# Copyright 2019 ducandu GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from surreal.algos.rl_algo import RLAlgo
from surreal.algos.sac import SAC, SACConfig
from surreal.components import FIFOBuffer, NegLogLikelihoodLoss, Network, MixtureDistribution, Optimizer, Preprocessor
from surreal.config import AlgoConfig
from surreal.spaces import Dict, Float, Int, Space


class DADS(RLAlgo):  # He, Hz, Hp, R, K, C, γ, L
    """
    The DADS algorithm.
    [1] Dynamics-Aware Unsupervised Discovery of Skills - A. Sharma∗, S. Gu, S. Levine, V. Kumar, K. Hausman - Google Brain 2019
        Compare to "Algorithm 1" and "Algorithm 2" pseudocodes in paper.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.inference = False  # True=planning mode. False="supervised+intrinsic-reward+model-learning" mode.
        self.he = 0  # Current step within He (total episode horizon).
        self.hz = 0  # Current step within Hz (repeat horizon for one selected skill)

        self.preprocessor = Preprocessor.make(config.preprocessor)
        self.s = self.preprocessor(Space.make(config.state_space).with_batch())  # preprocessed states
        self.a = Space.make(config.action_space).with_batch()  # actions (a)
        self.ri = Float(main_axes=[("Episode Horizon", config.episode_horizon)])  # intrinsic rewards in He
        self.z = Float(-1.0, 1.0, shape=(config.dim_skill_vectors,), main_axes="B") if config.discrete_skills is False \
            else Int(config.dim_skill_vectors, main_axes="B")
        self.s_and_z = Dict(dict(s=self.s, z=self.z), main_axes="B")
        self.pi = Network.make(input_space=self.s_and_z, output_space=self.a, **config.policy_network)
        self.q = Network.make(input_space=self.s_and_z, output_space=self.s,
                              distributions=dict(type="mixture", num_experts=config.num_q_experts), **config.q_network)
        self.B = FIFOBuffer(Dict(dict(s=self.s, z=self.z, a=self.a, t=bool)), config.episode_buffer_capacity,
                            when_full=self.event_buffer_full, next_record_setup=dict(s="s_"))
        self.SAC = SAC(config=SACConfig.make(config.sac_config), name="SAC-level0")  # Low-level SAC.
        self.q_optimizer = Optimizer.make(config.supervised_optimizer)  # supervised model optimizer
        self.Lsup = NegLogLikelihoodLoss(distribution=MixtureDistribution(num_experts=config.num_q_experts))

    def update(self, samples, time_percentage):
        parameters = self.q(dict(s=samples["s"], z=samples["z"]), parameters_only=True)

        # Update for K1 iterations on same batch.
        weights = self.q.get_weights(as_ref=True)
        s_ = samples["s_"] if self.config.q_predicts_states_diff is False else \
            tf.nest.map_structure(lambda s, s_: s_ - s, samples["s"], samples["s_"])
        for _ in range(self.config.num_steps_per_supervised_update):
            loss = self.Lsup(parameters, s_)
            self.q_optimizer.apply_gradients(loss, weights, time_percentage=time_percentage)

        # Calculate intrinsic rewards.
        # Pull a batch of zs of size batch * (L - 1) (b/c 1 batch is the `z` of the sample (numerator's z)).
        batch_size = len(samples["s"])
        zs = tf.concat([samples["z"], self.z.sample(batch_size * (self.config.num_denominator_samples_for_ri - 1))])
        s = tf.nest.map_structure(lambda s: tf.tile(s, [self.config.num_denominator_samples_for_ri] + ([1] * (len(s.shape) - 1))), samples["s"])
        s_ = tf.nest.map_structure(lambda s: tf.tile(s, [self.config.num_denominator_samples_for_ri] + ([1] * (len(s.shape) - 1))), samples["s_"])
        # Single (efficient) forward pass yielding s' likelihoods.
        all_s__llhs = tf.stack(tf.split(self.q(dict(s=s, z=zs), s_, likelihood=True), self.config.num_denominator_samples_for_ri))
        r = tf.math.log(all_s__llhs[0] / tf.reduce_sum(all_s__llhs, axis=0)) + \
            tf.math.log(self.config.num_denominator_samples_for_ri)
        # Update RL-algo's policy (same as π) from our batch (using intrinsic rewards).
        self.SAC.update(
            dict(s=samples["s"], z=samples["z"], a=samples["a"], r=r, s_=samples["s_"], t=samples["t"]), time_percentage
        )

    # When buffer full -> Update transition model q.
    def event_buffer_full(self):
        self.update(self.B.flush(), time_percentage=time_percentage)

    def event_episode_starts(self, env, time_steps, batch_position, s):
        if self.inference is False:
            self.z.value[batch_position] = self.z.sample()  # Sample a new skill from Space z and store it in z (assume uniform).

    # Fill the buffer with M samples.
    def event_tick(self, env, actor_time_steps, batch_positions, r, t, s_):
        # If we are in inference mode -> do a planning step (rather than just act).
        if self.inference:
            self.he += 1
            if self.he >= self.config.He:  # We have reached the end of the total episode horizon -> reset.
                env.reset()  # Send reset request to env.
                return
            self.plan(env.s)
            # Execute selected skill for Hz steps.
            if self.hz == self.config.Hz - 1:
                zi = self.N.sample()   # ?? ~ N[he/Hz]
                hz = 0  # reset counter
            hz += 1
        else:
            for i in batch_positions:
                if self.hz[i] >= self.config.skill_horizon:
                    self.z.value[i] = self.z.sample()

        # Add single(!) szas't-tuple to buffer.
        if actor_time_steps > 0:
            self.B.add_records(dict(s=self.s.value, z=self.z.value, a=self.a.value, t=t, s_=s_))

        # Query policy for an action.
        a_ = self.pi(dict(s=s_, z=self.z.value))

        # Send the new action back to the env.
        env.act(a_)

        # Store action and state for next tick.
        self.s.assign(s_)
        self.a.assign(a_)

    #def plan(self, s0):
    #    for j in range(R):
    #        # Sample z (0 to Hp-1) from learnt N.
    #        zk ~ N[+K@1=K]  # Add a rank K at position 1 (0 is the Hp position).
    #        # Simulate trajectory using q.
    #        roll_out()
    #        # Calculate rewards from reward function (TODO: How to do that if env is external?!!)
    #        renv = env.get_reward(s0)
    #        # Update μ.
    #        for i in range(Hp):
    #            μ[i] = sum[k=0->K-1](exp(γ renv[k]) / (sum[p=0->K-1](exp(γ renv[p]))) * zk[i])
    #    # return best next plan (z).


class DADSConfig(AlgoConfig):
    """
    Config object for a DADS Algo.
    """
    def __init__(
            self, *,
            policy_network, q_network,
            state_space, action_space,
            sac_config,
            num_q_experts=4,  # 4 used in paper.
            q_predicts_states_diff=False,
            num_denominator_samples_for_ri=250,  # 50-500 used in paper
            dim_skill_vectors=10, discrete_skills=False, episode_horizon=200, skill_horizon=None,
            preprocessor=None,
            supervised_optimizer=None,
            num_steps_per_supervised_update=1,
            episode_buffer_capacity=200,
            summaries=None
    ):
        """
        Args:
            policy_network (Network): The policy-network (pi) to use as a function approximator for the learnt policy.

            q_network (Network): The dynamics-network (q) to use as a function approximator for the learnt env
                dynamics. NOTE: Not to be confused with a Q-learning Q-net! In the paper, the dynamics function is
                called `q`, hence the same nomenclature here.

            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.
            sac_config (SACConfig): The config for the internal SAC-Algo used to learn the skills using intrinsic rewards.

            num_q_experts (int): The number of experts used in the Mixture distribution output bz the q-network to
                predict the next state (s') given s (state) and z (skill vector).

            q_predicts_states_diff (bool): Whether the q-network predicts the different between s and s' rather than
                s' directly. Default: False.

            num_denominator_samples_for_ri (int): The number of samples to calculate for the denominator of the
                intrinsic reward function (`L` in the paper).

            dim_skill_vectors (int): The number of dimensions of the learnt skill vectors.
            discrete_skills (bool): Whether skill vectors are discrete (one-hot).
            episode_horizon (int): The episode horizon (He) to move within, when gathering episode samples.

            skill_horizon (Optional[int]): The horizon for which to use one skill vector (before sampling a new one).
                Default: Use value of `episode_horizon`.

            preprocessor (Preprocessor): The preprocessor (if any) to use.
            supervised_optimizer (Optimizer): The optimizer to use for the supervised (q) model learning task.

            num_steps_per_supervised_update (int): The number of gradient descent iterations per update
                (each iteration uses the same environment samples).

            episode_buffer_capacity (int): The capacity of the episode (experience) FIFOBuffer.

            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
                In the simplest case, this is a list of `self.[...]`-property names of the SAC object that should
                be tracked after each tick.
        """
        # Clean up network configs to be passable as **kwargs to `make`.
        # Networks are given as sequential config or directly as Keras objects -> prepend "network" key to spec.
        if isinstance(policy_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            policy_network = dict(network=policy_network)
        if isinstance(q_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            q_network = dict(network=q_network)

        # Make state/action space.
        state_space = Space.make(state_space)
        action_space = Space.make(action_space)

        # Fix SAC config, add correct state- and action-spaces.
        sac_config["state_space"] = Dict(s=state_space, z=Float(-1.0, 1.0, shape=(dim_skill_vectors,)))
        sac_config["action_space"] = action_space
        sac_config["memory_capacity"] = 1  # Use no memory. Updates are done from DADS' own buffer.
        sac_config["memory_batch_size"] = 1
        sac_config["policy_network"] = policy_network  # Share policy network between DADS and underlying learning SAC.

        if skill_horizon is None:
            skill_horizon = episode_horizon

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
