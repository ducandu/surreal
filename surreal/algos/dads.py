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
from surreal.algos.sac import SAC
from surreal.components import FIFOBuffer, LossFunction, Network, Normal, Optimizer, Preprocessor
from surreal.config import AlgoConfig
from surreal.spaces import Dict, Float, Space


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
        self.a  = Space.make(config.action_space).with_batch()  # actions (a)
        self.ri = Float(main_axes=[("Episode Horizon", config.He)])  # intrinsic rewards
        self.z = Float(-1.0, 1.0, shape=(config.n,), main_axes=["B", ("T", config.Hp)])  # skill vectors
        #self.mu = Float(-1.0, 1.0, main_axes=[("Planning Horizon", config.Hp)])  # mean values

        self.pi = Network.make(input_space=Dict(dict(s=self.s, z=self.z)), output_space=self.a, **config.policy_network)
        self.q = Network(input_space=Dict(dict(s=self.s, z=self.z)), output_space=self.s, distributions=True)  # Output distribution over s' Space.
        self.B = FIFOBuffer(Dict(dict(s=self.s, z=self.z, a=self.a, t=bool)), config.memory_capacity,
                            when_full=self.event_buffer_full, next_record_setup=dict(s="s_"))
        self.SAC = SAC(config=config.sac, state_space=Tuple(env.state_space, z), action_space=env.action_space, policy_network=self.pi, memory=None)
        self.q_optimizer = Optimizer.make(config.optimizer)  # supervised model optimizer
        self.Lsup = DADSLoss()  # TODO: specify more
        #N = Normal(mean=μ)  # use standard covariance matrix

    def update(self, batch=None):
        # Update the time-percentage value.
        #time_percentage = env.time_step / max_time_steps
        # Get batch from buffer and clear the buffer.
        if batch is None:
            batch = self.B.flush()
        # Update for K1 iterations on same batch.
        self.optimizer.optimize(self.Lsup(batch), steps=self.config.K1, time_percentage=time_percentage)
        # Calculate intrinsic rewards.
        # TODO: according to paper, batch[z] should be part of the sum in the denominator.
        ri = tf.math.log(
            self.q(dict(a=batch["s"], z=batch["z"])) / tf.reduce_sum(self.q(dict(s=batch["s"], z=self.z.sample(L)))
        ) + tf.math.log(L)
        # Update RL-algo's policy (same as π) from our batch.
        SAC.update((batch["s"], batch["z"]), batch["a"], ri, batch["s'"], batch["t"])

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

        # Add single(!) szas't-tuple to buffer.
        if actor_time_steps > 0:
            self.B.add_records(dict(s=self.s.value, z=self.z.value, a=self.a.value, t=t, s_=s_))

        # Query policy for an action.
        a_ = self.pi(dict(s=s_, z=zi))

        # Send the new action back to the env.
        env.act(a_)

        # Store action and state for next tick.
        self.s.assign(s)
        self.a.assign(a_)

    # When buffer full -> learn transition model q.
    def event_buffer_full(self):  # TODO: how do we link this to B
        self.update(self.B.flush())

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


class DADSLoss(LossFunction):
    def call(self):
        return 0.0


class DADSConfig(AlgoConfig):
    """
    Config object for a DADS Algo.
    """
    def __init__(
            self, *,
            policy_network, state_space, action_space,
            preprocessor=None,
            supervised_optimizer=None,
            episode_buffer_length=200,
            summaries=None
    ):
        """
        Args:
            policy_network (Network): The policy-network (pi) to use as a function approximator for the learnt policy.
            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.
            preprocessor (Preprocessor): The preprocessor (if any) to use.
            supervised_optimizer (Optimizer): The optimizer to use for the supervised (q) model learning task.

            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
                In the simplest case, this is a list of `self.[...]`-property names of the SAC object that should
                be tracked after each tick.
        """
        # If one not given, use a copy of the other NN and make sure the given network is not a done Keras object yet.
        if policy_network is None:
            assert isinstance(q_network, (dict, list, tuple))
            policy_network = q_network
        elif q_network is None:
            assert isinstance(policy_network, (dict, list, tuple))
            q_network = policy_network

        # Clean up network configs to be passable as **kwargs to `make`.
        # Networks are given as sequential config or directly as Keras objects -> prepend "network" key to spec.
        if isinstance(q_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            q_network = dict(network=q_network)
        if isinstance(policy_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            policy_network = dict(network=policy_network)

        # Make sure our optimizers are defined ok.
        if default_optimizer is None:
            assert q_optimizer and policy_optimizer and alpha_optimizer
        if q_optimizer and policy_optimizer and alpha_optimizer:
            if default_optimizer:
                logging.warning(
                    "***WARNING: `default_optimizer` defined, but has no effect b/c `q_optimizer`, `policy_optimizer` "
                    "and `alpha_optimizer` are already provided!"
                )
        if q_optimizer is None:
            q_optimizer = default_optimizer
        if policy_optimizer is None:
            policy_optimizer = default_optimizer
        if alpha_optimizer is None:
            alpha_optimizer = default_optimizer

        assert time_unit in ["time_step", "env_tick"]

        # Special value for start-train parameter -> When memory full.
        if update_after == "when-memory-full":
            update_after = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif update_after == "when-memory-ready":
            update_after = memory_batch_size
        assert isinstance(update_after, int)

        # Make sure sync-freq >= update-freq:
        assert sync_frequency >= update_frequency
        # Make sure memory batch size is less than capacity.
        assert memory_batch_size <= memory_capacity

        # Derive memory_spec for SAC c'tor.
        # If PR -> Check that alpha is not 0.0.
        if use_prioritized_replay is True:
            if memory_alpha == 0.0:
                logging.warning(
                    "***WARNING: `use_prioritized_replay` is True, but memory's alpha is set to 0.0 (which implies no "
                    "prioritization whatsoever)!"
                )
            memory_spec = dict(type="prioritized-replay-buffer", alpha=memory_alpha, beta=memory_beta)
        else:
            memory_spec = dict(type="replay-buffer")
        memory_spec["capacity"] = memory_capacity
        memory_spec["next_record_setup"] = dict(s="s_",
                                                n_step=n_step)  # setup: s' is next-record of s (after n-steps).

        # Make action space.
        action_space = Space.make(action_space)

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
        self.last_sync = 0
