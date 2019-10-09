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

import numpy as np
import tensorflow as tf

from surreal.algos.rl_algo import RLAlgo
from surreal.components import Network, ReplayBuffer, Optimizer, Preprocessor, LossFunction
from surreal.config import Config
from surreal.spaces import Dict, Int, Space, ContainerSpace


class SAC(RLAlgo):
    """
    An implementation of the soft-actor critic algorithm, following the paper:

    [1] Soft Actor-Critic Algorithms and Applications - Haarnoja, Zhou, Hartikainen et al. -
        UC Berkeley, Google Brain - Jan 2019.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.preprocessor = Preprocessor.make(config.preprocessor)
        self.s = self.preprocessor(Space.make(self.config.state_space).with_batch())  # preprocessed states (x)
        self.a = Space.make(self.config.action_space).with_batch()  # actions (a)
        self.pi = Network.make(distributions=dict(  # policy (π)
            bounded_distribution_type=self.config.bounded_distribution_type,discrete_distribution_type="gumbel-softmax",
            gumbel_softmax_temperature=self.config.gumbel_softmax_temperature
        ), input_space=self.s, output_space=self.a, **config.policy_network)
        self.Q = []  # the Q-networks
        for i in range(config.num_q_networks):
            self.Q.append(Network.make(input_space=Dict(s=self.s, a=self.a), output_space=float, **config.q_network))
        self.Qt = [self.Q[i].copy(trainable=False) for i in range(config.num_q_networks)]  # target q-network(s)
        self.memory = ReplayBuffer.make(
            record_space=Dict(dict(s=self.s, a=self.a, r=float, s_=self.s, t=bool), main_axes="B"),
            capacity=config.memory_capacity
        )
        self.alpha = tf.Variable(config.initial_alpha, name="alpha", dtype=tf.float32)  # the temperature parameter α
        self.L = SACLoss()  # SAC loss functions.
        self.optimizers = dict(
            q=Optimizer.make(self.config.q_optimizer), pi=Optimizer.make(self.config.policy_optimizer),
            alpha=Optimizer.make(self.config.alpha_optimizer)
        )
        self.preprocessor.reset()  # make sure, Preprocessor is clean

    def event_episode_starts(self, env, time_steps, batch_position, s):
        # Reset Phi at beginning of each episode (only at given batch position).
        self.preprocessor.reset(batch_position)

    def event_tick(self, env, actor_time_steps, batch_positions, r, t, s_):
        # Update time-percentage value (for decaying parameters, e.g. learning-rate).
        time_percentage = actor_time_steps / (self.config.max_time_steps or env.max_time_steps)

        # Preprocess states.
        s_ = self.preprocessor(s_)

        # Add now-complete sars't-tuple to memory (batched).
        if actor_time_steps > 0:
            self.memory.add_records(dict(s=self.s.value, a=self.a.value, r=r, t=t, s_=s_))

        # Query policy for an action sample and send the new actions back to the env.
        a__soft_categorical = self.pi(s_).numpy()  # a_ is a tensor due to it being direct NN output.
        a_ = argmax_if_applicable(a__soft_categorical, self.a)
        # TEST:
        if np.isnan(a_.min()) or np.isnan(a_.max()):
            a_ = self.pi(s_).numpy()
            print("here")
        #print("a-min={} a-max={}".format(a_.min(), a_.max()))
        # END TEST
        env.act(a_)

        # Every nth tick event -> Update all our networks, based on the losses (i iterations per update step).
        if self.is_time_to("update", env.tick, actor_time_steps):
            for _ in range(self.config.num_steps_per_update):
                records = self.memory.get_records(self.config.memory_batch_size)
                Ls_critic, td_errors_critic, tapes_critic, L_actor, tape_actor, L_alpha, tape_alpha = self.L(
                    records, self.alpha, self.pi, self.Q, self.Qt, self.config)
                for i in range(self.config.num_q_networks):
                    weights = self.Q[i].get_weights(as_ref=True)
                    self.optimizers["q"].apply_gradients(
                        list(zip(tapes_critic[i].gradient(Ls_critic[i], weights), weights)), time_percentage
                    )
                #print("Q(0,->)={}".format(self.Q[0](dict(a=np.array([1]), s=np.array([[1.0, 0.0, 0.0, 0.0]])))))
                #print("Q(1,->)={}".format(self.Q[0](dict(a=np.array([1]), s=np.array([[0.0, 1.0, 0.0, 0.0]])))))
                #print("Qt(0,->)={}".format(self.Qt[0](dict(a=np.array([1]), s=np.array([[1.0, 0.0, 0.0, 0.0]])))))
                #print("Qt(1,->)={}".format(self.Qt[0](dict(a=np.array([1]), s=np.array([[0.0, 1.0, 0.0, 0.0]])))))

                weights = self.pi.get_weights(as_ref=True)
                # TEST
                old_weights = self.pi.get_weights()
                grads_and_vars = list(zip(tape_actor.gradient(L_actor, weights), weights))
                self.optimizers["pi"].apply_gradients(grads_and_vars)
                #print("pi(0)={}".format(self.pi(np.array([[1.0, 0.0, 0.0, 0.0]]))))
                #print("pi(1)={}".format(self.pi(np.array([[0.0, 1.0, 0.0, 0.0]]))))
                new_weights = self.pi.get_weights()
                if np.isnan(new_weights[0].min()) or np.isnan(new_weights[0].max()):
                    print("here, too")
                # END: TEST
                self.optimizers["alpha"].apply_gradients([(tape_alpha.gradient(L_alpha, self.alpha), self.alpha)])
                #print("alpha={}".format(float(self.alpha.numpy())))

        # Every mth tick event -> Synchronize target Q-net(s) using tau-syncing.
        if self.is_time_to("sync", env.tick, actor_time_steps, only_after=self.config.update_after):
            for i in range(self.config.num_q_networks):
                self.Qt[i].sync_from(self.Q[i], tau=self.config.sync_tau)

        # Store actions and states for next tick (they form the incomplete next *s**a*rs't-tuple).
        self.s.assign(s_)
        self.a.assign(a_)


def argmax_if_applicable(a, space):
    # In case we have a discrete action space, convert back to Int via argmax.
    if isinstance(space, Int):
        return np.argmax(a, axis=-1)
    # In case of a ContainerSpace, do this for all sub-components that are Int spaces.
    elif isinstance(space, ContainerSpace):
        return tf.nest.map(lambda s: np.argmax(s, axis=-1) if s.dtype == np.int else s, a)
    return a


class SACLoss(LossFunction):
    """
    The SAC loss function consisting of three parts:
    a) Critic Loss (for each Q-function):
        JQ = E(s,a)~D [ 1/2( Q(s,a) - (r(s,a) + γE(s'~p)[V(s')]) )² ]

    b) Actor Loss (for the policy π):
        Jπ =

    c) "Alpha Loss" (for the entropy weight-parameter α):
        Jα = E(a)~π [-α log π(a|s) - α `Hbar`]
        Note: Optimizer will simply apply the gradient of dJα/dα to the α variable.

    Where:
        E = expectation values.
        V = soft state value function: V(s) = E(a~π) [Qt(s,a) - α log π(a|s)], which
            adds an entropy bonus term to the reward weighted with α.
        Qt = target Q-network (synchronized every m time steps using tau-syncing).
        γ = discount factor
    """
    def call(self, samples, alpha, pi, q_nets, qt_nets, config):
        """
        Args:
            samples (Dict[states,actions,rewards,next-states,terminals]): The batch to push through the loss function
                to get an expectation value (mean over all batch items).

            alpha (tf.Variable): The alpha variable.
            pi (Network): The policy network (π).
            q_nets (List[Network]): The Q-network(s).
            qt_nets (List[Network]): The target Q-network(s).
            config (SACConfig): A SACConfig object, of which this LossFunction uses some properties.

        Returns:
            Tuple:
                List[tf.Tensor]: The critic loss value(s) JQ. See formula above. This may be a list of 1 or 2 entries
                    depending on config.num_q_networks.
                tf.Tensor: The actor loss value Jπ. See formula above.
                tf.Tensor: The alpha loss value Jα. See formula above.
        """
        s, a, r, s_, t = samples["s"], samples["a"], samples["r"], samples["s_"], samples["t"]
        # Get a' (soft one-hot if action Space is discrete) and its log-likelihood.
        a__soft, log_likelihood_a_ = pi(s_, log_likelihood=True)
        # Take the min of the two q-target nets to calculate the Q-loss-target (critic loss).
        target_qs = tf.reduce_min(tuple([target_q_net(dict(s=s_, a=a__soft)) for target_q_net in qt_nets]), axis=0)
        losses_critic = []
        td_errors_critic = []
        tapes_critic = []
        for q_net in q_nets:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(q_net.get_weights(as_ref=True))
                td_errors_critic.append((r + config.gamma * tf.where(t, tf.zeros_like(target_qs), target_qs - alpha * log_likelihood_a_)) - q_net(dict(s=s, a=a)))
                losses_critic.append(0.5 * tf.reduce_mean(td_errors_critic[-1] ** 2))
            tapes_critic.append(tape)

        with tf.GradientTape(watch_accessed_variables=False) as tape_actor:
            tape_actor.watch(pi.get_weights(as_ref=True))
            # Get current pi output for the states.
            a_sampled_soft, log_likelihood_a_sampled = pi(s, log_likelihood=True)
            q_values_sampled = tf.reduce_min(tuple([q_net(dict(s=s, a=a_sampled_soft)) for q_net in q_nets]), axis=0)
            loss_actor = tf.reduce_mean(alpha * log_likelihood_a_sampled - q_values_sampled)

        with tf.GradientTape(watch_accessed_variables=False) as tape_alpha:
            tape_alpha.watch(alpha)
            # In [1], α is used directly, however the implementation uses log(α).
            # See the discussion in https://github.com/rail-berkeley/softlearning/issues/37.
            loss_alpha = -tf.math.log(alpha) * (tf.reduce_mean(log_likelihood_a_sampled) + config.entropy_target)

        return losses_critic, td_errors_critic, tapes_critic, loss_actor, tape_actor, loss_alpha, tape_alpha


class SACConfig(Config):
    """
    Config object for a SAC Algo.
    """
    def __init__(
            self, *,
            q_network, state_space, action_space,
            policy_network=None,
            preprocessor=None,
            default_optimizer=None, q_optimizer=None, policy_optimizer=None, alpha_optimizer=None,
            bounded_distribution_type="squashed-normal", gumbel_softmax_temperature=1.0,
            gamma=0.99,
            num_q_networks=2,
            memory_capacity=10000, memory_batch_size=256,  # memory_alpha=1.0, memory_beta=0.0,
            initial_alpha=1.0, entropy_target=None,  # default: -dim(A)
            #n_step=1, n_step_only=True,
            max_time_steps=None, update_after=0, update_frequency=1, num_steps_per_update=1,
            sync_frequency=1, sync_tau=0.005,
            time_unit="time_steps"
    ):
        """
        Args:
            q_network (Network): The Q-network to use as a function approximator for the learnt Q-function.
            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.

            policy_network (Network): The policy-network (pi) to use as a function approximator for the learnt policy.
                Default: Use the same setup as the q-network(s).

            preprocessor (Preprocessor): The preprocessor (if any) to use.
            default_optimizer (Optimizer): The optimizer to use for any Q/pi/alpha, which don't have their own defined.
            q_optimizer (Optimizer): The optimizer to use for the Q-network. If None, use `optimizer`.
            policy_optimizer (Optimizer): The optimizer to use for the policy (pi). If None, use `optimizer`.
            alpha_optimizer (Optimizer): The optimizer to use for the alpha parameter. If None, use `optimizer`.

            bounded_distribution_type (str): Which distribution type to use for continuous, bounded output spaces.
                Must be a Distribution class type string. See components/distributions/__init__.py

            gumbel_softmax_temperature (float): Iff `discrete_distribution_type`="gumbel-softmax" (which is fixed and
                required for SAC), which temperature parameter to use.

            gamma (float): The discount factor (gamma).
            memory_capacity (int): The memory's capacity (max number of records to store).
            #memory_alpha (float): The alpha value for the PrioritizedReplayBuffer.
            #memory_beta (float): The beta value for the PrioritizedReplayBuffer.
            memory_batch_size (int): The batch size to use for updating from memory.

            initial_alpha (float): The initial value for alpha (before optimization).
            entropy_target (float): The value of "Hbar" in the loss function for alpha. Default is -dim(A).

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_steps"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.
            num_steps_per_update (int): The number of gradient descent iterations per update.
            sync_frequency (int): The frequency (in `time_unit`) with which to sync our target network.
            sync_tau (float): The target smoothing coefficient with which to synchronize the target Q-network.
            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.
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
                raise Warning(
                    "***WARNING: `default_optimizer` defined, but has no effect b/c `q_optimizer`, `policy_optimizer` "
                    "and `alpha_optimizer` are already provided!"
                )
        if q_optimizer is None:
            q_optimizer = default_optimizer
        if policy_optimizer is None:
            policy_optimizer = default_optimizer
        if alpha_optimizer is None:
            alpha_optimizer = default_optimizer

        # Special value for start-train parameter -> When memory full.
        if update_after == "when-memory-full":
            assert time_unit == "time_steps"
            update_after = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif update_after == "when-memory-ready":
            assert time_unit == "time_steps"
            update_after = memory_batch_size
        assert isinstance(update_after, int)

        # Make sure sync-freq >= update-freq:
        assert sync_frequency >= update_frequency
        # Make sure memory batch size is less than capacity.
        assert memory_batch_size <= memory_capacity

        # Make action space.
        action_space = Space.make(action_space)

        # Default Hbar: -dim(A) (according to the paper).
        if entropy_target is None:
            entropy_target = -(action_space.flat_dim_with_categories if isinstance(action_space, Int) else
                               action_space.flat_dim)  # TODO: check, whether correct

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
        self.last_sync = 0
