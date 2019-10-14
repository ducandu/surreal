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

import logging
import numpy as np
import tensorflow as tf

from surreal.algos.rl_algo import RLAlgo
from surreal.components import Network, Memory, Optimizer, Preprocessor, LossFunction, NStep
from surreal.config import Config
from surreal.spaces import Dict, Int, Space, ContainerSpace
from surreal.utils.util import default_dict


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
        self.a_soft = self.a.as_one_hot_float_space()  # soft-one-hot actions (if Int elements in action space)
        self.pi = Network.make(distributions=dict(  # policy (π)
            bounded_distribution_type=self.config.bounded_distribution_type, discrete_distribution_type="gumbel-softmax",
            gumbel_softmax_temperature=self.config.gumbel_softmax_temperature
        ), input_space=self.s, output_space=self.a, **config.policy_network)
        self.Q = []  # the Q-networks
        for i in range(config.num_q_networks):
            self.Q.append(Network.make(input_space=Dict(s=self.s, a=self.a), output_space=float, **config.q_network))
        self.Qt = [self.Q[i].copy(trainable=False) for i in range(config.num_q_networks)]  # target q-network(s)
        record_space = Dict(default_dict(dict(s=self.s, a=self.a_soft, r=float, t=bool),
                                         {"n": int} if config.n_step > 1 else {}), main_axes="B")
        self.memory = Memory.make(record_space=record_space, **config.memory_spec)
        self.alpha = tf.Variable(config.initial_alpha, name="alpha", dtype=tf.float32)  # the temperature parameter α
        self.n_step = NStep(config.gamma, n_step=config.n_step, n_step_only=True)
        self.L, self.Ls_critic, self.L_actor, self.L_alpha = SACLoss(), [0, 0], 0, 0  # SAC loss function and values.
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
            records = self.n_step(self.s.value, self.a_soft.value, r, t, s_) if self.config.n_step > 1 else \
                dict(s=self.s.value, a=self.a_soft.value, r=r, t=t, s_=s_)
            if records:
                self.memory.add_records(records)

        # Query policy for an action sample and send the new actions back to the env.
        a__soft_categorical = self.pi(s_)  # a_ is a tensor due to it being direct NN output.
        a_ = self.argmax_if_applicable(a__soft_categorical).numpy()
        env.act(a_)

        # Every nth tick event -> Update all our networks, based on the losses (i iterations per update step).
        if self.is_time_to("update", env.tick, actor_time_steps):
            for _ in range(self.config.num_steps_per_update):
                records, indices = self.memory.get_records_with_indices(self.config.memory_batch_size)
                self.Ls_critic, abs_td_errors, tapes_critic, self.L_actor, tape_actor, self.L_alpha, tape_alpha = \
                    self.L(records, self.alpha, self.pi, self.Q, self.Qt, self.config)
                for i in range(self.config.num_q_networks):
                    weights = self.Q[i].get_weights(as_ref=True)
                    self.optimizers["q"].apply_gradients(
                        list(zip(tapes_critic[i].gradient(self.Ls_critic[i], weights), weights)), time_percentage
                    )
                # Update our memory's priority weights with the abs(td-error) values (if we use a PR).
                if self.config.use_prioritized_replay is True:
                    self.memory.update_records(indices, abs_td_errors)
                weights = self.pi.get_weights(as_ref=True)
                grads_and_vars = list(zip(tape_actor.gradient(self.L_actor, weights), weights))
                self.optimizers["pi"].apply_gradients(grads_and_vars, time_percentage)
                if self.L_alpha is not None:
                    self.optimizers["alpha"].apply_gradients(
                        [(tape_alpha.gradient(self.L_alpha, self.alpha), self.alpha)], time_percentage
                    )

        # Every mth tick event -> Synchronize target Q-net(s) using soft (tau) syncing.
        if self.is_time_to("sync", env.tick, actor_time_steps, only_after=self.config.update_after):
            for i in range(self.config.num_q_networks):
                self.Qt[i].sync_from(self.Q[i], tau=self.config.sync_tau)

        # Store actions and states for next tick (they form the incomplete next *s**a*rs't-tuple).
        self.s.assign(s_)
        self.a_soft.assign(a__soft_categorical)

    def argmax_if_applicable(self, a):
        # In case we have a discrete action space, convert back to Int via argmax.
        if isinstance(self.a, Int):
            return tf.argmax(a, axis=-1)
        # In case of a ContainerSpace, do this for all sub-components that are Int spaces.
        elif isinstance(self.a, ContainerSpace):
            return tf.nest.map(lambda a_comp: tf.argmax(a_comp, axis=-1) if a_comp.dtype == np.int else a_comp, a)
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
        s, a_soft, r, s_, t = samples["s"], samples["a"], samples["r"], samples["s_"], samples["t"]
        # Get a' (soft one-hot if action Space is discrete) and its log-likelihood.
        a__soft, log_likelihood_a_ = pi(s_, log_likelihood=True)
        # Take the min of the two q-target nets to calculate the Q-loss-target (critic loss).
        target_qs = tf.reduce_min(tuple([target_q_net(dict(s=s_, a=a__soft)) for target_q_net in qt_nets]), axis=0)
        losses_critic = []
        abs_td_errors = []
        tapes_critic = []
        for q_net in q_nets:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(q_net.get_weights(as_ref=True))
                td_error = (r + config.gamma * tf.where(
                    t, tf.zeros_like(target_qs), target_qs - alpha * log_likelihood_a_
                )) - q_net(dict(s=s, a=a_soft))
                abs_td_errors.append(tf.abs(td_error))
                losses_critic.append(0.5 * tf.reduce_mean(td_error ** 2))
            tapes_critic.append(tape)
        abs_td_errors = tf.reduce_mean(abs_td_errors, axis=0)

        with tf.GradientTape(watch_accessed_variables=False) as tape_actor:
            tape_actor.watch(pi.get_weights(as_ref=True))
            # Get current pi output for the states.
            a_sampled_soft, log_likelihood_a_sampled = pi(s, log_likelihood=True)
            q_values_sampled = tf.reduce_min(tuple([q_net(dict(s=s, a=a_sampled_soft)) for q_net in q_nets]), axis=0)
            loss_actor = tf.reduce_mean(alpha * log_likelihood_a_sampled - q_values_sampled)

        loss_alpha = None
        tape_alpha = None
        if config.optimize_alpha is True:
            with tf.GradientTape(watch_accessed_variables=False) as tape_alpha:
                tape_alpha.watch(alpha)
                # In [1], α is used directly, however the implementation uses log(α).
                # See the discussion in https://github.com/rail-berkeley/softlearning/issues/37.
                #loss_alpha = -tf.math.log(alpha) * (tf.reduce_mean(log_likelihood_a_sampled) + config.entropy_target)
                loss_alpha = -tf.reduce_mean(tf.math.log(alpha) * (log_likelihood_a_sampled * config.entropy_target))

        return losses_critic, abs_td_errors, tapes_critic, loss_actor, tape_actor, loss_alpha, tape_alpha


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
            optimize_alpha=True,
            bounded_distribution_type="squashed-normal", gumbel_softmax_temperature=1.0,
            gamma=0.99,
            num_q_networks=2,
            memory_capacity=10000, memory_batch_size=256,
            use_prioritized_replay=False, memory_alpha=1.0, memory_beta=0.0,
            initial_alpha=1.0, entropy_target=None,  # default: -dim(A)
            n_step=1,
            max_time_steps=None, update_after=0, update_frequency=1, num_steps_per_update=1,
            sync_frequency=1, sync_tau=0.005,
            time_unit="time_step",
            summaries=None
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

            optimize_alpha (bool): Whether to use the alpha loss term and an optimizer step to update alpha. False
                for keeping alpha constant at `initial_alpha`.

            bounded_distribution_type (str): Which distribution type to use for continuous, bounded output spaces.
                Must be a Distribution class type string. See components/distributions/__init__.py

            gumbel_softmax_temperature (float): Iff `discrete_distribution_type`="gumbel-softmax" (which is fixed and
                required for SAC), which temperature parameter to use.

            gamma (float): The discount factor (gamma).
            memory_capacity (int): The memory's capacity (max number of records to store).
            memory_batch_size (int): The batch size to use for updating from memory.
            use_prioritized_replay (bool): Whether to use a PrioritizedReplayBuffer (instead of a plain ReplayBuffer).
            memory_alpha (float): The alpha value for the PrioritizedReplayBuffer.
            memory_beta (float): The beta value for the PrioritizedReplayBuffer.

            initial_alpha (float): The initial value for alpha (before optimization).
            entropy_target (float): The value of "Hbar" in the loss function for alpha. Default is -dim(A).
            n_step (int): The number of steps (n) to "look ahead/back" when converting 1-step tuples into n-step ones.

            #n_step_only (bool): Whether to exclude samples that are shorter than `n_step` AND don't have a terminal
            #    at the end.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_step"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.
            num_steps_per_update (int): The number of gradient descent iterations per update.
            sync_frequency (int): The frequency (in `time_unit`) with which to sync our target network.
            sync_tau (float): The target smoothing coefficient with which to synchronize the target Q-network.
            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.
            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
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
        memory_spec["next_record_setup"] = dict(s="s_", n_step=n_step)  # setup: s' is next-record of s (after n-steps).

        # Make action space.
        action_space = Space.make(action_space)

        # Default Hbar: -dim(A) (according to the paper).
        if entropy_target is None:
            entropy_target = -(action_space.flat_dim_with_categories if isinstance(action_space, Int) else
                               action_space.flat_dim)
            print("entropy_target={}".format(entropy_target))

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
        self.last_sync = 0
