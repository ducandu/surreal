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
from surreal.components import Decay, GeneralizedAdvantages, Network, ReplayBuffer, Optimizer, Preprocessor, \
    LossFunction
from surreal.config import AlgoConfig
from surreal.spaces import Dict, Int, Space


class PPO(RLAlgo):
    """
    An implementation of the proximal-policy optimization algorithm, following the paper:

    [1] Proximal Policy Optimization Algorithms - J. Schulman et al. - OpenAI 2017.
        https://arxiv.org/abs/1707.06347
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.preprocessor = Preprocessor.make(config.preprocessor)
        self.s = self.preprocessor(Space.make(config.state_space).with_batch())  # preprocessed states
        self.a = Space.make(config.action_space).with_batch()  # actions (a)
        self.V = None
        if config.use_separate_value_function_network is True:
            self.V = Network.make()
        self.pi = Network.make(distributions=dict(), input_space=self.s, output_space=self.a, **config.policy_network)  # policy (π)
        record_space = Dict(dict(s=self.s, a=self.a, r=float, t=bool), main_axes="B")
        self.memory = ReplayBuffer.make(capacity=config.memory_capacity, record_space=record_space, episodes=True)
        self.gae = GeneralizedAdvantages(config.gamma, config.gae_lambda, config.clip_rewards)
        self.L = PPOLoss()
        self.optimizers = dict(
            pi=Optimizer.make(self.config.policy_optimizer), v=Optimizer.make(self.config.value_function_optimizer)
        )
        self.preprocessor.reset()  # make sure, Preprocessor is clean

    def update(self, samples, time_percentage):
        s, a, r, t, i = samples["s"], samples["a"], samples["r"], samples["t"], samples["i"]
        
        prev_log_likelihoods = self.pi(s, a, log_likelihood=True)
        if self.V is not None:
            prev_V = self.V(self.s.value)
        else:
            prev_V = prev_log_likelihoods[0]["V"]  # 1st return value is the actual output (take "V" key of that).
            # 2nd return value is the composite log-llh for all distributions
            # (only pi, not V, as V does not have a distribution).
            prev_log_likelihoods = prev_log_likelihoods[1]

        # Post process all rewards (replace them by generalized advantage estimation values).
        A = r
        if self.config.apply_postprocessing is True:
            A = self.gae.get_gae_values(prev_V, r, t, i)
        if self.config.standardize_advantages:
            mean, std = tf.nn.moments(x=A, axes=[0])
            A = (A - mean) / std

        policy_loss, value_function_loss = None, None
        for _ in range(self.config.num_iterations):
            # Figure out the indices for the sub-sampling iteration.
            start = np.random.randint(self.config.memory_batch_size)
            indices = np.arange(start, start + self.config.sample_size) % self.config.memory_batch_size
            s_subsample = tf.nest.map_structure(lambda v: tf.gather(v, indices), s)
            a_subsample = tf.nest.map_structure(lambda v: tf.gather(v, indices), a)
            prev_log_likelihoods_subsample = tf.gather(params=prev_log_likelihoods, indices=indices)
            prev_V_subsample = tf.gather(params=prev_V, indices=indices)
            A_subsample = tf.gather(params=A, indices=indices)
            # Calculate losses and update out network(s).
            loss_pi, tape_pi, loss_V, tape_V = self.L(
                s_subsample, a_subsample, prev_log_likelihoods_subsample, prev_V_subsample, A_subsample,
                self.pi, self.V, self.config
            )
            if self.V is None:
                loss_pi = loss_pi + loss_V
            else:
                self.optimizers["V"].apply_gradients(
                    self.V.get_weights(as_ref=True), loss_V, time_percentage=time_percentage
                )
            self.optimizers["pi"].apply_gradients(
                self.pi.get_weights(as_ref=True), loss_pi, time_percentage=time_percentage
            )

        return policy_loss, value_function_loss

    def event_episode_starts(self, event):
        # Reset preprocessor at beginning of each episode (only at given batch position).
        self.preprocessor.reset(event.current_actor_slot)

    def event_tick(self, event):
        # Update time-percentage value (for decaying parameters, e.g. learning-rate).
        time_percentage = event.actor_time_steps / (self.config.max_time_steps or event.env.max_time_steps)

        # Preprocess states.
        s_ = self.preprocessor(event.s_)

        # Add now-complete sars't-tuple to memory (batched).
        if event.actor_time_steps > 0:
            self.memory.add_records(dict(s=self.s.value, a=self.a.value, r=event.r, t=event.t, s_=s_))

        # Query policy for an action sample and send the new actions back to the env.
        a_ = self.pi(s_)  # a_ is a tensor due to it being direct NN output.
        event.env.act(a_)

        # Every nth tick event -> Update all our networks, based on the losses (i iterations per update step).
        if self.is_time_to("update", event.env.tick, event.actor_time_steps):
            for _ in range(self.config.num_steps_per_update):
                self.update(self.memory.get_records(self.config.memory_batch_size), time_percentage)

        # Store actions and states for next tick (they form the incomplete next *s**a*rs't-tuple).
        self.s.assign(s_)
        self.a.assign(a_)


class PPOLoss(LossFunction):
    """
    The PPO loss function:
    TODO: document.
    """
    def call(self, s, a, prev_log_likelihoods, prev_V, A, pi, value_function, config, time_percentage):
        """
        Args:
            s (any): A (sub-sampled) batch of states.
            a (any): A (sub-sampled) batch of states.
            prev_log_likelihoods (any): The result of having passed s and a through `pi` prior to the upcoming update.
            prev_V (any): The result of having passed s through the `value_function` prior to the upcoming update.
            A (any): The (sub-sampled) generalized advantage estimation values.
            pi (Network): The policy network.
            value_function (Optional[Network]): The (separate) value function Network, if not shared with `pi`.
            config (PPOConfig): The configuration of the PPO Algo.

        Returns:
            Tuple:
                tf.Tensor: The policy loss value. See formula above.
                tf.Tensor: The value function's loss value. See formula above.
        """
        log_likelihoods = pi(s, a, log_likelihood=True)
        values = value_function(s)
        entropy = pi.entropy(s)

        # Policy loss.
        # Likelihood ratio and clipped objective.
        ratio = tf.math.exp(log_likelihoods - prev_log_likelihoods)
        # Make sure the pg_advantages vector (batch) is broadcast correctly.
        for _ in range(get_rank(ratio) - 1):
            advantages = tf.expand_dims(A, -1)

        clipped_A = tf.where(
            A > 0, (1 + config.clip_ratio(time_percentage)) * A, (1 - config.clip_ratio(time_percentage)) * A
        )
        policy_loss = -tf.min(ratio * A, clipped_A)

        # Subtract the entropy bonus from the loss (the larger the entropy the smaller the loss).
        policy_loss -= config.entropy_weight(time_percentage) * entropy

        # Reduce over the composite actions, if any.
        if get_rank(ratio) > 1:
            policy_loss = tf.reduce_mean(policy_loss, tuple(range(1, get_rank(ratio))))

        # Value function loss.
        v_targets = A + prev_V
        v_targets = v_targets.detach()
        value_function_loss = (values - v_targets) ** 2
        if config.clip_value_function:
            vf_clipped = prev_V + tf.clamp(
                values - prev_V, -config.clip_value_function, config.clip_value_function
            )
            clipped_loss = (vf_clipped - v_targets) ** 2
            value_function_loss = tf.max(value_function_loss, clipped_loss)

        return policy_loss, value_function_loss


class PPOConfig(AlgoConfig):
    """
    Config object for a SAC Algo.
    """
    def __init__(
            self, *,
            policy_network, state_space, action_space,
            value_function_network=None, use_shared_value_function=True,
            preprocessor=None,
            default_optimizer=None, policy_optimizer=None, value_function_optimizer=None,
            gamma=0.99,
            clip_ratio=0.2, gae_lambda=1.0, clip_rewards=0.0, clip_value_function=0.0,
            standardize_advantages=False, entropy_weight=None,
            memory_capacity=10000, memory_batch_size=256,
            use_prioritized_replay=False, memory_alpha=1.0, memory_beta=0.0,
            max_time_steps=None, update_after=0, update_frequency=1, num_steps_per_update=1,
            time_unit="time_step",
            summaries=None
    ):
        """
        Args:
            policy_network (Network): The pi-network to use as a function approximator for the learnt policy.
            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.

            value_function_network (Network): The value-function-network (V) to use as a function approximator for the
                learnt value-function. Default: Use the same setup as the policy-network.

            use_shared_value_function (bool): Whether to not use the `value_function_network` (must be None then) and
                instead share the network between pi and V.

            preprocessor (Preprocessor): The preprocessor (if any) to use.
            default_optimizer (Optimizer): The optimizer to use for any Q/pi/alpha, which don't have their own defined.
            policy_optimizer (Optimizer): The optimizer to use for the pi-network. If None, use `default_optimizer`.

            value_function_optimizer (Optimizer): The optimizer to use for the value-function (V).
                If None, use `default_optimizer`.

            gamma (float): The discount factor (gamma).
            clip_ratio (float): Clipping parameter for the importance sampling (IS) likelihood ratio.
            gae_lambda (float): Lambda for generalized advantage estimation.
            clip_rewards (Optional[float]): If not 0 or None, rewards will be clipped within a +/- `clip_rewards` range.
            clip_value_function (Optional[float]): If not 0 or None, V outputs will be clipped within a +/- range.
            standardize_advantages (bool): If true, standardize advantage values in update.
            entropy_weight (float): The coefficient used for the entropy regularization term (L[E]).
            memory_capacity (int): The memory's capacity (max number of records to store).
            memory_batch_size (int): The batch size to use for updating from memory.
            use_prioritized_replay (bool): Whether to use a PrioritizedReplayBuffer (instead of a plain ReplayBuffer).
            memory_alpha (float): The alpha value for the PrioritizedReplayBuffer.
            memory_beta (float): The beta value for the PrioritizedReplayBuffer.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_step"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.

            num_steps_per_update (int): The number of gradient descent iterations per update (each iteration uses
                a different sample).

            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.

            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
                In the simplest case, this is a list of `self.[...]`-property names of the SAC object that should
                be tracked after each tick.
        """
        # If one not given, use a copy of the other NN and make sure the given network is not a done Keras object yet.
        if policy_network is None:
            assert isinstance(value_function_network, (dict, list, tuple))
            policy_network = value_function_network
        if use_shared_value_function is True:
            assert value_function_network is None
        elif value_function_network is None:
            assert isinstance(policy_network, (dict, list, tuple))
            value_function_network = policy_network

        # Clean up network configs to be passable as **kwargs to `make`.
        # Networks are given as sequential config or directly as Keras objects -> prepend "network" key to spec.
        if isinstance(policy_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            policy_network = dict(network=policy_network)
        if isinstance(value_function_network, (list, tuple, tf.keras.models.Model, tf.keras.layers.Layer)):
            value_function_network = dict(network=value_function_network)

        # Some settings could be decaying values.
        clip_ratio = Decay.make(clip_ratio)
        entropy_weight = Decay.make(entropy_weight)

        clip_rewards = clip_rewards or 0.0
        clip_value_function = clip_value_function or 0.0

        # Make sure our optimizers are defined ok.
        if default_optimizer is None:
            assert policy_optimizer and value_function_optimizer
        if policy_optimizer and value_function_optimizer:
            if default_optimizer:
                logging.warning(
                    "***WARNING: `default_optimizer` defined, but has no effect b/c `policy_optimizer` "
                    "and `value_functionn_optimizer` are already provided!"
                )
        if policy_optimizer is None:
            policy_optimizer = default_optimizer
        if value_function_optimizer is None:
            value_function_optimizer = default_optimizer

        assert time_unit in ["time_step", "env_tick"]

        # Special value for start-train parameter -> When memory full.
        if update_after == "when-memory-full":
            update_after = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif update_after == "when-memory-ready":
            update_after = memory_batch_size
        assert isinstance(update_after, int)

        # Make sure memory batch size is less than capacity.
        assert memory_batch_size <= memory_capacity

        # Make action space.
        action_space = Space.make(action_space)

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
        self.last_sync = 0
