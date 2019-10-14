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
from random import random
import tensorflow as tf

from surreal.algos.rl_algo import RLAlgo
from surreal.components import Network, ReplayBuffer, Optimizer, Decay, Preprocessor, LossFunction
from surreal.config import Config
from surreal.spaces import Dict, Int, Space


class DQN2015(RLAlgo):
    """
    An implementation of the "classic" DQN from the 2015 paper:
    [1] Human-level control through deep reinforcement learning - V. Mnih, K. Kavukcuoglu, D. Silver, et al. - Deepmind
        Nature Feb 2015.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.Phi = Preprocessor.make(config.preprocessor)  # states preprocessor (Phi, like in paper)
        self.x = self.Phi(Space.make(self.config.state_space).with_batch())  # preprocessed states ('x', like in paper)
        self.a = Space.make(self.config.action_space).with_batch()  # actions (a)
        self.Q = Network.make(network=config.q_network, output_space=self.a, input_space=self.x)  # Q-network
        self.Qt = self.Q.copy(trainable=False)  # target Q-network
        self.memory = ReplayBuffer.make(  # simple replay buffer
            record_space=Dict(dict(x=self.x, a=self.a, r=float, t=bool), main_axes="B"),
            capacity=config.memory_capacity, next_record_setup=dict(x="x_")
        )
        self.L = DQN2015Loss()  # plain Q-loss (quadratic, 1-step TD)
        self.optimizer = Optimizer.make(self.config.optimizer)
        self.epsilon = Decay.make(self.config.epsilon)  # for epsilon greedy learning
        self.Phi.reset()  # make sure, Preprocessor is clean

    def event_episode_starts(self, env, actor_time_steps, batch_position, s):
        # Reset Phi at beginning of each episode (only at given batch positions).
        self.Phi.reset(batch_position)

    # Env tick event -> Act in env and collect samples in replay-buffer.
    def event_tick(self, env, actor_time_steps, batch_positions, r, t, s_):
        # Update time-percentage value (for decaying parameters, e.g. learning-rate).
        time_percentage = actor_time_steps / (self.config.max_time_steps or env.max_time_steps)

        # Preprocess states. Call preprocessed states 'x', just like in the paper.
        x_ = self.Phi(s_)

        # Add now-complete sars't-tuple to memory (batched).
        if actor_time_steps > 0:
            self.memory.add_records(dict(x=self.x.value, a=self.a.value, r=r, x_=x_, t=t))

        # Handle ε-greedy exploration (should an ε case always be across the entire batch?).
        if random() > self.epsilon(time_percentage):
            a_ = np.argmax(self.Q(x_), axis=-1)
        else:
            a_ = self.a.sample(len(batch_positions))
        # Send the new actions back to the env.
        env.act(a_)

        # Every nth tick event -> Update network, based on Loss.
        if self.is_time_to("update", env.tick, actor_time_steps):
            weights = self.Q.get_weights(as_ref=True)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(weights)  # Only watch main Q-weights, not the target weights.
                L = self.L(self.memory.get_records(self.config.memory_batch_size), self.Q, self.Qt, self.config)
                self.optimizer.apply_gradients(list(zip(tape.gradient(L, weights), weights)), time_percentage)

        # Every mth tick event -> Synchronize target Q-net.
        if self.is_time_to("sync", env.tick, actor_time_steps):
            self.Qt.sync_from(self.Q)

        # Store actions and states for next tick (they form the incomplete next sars't-tuple).
        self.x.assign(x_)
        self.a.assign(a_)


class DQN2015Loss(LossFunction):
    """
    The DQN2015 loss function (expected (over some batch) quadratic, 1-step TD loss):

    L = E[(TDtarget(s') - Q(s,a))²]

    Where:
        E = expectation over some uniform memory batch.
        TDtarget(s') = r + γ max a' (Qt(s'))
        Qt = target Q-network (synchronized every m time steps, where m >> Q update frequency).
        γ = discount factor
    """
    def call(self, samples, q_net, target_q_net, config):
        """
        Args:
            samples (Dict[states,actions,rewards,next-states,terminals]): The batch to push through the loss function
                to get an expectation value (mean over all batch items).

            q_net (Network): The Q-network.
            target_q_net (Network): The target Q-network.
            config (DDDQNConfig): A DQN2015Config object, of which this LossFunction uses some properties.

        Returns:
            tf.Tensor: The single loss value (0D). See formula above.
        """
        x, a, r, x_, t = samples["x"], samples["a"], samples["r"], samples["x_"], samples["t"]
        target_q_xp_ap = tf.reduce_max(target_q_net(x_), axis=-1)  # max a'(target-q(x',a'))
        td_targets = r + config.gamma * tf.where(t, tf.zeros_like(target_q_xp_ap), target_q_xp_ap)
        return 0.5 * tf.reduce_mean((td_targets - q_net(x, a)) ** 2)


class DQN2015Config(Config):
    """
    Config object for a DQN2015 Algorithm.
    """
    def __init__(
            self, *,
            q_network, optimizer,
            state_space, action_space,
            preprocessor=None,
            gamma=0.99, epsilon=(1.0, 0.0), memory_capacity=10000,
            memory_batch_size=512,
            max_time_steps=None, update_after=0,
            update_frequency=16, sync_frequency=4, time_unit="time_step"
    ):
        """
        Args:
            q_network (Network): The Q-network to use as a function approximator for the learnt Q-function.
            optimizer (Optimizer): The optimizer to use for the Q-network.
            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.
            preprocessor (Preprocessor): The preprocessor (if any) to use.
            gamma (float): The discount factor (gamma).
            epsilon (Decay): The epsilon spec used for epsilon-greedy learning.
            memory_capacity (int): The memory's capacity (max number of records to store).
            memory_batch_size (int): The batch size to use for updating from memory.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_step"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.
            sync_frequency (int): The frequency (in `time_unit`) with which to sync our target network.
            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.
        """
        assert time_unit in ["time_step", "env_tick"]

        # Special value for start-train parameter -> When memory full.
        if update_after == "when-memory-full":
            assert time_unit == "time_step"
            update_after = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif update_after == "when-memory-ready":
            assert time_unit == "time_step"
            update_after = memory_batch_size
        assert isinstance(update_after, int)

        # Make sure sync-freq > update-freq:
        assert sync_frequency > update_frequency

        # Make sure memory batch size is less than capacity.
        assert memory_batch_size <= memory_capacity

        # Make sure action space is single Int(1D) with categories.
        assert isinstance(action_space, Int) and action_space.rank == 0 and action_space.num_categories > 1

        super().__init__(locals())  # Config will store all c'tor variables automatically.

        # Keep track of which time-step stuff happened. Only important for by-time-step frequencies.
        self.last_update = 0
        self.last_sync = 0
