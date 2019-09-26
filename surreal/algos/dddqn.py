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

from collections import deque
import numpy as np
from random import random
import tensorflow as tf

from surreal.algos.rl_algo import RLAlgo
from surreal.components import Network, PrioritizedReplayBuffer, Optimizer, Decay, Preprocessor, LossFunction
from surreal.config import Config
from surreal.spaces import Dict, Float, Bool, Int, Space


class DDDQN(RLAlgo):
    """
    An implementation of an "advanced" double/dueling/n-step/priority-buffer-based DQN, following the papers:

    [1] Human-level control through deep reinforcement learning - V. Mnih, K. Kavukcuoglu, D. Silver, et al. - Deepmind
        Nature Feb 2015.
    [2] Deep Reinforcement Learning with Double Q-learning - Hasselt, Guez, Silver - DeepMind 2015.
    [3] Dueling Network Architectures for Deep Reinforcement Learning - Wang et al. - DeepMind 2016.
    [4] Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target - Hernandez-Garcia,
        Sutton - University of Alberta 2019.
    """
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.Phi = Preprocessor.make(config.preprocessor)
        self.x = self.Phi(Space.make(self.config.state_space).with_batch())  # preprocessed states (x)
        self.a = Space.make(self.config.action_space).with_batch()  # actions (a)
        self.r = Float().with_batch()
        self.t = Bool().with_batch()
        self.Q = Network.make(
            network=config.q_network, input_space=self.x,
            output_space=Dict(A=self.a, V=Float().with_batch()),  # dueling network outputs
            adapters=dict(A=dict(pre_network=config.dueling_a_network), V=dict(pre_network=config.dueling_v_network))
        )
        self.Qt = self.Q.copy(trainable=False)
        self.memory = PrioritizedReplayBuffer.make(
            record_space=Dict(dict(x=self.x, a=self.a, r=self.r, x_=self.x, t=self.t, num_steps=Int().with_batch())),
            capacity=config.memory_capacity, alpha=config.memory_alpha, beta=config.memory_beta
        )
        self.queue = deque([], maxlen=config.n_step)  # Our n-step buffer.
        self.queue_index = -1
        self.L = DDDQNLoss()
        self.optimizer = Optimizer.make(self.config.optimizer)
        self.epsilon = Decay.make(self.config.epsilon)
        self.Phi.reset()

    def event_episode_starts(self, env, time_steps, batch_position, s):
        # Reset Phi at beginning of each episode (only at given batch positions).
        self.Phi.reset(batch_position)

    def event_tick(self, env, time_steps, batch_positions, r, t, s_):
        # Update time-percentage value.
        time_percentage = time_steps / (self.config.max_time_steps or env.max_time_steps)

        # Preprocess states.
        x_ = self.Phi(s_)

        # Add sars't-tuples to memory (batched).
        if time_steps > 0:
            records = self.n_step()
            self.memory.add_records(dict(x=self.x.value, a=self.a.value, r=r, x_=x_, t=t))

        # Handle ε-greedy exploration (should an ε case always be across the entire batch?).
        if random() > self.epsilon(time_percentage):
            a_ = np.argmax(self.Q(x_)["A"], axis=-1)  # "A" -> advantage values (for argmax, same as Q-values).
        else:
            a_ = self.a.sample(len(batch_positions))
        # Send the new actions back to the env.
        env.act(a_)

        # Every nth tick event -> Update network, based on Loss.
        if env.tick % (int(self.config.update_frequency / len(batch_positions)) or 1) == 0 and \
                time_steps > self.config.steps_before_update:
            weights = self.Q.get_weights(as_ref=True)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(weights)  # Only watch main Q-weights, not the target weights.
                records, indices = self.memory.get_records_with_indices(self.config.memory_batch_size)
                L, abs_td_errors = self.L(records, self.Q, self.Qt, self.config)
                self.optimizer.apply_gradients(list(zip(tape.gradient(L, weights), weights)), time_percentage)
                # Update prioritized replay records with Add td_errors to prioritized replay as
                self.memory.update_records(indices, abs_td_errors)

        # Every mth tick event -> Synchronize target Q-net.
        if env.tick % (int(self.config.sync_frequency / len(batch_positions)) or 1) == 0:
            self.Qt.sync_from(self.Q)

        # Store sars't-tuple in deque for n-step estimation.
        self.queue.append([x_, a_, ])
        #self.x.assign(x_)
        #self.a.assign(a_)

    def n_step(self):
        pass


def dueling(output, a, num_actions):
    """
    Dueling layer logic (output is split between advantage nodes and single value node), then
    combined as:
    Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM a* over all A(s,a*)

    Args:
        output (any): The NN output (should be a dict with "A" (advantage) and "V" (value) keys).
        a (np.array): The (int) actions to pick the Q-value for.
        num_actions (int): The number of possible actions (property `num_categories` of the action Space).

    Returns:
        any: The q-value for the given action using the NN's dueling output (advantages and single value outputs).
    """
    q_values = output["V"] + tf.gather_nd(output["A"], tf.reshape(a, (-1, 1)), batch_dims=1) - \
               (tf.reduce_sum(output["A"], axis=-1) / num_actions)
    return q_values


class DDDQNLoss(LossFunction):
    """
    The DDDQN loss function (double-Q learning loss):

    L = E[(TDtarget(s') - Q(s,a)) ** 2]

    Where:
        E = expectation over some uniform memory batch.
        TDtarget(s') = r + γ Qt(s', argmax a'(Q(s')))
        Qt = target Q-network (synchronized every n time steps, where n >> Q update frequency).
    """
    def call(self, samples, q_net, target_q_net, config):
        """
        Args:
            samples (Dict[states,actions,rewards,next-states,terminals]): The batch to push through the loss function
                to get an expectation value (mean over all batch items).

            q_net (Network): The Q-network.
            target_q_net (Network): The target Q-network.
            config (DDDQNConfig): A DDQNConfig object, of which this LossFunction uses some properties.

        Returns:
            Tuple:
                tf.Tensor: The single loss value (0D). See formula above.
                tf.Tensor: The (already abs'd) TD-errors, e.g. useful as weights in a prioritized replay buffer.
        """
        x, a, r, x_, t = samples["x"], samples["a"], samples["r"], samples["x_"], samples["t"]
        # "A" -> advantage values (for argmax, this is the same as argmaxing over the Q-values).
        a_ = tf.argmax(q_net(x_)["A"], axis=-1, output_type=tf.int32)  # argmax a' (Q(s'))
        target_q_xp_ap = dueling(target_q_net(x_), a_, config.action_space.num_categories)  # Qt(s',a')
        td_targets = r + (config.gamma ** config.n_step) * tf.where(t, tf.zeros_like(target_q_xp_ap), target_q_xp_ap)
        td_errors = td_targets - dueling(q_net(x), a, config.action_space.num_categories)  # Q(s,a)
        return 0.5 * tf.reduce_mean(td_errors ** 2), tf.abs(td_errors)


class DDDQNConfig(Config):
    """
    Config object for a DDDQN Algorithm.
    """
    def __init__(
            self, *,
            q_network, dueling_a_network, dueling_v_network, optimizer,
            state_space, action_space,
            preprocessor=None,
            gamma=0.99, epsilon=(1.0, 0.0),
            memory_capacity=10000, memory_alpha=1.0, memory_beta=0.0, memory_batch_size=512,
            n_step=1,
            max_time_steps=None, update_frequency=16, steps_before_update=0, sync_frequency=4
    ):
        """
        Args:
            q_network (Network): The Q-network to use as a function approximator for the learnt Q-function.
            dueling_a_network (Network): The Q-network to use as a function approximator for the learnt Q-function.
            dueling_a_network  (Network): The Q-network to use as a function approximator for the learnt Q-function.
            optimizer (Optimizer): The optimizer to use for the Q-network.
            state_space (Space): The state/observation Space.
            action_space (Space): The action Space.
            preprocessor (Preprocessor): The preprocessor (if any) to use.
            gamma (float): The discount factor (gamma).
            epsilon (Decay): The epsilon spec used for epsilon-greedy learning.
            memory_capacity (int): The memory's capacity (max number of records to store).
            memory_alpha (float): The alpha value for the PrioritizedReplayBuffer.
            memory_beta (float): The beta value for the PrioritizedReplayBuffer.
            memory_batch_size (int): The batch size to use for updating from memory.

            n_step (int): How many steps to "look ahead" in an n-step discounted Q-learning setup.
                "Normal" Q-learning or TD(0) has `n_step` of 1.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_frequency (int): The frequency (in all-actor time steps) with which to update our Q-network.

            steps_before_update (Union[int,str]): The steps (across all actors) to take before starting any updates.
                Special values: "when-memory-full" for same as `memory_capacity`, "when-memory-ready" for same
                as `memory_batch_size`.

            sync_frequency (int): The frequency (in all-actor time steps) with which to synch our target network.
        """
        # Special value for start-train parameter -> When memory full.
        if steps_before_update == "when-memory-full":
            steps_before_update = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif steps_before_update == "when-memory-ready":
            steps_before_update = memory_batch_size
        assert isinstance(steps_before_update, int)

        # Make sure sync-freq > update-freq:
        assert sync_frequency > update_frequency
        # Make sure memory batch size is less than capacity.
        assert memory_batch_size <= memory_capacity

        # Make sure action space is single Int(1D) with categories.
        assert isinstance(action_space, Int) and action_space.rank == 0 and action_space.num_categories > 1

        super().__init__(locals())  # Config will store all c'tor variables automatically.
