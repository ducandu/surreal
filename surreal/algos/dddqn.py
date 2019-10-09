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
import copy
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
        self.Q = Network.make(
            network=config.q_network, input_space=self.x,
            output_space=Dict(A=self.a, V=Float().with_batch()),  # dueling network outputs
            adapters=dict(A=dict(pre_network=config.dueling_a_network), V=dict(pre_network=config.dueling_v_network))
        )
        self.Qt = self.Q.copy(trainable=False)
        self.memory = PrioritizedReplayBuffer.make(
            record_space=Dict(dict(x=self.x, a=self.a, r=float, x_=self.x, t=bool, num_steps=int), main_axes="B"),
            capacity=config.memory_capacity, alpha=config.memory_alpha, beta=config.memory_beta
        )
        self.queue = deque([], maxlen=config.n_step)  # Our n-step buffer.
        self.L = DDDQNLoss()  # double/dueling/n-step Q-loss
        self.optimizer = Optimizer.make(self.config.optimizer)
        self.epsilon = Decay.make(self.config.epsilon)  # for epsilon greedy learning
        self.Phi.reset()  # make sure, Preprocessor is clean

    def event_episode_starts(self, env, actor_time_steps, batch_position, s):
        # Reset Phi at beginning of each episode (only at given batch positions).
        self.Phi.reset(batch_position)

    def event_tick(self, env, actor_time_steps, batch_positions, r, t, s_):
        # Update time-percentage value (for decaying parameters, e.g. learning-rate).
        time_percentage = actor_time_steps / (self.config.max_time_steps or env.max_time_steps)

        # Preprocess states.
        x_ = self.Phi(s_)

        # Add now-complete sars't-tuple to memory (batched).
        if actor_time_steps > 0:
            records = self.n_step(self.x.value, self.a.value, r, t, x_)
            if records:
                self.memory.add_records(records)

        # Handle ε-greedy exploration (should an ε case always be across the entire batch?).
        if random() > self.epsilon(time_percentage):
            a_ = np.argmax(self.Q(x_)["A"], axis=-1)  # "A" -> advantage values (for argmax, same as Q-values).
        else:
            a_ = self.a.sample(len(batch_positions))
        # Send the new actions back to the env.
        env.act(a_)

        # Every nth tick event -> Update network, based on Loss.
        if self.is_time_to("update", env.tick, actor_time_steps):
            weights = self.Q.get_weights(as_ref=True)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(weights)  # Only watch main Q-weights, not the target weights.
                records, indices = self.memory.get_records_with_indices(self.config.memory_batch_size)
                L, abs_td_errors = self.L(records, self.Q, self.Qt, self.config)
                self.optimizer.apply_gradients(list(zip(tape.gradient(L, weights), weights)), time_percentage)
                # Update prioritized replay records with Add td_errors to prioritized replay as
                self.memory.update_records(indices, abs_td_errors)

        # Every mth tick event -> Synchronize target Q-net.
        if self.is_time_to("sync", env.tick, actor_time_steps):
            self.Qt.sync_from(self.Q)

        # Store actions and states for next tick (they form the incomplete next sars't-tuple).
        self.x.assign(x_)
        self.a.assign(a_)

    def n_step(self, x, a, r, t, x_):
        self.queue.append(dict(x=x, a=a, r=r, t=t))
        records = {"x": [], "a": [], "r": [], "t": [], "x_": [], "n": []}
        if isinstance(x_, tf.Tensor):
            x_ = x_.numpy()
        else:
            x_ = copy.deepcopy(x_)
        if isinstance(t, tf.Tensor):
            t = t.numpy()
        else:
            t = copy.deepcopy(t)
        #num_steps_list = []
        batch_size = t.shape[0]
        r_sum = 0.0
        num_steps = np.array([1] * batch_size)
        # N-step loop (moving back in deque).
        for i in reversed(range(len(self.queue))):
            record = self.queue[i]
            # Add up rewards as we move back.
            r_sum += record["r"]

            # Batch loop.
            for batch_index in range(batch_size):
                # Reached n-steps OR a terminal (s' at i is already a reset-state (first one in episode)).
                if i == 0 or self.queue[i-1]["t"][batch_index]:
                    # Do not include samples smaller than n-steps w/o a terminal.
                    if self.config.n_step_only is False or num_steps[batch_index] == self.config.n_step or t[batch_index]:
                        # Add done n-step record to our records.
                        records["x"].append(record["x"][batch_index])
                        records["a"].append(record["a"][batch_index])
                        records["r"].append(r_sum[batch_index])
                        records["t"].append(t[batch_index])
                        records["x_"].append(x_[batch_index])
                        records["n"].append(num_steps[batch_index])
                    #if record["x"][batch_index][0] == 1.0 and record["a"][batch_index] == 2 and num_steps[batch_index] == 1 and r_sum[batch_index] == -0.1 and t[batch_index]:
                    #    print("here")
                    if i > 0 and self.queue[i-1]["t"][batch_index]:
                        r_sum[batch_index] = 0.0
                        num_steps[batch_index] = 0
                        x_[batch_index] = record["x"][batch_index]  # the reset-state
                        t[batch_index] = True

            # Keep multiplying by discount factor.
            r_sum *= self.config.gamma
            num_steps += 1

        # Return all records (non-horizontally).
        if len(records["x"]) > 0:
            return dict(
                x=np.array(records["x"]), a=np.array(records["a"]), r=np.array(records["r"]),
                t=np.array(records["t"]), x_=np.array(records["x_"]), num_steps=np.array(records["n"])
            )


def dueling(output, a):
    """
    Dueling layer logic (output is split between advantage nodes and single value node), then
    combined as:
    Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM a* over all A(s,a*)

    Args:
        output (any): The NN output (should be a dict with "A" (advantage) and "V" (value) keys).
        a (np.array): The (int) actions to pick the Q-value for.

    Returns:
        any: The q-value for the given action using the NN's dueling output (advantages and single value outputs).
    """
    q_values = output["V"] + tf.gather_nd(output["A"], tf.reshape(a, (-1, 1)), batch_dims=1) - \
        (tf.reduce_sum(output["A"], axis=-1) / output["A"].shape[-1])  # -1: last dim of a == |A|
    return q_values


class DDDQNLoss(LossFunction):
    """
    The DDDQN loss function (expected (over some batch) dueling/double-Q/n-step TD learning loss):

    L = E[(TDtarget(s') - Q(s,a))²]

    Where:
        E = expectation over a prioritized(!) memory batch. Prioritization according to previous abs(TD-error) terms.
        TDtarget(s') = r0 + γr1 + γ²*r2 + ... + γ^n Qt(s', argmax a' Q(s',a'))
        s' = state after n-steps.
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
            config (DDDQNConfig): A DDQNConfig object, of which this LossFunction uses some properties.

        Returns:
            Tuple:
                tf.Tensor: The single loss value (0D). See formula above.
                tf.Tensor: The (already abs'd) TD-errors, e.g. useful as weights in a prioritized replay buffer.
        """
        x, a, r, x_, t, n = samples["x"], samples["a"], samples["r"], samples["x_"], samples["t"], samples["num_steps"]
        # "A" -> advantage values (for argmax, this is the same as argmaxing over the Q-values).
        a_ = tf.argmax(q_net(x_)["A"], axis=-1, output_type=tf.int32)  # argmax a' (Q(s'))
        target_q_xp_ap = dueling(target_q_net(x_), a_)  # Qt(s',a')
        td_targets = r + (config.gamma ** n) * tf.where(t, tf.zeros_like(target_q_xp_ap), target_q_xp_ap)
        td_errors = td_targets - dueling(q_net(x), a)  # Q(s,a)
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
            n_step=1, n_step_only=True,
            max_time_steps=None, update_after=0,
            update_frequency=16, sync_frequency=4, time_unit="time_steps"
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

            n_step_only (bool): Whether to exclude samples that are shorter than `n_step` AND don't have a terminal
                at the end.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_steps"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.
            sync_frequency (int): The frequency (in `time_unit`) with which to sync our target network.
            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.
        """
        # Special value for start-train parameter -> When memory full.
        if update_after == "when-memory-full":
            assert time_unit == "time_steps"
            update_after = memory_capacity
        # Special value for start-train parameter -> When memory has enough records to pull a batch.
        elif update_after == "when-memory-ready":
            assert time_unit == "time_steps"
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
