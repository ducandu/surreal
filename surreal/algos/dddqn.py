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
from surreal.components import Network, PrioritizedReplayBuffer, Optimizer, Decay, Preprocessor, LossFunction, NStep
from surreal.config import AlgoConfig
from surreal.spaces import Dict, Float, Int, Space


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
        self.x = self.Phi(Space.make(config.state_space).with_batch())  # preprocessed states (x)
        self.a = Space.make(config.action_space).with_batch()  # actions (a)
        self.Q = Network.make(
            network=config.q_network, input_space=self.x,
            output_space=Dict(A=self.a, V=Float().with_batch()),  # dueling network outputs
            adapters=dict(A=dict(pre_network=config.dueling_a_network), V=dict(pre_network=config.dueling_v_network))
        )
        self.Qt = self.Q.copy(trainable=False)
        self.memory = PrioritizedReplayBuffer.make(
            record_space=Dict(dict(s=self.x, a=self.a, r=float, t=bool, n=int), main_axes="B"),
            capacity=config.memory_capacity, alpha=config.memory_alpha,
            beta=config.memory_beta, next_record_setup=dict(s="s_", n_step=config.n_step)
        )
        self.n_step = NStep(config.gamma, n_step=config.n_step, n_step_only=True)  # N-step component
        self.L = DDDQNLoss()  # double/dueling/n-step Q-loss
        self.optimizer = Optimizer.make(self.config.optimizer)
        self.epsilon = Decay.make(self.config.epsilon)  # for epsilon greedy learning
        self.Phi.reset()  # make sure, Preprocessor is clean

    def update(self, samples, time_percentage):
        weights = self.Q.get_weights(as_ref=True)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(weights)  # Only watch main Q-weights, not the target weights.
            L, abs_td_errors = self.L(samples, self.Q, self.Qt, self.config)
            self.optimizer.apply_gradients(list(zip(tape.gradient(L, weights), weights)), time_percentage)
        return L, abs_td_errors

    def event_episode_starts(self, event):
        # Reset Phi at beginning of each episode (only at given batch positions).
        self.Phi.reset(event.current_actor_slot)

    def event_tick(self, event):
        # Update time-percentage value (for decaying parameters, e.g. learning-rate).
        time_percentage = event.actor_time_steps / (self.config.max_time_steps or event.env.max_time_steps)

        # Preprocess states.
        x_ = self.Phi(event.s_)

        # Add now-complete sars't-tuple to memory (batched).
        if event.actor_time_steps > 0:
            records = self.n_step(self.x.value, self.a.value, event.r, event.t, x_)
            if records:
                self.memory.add_records(records)

        # Handle ε-greedy exploration (should an ε case always be across the entire batch?).
        if random() > self.epsilon(time_percentage):
            a_ = np.argmax(self.Q(x_)["A"], axis=-1)  # "A" -> advantage values (for argmax, same as Q-values).
        else:
            a_ = self.a.sample(len(event.actor_slots))
        # Send the new actions back to the env.
        event.env.act(a_)

        # Every nth tick event -> Update network, based on Loss and update memory's priorities based on the TD-errors.
        if self.is_time_to("update", event.env.tick, event.actor_time_steps):
            samples, indices = self.memory.get_records_with_indices(self.config.memory_batch_size)
            _, abs_td_errors = self.update(samples, time_percentage)
            # Update prioritized replay records.
            self.memory.update_records(indices, abs_td_errors)

        # Every mth tick event -> Synchronize target Q-net.
        if self.is_time_to("sync", event.env.tick, event.actor_time_steps):
            self.Qt.sync_from(self.Q)

        # Store actions and states for next tick (they form the incomplete next sars't-tuple).
        self.x.assign(x_)
        self.a.assign(a_)


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
        x, a, r, x_, t, n = samples["s"], samples["a"], samples["r"], samples["s_"], samples["t"], samples["n"]
        # "A" -> advantage values (for argmax, this is the same as argmaxing over the Q-values).
        a_ = tf.argmax(q_net(x_)["A"], axis=-1, output_type=tf.int32)  # argmax a' (Q(s'))
        target_q_xp_ap = dueling(target_q_net(x_), a_)  # Qt(s',a')
        td_targets = r + (config.gamma ** n) * tf.where(t, tf.zeros_like(target_q_xp_ap), target_q_xp_ap)
        td_errors = td_targets - dueling(q_net(x), a)  # Q(s,a)
        return 0.5 * tf.reduce_mean(td_errors ** 2), tf.abs(td_errors)


class DDDQNConfig(AlgoConfig):
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
            n_step=1, #n_step_only=True,
            max_time_steps=None, update_after=0,
            update_frequency=16, sync_frequency=4, time_unit="time_step",
            summaries=None
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

            n_step (int): The number of steps (n) to "look ahead/back" when converting 1-step tuples into n-step ones.
                "Normal" Q-learning or TD(0) has `n_step` of 1.

            #n_step_only (bool): Whether to exclude samples that are shorter than `n_step` AND don't have a terminal
            #    at the end.

            max_time_steps (Optional[int]): The maximum number of time steps (across all actors) to learn/update.
                If None, use a value given by the environment.

            update_after (Union[int,str]): The `time_unit`s to wait before starting any updates.
                Special values (only valid iff time_unit == "time_step"!):
                - "when-memory-full" for same as `memory_capacity`.
                - when-memory-ready" for same as `memory_batch_size`.

            update_frequency (int): The frequency (in `time_unit`) with which to update our Q-network.
            sync_frequency (int): The frequency (in `time_unit`) with which to sync our target network.
            time_unit (str["time_step","env_tick"]): The time units we are using for update/sync decisions.

            summaries (List[any]): A list of summaries to produce if `UseTfSummaries` in debug.json is true.
                In the simplest case, this is a list of `self.[...]`-property names of the SAC object that should
                be tracked after each tick.
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
