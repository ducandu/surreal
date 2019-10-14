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

from abc import ABCMeta, abstractmethod

from surreal.algos.algo import Algo


class RLAlgo(Algo, metaclass=ABCMeta):
    """
    A reinforcement learning Algo that interacts with an environment to learn how to "predict" actions such that
    the sum of received reward signals (coming from the environment) is maximized.
    See [1].
    [1] Reinforcement Learning - An Introduction, 2nd Edition; A.G. Barto, R.S. Sutton 2018
    """
    @abstractmethod
    def event_tick(self, env, actor_time_steps, batch_positions, r, t, s_):
        """
        Tick event handler. Ticks are generated by the environment(s), which this Algo serves as a decision learner
        and maker. Somewhere inside this handler, the Algo is supposed to make an action decision - based on
        the environment's current or historic state(s) and rewards - and call back `env.act()` passing the
        Algo's resulting action decisions.

        Args:
            env (Env): The Env that triggered this tick.
            actor_time_steps (int): The time steps (across all actors that share this Algo).
            batch_positions (List[int]): The positions in the actor-batch for which the tick has happened.
            r (np.ndarray): The batch of rewards received when reaching the next state (s').
            t (np.ndarray): The terminal signals received when reaching the next state (s').

            s_ (any): The batch of next states (s'). NOTE: If `t` is True for an s' in the batch, then that s' is
                the first state of a new (reset) episode and `r` is the last reward received in the old
                episode.
        """
        raise NotImplementedError

    def event_episode_starts(self, env, actor_time_steps, batch_positions, s):
        """
        Called whenever a new episode starts in the Env.

        Args:
            env (Env): The Env that triggered this event.
            actor_time_steps (int): The time steps (across all actors that share this Algo).
            batch_positions (int): The position in the actor-batch at which a reset has happened.
            s (any): The new state (s) after a reset.
        """
        pass

    def event_episode_ends(self, env, actor_time_steps, batch_position):
        """
        Called whenever an episode ended in the Env.

        Args:
            env (Env): The Env that triggered this event.
            actor_time_steps (int): The time steps (across all actors that share this Algo).
            batch_position (int): The position in the actor-batch at which a terminal=True was observed.
            s (any): The terminal state (s') before(!) any reset.
        """
        pass

    def is_time_to(self, what, env_tick, actor_time_steps, only_after=None):
        """
        Helper method to figure out (according to some special config values), whether it's time to do something.
        E.g. update or sync a network.

        Args:
            what (str): The config to look for.
            env_tick (int): The current env tick.
            actor_time_steps (int): The current env all-actor time step.

        Returns:
            bool: Whether it is time to do `what`.
        """
        assert hasattr(self.config, "{}_frequency".format(what))
        frequency = self.config.__getattribute__("{}_frequency".format(what))
        assert hasattr(self.config, "time_unit")

        # Check, whether `what` should already be done at all.
        do_not_what_before = None
        if hasattr(self.config, "{}_after".format(what)) or only_after is not None:
            do_not_what_before = only_after if only_after is not None else \
                self.config.__getattribute__("{}_after".format(what))

        # Go by env ticks.
        if self.config.time_unit == "env_tick":
            # Not ready yet.
            if do_not_what_before is not None and env_tick < do_not_what_before:
                return False
            # `env_tick` is contiguous, no danger missing any by modulus.
            return env_tick % frequency == 0
        # Go by actor time steps.
        else:
            assert self.config.time_unit == "time_step"
            assert hasattr(self.config, "last_{}".format(what))
            # Not ready yet.
            if do_not_what_before is not None and actor_time_steps < do_not_what_before:
                return False
            # As time-steps are not contiguous, make sure, we don't miss any `what` ever.
            if actor_time_steps - self.config.__getattribute__("last_{}".format(what)) >= frequency:
                self.config.__setattr__("last_{}".format(what), actor_time_steps)
                return True
            return False

