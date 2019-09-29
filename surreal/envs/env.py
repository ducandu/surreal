# Copyright 2019 ducandu GmbH, All Rights Reserved
# (this is a modified version of the Apache 2.0 licensed RLgraph file of the same name).
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
from collections import defaultdict
import cv2
import json
import numpy as np
import threading

from surreal import PATH_EPISODE_LOGS
from surreal.debug import StoreEveryNthEpisode
from surreal.makeable import Makeable


class Env(Makeable, metaclass=ABCMeta):
    """
    An Env class used to run experiment-based RL.
    """
    def __init__(self, actors=None, render=False, action_map=None):
        """
        Args:
            actors (Union[int,List[Actor]]): The number of Actors to create and add to this env or a list of
                already existing Actors.

            render: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probably going to slow down the training.

            action_map (Optional[callable): An optional callable taking `actions` as inputs to enable e.g. discrete
                learning in continuous envs or vice-versa. The callable must output the mapped actions, ready to be
                applied in the underlying env.
        """
        super().__init__()

        # Our actors.
        self.actors = actors or []

        # The action map (if any).
        self.action_map = action_map

        # Global tick counter. Per env tick (for all actors).
        self.tick = 0
        # Time steps over all actors. n per env tick (n = number of actors).
        self.time_step_all_actors = 0
        # Time steps over all actors that share the same algo.
        self.time_steps_algos = defaultdict(int)
        # Time step(s) per episode (batched via actors axis).
        self.episodes_time_steps = None
        self.historic_episodes_lengths = []

        # Maximum number of actor-time-steps/env-ticks for some run (may not be defined, in which case,
        # algo needs to figure out what to do).
        self.max_ticks = None
        self.max_time_steps = None

        # Whether the Env is currently running (or paused).
        self.running = False
        # Running in `run_thread`. No thread used if run synchronously.
        self.run_thread = None

        # Whether to render this env.
        self.do_render = render
        # Which actor slot should we trajectory-log right now? False for none.
        self.debug_store_episode = False
        self.num_episodes = 0

        # Current state: Must be implemented as a flattened list of buffer(s) (one for each container item).
        self.state = None
        # Current rewards per actor.
        self.reward = np.zeros(shape=(len(actors),))
        # Current accumulated episode returns per actor.
        self.episodes_returns = np.zeros(shape=len(actors))
        # Historic episode returns in chronological order.
        self.historic_episodes_returns = []
        # Current terminals per actor.
        self.terminal = np.array([True] * len(actors))

    @property
    def rl_algos_to_actors(self):
        """
        Unique algos by name mapped to the different actor-slots that point to them.
        """
        ret = {}
        for i, actor in enumerate(self.actors):
            algo_name = actor.rl_algo.name if actor.rl_algo is not None else "_default_"
            if algo_name not in ret:
                ret[algo_name] = [i]
            else:
                ret[algo_name].append(i)
        return ret

    def run(self, ticks=None, actor_time_steps=None, sync=True, render=None):
        """
        Runs this Env for n time_steps (or infinitely if `time_steps` is not given or 0).

        Args:
            ticks (Optional[int]): The number of time steps (ticks) to run for.
            sync (bool): Whether to run synchronously (wait for execution to be done) or not (run in separate thread).
            render (bool): Whether to render this run. If not None, will override `self.do_render`.
        """
        # Set up time_step counters per episode (per actor).
        if self.episodes_time_steps is None:
            self.episodes_time_steps = np.zeros(shape=(len(self.actors),), dtype=np.int32)

        self.running = True
        if sync is True:
            self._run(ticks, actor_time_steps, render=render)
        else:
            # TODO: What if `run` is called, while this env is still running?
            self.run_thread = threading.Thread(target=self._run, args=[ticks, actor_time_steps])
            self.run_thread.run()

    def _run(self, ticks=None, actor_time_steps=None, render=None):
        """
        The actual loop scaffold implementation (to run in thread or synchronously).

        Args:
            ticks (Optional[int]): The number of time steps (ticks) to run for.
        """
        # Set max-time-steps.
        self.max_ticks = (actor_time_steps / len(self.actors)) if actor_time_steps is not None else \
            (ticks or float("inf"))
        self.max_time_steps = self.max_ticks * len(self.actors)

        # Build a algo-map for faster non-repetitive lookup.
        quick_algo_map = {}
        for algo_name, actor_slots in self.rl_algos_to_actors.items():
            quick_algo_map[algo_name] = self.actors[self.rl_algos_to_actors[algo_name][0]].rl_algo  # type: RLAlgo

        tick = 0
        while self.running is True and tick < self.max_ticks:
            # Loop through Actors, gather their observations/rewards/terminals and then tick each one of their
            # algos exactly once.
            for algo_name, actor_slots in self.rl_algos_to_actors.items():
                # If episode ended, send new-episode event to algo.
                for slot in actor_slots:
                    if self.terminal[slot]:
                        if tick > 0:
                            self.num_episodes += 1

                            # Switch on/off debug trajectory logging.
                            if StoreEveryNthEpisode is not False and self.debug_store_episode is False and \
                                    self.num_episodes % StoreEveryNthEpisode == 0:
                                self.debug_store_episode = (self.num_episodes, slot)
                            elif self.debug_store_episode is not False and self.debug_store_episode[1] == slot:
                                self.debug_store_episode = False

                            # Send `episode_ends` event.
                            quick_algo_map[algo_name].event_episode_ends(self, self.time_steps_algos[algo_name], slot)
                            # Log all historic returns.
                            self.historic_episodes_returns.append(self.episodes_returns[slot])
                            # Log all historic episode lengths.
                            self.historic_episodes_lengths.append(self.episodes_time_steps[slot])

                            # Log stats sometimes.
                            if slot == 0:
                                print(
                                    "Tick={} (x{} Actors); Episodes done: {}; "
                                    "Avg episode len ~ {}; Avg R ~ {:.4f}".format(
                                        self.tick, len(self.actors),
                                        self.num_episodes,
                                        int(np.mean(self.historic_episodes_lengths[-len(self.actors):])),
                                        np.mean(self.historic_episodes_returns[-len(self.actors):])
                                    )
                                )

                        # Reset episode stats.
                        self.episodes_time_steps[slot] = 0
                        self.episodes_returns[slot] = 0.0

                        # Send `episode_starts` event.
                        quick_algo_map[algo_name].event_episode_starts(
                            self, self.time_steps_algos[algo_name], slot, self.state[slot]
                        )

                # Tick the algorithm passing self.
                slots = np.array(actor_slots)
                # TODO: This may become asynchronous in the future:
                # TODO: Need to make sure that we do not expect `self.act` to be called by the algo within this tick.
                quick_algo_map[algo_name].event_tick(
                    self, self.time_steps_algos[algo_name], slots,
                    self.reward[slots], self.terminal[slots], self.state[slots]
                )
                # Accumulate episode rewards.
                self.episodes_returns[slots] += self.reward[slots]

                # Time steps (all actors with this algo).
                self.time_steps_algos[algo_name] += len(actor_slots)

                if render is True or (render is None and self.do_render is True):
                    self.render()

            # Time step for just this `run`.
            tick += 1
            # Global time step.
            self.tick += 1
            # Global time step (all actors).
            self.time_step_all_actors += len(self.actors)
            # Single episode (per actor) time_steps.
            self.episodes_time_steps += 1

        # Done with the run.
        self.running = False

        # Interrupted.
        if tick < self.max_ticks:
            # TODO: What if paused, may one resume?
            print("Run paused at tick {}.".format(tick))
        # Cleanly finished run.
        else:
            print("Run done after {} ticks.".format(tick))

        # Reset max-time-steps to undefined after each run.
        self.max_ticks = None
        self.max_time_steps = None

    def pause(self):
        self.running = False

    def act(self, actions):
        """
        Executes actions in this Env by calling the abstract `_act` method, which must be implemented by children
        of this class.

        Args:
            actions (Dict[str,any]): The action(s) to be executed by the environment.
                Keys are the Actors' names, values are the actual action structures (depending on the Actors' action
                Spaces).
        """
        # Handle debug logging of trajectories.
        s = None
        if self.debug_store_episode is not False:
            episode, slot = self.debug_store_episode
            s = self.state[slot].copy()

        # Action translations?
        if self.action_map is not None:
            actions = self.action_map(actions)
            #print(actions)

        # Call main action handler.
        self._act(actions)

        if s is not None:
            self._debug_store(
                PATH_EPISODE_LOGS + "ep_{:03d}_ts{:03d}_a{}_r{}".format(
                    episode, self.episodes_time_steps[slot], actions[slot], self.reward[slot]
                ), s
            )

    @abstractmethod
    def _act(self, actions):
        """
        Executes actions in this Env.

        Args:
            actions (Dict[str,any]): The action(s) to be executed by the environment.
                Keys are the Actors' names, values are the actual action structures (depending on the Actors' action
                Spaces).
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """
        Renders the env according to some specs given in kwargs (e.g. mode, which sub-env, etc..).
        May be implemented or not.
        """
        pass

    def terminate(self):
        """
        Clean up operation.
        May be implemented or not.
        """
        pass

    @staticmethod
    def _debug_store(path, state):
        # TODO: state Dict or Tuple, etc..
        # Probably an image.
        if len(state.shape) == 3 and (state.shape[2] == 1 or state.shape[2] == 3):
            cv2.imwrite(path+".png", state)
        # Some other data.
        else:
            with open(path, "w") as file:
                json.dump(file, state)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
