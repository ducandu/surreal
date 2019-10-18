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
import logging
import numpy as np
import re
import tensorflow as tf
import threading
import time

from surreal import PATH_EPISODE_LOGS
from surreal.debug import StoreEveryNthEpisode
from surreal.makeable import Makeable
from surreal.utils.util import SMALL_NUMBER


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

    def run(self, ticks=None, actor_time_steps=None, episodes=None, sync=True, render=None):
        """
        Runs this Env for n time_steps (or infinitely if `time_steps` is not given or 0).

        Args:
            ticks (Optional[int]): The number of env-ticks to run for.
            actor_time_steps (Optional[int]): The number of single actor time-steps to run for.
            episodes (Optional[int]): The number of episodes (across all actors) to execute.
            sync (bool): Whether to run synchronously (wait for execution to be done) or not (run in separate thread).
            render (bool): Whether to render this run. If not None, will override `self.do_render`.
        """
        # Set up time_step counters per episode (per actor).
        if self.episodes_time_steps is None:
            self.episodes_time_steps = np.zeros(shape=(len(self.actors),), dtype=np.int32)

        self.running = True
        if sync is True:
            self._run(ticks, actor_time_steps, episodes, render=render)
        else:
            # TODO: What if `run` is called, while this env is still running?
            self.run_thread = threading.Thread(target=self._run, args=[ticks, actor_time_steps, episodes])
            self.run_thread.run()

    def _run(self, ticks=None, actor_time_steps=None, episodes_x=None, render=None):
        """
        The actual loop scaffold implementation (to run in thread or synchronously).

        Args:
            ticks (Optional[int]): The number of env-ticks to run for.
            actor_time_steps (Optional[int]): The number of single actor time-steps to run for.
            #currently not supported! episodes (Optional[int]): The number of episodes (across all actors) to execute.
        """
        # Set max-time-steps.
        # TODO: Have env keep track of time_percentage, not algo!
        # TODO: Distinguish between time_percentage this run and time_percentage overall (through many different `run` calls).
        # TODO: Solve dilemma of when only `episodes` given. What's the number of ticks then? Do we know the lengths of episodes up front? We could calculate time_percentage based on episodes done.
        self.max_ticks = (actor_time_steps / len(self.actors)) if actor_time_steps is not None else \
            (ticks or float("inf"))
        self.max_time_steps = self.max_ticks * len(self.actors)
        #self.max_episodes = episodes if episodes is not None else float("inf")

        # Build a algo-map for faster non-repetitive lookup.
        quick_algo_map = {}
        for algo_name, actor_slots in self.rl_algos_to_actors.items():
            quick_algo_map[algo_name] = self.actors[self.rl_algos_to_actors[algo_name][0]].rl_algo  # type: RLAlgo

        tick = 0
        #episode = 0
        last_time_measurement = time.time()
        last_episode_measurement = 0
        last_actor_ts_measurement = 0
        last_tick_measurement = 0
        while self.running is True and tick < self.max_ticks:  # and episode < self.max_episodes:
            # Loop through Actors, gather their observations/rewards/terminals and then tick each one of their
            # algos exactly once.
            for algo_name, actor_slots in self.rl_algos_to_actors.items():
                algo = quick_algo_map[algo_name]
                # If episode ended, send new-episode event to algo.
                for slot in actor_slots:
                    if self.terminal[slot]:
                        if tick > 0:
                            self.num_episodes += 1
                            #episode += 1
                            last_episode_measurement += 1

                            # Switch on/off debug trajectory logging.
                            if StoreEveryNthEpisode is not False and self.debug_store_episode is False and \
                                    self.num_episodes % StoreEveryNthEpisode == 0:
                                self.debug_store_episode = (self.num_episodes, slot)
                            elif self.debug_store_episode is not False and self.debug_store_episode[1] == slot:
                                self.debug_store_episode = False

                            # Log all historic returns.
                            self.historic_episodes_returns.append(self.episodes_returns[slot])
                            # Log all historic episode lengths.
                            self.historic_episodes_lengths.append(self.episodes_time_steps[slot])

                            # Send `episode_ends` event.
                            algo.event_episode_ends(self, self.time_steps_algos[algo_name], slot)
                            self.summarize_episode(algo)

                            # Log stats sometimes.
                            if slot == 0:
                                t = time.time()
                                delta_t = (t - last_time_measurement) or SMALL_NUMBER
                                logging.info(
                                    "Ticks(tx)={} Time-Steps(ts)={} ({} Actors); Episodes(ep)={}; "
                                    "Avg ep len~{}; Avg R~{:.4f}; tx/s={:d}; ts/s={:d}; ep/s={:.2f}".format(
                                        self.tick, self.time_step_all_actors, len(self.actors),
                                        self.num_episodes,
                                        int(np.mean(self.historic_episodes_lengths[-len(self.actors):])),
                                        np.mean(self.historic_episodes_returns[-len(self.actors):]),
                                        int(last_tick_measurement / delta_t),
                                        int(last_actor_ts_measurement / delta_t),  # TODO: these two are wrong
                                        last_episode_measurement / delta_t
                                    )
                                )
                                last_time_measurement = t
                                last_episode_measurement = 0
                                last_actor_ts_measurement = 0
                                last_tick_measurement = 0

                        # Reset episode stats.
                        self.episodes_time_steps[slot] = 0
                        self.episodes_returns[slot] = 0.0

                        # Send `episode_starts` event.
                        algo.event_episode_starts(self, self.time_steps_algos[algo_name], slot, self.state[slot])
                        #self.summarize(algo)

                # Tick the algorithm passing self.
                slots = np.array(actor_slots)

                # TODO: This may become asynchronous in the future:
                # TODO: Need to make sure that we do not expect `self.act` to be called by the algo within this tick.
                algo.event_tick(self, self.time_steps_algos[algo_name], slots, self.reward[slots], self.terminal[slots],
                                self.state[slots])
                self.summarize_tick(algo)

                # Accumulate episode rewards.
                self.episodes_returns[slots] += self.reward[slots]

                # Time steps (all actors with this algo).
                self.time_steps_algos[algo_name] += len(actor_slots)

                if render is True or (render is None and self.do_render is True):
                    self.render()

            # Time step for just this `run`.
            tick += 1
            last_tick_measurement += 1
            # Global time step.
            self.tick += 1
            # Global time step (all actors).
            self.time_step_all_actors += len(self.actors)
            last_actor_ts_measurement += len(self.actors)
            # Single episode (per actor) time_steps.
            self.episodes_time_steps += 1

        # Done with the run.
        self.running = False

        # Interrupted.
        if tick < self.max_ticks:
            # TODO: What if paused, may one resume?
            logging.info("Run paused at tick {}.".format(tick))
        # Cleanly finished run.
        else:
            logging.info("Run done after {} ticks.".format(tick))

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

    def point_all_actors_to_algo(self, rl_algo):
        """
        Points all of this Env's Actors to the given RLAlgo object.

        Args:
            rl_algo (RLAlgo): The RLAlgo to point to.
        """
        for actor in self.actors:
            actor.set_algo(rl_algo)

    def summarize_tick(self, algo):
        """
        Writes summary information (iff debug.UseTfSummaries is true) to the Algo's `summary_writer` object.
        Summary information and setup can be passed into the Algo via `config.summaries`, which is a list of items,
        that will simply be executed on the Algo context (prepending "algo."):
        E.g.:
        "Q[0](np.array([[1.0, 0.0]]))": Will summarize the result of calling `self.Q[0](...)` on the Algo object.
        "L_critic": Will summarize the value of `self.L_critic` on the Algo object.

        Args:
            algo (Algo): The Algo to summarize.
        """
        # Summaries not setup.
        if algo.summary_writer is None:
            return

        with algo.summary_writer.as_default():
            for summary in algo.config.summaries:
                name = code_ = summary
                # Tuple/List of 2: Summary name + prop.
                if isinstance(summary, (list, tuple)) and len(summary) == 2:
                    name = summary[0]
                    code_ = summary[1]

                # Ignore episode stats.
                if re.match(r'^episode\..+$', name):
                    continue

                l_dict = {"algo": algo}
                # Execute the code.
                try:
                    exec("result = algo.{}".format(code_), None, l_dict)
                # This should never really fail.
                except Exception as e:
                    logging.error("Summary ERROR '{}' in '{}'!".format(e, code_))
                    continue

                result = l_dict["result"]
                # Array or Tensor?
                if isinstance(result, (np.ndarray, tf.Tensor)):
                    if result.shape == ():
                        tf.summary.scalar(name, result, step=self.tick)
                    elif result.shape == (1,):
                        tf.summary.scalar(name, tf.squeeze(result), step=self.tick)
                    # TODO: Add images, etc..?
                    else:
                        tf.summary.histogram(name, result, step=self.tick)
                # Assume scalar.
                else:
                    tf.summary.scalar(name, result, step=self.tick)

    def summarize_episode(self, algo):
        """
        See `summarize`.
        This method only considers entries in `algo.summaries` that start with "episode.[some prop]".
        Currently only supports props: `episode.return` and `episode.time_steps`.

        Args:
            algo (Algo): The Algo to summarize.
            #num_episodes (int): The number of episodes that have been finished.
            #episode_return (float): The overall return (undiscounted) of the finished episode.
            #episode_time_steps (int): The length of the finished episode in time-steps.
        """
        # Summaries not setup.
        if algo.summary_writer is None:
            return

        with algo.summary_writer.as_default():
            for summary in algo.config.summaries:
                name = code_ = summary
                # Tuple/List of 2: Summary name + prop.
                if isinstance(summary, (list, tuple)) and len(summary) == 2:
                    name = summary[0]
                if not re.match(r'^episode\..+', name):
                    continue
                value = self.historic_episodes_returns[-1] if name == "episode.return" else \
                    self.historic_episodes_lengths[-1]
                tf.summary.scalar(name, value, step=self.num_episodes)

    @staticmethod
    def _debug_store(path, state):
        # TODO: state Dict or Tuple, etc..
        # Probably an image.
        if len(state.shape) == 3 and (state.shape[2] == 1 or state.shape[2] == 3):
            cv2.imwrite(path+".png", state)
        # Some other data.
        else:
            logging.warning("***WARNING: No mechanism yet for state debug-saving if not image!")
            #with open(path, "w") as file:
            #    json.dump(file, state)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
