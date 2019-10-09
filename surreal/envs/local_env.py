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
import multiprocessing
import numpy as np
import tensorflow as tf

from surreal.actors.actor import Actor
from surreal.debug import IfRenderingRenderMaximallyNActors
from surreal.envs.env import Env


class LocalEnv(Env, metaclass=ABCMeta):
    """
    An locally running Env with multiple single-actor copies of some underlying env.
    Uses multiprocessing to use up n available cores for synchronous stepping through all envs at the same time.
    """
    def __init__(self, *, actors=1, num_cores=1, state_space, action_space, reward_space=None, process_class=None,
                 render=False, action_map=None, **process_kwargs):
        """
        Args:
            actors (int): The number of Actors to use.
            state_space (Space): The (single actor) state Space used for all copies of this Env and their Actors.
            action_space (Space): The (single actor) action Space used for all copies of this Env and their Actors.
            reward_space (Space): The (single actor) reward Space used for all copies of this Env and their Actors.

            num_cores (Optional[int]): The max number of CPU cores to use. If set to more than the available CPUs, will
                use all available ones. None for using all available ones. 1 for not utilizing multi-processing.
                Default: 1.

            args (Optional[List[any]]): An optional args list to be passed to `process_setup`.

        Keyword Args:
            process_kwargs: An optional kwargs dict to be passed to the `process_class` c'tor.
        """
        # Actors given as number. Create new Actors.
        self.num_actors = actors
        actors = []
        for i in range(self.num_actors):
            actors.append(Actor(
                name="actor{}".format(i),
                action_space=action_space, state_space=state_space,
                reward_space=reward_space
            ))

        # Call the super's constructor.
        super().__init__(actors=actors, render=render, action_map=action_map)

        # Create the processes and queue.
        max_num_cores = multiprocessing.cpu_count()
        self.num_cores = min(min(num_cores, max_num_cores) if num_cores else max_num_cores, self.num_actors)

        # Build process to actor-slots map.
        self.process_to_actors = {}
        a = 0
        a_left = self.num_actors
        for c in range(self.num_cores):
            num = int(np.ceil(a_left / self.num_cores))
            self.process_to_actors[c] = np.arange(start=a, stop=a + num)
            a_left -= num
            a += num

        self.processes = []
        assert process_class is not None and issubclass(process_class, multiprocessing.Process)
        # Our results Queue (Processes will send back results here).
        self.results_queue = multiprocessing.Queue()
        for i in range(self.num_cores):
            pipe = multiprocessing.Pipe()
            # Generate the Process object.
            process = process_class(
                id_=i, num_actors=len(self.process_to_actors[i]), pipe=pipe, results_queue=self.results_queue,
                **process_kwargs
            )
            # Store Process and its connections.
            self.processes.append((process, pipe))
            # Run the Process.
            process.start()

    def _act(self, actions):
        """
        Executes actions on all our Processes and Actors.

        Args:
            actions (any): The actions to execute. Must be given as a batch (one item per actor).
        """
        # Send act command to all processes.
        for i in range(self.num_cores):
            actors = self.process_to_actors[i]
            self.processes[i][1][1].send(dict(cmd="act", actions=actions[actors]))

        # Wait on our queue for all responses.
        num_responses = 0
        while num_responses < self.num_cores:
            response = self.results_queue.get()
            actors = self.process_to_actors[response["id"]]
            for i, actor in enumerate(actors):
                self.reward[actor], self.terminal[actor] = response["r"][i], response["t"][i]
                self.state[actor] = response["s"][i]
            num_responses += 1

    def reset_all(self):
        """
        Resets all the underlying Actors by sending commands to the respective Processes and waiting for the reset
        states to be returned via our queue.
        """
        # Send act command to all processes.
        for i in range(self.num_cores):
            self.processes[i][1][1].send(dict(cmd="reset_all"))  # [1][1] = pipe.sender

        # Wait on our queue for all responses.
        num_responses = 0
        while num_responses < self.num_cores:
            response = self.results_queue.get()
            actors = self.process_to_actors[response["id"]]
            for i, actor in enumerate(actors):
                self.state[actor] = response["s"][i]
            num_responses += 1

    def terminate(self):
        for i in range(self.num_cores):
            self.processes[i][1][1].send(dict(cmd="terminate"))  # [1][1] = pipe.sender

    def render(self, num_actors=None):
        # TODO: not actor specific: process specific!
        for i in range(num_actors or IfRenderingRenderMaximallyNActors):
            self.processes[i][1][1].send(dict(cmd="render", type="human"))  # [1][1] = pipe.sender


class LocalEnvProcess(multiprocessing.Process):
    """
    The actual underlying Env as a python multiprocessing Process subclass.
    """
    def __init__(self, id_, num_actors, pipe, results_queue):
        """
        Args:
            id_ (int): The ID of this process within the controlling `LocalEnv` object.
            num_actors (int): The number of Actors that live in this Process.
            pipe (Tuple[multiprocessing.Connection]): The connection tuple generated by the `LocalEnv` object.
            results_queue (multiprocessing.Queue): The results queue to send back results to the `LocalEnv` object.
        """
        super().__init__()

        self.id = id_
        self.num_actors = num_actors
        self.pipe = pipe
        self.results_queue = results_queue

    def run(self):
        """
        Overwrite of the Process `run` method. Handles the main loop waiting on incoming queries and sending back
        results through the `results_queue`.
        """
        receiver, sender = self.pipe
        # Loop endlessly until terminated.
        while True:
            msg = receiver.recv()
            # Act command. Send back state, reward, and terminal.
            if msg["cmd"] == "act":
                s, r, t = self._synchronous_act(msg["actions"])
                self.results_queue.put(dict(id=self.id, s=s, r=r, t=t))
            # Reset command. Send back reset-state.
            elif msg["cmd"] == "reset":
                s = self._single_reset(msg["actor_slot"])
                self.results_queue.put(dict(id=self.id, actor_slot=msg["actor_slot"], s=s))
            # Reset-all command. Send back reset-states.
            elif msg["cmd"] == "reset_all":
                s = []
                for actor_slot in range(self.num_actors):
                    s.append(self._single_reset(actor_slot))
                self.results_queue.put(dict(id=self.id, s=s))
            # Rendering command. No results being sent back.
            elif msg["cmd"] == "render":
                self._single_render()
            # Terminal command. Exit process.
            elif msg["cmd"] == "terminate":
                return 0

    @abstractmethod
    def _synchronous_act(self, actions):
        """
        Executes an act command on all our actors and returns the results tuple.

        Args:
            actions (any): The actions (per Actor) to execute.

        Returns:
            Tuple: States, rewards, terminals for each actor (as batched).
        """
        raise NotImplementedError

    @abstractmethod
    def _single_reset(self, actor_slot):
        """
        Executes a reset command on one of our actors and returns the results tuple.

        Args:
            actor_slot (any): The actor slot to reset.

        Returns:
            any: The reset state after resetting the given actor.
        """
        raise NotImplementedError

    def _single_render(self, type=None):
        pass
