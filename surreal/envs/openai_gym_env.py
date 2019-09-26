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

import gym
import numpy as np
import tensorflow.nest as nest

from surreal.actors.actor import Actor
from surreal.debug import IfRenderingRenderMaximallyNActors
from surreal.envs import Env
from surreal.spaces import *
from surreal.utils.errors import SurrealError


class OpenAIGymEnv(Env):
    """
    OpenAI Gym adapter for RLgraph: https://gym.openai.com/.
    """
    def __init__(
            self, openai_gym_id, actors=None, frame_skip=None, max_num_noops_after_reset=0,
            noop_action=0, episodic_life=False, lost_life_reward=-1.0,
            fire_after_reset=False, monitor=None, monitor_safe=False, monitor_video=0,
            force_float32=True, **kwargs
    ):
        """
        Args:
            openai_gym_id (str]): OpenAI Gym environment ID. See https://gym.openai.com/envs

            frame_skip (Optional[Tuple[int,int],int]): Number of game frames that should be skipped with each action
                (repeats given action for this number of game frames and accumulates reward).
                Default: (2,5) -> Uniformly pull from set [2,3,4].

            max_num_noops_after_reset (Optional[int]): How many no-ops to maximally perform when resetting
                the environment before returning the reset state. This is internal openAI gym env single-frame count
                (not caring about `frame_skip`).

            noop_action (any): The action representing no-op. 0 for Atari.

            episodic_life (bool): If true, losing a life will lead to episode end from the perspective
                of the agent. Internally, th environment will keep stepping the game and manage the true
                termination (end of game).

            lost_life_reward (float): The reward that should be received if a life is lost. Only if
                `episodic_life` is True.

            fire_after_reset (Optional[bool]): If True, auto-apply action fire (1) after reset to get game started.
                Otherwise, game would wait for this action to come in before reacting to input.

            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.

            force_float32 (bool): Whether to convert all state signals (iff the state space is of dtype float64) into
                float32. Note: This does not affect any int-type state core.
                Default: True.
        """
        # Create a first actual gym-env object.
        self.gym_envs = [gym.make(openai_gym_id)]  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv

        # Derive action- and state Spaces.
        action_space = self.translate_space(self.gym_envs[0].action_space)
        state_space = self.translate_space(self.gym_envs[0].observation_space, force_float32=force_float32)
        # Create Actors.
        if isinstance(actors, int):
            num_actors = actors
            actors = []
            for i in range(num_actors):
                actors.append(Actor("actor{}".format(i), action_space=action_space, state_space=state_space))
        super(OpenAIGymEnv, self).__init__(actors=actors, **kwargs)

        # In Atari envs, 0 is no-op.
        self.noop_action = noop_action
        self.max_num_noops_after_reset = max_num_noops_after_reset

        # Manage life as episodes.
        self.episodic_life = episodic_life
        self.lost_life_reward = lost_life_reward
        self.true_terminal = [True] * len(actors)
        self.lives = [0] * len(actors)

        self.fire_after_reset = fire_after_reset
        self.force_float32 = False  # Set to False for now, later overwrite with a correct value.

        # Add other gym-envs to our list.
        for _ in range(len(actors) - 1):
            self.gym_envs.append(gym.make(openai_gym_id))

        # Buffers for execution returns from gym-envs.
        self.state = np.array([state_space.zeros()] * len(actors))
        self.info = [None] * len(actors)

        # Manually set the `frame_skip` property.
        self.frame_skip = None
        if frame_skip is not None:
            # Skip externally and always maximize pixels over 2 consecutive frames.
            #if "NoFrameskip" in self.gym_envs[0].env:
            #    self.frame_skip = frame_skip
            # Set gym property.
            #else:
            for env in self.gym_envs:
                env.env.frameskip = frame_skip

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            for i in range(len(self.gym_envs)):
                self.gym_envs[i] = gym.wrappers.Monitor(self.gym_envs[i], monitor, force=not monitor_safe,
                                                        video_callable=video_callable)

        if self.fire_after_reset is True:
            assert all(env.unwrapped.get_action_meanings()[1] == 'FIRE' for env in self.gym_envs)
            assert all(len(env.unwrapped.get_action_meanings()) >= 3 for env in self.gym_envs)

        # If state_space is not a Float -> Set force_float32 to False.
        if not isinstance(state_space, Float):
            force_float32 = False
        self.force_float32 = force_float32

        for i in range(len(self.gym_envs)):
            self.state[i] = self._reset(i)

    def _act(self, actions):
        for i, gym_env in enumerate(self.gym_envs):
            self.state[i], self.reward[i], self.terminal[i], _ = self._act_and_skip(self.gym_envs[i], actions[i])

            # Manage lives if necessary.
            if self.episodic_life is True:
                self.true_terminal[i] = self.terminal[i]
                lives = self.gym_envs[i].unwrapped.ale.lives()
                # lives < self.lives -> lost a life so show terminal = true to learner.
                if self.lives[i] > lives > 0:
                    self.terminal[i] = True
                    self.reward[i] = self.lost_life_reward
                self.lives[i] = lives

            # Episode truly ended -> reset env (and only update state (as the new next state in the new episode).
            # reward and terminal stay (b/c the algo still needs to see them as terminal=True and r=[some last reward]).
            if self.terminal[i]:
                self.state[i] = self._reset(i)

    def _act_and_skip(self, gym_env, actions):
        # TODO - allow for goal reward substitution for multi-goal envs
        # Frame skipping is unset or set as env property.
        if self.frame_skip is None:
            s, r, t, info = gym_env.step(actions)
        else:
            # Do frame skip loop in our wrapper class.
            r = 0.0
            t = None
            info = None
            # State is the maximum color value over the last two frames to avoid the Atari2600 flickering problem.
            s_last = None
            s_pre_last = None
            for i in range(self.frame_skip):
                state, reward, terminal, info = gym_env.step(actions)
                if i == self.frame_skip - 2:
                    s_pre_last = state
                if i == self.frame_skip - 1:
                    s_last = state
                r += reward
                if terminal:
                    break

            # Take the max over last two states.
            s = nest.map_structure(lambda s1, s2: np.maximum(s1, s2), s_last, s_pre_last)

        if self.force_float32 is True:
            s = np.array(s, dtype=np.float32)

        return s, np.asarray(r, dtype=np.float32), bool(t), info

    def _reset(self, actor_slot):
        gym_env = self.gym_envs[actor_slot]
        # Some Envs need to hit the fire button (action=1) to actually start.
        if self.fire_after_reset is True:
            self._episodic_reset(gym_env, actor_slot)
            state, _, terminal, _ = self._act_and_skip(gym_env, 1)
            if terminal:
                raise ValueError
                #self._episodic_reset(gym_env, actor_slot)
            #state, _, terminal, _ = self._act_and_skip(gym_env, 2)
            #if terminal:
            #    self._episodic_reset(gym_env, actor_slot)
            return state
        # Normal reset.
        else:
            return self._episodic_reset(gym_env, actor_slot)

    def _episodic_reset(self, gym_env, gym_env_slot):
        reset_state = None
        # Actually reset the underlying gym-env.
        if self.episodic_life is False or self.true_terminal[gym_env_slot]:
            if isinstance(gym_env, gym.wrappers.Monitor):
                gym_env.stats_recorder.done = True
            reset_state = gym_env.reset()

        # Do we have to wait for n noops after each reset?
        if self.max_num_noops_after_reset > 0:
            num_noops = np.random.randint(low=1, high=self.max_num_noops_after_reset + 1)
            # Do a number of noops to vary starting positions.
            for _ in range(num_noops):
                reset_state, _, terminal, _ = gym_env.step(self.noop_action)
                if terminal:
                    reset_state = gym_env.reset()

        # If the last terminal was just a live lost (not actual end of episode):
        # Do 1 step (with noop) and pretend that next state is the reset-state.
        if self.episodic_life is True and not self.true_terminal[gym_env_slot]:
            reset_state, _, _, _ = self._act_and_skip(gym_env, self.noop_action)

        assert reset_state is not None
        return reset_state if self.force_float32 is False else np.array(reset_state, dtype=np.float32)

    def terminate(self):
        for gym_env in self.gym_envs:
            gym_env.close()
        self.gym_envs = None

    def render(self, num_actors=None):
        for i in range(num_actors or IfRenderingRenderMaximallyNActors):
            self.gym_envs[i].render("human")

    @staticmethod
    def translate_space(space, force_float32=False):
        """
        Translates openAI core into RLGraph Space classes.

        Args:
            space (gym.core.Space): The openAI Space to be translated.

        Returns:
            Space: The translated rlgraph Space.
        """
        if isinstance(space, gym.spaces.Discrete):
            return Int(space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return Bool(shape=(space.n,))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return Int(low=np.zeros((space.nvec.ndim,), dtype=np.uint8), high=space.nvec)
        elif isinstance(space, gym.spaces.Box):
            # Decide by dtype:
            box_dtype = str(space.low.dtype)
            if "int" in box_dtype:
                return Int(low=space.low, high=space.high, dtype=box_dtype)
            elif "float" in box_dtype:
                return Float(
                    low=space.low, high=space.high, dtype="float32" if force_float32 is True else box_dtype
                )
            elif "bool" in box_dtype:
                return Bool(shape=space.shape)
        elif isinstance(space, gym.spaces.Tuple):
            return Tuple(*[OpenAIGymEnv.translate_space(s) for s in space.spaces])
        elif isinstance(space, gym.spaces.Dict):
            return Dict({key: OpenAIGymEnv.translate_space(value, force_float32=force_float32)
                         for key, value in space.spaces.items()})

        raise SurrealError("Unknown openAI gym Space class ({}) for state_space!".format(space))

    def __str__(self):
        return "OpenAIGym({}x{})".format(len(self.actors), self.gym_envs[0].env)
