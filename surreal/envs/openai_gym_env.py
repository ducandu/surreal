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
import random
import tensorflow as tf

from surreal.envs.local_env import LocalEnv, LocalEnvProcess
from surreal.spaces import *
from surreal.utils.errors import SurrealError


class OpenAIGymEnv(LocalEnv):
    """
    OpenAI Gym adapter for RLgraph: https://gym.openai.com/.
    """
    def __init__(
            self, openai_gym_id, *,
            actors=1, num_cores=None, frame_skip=None, max_episode_steps=None, max_num_noops_after_reset=0,
            noop_action=0, episodic_life=False, lost_life_reward=-1.0, fire_after_reset=False, force_float32=False
            #monitor=None, monitor_safe=False, monitor_video=0,
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

            #monitor: Output directory. Setting this to None disables monitoring.
            #monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            #monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.

            force_float32 (bool): Whether to convert all state signals (iff the state space is of dtype float64) into
                float32. Note: This does not affect any int-type state core.
                Default: True.
        """
        # Create a first actual gym-env object.
        self.dummy_env = gym.make(openai_gym_id)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv

        # Derive action- and state Spaces.
        state_space = self.translate_space(self.dummy_env.observation_space, force_float32=force_float32)

        # If state_space is not a Float -> Set force_float32 to False.
        if not isinstance(state_space, Float):
            force_float32 = False

        super().__init__(
            actors=actors, num_cores=num_cores,
            state_space=state_space, action_space=self.translate_space(self.dummy_env.action_space),
            process_class=OpenAIGymEnvProcess,
            # kwargs to be passed to `process_class` c'tor.
            openai_gym_id=openai_gym_id,
            max_episode_steps=max_episode_steps,
            noop_action=noop_action,
            max_num_noops_after_reset=max_num_noops_after_reset,
            episodic_life=episodic_life,
            lost_life_reward=lost_life_reward,
            fire_after_reset=fire_after_reset,
            frame_skip=frame_skip,
            force_float32=force_float32
            # , monitor=monitor, monitor_safe=monitor_safe, monitor_video=monitor_video
        )

        # Buffers for execution returns from processes.
        self.state = np.array([state_space.zeros()] * actors)
        # Reset ourselves.
        self.reset_all()

        #if monitor:
        #    if monitor_video == 0:
        #        video_callable = False
        #    else:
        #        video_callable = (lambda x: x % monitor_video == 0)
        #    for i in range(len(self.gym_envs)):
        #        self.gym_envs[i] = gym.wrappers.Monitor(self.gym_envs[i], monitor, force=not monitor_safe,
        #                                                video_callable=video_callable)

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
        return "OpenAIGym({}x{})".format(self.num_actors, self.dummy_env.env)


class OpenAIGymEnvProcess(LocalEnvProcess):
    def __init__(self, openai_gym_id, *, noop_action, max_num_noops_after_reset, episodic_life,
                 lost_life_reward, max_episode_steps, fire_after_reset, frame_skip, force_float32, **kwargs):
        super().__init__(**kwargs)

        # Create the env.
        self.envs = [gym.make(openai_gym_id) for _ in range(self.num_actors)]

        self.true_terminal = [True] * self.num_actors
        self.lives = [0] * self.num_actors

        self.noop_action = noop_action
        self.max_num_noops_after_reset = max_num_noops_after_reset
        self.episodic_life = episodic_life
        self.lost_life_reward = lost_life_reward

        # Set max-episode cutoff for all our envs.
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps is not None:
            for i in range(self.num_actors):
                self.envs[i]._max_episode_steps = self.max_episode_steps

        self.fire_after_reset = fire_after_reset
        if self.fire_after_reset is True:
            for i in range(self.num_actors):
                assert self.envs[i].unwrapped.get_action_meanings()[1] == 'FIRE'
                assert len(self.envs[i].unwrapped.get_action_meanings()) >= 3

        # Manually set the `frame_skip` property.
        self.frame_skip = frame_skip if isinstance(frame_skip, (tuple, list)) else None if frame_skip is None else \
            (frame_skip, frame_skip + 1)
        #if self.frame_skip is not None:
            # Skip externally and always maximize pixels over 2 consecutive frames.
            #if "NoFrameskip" in self.gym_envs[0].env:
            #    self.frame_skip = frame_skip
            # Set gym property.
            #else:
            #for i in range(self.num_actors):
            #    self.envs[i].env.frameskip = self.frame_skip

        self.force_float32 = force_float32

    def _synchronous_act(self, actions):
        states, rewards, terminals = [], [], []
        for i in range(self.num_actors):
            s, r, t = self.act_and_skip(i, actions[i])
            # Manage lives if necessary.
            if self.episodic_life is True:
                self.true_terminal[i] = t
                lives = self.envs[i].unwrapped.ale.lives()
                # lives < self.lives -> lost a life so show terminal = true to learner.
                if self.lives[i] > lives:
                    t = True
                    r = self.lost_life_reward
                self.lives[i] = lives

            # Episode ended -> reset env (and only update state (as the new next state in the new episode).
            # reward and terminal stay (b/c the algo still needs to see them as terminal=True and r=[some last reward]).
            if t:
                s = self._single_reset(i)

            states.append(s)
            rewards.append(r)
            terminals.append(t)

        return states, rewards, terminals

    def act_and_skip(self, actor_slot, action):
        # Frame skipping is unset or set as env property.
        if self.frame_skip is None:
            s, r, t, _ = self.envs[actor_slot].step(action)
        else:
            # Do frame skip manually.
            # State is the maximum color value over the last two frames to avoid the Atari2600 flickering problem.
            r = 0.0
            s_last = None
            s_pre_last = None
            for i in range(random.randint(self.frame_skip[0], self.frame_skip[1] - 1)):
                s, r_frame, t, _ = self.envs[actor_slot].step(action)
                s_pre_last = s_last if s_last is not None else s
                s_last = s
                r += r_frame
                if t:
                    break

            # Take the max over last two states.
            s = tf.nest.map_structure(lambda s1, s2: np.maximum(s1, s2), s_last, s_pre_last)

        if self.force_float32 is True:
            s = np.array(s, dtype=np.float32)

        return s, np.asarray(r, dtype=np.float32), bool(t)

    def _single_reset(self, actor_slot):
        #gym_env = self.gym_envs[actor_slot]
        # Some Envs need to hit the fire button (action=1) to actually start.
        if self.fire_after_reset is True:
            self.episodic_reset(actor_slot)
            state, _, terminal = self.act_and_skip(actor_slot, 1)
            if terminal:
                raise ValueError
                #self._episodic_reset(self.true_terminal[actor_slot])
            #state, _, terminal = self._act_and_skip(2)
            #if terminal:
            #    self._episodic_reset(self.true_terminal[actor_slot])
            return state
        # Normal reset.
        else:
            return self.episodic_reset(actor_slot)

    def episodic_reset(self, actor_slot):
        reset_state = None
        # Actually reset the underlying gym-env.
        if self.episodic_life is False or self.true_terminal[actor_slot]:
            if isinstance(self.envs[actor_slot], gym.wrappers.Monitor):
                self.envs[actor_slot].stats_recorder.done = True
            reset_state = self.envs[actor_slot].reset()

        # Do we have to wait for n noops after each reset?
        if self.max_num_noops_after_reset > 0:
            num_noops = np.random.randint(low=1, high=self.max_num_noops_after_reset + 1)
            # Do a number of noops to vary starting positions.
            for _ in range(num_noops):
                reset_state, _, terminal, _ = self.envs[actor_slot].step(self.noop_action)
                if terminal:
                    reset_state = self.envs[actor_slot].reset()

        # If the last terminal was just a live lost (not actual end of episode):
        # Do 1 step (with noop) and pretend that next state is the reset-state.
        if self.episodic_life is True and not self.true_terminal[actor_slot]:
            reset_state, _, _ = self.act_and_skip(actor_slot, self.noop_action)

        assert reset_state is not None
        return reset_state if self.force_float32 is False else np.array(reset_state, dtype=np.float32)

    def _single_render(self, type="human"):
        self.envs[0].render(type)
