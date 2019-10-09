# Copyright 2019 ducandu GmbH, All Rights Reserved
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

from collections import namedtuple
import numpy as np
import tensorflow as tf
import unittest

from surreal.algos.dddqn import DDDQN, DDDQNConfig, DDDQNLoss
from surreal.components.preprocessors import Preprocessor
from surreal.envs import GridWorld
from surreal.tests.test_util import check


class TestDDDQNFunctionality(unittest.TestCase):
    """
    Tests the DDDQN algo functionality (loss functions, execution logic, etc.).
    """
    def test_dddqn_loss_function(self):
        """
        Tests the dueling/double q-loss function assuming an n-step of 1.
        """
        # Batch of size=2.
        input_ = {
            "x": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "a": np.array([2, 1]),
            "r": np.array([10.3, -4.25]),
            "t": np.array([False, True]),
            # make s' distinguishable from s via its values for the fake q-net to notice.
            "x_": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            "num_steps": np.array([1, 1])
        }

        # Fake q-nets. Just have to be callables.
        # The q-net is first called by the loss function with s', then with s. Detect this difference here and return
        # different q-values for the different states.
        q_net = lambda s: dict(A=np.array([[-12.3, 1.2, 1.4], [12.2, -11.5, 9.2]]) if s[0][0] == 1.0 else \
            np.array([[10.0, -10.0, 12.4], [-0.101, -4.6, -9.3]]), V=np.array([1.0, -2.0]))
        target_q_net = lambda s_: dict(A=np.array([[-10.3, 1.5, 1.4], [8.2, -10.9, 9.3]]), V=np.array([0.1, -0.2]))

        """
        Calculation:
        batch of 2, gamma=0.9
        a' = [2 0]  <- argmax(a')Q(s'a')

        Qt(s'.) = 0.1+[-10.3, 1.5, 1.4]--2.4666(A-avg) -0.2+[8.2, -10.9, 9.3]-2.2(A-avg) -> Qt(s'a') = \ 
            [0.1+1.4+2.4666=3.9666] [0.0 <- terminal=True]

        a = [2 1]
        Q(s,a)  = 1.0+[12.4]-4.1333(A-avg) -2.0+[-4.6]--4.667(A-avg) = [9.2667 -1.933]

        L = E(batch)| 0.5((r + gamma Qt(s'( argmax(a') Q(s'a') )) ) - Q(s,a))^2 |
        L = (0.5(10.3 + 0.9*3.9666 - 9.2667)^2 + 0.5(-4.25 + 0.9*0.0 - -1.933)^2) / 2
        L = (0.5(4.60324)^2 + 0.5(-2.317)^2) / 2  <- td-errors are the numbers inside the (...)^2 brackets
        L = (21.1898184976 + 5.368489) / 4
        L = 26.5583074976 / 4 
        L = 6.6395768744
        """

        # Expect the mean over the batch.
        expected_loss = 6.6395768744
        expected_td_errors = [4.60333333, 2.317]  # absolute values
        out = DDDQNLoss()(input_, q_net, target_q_net, namedtuple("FakeDDDQNConfig", ["gamma"])(gamma=0.9))
        check(out[0].numpy(), expected_loss, decimals=3)
        check(out[1].numpy(), expected_td_errors, decimals=2)

    def test_dddqn_n_step_memory_insertion(self):
        """
        Tests the n-step post-processing and memory-insertions of DDQN.
        """
        # Create an Env object.
        env = GridWorld("2x2", actors=1)
        # Create a very standard DDQN.
        dqn_config = DDDQNConfig.make(
            "../configs/dddqn_grid_world_2x2_learning.json",
            gamma=0.5,  # fix gamma for unique-memory-checks purposes
            epsilon=[1.0, 0.5],  # fix epsilon to get lots of random actions.
            preprocessor=Preprocessor(
                lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
            ),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )
        algo = DDDQN(config=dqn_config, name="my-dddqn")
        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run for n ticks, then check memory contents for correct n-step tuples.
        for _ in range(5):
            env.run(ticks=100, sync=True, render=False)
            self._check_2x2_grid_world_mem(algo.memory.memory)

        env.terminate()

    def test_dddqn_n_step_memory_insertion_n_step_samples_only(self):
        """
        Tests the n-step post-processing and memory-insertions of DDQN (with the n_step_only option set to True).
        """
        # Create an Env object.
        env = GridWorld("2x2", actors=1)
        # Create a very standard DDQN.
        dqn_config = DDDQNConfig.make(
            "../configs/dddqn_grid_world_2x2_learning.json",
            n_step_only=True,
            gamma=0.5,  # fix gamma for unique-memory-checks purposes
            epsilon=[1.0, 0.5],  # fix epsilon to get lots of random actions.
            preprocessor=Preprocessor(
                lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
            ),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )
        algo = DDDQN(config=dqn_config, name="my-dddqn")
        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run for n ticks, then check memory contents for correct n-step tuples.
        for _ in range(5):
            env.run(ticks=100, sync=True, render=False)
            self._check_2x2_grid_world_mem(algo.memory.memory, n_step_only=True)

        env.terminate()

    def _check_2x2_grid_world_mem(self, memory, n_step_only=False):
        for m in memory:
            # s=0 (start state).
            if m[4][0] == 1.0:
                # action=up
                if m[0] == 0:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][0] == 1.0)  # s'=0
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=-2.6 -> up, then right (death)
                        elif np.allclose(m[2], -2.6):
                            self.assertTrue(m[3])  # t=True
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                # action=right -> assert death
                elif m[0] == 1:
                    self.assertTrue(m[2] == -5.0)  # r
                    self.assertTrue(m[3])  # t=True
                    self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                    self.assertTrue(m[1] == 1)  # 1-step tuple
                # action=down
                elif m[0] == 2:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][1] == 1.0)  # s'=1
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=0.4 -> down, then into goal
                        elif np.allclose(m[2], 0.4):
                            self.assertTrue(m[3])
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                # action=left
                elif m[0] == 3:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][0] == 1.0)  # s'=0
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=-2.6 -> left, then into death
                        elif np.allclose(m[2], -2.6):
                            self.assertTrue(m[3])
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                else:
                    raise ValueError
            # s=1
            elif m[4][1] == 1.0:
                # action=up
                if m[0] == 0:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][0] == 1.0)  # s'=0
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=-2.6 -> up, then into death
                        elif np.allclose(m[2], -2.6):
                            self.assertTrue(m[3])
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                # action=right -> assert goal
                elif m[0] == 1:
                    self.assertTrue(m[2] == 1.0)  # r
                    self.assertTrue(m[3])  # t=True
                    self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                    self.assertTrue(m[1] == 1)  # 1-step tuple
                # action=down
                elif m[0] == 2:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][1] == 1.0)  # s'=1
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=0.4 -> down, then into goal
                        elif np.allclose(m[2], 0.4):
                            self.assertTrue(m[3])
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                # action=left
                elif m[0] == 3:
                    # 1-step
                    if m[1] == 1:
                        if n_step_only:
                            raise ValueError
                        self.assertTrue(m[2] == -0.1)  # r
                        self.assertFalse(m[3])  # t=False
                        self.assertTrue(m[5][1] == 1.0)  # s'=1
                    else:
                        self.assertTrue(m[1] == 2)  # num_steps
                        # r=-0.1-0.5*0.1 = -0.15 -> check for "stultus currit" (walking around with -0.1 rewards)
                        if np.allclose(m[2], -0.15):  # 2*-0.1 discuonted
                            self.assertFalse(m[3])  # we can only assert that episode is still ongoing
                        # r=0.4 -> left, then into goal
                        elif np.allclose(m[2], 0.4):
                            self.assertTrue(m[3])
                            self.assertTrue(m[5][0] == 1.0)  # s'=0 (reset-state)
                        else:
                            raise ValueError
                else:
                    raise ValueError
            else:
                raise ValueError
