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

import numpy as np
import unittest

from surreal.algos.dddqn import DDDQNLoss
from surreal.tests.test_util import check


class TestDDDQNFunctionality(unittest.TestCase):
    """
    Tests the DDDQN algo functionality (loss functions, execution logic, etc.).
    """
    def test_dddqn_loss_function(self):
        # Batch of size=2.
        input_ = {
            "x": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "a": np.array([2, 1]),
            "r": np.array([10.3, -4.25]),
            "t": np.array([False, True]),
            # make s' distinguishable from s via its values for the fake q-net to notice.
            "x_": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
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
        out = DDDQNLoss()(input_, 0.9, q_net, target_q_net, num_categories=3)
        check(out[0].numpy(), expected_loss, decimals=3)
        check(out[1].numpy(), expected_td_errors, decimals=2)
