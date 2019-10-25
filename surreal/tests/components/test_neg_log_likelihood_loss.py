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
import scipy.stats as sts
import unittest

from surreal.components.loss_functions.neg_log_likelihood_loss import NegLogLikelihoodLoss
from surreal.spaces import *
from surreal.spaces.space_utils import get_default_distribution_from_space
from surreal.tests.test_util import check
from surreal.utils.numpy import softmax


class TestSupervisedLossFunctions(unittest.TestCase):

    def test_neg_log_likelihood_loss_function_w_simple_space(self):
        shape = (5, 4, 3)
        parameters_space = Tuple(Float(shape=shape), Float(shape=shape), main_axes="B")
        labels_space = Float(shape=shape, main_axes="B")

        loss_function = NegLogLikelihoodLoss(distribution=get_default_distribution_from_space(labels_space))

        parameters = parameters_space.sample(10)
        # Make sure stddev params are not too crazy (just like our adapters do clipping for the raw NN output).
        parameters = (parameters[0], np.clip(parameters[1], 0.1, 1.0))
        labels = labels_space.sample(10)

        expected_loss_per_item = np.sum(-np.log(sts.norm.pdf(labels, parameters[0], parameters[1])), axis=(-1, -2, -3))

        out = loss_function(parameters, labels)
        check(out, expected_loss_per_item, decimals=4)

    def test_neg_log_likelihood_loss_function_w_container_space(self):
        parameters_space = Dict({
            # Make sure stddev params are not too crazy (just like our adapters do clipping for the raw NN output).
            "a": Tuple(Float(shape=(2, 3)), Float(0.5, 1.0, shape=(2, 3))),  # normal (0.0 to 1.0)
            "b": Float(shape=(4,), low=-1.0, high=1.0)  # 4-discrete
        }, main_axes="B")

        labels_space = Dict({
            "a": Float(shape=(2, 3)),
            "b": Int(4)
        }, main_axes="B")

        loss_function = NegLogLikelihoodLoss(distribution=get_default_distribution_from_space(labels_space))

        parameters = parameters_space.sample(2)
        # Softmax the discrete params.
        probs_b = softmax(parameters["b"])
        # probs_b = parameters["b"]
        labels = labels_space.sample(2)

        # Expected loss: Sum of all -log(llh)
        log_prob_per_item_a = np.sum(np.log(sts.norm.pdf(labels["a"], parameters["a"][0], parameters["a"][1])),
                                     axis=(-1, -2))
        log_prob_per_item_b = np.array([np.log(probs_b[0][labels["b"][0]]), np.log(probs_b[1][labels["b"][1]])])

        expected_loss_per_item = -(log_prob_per_item_a + log_prob_per_item_b)

        out = loss_function(parameters, labels)
        check(out, expected_loss_per_item, decimals=4)
