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
from scipy.stats import norm, beta, multivariate_normal
import scipy
import unittest

from surreal.components.distributions import *
from surreal.spaces import Dict, Float, Int, Tuple
from surreal.tests import check
from surreal.utils.numpy import softmax, sigmoid
from surreal.utils.util import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT


class TestDistributions(unittest.TestCase):
    """
    Tests our various distribution classes passing them parameterization input that would normally come from
    their respective adapters.
    """
    def test_bernoulli(self):
        # Create 5 bernoulli distributions (or a multiple thereof if we use batch-size > 1).
        param_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")

        # The Component to test.
        bernoulli = Bernoulli()
        # Batch of size=6 and deterministic (True).
        input_ = param_space.sample(6)
        expected = sigmoid(input_) > 0.5
        # Sample n times, expect always max value (max likelihood for deterministic draw).
        for _ in range(10):
            out = bernoulli.sample(input_, deterministic=True)
            check(out, expected)
            out = bernoulli.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=6 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(6)
        outs = []
        for _ in range(100):
            out = bernoulli.sample(input_, deterministic=False)
            outs.append(out)
            out = bernoulli.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs), 0.5, decimals=1)

        logits = np.array([[0.1, -0.2, 0.3, -4.4, 2.0]])
        probs = sigmoid(logits)

        # Test log-likelihood outputs.
        values = np.array([[True, False, False, True, True]])
        out = bernoulli.log_prob(logits, values=values)
        expected_log_probs = np.log(np.where(values, probs, 1.0 - probs))
        check(out, expected_log_probs)

        # Test entropy outputs.
        # Binary Entropy with natural log.
        expected_entropy = -(probs * np.log(probs)) - ((1.0 - probs) * np.log(1.0 - probs))
        out = bernoulli.entropy(logits)
        check(out, expected_entropy)

    def test_categorical(self):
        # Create 5 categorical distributions of 3 categories each.
        param_space = Float(shape=(5, 3), low=-1.0, high=2.0, main_axes="B")
        values_space = Int(3, shape=(5,), main_axes="B")

        # The Component to test.
        categorical = Categorical()

        # Batch of size=3 and deterministic (True).
        input_ = param_space.sample(3)
        expected = np.argmax(input_, axis=-1)
        # Sample n times, expect always max value (max likelihood for deterministic draw).
        for _ in range(10):
            out = categorical.sample(input_, deterministic=True)
            check(out, expected)
            out = categorical.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=3 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(3)
        outs = []
        for _ in range(100):
            out = categorical.sample(input_, deterministic=False)
            outs.append(out)
            out = categorical.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs), 1.0, decimals=0)

        input_ = param_space.sample(1)
        probs = softmax(input_)
        values = values_space.sample(1)

        # Test log-likelihood outputs.
        out = categorical.log_prob(input_, values)
        check(out, np.log(np.array([[
            probs[0][0][values[0][0]], probs[0][1][values[0][1]], probs[0][2][values[0][2]],
            probs[0][3][values[0][3]], probs[0][4][values[0][4]]
        ]])), decimals=4)

        # Test entropy outputs.
        out = categorical.entropy(input_)
        expected_entropy = - np.sum(probs * np.log(probs), axis=-1)
        check(out, expected_entropy)

    def test_normal(self):
        # Create 5 normal distributions (2 parameters (mean and stddev) each).
        param_space = Tuple(
            Float(shape=(5,)),  # mean
            Float(0.5, 1.0, shape=(5,)),  # stddev
            main_axes="B"
        )
        values_space = Float(shape=(5,), main_axes="B")

        # The Component to test.
        normal = Normal()

        # Batch of size=2 and deterministic (True).
        input_ = param_space.sample(2)
        expected = input_[0]  # 0 = mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            out = normal.sample(input_, deterministic=True)
            check(out, expected)
            normal.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        expected = input_[0][0]  # 0 = mean
        outs = []
        for _ in range(100):
            out = normal.sample(input_, deterministic=False)
            outs.append(out)
            out = normal.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs), expected.mean(), decimals=1)

        means = np.array([[0.1, 0.2, 0.3, 0.4, 50.0]])
        log_stds = np.array([[0.8, -0.2, 0.3, -1.0, 10.0]])
        # The normal-adapter does this following line with the NN output (interpreted as log(stddev)):
        # Doesn't really matter here in this test case, though.
        stds = np.exp(np.clip(log_stds, a_min=MIN_LOG_NN_OUTPUT, a_max=MAX_LOG_NN_OUTPUT))
        values = np.array([[1.0, 2.0, 0.4, 10.0, 5.4]])

        # Test log-likelihood outputs.
        out = normal.log_prob((means, stds), values)
        expected_outputs = np.log(norm.pdf(values, means, stds))
        check(out, expected_outputs)

        # Test entropy outputs.
        out = normal.entropy((means, stds))
        # See: https://en.wikipedia.org/wiki/Normal_distribution#Maximum_entropy
        expected_entropy = 0.5 * (1 + np.log(2 * np.square(stds) * np.pi))
        check(out, expected_entropy)

    def test_multivariate_normal(self):
        # Create batch0=n (batch-rank), batch1=2 (can be used for m mixed Gaussians), num-events=3 (trivariate)
        # distributions (2 parameters (mean and stddev) each).
        num_events = 3  # 3=trivariate Gaussian
        num_mixed_gaussians = 2  # 2x trivariate Gaussians (mixed)
        param_space = Tuple(
            Float(shape=(num_mixed_gaussians, num_events)),  # mean
            Float(0.5, 1.0, shape=(num_mixed_gaussians, num_events)),  # diag (variance)
            main_axes="B"
        )
        values_space = Float(shape=(num_mixed_gaussians, num_events), main_axes="B")

        # The Component to test.
        distribution = MultivariateNormal()

        input_ = param_space.sample(4)
        expected = input_[0]  # 0=mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            out = distribution.sample(input_, deterministic=True)
            check(out, expected)
            out = distribution.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        expected = input_[0]  # 0=mean
        outs = []
        for _ in range(100):
            out = distribution.sample(input_, deterministic=False)
            outs.append(out)
            out = distribution.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs), expected.mean(), decimals=1)

        means = values_space.sample(2)
        stds = values_space.sample(2)
        values = values_space.sample(2)

        # Test log-likelihood outputs (against scipy).
        out = distribution.log_prob((means, stds), values)
        # Sum up the individual log-probs as we have a diag (independent) covariance matrix.
        check(out, np.sum(np.log(norm.pdf(values, means, stds)), axis=-1), decimals=4)

        # TODO: entropy and KL-Divergence test cases.

    def test_beta(self):
        # Create 5 beta distributions (2 parameters (alpha and beta) each).
        param_space = Tuple(
            Float(shape=(5,)),  # alpha
            Float(shape=(5,)),  # beta
            main_axes="B"
        )
        values_space = Float(shape=(5,), main_axes="B")

        # The Component to test.
        low, high = -1.0, 2.0
        beta_distribution = Beta(low=low, high=high)

        # Batch of size=2 and deterministic (True).
        input_ = param_space.sample(2)
        # Mean for a Beta distribution: 1 / [1 + (beta/alpha)]
        expected = (1.0 / (1.0 + input_[1] / input_[0])) * (high - low) + low
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(100):
            out = beta_distribution.sample(input_, deterministic=True)
            check(out, expected)
            out = beta_distribution.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        expected = (1.0 / (1.0 + input_[1] / input_[0])) * (high - low) + low
        outs = []
        for _ in range(100):
            out = beta_distribution.sample(input_, deterministic=False)
            outs.append(out)
            out = beta_distribution.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs), expected.mean(), decimals=1)

        alpha_ = values_space.sample(1)
        beta_ = values_space.sample(1)
        values = values_space.sample(1)
        values_scaled = values * (high - low) + low

        # Test log-likelihood outputs (against scipy).
        out = beta_distribution.log_prob((alpha_, beta_), values_scaled)
        check(out, np.log(beta.pdf(values, alpha_, beta_)), decimals=4)

        # TODO: Test entropy outputs (against scipy).
        out = beta_distribution.entropy((alpha_, beta_))
        # TODO: This is tricky and does not seem to match sometimes for all input-slots.
        #check(out, beta.entropy(alpha_, beta_), decimals=2)

    def test_mixture(self):
        # Create a mixture distribution consisting of 3 bivariate normals weighted by an internal
        # categorical distribution.
        num_distributions = 3
        num_events_per_multivariate = 2  # 2=bivariate
        param_space = Dict(
            {
                "categorical": Float(shape=(num_distributions,), low=-1.5, high=2.3),
                "parameters0": Tuple(
                    Float(shape=(num_events_per_multivariate,)),  # mean
                    Float(shape=(num_events_per_multivariate,), low=0.5, high=1.0),  # diag
                ),
                "parameters1": Tuple(
                    Float(shape=(num_events_per_multivariate,)),  # mean
                    Float(shape=(num_events_per_multivariate,), low=0.5, high=1.0),  # diag
                ),
                "parameters2": Tuple(
                    Float(shape=(num_events_per_multivariate,)),  # mean
                    Float(shape=(num_events_per_multivariate,), low=0.5, high=1.0),  # diag
                ),
            },
            main_axes="B"
        )
        values_space = Float(shape=(num_events_per_multivariate,), main_axes="B")
        # The Component to test.
        mixture = MixtureDistribution(
            # Try different spec types.
            MultivariateNormal(), "multi-variate-normal", "multivariate_normal"
        )

        # Batch of size=n and deterministic (True).
        input_ = param_space.sample(1)
        # Make probs for categorical.
        categorical_probs = softmax(input_["categorical"])

        # Note: Usually, the deterministic draw should return the max-likelihood value
        # Max-likelihood for a 3-Mixed Bivariate: mean-of-argmax(categorical)()
        # argmax = np.argmax(input_[0]["categorical"], axis=-1)
        #expected = np.array([input_[0]["parameters{}".format(idx)][0][i] for i, idx in enumerate(argmax)])
        #    input_[0]["categorical"][:, 1:2] * input_[0]["parameters1"][0] + \
        #    input_[0]["categorical"][:, 2:3] * input_[0]["parameters2"][0]

        # The mean value is a 2D vector (bivariate distribution).
        expected = categorical_probs[:, 0:1] * input_["parameters0"][0] + \
            categorical_probs[:, 1:2] * input_["parameters1"][0] + \
            categorical_probs[:, 2:3] * input_["parameters2"][0]

        for _ in range(20):
            out = mixture.sample(input_, deterministic=True)
            check(out, expected)
            out = mixture.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        # Make probs for categorical.
        categorical_probs = softmax(input_["categorical"])
        expected = categorical_probs[:, 0:1] * input_["parameters0"][0] + \
            categorical_probs[:, 1:2] * input_["parameters1"][0] + \
            categorical_probs[:, 2:3] * input_["parameters2"][0]
        outs = []
        for _ in range(500):
            out = mixture.sample(input_, deterministic=False)
            outs.append(out)
            out = mixture.sample_stochastic(input_)
            outs.append(out)
        check(np.mean(np.array(outs), axis=0), expected, decimals=1)

        return
        # TODO: prob/log-prob tests for Mixture.

        # Test log-likelihood outputs (against scipy).
        for i in range(20):
            params = param_space.sample(1)
            # Make sure categorical params are softmaxed.
            category_probs = softmax(params["categorical"][0])
            values = values_space.sample(1)
            expected = 0.0
            v = []
            for j in range(3):
                v.append(multivariate_normal.pdf(
                    values[0], mean=params["parameters{}".format(j)][0][0], cov=params["parameters{}".format(j)][1][0]
                ))
                expected += category_probs[j] * v[-1]
            out = mixture.prob(params, values)
            check(out[0], expected, atol=0.1)

            expected = np.zeros(shape=(3,))
            for j in range(3):
                expected[j] = np.log(category_probs[j]) + np.log(multivariate_normal.pdf(
                    values[0], mean=params["parameters{}".format(j)][0][0], cov=params["parameters{}".format(j)][1][0]
                ))
            expected = np.log(np.sum(np.exp(expected)))
            out = mixture.log_prob(params, values)
            print("{}: out={} expected={}".format(i, out, expected))
            check(out, np.array([expected]), atol=0.25)

    def test_squashed_normal(self):
        param_space = Tuple(Float(-1.0, 1.0, shape=(5,)), Float(0.5, 1.0, shape=(5,)), main_axes="B")

        low, high = -2.0, 1.0
        squashed_distribution = SquashedNormal(low=low, high=high)

        # Batch of size=2 and deterministic (True).
        input_ = param_space.sample(2)
        expected = ((np.tanh(input_[0]) + 1.0) / 2.0) * (high - low) + low   # [0] = mean
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(50):
            out = squashed_distribution.sample(input_, deterministic=True)
            check(out, expected)
            out = squashed_distribution.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        expected = ((np.tanh(input_[0]) + 1.0) / 2.0) * (high - low) + low  # [0] = mean
        outs = []
        for _ in range(500):
            out = squashed_distribution.sample(input_, deterministic=False)
            outs.append(out)
            self.assertTrue(np.max(out) <= high)
            self.assertTrue(np.min(out) >= low)
            out = squashed_distribution.sample_stochastic(input_)
            outs.append(out)
            self.assertTrue(np.max(out) <= high)
            self.assertTrue(np.min(out) >= low)

        check(np.mean(outs), expected.mean(), decimals=1)

        means = np.array([[0.1, 0.2, 0.3, 0.4, 50.0], [-0.1, -0.2, -0.3, -0.4, -1.0]])
        log_stds = np.array([[0.8, -0.2, 0.3, -1.0, 10.0], [0.7, -0.3, 0.4, -0.9, 8.0]])
        # The normal-adapter does this following line with the NN output (interpreted as log(stddev)):
        # Doesn't really matter here in this test case, though.
        stds = np.exp(np.clip(log_stds, a_min=MIN_LOG_NN_OUTPUT, a_max=MAX_LOG_NN_OUTPUT))
        # Make sure values are within low and high.
        values = np.array([[0.9, 0.2, 0.4, -0.1, -1.05], [-0.9, -0.2, 0.4, -0.1, -1.05]])

        # Test log-likelihood outputs.
        # TODO: understand and comment the following formula to get the log-prob.
        # Unsquash values, then get log-llh from regular gaussian.
        unsquashed_values = np.arctanh((values - low) / (high - low) * 2.0 - 1.0)
        log_prob_unsquashed = np.log(norm.pdf(unsquashed_values, means, stds))
        log_prob = log_prob_unsquashed - np.sum(np.log(1 - np.tanh(unsquashed_values) ** 2), axis=-1, keepdims=True)

        out = squashed_distribution.log_prob((means, stds), values)
        check(out, log_prob)

        # Test entropy outputs.
        # TODO
        return
        #out = squashed_distribution.entropy((means, stds))
        ## See: https://en.wikipedia.org/wiki/Normal_distribution#Maximum_entropy
        #expected_entropy = 0.5 * (1 + np.log(2 * np.square(stds) * np.pi))
        #check(out, expected_entropy)

    def test_gumbel_softmax_distribution(self):
        # 5-categorical Gumble-Softmax.
        param_space = Float(shape=(5,), main_axes="B")
        values_space = Float(shape=(5,), main_axes="B")

        gumble_softmax_distribution = GumbelSoftmax(temperature=1.0)

        # Batch of size=2 and deterministic (True).
        input_ = param_space.sample(2)
        expected = softmax(input_)
        # Sample n times, expect always argmax value (deterministic draw).
        for _ in range(50):
            out = gumble_softmax_distribution.sample(input_, deterministic=True)
            check(out, expected)
            out = gumble_softmax_distribution.sample_deterministic(input_)
            check(out, expected)

        # Batch of size=1 and non-deterministic -> expect roughly the vector of probs.
        input_ = param_space.sample(1)
        expected = softmax(input_)
        outs = []
        for _ in range(100):
            out = gumble_softmax_distribution.sample(input_)
            outs.append(out)
            out = gumble_softmax_distribution.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(outs, axis=0), expected, decimals=1)

        return  # TODO: Figure out Gumbel Softmax log-prob calculation (our current implementation does not correspond with paper's formula).

        def gumbel_log_density(y, probs, num_categories, temperature=1.0):
            # https://arxiv.org/pdf/1611.01144.pdf.
            density = np.math.factorial(num_categories - 1) * np.math.pow(temperature, num_categories - 1) * \
                (np.sum(probs / np.power(y, temperature), axis=-1) ** -num_categories) * \
                np.prod(probs / np.power(y, temperature + 1.0), axis=-1)
            return np.log(density)

        # Test log-likelihood outputs.
        input_ = param_space.sample(3)
        values = values_space.sample(3)
        expected = gumbel_log_density(values, softmax(input_), num_categories=param_space.shape[0])

        out = gumble_softmax_distribution.log_prob(input_, values)
        check(out, expected)

    def test_joint_cumulative_distribution(self):
        param_space = Dict({
            "a": Float(shape=(4,)),  # 4-discrete
            "b": Dict({"ba": Tuple([Float(shape=(3,)), Float(0.1, 1.0, shape=(3,))]),  # 3-variate normal
                       "bb": Tuple([Float(shape=(2,)), Float(shape=(2,))]),  # beta -1 to 1
                       "bc": Tuple([Float(shape=(4,)), Float(0.1, 1.0, shape=(4,))]),  # normal (dim=4)
                       })
        }, main_axes="B")

        values_space = Dict({
            "a": Int(4),
            "b": Dict({
                "ba": Float(shape=(3,)),
                "bb": Float(shape=(2,)),
                "bc": Float(shape=(4,))
            })
        }, main_axes="B")

        low, high = -1.0, 1.0
        cumulative_distribution = JointCumulativeDistribution(distributions={
            "a": Categorical(), "b": {"ba": MultivariateNormal(), "bb": Beta(low=low, high=high), "bc": Normal()}
        })

        # Batch of size=2 and deterministic (True).
        input_ = param_space.sample(2)
        input_["a"] = softmax(input_["a"])
        expected_mean = {
            "a": np.argmax(input_["a"], axis=-1),
            "b": {
                "ba": input_["b"]["ba"][0],  # [0]=Mean
                # Mean for a Beta distribution: 1 / [1 + (beta/alpha)] * range + low
                "bb": (1.0 / (1.0 + input_["b"]["bb"][1] / input_["b"]["bb"][0])) * (high - low) + low,
                "bc": input_["b"]["bc"][0],
            }
        }
        # Sample n times, expect always mean value (deterministic draw).
        for _ in range(20):
            out = cumulative_distribution.sample(input_, deterministic=True)
            check(out, expected_mean)
            out = cumulative_distribution.sample_deterministic(input_)
            check(out, expected_mean)

        # Batch of size=1 and non-deterministic -> expect roughly the mean.
        input_ = param_space.sample(1)
        input_["a"] = softmax(input_["a"])
        expected_mean = {
            "a": np.sum(input_["a"] * np.array([0, 1, 2, 3])),
            "b": {
                "ba": input_["b"]["ba"][0],  # [0]=Mean
                # Mean for a Beta distribution: 1 / [1 + (beta/alpha)] * range + low
                "bb": (1.0 / (1.0 + input_["b"]["bb"][1] / input_["b"]["bb"][0])) * (high - low) + low,
                "bc": input_["b"]["bc"][0],
            }
        }

        outs = []
        for _ in range(500):
            out = cumulative_distribution.sample(input_)
            outs.append(out)
            out = cumulative_distribution.sample_stochastic(input_)
            outs.append(out)

        check(np.mean(np.stack([o["a"][0] for o in outs], axis=0), axis=0), expected_mean["a"], atol=0.3)
        check(np.mean(np.stack([o["b"]["ba"][0] for o in outs], axis=0), axis=0),
              expected_mean["b"]["ba"][0], decimals=1)
        check(np.mean(np.stack([o["b"]["bb"][0] for o in outs], axis=0), axis=0),
              expected_mean["b"]["bb"][0], decimals=1)
        check(np.mean(np.stack([o["b"]["bc"][0] for o in outs], axis=0), axis=0),
              expected_mean["b"]["bc"][0], decimals=1)

        # Test log-likelihood outputs.
        params = param_space.sample(1)
        params["a"] = softmax(params["a"])
        # Make sure beta-values are within 0.0 and 1.0 for the numpy calculation (which doesn't have scaling).
        values = values_space.sample(1)
        log_prob_beta = np.log(beta.pdf(values["b"]["bb"], params["b"]["bb"][0], params["b"]["bb"][1]))
        # Now do the scaling for b/bb (beta values).
        values["b"]["bb"] = values["b"]["bb"] * (high - low) + low
        expected_log_llh = np.log(params["a"][0][values["a"][0]]) + \
            np.sum(np.log(norm.pdf(values["b"]["ba"][0], params["b"]["ba"][0], params["b"]["ba"][1]))) + \
            np.sum(log_prob_beta) + \
            np.sum(np.log(norm.pdf(values["b"]["bc"][0], params["b"]["bc"][0], params["b"]["bc"][1])))

        out = cumulative_distribution.log_prob(params, values)
        check(out, expected_log_llh, decimals=0)
