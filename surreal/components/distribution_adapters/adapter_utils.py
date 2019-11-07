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

import re

from surreal.components.distributions.distribution import Distribution
from surreal.utils.errors import SurrealError


def get_adapter_spec_from_distribution_spec(distribution_spec):
    """
    Args:
        distribution_spec (Union[dict,Distribution]): The spec of the Distribution object, for which to return an
            appropriate DistributionAdapter spec dict.

    Returns:
        dict: The spec-dict to make a DistributionAdapter.
    """
    # Create a dummy-distribution to get features from it.
    distribution = Distribution.make(distribution_spec)
    distribution_type_str = re.sub(r'[\W]|distribution$', "", type(distribution).__name__.lower())

    if distribution_type_str == "categorical":
        return dict(type="categorical-distribution-adapter")
    elif distribution_type_str == "gumbelsoftmax":
        return dict(type="gumbel-softmax-distribution-adapter")
    elif distribution_type_str == "bernoulli":
        return dict(type="bernoulli-distribution-adapter")
    elif distribution_type_str == "normal":
        return dict(type="normal-distribution-adapter")
    elif distribution_type_str == "multivariatenormal":
        return dict(type="multivariate-normal-distribution-adapter")
    elif distribution_type_str == "beta":
        return dict(type="beta-distribution-adapter")
    elif distribution_type_str == "squashednormal":
        return dict(type="squashed-normal-distribution-adapter")
    elif distribution_type_str == "mixture":
        return dict(
            type="mixture-distribution-adapter",
            _args=[get_adapter_spec_from_distribution_spec(re.sub(r'[\W]|distribution$', "", type(s).__name__.lower())) for
                   s in distribution.sub_distributions]
        )
    else:
        raise SurrealError("'{}' is an unknown Distribution type!".format(distribution_type_str))


def get_distribution_spec_from_adapter(distribution_adapter):
    distribution_adapter_type_str = type(distribution_adapter).__name__
    if distribution_adapter_type_str == "CategoricalDistributionAdapter":
        return dict(type="categorical")
    elif distribution_adapter_type_str == "GumbelSoftmaxDistributionAdapter":
        return dict(type="gumbel-softmax")
    elif distribution_adapter_type_str == "BernoulliDistributionAdapter":
        return dict(type="bernoulli")
    # TODO: What about multi-variate normal with non-trivial co-var matrices?
    elif distribution_adapter_type_str == "NormalDistributionAdapter":
        return dict(type="normal")
    elif distribution_adapter_type_str == "BetaDistributionAdapter":
        return dict(type="beta")
    elif distribution_adapter_type_str == "SquashedNormalDistributionAdapter":
        return dict(type="squashed-normal")
    elif distribution_adapter_type_str == "MixtureDistributionAdapter":
        # TODO: MixtureDistribution is generic (any sub-distributions, but its AA is not (only supports mixture-Normal))
        return dict(type="mixture", _args=["multivariate-normal" for _ in range(distribution_adapter.num_mixtures)])
    elif distribution_adapter_type_str == "PlainOutputAdapter":
        return None
    else:
        raise SurrealError("'{}' is an unknown DistributionAdapter type!".format(distribution_adapter_type_str))
