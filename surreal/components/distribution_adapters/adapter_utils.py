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

from surreal.utils.errors import SurrealError


def get_adapter_type_from_distribution_type(distribution_type_str):
    """
    Args:
        distribution_type_str (str): The type (str) of the Distribution object, for which to return an appropriate
            DistributionAdapter lookup-class string.

    Returns:
        str: The lookup-class string for an action-adapter.
    """
    distribution_type_str = re.sub(r'[\W]', "", distribution_type_str.lower())
    # Int: Categorical.
    if distribution_type_str == "categorical":
        return "categorical-distribution-adapter"
    elif distribution_type_str == "gumbelsoftmax":
        return "gumbel-softmax-distribution-adapter"
    # Bool: Bernoulli.
    elif distribution_type_str == "bernoulli":
        return "bernoulli-distribution-adapter"
    # Continuous action space: Normal/Beta/etc. distribution.
    # Unbounded -> Normal distribution.
    elif distribution_type_str == "normal":
        return "normal-distribution-adapter"
    # Bounded -> Beta.
    elif distribution_type_str == "beta":
        return "beta-distribution-adapter"
    # Bounded -> Squashed Normal.
    elif distribution_type_str == "squashednormal":
        return "squashed-normal-distribution-adapter"
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
    elif distribution_adapter_type_str == "NormalMixtureDistributionAdapter":
        # TODO: MixtureDistribution is generic (any sub-distributions, but its AA is not (only supports mixture-Normal))
        return dict(type="mixture", _args=["multivariate-normal" for _ in range(distribution_adapter.num_mixtures)])
    elif distribution_adapter_type_str == "PlainOutputAdapter":
        return None
    else:
        raise SurrealError("'{}' is an unknown DistributionAdapter type!".format(distribution_adapter_type_str))
