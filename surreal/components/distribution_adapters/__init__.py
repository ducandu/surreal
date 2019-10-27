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

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter
from surreal.components.distribution_adapters.adapter_utils import get_adapter_type_from_distribution_type, \
    get_distribution_spec_from_adapter
from surreal.components.distribution_adapters.bernoulli_distribution_adapter import BernoulliDistributionAdapter
from surreal.components.distribution_adapters.beta_distribution_adapter import BetaDistributionAdapter
from surreal.components.distribution_adapters.categorical_distribution_adapter import CategoricalDistributionAdapter
from surreal.components.distribution_adapters.gumbel_softmax_distribution_adapter import \
    GumbelSoftmaxDistributionAdapter
from surreal.components.distribution_adapters.normal_distribution_adapter import NormalDistributionAdapter
from surreal.components.distribution_adapters.mixture_distribution_adapter import \
    MixtureDistributionAdapter
from surreal.components.distribution_adapters.plain_output_adapter import PlainOutputAdapter
from surreal.components.distribution_adapters.squashed_normal_distribution_adapter import \
    SquashedNormalDistributionAdapter

DistributionAdapter.__lookup_classes__ = dict(
    distributionadapter=DistributionAdapter,
    bernoullidistributionadapter=BernoulliDistributionAdapter,
    categoricaldistributionadapter=CategoricalDistributionAdapter,
    betadistributionadapter=BetaDistributionAdapter,
    gumbelsoftmaxdistributionadapter=GumbelSoftmaxDistributionAdapter,
    gumbelsoftmaxadapter=GumbelSoftmaxDistributionAdapter,
    normaldistributionadapter=NormalDistributionAdapter,
    normalmixtureadapter=MixtureDistributionAdapter,
    normalmixturedistributionadapter=MixtureDistributionAdapter,
    plainoutputadapter=PlainOutputAdapter,
    squashednormaladapter=SquashedNormalDistributionAdapter,
    squashednormaldistributionadapter=SquashedNormalDistributionAdapter,
)

__all__ = ["DistributionAdapter", "get_adapter_type_from_distribution_type",
           "get_distribution_spec_from_action_adapter"] + \
          list(set(map(lambda x: x.__name__, DistributionAdapter.__lookup_classes__.values())))
