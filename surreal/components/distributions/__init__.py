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

from surreal.components.distributions.bernoulli import Bernoulli
from surreal.components.distributions.beta import Beta
from surreal.components.distributions.categorical import Categorical
from surreal.components.distributions.distribution import Distribution
from surreal.components.distributions.gumbel_softmax import GumbelSoftmax
from surreal.components.distributions.joint_cumulative_distribution import JointCumulativeDistribution
from surreal.components.distributions.mixture_distribution import MixtureDistribution
from surreal.components.distributions.multivariate_normal import MultivariateNormal
from surreal.components.distributions.normal import Normal
from surreal.components.distributions.squashed_normal import SquashedNormal

Distribution.__lookup_classes__ = dict(
    bernoulli=Bernoulli,
    bernoullidistribution=Bernoulli,
    categorical=Categorical,
    categoricaldistribution=Categorical,
    gaussian=Normal,
    gaussiandistribution=Normal,
    gumbelsoftmax=GumbelSoftmax,
    gumbelsoftmaxdistribution=GumbelSoftmax,
    jointcumulative=JointCumulativeDistribution,
    jointcumulativedistribution=JointCumulativeDistribution,
    mixture=MixtureDistribution,
    mixturedistribution=MixtureDistribution,
    multivariatenormal=MultivariateNormal,
    multivariategaussian=MultivariateNormal,
    normal=Normal,
    normaldistribution=Normal,
    beta=Beta,
    betadistribution=Beta,
    squashed=SquashedNormal,
    squashednormal=SquashedNormal,
    squashednormaldistribution=SquashedNormal
)

__all__ = ["Distribution"] + list(set(map(lambda x: x.__name__, Distribution.__lookup_classes__.values())))

