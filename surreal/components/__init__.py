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

from surreal.makeable import Makeable
from surreal.components.misc import Decay, Constant, LinearDecay, ExponentialDecay, PolynomialDecay
from surreal.components.distributions import *
from surreal.components.preprocessors import *
from surreal.components.loss_functions import LossFunction
from surreal.components.memories import *
from surreal.components.networks import *
from surreal.components.optimizers import *
from surreal.utils.util import default_dict


# Register all specific sub-classes to Makeable's lookup dict.
default_dict(Makeable.__lookup_classes__, Decay.__lookup_classes__)
default_dict(Makeable.__lookup_classes__, Distribution.__lookup_classes__)
default_dict(Makeable.__lookup_classes__, Memory.__lookup_classes__)
default_dict(Makeable.__lookup_classes__, Network.__lookup_classes__)
default_dict(Makeable.__lookup_classes__, Optimizer.__lookup_classes__)
default_dict(Makeable.__lookup_classes__, Preprocessor.__lookup_classes__)

__all__ = [] + \
          list(set(map(lambda x: x.__name__, Decay.__lookup_classes__.values()))) + \
          list(set(map(lambda x: x.__name__, Distribution.__lookup_classes__.values()))) + \
          list(set(map(lambda x: x.__name__, Memory.__lookup_classes__.values()))) + \
          list(set(map(lambda x: x.__name__, Network.__lookup_classes__.values()))) + \
          list(set(map(lambda x: x.__name__, Optimizer.__lookup_classes__.values()))) + \
          list(set(map(lambda x: x.__name__, Preprocessor.__lookup_classes__.values())))
