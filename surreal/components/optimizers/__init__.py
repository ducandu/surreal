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

from functools import partial

from surreal.components.optimizers.optimizer import Optimizer
from surreal.components.optimizers.adam import Adam
from surreal.components.optimizers.adadelta import Adadelta
from surreal.components.optimizers.adagrad import Adagrad
from surreal.components.optimizers.nadam import Nadam
from surreal.components.optimizers.rms_prop import RMSProp
from surreal.components.optimizers.sgd import SGD


Optimizer.__lookup_classes__ = dict(
    adagrad=Adagrad,
    adagradoptimizer=Adagrad,
    adadelta=Adadelta,
    adadeltaoptimizer=Adadelta,
    adam=Adam,
    adamoptimizer=Adam,
    nadam=Nadam,
    nadamoptimizer=Nadam,
    sgd=SGD,
    sgdoptimizer=SGD,
    gradientdescent=SGD,
    gradientdescentoptimizer=SGD,
    rmsprop=RMSProp,
    rmspropoptimizer=RMSProp
)

# The default Optimizer to use if a spec is None and no args/kwars are given.
Optimizer.__default_constructor__ = partial(SGD, learning_rate=0.0001)

__all__ = ["Optimizer"] + \
          list(set(map(lambda x: x.__name__, Optimizer.__lookup_classes__.values())))
