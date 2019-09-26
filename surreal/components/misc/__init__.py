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

#from surreal.components.misc.batch_apply import BatchApply
#from surreal.components.misc.batch_splitter import BatchSplitter
#from surreal.components.misc.container_merger import ContainerMerger
#from surreal.components.misc.multi_gpu_synchronizer import MultiGpuSynchronizer
#from surreal.components.misc.noise_components import NoiseComponent, ConstantNoise, GaussianNoise, \
#    OrnsteinUhlenbeckNoise
#from surreal.components.misc.repeater_stack import RepeaterStack
#from surreal.components.misc.sampler import Sampler
#from surreal.components.misc.slice import Slice
#from surreal.components.misc.staging_area import StagingArea
#from surreal.components.misc.stop_gradient import StopGradient
#from surreal.components.misc.synchronizable import Synchronizable
from surreal.components.misc.decay_components import Decay, Constant, LinearDecay, PolynomialDecay, ExponentialDecay

Decay.__lookup_classes__ = dict(
    decay=Decay,
    constant=Constant,
    constantparameter=Constant,
    constantdecay=Constant,
    lineardecay=LinearDecay,
    polynomialdecay=PolynomialDecay,
    exponentialdecay=ExponentialDecay
)
Decay.__default_constructor__ = Constant

#NoiseComponent.__lookup_classes__ = dict(
#    noise=NoiseComponent,
#    constantnoise=ConstantNoise,
#    gaussiannoise=GaussianNoise,
#    ornsteinuhlenbeck=OrnsteinUhlenbeckNoise,
#    ornsteinuhlenbecknoise=OrnsteinUhlenbeckNoise
#)
#NoiseComponent.__default_constructor__ = GaussianNoise


__all__ = ["BatchApply", "ContainerMerger",
           "Synchronizable", "StopGradient", "RepeaterStack", "Slice",
           "Sampler", "BatchSplitter", "MultiGpuSynchronizer"] + \
          list(set(map(lambda x: x.__name__,
                       list(Decay.__lookup_classes__.values()) #+
                       #list(NoiseComponent.__lookup_classes__.values())
                       )))
