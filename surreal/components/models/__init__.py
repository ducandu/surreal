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

from surreal.components.models.model import Model
#from surreal.components.models.intrinsic_curiosity_world_option_model import IntrinsicCuriosityWorldOptionModel
#from surreal.components.models.supervised_model import SupervisedModel

Model.__lookup_classes__ = dict(
    model=Model,
    #intrinsiccuriosityworldoptionmodel=IntrinsicCuriosityWorldOptionModel,
    #supervisedmodel=SupervisedModel,
)

__all__ = ["Model"]
          #list(set(map(lambda x: x.__name__, Model.__lookup_classes__.values())))


