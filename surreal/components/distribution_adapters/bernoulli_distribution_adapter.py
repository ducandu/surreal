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


class BernoulliDistributionAdapter(DistributionAdapter):
    """
    Action adapter for the Bernoulli distribution.
    """
    def get_units_and_shape(self):
        units = self.output_space.flat_dim
        #new_shape = self.output_space.get_shape()
        #return units, new_shape
        new_shape = self.output_space.get_shape(include_main_axes=True)  #+((1,) if self.output_space.shape == () else ())
        new_shape = tuple([i if i is not None else -1 for i in new_shape])
        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        # `adapter_outputs` are interpreted as log-odds. Passing these through a softmax will yield the probs of a
        # `True` event.
        return adapter_outputs  # Logits (parameters for Bernoulli distribution).
