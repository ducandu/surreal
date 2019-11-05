# Copyright 2019 ducandu GmbH, All Rights Reserved.
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

import copy

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter


class MixtureDistributionAdapter(DistributionAdapter):
    def __init__(self, output_space, *sub_adapters, num_experts=None, **kwargs):
        """
        Args:
            sub_adapters (List[Union[string,DistributionAdapter]]): The type-strings or actual DistributionAdapter
                objects that define the n sub-adapters of this MixtureDistributionAdapter.

            num_experts (Optional[int]): If provided and len(`sub_adapters`) == 1, clone the given single
                sub_adapters `num_experts` times to get all sub_adapters.
        """
        self.num_experts = (num_experts or len(sub_adapters))
        super().__init__(output_space=output_space, **kwargs)

        self.sub_adapters = []

        # Default is some Normal.
        if len(sub_adapters) == 0:
            sub_adapters = ["normal-distribution-adapter"]
        # If only one given AND num_experts is provided, clone the sub_distribution config.
        if len(sub_adapters) == 1 and self.num_experts is not None:
            self.sub_adapters = [DistributionAdapter.make(
                {"type": sub_adapters[0], "output_space": output_space} if isinstance(sub_adapters[0], str)
                else sub_adapters[0]
            ) for _ in range(self.num_experts)]
        # Sub-distributions are given as n single configs.
        else:
            for i, s in enumerate(sub_adapters):
                self.sub_adapters.append(DistributionAdapter.make(
                    {"type": s, "output_space": output_space} if isinstance(s, str) else s)
                )

    def get_units_and_shape(self):
        new_shape = list(self.output_space.get_shape(with_category_rank=True))
        # num_experts=categorical nodes.
        new_shape = tuple(new_shape[:-1] + [self.num_experts])

        return self.num_experts, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        pass

    def call(self, inputs):
        parameters = {}
        if self.pre_network is not None:
            parameters["categorical"] = self.output_layer(self.pre_network(inputs))
        else:
            parameters["categorical"] = self.output_layer(inputs)

        # Get outputs of sub-adapters.
        for i, s in enumerate(self.sub_adapters):
            parameters["parameters{}".format(i)] = s(inputs)

        # Return parameters.
        return parameters  #self.get_parameters_from_adapter_outputs(outputs)

    def copy(self, trainable=None):
        # Hide non-copyable members.
        output_layer = self.output_layer
        pre_network = self.pre_network
        self.output_layer = None
        self.pre_network = None
        sub_adapters = self.sub_adapters
        self.sub_adapters = None

        # Do the copy.
        copy_ = copy.deepcopy(self)
        copy_.output_layer = self.clone_component(output_layer, trainable=trainable)
        if pre_network is not None:
            copy_.pre_network = self.clone_component(pre_network, trainable=trainable)
        # Copy all our sub-adapters.
        copy_.sub_adapters = [s.copy(trainable=trainable) for s in sub_adapters]

        # Put everything back in place and clone keras models.
        self.output_layer = output_layer
        self.pre_network = pre_network
        self.sub_adapters = sub_adapters

        # Do a sample call to build the copy, then sync weights.
        if self.init_args is not None:
            copy_(*self.init_args, **self.init_kwargs)
            copy_.sync_from(self, tau=1.0)

        return copy_

    def _get_weights_list(self):
        # Own weights.
        weights_list = (self.pre_network.variables if self.pre_network is not None else []) + \
               self.output_layer.variables
        # Get weights of sub-adapters.
        for i, s in enumerate(self.sub_adapters):
            weights_list.extend(list(s._get_weights_list()))
        return tuple(weights_list)
