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

import copy
import inspect
import tensorflow as tf
import tensorflow.nest as nest

from surreal.components.distribution_adapters.adapter_utils import get_adapter_type_from_distribution_type, \
    get_distribution_spec_from_adapter
from surreal.components.distributions.distribution import Distribution
from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter
from surreal.components.models import Model
from surreal.spaces import Float, Int, Space, PrimitiveSpace
from surreal.spaces.space_utils import get_default_distribution_from_space
from surreal.utils.errors import SurrealError
from surreal.utils.keras import keras_from_spec
from surreal.utils.nest import flatten_alongside
from surreal.utils.util import complement_struct


# TODO: Change this into a multi-input-stream function, no matter, what the input_space is.
class Network(Model):
    """
    A generic function approximator holding a network and an output adapter and offering an intuitive call-API.
    """
    def __init__(
            self, network, output_space, adapters=None, distributions=False, deterministic=False,
            input_space=None
    ):
        """
        Args:
            network (Union[tf.keras.Model,tf.keras.Layer,callable]): The neural network callable
                (w/o the final action-layer) for this function approximator.

            output_space (Space): The output Space (may be a ContainerSpace).

            adapters (dict):

            distributions (Union[Dict,bool,str]): Distribution specification for the different output components.
                Supported values are:
                Dict[str,any]: A dictionary, matching the output space's structure and specifying for each component,
                    what the distribution should be (or False/None for no distribution).
                bool: True if all components should have the default distribution according to their Space type.
                    False if no component should have a distribution.
                "default": See True.
                None: See False.
                Values of True/False/"default"/None may also be given inside a nested dict (see Dict above) for
                    specific components of the output space.

            deterministic (bool): Whether to sample (from possible distributions) deterministically.
                Default: False (stochastic sampling).

            input_space (Optional[Space]): Input space may be provided to ensure immediate build of the network (and
                its variables).
        """
        super().__init__()

        # Store the given tf.keras.Model.
        self.network = network

        self.deterministic = deterministic

        # Create the output adapters.
        self.output_space = None
        self.flat_output_space = None
        # The adapters linking the main NN's output to the output layer(s)/distributions.
        self.adapters = []
        # The distributions to use (if any) for different components of the output space.
        self.distributions = []
        self._create_adapters_and_distributions(output_space, adapters, distributions)

        # If input space given, push through a sample to build our weights.
        self.input_space = input_space
        if self.input_space is not None:
            self(self.input_space.sample())

    def _create_adapters_and_distributions(self, output_space, adapters, distributions):
        if output_space is None:
            adapter = DistributionAdapter.make(adapters)
            self.output_space = adapter.output_space
            # Assert single component action space.
            assert isinstance(self.output_space, PrimitiveSpace), \
                "ERROR: Action space must not be ContainerSpace if no `action_space` is given in Policy constructor!"
        else:
            self.output_space = Space.make(output_space)
        self.flat_output_space = nest.flatten(self.output_space)

        # Find out whether we have a generic adapter-spec (one for all output components).
        generic_adapter_spec_for_all_output_components = None
        if isinstance(adapters, dict) and not any(key in adapters for key in self.output_space):
            generic_adapter_spec_for_all_output_components = adapters
        # adapters may be incomplete (add Nones to non-defined leafs).
        elif isinstance(adapters, dict):
            adapters = complement_struct(adapters, reference_struct=self.output_space)
        flat_output_adapter_spec = flatten_alongside(adapters, alongside=self.output_space)

        # Find out whether we have a generic distribution-spec (one for all output components).
        generic_distribution_spec_for_all_output_components = None
        if isinstance(distributions, dict) and not any(key in distributions for key in self.output_space):
            generic_distribution_spec_for_all_output_components = distributions
        # adapters may be incomplete (add Nones to non-defined leafs).
        elif isinstance(distributions, dict):
            distributions = complement_struct(distributions, reference_struct=self.output_space)
        # No distributions whatsoever.
        elif not distributions:
            distributions = complement_struct({}, reference_struct=self.output_space)
        elif distributions is True or distributions == "default":
            distributions = complement_struct({}, reference_struct=self.output_space, value=True)
        flat_distribution_spec = tf.nest.flatten(distributions)

        # Figure out our Distributions.
        for i, output_component in enumerate(self.flat_output_space):
            # Generic spec -> Use it.
            if generic_adapter_spec_for_all_output_components:
                da_spec = copy.deepcopy(generic_adapter_spec_for_all_output_components)
                da_spec["output_space"] = output_component
            # Spec dict -> find setting in possibly incomplete spec.
            elif isinstance(adapters, dict):
                # If not specified in dict -> auto-generate AA-spec.
                da_spec = flat_output_adapter_spec[i]
                da_spec["output_space"] = output_component
            # Simple type spec.
            elif not isinstance(adapters, DistributionAdapter):
                da_spec = dict(output_space=output_component)
            # Direct object.
            else:
                da_spec = adapters

            # We have to get the type of the adapter from a distribution.
            dist_spec = None
            if isinstance(da_spec, dict) and "type" not in da_spec:
                # Single distribution settings for all output components.
                if generic_distribution_spec_for_all_output_components is not None:
                    if generic_distribution_spec_for_all_output_components is not False:
                        dist_spec = get_default_distribution_from_space(
                            output_component, **generic_distribution_spec_for_all_output_components
                        )

                else:
                    settings = flat_distribution_spec[i] if isinstance(flat_distribution_spec[i], dict) else {}
                    dist_spec = get_default_distribution_from_space(output_component, **settings)

                # No distribution.
                if not flat_distribution_spec[i]:
                    self.distributions.append(None)
                # Some distribution.
                else:
                    self.distributions.append(Distribution.make(dist_spec))
                    if self.distributions[-1] is None:
                        raise SurrealError(
                            "`output_component` is of type {} and not allowed in {} Component!".
                            format(type(output_space).__name__, type(self).__name__)
                        )
                # Special case: No distribution AND float -> plain output adapter.
                if not flat_distribution_spec[i] and isinstance(da_spec["output_space"], Float):
                    da_spec["type"] = "plain-output-adapter"
                # All other cases: Get adapter type from distribution spec
                # (even if we don't use a distribution in the end).
                else:
                    da_spec["type"] = get_adapter_type_from_distribution_type(dist_spec["type"])
                self.adapters.append(DistributionAdapter.make(da_spec))

            # da_spec is completely defined  -> Use it to get distribution.
            else:
                self.adapters.append(DistributionAdapter.make(da_spec))
                if distributions[i]:
                    dist_spec = get_distribution_spec_from_adapter(self.adapters[-1])
                    self.distributions.append(Distribution.make(dist_spec))

        # TODO: Can we avoid this registration process somehow?
        #for a in self.adapters:
        #    for k in a.keras_models:
        #        self.add_keras_model(k)

    def call(self, inputs, values=None, *, deterministic=None, likelihood=False, log_likelihood=False):
        """
        Computes Q(s) -> a by passing the inputs through our model
        """
        deterministic = deterministic if deterministic is not None else self.deterministic

        # Return struct according to output Space.
        nn_out = self.network(inputs)

        # Simple output -> Push through each of our output-adapters.
        if not isinstance(nn_out, (tuple, dict)):
            # No values given -> Sample from distribution or return plain adapter-output (if no distribution given).
            if values is None:
                adapter_outputs = [a(nn_out) for a in self.adapters]
                tfp_distributions = [
                    distribution.parameterize_distribution(adapter_outputs[i])
                    if distribution is not None else None
                    for i, distribution in enumerate(self.distributions)
                ]
                sample = [
                    distribution._sample(tfp_distributions[i], deterministic=deterministic)
                    if distribution is not None else adapter_outputs[i]
                    for i, distribution in enumerate(self.distributions)
                ]
                packed_sample = nest.pack_sequence_as(self.output_space.structure, sample)
                # Return (combined?) likelihood values for each sample along with sample.
                if likelihood is True:
                    llhs = [
                        distribution._prob(tfp_distributions[i], sample[i])
                        for i, distribution in enumerate(self.distributions) if distribution
                    ]
                    total_llh = 1.0
                    for llh in llhs:
                        total_llh *= llh
                    return packed_sample, total_llh
                else:
                    return packed_sample
            # Values given -> Return probabilities/likelihoods or plain outputs for given values (if no distribution).
            else:
                values = complement_struct(values, self.output_space, "_undef_")
                flat_values = tf.nest.flatten(values)
                combined_likelihood = None
                for i, distribution in enumerate(self.distributions):
                    if distribution is not None and flat_values[i] is not "_undef_" and flat_values[i] is not None:
                        llhs = distribution.prob(self.adapters[i](nn_out), flat_values[i])
                        llh = tf.math.reduce_prod(llhs, axis=[-i - 1 for i in range(len(self.flat_output_space[i].shape))])
                        combined_likelihood = (combined_likelihood if combined_likelihood is not None else 1.0) * llh

                outputs = []
                for i, distribution in enumerate(self.distributions):
                    if distribution is None and flat_values[i] is not "_undef_":
                        if flat_values[i] is not None and flat_values[i] is not False:
                            assert isinstance(self.flat_output_space[i], Int)
                            outputs.append(tf.math.reduce_sum(
                                self.adapters[i](nn_out) *
                                tf.one_hot(flat_values[i], depth=self.flat_output_space[i].num_categories), axis=-1
                            ))
                        else:
                            outputs.append(self.adapters[i](nn_out))
                    else:
                        outputs.append(None)

                if all(o is None for o in outputs):
                    return combined_likelihood

                packed_out = nest.pack_sequence_as(self.output_space.structure, outputs)
                if combined_likelihood is not None:
                    return packed_out, combined_likelihood
                else:
                    return packed_out

        # NN already outputs containers.
        else:
            # Must match self.output_space.
            nest.assert_same_structure(self.adapters, nn_out)
            adapter_outputs = [
                adapter(out) for out, adapter in zip(nest.flatten(nn_out), nest.flatten(self.adapters))
            ]
            return nest.pack_sequence_as(adapter_outputs, self.adapters)

    def entropy(self, inputs):
        pass  # TODO: implement

    def copy(self, trainable=True):
        # Hide non-copyable members.
        network = self.network
        adapters = self.adapters
        self.network = None
        self.adapters = None

        # Do the copy.
        copy_ = copy.deepcopy(self)
        copy_.network = self.clone_component(network, trainable=trainable)
        copy_.adapters = [a.copy(trainable=trainable) for a in adapters]

        # Put everything back in place and clone keras models.
        self.network = network
        self.adapters = adapters

        # Do a sample call to build the copy, then sync weights.
        if self.init_args is not None:
            copy_(*self.init_args, **self.init_kwargs)
            copy_.sync_from(self, tau=1.0)

        return copy_

    def _get_weights_list(self):
        ret = self.network.variables  # type: list
        for a in self.adapters:
            ret.extend(a.get_weights(as_ref=True))
        return ret

    @classmethod
    def make(cls, spec=None, **kwargs):
        """
        Override for simple Keras Sequential.
        """
        network = spec if isinstance(spec, (list, tuple)) else kwargs.get("network")
        # Layers are given as list -> Build a simple Keras sequential model using Keras configs.
        if isinstance(network, (list, tuple)):
            sequential = keras_from_spec(network)
            kwargs.pop("network", None)
            network = super().make(network=sequential, **kwargs)
        # Not sure what to do -> Pass on to parent's `make`.
        else:
            network = super().make(spec, **kwargs)

        # Inspect and add to Algo if caller is an algo.
        # [1] = direct caller of this method.
        # [0] = frame object
        caller_frame = inspect.stack()[1][0]
        # Caller has attribute `saveables` -> Add this component to `[caller].saveables`.
        if hasattr(caller_frame.f_locals["self"], "saveables") and \
                isinstance(caller_frame.f_locals["self"].saveables, list):
            caller_frame.f_locals["self"].saveables.append(network)
        return network
