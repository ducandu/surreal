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

from surreal.components.distribution_adapters.adapter_utils import get_adapter_spec_from_distribution_spec, \
    get_distribution_spec_from_adapter
from surreal.components.distributions.distribution import Distribution
from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter
from surreal.components.models import Model
from surreal.spaces import Bool, Float, Int, Space, PrimitiveSpace, ContainerSpace
from surreal.spaces.space_utils import get_default_distribution_from_space
from surreal.utils.errors import SurrealError
from surreal.utils.keras import keras_from_spec
from surreal.utils.nest import flatten_alongside
from surreal.utils.util import complement_struct, default_dict


class Network(Model):
    """
    A generic function approximator holding a network and an output adapter and offering an intuitive call-API.
    """
    def __init__(
            self, network, *, output_space, adapters=None, distributions=False, deterministic=False,
            input_space=None, pre_concat_networks=None, auto_flatten_inputs=True
    ):
        """
        Args:
            network (Union[tf.keras.models.Model,tf.keras.layers.Layer,callable]): The neural network callable
                (w/o the final action-layer) for this function approximator.

            output_space (Space): The output Space (may be a ContainerSpace).

            adapters (dict): A dict for custom output/distribution-adapters, in case non-standard ones should
                be used OR extra network components should be run through only for these output-components
                (e.g. a dueling Q-network with two separate (non-shared) layers for A and V).

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
                its variables). Also, if it's a ContainerSpace, will build additional "pre-concat" NNs, through
                which input components are passed befor ebeing concat'd and sent further through the main NN.

            pre_concat_networks (Union[Dict,Tuple]): The neural network callable(s) for the different input
                components. Only applicable if `input_space` is given an a ContainerSpace.

            auto_flatten_inputs (bool): If True, will try to automatically flatten (or one-hot) all input components,
                but only if for that input-component, no `pre_concat_network` has been specified.
                For Int: One-hot along all non-main-axes. E.g. [[2, 3], [1, 2]] -> [0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
                For Float: Flatten along all non-main axes. E.g. [[2.0, 3.0], [1.0, 2.0]] -> [2.0 3.0 1.0 2.0]
                For Bool: Flatten along all non-main axes and convert to 0.0 (False) or 1.0 (True).
                Default: True.
        """
        super().__init__()

        # Store the given tf.keras.Model.
        self.network = network

        # Whether distribution outputs should be sampled deterministically.
        self.deterministic = deterministic

        # Create the output adapters.
        self.output_space = None
        self.flat_output_space = None
        # The adapters linking the main NN's output to the output layer(s)/distributions.
        self.adapters = []
        # The distributions to use (if any) for different components of the output space.
        self.distributions = []
        self._create_adapters_and_distributions(output_space, adapters, distributions)

        # Input space given explicitly.
        self.input_space = Space.make(input_space).with_batch() if input_space is not None else None
        self.flat_input_space = None
        self.pre_concat_networks = []  # One per input component.
        if self.input_space is not None:
            # If container space, build input NNs, then concat and connect to `self.network`.
            if isinstance(self.input_space, ContainerSpace):
                self._create_pre_concat_networks(pre_concat_networks, auto_flatten_inputs)
            # Push through a sample to build our weights.
            self(self.input_space.sample())

    def _create_adapters_and_distributions(self, output_space, adapters, distributions):
        if output_space is None:
            adapter = DistributionAdapter.make(adapters)
            self.output_space = adapter.output_space
            # Assert single component output space.
            assert isinstance(self.output_space, PrimitiveSpace), \
                "ERROR: Output space must not be ContainerSpace if no `output_space` is given in Network constructor!"
        else:
            self.output_space = Space.make(output_space)
        self.flat_output_space = tf.nest.flatten(self.output_space)

        # Find out whether we have a generic adapter-spec (one for all output components).
        generic_adapter_spec = None
        if isinstance(adapters, dict) and not any(key in adapters for key in self.output_space):
            generic_adapter_spec = adapters
        # adapters may be incomplete (add Nones to non-defined leafs).
        elif isinstance(adapters, dict):
            adapters = complement_struct(adapters, reference_struct=self.output_space)
        flat_output_adapter_spec = flatten_alongside(adapters, alongside=self.output_space)

        # Find out whether we have a generic distribution-spec (one for all output components).
        generic_distribution_spec = None
        if isinstance(self.output_space, PrimitiveSpace) or \
                (isinstance(distributions, dict) and not any(key in distributions for key in self.output_space)):
            generic_distribution_spec = distributions
            flat_distribution_spec = tf.nest.map_structure(lambda s: distributions, self.flat_output_space)
        else:
            # adapters may be incomplete (add Nones to non-defined leafs).
            if isinstance(distributions, dict):
                distributions = complement_struct(distributions, reference_struct=self.output_space)
            # No distributions whatsoever.
            elif not distributions:
                distributions = complement_struct({}, reference_struct=self.output_space)
            # Use default distributions (depending on output-space(s)).
            elif distributions is True or distributions == "default":
                distributions = complement_struct({}, reference_struct=self.output_space, value=True)
            flat_distribution_spec = tf.nest.flatten(distributions)

        # Figure out our Distributions.
        for i, output_component in enumerate(self.flat_output_space):
            # Generic spec -> Use it.
            if generic_adapter_spec:
                da_spec = copy.deepcopy(generic_adapter_spec)
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
            if isinstance(da_spec, dict) and "type" not in da_spec:
                # Single distribution settings for all output components.
                if generic_distribution_spec is not None:
                    settings = {} if generic_distribution_spec in ["default", True, False] else (generic_distribution_spec or {})
                else:
                    settings = flat_distribution_spec[i] if isinstance(flat_distribution_spec[i], dict) else {}
                # `distributions` could be simply a direct spec dict.
                if (isinstance(settings, dict) and "type" in settings) or isinstance(settings, Distribution):
                    dist_spec = settings
                else:
                    dist_spec = get_default_distribution_from_space(output_component, **settings)

                # No distribution.
                if not generic_distribution_spec and not flat_distribution_spec[i]:
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
                if not generic_distribution_spec and \
                        (not flat_distribution_spec[i] and isinstance(da_spec["output_space"], Float)):
                    da_spec["type"] = "plain-output-adapter"
                # All other cases: Get adapter type from distribution spec
                # (even if we don't use a distribution in the end).
                else:
                    default_dict(da_spec, get_adapter_spec_from_distribution_spec(dist_spec))

                self.adapters.append(DistributionAdapter.make(da_spec))

            # da_spec is completely defined  -> Use it to get distribution.
            else:
                self.adapters.append(DistributionAdapter.make(da_spec))
                if distributions[i]:
                    dist_spec = get_distribution_spec_from_adapter(self.adapters[-1])
                    self.distributions.append(Distribution.make(dist_spec))

    def _create_pre_concat_networks(self, pre_concat_networks, auto_flatten_inputs=True):
        # Find out whether we have a generic pre-concat-spec (one for all input components).
        generic_pre_concat_spec_for_all_input_components = None
        if isinstance(pre_concat_networks, dict) and not any(key in pre_concat_networks for key in self.input_space):
            generic_pre_concat_spec_for_all_input_components = pre_concat_networks
        # Spec may be incomplete (add Nones to non-defined leafs).
        elif isinstance(pre_concat_networks, dict):
            pre_concat_networks = complement_struct(pre_concat_networks, reference_struct=self.input_space.structure)
        # No distributions whatsoever.
        elif not pre_concat_networks:
            pre_concat_networks = complement_struct({}, reference_struct=self.input_space.structure)
        flat_pre_concat_nn_spec = flatten_alongside(pre_concat_networks, alongside=self.input_space)

        self.flat_input_space = tf.nest.flatten(self.input_space)

        # Figure out our pre-concat NNs.
        for i, input_component in enumerate(self.flat_input_space):
            # Generic spec -> Use it.
            if generic_pre_concat_spec_for_all_input_components:
                nn = keras_from_spec(generic_pre_concat_spec_for_all_input_components)
            # Spec dict -> find setting in possibly incomplete spec.
            else:
                # If not specified in dict.
                if flat_pre_concat_nn_spec[i] is None:
                    # Automatically pre-process inputs (flatten/one-hot/bool-to-float, etc..).
                    if auto_flatten_inputs is True:
                        nn = tf.keras.layers.Lambda(self._auto_input_lambda(input_component))
                    # No auto-preprocessing.
                    else:
                        nn = None
                # Manual preprocessing.
                else:
                    nn = keras_from_spec(flat_pre_concat_nn_spec[i])

            self.pre_concat_networks.append(nn)

    @staticmethod
    def _auto_input_lambda(input_component):
        """
        Creates automatic lambda Keras layers for certain input space components (e.g. int -> one-hot).
        This helps simplifying the generation of Networks from arbitrarily nested input-spaces.

        Args:
            input_component (Space): The input-space (sub)-component to create a Keras Lambda for.

        Returns:
            tf.keras.layers.Lambda: The Keras Lambda layer to use for processing the given input component.
        """
        new_shape = tuple([-1 for _ in range(len(input_component.main_axes))]) + \
                    (int(tf.reduce_prod(input_component.get_shape(with_category_rank=True))),)
        # Int -> One-hot and flatten down to main_axes.
        if isinstance(input_component, Int):
            return lambda i_: tf.reshape(tf.one_hot(i_, input_component.num_categories) if i_.dtype in [tf.int32, tf.int64] else i_, new_shape)
        # Float -> Flatten down to main_axes.
        elif isinstance(input_component, Float):
            return lambda i_: tf.reshape(i_, new_shape)
        # Bool -> Convert to float (0.0 and 1.0) and flatten down to main_axes.
        elif isinstance(input_component, Bool):
            return lambda i_: tf.reshape(tf.cast(i_, tf.float32), new_shape)
        # Unknown component Space -> Error.
        else:
            raise SurrealError("Unsupported input-space type: {}!".format(type(input_component).__name__))

    def call(self, inputs, values=None, *, deterministic=None, likelihood=False, log_likelihood=False,
             parameters_only=False):
        """
        Computes a forward pass through the neural network, plus (optionally) a distribution sampling step
        (deterministic or stochastic), plus (optionally) a (log)?-likelihood value for given `values` or the drawn
        sample.
        In other words, emulates common pseudocode:
        - q(s,a) <- Q-value of s,a.
        - pi(a|s) <- log likelihood/prob of a given s.
        - pi(s) <- action sample, given s.
        - pi(s, log_likelihood=True) <- action sample (given s), plus the log-likelihood of the drawn action.

        Args:
            inputs (any): The inputs to this Network (may be an arbitrarily nested structure).
            values (Optional[any]): The values to get (log)?-probs/likelihoods for.

            deterministic (Optional[bool]): If not None, use this setting (instead of `self.deterministic`) to determine
                whether a possible sampling from a distribution should be done deterministically or not.

            likelihood (bool): Whether to also return the likelihood (prob) when sampling, or the likelihood (prob)
                for the provided `values` for those output components that have distributions.

            log_likelihood (bool): Whether to also return the log-likelihood (log-prob) when sampling, or the
                log-likelihood (log-prob) for the provided `values` for those output components that have distributions.

            parameters_only (bool): Whether to only return the raw distribution parameters (no sampling) for all
                output components. Only meaningful if `values` is None.

        Returns:
            Depending on the given options, returns:
                1) If `values` is None: Plain output for output-components w/o a distribution, a (deterministic or
                    stochastic) sample for those with distribution.
                    If likelihood/log_likelihood are True, a second return item (the (log)?-likelihood) is returned.
                2) If `values` is not None: The likelihood/log-likelihood of the given value against the output
                    distributions. If some output components do not have distributions, a tuple:
                    ([output-components], [(log)?-likelihoods of those output component that do have a distribution])
        """
        deterministic = deterministic if deterministic is not None else self.deterministic

        # If complex input -> pass through pre_concat_nns, then concat, then move on through core nn.
        if len(self.pre_concat_networks) > 0:
            inputs = tf.nest.flatten(inputs)
            # Make sure input is roughly in line with `self.input_space`.
            assert len(self.flat_input_space) == len(inputs)
            inputs = tf.concat([
                self.pre_concat_networks[i](in_) if self.pre_concat_networks[i] is not None else in_
                for i, in_ in enumerate(inputs)
            ], axis=-1)

        # Return struct according to output Space.
        nn_out = self.network(inputs)

        # Simple output -> Push through each of our output-adapters.
        if not isinstance(nn_out, (tuple, dict)):
            # No values given -> Sample from distribution or return plain adapter-output (if no distribution given).
            if values is None:
                adapter_outputs = [a(nn_out) for a in self.adapters]
                # Only raw adapter outputs wanted.
                if parameters_only is True:
                    return tf.nest.pack_sequence_as(self.output_space.structure, adapter_outputs)

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
                packed_sample = tf.nest.pack_sequence_as(self.output_space.structure, sample)
                # Return (combined?) likelihood values for each sample along with sample.
                if likelihood is True or log_likelihood is True:
                    # Calculate probs/likelihoods (all in log-space for increased accuracy (for very small probs)).
                    log_llhs_components = [
                        # Reduce all axes that are not main_axes, so that we get only one
                        # (log)?-prob/likelihood per (composite)-action.
                        tf.reduce_sum(
                            distribution._log_prob(tfp_distributions[i], sample[i]),
                            axis=self.flat_output_space[i].reduction_axes
                        )
                        if distribution is not None else 0.0
                        for i, distribution in enumerate(self.distributions)
                    ]
                    # Combine all probs/likelihoods by multiplying up.
                    log_llh_sum = 0.0
                    for log_llh_component in log_llhs_components:
                        log_llh_sum += log_llh_component
                    return packed_sample, log_llh_sum if log_likelihood else tf.exp(log_llh_sum)

                else:
                    return packed_sample
            # Values given -> Return probabilities/likelihoods or plain outputs for given values (if no distribution).
            else:
                values = complement_struct(values, self.output_space.structure, "_undef_")
                flat_values = tf.nest.flatten(values)
                combined_likelihood_return = None
                for i, distribution in enumerate(self.distributions):
                    if distribution is not None and flat_values[i] is not "_undef_" and flat_values[i] is not None:
                        log_llh_sum = distribution.log_prob(self.adapters[i](nn_out), flat_values[i])
                        # `log_llh_sum` has not been reduced to the main axes yet (by the distribution) -> Do this here.
                        if len(self.flat_output_space[i].main_axes) < len(log_llh_sum.shape):
                            log_llh_sum = tf.math.reduce_sum(log_llh_sum, axis=self.flat_output_space[i].reduction_axes)
                        combined_likelihood_return = (combined_likelihood_return if combined_likelihood_return is not None else 0.0) + log_llh_sum
                if combined_likelihood_return is not None and not log_likelihood:
                    combined_likelihood_return = tf.math.exp(combined_likelihood_return)

                outputs = []
                for i, distribution in enumerate(self.distributions):
                    # No distribution.
                    if distribution is None and flat_values[i] is not "_undef_":
                        # Some value for this component was given.
                        if flat_values[i] is not None and flat_values[i] is not False:
                            # Make sure it's an Int space.
                            if not isinstance(self.flat_output_space[i], Int):
                                raise SurrealError(
                                    "Component {} of output space does not have a distribution and is not an Int. "
                                    "Hence, values for this component (to get outputs of likelihoodsfor) are not "
                                    "allowed in `call`."
                                )
                            # Return outputs for the discrete values by doing the sum-over-Hadamard-trick.
                            outputs.append(tf.math.reduce_sum(
                                self.adapters[i](nn_out) *
                                tf.one_hot(flat_values[i], depth=self.flat_output_space[i].num_categories), axis=-1
                            ))
                        # No value given, return plain adapter output.
                        else:
                            outputs.append(self.adapters[i](nn_out))
                    # Distribution: Already handled by likelihood block above.
                    else:
                        outputs.append(None)

                # Only likelihood expected (there are no non-distribution components in our output space).
                if all(o is None for o in outputs):
                    return combined_likelihood_return

                packed_out = tf.nest.pack_sequence_as(self.output_space.structure, outputs)
                if combined_likelihood_return is not None:
                    return packed_out, combined_likelihood_return
                else:
                    return packed_out

        # NN already outputs containers.
        else:
            # Must match self.output_space.
            tf.nest.assert_same_structure(self.adapters, nn_out)
            adapter_outputs = [
                adapter(out) for out, adapter in zip(tf.nest.flatten(nn_out), tf.nest.flatten(self.adapters))
            ]
            return tf.nest.pack_sequence_as(adapter_outputs, self.adapters)

    def entropy(self, inputs):
        pass  # TODO: implement

    def copy(self, trainable=True):
        # Hide non-copyable members.
        network = self.network
        pre_concat_nns = self.pre_concat_networks
        adapters = self.adapters
        self.network = None
        self.pre_concat_networks = None
        self.adapters = None

        # Do the copy.
        copy_ = copy.deepcopy(self)
        copy_.network = self.clone_component(network, trainable=trainable)
        copy_.pre_concat_networks = [
            self.clone_component(pre_concat_nn, trainable=trainable) if pre_concat_nn is not None else None
            for pre_concat_nn in pre_concat_nns
        ]
        copy_.adapters = [a.copy(trainable=trainable) for a in adapters]

        # Put everything back in place and clone keras models.
        self.network = network
        self.pre_concat_networks = pre_concat_nns
        self.adapters = adapters

        # Do a sample call to build the copy, then sync weights.
        if self.init_args is not None:
            copy_(*self.init_args, **self.init_kwargs)
            copy_.sync_from(self, tau=1.0)

        return copy_

    def _get_weights_list(self):
        ret = self.network.variables  # type: list
        for pre_nn in self.pre_concat_networks:
            if hasattr(pre_nn, "get_weights") and callable(pre_nn.get_weights):
                # Try whether get_weights has the `as_ref` option, if not, must be a native keras object ...
                try:
                    ret.extend(pre_nn.get_weights(as_ref=True))
                # ... in which case, we simply get `variables`.
                except TypeError as e:
                    ret.extend(pre_nn.variables)
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

        # Inspect and add to Algo if caller has `savables` property and is list.
        # [1] = direct caller of this method.
        # [0] = frame object
        caller_frame = inspect.stack()[1][0]
        # Caller has attribute `savables` -> Add this component to `[caller].savables`.
        if hasattr(caller_frame.f_locals["self"], "savables") and \
                isinstance(caller_frame.f_locals["self"].savables, list):
            caller_frame.f_locals["self"].savables.append(network)
        return network
