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


class SurrealError(Exception):
    """
    Simple Error class.
    """
    pass


class SurrealObsoletedError(SurrealError):
    """
    An error raised when some obsoleted method, property, etc. is used.
    """
    def __init__(self, type_, old_value, new_value):
        """
        Args:
            type_ (str): Some type description of what exactly is obsoleted.
            old_value (str): The obsoleted value used.
            new_value (str): The new (replacement) value that should have been used instead.
        """
        msg = "The {} '{}' you are using has been obsoleted! Use '{}' instead.".format(type_, old_value, new_value)
        super().__init__(msg)


class SurrealSpaceError(SurrealError):
    """
    A Space related error. Raises together with a message and Space information.
    """
    def __init__(self, space, msg=None):
        """
        Args:
            space (Space): The Space that failed some check.
            input_arg (Optional[str]): An optional API-method input arg name.
            msg (Optional[str]): The error message.
        """
        super().__init__(msg)
        self.space = space


class RLGraphKerasStyleAssemblyError(SurrealError):
    """
    Special error to raise when constructing a NeuralNetwork using our Keras-style assembly support.
    """
    pass
