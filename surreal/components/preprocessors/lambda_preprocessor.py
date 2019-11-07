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

# Import these just in case they are needed inside the lambda code.
import numpy as np
import tensorflow as tf

from surreal.components.preprocessors.preprocessor import Preprocessor


class LambdaPreprocessor(Preprocessor):
    """
    Uses a given lambda function to preprocess some input.
    """
    def __init__(self, code):
        super().__init__()

        l_dict = {}
        # Execute the code.
        exec("lambda_function = {}".format(code), None, l_dict)
        self.lambda_ = l_dict["lambda_function"]

    def call(self, inputs):
        # Pass inputs through our lambda function.
        return self.lambda_(inputs)
