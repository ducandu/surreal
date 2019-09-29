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

from abc import ABCMeta, abstractmethod

#from surreal import PATH_LOSS_LOGS
from surreal.debug import StoreLosses
from surreal.makeable import Makeable


class LossFunction(Makeable, metaclass=ABCMeta):
    """
    A generic (callable) LossFunction with debug logging functionality.
    Children must only implement the `call` method with the respective loss math.
    """
    def __init__(self):
        super().__init__()
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        out = self.call(*args, **kwargs)

        # For now, just print. Later, store to file, plot, etc..
        if StoreLosses is True and self.num_calls % 100 == 0:
            # If tuple is returned, assume that first slot is always the actual loss.
            # Some loss functions return additional information.
            print("Loss={}".format(out[0] if isinstance(out, tuple) else out))

        return out

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError
