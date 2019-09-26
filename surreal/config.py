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

from surreal.makeable import Makeable


class Config(Makeable):
    """
    A Config class allowing for passing in a locals-dict, which it will iterate over to assign attributes to the given
    values. This saves a lot of boilerplate/duplicate code in those constructors that only do attribute assignment from
    c'tor args/kwargs.
    """
    def __init__(self, locals_):
        for k, v in locals_.items():
            if k == "self" or k == "__class__":
                continue
            setattr(self, k, v)
