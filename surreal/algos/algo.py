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
import json
import yaml

from surreal.makeable import Makeable


class Algo(Makeable, metaclass=ABCMeta):

    def __init__(self, config, name=None):
        super().__init__()

        self.config = config
        self.name = name or config.name

        # Saveable components (e.g. networks).
        self.saveables = []

    def load(self, path):
        pass
        #data =
        #for component in self.saveables:
        #    data.append(component.serialize(data_format))

    def save(self, path, data_format="json"):
        """
        Saves this Algorithm by converting all its saveable components to json and storing everything in the given
        path/file.

        Args:
            path (str): The path/file to store the Algo's state in.
            data_format (str): One of "json" or "yaml".
        """
        assert data_format == "json" or data_format == "yaml"

        data = []
        for component in self.saveables:
            data.append(component.serialize(data_format))

        with open(path, "w") as file:
            if data_format == "json":
                json.dump(dict(data=data), file)
            else:
                yaml.dump(dict(data=data), file)
