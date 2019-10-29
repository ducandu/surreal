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
import tensorflow as tf
import yaml

from surreal import PATH_SUMMARIES
from surreal.debug import UseTfSummaries
from surreal.makeable import Makeable


class Algo(Makeable, metaclass=ABCMeta):

    def __init__(self, config, name=None):
        """
        Args:
            config (AlgoConfig): The Algo's config object.
            name (str): Some name for this Algo (used by the Env to send events to this Algo).
        """
        super().__init__()

        self.config = config
        self.name = name or config.name

        # Savable components (e.g. networks).
        self.savables = []

        # Create a summary writer if necessary.
        self.summary_writer = None
        if UseTfSummaries is True and self.config.summaries is not None:
            self.summary_writer = tf.summary.create_file_writer(PATH_SUMMARIES)

    @abstractmethod
    def update(self, samples, time_percentage):
        """
        Performs an update of some form using the given batch (`samples`).

        Args:
            samples (any): The samples to use for the update.

            time_percentage (float): The time-percentage value (starting from 0.0 e.g. at beginning of learning to
                1.0 at the end of learning).

        Returns:
            any: Some data that may be important for the Algo for further processing (e.g. loss).
        """
        raise NotImplementedError

    @staticmethod
    def load(path, include_weights=True):
        """
        Loads an algorithm from the specified algo file, including (or excluding) current weight values.

        Args:
            path (str): The path/filename of the algo to load.
            include_weights (bool): Whether to include the current weight values of the Algo.

        Returns:
            Algo: The Algo after loading it from disk.
        """
        # Open the given file and load the data into a json/yaml struct.
        with open(path, 'r') as file:
            try:
                struct = json.load(file)
            except Exception as e:
                struct = yaml.load(file)

        print(struct)

    def save(self, path, data_format=None):
        """
        Saves this Algorithm by converting all its saveable components to json and storing everything in the given
        path/file.

        Args:
            path (str): The path/file to store the Algo's state in.
            data_format (Optional[str]): One of "json" or "yaml" OR given through `path` (file extension).
        """
        if data_format is None:
            data_format = path[-4:]

        assert data_format == "json" or data_format == "yaml"

        data = []
        for component in self.savables:
            data.append(component.serialize(data_format))

        with open(path, "w") as file:
            if data_format == "json":
                json.dump(dict(data=data), file)
            else:
                yaml.dump(dict(data=data), file)
