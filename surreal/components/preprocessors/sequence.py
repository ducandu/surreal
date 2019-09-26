# Copyright 2019 ducandu GmbH. All Rights Reserved.
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

from collections import deque
import numpy as np
import tensorflow as tf

from surreal.components.preprocessors.preprocessor import Preprocessor


class Sequence(Preprocessor):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property (velocity of game objects as they move across the screen).
    """

    def __init__(self, sequence_length=2, adddim=True):
        """
        Args:
            sequence_length (int): The number of records to always concatenate together within the last rank or
                in an extra (added) rank.

            adddim (bool): Whether to add another rank to the end of the input with dim=length-of-the-sequence.
                If False, concatenates the sequence within the last rank. Default: True.
        """
        # Switch off split (it's switched on for all LayerComponents by default).
        # -> accept any Space -> flatten to OrderedDict -> input & return OrderedDict -> re-nest.
        super().__init__()

        # Sequence preprocessors are stateful.
        self.has_state = True

        # The shape of the last inputs. Must stay the same over consecutive calls util `reset` is called.
        self.last_shape = None
        self.reset_batch_positions = None  # In case single batch position(s) should be reset.

        self.sequence_length = sequence_length
        self.adddim = adddim

        # The buffer's deque to store data as a series.
        self.deque = deque([], maxlen=self.sequence_length)

        self.reset()

    def reset(self, batch_position=None):
        # Complete reset.
        if batch_position is None:
            self.last_shape = None
            self.reset_batch_positions = None
        # Single slot reset. Only reset index at that batch position. Leave other indices intact.
        else:
            # Already in reset state -> return.
            if self.last_shape is None:
                self.reset_batch_positions = None
                return
            # Ok to reset single batch position.
            else:
                if self.reset_batch_positions is None:
                    self.reset_batch_positions = [batch_position]
                else:
                    self.reset_batch_positions.append(batch_position)

    def call(self, inputs):
        """
        Sequences (stitches) together the incoming inputs by using our buffer (with stored older records).
        Sequencing happens within the last rank if `self.add_rank` is False, otherwise a new rank is added at the end
        for the sequencing.

        Args:
            inputs (any): The input to be sequenced.

        Returns:
            any: The sequenced inputs.
        """
        deque_data = tf.nest.flatten(inputs)
        # `last_shape` is None: Fill buffer with all the same inputs, `self.sequence_length` times and store new
        # `last_shape`.
        if self.last_shape is None:
            for _ in range(self.sequence_length):
                self.deque.append(deque_data)
            self.last_shape = [d.shape for d in deque_data]
        # Indices are already defined, insert new item.
        else:
            # Assert same shape.
            assert self.last_shape == [d.shape for d in deque_data]
            self.deque.append(deque_data)
            # Reset single batch positions in deque to `sequence_length` x deque_data[at that batch position].
            if self.reset_batch_positions is not None:
                for pos in self.reset_batch_positions:
                    # Replace everywhere in deque.
                    for record in self.deque:
                        for i, data_item in enumerate(record):
                            data_item[pos] = deque_data[i][pos]
                self.reset_batch_positions = None

        # Add a new dim.
        if self.adddim:
            sequence = [np.stack([self.deque[j][i] for j in range(self.sequence_length)], axis=-1)
                        for i in range(len(self.deque[0]))]
        # Concat the sequence items in the last rank.
        else:
            sequence = [np.concatenate([self.deque[j][i] for j in range(self.sequence_length)], axis=-1)
                        for i in range(len(self.deque[0]))]

        return tf.nest.pack_sequence_as(inputs, sequence)
