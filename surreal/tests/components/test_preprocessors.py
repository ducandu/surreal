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

import cv2
import numpy as np
import os
import unittest

from surreal.components.preprocessors.grayscale import GrayScale
from surreal.components.preprocessors.image_resize import ImageResize
from surreal.components.preprocessors.image_crop import ImageCrop
from surreal.components.preprocessors.preprocessor import Preprocessor
from surreal.components.preprocessors.sequence import Sequence
from surreal.spaces import Dict, Int
from surreal.tests.test_util import check


class TestPreprocessors(unittest.TestCase):
    """
    Tests different preprocessors alone and in combination.
    """
    def test_gray_scale_with_uint8_image(self):
        input_space = Int(256, shape=(2, 10, 3), dtype="uint8", main_axes="B")
        grayscale = GrayScale(keepdims=False)
        # Take large sample to make sure rounding is not a major issue.
        input_ = input_space.sample(400)
        expected = np.round(np.dot(input_[:, :, :, :3], [0.299, 0.587, 0.114]), 0).astype(dtype=input_.dtype)
        out = grayscale(input_)
        check(out, expected, atol=1)  # Ok to diverge by one (unit8) due to rounding.

    def test_image_resize(self):
        image_resize = ImageResize(width=4, height=4, interpolation="bilinear")
        input_ = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image.bmp"))
        expected = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/4x4x3_image_resized.bmp"))
        out = image_resize(input_)
        check(out, expected)

    def test_image_crop(self):
        image_crop = ImageCrop(x=7, y=1, width=8, height=12)
        input_ = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image.bmp"))
        expected = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/8x12x3_image_cropped.bmp"))
        out = image_crop(input_)
        check(out, expected)

    def test_sequence(self):
        seq_len = 3
        sequence = Sequence(sequence_length=seq_len, adddim=True)

        for _ in range(3):
            sequence.reset()
            self.assertTrue(sequence.last_shape is None)
            input_ = np.asarray([[1.0], [2.0], [3.0], [4.0]])
            out = sequence(input_)
            self.assertTrue(len(sequence.deque) == 3)
            check(
                out, np.asarray([[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]], [[4.0, 4.0, 4.0]]])
            )
            input_ = np.asarray([[1.1], [2.2], [3.3], [4.4]])
            out = sequence(input_)
            self.assertTrue(len(sequence.deque) == 3)
            check(
                out, np.asarray([[[1.0, 1.0, 1.1]], [[2.0, 2.0, 2.2]], [[3.0, 3.0, 3.3]], [[4.0, 4.0, 4.4]]])
            )
            input_ = np.asarray([[1.11], [2.22], [3.33], [4.44]])
            out = sequence(input_)
            self.assertTrue(len(sequence.deque) == 3)
            check(
                out, np.asarray([[[1.0, 1.1, 1.11]], [[2.0, 2.2, 2.22]], [[3.0, 3.3, 3.33]], [[4.0, 4.4, 4.44]]])
            )
            input_ = np.asarray([[10], [20], [30], [40]])
            out = sequence(input_)
            self.assertTrue(len(sequence.deque) == 3)
            check(
                out, np.asarray([[[1.1, 1.11, 10]], [[2.2, 2.22, 20]], [[3.3, 3.33, 30]], [[4.4, 4.44, 40]]])
            )

    def test_sequence_with_batch_pos_reset(self):
        sequence = Sequence(sequence_length=2, adddim=True)

        for i in range(3):
            sequence.reset()
            self.assertTrue(len(sequence.deque) == 0 if i == 0 else 2)
            expected = np.array([
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[5.0, 5.0], [6.0, 6.0]]
            ])
            out = sequence(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            check(out, expected)
            self.assertTrue(len(sequence.deque) == 2)

            expected = np.array([
                [[1.0, 0.1], [2.0, 0.2]],
                [[3.0, 0.3], [4.0, 0.4]],
                [[5.0, 0.5], [6.0, 0.6]]
            ])
            out = sequence(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
            check(out, expected)
            self.assertTrue(len(sequence.deque) == 2)

            # Reset at single batch position.
            sequence.reset(batch_position=1)

            expected = np.array([
                [[0.1, 10.0], [0.2, 20.0]],
                [[30.0, 30.0], [40.0, 40.0]],
                [[0.5, 50.0], [0.6, 60.0]]
            ])
            out = sequence(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]))
            check(out, expected)
            self.assertTrue(len(sequence.deque) == 2)

            # Check invalid batch shape.
            got_here = False
            try:
                # Push a different shape through the preprocessor and expect exception.
                sequence(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
            except AssertionError as e:
                got_here = True  # expected
            self.assertTrue(got_here and len(sequence.deque) == 2)

            expected = np.array([
                [[10.0, 100.0], [20.0, 200.0]],
                [[30.0, 300.0], [40.0, 400.0]],
                [[50.0, 500.0], [60.0, 600.0]]
            ])
            out = sequence(np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]))
            check(out, expected)
            self.assertTrue(len(sequence.deque) == 2)

    def test_sequence_preprocessor_with_container_space(self):
        sequence = Sequence(sequence_length=4, adddim=False)

        for i in range(3):
            sequence.reset()
            expected = (np.array([0.5, 0.5, 0.5, 0.5]), np.array([[0.6, 0.7] * 4,
                                                                  [0.8, 0.9] * 4]))
            out = sequence(tuple([np.array([0.5]), np.array([[0.6, 0.7], [0.8, 0.9]])]))
            check(out, expected)

            expected = (np.array([0.5, 0.5, 0.5, 0.6]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                   0.6, 0.7, 1.1, 1.1],
                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                   0.8, 0.9, 1.1, 1.1]]))
            out = sequence(tuple([np.array([0.6]), np.array([[1.1, 1.1], [1.1, 1.1]])]))
            check(out, expected)

            expected = (np.array([0.5, 0.5, 0.6, 0.7]), np.array([[0.6, 0.7, 0.6, 0.7,
                                                                   1.1, 1.1, 2.0, 2.1],
                                                                  [0.8, 0.9, 0.8, 0.9,
                                                                   1.1, 1.1, 2.2, 2.3]]))
            out = sequence((tuple([np.array([0.7]), np.array([[2.0, 2.1], [2.2, 2.3]])])))
            check(out, expected)

    def test_preprocessor_stack_with_non_batched_images(self):
        # Try some dict-specs as well.
        preprocessor = Preprocessor(
            {"type": "gray-scale", "keepdims": True},
            ImageCrop(x=7, y=1, width=8, height=12),
            {"type": "image__resize", "width": 4, "height": 4, "interpolation": "bilinear"},
            Sequence(sequence_length=4, adddim=True)
        )
        preprocessor.reset()

        input_ = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image.bmp"))
        assert input_ is not None
        out = preprocessor(input_)
        self.assertTrue(out.shape == (4, 4, 1, 4))
        check(out[:, :, :, 0], out[:, :, :, 1])
        check(out[:, :, :, 0], out[:, :, :, 2])
        check(out[:, :, :, 0], out[:, :, :, 3])

        input2 = cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image_2.bmp"))
        assert input2 is not None
        out = preprocessor(input2)
        check(out[:, :, :, 0], out[:, :, :, 1])
        check(out[:, :, :, 0], out[:, :, :, 2])
        got_here = False
        try:
            check(out[:, :, :, 0], out[:, :, :, 3])
        except AssertionError:
            got_here = True  # expected to fail
        self.assertTrue(got_here)
        out = preprocessor(input2)
        check(out[:, :, :, 0], out[:, :, :, 1])
        check(out[:, :, :, 2], out[:, :, :, 3])
        got_here = False
        try:
            check(out[:, :, :, 1], out[:, :, :, 2])
        except AssertionError:
            got_here = True  # expected to fail
        self.assertTrue(got_here)

    def test_preprocessor_stack_with_batched_images(self):
        preprocessor = Preprocessor(
            GrayScale(keepdims=True),
            ImageCrop(x=7, y=1, width=8, height=12),
            ImageResize(width=4, height=4, interpolation="bilinear"),
            Sequence(sequence_length=4, adddim=True)
        )

        # Some batch (2) of images.
        input_ = np.array([
            cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image.bmp")),
            cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image_2.bmp"))
        ])
        # Reverse the batch.
        input2 = np.array([
            cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image_2.bmp")),
            cv2.imread(os.path.join(os.path.dirname(__file__), "../images/16x16x3_image.bmp"))
        ])
        assert input_ is not None
        assert input2 is not None

        for _ in range(3):
            preprocessor.reset()

            out = preprocessor(input_)
            self.assertTrue(out.shape == (2, 4, 4, 1, 4))
            check(out[:, :, :, :, 0], out[:, :, :, :, 1])
            check(out[:, :, :, :, 0], out[:, :, :, :, 2])
            check(out[:, :, :, :, 0], out[:, :, :, :, 3])

            out = preprocessor(input2)
            check(out[:, :, :, :, 0], out[:, :, :, :, 1])
            check(out[:, :, :, :, 0], out[:, :, :, :, 2])
            got_here = False
            try:
                check(out[:, :, :, :, 0], out[:, :, :, :, 3])
            except AssertionError:
                got_here = True  # expected to fail
            self.assertTrue(got_here)

            out = preprocessor(input2)
            got_here = False
            check(out[:, :, :, :, 0], out[:, :, :, :, 1])
            try:
                check(out[:, :, :, :, 0], out[:, :, :, :, 2])
            except AssertionError:
                got_here = True  # expected to fail
            self.assertTrue(got_here)
            check(out[:, :, :, :, 2], out[:, :, :, :, 3])

            # Reset at single batch position.
            preprocessor.reset(batch_position=0)
            out = preprocessor(input_)
            # At slot 0, everything should be the same (we just reset that one.
            check(out[0, :, :, :, 0], out[0, :, :, :, 1])
            check(out[0, :, :, :, 0], out[0, :, :, :, 2])
            check(out[0, :, :, :, 0], out[0, :, :, :, 3])
            # But at slot 1, things will proceed normally.
            check(out[1, :, :, :, 1], out[1, :, :, :, 2])
            got_here = False
            try:
                check(out[1, :, :, :, 0], out[1, :, :, :, 1])
            except AssertionError:
                got_here = True
            self.assertTrue(got_here)

            got_here = False
            try:
                check(out[1, :, :, :, 2], out[1, :, :, :, 3])
            except AssertionError:
                got_here = True
            self.assertTrue(got_here)

    def test_preprocessor_stack_with_nested_dict_inputs(self):
        input_space = Dict(A=float, B=dict(B1=float, B2=float), main_axes="B")
        preprocessor = Preprocessor(
            dict(A=lambda i: i * 3, B=dict(B1=lambda i: i * 4, B2=lambda i: i * 5)),
            lambda i: i["A"] * 2 + i["B"]["B1"] * 3
        )

        for _ in range(5):
            preprocessor.reset()

            # Some varying batch of inputs.
            input_ = input_space.sample(np.random.randint(1, 5))

            expected = input_["A"] * 3 * 2 + input_["B"]["B1"] * 4 * 3
            out = preprocessor(input_)
            check(out, expected)
