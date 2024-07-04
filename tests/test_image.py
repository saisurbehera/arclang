import unittest

import numpy as np

from arclang.image import Piece
from arclang.image import Point
from arclang.image import Image  # Assuming your classes are in 'image_module.py'


class TestImageAdditional(unittest.TestCase):
    def test_hash_image(self):
        img1 = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        img2 = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        img3 = Image(0, 0, 2, 2, [[1, 2], [3, 5]])
        self.assertEqual(img1.hash_image(), img2.hash_image())
        self.assertNotEqual(img1.hash_image(), img3.hash_image())

    def test_full_static_method(self):
        img = Image.full(Point(1, 1), Point(2, 2), filling=3)
        self.assertEqual(img.x, 1)
        self.assertEqual(img.y, 1)
        self.assertEqual(img.w, 2)
        self.assertEqual(img.h, 2)
        self.assertTrue(np.all(img.mask == 3))

    def test_empty_static_method(self):
        img = Image.empty_p(Point(1, 1), Point(2, 2))
        self.assertEqual(img.x, 1)
        self.assertEqual(img.y, 1)
        self.assertEqual(img.w, 2)
        self.assertEqual(img.h, 2)
        self.assertTrue(np.all(img.mask == 0))

    def test_is_rectangle(self):
        img1 = Image(0, 0, 2, 2, [[1, 1], [1, 1]])
        img2 = Image(0, 0, 2, 2, [[1, 1], [0, 1]])
        self.assertTrue(Image.is_rectangle(img1))
        self.assertFalse(Image.is_rectangle(img2))

    def test_count_components(self):
        img = Image(0, 0, 3, 3, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(img.count_components_col(), 5)

    def test_majority_col(self):
        img = Image(0, 0, 3, 3, [[1, 2, 1], [2, 1, 2], [1, 2, 1]])
        self.assertEqual(img.majority_col(), 1)
        self.assertEqual(img.majority_col(include0=1), 1)

    def test_sub_image(self):
        img = Image(0, 0, 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub_img = img.sub_image(Point(1, 1), Point(2, 2))
        self.assertTrue(
            np.array_equal(sub_img.mask, np.array([[5, 6], [8, 9]], dtype=np.int8))
        )

    def test_split_cols(self):
        img = Image(0, 0, 2, 2, [[1, 2], [0, 1]])
        split_result = img.split_cols()
        self.assertEqual(len(split_result), 2)
        self.assertTrue(
            np.array_equal(
                split_result[0][0].mask, np.array([[1, 0], [0, 1]], dtype=np.int8)
            )
        )
        self.assertEqual(split_result[0][1], 1)
        self.assertTrue(
            np.array_equal(
                split_result[1][0].mask, np.array([[0, 1], [0, 0]], dtype=np.int8)
            )
        )
        self.assertEqual(split_result[1][1], 2)


if __name__ == "__main__":
    unittest.main()
