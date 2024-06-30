import unittest

import numpy as np

from arclang.function import *
from arclang.image import Image
from arclang.image import Point

# Import the new display_matrix_term function
from arclang.utils import display_matrix_term


class TestImageFunctions(unittest.TestCase):

    def test_square(self):
        img = square(5)
        self.assertEqual(img.w, 5)
        self.assertEqual(img.h, 5)
        self.assertTrue(np.all(img.mask == 1))
        display_matrix_term(img)

    def test_line(self):
        img_h = line(0, 7)
        self.assertEqual(img_h.w, 7)
        self.assertEqual(img_h.h, 1)
        display_matrix_term(img_h)

        img_v = line(1, 7)
        self.assertEqual(img_v.w, 1)
        self.assertEqual(img_v.h, 7)
        display_matrix_term(img_v)

    def test_filter_col(self):
        img = Image(0, 0, 5, 5, np.random.randint(0, 10, (5, 5)))
        palette = Image(0, 0, 3, 3, np.random.randint(0, 10, (3, 3)))
        filtered = filter_col(img, palette)
        display_matrix_term(filtered)

    def test_broadcast(self):
        col = Image(0, 0, 3, 3, np.random.randint(0, 10, (3, 3)))
        shape = Image(0, 0, 6, 6)
        result = broadcast(col, shape)
        display_matrix_term(result)

    # ... (other test methods)


if __name__ == "__main__":
    unittest.main()
