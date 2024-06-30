import numpy as np
from typing import List, Tuple
import unittest

# Assuming the Image class and all ported functions are imported here
from arclang.function import *

class TestImageFunctions(unittest.TestCase):
    def setUp(self):
        # Create some test images
        self.img1 = Image(0, 0, 5, 5, np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4]
        ]))
        self.img2 = Image(0, 0, 5, 5, np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]))
        self.img3 = Image(0, 0, 5, 5, np.array([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 3, 0, 3]
        ]))

    def test_split_all(self):
        result = split_all(self.img1)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(isinstance(img, Image) for img in result))

    def test_erase_col(self):
        result = erase_col(self.img2, 2)
        expected = Image(0, 0, 5, 5, np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 3, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]))
        self.assertTrue(np.array_equal(result.mask, expected.mask))

    def test_inside_marked(self):
        result = inside_marked(self.img2)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(img, Image) for img in result))

    def test_make_border(self):
        result = make_border(self.img1)
        self.assertEqual(result.w, 5)
        self.assertEqual(result.h, 5)
        self.assertTrue(np.any(result.mask == 1))

    def test_make_border2(self):
        result = make_border2(self.img1)
        self.assertEqual(result.w, 7)
        self.assertEqual(result.h, 7)

    def test_compress2(self):
        result = compress2(self.img1)
        self.assertEqual(result.w, 4)
        self.assertEqual(result.h, 4)


    def test_greedy_fill(self):
        pieces = [(1, [1, 1, 1, 1])]
        done = np.zeros((5, 5), dtype=int)
        result = greedy_fill(self.img1, pieces, done, 2, 2, 1000)
        self.assertIsNotNone(result)

    def test_greedy_fill_black(self):
        result = greedy_fill_black(self.img1)
        self.assertIsNotNone(result)

    def test_greedy_fill_black2(self):
        result = greedy_fill_black2(self.img1)
        self.assertIsNotNone(result)

    def test_extend2(self):
        room = Image(0, 0, 7, 7)
        result = extend2(self.img1, room)
        self.assertEqual(result.w, 7)
        self.assertEqual(result.h, 7)

    def test_connect(self):
        result = connect(self.img3, 0)
        self.assertTrue(np.any(result.mask != 0))

    def test_replace_template(self):
        template = Image(0, 0, 2, 2, np.array([[1, 1], [1, 1]]))
        marked = Image(0, 0, 2, 2, np.array([[2, 2], [2, 2]]))
        result = replace_template(self.img1, template, marked)
        self.assertTrue(np.any(result.mask == 2))

    def test_swap_template(self):
        a = Image(0, 0, 2, 2, np.array([[1, 1], [1, 1]]))
        b = Image(0, 0, 2, 2, np.array([[2, 2], [2, 2]]))
        result = swap_template(self.img1, a, b)
        self.assertTrue(np.any(result.mask == 2))

    def test_spread_cols(self):
        result = spread_cols(self.img1)
        self.assertTrue(np.all(result.mask != 0))

    def test_split_columns(self):
        result = split_columns(self.img1)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(img.w == 1 for img in result))

    def test_split_rows(self):
        result = split_rows(self.img1)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(img.h == 1 for img in result))

    def test_half(self):
        result = half(self.img1, 0)
        self.assertEqual(result.w, 2)
        self.assertEqual(result.h, 5)

    def test_smear_each(self):
        result = smear_each(self.img1, 0)
        self.assertTrue(np.any(result.mask != self.img1.mask))

    def test_mirror2(self):
        line = Image(0, 0, 5, 1)
        result = mirror2(self.img1, line)
        self.assertEqual(result.h, 5)
        self.assertEqual(result.w, 5)

    def test_gravity(self):
        result = gravity(self.img1, 0)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(img, Image) for img in result))

    def test_my_stack(self):
        lens = [self.img1, self.img2]
        result = my_stack_l(lens, 0)
        self.assertIsNotNone(result)

    def test_stack_line(self):
        shapes = [self.img1, self.img2]
        result = stack_line(shapes)
        self.assertIsNotNone(result)

    def test_compose_growing_slow(self):
        imgs = [self.img1, self.img2]
        result = compose_growing_slow(imgs)
        self.assertIsNotNone(result)

    def test_compose_growing(self):
        imgs = [self.img1, self.img2]
        result = compose_growing(imgs)
        self.assertIsNotNone(result)

    def test_pick_unique(self):
        imgs = [self.img1, self.img2]
        result = pick_unique(imgs, 0)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()