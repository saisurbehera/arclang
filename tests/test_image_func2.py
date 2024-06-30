import numpy as np
import unittest
from typing import List, Tuple
from arclang.function import *

class TestImageFunctions2(unittest.TestCase):
    def setUp(self):
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

    def test_swap_template(self):
        in_img = Image(0, 0, 5, 5, np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]))
        a = Image(0, 0, 3, 3, np.full((3, 3), 2))
        b = Image(0, 0, 3, 3, np.full((3, 3), 3))
        result = swap_template(in_img, a, b)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 3, 3, 3, 1],
            [1, 3, 3, 3, 1],
            [1, 3, 3, 3, 1],
            [1, 1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_spread_cols(self):
        img = Image(0, 0, 5, 5, np.array([
            [1, 0, 2, 0, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]))
        result = spread_cols(img)
        expected = np.array([
            [1, 1, 2, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 2, 3],
            [1, 1, 2, 2, 3]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_split_columns(self):
        result = split_columns(self.img1)
        self.assertEqual(len(result), 5)
        for i, col in enumerate(result):
            self.assertEqual(col.w, 1)
            self.assertEqual(col.h, 5)
            self.assertTrue(np.array_equal(col.mask, self.img1.mask[:, i:i+1]))

    def test_split_rows(self):
        result = split_rows(self.img1)
        self.assertEqual(len(result), 5)
        for i, row in enumerate(result):
            self.assertEqual(row.w, 5)
            self.assertEqual(row.h, 1)
            self.assertTrue(np.array_equal(row.mask, self.img1.mask[i:i+1, :]))

    def test_half(self):
        for i in range(4):
            result = half(self.img1, i)
            if i < 2:  # Vertical split
                self.assertEqual(result.w, 2)
                self.assertEqual(result.h, 5)
            else:  # Horizontal split
                self.assertEqual(result.w, 5)
                self.assertEqual(result.h, 2)
            if i % 2 == 0:  # Left or top half
                self.assertEqual(result.x, 0)
                self.assertEqual(result.y, 0)
            else:  # Right or bottom half
                self.assertEqual(result.x, 3 if i == 1 else 0)
                self.assertEqual(result.y, 3 if i == 3 else 0)

    def test_smear(self):
        img = Image(0, 0, 5, 5, np.array([
            [1, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 4]
        ]))
        result = smear_each(img, 6)  # All directions
        expected = np.array([[1, 2, 2, 2, 2],
       [3, 0, 0, 0, 4],
       [3, 0, 0, 0, 4],
       [3, 0, 0, 0, 4],
       [3, 4, 4, 4, 4]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_mirror2(self):
        img = Image(0, 0, 3, 3, np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
        line = Image(0, 0, 3, 1)  # Horizontal line
        result = mirror2(img, line)
        expected = np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))
        self.assertEqual(result.y, -2)  # Mirrored upwards

    def test_gravity(self):
        img = Image(0, 0, 5, 5, np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0],
            [0, 4, 0, 5, 0]
        ]))
        result = gravity(img, 2)  # Downwards
        self.assertEqual(len(result), 5)  # 5 components
        bottom_row = np.zeros(5)
        for component in result:
            y = component.y + component.h - 1
            x = component.x
            bottom_row[x] = component.mask[-1, 0]
        self.assertTrue(np.array_equal(bottom_row, [2, 4, 1, 5, 3]))

    def test_my_stack(self):
        imgs = [
            Image(0, 0, 2, 2, np.full((2, 2), 1)),
            Image(0, 0, 3, 3, np.full((3, 3), 2)),
            Image(0, 0, 4, 4, np.full((4, 4), 3))
        ]
        result = my_stack_l(imgs, 0)  # Horizontal stacking
        self.assertEqual(result.w, 9)
        self.assertEqual(result.h, 4)
        self.assertTrue(np.all(result.mask[:2, :2] == 1))
        self.assertTrue(np.all(result.mask[:4, 5:] == 3))

    def test_stack_line(self):
        imgs = [
            Image(0, 0, 2, 2, np.full((2, 2), 1)),
            Image(3, 0, 2, 2, np.full((2, 2), 2)),
            Image(6, 0, 2, 2, np.full((2, 2), 3))
        ]
        result = stack_line(imgs)
        self.assertEqual(result.w, 6)
        self.assertEqual(result.h, 2)


    def test_stack_line_v(self):
        imgs = [
            Image(0, 0, 2, 2, np.full((2, 2), 1)),
            Image(3, 0, 2, 2, np.full((2, 2), 2)),
            Image(6, 0, 2, 2, np.full((2, 2), 3))
        ]
        result = stack_line_v(imgs)
        self.assertEqual(result.w, 2)
        self.assertEqual(result.h, 6)

    def test_compose_growing_slow(self):
        imgs = [
            Image(0, 0, 3, 3, np.full((3, 3), 1)),
            Image(1, 1, 3, 3, np.full((3, 3), 2)),
            Image(2, 2, 3, 3, np.full((3, 3), 3))
        ]
        result = compose_growing_slow(imgs)
        expected = np.array([
            [1, 1, 1, 0, 0],
            [1, 2, 2, 2, 0],
            [1, 2, 3, 3, 3],
            [0, 2, 3, 3, 3],
            [0, 0, 3, 3, 3]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_compose_growing(self):
        imgs = [
            Image(0, 0, 3, 3, np.full((3, 3), 1)),
            Image(1, 1, 3, 3, np.full((3, 3), 2)),
            Image(2, 2, 3, 3, np.full((3, 3), 3))
        ]
        result = compose_growing(imgs)
        expected = np.array([
            [1, 1, 1, 0, 0],
            [1, 2, 2, 2, 0],
            [1, 2, 3, 3, 3],
            [0, 2, 3, 3, 3],
            [0, 0, 3, 3, 3]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_pick_unique(self):
        imgs = [
            Image(0, 0, 2, 2, np.full((2, 2), 1)),
            Image(0, 0, 2, 2, np.full((2, 2), 2)),
            Image(0, 0, 2, 2, np.array([[3, 3], [3, 4]]))
        ]
        result = pick_unique(imgs, 0)
        self.assertTrue(np.array_equal(result.mask, np.array([[3, 3], [3, 4]])))
    
    def test_greedy_fill(self):
        ret = Image(0, 0, 4, 4, np.zeros((4, 4), dtype=int))
        pieces = [(2, [1, 1, 1, 1])]
        done = np.zeros((4, 4), dtype=int)
        donew = 1000
        result = greedy_fill(ret, pieces, done, 2, 2, donew)
        expected = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_greedy_fill_black(self):
        img = Image(0, 0, 4, 4, np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        result = greedy_fill_black(img, N=2)
        expected = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_greedy_fill_black2(self):
        img = Image(0, 0, 4, 4, np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        result = greedy_fill_black2(img, N=2)
        expected = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_extend2(self):
        img = Image(0, 0, 3, 3, np.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ]))
        room = Image(0, 0, 5, 5)
        result = extend2(img, room)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_connect(self):
        img = Image(0, 0, 5, 5, np.array([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 3, 0, 3]
        ]))
        result = connect(img, 0)  # Horizontal
        expected = np.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_replace_template(self):
        in_img = Image(0, 0, 5, 5, np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]))
        need = Image(0, 0, 3, 3, np.array([
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ]))
        marked = Image(0, 0, 3, 3, np.array([
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ]))
        result = replace_template(in_img, need, marked)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 3, 3, 3, 1],
            [1, 3, 3, 3, 1],
            [1, 3, 3, 3, 1],
            [1, 1, 1, 1, 1]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    
if __name__ == '__main__':
    unittest.main()
