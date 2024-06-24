import unittest
import numpy as np
from arclang.function import *
from arclang.image import Image, Point

class TestImageFunctions(unittest.TestCase):

    def test_col(self):
        img = col(3)
        self.assertEqual(img.w, 1)
        self.assertEqual(img.h, 1)
        self.assertEqual(img[0, 0], 3)

    def test_pos(self):
        img = pos(2, 3)
        self.assertEqual(img.x, 2)
        self.assertEqual(img.y, 3)
        self.assertEqual(img.w, 1)
        self.assertEqual(img.h, 1)

    def test_square(self):
        img = square(3)
        self.assertEqual(img.w, 3)
        self.assertEqual(img.h, 3)
        self.assertTrue(np.all(img.mask == 1))

    def test_line(self):
        img_h = line(0, 3)
        self.assertEqual(img_h.w, 3)
        self.assertEqual(img_h.h, 1)
        img_v = line(1, 3)
        self.assertEqual(img_v.w, 1)
        self.assertEqual(img_v.h, 3)

    def test_get_pos(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        pos = get_pos(img)
        self.assertEqual(pos.x, 2)
        self.assertEqual(pos.y, 3)
        self.assertEqual(pos.w, 1)
        self.assertEqual(pos.h, 1)
        self.assertEqual(pos[0, 0], 1)  # Assuming 1 is the majority color

    def test_get_size(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        size = get_size(img)
        self.assertEqual(size.w, 2)
        self.assertEqual(size.h, 2)
        self.assertEqual(size[0, 0], 1)  # Assuming 1 is the majority color

    def test_hull(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        hull_img = hull(img)
        self.assertEqual(hull_img.x, 2)
        self.assertEqual(hull_img.y, 3)
        self.assertEqual(hull_img.w, 2)
        self.assertEqual(hull_img.h, 2)
        self.assertTrue(np.all(hull_img.mask == 1))  # Assuming 1 is the majority color

    def test_to_origin(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        origin_img = to_origin(img)
        self.assertEqual(origin_img.x, 0)
        self.assertEqual(origin_img.y, 0)
        self.assertEqual(origin_img.w, 2)
        self.assertEqual(origin_img.h, 2)
        self.assertTrue(np.array_equal(origin_img.mask, img.mask))

    def test_get_w(self):
        img = Image(2, 3, 4, 5, np.ones((5, 4)))
        w_img = get_w(img, 0)
        self.assertEqual(w_img.w, 4)
        self.assertEqual(w_img.h, 1)
        w_img = get_w(img, 1)
        self.assertEqual(w_img.w, 4)
        self.assertEqual(w_img.h, 4)

    def test_get_h(self):
        img = Image(2, 3, 4, 5, np.ones((5, 4)))
        h_img = get_h(img, 0)
        self.assertEqual(h_img.w, 1)
        self.assertEqual(h_img.h, 5)
        h_img = get_h(img, 1)
        self.assertEqual(h_img.w, 5)
        self.assertEqual(h_img.h, 5)

    def test_hull0(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        hull0_img = hull0(img)
        self.assertEqual(hull0_img.x, 2)
        self.assertEqual(hull0_img.y, 3)
        self.assertEqual(hull0_img.w, 2)
        self.assertEqual(hull0_img.h, 2)
        self.assertTrue(np.all(hull0_img.mask == 0))

    def test_get_size0(self):
        img = Image(2, 3, 2, 2, [[1, 2], [3, 4]])
        size0 = get_size0(img)
        self.assertEqual(size0.w, 2)
        self.assertEqual(size0.h, 2)
        self.assertTrue(np.all(size0.mask == 0))

    def test_move(self):
        img = Image(1, 1, 2, 2, [[1, 2], [3, 4]])
        p = Image(2, 3, 1, 1, [[0]])
        moved = move(img, p)
        self.assertEqual(moved.x, 3)
        self.assertEqual(moved.y, 4)
        self.assertTrue(np.array_equal(moved.mask, img.mask))

    def test_filter_col(self):
        img = Image(0, 0, 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        palette = Image(0, 0, 2, 2, [[1, 3], [5, 7]])
        filtered = filter_col(img, palette)
        expected = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 0]])
        self.assertTrue(np.array_equal(filtered.mask, expected))

    def test_filter_col_id(self):
        img = Image(0, 0, 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        filtered = filter_col_id(img, 3)
        expected = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(filtered.mask, expected))

    def test_broadcast(self):
        col = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        shape = Image(0, 0, 4, 4)
        result = broadcast(col, shape)
        expected = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_col_shape(self):
        col = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        shape = Image(1, 1, 3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = col_shape(col, shape)
        expected = np.array([[1, 2, 1], [3, 0, 3], [1, 2, 1]])
        self.assertTrue(np.array_equal(result.mask, expected))
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 1)

    def test_col_shape_id(self):
        shape = Image(1, 1, 3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = col_shape_id(shape, 2)
        expected = np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_compress(self):
        img = Image(1, 1, 4, 4, [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        compressed = compress(img)
        self.assertEqual(compressed.x, 1 + 1)  # new x = original x + xmi
        self.assertEqual(compressed.y, 1 + 0)  # new y = original y + ymi
        self.assertEqual(compressed.w, 2)      # new width
        self.assertEqual(compressed.h, 3)      # new height
        expected = np.array([[1, 0], [1, 1], [0, 1]])
        self.assertTrue(np.array_equal(compressed.mask, expected))

    def test_embed(self):
        img = Image(1, 1, 2, 2, [[1, 2], [3, 4]])
        shape = Image(0, 0, 4, 4)
        embedded = embed(img, shape)
        expected = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(embedded.mask, expected))

    def test_compose(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(1, 1, 2, 2, [[5, 6], [7, 8]])
        result = compose(a, b, lambda x, y: max(x, y), 0)
        expected = np.array([[1, 2, 0], [3, 5, 6], [0, 7, 8]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_compose_id(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(1, 1, 2, 2, [[5, 6], [7, 8]])
        result = compose_id(a, b, 0)
        expected = np.array([[1, 2, 0], [3, 5, 6], [0, 7, 8]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_outer_product_is(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(0, 0, 2, 2, [[1, 0], [1, 1]])
        result = outer_product_is(a, b)
        expected = np.array([
            [1, 0, 2, 0],
            [1, 1, 2, 2],
            [3, 0, 4, 0],
            [3, 3, 4, 4]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_outer_product_si(self):
        a = Image(0, 0, 2, 2, [[1, 2], [0, 4]])
        b = Image(0, 0, 2, 2, [[5, 6], [7, 8]])
        result = outer_product_si(a, b)
        expected = np.array([
            [5, 6, 5, 6],
            [7, 8, 7, 8],
            [0, 0, 5, 6],
            [0, 0, 7, 8]
        ])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_fill(self):
        a = Image(0, 0, 3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = fill(a)
        expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_interior(self):
        a = Image(0, 0, 3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = interior(a)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_border(self):
        a = Image(0, 0, 3, 3, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        result = border(a)
        expected = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.assertTrue(np.array_equal(result.mask, expected))

    def test_align_x(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(2, 2, 4, 4)
        result = align_x(a, b, 2)
        self.assertEqual(result.x, 3)

    def test_align_y(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(2, 2, 4, 4)
        result = align_y(a, b, 2)
        self.assertEqual(result.y, 3)

    def test_align(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(2, 2, 4, 4)
        result = align(a, b, 2, 2)
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 3)

    def test_align_images(self):
        a = Image(0, 0, 2, 2, [[1, 2], [3, 4]])
        b = Image(2, 2, 2, 2, [[1, 2], [3, 4]])
        result = align_images(a, b)
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 2)
