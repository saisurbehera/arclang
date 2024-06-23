import unittest
import numpy as np
from arclang.image import Image, Piece  # Assuming your classes are in 'image_module.py'

class TestImageOperations(unittest.TestCase):
    def test_image_equality(self):
        img1 = Image(x=0, y=0, w=10, h=10, mask=np.random.randint(-128, 128, size=(10, 10), dtype=np.int8))
        img2 = Image(x=0, y=0, w=10, h=10, mask=img1.mask.copy())
        self.assertEqual(img1, img2)

    def test_image_inequality(self):
        img1 = Image(x=0, y=0, w=10, h=10, mask=np.random.randint(-128, 128, size=(10, 10), dtype=np.int8))
        img2 = Image(x=0, y=0, w=10, h=10, mask=np.random.randint(-128, 128, size=(10, 10), dtype=np.int8))
        self.assertNotEqual(img1, img2)

    def test_boundary_check(self):
        img = Image(x=0, y=0, w=5, h=5, mask=np.random.randint(-128, 128, size=(5, 5), dtype=np.int8))
        with self.assertRaises(IndexError):
            _ = img[5, 0]  # This should raise an IndexError as the index is out of bounds


    def test_safe_method(self):
        img = Image(x=0, y=0, w=5, h=5, mask=np.random.randint(-128, 128, size=(5, 5), dtype=np.int8))
        self.assertEqual(img.safe(-1, -1), 0)  # Out of bounds check
        self.assertEqual(img.safe(0, 0), img.mask[0, 0])  # In bounds check

    def test_hash_consistency(self):
        mask = np.random.randint(-128, 128, size=(5, 5), dtype=np.int8)
        img1 = Image(x=0, y=0, w=5, h=5, mask=mask)
        img2 = Image(x=0, y=0, w=5, h=5, mask=mask)
        self.assertEqual(img1.hash_image(), img2.hash_image())

if __name__ == '__main__':
    unittest.main(verbosity=2)
