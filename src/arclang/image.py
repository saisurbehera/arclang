import numpy as np
from functools import reduce
from collections import namedtuple

MAXSIDE = 100
MAXAREA = 40 * 40
MAXPIXELS = 40 * 40 * 5

Point = namedtuple('Point', ['x', 'y'])

class Image:
    def __init__(self, x=0, y=0, w=0, h=0, mask=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if mask is None:
            self.mask = np.zeros((h, w), dtype=np.int8)
        else:
            self.mask = np.array(mask, dtype=np.int8).reshape(h, w)

    def __getitem__(self, idx):
        i, j = idx
        # Uncomment the below lines to include boundary checking
        # assert 0 <= i < self.h and 0 <= j < self.w, "Index out of bounds"
        return self.mask[i, j]

    def __setitem__(self, idx, value):
        i, j = idx
        # Uncomment the below lines to include boundary checking
        # assert 0 <= i < self.h and 0 <= j < self.w, "Index out of bounds"
        self.mask[i, j] = value

    def safe(self, i, j):
        if i < 0 or j < 0 or i >= self.h or j >= self.w:
            return 0
        return self.mask[i, j]

    def __eq__(self, other):
        return np.array_equal(self.mask, other.mask) and self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if (self.w, self.h) != (other.w, other.h):
            return (self.w, self.h) < (other.w, other.h)
        return self.mask.flatten().tolist() < other.mask.flatten().tolist()

    def hash_image(self):
        base = 137
        r = 1543
        r = (r * base + self.w) % 2**64
        r = (r * base + self.h) % 2**64
        r = (r * base + self.x) % 2**64
        r = (r * base + self.y) % 2**64
        for c in self.mask.flatten():
            r = (r * base + int(c)) % 2**64
        return r

class Piece:
    def __init__(self, imgs=None, node_prob=0.0, keepi=0, knowi=0):
        if imgs is None:
            imgs = []
        self.imgs = imgs
        self.node_prob = node_prob
        self.keepi = keepi
        self.knowi = knowi

def check_all(v, f):
    return all(f(it) for it in v)

def all_equal(v, f):
    needed = f(v[0])
    return all(f(it) == needed for it in v)
