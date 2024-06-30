from collections import namedtuple
from functools import reduce
from typing import List
from typing import Tuple
from typing import Union

import numpy as np


Point = namedtuple("Point", ["x", "y"])


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
        return self.mask[i, j]

    def __setitem__(self, idx, value):
        i, j = idx
        self.mask[i, j] = value

    def safe(self, i, j):
        if i < 0 or j < 0 or i >= self.h or j >= self.w:
            return 0
        return self.mask[i, j]

    def __eq__(self, other):
        return (
            np.array_equal(self.mask, other.mask)
            and self.x == other.x
            and self.y == other.y
            and self.w == other.w
            and self.h == other.h
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if (self.w, self.h) != (other.w, other.h):
            return (self.w, self.h) < (other.w, other.h)
        return self.mask.flatten().tolist() < other.mask.flatten().tolist()

    def copy(self) -> "Image":
        return Image(self.x, self.y, self.w, self.h, self.mask.copy())

    def col_mask(self) -> int:
        mask = 0
        for i in range(self.h):
            for j in range(self.w):
                mask |= 1 << self[i, j]
        return mask

    def count_cols(self, include0: int = 0) -> int:
        mask = self.col_mask()
        if not include0:
            mask &= ~1
        return bin(mask).count("1")

    def count(self) -> int:
        return np.sum(self.mask > 0)

    @staticmethod
    def full(p: Point, sz: Point, filling: int = 1) -> "Image":
        img = Image(p.x, p.y, sz.x, sz.y)
        img.mask.fill(filling)
        return img

    @staticmethod
    def full_i(p: Point, sz: Point, filling: int = 1) -> "Image":
        img = Image(p.x, p.y, sz.x, sz.y)
        img.mask.fill(filling)
        return img

    @staticmethod
    def empty_p(p: Point, sz: Point) -> "Image":
        return Image.full(p, sz, 0)

    @staticmethod
    def empty(x: int, y: int, w: int, h: int) -> "Image":
        return Image(x, y, w, h, np.zeros((h, w), dtype=np.int8))

    @staticmethod
    def is_rectangle(img: "Image") -> bool:
        return img.count() == img.w * img.h

    def count_components_dfs(self, r: int, c: int):
        self[r, c] = 0
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < self.h and 0 <= nc < self.w and self[nr, nc]:
                    self.count_components_dfs(nr, nc)

    def count_components(self) -> int:
        img_copy = self.mask.copy()
        ans = 0
        for i in range(self.h):
            for j in range(self.w):
                if img_copy[i, j]:
                    self.count_components_dfs(i, j)
                    ans += 1
        self.mask = img_copy  # Restore the original mask
        return ans

    def majority_col(self, include0: int = 0) -> int:
        unique, counts = np.unique(self.mask, return_counts=True)
        if not include0:
            zero_index = np.where(unique == 0)[0]
            if zero_index.size > 0:
                counts[zero_index[0]] = 0
        if counts.size == 0 or np.max(counts) == 0:
            return 0  # Return 0 if all colors were excluded or the image is empty
        return int(unique[np.argmax(counts)])

    def sub_image(self, p: Point, sz: Point) -> "Image":
        assert (
            p.x >= 0
            and p.y >= 0
            and p.x + sz.x <= self.w
            and p.y + sz.y <= self.h
            and sz.x >= 0
            and sz.y >= 0
        )
        ret = Image(p.x + self.x, p.y + self.y, sz.x, sz.y)
        ret.mask = self.mask[p.y : p.y + sz.y, p.x : p.x + sz.x].copy()
        return ret

    def split_cols(self, include0: int = 0) -> List[Tuple["Image", int]]:
        ret = []
        mask = self.col_mask()
        for c in range(int(not include0), 10):
            if mask >> c & 1:
                s = Image(self.x, self.y, self.w, self.h)
                s.mask = (self.mask == c).astype(np.int8)
                ret.append((s, c))
        return ret

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

    @staticmethod
    def empty_p2(p: Union[Point, int], sz: Union[Point, int], h: int = None) -> "Image":
        if isinstance(p, Point) and isinstance(sz, Point):
            return Image(p.x, p.y, sz.x, sz.y)
        elif isinstance(p, int) and isinstance(sz, int) and h is not None:
            return Image(p, sz, sz, h)
        else:
            raise ValueError("Invalid arguments for Image.empty")


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
