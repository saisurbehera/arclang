from typing import List
from typing import Tuple
from functools import reduce
from typing import Union, Literal
from collections import namedtuple

import numpy as np


Point = namedtuple("Point", ["x", "y"])
from scipy.ndimage import label, generate_binary_structure


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
    
    
    def find_same_color_components_dfs(self, r: int, c: int, visited: np.ndarray, component: np.ndarray, value: int):
        visited[r, c] = True
        component[r, c] = value
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < self.h and 0 <= nc < self.w and self.mask[nr, nc] == value and not visited[nr, nc]:
                    self.find_same_color_components_dfs(nr, nc, visited, component, value)

    def list_same_color_components(self) -> List["Image"]:
        visited = np.zeros_like(self.mask, dtype=bool)
        components = []
        for value in np.unique(self.mask):
            if value == 0:  # Skip background
                continue
            for i in range(self.h):
                for j in range(self.w):
                    if self.mask[i, j] == value and not visited[i, j]:
                        component = np.zeros_like(self.mask)
                        self.find_same_color_components_dfs(i, j, visited, component, value)
                        components.append(Image(self.x, self.y, self.w, self.h, component))
        return components

    def find_connected_components_dfs(self, r: int, c: int, color: int, visited: np.ndarray, component: np.ndarray):
        if r < 0 or r >= self.h or c < 0 or c >= self.w or visited[r, c] or self.mask[r, c] != color:
            return
        visited[r, c] = True
        component[r, c] = color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.find_connected_components_dfs(r + dr, c + dc, color, visited, component)

    def list_components(self, strategy: str = 'dfs', fit: bool = False) -> List["Image"]:
        components = []
        
        if strategy == 'dfs':
            visited = np.zeros_like(self.mask, dtype=bool)
            for i in range(self.h):
                for j in range(self.w):
                    if self.mask[i, j] != 0 and not visited[i, j]:
                        component = np.zeros_like(self.mask)
                        self.find_connected_components_dfs(i, j, self.mask[i, j], visited, component)
                        components.append(component)
        
        elif strategy == 'partition':
            unique_colors = np.unique(self.mask)
            unique_colors = unique_colors[unique_colors != 0]  # Exclude background color 0
            for color in unique_colors:
                component = np.zeros_like(self.mask)
                component[self.mask == color] = color
                components.append(component)
        
        else:
            raise ValueError("Invalid strategy. Choose 'dfs' or 'partition'.")

        return [self._process_component(comp, fit) for comp in components]

    def _process_component(self, component: np.ndarray, fit: bool) -> "Image":
        if fit:
            rows, cols = np.where(component != 0)
            if len(rows) == 0 or len(cols) == 0:
                return None  # Skip empty components
            top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
            cropped = component[top:bottom+1, left:right+1]
            return Image(left, top, cropped.shape[1], cropped.shape[0], cropped)
        else:
            return Image(self.x, self.y, self.w, self.h, component)
        
    def count_components(self) -> int:
        return len(self.list_components())
    
    def count_components_col(self) -> int:
        return len(self.list_same_color_components())

    
    def aggressive_connected_components(self, connectivity=2) -> List["Image"]:
        """
        Find connected components with a more aggressive connectivity.
        
        :param connectivity: 1 for 4-connectivity, 2 for 8-connectivity, 
                             can be increased for more aggressive connectivity
        :return: List of Image objects, each representing a component
        """
        # Create a structure for the given connectivity
        struct = generate_binary_structure(2, connectivity)
        
        # Label the connected components
        labeled_array, num_features = label(self.mask > 0, structure=struct)
        
        components = []
        for i in range(1, num_features + 1):
            component = np.where(labeled_array == i, self.mask, 0)
            components.append(Image(self.x, self.y, self.w, self.h, component))
        
        return components

    def list_distinct_components(self, fit: bool = False) -> List["Image"]:
        # Use scipy's label function to identify distinct regions
        labeled_array, num_features = label(self.mask)
        
        components = []
        for i in range(1, num_features + 1):
            component = np.zeros_like(self.mask)
            component[labeled_array == i] = self.mask[labeled_array == i]
            
            if fit:
                rows, cols = np.where(component != 0)
                top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
                cropped = component[top:bottom+1, left:right+1]
                components.append(Image(left, top, cropped.shape[1], cropped.shape[0], cropped))
            else:
                components.append(Image(self.x, self.y, self.w, self.h, component))
        
        return components

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
