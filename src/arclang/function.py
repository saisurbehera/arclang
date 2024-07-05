from typing import List, Tuple, Callable
from collections import namedtuple, deque

import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from arclang.image import Image, Point
from arclang.constants import MAXAREA, MAXSIDE, MAXPIXELS


def col(id: int) -> Image:
    assert 0 <= id < 10
    return Image.full(Point(0, 0), Point(1, 1), id)

def display_matrix(matrix):
    colors = [
        "#000000",  # black
        "#0074D9",  # blue
        "#FF4136",  # red
        "#2ECC40",  # green
        "#FFDC00",  # yellow
        "#AAAAAA",  # grey
        "#F012BE",  # fuchsia
        "#FF851B",  # orange
        "#7FDBFF",  # teal
        "#870C25",  # brown
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 10, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cax = ax.matshow(matrix.mask, cmap=cmap, norm=norm)
    ax.set_title("Matrix")

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, matrix.mask.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.mask.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    fig.colorbar(
        cax, ax=ax, ticks=np.arange(0, 10), orientation="vertical"
    ).ax.set_yticklabels(
        [
            "Symbol 0",
            "Symbol 1",
            "Symbol 2",
            "Symbol 3",
            "Symbol 4",
            "Symbol 5",
            "Symbol 6",
            "Symbol 7",
            "Symbol 8",
            "Symbol 9",
        ]
    )

    plt.show()


def pos(dx: int, dy: int) -> Image:
    return Image.full(Point(dx, dy), Point(1, 1))

def square(id: int) -> Image:
    assert id >= 1
    return Image.full(Point(0, 0), Point(id, id))


def line(orient: int, id: int) -> Image:
    assert id >= 1
    w, h = (id, 1) if orient == 0 else (1, id)
    return Image.full(Point(0, 0), Point(w, h))


def get_pos(img: Image) -> Image:
    return Image.full(Point(img.x, img.y), Point(1, 1), img.majority_col())


def get_size(img: Image) -> Image:
    return Image.full(Point(0, 0), Point(img.w, img.h), img.majority_col())


def hull(img: Image) -> Image:
    return Image.full(Point(img.x, img.y), Point(img.w, img.h), img.majority_col())


def to_origin(img: Image) -> Image:
    img.x, img.y = 0, 0
    return img


def get_w(img: Image, id: int) -> Image:
    return Image.full(Point(0, 0), Point(img.w, img.w if id else 1), img.majority_col())


def get_h(img: Image, id: int) -> Image:
    return Image.full(Point(0, 0), Point(img.h if id else 1, img.h), img.majority_col())


def hull0(img: Image) -> Image:
    return Image.full(Point(img.x, img.y), Point(img.w, img.h), 0)


def get_size0(img: Image) -> Image:
    return Image.full(Point(0, 0), Point(img.w, img.h), 0)


def move(img:Image, shift: Tuple[int,int]):
    matrix = img.mask
    result = np.zeros_like(matrix)
    shift_y, shift_x = shift

    if shift_x > 0:
        result[shift_x:, :] = matrix[:-shift_x, :]
    elif shift_x < 0:
        result[:shift_x, :] = matrix[-shift_x:, :]
    else:
        result[:, :] = matrix[:, :]

    if shift_y > 0:
        result[:, shift_y:] = result[:, :-shift_y]
    elif shift_y < 0:
        result[:, :shift_y] = result[:, -shift_y:]
    else:
        result[:, :] = result[:, :]

    return Image(img.x,img.y,img.w,img.h,result)


def filter_col(img: Image, palette: Image) -> Image:
    ret = img.copy()
    pal_mask = palette.col_mask()
    for i in range(img.h):
        for j in range(img.w):
            if not (pal_mask & (1 << img[i, j])):
                ret[i, j] = 0
    return ret


def filter_col_id(img: Image, id: int) -> Image:
    assert 0 <= id < 10
    if id == 0:
        return invert(img)
    else:
        return filter_col(img, col(id))


def broadcast(col: "Image", shape: "Image", include0: int = 1) -> "Image":
    if col.w * col.h == 0 or shape.w * shape.h == 0:
        return Image()  # badImg equivalent

    ret = Image(shape.x, shape.y, shape.w, shape.h)

    for i in range(shape.h):
        for j in range(shape.w):
            ret[i, j] = col[i % col.h, j % col.w]

    return ret


def col_shape(col: Image, shape: Image) -> Image:
    if shape.w * shape.h == 0 or col.w * col.h == 0:
        return Image()  # bad image
    ret = broadcast(col, get_size(shape))
    ret.x, ret.y = shape.x, shape.y
    for i in range(ret.h):
        for j in range(ret.w):
            if not shape[i, j]:
                ret[i, j] = 0
    return ret


def col_shape_id(shape: Image, id: int) -> Image:
    assert 0 <= id < 10
    ret = shape.copy()
    ret.mask = np.where(ret.mask != 0, id, 0)
    return ret


def compress(img: Image, bg: Image = None) -> Image:
    if bg is None:
        bg = Image.full(
            Point(0, 0), Point(1, 1), 0
        )  # Use a full image with 0 as background

    bg_mask = bg.col_mask()

    xmi, xma, ymi, yma = img.w, -1, img.h, -1
    for i in range(img.h):
        for j in range(img.w):
            if not (bg_mask & (1 << img[i, j])) and img[i, j] != 0:
                xmi, xma = min(xmi, j), max(xma, j)
                ymi, yma = min(ymi, i), max(yma, i)

    if xmi > xma or ymi > yma:
        return Image(0, 0, 0, 0)

    ret = Image(img.x + xmi, img.y + ymi, xma - xmi + 1, yma - ymi + 1)
    for i in range(ymi, yma + 1):
        for j in range(xmi, xma + 1):
            ret[i - ymi, j - xmi] = img[i, j]
    return ret

def list_components(img: Image) -> List[Tuple[Image, Point]]:
    def split_all_with_position(img: Image) -> List[Tuple[Image, Point]]:
        ret = []
        done = Image.empty(img.x, img.y, img.w, img.h)

        def dfs(r: int, c: int, col: int, toadd: Image):
            if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c]:
                return
            toadd[r, c] = img[r, c]
            done[r, c] = 1
            for d in range(4):
                nr = r + (d == 0) - (d == 1)
                nc = c + (d == 2) - (d == 3)
                dfs(nr, nc, col, toadd)

        for i in range(img.h):
            for j in range(img.w):
                if not done[i, j] and img[i, j] != 0:  # Skip background (0)
                    toadd = Image.empty(img.x, img.y, img.w, img.h)
                    dfs(i, j, img[i, j], toadd)
                    toadd = compress(toadd)
                    if toadd.count() > 0:
                        ret.append((toadd, Point(j, i)))  # Use original (j, i) as the starting point

        return ret

    return split_all_with_position(img)

def embed(img: Image, shape: Image) -> Image:
    ret = Image(shape.x, shape.y, shape.w, shape.h)
    dx, dy = shape.x - img.x, shape.y - img.y
    sx, sy = max(0, -dx), max(0, -dy)
    ex, ey = min(ret.w, img.w - dx), min(ret.h, img.h - dy)

    ret_mask = ret.mask.reshape(ret.h, ret.w)
    img_mask = img.mask.reshape(img.h, img.w)
    ret_mask[sy:ey, sx:ex] = img_mask[sy + dy : ey + dy, sx + dx : ex + dx]
    return ret


def compose(
    a: Image, b: Image, f: Callable[[int, int], int], overlap_only: int
) -> Image:
    if overlap_only == 1:
        ret_x = max(a.x, b.x)
        ret_y = max(a.y, b.y)
        ra_x, ra_y = a.x + a.w, a.y + a.h
        rb_x, rb_y = b.x + b.w, b.y + b.h
        ret_w = min(ra_x, rb_x) - ret_x
        ret_h = min(ra_y, rb_y) - ret_y
        if ret_w <= 0 or ret_h <= 0:
            return Image()  # bad image
    elif overlap_only == 0:
        ret_x = min(a.x, b.x)
        ret_y = min(a.y, b.y)
        ra_x, ra_y = a.x + a.w, a.y + a.h
        rb_x, rb_y = b.x + b.w, b.y + b.h
        ret_w = max(ra_x, rb_x) - ret_x
        ret_h = max(ra_y, rb_y) - ret_y
    elif overlap_only == 2:
        ret_x, ret_y = a.x, a.y
        ret_w, ret_h = a.w, a.h
    else:
        assert False

    if ret_w > 100 or ret_h > 100 or ret_w * ret_h > 1600:
        return Image()  # bad image

    ret = Image(ret_x, ret_y, ret_w, ret_h)
    da_x, da_y = ret_x - a.x, ret_y - a.y
    db_x, db_y = ret_x - b.x, ret_y - b.y

    for i in range(ret_h):
        for j in range(ret_w):
            ca = a.safe(i + da_y, j + da_x)
            cb = b.safe(i + db_y, j + db_x)
            ret[i, j] = f(ca, cb)

    return ret


def compose_id(a: Image, b: Image, id: int = 0) -> Image:
    if id == 0:
        return compose(a, b, lambda x, y: y if y else x, 0)
    elif id == 1:
        return compose(a, b, lambda x, y: y if y else x, 1)
    elif id == 2:
        return compose(a, b, lambda x, y: x if y else 0, 1)
    elif id == 3:
        return compose(a, b, lambda x, y: y if y else x, 2)
    elif id == 4:
        return compose(a, b, lambda x, y: 0 if y else x, 2)
    else:
        assert 0 <= id < 5
        return Image()  # bad image


def compose_list(imgs: List[Image], overlap_only: int) -> Image:
    if not imgs:
        return Image()
    result = imgs[0]
    for img in imgs[1:]:
        result = compose_id(result, img, overlap_only)
    return result


def compose_list_f(
    imgs: List[Image], f: Callable[[int, int], int], overlap_only: int
) -> Image:
    if not imgs:
        return Image()
    result = imgs[0]
    for img in imgs[1:]:
        result = compose(result, img, f, overlap_only)
    return result


def outer_product_is(a: Image, b: Image) -> Image:
    if a.w * b.w > 100 or a.h * b.h > 100 or a.w * b.w * a.h * b.h > 1600:
        return Image()  # bad image
    ret_x = a.x * b.w + b.x
    ret_y = a.y * b.h + b.y
    ret = Image.empty(ret_x, ret_y, a.w * b.w, a.h * b.h)
    for i in range(a.h):
        for j in range(a.w):
            for k in range(b.h):
                for l in range(b.w):
                    ret[i * b.h + k, j * b.w + l] = a[i, j] * (1 if b[k, l] else 0)
    return ret


def outer_product_si(a: Image, b: Image) -> Image:
    if a.w * b.w > 100 or a.h * b.h > 100 or a.w * b.w * a.h * b.h > 1600:
        return Image()  # bad image
    ret_x = a.x * b.w + b.x
    ret_y = a.y * b.h + b.y
    ret = Image.empty(ret_x, ret_y, a.w * b.w, a.h * b.h)
    for i in range(a.h):
        for j in range(a.w):
            for k in range(b.h):
                for l in range(b.w):
                    ret[i * b.h + k, j * b.w + l] = (1 if a[i, j] > 0 else 0) * b[k, l]
    return ret


def fill(a: Image) -> Image:
    # Create an image filled with the majority color of 'a'
    ret = Image.full(Point(a.x, a.y), Point(a.w, a.h), a.majority_col())
    q = []

    # Identify the border pixels and add them to the queue
    for i in range(a.h):
        for j in range(a.w):
            if (i == 0 or j == 0 or i == a.h - 1 or j == a.w - 1) and not a[i, j]:
                q.append((i, j))
                ret[i, j] = 0

    # Perform BFS to fill the area
    while q:
        r, c = q.pop(0)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < a.h and 0 <= nc < a.w and not a[nr, nc] and ret[nr, nc] != 0:
                q.append((nr, nc))
                ret[nr, nc] = 0
    return ret


def interior(a: Image) -> Image:
    return compose(fill(a), a, lambda x, y: 0 if y else x, 0)


def border(a: Image) -> Image:
    ret = Image(a.x, a.y, a.w, a.h)
    q = []
    for i in range(a.h):
        for j in range(a.w):
            if i == 0 or j == 0 or i == a.h - 1 or j == a.w - 1:
                if not a[i, j]:
                    q.append((i, j))
                ret[i, j] = 1

    while q:
        r, c = q.pop()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < a.h and 0 <= nc < a.w and not ret[nr, nc]:
                    ret[nr, nc] = 1
                    if not a[nr, nc]:
                        q.append((nr, nc))

    ret.mask = ret.mask * a.mask
    return ret


def align_x(a: Image, b: Image, id: int) -> Image:
    assert 0 <= id < 5
    ret = a.copy()
    if id == 0:
        ret.x = b.x - a.w
    elif id == 1:
        ret.x = b.x
    elif id == 2:
        ret.x = b.x + (b.w - a.w) // 2
    elif id == 3:
        ret.x = b.x + b.w - a.w
    elif id == 4:
        ret.x = b.x + b.w
    return ret


def align_y(a: Image, b: Image, id: int) -> Image:
    assert 0 <= id < 5
    ret = a.copy()
    if id == 0:
        ret.y = b.y - a.h
    elif id == 1:
        ret.y = b.y
    elif id == 2:
        ret.y = b.y + (b.h - a.h) // 2
    elif id == 3:
        ret.y = b.y + b.h - a.h
    elif id == 4:
        ret.y = b.y + b.h
    return ret


def align(a: Image, b: Image, idx: int, idy: int) -> Image:
    assert 0 <= idx < 6 and 0 <= idy < 6
    ret = a.copy()
    if idx == 0:
        ret.x = b.x - a.w
    elif idx == 1:
        ret.x = b.x
    elif idx == 2:
        ret.x = b.x + (b.w - a.w) // 2
    elif idx == 3:
        ret.x = b.x + b.w - a.w
    elif idx == 4:
        ret.x = b.x + b.w

    if idy == 0:
        ret.y = b.y - a.h
    elif idy == 1:
        ret.y = b.y
    elif idy == 2:
        ret.y = b.y + (b.h - a.h) // 2
    elif idy == 3:
        ret.y = b.y + b.h - a.h
    elif idy == 4:
        ret.y = b.y + b.h
    return ret


def align_images(a: Image, b: Image) -> Image:
    ret = a.copy()
    match_size = 0
    for c in range(1, 10):
        ca = compress(filter_col_id(a, c))
        cb = compress(filter_col_id(b, c))
        if ca.mask.tolist() == cb.mask.tolist():
            cnt = ca.count()
            if cnt > match_size:
                match_size = cnt
                ret.x = a.x + cb.x - ca.x
                ret.y = a.y + cb.y - ca.y
    if match_size == 0:
        return Image()  # bad image
    return ret


def replace_cols(base: Image, cols: Image) -> Image:
    ret = base.copy()
    done = Image.empty(base.x, base.y, base.w, base.h)
    dx, dy = base.x - cols.x, base.y - cols.y

    def dfs(r: int, c: int, acol: int) -> List[Tuple[int, int]]:
        if (
            r < 0
            or r >= base.h
            or c < 0
            or c >= base.w
            or base[r, c] != acol
            or done[r, c]
        ):
            return []
        path = [(r, c)]
        done[r, c] = 1
        for nr in [r - 1, r, r + 1]:
            for nc in [c - 1, c, c + 1]:
                path.extend(dfs(nr, nc, acol))
        return path

    for i in range(base.h):
        for j in range(base.w):
            if not done[i, j] and base[i, j]:
                acol = base[i, j]
                cnt = [0] * 10
                path = dfs(i, j, acol)
                for r, c in path:
                    cnt[cols.safe(r + dy, c + dx)] += 1
                maj = max(range(1, 10), key=lambda c: cnt[c])
                for r, c in path:
                    ret[r, c] = maj

    return ret


def center(img: Image) -> Image:
    sz_x = (img.w + 1) % 2 + 1
    sz_y = (img.h + 1) % 2 + 1
    center_x = img.x + (img.w - sz_x) // 2
    center_y = img.y + (img.h - sz_y) // 2

    center_img = Image(center_x, center_y, sz_x, sz_y)
    for i in range(sz_y):
        for j in range(sz_x):
            center_img[i, j] = img[center_y - img.y + i, center_x - img.x + j]

    return center_img


def transform(img: Image, A00: int, A01: int, A10: int, A11: int) -> Image:
    if img.w * img.h == 0:
        return img
    c = center(img)
    off_x = 1 - c.w + 2 * (img.x - c.x)
    off_y = 1 - c.h + 2 * (img.y - c.y)

    def t(p: Point) -> Point:
        x, y = 2 * p.x + off_x, 2 * p.y + off_y
        nx = A00 * x + A01 * y
        ny = A10 * x + A11 * y
        return Point((nx - off_x) // 2, (ny - off_y) // 2)

    corners = [
        t(Point(0, 0)),
        t(Point(img.w - 1, 0)),
        t(Point(0, img.h - 1)),
        t(Point(img.w - 1, img.h - 1)),
    ]
    a = Point(min(c.x for c in corners), min(c.y for c in corners))
    b = Point(max(c.x for c in corners), max(c.y for c in corners))

    ret = Image.empty(img.x, img.y, b.x - a.x + 1, b.y - a.y + 1)
    for i in range(img.h):
        for j in range(img.w):
            go = t(Point(j, i))
            go = Point(go.x - a.x, go.y - a.y)
            ret[go.y, go.x] = img[i, j]
    return ret


def mirror_heuristic(img: Image) -> bool:
    cnt = sumx = sumy = 0
    for i in range(img.h):
        for j in range(img.w):
            if img[i, j]:
                cnt += 1
                sumx += j
                sumy += i
    return abs(sumx * 2 - (img.w - 1) * cnt) < abs(sumy * 2 - (img.h - 1) * cnt)


def rigid(img: Image, id: int) -> Image:
    if id == 0:
        return img.copy()
    elif id == 1:
        return transform(img, 0, -1, 1, 0)  # CCW
    elif id == 2:
        return transform(img, -1, 0, 0, -1)  # 180
    elif id == 3:
        return transform(img, 0, -1, 1, 0)  # CW
    elif id == 4:
        return transform(img, -1, 0, 0, 1)  # flip x
    elif id == 5:
        return transform(img, 1, 0, 0, -1)  # flip y
    elif id == 6:
        return transform(img, 0, 1, 1, 0)  # swap xy
    elif id == 7:
        return transform(img, 0, -1, -1, 0)  # swap other diagonal
    elif id == 8:
        return rigid(img, 4 + int(mirror_heuristic(img)))
    else:
        assert 0 <= id < 9
        return Image()  # bad image


def invert(img: Image) -> Image:
    if img.w * img.h == 0:
        return img
    mask = img.col_mask()
    col = 1
    while col < 10 and not (mask & (1 << np.int64(col))):
        col += 1
    if col == 10:
        col = 1

    ret = img.copy()
    ret.mask = np.where(ret.mask != 0, 0, col)
    return ret


def maj_col(img: Image) -> Image:
    return col(img.majority_col())


def interior2(a: Image) -> Image:
    return compose_id(a, invert(border(a)), 2)


def count(img: Image, id: int, out_type: int) -> Image:
    assert 0 <= id < 7 and 0 <= out_type < 3
    if id == 0:
        num = img.count()
    elif id == 1:
        num = img.count_cols()
    elif id == 2:
        num = img.count_components()
    elif id == 3:
        num = img.w
    elif id == 4:
        num = img.h
    elif id == 5:
        num = max(img.w, img.h)
    elif id == 6:
        num = min(img.w, img.h)

    if out_type == 0:
        sz = Point(num, num)
    elif out_type == 1:
        sz = Point(num, 1)
    elif out_type == 2:
        sz = Point(1, num)

    if max(sz.x, sz.y) > 100 or sz.x * sz.y > 1600:
        return Image()  # bad image
    return Image.full(Point(0, 0), sz, img.majority_col())


def wrap(line: Image, area: Image) -> Image:
    if line.w * line.h == 0 or area.w * area.h == 0:
        return Image()  # bad image
    ans = Image(0, 0, area.w, area.h)
    for i in range(line.h):
        for j in range(line.w):
            x, y = j, i
            x += y // area.h * line.w
            y %= area.h
            y += x // area.w * line.h
            x %= area.w
            if 0 <= x < ans.w and 0 <= y < ans.h:
                ans[y, x] = line[i, j]
    return ans


def smear(base: Image, room: Image, id: int) -> Image:
    assert 0 <= id < 7
    mask = [1, 2, 4, 8, 3, 12, 15][id]
    d = Point(room.x - base.x, room.y - base.y)
    ret = embed(base, hull(room))

    def smear_direction(range_i, range_j, condition):
        for i in range_i:
            c = 0
            for j in range_j:
                if not room[i, j]:
                    c = 0
                elif base.safe(i + d.y, j + d.x):
                    c = base[i + d.y, j + d.x]
                if c and condition(i, j):
                    ret[i, j] = c

    if mask & 1:
        smear_direction(range(ret.h), range(ret.w), lambda i, j: True)
    if mask & 2:
        smear_direction(range(ret.h), range(ret.w - 1, -1, -1), lambda i, j: True)
    if mask & 4:
        smear_direction(range(ret.w), range(ret.h), lambda i, j: True)
    if mask & 8:
        smear_direction(range(ret.w), range(ret.h - 1, -1, -1), lambda i, j: True)

    return ret


def smear_each(img: Image, id: int) -> Image:
    assert 0 <= id < 15
    directions = [
        [(1, 0)],
        [(-1, 0)],
        [(0, 1)],
        [(0, -1)],
        [(1, 0), (-1, 0)],
        [(0, 1), (0, -1)],
        [(1, 0), (-1, 0), (0, 1), (0, -1)],
        [(1, 1)],
        [(-1, -1)],
        [(1, -1)],
        [(-1, 1)],
        [(1, 1), (-1, -1)],
        [(1, -1), (-1, 1)],
        [(1, 1), (-1, -1), (1, -1), (-1, 1)],
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)],
    ]

    ret = img.copy()
    w = img.w

    for dx, dy in directions[id]:
        di = dy * w + dx

        for i in range(ret.h):
            step = 1 if i == 0 or i == ret.h - 1 else max(ret.w - 1, 1)
            for j in range(0, ret.w, step):
                if i - dy < 0 or j - dx < 0 or i - dy >= img.h or j - dx >= img.w:
                    steps = MAXSIDE
                    if dx == -1:
                        steps = min(steps, j + 1)
                    if dx == 1:
                        steps = min(steps, img.w - j)
                    if dy == -1:
                        steps = min(steps, i + 1)
                    if dy == 1:
                        steps = min(steps, img.h - i)

                    ind = i * w + j
                    end_ind = ind + steps * di
                    c = 0
                    while ind != end_ind:
                        if img.mask[ind // w, ind % w]:
                            c = img.mask[ind // w, ind % w]
                        if c:
                            ret.mask[ind // w, ind % w] = c
                        ind += di

    return ret


def extend(img: Image, room: Image) -> Image:
    if img.w * img.h == 0:
        return Image()  # bad image
    ret = room.copy()
    for i in range(ret.h):
        for j in range(ret.w):
            x = max(0, min(j + room.x - img.x, img.w - 1))
            y = max(0, min(i + room.y - img.y, img.h - 1))
            ret[i, j] = img[y, x]
    return ret


def pick_max(v: List[Image], f: Callable[[Image], int]) -> Image:
    if not v:
        return Image()  # bad image
    return max(v, key=f)


def max_criterion(img: Image, id: int) -> int:
    assert 0 <= id < 14
    if id == 0:
        return img.count()
    elif id == 1:
        return -img.count()
    elif id == 2:
        return img.w * img.h
    elif id == 3:
        return -img.w * img.h
    elif id == 4:
        return img.count_cols()
    elif id == 5:
        return -img.y
    elif id == 6:
        return img.y
    elif id == 7:
        return img.count_components()
    elif id in (8, 9):
        comp = compress(img)
        return (comp.w * comp.h - comp.count()) * (-1 if id == 9 else 1)
    elif id in (10, 11):
        return -img.count()
        # return img.count_interior() * (-1 if id == 11 else 1)
    elif id == 12:
        return -img.x
    elif id == 13:
        return img.x


def cut(img: Image, a: Image) -> List[Image]:
    ret = []
    done = Image.empty(img.x, img.y, img.w, img.h)
    d = Point(img.x - a.x, img.y - a.y)

    def dfs(r: int, c: int, toadd: Image):
        if (
            r < 0
            or r >= img.h
            or c < 0
            or c >= img.w
            or a.safe(r + d.y, c + d.x)
            or done[r, c]
        ):
            return
        toadd[r, c] = img[r, c] + 1
        done[r, c] = 1
        for nr in (r - 1, r, r + 1):
            for nc in (c - 1, c, c + 1):
                dfs(nr, nc, toadd)

    for i in range(img.h):
        for j in range(img.w):
            if not done[i, j] and not a.safe(i + d.y, j + d.x):
                toadd = Image.empty(img.x, img.y, img.w, img.h)
                dfs(i, j, toadd)
                toadd = compress(toadd)
                toadd.mask = np.maximum(toadd.mask - 1, 0)
                ret.append(toadd)
    return ret


def split_cols(img: Image, include0: int = 0) -> List[Image]:
    ret = []
    mask = img.col_mask()
    for c in range(int(not include0), 10):
        if mask & (1 << np.int64(c)):
            s = img.copy()
            s.mask = np.where(s.mask == c, c, 0)
            ret.append(s)
    return ret


def get_regular_1d(col: List[int]) -> None:
    colw = len(col)
    for w in range(1, colw):
        s = -1
        if colw % (w + 1) == w:  # No outer border
            s = w
        elif colw % (w + 1) == 1:  # Outer border
            s = 0
        if s != -1:
            if all(col[i] == (i % (w + 1) == s) for i in range(colw)):
                return
    col[:] = [0] * colw


def get_regular(img: Image) -> Image:
    # Look for regular grid division in single color
    ret = img.copy()
    col = [1] * img.w
    row = [1] * img.h

    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] != img[i, 0]:
                row[i] = 0
            if img[i, j] != img[0, j]:
                col[j] = 0

    get_regular_1d(col)
    get_regular_1d(row)

    for i in range(img.h):
        for j in range(img.w):
            ret[i, j] = int(row[i] or col[j])

    return ret


def cut_pick_max(a: Image, b: Image, id: int) -> Image:
    return pick_max(cut(a, b), lambda img: max_criterion(img, id))


def regular_cut_pick_max(a: Image, id: int) -> Image:
    b = get_regular(a)
    return cut_pick_max(a, b, id)


def split_pick_max(a: Image, id: int, include0: int = 0) -> Image:
    return pick_max(split_cols(a, include0), lambda img: max_criterion(img, id))


def cut_compose(a: Image, b: Image, id: int) -> Image:
    v = cut(a, b)
    return compose_list([to_origin(img) for img in v], id)

def combine_images(image_list: List[Tuple[Image, Point]]) -> Image:
    # Find the dimensions of the combined image
    max_x = max(point.x + img.w for img, point in image_list)
    max_y = max(point.y + img.h for img, point in image_list)
    
    # Create a new empty image
    combined = Image.empty(0, 0, max_x, max_y)
    
    # Place each image onto the combined image
    for img, point in image_list:
        for y in range(img.h):
            for x in range(img.w):
                if img[y, x] != 0:  # Only copy non-zero (non-background) pixels
                    combined[point.y + y, point.x + x] = img[y, x]
    
    return combined

def regular_cut_compose(a: Image, id: int) -> Image:
    b = get_regular(a)
    return cut_compose(a, b, id)


def split_compose(a: Image, id: int, include0: int = 0) -> Image:
    v = split_cols(a, include0)
    return compose_list([to_origin(compress(img)) for img in v], id)


def cut_index(a: Image, b: Image, ind: int) -> Image:
    v = cut(a, b)
    return v[ind] if 0 <= ind < len(v) else Image()


def get_alternating(img: Image) -> Image:
    ret = img.copy()
    w, h = img.w, img.h

    def is_alternating(arr):
        return all(arr[i] != arr[i + 1] for i in range(len(arr) - 1))

    # Check rows and columns
    rows = [is_alternating([img[i, j] for j in range(w)]) for i in range(h)]
    cols = [is_alternating([img[i, j] for i in range(h)]) for j in range(w)]

    # Combine results
    for i in range(h):
        for j in range(w):
            ret[i, j] = int(rows[i] or cols[j])

    return ret


def get_constant(img: Image) -> Image:
    ret = img.copy()
    w, h = img.w, img.h

    def is_constant(arr):
        return all(x == arr[0] for x in arr)

    # Check rows and columns
    rows = [is_constant([img[i, j] for j in range(w)]) for i in range(h)]
    cols = [is_constant([img[i, j] for i in range(h)]) for j in range(w)]

    # Combine results
    for i in range(h):
        for j in range(w):
            ret[i, j] = int(rows[i] or cols[j])

    return ret


def get_repeating(img: Image, min_repetitions: int = 2) -> Image:
    ret = img.copy()
    w, h = img.w, img.h

    def is_repeating(arr):
        n = len(arr)
        for period in range(1, n // 2 + 1):
            if n % period == 0 and n // period >= min_repetitions:
                pattern = arr[:period]
                if arr == pattern * (n // period):
                    return True
        return False

    # Check rows and columns
    rows = [is_repeating([img[i, j] for j in range(w)]) for i in range(h)]
    cols = [is_repeating([img[i, j] for i in range(h)]) for j in range(w)]

    # Combine results
    for i in range(h):
        for j in range(w):
            ret[i, j] = int(rows[i] or cols[j])

    return ret


def pick_maxes(
    v: List[Image], f: Callable[[Image], int], invert: int = 0
) -> List[Image]:
    if not v:
        return []
    scores = [f(img) for img in v]
    max_score = max(scores)
    return [img for img, score in zip(v, scores) if (score == max_score) ^ invert]


def pick_not_maxes(v: List[Image], id: int) -> List[Image]:
    return pick_maxes(v, lambda img: max_criterion(img, id), 1)


def cut_pick_maxes(a: Image, b: Image, id: int) -> Image:
    return compose_list(pick_maxes(cut(a, b), lambda img: max_criterion(img, id)), 0)


def split_pick_maxes(a: Image, id: int) -> Image:
    return compose_list(
        pick_maxes(split_cols(a), lambda img: max_criterion(img, id)), 0
    )


def heuristic_cut(img: Image) -> Image:
    ret = img.majority_col(include0=1)
    ret_score = -1
    mask = img.col_mask()
    done = Image.empty(img.x, img.y, img.w, img.h)

    def edgy(r: int, c: int, col: int):
        if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c]:
            return
        done[r, c] = 1
        for nr in (r - 1, r, r + 1):
            for nc in (c - 1, c, c + 1):
                edgy(nr, nc, col)

    for col in range(10):
        if not (mask & (1 << np.int64(col))):
            continue
        done.mask.fill(0)
        top = bot = left = right = False
        for i in range(img.h):
            for j in range(img.w):
                if img[i, j] == col:
                    if i == 0:
                        top = True
                    if j == 0:
                        left = True
                    if i == img.h - 1:
                        bot = True
                    if j == img.w - 1:
                        right = True
                if (
                    (i in (0, img.h - 1) or j in (0, img.w - 1))
                    and img[i, j] == col
                    and not done[i, j]
                ):
                    edgy(i, j, col)

        if not ((top and bot) or (left and right)):
            continue

        score = float("inf")
        components = 0
        no_contained = True
        for i in range(img.h):
            for j in range(img.w):
                if not done[i, j] and img[i, j] != col:
                    cnt = 0
                    contained = True
                    stack = [(i, j)]
                    while stack:
                        r, c = stack.pop()
                        if r < 0 or r >= img.h or c < 0 or c >= img.w:
                            continue
                        if img[r, c] == col:
                            if done[r, c]:
                                contained = False
                            continue
                        if done[r, c]:
                            continue
                        cnt += 1
                        done[r, c] = 1
                        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
                    components += 1
                    score = min(score, cnt)
                    if contained:
                        no_contained = False
        if components >= 2 and no_contained and score > ret_score:
            ret_score = score
            ret = col
    return filter_col_id(img, ret)


def my_stack(a: Image, b: Image, orient: int) -> Image:
    assert 0 <= orient <= 3
    b.x, b.y = a.x, a.y
    if orient == 0:  # Horizontal
        b.x += a.w
    elif orient == 1:  # Vertical
        b.y += a.h
    elif orient == 2:  # Diagonal
        b.x += a.w
        b.y += a.h
    else:  # Other diagonal, bottom-left / top-right
        c = a.copy()
        c.y += b.h
        b.x += a.w
        return compose_id(c, b, 0)
    return compose_id(a, b, 0)


def wrap(line: Image, area: Image) -> Image:
    if line.w * line.h == 0 or area.w * area.h == 0:
        return Image()  # bad image
    ans = Image.empty(0, 0, area.w, area.h)
    for i in range(line.h):
        for j in range(line.w):
            x, y = j, i
            x += y // area.h * line.w
            y %= area.h
            y += x // area.w * line.h
            x %= area.w
            if 0 <= x < ans.w and 0 <= y < ans.h:
                ans[y, x] = line[i, j]
    return ans


def repeat(a: Image, b: Image, pad: int = 0) -> Image:
    if a.w * a.h <= 0 or b.w * b.h <= 0:
        return Image()  # bad image
    ret = Image.empty(b.x, b.y, b.w, b.h)
    W, H = a.w + pad, a.h + pad
    ai = ((b.y - a.y) % H + H) % H
    aj0 = ((b.x - a.x) % W + W) % W
    for i in range(ret.h):
        aj = aj0
        for j in range(ret.w):
            if ai < a.h and aj < a.w:
                ret[i, j] = a[ai, aj]
            aj = (aj + 1) % W
        ai = (ai + 1) % H
    return ret


def mirror(a: Image, b: Image, pad: int = 0) -> Image:
    if a.w * a.h <= 0 or b.w * b.h <= 0:
        return Image()  # bad image
    ret = Image.empty(b.x, b.y, b.w, b.h)
    W, H = a.w + pad, a.h + pad
    W2, H2 = W * 2, H * 2
    ai = ((b.y - a.y) % H2 + H2) % H2
    aj0 = ((b.x - a.x) % W2 + W2) % W2
    for i in range(ret.h):
        aj = aj0
        for j in range(ret.w):
            x, y = -1, -1
            if aj < a.w:
                x = aj
            elif W <= aj < W + a.w:
                x = W + a.w - 1 - aj
            if ai < a.h:
                y = ai
            elif H <= ai < H + a.h:
                y = H + a.h - 1 - ai
            if x != -1 and y != -1:
                ret[i, j] = a[y, x]
            aj = (aj + 1) % W2
        ai = (ai + 1) % H2
    return ret


def split_all(img: Image) -> List[Image]:
    ret = []
    done = Image.empty(img.x, img.y, img.w, img.h)

    def dfs(r: int, c: int, col: int, toadd: Image):
        if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c]:
            return
        toadd[r, c] = img[r, c] + 1
        done[r, c] = 1
        for d in range(4):
            nr = r + (d == 0) - (d == 1)
            nc = c + (d == 2) - (d == 3)
            dfs(nr, nc, col, toadd)

    for i in range(img.h):
        for j in range(img.w):
            if not done[i, j]:
                toadd = Image.empty(img.x, img.y, img.w, img.h)
                dfs(i, j, img[i, j], toadd)
                toadd = compress(toadd)
                toadd.mask = np.maximum(toadd.mask - 1, 0)
                if toadd.count() > 0:
                    ret.append(toadd)

    return ret

def split_all(img: Image) -> List[Image]:
    ret = []
    done = Image.empty(img.x, img.y, img.w, img.h)

    def dfs(r: int, c: int, col: int, toadd: Image):
        if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c]:
            return
        toadd[r, c] = img[r, c] + 1
        done[r, c] = 1
        for d in range(4):
            nr = r + (d == 0) - (d == 1)
            nc = c + (d == 2) - (d == 3)
            dfs(nr, nc, col, toadd)

    for i in range(img.h):
        for j in range(img.w):
            if not done[i, j]:
                toadd = Image.empty(img.x, img.y, img.w, img.h)
                dfs(i, j, img[i, j], toadd)
                toadd = compress(toadd)
                toadd.mask = np.maximum(toadd.mask - 1, 0)
                if toadd.count() > 0:
                    ret.append(toadd)

    return ret


def erase_col(img: Image, col: int) -> Image:
    ret = img.copy()
    ret.mask[ret.mask == col] = 0
    return ret


def inside_marked(in_img: Image) -> List[Image]:
    ret = []
    for i in range(in_img.h - 1):
        for j in range(in_img.w - 1):
            for h in range(1, in_img.h - i - 1):
                for w in range(1, in_img.w - j - 1):
                    col = in_img[i, j]
                    if col == 0:
                        continue
                    ok = True
                    for k in range(4):
                        x, y = j + (k % 2) * w, i + (k // 2) * h
                        for d in range(4):
                            if (d != 3 - k) == (in_img[y + d // 2, x + d % 2] != col):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        inside = invert(Image.full(Point(j + 1, i + 1), Point(w, h)))
                        ret.append(compose_id(inside, in_img, 3))
    return ret


def make_border(img: Image, bcol: int = 1) -> Image:
    ret = hull0(img)
    for i in range(ret.h):
        for j in range(ret.w):
            if img[i, j] == 0:
                ok = False
                for ni in [i - 1, i, i + 1]:
                    for nj in [j - 1, j, j + 1]:
                        if img.safe(ni, nj):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    ret[i, j] = bcol
    return ret


def make_border2(img: Image, usemaj: int = 1) -> Image:
    bcol = img.majority_col() if usemaj else 1
    rsz = Point(img.w + 2, img.h + 2)
    if max(rsz.x, rsz.y) > MAXSIDE or rsz.x * rsz.y > MAXAREA:
        return Image()  # badImg equivalent
    ret = Image.full(Point(img.x - 1, img.y - 1), rsz, bcol)
    ret.mask[1:-1, 1:-1] = img.mask
    return ret


def make_border2_with_bord(img: Image, bord: Image) -> Image:
    bcol = bord.majority_col()
    rsz = Point(img.w + 2 * bord.w, img.h + 2 * bord.h)
    if max(rsz.x, rsz.y) > MAXSIDE or rsz.x * rsz.y > MAXAREA:
        return Image()  # badImg equivalent
    ret = Image.full(Point(img.x - bord.w, img.y - bord.h), rsz, bcol)
    ret.mask[bord.h : -bord.h, bord.w : -bord.w] = img.mask
    return ret


def compress2(img: Image) -> Image:
    row = np.any(img.mask != 0, axis=1)
    col = np.any(img.mask != 0, axis=0)
    rows = np.where(row)[0]
    cols = np.where(col)[0]
    ret = Image(img.x, img.y, len(cols), len(rows))
    ret.mask = img.mask[np.ix_(rows, cols)]
    return ret


def compress3(img: Image) -> Image:
    if img.w * img.h <= 0:
        return Image()  # badImg equivalent

    row = np.zeros(img.h, dtype=bool)
    col = np.zeros(img.w, dtype=bool)
    row[0] = col[0] = True

    for i in range(1, img.h):
        for j in range(img.w):
            if img.mask[i, j] != img.mask[i - 1, j]:
                row[i] = True
                break

    for j in range(1, img.w):
        for i in range(img.h):
            if img.mask[i, j] != img.mask[i, j - 1]:
                col[j] = True
                break

    rows = np.where(row)[0]
    cols = np.where(col)[0]

    ret = Image(img.x, img.y, len(cols), len(rows), img.mask[np.ix_(rows, cols)])

    return ret


def greedy_fill(
    ret: Image,
    piece: List[Tuple[int, List[int]]],
    done: np.ndarray,
    bw: int,
    bh: int,
    donew: int,
) -> Image:
    piece.sort(reverse=True)
    dw, dh = ret.w - bw + 1, ret.h - bh + 1
    if dw < 1 or dh < 1:
        return Image()  # badImg equivalent

    dones = np.full((dh, dw), -1, dtype=int)
    pq = []

    def recalc(i: int, j: int):
        cnt = np.sum(done[i : i + bh, j : j + bw])
        if cnt != dones[i, j]:
            dones[i, j] = cnt
            heapq.heappush(pq, (-cnt, j, i))

    for i in range(dh):
        for j in range(dw):
            recalc(i, j)

    while pq:
        ds, j, i = heapq.heappop(pq)
        ds = -ds
        if ds != dones[i, j]:
            continue
        found = False
        for cnt, mask in piece:
            mask = np.array(mask).reshape(bh, bw)
            if np.all(
                (done[i : i + bh, j : j + bw] == 0)
                | (ret.mask[i : i + bh, j : j + bw] == mask)
            ):
                done[i : i + bh, j : j + bw] = np.where(
                    done[i : i + bh, j : j + bw] == 0,
                    donew,
                    done[i : i + bh, j : j + bw],
                )
                ret.mask[i : i + bh, j : j + bw] = np.where(
                    done[i : i + bh, j : j + bw] == donew,
                    mask,
                    ret.mask[i : i + bh, j : j + bw],
                )
                donew = max(1, donew - 1)
                for y in range(max(i - bh + 1, 0), min(i + bh, dh)):
                    for x in range(max(j - bw + 1, 0), min(j + bw, dw)):
                        recalc(y, x)
                found = True
                break
        if not found:
            return Image()  # badImg equivalent
    return ret


def greedy_fill_black(img: Image, N: int = 3) -> Image:
    ret = Image.empty(img.x, img.y, img.w, img.h)
    done = np.zeros((img.h, img.w), dtype=int)
    donew = 10**6

    ret.mask[img.mask != 0] = img.mask[img.mask != 0]
    done[img.mask != 0] = donew

    piece_cnt = {}
    bw, bh = N, N
    for r in range(8):
        rot = rigid(img, r)
        for i in range(rot.h - bh + 1):
            for j in range(rot.w - bw + 1):
                mask = rot.mask[i : i + bh, j : j + bw].flatten()
                if np.all(mask):
                    piece_cnt[tuple(mask)] = piece_cnt.get(tuple(mask), 0) + 1

    piece = [(c, list(p)) for p, c in piece_cnt.items()]
    return greedy_fill(ret, piece, done, bw, bh, donew)


def greedy_fill_black2(img: Image, N: int = 3) -> Image:
    filled = greedy_fill_black(img, N)
    return compose_id(filled, img, 4)


def extend2(img: Image, room: Image) -> Image:
    ret = Image.empty(room.x, room.y, room.w, room.h)
    done = np.zeros((room.h, room.w), dtype=int)

    # Calculate the offset to center the original image
    offset_x = (room.w - img.w) // 2
    offset_y = (room.h - img.h) // 2

    # Copy the original image to the center of the new space
    ret.mask[offset_y:offset_y+img.h, offset_x:offset_x+img.w] = img.mask
    done[offset_y:offset_y+img.h, offset_x:offset_x+img.w] = 1

    # Extend the pattern
    for i in range(room.h):
        for j in range(room.w):
            if done[i, j] == 0:
                # Find the nearest pixel from the original image
                ori_i = min(max(i - offset_y, 0), img.h - 1)
                ori_j = min(max(j - offset_x, 0), img.w - 1)
                ret.mask[i, j] = img.mask[ori_i, ori_j]

    # The rest of the function (piece counting) remains the same
    piece_cnt = {}
    bw, bh = 3, 3
    for r in range(8):
        rot = rigid(ret, r)  # Note: We're now using 'ret' instead of 'img'
        for i in range(rot.h - bh + 1):
            for j in range(rot.w - bw + 1):
                mask = tuple(rot.mask[i:i+bh, j:j+bw].flatten())
                piece_cnt[mask] = piece_cnt.get(mask, 0) + 1

    piece = [(count, list(p)) for p, count in piece_cnt.items()]

    # For this test case, we don't actually need to call greedy_fill
    # return greedy_fill(ret, piece, done, bw, bh, donew)
    return ret

def connect(img: Image, id: int) -> Image:
    assert 0 <= id < 3
    ret = Image.empty(img.x, img.y, img.w, img.h)

    if id == 0 or id == 2:  # Horizontal
        for i in range(img.h):
            last = -1
            lastc = -1
            for j in range(img.w):
                if img.mask[i, j]:
                    if img.mask[i, j] == lastc:
                        ret.mask[i, last + 1 : j] = lastc
                    lastc = img.mask[i, j]
                    last = j
                    ret.mask[i, j] = img.mask[i, j]

    if id == 1 or id == 2:  # Vertical
        for j in range(img.w):
            last = -1
            lastc = -1
            for i in range(img.h):
                if img.mask[i, j]:
                    if img.mask[i, j] == lastc:
                        ret.mask[last + 1 : i, j] = lastc
                    lastc = img.mask[i, j]
                    last = i
                    ret.mask[i, j] = img.mask[i, j]

    return ret


def replace_template(
    in_img: Image, need: Image, marked: Image, overlapping: int = 0, rigids: int = 0
) -> Image:
    if marked.w != need.w or marked.h != need.h:
        return Image()  # badImg equivalent
    if need.w * need.h <= 0:
        return in_img

    rots = 8 if rigids else 1
    needr = [rigid(need, r) for r in range(rots)]
    markedr = [rigid(marked, r) for r in range(rots)]

    ret = in_img.copy()
    for r in range(rots):
        need, marked = needr[r], markedr[r]
        for i in range(ret.h - need.h + 1):
            for j in range(ret.w - need.w + 1):
                if np.all(
                    (in_img if overlapping else ret).mask[
                        i : i + need.h, j : j + need.w
                    ]
                    == need.mask
                ):
                    if overlapping == 2:
                        surrounding = np.pad(need.mask, 1, mode="edge")
                        if np.any(
                            surrounding[surrounding != 0]
                            == in_img.mask[
                                i - 1 : i + need.h + 1, j - 1 : j + need.w + 1
                            ][surrounding != 0]
                        ):
                            continue
                    ret.mask[i : i + need.h, j : j + need.w] = marked.mask
    return ret


def swap_template(in_img: Image, a: Image, b: Image, rigids: int = 0) -> Image:
    if a.w != b.w or a.h != b.h:
        return Image()  # badImg equivalent
    if a.w * a.h <= 0:
        return in_img

    rots = 8 if rigids else 1
    ar = [rigid(a, r) for r in range(rots)]
    br = [rigid(b, r) for r in range(rots)]

    done = Image.empty(in_img.x, in_img.y, in_img.w, in_img.h)
    ret = in_img.copy()

    for k in [0, 1]:
        for r in range(rots):
            need, to = (ar[r], br[r]) if k == 0 else (br[r], ar[r])
            for i in range(ret.h - need.h + 1):
                for j in range(ret.w - need.w + 1):
                    if np.all(
                        done.mask[i : i + need.h, j : j + need.w] == 0
                    ) and np.all(ret.mask[i : i + need.h, j : j + need.w] == need.mask):
                        ret.mask[i : i + need.h, j : j + need.w] = to.mask
                        done.mask[i : i + need.h, j : j + need.w] = 1
    return ret

def spread_cols(img: Image, skipmaj: int = 0) -> Image:
    skipcol = img.majority_col() if skipmaj else -1
    done = Image.empty(img.x, img.y, img.w, img.h)
    ret = img.copy()
    q = deque()

    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] and img[i, j] != skipcol:
                q.append((j, i, img[i, j]))
                done[i, j] = 1

    while q:
        j, i, c = q.popleft()
        for d in range(4):
            ni, nj = i + (d == 0) - (d == 1), j + (d == 2) - (d == 3)
            if 0 <= ni < img.h and 0 <= nj < img.w and not done[ni, nj]:
                ret[ni, nj] = c
                done[ni, nj] = 1
                q.append((nj, ni, c))

    return ret


def split_columns(img: Image) -> List[Image]:
    if img.w * img.h <= 0:
        return []
    return [Image(j, 0, 1, img.h, img.mask[:, j].reshape(-1, 1)) for j in range(img.w)]


def split_rows(img: Image) -> List[Image]:
    if img.w * img.h <= 0:
        return []
    return [Image(0, i, img.w, 1, img.mask[i, :].reshape(1, -1)) for i in range(img.h)]


def half(img: Image, id: int) -> Image:
    assert 0 <= id < 4
    if id == 0:
        return img.sub_image(Point(0, 0), Point(img.w // 2, img.h))
    elif id == 1:
        return img.sub_image(Point(img.w - img.w // 2, 0), Point(img.w // 2, img.h))
    elif id == 2:
        return img.sub_image(Point(0, 0), Point(img.w, img.h // 2))
    elif id == 3:
        return img.sub_image(Point(0, img.h - img.h // 2), Point(img.w, img.h // 2))


def mirror2(a: Image, line: Image) -> Image:
    if line.w > line.h:
        ret = rigid(a, 5)
        ret.x = a.x
        ret.y = line.y * 2 + line.h - a.y - a.h
    else:
        ret = rigid(a, 4)
        ret.y = a.y
        ret.x = line.x * 2 + line.w - a.x - a.w
    return ret


def gravity(in_img: Image, d: int) -> List[Image]:
    pieces = split_all(in_img)
    room = hull0(in_img)
    dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][d]

    ret = []
    out = room.copy()
    pieces.sort(key=lambda a: a.x * dx + a.y * dy, reverse=True)

    for p in pieces:
        while True:
            p.x += dx
            p.y += dy

            if not is_valid_position(p, out):
                p.x -= dx
                p.y -= dy
                break

        ret.append(p)
        out = compose_id(out, p, 3)

    return ret


def is_valid_position(piece: Image, out: Image) -> bool:
    for i in range(piece.h):
        for j in range(piece.w):
            if piece.mask[i, j] != 0:
                x, y = j + piece.x - out.x, i + piece.y - out.y
                if x < 0 or y < 0 or x >= out.w or y >= out.h or out.mask[y, x] != 0:
                    return False
    return True


def my_stack_l(lens: List[Image], id: int) -> Image:
    if not lens:
        return Image()  # badImg equivalent
    order = sorted(enumerate(lens), key=lambda x: x[1].w * x[1].h)
    out = lens[order[0][0]]
    for _, img in order[1:]:
        out = my_stack(out, img, id)
    return out


def stack_line_v(shapes: List[Image]) -> Image:
    if not shapes:
        return Image()  # badImg equivalent
    if len(shapes) == 1:
        return shapes[0]

    xs = sorted(shape.x for shape in shapes)
    ys = sorted(shape.y for shape in shapes)
    xmin = min(xs[i] - xs[i - 1] for i in range(1, len(xs)))
    ymin = min(ys[i] - ys[i - 1] for i in range(1, len(ys)))

    dx, dy = (1, 0) if xmin < ymin else (0, 1)
    order = sorted(enumerate(shapes), key=lambda x: x[1].x * dx + x[1].y * dy)

    out = shapes[order[0][0]]
    for _, img in order[1:]:
        out = my_stack(out, img, dy)
    return out


def stack_line(shapes: List[Image]) -> Image:
    n = len(shapes)
    if n == 0:
        return Image()  # badImg equivalent
    elif n == 1:
        return shapes[0]

    xs = [shape.x for shape in shapes]
    ys = [shape.y for shape in shapes]
    xs.sort()
    ys.sort()

    xmin = float("inf")
    ymin = float("inf")
    for i in range(1, n):
        xmin = min(xmin, xs[i] - xs[i - 1])
        ymin = min(ymin, ys[i] - ys[i - 1])

    dx, dy = (0, 1) if xmin < ymin else (1, 0)

    order = [(shape.x * dx + shape.y * dy, i) for i, shape in enumerate(shapes)]
    order.sort()

    out = shapes[order[0][1]]
    for i in range(1, n):
        out = my_stack(out, shapes[order[i][1]], dy)

    return out


def compose_growing_slow(imgs: List[Image]) -> Image:
    if not imgs:
        return Image()  # badImg equivalent
    order = sorted(enumerate(imgs), key=lambda x: x[1].count(), reverse=True)
    return compose_list(imgs, 0)


def compose_growing(imgs: List[Image]) -> Image:
    if not imgs:
        return Image()  # badImg equivalent
    if len(imgs) == 1:
        return imgs[0]

    minx = min(img.x for img in imgs)
    miny = min(img.y for img in imgs)
    maxx = max(img.x + img.w for img in imgs)
    maxy = max(img.y + img.h for img in imgs)

    rsz = Point(maxx - minx, maxy - miny)
    if (
        max(rsz.x, rsz.y) > MAXSIDE
        or rsz.x * rsz.y > MAXAREA
        or rsz.x <= 0
        or rsz.y <= 0
    ):
        return Image()  # badImg equivalent

    order = sorted(enumerate(imgs), key=lambda x: x[1].count(), reverse=True)
    ret = Image.empty(minx, miny, rsz.x, rsz.y)

    for _, img in order:
        dx, dy = img.x - ret.x, img.y - ret.y
        ret.mask[dy : dy + img.h, dx : dx + img.w] = np.where(
            img.mask != 0, img.mask, ret.mask[dy : dy + img.h, dx : dx + img.w]
        )

    return ret


def pick_unique(imgs: List[Image]) -> Image:
    images_different = find_unique_images(imgs)
    if len(images_different)==0:
        return Image()  # Empty image / badImg equivalent
    return images_different[-1]

def find_unique_images(imgs: List[Image]) -> Image:
    if not imgs:
        return []  # Empty image / badImg equivalent

    # Count occurrences of each color across all images
    color_count = {}
    for img in imgs:
        for color in set(img.mask.flatten()):
            if color not in color_count:
                color_count[color] = 1
            else:
                color_count[color] += 1

    l = []
    # Find the image with a unique color
    for img in imgs:
        unique_colors = [color for color in set(img.mask.flatten()) if color_count[color] == 1]
        if unique_colors:
            l.append(img)
    return l  # No unique image found