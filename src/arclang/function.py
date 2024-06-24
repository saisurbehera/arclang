import numpy as np
from collections import namedtuple
from arclang.image import Image, Point
from typing import List, Tuple, Callable

def col(id: int) -> Image:
    assert 0 <= id < 10
    return Image.full(Point(0, 0), Point(1, 1), id)

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

def move(img: Image, p: Image) -> Image:
    img.x += p.x
    img.y += p.y
    return img

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


def broadcast(col: 'Image', shape: 'Image', include0: int = 1) -> 'Image':
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
        bg = Image.full(Point(0, 0), Point(1, 1), 0)  # Use a full image with 0 as background

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

def embed(img: Image, shape: Image) -> Image:
    ret = Image(shape.x, shape.y, shape.w, shape.h)
    dx, dy = shape.x - img.x, shape.y - img.y
    sx, sy = max(0, -dx), max(0, -dy)
    ex, ey = min(ret.w, img.w - dx), min(ret.h, img.h - dy)

    ret_mask = ret.mask.reshape(ret.h, ret.w)
    img_mask = img.mask.reshape(img.h, img.w)
    ret_mask[sy:ey, sx:ex] = img_mask[sy+dy:ey+dy, sx+dx:ex+dx]
    return ret

def compose(a: Image, b: Image, f: Callable[[int, int], int], overlap_only: int) -> Image:
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
                    ret[i*b.h + k, j*b.w + l] = a[i, j] * (1 if b[k, l] else 0)
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
                    ret[i*b.h + k, j*b.w + l] = (1 if a[i, j] > 0 else 0) * b[k, l]
    return ret

def fill(a: Image) -> Image:
    # Create an image filled with the majority color of 'a'
    ret = Image.full(Point(a.x, a.y), Point(a.w, a.h), a.majority_col())
    q = []

    # Identify the border pixels and add them to the queue
    for i in range(a.h):
        for j in range(a.w):
            if (i == 0 or j == 0 or i == a.h-1 or j == a.w-1) and not a[i, j]:
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
            if i == 0 or j == 0 or i == a.h-1 or j == a.w-1:
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
        if r < 0 or r >= base.h or c < 0 or c >= base.w or base[r, c] != acol or done[r, c]:
            return []
        path = [(r, c)]
        done[r, c] = 1
        for nr in [r-1, r, r+1]:
            for nc in [c-1, c, c+1]:
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
    return Image(img.x + (img.w - sz_x) // 2, img.y + (img.h - sz_y) // 2, sz_x, sz_y)

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

    corners = [t(Point(0, 0)), t(Point(img.w-1, 0)), t(Point(0, img.h-1)), t(Point(img.w-1, img.h-1))]
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
        return img
    elif id == 1:
        return transform(img, 0, 1, -1, 0)  # CCW
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
    while col < 10 and not (mask & (1 << col)):
        col += 1
    if col == 10:
        col = 1

    ret = img.copy()
    ret.mask = np.where(ret.mask != 0, 0, col)
    return ret

def maj_col(img: Image) -> Image:
    return col(img.majority_col())