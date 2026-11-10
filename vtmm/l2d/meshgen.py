"""Mesh generation and texture atlas packing for layer images."""
from __future__ import annotations

import cv2
import numpy as np


def grid_mesh(alpha: np.ndarray, x0: int, y0: int, cell: int, pad: int = 6, thr: int = 8, canvas_diag: bool = False):
    """Regular triangle grid covering every cell that touches (dilated) alpha.

    alpha: (h,w) uint8 of the layer crop located at canvas (x0,y0).
    canvas_diag: which way a cell is cut in two depends on the cell's place on the canvas, not on where the
    layer's box happens to begin - a layer that loses a few pixels at its edge keeps the mesh it had.
    Returns verts (N,2) float canvas px, tris (M,3) int.
    """
    h, w = alpha.shape
    # grid anchored on canvas multiples of `cell` so neighbouring layers share vertex lattices
    gx0 = (x0 - pad) // cell * cell
    gy0 = (y0 - pad) // cell * cell
    nx = int(np.ceil((x0 + w + pad - gx0) / cell))
    ny = int(np.ceil((y0 + h + pad - gy0) / cell))
    win = np.zeros((ny * cell, nx * cell), np.uint8)
    ox, oy = x0 - gx0, y0 - gy0
    win[oy:oy + h, ox:ox + w] = (alpha > thr)
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        win = cv2.dilate(win, k)
    occ = win.reshape(ny, cell, nx, cell).max(axis=(1, 3)) > 0
    vid = -np.ones((ny + 1, nx + 1), np.int64)
    verts, tris = [], []
    base = (gx0 // cell + gy0 // cell + 1) if canvas_diag else 0

    def v(j, i):
        if vid[j, i] < 0:
            vid[j, i] = len(verts)
            verts.append((gx0 + i * cell, gy0 + j * cell))
        return vid[j, i]

    for j in range(ny):
        for i in range(nx):
            if not occ[j, i]:
                continue
            a, b, c, d = v(j, i), v(j, i + 1), v(j + 1, i), v(j + 1, i + 1)
            if (i + j + base) % 2 == 0:
                tris += [(a, b, c), (b, d, c)]
            else:
                tris += [(a, b, d), (a, d, c)]
    return np.array(verts, np.float64).reshape(-1, 2), np.array(tris, np.int64).reshape(-1, 3)


def strip_mesh(centerline: np.ndarray, half_width, n_across: int = 1):
    """Ribbon mesh along a polyline (for lash / mouth lines). Returns verts, tris, (s,t) params."""
    pts = np.asarray(centerline, np.float64)
    n = len(pts)
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    hw = np.broadcast_to(np.asarray(half_width, np.float64), (n,))
    rows = 2 * n_across + 1
    ts = np.linspace(-1, 1, rows)
    verts = np.stack([pts + normal * (hw * t)[:, None] for t in ts], 1).reshape(-1, 2)
    st = np.stack(np.meshgrid(np.linspace(0, 1, n), ts, indexing="ij"), -1).reshape(-1, 2)
    tris = []
    for i in range(n - 1):
        for r in range(rows - 1):
            a = i * rows + r
            b = a + 1
            c = a + rows
            d = c + 1
            tris += [(a, c, b), (b, c, d)]
    return verts, np.array(tris, np.int64), st


# ----------------------------------------------------------------------------------------
class Atlas:
    """Shelf packer. Each item is an RGBA image; returns page index + pixel rect per item."""

    def __init__(self, size: int = 4096, gap: int = 4):
        self.size, self.gap = size, gap
        self.pages = []   # list of dict(shelves=[(y, h, x_cursor)], used_h)
        self.items = {}

    def pack(self, items: dict):
        """items: name -> (w, h). Returns name -> (page, x, y)."""
        order = sorted(items, key=lambda n: (-items[n][1], -items[n][0]))
        place = {}
        for name in order:
            w, h = items[name]
            assert w + 2 * self.gap <= self.size and h + 2 * self.gap <= self.size, (name, w, h)
            done = False
            for pi, page in enumerate(self.pages):
                pos = self._fit(page, w, h)
                if pos:
                    place[name] = (pi, *pos)
                    done = True
                    break
            if not done:
                self.pages.append({"shelves": [], "y": self.gap})
                pos = self._fit(self.pages[-1], w, h)
                place[name] = (len(self.pages) - 1, *pos)
        return place

    def _fit(self, page, w, h):
        g = self.gap
        for sh in page["shelves"]:
            if h <= sh["h"] and sh["x"] + w + g <= self.size:
                x = sh["x"]
                sh["x"] += w + g
                return x, sh["y"]
        if page["y"] + h + g <= self.size:
            sh = {"y": page["y"], "h": h, "x": g + w + g}
            page["shelves"].append(sh)
            page["y"] += h + g
            return g, sh["y"]
        return None


class NestAtlas:
    """Packs shapes, not boxes.

    The shelf packer above gives every layer the whole of its bounding box: a coat tail that runs slantwise across
    a 1672 x 3960 box takes all of it though it fills a quarter, and thirteen pages came out 41 % full. Here an item
    is a mask on a coarse grid - the cells its meshes' triangles reach, grown by a margin -, items go into each
    other's empty corners, and each may be turned by quarter turns (texel for texel, nothing is resampled). An
    item is put where it touches most of what is already there (the page's edge counts), biggest first; a new page
    is opened only when it fits nowhere."""

    def __init__(self, size: int = 4096, grid: int = 16):
        assert size % grid == 0
        self.size, self.grid, self.n = size, grid, size // grid
        self.pages = []                                   # occupancy, one float32 grid per page

    def pack(self, masks: dict, tries: int = 8):
        """masks: name -> 2D array on the grid (non-zero = the item needs that cell).
        Returns name -> (page, x px, y px, quarter turns as for np.rot90).
        Putting the biggest first and each where it touches most is a rule of thumb, and one item more or less
        can cost a page: the items are packed in several orders (by cells taken, by box, by longest side, and a
        few with neighbours in the order swapped), and the atlas with the fewest pages - and of those the one
        whose last page is emptiest - is kept."""
        m = {n: (np.asarray(v) > 0).astype(np.float32) for n, v in masks.items()}
        names = sorted(m)
        orders = [sorted(names, key=lambda n: (-float(m[n].sum()), n)),
                  sorted(names, key=lambda n: (-m[n].size, n)),
                  sorted(names, key=lambda n: (-max(m[n].shape), -float(m[n].sum()), n))]
        rng = np.random.default_rng(1)
        while len(orders) < tries:
            keys = np.arange(len(names)) + rng.normal(0.0, 4.0, len(names))
            orders.append([orders[0][i] for i in np.argsort(keys)])
        best = None
        for order in orders[:max(tries, 1)]:
            pages, place = self._pack_once(m, order)
            score = (len(pages), float(pages[-1].sum()))
            if best is None or score < best[0]:
                best = (score, pages, place)
        self.pages = best[1]
        return best[2]

    def _pack_once(self, m, order):
        k3 = np.ones((3, 3), np.float32)
        pages, place = [], {}
        for name in order:
            m0 = m[name]
            best = None
            for pi in range(len(pages) + 1):
                if pi == len(pages):
                    pages.append(np.zeros((self.n, self.n), np.float32))
                occ = pages[pi]
                occ_b = np.pad(occ, 1, constant_values=1.0)
                for k in range(4):
                    mk = np.ascontiguousarray(np.rot90(m0, k))
                    if mk.shape[0] > self.n or mk.shape[1] > self.n:
                        continue
                    free = cv2.matchTemplate(occ, mk, cv2.TM_CCORR) < 0.5
                    if not free.any():
                        continue
                    mp = np.pad(mk, 1)
                    ring = np.ascontiguousarray(cv2.dilate(mp, k3) - mp)
                    touch = cv2.matchTemplate(occ_b, ring, cv2.TM_CCORR)
                    ys, xs = np.nonzero(free)
                    sc = -touch[ys, xs] + 0.01 * (ys + 0.001 * xs)
                    i = int(np.argmin(sc))
                    if best is None or sc[i] < best[0]:
                        best = (float(sc[i]), pi, k, int(xs[i]), int(ys[i]))
                if best is not None:
                    break
            assert best is not None, (name, m0.shape)
            _, pi, k, x, y = best
            mk = np.rot90(m0, k)
            sub = pages[pi][y:y + mk.shape[0], x:x + mk.shape[1]]
            np.maximum(sub, mk, out=sub)
            place[name] = (pi, x * self.grid, y * self.grid, k)
        return [p for p in pages if p.any()], place


def bleed_far(rgba: np.ndarray, reach: int, done: int = 0) -> np.ndarray:
    """Colour for the see-through pixels further out: every pixel with alpha 0 that lies more than `done` px (as
    `bleed` counts, in steps of a 3 x 3 neighbourhood) and at most `reach` px from the picture gets the colour of
    the picture's nearest pixel. (When the picture is shown small, the graphics card reads it from reduced copies
    in which each texel is the mean of 4, 16, 64 ... texels: without colour out there an edge is mixed with black.)"""
    alpha0 = rgba[..., 3] == 0
    if not alpha0.any() or alpha0.all():
        return rgba
    src = alpha0.astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    cheb = cv2.distanceTransform(src, cv2.DIST_C, 3)
    todo = alpha0 & (cheb > done) & (dist <= reach)
    if not todo.any():
        return rgba
    flat = rgba.reshape(-1, 4)
    seeds = np.flatnonzero(~alpha0.ravel())
    lut = np.zeros(int(lab.max()) + 1, np.int64)
    lut[lab.ravel()[seeds]] = seeds
    out = rgba.copy()
    t = np.flatnonzero(todo.ravel())
    out.reshape(-1, 4)[t, :3] = flat[lut[lab.ravel()[t]], :3]
    return out


def bleed(rgba: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Push colours of opaque pixels into neighbouring fully transparent pixels (RGB only),
    so bilinear sampling at the alpha edge never mixes in an arbitrary colour."""
    out = rgba.copy()
    rgb = out[..., :3].astype(np.float32)
    known = (out[..., 3] > 0).astype(np.float32)
    k = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        if known.min() > 0:
            break
        s = cv2.filter2D(rgb * known[..., None], -1, k, borderType=cv2.BORDER_REPLICATE)
        c = cv2.filter2D(known, -1, k, borderType=cv2.BORDER_REPLICATE)
        new = (known == 0) & (c > 0)
        rgb[new] = s[new] / c[new][:, None]
        known[new] = 1
    out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return out
