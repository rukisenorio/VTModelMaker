"""The mouth's line as a ribbon: its geometry, shared by the rig (build_model) and by the stage that makes its
texture (tools/mouth_parts.py). (The first character's tools/mouth_rig.py, 2026-10-20: chara2's mouth had been eleven
drawings swapped by how open and how smiling it is - 「口の動きが不自然」. Here, as there, one line bent into every shape.)

The ribbon runs along the drawn line from the centre of one rounded end to the centre of the
other, with one more column past each end for the outer half of the end. At rest it lies on the
artwork exactly as the texture was taken from it, so the closed mouth is the artwork's own line;
in every other mouth shape the same ribbon is bent.
"""
import json, os
import numpy as np

ROOT = os.environ.get("L2D_PROJECT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = 21                      # points along the line
HW = 8.0                     # half-width of the ribbon at rest (px)
TEX_W, TEX_H = 160, 22       # the texture (1x)
U0, U1 = 8.0, TEX_W - 9.0    # texture columns of the two end points
CAPU = 5.0                   # texture columns past an end point


def grid_tris(n, rows):
    t = []
    for i in range(n - 1):
        for r_ in range(rows - 1):
            a0 = i * rows + r_
            t += [(a0, a0 + rows, a0 + 1), (a0 + 1, a0 + rows, a0 + rows + 1)]
    return t


class Geom:
    def __init__(self, mg=None):
        mg = mg or json.load(open(os.path.join(ROOT, "work", "mouth_line.json")))
        self.mg = mg
        self.xa, self.xb = float(mg["xa"]), float(mg["xb"])
        self.cxm, self.w0 = 0.5 * (self.xa + self.xb), 0.5 * (self.xb - self.xa)
        self.ss = 0.5 * (1 - np.cos(np.pi * np.arange(NS) / (NS - 1)))        # denser near the corners
        self.x_rest = self.xa + self.ss * (self.xb - self.xa)
        self.y_rest = np.polyval(np.array(mg["pf"], float), self.x_rest - float(mg["xc"]))
        self.cl = self.y_rest[0] + (self.y_rest[-1] - self.y_rest[0]) * self.ss
        self.Dp = self.y_rest - self.cl                                       # how the drawn line dips
        self.cap = CAPU * (self.xb - self.xa) / (U1 - U0)                     # the same scale past the ends as between them
        self.ink = np.interp(self.x_rest, np.array(mg["ink_x"], float), np.array(mg["ink"], float))   # the line's thickness at each point

    def strip(self, X, Y, hw, capf=1.0):
        """Ribbon along a curve with half-width `hw` (one value or one per point): 3 points across.
        capf: how far the ribbon runs on past its end points (1: as the line was drawn)."""
        pts = np.stack([X, Y], 1)
        tg = np.gradient(pts, axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-6)
        pts = np.concatenate([pts[:1] - tg[:1] * self.cap * capf, pts, pts[-1:] + tg[-1:] * self.cap * capf])
        tg = np.concatenate([tg[:1], tg, tg[-1:]])
        nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
        hw = np.broadcast_to(np.asarray(hw, float), (NS,))
        hwv = np.concatenate([hw[:1], hw, hw[-1:]])[:, None]
        return np.stack([pts - nrm * hwv, pts, pts + nrm * hwv], 1).reshape(-1, 2)

    def rest(self):
        return self.strip(self.x_rest, self.y_rest, HW)

    def uv(self):
        """Texture position (px, pixel centres at .5) of each ribbon point."""
        u = np.concatenate([[U0 - CAPU], U0 + self.ss * (U1 - U0), [U1 + CAPU]])
        v = (TEX_H - 1) / 2.0 + np.array([-HW, 0.0, HW])
        return np.stack([np.repeat(u, 3) + 0.5, np.tile(v, NS + 2) + 0.5], 1)

    def tris(self):
        return grid_tris(NS + 2, 3)

    def texture_map(self, k=1):
        """Where on the canvas each texel of the (k times) texture lies at rest: (map_x, map_y, covered)."""
        uv, pos = self.uv() * k, self.rest()
        H, W = TEX_H * k, TEX_W * k
        mx = np.zeros((H, W), np.float32); my = np.zeros((H, W), np.float32); ok = np.zeros((H, W), bool)
        for tri in self.tris():
            a, b, c = uv[list(tri)]
            x0, x1 = int(np.floor(min(a[0], b[0], c[0]))) - 1, int(np.ceil(max(a[0], b[0], c[0]))) + 1
            y0, y1 = int(np.floor(min(a[1], b[1], c[1]))) - 1, int(np.ceil(max(a[1], b[1], c[1]))) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            v0, v1 = b - a, c - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-9:
                continue
            l1 = ((gx - a[0]) * v1[1] - v1[0] * (gy - a[1])) / den
            l2 = (v0[0] * (gy - a[1]) - (gx - a[0]) * v0[1]) / den
            ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l1 + l2 <= 1 + 1e-6)
            pa, pb, pc = pos[list(tri)]
            px = pa[0] + l1 * (pb[0] - pa[0]) + l2 * (pc[0] - pa[0])
            py = pa[1] + l1 * (pb[1] - pa[1]) + l2 * (pc[1] - pa[1])
            sub = (slice(y0, y1), slice(x0, x1))
            mx[sub][ins] = px[ins]; my[sub][ins] = py[ins]; ok[sub][ins] = True
        return mx, my, ok
