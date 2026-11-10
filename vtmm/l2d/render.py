"""Software renderer for a Core-evaluated model: draw order, clipping masks, opacity.

Not a performance renderer - it exists so generated models can be inspected off-line with
exactly the geometry the Cubism Core produces.
"""
from __future__ import annotations

import cv2
import numpy as np

SHIFT = 4  # sub-pixel bits for cv2.fillConvexPoly


def premultiply(rgba_u8: np.ndarray) -> np.ndarray:
    f = rgba_u8.astype(np.float32) / 255.0
    f[..., :3] *= f[..., 3:4]
    return f


class Renderer:
    def __init__(self, model, textures, scale: float = 0.25, crop=None):
        """textures: list of uint8 RGBA arrays (straight alpha). crop: (x0,y0,x1,y1) canvas px."""
        self.m = model
        self.scale = scale
        cw, ch = model.canvas_size
        self.crop = crop or (0, 0, cw, ch)
        x0, y0, x1, y1 = self.crop
        self.out_w = int(round((x1 - x0) * scale))
        self.out_h = int(round((y1 - y0) * scale))
        self.tex = []
        for t in textures:
            if scale < 1.0:
                tw = max(1, int(round(t.shape[1] * scale)))
                th = max(1, int(round(t.shape[0] * scale)))
                pm = premultiply(t)
                pm = cv2.resize(pm, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                pm = premultiply(t)
            self.tex.append(pm)

    # ------------------------------------------------------------------
    def _raster(self, i: int):
        """Return (y0, x0, rgba premultiplied patch) for drawable i, or None."""
        m = self.m
        pos = m.positions_px(i).astype(np.float64)
        pos[:, 0] = (pos[:, 0] - self.crop[0]) * self.scale
        pos[:, 1] = (pos[:, 1] - self.crop[1]) * self.scale
        idx = m.indices[i].reshape(-1, 3).astype(np.int64)
        if idx.size == 0:
            return None
        bx0 = int(np.floor(pos[:, 0].min())) - 1
        by0 = int(np.floor(pos[:, 1].min())) - 1
        bx1 = int(np.ceil(pos[:, 0].max())) + 2
        by1 = int(np.ceil(pos[:, 1].max())) + 2
        bx0, by0 = max(bx0, 0), max(by0, 0)
        bx1, by1 = min(bx1, self.out_w), min(by1, self.out_h)
        if bx1 <= bx0 or by1 <= by0:
            return None
        w, h = bx1 - bx0, by1 - by0
        local = pos - np.array([bx0, by0])
        idmap = np.full((h, w), -1, dtype=np.int32)
        fixed = np.round(local * (1 << SHIFT)).astype(np.int32)
        for t in range(idx.shape[0]):
            cv2.fillConvexPoly(idmap, fixed[idx[t]], int(t), lineType=cv2.LINE_8, shift=SHIFT)
        ys, xs = np.nonzero(idmap >= 0)
        if ys.size == 0:
            return None
        tid = idmap[ys, xs]
        a = local[idx[tid, 0]]
        b = local[idx[tid, 1]]
        c = local[idx[tid, 2]]
        px = np.stack([xs + 0.5, ys + 0.5], axis=1)
        v0 = b - a
        v1 = c - a
        v2 = px - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        uv = m.uvs[i].astype(np.float64)
        ua, ub, uc = uv[idx[tid, 0]], uv[idx[tid, 1]], uv[idx[tid, 2]]
        u = ua[:, 0] + l1 * (ub[:, 0] - ua[:, 0]) + l2 * (uc[:, 0] - ua[:, 0])
        v = ua[:, 1] + l1 * (ub[:, 1] - ua[:, 1]) + l2 * (uc[:, 1] - ua[:, 1])
        tex = self.tex[int(m.texture_indices[i])]
        th, tw = tex.shape[:2]
        mapx = np.full((h, w), -10.0, dtype=np.float32)
        mapy = np.full((h, w), -10.0, dtype=np.float32)
        mapx[ys, xs] = (u * tw - 0.5).astype(np.float32)
        mapy[ys, xs] = ((1.0 - v) * th - 0.5).astype(np.float32)   # Core UVs are v-up
        patch = cv2.remap(tex, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return by0, bx0, patch

    def render(self, background=None, only=None, skip=None) -> np.ndarray:
        """Return uint8 RGBA (straight alpha) or RGB if background colour given."""
        m = self.m
        out = np.zeros((self.out_h, self.out_w, 4), dtype=np.float32)
        order = np.argsort(m.render_orders, kind="stable")
        mask_cache = {}
        for i in order:
            i = int(i)
            if only is not None and m.drawable_ids[i] not in only:
                continue
            if skip is not None and m.drawable_ids[i] in skip:
                continue
            op = float(m.opacities[i])
            if op <= 0.001 or not (int(m.dynamic_flags[i]) & 1):
                continue
            r = self._raster(i)
            if r is None:
                continue
            y0, x0, patch = r
            h, w = patch.shape[:2]
            patch = patch * op
            if m.mask_counts[i] > 0:
                key = tuple(int(x) for x in m.masks[i])
                if key not in mask_cache:
                    ma = np.zeros((self.out_h, self.out_w), dtype=np.float32)
                    for mi in key:
                        if mi < 0:
                            continue
                        rr = self._raster(mi)
                        if rr is None:
                            continue
                        my0, mx0, mp = rr
                        sl = ma[my0:my0 + mp.shape[0], mx0:mx0 + mp.shape[1]]
                        np.maximum(sl, mp[..., 3], out=sl)
                    mask_cache[key] = ma
                mview = mask_cache[key][y0:y0 + h, x0:x0 + w]
                if int(m.constant_flags[i]) & 8:
                    mview = 1.0 - mview
                patch = patch * mview[..., None]
            dst = out[y0:y0 + h, x0:x0 + w]
            cf = int(m.constant_flags[i])
            if cf & 2:
                # multiplicative (Cubism: colour = dst x (src_premul + 1 - src_alpha), alpha kept)
                dst[..., :3] *= (patch[..., :3] + 1.0 - patch[..., 3:4])
            elif cf & 1:
                # additive (colour added, alpha kept)
                dst[..., :3] += patch[..., :3]
            else:
                dst *= (1.0 - patch[..., 3:4])
                dst += patch
        if background is not None:
            bg = np.array(background, dtype=np.float32) / 255.0
            rgb = out[..., :3] + bg * (1.0 - out[..., 3:4])
            return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        a = out[..., 3:4]
        rgb = np.where(a > 1e-6, out[..., :3] / np.maximum(a, 1e-6), 0)
        return np.clip(np.concatenate([rgb, a], axis=2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
