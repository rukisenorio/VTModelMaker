"""Reference re-implementation of the Core's deformer evaluation, used to pin down the
coordinate conventions of the moc3 format by comparing against the real Core."""
from __future__ import annotations

import math

import numpy as np


class Evaluator:
    def __init__(self, m: dict, opts: dict | None = None):
        self.m = m
        self.o = {
            "first_fastest": True,
            "quad": None,           # None -> per-warp isQuadSource (False for v1)
            "angle_sign": 1.0,
            "use_base_angle": True,
        }
        if opts:
            self.o.update(opts)
        p = m["params"]
        self.pb_param = {}
        for pi, (b, c) in enumerate(zip(p["pbBegin"], p["pbCount"])):
            for j in range(b, b + c):
                self.pb_param[j] = pi
        self.values = np.array(p["default"], dtype=np.float64)

    # ------------------------------------------------------------------ keyforms
    def combos(self, kfb_idx: int):
        m = self.m
        b = int(m["kfb"]["pbiBegin"][kfb_idx])
        c = int(m["kfb"]["pbiCount"][kfb_idx])
        pbs = [int(x) for x in m["pbi"]["idx"][b:b + c]]
        if not self.o["first_fastest"]:
            pbs = pbs[::-1]
        out = [(0, 1.0)]
        stride = 1
        for pb in pbs:
            kb = int(m["pb"]["keysBegin"][pb])
            kc = int(m["pb"]["keysCount"][pb])
            keys = m["keys"]["value"][kb:kb + kc]
            v = self.values[self.pb_param[pb]]
            if kc == 1 or v <= keys[0]:
                i, t = 0, 0.0
            elif v >= keys[-1]:
                i, t = kc - 1, 0.0
            else:
                i = int(np.searchsorted(keys, v, side="right") - 1)
                t = float((v - keys[i]) / (keys[i + 1] - keys[i]))
            new = []
            for off, w in out:
                if t < 1.0:
                    new.append((off + i * stride, w * (1 - t)))
                if t > 0.0:
                    new.append((off + (i + 1) * stride, w * t))
            out = new
            stride *= kc
        return out

    def _blend(self, kfb_idx, begin, fn):
        acc = None
        for off, w in self.combos(kfb_idx):
            v = fn(begin + off)
            acc = v * w if acc is None else acc + v * w
        return acc

    # ------------------------------------------------------------------ deformers
    def deformer(self, di: int):
        cache = self.__dict__.setdefault("_dcache", {})
        if di in cache:
            return cache[di]
        m = self.m
        d = m["deformers"]
        parent = int(d["parentDeformer"][di])
        si = int(d["specificIdx"][di])
        xy = m["kfPos"]["xy"]
        pd = self.deformer(parent) if parent >= 0 else None
        pscale = pd["scale"] if pd else 1.0
        pop = pd["opacity"] if pd else 1.0
        if int(d["type"][di]) == 0:
            w = m["warp"]
            rows, cols, n = int(w["rows"][si]), int(w["cols"][si]), int(w["vertexCount"][si])
            kb = int(w["kfBegin"][si])
            kfb = int(w["kfbIdx"][si])
            pos = self._blend(kfb, kb, lambda k: xy[int(m["warpKf"]["posBegin"][k]):
                                                    int(m["warpKf"]["posBegin"][k]) + n * 2].astype(np.float64))
            op = self._blend(kfb, kb, lambda k: float(m["warpKf"]["opacity"][k]))
            grid = pos.reshape(-1, 2)
            if pd:
                grid = self.apply(pd, grid)
            quad = self.o["quad"]
            if quad is None:
                quad = bool(w["isQuadSource"][si]) if "isQuadSource" in w else False
            res = {"kind": "warp", "grid": grid.reshape(rows + 1, cols + 1, 2), "rows": rows,
                   "cols": cols, "quad": quad, "scale": pscale, "opacity": op * pop}
        else:
            r = m["rot"]
            kb = int(r["kfBegin"][si])
            k = m["rotKf"]
            kfb = int(r["kfbIdx"][si])
            angle = self._blend(kfb, kb, lambda q: float(k["angle"][q]))
            ox = self._blend(kfb, kb, lambda q: float(k["originX"][q]))
            oy = self._blend(kfb, kb, lambda q: float(k["originY"][q]))
            sc = self._blend(kfb, kb, lambda q: float(k["scale"][q]))
            op = self._blend(kfb, kb, lambda q: float(k["opacity"][q]))
            first = self.combos(kfb)[0][0] + kb
            rx, ry = bool(k["reflectX"][first]), bool(k["reflectY"][first])
            base = float(r["baseAngle"][si]) if self.o["use_base_angle"] else 0.0
            origin = np.array([[ox, oy]])
            ang = angle + base
            if pd is None:
                o_root = origin[0]
            else:
                o_root = self.apply(pd, origin)[0]
                if pd["kind"] == "rot":
                    ang += pd["angle"]
                else:
                    eps = self.o.get("eps", 0.01)
                    q = self.apply(pd, origin + np.array([[0.0, -eps]]))[0]
                    dvec = q - o_root
                    ang += math.degrees(math.atan2(dvec[0], -dvec[1])) * self.o.get("warp_dir_sign", 1.0)
            res = {"kind": "rot", "origin": o_root, "angle": ang, "scale": sc * pscale,
                   "rx": rx, "ry": ry, "opacity": op * pop}
        cache[di] = res
        return res

    def apply(self, dres, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if dres["kind"] == "warp":
            grid, rows, cols = dres["grid"], dres["rows"], dres["cols"]
            u = pts[:, 0] * cols
            v = pts[:, 1] * rows
            i = np.clip(np.floor(u).astype(int), 0, cols - 1)
            j = np.clip(np.floor(v).astype(int), 0, rows - 1)
            fx = (u - i)[:, None]
            fy = (v - j)[:, None]
            p00 = grid[j, i]
            p10 = grid[j, i + 1]
            p01 = grid[j + 1, i]
            p11 = grid[j + 1, i + 1]
            if dres["quad"]:
                return (p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy)
                        + p01 * (1 - fx) * fy + p11 * fx * fy)
            tri_a = p00 + (p10 - p00) * fx + (p01 - p00) * fy
            tri_b = p11 + (p01 - p11) * (1 - fx) + (p10 - p11) * (1 - fy)
            return np.where((fx + fy) <= 1.0, tri_a, tri_b)
        a = math.radians(dres["angle"] * self.o["angle_sign"])
        c, s = math.cos(a), math.sin(a)
        scale = dres["scale"]
        x = pts[:, 0] * scale * (-1.0 if dres["rx"] else 1.0)
        y = pts[:, 1] * scale * (-1.0 if dres["ry"] else 1.0)
        out = np.empty_like(pts)
        out[:, 0] = dres["origin"][0] + x * c - y * s
        out[:, 1] = dres["origin"][1] + x * s + y * c
        return out

    def to_root(self, parent: int, pts):
        if parent < 0:
            return np.asarray(pts, dtype=np.float64), 1.0
        d = self.deformer(parent)
        return self.apply(d, pts), d["opacity"]

    # ------------------------------------------------------------------ art meshes
    def art(self, ai: int):
        m = self.m
        a = m["art"]
        n = int(a["vertexCount"][ai])
        kb = int(a["kfBegin"][ai])
        kfb = int(a["kfbIdx"][ai])
        xy = m["kfPos"]["xy"]
        k = m["artKf"]
        pos = self._blend(kfb, kb, lambda q: xy[int(k["posBegin"][q]): int(k["posBegin"][q]) + n * 2]
                          .astype(np.float64)).reshape(-1, 2)
        op = self._blend(kfb, kb, lambda q: float(k["opacity"][q]))
        root, pop = self.to_root(int(a["parentDeformer"][ai]), pos)
        return root, op * pop

    def set_values(self, values):
        self.values = np.array(values, dtype=np.float64)
        self._dcache = {}
