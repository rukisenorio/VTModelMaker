"""High level builder that assembles a Live2D model (parameters, deformers, art meshes)
and serialises it with moc3.write().

Coordinates handed to the builder are always in the *parent's local space*:
  - no parent deformer : canvas pixels (x right, y down)
  - parent is a warp   : normalised [0,1]x[0,1] inside the warp's grid
  - parent is rotation : pixels relative to the rotation origin (parent's rotated frame)
"""
from __future__ import annotations

import itertools

import numpy as np

from . import moc3

WARP, ROT = 0, 1


class Binding:
    """Keyform axes of one object: list of (param_id, [key values])."""

    def __init__(self, axes):
        self.axes = [(pid, [float(k) for k in keys]) for pid, keys in axes]
        for _, keys in self.axes:
            assert keys == sorted(keys) and len(set(keys)) == len(keys), keys

    @property
    def shape(self):
        return tuple(len(k) for _, k in self.axes)

    @property
    def count(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def grid(self):
        """Iterate key-value tuples in storage order (first axis varies fastest)."""
        ranges = [keys for _, keys in self.axes]
        for combo in itertools.product(*reversed(ranges)):
            yield tuple(reversed(combo))


class Builder:
    def __init__(self, width: float, height: float, ppu: float | None = None):
        self.width, self.height = float(width), float(height)
        self.ppu = float(ppu or width)
        self.origin = (self.width / 2, self.height / 2)
        self.params = []      # dict(id,min,max,default)
        self.parts = []       # dict(id,parent)
        self.deformers = []   # dict
        self.arts = []        # dict
        self._ids = set()

    # ------------------------------------------------------------------ declarations
    def _uid(self, s):
        assert s not in self._ids, f"duplicate id {s}"
        assert len(s.encode()) < 64
        self._ids.add(s)
        return s

    def param(self, pid, lo, hi, default=0.0):
        self.params.append({"id": self._uid(pid), "min": float(lo), "max": float(hi),
                            "default": float(default)})
        return pid

    def part(self, pid, parent=None):
        self.parts.append({"id": self._uid(pid), "parent": parent})
        return pid

    def warp(self, did, part, parent, rows, cols, binding, grids, opacities=None, quad=True):
        """grids: array [n_keyforms, (rows+1)*(cols+1), 2] in parent-local coords."""
        binding = binding or Binding([])
        grids = np.asarray(grids, dtype=np.float64).reshape(binding.count, (rows + 1) * (cols + 1), 2)
        self.deformers.append({
            "id": self._uid(did), "type": WARP, "part": part, "parent": parent, "rows": rows,
            "cols": cols, "binding": binding, "grids": grids, "quad": quad,
            "opacities": np.ones(binding.count) if opacities is None else np.asarray(opacities, float),
        })
        return did

    def rotation(self, did, part, parent, binding, origins, angles, scales=None, opacities=None,
                 base_angle=0.0):
        """origins [n,2] parent-local; angles [n] degrees (clockwise on screen)."""
        binding = binding or Binding([])
        n = binding.count
        origins = np.asarray(origins, dtype=np.float64).reshape(n, 2)
        angles = np.broadcast_to(np.asarray(angles, dtype=np.float64), (n,)).copy()
        self.deformers.append({
            "id": self._uid(did), "type": ROT, "part": part, "parent": parent, "binding": binding,
            "origins": origins, "angles": angles, "base_angle": float(base_angle),
            "scales": np.ones(n) if scales is None else np.broadcast_to(np.asarray(scales, float), (n,)).copy(),
            "opacities": np.ones(n) if opacities is None else np.asarray(opacities, float),
        })
        return did

    def art(self, aid, part, parent, binding, positions, uvs, triangles, texture=0,
            opacities=None, draw_orders=None, masks=(), flags=4, inverted_mask=False):
        """positions [n_keyforms, n_vertices, 2] parent-local; uvs [n_vertices,2] (v down, 0..1)."""
        binding = binding or Binding([])
        uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        nv = uvs.shape[0]
        positions = np.asarray(positions, dtype=np.float64).reshape(binding.count, nv, 2)
        tris = np.asarray(triangles, dtype=np.int64).reshape(-1)
        assert tris.size % 3 == 0 and tris.max() < nv and nv < 32767
        n = binding.count
        self.arts.append({
            "id": self._uid(aid), "part": part, "parent": parent, "binding": binding,
            "positions": positions, "uvs": uvs, "tris": tris, "texture": int(texture),
            "opacities": np.ones(n) if opacities is None else np.broadcast_to(np.asarray(opacities, float), (n,)).copy(),
            "draw_orders": np.full(n, 500.0) if draw_orders is None else np.broadcast_to(np.asarray(draw_orders, float), (n,)).copy(),
            "masks": list(masks), "flags": int(flags) | (8 if inverted_mask else 0),
        })
        return aid

    # ------------------------------------------------------------------ serialisation
    def build(self) -> dict:
        ppu = self.ppu
        ox, oy = self.origin
        pidx = {p["id"]: i for i, p in enumerate(self.params)}
        partidx = {p["id"]: i for i, p in enumerate(self.parts)}
        didx = {d["id"]: i for i, d in enumerate(self.deformers)}
        aidx = {a["id"]: i for i, a in enumerate(self.arts)}

        # --- parameter bindings: unique (param, keys) pairs, grouped per parameter
        pb_keys = {}  # (param_index, keys tuple) -> None
        objects = self.deformers + self.arts
        for o in objects:
            for pid, keys in o["binding"].axes:
                assert pid in pidx, f"unknown parameter {pid}"
                pb_keys.setdefault((pidx[pid], tuple(keys)), None)
        pb_list = sorted(pb_keys.keys(), key=lambda t: (t[0], len(t[1]), t[1]))
        pb_index = {k: i for i, k in enumerate(pb_list)}
        keys_values, pb_begin, pb_count = [], [], []
        for _, keys in pb_list:
            pb_begin.append(len(keys_values))
            pb_count.append(len(keys))
            keys_values.extend(keys)
        par_pb_begin, par_pb_count = [], []
        for i in range(len(self.params)):
            mine = [j for j, (p, _) in enumerate(pb_list) if p == i]
            par_pb_begin.append(mine[0] if mine else -1)
            par_pb_count.append(len(mine))

        # --- keyform bindings: unique ordered tuples of parameter bindings; index 0 = no axes
        kfb_list = [()]
        kfb_index = {(): 0}

        def kfb_of(binding):
            key = tuple(pb_index[(pidx[pid], tuple(keys))] for pid, keys in binding.axes)
            if key not in kfb_index:
                kfb_index[key] = len(kfb_list)
                kfb_list.append(key)
            return kfb_index[key]

        pbi, kfb_begin, kfb_count = [], [], []
        obj_kfb = [kfb_of(o["binding"]) for o in objects]
        for key in kfb_list:
            kfb_begin.append(len(pbi))
            kfb_count.append(len(key))
            pbi.extend(key)

        kfpos = []

        def push_positions(arr):
            begin = len(kfpos)
            kfpos.extend(np.asarray(arr, dtype=np.float64).reshape(-1).tolist())
            return begin

        def to_parent_units(o, pts):
            """Root-level coordinates are canvas pixels in the builder, Core units in the file."""
            if o["parent"] is None:
                out = np.array(pts, dtype=np.float64)
                out[..., 0] = (out[..., 0] - ox) / ppu
                out[..., 1] = (out[..., 1] - oy) / ppu
                return out
            return np.asarray(pts, dtype=np.float64)

        # --- deformers
        D = {k: [] for k in ("ids", "kfbIdx", "isVisible", "isEnabled", "parentPart",
                             "parentDeformer", "type", "specificIdx")}
        Wp = {k: [] for k in ("kfbIdx", "kfBegin", "kfCount", "vertexCount", "rows", "cols", "isQuadSource")}
        Rt = {k: [] for k in ("kfbIdx", "kfBegin", "kfCount", "baseAngle")}
        WK = {"opacity": [], "posBegin": []}
        RK = {k: [] for k in ("opacity", "angle", "originX", "originY", "scale", "reflectX", "reflectY")}
        for oi, d in enumerate(self.deformers):
            kfb = obj_kfb[oi]
            n = d["binding"].count
            D["ids"].append(d["id"])
            D["kfbIdx"].append(kfb)
            D["isVisible"].append(1)
            D["isEnabled"].append(1)
            D["parentPart"].append(partidx[d["part"]])
            D["parentDeformer"].append(-1 if d["parent"] is None else didx[d["parent"]])
            D["type"].append(d["type"])
            if d["type"] == WARP:
                D["specificIdx"].append(len(Wp["kfbIdx"]))
                Wp["kfbIdx"].append(kfb)
                Wp["kfBegin"].append(len(WK["opacity"]))
                Wp["kfCount"].append(n)
                Wp["vertexCount"].append((d["rows"] + 1) * (d["cols"] + 1))
                Wp["rows"].append(d["rows"])
                Wp["cols"].append(d["cols"])
                Wp["isQuadSource"].append(1 if d["quad"] else 0)
                grids = to_parent_units(d, d["grids"])
                for k in range(n):
                    WK["opacity"].append(d["opacities"][k])
                    WK["posBegin"].append(push_positions(grids[k]))
            else:
                D["specificIdx"].append(len(Rt["kfbIdx"]))
                Rt["kfbIdx"].append(kfb)
                Rt["kfBegin"].append(len(RK["opacity"]))
                Rt["kfCount"].append(n)
                Rt["baseAngle"].append(d["base_angle"])
                origins = to_parent_units(d, d["origins"])
                # children of a rotation deformer live in pixels; a root rotation converts to units
                unit = (1.0 / ppu) if self._needs_unit_scale(d) else 1.0
                for k in range(n):
                    RK["opacity"].append(d["opacities"][k])
                    RK["angle"].append(d["angles"][k])
                    RK["originX"].append(origins[k, 0])
                    RK["originY"].append(origins[k, 1])
                    RK["scale"].append(d["scales"][k] * unit)
                    RK["reflectX"].append(0)
                    RK["reflectY"].append(0)

        # --- art meshes
        A = {k: [] for k in ("ids", "kfbIdx", "kfBegin", "kfCount", "isVisible", "isEnabled",
                             "parentPart", "parentDeformer", "textureNo", "flags", "vertexCount",
                             "uvBegin", "idxBegin", "idxCount", "maskBegin", "maskCount")}
        AK = {"opacity": [], "drawOrder": [], "posBegin": []}
        uvs, pos_idx, masks = [], [], []
        nd = len(self.deformers)
        for ai, a in enumerate(self.arts):
            kfb = obj_kfb[nd + ai]
            n = a["binding"].count
            nv = a["uvs"].shape[0]
            A["ids"].append(a["id"])
            A["kfbIdx"].append(kfb)
            A["kfBegin"].append(len(AK["opacity"]))
            A["kfCount"].append(n)
            A["isVisible"].append(1)
            A["isEnabled"].append(1)
            A["parentPart"].append(partidx[a["part"]])
            A["parentDeformer"].append(-1 if a["parent"] is None else didx[a["parent"]])
            A["textureNo"].append(a["texture"])
            A["flags"].append(a["flags"])
            A["vertexCount"].append(nv)
            A["uvBegin"].append(len(uvs))
            uvs.extend(a["uvs"].reshape(-1).tolist())   # stored v-down; the Core returns 1-v
            A["idxBegin"].append(len(pos_idx))
            A["idxCount"].append(a["tris"].size)
            pos_idx.extend(a["tris"].tolist())
            A["maskBegin"].append(len(masks))
            A["maskCount"].append(len(a["masks"]))
            masks.extend(aidx[mid] for mid in a["masks"])
            positions = to_parent_units(a, a["positions"])
            for k in range(n):
                AK["opacity"].append(a["opacities"][k])
                AK["drawOrder"].append(a["draw_orders"][k])
                AK["posBegin"].append(push_positions(positions[k]))

        na = len(self.arts)
        order_min = int(min((a["draw_orders"].min() for a in self.arts), default=500))
        order_max = int(max((a["draw_orders"].max() for a in self.arts), default=500))
        npart = len(self.parts)
        model = {
            "version": 3,
            "canvas": {"ppu": ppu, "originX": ox, "originY": oy, "width": self.width,
                       "height": self.height, "flags": 0},
            "parts": {
                "ids": [p["id"] for p in self.parts], "kfbIdx": np.zeros(npart, "i4"),
                "kfBegin": np.arange(npart, dtype="i4"), "kfCount": np.ones(npart, "i4"),
                "isVisible": np.ones(npart, "i4"), "isEnabled": np.ones(npart, "i4"),
                "parentPart": np.array([-1 if p["parent"] is None else partidx[p["parent"]]
                                        for p in self.parts], "i4"),
            },
            "deformers": D, "warp": Wp, "rot": Rt, "art": A,
            "params": {
                "ids": [p["id"] for p in self.params],
                "max": [p["max"] for p in self.params], "min": [p["min"] for p in self.params],
                "default": [p["default"] for p in self.params],
                "isRepeat": np.zeros(len(self.params), "i4"),
                "decimalPlaces": np.full(len(self.params), 3, "i4"),
                "pbBegin": par_pb_begin, "pbCount": par_pb_count,
            },
            "partKf": {"drawOrder": np.full(npart, 500.0)},
            "warpKf": WK, "rotKf": RK, "artKf": AK,
            "kfPos": {"xy": kfpos},
            "pbi": {"idx": pbi},
            "kfb": {"pbiBegin": kfb_begin, "pbiCount": kfb_count},
            "pb": {"keysBegin": pb_begin, "keysCount": pb_count},
            "keys": {"value": keys_values},
            "uvs": {"uv": uvs},
            "posIdx": {"idx": pos_idx},
            "masks": {"drawableIdx": masks},
            "dog": {"objBegin": [0], "objCount": [na], "objTotalCount": [na],
                    "maxDrawOrder": [order_max], "minDrawOrder": [order_min]},
            "dogo": {"type": np.zeros(na, "i4"), "index": np.arange(na, dtype="i4"),
                     "selfIndex": np.full(na, -1, "i4")},
            "glue": {"ids": [], "kfbIdx": [], "kfBegin": [], "kfCount": [], "artA": [], "artB": [],
                     "infoBegin": [], "infoCount": []},
            "glueInfo": {"weight": [], "posIdx": []},
            "glueKf": {"intensity": []},
        }
        return model

    def _needs_unit_scale(self, d) -> bool:
        """True when no rotation deformer above `d` has already converted pixels to units."""
        by_id = {x["id"]: x for x in self.deformers}
        p = d["parent"]
        while p is not None:
            if by_id[p]["type"] == ROT:
                return False
            p = by_id[p]["parent"]
        return True

    def write(self, path: str):
        moc3.write(self.build(), path, version=3)
