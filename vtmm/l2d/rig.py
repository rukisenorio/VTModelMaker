"""Procedural rig description -> Live2D runtime files.

Everything is authored in canvas pixels at rest. Each deformer/mesh may carry a function
that maps rest points + parameter values to deformed canvas points; the compiler samples it
on every keyform combination and converts the result into the parent's local space.
"""
from __future__ import annotations

import json
import os
import uuid

import cv2
import numpy as np
from PIL import Image

from .builder import Binding, Builder
from .meshgen import NestAtlas, bleed_far, Atlas, bleed, grid_mesh


class Node:
    def __init__(self, kind, name, parent, part):
        self.kind, self.name, self.parent, self.part = kind, name, parent, part
        self.axes = []

    def to_local(self, pts):
        pts = np.asarray(pts, np.float64)
        if self.kind == "warp":
            x0, y0, x1, y1 = self.rect
            return (pts - np.array([x0, y0])) / np.array([x1 - x0, y1 - y0])
        return pts - np.asarray(self.pivot, np.float64)


def _isotonic(y):
    """Least-squares non-decreasing fit of y (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v_ = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v_]; wts[-2:] = [w]; cnt[-2:] = [c]
    return np.repeat(np.array(vals), cnt)


def _unfold_line(x, xr, m):
    """x along a line of nodes, kept increasing with spacing >= m x the rest spacing (xr), nearest in least squares."""
    need = m * np.maximum(np.diff(xr), 0.0)
    if np.all(np.diff(x) >= need - 1e-9):
        return x
    c = np.concatenate([[0.0], np.cumsum(need)])
    return _isotonic(x - c) + c


def unfold_grid(pts, rest, nr, nc, m):
    P = np.asarray(pts, np.float64).reshape(nr, nc, 2).copy()
    R = np.asarray(rest, np.float64).reshape(nr, nc, 2)
    for _ in range(2):
        for j in range(nr):
            P[j, :, 0] = _unfold_line(P[j, :, 0], R[j, :, 0], m)
        for i in range(nc):
            P[:, i, 1] = _unfold_line(P[:, i, 1], R[:, i, 1], m)
    return P.reshape(-1, 2)


def _local(parent, pts):
    return np.asarray(pts, np.float64) if parent is None else parent.to_local(pts)


def _combos(axes):
    b = Binding(axes)
    names = [a[0] for a in axes]
    return b, [dict(zip(names, combo)) for combo in b.grid()]


class Rig:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.params = []          # (id, lo, hi, default, display name, group)
        self.param_defaults = {}
        self.parts = []
        self.nodes = []
        self.meshes = []
        self.layers = {}          # name -> dict(img RGBA, x0, y0, scale)
        self.physics = []
        self.expressions = {}
        self.param_groups = []

    # ------------------------------------------------------------------ declarations
    def param(self, pid, lo, hi, default=0.0, name=None, group=None):
        self.params.append((pid, lo, hi, default, name or pid, group))
        self.param_defaults[pid] = default
        return pid

    def part(self, pid, name=None):
        self.parts.append((pid, name or pid))
        return pid

    def layer(self, name, img, x0, y0, scale=1.0, hi=None):
        """Register a texture source: RGBA uint8 crop positioned at canvas (x0,y0).
        hi: the same picture at an integer multiple of the canvas resolution (k*h x k*w); the
        atlas then holds that one, and `scale` is ignored."""
        assert img.ndim == 3 and img.shape[2] == 4
        lay = {"img": img, "x0": int(x0), "y0": int(y0), "scale": float(scale)}
        if hi is not None:
            k = hi.shape[0] // img.shape[0]
            assert k >= 2 and hi.shape[:2] == (img.shape[0] * k, img.shape[1] * k), (name, hi.shape, img.shape)
            lay["hi"] = hi
            lay["scale"] = float(k)
        self.layers[name] = lay
        return name

    def warp(self, name, parent, part, rect, cols, rows, axes=(), fn=None, opacity=None, unfold=None):
        """opacity(values)->0..1: a deformer's opacity is passed on to everything that hangs on it.
        unfold: if given (a share of the rest spacing, e.g. 0.2), every keyform's lattice is kept from folding: along each
        row the nodes' x and down each column their y stay in order, at least that share of their rest spacing apart -
        the nearest such lattice in least squares (isotonic regression), so only nodes in a fold move."""
        n = Node("warp", name, parent, part)
        n.rect = tuple(float(v) for v in rect)
        n.cols, n.rows, n.axes, n.fn = cols, rows, list(axes), fn
        n.opacity = opacity
        n.unfold = unfold
        self.nodes.append(n)
        return n

    def rot(self, name, parent, part, pivot, axes=(), angle=None, origin=None, scale=None, opacity=None):
        """angle(values)->deg ; origin(values)->canvas (x,y) ; scale(values)->float ; opacity(values)->0..1
        (a deformer's opacity is passed on to everything that hangs on it)."""
        n = Node("rot", name, parent, part)
        n.pivot = tuple(float(v) for v in pivot)
        n.axes, n.angle, n.origin, n.scale = list(axes), angle, origin, scale
        n.opacity = opacity
        self.nodes.append(n)
        return n

    def mesh(self, name, layer, parent, part, verts, tris, z, axes=(), fn=None, opacity=None,
             masks=(), uv_px=None, inverted_mask=False, draw_order=500, blend="normal"):
        """verts: canvas rest positions. uv_px: per-vertex texture position in canvas px of the
        layer (defaults to the rest positions)."""
        self.meshes.append({
            "name": name, "layer": layer, "parent": parent, "part": part,
            "verts": np.asarray(verts, np.float64).reshape(-1, 2),
            "tris": np.asarray(tris, np.int64).reshape(-1, 3), "z": float(z), "axes": list(axes),
            "fn": fn, "opacity": opacity, "masks": list(masks),
            "uv_px": None if uv_px is None else np.asarray(uv_px, np.float64).reshape(-1, 2),
            "inverted_mask": inverted_mask, "draw_order": draw_order,
            # (Cubism's constant flags: 1 additive, 2 multiplicative blending)
            "blend": {"normal": 0, "add": 1, "multiply": 2}[blend],
        })
        return name

    def layer_mesh(self, name, parent, part, z, cell=48, pad=8, layer=None, canvas_diag=False, thr=8, **kw):
        """Grid mesh that covers a registered layer image (every cell with coverage above `thr` of 255)."""
        lay = self.layers[layer or name]
        verts, tris = grid_mesh(lay["img"][..., 3], lay["x0"], lay["y0"], cell, pad=pad, thr=thr, canvas_diag=canvas_diag)
        return self.mesh(name, layer or name, parent, part, verts, tris, z, **kw)

    # ------------------------------------------------------------------ compile
    def compile(self, out_dir, model_name, atlas_size=4096, texture_dir=None):
        os.makedirs(out_dir, exist_ok=True)
        tex_dir_name = texture_dir or f"{model_name}.{atlas_size}"
        os.makedirs(os.path.join(out_dir, tex_dir_name), exist_ok=True)

        # ---- atlas -----------------------------------------------------------------
        used = sorted({m["layer"] for m in self.meshes})
        ext = {}
        for lname in used:
            lay = self.layers[lname]
            h, w = lay["img"].shape[:2]
            ex0, ey0, ex1, ey1 = lay["x0"], lay["y0"], lay["x0"] + w, lay["y0"] + h
            for m in self.meshes:
                if m["layer"] != lname:
                    continue
                src = m["uv_px"] if m["uv_px"] is not None else m["verts"]
                ex0 = min(ex0, int(np.floor(src[:, 0].min())))
                ey0 = min(ey0, int(np.floor(src[:, 1].min())))
                ex1 = max(ex1, int(np.ceil(src[:, 0].max())))
                ey1 = max(ey1, int(np.ceil(src[:, 1].max())))
            ext[lname] = (ex0, ey0, ex1, ey1)
        sizes = {}
        for lname in used:
            ex0, ey0, ex1, ey1 = ext[lname]
            s = self.layers[lname]["scale"]
            sizes[lname] = (int(np.ceil((ex1 - ex0) * s)) + 2, int(np.ceil((ey1 - ey0) * s)) + 2)
        if os.environ.get("L2D_ATLAS_DUMP"):
            # (what a packer needs to know, for trying packers without building: every layer's box, scale and
            # where its picture has anything, and the triangles of the meshes that use it, in the layer's own px)
            import pickle
            dump = {}
            for lname in used:
                lay = self.layers[lname]
                ex0, ey0, ex1, ey1 = ext[lname]
                occ = np.zeros((ey1 - ey0, ex1 - ex0), bool)
                ih, iw = lay["img"].shape[:2]
                occ[lay["y0"] - ey0: lay["y0"] - ey0 + ih, lay["x0"] - ex0: lay["x0"] - ex0 + iw] = lay["img"][..., 3] > 0
                dump[lname] = {"ext": ext[lname], "scale": lay["scale"], "occ": occ[::4, ::4].copy(),
                               "meshes": [((m["uv_px"] if m["uv_px"] is not None else m["verts"]).copy(), m["tris"].copy(), m["name"])
                                          for m in self.meshes if m["layer"] == lname]}
            pickle.dump(dump, open(os.environ["L2D_ATLAS_DUMP"], "wb"))
        NEST = os.environ.get("L2D_ATLAS", "nest") != "shelf"
        MARGIN, GRID = 16, 16             # px of the atlas around what a layer's meshes reach; the packing grid

        def picture(lname):
            """The layer's picture at the atlas's scale, over its box `ext`, with colour bled into the see-through."""
            lay = self.layers[lname]
            ex0, ey0, ex1, ey1 = ext[lname]
            s = lay["scale"]
            if "hi" in lay:
                k = int(round(s))
                full = np.zeros(((ey1 - ey0) * k, (ex1 - ex0) * k, 4), np.uint8)
                ih, iw = lay["hi"].shape[:2]
                full[(lay["y0"] - ey0) * k: (lay["y0"] - ey0) * k + ih, (lay["x0"] - ex0) * k: (lay["x0"] - ex0) * k + iw] = lay["hi"]
                return bleed(full, 6 * k), 6 * k
            full = np.zeros((ey1 - ey0, ex1 - ex0, 4), np.uint8)
            ih, iw = lay["img"].shape[:2]
            full[lay["y0"] - ey0: lay["y0"] - ey0 + ih, lay["x0"] - ex0: lay["x0"] - ex0 + iw] = lay["img"]
            full = bleed(full, 6)
            if s == 1.0:
                return full, 6
            tw, th = sizes[lname][0] - 2, sizes[lname][1] - 2
            pm = full.astype(np.float32)
            pm[..., :3] *= pm[..., 3:4] / 255.0
            pm = cv2.resize(pm, (tw, th), interpolation=cv2.INTER_AREA)
            a = pm[..., 3:4]
            rgb = np.where(a > 0.5, pm[..., :3] * 255.0 / np.maximum(a, 0.5), 0)
            full = np.clip(np.concatenate([rgb, a], 2) + 0.5, 0, 255).astype(np.uint8)
            return bleed(full, 4), 4

        turn = {}
        if not NEST:
            atlas = Atlas(atlas_size, gap=4)
            place = atlas.pack(sizes)
            pages = [np.zeros((atlas_size, atlas_size, 4), np.uint8) for _ in atlas.pages]
            for lname in used:
                full, _ = picture(lname)
                pi, px, py = place[lname]
                pages[pi][py + 1: py + 1 + full.shape[0], px + 1: px + 1 + full.shape[1]] = full
                turn[lname] = (pi, px + 1, py + 1, 0, 0, 0)
        else:
            # Shapes, not boxes (meshgen.NestAtlas): a layer takes the cells of the atlas that its meshes' triangles
            # reach, grown by MARGIN - what else of its box is see-through belongs to whoever fits in there.
            frames, masks = {}, {}
            for lname in used:
                lay = self.layers[lname]
                ex0, ey0, ex1, ey1 = ext[lname]
                s = lay["scale"]
                full, bled = picture(lname)
                th, tw = full.shape[:2]
                hp, wp = -(-(th + 2 * MARGIN) // GRID) * GRID, -(-(tw + 2 * MARGIN) // GRID) * GRID
                assert hp <= atlas_size and wp <= atlas_size, (lname, wp, hp)
                frame = np.zeros((hp, wp, 4), np.uint8)
                frame[MARGIN:MARGIN + th, MARGIN:MARGIN + tw] = full
                frame = bleed_far(frame, 2 * MARGIN, done=bled)
                cov = np.zeros((hp, wp), np.uint8)
                for m in self.meshes:
                    if m["layer"] != lname:
                        continue
                    src = m["uv_px"] if m["uv_px"] is not None else m["verts"]
                    P = np.round((src - (ex0, ey0)) * s + MARGIN).astype(np.int32)
                    cv2.fillPoly(cov, [P[t] for t in m["tris"]], 1)
                cov = cv2.dilate(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN + 1, 2 * MARGIN + 1)))
                frames[lname] = frame
                masks[lname] = cov.reshape(hp // GRID, GRID, wp // GRID, GRID).max((1, 3))
            atlas = NestAtlas(atlas_size, GRID)
            place = atlas.pack(masks)
            pages = [np.zeros((atlas_size, atlas_size, 4), np.uint8) for _ in atlas.pages]
            for lname in used:
                pi, px, py, k = place[lname]
                fr = np.rot90(frames[lname], k)
                own = cv2.resize(np.ascontiguousarray(np.rot90(masks[lname], k)), None, fx=GRID, fy=GRID, interpolation=cv2.INTER_NEAREST) > 0
                sub = pages[pi][py: py + fr.shape[0], px: px + fr.shape[1]]
                sub[own] = fr[own]
                turn[lname] = (pi, px, py, k, frames[lname].shape[1], frames[lname].shape[0])
            del frames
            used_cells = sum(float(p.sum()) for p in atlas.pages)
            print(f"テクスチャ: レイヤー {len(used)} 枚を {atlas_size} px のページ {len(pages)} 枚に詰めました"
                  f"（ページの {100.0 * used_cells / (len(pages) * atlas.n ** 2):.0f} % を使用、"
                  f"{sum(1 for v in place.values() if v[3])} 枚は回して配置）")
        tex_files = []
        for i, pg in enumerate(pages):
            rel = f"{tex_dir_name}/texture_{i:02d}.png"
            Image.fromarray(pg).save(os.path.join(out_dir, rel), compress_level=6)
            tex_files.append(rel)
        # (pages of an earlier, larger atlas must not stay behind in the folder)
        for f_ in os.listdir(os.path.join(out_dir, tex_dir_name)):
            if f_.startswith("texture_") and f"{tex_dir_name}/{f_}" not in tex_files:
                os.remove(os.path.join(out_dir, tex_dir_name, f_))
        self.atlas_pages = pages

        def uv_of(m):
            lname = m["layer"]
            ex0, ey0, _, _ = ext[lname]
            s = self.layers[lname]["scale"]
            pi, px, py, k, wp, hp = turn[lname]
            src = m["uv_px"] if m["uv_px"] is not None else m["verts"]
            fx, fy = (src[:, 0] - ex0) * s, (src[:, 1] - ey0) * s
            if NEST:
                fx, fy = fx + MARGIN, fy + MARGIN
                if k == 1:
                    fx, fy = fy, wp - fx
                elif k == 2:
                    fx, fy = wp - fx, hp - fy
                elif k == 3:
                    fx, fy = hp - fy, fx
            return np.stack([(px + fx) / atlas_size, (py + fy) / atlas_size], 1), pi

        # ---- model -----------------------------------------------------------------
        b = Builder(self.w, self.h)
        for pid, lo, hi, default, _, _ in self.params:
            b.param(pid, lo, hi, default)
        for pid, _ in self.parts:
            b.part(pid)

        for n in self.nodes:
            binding, combos = _combos(n.axes)
            pname = None if n.parent is None else n.parent.name
            if n.kind == "warp":
                x0, y0, x1, y1 = n.rect
                ys, xs = np.meshgrid(np.linspace(y0, y1, n.rows + 1), np.linspace(x0, x1, n.cols + 1), indexing="ij")
                rest = np.stack([xs, ys], -1).reshape(-1, 2)
                grids, dop = [], []
                for values in combos:
                    full = dict(self.param_defaults)
                    full.update(values)
                    pts = rest if n.fn is None else np.asarray(n.fn(rest.copy(), full), np.float64)
                    if getattr(n, "unfold", None):
                        pts = unfold_grid(pts, rest, n.rows + 1, n.cols + 1, n.unfold)
                    grids.append(_local(n.parent, pts))
                    dop.append(1.0 if n.opacity is None else float(n.opacity(full)))
                b.warp(n.name, n.part, pname, n.rows, n.cols, binding if n.axes else None, grids,
                       opacities=None if n.opacity is None else dop)
            else:
                origins, angles, scales, dop = [], [], [], []
                for values in combos:
                    full = dict(self.param_defaults)
                    full.update(values)
                    dop.append(1.0 if n.opacity is None else float(n.opacity(full)))
                    o = np.asarray(n.pivot if n.origin is None else n.origin(full), np.float64)
                    origins.append(_local(n.parent, o.reshape(1, 2))[0])
                    angles.append(0.0 if n.angle is None else float(n.angle(full)))
                    scales.append(1.0 if n.scale is None else float(n.scale(full)))
                b.rotation(n.name, n.part, pname, binding if n.axes else None, origins, angles, scales,
                           opacities=None if n.opacity is None else dop)

        order = sorted(range(len(self.meshes)), key=lambda i: self.meshes[i]["z"])
        for i in order:
            m = self.meshes[i]
            binding, combos = _combos(m["axes"])
            positions, opac = [], []
            for values in combos:
                full = dict(self.param_defaults)
                full.update(values)
                pts = m["verts"] if m["fn"] is None else np.asarray(m["fn"](m["verts"].copy(), full), np.float64)
                positions.append(_local(m["parent"], pts))
                opac.append(1.0 if m["opacity"] is None else float(m["opacity"](full)))
            uv, page = uv_of(m)
            b.art(m["name"], m["part"], None if m["parent"] is None else m["parent"].name,
                  binding if m["axes"] else None, positions, uv, m["tris"], texture=page,
                  opacities=opac, masks=m["masks"], inverted_mask=m["inverted_mask"],
                  draw_orders=m["draw_order"], flags=4 | m.get("blend", 0))
        moc_name = f"{model_name}.moc3"
        b.write(os.path.join(out_dir, moc_name))

        # ---- json side files ----------------------------------------------------------
        model3 = {
            "Version": 3,
            "FileReferences": {"Moc": moc_name, "Textures": tex_files,
                               "DisplayInfo": f"{model_name}.cdi3.json"},
            "Groups": [
                {"Target": "Parameter", "Name": "EyeBlink",
                 "Ids": [p for p in ("ParamEyeLOpen", "ParamEyeROpen") if p in self.param_defaults]},
                {"Target": "Parameter", "Name": "LipSync",
                 "Ids": [p for p in ("ParamMouthOpenY",) if p in self.param_defaults]},
            ],
            "HitAreas": [],
        }
        if self.physics:
            model3["FileReferences"]["Physics"] = f"{model_name}.physics3.json"
            with open(os.path.join(out_dir, f"{model_name}.physics3.json"), "w", encoding="utf-8") as fh:
                json.dump(self._physics_json(), fh, indent=1)
        if self.expressions:
            os.makedirs(os.path.join(out_dir, "expressions"), exist_ok=True)
            model3["FileReferences"]["Expressions"] = []
            for ename, e in self.expressions.items():
                rel = f"expressions/{ename}.exp3.json"
                with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as fh:
                    json.dump({"Type": "Live2D Expression", "FadeInTime": e.get("fade", 0.15),
                               "FadeOutTime": e.get("fade", 0.15),
                               # (a value given as (value, blend) has its own blend mode)
                               "Parameters": [{"Id": k, "Value": (v[0] if isinstance(v, (tuple, list)) else v),
                                               "Blend": (v[1] if isinstance(v, (tuple, list)) else e.get("blend", "Overwrite"))}
                                              for k, v in e["params"].items()]}, fh, indent=1)
                model3["FileReferences"]["Expressions"].append({"Name": ename, "File": rel})
        with open(os.path.join(out_dir, f"{model_name}.model3.json"), "w", encoding="utf-8") as fh:
            json.dump(model3, fh, indent=1)
        groups = []
        seen = []
        for _, _, _, _, _, g in self.params:
            if g and g not in seen:
                seen.append(g)
        cdi = {
            "Version": 3,
            "Parameters": [{"Id": pid, "GroupId": (f"ParamGroup{seen.index(g) + 1}" if g else ""), "Name": nm}
                           for pid, _, _, _, nm, g in self.params],
            "ParameterGroups": [{"Id": f"ParamGroup{i + 1}", "GroupId": "", "Name": g} for i, g in enumerate(seen)],
            "Parts": [{"Id": pid, "Name": nm} for pid, nm in self.parts],
        }
        with open(os.path.join(out_dir, f"{model_name}.cdi3.json"), "w", encoding="utf-8") as fh:
            json.dump(cdi, fh, indent=1, ensure_ascii=False)
        return os.path.join(out_dir, moc_name)

    # ------------------------------------------------------------------ physics
    def add_physics(self, name, inputs, outputs, vertices, norm_pos=(-10, 0, 10), norm_ang=(-10, 0, 10)):
        """inputs: [(param, weight, 'X'|'Y'|'Angle', reflect)], outputs: [(param, vertex_index,
        scale, weight, 'X'|'Y'|'Angle', reflect)], vertices: [(x, y, mobility, delay, accel, radius)]."""
        self.physics.append({"name": name, "inputs": inputs, "outputs": outputs, "vertices": vertices,
                             "norm_pos": norm_pos, "norm_ang": norm_ang})

    def _physics_json(self):
        settings, dictionary = [], []
        ti = to = tv = 0
        for i, p in enumerate(self.physics):
            sid = f"PhysicsSetting{i + 1}"
            dictionary.append({"Id": sid, "Name": p["name"]})
            ins = [{"Source": {"Target": "Parameter", "Id": a[0]}, "Weight": a[1], "Type": a[2],
                    "Reflect": bool(a[3]) if len(a) > 3 else False} for a in p["inputs"]]
            outs = [{"Destination": {"Target": "Parameter", "Id": a[0]}, "VertexIndex": a[1],
                     "Scale": a[2], "Weight": a[3], "Type": a[4],
                     "Reflect": bool(a[5]) if len(a) > 5 else False} for a in p["outputs"]]
            verts = [{"Position": {"X": v[0], "Y": v[1]}, "Mobility": v[2], "Delay": v[3],
                      "Acceleration": v[4], "Radius": v[5]} for v in p["vertices"]]
            settings.append({
                "Id": sid, "Input": ins, "Output": outs, "Vertices": verts,
                "Normalization": {
                    "Position": {"Minimum": p["norm_pos"][0], "Default": p["norm_pos"][1], "Maximum": p["norm_pos"][2]},
                    "Angle": {"Minimum": p["norm_ang"][0], "Default": p["norm_ang"][1], "Maximum": p["norm_ang"][2]},
                },
            })
            ti += len(ins)
            to += len(outs)
            tv += len(verts)
        return {
            "Version": 3,
            "Meta": {"PhysicsSettingCount": len(settings), "TotalInputCount": ti, "TotalOutputCount": to,
                     "VertexCount": tv, "Fps": 60,
                     "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
                     "PhysicsDictionary": dictionary},
            "PhysicsSettings": settings,
        }


def new_model_id() -> str:
    return uuid.uuid4().hex
