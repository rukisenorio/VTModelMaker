"""Port of the Cubism physics3 pendulum solver, so previews show the real follow-through."""
from __future__ import annotations

import json
import math

AIR_RESISTANCE = 5.0
MOVEMENT_THRESHOLD = 0.001


def _norm(value, pmin, pmax, nmin, nmax, ndef, inverted):
    maxv, minv = max(pmax, pmin), min(pmax, pmin)
    value = min(max(value, minv), maxv)
    minn, maxn = min(nmin, nmax), max(nmin, nmax)
    mid = minv + (maxv - minv) / 2.0
    pv = value - mid
    if pv > 0:
        res = pv * ((maxn - ndef) / (maxv - mid)) + ndef
    elif pv < 0:
        res = pv * ((minn - ndef) / (minv - mid)) + ndef
    else:
        res = ndef
    return res if inverted else -res


def _dir_to_rad(frm, to):
    r = math.atan2(to[1], to[0]) - math.atan2(frm[1], frm[0])
    while r < -math.pi:
        r += 2 * math.pi
    while r > math.pi:
        r -= 2 * math.pi
    return r


class Physics:
    def __init__(self, path: str, ranges: dict):
        """ranges: param id -> (min, max, default)."""
        self.j = json.load(open(path))
        self.ranges = ranges
        self.settings = []
        for s in self.j["PhysicsSettings"]:
            parts = []
            for i, v in enumerate(s["Vertices"]):
                pos = [0.0, 0.0] if i == 0 else [parts[-1]["pos"][0], parts[-1]["pos"][1] + v["Radius"]]
                parts.append({"pos": list(pos), "last": list(pos), "lastg": [0.0, 1.0], "vel": [0.0, 0.0],
                              "mob": v["Mobility"], "delay": v["Delay"], "acc": v["Acceleration"], "rad": v["Radius"]})
            self.settings.append({"s": s, "p": parts})

    def step(self, values: dict, dt: float) -> dict:
        out = {}
        for st in self.settings:
            s, p = st["s"], st["p"]
            npos, nang = s["Normalization"]["Position"], s["Normalization"]["Angle"]
            tx = ty = ang = 0.0
            for inp in s["Input"]:
                pid = inp["Source"]["Id"]
                pmin, pmax, _ = self.ranges[pid]
                w = inp["Weight"] / 100.0
                n = npos if inp["Type"] in ("X", "Y") else nang
                val = _norm(values.get(pid, 0.0), pmin, pmax, n["Minimum"], n["Maximum"], n["Default"], inp.get("Reflect", False)) * w
                if inp["Type"] == "X":
                    tx += val
                elif inp["Type"] == "Y":
                    ty += val
                else:
                    ang += val
            rad = math.radians(-ang)
            tx = tx * math.cos(rad) - ty * math.sin(rad)
            ty = tx * math.sin(rad) + ty * math.cos(rad)
            # ---- particles
            p[0]["pos"] = [tx, ty]
            tr = math.radians(ang)
            g = (math.sin(tr), math.cos(tr))
            thr = MOVEMENT_THRESHOLD * npos["Maximum"]
            for i in range(1, len(p)):
                q, prev = p[i], p[i - 1]
                force = (g[0] * q["acc"], g[1] * q["acc"])
                q["last"] = list(q["pos"])
                delay = q["delay"] * dt * 30.0
                d = [q["pos"][0] - prev["pos"][0], q["pos"][1] - prev["pos"][1]]
                r = _dir_to_rad(q["lastg"], g) / AIR_RESISTANCE
                d[0] = math.cos(r) * d[0] - d[1] * math.sin(r)
                d[1] = math.sin(r) * d[0] + d[1] * math.cos(r)
                q["pos"] = [prev["pos"][0] + d[0], prev["pos"][1] + d[1]]
                q["pos"][0] += q["vel"][0] * delay + force[0] * delay * delay
                q["pos"][1] += q["vel"][1] * delay + force[1] * delay * delay
                nd = [q["pos"][0] - prev["pos"][0], q["pos"][1] - prev["pos"][1]]
                ln = math.hypot(*nd) or 1.0
                q["pos"] = [prev["pos"][0] + nd[0] / ln * q["rad"], prev["pos"][1] + nd[1] / ln * q["rad"]]
                if abs(q["pos"][0]) < thr:
                    q["pos"][0] = 0.0
                if delay != 0.0:
                    q["vel"] = [(q["pos"][0] - q["last"][0]) / delay * q["mob"], (q["pos"][1] - q["last"][1]) / delay * q["mob"]]
                q["lastg"] = list(g)
            # ---- outputs
            for o in s["Output"]:
                i = o["VertexIndex"]
                if i < 1 or i >= len(p):
                    continue
                t = (p[i]["pos"][0] - p[i - 1]["pos"][0], p[i]["pos"][1] - p[i - 1]["pos"][1])
                if o["Type"] == "Angle":
                    pg = (p[i - 1]["pos"][0] - p[i - 2]["pos"][0], p[i - 1]["pos"][1] - p[i - 2]["pos"][1]) if i >= 2 else (0.0, 1.0)
                    val = _dir_to_rad(pg, t)
                else:
                    val = t[0] if o["Type"] == "X" else t[1]
                if o.get("Reflect", False):
                    val = -val
                pid = o["Destination"]["Id"]
                pmin, pmax, _ = self.ranges[pid]
                val = min(max(val * o["Scale"], pmin), pmax)
                w = o["Weight"] / 100.0
                out[pid] = val if w >= 1 else values.get(pid, 0.0) * (1 - w) + val * w
        return out
