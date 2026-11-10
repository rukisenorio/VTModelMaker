"""How far every physics output swings under a typical head and body motion (and at rest, from the idle wind), stepped
at 30 and 60 per second as VTube Studio does per drawn frame: python3 phys_check.py <project>
A part whose output stays under ~0.15 does not visibly move at display size; one that sits at +-1 saturates."""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, preview
from l2d.physics import Physics

P = C.open_project()
L = preview.Look(P)
m = L.m
ranges = {pid: (float(m.param_min[i]), float(m.param_max[i]), float(m.param_default[i])) for i, pid in enumerate(m.param_ids)}
ph_path = os.path.join(L.d, L.mj["FileReferences"]["Physics"])
for fps in (30, 60):
    ph = Physics(ph_path, ranges)
    outs = sorted({o["Destination"]["Id"] for s_ in ph.j["PhysicsSettings"] for o in s_["Output"]})
    rec = {o: [] for o in outs}
    for phase, secs in (("motion", 10.0), ("rest", 6.0)):
        for i in range(int(secs * fps)):
            t = i / fps
            v = {k: r[2] for k, r in ranges.items() if k not in outs}
            if phase == "motion":
                v["ParamAngleX"] = 26 * math.sin(2 * math.pi * t / 4.0)
                v["ParamAngleY"] = 18 * math.sin(2 * math.pi * t / 3.1 + 1.0)
                v["ParamAngleZ"] = 10 * math.sin(2 * math.pi * t / 5.3)
                v["ParamBodyAngleX"] = v["ParamAngleX"] / 3; v["ParamBodyAngleZ"] = v["ParamAngleZ"] / 3
            v["ParamBreath"] = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3.4)
            v["ParamWind"] = 0.32 * math.sin(2 * math.pi * 2 * t / 8.0) + 0.22 * math.sin(2 * math.pi * 5 * t / 8.0 + 1.3)
            res = ph.step(v, 1.0 / fps)
            for o in outs:
                rec[o].append((phase, res.get(o, 0.0)))
    print(f"--- 1秒に {fps} 回計算: 出力  [動いている時 p5..p95, 最大|.|]  [止まっている時 p5..p95]")
    for o in outs:
        a = np.array([x for ph_, x in rec[o] if ph_ == "motion"]); b = np.array([x for ph_, x in rec[o] if ph_ == "rest"][fps:])
        flag = " 振り切れています" if np.mean(np.abs(a) > 0.98) > 0.05 else (" ほとんど動きません" if np.percentile(np.abs(a), 95) < 0.15 else "")
        print(f"   {o:22s} [{np.percentile(a, 5):+.2f}..{np.percentile(a, 95):+.2f}, {np.abs(a).max():.2f}]  [{np.percentile(b, 5):+.2f}..{np.percentile(b, 95):+.2f}]{flag}")
