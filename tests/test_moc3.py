"""Smoke tests for the moc3 engine - no VTube Studio needed (the Cubism Core check runs only when one is found).

usage: python -X utf8 tests/test_moc3.py      (or: python -m pytest tests)
A tiny rig - one picture on a warp turned by a parameter, one on a rotation - is compiled; the files must exist, the
.moc3 must parse back with the same counts, and, when a Cubism Core is installed, pass its consistency check and move.
"""
import os, sys, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vtmm"))
import numpy as np
from l2d.rig import Rig
from l2d import moc3


def tiny_model(out_dir):
    W = H = 512
    rig = Rig(W, H)
    rig.param("ParamAngleX", -30, 30, 0, "Angle X")
    rig.param("ParamSwing", -1, 1, 0, "Swing")
    rig.part("PartMain", "main")
    img = np.zeros((200, 200, 4), np.uint8)
    img[20:180, 20:180] = (230, 120, 90, 255)
    rig.layer("square", img, 156, 100)
    disc = np.zeros((80, 80, 4), np.uint8)
    yy, xx = np.mgrid[0:80, 0:80]
    disc[(yy - 40) ** 2 + (xx - 40) ** 2 < 38 ** 2] = (60, 90, 200, 255)
    rig.layer("disc", disc, 216, 330)

    def turn(pts, v):
        out = pts.copy()
        out[:, 0] += 40.0 * v["ParamAngleX"] / 30.0 * (1.0 - abs(pts[:, 1] - 200) / 300.0)
        return out
    w = rig.warp("W_Turn", None, "PartMain", (120, 80, 400, 320), 6, 6, axes=[("ParamAngleX", [-30, 0, 30])], fn=turn)
    rig.layer_mesh("square", w, "PartMain", z=1.0, cell=40)
    r = rig.rot("R_Swing", None, "PartMain", (256, 330), axes=[("ParamSwing", [-1, 0, 1])], angle=lambda v: 20.0 * v["ParamSwing"])
    rig.layer_mesh("disc", r, "PartMain", z=2.0, cell=20)
    rig.expressions["Tilt"] = {"params": {"ParamAngleX": 15.0}}
    return rig.compile(out_dir, "Tiny", atlas_size=1024)


def test_compile_and_parse():
    with tempfile.TemporaryDirectory() as d:
        moc_path = tiny_model(d)
        assert os.path.exists(moc_path)
        mj = json.load(open(os.path.join(d, "Tiny.model3.json"), encoding="utf-8"))
        for f in [mj["FileReferences"]["Moc"]] + mj["FileReferences"]["Textures"]:
            assert os.path.exists(os.path.join(d, f)), f
        m = moc3.parse(moc_path)
        counts = m["counts"] if "counts" in m else m.get("count", {})
        assert counts, "no counts in the parsed moc3"
        assert counts.get("params") == 2 and counts.get("art") == 2, counts
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vtmm"))
            import platform_env
            if not platform_env.core_lib():
                print("no Cubism Core on this computer - the Core check is skipped")
                return
            from l2d import core
            model = core.Model(moc_path)
            model.reset(); model.update()
            rest = model.positions(0).copy()
            model.reset(); model.set(ParamAngleX=30.0); model.update()
            moved = model.positions(0)
            assert np.isfinite(moved).all() and np.abs(moved - rest).max() > 0, "the warp did not move the mesh"
            print("Cubism Core: consistent, and the mesh moves")
        except ImportError:
            pass


if __name__ == "__main__":
    test_compile_and_parse()
    print("ok")
