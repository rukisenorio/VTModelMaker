"""A built model IN MOTION, as the user sees it in VTube Studio: a head shake with the model's own physics stepped at
30/s, frames at view size on a dark ground, side by side. Judge every build with this before calling anything done -
pose sheets and counts (holes, rest diff, check ok) hide what the motion shows: fraying hair, flickering bits, parts
that break up. (The user's verdict on three models that had passed counts and sheets: 「全部雑です、何をやってるんですか」)

usage: python3 motion_strip.py <build dir> <out.png> [--box x0,y0,x1,y1] [--scale 0.16] [--seconds 3.6] [--every 0.3]
                               [--compare <older build dir>] [--blink T] [--set Param=value,...]
  build dir   the folder with <name>.model3.json (out/<Name> of a project, or an installed copy)
  --box       canvas box to show (default: the head and shoulders, from the model's canvas: top 1/3, middle 55 %)
  --scale     view size (0.16-0.35 is what VTube Studio shows of a full-body 8000 px canvas on a 1080p screen)
  --compare   a second row with an older build at the same frames (before/after)
  --blink T   a blink at T s (shut in 0.08 s, 0.04 s shut, open in 0.14 s) and every frame from T-0.03 for 0.4 s at 30/s:
              the blink as the user sees it, at view size, while the head moves (frame by frame, not at chosen open values)
  --set       parameters held through the run (ParamEyeType=1 for a switch, ParamEyeLSmile=1,ParamEyeRSmile=1 ...)
"""
import os, sys, json, math, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                   # (the engine folder: l2d, platform_env)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from l2d import core
from l2d.render import Renderer
from l2d.physics import Physics

Image.MAX_IMAGE_PIXELS = None


def load(d):
    mj_path = sorted(glob.glob(os.path.join(d, "*.model3.json")))[0]
    mj = json.load(open(mj_path))
    tex = [np.asarray(Image.open(os.path.join(d, t)).convert("RGBA")) for t in mj["FileReferences"]["Textures"]]
    m = core.Model(os.path.join(d, mj["FileReferences"]["Moc"]))
    phys = mj["FileReferences"].get("Physics")
    return m, tex, (os.path.join(d, phys) if phys else None)


def blink_open(t, t0):
    """The eyes' openness through one blink starting at t0 (1 open .. 0 shut)."""
    if t0 is None or t < t0:
        return 1.0
    u = t - t0
    if u < 0.08:
        return 1.0 - u / 0.08
    if u < 0.12:
        return 0.0
    if u < 0.26:
        return (u - 0.12) / 0.14
    return 1.0


def frames(d, box, scale, seconds, every, label, blink=None, hold=None):
    m, tex, phys = load(d)
    ranges = {pid: (float(m.param_min[i]), float(m.param_max[i]), float(m.param_default[i])) for i, pid in enumerate(m.param_ids)}
    ph = Physics(phys, ranges) if phys else None
    R = Renderer(m, tex, scale=scale, crop=box)
    import platform_env
    F = platform_env.font(16)
    fps = 30
    out = []
    want = [every * (k + 1) for k in range(int(seconds / every))]
    if blink is not None:
        want = [blink - 0.03 + k / fps for k in range(12)]
        seconds = max(seconds, blink + 0.4)
    for i in range(int((seconds + 0.1) * fps)):
        t = i / fps
        # (a head shake as tracking gives it: turn, nod and lean out of phase, the body following a third)
        ax = 28 * math.sin(2 * math.pi * 0.6 * t)
        ay = 12 * math.sin(2 * math.pi * 0.45 * t + 1.0)
        az = 10 * math.sin(2 * math.pi * 0.5 * t + 0.5)
        v = {"ParamAngleX": ax, "ParamAngleY": ay, "ParamAngleZ": az, "ParamBodyAngleX": ax / 3, "ParamBodyAngleZ": az / 3,
             "ParamBreath": 0.5 + 0.5 * math.sin(2 * math.pi * 0.3 * t)}
        if blink is not None:
            v["ParamEyeLOpen"] = v["ParamEyeROpen"] = blink_open(t, blink)
        v.update(hold or {})
        if ph is not None:
            v.update(ph.step(v, 1.0 / fps))
        if any(abs(t - w) < 0.5 / fps for w in want):
            m.reset()
            m.set(**{k: val for k, val in v.items() if k in m.param_index})
            m.update()
            im = Image.fromarray(R.render(background=(30, 30, 38))[..., :3])
            tl = f"{t:.2f}秒" if blink is not None else f"{t:.1f}秒"
            ImageDraw.Draw(im).text((4, 2), f"{label} {tl}" if not out else tl, font=F, fill=(255, 220, 120))
            out.append(np.asarray(im))
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build"); ap.add_argument("out")
    ap.add_argument("--box"); ap.add_argument("--scale", type=float, default=0.16)
    ap.add_argument("--seconds", type=float, default=3.6); ap.add_argument("--every", type=float, default=0.3)
    ap.add_argument("--compare")
    ap.add_argument("--blink", type=float)
    ap.add_argument("--set")
    a = ap.parse_args()
    hold = {k: float(v) for k, v in (kv.split("=") for kv in a.set.split(","))} if a.set else None
    m0, _, _ = load(a.build)
    W, H = int(m0.canvas_size[0]), int(m0.canvas_size[1])
    box = tuple(int(v) for v in a.box.split(",")) if a.box else (int(0.22 * W), 0, int(0.78 * W), int(0.34 * H))
    rows = []
    runs = ([(a.compare, "前の版")] if a.compare else []) + [(a.build, "今の版" if a.compare else "")]
    for d, lab in runs:
        fr, _ = frames(d, box, a.scale, a.seconds, a.every, lab, a.blink, hold)
        per = 6 if len(fr) > 6 else len(fr)
        for k in range(0, len(fr), per):
            chunk = fr[k:k + per]
            while len(chunk) < per:
                chunk.append(np.full_like(fr[0], 20))
            rows.append(np.concatenate(chunk, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    print("保存しました:", a.out, Image.open(a.out).size)


if __name__ == "__main__":
    main()
