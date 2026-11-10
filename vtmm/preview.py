"""Looks at a build: a sheet of poses, the rest pose against the picture, a short motion clip.

usage: preview.py <project> sheet [out.png] [--full]   poses at bust size on a dark ground (--full: the whole figure)
       preview.py <project> rest                       the rest pose vs the picture: work/rest_vs_art.png + the count
       preview.py <project> clip [out.mp4] [seconds]   head motion, blinks, talking, physics stepped at 30/s
       preview.py <project> zoom <x0,y0,x1,y1> "<pose>;<pose>;..." [out.png]   poses side by side in a canvas box, on a
                                                       dark ground (a pose: ParamAngleX=30,ParamHairFront=1 - empty = rest)
       preview.py <project> peel <x0,y0,x1,y1> "<pose>" [out.png]   that box in that pose with every drawable that covers
                                                       part of it shown alone - which part paints a defect
       preview.py <project> holes                      gaps: see-through pixels enclosed by the figure (0.25x, round the
                                                       head and shoulders) at rest and over 81 poses - the head turn and
                                                       nod x the back and front hair swings at -1/0/1 - the worst three
                                                       painted into preview/holes.png; and the body drawn alone at rest
                                                       above the chin (it should have nothing there)
The build is out/<name> (env L2D_MODEL_DIR to look at another one). Judge at view size first (method rule 7).
"""
import os, sys, json, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import common as C

POSES = [("正面", {}), ("首 X-30", {"ParamAngleX": -30}), ("首 X+30", {"ParamAngleX": 30}), ("見上げ Y+30", {"ParamAngleY": 30}),
         ("見下ろし Y-30", {"ParamAngleY": -30}), ("首かしげ Z+30", {"ParamAngleZ": 30}), ("首 X+30 Y+30", {"ParamAngleX": 30, "ParamAngleY": 30}),
         ("首 X-30 Y-30", {"ParamAngleX": -30, "ParamAngleY": -30}), ("まばたき 0.4", {"ParamEyeLOpen": 0.4, "ParamEyeROpen": 0.4}),
         ("閉じた目", {"ParamEyeLOpen": 0, "ParamEyeROpen": 0}), ("笑って閉じた目", {"ParamEyeLOpen": 0, "ParamEyeROpen": 0, "ParamEyeLSmile": 1, "ParamEyeRSmile": 1, "ParamMouthForm": 1}),
         ("視線 向かって左上", {"ParamEyeBallX": -1, "ParamEyeBallY": 1}), ("あ", {"ParamA": 1, "ParamMouthOpenY": 0.9}), ("い", {"ParamI": 1, "ParamMouthOpenY": 0.55}),
         ("う", {"ParamU": 1, "ParamMouthOpenY": 0.5}), ("お", {"ParamO": 1, "ParamMouthOpenY": 0.8}), ("笑顔で口を開く", {"ParamMouthOpenY": 1, "ParamMouthForm": 1}),
         ("眉を上げる", {"ParamBrowLY": 1, "ParamBrowRY": 1}), ("髪の揺れ +1", {"ParamHairFront": 1, "ParamHairFrontTip": 1, "ParamHairBack": 1, "ParamHairSideL": 1, "ParamHairSideR": 1}),
         ("髪の揺れ -1", {"ParamHairFront": -1, "ParamHairFrontTip": -1, "ParamHairBack": -1, "ParamHairSideL": -1, "ParamHairSideR": -1}),
         ("体 X+10 Z+10", {"ParamBodyAngleX": 10, "ParamBodyAngleZ": 10}), ("腕 +1", {"ParamArmL": 1, "ParamArmR": 1}),
         ("頬の赤み", {"ParamCheek": 1}), ("驚き", {"ParamIrisSmall": 1, "ParamMouthOpenY": 0.45, "ParamMouthForm": -0.7, "ParamBrowLY": 0.9, "ParamBrowRY": 0.9})]


class Look:
    def __init__(self, P, d=None):
        from l2d import core
        from l2d.render import Renderer
        self.d = d or os.environ.get("L2D_MODEL_DIR") or P.p("out", P.name)
        mj = json.load(open(os.path.join(self.d, f"{P.name}.model3.json")))
        self.mj = mj
        self.m = core.Model(os.path.join(self.d, mj["FileReferences"]["Moc"]))
        self.tex = [np.asarray(Image.open(os.path.join(self.d, t)).convert("RGBA")) for t in mj["FileReferences"]["Textures"]]
        self.R = Renderer
        self.rend = {}

    def render(self, vals, scale, crop):
        key = round(scale, 3)
        if key not in self.rend:
            self.rend = {key: self.R(self.m, self.tex, scale=key)}
        R = self.rend[key]
        R.crop = crop
        R.out_w, R.out_h = int(round((crop[2] - crop[0]) * key)), int(round((crop[3] - crop[1]) * key))
        self.m.reset(); self.m.set(**vals); self.m.update()
        return R.render(background=None)


def font(n):
    import platform_env
    return platform_env.font(n)


def sheet(P, out, full=False):
    L = Look(P)
    W, H = P.canvas_size()
    HX, HY, HR = P.head()
    if full:
        crop = (0, 0, W, H); sc = 900.0 / H
    else:
        crop = (HX - 3.2 * HR, HY - 2.0 * HR, HX + 3.2 * HR, HY + 4.4 * HR); sc = 380.0 / (crop[3] - crop[1])
    tiles = []
    f = font(16)
    for name, v in POSES:
        img = L.render(v, sc, crop)
        t = Image.fromarray(C.on_dark(img))
        ImageDraw.Draw(t).text((6, 4), name, fill=(255, 230, 0), font=f)
        tiles.append(np.asarray(t))
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("ポーズの確認画像:", out)


def rest(P):
    L = Look(P)
    W, H = P.canvas_size()
    s = 0.25
    img = L.render({}, s, (0, 0, W, H)).astype(np.float32)
    fig = C.load_rgba(P.p("work", "figure.png"))
    small = cv2.resize(fig, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    a1, a2 = img[..., 3:4] / 255, small[..., 3:4] / 255
    m1 = img[..., :3] * a1 + 40 * (1 - a1)
    m2 = small[..., :3] * a2 + 40 * (1 - a2)
    d = np.abs(m1 - m2).max(2)
    bad = d > 40
    print(f"正面の姿勢と元の絵の比べ（{s} 倍）: 色の差が 14 より大きい所 {int((d > 14).sum())} px、40 より大きい所 {int(bad.sum())} px（全部で {d.size} px）")
    vis = np.concatenate([m1, m2, np.repeat((np.clip(d * 3, 0, 255))[..., None], 3, 2)], 1).astype(np.uint8)
    Image.fromarray(vis).save(P.p("work", "rest_vs_art.png"))


def parse_pose(ps):
    v = {}
    for kv in ps.split(","):
        if kv.strip():
            k_, x_ = kv.split("=")
            v[k_.strip()] = float(x_)
    return v


def zoom(P, box, poses, out, height=560):
    """Poses side by side, each the canvas box at `height` px on a dark ground, labelled."""
    L = Look(P)
    tiles = []
    for ps in poses:
        img = L.render(parse_pose(ps), height / float(box[3] - box[1]), tuple(box))
        t = Image.fromarray(C.on_dark(img))
        ImageDraw.Draw(t).text((6, 4), ps or "正面", fill=(255, 230, 0), font=font(15))
        tiles.append(np.asarray(t))
    hh = min(t.shape[0] for t in tiles)
    Image.fromarray(np.concatenate([t[:hh] for t in tiles], 1)).save(out)
    print("確認画像:", out)


def peel(P, box, pose, out, scale=1.5, cols=5):
    """The box in one pose, then every drawable that paints part of it alone (a grid): who owns a defect."""
    from l2d.render import Renderer
    L = Look(P)
    v = parse_pose(pose)
    R = Renderer(L.m, L.tex, scale=scale, crop=tuple(box))
    L.m.reset(); L.m.set(**v); L.m.update()
    tiles, names = [C.on_dark(R.render(background=None))], ["全部"]
    for d in L.m.drawable_ids:
        img = R.render(background=None, only={d})
        if (img[..., 3] > 30).sum() > 20:
            tiles.append(C.on_dark(img)); names.append(d)
    out_t = []
    for t, n in zip(tiles, names):
        im = Image.fromarray(t); ImageDraw.Draw(im).text((4, 2), n, fill=(255, 230, 0), font=font(14)); out_t.append(np.asarray(im))
    while len(out_t) % cols:
        out_t.append(np.zeros_like(out_t[0]))
    rows = [np.concatenate(out_t[i:i + cols], 1) for i in range(0, len(out_t), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("確認画像:", out, names)


def holes(P):
    """The other session's review measure (2026-10-22): a pose is worse than rest by the see-through pixels enclosed by
    the figure that rest does not have - a part moved off what lay behind it and nothing was drawn there."""
    from scipy import ndimage
    L = Look(P)
    W, H = P.canvas_size()
    HX, HY, HR = P.head()
    chin = P.pt("chin")
    box = (max(0, int(HX - 3.2 * HR)), 0, min(W, int(HX + 3.2 * HR)), min(H, int(chin[1] + 1.4 * HR)))
    s = 0.25

    def enclosed(v):
        img = L.render(v, s, box)
        lab, n = ndimage.label(img[..., 3] <= 127)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        return img, np.isin(lab, [i for i in range(1, n + 1) if i not in border])
    img0, h0 = enclosed({})
    # (counted against rest: the picture's own page pockets - between fly-away strands, between the hair and an arm -
    # move and change shape with the hair, so a pose is judged by how much more it has, and by looking)
    res = []
    for ax in (-30, 0, 30):
        for ay in (-30, 0, 30):
            for hb in (-1, 0, 1):
                for hf in (-1, 0, 1):
                    v = {"ParamAngleX": ax, "ParamAngleY": ay, "ParamHairBack": hb, "ParamHairFront": hf, "ParamHairFrontTip": hf}
                    img, h = enclosed(v)
                    res.append((int(h.sum()) - int(h0.sum()), v, img, h))
    res.sort(key=lambda t: -t[0])
    worse = [r for r in res if r[0] > 6]
    print(f"すき間（囲まれた透明な所）: 正面で {int(h0.sum())} px（0.25 倍、範囲 {box}）。{len(res)} 通りの姿勢のうち {len(worse)} 通りで増えます。多いもの:")
    for d, v, _, _ in res[:5]:
        print(f"   +{d} {dict((k.replace('Param', ''), x) for k, x in v.items() if x)}")
    tiles = []
    for d, v, img, h in [(0, {}, img0, h0)] + res[:3]:
        a_ = img[..., 3:4].astype(np.float32) / 255
        t = (img[..., :3] * a_ + 40 * (1 - a_)).astype(np.uint8)
        t[h] = (255, 0, 200)
        im = Image.fromarray(t)
        short = {"ParamAngleX": "首X", "ParamAngleY": "首Y", "ParamHairBack": "後髪", "ParamHairFront": "前髪", "ParamHairFrontTip": "毛先"}
        ImageDraw.Draw(im).text((5, 3), (" ".join(f"{short.get(k, k.replace('Param', ''))}{x:+g}" for k, x in v.items() if x) or "正面")
                                + f"  +{d}", font=font(15), fill=(255, 220, 120))
        tiles.append(np.asarray(im))
    os.makedirs(P.p("preview"), exist_ok=True)
    Image.fromarray(np.concatenate(tiles, 1)).save(P.p("preview", "holes.png"))
    # the body alone at rest above the chin: a piece of the head left in it stays behind when the head moves
    bl = [d_ for d_ in L.m.drawable_ids if d_ == "body"]
    if bl:
        from l2d.render import Renderer
        R = Renderer(L.m, L.tex, scale=0.5, crop=(0, 0, W, int(chin[1] - 0.3 * HR)))
        L.m.reset(); L.m.update()
        nb = int((R.render(background=None, only=set(bl))[..., 3] > 40).sum() * 4)
        print(f"CHECK 正面の姿勢で体だけを描いたとき、あごより上に出る所: キャンバスで {nb} px（首まわりの服のほかは 0 のはず）")


def clip(P, out, secs=8.0):
    from l2d.physics import Physics
    L = Look(P)
    W, H = P.canvas_size()
    HX, HY, HR = P.head()
    crop = (HX - 3.2 * HR, HY - 2.0 * HR, HX + 3.2 * HR, HY + 4.4 * HR)
    sc = 720.0 / (crop[3] - crop[1])
    m = L.m
    ranges = {pid: (float(m.param_min[i]), float(m.param_max[i]), float(m.param_default[i])) for i, pid in enumerate(m.param_ids)}
    ph = Physics(os.path.join(L.d, L.mj["FileReferences"]["Physics"]), ranges) if "Physics" in L.mj["FileReferences"] else None
    outs = {o["Destination"]["Id"] for s_ in ph.j["PhysicsSettings"] for o in s_["Output"]} if ph else set()
    FPS = 30
    vals = {k: r[2] for k, r in ranges.items()}
    frames = []
    for i in range(int(-2 * FPS), int(secs * FPS)):
        t = max(i, 0) / FPS
        v = {k: r[2] for k, r in ranges.items() if k not in outs}
        v["ParamAngleX"] = 26 * math.sin(2 * math.pi * t / 4.0)
        v["ParamAngleY"] = 18 * math.sin(2 * math.pi * t / 3.1 + 1.0)
        v["ParamAngleZ"] = 10 * math.sin(2 * math.pi * t / 5.3)
        v["ParamBodyAngleX"] = v["ParamAngleX"] / 3; v["ParamBodyAngleZ"] = v["ParamAngleZ"] / 3
        v["ParamBreath"] = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3.4)
        bl = min(abs(((t + 0.7) % 3.3) - 0.12) / 0.12, 1.0)
        v["ParamEyeLOpen"] = v["ParamEyeROpen"] = bl
        talk = max(0.0, math.sin(2 * math.pi * t * 4.3) * math.sin(2 * math.pi * t * 0.7))
        v["ParamMouthOpenY"] = talk * 0.8
        v["ParamMouthForm"] = 0.3
        for k in v:
            v[k] = min(max(v[k], ranges[k][0]), ranges[k][1])
        vals.update(v)
        if ph:
            vals.update(ph.step(vals, 1.0 / FPS))
        if i >= 0:
            frames.append(C.on_dark(L.render(vals, sc, crop)))
    h, w = frames[0].shape[:2]
    w2, h2 = w // 2 * 2, h // 2 * 2
    import platform_env
    ff = platform_env.ffmpeg_bin()
    if not ff:
        print("ffmpeg が見つからないので動画は作りません（絵はできています）")
        return
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr[:h2, :w2]).tobytes())
    p.stdin.close(); p.wait()
    print(f"動画: {out}（{len(frames)} コマ）")


def main():
    P = C.open_project()
    mode = sys.argv[2] if len(sys.argv) > 2 else "sheet"
    if mode == "sheet":
        sheet(P, sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else P.p("preview", "poses.png"), "--full" in sys.argv)
    elif mode == "rest":
        rest(P)
    elif mode == "holes":
        holes(P)
    elif mode in ("zoom", "peel"):
        box = [int(float(q)) for q in sys.argv[3].split(",")]
        arg = sys.argv[4] if len(sys.argv) > 4 else ""
        out = sys.argv[5] if len(sys.argv) > 5 else P.p("preview", f"{mode}.png")
        if mode == "zoom":
            zoom(P, box, arg.split(";"), out)
        else:
            peel(P, box, arg, out)
    elif mode == "clip":
        clip(P, sys.argv[3] if len(sys.argv) > 3 else P.p("preview", "clip.mp4"), float(sys.argv[4]) if len(sys.argv) > 4 else 8.0)


if __name__ == "__main__":
    main()
