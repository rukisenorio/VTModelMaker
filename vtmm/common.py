"""Shared bits of the kit: the project folder and its character.json, landmarks in canvas px, small image helpers.

Every stage runs as `python3 <kit>/<stage>.py <project dir> [args]`. open_project() also puts the project folder into the
environment (L2D_PROJECT), so that the helper modules taken over from the characters (layerio, sr2x, mouth_rig) work on
it. Coordinates in character.json are SOURCE pixels (what one reads off src/art.png and its guide pictures); the
stages work in CANVAS pixels = source x canvas_scale (the picture super-resolved, tools: prep.py).
"""
import os, sys, json, math
KIT = os.path.dirname(os.path.abspath(__file__))
if KIT not in sys.path:
    sys.path.insert(0, KIT)
import numpy as np

# the landmarks a character needs (source px of src/art.png); L = the viewer's left (the character's right)
POINTS = {
    "hair_top": "髪のいちばん上の点（帽子をかぶっていれば帽子のいちばん上）",
    "hair_bottom": "髪のいちばん下の毛先（短い髪なら null。そのときは、あごから頭の半径の約4倍下とみなす）",
    "crown": "髪の下の頭蓋骨のてっぺん（髪のてっぺんではない。見当でよい: 髪がなかったら頭の上の端がくる所）",
    "brow_l": "向かって左の眉の真ん中", "brow_r": "向かって右の眉の真ん中",
    "eye_l": "向かって左の目の中心（黒目の中心）", "eye_r": "向かって右の目の中心（黒目の中心）",
    "nose": "鼻の先", "mouth": "口の線の真ん中",
    "chin": "あごのいちばん下の点（顔の輪郭線の上）",
    "face_l": "目の高さでの、向かって左の顔の輪郭", "face_r": "目の高さでの、向かって右の顔の輪郭",
    "ear_l": "向かって左の耳の真ん中（隠れていれば null）", "ear_r": "向かって右の耳の真ん中（隠れていれば null）",
    "neck": "首の前側で、襟と接する所（その真ん中）",
    "shoulder_l": "向かって左の肩の付け根（関節）", "shoulder_r": "向かって右の肩の付け根（関節）",
    "elbow_l": "向かって左のひじ", "elbow_r": "向かって右のひじ",
    "wrist_l": "向かって左の手首", "wrist_r": "向かって右の手首",
    "hip": "腰の真ん中（股の高さ）",
    "knee_l": "向かって左のひざ", "knee_r": "向かって右のひざ",
    "foot_l": "向かって左の足の裏", "foot_r": "向かって右の足の裏",
}
TEMPLATE_HR = 420.0      # the head radius (canvas px) the rig's distances were tuned for (the second character)


def sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def open_project(argv=None):
    argv = sys.argv if argv is None else argv
    # (Windows: every stage runs in Python's UTF-8 mode - else text files default to the locale's encoding, and the
    # Japanese in character.json, the prompts and the VTube Studio settings would be written in one encoding and read in
    # another. Re-run this stage in UTF-8 mode and hand on its exit code)
    if os.name == "nt" and not sys.flags.utf8_mode and argv is sys.argv and not os.environ.get("VTMM_NO_RERUN"):
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-X", "utf8"] + sys.argv, env=dict(os.environ, VTMM_NO_RERUN="1")))
    if len(argv) < 2:
        sys.exit(f"使い方: {os.path.basename(argv[0])} <project dir> ...")
    d = os.path.abspath(os.path.expanduser(argv[1]))
    os.environ["L2D_PROJECT"] = d
    return Project(d)


class Project:
    def __init__(self, d):
        self.d = d
        self.cfg_path = os.path.join(d, "character.json")
        self.cfg = json.load(open(self.cfg_path, encoding="utf-8")) if os.path.exists(self.cfg_path) else {}

    def p(self, *parts):
        return os.path.join(self.d, *parts)

    def save_cfg(self):
        with open(self.cfg_path, "w", encoding="utf-8") as f_:
            json.dump(self.cfg, f_, indent=1, ensure_ascii=False)

    @property
    def name(self):
        return self.cfg["name"]

    @property
    def k(self):
        return float(self.cfg.get("canvas_scale", 4))

    @property
    def kpx(self):
        """The pixel unit for tolerances (line widths, registration slack): the super-resolution factor - but never less
        than the canvas height / 2000. A picture given at full size has canvas_scale 1, and every tolerance had been four
        times smaller than on an upscaled one of the same canvas size (a red jaw line stayed in the hair, masks frayed)."""
        try:
            h = self.canvas_size()[1]
        except Exception:
            h = 0
        return max(self.k, h / 2000.0)

    def pt(self, key, default=None):
        """A landmark in canvas px (None if not given)."""
        v = self.cfg.get("points", {}).get(key)
        if v is None:
            return default
        return np.array(v, float) * self.k

    def need(self, *keys):
        miss = [k for k in keys if self.cfg.get("points", {}).get(k) is None]
        if miss:
            sys.exit("character.json に位置合わせの点が足りません: " + ", ".join(miss) + "（work/guide_*.png を見て入れてください）")

    def canvas_size(self):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        return Image.open(self.p("work", "canvas.png")).size

    # ---- the head, from the landmarks (refined by measure.py into work/geom.json when it has run)
    def geom(self):
        g = self.p("work", "geom.json")
        return json.load(open(g)) if os.path.exists(g) else {}

    def head(self):
        """(HX, HY, HR): the head's vertical axis, the skull's middle and radius, canvas px."""
        g = self.geom()
        if "head" in g:
            return tuple(g["head"])
        fl, fr, crown, chin = self.pt("face_l"), self.pt("face_r"), self.pt("crown"), self.pt("chin")
        hx = 0.5 * (fl[0] + fr[0])
        hr = 0.5 * (fr[0] - fl[0]) * 1.04
        hy = crown[1] + hr
        return float(hx), float(hy), float(hr)

    def unit(self):
        """Canvas px of this character per canvas px of the template (distances tuned on the second character)."""
        return self.head()[2] / TEMPLATE_HR


def load_rgba(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("RGBA")).copy()


def save_rgba(path, a):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.ascontiguousarray(a)).save(path, compress_level=3)


def box_around(c, half_w, half_h, W, H):
    x0, y0 = int(max(0, c[0] - half_w)), int(max(0, c[1] - half_h))
    x1, y1 = int(min(W, c[0] + half_w)), int(min(H, c[1] + half_h))
    return x0, y0, x1, y1


def on_dark(rgba, bg=(30, 30, 36)):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(np.array(bg, np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)


def hull_mask(m):
    """The convex hull of a mask, filled (an eye's outline across the strands of hair that cut it into pieces - a closing
    cannot bridge a strand wider than itself)."""
    import cv2
    ys, xs = np.nonzero(m)
    out = np.zeros(m.shape, np.uint8)
    if len(xs) < 3:
        return out.astype(bool)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    cv2.fillPoly(out, [hull], 1)
    return out.astype(bool)
