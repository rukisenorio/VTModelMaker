"""段階 0: 1 枚の絵から新しいキャラのプロジェクトを作ります。

使い方: setup.py <project dir> <name> <picture> [--scale k]

<project dir>（まだないフォルダか、空のフォルダ）を作り、中に src/ work/ gen/{in,out,logs} layers/ out/ preview/
backups/ と、src/art.png（絵を RGBA の PNG にしたもの）、位置合わせの点がすべて空の character.json、位置合わせの点を
読み取るための目盛りの絵 2 枚を置きます: work/guide_full.png（絵の全体。50 px ごとに線、100 px ごとに数字）と
work/guide_head.png（頭のあたりを 3 倍に拡大。10 px ごとに線、50 px ごとに数字）。座標は元の絵の px です。
canvas_scale（prep.py が絵を何倍に拡大するか）は、キャンバスの高さが 6000〜9000 px になるように決めます。--scale で変えられます。
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import common as C

FONT = None


def font(size):
    import platform_env
    return platform_env.font(size, jp=False)


def guide(img, step, label, zoom=1, origin=(0, 0)):
    """A copy of img (zoomed) with a grid: thin lines every `step` source px, numbered every `label`."""
    im = Image.fromarray(img).convert("RGB")
    if zoom != 1:
        im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST if zoom >= 3 else Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    f = font(max(11, 9 + zoom * 3))
    ox, oy = origin
    x0 = (-ox) % step
    for x in range(int(x0), img.shape[1], step):
        sx = ox + x
        strong = sx % label == 0
        dr.line([(x * zoom, 0), (x * zoom, im.height)], fill=(255, 0, 80, 150 if strong else 60), width=1)
        if strong:
            dr.text((x * zoom + 2, 2), str(sx), fill=(200, 0, 60, 255), font=f)
            dr.text((x * zoom + 2, im.height - 16 - zoom * 2), str(sx), fill=(200, 0, 60, 255), font=f)
    y0 = (-oy) % step
    for y in range(int(y0), img.shape[0], step):
        sy = oy + y
        strong = sy % label == 0
        dr.line([(0, y * zoom), (im.width, y * zoom)], fill=(0, 120, 255, 150 if strong else 60), width=1)
        if strong:
            dr.text((2, y * zoom + 2), str(sy), fill=(0, 80, 220, 255), font=f)
            dr.text((im.width - 46, y * zoom + 2), str(sy), fill=(0, 80, 220, 255), font=f)
    return im


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    d, name, pic = os.path.abspath(os.path.expanduser(sys.argv[1])), sys.argv[2], os.path.expanduser(sys.argv[3])
    if os.path.exists(d) and os.listdir(d) and not os.path.exists(os.path.join(d, "gen")):
        sys.exit(f"{d} はもうあって、空ではありません。新しいフォルダを選んでください")
    for sub in ("src", "work", "gen/in", "gen/out", "gen/logs", "layers", "out", "preview", "backups"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    art = np.asarray(Image.open(pic).convert("RGBA"))
    Image.fromarray(art).save(os.path.join(d, "src", "art.png"))
    if os.path.abspath(pic) != os.path.abspath(os.path.join(d, "src", os.path.basename(pic))):
        shutil.copy2(pic, os.path.join(d, "src", "original" + os.path.splitext(pic)[1].lower()))
    h, w = art.shape[:2]
    k = 4 if h <= 2300 else 2 if h <= 4600 else 1
    if "--scale" in sys.argv:
        k = int(sys.argv[sys.argv.index("--scale") + 1])
    cfg_path = os.path.join(d, "character.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    cfg.update({"name": name, "src": "src/art.png", "source_size": [w, h], "canvas_scale": k})
    cfg.setdefault("points", {p: None for p in C.POINTS})
    cfg.setdefault("point_help", C.POINTS)
    cfg.setdefault("parts", {})          # extra parts (accessories...) - see references/new-character.md
    cfg.setdefault("options", {})
    with open(cfg_path, "w", encoding="utf-8") as f_:
        json.dump(cfg, f_, indent=1, ensure_ascii=False)
    rgb = art[..., :3].copy()
    if (art[..., 3] < 250).any():                       # (see-through: show it on mid grey)
        a = art[..., 3:4].astype(np.float32) / 255
        rgb = (rgb * a + 160 * (1 - a)).astype(np.uint8)
    guide(rgb, 50, 100).save(os.path.join(d, "work", "guide_full.png"))
    # the head region: the top 30 % of the picture's figure, at 3x (where the head is unknown yet: the upper part)
    ys, xs = np.nonzero(art[..., 3] > 0) if (art[..., 3] < 250).any() else np.nonzero((rgb.astype(int).sum(2) < 720))
    top = int(ys.min()) if len(ys) else 0
    fig_h = (int(ys.max()) - top) if len(ys) else h
    # (the head's region from the hair's width near the top: 2.4 widths down covers the head and the neck in a
    # full-body picture and in a waist-up one alike - a fixed share of the figure's height cut a waist-up head at the nose)
    fig_m = np.zeros((h, w), bool)
    fig_m[ys, xs] = True
    rows_top = fig_m[top:top + max(4, int(0.06 * fig_h))]
    cols_t = np.nonzero(rows_top.any(0))[0]
    w_top = (cols_t.max() - cols_t.min() + 1) if len(cols_t) else w // 3
    cx = int(np.median(cols_t)) if len(cols_t) else w // 2
    hh = int(np.clip(2.4 * w_top, 0.25 * fig_h, 0.55 * fig_h))
    half = max(80, int(0.6 * hh))
    hb = (max(0, cx - half), top, min(w, cx + half), min(h, top + hh))
    # (kept: the app zooms to it before the face's points are placed)
    cfg["head_guess"] = [int(v) for v in hb]
    with open(cfg_path, "w", encoding="utf-8") as f_:
        json.dump(cfg, f_, indent=1, ensure_ascii=False)
    guide(rgb[hb[1]:hb[3], hb[0]:hb[2]], 10, 50, zoom=3, origin=(hb[0], hb[1])).save(os.path.join(d, "work", "guide_head.png"))
    # the project's own copy of the kit (per-character changes go there, never into the skill's kit - they would change
    # every character) and of the 2x net
    kit_src = os.path.dirname(os.path.abspath(__file__))
    kit_dst = os.path.join(d, "kit")
    if not os.path.exists(kit_dst):
        shutil.copytree(kit_src, kit_dst, ignore=shutil.ignore_patterns("__pycache__", "app"))
        # (the files' hashes as copied: update_kit.py later replaces only the ones this project did not change)
        import update_kit
        update_kit.record(kit_dst)
    net = os.path.join(d, "work", "sr", "net.pt")
    if not os.path.exists(net):
        os.makedirs(os.path.dirname(net), exist_ok=True)
        for cand in (os.path.join(kit_src, "sr_net.pt"), os.path.join(os.path.dirname(kit_src), "assets", "sr_net.pt")):
            if os.path.exists(cand):
                shutil.copy2(cand, net)
                break
    print(f"プロジェクト {d}: 絵 {w}x{h}、canvas_scale {k}（キャンバス {w * k}x{h * k}）。キャラ専用の本体の写しを {kit_dst} に作りました")
    print("次に: work/guide_full.png と work/guide_head.png を見て character.json の 'points'（元の絵の px）を入れ、'describe' も書いてください。")
    print(f"そのあと: python3 {os.path.join(kit_dst, 'run.py')} {d} --to prep （囲まれた背景を選ぶ）、それから --from plates")


if __name__ == "__main__":
    main()
