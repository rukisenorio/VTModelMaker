"""Stage 1: the canvas and the cut-out.

usage: prep.py <project>

  work/canvas.png   the picture super-resolved by canvas_scale (sr2x: a small net that learned "lines of this kind of
                    drawing at twice the size" - 2x per pass, so 4 = two passes; cubic for odd factors)
  work/figure.png   RGBA cut-out of the whole figure on the canvas, edges unmixed from the page (no pale fringe on a
                    dark ground), work/bg.png (255 = page)
  work/pockets.png  the enclosed pieces of page-coloured area NOT reached from the picture's border, numbered: page
                    between an arm and the body, between strands of hair - or white clothing / the whites of the eyes!
                    Colour alone cannot tell them apart. List the ones that are page in character.json
                    "bg_pockets": [[x, y], ...] (source px of a point inside each) and run prep.py again.
  work/figure_look.png  the cut-out on a dark and on a loud ground (look for white rims, holes, page left inside)

A picture with transparency keeps its own alpha (nothing is flooded). A coloured page should be removed beforehand (a
transparent PNG is best); a light page (near white or light grey) is handled here.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import common as C


def page_colour(img):
    b = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    return np.median(b, 0)


def page_1x(img, P, pockets, tol=20.0):
    """The page at the source's size: what is close to the page colour and joined to the border (a 5 px disc cannot slip
    through an outline's one-pixel breaks there), plus the pockets named in the config."""
    f = img.astype(np.float32)
    d = np.abs(f - P[None, None, :]).max(2)
    pagey = d <= tol
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(pagey.astype(np.uint8), disc)
    n, lab = cv2.connectedComponents(core, connectivity=4)
    ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    reach = np.isin(lab, ids[ids > 0])
    bg = cv2.dilate(reach.astype(np.uint8), disc).astype(bool) & pagey
    n2, lab2 = cv2.connectedComponents((pagey & ~bg).astype(np.uint8), connectivity=4)
    used = []
    for x, y in pockets:
        k = lab2[int(round(y)), int(round(x))]
        if k > 0:
            bg |= lab2 == k
            used.append((x, y))
        else:
            print("  囲まれた背景の点", (x, y), "は背景の色の所にないので、何も取りません")
    # candidates for the report: enclosed page-coloured pieces of some size
    sizes = ndimage.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
    cands = []
    for i, s in enumerate(sizes):
        if s >= 0.00004 * img.shape[0] * img.shape[1] and not (bg & (lab2 == i + 1)).any():
            ys, xs = np.nonzero(lab2 == i + 1)
            j = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
            cands.append((int(xs[j]), int(ys[j]), int(s)))
    return bg, used, cands


def main():
    P_ = C.open_project()
    cfg = P_.cfg
    k = int(cfg.get("canvas_scale", 4))
    art = C.load_rgba(P_.p(cfg.get("src", "src/art.png")))
    has_alpha = (art[..., 3] < 250).any()
    rgb1 = art[..., :3].copy()
    import sr2x
    # ---- the canvas
    out = P_.p("work", "canvas.png")
    if not os.path.exists(out) or "--redo" in sys.argv:
        img = art if has_alpha else rgb1
        f = 1
        while f * 2 <= k:
            img = sr2x.up(img)
            f *= 2
        if f != k:
            img = cv2.resize(img, (art.shape[1] * k, art.shape[0] * k), interpolation=cv2.INTER_CUBIC)
        C.save_rgba(out, img if img.shape[2] == 4 else np.dstack([img, np.full(img.shape[:2], 255, np.uint8)]))
        print("キャンバス:", img.shape[1], "x", img.shape[0])
    big4 = C.load_rgba(out)
    big = big4[..., :3]
    H, W = big.shape[:2]
    # ---- the cut-out
    import layerio
    if has_alpha:
        fig = big4[..., 3] > 0
        bg = ~fig
        rgba = big4.copy()
        cands = []
        print("絵に透明な部分があるので、その透明さをそのまま使います")
    else:
        Pc = page_colour(rgb1)
        print("背景の色", Pc.round(1))
        # (a light page, not exactly white: brought to white so that the edge unmixing below works against white)
        gain = 255.0 / np.maximum(Pc, 1.0)
        wb = lambda im: np.clip(im.astype(np.float32) * gain[None, None, :], 0, 255).astype(np.uint8)
        one, bigw = wb(rgb1), wb(big)
        pockets = [tuple(p) for p in cfg.get("bg_pockets", [])]
        bg1, used, cands = page_1x(one, np.array([255.0, 255, 255]), pockets)
        bgk = cv2.resize(bg1.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sure = cv2.erode(bgk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 5, 2 * k + 5))).astype(bool)
        fb = bigw.astype(np.int16)
        mn, mx = fb.min(2), fb.max(2)
        white = (mn >= 236) & (mx - mn <= 16)
        n, lab = cv2.connectedComponents(white.astype(np.uint8), connectivity=4)
        ids = np.unique(lab[sure & white])
        bg = np.isin(lab, ids[ids > 0]) | sure
        bg &= cv2.dilate(bgk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * k + 1, 4 * k + 1))).astype(bool)
        # (the light rim between page and outline - source colour fringes - also page, but only a few px deep and stopped
        # by the outline)
        lum = bigw.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        light = (lum > 182) & (mx - mn <= 95)
        RIM = 2 * k + 1
        dist = ndimage.distance_transform_edt(~bg)
        band = light & (dist <= RIM)
        reach = bg.copy()
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(RIM + 2):
            nxt = cv2.dilate(reach.astype(np.uint8), k3).astype(bool) & band & ~reach
            if not nxt.any():
                break
            reach |= nxt
        bg = reach
        fig = ~bg
        lab3, n3 = ndimage.label(fig)
        sizes = ndimage.sum(fig, lab3, range(1, n3 + 1))
        keep = [i + 1 for i in range(n3) if sizes[i] >= 20 * k * k]
        fig &= np.isin(lab3, keep)
        bg = ~fig
        rgba = layerio.cutout(bigw, fig, bg=bg, r=k, soften=0.8)
        # (back to the picture's own colours inside: only the page was white-balanced for the unmixing)
        inside = rgba[..., 3] == 255
        rgba[inside, :3] = big[inside]
        print(f"背景はキャンバスの {100 * bg.mean():.1f} %。囲まれた背景を {len(used)} か所取りました。キャラは {len(keep)} 個のかたまりです")
    C.save_rgba(P_.p("work", "figure.png"), rgba)
    Image.fromarray((bg * 255).astype(np.uint8)).save(P_.p("work", "bg.png"))
    # ---- the looks
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    dark = (rgba[..., :3] * a + np.array([40, 42, 52]) * (1 - a)).astype(np.uint8)
    loud = (rgba[..., :3] * a + np.array([0, 220, 80]) * (1 - a)).astype(np.uint8)
    s = max(1, H // 1600)
    sm = lambda im: cv2.resize(im, (W // s, H // s), interpolation=cv2.INTER_AREA)
    Image.fromarray(np.concatenate([sm(dark), sm(loud)], 1)).save(P_.p("work", "figure_look.png"))
    # the pocket candidates, numbered on the source
    rep = Image.fromarray(rgb1.copy())
    from PIL import ImageDraw
    dr = ImageDraw.Draw(rep)
    for i, (x, y, s_) in enumerate(cands):
        dr.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 0, 80), width=2)
        dr.text((x + 8, y - 8), str(i), fill=(255, 0, 80))
    rep.save(P_.p("work", "pockets.png"))
    json.dump([{"i": i, "point": [x, y], "area": s_} for i, (x, y, s_) in enumerate(cands)], open(P_.p("work", "pockets.json"), "w"), indent=1)
    # ... and each one close up (3x, the piece tinted): page between strands / arm and body, or white artwork?
    if cands:
        f1 = np.abs(rgb1.astype(np.float32) * (255.0 / np.maximum(page_colour(rgb1), 1.0)) - 255.0).max(2) <= 20.0
        n2, lab2 = cv2.connectedComponents(f1.astype(np.uint8), connectivity=4)
        R, Z, cols = 40, 3, 8
        tiles = []
        for i, (x, y, s_) in enumerate(cands):
            x0, y0 = max(0, x - R), max(0, y - R)
            crop = rgb1[y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.float32)
            piece = (lab2 == lab2[y, x])[y0:y0 + 2 * R, x0:x0 + 2 * R]
            crop[piece] = crop[piece] * 0.45 + np.array([255, 0, 160]) * 0.55
            t = np.full((2 * R, 2 * R, 3), 90, np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop.astype(np.uint8)
            ti = Image.fromarray(t).resize((2 * R * Z, 2 * R * Z), Image.NEAREST)
            ImageDraw.Draw(ti).text((4, 2), f"{i} ({x},{y})", fill=(255, 255, 0))
            tiles.append(np.asarray(ti))
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)]
        Image.fromarray(np.concatenate(rows, 0)).save(P_.p("work", "pockets_sheet.png"))
    print(f"背景の色で囲まれた所が {len(cands)} か所、背景にしていません（work/pockets.png）。本当に背景のものは bg_pockets に加えてください")


if __name__ == "__main__":
    main()
