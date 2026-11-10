"""Layer storage: RGBA crops positioned on the canvas + a JSON manifest."""
import json, os
import numpy as np, cv2
from PIL import Image

# (the kit: the project folder comes from the environment - common.open_project sets it)
ROOT = os.environ.get("L2D_PROJECT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LDIR = os.path.join(ROOT, "layers")
MAN = os.path.join(LDIR, "manifest.json")


def _load():
    return json.load(open(MAN)) if os.path.exists(MAN) else {}


def save_layer(name, rgba, x0, y0, pad=2, crop=True):
    """Crop to the alpha bbox (unless crop=False) and store. rgba: (h,w,4) uint8 at canvas (x0,y0)."""
    a = rgba[..., 3]
    ys, xs = np.nonzero(a > 0)
    if len(ys) == 0:
        raise ValueError(f"レイヤー {name} が空です")
    by0, by1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, rgba.shape[0])
    bx0, bx1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, rgba.shape[1])
    if not crop:
        by0, by1, bx0, bx1 = 0, rgba.shape[0], 0, rgba.shape[1]
    c = rgba[by0:by1, bx0:bx1]
    Image.fromarray(c).save(os.path.join(LDIR, name + ".png"), compress_level=3)
    man = _load()
    man[name] = {"x0": int(x0 + bx0), "y0": int(y0 + by0), "w": int(c.shape[1]), "h": int(c.shape[0])}
    json.dump(man, open(MAN, "w"), indent=1)
    return man[name]


def load_layer(name):
    man = _load()
    e = man[name]
    return np.asarray(Image.open(os.path.join(LDIR, name + ".png")).convert("RGBA")), e["x0"], e["y0"]


def drop_layer(name):
    """Forget a layer an earlier run made and this one does not (its png too): left in the manifest, the build kept
    rigging it - a brow made of eyelashes that stayed up when the eye closed."""
    man = _load()
    if name in man:
        man.pop(name)
        json.dump(man, open(MAN, "w"), indent=1)
    f = os.path.join(LDIR, name + ".png")
    if os.path.exists(f):
        os.remove(f)


def manifest():
    return _load()


def feather(mask, sigma=1.2):
    """Soft alpha (0..255) from a boolean mask."""
    m = mask.astype(np.float32)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rgba(rgb, alpha):
    return np.dstack([rgb, alpha]).astype(np.uint8)


def fill_from(rgb, known, target, iters=400):
    """Extend colours of `known` pixels into `target` pixels (nearest-colour + smoothing)."""
    from scipy import ndimage
    known = known.astype(bool)
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    near = rgb[idx[0], idx[1]]
    out = rgb.copy()
    t = target & ~known
    out[t] = near[t]
    # soften the Voronoi streaks inside the filled area only
    blur = cv2.GaussianBlur(out, (0, 0), 6)
    w = cv2.GaussianBlur(t.astype(np.float32), (0, 0), 3)[..., None]
    w = np.where(t[..., None], np.clip(w, 0.0, 1.0), 0.0)
    return np.clip(out * (1 - w) + blur * w + 0.5, 0, 255).astype(np.uint8)


def composite(names, box, bg=(255, 255, 255)):
    """Alpha-composite stored layers (in the given order, back to front) inside canvas box."""
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    out[:] = bg
    for n in names:
        img, lx, ly = load_layer(n)
        h, w = img.shape[:2]
        ix0, iy0, ix1, iy1 = max(x0, lx), max(y0, ly), min(x1, lx + w), min(y1, ly + h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        src = img[iy0 - ly:iy1 - ly, ix0 - lx:ix1 - lx].astype(np.float32)
        a = src[..., 3:4] / 255.0
        dst = out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
        dst[:] = src[..., :3] * a + dst * (1 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def defringe(img, r=3, margin=10.0):
    """Remove the white background halo baked into the rim of a cut-out layer: rim pixels that
    are brighter than the nearest solid interior pixel take that interior colour instead."""
    from scipy import ndimage
    a = img[..., 3]
    solid = a >= 250
    core = cv2.erode(solid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    if core.sum() < 50:
        return img
    idx = ndimage.distance_transform_edt(~core, return_distances=False, return_indices=True)
    near = img[idx[0], idx[1], :3].astype(np.float32)
    rgb = img[..., :3].astype(np.float32)
    w = np.array([0.299, 0.587, 0.114], np.float32)
    rim = (a > 0) & ~core
    brighter = (rgb @ w) > (near @ w) + margin
    out = img.copy()
    sel = rim & brighter
    out[..., :3][sel] = near[sel].astype(np.uint8)
    return out


def matte_white(rgb, mask, bg, r=3):
    """RGBA for a cut-out whose outer rim was anti-aliased against a white background.

    rgb: source colours; mask: layer region; bg: pixels known to be pure background.
    Rim pixels near the background get the nearest solid colour and an alpha recovered from how
    far they sit between that colour and white, so the layer carries no white halo.
    """
    from scipy import ndimage
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    m = mask & ~bg
    near_bg = cv2.dilate(bg.astype(np.uint8), k(r)).astype(bool)
    grow = cv2.dilate(m.astype(np.uint8), k(2)).astype(bool) & ~bg & near_bg     # AA tail just outside the mask
    region = m | grow
    solid = m & ~near_bg
    alpha = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) * 255.0, 0, 255)
    out_rgb = rgb.copy()
    rim = region & near_bg
    if solid.sum() > 20 and rim.any():
        idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        near = rgb[idx[0], idx[1]].astype(np.float32)
        p = rgb.astype(np.float32)
        dp = (255.0 - p).sum(2)
        df = np.maximum((255.0 - near).sum(2), 45.0)
        a = np.clip(dp / df, 0, 1) * 255.0
        alpha[rim] = a[rim]
        out_rgb[rim] = near[rim].astype(np.uint8)
    alpha[bg] = 0
    alpha[~region] = np.minimum(alpha[~region], 0)
    return np.dstack([out_rgb, np.clip(alpha + 0.5, 0, 255).astype(np.uint8)])


def white_bg(rgb, thr=250, borders="tlrb"):
    """Background = near-white pixels connected to the chosen image borders."""
    white = (rgb.min(2) >= thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(white, connectivity=4)
    ids = []
    if "t" in borders: ids.append(lab[0])
    if "b" in borders: ids.append(lab[-1])
    if "l" in borders: ids.append(lab[:, 0])
    if "r" in borders: ids.append(lab[:, -1])
    ids = np.unique(np.concatenate(ids))
    return np.isin(lab, ids[ids > 0])


def cutout(rgb, mask, bg=None, r=3, soften=0.7, light=None):
    """RGBA cut-out with a decontaminated rim.

    bg: pixels that are true background (white). They are removed from the mask. The layer's
    pixels right next to the background are anti-aliased against white; each one is rewritten as
    "local line colour x coverage", the line colour being the least white pixel of the layer
    within 3 px. On white the layer looks exactly as drawn, and on any other page thin strokes
    and outlines keep their colour instead of carrying a pale fringe. Light artwork is never
    treated as background, and rims that border other artwork are left alone.

    light: artwork outside the layer that is nearly white (a silver tuft behind an ear's outline).
    Edge pixels of the layer next to it get the same treatment - they are the outline blended with
    that white, and would stay behind as a pale fringe when the light thing moves away - but only
    where they really are such a blend; nothing else changes.
    """
    from scipy import ndimage
    ell = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    m = mask.astype(bool)
    if bg is None:
        bg = np.zeros_like(m)
    m = m & ~bg
    H_, W_ = m.shape
    near_bg = cv2.dilate(bg.astype(np.uint8), ell(r)).astype(bool)
    solid = m & ~near_bg
    alpha = m.astype(np.float32) * 255.0
    out = rgb.copy()
    done = np.zeros_like(m)
    edge = m & cv2.dilate(bg.astype(np.uint8), ell(2)).astype(bool)        # the anti-aliased band itself
    edge2 = np.zeros_like(m)
    if light is not None:
        edge2 = m & cv2.dilate((light & ~m).astype(np.uint8), ell(2)).astype(bool) & ~edge
    if edge.any() or edge2.any():
        white_dist = 765.0 - rgb.astype(np.float64).sum(2)                 # 0 = white
        key = np.where(m, white_dist, -1.0) * 67108864.0 + np.arange(m.size, dtype=np.float64).reshape(m.shape)
        best = cv2.dilate(key, ell(3))
        for e_, strict in ((edge, False), (edge2, True)):
            ry, rx = np.nonzero(e_)
            if not len(ry):
                continue
            bv = best[ry, rx]
            bi = (bv - np.floor(bv / 67108864.0) * 67108864.0).astype(np.int64)
            fy, fx = np.divmod(bi, W_)
            P = rgb[ry, rx].astype(np.float32)
            F = rgb[fy, fx].astype(np.float32)
            dP, dF = (255.0 - P).sum(1), (255.0 - F).sum(1)
            cov = np.clip(dP / np.maximum(dF, 45.0), 0, 1)
            pred = cov[:, None] * F + (1 - cov[:, None]) * 255.0
            ok = (dF > dP + 18.0) & (np.abs(pred - P).max(1) <= 26.0)      # really "line colour over white"
            if strict:
                ok &= (dF > 300.0) & (cov < 0.85)                          # a dark line's soft edge, nothing else
            alpha[ry[ok], rx[ok]] = cov[ok] * 255.0
            out[ry[ok], rx[ok]] = F[ok].astype(np.uint8)
            done[ry[ok], rx[ok]] = True
    rim = m & near_bg & ~done
    if solid.sum() > 20 and rim.any():
        idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        near = rgb[idx[0], idx[1]].astype(np.float32)
        p = rgb.astype(np.float32)
        w = np.array([0.299, 0.587, 0.114], np.float32)
        brighter = rim & ((p @ w) > (near @ w) + 6.0)
        dp = (255.0 - p).sum(2)
        df = np.maximum((255.0 - near).sum(2), 45.0)
        a = np.clip(dp / df, 0, 1) * 255.0
        alpha[brighter] = a[brighter]
        out[brighter] = near[brighter].astype(np.uint8)
    if soften:
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), soften) * 255.0
        alpha = np.minimum(alpha, np.clip(soft * 1.35, 0, 255))
    return np.dstack([out, np.clip(alpha + 0.5, 0, 255).astype(np.uint8)])


def smooth_fill(rgb, src, target):
    """`target` filled with the colours of `src` around it, blended: close to the source its nearest colours,
    further in ever wider averages - a soft continuation without the streaks that nearest colours leave."""
    f = rgb.astype(np.float32)
    w = src.astype(np.float32)
    res = np.zeros_like(f)
    got = np.zeros(src.shape, np.float32)
    for sigma, k in ((5, 1), (15, 2), (45, 4), (140, 8)):
        fs = cv2.resize(f * w[..., None], None, fx=1.0 / k, fy=1.0 / k, interpolation=cv2.INTER_AREA)
        ws = cv2.resize(w, None, fx=1.0 / k, fy=1.0 / k, interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(fs, (0, 0), sigma / k)
        den = cv2.GaussianBlur(ws, (0, 0), sigma / k)
        num = cv2.resize(num, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(den, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
        est = num / np.maximum(den[..., None], 1e-4)
        conf = np.clip(den * 5.0, 0, 1) * (1 - got)
        res += est * conf[..., None]
        got += conf
    res /= np.maximum(got[..., None], 1e-4)
    return np.where(target[..., None], np.clip(res + 0.5, 0, 255).astype(np.uint8), rgb)


def kept_input(name, snap_dir):
    """A layer as the stages before left it, for a stage that changes the layer in place and must be able to run
    again: a copy kept in snap_dir. The copy is renewed only when the layer on disk is not what the stage itself
    last wrote (record_output) - then an earlier stage has made it anew. Returns (rgba, x0, y0)."""
    import hashlib, shutil
    os.makedirs(snap_dir, exist_ok=True)
    cur, snap, bp = os.path.join(LDIR, name + ".png"), os.path.join(snap_dir, name + ".png"), os.path.join(snap_dir, "book.json")
    book = json.load(open(bp)) if os.path.exists(bp) else {}
    now = hashlib.md5(open(cur, "rb").read()).hexdigest()
    mine = book.get(name) == now
    if not mine:
        # The layer on disk may be what a LATER stage made of this stage's output (the snapshot folders are numbered
        # in the pipeline's order: work/s9 ... work/s15). Then this stage's copy still holds - taking a new one would
        # feed the later stage's work back in as this stage's input. (2026-10-12: tools/gen_stage.py tidy, run again
        # after its own later step cross, took the torso without its lobes for the torso as the stages before left it.)
        base, me = os.path.dirname(os.path.abspath(snap_dir)), os.path.basename(os.path.abspath(snap_dir))
        if me[:1] == "s" and me[1:].isdigit() and os.path.isdir(base):
            for d in os.listdir(base):
                lb = os.path.join(base, d, "book.json")
                if d[:1] == "s" and d[1:].isdigit() and int(d[1:]) > int(me[1:]) and os.path.exists(lb) and json.load(open(lb)).get(name) == now:
                    mine = True
    if not os.path.exists(snap) or not mine:
        shutil.copy(cur, snap)
        json.dump(_load()[name], open(os.path.join(snap_dir, name + ".json"), "w"))
    e = json.load(open(os.path.join(snap_dir, name + ".json")))
    return np.asarray(Image.open(snap).convert("RGBA")), e["x0"], e["y0"]


def record_output(name, snap_dir):
    """Note that the layer on disk is now the stage's own work (see kept_input)."""
    import hashlib
    bp = os.path.join(snap_dir, "book.json")
    book = json.load(open(bp)) if os.path.exists(bp) else {}
    book[name] = hashlib.md5(open(os.path.join(LDIR, name + ".png"), "rb").read()).hexdigest()
    json.dump(book, open(bp, "w"), indent=1)
