"""Stage 3: the figure cut into the parts the rig moves, each with its hidden part drawn from the plates.

usage: segment.py <project> [--look]

From the canvas, the cut-out and the plates (bald, backhair, face_blank, headless, noarms; noiris for the eyes later):
  hair        what the bald plate does not have: figure pixels that differ from it (tolerant of the plate's 1-2 px
              misregistration) and are not skin
  hair_front / hair_back   hair lying on the head, the face or the body (where the bald plate shows skin or clothing
              under it) is FRONT; hair standing out against the page (the plate shows background there) is BACK; the
              backhair plate's hair under the front hair is the back hair's hidden part
  face        the head's skin (the bald plate's skin round the face's middle) above the jaw line; under the hair the bald
              plate's skin, under the eyes / brows / mouth the face_blank plate's
  eye_l/r, brow_l/r, mouth   what the face_blank plate removed, given to the nearest feature landmark
              (masks only here: work/feature_masks.npz - eyes.py and mouth.py make their layers)
  neck        skin below the jaw line, carried up under the face
  arm_l/arm_r what the noarms plate removed, on each side (not hair, not head)
  body        everything else, with the noarms plate under the arms, the headless plate under the neck and head, the
              bald plate under hair that lies on the body
Every hidden part runs on under the parts in front of it (method rule 3: the line between two parts belongs to the one
in front, the one behind continues under it).
  -> layers/*.png + manifest (layerio), work/labels.png (+ labels.json), work/layer_order.json, work/segment_look.png
Check segment_look.png: each part tinted, then the parts restacked against the picture (must be the picture).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import common as C

LABELS = ["page", "hair_back", "body", "neck", "arm_l", "arm_r", "face", "eye_l", "eye_r", "brow_l", "brow_r", "mouth", "hair_front"]
Z = {"hair_back": 0, "body": 10, "neck": 12, "arm_l": 20, "arm_r": 20, "face": 30, "eye_l": 31, "eye_r": 31, "brow_l": 33,
     "brow_r": 33, "mouth": 32, "hair_front": 40}


def tol_diff(a, b, r=4, sigma=0.0):
    """Per pixel, the smallest colour difference between a and b anywhere within r px of it in b (a small misregistration
    of the plate explained away); with sigma the difference is averaged over a patch first (chara3's layerio.tol_diff)."""
    ai = a.astype(np.float32)
    h, w = a.shape[:2]
    bp = np.pad(b.astype(np.float32), ((r, r), (r, r), (0, 0)), mode="edge")
    best = np.full((h, w), 1e9, np.float32)
    step = max(1, r // 3)
    for dy in range(-r, r + 1, step):
        for dx in range(-r, r + 1, step):
            if dx * dx + dy * dy > r * r:
                continue
            d = np.abs(ai - bp[r + dy:r + dy + h, r + dx:r + dx + w]).max(2)
            if sigma > 0:
                d = cv2.GaussianBlur(d, (0, 0), sigma)
            np.minimum(best, d, out=best)
    return best


def plate_full(P, name, shape):
    """A plate placed on the canvas: (RGB array of the canvas size, bool mask where the plate has pixels)."""
    idx = json.load(open(P.p("work", "plates", "index.json")))
    if name not in idx:
        return None, None
    x0, y0, x1, y1 = idx[name]["box"]
    pl = np.asarray(Image.open(P.p("work", "plates", f"{name}.png")).convert("RGB"))
    H, W = shape
    out = np.full((H, W, 3), 255, np.uint8)
    m = np.zeros((H, W), bool)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    out[ya:yb, xa:xb] = pl[ya - y0:yb - y0, xa - x0:xb - x0]
    # (a margin round the plate's edge is not trusted: the registration and the crop's border)
    mm = np.zeros((y1 - y0, x1 - x0), bool)
    e = max(8, (x1 - x0) // 60)
    # (no margin on a side where the box reaches the canvas's edge: there the crop's border is the canvas's own - the
    # crown's top 60 px were "no plate", and a strand there stayed in the body)
    el_, et_ = (0 if x0 <= 0 else e), (0 if y0 <= 0 else e)
    er_, eb_ = (0 if x1 >= W else e), (0 if y1 >= H else e)
    mm[et_:(y1 - y0) - eb_, el_:(x1 - x0) - er_] = True
    m[ya:yb, xa:xb] = mm[ya - y0:yb - y0, xa - x0:xb - x0]
    return out, m


def plate_page(P, pl, m, k):
    """A plate's page: its white JOINED to the page round the figure (white clothing is white too, but enclosed by its
    outlines) - flooded with a disc so that it cannot slip through a line's small breaks."""
    wp = (pl.min(2) > 236) & ((pl.max(2).astype(int) - pl.min(2)) < 16) & m
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * k) | 1, int(2 * k) | 1))
    core = cv2.erode(wp.astype(np.uint8), disc)
    nl, lbl = cv2.connectedComponents(core, connectivity=4)
    page_c = np.asarray(Image.open(P.p("work", "bg.png"))) > 127
    sid = np.unique(lbl[page_c & (core > 0)])
    return cv2.dilate(np.isin(lbl, sid[sid > 0]).astype(np.uint8), disc).astype(bool) & wp


def tone_match(src, A, ref, target):
    """A plate's pixels brought to the picture's tone where they fill in (rule 2: the plate "brought to the artwork's
    tone right beside the line"): the picture-minus-plate difference over ref (where both show the same thing) carried
    smoothly into target and added there. -> src with target adjusted."""
    import layerio
    if not target.any() or ref.sum() < 50:
        return src
    ys, xs = np.nonzero(ref | target)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    d = np.clip(A[y0:y1, x0:x1].astype(np.float32) - src[y0:y1, x0:x1].astype(np.float32) + 128.0, 0, 255).astype(np.uint8)
    f = layerio.smooth_fill(d, ref[y0:y1, x0:x1], np.ones((y1 - y0, x1 - x0), bool)).astype(np.float32) - 128.0
    out = src.copy()
    t = target[y0:y1, x0:x1]
    sub = out[y0:y1, x0:x1].astype(np.float32)
    sub[t] = np.clip(sub[t] + f[t], 0, 255)
    out[y0:y1, x0:x1] = sub.astype(np.uint8)
    return out


def crop_diff(A, Pl, m, r, sigma):
    """tol_diff only over the plate's bounding box (speed)."""
    D = np.zeros(A.shape[:2], np.float32)
    ys, xs = np.nonzero(m)
    if not len(ys):
        return D
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    D[y0:y1, x0:x1] = tol_diff(A[y0:y1, x0:x1], Pl[y0:y1, x0:x1], r=r, sigma=sigma)
    D[~m] = 0
    return D


def colour_model(pix, k=4):
    """k colour centres of a set of pixels (k-means), for 'is this pixel one of these colours'."""
    pix = pix.astype(np.float32)
    if len(pix) > 60000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 60000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, cen = cv2.kmeans(pix, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return cen


def colour_dist(img, cen):
    """Per pixel, the distance (largest channel difference) to the nearest of the colour centres."""
    f = img.reshape(-1, 3).astype(np.float32)
    best = np.full(len(f), 1e9, np.float32)
    for c in cen:
        best = np.minimum(best, np.abs(f - c[None, :]).max(1))
    return best.reshape(img.shape[:2])


def near_colours(img, cen, tol):
    return colour_dist(img, cen) <= tol


def biggest(m, n=1, min_px=0):
    lb, k = ndimage.label(m)
    if k == 0:
        return m
    s = ndimage.sum(m, lb, range(1, k + 1))
    order = np.argsort(s)[::-1][:n]
    return np.isin(lb, [i + 1 for i in order if s[i] >= min_px])


def jaw_curve(face_skin, HX, cy, HR, W):
    """The face's lower outline per column: walking down from the face's middle row, the last row of the head's skin
    before the skin stops (the jaw line is the face's own outline - the neck below is skin too, so the skin is read in
    the bald plate where the jaw's outline line separates them, given as face_skin = skin minus dark line)."""
    xs = np.arange(int(HX - 1.15 * HR), int(HX + 1.15 * HR))
    jaw = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        if x < 0 or x >= W:
            continue
        col = face_skin[int(cy):int(cy + 1.6 * HR), x]
        if not col.any():
            continue
        # the first gap below the start (the outline), counted from the face's middle
        run = np.nonzero(~col)[0]
        jaw[i] = cy + (run[0] if len(run) else len(col))
    return xs, jaw


def grow_round(m, r):
    """m grown by r px in every direction - round. (A 3x3 dilation repeated r times is a diamond: where its edge is seen -
    a hidden part's border bared by a turn - it showed as stairs and 45-degree cuts.)"""
    return ndimage.distance_transform_edt(~m) <= r


def side_locks(P, A, hair, ws, elev, refs, k, HR):
    """The side locks, out of the hair the lockmap left grey: the locks that hang beside the cheeks and over the ears. The
    lockmap paints the bangs only; left grey, the side locks were back hair, BEHIND the face: at a turn the cheek slid over
    them, the face had nothing under them (a transparent band by the ear) and there was no ear at all (chara4,
    2026-10-22). The plate sidemap paints each side lock in its own colour and the rest grey: a watershed on the drawn
    lines from its colours' cores (the locks) and its grey's (the hair that stays), over the grey hair - as the bangs.
    Hair that the plates nosideL / nosideR take away and that no colour reached is a lock of its own. New lock ids follow
    the bangs' in ws. -> ws, the side locks' mask"""
    from skimage.segmentation import watershed
    H, W = hair.shape
    grey_h = hair & (ws == 1)
    removed = np.zeros((H, W), bool)
    for pn_ in ("nosideL", "nosideR"):
        pl_s, m_s = plate_full(P, pn_, (H, W))
        if pl_s is None:
            continue
        D_s = crop_diff(A, pl_s, m_s, int(round(1.5 * k)) + 2, 0.6 * k)
        rem = grey_h & m_s & (D_s > 40)
        rem = ndimage.binary_opening(rem, iterations=max(1, int(k)))
        lbs, ns_ = ndimage.label(rem)
        if ns_:
            szs = np.bincount(lbs.ravel())[1:]
            rem = np.isin(lbs, [i + 1 for i in range(ns_) if szs[i] >= (0.12 * HR) ** 2])
        removed |= rem
        print(f"下絵 {pn_} で消える髪: {int(rem.sum())} px", flush=True)
    names = list(refs)
    smp, msmp = plate_full(P, "sidemap", (H, W))
    mk = np.zeros((H, W), np.int32)
    er_ = max(2, int(1.2 * k))
    if smp is not None:
        q = smp.astype(np.float32)
        satq = q.max(2) - q.min(2)
        lumq = q @ np.array([0.299, 0.587, 0.114], np.float32)
        dist_c = np.stack([np.abs(q - np.array(refs[n_], np.float32)[None, None, :]).max(2) for n_ in names], 0)
        best = np.argmin(dist_c, 0)
        painted = msmp & (np.abs(q - A.astype(np.float32)).max(2) > 50)
        # (a lock colour on the hair counts even where the picture has that colour itself: a red-streaked lock painted
        # red is the picture's red - required to differ, the red side lock got no seed and stayed back hair. Only the
        # hair is looked at here, so no red ornament can pass for a lock)
        coloured = msmp & (satq > 110) & (dist_c.min(0) < 90)
        # (but the hat's ornaments are hair here too - the bald plate takes them away with the hair, and a hat stage
        # takes them out later: their red leaves, unpainted, were "red locks". Only what the plate hairmask paints as hair)
        hmp, mhm = plate_full(P, "hairmask", (H, W))
        if hmp is not None:
            g_ = hmp.astype(np.int16)
            green = mhm & (g_[..., 1] > 170) & (g_[..., 0] < 120) & (g_[..., 2] < 120)
            coloured &= ndimage.binary_dilation(green, iterations=max(1, int(k))) & (painted | ndimage.binary_erosion(green, iterations=max(1, int(2 * k))))
        grey = painted & (satq < 40) & (lumq > 80) & (lumq < 180)
        mk[ndimage.binary_erosion(grey & grey_h, iterations=er_) & ~removed] = 1
        for i_ in range(len(names)):
            mi = ndimage.binary_opening(ndimage.binary_erosion(coloured & (best == i_) & grey_h, iterations=er_), iterations=1)
            lbm, nm_ = ndimage.label(mi)
            for j in range(1, nm_ + 1):
                part = lbm == j
                if part.sum() >= (0.04 * HR) ** 2:
                    mk[part] = max(int(mk.max()), 1) + 1
    else:
        mk[ndimage.binary_erosion(grey_h & ~ndimage.binary_dilation(removed, iterations=int(0.05 * HR)), iterations=er_)] = 1
    # (hair the noside plates take away that no colour reached: a lock of its own per piece)
    lbp, npc = ndimage.label(removed)
    for j in range(1, npc + 1):
        pc = lbp == j
        if not (mk[pc] > 1).any():
            core = ndimage.binary_erosion(pc, iterations=er_)
            mk[core if core.any() else pc] = max(int(mk.max()), 1) + 1
    if mk.max() <= 1:
        return ws, np.zeros((H, W), bool)
    wsS = watershed(elev, mk, mask=grey_h)
    # (small pieces go to the neighbour they share the longest edge with, as the bangs' do)
    for _ in range(3):
        changed = False
        for v_ in range(2, int(wsS.max()) + 1):
            m_ = wsS == v_
            n_px = int(m_.sum())
            if n_px == 0 or n_px >= (0.12 * HR) ** 2:
                continue
            ring_ = ndimage.binary_dilation(m_, iterations=2) & ~m_ & grey_h
            nb = wsS[ring_]
            nb = nb[(nb > 0) & (nb != v_)]
            if len(nb):
                wsS[m_] = int(np.bincount(nb).argmax())
                changed = True
        if not changed:
            break
    # (all the hair the noside plates take away is side lock: it hangs in front of the ear and the temple. Flooded from
    # the grey of the map, a piece over the ear stayed back hair - behind the ear - and the ear, which may not show at
    # rest where back hair is seen, was cut to a jagged shape)
    stray = removed & (wsS <= 1)
    if stray.any() and (wsS > 1).any():
        idx_ = ndimage.distance_transform_edt(~(wsS > 1), return_distances=False, return_indices=True)
        wsS[stray] = wsS[idx_[0][stray], idx_[1][stray]]
        print(f"noside の下絵で消える髪のうち、いちばん近い横髪に入れた分: {int(stray.sum())} px", flush=True)
    side = wsS > 1
    nid = int(ws.max()) + 1
    n_side = 0
    for v_ in np.unique(wsS):
        if v_ <= 1:
            continue
        ws[wsS == v_] = nid
        nid += 1
        n_side += 1
    print(f"横髪: {int(side.sum())} px、{n_side} 束{'（下絵 sidemap から）' if smp is not None else ''}。"
          f"noside の下絵で消える髪の {100.0 * (side & removed).sum() / max(removed.sum(), 1):.0f} % が横髪に入っています", flush=True)
    return ws, side


def main():
    P = C.open_project()
    import layerio
    look = "--look" in sys.argv
    fig = C.load_rgba(P.p("work", "figure.png"))
    H, W = fig.shape[:2]
    alpha = fig[..., 3]
    F = alpha > 0
    a3 = alpha[..., None].astype(np.float32) / 255.0
    A = (fig[..., :3].astype(np.float32) * a3 + 255.0 * (1 - a3)).astype(np.uint8)     # the figure on white
    k = P.kpx                    # (the pixel unit for tolerances, not the point scale)
    HX, HY, HR = P.head()
    el, er, mouth, chin = P.pt("eye_l"), P.pt("eye_r"), P.pt("mouth"), P.pt("chin")
    nose = P.pt("nose", (el + er) / 2 + np.array([0, 0.3 * HR]))
    EYE_Y_ = float(0.5 * (el[1] + er[1]))
    lum = A.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    r = int(round(1.5 * k)) + 2
    sg = 0.6 * k
    # ---- the skin colour: the cheeks under the eyes, beside the nose
    sk = []
    for e in (el, er):
        y0, y1 = int(e[1] + 0.35 * HR), int(e[1] + 0.55 * HR)
        x0, x1 = int(e[0] - 0.12 * HR), int(e[0] + 0.12 * HR)
        sk.append(A[y0:y1, x0:x1].reshape(-1, 3))
    skin_px = np.concatenate(sk)
    skin_c = np.median(skin_px, 0)
    skin_warm = float(skin_c[0] - skin_c[2])
    WARM_TOL = max(5.0, 0.45 * skin_warm)
    print("肌の色", skin_c.round(), flush=True)

    def skinlike(img, tol=38):
        """Near the skin's colour - and as warm as skin: silver or white hair is as light as pale skin, but not as warm
        (red minus blue; the third character's rule)."""
        f = img.astype(np.float32)
        d = np.abs(f - skin_c[None, None, :]).max(2)
        warm = f[..., 0] - f[..., 2]
        # (one-sided and relative to this skin: cooler than the skin by WARM_TOL is not skin - pale skin is only ~15
        # warmer than silver hair; pinker shading is still skin)
        return (d <= tol) & (warm >= skin_warm - WARM_TOL)

    # ---- hair: figure pixels the bald plate does not have
    bald, mb = plate_full(P, "bald", (H, W))
    Db = crop_diff(A, bald, mb, r, sg)
    # (and by warmth: silver or white hair over the bald plate's scalp is as light as the skin there - by colour
    # distance alone it "did not differ", and the face took the hair; the third character's R-B rule)
    warm_A = (A[..., 0].astype(np.float32) - A[..., 2])
    warm_b = (bald[..., 0].astype(np.float32) - bald[..., 2])
    Dw = crop_diff(np.repeat(warm_A[..., None], 3, 2), np.repeat(warm_b[..., None], 3, 2), mb, r, sg)
    cooler = Dw > WARM_TOL
    cooler &= (warm_b - warm_A) > WARM_TOL                 # (the picture cooler than the plate: hair over skin)
    Db_any = (Db > 34) | cooler
    hair = F & mb & Db_any
    # (not skin: the plate redraws skin a little differently - and the face's own features are in the plate)
    hair &= ~(skinlike(A, 34) & skinlike(bald, 40))
    # hair colours: the hair seen above the eyes (the bangs, the top) - the hair is those colours, its dark outlines
    # and light highlights; what the plate merely redrew (a collar, a sleeve's edge) is other colours
    top_band = hair.copy()
    top_band[int(min(el[1], er[1]) - 0.1 * HR):] = False
    cen = colour_model(A[top_band], 5) if top_band.sum() > 500 else colour_model(A[hair], 5)
    huey = near_colours(A, cen, 46) | (lum < 80)
    hair &= huey
    # (hair highlights: light, unsaturated and NOT warm - skin is - differing from the bald plate, beside the hair. A
    # white highlight stroke is no hair colour; left out, it went to the face or the body and stayed there - white wedges
    # at the forehead, white dots over the head when the hair moved)
    sat_A = A.max(2).astype(np.int16) - A.min(2)
    hl = F & mb & Db_any & (lum > 185) & (sat_A < 45) & ~skinlike(A, 34) & ndimage.binary_dilation(hair, iterations=int(2 * k))
    hair |= hl
    hair = ndimage.binary_opening(hair, iterations=max(1, int(k / 2)))
    hair = ndimage.binary_closing(hair, iterations=int(k)) & (hair | huey)
    # (strands over the face: inside the head the bald plate is skin, so whatever there is neither skin nor a feature
    # and differs from it at all is hair - by the plain threshold thin light strands over pale skin stayed face, pale
    # wedges and lines that kept still when the hair swung)
    over_face = ndimage.binary_dilation(skinlike(bald, 44) & mb, iterations=int(k)) & F & ~skinlike(A, 34) & ((Db > 18) | cooler) & huey
    hair |= over_face
    # (round each eye, hair is what the eyes_clear plate took away - the strands over the eye - not what differs from
    # the bald plate: light grey eye-white and the dark lash are "hair colours" for silver or black hair, and where the
    # bald plate redrew an eye's corner a little differently, the corner became hair)
    ecl, mecl = plate_full(P, "eyes_clear", (H, W))
    fb0, mfb0 = plate_full(P, "face_blank", (H, W))
    if ecl is not None:
        for e_pt in (el, er):
            rx, ry = 0.62 * HR, 0.36 * HR
            x0w, y0w = int(max(0, e_pt[0] - rx)), int(max(0, e_pt[1] - ry))
            x1w, y1w = int(min(W, e_pt[0] + rx)), int(min(H, e_pt[1] + ry))
            gy, gx = np.mgrid[y0w:y1w, x0w:x1w]
            ell = (((gx - e_pt[0]) / rx) ** 2 + ((gy - e_pt[1]) / ry) ** 2 <= 1.0) & mecl[y0w:y1w, x0w:x1w]
            Dc = tol_diff(A[y0w:y1w, x0w:x1w], ecl[y0w:y1w, x0w:x1w], r=r, sigma=sg)
            strand = ell & F[y0w:y1w, x0w:x1w] & (Dc > 34) & huey[y0w:y1w, x0w:x1w] & ~skinlike(A[y0w:y1w, x0w:x1w], 34)
            # (the eye itself: what the blank face does not have and the eyes_clear plate keeps - only that leaves the
            # hair; a side strand that eyes_clear kept stays hair)
            if fb0 is not None:
                Df_w = tol_diff(A[y0w:y1w, x0w:x1w], fb0[y0w:y1w, x0w:x1w], r=r, sigma=sg)
                is_eye = ell & (Df_w > 30) & (Dc <= 34) & mfb0[y0w:y1w, x0w:x1w]
            else:
                is_eye = np.zeros_like(ell)
            hw = hair[y0w:y1w, x0w:x1w]
            hair[y0w:y1w, x0w:x1w] = np.where(ell, (hw & ~is_eye) | strand, hw)
    # (above the shoulders, where the bald plate shows the page, what the figure has there is hair - fine fly-away
    # strands of light hair on a light page did not "differ" enough, went to the body and stayed behind as arcs)
    sh_pts0 = [P.pt(f"shoulder_{s_}") for s_ in ("l", "r") if P.pt(f"shoulder_{s_}") is not None]
    sh_y0 = float(np.mean([q[1] for q in sh_pts0])) if sh_pts0 else chin[1] + 1.2 * HR
    bpage0 = plate_page(P, bald, mb, k)
    fly = F & bpage0 & (np.arange(H)[:, None] < sh_y0 - 0.1 * HR) & ndimage.binary_dilation(hair, iterations=int(1.0 * HR))
    hair |= fly
    hair = biggest(hair, n=12, min_px=int(400 * k * k))
    # (clothing whatever the plates say: work/force_body.png, made by hand for one character - a cape or a collar that the
    # bald plate took away with the hair moved with the head as hair and left a hole where it had been)
    force_body = np.zeros((H, W), bool)
    fb_p = P.p("work", "force_body.png")
    if os.path.exists(fb_p):
        force_body = np.asarray(Image.open(fb_p).convert("L")) > 127
        if force_body.shape == (H, W):
            print("体として扱う所（work/force_body.png）:", int((hair & force_body).sum()), "px を髪から外しました", flush=True)
            hair &= ~force_body
        else:
            force_body = np.zeros((H, W), bool)
    hair = ndimage.binary_fill_holes(hair) & F if False else hair
    print("髪:", int(hair.sum()), "px", flush=True)

    # ---- the head's skin in the bald plate (round the face's middle), the jaw line, face vs neck
    bskin = skinlike(bald, 44) & mb
    seed = np.zeros_like(bskin)
    seed[int(nose[1]) - 3:int(nose[1]) + 3, int(nose[0]) - 3:int(nose[0]) + 3] = True
    darkline = (bald.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) < 150
    sk_open = bskin & ~ndimage.binary_dilation(darkline, iterations=1)
    lb, _ = ndimage.label(sk_open)
    ids = np.unique(lb[seed & sk_open])
    head_skin = np.isin(lb, ids[ids > 0])
    head_skin = ndimage.binary_closing(head_skin, iterations=int(2 * k))
    # (the features inside the face: holes of the skin)
    head_skin = ndimage.binary_fill_holes(head_skin)
    # the face is the head's skin above the jaw: the skin region joined to the face's middle in the bald plate stops at the
    # jaw's outline (the neck is cut off by that line) - check: it must not run far below the chin
    below = head_skin.copy()
    below[: int(chin[1] + 0.25 * HR)] = False
    if below.sum() > 0.15 * head_skin.sum():
        # (the jaw's line did not separate face and neck: cut along the chin's row, rounded towards the face's sides)
        yy, xx = np.mgrid[0:H, 0:W]
        cut = chin[1] - 0.55 * HR * ((xx - HX) / (1.0 * HR)) ** 2
        head_skin &= yy < cut
        print("  下絵 bald であごの線が閉じていないので、あごを通る曲線で切りました", flush=True)
    # (its outline line with it - and nothing beyond: the face's region grows from its skin into DARK pixels only, so it
    # ends at the outer edge of its own outline; a plain dilation took the light skin of the neck beyond the jaw line
    # along, a pale rim that slid over the back hair at a turn)
    lum_A = A.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    skin_l = float(skin_c @ np.array([0.299, 0.587, 0.114], np.float32))
    grow = ndimage.binary_dilation(head_skin, iterations=int(2.5 * k)) & (lum_A < skin_l - 40)
    face_reg = head_skin | grow
    face_reg |= ndimage.binary_dilation(face_reg, iterations=1) & (lum_A < skin_l - 12)
    # (the face's own lower outline is no hair: the bald plate keeps the face, and where the picture's jaw line is drawn
    # by the plate too within a few px - a little misregistered or thinner, so that it "differed" - the line is the
    # plate's own. As hair it went with the hair: a double jaw line at every turn, a loose line under a raised chin.
    # Only round the jaw and the cheeks below the eyes (higher up the plate's line is the bald skull's, under the hair),
    # and only lines: dark beside light in the plate)
    W3h = np.array([0.299, 0.587, 0.114], np.float32)
    lum_b = bald.astype(np.float32) @ W3h
    size_l = 2 * int(2.5 * k) + 1
    bmin, bmax = ndimage.minimum_filter(lum_b, size=size_l), ndimage.maximum_filter(lum_b, size=size_l)
    jaw_band = (ndimage.binary_dilation(head_skin, iterations=int(2.5 * k) + 2) & ~ndimage.binary_erosion(head_skin, iterations=int(1.5 * k))
                & (np.arange(H)[:, None] > EYE_Y_ + 0.25 * HR))
    own_line = jaw_band & mb & F & (lum_A < skin_l - 60) & (bmin <= lum_A + 30) & (bmax >= lum_A + 90)
    print("あごの輪郭線を髪から外しました:", int((hair & own_line).sum()), "px", flush=True)
    hair &= ~own_line
    face_reg |= own_line                       # (the face's: left to the body, bits of it stayed at the cheeks' edges)
    # (and the light fringe beside that line - left to the neck, it stayed as white specks under a raised chin)
    face_reg |= ndimage.binary_dilation(own_line, iterations=max(1, int(1.5 * k))) & F & ~hair & (lum_A >= skin_l - 10)
    # (the ears and the skull's rim: skin of the bald plate beside the head's skin, above the chin - the face's)
    yy_ = np.arange(H)[:, None]
    # (a round reach - a dilation by a small cross repeated 0.45 HR times is a diamond, and the ear pieces it cut came out
    # with straight 45-degree edges: crosses of skin in the face layer beside the ear)
    near_head = (ndimage.distance_transform_edt(~head_skin) < 0.45 * HR) & (yy_ < (chin[1] + 0.15 * HR))
    hz = (skinlike(bald, 44) & mb & near_head) & ~head_skin
    lbz, nz = ndimage.label(hz)
    seen_now = skinlike(A, 40) & F
    keep_z = [i + 1 for i in range(nz) if (seen_now & (lbz == i + 1)).sum() > 0.25 * (lbz == i + 1).sum()]
    head_zone = ndimage.binary_closing(np.isin(lbz, keep_z) | head_skin, iterations=int(k))
    # (the head's own skin seen in the picture is never hair: an ear between strands that the bald plate redrew a little
    # elsewhere "differed", and light skin is within reach of light brown hair's colours - the ear went with the hair.
    # Clear skin near the face below the brows; and inside the plate's ears, made a little larger, skin in the shade too
    # - the ear under the hair is darker than the cheek, as dark as the hair's highlights. Opaque pixels only: the hair's
    # edge blended with a white page looks like light skin)
    near_face_h = ndimage.distance_transform_edt(~head_skin) < 0.5 * HR
    fs_ = head_skin & F & skinlike(A, 60) & (lum > 120)
    cen_skin = colour_model(A[fs_], 4) if fs_.sum() > 500 else skin_c[None, :]
    skin_broad = (lum > 110) & (colour_dist(A, cen_skin) < 40) & ((A[..., 0].astype(np.float32) - A[..., 2]) >= skin_warm - WARM_TOL)
    ear_zone = ndimage.binary_dilation(hz, iterations=int(0.1 * HR))
    yy_s = np.arange(H)[:, None]
    lower_h = (yy_s > EYE_Y_ - 0.3 * HR) & (yy_s < chin[1] - 0.05 * HR)
    # (skin's hue: orange to pink - green above blue by less than red above green; a cream or yellow ornament is the
    # other way round)
    Af_ = A.astype(np.float32)
    skin_hue = (Af_[..., 1] - Af_[..., 2]) <= (Af_[..., 0] - Af_[..., 1]) + 10
    seen_skin_h = F & (alpha > 230) & lower_h & skin_hue & ((((head_zone | near_face_h) & skinlike(A, 30))) | (ear_zone & skin_broad))
    print("見えている頭の肌を髪から外しました:", int((hair & seen_skin_h).sum()), "px", flush=True)
    hair &= ~seen_skin_h
    # (the plate hairmask - all the hair painted flat green - says what is hair where colour cannot: silver or white hair
    # over pale skin is the skin's own colour inside its outlines; the strands' insides went to the face and only their
    # outlines moved with the hair. Its edge a little inside: the plate's outline lies a few px off)
    hmp, mhm = plate_full(P, "hairmask", (H, W))
    if hmp is not None:
        g_ = hmp.astype(np.int16)
        green = mhm & (g_[..., 1] > 170) & (g_[..., 0] < 120) & (g_[..., 2] < 120)
        green = ndimage.binary_erosion(ndimage.binary_opening(green, iterations=1), iterations=max(1, int(0.5 * k)))
        add_h = F & green & ~hair
        hair |= add_h
        print("下絵 hairmask から髪に加えました:", int(add_h.sum()), "px", flush=True)
    head_inner = ndimage.binary_erosion(head_zone, iterations=int(0.12 * HR))
    # (everything of the head in the bald plate: its skin round the face - the ears too, seen or not - with its outline)
    head_any = ndimage.binary_dilation(head_zone | (skinlike(bald, 44) & mb & near_head), iterations=int(1.5 * k))
    Image.fromarray((head_skin * 255).astype(np.uint8)).save(P.p("work", "head_skin.png"))
    # ---- features: what the face_blank plate removed
    fb, mf = plate_full(P, "face_blank", (H, W))
    Df = crop_diff(A, fb, mf, r, sg) if fb is not None else np.zeros((H, W), np.float32)
    feat = F & mf & (Df > 30) & ~hair & face_reg
    feat = ndimage.binary_opening(feat, iterations=1)
    marks = {"eye_l": el, "eye_r": er, "brow_l": P.pt("brow_l"), "brow_r": P.pt("brow_r"), "mouth": mouth}
    reach = {"eye_l": 0.62, "eye_r": 0.62, "brow_l": 0.5, "brow_r": 0.5, "mouth": 0.42}
    yy, xx = np.mgrid[0:H, 0:W]
    fl = {}
    ys, xs = np.nonzero(feat)
    best = np.full(len(ys), 1e9); who = np.full(len(ys), -1)
    names = [n for n in marks if marks[n] is not None]
    for i, n in enumerate(names):
        c = marks[n]
        d = np.hypot(xs - c[0], (ys - c[1]) * (1.3 if n.startswith("brow") else 1.0)) / HR
        ok = (d < reach[n]) & (d < best)
        best[ok] = d[ok]; who[ok] = i
    for i, n in enumerate(names):
        m = np.zeros((H, W), bool)
        m[ys[who == i], xs[who == i]] = True
        m = ndimage.binary_closing(m, iterations=int(k))
        # (an eye cut into pieces by strands of hair keeps all of them: with only the biggest few, bits of the iris went
        # to the face)
        fl[n] = biggest(m, n=(40 if n.startswith("eye") else 6 if n.startswith("brow") else 3),
                        min_px=int((3 if n.startswith("eye") else 20) * k * k)) & ~hair
        print(f"{n}: {int(fl[n].sum())} px", flush=True)
    # (each eye's own mask, before the pieces below are added to it: its outline, its window and its zones come from
    # this. With the brow pieces and the lines on the hair in it, the eye's hull reached up into the hair, and the hair
    # there went into the eye's layers - silver patches and hair outlines that came down with the lid)
    fl_core = {e_: fl[e_].copy() for e_ in ("eye_l", "eye_r") if e_ in fl}
    Ac_ = A.astype(np.float32)
    lum_c = Ac_ @ np.array([0.299, 0.587, 0.114], np.float32)
    # (which pieces are the hair's own outlines: what the plate hairmask paints as hair, a little wider. By colour alone
    # neither works - the lash's darkest strokes are as near-black as the hair's outline, and a black line on warm skin
    # is as warm as a lash)
    hmp_e, mhm_e = plate_full(P, "hairmask", (H, W))
    if hmp_e is not None:
        g_e = hmp_e.astype(np.int16)
        hair_pl = grow_round(mhm_e & (g_e[..., 1] > 170) & (g_e[..., 0] < 120) & (g_e[..., 2] < 120), 1.5 * k)
    else:
        Ac_ = A.astype(np.float32)
        hair_pl = ((Ac_[..., 0] - Ac_[..., 1]) < 14) & ((Ac_ @ np.array([0.299, 0.587, 0.114], np.float32)) < 120)

    def eye_like(m_):
        """Of a mask's pieces, the ones the plate hairmask does not paint as hair (under half of a piece); the rest - the
        hair's outlines - goes back to the hair."""
        lb_, n_ = ndimage.label(m_, structure=np.ones((3, 3), bool))
        if not n_:
            return m_, np.zeros_like(m_)
        share = ndimage.mean(hair_pl, lb_, range(1, n_ + 1))
        ok_ = np.isin(lb_, [i + 1 for i in range(n_) if share[i] < 0.5])
        return m_ & ok_, m_ & ~ok_
    # (a "brow" piece that touches its eye is the eye's: the lash's upper strokes and its red accent mark went to the
    # brow by distance to the brow's point - with the real brows hidden under the bangs, the brow's point sits on the
    # lash - and stayed up when the eye closed: 「目を瞑る時に目が残っちゃってる」 (chara4, 2026-10-23). Pieces that
    # touch a piece already taken go with it too)
    for s_ in ("l", "r"):
        e_, b_ = f"eye_{s_}", f"brow_{s_}"
        if e_ not in fl or b_ not in fl or not fl[e_].any() or not fl[b_].any():
            continue
        lbb, nbb = ndimage.label(ndimage.binary_closing(fl[b_], iterations=max(1, int(k))), structure=np.ones((3, 3), bool))
        took = np.zeros((H, W), bool)
        for _ in range(4):
            reach_e = ndimage.binary_dilation(fl[e_] | took, iterations=max(2, int(2 * k)))
            ids_t = [i for i in np.unique(lbb[reach_e & (lbb > 0)]) if i > 0]
            new_t = np.isin(lbb, ids_t) & fl[b_] & ~took
            if not new_t.any():
                break
            took |= new_t
        if took.any():
            took_e, took_h = eye_like(took)
            fl[e_] |= took_e
            fl[b_] &= ~took
            hair |= took_h
            print(f"{b_}: 目に触れている {int(took_e.sum())} px を {e_} に、髪の輪郭 {int(took_h.sum())} px を髪に移しました"
                  f"（眉として残ったのは {int(fl[b_].sum())} px）", flush=True)
    # (character.json "options": {"brows_with_eyes": true} - the real brows are hidden under the bangs, and every stroke
    # above an eye is the eye's: it closes with it. chara4, 「全部つぶってくださいよ」: a thin line left as the brow stood
    # above the shut eye)
    if P.cfg.get("options", {}).get("brows_with_eyes"):
        for s_ in ("l", "r"):
            e_, b_ = f"eye_{s_}", f"brow_{s_}"
            if e_ in fl and b_ in fl and fl[b_].any():
                be_, bh_ = eye_like(fl[b_])
                fl[e_] |= be_
                hair |= bh_
                print(f"{b_}: {int(be_.sum())} px を {e_} に、髪の輪郭 {int(bh_.sum())} px を髪に移しました（brows_with_eyes）", flush=True)
                fl[b_] = np.zeros((H, W), bool)
    # (inside an eye's outline whatever is neither hair nor skin is the eye: bits of the iris and the white between strands
    # of hair barely differ from the blank face's redrawn strands and went to the face - lilac and white patches that
    # stayed when the eye blinked)
    for e_ in ("eye_l", "eye_r"):
        if e_ in fl_core and fl_core[e_].any():
            zone_e = C.hull_mask(fl_core[e_])
            add_e = zone_e & F & ~hair & ~skinlike(A, 34)
            fl[e_] |= add_e
            fl_core[e_] |= add_e
    # (the eyes' lines on and in the hair: hair over the eyes is drawn half see-through, and the crease line and the
    # lash's ends show on it and in it - near an eye, where the face_blank plate (eyes and brows taken away, the hair
    # kept) differs from the picture and the picture is darker and warmer than the plate. In the hair layers they stayed
    # when the eye closed (chara4, 「そこも一緒に瞑るようにしてください」, 2026-10-23). At the hair's edge and
    # lash-coloured - the line itself - it is the eye's; deeper in, the hair gets the plate's plain hair there)
    hair_repaint, hair_repaint_col = None, None
    if fb is not None:
        Af_l = A.astype(np.float32)
        fbf = fb.astype(np.float32)
        W3l = np.array([0.299, 0.587, 0.114], np.float32)
        # (a brown line seen through silver hair darkens it a little and warms it: darker by 6, warmer by 3. The hair's
        # own outlines, which the plate redraws a pixel off, are darker too but neutral)
        darker = (fbf @ W3l) - (Af_l @ W3l) > 6
        warmer = (Af_l[..., 0] - Af_l[..., 2]) - (fbf[..., 0] - fbf[..., 2]) > 3
        near_eyes = np.zeros((H, W), bool)
        for e_ in ("eye_l", "eye_r"):
            if e_ in fl_core and fl_core[e_].any():
                ys_e, xs_e = np.nonzero(fl_core[e_])
                ze = grow_round(C.hull_mask(fl_core[e_]), 0.35 * HR) & (np.arange(H)[:, None] < np.percentile(ys_e, 70))
                near_eyes |= ze
        cand_l = hair & mf & near_eyes & (Df > 12) & darker & warmer
        # (with its soft fringe)
        cand_l |= ndimage.binary_dilation(cand_l, iterations=max(1, int(k))) & hair & mf & near_eyes & ((fbf @ W3l) - (Af_l @ W3l) > 3) & \
                  ((Af_l[..., 0] - Af_l[..., 2]) - (fbf[..., 0] - fbf[..., 2]) > 1)
        if cand_l.any():
            edge_h = hair & ~ndimage.binary_erosion(hair, iterations=max(2, int(2 * k)))
            lashy_l = (Af_l @ W3l < 140) & ((Af_l[..., 0] - Af_l[..., 2]) > 14)
            to_eye = cand_l & edge_h & lashy_l
            # (and a line's end labelled hair where the plate shows skin - no hair there at all, only the eye's line over
            # the skin: darker than that skin by far, but no warmer, so the see-through test above missed it, and it
            # stayed up as a dash over the shut eye)
            on_skin_l = skinlike(fb, 44) & ((fbf @ W3l) - (Af_l @ W3l) > 30) & (Af_l @ W3l < 170) & ((Af_l[..., 0] - Af_l[..., 2]) > 10)
            to_eye |= hair & mf & near_eyes & (Df > 25) & on_skin_l
            for e_ in ("eye_l", "eye_r"):
                if e_ in fl_core and fl_core[e_].any():
                    mine = to_eye & grow_round(C.hull_mask(fl_core[e_]), 0.35 * HR)
                    fl[e_] |= mine
            hair &= ~to_eye
            hair_repaint = cand_l & ~to_eye
            ref_h = hair & mf & ~grow_round(cand_l, 0.05 * HR) & grow_round(cand_l, 0.3 * HR) & (Df < 15)
            hair_repaint_col = tone_match(fb, A, ref_h, hair_repaint) if ref_h.sum() > 200 else fb
            print(f"髪の上の目の線: {int(to_eye.sum())} px を目に移し、髪 {int(hair_repaint.sum())} px を"
                  f"下絵 face_blank で無地に塗り直しました", flush=True)
    feat_all = np.zeros((H, W), bool)
    for m in fl.values():
        feat_all |= m
    # (the features' hidden parts: the bald plate draws every feature whole; where hair hides one at rest - an eyelash's
    # tip under a side lock, a brow's end under the bangs - that stroke belongs to the feature, not to the face's fill:
    # in the face it stayed behind as a loose dark line when the hair swung, and did not blink or rise with its feature)
    W3s = np.array([0.299, 0.587, 0.114], np.float32)
    skin_l0 = float(skin_c @ W3s)
    bald_dark = mb & ((bald.astype(np.float32) @ W3s) < skin_l0 - 40) & ~skinlike(bald, 34)
    hid_feat = {}
    for n_, m_ in fl.items():
        if not m_.any():
            hid_feat[n_] = np.zeros((H, W), bool)
            continue
        near = ndimage.binary_dilation(C.hull_mask(m_), iterations=int(0.12 * HR))
        cand_h = near & bald_dark & hair & ~feat_all
        lbh2, nh2 = ndimage.label(cand_h)
        touch = ndimage.binary_dilation(m_, iterations=max(int(2 * k), int(0.06 * HR)))
        ids_h = np.unique(lbh2[touch & cand_h])
        hid_feat[n_] = np.isin(lbh2, ids_h[ids_h > 0])
        if hid_feat[n_].any():
            hid_feat[n_] = ndimage.binary_dilation(hid_feat[n_], iterations=max(1, int(k / 2))) & near & hair
            print(f"{n_}: 髪に隠れた {int(hid_feat[n_].sum())} px を下絵 bald から描きました", flush=True)
    hid_feat_all = np.zeros((H, W), bool)
    for m_ in hid_feat.values():
        hid_feat_all |= m_
    # ---- face: the head's skin with its outline, not hair, not the features (they are their own parts) - and every
    # other seen bit of the head (an ear's sliver between strands)
    yy_ = np.arange(H)[:, None]
    near_head0 = ndimage.binary_dilation(head_skin, iterations=int(0.45 * HR)) & (yy_ < (chin[1] + 0.15 * HR))
    head_bits = ndimage.binary_dilation(skinlike(bald, 44) & mb & near_head0, iterations=int(1.5 * k)) & (yy_ < chin[1] + 0.1 * HR)
    # (not the neck: under the face's own lower outline nothing is the head's; beside the face only above the chin)
    jaw_c = np.full(W, -1)
    cs_ = np.nonzero(head_skin.any(0))[0]
    jaw_c[cs_] = [np.nonzero(head_skin[:, c])[0].max() for c in cs_]
    below_jaw = (jaw_c[None, :] >= 0) & (yy_ > jaw_c[None, :] + int(1.5 * k))
    beside = (jaw_c[None, :] < 0) & (yy_ > chin[1] - 0.2 * HR)
    # (but never the ears: an ear hangs beside the face, below the face's skin in its own columns - cut as "under the
    # jaw" it went to the body and then with the hair. The ears only - the plate's skin pieces beside the face, not the
    # neck's top under the chin: exempted too, the face took a strip of the neck down to a straight edge)
    ears_z = np.zeros((H, W), bool)
    if keep_z:
        cms = ndimage.center_of_mass(np.ones((H, W)), lbz, keep_z)
        ear_ids = [i for i, (cy_, cx_) in zip(keep_z, cms) if abs(cx_ - HX) > 0.75 * HR]
        if ear_ids:
            ears_z = ndimage.binary_dilation(np.isin(lbz, ear_ids), iterations=int(k))
    head_bits &= ~((below_jaw | beside) & ~ears_z)
    face = F & (face_reg | head_bits | seen_skin_h) & ~hair & ~feat_all
    face_reg = face_reg | (head_bits & F & ~hair)
    # ---- arms: SAM on the picture with the bald plate in place of the hair (the plate shows the sleeves whole - a white
    # sleeve over a white blouse does not differ from the noarms plate, so the plate difference alone leaves holes),
    # with points along shoulder - elbow - wrist - hand; joined with what the noarms plate removed along the limb
    na, mn_ = plate_full(P, "noarms", (H, W))
    arm = {"arm_l": np.zeros((H, W), bool), "arm_r": np.zeros((H, W), bool)}
    arm_full = {"arm_l": np.zeros((H, W), bool), "arm_r": np.zeros((H, W), bool)}
    hairless = A.copy()
    hairless[mb] = bald[mb]
    Da = crop_diff(A, na, mn_, r, sg) if na is not None else np.zeros((H, W), np.float32)
    sam = None
    for side in ("l", "r"):
        js = [P.pt(f"{j}_{side}") for j in ("shoulder", "elbow", "wrist")]
        if any(j is None for j in js):
            continue
        sh, eb, wr = js
        hand = wr + (wr - eb) * 0.45
        pos = [sh + 0.6 * (eb - sh), eb, eb + 0.5 * (wr - eb), wr, wr + 0.6 * (hand - wr)]
        pos = [p_ for p_ in pos if F[int(p_[1]), int(p_[0])]]
        neg = [np.array([HX, sh[1] + 1.4 * HR]), P.pt("hip", np.array([HX, wr[1]]))]
        xs_ = [q[0] for q in (sh, eb, wr, hand)]; ys_ = [q[1] for q in (sh, eb, wr, hand)]
        bx = [int(max(0, min(xs_) - 0.9 * HR)), int(max(0, min(ys_) - 0.7 * HR)), int(min(W, max(xs_) + 0.9 * HR)), int(min(H, max(ys_) + 0.9 * HR))]
        # (the limb's capsule: nothing of the arm lies far from its bones)
        cap = np.zeros((H, W), np.uint8)
        for a_, b_ in ((sh, eb), (eb, wr), (wr, hand)):
            cv2.line(cap, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), 1, int(1.25 * HR))
        cap = cap.astype(bool)
        try:
            if sam is None:
                from sam import Sam
                sam = Sam(hairless)
            m_, sc = sam.mask((0, 0, W, H), bx, pos=[tuple(map(float, q)) for q in pos], neg=[tuple(map(float, q)) for q in neg], box=bx)
            print(f"  SAM arm_{side}: スコア {sc:.2f}、{int(m_.sum())} px", flush=True)
        except Exception as e:
            print("  SAM が使えません（" + str(e)[:80] + "）。下絵との差だけで分けます", flush=True)
            m_ = np.zeros((H, W), bool)
        dm = F & (Da > 34) & cap
        dm = ndimage.binary_opening(dm, iterations=int(k))
        full = (m_ & cap) | dm
        full = ndimage.binary_closing(full, iterations=int(2 * k))
        full = ndimage.binary_fill_holes(full) & F & ~face_reg & ~feat_all
        full = biggest(full, n=1)
        arm_full[f"arm_{side}"] = full
        arm[f"arm_{side}"] = full & ~hair
        print(f"arm_{side}: {int(arm[f'arm_{side}'].sum())} px（髪に隠れた所も入れた全体: {int(full.sum())} px）", flush=True)
    # ---- neck: skin below the face's skin, above the collar, near the axis
    # (and its shadowed skin under the jaw: darker than the cheeks but as warm - left to the body, the jaw's shadow and
    # the neck under it stayed behind as a pale band with a dark edge when the face turned)
    fA = A.astype(np.float32)
    lumA = fA @ np.array([0.299, 0.587, 0.114], np.float32)
    skin_lum = float(skin_c @ np.array([0.299, 0.587, 0.114], np.float32))
    shadow_skin = ((fA[..., 0] - fA[..., 2]) >= skin_warm - WARM_TOL) & (lumA > 0.5 * skin_lum) & (lumA < skin_lum + 15) & \
                  (fA[..., 0] > fA[..., 1]) & (np.abs(fA - skin_c[None, None, :]).max(2) < 110)
    neck = F & (skinlike(A, 40) | shadow_skin) & ~face_reg & ~hair & ~arm["arm_l"] & ~arm["arm_r"]
    neck &= (yy > chin[1] - 0.5 * HR) & (yy < chin[1] + 1.2 * HR) & (np.abs(xx - HX) < 0.9 * HR)
    neck = ndimage.binary_closing(neck, iterations=int(k))
    lbn, nn = ndimage.label(neck)
    axis_band = np.abs(xx - HX) < 0.25 * HR
    keep_n = [i + 1 for i in range(nn) if ((lbn == i + 1) & axis_band).sum() > 50 * k * k]
    neck = np.isin(lbn, keep_n) if keep_n else biggest(neck, n=1, min_px=int(100 * k * k))
    # (the neck's outline lines beside it)
    neck_reg = ndimage.binary_dilation(neck, iterations=int(1.5 * k)) & F & ~face_reg & ~hair & (yy > chin[1] - 0.5 * HR)
    print("首:", int(neck_reg.sum()), "px", flush=True)
    # ---- hair ornaments: pieces of the figure that are neither hair nor head, inside the hair above the shoulders and
    # bordered mostly by hair (a ribbon or a clip on a braid) - they move with the hair (left in the body, they stayed
    # where they were when the hair swung)
    sh_pts = [P.pt(f"shoulder_{s_}") for s_ in ("l", "r") if P.pt(f"shoulder_{s_}") is not None]
    sh_y = float(np.mean([q[1] for q in sh_pts])) if sh_pts else chin[1] + 1.2 * HR
    cand = F & ~hair & ~face_reg & ~feat_all & ~neck_reg & ~seen_skin_h & (yy_ < sh_y) & ndimage.binary_dilation(hair, iterations=int(0.06 * HR))
    lab_o, n_o = ndimage.label(cand)
    orn = 0
    for i, sl in enumerate(ndimage.find_objects(lab_o)):
        if sl is None:
            continue
        sl = tuple(slice(max(0, q.start - 3), q.stop + 3) for q in sl)
        comp = lab_o[sl] == i + 1
        sz_ = int(comp.sum())
        if sz_ < 4 or sz_ > 0.6 * HR * HR:
            continue
        # (an ornament is gone in the bald plate with the hair; what the plate keeps there - a collar, a sleeve seen
        # between strands - is the clothing behind the hair: taken into the hair, collar pieces went with the hair, navy
        # flakes over the collar at every nod)
        if (Db[sl][comp] > 34).mean() < 0.5:
            continue
        ring_ = ndimage.binary_dilation(comp, iterations=2) & ~comp
        if (ring_ & hair[sl]).sum() > 0.55 * ring_.sum():
            hair[sl] |= comp
            orn += sz_
    if orn:
        print("髪飾りを髪に含めました:", orn, "px", flush=True)
    # ---- hair front / back: on skin or clothing in the bald plate = front; against the page there = back
    # (the plate's page: its white JOINED to the page round the figure - a white blouse is white too, but enclosed by its
    # outlines; flooded with a disc so that it cannot slip through a line's small breaks)
    bald_page = plate_page(P, bald, mb, k)
    lmp, mlmp = plate_full(P, "lockmap", (H, W))
    lock_ids = None
    side_m = np.zeros((H, W), bool)
    if lmp is not None and mlmp[hair].mean() > 0.9:
        # (the front hair and its locks from the plate lockmap - Codex paints every front lock in its own flat colour and
        # the rest of the hair flat grey - snapped to the picture's own strand outlines by a watershed: seeds are the
        # colours' cores, the ridges the drawn lines. Pixel by pixel along the bald plate's skull, single strands had been
        # cut in two along a circle, half front and half back; at a turn they came apart and the hair frayed and stretched
        # like one sheet (chara4, 2026-10-22: 「全部雑です」). A light-region vote against the backhair plate failed: the
        # strands' lines have gaps and silver hair is as light as the forehead behind it)
        from skimage.segmentation import watershed
        REFS = {"red": (230, 30, 30), "green": (40, 210, 40), "blue": (30, 60, 230), "yellow": (240, 230, 30), "magenta": (230, 30, 220),
                "cyan": (30, 220, 230), "orange": (250, 140, 20), "purple": (140, 40, 220)}
        q = lmp.astype(np.float32)
        satq = q.max(2) - q.min(2)
        lumq = q @ np.array([0.299, 0.587, 0.114], np.float32)
        # (only what Codex repainted: the picture's own saturated colours - a red hat ornament, a red streak - stay as they
        # are in the map and are no lock)
        painted = mlmp & (np.abs(q - A.astype(np.float32)).max(2) > 50)
        grey = painted & (satq < 40) & (lumq > 80) & (lumq < 180)
        names = list(REFS)
        dist_c = np.stack([np.abs(q - np.array(REFS[n_], np.float32)[None, None, :]).max(2) for n_ in names], 0)
        best = np.argmin(dist_c, 0)
        coloured = painted & (satq > 110) & (dist_c.min(0) < 90)
        markers = np.zeros((H, W), np.int32)
        er_ = max(2, int(1.2 * k))
        gm = ndimage.binary_erosion(grey & hair, iterations=er_)
        markers[gm] = 1
        for i_, n_ in enumerate(names):
            mi = ndimage.binary_erosion(coloured & (best == i_) & hair, iterations=er_)
            mi = ndimage.binary_opening(mi, iterations=1)
            # (one lock per colour region: two separate regions of one colour are two locks)
            lbm, nm_ = ndimage.label(mi)
            for j in range(1, nm_ + 1):
                part = lbm == j
                if part.sum() >= (0.04 * HR) ** 2:
                    markers[part] = markers.max() + 1
        elev = cv2.GaussianBlur(255.0 - lum, (0, 0), max(1.0, 0.3 * k))
        # (the hair proper - what the plate hairmask paints as hair: a hat that the bald plate took away with the hair is
        # hair here until a hat stage takes it out, and without seeds of its own it was flooded from the nearest lock -
        # a beret's flap and tassels became a side lock and swung as hair (chara4). What is not hair proper stays back)
        hm_ok = hair
        hmp2, mhm2 = plate_full(P, "hairmask", (H, W))
        if hmp2 is not None:
            g2 = hmp2.astype(np.int16)
            green2 = mhm2 & (g2[..., 1] > 170) & (g2[..., 0] < 120) & (g2[..., 2] < 120)
            hm_ok = hair & (ndimage.binary_dilation(green2, iterations=int(2 * k)) | ~mhm2)
            markers[~hm_ok & (markers > 1)] = 0
        ws = watershed(elev, markers, mask=hm_ok)
        ws[hair & ~hm_ok] = 1
        # (small pieces - a lock's sliver beyond a crossing strand, a fragment at an eye's corner - go to the neighbour
        # they share the longest edge with: as locks of their own they swung apart from the lock they belong to)
        small_px = (0.12 * HR) ** 2
        for _ in range(3):
            changed = False
            for v_ in range(2, int(ws.max()) + 1):
                m_ = ws == v_
                n_px = int(m_.sum())
                if n_px == 0 or n_px >= small_px:
                    continue
                ring_ = ndimage.binary_dilation(m_, iterations=2) & ~m_ & hair
                nb = ws[ring_]
                nb = nb[(nb > 0) & (nb != v_)]
                if len(nb):
                    ws[m_] = int(np.bincount(nb).argmax())
                    changed = True
            if not changed:
                break
        # (the ids made consecutive again: 1 = back, 2.. = the locks)
        uniq = [v_ for v_ in np.unique(ws) if v_ > 1]
        remap = np.zeros(int(ws.max()) + 1, np.int32)
        remap[1] = 1
        for j_, v_ in enumerate(uniq):
            remap[v_] = j_ + 2
        ws = remap[ws]
        ws, side_m = side_locks(P, A, hm_ok, ws, elev, REFS, k, HR)
        front = hair & (ws > 1)
        back = hair & ~front
        lock_ids = np.where(front, ws - 1, 0).astype(np.uint8)
        print("下絵 lockmap から前髪と後ろ髪、前髪の束", int(lock_ids.max()), "本を分けました（描かれた線に沿って）", flush=True)
    else:
        back = hair & bald_page
        back = ndimage.binary_opening(back, iterations=int(k))
        front = hair & ~back
    # (small islands of either kind inside the other go with it)
    for _ in range(2):
        lbf, nf = ndimage.label(front)
        sf = ndimage.sum(front, lbf, range(1, nf + 1))
        small = np.isin(lbf, [i + 1 for i in range(nf) if sf[i] < 600 * k * k])
        front &= ~small; back |= small & hair
        lbb, nb = ndimage.label(back)
        sb = ndimage.sum(back, lbb, range(1, nb + 1))
        small = np.isin(lbb, [i + 1 for i in range(nb) if sb[i] < 600 * k * k])
        back &= ~small; front |= small & hair
    print(f"前髪: {int(front.sum())} px、後ろ髪: {int(back.sum())} px", flush=True)
    # ---- body: the rest of the figure
    body = F & ~hair & ~face_reg & ~feat_all & ~neck_reg & ~arm_full["arm_l"] & ~arm_full["arm_r"]
    # ---- the label map (seen parts, back to front)
    lab = np.zeros((H, W), np.uint8)
    seen = {"hair_back": back, "body": body, "neck": neck_reg, "arm_l": arm["arm_l"], "arm_r": arm["arm_r"], "face": face,
            "eye_l": fl.get("eye_l", np.zeros((H, W), bool)), "eye_r": fl.get("eye_r", np.zeros((H, W), bool)),
            "brow_l": fl.get("brow_l", np.zeros((H, W), bool)), "brow_r": fl.get("brow_r", np.zeros((H, W), bool)),
            "mouth": fl.get("mouth", np.zeros((H, W), bool)), "hair_front": front}
    for n in sorted(seen, key=lambda n: Z[n]):
        lab[seen[n] & F] = LABELS.index(n)
    # (every figure pixel has an owner: what is left goes to the nearest labelled part)
    left = F & (lab == 0)
    if left.any():
        idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab[left] = lab[idx[0][left], idx[1][left]]
    # (the face's own lower outline is the face's, not the neck's: lines of the picture right under the face - the chin's
    # outline, drawn red or dark - labelled neck stayed on the neck when the chin rose: a second jaw line under the face)
    nid, fid0 = LABELS.index("neck"), LABELS.index("face")
    near_face_n = ndimage.distance_transform_edt(lab != fid0) <= max(2.0, 1.5 * k)
    Af0 = A.astype(np.float32)
    line_px = ((Af0 @ np.array([0.299, 0.587, 0.114], np.float32)) < float(skin_c @ np.array([0.299, 0.587, 0.114], np.float32)) - 50) | \
              ((Af0.max(2) - Af0.min(2)) > 90)
    to_face = (lab == nid) & near_face_n & line_px
    if to_face.any():
        lab[to_face] = fid0
        print("あごの輪郭線を顔に移しました（首になっていた分）:", int(to_face.sum()), "px", flush=True)
    # (a feature's bit seen among the hair - an eyelash's tip drawn across a strand - is the feature's: as a small piece of
    # the face cut off by the hair it stayed behind as a grey flake when the eye blinked, the head turned or the hair swung.
    # Small pieces apart from the face, not skin but dark or cool, near an eye or a brow)
    fid = LABELS.index("face")
    lf, nf = ndimage.label(lab == fid)
    if nf > 1:
        szf = np.bincount(lf.ravel(), minlength=nf + 1)[1:]
        main_f = int(np.argmax(szf)) + 1
        lumA = A.astype(np.float32) @ W3s
        dist_f, moved = {}, 0
        for i, sl in enumerate(ndimage.find_objects(lf)):
            if sl is None or i + 1 == main_f or szf[i] > (0.1 * HR) ** 2:
                continue
            comp = lf[sl] == i + 1
            a_, l_ = A[sl][comp], lumA[sl][comp]
            featlike = ~skinlike(a_[None], 34)[0] & ((l_ < skin_l0 - 40) | ((a_[:, 0].astype(np.float32) - a_[:, 2]) < skin_warm - WARM_TOL))
            if featlike.mean() < 0.5:
                continue
            best = None
            for f_ in ("eye_l", "eye_r", "brow_l", "brow_r"):
                if f_ not in dist_f:
                    fm_ = lab == LABELS.index(f_)
                    dist_f[f_] = ndimage.distance_transform_edt(~fm_) if fm_.any() else None
                if dist_f[f_] is None:
                    continue
                dmin = float(dist_f[f_][sl][comp].min())
                if dmin < 0.12 * HR and (best is None or dmin < best[1]):
                    best = (f_, dmin)
            if best is not None:
                lab[sl][comp] = LABELS.index(best[0])
                moved += int(comp.sum())
        if moved:
            print(f"髪の中にあった目・眉・口のかけらを元のパーツに戻しました: {moved} px", flush=True)
    # (no head in the body: a piece of the body apart from its main mass, above the shoulders and not of the clothing's
    # colours, is a bit of the head - a fly-away strand, a cheek's outline between strands - and stayed where it was when
    # the head moved. It goes to the part it borders most. Clothing seen between strands stays the body's)
    bid = LABELS.index("body")
    lb_b, nb_b = ndimage.label(lab == bid)
    if nb_b > 1:
        szb = np.bincount(lb_b.ravel(), minlength=nb_b + 1)[1:]
        main_b = int(np.argmax(szb)) + 1
        cen_body = colour_model(A[lb_b == main_b], 6)
        d_main = ndimage.distance_transform_edt(lb_b != main_b)
        cand_l = [LABELS.index(n_) for n_ in ("hair_front", "hair_back", "face", "neck")]
        moved_b, d_hair = 0, {}
        for i, sl in enumerate(ndimage.find_objects(lb_b)):
            if sl is None or i + 1 == main_b:
                continue
            sl2 = tuple(slice(max(0, q.start - 3), q.stop + 3) for q in sl)
            comp = lb_b[sl2] == i + 1
            if np.nonzero(comp)[0].mean() + sl2[0].start > sh_y - 0.1 * HR:
                continue
            if force_body[sl2][comp].any():
                continue                              # (clothing by hand)
            a_ = A[sl2][comp][None]
            # (clothing only beside the body's main mass - a light strand far above it has the colour of a white cord)
            cloth = (float(d_main[sl2][comp].min()) < 0.6 * HR and (colour_dist(a_, cen_body) < colour_dist(a_, cen))[0].mean() > 0.5
                     and skinlike(a_, 34)[0].mean() < 0.5)
            if cloth:
                continue
            ring_ = ndimage.binary_dilation(comp, iterations=2) & ~comp
            nbl = lab[sl2][ring_]
            counts = [int((nbl == c).sum()) for c in cand_l]
            if skinlike(a_, 40)[0].mean() > 0.5 and head_any[sl2][comp].any():
                # (skin of the head - an ear's sliver between strands - is the face's, however much hair is round it:
                # given to the front hair, an ear went with the hair)
                lab[sl2][comp] = LABELS.index("face")
            elif max(counts) > 0:
                lab[sl2][comp] = cand_l[int(np.argmax(counts))]
            else:
                # (a fly-away strand out in the page touches nothing: it goes to the nearest hair, if near)
                if not d_hair:
                    for hn_ in ("hair_front", "hair_back"):
                        hm_ = lab == LABELS.index(hn_)
                        d_hair[hn_] = ndimage.distance_transform_edt(~hm_) if hm_.any() else None
                best_ = min(((float(d_[sl2][comp].min()), hn_) for hn_, d_ in d_hair.items() if d_ is not None), default=None)
                if best_ is None or best_[0] > 1.0 * HR:
                    continue
                lab[sl2][comp] = LABELS.index(best_[1])
            moved_b += int(comp.sum())
        if moved_b:
            print(f"体に残っていた頭のかけらを周りのパーツに移しました: {moved_b} px", flush=True)
    # (the check: the body has nothing above the shoulders but its main mass - a hood, a collar)
    lb_b, nb_b = ndimage.label(lab == bid)
    if nb_b > 1:
        szb = np.bincount(lb_b.ravel(), minlength=nb_b + 1)[1:]
        stray = (lb_b > 0) & (lb_b != int(np.argmax(szb)) + 1) & (np.arange(H)[:, None] < sh_y - 0.1 * HR)
        print(f"CHECK 肩より上で体のかたまりから離れている所: {int(stray.sum())} px（髪の間に見える服だけならよい）", flush=True)
    Image.fromarray(lab).save(P.p("work", "labels.png"))
    json.dump({"id": {n: i for i, n in enumerate(LABELS)}, "z": Z}, open(P.p("work", "labels.json"), "w"), indent=1)
    if lock_ids is not None:
        Image.fromarray(np.where(lab == LABELS.index("hair_front"), lock_ids, 0).astype(np.uint8)).save(P.p("work", "lock_ids.png"))
    np.savez_compressed(P.p("work", "feature_masks.npz"), **{n: (lab == LABELS.index(n)) for n in ("eye_l", "eye_r", "brow_l", "brow_r", "mouth")},
                        **{n + "_core": (lab == LABELS.index(n)) & fl_core[n] for n in fl_core},
                        **{n + "_hidden": hid_feat.get(n, np.zeros((H, W), bool)) for n in ("eye_l", "eye_r", "brow_l", "brow_r", "mouth")})
    # ---- the layers, with their hidden parts
    hb, mhb = plate_full(P, "backhair", (H, W))
    hl, mhl = plate_full(P, "headless", (H, W))
    owner = {n: lab == LABELS.index(n) for n in LABELS[1:]}
    reach_px = int(0.18 * HR)                  # how far a part runs on under the parts in front of it

    def under_front(n):
        """What lies in front of part n (higher z), as a mask."""
        m = np.zeros((H, W), bool)
        for o in owner:
            if Z[o] > Z[n]:
                m |= owner[o]
        return m

    def carry(rgb, known, target):
        """target pixels get the colour of the nearest known pixel (then a light blur), the plain way to run a part on."""
        if not target.any() or not known.any():
            return rgb
        idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        out = rgb.copy()
        out[target] = rgb[idx[0][target], idx[1][target]]
        return out

    # (the skin the backhair plate draws under the front hair, as one smooth region: its hairline is where the face's
    # hidden part ends and the back hair's begins. It was opened with a small cross fifteen times over - diamonds and
    # stairs - and the face under the bangs came out as a patchwork of skin squares and grey blocks: what every bang
    # showed when it swung (chara4, 2026-10-22, the rest drawn without the front hair))
    def smooth_region(m, sig, min_px):
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig) > 0.5
        lb_, n_ = ndimage.label(m)
        if n_:
            sz_ = np.bincount(lb_.ravel())[1:]
            m = np.isin(lb_, [i + 1 for i in range(n_) if sz_[i] >= min_px])
        # (and its small holes - a brow, a lash, a thin strand the plate kept - closed)
        holes_ = ndimage.binary_fill_holes(m) & ~m
        lbh, nh = ndimage.label(holes_)
        if nh:
            szh = np.bincount(lbh.ravel())[1:]
            m |= np.isin(lbh, [i + 1 for i in range(nh) if szh[i] < min_px])
        return m
    # (which of the backhair plate is hair: its own hair mask, a plate made on it (bh_hairmask) - its pinkish silver
    # hair is the skin's colour (red - blue 16 against the skin's 17 on chara4) and its hairline no closed line, so the
    # skin found by colour took the crown and the face layer drew the combed-back hair)
    bh_hair = None
    bhm, mbhm = plate_full(P, "bh_hairmask", (H, W))
    if hb is not None and bhm is not None:
        g_ = bhm.astype(np.int16)
        bh_hair = mbhm & (g_[..., 1] > 170) & (g_[..., 0] < 120) & (g_[..., 2] < 120)
        bh_hair = cv2.GaussianBlur(bh_hair.astype(np.float32), (0, 0), max(1.5, 1.5 * k)) > 0.5
        print(f"下絵 backhair の髪の部分（下絵 bh_hairmask で判断）: {int(bh_hair.sum())} px", flush=True)
    # (the same for the noside plates: their own hair masks, plates made on them (ns_hairmaskL / R))
    ns_hair = {}
    for side_ in ("L", "R"):
        nm_p, nm_m = plate_full(P, "ns_hairmask" + side_, (H, W))
        if nm_p is not None:
            g_ = nm_p.astype(np.int16)
            hm_ = nm_m & (g_[..., 1] > 170) & (g_[..., 0] < 120) & (g_[..., 2] < 120)
            ns_hair["noside" + side_] = cv2.GaussianBlur(hm_.astype(np.float32), (0, 0), max(1.5, 1.5 * k)) > 0.5
    if hb is not None:
        sk_src = skinlike(hb, 44) & mhb
        if bh_hair is not None:
            sk_src &= ~ndimage.binary_dilation(bh_hair, iterations=max(1, int(k)))
        sk_bh_s = smooth_region(sk_src, max(2.0, 2.5 * k), (0.1 * HR) ** 2)
    else:
        sk_bh_s = None
    # (the side of the head under the side locks - the temple, the skin before the ear, the ear - as each noside plate
    # draws it: what its hair mask says is no hair, skin and its outlines, within the bald head's outline (no neck, no
    # collar). It goes into the face layer: with the face stopping at its own outline there, the skin seen between the
    # strands floated as pink islands when the locks swung, and the bald plate's ear sat lower and further out)
    side_head = {}
    if side_m.any():
        # (where an ear and the temple can be: beside the head's skin by up to half a head radius - an ear stands out about
        # 0.45 HR from the cheek's outline - from above the brows to the mouth's height; the neck below is the neck's.
        # The bald plate's own outline was no guide: its ear sat elsewhere, and the plate's ear hardly counted as skin)
        yy_h = np.arange(H)[:, None]
        brow_y_ = min(P.pt("brow_l")[1], P.pt("brow_r")[1]) if P.pt("brow_l") is not None and P.pt("brow_r") is not None else EYE_Y_ - 0.4 * HR
        head_any_d = (ndimage.distance_transform_edt(~head_skin) < 0.5 * HR) & (yy_h > brow_y_ - 0.3 * HR) & (yy_h < mouth[1])
        # (under any front hair: the left ear lies under a bang, not under a side lock)
        under_s0 = grow_round(front, 0.06 * HR)
        for pn_ in ("nosideL", "nosideR"):
            pl_s, m_s = plate_full(P, pn_, (H, W))
            if pl_s is None:
                continue
            f_s = pl_s.astype(np.float32)
            sat_s = f_s.max(2) - f_s.min(2)
            lum_s = f_s @ np.array([0.299, 0.587, 0.114], np.float32)
            page_s = plate_page(P, pl_s, m_s, k)
            # (skin - an ear's inner shading is pinker than the cheek: a broad skin test - and the outlines; never the
            # page's white, also where it is a pocket enclosed by hair and collar that the flood from the page does not
            # reach: taken as skin, it gave the face layer arms beside the ear that broke up at every turn)
            whiteish = (pl_s.min(2) > 240) & (sat_s < 14)
            cand_ = (skinlike(pl_s, 60) | ((lum_s < 110) & (sat_s < 130))) & ~whiteish & ~page_s
            if pn_ in ns_hair:
                # (with its hair mask: the silver hair is the skin's colour)
                hair_s = ndimage.binary_dilation(ns_hair[pn_], iterations=max(1, int(k)))
                cand_ &= ~hair_s
            else:
                hair_s = np.zeros((H, W), bool)
            # (inside the head's skin too: an ear the bald plate joins to the face is the head's skin there. The features
            # are drawn over it afterwards from the blank face)
            core = smooth_region(cand_ & m_s & under_s0 & head_any_d, max(2.0, 2.5 * k), (0.05 * HR) ** 2)
            # (only what joins the head's skin: the ear and the temple, no island of the plate beyond them)
            lbc, nbc = ndimage.label(core)
            if nbc:
                touch_c = np.unique(lbc[grow_round(head_skin, 3 * k) & (lbc > 0)])
                core = np.isin(lbc, touch_c[touch_c > 0])
            sh_ = core & ~hair_s & ~page_s
            side_head[pn_] = (pl_s, m_s, sh_)
            print(f"下絵 {pn_} にある、横髪の下の頭の横の部分: {int(sh_.sum())} px"
                  f"{'' if pn_ in ns_hair else '（この下絵の髪のマスクがないので、色だけで判断）'}", flush=True)
    order = []
    face_eye_seen = None
    for n in sorted(owner, key=lambda n: Z[n]):
        if n in ("eye_l", "eye_r", "brow_l", "brow_r", "mouth"):
            continue                                  # (eyes.py / mouth.py / brows: their own stages)
        vis = owner[n]
        if not vis.any():
            continue
        rgb = fig[..., :3].copy()
        al = np.where(vis, alpha, 0).astype(np.uint8)
        hid = np.zeros((H, W), bool)
        src = np.zeros((H, W, 3), np.uint8)
        dbg = {}                                       # (L2D_DEBUG=1: where each layer's pixels came from, work/debug_<n>.npz)
        front_m = under_front(n)
        near = ndimage.binary_dilation(vis, iterations=reach_px) & front_m & ~vis
        if n == "face":
            # under the hair: the bald plate's skin (forehead, temples, ears); under the features: the blank face
            fill_reg = head_zone & ~vis
            # (beside the head's skin under the side locks: the noside plate's side of the head, not the bald plate's ear)
            sh_all = np.zeros((H, W), bool)
            for pn_, (pl_s, m_s, sh_) in side_head.items():
                fill_reg &= ~(m_s & ~head_skin & ~sh_)
                sh_all |= sh_
            fill_reg |= sh_all & ~vis
            if hb is not None:
                seen_bh = ndimage.binary_dilation(sk_bh_s | ~mhb, iterations=int(2 * k))
                # (the face proper - from the brows down - is always whole: where the backhair plate showed no skin
                # there (an eye's lashes, a strand it kept) the face had a hole, a dark diamond when the hair swung)
                face_proper = head_skin & (np.arange(H)[:, None] > EYE_Y_ - 0.35 * HR)
                allow_f = seen_bh | face_proper
                # (under the side locks the plates nosideL / nosideR draw the skin: the backhair plate keeps the side
                # locks, and the temples and cheeks under them had no face)
                # (only under the side locks themselves: their pinkish silver hair is "skin" to the colours as well, and
                # the whole box of each plate became face - a straight edge across the crown at the boxes' top)
                allow_f |= sh_all
                fill_reg &= allow_f
            hid |= fill_reg
            ref = vis & skinlike(A, 40) & skinlike(bald, 44) & mb
            adj = tone_match(bald, A, ref, fill_reg)
            src[fill_reg] = adj[fill_reg]
            # (under the front hair the backhair plate's own drawing - the forehead and the temples as far as its hairline,
            # and the hairline itself: the back hair's hidden part is drawn from the same plate beyond it, so the two
            # meet as the plate draws them. The bald plate redraws the whole head, a little off the picture)
            if hb is not None:
                use_b = fill_reg & mhb & ndimage.binary_dilation(sk_bh_s, iterations=int(2 * k))
                if use_b.any():
                    ref_b = vis & mhb & skinlike(A, 40) & (np.abs(A.astype(np.int16) - hb.astype(np.int16)).max(2) < 40)
                    adj_b = tone_match(hb, A, ref_b, use_b)
                    src[use_b] = adj_b[use_b]
                    print(f"   前髪の下の顔を下絵 backhair から描きました: {int(use_b.sum())} px", flush=True)
            # (under the side locks the noside plates' skin, outlines and ears: made from the picture with only those
            # locks taken away, they lie on it - the bald plate redraws the whole head, and its ear sat lower and
            # further out than the picture's earlobe under the earring)
            # (skin and the outlines only - dark and not strongly coloured: a red collar is dark too, and went into the face
            # as a red smudge by the jaw)
            for pn_, (pl_s, m_s, sh_) in side_head.items():
                use = fill_reg & sh_
                if use.any():
                    ref_s = vis & m_s & skinlike(A, 40) & (np.abs(A.astype(np.int16) - pl_s.astype(np.int16)).max(2) < 40)
                    adj_s = tone_match(pl_s, A, ref_s, use)
                    src[use] = adj_s[use]
                    print(f"   横髪の下の顔を下絵 {pn_} から描きました: {int(use.sum())} px", flush=True)
            if fb is not None:
                fr = feat_all & mf
                src[fr] = fb[fr]
                hid |= fr
            # (under hair that crosses an eye the bald plate draws the eye - a ghost eye in the face that a blink bared:
            # there plain skin, carried smoothly from the skin round the eye)
            ez = np.zeros((H, W), bool)
            for e_ in ("eye_l", "eye_r"):
                if e_ in fl and fl[e_].any():
                    ez |= ndimage.binary_fill_holes(ndimage.binary_closing(fl[e_], iterations=int(0.07 * HR)))
            # (and inside the eye itself the face is skin even where the picture's eye was labelled face - bits of the
            # white and the highlights, light like skin. Kept, they stayed when the eye layers closed over them, and the
            # skin filled round them took their light: a glossy ghost of each eye at every blink, 「目を瞑る時に目が残っ
            # ちゃってる」 (chara4, 2026-10-23). At rest the eye's layers lie over them)
            face_eye_seen = ez & vis
            # (round the eye too, a little beyond its outline: the picture's lines labelled face there - the lower lid's
            # red line ends - stayed as an arc under the closed eye)
            ez_big = grow_round(ez, 0.05 * HR)
            face_eye_seen |= ez_big & vis & ~skinlike(A, 34)
            hid |= face_eye_seen
            ez = ndimage.binary_dilation(ez, iterations=int(0.04 * HR))
            ez_fill = ((ez | ndimage.binary_dilation(hid_feat_all, iterations=int(1.5 * k))) & hid & ~vis) | face_eye_seen
            # (the eye's zone itself: the blank face plate - drawn skin, like all round it. Filled smoothly from the skin
            # round the eye it was a flat disc with a rim, a ghost of the eye under every closed lid)
            if fb is not None:
                use_fb = (ez_big | ez) & mf & (hid | face_eye_seen)
                ref_fb = vis & mf & ~ez_big & grow_round(ez_big, 0.15 * HR) & skinlike(A, 40) & \
                         (np.abs(A.astype(np.int16) - fb.astype(np.int16)).max(2) < 40)
                adj_fb = tone_match(fb, A, ref_fb, use_fb) if ref_fb.sum() > 200 else fb
                src[use_fb] = adj_fb[use_fb]
                ez_fill &= ~use_fb
                print(f"   顔の目のあたりを下絵 face_blank から描きました: {int(use_fb.sum())} px", flush=True)
            if ez_fill.any():
                cur = np.where(hid[..., None], src, rgb)
                known_s = (vis | hid) & ~ez_fill & skinlike(cur, 40)
                ys_e, xs_e = np.nonzero(ez_fill)
                y0e, y1e = max(0, ys_e.min() - int(0.2 * HR)), ys_e.max() + int(0.2 * HR)
                x0e, x1e = max(0, xs_e.min() - int(0.2 * HR)), xs_e.max() + int(0.2 * HR)
                sub = layerio.smooth_fill(cur[y0e:y1e, x0e:x1e], known_s[y0e:y1e, x0e:x1e], ez_fill[y0e:y1e, x0e:x1e])
                src[y0e:y1e, x0e:x1e][ez_fill[y0e:y1e, x0e:x1e]] = sub[ez_fill[y0e:y1e, x0e:x1e]]
        elif n == "body":
            # under the arms the noarms plate, under head / neck / hair the headless and bald plates
            for pn_, pl, m0 in (("bald", bald, mb), ("headless", hl, mhl), ("noarms", na, mn_)):
                if pl is None:
                    continue
                mm = m0
                if pl is bald:
                    # (nothing of the head - its skin, the ears - but what lies beside it, e.g. a hood under the hair tips)
                    mm = mm & ~grow_round(head_any, 0.05 * HR)
                elif pl is na:
                    mm = mm & (np.arange(H)[:, None] > chin[1] + 0.15 * HR) & ~grow_round(head_any, 0.1 * HR)
                elif pl is hl:
                    behind_neck = grow_round(owner["neck"] | (owner["face"] & (np.arange(H)[:, None] > chin[1] - 0.6 * HR)), 0.06 * HR)
                    mm = mm & (np.arange(H)[:, None] > chin[1] - 0.6 * HR) & behind_neck
                # (not the plate's page - white clothing is no page. Its page is flooded over the WHOLE plate: flooded
                # only inside the strip behind the neck, it had no outer page to start from, and the white the headless
                # plate draws where the neck was went into the body - a pure white patch beside the jaw at every turn)
                okp = mm & ~plate_page(P, pl, m0, k)
                reg = okp & front_m & ~vis & ~head_zone
                src[reg] = pl[reg]
                hid |= reg
                dbg["from_" + pn_] = reg
        elif n in ("arm_l", "arm_r"):
            reg = arm_full[n] & ~vis & mb
            src[reg] = bald[reg]
            hid |= reg
        elif n == "hair_back":
            # (hair, not clothing: the plates draw the collar and the hood behind the head too, and a dark collar is
            # within reach of the hair's dark outline colour - nearer the hair's colours than the clothing's. Navy
            # collar pieces in the back hair floated beside the jaw at every nod)
            bpx = rgb[owner["body"] & (alpha > 200)]
            cen_b = colour_model(bpx, 6) if len(bpx) > 500 else None
            def hairish(img, tol):
                dh_ = colour_dist(img, cen)
                return (dh_ <= tol) & (dh_ < colour_dist(img, cen_b)) if cen_b is not None else dh_ <= tol
            if hb is not None:
                # (inside the head's outline too, where the plate draws hair - the crown under the bangs: kept out of the
                # whole head, the bangs bared grey blocks there. Not deep inside the plate's own skin, the face's)
                hcol = hairish(hb, 46) & ~skinlike(hb, 30)
                if bh_hair is not None:
                    # (its hair as its mask says, with its outlines - the pinkish silver is "skin" to the colours)
                    # (and on a little under the face's edge - the hairline: the face turns on its own surface and the
                    # hair on the head's, and the forehead's edge slides a few px over the hair; with the back hair
                    # ending where the face begins, a gap opened there)
                    hcol = (hcol | ndimage.binary_dilation(bh_hair, iterations=max(1, int(1.5 * k)))) & \
                           ~ndimage.binary_erosion(sk_bh_s, iterations=int(0.1 * HR))
                near_h = grow_round(hair, 0.25 * HR)
                reg = mhb & hcol & front_m & ~vis & near_h & ~ndimage.binary_erosion(sk_bh_s, iterations=int(0.1 * HR))
                # (never the face's features: the plate keeps the eyes, and their whites and red lines passed for this
                # hair's colours - bits of eye in the back hair, behind the face)
                reg &= ~grow_round(feat_all, 0.1 * HR)
                # (not under the side locks: the backhair plate keeps them - drawn again here, they showed as a second set
                # of strands beside the swinging locks. The noside plates draw what is there, below)
                for pn_ in ("nosideL", "nosideR"):
                    pl_s, m_s = plate_full(P, pn_, (H, W))
                    if pl_s is not None:
                        reg &= ~(m_s & grow_round(side_m, 0.06 * HR))
                reg = ndimage.binary_opening(reg, iterations=int(k))
                ref = vis & mhb & (np.abs(A.astype(np.int16) - hb.astype(np.int16)).max(2) < 50)
                adj = tone_match(hb, A, ref, reg)
                src[reg] = adj[reg]
                hid |= reg
            # (behind the side locks: the hair the plates nosideL / nosideR draw there - the back of the head behind and
            # above the ears, darker - brought to the picture's tone; where they show skin, the page or the collar,
            # nothing: the face, the ears, the page)
            for pn_ in ("nosideL", "nosideR"):
                pl_s, m_s = plate_full(P, pn_, (H, W))
                if pl_s is None or not side_m.any():
                    continue
                fq_ = pl_s.astype(np.float32)
                # (hair as the plate draws it behind the ears: silver - this hair has red streaks, and the plate's red
                # collar passed for hair: red patches by the jaw at every turn)
                hcol_s = hairish(pl_s, 46) & ~skinlike(pl_s, 30) & ~plate_page(P, pl_s, m_s, k) & ((fq_.max(2) - fq_.min(2)) < 70)
                if pn_ in ns_hair:
                    # (by its own hair mask: the silver hair behind the ear is the skin's colour)
                    hcol_s = ndimage.binary_dilation(ns_hair[pn_], iterations=max(1, int(1.5 * k))) & ~plate_page(P, pl_s, m_s, k) & \
                             ((fq_.max(2) - fq_.min(2)) < 70)
                if pn_ in side_head:
                    hcol_s &= ~side_head[pn_][2]
                reg_s = m_s & hcol_s & front_m & ~vis & ~hid & grow_round(side_m, 0.06 * HR)
                reg_s = ndimage.binary_opening(reg_s, iterations=int(k))
                if reg_s.any():
                    ref_s = vis & m_s & (np.abs(A.astype(np.int16) - pl_s.astype(np.int16)).max(2) < 50)
                    adj_s = tone_match(pl_s, A, ref_s, reg_s)
                    src[reg_s] = adj_s[reg_s]
                    hid |= reg_s
                    print(f"   横髪の後ろの後ろ髪を下絵 {pn_} から描きました: {int(reg_s.sum())} px", flush=True)
            # (behind the front hair, outside the face, the back hair is always there: where the plates show none - the
            # page behind a side lock - a darker copy of the front hair itself. With nothing there a turn with the hair
            # swinging opened thin gaps along the hair's edge; a smooth fill there read as a smear, a copy reads as hair)
            U = owner["hair_front"] & ~head_zone & front_m & ~vis & ~hid & ~skinlike(fig[..., :3], 40)
            # (not with the front hair cut into locks: the copy kept the locks' strands in place, and every swing showed
            # them twice - the hair base under all the hair (hair_base.py) holds the gaps, without lines)
            if lock_ids is not None:
                U &= False
            if U.any():
                src[U] = (fig[..., :3][U].astype(np.float32) * 0.85).astype(np.uint8)
                hid |= U
                print(f"   前髪の後ろの後ろ髪を、前髪を暗くした写しで描きました: {int(U.sum())} px", flush=True)
            # behind the jaw, the ears and the neck, and behind the lower face: the back hair carried on. (A nod moves
            # the face up and the back hair down - by about 0.45 HR between them at full tilt - and a turn slides them
            # apart: what lay behind the face's lower half is bared as a sliver; with nothing there the page showed as a
            # black wedge under the jaw.) Near the seen back hair a smooth continuation of its own colours (its dark
            # outlines left out - averaged in, they made the fill near-black); deeper, its mid-tone.
            yy2 = np.arange(H)[:, None]
            W3_ = np.array([0.299, 0.587, 0.114], np.float32)
            hp = rgb[vis & (alpha > 200)].astype(np.float32)
            lh = hp @ W3_
            mid_c = np.median(hp[(lh >= np.percentile(lh, 35)) & (lh <= np.percentile(lh, 75))], 0)
            # (round: a dilation by a small cross repeated is a diamond, and its grey showed as blocks with straight
            # edges beside the jaw at a turn)
            ring = (ndimage.distance_transform_edt(~(head_zone | neck_reg)) < 0.4 * HR) & ~head_inner & (yy2 > EYE_Y_) & (yy2 < chin[1] + 0.9 * HR)
            deep = head_inner & (yy2 > EYE_Y_ + 0.15 * HR)
            behind = (ring | deep) & front_m & ~vis & ~hid
            # (where the nape plate draws hair behind the lower face, the drawing - brought to the picture's tone; the
            # smooth fill below only where it has none: a fill there read as a grey smear beside a raised chin)
            npl, mnp = plate_full(P, "nape", (H, W))
            if npl is not None and behind.any():
                nape_hair = ndimage.binary_opening(mnp & hairish(npl, 46) & ~skinlike(npl, 40), iterations=int(k))
                nh_ = behind & nape_hair
                # (and no back hair below where the nape plate's hair ends: short hair ends at the nape - filled on
                # down to the shoulders, a hair-coloured slab showed beside the neck at every nod)
                low_ = np.where(nape_hair.any(0), H - 1 - np.argmax(nape_hair[::-1], 0), -1).astype(np.float32)
                low_ = ndimage.median_filter(low_, size=int(0.1 * HR) | 1)
                behind &= (np.arange(H)[:, None] <= low_[None, :] + 0.05 * HR) | ~mnp.any(0)[None, :]
                # (nor where the plate shows the page between the hair's tips, or clothing: the smooth fill closed those
                # gaps with the hair's mid tone - a light patch beside the neck that rose with the head)
                cloth_np = (colour_dist(npl, cen_b) < colour_dist(npl, cen)) & ~skinlike(npl, 40) if cen_b is not None else np.zeros((H, W), bool)
                behind &= ~(mnp & (plate_page(P, npl, mnp, k) | cloth_np))
                if nh_.any():
                    ref_n = vis & mnp & (np.abs(A.astype(np.int16) - npl.astype(np.int16)).max(2) < 50)
                    adj_n = tone_match(npl, A, ref_n, nh_)
                    src[nh_] = adj_n[nh_]
                    hid |= nh_
                    behind &= ~nh_
                    print(f"   顔の下半分の後ろの後ろ髪を下絵 nape から描きました: {int(nh_.sum())} px", flush=True)
            if behind.any():
                lum_all = rgb.astype(np.float32) @ W3_
                src_l = np.where(hid, src.astype(np.float32) @ W3_, lum_all)
                cur_h = np.where(hid[..., None], src, rgb)
                known_h = ((vis & (alpha > 200)) | hid) & (src_l >= np.percentile(lh, 30)) & hairish(cur_h, 46) & ~skinlike(cur_h, 40)
                ys_b, xs_b = np.nonzero(behind | known_h)
                y0b, y1b, x0b, x1b = max(0, ys_b.min() - 4), ys_b.max() + 5, max(0, xs_b.min() - 4), xs_b.max() + 5
                base_ = np.where(hid[..., None], src, rgb)
                sub = layerio.smooth_fill(base_[y0b:y1b, x0b:x1b], known_h[y0b:y1b, x0b:x1b], behind[y0b:y1b, x0b:x1b]).astype(np.float32)
                # (deep inside: going over to the mid-tone, a little darker downwards)
                dd = ndimage.distance_transform_edt(~known_h[y0b:y1b, x0b:x1b])
                w_mid = np.clip(dd / (0.25 * HR), 0, 1)[..., None]
                # (behind the head, in its shadow: well darker than the hair in view - at the hair's own light tone the
                # plain fill read as a pale or brown smear beside the face at a turn, not as hair in shadow)
                shade = 0.74 * (0.94 + 0.06 * np.clip((np.arange(y0b, y1b)[:, None, None] - EYE_Y_) / HR, 0, 1))
                sub = sub * (1 - w_mid) + (mid_c[None, None, :] * shade) * w_mid
                bm = behind[y0b:y1b, x0b:x1b]
                src[y0b:y1b, x0b:x1b][bm] = np.clip(sub[bm], 0, 255).astype(np.uint8)
                hid |= behind
        elif n == "neck":
            # carried up under the face (the first character's rule: its own width and side lines, widening toward the
            # jaw, a soft shade under the chin): up to the jaw line's own height and 0.55 HR more - turned or looking up
            # the face lifts that far; at a constant height the jaw's corners rose above it and the page showed
            W3_ = np.array([0.299, 0.587, 0.114], np.float32)
            rows_n = np.nonzero(vis.any(1))[0]
            ytop = int(rows_n.min())
            # (its width from the rows just under the chin - not from its topmost rows: a strip of shadowed skin beside
            # the jaw reached higher than the neck, and the width and the top came from the strip)
            band_rows = [y for y in range(int(chin[1] + 0.05 * HR), min(int(chin[1] + 0.35 * HR), H)) if vis[y].any()]
            if not band_rows:
                band_rows = [y for y in range(ytop, min(ytop + int(0.25 * HR), H)) if vis[y].any()]
            lefts = np.array([np.nonzero(vis[y])[0].min() for y in band_rows], float)
            rights = np.array([np.nonzero(vis[y])[0].max() for y in band_rows], float)
            xl0, xr0 = float(np.median(lefts)), float(np.median(rights))
            cxn = 0.5 * (xl0 + xr0)
            # (never narrower than a neck - an option, "rig": {"neck_min_width": 0.9} (head radii): under a high collar
            # only a sliver of the neck shows under the chin, and the neck's hidden part made from that sliver was a thin
            # bar (chara4). Not by default: where the picture draws the whole neck with its side lines, the widened part
            # stood outside them as a straight vertical edge when the chin rose (chara5))
            nmw = float(P.cfg.get("rig", {}).get("neck_min_width", 0.0))
            if nmw > 0 and xr0 - xl0 < nmw * HR:
                xl0, xr0 = cxn - 0.5 * nmw * HR, cxn + 0.5 * nmw * HR
            jaw = np.full(W, np.inf)
            hs_cols = np.nonzero(head_skin.any(0))[0]
            for x in hs_cols:
                jaw[x] = np.nonzero(head_skin[:, x])[0].max()
            nk = rgb[vis].astype(np.float32)
            lum_n = nk @ W3_
            skin_n = np.median(nk[(lum_n >= np.percentile(lum_n, 40)) & (lum_n <= np.percentile(lum_n, 85))], 0)
            shadow_n = np.median(nk[lum_n <= np.percentile(lum_n, 20)], 0)
            shade_c = skin_n * 0.82 + shadow_n * 0.18
            line_c = np.median(nk[lum_n <= np.percentile(lum_n, 4)], 0)
            line_w = max(2.0, 0.012 * HR)
            # (how far the jaw rises at a full nod up, per column: the rig shortens what lies below the head's middle by
            # the cosine and lifts it by its depth (the nod's plate, ~0.8 HR) times the sine - 32 deg)
            ph_ = np.radians(32.0)
            lift = lambda jy_: 1.15 * ((jy_ - HY) * (1 - np.cos(ph_)) + 0.8 * HR * np.sin(ph_))
            # (the neck goes up behind the face as itself - its own width, widening a little toward the jaw (the first
            # character's neck: about 0.12 HR each side), never to the face's width - and only as far as the jaw ABOVE each column
            # can lift away from it: it went up to the eyes behind the cheeks, a bright slab that showed at every turn.
            # What a raised chin bares is the jaw's underside: the neck's shadow tone, not its bright skin)
            jaw_mid = float(jaw[int(cxn)]) if np.isfinite(jaw[int(cxn)]) else float(ytop)
            WIDEN = 0.12 * HR
            # (column by column: from the neck's own top in that column up to where the jaw above it can lift away - one
            # top row for all columns came from a stray strip of shadowed skin beside the jaw, and the middle under the
            # chin got nothing: the hair behind showed there like a beard)
            ctop = np.full(W, -1)
            vc = np.nonzero(vis.any(0))[0]
            for x in vc:
                ctop[x] = np.nonzero(vis[:, x])[0].min()
            mid_cols = vc[np.abs(vc - cxn) < 0.6 * (xr0 - xl0)]
            top_med = float(np.median(ctop[mid_cols])) if len(mid_cols) else float(ytop)
            xa_all, xb_all = max(0, int(xl0 - WIDEN)), min(W, int(xr0 + WIDEN) + 1)
            # (from the colour of the neck's own top, going darker - never lighter: from its light median skin it showed
            # a pale jagged seam across the neck under a raised chin. That colour smoothed along the neck - taken column
            # by column it streaked the whole raised neck)
            t0s = np.array([min(ctop[x] if ctop[x] >= 0 else top_med, top_med + 0.15 * HR) for x in range(xa_all, xb_all)], float)
            TC = np.full((xb_all - xa_all, 3), np.nan, np.float32)
            nt_ = max(2, int(0.04 * HR))
            for i_, x in enumerate(range(xa_all, xb_all)):
                y_ = int(t0s[i_])
                tc_ = rgb[y_:y_ + nt_, x][vis[y_:y_ + nt_, x]]
                if len(tc_):
                    TC[i_] = np.median(tc_.astype(np.float32), 0)
            okc = ~np.isnan(TC[:, 0])
            if okc.any():
                xi_ = np.arange(len(TC))
                for ch in range(3):
                    TC[:, ch] = np.interp(xi_, xi_[okc], TC[okc, ch])
                from scipy.ndimage import gaussian_filter1d
                TC = gaussian_filter1d(TC, sigma=max(1.0, 0.06 * HR), axis=0, mode="nearest")
            else:
                TC[:] = shade_c
            dark_c = shade_c * 0.4 + shadow_n * 0.6
            if P.cfg.get("rig", {}).get("neck_tone") == "lit":
                # (lit skin: a neck seen only as a sliver in a high collar's shadow gave a dark brown block under a raised
                # chin; the drawn view from below shows the neck lit, the jaw's cast shadow on it - a layer of its own)
                skin_n = skin_c.astype(np.float32) * 0.95
                shade_c = skin_c.astype(np.float32) * 0.90
                dark_c = skin_c.astype(np.float32) * 0.88
            # (skin in shadow, never darker than 0.72 of the neck's skin: where a high collar leaves only a sliver of neck,
            # deep in the collar's shadow, its darkest pixels gave a brown block under a raised chin)
            l_sk, l_dk = float(skin_n @ W3_), float(dark_c @ W3_)
            if l_dk < 0.72 * l_sk:
                t_dk = (0.72 * l_sk - l_dk) / max(l_sk - l_dk, 1.0)
                dark_c = dark_c * (1 - t_dk) + skin_n * t_dk
            top_mid = np.median(TC, 0)
            if float(dark_c @ W3_) > float(top_mid @ W3_):
                dark_c = top_mid
            # (the top colours too: under a high collar the neck's first rows are the collar's cast shadow)
            l_tc = TC @ W3_
            low_tc = l_tc < (0.92 if P.cfg.get("rig", {}).get("neck_tone") == "lit" else 0.72) * l_sk
            if low_tc.any():
                lim_ = 0.92 if P.cfg.get("rig", {}).get("neck_tone") == "lit" else 0.72
                t_tc = ((lim_ * l_sk - l_tc) / np.maximum(l_sk - l_tc, 1.0)).clip(0, 1)[:, None]
                TC = np.where(low_tc[:, None], TC * (1 - t_tc) + skin_n[None, :] * t_tc, TC)
            for i_, x in enumerate(range(xa_all, xb_all)):
                t0 = t0s[i_]                                 # (never from deep down a column without neck)
                jy = jaw[x] if np.isfinite(jaw[x]) else jaw_mid
                y_from = int(max(0, jy - lift(jy)))
                inside = 1.0 - max(0.0, max(xl0 - x, x - xr0)) / max(WIDEN, 1.0)     # (1 inside its own width)
                top_c = TC[i_]
                for y in range(y_from, int(t0) + int(0.05 * HR)):
                    if vis[y, x]:
                        continue
                    f_ = float(np.clip((t0 - y) / max(t0 - (jaw_mid - 0.3 * HR), 1.0), 0, 1))
                    if inside < 1.0 and inside < 1.0 - f_:        # (the widening only toward the jaw)
                        continue
                    # (in the shadow tone almost at once above its own top: behind the face it is under the jaw, and
                    # bright skin there showed as a pale smear at the face's edge when it turned)
                    # (in the shadow tone almost at once above its own top, flat and crisp: behind the face it is under
                    # the jaw. Bright, it showed as a pale smear at the face's edge at a turn; smoothed across to keep the
                    # neck's shading, it read as a blurred stain - a hidden part is drawn, not blurred, method rule 2)
                    t_ = float(np.clip((t0 - y) / (0.08 * HR), 0, 1))
                    c = top_c * (1 - t_) + dark_c * t_
                    dl = min(x - (xl0 - WIDEN * f_), (xr0 + WIDEN * f_) - x)
                    if 0 <= dl < line_w:
                        lw_ = (1.0 - dl / line_w * 0.5) * (1.0 - f_)
                        c = line_c * lw_ + c * (1.0 - lw_)
                    hid[y, x] = True
                    src[y, x] = c
            hid &= front_m
        # (a hidden part is hidden: only where a part IN FRONT covers it at rest - method rule 1, rest = the picture;
        # the sleeves' hidden edge under hair that lies BEHIND the arms showed as white bands)
        hidden = hid & ~vis & front_m
        rgb[hidden] = src[hidden]
        al[hidden] = 255
        if n in ("hair_front", "hair_back") and hair_repaint is not None:
            rp_ = hair_repaint & vis
            rgb[rp_] = hair_repaint_col[rp_]
        if n == "face" and face_eye_seen is not None and face_eye_seen.any():
            rgb[face_eye_seen] = src[face_eye_seen]
            print(f"   顔に入っていた絵の目のかけらは、目のレイヤーの下なので肌にしました: {int(face_eye_seen.sum())} px", flush=True)
        # run on under what is in front - from the seen part only (a hidden part is drawn as far as it should go) - in
        # a smooth continuation of its colours (nearest colours left streaks and blocks)
        known = vis | hidden
        run = ndimage.binary_dilation(vis, iterations=reach_px) & front_m & ~known
        if n in ("body", "arm_l", "arm_r"):
            run &= ~head_any
        if run.any():
            ys_r, xs_r = np.nonzero(run | known)
            y0r, y1r, x0r, x1r = max(0, ys_r.min() - 4), ys_r.max() + 5, max(0, xs_r.min() - 4), xs_r.max() + 5
            sub = layerio.smooth_fill(rgb[y0r:y1r, x0r:x1r], (known & (al > 200))[y0r:y1r, x0r:x1r], run[y0r:y1r, x0r:x1r])
            rgb[y0r:y1r, x0r:x1r] = sub
        al[run] = 255
        if os.environ.get("L2D_DEBUG"):
            np.savez_compressed(P.p("work", f"debug_{n}.npz"), vis=vis, hidden=hidden, run=run, **dbg)
        layer = np.dstack([rgb, al])
        ys_, xs_ = np.nonzero(al)
        x0_, y0_, x1_, y1_ = xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1
        layerio.save_layer(n, np.ascontiguousarray(layer[y0_:y1_, x0_:x1_]), x0_, y0_)
        order.append(n)
        print(f"レイヤー {n}: 見えている所 {int(vis.sum())} px、隠れた所（描き足し） {int(hidden.sum())} px、"
              f"前のパーツの下へのばした所 {int(run.sum())} px", flush=True)
    json.dump({"order": order, "z": Z}, open(P.p("work", "layer_order.json"), "w"), indent=1)
    # ---- the look: parts tinted + restacked vs the picture
    tint = {"hair_back": (90, 60, 200), "body": (60, 160, 60), "neck": (200, 140, 60), "arm_l": (220, 60, 60), "arm_r": (220, 60, 160),
            "face": (240, 200, 60), "eye_l": (0, 200, 255), "eye_r": (0, 200, 255), "brow_l": (255, 0, 255), "brow_r": (255, 0, 255),
            "mouth": (255, 80, 0), "hair_front": (40, 40, 255)}
    T = A.astype(np.float32).copy()
    for n, c in tint.items():
        m = owner[n]
        T[m] = T[m] * 0.45 + np.array(c, np.float32) * 0.55
    st = np.zeros((H, W, 3), np.float32) + 40
    for n in order:
        L, x0_, y0_ = layerio.load_layer(n)
        h_, w_ = L.shape[:2]
        a_ = L[..., 3:4].astype(np.float32) / 255
        st[y0_:y0_ + h_, x0_:x0_ + w_] = st[y0_:y0_ + h_, x0_:x0_ + w_] * (1 - a_) + L[..., :3] * a_
    for n in ("eye_l", "eye_r", "brow_l", "brow_r", "mouth"):
        m = owner[n]
        st[m] = fig[..., :3][m] * a3[m] + st[m] * (1 - a3[m])
    s = max(1, H // 1400)
    sm = lambda im: cv2.resize(im.astype(np.uint8), (W // s, H // s), interpolation=cv2.INTER_AREA)
    Image.fromarray(np.concatenate([sm(T), sm(st)], 1)).save(P.p("work", "segment_look.png"))
    print("確認用の絵: work/segment_look.png", flush=True)


if __name__ == "__main__":
    main()
