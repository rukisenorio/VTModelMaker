"""Stage 4a: the eyes in parts that blink and look about (the second character's tools/eyes.py, made general).

usage: eyes.py <project>

Per eye (eye_l = the viewer's left), from segment.py's eye mask and the noiris plate:
  eye_?_white    the eyeball: the plate's white inside the opening (+ the picture's shading round it) - squeezed between
                 the lids as the eye closes, and the mask the iris is clipped by
  eye_?_iris     what differs from the plate's white inside the opening, filled, with its soft rim
  eye_?_lash_up  everything of the eye above the opening's middle (the upper lash band), minus the crease
  eye_?_crease   separate dark strokes above the lash (a double eyelid line): a part of its own that fades as the lid falls
  eye_?_lash_low the lower lash
  eye_?_closed / eye_?_closed_smile   the closed eye's line from the eyes_closed / eyes_smile plates (what they draw that
                 the blank face has not), unmixed from the skin
  -> work/eye_geom.json (the opening's top/bottom per column, the iris centre) for the rig's blink (build.py band map)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import common as C
from segment import plate_full, tol_diff
W3 = np.array([0.299, 0.587, 0.114], np.float32)


def closed_line(P, plate, blank, F, hair, win, skin_c, name, k, zone=None):
    """The closed eye's line in a plate: dark strokes on the skin that the blank face does not have, in the window."""
    import layerio
    x0, y0, x1, y1 = win
    p = plate[y0:y1, x0:x1].astype(np.float32)
    b = blank[y0:y1, x0:x1].astype(np.float32)
    lp, lb = p @ W3, b @ W3
    s_l = float(np.array(skin_c, np.float32) @ W3)
    dark = (lp < s_l - 38) & (lb - lp > 28) & ~hair[y0:y1, x0:x1]
    if zone is not None:
        dark &= zone[y0:y1, x0:x1]                    # (only where the eye is: strands' outlines redrawn elsewhere are no line)
    dark = ndimage.binary_opening(dark, iterations=1)
    lab, n = ndimage.label(ndimage.binary_closing(dark, iterations=int(k)))
    if n == 0:
        print("  閉じた目の線が見つかりません:", name)
        return None
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    cy, cx = (y1 - y0) / 2, (x1 - x0) / 2
    keep = []
    for i in range(n):
        if sizes[i] < 25 * k * k:
            continue
        ys, xs = np.nonzero(lab == i + 1)
        if abs(ys.mean() - cy) < 0.45 * (y1 - y0) and abs(xs.mean() - cx) < 0.45 * (x1 - x0):
            keep.append(i + 1)
    m = np.isin(lab, keep)
    if not m.any():
        print("  目の近くに閉じた目の線がありません:", name)
        return None
    # (the line is the longest stroke across the eye; other bits only where they lie along it - within a few px of its
    # band; loose strokes below it pulled the meeting line down and the half-shut lid slid under the eye)
    widths = {i: np.ptp(np.nonzero(lab == i)[1]) for i in keep}
    main = max(keep, key=lambda i: widths[i])
    mm_ = lab == main
    band = ndimage.binary_dilation(mm_, iterations=int(3 * k))
    m = mm_ | (m & band)
    m = ndimage.binary_dilation(m, iterations=int(1.5 * k))
    # unmix from the skin: the line's own colour x coverage along the skin->line axis
    line = np.median(p[m & (lp < np.percentile(lp[m], 35))], 0)
    S = b
    axis = line[None, None, :] - S
    cov = ((p - S) * axis).sum(2) / np.maximum((axis * axis).sum(2), 1.0)
    cov = np.clip(cov, 0, 1) * m
    out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    col = np.where(cov[..., None] > 0.05, np.clip((p - S * (1 - cov[..., None])) / np.maximum(cov[..., None], 0.05), 0, 255), line)
    out[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
    out[..., 3] = (cov * 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    if out[..., 3].max() == 0:
        return None
    mm = layerio.save_layer(name, out, x0, y0)
    print(f"{name:20s} {mm['w']}x{mm['h']}、位置 ({mm['x0']},{mm['y0']})")
    return mm


SEAT = 0.88        # (where the lids meet: about this far down the open eye's opening, column by column ...)
SEAT_BOW = 0.4     # (... as a gentle arc about the line fitted through that curve: this much of its bow kept. A round
                   # eye's lower half is a deep U - followed fully, the closed eye was a U or, tilted, a swoosh; drawn
                   # from corner to corner, the lids met in the eye's upper half and the lower lid rose half the eye.
                   # So the corners come down a little as the eye closes, as drawn closed eyes sit lower than open ones)


def seat_closed(name, gx, gu, gl):
    """The drawn closed line moved, column by column, onto the line where the lids meet: SEAT of the way down the open
    eye's opening, its own shape and weight kept. Codex draws the closed eye where it likes - chara4's left one in the
    opening's upper third, its right one as a straight diagonal across the iris: the lower lid rose half the eye to meet
    it, and the eye slid up and sideways as it closed (「目の開閉がずれてます」, 2026-10-23). Beyond the eye's corners
    the line's ends move as the corner next to them."""
    import layerio
    if name not in layerio.manifest():
        return
    L, lx0, ly0 = layerio.load_layer(name)
    a = L[..., 3] > 100
    cols = np.nonzero(a.any(0))[0]
    if len(cols) < 8:
        return
    bottom = np.array([np.nonzero(a[:, c])[0].max() for c in cols], np.float32) + ly0
    bottom = ndimage.gaussian_filter1d(ndimage.median_filter(bottom, size=9, mode="nearest"), 3.0, mode="nearest")
    xs = (cols + lx0).astype(np.float32)
    inside = (xs >= gx[0]) & (xs <= gx[-1])
    if inside.sum() < 5:
        return
    curve = gu + SEAT * (gl - gu)
    axis_ = np.polyval(np.polyfit(gx, curve, 1), gx)                    # (the eye's tilt, through its lower part)
    T = np.interp(xs, gx, axis_ + SEAT_BOW * (curve - axis_))
    sh = np.where(inside, T - bottom, np.nan)
    ii = np.nonzero(inside)[0]
    sh[:ii[0]] = sh[ii[0]]
    sh[ii[-1] + 1:] = sh[ii[-1]]
    sh = ndimage.gaussian_filter1d(sh, 4.0, mode="nearest")
    shift = np.interp(np.arange(L.shape[1]), cols, sh).astype(np.float32)
    pad = int(np.ceil(np.abs(shift).max())) + 4
    Lp = np.pad(L, ((pad, pad), (0, 0), (0, 0)))
    hh, ww = Lp.shape[:2]
    mx, my = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    out = cv2.remap(Lp, mx, my - shift[None, :], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mm = layerio.save_layer(name, out, lx0, ly0 - pad)
    print(f"{name:20s} まぶたの合う線（目の開きの {SEAT:.2f} の高さ）に合わせて {float(sh.min()):+.0f}〜{float(sh.max()):+.0f} px "
          f"動かしました → {mm['w']}x{mm['h']}、位置 ({mm['x0']},{mm['y0']})")


def drop_hair_from_eyes(P):
    """Hair in the eye layers, where the front hair covers them at rest: the upper lash loses what copies the hair there
    (within 40 of its colour) and keeps its own drawn band under it - the band comes down into view; the lower lash and
    the crease keep nothing under the hair - they only fade. Strands' outlines crossing the eyes had gone into those
    layers and slid out from under the hair with the lids at every blink (chara4's left crease was 80 % hair lines;
    the other session's chara6, 2026-10-23). Run after the locks stage: the hair is its layers hair_front and lock_*."""
    import layerio
    man = layerio.manifest()
    order = json.load(open(P.p("work", "layer_order.json")))
    hair = sorted([n for n in man if n.startswith("hair_front") or n.startswith("lock_")], key=lambda n: order["z"].get(n, 40))
    if not hair:
        return
    for n in sorted(man):
        kind = n[6:] if n.startswith("eye_") and len(n) > 6 else ""
        if kind not in ("lash_up", "lash_low", "crease"):
            continue
        L, lx0, ly0 = layerio.load_layer(n)
        h_, w_ = L.shape[:2]
        Ha = np.zeros((h_, w_), np.float32)
        Hc = np.zeros((h_, w_, 3), np.float32)
        for hn in hair:                                    # (back to front: the topmost hair's colour wins)
            Hl, hx0, hy0 = layerio.load_layer(hn)
            x0, y0 = max(lx0, hx0), max(ly0, hy0)
            x1, y1 = min(lx0 + w_, hx0 + Hl.shape[1]), min(ly0 + h_, hy0 + Hl.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            hs = Hl[y0 - hy0:y1 - hy0, x0 - hx0:x1 - hx0]
            ls = (slice(y0 - ly0, y1 - ly0), slice(x0 - lx0, x1 - lx0))
            top = hs[..., 3] > 128
            Hc[ls][top] = hs[..., :3][top]
            Ha[ls] = np.maximum(Ha[ls], hs[..., 3].astype(np.float32))
        under = (Ha > 128) & (L[..., 3] > 0)
        if kind == "lash_up":
            under &= np.abs(L[..., :3].astype(np.float32) - Hc).max(2) < 40
        if not under.any():
            continue
        # (and the crumbs that leave: bits under 12 px next to it - the hair line's soft edge beyond the hair's cover)
        rest = (L[..., 3] > 0) & ~under
        lb_, n_ = ndimage.label(rest, structure=np.ones((3, 3), bool))
        if n_:
            near = ndimage.binary_dilation(under, iterations=3)
            szs = ndimage.sum(rest, lb_, range(1, n_ + 1))
            small = [k_ for k_ in np.unique(lb_[near & rest]) if k_ > 0 and szs[k_ - 1] < 12]
            under |= np.isin(lb_, small)
        L = L.copy()
        L[under] = 0
        if not (L[..., 3] > 0).any():
            # (all of it was hair: no such part)
            layerio.drop_layer(n)
            print(f"{n:20s} 前髪の下で全部が髪だったので、このレイヤーはなくしました", flush=True)
            continue
        layerio.save_layer(n, L, lx0, ly0, crop=False)
        print(f"{n:20s} 髪を取り除きました: {int(under.sum())} px", flush=True)


def main():
    P = C.open_project()
    import layerio
    fig = C.load_rgba(P.p("work", "figure.png"))
    H, W = fig.shape[:2]
    a3 = fig[..., 3:4].astype(np.float32) / 255.0
    A = (fig[..., :3].astype(np.float32) * a3 + 255.0 * (1 - a3)).astype(np.uint8)
    k = P.kpx                    # (the pixel unit for tolerances, not the point scale)
    HX, HY, HR = P.head()
    fm = np.load(P.p("work", "feature_masks.npz"))
    lab = np.asarray(Image.open(P.p("work", "labels.png")))
    ids = json.load(open(P.p("work", "labels.json")))["id"]
    hair = (lab == ids["hair_front"]) | (lab == ids["hair_back"])
    noiris, mni = plate_full(P, "noiris", (H, W))
    blank, mbl = plate_full(P, "face_blank", (H, W))
    closed, _ = plate_full(P, "eyes_closed", (H, W))
    smile, _ = plate_full(P, "eyes_smile", (H, W))
    clear, mcl = plate_full(P, "eyes_clear", (H, W))
    el, er = P.pt("eye_l"), P.pt("eye_r")
    skin_c = np.median(np.concatenate([A[int(e[1] + 0.35 * HR):int(e[1] + 0.55 * HR), int(e[0] - 0.12 * HR):int(e[0] + 0.12 * HR)].reshape(-1, 3) for e in (el, er)]), 0)
    HALF = (int(0.58 * HR), int(0.45 * HR))
    geom = {}
    for side, c in (("l", el), ("r", er)):
        # (the eye's own mask for its window, outline and zones; what segment added to it beyond - lash strokes above it,
        # its lines on the hair - joins the lashes at the end. Taken into the zones, they pulled the hair round the eye
        # into its layers)
        eye_all = fm[f"eye_{side}"]
        eye_m = fm[f"eye_{side}_core"] if f"eye_{side}_core" in fm.files and fm[f"eye_{side}_core"].any() else eye_all
        extras = eye_all & ~eye_m
        ys_, xs_ = np.nonzero(eye_m)
        if len(ys_) < 50:
            print(f"eye_{side}: 目のマスクがないので飛ばします"); continue
        cx, cy = 0.5 * (xs_.min() + xs_.max()), 0.5 * (ys_.min() + ys_.max())
        x0, y0, x1, y1 = int(cx - HALF[0]), int(cy - HALF[1]), int(cx + HALF[0]), int(cy + HALF[1])
        if extras.any():
            ye_, xe_ = np.nonzero(extras)
            x0, y0 = min(x0, int(xe_.min()) - 4), min(y0, int(ye_.min()) - 4)
            x1, y1 = max(x1, int(xe_.max()) + 5), max(y1, int(ye_.max()) + 5)
            print(f"eye_{side}: 目のマスクの外の {int(extras.sum())} px をまつ毛に加えました")
        p = noiris[y0:y1, x0:x1].astype(np.int16)
        # (hair over the eye: what the eye is under it comes from the eyes_clear plate, brought to the picture's tone -
        # with holes there, the eye showed dark marks when the hair swung away; rule 2: hidden parts are drawn)
        Ac = A.copy()
        figc = fig.copy()
        eye_full = eye_m.copy()
        zone = C.hull_mask(eye_m)
        cover = zone & hair & ~eye_m
        if clear is not None and cover.any():
            from segment import tone_match
            ref = eye_m & mcl & (np.abs(A.astype(np.int16) - clear.astype(np.int16)).max(2) < 50)
            adj = tone_match(clear, A, ref, cover)
            under = cover & mcl
            Ac[under] = adj[under]
            not_skin = np.abs(Ac.astype(np.float32) - skin_c[None, None, :]).max(2) > 34
            add = under & not_skin
            eye_full |= add
            figc[add, :3] = Ac[add]
            figc[add, 3] = 255
            print(f"eye_{side}: 髪の下の目 {int(add.sum())} px を下絵 eyes_clear から描きました")
        hk = f"eye_{side}_hidden"
        if hk in fm.files and fm[hk].any():
            from segment import tone_match
            bald_p, mbp = plate_full(P, "bald", (H, W))
            hsm = fm[hk] & ~eye_full
            if bald_p is not None and hsm.any():
                ref = eye_m & mbp & (np.abs(A.astype(np.int16) - bald_p.astype(np.int16)).max(2) < 50)
                adj = tone_match(bald_p, A, ref, hsm)
                Ac[hsm] = adj[hsm]
                eye_full |= hsm
                figc[hsm, :3] = adj[hsm]
                figc[hsm, 3] = 255
                print(f"eye_{side}: 髪の下の {int(hsm.sum())} px を下絵 bald から描きました（目の輪郭の外）")
        b = Ac[y0:y1, x0:x1].astype(np.int16)
        eye = eye_full[y0:y1, x0:x1]
        Fw = figc[y0:y1, x0:x1]
        # the plate's white: light and neutral inside the eye's outline, the piece nearest the eye's middle
        mn, mx = p.min(2), p.max(2)
        # (cooler than the skin, and inside the eye: a pale skin is as light and as grey as the white of an eye - taken
        # with it, the eye's opening ran out over the cheek and the lids met in the wrong place)
        warm_p = p[..., 0].astype(np.int16) - p[..., 2]
        skin_w = float(skin_c[0] - skin_c[2])
        cool = warm_p < skin_w - max(5.0, 0.45 * skin_w)
        neutral = ndimage.binary_opening((mn > 150) & (mx - mn < 26) & cool, iterations=2) & ndimage.binary_dilation(eye, iterations=int(2 * k))
        neutral &= ndimage.binary_dilation(zone[y0:y1, x0:x1], iterations=int(1.5 * k))
        lb, n = ndimage.label(neutral)
        sizes = ndimage.sum(neutral, lb, range(1, n + 1)) if n else []
        zone_w = ndimage.binary_dilation(zone[y0:y1, x0:x1], iterations=int(2 * k))
        keep_w = []
        for i in range(n):
            if sizes[i] < 60 * k * k:
                continue
            if ((lb == i + 1) & zone_w).sum() > 0.5 * sizes[i]:
                keep_w.append(i + 1)
        if not keep_w:
            print(f"eye_{side}: 下絵 noiris に白目が見つからないので飛ばします"); continue
        # (every piece of white inside the eye: a strand left in the plate cut the white in two, and the far piece was
        # dropped - a notch of the eye that showed as a hole)
        O = ndimage.binary_fill_holes(ndimage.binary_closing(np.isin(lb, keep_w), iterations=max(int(1.5 * k), int(0.04 * HR))))
        cols = np.nonzero(O.any(0))[0]
        top = np.array([np.nonzero(O[:, cc])[0].min() for cc in cols], float)
        bot = np.array([np.nonzero(O[:, cc])[0].max() for cc in cols], float)
        # the eyeball reaches up to the lash's dark: the grey band under it (the iris's top seen through) is the eyeball's
        lp = (p * W3).sum(2)
        walk = np.zeros(len(cols))
        for j, cc in enumerate(cols):
            t = int(top[j]); n_ = 0
            while n_ < int(6.5 * k) and t - n_ - 1 >= 0 and lp[t - n_ - 1, cc] > 120:
                n_ += 1
            walk[j] = top[j] - n_
        walk = ndimage.median_filter(walk, size=max(3, int(4 * k) | 1), mode="nearest")
        keep = np.ones(len(cols), bool)
        for _ in range(3):
            cf = np.polyfit(cols[keep], walk[keep], 3)
            res = np.abs(np.polyval(cf, cols) - walk)
            keep = res < max(3.0, 2.5 * np.median(res[keep]))
        top_e = np.minimum(np.polyval(cf, cols) + 0.5 * k, top)
        rows_ = np.arange(O.shape[0])[:, None]
        band = np.zeros_like(O)
        band[:, cols] = (rows_ >= np.ceil(top_e)[None, :]) & (rows_ <= top[None, :])
        E = (O | (band & eye)) & ndimage.binary_dilation(eye, iterations=int(k))
        # (the eyeball never takes the picture's own lash band: a plate that also removed strands redrew the eye a little
        # higher, its white reached into the band, and that part of the lash went into the white layer - squeezed with
        # it, the lash fell apart when the eye closed. The band = the eye's darkest pixels in its upper half that hang
        # on to its top edge)
        lumP = b.astype(np.float32) @ W3
        if eye.any():
            thr = float(np.percentile(lumP[eye], 30))
            rows_w = np.arange(eye.shape[0])[:, None]
            midc = np.full(eye.shape[1], np.inf)
            midc[cols] = 0.5 * (top + bot)
            top_eye = np.full(eye.shape[1], np.inf)
            ce = np.nonzero(eye.any(0))[0]
            top_eye[ce] = [np.nonzero(eye[:, c_])[0].min() for c_ in ce]
            dark_up = eye & (lumP <= thr) & (rows_w < midc[None, :])
            # (lash, not iris: nearer the lash's colour - its top edge - than the iris's; the iris's dark rim under the
            # lash is as dark as the lash, and cut away with it the iris had a bite)
            top_band = eye & (rows_w <= top_eye[None, :] + 2 * k)
            iris_guess = E & (np.abs(b - p).max(2) > 34)
            if top_band.sum() > 20 and iris_guess.sum() > 20:
                lash_c = np.median(b[top_band].astype(np.float32), 0)
                iris_c = np.median(b[iris_guess].astype(np.float32), 0)
                bf = b.astype(np.float32)
                dl_ = np.abs(bf - lash_c).max(2); di_ = np.abs(bf - iris_c).max(2)
                dark_up &= dl_ < di_
            lbd, nd_ = ndimage.label(dark_up)
            near_top = dark_up & (rows_w <= top_eye[None, :] + 3 * k)
            keep_d = np.unique(lbd[near_top])
            band_pic = np.isin(lbd, keep_d[keep_d > 0])
            cut = ndimage.binary_dilation(band_pic, iterations=1) & E
            E &= ~cut
            if cut.sum():
                print(f"eye_{side}: 絵のまつ毛の帯 {int(cut.sum())} px を白目から外しました")
        # the iris: what differs from the plate's white inside the eyeball, filled, with its soft rim
        d = np.abs(b - p).max(2)
        ir = E & (d > 34)
        ir = ndimage.binary_opening(ir, iterations=max(1, int(k / 2)))
        lbi, ni = ndimage.label(ir)
        if ni:
            si = ndimage.sum(ir, lbi, range(1, ni + 1))
            ir = lbi == (int(np.argmax(si)) + 1)
        ib = ndimage.binary_fill_holes(ndimage.binary_closing(ir, iterations=int(1.5 * k))) & E
        ib = ndimage.binary_dilation(ib, iterations=max(1, int(k / 2))) & E

        def put(name, rgb, m, soft_from=None):
            out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            out[m, :3] = rgb[m]
            a = np.full(m.shape, 255, np.uint8) if soft_from is None else soft_from
            out[m, 3] = a[m]
            out[out[..., 3] == 0] = 0
            if out[..., 3].max() == 0:
                return
            mm = layerio.save_layer(name, out, x0, y0)
            print("%-16s %4d x %4d、位置 (%d, %d)  %d px" % (name, mm["w"], mm["h"], mm["x0"], mm["y0"], int(m.sum())))
        ring = ndimage.binary_dilation(E, iterations=max(1, int(0.8 * k))) & eye & ~E

        def carried(rgb, m, more):
            idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
            out = rgb.astype(np.uint8).copy()
            out[more] = rgb[idx[0][more], idx[1][more]]
            return out
        near = ndimage.binary_dilation(ib, iterations=int(2.5 * k)) & E & ~ib
        off = np.median((b - p)[near], axis=0) if near.any() else np.zeros(3)
        p_adj = np.clip(p + off, 0, 255)
        white_src = np.where(ib[..., None], p_adj, b).astype(np.int16)
        put(f"eye_{side}_white", carried(white_src, E, ring), E | ring)
        more = ndimage.binary_dilation(ib, iterations=max(1, int(0.8 * k))) & ring
        put(f"eye_{side}_iris", carried(b, ib, more), ib | more)
        # the lashes: the eye's own pixels outside the eyeball, split at the eyeball's middle
        mid = np.full(E.shape[1], np.nan)
        for j, cc in enumerate(cols):
            mid[cc] = 0.5 * (top_e[j] + bot[j])
        xs_all = np.arange(E.shape[1])
        good = ~np.isnan(mid)
        midf = np.interp(xs_all, xs_all[good], mid[good])
        rows = np.arange(E.shape[0])[:, None]
        rest = (eye | (extras[y0:y1, x0:x1] & (Fw[..., 3] > 0))) & ~E
        up = rest & (rows < midf[None, :])
        low = rest & (rows >= midf[None, :])
        # the crease: separate dark strokes above the upper lash
        lumF = Fw[..., :3].astype(np.float32) @ W3
        dark = up & ((lumF < 185) | (lumF < ndimage.median_filter(lumF, size=15) - 6))
        ld, nd = ndimage.label(dark, structure=np.ones((3, 3), bool))
        crease = np.zeros_like(up)
        if nd > 1:
            szd = ndimage.sum(dark, ld, range(1, nd + 1))
            main_ = ld == (int(np.argmax(szd)) + 1)
            mtop = np.full(up.shape[1], np.inf)
            mc = np.nonzero(main_.any(0))[0]
            mtop[mc] = [np.nonzero(main_[:, cc])[0].min() for cc in mc]
            for i in range(nd):
                if szd[i] < 8 * k * k:
                    continue
                comp = ld == i + 1
                if (comp & main_).any():
                    continue
                yy, xx = np.nonzero(comp)
                if np.median(yy) < np.median(np.where(np.isfinite(mtop[xx]), mtop[xx], np.inf)) or not np.isfinite(mtop[xx]).any():
                    crease |= comp
            crease = ndimage.binary_dilation(crease, iterations=max(1, int(0.8 * k))) & up & ~ndimage.binary_dilation(main_, iterations=max(1, int(k / 2)))
        if crease.any():
            put(f"eye_{side}_crease", Fw[..., :3], crease, soft_from=Fw[..., 3])
        up &= ~crease
        put(f"eye_{side}_lash_up", Fw[..., :3], up, soft_from=Fw[..., 3])
        put(f"eye_{side}_lash_low", Fw[..., :3], low, soft_from=Fw[..., 3])
        geom[side] = {"x": [int(cc + x0) for cc in cols], "up": [float(t + y0) for t in top_e], "low": [float(t + y0) for t in bot],
                      "iris_c": [float(np.nonzero(ib)[1].mean() + x0), float(np.nonzero(ib)[0].mean() + y0)] if ib.any() else [float(c[0]), float(c[1])]}
        # the closed eye's lines from the plates (window a little larger: the closed line lies along the lower lid)
        win = (int(cx - 1.25 * HALF[0]), int(cy - HALF[1]), int(cx + 1.25 * HALF[0]), int(cy + 1.3 * HALF[1]))
        zc = ndimage.binary_dilation(zone, structure=np.ones((int(0.12 * HR) | 1, int(0.03 * HR) | 1), bool))
        if closed is not None and blank is not None:
            if closed_line(P, closed, blank, None, hair, win, skin_c, f"eye_{side}_closed", k, zone=zc):
                g_ = geom[side]
                seat_closed(f"eye_{side}_closed", np.array(g_["x"], np.float32), np.array(g_["up"], np.float32), np.array(g_["low"], np.float32))
        if smile is not None and blank is not None:
            closed_line(P, smile, blank, None, hair, win, skin_c, f"eye_{side}_closed_smile", k, zone=zc)
    json.dump(geom, open(P.p("work", "eye_geom.json"), "w"), indent=1)
    print("目の形のデータ: work/eye_geom.json")
    drop_hair_from_eyes(P)


if __name__ == "__main__":
    main()
