"""Stage 4b: brows and the mouth (the mouth as the first and second characters have it: one line bent into every shape).

usage: face_parts.py <project>

  brow_l / brow_r   the picture's brows (segment.py's masks), soft edges from the cut-out
  mouth_closed      the closed mouth's line: the picture's own when its mouth is closed (a line, much wider than tall);
                    when the picture's mouth is open, the "mouth_closed" plate's (made by plates.py on request:
                    character.json "plates": {"mouth_closed": {...}}) - unmixed from the skin
  work/mouth_line.json   the line's geometry (ends, polynomial, thickness) for kit/mouth_rig.py and the rig
  mouth_stroke / mouth_in / mouth_tongue / mouth_teeth / mouth_teethlow / mouth_lipring   texture sources the rig bends
                    (placed off the figure; only their pictures matter)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import common as C
from segment import plate_full
W3 = np.array([0.299, 0.587, 0.114], np.float32)


def unmix_line(pix, skin, m):
    """RGBA of a dark line drawn on skin (pix, skin: HxWx3 float), coverage along the skin->line axis inside m."""
    lp = pix @ W3
    line = np.median(pix[m & (lp <= np.percentile(lp[m], 35))], 0)
    axis = line[None, None, :] - skin
    cov = np.clip(((pix - skin) * axis).sum(2) / np.maximum((axis * axis).sum(2), 1.0), 0, 1) * m
    col = np.where(cov[..., None] > 0.05, np.clip((pix - skin * (1 - cov[..., None])) / np.maximum(cov[..., None], 0.05), 0, 255), line)
    out = np.zeros(pix.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
    out[..., 3] = (cov * 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    return out


def main():
    P = C.open_project()
    import layerio
    fig = C.load_rgba(P.p("work", "figure.png"))
    H, W = fig.shape[:2]
    a3 = fig[..., 3:4].astype(np.float32) / 255.0
    A = (fig[..., :3].astype(np.float32) * a3 + 255.0 * (1 - a3)).astype(np.uint8)
    k = P.kpx                    # (the pixel unit for tolerances, not the point scale)
    HX, HY, HR = P.head()
    u = P.unit()
    fm = np.load(P.p("work", "feature_masks.npz"))
    blank, mbl = plate_full(P, "face_blank", (H, W))
    # ---- brows: the picture's pixels, a margin of their soft edge
    bald_p, mbp = plate_full(P, "bald", (H, W))
    skin_c = np.median(A[int(P.pt("eye_l")[1] + 0.35 * HR):int(P.pt("eye_l")[1] + 0.55 * HR),
                         int(P.pt("eye_l")[0] - 0.12 * HR):int(P.pt("eye_l")[0] + 0.12 * HR)].reshape(-1, 3), 0).astype(np.float32)
    for s in ("l", "r"):
        m = fm[f"brow_{s}"]
        hk = f"brow_{s}_hidden"
        mh = fm[hk] if hk in fm.files else np.zeros_like(m)
        if m.sum() < 20:
            layerio.drop_layer(f"brow_{s}")
            print(f"brow_{s}: 眉が見つからないので、なしにします"); continue
        m = ndimage.binary_dilation(m, iterations=max(1, int(k / 2))) & (fig[..., 3] > 0)
        # (what the hair hides of the brow at rest, from the bald plate: under the hair, it rises with the brow - left
        # in the face, the brow came apart when it moved)
        mh = mh & ~m
        allm = m | mh
        ys, xs = np.nonzero(allm)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        pix = A[y0:y1, x0:x1].astype(np.float32)
        sk = blank[y0:y1, x0:x1].astype(np.float32) if blank is not None else np.full_like(pix, 230)
        if mh.any() and bald_p is not None:
            mhw = mh[y0:y1, x0:x1]
            pix[mhw] = bald_p[y0:y1, x0:x1][mhw]
            sk[mhw] = skin_c
            print(f"brow_{s}: 髪の下の {int(mh.sum())} px を下絵 bald から描きました")
        out = unmix_line(pix, sk, allm[y0:y1, x0:x1])
        mm = layerio.save_layer(f"brow_{s}", out, x0, y0)
        print(f"brow_{s}: {mm['w']}x{mm['h']}、位置 ({mm['x0']},{mm['y0']})")
    # ---- the closed mouth's line
    m = fm["mouth"]
    ys, xs = np.nonzero(m)
    use_plate = False
    if len(ys) and (ys.max() - ys.min() + 1) > 0.38 * (xs.max() - xs.min() + 1):
        use_plate = True
    if use_plate:
        mc, mmc = plate_full(P, "mouth_closed", (H, W))
        if mc is None:
            sys.exit("絵の口が開いています。下絵 'mouth_closed' を加えて（character.json の plates）plates.py を実行してから、もう一度これを実行してください")
        mouth_c = P.pt("mouth")
        x0, y0 = int(mouth_c[0] - 0.5 * HR), int(mouth_c[1] - 0.3 * HR)
        x1, y1 = int(mouth_c[0] + 0.5 * HR), int(mouth_c[1] + 0.3 * HR)
        pix = mc[y0:y1, x0:x1].astype(np.float32)
        sk = blank[y0:y1, x0:x1].astype(np.float32)
        lm = ((sk @ W3) - (pix @ W3) > 28)
        lm = ndimage.binary_dilation(ndimage.binary_opening(lm, iterations=1), iterations=max(1, int(k / 2)))
        out = unmix_line(pix, sk, lm)
    else:
        m = ndimage.binary_dilation(m, iterations=max(1, int(k / 2)))
        x0, y0, x1, y1 = xs.min() - int(k), ys.min() - int(k), xs.max() + 1 + int(k), ys.max() + 1 + int(k)
        pix = A[y0:y1, x0:x1].astype(np.float32)
        sk = blank[y0:y1, x0:x1].astype(np.float32) if blank is not None else np.full_like(pix, 230)
        out = unmix_line(pix, sk, m[y0:y1, x0:x1])
    layerio.save_layer("mouth_closed", out, x0, y0)
    Lc, cx0, cy0 = layerio.load_layer("mouth_closed")
    a = Lc[..., 3].astype(np.float32) / 255.0
    ink = a.sum(0)
    cols = np.nonzero(ink > 0.5)[0]
    cols = np.arange(cols.min(), cols.max() + 1)
    yy = np.arange(a.shape[0])[:, None]
    yc = (a * yy).sum(0)[cols] / np.maximum(ink[cols], 1e-6) + 0.5
    xs_ = cols + cx0 + 0.5
    xc = float(xs_.mean())
    pf = np.polyfit(xs_ - xc, yc + cy0, 4)
    r_a, r_b = 0.5 * float(ink[cols[1:5]].mean()), 0.5 * float(ink[cols[-5:-1]].mean())
    xa, xb = float(cols.min() + cx0) + r_a, float(cols.max() + 1 + cx0) - r_b
    line_rgb = np.median(Lc[..., :3][a > 0.9], axis=0) if (a > 0.9).any() else np.median(Lc[..., :3][a > 0.5], axis=0)
    hw = float(np.clip(1.6 * np.percentile(ink[cols], 90), 3.0, 14.0))
    mg = {"xa": xa, "xb": xb, "xc": xc, "pf": pf.tolist(), "ink_x": xs_.tolist(), "ink": gaussian_filter1d(ink[cols], 2.0, mode="nearest").tolist(),
          "line_rgb": line_rgb.tolist(), "y_mid": float(np.polyval(pf, 0.5 * (xa + xb) - xc)), "hw": hw, "from_plate": use_plate}
    json.dump(mg, open(P.p("work", "mouth_line.json"), "w"))
    print("口の線: x %.1f〜%.1f、真ん中の y %.1f、太さ %.1f〜%.1f px、帯の半分の幅 %.1f、色 %s" % (
        xa, xb, mg["y_mid"], ink[cols].min(), ink[cols].max(), hw, np.round(line_rgb).astype(int).tolist()))
    # ---- the textures (the second character's tools/mouth_parts.py)
    import mouth_rig
    mouth_rig.HW = hw
    G = mouth_rig.Geom(mg)
    mx, my, ok = G.texture_map(1)
    sx, sy = mx - cx0 - 0.5, my - cy0 - 0.5
    pm = cv2.remap(Lc[..., :3].astype(np.float32) * a[..., None], sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    at = np.clip(cv2.remap(a, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1) * ok
    col = np.where(at[..., None] > 0.1, pm / np.maximum(at[..., None], 0.1), 0.0)
    seen = at > 0.1
    if seen.any():
        idx = ndimage.distance_transform_edt(~seen, return_distances=False, return_indices=True)
        col = col[idx[0], idx[1]]
    half = 1.0 - np.sqrt(1.0 - np.clip(at, 0, 1))
    stroke = np.clip(np.dstack([col, half * 255.0]) + 0.5, 0, 255).astype(np.uint8)
    SPOT = (20, H - 300)
    layerio.save_layer("mouth_stroke", stroke, SPOT[0], SPOT[1], crop=False)
    line = np.array(line_rgb, float)
    INTERIOR = np.clip(0.55 * np.array([178.0, 78.0, 78.0]) + 0.45 * (line * 1.6), 0, 255)
    TONGUE = np.array([226.0, 146.0, 146.0])
    TEETH = np.array([250.0, 244.0, 240.0])
    LINE_OPEN = np.clip(line * 0.9, 0, 255)
    h, w = 64, 96
    g = np.linspace(0.84, 1.05, h)[:, None, None]
    tex = np.clip(INTERIOR[None, None, :] * g, 0, 255) * np.ones((h, w, 1))
    layerio.save_layer("mouth_in", np.dstack([tex, np.full((h, w), 255.0)]).astype(np.uint8), SPOT[0] + 200, SPOT[1], crop=False)
    h, w = 80, 120
    yy2, xx2 = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx2 - 59.5) / 55.0) ** 2 + ((yy2 - 66.0) / 60.0) ** 2)
    al = np.clip((1.0 - d) * 55.0 / 1.2, 0, 1) * 255.0
    shade = 1.0 - 0.10 * np.clip(d, 0, 1) ** 2
    hl = 0.06 * np.exp(-(((xx2 - 59.5) / 26.0) ** 2 + ((yy2 - 34.0) / 12.0) ** 2))
    tex = np.clip(TONGUE[None, None, :] * (shade + hl)[..., None], 0, 255)
    layerio.save_layer("mouth_tongue", np.dstack([tex, al]).astype(np.uint8), SPOT[0] + 350, SPOT[1], crop=False)
    # the upper teeth: one plain white band along the upper lip (no lines between teeth - 「歯は一枚でいい」)
    h, w = 48, 200
    BAND = 0.36
    yy3, xx3 = np.mgrid[0:h, 0:w].astype(np.float32)
    band = np.clip((BAND * h - yy3) / 1.2 + 0.5, 0, 1)
    edge = band - ndimage.grey_erosion(band, size=(3, 3))
    tex = TEETH[None, None, :] * np.ones((h, w, 1)) * (1 - 0.06 * np.clip((yy3 - BAND * h + 6) / 6, 0, 1))[..., None]
    line_a = np.clip(edge * 1.4, 0, 1) * (yy3 > 3)
    tex = tex * (1 - line_a[..., None]) + LINE_OPEN[None, None, :] * line_a[..., None]
    layerio.save_layer("mouth_teeth", np.dstack([tex, band * 255.0]).astype(np.uint8), SPOT[0] + 520, SPOT[1], crop=False)
    json.dump({"band": BAND, "fangs_u": []}, open(P.p("work", "mouth_teeth.json"), "w"))
    h, w = 24, 200
    yy5, xx5 = np.mgrid[0:h, 0:w].astype(np.float32)
    tex = TEETH[None, None, :] * np.ones((h, w, 1)) * (1 - 0.05 * np.clip((h - 1 - yy5) / 10.0, 0, 1))[..., None]
    la = np.clip(1.0 - (yy5 - 0.5) / 2.0, 0, 1)
    tex = tex * (1 - la[..., None]) + LINE_OPEN[None, None, :] * la[..., None]
    layerio.save_layer("mouth_teethlow", np.dstack([tex, np.full((h, w), 255.0)]).astype(np.uint8), SPOT[0] + 800, SPOT[1], crop=False)
    h, w = 64, 64
    V_, U_ = np.mgrid[0:h, 0:w].astype(np.float32)
    U_ = 2 * (U_ + 0.5) / w - 1; V_ = 2 * (V_ + 0.5) / h - 1
    rr = np.sqrt(U_ ** 2 + V_ ** 2 * (1 - U_ ** 2))
    RHO = 0.56
    band = np.clip((rr - RHO) / 0.08 + 0.5, 0, 1)
    LIP = np.array([234.0, 160.0, 160.0])
    tex = LIP[None, None, :] * (1.0 - 0.10 * np.clip(rr - RHO, 0, 1)[..., None]) * np.ones((h, w, 1))
    layerio.save_layer("mouth_lipring", np.dstack([tex, band * 255.0]).astype(np.uint8), SPOT[0] + 1020, SPOT[1], crop=False)
    print("口のテクスチャを作りました（線 %dx%d、口の中、舌、上の歯、下の歯、唇の輪）" % (stroke.shape[1], stroke.shape[0]))


if __name__ == "__main__":
    main()
