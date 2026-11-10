"""A hair base under all the hair (first made for chara4, 2026-10-22): where the hair is seen at rest, a soft, darker version of
its own colours, on the head's turn without any swing, behind everything of the head (z -2).

When a lock swings away, the place it covered at rest showed the page (in the hair's interior and along its outline: gaps
between locks at a turn with a swing). With this layer it shows hair in shadow, as the thick hair mass behind the locks
would. It is fully covered at rest (the seen hair, shrunk a little, lies over it), so the rest pose is unchanged; the page
between strands stays page (it was no hair at rest). Run after segment and after any per-character stage that changes the hair (a hat split)."""
import os, sys, json
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

Image.MAX_IMAGE_PIXELS = None
Z = -2.0
DARK = 0.78


def main():
    P = C.open_project()
    import layerio                     # (after open_project: layerio finds the project's layers folder from it)
    k = P.kpx
    fig = C.load_rgba(P.p("work", "figure.png"))
    H, W = fig.shape[:2]
    A = fig[..., :3]
    lab = np.asarray(Image.open(P.p("work", "labels.png")))
    meta = json.load(open(P.p("work", "labels.json")))
    ids = meta["id"]
    hair = (lab == ids["hair_front"]) | (lab == ids["hair_back"])
    hair &= fig[..., 3] > 200
    base = ndimage.binary_erosion(ndimage.binary_closing(hair, iterations=int(2 * k)), iterations=max(2, int(1.5 * k)))
    # (not where the side locks hang over the page: the plates nosideL / nosideR show nothing behind them there - a hair
    # base under a lock that flares out over the page showed as a dark ghost of it when it swung)
    from segment import plate_full, plate_page
    # (but next to the face it stays: between the cheek and the locks hanging beside it there is more hair in 3D, and with
    # nothing there a crescent of page opened along the far cheek at a full turn with the hair swung out (chara4,
    # 2026-10-23))
    HX_, HY_, HR_ = P.head()
    hs_p = P.p("work", "head_skin.png")
    if os.path.exists(hs_p):
        head_skin = np.asarray(Image.open(hs_p)) > 127
        near_face = ndimage.distance_transform_edt(~head_skin) < 0.35 * HR_
    else:
        near_face = np.zeros((H, W), bool)
    for pn_ in ("nosideL", "nosideR"):
        pl_s, m_s = plate_full(P, pn_, (H, W))
        if pl_s is not None:
            pg_ = m_s & plate_page(P, pl_s, m_s, k) & ~near_face
            print(f"髪の下地: 下絵 {pn_} で横髪の後ろが背景になっている所には敷きません: {int((base & pg_).sum())} px", flush=True)
            base &= ~pg_
    ys, xs = np.nonzero(base)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sl = (slice(y0, y1), slice(x0, x1))
    # (its colour: the hair's own colours, the dark outlines left out, averaged softly - no line of the hair shows twice -
    # then a step darker: the hair behind the locks is in their shadow)
    lum = A.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    known = hair & (lum > np.percentile(lum[hair], 25))
    # (the hair's own colours only: a ribbon, a clip, an earring or a hat counted as hair would leave a ghost of itself
    # in the base when it moves - the other session's chara5 lesson. The hair's main colour clusters, each at least 8 %)
    from segment import colour_model, colour_dist
    px = A[known]
    cen = colour_model(px, 6)
    lab_c = np.argmin(np.stack([np.abs(px.astype(np.float32) - c[None, :]).max(1) for c in cen], 1), 1)
    share = np.bincount(lab_c, minlength=len(cen)) / max(len(px), 1)
    main = cen[share >= 0.08]
    known &= colour_dist(A, main) <= 46
    sub = layerio.smooth_fill(A[sl], known[sl], np.ones((y1 - y0, x1 - x0), bool)).astype(np.float32)
    sub = cv2.GaussianBlur(sub, (0, 0), max(2.0, 3 * k))
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    bm = base[sl]
    rgba[bm, :3] = np.clip(sub[bm] * DARK, 0, 255).astype(np.uint8)
    rgba[bm, 3] = 255
    layerio.save_layer("hair_base", rgba, x0, y0, pad=0, crop=False)
    # ---- the back hair's hidden part is never skin-coloured: next to the face the plates' fills took the face's shaded
    # edge for hair (pale skin and pinkish silver hair are close), a mauve smear beside the jaw at every turn. Hidden back
    # hair warmer than the seen hair goes to this same soft hair colour in shadow
    Lb, bx0, by0 = layerio.load_layer("hair_back")
    hb = Lb.copy()
    hh, ww = hb.shape[:2]
    seen_hb = lab[by0:by0 + hh, bx0:bx0 + ww] == ids["hair_back"]
    hp = A[hair].astype(np.float32)
    warm_seen = float(np.percentile(hp[:, 0] - hp[:, 2], 75))
    c = hb[..., :3].astype(np.float32)
    warm = c[..., 0] - c[..., 2]
    lum_b = c @ np.array([0.299, 0.587, 0.114], np.float32)
    bad = (hb[..., 3] > 0) & ~seen_hb & (warm > warm_seen + 6) & (lum_b > 100)
    if bad.any():
        full_sub = np.zeros((H, W, 3), np.float32)
        full_sub[sl] = sub
        col = np.clip(full_sub[by0:by0 + hh, bx0:bx0 + ww][bad] * DARK, 0, 255).astype(np.uint8)
        hb[bad, :3] = col
        layerio.save_layer("hair_back", hb, bx0, by0, pad=0, crop=False)
    print(f"見えている髪より赤みの強い隠れた後ろ髪（赤み > {warm_seen + 6:.0f}）を髪の影の色にしました: {int(bad.sum())} px", flush=True)
    order = json.load(open(P.p("work", "layer_order.json")))
    order["z"]["hair_base"] = Z
    if "hair_base" not in order["order"]:
        order["order"].append("hair_base")
    order["order"].sort(key=lambda n: order["z"].get(n, 30))
    json.dump(order, open(P.p("work", "layer_order.json"), "w"), indent=1)
    print(f"髪の下地: {int(base.sum())} px、z {Z}", flush=True)


if __name__ == "__main__":
    main()
