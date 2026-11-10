"""The ears as their own layers, ear_l and ear_r, drawn by the plates nosideL / nosideR (the side locks taken away: the
whole ear, the side of the head) - the first character's rule "ears from close-ups with only the covering locks removed, on the face's
depth". chara4's ears lie under the side locks; nothing of them was drawn anywhere, and what a swinging lock or a turn
bared there was a transparent band (2026-10-22). Here, after segment:
  - the ear in the plate: its skin beside the head's skin (work/head_skin.png - the bald plate's face, which stops at the
    ear's front outline) at the ear's height, with its own outline (dark pixels joined to it) and its inner lines;
  - brought to the picture's tone (the skin both show near the ear); where the picture shows the ear at rest, its own
    pixels (rule 1);
  - only where something lies in front of it at rest or the picture shows it - never over the page beside a lock;
  - z just under the face: its root goes under the face's edge, and the side locks and the earring lie over it.
  -> layers ear_l / ear_r, work/layer_order.json, work/ear_geom.json (boxes)
usage: ear_split.py <project>      (optional - not in run.py's stages: segment draws the side of the head, the ears
       with it, into the face layer from the noside plates. Use this when an ear must move on its own - elf ears that
       lift and droop - and take the ear out of the face layer then)"""
import os, sys, json
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import layerio
from segment import plate_full, tone_match, plate_page

Image.MAX_IMAGE_PIXELS = None
Z_BELOW_FACE = 0.4
W3 = np.array([0.299, 0.587, 0.114], np.float32)


def skinlike(img, cen, tol):
    d = np.abs(img.astype(np.float32) - cen[None, None, :]).max(2)
    f = img.astype(np.float32)
    return (d < tol) & ((f[..., 0] - f[..., 2]) > 8)


def main():
    P = C.open_project()
    k = P.kpx
    HX, HY, HR = P.head()
    fig = C.load_rgba(P.p("work", "figure.png"))
    H, W = fig.shape[:2]
    A = fig[..., :3]
    alpha = fig[..., 3]
    F = alpha > 0
    lab = np.asarray(Image.open(P.p("work", "labels.png")))
    meta = json.load(open(P.p("work", "labels.json")))
    ids = meta["id"]
    order = json.load(open(P.p("work", "layer_order.json")))
    z_face = float(order["z"].get("face", 30))
    head_skin = np.asarray(Image.open(P.p("work", "head_skin.png"))) > 127
    eye_y = 0.5 * (P.pt("eye_l")[1] + P.pt("eye_r")[1])
    mouth_y = P.pt("mouth")[1]
    nose_y = P.pt("nose")[1] if P.pt("nose") is not None else 0.5 * (eye_y + mouth_y)
    # (where the ears are: a point given for each, else the other one mirrored about the head's middle)
    er_p, el_p = P.pt("ear_r"), P.pt("ear_l")
    ear_pt = {"r": er_p if er_p is not None else (None if el_p is None else np.array([2 * HX - el_p[0], el_p[1]])),
              "l": el_p if el_p is not None else (None if er_p is None else np.array([2 * HX - er_p[0], er_p[1]]))}
    # (the skin colour: the face's seen skin, its lighter half)
    fs = (lab == ids["face"]) & (alpha > 230)
    px = A[fs].astype(np.float32)
    lum_px = px @ W3
    skin_c = np.median(px[lum_px > np.percentile(lum_px, 40)], 0)
    skin_l = float(skin_c @ W3)
    # (what lies in front of an ear at rest: every layer above it but the page and the layers behind the head)
    behind = {"page", "hair_back", "hat", "hair_base", "body", "neck", "coat_l", "coat_r", "coat_tassel", "arm_l", "arm_r"}
    front_rest = np.zeros((H, W), bool)
    for n_, v_ in ids.items():
        if n_ not in behind:
            front_rest |= lab == v_
    yy, xx = np.mgrid[0:H, 0:W]
    rows_hs = np.nonzero(head_skin.any(1))[0]
    geom = {}
    for side, pn_, sgn in (("l", "nosideL", -1), ("r", "nosideR", 1)):
        pl, m = plate_full(P, pn_, (H, W))
        name = f"ear_{side}"
        if pl is None:
            print(f"{name}: 下絵 {pn_} がありません")
            continue
        lum_p = pl.astype(np.float32) @ W3
        # the head's side at each row: the farthest column of its skin on this side
        edge = np.full(H, np.nan)
        for y in rows_hs:
            xs_ = np.nonzero(head_skin[y])[0]
            edge[y] = xs_.max() if sgn > 0 else xs_.min()
        band = (yy > eye_y - 0.45 * HR) & (yy < nose_y + 0.2 * HR)
        e_ = np.nan_to_num(edge, nan=float(HX))[:, None]
        beside = band & (sgn * (xx - e_) > -0.08 * HR) & (sgn * (xx - e_) < 0.75 * HR)
        sk = skinlike(pl, skin_c, 70) & (lum_p > skin_l - 75) & m & ~plate_page(P, pl, m, k)
        cand = sk & beside & ~ndimage.binary_dilation(head_skin, iterations=max(2, int(1.5 * k)))
        cand = ndimage.binary_opening(cand, iterations=max(1, int(k)))
        lb, n = ndimage.label(cand)
        if not n:
            print(f"{name}: 下絵 {pn_} に耳が見つかりません")
            continue
        # (an ear reaches up to about the eyes' height; skin that starts lower is the neck under the jaw. Of the pieces
        # that do, the one nearest the ear's point. None: the plate's ear is joined to the head's skin there - the face
        # layer carries it, drawn from the same plate)
        best, best_d = None, None
        for j in range(1, n + 1):
            pc = lb == j
            ys_, xs_ = np.nonzero(pc)
            if len(ys_) < (0.08 * HR) ** 2 or ys_.min() > eye_y + 0.1 * HR:
                continue
            ep = ear_pt[side]
            d_ = 0.0 if ep is None else float(np.hypot(xs_.mean() - ep[0], ys_.mean() - ep[1]))
            if best is None or d_ < best_d:
                best, best_d = j, d_
        if best is None:
            print(f"{name}: 下絵の耳が頭の肌とつながっているので、顔のレイヤーに入れたままにします", flush=True)
            continue
        ear = lb == best
        # (its inner lines and shading: holes of it; its outline: dark pixels of the plate joined to it, a stroke wide -
        # on the side of the head's skin only up to the face's own outline, which stays the face's)
        ear = ndimage.binary_fill_holes(ndimage.binary_closing(ear, iterations=int(2 * k)))
        fp_ = pl.astype(np.float32)
        # (outline colours: dark, not strongly coloured - the red collar beside the jaw is dark as well)
        dark = (lum_p < skin_l - 40) & ((fp_.max(2) - fp_.min(2)) < 130)
        out_line = ndimage.binary_dilation(ear, iterations=int(3 * k)) & dark & m & ~head_skin
        ear = ndimage.binary_fill_holes(ear | out_line)
        ear &= m & ~head_skin
        # (rule 1 and nothing new at rest: only where the picture shows it or something in front of it covers it)
        seen_ear = ear & F & (lab == ids["face"])
        ok = ear & F & (front_rest | seen_ear) & ~(lab == ids.get("page", -1))
        ys, xs = np.nonzero(ok)
        if len(ys) < 50:
            print(f"{name}: 耳が小さすぎます（{len(ys)} px）"); continue
        # its colours: the plate's, brought to the picture's tone on the skin both show near the ear
        near = ndimage.binary_dilation(ear, iterations=int(0.3 * HR)) & ~ear
        ref = near & fs & m & (np.abs(A.astype(np.int16) - pl.astype(np.int16)).max(2) < 40)
        if ref.sum() < 200:
            ref = fs & m & (np.abs(A.astype(np.int16) - pl.astype(np.int16)).max(2) < 40)
        adj = tone_match(pl, A, ref, ok)
        L = np.zeros((H, W, 4), np.uint8)
        L[ok, :3] = adj[ok]
        L[ok, 3] = 255
        # (the picture's own pixels where it shows the ear)
        L[seen_ear & ok, :3] = A[seen_ear & ok]
        L[seen_ear & ok, 3] = alpha[seen_ear & ok]
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        layerio.save_layer(name, np.ascontiguousarray(L[y0:y1, x0:x1]), int(x0), int(y0))
        order["z"][name] = z_face - Z_BELOW_FACE
        if name not in order["order"]:
            order["order"].append(name)
        geom[name] = {"box": [int(x0), int(y0), int(x1), int(y1)], "px": int(ok.sum()), "seen": int((seen_ear & ok).sum())}
        print(f"{name}: 下絵 {pn_} から {int(ok.sum())} px（そのうち絵で見えている所 {int((seen_ear & ok).sum())} px）、範囲 {geom[name]['box']}", flush=True)
    order["order"].sort(key=lambda n: order["z"].get(n, 30))
    json.dump(order, open(P.p("work", "layer_order.json"), "w"), indent=1)
    json.dump(geom, open(P.p("work", "ear_geom.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
