"""The front hair cut into its locks, each a layer that swings on its own (the second character's lock_cut, made general
by the plate lockmap; 2026-10-22).

segment.py splits the hair by the lockmap plate and writes work/lock_ids.png (0 = no lock, 1.. = a front lock, along the
drawn lines). Here each lock becomes a layer lock_<i> cut from layers/hair_front.png (what is left there stays as
hair_front), with
  - its motion field: 0 at its root - the end nearest the crown, where it grows out of the hair under it - rising
    through the lock (geodesic distance) to 1 at its farthest tip; carried a little beyond the lock by nearest value, so
    the mesh cells over its edge move with it;
  - the direction it swings in (across its axis from root to tip) and how far its tip swings (a share of its free
    length).
  -> layers lock_<i>, layers/hair_front without them, work/locks.json (+ work/locks.npz: the fields at RES px),
     work/layer_order.json (the locks just above hair_front), work/locks_look.png
usage: locks.py <project>      (after segment and every stage that edits hair_front; before build)"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

Image.MAX_IMAGE_PIXELS = None
RES = 4
ROOT_BAND = 0.10      # head radii: the lock's root band, from its point nearest the crown
AMP_SHARE = 0.08      # its tip swings this share of its free length ...
AMP_MIN, AMP_MAX = 0.04, 0.16   # ... in head radii, within these
MIN_PX = 0.05         # (head radii squared) a lock smaller than this stays in hair_front


def geodesic(mask, seeds):
    from skimage.graph import MCP_Geometric
    cost = np.where(mask, 1.0, np.inf)
    m = MCP_Geometric(cost, fully_connected=True)
    d, _ = m.find_costs([tuple(s) for s in seeds])
    return d


def main():
    P = C.open_project()
    import layerio
    ip = P.p("work", "lock_ids.png")
    if not os.path.exists(ip):
        print("work/lock_ids.png がありません（下絵 lockmap がない）: 前髪は1枚のレイヤーのままです")
        return
    HX, HY, HR = P.head()
    crown = P.pt("crown")
    crown = (HX, HY - HR) if crown is None else (float(crown[0]), float(crown[1]))
    ids_full = np.asarray(Image.open(ip))
    HF, hx, hy = layerio.load_layer("hair_front")
    h, w = HF.shape[:2]
    ids = np.zeros((h, w), np.uint8)
    H_, W_ = ids_full.shape
    ax0, ay0, ax1, ay1 = max(hx, 0), max(hy, 0), min(hx + w, W_), min(hy + h, H_)
    ids[ay0 - hy:ay1 - hy, ax0 - hx:ax1 - hx] = ids_full[ay0:ay1, ax0:ax1]
    alpha = HF[..., 3] > 0
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = xx + hx, yy + hy
    base = HF.copy()
    info, fields = {}, {}
    order = json.load(open(P.p("work", "layer_order.json")))
    z0 = float(order["z"].get("hair_front", 40))
    # (old lock layers of an earlier run go first)
    for n_ in list(order["order"]):
        if n_.startswith("lock_"):
            order["order"].remove(n_); order["z"].pop(n_, None)
    k_ = 0
    for i in range(1, int(ids.max()) + 1):
        region = (ids == i) & alpha
        if region.sum() < MIN_PX * HR * HR:
            continue
        lb, n = ndimage.label(region, structure=np.ones((3, 3), bool))
        if n > 1:
            sz = np.bincount(lb.ravel())[1:]
            region = lb == int(np.argmax(sz)) + 1
        k_ += 1
        name = f"lock_{k_}"
        L = np.zeros_like(HF)
        L[region] = HF[region]
        base[region] = 0
        # the root: the lock's point nearest the crown and a band from it
        dc = np.hypot(cx - crown[0], cy - crown[1])
        dmin = float(dc[region].min())
        root = region & (dc < dmin + ROOT_BAND * HR)
        d = geodesic(region, np.argwhere(root))
        d[~np.isfinite(d)] = 0.0
        far = float(d[region].max()) if region.any() else 1.0
        f = np.clip(d / max(far, 1.0), 0, 1) ** 1.3
        f = np.where(region, f, 0.0)
        ind = ndimage.distance_transform_edt(~region, return_distances=False, return_indices=True)
        f = f[ind[0], ind[1]]
        fields[name] = cv2.resize(f.astype(np.float32), (int(np.ceil(w / RES)), int(np.ceil(h / RES))), interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(region)
        ti = int(np.argmax(np.where(region, d, -1)[ys, xs]))
        tip = (float(xs[ti] + hx), float(ys[ti] + hy))
        ry, rx = np.nonzero(root)
        rmid = (float(rx.mean() + hx), float(ry.mean() + hy))
        ax_ = np.array(tip) - np.array(rmid); ax_ /= max(np.hypot(*ax_), 1.0)
        perp = np.array([ax_[1], -ax_[0]])
        if perp[0] < 0:
            perp = -perp
        amp = float(np.clip(AMP_SHARE * far, AMP_MIN * HR, AMP_MAX * HR))
        mm = layerio.save_layer(name, L, hx, hy)
        info[name] = {"root": rmid, "tip": tip, "free": far, "perp": perp.tolist(), "amp": amp, "px": int(region.sum()),
                      "box": [mm["x0"], mm["y0"], mm["x0"] + mm["w"], mm["y0"] + mm["h"]]}
        order["z"][name] = z0 + 0.01 * k_
        order["order"].append(name)
        print(f"{name}: {int(region.sum())} px、根元 ({rmid[0]:.0f}, {rmid[1]:.0f})、毛先 ({tip[0]:.0f}, {tip[1]:.0f})、"
              f"自由に動く長さ {far:.0f} px、揺れ幅 {amp:.0f} px", flush=True)
    if not info:
        print("十分に大きい髪の束がありません: 前髪は1枚のレイヤーのままです")
        return
    left = base[..., 3] > 0
    if left.any():
        layerio.save_layer("hair_front", base, hx, hy, crop=False)
    else:
        # (all of the front hair is in the locks: hair_front keeps one transparent pixel, so the build still finds it)
        base[0, 0] = (0, 0, 0, 1)
        layerio.save_layer("hair_front", base, hx, hy, crop=False)
    order["order"].sort(key=lambda n: order["z"].get(n, 30))
    json.dump(order, open(P.p("work", "layer_order.json"), "w"), indent=1)
    json.dump({"x0": hx, "y0": hy, "res": RES, "locks": info}, open(P.p("work", "locks.json"), "w"), indent=1)
    np.savez_compressed(P.p("work", "locks.npz"), **fields)
    # a look: each lock tinted over the rest of the head
    rng = np.random.default_rng(3)
    look = np.zeros((h, w, 3), np.float32) + 40
    bb = base.astype(np.float32); ba = bb[..., 3:4] / 255.0
    look = bb[..., :3] * ba + look * (1 - ba)
    for name in info:
        Lk, lx, ly = layerio.load_layer(name)
        full = np.zeros((h, w, 4), np.float32)
        full[ly - hy:ly - hy + Lk.shape[0], lx - hx:lx - hx + Lk.shape[1]] = Lk
        a_ = full[..., 3:4] / 255.0
        col = rng.integers(60, 255, 3).astype(np.float32)
        look = (full[..., :3] * 0.45 + col * 0.55) * a_ + look * (1 - a_)
    s_ = 900.0 / max(h, w)
    Image.fromarray(cv2.resize(np.clip(look, 0, 255).astype(np.uint8), (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)).save(P.p("work", "locks_look.png"))
    print(f"髪の束 {len(info)} 本。hair_front に残ったのは {int(left.sum())} px。確認用の絵: work/locks_look.png", flush=True)


if __name__ == "__main__":
    main()
