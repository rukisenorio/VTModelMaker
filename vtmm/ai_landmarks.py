"""AI（Claude Code、なければ Codex）が置く位置合わせの点。人が画面で（または Claude がセッションで）確かめる下書き。
二回見る: 絵全体と頭（元の絵の画素で目盛りを振った格子つき）、そのあと顔だけを拡大して。

使い方: ai_landmarks.py <プロジェクト> [--force] [--no-refine]
  --force       もう置いてある点も置き直す（ふだんは空の点だけを埋める）
  --no-refine   二回目の、顔を拡大して見るのを省く
→ character.json の "points"（＋絵にないものは "points_none"、AI の返事そのものは "points_ai"）、
  work/ai_points_check.png（点を絵の上に描いたもの）
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
import common as C
import platform_env
import ai

Image.MAX_IMAGE_PIXELS = None
FACE = ("crown", "brow_l", "brow_r", "eye_l", "eye_r", "nose", "mouth", "chin", "face_l", "face_r", "ear_l", "ear_r")


def grid_picture(img, box, max_side, step_px=55):
    """The box (source px) of img scaled to max_side, with a grid every `step` SOURCE px (about step_px on screen) and
    numbers in source px on every other line. -> PIL image, scale"""
    x0, y0, x1, y1 = box
    crop = Image.fromarray(img[y0:y1, x0:x1]).convert("RGB")
    s = min(max_side / float(max(crop.size)), 4.0)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    raw = step_px / s
    step = next(v for v in (2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000) if v >= raw)
    dr = ImageDraw.Draw(crop, "RGBA")
    f = platform_env.font(13, jp=False)
    for k, xs in enumerate(range((x0 // step + 1) * step, x1, step)):
        X = (xs - x0) * s
        strong = (xs // step) % 2 == 0
        dr.line([(X, 0), (X, crop.height)], fill=(255, 0, 80, 140 if strong else 60), width=1)
        if strong:
            dr.text((X + 2, 2), str(xs), fill=(200, 0, 60, 255), font=f)
            dr.text((X + 2, crop.height - 16), str(xs), fill=(200, 0, 60, 255), font=f)
    for ys in range((y0 // step + 1) * step, y1, step):
        Y = (ys - y0) * s
        strong = (ys // step) % 2 == 0
        dr.line([(0, Y), (crop.width, Y)], fill=(0, 120, 255, 140 if strong else 60), width=1)
        if strong:
            dr.text((2, Y + 2), str(ys), fill=(0, 80, 220, 255), font=f)
            dr.text((crop.width - 44, Y + 2), str(ys), fill=(0, 80, 220, 255), font=f)
    return crop, s


def describe_points(keys):
    return "\n".join(f'  "{k}": {C.POINTS[k]}' for k in keys)


def ask_points(P, keys, pictures, note):
    prompt = f"""You are placing landmarks on a character illustration for rigging a VTube Studio (Live2D) model.
{note}
Coordinates are SOURCE pixels of the original picture: read them off the grid numbers printed along the edges of each
picture (red numbers = x, blue numbers = y). Be precise - interpolate between grid lines.
"L" means the VIEWER'S left (the left side of the image), "R" the viewer's right.
Give each point as [x, y], or null if the picture does not show it (a hidden ear, no legs in a waist-up picture...).

The points:
{describe_points(keys)}

Reply with ONLY a JSON object with exactly these keys: {", ".join(keys)}"""
    text, who = ai.ask(P.d, prompt, pictures, tag="landmarks")
    ans = ai.json_in(text)
    return {k: ans.get(k) for k in keys}, who


def clean(pts, W, H):
    out = {}
    for k, v in pts.items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(t, (int, float)) for t in v):
            x, y = float(v[0]), float(v[1])
            out[k] = [round(x, 1), round(y, 1)] if 0 <= x <= W and 0 <= y <= H else None
        else:
            out[k] = None
    # (left / right swapped by the viewer-character confusion: put them the viewer's way round)
    for a, b in (("eye_l", "eye_r"), ("brow_l", "brow_r"), ("face_l", "face_r"), ("ear_l", "ear_r"), ("shoulder_l", "shoulder_r"),
                 ("elbow_l", "elbow_r"), ("wrist_l", "wrist_r"), ("knee_l", "knee_r"), ("foot_l", "foot_r")):
        if out.get(a) and out.get(b) and out[a][0] > out[b][0]:
            out[a], out[b] = out[b], out[a]
    return out


PT_JA = {"hair_top": "髪の上端", "hair_bottom": "髪の下端", "crown": "頭頂", "brow_l": "眉（左）", "brow_r": "眉（右）",
         "eye_l": "目（左）", "eye_r": "目（右）", "nose": "鼻", "mouth": "口", "chin": "あご", "face_l": "輪郭（左）",
         "face_r": "輪郭（右）", "ear_l": "耳（左）", "ear_r": "耳（右）", "neck": "首", "shoulder_l": "肩（左）",
         "shoulder_r": "肩（右）", "elbow_l": "ひじ（左）", "elbow_r": "ひじ（右）", "wrist_l": "手首（左）", "wrist_r": "手首（右）",
         "hip": "腰", "knee_l": "ひざ（左）", "knee_r": "ひざ（右）", "foot_l": "足（左）", "foot_r": "足（右）"}   # (左右は向かって)


def checks(p):
    warn = []
    def y(k): return p[k][1] if p.get(k) else None
    def x(k): return p[k][0] if p.get(k) else None
    if p.get("eye_l") and p.get("mouth") and not y("eye_l") < y("mouth"):
        warn.append("口が目より下にありません")
    if p.get("mouth") and p.get("chin") and not y("mouth") < y("chin"):
        warn.append("あごが口より下にありません")
    if p.get("crown") and p.get("eye_l") and not y("crown") < y("eye_l"):
        warn.append("頭頂が目より上にありません")
    if p.get("face_l") and p.get("eye_l") and p.get("eye_r") and p.get("face_r") and not (x("face_l") < x("eye_l") < x("eye_r") < x("face_r")):
        warn.append("輪郭と目が、左から 輪郭・目・目・輪郭 の順に並んでいません")
    return warn


def main():
    P = C.open_project()
    art = np.asarray(Image.open(P.p("src", "art.png")).convert("RGBA"))
    H, W = art.shape[:2]
    rgb = art[..., :3].copy()
    if (art[..., 3] < 250).any():
        a = art[..., 3:4].astype(np.float32) / 255
        rgb = (rgb * a + 160 * (1 - a)).astype(np.uint8)
    hb = P.cfg.get("head_guess") or [0, 0, W, int(H * 0.4)]
    full, _ = grid_picture(rgb, (0, 0, W, H), 1500, 60)
    full.save(P.p("work", "ai_guide_full.png"))
    head, _ = grid_picture(rgb, tuple(int(v) for v in hb), 1400, 55)
    head.save(P.p("work", "ai_guide_head.png"))
    keys = list(C.POINTS)
    pts, who = ask_points(P, keys, ["work/ai_guide_full.png", "work/ai_guide_head.png"],
                          f"The first picture is the whole illustration; the second is the head region (x {hb[0]}-{hb[2]}, y {hb[1]}-{hb[3]}) "
                          "zoomed in - read the face's points there.")
    pts = clean(pts, W, H)
    print(f"全体を見ました（{who}）: {sum(v is not None for v in pts.values())} 点", flush=True)
    raw = {"first": pts, "by": who}
    if "--no-refine" not in sys.argv and pts.get("eye_l") and pts.get("eye_r") and pts.get("chin"):
        # a closer look at the face: crown to below the chin, a face width either side
        ex = [pts[k][0] for k in ("eye_l", "eye_r", "face_l", "face_r") if pts.get(k)]
        top = (pts.get("crown") or [0, pts["eye_l"][1] - 1.2 * (pts["chin"][1] - pts["eye_l"][1])])[1]
        fw = max(ex) - min(ex)
        fb = [int(max(0, min(ex) - 0.55 * fw)), int(max(0, top - 0.25 * fw)), int(min(W, max(ex) + 0.55 * fw)), int(min(H, pts["chin"][1] + 0.35 * fw))]
        face, _ = grid_picture(rgb, tuple(fb), 1400, 50)
        face.save(P.p("work", "ai_guide_face.png"))
        fk = [k for k in FACE]
        p2, who2 = ask_points(P, fk, ["work/ai_guide_face.png"],
                              f"The picture is the character's face, zoomed in (x {fb[0]}-{fb[2]}, y {fb[1]}-{fb[3]}).")
        p2 = clean(p2, W, H)
        raw["face"] = p2
        for k in fk:
            if p2.get(k) is not None or pts.get(k) is None:
                pts[k] = p2.get(k)
        print(f"顔を拡大して置き直しました（{who2}）", flush=True)
    pts = clean(pts, W, H)
    cur = P.cfg.setdefault("points", {})
    force = "--force" in sys.argv
    none = set(P.cfg.get("points_none", []))
    for k in keys:
        if force or cur.get(k) is None:
            cur[k] = pts.get(k)
            if pts.get(k) is None:
                none.add(k)
            else:
                none.discard(k)
    P.cfg["points_none"] = sorted(none)
    P.cfg["points_ai"] = raw
    P.save_cfg()
    # the check picture: the points on the picture
    im = Image.fromarray(rgb).convert("RGB")
    s = min(1.0, 1600.0 / max(W, H))
    im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    f = platform_env.font(13)
    for k, v in cur.items():
        if v:
            X, Y = v[0] * s, v[1] * s
            dr.ellipse([X - 4, Y - 4, X + 4, Y + 4], outline=(255, 40, 90), width=2)
            dr.text((X + 6, Y - 8), PT_JA.get(k, k), fill=(255, 40, 90), font=f)
    im.save(P.p("work", "ai_points_check.png"))
    w = checks(cur)
    print(f"置いた点: {sum(v is not None for v in cur.values())} 点 ／ 絵にないもの: " + ("、".join(PT_JA.get(k, k) for k in sorted(none)) or "なし"))
    print("確かめてください: work/ai_points_check.png（画面の「位置合わせ」でも直せます）" + ("\n注意: " + "／".join(w) if w else ""))


if __name__ == "__main__":
    main()
