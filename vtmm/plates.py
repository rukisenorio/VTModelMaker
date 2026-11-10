"""Stage 2: the "X removed" plates - what the picture hides, drawn by image generation, fitted back onto the canvas.

usage: plates.py <project> [make|fit|all] [name ...]      (default: all plates that are not made yet, then fit)

Each plate is a crop of the canvas (the figure on white) brought to 1024 px, an edit by Codex CLI ("remove X, keep
everything else exactly"), registered back onto its input (SIFT + affine + dense flow, align.register: Codex comes back
1-2 px off), super-resolved to the box's canvas size: work/plates/<name>.png (RGB, box size), <name>_changed.png (where
it differs from the canvas), index.json (box, inliers, residual). The layers take from a plate ONLY what the picture
does not show - a plate redraws its neighbours too (method rule 2).

The boxes come from the landmarks; the prompts from character.json "describe" (character / hair / eyes / skin /
outfit). Any plate's prompt or box can be overridden in character.json "plates": {"bald": {"prompt": ..., "box": [x0,
y0, x1, y1] canvas px, "agy": true, "on": "<plate>"}} ("on": made from the picture with that plate in place, e.g. a mask
of the backhair plate). A plate made by Antigravity stays on the input's pixel grid (better for anything whose edges are
used), but its quota is small.
Two Codex calls run at a time (about a minute each).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import common as C
import imagegen

KEEP = ("KEEP everything else exactly as it is, pixel for pixel: the same drawing, the same line weight, the same colouring, "
        "the same position and size of every part. The white background stays plain white.")


def plate_table(P):
    d = dict(P.cfg.get("describe", {}))
    hair = d.get("hair", "the hair")
    eyes = d.get("eyes", "the eyes")
    skin = d.get("skin", "the same skin colour as the face")
    outfit = d.get("outfit", "the clothing")
    who = d.get("character", "the character")
    W, H = P.canvas_size()
    HX, HY, HR = P.head()
    el, er, mouth, chin, neck = P.pt("eye_l"), P.pt("eye_r"), P.pt("mouth"), P.pt("chin"), P.pt("neck")
    nose = P.pt("nose")
    fig = np.asarray(Image.open(P.p("work", "figure.png")))[..., 3] > 0
    # the hair's extent: the figure above the shoulders, and below them as far as the cfg's hair_bottom (or the hips)
    top = P.pt("hair_top")[1]
    hb = P.pt("hair_bottom")
    hair_bottom = hb[1] if hb is not None else min(H, chin[1] + 4.2 * HR)
    rows = fig[int(top):int(min(H, hair_bottom))]
    cols = np.nonzero(rows.any(0))[0]
    # (the hair band: the figure's columns near the head - not the arms - judged at the shoulders' height)
    sh_y = int(0.5 * (P.pt("shoulder_l")[1] + P.pt("shoulder_r")[1])) if P.pt("shoulder_l") is not None else int(chin[1] + HR)
    hx0 = max(0, int(HX - 3.4 * HR)); hx1 = min(W, int(HX + 3.4 * HR))
    # (down to the hair's tips - and never less than a head radius below the chin: for short hair the tips end above the
    # chin, and the bald plate then had no chin, jaw or neck - the face's hidden part ended in a straight cut at a turn)
    head_box = [hx0, max(0, int(top - 0.15 * HR)), hx1, int(min(H, max(hair_bottom + 0.2 * HR, chin[1] + 1.0 * HR)))]
    eye_c = 0.5 * (el + er)
    ed = float(np.linalg.norm(er - el))
    face_c = np.array([HX, 0.5 * (eye_c[1] + mouth[1]) - 0.05 * HR])
    face_half = 1.35 * HR
    eye_half = max(1.05 * ed, 0.9 * HR)
    face_box = [int(face_c[0] - face_half), int(face_c[1] - face_half), int(face_c[0] + face_half), int(face_c[1] + face_half)]
    eyes_box = [int(eye_c[0] - eye_half), int(eye_c[1] - eye_half), int(eye_c[0] + eye_half), int(eye_c[1] + eye_half)]
    nk = neck if neck is not None else np.array([HX, chin[1] + 0.3 * HR])
    hl_half = 2.4 * HR
    headless_box = [int(nk[0] - hl_half), int(nk[1] - 2.9 * HR), int(nk[0] + hl_half), int(nk[1] + 1.9 * HR)]
    t = {}
    t["bald"] = (head_box, f"Remove ALL of the hair of {who}: {hair}, with every strand, lock and braid and any hair ornament fixed "
                 f"in the hair. The head is now bald: show the smooth, round bald scalp and the forehead in {skin}, both ears complete, "
                 f"the neck, and everything the hair covered - the shoulders, the back of the clothing and the arms - continued "
                 f"naturally; where hair stood out against the background, plain white. The face itself - the eyes, the eyebrows, the "
                 f"nose, the mouth - stays exactly as it is. " + KEEP)
    # (all the hair painted flat green: the only sure way to tell hair from skin where they are the same colours - silver
    # or white hair over pale skin; segment uses it to add the strands' insides to the hair)
    t["hairmask"] = (head_box, f"Fill ALL of the hair of {who} - {hair} - with one flat, pure green colour (#00FF00): every "
                     f"strand and lock, the bangs over the forehead and the eyes, the side locks, the hair at the back, keeping "
                     f"the hair's exact outline and shape, with no lines or shading inside the green. Do NOT paint anything that "
                     f"is not hair: a hat, hair ornaments, the face and its skin, the eyes, the eyebrows, the ears, earrings and "
                     f"the clothing stay exactly as they are, with their own colours. " + KEEP)
    # (the front hair's locks and the rest of the hair as a colour map: segment splits front and back by it, strand by
    # strand along the drawn lines, and the locks stage cuts the front hair into locks that swing on their own)
    t["lockmap"] = (head_box, f"This is a production colour map for Live2D rigging. Paint the hair of {who} - {hair} - as flat "
                    f"colour regions, keeping every outline exactly where it is. The BANGS and every lock of hair that hangs in "
                    f"front of the forehead, the eyes, the cheeks or in front of the ears: paint each separate lock with its own "
                    f"flat, saturated colour, neighbouring locks in clearly different colours (use pure red, green, blue, yellow, "
                    f"magenta, cyan, orange and purple; a lock that lies over another one gets its own colour). All the rest of "
                    f"the hair - the top of the head, the hair at the sides behind the face and the ears, and the hair at the "
                    f"back - paint flat mid grey (#808080). Paint inside the hair's outlines only, with no lines or shading inside "
                    f"the colours. Do not paint anything that is not hair: hats and hair ornaments, the face and its skin, the "
                    f"eyes, the eyebrows, the ears, earrings and the clothing stay exactly as they are.")
    # (the side locks: the lockmap paints the bangs; the locks that hang beside the cheeks and over the ears come from a
    # map of their own, and what lies behind them - the ears, the temples, the back hair behind the ears - from a close-up
    # of each side with only those locks taken away. Left grey in the lockmap, they were back hair BEHIND the face: the
    # cheek slid over them at a turn, and there was no ear at all (chara4, 2026-10-22))
    t["sidemap"] = (head_box, f"This is a production colour map for Live2D rigging. Paint ONLY the SIDE LOCKS of the hair of {who} - "
                    f"{hair} - as flat colour regions, keeping every outline exactly where it is: the locks that hang down at both "
                    f"sides of the face - beside the cheeks, the jaw and the neck, over and in front of the ears - and the locks that "
                    f"flare out sideways at the sides. Paint each separate side lock with its own flat, saturated colour, neighbouring "
                    f"locks in clearly different colours (use pure red, green, blue, yellow, magenta, cyan, orange and purple). Paint "
                    f"the BANGS over the forehead and the eyes, the top of the head and all the rest of the hair flat mid grey "
                    f"(#808080). Paint inside the hair's outlines only, with no lines or shading inside the colours. Do not paint "
                    f"anything that is not hair: hats and hair ornaments, the face and its skin, the eyes, the eyebrows, the ears, "
                    f"earrings and the clothing stay exactly as they are.")
    side_y0, side_y1 = int(eye_c[1] - 1.35 * HR), int(min(H, chin[1] + 0.5 * HR))
    for side_, (sx0, sx1) in (("L", (HX - 2.8 * HR, HX + 0.15 * HR)), ("R", (HX - 0.15 * HR, HX + 2.8 * HR))):
        where_ = "left" if side_ == "L" else "right"
        t["noside" + side_] = ([int(max(0, sx0)), side_y0, int(min(W, sx1)), side_y1],
                               f"This is a close-up of the side of the head of {who} - the side on the viewer's {where_}. Remove the "
                               f"SIDE LOCKS on this side: the locks of hair that hang down beside the cheek and the jaw and in front of "
                               f"and over the ear, and the locks that flare out sideways there. Show what is behind them: the whole "
                               f"ear, complete, in {skin} and drawn in the same thin line style as the face, the side of the head and "
                               f"the temple in {skin}, the outline of the cheek and the jaw, the neck, and - behind the ear and above "
                               f"it - the hair at the back of the head ({hair}), a little darker because it is behind; beyond the head "
                               f"the plain white background or the clothing as it continues. The bangs over the forehead and the eyes, "
                               f"the eyes, hats, hair ornaments, earrings and the clothing stay exactly as they are. " + KEEP)
        # (and which of it is hair: a mask made ON it - the silver hair behind the ear is the skin's colour)
        t["ns_hairmask" + side_] = (t["noside" + side_][0],
                                    f"Fill ALL of the HAIR of {who} with one flat, pure green colour (#00FF00): every strand and lock - "
                                    f"the bangs, the hair at the side of the head and behind the ear - keeping the hair's exact "
                                    f"outline, and drawing no lines or shading inside the green. Do NOT paint anything that is not "
                                    f"hair: the skin of the face, the temple and the neck, the ear, the eyes, the eyebrows, earrings, "
                                    f"hats and hair ornaments and the clothing all stay exactly as they are, with their own colours.")
    # (which of the backhair plate is hair: a mask made ON it - pale skin and pinkish silver hair are one colour, and the
    # hairline under the bangs is no closed line: found by colour, the crown became face)
    t["bh_hairmask"] = (head_box, f"Fill ALL of the HAIR of {who} with one flat, pure green colour (#00FF00): every strand and lock - the "
                        f"hair over the top of the head, at the sides and behind the ears - keeping the hair's exact outline and its "
                        f"hairline over the forehead exactly where they are, and drawing no lines or shading inside the green. Do NOT "
                        f"paint anything that is not hair: the forehead and the skin of the face, the temples, the ears, the eyes, the "
                        f"eyebrows, earrings, hats and hair ornaments and the clothing all stay exactly as they are, with their own "
                        f"colours.")
    t["backhair"] = (head_box, f"Remove the FRONT hair only: the bangs that fall over the forehead and between the eyes, and the "
                     f"locks that hang in front of the face, the ears, the shoulders and the chest. What was behind them shows: the "
                     f"forehead and the temples in {skin}, the clothing on the chest and shoulders, and the BACK hair - the hair at "
                     f"the back of the head, seen behind the head and the shoulders as one mass in the same colour, with the same "
                     f"outline the whole hairstyle has now around the top and the sides of the head. The face - the eyes, the "
                     f"eyebrows, the nose, the mouth - and the ears stay exactly as they are. " + KEEP)
    t["face_blank"] = (face_box, f"Remove the facial features: both eyes completely ({eyes}: the whites, the irises, the eyelashes, "
                       f"the eyelid lines), both eyebrows and the mouth. Leave a plain smooth face in {skin} with the same soft "
                       f"shading, as if the face had no eyes, brows or mouth drawn yet. The nose stays. The hair - also the bangs and "
                       f"strands that fall over the face - the ears and the neck stay exactly as they are. " + KEEP)
    t["noiris"] = (eyes_box, "In both eyes remove the iris and the pupil together with their highlights and sparkles, and also the "
                   "strands of hair that lie in front of the eyes, so that only the white of the eye is left inside each eye, complete, "
                   "softly shaded grey under the upper eyelid. The eyelashes, the eyelid lines and the shape of each eye stay exactly as "
                   "they are, and so does everything around the eyes. " + KEEP)
    # (bangs over the eyes are common: what lies under them is drawn, never guessed - the eye layers had holes there and
    # showed them as dark marks when the hair swung)
    t["eyes_clear"] = (eyes_box, "Remove only the hair that lies IN FRONT OF the eyes: the strands and locks of the bangs that cross the "
                       "eyes, the eyelashes and the eyelids. Show both eyes complete underneath, exactly as their visible parts are "
                       "drawn: the whole iris with its highlights, the whole white of the eye, the whole upper and lower eyelash lines "
                       "with their thickness and colour. Where the removed strands lay over the skin around the eyes, show the skin. "
                       "The eyebrows, the rest of the hair and everything else stay exactly as they are. " + KEEP)
    t["eyes_closed"] = (eyes_box, "Close both eyes, peacefully, as in a calm blink: draw each eye shut as ONE gently curved dark eyelash "
                        "line, bowing slightly downward, lying where the lower edge of the open eye was, as wide as the open eye, thickest "
                        "in the middle and tapering to both ends, with a small flick of lashes at its outer end. No white of the eye, no "
                        "iris and no second line above it - a single eyelid line only. The eyebrows, the hair over the face, the nose and "
                        "everything else stay exactly as they are. " + KEEP)
    t["eyes_smile"] = (eyes_box, "Close both eyes in a happy smile: draw each eye as ONE dark eyelash line curved UPWARD like an arch "
                       "('^'), as wide as the open eye, lying where the open eye was, thickest in the middle, tapering to both ends. No "
                       "white, no iris, no second line. The eyebrows, the hair over the face and everything else stay exactly as they "
                       "are. " + KEEP)
    t["headless"] = (headless_box, f"Remove the head completely - the hair, the face, the ears and the neck - so that what was behind "
                     f"the head shows: the collar and the clothing of the shoulders ({outfit}) continued naturally behind where the neck "
                     f"was, and the plain white background above. The shoulders, the clothing and the arms stay exactly as they are. "
                     + KEEP)
    # (what a raised or turned chin bares: the back hair behind the lower face and the neck, then the clothing - drawn;
    # a smooth fill there read as a grey smear beside the jaw. The second character's nape plate)
    nape_box = [int(HX - 1.5 * HR), int(nose[1]) if nose is not None else int(eye_c[1] + 0.4 * HR),
                int(HX + 1.5 * HR), int(min(H, chin[1] + 1.3 * HR))]
    t["nape"] = (nape_box, f"This is a close-up of {who}: the lower half of the face, the neck and the collar. Remove the lower half of "
                 f"the face (from below the nose: the mouth, the chin and the jaw line) and the whole neck, and show what is behind "
                 f"them: the back of the head - {hair} - continued behind where the face and the neck were, with the same colours "
                 f"and strands, a little darker because it is behind; and where the hair ends, the back of the collar and the "
                 f"clothing ({outfit}). The eyes, the ears, the hair in front and everything else stay exactly as they are. " + KEEP)
    if P.pt("shoulder_l") is not None and P.pt("wrist_l") is not None:
        xs = [P.pt(k)[0] for k in ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r") if P.pt(k) is not None]
        ys = [P.pt(k)[1] for k in ("shoulder_l", "shoulder_r", "wrist_l", "wrist_r") if P.pt(k) is not None]
        ax0, ax1 = min(xs) - 1.1 * HR, max(xs) + 1.1 * HR
        ay0, ay1 = min(ys) - 0.9 * HR, max(ys) + 2.4 * HR
        t["noarms"] = ([int(ax0), int(ay0), int(ax1), int(ay1)],
                       f"Remove BOTH arms completely - the sleeves, the hands and the fingers - so that what was behind them shows: the "
                       f"sides of the body and the clothing ({outfit}) continued naturally where the arms were, and the plain white "
                       f"background beyond the body. The hair, the head and the rest of the clothing stay exactly as they are. " + KEEP)
    over = P.cfg.get("plates", {})
    out = {}
    for n, (box, prompt) in t.items():
        o = over.get(n, {})
        if o.get("skip"):
            continue
        out[n] = {"box": o.get("box", box), "prompt": o.get("prompt", prompt), "agy": bool(o.get("agy", False))}
    for n, o in over.items():                       # (extra plates defined in the config only)
        if n not in out and not o.get("skip") and "box" in o and "prompt" in o:
            out[n] = {"box": o["box"], "prompt": o["prompt"] + " " + KEEP, "agy": bool(o.get("agy", False))}
    return out


HAIRLESS = {"noarms"}          # (made from the picture with the bald plate in place: under the arms neither arm nor hair)
ON = {"noarms": "bald", "bh_hairmask": "backhair", "ns_hairmaskL": "nosideL", "ns_hairmaskR": "nosideR"}   # (plates made on another plate; character.json "plates": {name: {"on": ...}})


def on_of(P, name):
    return P.cfg.get("plates", {}).get(name, {}).get("on") or ON.get(name)


def make_input(P, name, box):
    fig = np.asarray(Image.open(P.p("work", "figure.png")))
    a = fig[..., 3:4].astype(np.float32) / 255.0
    onwhite = (fig[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    H, W = onwhite.shape[:2]
    # (a plate made on another plate: its input is the picture with that plate in place - "noarms" on the bald plate, the
    # backhair plate's hair mask on the backhair plate, any plate with "on": <plate> in character.json)
    on = on_of(P, name)
    if on and os.path.exists(P.p("work", "plates", f"{on}.png")):
        idx = json.load(open(P.p("work", "plates", "index.json")))
        bx0, by0, bx1, by1 = idx[on]["box"]
        pl = np.asarray(Image.open(P.p("work", "plates", f"{on}.png")).convert("RGB"))
        e = max(8, (bx1 - bx0) // 60)
        xa, ya, xb, yb = max(bx0 + e, 0), max(by0 + e, 0), min(bx1 - e, W), min(by1 - e, H)
        onwhite = onwhite.copy()
        onwhite[ya:yb, xa:xb] = pl[ya - by0:yb - by0, xa - bx0:xb - bx0]
    x0, y0, x1, y1 = box
    crop = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    crop[ya - y0:yb - y0, xa - x0:xb - x0] = onwhite[ya:yb, xa:xb]
    w, h = x1 - x0, y1 - y0
    size = (1024, 1024) if abs(w - h) < 0.15 * max(w, h) else ((1024, 1536) if h > w else (1536, 1024))
    # (the box is brought to the request's aspect: pad the crop with white on the short side, centred)
    tw, th = size
    s = min(tw / w, th / h)
    small = cv2.resize(crop, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    inp = np.full((th, tw, 3), 255, np.uint8)
    ox, oy = (tw - small.shape[1]) // 2, (th - small.shape[0]) // 2
    inp[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    os.makedirs(P.p("gen", "in"), exist_ok=True)
    pin = P.p("gen", "in", f"{name}_in.png")
    Image.fromarray(inp).save(pin)
    json.dump({"box": [int(v) for v in box], "size": [tw, th], "scale": s, "offset": [ox, oy]}, open(P.p("gen", "in", f"{name}_in.json"), "w"))
    return pin


# what each plate is, for the person making it by hand (manual mode) - and which ones the build cannot do without
PLATE_JA = {
    "bald": "髪と帽子を全部消して、つるつるの頭にした絵", "backhair": "前髪と横髪だけ消して、おでこと後ろ髪を見せた絵",
    "face_blank": "目・眉・口を消した、のっぺらぼうの顔", "noiris": "黒目を消して白目だけにした目",
    "eyes_clear": "目にかかる髪を消して、目を全部見せた絵", "eyes_closed": "目を閉じた絵", "eyes_smile": "笑って目を閉じた絵（＾＾）",
    "headless": "頭と首を消して、襟や肩を見せた絵", "nape": "あごと首を消して、後ろ髪と襟を見せた絵", "noarms": "両腕を消した絵",
    "hairmask": "髪だけ緑一色に塗った絵", "lockmap": "前髪の束を1本ずつ別の色に塗った絵", "sidemap": "横髪の束を1本ずつ別の色に塗った絵",
    "nosideL": "向かって左の横髪を消して、耳を見せた絵", "nosideR": "向かって右の横髪を消して、耳を見せた絵",
    "bh_hairmask": "（前髪を消した絵の）髪だけ緑一色に塗った絵", "ns_hairmaskL": "（左の横髪を消した絵の）髪だけ緑に塗った絵",
    "ns_hairmaskR": "（右の横髪を消した絵の）髪だけ緑に塗った絵",
}
REQUIRED = ("bald", "face_blank", "noiris")


def make(P, names, table):
    import platform_env
    skipped = set(P.cfg.get("plates_skipped", []))
    jobs = []
    for n in names:
        if n in skipped or os.path.exists(P.p("gen", "out", f"{n}.png")) or imagegen.adopt_result(P.d, n):
            continue
        on = on_of(P, n)
        if on and not os.path.exists(P.p("work", "plates", f"{on}.png")):
            # (made on another plate: its input is the picture with that plate in place - not before that plate is in)
            print("  ", n, f"は下絵 {on} ができるのを待っています", flush=True)
            continue
        pin = make_input(P, n, table[n]["box"])
        open(P.p("gen", f"p_{n}.txt"), "w").write(table[n]["prompt"] + "\n")
        jobs.append((n, pin))

    def run(job):
        n, pin = job
        try:
            if table[n]["agy"]:
                imagegen.agy(P.d, n, pin, table[n]["prompt"])
            else:
                imagegen.codex(P.d, "edit", n, table[n]["prompt"], input_png=pin)
            return n, "できました"
        except imagegen.PlateRequested:
            return n, "依頼を書きました（入力の絵と文章は gen/requests/" + n + " にあります）"
        except Exception as e:
            return n, "失敗しました: " + str(e)[:300]
    with ThreadPoolExecutor(max_workers=1 if platform_env.imagegen_backend() == "manual" else 2) as ex:
        for n, r in ex.map(run, jobs):
            print("  ", n, r, flush=True)


def fit(P, name):
    import sr2x
    from align import register
    meta = json.load(open(P.p("gen", "in", f"{name}_in.json")))
    x0, y0, x1, y1 = meta["box"]
    inp = np.asarray(Image.open(P.p("gen", "in", f"{name}_in.png")).convert("RGB"))
    out = np.asarray(Image.open(P.p("gen", "out", f"{name}.png")).convert("RGB"))
    reg, info = register(out, inp)
    ox, oy = meta["offset"]
    s = meta["scale"]
    w, h = x1 - x0, y1 - y0
    sub = reg[oy:oy + int(round(h * s)), ox:ox + int(round(w * s))]
    up = sub
    while up.shape[1] * 2 <= w * 1.05:
        up = sr2x.up(up)
    big = cv2.resize(up, (w, h), interpolation=cv2.INTER_CUBIC if up.shape[1] < w else cv2.INTER_AREA)
    fig = np.asarray(Image.open(P.p("work", "figure.png")))
    Hc, Wc = fig.shape[:2]
    a = fig[..., 3:4].astype(np.float32) / 255.0
    onwhite = (fig[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    ref = np.full((h, w, 3), 255, np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, Wc), min(y1, Hc)
    ref[ya - y0:yb - y0, xa - x0:xb - x0] = onwhite[ya:yb, xa:xb]
    k = max(w, h) / 1024.0
    d = np.abs(cv2.GaussianBlur(big, (0, 0), 1.2 * k).astype(np.int16) - cv2.GaussianBlur(ref, (0, 0), 1.2 * k).astype(np.int16)).max(2)
    ch = ndimage.binary_opening(d > 30, iterations=max(int(round(1.5 * k)), 1))
    os.makedirs(P.p("work", "plates"), exist_ok=True)
    Image.fromarray(big).save(P.p("work", "plates", f"{name}.png"), compress_level=3)
    Image.fromarray((ch * 255).astype(np.uint8)).save(P.p("work", "plates", f"{name}_changed.png"))
    keep = ~ndimage.binary_dilation(ch, iterations=int(6 * k))
    res = float(np.abs(big.astype(np.int16) - ref.astype(np.int16)).max(2)[keep].mean()) if keep.any() else -1
    return {"box": [x0, y0, x1, y1], "inliers": info["inliers"], "flow": round(info.get("flow_px_mean", 0), 2),
            "changed_px": int(ch.sum()), "residual": round(res, 1)}


def main():
    P = C.open_project()
    what = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("make", "fit", "all") else "all"
    names = [a for a in sys.argv[2:] if a not in ("make", "fit", "all")]
    table = plate_table(P)
    with open(P.p("work", "plate_table.json"), "w", encoding="utf-8") as f_:
        json.dump(table, f_, indent=1, ensure_ascii=False)
    names = names or list(table)
    if what in ("make", "all"):
        first = [n for n in names if not on_of(P, n)]
        make(P, first, table)
        later = [n for n in names if on_of(P, n)]
        if later:
            # (the plates they are made on, fitted first - their input is the picture with that plate in place)
            idx_p = P.p("work", "plates", "index.json")
            os.makedirs(P.p("work", "plates"), exist_ok=True)
            idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
            for base_ in sorted({on_of(P, n) for n in later}):
                if base_ in table and os.path.exists(P.p("gen", "out", f"{base_}.png")) and base_ not in idx:
                    idx[base_] = fit(P, base_)
                    json.dump(idx, open(idx_p, "w"), indent=1)
            make(P, later, table)
    if what in ("fit", "all"):
        idx_p = P.p("work", "plates", "index.json")
        os.makedirs(P.p("work", "plates"), exist_ok=True)
        idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
        for n in names:
            src = P.p("gen", "out", f"{n}.png")
            if os.path.exists(src):
                done = P.p("work", "plates", f"{n}.png")
                # (fitted already and the picture not changed since: kept - a re-run in manual mode fits only the new ones)
                if what == "all" and n in idx and os.path.exists(done) and os.path.getmtime(done) >= os.path.getmtime(src):
                    continue
                idx[n] = fit(P, n)
                print(f"下絵 {n} をキャンバスに合わせました:", json.dumps(idx[n]), flush=True)
        json.dump(idx, open(idx_p, "w"), indent=1)
    # what is still missing (manual mode: the person's pictures; the build waits for them - exit 3)
    import platform_env
    skipped = set(P.cfg.get("plates_skipped", []))
    missing = [n for n in names if n not in skipped and not os.path.exists(P.p("gen", "out", f"{n}.png"))]
    status = {"backend": platform_env.imagegen_backend(), "waiting": missing, "required": list(REQUIRED),
              "skipped": sorted(skipped), "done": [n for n in names if os.path.exists(P.p("gen", "out", f"{n}.png"))],
              "blocked": {n: on_of(P, n) for n in missing if on_of(P, n) and not os.path.exists(P.p("work", "plates", f"{on_of(P, n)}.png"))},
              "ja": {n: PLATE_JA.get(n, n) for n in names}}
    os.makedirs(P.p("gen", "requests"), exist_ok=True)
    with open(P.p("gen", "requests", "status.json"), "w", encoding="utf-8") as f_:
        json.dump(status, f_, indent=1, ensure_ascii=False)
    if missing and platform_env.imagegen_backend() == "manual":
        print(f"下絵の絵を {len(missing)} 枚待っています: " + ", ".join(missing))
        print("それぞれの絵を gen/requests/<name>/ に入れてから（なくてもよいものは使わないことにしてもかまいません）、plates の段階をもう一度実行してください")
        sys.exit(3)
    if [n for n in REQUIRED if n in missing]:
        sys.exit("組み立てに必要な下絵がありません: " + ", ".join(n for n in REQUIRED if n in missing))


if __name__ == "__main__":
    main()
