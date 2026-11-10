"""高品質モード：見本の向き絵、それに合わせた首の回り方と見上げ方、動きで見る確認パック、そのAIの確認、比べるために
残す版。絵は Codex CLI が描き、確認は Claude Code CLI がします（Claude がないときは Codex）。

使い方: hq.py <プロジェクト> refs                 同じキャラを左右に30度向いた絵・見上げ・見下ろしで描く（Codex か
                                                手作業の依頼）→ gen/out/ref_*.png
        hq.py <プロジェクト> calibrate [--apply]  rig.yaw_amp（首の回り方）と rig.nod_up_scale（見上げ方）の候補を組み立て、
                                                見本と同じ角度で描いて、体の位置でそろえた見本と目の位置を比べる
                                                → 見本に合う値（--apply: character.json に書いて組み立て直す）
                                                ＋ preview/hq/calibrate.png
        hq.py <プロジェクト> qa [--label L] [--compare 前の版]   確認パック → preview/qa/<L>/（qa.md: 見る所）
        hq.py <プロジェクト> review [--label L]   AIが確認パックを見る → preview/qa/<L>/review.md
        hq.py <プロジェクト> snapshot [名前]      out/<名前> を versions/<名前>/ に残す（あとで比べるため）
"""
import os, sys, json, time, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image, ImageDraw
import common as C
import platform_env

Image.MAX_IMAGE_PIXELS = None
KIT = os.path.dirname(os.path.abspath(__file__))
VIEWS = {   # name: (what to draw, the parameter, its value's sign relative to the view, axis of the eyes' shift)
    "ref_turn_L": "Only the HEAD is turned about 30 degrees toward the LEFT side of the picture (a three-quarter view: the side of the face "
                  "on the right of the picture is nearer to us, the side on the left of the picture turns away).",
    "ref_turn_R": "Only the HEAD is turned about 30 degrees toward the RIGHT side of the picture (a three-quarter view: the side of the face "
                  "on the left of the picture is nearer to us, the side on the right of the picture turns away).",
    "ref_pitch_up": "Only the HEAD is TILTED UP about 30 degrees: the chin raised, looking up. No turn to a side and no lean - the face stays "
                    "symmetric. The face is foreshortened: eyes, nose and mouth higher and closer together, the underside of the jaw "
                    "visible as a softly shaded plane between the chin and the neck. The head is the same size as in the reference.",
    "ref_pitch_down": "Only the HEAD is TILTED DOWN about 30 degrees: the chin lowered, looking down. No turn to a side and no lean - the face "
                      "stays symmetric. More of the top of the head is visible, the bangs cover more of the forehead, the chin is tucked "
                      "in toward the collar. The head is the same size as in the reference.",
}


KEY_JA = {"yaw_amp": "首の回り方", "nod_up_scale": "見上げ方"}
VIEW_JA = {"ref_turn_L": "左向き", "ref_turn_R": "右向き", "ref_pitch_up": "見上げ", "ref_pitch_down": "見下ろし"}
EXP_JA = {"Blush": "頬の赤み", "Face_Angry": "怒り", "Face_Jitome": "ジト目", "Face_Sad": "悲しみ", "Face_Shy": "照れ",
          "Face_Sleepy": "眠い", "Face_Smile": "笑顔", "Face_Surprised": "驚き", "Face_Wink": "ウインク"}


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONUTF8="1", **(env or {})))
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1])} failed:\n{(r.stdout + r.stderr)[-1500:]}")
    return r.stdout


def kit_of(P):
    k = P.p("kit")
    return k if os.path.isdir(k) else KIT


# ------------------------------------------------------------------------------------------------------- refs
def head_crop_box(P):
    HX, HY, HR = P.head()
    W, H = P.canvas_size()
    top = P.pt("hair_top")
    chin = P.pt("chin")
    y0 = (top[1] if top is not None else HY - 1.6 * HR) - 0.15 * HR
    y1 = chin[1] + 2.0 * HR
    side = max(y1 - y0, 5.0 * HR)
    x0 = HX - side / 2
    return [int(max(0, x0)), int(max(0, y0)), int(min(W, x0 + side)), int(min(H, y0 + side))]


def refs(P):
    import imagegen
    d = dict(P.cfg.get("describe", {}))
    who = d.get("character", "the character")
    design = ", ".join(v for v in (d.get("hair"), d.get("eyes"), d.get("outfit")) if v) or "the same design"
    box = head_crop_box(P)
    fig = Image.open(P.p("work", "figure.png")).convert("RGBA")
    bg = Image.new("RGBA", fig.size, (255, 255, 255, 255))
    bg.alpha_composite(fig)
    crop = bg.crop(box).convert("RGB")
    crop = crop.resize((1024, int(1024 * crop.height / crop.width)), Image.LANCZOS)
    os.makedirs(P.p("gen", "in"), exist_ok=True)
    src = P.p("gen", "in", "head_ref.png")
    crop.save(src)
    json.dump({"box": box}, open(P.p("gen", "in", "head_ref.json"), "w", encoding="utf-8"))
    made, waiting = [], []
    for name, what in VIEWS.items():
        instr = (f"The same {who} from the same distance and the same height, head and shoulders, in the same framing as the reference "
                 f"(the collar and the shoulders stay where they are, facing the viewer). {what} Keep the same design exactly: {design}, "
                 "the same thin outlines and flat anime cel colouring. Neutral calm expression, mouth closed. Plain white background. No text.")
        try:
            imagegen.codex(P.d, "ref", name, instr, input_png=src, aspect="square")
            made.append(name)
        except imagegen.PlateRequested:
            waiting.append(name)
        except Exception as e:
            print(f"{VIEW_JA.get(name, name)}: 描けませんでした: {str(e)[:300]}", flush=True)
    print("見本の向き絵: " + ("、".join(VIEW_JA.get(n, n) for n in made) or "なし"))
    if waiting:
        print("手作業モード: 次の絵を待っています: " + "、".join(f"gen/requests/{n}" for n in waiting))
        sys.exit(3)


# ------------------------------------------------------------------------------------------------- calibrate
def render_head(P, out_dir, vals, width=700):
    import preview as PV
    L = PV.Look(P, out_dir)
    box = head_crop_box(P)
    s = width / float(box[2] - box[0])
    img = L.render(vals, s, tuple(box))
    eyes = []
    ids = list(L.m.drawable_ids)
    for side in ("l", "r"):
        cands = [i for i, d in enumerate(ids) if d.startswith(f"eye_{side}_iris") and not d.endswith("_star")] or \
                [i for i, d in enumerate(ids) if d.startswith(f"eye_{side}_white")]
        if cands:
            p = L.m.positions_px(cands[0])
            eyes.append(((p[:, 0].mean() - box[0]) * s, (p[:, 1].mean() - box[1]) * s))
    return C.on_dark(img, (255, 255, 255)), eyes, s


def align_ref(render, ref):
    """The drawing brought onto the render: SIFT matches on the lower half (the body, which does not turn), a similarity
    transform. -> warped drawing, number of inliers"""
    cv2.setRNGSeed(0)                              # (RANSAC: the same answer every time)
    g1 = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(4000)
    m1 = np.zeros_like(g1); m1[int(g1.shape[0] * 0.5):] = 255
    m2 = np.zeros_like(g2); m2[int(g2.shape[0] * 0.5):] = 255
    k1, d1 = sift.detectAndCompute(g1, m1)
    k2, d2 = sift.detectAndCompute(g2, m2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        k1, d1 = sift.detectAndCompute(g1, None)
        k2, d2 = sift.detectAndCompute(g2, None)
    good = []
    for pair in cv2.BFMatcher().knnMatch(d2, d1, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 6:
        return cv2.resize(ref, (render.shape[1], render.shape[0])), 0
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return cv2.resize(ref, (render.shape[1], render.shape[0])), 0
    warped = cv2.warpAffine(ref, M, (render.shape[1], render.shape[0]), borderValue=(255, 255, 255))
    return warped, int(inl.sum())


def find_eyes(render, eyes, ref_aligned, HRs):
    """Where the model's eyes are in the aligned drawing: each eye's patch from the render, searched for nearby (less than
    the eyes' spacing, so one eye is never taken for the other). -> [(x, y, score)] per eye in render px, or None"""
    g1 = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(ref_aligned, cv2.COLOR_RGB2GRAY)
    half = max(8, int(0.2 * HRs))
    gap = abs(eyes[0][0] - eyes[1][0]) if len(eyes) == 2 else 2 * HRs
    reach = int(max(4, min(0.55 * HRs, 0.45 * gap)))
    out = []
    for ex, ey in eyes:
        ex, ey = int(ex), int(ey)
        if ex - half < 0 or ey - half < 0 or ex + half >= g1.shape[1] or ey + half >= g1.shape[0]:
            return None
        tpl = g1[ey - half:ey + half, ex - half:ex + half]
        x0, y0 = max(0, ex - half - reach), max(0, ey - half - reach)
        x1, y1 = min(g2.shape[1], ex + half + reach), min(g2.shape[0], ey + half + reach)
        win = g2[y0:y1, x0:x1]
        if win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]:
            return None
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        out.append((x0 + loc[0] + half, y0 + loc[1] + half, float(score)))
    return out


def tear(P, out_dir, poses):
    """How much a build tears at the given poses: the see-through pixels enclosed by the figure plus the half-transparent
    ones around the head and neck, more than at rest (summed over the poses). A part that moved off what lay behind it,
    a stretched neck with nothing drawn under the chin, hair smeared thin - they all raise it."""
    from scipy import ndimage
    import preview as PV
    L = PV.Look(P, out_dir)
    HX, HY, HR = P.head()
    chin = P.pt("chin")
    box = (int(HX - 2.2 * HR), int(HY - 1.8 * HR), int(HX + 2.2 * HR), int(chin[1] + 1.2 * HR))
    sc = float(np.clip(80.0 / HR, 0.1, 0.6))     # (a head about 160 px wide: thin tears still show)

    def amount(v):
        a_ = L.render(v, sc, box)[..., 3]
        lab, n = ndimage.label(a_ <= 127)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        enc = int(np.isin(lab, [i for i in range(1, n + 1) if i not in border]).sum())
        return enc + int(((a_ > 20) & (a_ < 235)).sum())
    rest = amount({})
    floor = 0.002 * (box[2] - box[0]) * (box[3] - box[1]) * sc * sc       # (0.2 % of the box: noise)
    return sum(amount(v) - rest for v in poses), floor


def calibrate(P, apply=False):
    have = {n: P.p("gen", "out", n + ".png") for n in VIEWS if os.path.exists(P.p("gen", "out", n + ".png"))}
    if not have:
        sys.exit("見本の向き絵がまだありません。先に「見本の向き絵を描く」（hq.py <プロジェクト> refs）をしてください")
    HX, HY, HR = P.head()
    kit = kit_of(P)
    name = P.name
    # (the project's own copy of the engine must know these options - a copy made by an older version does not: bring
    # its unchanged files up to date first)
    bp = os.path.join(kit, "build.py")
    if "nod_up_scale" not in open(bp, encoding="utf-8").read():
        print(sh([sys.executable, "-X", "utf8", os.path.join(KIT, "update_kit.py"), P.d]).strip(), flush=True)
        if "nod_up_scale" not in open(bp, encoding="utf-8").read():
            sys.exit("このキャラ専用の本体の写し（kit/build.py）が古く、しかもこのキャラのために直されています。"
                     f"`update_kit.py {P.d} --all` で入れ替えるか（古い写しは backups/ に残ります）、"
                     "yaw_amp と nod_up_scale の設定を手で移してください")
    # which sign of ParamAngleX turns the face toward the left of the picture
    base_out = P.p("out", name)
    if not os.path.exists(os.path.join(base_out, f"{name}.moc3")):
        sys.exit("先にモデルを組み立ててください（run.py --from segment）")
    _, e0, _ = render_head(P, base_out, {})
    _, e1, _ = render_head(P, base_out, {"ParamAngleX": 30})
    left_sign = -1.0 if np.mean([b[0] - a[0] for a, b in zip(e0, e1)]) > 0 else 1.0
    pose = {"ref_turn_L": {"ParamAngleX": 30 * left_sign}, "ref_turn_R": {"ParamAngleX": -30 * left_sign},
            "ref_pitch_up": {"ParamAngleY": 30}, "ref_pitch_down": {"ParamAngleY": -30}}
    cur = dict(P.cfg.get("rig", {}))
    with open(P.cfg_path, encoding="utf-8") as f_:
        cfg_text = f_.read()                       # (restored after the candidates, whatever happens)
    # the candidates, and how far more may be tried when the drawing asks for more (beyond that the model tears: the
    # drawing is then probably exaggerated, and nothing is applied)
    plans = {"yaw_amp": ([0.82, 1.0, 1.2, 1.45, 1.7], ["ref_turn_L", "ref_turn_R"], 0, (0.5, 2.2)),
             "nod_up_scale": ([1.0, 1.3, 1.6, 1.9], ["ref_pitch_up"], 1, (0.5, 3.0))}
    default = {"yaw_amp": 0.82, "nod_up_scale": 1.0}            # (build.py's own defaults)
    rows, found, unsure, measured, target, tears, notes = [], {}, {}, {}, {}, {}, {}
    for key, (vals, views, axis, (lo, hi)) in plans.items():
        views = [v for v in views if v in have]
        if not views:
            continue
        now = float(cur.get(key, default[key]))
        # (matching the eyes alone is not enough: a look-up drawn with a long neck asks for a lift the model cannot make
        # without tearing under the chin. Every candidate is also measured for tearing at the views' poses, and one that
        # tears clearly more than the least torn version is not chosen)
        tposes = [pose[vw] for vw in views]
        t_now, t_floor = tear(P, base_out, tposes)
        print(f"{KEY_JA[key]}: 候補を組み立てて、見本の絵と目の位置を比べます"
              "（目の差は頭の半径を1とした値。＋はモデルの動きが足りない、－は動きすぎ。"
              f"裂け目は、見本の角度で正面より増える、すき間と半透明の量。今の版で {t_now:+d}）", flush=True)
        tears[key] = {}

        def measure(v):
            label = f"_cal_{key}_{v}"
            # (the candidate's value written into character.json for this build - a project's own copy of the engine may
            # be older than this script and know no other way in - and the file put back right after)
            cand = json.loads(cfg_text)
            cand.setdefault("rig", {})[key] = v
            try:
                with open(P.cfg_path, "w", encoding="utf-8") as f_:
                    json.dump(cand, f_, indent=1, ensure_ascii=False)
                sh([sys.executable, "-X", "utf8", os.path.join(kit, "build.py"), P.d], env={"L2D_OUT": label})
            finally:
                with open(P.cfg_path, "w", encoding="utf-8") as f_:
                    f_.write(cfg_text)
            out_dir = P.p("out", label)
            ds = []
            for vw in views:
                img, eyes, s = render_head(P, out_dir, pose[vw])
                if vw not in target:
                    # the drawing: aligned on the body and its eyes found ONCE (with the first candidate's eyes as the
                    # patterns), so that every candidate is measured against the same points
                    ref = np.asarray(Image.open(have[vw]).convert("RGB"))
                    al, inl = align_ref(img, ref)
                    fe = find_eyes(img, eyes, al, HR * s) if inl >= 12 and len(eyes) == 2 else None
                    if fe is None or min(e[2] for e in fe) < 0.25:
                        print(f"{VIEW_JA[vw]}の見本: モデルとうまく重なりません（体の一致 {inl} 点"
                              + (f"、目の一致 {min(e[2] for e in fe):.2f}" if fe else "")
                              + "）。この見本は使いません。「見本の向き絵を描く」で描き直してください", flush=True)
                        target[vw] = None
                    else:
                        target[vw] = (al, inl, fe)
                if not target[vw] or len(eyes) != 2:
                    continue
                al, inl, fe = target[vw]
                d = float(np.mean([fe[i][axis] - eyes[i][axis] for i in range(2)]))
                # (signed so that "the drawing's eyes are further along the view's direction" is positive: the model
                # turns or lifts too little)
                if vw in ("ref_turn_L", "ref_pitch_up"):
                    d = -d
                ds.append(d / (HR * s))
                rows.append((key, v, vw, img, al, inl, (fe[0][0] + fe[1][0] - eyes[0][0] - eyes[1][0]) / 2,
                             (fe[0][1] + fe[1][1] - eyes[0][1] - eyes[1][1]) / 2, eyes, fe))
            t_v, _ = tear(P, out_dir, tposes)
            tears[key][v] = t_v
            shutil.rmtree(out_dir, ignore_errors=True)
            if not ds:
                return np.nan
            print(f"  {v}: 目の差 {float(np.mean(ds)):+.3f} ／ 裂け目 {t_v:+d}", flush=True)
            return float(np.mean(ds))

        meas = {}
        mid = vals[len(vals) // 2]
        for v in [mid] + [x for x in vals if x != mid]:
            meas[v] = measure(v)
            if not np.isfinite(meas[v]):
                break                              # (the drawings are not usable: no point building the rest)
        # the limit: a quarter more than the least tearing version (the model now, or a candidate - not the model now
        # alone, which may already be torn by an earlier calibration). Strict on purpose: the tool must not make the
        # model worse; what it holds back, the person can still try and judge in motion
        t_limit = 1.25 * max(min([t_now] + list(tears[key].values())), 0) + t_floor
        over = [v for v, t in sorted(tears[key].items()) if t > t_limit]
        print(f"  （裂け目の上限 {int(t_limit):+d}: いちばん少ない版の1.25倍。" + (f"超える候補 {', '.join(str(v) for v in over)} は崩れるので選びません）" if over else "どの候補も超えていません）"),
              flush=True)
        # every candidate on the same side of the drawing: more are tried beyond them, within the safe range
        for _ in range(4):
            ok = sorted((v, d) for v, d in meas.items() if np.isfinite(d))
            if len(ok) < 2:
                break
            vs, dd = np.array(ok).T
            if np.any(dd == 0) or np.any(dd[:-1] * dd[1:] < 0):
                break
            slope = (dd[-1] - dd[0]) / (vs[-1] - vs[0])
            if abs(slope) < 1e-6:
                break
            up = (dd[0] > 0) == (slope < 0)        # the way the difference goes to zero
            nxt = round(float(min(hi, vs[-1] + 1.5 * (vs[-1] - vs[-2]))) if up else float(max(lo, vs[0] - 1.5 * (vs[1] - vs[0]))), 2)
            if nxt in meas:
                break
            edge = vs[-1] if up else vs[0]
            if tears[key].get(float(edge), 0) > t_limit:
                break                              # (the last one already tears: further would tear more)
            print(f"  （どの候補も見本に届かないので、{nxt} も試します）", flush=True)
            meas[nxt] = measure(nxt)
            if tears[key].get(nxt, 0) > t_limit:
                print(f"  （{nxt} は崩れます）", flush=True)
        ok = sorted((v, d) for v, d in meas.items() if np.isfinite(d))
        measured[key] = [[float(v), round(float(d), 4)] for v, d in ok]
        if len(ok) < 2:
            continue
        vs, dd = np.array(ok).T
        holds = {float(v): tears[key].get(float(v), 0) <= t_limit for v in vs}
        best, i_best = None, None
        for i in range(len(vs) - 1):               # where the difference changes sign: between those two candidates
            if dd[i] == 0 or dd[i] * dd[i + 1] < 0:
                best = vs[i] + (vs[i + 1] - vs[i]) * dd[i] / (dd[i] - dd[i + 1]) if dd[i] != dd[i + 1] else vs[i]
                i_best = i
                break
        if best is not None and holds[float(vs[i_best])] and holds[float(vs[i_best + 1])]:
            found[key] = round(float(best), 2)
            notes[key] = f"{KEY_JA[key]}: 見本どおりの {found[key]} にしました（前は {now}）"
            print(f"=> {KEY_JA[key]} = {found[key]}（今は {now}）", flush=True)
            continue
        target_v = best if best is not None else float(vs[int(np.argmin(np.abs(dd)))])
        safe = [float(v) for v in vs if holds[float(v)]]
        near = min(safe, key=lambda v: abs(v - target_v)) if safe else None
        torn = [float(v) for v in vs if not holds[float(v)]]
        if best is not None:
            why = f"見本どおり（{round(float(best), 2)}）にすると崩れます"
        elif torn:
            why = f"見本に近づける候補（{'、'.join(str(v) for v in torn)}）は崩れます"
        else:
            why = f"安全な範囲（{lo}〜{hi}）の中では見本に届きませんでした"
        if near is not None and abs(near - now) > 1e-6 and abs(near - target_v) < abs(now - target_v):
            found[key] = round(near, 2)
            notes[key] = f"{KEY_JA[key]}: {why}。崩れない候補のうち見本にいちばん近い {found[key]} にしました（前は {now}）"
            print(f"=> {KEY_JA[key]}: {why}。崩れない候補のうち見本にいちばん近い {found[key]} にします（今は {now}）", flush=True)
        else:
            unsure[key] = round(float(target_v), 2)
            notes[key] = f"{KEY_JA[key]}: {why}。今の {now} のままです。見本の絵が大げさなのかもしれません"
            print(f"=> {KEY_JA[key]}: {why}。今の {now} のままにします。見本の絵が大げさなのかもしれません。比較図を見て決めてください",
                  flush=True)
    # the sheet: for each tried value, the render beside the aligned drawing (model's eyes red, the drawing's blue)
    if rows:
        F = platform_env.font(18)
        W_ = rows[0][3].shape[1]
        head = Image.new("RGB", (3 * W_, 44), (255, 255, 255))
        dr = ImageDraw.Draw(head)
        for k, t in enumerate(("モデル（赤丸＝目）", "見本の絵（体の位置でそろえた・青丸＝目）", "重ねた図（2枚を半分ずつ）")):
            dr.text((k * W_ + 10, 10), t, font=platform_env.font(24), fill=(40, 40, 40))
        tiles = [np.asarray(head.resize((head.width // 2, head.height // 2), Image.LANCZOS))]
        for key, v, vw, img, al, inl, dx, dy, eyes, fe in sorted(rows, key=lambda r: (r[0], r[2], r[1])):
            blend = (0.5 * img.astype(np.float32) + 0.5 * al.astype(np.float32)).astype(np.uint8)
            im = Image.fromarray(np.concatenate([img, al, blend], 1))
            dr = ImageDraw.Draw(im)
            for (x, y), (fx, fy, _) in zip(eyes, fe):
                for ox, col, (px, py) in ((0, (230, 0, 40), (x, y)), (W_, (0, 90, 255), (fx, fy)),
                                          (2 * W_, (230, 0, 40), (x, y)), (2 * W_, (0, 90, 255), (fx, fy))):
                    dr.ellipse((ox + px - 6, py - 6, ox + px + 6, py + 6), outline=col, width=3)
            tv = tears.get(key, {}).get(v)
            dr.text((8, 6), f"{KEY_JA[key]} {v}・{VIEW_JA[vw]}（体の一致 {inl} 点、目の差 横 {dx:.0f} 縦 {dy:.0f}"
                    + (f"、裂け目 {tv:+d}" if tv is not None else "") + "）", font=F, fill=(200, 0, 60))
            tiles.append(np.asarray(im.resize((im.width // 2, im.height // 2), Image.LANCZOS)))
        os.makedirs(P.p("preview", "hq"), exist_ok=True)
        Image.fromarray(np.concatenate(tiles, 0)).save(P.p("preview", "hq", "calibrate.png"))
        print("比較図: preview/hq/calibrate.png（左＝モデル、中＝体の位置でそろえた見本、右＝重ねた図。赤丸＝モデルの目、青丸＝見本の目）")
    os.makedirs(P.p("preview", "hq"), exist_ok=True)
    json.dump({"found": found, "unsure": unsure, "notes": notes, "measured": measured, "tears": {k: {str(v): t for v, t in d.items()} for k, d in tears.items()},
               "now": cur},
              open(P.p("preview", "hq", "calibrate.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if apply and found:
        P.cfg = json.loads(cfg_text)
        rig = P.cfg.setdefault("rig", {})
        rig.update(found)
        P.save_cfg()
        print("character.json に書きました: " + "、".join(f"{KEY_JA[k]} = {v}" for k, v in found.items()) + "。組み立て直します", flush=True)
        sh([sys.executable, "-X", "utf8", os.path.join(kit, "run.py"), P.d, "--from", "build", "--to", "preview"])
        print("組み立て直しました。動きの確認画像（「確認画像を作る」）で見てください", flush=True)


# ------------------------------------------------------------------------------------------------------- qa
CHECKS = {
    "motion.png": "首振り（物理込み、表示サイズ）: 髪の束がほつれない・顔が前髪の下から滑り出ない・横髪が頬から離れない・帽子や飾りが頭から浮かない",
    "blink.png": "まばたき1コマずつ: 上まぶたが上から下りて目の下寄りで閉じる・途中でまつ毛がジグザグに折れない・閉じたあと目の上に線や印が残らない",
    "closed_eyes.png": "閉じた目の拡大: 白目・黒目・まつ毛の端・二重の線・前髪に透けた線が残っていない",
    "bare.png": "揺れる髪（hair_front と lock_*、外した層は図の上に書いてある）を外した状態＝揺れたときに見えるもの: おでこ・生え際・耳・こめかみが描かれている、四角い穴や灰色の塊がない。揺れない髪（hair_base）は残っていてよい",
    "far_sides.png": "首を回し切ったときの奥側: 頬の横にすき間がない、耳がちぎれない、後ろ髪が二重にならない",
    "nod.png": "見上げ・見下ろし: 首が不自然に長くない、あごの下に影、帽子が頭から浮かない、前髪と顔がずれない",
    "mouth.png": "あいうえお: 口の形が自然、歯や舌がはみ出さない",
    "expressions.png": "表情: それぞれ崩れていない",
    "holes.png": "81通りの姿勢でのすき間（ピンク）: 大きなピンクがない",
}


def qa(P, label=None, compare=None):
    import preview as PV
    from l2d.render import Renderer
    name = P.name
    out_dir = P.p("out", name)
    label = label or time.strftime("%Y%m%d_%H%M%S")
    d = P.p("preview", "qa", label)
    os.makedirs(d, exist_ok=True)
    HX, HY, HR = P.head()
    box = head_crop_box(P)
    s_view = round(min(0.6, max(0.12, 700.0 / (box[3] - box[1]))), 3)
    boxs = ",".join(str(v) for v in box)
    ms = os.path.join(kit_of(P), "motion_strip.py")
    cmp_ = ["--compare", compare] if compare else []
    sh([sys.executable, "-X", "utf8", ms, out_dir, os.path.join(d, "motion.png"), "--box", boxs, "--scale", str(s_view)] + cmp_)
    eye_box = [int(HX - 1.1 * HR), int(P.pt("eye_l")[1] - 0.7 * HR), int(HX + 1.1 * HR), int(P.pt("eye_l")[1] + 0.6 * HR)]
    sh([sys.executable, "-X", "utf8", ms, out_dir, os.path.join(d, "blink.png"), "--box", ",".join(str(v) for v in eye_box),
        "--scale", str(round(min(1.0, 900.0 / (eye_box[2] - eye_box[0])), 3)), "--blink", "1.0"] + cmp_)
    L = PV.Look(P, out_dir)
    ids = list(L.m.drawable_ids)
    F = platform_env.font(16)

    def tile(vals, crop, scale, skip=None, title=""):
        L.m.reset(); L.m.set(**vals); L.m.update()
        R = Renderer(L.m, L.tex, scale=scale, crop=tuple(int(v) for v in crop))
        im = Image.fromarray(C.on_dark(R.render(background=None, skip=skip)))
        ImageDraw.Draw(im).text((6, 4), title, font=F, fill=(255, 220, 120))
        return np.asarray(im)

    def row(tiles):
        h = max(t.shape[0] for t in tiles)
        return np.concatenate([np.pad(t, ((0, h - t.shape[0]), (0, 4), (0, 0))) for t in tiles], 1)

    def save(name_, rows_):
        w = max(r.shape[1] for r in rows_)
        Image.fromarray(np.concatenate([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows_], 0)).save(os.path.join(d, name_))

    sc_eye = min(1.5, 900.0 / (eye_box[2] - eye_box[0]))
    save("closed_eyes.png", [row([tile({}, eye_box, sc_eye, title="開いた目"), tile({"ParamEyeLOpen": 0, "ParamEyeROpen": 0}, eye_box, sc_eye, title="閉じた目")]),
                             row([tile({"ParamEyeLOpen": 0.5, "ParamEyeROpen": 0.5}, eye_box, sc_eye, title="半分"),
                                  tile({"ParamEyeLOpen": 0, "ParamEyeROpen": 0, "ParamEyeLSmile": 1, "ParamEyeRSmile": 1}, eye_box, sc_eye, title="笑って閉じた目")])])
    front = {i for i in ids if i.startswith("lock_") or i.startswith("hair_front")}
    n_locks = sum(1 for i in front if i.startswith("lock_"))
    gone = "・".join(x for x in (("hair_front" if any(i.startswith("hair_front") for i in front) else ""),
                                 (f"髪の束 {n_locks} 本" if n_locks else "")) if x) or "なし"
    sc_head = min(0.5, 600.0 / (box[2] - box[0]))
    save("bare.png", [row([tile(v, box, sc_head, skip=front, title=f"{t}・外した髪: {gone}") for v, t in (({}, "正面"), ({"ParamAngleX": 30}, "首 X+30"),
                                                                                                   ({"ParamAngleX": -30}, "首 X-30"))])])
    sc_side = min(0.7, 500.0 / (2.4 * HR))
    far = []
    for ax in (30, -30):
        for ay in (0, -30, 30):
            far.append(tile({"ParamAngleX": ax, "ParamAngleY": ay, "ParamHairFront": 1 if ax > 0 else -1, "ParamHairBack": 1 if ax > 0 else -1},
                            (HX - 1.9 * HR, HY - 0.3 * HR, HX + 1.9 * HR, P.pt("chin")[1] + 0.9 * HR), sc_side, title=f"首 X{ax:+d} Y{ay:+d}・髪を振った"))
    save("far_sides.png", [row(far[:3]), row(far[3:])])
    save("nod.png", [row([tile({"ParamAngleY": v}, box, sc_head, title=("見上げ" if v > 0 else "見下ろし" if v < 0 else "正面") + f" Y{v:+d}") for v in (30, 15, 0, -15, -30)])])
    mb = (P.pt("mouth")[0] - 0.7 * HR, P.pt("mouth")[1] - 0.4 * HR, P.pt("mouth")[0] + 0.7 * HR, P.pt("mouth")[1] + 0.45 * HR)
    sc_m = min(2.0, 500.0 / (1.4 * HR))
    vow = [({"ParamMouthOpenY": 0}, "閉じた口")] + [({"ParamMouthOpenY": 1, p: 1}, k) for p, k in (("ParamA", "あ"), ("ParamI", "い"), ("ParamU", "う"), ("ParamE", "え"), ("ParamO", "お"))]
    save("mouth.png", [row([tile(v, mb, sc_m, title=t) for v, t in vow if all(k in L.m.param_index for k in v)])])
    exps = []
    mj = json.load(open(os.path.join(out_dir, f"{name}.model3.json"), encoding="utf-8"))
    for e in mj["FileReferences"].get("Expressions", [])[:9]:
        ex = json.load(open(os.path.join(out_dir, e["File"]), encoding="utf-8"))
        # (as VTube Studio applies them to the face at rest: Add to the rest value, Multiply it, Overwrite replaces it)
        vals = {}
        for p in ex.get("Parameters", []):
            if p["Id"] not in L.m.param_index:
                continue
            rest = float(L.m.param_default[L.m.param_index[p["Id"]]])
            blend = p.get("Blend", "Overwrite")
            vals[p["Id"]] = rest + p["Value"] if blend == "Add" else rest * p["Value"] if blend == "Multiply" else p["Value"]
        exps.append(tile(vals, box, sc_head * 0.8, title=EXP_JA.get(e["Name"], e["Name"])))
    if exps:
        save("expressions.png", [row(exps[i:i + 5]) for i in range(0, len(exps), 5)])
    if os.path.exists(P.p("preview", "holes.png")):
        shutil.copy2(P.p("preview", "holes.png"), os.path.join(d, "holes.png"))
    try:
        phys = sh([sys.executable, "-X", "utf8", os.path.join(kit_of(P), "phys_check.py"), P.d])
        open(os.path.join(d, "phys.txt"), "w", encoding="utf-8").write(phys)
    except Exception as e:
        open(os.path.join(d, "phys.txt"), "w", encoding="utf-8").write(f"物理演算の確認ができませんでした: {e}")
    lines = [f"# 確認パック {label}", "", f"モデル: out/{name}" + (f"（比較: {compare}）" if compare else ""), ""]
    for f_, what in CHECKS.items():
        if os.path.exists(os.path.join(d, f_)):
            lines.append(f"- **{f_}** — {what}")
    lines += ["- **phys.txt** — 揺れものの物理の振れ幅（動きすぎ・動かなさすぎがないか）", ""]
    open(os.path.join(d, "qa.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"確認パック: preview/qa/{label}/（{len(os.listdir(d))} ファイル）")
    return label


def review(P, label=None):
    import ai
    base = P.p("preview", "qa")
    if not label:
        labels = sorted((x for x in os.listdir(base) if not x.startswith(".")),
                        key=lambda x: os.path.getmtime(os.path.join(base, x))) if os.path.isdir(base) else []
        if not labels:
            sys.exit("確認パックがまだありません。先に「確認画像を作る」（hq.py <プロジェクト> qa）をしてください")
        label = labels[-1]
    d = os.path.join(base, label)
    imgs = [f"preview/qa/{label}/{f}" for f in CHECKS if os.path.exists(os.path.join(d, f))]
    phys = open(os.path.join(d, "phys.txt"), encoding="utf-8").read()[-2500:] if os.path.exists(os.path.join(d, "phys.txt")) else ""
    checklist = "\n".join(f"- {f}: {w}" for f, w in CHECKS.items())
    prompt = f"""You review a VTube Studio (Live2D) model made automatically from one illustration. The pictures are its check pack:
renders of the model in many poses (the head shake and the blink are frame by frame, left to right, top to bottom; where a
set of rows is labelled "前の版", it is the previous version and the rows labelled "今の版" are this one). The labels and
the physics notes are in Japanese. What each picture is for:
{checklist}

Look at every picture carefully, at the size it is shown and zoomed in on details. List every visible defect: something
that breaks, tears, shows a gap or the dark background where the character should be, stays behind when it should move,
moves when it should stay, looks stretched, or does not match the illustration's look. For each: which picture and which
frame or pose, where on the character, how bad (大 = obvious at normal size / 中 / 小 = only zoomed in), and the likely
cause in the rig (which part or layer). Do not list things that look right. If nothing is wrong in a picture, say so in
one line.

Physics ranges (phys_check):
{phys}

Write the answer in JAPANESE as a short markdown report: first a one-line overall verdict (VTube Studio に入れてよいか),
then the defects ordered by severity, then one line per picture that looked fine."""
    text, who = ai.ask(P.d, prompt, imgs, tag="review", timeout=1500)
    out = os.path.join(d, "review.md")
    open(out, "w", encoding="utf-8").write(f"<!-- reviewed by {who}, {time.strftime('%Y-%m-%d %H:%M')} -->\n" + text.strip() + "\n")
    print(f"AIの確認（{who}）: preview/qa/{label}/review.md")
    print(text.strip()[:3000])


def snapshot(P, label=None):
    label = label or time.strftime("%Y%m%d_%H%M%S")
    src = P.p("out", P.name)
    dst = P.p("versions", label)
    if os.path.exists(dst):
        sys.exit(f"versions/{label} はもうあります。別の名前にしてください")
    shutil.copytree(src, dst)
    shutil.copy2(P.cfg_path, os.path.join(dst, "character.json"))
    print(f"残しました: versions/{label}")


def main():
    P = C.open_project()
    cmd = sys.argv[2] if len(sys.argv) > 2 else ""
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    if cmd == "refs":
        refs(P)
    elif cmd == "calibrate":
        calibrate(P, apply="--apply" in sys.argv)
    elif cmd == "qa":
        qa(P, arg("--label"), arg("--compare"))
    elif cmd == "review":
        review(P, arg("--label"))
    elif cmd == "snapshot":
        snapshot(P, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
