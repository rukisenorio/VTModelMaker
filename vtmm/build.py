"""Stage 5: the rig - layers/ -> out/<name>/ (moc3, textures, physics3, cdi3, expressions), with no Cubism Editor.

usage: build.py <project> [--lowres]

The second character's rig (chara2/build_model.py) made general: every distance is in head radii (measured from the
landmarks and the face layer), every place from character.json / work/*.json. The tree:
  W_Body       lean (feet stay), turn (the chest as a cylinder about the spine), bow, breath
    arms       rotation at each shoulder (physics sway + breath)
    neck       its top goes a little with the head's turn
    body       everything else
    W_Face     the head's turn / nod / lean: the face's own flatter surface for turning (z_head_yaw), a plate of one
               depth from the brows to the chin for nodding (the face slides inside the head, eyes only foreshorten),
               the chin drops with the mouth
      eyes     band-map blink (the whole upper lash maps onto the drawn closed line and thins to its weight; the lids
               meet at 0.1; the drawn closed eye takes over only where they coincide), gaze, smile-closing
      brows, mouth (the line's ribbon bent into every shape + interior / tongue / teeth), blush
    W_HairBack / W_HairFront   the hair: turned with the head on their own depths (KEEP of the cosine at the outline),
               swung as pendulums about the crown in two stages, side hair bobbing; near the cut between them the back
               hair moves as the front hair does (JOIN)
Options (character.json "rig"): yaw, pitch, roll (degrees at 30), corner, front_deg, back_deg, side_px, blush (bool).
"""
import os, sys, json, math, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy import ndimage
import common as C
from common import sstep
rad = math.radians

P_ = C.open_project()
import layerio
from l2d.rig import Rig

OPT = dict(P_.cfg.get("rig", {}))
# (a run may try other values without touching character.json - hq.py calibrate builds candidates this way)
OPT.update(json.loads(os.environ.get("VTMM_RIG") or "{}"))
LOWRES = "--lowres" in sys.argv
MODEL = P_.name
W, H = P_.canvas_size()
HX, HY, HR = P_.head()
u = HR / C.TEMPLATE_HR                      # (distances below were tuned on the second character, HR 420)
pt = P_.pt
EL, ER = pt("eye_l"), pt("eye_r")
EYE_Y = float(0.5 * (EL[1] + ER[1]))
EYES_X = (float(EL[0]), float(ER[0]))
MOUTH = pt("mouth")
MOUTH_Y = float(MOUTH[1])
CHIN_Y = float(pt("chin")[1])
NOSE = pt("nose", np.array([HX, 0.5 * (EYE_Y + MOUTH_Y)]))
bl_, br_ = pt("brow_l"), pt("brow_r")
BROW_Y = float(0.5 * (bl_[1] + br_[1])) if bl_ is not None and br_ is not None else EYE_Y - 0.4 * HR
CROWN = (HX, float(pt("crown")[1]) + 0.05 * HR) if pt("crown") is not None else (HX, HY - HR)
NECK = (HX, CHIN_Y)
HEAD_C = (HX, HY + 0.14 * HR)
HIP = tuple(pt("hip")) if pt("hip") is not None else (HX, float(H))
FEET_Y = float(max(pt("foot_l")[1], pt("foot_r")[1])) if pt("foot_l") is not None else float(H) + 1000.0
SHOULDER = {s: tuple(pt(f"shoulder_{s}")) for s in ("l", "r") if pt(f"shoulder_{s}") is not None}
SH_Y = float(np.mean([v[1] for v in SHOULDER.values()])) if SHOULDER else CHIN_Y + 1.6 * HR
YAW = float(OPT.get("yaw", 32.0)); PITCH = float(OPT.get("pitch", 28.0)); ROLL = float(OPT.get("roll", 13.0))
CORNER = float(OPT.get("corner", 0.65))
# (rig.yaw_amp: the depth of the face's turn surface - matched to drawn 3/4 views, hq.py calibrate; chara4 needed 1.3)
YAW_POWER, YAW_AMP, YAW_REACH = 0.6, float(OPT.get("yaw_amp", 0.82)), 1.25
YAW_BLEND = (0.95, 1.6)
KEEP, OUTER = 0.25, (1.3, 1.7)
OUT_Z = 0.10 * HR
# (rig.nod_up_scale: how far the face rises looking up - matched to a drawn view from below; chara4 needed 1.6)
NOD_UP = tuple(v * u * float(OPT.get("nod_up_scale", 1.0)) for v in (377.0, 368.0, 334.0))
NOD_DN = tuple(v * u for v in (520.0, 415.0, 395.0))
FRONT_W = 360.0 * u
JAW_DROP = 24.0 * u
FRONT_DEG = tuple(OPT.get("front_deg", (5.2, 4.8)))
BACK_DEG = tuple(OPT.get("back_deg", (6.5, 5.5)))
FRONT_D = ((380.0 * u, 820.0 * u), (600.0 * u, 860.0 * u))
BACK_D = ((300.0 * u, 820.0 * u), (560.0 * u, 1000.0 * u))
SIDE_DX, SIDE_DY = (float(v) * u for v in OPT.get("side_px", (30.0, 22.0)))
TOP_DROP = 90.0 * u
TOP_DROP_Y = (BROW_Y - 30.0 * u, CROWN[1] - 260.0 * u)
UNFOLD, UNFOLD_FACE = 0.2, 0.03
JOIN_FULL, JOIN_BAND = 100.0 * u, 260.0 * u
JOIN_Y = (EYE_Y + 17.0 * u, MOUTH_Y)
BODY_TURN, BODY_R, NECK_DEPTH, BODY_SHIFT = 10.0, 640.0 * u, 0.25, 10.0 * u
SWING_L = 3.4 * HR                  # (the second character's hair reaches about this far from the crown)


def sz(v):
    """A px size of the template in this character's px."""
    return v * u


# ---------------------------------------------------------------------------------------------- the head's surfaces
_HEAD_R = None


def head_radius():
    """The head's radius per direction from HEAD_C (one value a degree, smoothed), from the face layer (which holds the
    whole head's skin: segment.py gives it the bald plate's scalp under the hair)."""
    global _HEAD_R
    if _HEAD_R is None:
        hs = P_.p("work", "head_skin.png")
        if os.path.exists(hs):
            m = np.asarray(Image.open(hs))[::2, ::2] > 127
            fy, fx = np.nonzero(m)
            X, Y = fx * 2.0, fy * 2.0
        else:
            L, x0, y0 = layerio.load_layer("face")
            fy, fx = np.nonzero(L[::2, ::2, 3] > 128)
            X, Y = fx * 2.0 + x0, fy * 2.0 + y0
        dx, dy = X - HEAD_C[0], Y - HEAD_C[1]
        ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        R = np.zeros(360)
        np.maximum.at(R, ang.astype(int) % 360, np.hypot(dx, dy))
        R = np.where(R > 0, R, np.median(R[R > 0]))
        Rp = np.concatenate([R[-15:], R, R[:15]])
        _HEAD_R = np.convolve(Rp, np.ones(9) / 9.0, mode="same")[15:-15]
    return _HEAD_R


def head_r(p):
    R = head_radius()
    dx, dy = p[:, 0] - HEAD_C[0], p[:, 1] - HEAD_C[1]
    ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    return np.hypot(dx, dy) / np.maximum(np.interp(ang, np.arange(360) + 0.5, R, period=360), 1.0)


def z_head(p):
    r = head_r(p)
    return 0.85 * HR * np.clip(1 - r * r, 0, None) ** 0.75


def z_head_yaw(p):
    r = head_r(p) / YAW_REACH
    return YAW_AMP * HR * np.clip(1 - r * r, 0, None) ** YAW_POWER


def yaw_w(p):
    return 1.0 - sstep((head_r(p) - YAW_BLEND[0]) / (YAW_BLEND[1] - YAW_BLEND[0]))


def nose_bump(p, a, sx, sy):
    return a * np.exp(-(((p[:, 0] - NOSE[0]) / sx) ** 2 + ((p[:, 1] - NOSE[1]) / sy) ** 2))


def z_face(p):
    return z_head(p) + nose_bump(p, sz(40), sz(90), sz(90))


_Z_EYES = None


def z_eyes():
    global _Z_EYES
    if _Z_EYES is None:
        _Z_EYES = float(z_head(np.array([[EYES_X[0], EYE_Y], [EYES_X[1], EYE_Y]])).mean())
    return _Z_EYES


def z_head_nod(p):
    zd = z_head(p)
    q = np.stack([p[:, 0], np.full(len(p), EYE_Y)], 1)
    front = np.minimum(z_head(q), z_eyes()) * (1.0 - 0.06 * sstep((p[:, 1] - (MOUTH_Y - sz(2))) / sz(190)))
    t = sstep((p[:, 1] - EYE_Y) / sz(220))
    return zd * (1 - t) + np.maximum(zd, front) * t


def z_face_nod_dome(p):
    return z_head_nod(p) + nose_bump(p, sz(40), sz(90), sz(90))


_Z_MID_YAW = None


def plate_profile(p):
    global _Z_MID_YAW
    q = np.stack([p[:, 0], np.full(len(p), EYE_Y)], 1)
    if _Z_MID_YAW is None:
        _Z_MID_YAW = float(z_head_yaw(np.array([[HX, EYE_Y]]))[0])
    return np.clip(z_head_yaw(q) / _Z_MID_YAW, 0, 1)


def z_face_plate(p, down=False):
    """For the nod: one depth from the brows to below the eyes (they only foreshorten), the lower face forward of it
    looking up (the jaw comes toward one) and behind it looking down (the chin tucks in)."""
    from scipy.interpolate import PchipInterpolator
    ze, zm, zc = NOD_DN if down else NOD_UP
    ys = [BROW_Y - sz(30), EYE_Y + sz(55), MOUTH_Y - sz(15), max(CHIN_Y, MOUTH_Y + sz(20))]
    y = p[:, 1]
    zmid = PchipInterpolator(ys, [ze, ze, zm, zc])(np.clip(y, ys[0], ys[-1]))
    plate = zmid * plate_profile(p)
    t = sstep((y - (BROW_Y - sz(190))) / sz(160))
    return z_head(p) * (1 - t) + plate * t + nose_bump(p, sz(30), sz(70), sz(55))


_FACE_HW = None


def face_half_widths():
    global _FACE_HW
    if _FACE_HW is None:
        L, x0, y0 = layerio.load_layer("face")
        a = L[..., 3] > 128
        rows = np.arange(a.shape[0]) + y0
        hl = np.zeros(len(rows)); hr = np.zeros(len(rows))
        for i in range(a.shape[0]):
            xs = np.nonzero(a[i])[0]
            if len(xs):
                hl[i] = HX - (xs.min() + x0); hr[i] = xs.max() + x0 - HX
        kk = np.ones(31) / 31.0
        hl = np.convolve(np.pad(hl, 15, mode="edge"), kk, mode="valid"); hr = np.convolve(np.pad(hr, 15, mode="edge"), kk, mode="valid")
        _FACE_HW = (rows.astype(float), np.maximum(hl, sz(60)), np.maximum(hr, sz(60)))
    return _FACE_HW


def z_face_yaw(p):
    return z_head_yaw(p) + nose_bump(p, sz(40), sz(90), sz(90))


def z_face_turn(p):
    """The face for the turn: the flatter dome above the eyes; below them a front, so that nose, mouth and chin go aside
    with the eyes (with less, the lower face sheared - it reads as stretched)."""
    zd = z_face_yaw(p)
    q = np.stack([np.full(len(p), HX), np.minimum(p[:, 1], MOUTH_Y)], 1)
    zmid = np.maximum(z_face_nod_dome(q), z_face_yaw(q))
    dx = p[:, 0] - HX
    prof = 0.5 * (1.0 + np.cos(np.pi * np.clip(np.abs(dx) / FRONT_W, 0, 1)))
    cyl = zmid * prof
    t = sstep((p[:, 1] - EYE_Y) / sz(220))
    return zd * (1 - t) + np.maximum(zd, cyl) * t


def z_face_down(p):
    return z_face_plate(p, down=True)


def keep_hair(p):
    r = head_r(p)
    return KEEP + (1.0 - KEEP) * (1.0 - sstep((r - OUTER[0]) / (OUTER[1] - OUTER[0])))


def fringe_w(p):
    return sstep((p[:, 1] - (BROW_Y - sz(135))) / sz(170)) * (1.0 - sstep((np.abs(p[:, 0] - HX) - sz(300)) / sz(110)))


def z_side(p, zf=None):
    """Beside the face - side locks over the cheeks and the ears - the depth of the face's edge next to it, fading out
    0.6 HR beyond it (chara3's rule). On the head sphere's depth, all but nothing out there, the face's edge moved 35 px
    at a 10-degree turn and the side locks beside it 10-20: the face slid out from under them at every turn (chara4 and
    the other session's chara5, 2026-10-23)."""
    rows, hl, hr = face_half_widths()
    y = p[:, 1]
    dxp = p[:, 0] - HX
    left = dxp < 0
    hw = np.where(left, np.interp(y, rows, hl), np.interp(y, rows, hr))
    edge = np.stack([HX + np.where(left, -hw, hw), y], 1)
    beyond = np.abs(dxp) - hw
    w = (1.0 - sstep(beyond / (0.6 * HR))) * sstep((y - (BROW_Y - 0.2 * HR)) / (0.3 * HR)) * \
        (1.0 - sstep((y - (CHIN_Y + 0.1 * HR)) / (0.3 * HR)))
    return np.where(beyond > 0, (zf or z_face_turn)(edge) * w, 0.0)


def z_hair_front(p):
    w0 = yaw_w(p)
    return np.maximum(z_head_yaw(p) * w0 + z_head(p) * (1 - w0), z_side(p)) + sz(30)


def z_hair_front_nod(p, down=False):
    z = z_head(p) + sz(30)
    fw = fringe_w(p)
    return z * (1 - fw) + (z_face_plate(p, down) + sz(14)) * fw


def z_hair_back(p):
    """The back hair's depth: over the skull's outline it is the skull's own hair surface (the crown seen from the front
    lies on the head's top-front, as the front hair does); beyond the outline it falls behind the head. One field for
    the whole layer, so nothing in it tears: with the crown in the back layer (the strand-by-strand split) a depth
    behind the head there had moved the crown against the face at a turn."""
    r = head_r(p)
    behind = -0.4 * z_head(p) - OUT_Z * sstep((r - OUTER[0]) / (OUTER[1] - OUTER[0]))
    s_ = sstep((r - 0.9) / 0.4)
    return z_hair_front(p) * (1.0 - s_) + behind * s_


_JOIN_D = None


def join_w(p):
    """Near the cut between front and back hair the back hair moves as the front hair does."""
    global _JOIN_D
    if _JOIN_D is None:
        lab = np.asarray(Image.open(P_.p("work", "labels.png")))
        ids = json.load(open(P_.p("work", "labels.json")))["id"]
        hf = lab[::4, ::4] == ids["hair_front"]
        _JOIN_D = ndimage.distance_transform_edt(~hf) * 4.0
    from scipy.ndimage import map_coordinates
    d = map_coordinates(_JOIN_D, [p[:, 1] / 4.0, p[:, 0] / 4.0], order=1, mode="nearest")
    t = np.clip((d - JOIN_FULL) / JOIN_BAND, 0, 1)
    j = 1.0 - t * t * (3 - 2 * t)
    return j * (1.0 - sstep((p[:, 1] - JOIN_Y[0]) / max(JOIN_Y[1] - JOIN_Y[0], 1.0)))


def top_drop(pts, ph):
    if ph <= 0 or TOP_DROP <= 0:
        return 0.0
    t = sstep((TOP_DROP_Y[0] - pts[:, 1]) / (TOP_DROP_Y[0] - TOP_DROP_Y[1]))
    return TOP_DROP * (ph / rad(PITCH)) * t


def lean_w(p):
    """How much of the head's lean a point takes: all of the head, less and less of hair hanging far below the chin."""
    return 1.0 - 0.65 * sstep((p[:, 1] - (CHIN_Y + 0.4 * HR)) / (2.6 * HR))


_BODY_TOP = None


def body_top(x):
    """Per column, the top row of the body's main mass (canvas px; inf where the column has no body) - where the hair
    comes to lie on it."""
    global _BODY_TOP
    if _BODY_TOP is None:
        lab = np.asarray(Image.open(P_.p("work", "labels.png")))[::4, ::4]
        ids = json.load(open(P_.p("work", "labels.json")))["id"]
        b = lab == ids["body"]
        lb, n = ndimage.label(b)
        if n:
            b = lb == (int(np.argmax(np.bincount(lb.ravel())[1:])) + 1)
        top = np.where(b.any(0), np.argmax(b, 0) * 4.0, 1e9)
        # (never above the chin: a hat, a high collar or wings labelled body there made all the hair "rest" on the body -
        # the other session's chara5 turned its hair 15 % of the head's turn)
        _BODY_TOP = np.maximum(ndimage.median_filter(top, size=9), CHIN_Y - 0.1 * HR)
    t = _BODY_TOP[np.clip((x / 4.0).astype(int), 0, len(_BODY_TOP) - 1)]
    return np.where(t > 1e8, np.inf, t)


def turn(zf, sway=None, zf_nod=None, zf_down=None, keep=None, drop=False, rests=False):
    def fn(pts, v):
        cf = 1.0 - (1.0 - CORNER) * min(abs(v["ParamAngleX"]) / 30.0, 1.0) * min(abs(v["ParamAngleY"]) / 30.0, 1.0)
        th = rad(YAW * v["ParamAngleX"] / 30.0) * cf
        ph = rad(PITCH * v["ParamAngleY"] / 30.0) * cf
        x = pts[:, 0] - HX
        y = pts[:, 1] - HY
        z = zf(pts)
        zn = z if zf_nod is None else zf_nod(pts)
        if ph < 0 and zf_down is not None:
            zn = zf_down(pts)
        kk = np.ones(len(pts)) if keep is None else keep(pts)
        # (below the chin - long hair - the turn and nod fade: it hangs from the head but lies on the shoulders and along
        # the body. Faded only far below, the hair on the shoulders followed a nod and lifted off them: the page showed
        # in a band over each shoulder)
        y_a, y_b = CHIN_Y + 0.2 * HR, max(SH_Y + 0.3 * HR, CHIN_Y + 0.8 * HR)
        off = sstep((pts[:, 1] - y_a) / (y_b - y_a))
        if rests:
            # (hair: where it comes to lie on the body - the collar's top beside the neck, the shoulders - it stays with the
            # body; at the collar it still took 90 % of a nod and a band of page opened over each shoulder)
            bt = body_top(pts[:, 0])
            off = np.maximum(off, np.where(np.isfinite(bt), sstep((pts[:, 1] - (bt - 0.15 * HR)) / (0.35 * HR)), 0.0))
        hang = 1.0 - 0.85 * off
        th_p, ph_p = th * hang, ph * hang
        cx = 1.0 - (1.0 - np.cos(th_p)) * kk
        cy = 1.0 - (1.0 - np.cos(ph_p)) * np.where(ph_p > 0, kk, 1.0)
        x1 = x * cx + z * np.sin(th_p)
        z1 = -x * np.sin(th_p) + zn * np.cos(th_p)
        y1 = y * cy - z1 * np.sin(ph_p)
        if drop:
            y1 = y1 + top_drop(pts, ph) * hang
        out = np.stack([x1 + HX, y1 + HY], 1)
        if sway is not None:
            out = sway(out, pts, v)
        # the head's lean about the neck
        a = rad(-ROLL * v["ParamAngleZ"] / 30.0) * lean_w(pts)
        qx, qy = out[:, 0] - NECK[0], out[:, 1] - NECK[1]
        return np.stack([NECK[0] + qx * np.cos(a) - qy * np.sin(a), NECK[1] + qx * np.sin(a) + qy * np.cos(a)], 1)
    return fn


def pendulum(D, DEG, key1, key2, damp=None):
    """Hair hung from the crown: every point turned about it by an angle that grows with its distance from it, in two
    stages - the mass, and its ends trailing it."""
    def sway(out, pts, v):
        dx, dy = pts[:, 0] - CROWN[0], pts[:, 1] - CROWN[1]
        d = np.hypot(dx, dy)
        root = sstep((d - D[0][0]) / (D[0][1] - D[0][0])) ** 1.12
        tip = sstep((d - D[1][0]) / (D[1][1] - D[1][0])) ** 1.45
        dphi = np.radians(DEG[0] * root * v[key1] + DEG[1] * tip * v[key2])
        # (long hair: beyond SWING_L the tips' displacement no longer grows with the length - 340 px at the elbows of a
        # waist-long mane flung it off the body)
        dphi = dphi * np.minimum(1.0, SWING_L / np.maximum(d, 1.0))
        if damp is not None:
            dphi = dphi * damp(pts)
        phi = np.arctan2(dx, dy)
        out = out.copy()
        out[:, 0] += d * np.sin(phi + dphi) - dx
        out[:, 1] += d * np.cos(phi + dphi) - dy
        return out
    return sway


_FACE_D = None


def face_dist(p):
    """Distance (canvas px) from the face as it is at rest (the head's skin)."""
    global _FACE_D
    if _FACE_D is None:
        hs = P_.p("work", "head_skin.png")
        m = (np.asarray(Image.open(hs))[::4, ::4] > 127) if os.path.exists(hs) else np.zeros((H // 4 + 1, W // 4 + 1), bool)
        _FACE_D = ndimage.distance_transform_edt(~m) * 4.0
    from scipy.ndimage import map_coordinates
    return map_coordinates(_FACE_D, [p[:, 1] / 4.0, p[:, 0] / 4.0], order=1, mode="nearest")


def side_damp(pts):
    """Beside the face below the eyes - the side locks along the cheeks and the jaw - the front hair's pendulum swings a
    third as far. Hung from the crown, the locks reaching the jaw swung 140 px and left the cheek: a crescent of page
    opened along the far cheek at a turn (chara4, 2026-10-23; the drawn 3/4 view has them lying along the cheek). Their
    own lock pendulums keep them moving."""
    fd = face_dist(pts)
    beside = (1.0 - sstep((fd - 0.25 * HR) / (0.5 * HR))) * sstep((np.abs(pts[:, 0] - HX) - 0.45 * HR) / (0.3 * HR))
    low = sstep((pts[:, 1] - EYE_Y) / (0.4 * HR))
    return 1.0 - 0.65 * beside * low


def nape_damp(pts):
    nape = sstep((pts[:, 1] - (CHIN_Y - sz(300))) / sz(180)) * (1.0 - sstep((np.abs(pts[:, 0] - HX) - sz(430)) / sz(160)))
    # (and next to the face - the hair at the temples and above the ears grows there: it swings only a fifth; swung fully,
    # it opened the page between itself and the face at every swing of the back hair)
    near_face = 1.0 - sstep(face_dist(pts) / (0.4 * HR))
    return (1.0 - 0.75 * nape) * (1.0 - 0.8 * near_face)


def side_bob(pts, v, join=None):
    """Hair beside the face (beyond its width) bobs out and up on its own pendulums."""
    x, y = pts[:, 0], pts[:, 1]
    dl = np.clip((HX - 0.95 * HR - x) / sz(380), 0, 1) ** 1.3 * sstep((y - (EYE_Y - sz(500))) / sz(300))
    dr = np.clip((x - HX - 0.95 * HR) / sz(380), 0, 1) ** 1.3 * sstep((y - (EYE_Y - sz(500))) / sz(300))
    dyv = SIDE_DY * v["ParamHairSideL"] * dl + SIDE_DY * v["ParamHairSideR"] * dr
    dxv = SIDE_DX * v["ParamHairSideL"] * dl - SIDE_DX * v["ParamHairSideR"] * dr
    if join is not None:
        kk = 1.0 - join
        dxv, dyv = dxv * kk, dyv * kk
    return np.stack([x + dxv, y + dyv], 1)


# ---------------------------------------------------------------------------------------------- the mouth, as a rig
#            w     cy    dip   ou     ol    eu    el    teeth tongue taper cu    cat  tl   (the second character's shapes)
M_BASE = {
    (0.0, 0): (1.00, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.5, 0): (1.02, 0.0, 0.4, -3.0, 50.0, 0.6, 0.6, 8.0, 0.5, 0.25, -20.0, 0.0, 0.0),
    (1.0, 0): (1.04, 0.0, 0.3, -7.0, 68.0, 0.5, 0.5, 10.0, 1.0, 0.15, -30.0, 0.0, 0.0),
    (0.0, 1): (1.04, -2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 2.6, 0.0),
    (0.5, 1): (1.26, -2.0, 0.8, -3.0, 40.0, 1.0, 0.8, 12.0, 0.4, 0.25, 4.0, 0.0, 0.0),
    (1.0, 1): (1.44, -4.0, 0.6, -4.0, 56.0, 1.0, 0.7, 14.0, 1.0, 0.15, 6.0, 0.0, 0.0),
    (0.0, -1): (0.82, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -11.0, 0.0, 0.0),
    (0.5, -1): (0.62, 0.0, 0.0, -2.0, 30.0, 0.5, 0.5, 0.0, 0.25, 0.0, -16.0, 0.0, 0.0),
    (1.0, -1): (0.74, -4.0, 0.0, -2.0, 46.0, 0.5, 0.5, 0.0, 0.45, 0.0, -24.0, 0.0, 0.0),
}
M_VOWEL = {"ParamA": {1.0: (1.04, 0.0, 0.3, -7.0, 83.0, 0.5, 0.5, 10.0, 1.0, 0.15, -35.0, 0.0, 0.0)},
           "ParamI": {1.0: (1.35, 6.0, 0.6, 0.0, 40.0, 0.8, 0.8, 16.0, 0.0, 0.3, -14.0, 0.0, 22.0)},
           "ParamU": {0.5: (0.42, 2.0, 0.0, 0.0, 22.0, 0.5, 0.5, 0.0, 0.0, 0.0, -11.0, 0.0, 0.0), 1.0: (0.46, 3.0, 0.0, 0.0, 36.0, 0.5, 0.5, 0.0, 0.1, 0.0, -18.0, 0.0, 0.0)},
           "ParamE": {1.0: (1.14, -3.0, 0.6, 0.0, 67.0, 1.0, 0.75, 12.0, 0.7, 0.2, -5.0, 0.0, 0.0)},
           "ParamO": {0.5: (0.62, -2.0, 0.0, 0.0, 32.0, 0.5, 0.5, 0.0, 0.2, 0.0, -16.0, 0.0, 0.0), 1.0: (0.76, -5.0, 0.0, 0.0, 56.0, 0.5, 0.5, 0.0, 0.45, 0.0, -28.0, 0.0, 0.0)}}
PX_COLS = (1, 3, 4, 7, 10, 11, 12)          # (the columns in px: they scale with the mouth's size)


def mouth_rig_parts(rig, wface, zf, K3):
    import mouth_rig as MR
    mg = json.load(open(P_.p("work", "mouth_line.json")))
    MR.HW = float(mg.get("hw", 8.0))
    MG = MR.Geom(mg)
    NS, HW_LINE = MR.NS, MR.HW
    cxm, w0, ss, x_rest = MG.cxm, MG.w0, MG.ss, MG.x_rest
    cl, Dp = MG.cl, MG.Dp
    sm = w0 / 48.18                                 # (the second character's mouth: 48 px half-width)
    scale_row = lambda r: np.array([x * (sm if i in PX_COLS else 1.0) for i, x in enumerate(r)], float)
    prof = lambda e: np.clip(1 - (2 * ss - 1) ** 2, 0, 1) ** e
    corner = (2 * ss - 1) ** 2
    BASE = {k: scale_row(v) for k, v in M_BASE.items()}
    VOWEL = {k: {kk: scale_row(vv) for kk, vv in d.items()} for k, d in M_VOWEL.items()}
    AXM = [("ParamMouthOpenY", [0, 0.5, 1]), ("ParamMouthForm", K3)] + [(k, [0.0] + sorted(VOWEL[k])) for k in VOWEL]

    def mouth_shape(v):
        on = [VOWEL[k][float(v[k])] for k in VOWEL if v[k] > 0]
        if on:
            return np.mean(on, axis=0)
        return BASE[(float(v["ParamMouthOpenY"]), int(round(v["ParamMouthForm"])))]
    wave = np.cos(4.0 * np.pi * ss)
    corner_round = 1.0 - np.sqrt(np.clip(1.0 - (2 * ss - 1) ** 2, 0, 1))

    def mouth_curves(v):
        w, cy, dip, ou, ol, eu, el, teeth, tongue, taper, cu, cat, tl = mouth_shape(v)
        X = cxm + (x_rest - cxm) * w
        base = cl + cy + dip * Dp - cu * (corner if cu >= 0 else corner_round) - cat * wave
        return X, base + ou * prof(eu), base + ol * prof(el), teeth, tongue, w, taper
    EVEN = np.clip(4.2 * sm / np.maximum(MG.ink, 0.5), 0.75, 1.6)

    def strip(X, Y, taper=1.0):
        return MG.strip(X, Y, HW_LINE * (taper + (1 - taper) * EVEN), capf=taper + (1 - taper) * 0.15)

    def line_pos(v, lower):
        X, Yu, Yl, _, _, _, taper = mouth_curves(v)
        return strip(X, Yl if lower else Yu, taper)
    tris = MR.grid_tris
    line_rest = MG.rest()

    def lay_uv(name, fu, fv):
        lay = rig.layers[name]
        h, w = lay["img"].shape[:2]
        return np.stack([lay["x0"] + 3 + fu * (w - 6), lay["y0"] + 3 + fv * (h - 6)], 1)

    def stroke_uv():
        lay = rig.layers["mouth_stroke"]
        return MG.uv() + np.array([lay["x0"], lay["y0"]], float)
    NT = 4
    f_in = np.repeat(ss, NT), np.tile(np.linspace(0, 1, NT), NS)

    def interior_pos(v):
        X, Yu, Yl, *_ = mouth_curves(v)
        t = np.linspace(0, 1, NT)
        return np.stack([np.repeat(X, NT), (Yu[:, None] + (Yl - Yu)[:, None] * t[None, :]).reshape(-1)], 1)
    tj = json.load(open(P_.p("work", "mouth_teeth.json")))
    TEETH_H = 0.9

    def teeth_pos(v):
        X, Yu, Yl, teeth, *_ = mouth_curves(v)
        bot = Yu + 0.5 + TEETH_H * teeth / tj["band"] * np.ones_like(Yu)
        return np.stack([np.repeat(X, 2), np.stack([Yu + 0.5, bot], 1).reshape(-1)], 1)
    tprof_low = np.clip(1 - (2 * ss - 1) ** 2, 0, 1) ** 0.35

    def teeth_low_pos(v):
        X, Yu, Yl, *_ = mouth_curves(v)
        tl = mouth_shape(v)[12]
        mid = NS // 2
        gap = max(Yl[mid] - Yu[mid], 0.0)
        h_ = min(tl, 0.5 * gap) * tprof_low
        return np.stack([np.repeat(X, 2), np.stack([Yl - 0.5 - h_, Yl - 0.5 + 0 * Yl], 1).reshape(-1)], 1)
    NK = 9

    def tongue_pos(v):
        X, Yu, Yl, _, tongue, w, _ = mouth_curves(v)
        mid = NS // 2
        hgt = max(Yl[mid] - Yu[mid], 0.0)
        top = Yl[mid] - tongue * 0.48 * hgt
        bot = Yl[mid] + 6.0 * sm
        tw = 0.70 * w0 * w
        xs_ = cxm + tw * np.linspace(-1, 1, NK)
        t = np.linspace(0, 1, 3)
        return np.stack([np.repeat(xs_, 3), np.tile(top + (bot - top) * t, NK)], 1)
    defaults = rig.param_defaults
    rig.mesh("mouth_in", "mouth_in", wface, "PartFace", interior_pos(defaults), tris(NS, NT), z=zf + 0.50, axes=AXM,
             fn=lambda pts, v: interior_pos(v), uv_px=lay_uv("mouth_in", *f_in))

    def lip_op(v):
        vs = sum(v[k] for k in VOWEL)
        if vs > 0:
            return float(np.clip(v["ParamU"] / vs, 0, 1))
        return float(np.clip(-v["ParamMouthForm"], 0, 1) * np.clip(v["ParamMouthOpenY"] / 0.5, 0, 1))
    if "mouth_lipring" in rig.layers:
        rig.mesh("mouth_lipring", "mouth_lipring", wface, "PartFace", interior_pos(defaults), tris(NS, NT), z=zf + 0.505, axes=AXM,
                 fn=lambda pts, v: interior_pos(v), uv_px=lay_uv("mouth_lipring", *f_in), masks=["mouth_in"], opacity=lip_op)
    rig.mesh("mouth_tongue", "mouth_tongue", wface, "PartFace", tongue_pos(defaults), tris(NK, 3), z=zf + 0.51, axes=AXM,
             fn=lambda pts, v: tongue_pos(v), uv_px=lay_uv("mouth_tongue", np.repeat(np.linspace(0, 1, NK), 3), np.tile(np.linspace(0, 1, 3), NK)),
             masks=["mouth_in"])
    rig.mesh("mouth_teeth", "mouth_teeth", wface, "PartFace", teeth_pos(defaults), tris(NS, 2), z=zf + 0.52, axes=AXM,
             fn=lambda pts, v: teeth_pos(v), uv_px=lay_uv("mouth_teeth", np.repeat(ss, 2), np.tile(np.array([0.0, 1.0]), NS)),
             masks=["mouth_in"])
    if "mouth_teethlow" in rig.layers:
        rig.mesh("mouth_teethlow", "mouth_teethlow", wface, "PartFace", teeth_low_pos(defaults), tris(NS, 2), z=zf + 0.515, axes=AXM,
                 fn=lambda pts, v: teeth_low_pos(v), uv_px=lay_uv("mouth_teethlow", np.repeat(ss, 2), np.tile(np.array([0.0, 1.0]), NS)),
                 masks=["mouth_in"])
    rig.mesh("mouth_line_low", "mouth_stroke", wface, "PartFace", line_rest, tris(NS + 2, 3), z=zf + 0.54, axes=AXM,
             fn=lambda pts, v: line_pos(v, True), uv_px=stroke_uv())
    rig.mesh("mouth_line_up", "mouth_stroke", wface, "PartFace", line_rest, tris(NS + 2, 3), z=zf + 0.55, axes=AXM,
             fn=lambda pts, v: line_pos(v, False), uv_px=stroke_uv())


# ---------------------------------------------------------------------------------------------- the eyes (band map)
def eye_rig(rig, wface, zf, mesh):
    from scipy.ndimage import median_filter, gaussian_filter1d, grey_closing
    geom = json.load(open(P_.p("work", "eye_geom.json")))
    man = layerio.manifest()
    for s, sd in (("l", "ParamEyeLOpen"), ("r", "ParamEyeROpen")):
        if s not in geom or f"eye_{s}_lash_up" not in man:
            continue
        g = geom[s]
        gx, gu, gl = np.array(g["x"], float), np.array(g["up"], float), np.array(g["low"], float)
        upf = lambda x, gx=gx, gu=gu: np.interp(x, gx, gu)
        lowf = lambda x, gx=gx, gl=gl: np.interp(x, gx, gl)

        def col_profile(layer, fn, thr=128, main=False):
            L_, lx0, ly0 = layerio.load_layer(layer)
            a_ = L_[..., 3] > thr
            if main:
                lb_, n_ = ndimage.label(a_)
                if n_ > 1:
                    a_ = lb_ == (int(np.argmax(ndimage.sum(a_, lb_, range(1, n_ + 1)))) + 1)
            cols_ = np.nonzero(a_.any(0))[0]
            vals = np.array([fn(np.nonzero(a_[:, c])[0]) for c in cols_], float) + ly0
            return cols_ + lx0, vals

        def line_profile(layer):
            lx_, lt_ = col_profile(layer, np.min, thr=100)
            _, lb_ = col_profile(layer, np.max, thr=100)
            w_ = gaussian_filter1d(median_filter(lb_ - lt_, size=9, mode="nearest"), 4.0, mode="nearest")
            lb_ = gaussian_filter1d(median_filter(lb_, size=9, mode="nearest"), 4.0, mode="nearest")
            return lx_, lb_, np.clip(w_, 0, None)
        bx_, bv_ = col_profile(f"eye_{s}_lash_up", np.max)
        if f"eye_{s}_lash_low" in man:
            tx_, tv_ = col_profile(f"eye_{s}_lash_low", np.min)
        else:
            tx_, tv_ = gx, gl
        bv_ = median_filter(bv_, size=7, mode="nearest"); tv_ = median_filter(tv_, size=7, mode="nearest")
        xs_ = np.arange(min(bx_[0], tx_[0], gx[0]) - sz(60), max(bx_[-1], tx_[-1], gx[-1]) + sz(61))
        Tl = np.where((xs_ >= gx[0]) & (xs_ <= gx[-1]), np.interp(xs_, gx, gl), np.interp(xs_, tx_, tv_))

        def star_tip(layer):
            tx2, tt2 = col_profile(layer, np.min, main=True)
            base2 = gaussian_filter1d(grey_closing(tt2, size=int(61 * u) | 1, mode="nearest"), 3.0, mode="nearest")
            tip = gaussian_filter1d(ndimage.binary_dilation(tt2 < base2 - 3.0, iterations=6).astype(float), 2.0, mode="nearest")
            return (lambda x, a=tx2, b=base2: np.interp(x, a, b)), (lambda x, a=tx2, b=tip: np.interp(x, a, b, left=0.0, right=0.0))
        base_plain = star_tip(f"eye_{s}_lash_up")[0]
        # the upper lid as two edges (the other session's chara5, 2026-10-23): the band's top T (its lashes' spikes left
        # out, smoothed) goes to the closed line's top CT, the lid's edge R (the white's top) to the closed line's bottom
        # Mb, every point between keeps its place in the band (scaled by k). Beyond the white T still goes to CT and k
        # carries on from the white's ends, so the corners - the outer flick, the inner stroke - come down with the lid.
        # Measured column by column from the band's own lower edge beyond the white, the half-shut lash bent into a V and
        # a Z
        inside_ = (xs_ >= gx[0]) & (xs_ <= gx[-1])
        R_ = np.interp(xs_, gx, gu)
        T_ = gaussian_filter1d(base_plain(xs_), sz(12), mode="nearest")
        Tx = lambda x, xs_=xs_, T_=T_: np.interp(x, xs_, T_)
        Rx = lambda x, xs_=xs_, R_=R_: np.interp(x, xs_, R_)

        def core_line(layer):
            """the drawn closed line's core per column - the column's longest run (a hanging lash or a stray dot is a
            shorter piece) - its top and bottom smoothed along x"""
            L_, lx0, ly0 = layerio.load_layer(layer)
            a_ = L_[..., 3] > 100
            cx_, ct_, cb_ = [], [], []
            for c_ in range(a_.shape[1]):
                ys = np.nonzero(a_[:, c_])[0]
                if len(ys) == 0:
                    continue
                br = np.nonzero(np.diff(ys) > 1)[0]
                st, en = np.r_[ys[0], ys[br + 1]], np.r_[ys[br], ys[-1]]
                i_ = int(np.argmax(en - st))
                cx_.append(c_ + lx0); ct_.append(st[i_] + ly0); cb_.append(en[i_] + ly0 + 1)
            k_ = int(sz(15)) | 1
            ct_ = gaussian_filter1d(median_filter(np.array(ct_, float), size=k_, mode="nearest"), sz(3), mode="nearest")
            cb_ = gaussian_filter1d(median_filter(np.array(cb_, float), size=k_, mode="nearest"), sz(3), mode="nearest")
            return np.array(cx_, float), ct_, cb_

        def from_line(M_, CT_):
            # (the lid's edge never ends above where it starts - near the white's ends the drawn line can pass above it)
            # (smooth: the drawn closed line's underside has its lashes' bumps, and the lid's travel took them - the
            # half-shut band buckled into a W)
            Mb = gaussian_filter1d(np.maximum(M_, R_), sz(16), mode="nearest")
            k_in = np.clip((Mb - CT_) / np.maximum(R_ - T_, sz(4)), 0.12, 3.0)
            k_ = gaussian_filter1d(np.interp(xs_, xs_[inside_], k_in[inside_]), sz(16), mode="nearest")
            # (the lower lid's rise: from the white's bottom inside its columns, held beyond its tips, smoothed wide - near
            # a tip the white's bottom drops steeply, and the outline beside it stayed while the column next to it rose
            # 30 px: the half-shut eye's lower outline went wavy)
            dl_in = np.clip(Tl - Mb, 0.0, None)
            dl_ = gaussian_filter1d(np.interp(xs_, xs_[inside_], dl_in[inside_]), sz(12), mode="nearest")
            return CT_, k_, Mb, dl_

        def target(layer):
            if layer not in man:
                return None
            cx_, ct_, cb_ = core_line(layer)
            M_ = np.interp(xs_, cx_, cb_)
            # (never far below the open eye's lower lid: a stray stroke in the drawn closed eye pulled the lids down past
            # the eye - the half-shut eye's lash band slid under the iris)
            M_ = np.minimum(M_, Tl + 0.25 * np.maximum(Tl - R_, 0.0))
            CT_ = M_ - np.interp(xs_, cx_, np.clip(cb_ - ct_, sz(3), None))
            return from_line(M_, CT_)
        tc = target(f"eye_{s}_closed")
        ts = target(f"eye_{s}_closed_smile") or tc
        if tc is None:
            # (no drawn closed eye: the lids meet in the opening's middle, the band keeps 60 % of its weight)
            M_ = 0.5 * (R_ + Tl) + 0.15 * (Tl - R_)
            tc = from_line(M_, M_ - 0.6 * np.maximum(R_ - T_, 0.0))
            ts = tc
        CTc, Kc, Mc, DLc = tc
        CTs, Ks, Ms, DLs = ts

        def mix(a, b, xs_=xs_):
            return lambda x, s_, a=a, b=b: np.interp(x, xs_, a) * (1.0 - s_) + np.interp(x, xs_, b) * s_
        ctx, kx, mlx, dlx = mix(CTc, CTs), mix(Kc, Ks), mix(Mc, Ms), mix(DLc, DLs)
        O_MEET = 0.10

        def shut(o):
            return float(np.clip((1.0 - o) / (1.0 - O_MEET), 0.0, 1.0))
        sm_k = f"ParamEye{s.upper()}Smile"
        SMILE_PUSH = sz(9)

        def white_fn(pts, v, sd=sd, upf=upf, lowf=lowf, mlx=mlx, sm_k=sm_k):
            o, s_ = v[sd], v[sm_k]
            c = shut(o)
            x = pts[:, 0]
            uu, lo = upf(x), lowf(x)
            # (where the lids meet below the white - at its outer tip - the white goes down with them: kept at its own
            # bottom it showed as a sliver above the half-shut lid)
            m_ = np.maximum(mlx(x, s_), uu)
            t1, b1 = uu + (m_ - uu) * c, lo - (lo - m_) * c - SMILE_PUSH * s_ * o
            b1 = np.maximum(b1, t1 + 0.02 * (lo - uu))
            f = (pts[:, 1] - uu) / np.maximum(lo - uu, 1e-3)
            y = t1 + f * (b1 - t1)
            y = np.where(f < 0, t1 + (pts[:, 1] - uu) * o * o, y)
            return np.stack([x, y], 1)

        LATE = 0.7                     # (the band thins only after the lid has come this share of the way)

        def band_y(x, y, c, s_, Rx=Rx, kx=kx, mlx=mlx):
            """Where a point of the upper lid's band goes at closing c: its lid edge R comes down to the line the lids meet
            on (Mb) as the white's top does, the band keeping its own weight above it and thinning to the closed line's
            only at the end; what hangs below the lid edge (the outer corner's lashes wrapping down, the inner stroke) folds
            onto that line. Thinned from the start - column by column, by the closed line's weight over the band's - the
            band's strokes jammed into a zigzag at every half-shut frame (chara4, 「右目が変」「左も変」, 2026-10-23).
            (Every curve is bound as a default: the keyforms are made after the loop, and a closure over its names gave
            the left eye the right eye's curves - its lid came down 10 px.)"""
            g = float(sstep((c - LATE) / (1.0 - LATE)))
            r_ = Rx(x)
            d = y - r_
            k_ = kx(x, s_)
            return r_ + c * (mlx(x, s_) - r_) + np.where(d < 0, d * (1.0 + g * (k_ - 1.0)), d * (1.0 - g))

        def lash_fn(layer):
            def fn(pts, v, sd=sd, sm_k=sm_k, band_y=band_y):
                o, s_ = v[sd], v[sm_k]
                c = shut(o)
                x, y = pts[:, 0], pts[:, 1]
                return np.stack([x, band_y(x, y, c, s_)], 1)
            return fn

        def crease_fn(pts, v, sd=sd, sm_k=sm_k, Tx=Tx, band_y=band_y):
            # (the crease goes as the band's top below it does, and closes up toward it)
            o, s_ = v[sd], v[sm_k]
            c = shut(o)
            x, y = pts[:, 0], pts[:, 1]
            t_ = Tx(x)
            top = band_y(x, t_, c, s_)
            return np.stack([x, y + (top - t_) + 0.3 * c * np.minimum(np.clip(t_ - y, 0, None), sz(60))], 1)

        def lash_low_fn(pts, v, sd=sd, dlx=dlx, mlx=mlx, sm_k=sm_k):
            # (what of the lower lid lies above the line the lids meet on - the outline's ends round the white's tips -
            # folds onto that line as the upper lid comes down over it: left where it was, it stood up out of the
            # half-shut eye; below the line the lid rises to it)
            o, s_ = v[sd], v[sm_k]
            c = shut(o)
            x, y = pts[:, 0], pts[:, 1]
            mb = mlx(x, s_)
            above = np.clip(mb - y, 0.0, None)
            ramp = np.clip((y - mb) / sz(8), 0.0, 1.0)
            return np.stack([x, y + c * (above - dlx(x, s_) * ramp) - SMILE_PUSH * s_ * o], 1)
        icx, icy = g["iris_c"]

        def iris_fn(pts, v, icx=icx, icy=icy):
            k_ = 1.0 - 0.2 * v["ParamIrisSmall"]
            x = icx + (pts[:, 0] - icx) * k_ + sz(15) * v["ParamEyeBallX"]
            y = icy + (pts[:, 1] - icy) * k_ - sz(9) * v["ParamEyeBallY"]
            return np.stack([x, y], 1)
        EO = [(sd, [0, 0.065, 0.08, 0.1, 0.115, 0.2, 0.32, 0.45, 0.6, 0.72, 0.84, 0.905, 0.97, 1])]
        EO_O = [(sd, [0, 0.065, 0.08, 0.1, 0.115, 1])]

        def op_open(v, sd=sd):
            return float(sstep((v[sd] - 0.065) / 0.035))

        def op_closed(v, sd=sd):
            return 1.0 - float(sstep((v[sd] - 0.08) / 0.035))
        GZ = [("ParamEyeBallX", [-1, 0, 1]), ("ParamEyeBallY", [-1, 0, 1])]
        ES = [(sm_k, [0, 1])]
        IS = [("ParamIrisSmall", [0, 1])]
        cell = max(6, int(round(12 * u)))
        mesh(f"eye_{s}_white", wface, "PartEyes", z=zf + 0.1, cell=cell, axes=EO + ES, fn=white_fn, opacity=op_open)
        mesh(f"eye_{s}_iris", wface, "PartEyes", z=zf + 0.2, cell=cell, axes=GZ + EO_O + IS, fn=iris_fn, masks=[f"eye_{s}_white"], opacity=op_open)
        mesh(f"eye_{s}_lash_low", wface, "PartEyes", z=zf + 0.3, cell=max(8, int(16 * u)), axes=EO + ES, fn=lash_low_fn, opacity=op_open)
        mesh(f"eye_{s}_lash_up", wface, "PartEyes", z=zf + 0.4, cell=cell, axes=EO + ES, fn=lash_fn(f"eye_{s}_lash_up"), opacity=op_open)
        if f"eye_{s}_closed" in man:
            mesh(f"eye_{s}_closed", wface, "PartEyes", z=zf + 0.45, cell=cell, axes=EO_O + ES, fn=lambda pts, v: pts,
                 opacity=lambda v, op=op_closed, sm_k=sm_k: (1.0 - v[sm_k]) * op(v))
        if f"eye_{s}_closed_smile" in man:
            mesh(f"eye_{s}_closed_smile", wface, "PartEyes", z=zf + 0.47, cell=cell, axes=EO_O + ES, fn=lambda pts, v: pts,
                 opacity=lambda v, op=op_closed, sm_k=sm_k: v[sm_k] * op(v))
        if f"eye_{s}_crease" in man:
            mesh(f"eye_{s}_crease", wface, "PartEyes", z=zf + 0.39, cell=cell, axes=[(sd, [0, 0.1, 0.3, 0.45, 0.6, 0.75, 0.85, 1])] + ES,
                 fn=crease_fn, opacity=lambda v, sd=sd: float(sstep((v[sd] - 0.7) / 0.27)))


def blush_layer():
    """A soft pink flush over each cheek (for the shy face and the smile)."""
    from scipy.ndimage import gaussian_filter
    x0, y0 = int(HX - 1.1 * HR), int(EYE_Y + 0.15 * HR)
    w, h = int(2.2 * HR), int(0.7 * HR)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.zeros((h, w), np.float32)
    for e in (EL, ER):
        cx, cy = e[0] - x0 + (-0.05 if e[0] < HX else 0.05) * HR, e[1] - y0 + 0.4 * HR
        a = np.maximum(a, np.exp(-(((xx - cx) / (0.26 * HR)) ** 2 + ((yy - cy) / (0.11 * HR)) ** 2)))
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = 245, 110, 125
    out[..., 3] = np.clip(a * 120, 0, 255).astype(np.uint8)
    return out, x0, y0


# ---------------------------------------------------------------------------------------------- build
LOCK_GAIN = float(OPT.get("lock_gain", 1.0))


def load_locks():
    """The front hair's locks (locks.py): per lock its motion field (0 at its root, 1 at its farthest tip), the direction
    it swings in and how far its tip swings."""
    pj, pz = P_.p("work", "locks.json"), P_.p("work", "locks.npz")
    if not (os.path.exists(pj) and os.path.exists(pz)):
        return {}
    j = json.load(open(pj))
    z = np.load(pz)
    out = {}
    for n, it in j["locks"].items():
        if n in z.files:
            out[n] = {"field": z[n], "x0": j["x0"], "y0": j["y0"], "res": j["res"], "perp": np.array(it["perp"], float),
                      "amp": float(it["amp"]), "free": float(it["free"])}
    return out


def lock_weight(t, pts):
    from scipy.ndimage import map_coordinates
    return map_coordinates(t["field"], [(pts[:, 1] - t["y0"]) / t["res"], (pts[:, 0] - t["x0"]) / t["res"]], order=1, mode="nearest")


def main():
    man = layerio.manifest()
    LOCKS = {n: t for n, t in load_locks().items() if n in man}
    rig = Rig(W, H)
    P = rig.param
    G1, G2, G3 = "顔", "体", "揺れもの"
    P("ParamAngleX", -30, 30, 0, "角度 X", G1); P("ParamAngleY", -30, 30, 0, "角度 Y", G1); P("ParamAngleZ", -30, 30, 0, "角度 Z", G1)
    P("ParamEyeLOpen", 0, 1, 1, "左目 開閉", G1); P("ParamEyeROpen", 0, 1, 1, "右目 開閉", G1)
    P("ParamEyeBallX", -1, 1, 0, "目玉 X", G1); P("ParamEyeBallY", -1, 1, 0, "目玉 Y", G1)
    P("ParamMouthOpenY", 0, 1, 0, "口 開閉", G1); P("ParamMouthForm", -1, 1, 0, "口 変形", G1)
    P("ParamBrowLY", -1, 1, 0, "左眉 上下", G1); P("ParamBrowRY", -1, 1, 0, "右眉 上下", G1)
    P("ParamBrowLAngle", -1, 1, 0, "左眉 角度", G1); P("ParamBrowRAngle", -1, 1, 0, "右眉 角度", G1)
    P("ParamEyeLSmile", 0, 1, 0, "左目 笑顔", G1); P("ParamEyeRSmile", 0, 1, 0, "右目 笑顔", G1)
    P("ParamIrisSmall", 0, 1, 0, "瞳 小さく", G1)
    P("ParamCheek", 0, 1, 0, "照れ", G1)
    for n_ in LOCKS:
        P("ParamHair" + n_.replace("lock_", "Lock"), -1, 1, 0, "前髪の束 " + n_.replace("lock_", ""), "揺れもの")
    for pid_, nm_ in (("ParamA", "あ"), ("ParamI", "い"), ("ParamU", "う"), ("ParamE", "え"), ("ParamO", "お")):
        P(pid_, 0, 1, 0, "口 " + nm_, G1)
    P("ParamBodyAngleX", -10, 10, 0, "体の回転 X", G2); P("ParamBodyAngleY", -10, 10, 0, "体の回転 Y", G2); P("ParamBodyAngleZ", -10, 10, 0, "体の回転 Z", G2)
    P("ParamBreath", 0, 1, 0, "呼吸", G2)
    for pid, nm in (("ParamHairFront", "前髪 揺れ"), ("ParamHairFrontTip", "前髪 揺れ・毛先"), ("ParamHairBack", "後ろ髪 揺れ"),
                    ("ParamHairSideL", "横髪 左"), ("ParamHairSideR", "横髪 右"), ("ParamArmL", "腕 左"), ("ParamArmR", "腕 右"), ("ParamWind", "風")):
        P(pid, -1, 1, 0, nm, G3)
    for pid, nm in (("PartBody", "体"), ("PartArms", "腕"), ("PartFace", "顔"), ("PartEyes", "目"), ("PartHair", "髪")):
        rig.part(pid, nm)
    order = json.load(open(P_.p("work", "layer_order.json")))
    Zo = order["z"]
    eye_names = [n for n in man if n.startswith("eye_")]
    mouth_tex = [n for n in ("mouth_stroke", "mouth_in", "mouth_tongue", "mouth_teeth", "mouth_teethlow", "mouth_lipring") if n in man]
    brows = [n for n in ("brow_l", "brow_r") if n in man]
    names = [n for n in order["order"]] + eye_names + mouth_tex + brows
    head = set(["face", "hair_front", "hair_back", "neck"] + eye_names + mouth_tex + brows)
    for n in names:
        img, x0, y0 = layerio.load_layer(n)
        s = (0.5 if n in head else 0.35) if LOWRES else 1.0
        rig.layer(n, img, x0, y0, s)
    if OPT.get("blush", True):
        bimg, bx0, by0 = blush_layer()
        rig.layer("blush", bimg, bx0, by0, 0.5)
    K3 = [-1, 0, 1]

    def z_of(n, d=0.0):
        return float(Zo.get(n, 30)) + d

    def mesh(n, parent, part, cell=None, **kw):
        if n not in rig.layers:
            return None
        lay = rig.layers[n]
        h, w = lay["img"].shape[:2]
        c = cell or max(8, int(round((24 if max(h, w) < 300 else 40 if max(h, w) < 1200 else 72 if max(h, w) < 2500 else 110) * max(u, 0.6))))
        return rig.layer_mesh(n, parent, part, z=kw.pop("z", z_of(n)), cell=c, pad=kw.pop("pad", 4), thr=kw.pop("thr", 2), **kw)

    # ---- the body
    def body_fn(pts, v):
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        up = sstep((HIP[1] + sz(200) - y) / sz(900))
        legs = np.clip((FEET_Y - y) / max(FEET_Y - HIP[1], 1.0), 0, 1)
        bz, bx, by = v["ParamBodyAngleZ"] / 10.0, v["ParamBodyAngleX"] / 10.0, v["ParamBodyAngleY"] / 10.0
        a1 = rad(-2.0 * bz) * legs
        ox, oy = HIP[0], FEET_Y
        qx, qy = x - ox, y - oy
        x1 = ox + qx * np.cos(a1) - qy * np.sin(a1)
        y1 = oy + qx * np.sin(a1) + qy * np.cos(a1)
        a2 = rad(-4.0 * bz) * up
        qx, qy = x1 - HIP[0], y1 - HIP[1]
        x2 = HIP[0] + qx * np.cos(a2) - qy * np.sin(a2)
        y2 = HIP[1] + qx * np.sin(a2) + qy * np.cos(a2)
        th = rad(BODY_TURN * bx) * up
        dx = x2 - HIP[0]
        q = np.clip(dx / BODY_R, -1, 1)
        depth = BODY_R * np.sqrt(np.clip(1 - q * q, 0, 1))
        depth *= NECK_DEPTH + (1 - NECK_DEPTH) * sstep((y2 - (CHIN_Y + sz(10))) / sz(350))
        x2 = HIP[0] + dx * np.cos(th) + depth * np.sin(th) + BODY_SHIFT * bx * up
        y2 = y2 + sz(14) * by * up
        br = v["ParamBreath"]
        chest = sstep((y2 - (CHIN_Y + sz(60))) / sz(400)) * (1 - sstep((y2 - (SH_Y + sz(350))) / sz(500)))
        y2 = y2 - sz(10) * br * chest - sz(6) * br * up * (1 - chest)
        return np.stack([x2, y2], 1)
    wbody = rig.warp("W_Body", None, "PartBody", (0, 0, W, H), 36, 24,
                     [("ParamBodyAngleX", [-10, 0, 10]), ("ParamBodyAngleY", [-10, 0, 10]), ("ParamBodyAngleZ", [-10, 0, 10]), ("ParamBreath", [0, 1])], body_fn)
    mesh("body", wbody, "PartBody")
    # ---- the arms: turned at the shoulder (sway + breath)
    for s, sg in (("l", 1.0), ("r", -1.0)):
        if f"arm_{s}" not in rig.layers or s not in SHOULDER:
            continue
        key = f"ParamArm{s.upper()}"
        ra = rig.rot(f"R_Arm{s.upper()}", wbody, "PartArms", SHOULDER[s], [(key, K3), ("ParamBreath", [0, 1])],
                     angle=lambda v, key=key, sg=sg: sg * (5.0 * v[key] + 0.8 * v["ParamBreath"]))
        mesh(f"arm_{s}", ra, "PartArms")
    # ---- the neck: its top goes a little with the head
    def neck_fn(pts, v):
        w = sstep((CHIN_Y + sz(80) - pts[:, 1]) / sz(300))
        return np.stack([pts[:, 0] + sz(48) * v["ParamAngleX"] / 30.0 * w, pts[:, 1] - sz(6) * v["ParamAngleY"] / 30.0 * w], 1)
    mesh("neck", wbody, "PartBody", axes=[("ParamAngleX", [-30, 0, 30]), ("ParamAngleY", [-30, 0, 30])], fn=neck_fn)
    # ---- the head
    AX = [("ParamAngleX", [-30, 0, 30]), ("ParamAngleY", [-30, 0, 30]), ("ParamAngleZ", [-30, 0, 30])]
    HEADBOX = (HX - 2.6 * HR, max(0.0, CROWN[1] - 1.6 * HR), HX + 2.6 * HR, CHIN_Y + 1.1 * HR)
    face_turn_fn = turn(z_face_turn, zf_nod=z_face_plate, zf_down=z_face_down)

    def face_fn(pts, v):
        o = v["ParamMouthOpenY"]
        if o > 0:
            wy = sstep((pts[:, 1] - (MOUTH_Y - sz(6))) / sz(96))
            wx = 1.0 - sstep((np.abs(pts[:, 0] - MOUTH[0]) - sz(170)) / sz(160))
            pts = pts + np.stack([np.zeros(len(pts)), JAW_DROP * o * wy * wx], 1)
        return face_turn_fn(pts, v)
    wface = rig.warp("W_Face", wbody, "PartFace", HEADBOX, 26, 26, AX + [("ParamMouthOpenY", [0, 1])], face_fn, unfold=UNFOLD_FACE)
    zf = z_of("face")
    mesh("face", wface, "PartFace", cell=max(16, int(40 * u)))
    # (the ears, where they are layers of their own (ear_split.py: the noside plates' ears beside the face's skin), under
    # the face's edge on the face's own turn - the side of the head they sit on; a swinging side lock or a turn bares them)
    for n_ in ("ear_l", "ear_r"):
        if n_ in rig.layers:
            mesh(n_, wface, "PartFace", cell=max(8, int(16 * u)))
    if "blush" in rig.layers:
        mesh("blush", wface, "PartFace", z=zf + 0.05, cell=max(10, int(30 * u)), axes=[("ParamCheek", [0, 1])], fn=lambda pts, v: pts,
             opacity=lambda v: v["ParamCheek"])
    eye_rig(rig, wface, zf, mesh)
    for s, sd in (("l", "ParamEyeLOpen"), ("r", "ParamEyeROpen")):
        if f"brow_{s}" not in rig.layers:
            continue
        bl, bx0, by0 = layerio.load_layer(f"brow_{s}")
        bcx, bcy = bx0 + bl.shape[1] / 2.0, by0 + bl.shape[0] / 2.0
        sgn = 1.0 if s == "l" else -1.0
        S = s.upper()

        def brow_fn(pts, v, S=S, sd=sd, bcx=bcx, bcy=bcy, sgn=sgn):
            th = rad(14.0 * sgn * v[f"ParamBrow{S}Angle"])
            dx, dy = pts[:, 0] - bcx, pts[:, 1] - bcy
            x = bcx + dx * math.cos(th) - dy * math.sin(th)
            y = bcy + dx * math.sin(th) + dy * math.cos(th)
            return np.stack([x, y - sz(16) * v[f"ParamBrow{S}Y"] + sz(6) * (1.0 - v[sd])], 1)
        mesh(f"brow_{s}", wface, "PartFace", z=z_of(f"brow_{s}"), cell=max(6, int(12 * u)),
             axes=[(f"ParamBrow{S}Y", K3), (f"ParamBrow{S}Angle", K3), (sd, [0, 1])], fn=brow_fn)
    if "mouth_stroke" in rig.layers:
        mouth_rig_parts(rig, wface, zf, K3)
    # ---- the hair
    hb_ = [n for n in ("hair_back", "hair_front") if n in man]
    ys = []
    for n in hb_:
        L, x0, y0 = layerio.load_layer(n)
        ys.append(y0 + L.shape[0])
    HAIRBOX = (HX - 3.6 * HR, max(0.0, CROWN[1] - 1.6 * HR), HX + 3.6 * HR, max(ys + [CHIN_Y + 1.1 * HR]) + 0.2 * HR)
    rows = int(np.clip(26 * (HAIRBOX[3] - HAIRBOX[1]) / (HEADBOX[3] - HEADBOX[1]), 26, 40))
    FAX = [("ParamHairFront", K3), ("ParamHairFrontTip", K3)]
    SAX = [("ParamHairSideL", K3), ("ParamHairSideR", K3)]
    f_front = turn(z_hair_front, pendulum(FRONT_D, FRONT_DEG, "ParamHairFront", "ParamHairFrontTip", damp=side_damp), keep=keep_hair,
                   zf_nod=z_hair_front_nod, zf_down=lambda p: z_hair_front_nod(p, True), drop=True, rests=True)
    whf = rig.warp("W_HairFront", wbody, "PartHair", HAIRBOX, 30, rows, AX + FAX, f_front, unfold=UNFOLD)
    back_sway = pendulum(BACK_D, BACK_DEG, "ParamHairBack", "ParamHairFrontTip", damp=nape_damp)
    f_back = turn(z_hair_back, back_sway, keep=keep_hair, drop=True, rests=True)
    # (near the cut the back hair TURNS as the front hair does - no crack at a turn - but swings with its own pendulum:
    # joined to the front hair's swing too, it was dragged open with it and the page showed at the temple)
    f_back_as_front = turn(z_hair_front, back_sway, keep=keep_hair, zf_nod=z_hair_front_nod,
                           zf_down=lambda p: z_hair_front_nod(p, True), drop=True, rests=True)

    def back_fn(pts, v):
        jw = join_w(pts)[:, None]
        return jw * f_back_as_front(pts, v) + (1.0 - jw) * f_back(pts, v)
    whb = rig.warp("W_HairBack", wbody, "PartHair", HAIRBOX, 30, rows, AX + [("ParamHairBack", K3), ("ParamHairFrontTip", K3)], back_fn, unfold=UNFOLD)
    # the side hair's bob on child lattices of the same box and grid (node on node, so they hand on exactly what they are
    # given): on the turn's own lattice its two axes multiplied every keyform - a 106 MB moc3 for 31 drawables
    wsf = rig.warp("W_HairSideF", whf, "PartHair", HAIRBOX, 30, rows, SAX, lambda pts, v: side_bob(pts, v))
    wsb = rig.warp("W_HairSideB", whb, "PartHair", HAIRBOX, 30, rows, SAX, lambda pts, v: side_bob(pts, v, join=join_w(pts)))
    mesh("hair_back", wsb, "PartHair", cell=max(12, int(28 * u)))
    mesh("hair_front", wsf, "PartHair", cell=max(12, int(24 * u)))
    # the front hair's locks (locks.py): on the front hair's chain - they turn and swing with it - each also swinging
    # across itself by its own pendulum, the more the farther from its root. Two keys each: exactly linear
    for n_, t in LOCKS.items():
        pid = "ParamHair" + n_.replace("lock_", "Lock")

        def lock_fn(pts, v, t=t, pid=pid):
            d = LOCK_GAIN * t["amp"] * v[pid] * lock_weight(t, pts)
            return np.stack([pts[:, 0] + d * t["perp"][0], pts[:, 1] + d * t["perp"][1]], 1)
        mesh(n_, wsf, "PartHair", cell=max(10, int(18 * u)), axes=[(pid, [-1, 1])], fn=lock_fn)
    # ---- the hair base: under all the hair, on the head's turn and nod as the hair moves but without its swing - what a
    # swinging lock bares is hair in shadow, not the page (hair_base.py)
    if "hair_base" in rig.layers:
        f_hb_back = turn(z_hair_back, None, keep=keep_hair, drop=True)
        f_hb_front = turn(z_hair_front, None, keep=keep_hair, zf_nod=z_hair_front_nod, zf_down=lambda p: z_hair_front_nod(p, True), drop=True)

        def hair_base_fn(pts, v):
            jw = join_w(pts)[:, None]
            return jw * f_hb_front(pts, v) + (1.0 - jw) * f_hb_back(pts, v)
        whbase = rig.warp("W_HairBase", wbody, "PartHair", HAIRBOX, 24, rows, AX, hair_base_fn, unfold=UNFOLD)
        mesh("hair_base", whbase, "PartHair", cell=max(16, int(40 * u)))
    # ---- physics (the second character's, as the first's: two-stage front hair, back and side hair, arms)
    V2 = lambda r_, mob, delay, acc: [(0, 0, 1.0, 1.0, 1.0, 0), (0, r_, mob, delay, acc, r_)]
    V3 = lambda r_, mob, delay, acc: [(0, 0, 1.0, 1.0, 1.0, 0), (0, r_, mob, delay, acc, r_), (0, r_ * 2, mob * 0.95, delay * 0.9, acc, r_)]
    NP, NA = (-3, 0, 3), (-20, 0, 20)
    A = "Angle"

    def head_in(breath=0, wind=0):
        r = [("ParamAngleX", 60, "X"), ("ParamAngleZ", 60, "Angle"), ("ParamBodyAngleX", 25, "X"), ("ParamBodyAngleZ", 25, "Angle")]
        if breath:
            r.append(("ParamBreath", breath, "Angle"))
        if wind:
            r.append(("ParamWind", wind, "Angle"))
        return r

    def body_in(breath=0, wind=0, mirror=False):
        r = [("ParamBodyAngleX", 60, "X", mirror), ("ParamBodyAngleZ", 60, "Angle", mirror), ("ParamAngleX", 25, "X", mirror), ("ParamAngleZ", 20, "Angle", mirror)]
        if breath:
            r.append(("ParamBreath", breath, "Angle"))
        if wind:
            r.append(("ParamWind", wind, "Angle"))
        return r
    rig.add_physics("前髪", head_in(9, 30), [("ParamHairFront", 1, 3.6, 100, A), ("ParamHairFrontTip", 2, 3.8, 100, A)], V3(8, 0.90, 0.62, 1.2), NP, NA)
    rig.add_physics("後ろ髪", head_in(10, 34), [("ParamHairBack", 1, 3.6, 100, A)], V2(12, 0.90, 0.72, 1.0), NP, NA)
    rig.add_physics("横髪 左", head_in(12, 34), [("ParamHairSideL", 1, 3.6, 100, A)], V2(10, 0.90, 0.68, 1.15), NP, NA)
    rig.add_physics("横髪 右", head_in(12, 34), [("ParamHairSideR", 1, 3.6, 100, A, True)], V2(11, 0.89, 0.74, 1.1), NP, NA)
    for k_, (n_, t) in enumerate(LOCKS.items()):
        ln = float(np.clip(t["free"] / max(HR, 1.0), 0.5, 3.0))
        rig.add_physics("前髪の束 " + n_.replace("lock_", ""), head_in(8 + k_ % 3, 26 + 2 * (k_ % 4)),
                        [("ParamHair" + n_.replace("lock_", "Lock"), 1, 3.2, 100, A, k_ % 2 == 1)],
                        V2(6.0 + 3.0 * ln, 0.9 + 0.01 * (k_ % 4), 0.6 + 0.04 * (k_ % 3), 1.15 + 0.03 * (k_ % 2)), NP, NA)
    rig.add_physics("腕 左", body_in(0, 14), [("ParamArmL", 1, 2.4, 100, A)], V2(9, 0.7, 0.8, 1.0), NP, NA)
    rig.add_physics("腕 右", body_in(0, 14, True), [("ParamArmR", 1, 2.4, 100, A, True)], V2(9, 0.7, 0.8, 1.0), NP, NA)
    # ---- expressions (on top of tracking: Add / Multiply keep it alive; Overwrite only where a face must hold)
    BR = lambda y, a: {"ParamBrowLY": (y, "Add"), "ParamBrowRY": (y, "Add"), "ParamBrowLAngle": (a, "Add"), "ParamBrowRAngle": (a, "Add")}
    EO = lambda m: {"ParamEyeLOpen": (m, "Multiply"), "ParamEyeROpen": (m, "Multiply")}
    EX = rig.expressions
    EX["Face_Smile"] = {"params": {"ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0, "ParamEyeLSmile": 1.0, "ParamEyeRSmile": 1.0,
                                   "ParamMouthForm": 1.0, "ParamCheek": 0.55, **BR(0.3, -0.25)}, "fade": 0.2}
    EX["Face_Wink"] = {"params": {"ParamEyeROpen": (0.0, "Multiply"), "ParamEyeRSmile": 1.0, "ParamMouthForm": (0.7, "Add"), "ParamCheek": 0.35}, "fade": 0.15}
    EX["Face_Angry"] = {"params": {**BR(-0.6, 1.0), **EO(0.8), "ParamMouthForm": (-0.9, "Add")}, "fade": 0.2}
    EX["Face_Sad"] = {"params": {**BR(0.3, -1.0), **EO(0.78), "ParamMouthForm": (-0.8, "Add")}, "fade": 0.25}
    EX["Face_Surprised"] = {"params": {**BR(0.9, -0.3), "ParamIrisSmall": 1.0, "ParamMouthOpenY": 0.45, "ParamMouthForm": -0.7}, "fade": 0.12}
    EX["Face_Jitome"] = {"params": {**EO(0.5), **BR(-0.35, 0.25), "ParamMouthForm": -0.3, "ParamMouthOpenY": (0.3, "Multiply")}, "fade": 0.2}
    EX["Face_Shy"] = {"params": {"ParamCheek": 1.0, "ParamEyeBallX": -0.6, "ParamEyeBallY": -0.4, **BR(0.25, -0.5), "ParamMouthForm": (0.3, "Add")}, "fade": 0.25}
    EX["Face_Sleepy"] = {"params": {**EO(0.35), **BR(-0.15, -0.35), "ParamMouthForm": 0.0}, "fade": 0.35}
    EX["Blush"] = {"params": {"ParamCheek": 1.0}, "fade": 0.3}
    out = P_.p("out", os.environ.get("L2D_OUT", MODEL))
    shutil.rmtree(out, ignore_errors=True)
    # (a layer must fit on one atlas page with its meshes' reach: the largest scale that fits)
    for ln_, lay_ in rig.layers.items():
        ih_, iw_ = lay_["img"].shape[:2]
        ex0_, ey0_, ex1_, ey1_ = lay_["x0"], lay_["y0"], lay_["x0"] + iw_, lay_["y0"] + ih_
        for m_ in rig.meshes:
            if m_["layer"] == ln_:
                src_ = m_["uv_px"] if m_["uv_px"] is not None else m_["verts"]
                ex0_, ey0_ = min(ex0_, src_[:, 0].min()), min(ey0_, src_[:, 1].min())
                ex1_, ey1_ = max(ex1_, src_[:, 0].max()), max(ey1_, src_[:, 1].max())
        big_ = max(ex1_ - ex0_, ey1_ - ey0_) + 2
        if "hi" not in lay_ and lay_["scale"] * big_ > 4096 - 48:
            lay_["scale"] = (4096.0 - 48.0) / big_
            print("   %s はテクスチャに元の %.2f 倍の大きさで入れました（メッシュの届く範囲 %d px）" % (ln_, lay_["scale"], int(big_)))
    moc = rig.compile(out, MODEL, atlas_size=4096)
    print("組み立てました:", moc, "| 描画パーツ", len(rig.meshes), "| パラメータ", len(rig.params))


if __name__ == "__main__":
    main()
