"""Stage 6: VTube Studio settings for the build (tracking, hotkeys, icon, idle motion), the check, and the install.

usage: package.py <project> [--install]

out/<name>/ gets icon.png, <name>.vtube.json (face tracking mapped to the model's parameters - vowels too, for the
Advanced Lipsync; one hotkey per expression on the number keys, 0 = all off), animations/<name>_idle.motion3.json (a
light wind for the hair's physics). Then check(): every file the settings and model3 name exists, every parameter that
physics, expressions, motions and tracking use is in the moc, the Core gives finite vertices for random poses.
--install copies it to VTube Studio's Live2DModels/<vts_folder> (character.json "vts_folder", default the name) - only
while VTube Studio is NOT running; an existing settings file there is merged into (only what is new is added, a copy
goes to backups/ first), never overwritten; texture pages the build no longer has are moved to backups/, never deleted.
The ModelID and hotkey ids are kept across rebuilds (out/<name>/<name>.vtube.json of the last run).
"""
import os, sys, json, math, time, uuid, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import common as C

import platform_env
VTS_MODELS = platform_env.vts_models_dir()      # (macOS or Windows; None when VTube Studio is not installed)
TAU = 2 * math.pi
DUR, STEP = 16.0, 0.05
EXPR_KEYS = [("Face_Smile", "表情：にっこり"), ("Face_Wink", "表情：ウインク"), ("Face_Angry", "表情：怒り"), ("Face_Sad", "表情：悲しみ"),
             ("Face_Surprised", "表情：驚き"), ("Face_Jitome", "表情：ジト目"), ("Face_Shy", "表情：照れ"), ("Face_Sleepy", "表情：ねむい"),
             ("Blush", "頬を赤らめる")]


def idle_curves(t):
    return {"ParamWind": 0.32 * math.sin(TAU * 2 * t / DUR) + 0.22 * math.sin(TAU * 5 * t / DUR + 1.3) + 0.12 * math.sin(TAU * 9 * t / DUR + 0.5)}


def build_idle(param_ids):
    n = int(round(DUR / STEP))
    ts = [round(i * STEP, 4) for i in range(n + 1)]
    out = []
    for pid in sorted(idle_curves(0.0)):
        if pid not in param_ids:
            continue
        seg = []
        for j, t in enumerate(ts):
            val = round(idle_curves(t if j < n else 0.0)[pid], 4)
            seg += [t, val] if j == 0 else [0, t, val]
        out.append({"Target": "Parameter", "Id": pid, "Segments": seg})
    segs = sum((len(c["Segments"]) - 2) // 3 for c in out)
    meta = {"Duration": DUR, "Fps": 30.0, "Loop": True, "AreBeziersRestricted": True, "CurveCount": len(out),
            "TotalSegmentCount": segs, "TotalPointCount": segs + len(out), "UserDataCount": 0, "TotalUserDataSize": 0}
    return {"Version": 3, "Meta": meta, "Curves": out}


def ps(name, inp, lo_i, hi_i, out, lo_o, hi_o, smooth=15, clamp=False, breathing=False):
    return {"Folder": "", "Name": name, "Input": inp, "InputRangeLower": float(lo_i), "InputRangeUpper": float(hi_i),
            "OutputRangeLower": float(lo_o), "OutputRangeUpper": float(hi_o), "ClampInput": clamp, "ClampOutput": clamp,
            "UseBlinking": False, "UseBreathing": breathing, "OutputLive2D": out, "Smoothing": smooth, "Minimized": False}


def hotkey(name, action, file, key, button=-1):
    return {"HotkeyID": uuid.uuid4().hex, "Name": name, "Action": action, "File": file, "Folder": "", "AutoCreated": False,
            "Position": {"X": 0.0, "Y": 0.0, "Z": 0.0, "Rotation": 0.0},
            "ColorOverlay": {"On": False, "IsStaticColor": False, "Display": -1, "WindowName": "", "IncludeLeft": False, "IncludeMid": False,
                             "IncludeRight": False, "BaseValue": 0, "OverlayValue": 0, "Smoothing": 0, "IncludeItems": False,
                             "StaticColor": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}},
            "ColorScreenMultiplyPreset": {"ArtMeshMultiplyAndScreenColors": []},
            "HandGestureSettings": {"GestureLeft": "", "GestureRight": "", "GestureCombinator": "AND", "AllowMirroredGesture": False,
                                    "DeactivateExpWhenGestureNotDetected": False, "SecondsUntilDetection": 0.5, "SecondsDetected": 0.0, "PercentDetected": 0.0},
            "LoadModelSettings": {"LoadAtCurrentPosition": True},
            "Triggers": {"Trigger1": key, "Trigger2": "", "Trigger3": "", "ScreenButton": button},
            "SoundConfig": {"SoundActive": False, "StopWhenInactive": False, "StopFadeOutSeconds": 2.0, "SoundSlots": []},
            "IsGlobal": False, "IsActive": True, "Minimized": False, "StopsOnLastFrame": False, "DeactivateAfterKeyUp": False,
            "OnlyLoadOneRandomItem": False, "DeactivateAfterSeconds": False, "DeactivateAfterSecondsAmount": 10.0, "FadeSecondsAmount": 0.1,
            "OnScreenHotkeyColor": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}}


def hotkeys(src):
    have = {f[:-len(".exp3.json")] for f in os.listdir(os.path.join(src, "expressions"))} if os.path.isdir(os.path.join(src, "expressions")) else set()
    out = [hotkey("表情を全解除（普通の顔に戻す）", "RemoveAllExpressions", "", "N0", button=1)]
    keys = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9"]
    k = 0
    for e, nm in EXPR_KEYS:
        if e in have:
            out.append(hotkey(nm + " ON/OFF", "ToggleExpression", e + ".exp3.json", keys[k] if k < len(keys) else "", button=k + 2 if k < 7 else -1))
            k += 1
    return out


def vtube_json(P, src, param_ids):
    name = P.name
    W, H = P.canvas_size()
    HX, HY, HR = P.head()
    # (the bust in view: the head about as large as the earlier characters' were - their hand-set values fit
    # scale ~ 0.136 H / HR and the bust's middle raised by ~283 units per (0.5 - its height fraction) and scale)
    scale = float(np.clip(0.136 * H / HR, 0.8, 6.0))
    frac = (HY + 1.2 * HR) / H
    y = float(-283.5 * (0.5 - frac) * scale)
    params = [
        ps("顔の左右の向き", "FaceAngleX", -30, 30, "ParamAngleX", -30, 30, 15),
        ps("顔の上下の向き", "FaceAngleY", -20, 20, "ParamAngleY", -30, 30, 15),
        ps("顔の傾き", "FaceAngleZ", -30, 30, "ParamAngleZ", -30, 30, 20),
        ps("左目の開閉", "EyeOpenLeft", 0, 1, "ParamEyeLOpen", 0, 1, 8, clamp=True),
        ps("右目の開閉", "EyeOpenRight", 0, 1, "ParamEyeROpen", 0, 1, 8, clamp=True),
        ps("視線（左右）", "EyeRightX", -1, 1, "ParamEyeBallX", 1, -1, 12, clamp=True),
        ps("視線（上下）", "EyeRightY", -1, 1, "ParamEyeBallY", -1, 1, 12, clamp=True),
        ps("左眉の上下", "Brows", 0, 1, "ParamBrowLY", -1, 1, 12, clamp=True),
        ps("右眉の上下", "Brows", 0, 1, "ParamBrowRY", -1, 1, 12, clamp=True),
        ps("口の笑顔", "MouthSmile", 0, 1, "ParamMouthForm", -1, 1, 6, clamp=True),
        ps("左目の笑顔", "MouthSmile", 0.55, 1, "ParamEyeLSmile", 0, 1, 14, clamp=True),
        ps("右目の笑顔", "MouthSmile", 0.55, 1, "ParamEyeRSmile", 0, 1, 14, clamp=True),
        ps("頬の赤み", "MouthSmile", 0.6, 1, "ParamCheek", 0, 0.6, 30, clamp=True),
        # (VoiceVolumePlusMouthOpen: the mouth opens with the camera AND with the microphone - a mic-only user's
        # mouth stayed shut with MouthOpen alone)
        ps("口の開閉（声でも開く）", "VoiceVolumePlusMouthOpen", 0, 1, "ParamMouthOpenY", 0, 1, 4, clamp=True),
        ps("母音 あ", "VoiceA", 0, 1, "ParamA", 0, 1, 8, clamp=True),
        ps("母音 い", "VoiceI", 0, 1, "ParamI", 0, 1, 8, clamp=True),
        ps("母音 う", "VoiceU", 0, 1, "ParamU", 0, 1, 8, clamp=True),
        ps("母音 え", "VoiceE", 0, 1, "ParamE", 0, 1, 8, clamp=True),
        ps("母音 お", "VoiceO", 0, 1, "ParamO", 0, 1, 8, clamp=True),
        ps("体の回転 X", "FaceAngleX", -30, 30, "ParamBodyAngleX", -10, 10, 28),
        ps("体の回転 Y", "FaceAngleY", -30, 30, "ParamBodyAngleY", -10, 10, 28),
        ps("体の回転 Z", "FaceAngleZ", -30, 30, "ParamBodyAngleZ", -10, 10, 28),
        ps("呼吸（自動）", "", 0, 1, "ParamBreath", 0, 1, 0, breathing=True),
    ]
    params = [p for p in params if p["OutputLive2D"] in param_ids]
    idle = f"{name}_idle.motion3.json"
    return {
        "Version": 1, "Name": name, "ModelID": uuid.uuid4().hex,
        "FileReferences": {"Icon": "icon.png", "Model": f"{name}.model3.json", "IdleAnimation": idle, "IdleAnimationWhenTrackingLost": ""},
        "ModelSaveMetadata": {"LastSavedVTubeStudioVersion": "1.35.10", "LastSavedPlatform": "Steam",
                              "LastSavedDateUTC": time.strftime("%A, %d %B %Y, %H:%M:%S", time.gmtime()),
                              "LastSavedDateLocalTime": time.strftime("%A, %d %B %Y, %H:%M:%S"),
                              "LastSavedDateUnixMillisecondTimestamp": str(int(time.time() * 1000)), "TotalSaveCount": 1},
        "SavedModelPosition": {"Position": {"x": 0.0, "y": round(y, 1), "z": 0.0}, "Rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                               "Scale": {"x": round(scale, 2), "y": round(scale, 2), "z": 1.0}},
        "ModelPositionMovement": {"Use": False, "X": 6, "Y": 8, "Z": 11, "SmoothingX": 10, "SmoothingY": 10, "SmoothingZ": 10},
        "ItemSettings": {"OnlyMoveWhenPinned": False, "AllowNormalHotkeyTriggers": True, "Multiplier_HeadAngleX": 1.0, "Multiplier_HeadAngleY": 1.0,
                         "Multiplier_HeadAngleZ": 1.0, "Shift_HeadAngleX": 0.0, "Shift_HeadAngleY": 0.0, "Smoothing_HeadAngleX": 15.0,
                         "Smoothing_HeadAngleY": 15.0, "Smoothing_HeadAngleZ": 15.0},
        "PhysicsSettings": {"Use": True, "UseLegacyPhysics": False, "Live2DPhysicsFPS": 3, "PhysicsStrength": 50, "WindStrength": 0, "DraggingPhysicsStrength": 10},
        "GeneralSettings": {"ModelVolume": 90, "TimeUntilTrackingLostIdleAnimation": 0.0, "WorkshopSharingForbidden": True, "EnableExpressionSaving": True},
        "ParameterSettings": params,
        "Hotkeys": hotkeys(src),
        "HotkeySettings": {"UseOnScreenHotkeys": False, "UseKeyboardHotkeys": True, "SendOnScreenHotkeysToPC": True, "OnScreenHotkeyAlpha": 76},
        "ArtMeshDetails": {"ArtMeshesExcludedFromPinning": [], "ArtMeshesThatDeleteItemsOnDrop": [], "ArtMeshSceneLightingMultipliers": [],
                           "ArtMeshMultiplyAndScreenColors": [], "ModelArtMeshGroups": []},
        "ParameterCustomization": {"ParametersExcludedFromVNetSmoothing": [], "SoundTriggers": []},
        "PhysicsCustomizationSettings": {"PhysicsMultipliersPerPhysicsGroup": [], "WindMultipliersPerPhysicsGroup": [], "DraggingPhysicsMultipliersPerPhysicsGroup": []},
        "FolderInfo": {"HotkeyFolders": [], "ConfigItemFolders": []},
        "UserCustomization": {"PreferredLanguageSetBy": "", "PreferredLanguage": "", "AvailableLanguages": [], "UserParamFolderInfo": [], "UserParamInfo": []},
        "SavedActiveExpressions": [], "ModelFolder": "<VTS_MODEL_BASE_PATH_PLACEHOLDER>",
    }


def check(P, src):
    from l2d import core
    name = P.name
    mj = json.load(open(os.path.join(src, f"{name}.model3.json"), encoding="utf-8"))
    vt = json.load(open(os.path.join(src, f"{name}.vtube.json"), encoding="utf-8"))
    refs = mj["FileReferences"]
    files = [refs["Moc"], refs.get("Physics"), refs["DisplayInfo"]] + refs["Textures"] + [e["File"] for e in refs.get("Expressions", [])]
    files += [vt["FileReferences"]["Icon"], vt["FileReferences"]["Model"], os.path.join("animations", vt["FileReferences"]["IdleAnimation"])]
    missing = [f for f in files if f and not os.path.exists(os.path.join(src, f))]
    m = core.Model(os.path.join(src, refs["Moc"]))
    pids = set(m.param_ids)
    used = {}
    if refs.get("Physics"):
        for s_ in json.load(open(os.path.join(src, refs["Physics"]), encoding="utf-8"))["PhysicsSettings"]:
            for i in s_["Input"]:
                used.setdefault(i["Source"]["Id"], "物理演算の入力")
            for o in s_["Output"]:
                used.setdefault(o["Destination"]["Id"], "物理演算の出力")
    for e in refs.get("Expressions", []):
        for p in json.load(open(os.path.join(src, e["File"]), encoding="utf-8"))["Parameters"]:
            used.setdefault(p["Id"], "表情 " + e["Name"])
    for c in json.load(open(os.path.join(src, "animations", vt["FileReferences"]["IdleAnimation"]), encoding="utf-8"))["Curves"]:
        used.setdefault(c["Id"], "待機モーション")
    for p in vt["ParameterSettings"]:
        used.setdefault(p["OutputLive2D"], "トラッキング " + p["Name"])
    unknown = sorted((k, v) for k, v in used.items() if k not in pids)
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(40):
        m.reset()
        m.set(**{pid: float(rng.uniform(m.param_min[i], m.param_max[i])) for i, pid in enumerate(m.param_ids)})
        m.update()
        for d in range(len(m.drawable_ids)):
            if not np.isfinite(m.positions(d)).all():
                bad += 1
    ok = not missing and not unknown and bad == 0
    print("チェック:", "問題なし" if ok else "問題あり", "| 足りないファイル:", missing, "| モデルにないパラメータ:", unknown, "| 値がおかしい頂点:", bad)
    return ok


def merge_settings(P, src_file, dst_file):
    new, cur = json.load(open(src_file, encoding="utf-8")), json.load(open(dst_file, encoding="utf-8"))
    os.makedirs(P.p("backups"), exist_ok=True)
    shutil.copy2(dst_file, P.p("backups", time.strftime(f"{P.name}.vtube.%Y%m%d_%H%M%S.json")))
    have_hk = {(h["Action"], h["File"]) for h in cur.get("Hotkeys", [])}
    btn = {h["Triggers"].get("ScreenButton", -1) for h in cur.get("Hotkeys", [])}
    added = []
    for h in new["Hotkeys"]:
        if (h["Action"], h["File"]) not in have_hk:
            if h["Triggers"].get("ScreenButton", -1) in btn:
                h["Triggers"]["ScreenButton"] = -1
            cur.setdefault("Hotkeys", []).append(h); added.append(h["Name"])
    have_ps = {p["OutputLive2D"] for p in cur.get("ParameterSettings", [])}
    for p in new["ParameterSettings"]:
        if p["OutputLive2D"] not in have_ps:
            cur["ParameterSettings"].append(p); added.append(p["OutputLive2D"])
    refs = cur.setdefault("FileReferences", {})
    if not refs.get("IdleAnimation"):
        refs["IdleAnimation"] = new["FileReferences"]["IdleAnimation"]; added.append("待機アニメーション")
    json.dump(cur, open(dst_file, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("インストール済みの設定に、新しい分だけ加えました:", added)


def vts_running():
    """True when VTube Studio runs - and also when that cannot be told (never install into a running app)."""
    r = platform_env.vts_running()
    return True if r is None else r


def main():
    P = C.open_project()
    from l2d import core
    name = P.name
    src = P.p("out", name)
    mj = json.load(open(os.path.join(src, f"{name}.model3.json"), encoding="utf-8"))
    param_ids = set(core.Model(os.path.join(src, mj["FileReferences"]["Moc"])).param_ids)
    os.makedirs(os.path.join(src, "animations"), exist_ok=True)
    json.dump(build_idle(param_ids), open(os.path.join(src, "animations", f"{name}_idle.motion3.json"), "w", encoding="utf-8"), separators=(",", ":"))
    # the icon: the head of the picture
    HX, HY, HR = P.head()
    can = Image.open(P.p("work", "figure.png")).convert("RGBA")
    box = (int(HX - 1.7 * HR), int(HY - 1.5 * HR), int(HX + 1.7 * HR), int(HY + 1.9 * HR))
    ic = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (236, 232, 240, 255))
    ic.alpha_composite(can.crop(box))
    ic.convert("RGB").resize((512, 512), Image.LANCZOS).save(os.path.join(src, "icon.png"))
    vt = os.path.join(src, f"{name}.vtube.json")
    keep = P.p("work", f"{name}.vtube.ids.json")          # (ids of the last package: the build wipes out/<name>)
    new = vtube_json(P, src, param_ids)
    old = json.load(open(keep, encoding="utf-8")) if os.path.exists(keep) else (json.load(open(vt, encoding="utf-8")) if os.path.exists(vt) else None)
    if old:
        new["ModelID"] = old["ModelID"]
        ids = {(h["Action"], h["File"]): h["HotkeyID"] for h in old.get("Hotkeys", [])}
        for h in new["Hotkeys"]:
            h["HotkeyID"] = ids.get((h["Action"], h["File"]), h["HotkeyID"])
    json.dump(new, open(vt, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"ModelID": new["ModelID"], "Hotkeys": [{"Action": h["Action"], "File": h["File"], "HotkeyID": h["HotkeyID"]} for h in new["Hotkeys"]]},
              open(keep, "w", encoding="utf-8"), indent=1)
    print("VTube Studio 用の設定を書きました:", sorted(os.listdir(src)))
    if not check(P, src):
        print("インストールしませんでした: 先にチェックで見つかった問題を直してください")
        return
    if "--install" not in sys.argv:
        return
    if vts_running():
        print("VTube Studio が起動中なので、インストールしませんでした（終了してからインストールしてください）")
        return
    dst = os.path.join(VTS_MODELS, P.cfg.get("vts_folder", name))
    if os.path.isdir(dst):
        bk = P.p("backups", time.strftime("installed_%Y%m%d_%H%M%S"))
        shutil.copytree(dst, bk)
        print("前にインストールしたものを残しました:", bk)
    os.makedirs(dst, exist_ok=True)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s_, d_ = os.path.join(root, f), os.path.join(dst, rel, f)
            if f.endswith(".vtube.json") and os.path.exists(d_):
                merge_settings(P, s_, d_)
                continue
            shutil.copy2(s_, d_)
    tdir = f"{name}.4096"
    if os.path.isdir(os.path.join(dst, tdir)):
        stale = sorted(f for f in os.listdir(os.path.join(dst, tdir)) if f.startswith("texture_") and not os.path.exists(os.path.join(src, tdir, f)))
        if stale:
            keepd = P.p("backups", time.strftime("replaced_textures_%Y%m%d_%H%M%S"))
            os.makedirs(keepd, exist_ok=True)
            for f in stale:
                shutil.move(os.path.join(dst, tdir, f), os.path.join(keepd, f))
            print("使わなくなったテクスチャを移しました:", stale)
    print("VTube Studio のフォルダにインストールしました:", os.path.basename(dst))


if __name__ == "__main__":
    main()
