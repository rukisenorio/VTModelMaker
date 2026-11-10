"""Where things are on this computer (macOS or Windows): VTube Studio and the Live2D Cubism Core it ships, fonts, the
Codex CLI, the torch device - and the user's settings in ~/.vtmm/config.json (written by the installer):

  {"home": "<the tool's folder>", "imagegen": "codex" | "manual", "vts_dir": ..., "vts_models_dir": ...,
   "core_lib": ..., "codex": ..., "font": ...}

Every key is optional; what is missing is found on its own. Stages import this module from the engine folder, also from a
project's own copy of the engine (<project>/kit), so the settings live in the user's home, not next to the code.
"""
import os, sys, re, glob, json, shutil, subprocess

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".vtmm", "config.json")


def config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)


def home():
    """The tool's folder (the one with install.command / install.bat), or None if unknown."""
    h = config().get("home")
    if h and os.path.isdir(h):
        return h
    here = os.path.dirname(os.path.abspath(__file__))
    up = os.path.dirname(here)
    return up if os.path.exists(os.path.join(up, "install.command")) or os.path.exists(os.path.join(up, "install.bat")) else None


# ---------------------------------------------------------------------------------------------------- VTube Studio
def _steam_dirs():
    """Steam's own folder and every library folder it lists (VTube Studio may be installed on another drive)."""
    roots = []
    if IS_MAC:
        roots.append(os.path.expanduser("~/Library/Application Support/Steam"))
    elif IS_WIN:
        try:
            import winreg
            for hive, key, val in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                                   (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                                   (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath")):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(os.path.normpath(winreg.QueryValueEx(k, val)[0]))
                except OSError:
                    pass
        except ImportError:
            pass
        for env in ("ProgramFiles(x86)", "ProgramFiles"):
            if os.environ.get(env):
                roots.append(os.path.join(os.environ[env], "Steam"))
    else:
        roots += [os.path.expanduser("~/.steam/steam"), os.path.expanduser("~/.local/share/Steam")]
    out = []
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        if r not in out:
            out.append(r)
        vdf = os.path.join(r, "steamapps", "libraryfolders.vdf")
        if os.path.exists(vdf):
            try:
                text = open(vdf, encoding="utf-8", errors="ignore").read()
            except OSError:
                text = ""
            for m in re.finditer(r'"path"\s+"([^"]+)"', text):
                p = os.path.normpath(m.group(1).replace("\\\\", "\\"))
                if os.path.isdir(p) and p not in out:
                    out.append(p)
    return out


def vts_dir():
    """VTube Studio's install folder (steamapps/common/VTube Studio), or None."""
    c = config().get("vts_dir")
    if c and os.path.isdir(c):
        return c
    for s in _steam_dirs():
        d = os.path.join(s, "steamapps", "common", "VTube Studio")
        if os.path.isdir(d):
            return d
    return None


def vts_models_dir():
    """VTube Studio's Live2DModels folder (where a model folder goes), or None."""
    c = config().get("vts_models_dir")
    if c and os.path.isdir(c):
        return c
    d = vts_dir()
    if not d:
        return None
    if IS_MAC:
        cands = [os.path.join(d, "VTubeStudio.app", "Contents", "Resources", "Data", "StreamingAssets", "Live2DModels")]
    else:
        cands = [os.path.join(d, "VTube Studio_Data", "StreamingAssets", "Live2DModels")]
    for p in cands:
        if os.path.isdir(p):
            return p
    hits = glob.glob(os.path.join(d, "**", "StreamingAssets", "Live2DModels"), recursive=True)
    return hits[0] if hits else None


def core_lib():
    """The Live2D Cubism Core library to run models with: VTMM_CORE or config "core_lib" first (e.g. the one from the
    official Cubism SDK for Native), else the one VTube Studio ships. None if there is none."""
    for c in (os.environ.get("VTMM_CORE"), config().get("core_lib")):
        if c and os.path.exists(c):
            return c
    d = vts_dir()
    if not d:
        return None
    if IS_MAC:
        pats = ["VTubeStudio.app/Contents/PlugIns/Live2DCubismCore.bundle", "**/Live2DCubismCore.bundle",
                "**/libLive2DCubismCore.dylib"]
    elif IS_WIN:
        pats = ["VTube Studio_Data/Plugins/x86_64/Live2DCubismCore.dll", "VTube Studio_Data/Plugins/Live2DCubismCore.dll",
                "**/Live2DCubismCore.dll"]
    else:
        pats = ["**/libLive2DCubismCore.so"]
    for pat in pats:
        hits = [h for h in glob.glob(os.path.join(d, pat), recursive=True) if os.path.isfile(h)]
        if hits:
            return hits[0]
    return None


def vts_running():
    """True / False - or None when it cannot be told (then treat it as running: never install into a running app)."""
    try:
        if IS_WIN:
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="ignore",
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        else:
            out = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True).stdout
    except Exception:
        return None
    if not out.strip():
        return None
    return "vtube" in out.lower()


# ---------------------------------------------------------------------------------------------------- tools, fonts
def codex_bin():
    """The Codex CLI (image generation for the plates), or None."""
    c = config().get("codex")
    if c and os.path.exists(c):
        return c
    for name in ("codex", "codex.cmd", "codex.exe"):
        p = shutil.which(name)
        if p:
            return p
    cands = ["/opt/homebrew/bin/codex", "/usr/local/bin/codex", os.path.expanduser("~/.npm-global/bin/codex"),
             os.path.expanduser("~/.local/bin/codex")]
    if IS_WIN and os.environ.get("APPDATA"):
        cands.append(os.path.join(os.environ["APPDATA"], "npm", "codex.cmd"))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def claude_bin():
    """The Claude Code CLI (the high-quality mode's eyes: landmarks, choosing the turn, reviewing the checks), or None."""
    c = config().get("claude")
    if c and os.path.exists(c):
        return c
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = shutil.which(name)
        if p:
            return p
    cands = [os.path.expanduser("~/.local/bin/claude"), os.path.expanduser("~/.claude/local/claude"), "/opt/homebrew/bin/claude",
             "/usr/local/bin/claude"]
    if IS_WIN:
        cands += [os.path.join(os.environ.get("USERPROFILE", ""), ".local", "bin", "claude.exe")]
        if os.environ.get("APPDATA"):
            cands.append(os.path.join(os.environ["APPDATA"], "npm", "claude.cmd"))
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def imagegen_backend():
    """"codex" (the plates are drawn by the Codex CLI) or "manual" (the tool writes each request - picture and prompt - to
    gen/requests/ and the person makes the picture in ChatGPT and puts it back)."""
    b = os.environ.get("VTMM_IMAGEGEN") or config().get("imagegen")
    if b in ("codex", "manual"):
        return b
    return "codex" if codex_bin() else "manual"


def ffmpeg_bin():
    return config().get("ffmpeg") or shutil.which("ffmpeg")


def font_path(jp=True):
    c = config().get("font")
    cands = [c] if c else []
    if IS_MAC:
        cands += ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                  "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                  "/System/Library/Fonts/Helvetica.ttc"]
    elif IS_WIN:
        w = os.environ.get("WINDIR", r"C:\Windows")
        cands += [os.path.join(w, "Fonts", f) for f in ("meiryo.ttc", "YuGothM.ttc", "YuGothR.ttc", "msgothic.ttc", "arial.ttf")]
    else:
        cands += glob.glob("/usr/share/fonts/**/NotoSansCJK*.tt*", recursive=True) + glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    for f in cands:
        if f and os.path.exists(f):
            return f
    return None


def font(size, jp=True):
    from PIL import ImageFont
    f = font_path(jp)
    if f:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_path(p):
    """Show a file or folder in Finder / Explorer."""
    try:
        if IS_MAC:
            subprocess.Popen(["open", p])
        elif IS_WIN:
            os.startfile(p)
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception:
        pass


def summary():
    """What was found, for the installer, the app and the skill."""
    return {"os": "mac" if IS_MAC else "windows" if IS_WIN else sys.platform, "python": sys.version.split()[0],
            "home": home(), "config": CONFIG_PATH, "vts_dir": vts_dir(), "vts_models_dir": vts_models_dir(),
            "core_lib": core_lib(), "codex": codex_bin(), "claude": claude_bin(), "imagegen": imagegen_backend(), "ffmpeg": ffmpeg_bin(),
            "font": font_path()}


if __name__ == "__main__":
    JA = {"os": "OS", "python": "Python", "home": "道具のフォルダ", "config": "設定ファイル", "vts_dir": "VTube Studio のフォルダ",
          "vts_models_dir": "VTube Studio のモデルのフォルダ", "core_lib": "Live2D Cubism Core", "codex": "Codex CLI",
          "claude": "Claude Code", "imagegen": "下絵の作り方", "ffmpeg": "ffmpeg", "font": "文字のフォント"}
    IG = {"codex": "自動（Codex）", "manual": "手作業（ChatGPT）"}
    for k, v in summary().items():
        if k == "imagegen":
            v = IG.get(v, v)
        print(f"{JA.get(k, k)}: {'見つかりません' if v is None else v}")
