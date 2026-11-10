"""画像生成：Codex CLI（draw / edit / style / ref）と Antigravity CLI（画素の位置を保った編集）。手作業モードでは、
同じ依頼を gen/requests/<名前>/ に書いて、人が ChatGPT で作った絵を待つ。

どの呼び出しも <プロジェクト>/gen/out/<名前>.png とログ <プロジェクト>/gen/logs/<名前>.log を書き、出来上がりがもうあれば
何もしない（作り直すには消すか名前を変える）。入力は先に gen/in/ に写すので、下絵はいつでも作ったときの絵そのものに
重ねられる。下絵ができたあとで入力を作り直さないこと。

  draw  <プロジェクト> <名前> <指示ファイル> [縦横比]         参照なしの新しい絵（テスト用のキャラなど）
  edit  <プロジェクト> <名前> <入力png> <指示ファイル>        入力の構図を保つ Codex の編集（1〜2px ずれる: 重ね直す）
  style <プロジェクト> <名前> <見本png> <指示ファイル>        見本の絵柄で新しい絵（記号、シート）
  ref   <プロジェクト> <名前> <キャラpng> <指示ファイル>      同じキャラの描き直し（斜め向き・見上げ・見下ろしの見本）
  agy   <プロジェクト> <名前> <入力png> <指示ファイル>        Antigravity の編集。入力と同じ画素の格子のまま（回数が少ない！）

経験から: Codex の安全機能に断られた依頼は、それで終わり。言い換えて頼み直さず、別の素材（反対側を反転したもの、別の
下絵）を使う。agy の画像は5〜7回ほどで使い切り、数時間から数日戻らないことがある。いちばん大事な編集から先に使い、
Codex を控えにする。
"""
import os, sys, shutil, subprocess, time, re, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_env

CODEX = platform_env.codex_bin()
AGY = os.path.expanduser("~/.local/bin/agy")


class PlateRequested(Exception):
    """Manual mode: the request (picture + prompt) is in gen/requests/<name>/ and waits for the person's picture."""


# (manual mode: the same requests, worded for a person pasting them into ChatGPT with the input picture attached)
MANUAL = {
    "draw": "Draw a new image.\n\nWhat to draw:\n{instr}\n\nMake it a {aspect} image at the highest quality. Generate exactly one image.",
    "edit": ("Edit the attached image. This is a production asset for Live2D rigging, so the result must stay pixel-aligned "
             "with the input: same framing, same scale, every part that is not to change exactly where it was.\n\n"
             "Edit instruction:\n{instr}\n\nKeep the input's aspect ratio ({aspect}) at the highest quality. Generate exactly "
             "one image."),
    "style": ("Draw a new image. The attached image is a STYLE REFERENCE only: match its line weight, outline colour and flat "
              "anime cel colouring, but do not copy its content.\n\nWhat to draw:\n{instr}\n\nMake it a {aspect} image at "
              "the highest quality. Generate exactly one image."),
    "ref": ("Draw a new image of the SAME CHARACTER as in the attached image: keep the design exactly - the same face, eyes, "
            "hair, colours, outlines and colouring - and draw what is asked below.\n\nWhat to draw:\n{instr}\n\nMake it a "
            "{aspect} image at the highest quality. Generate exactly one image."),
}
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def request_dir(proj, name):
    return os.path.join(proj, "gen", "requests", name)


def adopt_result(proj, name):
    """A picture the person put into gen/requests/<name>/ (any name but input.png) becomes gen/out/<name>.png."""
    d = request_dir(proj, name)
    out = os.path.join(proj, "gen", "out", name + ".png")
    if not os.path.isdir(d):
        return None
    cands = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and f.lower() != "input.png"]
    if not cands:
        return None
    from PIL import Image
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.open(max(cands, key=os.path.getmtime)).convert("RGB").save(out)
    return out


def manual_request(proj, kind, name, instr, rel_input, aspect):
    d = request_dir(proj, name)
    os.makedirs(d, exist_ok=True)
    if rel_input:
        shutil.copy2(os.path.join(proj, "gen", rel_input), os.path.join(d, "input.png"))
    text = MANUAL[kind].format(instr=instr, aspect=aspect) + "\n\n（文章で返事をするときは日本語でお願いします。）"
    with open(os.path.join(d, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(d, "request.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "kind": kind, "aspect": aspect, "has_input": bool(rel_input), "time": time.time()}, f, indent=1)
    raise PlateRequested(name)

_TAIL = ("After the image is generated, copy the generated PNG file to out/{name}.png inside the current working directory. "
         "Do not modify any input. Then reply with exactly one line: the output path and its pixel dimensions.")

PROMPTS = {
    "draw": ("Use your built-in image generation tool to DRAW A NEW IMAGE.\n\nWhat to draw:\n\"{instr}\"\n\n"
             "Request a {aspect} image at the highest quality available. Generate exactly one image.\n\n" + _TAIL),
    "edit": ("Use your built-in image generation tool to EDIT the attached image ({input}). This is a production asset for "
             "Live2D rigging, so the result must stay pixel-aligned with the input: same framing, same scale, every part that "
             "is not to change exactly where it was.\n\nEdit instruction for the image tool:\n\"{instr}\"\n\n"
             "Request the output at the same aspect ratio as the input ({aspect}) at the highest quality available. Generate "
             "exactly one image.\n\n" + _TAIL),
    "style": ("Use your built-in image generation tool to DRAW A NEW IMAGE. The attached image ({input}) is a STYLE REFERENCE "
              "only: match its line weight, outline colour and flat anime cel colouring, but do not copy its content.\n\n"
              "What to draw:\n\"{instr}\"\n\nRequest a {aspect} image at the highest quality available. Generate exactly one "
              "image.\n\n" + _TAIL),
    "ref": ("Use your built-in image generation tool to DRAW A NEW IMAGE of the SAME CHARACTER as in the attached image "
            "({input}). The attached image is the character reference: keep the design exactly - the same face, eyes, hair, "
            "colours, the same outlines and colouring, the same line weight - and draw what is asked below. This is a "
            "production reference for Live2D rigging.\n\nWhat to draw:\n\"{instr}\"\n\nRequest a {aspect} image at the "
            "highest quality available. Generate exactly one image.\n\n" + _TAIL),
}


def _aspect(path):
    from PIL import Image
    w, h = Image.open(path).size
    if abs(w - h) < 0.05 * max(w, h):
        return "square"
    return "portrait 2:3" if h > w else "landscape 3:2"


def codex(proj, kind, name, instr, input_png=None, aspect=None):
    gen = os.path.join(proj, "gen")
    os.makedirs(os.path.join(gen, "out"), exist_ok=True); os.makedirs(os.path.join(gen, "logs"), exist_ok=True)
    os.makedirs(os.path.join(gen, "in"), exist_ok=True)
    out = os.path.join(gen, "out", name + ".png")
    if os.path.exists(out):
        return out
    if adopt_result(proj, name):
        return out
    rel = None
    if input_png is not None:
        rel = os.path.join("in", name + "_in.png")
        if os.path.abspath(input_png) != os.path.abspath(os.path.join(gen, rel)):
            shutil.copy2(input_png, os.path.join(gen, rel))
    aspect = aspect or (_aspect(os.path.join(gen, rel)) if rel else "portrait 2:3")
    text = instr if not os.path.exists(str(instr)) else open(instr, encoding="utf-8").read().strip()
    if platform_env.imagegen_backend() == "manual":
        manual_request(proj, kind, name, text, rel, aspect)
    if not CODEX:
        raise RuntimeError("Codex CLI が見つかりません。手作業モードにするには、画面の「下絵」で「手作業（ChatGPT）」を選ぶか、"
                           + platform_env.CONFIG_PATH + " に \"imagegen\": \"manual\" と書いてください")
    prompt = PROMPTS[kind].format(instr=text, input=rel, aspect=aspect, name=name)
    with open(os.path.join(gen, "logs", name + ".prompt.txt"), "w", encoding="utf-8") as f_:
        f_.write(prompt)
    cmd = [CODEX, "exec", "-C", gen, "--skip-git-repo-check", "--sandbox", "workspace-write", "-c", 'model_reasoning_effort="low"']
    if rel:
        cmd += ["-i", rel]
    cmd += ["-"]
    t0 = time.time()
    with open(os.path.join(gen, "logs", name + ".log"), "w", encoding="utf-8", errors="replace") as lg:
        subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", cwd=gen, stdout=lg, stderr=subprocess.STDOUT)
    if not os.path.exists(out):
        # (Codex sometimes leaves the picture in its own folder only: take the newest one made during the call)
        cands = [p for p in glob.glob(os.path.expanduser("~/.codex/generated_images/*/*.png")) if os.path.getmtime(p) >= t0 - 1]
        if cands:
            shutil.copy2(max(cands, key=os.path.getmtime), out)
    if not os.path.exists(out):
        log = open(os.path.join(gen, "logs", name + ".log"), encoding="utf-8", errors="replace").read()[-1500:]
        raise RuntimeError(f"{name}: Codex が絵を返しませんでした（断られたか、失敗）。断られたときは言い換えて頼み直さず、"
                           f"別の下絵で代えるか省いてください:\n{log}")
    return out


def agy(proj, name, input_png, instr, tries=3):
    """Antigravity edit on the input's own pixel grid -> gen/out/<name>.png (converted from its JPEG)."""
    from PIL import Image
    gen = os.path.join(proj, "gen")
    out = os.path.join(gen, "out", name + ".png")
    if os.path.exists(out):
        return out
    work = os.path.join(gen, "agy_work", name)
    os.makedirs(work, exist_ok=True); os.makedirs(os.path.join(gen, "logs"), exist_ok=True)
    shutil.copy2(input_png, os.path.join(work, "input.png"))
    os.makedirs(os.path.join(gen, "in"), exist_ok=True)
    shutil.copy2(input_png, os.path.join(gen, "in", name + "_in.png"))
    text = instr if not os.path.exists(str(instr)) else open(instr).read().strip()
    prompt = ("Edit the image @input.png with your image tool, exactly once. Do not run shell commands, do not write files, only call "
              "the image tool once and report the path of the image it made.\n\nEdit: " + text +
              "\nKeep everything else exactly as it is, at the same pixel positions and the same size.")
    for k in range(tries):
        r = subprocess.run([AGY, "-p", prompt, "--sandbox", "--print-timeout", "10m0s"], cwd=work, capture_output=True, text=True)
        log = (r.stdout or "") + (r.stderr or "")
        open(os.path.join(gen, "logs", name + ".agy.log"), "a").write(log)
        m = re.findall(r"(/[^\s'\"]*antigravity-cli/brain/[^\s'\"]+\.(?:jpg|jpeg|png))", log)
        if m and os.path.exists(m[-1]):
            Image.open(m[-1]).convert("RGB").save(out)
            return out
        if "RESOURCE_EXHAUSTED" in log or "429" in log:
            raise RuntimeError("Antigravity の画像の回数を使い切りました。これは Codex で描いてください")
        time.sleep(8)
    raise RuntimeError(f"{name}: Antigravity が絵を返しませんでした")


def main():
    kind, proj, name = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
    if kind == "draw":
        print(codex(proj, "draw", name, sys.argv[4], aspect=sys.argv[5] if len(sys.argv) > 5 else None))
    elif kind in ("edit", "style", "ref"):
        print(codex(proj, kind, name, sys.argv[5], input_png=sys.argv[4]))
    elif kind == "agy":
        print(agy(proj, name, sys.argv[4], sys.argv[5]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
