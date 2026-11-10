"""install.command / install.bat の最後の段（道具の Python で動く）: 設定ファイル、Claude 用のスキル、このパソコンに
そろっているものの確認。

使い方: install_finish.py [--no-skill]
  - ~/.vtmm/config.json: 道具の場所（"home"）と下絵の描き方（"imagegen": Codex CLI があれば codex、なければ manual）。
    前からある設定は残す
  - Claude Code 用のスキル: <道具>/skill/vtmodel → ~/.claude/skills/vtmodel（前の写しは消さずに横へ移す）
  - 確認: Python の部品、VTube Studio とその Live2D Cubism Core、Codex CLI、Claude Code、ffmpeg
"""
import os, sys, shutil, time, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import platform_env as pe

HOME = os.path.dirname(HERE)


def say(ok, text):
    print(("  OK   " if ok else "  !!   ") + text, flush=True)


def main():
    print("\n== 設定を書いています")
    cfg = pe.config()
    cfg["home"] = HOME
    if cfg.get("imagegen") not in ("codex", "manual"):
        cfg["imagegen"] = "codex" if pe.codex_bin() else "manual"
    pe.save_config(cfg)
    say(True, f"設定ファイル: {pe.CONFIG_PATH}")

    if "--no-skill" not in sys.argv:
        print("\n== Claude Code 用のスキルを入れています")
        src = os.path.join(HOME, "skill", "vtmodel")
        dst = os.path.join(os.path.expanduser("~"), ".claude", "skills", "vtmodel")
        if os.path.isdir(src):
            if os.path.exists(dst):
                old = dst + time.strftime("_old_%Y%m%d_%H%M%S")
                shutil.move(dst, old)
                say(True, f"前のスキルは {old} に移しました")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
            say(True, f"スキル: {dst}（Claude Code に「この絵をVTube Studioのモデルにして」と頼むと使われます）")
        else:
            say(False, "skill/vtmodel が見つかりません")

    print("\n== このパソコンの確認")
    bad = 0
    for mod, why in (("numpy", "計算"), ("scipy", "計算"), ("cv2", "画像処理"), ("PIL", "画像"), ("skimage", "画像処理"),
                     ("torch", "絵の拡大")):
        try:
            importlib.import_module(mod)
            say(True, f"{mod}（{why}）")
        except Exception as e:
            bad += 1
            say(False, f"{mod}（{why}）が使えません: {e}")
    s = pe.summary()
    if s["vts_models_dir"]:
        say(True, f"VTube Studio: {s['vts_dir']}")
    else:
        bad += 1
        say(False, "VTube Studio が見つかりません（Steam から入れてください。入れたあとでも、このツールはそのまま使えます）")
    if s["core_lib"]:
        try:
            from l2d import core
            v = core.lib().csmGetVersion()
            say(True, f"確認用エンジン（VTube Studio の中の Cubism Core {v >> 24}.{(v >> 16) & 255}.{v & 0xffff}）")
        except Exception as e:
            bad += 1
            say(False, f"確認用エンジンが読み込めません: {e}")
    else:
        say(False, "確認用エンジンがありません（VTube Studio を入れると使えます）")
    if s["codex"]:
        say(True, f"Codex CLI: {s['codex']}（下絵は自動で描かれます。初回は `codex login` が要ります）")
    else:
        say(True, "Codex CLI はありません → 下絵は手作業（ChatGPT で作って入れる）モードになります")
    if s["claude"]:
        say(True, f"Claude Code: {s['claude']}（高品質モードで確認画像を見ます。ログインしていなければ、ターミナルで `claude` を起動して /login）")
    else:
        say(True, "Claude Code はありません → 高品質モードの確認は Codex がします")
    say(bool(s["ffmpeg"]), "ffmpeg（動画を作るときだけ使います。なくても大丈夫）" if s["ffmpeg"] else "ffmpeg なし（動画は作れませんが、ほかは大丈夫）")
    print()
    if bad:
        print("いくつか足りないものがあります。上の !! の行を見てください。")
    else:
        print("準備ができました。")
    print("使い方: " + ("start.bat" if pe.IS_WIN else "start.command") + " をダブルクリック（ブラウザで画面が開きます）")


if __name__ == "__main__":
    main()
