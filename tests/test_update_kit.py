"""update_kit.py が、キャラ専用の本体の写しのうち直した所は残し、ほかを最新にすることを確かめる。

使い方: python -X utf8 tests/test_update_kit.py      （または python -m pytest tests）
"""
import os, sys, json, shutil, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "vtmm")
sys.path.insert(0, ENGINE)
import update_kit


def run(proj, *args):
    r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(ENGINE, "update_kit.py"), proj, *args],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_update_kit():
    with tempfile.TemporaryDirectory() as tmp:
        proj = os.path.join(tmp, "p")
        kit = os.path.join(proj, "kit")
        shutil.copytree(ENGINE, kit, ignore=shutil.ignore_patterns("__pycache__", "app"))
        update_kit.record(kit)
        assert os.path.exists(os.path.join(kit, ".engine.json"))
        # an old engine file the character never touched, one the character changed, one the old copy lacked
        stale, mine, missing = "preview.py", "eyes.py", "hq.py"
        for f, txt in ((stale, "# old engine\n"), (mine, "# changed for this character\n")):
            open(os.path.join(kit, f), "w", encoding="utf-8").write(txt)
        man = json.load(open(os.path.join(kit, ".engine.json"), encoding="utf-8"))
        man[stale] = update_kit.sha1(os.path.join(kit, stale))         # as if the old engine had been copied
        json.dump(man, open(os.path.join(kit, ".engine.json"), "w", encoding="utf-8"))
        os.remove(os.path.join(kit, missing))
        out = run(proj)
        same = lambda f: update_kit.sha1(os.path.join(kit, f)) == update_kit.sha1(os.path.join(ENGINE, f))
        assert same(stale), out                                         # refreshed
        assert same(missing), out                                       # added
        assert open(os.path.join(kit, mine), encoding="utf-8").read().startswith("# changed"), out   # kept
        assert mine in out.split("残したファイル")[-1] and "残したファイル" in out, out
        # --all replaces it too and keeps the old copy
        out = run(proj, "--all")
        assert same(mine), out
        bk = os.listdir(os.path.join(proj, "backups"))
        assert len(bk) == 1 and open(os.path.join(proj, "backups", bk[0], mine), encoding="utf-8").read().startswith("# changed")


if __name__ == "__main__":
    test_update_kit()
    print("ok")
