"""キャラ専用の本体の写し（<プロジェクト>/kit）を、道具の本体（vtmm/）に合わせて最新にする。そのキャラのために直した
所は消さない。

使い方: update_kit.py <プロジェクト> [--all]
  そのキャラで一度も変えていないファイル（写したときに記録したハッシュ kit/.engine.json と同じもの）は本体のものに
  入れ替え、本体に増えたファイルは足す。変えてあるファイルは残して、名前を出す。
  --all   全部を入れ替える（先に古い写しをまるごと backups/kit_<日時> に残す）。ハッシュを記録する前に作ったキャラや、
          キャラ専用の直しを捨てたいときに。
"""
import os, sys, json, time, shutil, hashlib
ENGINE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"__pycache__", "app"}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def engine_files():
    out = []
    for root, dirs, files in os.walk(ENGINE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith((".py", ".pt")) and not f.startswith("."):
                out.append(os.path.relpath(os.path.join(root, f), ENGINE))
    return sorted(out)


def record(kit):
    """The hashes of the kit's files as they are now (written right after a copy from the engine)."""
    man = {rel: sha1(os.path.join(kit, rel)) for rel in engine_files() if os.path.exists(os.path.join(kit, rel))}
    with open(os.path.join(kit, ".engine.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    proj = os.path.abspath(sys.argv[1])
    kit = os.path.join(proj, "kit")
    if not os.path.isdir(kit):
        sys.exit("このキャラには専用の本体の写し（kit/）がありません。道具の本体をそのまま使っています")
    man_p = os.path.join(kit, ".engine.json")
    man = json.load(open(man_p, encoding="utf-8")) if os.path.exists(man_p) else {}
    force = "--all" in sys.argv
    if force:
        bk = os.path.join(proj, "backups", time.strftime("kit_%Y%m%d_%H%M%S"))
        shutil.copytree(kit, bk, ignore=shutil.ignore_patterns("__pycache__"))
        print("古い写しを残しました:", os.path.relpath(bk, proj))
    updated, added, kept = [], [], []
    for rel in engine_files():
        src, dst = os.path.join(ENGINE, rel), os.path.join(kit, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst); added.append(rel)
        elif sha1(src) == sha1(dst):
            continue
        elif force or (rel in man and man[rel] == sha1(dst)):
            shutil.copy2(src, dst); updated.append(rel)
        else:
            kept.append(rel)
    record(kit)
    print("新しくしたファイル:", "、".join(updated) or "なし")
    print("足したファイル:", "、".join(added) or "なし")
    if kept:
        print("このキャラのために直してあるので残したファイル（本体の変更が要るなら手で移してください）:", "、".join(kept))
    if not updated and not added and not kept:
        print("もう最新です")


if __name__ == "__main__":
    main()
