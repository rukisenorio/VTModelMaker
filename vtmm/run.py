"""プロジェクトに本体の段階を順に実行します（setup.py と位置合わせの点のあとで）。段階ごとにかかった時間を出し、
エラーがあればそこで止まります。

使い方: run.py <project> [--from STAGE] [--to STAGE] [--install]
段階: prep plates segment hair_base locks eyes face_parts build package preview
  prep      キャンバスと切り抜き（そのあと work/figure_look.png と work/pockets_sheet.png を見て、囲まれた背景を
            bg_pockets に入れ、--from prep でもう一度実行）
  plates    Codex が描く「X を消した」下絵（2 枚ずつ同時に。できているものはそのまま使います）
  ...       segment → eyes → face_parts → build → package（--install: VTube Studio にインストール。VTube Studio を
            終了しているときだけ）
  preview   preview/poses.png（24 のポーズを胸から上で並べた一覧）＋ 正面の姿勢と元の絵の違いの数 ＋ 81 通りの姿勢での
            すき間（preview/holes.png）と、体に頭の一部が残っていないかの確認
各段階の出力は全部 work/logs/<stage>.txt に残ります（ここには最後の数行だけ出ます）。
プロジェクトにキャラ専用の本体の写し（<project>/kit/）があれば、どの段階もそれを使います。キャラごとの直しはそこに入れ、
本体には入れません（全部のキャラが変わってしまいます）。
"""
import os, sys, time, subprocess

STAGES = ["prep", "plates", "segment", "hair_base", "locks", "eyes", "face_parts", "build", "package", "preview"]
STAGE_JA = {"prep": "切り抜き", "plates": "下絵", "segment": "パーツ分け", "hair_base": "髪の下地", "locks": "髪の束", "eyes": "目",
            "face_parts": "眉と口", "build": "組み立て", "package": "VTube Studio 用の設定", "preview": "確認画像"}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    proj = os.path.abspath(os.path.expanduser(sys.argv[1]))
    here = os.path.dirname(os.path.abspath(__file__))
    kit = os.path.join(proj, "kit") if os.path.isdir(os.path.join(proj, "kit")) else here
    a = sys.argv.index("--from") + 1 if "--from" in sys.argv else None
    b = sys.argv.index("--to") + 1 if "--to" in sys.argv else None
    i0 = STAGES.index(sys.argv[a]) if a else 0
    i1 = STAGES.index(sys.argv[b]) if b else len(STAGES) - 1
    # (UTF-8 for every stage: on Windows the default text encoding is the locale's, and Japanese text in character.json
    # or a prompt would be read back wrong)
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    for st in STAGES[i0:i1 + 1]:
        t0 = time.time()
        if st == "preview":
            cmds = [[sys.executable, os.path.join(kit, "preview.py"), proj, m_] for m_ in ("sheet", "rest", "holes")]
        elif st == "package":
            cmds = [[sys.executable, os.path.join(kit, "package.py"), proj] + (["--install"] if "--install" in sys.argv else [])]
        else:
            cmds = [[sys.executable, os.path.join(kit, st + ".py"), proj]]
        print(f"== {st}（{STAGE_JA.get(st, st)}）", flush=True)
        os.makedirs(os.path.join(proj, "work", "logs"), exist_ok=True)
        for j, cmd in enumerate(cmds):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            # (the whole output kept: only its last lines are shown, and a stage's own checks print earlier)
            with open(os.path.join(proj, "work", "logs", f"{st}{'_' + str(j) if j else ''}.txt"), "w", encoding="utf-8") as f_:
                f_.write(r.stdout + r.stderr)
            lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "Warning" not in l and "warn(" not in l
                     and "Core Version" not in l and "preprocess(" not in l]
            for l in lines[-12:]:
                print("   " + l)
            if r.returncode == 3 and st == "plates":
                print("   下絵の絵を待っています（gen/requests/）。絵を入れたら --from plates でもう一度実行してください", flush=True)
                sys.exit(3)
            if r.returncode != 0:
                sys.exit(f"段階 {st}（{STAGE_JA.get(st, st)}）が失敗しました（終了コード {r.returncode}）。直してから --from {st} でもう一度実行してください")
        print(f"   （{time.time() - t0:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
