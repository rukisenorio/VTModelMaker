---
name: vtmodel
description: VTModelMaker で、キャラクターの絵から VTube Studio（Live2D）のモデルを作る。作ったモデルの仕上げ・直しと、Claude Code と Codex CLI を使う高品質モードもこれ。絵を VTube Studio / VTuber のモデルにしたいと言われたとき（「この絵をVTube Studioのモデルにして」「この絵をモデルにして」「VTuberにしたい」「動かしたい」）、VTModelMaker やモデルメーカーの話が出たとき、VTModelMaker の projects にいるキャラの仕上げや直しを頼まれたとき（「〇〇を仕上げて」「高品質で」「目の開閉がずれる」「髪が透ける」「首を振ると崩れる」）には必ず使う。「Live2D」と言われなくても使う。
---

# Claude と使う VTModelMaker

VTModelMaker は、キャラクターの絵1枚から VTube Studio のモデルを作る道具。`.moc3` を自分で書く（Cubism Editor は
使わない）。絵に隠れた部分は画像生成の下絵で描き、首の向きとうなずき、まばたき、視線、母音つきの口、髪の束と物理演算、
表情、VTube Studio の設定まで作る。このスキルは「Claude に頼む」使い方。もう一つは道具の画面（`start.command` /
`start.bat`）。どちらも同じプロジェクトのフォルダを使うので、片方で始めたキャラをもう片方で仕上げられる。

**相手は日本語話者。** 返事も、見せる説明や確認画像の読み方も日本語で。左右は「向かって左／右」で言う。

## どこに何があるか

- `~/.vtmm/config.json` を読む。`home` が道具のフォルダ（インストーラーが書く）。なければ、VTModelMaker をどこに
  展開したかを聞く。
- Python: `<home>/.venv/bin/python`（macOS）か `<home>\.venv\Scripts\python.exe`（Windows）。いつも `-X utf8` を付けて
  動かす。以下、`PY` はそのコマンド、`V` は `<home>/vtmm`（本体）、`<proj>` はプロジェクトのフォルダ。
- プロジェクト: `<home>/projects/<名前>`。プロジェクトごとに本体の写しが `<proj>/kit/` にある。キャラのための直しは
  そこへ入れ、`V` には入れない（入れるとすべてのキャラが変わる）。
- このパソコンで見つかったもの: `PY V/platform_env.py`（VTube Studio、その Cubism Core、Codex CLI、ffmpeg）。

## 新しい絵

1. `PY V/setup.py <home>/projects/<名前> <名前> <絵>` ― `<名前>` は半角の英数字と `-` `_`。
2. **位置合わせ**: `<proj>/work/guide_full.png` と `<proj>/work/guide_head.png`（`src/art.png` の画素で目盛りを振った
   格子）を開き、`character.json` → `points` に `[x, y]` を書く。各点の意味は `point_help`。`_l` は向かって左。絵にない
   もの（隠れた耳、脚がないなど）は `null`。必須: hair_top, crown, eye_l, eye_r, nose, mouth, chin, face_l, face_r, neck。
   ガイドの絵を拡大して一つずつ置き、先へ進む前に絵の上に描いて確かめる。下書きは `PY V/ai_landmarks.py <proj>` でも
   作れる（そのあと `work/ai_points_check.png` で全部確かめる）。
3. **説明（describe）**: character, hair, eyes, skin, outfit を日本語で短く一つずつ（下絵の指示に入る）。絵を見て書くか、
   `PY V/describe.py <proj>`（Codex）の下書きを直す。色と、向かって左右どちらかをはっきり書く。
4. `PY V/run.py <proj> --to prep` のあと、`work/figure_look.png` と `work/pockets_sheet.png` を見る（キャラに囲まれた
   背景のかけらに番号がある）。背景のものを `bg_pockets` に入れ（`point` は `work/pockets.json` から）、もう一度
   `--to prep`。
5. **下絵**: `PY V/run.py <proj> --from plates --to plates`。
   - Codex CLI があるとき（config `imagegen: codex`）: 自動で描く。10〜30分ほど。
   - 手作業モード（`imagegen: manual`、終了コード 3）: 依頼は一つずつ `gen/requests/<名前>/input.png` と `prompt.txt`。
     ChatGPT で作ってもらい（絵を添付し、文章を貼る）、できた絵をそのフォルダ（ファイル名は何でもよい）か画面に入れて
     もらう。あるとよい下絵は `character.json` → `plates_skipped` に入れて省ける。`bald`、`face_blank`、`noiris` は必須。
     終了コード 0 になるまで、この段をくり返す。
   - 画像生成に断られた依頼は、それで終わり。言い換えて頼み直さず、別の素材を使う。
6. `PY V/run.py <proj> --from segment --to preview` のあと、`preview/poses.png` と `preview/holes.png` を見る。
7. **動きで判断する**（表示サイズで。それまで「できた」と言わない）:
   `PY V/motion_strip.py <proj>/out/<名前> <proj>/preview/motion.png --box x0,y0,x1,y1 --scale 0.3`（物理込みの首振り）と、
   同じものに `--blink 1.0`（まばたき1回を1コマずつ）。並びの絵を相手に見せる。
8. **インストール** ― VTube Studio を閉じているときだけ:
   `PY -c "import sys; sys.path.insert(0, r'<V>'); import platform_env as p; print(p.vts_running())"` が `False` を出すこと。
   そのあと `PY V/package.py <proj> --install`。前に入れたとき VTube Studio が保存した設定に足す形で入れ（控えは
   `<proj>/backups/`）、VTube Studio のフォルダの中のものは何も消さない。

## 高品質（Claude Code ＋ Codex CLI）

高品質を頼まれたとき（「高品質で」「しっかり仕上げて」「もっと頑張って」）や、最初の一体に満足していないときは
`references/hq.md` に従う。AI の位置合わせの下書きを自分で確かめ、下絵は全部確かめ、見本の向き絵を描き
（`hq.py refs`）、首の回り方と見上げ方をそれに合わせ（`hq.py calibrate --apply`）、そのあと確認パック（`hq.py qa`）、
確認（`hq.py review` と自分の目）、部位ごとの直し、残した版との比較（`hq.py snapshot`）をくり返す。

## 仕上げと直し

- 直す前に、崩れを描いているのが誰かを突き止める: `PY V/preview.py <proj> peel x0,y0,x1,y1 "ParamAngleX=30,ParamEyeLOpen=0"`
  は、その姿勢でアートメッシュを一つずつ描く。`PY V/preview.py <proj> zoom <box> "<姿勢>;<姿勢>"` は姿勢を並べて比べる。
- 変えた段からやり直す（`run.py <proj> --from <段>`）。段は prep, plates, segment, hair_base, locks, eyes, face_parts,
  build, package, preview。
- 毎回、前の組み立てと比べる（`motion_strip.py ... --compare <前の out フォルダ>`）。危ない変更の前に `out/<名前>` を
  残しておく。
- `references/method.md`: 作り方の9つの決まり。最初に読む。
- `references/lessons.md`: 各段がすることと、これまでに出た崩れと直し方の全部（目が閉じ切らない、首を回すと髪が
  ほつれる、横髪が顔の後ろに入る、耳、襟、帽子…）。目の前の崩れで検索する。
- 道具のフォルダの `docs/engine.md`: moc3 を書く本体の説明（リグ、デフォーマ、キーフォーム、物理演算）。
- 道具を新しくしたあとは `PY V/update_kit.py <proj>` でキャラ専用の本体の写しを最新にする（直したファイルは残る）。

## 決まり

- 相手が使う権利を持つ絵だけを使う。下絵は画像生成（AI）で描く。公開・販売するモデルは、販売先の AI 作品の決まりに
  従う。
- 数や止まった絵でモデルを「できた」と言わない。表示サイズで、動かして崩れないこと。
- インストールは VTube Studio を閉じているときだけ。VTube Studio のフォルダにある相手のファイルを消したり上書きしたり
  しない。
- Live2D Cubism Core は VTube Studio のもの（または公式 SDK のもの）。そこから読み込み、写したり配ったりしない。
