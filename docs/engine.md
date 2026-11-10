<!-- VTModelMaker の注記：moc3 エンジンの開発者向けリファレンス。VTModelMaker ではエンジンは vtmm/l2d/ にある
（以下の template/ と chara3/ のパスは、このエンジンの出どころのプロジェクトのもの）。ここで kit/l2d と書いている
近道のエンジンが、vtmm/l2d のもの。 -->

# l2d エンジン リファレンス（.moc3 を直接書く。Cubism Editor は使わない）

新しいキャラの組み立てスクリプトを手早く書くためのもの。正とするのは、スキルの `template/l2d/*.py`、
`template/build_c3.py`（約2,200行）、`template/package_c3.py`（どれも3体目のキャラ chara3 のものの写し）と、
scripts/new_character.sh がそれらから作るプロジェクトの写し（各プロジェクトが自分の `l2d/` を持つ）。
以下のシグネチャと動作は、2026-10-21 にそのソースと VTube Studio の Core で確かめた。以下で `chara3/...`
と書いたパスは、テンプレートやプロジェクトの中の同じファイルを指す。
近道のエンジン `kit/l2d/`（chara2 の新しい写し。kit の各プロジェクトにも写される）は、これに2つを足したもの。
`rig.warp(..., unfold=m)` は、どのキーフォームでも格子が折れないようにする（各行の x と各列の y を、静止時の間隔の m
倍以上あけて順に保つ。単調回帰（isotonic regression）による。chara2 は髪に 0.2、顔に 0.03 を使った）。
`rig.mesh(..., blend="add"|"multiply")` は Cubism のブレンドのフラグを立てる（Cubism のレンダラはそれを描画する）。
以下はどちらにも当てはまる。テンプレートのリグのコードは、kit のエンジンでそのまま動く。

**目次：** 1 モジュールとデータの流れ | 2 Rig API | 3 キーフォーム：サンプリング、補間、キーの置き方 | 4 座標、
描画順、マスク、テクスチャ | 5 レンダラ | 6 core.Model（VTS 自身の Core） | 7 物理演算の移植 | 8 package_c3.py（VTS
用のパッケージ化、インストール） | 9 build_c3.py のパターン（コード片）

## 1. モジュールとデータの流れ

| モジュール | 役割 |
|---|---|
| `rig.py` | `Rig`：モデルを手続きで記述したもの（パラメータ、パーツ、レイヤー、デフォーマの木、メッシュ、物理演算、表情）。`compile()` は、すべての `fn` をすべてのキーフォームでサンプリングし、親のローカル座標に変換し、アトラスを詰め、全ファイルを書く |
| `builder.py` | `Builder`、`Binding`：親ローカルの配列 -> moc3 の表（パラメータ／キーフォームのバインディング、描画順グループは1つ）。ルートの px -> Core の単位 |
| `moc3.py` | `parse(path) -> dict`、`write(model, path, version=3)`：バイナリのコンテナ |
| `meshgen.py` | `grid_mesh`（`layer_mesh` が内部で使う）、`strip_mesh`（リボン）、`NestAtlas`（既定のパッカー）、`Atlas`（棚詰め）、`bleed`、`bleed_far` |
| `core.py` | VTube Studio の中の Cubism Core への ctypes バインディング：読み込みと整合性チェック、パラメータの設定、変形後の頂点の読み出し |
| `render.py` | `Renderer`：`core.Model` の状態を描くソフトウェアのラスタライザ（レンダー順、マスク、不透明度） |
| `physics.py` | `Physics`：Cubism の physics3 の振り子の移植。呼び出し側が1ステップずつ進める |
| `evalref.py` | `Evaluator`：Core のデフォーマの計算を、`moc3.parse()` の出力の上に実装し直したもの。規約を突き止めるためだけに使う |

流れ：レイヤー（`tools/layerio.py`：`manifest()`、`load_layer(name) -> (rgba, x0, y0)`）-> `rig.layer` -> デフォーマ
-> メッシュ -> 物理演算、表情 -> `rig.compile(out_dir, MODEL)`。compile は `out/<MODEL>/` に次を書く：
```
<MODEL>.moc3                    moc バージョン 3（Cubism 4.0 の形式。VTS の Core 5.0.0 は <= 5 を読む）
<MODEL>.<atlas_size>/texture_NN.png
<MODEL>.model3.json             Moc, Textures, DisplayInfo, Physics, Expressions; Groups EyeBlink / LipSync
<MODEL>.physics3.json           add_physics を呼んだときだけ
expressions/<name>.exp3.json    rig.expressions の項目ごとに1つ
<MODEL>.cdi3.json               パラメータ、パラメータグループ、パーツの表示名
```
そのあと `core.Model(moc)`（整合性チェック）、確認（§5〜7）、`package_c3.py`（§8）。書かないもの：グルー、
乗算色／スクリーン色、ブレンドシェイプ、加算／乗算ブレンド、パーツのキーフォーム、pose3.json。
メッシュはすべて両面・通常ブレンド（constant flags 4、+8 = 反転マスク）。

## 2. Rig API（l2d/rig.py）

| 呼び出し | 戻り値 |
|---|---|
| `Rig(width, height)` | キャンバス px。モデルの ppu = width、原点 = キャンバスの中央 |
| `rig.param(pid, lo, hi, default=0.0, name=None, group=None)` | pid |
| `rig.part(pid, name=None)` | pid |
| `rig.layer(name, img, x0, y0, scale=1.0, hi=None)` | name |
| `rig.warp(name, parent, part, rect, cols, rows, axes=(), fn=None, opacity=None)` | Node |
| `rig.rot(name, parent, part, pivot, axes=(), angle=None, origin=None, scale=None, opacity=None)` | Node |
| `rig.layer_mesh(name, parent, part, z, cell=48, pad=8, layer=None, canvas_diag=False, thr=8, **kw)` | name |
| `rig.mesh(name, layer, parent, part, verts, tris, z, axes=(), fn=None, opacity=None, masks=(), uv_px=None, inverted_mask=False, draw_order=500)` | name |
| `rig.expressions[name] = {"params": {...}, "fade": s}` | |
| `rig.add_physics(name, inputs, outputs, vertices, norm_pos=(-10, 0, 10), norm_ang=(-10, 0, 10))` | None |
| `rig.compile(out_dir, model_name, atlas_size=4096, texture_dir=None)` | .moc3 のパス |

共通：`parent` は warp/rot が返す Node か None（ルート）。`part` は宣言済みのパーツ id。
`axes = [(param_id, [keys])]`、キーは昇順で重複なし（assert で確かめる）。`fn(pts, v)` は静止時の点（float64 の
(N, 2)、コピー）と、`v` = 全パラメータの値（既定値 + このキーフォームのキー）を受け取り、変形後の点をキャンバス px
で返す（§3、§4）。`opacity(v) -> 0..1`。

**名前。** パラメータ、パーツ、デフォーマ、メッシュは1つの id 名前空間を共有する（一意、64 バイト未満、
assert で確かめる）。レイヤー名は別（メッシュは普通、そのレイヤーと同じ名前にする）。`Param*`、`Part*`、
ワープは `W_*`、回転は `R_*`。**クロージャ：** compile() は、build() が戻ったあとですべての fn を呼ぶ。
ループ変数は既定引数で束縛する（`lambda v, OP=OP: ...`）。

**param。** 宣言の順 = moc と cdi3 での順。`name` / `group` は cdi3 に入る（グループは最初に使われた順に ParamGroup1..
になる）。`rig.param_defaults` = {pid: default}。トラッキングが動かすものには標準の id を使う（ParamAngleX/Y/Z、
ParamEyeL/ROpen、ParamEyeL/RSmile、ParamEyeBallX/Y、ParamBrowL/RY、ParamMouthForm/OpenY/X、ParamA..ParamO、
ParamCheek、ParamBodyAngleX/Y/Z、ParamBreath）。model3 の EyeBlink は ParamEyeL/ROpen を、LipSync は ParamMouthOpenY
を取る。Core は値を [lo, hi] に切り詰める。**part。** 平らなリスト。パーツは cdi3 と VTS の UI
のためにまとめるもので、描画順も不透明度も決めない。

**layer。** `img`：uint8 の RGBA、ストレートアルファ、左上がキャンバスの (x0, y0)。`scale`：
アトラスでのキャンバス 1 px あたりのテクセル数（build_c3 は体、腕、クリスタル、ストラップを 0.5 で格納する）。`hi`：
同じ絵を整数倍 k >= 2 にしたもの（ちょうど k*h x k*w）で、代わりにこちらを格納する（scale = k）。`rig.layers[name]`
= {img, x0, y0, scale[, hi]}。アトラスに入るのは、どれかのメッシュが使うレイヤーだけ。
同じ絵で2つ目のメッシュを作るには、新しい名前でもう一度登録するか、layer_mesh に `layer=` を渡す。

**warp。** `rect = (x0, y0, x1, y1)`：静止時の格子のキャンバス px。**cols が rows より先**。fn は (rows+1)*(cols+1)
個の格子点を、行優先（y が外側）で受け取る。fn=None：恒等、つまり入れ物。
子は各セルの中で格子に双線形に従う（isQuadSource）：首振りのワープは頭の箱に 30 x 26（セル約 70 px。
隣どうしが同じように変形するよう、頭のどの面にも同じ rect と格子を使う）、揺れは 16〜30 の正方、
目は 6 x 5 〜 14 x 10、純粋な平行移動は 2 x 2。`opacity` は下にあるものすべてに掛かる。

**rot。** `pivot`：静止時の原点、キャンバス px。`angle(v)`：度、正 = 画面上で時計回り（Core で確認：
ParamAngleZ +30 で、頭頂が首を中心に 383 px 右へ動く）。`origin(v)`：キャンバスの (x, y)、子を一緒に運ぶ。`scale(v)`：
倍率。子は剛体として動く。ワープの下では、Core がワープを通して運ぶのは回転の原点と角度だけで、
局所的な縮みはない（恒等の rot の下に置いた目は、首振りで遠近の縮みがなくなった）。剛体の回転（R_Head のロール）
には rot を使い、ワープの下の入れ物には恒等のワープを使う。

**layer_mesh。** レイヤー（`layer or name`）を覆う格子メッシュ：`pad` px 膨らませたあとで alpha > `thr` が触れる、
`cell` px のセルすべて（格子はキャンバス上の `cell` の倍数に固定されるので、隣り合うレイヤーは頂点を共有する）。
`canvas_diag=True`：各セルの対角線をキャンバス上の位置だけで決める（切り直したレイヤーでもメッシュが変わらない）。
`**kw` は mesh に渡る。build_c3 のセル：体 64、腕／後ろ髪 40、前髪 36、耳 24、顔 20、細い部品 14〜16、目 10〜12、
鼻 8。親ワープのセルより粗いメッシュは、ワープの曲がりを失う。細かいメッシュにキーフォームが多いと重い（§3）。

**mesh。** 手で作るもの：`verts` は静止時のキャンバス px (N, 2)、`tris` は (M, 3)、N < 32767。`uv_px`：
各頂点がレイヤーのどこを参照するか、レイヤーのキャンバス px で（既定：verts）。
描いたシートを別の場所の形に貼るのに使う（口の中、歯、舌、唇の線のリボン）。fn は `pts` を無視して `v`
から位置を作ってもよい。fn が見るのはキーの値だけなので、
キーの値で引く表が使える（`MBASE[(v["ParamMouthOpenY"], round(v["ParamMouthForm"]))]`）。
`verts` = `rig.param_defaults` での fn の値。

**expressions。** `{"params": {pid: value | (value, blend)}, "fade": s, "blend": default}`
-> expressions/<name>.exp3.json。FadeIn = FadeOut = fade（既定 0.15）、Blend は値ごと（既定は "Overwrite"、
ほかに "Add"、"Multiply"）。VTS では、表情はトラッキングの上に重なる。Add / Multiply
なら下でトラッキングが続き（眉と口の形は足し算、目の開きは掛け算）、Overwrite は値を固定する（切り替え、記号）。
保たなければならないポーズは表情に入れる（§8）。

**add_physics。** `inputs`：[(param, weight %, "X" | "Y" | "Angle", reflect=False)]。`outputs`：
[(param, vertex_index >= 1, scale, weight %, "X" | "Y" | "Angle", reflect=False)]。粒子 i-1 -> i
の線分から読む（Angle：その前の線分に対する角度、ラジアン x scale、パラメータの範囲に切り詰め）。`vertices`：
[(x, y, mobility, delay, acceleration, radius)]、最初が固定点 (0, 0, 1, 1, 1, 0)。`norm_pos` / `norm_ang`：
正規化した入力の (min, default, max)。1回の呼び出し = PhysicsDictionary に名前の載る PhysicsSetting 1つ（VTS
に一覧が出る）。Meta Fps 60、重力 (0, -1)、風なし。出力は、ほかに何も書かないパラメータを動かす。
入力はどのパラメータでもよい（ParamWind：書くのは待機モーションだけ）。

**compile。** fn / opacity / angle / origin / scale を、
キーフォームの組み合わせごとに1回ずつ呼び（重い表はキャッシュする。build_c3 の `lip_x` のように）、
結果を親ローカルの座標系に写し、§1 のファイルを書き、ページを `rig.atlas_pages` に残す。アトラス：
使われる各レイヤーの箱（メッシュの verts / uv_px まで広げる）をそのレイヤーの scale で置き、
透明なテクセルに色をにじませ（6 px、次に 32 px まで）、メッシュが覆う範囲で 16 px の格子の上に入れ子に詰める。
余白 16 px、90度の回転を許す。古い texture_NN.png は消す。`L2D_ATLAS=shelf`：古い棚詰めのパッカー。

## 3. キーフォーム：サンプリング、補間、キーの置き方

**サンプリング。** オブジェクトの軸 [(p1, K1), ..., (pn, Kn)] から、|K1| x ... x |Kn| 個のキーフォームができる。
compile() は組み合わせごとに1回、`v` = 全既定値 + そのキーの値で fn を呼ぶ（最初の軸がいちばん速く変わる）。
キーの間は一切サンプリングしない。**実行時。** Core は軸ごとに値を挟む2つのキーを取り（端のキーで切り詰め）、
周りの 2^n 個のキーフォームを多重線形に混ぜる。ワープの点、メッシュの頂点、
回転の角度／原点／倍率、不透明度、どれも同じ。

実際にわかったこと：
1. fn や opacity が読むパラメータは、すべてそのオブジェクトの軸にする。それ以外は compile のあいだ既定値に見えるので、
   実行時の動きは黙ってそれを無視する（arm_fn は ParamPoseWave を読むので、腕はそれを軸に持つ）。
2. キーの間では、各頂点は直線の上を動く。fn の中でした回転は弦になる：キーの間の刻みが a なら、中ほどで点は
   r(1 - cos(a/2)) だけ内側に寄る（10度で 0.4 %、30度で 3.4 %）。数度の揺れなら問題ない。大きな回転では、
   キーを足すか回転デフォーマを使う（補間されるのはその角度）。
3. fn や opacity の中の傾斜には、どれも両端にキーを置く。そうすれば区分線形の関数はそのまま再現される：白目の `seen`
   = clip((open - 0.06) / 0.09) には EYE_KEYS [0, 0.06, 0.15, 1]。記号は [0, 0.5, 1]（0〜0.5 で現れる）。
   あごのパッチの AngleY のキーは FLARE_ON (2, 6) に。奥のゴーグルのストラップの AngleX のキーは +-6、
   +-16 に（そのフェードアウト）。曲線の区間には中にもキーが要る（WAVE_KEYS には、
   腕の予備動作と持ち上げのために 0.25 と 0.75 がある）。
4. 既定値をキーにする。[-1, 1] だけだと、静止ポーズは両端の平均になる（chara2：|d| の項が静止時に 2.6 px
   ずれていた）。範囲 -1..1 のパラメータには、すべて K3 = [-1, 0, 1]。
5. 瞬時の切り替え = 0.01 離した2つのキーと、その間の段差：WAVE_KEYS [0, .25, .5, .51, .75, 1] で、
   入れ替えは WAVE_SWAP 0.505。GOG_KEYS [0, .5, .51, .85, 1]。GOG_OFF_KEYS [0, .49, .5, 1] で、段差は 0.5（Core：
   ParamGoggles 0.5 -> 不透明度 1、0.495 -> 0.5、0.49 -> 0）。表情のフェード（0.3〜0.4秒）やモーションは、
   その 1 % の幅を数ミリ秒で通り過ぎる。入れ替えの前の動き（滑る、持ち上がる、予備動作）は、
   同じパラメータをキーにした形の変化。入ってくるパーツは 0.85 までに定位置に着く（保持した切り替えは
   0.88〜1 で揺らぐ）。両方向を確かめる。
6. 不透明度のクロスフェードは2枚の絵を見せる。キーが [0, 1] だけだと、
   段差状の不透明度も全範囲にわたるクロスフェードになる（Core：ParamEyeType 0.5 -> 両方の黒目が 0.5。
   chara3 のしいたけ目の切り替えは、そういう 0.1秒のクロスフェード）。実際に見たもの：髪の上に透けて見えるゴーグル、
   ベースと明るい版の組が暗くなって箱に見える、口の絵のクロスフェードが、混ざった母音で二重に透ける。
   切り替えは入れ替えのキーで行う。フェードさせるのは、透けてよいものだけ（頬の赤み、
   どんより（gloom）、記号、肌の上の光）。
7. 数が掛け算になるのは「1つの」オブジェクトの軸どうしで、鎖に沿ってではない。子は自分の軸だけを持ち、
   Core が毎フレーム鎖を合成する。だから前髪の鎖（首振り 9、弾み 3、左の横髪 9、前髪中央 9、右の横髪 9、
   髪の束のワープ3つが各 27）の費用は和で済む。18軸すべてを1つのワープに持たせれば 3^18 になる。
   ぶら下がっていないデフォーマをメッシュが再現しなければならないときは、掛け算が戻ってくる：jaw_flare（W_Body の上）
   は W_Face のキーフォームを自分で混ぜる（`face_lin`）ので、AngleX x Y x Z を 7 x 8 x 5 = 280 キーフォーム持つ。
   親と軸を共有する子（W_Face の下の W_Mouth）には、まず自分のキーフォームが掛かり、次に同じキーで親のものが掛かる。
   狙いどおりの所に着かせるには、親の写像の逆をとる（§9.7）。
8. 予算：オブジェクトごとに約 8 バイト x 頂点数 x キーフォーム数。Chara3（10-21 時点）：moc3 22 MB、
   メッシュ85個／頂点 19,748、メッシュのキーフォーム 6,716 + ワープのキーフォーム 409。arm_L / arm_R だけで（6軸、
   3*3*2*3*3*6 = 972 キーフォーム x 約 1,010 頂点）15.8 MB、7軸の口のメッシュ（648 キーフォーム、頂点 27〜84）
   で 1.5 MB。VTS の Core はこれを受け付ける（core.check）。軸の多いものは、小さなメッシュか粗いワープに置く。
   独立した動きは鎖に分ける。組み立てたモデルの予算は `a = moc3.parse(path)["art"]` で読む：`a["ids"]`、
   `a["kfCount"]`、`a["vertexCount"]`（合計は `m["counts"]`）。

## 4. 座標、描画順、マスク、テクスチャ

- **キャンバス px**、原点は左上、x は右、**y は下**：rect、pivot、頂点、レイヤーの位置、fn の入力と出力。
  moc はルートの座標を (px - centre) / ppu（ppu = width）で格納する。Core は y が上向きの単位で返し、
  `positions_px` は px で返す。
- **fn は、静止時の親のキャンバス座標で働く。** compile() は、親の静止時の形を使って fn
  の出力を親のローカル座標に写す（warp：その rect での (p - (x0, y0)) / (x1 - x0, y1 - y0)、rot：p - pivot）。
  その上に、Core が親の今の変形を掛ける。だから各 fn は自分の動きだけを書けばよく、鎖は子から親へと、
  ルートまで合成される：hair_back（fn なし）-> W_HairBackR -> W_HairBackL -> W_HairBackBounce -> W_HairBack（首振り）
  -> R_Head（ロール）-> W_Body（体のひねり、傾き、呼吸）。静止時の絵の上で決めた動き（揺れ）は、
  見え方を大きく変えるもの（首振り、ロール、体）の下に置く。
- 子は、親ワープの rect の内側に余白をとって収める（build_c3 は 60〜200 px あける）。外に出ると、
  Core は格子の縁から外挿する。
- **z**：メッシュは z の昇順 = 奥から手前に並べて書く。どれも描画順 500 のままなので、書く順で決まる（`draw_order=`
  はあるが、build_c3 は一度も変えない）。小数の z で間に差し込む（neck 40.5、neck_lit 40.55、jaw_flare 40.6、ears 41、
  face 42、hair_front 50）。前後の順はメッシュごとに固定。奥から手前へ回るには、もう一方の z に2つ目のメッシュを置き、
  入れ替えのキーで切り替える。
- **マスク**：`masks=["a", "b"]` は、それらのアートメッシュにアルファがある所（その和集合）にだけメッシュを描く。
  `inverted_mask=True` は、アルファがない所にだけ描く。名前はメッシュの id。マスクに使うメッシュも普通に描かれる。
  使った例：白目の中の黒目、mouth_in の中の舌と歯、顔の上の fx_gloom、体に落ちる髪の束の影、neck + jaw_flare の上の
  jaw_under。反転マスクのレイヤーは、マスクのある所では何も覆わない。そこで隠していたものは、すべて見えてしまう。
- **テクスチャはストレートアルファ**（レイヤーもアトラスのページも）。alpha 0 の下の色は問わない。
  アトラスがそこに色をにじませる（そうしないと、ミップマップで縁に黒が混ざる）。生成したレイヤーは、build_c3 と同じく
  `img[img[..., 3] == 0] = 0` とする。

## 5. レンダラ（l2d/render.py）

`Renderer(model, textures, scale=0.25, crop=None)`：`model` は core.Model。`textures` はページを uint8 の
RGBA・ストレートアルファの配列にしたもの。`scale` はキャンバス 1 px あたりの出力 px。`crop` はキャンバス px の
(x0, y0, x1, y1)（既定：キャンバス全体）。出力の大きさは round((x1 - x0) * scale) x round((y1 - y0) * scale)。
scale が 1 未満なら、ページは作成時に1回だけ縮める（乗算済みアルファ、INTER_AREA）。scale／crop ごとに Renderer
を1つ作り、使い回す。scale 1 では、4096 のページ1枚に約 270 MB かかる。

`render(background=None, only=None, skip=None)` は、モデルの「今の」Core の状態を描く（先に set + update）。
アートメッシュを Core のレンダー順に描き、不透明度 <= 0.001 や非表示のものは飛ばし、
マスクを掛け（マスクのアートメッシュのテクスチャのアルファの和集合。フラグ 8 で反転）、乗算済みアルファの "over"
で重ねる。戻り値は uint8 の RGBA・ストレートアルファ（background が None のとき）か、`background` = (r, g, b) の上の
RGB。`only` / `skip`：アートメッシュの id の集合（`only` のときもマスクは効く）。

```python
from l2d import core
from l2d.render import Renderer
mj = json.load(open(f"{d}/Chara3.model3.json"))
tex = [np.asarray(Image.open(os.path.join(d, t)).convert("RGBA")) for t in mj["FileReferences"]["Textures"]]
m = core.Model(os.path.join(d, mj["FileReferences"]["Moc"]))
pose = {"ParamAngleX": 30, "ParamAngleY": -20, "ParamEyeLOpen": 0.3}
m.reset(); m.set(**{k: v for k, v in pose.items() if k in m.param_index}); m.update()
r = Renderer(m, tex, scale=1.0, crop=(1700, 600, 2900, 1800))         # 頭を等倍で
Image.fromarray(r.render(background=(20, 22, 30))).save("pose.png")    # 暗い地の上で判断する
alone = r.render(only={"hair_front"}, background=(255, 0, 255))        # 剥がして見る。skip={...}：それが隠しているもの
```
見る大きさ：バストアップ約 0.3〜0.4、全身約 0.15、細部は等倍を 2〜8倍に拡大。chara3 の道具：`work/peel_c3.py <out.png>
x0,y0,x1,y1 "Param=v,..." [scale]`（そこにあるアートメッシュを1つずつ単独で、大きいものから）、
`tools/clip_c3.py`（物理演算込みの動画）。

## 6. core.Model（l2d/core.py）

`CORE_PATH` = `~/Library/Application Support/Steam/steamapps/common/VTube Studio/VTubeStudio.app/Contents/PlugIns/
Live2DCubismCore.bundle`：アプリが実行する Core（10-21 時点で 5.0.0）。形、切り詰め、補間、
レンダー順は、アプリのものと同じ。
- `core.check(path)` -> {core, latest_moc_version, moc_version, consistent}。
  インスタンスを作らずに調べる。`core.core_version()`。
- `m = core.Model(path, check_consistency=True)`：moc が Core の整合性チェックに通らなければ ValueError。
  compile のたびに最初にする確認（build_c3 の `__main__` は、モデルを読み込んで数を表示する）。
- 静的なもの：`canvas_size`（w, h px）、`canvas_origin`、`ppu`。`param_ids`、`param_min/max/default`、`param_index`
  {id: i}。`part_ids`、`part_parents`。`n_drawables`、`drawable_ids`（リスト）、`constant_flags`、`texture_indices`、
  `vertex_counts`、`index_counts`、`mask_counts`、`drawable_parts`。アートメッシュごとに `uvs[i]`（v は上向き）、
  `indices[i]`（平らな uint16）、`masks[i]`。
- ライブビュー（update() のあとで有効）：`param_values`、`part_opacities`、`opacities`、`render_orders`、
  `draw_orders`、`dynamic_flags`（bit 0 = 表示中）。
- `set(**values)` は値を書く（知らない id は KeyError。`param_index` で絞る）。`reset()` は全部を既定値に戻す。
  `update()` は Core を走らせる。`positions(i)`：ライブビュー、Core の単位、y は上向き（次の update
  のあとまで残すならコピーする）。`positions_px(i)`：新しい配列、キャンバス px、y は下向き。
使い道：ランダムなポーズで頂点が有限か。折れ = 静止ポーズと比べて符号付き面積が反転した三角形（positions_px の上の
`indices[i].reshape(-1, 3)`）（work/fold_motion.py）。目鼻の位置を見本の向き絵と比べる。

## 7. 物理演算の移植（l2d/physics.py）

`Physics(path, ranges)` は physics3.json を読む。`ranges` = {param: (min, max, default)}、入力と出力のすべてについて。
`step(values, dt) -> {output param: value}`：全設定を1ステップ進める（入力は正規化 x weight。
粒子は delay * dt * 30 で動き、速度 x mobility。出力 = scale x 線分の値、切り詰め、weight で混ぜる）。
結果は `vals.update(...)` で合わせる。状態はオブジェクトの中にある：シミュレーション1つにインスタンス1つ。VTube Studio
は物理演算を **描画1フレームに1回** 進める（Live2DPhysicsFPS 3 = "FPS_SAME_AS_GAME"。ユーザーの Mac
で毎秒28〜60ステップ）。そして振り子はフレームレートに左右される：同じファイルでも、
毎秒30ステップのほうが60ステップより大きく、長く揺れる。プレビューも調整も両方で行う。
```python
from l2d.physics import Physics
ranges = {p: (float(m.param_min[i]), float(m.param_max[i]), float(m.param_default[i])) for i, p in enumerate(m.param_ids)}
for steps in (60, 30):
    phys = Physics(f"{d}/Chara3.physics3.json", ranges); vals = {p: ranges[p][2] for p in ranges}
    for k in range(int(30 * dur)):                                      # 描画フレームは1秒に30
        vals.update({a: b for a, b in motion(k / 30).items() if a in ranges})   # トラッキングのような入力
        for _ in range(steps // 30): vals.update(phys.step(vals, 1.0 / steps))
        m.reset(); m.set(**vals); m.update()                            # そのあと描く、または折れを測る
```

## 8. package_c3.py（VTube Studio 用のパッケージ化とインストール）

`python3 package_c3.py [--install]` は `out/<MODEL>/`（MODEL "Chara3"、FOLDER "chara3"）に対して働き、次を書く：
- `animations/<MODEL>_idle.motion3.json`：ほかに何も書かないパラメータの16秒のループ（キーは0.05秒ごと、
  最後 = 最初）：ParamWind（3つのサイン波。物理演算の入力なので、髪、クリスタル、ストラップが揺れる）、
  ParamEarL/R のぴくっとした動き、ParamEyeSparkle。トラッキングの下に重なる。`.vtube.json` は
  FileReferences.IdleAnimation でこれを指定する。
- `_check`（31.5秒、フェードイン／アウト 0.5/0.6）：見回す、見上げ、見下ろし、傾く、揺れる、あいうえおと話す、笑う、
  手を振る + ピース、目にゴーグル。**ホットキーのモーションがトラッキングされるパラメータを押さえるのは、
  値が変わり続けているあいだだけ**：だから中で止まっているものはない（保持の所もゆっくり動かす）。
  本当に保つものは表情に入れる。`_wave`（3.3秒）、`_peace`（3.4秒）：
  ポーズの切り替えを0.45秒で上げ、身ぶり、下げて戻す。
- `icon.png`（work/canvas_base.png から 512 x 512 を切り出し）と `<MODEL>.vtube.json`（`vtube_json()`）：
  FileReferences（Icon、Model、IdleAnimation）。ParameterSettings は
  `ps(name, inp, lo_i, hi_i, out, lo_o, hi_o, smooth=15, clamp=False,
  breathing=False)` から。Hotkeys は `hotkey(name, action, file, key, button=-1)` から：
  ToggleExpression（file = exp3 のベース名）、TriggerAnimation（モーションのファイル）、RemoveAllExpressions。
  主な切り替えにはキー N1〜N8／画面上のボタン 1〜8、顔の表情にはキーなし（ユーザーが割り当てる）。
  PhysicsSettings {Use, Live2DPhysicsFPS 3, PhysicsStrength 50, WindStrength 0}。ModelID と HotkeyIDs
  は前のファイルから引き継ぐ（ほかのモデルの .vtube.json は決して写さない）。
- 写す価値のあるトラッキング：FaceAngleY +-20 -> ParamAngleY +-30。EyeRightX -> ParamEyeBallX を反転（出力 1, -1）。
  MouthSmile 0.55〜1 -> EyeL/RSmile 0〜1、0.6〜1 -> ParamCheek 0〜0.6。Brows 0〜1 -> BrowL/RY -1..1。
  FaceAngleX/Y/Z -> BodyAngleX/Y/Z +-10、smoothing 28。"Auto Breath"（入力 ""、breathing=True）-> ParamBreath。

`check()`（静的な確認）：model3 / vtube.json が名指すファイルがすべてある。物理演算、表情、モーション、トラッキング、
グループが使うパラメータがすべて moc にある。ホットキーが名指す exp3 がある。ランダムな40ポーズで頂点が有限。
`--install` にはこれが要る。

`--install`（`install()`）：VTube Studio が動いているあいだは断る（`ps -axo comm` に "vtube" が含まれるとき）。
`out/<MODEL>/` を `.../VTubeStudio.app/Contents/Resources/Data/StreamingAssets/Live2DModels/<FOLDER>/` に写す。
組み立てで作ったファイルは置き換える。そこに既にある `<MODEL>.vtube.json`（VTS がユーザーの編集をそこに保存する）
は決して上書きしない：`merge_settings()` がそれを `backups/<MODEL>.vtube.<time>.json` に控えてから、
足りないホットキー（Action + File で見る）とトラッキングの対応（出力パラメータで見る）だけを足す。そのとき、
ぶつかる画面上のボタンは外す。待機アニメーションは、何も設定されていないときだけ足す。
新しい組み立てにもうないテクスチャのページは、消さずに `backups/replaced_textures_<time>/` へ移す。

## 9. build_c3.py のパターン（抜粋）

**9.1 骨組み**（sys.path にはそのフォルダと tools/ が入っている。BODY_AX、body_field、NECK、
ROLL は build() より上で定義してある）
```python
from l2d.rig import Rig
import layerio
MODEL, CW, CH = "Chara3", 4532, 8000
K3 = [-1, 0, 1]
def build():
    rig = Rig(CW, CH)
    P = rig.param
    P("ParamAngleZ", -30, 30, 0, "角度 Z", "頭"); P("ParamHairFront", -1, 1, 0, "前髪", "揺れもの")
    for pid, nm in (("PartBody", "体"), ("PartFace", "顔"), ("PartHair", "前髪")):
        rig.part(pid, nm)
    L = {n: layerio.load_layer(n) for n in layerio.manifest()}            # name -> (rgba, x0, y0)
    for n, (img, x0, y0) in L.items():
        rig.layer(n, img, x0, y0, 0.5 if n.startswith(("body", "arm_")) else 1.0)
    wbody = rig.warp("W_Body", None, "PartBody", (0, 0, CW, CH), 12, 20, BODY_AX, body_field)
    rig.layer_mesh("body", wbody, "PartBody", z=20, cell=64, pad=2)
    rhead = rig.rot("R_Head", wbody, "PartFace", tuple(NECK), [("ParamAngleZ", [-30, 0, 30])],
                    angle=lambda v: v["ParamAngleZ"] / 30.0 * ROLL)       # 度。+ = 画面上で時計回り
    ...                                     # rhead の上に頭のワープ、その上にメッシュ、物理演算、表情
    return rig
if __name__ == "__main__":
    out = os.path.join(HERE, "out", MODEL); build().compile(out, MODEL, atlas_size=4096)
    from l2d import core
    m = core.Model(os.path.join(out, MODEL + ".moc3")); print(len(m.drawable_ids), len(m.param_ids))   # 整合性チェック
```

**9.2 首振りのワープ**（レリーフとしての回転。build_c3 の `turn` は、うなずき／見上げ用の別の奥行きも取る）
```python
YAW, PITCH, CORNER = rad(36.0), rad(26.0), 0.66          # 回し切り「かつ」うなずき切ったとき：それぞれ 0.66 倍
def head_angles(v):
    ax, ay = v["ParamAngleX"] / 30.0, v["ParamAngleY"] / 30.0
    k = 1.0 - (1.0 - CORNER) * abs(ax) * abs(ay)
    return ax * YAW * k, ay * PITCH * k
def turn(zfun):                            # zfun(pts) -> 見る人の方への奥行き px。PIV：首振りとうなずきの中心
    def fn(pts, v):
        th, ph = head_angles(v)
        z = zfun(pts)
        x, y = pts[:, 0] - PIV[0], pts[:, 1] - PIV[1]
        x1 = x * (1.0 - (1.0 - np.cos(th)) * fore(pts)) + z * np.sin(th)   # fore：輪郭では遠近の縮みを「残す」
        z1 = -x * np.sin(th) + z * np.cos(th)
        y1 = y * np.cos(ph) - z1 * np.sin(ph)
        return np.stack([x1 + PIV[0], y1 + PIV[1]], 1)
    return fn
AX = [("ParamAngleX", [-30, 0, 30]), ("ParamAngleY", [-30, 0, 30])]
HB = (C[0] - 1050, C[1] - 700, C[0] + 1050, CHIN + 380)                  # 頭のどの面にも同じ箱1つ
wback = rig.warp("W_HairBack", rhead, "PartHairBack", HB, 30, 26, AX, turn(z_back))
```
面ごとにワープを1つ（後ろ髪、耳、顔、前髪、ゴーグル）。それぞれ自分の奥行きを持つ。奥行きが 1 px あたり約 cot(angle)
より速く変わる所で、レイヤーは折れる（横方向：首振り、縦方向：うなずき）。
build_c3 は傾きに上限を設ける（`lip_x` / `lip_y`）。

**9.3 髪の束のグループ：物理演算で動く揺れのキーフォーム**
```python
def group_sway(fld, gi, piv, deg_root, deg_tip, p_root, p_tip, d_root=(60.0, 420.0), d_tip=(260.0, 720.0), gain=None):
    piv = np.asarray(piv, float)
    def fn(pts, v):
        wgt = fld(pts)[:, gi]                                  # このグループの、ぼかして正規化した重み
        d = np.hypot(pts[:, 0] - piv[0], pts[:, 1] - piv[1])
        root = sstep((d - d_root[0]) / (d_root[1] - d_root[0])) ** 1.15
        tip = sstep((d - d_tip[0]) / (d_tip[1] - d_tip[0])) ** 1.5
        return rot_about(pts, piv, rad(HAIR_GAIN if gain is None else gain) * wgt
                         * (deg_root * root * v[p_root] + deg_tip * tip * v[p_tip]))
    return fn
ffld = lock_fields(L["hair_front"], lambda mx: 0 if mx < C[0] - side_x else (2 if mx > C[0] + side_x else 1),
                   C[1] + 60, sigma=FRONT_FIELD_SIGMA)            # レイヤー上の3グループの重み
wfront = rig.warp("W_HairFront", rhead, "PartHair", HB, 30, 26, AX, turn_hair())
wfb = rig.warp("W_HairFrontBounce", wfront, "PartHair", FRECT, 16, 16, [("ParamHairBounce", K3)], front_bounce)
wsl = rig.warp("W_HairSideL", wfb, "PartHair", FRECT, 30, 30, [("ParamHairSideL", K3), ("ParamHairSideLTip", K3)],
               group_sway(ffld, 0, (C[0] - 0.95 * R, C[1] - 0.15 * R), 3.4, 4.6, "ParamHairSideL", "ParamHairSideLTip",
                          (60.0, 380.0), (240.0, 700.0)))
# ... W_HairFringe（グループ 1）、W_HairSideR（グループ 2）、W_HairLocksL/C/R（束ごとに揺れ1つ）も同じようにつなぐ
rig.layer_mesh("hair_front", wlk, "PartHair", z=50, cell=36, pad=2)     # メッシュは鎖の端にぶら下がる
rig.add_physics("横髪 左", head_in(8, 34, True), [("ParamHairSideL", 1, 3.4, 100, "Angle"),
                ("ParamHairSideLTip", 2, 3.8, 100, "Angle")], V3(11, 0.88, 0.70, 1.1), NP, NA)
```
r px 離れたピボットのまわりに角度 a 回すと、点は a*r px 動く。長さ l の重みの傾斜をまたぐと、変化は a*r/l。
これを約 0.6 未満に抑えないと、2つのグループの間の髪が折れる（だから広くぼかす：sigma 64、
後ろ髪では 500 px の継ぎ目）。

**9.4 物理演算の宣言**
```python
V2 = lambda rad_, mob, delay, acc: [(0, 0, 1.0, 1.0, 1.0, 0), (0, rad_, mob, delay, acc, rad_)]
V3 = lambda rad_, mob, delay, acc: [(0, 0, 1.0, 1.0, 1.0, 0), (0, rad_, mob, delay, acc, rad_),
                                    (0, rad_ * 2, mob * 0.95, delay * 0.9, acc, rad_)]
NP, NA, A = (-3, 0, 3), (-20, 0, 20), "Angle"
def head_in(breath=0, wind=0, reflect=False):
    r = [("ParamAngleX", 60, "X"), ("ParamAngleZ", 60, "Angle"), ("ParamBodyAngleX", 25, "X"), ("ParamBodyAngleZ", 25, "Angle")]
    if breath: r.append(("ParamBreath", breath * HAIR_WIND, "Angle", reflect))    # reflect：左右で揺れをずらす
    if wind: r.append(("ParamWind", wind * HAIR_WIND, "Angle"))                    # 書くのは待機モーションだけ
    return r
rig.add_physics("前髪", head_in(9, 30), [("ParamHairFront", 1, 3.6, 100, A), ("ParamHairFrontTip", 2, 3.8, 100, A)],
                V3(8, 0.90, 0.62, 1.2), NP, NA)
rig.add_physics("耳 右", [("ParamAngleZ", 60, "Angle", True), ("ParamAngleY", 45, "X"), ("ParamAngleX", 20, "X", True),
                ("ParamBodyAngleZ", 20, "Angle", True)], [("ParamEarSwingR", 1, 2.2, 100, A)], V2(4, 0.93, 0.58, 1.7), NP, NA)
```
硬く速いもの（耳、ペンダント、クリスタル、アホ毛）：radius 4〜7、mobility 0.92〜0.93、delay 0.55〜0.65、
acceleration 1.5〜1.7。長く重いもの（横髪／後ろ髪、腕、袖）：radius 11〜13.5、mobility 0.82〜0.90、delay 0.70〜0.92、
acceleration 0.8〜1.1。左右で radius / delay を少し変え、決して同じ調子で揺れないようにする。

**9.5 切り替え：入れ替えのキーと滑り**（ゴーグルを外す、ゴーグルを目に。腕のポーズも同じしくみ）
```python
GOG_KEYS = [0.0, 0.5, 0.51, 0.85, 1.0]      # ParamGogglesEyes：0.5 まで滑り、0.5/0.51 で入れ替え、0.85 までに落ち着く
GOG_OFF_KEYS = [0.0, 0.49, 0.5, 1.0]        # ParamGoggles（1 = 着用）：1 -> 0.5 で持ち上がり、0.5 未満で一度に消える
def gog_slide(pts, v):
    out = pts.copy()
    out[:, 1] += GOG_SLIDE * sstep(min(v["ParamGogglesEyes"], 0.5) / 0.5)       # 額から下へ
    out[:, 1] -= GOG_LIFT * sstep((1.0 - max(v["ParamGoggles"], 0.5)) / 0.5)    # 消える前に上へ
    return out
rig.layer_mesh("goggles", wgog, "PartAcc", z=53, cell=32, pad=2,
               axes=[("ParamGoggles", GOG_OFF_KEYS), ("ParamGogglesEyes", GOG_KEYS)], fn=gog_slide,
               opacity=lambda v: (1.0 if v["ParamGoggles"] >= 0.5 else 0.0) * (1.0 if v["ParamGogglesEyes"] <= 0.5 else 0.0))
rig.layer_mesh("goggles_eyes", wface, "PartFace", z=49.9, cell=28, pad=2, axes=[("ParamGogglesEyes", GOG_KEYS)],
               fn=gog_eyes, opacity=lambda v: 1.0 if v["ParamGogglesEyes"] > 0.5 else 0.0)   # 上から入ってくる
```

**9.6 別のメッシュでマスクするメッシュ**
```python
rig.layer_mesh(f"sclera_{side}", ew, "PartEye", z=43, cell=10, pad=4, axes=[(OP, EYE_KEYS), (SM, [0, 1])],
               fn=lid_map, opacity=seen)                              # OP / SM = ParamEye<side>Open / Smile
for k, nm in enumerate(("iris", "irisS")):                            # 普通の黒目／しいたけ目の黒目
    rig.layer_mesh(f"{nm}_{side}", wiris, "PartEye", z=44 + 0.1 * k, cell=12, pad=4,
                   axes=[("ParamEyeType", [0, 1]), (OP, EYE_KEYS)],
                   opacity=lambda v, k=k, seen=seen: (1.0 if abs(v["ParamEyeType"] - k) < 0.5 else 0.0) * seen(v),
                   masks=[f"sclera_{side}"])                          # 白目の中にだけ描く
rig.layer_mesh("fx_gloom", wface, "PartFace", z=49.5, cell=60, pad=2, axes=[("ParamGloom", [0, 1])],
               opacity=lambda v: v["ParamGloom"], masks=["face"])
```

**9.7 親が運ぶ先にぴったり着かなければならない子**（W_Face の下の W_Mouth、どちらも AX の上）
```python
Fz = turn_face(z_hair, nod_l("face"), ear=False, znod_up=nod_l("face_up"))     # W_Face 自身の fn
def keep_thick(pts, v):                       # 頭がうなずいても口の厚みを保つ
    c = np.stack([pts[:, 0], np.full(len(pts), mcy)], 1); Fc = Fz(c, v)       # 口の中心線を回したもの
    n_ = Fz(c + np.array([0.0, 1.0]), v) - Fc; n_ /= np.maximum(np.linalg.norm(n_, axis=1, keepdims=True), 1e-6)
    T = Fc + (pts[:, 1] - mcy)[:, None] * n_                    # 各点が最後に「着く」べき所
    q = pts.astype(float).copy()
    for _ in range(5):                                          # ニュートン法：Fz(q) = T
        Fq = Fz(q, v); e = T - Fq
        jx = Fz(q + np.array([1.0, 0.0]), v) - Fq; jy = Fz(q + np.array([0.0, 1.0]), v) - Fq
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        q[:, 0] += (jy[:, 1] * e[:, 0] - jy[:, 0] * e[:, 1]) / det
        q[:, 1] += (-jx[:, 1] * e[:, 0] + jx[:, 0] * e[:, 1]) / det
    return q                                                    # W_Face の静止時の座標で
wm = rig.warp("W_Mouth", wface, "PartFace", MRECT, 6, 6, AX, keep_thick)      # キーの上では正確
```

**9.8 表情**
```python
rig.expressions["Eye_Shiitake"] = {"params": {"ParamEyeType": 1}, "fade": 0.1}         # 値だけ = Overwrite
BR = lambda y, a: {"ParamBrowLY": (y, "Add"), "ParamBrowRY": (y, "Add"), "ParamBrowLAngle": (a, "Add"), "ParamBrowRAngle": (a, "Add")}
EO = lambda k: {"ParamEyeLOpen": (k, "Multiply"), "ParamEyeROpen": (k, "Multiply")}
EAR = lambda k: {"ParamEarL": (k, "Add"), "ParamEarR": (k, "Add")}
rig.expressions["Face_Angry"] = {"params": {**BR(-0.7, 0.8), **EO(0.82), "ParamMouthForm": (-0.8, "Add"), **EAR(-0.55),
                                            "ParamFxAnger": 1.0}, "fade": 0.2}           # 下でトラッキングが続く
```

**9.9 モーションのカーブ**（package_c3.py）
```python
def count(curve_list):                        # 線形セグメントのときの Meta の各数
    seg = sum((len(c["Segments"]) - 2) // 3 for c in curve_list); n = len(curve_list)
    return {"CurveCount": n, "TotalSegmentCount": seg, "TotalPointCount": seg + n}
def build_motion(fn, dur):                    # fn(t) -> {param id: value}; STEP = 0.05
    n = int(round(dur / STEP))
    ts = [round(i * STEP, 4) for i in range(n + 1)]
    out = []
    for pid in sorted(fn(0.0)):
        seg = []
        for j, t in enumerate(ts):
            val = round(fn(t)[pid], 4)
            seg += [t, val] if j == 0 else [0, t, val]        # 0 = (t, val) までの線形セグメント
        out.append({"Target": "Parameter", "Id": pid, "Segments": seg})
    meta = {"Duration": dur, "Fps": 30.0, "Loop": False, "AreBeziersRestricted": True, "FadeInTime": 0.0, "FadeOutTime": 0.0}
    meta.update(count(out)); meta.update({"UserDataCount": 0, "TotalUserDataSize": 0})
    return {"Version": 3, "Meta": meta, "Curves": out}
json.dump(build_motion(peace_curves, PEACE_DUR), open(os.path.join(SRC, "animations", PEACE), "w"), separators=(",", ":"))
```
`peace_curves` / `wave_curves`：切り替えを0.45秒で上げ、保ち、0.5秒で下げる。それに包絡線を掛けたサイン波を加える。
待機のループ：同じものを `"Loop": True` にし、最後のキー = 最初のキー。
