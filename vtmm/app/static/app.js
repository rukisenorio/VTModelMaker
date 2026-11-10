// VTModelMaker - the page. Plain JavaScript, no libraries.
"use strict";

const LM = [  // [key, label, instruction, optional]
  ["eye_l", "目（向かって左）", "向かって左の目の、黒目の中心をクリック", false],
  ["eye_r", "目（向かって右）", "向かって右の目の、黒目の中心をクリック", false],
  ["brow_l", "眉（向かって左）", "向かって左の眉の真ん中をクリック（前髪で見えなければ「なし」）", true],
  ["brow_r", "眉（向かって右）", "向かって右の眉の真ん中をクリック（見えなければ「なし」）", true],
  ["nose", "鼻", "鼻の先をクリック", false],
  ["mouth", "口", "口の線の真ん中をクリック", false],
  ["chin", "あご", "あごの先（顔の輪郭のいちばん下）をクリック", false],
  ["face_l", "顔の輪郭（向かって左）", "目の高さで、向かって左の顔の輪郭（ほほの端）をクリック", false],
  ["face_r", "顔の輪郭（向かって右）", "目の高さで、向かって右の顔の輪郭（ほほの端）をクリック", false],
  ["crown", "頭のてっぺん", "髪がなかったとしたら頭の上の端はどこか、見当でクリック（髪のてっぺんより少し下）", false],
  ["hair_top", "髪のいちばん上", "髪（帽子をかぶっていれば帽子）のいちばん上をクリック", false],
  ["ear_l", "耳（向かって左）", "向かって左の耳の真ん中をクリック（髪で隠れていれば「なし」）", true],
  ["ear_r", "耳（向かって右）", "向かって右の耳の真ん中をクリック（隠れていれば「なし」）", true],
  ["neck", "首", "首の前側、襟と接するところの真ん中をクリック", false],
  ["hair_bottom", "髪のいちばん下", "いちばん長い毛先をクリック（短い髪なら「なし」）", true],
  ["shoulder_l", "肩（向かって左）", "向かって左の肩の付け根をクリック", true],
  ["shoulder_r", "肩（向かって右）", "向かって右の肩の付け根をクリック", true],
  ["elbow_l", "ひじ（向かって左）", "向かって左のひじをクリック（絵になければ「なし」）", true],
  ["elbow_r", "ひじ（向かって右）", "向かって右のひじをクリック（なければ「なし」）", true],
  ["wrist_l", "手首（向かって左）", "向かって左の手首をクリック（なければ「なし」）", true],
  ["wrist_r", "手首（向かって右）", "向かって右の手首をクリック（なければ「なし」）", true],
  ["hip", "腰", "腰の真ん中（股の高さ）をクリック（上半身だけの絵なら「なし」）", true],
  ["knee_l", "ひざ（向かって左）", "向かって左のひざをクリック（なければ「なし」）", true],
  ["knee_r", "ひざ（向かって右）", "向かって右のひざをクリック（なければ「なし」）", true],
  ["foot_l", "足（向かって左）", "向かって左の足の裏をクリック（なければ「なし」）", true],
  ["foot_r", "足（向かって右）", "向かって右の足の裏をクリック（なければ「なし」）", true],
];
const DESC = [
  ["character", "どんなキャラか", "例: 落ち着いた表情の少年"],
  ["hair", "髪", "例: 毛先のはねた銀髪のショート、左に赤いメッシュ"],
  ["eyes", "目", "例: 左右で色の違う目（向かって左が赤、右が薄い灰色）"],
  ["skin", "肌", "例: 色白"],
  ["outfit", "服・持ち物", "例: 裏地の赤い黒のコート、赤いリボンタイ、黒の短パン、ブーツ"],
];
const STEPS = ["位置合わせ", "説明", "切り抜き", "下絵", "組み立て", "確認・インストール", "高品質"];

let ENV = {}, P = null, STEP = 1, POINTS = {}, CUR = 0, IMG = null, IMGSCALE = 1;
let VIEW = {s: 1, x: 0, y: 0}, DRAG = null, LASTJOB = null, LOGOPEN = true;

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  let j = {};
  try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}
const fileUrl = (rel, max) => `/api/p/${P.name}/file?path=${encodeURIComponent(rel)}${max ? "&max=" + max : ""}&t=${Date.now()}`;
function readFile(f) { return new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = bad; r.readAsDataURL(f); }); }

// ------------------------------------------------------------------------------------------------ environment
async function loadEnv() {
  ENV = await api("/api/env");
  const c = [];
  c.push(ENV.vts_models_dir ? ["ok", "VTube Studio: 見つかりました"] : ["bad", "VTube Studio: 見つかりません"]);
  c.push(ENV.core_lib ? ["ok", "確認用エンジン: あり"] : ["bad", "確認用エンジン: なし"]);
  c.push(ENV.imagegen === "codex" ? ["ok", "下絵: 自動（Codex）"] : ["warn", "下絵: 手作業（ChatGPT）"]);
  if (ENV.vts_running) c.push(["warn", "VTube Studio 起動中"]);
  $("envchips").innerHTML = c.map(([k, t]) => `<span class="chip ${k}">${t}</span>`).join("");
}

// -------------------------------------------------------------------------------------------------------- home
async function goHome() {
  P = null;
  $("home").classList.remove("hidden"); $("proj").classList.add("hidden");
  const list = await api("/api/projects");
  $("plist").innerHTML = list.length ? list.map(p => `<div class="pitem" onclick="openProject('${p.name}')"><b>${p.name}</b><br>
    <span class="muted">${p.installed ? "VTube Studio に入っています" : p.built ? "モデルができています" : p.prepped ? "作業中" : p.points_done ? "位置合わせ済み" : "位置合わせから"}</span></div>`).join("")
    : `<span class="muted">まだありません</span>`;
}
async function newProject() {
  const name = $("newname").value.trim(), f = $("newfile").files[0];
  if (!/^[A-Za-z0-9_\-]{1,40}$/.test(name)) { $("newmsg").textContent = "名前は半角の英数字と - _ だけにしてください"; return; }
  if (!f) { $("newmsg").textContent = "絵のファイルを選んでください"; return; }
  $("newbtn").disabled = true; $("newmsg").textContent = "準備しています…";
  try {
    await api("/api/projects", {name, filename: f.name, image: await readFile(f)});
    $("newmsg").textContent = ""; await openProject(name);
  } catch (e) { $("newmsg").textContent = "できませんでした: " + e.message; }
  $("newbtn").disabled = false;
}
function openFolder(name, sub) { api("/api/open", {project: name || null, sub: sub || null}).catch(() => {}); }

// ----------------------------------------------------------------------------------------------------- project
async function openProject(name) {
  $("home").classList.add("hidden"); $("proj").classList.remove("hidden");
  P = await api(`/api/p/${name}/state`);
  POINTS = Object.assign({}, P.cfg.points || {});
  NONE = {}; (P.cfg.points_none || []).forEach(k => { if (!POINTS[k]) NONE[k] = true; });
  $("ptitle").textContent = name; $("pdir").textContent = P.dir; $("pname2").textContent = name;
  IMG = null;
  const im = new Image();
  im.onload = () => { IMG = im; IMGSCALE = P.source_size[0] / im.width; fitView(); };
  im.src = fileUrl("src/art.png", 2400);
  buildDesc();
  CUR = LM.findIndex(([k]) => !POINTS[k] && !NONE[k]);
  if (CUR < 0) CUR = 0;
  showStep(guessStep());
}
async function refresh() {
  if (!P) return;
  P = await api(`/api/p/${P.name}/state`);
  renderTabs(); renderStep();
}
function guessStep() {
  if (!P.points_done) return 1;
  if (!P.prepped) return 3;
  if (P.built) return 6;
  if (P.plates && (P.plates.waiting || []).length) return 4;
  if (P.plates) return 5;
  return 3;
}
function renderTabs() {
  const done = [P.points_done, !!(P.cfg.describe && Object.keys(P.cfg.describe).length), P.prepped,
                P.plates && !(P.plates.waiting || []).length, P.built, P.installed, !!(P.hq && P.hq.review)];
  $("tabs").innerHTML = STEPS.map((t, i) => `<span class="tab ${STEP === i + 1 ? "on" : ""} ${done[i] ? "done" : ""}" onclick="showStep(${i + 1})">${i + 1}. ${t}</span>`).join("");
}
function showStep(n) {
  STEP = n;
  for (let i = 1; i <= 7; i++) $("step" + i).classList.toggle("hidden", i !== n);
  renderTabs(); renderStep();
}
function renderStep() {
  if (STEP === 1) { renderLM(); draw(); }
  if (STEP === 3) renderPrep();
  if (STEP === 4) renderPlates();
  if (STEP === 6) renderPreviews();
  if (STEP === 7) renderHQ();
  $("descai").disabled = !ENV.codex;
  $("aiptbtn").disabled = !(ENV.claude || ENV.codex);
}

// ----------------------------------------------------------------------------------------------- landmarks
function renderLM() {
  $("lmlist").innerHTML = LM.map(([k, label, , opt], i) => {
    const v = POINTS[k];
    return `<div class="lmrow ${i === CUR ? "cur" : ""} ${v ? "set" : ""}" onclick="pickLM(${i})">${label}${opt ? "" : ' <span class="req">必須</span>'}<span class="st">${v ? "✔" : NONE[k] ? "なし" : ""}</span></div>`;
  }).join("");
  const [k, label, instr, opt] = LM[CUR];
  $("instr").textContent = `【${label}】 ${instr}`;
  $("nonebtn").disabled = !opt;
}
let NONE = {};
function pickLM(i) { CUR = i; renderLM(); draw(); }
function setNone() { const [k, , , opt] = LM[CUR]; if (!opt) return; POINTS[k] = null; NONE[k] = true; nextLM(); }
function nextLM() {
  for (let j = 1; j <= LM.length; j++) {
    const i = (CUR + j) % LM.length, k = LM[i][0];
    if (!POINTS[k] && !NONE[k]) { CUR = i; break; }
  }
  renderLM(); draw();
}
async function savePoints() {
  const miss = LM.filter(([k, , , opt]) => !opt && !POINTS[k]).map(x => x[1]);
  try {
    await api(`/api/p/${P.name}/points`, {points: POINTS, none: Object.keys(NONE).filter(k => NONE[k])});
    $("ptmsg").textContent = miss.length ? "保存しました（まだ必須の点があります: " + miss.join("・") + "）" : "保存しました";
    await refresh();
  } catch (e) { $("ptmsg").textContent = "保存できませんでした: " + e.message; }
}

// canvas: the picture with pan / zoom; points in SOURCE px
const cv = () => $("cv");
function fitView() {
  if (!IMG) return;
  const c = cv(), w = c.clientWidth, h = c.clientHeight;
  c.width = w; c.height = h;
  VIEW.s = Math.min(w / IMG.width, h / IMG.height) * 0.96;
  VIEW.x = (w - IMG.width * VIEW.s) / 2; VIEW.y = (h - IMG.height * VIEW.s) / 2;
  draw();
}
function zoomBy(f, cx, cy) {
  const c = cv();
  cx = cx === undefined ? c.width / 2 : cx; cy = cy === undefined ? c.height / 2 : cy;
  VIEW.x = cx - (cx - VIEW.x) * f; VIEW.y = cy - (cy - VIEW.y) * f; VIEW.s *= f; draw();
}
function zoomFace() {
  if (!IMG) return;
  const keys = ["eye_l", "eye_r", "nose", "mouth", "chin", "face_l", "face_r"].filter(k => POINTS[k]);
  let cx, cy, span;
  if (keys.length >= 2) {
    const xs = keys.map(k => POINTS[k][0] / IMGSCALE), ys = keys.map(k => POINTS[k][1] / IMGSCALE);
    cx = (Math.min(...xs) + Math.max(...xs)) / 2; cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 2.6 + 40;
  } else if (P.cfg.head_guess) {
    const [x0, y0, x1, y1] = P.cfg.head_guess.map(v => v / IMGSCALE);
    cx = (x0 + x1) / 2; cy = y0 + (y1 - y0) * 0.35; span = (y1 - y0) * 0.45;
  } else { cx = IMG.width / 2; cy = IMG.height * 0.12; span = Math.min(IMG.width, IMG.height) * 0.5; }
  const c = cv();
  VIEW.s = Math.min(c.width, c.height) / span;
  VIEW.x = c.width / 2 - cx * VIEW.s; VIEW.y = c.height / 2 - cy * VIEW.s; draw();
}
function draw() {
  const c = cv(); if (!c || !IMG) return;
  if (c.width !== c.clientWidth || c.height !== c.clientHeight) { c.width = c.clientWidth; c.height = c.clientHeight; }
  const g = c.getContext("2d");
  g.fillStyle = "#2a2a30"; g.fillRect(0, 0, c.width, c.height);
  g.imageSmoothingEnabled = VIEW.s < 2;
  g.drawImage(IMG, VIEW.x, VIEW.y, IMG.width * VIEW.s, IMG.height * VIEW.s);
  g.font = "12px sans-serif";
  LM.forEach(([k, label], i) => {
    const v = POINTS[k]; if (!v) return;
    const x = VIEW.x + v[0] / IMGSCALE * VIEW.s, y = VIEW.y + v[1] / IMGSCALE * VIEW.s;
    g.beginPath(); g.arc(x, y, i === CUR ? 7 : 5, 0, 7);
    g.fillStyle = i === CUR ? "#ff3b6b" : "#3bd16f"; g.fill(); g.lineWidth = 2; g.strokeStyle = "#fff"; g.stroke();
    g.fillStyle = "#fff"; g.strokeStyle = "#000"; g.lineWidth = 3;
    g.strokeText(label, x + 9, y - 6); g.fillText(label, x + 9, y - 6);
  });
}
function setupCanvas() {
  const c = cv();
  c.addEventListener("mousedown", e => { DRAG = {x: e.offsetX, y: e.offsetY, vx: VIEW.x, vy: VIEW.y, moved: false}; });
  c.addEventListener("mousemove", e => {
    if (!DRAG) return;
    const dx = e.offsetX - DRAG.x, dy = e.offsetY - DRAG.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) DRAG.moved = true;
    if (DRAG.moved) { VIEW.x = DRAG.vx + dx; VIEW.y = DRAG.vy + dy; draw(); }
  });
  window.addEventListener("mouseup", e => {
    if (!DRAG) return;
    const d = DRAG; DRAG = null;
    if (d.moved || e.target !== c || !IMG) return;
    const sx = (e.offsetX - VIEW.x) / VIEW.s * IMGSCALE, sy = (e.offsetY - VIEW.y) / VIEW.s * IMGSCALE;
    if (sx < 0 || sy < 0 || sx > P.source_size[0] || sy > P.source_size[1]) return;
    const k = LM[CUR][0];
    POINTS[k] = [Math.round(sx * 10) / 10, Math.round(sy * 10) / 10]; delete NONE[k];
    nextLM();
  });
  c.addEventListener("wheel", e => { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.offsetX, e.offsetY); }, {passive: false});
  window.addEventListener("resize", () => { if (STEP === 1) draw(); });
}

// ------------------------------------------------------------------------------------------------- describe
function buildDesc() {
  const d = (P.cfg.describe) || {};
  $("descfields").innerHTML = DESC.map(([k, label, ex]) => `<div><div class="muted">${label}</div><input type="text" id="d_${k}" placeholder="例: ${ex}" value="${(d[k] || "").replace(/"/g, "&quot;")}"></div>`).join("");
}
async function saveDescribe() {
  const describe = {}; DESC.forEach(([k]) => describe[k] = $("d_" + k).value);
  await api(`/api/p/${P.name}/describe`, {describe}); $("descmsg").textContent = "保存しました"; await refresh();
}
async function describeAI() {
  await saveDescribe();
  try { await api(`/api/p/${P.name}/describe_ai`, {}); AFTER = async () => { await refresh(); buildDesc(); }; } catch (e) { alert(e.message); }
}

// ----------------------------------------------------------------------------------------------------- prep
function renderPrep() {
  $("prepres").classList.toggle("hidden", !P.prepped);
  if (!P.prepped) return;
  $("figlook").src = fileUrl("work/figure_look.png", 1600);
  $("pocketsimg").src = fileUrl("work/pockets.png", 1600);
  $("pocketsheet").src = fileUrl("work/pockets_sheet.png", 1600);
  const chosen = new Set((P.cfg.bg_pockets || []).map(p => p.join(",")));
  $("pocketlist").innerHTML = (P.pockets || []).length ? P.pockets.map(c => `<label><input type="checkbox" value="${c.i}" ${chosen.has(c.point.join(",")) ? "checked" : ""}> ${c.i}番</label>`).join("")
    : `<span class="muted">候補はありません</span>`;
}
async function savePockets() {
  const selected = [...document.querySelectorAll("#pocketlist input:checked")].map(x => +x.value);
  await api(`/api/p/${P.name}/pockets`, {selected});
  await runStage("prep", "prep", "切り抜き（作り直し）");
}

// --------------------------------------------------------------------------------------------------- plates
function renderPlates() {
  document.querySelectorAll('input[name=igm]').forEach(r => r.checked = (r.value === ENV.imagegen));
  const manual = ENV.imagegen !== "codex";
  $("platenote").innerHTML = manual
    ? "手作業モード: 下の各カードで、<b>入力の絵</b>を保存して ChatGPT に添付し、<b>文章をコピー</b>して貼り付けて送ってください。できた絵を保存して、カードの枠に入れます。全部入れたら（または使わないものを選んだら）「下絵を作る（続きから）」を押してください。"
    : "自動モード: Codex が下絵を順に描きます（全部で10〜30分）。終わったら「組み立て」へ進みます。";
  const st = P.plates;
  $("platestatus").textContent = st ? `できた下絵: ${(st.done || []).length} 枚 / 待っている: ${(st.waiting || []).length} 枚 / 使わない: ${(st.skipped || []).length} 枚` : "まだ始めていません";
  $("skipallrow").classList.toggle("hidden", !(P.requests || []).some(r => !r.required));
  $("reqs").innerHTML = (P.requests || []).map(r => {
    if (r.blocked_by) return `<div class="req-card"><b>${r.ja}</b> <span class="badge opt">あとで</span><div class="muted">先に「${(st.ja || {})[r.blocked_by] || r.blocked_by}」が要ります。</div></div>`;
    if (!r.has_request) return `<div class="req-card"><b>${r.ja}</b><div class="muted">「下絵を作る」を押すと依頼ができます。</div></div>`;
    return `<div class="req-card"><b>${r.ja}</b>${r.required ? '<span class="badge">必須</span>' : '<span class="badge opt">あるとよい</span>'}
      ${r.has_input ? `<img src="${fileUrl("gen/requests/" + r.name + "/input.png", 600)}"><a href="${fileUrl("gen/requests/" + r.name + "/input.png")}" download="${r.name}_input.png">入力の絵を保存</a>` : ""}
      <textarea readonly id="pr_${r.name}">${r.prompt.replace(/</g, "&lt;")}</textarea>
      <div class="row"><button class="sec small" onclick="copyPrompt('${r.name}')">文章をコピー</button>
      ${r.required ? "" : `<button class="sec small" onclick="skipPlate('${r.name}')">使わない</button>`}</div>
      <div class="drop" id="dr_${r.name}" ondragover="event.preventDefault();this.classList.add('hover')" ondragleave="this.classList.remove('hover')" ondrop="dropPlate(event,'${r.name}')">
        できた絵をここにドロップ／<label style="color:var(--accent);cursor:pointer">選ぶ<input type="file" accept="image/*" class="hidden" onchange="uploadPlate('${r.name}', this.files[0])"></label></div></div>`;
  }).join("");
}
async function setImagegen(v) { await api("/api/settings", {imagegen: v}); await loadEnv(); renderPlates(); }
function copyPrompt(n) { const t = $("pr_" + n); t.select(); navigator.clipboard ? navigator.clipboard.writeText(t.value) : document.execCommand("copy"); }
async function uploadPlate(n, f) {
  if (!f) return;
  $("dr_" + n).textContent = "入れています…";
  try { await api(`/api/p/${P.name}/plate/${n}`, {image: await readFile(f)}); $("dr_" + n).textContent = "入れました ✓"; }
  catch (e) { $("dr_" + n).textContent = "入れられませんでした: " + e.message; }
}
function dropPlate(e, n) { e.preventDefault(); const f = e.dataTransfer.files[0]; uploadPlate(n, f); }
async function skipPlate(n) { await api(`/api/p/${P.name}/skip`, {names: [n]}); await refresh(); }
async function skipOptional() { await api(`/api/p/${P.name}/skip`, {names: (P.requests || []).filter(r => !r.required).map(r => r.name)}); await refresh(); }

// -------------------------------------------------------------------------------------------- preview, install
function renderPreviews() {
  const titles = {"preview/poses.png": "いろいろな向きと表情", "preview/holes.png": "すき間の確認（ピンク＝すき間）",
                  "preview/motion.png": "首振り（上から順に時間が進みます）", "preview/blink.png": "まばたき（1コマずつ）"};
  $("previews").innerHTML = (P.previews || []).map(p => `<h3>${titles[p] || p}</h3><a href="${fileUrl(p)}" target="_blank"><img class="preview" src="${fileUrl(p, 2000)}"></a>`).join("")
    || `<span class="muted">まだありません。「組み立て」を先に。</span>`;
  $("instbtn").disabled = !P.built;
}
async function motion(blink) { try { await api(`/api/p/${P.name}/motion`, {blink}); AFTER = refresh; } catch (e) { alert(e.message); } }
async function installVTS() {
  await loadEnv();
  if (ENV.vts_running) { alert("VTube Studio が起動中です。終了してからもう一度押してください。"); return; }
  if (!confirm("VTube Studio に入れます（前に入れた同じ名前のモデルは、設定を残したまま入れ替わります）。よろしいですか？")) return;
  try { await api(`/api/p/${P.name}/install`, {}); AFTER = refresh; } catch (e) { alert(e.message); }
}

// ------------------------------------------------------------------------------------------------- high quality
async function aiPoints() {
  const any = LM.some(([k]) => POINTS[k]);
  if (any && !confirm("AI に位置を置き直してもらいますか？（今の点は上書きされます）")) return;
  try {
    await api(`/api/p/${P.name}/ai_landmarks`, {force: any});
    AFTER = async () => { await refresh(); POINTS = Object.assign({}, P.cfg.points || {}); NONE = {};
      (P.cfg.points_none || []).forEach(k => { if (!POINTS[k]) NONE[k] = true; }); renderLM(); draw();
      $("ptmsg").textContent = "AI が置きました。ずれている点は、選んでクリックし直してから「保存」を押してください。"; };
  } catch (e) { alert(e.message); }
}
async function hq(cmd) {
  try { await api(`/api/p/${P.name}/hq/${cmd}`, {}); AFTER = refresh; }
  catch (e) { alert(e.message.includes("409") ? "ほかの作業が動いています。終わるまで待ってください。" : e.message); }
}
function copyText(t) { navigator.clipboard ? navigator.clipboard.writeText(t) : null; }
function renderHQ() {
  const h = P.hq || {};
  $("hqavail").textContent = ENV.claude ? "" : "（Claude Code が見つからないので、確認は Codex がします）";
  $("hqrefs").innerHTML = (h.refs || []).map(r => `<a href="${fileUrl(r)}" target="_blank"><img src="${fileUrl(r, 300)}" style="height:150px;border-radius:6px;border:1px solid #ddd"></a>`).join("") || '<span class="muted">まだありません</span>';
  const KJ = {yaw_amp: "首の回り方", nod_up_scale: "見上げ方"};
  const cal = h.calibrate ? (h.calibrate.found ? h.calibrate : {found: h.calibrate, unsure: {}}) : null;
  const fnd = cal ? Object.entries(cal.found || {}).map(([k, v]) => `${KJ[k] || k} = ${v}`).join("、") : "";
  const uns = cal ? Object.entries(cal.unsure || {}).map(([k, v]) => `${KJ[k] || k}（いちばん近いのは ${v}）`).join("、") : "";
  const notes = cal && cal.notes ? Object.values(cal.notes) : [];
  $("hqcal").innerHTML = cal ? `<div>見本に合わせた値: ${fnd || "なし"}</div>` +
    (notes.length ? `<ul class="muted" style="margin:4px 0 6px 18px;padding:0">${notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>` : "") +
    (uns ? `<div class="muted">見本に届かなかったので書き込まなかったもの: ${uns}。見本の絵が大げさなのかもしれません。下の比較図で確かめてください。</div>` : "") +
    (h.calibrate_img ? `<div class="muted">比較図（左＝モデル、中＝体の位置でそろえた見本、右＝2枚を重ねた図。赤丸＝モデルの目、青丸＝見本の目）</div><div style="max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px"><a href="${fileUrl(h.calibrate_img)}" target="_blank"><img src="${fileUrl(h.calibrate_img, 1600)}" style="width:100%;display:block"></a></div>` : "") : "";
  $("hqreview").textContent = h.review ? h.review.replace(/^<!--.*?-->\n?/, "") : "（まだ確認していません）";
  $("hqimgs").innerHTML = (h.qa_images || []).map(p => `<h3>${p.split("/").pop()}</h3><a href="${fileUrl(p)}" target="_blank"><img class="preview" src="${fileUrl(p, 1800)}"></a>`).join("");
  $("hqprompt").textContent = `VTModelMaker の ${P.name} を高品質で仕上げて`;
}

// ------------------------------------------------------------------------------------------------------- jobs
let AFTER = null;
async function runStage(from, to, title) {
  try { await api(`/api/p/${P.name}/run`, {from, to, title}); AFTER = refresh; }
  catch (e) { alert(e.message.includes("409") ? "ほかの作業が動いています。終わるまで待ってください。" : e.message); }
}
function toggleLog() { LOGOPEN = !LOGOPEN; $("joblog").classList.toggle("hidden", !LOGOPEN); }
async function pollJob() {
  try {
    const j = await api("/api/job");
    $("jobspin").classList.toggle("hidden", !j.running);
    const t = j.running ? Math.round(Date.now() / 1000 - j.started) : (j.ended ? Math.round(j.ended - j.started) : 0);
    $("jobtitle").textContent = j.title ? `${j.project}: ${j.title}${j.running ? " を実行中…" : j.code === 0 ? " 完了" : j.code === 3 ? " 絵を待っています（手作業モード）" : " 失敗（ログを見てください）"}` : "待機中";
    $("jobtime").textContent = t ? `${Math.floor(t / 60)}分${t % 60}秒` : "";
    const log = $("joblog"), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = (j.lines || []).join("\n");
    if (atEnd) log.scrollTop = log.scrollHeight;
    if (LASTJOB && LASTJOB.running && !j.running) {
      if (AFTER) { const f = AFTER; AFTER = null; await f(); }
      await loadEnv();
      if (P && STEP === 3 && j.code === 0) renderPrep();
      if (P && j.title === "組み立て" && j.code === 0) showStep(6);
      if (P && j.title === "下絵" && j.code === 0) showStep(5);
    }
    LASTJOB = j;
  } catch (e) {}
  setTimeout(pollJob, LASTJOB && LASTJOB.running ? 1000 : 2500);
}

// --------------------------------------------------------------------------------------------------------- go
setupCanvas();
loadEnv().then(goHome);
pollJob();
