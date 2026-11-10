"""The tool's own window, without Claude: a small web server on this computer (http://127.0.0.1:<port>), opened in the
browser. Python's standard library only. One job (a stage chain) runs at a time; the page polls /api/job for its log.

usage: server.py [--port 8765] [--no-browser]

Every project is a folder in <home>/projects/<name> (the same layout the Claude skill uses), so a character started here
can be finished with Claude and the other way round.
"""
import os, sys, json, time, re, base64, threading, subprocess, socket, webbrowser, mimetypes, shutil, io
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

APP = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(APP)
sys.path.insert(0, ENGINE)
import platform_env  # noqa: E402

HOME = platform_env.home() or os.path.dirname(ENGINE)
PROJECTS = platform_env.config().get("projects_dir") or os.path.join(HOME, "projects")
STATIC = os.path.join(APP, "static")
PY = sys.executable
SAFE_DIRS = ("src", "work", "gen", "preview", "out", "layers")
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")

# ------------------------------------------------------------------------------------------------------------- jobs
JOB = {"running": False, "project": None, "title": "", "cmd": None, "lines": [], "code": None, "started": 0.0, "ended": 0.0}
LOCK = threading.Lock()


def _env():
    return dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")


def start_job(project, title, cmd):
    with LOCK:
        if JOB["running"]:
            return False
        JOB.update(running=True, project=project, title=title, cmd=cmd, lines=[], code=None, started=time.time(), ended=0.0)

    def run():
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", env=_env(), cwd=ENGINE,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                line = line.rstrip()
                if not line or "Core Version" in line or "Warning" in line or "warn(" in line:
                    continue
                with LOCK:
                    JOB["lines"].append(line)
                    del JOB["lines"][:-400]
            p.wait()
            code = p.returncode
        except Exception as e:                                  # (the job itself could not start)
            with LOCK:
                JOB["lines"].append(f"エラー: 作業を始められませんでした: {e}")
            code = -1
        with LOCK:
            JOB.update(running=False, code=code, ended=time.time())
    threading.Thread(target=run, daemon=True).start()
    return True


# ------------------------------------------------------------------------------------------------------- projects
def proj_dir(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("キャラの名前が正しくありません（半角の英数字と - _ だけ）")
    return os.path.join(PROJECTS, name)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def kit_of(d):
    k = os.path.join(d, "kit")
    return k if os.path.isdir(k) else ENGINE


def project_state(name):
    d = proj_dir(name)
    cfg = read_json(os.path.join(d, "character.json"), {})
    model = cfg.get("name", name)
    out = os.path.join(d, "out", model)
    pts = cfg.get("points", {})
    status = read_json(os.path.join(d, "gen", "requests", "status.json"), None)
    reqs = []
    if status:
        for n in status.get("waiting", []):
            rd = os.path.join(d, "gen", "requests", n)
            reqs.append({"name": n, "ja": status.get("ja", {}).get(n, n), "required": n in status.get("required", []),
                         "blocked_by": status.get("blocked", {}).get(n), "has_request": os.path.isdir(rd),
                         "prompt": open(os.path.join(rd, "prompt.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(rd, "prompt.txt")) else "",
                         "has_input": os.path.exists(os.path.join(rd, "input.png"))})
    return {
        "name": name, "dir": d, "cfg": cfg,
        "source_size": cfg.get("source_size"),
        "points_done": all(pts.get(k) is not None for k in REQUIRED_POINTS),
        "prepped": os.path.exists(os.path.join(d, "work", "figure.png")),
        "pockets": read_json(os.path.join(d, "work", "pockets.json"), []),
        "plates": status, "requests": reqs,
        "built": os.path.exists(os.path.join(out, f"{model}.moc3")),
        "previews": [p for p in ("preview/poses.png", "preview/holes.png", "preview/motion.png", "preview/blink.png")
                     if os.path.exists(os.path.join(d, p))],
        "installed": bool(platform_env.vts_models_dir() and os.path.isdir(os.path.join(platform_env.vts_models_dir(),
                                                                                       cfg.get("vts_folder", model)))),
        "mtime": max((os.path.getmtime(os.path.join(d, f)) for f in ("character.json",) if os.path.exists(os.path.join(d, f))), default=0),
        "hq": hq_state(d),
    }


def by_time(folder):
    """The folder's entries, oldest first (labels may be dates or names like v2)."""
    if not os.path.isdir(folder):
        return []
    return sorted((x for x in os.listdir(folder) if not x.startswith(".")), key=lambda x: os.path.getmtime(os.path.join(folder, x)))


def hq_state(d):
    refs = [f"gen/out/{n}.png" for n in ("ref_turn_L", "ref_turn_R", "ref_pitch_up", "ref_pitch_down") if os.path.exists(os.path.join(d, "gen", "out", n + ".png"))]
    qa_dir = os.path.join(d, "preview", "qa")
    labels = by_time(qa_dir)
    last = labels[-1] if labels else None
    imgs, review = [], ""
    if last:
        imgs = [f"preview/qa/{last}/{f}" for f in sorted(os.listdir(os.path.join(qa_dir, last))) if f.endswith(".png")]
        rv = os.path.join(qa_dir, last, "review.md")
        if os.path.exists(rv):
            review = open(rv, encoding="utf-8").read()
    vers = by_time(os.path.join(d, "versions"))
    return {"refs": refs, "calibrate": read_json(os.path.join(d, "preview", "hq", "calibrate.json"), None),
            "calibrate_img": "preview/hq/calibrate.png" if os.path.exists(os.path.join(d, "preview", "hq", "calibrate.png")) else None,
            "qa_label": last, "qa_images": imgs, "review": review, "versions": vers,
            "ai_points": os.path.exists(os.path.join(d, "work", "ai_points_check.png"))}


REQUIRED_POINTS = ("hair_top", "crown", "eye_l", "eye_r", "nose", "mouth", "chin", "face_l", "face_r", "neck")


def list_projects():
    os.makedirs(PROJECTS, exist_ok=True)
    out = []
    for n in sorted(os.listdir(PROJECTS)):
        if NAME_RE.match(n) and os.path.exists(os.path.join(PROJECTS, n, "character.json")):
            try:
                s = project_state(n)
                out.append({k: s[k] for k in ("name", "points_done", "prepped", "built", "installed", "mtime")})
            except Exception:
                pass
    return sorted(out, key=lambda x: -x["mtime"])


def decode_image(data_url):
    """A data: URL (or plain base64) from the page -> bytes."""
    if "," in data_url[:100]:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


# --------------------------------------------------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    server_version = "VTModelMaker/1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 200 * 1024 * 1024:
            raise ValueError("送られたファイルが大きすぎます（200MBまで）")
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _local_only(self):
        # (the page is served on 127.0.0.1 only; refuse requests made by other sites in the same browser)
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
            return False
        return host.split(":")[0] in ("127.0.0.1", "localhost")

    # ---- GET
    def do_GET(self):
        if not self._local_only():
            return self._send(403, {"error": "この操作はできません（このパソコンの中からだけ使えます）"})
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"))
            if path.startswith("/static/"):
                f = os.path.normpath(os.path.join(STATIC, path[len("/static/"):]))
                if not f.startswith(STATIC):
                    return self._send(403, {"error": "この操作はできません（このパソコンの中からだけ使えます）"})
                return self._file(f)
            if path == "/api/env":
                s = platform_env.summary()
                s["vts_running"] = platform_env.vts_running()
                s["projects_dir"] = PROJECTS
                return self._send(200, s)
            if path == "/api/projects":
                return self._send(200, list_projects())
            if path == "/api/job":
                with LOCK:
                    j = dict(JOB)
                    j["lines"] = JOB["lines"][-200:]
                return self._send(200, j)
            m = re.match(r"^/api/p/([^/]+)/(state|file)$", path)
            if m:
                name, what = m.groups()
                if what == "state":
                    return self._send(200, project_state(name))
                rel = q.get("path", [""])[0].replace("\\", "/")
                d = proj_dir(name)
                f = os.path.normpath(os.path.join(d, rel))
                if not rel or rel.split("/")[0] not in SAFE_DIRS or not f.startswith(os.path.normpath(d) + os.sep):
                    return self._send(403, {"error": "この操作はできません（このパソコンの中からだけ使えます）"})
                if q.get("max"):
                    return self._thumb(f, int(q["max"][0]))
                return self._file(f)
            return self._send(404, {"error": "見つかりません"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def _file(self, f):
        if not os.path.isfile(f):
            return self._send(404, {"error": "見つかりません"})
        ctype = mimetypes.guess_type(f)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(f, "rb") as fh:
            return self._send(200, fh.read(), ctype)

    def _thumb(self, f, mx):
        """A smaller copy of a picture (the canvas pictures are thousands of px high)."""
        if not os.path.isfile(f):
            return self._send(404, {"error": "見つかりません"})
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(f)
        s = min(1.0, mx / float(max(im.size)))
        if s < 1:
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        b = io.BytesIO()
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        im.save(b, "PNG")
        return self._send(200, b.getvalue(), "image/png")

    # ---- POST
    def do_POST(self):
        if not self._local_only():
            return self._send(403, {"error": "この操作はできません（このパソコンの中からだけ使えます）"})
        path = unquote(urlparse(self.path).path)
        try:
            body = self._json()
            if path == "/api/projects":
                return self._send(200, self.new_project(body))
            if path == "/api/settings":
                cfg = platform_env.config()
                for k in ("imagegen",):
                    if k in body:
                        cfg[k] = body[k]
                platform_env.save_config(cfg)
                return self._send(200, {"ok": True})
            if path == "/api/open":
                name = body.get("project")
                target = proj_dir(name) if name else PROJECTS
                sub = body.get("sub")
                if sub:
                    target = os.path.normpath(os.path.join(target, sub))
                os.makedirs(target, exist_ok=True)
                platform_env.open_path(target)
                return self._send(200, {"ok": True})
            m = re.match(r"^/api/p/([^/]+)/([a-z_]+)(?:/([A-Za-z0-9_]+))?$", path)
            if not m:
                return self._send(404, {"error": "見つかりません"})
            name, action, arg = m.groups()
            d = proj_dir(name)
            cfg_path = os.path.join(d, "character.json")
            cfg = read_json(cfg_path, {})
            if action == "points":
                pts = cfg.setdefault("points", {})
                for k, v in (body.get("points") or {}).items():
                    if k in pts or k in REQUIRED_POINTS:
                        pts[k] = None if v is None else [round(float(v[0]), 1), round(float(v[1]), 1)]
                if "none" in body:
                    cfg["points_none"] = sorted(set(body.get("none") or []))
                write_json(cfg_path, cfg)
                return self._send(200, {"ok": True})
            if action == "describe":
                cfg["describe"] = {k: str(v).strip() for k, v in (body.get("describe") or {}).items() if str(v).strip()}
                write_json(cfg_path, cfg)
                return self._send(200, {"ok": True})
            if action == "pockets":
                cands = read_json(os.path.join(d, "work", "pockets.json"), [])
                sel = set(int(i) for i in body.get("selected", []))
                cfg["bg_pockets"] = [c["point"] for c in cands if c["i"] in sel] + [p for p in cfg.get("bg_pockets", []) if body.get("keep_old")]
                write_json(cfg_path, cfg)
                return self._send(200, {"ok": True, "count": len(cfg["bg_pockets"])})
            if action == "plate" and arg:
                data = decode_image(body["image"])
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                im = Image.open(io.BytesIO(data)).convert("RGB")
                os.makedirs(os.path.join(d, "gen", "out"), exist_ok=True)
                im.save(os.path.join(d, "gen", "out", f"{arg}.png"))
                return self._send(200, {"ok": True, "size": im.size})
            if action == "skip":
                sk = set(cfg.get("plates_skipped", []))
                for n in body.get("names", []):
                    if body.get("undo"):
                        sk.discard(n)
                    else:
                        sk.add(n)
                cfg["plates_skipped"] = sorted(sk)
                write_json(cfg_path, cfg)
                return self._send(200, {"ok": True})
            if action == "run":
                kit = kit_of(d)
                frm, to = body.get("from", "prep"), body.get("to", "preview")
                cmd = [PY, "-X", "utf8", os.path.join(kit, "run.py"), d, "--from", frm, "--to", to]
                ok = start_job(name, body.get("title") or f"{frm} → {to}", cmd)
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "ai_landmarks":
                cmd = [PY, "-X", "utf8", os.path.join(kit_of(d), "ai_landmarks.py"), d] + (["--force"] if body.get("force") else [])
                ok = start_job(name, "AIで位置合わせ", cmd)
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "hq" and arg == "update":
                ok = start_job(name, "本体を最新にする", [PY, "-X", "utf8", os.path.join(ENGINE, "update_kit.py"), d])
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "hq" and arg in ("refs", "calibrate", "qa", "review", "snapshot"):
                # (the tool's own hq.py, also for a project made by an older version: it builds with the project's kit)
                cmd = [PY, "-X", "utf8", os.path.join(ENGINE, "hq.py"), d, arg]
                if arg == "calibrate":
                    cmd.append("--apply")
                if arg == "qa":
                    vd = os.path.join(d, "versions")
                    vers = by_time(vd)
                    if vers:
                        cmd += ["--compare", os.path.join(vd, vers[-1])]
                titles = {"refs": "見本の向き絵", "calibrate": "向きを見本に合わせる", "qa": "確認画像", "review": "AIの確認", "snapshot": "今の版を残す"}
                ok = start_job(name, titles[arg], cmd)
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "describe_ai":
                ok = start_job(name, "説明文の下書き", [PY, "-X", "utf8", os.path.join(kit_of(d), "describe.py"), d])
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "motion":
                model = cfg.get("name", name)
                box = ",".join(str(int(v)) for v in self.head_box(d, cfg))
                blink = bool(body.get("blink"))
                out_png = os.path.join(d, "preview", "blink.png" if blink else "motion.png")
                cmd = [PY, "-X", "utf8", os.path.join(kit_of(d), "motion_strip.py"), os.path.join(d, "out", model), out_png,
                       "--box", box, "--scale", str(self.view_scale(self.head_box(d, cfg)))]
                if blink:
                    cmd += ["--blink", "1.0"]
                ok = start_job(name, "まばたきの確認" if blink else "動きの確認", cmd)
                return self._send(200 if ok else 409, {"ok": ok})
            if action == "install":
                r = platform_env.vts_running()
                if r is not False:
                    return self._send(409, {"ok": False, "error": "VTube Studio が起動中です（または確認できません）。終了してからもう一度押してください。"})
                cmd = [PY, "-X", "utf8", os.path.join(kit_of(d), "package.py"), d, "--install"]
                ok = start_job(name, "VTube Studio に入れる", cmd)
                return self._send(200 if ok else 409, {"ok": ok})
            return self._send(404, {"error": "知らない操作です"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def head_box(self, d, cfg):
        k = float(cfg.get("canvas_scale", 1))
        p = cfg.get("points", {})
        if p.get("face_l") and p.get("face_r") and p.get("crown") and p.get("chin"):
            hx = 0.5 * (p["face_l"][0] + p["face_r"][0]) * k
            hr = 0.5 * (p["face_r"][0] - p["face_l"][0]) * k
            top = (p.get("hair_top") or p["crown"])[1] * k
            return (hx - 3.2 * hr, top - 0.2 * hr, hx + 3.2 * hr, p["chin"][1] * k + 2.4 * hr)
        w, h = cfg.get("source_size", [1000, 2000])
        return (0, 0, w * k, h * k * 0.35)

    def view_scale(self, box):
        return round(min(0.6, max(0.12, 700.0 / max(1.0, box[3] - box[1]))), 3)

    def new_project(self, body):
        name = (body.get("name") or "").strip()
        if not NAME_RE.match(name):
            raise ValueError("名前は半角の英数字と - _ だけにしてください（例: chara7）")
        d = proj_dir(name)
        if os.path.exists(d) and os.listdir(d):
            raise ValueError("その名前のキャラはもうあります")
        import tempfile
        ext = os.path.splitext(body.get("filename") or "art.png")[1].lower() or ".png"
        tmpd = tempfile.mkdtemp(prefix="vtmm_")
        tmp = os.path.join(tmpd, "art" + ext)
        with open(tmp, "wb") as f:
            f.write(decode_image(body["image"]))
        r = subprocess.run([PY, "-X", "utf8", os.path.join(ENGINE, "setup.py"), d, name, tmp], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        shutil.rmtree(tmpd, ignore_errors=True)
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr)[-800:])
        return project_state(name)


def free_port(start, tries=40):
    """The first port from `start` that can be listened on (SO_REUSEADDR as the server itself uses: a port the previous
    run just closed is free)."""
    for p in range(start, start + tries):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("空いているポートが見つかりません（8765番から順に試しました）。ほかに開いている VTModelMaker を閉じてください")


def main():
    # (a port given with --port is used as it is; else the first free one from 8765)
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else free_port(8765)
    os.makedirs(PROJECTS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"VTModelMaker: {url}  （このウィンドウを閉じると止まります）", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
