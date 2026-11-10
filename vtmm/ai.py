"""Asking an AI to look at pictures: the Claude Code CLI (preferred) or the Codex CLI, both read-only.

    text, who = ask(project_dir, prompt, ["work/ai_guide_full.png", ...], tag="landmarks")

Claude Code runs headless (`claude -p`) in the project folder with only its read-only tools - it opens the pictures with
its Read tool and cannot run commands or change files. Codex runs `codex exec` with the pictures attached and a read-only
sandbox. Both are the person's own accounts on this computer; nothing is sent anywhere else. Every call is logged in
gen/logs/ai_<tag>.log. Raises RuntimeError when neither is installed or the answer is empty.
"""
import os, sys, json, re, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platform_env

READ_ONLY_TOOLS = ["Read", "Glob", "Grep"]


def available():
    return [w for w, b in (("claude", platform_env.claude_bin()), ("codex", platform_env.codex_bin())) if b]


def _log(proj, tag, text):
    d = os.path.join(proj, "gen", "logs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"ai_{tag}.log"), "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")


def ask_claude(proj, prompt, images, tag, timeout):
    exe = platform_env.claude_bin()
    files = "\n".join(f"- {p}" for p in images)
    full = (prompt + "\n\nOpen these pictures with your Read tool (paths are relative to the current folder) and look at them "
            "carefully before you answer:\n" + files + "\n\nDo not run commands and do not change any file. Reply with the "
            "answer only.")
    cmd = [exe, "-p", full, "--output-format", "text", "--allowedTools", *READ_ONLY_TOOLS,
           "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch"]
    r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _log(proj, tag, f"[claude] exit {r.returncode}\n--- prompt\n{full}\n--- answer\n{r.stdout}\n--- stderr\n{r.stderr[-2000:]}")
    if "Not logged in" in r.stdout[:200] or "/login" in r.stdout[:200]:
        raise RuntimeError("Claude Code にログインしていません。ターミナルで `claude` を一度起動して、/login でログインしてください")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"Claude Code から返事がありませんでした（終了コード {r.returncode}）: {r.stderr.strip()[-300:]}")
    return r.stdout


def ask_codex(proj, prompt, images, tag, timeout):
    exe = platform_env.codex_bin()
    out_txt = os.path.join(proj, "gen", "logs", f"ai_{tag}.answer.txt")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    if os.path.exists(out_txt):
        os.remove(out_txt)
    cmd = [exe, "exec", "-C", proj, "--skip-git-repo-check", "--sandbox", "read-only", "-c", 'model_reasoning_effort="medium"']
    for p in images:
        cmd += ["-i", p]
    cmd += ["-o", out_txt, "-"]
    r = subprocess.run(cmd, input=prompt + "\n\nReply with the answer only.", cwd=proj, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = open(out_txt, encoding="utf-8", errors="replace").read() if os.path.exists(out_txt) else ""
    _log(proj, tag, f"[codex] exit {r.returncode}\n--- prompt\n{prompt}\n--- answer\n{text}\n--- output\n{(r.stdout + r.stderr)[-2000:]}")
    if not text.strip():
        raise RuntimeError(f"Codex から返事がありませんでした（終了コード {r.returncode}）")
    return text


def ask(proj, prompt, images, tag="ask", prefer=("claude", "codex"), timeout=900):
    have = available()
    errors = []
    for who in prefer:
        if who not in have:
            continue
        try:
            fn = ask_claude if who == "claude" else ask_codex
            return fn(proj, prompt, images, tag, timeout), who
        except Exception as e:
            errors.append(f"{who}: {e}")
    if not have:
        raise RuntimeError("Claude Code CLI も Codex CLI も見つかりません。どちらかを入れてください")
    raise RuntimeError("; ".join(errors))


def json_in(text):
    """The first JSON object in an answer (models sometimes wrap it in a code fence or a sentence)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("AIの返事に JSON がありませんでした")
    return json.loads(m.group(0))
