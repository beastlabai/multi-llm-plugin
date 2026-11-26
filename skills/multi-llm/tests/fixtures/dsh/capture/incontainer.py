#!/usr/bin/env python3
"""In-container capture driver for one dsh row (README.md here).

The row-capture kit's driver (``tests/fixtures/row_capture/incontainer.py``)
without the PATH wrapper: the image is this directory's throwaway Dockerfile,
not the e2e image, so the driver itself applies what the wrapper would. One
server per row on 127.0.0.1:8099 (``/rig/stub_server.py`` with a scenario,
``/rig/failserver.py --mode <m>``, or the row kit's ``scriptserver.py`` with a
tool and its arguments), then the adapter's rendered argv (``render.py``) with
the prompt on stdin, a fresh ``DSH_HOME`` holding the rendered profile patch,
the adapter's environment, and the stub key. The workspace is a git repository;
``/work/outside`` is a writable directory outside it, and both are recorded
after the run so a tool row shows what a tool call could change.

A CLI still running at ``CAPTURE_DEADLINE_S`` gets SIGTERM, then SIGKILL 10 s
later, and the meta records ``killed: true``.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROW = os.environ["ROW"]
SERVER = os.environ["SERVER"]          # stub | fail | script
MODE = os.environ.get("MODE", "")
TOOL = os.environ.get("TOOL", "")
TOOL_ARGS = os.environ.get("TOOL_ARGS", "{}")
PORT = "8099"
DEADLINE = float(os.environ.get("CAPTURE_DEADLINE_S") or 240)
KIT = Path("/work/out")
OUT = KIT / "rows" / "dsh"
OUT.mkdir(parents=True, exist_ok=True)

work = Path("/work/sample-project")
(work / "plans").mkdir(parents=True, exist_ok=True)
(work / "plans" / "auth-feature.md").write_text(
    "# Auth feature\n\n## Tasks\n\n### Task 1: Add a retry ladder\n\nRetry the "
    "token refresh.\n")
git_env = dict(os.environ, HOME="/tmp", GIT_AUTHOR_NAME="rig", GIT_AUTHOR_EMAIL="rig@invalid",
               GIT_COMMITTER_NAME="rig", GIT_COMMITTER_EMAIL="rig@invalid")
for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "seed"]):
    subprocess.run(cmd, cwd=str(work), env=git_env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
outside = Path("/work/outside")
outside.mkdir(parents=True, exist_ok=True)

log = OUT / f"{ROW}.server.jsonl"
if log.exists():
    log.unlink()
if SERVER == "stub":
    scenario = KIT / "scenarios" / f"{os.environ.get('SCENARIO') or ROW}.yaml"
    cmd = ["python3", "/rig/stub_server.py", "--port", PORT, "--log", str(log),
           "--scenario", str(scenario)]
elif SERVER == "script":
    cmd = ["python3", str(KIT / "scriptserver.py"), "--port", PORT, "--log", str(log),
           "--tool", TOOL, "--args", TOOL_ARGS]
else:
    cmd = ["python3", "/rig/failserver.py", "--port", PORT, "--mode", MODE, "--log", str(log)]
srv_out = open(OUT / f"{ROW}.server.log", "wb")
server = subprocess.Popen(cmd, stdout=srv_out, stderr=subprocess.STDOUT)
deadline = time.time() + 30
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/__health", timeout=1).read()
        break
    except Exception:
        time.sleep(0.1)
else:
    print("server never became healthy", file=sys.stderr)
    sys.exit(2)

dsh_home = Path("/work/homes/dsh")
patch = dsh_home / "profiles" / "headless" / "cordis.patch.yml"
patch.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(KIT / "dsh" / "patch.yml", patch)
home = Path("/work/homes/home")
home.mkdir(parents=True, exist_ok=True)
tmp = Path("/work/homes/tmp")
tmp.mkdir(parents=True, exist_ok=True)

env = {
    "PATH": os.environ["PATH"], "LANG": "C.UTF-8", "HOME": str(home), "TMPDIR": str(tmp),
    "OPENROUTER_API_KEY": "stub-key-not-a-credential", "DSH_HOME": str(dsh_home),
}
env.update(json.loads((KIT / "dsh" / "env.json").read_text()))
if os.environ.get("MODEL_OVERRIDE"):
    env["DSH_MULTI_LLM_MODEL"] = os.environ["MODEL_OVERRIDE"]
if "REASONING_OVERRIDE" in os.environ:
    env["DSH_MULTI_LLM_REASONING"] = os.environ["REASONING_OVERRIDE"]
argv = json.loads((KIT / "dsh" / "argv.json").read_text())
prompt = (KIT / "prompt.txt").read_bytes()

help_out = subprocess.run(["dsh", "--profile", "headless", "--help"], env=dict(env, DSH_HOME="/tmp/help-home"),
                          capture_output=True, text=True)
missing = [f for f in ("--json", "--permission-mode", "--approval-policy") if f not in help_out.stdout]

killed = False
started = time.time()
with open(OUT / f"{ROW}.stdout.txt", "wb") as so, open(OUT / f"{ROW}.stderr.txt", "wb") as se:
    child = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.PIPE, cwd=str(work), env=env)
    child.stdin.write(prompt)
    child.stdin.close()
    try:
        rc = child.wait(timeout=DEADLINE)
    except subprocess.TimeoutExpired:
        killed = True
        child.send_signal(signal.SIGTERM)
        try:
            rc = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            rc = child.wait()
wall = time.time() - started

requests = 0
for line in log.read_text().splitlines() if log.is_file() else []:
    try:
        entry = json.loads(line)
    except ValueError:
        continue
    if entry.get("method") == "GET" or str(entry.get("path", "")).endswith("__health"):
        continue
    requests += 1

porcelain = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=str(work), env=git_env, capture_output=True, text=True).stdout
meta = {"cli": "dsh", "row": ROW, "server": SERVER, "mode": MODE, "tool": TOOL,
        "exit_code": rc, "killed": killed, "deadline_s": DEADLINE,
        "wall_s": round(wall, 2), "requests": requests,
        "stdout_bytes": (OUT / f"{ROW}.stdout.txt").stat().st_size,
        "stderr_bytes": (OUT / f"{ROW}.stderr.txt").stat().st_size,
        "workspace_changes": porcelain.splitlines(),
        "outside_files": sorted(p.name for p in outside.iterdir()),
        "help_flags_missing": missing,
        "dsh_version": subprocess.run(["dsh", "--version"], env=env, capture_output=True,
                                      text=True).stdout.strip(),
        "argv": argv}
(OUT / f"{ROW}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
print(json.dumps(meta))
server.terminate()
try:
    server.wait(timeout=5)
except Exception:
    server.kill()
