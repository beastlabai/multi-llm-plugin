#!/usr/bin/env python3
"""In-container capture driver for one §6.2 row (README.md here).

pi's capture driver (tests/fixtures/pi/capture/incontainer.py), generalised:
one server per row on 127.0.0.1:8099 (``/rig/stub_server.py`` with a scenario,
``/rig/failserver.py --mode <m>``, or this kit's ``scriptserver.py`` with the
CLI's row of ``scripts.json``), polled until healthy, then the CLI
through the PATH wrapper (``/opt/mllm/bin/<cli>`` IS wrapper.py, so
redirect.yaml's row applies itself). The workspace is a git repository, as
seed_workspace.py makes it for the container lane. stdin is /dev/null and the
CLI is never wrapped in timeout(1); a CLI still running at
``CAPTURE_DEADLINE_S`` gets SIGTERM (the wrapper forwards it), then SIGKILL
10 s later, and the meta records ``killed: true`` with the exit status the
kill left.
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

CLI = os.environ["CLI"]
ROW = os.environ["ROW"]
SERVER = os.environ["SERVER"]          # stub | fail
MODE = os.environ.get("MODE", "")
PORT = os.environ.get("MLLM_STUB_PORT", "8099")
DEADLINE = float(os.environ.get("CAPTURE_DEADLINE_S") or 240)
OUT = Path("/work/out/rows") / CLI
OUT.mkdir(parents=True, exist_ok=True)

# The wrapper seeds each invocation root from $MLLM_HOMES_ROOT/<cli>; the
# container lane's entrypoint fills that from the image's home_seed, and so
# does this.
seed = Path("/opt/mllm/rig/home_seed") / CLI
home = Path("/work/homes") / CLI
home.mkdir(parents=True, exist_ok=True)
if seed.is_dir():
    shutil.copytree(seed, home, dirs_exist_ok=True)

work = Path("/work/sample-project")
work.mkdir(parents=True, exist_ok=True)
(work / "plans").mkdir(exist_ok=True)
(work / "plans" / "auth-feature.md").write_text(
    "# Auth feature\n\n## Tasks\n\n### Task 1: Add a retry ladder\n\nRetry the "
    "token refresh.\n")
git_env = dict(os.environ, HOME="/tmp", GIT_AUTHOR_NAME="rig", GIT_AUTHOR_EMAIL="rig@invalid",
               GIT_COMMITTER_NAME="rig", GIT_COMMITTER_EMAIL="rig@invalid")
for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
            ["git", "commit", "-q", "-m", "seed"]):
    subprocess.run(cmd, cwd=str(work), env=git_env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

log = OUT / f"{ROW}.server.jsonl"
if log.exists():
    log.unlink()
if SERVER == "stub":
    cmd = ["python3", "/rig/stub_server.py", "--port", PORT, "--log", str(log)]
    scenarios = Path("/work/out/scenarios")
    scenario = scenarios / f"{ROW}.yaml"
    if not scenario.is_file():
        print(f"no scenario for {ROW}", file=sys.stderr)
        sys.exit(2)
    cmd += ["--scenario", str(scenario)]
elif SERVER == "script":
    script = json.loads((Path("/work/out") / "scripts.json").read_text())[CLI]
    cmd = ["python3", "/work/out/scriptserver.py", "--port", PORT, "--log", str(log),
           "--tool", script["tool"], "--args", script["args"]]
else:
    cmd = ["python3", "/rig/failserver.py", "--port", PORT, "--mode", MODE,
           "--log", str(log)]
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

argv = json.loads((Path("/work/out") / "argv.json").read_text())[CLI]
prompt = (Path("/work/out") / "prompt.txt").read_text()
argv = [prompt if tok == "PROMPT_PLACEHOLDER" else tok for tok in argv]

killed = False
started = time.time()
with open(OUT / f"{ROW}.stdout.txt", "wb") as so, \
        open(OUT / f"{ROW}.stderr.txt", "wb") as se:
    child = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                             cwd=str(work))
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
    if str(entry.get("path", "")).endswith("__health") or entry.get("health"):
        continue
    if entry.get("method") == "GET":
        continue
    requests += 1

meta = {"cli": CLI, "row": ROW, "server": SERVER, "mode": MODE, "exit_code": rc,
        "killed": killed, "deadline_s": DEADLINE,
        "wall_s": round(wall, 2), "requests": requests,
        "stdout_bytes": (OUT / f"{ROW}.stdout.txt").stat().st_size,
        "stderr_bytes": (OUT / f"{ROW}.stderr.txt").stat().st_size,
        "argv_head": [t if len(t) < 200 else f"<prompt len={len(t)}>" for t in argv]}
(OUT / f"{ROW}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
print(json.dumps(meta))
server.terminate()
try:
    server.wait(timeout=5)
except Exception:
    server.kill()
