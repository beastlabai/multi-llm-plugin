#!/usr/bin/env python3
"""In-container capture driver (pi json-mode rows + omp compaction row).

Adapted from plans/harness-expansion/probe-out/incontainer.py: one server per
row on 127.0.0.1:8099, polled with urllib, then the CLI through the PATH wrapper
(/opt/mllm/bin/<cli> IS wrapper.py). Never wrapped in timeout(1); stdin is
/dev/null.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLI = os.environ["CLI"]
ROW = os.environ["ROW"]
SERVER = os.environ["SERVER"]          # stub | fail | script
MODE = os.environ.get("MODE", "")
PORT = os.environ.get("MLLM_STUB_PORT", "8099")
OUT = Path("/work/out/rows") / CLI
OUT.mkdir(parents=True, exist_ok=True)

seed = Path("/opt/mllm/rig/home_seed") / CLI
home = Path("/work/homes") / CLI
home.mkdir(parents=True, exist_ok=True)
if seed.is_dir():
    shutil.copytree(seed, home, dirs_exist_ok=True)

if os.environ.get("DROP_API_KEY") == "1":
    models = home / ".pi" / "agent" / "models.json"
    data = json.loads(models.read_text())
    for provider in data["providers"].values():
        provider.pop("apiKey", None)
    models.write_text(json.dumps(data, indent=2))

if os.environ.get("PI_SETTINGS"):
    target = home / ".pi" / "agent" / "settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(os.environ["PI_SETTINGS"])

work = Path("/work/sample-project")
work.mkdir(parents=True, exist_ok=True)
(work / "plans").mkdir(exist_ok=True)
(work / "plans" / "auth-feature.md").write_text(
    "# Auth feature\n\n## Tasks\n\n### Task 1: Add a retry ladder\n\nRetry the "
    "token refresh.\n")
(work / "notes").mkdir(exist_ok=True)
(work / "notes" / "incident.md").write_text(
    "# Incident notes\n\nThe gateway answered `401 Unauthorized: Incorrect API key "
    "provided` for the first minute, then `429 Too Many Requests: Rate limit "
    "reached for requests per min`. Also seen: `rate limit exceeded`, `quota "
    "exceeded`, `model_not_found`, `context_length_exceeded`.\n")
(work / "big").mkdir(exist_ok=True)
for n in range(1, 5):
    line = f"part {n}: " + ("lorem ipsum dolor sit amet " * 3).strip() + "\n"
    size = int(os.environ.get("PART_BYTES") or 40000)
    (work / "big" / f"part{n}.txt").write_text(line * max(1, size // len(line)))

log = OUT / f"{ROW}.server.jsonl"
if log.exists():
    log.unlink()
if SERVER == "stub":
    cmd = ["python3", "/rig/stub_server.py", "--port", PORT, "--log", str(log)]
    scenario = Path("/work/out/scenarios") / f"{ROW}.yaml"
    if scenario.is_file():
        cmd += ["--scenario", str(scenario)]
elif SERVER == "script":
    cmd = ["python3", "/work/out/scriptserver.py", "--port", PORT, "--mode", MODE,
           "--log", str(log)]
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

started = time.time()
with open(OUT / f"{ROW}.stdout.txt", "wb") as so, \
        open(OUT / f"{ROW}.stderr.txt", "wb") as se:
    rc = subprocess.call(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                         cwd=str(work))
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
