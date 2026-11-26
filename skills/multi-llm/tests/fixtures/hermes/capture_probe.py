"""Credential-free capture of one hermes row through the production adapter.

Runs inside the probe image (README.md here), ``--network none``: the skill is
mounted at ``/skill``, ``tests/container`` at ``/rig``. Only the adapter's
``BASE_URL`` changes, to the localhost server; the argv, environment, private
home, ``config.yaml`` and the gateway stop are the adapter's own. Prints one
JSON object: the streams, exit code, wall time, request counts and what was
left running after the dispatch closed.

    python3 capture_probe.py <row> <server> [mode] [--model M] [--kill-after S] [--no-key]
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, "/skill")
from utils.providers import hermes, private_home  # noqa: E402
from utils.providers.base import build_child_env  # noqa: E402
from utils.providers.types import PROFILE_GENERATE  # noqa: E402

PROMPT = Path(__file__).with_name("prompt.txt")
PORT = 18900


def start(server, mode, log):
    if server == "stub":
        scenario = f"/kit/scenarios/{mode}.yaml"
        cmd = [sys.executable, "/rig/stub_server.py", "--port", str(PORT), "--log", log, "--scenario", scenario]
    elif server == "fail":
        cmd = [sys.executable, "/rig/failserver.py", "--port", str(PORT), "--mode", mode, "--log", log]
    else:  # script: the tool-turn server beside this file
        cmd = [sys.executable, str(Path(__file__).with_name("toolserver.py")), "--port", str(PORT), "--log", log, "--mode", mode]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/rig")
    time.sleep(1.0)
    return process


def requests(log):
    try:
        return [json.loads(line) for line in open(log, encoding="utf-8")]
    except FileNotFoundError:
        return []


def daemons():
    out = subprocess.run(["ps", "-eo", "stat,args"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "gateway run" in line and not line.startswith("Z")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("row")
    ap.add_argument("server", choices=("stub", "fail", "script"))
    ap.add_argument("mode")
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--kill-after", type=float, default=None)
    ap.add_argument("--deadline", type=float, default=240.0)
    ap.add_argument("--no-key", action="store_true", help="dispatch with OPENROUTER_API_KEY unset")
    args = ap.parse_args()

    work = Path("/tmp/work")
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    if args.no_key:
        os.environ.pop("OPENROUTER_API_KEY", None)
    else:
        os.environ["OPENROUTER_API_KEY"] = "stub-key-not-a-credential"
    hermes.BASE_URL = f"http://127.0.0.1:{PORT}/v1"
    log = "/tmp/requests.jsonl"
    server = start(args.server, args.mode, log)
    provider = hermes.HermesProvider()
    prompt = PROMPT.read_text(encoding="utf-8")

    start_s = time.monotonic()
    killed = False
    with private_home.dispatch_homes():
        with provider.hermetic_dispatch(PROFILE_GENERATE, args.model) as refusal:
            assert refusal is None, refusal
            root = hermes._ACTIVE_HOME.get()
            env = build_child_env(provider, provider_env=provider.resolve_env(args.model, PROFILE_GENERATE))
            cmd, stdin = provider.build_command_for_transport(prompt, args.model, "stdin")
            child = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, cwd=work, start_new_session=True,
            )
            try:
                stdout, stderr = child.communicate(stdin, timeout=args.kill_after or args.deadline)
            except subprocess.TimeoutExpired:
                killed = True
                os.killpg(child.pid, signal.SIGTERM)
                stdout, stderr = child.communicate(timeout=20)
            wall = round(time.monotonic() - start_s, 2)
            at_exit = len(requests(log))
            running_at_exit = daemons()
        # hermetic_dispatch has stopped the gateway here.
        running_after_stop = daemons()
        after_stop = len(requests(log))
    time.sleep(5)
    chat = [r for r in requests(log) if "chat/completions" in (r.get("path") or "")]
    server.terminate()
    print(json.dumps({
        "row": args.row, "argv": cmd, "exit_code": child.returncode, "wall_s": wall,
        "killed": killed, "stdout": stdout, "stderr": stderr,
        "requests": at_exit, "chat_requests": len(chat),
        "requests_after_stop_plus_5s": len(requests(log)) - after_stop,
        "daemon_running_at_exit": len(running_at_exit),
        "daemon_running_after_stop": len(running_after_stop),
        "root_removed": not Path(root).exists(),
        "workspace": sorted(p.name for p in work.iterdir()),
        "parsed_success": provider.parse_output(stdout, stderr).get("success"),
        "classify_failure": provider.classify_failure(stdout, stderr, child.returncode),
    }))


if __name__ == "__main__":
    main()
