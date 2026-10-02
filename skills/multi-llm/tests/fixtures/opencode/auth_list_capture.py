#!/usr/bin/env python3
"""Capture ``opencode auth list`` offline ($0, no model call, no network).

Writes ``auth_list.<version>.json`` beside this file (``opencode_version``,
``argv``, ``captured``, ``cases``), the envelope the auth probe's classifier
(``OpenCodeProvider.classify_auth_probe``) is graded on. Every case runs the real
binary from a scratch HOME under a cleared environment (only ``HOME``, ``PATH``
and the variables the case names; no ``XDG_*``), stdin ``/dev/null``, no TTY.
Nothing reads or writes the invoking user's own opencode config or data.

Every key is a canary (``sk-canary-not-a-key-<n>``); ``auth list`` prints
provider names, never keys. Every absolute scratch path is rewritten to
``/rig``; nothing else is edited.

usage: auth_list_capture.py [--bin PATH]
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "https://opencode.ai/config.json"

#: name -> (description, files under HOME, env)
CASES = {
    "empty": ("nothing stored, no key in the environment", {}, {}),
    "stored": (
        "an OpenRouter API key stored in ~/.local/share/opencode/auth.json",
        {".local/share/opencode/auth.json": {
            "openrouter": {"type": "api", "key": "sk-canary-not-a-key-1"}}},
        {},
    ),
    "env_only": (
        "nothing stored, OPENROUTER_API_KEY in the environment",
        {}, {"OPENROUTER_API_KEY": "sk-canary-not-a-key-2"},
    ),
    "config_provider_only": (
        "nothing stored; ~/.config/opencode/opencode.json declares a custom "
        "provider carrying its own options.apiKey",
        {".config/opencode/opencode.json": {
            "$schema": SCHEMA,
            "provider": {"stub": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://127.0.0.1:9/v1",
                            "apiKey": "sk-canary-not-a-key-3"},
                "models": {"stub-model": {}},
            }},
        }},
        {},
    ),
    "empty_no_color": (
        "as empty, with NO_COLOR=1 and TERM=dumb", {},
        {"NO_COLOR": "1", "TERM": "dumb"},
    ),
}


def run_case(binary, name, spec, scratch):
    _desc, files, env_extra = spec
    home = os.path.join(scratch, name, "home")
    os.makedirs(home)
    for rel, body in files.items():
        path = os.path.join(home, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2)
    env = {"HOME": home, "PATH": "/usr/bin:/bin", **env_extra}
    with open(os.devnull, "rb") as devnull:
        done = subprocess.run(
            [binary, "auth", "list"], cwd=home, env=env, stdin=devnull,
            capture_output=True, text=True, timeout=60,
        )
    root = os.path.join(scratch, name)

    def scrub(text):
        return (text or "").replace(root, "/rig")

    return {"exit": done.returncode, "stdout": scrub(done.stdout), "stderr": scrub(done.stderr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=shutil.which("opencode") or "opencode")
    args = parser.parse_args()
    scratch = tempfile.mkdtemp(prefix="oc-auth-list-")
    try:
        with open(os.devnull, "rb") as devnull:
            version = subprocess.run(
                [args.bin, "--version"], stdin=devnull, capture_output=True, text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": scratch},
            ).stdout.strip()
        cases = {name: run_case(args.bin, name, spec, scratch) for name, spec in CASES.items()}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out = {
        "opencode_version": version,
        "argv": ["opencode", "auth", "list"],
        "captured": (
            f"{datetime.date.today().isoformat()}, auth_list_capture.py: host, scratch "
            "HOME per case, no XDG_* set, stdin /dev/null, no TTY, every key a "
            "sk-canary-not-a-key-<n> canary"
        ),
        "setup": {name: spec[0] for name, spec in CASES.items()},
        "cases": cases,
    }
    for name, run in cases.items():
        for leak in ("sk-canary", scratch):
            if leak in run["stdout"] + run["stderr"]:
                raise SystemExit(f"{name}: {leak!r} reached the capture")
    path = os.path.join(HERE, f"auth_list.{version}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(path)


if __name__ == "__main__":
    main()
