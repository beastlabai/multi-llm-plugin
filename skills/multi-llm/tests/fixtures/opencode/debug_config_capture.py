#!/usr/bin/env python3
"""Capture ``opencode debug config`` offline for plan §8 T8.1 ($0, no model call).

Writes ``debug_config.<version>.json`` beside this file, in the
``auth_list.1.18.32.json`` style (``opencode_version``, ``argv``, ``captured``,
``cases``). Every case runs the real binary from a scratch HOME under a cleared
environment (``env -i`` equivalent: only the variables the case names, plus
``HOME``/``PATH``), with ``OPENCODE_DISABLE_MODELS_FETCH=1`` and stdin
``/dev/null``. Nothing reads or writes the invoking user's own opencode config
or data. ``debug config`` makes no model call and needs no network.

**Every ``apiKey`` value is a canary** (``sk-canary-not-a-key-<n>``): the
command prints secrets in plaintext, which is exactly what the reducer in
``utils/providers/opencode.py`` (``_reduce_debug_config``) must never let out,
and the secret-hygiene tests assert the canaries appear in no refusal, doctor
detail or log line built from these captures.

Every absolute scratch path is rewritten to ``/rig``; nothing else is edited.

The in-image repeat (``mllm-e2e:local`` through ``tests/container/wrapper.py``)
is a separate, container-lane step and is not run by this script.

usage: TMPDIR=<scratch> debug_config_capture.py [--bin PATH]
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

#: The §4.2 step 1 permission block T4.1 emits under ``agent.<name>.permission``
#: — written out by hand here, so the capture shows the reducer ignoring it.
PERMISSION_BLOCK = {
    "external_directory": {"*": "deny"},
    "read": {"*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
    "doom_loop": "deny",
}

STUB_INLINE = {
    "npm": "@ai-sdk/openai-compatible",
    "options": {"baseURL": "http://127.0.0.1:9/v1", "apiKey": "sk-canary-not-a-key-1"},
    "models": {"stub-model": {}},
}
KEYLESS = {
    "npm": "@ai-sdk/openai-compatible",
    "options": {"baseURL": "http://127.0.0.1:9/v1"},
    "models": {"m1": {}},
}
ENVKEY = {
    "npm": "@ai-sdk/openai-compatible",
    "options": {"baseURL": "http://127.0.0.1:9/v1", "apiKey": "{env:MY_TEST_KEY}"},
    "models": {"m2": {}},
}
#: ``config_providers``-shaped: no credential, as multi-llm rebuilds it.
DECLARED_STUB = {
    "npm": "@ai-sdk/openai-compatible",
    "options": {"baseURL": "http://127.0.0.1:9/v1"},
    "models": {"stub-model": {}},
}

SCHEMA = "https://opencode.ai/config.json"


def doc(**body):
    return {"$schema": SCHEMA, **body}


#: name -> (description, argv tail, global files, project files, env, hermetic)
CASES = {
    "custom_inline_key": (
        "global opencode.json: custom `stub` (npm + baseURL) with an inline apiKey",
        ["debug", "config"],
        {"opencode.json": doc(provider={"stub": STUB_INLINE})}, {}, {}, False,
    ),
    "custom_keyless": (
        "global opencode.json: custom `keyless` with no apiKey",
        ["debug", "config"],
        {"opencode.json": doc(provider={"keyless": KEYLESS})}, {}, {}, False,
    ),
    "custom_envkey_unset": (
        "custom `envkey`, apiKey `{env:MY_TEST_KEY}`, variable unset",
        ["debug", "config"],
        {"opencode.json": doc(provider={"envkey": ENVKEY})}, {}, {}, False,
    ),
    "custom_envkey_set": (
        "custom `envkey`, apiKey `{env:MY_TEST_KEY}`, variable set",
        ["debug", "config"],
        {"opencode.json": doc(provider={"envkey": ENVKEY})}, {},
        {"MY_TEST_KEY": "sk-canary-not-a-key-3"}, False,
    ),
    "builtin_override_keyed": (
        "built-in `openrouter` in config with options.apiKey",
        ["debug", "config"],
        {"opencode.json": doc(provider={"openrouter": {
            "options": {"apiKey": "sk-canary-not-a-key-4"}}})}, {}, {}, False,
    ),
    "builtin_override_keyless": (
        "built-in `openrouter` in config, options without a key",
        ["debug", "config"],
        {"opencode.json": doc(provider={"openrouter": {"options": {"timeout": 30000}}})},
        {}, {}, False,
    ),
    "nothing_declared": (
        "no config anywhere (built-in `openrouter` with nothing is not declared)",
        ["debug", "config"], {}, {}, {}, False,
    ),
    "jsonc_commented": (
        "global opencode.jsonc with comments and trailing commas declaring `stub`",
        ["debug", "config"],
        {"opencode.jsonc": (
            '{\n  // a comment opencode tolerates\n  "$schema": "' + SCHEMA + '",\n'
            '  "provider": {\n    /* block comment */\n    "stub": {\n'
            '      "npm": "@ai-sdk/openai-compatible",\n'
            '      "options": {"baseURL": "http://127.0.0.1:9/v1", '
            '"apiKey": "sk-canary-not-a-key-5",},\n'
            '      "models": {"stub-model": {},},\n    },\n  },\n}\n'
        )}, {}, {}, False,
    ),
    "project_layer": (
        "project ./opencode.json only, declaring `stub`",
        ["debug", "config"],
        {}, {"opencode.json": doc(provider={"stub": STUB_INLINE})}, {}, False,
    ),
    "project_layer_disabled": (
        "project ./opencode.json with OPENCODE_DISABLE_PROJECT_CONFIG=1",
        ["debug", "config"],
        {}, {"opencode.json": doc(provider={"stub": STUB_INLINE})},
        {"OPENCODE_DISABLE_PROJECT_CONFIG": "1"}, False,
    ),
    "opencode_config_file": (
        "OPENCODE_CONFIG naming a file that declares `stub`",
        ["debug", "config"],
        {}, {}, {"OPENCODE_CONFIG": "@rig/explicit.json"}, False,
    ),
    "user_external_directory_rules": (
        "user rules for T8.2's external_directory reduction: top-level map plus "
        "agent.plan / agent.build layers, insertion order not alphabetical",
        ["debug", "config"],
        {"opencode.json": doc(
            permission={
                "external_directory": {
                    "/zz/*": "allow", "/shared/*": "allow",
                    "/shared/private/*": "deny", "*": "ask",
                },
                "*": "allow",
            },
            agent={
                "plan": {"permission": {"external_directory": {"/aa/*": "ask"}}},
                "build": {"permission": {"external_directory": "deny"}},
            },
        )}, {}, {}, False,
    ),
    "hermetic_config_content": (
        "the T4.1 hermetic env: empty XDG_CONFIG_HOME, project layer disabled, "
        "OPENCODE_CONFIG_CONTENT = config_providers `stub` block (no credential) "
        "plus the §4.2 permission block under agent.plan.permission; the user's "
        "global config (declaring `userstub`) is present under HOME and must vanish",
        ["debug", "config"],
        {"opencode.json": doc(provider={"userstub": STUB_INLINE})}, {},
        {"OPENCODE_CONFIG_CONTENT": json.dumps(doc(
            provider={"stub": DECLARED_STUB},
            agent={"plan": {"permission": PERMISSION_BLOCK}},
        ))}, True,
    ),
    "hermetic_pure_config_content": (
        "as hermetic_config_content, with the global --pure the hermetic argv renders",
        ["--pure", "debug", "config"],
        {"opencode.json": doc(provider={"userstub": STUB_INLINE})}, {},
        {"OPENCODE_CONFIG_CONTENT": json.dumps(doc(
            provider={"stub": DECLARED_STUB},
            agent={"plan": {"permission": PERMISSION_BLOCK}},
        ))}, True,
    ),
    "hermetic_pure_permission_only": (
        "hermetic --pure with a built-in prefix: OPENCODE_CONFIG_CONTENT holds "
        "only the permission block, so no provider is declared",
        ["--pure", "debug", "config"],
        {"opencode.json": doc(provider={"userstub": STUB_INLINE})}, {},
        {"OPENCODE_CONFIG_CONTENT": json.dumps(doc(
            agent={"plan": {"permission": PERMISSION_BLOCK}},
        ))}, True,
    ),
}


def write_files(root, files):
    os.makedirs(root, exist_ok=True)
    for name, body in files.items():
        text = body if isinstance(body, str) else json.dumps(body, indent=2)
        with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
            handle.write(text)


def run_case(binary, name, spec, scratch):
    _desc, tail, global_files, project_files, env_extra, hermetic = spec
    rig = os.path.join(scratch, name)
    home = os.path.join(rig, "home")
    workspace = os.path.join(rig, "ws")
    write_files(os.path.join(home, ".config", "opencode"), global_files)
    write_files(workspace, project_files)
    env = {
        "HOME": home,
        "PATH": "/usr/bin:/bin",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    if hermetic:
        empty = os.path.join(rig, "hermetic-config")
        os.makedirs(empty)
        env["XDG_CONFIG_HOME"] = empty
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    for key, value in env_extra.items():
        env[key] = value.replace("@rig", rig)
    if "OPENCODE_CONFIG" in env_extra:
        write_files(rig, {"explicit.json": doc(provider={"stub": STUB_INLINE})})
    with open(os.devnull, "rb") as devnull:
        done = subprocess.run(
            [binary, *tail], cwd=workspace, env=env, stdin=devnull,
            capture_output=True, text=True, timeout=60,
        )

    def scrub(text):
        return (text or "").replace(rig, "/rig")

    shown_env = {
        k: scrub(v) for k, v in env.items() if k not in ("HOME", "PATH")
    }
    return {
        "setup": _desc,
        "argv": ["opencode", *tail],
        "env": shown_env,
        "global_files": global_files,
        "project_files": project_files,
        "exit": done.returncode,
        "stdout": scrub(done.stdout),
        "stderr": scrub(done.stderr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=shutil.which("opencode") or "opencode")
    args = parser.parse_args()
    with open(os.devnull, "rb") as devnull:
        version = subprocess.run(
            [args.bin, "--version"], stdin=devnull, capture_output=True,
            text=True, env={"PATH": "/usr/bin:/bin", "HOME": tempfile.gettempdir()},
        ).stdout.strip()
    scratch = tempfile.mkdtemp(prefix="oc-debug-config-")
    try:
        cases = {
            name: run_case(args.bin, name, spec, scratch)
            for name, spec in CASES.items()
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out = {
        "opencode_version": version,
        "argv": ["opencode", "debug", "config"],
        "captured": (
            "2026-09-25, host, scratch HOME per case under a cleared env "
            "(HOME, PATH=/usr/bin:/bin, OPENCODE_DISABLE_MODELS_FETCH=1 plus the "
            "case's own env), stdin /dev/null, no network, no model call; every "
            "apiKey a sk-canary-not-a-key-<n> canary; scratch paths rewritten to /rig"
        ),
        "cases": cases,
    }
    path = os.path.join(HERE, f"debug_config.{version}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(path)


if __name__ == "__main__":
    main()
