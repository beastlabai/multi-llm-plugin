#!/usr/bin/env python3
"""Render the dsh adapter's own dispatch into the capture kit's out dir.

Writes ``<out>/dsh/argv.json`` (``build_command_for_transport(..., "stdin")``),
``<out>/dsh/env.json`` (``get_env`` without ``DSH_HOME``, which the driver
makes per row) and ``<out>/dsh/patch.yml`` (:data:`DSH_PROFILE_PATCH` with only
its ``baseURL`` pointed at the in-container server). Nothing else is changed,
so a capture runs the argv, environment and profile a real dispatch runs.

    render.py <out dir> [<model spec>]
"""
import json
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SKILL))

from utils.providers import dsh  # noqa: E402

PRODUCTION_URL = "https://openrouter.ai/api/v1"
STUB_URL = "http://127.0.0.1:8099/v1"


def main() -> int:
    out = Path(sys.argv[1]) / "dsh"
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash-1017"
    out.mkdir(parents=True, exist_ok=True)
    provider = dsh.DshProvider()
    # The gate probe needs a dsh on PATH; the image's binary is the one the
    # rows run, so the flags are confirmed there (incontainer.py), not here.
    provider.missing_flags = lambda flags: ()  # type: ignore[method-assign]
    argv, stdin = provider.build_command_for_transport("PROMPT", model, "stdin")
    assert stdin == "PROMPT" and argv[-1] == "-", argv
    env = provider.get_env(model)
    # Never a real key in the kit: the driver sets the stub key itself.
    env.pop(dsh.DSH_CREDENTIAL_ENV, None)
    home = env.pop(dsh.HOME_ENV, None)
    if home:
        dsh._HOMES.release(home)
    assert dsh.DSH_PROFILE_PATCH.count(PRODUCTION_URL) == 1
    (out / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    (out / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    (out / "patch.yml").write_text(dsh.DSH_PROFILE_PATCH.replace(PRODUCTION_URL, STUB_URL))
    print(json.dumps({"argv": argv, "env": env}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
