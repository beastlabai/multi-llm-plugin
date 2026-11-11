#!/usr/bin/env python3
"""Write argv.json: each CLI's argv as its adapter's ``build_command`` renders
it for redirect.yaml's ``model_arg`` and a dispatch timeout of
:data:`TIMEOUT_S`, with the prompt as ``PROMPT_PLACEHOLDER``.

The review fan-out always passes its timeout to ``build_command``
(``llm_client``, ``timeout_s=effective_timeout``); of these five adapters only
qwen-code renders it, as ``--max-wall-time`` (30 s under the timeout), which is
what ends its 429-forever row on qwen's own budget inside the capture deadline.

``build_command`` runs against the image's executables, never the host's PATH
(:func:`image_which`): kilocode's adapter names whichever of ``kilocode`` and
``kilo`` resolves, and the argv a container runs must name the binary the image
has, whatever the machine rendering it has installed.

Run from the skill root: ``uv run --project . -- python
tests/fixtures/row_capture/make_argv.py``. ``tests/test_row_capture_kit.py``
asserts the committed file is still what the adapters render.
"""
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional
from unittest import mock

import yaml

HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[2]
sys.path.insert(0, str(SKILL))

from utils.provider_registry import _PROVIDERS  # noqa: E402

#: The adapters this kit captures (README.md here).
CLIS = ("codex", "kilocode", "kimi", "opencode", "qwen-code")

#: The dispatch timeout the argv is rendered for; under incontainer.py's
#: default deadline (240 s), as a real slot's timeout is under its kill.
TIMEOUT_S = 230


#: Where the image puts each CLI's PATH wrapper (tests/container/Dockerfile).
IMAGE_BIN_DIR = "/opt/mllm/bin"


def _redirect_rows() -> list:
    rows = yaml.safe_load(
        (SKILL / "tests" / "container" / "redirect.yaml").read_text(encoding="utf-8")
    )["clis"]
    return [row for row in rows if row["cli"] in CLIS]


def model_args() -> dict:
    return {row["cli"]: row["model_arg"] for row in _redirect_rows()}


def image_binaries() -> dict:
    """Each CLI's executable in the image: redirect.yaml's ``binary``, else the
    CLI id (the wrapper contract), which is the name the Dockerfile links."""
    return {row["cli"]: row.get("binary", row["cli"]) for row in _redirect_rows()}


def image_which() -> Callable[..., Optional[str]]:
    """``shutil.which`` as it answers inside the image, for these CLIs: their
    :func:`image_binaries` resolve under :data:`IMAGE_BIN_DIR`, nothing else."""
    present = frozenset(image_binaries().values())

    def which(name: str, *_args, **_kwargs) -> Optional[str]:
        return f"{IMAGE_BIN_DIR}/{name}" if name in present else None

    return which


def render() -> dict:
    models = model_args()
    with mock.patch.object(shutil, "which", image_which()):
        return {
            cli: _PROVIDERS[cli].build_command(
                "PROMPT_PLACEHOLDER", models[cli], timeout_s=TIMEOUT_S
            )
            for cli in CLIS
        }


if __name__ == "__main__":
    (HERE / "argv.json").write_text(
        json.dumps(render(), indent=2) + "\n", encoding="utf-8"
    )
