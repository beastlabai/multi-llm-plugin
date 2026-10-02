"""Where the opencode captures for the one supported release live.

opencode supports its newest release only, and that release is both
``OpenCodeProvider.min_version`` and ``verified_version`` (the pin). Every
capture the adapter is graded on is named for the pin, so a bump
(``make opencode-bump VERSION=<x.y.z>``) re-captures them under the new name and
every consumer here follows with no edit:

* ``capture/<pin>/<case>.*`` — the offline rig (``capture/run.sh``): the
  permission cases plus the parser's stream cases (``run_review``,
  ``denied_turn``, ``auth_error``, ``invalid_model``);
* ``auth_list.<pin>.json`` — ``auth_list_capture.py``;
* ``debug_config.<pin>.json`` — ``debug_config_capture.py``.
"""

from __future__ import annotations

from pathlib import Path

from utils.providers.opencode import OpenCodeProvider

PIN = OpenCodeProvider.verified_version
ROOT = Path(__file__).parent / "fixtures" / "opencode"
CAPTURE = ROOT / "capture" / PIN
AUTH_LIST = ROOT / f"auth_list.{PIN}.json"
DEBUG_CONFIG = ROOT / f"debug_config.{PIN}.json"


def capture(case: str, suffix: str = ".jsonl") -> Path:
    """``capture/<pin>/<case><suffix>``: ``.jsonl`` is stdout, ``.stderr.txt`` stderr."""
    return CAPTURE / f"{case}{suffix}"


def read(case: str, suffix: str = ".jsonl") -> str:
    return capture(case, suffix).read_bytes().decode("utf-8")
