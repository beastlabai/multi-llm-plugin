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

import re
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


def did_you_mean(stderr: str) -> str:
    """The ``Did you mean: …?`` suggestion a captured model-not-found printed.

    Read from the capture, never written down: the list and its order come
    from the model catalog bundled with the release (it changed between 1.18.32
    and 1.18.34), so a test that spelled it would go red on a bump for a reason
    that is not a parser change.
    """
    found = re.search(r"Did you mean: [^?\"]+\?", stderr)
    assert found, "the capture carries no `Did you mean` suggestion"
    return found.group(0)
