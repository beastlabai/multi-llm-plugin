"""The helpers several provider test files share.

Imported by the ``test_provider_*.py`` files; not a test module itself.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider
from tests import opencode_fixtures as oc_pin


def _envelope_sweep_names():
    """Every provider id in the registry, resolved at class-definition time.

    A local twin of `_registered_provider_names()` below: `@pytest.mark.parametrize`
    evaluates its argument while a class body runs, so a class that uses this one
    cannot wait on a later definition.
    """
    from utils.provider_registry import _PROVIDERS

    return sorted(_PROVIDERS)


def _registered_provider_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one."""
    from utils.provider_registry import _PROVIDERS

    return list(_PROVIDERS)


C1_FIXTURES = Path(__file__).parent / "fixtures"


def _c1_capture(provider, name):
    return (C1_FIXTURES / provider / name).read_bytes().decode("utf-8")


def _c1_adapter(provider):
    from utils.provider_registry import _PROVIDERS

    return _PROVIDERS[provider]


def _c1_without_the_hook(monkeypatch, provider):
    """Put one adapter back on the pre-C1 haystack, in place."""
    monkeypatch.setattr(type(_c1_adapter(provider)), "vendor_text",
                        LLMProvider.vendor_text)


def _oc_event(event_type, **fields):
    """One compact `run --format json` line, `type` first like opencode writes."""
    return json.dumps({"type": event_type, **fields}, separators=(",", ":"))


#: A vendor `error` event carrying a 429, in the shape `describe_failure` reads.
OC_429_EVENT = _oc_event(
    "error", timestamp=1, sessionID="ses_0",
    error={"name": "APIError", "data": {
        "message": "Rate limit exceeded", "statusCode": 429, "isRetryable": True,
    }},
)


#: A `--print-logs` stderr line for a real opencode 429.
OC_429_STDERR = (
    'timestamp=x level=ERROR message=failed '
    'error="AI_APICallError: Too Many Requests" statusCode=429\n'
)


#: T2.4: every committed opencode capture, graded through `classify_failure`
#: before T2.1/T4.3 (identity `vendor_text`, no structural check) and after.
#: One cell moves: the b38de8bb denied turn, from `rate_limited` to
#: `runner_error`. `stdout`/`stderr` name fixture files (or carry literal
#: synthetic text); `None` is an empty stream.
OC_GRADED_CAPTURES = (
    # id,                       stdout,                                   stderr,                                    exit, before,          after
    ("auth_error",             ("rig", "auth_error"),                    ("rig", "auth_error.stderr"),             1,    "auth_failed",   "auth_failed"),
    ("invalid_model",          ("rig", "invalid_model"),                 ("rig", "invalid_model.stderr"),          1,    "invalid_model", "invalid_model"),
    ("run_review",             ("rig", "run_review"),                    ("rig", "run_review.stderr"),             0,    None,            None),
    ("denied_turn",            ("rig", "denied_turn"),                   ("rig", "denied_turn.stderr"),            0,    "rate_limited",  "runner_error"),
    ("continue_loop_on_deny",  ("rig", "s41_cont_read"),                 ("rig", "s41_cont_read.stderr"),          0,    None,            None),
    ("synthetic 429 event",    ("literal", OC_429_EVENT),                None,                                      1,    "rate_limited",  "rate_limited"),
    ("synthetic 429 stderr",   None,                                     ("literal", OC_429_STDERR),                1,    "rate_limited",  "rate_limited"),
)


def _oc_stream(spec):
    if spec is None:
        return ""
    kind, value = spec
    if kind == "rig":
        case, _, stream = value.partition(".")
        return oc_pin.read(case, ".stderr.txt" if stream == "stderr" else ".jsonl")
    return value
