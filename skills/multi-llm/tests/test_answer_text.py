"""A prose answer comes back whole, through every adapter family's real parser.

Plan: ``plans/ask-answer-recovery-and-false-success.md`` D2.2, task T2.1 (and
T2.4's rows for the adapters whose hooks it adds).

**Why a file of its own.** ``tests/test_salvage_text.py`` grades each capture's
``expect.salvage`` label through ``has_salvageable_json``, a JSON gate, so a
prose answer can be labelled neither ``answer`` nor ``partial`` there. The ask
lane's answers are prose. So each row here is a synthetic prose answer planted
in the adapter's **real** stdout shape — the committed success capture with its
answer string swapped for the prose (``_swap_answer``), or, for the two
adapters with no success capture, a shape built from what is committed — and
run through the real ``invoke_with_provider`` with only the subprocess stubbed
(``llm_client._run_provider_cli``, as ``tests/test_llm_client.py`` does). The
adapter's own ``parse_output``, its ``salvage_text`` hook or
``generic_answer_text``, the redaction and the cap all run, and
``details["answer_text"]`` must be the prose byte for byte.

**Two prose answers per row.** One has no JSON in it, so the parse ends
``PARSE_ERROR`` and the exit-0 failure return carries the key. One holds a
fenced JSON block — nemotron's shape (§2.1): the parse succeeds with dict
``data`` (and, but for pi and omp, no ``raw``), and the success return carries
the key. The two returns are different sites in ``invoke_with_provider``, and
each row pins which one it reached.

**The families.** The six hooks from ``cab9f89`` (pi, omp, codex, opencode,
copilot, muse); the generic rules (claude-code, cursor-agent, gemini and agy
unwrap a clean envelope, kilocode is ``"plain"``); and the adapters whose hooks
land with this plan — aider (T4.1) and goose, grok, vibe, cline, kimi and
qwen-code (T2.4). Without its hook, one of those last seven reaches
``generic_answer_text``: rule 1 or 2 gives ``""`` for the six JSON shapes, and
rule 3 keeps aider's banner. Either way its row is red.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client  # noqa: E402
from utils.llm_client import ANSWER_TEXT_KEY, invoke_with_provider  # noqa: E402
from tests.opencode_fixtures import PIN as _OPENCODE_PIN  # noqa: E402
from utils.provider_registry import get_provider  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

#: No JSON anywhere: every adapter's parse ends ``success: False`` / PARSE_ERROR.
PROSE_ONLY = (
    "## Short answer\n"
    "\n"
    "The retry ladder stops after the third attempt, so the fourth call in the\n"
    "plan never runs.\n"
    "\n"
    "- Step 2 reads the budget before it is set.\n"
    "- Step 5 already covers the rollback case.\n"
    "\n"
    "I would move the budget read after the config load and leave the rest as\n"
    "written."
)

#: A fenced JSON block inside the prose: the parse succeeds on the block, and
#: ``data`` is a dict that is not the answer (nemotron's loss, §2.1).
PROSE_WITH_JSON = (
    "Two options fit the plan.\n"
    "\n"
    "The second one needs this config block:\n"
    "\n"
    "```json\n"
    '{"retries": 3, "backoff": [5, 15]}\n'
    "```\n"
    "\n"
    "I would take the first: it changes one file and keeps the ladder as it is."
)

PROSES = {"prose-only": PROSE_ONLY, "prose-with-fenced-json": PROSE_WITH_JSON}


def _read(relative: str) -> str:
    return (FIXTURES / relative).read_bytes().decode("utf-8")


def _swap_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        return [_swap_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _swap_strings(item, old, new) for key, item in value.items()}
    return value


def _dump(value: Any, like: str) -> str:
    """Re-serialize in the capture's own layout: pretty when it was pretty."""
    if "\n  " in like:
        return json.dumps(value, indent=2, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _swap_answer(stdout: str, old: str, new: str) -> str:
    """The capture with every JSON string equal to ``old`` replaced by ``new``.

    A whole-stdout JSON value is walked as one; an event stream line by line,
    with every line that does not carry ``old`` kept byte for byte; plain text
    (aider) by substring. Streamed deltas that carry only a *chunk* of ``old``
    are left as they were: the adapters here read the finished message.
    """
    try:
        whole = json.loads(stdout)
    except ValueError:
        whole = None
    if isinstance(whole, (dict, list)):
        swapped = _swap_strings(whole, old, new)
        assert swapped != whole, "the capture does not carry the answer string"
        return _dump(swapped, stdout) + "\n"
    out = []
    hit = False
    for line in stdout.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        swapped = _swap_strings(record, old, new)
        if swapped == record:
            out.append(line)
            continue
        hit = True
        out.append(_dump(swapped, line) + ("\n" if line.endswith("\n") else ""))
    if hit:
        return "".join(out)
    assert old in stdout, "the capture does not carry the answer string"
    return stdout.replace(old, new)


def _hook_answer(provider: str) -> Callable[[str], str]:
    """The six ``cab9f89`` hooks already say where the answer is."""
    return lambda stdout: get_provider(provider).salvage_text(stdout)


def _last(records, key: str, value: str) -> Dict[str, Any]:
    return [r for r in records if isinstance(r, dict) and r.get(key) == value][-1]


def _lines(stdout: str):
    out = []
    for line in stdout.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


#: Where each capture's answer string sits, read without the adapter's hook, so
#: a missing hook turns its row red on the assertion rather than in setup.
_FIXTURE_ANSWER: Dict[str, Callable[[str], str]] = {
    "fx": lambda s: json.loads(s)["final_output"],
    "pi": _hook_answer("pi"),
    "omp": _hook_answer("omp"),
    "codex": _hook_answer("codex"),
    "opencode": _hook_answer("opencode"),
    "copilot": _hook_answer("copilot"),
    "muse": _hook_answer("muse"),
    "claude-code": lambda s: json.loads(s)["result"],
    "cursor-agent": lambda s: json.loads(s)["result"],
    "gemini": lambda s: json.loads(s)["response"],
    "aider": lambda s: next(ln for ln in s.splitlines() if ln.startswith("[{")),
    "goose": lambda s: _last(json.loads(s)["messages"], "role", "assistant")[
        "content"][0]["text"],
    "grok": lambda s: json.loads(s)["text"],
    "vibe": lambda s: _last(json.loads(s), "role", "assistant")["content"][0]["text"],
    "cline": lambda s: _last(_lines(s), "type", "run_result")["text"],
    "kimi": lambda s: _last(_lines(s), "role", "assistant")["content"],
    "qwen-code": lambda s: _last(json.loads(s), "type", "result")["result"],
}

#: provider -> (model spec, committed success capture).
_CAPTURED: Dict[str, Tuple[str, str]] = {
    "fx": ("fx:test/model", "fx/success.stdout.txt"),
    "pi": ("pi:openrouter/deepseek/deepseek-v4.1-flash", "pi/success.stdout.txt"),
    "omp": ("omp:stub/stub-model", "omp/success.stdout.txt"),
    "codex": ("codex:gpt-5", "codex/gpt5_sol_single_message.jsonl"),
    # opencode has no success row: its capture-rig reply is the prose
    # `FINAL ANSWER`, which is what gets swapped.
    "opencode": ("opencode:openrouter/stub", f"opencode/capture/{_OPENCODE_PIN}/s41_cont_read.jsonl"),
    "copilot": ("copilot:gpt-5", "copilot/success.jsonl"),
    "muse": ("muse:stub-model", "muse/success.stdout.txt"),
    "claude-code": ("claude-code:haiku", "claude-code/success.stdout.txt"),
    "cursor-agent": ("cursor-agent:gpt-4", "cursor-agent/success.stdout.txt"),
    "gemini": ("gemini:gemini-2.5-flash", "gemini/success.stdout.txt"),
    "aider": ("aider:openai/stub-model", "aider/success.stdout.txt"),
    "goose": ("goose:stub-model", "goose/success.stdout.txt"),
    "grok": ("grok:stub-model", "grok/success.stdout.txt"),
    "vibe": ("vibe:stub-model", "vibe/success-json.json"),
    "cline": ("cline:stub-model", "cline/success.stdout.txt"),
    "kimi": ("kimi:stub-model", "kimi/success.stdout.txt"),
    "qwen-code": ("qwen-code:stub-model", "qwen-code/success.stdout.json"),
}


def _captured_row(provider: str, prose: str) -> Tuple[str, str]:
    spec, relative = _CAPTURED[provider]
    capture = _read(relative)
    old = _FIXTURE_ANSWER[provider](capture)
    assert isinstance(old, str) and old.strip(), f"{relative}: no answer string"
    stdout = _swap_answer(capture, old, prose)
    assert prose in stdout or json.dumps(prose)[1:-1] in stdout
    return spec, stdout


def _agy_row(prose: str, *, error_key: bool) -> Tuple[str, str]:
    """agy has no success capture (§14): its one capture's envelope, made clean.

    ``error_key`` keeps the ``error`` field the captured envelope always
    carries, empty — the uncaptured success shape could look either way.
    """
    envelope = json.loads(_read("agy/auth-error.stdout.txt"))
    envelope.update(conversation_id="conv-1", status="SUCCESS", response=prose)
    if error_key:
        envelope["error"] = ""
    else:
        envelope.pop("error", None)
    return "agy:stub-model", json.dumps(envelope)


#: (row id, builder). The id names the adapter; the builder takes the prose.
ROWS = [
    *[(provider, lambda prose, p=provider: _captured_row(p, prose)) for provider in _CAPTURED],
    ("agy", lambda prose: _agy_row(prose, error_key=False)),
    ("agy-empty-error-field", lambda prose: _agy_row(prose, error_key=True)),
    # kilocode has no captures (§14); its stdout IS the reply ("plain").
    ("kilocode", lambda prose: ("kilocode:moonshotai/kimi-k3", prose)),
]

def _invoke(spec: str, stdout: str) -> Dict[str, Any]:
    binary = spec.split(":", 1)[0]

    def spawn(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    with patch.object(llm_client, "_run_provider_cli", side_effect=spawn), \
         patch("shutil.which", return_value=f"/usr/bin/{binary}"):
        return invoke_with_provider("Q", spec, want_answer_text=True)


@pytest.mark.parametrize("prose_id", sorted(PROSES))
@pytest.mark.parametrize("row,build", ROWS, ids=[row for row, _ in ROWS])
def test_a_prose_answer_comes_back_whole(row: str, build, prose_id: str) -> None:
    prose = PROSES[prose_id]
    spec, stdout = build(prose)

    result = _invoke(spec, stdout)

    # Each prose exercises its own return site (module docstring).
    if prose_id == "prose-only":
        assert result["error_code"] == llm_client.ERROR_PARSE_ERROR, result.get("error")
    else:
        assert result["success"] is True, result.get("error")
        assert isinstance(result["data"], dict)
    answer = result["details"].get(ANSWER_TEXT_KEY)
    assert answer is not None, f"{row}: no answer_text key ({result.get('error')})"
    assert answer == prose, f"{row}: {answer[:300]!r}"


def test_every_adapter_family_has_a_row() -> None:
    """A registry adapter with no row here is a family nobody checked."""
    from utils.provider_registry import _PROVIDERS

    covered = {row.split("-empty-")[0] for row, _ in ROWS}
    assert covered == set(_PROVIDERS)


@pytest.mark.parametrize("provider", sorted(_CAPTURED))
def test_the_swap_really_replaced_the_captured_answer(provider: str) -> None:
    """The row is not the capture: its own answer is gone from the stream."""
    spec, relative = _CAPTURED[provider]
    capture = _read(relative)
    old = _FIXTURE_ANSWER[provider](capture)

    _spec, stdout = _captured_row(provider, PROSE_ONLY)

    assert stdout != capture
    result = _invoke(spec, stdout)
    assert old.strip() not in (result["details"].get(ANSWER_TEXT_KEY) or "")
