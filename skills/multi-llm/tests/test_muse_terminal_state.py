"""C8 — a muse run that exited 1 never reaches a fan-out slot as an answer.

``MuseProvider.parse_output``'s first branch used to fail a non-``completed``
run only ``and not full``, so a run whose salvaged text happened to be valid
JSON came back ``success: True``. ``tests/fixtures/muse/manifest.yaml``'s
``truncated-json-answer`` row grades the adapter half of that; this module
grades the half the row cannot reach — **what Lane B does with those bytes** —
because the defect only ever mattered one caller deep:

* ``llm_client`` (Lane A) is safe by accident: it short-circuits on
  ``returncode != 0`` and never calls ``parse_output`` at all;
* ``subagent_executor`` (Lane B) is not, and deliberately so. Its
  ``extract_result`` never sees an exit code, because the executor's
  registry-wide rule is that an exit code is *evidence, never the verdict* —
  several CLIs exit non-zero on a warning while still printing the answer. So a
  schema-valid result recovered from a failed muse run was accepted as the
  task's answer with only an annotation.

That is why the fix is in the adapter and why ``utils/subagent_executor.py`` is
byte-identical after C8: the evidence that condemns this run is muse's own
``run.terminal.failed`` record and its reason sentence, which only the adapter
parses. The tests below pin the outcome in **both** directions — a recorded
non-``completed`` terminal fails whatever text was salvaged, and a *missing*
terminal record still parses — because "drop a conjunct" is only safe if the
``None`` half of the same condition provably did not move.
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.providers.muse import MuseProvider  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "muse"

#: muse's own words on the failed run, from the capture's ``run.terminal.failed``
#: record. The whole point of the fix is that this sentence — not "no valid JSON
#: found" and not a fabricated success — is what the slot is told.
TERMINAL_REASON = "model failed after 10 attempts: stream ended before terminal event"


def _capture(name):
    """The committed bytes of one row, exactly as muse wrote them."""
    stdout = (FIXTURES / f"{name}.stdout.txt").read_text(encoding="utf-8")
    stderr_path = FIXTURES / f"{name}.stderr.txt"
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    return stdout, stderr


def _without_terminal_record(stdout):
    """The same stream with every ``run.terminal.*`` record dropped.

    muse writes no terminal record on some clean runs, and ``None`` is the case
    the fixed condition still lets through. No committed capture has that shape
    — every row in the set reaches a terminal — so the guard is built by
    *removing* the record from a real green run rather than by inventing a
    stream: every other byte is still muse's own.
    """
    kept = []
    for line in stdout.split("\n"):
        stripped = line.strip()
        if stripped:
            try:
                event = json.loads(stripped)
            except ValueError:
                event = None
            if isinstance(event, dict) and str(
                event.get("payload_type") or ""
            ).startswith("run.terminal."):
                continue
        kept.append(line)
    return "\n".join(kept)


class TestAFailedRunIsNeverAnAnswer:
    """The defect, from the adapter out to the rung Lane B lands it on."""

    def test_the_adapter_refuses_the_salvaged_fragment(self):
        stdout, stderr = _capture("truncated-json-answer")

        parsed = MuseProvider().parse_output(stdout, stderr)

        assert parsed["success"] is False, (
            "a run that exited 1 with `run.terminal.failed` was reported as a "
            "success because its salvaged text happened to parse"
        )
        assert parsed["data"] is None
        assert TERMINAL_REASON in parsed["error"]

    def test_the_fragment_really_does_still_parse(self):
        """Otherwise this row would prove nothing.

        The capture's cut point (``--trunc-bytes 2100``, inside
        ``response.output_item.done``) was chosen so the delivered deltas hold a
        *syntactically complete* array — one per failed attempt, ten of them
        concatenated with no separator, which is why the trap is the extractor's
        salvage rather than a whole-string parse. If that ever stopped being
        true the tests above would pass for the wrong reason: the text would
        simply fail to parse and the terminal state would never be what decided
        it. So the recoverable answer is asserted rather than assumed.
        """
        from utils.json_extractor import extract_json_from_text

        stdout, _stderr = _capture("truncated-json-answer")
        scan = MuseProvider()._scan(stdout)

        salvaged = extract_json_from_text(scan["tail"].strip(), prefer_arrays=True)

        assert salvaged["success"] is True
        assert salvaged["data"] == [
            {"id": "S001", "title": "Bound the retry ladder", "severity": "high"}
        ]

    def test_lane_b_rejects_it_at_rung_5(self):
        """``extract_result`` is the executor's parse path (D3k rungs 4 and 5).

        Before C8 this call returned the salvaged array and the slot recorded a
        result. It now raises, and the rung matters: text WAS recovered, so this
        is rung 5 (``unparseable_result``) rather than rung 4
        (``runner_error``) — the split the executor makes on ``raw``.
        """
        stdout, stderr = _capture("truncated-json-answer")

        with pytest.raises(executor.ResultRejected) as excinfo:
            executor.extract_result(
                MuseProvider(), stdout, stderr, prefer_arrays=True
            )

        assert excinfo.value.status == executor.STATUS_UNPARSEABLE
        assert TERMINAL_REASON in excinfo.value.message

    def test_the_failing_branch_keeps_raw_so_the_rung_split_can_read_it(self):
        """``raw: None`` would silently demote this to rung 4.

        The executor chooses rung 5 over rung 4 by asking whether the adapter
        recovered any text at all, and it asks ``parsed["raw"]``. muse's failing
        branch returns the whole stream, which is what keeps a failed run that
        DID produce output distinguishable from one that produced none.
        """
        stdout, stderr = _capture("truncated-json-answer")

        parsed = MuseProvider().parse_output(stdout, stderr)

        assert parsed["raw"] == stdout
        assert parsed["raw"].strip(), "an empty `raw` is rung 4, not rung 5"

    def test_every_failure_row_in_the_set_keeps_its_stream_on_raw(self):
        """The same property across the whole committed set, not one row.

        A failed muse run must never answer ``raw: None``: rung 4 means "the
        runner produced nothing", and every one of these captures produced 17 KB
        or more.
        """
        provider = MuseProvider()
        for name in (
            "auth-error",
            "invalid-model",
            "context-overflow",
            "error-envelope",
            "stdout-error",
            "rate-limit",
            "truncated",
            "truncated-after-deltas",
            "truncated-json-answer",
        ):
            stdout, stderr = _capture(name)
            parsed = provider.parse_output(stdout, stderr)

            assert parsed["success"] is False, name
            assert parsed["raw"] == stdout, name


class TestTheOtherTwoTerminalCasesDidNotMove:
    """Terminal state before salvaged text — but in **one direction only**."""

    def test_a_completed_run_still_parses(self):
        stdout, stderr = _capture("success")

        data = executor.extract_result(
            MuseProvider(), stdout, stderr, prefer_arrays=True
        )

        assert data == [
            {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
            {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
        ]

    def test_a_run_with_no_terminal_record_at_all_still_parses(self):
        """The absence of a verdict is not a verdict.

        ``None`` is excluded by the same condition C8 edited, so this is the
        regression guard on the half that must NOT move: muse writes no terminal
        record on some clean runs, and failing those would turn a green fan-out
        slot into a rung-5 rejection.
        """
        stdout, stderr = _capture("success")
        no_terminal = _without_terminal_record(stdout)
        provider = MuseProvider()

        assert provider._scan(no_terminal)["terminal"] is None

        parsed = provider.parse_output(no_terminal, stderr)

        assert parsed["success"] is True
        assert parsed["data"] == [
            {"id": "S001", "title": "Bound the retry ladder", "severity": "high"},
            {"id": "S002", "title": "Strip the vendor token", "severity": "medium"},
        ]

    def test_an_empty_completed_run_is_still_the_no_output_failure(self):
        """``ephemeral-only``: exit 0, ``completed``, and an empty answer.

        It must keep landing on the "no output events" branch with muse's
        background-task diagnosis attached, rather than being swept into the new
        terminal-state refusal — its terminal really is ``completed``.
        """
        stdout, stderr = _capture("ephemeral-only")

        parsed = MuseProvider().parse_output(stdout, stderr)

        assert parsed["success"] is False
        assert "Muse reported an error" in parsed["error"]
        assert "did not reach a terminal state" in parsed["error"]


class TestTheFixIsInTheAdapterAndNotInTheExecutor:
    """The rejected candidate, pinned so it cannot be taken later by accident."""

    def test_the_executors_parse_path_cannot_see_an_exit_code(self):
        """Which is the whole argument for fixing this in the adapter.

        The alternative was to give ``subagent_executor`` the exit-code check
        ``llm_client`` has by accident. It is rejected rather than held open:
        the executor's "exit code is evidence, never the verdict" rule is
        registry-wide and deliberate, so making a non-zero exit fatal there
        would fix one adapter by changing how *every* adapter's exit code is
        read — and would newly fail runs that are answering correctly today.
        ``extract_result`` taking no exit code at all is that decision as a
        signature: this lane's verdict comes from the streams and the adapter
        that understands them.
        """
        parameters = inspect.signature(executor.extract_result).parameters

        assert "exit_code" not in parameters
        assert "returncode" not in parameters
