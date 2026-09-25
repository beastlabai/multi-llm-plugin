"""Provider lane, the denied turn: an out-of-workspace read goes BACK TO THE MODEL.

plans/opencode-validator-false-rate-limit-and-denied-turn.md §4.5, T4.9. On
opencode 1.18.32, `opencode run` answers every permission `ask` with an
auto-reject, and a rejected permission ENDS the agent loop: exit 0, no answer
(run b38de8bb, a validator that probed `/etc/postgresql`). `OpenCodeProvider`
therefore ships an agent-layer `external_directory: {"*": "deny"}` row through
`OPENCODE_CONFIG_CONTENT` — T4.1 on `agent-readonly` (and `generate`), T4.7 on
`agent-write` — because a `deny` is reported to the model and the loop goes on.

Until T4.9 this lane could not see any of it: `tests/container/wrapper.py`
rebuilt every child's environment from an allowlist that dropped
`OPENCODE_CONFIG_CONTENT`, so the in-image opencode ran without the block. The
wrapper now carries it (and refuses a value holding an `apiKey`), and these rows
prove the delivery end to end, at $0:

* the scenario (`scripted_opencode_read_deny.yaml`) is `container_full_chain.yaml`'s
  rules plus ONE scripted chat tool call: every opencode request advertising
  `read` is handed `read /etc/hostname` first;
* a second request must then reach the stub in the SAME exec session, carrying
  opencode's `deny` text as the tool result — the model received the refusal;
  an auto-reject would have ended the turn with no second request at all;
* `after_output: text` answers that second request with the phase's own fixture,
  so the phase completes and the row exits 0 — which, with T4.3's structural
  classifier, also means no turn ended on a non-`stop` `step_finish` (that is a
  recorded `runner_error` attempt, and a failed phase);
* no failed-attempt capture the executor kept carries an `auto-rejecting` line.

Two rows, one per profile the plan names: `deny-opencode-readonly` runs
`review-plan --exec-subagents` (the `generate` fan-out and the `agent-readonly`
validators, `--agent plan`), and `deny-opencode-write` adds `apply-suggestions
--exec-subagents` (the `agent-write` batches, `--agent build`). Both graded
structurally here; neither is a golden bundle, because every fixture it answers
with is the chain row's, which `chain-opencode`'s bundle already pins.
"""

from __future__ import annotations

import json

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, requires_docker()]

REDIRECT = harness.HERE / "redirect.yaml"

#: In-image path; the stub resolves the scenario's `fixture:` paths relative to it.
SCENARIO = ("/opt/mllm/skill/tests/fixtures/e2e/scenarios/"
            "scripted_opencode_read_deny.yaml")
PLAN = "plans/auth-feature.md"
CALL_ID = "call_mllm_read_deny"

#: How opencode words a `deny` rule to the model (T4.6 rig, `SCHEMA.md`), and
#: how it words an auto-rejected `ask` — the outcome these rows exist to rule out.
DENY_TEXT = ("The user has specified a rule which prevents you from using this "
             "specific tool call")
REJECT_TEXT = "The user rejected permission to use this specific tool call"
AUTO_REJECT_LINE = "auto-rejecting"

#: Row id -> (commands, the `--agent` every graded exec of that profile names,
#: the agent-layer key its `OPENCODE_CONFIG_CONTENT` must carry the rows under).
ROWS = {
    "deny-opencode-readonly": (
        [f"review-plan {PLAN} --quick --yes --exec-subagents"],
        "plan"),
    "deny-opencode-write": (
        [f"review-plan {PLAN} --quick --yes --exec-subagents",
         f"apply-suggestions {PLAN} --yes --exec-subagents"],
        "build"),
}


def _opencode_entry() -> dict:
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return next(r for r in data["clis"] if r.get("cli") == "opencode")


ENTRY = _opencode_entry()
SPEC = f"opencode:{ENTRY.get('model_arg', 'stub-model')}"


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module", params=sorted(ROWS), ids=sorted(ROWS))
def deny(request, posture, utc):
    row_id = request.param
    commands, _agent = ROWS[row_id]
    row = harness.Row(
        row_id=row_id,
        lane="provider",
        cli="opencode",
        spec=[SPEC],
        # Both driver keys: every validator and every write batch is opencode.
        runner=SPEC,
        command=commands[0],
        commands=commands[1:],
        scenario=SCENARIO,
        row_timeout_s=int(ENTRY.get("chain_row_timeout_s",
                                    ENTRY.get("row_timeout_s", 600))),
    )
    return harness.run_row(row, posture=posture, utc=utc)


def _runs(result: harness.RowResult) -> list[dict]:
    """The call records of opencode MODEL turns (`opencode run …`).

    `auth list`, `--version`, `models` and `debug config` spawn through the
    same wrapper and make no model call, so they are not graded here.
    """
    return [r for r in result.call_log
            if r.get("cli") == "opencode" and "run" in (r.get("argv") or [])[1:2]]


def _agent(record: dict) -> str | None:
    argv = record.get("argv") or []
    if "--agent" in argv:
        index = argv.index("--agent")
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


def _scripted(result: harness.RowResult, exec_id: str) -> list[dict]:
    return [e["scripted"] for e in result.stub_log
            if e.get("session") == exec_id and isinstance(e.get("scripted"), dict)]


class TestTheRowFinished:
    def test_the_row_did_not_hit_its_cap(self, deny):
        assert not deny.timed_out, (
            f"{deny.row.row_id}: {deny.cap_reason} after {deny.wall_clock_s}s")

    def test_the_row_exited_zero(self, deny):
        """A turn that ended on the refusal leaves no answer: the phase fails.

        On `deny-opencode-write` this is red until T4.7 ships the `build`
        agent's deny rows — the built-in `ask` auto-rejects and ends the turn.
        """
        assert deny.exit_code == 0, (
            f"{deny.row.row_id}: exited {deny.exit_code}\n"
            f"--- container stderr (tail) ---\n{deny.stderr[-2000:]}")

    def test_no_prompt_went_unmatched_or_unserved(self, deny):
        assert not [e for e in deny.stub_log if e.get("unmatched")]
        assert not [(e["method"], e["path"]) for e in deny.stub_log
                    if e.get("unhandled")]


class TestTheBlockReachedTheChild:
    """T4.9's carry, read off the wrapper's own call record."""

    def test_every_graded_turn_carried_the_agent_layer_deny(self, deny):
        _commands, agent = ROWS[deny.row.row_id]
        graded = [r for r in _runs(deny) if _agent(r) == agent]
        assert graded, (
            f"{deny.row.row_id}: no `opencode run --agent {agent}` exec at all — "
            f"the row never dispatched the profile it grades")
        for record in graded:
            content = (record.get("captured_env") or {}).get(
                "OPENCODE_CONFIG_CONTENT")
            assert content, (
                f"exec {record['exec_id']} (--agent {agent}) received no "
                f"OPENCODE_CONFIG_CONTENT — the wrapper did not carry it, or the "
                f"adapter did not emit it for this profile")
            document = json.loads(content)
            rows = document["agent"][agent]["permission"]["external_directory"]
            assert next(iter(rows.items())) == ("*", "deny"), (
                f"exec {record['exec_id']}: `external_directory` does not open "
                f"with the catch-all deny: {rows}")


class TestTheRefusalWentBackToTheModel:
    """The whole point: a second request, carrying the deny, in the same session."""

    def test_every_called_turn_came_back_with_the_deny(self, deny):
        called = 0
        for record in _runs(deny):
            phases = _scripted(deny, record["exec_id"])
            if not any(p.get("phase") == "call" for p in phases):
                continue
            called += 1
            returned = [p for p in phases if p.get("phase") == "output-to-text"]
            assert returned, (
                f"exec {record['exec_id']} (--agent {_agent(record)}): the stub "
                f"served the `read` call and no second request followed — the "
                f"turn ended on the refusal (the auto-reject shape), and the "
                f"model never saw it")
            for phase in returned:
                output = phase.get("output") or ""
                assert DENY_TEXT in output, (
                    f"exec {record['exec_id']}: the tool result is not "
                    f"opencode's deny text: {output[:300]!r}")
                assert REJECT_TEXT not in output
        assert called, (
            f"{deny.row.row_id}: no opencode turn was handed the scripted "
            f"`read` — the scenario did not arm, or no request advertised `read`")

    def test_the_graded_profile_itself_was_called_and_denied(self, deny):
        _commands, agent = ROWS[deny.row.row_id]
        denied = [r for r in _runs(deny) if _agent(r) == agent
                  and any(p.get("phase") == "output-to-text"
                          for p in _scripted(deny, r["exec_id"]))]
        assert denied, (
            f"{deny.row.row_id}: no `--agent {agent}` turn received the deny "
            f"and continued")

    def test_no_kept_capture_names_an_auto_reject(self, deny):
        """stderr of a successful dispatch is not kept; a failed attempt's is.

        So this is the stderr half of the plan's expectation wherever the
        executor recorded any, and the stub half above covers the rest: an
        auto-reject ends the turn before a second request can exist.
        """
        artifacts = deny.out_dir / "artifacts"
        offenders = []
        for path in sorted(artifacts.rglob("failed/*.json")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if AUTO_REJECT_LINE in text or REJECT_TEXT in text:
                offenders.append(str(path.relative_to(artifacts)))
        assert not offenders, (
            f"{deny.row.row_id}: attempt captures carry an auto-reject: "
            f"{offenders}")
