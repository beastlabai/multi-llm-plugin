"""The one-shot retry, and the one shape it is allowed to fire on.

The container host lane re-runs exactly one failure: a credentialed host row
whose turn **never started** — the host printed a clean `subtype: "success"` /
`is_error: false` / `stop_reason: "end_turn"` envelope with an EMPTY `result`,
never invoked our dispatcher, exported nothing and spent a cent or less. Such a
row observed nothing about the product, so there is nothing in it to grade and
re-running it is nearly free.

The shape it must NOT fire on was measured in the same week: a row that does
most of the work, returns from a backgrounded native `Task` batch and ends its
turn early, having already spent $0.10-$0.48 and left a half-written workspace.
Auto-retrying that would double the tier's cost and paper over a real product
bug.

Two obligations, and this module exists for both:

* **the predicate separates them**, on the real recorded evidence rather than on
  a story about it — :data:`MEASURED` is transcribed from the bundles named in
  each row, and `TestTheSweepOverTheRealBundles` re-reads those bundles
  themselves whenever the developer still has them;
* **a retry is never silent**. Both attempts land in the bundle, both land in
  the record, and the lane's own census says a row was re-run. A rig that
  retried quietly would be lying about its own reliability, which is the one
  thing this suite exists not to do.

**The filename is deliberately not `test_container_*`** — the same reason
`test_capability_promotions.py` gives. That glob is what the gating check
enumerates as container-marked, and a plain-CI module answering to it would be
deselected everywhere it matters: this needs no credential, no Docker daemon and
no run bundle on disk, and it is the guard for a mechanism that SPENDS MONEY on
its own initiative. It has to run on the PR that widens it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .container import harness

#: A stubbed image block, so no test here shells out to a Docker daemon.
IMAGE = {"tag": "mllm-e2e:local", "digest": "sha256:" + "cd" * 32,
         "platform": "linux/amd64", "pin_mode": "pinned"}

#: Real host-lane row ids, so the records these tests render go through the same
#: `host_matrix.yaml` lookups (`declared_command`, the model pin) a live row does.
ROW_A = "claude-code-review-plan"
ROW_B = "claude-code-ask"

#: The rig's own staging invocation, present on EVERY row and countable on none:
#: `install-host --check` is the entrypoint putting the shim in place, not the
#: host running our dispatcher.
STAGING_ARGV = ["run", "--project", "/opt/mllm/skill", "--", "python",
                "/opt/mllm/skill/multi_llm.py", "install-host", "claude-code",
                "--check", "--project-root", "/work/sample-project"]

#: A real dispatcher invocation — what a receipt has to look like to count.
DISPATCH_ARGV = ["run", "--project", "/opt/mllm/skill", "--", "python",
                 "/opt/mllm/skill/multi_llm.py", "ask", "plans/auth-feature.md"]

# --------------------------------------------------------------------------- #
# WHAT WAS MEASURED. One entry per run bundle recorded by
# `skills/multi-llm/tests/container/runs/` across seven live runs, transcribed
# here so the predicate is pinned in plain CI — `runs/` is gitignored, prunable
# and erased by `make e2e-clean`, so a guard that could only read it is a guard
# that stops running the day someone cleans up.
#
# `shape` is the classification a human made from the bundles, and it is the
# expectation: the detector must fire on `never-started` and on nothing else.
# `TestTheSweepOverTheRealBundles` re-derives all of it from the bundles
# themselves when they are still on disk, which is what keeps this table from
# drifting into fiction.
# --------------------------------------------------------------------------- #
MEASURED = [
    dict(run='20260921T224742Z', row='chain-claude-code',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224742Z', row='chain-writer-claude-code',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-aider',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-cline',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-codex',
         lane='provider', host=None, shape='other', exit_code=3, timed_out=False,
         receipts=1, artifact_files=29, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-goose',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-writer-claude-code',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-writer-codex',
         lane='provider', host=None, shape='other', exit_code=3, timed_out=False,
         receipts=1, artifact_files=29, spend_usd=None,
         envelope=None),
    dict(run='20260921T224840Z', row='chain-writer-opencode',
         lane='provider', host=None, shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=138, spend_usd=None,
         envelope=None),
    dict(run='20260922T084502Z', row='claude-code-apply-native',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=16, artifact_files=20, spend_usd=1.079751,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=1379,
             num_turns=13, output_tokens=4061)),
    dict(run='20260922T084502Z', row='claude-code-ask',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.01211,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=1570,
             num_turns=21, output_tokens=7791)),
    dict(run='20260922T084502Z', row='claude-code-review-plan',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=8, artifact_files=29, spend_usd=0.045408,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=2395,
             num_turns=8, output_tokens=3173)),
    dict(run='20260922T084502Z', row='codex-apply-executor',
         lane='host', host='codex', shape='other', exit_code=3, timed_out=False,
         receipts=0, artifact_files=25, spend_usd=None,
         envelope=None),
    dict(run='20260922T084502Z', row='codex-ask',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=15, spend_usd=0.010383,
         envelope=None),
    dict(run='20260922T084502Z', row='codex-auth-negative',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=31, spend_usd=0.057366,
         envelope=None),
    dict(run='20260922T084502Z', row='codex-full',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=30, spend_usd=0.010282,
         envelope=None),
    dict(run='20260922T084502Z', row='codex-review-plan',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=6, artifact_files=27, spend_usd=0.01558,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-apply-executor',
         lane='host', host='opencode', shape='other', exit_code=3, timed_out=False,
         receipts=0, artifact_files=25, spend_usd=None,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.005546,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-auth-negative',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=31, spend_usd=0.00503,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=30, spend_usd=0.006893,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-package-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=15, spend_usd=0.001838,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-package-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=30, spend_usd=0.001683,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-package-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=27, spend_usd=0.001303,
         envelope=None),
    dict(run='20260922T084502Z', row='opencode-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=6, artifact_files=27, spend_usd=0.006637,
         envelope=None),
    dict(run='20260922T124956Z', row='claude-code-apply-native',
         lane='host', host='claude-code', shape='half-finished', exit_code=0, timed_out=False,
         receipts=2, artifact_files=18, spend_usd=0.479441,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=894,
             num_turns=5, output_tokens=1203)),
    dict(run='20260922T124956Z', row='claude-code-ask',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.024932,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=1728,
             num_turns=16, output_tokens=8948)),
    dict(run='20260922T124956Z', row='claude-code-review-plan',
         lane='host', host='claude-code', shape='half-finished', exit_code=0, timed_out=False,
         receipts=6, artifact_files=23, spend_usd=0.096642,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=53,
             num_turns=2, output_tokens=250)),
    dict(run='20260922T124956Z', row='codex-apply-executor',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=52, spend_usd=0.009834,
         envelope=None),
    dict(run='20260922T124956Z', row='codex-ask',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.002902,
         envelope=None),
    dict(run='20260922T124956Z', row='codex-auth-negative',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=31, spend_usd=0.064575,
         envelope=None),
    dict(run='20260922T124956Z', row='codex-full',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=167, spend_usd=0.032672,
         envelope=None),
    dict(run='20260922T124956Z', row='codex-review-plan',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=30, spend_usd=0.018709,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-apply-executor',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=50, spend_usd=0.006989,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=13, spend_usd=0.002086,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-auth-negative',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=31, spend_usd=0.006149,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=11, artifact_files=169, spend_usd=0.026206,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-package-apply-executor',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=50, spend_usd=0.000251,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-package-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=13, spend_usd=0.0,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-package-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=170, spend_usd=0.001202,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-package-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=27, spend_usd=0.001393,
         envelope=None),
    dict(run='20260922T124956Z', row='opencode-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=26, spend_usd=0.019093,
         envelope=None),
    dict(run='20260922T143440Z', row='claude-code-apply-native',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=17, artifact_files=20, spend_usd=1.700198,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=870,
             num_turns=11, output_tokens=3469)),
    dict(run='20260922T143440Z', row='claude-code-review-plan',
         lane='host', host='claude-code', shape='never-started', exit_code=0, timed_out=False,
         receipts=0, artifact_files=0, spend_usd=0.000711,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=0,
             num_turns=4, output_tokens=95)),
    dict(run='20260922T154654Z', row='claude-code-apply-native',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=16, artifact_files=20, spend_usd=1.274321,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=847,
             num_turns=12, output_tokens=3821)),
    dict(run='20260922T154654Z', row='claude-code-ask',
         lane='host', host='claude-code', shape='never-started', exit_code=0, timed_out=False,
         receipts=0, artifact_files=0, spend_usd=0.0,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=0,
             num_turns=4, output_tokens=165)),
    dict(run='20260922T154654Z', row='claude-code-review-plan',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=7, artifact_files=29, spend_usd=0.102181,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=2235,
             num_turns=6, output_tokens=1641)),
    dict(run='20260922T154654Z', row='codex-apply-executor',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=6, artifact_files=51, spend_usd=0.016202,
         envelope=None),
    dict(run='20260922T154654Z', row='codex-ask',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.012848,
         envelope=None),
    dict(run='20260922T154654Z', row='codex-auth-negative',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=32, spend_usd=0.108783,
         envelope=None),
    dict(run='20260922T154654Z', row='codex-full',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=168, spend_usd=0.083488,
         envelope=None),
    dict(run='20260922T154654Z', row='codex-review-plan',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=29, spend_usd=0.026481,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-apply-executor',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=51, spend_usd=0.005174,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=13, spend_usd=0.008865,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-auth-negative',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=32, spend_usd=0.004891,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=169, spend_usd=0.013589,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-package-apply-executor',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=51, spend_usd=0.000268,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-package-ask',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=13, spend_usd=0.0,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-package-full',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=171, spend_usd=0.001512,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-package-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=3, artifact_files=27, spend_usd=0.001677,
         envelope=None),
    dict(run='20260922T154654Z', row='opencode-review-plan',
         lane='host', host='opencode', shape='other', exit_code=0, timed_out=False,
         receipts=5, artifact_files=27, spend_usd=0.021464,
         envelope=None),
    dict(run='20260922T172155Z', row='claude-code-ask',
         lane='host', host='claude-code', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=15, spend_usd=0.017045,
         envelope=dict(
             subtype='success', is_error=False, stop_reason='end_turn', result_chars=2067,
             num_turns=28, output_tokens=13279)),
    dict(run='20260922T172155Z', row='codex-apply-executor',
         lane='host', host='codex', shape='other', exit_code=0, timed_out=False,
         receipts=4, artifact_files=53, spend_usd=0.008178,
         envelope=None),
]

#: The two rows the retry exists for, as `(run, row)`.
NEVER_STARTED = [(r["run"], r["row"]) for r in MEASURED
                 if r["shape"] == "never-started"]
#: The two it must never fire on, however much they look like a failure.
HALF_FINISHED = [(r["run"], r["row"]) for r in MEASURED
                 if r["shape"] == "half-finished"]
#: Every other recorded row — greens, refusals, negatives, provider-lane chains.
OTHER = [(r["run"], r["row"]) for r in MEASURED if r["shape"] == "other"]

RUNS_IN_TABLE = sorted({r["run"] for r in MEASURED})


def _facts(run: str, row: str) -> dict:
    for entry in MEASURED:
        if entry["run"] == run and entry["row"] == row:
            return entry
    raise AssertionError(f"{run}/{row} is not in MEASURED")


def _materialise(out_dir: Path, facts: dict, **override) -> Path:
    """Write the bundle files the predicate reads, and only those.

    Deliberately minimal: if the detector ever starts depending on a file this
    does not write, the tests here go red rather than quietly grading a richer
    bundle than the one whose facts were measured.
    """
    facts = {**facts, **override}
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope = facts["envelope"]
    if envelope is not None:
        (out_dir / "host-stdout.txt").write_text(json.dumps({
            "type": envelope.get("type", "result"),
            "subtype": envelope["subtype"],
            "is_error": envelope["is_error"],
            "stop_reason": envelope["stop_reason"],
            "num_turns": envelope["num_turns"],
            "result": "x" * envelope["result_chars"],
            "usage": {"output_tokens": envelope["output_tokens"]},
        }) + "\n", encoding="utf-8")
    receipts = out_dir / "receipts" / facts["row"]
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "001.receipt.json").write_text(
        json.dumps({"argv": STAGING_ARGV, "exit": 0}) + "\n", encoding="utf-8")
    for index in range(int(facts["receipts"])):
        (receipts / f"{index + 2:03d}.receipt.json").write_text(
            json.dumps({"argv": DISPATCH_ARGV, "exit": 0}) + "\n",
            encoding="utf-8")
    artifacts = out_dir / "artifacts" / "auth-feature"
    if facts["artifact_files"]:
        artifacts.mkdir(parents=True, exist_ok=True)
    for index in range(int(facts["artifact_files"])):
        (artifacts / f"file-{index:03d}.json").write_text("{}\n", encoding="utf-8")
    if facts["spend_usd"] is not None:
        (out_dir / "usage.json").write_text(json.dumps({
            "row": facts["row"], "delta_usd": facts["spend_usd"]}) + "\n",
            encoding="utf-8")
    (out_dir / "row-result.json").write_text(json.dumps({
        "row_id": facts["row"], "lane": facts["lane"], "host": facts["host"],
        "exit_code": facts["exit_code"], "timed_out": facts["timed_out"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir


def _row(facts: dict, **override) -> harness.Row:
    facts = {**facts, **override}
    return harness.Row(
        row_id=facts["row"], lane=facts["lane"], host=facts["host"],
        creds="openrouter" if facts["lane"] == "host" else "none",
        dry_run_host=bool(facts.get("dry_run_host")),
        documented_invocation=bool(facts.get("documented_invocation")),
    )


def _result(tmp_path: Path, facts: dict, **override) -> harness.RowResult:
    """A `RowResult` over a bundle carrying exactly the measured facts."""
    merged = {**facts, **override}
    out_dir = _materialise(tmp_path / merged["row"], merged)
    return harness.RowResult(
        row=_row(merged), exit_code=merged["exit_code"], wall_clock_s=43.9,
        out_dir=out_dir, timed_out=merged["timed_out"])


class TestTheDetectorFiresOnTheOneShapeItIsFor:
    """Both recorded never-started rows, clause by clause."""

    @pytest.mark.parametrize("run,row", NEVER_STARTED,
                             ids=[f"{r}-{n}" for r, n in NEVER_STARTED])
    def test_it_fires_on_every_recorded_never_started_row(self, run, row,
                                                          tmp_path):
        signature = harness.never_started_signature(
            _result(tmp_path, _facts(run, row)))
        assert signature is not None, (
            f"{run}/{row} is the shape the retry exists for and the detector "
            f"did not recognise it")
        assert signature["shape"] == "never-started"
        assert signature["dispatcher_receipts"] == 0
        assert signature["artifact_files"] == 0
        assert signature["envelope"]["result_chars"] == 0
        assert signature["spend_usd"] <= harness.NEVER_STARTED_MAX_USD

    @pytest.mark.parametrize("run,row", NEVER_STARTED,
                             ids=[f"{r}-{n}" for r, n in NEVER_STARTED])
    def test_the_reason_quotes_what_it_measured(self, run, row, tmp_path):
        """A retry nobody can audit later is a retry that should not happen."""
        signature = harness.never_started_signature(
            _result(tmp_path, _facts(run, row)))
        reason = harness.never_started_reason(signature)
        assert "never-started" in reason
        assert "end_turn" in reason and "EMPTY result" in reason
        assert str(signature["spend_usd"]) in reason


class TestTheDetectorRefusesTheShapeThatCostMoney:
    """The half-finished rows: already $0.10-$0.48 in, and a product bug."""

    @pytest.mark.parametrize("run,row", HALF_FINISHED,
                             ids=[f"{r}-{n}" for r, n in HALF_FINISHED])
    def test_it_does_not_fire_on_a_half_finished_row(self, run, row, tmp_path):
        facts = _facts(run, row)
        assert harness.never_started_signature(
            _result(tmp_path, facts)) is None, (
            f"{run}/{row} had already spent ${facts['spend_usd']} and left "
            f"{facts['artifact_files']} artifacts behind it. Retrying it "
            f"doubles the tier's cost and papers over a real product bug")

    @pytest.mark.parametrize("run,row", HALF_FINISHED,
                             ids=[f"{r}-{n}" for r, n in HALF_FINISHED])
    def test_the_receipts_are_what_separate_it(self, run, row, tmp_path):
        """Zero-out its receipts and artifacts and it still does not fire.

        The spend clause is the independent second lock: the half-finished rows
        are 10x and 48x over the ceiling, so even a bundle whose receipts went
        missing cannot buy a re-run of a row that spent real money.
        """
        facts = _facts(run, row)
        assert harness.never_started_signature(_result(
            tmp_path, facts, receipts=0, artifact_files=0)) is None

    @pytest.mark.parametrize("run,row", OTHER,
                             ids=[f"{r}-{n}" for r, n in OTHER])
    def test_it_does_not_fire_on_any_other_recorded_row(self, run, row,
                                                        tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, _facts(run, row))) is None


class TestEveryClauseIsLoadBearing:
    """Flip one measured fact at a time; the detector must go quiet each time.

    A predicate whose clauses are not individually necessary is a predicate that
    fires on something nobody measured.
    """

    @pytest.fixture
    def facts(self):
        return _facts(*NEVER_STARTED[0])

    def test_a_non_zero_exit_is_someone_elses_diagnosis(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, exit_code=3)) is None

    def test_a_capped_row_is_a_cap_hit(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, timed_out=True, exit_code=None)) is None

    def test_a_reported_error_is_not_this_shape(self, facts, tmp_path):
        envelope = {**facts["envelope"], "is_error": True}
        assert harness.never_started_signature(
            _result(tmp_path, facts, envelope=envelope)) is None

    def test_a_max_turns_envelope_is_not_this_shape(self, facts, tmp_path):
        envelope = {**facts["envelope"], "subtype": "error_max_turns"}
        assert harness.never_started_signature(
            _result(tmp_path, facts, envelope=envelope)) is None

    def test_an_interrupted_turn_is_not_this_shape(self, facts, tmp_path):
        envelope = {**facts["envelope"], "stop_reason": "max_tokens"}
        assert harness.never_started_signature(
            _result(tmp_path, facts, envelope=envelope)) is None

    def test_one_character_of_answer_is_an_answer(self, facts, tmp_path):
        envelope = {**facts["envelope"], "result_chars": 1}
        assert harness.never_started_signature(
            _result(tmp_path, facts, envelope=envelope)) is None

    def test_one_dispatcher_receipt_is_a_row_that_started(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, receipts=1)) is None

    def test_the_rigs_own_staging_receipt_still_does_not_count(self, facts,
                                                               tmp_path):
        """Every bundle `_materialise` writes carries `install-host --check`.

        If it ever started counting, this whole mechanism would stop firing —
        and the lane would look reliable because it had gone blind.
        """
        result = _result(tmp_path, facts)
        assert len(result.receipts) == 1
        assert harness.dispatcher_receipts(result) == []
        assert harness.never_started_signature(result) is not None

    def test_one_exported_artifact_is_work_done(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, artifact_files=1)) is None

    def test_a_row_that_spent_more_than_the_ceiling(self, facts, tmp_path):
        over = harness.NEVER_STARTED_MAX_USD + 0.000001
        assert harness.never_started_signature(
            _result(tmp_path, facts, spend_usd=over)) is None

    def test_the_ceiling_itself_is_inside(self, facts, tmp_path):
        assert harness.never_started_signature(_result(
            tmp_path, facts, spend_usd=harness.NEVER_STARTED_MAX_USD)) is not None

    def test_an_unmeasured_spend_is_never_a_small_one(self, facts, tmp_path):
        """No usage readback, no retry. A bill this rig cannot bound, it does not sign."""
        assert harness.never_started_signature(
            _result(tmp_path, facts, spend_usd=None)) is None

    def test_a_host_that_prints_no_result_envelope(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, envelope=None)) is None

    def test_a_jsonl_host_stream_with_no_result_line(self, facts, tmp_path):
        """codex and opencode write events, not a verdict — and events are not one."""
        result = _result(tmp_path, facts, envelope=None)
        (result.out_dir / "host-stdout.txt").write_text("\n".join(
            json.dumps({"type": "item.completed", "subtype": "success",
                        "is_error": False, "stop_reason": "end_turn",
                        "result": ""}) for _ in range(3)) + "\n",
            encoding="utf-8")
        assert harness.never_started_signature(result) is None

    def test_the_provider_lane_is_not_this_lanes_business(self, facts, tmp_path):
        assert harness.never_started_signature(
            _result(tmp_path, facts, lane="provider")) is None

    @pytest.mark.parametrize("tier", ["dry_run_host", "documented_invocation"])
    def test_the_credential_free_tiers_are_graded_on_other_evidence(
            self, facts, tmp_path, tier):
        assert harness.never_started_signature(
            _result(tmp_path, facts, **{tier: True})) is None


class TestTheSweepOverTheRealBundles:
    """The same question asked of the bytes, whenever they are still on disk.

    `runs/` is gitignored and prunable, so this skips by name rather than
    failing when the bundles are gone — but where they exist it is the only
    check here that cannot be fooled by a transcription error in `MEASURED`.
    """

    @staticmethod
    def _bundles():
        root = harness.RUNS
        if not root.is_dir():
            return []
        found = []
        for run in sorted(root.iterdir()):
            if run.name not in RUNS_IN_TABLE:
                # A run recorded after this table was written. Reading it would
                # make the assertion below depend on evidence nobody classified.
                continue
            for bundle in sorted(run.iterdir()):
                if not bundle.is_dir():
                    continue
                marker = bundle / "row-result.json"
                if not marker.is_file():
                    continue
                try:
                    record = json.loads(marker.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                found.append((run.name, bundle, record))
        return found

    @staticmethod
    def _result_for(bundle: Path, record: dict) -> harness.RowResult:
        return harness.RowResult(
            row=harness.Row(row_id=record.get("row_id") or bundle.name,
                            lane=record.get("lane") or "host",
                            host=record.get("host"),
                            cli=record.get("cli"),
                            creds="openrouter"),
            exit_code=record.get("exit_code"),
            wall_clock_s=float(record.get("wall_clock_s") or 0.0),
            out_dir=bundle, timed_out=bool(record.get("timed_out")))

    def test_it_fires_on_exactly_the_rows_it_was_written_for(self):
        bundles = self._bundles()
        if not bundles:
            pytest.skip("no classified run bundles on this machine — `runs/` is "
                        "gitignored and prunable, so the transcribed MEASURED "
                        "table above is the pin here")
        fired = {(run, bundle.name) for run, bundle, record in bundles
                 if harness.never_started_signature(
                     self._result_for(bundle, record)) is not None}
        expected = {key for key in NEVER_STARTED
                    if any(key == (run, bundle.name)
                           for run, bundle, _ in bundles)}
        assert fired == expected, (
            f"the detector fired on {sorted(fired - expected)} and missed "
            f"{sorted(expected - fired)} across {len(bundles)} recorded "
            f"bundles")

    def test_the_table_still_describes_the_bundles(self):
        """MEASURED is a transcription, and a transcription can rot."""
        bundles = self._bundles()
        if not bundles:
            pytest.skip("no classified run bundles on this machine")
        wrong = []
        for run, bundle, record in bundles:
            try:
                facts = _facts(run, bundle.name)
            except AssertionError:
                continue
            result = self._result_for(bundle, record)
            observed = (len(harness.dispatcher_receipts(result)),
                        len(harness.artifact_files(result)),
                        result.usage.get("delta_usd"))
            claimed = (facts["receipts"], facts["artifact_files"],
                       facts["spend_usd"])
            if observed != claimed:
                wrong.append(f"{run}/{bundle.name}: table says {claimed}, "
                             f"bundle says {observed}")
        assert not wrong, "MEASURED disagrees with the bundles:\n" + "\n".join(wrong)


# --------------------------------------------------------------------------- #
# The other half: a retry that HAPPENED, and what it is obliged to leave behind.
#
# `_run_attempt` is the one function in `run_row` that needs a Docker daemon, so
# it is the one function these tests replace. Everything above and around it —
# the detection, the shelving, the manifest, the two census entries, the record
# with two rows in it — is the real code path, driven end to end.
# --------------------------------------------------------------------------- #

UTC = "20260101T000000Z"

#: Provenance stubbed all-`documented`, so "does a retried row still promote?"
#: has a deterministic answer here instead of tracking today's capabilities file.
PROVENANCE = {"shim": "documented", "install_paths": "documented",
              "arguments": "documented", "reload": "documented",
              "multi_turn": "documented"}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """The harness with every daemon- and tracked-file write redirected."""
    monkeypatch.setattr(harness, "image_evidence",
                        lambda image=harness.DEFAULT_IMAGE: dict(IMAGE))
    monkeypatch.setattr(harness, "cli_version",
                        lambda cli, image=harness.DEFAULT_IMAGE: None)
    monkeypatch.setattr(harness, "load_capabilities",
                        lambda: {"claude-code": {"provenance": dict(PROVENANCE)}})
    # The records directory is TRACKED. A test that wrote into it would commit
    # its own fixtures by accident.
    monkeypatch.setattr(harness, "RECORDS", tmp_path / "records")
    monkeypatch.delenv("MLLM_E2E_RETRY_NEVER_STARTED", raising=False)
    monkeypatch.setenv("MLLM_E2E_MATRIX_TIMEOUT_S", "0")     # uncapped
    harness.MATRIX.reset()
    harness.GRADES.reset()
    yield tmp_path / "runs"
    harness.MATRIX.reset()
    harness.GRADES.reset()


class _Driver:
    """A `_run_attempt` that materialises a recorded shape instead of running Docker."""

    def __init__(self, monkeypatch, *shapes, expire_after: int | None = None):
        self.shapes = [dict(shape) for shape in shapes]
        self.out_dirs: list[Path] = []
        self.expire_after = expire_after
        monkeypatch.setattr(harness, "_run_attempt", self)

    def __call__(self, row, posture, image, out_dir, runs_root, run_root):
        facts = self.shapes[min(len(self.out_dirs), len(self.shapes) - 1)]
        self.out_dirs.append(Path(out_dir))
        _materialise(Path(out_dir), {**facts, "row": row.row_id})
        result = harness.RowResult(
            row=row, exit_code=facts["exit_code"],
            wall_clock_s=40.0 + len(self.out_dirs), out_dir=Path(out_dir),
            timed_out=facts["timed_out"])
        if self.expire_after == len(self.out_dirs):
            # The lane's clock, spent by the attempt that just ran — which is
            # how it really expires.
            harness.MATRIX.started_at -= 10_000
        return result

    @property
    def attempts(self) -> int:
        return len(self.out_dirs)


def _drive(rig, driver_row: str = ROW_A) -> harness.RowResult:
    row = harness.Row(row_id=driver_row, lane="host", host="claude-code",
                      creds="openrouter", row_timeout_s=900)
    return harness.run_row(row, posture=harness.load_posture(),
                           runs_root=rig, utc=UTC)


def _manifest(result) -> dict:
    path = result.out_dir / harness.RETRY_MANIFEST_NAME
    assert path.is_file(), f"no {harness.RETRY_MANIFEST_NAME} beside {result.out_dir}"
    return json.loads(path.read_text(encoding="utf-8"))


def _record(result) -> dict:
    assert result.record_path is not None, "no record was written"
    return json.loads(Path(result.record_path).read_text(encoding="utf-8"))


class TestARetryIsNeverSilent:
    """Both attempts in the bundle, both in the record, the retry in the census."""

    @pytest.fixture
    def retried(self, rig, monkeypatch):
        driver = _Driver(monkeypatch, _facts(*NEVER_STARTED[0]),
                         _facts("20260922T154654Z", "claude-code-review-plan"))
        result = _drive(rig)
        return driver, result

    def test_the_row_is_run_exactly_twice(self, retried):
        driver, result = retried
        assert driver.attempts == 2
        assert result.attempt == 2
        assert len(result.attempts) == 1

    def test_the_discarded_attempt_keeps_its_whole_bundle(self, retried):
        _driver, result = retried
        shelved = result.out_dir / "attempt-1"
        assert shelved.is_dir()
        assert result.attempts[0].out_dir == shelved
        envelope = harness.host_result_envelope(result.attempts[0])
        assert envelope["result"] == "", (
            "the discarded attempt's own evidence has to still be there — it is "
            "the only thing that says WHY the row was re-run")
        assert harness.never_started_signature(result.attempts[0]) is not None

    def test_the_graded_attempt_is_where_every_grader_already_looks(self, retried):
        """The shelving must not move the bundle a passing row is read from."""
        _driver, result = retried
        assert result.out_dir.name == ROW_A
        assert (result.out_dir / "host-stdout.txt").is_file()
        assert harness.dispatcher_receipts(result), (
            "the graded attempt is the one that did the work")
        assert harness.never_started_signature(result) is None

    def test_each_attempt_says_which_one_it_is(self, retried):
        _driver, result = retried
        first = json.loads((result.out_dir / "attempt-1" / "row-result.json")
                           .read_text(encoding="utf-8"))
        second = json.loads((result.out_dir / "row-result.json")
                            .read_text(encoding="utf-8"))
        assert (first["attempt"], first["graded"]) == (1, False)
        assert (second["attempt"], second["graded"]) == (2, True)

    def test_the_manifest_names_both_attempts_and_the_graded_one(self, retried):
        _driver, result = retried
        manifest = _manifest(result)
        assert manifest["state"] == "retried"
        assert manifest["graded_attempt"] == 2
        assert manifest["max_attempts"] == 2
        assert [entry["attempt"] for entry in manifest["attempts"]] == [1, 2]
        assert [entry["graded"] for entry in manifest["attempts"]] == [False, True]
        assert manifest["attempts"][0]["bundle"] == "attempt-1"
        assert manifest["attempts"][0]["never_started_signature"], (
            "the manifest has to say what the first attempt looked like, or "
            "nobody can audit the retry afterwards")
        assert manifest["attempts"][1]["never_started_signature"] is None

    def test_the_record_carries_both_attempts(self, retried):
        _driver, result = retried
        record = _record(result)
        assert [row["row_id"] for row in record["rows"]] == [ROW_A, ROW_A]
        assert record["rows"][0]["outcome"] == "fail"
        assert "never-started" in record["rows"][0]["reason"]
        assert record["rows"][-1]["outcome"] == "pass"
        assert "attempt 2 of 2" in record["rows"][-1]["reason"]

    def test_the_record_still_validates(self, retried):
        _driver, result = retried
        assert harness.validate_record(_record(result)) == []

    def test_the_digest_covers_the_discarded_attempt(self, retried):
        """A sibling directory would leave attempt 1 rewritable after the fact."""
        _driver, result = retried
        files = _record(result)["bundle"]["files"]
        assert any(name.startswith("attempt-1/") for name in files), (
            f"the bundle digest covers {sorted(files)[:5]}… and not the "
            f"discarded attempt")

    def test_the_lane_census_counts_the_retry_apart(self, retried):
        _driver, result = retried
        census = harness.MATRIX.census()
        assert [entry["row"] for entry in census["retried"]] == \
            [f"{ROW_A}#attempt-1"]
        assert "never-started" in census["retried"][0]["reason"]
        assert [entry["row"] for entry in census["ran"]] == [ROW_A]
        assert census["failed"] == [], (
            "a discarded attempt is not a failed row — it left its question "
            "unanswered, which is a fourth thing")
        assert f"{ROW_A}#attempt-1" in harness.MATRIX.report()


class TestTheRetryIsOneShot:
    """Twice is a finding for a human, not a third container."""

    def test_a_second_never_started_attempt_is_graded_not_retried(
            self, rig, monkeypatch):
        shape = _facts(*NEVER_STARTED[0])
        driver = _Driver(monkeypatch, shape, shape)
        result = _drive(rig)
        assert driver.attempts == 2
        manifest = _manifest(result)
        assert manifest["graded_attempt"] == 2
        assert manifest["attempts"][1]["never_started_signature"], (
            "a row that never started twice running is a broken row, and the "
            "manifest is where that has to read at a glance")
        assert harness.never_started_signature(result) is not None


class TestNothingElseChanges:
    """A row that ran once renders exactly the bundle and record it always did."""

    @pytest.fixture
    def green(self, rig, monkeypatch):
        driver = _Driver(monkeypatch,
                         _facts("20260922T084502Z", "claude-code-ask"))
        return driver, _drive(rig, ROW_B)

    def test_one_attempt_one_bundle(self, green):
        driver, result = green
        assert driver.attempts == 1
        assert result.attempt == 1
        assert result.attempts == []
        assert not (result.out_dir / "attempt-1").exists()
        assert not (result.out_dir / harness.RETRY_MANIFEST_NAME).exists()

    def test_one_row_in_the_record(self, green):
        _driver, result = green
        record = _record(result)
        assert len(record["rows"]) == 1
        assert "attempt" not in (record["rows"][0].get("reason") or "")

    def test_nothing_lands_in_the_retried_bucket(self, green):
        assert harness.MATRIX.census()["retried"] == []


class TestTheRetryCanBeDeclined:
    """Two ways to not re-run a row — and both of them still say so."""

    def test_the_env_switch_disarms_the_retry_not_the_detector(
            self, rig, monkeypatch):
        monkeypatch.setenv("MLLM_E2E_RETRY_NEVER_STARTED", "0")
        driver = _Driver(monkeypatch, _facts(*NEVER_STARTED[0]))
        result = _drive(rig)
        assert driver.attempts == 1
        assert result.attempt == 1
        manifest = _manifest(result)
        assert manifest["state"] == "not-retried"
        assert manifest["graded_attempt"] == 1
        assert "MLLM_E2E_RETRY_NEVER_STARTED" in manifest["declined"]
        assert not (result.out_dir / "attempt-1").exists()

    def test_a_spent_lane_budget_declines_it(self, rig, monkeypatch):
        monkeypatch.setenv("MLLM_E2E_MATRIX_TIMEOUT_S", "600")
        driver = _Driver(monkeypatch, _facts(*NEVER_STARTED[0]),
                         expire_after=1)
        result = _drive(rig)
        assert driver.attempts == 1, (
            "the retry is a row like any other: once the lane's clock is spent "
            "the driver starts no further container")
        manifest = _manifest(result)
        assert manifest["state"] == "not-retried"
        assert "budget" in manifest["declined"]
        assert harness.MATRIX.census()["never_started"] == [], (
            "the row DID start once — filing it as never-started would describe "
            "a bundle that is sitting right there")


class TestPassWithAFlag:
    """What a row that only passed on its retry is reported as, and why.

    The decision this module encodes: **pass, with a flag that is countable.**

    The row's verdict answers a question about the PRODUCT — did the host load
    our shim, find the command, run the workflow — and attempt 2 answered it on
    evidence that is whole. Grading it `fail` would throw away a real answer
    because the first attempt produced none, and would make the honest thing
    (re-running a row that observed nothing) more expensive than the dishonest
    one.

    The flag answers a different question, about the LANE: does this row pass
    when you run it? A retried pass is a false answer to that one, so it is
    marked in every place a reader counts from — the record's rows array, the
    lane census, the matrix census — and the mark is a count rather than a word,
    so a retry rate that starts climbing shows up as a number instead of a
    feeling.
    """

    @pytest.fixture
    def graded(self, rig, monkeypatch):
        _Driver(monkeypatch, _facts(*NEVER_STARTED[0]),
                _facts("20260922T154654Z", "claude-code-review-plan"))
        result = _drive(rig)
        harness.GRADES.note(ROW_A, "tests/test_x.py::test_the_row_did_its_job",
                            True)
        path = harness.GRADES.resolve(ROW_A)
        return result, json.loads(Path(path).read_text(encoding="utf-8"))

    def test_the_run_is_a_pass(self, graded):
        _result, record = graded
        assert record["outcome"] == "pass"
        assert record["rows"][-1]["outcome"] == "pass"

    def test_and_it_says_it_was_retried_where_a_reader_counts(self, graded):
        _result, record = graded
        assert len(record["rows"]) == 2
        assert record["rows"][0]["outcome"] == "fail"
        assert "attempt 2 of 2" in record["rows"][-1]["reason"]
        assert "retry.json" in record["rows"][-1]["reason"]

    def test_the_promotion_still_stands_and_carries_the_flag_with_it(self, graded):
        """The capability was demonstrated — by an attempt whose bundle is whole.

        Withholding the promotion would say the opposite of what happened: the
        never-started attempt is evidence about nothing, and the graded one is
        evidence about the host. What the promotion may not do is arrive without
        the retry attached, and it cannot: the record that carries it is the
        record that carries both rows.
        """
        _result, record = graded
        assert record["promotions"], (
            "a graded, green attempt promotes what it demonstrated")
        assert {edit["evidence"] for edit in record["promotions"]} == {ROW_A}
        assert len(record["rows"]) == 2
