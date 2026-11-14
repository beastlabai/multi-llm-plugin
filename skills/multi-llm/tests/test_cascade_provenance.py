"""``utils/cascade_provenance``: the stage manifest and the review round's ``cascade`` block."""

from __future__ import annotations

import json

import pytest

from utils import cascade_provenance as cp
from utils import status_file


def _inputs(tmp_path, ledger="## Earlier review rounds\n\nR1-1 …\n"):
    inputs = tmp_path / "stage-2" / "inputs"
    inputs.mkdir(parents=True)
    # Bytes, as the cascade writes them: text mode would write CRLF on Windows.
    (inputs / cp.REVIEWER_LEDGER).write_bytes(ledger.encode("utf-8"))
    (inputs / cp.VALIDATOR_LEDGER).write_bytes(b"validator view\n")
    (inputs / cp.JUDGE_LEDGER).write_bytes(b"judge view\n")
    (inputs / cp.FULL_LEDGER).write_bytes(b"everything\n")
    return inputs


def _manifest(inputs, **over):
    kwargs = dict(cascade_id="abc123def456", mode="improve-plan", stage=2, of=3,
                  model="codex:gpt-x", names=cp.LEDGER_NAMES)
    kwargs.update(over)
    return cp.write_manifest(inputs, **kwargs)


def test_manifest_round_trip_and_block(tmp_path):
    inputs = _inputs(tmp_path)
    path = _manifest(inputs)
    assert path == inputs / cp.MANIFEST_NAME
    doc = cp.load_manifest(path)
    assert doc["stage"] == 2 and doc["of"] == 3
    block = cp.block_from_manifest(path)
    assert block["manifest"] == str(path)
    assert set(block["files"]) == set(cp.LEDGER_NAMES)
    assert cp.check_block(block, "x") is block
    assert cp.ledger_text(block).startswith("## Earlier review rounds")
    assert cp.ledger_text(block, cp.JUDGE_LEDGER) == "judge view\n"
    # An input the stage does not have (improve-code: no plan files) is "".
    assert cp.ledger_text(block, cp.PLAN_BEFORE) == ""
    assert cp.ledger_text(None) == ""


def test_changed_input_fails_closed(tmp_path):
    inputs = _inputs(tmp_path)
    path = _manifest(inputs)
    block = cp.block_from_manifest(path)
    (inputs / cp.REVIEWER_LEDGER).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(cp.CascadeProvenanceError) as caught:
        cp.ledger_text(block)
    assert caught.value.path.endswith(cp.REVIEWER_LEDGER)
    with pytest.raises(cp.CascadeProvenanceError):
        cp.load_manifest(path)
    assert "ERROR: cascade stage input" in cp.error_line(caught.value)


def test_missing_and_malformed_manifest(tmp_path):
    with pytest.raises(cp.CascadeProvenanceError, match="does not exist"):
        cp.load_manifest(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(cp.CascadeProvenanceError, match="cascade_id"):
        cp.load_manifest(bad)


def test_unknown_input_name_refused(tmp_path):
    inputs = _inputs(tmp_path)
    (inputs / "other.md").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        _manifest(inputs, names=("other.md",))


def test_read_block_from_status(tmp_path):
    inputs = _inputs(tmp_path)
    block = cp.block_from_manifest(_manifest(inputs))
    review = tmp_path / "review-plan"
    review.mkdir()
    assert cp.read_block(review) is None
    status_file.write_status(str(review), {"round_id": "r1", cp.STATUS_KEY: block})
    assert cp.read_block(review) == block
    status_file.write_status(str(review), {"round_id": "r1", cp.STATUS_KEY: {"stage": 1}})
    with pytest.raises(cp.CascadeProvenanceError, match="lacks"):
        cp.read_block(review)


def test_next_block_rules(tmp_path):
    inputs = _inputs(tmp_path)
    one = cp.block_from_manifest(_manifest(inputs))
    two = dict(one, stage=3)
    stage_model = [one["model"]]
    # Flag given: recorded, whatever the models.
    assert cp.next_block(None, one, rerun_all=False, models=()) is one
    assert cp.next_block(two, one, rerun_all=True, models=()) is one
    # Same stage, results kept: fine.
    assert cp.next_block(one, dict(one), rerun_all=False, models=stage_model) == one
    # Another stage's results would be kept: refused.
    with pytest.raises(cp.CascadeProvenanceError, match="would keep its results"):
        cp.next_block(two, one, rerun_all=False, models=stage_model)
    # Flag absent: carried over only when the fan-out resumes the stage's
    # round with exactly its one model; cleared on --rerun-all.
    assert cp.next_block(one, None, rerun_all=False, models=stage_model) is one
    assert cp.next_block(one, None, rerun_all=True, models=stage_model) is None
    assert cp.same_stage(one, dict(one)) and not cp.same_stage(one, two)


@pytest.mark.parametrize("models", [
    ["codex:other"],                      # another single model
    ["codex:gpt-x", "codex:other"],       # the stage's model and another
    ["codex:other", "codex:gpt-x"],
    [],
])
def test_next_block_without_the_flag_drops_a_round_started_by_hand(tmp_path, models):
    """A round with any model list but the stage's one model is not the
    stage's round, so it records no block."""
    one = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    assert one["model"] == "codex:gpt-x"
    assert cp.next_block(one, None, rerun_all=False, models=models) is None
    assert not cp.resumes_stage(one, models, rerun_all=False)


def test_resumes_stage(tmp_path):
    one = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    assert cp.resumes_stage(one, ("codex:gpt-x",), rerun_all=False)
    assert not cp.resumes_stage(one, ["codex:gpt-x"], rerun_all=True)
    assert not cp.resumes_stage(None, ["codex:gpt-x"], rerun_all=False)


@pytest.mark.parametrize("flags, expected", [
    ([], ["--models", "codex:gpt-x"]),
    (["--yes"], ["--yes", "--models", "codex:gpt-x"]),
    (["--models", "codex:gpt-x"], ["--models", "codex:gpt-x"]),
    (["--models", "a", "b", "--timeout", "9"], ["--timeout", "9", "--models", "codex:gpt-x"]),
    (["--models=a", "--quick", "--host", "codex"], ["--host", "codex", "--models", "codex:gpt-x"]),
])
def test_retry_flags_name_the_stage_model_when_a_block_is_on_record(tmp_path, flags, expected):
    """A printed ``--force`` retry of a stage's round names its one model in
    place of any model selector, so its fan-out resumes the round and the
    block carries over (``resumes_stage``)."""
    block = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    review = tmp_path / "review-plan"
    review.mkdir()
    status_file.write_status(str(review), {"round_id": "r1", cp.STATUS_KEY: block})
    assert cp.retry_flags(review, flags) == expected
    assert cp.resumes_stage(block, expected[-1:], rerun_all=False)


def test_retry_flags_without_a_block_are_unchanged(tmp_path):
    review = tmp_path / "review-plan"
    flags = ["--models", "a", "b", "--quick"]
    assert cp.retry_flags(review, flags) == flags  # no checkpoint at all
    review.mkdir()
    status_file.write_status(str(review), {"round_id": "r1"})
    assert cp.retry_flags(review, flags) == flags
    # A malformed block is the run's to refuse; the printed line stays as it was.
    status_file.write_status(str(review), {"round_id": "r1", cp.STATUS_KEY: {"stage": 1}})
    assert cp.retry_flags(review, flags) == flags


def test_retry_flags_read_the_given_manifest_before_the_round_records_one(tmp_path):
    manifest = _manifest(_inputs(tmp_path))
    review = tmp_path / "review-plan"
    assert cp.retry_flags(review, ["--quick"], stage_file=manifest) == [
        "--models", "codex:gpt-x"]
    assert cp.retry_flags(review, ["--quick"], stage_file=tmp_path / "missing.json") == [
        "--quick"]


def _current(tmp_path, *, pointer=None, record=None):
    """A plan output dir whose improve-plan run dir holds *pointer* and *record*
    (``None``: the well-formed running ones for cascade ``abc123def456``)."""
    out = tmp_path / "plan"
    run_dir = out / "improve-plan"
    cascade = run_dir / "cascades" / "20261114T000000-abc123de"
    cascade.mkdir(parents=True)
    pointer = pointer if pointer is not None else {
        "schema_version": 1, "cascade_id": "abc123def456",
        "dir": "cascades/20261114T000000-abc123de"}
    record = record if record is not None else {
        "cascade_id": "abc123def456", "state": "running",
        "stages": [{"index": 1, "state": "done"}, {"index": 2, "state": "running"},
                   {"index": 3, "state": "pending"}]}
    for path, doc in ((run_dir / cp.POINTER_NAME, pointer), (cascade / cp.RECORD_NAME, record)):
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return out / "review-plan"


def test_check_current_takes_the_running_stage(tmp_path):
    block = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    cp.check_current(block, _current(tmp_path))


@pytest.mark.parametrize("pointer, record, detail", [
    ("{", None, "cannot be read as JSON"),
    ([], None, "is not a JSON object"),
    ({"cascade_id": "fedcba987654", "dir": "cascades/x"}, None, "superseded"),
    ({"cascade_id": "abc123def456", "dir": "../elsewhere"}, None, "not a path inside"),
    ({"cascade_id": "abc123def456", "dir": "/abs"}, None, "not a path inside"),
    (None, {"cascade_id": "other", "state": "running"}, "not the one the pointer names"),
    (None, {"cascade_id": "abc123def456", "state": "abandoned"}, "is abandoned, not running"),
    (None, {"cascade_id": "abc123def456", "state": "running",
            "stages": [{}, {"state": "skipped"}]}, "stage 2 of cascade abc123de is already"),
])
def test_check_current_refuses_a_stale_handoff(tmp_path, pointer, record, detail):
    block = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    with pytest.raises(cp.StaleHandoff) as caught:
        cp.check_current(block, _current(tmp_path, pointer=pointer, record=record))
    assert detail in caught.value.detail
    assert cp.error_line(caught.value).startswith("ERROR: cascade stage handoff refused: ")


def test_check_current_with_no_pointer_refuses(tmp_path):
    block = cp.block_from_manifest(_manifest(_inputs(tmp_path)))
    with pytest.raises(cp.StaleHandoff) as caught:
        cp.check_current(block, tmp_path / "plan" / "review-plan")
    assert caught.value.path.endswith(cp.POINTER_NAME)
    assert "does not exist" in caught.value.detail
