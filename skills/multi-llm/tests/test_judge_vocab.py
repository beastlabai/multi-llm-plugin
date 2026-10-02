"""The judge's persisted names are host-neutral, and the old ones still load (issue 18).

The 2026-10-01 three-host run judged its Codex leg with ``codex:gpt-6-astra``, yet
everything that judge left on disk said Claude: ``decision_source:
claude_auto_decide``, ``status: claude_skipped``, ``claude_decide_item_ids``, a
``claudeDecided`` badge flag. ``utils/judge_vocab.py`` now owns the spelling.

Two halves, both asserted here:

* **Writers use only the new names.** A source guard over every runtime module,
  template and instruction file, plus the writers themselves.
* **Readers accept both.** ``tests/fixtures/legacy_judge_keys/`` holds what an
  earlier version wrote — recorded decisions, an ``orchestrator_output.json``, a
  ``user_selections.json`` and a ``report_data.json`` sidecar. Every test below
  loads the fixture *and* its current-spelling twin and requires the same
  behaviour from both. The twin is made by a plain text substitution written out
  in this file, never by ``judge_vocab`` itself, so the map cannot vouch for
  itself.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import display_decisions  # noqa: E402
from tests.test_apply_orchestrator_base import (  # noqa: E402
    MockApplyOrchestrator,
    _FakeOutcome,
    _FakeSchedulerResult,
    _make_args,
    _setup_plan_with_output,
)
from utils import html_report_generator  # noqa: E402
from utils.filtering import validate_judge_decide_items_honored  # noqa: E402
from utils.human_decisions import (  # noqa: E402
    VerdictIngest,
    judge_outcome,
    judge_source_label,
    judge_status_label,
    record_verdicts,
)
from utils.judge_vocab import (  # noqa: E402
    JUDGE_AUTO_DECIDE,
    JUDGE_AUTO_DECIDE_SALVAGE,
    JUDGE_DECIDE,
    JUDGE_DECIDE_ITEM_IDS,
    JUDGE_DECIDED,
    JUDGE_SKIPPED,
    LEGACY_KEYS,
    LEGACY_VALUES,
    canonical_value,
    upgrade_legacy,
)
from utils.report_parser import (  # noqa: E402
    load_consolidated_html_selections,
    parse_validation_overrides_groups,
)
from utils.state_manager import StateManager  # noqa: E402

FIXTURES = SKILL_ROOT / "tests" / "fixtures" / "legacy_judge_keys"

#: Every legacy spelling, as a regex over serialized text. ``claude_decide_<hash>``
#: batch ids are excluded on purpose: an id is opaque and old ones stay old.
LEGACY_TOKEN = re.compile(
    r"claude_auto_decide|claude_skipped|claude_decide_item_ids|claudeDecided"
    r"|\bclaude_decide\b(?!_)"
)

#: The rename, spelled out by hand so the twin does not depend on judge_vocab.
_TEXTUAL_RENAME = (
    ("claude_decide_item_ids", "judge_decide_item_ids"),
    ("claude_auto_decide", "judge_auto_decide"),
    ("claude_skipped", "judge_skipped"),
    ('"claude_decide"', '"judge_decide"'),
    ("claudeDecided", "judgeDecided"),
)

SPELLINGS = ("legacy", "current")


def _fixture_text(name: str, spelling: str) -> str:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    if spelling == "current":
        for old, new in _TEXTUAL_RENAME:
            text = text.replace(old, new)
        assert not LEGACY_TOKEN.search(text), name
    else:
        assert LEGACY_TOKEN.search(text), f"{name} no longer carries a legacy spelling"
    return text


def _fixture(name: str, spelling: str):
    return json.loads(_fixture_text(name, spelling))


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------

class TestTheMap:
    def test_every_legacy_spelling_maps_onto_a_judge_name(self):
        for old, new in {**LEGACY_VALUES, **LEGACY_KEYS}.items():
            assert "claude" in old.lower()
            assert "claude" not in new.lower() and "judge" in new.lower()
        assert set(LEGACY_VALUES.values()) == {
            JUDGE_DECIDE, JUDGE_AUTO_DECIDE, JUDGE_AUTO_DECIDE_SALVAGE, JUDGE_SKIPPED,
        }
        assert set(LEGACY_KEYS.values()) == {
            JUDGE_DECIDE, JUDGE_DECIDE_ITEM_IDS, JUDGE_DECIDED,
        }

    def test_canonical_value_maps_exact_spellings_only(self):
        assert canonical_value("claude_auto_decide_salvage") == JUDGE_AUTO_DECIDE_SALVAGE
        assert canonical_value(JUDGE_AUTO_DECIDE) == JUDGE_AUTO_DECIDE
        assert canonical_value("claude_decide_efabfefb") == "claude_decide_efabfefb"
        assert canonical_value("Let claude_decide run") == "Let claude_decide run"
        assert canonical_value(None) is None
        assert canonical_value(3) == 3

    def test_upgrade_renames_keys_and_values_and_leaves_ids_and_text(self):
        document = {
            "claude_decide_item_ids": ["h1"],
            "decision_mode": "claude_auto_decide",
            "groups": [{"claude_decide": True, "claudeDecided": True}],
            "batch_context": {"batch_id": "claude_decide_efabfefb",
                              "decision_source": "claude_auto_decide_salvage"},
            "reason": "claude_auto_decide was the old name",
            "overrides": {"G1S1": "claude_decide", "h2": "valid"},
            "items": ["claude_skipped", "invalid"],
        }
        assert upgrade_legacy(document) is document
        assert document == {
            "judge_decide_item_ids": ["h1"],
            "decision_mode": "judge_auto_decide",
            "groups": [{"judge_decide": True, "judgeDecided": True}],
            "batch_context": {"batch_id": "claude_decide_efabfefb",
                              "decision_source": "judge_auto_decide_salvage"},
            "reason": "claude_auto_decide was the old name",
            "overrides": {"G1S1": "judge_decide", "h2": "valid"},
            "items": ["judge_skipped", "invalid"],
        }

    def test_a_new_key_wins_over_its_legacy_twin(self):
        document = {"claude_decide_item_ids": ["stale"], "judge_decide_item_ids": ["new"]}
        assert upgrade_legacy(document) == {"judge_decide_item_ids": ["new"]}

    @pytest.mark.parametrize("legacy_first", [True, False])
    @pytest.mark.parametrize("old_key, new_key, old_value, empty", [
        ("claude_decide_item_ids", JUDGE_DECIDE_ITEM_IDS, ["a", "b"], []),
        ("claude_decide_item_ids", JUDGE_DECIDE_ITEM_IDS, ["a"], None),
        ("claude_decide", JUDGE_DECIDE, True, False),
        ("claudeDecided", JUDGE_DECIDED, 3, 0),
    ])
    def test_an_empty_new_twin_does_not_erase_a_populated_legacy_key(
        self, old_key, new_key, old_value, empty, legacy_first
    ):
        """A writer that filled in the new key's default over an old document
        must not cost it the value the old key carried (review 2026-10-02)."""
        pairs = [(old_key, old_value), (new_key, empty)]
        document = dict(pairs if legacy_first else reversed(pairs))
        assert upgrade_legacy(document) == {new_key: old_value}

    def test_an_empty_legacy_twin_never_replaces_the_new_key(self):
        for new_value in ([], ["new"]):
            document = {"judge_decide_item_ids": new_value, "claude_decide_item_ids": []}
            assert upgrade_legacy(document) == {"judge_decide_item_ids": new_value}

    def test_a_kept_legacy_value_is_upgraded_too(self):
        document = {"human_review_config": {
            "claude_decide_item_ids": ["claude_skipped"], "judge_decide_item_ids": []}}
        assert upgrade_legacy(document) == {
            "human_review_config": {"judge_decide_item_ids": ["judge_skipped"]}}

    def test_the_label_helpers_read_both_spellings(self):
        for source in ("claude_auto_decide", JUDGE_AUTO_DECIDE):
            assert judge_source_label(source) == "judge agent"
        for source in ("claude_auto_decide_salvage", JUDGE_AUTO_DECIDE_SALVAGE):
            assert judge_source_label(source) == "judge agent (salvaged)"
        for status in ("claude_skipped", JUDGE_SKIPPED):
            assert judge_status_label(status) == "skipped by judge"
        for marker in ("claude_auto_decide_salvage", JUDGE_AUTO_DECIDE_SALVAGE):
            record = {"decision": "approved", "batch_context": {"batch_action": marker}}
            assert judge_outcome(record) == "salvaged"


# ---------------------------------------------------------------------------
# Writers use only the new names
# ---------------------------------------------------------------------------

#: Runtime string constants that would write a legacy spelling. Comments and
#: docstrings are not constants an output can carry, so they are not scanned.
_LEGACY_LITERAL = re.compile(
    r"^(claude_decide|claude_auto_decide|claude_auto_decide_salvage|claude_skipped"
    r"|claude_decide_item_ids|claudeDecided|claude_decide_)$"
)


def _runtime_modules():
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        rel = path.relative_to(SKILL_ROOT).as_posix()
        if rel.startswith(("tests/", ".venv/", "legacy/")) or "/." in rel:
            continue
        yield rel, path


class TestWritersUseOnlyTheNewNames:
    def test_no_runtime_module_carries_a_legacy_spelling(self):
        """`judge_vocab.py`'s map is the only place a `claude_*` judge name lives."""
        offenders = []
        for rel, path in _runtime_modules():
            if rel == "utils/judge_vocab.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and _LEGACY_LITERAL.match(node.value)):
                    offenders.append(f"{rel}:{node.lineno}: {node.value!r}")
        assert not offenders, "\n".join(offenders)

    def test_the_templates_write_the_new_override_value(self):
        """The one legacy literal a template may hold is in its localStorage
        migration, which turns an old saved selection into the new value."""
        for name in ("report_template.html", "pr_report_template.html",
                     "consolidated_report_template.html"):
            text = (SKILL_ROOT / "templates" / name).read_text(encoding="utf-8")
            assert "value=\"judge_decide\"" in text, name
            for line in text.splitlines():
                if LEGACY_TOKEN.search(line):
                    assert ("overrides[k] === 'claude_decide'" in line
                            or "before the judge rename" in line), (name, line)
        pr = (SKILL_ROOT / "templates" / "pr_report_template.html").read_text(
            encoding="utf-8")
        assert "hd.judgeDecided" in pr and "claudeDecided" not in pr

    def test_the_documents_a_host_reads_name_only_the_new_keys(self):
        """The instructions are the interface the host agent reads the keys by."""
        paths = [*sorted((SKILL_ROOT / "instructions").glob("apply-*.md")),
                 SKILL_ROOT / "references" / "human-decision-batch.md",
                 *sorted((SKILL_ROOT / "legacy" / "instructions").glob("apply-*.md")),
                 SKILL_ROOT / "legacy" / "references" / "human-decision-batch.md"]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert not LEGACY_TOKEN.search(text), path
        reference = paths[3].read_text(encoding="utf-8")
        for name in (JUDGE_AUTO_DECIDE, JUDGE_AUTO_DECIDE_SALVAGE, JUDGE_DECIDE_ITEM_IDS):
            assert name in reference

    def test_recorded_verdicts_carry_the_new_markers_and_batch_prefix(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        state = StateManager(plan)
        ingest = VerdictIngest(verdicts={
            "g1": {"group_id": "g1", "decision": "skip", "reason": "subjective",
                   "salvaged_description": None, "dropped": None},
            "g2": {"group_id": "g2", "decision": "salvage", "reason": "half",
                   "salvaged_description": "the core", "dropped": "the rest"},
        })
        record_verdicts(state=state, phase="apply-suggestions", ingest=ingest,
                        items_by_group={}, default_judge="codex:gpt-6-astra")
        state.save()
        text = state.state_file.read_text(encoding="utf-8")
        assert not LEGACY_TOKEN.search(text)
        assert "claude_decide_" not in text
        records = state.get_all_human_decisions("apply-suggestions")
        assert records["g1"]["batch_context"]["decision_source"] == JUDGE_AUTO_DECIDE
        assert records["g2"]["batch_context"]["batch_action"] == JUDGE_AUTO_DECIDE_SALVAGE
        assert records["g1"]["batch_context"]["batch_id"].startswith("judge_decide_")

    def test_the_human_review_config_and_the_skip_status(self, tmp_path):
        plan_path = _setup_plan_with_output(tmp_path)
        orch = MockApplyOrchestrator(_make_args(plan_file=str(plan_path), yes=True))
        assert orch.args.agent_decide is True
        orch.formatted_human = [{"group_id": "h1", "importance": "LOW",
                                 "decision_mode": JUDGE_AUTO_DECIDE}]
        config = orch.build_human_review_config()
        assert config["decision_mode"] == JUDGE_AUTO_DECIDE
        assert config[JUDGE_DECIDE_ITEM_IDS] == ["h1"]
        assert not LEGACY_TOKEN.search(json.dumps(config))

        orch._setup()
        orch.skipped = [{"group_hash": "h2", "theme": "T", "suggestions": [],
                         "validation_status": "skipped", "reason": "r",
                         "decision_source": JUDGE_AUTO_DECIDE}]
        orch._rebuild_formatted_skipped()
        assert orch.formatted_skipped[0]["status"] == JUDGE_SKIPPED

    def test_a_ticked_agent_box_parses_to_the_new_value(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(
            "## G1 [aabb000000000001]: Theme\n- [x] Let the agent decide\n",
            encoding="utf-8")
        assert parse_validation_overrides_groups(str(report)) == {
            "aabb000000000001": JUDGE_DECIDE}

    def test_the_report_overlay_writes_judge_decided(self):
        info = html_report_generator._normalize_human_decision(
            {"decision": "skipped",
             "batch_context": {"decision_source": JUDGE_AUTO_DECIDE}})
        assert info[JUDGE_DECIDED] is True and "claudeDecided" not in info


# ---------------------------------------------------------------------------
# Readers accept the old spelling and behave identically
# ---------------------------------------------------------------------------

def _plan_with_recorded_decisions(root: Path, spelling: str):
    """A plan whose state.json was saved by a run that recorded judge decisions.

    The records are injected into the file on disk, exactly as the earlier
    version's ``StateManager.save`` left them.
    """
    plan_path = _setup_plan_with_output(root)
    seed = StateManager(plan_path)
    seed.save()
    raw = json.loads(seed.state_file.read_text(encoding="utf-8"))
    raw[f"human_decisions_{MockApplyOrchestrator.phase_name}"] = _fixture(
        "human_decisions.json", spelling)
    seed.state_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return plan_path, seed.state_file


def _resumed_orchestrator(plan_path: Path):
    orch = MockApplyOrchestrator(
        _make_args(plan_file=str(plan_path), resume=True, yes=True))
    orch._setup()
    orch.write_tasks = [{"task_id": "apply-batch-0", "expected_ids": ["a"]}]
    orch.write_result = _FakeSchedulerResult(
        [_FakeOutcome("apply-batch-0", duration_seconds=4.0)])
    orch.formatted_valid = [
        {"group_id": "9a9a9a9a9a9a9a9a", "title": "Retry uploads"},
        {"group_id": "5c5c5c5c5c5c5c5c", "title": "Guard the queue"},
    ]
    return orch


def _normalized(text: str, root: Path) -> str:
    return text.replace(str(root), "<ROOT>")


class TestAnEarlierRunLoadsTheSame:
    def test_a_resume_across_the_upgrade_reports_the_same_decisions(
        self, tmp_path, capsys, monkeypatch
    ):
        from utils import provider_registry

        seen = {}
        for spelling in SPELLINGS:
            # One process pins one plan (the run anchor); this test runs two.
            monkeypatch.setattr(provider_registry, "_RUN_ANCHOR", None)
            root = tmp_path / spelling
            root.mkdir()
            plan_path, state_file = _plan_with_recorded_decisions(root, spelling)
            orch = _resumed_orchestrator(plan_path)
            capsys.readouterr()  # the constructor's and the previous pass's lines

            decided = orch.judge_decisions()
            orch.print_write_outcome_summary()
            out = capsys.readouterr().out
            summary = Path(orch.write_apply_summary_report()).read_text(encoding="utf-8")
            # No save() by hand: on the native leg nothing saves after the
            # load (completion is the host's --mark-completed, later), and the
            # host reads this file raw for its "Decided by the judge agent"
            # section. The resume itself must have left the new names on disk.
            on_disk = state_file.read_text(encoding="utf-8")

            seen[spelling] = (
                sorted(decided),
                orch._judge_breakdown(decided),
                _normalized(out, root),
                _normalized(summary, root),
            )
            # Only the new names on disk; old batch ids stay as ids.
            assert not LEGACY_TOKEN.search(on_disk), spelling
            assert "judge_auto_decide_salvage" in on_disk
            assert "claude_decide_efabfefb" in on_disk

        assert seen["legacy"] == seen["current"]
        ids, breakdown, out, summary = seen["legacy"]
        assert ids == ["3b810046f0a56e8c", "5c5c5c5c5c5c5c5c", "9a9a9a9a9a9a9a9a"]
        assert breakdown == " (1 approved, 1 salvaged, 1 skipped)"
        assert "Decided by the judge agent (3)" in out
        assert "source: judge agent (salvaged)" in out
        assert "## Decided by the judge agent (3)" in summary

    def test_the_pre_mark_audit_reads_an_old_config_and_old_records(self):
        results = {}
        for spelling in SPELLINGS:
            config = _fixture("orchestrator_output.json", spelling)["human_review_config"]
            records = _fixture("human_decisions.json", spelling)
            # The pre-marked item was judged, and one that was not.
            records["6d6d6d6d6d6d6d6d"] = dict(records["5c5c5c5c5c5c5c5c"])
            results[spelling] = validate_judge_decide_items_honored(config, records)
            config[JUDGE_DECIDE_ITEM_IDS if spelling == "current"
                   else "claude_decide_item_ids"].append("deadbeef00000000")
            ok, warnings = validate_judge_decide_items_honored(config, records)
            assert not ok and "deadbeef00000000" in warnings[0], spelling
        assert results["legacy"] == results["current"] == (True, [])

    def test_util_decisions_reads_an_old_orchestrator_output(self, tmp_path, capsys):
        printed = {}
        for spelling in SPELLINGS:
            path = tmp_path / f"{spelling}.json"
            path.write_text(_fixture_text("orchestrator_output.json", spelling),
                            encoding="utf-8")
            assert display_decisions.main(
                ["--output-file", str(path), "--phase", "apply-suggestions"]) == 0
            printed[spelling] = capsys.readouterr().out
        assert printed["legacy"] == printed["current"]
        assert "! SKIPPED BY THE JUDGE AGENT (1 suggestions)" in printed["legacy"]
        assert "(judge agent)" in printed["legacy"]
        assert "x INVALID (1 suggestions)" in printed["legacy"]

    def test_an_old_selections_file_still_routes_to_the_judge(self, tmp_path):
        routed = {}
        for spelling in SPELLINGS:
            review_dir = tmp_path / spelling / "review-plan"
            review_dir.mkdir(parents=True)
            (review_dir / "user_selections.json").write_text(
                _fixture_text("user_selections.json", spelling), encoding="utf-8")
            orch = MockApplyOrchestrator(_make_args())
            orch.groups = [
                {"group_hash": "aaaa1111bbbb2222", "theme": "First",
                 "validation_status": "needs-human-decision",
                 "suggestions": [{"suggestion_hash": "s1"}]},
                {"group_hash": "bbbb2222cccc3333", "theme": "Second",
                 "validation_status": "needs-human-decision",
                 "suggestions": [{"suggestion_hash": "s2"}, {"suggestion_hash": "s3"}]},
            ]
            orch.skipped_group_indices = set()
            orch.skipped_suggestion_ids = set()
            orch.load_html_selections(str(review_dir), {})
            orch.merged = json.loads(json.dumps(orch.groups))
            orch.apply_group_validation_overrides()
            orch.apply_suggestion_validation_overrides()
            routed[spelling] = (
                orch.validation_overrides,
                orch.suggestion_validation_overrides,
                orch.merged,
                sorted(orch.judge_decide_overrides),
            )
        assert routed["legacy"] == routed["current"]
        merged = routed["legacy"][2]
        assert [g.get(JUDGE_DECIDE) for g in merged] == [True, True]
        assert [g["validation_status"] for g in merged] == [
            "needs-human-decision", "needs-human-decision"]
        assert routed["legacy"][1] == {"G2S1": JUDGE_DECIDE}

    def test_an_old_consolidated_selections_file_reads_the_new_value(self, tmp_path):
        loaded = {}
        for spelling in SPELLINGS:
            phase_dir = tmp_path / spelling
            phase_dir.mkdir()
            document = {"skipped_groups": [],
                        "validation_overrides": {"abc123def456": "claude_decide"}}
            text = json.dumps(document)
            if spelling == "current":
                text = text.replace('"claude_decide"', '"judge_decide"')
            (phase_dir / "consolidated_user_selections.json").write_text(
                text, encoding="utf-8")
            loaded[spelling] = load_consolidated_html_selections(phase_dir)
        assert loaded["legacy"] == loaded["current"]
        assert loaded["legacy"]["validation_overrides"] == {"abc123def456": JUDGE_DECIDE}

    def test_the_report_overlay_reads_old_records_and_an_old_sidecar(self, tmp_path):
        sidecars = {}
        for spelling in SPELLINGS:
            phase_dir = tmp_path / spelling
            phase_dir.mkdir()
            (phase_dir / "report_data.json").write_text(
                _fixture_text("report_data.json", spelling), encoding="utf-8")
            report, matched = html_report_generator.regenerate_report_with_human_decisions(
                phase_dir, _fixture("human_decisions.json", spelling))
            assert report is not None and matched == 3
            sidecar = (phase_dir / "report_data.json").read_text(encoding="utf-8")
            assert not LEGACY_TOKEN.search(sidecar), spelling
            sidecars[spelling] = json.loads(sidecar)
        assert sidecars["legacy"] == sidecars["current"]
        summary = sidecars["legacy"]["humanDecisionsSummary"]
        assert summary == {"approved": 1, "salvaged": 1, "skipped": 1,
                           JUDGE_DECIDED: 2, "total": 3}

    def test_regenerate_decisions_reads_an_old_state_file(self, tmp_path):
        """The CLI leg reads state.json raw, not through StateManager."""
        sidecars = {}
        for spelling in SPELLINGS:
            phase_dir = tmp_path / spelling / "review-plan"
            phase_dir.mkdir(parents=True)
            (phase_dir / "report_data.json").write_text(
                _fixture_text("report_data.json", "current"), encoding="utf-8")
            state_file = tmp_path / spelling / "state.json"
            state_file.write_text(json.dumps(
                {"human_decisions_apply-suggestions":
                 _fixture("human_decisions.json", spelling)}), encoding="utf-8")
            assert html_report_generator.main(
                ["regenerate-decisions", "--phase-dir", str(phase_dir),
                 "--state-file", str(state_file),
                 "--apply-phase", "apply-suggestions"]) == 0
            sidecars[spelling] = json.loads(
                (phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert sidecars["legacy"] == sidecars["current"]
        assert sidecars["legacy"]["humanDecisionsSummary"][JUDGE_DECIDED] == 2


# ---------------------------------------------------------------------------
# The upgrade reaches the disk, and only on a load that may write
# ---------------------------------------------------------------------------

def _identity(path: Path):
    """What a rewrite of *path* would change: its bytes, inode and mtime."""
    stat = path.stat()
    return path.read_bytes(), stat.st_ino, stat.st_mtime_ns


def _pin_file(path: Path):
    """Back-date *path* (so a same-second rewrite still shows) and return its identity."""
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    return _identity(path)


class TestTheUpgradeIsWrittenBack:
    """A load that upgraded a legacy spelling writes it back, once.

    The host reads ``state.json`` raw (the apply instructions' "Decided by the
    judge agent" section), and a native-leg ``--resume`` saves nothing after
    its load, so an upgrade held only in memory never reached that reader
    (review 2026-10-02).
    """

    def test_the_native_leg_resume_leaves_only_the_new_names_on_disk(self, tmp_path):
        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "legacy")
        orch = MockApplyOrchestrator(
            _make_args(plan_file=str(plan_path), resume=True, yes=True))
        assert orch.run() == 0
        on_disk = state_file.read_text(encoding="utf-8")
        assert not LEGACY_TOKEN.search(on_disk)
        # The host's own filter, over the raw file, finds all three judgements.
        records = json.loads(on_disk)[f"human_decisions_{orch.phase_name}"]
        judged = [gid for gid, r in records.items()
                  if r["batch_context"]["decision_source"]
                  in (JUDGE_AUTO_DECIDE, JUDGE_AUTO_DECIDE_SALVAGE)]
        assert sorted(judged) == [
            "3b810046f0a56e8c", "5c5c5c5c5c5c5c5c", "9a9a9a9a9a9a9a9a"]

    def test_the_write_is_the_upgraded_document_and_nothing_else(self, tmp_path):
        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "legacy")
        before = json.loads(state_file.read_text(encoding="utf-8"))
        state = StateManager(plan_path)
        on_disk = json.loads(state_file.read_text(encoding="utf-8"))
        assert on_disk == upgrade_legacy(before)  # updated_at untouched too
        assert state.state == on_disk

    def test_a_current_file_is_never_rewritten(self, tmp_path):
        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "current")
        pinned = _pin_file(state_file)
        StateManager(plan_path).reload()
        assert _identity(state_file) == pinned

    def test_a_read_only_load_upgrades_in_memory_and_writes_nothing(self, tmp_path):
        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "legacy")
        pinned = _pin_file(state_file)
        state = StateManager(plan_path, read_only=True)
        records = state.get_all_human_decisions(MockApplyOrchestrator.phase_name)
        assert not LEGACY_TOKEN.search(json.dumps(records))
        state.reload()
        with pytest.raises(RuntimeError, match="read-only"):
            state.save()
        assert _identity(state_file) == pinned

    def test_reload_writes_an_upgrade_back(self, tmp_path):
        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "current")
        state = StateManager(plan_path)
        # A file-level writer from before the rename puts the old names back.
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        raw[f"human_decisions_{MockApplyOrchestrator.phase_name}"] = _fixture(
            "human_decisions.json", "legacy")
        state_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        state.reload()
        assert not LEGACY_TOKEN.search(state_file.read_text(encoding="utf-8"))
        assert not LEGACY_TOKEN.search(json.dumps(state.state))

    def test_a_save_that_lands_after_the_read_is_not_overwritten(
        self, tmp_path, monkeypatch
    ):
        from utils import state_manager

        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "legacy")
        newer = json.loads(_fixture_text("human_decisions.json", "current"))
        real_upgrade = state_manager.upgrade_legacy

        def upgrade_while_another_process_saves(document):
            state_file.write_text(json.dumps({"concurrent": newer}), encoding="utf-8")
            return real_upgrade(document)

        monkeypatch.setattr(state_manager, "upgrade_legacy",
                            upgrade_while_another_process_saves)
        StateManager(plan_path)
        assert json.loads(state_file.read_text(encoding="utf-8")) == {"concurrent": newer}

    def test_the_read_only_commands_never_write(self, tmp_path, capsys):
        """`status`, `util next-command`, `--status` and the tasks staleness probe."""
        import check_workflow_prerequisites as prereq
        from utils import next_phase, run_status
        from utils.tasks_file import _plan_edited_after_generate_tasks

        plan_path, state_file = _plan_with_recorded_decisions(tmp_path, "legacy")
        pinned = _pin_file(state_file)
        assert run_status.main(["--plan-file", str(plan_path), "--view", "workflow"]) == 0
        assert next_phase.main([str(plan_path)]) == 0
        with pytest.raises(SystemExit) as exit_:
            prereq.main(["--plan-file", str(plan_path), "--status"])
        assert exit_.value.code == 0
        _plan_edited_after_generate_tasks(str(plan_path), 0.0, 0.0)
        capsys.readouterr()
        assert _identity(state_file) == pinned
