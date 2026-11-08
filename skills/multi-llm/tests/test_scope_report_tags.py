"""Scope tags in the review reports (plans/scope-policy-and-report-tags.md §6, §7
"Reports" and "Persistence/UI").

Every report kind — plan, tasks, code and commit ``report.md``, the consolidated
``consolidated-report.md``, and the flat, PR and consolidated HTML pages — shows
each finding's own (reviewer, provisional) tag and each group's validated tag
with the same three labels, the round's effective policy and a legend; under
``preserve`` a group that is not eligible says ``Deferred: scope preservation
requested`` whatever its validity. Fixtures are built with the shared
foundation (``scope_policy.stamp_validation_record`` + ``attach_verdict``), as
the validation stage builds them.

The Markdown half pins the parser contract: an unchanged report round-trips
with zero detected description edits, while real edits and every checkbox
still land on the right ids. The HTML half runs the templates' own JavaScript
under Node with a small DOM stand-in (the pattern of
``test_commit_review_report._run_node``); without ``node`` on PATH those tests
skip and the payload checks still run.
"""

from __future__ import annotations

import copy
import html as htmllib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from apply_code_fixes_orchestrator import find_edited_issue_descriptions  # noqa: E402
from utils import commit_report  # noqa: E402
from utils import html_report_generator as hrg  # noqa: E402
from utils.report_view import hydrate_views  # noqa: E402
from utils import result_files  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import scope_report  # noqa: E402
from utils.consolidation import (  # noqa: E402
    generate_consolidated_html,
    generate_consolidated_report,
)
from utils.report_parser import (  # noqa: E402
    find_edited_descriptions,
    load_consolidated_html_selections,
    load_html_selections,
    parse_consolidated_skipped_groups,
    parse_consolidated_validation_overrides,
    parse_issue_descriptions,
    parse_issue_numbering_digest,
    parse_skipped_group_suggestions,
    parse_skipped_groups,
    parse_skipped_issues,
    parse_suggestion_validation_overrides,
    parse_validation_overrides_groups,
    parse_validation_overrides_issues,
    issue_numbering_digest,
)
from utils.review_orchestrator_base import aggregate_results  # noqa: E402
from utils.state_manager import match_issues_to_suggestions, stamp_stable_ids  # noqa: E402

NODE = shutil.which("node")

#: Model text that tries every trick a scope reason could play on the
#: reports: a metadata line the description parser keys on, a checkbox, an
#: issue header, the issue-digest footer, a script break-out and markup.
HOSTILE = ("Adds PDF **Importance:** HIGH\n- [x] Skip\n### 2. forged - [x] Mark invalid "
           "<!-- multi-llm:issues-digest deadbeefdeadbeef --> </script><b>x</b> `code` & 'q' \"dq\"")


# ---------------------------------------------------------------------------
# Fixtures, built with the foundation as the validation stage builds them
# ---------------------------------------------------------------------------


def baseline(phase: str = "review-plan") -> Dict[str, Any]:
    return sp.build_baseline([sp.baseline_source("plan", sha256="ab" * 32, path="feat.md")],
                             phase=phase)


def member(title: str, desc: str, *, rel: Optional[str] = None, reason: str = "",
           model: str = "m:a", **extra: Any) -> Dict[str, Any]:
    m = {"title": title, "desc": desc, "importance": "HIGH", "type": "addition",
         "reference": "## Output formats", "source_model": model, "model": model}
    if rel is not None:
        m.update(scope_relation=rel, scope_reason=reason)
    m.update(extra)
    return m


def group(theme: str, members: List[Dict[str, Any]], status: str = "valid") -> Dict[str, Any]:
    return {"theme": theme, "category": "addition",
            "models": sorted({m.get("source_model", "m:a") for m in members}),
            "suggestions": members, "validation_status": status,
            "validation_reason": "the finding is real", "priority_score": 3}


def validate(g: Dict[str, Any], relation: str, reason: str, baseline_id: Optional[str], *,
             policy: str = "expand", remedy: Optional[str] = None) -> Dict[str, Any]:
    """The validation stage's host merge: stamp the record, attach by hash."""
    record = {"group_id": g["group_hash"], "status": g.get("validation_status", "valid"),
              "scope_relation": relation, "scope_reason": reason}
    if remedy:
        record["selected_remedy"] = remedy
    stamped = sp.stamp_validation_record(record, g, baseline_id=baseline_id,
                                         allow_selected_remedy=(policy == "preserve"))
    sp.attach_verdict(g, stamped, hash_matched=True)
    return stamped


def write_state(out_dir: Path, phase: str, policy: str, base: Optional[Dict[str, Any]],
                *, apply_policy: Optional[str] = None,
                deferrals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    sp.record_review_round(state, phase, policy, "explicit", base)
    if apply_policy:
        sp.record_apply_policy(state, phase, apply_policy, "explicit")
    if deferrals is not None:
        sp.record_deferrals(state, phase, deferrals)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return state


def plan_groups(baseline_id: Optional[str], *, policy: str = "expand",
                hostile: bool = False) -> List[Dict[str, Any]]:
    """Three groups: a validated extension (whose reviewer said in scope — a
    leaked extension), a validated in-scope fix with two members, and a
    legacy group no one classified."""
    g1 = group("PDF export", [member("Add PDF export", "Render a PDF next to the CSV.",
                                     rel="in_scope", reason="the reviewer calls it required")])
    g2 = group("CSV header", [
        member("Fix CSV header", "The header row is missing.", rel="in_scope",
               reason="completes the CSV outcome"),
        member("Header order", "Columns are out of order.", rel="scope_extension",
               reason="reorders columns", model="m:b"),
    ], status="needs-human-decision")
    g3 = group("Legacy finding", [member("Old one", "Found before scope tags existed.")])
    groups = [g1, g2, g3]
    stamp_stable_ids(groups)
    validate(g1, "scope_extension", HOSTILE if hostile else "Adds PDF beyond the CSV outcome",
             baseline_id, policy=policy)
    validate(g2, "in_scope", "Completes the CSV requirement", baseline_id, policy=policy)
    return groups


def plan_review(tmp_path: Path, *, policy: Optional[str] = "preserve", phase: str = "review-plan",
                hostile: bool = False, apply_policy: Optional[str] = None,
                groups: Optional[List[Dict[str, Any]]] = None, style: str = "pr"):
    plan = tmp_path / "feat.md"
    plan.write_text("# Feat\n\n## Output formats\nCSV only.\n", encoding="utf-8")
    out = tmp_path / "feat"
    phase_dir = out / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    base = baseline(phase)
    if policy is not None:
        write_state(out, phase, policy, base, apply_policy=apply_policy)
    if groups is None:
        groups = plan_groups(base["baseline_id"] if policy is not None else None,
                             policy=policy or "expand", hostile=hostile)
    (phase_dir / "grouped.json").write_text(json.dumps(groups), encoding="utf-8")
    path = aggregate_results("feat", str(out), str(phase_dir), phase, ["m:a", "m:b"], {},
                             validated_groups=copy.deepcopy(groups), plan_path=str(plan),
                             template_style=style)
    return Path(path), groups, phase_dir, plan


def blocks(report: str) -> Dict[str, str]:
    """``## G<n>`` / ``## CG<n>`` block text keyed by its theme."""
    out = {}
    for match in re.finditer(r"^## C?G\d+(?: \[[0-9a-f]+\])?: ([^\n]+)\n(.*?)(?=^## |\Z)",
                             report, re.M | re.S):
        out[match.group(1)] = match.group(2)
    return out


# ---------------------------------------------------------------------------
# Plan and task Markdown
# ---------------------------------------------------------------------------


class TestPlanAndTaskMarkdown:
    @pytest.mark.parametrize("phase", ["review-plan", "review-tasks"])
    def test_tags_policy_line_legend_and_distinct_group_counts(self, tmp_path, phase):
        path, groups, _phase_dir, _plan = plan_review(tmp_path, phase=phase)
        report = path.read_text(encoding="utf-8")
        assert ("**Scope policy:** preserve. Additional scope was not solicited from "
                "reviewers; extension counts cover returned findings only.") in report
        # Three groups, four suggestions: the counts are per group.
        assert ("**Scope tags:** 1 `Within scope`, 1 `Scope extension`, 1 `Scope unclear` "
                "(validated, of 3 groups); 2 deferred: scope preservation requested") in report
        assert "> " + sp.legend_text() in report
        by_theme = blocks(report)
        pdf = by_theme["PDF export"]
        assert "**Group scope:** `Scope extension` (validated) — Adds PDF beyond the CSV outcome" in pdf
        # The reviewer said in scope: shown as its own provisional tag, never as the verdict.
        assert ("**Scope:** `Within scope` (reviewer, provisional) — the reviewer calls it "
                "required") in pdf
        # A real (valid) finding rejected by policy is deferred, not invalid.
        assert "**Validation:** Valid" in pdf
        assert "**Scope outcome:** Deferred: scope preservation requested — the proposed remedy extends" in pdf
        csv = by_theme["CSV header"]
        assert "**Group scope:** `Within scope` (validated)" in csv
        assert "**Scope outcome:** Eligible under scope preservation" in csv
        assert "**Scope:** `Scope extension` (reviewer, provisional) — reorders columns" in csv
        legacy = by_theme["Legacy finding"]
        assert "**Group scope:** `Scope unclear` — no validated scope classification" in legacy
        assert "**Scope:** `Scope unclear` (reviewer, provisional) — not classified" in legacy
        assert "fresh classified review, or apply with --scope-policy expand" in legacy

    def test_scope_lines_sit_between_the_boxes_and_the_metadata_line(self, tmp_path):
        path, _groups, _pd, _plan = plan_review(tmp_path)
        report = path.read_text(encoding="utf-8")
        for header in re.finditer(r"^### (G\d+S\d+) \[[0-9a-f]+\]: [^\n]*\n", report, re.M):
            after = report[header.end():].split("\n")
            assert after[0] == "- [ ] Skip", after[:3]
            meta = next(i for i, line in enumerate(after) if line.startswith("**Importance:**"))
            assert any(line.startswith("**Scope:** `") for line in after[:meta])
        for header in re.finditer(r"^## (G\d+) \[[0-9a-f]+\]: [^\n]*\n\n", report, re.M):
            after = report[header.end():].split("\n")
            assert after[0] == "- [ ] Skip this group"
            meta = next(i for i, line in enumerate(after) if line.startswith("**Validation:**"))
            assert any(line.startswith("**Group scope:**") for line in after[:meta])

    def test_legacy_data_reads_expand_and_scope_unclear(self, tmp_path):
        groups = [group("Old", [member("One", "first")]), group("Older", [member("Two", "second")])]
        stamp_stable_ids(groups)
        path, _g, _pd, _plan = plan_review(tmp_path, policy=None, groups=groups)
        report = path.read_text(encoding="utf-8")
        assert "**Scope policy:** expand. Scope tags are informational." in report
        assert ("**Scope tags:** 0 `Within scope`, 0 `Scope extension`, 2 `Scope unclear` "
                "(validated, of 2 groups)") in report
        assert "deferred" not in report.split("> **Note:**")[0]
        assert "**Scope outcome:**" not in report
        assert report.count("(reviewer, provisional) — not classified") == 2

    def test_an_apply_override_names_the_review_policy(self, tmp_path):
        path, _g, _pd, _plan = plan_review(tmp_path, apply_policy="expand")
        report = path.read_text(encoding="utf-8")
        assert ("**Scope policy:** expand (the review was requested under preserve). "
                "Additional scope was not solicited") in report
        # Eligibility follows the apply policy: nothing deferred under expand.
        assert "**Scope outcome:**" not in report
        assert "`Scope extension` (validated)" in report

    def test_a_proposal_edited_after_validation_is_never_a_confident_tag(self, tmp_path):
        base = baseline()
        groups = plan_groups(base["baseline_id"])
        groups[1]["suggestions"][0]["desc"] = "The header row is missing; also add XLSX."
        path, _g, _pd, _plan = plan_review(tmp_path, groups=groups)
        csv = blocks(path.read_text(encoding="utf-8"))["CSV header"]
        assert ("**Group scope:** `Scope unclear` (may be stale: the proposal changed) — "
                "validated as `Within scope` before the change: Completes the CSV requirement") in csv
        assert "`Within scope` (validated)" not in csv
        assert "**Scope outcome:** Deferred: scope preservation requested — the proposal changed" in csv

    def test_hostile_reason_text_cannot_reach_the_parsers(self, tmp_path):
        path, groups, _pd, _plan = plan_review(tmp_path, hostile=True)
        report = path.read_text(encoding="utf-8")
        line = next(l for l in report.splitlines() if l.startswith("**Group scope:** `Scope extension`"))
        assert "\n" not in line
        for escaped in (r"\*\*Importance:\*\*", r"- \[x\] Skip", r"\#\#\# 2.", r"\<!--",
                        r"\</script\>"):
            assert escaped in line, escaped
        assert "`code`" not in line and "'code'" in line
        assert parse_skipped_groups(str(path)) == set()
        assert parse_skipped_group_suggestions(str(path)) == set()
        assert parse_validation_overrides_groups(str(path)) == {}
        assert parse_suggestion_validation_overrides(str(path)) == {}
        assert find_edited_descriptions(str(path), copy.deepcopy(groups)) == {}


class TestPlanMarkdownRoundTrip:
    def test_an_unchanged_report_has_zero_description_edits(self, tmp_path):
        path, groups, _pd, _plan = plan_review(tmp_path)
        assert find_edited_descriptions(str(path), copy.deepcopy(groups)) == {}
        assert parse_skipped_groups(str(path)) == set()
        assert parse_skipped_group_suggestions(str(path)) == set()
        assert parse_validation_overrides_groups(str(path)) == {}

    def test_a_description_holding_a_rule_or_header_is_not_an_edit(self, tmp_path):
        base = baseline()
        groups = plan_groups(base["baseline_id"])
        groups[2]["suggestions"][0]["desc"] = "First part.\n---\nSecond part.\n### Notes\nMore."
        stamp_stable_ids(groups)
        path, _g, _pd, _plan = plan_review(tmp_path, groups=groups)
        assert find_edited_descriptions(str(path), copy.deepcopy(groups)) == {}
        # An edit to the readable part is still an edit.
        report = path.read_text(encoding="utf-8").replace("First part.", "First part, revised.")
        path.write_text(report, encoding="utf-8")
        edits = find_edited_descriptions(str(path), copy.deepcopy(groups))
        assert list(edits) == [groups[2]["suggestions"][0]["suggestion_hash"]]

    def test_real_edits_and_every_checkbox_round_trip_by_id(self, tmp_path):
        path, groups, _pd, _plan = plan_review(tmp_path)
        report = path.read_text(encoding="utf-8")
        g1, g2, g3 = groups
        csv_hash = g2["suggestions"][0]["suggestion_hash"]
        order_hash = g2["suggestions"][1]["suggestion_hash"]
        # Tick the PDF group's Skip box, the CSV group's Mark invalid, one
        # member's Skip, and edit one member's description.
        report = report.replace(f"## {g1['display_label']} [{g1['display_hash']}]: PDF export\n\n- [ ] Skip this group",
                                f"## {g1['display_label']} [{g1['display_hash']}]: PDF export\n\n- [x] Skip this group")
        csv_block_start = report.index(f"[{g2['display_hash']}]: CSV header")
        head, tail = report[:csv_block_start], report[csv_block_start:]
        tail = tail.replace("- [ ] Mark invalid", "- [x] Mark invalid", 1)
        report = head + tail
        order_header = next(m for m in re.finditer(r"^### (G\d+S\d+) \[([0-9a-f]+)\]: Header order\n- \[ \] Skip",
                                                   report, re.M))
        report = report.replace(order_header.group(0), order_header.group(0).replace("- [ ] Skip", "- [x] Skip"))
        report = report.replace("The header row is missing.", "The header row is missing; keep it first.")
        path.write_text(report, encoding="utf-8")

        # The parsers return the headers' display hashes (the apply phase's
        # filtering resolves each by its unique prefix); every one names the
        # box's own group or suggestion.
        assert parse_skipped_groups(str(path)) == {g1["display_hash"]}
        assert g1["group_hash"].startswith(g1["display_hash"])
        order = g2["suggestions"][1]
        assert parse_skipped_group_suggestions(str(path)) == {order["display_hash"]}
        assert order_hash.startswith(order["display_hash"])
        assert parse_validation_overrides_groups(str(path)) == {g2["display_hash"]: "invalid"}
        assert parse_suggestion_validation_overrides(str(path)) == {}
        edits = find_edited_descriptions(str(path), copy.deepcopy(groups))
        assert set(edits) == {csv_hash}
        assert edits[csv_hash][1] == "The header row is missing; keep it first."
        assert g3["display_hash"] not in parse_skipped_groups(str(path))


# ---------------------------------------------------------------------------
# Code and commit Markdown
# ---------------------------------------------------------------------------


def issue(title: str, file: str, model: str, *, rel: Optional[str] = None, reason: str = "",
          desc: Optional[str] = None) -> Dict[str, Any]:
    i = {"title": title, "desc": desc or f"{title}: the details.", "importance": "high",
         "file": file, "line_range": [3, 4], "type": "bug", "model": model, "source_model": model}
    if rel is not None:
        i.update(scope_relation=rel, scope_reason=reason)
    return i


def code_review(tmp_path: Path, *, policy: Optional[str] = "preserve", commit: bool = False,
                hostile: bool = False):
    plan = tmp_path / "feat.md"
    plan.write_text("# Feat\n", encoding="utf-8")
    phase_dir = cro.get_phase_dir(plan, "code-review")
    base = baseline("code-review")
    if policy is not None:
        write_state(phase_dir.parent, "code-review", policy, base)
    # One finding two models agree on (a duplicate pair in one group), with
    # different provisional tags, and one single-model finding.
    dup_a = issue("Null check", "a.py", "m:a", rel="in_scope", reason="fixes a regression")
    dup_b = issue("Null check", "a.py", "m:b", rel="scope_extension", reason="adds a guard layer")
    solo = issue("Retry loop", "b.py", "m:b", rel="scope_extension", reason="adds retries")
    results = {"m:a": (True, [dict(dup_a)], None), "m:b": (True, [dict(dup_b), dict(solo)], None)}
    g_dup = {"theme": "Null check", "category": "bug", "models": ["m:a", "m:b"],
             "suggestions": [dict(dup_a), dict(dup_b)], "validation_status": "valid",
             "validation_reason": "ok"}
    g_solo = {"theme": "Retry loop", "category": "bug", "models": ["m:b"],
              "suggestions": [dict(solo)], "validation_status": "needs-human-decision",
              "validation_reason": "unsure"}
    groups = [g_dup, g_solo]
    stamp_stable_ids(groups)
    bid = base["baseline_id"] if policy is not None else None
    if policy is not None:
        validate(g_dup, "in_scope", "Fixes the regression the change introduced", bid)
        validate(g_solo, "scope_extension", HOSTILE if hostile else "Adds retries the plan never asked for", bid)
    payload = None
    if commit:
        head, base_sha = "c" * 40, "b" * 40
        payload = commit_report.report_payload({
            "base": base_sha, "head": head,
            "commits": [{"sha": head, "date": "2026-10-31", "subject": "Fix it",
                         "merge": False, "outside_window": False}],
            "phrase": "last commit", "window": None, "notes": [],
            "size": {"commits": 1, "files": 2, "added": 3, "deleted": 1, "lines": 4, "large": False},
            "brief": "plans/commit-reviews/x.md", "drifted": [],
        })
    report = cro.generate_review_report(plan, results, ["a.py", "b.py"], groups,
                                        phase_dir=phase_dir, commit_review=payload)
    path = phase_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    issues = [dict(dup_a), dict(dup_b), dict(solo)]
    return path, report, groups, issues, phase_dir


def issue_blocks(report: str) -> Dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(
        r"^### (\d+)\. [^\n]*\n(.*?)(?=^### \d+\.|^## |\Z)", report, re.M | re.S)}


class TestCodeAndCommitMarkdown:
    @pytest.mark.parametrize("commit", [False, True], ids=["code", "commits"])
    def test_duplicate_issues_share_the_group_tag_and_count_once(self, tmp_path, commit):
        path, report, groups, _issues, _pd = code_review(tmp_path, commit=commit)
        assert "**Total Issues:** 3" in report
        assert ("**Scope tags:** 1 `Within scope`, 1 `Scope extension`, 0 `Scope unclear` "
                "(validated, of 2 groups); 1 deferred: scope preservation requested") in report
        assert "> " + sp.legend_text() in report
        by_number = issue_blocks(report)
        null_checks = [b for b in by_number.values() if "Null check: the details." in b]
        assert len(null_checks) == 2
        for block in null_checks:
            assert "**Group scope:** `Within scope` (validated) — Fixes the regression" in block
            assert "**Scope outcome:** Eligible under scope preservation" in block
        # Each duplicate keeps its own reviewer's provisional tag.
        provisional = sorted(re.search(r"\*\*Scope:\*\* `([^`]+)` \(reviewer, provisional\)", b).group(1)
                             for b in null_checks)
        assert provisional == ["Scope extension", "Within scope"]
        retry = next(b for b in by_number.values() if "Retry loop: the details." in b)
        assert "**Group scope:** `Scope extension` (validated)" in retry
        assert "**Scope outcome:** Deferred: scope preservation requested" in retry
        if commit:
            assert "Commit brief" in report

    def test_legacy_code_report_reads_scope_unclear(self, tmp_path):
        _path, report, _g, _i, _pd = code_review(tmp_path, policy=None)
        assert "**Scope policy:** expand. Scope tags are informational." in report
        assert "of 2 groups)" in report and "2 `Scope unclear`" in report
        assert report.count("**Group scope:** `Scope unclear` — no validated scope classification") == 3

    def test_unchanged_report_round_trips_with_zero_edits_and_real_ones_land(self, tmp_path):
        path, report, groups, issues, _pd = code_review(tmp_path, hostile=True)
        mapping = match_issues_to_suggestions(issues, groups)
        assert len(mapping) == 3
        # Unchanged: no edits, no ticks, the digest footer is the real one.
        assert find_edited_issue_descriptions(str(path), copy.deepcopy(groups), mapping) == {}
        assert parse_skipped_issues(str(path)) == set()
        assert parse_validation_overrides_issues(str(path)) == {}
        assert parse_issue_numbering_digest(str(path)) == issue_numbering_digest(issues)
        descriptions = parse_issue_descriptions(str(path))
        assert len(descriptions) == 3
        for number, text in descriptions.items():
            # The needs-human issue's validation reason blockquote is part of
            # its block as before; `normalize_description` drops it.
            assert text.split("\n\n")[-1] in {i["desc"] for i in issues}, (number, text)
        # Tick Skip on #2, Mark valid on the needs-human issue, edit #1's description.
        retry_no = next(n for n, b in issue_blocks(report).items() if "Retry loop: the details." in b)
        edited = report.replace("### 2. Null check\n- [ ] Skip", "### 2. Null check\n- [x] Skip")
        start = edited.index(f"### {retry_no}. Retry loop")
        edited = edited[:start] + edited[start:].replace("- [ ] Mark valid", "- [x] Mark valid", 1)
        first = edited.index("### 1. Null check")
        edited = edited[:first] + edited[first:].replace("Null check: the details.",
                                                         "Null check: guard user.profile only.", 1)
        path.write_text(edited, encoding="utf-8")
        assert parse_skipped_issues(str(path)) == {2}
        assert parse_validation_overrides_issues(str(path)) == {retry_no: "valid"}
        edits = find_edited_issue_descriptions(str(path), copy.deepcopy(groups), mapping)
        assert set(edits) == {1}
        assert edits[1][1] == "Null check: guard user.profile only."


# ---------------------------------------------------------------------------
# Consolidated Markdown and HTML payload
# ---------------------------------------------------------------------------


def consolidated(tmp_path: Path, *, policy: Optional[str] = "preserve"):
    out = tmp_path / "feat"
    phase_dir = out / "review-plan"
    phase_dir.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "feat.md"
    plan.write_text("# Feat\n", encoding="utf-8")
    base = baseline()
    if policy is not None:
        write_state(out, "review-plan", policy, base)
    groups = plan_groups(base["baseline_id"] if policy is not None else None)
    validation = [{"group_id": g["group_hash"], "group_index": i, "status": g["validation_status"],
                   "reason": "r", "confidence": 0.9} for i, g in enumerate(groups)]
    cgs = [
        {"consolidated_id": "a1b2c3d4e5f6", "display_index": 1, "title": "Output fixes",
         "description": "Fix the CSV and add PDF.", "importance": "HIGH", "reference": "## Output formats",
         "underlying_group_indices": [0, 1],
         "underlying_group_ids": [groups[0]["group_hash"], groups[1]["group_hash"]],
         "model_count": 2, "is_singleton": False, "original_suggestion_count": 3, "reasoning": "related"},
        {"consolidated_id": "0f0e0d0c0b0a", "display_index": 2, "title": "Legacy",
         "description": "Old.", "importance": "LOW", "reference": "", "underlying_group_indices": [2],
         "underlying_group_ids": [groups[2]["group_hash"]], "model_count": 1, "is_singleton": True,
         "original_suggestion_count": 1, "reasoning": ""},
    ]
    return phase_dir, plan, groups, validation, cgs


class TestConsolidatedMarkdown:
    def test_umbrella_is_derived_and_mixed_members_keep_their_tags(self, tmp_path):
        phase_dir, _plan, groups, validation, cgs = consolidated(tmp_path)
        path = Path(generate_consolidated_report(cgs, groups, str(phase_dir), "feat",
                                                 validation=validation))
        report = path.read_text(encoding="utf-8")
        by_theme = blocks(report)
        output = by_theme["Output fixes"]
        g1, g2 = groups[0]["display_label"], groups[1]["display_label"]
        assert (f"**Scope:** `Scope extension` (derived from its groups; mixed members) — "
                f"contains extension ({g1})") in output
        assert (f"**Scope outcome:** Deferred: scope preservation requested for 1 of 2 groups "
                f"({g1}); each group is decided on its own") in output
        # Member detail survives: each underlying group's validated tag, each
        # member's provisional tag.
        assert "**Group scope:** `Scope extension` (validated)" in output
        assert "**Group scope:** `Within scope` (validated)" in output
        assert output.count("(reviewer, provisional)") == 3
        legacy = by_theme["Legacy"]
        assert "`Scope unclear` (derived from its groups) — contains members whose scope is unclear" in legacy
        # Distinct underlying groups, not umbrellas or suggestions.
        assert ("**Scope tags:** 1 `Within scope`, 1 `Scope extension`, 1 `Scope unclear` "
                "(validated, of 3 underlying groups); 2 deferred") in report
        assert "> " + sp.legend_text() in report
        # The checkboxes still parse by consolidated id.
        assert parse_consolidated_skipped_groups(str(path)) == set()
        assert parse_consolidated_validation_overrides(str(path)) == {}
        ticked = report.replace("## CG1 [a1b2c3d4e5f6]: Output fixes\n- [ ] Skip this group",
                                "## CG1 [a1b2c3d4e5f6]: Output fixes\n- [x] Skip this group")
        path.write_text(ticked, encoding="utf-8")
        assert parse_consolidated_skipped_groups(str(path)) == {"a1b2c3d4e5f6"}

    def test_a_verdict_joins_from_validation_by_hash_only(self, tmp_path):
        phase_dir, _plan, groups, validation, cgs = consolidated(tmp_path)
        stamped = []
        for g in groups:
            verdict = g.pop(sp.VERDICT_KEY, None)
            record = {"group_id": g["group_hash"], "group_index": len(stamped), "status": "valid"}
            if verdict:
                record.update(scope_relation=verdict["relation"], scope_reason=verdict["reason"],
                              scope_baseline_id=verdict["baseline_id"],
                              scope_proposal_digest=verdict["proposal_digest"], scope_source="validation")
            stamped.append(record)
        scoped = scope_report.with_verdicts(groups, stamped)
        assert sp.authority(scoped[0], baseline()["baseline_id"]).relation == sp.SCOPE_EXTENSION
        # An index-only record (no hash) carries no scope authority.
        no_hash = [dict(r, group_id=None) for r in stamped]
        assert sp.VERDICT_KEY not in scope_report.with_verdicts(groups, no_hash)[0]


# ---------------------------------------------------------------------------
# HTML payloads, regeneration and the sidecar
# ---------------------------------------------------------------------------


class TestHtmlPayload:
    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_report_data_persists_tags_baseline_bindings_and_outcomes(self, tmp_path, style):
        _path, groups, phase_dir, _plan = plan_review(tmp_path, style=style)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        block = data["scopePolicy"]
        assert block["effectivePolicy"] == "preserve" and block["reviewPolicy"] == "preserve"
        assert block["baselineId"] == baseline()["baseline_id"]
        assert block["baseline"]["sources"][0]["role"] == "plan"
        assert block["counts"] == {"in_scope": 1, "scope_extension": 1, "unclear": 1, "deferred": 2}
        by_hash = {g["groupHash"]: g for g in data["groups"]}
        pdf = by_hash[groups[0]["group_hash"]]["scope"]
        assert pdf["relation"] == "scope_extension" and pdf["authority"] == "bound"
        assert pdf["proposalDigest"] == groups[0][sp.VERDICT_KEY]["proposal_digest"]
        assert pdf["decision"]["outcome"] == "deferred" and not pdf["decision"]["eligible"]
        assert by_hash[groups[0]["group_hash"]]["suggestions"][0]["scope"] == {
            "relation": "in_scope", "label": "Within scope",
            "reason": "the reviewer calls it required", "reference": "", "classified": True,
            "provisional": True}
        # Every view's group, rebuilt from the sidecar, carries the same entry.
        for bucket in hydrate_views(data)["sectionView"].values():
            for copy_ in bucket["suggestions"]:
                assert copy_["scope"] == by_hash[copy_["groupHash"]]["scope"]

    def test_selected_remedy_is_persisted_under_preserve(self, tmp_path):
        base = baseline()
        groups = plan_groups(base["baseline_id"], policy="preserve")
        validate(groups[1], "in_scope", "Completes the CSV requirement", base["baseline_id"],
                 policy="preserve", remedy="Add the header row only")
        _path, _g, phase_dir, _plan = plan_review(tmp_path, groups=groups)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        entry = next(g for g in data["groups"] if g["groupHash"] == groups[1]["group_hash"])["scope"]
        assert entry["selectedRemedy"] == "Add the header row only"
        assert entry["authority"] == "bound"
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Selected remedy:** Add the header row only" in report

    def test_the_payload_escapes_script_breakouts(self, tmp_path):
        _path, _g, phase_dir, _plan = plan_review(tmp_path, hostile=True)
        page = (phase_dir / "report.html").read_text(encoding="utf-8")
        data_text = re.search(r"const reportData = (\{.*?\n\});", page, re.S).group(1)
        assert "</" not in data_text.replace("<\\/", "") and "<!--" not in data_text
        assert "<\\/script>" in data_text and "<\\!--" in data_text
        # Once the browser's JS parser reads the literal, the text is whole again.
        parsed = json.loads(data_text.replace("<\\!--", "<!--"))
        assert sp.flatten_text(HOSTILE) in [g["scope"]["reason"] for g in parsed["groups"]]


class TestRegeneration:
    def _no_spawn(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError(f"regeneration spawned {args!r}")
        monkeypatch.setattr(subprocess, "run", refuse)
        monkeypatch.setattr(subprocess, "Popen", refuse)

    def test_an_apply_override_recomputes_eligibility_without_the_plan(self, tmp_path, monkeypatch):
        _path, groups, phase_dir, plan = plan_review(tmp_path)
        before = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        plan.unlink()  # the source plan is unavailable
        self._no_spawn(monkeypatch)
        state = json.loads((phase_dir.parent / "state.json").read_text(encoding="utf-8"))
        assert not hrg.scope_regeneration_needed(phase_dir, state)
        sp.record_apply_policy(state, "apply-suggestions", "expand", "explicit")
        assert hrg.scope_regeneration_needed(phase_dir, state)
        path, matched = hrg.regenerate_report_with_human_decisions(phase_dir, {}, scope_state=state)
        assert matched == 0
        after = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert after["scopePolicy"]["effectivePolicy"] == "expand"
        assert after["scopePolicy"]["policyLine"].startswith(
            "Scope policy: expand (the review was requested under preserve)")
        for old, new in zip(before["groups"], after["groups"]):
            assert new["scope"]["decision"]["eligible"] is True
            # The historical classification is untouched.
            assert {k: v for k, v in new["scope"].items() if k != "decision"} == \
                {k: v for k, v in old["scope"].items() if k != "decision"}
        assert after["scopePolicy"]["counts"]["deferred"] == 0
        page = path.read_text(encoding="utf-8")
        assert '"effectivePolicy": "expand"' in page

    def test_preserve_override_and_recorded_deferral_reasons(self, tmp_path):
        _path, groups, phase_dir, _plan = plan_review(tmp_path, policy="expand")
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert all(g["scope"]["decision"]["eligible"] for g in data["groups"])
        state = json.loads((phase_dir.parent / "state.json").read_text(encoding="utf-8"))
        sp.record_apply_policy(state, "apply-suggestions", "preserve", "explicit")
        sp.record_deferrals(state, "apply-suggestions", [{
            "group_id": groups[0]["group_hash"], "policy": "preserve",
            "reason": "Deferred: scope preservation requested: the apply phase's own words"}])
        hrg.regenerate_report_with_human_decisions(phase_dir, {}, scope_state=state)
        after = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        by_hash = {g["groupHash"]: g["scope"] for g in after["groups"]}
        assert by_hash[groups[0]["group_hash"]]["decision"]["detail"] == "the apply phase's own words"
        assert by_hash[groups[1]["group_hash"]]["decision"]["eligible"] is True
        assert by_hash[groups[2]["group_hash"]]["decision"]["outcome"] == "deferred"

    def test_a_legacy_payload_without_a_round_renders_as_before(self, tmp_path):
        groups = [group("Old", [member("One", "first")])]
        stamp_stable_ids(groups)
        _path, _g, phase_dir, _plan = plan_review(tmp_path, policy=None, groups=groups)
        sidecar = phase_dir / "report_data.json"
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data.pop("scopePolicy")
        for g in hrg._payload_groups(data):
            g.pop("scope", None)
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        assert not hrg.scope_regeneration_needed(phase_dir, {})
        hrg.regenerate_report_with_human_decisions(phase_dir, {}, scope_state={})
        assert "scopePolicy" not in json.loads(sidecar.read_text(encoding="utf-8"))

    def test_consolidated_sidecar_regenerates_by_underlying_group(self, tmp_path, monkeypatch):
        phase_dir, plan, groups, validation, cgs = consolidated(tmp_path)
        generate_consolidated_html(cgs, groups, str(phase_dir), str(plan), ["m:a", "m:b"],
                                   validation=validation)
        sidecar = phase_dir / hrg.CONSOLIDATED_REPORT_DATA
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["scopePolicy"]["counts"]["deferred"] == 2
        ug = data["consolidatedGroups"][0]["underlyingGroups"]
        assert [u["scope"]["relation"] for u in ug] == ["scope_extension", "in_scope"]
        assert data["consolidatedGroups"][0]["scope"]["mixed"] is True
        plan.unlink()
        self._no_spawn(monkeypatch)
        decisions = {groups[1]["group_hash"]: {"decision": "approved", "reason": "fine",
                                               "batch_context": {"decision_source": "user"}}}
        state = json.loads((phase_dir.parent / "state.json").read_text(encoding="utf-8"))
        sp.record_apply_policy(state, "apply-suggestions", "expand", "explicit")
        page, matched, status = hrg.regenerate_consolidated_with_human_decisions(
            phase_dir, decisions, scope_state=state)
        assert (status, matched) == ("regenerated", 1)
        after = json.loads(sidecar.read_text(encoding="utf-8"))
        ug = after["consolidatedGroups"][0]["underlyingGroups"]
        assert ug[1]["humanDecision"]["outcome"] == "approved" and "humanDecision" not in ug[0]
        assert all(u["scope"]["decision"]["eligible"] for u in ug)
        assert after["consolidatedGroups"][0]["scope"]["deferred"] == []
        assert after["consolidatedGroups"][0]["scope"]["relation"] == "scope_extension"
        assert after["scopePolicy"]["effectivePolicy"] == "expand"
        assert '"outcome": "approved"' in page.read_text(encoding="utf-8")

    def test_a_legacy_consolidated_page_is_a_reported_no_op(self, tmp_path, capsys):
        phase_dir, plan, groups, validation, cgs = consolidated(tmp_path)
        (phase_dir / hrg.CONSOLIDATED_REPORT_HTML).write_text("<html>old page</html>", encoding="utf-8")
        assert hrg.regenerate_consolidated_with_human_decisions(phase_dir, {})[2] == "legacy"
        assert (phase_dir / hrg.CONSOLIDATED_REPORT_HTML).read_text(encoding="utf-8") == "<html>old page</html>"
        # Through the CLI, beside a regenerated main report.
        _p, _g, _pd, _plan = plan_review(tmp_path)
        state_file = phase_dir.parent / "state.json"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["human_decisions_apply-suggestions"] = {
            groups[0]["group_hash"]: {"decision": "skipped", "reason": "no"}}
        state_file.write_text(json.dumps(state), encoding="utf-8")
        rc = hrg.main(["regenerate-decisions", "--phase-dir", str(phase_dir),
                       "--state-file", str(state_file), "--apply-phase", "apply-suggestions"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "HTML report regenerated with 1 of 1 human decision(s) matched" in out
        assert "predates consolidated_report_data.json" in out
        assert (phase_dir / hrg.CONSOLIDATED_REPORT_HTML).read_text(encoding="utf-8") == "<html>old page</html>"

    def test_cli_regenerates_for_a_policy_change_alone(self, tmp_path, capsys):
        _path, _groups, phase_dir, _plan = plan_review(tmp_path)
        state_file = phase_dir.parent / "state.json"
        argv = ["regenerate-decisions", "--phase-dir", str(phase_dir),
                "--state-file", str(state_file), "--apply-phase", "apply-suggestions"]
        assert hrg.main(argv) == 0
        assert "No human or judge decisions are recorded" in capsys.readouterr().out
        state = json.loads(state_file.read_text(encoding="utf-8"))
        sp.record_apply_policy(state, "apply-suggestions", "expand", "explicit")
        state_file.write_text(json.dumps(state), encoding="utf-8")
        assert hrg.main(argv) == 0
        out = capsys.readouterr().out
        assert "regenerated for this round's recorded scope policy" in out and "0 of 0" not in out
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["scopePolicy"]["effectivePolicy"] == "expand"


class TestSidecarIsNeverAModelResult:
    """``consolidated_report_data.json`` beside real model results is excluded
    by every reaggregation input scan."""

    NAMES = ("consolidated_report_data.json", "report_data.json")

    def _phase_dir(self, tmp_path: Path) -> Path:
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        (phase_dir / "m_a.json").write_text("[]", encoding="utf-8")
        for name in self.NAMES:
            (phase_dir / name).write_text(json.dumps({"groups": []}), encoding="utf-8")
        return phase_dir

    def test_the_shared_scanners_skip_it(self, tmp_path):
        phase_dir = self._phase_dir(tmp_path)
        assert set(result_files.snapshot_candidates(phase_dir)) == {"m_a.json"}
        assert [Path(p).name for p in cro.leftover_model_results(phase_dir)] == ["m_a.json"]

    def test_the_reaggregation_rule_skips_it(self, tmp_path):
        """Both reaggregates read ``result_files.model_result_paths``."""
        phase_dir = self._phase_dir(tmp_path)
        assert [Path(p).name for p in result_files.model_result_paths(phase_dir)] == ["m_a.json"]


class TestImportedSelectionsCarryNoScope:
    def test_scope_and_policy_keys_are_dropped(self, tmp_path, capsys):
        payload = {"format_version": 2, "plan_path": "p.md", "skipped_groups": ["abc"],
                   "scope_policy": "expand", "scopePolicy": {"effectivePolicy": "expand"},
                   "scope_verdict": {"relation": "in_scope"}, "selected_remedy": "x"}
        (tmp_path / "user_selections.json").write_text(json.dumps(payload), encoding="utf-8")
        data = load_html_selections(tmp_path)
        assert data["skipped_groups"] == ["abc"]
        assert not [k for k in data if k.lower().startswith(("scope", "selected"))]
        assert "are ignored" in capsys.readouterr().err
        cons = {"format_version": 2, "skipped_groups": ["a1b2c3d4e5f6"],
                "validation_overrides": {}, "scope_policy": "expand"}
        (tmp_path / "consolidated_user_selections.json").write_text(json.dumps(cons), encoding="utf-8")
        data = load_consolidated_html_selections(tmp_path)
        assert data["skipped_groups"] == ["a1b2c3d4e5f6"] and "scope_policy" not in data


# ---------------------------------------------------------------------------
# The templates' own JavaScript, under Node
# ---------------------------------------------------------------------------

#: A DOM stand-in just big enough for the templates' init paths: persistent
#: elements by id, text escaping as a browser serialises it, a localStorage,
#: a Blob that records what an export wrote, and querySelectorAll over
#: elements a test registers (``__register``) by class and attribute.
DOM_SHIM = r"""
function __escape(v) { return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function ClassList() { this._c = {}; }
ClassList.prototype = {
  add: function (c) { this._c[c] = true; }, remove: function (c) { delete this._c[c]; },
  toggle: function (c, f) { var on = f === undefined ? !this._c[c] : !!f; if (on) this._c[c] = true; else delete this._c[c]; return on; },
  contains: function (c) { return !!this._c[c]; }
};
function El(tag) {
  this.tagName = tag || 'div'; this._html = ''; this.style = {}; this.dataset = {}; this.value = '';
  this.checked = false; this.classList = new ClassList(); this.children = []; this._attrs = {};
  this.scrollHeight = 10;
}
El.prototype = {
  get innerHTML() { return this._html; }, set innerHTML(v) { this._html = String(v); },
  get textContent() { return this._text || ''; },
  set textContent(v) { this._text = String(v); this._html = __escape(v); },
  appendChild: function (n) { if (n && n.__text !== undefined) this._html += __escape(n.__text); else this.children.push(n); return n; },
  removeChild: function () {}, addEventListener: function () {}, removeEventListener: function () {},
  setAttribute: function (k, v) { this._attrs[k] = String(v); }, getAttribute: function (k) { return this._attrs.hasOwnProperty(k) ? this._attrs[k] : null; },
  removeAttribute: function (k) { delete this._attrs[k]; }, querySelector: function () { return new El(); },
  querySelectorAll: function () { return []; }, closest: function () { return null; }, contains: function () { return false; },
  focus: function () {}, setSelectionRange: function () {}, click: function () {}, scrollIntoView: function () {},
  matches: function () { return false; }, getBoundingClientRect: function () { return { top: 0, left: 0, height: 0, width: 0 }; }
};
var __byId = {}, __registered = [], __blobs = [], __store = {};
function __register(classes, attrs) { var e = new El(); e.__classes = classes; e.__attrs = attrs; __registered.push(e); return e; }
function __matches(e, sel) {
  var classes = sel.match(/\.[\w-]+/g) || [], ok = true;
  classes.forEach(function (c) { if ((e.__classes || []).indexOf(c.slice(1)) < 0) ok = false; });
  var re = /\[([\w-]+)="((?:[^"\\]|\\.)*)"\]/g, m;
  while ((m = re.exec(sel))) { if ((e.__attrs || {})[m[1]] !== m[2].replace(/\\(.)/g, '$1')) ok = false; }
  return ok && (classes.length > 0 || /\[/.test(sel));
}
var document = {
  readyState: 'complete', body: new El('body'),
  getElementById: function (id) { if (!__byId[id]) __byId[id] = new El(); return __byId[id]; },
  createElement: function (t) { return new El(t); }, createTextNode: function (v) { return { __text: String(v) }; },
  querySelector: function (sel) { return __registered.filter(function (e) { return __matches(e, sel); })[0] || new El(); },
  querySelectorAll: function (sel) { return __registered.filter(function (e) { return __matches(e, sel); }); },
  addEventListener: function () {}
};
var window = globalThis;
window.addEventListener = function () {}; window.dispatchEvent = function () {};
window.markedParse = function (t) { return '<p>' + __escape(t) + '</p>'; };
var marked = { setOptions: function () {}, parse: function (t) { return '<p>' + __escape(t) + '</p>'; } };
var DOMPurify = { sanitize: function (x) { return x; } };
var localStorage = { getItem: function (k) { return __store.hasOwnProperty(k) ? __store[k] : null; },
                     setItem: function (k, v) { __store[k] = String(v); }, removeItem: function (k) { delete __store[k]; } };
function Blob(parts) { __blobs.push(parts.join('')); }
URL.createObjectURL = function () { return 'blob:test'; }; URL.revokeObjectURL = function () {};
function IntersectionObserver() { this.observe = function () {}; this.disconnect = function () {}; }
function setTimeout() { return 0; }
console.error = function () {}; console.warn = function () {};
"""


def _template_script(page: str) -> str:
    """The page's own code, in page order: the shared script
    (``templates/shared/report_core.js``) and the template's, without the
    vendored libraries."""
    scripts = re.findall(r'<script(?: data-shared="report_core\.js")?>(.*?)</script>', page, re.S)
    assert len(scripts) == 2, "expected the shared script and the page's"
    return "\n;\n".join(scripts)


def run_page(page: str, body: str, store: Optional[Dict[str, str]] = None) -> Any:
    """Run a generated page's main script under Node with the DOM stand-in,
    then *body*, which must ``__out(value)`` once."""
    if NODE is None:
        pytest.skip("node is not on PATH; the payload checks still ran")
    script = (DOM_SHIM + f"__store = {json.dumps(store or {})};\n"
              + "function __out(v) { process.stdout.write('@@OUT@@' + JSON.stringify(v) + '\\n'); }\n"
              + _template_script(page) + "\n;\n" + body)
    # A file, not ``-e``: the PR template's script exceeds one argv string.
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "page.js"
        path.write_text(script, encoding="utf-8")
        done = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    lines = [l for l in done.stdout.splitlines() if l.startswith("@@OUT@@")]
    assert lines, done.stdout[-2000:]
    return json.loads(lines[-1][len("@@OUT@@"):])


class Badges(HTMLParser):
    """Every scope badge in rendered markup, with the scope container it sits in."""

    VOID = {"img", "input", "br", "hr", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, Dict[str, str]]] = []
        self.badges: List[Dict[str, Any]] = []
        self.current: Optional[Dict[str, Any]] = None
        self.tags: List[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        self.tags.append(tag)
        if tag in self.VOID:
            return
        self.stack.append((tag, a))
        if tag == "span" and "scope-badge" in a.get("class", "").split():
            group = next((s[1].get("data-scope-group") for s in reversed(self.stack)
                          if "data-scope-group" in s[1]), None)
            sugg = next((s[1].get("data-scope-suggestion") for s in reversed(self.stack)
                         if "data-scope-suggestion" in s[1]), None)
            self.current = {"text": "", "class": a["class"], "title": a.get("title", ""),
                            "group": group, "suggestion": sugg}

    def handle_endtag(self, tag):
        if tag == "span" and self.current is not None:
            self.badges.append(self.current)
            self.current = None
        while self.stack:
            if self.stack.pop()[0] == tag:
                break

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data


def badges_in(markup: str) -> Badges:
    parser = Badges()
    parser.feed(markup)
    return parser


def _report(tmp_path: Path, style: str, **kw):
    _path, groups, phase_dir, _plan = plan_review(tmp_path, style=style, **kw)
    page = (phase_dir / "report.html").read_text(encoding="utf-8")
    data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
    return page, groups, data


FLAT_RENDER = "init(); __out({groups: document.getElementById('groups-container').innerHTML, " \
              "summary: document.getElementById('scope-summary').innerHTML});"
PR_RENDER = "init(); __out({pr: document.getElementById('pr-content').innerHTML, " \
            "flat: document.getElementById('flat-content').innerHTML, " \
            "summary: document.getElementById('summary-left').innerHTML});"


def _storage_key(data: Dict[str, Any]) -> str:
    return "multi-llm-selections-" + data["phase"] + "-" + data["planPath"]


class TestTemplateBadges:
    @pytest.mark.parametrize("style,render,keys", [
        ("flat", FLAT_RENDER, ["groups"]), ("pr", PR_RENDER, ["pr", "flat"])])
    def test_badges_render_in_every_view(self, tmp_path, style, render, keys):
        page, groups, _data = _report(tmp_path, style, hostile=True)
        out = run_page(page, render)
        assert "Scope policy: preserve." in htmllib.unescape(out["summary"])
        assert "(validated, distinct groups)" in out["summary"]
        for key in keys:
            found = badges_in(out[key])
            assert "script" not in found.tags and "b" not in found.tags  # hostile text stayed text
            by_group = {}
            for b in found.badges:
                if b["group"] is not None:
                    by_group.setdefault(b["group"], []).append(b)
            pdf = [b["text"] for b in by_group[groups[0]["group_hash"]]]
            assert pdf == ["Scope extension (validated)", "Deferred: scope preservation requested"]
            title = next(b["title"] for b in by_group[groups[0]["group_hash"]] if "validated" in b["text"])
            assert "</script><b>x</b>" in title  # the reason, escaped as an attribute
            assert [b["text"] for b in by_group[groups[1]["group_hash"]]] == ["Within scope (validated)"]
            assert [b["text"] for b in by_group[groups[2]["group_hash"]]] == [
                "Scope unclear (not classified)", "Deferred: scope preservation requested"]
            provisional = {b["suggestion"]: b["text"] for b in found.badges if b["suggestion"]}
            assert provisional[groups[0]["suggestions"][0]["suggestion_hash"]] == "Within scope (provisional)"
            assert provisional[groups[2]["suggestions"][0]["suggestion_hash"]] == "Scope unclear (provisional)"
            for b in found.badges:
                assert "scope-badge" in b["class"]

    @pytest.mark.parametrize("style,render,key", [("flat", FLAT_RENDER, "groups"), ("pr", PR_RENDER, "pr")])
    def test_a_legacy_payload_renders_unknown_tags(self, tmp_path, style, render, key):
        page, groups, data = _report(tmp_path, style)
        data.pop("scopePolicy")
        for g in hrg._payload_groups(data):
            g.pop("scope", None)
            for s in g.get("suggestions") or []:
                s.pop("scope", None)
        legacy = hrg.embed_report_data(data, hrg.load_report_template(style))
        out = run_page(legacy, render)
        texts = {b["text"] for b in badges_in(out[key]).badges}
        assert texts == {"Scope unclear (not classified)", "Scope unclear (provisional)"}
        assert "predates scope classification" in out["summary"]

    @pytest.mark.parametrize("style,render,key,editor", [
        ("flat", FLAT_RENDER, "groups", "handleDescriptionEdit"),
        ("pr", PR_RENDER, "pr", "handleDescriptionEditPr")])
    def test_an_edited_description_marks_the_classification_stale(self, tmp_path, style, render,
                                                                   key, editor):
        page, groups, data = _report(tmp_path, style)
        csv = groups[1]
        sid = csv["suggestions"][0]["suggestion_hash"]
        # Restored from localStorage: the page opens with the edit applied.
        store = {_storage_key(data): json.dumps({"edited_descriptions": {sid: "Also add XLSX."},
                                                  "skipped_groups": {csv["group_hash"]: True}})}
        out = run_page(page, render, store)
        mine = [b["text"] for b in badges_in(out[key]).badges if b["group"] == csv["group_hash"]]
        assert mine == ["Scope unclear (may be stale: description edited)",
                        "Deferred: scope preservation requested"]
        assert "Within scope (validated)" not in mine
        member = [b["text"] for b in badges_in(out[key]).badges if b["suggestion"] == sid]
        assert member == ["Within scope (provisional; may be stale: description edited)"]
        # The restored skip still shows, and its control stays usable.
        assert re.search(r'data-group-id="%s"' % csv["group_hash"], out[key])
        assert "checked" in out[key]
        # Live: an edit in the page re-renders the tags it invalidates, and
        # undoing it restores the validated tag.
        live = run_page(page, f"""
            init();
            var g = __register(['scope-tags'], {{'data-scope-group': {json.dumps(csv['group_hash'])}}});
            var s = __register(['scope-tags'], {{'data-scope-suggestion': {json.dumps(sid)}}});
            var original = {json.dumps(csv['suggestions'][0]['desc'])};
            var ta = {{ value: 'Rewritten', dataset: {{ original: original }}, classList: new ClassList(),
                       style: {{}}, scrollHeight: 10,
                       getAttribute: function (k) {{ return k === 'data-original' ? original : null; }} }};
            {editor}({json.dumps(sid)}, ta);
            var edited = [g.innerHTML, s.innerHTML];
            ta.value = original;
            {editor}({json.dumps(sid)}, ta);
            __out({{edited: edited, restored: [g.innerHTML, s.innerHTML]}});
        """)
        assert [b["text"] for b in badges_in(live["edited"][0]).badges] == [
            "Scope unclear (may be stale: description edited)", "Deferred: scope preservation requested"]
        assert [b["text"] for b in badges_in(live["edited"][1]).badges] == [
            "Within scope (provisional; may be stale: description edited)"]
        assert [b["text"] for b in badges_in(live["restored"][0]).badges] == ["Within scope (validated)"]

    @pytest.mark.parametrize("style,render", [("flat", FLAT_RENDER), ("pr", PR_RENDER)])
    def test_export_keeps_the_selection_schema_and_drops_scope_fields(self, tmp_path, style, render):
        page, groups, data = _report(tmp_path, style)
        pdf = groups[0]
        store = {_storage_key(data): json.dumps({
            "skipped_groups": {pdf["group_hash"]: True},
            "validation_overrides": {pdf["group_hash"]: "valid"},
            "scope_policy": "expand", "scopePolicy": {"effectivePolicy": "expand"},
            "scope_verdict": {pdf["group_hash"]: {"relation": "in_scope"}}})}
        out = run_page(page, render.replace("__out({", "exportSelections(); __out({blobs: __blobs, "), store)
        exported = json.loads(out["blobs"][-1])
        assert set(exported) == {"plan_path", "phase", "format_version", "exported_at",
                                 "last_file_export_at", "skipped_groups", "skipped_suggestions",
                                 "edited_descriptions", "validation_overrides", "_labels"}
        assert exported["skipped_groups"] == {pdf["group_hash"]: True}
        assert exported["validation_overrides"] == {pdf["group_hash"]: "valid"}
        # A validity override never releases a preserve deferral, and an
        # imported policy-like field never changes a tag.
        key = "groups" if style == "flat" else "pr"
        mine = [b["text"] for b in badges_in(out[key]).badges if b["group"] == pdf["group_hash"]]
        assert mine == ["Scope extension (validated)", "Deferred: scope preservation requested"]
        # The exported file loads back as a selection with the same ids.
        (tmp_path / "sel").mkdir()
        (tmp_path / "sel" / "user_selections.json").write_text(json.dumps(exported), encoding="utf-8")
        loaded = load_html_selections(tmp_path / "sel")
        assert loaded["skipped_groups"] == {pdf["group_hash"]: True}

    def test_pr_umbrella_banner_shows_the_derived_tag(self, tmp_path):
        out_dir = tmp_path / "feat" / "review-plan"
        out_dir.mkdir(parents=True)
        base = baseline()
        groups = plan_groups(base["baseline_id"])
        (out_dir / "consolidated.json").write_text(json.dumps({"consolidated_groups": [{
            "consolidated_id": "a1b2c3d4e5f6", "title": "Output fixes", "description": "d",
            "importance": "HIGH", "is_singleton": False, "display_index": 1,
            "underlying_group_ids": [groups[0]["group_hash"], groups[1]["group_hash"]]}]}),
            encoding="utf-8")
        _path, _g, phase_dir, _plan = plan_review(tmp_path, groups=groups)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        umbrella = data["umbrellaMap"][groups[0]["group_hash"]]["scope"]
        assert umbrella["relation"] == "scope_extension" and umbrella["mixed"]
        page = (phase_dir / "report.html").read_text(encoding="utf-8")
        out = run_page(page, PR_RENDER)
        texts = [b["text"] for b in badges_in(out["pr"]).badges if b["group"] is None and b["suggestion"] is None]
        assert "Scope extension (derived; mixed members)" in texts
        assert "Deferred: scope preservation requested (1 of 2)" in texts


class TestConsolidatedTemplate:
    def _page(self, tmp_path):
        phase_dir, plan, groups, validation, cgs = consolidated(tmp_path)
        path = Path(generate_consolidated_html(cgs, groups, str(phase_dir), str(plan),
                                               ["m:a", "m:b"], validation=validation))
        return path, groups, phase_dir

    RENDER = ("__out({groups: document.getElementById('groups-container').innerHTML, "
              "summary: document.getElementById('scope-summary').innerHTML});")

    def test_umbrella_members_and_decisions_render(self, tmp_path):
        path, groups, phase_dir = self._page(tmp_path)
        hrg.regenerate_consolidated_with_human_decisions(phase_dir, {
            groups[1]["group_hash"]: {"decision": "approved", "reason": "ok"}})
        out = run_page(path.read_text(encoding="utf-8"), self.RENDER)
        found = badges_in(out["groups"])
        umbrella = [b["text"] for b in found.badges if b["group"] is None and b["suggestion"] is None]
        assert umbrella[:2] == ["Scope extension (derived; mixed members)",
                                "Deferred: scope preservation requested (1 of 2)"]
        by_group = {}
        for b in found.badges:
            if b["group"]:
                by_group.setdefault(b["group"], []).append(b["text"])
        assert by_group[groups[0]["group_hash"]] == ["Scope extension (validated)",
                                                     "Deferred: scope preservation requested"]
        assert by_group[groups[1]["group_hash"]] == ["Within scope (validated)"]
        assert len([b for b in found.badges if b["suggestion"]]) == 4
        assert "User: approved" in htmllib.unescape(out["groups"])
        assert "Scope policy: preserve." in out["summary"]

    def test_export_schema_is_unchanged_and_scope_fields_never_load(self, tmp_path):
        path, groups, _pd = self._page(tmp_path)
        page = path.read_text(encoding="utf-8")
        data = json.loads((path.parent / hrg.CONSOLIDATED_REPORT_DATA).read_text(encoding="utf-8"))
        key = "multi-llm-selections-consolidated-" + data["planPath"]
        store = {key: json.dumps({"skipped_groups": {"a1b2c3d4e5f6": True},
                                  "validation_overrides": {}, "scope_policy": "expand"})}
        out = run_page(page, "exportSelections(); __out({blobs: __blobs, selections: selections});", store)
        exported = json.loads(out["blobs"][-1])
        assert set(exported) == {"plan_path", "format_version", "exported_at", "skipped_groups",
                                 "validation_overrides", "_labels"}
        assert exported["skipped_groups"] == {"a1b2c3d4e5f6": True}
        assert "scope_policy" not in out["selections"]


def test_the_three_pages_share_one_scope_block():
    """One block, in the shared script every page inlines: the badge words
    cannot drift between pages, and they are the generator's."""
    core = (SKILL_ROOT / "templates" / "shared" / "report_core.js").read_text(encoding="utf-8")
    assert core.count("BEGIN SCOPE TAGS") == 1 and core.count("// END SCOPE TAGS") == 1
    for label in sp.LABELS.values():
        assert f"'{label}'" in core
    assert f"'{sp.DEFERRED_OUTCOME}'" in core
    for name in ("report_template.html", "pr_report_template.html", "consolidated_report_template.html"):
        text = (SKILL_ROOT / "templates" / name).read_text(encoding="utf-8")
        assert "SCOPE TAGS" not in text and "function scope" not in text, name
        assert text.count("<!--MLLM:report-core-script-->") == 1, name


class TestEveryPrView:
    """The PR page's code, plan and task views each render their group cards
    through the one placeholder; each shows the same tags as the flat view."""

    def test_code_review_view(self, tmp_path):
        _path, _report, groups, _issues, phase_dir = code_review(tmp_path)
        out = run_page((phase_dir / "report.html").read_text(encoding="utf-8"), PR_RENDER)
        for key in ("pr", "flat"):
            by_group: Dict[str, List[str]] = {}
            for b in badges_in(out[key]).badges:
                if b["group"]:
                    by_group.setdefault(b["group"], []).append(b["text"])
            assert by_group[groups[0]["group_hash"]] == ["Within scope (validated)"]
            assert by_group[groups[1]["group_hash"]] == ["Scope extension (validated)",
                                                         "Deferred: scope preservation requested"]
            members = [b["text"] for b in badges_in(out[key]).badges if b["suggestion"]]
            assert sorted(members) == ["Scope extension (provisional)", "Scope extension (provisional)",
                                       "Within scope (provisional)"]

    def test_task_review_view(self, tmp_path):
        page, groups, _data = _report(tmp_path, "pr", phase="review-tasks")
        out = run_page(page, PR_RENDER)
        texts = {b["group"]: b["text"] for b in badges_in(out["pr"]).badges
                 if b["group"] and "validated" in b["text"]}
        assert texts[groups[0]["group_hash"]] == "Scope extension (validated)"
        assert texts[groups[1]["group_hash"]] == "Within scope (validated)"


def test_in_memory_state_takes_precedence_over_the_saved_file(tmp_path):
    """A producer may hand the round it holds in memory to the reports."""
    base = baseline()
    groups = plan_groups(base["baseline_id"])
    plan = tmp_path / "feat.md"
    plan.write_text("# Feat\n", encoding="utf-8")
    phase_dir = tmp_path / "feat" / "review-plan"
    phase_dir.mkdir(parents=True)
    state: Dict[str, Any] = {}
    sp.record_review_round(state, "review-plan", "preserve", "explicit", base)
    path = aggregate_results("feat", str(tmp_path / "feat"), str(phase_dir), "review-plan", ["m:a"],
                             {}, validated_groups=groups, plan_path=str(plan), scope_state=state)
    assert "**Scope policy:** preserve." in Path(path).read_text(encoding="utf-8")
    data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
    assert data["scopePolicy"]["effectivePolicy"] == "preserve"


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def test_scope_badges_have_accessible_contrast_in_both_schemes():
    """The badge colours live once, in the shared style every page inlines."""
    text = (SKILL_ROOT / "templates" / "shared" / "report_core.css").read_text(encoding="utf-8")
    rules = re.findall(r"\.(scope-(?:in|extension|unclear|deferred)) \{ background: (#[0-9a-f]{6}); "
                       r"color: (#[0-9a-f]{6}); \}", text)
    assert len(rules) == 8, rules  # four classes, light and dark
    for cls, bg, fg in rules:
        hi, lo = sorted((_luminance(bg), _luminance(fg)), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= 7.0, (cls, bg, fg)
    assert "text-transform: none" in text  # the label reads exactly as written
