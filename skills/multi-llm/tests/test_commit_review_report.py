"""Tests for a commit review's reports (plans/review-commits.md §2.8 "Commit
brief", never "plan", §2.10 "What changes for commits", §4 T4 "Report tests").

``report.md`` gets the commit header and per-finding "Introduced by" lines;
``report.html`` gets the header, the commit filter chips and the "Group by
commit" toggle, the drift badge and an export help naming the apply command
the generator passes in; ``report_data.json`` keeps all of it, so a
regenerated report renders the same without git. A plan's reports are
unchanged. The last class drives the two attribution call sites (the
fan-out's aggregation and ``--reaggregate``) over a real repository.
"""

from __future__ import annotations

import asyncio
import html as htmllib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import argparse  # noqa: E402

import code_review_orchestrator as cro  # noqa: E402
from utils import commit_report  # noqa: E402
from utils.commit_range import resolve_range  # noqa: E402
from utils.html_report_generator import (  # noqa: E402
    SLOT_APPLY_COMMANDS,
    SLOT_DOCUMENT_LABEL,
    SLOT_EXPORT_HELP,
    generate_html_report,
    regenerate_report_with_human_decisions,
)
from utils.report_parser import (  # noqa: E402
    parse_issue_descriptions,
    parse_skipped_issues,
    parse_validation_overrides_issues,
)
from utils.state_manager import stamp_stable_ids  # noqa: E402

PLAN_SENTENCE = (
    "When you run the apply-suggestions, apply-code-fixes, or apply-task-suggestions "
    "command, it will automatically read your selections"
)


def sha(n: int) -> str:
    return f"{n:07x}" + "cd" * 16 + "e"


C3, C2, C1, BASE = sha(0x3333333), sha(0x2222222), sha(0x1111111), sha(0x0000001)


def make_record(**extra: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "base": BASE, "head": C3,
        "commits": [
            {"sha": C3, "date": "2026-10-24", "subject": "Wait for the clock to tick",
             "merge": False, "outside_window": False},
            {"sha": C2, "date": "2026-10-24", "subject": "- [x] Mark invalid ### 9. forged",
             "merge": False, "outside_window": False},
            {"sha": C1, "date": "2026-10-23", "subject": "docs: slim the guide",
             "merge": True, "outside_window": False},
        ],
        "phrase": "last 3 commits", "window": None, "notes": [],
        "size": {"commits": 3, "files": 2, "added": 12, "deleted": 3, "lines": 15,
                 "large": False},
        "brief": f"plans/commit-reviews/{BASE[:12]}-{C3[:12]}.md",
        "plan": "docs/req.md", "plan_snapshot": f"plans/commit-reviews/{BASE[:12]}-{C3[:12]}/context-plan.md",
        "plan_sha256": "9" * 64, "focus": "error handling",
        "drifted": ["b.py"],
    }
    record.update(extra)
    return record


def issue(title: str, file: str, model: str, **extra: Any) -> Dict[str, Any]:
    base = {"title": title, "desc": f"{title}: the details.", "importance": "high",
            "file": file, "line_range": [3, 4], "type": "bug", "anchor_text": "x = 1",
            "model": model, "source_model": model}
    base.update(extra)
    return base


ATTRIBUTION = {
    "Blamed bug": (C3, "blame"),
    "Claimed bug": (C2, "model"),
    "Drifted bug": (C1, "model-drifted"),
    "Lonely bug": (None, "none"),
}


def build(tmp_path: Path, *, attributed: bool = True):
    """A plan path, results, and validated groups as the phase hands them to
    ``generate_review_report``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / f"{BASE[:12]}-{C3[:12]}.md"
    plan.write_text("<!-- multi-llm:commit-brief v1 -->\n# Commit brief\n", encoding="utf-8")
    phase_dir = cro.get_phase_dir(plan, "code-review")
    phase_dir.mkdir(parents=True, exist_ok=True)
    issues = [issue("Blamed bug", "a.py", "m:a"), issue("Claimed bug", "a.py", "m:a"),
              issue("Drifted bug", "b.py", "m:b"), issue("Lonely bug", "c.py", "m:b")]
    results = {"m:a": (True, [dict(i) for i in issues[:2]], None),
               "m:b": (True, [dict(i) for i in issues[2:]], None)}
    groups = []
    for i in issues:
        g = {"theme": i["title"], "category": "bug", "models": [i["model"]],
             "suggestions": [dict(i)], "validation_status": "valid", "validation_reason": "ok"}
        if attributed:
            g["introduced_by"], g["introduced_by_source"] = ATTRIBUTION[i["title"]]
        groups.append(g)
    stamp_stable_ids(groups)
    return plan, phase_dir, results, groups


def report_for(tmp_path: Path, *, style: str = "pr", record: Optional[Dict[str, Any]] = None,
               commit: bool = True):
    plan, phase_dir, results, groups = build(tmp_path, attributed=commit)
    payload = (commit_report.report_payload(record or make_record(), drifted=["a.py"],
                                            scope_total=3) if commit else None)
    report = cro.generate_review_report(plan, results, ["a.py", "b.py", "c.py"], groups,
                                        phase_dir=phase_dir, template_style=style,
                                        commit_review=payload)
    html = (phase_dir / "report.html").read_text(encoding="utf-8")
    data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
    return report, html, data, phase_dir


_DATA = re.compile(r"const reportData = (\{.*?\n\});", re.S)


def _embedded(html: str) -> Dict[str, Any]:
    return json.loads(_DATA.search(html).group(1).replace("<\\/", "</"))


def _without_data(html: str) -> str:
    return _DATA.sub("const reportData = DATA;", html)


def blocks(report: str) -> Dict[str, str]:
    """``{title: issue block}`` of a report.md."""
    out = {}
    for match in re.finditer(r"^### \d+\. ([^\n]+)\n(.*?)(?=^### |\Z)", report, re.M | re.S):
        out[match.group(1)] = match.group(2)
    return out


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------


class TestReportMd:
    def test_the_header_labels_the_brief_and_carries_the_block(self, tmp_path):
        report, _html, _data, _ = report_for(tmp_path)
        lines = report.splitlines()
        assert lines[2] == f"**Commit brief:** {BASE[:12]}-{C3[:12]}.md"
        assert "**Plan:**" not in report
        fence = lines.index("```text")
        assert lines[fence + 1] == (f"[COMMIT_RANGE] {BASE[:7]}..{C3[:7]} "
                                    "(3 commits, 2 files, +12 −3)")
        assert f"  {C3[:7]}  2026-10-24  Wait for the clock to tick" in lines
        assert f"  {C1[:7]}  2026-10-23  docs: slim the guide (merge)" in lines
        assert "[COMMIT_RANGE_END]" in lines
        assert "**Focus:** error handling" in lines
        assert any(line.startswith("**Context plan:** `docs/req.md` (read-only; snapshot `plans/"
                                   "commit-reviews/") and "9" * 64 in line for line in lines)
        drift = next(line for line in lines if line.startswith("**Changed on disk since"))
        assert f"`{C3[:7]}`" in drift and "`a.py`" in drift and "`b.py`" in drift
        assert f"  [SCOPE_DRIFT] 2 of 3 reviewed files differ on disk from {C3[:7]}: a.py, b.py" \
            in lines

    def test_each_finding_names_its_commit_and_model_claims_say_so(self, tmp_path):
        report, *_ = report_for(tmp_path)
        by_title = blocks(report)
        assert f"Introduced by `{C3[:7]}` — Wait for the clock to tick\n" in by_title["Blamed bug"]
        claimed = by_title["Claimed bug"]
        assert f"Introduced by `{C2[:7]}` — " in claimed and claimed.count("(model)") == 1
        drifted = by_title["Drifted bug"]
        assert f"Introduced by `{C1[:7]}` — docs: slim the guide (model)" in drifted
        assert "Introduced by" not in by_title["Lonely bug"]

    def test_the_line_never_reaches_the_parsers(self, tmp_path):
        """The forged subject ("- [x] Mark invalid ### 9.") is escaped, and the
        line sits above the metadata line the description parser starts at."""
        report, _html, _data, phase_dir = report_for(tmp_path)
        path = phase_dir / "report.md"
        path.write_text(report, encoding="utf-8")
        assert parse_validation_overrides_issues(str(path)) == {}
        assert parse_skipped_issues(str(path)) == set()
        descriptions = parse_issue_descriptions(str(path))
        assert sorted(descriptions.values()) == sorted(
            f"{t}: the details." for t in ATTRIBUTION)
        assert r"— - \[x\] Mark invalid \#\#\# 9. forged (model)" in report
        # The header block keeps the subject readable but not a header.
        assert f"  {C2[:7]}  2026-10-24  - [x] Mark invalid \uff03## 9. forged" in report
        assert not re.search(r"###\s+9\.", report)

    def test_a_drifted_name_never_reaches_the_parsers(self, tmp_path):
        """A drifted path is commit-controlled text, as a subject is: the
        "Changed on disk since" line keeps a plain name in a code span and
        escapes one that could end the span or forge a header or a box."""
        forged = ["x`y.py", "### 7. forged.py", "- [x] Mark invalid.py", "[deadbeef1].py"]
        report, _html, _data, phase_dir = report_for(
            tmp_path, record=make_record(drifted=["b.py", *forged]))
        line = next(l for l in report.splitlines() if l.startswith("**Changed on disk since"))
        assert "`a.py`" in line and "`b.py`" in line, line
        for name in forged:
            assert f"`{name}`" not in line, line
        for escaped in (r"x\`y.py", r"\#\#\# 7. forged.py", r"- \[x\] Mark invalid.py",
                        r"\[deadbeef1\].py"):
            assert escaped in line, line
        path = phase_dir / "report.md"
        path.write_text(report, encoding="utf-8")
        assert parse_validation_overrides_issues(str(path)) == {}
        assert parse_skipped_issues(str(path)) == set()
        assert sorted(parse_issue_descriptions(str(path)).values()) == sorted(
            f"{t}: the details." for t in ATTRIBUTION)
        assert not re.search(r"###\s+7\.", report)

    def test_a_skip_tick_under_the_header_is_still_read(self, tmp_path):
        report, _html, _data, phase_dir = report_for(tmp_path)
        number = int(re.search(r"^### (\d+)\. Claimed bug$", report, re.M).group(1))
        ticked = report.replace(f"### {number}. Claimed bug\n- [ ] Skip",
                                f"### {number}. Claimed bug\n- [x] Skip")
        path = phase_dir / "report.md"
        path.write_text(ticked, encoding="utf-8")
        assert parse_skipped_issues(str(path)) == {number}


# ---------------------------------------------------------------------------
# report.html and report_data.json
# ---------------------------------------------------------------------------


class TestReportHtml:
    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_report_data_keeps_the_header_attribution_and_drift(self, tmp_path, style):
        _report, _html, data, _ = report_for(tmp_path, style=style)
        cr = data["commitRange"]
        assert cr["documentLabel"] == "Commit brief"
        assert cr["blockLines"][0].startswith(f"[COMMIT_RANGE] {BASE[:7]}..{C3[:7]}")
        assert [c["sha"] for c in cr["commits"]] == [C3, C2, C1]
        assert cr["drifted"] == ["a.py", "b.py"]
        assert cr["applyCommands"] == [f"--apply-commit-fixes {BASE[:12]}..{C3[:12]}"]
        assert cr["focus"] == "error handling" and cr["plan"] == "docs/req.md"
        by_theme = {g["theme"]: g for g in data["groups"]}
        assert cr["attribution"][by_theme["Claimed bug"]["groupHash"]] == {
            "introducedBy": C2, "source": "model"}
        assert by_theme["Blamed bug"]["introducedByShort"] == C3[:7]
        assert by_theme["Blamed bug"]["introducedBySubject"] == "Wait for the clock to tick"
        assert by_theme["Lonely bug"]["introducedBy"] is None
        assert "introducedByShort" not in by_theme["Lonely bug"]

    def test_the_flat_report_has_the_header_filter_toggle_badge_and_help(self, tmp_path):
        _report, html, _data, _ = report_for(tmp_path, style="flat")
        assert "<strong>Commit brief:</strong>" in html
        assert "<strong>Plan:</strong>" not in html
        assert (f"When you run the --apply-commit-fixes {BASE[:12]}..{C3[:12]} command, "
                "it will automatically read your selections") in html
        assert PLAN_SENTENCE not in html
        for needle in ('id="commit-range-header"', 'id="commit-filter-group"',
                       'id="commit-filters"', 'id="group-by-commit"', ">Group by commit<",
                       "function renderCommitFilter()", "function renderCommitHeader()",
                       "data-commit=", "unattributed (", "changed on disk since ",
                       "renderCommitHeader();", "renderCommitFilter();"):
            assert needle in html, needle

    def test_the_pr_report_has_the_header_filter_toggle_badge_and_help(self, tmp_path):
        _report, html, _data, _ = report_for(tmp_path, style="pr")
        assert ('<span class="export-help" id="export-help">Export saves '
                '<code>user_selections.json</code>: put it next to <code>report.md</code>, '
                f'then run <code>--apply-commit-fixes {BASE[:12]}..{C3[:12]}</code>') in html
        for needle in ('id="commit-range-header"', 'id="commit-filter-bar"',
                       "Group by commit", "function applyCommitFilter()",
                       "function flatGroupOrder(groups)", "driftBadgeHtml([filePath])",
                       "introducedByHtml(group)", "applyCommitFilter();"):
            assert needle in html, needle

    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_no_slot_marker_is_left_in_either_template(self, tmp_path, style):
        for commit in (True, False):
            _report, html, _data, _ = report_for(tmp_path / f"{style}{commit}", style=style,
                                                 commit=commit)
            for slot in (SLOT_DOCUMENT_LABEL, SLOT_APPLY_COMMANDS, SLOT_EXPORT_HELP):
                assert slot not in html

    def test_a_root_range_names_the_brief_as_the_apply_target(self, tmp_path):
        record = make_record(base=None)
        _report, html, data, _ = report_for(tmp_path, style="flat", record=record)
        assert data["commitRange"]["applyCommands"] == [
            f"--apply-commit-fixes {record['brief']}"]
        assert f"When you run the --apply-commit-fixes {record['brief']} command" in html

    @pytest.mark.parametrize("style", ["pr", "flat"])
    @pytest.mark.parametrize("folder", ["review notes", "reviewer's notes"])
    def test_a_root_apply_command_quotes_the_brief_path(self, tmp_path, style, folder):
        record = make_record(base=None)
        record["brief"] = f"{folder}/root-{record['head'][:12]}.md"
        _report, html, data, _ = report_for(tmp_path, style=style, record=record)
        command, = data["commitRange"]["applyCommands"]
        assert shlex.split(command) == ["--apply-commit-fixes", record["brief"]]
        assert command in htmllib.unescape(html)

    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_a_regenerated_report_renders_the_same_without_git(self, tmp_path, monkeypatch,
                                                                style):
        plan, phase_dir, _results, groups = build(tmp_path)
        payload = commit_report.report_payload(make_record(), drifted=["a.py"], scope_total=3)
        html = generate_html_report(groups=groups, plan_path=plan, phase_dir=phase_dir,
                                    phase_type="code-review", models=["m:a", "m:b"],
                                    template_style=style, commit_review=payload)
        before = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))

        def no_spawn(*args, **kwargs):
            raise AssertionError(f"a regenerated report spawned {args!r}")

        monkeypatch.setattr(subprocess, "run", no_spawn)
        monkeypatch.setattr(subprocess, "Popen", no_spawn)
        path, matched = regenerate_report_with_human_decisions(phase_dir, {})
        assert matched == 0
        regenerated = path.read_text(encoding="utf-8")
        # The page around the data is the same page: label, help, filter, toggle.
        assert _without_data(regenerated) == _without_data(html)
        after = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert after["commitRange"] == before["commitRange"]
        assert _embedded(regenerated)["commitRange"] == _embedded(html)["commitRange"]
        assert [g.get("introducedBySubject") for g in after["groups"]] == \
            [g.get("introducedBySubject") for g in before["groups"]]
        label = "<strong>Commit brief:</strong>" if style == "flat" else 'id="export-help"'
        assert label in regenerated


# ---------------------------------------------------------------------------
# Commit-controlled text in attributes, and the absent-at-head label
# ---------------------------------------------------------------------------

#: A file name a commit controls: both quotes, markup and an ampersand.
HOSTILE = "src/a\"b'c<d>&e.py"
NODE = shutil.which("node")

#: Enough of ``document`` for both templates' ``escapeHtml`` (``textContent``
#: then ``innerHTML``, or ``appendChild(createTextNode(...))``): text
#: serialised as a browser does, escaping ``&``, ``<`` and ``>`` only.
DOM_SHIM = r"""
var document = {
  createElement: function () {
    var text = '';
    return {
      set textContent(v) { text = String(v); },
      appendChild: function (node) { text += node.data; },
      get innerHTML() {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      }
    };
  },
  createTextNode: function (v) { return { data: String(v) }; }
};
"""

#: An ``escapeHtml(`` call whose output lands inside an attribute value: the
#: string literal concatenated just before it leaves an attribute open.
_LITERAL_BEFORE = re.compile(r"(['\"])((?:\\.|(?!\1).)*)\1\s*\+\s*$", re.S)


def _escape_html_in_attributes(source: str):
    found = []
    for match in re.finditer(r"\bescapeHtml\(", source):
        before = _LITERAL_BEFORE.search(source[max(0, match.start() - 400):match.start()])
        if not before:
            continue
        literal = before.group(2)
        tail = literal[literal.rfind(">") + 1:]
        if re.search(r'=\s*"[^"]*$', tail) or re.search(r"=\s*\\?'[^']*$", tail):
            found.append((source.count("\n", 0, match.start()) + 1, tail[-60:]))
    return found


def _js_function(source: str, name: str) -> str:
    """The source of ``function name(...) {...}`` (braces counted; the
    functions read here hold no brace inside a string)."""
    start = source.index(f"function {name}(")
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced function {name}")


def _run_node(html: str, names, body: str) -> Any:
    """Evaluate *body* (which ``console.log``s one JSON value) with the named
    template functions, ``reportData`` from the page and the DOM shim."""
    if NODE is None:
        pytest.skip("node is not on PATH; the static checks above still ran")
    script = (DOM_SHIM + f"var reportData = {json.dumps(_embedded(html))};\n"
              + "\n".join(_js_function(html, name) for name in names) + "\n" + body)
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8",
                            timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def hostile_report(tmp_path: Path, style: str, *, labels: Optional[Dict[str, str]] = None):
    plan, phase_dir, _results, _groups = build(tmp_path)
    groups = [{"theme": "Hostile bug", "category": "bug", "models": ["m:a"],
               "suggestions": [issue("Hostile bug", HOSTILE, "m:a")],
               "validation_status": "valid", "validation_reason": "ok"},
              {"theme": "Gone bug", "category": "bug", "models": ["m:a"],
               "suggestions": [issue("Gone bug", "old.py", "m:a")],
               "validation_status": "valid", "validation_reason": "ok"}]
    stamp_stable_ids(groups)
    html = generate_html_report(groups=groups, plan_path=plan, phase_dir=phase_dir,
                                phase_type="code-review", models=["m:a"],
                                template_style=style, file_labels=labels,
                                commit_review=commit_report.report_payload(
                                    make_record(), scope=["a.py"]))
    return html, groups, phase_dir


class TestTheBlocksDriftLine:
    """The reports' block is the block the front door printed (§2.10): its
    ``[SCOPE_DRIFT]`` line names reviewed files only, while the badges and the
    "Changed on disk since" line take every file the guard will skip."""

    def test_it_names_reviewed_files_and_the_badges_take_the_rest(self):
        payload = commit_report.report_payload(
            make_record(drifted=["b.py"]), drifted=["a.py", "outside.py"],
            scope=["a.py", "b.py", "c.py"])
        line = [l for l in payload["blockLines"] if "[SCOPE_DRIFT]" in l]
        assert line == [f"  [SCOPE_DRIFT] 2 of 3 reviewed files differ on disk from "
                        f"{C3[:7]}: a.py, b.py"]
        assert payload["drifted"] == ["a.py", "b.py", "outside.py"]

    def test_a_finding_outside_the_scope_adds_no_line(self):
        payload = commit_report.report_payload(
            make_record(drifted=[]), drifted=["outside.py"], scope=["a.py"])
        assert not [l for l in payload["blockLines"] if "[SCOPE_DRIFT]" in l]
        assert payload["drifted"] == ["outside.py"]

    def test_report_context_passes_the_scope(self):
        attribution = MagicMock(drifted=("outside.py",))
        payload = commit_report.report_context(
            {"commit_range": make_record(drifted=[])}, attribution, ["a.py"])
        assert not [l for l in payload["blockLines"] if "[SCOPE_DRIFT]" in l]
        assert payload["drifted"] == ["outside.py"]


class TestAttributesAreEscapedAsAttributes:
    """A file name a commit controls reaches HTML attributes (``title``,
    ``data-target``, ``id``, inline handlers): ``escapeHtml`` leaves quotes
    alone, so every such site takes ``escapeAttr``, and a value inside an
    inline handler's JS string takes ``escapeJsAttr``."""

    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_no_attribute_takes_the_text_escaper(self, tmp_path, style):
        html, _groups, _ = hostile_report(tmp_path, style)
        data = _embedded(html)
        refs = [s["fileRef"] for g in data["groups"] for s in g["suggestions"]]
        assert any(ref.startswith(HOSTILE) for ref in refs), refs
        assert _escape_html_in_attributes(html) == []
        assert "function escapeAttr(" in html
        if style == "pr":
            for site in (
                "title=\"' +\n          escapeAttr((guard.files || []).join(', '))",
                "' title=\"' + escapeAttr(info.reason) + '\">",
                "data-target=\"file-' + escapeAttr(filePath) + '\"",
                "onclick=\"scrollToBlock(\\'file-' + escapeJsAttr(filePath) + '\\')\"",
                "id=\"file-' + escapeAttr(filePath) + '\"",
                "onclick=\"batchSkipInBlock(\\'' + escapeJsAttr(blockContext) + '\\')\"",
            ):
                assert site in html, site

    def test_the_guard_badge_title_holds_the_name_whole(self, tmp_path):
        html, groups, phase_dir = hostile_report(tmp_path, "pr")
        gid = groups[0]["group_hash"]
        path, _ = regenerate_report_with_human_decisions(phase_dir, {}, guard={
            "head": C3, "skipped": {gid: {"reason": "changed", "files": [HOSTILE, "b.py"]}}})
        html = path.read_text(encoding="utf-8")
        badge = _run_node(
            html, ["escapeHtml", "escapeAttr", "renderHumanDecisionBadge"],
            "var g = reportData.groups.filter(function (x) { return x.guardSkip; })[0];\n"
            "console.log(JSON.stringify(renderHumanDecisionBadge(g)));")
        match = re.fullmatch(r'<span class="badge badge-judge-decision badge-decision-skipped" '
                             r'title="([^"<>]*)">⚙ ([^<"]*)</span>', badge)
        assert match, badge
        assert htmllib.unescape(match.group(1)) == f"{HOSTILE}, b.py"
        assert match.group(2) == f"skipped: file changed since {C3[:7]}"

    def test_an_inline_handler_gets_the_name_back_whole(self, tmp_path):
        html, _groups, _ = hostile_report(tmp_path, "pr")
        attribute = _run_node(
            html, ["escapeHtml", "escapeAttr", "escapeJsAttr"],
            f"console.log(JSON.stringify(\"scrollToBlock('file-\" + "
            f"escapeJsAttr({json.dumps(HOSTILE + chr(10) + chr(92))}) + \"')\"));")
        assert '"' not in attribute and "<" not in attribute
        # The browser decodes the attribute, then runs it as JS.
        handler = htmllib.unescape(attribute)
        called = _run_node(
            html, [],
            "var seen = null; function scrollToBlock(id) { seen = id; }\n"
            f"{handler};\nconsole.log(JSON.stringify(seen));")
        assert called == "file-" + HOSTILE + "\n\\"


class TestTheAbsentAtHeadLabel:
    """§2.5 "The file's header carries the same label as the excerpt"."""

    LABEL = "renamed to new\"<i>.py in 0000001..3333333"

    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_the_header_renders_the_stored_label(self, tmp_path, style):
        html, _groups, _ = hostile_report(tmp_path, style, labels={"old.py": self.LABEL})
        assert _embedded(html)["fileContexts"]["old.py"]["label"] == self.LABEL
        if style == "pr":
            assert ("'<span class=\"block-title mono\">' + escapeHtml(filePath) + "
                    "fileLabelHtml(filePath) + '</span>'") in html
        else:
            assert ("'<span class=\"file-ref\">File: <code>' + escapeHtml(fileRef) + "
                    "'</code>' + fileLabelHtml(fileRef) + '</span>'") in html
        rendered = _run_node(
            html, ["escapeHtml", "fileLabelHtml"],
            "console.log(JSON.stringify([fileLabelHtml('old.py'), fileLabelHtml('old.py:3-4'),"
            " fileLabelHtml(" + json.dumps(HOSTILE) + ")]));")
        label = ' <span class="file-absent-label">(renamed to new"&lt;i&gt;.py in 0000001..3333333)</span>'
        assert rendered == [label, label, ""]

    def test_a_plan_report_has_no_label(self, tmp_path):
        plan, phase_dir, _results, groups = build(tmp_path, attributed=False)
        html = generate_html_report(groups=groups, plan_path=plan, phase_dir=phase_dir,
                                    phase_type="code-review", models=["m:a"])
        contexts = _embedded(html).get("fileContexts") or {}
        assert not [path for path, entry in contexts.items() if "label" in entry]


# ---------------------------------------------------------------------------
# A plan's reports are unchanged
# ---------------------------------------------------------------------------


class TestPlanReportsAreUnchanged:
    def test_report_md_has_no_commit_line(self, tmp_path):
        report, _html, _data, _ = report_for(tmp_path, commit=False)
        lines = report.splitlines()
        assert lines[:3] == ["# Code Review Report", "", f"**Plan:** {BASE[:12]}-{C3[:12]}.md"]
        assert lines[3].startswith("**Generated:** ")
        assert lines[4:6] == ["**Files Changed:** 3", ""]
        assert lines[6].startswith("**Models:** ")
        for needle in ("Commit brief", "Introduced by", "**Commits:**", "[COMMIT_RANGE]",
                       "Changed on disk"):
            assert needle not in report

    @pytest.mark.parametrize("style", ["pr", "flat"])
    def test_report_data_gains_no_key(self, tmp_path, style):
        _report, html, data, _ = report_for(tmp_path, style=style, commit=False)
        assert "commitRange" not in data
        for group in data["groups"]:
            assert not [key for key in group if key.startswith("introducedBy")]
        if style == "flat":
            assert "<strong>Plan:</strong>" in html
            assert PLAN_SENTENCE in html
        else:
            assert '<span class="export-help"' not in html
            assert ('<span class="export-indicator" id="export-indicator" style="display:none">'
                    '</span>\n') in html

    def test_attribution_fields_without_a_commit_payload_render_no_line(self, tmp_path):
        """A group that carries the fields (a hand-copied grouped.json) in a
        plan's report still gets no "Introduced by" line in report.md."""
        plan, phase_dir, results, groups = build(tmp_path, attributed=True)
        report = cro.generate_review_report(plan, results, ["a.py"], groups,
                                            phase_dir=phase_dir)
        assert "Introduced by" not in report and "**Plan:**" in report


# ---------------------------------------------------------------------------
# The call sites: after aggregation, and on --reaggregate
# ---------------------------------------------------------------------------


@pytest.fixture
def git_env(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.invalid"),
                        ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.invalid")):
        monkeypatch.setenv(name, value)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def commit_repo(tmp_path, git_env, monkeypatch):
    root = Path(os.path.realpath(tmp_path)) / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("".join(f"line {i}\n" for i in range(1, 11)), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "a.py").write_text("".join(("three\n" if i == 3 else f"line {i}\n")
                                       for i in range(1, 11)), encoding="utf-8")
    _git(root, "commit", "-q", "-am", "c1: rewrite line three")
    head = _git(root, "rev-parse", "HEAD")
    record = resolve_range(f"{base}..{head}", cwd=str(root)).to_record()
    record.update({"drifted": [], "brief": "plans/commit-reviews/x.md", "focus": None,
                   "plan": None, "plan_snapshot": None, "plan_sha256": None,
                   "size": {"commits": 1, "files": 1, "added": 1, "deleted": 1, "lines": 2,
                            "large": False}})
    plan = root / "plans" / "commit-reviews" / f"{base[:12]}-{head[:12]}.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("<!-- multi-llm:commit-brief v1 -->\n# Brief\n", encoding="utf-8")
    (plan.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root, base, head, record, plan


def _fake_state(record: Dict[str, Any], base: str) -> MagicMock:
    values = {"tracked_files": [{"path": "a.py"}], "head_before_implement": base,
              "head_at_start": base, "pre_existing_changes": [], "commit_range": record}
    state = MagicMock()
    state.get.side_effect = lambda key, default=None: values.get(key, default)
    state.is_phase_completed.return_value = False
    state.state = {"phases_completed": {}}
    return state


def _args(plan: Path, **overrides: Any) -> argparse.Namespace:
    values = dict(plan_file=str(plan), models=["cursor-agent:auto"], interactive=False,
                  quick=False, timeout=None, max_parallel=2, base_ref=None,
                  skip_validation=True, validation_model=None, apply_fixes=False, force=True,
                  rerun_all=False, reaggregate=False, report_style="pr", files=[],
                  files_from=None)
    values.update(overrides)
    return argparse.Namespace(**values)


FINDING = {"title": "Line three is wrong", "desc": "It should stay line 3.", "importance": "high",
           "file": "a.py", "line_range": [3, 3], "type": "bug", "anchor_text": "three",
           "introduced_by": "deadbee"}


class TestTheCallSites:
    @pytest.mark.parametrize("reaggregate", [False, True])
    def test_groups_are_attributed_and_the_report_carries_it(self, commit_repo, reaggregate):
        root, base, head, record, plan = commit_repo
        phase_dir = cro.get_phase_dir(plan, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "cursor-agent_auto.json").write_text(json.dumps([FINDING]),
                                                          encoding="utf-8")
        args = _args(plan, reaggregate=reaggregate)
        state = _fake_state(record, base)
        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=state), \
             patch.object(cro, "resolve_models", return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda ref: ref or "HEAD~1"), \
             patch.object(cro, "build_unsanitize_map",
                          return_value={"cursor-agent_auto": "cursor-agent:auto"}), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(return_value={
                 "cursor-agent:auto": (True, [dict(FINDING)], None)})):
            if reaggregate:
                asyncio.run(cro.reaggregate_from_existing_files(
                    "x", phase_dir.parent, plan, args))
            else:
                assert asyncio.run(cro.main()) == 0

        grouped = cro.load_groups_payload(json.loads((phase_dir / "grouped.json").read_text()))
        assert [(g["introduced_by"], g["introduced_by_source"]) for g in grouped] == [
            (head, "blame")]
        # The model's own (refused) claim stays on its suggestion, untouched.
        assert grouped[0]["suggestions"][0]["introduced_by"] == "deadbee"
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert f"**Commit brief:** {plan.name}" in report
        assert f"[COMMIT_RANGE] {base[:7]}..{head[:7]} (1 commit, 1 file, +1 −1)" in report
        if not reaggregate:
            # The fan-out's --skip-validation report keeps its groups.
            assert f"Introduced by `{head[:7]}` — c1: rewrite line three" in report
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert data["commitRange"]["head"] == head
        # A --skip-validation reaggregate reports no groups (as for any plan),
        # so only the fan-out's page carries a group to attribute.
        assert list(data["commitRange"]["attribution"].values()) == (
            [] if reaggregate else [{"introducedBy": head, "source": "blame"}])

    def test_a_plan_run_is_left_unattributed(self, commit_repo):
        root, base, head, record, plan = commit_repo
        phase_dir = cro.get_phase_dir(plan, "code-review")
        args = _args(plan)
        state = _fake_state(record, base)
        state.get.side_effect = lambda key, default=None: {
            "tracked_files": [{"path": "a.py"}], "head_before_implement": base,
            "pre_existing_changes": []}.get(key, default)
        with patch.object(cro, "parse_args", return_value=args), \
             patch.object(cro, "get_or_create_state", return_value=state), \
             patch.object(cro, "resolve_models", return_value=["cursor-agent:auto"]), \
             patch.object(cro, "is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda ref: ref or "HEAD~1"), \
             patch.object(cro, "run_all_reviews", new=AsyncMock(return_value={
                 "cursor-agent:auto": (True, [dict(FINDING)], None)})):
            assert asyncio.run(cro.main()) == 0
        grouped = cro.load_groups_payload(json.loads((phase_dir / "grouped.json").read_text()))
        assert "introduced_by" not in grouped[0]
        report = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert "**Plan:**" in report and "Introduced by" not in report
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert "commitRange" not in data


class TestSalvageKeepsTheClaim:
    """§2.6: a commit review's reviewers answer ``introduced_by``. A salvage
    request names the fields its subagent rebuilds the answer with, so a
    commit review's names it too; a plan's run keeps the fields it had."""

    RAW = '[{"title": "t", "desc": "d", "importance": "high", "file": "a.py", ' \
          '"line_range": [3, 3], "type": "bug", "anchor_text": "three", ' \
          '"introduced_by": "abc1234"'

    def _salvage(self, plan: Path, monkeypatch, outcome: Dict[str, Any]) -> Dict[str, Any]:
        monkeypatch.setattr(cro, "invoke_with_file_output", lambda **_kw: dict(outcome))
        out_dir = cro.get_phase_dir(plan, "code-review").parent
        success, _issues, _error = asyncio.run(cro.run_single_review(
            "cursor-agent:auto", plan, out_dir))
        assert success is False
        request = cro.get_phase_dir(plan, "code-review") / "salvage_cursor-agent_auto.json"
        return json.loads(request.read_text(encoding="utf-8"))

    OUTCOMES = {
        "unparsed answer": {"success": True, "data": {"raw": RAW}, "source": "stdout"},
        "failed slot": {"success": False, "error": "killed",
                        "details": {"stdout_excerpt": RAW}},
    }

    @pytest.mark.parametrize("path", sorted(OUTCOMES))
    def test_a_commit_reviews_request_names_introduced_by(self, commit_repo, monkeypatch,
                                                           path):
        root, base, head, record, plan = commit_repo
        out = cro.get_phase_dir(plan, "code-review")
        out.mkdir(parents=True, exist_ok=True)
        (out.parent / "state.json").write_text(json.dumps({"commit_range": record}),
                                                encoding="utf-8")
        request = self._salvage(plan, monkeypatch, self.OUTCOMES[path])
        assert request["expected_schema"]["fields"] == [
            "title", "desc", "importance", "file", "line_range", "type", "anchor_text",
            "introduced_by"]

    @pytest.mark.parametrize("path", sorted(OUTCOMES))
    def test_a_plan_runs_request_is_unchanged(self, commit_repo, monkeypatch, path):
        root, base, head, record, _brief = commit_repo
        plan = root / "plans" / "feature.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        cro.get_phase_dir(plan, "code-review").mkdir(parents=True, exist_ok=True)
        request = self._salvage(plan, monkeypatch, self.OUTCOMES[path])
        assert request["expected_schema"]["fields"] == [
            "title", "desc", "importance", "file", "line_range", "type", "anchor_text"]
