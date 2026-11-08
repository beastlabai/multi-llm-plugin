"""The report payload's views: each group stored once, rebuilt by the page.

``utils/report_view.py`` writes the fileView/sectionView/taskView buckets and
``globalSuggestions`` as positions in ``groups``, into ``report.html`` and
``report_data.json`` alike. The PR template's ``hydrateViews`` rebuilds them
before anything renders, and ``report_view.hydrate_views`` is its Python twin.
The Node checks run the template's own function on a generated page; they
skip without ``node``, and the Python checks still run.
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.html_report_generator import (  # noqa: E402
    generate_html_report,
    regenerate_report_with_human_decisions,
)
from utils.report_view import (  # noqa: E402
    GLOBAL_SUGGESTIONS,
    GROUP_INDEXES,
    VIEW_KEYS,
    compact_views,
    hydrate_views,
)

TEMPLATES = Path(__file__).parent.parent / "templates"
NODE = shutil.which("node")

PLAN = """# Plan

Status: draft; the next step is review.

## Overview
Overview text.

## Design
Design text.

### Details
Detail text.

## Testing
Testing text.
"""

HASH_A = "a" * 16


def _group(theme: str, reference: str = "", group_hash: str = "", **extra: Any) -> Dict[str, Any]:
    suggestion = {"title": f"{theme} fix", "desc": f"Fix {theme}.", "importance": "MEDIUM",
                  "type": "modification", "reference": reference, "source_model": "m1"}
    suggestion.update(extra)
    group = {"theme": theme, "models": ["m1"], "validation_status": "valid",
             "suggestions": [suggestion]}
    if group_hash:
        group["group_hash"] = group_hash
    return group


GROUPS = [
    _group("Both sections", "## Overview / ## Design", HASH_A),
    _group("Testing gap", "## Testing", "b" * 16, file="src/app.py", line_range=[3, 4]),
    _group("Cross-cutting", "", "c" * 16),
    _group("Status words", "status", "d" * 16),
]


def _report(tmp_path: Path, style: str = "pr"):
    plan = tmp_path / "plan.md"
    plan.write_text(PLAN, encoding="utf-8")
    phase_dir = tmp_path / "review-plan"
    phase_dir.mkdir()
    html = generate_html_report(
        groups=copy.deepcopy(GROUPS), plan_path=plan, phase_dir=phase_dir,
        phase_type="review-plan", models=["m1"], template_style=style, scope_state={},
    )
    return html, phase_dir


def _embedded(html: str) -> Dict[str, Any]:
    """The page's payload as the generator embedded it (views still indexed)."""
    start = html.index("const reportData = ") + len("const reportData = ")
    return json.JSONDecoder().raw_decode(html, start)[0]


def _themes(bucket: Dict[str, Any]):
    return [group["theme"] for group in bucket["suggestions"]]


# ---------------------------------------------------------------------------
# The stored payload
# ---------------------------------------------------------------------------


class TestStoredOnce:
    def test_sidecar_is_compact_and_views_name_groups_by_index(self, tmp_path):
        html, phase_dir = _report(tmp_path)
        text = (phase_dir / "report_data.json").read_text(encoding="utf-8")
        assert "\n" not in text and '": ' not in text
        data = json.loads(text)
        for name in VIEW_KEYS:
            for key, bucket in data[name].items():
                assert "suggestions" not in bucket, (name, key)
                assert all(type(i) is int for i in bucket[GROUP_INDEXES]), (name, key)
        # The suggestions of the groups with no file: fileView's `_global`.
        no_file = [i for i, g in enumerate(data["groups"]) if g["theme"] != "Testing gap"]
        assert sorted(data["fileView"]["_global"][GROUP_INDEXES]) == no_file
        assert data[GLOBAL_SUGGESTIONS] == {GROUP_INDEXES: data["fileView"]["_global"][GROUP_INDEXES]}
        # Each group's text once, in the page and in the sidecar.
        assert text.count('"theme":"Both sections"') == 1
        assert html.count('"theme": "Both sections"') == 1
        assert _embedded(html) == data

    def test_a_default_run_gains_no_key(self, tmp_path):
        _html, phase_dir = _report(tmp_path)
        data = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        for absent in ("servedModels", "usageTotal", "usageTotalLabel", "commitRange",
                       "requestedModels", "umbrellaMap", "diffData"):
            assert absent not in data

    def test_hydrated_views_hold_the_groups_themselves(self, tmp_path):
        _html, phase_dir = _report(tmp_path)
        data = hydrate_views(json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8")))
        by_theme = {g["theme"]: g for g in data["groups"]}
        section_view = data["sectionView"]
        assert _themes(section_view["## Overview"]) == ["Both sections"]
        assert _themes(section_view["## Design"]) == ["Both sections"]
        assert section_view["## Design"]["suggestions"][0] is by_theme["Both sections"]
        assert _themes(section_view["_global"]) == ["Cross-cutting"]
        assert _themes(data["fileView"]["src/app.py"]) == ["Testing gap"]
        assert [s["title"] for s in data[GLOBAL_SUGGESTIONS]] == [
            s["title"] for g in data["fileView"]["_global"]["suggestions"] for s in g["suggestions"]]
        assert len(data[GLOBAL_SUGGESTIONS]) == 3
        # A plan section nobody cited is still a bucket, empty.
        assert section_view["### Details"]["suggestions"] == []
        assert section_view["### Details"]["hasSuggestions"] is False

    def test_a_reference_naming_no_heading_keeps_its_suggestions(self, tmp_path):
        """``status`` is body text: no full-section text (the base copied the
        plan from that line to the end), but its block still lists the group
        and the hover context still shows the words."""
        html, _phase_dir = _report(tmp_path)
        data = hydrate_views(_embedded(html))
        assert _themes(data["sectionView"]["status"]) == ["Status words"]
        assert "status" not in data["fullSectionContexts"]
        assert "Status: draft" in data["sectionContexts"]["status"]
        assert data["fullSectionContexts"]["## Design"].endswith("Detail text.")

    def test_compact_views_keeps_a_bucket_whose_members_are_copies(self):
        """A payload whose views hold copies (a sidecar read back from JSON)
        has nothing to index, and both functions leave it as it is."""
        group = {"theme": "x", "suggestions": [{"title": "t"}]}
        legacy = {"groups": [group],
                  "sectionView": {"## A": {"suggestions": [copy.deepcopy(group)],
                                           "suggestionCount": 1}},
                  GLOBAL_SUGGESTIONS: [{"title": "t"}]}
        assert compact_views(legacy) == legacy
        assert hydrate_views(copy.deepcopy(legacy)) == legacy

    def test_hydrate_drops_an_index_outside_groups(self):
        data = {"groups": [{"theme": "x", "suggestions": [{"title": "t"}]}],
                "taskView": {"T001": {GROUP_INDEXES: [0, 5, -1, True]}},
                GLOBAL_SUGGESTIONS: {GROUP_INDEXES: [0, 9]}}
        hydrate_views(data)
        assert data["taskView"]["T001"]["suggestions"] == [data["groups"][0]]
        assert data[GLOBAL_SUGGESTIONS] == [{"title": "t"}]


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


class TestRegeneration:
    def test_a_decision_reaches_every_view(self, tmp_path):
        """The overlay stamps ``groups`` only; the views share those objects
        now, so the PR view's section blocks show the badge too (their copies
        never carried it)."""
        _html, phase_dir = _report(tmp_path)
        path, matched = regenerate_report_with_human_decisions(
            phase_dir, {HASH_A: {"decision": "approved", "reason": "ok"}}, scope_state={})
        assert matched == 1
        data = hydrate_views(_embedded(path.read_text(encoding="utf-8")))
        for section in ("## Overview", "## Design"):
            decided = data["sectionView"][section]["suggestions"][0]
            assert decided["humanDecision"]["outcome"] == "approved"
        sidecar = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        assert "suggestions" not in sidecar["sectionView"]["## Design"]

    def test_a_legacy_sidecar_keeps_its_view_copies(self, tmp_path):
        """A sidecar written before the views were indexed still regenerates,
        its view copies untouched and its groups stamped."""
        _html, phase_dir = _report(tmp_path)
        sidecar = phase_dir / "report_data.json"
        legacy = copy.deepcopy(hydrate_views(json.loads(sidecar.read_text(encoding="utf-8"))))
        for name in VIEW_KEYS:
            for bucket in legacy[name].values():
                bucket.pop(GROUP_INDEXES, None)
        sidecar.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        path, matched = regenerate_report_with_human_decisions(
            phase_dir, {HASH_A: {"decision": "skipped"}}, scope_state={})
        assert matched == 1
        data = _embedded(path.read_text(encoding="utf-8"))
        assert _themes(data["sectionView"]["## Design"]) == ["Both sections"]
        assert GROUP_INDEXES not in data["sectionView"]["## Design"]
        stamped = next(g for g in data["groups"] if g.get("groupHash") == HASH_A)
        assert stamped["humanDecision"]["outcome"] == "skipped"


# ---------------------------------------------------------------------------
# The templates
# ---------------------------------------------------------------------------


def _js_function(source: str, name: str) -> str:
    """The source of ``function name(...) {...}`` (braces counted; the
    function read here holds no brace inside a string)."""
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


def _run_node(payload: Dict[str, Any], body: str) -> Any:
    if NODE is None:
        pytest.skip("node is not on PATH; the Python checks above still ran")
    template = (TEMPLATES / "pr_report_template.html").read_text(encoding="utf-8")
    script = (f"var reportData = {json.dumps(payload)};\n"
              + _js_function(template, "hydrateViews") + "\n" + body)
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestTemplates:
    @pytest.mark.parametrize("name", ["report_template.html", "pr_report_template.html",
                                      "consolidated_report_template.html"])
    def test_a_template_that_reads_a_view_hydrates_first(self, name):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        reads = [key for key in VIEW_KEYS + (GLOBAL_SUGGESTIONS,) if f"reportData.{key}" in text]
        if not reads:
            return
        declared = "const reportData = /* REPORT_DATA_PLACEHOLDER */null;\n"
        assert declared + "    hydrateViews(reportData);\n" in text, (name, reads)

    def test_hydrate_views_matches_python(self, tmp_path):
        html, _phase_dir = _report(tmp_path)
        payload = _embedded(html)
        page = _run_node(payload, """
hydrateViews(reportData);
var design = reportData.sectionView['## Design'].suggestions[0];
console.log(JSON.stringify({data: reportData,
  shared: reportData.groups.indexOf(design) >= 0}));
""")
        assert page["shared"] is True
        assert page["data"] == hydrate_views(copy.deepcopy(payload))

    def test_hydrate_views_leaves_a_legacy_payload_and_null_alone(self):
        legacy = {"groups": [{"theme": "x", "suggestions": [{"title": "t"}]}],
                  "sectionView": {"## A": {"suggestions": [{"theme": "copy"}]}},
                  GLOBAL_SUGGESTIONS: [{"title": "t"}]}
        result = _run_node(legacy, """
console.log(JSON.stringify({data: hydrateViews(reportData), nothing: hydrateViews(null)}));
""")
        assert result == {"data": legacy, "nothing": None}

    def test_the_pr_view_renders_each_section_with_its_decided_group(self, tmp_path):
        """In headless Chrome: the PR view's section blocks render from the
        hydrated views (the flat view renders ``groups`` directly, so it
        cannot tell), and the decision badge the overlay stamped on the group
        shows in both blocks it is filed under."""
        from tests.test_html_report_scripts import rendered_dom

        _html, phase_dir = _report(tmp_path)
        path, _matched = regenerate_report_with_human_decisions(
            phase_dir, {HASH_A: {"decision": "approved"}}, scope_state={})
        dom = rendered_dom(path.read_text(encoding="utf-8"), tmp_path)
        pr_view = dom[dom.index('id="pr-content"'):dom.index('id="flat-view"')]
        for section in ("## Overview", "## Design"):
            block = pr_view[pr_view.index(f'id="section-{section}"'):]
            end = block.find('id="section-', 1)
            block = block if end < 0 else block[:end]
            assert f'data-group-id="{HASH_A}"' in block, section
            assert "badge-decision-approved" in block, section


# ---------------------------------------------------------------------------
# The builders
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_report_inputs_mirror_generate_html_report(self):
        """One field per parameter, same names, order and defaults, so a
        caller's keyword reaches the view unchanged."""
        import dataclasses
        import inspect

        from utils.report_view import ReportInputs

        params = inspect.signature(generate_html_report).parameters
        fields = dataclasses.fields(ReportInputs)
        assert [f.name for f in fields] == list(params)
        for field in fields:
            default = params[field.name].default
            expected = dataclasses.MISSING if default is inspect.Parameter.empty else default
            assert field.default == expected, field.name

    def test_model_metadata_names_the_fallback_the_sandbox_and_the_spend(self):
        from utils.html_report_generator import get_model_metadata, sandbox_fallback_label
        from utils.report_view import build_model_metadata

        models = ["p:a", "p:b", "p:c", "p:d"]
        served = {"p:a": {"served_model": "p:z", "fallback": True},
                  "p:b": {"sandbox": "bypassed"},
                  "p:c": {"sandbox": "fallback-plugin-landlock"}}
        usage = {"p:d": {"input_tokens": 10, "output_tokens": 5}}
        metadata = build_model_metadata(models, served, usage)
        assert metadata["p:a"]["model"].endswith("(fallback: p:z)")
        assert metadata["p:a"]["served_model"] == "p:z" and metadata["p:a"]["fallback"] is True
        assert metadata["p:b"]["model"].endswith("(unsandboxed)")
        assert metadata["p:c"]["model"].endswith(
            f"({sandbox_fallback_label('fallback-plugin-landlock')})")
        assert "unsandboxed" not in metadata["p:c"]["model"]
        assert metadata["p:d"]["usage"] == usage["p:d"] and metadata["p:d"]["usage_label"]
        # A default run: exactly the plain badge data.
        assert build_model_metadata(models) == {m: get_model_metadata(m) for m in models}

    @pytest.mark.parametrize("text, expected", [
        ("# Title\n## A\n", "Title"),
        ("intro\n## A\n# Late title\n", "Late title"),
        ("## Only sections\n", "stem"),
        ("intro\n#   Spaced out  \n", "Spaced out"),
        ("", "stem"),
    ])
    def test_plan_title(self, text, expected):
        from utils.report_view import plan_title

        assert plan_title(text, "stem") == expected

    def test_file_contexts_use_only_the_reader_they_are_given(self):
        from utils.report_view import build_file_contexts

        calls = []

        def reader(path, line_range, context_lines=5):
            calls.append((path, list(line_range), context_lines))
            if path == "boom.py":
                raise OSError("unreadable")
            return [{"line_number": line_range[0], "content": path}] if path == "read.py" else None

        def sugg(path, line_range=None, anchor=None):
            return {"fileRef": f"{path}:1", "lineRange": line_range, "anchorText": anchor}

        groups = [{"suggestions": [sugg("diffed.py", [1, 2]), sugg("read.py", [4, 6])]},
                  {"suggestions": [sugg("boom.py", [1, 1], anchor="x = 1"),
                                   sugg("gone.py", [2, 3])]}]
        contexts = build_file_contexts(groups, {"diffed.py": {}}, reader, {"gone.py": "deleted"})
        assert contexts["diffed.py"]["source"] == "diff"
        assert contexts["read.py"] == {"source": "file", "available": True,
                                       "lines": [{"line_number": 4, "content": "read.py"}]}
        assert contexts["boom.py"] == {"source": "anchor", "anchorText": "x = 1", "available": True}
        assert contexts["gone.py"] == {"source": "none", "available": False, "label": "deleted"}
        # The diffed file's outside-the-diff pass asked the reader too, with ±5 lines.
        assert ("diffed.py", [1, 2], 5) in calls

    def test_a_pinned_reader_never_reads_the_disk(self, tmp_path):
        from utils.report_view import pinned_file_reader

        on_disk = tmp_path / "a.py"
        on_disk.write_text("disk\n" * 10, encoding="utf-8")
        assert pinned_file_reader("abc123", {})(str(on_disk), [1, 2]) is None
        assert pinned_file_reader("abc123", None)(str(on_disk), [1, 2]) is None
        held = pinned_file_reader("abc123", {str(on_disk): ["head\n"] * 10})(str(on_disk), [1, 1])
        assert held and {line["content"].strip() for line in held} == {"head"}
        unpinned = pinned_file_reader(None, None)(str(on_disk), [1, 1])
        assert unpinned and unpinned[0]["content"].strip() == "disk"

    def test_merge_task_view_adds_each_task_once_by_bare_id(self):
        from utils.report_view import merge_task_view

        meta = {"title": "One"}
        view = {"T002: x": {"suggestions": [{"theme": "g"}]}, "_coverageGaps": {"suggestions": []}}
        merge_task_view(view, {"T001": meta, "### Task T001: One": meta, "T002": {}})
        assert view["T001"] == {"taskMetadata": meta, "suggestions": [], "suggestionCount": 0,
                                "maxImportance": "", "hasSuggestions": False}
        assert "T002" not in view and view["T002: x"]["hasSuggestions"] is True
        assert "### Task T001: One" not in view and "hasSuggestions" not in view["_coverageGaps"]


# ---------------------------------------------------------------------------
# The consolidated page shares the badges
# ---------------------------------------------------------------------------


class TestConsolidatedBadges:
    SPEC = "codex:gpt-x:high"
    # Two groups merged into one: below a 10% reduction `generate_consolidated_json`
    # deletes the report instead of writing it.
    CGS = [{"consolidated_id": "0123456789ab", "display_index": 1, "title": "Combined",
            "description": "Both.", "importance": "MEDIUM", "reference": "## Design",
            "type": "modification", "is_singleton": False, "model_count": 1,
            "original_suggestion_count": 2, "reasoning": "One finding, twice.",
            "underlying_group_indices": [0, 1], "underlying_group_ids": [HASH_A, "e" * 16]}]

    def _groups(self):
        return [dict(copy.deepcopy(GROUPS[0]), models=[self.SPEC]),
                dict(copy.deepcopy(GROUPS[0]), models=[self.SPEC], group_hash="e" * 16)]

    @staticmethod
    def _payload(html: str) -> Dict[str, Any]:
        start = html.index("var reportData = ") + len("var reportData = ")
        return json.JSONDecoder().raw_decode(html, start)[0]

    def test_the_badges_say_what_report_html_says(self, tmp_path):
        from utils.consolidation import generate_consolidated_html
        from utils.report_view import build_model_metadata

        plan = tmp_path / "plan.md"
        plan.write_text(PLAN, encoding="utf-8")
        served = {self.SPEC: {"served_model": "codex:gpt-y", "fallback": True,
                              "sandbox": "bypassed"}}
        page = Path(generate_consolidated_html(
            self.CGS, self._groups(), str(tmp_path), str(plan), [self.SPEC],
            served_models=served)).read_text(encoding="utf-8")
        data = self._payload(page)
        badge = data["modelMetadata"][self.SPEC]["model"]
        assert badge.endswith("(fallback: codex:gpt-y) (unsandboxed)")
        assert data["modelMetadata"] == build_model_metadata([self.SPEC], served)
        assert data["planTitle"] == "Plan"
        sidecar = (tmp_path / "consolidated_report_data.json").read_text(encoding="utf-8")
        assert "\n" not in sidecar and json.loads(sidecar) == data

    def test_the_consolidation_pass_reads_models_served_off_the_checkpoint(
        self, tmp_path, monkeypatch
    ):
        import review_plan_orchestrator as rpo
        from utils.html_report_generator import sandbox_fallback_label
        from utils.status_file import update_status

        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = tmp_path / "plan.md"
        plan.write_text(PLAN, encoding="utf-8")
        (phase_dir / "grouped.json").write_text(json.dumps(self._groups()), encoding="utf-8")
        (phase_dir / "consolidation_tasks.json").write_text(
            json.dumps({"total_batches": 1, "batches": [], "singleton_sections": {}}),
            encoding="utf-8")
        update_status(str(phase_dir), {
            "state": "models_complete",
            "models_served": {self.SPEC: {"sandbox": "fallback-plugin-landlock"}},
        })
        monkeypatch.setattr(rpo, "merge_consolidation_results",
                            lambda *_a, **_k: ([dict(cg) for cg in self.CGS], {"count": 0}))
        rpo.reaggregate_consolidation_mode(str(phase_dir), str(plan), "plan")
        data = self._payload((phase_dir / "consolidated-report.html").read_text(encoding="utf-8"))
        badge = data["modelMetadata"][self.SPEC]
        assert badge["sandbox"] == "fallback-plugin-landlock"
        assert badge["model"].endswith(f"({sandbox_fallback_label('fallback-plugin-landlock')})")
