"""Importance reads the same in either case, on both review lanes.

The plan lane stores upper case and the code lane's reviewers answer in lower
case (the schema enum); artifacts from older runs carry either. These tests pin
that one normaliser (``utils/importance.py``) decides the priority score, the
group theme, the code report's sections and the ``--fail-on-high`` gate.
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
from utils import cli_common
from utils import importance
from utils.html_report_generator import compute_max_importance
from utils.state_manager import stamp_stable_ids
from utils.suggestion_processor import SuggestionGroup, group_similar_suggestions


SPELLINGS = [
    pytest.param(("HIGH", "MEDIUM", "LOW"), id="plan-lane-upper"),
    pytest.param(("high", "medium", "low"), id="code-lane-lower"),
    pytest.param(("High", " medium ", "Low"), id="mixed"),
]


class TestNormaliser:
    @pytest.mark.parametrize("value", ["HIGH", "high", "High", " high\n"])
    def test_every_spelling_of_high_is_high(self, value):
        assert importance.canonical(value) == "HIGH"
        assert importance.normalize(value) == "HIGH"
        assert importance.is_high(value)
        assert importance.rank(value) == 0
        assert importance.weight(value) == 3

    @pytest.mark.parametrize("value", [None, "", "critical", 3, ["HIGH"]])
    def test_a_value_naming_no_level_takes_the_default(self, value):
        assert importance.canonical(value) is None
        assert importance.normalize(value) == "MEDIUM"
        assert importance.normalize(value, "LOW") == "LOW"
        assert importance.normalize(value, "") == ""
        assert importance.rank(value, "") == importance.UNKNOWN_RANK
        assert not importance.is_high(value)

    def test_one_order_and_one_score_table(self):
        assert importance.LEVELS == ("HIGH", "MEDIUM", "LOW")
        assert [importance.rank(level) for level in importance.LEVELS] == [0, 1, 2]
        assert [importance.weight(level) for level in importance.LEVELS] == [3, 2, 1]
        assert importance.MODEL_VALUES == ("high", "medium", "low")

    def test_highest_reads_records_and_values_in_either_case(self):
        assert importance.highest([{"importance": "low"}, {"importance": "HIGH"}]) == "HIGH"
        assert importance.highest(["low", "Medium"]) == "MEDIUM"
        assert importance.highest([]) == "MEDIUM"
        assert importance.highest([{"importance": "weird"}], default="") == ""
        assert importance.group_importance(
            {"issues": [{"importance": "low"}, {"importance": "high"}]}
        ) == "HIGH"

    def test_canonicalize_rewrites_in_place_and_keeps_key_order(self):
        records = [
            {"title": "a", "importance": "high", "model": "m"},
            {"title": "b", "importance": "critical"},
            "not a record",
        ]
        importance.canonicalize(records)
        assert records[0] == {"title": "a", "importance": "HIGH", "model": "m"}
        assert list(records[0]) == ["title", "importance", "model"]
        assert records[1]["importance"] == "critical"  # left as written


def _suggestion(title, level, model="model-a", **extra):
    return {
        "title": title,
        "desc": f"{title} description with enough words to tokenize",
        "importance": level,
        "type": "bug",
        "reference": "## Section",
        "source_model": model,
        **extra,
    }


class TestGroupingIgnoresCase:
    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_priority_score_is_the_same_for_the_same_importance(self, spelling):
        high, medium, low = spelling
        for level, expected in ((high, 1 + 3), (medium, 1 + 2), (low, 1 + 1)):
            group = SuggestionGroup("bug", "theme")
            group.add_suggestion({"importance": level}, "model-a")
            assert group.priority_score == expected

    def test_upper_and_lower_case_groups_score_and_hash_alike(self):
        """The same review in either spelling groups to the same theme, the
        same score and the same stable ids: case is not identity."""
        def review(high, low):
            return [
                _suggestion("alpha retry ladder budget", low, model="model-a"),
                _suggestion("zeta retry ladder budget", high, model="model-b"),
            ]

        stamped = []
        for high, low in (("HIGH", "LOW"), ("high", "low")):
            groups = [g.to_dict() for g in group_similar_suggestions(review(high, low))]
            stamp_stable_ids(groups)
            stamped.append(groups)

        upper, lower = stamped
        assert len(upper) == len(lower) == 1
        # The theme comes from the HIGH member, in both spellings.
        assert upper[0]["theme"] == lower[0]["theme"]
        assert upper[0]["theme"].startswith("zeta")
        assert upper[0]["priority_score"] == lower[0]["priority_score"] == 2 + 3 + 1
        assert upper[0]["group_hash"] == lower[0]["group_hash"]
        assert [s["suggestion_hash"] for s in upper[0]["suggestions"]] == [
            s["suggestion_hash"] for s in lower[0]["suggestions"]
        ]


def _issue(title, level, file="src/a.py"):
    return {
        "title": title,
        "desc": f"{title}: the full explanation of the finding",
        "importance": level,
        "file": file,
        "line_range": [1, 2],
        "type": "bug",
    }


def _section(report, heading):
    """The text of one ``## <heading>`` section of a markdown report."""
    start = report.index(f"## {heading}")
    end = report.find("\n## ", start + 1)
    return report[start:end if end != -1 else len(report)]


class TestCodeReportKeepsEveryIssue:
    def test_a_high_issue_in_either_case_is_listed_under_high(self, tmp_path):
        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        results = {
            "model-a": (True, [_issue("Upper case finding", "HIGH")], None),
            "model-b": (True, [_issue("Lower case finding", "high", file="src/b.py")], None),
            "model-c": (True, [_issue("Unknown level finding", "critical", file="src/c.py")], None),
        }

        report = cro.generate_review_report(plan_path, results, changed_files=["src/a.py"])

        assert "**Total Issues:** 3" in report
        high = _section(report, "HIGH Priority")
        assert "Upper case finding" in high
        assert "Lower case finding" in high
        # A level the schema does not name reads as MEDIUM rather than vanishing.
        assert "Unknown level finding" in _section(report, "MEDIUM Priority")


class TestFailOnHighCountsEitherLane:
    @pytest.mark.parametrize("level", ["HIGH", "high"])
    def test_a_high_group_fires_the_gate(self, tmp_path, monkeypatch, level):
        context = cli_common.ResultContext(mode="review-code", phase="code-review")
        monkeypatch.setattr(cli_common, "_ACTIVE", context)
        groups = [g.to_dict() for g in group_similar_suggestions([
            _issue("Retry ladder never stops", level) | {"source_model": "model-a"},
        ])]
        assert compute_max_importance(groups[0]["suggestions"]) == "HIGH"

        args = argparse.Namespace(fail_on_high=True)
        code = cro.finish_code_review(tmp_path, groups, args, total_issues=1)

        assert code == cro.EXIT_NEEDS_HUMAN
        assert context.high_priority_groups == {"count": 1}


class TestGroupImportanceIsOneRule:
    """A group's importance reads the same in its HTML data, in the priority
    sorts and in apply filtering: the highest member, a member naming no level
    counting as MEDIUM; ``""`` (no badge) only for a group with no members."""

    MEMBERS = {
        "low": [{"importance": "LOW"}],
        "junk-only": [{"importance": "junk"}],
        "low-and-junk": [{"importance": "low"}, {"importance": "junk"}],
        "missing": [{"title": "no importance field"}],
        "high-lower-case": [{"importance": "medium"}, {"importance": "high"}],
    }
    EXPECTED = {
        "low": "LOW",
        "junk-only": "MEDIUM",
        "low-and-junk": "MEDIUM",
        "missing": "MEDIUM",
        "high-lower-case": "HIGH",
    }

    def _groups(self):
        return [
            {"theme": name, "validation_status": "valid", "suggestions": [dict(m) for m in members]}
            for name, members in self.MEMBERS.items()
        ]

    def test_html_data_member_data_and_apply_agree(self):
        from utils.filtering import filter_items
        from utils.html_report_generator import _build_group_data

        groups = self._groups()
        kept, _, skipped, _ = filter_items(
            [dict(g) for g in groups], min_priority="medium"
        )
        kept_themes = {g["theme"] for g in kept}
        assert {g["theme"] for g in skipped} == {"low"}

        for index, group in enumerate(groups):
            expected = self.EXPECTED[group["theme"]]
            html = _build_group_data(group, index)
            assert html["maxImportance"] == expected, group["theme"]
            # The badge on the group is the highest of its members' badges.
            assert importance.highest(html["suggestions"]) == expected
            assert importance.get_highest_importance(group) == expected
            assert (group["theme"] in kept_themes) == importance.at_least(expected, "MEDIUM")

    def test_priority_sorts_rank_by_the_same_level(self):
        from utils.html_report_generator import (
            _build_group_data,
            sort_groups_by_priority,
            sort_raw_groups_by_priority,
        )

        groups = self._groups()
        by_apply = sorted(
            range(len(groups)),
            key=lambda i: (importance.rank(self.EXPECTED[groups[i]["theme"]]), i),
        )
        expected_order = [groups[i]["theme"] for i in by_apply]

        html_sorted = sort_groups_by_priority(
            [_build_group_data(g, i) for i, g in enumerate(groups)]
        )
        raw_sorted = sort_raw_groups_by_priority([dict(g) for g in groups])

        assert [g["theme"] for g in html_sorted] == expected_order
        assert [g["theme"] for g in raw_sorted] == expected_order
        # A singleton naming no level is MEDIUM, so it outranks a LOW group.
        assert expected_order.index("junk-only") < expected_order.index("low")

    def test_only_a_group_with_no_members_has_no_level(self):
        assert compute_max_importance([]) == ""
        assert compute_max_importance(None) == ""
        assert importance.group_importance({"suggestions": []}, empty="") == ""
        assert importance.group_importance({"issues": [{"importance": "junk"}]}, empty="") == "MEDIUM"


class TestTemplatesUseTheGeneratorsImportance:
    """The report templates take a group's importance and the rank table from
    the generator's data (``maxImportance``, ``sortConfig``) rather than
    deriving their own."""

    TEMPLATES = ("report_template.html", "pr_report_template.html",
                 "consolidated_report_template.html")

    def _template(self, name):
        """The template with the shared code it inlines (templates/shared/)."""
        from utils.html_report_generator import inline_shared_slots

        root = Path(__file__).parent.parent / "templates"
        return inline_shared_slots((root / name).read_text(encoding="utf-8"))

    @pytest.mark.parametrize("name", TEMPLATES)
    def test_no_template_derives_a_groups_importance(self, name):
        import re

        text = self._template(name)
        # The old pattern: walk the members, promoting a running maximum, or
        # rank with a table of the template's own.
        assert not re.search(r"maxImp(?:ortance)?\s*=\s*'(?:HIGH|high)'", text)
        assert not re.search(r"\bimp\s*===\s*'(?:HIGH|high)'", text)
        assert not re.search(r"impOrder\s*=\s*\{", text)

    @pytest.mark.parametrize("name", TEMPLATES)
    def test_fallback_rank_tables_are_the_generators(self, name):
        import json
        import re

        from utils.html_report_generator import build_sort_config

        config = build_sort_config()
        text = self._template(name)
        literals = re.findall(r"importanceOrder\s*[=:]\s*(\{[^}]*\})", text)
        assert literals, name  # the shared card order's fallback, on every page
        for literal in literals:
            assert json.loads(literal) == config["importanceOrder"]
        for number in re.findall(r"unknownImportanceRank\s*(?:=|:|\|\|)\s*(\d+)", text):
            assert int(number) == config["unknownImportanceRank"]
