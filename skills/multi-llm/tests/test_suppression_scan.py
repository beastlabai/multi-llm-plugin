"""The suppression scan: what a code batch added that quiets a check.

``utils.suppression_scan`` is pure: each case hands it a file's text before
and after a batch and reads the hits. The Python detection is the one a
seventeen-run experiment validated (every suppression listed, no false
positive, every reason tag as a person graded it); these cases pin its rules
on small fixtures: real comments only, the marker at a segment's start,
netting so a moved line is not reported, skips with their reasons, and
configuration compared key by key. The C-like markers and JavaScript skips
are covered here only, so their decoys (a marker inside a string, prose that
names one) each have a case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import suppression_scan as scan  # noqa: E402


def _one(path, before, after):
    return scan.scan_changes([(path, before, after)])


def _kinds(hits):
    return [(hit["path"], hit["line"], hit["kind"]) for hit in hits]


# ---------------------------------------------------------------------------
# Python comment markers
# ---------------------------------------------------------------------------

class TestPythonCommentMarkers:

    def test_an_added_noqa_without_a_reason_is_listed_as_reasonless(self):
        before = "def f(a, b, c, d, e):\n    return a\n"
        after = "def f(a, b, c, d, e, g):  # noqa: PLR0913\n    return a\n"
        hits = _one("src/m.py", before, after)
        assert hits == [{
            "path": "src/m.py", "line": 1, "kind": "comment-marker:noqa",
            "marker": "# noqa: PLR0913",
            "text": "def f(a, b, c, d, e, g):  # noqa: PLR0913", "reason": None,
        }]

    def test_the_rest_of_the_comment_is_its_reason(self):
        after = "try:\n    run()\nexcept Exception:  # noqa: BLE001 - handlers must not stop delivery\n    pass\n"
        hits = _one("src/m.py", "try:\n    run()\nexcept ValueError:\n    pass\n", after)
        assert [hit["reason"] for hit in hits] == ["handlers must not stop delivery"]

    def test_a_comment_line_directly_above_is_its_reason(self):
        after = "# The plan fixes six arguments.\ndef f(a, b, c, d, e, g):  # noqa: PLR0913\n    pass\n"
        hits = _one("src/m.py", None, after)
        assert hits[0]["reason"] == "above: The plan fixes six arguments."
        assert hits[0]["line"] == 2

    def test_another_hash_segment_is_a_reason(self):
        after = "x = f(y)  # type: ignore[arg-type]  # the stub is wrong upstream\n"
        hits = _one("src/m.py", "x = f(y)\n", after)
        assert hits[0]["marker"] == "# type: ignore[arg-type]"
        assert hits[0]["reason"] == "the stub is wrong upstream"

    @pytest.mark.parametrize("comment, kind", [
        ("# noqa", "comment-marker:noqa"),
        ("# ruff: noqa", "comment-marker:ruff-noqa-file"),
        ("# type: ignore", "comment-marker:type-ignore"),
        ("# pyright: ignore[reportGeneralTypeIssues]", "comment-marker:pyright-ignore"),
        ("# pragma: no cover", "comment-marker:pragma-no-cover"),
        ("# pylint: disable=too-many-arguments", "comment-marker:pylint-disable"),
        ("# fmt: off", "comment-marker:fmt-off"),
        ("# nosec", "comment-marker:nosec"),
    ])
    def test_each_marker_kind(self, comment, kind):
        hits = _one("src/m.py", "x = 1\n", f"x = 1  {comment}\n")
        assert [hit["kind"] for hit in hits] == [kind]

    def test_a_marker_inside_a_string_is_not_a_comment(self):
        after = 'HELP = "add # noqa to silence it"\nOTHER = """\n# type: ignore\n"""\n'
        assert _one("src/m.py", None, after) == []

    def test_prose_that_names_a_marker_is_not_one(self):
        after = "x = 1  # we never add a noqa here\n# see the type: ignore docs\n"
        assert _one("src/m.py", "x = 1\n", after) == []

    def test_a_moved_line_is_netted_out(self):
        before = "import os  # noqa: F401\n\nx = 1\n"
        after = "x = 1\n\n\nimport os  # noqa: F401\n"
        assert _one("src/m.py", before, after) == []

    def test_a_line_moved_to_another_file_of_the_batch_is_netted_out(self):
        hits = scan.scan_changes([
            ("src/a.py", "import os  # noqa: F401\n", "\n"),
            ("src/b.py", "\n", "import os  # noqa: F401\n"),
        ])
        assert hits == []

    def test_only_the_added_one_of_two_alike_is_listed(self):
        before = "a = f()  # type: ignore\n"
        after = "a = f()  # type: ignore\na = f()  # type: ignore\n"
        assert _kinds(_one("src/m.py", before, after)) == [
            ("src/m.py", 2, "comment-marker:type-ignore")]

    def test_a_marker_the_batch_kept_is_not_listed(self):
        before = "import os  # noqa: F401\n"
        after = "import os  # noqa: F401\nx = 2\n"
        assert _one("src/m.py", before, after) == []

    def test_an_untokenizable_file_is_still_scanned(self):
        after = 'x = (1,\ny = "unterminated  # noqa: E501\n'
        hits = _one("src/m.py", None, after)
        assert [hit["kind"] for hit in hits] == ["unparseable", "comment-marker:noqa"]
        assert hits[0]["line"] == 0 and hits[0]["marker"] == scan.UNPARSEABLE_MARKER

    def test_a_file_that_did_not_parse_before_either_is_not_named_again(self):
        broken = "def f(:\n"
        assert _one("src/m.py", broken, broken + "x = 1\n") == []


# ---------------------------------------------------------------------------
# A marker the file already held, on a line the batch edited
# ---------------------------------------------------------------------------

class TestAnEditedLine:

    def test_editing_the_code_beside_an_existing_marker_does_not_add_it(self):
        before = "import os  # noqa: F401\n"
        after = "import os, sys  # noqa: F401\n"
        assert _one("src/m.py", before, after) == []

    def test_the_recorded_before_side_nets_an_edited_line_too(self):
        before = "import os  # noqa: F401\nx = 1\n"
        after = "import os, sys  # noqa: F401\nx = 2\n"
        side = scan.before_side("src/m.py", before)
        assert scan.scan_changes([("src/m.py", side, after)]) == []

    @pytest.mark.parametrize("after, line", [
        ("import os, sys  # noqa: F401\nimport json  # noqa: F401\n", 2),
        ("import json  # noqa: F401\nimport os, sys  # noqa: F401\n", 1),
    ])
    def test_a_second_copy_beside_an_edited_one_is_listed_where_it_is_new(self, after, line):
        hits = _one("src/m.py", "import os  # noqa: F401\n", after)
        assert [(hit["line"], hit["text"]) for hit in hits] == [
            (line, "import json  # noqa: F401")]

    def test_another_marker_on_an_edited_line_is_still_new(self):
        before = "import os  # noqa: F401\n"
        after = "import os, sys  # noqa: F401  # type: ignore\n"
        assert [hit["marker"] for hit in _one("src/m.py", before, after)] == ["# type: ignore"]

    @pytest.mark.parametrize("order", [("src/a.py", "src/b.py"), ("src/b.py", "src/a.py")])
    def test_a_copy_added_in_another_file_is_listed_there_in_either_order(self, order):
        sides = {
            "src/a.py": ("import os  # noqa: F401\n", "import os  # noqa: F401\nx = 1\n"),
            "src/b.py": ("\n", "import os  # noqa: F401\n"),
        }
        hits = scan.scan_changes([(path, *sides[path]) for path in order])
        assert _kinds(hits) == [("src/b.py", 1, "comment-marker:noqa")]

    def test_an_edited_line_does_not_also_net_a_new_copy_in_another_file(self):
        hits = scan.scan_changes([
            ("src/a.py", "import os  # noqa: F401\n", "import os, sys  # noqa: F401\n"),
            ("src/b.py", "\n", "import os  # noqa: F401\n"),
        ])
        assert _kinds(hits) == [("src/b.py", 1, "comment-marker:noqa")]

    def test_a_c_like_marker_on_an_edited_line(self):
        before = "console.log(a); // eslint-disable-line no-console\n"
        after = "console.log(a, b); // eslint-disable-line no-console\n"
        assert _one("src/a.ts", before, after) == []

    @pytest.mark.parametrize("before, after", [
        # Nothing alike: the vanished marker's line is gone, a new one is added.
        ("import os  # noqa: F401\nx = 1\n", "x = compute()  # noqa: F401\n"),
        # Half the words alike is not more than half (PAIRING_MIN_ALIKE).
        ("import os  # noqa: F401\n", "import json  # noqa: F401\n"),
        ("from a import b  # noqa: F401\n", "from c import d  # noqa: F401\n"),
        # Only punctuation alike: words are what count.
        ("a = f()  # type: ignore\n", "b = g()  # type: ignore\n"),
    ], ids=["unrelated", "import-swapped", "from-import-swapped", "punctuation-only"])
    def test_a_marker_on_an_unlike_line_is_not_netted_by_one_removed(self, before, after):
        """Fable review 2 of 2026-11-19, N2: the pairing took the most alike
        candidate even at zero likeness, so a marker dropped from one line hid
        the same marker added on an unrelated one."""
        hits = _one("src/m.py", before, after)
        assert [hit["line"] for hit in hits] == [1]

    def test_each_marker_kind_is_paired_on_its_own_likeness(self):
        before = "import os  # noqa: F401\nself.x = x  # type: ignore\n"
        after = "self.y = y  # type: ignore\nimport os, sys  # noqa: F401\n"
        assert _kinds(_one("src/m.py", before, after)) == [
            ("src/m.py", 1, "comment-marker:type-ignore")]

    def test_a_skip_on_a_renamed_test_is_listed(self):
        """Skips are netted on the whole key only (Fable review 2 of
        2026-11-19, N2): the function around a skip is its place, so a
        renamed test is a new place."""
        before = "import pytest\n@pytest.mark.skip(reason='slow')\ndef test_a():\n    pass\n"
        after = "import pytest\n@pytest.mark.skip(reason='slow')\ndef test_a_renamed():\n    pass\n"
        assert _kinds(_one("tests/test_m.py", before, after)) == [
            ("tests/test_m.py", 2, "skip-marker:pytest.mark.skip")]

    def test_a_javascript_skip_on_an_edited_line_is_listed(self):
        before = "it.skip('handles x', () => {});\n"
        after = "it.skip('handles x', async () => {});\n"
        assert [hit["line"] for hit in _one("src/a.test.js", before, after)] == [1]

    def test_a_skip_moved_to_another_test_of_the_file_is_listed(self):
        """Fable review 2 of 2026-11-19, N2: un-skipping test_a and skipping
        test_b in one batch listed nothing."""
        before = ("import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n\n"
                  "def test_b():\n    pass\n")
        after = ("import pytest\ndef test_a():\n    pass\n\n"
                 "@pytest.mark.skip\ndef test_b():\n    pass\n")
        assert _kinds(_one("tests/test_m.py", before, after)) == [
            ("tests/test_m.py", 5, "skip-marker:pytest.mark.skip")]

    def test_a_module_skip_dropped_for_a_function_skip_is_listed(self):
        before = "import pytest\npytestmark = pytest.mark.skip\ndef test_a():\n    pass\n"
        after = "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n"
        assert _kinds(_one("tests/test_m.py", before, after)) == [
            ("tests/test_m.py", 2, "skip-marker:pytest.mark.skip")]

    def test_a_skip_moved_to_another_file_of_the_batch_is_still_netted(self):
        skip = "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n"
        hits = scan.scan_changes([
            ("tests/test_old.py", skip, "import pytest\n"),
            ("tests/test_new.py", None, skip),
        ])
        assert hits == []

    def test_past_the_pairing_cap_the_count_still_nets(self, monkeypatch):
        monkeypatch.setattr(scan, "PAIRING_MAX", 0)
        before = "import os  # noqa: F401\n"
        after = "import os, sys  # noqa: F401\nimport json  # noqa: F401\n"
        assert len(_one("src/m.py", before, after)) == 1

    def test_past_the_pairing_cap_no_likeness_is_weighed(self, monkeypatch):
        """Past the cap the netting is by count alone: the vanished side has
        no line to pair by, so a floor over arbitrary pairs would list two
        edited lines that the weighed pairing nets."""
        before = "b = f(x)  # type: ignore\na = g(y)  # type: ignore\n"
        after = "b = f(x, 1)  # type: ignore\na = g(y, 1)  # type: ignore\n"
        assert _one("src/m.py", before, after) == []
        monkeypatch.setattr(scan, "PAIRING_MAX", 0)
        assert _one("src/m.py", before, after) == []
        # By count, an unlike line nets too: the cost of the fallback.
        assert _one("src/m.py", "import os  # noqa: F401\n", "x = compute()  # noqa: F401\n") == []

    def test_a_second_skip_beside_an_existing_one_is_still_listed(self):
        before = "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n"
        after = before + "@pytest.mark.skip\ndef test_b():\n    pass\n"
        assert _kinds(_one("tests/test_m.py", before, after)) == [
            ("tests/test_m.py", 5, "skip-marker:pytest.mark.skip")]


# ---------------------------------------------------------------------------
# One odd file never costs the batch its other hits
# ---------------------------------------------------------------------------

class TestOneOddFile:

    NOQA = ("src/m.py", "x = 1\n", "x = 1  # noqa: E501\n")

    def test_a_toml_date_is_spelled_as_its_text(self):
        before = "[tool.ruff]\nline-length = 79\n"
        after = "[tool.ruff]\nline-length = 79\nstamp = 2024-01-01\n"
        hits = scan.scan_changes([("pyproject.toml", before, after), self.NOQA])
        assert [(hit["path"], hit["marker"]) for hit in hits] == [
            ("pyproject.toml", 'ruff.stamp: "<absent>" -> "2024-01-01"'),
            ("src/m.py", "# noqa: E501"),
        ]

    def test_a_nested_select_entry_is_compared_without_a_set(self):
        before = "[lint]\nselect = [['E'], 'F']\n"
        after = "[lint]\nselect = ['F']\n"
        hits = scan.scan_changes([("ruff.toml", before, after), self.NOQA])
        assert [(hit["path"], hit["kind"]) for hit in hits] == [
            ("ruff.toml", "cfg-select-removed"), ("src/m.py", "comment-marker:noqa")]

    def test_a_file_the_scan_fails_on_is_named_and_the_others_still_report(self, monkeypatch):
        real = scan._skip_markers

        def odd(path, src):
            if path == "src/bad.py":
                raise RecursionError("maximum recursion depth exceeded")
            return real(path, src)

        monkeypatch.setattr(scan, "_skip_markers", odd)
        # The failed file lost a marker another file gained: its before side
        # is never counted, so the other file's hit is not netted away.
        changes = [("src/bad.py", "import os  # noqa: F401\n", "\n"),
                   ("src/b.py", "\n", "import os  # noqa: F401\n"), self.NOQA]
        not_scanned = []
        hits = scan.scan_changes(changes, not_scanned)
        assert _kinds(hits) == [("src/b.py", 1, "comment-marker:noqa"),
                                ("src/m.py", 1, "comment-marker:noqa")]
        assert not_scanned == [{
            "path": "src/bad.py",
            "why": "the scan failed on it (RecursionError: maximum recursion depth exceeded)",
        }]
        # Without a list to name it in, the others still report.
        assert scan.scan_changes(changes) == hits


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------

class TestSkipMarkers:

    def test_a_skip_decorator_carries_its_reason(self):
        before = "def test_a():\n    assert f() == 1\n"
        after = (
            "import pytest\n\n@pytest.mark.skip(reason=\"flaky on CI\")\n"
            "def test_a():\n    assert f() == 1\n"
        )
        hits = _one("tests/test_m.py", before, after)
        assert _kinds(hits) == [("tests/test_m.py", 3, "skip-marker:pytest.mark.skip")]
        assert hits[0]["reason"] == "flaky on CI"

    def test_xfail_skipif_and_calls_are_skips(self):
        after = (
            "import pytest, unittest\n"
            "@pytest.mark.xfail\ndef test_a():\n    pass\n"
            "@pytest.mark.skipif(True, reason='windows')\ndef test_b():\n    pass\n"
            "def test_c():\n    pytest.skip('needs network')\n"
            "@unittest.skip('later')\ndef test_d():\n    pass\n"
        )
        hits = _one("tests/test_m.py", None, after)
        assert {hit["kind"] for hit in hits} == {
            "skip-marker:pytest.mark.xfail", "skip-marker:pytest.mark.skipif",
            "skip-marker:pytest.skip()", "skip-marker:unittest.skip()",
        }
        reasons = {hit["kind"]: hit["reason"] for hit in hits}
        assert reasons["skip-marker:pytest.mark.skipif"] == "windows"
        assert reasons["skip-marker:pytest.skip()"] == "needs network"
        assert reasons["skip-marker:pytest.mark.xfail"] is None

    def test_a_skip_the_test_already_had_is_not_listed(self):
        text = "import pytest\n@pytest.mark.skip\ndef test_a():\n    pass\n"
        assert _one("tests/test_m.py", text, text + "\nX = 1\n") == []

    def test_a_skip_named_in_a_string_is_not_one(self):
        after = 'DOC = "@pytest.mark.skip(reason=1)"\n'
        assert _one("tests/test_m.py", None, after) == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_a_raised_threshold_is_a_loosening_on_its_line(self):
        before = "line-length = 79\n[lint.mccabe]\nmax-complexity = 6\n"
        after = "line-length = 79\n[lint.mccabe]\nmax-complexity = 9  # the parser is big\n"
        hits = _one("ruff.toml", before, after)
        assert _kinds(hits) == [("ruff.toml", 3, "cfg-loosened-threshold")]
        assert hits[0]["marker"] == "lint.mccabe.max-complexity: 6 -> 9"
        assert hits[0]["reason"] == "the parser is big"

    def test_pyproject_ignore_entry_and_dropped_select(self):
        before = (
            "[project]\nname = 'x'\n[tool.ruff.lint]\nselect = ['E', 'F', 'BLE']\n"
        )
        after = (
            "[project]\nname = 'y'\n[tool.ruff.lint]\nselect = ['E', 'F']\n"
            "ignore = ['E501']\n"
        )
        hits = _one("pyproject.toml", before, after)
        kinds = {hit["kind"] for hit in hits}
        assert kinds == {"cfg-select-removed", "cfg-ignore-entry"}
        # [project] is not a check's configuration.
        assert not any("name" in hit["marker"] for hit in hits)

    def test_a_new_ruff_config_is_named(self):
        hits = _one("pkg/ruff.toml", None, "[lint]\nignore = ['ARG001']\n")
        assert "new-lint-config" in {hit["kind"] for hit in hits}

    def test_an_ini_config_is_compared_line_by_line(self):
        before = "[flake8]\nmax-line-length = 79\n"
        after = "[flake8]\n# generated code\nmax-line-length = 79\nper-file-ignores = gen.py:E501\n"
        hits = _one("setup.cfg", before, after)
        assert _kinds(hits) == [("setup.cfg", 4, "cfg-changed")]
        assert hits[0]["marker"] == "+ per-file-ignores = gen.py:E501"

    def test_an_eslint_or_tsconfig_change_is_listed(self):
        hits = _one("tsconfig.json", '{\n  "strict": true\n}\n', '{\n  "strict": false\n}\n')
        assert {hit["marker"] for hit in hits} == {'+ "strict": false', '- "strict": true'}


# ---------------------------------------------------------------------------
# C-like comment markers and JavaScript skips
# ---------------------------------------------------------------------------

class TestCLikeMarkers:

    def test_eslint_disable_with_a_description(self):
        after = "// eslint-disable-next-line no-console -- debugging aid\nconsole.log(x);\n"
        hits = _one("src/a.ts", "console.log(x);\n", after)
        assert [(h["kind"], h["marker"], h["reason"]) for h in hits] == [
            ("comment-marker:eslint-disable", "eslint-disable-next-line no-console",
             "debugging aid")]

    def test_a_block_comment_marker_without_a_reason(self):
        hits = _one("src/a.js", "x();\n", "/* eslint-disable */\nx();\n")
        assert [(h["marker"], h["reason"]) for h in hits] == [("eslint-disable", None)]

    def test_typescript_directives(self):
        after = (
            "// @ts-ignore\nconst a: number = b;\n"
            "// @ts-expect-error the generated types lag the API\nconst c: string = d;\n"
        )
        hits = _one("src/a.ts", None, after)
        assert [(h["kind"], h["reason"]) for h in hits] == [
            ("comment-marker:ts-ignore", None),
            ("comment-marker:ts-expect-error", "the generated types lag the API"),
        ]

    def test_go_nolint_with_a_reason_after_it(self):
        after = "defer f.Close() //nolint:errcheck // closing a read-only file\n"
        hits = _one("main.go", "defer f.Close()\n", after)
        assert [(h["marker"], h["reason"]) for h in hits] == [
            ("nolint:errcheck", "closing a read-only file")]

    def test_coverage_ignore(self):
        hits = _one("src/a.js", "if (x) y();\n", "/* istanbul ignore next */\nif (x) y();\n")
        assert [h["kind"] for h in hits] == ["comment-marker:coverage-ignore"]

    def test_a_marker_inside_a_string_is_not_a_comment(self):
        after = (
            'const help = "// eslint-disable-next-line no-console";\n'
            "const url = 'http://x/@ts-ignore';\n"
            "const t = `/* eslint-disable */`;\n"
        )
        assert _one("src/a.ts", None, after) == []

    def test_prose_that_names_a_marker_is_not_one(self):
        after = "// we never use eslint-disable here\n/* see @ts-ignore docs */\nx();\n"
        assert _one("src/a.ts", "x();\n", after) == []

    def test_a_moved_marker_is_netted_out(self):
        before = "// @ts-ignore\nconst a = b;\n\nfoo();\n"
        after = "foo();\n\n// @ts-ignore\nconst a = b;\n"
        assert _one("src/a.ts", before, after) == []

    def test_javascript_skips(self):
        after = (
            "it.skip('handles x', () => {});\n"
            "xit('handles y', () => {});\n"
            "describe.skip('suite', () => {});\n"
            "it('mentions it.skip( in a string', () => {});\n"
            "// it.skip('commented out', () => {});\n"
        )
        hits = _one("src/a.test.js", None, after)
        assert [(h["line"], h["kind"]) for h in hits] == [
            (1, "skip-marker:it.skip()"), (2, "skip-marker:xit()"),
            (3, "skip-marker:describe.skip()"),
        ]


# ---------------------------------------------------------------------------
# The before side, scope and what a person reads
# ---------------------------------------------------------------------------

class TestTheBeforeSide:

    def test_recorded_counts_net_as_the_text_does(self):
        before = "import os  # noqa: F401\n@pytest.mark.skip\ndef test_a():\n    pass\n"
        after = before + "import sys  # noqa: F401\n"
        side = scan.before_side("tests/test_m.py", before)
        assert json.loads(json.dumps(side)) == side
        assert scan.scan_changes([("tests/test_m.py", side, after)]) == \
            _one("tests/test_m.py", before, after)

    def test_a_config_file_keeps_its_text(self):
        side = scan.before_side("ruff.toml", "line-length = 79\n")
        assert side["config"] == "line-length = 79\n"
        hits = scan.scan_changes([("ruff.toml", side, "line-length = 120\n")])
        assert [hit["kind"] for hit in hits] == ["cfg-loosened-threshold"]

    def test_an_absent_file(self):
        assert scan.before_side("src/m.py", None) is None


class TestScope:

    @pytest.mark.parametrize("path, wanted", [
        ("src/m.py", True), ("stubs/m.pyi", True), ("web/a.tsx", True), ("cmd/main.go", True),
        ("pyproject.toml", True), (".eslintrc.json", True), ("tsconfig.build.json", True),
        ("README.md", False), ("data.json", False), (".venv/lib/x.py", False),
        ("node_modules/x/index.js", False), (".multi-llm/tmp/x.py", False),
    ])
    def test_scannable(self, path, wanted):
        assert scan.scannable(path) is wanted

    def test_an_unscannable_file_is_skipped(self):
        assert _one("notes.md", None, "x  # noqa\n") == []

    def test_a_secret_on_the_line_is_redacted(self):
        token = "sk-" + "a1B2c3D4" * 6
        after = f'API_KEY = "{token}"  # noqa: S105\n'
        hits = _one("src/m.py", None, after)
        assert hits and token not in hits[0]["text"]


class TestWhatAPersonReads:

    HIT = {"path": "src/m.py", "line": 3, "kind": "comment-marker:noqa",
           "marker": "# noqa: E501", "text": "x = 1  # noqa: E501", "reason": None}

    def test_the_section_lists_each_hit_and_the_unscanned_files(self):
        lines = scan.section_lines([
            ("apply-fix-batch-0", [self.HIT], [{"path": "big.py", "why": "it is too big"}]),
        ])
        text = "\n".join(lines)
        assert lines[1] == scan.CHECK_THESE_HEADING
        assert "- `src/m.py:3` `# noqa: E501` — **no reason given** — apply-fix-batch-0 — " \
            "`x = 1  # noqa: E501`" in text
        assert "- `big.py` — apply-fix-batch-0: not scanned, it is too big" in text

    def test_nothing_to_list(self):
        assert scan.section_lines([]) == []
        assert scan.notice_clause([]) is None

    def test_a_line_with_a_backquote_stays_one_code_span(self):
        hit = dict(self.HIT, text="msg = `x`  # noqa", reason="kept")
        line = scan.section_lines([("T001", [hit], [])])[-1]
        assert "``msg = `x`  # noqa``" in line and "reason: kept" in line

    def test_the_notice_names_the_first_three(self):
        hits = [dict(self.HIT, line=n) for n in range(1, 6)]
        clause = scan.notice_clause([("T001", hits, [])])
        assert clause.startswith("check these: the batches added 5 suppression(s)")
        assert "5 with no reason given" in clause
        assert "src/m.py:3 # noqa: E501, +2 more" in clause
        assert "Check these: suppressions added" in clause

    def test_from_rows_names_an_earlier_run(self):
        rows = [{"task_id": "apply-fix-batch-0", "from_earlier_run": True,
                 "suppressions": [self.HIT]}, {"task_id": "apply-fix-batch-1"}]
        assert scan.from_rows(rows) == [("apply-fix-batch-0 (earlier run)", [self.HIT], [])]

    def test_hit_line(self):
        assert scan.hit_line(dict(self.HIT, reason="generated")) == (
            "src/m.py:3 # noqa: E501 — reason: generated: x = 1  # noqa: E501")
