"""``tests/tools/junit_times.py``: the CI legs' JUnit files summed by test file.

The parallel CI leg's log cannot be profiled by file, so this tool is the only
per-file record of it; a wrong grouping would compare the wrong files.
"""

from __future__ import annotations

import json

from tests.tools import junit_times

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="5">
<testcase classname="tests.test_alpha" name="test_one" time="1.5"/>
<testcase classname="tests.test_alpha.TestGroup" name="test_two" time="0.5"/>
<testcase classname="tests.container.test_beta" name="test_three" time="2.0"/>
<testcase classname="tests.test_gamma" name="test_skipped" time="0.000"><skipped/></testcase>
<testcase classname="tests.conftest" name="setup" time="0.25"/>
</testsuite></testsuites>
"""


def _write(tmp_path, name: str, text: str = JUNIT):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_cases_are_summed_by_the_file_their_classname_names(tmp_path):
    seconds, counts = junit_times.per_file([_write(tmp_path, "a.xml")])
    assert seconds == {"test_alpha.py": 2.0, "test_beta.py": 2.0, "test_gamma.py": 0.0,
                       "tests.conftest": 0.25}
    assert counts == {"test_alpha.py": 2, "test_beta.py": 1, "test_gamma.py": 1,
                      "tests.conftest": 1}


def test_several_files_add_up_as_the_shards_of_one_leg(tmp_path):
    paths = [_write(tmp_path, "a.xml"), _write(tmp_path, "b.xml")]
    seconds, counts = junit_times.per_file(paths)
    assert seconds["test_alpha.py"] == 4.0
    assert counts["test_alpha.py"] == 4


def test_json_lists_every_file(tmp_path, capsys):
    assert junit_times.main([_write(tmp_path, "a.xml"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "test_alpha.py": 2.0, "test_beta.py": 2.0, "test_gamma.py": 0.0, "tests.conftest": 0.25}


def test_the_table_puts_the_slowest_file_first(tmp_path, capsys):
    assert junit_times.main([_write(tmp_path, "a.xml"), "--top", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "4 files, 5 tests, 4 s summed"
    assert len(out) == 3
    assert out[1].endswith("test_alpha.py") or out[1].endswith("test_beta.py")
