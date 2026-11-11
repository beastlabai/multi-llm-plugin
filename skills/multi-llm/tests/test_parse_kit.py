"""The parse result every adapter returns (``utils/providers/parse_kit.py``).

The adapters' own rows are graded by ``test_provider_fixture_rows.py``; this
file pins the kit's shapes, including key order, which logs and result files
serialise as built.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_extractor import extract_json_from_text  # noqa: E402
from utils.providers.parse_kit import (  # noqa: E402
    answer_from_text,
    answer_json,
    direct_json,
    fail,
    loads_or_extract,
    ok,
)


def test_a_failure_has_four_keys_in_order_then_the_adapters():
    assert list(fail("boom", "out")) == ["success", "error", "raw", "data"]
    assert fail("boom", "out") == {"success": False, "error": "boom", "raw": "out", "data": None}
    extended = fail("cut", "text", incomplete=True, error_code="E")
    assert list(extended) == ["success", "error", "raw", "data", "incomplete", "error_code"]


def test_a_success_carries_only_its_data():
    assert ok([1]) == {"success": True, "data": [1]}
    assert ok(None) == {"success": True, "data": None}


def test_each_call_builds_a_fresh_dict():
    first, second = fail("e", "r"), fail("e", "r")
    first["error"] = "changed"
    assert second["error"] == "e"


@pytest.mark.parametrize("text, expected", [
    ('[{"a": 1}]', {"success": True, "data": [{"a": 1}]}),
    ('{"findings": [1]}', {"success": True, "data": {"findings": [1]}}),
    ('{"cut"', None),
    ("prose [1]", None),
    (' [1]', None),  # read as given: the caller strips
])
def test_direct_json_is_the_whole_document_or_nothing(text, expected):
    assert direct_json(text) == expected


def test_answer_json_serves_a_whole_document_before_the_extractor_would():
    """The extractor prefers arrays, so it would serve the inner list."""
    text = '{"findings": [{"id": 1}]}'
    assert answer_json(text) == {"success": True, "data": {"findings": [{"id": 1}]}}
    assert extract_json_from_text(text, prefer_arrays=True)["data"] != {"findings": [{"id": 1}]}


def test_answer_json_falls_back_to_the_extractor():
    text = "Here it is:\n```json\n[{\"id\": 2}]\n```"
    assert answer_json(text) == extract_json_from_text(text, prefer_arrays=True)


class TestAnswerFromText:
    def test_strips_then_parses(self):
        assert answer_from_text('  [1, 2]\n', raw="stdout") == {"success": True, "data": [1, 2]}

    def test_an_empty_answer_fails_with_the_callers_raw(self):
        assert answer_from_text("  \n", raw="the envelope") == fail(
            "Empty text response", "the envelope"
        )

    def test_raw_defaults_to_the_stripped_text(self):
        assert answer_from_text("   ", empty_error="Empty output") == fail("Empty output", "")

    def test_no_empty_check_hands_empty_text_to_the_extractor(self):
        assert answer_from_text("  ", empty_error=None) == extract_json_from_text(
            "", prefer_arrays=True
        )


class TestLoadsOrExtract:
    def test_one_document(self):
        assert loads_or_extract('{"text": "x"}') == ({"text": "x"}, None)

    def test_a_null_document_is_still_a_document(self):
        assert loads_or_extract("null") == (None, None)

    def test_an_extractable_stdout_is_the_extractors_success(self):
        stdout = 'log line\n[{"id": 1}]'
        envelope, result = loads_or_extract(stdout)
        assert envelope is None
        assert result == extract_json_from_text(stdout, prefer_arrays=True)
        assert result["success"] is True

    def test_nothing_parseable_is_the_json_parse_error(self):
        envelope, result = loads_or_extract("no json at all")
        assert envelope is None
        try:
            json.loads("no json at all")
        except json.JSONDecodeError as exc:
            expected = f"JSON parse error: {exc}"
        assert result == fail(expected, "no json at all")
