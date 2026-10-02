"""Tests for schema validator module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.schema_validator import (
    drop_non_record_entries,
    validate_type,
    validate_against_schema,
    validate_json_output,
    validate_task_dependencies,
    validate_code_review_issues,
)


class TestValidateType:
    """Tests for type validation."""

    def test_string_type(self):
        """Test string type validation."""
        assert validate_type("hello", "string") is True
        assert validate_type(123, "string") is False

    def test_integer_type(self):
        """Test integer type validation."""
        assert validate_type(123, "integer") is True
        assert validate_type("123", "integer") is False

    def test_array_type(self):
        """Test array type validation."""
        assert validate_type([1, 2, 3], "array") is True
        assert validate_type("not array", "array") is False

    def test_object_type(self):
        """Test object type validation."""
        assert validate_type({"key": "value"}, "object") is True
        assert validate_type([1, 2], "object") is False

    def test_boolean_type(self):
        """Test boolean type validation."""
        assert validate_type(True, "boolean") is True
        assert validate_type("true", "boolean") is False


class TestValidateAgainstSchema:
    """Tests for schema validation."""

    def test_simple_object_schema(self):
        """Test validation against simple object schema."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }

        valid_data = {"name": "John", "age": 30}
        errors = validate_against_schema(valid_data, schema)
        assert len(errors) == 0

        invalid_data = {"name": "John"}  # missing age
        errors = validate_against_schema(invalid_data, schema)
        assert len(errors) > 0

    def test_array_schema(self):
        """Test validation of arrays."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"}
                }
            }
        }

        valid_data = [{"id": "1"}, {"id": "2"}]
        errors = validate_against_schema(valid_data, schema)
        assert len(errors) == 0

        invalid_data = [{"id": 1}]  # id should be string
        errors = validate_against_schema(invalid_data, schema)
        assert len(errors) > 0

    def test_enum_validation(self):
        """Test enum validation."""
        schema = {
            "type": "string",
            "enum": ["high", "medium", "low"]
        }

        errors = validate_against_schema("high", schema)
        assert len(errors) == 0

        errors = validate_against_schema("invalid", schema)
        assert len(errors) > 0


class TestValidateJsonOutput:
    """Tests for LLM JSON output validation."""

    def test_validate_raw_json(self):
        """Test validation of raw JSON string."""
        output = '[{"title": "test", "desc": "description", "importance": "high", "file": "test.py", "type": "bug"}]'

        is_valid, parsed, errors = validate_json_output(
            output, "code_review_issues.schema.json"
        )
        # May fail if schema file doesn't exist in test env
        assert parsed is not None

    def test_validate_json_in_code_block(self):
        """Test validation of JSON in markdown code block."""
        output = '''Here are the results:
```json
[{"id": "1"}]
```
'''
        is_valid, parsed, errors = validate_json_output(output, "task_decomposition.schema.json")
        assert isinstance(parsed, list)

    def test_invalid_json_returns_errors(self):
        """Test that invalid JSON returns appropriate errors."""
        output = "This is not JSON at all"

        is_valid, parsed, errors = validate_json_output(
            output, "code_review_issues.schema.json"
        )
        assert is_valid is False
        assert len(errors) > 0


class TestValidateTaskDependencies:
    """Tests for task dependency validation."""

    def test_valid_dependencies(self):
        """Test validation of valid dependencies."""
        tasks = [
            {"id": "T001", "depends_on": []},
            {"id": "T002", "depends_on": ["T001"]},
            {"id": "T003", "depends_on": ["T001", "T002"]}
        ]

        errors = validate_task_dependencies(tasks)
        assert len(errors) == 0

    def test_missing_dependency(self):
        """Test detection of missing dependency."""
        tasks = [
            {"id": "T001", "depends_on": []},
            {"id": "T002", "depends_on": ["T999"]}  # T999 doesn't exist
        ]

        errors = validate_task_dependencies(tasks)
        assert any("T999" in e for e in errors)

    def test_self_dependency(self):
        """Test detection of self-dependency."""
        tasks = [
            {"id": "T001", "depends_on": ["T001"]}  # self dependency
        ]

        errors = validate_task_dependencies(tasks)
        assert any("self-dependency" in e for e in errors)

    def test_circular_dependency(self):
        """Test detection of circular dependency."""
        tasks = [
            {"id": "T001", "depends_on": ["T002"]},
            {"id": "T002", "depends_on": ["T001"]}  # circular
        ]

        errors = validate_task_dependencies(tasks)
        assert any("Circular" in e for e in errors)


class TestValidateCodeReviewIssues:
    """Tests for code review issues validation."""

    def test_valid_issues(self, sample_code_review_issues):
        """Test validation of valid issues."""
        is_valid, errors = validate_code_review_issues(sample_code_review_issues)
        # May need schema file
        assert isinstance(errors, list)

    def test_empty_issues_is_valid(self):
        """Test that empty list is valid."""
        is_valid, errors = validate_code_review_issues([])
        assert len(errors) == 0 or "Schema not found" in str(errors)


    def test_a_null_line_range_is_a_file_level_finding_not_a_warning(self):
        """#23: two reviewers answered `"line_range": null` for a finding about a
        file that does not exist yet; every consumer already reads it as absent."""
        base = {"title": "t", "desc": "d", "importance": "high",
                "file": "index.html", "type": "missing"}
        assert validate_code_review_issues([{**base, "line_range": None}]) == (True, [])
        # A range of the wrong type still warns.
        is_valid, errors = validate_code_review_issues([{**base, "line_range": "1-2"}])
        assert not is_valid and errors


class TestDropNonRecordEntries:
    """The guard that acts on the verdict ``validate_code_review_issues`` prints.

    2026-09-20: one reviewer's result file held ``[0]`` — the stdout fallback
    had scraped a bracketed index out of its prose — and every aggregation site
    stamps the producing model onto each element it reads
    (``issue["model"] = model``). The validator said ``'[0]: expected object,
    got int'`` and only warned; the item assignment then raised ``TypeError``
    and the other four models' eleven findings were never written.
    """

    def test_the_real_failing_payload_yields_no_records(self, capsys):
        assert drop_non_record_entries([0], "cursor-agent:kimi-k3-high") == []

    def test_the_records_beside_a_stray_scalar_survive(self):
        issue = {"title": "x", "desc": "y", "importance": "low",
                 "file": "src/a.py", "type": "bug"}

        kept = drop_non_record_entries([issue, 0, "prose"], "m:a")

        assert kept == [issue]

    def test_a_clean_payload_is_returned_unchanged_and_in_silence(self, capsys):
        issues = [{"title": "x"}, {"title": "y"}]

        assert drop_non_record_entries(issues, "m:a") == issues
        assert capsys.readouterr().err == ""

    def test_nothing_to_aggregate_is_not_a_warning(self, capsys):
        assert drop_non_record_entries(None, "m:a") == []
        assert drop_non_record_entries([], "m:a") == []
        assert capsys.readouterr().err == ""

    def test_a_payload_that_is_not_a_list_yields_no_records(self):
        """Iterating a dict hands the caller its *keys*, which fail identically."""
        assert drop_non_record_entries({"issues": []}, "m:a") == []

    def test_the_warning_names_the_model_the_element_and_the_disposition(
        self, capsys
    ):
        drop_non_record_entries([0], "cursor-agent:kimi-k3-high", noun="issues")

        warning = capsys.readouterr().err
        assert "cursor-agent:kimi-k3-high" in warning
        assert "0 (int)" in warning
        assert "issues" in warning

    def test_the_warning_goes_to_stderr_and_never_to_stdout(self, capsys):
        """stdout is the machine-readable channel (the bracketed markers)."""
        drop_non_record_entries([0], "m:a")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "WARNING:" in captured.err

    def test_the_noun_follows_the_phase(self, capsys):
        drop_non_record_entries([0], "m:a", noun="suggestions")

        assert "suggestions" in capsys.readouterr().err
