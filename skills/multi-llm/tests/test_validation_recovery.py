"""Unit tests for validation error recovery features in utils/validation.py."""
import json
import tempfile
from pathlib import Path

import pytest

from utils.validation import (
    ERROR_TYPE_AMBIGUOUS,
    ERROR_TYPE_MODEL_FAILURE,
    ERROR_TYPE_PARSING,
    ERROR_TYPE_RATE_LIMITED,
    ERROR_TYPE_TIMEOUT,
    ERROR_TYPE_UNKNOWN,
    RECOVERABLE_ERROR_TYPES,
    _classify_validation_error,
    load_validation_results,
    save_validation_results,
)


class TestErrorTypeConstants:
    """Tests for error type constants and RECOVERABLE_ERROR_TYPES."""

    def test_error_type_parsing_exists(self):
        """ERROR_TYPE_PARSING constant exists with expected value."""
        assert ERROR_TYPE_PARSING == "parsing_error"

    def test_error_type_timeout_exists(self):
        """ERROR_TYPE_TIMEOUT constant exists with expected value."""
        assert ERROR_TYPE_TIMEOUT == "timeout"

    def test_error_type_rate_limited_exists(self):
        """ERROR_TYPE_RATE_LIMITED constant exists with expected value."""
        assert ERROR_TYPE_RATE_LIMITED == "rate_limited"

    def test_error_type_ambiguous_exists(self):
        """ERROR_TYPE_AMBIGUOUS constant exists with expected value."""
        assert ERROR_TYPE_AMBIGUOUS == "real_ambiguity"

    def test_error_type_model_failure_exists(self):
        """ERROR_TYPE_MODEL_FAILURE constant exists with expected value."""
        assert ERROR_TYPE_MODEL_FAILURE == "model_failure"

    def test_error_type_unknown_exists(self):
        """ERROR_TYPE_UNKNOWN constant exists with expected value."""
        assert ERROR_TYPE_UNKNOWN == "unknown"

    def test_recoverable_error_types_contains_parsing(self):
        """RECOVERABLE_ERROR_TYPES contains parsing_error."""
        assert ERROR_TYPE_PARSING in RECOVERABLE_ERROR_TYPES

    def test_recoverable_error_types_contains_timeout(self):
        """RECOVERABLE_ERROR_TYPES contains timeout."""
        assert ERROR_TYPE_TIMEOUT in RECOVERABLE_ERROR_TYPES

    def test_recoverable_error_types_contains_rate_limited(self):
        """RECOVERABLE_ERROR_TYPES contains rate_limited."""
        assert ERROR_TYPE_RATE_LIMITED in RECOVERABLE_ERROR_TYPES

    def test_recoverable_error_types_is_frozenset(self):
        """RECOVERABLE_ERROR_TYPES is a frozenset (immutable)."""
        assert isinstance(RECOVERABLE_ERROR_TYPES, frozenset)

    def test_recoverable_error_types_exactly_three(self):
        """RECOVERABLE_ERROR_TYPES contains exactly three error types."""
        assert len(RECOVERABLE_ERROR_TYPES) == 3


class TestClassifyValidationError:
    """Tests for _classify_validation_error function."""

    # HTTP status code tests
    def test_http_429_returns_rate_limited(self):
        """HTTP 429 status returns rate_limited."""
        result = _classify_validation_error("Some error", http_status=429)
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_http_408_returns_timeout(self):
        """HTTP 408 status returns timeout."""
        result = _classify_validation_error("Some error", http_status=408)
        assert result == ERROR_TYPE_TIMEOUT

    def test_http_504_returns_timeout(self):
        """HTTP 504 status returns timeout."""
        result = _classify_validation_error("Some error", http_status=504)
        assert result == ERROR_TYPE_TIMEOUT

    def test_http_500_returns_model_failure(self):
        """HTTP 500 status returns model_failure."""
        result = _classify_validation_error("Some error", http_status=500)
        assert result == ERROR_TYPE_MODEL_FAILURE

    def test_http_502_returns_model_failure(self):
        """HTTP 502 status returns model_failure."""
        result = _classify_validation_error("Some error", http_status=502)
        assert result == ERROR_TYPE_MODEL_FAILURE

    def test_http_503_returns_model_failure(self):
        """HTTP 503 status returns model_failure."""
        result = _classify_validation_error("Some error", http_status=503)
        assert result == ERROR_TYPE_MODEL_FAILURE

    # Parsing error tests
    def test_json_in_error_returns_parsing_error(self):
        """'json' in error message returns parsing_error."""
        result = _classify_validation_error("Invalid JSON in response")
        assert result == ERROR_TYPE_PARSING

    def test_parse_error_in_error_returns_parsing_error(self):
        """'parse error' in error message returns parsing_error."""
        result = _classify_validation_error("Parse error occurred")
        assert result == ERROR_TYPE_PARSING

    def test_decode_error_in_error_returns_parsing_error(self):
        """'decode error' in error message returns parsing_error."""
        result = _classify_validation_error("Decode error while processing")
        assert result == ERROR_TYPE_PARSING

    def test_parsing_case_insensitive(self):
        """Parsing error detection is case insensitive."""
        result = _classify_validation_error("INVALID JSON FORMAT")
        assert result == ERROR_TYPE_PARSING

    # Timeout error tests
    def test_timeout_in_error_returns_timeout(self):
        """'timeout' in error message returns timeout."""
        result = _classify_validation_error("Request timeout")
        assert result == ERROR_TYPE_TIMEOUT

    def test_timed_out_in_error_returns_timeout(self):
        """'timed out' in error message returns timeout."""
        result = _classify_validation_error("Connection timed out")
        assert result == ERROR_TYPE_TIMEOUT

    def test_timeout_case_insensitive(self):
        """Timeout detection is case insensitive."""
        result = _classify_validation_error("REQUEST TIMEOUT EXCEEDED")
        assert result == ERROR_TYPE_TIMEOUT

    # Rate limiting tests
    def test_rate_limit_in_error_returns_rate_limited(self):
        """'rate limit' in error message returns rate_limited."""
        result = _classify_validation_error("Rate limit exceeded")
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_too_many_requests_returns_rate_limited(self):
        """'too many requests' returns rate_limited."""
        result = _classify_validation_error("Too many requests")
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_throttled_returns_rate_limited(self):
        """'throttled' returns rate_limited."""
        result = _classify_validation_error("Request was throttled")
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_quota_exceeded_returns_rate_limited(self):
        """'quota exceeded' returns rate_limited."""
        result = _classify_validation_error("Quota exceeded for the day")
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_rate_limit_case_insensitive(self):
        """Rate limit detection is case insensitive."""
        result = _classify_validation_error("RATE LIMIT REACHED")
        assert result == ERROR_TYPE_RATE_LIMITED

    # Model failure tests
    def test_binary_not_found_returns_model_failure(self):
        """'binary not found' returns model_failure."""
        result = _classify_validation_error("Binary not found in PATH")
        assert result == ERROR_TYPE_MODEL_FAILURE

    def test_service_unavailable_returns_model_failure(self):
        """'service unavailable' returns model_failure."""
        result = _classify_validation_error("Service unavailable")
        assert result == ERROR_TYPE_MODEL_FAILURE

    def test_connection_refused_returns_model_failure(self):
        """'connection refused' returns model_failure."""
        result = _classify_validation_error("Connection refused by server")
        assert result == ERROR_TYPE_MODEL_FAILURE

    def test_model_failure_case_insensitive(self):
        """Model failure detection is case insensitive."""
        result = _classify_validation_error("SERVICE UNAVAILABLE")
        assert result == ERROR_TYPE_MODEL_FAILURE

    # Unknown error tests
    def test_unknown_error_returns_unknown(self):
        """Unknown error message returns unknown."""
        result = _classify_validation_error("Some random error happened")
        assert result == ERROR_TYPE_UNKNOWN

    def test_empty_error_returns_unknown(self):
        """Empty error message returns unknown."""
        result = _classify_validation_error("")
        assert result == ERROR_TYPE_UNKNOWN

    # Precedence tests
    def test_http_status_takes_precedence_over_error_text(self):
        """HTTP status code takes precedence over error text matching."""
        # Error text says "timeout" but HTTP status says rate limited
        result = _classify_validation_error("Request timeout", http_status=429)
        assert result == ERROR_TYPE_RATE_LIMITED

    def test_http_status_takes_precedence_parsing_text(self):
        """HTTP status takes precedence even with parsing-related error text."""
        result = _classify_validation_error("Invalid JSON", http_status=503)
        assert result == ERROR_TYPE_MODEL_FAILURE


class TestSaveValidationResults:
    """Tests for save_validation_results function."""

    def test_creates_output_directory_if_needed(self):
        """Creates output directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "validation.json"
            validation_results = [
                {"group_index": 0, "status": "valid", "reason": "Test", "confidence": 0.9}
            ]
            save_validation_results(validation_results, output_path, model="test-model")
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_saves_with_correct_schema_version(self):
        """Saves with schema version 2.1 (includes group_id support)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            validation_results = [
                {"group_index": 0, "status": "valid", "reason": "Test", "confidence": 0.9}
            ]
            save_validation_results(validation_results, output_path, model="test-model")

            with open(output_path, 'r', encoding="utf-8") as f:
                data = json.load(f)

            assert data["metadata"]["schema_version"] == "2.1"

    def test_includes_metadata_with_model_and_timestamp(self):
        """Includes metadata with model and timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            validation_results = [
                {"group_index": 0, "status": "valid", "reason": "Test", "confidence": 0.9}
            ]
            save_validation_results(validation_results, output_path, model="my-model")

            with open(output_path, 'r', encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert data["metadata"]["model"] == "my-model"
            assert "timestamp" in data["metadata"]

    def test_saves_all_validation_fields(self):
        """Saves all validation fields including error_type and recoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            validation_results = [
                {
                    "group_index": 0,
                    "status": "validation_failed",
                    "reason": "Timeout occurred",
                    "confidence": 0.0,
                    "error_type": ERROR_TYPE_TIMEOUT,
                    "recoverable": True,
                    "revalidated": False,
                }
            ]
            save_validation_results(validation_results, output_path, model="test-model")

            with open(output_path, 'r', encoding="utf-8") as f:
                data = json.load(f)

            group = data["groups"][0]
            assert group["group_index"] == 0
            assert group["status"] == "validation_failed"
            assert group["reason"] == "Timeout occurred"
            assert group["confidence"] == 0.0
            assert group["error_type"] == ERROR_TYPE_TIMEOUT
            assert group["recoverable"] is True
            assert group["revalidated"] is False


class TestLoadValidationResults:
    """Tests for load_validation_results function."""

    def test_loads_v2_format_correctly(self):
        """Loads v2 format correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v2_data = {
                "groups": [
                    {
                        "group_index": 0,
                        "status": "valid",
                        "reason": "Good",
                        "confidence": 0.95,
                        "error_type": ERROR_TYPE_UNKNOWN,
                        "recoverable": False,
                        "revalidated": False,
                    }
                ],
                "metadata": {
                    "model": "test",
                    "timestamp": "2025-01-01T00:00:00",
                    "schema_version": "2.0",
                }
            }
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v2_data, f)

            results = load_validation_results(output_path)

            assert len(results) == 1
            assert results[0]["group_index"] == 0
            assert results[0]["status"] == "valid"
            assert results[0]["error_type"] == ERROR_TYPE_UNKNOWN
            assert results[0]["recoverable"] is False

    def test_migrates_v1_list_format_to_v2(self):
        """Migrates v1 format (list) to v2 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            # V1 format was a direct list
            v1_data = [
                {
                    "group_index": 0,
                    "status": "valid",
                    "reason": "Looks good",
                    "confidence": 0.9,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            assert len(results) == 1
            # Migration should add missing fields
            assert "error_type" in results[0]
            assert "recoverable" in results[0]
            assert "revalidated" in results[0]

    def test_adds_error_type_for_old_needs_human_decision(self):
        """Adds error_type for old needs-human-decision items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "needs-human-decision",
                    "reason": "Could not determine",
                    "confidence": 0.0,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            # Default for needs-human-decision without ambiguity keywords is parsing_error
            assert results[0]["error_type"] == ERROR_TYPE_PARSING

    def test_adds_error_type_for_old_validation_failed(self):
        """Adds error_type for old validation_failed items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "validation_failed",
                    "reason": "Failed to validate",
                    "confidence": 0.0,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            # validation_failed defaults to parsing_error
            assert results[0]["error_type"] == ERROR_TYPE_PARSING

    def test_sets_recoverable_based_on_error_type(self):
        """Sets recoverable based on error_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "validation_failed",
                    "reason": "Parsing issue",
                    "confidence": 0.0,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            # parsing_error is recoverable
            assert results[0]["recoverable"] is True

    def test_sets_revalidated_false_for_migrated_items(self):
        """Sets revalidated=False for migrated items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "valid",
                    "reason": "Good",
                    "confidence": 0.9,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            assert results[0]["revalidated"] is False

    def test_detects_ambiguity_in_reason_text(self):
        """Detects ambiguity in reason text for error_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "needs-human-decision",
                    "reason": "This is ambiguous and requires judgment",
                    "confidence": 0.5,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            # Should detect "ambiguous" and "judgment" keywords
            assert results[0]["error_type"] == ERROR_TYPE_AMBIGUOUS

    def test_detects_unclear_in_reason_text(self):
        """Detects 'unclear' keyword in reason text for error_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation.json"
            v1_data = [
                {
                    "group_index": 0,
                    "status": "needs-human-decision",
                    "reason": "The intent is unclear",
                    "confidence": 0.5,
                }
            ]
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(v1_data, f)

            results = load_validation_results(output_path)

            assert results[0]["error_type"] == ERROR_TYPE_AMBIGUOUS


class TestRecoverableErrorTypes:
    """Tests for RECOVERABLE_ERROR_TYPES membership."""

    def test_parsing_error_is_recoverable(self):
        """parsing_error is recoverable."""
        assert ERROR_TYPE_PARSING in RECOVERABLE_ERROR_TYPES

    def test_timeout_is_recoverable(self):
        """timeout is recoverable."""
        assert ERROR_TYPE_TIMEOUT in RECOVERABLE_ERROR_TYPES

    def test_rate_limited_is_recoverable(self):
        """rate_limited is recoverable."""
        assert ERROR_TYPE_RATE_LIMITED in RECOVERABLE_ERROR_TYPES

    def test_model_failure_is_not_recoverable(self):
        """model_failure is NOT recoverable."""
        assert ERROR_TYPE_MODEL_FAILURE not in RECOVERABLE_ERROR_TYPES

    def test_real_ambiguity_is_not_recoverable(self):
        """real_ambiguity is NOT recoverable."""
        assert ERROR_TYPE_AMBIGUOUS not in RECOVERABLE_ERROR_TYPES

    def test_unknown_is_not_recoverable(self):
        """unknown is NOT recoverable."""
        assert ERROR_TYPE_UNKNOWN not in RECOVERABLE_ERROR_TYPES
