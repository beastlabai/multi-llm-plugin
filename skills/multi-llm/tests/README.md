# Test Suite Documentation

This directory contains unit tests and integration tests for the multi-llm skill.

## Quick Start

```bash
# Run all tests
uv run -- pytest

# Run with verbose output
uv run -- pytest -v

# Run a specific test file
uv run -- pytest tests/test_filtering.py -v

# Run a specific test class
uv run -- pytest tests/test_filtering.py::TestFilterItems -v

# Run a specific test
uv run -- pytest tests/test_filtering.py::TestFilterItems::test_filter_valid_items -v
```

## Test Categories

### Unit Tests

These test individual functions and classes in isolation:

| File | Module Tested | Description |
|------|---------------|-------------|
| `test_importance.py` | `utils/importance.py` | Importance level calculations and comparisons |
| `test_filtering.py` | `utils/filtering.py` | Bulk approval conflict resolution, item filtering |
| `test_validation_recovery.py` | `utils/validation.py` | Error classification, validation result loading/saving |
| `test_state_manager_decisions.py` | `utils/state_manager.py` | Group ID generation, human decision tracking, resume state |
| `test_schema_validator.py` | `utils/schema_validator.py` | JSON schema validation |
| `test_json_extractor.py` | `utils/json_extractor.py` | JSON extraction from LLM responses |
| `test_output_handler.py` | `utils/output_handler.py` | Output file path handling |
| `test_provider_registry.py` | `utils/provider_registry.py` | LLM provider configuration |

### Integration Tests

These test the orchestrators end-to-end using subprocess calls:

| File | Description |
|------|-------------|
| `test_apply_suggestions_integration.py` | Tests `apply_suggestions_orchestrator.py` with various CLI options |

## Running Integration Tests

Integration tests use mock data in `plans/test-plan/` to avoid calling real LLMs.

```bash
# Run integration tests
uv run -- pytest tests/test_apply_suggestions_integration.py -v

# Regenerate test fixtures if they become stale
uv run -- python tests/test_apply_suggestions_integration.py --setup-fixtures
```

### Integration Test Fixtures

The integration tests use a self-contained test plan at `plans/test-plan/`:

```
plans/
├── test-plan.md                    # Mock plan file
└── test-plan/
    ├── state.json                  # Session state (for resume tests)
    └── review-plan/
        ├── backup.md               # Original plan backup
        ├── grouped.json            # 10 suggestion groups
        └── validation.json         # Validation results (v2 format)
```

The fixtures include:
- **10 suggestion groups** with varying importance (HIGH, MEDIUM, LOW)
- **Validation results** covering all statuses:
  - 4 `valid`
  - 2 `needs-human-decision` (real ambiguity)
  - 3 `validation_failed` (parsing_error, timeout, rate_limited)
  - 1 `invalid`
- **State file** with:
  - 1 previously processed item
  - 1 previous human decision (approved)

## Test Patterns

### Import Pattern

All test files use this import pattern to access the skill modules:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.filtering import filter_items
```

### Fixture Pattern

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

@pytest.fixture
def sample_plan(temp_dir):
    plan_file = temp_dir / "test-plan.md"
    plan_file.write_text("# Test Plan\n...")
    return plan_file
```

### Class-Based Test Organization

Tests are organized into classes by feature:

```python
class TestFilterItems:
    """Tests for filter_items() function."""

    def test_filter_valid_items(self):
        ...

    def test_filter_with_bulk_approval(self):
        ...
```

## Coverage Report

To generate a coverage report:

```bash
# Run with coverage
uv run -- pytest --cov=. --cov-report=html

# Open the report
open htmlcov/index.html
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you're running from the skill directory:

```bash
cd skills/multi-llm
uv run -- pytest
```

### Stale Fixtures

If integration tests fail unexpectedly, regenerate the fixtures:

```bash
uv run -- python tests/test_apply_suggestions_integration.py --setup-fixtures
```

### Plan Hash Mismatch

The state file includes a `plan_hash` that must match the actual plan content. If you modify `plans/test-plan.md`, regenerate fixtures to update the hash.

## Adding New Tests

1. **Unit tests**: Add to the appropriate `test_*.py` file or create a new one
2. **Integration tests**: Add to `test_apply_suggestions_integration.py` or create a new integration test file
3. **Fixtures**: Add common fixtures to `conftest.py`

### Example: Adding a Unit Test

```python
# tests/test_my_module.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.my_module import my_function


class TestMyFunction:
    def test_basic_case(self):
        result = my_function("input")
        assert result == "expected"

    def test_edge_case(self):
        with pytest.raises(ValueError):
            my_function(None)
```

### Example: Adding an Integration Test

```python
# In test_apply_suggestions_integration.py

class TestNewFeature:
    def test_new_option_works(self):
        result = run_orchestrator("--new-option", "--dry-run")
        assert result.returncode == 0
        assert "expected output" in result.stderr
```
