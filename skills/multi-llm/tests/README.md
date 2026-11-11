# Test Suite Documentation

This directory contains unit tests and integration tests for the multi-llm skill.

## Quick Start

Run from the **repository root**, with `uv run --project` (not `--directory`):

```bash
# Which tests cover a module, then run only those (advisory, -n 4)
make test-for MODULE=utils/status_file.py

# One file, one class, one test
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py -v
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py::TestFilterItems -v

# The ordinary CI selection (the gate; `make test-for` never replaces it),
# in parallel with --dist loadfile, or serially as the gating CI job runs it
make test
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests -m "not live and not container and not package and not codex_jail"
```

A bare `pytest` applies the `addopts` default in `pyproject.toml`
(`-m 'not container and not package and not codex_jail'`).

## Which tests cover a module

File names do not follow the module layout, so use the generated map rather than
guessing:

- [`MODULE_TESTS.md`](MODULE_TESTS.md) lists, for every source module, how many
  test files cover it and the best matches. It is generated; do not edit it.
- `make test-for MODULE=utils/foo.py [TOP=n]` (backed by
  [`tools/tests_for.py`](tools/tests_for.py)) prints the ranked files, best name
  match first, and runs only those. It marks a **hub** (more than 40 covering
  files, for example `utils.state_manager`), which the map cannot narrow.
- The map is a static AST scan. It counts `import x`, `from x import y`, and a
  literal `importlib.import_module("x")` / `__import__("x")` under any alias,
  plus two declared tables in `tools/tests_for.py`, each row with a reason:
  `SWEEPS` (a test that loads modules by a computed name, such as every
  `utils.providers.*` adapter) and `HARNESS_SUITES` (every `test_e2e_*.py`,
  which drive the skill as a subprocess; the orchestrators a suite runs are
  derived from the mode words and orchestrator names it passes). A new
  `test_e2e_*.py` needs a `HARNESS_SUITES` row or `test_module_tests_map.py` fails.
- A `tests/support` module a test imports counts its own imports as the test's
  (followed through the support modules it imports in turn).
- Still not covered: `runpy`, any other helper module that re-exports a source
  module, a suite-wide fixture a test names without importing it, a computed
  import nobody declared, and coverage through another source module.
  The map picks a focused first run; it is never a reason to skip the CI command.
- After a test's imports or a declared route change, run `make tests-map` and
  commit `MODULE_TESTS.md`; `test_module_tests_map.py` fails until you do. Adding
  a test that maps to no module leaves the file unchanged.

## Running Integration Tests

Integration tests use mock data in `plans/test-plan/` to avoid calling real LLMs.
Each test builds its own copy under pytest's `tmp_path`, so the tests never
touch the skill's `plans/` directory and are safe to split across xdist workers.

```bash
# Run integration tests
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_apply_suggestions_integration.py -v

# Write the same fixture tree into the skill's plans/ to inspect it
uv run --project skills/multi-llm -- python skills/multi-llm/tests/test_apply_suggestions_integration.py --setup-fixtures
```

### Integration Test Fixtures

The integration tests use a self-contained test plan at `plans/test-plan/` (per test, under `tmp_path`):

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

### Shared fixtures and helpers: `tests/support/`

A helper or fixture that more than one test file uses lives in
[`support/`](support/), never in a `test_*.py` file another test imports: a
test module changes with its own tests, and one used as a library broke its
importers silently. Each module there is one "world" (the check-gate repository,
the implement world, transcripts for the report-check hook, ...), described in
its docstring. [`support/__init__.py`](support/__init__.py) has the rules, and
`test_support_package.py` checks them:

- A module listed in `conftest.py`'s `pytest_plugins` registers its fixtures for
  every test: name them as parameters and import nothing (`repo` is the
  check-gate world's, `workdir`, `plan_dir`, `runner_env`, `stub_phases`). A
  file that defines a fixture of the same name overrides it.
- A module that is not registered defines each fixture as
  `@pytest.fixture(name="repo") def implement_repo(...)`; import the function
  name (`from tests.support.implement_world import implement_repo  # noqa:
  F401 - the repo fixture`) to get the fixture in that file. To reuse one under
  another name, go through the module (`write_orchestrators.orchestrator_repo.__wrapped__`),
  since importing the function binds its fixture name.
- [`support/gitrepo.py`](support/gitrepo.py) runs git hermetically (`git`,
  `init_repo`, `commit_file`, `make_repo_fixture`): the developer's
  `~/.gitconfig` never reaches a test repository.
- Some older test modules still re-export a moved name (a `__getattr__` at the
  top, or a `# noqa: F401` import) because a file another change is editing
  imports it from there; import it from `tests.support` instead.

### Fixture Pattern

Fixtures every test may need, and the isolation that runs for every test, are
defined in `conftest.py`:

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
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests --cov=skills/multi-llm --cov-report=html

# Open the report
open htmlcov/index.html
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, run from the repository root with the `--project` form
shown under Quick Start; `tests/conftest.py` puts the skill directory on `sys.path`.

### Stale Fixtures

If integration tests fail unexpectedly, regenerate the fixtures:

```bash
uv run --project skills/multi-llm -- python skills/multi-llm/tests/test_apply_suggestions_integration.py --setup-fixtures
```

### Plan Hash Mismatch

The state file includes a `plan_hash` that must match the actual plan content. If you modify `plans/test-plan.md`, regenerate fixtures to update the hash.

## Adding New Tests

1. **Unit tests**: Add to the appropriate `test_*.py` file or create a new one
2. **Integration tests**: Add to `test_apply_suggestions_integration.py` or create a new integration test file
3. **Fixtures**: Add a fixture or helper that several files share to a
   `support/` module (see above); `conftest.py` holds the suite-wide isolation

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
