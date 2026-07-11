"""FixtureManager class for creating isolated test directories and loading fixtures.

Manages test fixtures including plan files, response files, and scenarios.
Creates isolated test directories and validates fixtures against JSON schemas.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FixturePlan:
    """Represents a test plan fixture with all associated files.

    Attributes:
        plan_path: Path to the plan file
        output_dir: Path to the output directory (plan_name/ subdirectory)
        name: Name of the plan (without extension)
        content: The plan content (markdown)
        metadata: Additional metadata about the fixture
    """

    plan_path: Path
    output_dir: Path
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_state_path(self) -> Path:
        """Get the path to the state.json file."""
        return self.output_dir / "state.json"

    def get_phase_dir(self, phase: str) -> Path:
        """Get the directory for a specific phase's outputs.

        Args:
            phase: Phase name (e.g., "review-plan", "apply-suggestions")

        Returns:
            Path to the phase output directory
        """
        return self.output_dir / phase

    def ensure_phase_dir(self, phase: str) -> Path:
        """Ensure a phase directory exists and return its path.

        Args:
            phase: Phase name (e.g., "review-plan", "apply-suggestions")

        Returns:
            Path to the created phase directory
        """
        phase_dir = self.get_phase_dir(phase)
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir


class FixtureManager:
    """Manages test fixtures for end-to-end tests.

    Creates isolated test directories, loads fixture files, and validates
    fixtures against JSON schemas.

    Usage:
        manager = FixtureManager(tmp_path)
        plan = manager.create_plan("my-plan", "# Plan content...")
        # or
        plan = manager.load_plan("auth-feature")  # loads from fixtures/e2e/plans/
    """

    # Paths relative to the test directory
    E2E_FIXTURES_DIR = "fixtures/e2e"
    PLANS_SUBDIR = "plans"
    RESPONSES_SUBDIR = "responses"
    SCENARIOS_SUBDIR = "scenarios"
    SCHEMAS_DIR = "schemas"

    def __init__(
        self,
        tmp_path: Path,
        fixtures_base: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        validate_on_load: bool = True,
    ):
        """Initialize FixtureManager.

        Args:
            tmp_path: Temporary directory for this test (from pytest tmp_path fixture)
            fixtures_base: Base path for fixture files. If None, auto-detected from
                tests/fixtures relative to this file.
            skill_dir: Path to the skill directory for schema validation.
                If None, auto-detected relative to this file.
            validate_on_load: Whether to validate fixtures against schemas on load.
        """
        self.tmp_path = Path(tmp_path)
        self.validate_on_load = validate_on_load

        # Auto-detect fixtures base path
        if fixtures_base is None:
            # Relative to this file: ../fixtures
            self.fixtures_base = Path(__file__).parent.parent / "fixtures"
        else:
            self.fixtures_base = Path(fixtures_base)

        # Auto-detect skill directory
        if skill_dir is None:
            # Relative to this file: ../.. (up from tests/harness to skill root)
            self.skill_dir = Path(__file__).parent.parent.parent
        else:
            self.skill_dir = Path(skill_dir)

        # Paths to fixture subdirectories
        self.e2e_dir = self.fixtures_base / "e2e"
        self.plans_dir = self.e2e_dir / self.PLANS_SUBDIR
        self.responses_dir = self.e2e_dir / self.RESPONSES_SUBDIR
        self.scenarios_dir = self.e2e_dir / self.SCENARIOS_SUBDIR
        self.schemas_dir = self.skill_dir / self.SCHEMAS_DIR

        # Create tmp_path structure
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        # Make the fixture dir a git work tree: real plans always live inside a
        # repository, and implement_orchestrator's default --output fails fast
        # when the plan is outside one (its default path anchors at the git root).
        if not (self.tmp_path / ".git").exists():
            subprocess.run(
                ["git", "init", "-q"],
                cwd=str(self.tmp_path),
                check=True,
                capture_output=True,
            )

    def create_plan(
        self,
        name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FixturePlan:
        """Create a plan file in the test directory.

        Args:
            name: Name of the plan (without .md extension)
            content: Markdown content for the plan
            metadata: Optional metadata to associate with the fixture

        Returns:
            FixturePlan with paths to created files
        """
        # Sanitize name
        safe_name = name.replace("/", "_").replace("\\", "_")

        # Create plan file in tmp_path
        plan_path = self.tmp_path / f"{safe_name}.md"
        plan_path.write_text(content, encoding="utf-8")

        # Create output directory
        output_dir = self.tmp_path / safe_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return FixturePlan(
            plan_path=plan_path,
            output_dir=output_dir,
            name=safe_name,
            content=content,
            metadata=metadata or {},
        )

    def load_plan(self, name: str) -> FixturePlan:
        """Load a plan from the fixtures/e2e/plans/ directory.

        Args:
            name: Name of the plan file (without .md extension)

        Returns:
            FixturePlan with the plan copied to tmp_path

        Raises:
            FileNotFoundError: If the plan fixture doesn't exist
        """
        source_path = self.plans_dir / f"{name}.md"
        if not source_path.exists():
            raise FileNotFoundError(
                f"Plan fixture not found: {source_path}. "
                f"Available plans: {self._list_available_plans()}"
            )

        content = source_path.read_text(encoding="utf-8")
        return self.create_plan(name, content, metadata={"source": str(source_path)})

    def _list_available_plans(self) -> List[str]:
        """List available plan fixtures."""
        if not self.plans_dir.exists():
            return []
        return [p.stem for p in self.plans_dir.glob("*.md")]

    def load_response(
        self,
        phase: str,
        name: str,
        validate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Load a response fixture from fixtures/e2e/responses/{phase}/.

        Args:
            phase: Phase name (e.g., "review_plan", "validation")
            name: Response file name (without .json extension)
            validate: Whether to validate against schema. If None, uses
                the instance's validate_on_load setting.

        Returns:
            Parsed JSON data from the response file

        Raises:
            FileNotFoundError: If the response fixture doesn't exist
            ValueError: If validation fails (when validation is enabled)
        """
        response_path = self.responses_dir / phase / f"{name}.json"
        if not response_path.exists():
            raise FileNotFoundError(
                f"Response fixture not found: {response_path}. "
                f"Available responses in {phase}: {self._list_available_responses(phase)}"
            )

        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate if requested
        should_validate = validate if validate is not None else self.validate_on_load
        if should_validate:
            self._validate_response(phase, data, response_path)

        return data

    def _list_available_responses(self, phase: str) -> List[str]:
        """List available response fixtures for a phase."""
        phase_dir = self.responses_dir / phase
        if not phase_dir.exists():
            return []
        return [p.stem for p in phase_dir.glob("*.json")]

    def _validate_response(
        self,
        phase: str,
        data: Any,
        source_path: Path,
    ) -> None:
        """Validate a response against its schema.

        Args:
            phase: Phase name to determine which schema to use
            data: The data to validate
            source_path: Path to the source file (for error messages)

        Raises:
            ValueError: If validation fails
        """
        # Map phases to schema files
        schema_map = {
            "review_plan": None,  # No schema yet for review suggestions
            "validation": None,  # Validation results don't have a schema
            "code_review": "code_review_issues.schema.json",
            "generate_tasks": "task_decomposition.schema.json",
        }

        schema_name = schema_map.get(phase)
        if schema_name is None:
            # No schema for this phase, skip validation
            return

        schema_path = self.schemas_dir / schema_name
        if not schema_path.exists():
            # Schema doesn't exist, skip validation
            return

        try:
            import jsonschema

            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            jsonschema.validate(data, schema)
        except ImportError:
            # jsonschema not installed, skip validation
            pass
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"Response fixture validation failed for {source_path}: {e.message}"
            ) from e

    def load_scenario(self, name: str) -> Path:
        """Load a scenario file path from fixtures/e2e/scenarios/.

        Args:
            name: Scenario file name (without .yaml extension)

        Returns:
            Path to the scenario file

        Raises:
            FileNotFoundError: If the scenario doesn't exist
        """
        scenario_path = self.scenarios_dir / f"{name}.yaml"
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario not found: {scenario_path}. "
                f"Available scenarios: {self._list_available_scenarios()}"
            )
        return scenario_path

    def _list_available_scenarios(self) -> List[str]:
        """List available scenario files."""
        if not self.scenarios_dir.exists():
            return []
        return [p.stem for p in self.scenarios_dir.glob("*.yaml")]

    def create_with_review_phase(
        self,
        name: str,
        plan_content: str,
        suggestions: Union[List[Dict[str, Any]], Path, str],
        validation: Optional[Union[List[Dict[str, Any]], Path, str]] = None,
    ) -> FixturePlan:
        """Create a plan fixture with pre-populated review phase outputs.

        This is useful for testing phases that depend on review-plan output
        (like apply-suggestions) without running the full review-plan phase.

        Args:
            name: Name of the plan
            plan_content: Markdown content for the plan
            suggestions: Either a list of suggestion groups, a path to a JSON file,
                or the name of a response fixture (e.g., "valid_suggestions")
            validation: Optional validation results in the same format as suggestions

        Returns:
            FixturePlan with pre-populated review-plan/ directory
        """
        # Create base plan
        fixture = self.create_plan(name, plan_content)

        # Create review-plan directory
        review_dir = fixture.ensure_phase_dir("review-plan")

        # Load suggestions
        suggestions_data = self._resolve_data(suggestions, "review_plan")
        grouped_path = review_dir / "grouped.json"
        with open(grouped_path, "w", encoding="utf-8") as f:
            json.dump(suggestions_data, f, indent=2)

        # Load validation if provided
        if validation is not None:
            validation_data = self._resolve_data(validation, "validation")
            validation_path = review_dir / "validation.json"
            with open(validation_path, "w", encoding="utf-8") as f:
                json.dump(validation_data, f, indent=2)

        return fixture

    def create_with_tasks(
        self,
        name: str,
        plan_content: str,
        tasks: Union[List[Dict[str, Any]], Path, str],
    ) -> FixturePlan:
        """Create a plan fixture with pre-populated task outputs.

        This is useful for testing phases that depend on tasks
        (like implement) without running generate-tasks.

        Args:
            name: Name of the plan
            plan_content: Markdown content for the plan
            tasks: Either a list of tasks, a path to a JSON file,
                or the name of a response fixture (e.g., "tasks")

        Returns:
            FixturePlan with pre-populated tasks/ directory
        """
        # Create base plan
        fixture = self.create_plan(name, plan_content)

        # Create tasks directory
        tasks_dir = fixture.ensure_phase_dir("tasks")

        # Load tasks
        tasks_data = self._resolve_data(tasks, "generate_tasks")
        tasks_path = tasks_dir / "tasks.json"
        with open(tasks_path, "w", encoding="utf-8") as f:
            json.dump(tasks_data, f, indent=2)

        return fixture

    def _resolve_data(
        self,
        data: Union[List[Dict[str, Any]], Path, str],
        default_phase: str,
    ) -> Any:
        """Resolve data from various input formats.

        Args:
            data: Either raw data, a Path, or a fixture name string
            default_phase: Default phase to use when loading by fixture name

        Returns:
            The resolved data
        """
        if isinstance(data, list):
            return data
        if isinstance(data, Path):
            with open(data, "r", encoding="utf-8") as f:
                return json.load(f)
        if isinstance(data, str):
            # Treat as fixture name
            return self.load_response(default_phase, data, validate=False)
        raise TypeError(f"Invalid data type: {type(data)}")

    def copy_fixture_to_tmp(self, relative_path: str) -> Path:
        """Copy a fixture file to tmp_path.

        Args:
            relative_path: Path relative to fixtures_base

        Returns:
            Path to the copied file in tmp_path
        """
        source = self.fixtures_base / relative_path
        dest = self.tmp_path / Path(relative_path).name

        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)

        return dest

    def create_state_file(
        self,
        fixture: FixturePlan,
        phases_completed: Optional[List[str]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create a state.json file for a fixture.

        Args:
            fixture: The FixturePlan to create state for
            phases_completed: List of phase names to mark as completed
            extra_state: Additional state fields to include

        Returns:
            Path to the created state.json file
        """
        import hashlib
        from datetime import datetime

        # Compute plan hash
        plan_hash = hashlib.sha256(fixture.content.encode()).hexdigest()[:16]

        state = {
            "schema_version": "1.0",
            "plan_path": str(fixture.plan_path),
            "plan_hash": plan_hash,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "head_at_start": "test-commit-hash",
            "branch_name": "test-branch",
            "review_phase_completed": False,
            "tracked_files": [],
            "task_status": {},
            "phases_completed": {},
            "phases_skipped": {},
        }

        # Mark phases as completed
        if phases_completed:
            for phase in phases_completed:
                state["phases_completed"][phase] = datetime.now().isoformat()

            # Also set review_phase_completed if review-plan is in the list
            if "review-plan" in phases_completed:
                state["review_phase_completed"] = True

        # Merge extra state
        if extra_state:
            state.update(extra_state)

        # Write state file
        state_path = fixture.get_state_path()
        fixture.output_dir.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return state_path
