"""Schema validation utilities for LLM outputs."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class ValidationError(Exception):
    """Raised when schema validation fails."""
    pass


def load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Load a JSON schema from the schemas directory.

    Args:
        schema_name: Name of schema file (e.g., "task_decomposition.schema.json")

    Returns:
        Schema as dictionary
    """
    schema_path = SCHEMA_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_type(value: Any, expected_type: str) -> bool:
    """Validate a value against a JSON schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    if expected_type not in type_map:
        return True  # Unknown type, assume valid

    expected = type_map[expected_type]
    return isinstance(value, expected)


def validate_against_schema(
    data: Any,
    schema: Dict[str, Any],
    path: str = ""
) -> List[str]:
    """
    Validate data against a JSON schema (simplified validation).

    Args:
        data: Data to validate
        schema: JSON schema
        path: Current path for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    # Type validation
    if "type" in schema:
        expected_type = schema["type"]
        if not validate_type(data, expected_type):
            errors.append(f"{path}: expected {expected_type}, got {type(data).__name__}")
            return errors  # Can't continue if type mismatch

    # Required fields for objects
    if schema.get("type") == "object" and isinstance(data, dict):
        for required_field in schema.get("required", []):
            if required_field not in data:
                errors.append(f"{path}: missing required field '{required_field}'")

        # Validate properties
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                prop_path = f"{path}.{prop_name}" if path else prop_name
                errors.extend(validate_against_schema(data[prop_name], prop_schema, prop_path))

    # Array items validation
    if schema.get("type") == "array" and isinstance(data, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            item_path = f"{path}[{i}]"
            errors.extend(validate_against_schema(item, items_schema, item_path))

    # Enum validation
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: value '{data}' not in enum {schema['enum']}")

    return errors


def validate_json_output(
    output: str,
    schema_name: str
) -> Tuple[bool, Union[Dict[str, Any], List[Any]], List[str]]:
    """
    Validate LLM JSON output against a schema.

    Args:
        output: Raw string output from LLM
        schema_name: Name of schema file to validate against

    Returns:
        Tuple of (is_valid, parsed_data, error_messages)
    """
    # Try to extract JSON from the output
    parsed = None
    errors = []

    # Try direct parsing
    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    if parsed is None:
        code_block_match = re.search(
            r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```',
            output
        )
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

    # Try finding JSON structure
    if parsed is None:
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, output)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    break
                except json.JSONDecodeError:
                    continue

    if parsed is None:
        return False, {}, ["Could not extract valid JSON from output"]

    # Load and validate against schema
    try:
        schema = load_schema(schema_name)
    except FileNotFoundError as e:
        return False, parsed, [str(e)]

    errors = validate_against_schema(parsed, schema)

    return len(errors) == 0, parsed, errors


def validate_task_dependencies(tasks: List[Dict[str, Any]]) -> List[str]:
    """
    Validate task dependencies form a valid DAG.

    Args:
        tasks: List of task dictionaries with 'id' and 'depends_on' fields

    Returns:
        List of validation errors
    """
    errors = []
    task_ids = {task.get("id") for task in tasks if "id" in task}

    for task in tasks:
        task_id = task.get("id")
        depends_on = task.get("depends_on", [])

        if not task_id:
            errors.append("Task missing 'id' field")
            continue

        for dep_id in depends_on:
            if dep_id not in task_ids:
                errors.append(f"Task {task_id}: dependency '{dep_id}' not found")
            if dep_id == task_id:
                errors.append(f"Task {task_id}: self-dependency")

    # Check for cycles using DFS
    visited = set()
    rec_stack = set()

    def has_cycle(task_id: Any) -> bool:
        visited.add(task_id)
        rec_stack.add(task_id)

        task = next((t for t in tasks if t.get("id") == task_id), None)
        if task:
            for dep_id in task.get("depends_on", []):
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

        rec_stack.remove(task_id)
        return False

    for task in tasks:
        task_id = task.get("id")
        if task_id and task_id not in visited:
            if has_cycle(task_id):
                errors.append(f"Circular dependency detected involving task {task_id}")

    return errors


def validate_code_review_issues(issues: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate code review issues format.

    Args:
        issues: List of issue dictionaries

    Returns:
        Tuple of (is_valid, error_messages)
    """
    is_valid, _, errors = validate_json_output(
        json.dumps(issues),
        "code_review_issues.schema.json"
    )
    return is_valid, errors


def validate_state_file(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate state file format.

    Args:
        state: State dictionary

    Returns:
        Tuple of (is_valid, error_messages)
    """
    is_valid, _, errors = validate_json_output(
        json.dumps(state),
        "state_file.schema.json"
    )
    return is_valid, errors