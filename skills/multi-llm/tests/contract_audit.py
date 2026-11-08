"""Opt-in audit: validate every result line, checkpoint write and run view a test run makes.

``tests/test_result_contract.py`` holds the schemas to the code with its own
scenarios and a source inventory; this plugin is the wide net beside it. Load it
on any selection and every ``MULTI_LLM_RESULT`` payload
(``cli_common.ResultContext.payload``), every ``.status.json`` after a write
(``status_file.write_status``) and every run view (``detach.classify_run``) the
tests produce is validated against ``schemas/``::

    MULTI_LLM_CONTRACT_AUDIT=/tmp/audit.jsonl \\
      uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_detach.py \\
      -p tests.contract_audit -q

**Warning-only by design.** It never fails a test: several suites write hostile
checkpoints on purpose (a torn ``progress.models``, a string where a number
goes) to prove the readers survive them, and those show up here as expected.
What to look for is a violation from a real writer. Each process appends one JSON
line per violation to the file the variable names (so ``-n 4`` works), and the
run's terminal summary groups them. Without the variable the plugin does nothing.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ENV = "MULTI_LLM_CONTRACT_AUDIT"
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCHEMAS = ("result_line.schema.json", "status_file.schema.json", "status_json.schema.json")

_state: Dict[str, Any] = {"test": "?", "validators": None, "restore": []}


def _report_path() -> Optional[Path]:
    value = os.environ.get(ENV)
    return Path(value) if value else None


def _validators() -> Dict[str, Any]:
    if _state["validators"] is None:
        import jsonschema
        from referencing import Registry, Resource

        loaded = {name: json.loads((SCHEMA_DIR / name).read_text("utf-8")) for name in SCHEMAS}
        registry = Registry().with_resources(
            [(name, Resource.from_contents(doc)) for name, doc in loaded.items()])
        _state["validators"] = {
            "result line": jsonschema.Draft7Validator(loaded[SCHEMAS[0]], registry=registry),
            ".status.json": jsonschema.Draft7Validator(loaded[SCHEMAS[1]], registry=registry),
            "run view": jsonschema.Draft7Validator(
                {"$ref": f"{SCHEMAS[2]}#/definitions/run_view"}, registry=registry),
        }
    return _state["validators"]


def _leaves(error) -> List[Any]:
    """The innermost errors: an ``anyOf`` miss names its alternatives' causes."""
    if not error.context:
        return [error]
    return [leaf for child in error.context for leaf in _leaves(child)]


def _audit(kind: str, document: Any) -> None:
    path = _report_path()
    if path is None:
        return
    try:
        document = json.loads(json.dumps(document, default=str))
        messages = sorted({
            f"{'/'.join(str(p) for p in leaf.absolute_path) or '<root>'}: {leaf.message[:200]}"
            for error in _validators()[kind].iter_errors(document)
            for leaf in _leaves(error)
            if not leaf.message.endswith("is not of type 'null'")
        })
        if messages:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"test": _state["test"], "kind": kind, "errors": messages}) + "\n")
    except Exception:  # noqa: BLE001 - an audit never fails the run it watches
        pass


def _wrap(owner: Any, name: str, after) -> None:
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            after(result, *args, **kwargs)
        except Exception:  # noqa: BLE001 - an audit never fails the run it watches
            pass
        return result

    setattr(owner, name, wrapper)
    _state["restore"].append((owner, name, original))


def pytest_configure(config) -> None:
    if _report_path() is None:
        return
    from utils import cli_common, detach, status_file

    _wrap(cli_common.ResultContext, "payload",
          lambda payload, *a, **k: _audit("result line", payload))
    _wrap(status_file, "write_status",
          lambda _, phase_dir, *a, **k: _audit(".status.json", status_file.read_status(phase_dir)))
    _wrap(detach, "classify_run", lambda view, *a, **k: _audit("run view", view))


def pytest_unconfigure(config) -> None:
    for owner, name, original in reversed(_state["restore"]):
        setattr(owner, name, original)
    _state["restore"].clear()


def pytest_runtest_setup(item) -> None:
    _state["test"] = item.nodeid


def pytest_terminal_summary(terminalreporter) -> None:
    path = _report_path()
    if path is None or hasattr(terminalreporter.config, "workerinput"):
        return
    grouped: Dict[tuple, set] = collections.defaultdict(set)
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            record = json.loads(line)
            for message in record["errors"]:
                grouped[(record["kind"], message)].add(record["test"].split("::")[0])
    terminalreporter.section("contract audit (warning only)")
    if not grouped:
        terminalreporter.write_line(f"no schema violations ({path})")
        return
    for (kind, message), files in sorted(grouped.items()):
        terminalreporter.write_line(f"{kind}: {message}  [{', '.join(sorted(files)[:3])}]")
    terminalreporter.write_line(f"{len(grouped)} distinct violation(s); details in {path}")
