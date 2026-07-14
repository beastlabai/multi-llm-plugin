"""Guard test: every CLI entry point must bootstrap its stdout/stderr.

Any module with an ``if __name__ == "__main__"`` block can be invoked as a
process (the instruction files run several of them directly, e.g.
``python "${CLAUDE_SKILL_DIR}/utils/metrics.py" record ...``). On Windows an
unbootstrapped process writes its stdout/stderr with the locale codec (cp1252),
so a non-cp1252 character (the ETA line's ``·``, em dashes, model output) is
either mangled into bytes that are not valid UTF-8 or raises UnicodeEncodeError
outright — breaking any parent that captures the output as UTF-8.

The checks are AST-based rather than substring greps so a mention of
``bootstrap_streams()`` in a comment or docstring cannot satisfy them.
"""

import ast
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent

BOOTSTRAP = "bootstrap_streams"

# Directories that hold no first-party runtime code.
_EXCLUDED_DIRS = {"tests", "__pycache__", "node_modules", "build", "dist"}

_FIX_HINT = (
    "Add the stream bootstrap to this entry point:\n"
    "    try:\n"
    "        from .stream_bootstrap import bootstrap_streams\n"
    "    except ImportError:  # direct script invocation\n"
    "        from stream_bootstrap import bootstrap_streams\n"
    "(top-level scripts use `from utils.stream_bootstrap import "
    "bootstrap_streams`)\n"
    "and call `bootstrap_streams()` as the first statement of main()."
)


def _is_excluded(path: Path) -> bool:
    rel_parts = path.relative_to(SKILL_ROOT).parts
    return any(
        part in _EXCLUDED_DIRS or part.startswith(".") for part in rel_parts
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_main_guard(node: ast.AST) -> bool:
    """True for an `if __name__ == "__main__":` statement."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    names = {
        n.id for n in (left, right) if isinstance(n, ast.Name)
    }
    consts = {
        n.value for n in (left, right) if isinstance(n, ast.Constant)
    }
    return "__name__" in names and "__main__" in consts


def _has_main_guard(tree: ast.Module) -> bool:
    return any(_is_main_guard(node) for node in ast.walk(tree))


def _bootstrap_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == BOOTSTRAP
    ]


def _import_time_bootstrap_calls(node: ast.AST) -> list[ast.Call]:
    """bootstrap_streams() calls that would run on plain `import module`.

    Descends the module body but skips function bodies and the __main__
    guard — the only two places the call may legally live.
    """
    found: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_main_guard(child):
            continue
        if isinstance(child, ast.Call) and _call_name(child) == BOOTSTRAP:
            found.append(child)
        found.extend(_import_time_bootstrap_calls(child))
    return found


def _source_files() -> list[Path]:
    return sorted(
        p for p in SKILL_ROOT.rglob("*.py") if not _is_excluded(p)
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _entry_points() -> list[Path]:
    return [p for p in _source_files() if _has_main_guard(_parse(p))]


def _rel(path: Path) -> str:
    return path.relative_to(SKILL_ROOT).as_posix()


ENTRY_POINTS = _entry_points()
SOURCE_FILES = _source_files()


class TestEntryPointsBootstrapStreams:
    """Every runnable script calls bootstrap_streams(); no module does at import."""

    def test_discovery_is_not_vacuous(self):
        """The globbing must actually find the known entry points.

        Without this, a broken discovery would make the parametrized test below
        pass with zero cases.
        """
        found = {_rel(p) for p in ENTRY_POINTS}
        expected = {
            "utils/metrics.py",
            "utils/backup.py",
            "utils/prior_changes.py",
            "utils/html_report_generator.py",
            "review_plan_orchestrator.py",
            "code_review_orchestrator.py",
            "implement_orchestrator.py",
        }
        missing = expected - found
        assert not missing, f"entry-point discovery missed: {sorted(missing)}"
        assert len(found) >= 10, f"suspiciously few entry points found: {found}"

    @pytest.mark.parametrize("path", ENTRY_POINTS, ids=_rel)
    def test_entry_point_calls_bootstrap_streams(self, path: Path):
        """A module runnable as a script must bootstrap its streams."""
        tree = _parse(path)
        calls = _bootstrap_calls(tree)
        assert calls, (
            f"{_rel(path)} has an `if __name__ == \"__main__\"` block but never "
            f"calls {BOOTSTRAP}(). Its stdout/stderr would use the Windows "
            f"locale codec (cp1252), emitting bytes that are not valid UTF-8 "
            f"(e.g. the '·' separator) and breaking UTF-8 capture in the "
            f"parent process.\n{_FIX_HINT}"
        )

    @pytest.mark.parametrize("path", SOURCE_FILES, ids=_rel)
    def test_bootstrap_is_not_called_at_import_time(self, path: Path):
        """The bootstrap must only run on the CLI path, never on import.

        These modules double as libraries (utils.metrics is imported by the
        orchestrators and the tests). Calling bootstrap_streams() at module
        scope would reconfigure the streams of every importing process —
        including pytest, breaking its capture.
        """
        offenders = _import_time_bootstrap_calls(_parse(path))
        assert not offenders, (
            f"{_rel(path)} calls {BOOTSTRAP}() at module import time "
            f"(line {offenders[0].lineno}). Move the call inside main() so it "
            f"only runs when the module is executed as a script."
        )
