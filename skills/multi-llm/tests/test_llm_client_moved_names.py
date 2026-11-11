"""The contract for the names `utils.llm_client` re-exports from
`utils.prompt_channels`, the prompt text and diff-access rendering split out
of it (docs/development/providers.md#provider-output-channels):

* **Reads work.** `llm_client.X`, `from utils.llm_client import X` and `dir()`
  see every moved name, as the owner's current object.
* **Writes do not.** A patch or assignment of a moved name on `llm_client` only
  shadows it there and reaches no code. The guard at the end fails any test
  that writes one, naming the owner to patch.
* **The table is the whole module**: every top-level name of
  `prompt_channels` is listed, and `llm_client` defines none of them itself.
* **`prompt_channels` spawns nothing** and never imports `llm_client`.
"""

import ast
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils import llm_client, prompt_channels  # noqa: E402

OWNER_SOURCE = SKILL_ROOT / "utils" / "prompt_channels.py"


def _top_level_definitions(path):
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_every_name_prompt_channels_defines_is_re_exported():
    assert set(llm_client._MOVED) == {"prompt_channels"}
    assert set(llm_client._MOVED_TO) == _top_level_definitions(OWNER_SOURCE)


@pytest.mark.parametrize("name", sorted(llm_client._MOVED_TO))
def test_a_moved_name_reads_as_the_owners_object(name):
    assert name not in vars(llm_client), f"llm_client still defines {name} itself"
    assert getattr(llm_client, name) is getattr(prompt_channels, name)
    assert name in dir(llm_client)
    namespace = {}
    exec(f"from utils.llm_client import {name} as found", namespace)
    assert namespace["found"] is getattr(prompt_channels, name)


def test_a_read_sees_a_patch_of_the_owner(monkeypatch):
    marker = object()
    monkeypatch.setattr(prompt_channels, "STDOUT_OUTPUT_LOCATION", marker)
    assert llm_client.STDOUT_OUTPUT_LOCATION is marker


def test_an_unknown_name_still_raises():
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        llm_client.not_a_name  # noqa: B018


def test_the_file_output_prompt_renders_through_the_owner(monkeypatch, tmp_path):
    """`invoke_with_file_output`'s context reads the blocks from the owner, so a
    patch there reaches the dispatched prompt."""
    monkeypatch.setattr(prompt_channels, "render_working_limits", lambda _s: "LIMITS")
    monkeypatch.setattr(
        prompt_channels, "render_output_location", lambda *_a, **_k: "LOCATION"
    )
    context, _ = llm_client._file_output_context(
        "codex:gpt-5", {"prefix": "p"}, tmp_path, "code-review", 600, None,
    )
    assert context["working_limits"] == "LIMITS"
    assert context["output_location"] == "LOCATION"


def test_prompt_channels_spawns_nothing_and_never_imports_llm_client():
    tree = ast.parse(OWNER_SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert not imported & {"subprocess", ".llm_client", "utils.llm_client"}, imported


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on llm_client
# ---------------------------------------------------------------------------


def _guard_message(path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.llm_client, but {name} "
        f"lives in utils/{owner}.py and llm_client only re-exports it for reading: "
        f"the code that uses it never sees this patch. Patch utils.{owner}.{name} "
        f"instead."
    )


def test_no_test_writes_a_moved_name_on_llm_client():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(llm_client._MOVED_TO, "llm_client")
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(encoding="utf-8"), str(path))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom utils import llm_client\n"


class TestTheGuardSeesLlmClient:
    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(llm_client, "render_working_limits", None)',
         "render_working_limits"),
        ('monkeypatch.setattr("utils.llm_client.render_diff_access", None)',
         "render_diff_access"),
        ('from utils import llm_client as lc\nmock.patch.object(lc, "_lies_within")',
         "_lies_within"),
        ('import utils.llm_client as client\nclient.INLINE_DIFF_CAP_BYTES = 1',
         "INLINE_DIFF_CAP_BYTES"),
        ('mock.patch(target="utils.llm_client.STDOUT_OUTPUT_LOCATION")',
         "STDOUT_OUTPUT_LOCATION"),
    ])
    def test_a_write(self, snippet, name):
        hits = MovedNameWriteScanner(llm_client._MOVED_TO, "llm_client").scan(
            _IMPORTS + snippet
        )
        assert [hit[2] for hit in hits] == [name]
        assert (
            f"Patch utils.prompt_channels.{name} instead"
            in _guard_message("t.py", *hits[0])
        )

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(llm_client, "get_provider", None)',
        'monkeypatch.setattr(prompt_channels, "render_diff_access", None)',
        'found = llm_client.render_working_limits(60)',
    ])
    def test_not_a_write(self, snippet):
        assert MovedNameWriteScanner(llm_client._MOVED_TO, "llm_client").scan(
            _IMPORTS + snippet
        ) == []
