"""The codex plugin: its build, its manifests, its catalog entry, its thin skill.

Scope: the build-and-ship half of ``plans/codex-native-plugin.md`` (D1, D1a, D6,
D7, D9, A2, A2a, A2b). The stdio server's behaviour is graded in
``test_codex_mcp_server.py``.

**Three input classes**, and every row says which it reads:

* **(src)** — the committed sources: ``packaging/codex/{plugin.json.tmpl,
  mcp.json, ingestion_rules.py, build.py}``, the skill template, and the
  in-process renders ``build.py`` exposes.
* **(tree)** — the **checked-in** ``packaging/codex/multi-llm/``. Unlike the npm
  package this tree is committed (the Git marketplace route serves it with no
  build), so a row may read it on a clean checkout.
* **(build)** — ``build.py``'s output in a temp directory, from the session
  fixture ``built_plugin`` (dirty inputs allowed, so it grades the working
  tree). Needs git and nothing else, so it runs in plain CI like the opencode
  suite's (build) rows. Rows that also need ``uv`` or ``make`` carry the
  ``package`` marker and run in the Linux package job.
"""

from __future__ import annotations

import ast
import datetime
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
CODEX_PKG = REPO_ROOT / "packaging" / "codex"
PLUGIN_DIR = CODEX_PKG / "multi-llm"
PAYLOAD = "skills/multi-llm"
AGENTS_CATALOG = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
SCHEMA_DIR = SKILL_ROOT / "tests" / "fixtures" / "codex" / "schemas"

for _path in (str(SKILL_ROOT), str(REPO_ROOT / "packaging"), str(CODEX_PKG)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import modes as mode_table  # noqa: E402
from utils import cli_common, host_install, unattended  # noqa: E402

from common import payload  # noqa: E402

# Imported under its own name: the opencode suite imports *its* build.py as
# ``build`` from a different directory, and one pytest session holds both.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "codex_build", str(CODEX_PKG / "build.py")
)
codex_build = importlib.util.module_from_spec(_spec)
sys.modules["codex_build"] = codex_build
_spec.loader.exec_module(codex_build)
import ingestion_rules  # noqa: E402

#: Set by the package job: a row that would skip fails instead.
REQUIRE_ROWS_ENV = "MLLM_REQUIRE_PACKAGE_ROWS"


def _unavailable(reason: str) -> None:
    if os.environ.get(REQUIRE_ROWS_ENV):
        pytest.fail(f"{REQUIRE_ROWS_ENV} is set but this row cannot run: {reason}")
    pytest.skip(reason)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _files(tree: Path) -> dict:
    """``{relative: path}`` for every shipped file — by-products excluded."""
    found = {}
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(tree).as_posix()
        if codex_build._is_by_product(relative):  # noqa: SLF001
            continue
        found[relative] = path
    return found


@pytest.fixture(scope="session")
def built_plugin(tmp_path_factory) -> Path:
    """``build.py``'s output, in a temp directory. Never the checked-in tree."""
    if not codex_build.SERVER_SOURCE.is_file():
        pytest.fail(f"{codex_build.SERVER_SOURCE} is missing; the plugin cannot build")
    destination = tmp_path_factory.mktemp("codex-plugin") / "multi-llm"
    codex_build.build(destination=destination, allow_dirty=True)
    return destination


def _frontmatter(text: str) -> dict:
    import yaml

    assert text.startswith("---\n"), "no frontmatter"
    end = text.index("\n---", 4)
    return yaml.safe_load(text[4:end])


def _body(text: str) -> str:
    end = text.index("\n---", 4)
    return text[end + 4:]


# ===========================================================================
# Manifest (A2) and the restated ingestion rules
# ===========================================================================

class TestManifest:
    def test_plugin_manifest_valid(self):
        """(src + tree) R2's key set only; no ``hooks``; every required
        ``interface`` field; ``author.name``; strict semver; ``skills`` and
        ``mcpServers`` at their fixed conventions — restated, not copied, from
        ``validate_plugin.py``.

        **The schema clause is not claimed** until the vendored
        ``plugin.schema.json`` is committed (see ``test_schema_vendoring_state``
        and ``tests/fixtures/codex/schemas/provenance.json``): until then this
        row stands on the hand-restated assertions below alone.
        """
        for manifest in (
            json.loads(codex_build.manifest_text()),
            _load(PLUGIN_DIR / ".codex-plugin" / "plugin.json"),
        ):
            assert ingestion_rules.validate_manifest(manifest, PLUGIN_DIR) == []
            assert set(manifest) <= ingestion_rules.ACCEPTED_KEYS
            assert "hooks" not in manifest
            assert "id" not in manifest, "A2 ships no id (optional, R2)"
            for key in ingestion_rules.REQUIRED_KEYS:
                assert key in manifest, key
            assert manifest["author"]["name"] == "beastlabai"
            assert ingestion_rules.SEMVER_RE.fullmatch(manifest["version"])
            assert "+" not in manifest["version"], (
                "a +codex.<n> build suffix is the dev loop (R48) and is never "
                "committed"
            )
            assert ingestion_rules.normalize_contract_path(manifest["skills"]) == "skills"
            assert (
                ingestion_rules.normalize_contract_path(manifest["mcpServers"])
                == ".mcp.json"
            )
            interface = manifest["interface"]
            for key in ingestion_rules.INTERFACE_REQUIRED_STRINGS:
                assert isinstance(interface[key], str) and interface[key].strip(), key
            assert interface["displayName"] == "Multi-LLM"
            assert isinstance(interface["capabilities"], list)
            assert interface.get("defaultPrompt") or interface.get("default_prompt")
            # Asset paths: vacuous until A5, live the moment one is declared.
            for key in ingestion_rules.INTERFACE_ASSET_KEYS:
                if key in interface:
                    assert (PLUGIN_DIR / interface[key]).is_file(), key

    def test_the_restated_rules_reject_what_upstream_rejects(self):
        """(src) The checker is not a tautology: each rule fires on its case."""
        good = json.loads(codex_build.manifest_text())

        def broken(**changes):
            manifest = json.loads(json.dumps(good))
            for key, value in changes.items():
                if value is None:
                    manifest.pop(key, None)
                else:
                    manifest[key] = value
            return ingestion_rules.validate_manifest(manifest)

        assert any("hooks" in e for e in broken(hooks="./hooks.json"))
        assert any("`extra`" in e for e in broken(extra=1))
        assert any("version" in e for e in broken(version="1.0"))
        assert broken(version="0.1.0+codex.3") == [], "R48: +build is accepted"
        assert any("author" in e for e in broken(author={"email": "x@y"}))
        assert any("skills" in e for e in broken(skills="./other/"))
        assert any("mcpServers" in e for e in broken(mcpServers="./servers.json"))
        interface = dict(good["interface"])
        interface.pop("category")
        assert any("category" in e for e in broken(interface=interface))
        interface = dict(good["interface"])
        interface.pop("defaultPrompt")
        assert any("defaultPrompt" in e for e in broken(interface=interface))
        interface = dict(good["interface"], default_prompt=["x"])
        interface.pop("defaultPrompt")
        assert broken(interface=interface) == [], "R48: snake_case alias accepted"

    def test_the_manifest_template_fills_only_the_version(self):
        """(src) D1's mapping row: ``version`` filled, nothing else rewritten."""
        template = codex_build.MANIFEST_TEMPLATE.read_text(encoding="utf-8")
        assert template.count(codex_build.VERSION_SLOT) == 1
        rendered = codex_build.manifest_text("9.9.9")
        assert rendered == template.replace(codex_build.VERSION_SLOT, "9.9.9")

    def test_ingestion_rules_declare_provenance(self):
        """(src) The four stamp constants, and the stamp is not behind the pin.

        ``SOURCE_CODEX_VERSION`` must be at or above the container's pinned
        codex (``tests/container/package.json``): a pin bump that skipped the
        re-sync goes red here rather than waiting on a container run.
        """
        assert ingestion_rules.SEMVER_RE.fullmatch(ingestion_rules.SOURCE_CODEX_VERSION)
        assert ingestion_rules.SOURCE_PATH.endswith(
            "plugin-creator/scripts/validate_plugin.py"
        )
        assert not ingestion_rules.SOURCE_PATH.startswith("/"), (
            "SOURCE_PATH is $CODEX_HOME-relative; an absolute path is a build "
            "machine's"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", ingestion_rules.SOURCE_SHA256)
        datetime.date.fromisoformat(ingestion_rules.SOURCE_READ_DATE)

        pins = _load(SKILL_ROOT / "tests" / "container" / "package.json")
        pinned = pins["dependencies"]["@openai/codex"]
        assert codex_build._version_key(  # noqa: SLF001
            ingestion_rules.SOURCE_CODEX_VERSION
        ) >= codex_build._version_key(pinned), (  # noqa: SLF001
            f"the ingestion rules were restated against codex "
            f"{ingestion_rules.SOURCE_CODEX_VERSION}, but the container pins "
            f"{pinned}. Re-read validate_plugin.py, fix the rules, re-stamp."
        )

    def test_the_server_floor_has_one_owner(self):
        """(src) ``MIN_PYTHON`` is owned by ``ingestion_rules``; every copy agrees."""
        assert ingestion_rules.MIN_PYTHON == (3, 9)
        assert host_install.CODEX_SERVER_MIN_PYTHON == ingestion_rules.MIN_PYTHON
        floor = ".".join(str(part) for part in ingestion_rules.MIN_PYTHON)
        assert f"`python3 >= {floor}`" in codex_build.skill_md()

    def test_schema_vendoring_state(self):
        """(src) The vendoring note's state is written down, never silent.

        Either both schemas are committed with a matching stamp — each file's
        sha256 equal to the recorded one, each recorded codex version at or
        above the ingestion-rules floor — or ``provenance.json`` says they are
        not vendored and why, and **no** schema file sits here unstamped. A
        missing tracked schema is red, never a skip.
        """
        import hashlib

        provenance = _load(SCHEMA_DIR / "provenance.json")
        files = provenance["files"]
        assert set(files) == {"plugin.schema.json", "mcp.schema.json"}
        if provenance["status"] == "not_vendored":
            assert provenance["reason"].strip()
            for name, record in files.items():
                assert record["vendored"] is False
                assert not (SCHEMA_DIR / name).exists(), (
                    f"{name} is on disk but provenance.json says it is not "
                    "vendored; stamp it"
                )
            return
        assert provenance["status"] == "vendored"
        for name, record in files.items():
            path = SCHEMA_DIR / name
            assert path.is_file(), f"{path} is tracked by provenance.json and missing"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            assert codex_build._version_key(record["codex_version"]) >= (  # noqa: SLF001
                codex_build._version_key(ingestion_rules.SOURCE_CODEX_VERSION)  # noqa: SLF001
            )
            datetime.date.fromisoformat(record["fetched"])

    def test_mcp_json_shape(self):
        """(src + tree) A2b exactly: one server, R39's contained forms, D9's
        allowlist, ``tool_timeout_sec`` set, the tool pre-approved — and the
        checked-in copy verbatim.

        ``default_tools_approval_mode: approve`` is deliberate (the user's
        decision, 2026-09-24): without it codex refuses the call outright under
        ``approval_policy = "never"`` with a sandbox, measured on 0.156.1 by
        rows 1b(a) and 12.

        The schema clause is not claimed (``test_schema_vendoring_state``).
        """
        source = _load(codex_build.MCP_SOURCE)
        assert ingestion_rules.validate_mcp(source) == []
        assert set(source) == {"mcpServers"}
        assert list(source["mcpServers"]) == ["multi_llm"]
        server = source["mcpServers"]["multi_llm"]
        assert set(server) == {
            "command", "args", "cwd", "env_vars", "tool_timeout_sec",
            "default_tools_approval_mode",
        }
        assert server["default_tools_approval_mode"] == "approve"
        assert server["command"] == "python3"
        assert "/" not in server["command"], "a bare executable name (R39)"
        assert server["args"] == ["./mcp/server.py"]
        assert server["cwd"] in (".", "${PLUGIN_ROOT}", "${PLUGIN_DATA}")
        assert server["tool_timeout_sec"] == 900
        assert server["env_vars"] == list(A2B_ENV_VARS)
        assert (PLUGIN_DIR / ".mcp.json").read_bytes() == (
            codex_build.MCP_SOURCE.read_bytes()
        )
        # No user credential is ever forwarded by name.
        for name in server["env_vars"]:
            assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD", name), name


# ===========================================================================
# D9 — the environment allowlist, discovered from the tree
# ===========================================================================

#: A2b's list, in A2b's order.
A2B_ENV_VARS = (
    "CODEX_HOME",
    "MULTI_LLM_PROVIDERS_CONFIG", "MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE",
    "MULTI_LLM_GIT_CREDENTIAL_HELPER", "MULTI_LLM_HOST", "MULTI_LLM_UV",
    "MULTI_LLM_UV_RUN_ARGS", "MULTI_LLM_SKILL_DIR", "MULTI_LLM_MAX_DETACHED",
    "MULTI_LLM_HEARTBEAT_INTERVAL_S", "MULTI_LLM_HANDSHAKE_TIMEOUT_S",
    "MULTI_LLM_REPLAY_FLAGS", "MULTI_LLM_LEGACY_INSTRUCTIONS",
    "MULTI_LLM_CODEX_UNSANDBOXED_WRITES",
    "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_OFFLINE",
    "PWD",
)

#: The non-``MULTI_LLM_*`` remainder, declared (D9). ``PWD`` is D3f's rung 2:
#: codex's own working directory, forwarded so the env rung can answer at all.
NON_MULTI_LLM_FORWARDED = frozenset({
    "CODEX_HOME", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_OFFLINE", "PWD",
})

#: Real reads deliberately left out of the allowlist (D9).
NOT_FORWARDED = frozenset({
    # The server sets it per call from the turn metadata; a forwarded one would outlive the turn.
    "MULTI_LLM_HOST_SESSION_MODEL",
    # The same rule for the chat's sandbox posture (plan codex-sandbox-unavailable
    # step 6, F5, 2026-09-30): set or removed on every call, never inherited —
    # a forwarded one would hand F5's consent to a chat that never gave it.
    "MULTI_LLM_HOST_SANDBOX_POSTURE",
    # A test switch, but read by shipped code, so the walk finds it.
    "MULTI_LLM_TEST_FAST_BACKOFF",
    # The parent->worker handshake; an inherited stale one is a bug.
    "MULTI_LLM_DETACHED_WORKER",
})

#: Protocol tokens and a comment-only mention — never environment reads, and
#: asserted ABSENT from the walk so a real read of one fails loudly instead of
#: being swept into the allowlist.
NOT_ENVIRONMENT = frozenset({
    "MULTI_LLM_RESULT", "MULTI_LLM_NESTED_EXEC_OK", "MULTI_LLM_DEBUG",
})

#: Files allowed to read an environment name the walk cannot resolve to a
#: literal, each with its reason. A new file here is a decision, not a drift.
DYNAMIC_READ_FILES = {
    # `--question-env NAME`: the user names the variable per call.
    "ask_orchestrator.py": "a per-call variable named on the command line",
    # iterates a tuple of cursor's own profile variables (none MULTI_LLM_*).
    "cursor_agent.py": "cursor's profile variables, iterated",
    # `_truthy_env(name)`: every caller passes a module constant the walk
    # already finds at its own read site.
    "provider_registry.py": "a helper whose callers pass named constants",
    # names read out of the generated surface (D3e), never spelled.
    "mcp_server.py": "names read from dist/cli-surface.json",
}


def _shipped_python() -> list:
    files = [
        REPO_ROOT / relative
        for relative in payload.selected_files(REPO_ROOT, PAYLOAD)
        if relative.endswith(".py")
    ]
    files.append(codex_build.SERVER_SOURCE)
    return files


def _module_constants(tree: ast.Module) -> dict:
    found = {}
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = value.value
    return found


def _is_environ(node) -> bool:
    return (
        isinstance(node, ast.Attribute) and node.attr == "environ"
        and isinstance(node.value, ast.Name) and node.value.id == "os"
    ) or (isinstance(node, ast.Name) and node.id == "environ")


def _env_read_keys(tree: ast.Module):
    """Every expression used as a *read* key: ``os.environ[k]`` (load context),
    ``os.environ.get/setdefault/pop(k)`` and ``os.getenv(k)``."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
            and _is_environ(node.value)
        ):
            yield node.slice
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("get", "setdefault", "pop")
                and _is_environ(func.value)
            ):
                yield node.args[0]
            elif (
                isinstance(func, ast.Attribute) and func.attr == "getenv"
                and isinstance(func.value, ast.Name) and func.value.id == "os"
            ) or (isinstance(func, ast.Name) and func.id == "getenv"):
                yield node.args[0]


def env_reads():
    """``(names read, {file: unresolved count})`` over the shipped tree.

    An AST walk, never a text grep: a comment naming a variable, a protocol
    token and a test-only ``MLLM_*`` name are not reads (D9). A key resolves
    when it is a string literal, a module-level constant assigned one (the
    ``WORKER_ENV`` indirection in ``utils/detach.py``), or an attribute or
    imported name whose target is such a constant somewhere in the tree.
    """
    parsed = {path: ast.parse(path.read_text(encoding="utf-8")) for path in _shipped_python()}
    local = {path: _module_constants(tree) for path, tree in parsed.items()}
    global_constants: dict = {}
    for constants in local.values():
        for name, value in constants.items():
            global_constants.setdefault(name, set()).add(value)

    names, unresolved = set(), {}
    for path, tree in parsed.items():
        for key in _env_read_keys(tree):
            value = None
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                value = key.value
            elif isinstance(key, ast.Name):
                value = local[path].get(key.id)
                if value is None and len(global_constants.get(key.id, ())) == 1:
                    value = next(iter(global_constants[key.id]))
            elif isinstance(key, ast.Attribute):
                if len(global_constants.get(key.attr, ())) == 1:
                    value = next(iter(global_constants[key.attr]))
            if value is None:
                unresolved[path.name] = unresolved.get(path.name, 0) + 1
            else:
                names.add(value)
    return names | surface_env_names(), unresolved


def surface_env_names() -> set:
    """Environment names the server reads *through the surface* (D3e).

    The server spells no variable name of its own: the uv override, its run-args
    override and the write-posture lever arrive as values of ``env`` /
    ``*_env`` keys in ``dist/cli-surface.json``, and the server reads them from
    its environment by those values — which is why ``mcp_server.py`` sits in
    :data:`DYNAMIC_READ_FILES`. Read from the emitted surface, so a name the
    emitter adds is discovered the same day.
    """
    from common import surface

    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if (key == "env" or key.endswith("_env")) and isinstance(value, str):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]+", value):
                        found.add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(surface.cli_surface("codex"))
    return found


class TestEnvironment:
    def test_env_allowlist_matches_the_tree(self):
        """(src) Exact in both directions (D9): the ``MULTI_LLM_*`` subset of
        the shipped ``env_vars`` equals the names shipped code reads minus
        :data:`NOT_FORWARDED`, and the remainder equals the declared four."""
        allowlist = _load(codex_build.MCP_SOURCE)["mcpServers"]["multi_llm"]["env_vars"]
        names, unresolved = env_reads()

        assert not (names & NOT_ENVIRONMENT), (
            f"{sorted(names & NOT_ENVIRONMENT)} became real environment reads; "
            "decide whether they are forwarded rather than sweeping them in"
        )
        assert NOT_FORWARDED <= names, (
            f"{sorted(NOT_FORWARDED - names)} is no longer read; drop it from "
            "NOT_FORWARDED"
        )
        read = {n for n in names if n.startswith("MULTI_LLM_")} - NOT_FORWARDED
        forwarded = {n for n in allowlist if n.startswith("MULTI_LLM_")}
        assert forwarded == read, (
            f"read but not forwarded: {sorted(read - forwarded)}; forwarded but "
            f"never read: {sorted(forwarded - read)}. A name missing from "
            "env_vars is silently absent from the server's environment (R12′)."
        )
        assert set(allowlist) - forwarded == NON_MULTI_LLM_FORWARDED
        assert len(allowlist) == len(set(allowlist))
        assert set(unresolved) <= set(DYNAMIC_READ_FILES), (
            f"unresolvable environment reads in {sorted(set(unresolved) - set(DYNAMIC_READ_FILES))}; "
            "read a literal or a module constant, or record the file in "
            "DYNAMIC_READ_FILES with its reason"
        )

    def test_the_walk_sees_through_a_module_constant(self):
        """(src) The one indirection the tree carries (``WORKER_ENV``) resolves,
        and a comment naming a variable does not count as a read."""
        tree = ast.parse(
            'import os\nWORKER = "MULTI_LLM_X"\n'
            '# os.environ.get("MULTI_LLM_COMMENT")\n'
            'def f():\n    return os.environ.get(WORKER)\n'
        )
        keys = list(_env_read_keys(tree))
        assert len(keys) == 1 and isinstance(keys[0], ast.Name)
        assert _module_constants(tree)["WORKER"] == "MULTI_LLM_X"


# ===========================================================================
# The thin skill (D3, D7)
# ===========================================================================

def _mode_spellings():
    """Mode words and aliases in the spellings a skill must not carry.

    Hyphenated names are distinctive, so they are forbidden as whole words;
    single English words (``ask``, ``implement``, ``review``) are forbidden only
    in a CLI spelling — as a ``--flag`` or back-quoted — since the exit-3 rule
    this skill renders legitimately says "review, plan, or implement".
    """
    words = {mode.name for mode in mode_table.MODES} | set(mode_table.ALIASES)
    bare = {word for word in words if "-" in word}
    cli = {f"--{word}" for word in words} | {f"`{word}`" for word in words}
    return bare, cli


class TestPluginSkill:
    """D7 item 2's list, graded on the render and on the checked-in copy."""

    @pytest.fixture(params=["render", "tree"])
    def skill(self, request) -> str:
        if request.param == "render":
            return codex_build.skill_md()
        return (PLUGIN_DIR / PAYLOAD / "SKILL.md").read_text(encoding="utf-8")

    def test_frontmatter_names_the_skill(self, skill):
        meta = _frontmatter(skill)
        assert meta["name"] == "multi-llm"
        assert set(meta) == {"name", "description"}

    def test_description_is_within_its_budget(self, skill):
        """R42 publishes no number, so this plan fixed one (D3e) — and the
        constant itself is asserted, so it cannot be widened to fit."""
        assert host_install.PLUGIN_SKILL_DESCRIPTION_MAX == 400
        description = _frontmatter(skill)["description"]
        assert description.strip()
        assert len(description) <= host_install.PLUGIN_SKILL_DESCRIPTION_MAX, len(description)

    def test_names_the_tool_exactly_once(self, skill):
        assert skill.count(f"`{mode_table.TOOL_NAME}`") == 1

    def test_the_needs_human_rule_is_the_owner_s_text(self, skill):
        """Filled with ``modes.NEEDS_HUMAN_RULE`` itself, not a retyped copy."""
        template = (SKILL_ROOT / codex_build.SKILL_TEMPLATE).read_text(encoding="utf-8")
        assert template.count(cli_common.NEEDS_HUMAN_RULE_TOKEN) == 1
        assert mode_table.NEEDS_HUMAN_WORK_SENTENCE not in " ".join(template.split())
        flat = " ".join(skill.split())
        assert mode_table.NEEDS_HUMAN_RULE in flat
        assert cli_common.NEEDS_HUMAN_RULE_TOKEN not in skill

    def test_carries_no_dispatcher_word(self, skill):
        bare, cli = _mode_spellings()
        body = skill
        for token in ("--project", "multi_llm.py", "--detach", "--host", "--yes",
                      "{SKILL_DIR}", "```"):
            assert token not in body, token
        assert not re.search(r"\buv\b", body), "no uv in the skill"
        for word in sorted(bare):
            assert not re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", body), word
        for spelling in sorted(cli):
            assert spelling not in body, spelling

    def test_names_the_fallback(self, skill):
        """D5 condition 5: a server that never started leaves prose as the only
        recovery, so the sentence names the command and the floor exactly."""
        assert "`codex mcp get multi_llm`" in skill
        assert "`python3 >= 3.9`" in skill

    def test_says_the_tool_writes_outside_the_sandbox(self, skill):
        flat = " ".join(skill.split())
        assert "outside the exec sandbox" in flat
        assert "writes to the workspace" in flat

    def test_is_under_twenty_lines(self, skill):
        assert len(skill.splitlines()) < 20

    def test_no_placeholder_survives(self, skill):
        """``test_shim_correctness.py``'s full ``PLACEHOLDERS`` sweep, re-homed."""
        from tests.test_shim_correctness import PLACEHOLDERS

        for token in PLACEHOLDERS:
            assert token not in skill, token
        for token in host_install.FORBIDDEN_IN_INSTALLED:
            assert token not in skill, token

    def test_no_core_paragraph_is_duplicated_into_the_shim(self, skill):
        """D7's own rule, asserted rather than assumed (re-homed from
        ``TestShim``, which does not collect this file)."""
        paragraphs = {
            " ".join(block.split())
            for block in re.split(r"\n\s*\n", _body(skill))
            if len(block.split()) >= 25
        }
        offenders = []
        for core in sorted(SKILL_ROOT.glob("instructions/*.md")):
            text = " ".join(core.read_text(encoding="utf-8").split())
            offenders += [f"{core.name}: {p[:80]}" for p in paragraphs if p in text]
        assert not offenders, offenders


class TestRender:
    def test_skill_md_matches_render(self):
        """(tree) Byte-identical to D1's mapping-table call, spelled once."""
        text = (SKILL_ROOT / codex_build.SKILL_TEMPLATE).read_text(encoding="utf-8")
        fresh = host_install.render(
            "codex", base=str(SKILL_ROOT), skill="{SKILL_DIR}", template=text
        )
        assert (PLUGIN_DIR / PAYLOAD / "SKILL.md").read_text(encoding="utf-8") == fresh
        assert codex_build.skill_md() == fresh

    def test_the_file_route_shim_is_untouched(self):
        """D7 item 1: the plugin skill is a new template; ``shim.md.tmpl`` stays
        the file route's, and the codex capability entry still points at it."""
        from utils import host_capabilities as caps

        assert caps.entry("codex")["shim"]["template"] == "hosts/codex/shim.md.tmpl"


# ===========================================================================
# The build (D1, D1a)
# ===========================================================================

#: An absolute-path shape: rooted at a filesystem top-level directory (the
#: build machine's, a developer's, a CI runner's), or a Windows user profile.
_TOP_LEVEL = (
    "tmp", "Users", "home", "root", "opt", "etc", "usr", "var", "private", "srv",
    "mnt", "media", "bin", "sbin", "lib", "lib64", "nix", "snap", "proc", "dev",
    "run", "Volumes", "Library", "System", "Applications", "workspace",
    "workspaces", "github", "runner", "__w", "builds", "build", "data", "app",
    "code",
)
_ABSOLUTE_SHAPE = re.compile(
    r"(?<![\w.~$}/\\<-])/(?:%s)(?![\w.-])(?:/[\w.+@-]+)*" % "|".join(_TOP_LEVEL)
)
_WINDOWS_PROFILE = re.compile(r"\b[A-Za-z]:[\\/]+(?:Users|home)\b")

#: The uv candidate ladder's POSIX members, read from its owner (never
#: re-spelled here: a literal trips the discovery grep's exclusion set).
_UV_LADDER = frozenset(c for c in unattended.UV_CANDIDATES if c.startswith("/"))
_SANDBOX = frozenset({"/usr/bin/bwrap", "/etc/apparmor.d/bwrap"})

#: ``{staged path: {absolute strings it may carry}}`` — every absolute path
#: the payload ships on purpose, with its reason. Anything else is a failure.
_ALLOWED_ABSOLUTE = {
    # (1) the uv ladder the server reads from the surface, and its message.
    "dist/cli-surface.json": _UV_LADDER,
    f"{PAYLOAD}/hosts/uv-missing.txt": _UV_LADDER,
    f"{PAYLOAD}/utils/unattended.py": _UV_LADDER,
    # (2) the same ladder as documentation, plus the ``2>/dev/null`` probe.
    f"{PAYLOAD}/references/uv-check.md": _UV_LADDER | {"/dev/null"},
    f"{PAYLOAD}/hosts/snippets/uv-preflight.md": _UV_LADDER | {"/dev/null"},
    f"{PAYLOAD}/hosts/codex/shim.md.tmpl": _UV_LADDER | {"/dev/null"},
    f"{PAYLOAD}/hosts/opencode/shim.md.tmpl": _UV_LADDER | {"/dev/null"},
    f"{PAYLOAD}/utils/detach.py": {"/usr/local", "/opt/local"},
    # (3) interpreter, shell and device spellings.
    "mcp/server.py": {"/bin/bash", "/dev/null"},
    f"{PAYLOAD}/utils/write_batch_scheduler.py": {"/bin/sh"},
    f"{PAYLOAD}/multi_llm.py": {"/dev/null"},
    f"{PAYLOAD}/utils/git_utils.py": {"/dev/null"},
    f"{PAYLOAD}/utils/process_control.py": {
        "/dev/fd", "/dev/null", "/proc", "/proc/self/fd"},
    f"{PAYLOAD}/utils/run_registry.py": {"/proc"},
    f"{PAYLOAD}/utils/providers/codex.py": {"/dev/null", "/tmp", "/dev/ptmx", "/dev/pts"},
    # the shell wrappers codex puts around each command (plan §4.2 step 2) and
    # the ``2>/dev/null`` suffix the read-only classifier admits.
    f"{PAYLOAD}/utils/providers/codex_stream_audit.py": {
        "/bin/bash", "/bin/sh", "/dev/null", "/usr/bin/bash", "/usr/bin/sh"},
    # (4) the sandbox names.
    # `/etc` in the plugin-landlock rung's "System-level alternative (needs
    # /etc and sudo …)" repair heading (plan codex-sandbox-unavailable step 2).
    f"{PAYLOAD}/utils/doctor.py": _SANDBOX | {"/bin/true", "/dev", "/etc"},
    # the plugin-landlock wrapper's device entries: `/dev/null` is a required
    # file rule, `/dev/ptmx` and `/dev/pts` the optional tty entries it notes.
    f"{PAYLOAD}/utils/landlock_exec.py": {"/dev/null", "/dev/ptmx", "/dev/pts"},
    # its dispatch half: the same device entries, `/dev/tty` in the step-1c
    # note, and the scratch directory's name pattern in a docstring.
    f"{PAYLOAD}/utils/landlock_dispatch.py": {
        "/dev/null", "/dev/ptmx", "/dev/pts", "/dev/tty", "/tmp/multi-llm-slot-"},
    f"{PAYLOAD}/utils/providers/base.py": {"/usr/bin/bwrap", "/dev/tty", "/tmp"},
    # (5) the container image's paths, in provider docstrings.
    f"{PAYLOAD}/hosts/capabilities.yaml": {"/opt/mllm/opencode-plugin"},
    f"{PAYLOAD}/utils/providers/aider.py": {"/opt/mllm/real/aider", "C:\\Users"},
    f"{PAYLOAD}/utils/providers/claude_code.py": {"/opt/mllm/real/claude"},
    f"{PAYLOAD}/utils/providers/cline.py": {"/opt/mllm/real/cline"},
    f"{PAYLOAD}/utils/providers/goose.py": {"/opt/mllm/real/goose"},
    f"{PAYLOAD}/utils/providers/pi.py": {"/opt/mllm/real/pi"},
    f"{PAYLOAD}/utils/providers/copilot.py": {"/opt"},
    # (6) prose about temp directories and generic examples, no machine's.
    f"{PAYLOAD}/ask_orchestrator.py": {"/tmp/q.txt"},
    f"{PAYLOAD}/implement_orchestrator.py": {"/tmp"},
    f"{PAYLOAD}/utils/cli_common.py": {"/tmp"},
    f"{PAYLOAD}/utils/llm_client.py": {"/tmp"},
    f"{PAYLOAD}/utils/providers/omp.py": {"/tmp"},
    f"{PAYLOAD}/utils/subagent_executor.py": {"/private/tmp", "/tmp"},
    f"{PAYLOAD}/utils/providers/gemini.py": {"/tmp/mllm-inv", "/tmp/mllm-inv/gemini-"},
    f"{PAYLOAD}/utils/providers/kilocode.py": {"/tmp/mllm-inv"},
    f"{PAYLOAD}/utils/provider_registry.py": {"/home/me/repo/.git"},
}


def _absolute_shapes(text: str) -> set:
    return set(_ABSOLUTE_SHAPE.findall(text)) | set(_WINDOWS_PROFILE.findall(text))


def _allowed_absolute(relative: str) -> frozenset:
    return frozenset(_ALLOWED_ABSOLUTE.get(relative, ()))


def _json_strings(node, keys=()):
    """``(key path, string)`` for every string leaf of a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _json_strings(value, keys + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from _json_strings(value, keys)
    elif isinstance(node, str):
        yield keys, node


def _server_payload_inventory() -> tuple:
    """The server's own call-time ``PAYLOAD_INVENTORY``, read by AST."""
    tree = ast.parse(codex_build.SERVER_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "PAYLOAD_INVENTORY"):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("mcp_server.py defines no PAYLOAD_INVENTORY")


def _server_env(base: Path) -> dict:
    """A stripped environment for the server: a workspace anchor and a uv
    that records being spawned (it never should be, on these rows)."""
    home = base / "home"
    workspace = base / "workspace"
    for directory in (home, workspace):
        directory.mkdir(exist_ok=True)
    uv = base / "uv-tripwire"
    if not uv.exists():
        uv.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(base / 'uv-spawned')!r}).write_text('x')\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
    return {
        "PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
        "LANG": "C.UTF-8", "SHELL": "/bin/bash", "TERM": "dumb",
        "USER": os.environ.get("USER", "u"), "PWD": str(workspace),
        "MULTI_LLM_UV": str(uv),
    }


def _drive_server(root: Path, base: Path, tool: str):
    """``initialize``, ``tools/list``, one ``tools/call``, ``tools/list`` over
    the server's stdio. Returns ``({step: response}, uv_spawned)``."""
    spawned = base / "uv-spawned"
    if spawned.exists():
        spawned.unlink()
    proc = subprocess.Popen(
        [sys.executable, str(root / "mcp" / "server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(base), env=_server_env(base),
    )
    responses = {}

    def exchange(key, rid, method, params=None):
        message = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            message["params"] = params
        proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        proc.stdin.flush()
        for raw in iter(proc.stdout.readline, b""):
            answer = json.loads(raw)
            if answer.get("id") == rid and "method" not in answer:
                responses[key] = answer
                return
        raise AssertionError(f"the server closed fd 1 before answering {method}")

    watchdog = threading.Timer(60, proc.kill)
    watchdog.start()
    try:
        exchange("initialize", 1, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"}})
        proc.stdin.write(b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
        proc.stdin.flush()
        exchange("list_before", 2, "tools/list")
        exchange("call", 3, "tools/call", {"name": tool, "arguments": {
            "mode": "review-plan", "plan": "p.md", "question": "",
            "extra_flags": []}})
        exchange("list_after", 4, "tools/list")
        proc.stdin.close()
        assert proc.wait(30) == 0, proc.stderr.read().decode("utf-8", "replace")
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
        proc.stdout.close()
        proc.stderr.close()
    return responses, spawned.exists()


class TestBuild:
    def test_the_build_names_a_missing_server(self, tmp_path):
        """(src) No tree without the server `.mcp.json` names."""
        missing = tmp_path / "mcp_server.py"
        with pytest.raises(payload.PayloadError, match="mcp_server.py"):
            codex_build.build(
                destination=tmp_path / "out", allow_dirty=True, server=missing
            )
        assert not (tmp_path / "out").exists()

    def test_allow_dirty_refuses_an_untracked_admitted_input(self, tmp_path):
        """--allow-dirty tolerates a modified input, never an untracked one:
        staging reads ``git ls-files``, so the file would silently be absent
        and the digest would attest to a tree no clean build reproduces."""
        repo = tmp_path / "repo"
        manifest = repo / payload.MANIFEST_RELATIVE
        manifest.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / payload.MANIFEST_RELATIVE, manifest)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        template = repo / "skills/multi-llm/hosts/codex/plugin-skill.md.tmpl"
        template.parent.mkdir(parents=True)
        template.write_text("x\n", encoding="utf-8")
        scratch = repo / "skills/multi-llm/notes.tmp"  # not admitted
        scratch.write_text("x\n", encoding="utf-8")

        assert codex_build.untracked_admitted(repo) == (
            "skills/multi-llm/hosts/codex/plugin-skill.md.tmpl",
        )
        with pytest.raises(payload.PayloadError, match="plugin-skill.md.tmpl"):
            codex_build.build(
                destination=tmp_path / "out", root=repo, allow_dirty=True
            )
        assert not (tmp_path / "out").exists()

        subprocess.run(["git", "-C", str(repo), "add", str(template)], check=True)
        assert codex_build.untracked_admitted(repo) == ()

    def test_allow_dirty_env_is_read_by_exact_spelling(self, monkeypatch):
        """``main()``'s ``--allow-dirty`` default, driven through the parser
        with ``build()`` stubbed: ``=1`` and nothing else turns it on."""
        seen = []

        def fake_build(*, destination, allow_dirty):
            seen.append(allow_dirty)
            return {"payload_files": 0, "destination": destination,
                    "version": "0", "digest": "0"}

        monkeypatch.setattr(codex_build, "build", fake_build)
        cases = (("1", True), ("0", False), ("true", False), ("", False), (None, False))
        for value, expected in cases:
            if value is None:
                monkeypatch.delenv(codex_build.ALLOW_DIRTY_ENV, raising=False)
            else:
                monkeypatch.setenv(codex_build.ALLOW_DIRTY_ENV, value)
            assert codex_build.allow_dirty_from_env() is expected, value
            assert codex_build.allow_dirty_from_env(
                {} if value is None else {codex_build.ALLOW_DIRTY_ENV: value}
            ) is expected, value
            seen.clear()
            assert codex_build.main([]) == 0
            assert seen == [expected], value
        # The flag itself still wins over an unset or off environment.
        seen.clear()
        assert codex_build.main(["--allow-dirty"]) == 0
        assert seen == [True]

    def test_version_is_read_from_the_owner(self):
        text = (SKILL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"', text, re.M)
        assert codex_build.project_version() == match.group(1)

    def test_packaging_inputs_name_every_input(self):
        """(src) Everything the build reads outside the payload prefix is on
        the dirty-input list, so an uncommitted edit to it cannot ship."""
        for path in (
            codex_build.MANIFEST_TEMPLATE, codex_build.MCP_SOURCE,
            codex_build.SERVER_SOURCE, CODEX_PKG / "build.py",
            CODEX_PKG / "ingestion_rules.py",
        ):
            relative = path.relative_to(REPO_ROOT).as_posix()
            assert relative in codex_build.PACKAGING_INPUTS, relative

    def test_checked_in_plugin_matches_build(self, built_plugin):
        """(build vs tree) Re-running ``build.py`` produces no diff.

        Built with the dirty allowance, so it grades the *working* tree: it goes
        red the moment a source edit lands without ``make codex-regen``.
        """
        built, checked_in = _files(built_plugin), _files(PLUGIN_DIR)
        assert set(built) == set(checked_in), (
            f"only in a fresh build: {sorted(set(built) - set(checked_in))[:10]}; "
            f"only checked in: {sorted(set(checked_in) - set(built))[:10]}. "
            "Run `make codex-regen`."
        )
        differing = [
            name for name in built
            if not filecmp.cmp(built[name], checked_in[name], shallow=False)
        ]
        assert not differing, (
            f"{len(differing)} file(s) differ from a fresh build: {differing[:10]}. "
            "Run `make codex-regen` and commit the tree with the edit."
        )

    def test_the_mapping_table(self, built_plugin):
        """(build) D1's six exceptions, each from its named source."""
        assert (built_plugin / "mcp" / "server.py").read_bytes() == (
            codex_build.SERVER_SOURCE.read_bytes()
        )
        assert (built_plugin / ".mcp.json").read_bytes() == codex_build.MCP_SOURCE.read_bytes()
        assert (built_plugin / "LICENSE").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes()
        assert (built_plugin / "dist" / "cli-surface.json").read_text("utf-8") == (
            codex_build.cli_surface_json()
        )
        assert (built_plugin / PAYLOAD / "SKILL.md").read_text("utf-8") == (
            codex_build.skill_md()
        )
        manifest = _load(built_plugin / ".codex-plugin" / "plugin.json")
        assert manifest["version"] == codex_build.project_version()
        # Everything else under skills/multi-llm/ is the source, verbatim.
        for relative in payload.expected_payload(REPO_ROOT):
            if relative == "SKILL.md":
                continue
            assert (built_plugin / PAYLOAD / relative).read_bytes() == (
                SKILL_ROOT / relative
            ).read_bytes(), relative

    def test_the_surface_is_the_opencode_schema_version(self, built_plugin):
        from common import surface

        document = _load(built_plugin / "dist" / "cli-surface.json")
        assert document["schema_version"] == surface.SURFACE_SCHEMA_VERSION
        assert document["host"] == "codex"
        assert document["tool_name"] == mode_table.TOOL_NAME

    def test_the_digest_covers_the_tree(self, built_plugin):
        recorded = (built_plugin / "dist" / "tree.sha256").read_text("utf-8").strip()
        assert recorded == codex_build.digest_of(built_plugin)
        assert (PLUGIN_DIR / "dist" / "tree.sha256").read_text("utf-8").strip() == (
            codex_build.digest_of(PLUGIN_DIR)
        ), "the checked-in digest does not match its own tree; `make codex-regen`"

    def test_no_absolute_path_in_the_payload(self, built_plugin):
        """(build) Two absences about *this machine*, and a positive, path-keyed
        allowlist for everything absolute that ships on purpose.

        Every staged text file is swept for absolute-path shapes (a path rooted
        at a filesystem top-level directory, or a Windows ``X:\\Users``); each
        hit must be enumerated for that staged path in
        :data:`_ALLOWED_ABSOLUTE`, so a ``/tmp``, ``/Users`` or CI-runner path
        nobody listed is a failure. A ``#!/usr/bin/env`` shebang on line 1 is
        the one shape allowed everywhere. The checkout root and an expanded
        ``$HOME``/``$CODEX_HOME`` are refused outright, listed or not.
        """
        machine = {str(REPO_ROOT), str(Path.home())}
        if os.environ.get("CODEX_HOME"):
            machine.add(os.environ["CODEX_HOME"])
        machine.discard("/")
        offenders, unlisted, stale = [], [], []
        seen_keys = set()
        for relative, path in _files(built_plugin).items():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # binary (icons): not a text file
            offenders += [f"{relative}: {m}" for m in machine if m and m in text]
            if text.startswith("#!/usr/bin/env "):
                text = text.split("\n", 1)[1] if "\n" in text else ""
            found = _absolute_shapes(text)
            allowed = _allowed_absolute(relative)
            if relative in _ALLOWED_ABSOLUTE:
                seen_keys.add(relative)
            unlisted += [f"{relative}: {hit}" for hit in sorted(found - allowed)]
            stale += [f"{relative}: {hit}" for hit in sorted(allowed - found)]
        assert not offenders, offenders
        assert not unlisted, (
            "absolute paths not on the allowlist (add a row with its reason, or "
            f"spell the path relative/at run time): {unlisted}"
        )
        assert not stale, f"allowlisted paths no longer shipped: {stale}"
        missing_keys = set(_ALLOWED_ABSOLUTE) - seen_keys
        assert not missing_keys, f"allowlist rows for files not shipped: {sorted(missing_keys)}"

        # The surface: absolute paths only under the uv.candidates key.
        surface = _load(built_plugin / "dist" / "cli-surface.json")
        misplaced = [
            f"{'.'.join(keys)}: {value}"
            for keys, value in _json_strings(surface)
            if _absolute_shapes(value) and keys[:2] != ("uv", "candidates")
        ]
        assert not misplaced, misplaced

    def test_payload_manifest(self):
        """(tree) What the Git route installs: the required set present, and
        nothing else — no symlink, no test/legacy/plan file, a clean sweep."""
        files = _files(PLUGIN_DIR)
        for required in codex_build.PLUGIN_FILES:
            assert required in files, required
        for name in (*payload.REQUIRED_INVENTORY, *payload.target_module_paths()):
            assert f"{PAYLOAD}/{name}" in files, name
        links = [p for p in PLUGIN_DIR.rglob("*") if p.is_symlink()]
        assert not links, links
        for relative in files:
            inner = relative[len(PAYLOAD) + 1:] if relative.startswith(PAYLOAD + "/") else ""
            assert not inner.startswith(("tests/", "legacy/", "plans/")), relative
            assert inner not in ("AGENTS.md", "CLAUDE.md"), relative
        payload_files = {
            name[len(PAYLOAD) + 1:] for name in files if name.startswith(PAYLOAD + "/")
        }
        assert payload_files == set(payload.expected_payload(REPO_ROOT)), (
            "the checked-in payload is not the allowlist's expansion"
        )

    def test_the_sweep_is_clean_on_the_checked_in_bytes(self, tmp_path):
        """(tree) A4 row 11a over the exact bytes, by-products aside."""
        for relative, path in _files(PLUGIN_DIR).items():
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        payload.scan_tree(tmp_path)

    def test_partial_payload_fails_by_name(self, built_plugin, tmp_path):
        """(build) Each required file deleted from a fresh copy of the built
        tree, in turn: the inventory refuses it by name, and the **server**,
        driven over stdio from that copy, still lists the one ``multi_llm``
        tool on every deletion — so D5's health contract is graded file by
        file. A file the server's call-time check covers (the dispatcher
        script plus ``PAYLOAD_INVENTORY``) is a pre-spawn tool error naming its
        absolute path. The plan's two exemptions: ``dist/cli-surface.json``
        answers with the degraded definition and the fail-closed envelope, and
        ``mcp/server.py`` absent means nothing of ours runs — no listing.
        """
        surface = _load(built_plugin / "dist" / "cli-surface.json")
        tool = surface["tool_name"]
        named_by_server = {
            f"{PAYLOAD}/{name}"
            for name in (surface["dispatcher"]["script"], *_server_payload_inventory())
        }
        required = list(codex_build.PLUGIN_FILES) + [
            f"{PAYLOAD}/{name}" for name in codex_build.RUNNABLE_PAYLOAD
        ]
        assert named_by_server <= set(required), sorted(named_by_server - set(required))
        for name in required:
            copy = tmp_path / name.replace("/", "_")
            shutil.copytree(built_plugin, copy)
            target = copy / name
            target.unlink()
            with pytest.raises(payload.PayloadError) as excinfo:
                codex_build._inventory(copy)  # noqa: SLF001
            assert name in str(excinfo.value), name

            if name == "mcp/server.py":
                proc = subprocess.run(
                    [sys.executable, str(target)], input=b"", capture_output=True,
                    timeout=30, cwd=str(tmp_path), env=_server_env(tmp_path),
                )
                assert proc.returncode != 0 and proc.stdout == b"", name
                shutil.rmtree(copy)
                continue

            responses, spawned = _drive_server(copy, tmp_path, tool)
            for key in ("list_before", "list_after"):
                tools = responses[key]["result"]["tools"]
                assert [t["name"] for t in tools] == [tool], (name, key)
            if name == "dist/cli-surface.json":
                result = responses["call"]["result"]
                envelope = json.loads(result["content"][0]["text"])
                assert result["isError"] is True and envelope["ok"] is False, name
                assert envelope["state"] == surface["dispatcher"]["states"][
                    "needs_human"], name
                assert envelope["exit"] is None and not spawned, name
            elif name in named_by_server:
                result = responses["call"]["result"]
                envelope = json.loads(result["content"][0]["text"])
                assert result["isError"] is True and envelope["ok"] is False, name
                assert envelope["state"] == surface["dispatcher"]["states"][
                    "needs_human"], name
                assert envelope["exit"] is None and not spawned, name
                assert os.path.abspath(str(target)) in envelope["message"], name
            shutil.rmtree(copy)
        # And a surplus file is refused as surely as a missing one.
        copy = tmp_path / "surplus"
        shutil.copytree(built_plugin, copy)
        (copy / "stray.txt").write_text("x", encoding="utf-8")
        with pytest.raises(payload.PayloadError, match="stray.txt"):
            codex_build._inventory(copy)  # noqa: SLF001

    def test_the_static_inventory_is_asserted_at_staging(self, tmp_path):
        """(src) A required payload file missing at staging time is named too."""
        staged = ["multi_llm.py"]
        with pytest.raises(payload.PayloadError, match="uv.lock"):
            payload._assert_inventory(  # noqa: SLF001
                tmp_path, staged, payload.REQUIRED_INVENTORY, ()
            )

    def test_a_payload_without_the_landlock_wrapper_fails_the_inventory(self, tmp_path):
        """(src) The ``plugin-landlock`` rung runs ``utils/landlock_exec.py``
        by path. A payload short of it alone must fail the build, not exit 2
        (``can't open file``) at every jailed launch, where a packaging fault
        reads as a probe result (plan codex-sandbox-unavailable-findings-and-
        fixes §7.1)."""
        wrapper = "utils/landlock_exec.py"
        assert wrapper in payload.REQUIRED_INVENTORY
        short = [name for name in payload.REQUIRED_INVENTORY if name != wrapper]
        with pytest.raises(payload.PayloadError) as excinfo:
            payload._assert_inventory(  # noqa: SLF001
                tmp_path, short, payload.REQUIRED_INVENTORY, ()
            )
        missing = [
            line.strip() for line in str(excinfo.value).splitlines()
            if line.startswith("    ")
        ]
        assert missing == [wrapper]
        payload._assert_inventory(  # noqa: SLF001
            tmp_path, [*short, wrapper], payload.REQUIRED_INVENTORY, ()
        )


@pytest.mark.package
class TestStagedPayloadRuns:
    def test_staged_tree_syncs_runtime_only(self, built_plugin, tmp_path):
        """(build, uv) ``--frozen --no-dev`` installs the runtime set only; the
        counter-case without ``--no-dev`` pulls the dev group in."""
        uv = unattended.find_uv()
        if uv is None:
            _unavailable("uv is not installed")
        project = built_plugin / PAYLOAD
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(tmp_path / "venv"))
        env.pop("VIRTUAL_ENV", None)

        def would_install(*flags):
            result = subprocess.run(
                [uv, "sync", *flags, "--dry-run", "--project", str(project)],
                capture_output=True, text=True, env=env, timeout=300,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            listing = result.stdout + result.stderr
            return {m.lower() for m in re.findall(r"^\s*\+\s+([A-Za-z0-9_.-]+)==", listing, re.M)}

        runtime = would_install("--frozen", "--no-dev")
        assert runtime == {"pyyaml"}, runtime
        everything = would_install("--frozen")
        assert "pytest" in everything and "pytest" not in runtime


# ===========================================================================
# Versioning (D6)
# ===========================================================================

def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check,
        env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
                 GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid"),
    )


class TestVersioning:
    def test_the_manifest_carries_the_owner_s_version(self):
        manifest = _load(PLUGIN_DIR / ".codex-plugin" / "plugin.json")
        assert manifest["version"] == codex_build.project_version()

    def test_version_bumps_with_payload(self):
        """History half, over this repository: adjacent ``codex-v*`` tags whose
        digests differ carry different versions, and no version is carried by
        two tags. With no tag at all — the pre-first-release state — ``release``
        must be absent too, asserted rather than passed vacuously."""
        tags = codex_build.release_tags()
        if not tags:
            for ref in ("refs/heads/release", "refs/remotes/origin/release"):
                probe = _git(REPO_ROOT, "rev-parse", "-q", "--verify", ref, check=False)
                assert probe.returncode != 0, (
                    f"{ref} exists but no codex-v* tag does: the release ref may "
                    "only ever point at a tagged commit"
                )
            return
        assert codex_build.bump_history_problems() == ()

    def test_the_bump_rule_reads_git_both_ways(self, tmp_path):
        """The rule itself, over a throwaway repository: digest moved under an
        unmoved version is refused, a bumped version or an unmoved digest is
        not — and a docs-only bump over an unchanged digest is legal."""
        repo = tmp_path / "repo"
        (repo / "skills" / "multi-llm").mkdir(parents=True)
        (repo / "packaging" / "codex" / "multi-llm" / "dist").mkdir(parents=True)
        _git(tmp_path, "init", "-q", str(repo))

        def commit(version: str, digest: str, tag: str = ""):
            (repo / codex_build.VERSION_PATH).write_text(
                f'[project]\nname = "multi-llm"\nversion = "{version}"\n', encoding="utf-8"
            )
            (repo / codex_build.DIGEST_PATH).write_text(digest + "\n", encoding="utf-8")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", f"{version} {digest}")
            if tag:
                _git(repo, "tag", tag)

        commit("0.1.0", "aaa", "codex-v0.1.0")
        commit("0.1.1", "aaa", "codex-v0.1.1")  # docs-only bump: legal
        commit("0.1.2", "bbb", "codex-v0.1.2")
        assert codex_build.bump_history_problems(repo) == ()
        commit("0.1.2", "ccc", "codex-v0.1.2+rebuild")  # same number, new bytes
        problems = codex_build.bump_history_problems(repo)
        assert any("0.1.2" in p for p in problems), problems

    def test_the_publish_half_compares_against_the_newest_tag(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "skills" / "multi-llm").mkdir(parents=True)
        (repo / "packaging" / "codex" / "multi-llm" / "dist").mkdir(parents=True)
        _git(tmp_path, "init", "-q", str(repo))
        (repo / codex_build.VERSION_PATH).write_text(
            '[project]\nversion = "0.1.0"\n', encoding="utf-8"
        )
        (repo / codex_build.DIGEST_PATH).write_text("aaa\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "r")
        _git(repo, "tag", "codex-v0.1.0")
        tree = tmp_path / "tree"
        (tree / "dist").mkdir(parents=True)
        (tree / "dist" / "tree.sha256").write_text("bbb\n", encoding="utf-8")
        monkeypatch.setattr(codex_build, "project_version", lambda base=None: "0.1.0")
        assert "0.1.0" in codex_build.bump_ahead_problem(repo, tree)
        monkeypatch.setattr(codex_build, "project_version", lambda base=None: "0.1.1")
        assert codex_build.bump_ahead_problem(repo, tree) is None


# ===========================================================================
# The catalog (A2a)
# ===========================================================================

class TestCatalog:
    def test_marketplace_entry_resolves(self):
        """A2a's shape: the name equals the manifest's (R29); ``source.path``
        relative, ``./``-prefixed and inside the root (R20′); no policy, or a
        legal one; no version anywhere."""
        catalog = _load(AGENTS_CATALOG)
        assert ingestion_rules.validate_catalog(catalog) == []
        assert catalog["name"] == "beastlabai"
        assert catalog["name"] not in ingestion_rules.RESERVED_MARKETPLACE_NAMES
        entries = catalog["plugins"]
        assert [entry["name"] for entry in entries] == ["multi-llm"]
        entry = entries[0]
        assert entry["source"] == {"source": "local", "path": "./packaging/codex/multi-llm"}
        assert "version" not in json.dumps(entry)
        assert "policy" not in entry
        plugin_dir = (REPO_ROOT / entry["source"]["path"]).resolve()
        assert plugin_dir == PLUGIN_DIR.resolve()
        assert REPO_ROOT.resolve() in plugin_dir.parents
        manifest = _load(plugin_dir / ".codex-plugin" / "plugin.json")
        assert manifest["name"] == entry["name"]
        assert entry["category"] == manifest["interface"]["category"]

    def test_the_catalog_rules_fire(self):
        base = {"name": "beastlabai", "plugins": [
            {"name": "x", "source": {"source": "local", "path": "./p"}}]}
        assert ingestion_rules.validate_catalog(base) == []
        for bad in (
            {"name": "openai-curated", "plugins": []},
            {"name": "beastlabai", "plugins": [{"name": "x", "source": "/abs"}]},
            {"name": "beastlabai", "plugins": [{"name": "x", "source": "./../up"}]},
            {"name": "beastlabai", "plugins": [
                {"name": "x", "source": "./p", "policy": {"authentication": "NONE"}}]},
        ):
            assert ingestion_rules.validate_catalog(bad), bad

    def test_gitattributes_marks_the_tree_generated(self):
        """D1a: one rule, and nothing else."""
        text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        rules = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
        assert rules == ["packaging/codex/multi-llm/** linguist-generated=true -diff"]


# ===========================================================================
# Make and CI
# ===========================================================================

def _makefile_recipes() -> dict:
    recipes, current = {}, None
    for line in (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+):(?!=)", line)
        if match:
            current = match.group(1)
            recipes[current] = []
        elif line.startswith("\t") and current:
            recipes[current].append(line.strip())
        elif line.strip() and not line.startswith(("#", " ")):
            current = None
    return recipes


class TestMake:
    TARGETS = ("codex-regen", "codex-package", "codex-package-check-ci",
               "codex-package-check", "codex-publish")

    def test_the_five_targets_exist_and_are_phony(self):
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        joined = makefile.replace("\\\n", " ")
        phony = " ".join(re.findall(r"^\.PHONY:(.*)$", joined, re.M))
        recipes = _makefile_recipes()
        for target in self.TARGETS:
            assert target in recipes, target
            assert re.search(rf"(?<![\w-]){re.escape(target)}(?![\w-])", phony), target
            assert f"make {target}" in makefile, f"{target} has no help line"
        assert "codex-stage" not in recipes, "the tree is committed; nothing stages it"

    def test_codex_package_pytest_names_this_route_s_suite(self):
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        block = re.search(r"^CODEX_PACKAGE_PYTEST :=(.*?)(?:\n\n)", makefile, re.M | re.S)
        assert block
        for name in ("test_codex_package.py", "test_codex_mcp_server.py",
                     "test_uv_candidates.py"):
            assert name in block.group(1), name
        assert "test_opencode_package.py" not in block.group(1)

    def test_the_clean_build_ignores_every_dirty_opt_in(self):
        recipes = _makefile_recipes()
        assert any("--allow-dirty" in line for line in recipes["codex-regen"])
        package = " ".join(recipes["codex-package"])
        assert "--allow-dirty" not in package
        assert "MULTI_LLM_ALLOW_DIRTY= " in package
        assert "--require-no-diff" in package

    def test_publish_refuses_before_it_tags(self):
        recipe = _makefile_recipes()["codex-publish"]
        assert "--check-stamp" in recipe[0]
        joined = "\n".join(recipe)
        assert joined.index("--check-stamp") < joined.index("git tag")
        assert "codex-v" in joined and "opencode-v" not in joined

    def test_the_full_gate_refuses_rows_it_cannot_run(self):
        recipe = "\n".join(_makefile_recipes()["codex-package-check"])
        assert "REFUSING" in recipe
        assert recipe.index("REFUSING (rows 6b, 14)") < recipe.index("--write-stamp")

    def test_ci_runs_the_codex_tier_beside_opencode(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text("utf-8")
        assert "make codex-package-check-ci" in workflow
        assert "make opencode-package-check-ci" in workflow

    def test_the_publish_workflow_is_split_and_unprivileged(self):
        import yaml

        path = REPO_ROOT / ".github" / "workflows" / "publish-codex-plugin.yml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = document[True] if True in document else document["on"]
        assert triggers["push"]["tags"] == ["codex-v*"]
        assert "tag" in triggers["workflow_dispatch"]["inputs"]
        jobs = document["jobs"]
        assert jobs["gate"]["permissions"] == {"contents": "read"}
        # The one write, plus the read-only gate-run lookup on the tagged sha.
        assert jobs["promote"]["permissions"] == {"contents": "write", "actions": "read"}
        text = path.read_text(encoding="utf-8")
        assert "id-token" not in text.replace("`id-token: write`", "")
        for use in re.findall(r"uses:\s*(\S+)", text):
            assert re.search(r"@[0-9a-f]{40}$", use), f"{use} is not pinned by SHA"
        assert "codex-release-network-check.json" in text


# ===========================================================================
# The promote job's record gate (Phase 4), exercised offline
# ===========================================================================

PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-codex-plugin.yml"
RECORDS_DIR = SKILL_ROOT / "tests" / "container" / "records"
REHEARSAL_GLOB = "*-codex-release-network-check-rehearsal.json"


def _promote_validator() -> str:
    """The Python the promote job feeds ``python3 -`` from its heredoc."""
    import yaml

    document = yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    for step in document["jobs"]["promote"]["steps"]:
        match = re.search(r"<<'PY'\n(.*?)\nPY\b", step.get("run") or "", re.S)
        if match:
            return match.group(1)
    raise AssertionError("the promote job has no python3 heredoc")


def _promote_required(source: str) -> list:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "REQUIRED"):
            return ast.literal_eval(node.value)
    raise AssertionError("the promote validator pins no REQUIRED list")


def _rehearsal_record() -> tuple:
    found = sorted(RECORDS_DIR.glob(REHEARSAL_GLOB))
    assert found, f"no {REHEARSAL_GLOB} under {RECORDS_DIR}"
    return found[-1], json.loads(found[-1].read_text(encoding="utf-8"))


class TestPromoteGate:
    """The promote job reads the network-check record; these run its own
    validator (extracted from the workflow, never retyped) against files in a
    temporary tree. Nothing here reaches GitHub."""

    def _run(self, tmp_path, records: dict, tag: str, sha: str):
        target = tmp_path / "skills" / "multi-llm" / "tests" / "container" / "records"
        target.mkdir(parents=True)
        for name, record in records.items():
            (target / name).write_text(json.dumps(record), encoding="utf-8")
        return subprocess.run([sys.executable, "-", tag, sha], input=_promote_validator(),
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)

    def test_criterion_names_are_the_rehearsal_records(self):
        _, record = _rehearsal_record()
        names = [c["name"] for c in record["criteria"]]
        assert len(names) == 6 == len(set(names))
        assert _promote_required(_promote_validator()) == names
        assert all(set(c) >= {"name", "pass", "evidence"} for c in record["criteria"])

    def test_the_rehearsal_record_is_marked_as_one(self):
        path, record = _rehearsal_record()
        assert record["rehearsal"] is True
        assert "not GitHub" in record["source"]
        for key in ("tag", "sha", "machine", "commands", "plugin_list", "mcp_get",
                    "locator_line"):
            assert record.get(key), key

    def test_a_genuine_record_promotes(self, tmp_path):
        _, record = _rehearsal_record()
        genuine = {k: v for k, v in record.items() if k != "rehearsal"}
        done = self._run(tmp_path, {"20990101T000000Z-codex-release-network-check.json": genuine},
                         record["tag"], record["sha"])
        assert done.returncode == 0, done.stderr
        assert "all six criteria pass" in done.stdout

    @pytest.mark.parametrize("flag", [True, "true", 1, None])
    def test_the_rehearsal_flag_is_refused(self, tmp_path, flag):
        _, record = _rehearsal_record()
        done = self._run(tmp_path, {"20990101T000000Z-codex-release-network-check.json":
                                    {**record, "rehearsal": flag}}, record["tag"], record["sha"])
        assert done.returncode != 0 and "rehearsal record" in done.stderr

    def test_a_rehearsal_file_name_is_refused(self, tmp_path):
        _, record = _rehearsal_record()
        genuine = {k: v for k, v in record.items() if k != "rehearsal"}
        done = self._run(tmp_path, {"20990101T000000Z-codex-release-network-check-rehearsal.json":
                                    genuine}, record["tag"], record["sha"])
        assert done.returncode != 0 and "-rehearsal.json" in done.stderr

    def test_a_red_or_renamed_criterion_is_refused(self, tmp_path):
        _, record = _rehearsal_record()
        genuine = {k: v for k, v in record.items() if k != "rehearsal"}
        red = json.loads(json.dumps(genuine))
        red["criteria"][0]["pass"] = False
        done = self._run(tmp_path, {"20990101T000000Z-codex-release-network-check.json": red},
                         record["tag"], record["sha"])
        assert done.returncode != 0 and "does not record a pass" in done.stderr
        renamed = json.loads(json.dumps(genuine))
        renamed["criteria"][0]["name"] = "marketplace_added"
        done = self._run(tmp_path / "b", {"20990101T000000Z-codex-release-network-check.json": renamed},
                         record["tag"], record["sha"])
        assert done.returncode != 0 and "expected exactly" in done.stderr

    def test_another_tag_or_sha_is_refused(self, tmp_path):
        _, record = _rehearsal_record()
        genuine = {k: v for k, v in record.items() if k != "rehearsal"}
        done = self._run(tmp_path, {"20990101T000000Z-codex-release-network-check.json": genuine},
                         record["tag"], "0" * 40)
        assert done.returncode != 0 and "no committed" in done.stderr


# ===========================================================================
# A6 steps 1-5, driven end to end
# ===========================================================================

def _copy_worktree(destination: Path) -> None:
    listed = _git(
        REPO_ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).stdout
    for relative in (p for p in listed.split("\0") if p):
        source = REPO_ROOT / relative
        if not source.is_file() or source.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _bump(text: str, old: str, new: str, pattern: str) -> str:
    replaced, count = re.subn(pattern.format(re.escape(old)), lambda m: m.group(0).replace(old, new), text, flags=re.M)
    assert count >= 1, pattern
    return replaced


@pytest.mark.package
class TestReleaseSequence:
    def test_release_sequence_runs_as_documented(self, tmp_path):
        """A6 steps 1-5 in a throwaway repository seeded from this working tree:
        bump (with every mirror step 1 names), ``codex-regen``, commit,
        ``codex-package``, stamp. ``make`` runs the real recipes, with
        ``CODEX_BUILD`` pointed at this interpreter so no second venv is synced.
        """
        if not shutil.which("make"):
            _unavailable("make is not on PATH")
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(tmp_path, "init", "-q", "-b", "master", str(repo))
        _copy_worktree(repo)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "baseline")
        build_cmd = f"{sys.executable} packaging/codex/build.py"
        env = dict(os.environ)
        for name in ("ALLOW_DIRTY", "MULTI_LLM_ALLOW_DIRTY", "MAKEFLAGS"):
            env.pop(name, None)

        def make(*targets: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["make", "--no-print-directory", *targets, f"CODEX_BUILD={build_cmd}"],
                cwd=str(repo), capture_output=True, text=True, env=env, timeout=600,
            )

        def singular() -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "skills/multi-llm/tests/test_opencode_package.py",
                 "-k", "test_version_source_is_singular"],
                cwd=str(repo), capture_output=True, text=True, env=env, timeout=600,
            )

        old = codex_build.project_version()
        major, minor, patch = (int(x) for x in old.split("+")[0].split("-")[0].split("."))
        new = f"{major}.{minor}.{patch + 1}"

        # Step 1 — the owner and every mirror it names, except the npm manifest.
        edits = {
            "skills/multi-llm/pyproject.toml": r'^version = "{}"',
            "skills/multi-llm/uv.lock": r'name = "multi-llm"\nversion = "{}"',
            ".claude-plugin/plugin.json": r'"version": "{}"',
            "skills/multi-llm/hosts/capabilities.yaml": r"^\s+version:\s*['\"]?{}['\"]?\s*$",
        }
        for relative, pattern in edits.items():
            path = repo / relative
            path.write_text(_bump(path.read_text("utf-8"), old, new, pattern), "utf-8")
        render = subprocess.run(
            [sys.executable, "skills/multi-llm/hosts/render_capabilities.py"],
            cwd=str(repo), capture_output=True, text=True, env=env, timeout=300,
        )
        assert render.returncode == 0, render.stdout + render.stderr
        check = subprocess.run(
            [sys.executable, "skills/multi-llm/hosts/render_capabilities.py", "--check"],
            cwd=str(repo), capture_output=True, text=True, env=env, timeout=300,
        )
        assert check.returncode == 0, check.stdout + check.stderr

        # ...and a run that leaves packaging/opencode/package.json behind is red
        # at test_version_source_is_singular, before any gate.
        behind = singular()
        assert behind.returncode != 0 and "package.json" in behind.stdout, behind.stdout[-2000:]
        package_json = repo / "packaging" / "opencode" / "package.json"
        package_json.write_text(
            _bump(package_json.read_text("utf-8"), old, new, r'"version": "{}"'), "utf-8"
        )

        # Step 2 — the regeneration writes the tracked tree from dirty inputs.
        regen = make("codex-regen")
        assert regen.returncode == 0, regen.stdout + regen.stderr
        manifest = json.loads(
            (repo / "packaging/codex/multi-llm/.codex-plugin/plugin.json").read_text("utf-8")
        )
        assert manifest["version"] == new
        status = _git(repo, "status", "--porcelain", "--", "packaging/codex/multi-llm").stdout
        assert "plugin.json" in status and "tree.sha256" in status, status
        synced = singular()
        assert synced.returncode == 0, synced.stdout[-2000:]

        # ...and the clean build refuses those same inputs, naming them.
        refused = make("codex-package")
        assert refused.returncode != 0
        assert "pyproject.toml" in refused.stdout + refused.stderr, refused.stderr

        # Step 3 — one commit; step 4 — the clean re-run produces no diff.
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"release {new}")
        clean = make("codex-package")
        assert clean.returncode == 0, clean.stdout + clean.stderr
        assert _git(repo, "status", "--porcelain").stdout.strip() == ""

        # Step 5 — codex-publish refuses until a stamp covers these bytes. No
        # remote exists, so nothing can be pushed even if a refusal regressed.
        publish = make("codex-publish")
        assert publish.returncode != 0
        assert "no " in publish.stderr and ".package-check.stamp" in publish.stderr
        assert _git(repo, "tag", "--list", "codex-v*").stdout.strip() == ""
        stamp = subprocess.run(
            [sys.executable, "packaging/codex/build.py", "--write-stamp", "2026-01-01T00:00:00Z"],
            cwd=str(repo), capture_output=True, text=True, env=env, timeout=300,
        )
        assert stamp.returncode == 0, stamp.stdout + stamp.stderr
        checked = subprocess.run(
            [sys.executable, "packaging/codex/build.py", "--check-stamp"],
            cwd=str(repo), capture_output=True, text=True, env=env, timeout=300,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


# ===========================================================================
# Route-keyed promotion — rule (j), Phase 4's Git-route-only rule
# ===========================================================================

class TestRoutePromotion:
    """``package`` moves on the recorded install SOURCE, never the route word."""

    def _row(self, **kwargs):
        from tests.container import harness

        defaults = dict(
            row_id="codex-package-ask", lane="host", host="codex",
            creds="openrouter", route="package",
            command="ask plans/auth-feature.md 'What does this plan change?' "
                    "--yes --quick --detach",
        )
        defaults.update(kwargs)
        return harness.Row(**defaults)

    def _result(self, row, out_dir, *, install_source):
        """A real green bundle: two receipts, a terminal detached run, and the
        ``plugin-install.json`` the wiring step writes."""
        from tests.container import harness

        receipt_dir = out_dir / "receipts" / row.row_id
        receipt_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            (receipt_dir / f"{index}.receipt.json").write_text(
                json.dumps({"argv": ["multi_llm.py", "ask"]}), encoding="utf-8")
        (out_dir / "run-states.json").write_text(
            json.dumps({"ask": {"state": "done"}}), encoding="utf-8")
        (out_dir / "plugin-install.json").write_text(
            json.dumps({"install_source": install_source}), encoding="utf-8")
        return harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                 out_dir=out_dir)

    def _edits(self, row, result):
        from tests.container import harness

        return harness.promotions(row, result, "pass", harness.GRADED_PASS)

    def test_promotion_is_route_keyed(self, tmp_path):
        """Both halves spell ``route: package``; only the Git install moves
        ``package``. The local-path seed — every bit as green — earns the
        route-neutral behavioural cells and leaves ``package`` where it stood."""
        from tests.container import harness

        row = self._row()
        local = self._edits(row, self._result(row, tmp_path / "local",
                                              install_source="local-path"))
        local_fields = {edit["field"] for edit in local}
        assert "package" not in local_fields, (
            "a local-path install proposed promoting `package` (rule j)")
        assert {"multi_turn", "detach"} <= local_fields
        assert not ({"shim", "install_paths", "arguments", "reload"}
                    & local_fields)

        git = self._edits(row, self._result(row, tmp_path / "git",
                                            install_source="git"))
        assert {"host": "codex", "field": "package", "from": "documented",
                "to": "verified", "evidence": row.row_id} in git

        # No install source recorded at all is the local case, not the Git one.
        bare = self._result(row, tmp_path / "bare", install_source=None)
        assert "package" not in harness.promotable_fields(row, bare)

    def test_a_probe_row_promotes_nothing(self, tmp_path):
        """A zero-quota probe's receipts are the harness's own stdio client,
        not a host turn — even over a Git install it proposes no edit."""
        row = self._row(row_id="codex-package-probe-runnability", creds="none",
                        probe="runnability")
        assert self._edits(row, self._result(row, tmp_path,
                                             install_source="git")) == []

    def test_the_seed_records_a_local_path_source(self):
        """The container's wiring step names its install source, so the gate
        above has a datum to key on rather than defaulting."""
        source = (SKILL_ROOT / "tests" / "container" / "entrypoint.py").read_text(
            encoding="utf-8")
        assert '"install_source": "local-path"' in source
