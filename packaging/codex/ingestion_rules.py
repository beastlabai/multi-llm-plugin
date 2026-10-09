"""The codex plugin ingestion rules this build holds its output to, restated.

**Restated, not ported.** Upstream's own checker is ``validate_plugin.py`` in the
``plugin-creator`` system skill a codex install unpacks under
``$CODEX_HOME/skills/.system/``. Its redistribution licence is unstated and its
schema churns, so nothing here is copied from it: the rules below are the
findings the plan records (R2, R3, R5, R6, R20′, R29, R39, R48) written down as
data, plus a checker small enough to read in one sitting. ``build.py`` runs it
over every tree it builds, and ``test_plugin_manifest_valid`` runs it again.

**Stamped, so the restatement is dated.** The four ``SOURCE_*`` constants say
which upstream file the rules were read against: the codex version, the path
under ``$CODEX_HOME`` where that file lives, the SHA-256 of its bytes, and the
day it was read. ``test_ingestion_rules_declare_provenance`` fails in plain CI
when the stamp falls behind the container's pinned codex, and the container
ingestion-rules drift row goes red when the installed file no longer hashes to
``SOURCE_SHA256``. Whoever moves the pin re-reads the file, fixes the rules
here, and re-stamps, in one commit.

**2026-09-30 (0.156.1 -> 0.159.2): the file is gone, not just moved.**
``validate_plugin.py`` no longer ships anywhere reachable: it is absent from
this machine's installed codex 0.159.2 (``$CODEX_HOME/skills/.system/`` unpacks
``imagegen``, ``openai-docs``, ``review-agent``, ``skill-creator`` and
``skill-installer`` -- no ``plugin-creator`` at all), and the public mirror
(``github.com/openai/skills``, tip ``49f948f``) still carries a
``skills/.system/plugin-creator/`` skill but only ``scripts/create_basic_plugin.py``
and ``references/plugin-json-spec.md`` -- no ``validate_plugin.py`` in that
repo's history at any commit. There is therefore nothing left to re-read or
re-hash: ``SOURCE_SHA256`` below is carried forward unchanged from the 0.156.1
read (the last time the file was known to exist) rather than restamped against
a live file. The reference doc that replaced it (``plugin-json-spec.md``, a
generation guide, not a validator) contradicts none of the rules below -- same
top-level keys, same ``interface`` keys, same marketplace shape -- so nothing
here is known to have drifted; only the standalone checker is confirmed gone.
``SOURCE_CODEX_VERSION``/``SOURCE_READ_DATE`` move to record *this* re-check,
which is a version-floor bump and an absence confirmation, not a fresh read.

**2026-10-08 (0.159.2 -> 0.161.0): still absent.** The container pin moved to
0.161.0 (``make e2e-lock-update``). A throwaway ``$CODEX_HOME`` that the installed
codex 0.161.0 populated (the parser-fixture rig's) unpacks the same five system
skills under ``skills/.system/`` and no ``plugin-creator``, so there is still no
``validate_plugin.py`` to read; the stamp moves as an absence re-confirmation
again, and ``SOURCE_SHA256`` stays the 0.156.1 value.

**2026-10-09 (0.161.0 -> 0.162.0): still absent.** The container pin moved to
0.162.0 (``make e2e-lock-update``). A throwaway ``$CODEX_HOME`` that the
in-image codex 0.162.0 populated (``codex exec`` run offline, ``--network none``,
which unpacks the system skills before it fails to reach the model) holds the
same five system skills under ``skills/.system/`` and no ``plugin-creator``, so
there is still no ``validate_plugin.py`` to read; the stamp moves as an absence
re-confirmation again, and ``SOURCE_SHA256`` stays the 0.156.1 value.

**The validator is stricter than local ingestion (R5).** ``codex plugin add``
installs a manifest carrying ``hooks`` that the validator rejects by name. What
ships is held to the validator's stricter rules anyway, so the tree is clean if
it is ever submitted.

This module is a build-time and test-time input, never shipped: the payload
stages ``skills/multi-llm`` by allowlist and ``build.py`` copies four named
files beside it, and this is none of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# The provenance stamp
# ---------------------------------------------------------------------------

#: The codex release the rules were last checked against. At or above the
#: container pin (``tests/container/package.json``), asserted. As of this
#: value the file at SOURCE_PATH is confirmed ABSENT (see the module
#: docstring's 2026-09-30 note) -- the version records the check, not a read.
SOURCE_CODEX_VERSION = "0.162.0"

#: Where that file used to live, relative to ``$CODEX_HOME``. The container
#: drift row looks here; off the image, or on any codex build that no longer
#: ships it, it is legitimately absent and that row skips naming this path.
SOURCE_PATH = "skills/.system/plugin-creator/scripts/validate_plugin.py"

#: SHA-256 of those bytes, as read at 0.156.1 -- carried forward unchanged
#: since no later build ships the file to re-hash (module docstring).
SOURCE_SHA256 = "f4eeadb733b28b0c3e714de263a76d6542866a672f3e99bdffcf4dbcdf85e944"

#: The day this stamp was last checked, ISO 8601 (an absence re-confirmation
#: on 2026-09-30, 2026-10-08 and again on 2026-10-09, not a fresh read -- the file has had no bytes to read since
#: at least 0.156.1's successor).
SOURCE_READ_DATE = "2026-10-09"

# ---------------------------------------------------------------------------
# The server's own interpreter floor
# ---------------------------------------------------------------------------

#: ``python3 >= 3.9``: the floor ``mcp/server.py`` is written to, one minor
#: below the payload's ``requires-python`` on purpose (A2b). The server runs on
#: the launching ``PATH``'s bare ``python3`` before uv exists, so nothing
#: fetches a newer one for it. This constant is the owner; the skill's fallback
#: sentence, the README and ``host_install.CODEX_SERVER_MIN_PYTHON`` agree with
#: it, and a test asserts the last of those.
MIN_PYTHON: Tuple[int, int] = (3, 9)

# ---------------------------------------------------------------------------
# R2 / R3 / R6 / R48 — the manifest
# ---------------------------------------------------------------------------

#: Every top-level key ``.codex-plugin/plugin.json`` may carry (R2). Anything
#: else is rejected.
ACCEPTED_KEYS = frozenset({
    "id", "name", "version", "description", "skills", "apps", "mcpServers",
    "interface", "author", "homepage", "repository", "license", "keywords",
})

#: Accepted is not required (R2): ``id`` is optional, and these five are not.
REQUIRED_KEYS: Tuple[str, ...] = (
    "name", "version", "description", "author", "interface",
)

#: Rejected by name (R3); plugin-bundled hooks are also a removed feature (R4).
REJECTED_KEYS = frozenset({"hooks"})

#: ``author`` sub-keys; ``name`` required, the ``url`` an absolute https URL.
AUTHOR_KEYS = frozenset({"name", "email", "url"})

#: ``interface`` sub-keys the validator accepts.
INTERFACE_KEYS = frozenset({
    "displayName", "shortDescription", "longDescription", "developerName",
    "category", "capabilities", "websiteURL", "privacyPolicyURL",
    "termsOfServiceURL", "brandColor", "composerIcon", "logo", "logoDark",
    "screenshots", "defaultPrompt", "default_prompt",
})

#: Required as non-empty strings (R3). ``category`` has no enum.
INTERFACE_REQUIRED_STRINGS: Tuple[str, ...] = (
    "displayName", "shortDescription", "longDescription", "developerName",
    "category",
)

#: One of these must be present (R48: the snake_case spelling is an alias).
DEFAULT_PROMPT_KEYS: Tuple[str, ...] = ("defaultPrompt", "default_prompt")

#: Asset-path fields; optional, and each must resolve inside the plugin (A5).
INTERFACE_ASSET_KEYS: Tuple[str, ...] = ("composerIcon", "logo", "logoDark")

#: Path fields are fixed conventions, not free paths (R6).
CONTRACT_PATHS: Mapping[str, str] = {
    "skills": "skills",
    "apps": ".app.json",
    "mcpServers": ".mcp.json",
}

#: Strict semver, ``+build`` suffix accepted (R48 — the ``+codex.<n>`` dev loop).
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

#: Plugin identifier: ASCII letters, digits, ``.``, ``_``, ``-``; dots separate
#: non-empty segments.
PLUGIN_NAME_RE = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")

#: An unfilled template marker fails the validator anywhere in the manifest.
TODO_MARKER = "[TODO:"

# ---------------------------------------------------------------------------
# R19 / R20′ / R29 — the marketplace catalog
# ---------------------------------------------------------------------------

#: Reserved marketplace names (R19); ``marketplace add`` refuses them.
RESERVED_MARKETPLACE_NAMES = frozenset({"openai-curated", "openai-api-curated"})

#: ``policy.authentication``'s enum; any other value kills the whole catalog.
AUTH_POLICIES = frozenset({"ON_INSTALL", "ON_USE"})

MARKETPLACE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

# ---------------------------------------------------------------------------
# R39 — the plugin's stdio MCP server
# ---------------------------------------------------------------------------

#: ``cwd`` must be a contained ``./`` path, ``${PLUGIN_ROOT}`` or
#: ``${PLUGIN_DATA}``; ``.`` is what ships (R11) and is accepted.
CWD_PREFIXES: Tuple[str, ...] = ("./", "${PLUGIN_ROOT}", "${PLUGIN_DATA}")
CWD_EXACT: Tuple[str, ...] = (".", "${PLUGIN_ROOT}", "${PLUGIN_DATA}")


def _is_bare_executable(command: str) -> bool:
    return bool(command) and "/" not in command and "\\" not in command


def _is_contained_dot_path(value: str) -> bool:
    if not value.startswith("./"):
        return False
    parts = PurePosixPath(value).parts
    return ".." not in parts


def cwd_is_legal(cwd: str) -> bool:
    """R39's ``cwd`` rule."""
    if cwd in CWD_EXACT:
        return True
    if cwd.startswith(("${PLUGIN_ROOT}/", "${PLUGIN_DATA}/")):
        return ".." not in PurePosixPath(cwd).parts
    return _is_contained_dot_path(cwd)


def command_is_legal(command: str) -> bool:
    """R39's ``command`` rule: a bare executable name or a contained ``./`` path."""
    return _is_bare_executable(command) or _is_contained_dot_path(command)


# ---------------------------------------------------------------------------
# The checkers
# ---------------------------------------------------------------------------

def normalize_contract_path(raw: str) -> Optional[str]:
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute():
        return None
    parts = [part for part in path.parts if part != "."]
    normalized = "/".join(parts).rstrip("/")
    return normalized or None


def _todo_markers(value: Any, where: str, errors: List[str]) -> None:
    if isinstance(value, str):
        if TODO_MARKER in value:
            errors.append(f"{where} still contains a `[TODO: ...]` placeholder")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _todo_markers(item, f"{where}[{index}]", errors)
    elif isinstance(value, dict):
        for key, item in value.items():
            _todo_markers(item, f"{where}.{key}", errors)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _https(value: Any) -> bool:
    return isinstance(value, str) and re.match(r"^https://[^/\s]+", value) is not None


def _asset_ok(plugin_root: Path, raw: Any) -> bool:
    if not _non_empty_string(raw):
        return False
    candidate = PurePosixPath(str(raw).replace("\\", "/"))
    if candidate.is_absolute() or any(p in {"", ".", ".."} for p in candidate.parts):
        return False
    return (plugin_root / candidate.as_posix()).is_file()


def validate_manifest(
    manifest: Mapping[str, Any], plugin_root: Optional[Path] = None,
) -> List[str]:
    """Every rule above that *manifest* breaks, as readable lines.

    *plugin_root*, when given, also grades what the manifest points at: the
    companion ``.mcp.json`` and every asset path. Empty list means clean.
    """
    errors: List[str] = []
    if not isinstance(manifest, Mapping):
        return ["plugin.json must contain a JSON object"]
    _todo_markers(dict(manifest), "$", errors)

    for key in sorted(REJECTED_KEYS & set(manifest)):
        errors.append(f"plugin.json field `{key}` is rejected by name (R3)")
    for key in sorted(set(manifest) - ACCEPTED_KEYS - REJECTED_KEYS):
        errors.append(f"plugin.json field `{key}` is not an accepted key (R2)")
    for key in REQUIRED_KEYS:
        if key not in manifest:
            errors.append(f"plugin.json field `{key}` is required (R2)")

    if "id" in manifest and not _non_empty_string(manifest["id"]):
        errors.append("plugin.json field `id` must be a non-empty string")
    name = manifest.get("name")
    if not _non_empty_string(name) or not PLUGIN_NAME_RE.fullmatch(str(name)):
        errors.append("plugin.json field `name` must be a valid plugin identifier")
    version = manifest.get("version")
    if not _non_empty_string(version) or not SEMVER_RE.fullmatch(str(version)):
        errors.append("plugin.json field `version` must be strict semver")
    if not _non_empty_string(manifest.get("description")):
        errors.append("plugin.json field `description` must be a non-empty string")

    author = manifest.get("author")
    if not isinstance(author, Mapping):
        errors.append("plugin.json field `author` must be an object")
    else:
        for key in sorted(set(author) - AUTHOR_KEYS):
            errors.append(f"plugin.json field `author.{key}` is not accepted")
        if not _non_empty_string(author.get("name")):
            errors.append("plugin.json field `author.name` must be a non-empty string")
        if "url" in author and not _https(author["url"]):
            errors.append("plugin.json field `author.url` must be an https URL")

    for key, expected in CONTRACT_PATHS.items():
        if key not in manifest:
            continue
        value = manifest[key]
        if key == "mcpServers" and isinstance(value, Mapping):
            continue
        if not isinstance(value, str) or normalize_contract_path(value) != expected:
            errors.append(f"plugin.json field `{key}` must resolve to `{expected}` (R6)")

    interface = manifest.get("interface")
    if not isinstance(interface, Mapping):
        errors.append("plugin.json field `interface` must be an object")
    else:
        for key in sorted(set(interface) - INTERFACE_KEYS):
            errors.append(f"plugin.json field `interface.{key}` is not accepted")
        for key in INTERFACE_REQUIRED_STRINGS:
            if not _non_empty_string(interface.get(key)):
                errors.append(
                    f"plugin.json field `interface.{key}` must be a non-empty string (R3)"
                )
        if not any(key in interface for key in DEFAULT_PROMPT_KEYS):
            errors.append("plugin.json field `interface.defaultPrompt` is required (R3)")
        capabilities = interface.get("capabilities")
        if not isinstance(capabilities, list) or not all(
            _non_empty_string(item) for item in capabilities
        ):
            errors.append(
                "plugin.json field `interface.capabilities` must be an array of strings"
            )
        for key in ("websiteURL", "privacyPolicyURL", "termsOfServiceURL"):
            if key in interface and not _https(interface[key]):
                errors.append(f"plugin.json field `interface.{key}` must be an https URL")
        if plugin_root is not None:
            for key in INTERFACE_ASSET_KEYS:
                if key in interface and not _asset_ok(plugin_root, interface[key]):
                    errors.append(
                        f"plugin.json field `interface.{key}` does not resolve "
                        "inside the plugin"
                    )
            for index, raw in enumerate(interface.get("screenshots") or []):
                if not _asset_ok(plugin_root, raw):
                    errors.append(
                        f"plugin.json field `interface.screenshots[{index}]` does "
                        "not resolve inside the plugin"
                    )

    if plugin_root is not None and isinstance(manifest.get("mcpServers"), str):
        path = Path(plugin_root) / ".mcp.json"
        if not path.is_file():
            errors.append("`.mcp.json` is required when plugin.json names it")
        else:
            try:
                errors.extend(validate_mcp(json.loads(path.read_text(encoding="utf-8"))))
            except ValueError:
                errors.append("`.mcp.json` must contain valid JSON")
    return errors


def validate_mcp(document: Any) -> List[str]:
    """The ``.mcp.json`` rules: one object of servers, each in R39's form."""
    errors: List[str] = []
    if not isinstance(document, Mapping):
        return ["`.mcp.json` must contain a JSON object"]
    for key in sorted(set(document) - {"mcpServers"}):
        errors.append(f"`.mcp.json` field `{key}` is not accepted")
    servers = document.get("mcpServers")
    if not isinstance(servers, Mapping):
        return errors + ["`.mcp.json` field `mcpServers` must be an object"]
    for name, server in servers.items():
        if not _non_empty_string(name):
            errors.append("`.mcp.json` server names must be non-empty strings")
        if not isinstance(server, Mapping):
            errors.append(f"`.mcp.json` server `{name}` must be an object")
            continue
        command = server.get("command")
        if not isinstance(command, str) or not command_is_legal(command):
            errors.append(
                f"`.mcp.json` server `{name}`: command must be a bare executable "
                "name or a contained ./ path (R39)"
            )
        for index, arg in enumerate(server.get("args") or []):
            if not isinstance(arg, str):
                errors.append(f"`.mcp.json` server `{name}`: args[{index}] not a string")
        cwd = server.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not cwd_is_legal(cwd)):
            errors.append(
                f"`.mcp.json` server `{name}`: cwd must be a contained ./ path, "
                "${PLUGIN_ROOT} or ${PLUGIN_DATA} (R39)"
            )
    return errors


def validate_catalog(catalog: Any) -> List[str]:
    """``.agents/plugins/marketplace.json``: name legal, entries relative (R20′)."""
    errors: List[str] = []
    if not isinstance(catalog, Mapping):
        return ["marketplace.json must contain a JSON object"]
    name = catalog.get("name")
    if not _non_empty_string(name) or not MARKETPLACE_NAME_RE.fullmatch(str(name)):
        errors.append("marketplace name must be ASCII letters, digits, `_` or `-`")
    elif name in RESERVED_MARKETPLACE_NAMES:
        errors.append(f"marketplace name {name!r} is reserved (R19)")
    plugins = catalog.get("plugins")
    if not isinstance(plugins, list):
        return errors + ["marketplace.json field `plugins` must be an array"]
    for entry in plugins:
        label = entry.get("name") if isinstance(entry, Mapping) else entry
        if not isinstance(entry, Mapping):
            errors.append(f"catalog entry {label!r} must be an object")
            continue
        source = entry.get("source")
        path = source.get("path") if isinstance(source, Mapping) else source
        if not isinstance(path, str) or not _is_contained_dot_path(path):
            errors.append(
                f"catalog entry {label!r}: source path must be relative, `./`-"
                "prefixed and inside the root, or codex skips it silently (R20′)"
            )
        policy = entry.get("policy") or {}
        if isinstance(policy, Mapping) and "authentication" in policy:
            if policy["authentication"] not in AUTH_POLICIES:
                errors.append(
                    f"catalog entry {label!r}: policy.authentication must be one "
                    f"of {sorted(AUTH_POLICIES)}"
                )
        if "version" in entry:
            errors.append(f"catalog entry {label!r} carries a version; none is read")
    return errors


def stamp() -> Dict[str, str]:
    """The provenance stamp as one mapping, for a report or a record."""
    return {
        "codex_version": SOURCE_CODEX_VERSION,
        "path": SOURCE_PATH,
        "sha256": SOURCE_SHA256,
        "read_date": SOURCE_READ_DATE,
    }
