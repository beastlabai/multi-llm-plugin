#!/usr/bin/env python3
"""Capability data (D5b) and the interactive inventory (D5d).

Four properties, each of which the design leans on somewhere else:

1. **The data validates and is complete.** ``hosts/capabilities.yaml`` matches
   ``schemas/host_capabilities.schema.json``, every entry carries all three verbs
   plus a ``native_tool`` wherever a verb is native, a ``default_subagent_runner``
   present in its own ``runners`` map, ``subagent_tiers`` iff ``subagents: native``,
   ``env_guards`` as three lists of literal names, and a ``provenance`` key per
   field. Missing keys, unknown keys and out-of-place ``unknown`` values each fail —
   because a defaulted capability is exactly the silent guess the schema exists to
   remove.
2. **A `documented`-only value never selects the non-conservative path.** Native
   subagents and a trusted command-timeout ceiling both require ``verified``. This
   is what makes "the conservative resolution is what executes" a test rather than a
   promise.
3. **Consumers read the data, not a hardcoded table.** Mutating an entry in a
   fixture tree changes the flags that get chosen and the ceiling that binds; an
   unknown ``--host`` is exit 3; ``--exec-subagents`` on a host with no
   executor-capable runner is exit 3.
4. **The generated markdown is a build artifact.** The checked-in
   ``references/host-capabilities.md`` is byte-identical to a fresh render, so human
   docs stay in sync by construction.

Plus D5d: the inventory is closed, the fail-closed rows render on no host, and the
two presentations of a row resolve to an identical list of option commands.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "hosts"))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import host_capabilities as caps  # noqa: E402
from utils import unattended as un  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

import render_capabilities  # noqa: E402

CAPS_PATH = SKILL_ROOT / caps.CAPABILITIES_FILE
SCHEMA_PATH = SKILL_ROOT / caps.SCHEMA_FILE
RENDERED = SKILL_ROOT / render_capabilities.OUTPUT

DOCUMENT = yaml.safe_load(CAPS_PATH.read_text(encoding="utf-8"))
HOSTS = DOCUMENT["hosts"]
HOST_IDS = sorted(HOSTS)


def _flat_provenance(provenance):
    """``{name: value}`` with every map-valued row flattened into dotted names.

    ``next_command``'s provenance is a map, one value per route, so a promoted
    route is ``{"package": "verified"}`` — a dict, never equal to
    ``"verified"``. Without flattening, the ledger tests would let that
    promotion through with no recorded run. Written here rather than imported
    from the renderer, so a bug in one cannot hide the other's.
    """
    flat = {}
    for field, value in provenance.items():
        if isinstance(value, dict):
            for key, inner in value.items():
                flat[f"{field}.{key}"] = inner
        else:
            flat[field] = value
    return flat


def _capture(store):
    """A ``_load_target`` stand-in recording the residual argv verbatim."""

    def loader(module_path, attr):
        def target(argv):
            store["argv"] = list(argv)
            store["module"] = module_path
            return 0

        return target

    return loader


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    caps.clear_cache()
    yield
    caps.clear_cache()


@pytest.fixture
def fixture_tree(tmp_path, monkeypatch):
    """A copy of the capability data a test may mutate, wired as the skill dir.

    "Consumers read the data" is only meaningful if changing the data changes the
    behavior, and that cannot be shown against the shipped file.
    """
    root = tmp_path / "skill dir"
    (root / "hosts").mkdir(parents=True)
    (root / "schemas").mkdir(parents=True)
    shutil.copy(CAPS_PATH, root / caps.CAPABILITIES_FILE)
    shutil.copy(SCHEMA_PATH, root / caps.SCHEMA_FILE)

    def write(document):
        (root / caps.CAPABILITIES_FILE).write_text(
            yaml.safe_dump(document, sort_keys=True), encoding="utf-8"
        )
        caps.clear_cache()

    monkeypatch.setattr(caps, "_CACHE", {})
    monkeypatch.setattr("utils.cli_common._SKILL_DIR", root)
    return root, write


# ===========================================================================
# 1. The data validates and is complete
# ===========================================================================

class TestSchema:
    def test_the_document_validates(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(DOCUMENT, schema)

    def test_the_schema_rejects_a_missing_required_field(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        broken = json.loads(json.dumps(DOCUMENT))
        del broken["hosts"]["codex"]["detach"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    def test_the_schema_rejects_an_unknown_field(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        broken = json.loads(json.dumps(DOCUMENT))
        broken["hosts"]["codex"]["subagents_maybe"] = "native"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    def test_the_schema_rejects_a_wildcard_env_guard(self):
        """A `*` would strip the runner's own credentials with the guard (D3c)."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        broken = json.loads(json.dumps(DOCUMENT))
        broken["hosts"]["codex"]["env_guards"]["guards"] = ["CODEX_*"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    #: The only keys allowed to be absent. Each governs a lint or generated
    #: documentation and **never** a dispatch decision — that is the whole
    #: admission test, because a defaulted *capability* is the silent guess the
    #: schema exists to remove. Anything that could change what runs must be
    #: required, so its absence is a validation error rather than a guess.
    #:
    #: ``package`` passes the test, and the reasoning is worth keeping because
    #: it is the first entry that describes something *outside* the harness. It
    #: says how the skill is distributed to a host, not what the host can do, so
    #: it is optional for a plain reason: most hosts have no package route, and
    #: requiring it would force a dummy block into every one of them. It governs
    #: the generated "Install via" paragraph and ``--check``'s route-detection
    #: note; **no consumer branches on it to decide what runs**, which is what
    #: keeps it on the admissible side of the line. The moment something does,
    #: it stops being an optional key.
    OPTIONAL_KEYS = {"shim_max_lines", "headless_invocation", "package"}

    def test_only_lint_and_doc_keys_are_optional(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        host = schema["definitions"]["host"]
        optional = set(host["properties"]) - set(host["required"])
        assert optional == self.OPTIONAL_KEYS, (
            f"optional key set changed to {sorted(optional)}. A new optional key "
            "is only admissible if it cannot affect dispatch; if it can, make it "
            "required instead."
        )

    #: Fields a consumer *branches on*. A branch needs a stated meaning: without
    #: one nobody can say what a value claims, and a capability nobody can define
    #: is a capability nobody can falsify — which is how `multi_turn` shipped
    #: `false` for a host whose recorded run had already contradicted it.
    BRANCHING_KEYS = {"multi_turn", "subagents", "ask", "detach"}

    def test_every_branching_field_states_what_it_claims(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        properties = schema["definitions"]["host"]["properties"]
        for key in sorted(self.BRANCHING_KEYS):
            assert properties[key].get("description", "").strip(), (
                f"'{key}' selects a code path but the schema states no meaning "
                "for its values"
            )

    def test_the_runtime_floor_is_the_schemas_own_required_list(self):
        """`REQUIRED_HOST_KEYS` is a restatement, not a second opinion.

        `load` cannot use the schema: `jsonschema` is a dev-only extra and there
        is an active plan to ship a dependency-free package, so the runtime floor
        is a plain tuple. That only stays honest while this assertion holds — the
        tuple is allowed to be exactly the schema's `required` list, never a
        hand-curated subset that quietly stops covering a new key.
        """
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert set(caps.REQUIRED_HOST_KEYS) == set(
            schema["definitions"]["host"]["required"]
        ), (
            "utils/host_capabilities.REQUIRED_HOST_KEYS drifted from "
            "schemas/host_capabilities.schema.json. Add the new required key to "
            "the tuple; the runtime floor is what turns a malformed install into "
            "one actionable error instead of a KeyError mid-dispatch."
        )

    def test_every_optional_key_is_provenance_tracked(self):
        """Optional does not mean unattributed."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        prov = schema["definitions"]["host"]["properties"]["provenance"]["properties"]
        for key in self.OPTIONAL_KEYS:
            assert key in prov, f"optional key '{key}' has no provenance slot"


#: Every ``(host, route)`` the shipped ``next_command`` cells carry.
NEXT_COMMAND_ROUTES = [
    (host_id, route)
    for host_id in HOST_IDS
    for route in HOSTS[host_id]["next_command"]
]


class TestNextCommand:
    """The host-level ``next_command`` cell (plans/next-phase-command.md §2.1, T2).

    Required, keyed by the route vocabulary (``default``, ``package``,
    ``skill``), read at run time by ``utils/next_phase.py`` to spell the
    command a finished phase ends its report with.
    """

    @staticmethod
    def _validator():
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        return jsonschema, schema

    @staticmethod
    def _copy():
        return json.loads(json.dumps(DOCUMENT))

    @staticmethod
    def _package_host():
        hosts = [host_id for host_id in HOST_IDS if "package" in HOSTS[host_id]]
        assert hosts, "no host declares a package block"
        return hosts[0]

    def test_a_package_block_without_a_package_template_is_rejected(self):
        jsonschema, schema = self._validator()
        host_id = self._package_host()
        broken = self._copy()
        del broken["hosts"][host_id]["next_command"]["package"]
        del broken["hosts"][host_id]["provenance"]["next_command"]["package"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    def test_a_package_template_without_a_package_block_is_rejected(self):
        jsonschema, schema = self._validator()
        # A host that never had a block gains a template.
        broken = self._copy()
        host = broken["hosts"]["claude-code"]
        assert "package" not in host
        host["next_command"]["package"] = host["next_command"]["default"]
        host["provenance"]["next_command"]["package"] = "documented"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)
        # A host that had one loses the block and keeps the template.
        host_id = self._package_host()
        broken = self._copy()
        del broken["hosts"][host_id]["package"]
        del broken["hosts"][host_id]["provenance"]["package"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    @pytest.mark.parametrize("host_id, route", NEXT_COMMAND_ROUTES)
    def test_a_route_without_its_provenance_row_is_rejected(self, host_id, route):
        jsonschema, schema = self._validator()
        broken = self._copy()
        del broken["hosts"][host_id]["provenance"]["next_command"][route]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    @pytest.mark.parametrize("template", [
        "/multi-llm:multi-llm",                  # no {args}
        "{args}",                                # no prefix
        "/multi-llm {args} --yes",               # {args} not last
        "/multi-llm {plan} {args}",              # a second brace
        "/multi-llm {ARGS}",                     # not the token
    ])
    def test_a_template_ends_in_args_and_holds_no_other_brace(self, template):
        jsonschema, schema = self._validator()
        broken = self._copy()
        broken["hosts"]["claude-code"]["next_command"]["default"] = template
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    def test_an_unknown_route_key_is_rejected(self):
        """The keys are the route vocabulary; ``plugin`` is a product name."""
        jsonschema, schema = self._validator()
        broken = self._copy()
        broken["hosts"]["codex"]["next_command"]["plugin"] = "$x {args}"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_provenance_has_exactly_the_cells_routes(self, host_id):
        host = HOSTS[host_id]
        assert set(host["provenance"]["next_command"]) == set(host["next_command"])

    @pytest.mark.parametrize("host_id, route", NEXT_COMMAND_ROUTES)
    def test_each_prefix_is_in_the_prose_that_documents_its_route(
        self, host_id, route
    ):
        """``default``/``skill`` against ``invocation``, ``package`` against
        ``package.invocation``, so the display cell and the documented
        spelling cannot drift apart."""
        host = HOSTS[host_id]
        prose = (
            host["package"]["invocation"] if route == "package"
            else host["invocation"]
        )
        prefix = caps.next_command_prefix(host_id, route)
        assert prefix.strip(), (host_id, route)
        assert prefix in prose, (
            f"{host_id}.next_command.{route}'s prefix {prefix!r} is not in the "
            f"prose that documents that route: {prose!r}"
        )

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_none_reads_default(self, host_id):
        assert caps.next_command_template(host_id, None) == (
            HOSTS[host_id]["next_command"]["default"]
        )

    @pytest.mark.parametrize("host_id, route", NEXT_COMMAND_ROUTES)
    def test_each_present_key_returns_its_template(self, host_id, route):
        assert caps.next_command_template(host_id, route) == (
            HOSTS[host_id]["next_command"][route]
        )

    @pytest.mark.parametrize("host_id, route", [
        ("claude-code", "package"),
        ("codex", "skill"),
        ("opencode", "plugin"),
        ("codex", ""),
    ])
    def test_an_absent_route_raises_naming_the_host_and_the_route(
        self, host_id, route
    ):
        """The accessor never falls back: a wrong spelling is a caller bug."""
        with pytest.raises(ValueError) as raised:
            caps.next_command_template(host_id, route)
        assert host_id in str(raised.value)
        assert f"'{route}'" in str(raised.value)

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_the_route_list_is_the_cells_keys(self, host_id):
        assert caps.next_command_routes(host_id) == tuple(
            HOSTS[host_id]["next_command"]
        )

    def test_every_tool_route_is_a_key_on_some_host(self):
        """``modes.TOOL_ROUTES`` and the yaml keys are one vocabulary."""
        assert mode_table.TOOL_ROUTES
        for route in mode_table.TOOL_ROUTES:
            assert any(route in HOSTS[h]["next_command"] for h in HOST_IDS), (
                f"modes.TOOL_ROUTES names {route!r}, which no host's "
                "next_command carries"
            )


class TestLoadRefusesAMalformedDocument:
    """The schema is CI's guard; this is the one that fires in the field.

    A user never edits `hosts/capabilities.yaml` — it ships inside the skill — so
    every way it can be wrong is a partial or corrupt install, and each one used
    to surface as a raw exception from wherever a consumer happened to index
    first: a `yaml.ParserError` traceback, an `AttributeError` on a list, or a
    bare `KeyError('max_concurrent_subagents')` minutes into a fan-out with
    nothing on it naming the file. All four are now one `NeedsHuman` (exit 3) at
    load, naming the file and what is wrong with it.
    """

    @staticmethod
    def _load(tmp_path, text):
        (tmp_path / "hosts").mkdir(exist_ok=True)
        (tmp_path / caps.CAPABILITIES_FILE).write_text(text, encoding="utf-8")
        caps.clear_cache()
        return caps.load(base=str(tmp_path))

    def test_a_file_that_is_not_yaml_names_the_file(self, tmp_path):
        with pytest.raises(un.NeedsHuman) as excinfo:
            self._load(tmp_path, "hosts: [\n  broken")
        assert "not valid YAML" in str(excinfo.value)
        assert str(tmp_path / caps.CAPABILITIES_FILE) in str(excinfo.value)

    def test_a_document_that_is_not_a_mapping_is_refused(self, tmp_path):
        with pytest.raises(un.NeedsHuman) as excinfo:
            self._load(tmp_path, "- a\n- b\n")
        assert "not a mapping" in str(excinfo.value)

    def test_a_document_with_no_hosts_is_refused(self, tmp_path):
        """Otherwise `entry()` reports `Known hosts: ` and blames the caller."""
        with pytest.raises(un.NeedsHuman) as excinfo:
            self._load(tmp_path, "version: 1\n")
        assert "declares no hosts" in str(excinfo.value)

    def test_an_incomplete_entry_names_the_file_and_the_missing_key(
        self, tmp_path
    ):
        document = json.loads(json.dumps(DOCUMENT))
        del document["hosts"]["opencode"]["max_concurrent_subagents"]
        with pytest.raises(un.NeedsHuman) as excinfo:
            self._load(tmp_path, yaml.safe_dump(document, sort_keys=True))
        message = str(excinfo.value)
        assert "opencode" in message
        assert "max_concurrent_subagents" in message
        assert str(tmp_path / caps.CAPABILITIES_FILE) in message

    def test_a_broken_sibling_is_caught_by_a_run_that_never_names_it(
        self, tmp_path
    ):
        """The check belongs in `load`, not in `entry`.

        A dispatch on `claude-code` reads no opencode key, so an entry-scoped
        check would let a corrupt install run until something happened to touch
        the broken host — which on this data means `install-host --all` or the
        renderer, both of which run long after the install that broke it.
        """
        document = json.loads(json.dumps(DOCUMENT))
        del document["hosts"]["opencode"]["detach"]
        with pytest.raises(un.NeedsHuman) as excinfo:
            self._load(tmp_path, yaml.safe_dump(document, sort_keys=True))
        assert "opencode" in str(excinfo.value)
        assert "detach" in str(excinfo.value)

    def test_the_shipped_document_still_loads(self, tmp_path):
        """The floor is a floor: the real data passes it unchanged."""
        caps.clear_cache()
        assert caps.load()["hosts"], "the shipped capability data must load"


@pytest.mark.parametrize("host_id", HOST_IDS)
class TestProfileCompleteness:
    def test_all_three_verbs_are_present(self, host_id):
        host = HOSTS[host_id]
        for verb in caps.VERBS:
            assert verb in host, f"{host_id} answers no *{verb}* verb"

    def test_native_tool_exists_wherever_a_verb_is_native(self, host_id):
        host = HOSTS[host_id]
        native_like = {"subagents": "native", "ask": "structured", "detach": "native"}
        for verb, native_value in native_like.items():
            if host[verb] == native_value:
                assert (host.get("native_tool") or {}).get(verb), (
                    f"{host_id}.{verb} is '{native_value}' — the host's own tool — "
                    "but native_tool names none, so no sentence can be rendered"
                )

    def test_native_tool_names_nothing_a_verb_does_not_use(self, host_id):
        """The one place a host tool name may live must not accumulate extras."""
        host = HOSTS[host_id]
        native_like = {"subagents": "native", "ask": "structured", "detach": "native"}
        for verb in (host.get("native_tool") or {}):
            assert verb in caps.VERBS, f"{host_id}.native_tool.{verb} is not a verb"
            if host[verb] != native_like[verb]:
                # Legal (Codex documents its agent tool while dispatching through
                # the executor), but it must never be what gets dispatched.
                assert host[verb] != native_like[verb]

    def test_the_default_runner_is_in_its_own_runners_map(self, host_id):
        host = HOSTS[host_id]
        provider = host["default_subagent_runner"].split(":", 1)[0]
        assert provider in host["runners"], (
            f"{host_id}'s default_subagent_runner names provider '{provider}', "
            f"absent from its own runners map {sorted(host['runners'])}"
        )

    def test_the_default_runner_has_the_profiles_it_is_used_for(self, host_id):
        host = HOSTS[host_id]
        provider = host["default_subagent_runner"].split(":", 1)[0]
        profiles = set(host["runners"][provider]["profiles"])
        assert {"agent-readonly"} <= profiles, (
            f"{host_id}'s default runner cannot take a read/analyze dispatch "
            f"(profiles={sorted(profiles)}); a generate-only runner invents a "
            "result instead of failing (D3e)"
        )

    def test_subagent_tiers_iff_native(self, host_id):
        host = HOSTS[host_id]
        tiers = host["subagent_tiers"]
        if host["subagents"] == "native":
            assert tiers is None or ("full" in tiers), (
                f"{host_id} dispatches natively, so subagent_tiers must be a map "
                "with `full` or an explicit null (D2e)"
            )
        else:
            assert tiers is None, (
                f"{host_id} is subagents: {host['subagents']}, so its tiers come "
                "from providers.yaml driver.* — carrying a map here would be two "
                "sources for one fact"
            )

    def test_env_guards_are_three_lists_of_literal_names(self, host_id):
        guards = HOSTS[host_id]["env_guards"]
        assert set(guards) == {"guards", "session_scoped", "passthrough"}
        for key, names in guards.items():
            assert isinstance(names, list), (host_id, key)
            for name in names:
                assert "*" not in name and "?" not in name, (
                    f"{host_id}.env_guards.{key} contains the wildcard {name!r}; "
                    "literal names only"
                )

    def test_provenance_covers_every_field(self, host_id):
        host = HOSTS[host_id]
        governed = set(host) - {"provenance"}
        recorded = set(host["provenance"])
        assert governed <= recorded, (
            f"{host_id} has fields with no provenance: {sorted(governed - recorded)}"
        )
        assert recorded <= governed, (
            f"{host_id} records provenance for absent fields: "
            f"{sorted(recorded - governed)}"
        )

    def test_unknown_is_only_where_the_schema_permits_it(self, host_id):
        """`unknown` as a *value* belongs to nested_exec and timeout_override."""
        host = HOSTS[host_id]
        for field, value in host.items():
            if field in ("nested_exec", "timeout_override", "provenance"):
                continue
            assert value != "unknown", f"{host_id}.{field} is an unknown value"
        for key, value in host["timeout_override"].items():
            if key == "frontmatter":
                assert value != "unknown", (
                    f"{host_id}: whether a frontmatter timeout key exists is a "
                    "documented fact, not an open question"
                )


class TestConservativeResolution:
    """A `documented`-only value never selects the non-conservative path."""

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_native_subagents_require_verified_provenance(self, host_id):
        host = HOSTS[host_id]
        if host["subagents"] != "native":
            return
        assert host["provenance"]["subagents"] == "verified", (
            f"{host_id} dispatches subagents natively on `"
            f"{host['provenance']['subagents']}` provenance. Native dispatch is "
            "the non-conservative path: promote it only from a recorded smoke, "
            "because the failure is silent-then-fatal (a phase waits for a "
            "subagent that never starts until the host's cap kills it)."
        )

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_a_documented_cap_is_worked_around_not_trusted(self, host_id):
        host = HOSTS[host_id]
        ceiling = host["timeout_ceiling_s"]
        if not ceiling or host["provenance"]["timeout_ceiling_s"] == "verified":
            return
        if host["detach"] == "native":
            # The other legal workaround: the host's own background mechanism
            # steps out of the capped shell entirely. Legal only when that
            # mechanism is itself observed — otherwise the cap is being trusted
            # through a second unverified claim.
            assert host["provenance"]["detach"] == "verified", (
                f"{host_id} works a {ceiling}s cap around with a background "
                "mechanism whose own provenance is "
                f"`{host['provenance']['detach']}`. Two unverified claims do not "
                "make a conservative resolution."
            )
            return
        assert host["detach"] in ("python", "required"), (
            f"{host_id} documents a {ceiling}s command cap on unverified "
            "provenance but resolves *run detached* to neither `--detach` nor a "
            "verified background mechanism. A cap that is lower than documented "
            "must cost a poll cycle, not a killed fan-out."
        )

    def test_the_two_phase_three_hosts_are_executor_backed(self):
        """Codex and OpenCode: spawn-from-exec is unconfirmed, so `executor`."""
        for host_id in ("codex", "opencode"):
            assert HOSTS[host_id]["subagents"] == "executor", host_id
            assert HOSTS[host_id]["provenance"]["subagents"] == "documented"

    #: The ledger of what a *recorded* smoke actually observed, per host. A
    #: `verified` value outside this set is an unbacked claim; a value inside it
    #: that the smoke did not exercise is worse, because it retires the
    #: conservative resolution on nothing. Extend a row only together with the
    #: comment in `capabilities.yaml` that says which run promoted it.
    #:
    #: opencode, 1.18.12, 2026-08-08: installed shim discovered by name in a new
    #: session; uv preflight ran ahead of the first `uv run`; baked prefix
    #: resolved; `--` separator confirmed necessary headless; a detached `ask`
    #: fan-out polled through `status --wait` to terminal `done`; and four tool
    #: turns inside that single `opencode run`, the agent acting on each one's
    #: output and following `MULTI_LLM_RESULT.next` unprompted — which is the
    #: whole of the `multi_turn` claim. No subagent was dispatched and nothing
    #: was asked, so `subagents` and `ask` stay documented.
    #:
    #: codex 0.146.0 and opencode 1.18.18, 2026-08-24, the container
    #: documented-invocation row (`tests/test_container_documented_invocation.py`,
    #: `--creds none`): each host's shim was installed into a fresh `$HOME` and the
    #: published `/multi-llm …` string was typed into a fresh pty session. opencode
    #: expanded it — the turn it sent carried the shim's own text, `{SKILL_DIR}`
    #: substituted, the baked `--host … --yes --exec-subagents`, and the mode flag
    #: still adjacent to its argument. codex refused it — `Unrecognized command
    #: '/multi-llm'`, zero requests sent — while advertising the same skill to the
    #: model inside `<skills_instructions>`. That row promotes `invocation` on both
    #: hosts and nothing else: it graded a command surface, so codex's `shim`,
    #: `arguments` and `install_paths` stay where they were.
    #:
    #: codex 0.156.1, 2026-09-25, `codex-package-sandbox-posture`: every
    #: `tools/call` its mcp-tap recorded carried the session's `--model` as the turn
    #: metadata's `model`, which promotes `session_runner_provider` alone.
    #:
    #: codex 0.159.2 and opencode 1.18.33, 2026-10-01, T10 of
    #: plans/next-phase-command.md (`TestTheNextCommandLiveCheck`, rows
    #: `<host>[-package]-next-command`): on each host and route a session ran
    #: `generate-tasks` on `plans/next phase.md`, and the command its dispatcher
    #: printed was handed verbatim to a FRESH session of the same route; that
    #: session's receipt recorded `implement` with the fixture as the plan
    #: argument and no quote character in it. Twice per route, on two models
    #: (`~deepseek/deepseek-v4-flash-latest`, `deepseek/deepseek-v4.1-flash`):
    #: codex default runs/20261001T113914Z + 115800Z, codex package 113244Z +
    #: 115346Z, opencode default 113344Z + 115548Z, opencode package 112645Z +
    #: 115138Z. It promotes the four `next_command.<route>` cells and nothing
    #: else — a typed template, not the composer, not the model's copy of it.
    #:
    #: claude-code 2.1.285, 2026-10-01, T10 of plans/next-phase-command.md (row
    #: `claude-code-next-command`, `live_next_command` tier): the lane stages this
    #: host as a plain skill under `$HOME/.claude/skills/multi-llm`, so the route
    #: exercised is `skill`. A session ran `generate-tasks` on `plans/next phase.md`
    #: and the printed command was typed as the whole prompt of a FRESH `claude -p`
    #: session, which dispatched `implement` with the fixture unquoted. Twice:
    #: runs/20261001T114907Z on `openai/gpt-6-luna` through OpenRouter, and
    #: runs/20261001T121548Z on `sonnet` through the Claude subscription (`--creds
    #: fallback`). It promotes `next_command.skill` and nothing else: the plugin
    #: route (`next_command.default`), the Stop hook (not installed under plain-skill
    #: staging) and TUI typing were not exercised.
    #:
    #: codex 0.159.2 / 0.163.0 / 0.164.0, 2026-10-23, the first release
    #: (codex-v0.1.0, ae089856): `package` on the two Git-route records rule (j)
    #: names — the offline `codex-package-git-route` row in the stamped gate
    #: session, and the manual network check from GitHub
    #: (records/20261023T132656Z-codex-release-network-check.json, six of six),
    #: plus the published `--ref release` line re-run after promotion. It
    #: promotes `package` and nothing else.
    RECORDED_SMOKES = {
        "claude-code": {"next_command.skill"},
        "codex": {
            "invocation", "session_runner_provider",
            "next_command.default", "next_command.package", "package",
        },
        "opencode": {
            "shim", "installable", "install_paths", "reload", "arguments",
            "headless_invocation", "detach", "multi_turn", "invocation",
            "next_command.default", "next_command.package",
        },
    }

    #: Cells the reference host carried as `verified` before this ledger existed
    #: (D5b, 227af2d), by its own use as the host the skill was written on, with no
    #: recorded run id behind them. Exactly these two are exempt from the
    #: unexercised-verb check below; `next_command` is deliberately not here.
    PRE_LEDGER_VERIFIED = {("claude-code", "subagents"), ("claude-code", "ask")}

    @classmethod
    def _unbacked(cls, hosts):
        """``{host: [dotted field, …]}`` claiming ``verified`` with no ledger row.

        Map-valued rows are flattened first (``next_command.package``), so a
        promoted route is held to the ledger like any other field.
        """
        unbacked = {}
        for host_id, host in hosts.items():
            if host_id == "claude-code":
                continue
            verified = {
                field for field, value in _flat_provenance(host["provenance"]).items()
                if value == "verified"
            }
            extra = sorted(verified - cls.RECORDED_SMOKES.get(host_id, set()))
            if extra:
                unbacked[host_id] = extra
        return unbacked

    def test_no_value_is_verified_without_a_recorded_run(self):
        reference = _flat_provenance(HOSTS["claude-code"]["provenance"])
        assert "verified" in reference.values(), (
            "the reference host has observed *something*"
        )
        unbacked = self._unbacked(HOSTS)
        assert not unbacked, (
            f"verified provenance with no recorded smoke behind it: {unbacked}"
        )

    def test_a_nested_verified_route_is_held_to_the_ledger(self, monkeypatch):
        """A promoted ``next_command`` route is a dict row; it is caught anyway.

        Since T10 promoted every non-reference route, the unbacked case is made
        by taking the ledger row away rather than by promoting a cell.
        """
        ledger = {host: set(fields) for host, fields in self.RECORDED_SMOKES.items()}
        ledger["codex"].discard("next_command.package")
        monkeypatch.setattr(type(self), "RECORDED_SMOKES", ledger)
        hosts = json.loads(json.dumps(HOSTS))
        hosts["codex"]["provenance"]["next_command"]["package"] = "verified"
        assert self._unbacked(hosts) == {"codex": ["next_command.package"]}

    def test_the_smoke_ledger_has_no_stale_rows(self):
        """A ledger entry that nothing claims is a promotion silently reverted."""
        for host_id, fields in self.RECORDED_SMOKES.items():
            assert host_id in HOSTS, f"ledger names unknown host '{host_id}'"
            prov = _flat_provenance(HOSTS[host_id]["provenance"])
            for field in fields:
                assert prov.get(field) == "verified", (
                    f"{host_id}.{field} is recorded as smoke-verified but the "
                    f"entry says '{prov.get(field)}'"
                )

    def test_a_ledger_row_names_a_route_by_its_dotted_path(self):
        """The stale-row check reads a nested route the way the ledger names it."""
        flat = _flat_provenance({
            "invocation": "verified",
            "next_command": {"default": "documented", "package": "verified"},
        })
        assert flat == {
            "invocation": "verified",
            "next_command.default": "documented",
            "next_command.package": "verified",
        }

    def test_an_unexercised_verb_is_never_verified(self):
        """The two verbs the recorded smokes could not reach stay conservative."""
        for host_id in self.RECORDED_SMOKES:
            prov = HOSTS[host_id]["provenance"]
            for verb in ("subagents", "ask"):
                if (host_id, verb) in self.PRE_LEDGER_VERIFIED:
                    continue
                assert prov[verb] != "verified", (
                    f"{host_id}.{verb} was promoted to verified, but no recorded "
                    "smoke dispatched a subagent or rendered a question. "
                    "Promoting it retires the conservative resolution on nothing."
                )


# ===========================================================================
# 2. Shims and entries are in bijection
# ===========================================================================

class TestShimBijection:
    def test_every_entry_declares_a_distinct_shim_target(self):
        seen = {}
        for host_id, host in HOSTS.items():
            key = (host["shim"]["template"], host["shim"]["installed"])
            assert key not in seen, (
                f"{host_id} and {seen[key]} declare the same shim target {key}; "
                "entries and shims are 1:1"
            )
            seen[key] = host_id

    def test_the_one_host_nobody_renders_is_the_reference_host(self):
        """There is deliberately no hosts/claude-code/SKILL.md (D5c)."""
        not_rendered = [
            host_id for host_id, host in HOSTS.items()
            if host["shim"]["template"] is None
        ]
        assert not_rendered == ["claude-code"], not_rendered
        assert HOSTS["claude-code"]["installable"] is False
        assert HOSTS["claude-code"]["shim"]["installed"] == "SKILL.md"
        assert not (SKILL_ROOT / "hosts" / "claude-code").exists(), (
            "a copy of the Claude Code shim under hosts/ could only be dead text "
            "or a duplicate to keep in sync"
        )

    def test_an_installable_entry_declares_a_template(self):
        for host_id, host in HOSTS.items():
            if host["installable"]:
                assert host["shim"]["template"], host_id

    def test_every_shim_file_on_disk_resolves_to_an_entry(self):
        """The half of the bijection that is total today.

        The per-host `.tmpl` files are the install design's to add; until then the
        only shim on disk is the reference host's, and this assertion becomes total
        the moment a template appears without an entry.
        """
        declared = {
            host["shim"]["template"] for host in HOSTS.values()
            if host["shim"]["template"]
        }
        on_disk = {
            path.relative_to(SKILL_ROOT).as_posix()
            for path in SKILL_ROOT.glob("hosts/*/shim.md.tmpl")
        }
        assert on_disk <= declared, (
            f"shim template(s) with no capabilities.yaml entry: {on_disk - declared}"
        )
        assert (SKILL_ROOT / HOSTS["claude-code"]["shim"]["installed"]).exists()

    def test_snippets_are_host_independent(self):
        """A snippet that would need an `if` is two snippets, or is not a snippet."""
        snippets = sorted((SKILL_ROOT / "hosts" / "snippets").glob("*.md"))
        assert snippets, "the shared shim partials are missing"
        for path in snippets:
            text = path.read_text(encoding="utf-8")
            for host_id in HOST_IDS:
                assert host_id not in text, (
                    f"{path.name} names the host '{host_id}'; anything that varies "
                    "per host belongs in capabilities.yaml, in a verb slot, or in "
                    "that host's own template"
                )
            for tool in ("Task", "AskUserQuestion", "run_in_background"):
                assert tool not in text, (
                    f"{path.name} names the host tool '{tool}'"
                )


# ===========================================================================
# 3. The three verb sentences
# ===========================================================================

class TestVerbResolution:
    def test_the_vocabulary_is_closed_at_three(self):
        assert caps.VERBS == ("subagents", "ask", "detach")
        assert set(caps.VERB_WORDING) == set(caps.VERBS)
        assert set(caps.VERB_SLOTS) == set(caps.VERBS)

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_every_host_renders_exactly_three_sentences(self, host_id):
        sentences = caps.verb_sentences(host_id)
        assert len(sentences) == 3
        for verb, sentence in zip(caps.VERBS, sentences):
            wording = caps.VERB_WORDING[verb]
            assert sentence.lower().startswith(f"**{wording}**"), (
                f"{host_id}'s {verb} sentence does not open with the core's own "
                f"wording {wording!r}: {sentence[:60]!r}"
            )

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_no_sentence_defers_or_conditions(self, host_id):
        """The agent gets a concrete imperative, never a condition to evaluate."""
        for sentence in caps.verb_sentences(host_id):
            lowered = sentence.lower()
            for banned in ("if your host", "capabilities.yaml",
                           "host-capabilities.md", "depending on", "consult"):
                assert banned not in lowered, (
                    f"{host_id}: a verb sentence says {banned!r} — the resolution "
                    "is baked at render time, never deferred to the agent"
                )

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_a_sentence_names_a_tool_only_when_its_verb_is_native(self, host_id):
        host = HOSTS[host_id]
        native_like = {"subagents": "native", "ask": "structured", "detach": "native"}
        for verb in caps.VERBS:
            sentence = caps.verb_sentence(host_id, verb)
            tool = (host.get("native_tool") or {}).get(verb)
            if host[verb] == native_like[verb]:
                assert f"`{tool}`" in sentence
            elif tool:
                assert f"`{tool}`" not in sentence, (
                    f"{host_id}'s {verb} sentence names `{tool}` although the verb "
                    f"resolves to '{host[verb]}' — the tool is documented, not "
                    "dispatched"
                )

    def test_the_executor_hosts_are_told_to_use_the_flag(self):
        for host_id in ("codex", "opencode"):
            assert "--exec-subagents" in caps.verb_sentence(host_id, "subagents")

    def test_the_native_host_is_told_not_to_use_the_flag(self):
        sentence = caps.verb_sentence("claude-code", "subagents")
        assert "Never pass `--exec-subagents`" in sentence

    def test_the_native_detach_sentence_carries_the_whole_loop(self):
        """T8: on this host the launch is the *start* of *run detached*.

        A native background run writes no ``run`` object, so ``classify_run``
        can only ever answer ``no_run`` and the poll's exit code can never end
        the loop. Everything that makes the loop correct there is stated in this
        one sentence because nothing else on the host says it: the poll itself,
        the ``--host`` the wait clamp reads (no launcher printed a command to
        copy it from), the per-call timeout (a tool parameter, so none of
        ``host_install.TIMEOUT_CHANNELS`` protects it), and the background
        mechanism's own exit signal as the stop condition.
        """
        sentence = caps.verb_sentence("claude-code", "detach")
        assert "Never pass `--detach`" in sentence
        for fragment in (
            "--view run",          # it polls at all
            "--wait",
            "--timeout <interval>",
            "120-300 s",           # rule 5's range, not the host's hard cap
            "`progress:`",         # rule 1: relay the rendered line
            "Spell `--host` out yourself",
            # The call's own timeout, not the command's — and the margin is
            # `WAIT_CAP_MARGIN_S`, never a typed `30` (F6, 2026-09-20). Pinning
            # the literal here was the reason moving that constant could have
            # left the number standing on every surface an agent reads.
            f"plus {caps.WAIT_CAP_MARGIN_S:.0f} s",
            "except `1`",          # rule 2, narrowed for this host
            "the background mechanism reports the process exited",
            "orchestrator-run.log",
        ):
            assert fragment in sentence, (
                "the native *run detached* sentence lost "
                f"{fragment!r}; SKILL.md is a byte-identical copy of this "
                "string, so what is dropped here is dropped on the one host "
                "the poll loop was written for.\n"
                f"  sentence: {sentence}"
            )

    def test_the_native_detach_sentence_says_which_models_and_which_finished(self):
        """2026-09-20: the launch named a count and never a model.

        On this host the whole loop is this one sentence, so the two things a
        real run was missing have to be in it. The **roster** is a call of its
        own, made once between the launch and the loop and never waiting: the
        first poll waits 120-300 s, and a roster that reaches the user three
        minutes after launch is not a roster. And what a poll relays gains the
        `just finished:` line, which is the run's own answer to *which* of them
        finished \u2014 `models 4/5 done` names no model, and a name an agent derived
        by diffing two tables is a name it can get wrong.
        """
        sentence = caps.verb_sentence("claude-code", "detach")
        for fragment in (
            "one **non-waiting**",       # the roster call, and that it does not wait
            "no `--wait`, no `--timeout`",
            "`models:` table once",      # relayed, and relayed once
            "which CLI and model each of them runs",
            "repeat that one call once if no table printed yet",
            "`just finished:` line",     # the per-poll half
            "turned terminal during that wait",
        ):
            assert fragment in sentence, (
                "the native *run detached* sentence lost "
                f"{fragment!r}; SKILL.md is a byte-identical copy of this "
                "string, so what is dropped here is dropped on the one host "
                "the poll loop was written for.\n"
                f"  sentence: {sentence}"
            )

    def test_the_native_detach_sentence_overclaims_neither_reading(self):
        """F-2026-09-20: both halves of the sentence claimed more than the code.

        **The roster half.** "repeat that one call once if no table printed yet"
        was the only escape hatch, and on a re-used phase dir a table *always*
        prints — the predecessor's — so an agent following the sentence relayed
        a finished run's models as this run's roster. The discriminator is on
        screen: the head line's tense, which ``run_status._verdict_text`` reads
        off the checkpoint, and the ``progress:`` step, ``fanout`` while a
        fan-out runs and ``complete`` once one finished.

        **The relay half.** The absence of ``just finished:`` is not a claim
        that nothing finished — that line is a difference between two readings
        taken inside one ``--wait`` call, so a model that finishes in the gap
        between polls is already terminal when the next poll takes its baseline.
        The sentence hands the agent the fallback instead: copy the newly
        terminal row's spec and state word out of the table, which is a copy and
        not the recount the counts rule forbids.

        Every word the sentence tells the agent to look for is pinned against
        the code that emits it, because a grep for a phrase this same edit wrote
        proves only that the phrase is there.
        """
        from utils import detach as detach_mod
        from utils import run_status

        def head(state, step, done):
            return run_status._verdict_text({
                "verdict": detach_mod.VERDICT_NO_RUN,
                "run": {},
                "state": state,
                "progress": {
                    "step": step, "models_total": 2, "models_done": done,
                },
            })

        assert head("complete", "complete", 2).startswith("finished in foreground")
        assert head("models_running", "fanout", 0).startswith("running in foreground")
        assert "complete" in run_status.CHECKPOINT_TERMINAL_STEPS
        skill_root = Path(caps.__file__).parent.parent
        for src in (
            "utils/review_orchestrator_base.py",
            "ask_orchestrator.py",
            "code_review_orchestrator.py",
        ):
            body = (skill_root / src).read_text(encoding="utf-8")
            assert 'step="fanout"' in body, f"{src} no longer reports `fanout`"
            assert '"progress": None' in body, (
                f"{src} no longer drops the predecessor's `progress` in the "
                "write that claims the dir"
            )

        sentence = caps.verb_sentence("claude-code", "detach")
        for fragment, why in (
            ("a table that *does* print can still be the previous run's",
             "the sentence still treats any printed table as this run's roster"),
            ("`finished in foreground (…)` rather than `running`",
             "the sentence names no tense discriminator"),
            ("`progress:` step that is not `fanout`",
             "the sentence names no step discriminator"),
            ("That step is the whole test",
             "the sentence no longer says the step is the ONLY test. F-2026-"
             "09-21: as an OR with the head line's tense it never accepted "
             "this run's roster on a dir an earlier detached run had used, "
             "because only `claim_run` drops a predecessor's `run` record"),
            ("make that call again rather than relaying it",
             "the sentence says what to notice and not what to do"),
            ("Its absence claims nothing",
             "the sentence still reads a missing `just finished:` line as "
             "proof that nothing finished"),
            ("copying its spec and state word out of",
             "the sentence gives no fallback for a finish no line named"),
            ("a copy, never a recount",
             "the sentence does not reconcile the fallback with the counts "
             "rule, so the two read as a contradiction"),
            ("every poll compares its `models:` table",
             "the comparison is conditional again; one model finishing in "
             "the gap and a second during the wait share one "
             "`just finished:` line that names only the second"),
            ("names no verdict for this one",
             "a `0` or a `3` reads as a verdict again on the one host where "
             "no selector names this invocation"),
            ("relay the rendered line exactly as on a `2`",
             "the sentence tells the agent to relay NOTHING on a `0` or a "
             "`3`, which drops every `progress:` line of a run whose phase "
             "dir carries a predecessor's terminal record"),
        ):
            assert fragment in sentence, f"{why} (missing {fragment!r})"
        for gone, why in (
            ("rather than `running`, or under a `progress:` step",
             "the roster test ORs the head line's tense back in"),
            ("when the counts moved with no such line",
             "the `models:` comparison is conditional again"),
            ("belongs to an earlier run in the same dir and is relayed as "
             "nothing",
             "a poll whose exit code belongs to an earlier run relays "
             "nothing again, which is the silence Layer 2 exists to remove"),
        ):
            assert gone not in sentence, why

    def test_only_the_native_detach_sentence_names_a_call_timeout(self):
        """The two `--detach` resolutions inherit `--host` from the launcher.

        They also run their wait inside a shell command rather than a tool call,
        so the per-call timeout rule is meaningless there — and a rule that does
        not apply is a rule an agent has to decide about.
        """
        for value in ("python", "required"):
            assert f"plus {caps.WAIT_CAP_MARGIN_S:.0f} s" not in caps._DETACH[value]
            assert "--timeout" not in caps._DETACH[value]
            assert str(caps.NATIVE_LAUNCH_TIMEOUT_MS) not in caps._DETACH[value]

    def test_the_native_launch_names_its_own_timeout(self):
        """PD-1 (2026-10-27): the sentence sized every poll and never the launch.

        Claude Code stops a `run_in_background` command that names no `timeout`
        at the tool's default deadline, ten minutes in a headless session: a
        timeout-less `sleep 700` was stopped at 600.9 s, and a live
        `--review-commits` fan-out at 599.7 s (`"exit":143,"next":null`), after
        which the host improvised a `--force` re-run. The launch now names the
        parameter's documented maximum in its unit, derived from one constant,
        and the loop's stop condition covers a stop at that limit too.
        """
        sentence = caps.verb_sentence("claude-code", "detach")
        ms = caps.NATIVE_LAUNCH_TIMEOUT_MS
        # The tool's documented background maximum (2.1.311), whole hours so
        # the derived "N hours" reads true.
        assert ms == 7_200_000 and ms % 3_600_000 == 0
        for fragment in (
            "give that launch call a timeout of its own",
            f"the longest the parameter takes: {ms} here, in milliseconds",
            f"({ms // 3_600_000} hours;",
            "clamps it rather than refusing it",
            "exit `143`, `next` `null`",
        ):
            assert fragment in sentence, (
                f"the native *run detached* sentence lost {fragment!r}; SKILL.md "
                f"is a byte-identical copy of it.\n  sentence: {sentence}"
            )
        # The launch clause comes before the first status call it hands over to.
        assert sentence.index(str(ms)) < sentence.index("one **non-waiting**")
        # Derived, never typed into the sentence's source.
        source = Path(caps.__file__).read_text(encoding="utf-8")
        assert f"takes: {ms} here" not in source
        # Milliseconds and Claude Code's number: only claude-code may read it.
        assert [h for h in HOST_IDS if HOSTS[h]["detach"] == "native"] == ["claude-code"]

    def test_a_stop_at_the_time_limit_is_offered_its_real_ways_forward(self):
        """Fable review, finding 3 (2026-10-27): the maximum is still a ceiling.

        A run past it was left with "report … rather than retrying" and nothing
        to offer, which is the run-1 incident minus the improvisation. Probed
        on a copy of the phase dir that kill left, with no model call: `status`
        names no command, a plain relaunch over a dir holding a result is exit
        3, `--force` resumes with `[SKIP]`, and `--reaggregate` reports with what
        is there. The sentence offers those, and says which models have a file,
        because a `done` row can have none (`tests/test_e2e_failures.py`
        `TestAFanOutCutOffMidRun` pins all three on the real `main()`).
        """
        sentence = caps.verb_sentence("claude-code", "detach")
        for fragment in (
            "report a non-zero exit status with that path rather than retrying.",
            "`status` still reading `running in foreground`",
            "offer the user each of these and run none yourself",
            "made again the same way with `--force` added, which resumes",
            "(`[SKIP]`)",
            # Fable's second pass, finding 2: review-code's guard alone; the
            # other review modes and ask re-run plainly and `[SKIP]`.
            "without it, review-code and the commit modes refuse a dir holding "
            "results, exit `3`",
            "never `--rerun-all`",
            "on a review mode with at least one such file, its `--reaggregate`",
            "a `done` row can have none",
            "a composite's `[IMPROVE_RERUN]`), offer that instead",
        ):
            assert fragment in sentence, (
                f"the native *run detached* sentence lost {fragment!r}.\n"
                f"  sentence: {sentence}"
            )
        # The words the run prints, pinned against the code that prints them.
        import code_review_orchestrator as cro_mod
        from utils import run_status

        body = Path(cro_mod.__file__).read_text(encoding="utf-8")
        assert "- already has results" in body and "Partial completion detected" in body
        assert "a dir holding results is refused" not in sentence
        # The §6 rule "offer only the ways forward the stop itself names" would
        # leave this stop with none: SKILL.md carves it out (finding 4).
        skill = (Path(caps.__file__).parent.parent / "SKILL.md").read_text(encoding="utf-8")
        assert ("A stop at the background mechanism's time limit is the one stop whose "
                "ways forward are not its own to name: it names none, so offer the ones "
                "*Run detached* (section 5) lists for it") in skill
        assert skill.index("offer only the ways forward the stop itself names") < skill.index(
            "is the one stop whose ways forward are not its own to name")
        assert run_status._verdict_text({
            "verdict": "no_run", "run": {}, "state": "models_running",
            "progress": {"step": "fanout", "models_total": 2, "models_done": 2},
        }).startswith("running in foreground")

    def test_the_margin_is_one_constant_two_registers_and_no_typed_copy(self):
        """F6, 2026-09-20: the margin was a literal in six places.

        `run_status`'s comment said it was "named once here rather than twice so
        the two cannot drift" — but `30` was typed into the constant, this verb
        sentence, `SKILL.md`, `references/host-capabilities.md` and four
        instruction cores, and the only test that looked pinned the *string*
        `"plus 30 s"`. Moving the constant would have gone green with every
        agent-read surface still on the old number.

        Two registers, one owner. `status --wait` clamps a resolved deadline to
        `ceiling - margin`; a Claude Code poll gives its tool call
        `interval + margin`. The sentence interpolates the constant, so the two
        generated copies (`SKILL.md` is byte-compared to it,
        `references/host-capabilities.md` is generated from it) follow it; the
        four hand-written cores are pinned against it in
        `tests/test_instruction_portability.py`.
        """
        from utils import run_status

        assert run_status.WAIT_CAP_MARGIN_S is caps.WAIT_CAP_MARGIN_S, (
            "run_status re-exports the constant; it must not grow a second copy"
        )
        # Interpolated, not typed: raise the margin and the sentence moves.
        rendered = caps._DETACH["native"].format(tool="run_in_background")
        assert f"plus {caps.WAIT_CAP_MARGIN_S:.0f} s," in rendered
        source = Path(caps.__file__).read_text(encoding="utf-8")
        assert f"plus {caps.WAIT_CAP_MARGIN_S:.0f} s" not in source, (
            "the *run detached* sentence types the margin out instead of "
            "interpolating WAIT_CAP_MARGIN_S — which is how the number came to "
            "stand in six places while the comment claimed it stood in one"
        )

    def test_the_native_call_timeout_names_its_unit(self):
        """F-2026-09-21: "the interval plus 30 s" never said *in what*.

        The per-call timeout is the one number in this sentence that is a tool
        parameter rather than a flag, and on the only host the template renders
        for that parameter is **milliseconds** — the Bash tool's `timeout`,
        120000 by default and 600000 at the cap. An agent that read the seconds
        and passed `330` got a 0.33 s call, killed before `status` could answer,
        and rule 2 then told it to re-issue with the right call timeout: a spin,
        not a poll loop. So the unit, one worked value and the value it is *not*
        all have to be on the surface the agent reads, and both numbers are
        derived from `WAIT_CAP_MARGIN_S` so a changed margin cannot leave a
        stale example behind.
        """
        sentence = caps.verb_sentence("claude-code", "detach")
        # The widest interval the same sentence offers, so the largest call
        # timeout following it can ever ask for.
        seconds = int(300 + caps.WAIT_CAP_MARGIN_S)
        assert "milliseconds here" in sentence, (
            "the native *run detached* sentence states a call timeout in "
            "seconds without naming the unit its host's parameter takes; "
            f"sentence: {sentence}"
        )
        for fragment in (f"{seconds * 1000} for a 300 s interval", f"not {seconds},"):
            assert fragment in sentence, (
                f"the worked call timeout lost {fragment!r} — the unit alone "
                "still leaves the conversion to the agent"
            )
        # Following it can never name a value the host would clamp.
        assert seconds <= caps.timeout_ceiling_s("claude-code"), (
            "the sentence's own widest interval plus the margin now exceeds "
            "this host's documented per-call ceiling"
        )
        # Derived, never typed: raise the margin and both numbers move with it.
        source = Path(caps.__file__).read_text(encoding="utf-8")
        for fragment in (f"{seconds * 1000} for a ", f"not {seconds},"):
            assert fragment not in source, (
                "the worked call timeout is typed out instead of derived from "
                "WAIT_CAP_MARGIN_S, so the margin and its example can drift"
            )
        # The unit is host-specific; this template is keyed by the resolution.
        native = [h for h in HOST_IDS if HOSTS[h]["detach"] == "native"]
        assert native == ["claude-code"], (
            f"{native} resolve *run detached* to `native` and share one "
            "sentence that says the per-call timeout is in milliseconds — "
            "true of claude-code's Bash tool, a guess about any other host. "
            "Give the unit a slot before adding the host."
        )

    def test_a_fourth_verb_is_a_keyerror(self):
        with pytest.raises(KeyError):
            caps.verb_sentence("claude-code", "run-tests")

    def test_a_native_verb_without_a_tool_is_exit_three(self, fixture_tree):
        root, write = fixture_tree
        document = yaml.safe_load(
            (root / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
        )
        document["hosts"]["claude-code"]["native_tool"].pop("subagents")
        write(document)
        with pytest.raises(un.NeedsHuman) as excinfo:
            caps.verb_sentence("claude-code", "subagents", str(root))
        assert excinfo.value.exit_code == 3
        assert "native_tool.subagents" in str(excinfo.value)


class TestRecoveryInstruction:
    """`multi_turn` decides whom a failed batch's retry instruction addresses."""

    def test_the_recorded_run_makes_opencode_multi_turn(self):
        """Four tool turns inside one `opencode run`, acted on by its agent."""
        assert caps.is_multi_turn("opencode") is True
        assert HOSTS["opencode"]["provenance"]["multi_turn"] == "verified"

    def test_a_multi_turn_host_is_told_to_re_invoke(self):
        lines = caps.retry_instruction(
            "claude-code", failed_ids="t-1", retry_command="uv run … --force"
        )
        assert lines[0].startswith("Re-invoke to retry only the failures (t-1)")
        assert lines[1] == "  uv run … --force"

    def test_a_single_turn_host_addresses_the_human(self, fixture_tree):
        """The instruction must not be addressed to an agent that never reads it."""
        root, write = fixture_tree
        document = yaml.safe_load(
            (root / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
        )
        document["hosts"]["opencode"]["multi_turn"] = False
        write(document)
        lines = caps.retry_instruction(
            "opencode", failed_ids="t-1", retry_command="uv run … --force",
            base=str(root),
        )
        assert "no agent turn follows" in lines[0]
        assert "a human has to act on it" in lines[0]
        assert not lines[0].startswith("Re-invoke")
        # Same fix either way: the command is the payload, and it survives.
        assert lines[1] == "  uv run … --force"

    def test_an_unresolvable_host_still_names_the_retry_command(self):
        """A broken install must not cost the caller the command to re-run."""
        lines = caps.retry_instruction(
            "no-such-host", failed_ids="t-1", retry_command="uv run … --force"
        )
        assert lines[1] == "  uv run … --force"

    def test_a_missing_command_falls_back_rather_than_dangling(self):
        lines = caps.retry_instruction("claude-code", failed_ids="t-1")
        assert "re-run the same command" in lines[1]

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_neither_addressee_is_told_to_consult_the_matrix(self, host_id):
        """The addressee is resolved here; the agent reads a finished sentence."""
        for line in caps.retry_instruction(
            host_id, failed_ids="t-1", retry_command="cmd"
        ):
            lowered = line.lower()
            for banned in ("capabilities.yaml", "host-capabilities.md",
                           "if your host", "consult", "depending on"):
                assert banned not in lowered, (host_id, line)


# ===========================================================================
# 4. Consumers read the data
# ===========================================================================

class TestHostResolution:
    def test_the_default_is_the_reference_host(self, monkeypatch):
        monkeypatch.delenv(caps.HOST_ENV, raising=False)
        assert caps.resolve_host_id([]) == "claude-code"
        assert caps.DEFAULT_HOST == "claude-code"

    def test_the_flag_wins_over_the_env(self, monkeypatch):
        monkeypatch.setenv(caps.HOST_ENV, "opencode")
        assert caps.resolve_host_id(["--host", "codex"]) == "codex"
        assert caps.resolve_host_id(["--host=codex"]) == "codex"
        assert caps.resolve_host_id([]) == "opencode"

    def test_an_unknown_host_is_exit_three_naming_the_known_ones(self, capsys):
        assert multi_llm.main(["review-plan", "p.md", "--host", "nope"]) == 3
        err = capsys.readouterr().err
        assert "unknown host 'nope'" in err
        for host_id in HOST_IDS:
            assert host_id in err

    def test_an_omitted_flag_changes_nothing(self, monkeypatch, capsys):
        """The additive property: no `--host` behaves exactly as before it existed."""
        monkeypatch.delenv(caps.HOST_ENV, raising=False)
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capture(captured))
        assert multi_llm.main(["review-plan", "plans/a.md", "--quick"]) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", "plans/a.md", "--quick"]

    def test_the_flag_is_forwarded_not_consumed(self, monkeypatch, capsys):
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capture(captured))
        assert multi_llm.main(["review-plan", "p.md", "--host", "codex"]) == 0
        capsys.readouterr()
        assert captured["argv"] == ["--plan-file", "p.md", "--host", "codex"]

    def test_the_result_line_is_still_emitted_on_an_unknown_host(self, capsys):
        multi_llm.main(["review-plan", "p.md", "--host", "nope"])
        payload = parse_result_line(capsys.readouterr().out)
        assert payload["exit"] == 3


class TestExecutorEligibility:
    def test_every_shipped_host_has_an_executor_capable_runner(self):
        for host_id in HOST_IDS:
            assert caps.executor_capable_runners(host_id), host_id

    def test_a_generate_only_host_refuses_the_flag(self, fixture_tree, capsys,
                                                  monkeypatch):
        """D1a/D4c left this hook empty; the data fills it."""
        root, write = fixture_tree
        document = yaml.safe_load(
            (root / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
        )
        for spec in document["hosts"]["codex"]["runners"].values():
            spec["profiles"] = ["generate"]
        write(document)
        monkeypatch.setattr(caps, "load", lambda base=None: document)
        assert multi_llm.main([
            "review-plan", "p.md", "--host", "codex", "--exec-subagents",
        ]) == 3
        err = capsys.readouterr().err
        assert "--exec-subagents is not available on host 'codex'" in err
        assert "agent-readonly" in err

    def test_the_flag_is_accepted_where_a_runner_qualifies(self, monkeypatch,
                                                          capsys):
        captured = {}
        monkeypatch.setattr(multi_llm, "_load_target", _capture(captured))
        assert multi_llm.main([
            "review-plan", "p.md", "--host", "codex", "--exec-subagents",
        ]) == 0
        capsys.readouterr()
        assert "--exec-subagents" in captured["argv"]


class TestCeilingUsesTheData:
    def test_the_host_leg_of_the_minimum_comes_from_the_entry(self, monkeypatch):
        from utils import subagent_executor as executor

        monkeypatch.setenv(caps.HOST_ENV, "opencode")
        caps.clear_cache()
        record = executor.effective_ceiling([])
        assert record["host_max_concurrent_subagents"] == (
            HOSTS["opencode"]["max_concurrent_subagents"]
        )

    def test_mutating_the_entry_moves_the_ceiling(self, fixture_tree, monkeypatch):
        from utils import subagent_executor as executor

        root, write = fixture_tree
        document = yaml.safe_load(
            (root / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
        )
        document["hosts"]["opencode"]["max_concurrent_subagents"] = 1
        write(document)
        monkeypatch.setattr(caps, "load", lambda base=None: document)
        monkeypatch.setenv(caps.HOST_ENV, "opencode")
        record = executor.effective_ceiling([])
        assert record["host_max_concurrent_subagents"] == 1
        assert record["effective"] == 1

    def test_an_explicit_host_max_still_wins(self):
        from utils import subagent_executor as executor

        record = executor.effective_ceiling([], host_max=2)
        assert record["host_max_concurrent_subagents"] == 2

    def test_the_host_leg_follows_the_argv_a_shim_baked_in(
        self, monkeypatch, tmp_path
    ):
        """`--host` is the *primary* leg of `resolve_host_id`, not the env var.

        A rendered shim declares its host by baking `--host <id>` into the
        dispatcher line and sets no environment variable at all, so a run on any
        host but the reference one reaches `_host_ceiling` with `$MULTI_LLM_HOST`
        unset. Reading the env alone answered `claude-code`'s 10 on every one of
        them: an opencode host-lane run recorded
        `host_max_concurrent_subagents: 10` against an entry that declares 4.
        """
        from utils import subagent_executor as executor

        monkeypatch.delenv(caps.HOST_ENV, raising=False)
        monkeypatch.setattr(sys, "argv", [
            "multi_llm.py", "review-plan", str(tmp_path / "plan.md"),
            "--host", "opencode", "--exec-subagents", "--yes",
        ])
        assert HOSTS["opencode"]["max_concurrent_subagents"] != (
            HOSTS[caps.DEFAULT_HOST]["max_concurrent_subagents"]
        ), "this regression is only observable while the two entries differ"
        record = executor.effective_ceiling([])
        assert record["host_max_concurrent_subagents"] == (
            HOSTS["opencode"]["max_concurrent_subagents"]
        )

    def test_the_write_scheduler_takes_the_same_host_leg(
        self, monkeypatch, tmp_path
    ):
        """No orchestrator passes `host_max`, so `None` has to mean *look it up*.

        Read as "this phase has no host leg" it silently deleted the leg from
        every real write run — both construction sites
        (`implement_orchestrator`, `apply_orchestrator_base`) are keyword-complete
        without it — and left the batch report recording a null where the read
        half records a number.
        """
        from utils.write_batch_scheduler import WriteBatchScheduler

        monkeypatch.delenv(caps.HOST_ENV, raising=False)
        monkeypatch.setattr(sys, "argv", [
            "multi_llm.py", "implement", str(tmp_path / "plan.md"),
            "--host", "opencode", "--exec-subagents", "--yes",
        ])
        scheduler = WriteBatchScheduler(
            tasks=[],
            phase_dir=str(tmp_path),
            kind="implement",
            output_dir=str(tmp_path),
            repo_root=str(tmp_path),
            log=lambda line: None,
        )
        assert scheduler.host_max == HOSTS["opencode"]["max_concurrent_subagents"]

    def test_unreadable_data_bounds_nothing_rather_than_failing(self, monkeypatch):
        from utils import subagent_executor as executor

        monkeypatch.setattr(caps, "load", lambda base=None: (_ for _ in ()).throw(
            un.NeedsHuman("boom")
        ))
        record = executor.effective_ceiling([])
        assert record["host_max_concurrent_subagents"] is None
        assert record["effective"] >= 1


class TestTheForegroundCommandCeiling:
    """``timeout_ceiling_s``, the second thing read off a host entry at runtime.

    `capabilities.yaml`'s own "not on any agent read path" note names
    `multi_llm.py` as a legitimate consumer, so `status --wait` resolving the
    ceiling here — rather than a number being baked into a rendered surface — is
    the indirection this module exists for.
    """

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_the_accessor_answers_the_entry(self, host_id):
        assert caps.timeout_ceiling_s(host_id) == HOSTS[host_id]["timeout_ceiling_s"]

    def test_zero_means_no_documented_cap_not_a_cap_of_zero(self):
        """The falsy special case `host_install.timeout_note` and
        `render_capabilities._timeout_cell` already make, restated where the
        clamp reads it: codex ships `0`, and a caller that subtracted a margin
        from it would clamp every codex poll to nothing."""
        assert caps.timeout_ceiling_s("codex") == 0
        host, ceiling = caps.tightest_timeout_ceiling()
        assert host != "codex"
        assert ceiling > 0

    def test_the_tightest_ceiling_is_what_an_unrouted_command_clamps_against(self):
        """Pinned to the literal pair the shipped file resolves to.

        Re-deriving the minimum here would just restate the accessor's own body
        and pass by construction, so the number is written out instead:
        `opencode`'s documented 120s bash default is what an unrouted poll has
        to fit inside today. A deliberate ceiling change in
        `hosts/capabilities.yaml` is expected to update this literal -- and to
        look at `status --wait`'s clamp while doing so.
        """
        assert caps.timeout_ceiling_s("opencode") == 120
        assert caps.tightest_timeout_ceiling() == ("opencode", 120)

    def test_a_tie_on_the_ceiling_is_broken_by_host_id(self, monkeypatch):
        """Two hosts at the same lowest cap must resolve the same way twice.

        The clamp is identical either way, but the *host* named in the message
        is not, and a dict-order answer would flip whenever the file is
        reordered. `claude-code` sorts before `opencode`, so it wins the tie
        even though it is written second here.
        """
        document = {"hosts": {
            "opencode": dict(HOSTS["opencode"], timeout_ceiling_s=120),
            "claude-code": dict(HOSTS["claude-code"], timeout_ceiling_s=120),
            "codex": dict(HOSTS["codex"], timeout_ceiling_s=600),
        }}
        monkeypatch.setattr(caps, "load", lambda base=None: document)
        assert caps.tightest_timeout_ceiling() == ("claude-code", 120)

    def test_a_file_where_nobody_declares_a_cap_clamps_nothing(self, monkeypatch):
        """`None` is "do not clamp", which is the only honest answer when no
        host declares a bound — not a clamp to zero."""
        document = {"hosts": {
            host_id: dict(HOSTS[host_id], timeout_ceiling_s=0)
            for host_id in HOST_IDS
        }}
        monkeypatch.setattr(caps, "load", lambda base=None: document)
        assert caps.tightest_timeout_ceiling() is None

    def test_mutating_the_entry_moves_the_ceiling(self, fixture_tree, monkeypatch):
        root, write = fixture_tree
        document = yaml.safe_load(
            (root / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
        )
        document["hosts"]["claude-code"]["timeout_ceiling_s"] = 45
        write(document)
        monkeypatch.setattr(caps, "load", lambda base=None: document)
        assert caps.timeout_ceiling_s("claude-code") == 45
        assert caps.tightest_timeout_ceiling() == ("claude-code", 45)

    def test_an_unknown_host_is_exit_three_like_every_other_entry_read(self):
        with pytest.raises(caps.UnknownHost):
            caps.timeout_ceiling_s("not-a-host")


# ===========================================================================
# 5. The generated markdown is a build artifact
# ===========================================================================

class TestGeneratedReference:
    def test_the_checked_in_document_is_byte_identical_to_a_fresh_render(self):
        assert RENDERED.read_text(encoding="utf-8") == render_capabilities.render(), (
            "references/host-capabilities.md is generated from "
            "hosts/capabilities.yaml — re-run `python "
            "hosts/render_capabilities.py` and commit the result."
        )

    def test_check_mode_agrees(self, capsys):
        assert render_capabilities.main(["--check"]) == 0
        capsys.readouterr()

    def test_check_mode_fails_on_a_stale_document(self, monkeypatch, capsys):
        monkeypatch.setattr(
            render_capabilities, "render", lambda base=None: "different\n"
        )
        assert render_capabilities.main(["--check"]) == 1
        assert "stale" in capsys.readouterr().err

    def test_a_promoted_route_is_listed_by_its_dotted_name(self):
        """``_verified_cell`` flattens a map row, so a promotion is not dropped."""
        host = json.loads(json.dumps(HOSTS["codex"]))
        host["provenance"]["next_command"] = {"default": "documented",
                                              "package": "verified"}
        cell = render_capabilities._verified_cell(host)
        assert "`next_command.package`" in cell
        assert "`next_command.default`" not in cell
        assert "`next_command`" not in cell

    def test_the_render_is_deterministic(self):
        assert render_capabilities.render() == render_capabilities.render()

    def test_the_render_carries_every_host_and_every_verb(self):
        text = render_capabilities.render()
        for host_id in HOST_IDS:
            assert f"`{host_id}`" in text
            for sentence in caps.verb_sentences(host_id):
                assert sentence in text
        for wording in caps.VERB_WORDING.values():
            assert wording in text

    def test_the_render_shows_which_variables_get_stripped(self):
        text = render_capabilities.render()
        for name in HOSTS["claude-code"]["env_guards"]["guards"]:
            assert f"`{name}`" in text

    def test_the_render_distinguishes_audited_empty_from_unaudited(self):
        text = render_capabilities.render()
        assert "the env diff has not run yet" in text

    def test_the_generated_file_says_it_is_generated(self):
        head = RENDERED.read_text(encoding="utf-8")[:400]
        assert "GENERATED FILE" in head
        assert "render_capabilities.py" in head


# ===========================================================================
# 6. Neither the data nor the docs are on an agent read path
# ===========================================================================

class TestNotOnAnAgentReadPath:
    """D5's resolution rule: the shim's baked sentence *is* the matrix's answer."""

    def test_no_instruction_file_names_the_data_or_the_docs(self):
        offenders = []
        corpus = sorted(SKILL_ROOT.glob("instructions/*.md"))
        corpus += [
            path for path in SKILL_ROOT.glob("references/*.md")
            if path.name != "host-capabilities.md"
        ]
        corpus.append(SKILL_ROOT / "SKILL.md")
        corpus += sorted(SKILL_ROOT.glob("hosts/**/*.md"))
        corpus += sorted(SKILL_ROOT.glob("hosts/**/*.md.tmpl"))
        for path in corpus:
            text = path.read_text(encoding="utf-8")
            for needle in ("capabilities.yaml", "host-capabilities.md"):
                if needle in text:
                    offenders.append(f"{path.relative_to(SKILL_ROOT)}: {needle}")
        assert not offenders, (
            "an agent-read file points at the capability matrix. The whole design "
            "is that every verb is resolved at render time — an agent reasoning "
            "over the matrix mid-run trades one instruction-following failure mode "
            f"for a worse one: {offenders}"
        )


# ===========================================================================
# 7. D5d — the interactive inventory
# ===========================================================================

class TestInteractiveInventory:
    def test_the_inventory_covers_every_d6_row_exactly_once(self):
        covered = [number for entry in un.INVENTORY for number in entry.rows]
        assert sorted(covered) == list(range(2, 16)), covered
        assert len(covered) == len(set(covered))

    def test_row_one_is_absent_by_design(self):
        """uv missing is never a question on any host (D6c)."""
        assert 1 not in [n for e in un.INVENTORY for n in e.rows]

    def test_every_inventory_row_indexes_a_real_site(self):
        for entry in un.INVENTORY:
            row = un.row_for_site(entry.site_id)
            assert entry.rows[0] == row.number, entry.site_id

    def test_the_ask_vocabulary_is_closed_and_matches_the_schema(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert set(un.ASK_CLASSES) == set(
            schema["definitions"]["host"]["properties"]["ask"]["enum"]
        )

    @pytest.mark.parametrize("ask_class", un.ASK_CLASSES)
    @pytest.mark.parametrize("site_id", [e.site_id for e in un.INVENTORY])
    def test_every_row_has_a_documented_presentation_per_class(self, site_id,
                                                               ask_class):
        text = un.presentation(site_id, ask_class)
        assert text and text.strip() == text, (site_id, ask_class)

    @pytest.mark.parametrize("ask_class", (un.ASK_STRUCTURED, un.ASK_FREE_TEXT))
    def test_the_fail_closed_rows_render_on_no_class(self, ask_class):
        for entry in un.INVENTORY:
            if entry.renderable:
                continue
            assert un.presentation(entry.site_id, ask_class) == "never rendered", (
                f"{entry.site_id} is fail-closed in D6's disposition table; a host "
                "that turns it into a question has reintroduced the hang"
            )

    def test_the_fail_closed_rows_are_exactly_thirteen_to_fifteen(self):
        closed = sorted(e.rows[0] for e in un.INVENTORY if not e.renderable)
        assert closed == [13, 14, 15]

    @pytest.mark.parametrize(
        "site_id", [e.site_id for e in un.INVENTORY if e.renderable]
    )
    def test_a_renderable_row_has_a_decision_flag(self, site_id):
        """D5a: no decision may exist only as a question."""
        row = un.row_for_site(site_id)
        assert un.flags_for_row(row.number), site_id

    @pytest.mark.parametrize(
        "site_id",
        [e.site_id for e in un.INVENTORY if e.renderable and e.options],
    )
    def test_the_documented_option_count_is_what_gets_rendered(self, site_id):
        entry = un.inventory_row(site_id)
        options = [
            un.DecisionOption(f"option {i}", f"cmd-{i}", is_default=(i == 1))
            for i in range(1, entry.options + 1)
        ]
        block = un.render_option_block(site_id, options)
        for index in range(1, entry.options + 1):
            assert f"{index}) option {index}" in block
        assert f"{entry.options + 1})" not in block

    def test_one_renderer_two_presentations(self):
        """The option list is built once and is identical for both classes."""
        options = [
            un.DecisionOption("Accept the partial aggregate", "cmd --reaggregate",
                              is_default=True),
            un.DecisionOption("Resume the missing models", "cmd --force"),
            un.DecisionOption("Re-run everything", "cmd --force --rerun-all"),
        ]
        assert un.option_commands(options) == [
            "cmd --reaggregate", "cmd --force", "cmd --force --rerun-all",
        ]
        block = un.render_option_block("partial-results-recovery", options)
        for command in un.option_commands(options):
            assert command in block, "the command is the payload, in both classes"

    def test_the_default_option_is_marked_and_only_that_one(self):
        options = [
            un.DecisionOption("a", "cmd-a", is_default=True),
            un.DecisionOption("b", "cmd-b"),
        ]
        block = un.render_option_block("consolidation-offer", options)
        assert block.count(un.DEFAULT_SUFFIX) == 1
        assert f"1) a{un.DEFAULT_SUFFIX}" in block

    def test_an_option_with_no_command_says_so(self):
        """D5a's "skip = simply do not re-invoke" is the one empty payload."""
        options = [
            un.DecisionOption("Run it", "cmd --consolidate", is_default=True),
            un.DecisionOption("Skip", None),
        ]
        block = un.render_option_block("consolidation-offer", options)
        assert "do not re-invoke" in block

    def test_the_block_forbids_composing_and_paraphrasing(self):
        block = un.render_option_block(
            "consolidation-offer",
            [un.DecisionOption("Run it", "cmd", is_default=True),
             un.DecisionOption("Skip", None)],
        )
        assert "verbatim" in block
        assert "do not compose flags" in block.lower()
        assert "re-renders this block" in block

    def test_rendering_a_fail_closed_row_is_exit_three(self):
        with pytest.raises(un.NeedsHuman) as excinfo:
            un.render_option_block(
                "phase-failure-recovery", [un.DecisionOption("retry", "cmd")]
            )
        assert excinfo.value.exit_code == 3
        assert "fail-closed" in str(excinfo.value)

    def test_rendering_zero_options_is_exit_three(self):
        with pytest.raises(un.NeedsHuman) as excinfo:
            un.render_option_block("consolidation-offer", [])
        assert excinfo.value.exit_code == 3
        assert "row 15" in str(excinfo.value)

    def test_an_unknown_ask_class_is_a_keyerror(self):
        with pytest.raises(KeyError):
            un.presentation("consolidation-offer", "sometimes")

    @pytest.mark.parametrize("host_id", HOST_IDS)
    def test_every_host_ask_value_is_an_inventory_class(self, host_id):
        assert caps.ask_class(host_id) in un.ASK_CLASSES

    def test_free_text_rendering_is_gated_on_interactive(self):
        """Phase 3 ships the renderer and the flags; free-text asks stay opt-in."""
        for entry in un.INVENTORY:
            if not entry.renderable:
                continue
            if entry.site_id == "model-selection":
                assert "--interactive" in entry.unattended
        assert "--interactive" in caps.verb_sentence("codex", "ask")

    def test_row_two_enumerates_sets_rather_than_inviting_free_text(self):
        entry = un.inventory_row("model-selection")
        assert "--models a,b,c" in entry.free_text
        assert "type your models" not in entry.free_text.lower()


# ===========================================================================
# 8. Packaging
# ===========================================================================

class TestPackaging:
    def test_exactly_one_auto_discoverable_skill_manifest_ships(self):
        """`hosts/**` is `.tmpl`-suffixed so it cannot be a second manifest (D5c)."""
        manifests = [
            path.relative_to(SKILL_ROOT).as_posix()
            for path in SKILL_ROOT.rglob("SKILL.md")
            if ".venv" not in path.parts and "tests" not in path.parts
        ]
        assert manifests == ["SKILL.md"], manifests

    def test_the_capability_data_ships_inside_the_skill(self):
        """A shim must be able to name the core and the dispatcher from one artifact."""
        assert CAPS_PATH.is_file()
        assert SCHEMA_PATH.is_file()
        assert os.path.commonpath([str(CAPS_PATH), str(SKILL_ROOT)]) == str(SKILL_ROOT)
