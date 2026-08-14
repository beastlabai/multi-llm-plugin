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

    #: The only keys allowed to be absent. Both govern a lint or generated
    #: documentation and **never** a dispatch decision — that is the whole
    #: admission test, because a defaulted *capability* is the silent guess the
    #: schema exists to remove. Anything that could change what runs must be
    #: required, so its absence is a validation error rather than a guess.
    OPTIONAL_KEYS = {"shim_max_lines", "headless_invocation"}

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

    def test_every_optional_key_is_provenance_tracked(self):
        """Optional does not mean unattributed."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        prov = schema["definitions"]["host"]["properties"]["provenance"]["properties"]
        for key in self.OPTIONAL_KEYS:
            assert key in prov, f"optional key '{key}' has no provenance slot"


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
    RECORDED_SMOKES = {
        "opencode": {
            "shim", "installable", "install_paths", "reload", "arguments",
            "headless_invocation", "detach", "multi_turn",
        },
    }

    def test_no_value_is_verified_without_a_recorded_run(self):
        for host_id, host in HOSTS.items():
            verified = {
                field for field, value in host["provenance"].items()
                if value == "verified"
            }
            if host_id == "claude-code":
                assert verified, "the reference host has observed *something*"
                continue
            allowed = self.RECORDED_SMOKES.get(host_id, set())
            assert verified <= allowed, (
                f"{host_id} claims verified provenance for "
                f"{sorted(verified - allowed)} with no recorded smoke behind it"
            )

    def test_the_smoke_ledger_has_no_stale_rows(self):
        """A ledger entry that nothing claims is a promotion silently reverted."""
        for host_id, fields in self.RECORDED_SMOKES.items():
            assert host_id in HOSTS, f"ledger names unknown host '{host_id}'"
            prov = HOSTS[host_id]["provenance"]
            for field in fields:
                assert prov.get(field) == "verified", (
                    f"{host_id}.{field} is recorded as smoke-verified but the "
                    f"entry says '{prov.get(field)}'"
                )

    def test_an_unexercised_verb_is_never_verified(self):
        """The two verbs the recorded smokes could not reach stay conservative."""
        for host_id in self.RECORDED_SMOKES:
            prov = HOSTS[host_id]["provenance"]
            for verb in ("subagents", "ask"):
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

    def test_unreadable_data_bounds_nothing_rather_than_failing(self, monkeypatch):
        from utils import subagent_executor as executor

        monkeypatch.setattr(caps, "load", lambda base=None: (_ for _ in ()).throw(
            un.NeedsHuman("boom")
        ))
        record = executor.effective_ceiling([])
        assert record["host_max_concurrent_subagents"] is None
        assert record["effective"] >= 1


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
