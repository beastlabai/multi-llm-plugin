"""Provider housekeeping: generated exports, dead config, prompt index.

Three hand-maintained invariants that had all silently drifted (§9.1, §9.2,
§9.4):

* **§9.1 — the export lists.** ``utils/providers/__init__.py`` and
  ``utils/__init__.py`` each carried a hand-written copy of the adapter roster.
  Measured 2026-08-16: ``utils/providers`` omitted ``ClaudeCodeProvider`` and
  ``utils`` omitted **three** (``ClaudeCodeProvider``, ``KiloCodeProvider``,
  ``MuseProvider``) — and *deleting* an export left the whole suite green, which
  is the hole this file closes. Both lists are now GENERATED from the one
  adapter table (``provider_registry._PROVIDERS``), and the assertions below are
  set equalities in **both** directions, so an addition without an export and a
  deletion of an export are equally red.

* **§9.2 — dead config fields.** ``supports_json_output`` was declared on all 13
  providers with **0 readers** anywhere in the skill, and ``command:`` likewise
  (every ``is_available()`` is a hardcoded ``shutil.which``). Both are deleted
  from ``providers.yaml`` and from the generated ``--init`` template; the
  security guard that an injected ``command:`` can never move the binary
  (``test_provider_registry.TestCommandNeverLaunchesBinary``) stays.

* **§9.4 — where the prompt sits in argv.** ``build_agent_command_for_transport``
  and every ``cmd[:-1]`` override assume the prompt is the *last* argv element.
  Two adapters violated that: ``grok`` put it at index 4 and ``aider`` embeds it
  inside ``"/ask " + prompt`` (no standalone element at all), which is why an
  assert-only fix was impossible. Each adapter now *declares*
  ``prompt_argv_index``, and this file checks the declaration against the argv
  the adapter really renders.
"""
import re
import sys
from unittest import mock

import pytest

import utils.providers.base as base_module
import utils.providers.cursor_agent as cursor_agent_module
from utils.provider_registry import _PROVIDERS

PROMPT = "Analyze this plan carefully"

# One valid model per adapter. Deliberately a local table: `build_command` does
# not validate models, but several adapters split on "/" or on a trailing
# ":effort", so a shared placeholder would not exercise the real argv shape.
MODELS = {
    "agy": "Gemini 3.1 Pro (High)",
    "aider": "openrouter/z-ai/glm-5.2",
    "claude-code": "sonnet",
    "cline": "openrouter/z-ai/glm-5.2",
    "codex": "gpt-5.5",
    # The real default slug (harness-expansion.md §6.1). It USED to be chosen for
    # being NON-REASONING, because a stream carrying `reasoning_content` deltas
    # breaks copilot's completions parser (copilot-cli#4196); `openai/gpt-5.6-luna`
    # replaced it on 2026-08-19 as newer and cheaper, and it DOES advertise
    # reasoning parameters. That tension is real and is graded on the live lane,
    # not here -- this table only shapes argv.
    "copilot": "openai/gpt-5.6-luna",
    "cursor-agent": "auto",
    "gemini": "gemini-2.5-pro",
    "goose": "openrouter/z-ai/glm-5.2",
    "grok": "grok-4",
    "kilocode": "openai-native/gpt-5.5",
    # The real default slug (harness-expansion.md §6.1) rather than a
    # placeholder: kimi renders NO model token at all, and this row is what
    # keeps `TestDeclaredPromptIndex` checking that against the real argv.
    "kimi": "moonshotai/kimi-k2.5",
    "muse": "muse-spark-1.2",
    "opencode": "openai/gpt-5.5",
    "pi": "anthropic/claude-opus-4-8",
    # The real default slug (harness-expansion.md §6.1), not a placeholder: it is
    # what the §7.4 live smoke ran, and the slash matters — an OpenRouter-shaped
    # id was measured passing through `-m` verbatim.
    "qwen-code": "qwen/qwen3.8-27b",
    # A SEED ALIAS, not a wire slug -- vibe has no `--model` flag, so this value
    # reaches VIBE_ACTIVE_MODEL and never argv. It is still the real default from
    # providers.yaml, because a shared dummy would not exercise the real argv
    # shape and this row is exactly what `TestDeclaredPromptIndex` renders.
    "vibe": "vibe-default",
}


def _registered_class_names():
    """The adapter class names the registry really instantiates."""
    return {type(provider).__name__ for provider in _PROVIDERS.values()}


class TestGeneratedProviderExports:
    """§9.1 — the export lists are derived from the registry, not retyped."""

    def test_model_table_covers_every_registered_adapter(self):
        assert set(MODELS) == set(_PROVIDERS)

    def test_providers_package_all_is_exactly_the_registered_roster(self):
        import utils.providers as providers_pkg

        exported = set(providers_pkg.__all__) - {
            "LLMProvider",
            "PROVIDER_CLASSES",
        }
        assert exported == _registered_class_names()

    def test_utils_all_is_exactly_the_registered_roster(self):
        import utils

        exported = set(utils.__all__) & (
            _registered_class_names() | {"LLMProvider"}
        )
        assert exported - {"LLMProvider"} == _registered_class_names()

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_each_adapter_is_reachable_from_both_packages(self, name):
        import utils
        import utils.providers as providers_pkg

        cls = type(_PROVIDERS[name])
        assert getattr(providers_pkg, cls.__name__) is cls
        assert getattr(utils, cls.__name__) is cls

    def test_every_exported_name_resolves(self):
        """`__all__` may not name anything the module does not actually bind."""
        import utils
        import utils.providers as providers_pkg

        for module in (providers_pkg, utils):
            missing = [n for n in module.__all__ if not hasattr(module, n)]
            assert not missing, f"{module.__name__}.__all__ names {missing}"

    def test_deleting_an_export_cannot_pass(self):
        """The measured hole: a *removed* export used to leave the suite green.

        Simulated by removing one name from the generated roster and asserting
        the same comparison the tests above make now fails.
        """
        import utils.providers as providers_pkg

        truncated = (set(providers_pkg.__all__) - {"LLMProvider", "PROVIDER_CLASSES"}) - {
            "ClaudeCodeProvider"
        }
        assert truncated != _registered_class_names()


class TestDeadConfigFieldsAreGone:
    """§9.2 — 29 occurrences, 0 readers: deleted from base AND the template."""

    DEAD_FIELDS = ("supports_json_output", "command")

    def _base_text(self):
        from pathlib import Path

        import utils.provider_registry as registry

        return (
            Path(registry.__file__).parent.parent / "providers.yaml"
        ).read_text(encoding="utf-8")

    def _template_text(self):
        import init_config

        return init_config.TEMPLATE_PATH.read_text(encoding="utf-8")

    @pytest.mark.parametrize("field", DEAD_FIELDS)
    def test_no_provider_declares_the_dead_field(self, field):
        import yaml

        config = yaml.safe_load(self._base_text())
        offenders = [
            name
            for name, block in config["providers"].items()
            if isinstance(block, dict) and field in block
        ]
        assert not offenders, f"{field} still declared by {offenders}"

    @pytest.mark.parametrize("field", DEAD_FIELDS)
    def test_the_init_template_does_not_reintroduce_it(self, field):
        """Live *or* commented — the template's blocks are authored commented,
        so a `#   command: claude` line would be uncommented by the toggler."""
        offenders = [
            line
            for line in self._template_text().splitlines()
            if line.lstrip("#").strip().startswith(f"{field}:")
        ]
        assert not offenders, offenders

    def test_the_paragraph_no_longer_claims_supports_json_output_works(self):
        from pathlib import Path

        import init_config

        source = Path(init_config.__file__).read_text(encoding="utf-8")
        assert "supports_json_output" not in source


class TestDeclaredPromptIndex:
    """§9.4 — every adapter declares where `build_command` puts the prompt."""

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_every_adapter_declares_a_prompt_index(self, name):
        index = _PROVIDERS[name].prompt_argv_index
        assert index is None or isinstance(index, int)

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_declaration_matches_the_rendered_argv(self, name):
        provider = _PROVIDERS[name]
        argv = provider.build_command(PROMPT, MODELS[name])
        index = provider.prompt_argv_index

        if index is None:
            # No standalone element: the prompt is embedded in exactly one arg.
            assert PROMPT not in argv
            carriers = [arg for arg in argv if PROMPT in arg]
            assert len(carriers) == 1, f"{name}: prompt carried by {carriers}"
        else:
            assert argv[index] == PROMPT, (
                f"{name}: declared prompt_argv_index={index} but the prompt is at "
                f"{argv.index(PROMPT) if PROMPT in argv else None}"
            )

    def test_grok_prompt_is_the_last_argv_element(self):
        """§9.4's normalization: it used to be index 4."""
        provider = _PROVIDERS["grok"]
        argv = provider.build_command(PROMPT, "grok-4")
        assert provider.prompt_argv_index == -1
        assert argv[-1] == PROMPT

    def test_grok_effort_suffix_keeps_the_prompt_last(self):
        provider = _PROVIDERS["grok"]
        argv = provider.build_command(PROMPT, "grok-4:high")
        assert argv[-1] == PROMPT
        assert "--reasoning-effort" in argv

    def test_aider_declares_no_standalone_prompt_element(self):
        """aider embeds the prompt in `"/ask " + prompt` — index is None."""
        provider = _PROVIDERS["aider"]
        assert provider.prompt_argv_index is None
        argv = provider.build_command(PROMPT, MODELS["aider"])
        assert PROMPT not in argv
        assert f"/ask {PROMPT}" in argv

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_last_element_assumption_holds_wherever_it_is_relied_on(self, name):
        """`cmd[:-1]` is only safe for an adapter declaring index -1.

        Every adapter offering a non-argv prompt transport, or an agent profile
        (whose default transport helper drops `argv[-1]`), must declare -1.
        """
        from utils.providers.base import PROFILE_GENERATE

        provider = _PROVIDERS[name]
        relies_on_last = (
            tuple(provider.supported_prompt_transports) != ("argv",)
            or tuple(provider.agent_profiles) != (PROFILE_GENERATE,)
        )
        if relies_on_last:
            assert provider.prompt_argv_index == -1, (
                f"{name} relies on the prompt being argv[-1] but declares "
                f"{provider.prompt_argv_index}"
            )


class TestDocumentedEndpointRecipesStillWork:
    """The documented endpoint recipes (GLM / Z.ai) cannot rot silently.

    §5 of `plans/harness-expansion.md` ships those recipes as **documentation
    only** — there is no `env:` config key and no code path that validates them
    (§5.1: a key the base does not declare merges and is then ignored, so a
    commented example would fail silently and send the run to Anthropic on the
    user's own account). The whole mechanism is that nothing in the plugin
    strips or overwrites the variables the recipes tell users to export.

    That is exactly one assumption, and it is one edit away from being false:
    the day someone adds `ANTHROPIC_BASE_URL` to a strip list, every documented
    recipe stops working with nothing to read. So the guard reads the variable
    names **out of the recipe document itself** (`DOC` below) rather than
    duplicating a list nobody would maintain — a recipe that gains a variable
    gains its assertion for free, and a recipe that loses one loses it.
    """

    from pathlib import Path as _Path

    REPO_ROOT = _Path(__file__).resolve().parents[3]
    #: The recipes moved out of the README when it was trimmed to a front page.
    #: The guard follows them rather than being deleted — that is what the
    #: assertion below says in words.
    DOC = "docs/providers.md"
    HEADING = "## Pointing a provider CLI at a different endpoint"

    #: The two names `DOC` explicitly tells users NOT to build a recipe on,
    #: because `detach._STRIPPED_ENV` deletes them from every detached run. This
    #: guard runs in the opposite direction from the one below: these MUST stay
    #: stripped, or the documented warning is the thing that has gone stale.
    DOCUMENTED_AS_STRIPPED = ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")

    def _subsection(self) -> str:
        import re

        text = (self.REPO_ROOT / self.DOC).read_text(encoding="utf-8")
        start = text.find(self.HEADING)
        assert start != -1, (
            f"{self.DOC} no longer contains {self.HEADING!r}. The endpoint recipes "
            f"are a shipped deliverable; if the heading moved, move this guard "
            f"with it rather than deleting it."
        )
        rest = text[start + len(self.HEADING):]
        # Two or three hashes, never one: a recipe block legitimately contains
        # `# then select a GLM slug ...`, and a one-hash pattern ends the
        # subsection on a shell comment.
        end = re.search(r"^(?:#{2,3} |---$)", rest, re.MULTILINE)
        return rest[: end.start()] if end else rest

    def _exported_names(self) -> "set[str]":
        import re

        blocks = re.findall(r"```bash\n(.*?)```", self._subsection(), re.DOTALL)
        assert blocks, "the endpoint subsection has no ```bash recipe blocks left"
        names = {
            match.group(1)
            for block in blocks
            for match in re.finditer(
                r"^export ([A-Z_][A-Z0-9_]*)=", block, re.MULTILINE
            )
        }
        # A rename that broke the extraction would otherwise pass vacuously.
        assert len(names) >= 4, f"extracted only {sorted(names)} from the recipes"
        return names

    def test_no_documented_variable_is_stripped_from_a_detached_run(self):
        from utils import detach

        offenders = sorted(self._exported_names() & set(detach._STRIPPED_ENV))
        assert not offenders, (
            f"{self.DOC}'s endpoint recipes export {offenders}, which "
            f"detach._STRIPPED_ENV deletes — the recipes would work "
            f"interactively and silently fall back to the vendor default under "
            f"--detach."
        )

    def test_no_documented_variable_is_stripped_by_the_claude_code_adapter(self):
        from utils.providers.claude_code import ClaudeCodeProvider

        removed = set(ClaudeCodeProvider().get_remove_env())
        offenders = sorted(self._exported_names() & removed)
        assert not offenders, (
            f"{self.DOC}'s endpoint recipes export {offenders}, which "
            f"ClaudeCodeProvider.get_remove_env() deletes from the child env — "
            f"the documented recipe cannot reach the CLI."
        )

    @pytest.mark.parametrize("name", DOCUMENTED_AS_STRIPPED)
    def test_the_variables_the_readme_warns_against_are_still_stripped(self, name):
        """The other direction: the warning must stay true, not just unrefuted."""
        from utils import detach

        assert name in detach._STRIPPED_ENV, (
            f"{self.DOC} tells users never to pin models with {name} because a "
            f"detached run strips it. It is no longer stripped, so either the "
            f"warning or this guard is now wrong."
        )


# --------------------------------------------------------------------------- #
# The argv <-> `supports_file_output` guard
# --------------------------------------------------------------------------- #
#
# `TestDeclaredOutputChannel` (test_providers.py) pins WHICH adapters declare
# they cannot write the review result file. It cannot see WHY, so an adapter
# that later gains a read-only mode and forgets to declare it keeps that roster
# green while its users get "JSON parsing failed from both file and stdout" —
# the exact recurrence the cursor `--mode ask` handoff called out as still open.
#
# This class closes that by reading the argv instead of the roster: every
# restriction-shaped atom an adapter really renders must be ADJUDICATED in one of
# the two tables below, and the declaration must follow the adjudication. A new
# flag nobody classified fails with the flag named, so the question "does this
# stop the CLI writing files?" has to be answered when it is introduced rather
# than when a user's fan-out dies on it.

#: Words that select a *restricted* mode. Deliberately restriction-only: a
#: write-ENABLING flag (`--auto`, `--always-approve`, `--dangerously-skip-
#: permissions`) can never cause the failure this guard exists to prevent, and
#: sweeping those in would make the table a list of every flag in the repo.
RESTRICTIVE_ATOM_RE = re.compile(
    r"(?:^|[^a-z])"
    r"(ask|plan|read[-_]?only|no[-_]?write|dry[-_]?run|deny|restrict|sandbox|safe)"
    r"(?:$|[^a-z])",
    re.IGNORECASE,
)

#: A budget flag set to zero is a restriction the word list cannot see — qwen's
#: `--max-tool-calls 0` is a *zero* tool budget, and a model that cannot call a
#: tool cannot write a file. Only zero counts: `--max-turns 25` is a bound, not a
#: prohibition.
ZERO_BUDGET_FLAG_RE = re.compile(r"(tool|call|write|edit|turn|step)", re.IGNORECASE)

#: Atoms that STOP the CLI writing files. Membership here is what forces
#: `supports_file_output = False`; each value is the measurement behind it.
STOPS_FILE_WRITES = {
    ("aider", "--message /ask"): (
        "aider's `/ask` is its read-only chat mode: it answers and edits "
        "nothing, so stdout is the only output path it has. The atom is the "
        "`--message` pair because that is where the mode selector lives — aider "
        "has no standalone prompt element (`prompt_argv_index is None`)."
    ),
    ("cursor-agent", "--mode ask"): (
        "Measured 2026-08-18 on cursor-agent:gpt-5.6-sol-xhigh, three identical "
        "runs: a SUCCESSFUL envelope whose whole result is \"I can't write the "
        "requested file while Ask mode is active.\", and no file. Re-confirmed "
        "against the real binary after the fix: the stdout block returns the "
        "array in 11.7 s, the file block still fails to parse in 11.7 s."
    ),
    ("qwen-code", "--max-tool-calls 0"): (
        "A zero tool budget that aborts before the first call: "
        "FatalBudgetExceededError, exit 55, ZERO bytes on stdout — harsher than "
        "a refusal sentence, because there is nothing left to fall back to. "
        "Captured under plans/harness-expansion/probe-out/qwen-code/captures/"
        "block/, and hit in 4 of 14 live runs."
    ),
    ("copilot", "--no-ask-user"): (
        "The atom that flagged it, though not the mechanism. `--no-ask-user` "
        "removes the `ask_user` TOOL; what actually denies the write is the "
        "permission flag copilot deliberately does NOT ship — no "
        "`--allow-all-tools`, which upstream documents as required for "
        "non-interactive mode. Measured on the live lane 2026-08-19: the model "
        "called `create` with the right path and body, and the CLI answered "
        "`{\"message\": \"Permission denied and could not request permission "
        "from user\", \"code\": \"denied\"}`. Under `-p` there is no channel "
        "on which approval could arrive, so the request and the denial are the "
        "only two things that can happen."
    ),
    ("vibe", "--agent plan"): (
        "Upstream's read-only profile: `write_file` and `edit` are advertised to "
        "the model and then denied — permission `never`, allowlist "
        "$VIBE_HOME/plans/* alone, confirmed in a real run's own session "
        "meta.json. The one writable path is a per-slot temp home this adapter "
        "mints and deletes, never the review's output directory."
    ),
}

#: Atoms that LOOK restrictive and were checked to be compatible with the write.
#: A row here is a claim someone verified, not a flag someone waved through.
REVIEWED_STILL_WRITES = {
    ("codex", "--sandbox workspace-write"): (
        "Writable inside the workspace root, and the review output path is "
        "derived from the plan's own directory inside that repo — measured "
        "`(source: file)` in plans/harness-expansion/code-review/"
        "orchestrator-run.log. An explicit --out-dir outside the repo, or a plan "
        "in no git repo, would land outside it; the stdout fallback covers both."
    ),
}


def _argv_for(name):
    """The argv an adapter renders, host-independently.

    cursor's §5.4.1 gate reads the *installed* binary's --help before it will
    emit `--trust --mode ask`, so the probe is stubbed exactly as
    `TestDeclaredOutputChannel` stubs it — this reads the adapter, not the
    developer's cursor build. Since M14 the gate has one implementation
    (`base.read_help_tokens`), so that is what is stubbed.
    """
    if name == "cursor-agent":
        with mock.patch.object(
            base_module,
            "read_help_tokens",
            lambda _b, **_kw: frozenset({"--trust", "--mode"}),
        ), mock.patch.object(
            cursor_agent_module.shutil, "which", lambda _n: sys.executable
        ):
            return _PROVIDERS[name].build_command(PROMPT, MODELS[name])
    return _PROVIDERS[name].build_command(PROMPT, MODELS[name])


def restrictive_atoms(name):
    """Every restriction-shaped atom in this adapter's rendered argv.

    An *atom* is one argv element, or a `--flag value` pair when the flag takes
    one — because the restriction usually lives in the value (`--mode ask`,
    `--agent plan`) rather than in the flag. The prompt and the model slug are
    stripped out of every element first: both are data this test chose, and a
    prompt that happens to contain the word "plan" is not a mode selector.
    """
    argv = _argv_for(name)
    model = MODELS[name]
    scrubbed = [arg.replace(PROMPT, "").replace(model, "") for arg in argv]

    units, index = [], 0
    while index < len(scrubbed):
        element = scrubbed[index]
        takes_value = (
            element.startswith("-")
            and index + 1 < len(scrubbed)
            and not scrubbed[index + 1].startswith("-")
        )
        if takes_value:
            units.append((element, scrubbed[index + 1]))
            index += 2
        else:
            units.append((element,))
            index += 1

    atoms = []
    for unit in units:
        restrictive = any(RESTRICTIVE_ATOM_RE.search(part) for part in unit)
        zero_budget = (
            len(unit) == 2
            and unit[1].strip() == "0"
            and ZERO_BUDGET_FLAG_RE.search(unit[0])
        )
        if restrictive or zero_budget:
            atoms.append(" ".join(part.strip() for part in unit).strip())
    return atoms


#: Adapters that cannot write the result file for a reason **the argv cannot
#: show**, because nothing in it is restriction-shaped: the CLI simply does not
#: ship the tool. The argv tables above are the right guard for a *chosen* mode
#: (`--mode ask`, `/ask`, `--agent plan`, a zero tool budget); they are blind to
#: a default, and a blind spot that silently reads as "declares True, argv is
#: clean, nothing to see" is exactly how §14.1 shipped.
#:
#: A row here is a MEASUREMENT, not an opinion, and it must name the probe.
NO_WRITE_TOOL_AT_ALL = {
    "gemini": (
        "MEASURED 2026-08-19 (plans/harness-expansion/probe-out/gemini/"
        "gap5_probe.py, two offline runs). Under this adapter's exact argv — no "
        "`--yolo`, no `--approval-mode` — gemini-cli 0.25.2 advertises nine "
        "tools: list_directory, read_file, search_file_content, glob, "
        "activate_skill, save_memory, google_web_search, write_todos, "
        "delegate_to_agent. There is no `write_file`, no `replace` and no "
        "`run_shell_command`. Answering with a write_file functionCall anyway "
        "returns `Error executing tool write_file: Tool \"write_file\" not found "
        "in registry`, stats.tools.byName.write_file = {count:1, success:0, "
        "fail:1}, totalLinesAdded 0, and the file absent — while gemini EXITS 0 "
        "and prints the model's own `JSON written to ...` confirmation as its "
        "answer. That is worse than the four argv-restricted adapters, which "
        "refuse in words: this one reports success for a write that never "
        "happened, so the fan-out would record the confirmation sentence as the "
        "review. Closes harness-expansion.md §14.7 gap 5."
    ),
}


class TestReadOnlyArgvMatchesTheDeclaration:
    """§9.5 — `supports_file_output` is checked against the argv that causes it."""

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_every_restrictive_atom_is_adjudicated(self, name):
        for atom in restrictive_atoms(name):
            key = (name, atom)
            assert key in STOPS_FILE_WRITES or key in REVIEWED_STILL_WRITES, (
                f"{name} renders {atom!r}, which looks like it selects a "
                f"restricted mode, and no one has said whether it stops the CLI "
                f"writing the review result file. Answer that, then add the row "
                f"to STOPS_FILE_WRITES (and set supports_file_output = False) or "
                f"to REVIEWED_STILL_WRITES with the evidence."
            )

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_the_declaration_follows_the_adjudication(self, name):
        blocked = [a for a in restrictive_atoms(name) if (name, a) in STOPS_FILE_WRITES]
        no_tool = name in NO_WRITE_TOOL_AT_ALL
        assert _PROVIDERS[name].supports_file_output == (not (blocked or no_tool)), (
            f"{name} declares supports_file_output="
            f"{_PROVIDERS[name].supports_file_output} but its argv carries "
            f"{blocked or 'no write-stopping atom'}"
            + (" and it is in NO_WRITE_TOOL_AT_ALL" if no_tool else "")
        )

    def test_the_roster_the_argv_produces_is_the_roster_that_is_pinned(self):
        """This file's *why* and test_providers.py's *what* cannot drift apart."""
        from tests.test_providers import TestDeclaredOutputChannel

        derived = {
            name
            for name in _PROVIDERS
            if any((name, atom) in STOPS_FILE_WRITES for atom in restrictive_atoms(name))
        } | set(NO_WRITE_TOOL_AT_ALL)
        assert derived == TestDeclaredOutputChannel.CANNOT_WRITE_FILES

    @pytest.mark.parametrize(
        "key", sorted(set(STOPS_FILE_WRITES) | set(REVIEWED_STILL_WRITES))
    )
    def test_no_adjudication_row_is_stale(self, key):
        """A row whose flag the adapter no longer renders is a row to delete."""
        name, atom = key
        assert name in _PROVIDERS, f"{name} is not a registered adapter"
        assert atom in restrictive_atoms(name), (
            f"{name} no longer renders {atom!r} — the mode it adjudicates is "
            f"gone, so this row is describing a build that no longer ships."
        )
