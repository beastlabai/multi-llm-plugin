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
    # The real default SELECTOR (plans/add-omp-provider.md §1): omp matches the
    # catalog on `selector` (openrouter/openai/gpt-5.6-luna), never on `id`
    # (openai/gpt-5.6-luna), so a placeholder would render an argv shape no real
    # dispatch has. omp owns NO adjudication row and that is deliberate:
    # `RESTRICTIVE_ATOM_RE` matches nothing in its argv — `read,` is not
    # `read-only` and `edit`/`write` are not `no-write` — so
    # `restrictive_atoms("omp")` is empty and any row added below would fail
    # `test_no_adjudication_row_is_stale` on sight. The regex is NOT widened to
    # make one fit: this table is restriction-only, and a write-ENABLING flag
    # cannot cause the failure it guards. The measurement such a row would have
    # carried (under `--approval-mode yolo` with the explicit allow-list,
    # `write` and `edit` really created and edited files) is pinned in
    # `tests/test_providers.py::TestOmpProvider` instead.
    "omp": "openrouter/openai/gpt-5.6-luna",
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
    ("codex", "--sandbox workspace-write"): (
        "Not a read-only mode, and until D6 (2026-09-11) this atom sat in "
        "REVIEWED_STILL_WRITES with `(source: file)` measured in "
        "plans/harness-expansion/code-review/orchestrator-run.log. It moved "
        "because the write goes through bubblewrap, and on a stock Ubuntu 24.04 "
        "(kernel.apparmor_restrict_unprivileged_userns=1) bwrap cannot start: "
        "`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. "
        "Measured twice on codex:gpt-6-astra:xhigh, 2026-09-10 — seven minutes "
        "of review work each time, three `Failed to write file` attempts, and a "
        "final message with no JSON in it, which `--ephemeral` left nothing to "
        "salvage from. A review channel that depends on the machine's sandbox "
        "is not one the fan-out can bet a slot on; the array on stdout is what "
        "`parse_output` reads anyway. `supports_file_edits` stays True — the "
        "agent-write profile really edits, and D6 step 3 gates it with a probe."
    ),
}

#: Atoms that LOOK restrictive and were checked to be compatible with the write.
#: A row here is a claim someone verified, not a flag someone waved through.
REVIEWED_STILL_WRITES = {
    ("muse", "--disable-sandbox"): (
        "The word is `sandbox` but the flag REMOVES one. D6 (2026-09-11) turns "
        "muse's shell sandbox off: it is bubblewrap with --unshare-net, and on a "
        "stock Ubuntu 24.04 (kernel.apparmor_restrict_unprivileged_userns=1) "
        "bwrap cannot start, so every sandboxed write died on `bwrap: loopback: "
        "Failed RTM_NEWADDR` exactly as codex's did. Verified against `muse exec "
        "--help` 1.1.1: 'Disable shell filesystem/network sandboxing for this "
        "run'. Deliberately not `--yolo`, which also grants workspace trust."
    ),
}


#: Atoms an adapter renders ONLY under a dispatch-time override, never in its
#: default argv — so the two tables above (which read the default) cannot see
#: them, and `test_no_adjudication_row_is_stale` would rightly refuse them there.
#: A row here is adjudicated the same way: does the atom stop the CLI writing
#: the result file? The D6 override's answer is "no, the opposite" — and the
#: declaration does not move because of it, because `supports_file_output` is
#: a property of the adapter, not of one dispatch, and the fan-out keeps asking
#: a bypassed codex for stdout (measured 2026-09-17: the array was the final
#: `agent_message`, the `stdout_fallback` shape `parse_output` already reads).
#: `TestDispatchTimeOverrideAtoms` checks each row is really rendered under
#: the override and really absent from the default.
DISPATCH_TIME_OVERRIDES = {
    ("codex", "--sandbox danger-full-access"): (
        "D6 override (2026-09-17). Rendered only by `apply_sandbox_fallback` "
        "for `generate`/`agent-readonly`, behind a probe verdict of "
        "`unavailable` under `providers.codex.sandbox_fallback: auto` (the "
        "default; `off` restores the refusal). "
        "The word is `sandbox` but the value REMOVES one: `codex sandbox -c "
        "'sandbox_mode=\"danger-full-access\"' -- sh -c 'exit 0'` runs with no "
        "bwrap at all (rc 0, codex-cli 0.154.0) where `read-only` and "
        "`workspace-write` both die on `bwrap: loopback: Failed RTM_NEWADDR`. "
        "Under it a file write would SUCCEED, so `STDOUT_OUTPUT_LOCATION` no "
        "longer claims a write \"will be refused\"; the channel stays stdout "
        "and `supports_file_output` stays False."
    ),
}


def _argv_for(name, *, sandbox_fallback=False):
    """The argv an adapter renders, host-independently.

    cursor's §5.4.1 gate reads the *installed* binary's --help before it will
    emit `--trust --mode ask`, so the probe is stubbed exactly as
    `TestDeclaredOutputChannel` stubs it — this reads the adapter, not the
    developer's cursor build. Since M14 the gate has one implementation
    (`base.read_help_tokens`), so that is what is stubbed.

    ``sandbox_fallback=True`` renders through ``build_command_for_transport``
    with the D6 override set — the same stub, so the override class reads the
    adapter and not the machine either.
    """
    def render():
        if sandbox_fallback:
            argv, _ = _PROVIDERS[name].build_command_for_transport(
                PROMPT, MODELS[name], "argv", sandbox_fallback=True
            )
            return argv
        return _PROVIDERS[name].build_command(PROMPT, MODELS[name])

    if name == "cursor-agent":
        with mock.patch.object(
            base_module,
            "read_help_tokens",
            lambda _b, **_kw: frozenset({"--trust", "--mode"}),
        ), mock.patch.object(
            cursor_agent_module.shutil, "which", lambda _n: sys.executable
        ):
            return render()
    return render()


def restrictive_atoms(name, argv=None):
    """Every restriction-shaped atom in this adapter's rendered argv.

    An *atom* is one argv element, or a `--flag value` pair when the flag takes
    one — because the restriction usually lives in the value (`--mode ask`,
    `--agent plan`) rather than in the flag. The prompt and the model slug are
    stripped out of every element first: both are data this test chose, and a
    prompt that happens to contain the word "plan" is not a mode selector.
    ``argv`` defaults to the adapter's rendered default; the override class
    below hands in the argv a dispatch-time rewrite produced.
    """
    if argv is None:
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
        "review. Closes harness-expansion.md §14.7 gap 5. RE-MEASURED 2026-10-08 "
        "on gemini-cli 0.63.0 by the same probe: eight tools (enter_plan_mode, "
        "glob, google_web_search, grep_search, invoke_agent, list_directory, "
        "read_file, update_topic), still no write tool, the same exit-0 "
        "confirmation; the error now reads `Tool \"write_file\" not found. Did "
        "you mean one of: ...`."
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


# ---------------------------------------------------------------------------
# `wants_inline_plan` against the route that replaces the inline copy
# (plans/ask-answer-recovery-and-false-success.md D4.2)
# ---------------------------------------------------------------------------
#
# An adapter that declares `False` gets an ask prompt that names the plan's path
# and says its content is attached above. That sentence is true only if the
# adapter's own argv attaches the file; otherwise the model answers a question
# about a plan it was never shown.

PLAN_BODY = "PLAN BODY SENTINEL"


def _path_only_ask_prompt(name, plan_path):
    """The ask prompt a path-only adapter is meant to get: the path block and no
    size note. Built from the template directly, so the route is checked even
    where `render_plan_fields` would fall back to the inline block."""
    from ask_orchestrator import PATH_PLAN_BLOCK, render_output_fields
    from utils.prompt_loader import load_prompt

    return load_prompt("ask.txt").format(
        question=PROMPT,
        plan_block=PATH_PLAN_BLOCK.format(plan_path=plan_path),
        plan_size_note="",
        **render_output_fields(name, "/out/answer.md"),
    )


def assert_path_only_plan_has_a_route(name, provider, model, plan_path):
    argv = provider.build_command(_path_only_ask_prompt(name, plan_path), model)
    assert ("--read", plan_path) in list(zip(argv, argv[1:])), (
        f"{name} declares wants_inline_plan = False, but the argv it renders "
        f"for its path-only ask prompt carries no `--read {plan_path}`, so the "
        f"ask lane would send its model no plan at all. Give it a route that "
        f"attaches the plan, or declare wants_inline_plan = True."
    )


class TestPathOnlyAskPromptHasARoute:
    """D4.2 — `wants_inline_plan = False` is checked against the argv that
    replaces the inline copy."""

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_every_adapter_gets_the_plan_by_one_route(self, name, tmp_path):
        from ask_orchestrator import render_prompt
        from utils.prompt_loader import load_prompt

        plan = tmp_path / "plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")
        plan_path = str(plan.resolve())
        assert re.fullmatch(r"/[\w.\-/]+", plan_path), (
            f"precondition: {plan_path} must be a path aider's extractor matches whole"
        )
        rendered = render_prompt(
            load_prompt("ask.txt"), PROMPT, plan_path, PLAN_BODY,
            "/out/answer.md", name,
        )

        if _PROVIDERS[name].wants_inline_plan:
            assert PLAN_BODY in rendered
            return
        # The path-only prompt is what the renderer really sends, not a
        # fallback it happens to take on this path ...
        assert rendered == _path_only_ask_prompt(name, plan_path)
        assert PLAN_BODY not in rendered
        # ... and the argv built from it attaches the plan.
        assert_path_only_plan_has_a_route(
            name, _PROVIDERS[name], MODELS[name], plan_path
        )

    def test_a_path_only_declaration_with_no_route_fails_by_name(self, tmp_path):
        from utils.providers.claude_code import ClaudeCodeProvider

        class NoRouteAdapter(ClaudeCodeProvider):
            wants_inline_plan = False

        plan = tmp_path / "plan.md"
        plan.write_text(PLAN_BODY, encoding="utf-8")

        with pytest.raises(AssertionError, match="no-route-adapter"):
            assert_path_only_plan_has_a_route(
                "no-route-adapter", NoRouteAdapter(), MODELS["claude-code"],
                str(plan.resolve()),
            )


# ---------------------------------------------------------------------------
# `runs_shell_in_generate` against its evidence (plan: review-run-scope-
# salvage-and-failure-visibility D2.3)
# ---------------------------------------------------------------------------
#
# The code-review prompt's `{diff_access}` block grants `git diff` to a lane
# declaring `True`, tells a `False` lane it has no shell, and asserts nothing
# for `None`. A declaration nobody can trace to a source is therefore a prompt
# that lies to a model about its tools — the 2026-09-25 run's cursor lanes were
# told to run `git diff` and had no shell. So every non-`None` value needs a row
# here, and the row is either something this file can read out of the argv, or
# a measurement named well enough to be re-run.

from typing import NamedTuple, Optional  # noqa: E402


class ShellEvidence(NamedTuple):
    """One adapter's evidence for its `runs_shell_in_generate` value."""

    #: The value the evidence supports; the declaration must equal it.
    shell: bool
    #: Where the fact comes from: the measurement, or why the argv shows it.
    source: str
    #: An ARGV-TOKEN row: the flag whose comma-split value the adapter renders…
    flag: Optional[str] = None
    #: …and the shell-granting tool name that value must carry. A row with no
    #: token is a NAMED-MEASUREMENT row.
    token: Optional[str] = None


#: Hand-kept, one row per adapter that declares `True` or `False`; an adapter
#: declaring `None` has no row. A `True` value is backed by a shell-granting
#: token in its `generate` argv or by a named measurement; a `False` value only
#: by a named measurement, AND its argv must grant no shell tool — so a later
#: tool-list edit that grants one turns this red.
SHELL_EVIDENCE = {
    "pi": ShellEvidence(
        True,
        "`bash` in the `--tools` allow-list (PI_TOOLS, D3.1); pi has no "
        "permission prompt. Measured too: the 2026-09-25 review-code slot's "
        "stream ended inside a `bash` call running `timeout 1700 python -m "
        "pytest tests …` (plan §3.1).",
        flag="--tools",
        token="bash",
    ),
    "omp": ShellEvidence(
        True,
        "`bash` in the `--tools` allow-list (OMP_TOOLS) under `--approval-mode "
        "yolo`, rendered on every `generate` dispatch because every narrower "
        "approval mode denies `bash` headless.",
        flag="--tools",
        token="bash",
    ),
    "codex": ShellEvidence(
        True,
        "MEASURED: an audited review stream of the 2026-09-25 run ran the "
        "`command_execution` `pwd && rg --files …` under `generate`'s "
        "`--sandbox workspace-write` (plan §7.3; codex_stream_audit's "
        "ANY_FLAG_VERBS). The argv names a sandbox policy, not a tool list, so "
        "no argv token can show it.",
    ),
    "cursor-agent": ShellEvidence(
        False,
        "MEASURED 2026-09-25 in the review-code logs of both cursor lanes under "
        "`--mode ask`: kimi — \"Shell access is refused in this environment\"; "
        "grok — \"Shell diffs are blocked in this mode\" (plan §2.1).",
    ),
    "claude-code": ShellEvidence(
        False,
        "MEASURED 2026-09-25: the review-code envelope carries three `Bash` "
        "`permission_denials`; `--permission-mode acceptEdits` auto-accepts file "
        "edits only and headless `claude -p` denies Bash (plan §2.1).",
    ),
    "qwen-code": ShellEvidence(
        False,
        "MEASURED: `--max-tool-calls 0` aborts before the first tool call runs "
        "(FatalBudgetExceededError, exit 55, zero bytes on stdout; "
        "plans/harness-expansion/probe-out/qwen-code/captures/block/), and a "
        "shell is a tool call.",
    ),
    "gemini": ShellEvidence(
        False,
        "MEASURED 2026-08-19 (plans/harness-expansion/probe-out/gemini/"
        "gap5_probe.py): under this argv gemini-cli 0.25.2 advertises nine tools "
        "and `run_shell_command` is not among them (NO_WRITE_TOOL_AT_ALL above); "
        "re-measured 2026-10-08 on 0.63.0: eight tools, still without it.",
    ),
}

#: Tool names that hand a model a shell when an argv grants them: `bash`
#: (pi/omp tool lists), `Bash` (claude-code's tool), `run_shell_command`
#: (gemini-cli's). A `Bash(` prefix is claude-code's pattern grant
#: (`Bash(git diff:*)`), which grants a shell as surely as the bare name.
SHELL_TOOL_NAMES = frozenset({"bash", "Bash", "run_shell_command"})


def _generate_argv(name):
    """The adapter's `generate` argv, host-independently.

    `_argv_for`'s cursor stub, widened to the two other adapters whose argv
    depends on the INSTALLED binary's help: omp refuses to render unless every
    flag it needs is documented, and pi renders `--tools` only when its help
    documents it. Each gate is answered "documented", so this reads the
    adapter's code, never the developer's installed build — and pi's row sees
    the flag whether or not a pi binary is on this machine.
    """
    from utils.providers.omp import OMP_REQUIRED_FLAGS
    from utils.providers.pi import PI_TOOLS_FLAG

    gate_answers = {
        "cursor-agent": frozenset({"--trust", "--mode"}),
        "omp": frozenset(OMP_REQUIRED_FLAGS),
        "pi": frozenset({PI_TOOLS_FLAG}),
    }
    answer = gate_answers.get(name)
    if answer is None:
        return _argv_for(name)
    with mock.patch.object(
        base_module, "read_help_tokens", lambda _b, **_kw: answer
    ), mock.patch.object(base_module.shutil, "which", lambda _n: sys.executable):
        return _PROVIDERS[name].build_command(PROMPT, MODELS[name])


def _shell_grants(argv):
    """Every shell-granting tool name in ``argv``, split on commas.

    Each element but the prompt is split on commas (the `--tools a,b,c` form)
    and on whitespace (a space-separated `--allowed-tools` list); the prompt
    is data this test chose.
    """
    grants = set()
    for element in argv:
        if element == PROMPT:
            continue
        for piece in re.split(r"[,\s]+", element):
            if piece in SHELL_TOOL_NAMES or piece.startswith(("Bash(", "bash(")):
                grants.add(piece)
    return sorted(grants)


def _comma_split_value(argv, flag):
    """The comma-split value ``flag`` carries in ``argv``, or ``[]``."""
    if flag not in argv or argv.index(flag) + 1 >= len(argv):
        return []
    return argv[argv.index(flag) + 1].split(",")


TOKEN_ROWS = sorted(name for name, row in SHELL_EVIDENCE.items() if row.token)
MEASUREMENT_ROWS = sorted(name for name, row in SHELL_EVIDENCE.items() if not row.token)


class TestShellDeclarationMatchesTheEvidence:
    """`runs_shell_in_generate` is checked against its argv or its measurement."""

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_a_declared_value_has_exactly_its_evidence_row(self, name):
        declared = _PROVIDERS[name].runs_shell_in_generate
        row = SHELL_EVIDENCE.get(name)
        if declared is None:
            assert row is None, (
                f"{name} declares runs_shell_in_generate=None (not measured) but "
                f"SHELL_EVIDENCE carries a row for it — set the value the row "
                f"supports, or delete the row."
            )
            return
        assert row is not None, (
            f"{name} declares runs_shell_in_generate={declared} with no "
            f"SHELL_EVIDENCE row. A value leaves None only with a measured "
            f"source — a log, a probe or its argv — so add the row naming it."
        )
        assert row.shell is declared, (
            f"{name} declares runs_shell_in_generate={declared} but its evidence "
            f"supports {row.shell}: {row.source}"
        )

    def test_every_row_names_a_registered_adapter(self):
        assert set(SHELL_EVIDENCE) <= set(_PROVIDERS)

    @pytest.mark.parametrize("name", TOKEN_ROWS)
    def test_an_argv_token_row_is_still_rendered(self, name):
        """A row whose token the adapter no longer renders is stale, as in
        `test_no_adjudication_row_is_stale`: the shell it vouches for is gone."""
        row = SHELL_EVIDENCE[name]
        argv = _generate_argv(name)

        assert row.shell is True, f"{name}: an argv token can only show a shell"
        assert row.token in _comma_split_value(argv, row.flag), (
            f"{name} no longer renders {row.token!r} in its {row.flag} value "
            f"({_comma_split_value(argv, row.flag)}), so this row describes a "
            f"build that no longer ships — re-measure the shell and move "
            f"runs_shell_in_generate with it."
        )
        # And the detector the False rows rely on sees it: a positive control,
        # so `_shell_grants` cannot pass the rows below by seeing nothing.
        assert row.token in _shell_grants(argv)

    @pytest.mark.parametrize("name", MEASUREMENT_ROWS)
    def test_a_measurement_row_names_its_measurement(self, name):
        row = SHELL_EVIDENCE[name]

        assert row.flag is None and row.token is None
        assert "MEASURED" in row.source, (
            f"{name}: a named-measurement row must say what was measured"
        )

    @pytest.mark.parametrize(
        "name", sorted(n for n, row in SHELL_EVIDENCE.items() if row.shell is False)
    )
    def test_a_false_declaration_grants_no_shell_tool(self, name):
        """`False` is a measurement of THIS argv; an edit that hands the model a
        shell tool makes it false, and the prompt would then deny a shell the
        lane has."""
        row = SHELL_EVIDENCE[name]
        argv = _generate_argv(name)

        assert row.token is None, (
            f"{name}: a False value is backed by a measurement, never by an "
            f"argv token"
        )
        assert _shell_grants(argv) == [], (
            f"{name} declares runs_shell_in_generate=False but its generate argv "
            f"grants {_shell_grants(argv)}: {argv}"
        )


class TestDispatchTimeOverrideAtoms:
    """`DISPATCH_TIME_OVERRIDES` — adjudicated from both sides (D6, 2026-09-17)."""

    @staticmethod
    def _override_argv(name):
        return _argv_for(name, sandbox_fallback=True)

    @pytest.mark.parametrize("key", sorted(DISPATCH_TIME_OVERRIDES))
    def test_the_row_is_rendered_under_the_override_and_absent_by_default(self, key):
        name, atom = key
        assert name in _PROVIDERS, f"{name} is not a registered adapter"
        assert atom in restrictive_atoms(name, self._override_argv(name)), (
            f"{name} no longer renders {atom!r} under `sandbox_fallback=True` — "
            f"the override this row adjudicates is gone, so delete the row."
        )
        assert atom not in restrictive_atoms(name), (
            f"{name} renders {atom!r} in its DEFAULT argv. A dispatch-time "
            f"override that leaked into a default is a bypass on every run; "
            f"`tests/test_read_only_modes.py` should already be red on this."
        )

    @pytest.mark.parametrize("name", sorted(_PROVIDERS))
    def test_every_override_atom_an_adapter_renders_is_adjudicated(self, name):
        """A new override nobody classified fails by name, like a new flag."""
        default = set(restrictive_atoms(name))
        for atom in restrictive_atoms(name, self._override_argv(name)):
            if atom in default:
                continue
            assert (name, atom) in DISPATCH_TIME_OVERRIDES, (
                f"{name} renders {atom!r} only under `sandbox_fallback=True`, and "
                f"no one has said what it does to the CLI's writes. Add the row to "
                f"DISPATCH_TIME_OVERRIDES with the measurement."
            )

    @pytest.mark.parametrize("key", sorted(DISPATCH_TIME_OVERRIDES))
    def test_the_override_moves_no_declaration(self, key):
        """`supports_file_output` is per adapter; the fan-out keeps asking a
        bypassed slot for stdout, and the row says why."""
        name, _atom = key
        assert _PROVIDERS[name].supports_file_output is False
        assert "stdout" in DISPATCH_TIME_OVERRIDES[key]
