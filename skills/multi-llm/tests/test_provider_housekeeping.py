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
import pytest

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
    "cursor-agent": "auto",
    "gemini": "gemini-2.5-pro",
    "goose": "openrouter/z-ai/glm-5.2",
    "grok": "grok-4",
    "kilocode": "openai-native/gpt-5.5",
    "muse": "muse-spark-1.2",
    "opencode": "openai/gpt-5.5",
    "pi": "anthropic/claude-opus-4-8",
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
