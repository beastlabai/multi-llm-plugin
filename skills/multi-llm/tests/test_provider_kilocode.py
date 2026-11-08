"""Tests for the KiloCode adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import split_reasoning_effort
from utils.providers.kilocode import KiloCodeProvider


class TestKiloCodeProvider:
    """Tests for the KiloCodeProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a KiloCodeProvider instance."""
        return KiloCodeProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "kilocode"

    def test_default_timeout(self, provider):
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

    # -- Dual binary name (`kilocode` / `kilo`) -----------------------------
    #
    # Upstream ships the CLI under both spellings. This adapter knew only
    # `kilocode`, so on a `kilo`-only machine `is_available()` was False and the
    # provider was invisible to the fan-out — not degraded, absent. The three
    # cases below are the whole contract: canonical present (today's argv,
    # byte-identical), only the alias present (argv[0] moves, nothing else
    # does), and neither present (not available).

    @staticmethod
    def _which_finding(*names):
        """A `shutil.which` stub that resolves exactly `names`."""
        found = set(names)
        return lambda name: f"/usr/local/bin/{name}" if name in found else None

    @patch("shutil.which")
    def test_is_available_true(self, mock_which, provider):
        """Test is_available returns True when kilocode is found."""
        mock_which.return_value = "/usr/local/bin/kilocode"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("kilocode")

    @patch("shutil.which")
    def test_is_available_true_for_the_kilo_spelling(self, mock_which, provider):
        """A `kilo`-only install is available — the defect this pins."""
        mock_which.side_effect = self._which_finding("kilo")

        assert provider.is_available() is True
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_is_available_false(self, mock_which, provider):
        """is_available is False only when NEITHER spelling resolves."""
        mock_which.return_value = None

        assert provider.is_available() is False
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_build_command_uses_the_kilocode_spelling_when_present(self, mock_which, provider):
        """Canonical install: argv is byte-identical to what shipped before."""
        mock_which.side_effect = self._which_finding("kilocode")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "kilocode", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_prefers_kilocode_when_both_resolve(self, mock_which, provider):
        """Both installed ⇒ `kilocode` wins, so argv does not move under the user."""
        mock_which.side_effect = self._which_finding("kilocode", "kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    @patch("shutil.which")
    def test_build_command_moves_only_argv0_for_the_kilo_spelling(self, mock_which, provider):
        """`kilo`-only: argv[0] moves and every other element is identical.

        Asserted against the canonical argv element-by-element rather than as a
        literal, so this cannot drift from the row above.
        """
        mock_which.side_effect = self._which_finding("kilocode")
        canonical = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        mock_which.side_effect = self._which_finding("kilo")
        aliased = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert aliased[0] == "kilo"
        assert aliased[1:] == canonical[1:]

    @patch("shutil.which")
    def test_build_command_variant_branch_also_resolves_the_binary(self, mock_which, provider):
        """The `--variant` branch is the second hardcoded literal §2.3 names."""
        mock_which.side_effect = self._which_finding("kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilo", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2",
            "--variant", "xhigh", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_falls_back_to_the_canonical_name_with_neither_installed(
        self, mock_which, provider
    ):
        """Neither installed ⇒ the canonical spelling, so the argv contract holds.

        `is_available()` is False here, so this argv never reaches a subprocess;
        it is what the argv-contract tests, the housekeeping MODELS table and
        `seed_expected_argv.py` render on a binary-free host.
        """
        mock_which.return_value = None

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    def test_build_command(self, provider):
        """Test command building for Kilo Code CLI."""
        prompt = "Review this code"
        model = "openrouter/moonshotai/kimi-k2.5"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "openrouter/moonshotai/kimi-k2.5", "Review this code"]

    def test_build_command_different_model(self, provider):
        """Test command building with a different model."""
        prompt = "Analyze this"
        model = "kilo/z-ai/glm-5:free"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "kilo/z-ai/glm-5:free", "Analyze this"]

    def test_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the nine valid efforts."""
        from utils.providers.kilocode import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant",
        }

    def test_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --variant pair after -m <base>, prompt last."""
        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/z-ai/glm-5.2",
            "--variant",
            "xhigh",
            "Review this code",
        ]

    def test_build_command_thinking_variant_suffix(self, provider):
        """The kilocode-specific :thinking suffix maps to --variant thinking."""
        cmd = provider.build_command("Review this code", "openrouter/moonshotai/kimi-k2.7-code:thinking")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/moonshotai/kimi-k2.7-code",
            "--variant",
            "thinking",
            "Review this code",
        ]

    def test_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --variant form."""
        efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant")
        for effort in efforts:
            cmd = provider.build_command("Review this code", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                "openrouter/z-ai/glm-5.2",
                "--variant",
                effort,
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --variant flag.

        Critically, openrouter ":free" variants are NOT effort suffixes:
        the whole string stays the model.
        """
        for model in ("openrouter/z-ai/glm-5.2", "openrouter/deepseek/deepseek-r1:free"):
            cmd = provider.build_command("Review this code", model)

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                model,
                "Review this code",
            ], f"model {model!r} was not passed through verbatim"

    def test_parse_output_direct_json_array(self, provider):
        """Parse direct JSON array output."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_direct_json_object(self, provider):
        """Parse direct JSON object output."""
        data = {"status": "ok", "items": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_code_block(self, provider):
        """Extract JSON from markdown code block."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        stdout = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_code_block_without_json_label(self, provider):
        """Extract JSON from code block without json language label."""
        inner_json = {"status": "ok", "items": [1, 2, 3]}
        stdout = f"""Output:
```
{json.dumps(inner_json)}
```
"""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_embedded_json(self, provider):
        """Extract embedded JSON from plain text."""
        inner_json = [{"id": 1}, {"id": 2}]
        stdout = f"The results are: {json.dumps(inner_json)} end of results."

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_empty(self, provider):
        """Handle empty output."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_whitespace_only(self, provider):
        """Handle whitespace-only output."""
        result = provider.parse_output("   \n\t  ", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_no_json(self, provider):
        """Handle output with no extractable JSON."""
        stdout = "Just plain text with no JSON at all."

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_parse_output_whitespace_handling(self, provider):
        """Test that whitespace is handled correctly."""
        data = [1, 2, 3]
        stdout = f"  \n  {json.dumps(data)}  \n  "

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data


class TestKiloCodeDefaultModelSpecs:
    """Every kilocode `:variant` in `defaults.models` must be one the model publishes.

    **This is the only place the mistake can be caught.** `kilocode run
    --variant <name>` SILENTLY IGNORES an unsupported name — no warning, no
    non-zero exit, no marker in the stream — so a fan-out row asking for a
    variant the model does not have runs at the model's default reasoning level
    and reports success. Nothing at run time distinguishes it from a row that
    got what it asked for.

    Four of the six kilocode rows of `defaults.models` were in exactly that
    state (§13.4 #2): `kimi-k3:thinking`, `nemotron-3-ultra-550b-a55b:xhigh`,
    `minimax-m3:xhigh`, `mimo-v2.5-pro:xhigh`.

    The published set is a **captured** fixture, not a live probe: `kilocode
    models <backend> --verbose` is credential-free but it is still a network
    call, and a test that reaches the internet is a test that fails for reasons
    that are not about this repository. `tests/fixtures/kilocode/
    model_variants.json` carries the capture plus its own provenance, and this
    test refuses to pass over a model the capture does not cover — a stale
    fixture must be a failure, not a silent skip, or the guard evaporates the
    first time somebody adds a row.
    """

    VARIANTS_FIXTURE = (Path(__file__).parent / "fixtures" / "kilocode"
                        / "model_variants.json")

    @staticmethod
    def _kilocode_specs():
        """The `kilocode:<model>` rows of the BASE `defaults.models`.

        `load_base_config` and not `load_config`: a developer's own
        `<git-root>/.multi-llm/providers.yaml` deep-merges over base and would
        otherwise decide whether this test passes.
        """
        from utils.provider_registry import load_base_config

        models = (load_base_config().get("defaults") or {}).get("models") or []
        return [spec[len("kilocode:"):] for spec in models
                if isinstance(spec, str) and spec.startswith("kilocode:")]

    def _published(self):
        return json.loads(
            self.VARIANTS_FIXTURE.read_text(encoding="utf-8"))["variants"]

    def test_fixture_covers_every_kilocode_default(self):
        """A spec the capture does not know about fails loudly, naming the refresh."""
        published = self._published()
        specs = self._kilocode_specs()

        assert specs, "defaults.models lists no kilocode rows — did the key move?"
        for spec in specs:
            from utils.providers.kilocode import REASONING_EFFORTS

            base_model, _variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            assert base_model in published, (
                f"no captured `variants` for kilocode model {base_model!r}. "
                f"Refresh {self.VARIANTS_FIXTURE.name} with the commands in its "
                "`_provenance` block (`kilocode models <backend> --verbose`, "
                "credential-free) — an uncovered model is an unchecked variant, "
                "which the CLI would ignore in silence."
            )

    def test_every_default_variant_is_published_by_its_model(self):
        """The L11 #2 assertion: red on four of six before the catalog was corrected."""
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        for spec in self._kilocode_specs():
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is not None and variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not offenders, (
            "defaults.models asks kilocode for a variant the model does not "
            "publish. `kilocode run --variant` ignores an unsupported name "
            "SILENTLY, so these rows run at the model's default effort and "
            "still report success:\n" + "\n".join(offenders)
        )

    @staticmethod
    def _catalog_specs():
        """The `providers.kilocode.models` rows — the *selectable* catalog.

        Separate from `defaults.models` because they fail differently, not
        because the rule differs: a defaults row is silently mis-run on every
        fan-out, while a catalog row is silently mis-run only once a user or
        `--models` picks it. Same silent-ignore mechanism, same guard.
        """
        from utils.provider_registry import load_base_config

        providers = load_base_config().get("providers") or {}
        return list((providers.get("kilocode") or {}).get("models") or [])

    def test_every_catalog_variant_is_published_by_its_model(self):
        """Five catalog rows carried the same defect the defaults did.

        Found while L11 was correcting `defaults.models` and fixed with the
        same rule (the top variant the model actually publishes, preserving the
        row's maximum-reasoning intent): `kimi-k3:thinking` → `:max`,
        `minimax-m3:xhigh` → `:thinking`, `qwen3.7-max:xhigh` → `:max`,
        `hy3:xhigh` → `:high`, `step-3.7-flash:xhigh` → `:high`.
        """
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        uncovered = []
        for spec in self._catalog_specs():
            if not isinstance(spec, str):
                continue
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is None:
                uncovered.append(f"  {base_model}")
                continue
            if variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not uncovered, (
            "providers.kilocode.models carries a `:variant` on a model the "
            f"capture does not cover. Refresh {self.VARIANTS_FIXTURE.name} "
            "with the commands in its `_provenance` block:\n"
            + "\n".join(uncovered)
        )
        assert not offenders, (
            "providers.kilocode.models asks for a variant the model does not "
            "publish; `kilocode run --variant` ignores it SILENTLY:\n"
            + "\n".join(offenders)
        )
