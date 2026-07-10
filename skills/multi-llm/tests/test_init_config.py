"""Tests for the auto-detect `--init` config scaffolder (init_config.py).

`--init` is fully automatic and zero-prompt: it detects installed provider CLIs
and UNCOMMENTS their lines in an inert template. This module covers:

- The toggler (`auto_uncomment` / `_toggle_template`): per-provider sub-block
  uncommenting, `/`-bearing specs, sub-block boundaries, the `default_provider`
  rewrite (first detected in declaration order), and "nothing detected" inertness.
- The D2a empty-panel guard (inject + notice) for gemini/opencode-only machines.
- End-to-end init: detection → write → merged resolution through load_config.
- Re-init `--force` after a detection change (outside-marker preservation).
- The in-marker `modes:` example safety (never uncommented into orphan items).
- `--template-only` byte-identity and the inert nothing-detected write.
- Removal regression: the old interactive flow / flags are gone.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import init_config
import utils.provider_registry as registry


def _reset_cache():
    registry._config = None
    registry._config_key = None


# ---------------------------------------------------------------------------
# A small synthetic fixture template body, built via the same emitters the real
# template uses, so the toggler is tested against the exact authored format
# (indentation, `# ` prefixes) but independently of the shipped base config.
# Providers: alpha (curated), beta (a `/`-bearing model id), gamma (uncurated).
# ---------------------------------------------------------------------------

_FIXTURE_PROVIDERS = {
    "alpha": {"command": "alpha", "max_concurrent": 2, "models": ["a1", "a2"]},
    "beta": {"command": "beta", "models": ["openrouter/b/b1"]},
    "gamma": {"command": "gamma", "models": ["g1"]},
}
_FIXTURE_ORDER = list(_FIXTURE_PROVIDERS)
_FIXTURE_DEFAULT_MODELS = ["alpha:a1", "beta:openrouter/b/b1"]  # gamma intentionally absent
_FIXTURE_QUICK_MODELS = ["alpha:a1"]


def _fixture_body() -> str:
    lines = ["## intro prose", "## --- providers ---"]
    lines += init_config._emit_providers_block(_FIXTURE_PROVIDERS)
    lines += ["", "## --- default_provider ---", "# default_provider: beta", "", ""]
    lines += ["## --- defaults ---", "defaults:", ""]
    lines += ["  ## --- defaults.models ---", "  models:"]
    lines += init_config._emit_defaults_items(_FIXTURE_DEFAULT_MODELS)
    lines += ["", "  ## --- defaults.quick_models ---", "  quick_models:"]
    lines += init_config._emit_defaults_items(_FIXTURE_QUICK_MODELS)
    lines += [
        "",
        "  ## --- defaults.modes ---",
        "  ## modes:",
        "  ##   review-plan:",
        "  ##     models:",
        "  ##       - alpha:a1",
        "  ##       - beta:openrouter/b/b1",
    ]
    return "\n".join(lines)


def _toggle(detected):
    body = _fixture_body()
    return init_config._toggle_template(body, set(detected), _FIXTURE_ORDER)


def _parse(detected):
    text, _m, _q = _toggle(detected)
    return yaml.safe_load(text), text.splitlines()


class TestTogglerNothingDetected:
    def test_inert_when_nothing_detected(self):
        body = _fixture_body()
        out = init_config.auto_uncomment(body, set(), _FIXTURE_ORDER)
        # A toggle over an empty detection set is the identity (pristine = inert).
        assert out == body
        parsed = yaml.safe_load(out)
        assert set(parsed.keys()) == {"defaults"}
        assert parsed["defaults"] == {"models": None, "quick_models": None}


class TestTogglerSingleProvider:
    def test_single_detected_uncomments_only_its_block(self):
        parsed, lines = _parse(["alpha"])
        # alpha's full sub-block is live; beta/gamma stay commented.
        assert parsed["providers"]["alpha"] == {
            "command": "alpha",
            "max_concurrent": 2,
            "models": ["a1", "a2"],
        }
        assert "beta" not in parsed["providers"]
        assert "gamma" not in parsed["providers"]
        # default_provider rewritten to the (only) detected provider.
        assert parsed["default_provider"] == "alpha"
        # defaults.models keeps only alpha's spec; gamma is uncurated, beta undetected.
        assert parsed["defaults"]["models"] == ["alpha:a1"]
        # Specific line-level checks: alpha key uncommented, beta key still commented.
        assert "  alpha:" in lines
        assert "#   beta:" in lines

    def test_slash_bearing_model_uncomments_correctly(self):
        parsed, _lines = _parse(["beta"])
        # The `/`-bearing bare id round-trips through the providers block...
        assert parsed["providers"]["beta"]["models"] == ["openrouter/b/b1"]
        # ...and through the defaults spec (split on the FIRST colon only).
        assert parsed["defaults"]["models"] == ["beta:openrouter/b/b1"]
        assert parsed["default_provider"] == "beta"


class TestTogglerMultiProvider:
    def test_undetected_sandwiched_between_detected(self):
        # alpha + gamma detected; beta (declared between them) stays commented.
        parsed, _lines = _parse(["alpha", "gamma"])
        assert set(parsed["providers"]) == {"alpha", "gamma"}
        assert "beta" not in parsed["providers"]
        # default_provider is the FIRST detected in declaration order (alpha).
        assert parsed["default_provider"] == "alpha"
        # gamma is uncurated in defaults.models → only alpha:a1 survives there.
        assert parsed["defaults"]["models"] == ["alpha:a1"]

    def test_all_detected_uncomments_everything(self):
        parsed, _lines = _parse(["alpha", "beta", "gamma"])
        assert set(parsed["providers"]) == {"alpha", "beta", "gamma"}
        assert parsed["default_provider"] == "alpha"
        assert parsed["defaults"]["models"] == ["alpha:a1", "beta:openrouter/b/b1"]
        assert parsed["defaults"]["quick_models"] == ["alpha:a1"]


class TestTogglerStateMachine:
    def test_default_provider_rewritten_not_trapped_in_providers(self):
        # Regression: default_provider must be REWRITTEN (not left commented inside
        # the providers sub-block) whenever ≥1 provider is detected.
        _parsed, lines = _parse(["alpha"])
        assert "default_provider: alpha" in lines
        assert "# default_provider: beta" not in lines

    def test_default_provider_left_commented_when_none_detected(self):
        _parsed, lines = _parse([])
        assert "# default_provider: beta" in lines
        assert "default_provider: beta" not in lines

    def test_double_hash_prose_and_blanks_pass_through_verbatim(self):
        # `## ` prose and blank separators are never `# `-stripped, even inside a
        # detected provider's region.
        _parsed, lines = _parse(["alpha", "beta", "gamma"])
        assert "## intro prose" in lines
        assert "## --- providers ---" in lines
        assert "" in lines  # blank separators survive

    def test_detected_last_provider_does_not_bleed_into_default_provider(self):
        # gamma is the last provider; with it detected the sub-block ends cleanly at
        # the default_provider top-level key (gamma's toggle state does not bleed).
        parsed, lines = _parse(["gamma"])
        assert parsed["providers"]["gamma"] == {"command": "gamma", "models": ["g1"]}
        assert parsed["default_provider"] == "gamma"
        assert "default_provider: gamma" in lines

    def test_modes_example_never_uncommented(self):
        # The `## ` modes example lines are inert regardless of detection.
        parsed, lines = _parse(["alpha", "beta", "gamma"])
        assert "  ## modes:" in lines
        assert "  ##   review-plan:" in lines
        # defaults.modes is NOT materialized (init writes no per-mode panel).
        assert "modes" not in parsed["defaults"]

    def test_toggle_returns_uncommented_spec_lists(self):
        # _toggle_template reports the specs it uncommented (used by D2a + recheck).
        _text, models_specs, quick_specs = _toggle(["alpha", "beta"])
        assert models_specs == ["alpha:a1", "beta:openrouter/b/b1"]
        assert quick_specs == ["alpha:a1"]
        _text, models_specs, quick_specs = _toggle(["gamma"])
        assert models_specs == []  # gamma uncurated → empty (D2a territory)
        assert quick_specs == []


# ---------------------------------------------------------------------------
# D2a guard — exercised against the REAL template + base (gemini/opencode are the
# off-panel providers in the shipped base config).
# ---------------------------------------------------------------------------


def _real_init(detected_list):
    """Run the real template toggler + D2a guard for ``detected_list``.

    Returns (parsed_config_dict, expected_models, notices).
    """
    base = init_config.load_base_config()
    order = list(base["providers"])
    body = init_config._extract_managed_body(init_config.TEMPLATE_PATH.read_text())
    toggled, m, q = init_config._toggle_template(body, set(detected_list), order)
    toggled, expected, notices = init_config._apply_d2a_guard(
        toggled, detected_list, m, q, base
    )
    text = init_config.splice_managed_block(init_config.TEMPLATE_PATH.read_text(), toggled)
    return yaml.safe_load(text), expected, notices


class TestD2aGuard:
    def test_gemini_only_injects_first_model(self):
        parsed, expected, notices = _real_init(["gemini"])
        assert parsed["defaults"]["models"] == ["gemini:gemini-3-flash"]
        assert parsed["defaults"]["quick_models"] == ["gemini:gemini-3-flash"]
        assert expected == ["gemini:gemini-3-flash"]
        assert any("gemini" in n and "defaults.models" in n for n in notices)

    def test_opencode_only_injects_first_model(self):
        parsed, _expected, notices = _real_init(["opencode"])
        assert parsed["defaults"]["models"] == ["opencode:opencode/big-pickle"]
        assert parsed["defaults"]["quick_models"] == ["opencode:opencode/big-pickle"]
        assert any("opencode" in n for n in notices)

    def test_multi_offpanel_injects_both_and_names_both(self):
        parsed, _expected, notices = _real_init(["gemini", "opencode"])
        assert parsed["defaults"]["models"] == [
            "gemini:gemini-3-flash",
            "opencode:opencode/big-pickle",
        ]
        # quick_models gets only the FIRST detected provider's first model.
        assert parsed["defaults"]["quick_models"] == ["gemini:gemini-3-flash"]
        models_notice = next(n for n in notices if "defaults.models" in n)
        assert "gemini" in models_notice and "opencode" in models_notice

    def test_covered_provider_present_no_injection(self):
        # {gemini, claude-code}: claude-code:fable:high is already in the panel, so
        # the guard does NOT fire and gemini is simply absent from the panel.
        parsed, _expected, notices = _real_init(["gemini", "claude-code"])
        assert parsed["defaults"]["models"] == ["claude-code:fable:high"]
        assert notices == []

    def test_notice_text_single_vs_multi(self):
        _p1, _e1, n1 = _real_init(["gemini"])
        _p2, _e2, n2 = _real_init(["gemini", "opencode"])
        single = next(n for n in n1 if "defaults.models" in n)
        multi = next(n for n in n2 if "defaults.models" in n)
        assert "{gemini}" in single
        assert "{gemini, opencode}" in multi

    def test_injection_does_not_corrupt_provider_catalogs(self):
        # Regression: D2a inject must scope to defaults.models / quick_models and
        # NEVER splice the prefixed spec into a detected provider's OWN models
        # catalog (whose `models:` key is uncommented at a deeper indent). The
        # provider blocks must stay BARE model ids after D2a.
        opencode_catalog = [
            "opencode/big-pickle",
            "opencode/sonnet",
            "opencode/deepseek-v4-flash-free",
            "opencode/hy3-free",
            "opencode/nemotron-3-ultra-free",
            "google/gemini-3.1-pro-preview",
            "openai/gpt-5.5",
        ]
        for detected, catalogs in (
            (["opencode"], {"opencode": opencode_catalog}),
            (
                ["gemini"],
                {
                    "gemini": [
                        "gemini-3-flash",
                        "gemini-3-pro",
                        "gemini-3.1-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ]
                },
            ),
            (
                ["gemini", "opencode"],
                {
                    "gemini": [
                        "gemini-3-flash",
                        "gemini-3-pro",
                        "gemini-3.1-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ],
                    "opencode": opencode_catalog,
                },
            ),
        ):
            parsed, _expected, _notices = _real_init(detected)
            for name, expected_models in catalogs.items():
                got = parsed["providers"][name]["models"]
                assert got == expected_models, (detected, name, got)
                # No catalog entry carries a `provider:` prefix (would be a bare-id
                # corruption from the injection bleeding into the provider block).
                assert all(":" not in m for m in got), (detected, name, got)


# ---------------------------------------------------------------------------
# End-to-end init (real main(), monkeypatched detection).
# ---------------------------------------------------------------------------


def _git_init(path: Path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


@pytest.mark.config_override
class TestEndToEndInit:
    def test_writes_detected_specs_and_merges_full_blocks(self, tmp_path, monkeypatch):
        _git_init(tmp_path)
        monkeypatch.setattr(
            init_config, "_available_providers", lambda cfg: ["claude-code", "cursor-agent"]
        )
        _reset_cache()
        rc = init_config.main(["--dir", str(tmp_path)])
        assert rc == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert out.exists()

        written = yaml.safe_load(out.read_text())
        assert written["default_provider"] == "claude-code"
        # Detected providers' full blocks are present and live in the written file.
        assert "claude-code" in written["providers"]
        assert "cursor-agent" in written["providers"]
        assert written["providers"]["claude-code"]["command"] == "claude"
        # Only detected providers' defaults specs are uncommented.
        assert written["defaults"]["models"] == [
            "cursor-agent:composer-2.5",
            "cursor-agent:gemini-3.1-pro",
            "cursor-agent:grok-4.5-xhigh",
            "claude-code:fable:high",
        ]

        # The file resolves cleanly through load_config (fresh cache), with the
        # detected providers' full blocks merged over base and NO providers-block
        # warning (filter removed).
        _reset_cache()
        merged = registry.load_config(anchor=str(tmp_path))
        assert merged["defaults"]["models"] == written["defaults"]["models"]
        assert merged["providers"]["claude-code"]["command"] == "claude"
        # base providers the override omitted are still inherited.
        assert "gemini" in merged["providers"]
        _reset_cache()

    def test_no_providers_block_warning_on_subsequent_run(self, tmp_path, monkeypatch, capsys):
        _git_init(tmp_path)
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["claude-code"])
        _reset_cache()
        init_config.main(["--dir", str(tmp_path)])
        capsys.readouterr()
        _reset_cache()
        registry.load_config(anchor=str(tmp_path))
        assert "ignoring 'providers:' block" not in capsys.readouterr().err
        _reset_cache()

    def test_gemini_only_resolves_nonempty_defaults(self, tmp_path, monkeypatch):
        _git_init(tmp_path)
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["gemini"])
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        _reset_cache()
        assert registry.get_default_models(anchor=str(tmp_path)) == ["gemini:gemini-3-flash"]
        assert registry.get_quick_models(anchor=str(tmp_path)) == ["gemini:gemini-3-flash"]
        _reset_cache()

    def test_modes_example_safe_when_claude_and_cursor_detected(self, tmp_path, monkeypatch):
        # Both providers appear as items in the commented in-marker modes example;
        # the written body must still parse cleanly (items NOT uncommented).
        _git_init(tmp_path)
        monkeypatch.setattr(
            init_config, "_available_providers", lambda cfg: ["claude-code", "cursor-agent"]
        )
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        written = yaml.safe_load((tmp_path / ".multi-llm" / "providers.yaml").read_text())
        # defaults.modes is never materialized by init.
        assert "modes" not in written["defaults"]


@pytest.mark.config_override
class TestReinitForce:
    def test_reinit_reflects_new_detection_and_preserves_outside_edits(
        self, tmp_path, monkeypatch
    ):
        _git_init(tmp_path)
        out = tmp_path / ".multi-llm" / "providers.yaml"

        # First init: only cursor-agent detected.
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["cursor-agent"])
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        first = yaml.safe_load(out.read_text())
        assert first["default_provider"] == "cursor-agent"
        assert "claude-code" not in first["providers"]

        # Hand-add an OUTSIDE-marker comment.
        sentinel = "# HAND-EDITED-OUTSIDE-MARKER do not clobber\n"
        out.write_text(out.read_text() + sentinel)

        # Detection changes: claude-code now installed too. Re-init --force.
        monkeypatch.setattr(
            init_config, "_available_providers", lambda cfg: ["claude-code", "cursor-agent"]
        )
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path), "--force"]) == 0

        text = out.read_text()
        second = yaml.safe_load(text)
        # Regenerated region reflects the NEW detection.
        assert second["default_provider"] == "claude-code"
        assert "claude-code" in second["providers"]
        # Outside-marker hand-edit survived verbatim.
        assert sentinel.strip() in text
        _reset_cache()


@pytest.mark.config_override
class TestBackupOnOverwrite:
    def _backups(self, out: Path) -> "list[Path]":
        return sorted(out.parent.glob(out.name + ".bak.*"))

    def test_first_init_makes_no_backup(self, tmp_path, monkeypatch):
        _git_init(tmp_path)
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["cursor-agent"])
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert self._backups(out) == []
        _reset_cache()

    def test_force_overwrite_backs_up_and_notifies(self, tmp_path, monkeypatch, capsys):
        _git_init(tmp_path)
        out = tmp_path / ".multi-llm" / "providers.yaml"

        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["cursor-agent"])
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        original = out.read_text()
        capsys.readouterr()

        # Detection changes → the regenerated file differs → backup fires.
        monkeypatch.setattr(
            init_config, "_available_providers", lambda cfg: ["claude-code", "cursor-agent"]
        )
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path), "--force"]) == 0
        backups = self._backups(out)
        assert len(backups) == 1
        # The backup preserves the pre-overwrite content verbatim...
        assert backups[0].read_text() == original
        assert out.read_text() != original
        # ...and the user is told where it went.
        stdout = capsys.readouterr().out
        assert "Backed up the previous config to" in stdout
        assert str(backups[0]) in stdout
        _reset_cache()

    def test_identical_force_rerun_skips_backup(self, tmp_path, monkeypatch, capsys):
        _git_init(tmp_path)
        out = tmp_path / ".multi-llm" / "providers.yaml"
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: ["cursor-agent"])
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        capsys.readouterr()
        # Same detection → byte-identical regeneration → nothing to preserve.
        _reset_cache()
        assert init_config.main(["--dir", str(tmp_path), "--force"]) == 0
        assert self._backups(out) == []
        assert "Backed up" not in capsys.readouterr().out
        _reset_cache()

    def test_template_only_force_backs_up_custom_content(self, tmp_path, capsys):
        assert init_config.main(["--dir", str(tmp_path), "--template-only"]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        out.write_text("custom: content\n")
        capsys.readouterr()
        assert init_config.main(["--dir", str(tmp_path), "--template-only", "--force"]) == 0
        backups = self._backups(out)
        assert len(backups) == 1
        assert backups[0].read_text() == "custom: content\n"
        assert out.read_text() == init_config.TEMPLATE_PATH.read_text()
        assert "Backed up the previous config to" in capsys.readouterr().out

    def test_same_second_collision_gets_counter_suffix(self, tmp_path, monkeypatch):
        _git_init(tmp_path)
        out = tmp_path / ".multi-llm" / "providers.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("v1\n")
        fixed = datetime(2026, 7, 10, 12, 0, 0)

        class _FixedDatetime:
            @staticmethod
            def now():
                return fixed

        monkeypatch.setattr(init_config, "datetime", _FixedDatetime)
        first = init_config._backup_existing_config(out, "new1\n")
        out.write_text("v2\n")
        second = init_config._backup_existing_config(out, "new2\n")
        assert first is not None and second is not None
        assert first != second
        assert second.name.endswith("-1")
        assert first.read_text() == "v1\n"
        assert second.read_text() == "v2\n"


@pytest.mark.config_override
class TestTemplateOnlyAndNothingDetected:
    def test_template_only_byte_identical(self, tmp_path):
        assert init_config.main(["--dir", str(tmp_path), "--template-only"]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert out.read_text() == init_config.TEMPLATE_PATH.read_text()

    def test_nothing_detected_writes_inert_and_exits_0(self, tmp_path, monkeypatch, capsys):
        _git_init(tmp_path)
        monkeypatch.setattr(init_config, "_available_providers", lambda cfg: [])
        _reset_cache()
        rc = init_config.main(["--dir", str(tmp_path)])
        assert rc == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        parsed = yaml.safe_load(out.read_text())
        # Inert: no live providers / default_provider; defaults inherit base.
        assert set(parsed.keys()) == {"defaults"}
        assert parsed["defaults"] == {"models": None, "quick_models": None}
        assert "No supported provider CLIs found" in capsys.readouterr().out
        _reset_cache()

    def test_refuses_overwrite_without_force(self, tmp_path):
        assert init_config.main(["--dir", str(tmp_path), "--template-only"]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        out.write_text("custom: content\n")
        assert init_config.main(["--dir", str(tmp_path), "--template-only"]) == 1
        assert out.read_text() == "custom: content\n"


class TestValidation:
    def test_validate_rejects_unknown_provider_in_defaults(self):
        base = init_config.load_base_config()
        text = (
            "default_provider: claude-code\n"
            "providers:\n  claude-code:\n    default_timeout: 1800\n"
            "defaults:\n  models:\n    - bogusprov:x\n  quick_models:\n    - claude-code:opus\n"
        )
        err = init_config._validate_generated_config(text, {"claude-code"}, base)
        assert err and "unknown provider 'bogusprov'" in err

    def test_validate_rejects_default_provider_not_detected(self):
        base = init_config.load_base_config()
        text = (
            "default_provider: gemini\n"
            "defaults:\n  models:\n    - claude-code:opus\n"
            "  quick_models:\n    - claude-code:opus\n"
        )
        err = init_config._validate_generated_config(text, {"claude-code"}, base)
        assert err and "default_provider 'gemini'" in err

    def test_validate_rejects_empty_panel_despite_detection(self):
        base = init_config.load_base_config()
        text = (
            "default_provider: claude-code\n"
            "defaults:\n  models: []\n  quick_models: []\n"
        )
        err = init_config._validate_generated_config(text, {"claude-code"}, base)
        assert err and "empty despite detected providers" in err

    def test_validate_passes_for_inert_nothing_detected(self):
        base = init_config.load_base_config()
        text = "defaults:\n  models:\n  quick_models:\n"
        assert init_config._validate_generated_config(text, set(), base) is None


class TestRemovalRegression:
    @pytest.mark.parametrize(
        "flag",
        ["--emit-catalog", "--from-selections", "--timeout", "--show-all",
         "--json", "--non-interactive"],
    )
    def test_removed_flags_rejected(self, flag, tmp_path):
        with pytest.raises(SystemExit):
            init_config.main([flag, "x", "--dir", str(tmp_path)])

    def test_removed_functions_absent(self):
        for name in (
            "run_interactive", "run_from_selections", "emit_catalog",
            "render_managed_block", "_write_interactive", "_pick_provider_models",
        ):
            assert not hasattr(init_config, name), f"{name} should be removed"

    def test_removed_interactive_primitives_absent(self):
        import utils.interactive as interactive

        for name in (
            "select_with_actions", "ActionSelection", "select_one",
            "prompt_text", "prompt_yes_no", "ACTION_SHOW_ALL", "ACTION_ENTER_MANUAL",
        ):
            assert not hasattr(interactive, name), f"{name} should be removed"
        # The runtime cascade sentinel + helpers are KEPT.
        for name in ("UNAVAILABLE", "select_multi", "resolve_models", "is_tty"):
            assert hasattr(interactive, name)

    def test_removed_listing_helpers_absent(self):
        import utils.providers.base as base

        for name in (
            "ModelListing", "build_models_listing", "run_models_command",
            "strip_ansi", "is_valid_bare_id", "parse_line_ids", "float_curated",
        ):
            assert not hasattr(base, name), f"{name} should be removed"
        # Adapters no longer advertise a listing capability.
        for name in ("cursor-agent", "opencode", "kilocode"):
            provider = registry.get_provider(name)
            assert not hasattr(provider, "list_models")
            assert not hasattr(provider, "can_list_models")
