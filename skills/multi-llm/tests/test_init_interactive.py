"""Tests for the interactive `--init` flow, the per-adapter model listing, and
the sentinel-aware picker primitives.

Covers:
- Fixture-driven `list_models()` parsing per adapter (success / ANSI / empty /
  auth-error / JSON shapes) and the failure-tolerant curated fallback.
- `select_with_actions` sentinel exclusivity + precedence, and the other picker
  primitives.
- The init flow end-to-end (interactive simulated): selection-keys-only write,
  zero-models abort, cancel-at-confirm no-op, quick disable, non-TTY template-only,
  template byte-identity, and mode-shadowing warnings + resolution.
"""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import init_config
import utils.provider_registry as registry
from utils.interactive import (
    ACTION_ENTER_MANUAL,
    ACTION_SHOW_ALL,
    ActionSelection,
    UNAVAILABLE,
    select_one,
    select_with_actions,
)
from utils.providers.base import (
    ModelListing,
    build_models_listing,
    float_curated,
    is_valid_bare_id,
    parse_line_ids,
    run_models_command,
    strip_ansi,
    try_parse_json_ids,
)

FIXTURES = Path(__file__).parent / "fixtures" / "model_listings"


def _fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _reset_cache():
    registry._config = None
    registry._config_key = None


# ---------------------------------------------------------------------------
# base.py helpers
# ---------------------------------------------------------------------------


class TestBaseHelpers:
    def test_is_valid_bare_id(self):
        assert is_valid_bare_id("gpt-5.2-high")
        assert is_valid_bare_id("opencode/sonnet")          # slash allowed
        assert is_valid_bare_id("openrouter/moonshotai/kimi-k2")
        assert not is_valid_bare_id("foo:bar")              # colon rejected
        assert not is_valid_bare_id("has space")
        assert not is_valid_bare_id("")
        assert not is_valid_bare_id(None)

    def test_strip_ansi(self):
        assert strip_ansi("\x1b[32mhi\x1b[0m") == "hi"
        assert strip_ansi("plain") == "plain"

    def test_float_curated_floats_curated_to_top_dedup(self):
        out = float_curated(["x", "y", "a"], ["a", "z"])
        # curated first (a, z — z absent from full still kept), then remaining full
        assert out == ["a", "z", "x", "y"]

    def test_float_curated_drops_colon_curated(self):
        out = float_curated(["x"], ["bad:id", "good"])
        assert "bad:id" not in out and out[0] == "good"

    def test_try_parse_json_strings(self):
        ids = try_parse_json_ids(_fx("json_array_of_strings.json"))
        assert ids == [
            "opencode/big-pickle", "opencode/sonnet",
            "google/gemini-2.5-pro", "anthropic/claude-opus-4.8",
        ]

    def test_try_parse_json_objects(self):
        ids = try_parse_json_ids(_fx("json_array_of_objects.json"))
        assert ids == ["gpt-5.2-high", "composer-2.5", "claude-opus-4-8-thinking-high"]

    def test_try_parse_json_returns_none_for_plaintext(self):
        assert try_parse_json_ids("auto - Auto\ncomposer-2.5 - Composer") is None
        assert try_parse_json_ids("") is None

    def test_run_models_command_nonzero_returns_none(self):
        # `false` exits non-zero → None (never raises).
        assert run_models_command(["false"], timeout=5) is None

    def test_run_models_command_missing_binary_returns_none(self):
        assert run_models_command(["definitely-not-a-real-cli-xyz", "models"], timeout=2) is None

    def test_run_models_command_success(self):
        out = run_models_command(["printf", "a\\nb\\n"], timeout=5)
        assert out is not None and "a" in out


# ---------------------------------------------------------------------------
# Per-adapter list_models() parsing (fixture-driven)
# ---------------------------------------------------------------------------


class TestAdapterListModels:
    def test_capability_flags(self):
        assert registry.get_provider("cursor-agent").can_list_models is True
        assert registry.get_provider("opencode").can_list_models is True
        assert registry.get_provider("kilocode").can_list_models is True
        assert registry.get_provider("codex").can_list_models is False
        assert registry.get_provider("gemini").can_list_models is False
        assert registry.get_provider("claude-code").can_list_models is False

    def test_cursor_parses_success_fixture(self):
        ids = registry.get_provider("cursor-agent")._parse_models(_fx("cursor_agent_success.txt"))
        assert "auto" in ids and "composer-2.5" in ids
        assert "Available models" not in ids
        assert not any(":" in i for i in ids)            # Tip line / colon ids skipped
        assert not any(i.startswith("Tip") for i in ids)

    def test_cursor_parses_ansi_fixture(self):
        ids = registry.get_provider("cursor-agent")._parse_models(_fx("cursor_agent_ansi.txt"))
        assert ids == ["auto", "gpt-5.2-high", "composer-2.5", "claude-opus-4-8-thinking-high"]

    def test_opencode_parses_success(self):
        ids = parse_line_ids(_fx("opencode_success.txt"))
        assert "opencode/big-pickle" in ids
        assert all("/" in i for i in ids) and not any(":" in i for i in ids)

    def test_kilocode_parses_success_drops_colon_ids(self):
        raw = _fx("kilocode_success.txt")
        assert ":free" in raw and ":discounted" in raw      # fixture really has them
        ids = parse_line_ids(raw)
        assert "openrouter/moonshotai/kimi-k2" in ids
        assert not any(":" in i for i in ids)               # :free / :discounted dropped

    def test_empty_and_auth_error_parse_to_nothing(self):
        cur = registry.get_provider("cursor-agent")
        assert cur._parse_models(_fx("empty.txt")) == []
        assert cur._parse_models(_fx("auth_error.txt")) == []
        assert parse_line_ids(_fx("auth_error.txt")) == []

    def test_list_models_success_floats_curated(self, monkeypatch):
        cur = registry.get_provider("cursor-agent")
        monkeypatch.setattr(
            "utils.providers.base.run_models_command",
            lambda argv, *, timeout: _fx("cursor_agent_success.txt"),
        )
        ml = cur.list_models(["composer-2.5", "gpt-5.2-high"], timeout=5)
        assert ml.source == "cli" and ml.note is None
        assert ml.recommended == ["composer-2.5", "gpt-5.2-high"]
        assert ml.models[:2] == ["composer-2.5", "gpt-5.2-high"]   # curated floated to top
        assert "auto" in ml.models

    def test_list_models_command_failure_falls_back_to_curated(self, monkeypatch):
        cur = registry.get_provider("cursor-agent")
        monkeypatch.setattr(
            "utils.providers.base.run_models_command", lambda argv, *, timeout: None
        )
        ml = cur.list_models(["composer-2.5"], timeout=5)
        assert ml.source == "curated" and ml.models == []
        assert ml.recommended == ["composer-2.5"] and ml.note

    def test_list_models_garbled_output_falls_back(self, monkeypatch):
        cur = registry.get_provider("cursor-agent")
        monkeypatch.setattr(
            "utils.providers.base.run_models_command",
            lambda argv, *, timeout: "@@@ not a model list @@@\n???",
        )
        ml = cur.list_models(["composer-2.5"], timeout=5)
        assert ml.source == "curated" and ml.note

    def test_list_models_never_raises_on_timeout(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="cursor-agent models", timeout=1)

        monkeypatch.setattr("utils.providers.base.subprocess.run", boom)
        ml = registry.get_provider("cursor-agent").list_models(["composer-2.5"], timeout=1)
        assert ml.source == "curated" and ml.models == []

    def test_non_listing_default_returns_curated(self):
        ml = registry.get_provider("codex").list_models(["gpt-5.5"])
        assert ml.source == "curated" and ml.models == [] and ml.recommended == ["gpt-5.5"]

    def test_build_models_listing_empty_parser_falls_back(self):
        ml = build_models_listing(
            ["printf", "x"], lambda raw: [], ["c1", "c2"], timeout=5
        )
        assert ml.source == "curated" and ml.recommended == ["c1", "c2"] and ml.note

    def test_build_models_listing_parser_exception_falls_back(self, monkeypatch):
        # A parser that raises must not propagate (build_models_listing "Never
        # raises"); it degrades to the same curated fallback as empty output.
        monkeypatch.setattr(
            "utils.providers.base.run_models_command",
            lambda argv, *, timeout: "some raw output",
        )

        def boom(raw):
            raise ValueError("garbled CLI output")

        ml = build_models_listing(["printf", "x"], boom, ["c1", "c2"], timeout=5)
        assert ml.source == "curated" and ml.models == []
        assert ml.recommended == ["c1", "c2"] and ml.note


# ---------------------------------------------------------------------------
# select_with_actions sentinel contract
# ---------------------------------------------------------------------------


class TestSelectWithActions:
    def _patch(self, monkeypatch, returns):
        monkeypatch.setattr("utils.interactive.is_tty", lambda: True)
        monkeypatch.setattr("utils.interactive.select_multi", lambda opts, prompt: returns)

    def test_ordinary_rows_only(self, monkeypatch):
        self._patch(monkeypatch, ["composer-2.5", "gpt-5.2-high"])
        r = select_with_actions(["composer-2.5", "gpt-5.2-high"], "p",
                                show_all_label="SA", manual_label="MAN")
        assert r.action is None and r.selected == ["composer-2.5", "gpt-5.2-high"]
        assert not r.cancelled

    def test_show_all_wins_over_ordinary_and_manual(self, monkeypatch):
        self._patch(monkeypatch, ["composer-2.5", "SA", "MAN"])
        r = select_with_actions(["composer-2.5"], "p", show_all_label="SA", manual_label="MAN")
        assert r.action == ACTION_SHOW_ALL and r.selected == []

    def test_manual_alone(self, monkeypatch):
        self._patch(monkeypatch, ["MAN"])
        r = select_with_actions(["composer-2.5"], "p", show_all_label="SA", manual_label="MAN")
        assert r.action == ACTION_ENTER_MANUAL

    def test_zero_selected_is_cancel(self, monkeypatch):
        self._patch(monkeypatch, [])
        r = select_with_actions(["composer-2.5"], "p", show_all_label="SA", manual_label="MAN")
        assert r.cancelled and r.action is None

    def test_only_sentinel_no_show_all_label(self, monkeypatch):
        # Provider can't list → no show_all_label; manual only.
        self._patch(monkeypatch, ["MAN"])
        r = select_with_actions(["c"], "p", show_all_label=None, manual_label="MAN")
        assert r.action == ACTION_ENTER_MANUAL


# ---------------------------------------------------------------------------
# Other picker primitives
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_prompt_text(self, monkeypatch):
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda p="": "my-model")
        assert I.prompt_text("id:") == "my-model"
        monkeypatch.setattr("builtins.input", lambda p="": "   ")
        assert I.prompt_text("id:") is None

    def test_prompt_yes_no(self, monkeypatch):
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda p="": "y")
        assert I.prompt_yes_no("ok?") is True
        monkeypatch.setattr("builtins.input", lambda p="": "")
        assert I.prompt_yes_no("ok?", default=True) is True
        assert I.prompt_yes_no("ok?", default=False) is False

    def test_select_one_numbered(self, monkeypatch):
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr("shutil.which", lambda name: None)   # no gum/fzf → numbered
        monkeypatch.setattr("builtins.input", lambda p="": "2")
        assert I.select_one(["a", "b", "c"], "pick", default="a") == "b"
        monkeypatch.setattr("builtins.input", lambda p="": "")
        assert I.select_one(["a", "b", "c"], "pick", default="c") == "c"


class TestSelectOneCascade:
    """select_one cascade: UNAVAILABLE falls through; a cancel returns default."""

    def test_unavailable_backend_falls_through_to_next(self, monkeypatch):
        """gum UNAVAILABLE → cascade tries fzf, whose selection is returned."""
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr(I, "_try_gum_choose_one", lambda rows, p, default: UNAVAILABLE)
        monkeypatch.setattr(I, "_try_fzf_one", lambda rows, p: "b")

        called = {"numbered": False}

        def _numbered(rows, p, default):
            called["numbered"] = True
            return "should-not-happen"

        monkeypatch.setattr(I, "_numbered_prompt_one", _numbered)

        assert select_one(["a", "b", "c"], "pick", default="a") == "b"
        assert called["numbered"] is False

    def test_gum_cancel_returns_default_without_cascading(self, monkeypatch):
        """A cancelled gum (ran → None) stops the cascade and returns the default.

        Regression guard for the reported bug: an Esc / empty gum single-select
        must NOT fall through and re-prompt the user via fzf or the numbered
        fallback — it returns the documented default instead.
        """
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr(I, "_try_gum_choose_one", lambda rows, p, default: None)

        calls = {"fzf": False, "numbered": False}

        def _fzf(rows, p):
            calls["fzf"] = True
            return "leaked-from-fzf"

        def _numbered(rows, p, default):
            calls["numbered"] = True
            return "leaked-from-numbered"

        monkeypatch.setattr(I, "_try_fzf_one", _fzf)
        monkeypatch.setattr(I, "_numbered_prompt_one", _numbered)

        assert select_one(["a", "b", "c"], "pick", default="c") == "c"
        assert calls["fzf"] is False
        assert calls["numbered"] is False

    def test_fzf_cancel_returns_default_without_cascading(self, monkeypatch):
        """gum UNAVAILABLE, fzf ran-but-cancelled (→ None) → default; numbered unused."""
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr(I, "_try_gum_choose_one", lambda rows, p, default: UNAVAILABLE)
        monkeypatch.setattr(I, "_try_fzf_one", lambda rows, p: None)

        called = {"numbered": False}

        def _numbered(rows, p, default):
            called["numbered"] = True
            return "leaked"

        monkeypatch.setattr(I, "_numbered_prompt_one", _numbered)

        assert select_one(["a", "b", "c"], "pick", default="b") == "b"
        assert called["numbered"] is False

    def test_both_unavailable_falls_back_to_numbered(self, monkeypatch):
        """gum and fzf both UNAVAILABLE → numbered fallback is used."""
        import utils.interactive as I
        monkeypatch.setattr(I, "is_tty", lambda: True)
        monkeypatch.setattr(I, "_try_gum_choose_one", lambda rows, p, default: UNAVAILABLE)
        monkeypatch.setattr(I, "_try_fzf_one", lambda rows, p: UNAVAILABLE)
        monkeypatch.setattr(I, "_numbered_prompt_one", lambda rows, p, default: "a")

        assert select_one(["a", "b", "c"], "pick", default="c") == "a"


# ---------------------------------------------------------------------------
# Init flow end-to-end (interactive simulated)
# ---------------------------------------------------------------------------


def _git_repo(tmp_path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _all_available(monkeypatch, *, gum_fzf=False):
    """Make every provider CLI 'available'; gum/fzf optional."""
    real = __import__("shutil").which

    def which(name):
        if name in {"gum", "fzf"}:
            return f"/usr/bin/{name}" if gum_fzf else None
        if name in {"claude", "cursor-agent", "gemini", "opencode", "codex", "kilocode"}:
            return f"/usr/bin/{name}"
        return real(name)

    monkeypatch.setattr("shutil.which", which)


def _drive(monkeypatch, *, picks: dict, quick_yes=True, confirm=True,
           default_provider=None, edit=False):
    """Wire init_config's interactive primitives for a scripted run.

    picks: provider-name → ActionSelection (or list of bare ids → selected).
    """
    monkeypatch.setattr(init_config, "is_tty", lambda: True)

    def swa(rows, prompt, *, show_all_label=None, manual_label=None):
        for name, sel in picks.items():
            if prompt.startswith(f"Select {name} "):
                return sel if isinstance(sel, ActionSelection) else ActionSelection(selected=list(sel))
        return ActionSelection(cancelled=True)

    monkeypatch.setattr(init_config, "select_with_actions", swa)
    monkeypatch.setattr(init_config, "select_multi", lambda opts, prompt: list(opts))

    edited = {"done": False}

    def yes_no(prompt, default=False):
        if "Use these as your --quick panel" in prompt:
            return quick_yes
        if "Disable --quick entirely" in prompt:
            return False
        if "Write this config" in prompt:
            return confirm
        if "Edit your selections" in prompt:
            if edit and not edited["done"]:
                edited["done"] = True
                return True
            return False
        return default

    monkeypatch.setattr(init_config, "prompt_yes_no", yes_no)
    monkeypatch.setattr(
        init_config, "select_one",
        lambda rows, prompt, default=None: default_provider or default,
    )


@pytest.mark.config_override
class TestInitInteractiveFlow:
    def test_writes_expected_selection_keys(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={
            "cursor-agent": ["composer-2.5", "gpt-5.2-high"],
            "claude-code": ["opus"],
        })
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0

        out = repo / ".multi-llm" / "providers.yaml"
        cfg = yaml.safe_load(out.read_text())
        # Provider order follows base-config declaration order (claude-code first).
        assert cfg["defaults"]["models"] == [
            "claude-code:opus", "cursor-agent:composer-2.5", "cursor-agent:gpt-5.2-high",
        ]
        assert cfg["defaults"]["quick_models"] == ["claude-code:opus", "cursor-agent:composer-2.5"]
        assert cfg["default_provider"] == "claude-code"
        assert "providers" not in cfg                      # selection-keys only

        # Post-write reload through load_config (fresh cache) exposes the choices.
        _reset_cache()
        merged = registry.load_config(anchor=str(repo))
        assert merged["defaults"]["models"] == cfg["defaults"]["models"]
        assert merged["default_provider"] == "claude-code"

    def test_generated_file_yaml_loads_cleanly(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]})
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        text = (repo / ".multi-llm" / "providers.yaml").read_text()
        assert isinstance(yaml.safe_load(text), dict)

    def test_show_all_noncatalog_pick_is_written_and_flagged(self, tmp_path, monkeypatch, capsys):
        _all_available(monkeypatch)
        cur = registry.get_provider("cursor-agent")
        monkeypatch.setattr(
            cur, "list_models",
            lambda curated, *, timeout=10: ModelListing(
                models=["made-up-model", "composer-2.5"], source="cli", recommended=list(curated)
            ),
        )
        _drive(monkeypatch, picks={"cursor-agent": ActionSelection(action=ACTION_SHOW_ALL)})
        # select_multi returns all options; restrict to the non-catalog id
        monkeypatch.setattr(init_config, "select_multi", lambda opts, prompt: ["made-up-model"])
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        out = capsys.readouterr().out
        assert "(unverified id)" in out
        cfg = yaml.safe_load((repo / ".multi-llm" / "providers.yaml").read_text())
        assert cfg["defaults"]["models"] == ["cursor-agent:made-up-model"]

    def test_manual_entry_accepted_verbatim(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ActionSelection(action=ACTION_ENTER_MANUAL)})
        ids = iter(["hand-typed-model", None])
        monkeypatch.setattr(init_config, "prompt_text", lambda prompt: next(ids))
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        cfg = yaml.safe_load((repo / ".multi-llm" / "providers.yaml").read_text())
        assert cfg["defaults"]["models"] == ["cursor-agent:hand-typed-model"]

    def test_zero_models_exits_1_no_write(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={})            # every provider cancelled → nothing
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 1
        assert not (repo / ".multi-llm" / "providers.yaml").exists()

    def test_cancel_at_confirm_exits_0_no_write(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]}, confirm=False)
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        assert not (repo / ".multi-llm" / "providers.yaml").exists()

    def test_quick_disable_writes_empty_list(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        monkeypatch.setattr(init_config, "is_tty", lambda: True)

        def swa(rows, prompt, *, show_all_label=None, manual_label=None):
            if prompt.startswith("Select cursor-agent "):
                return ActionSelection(selected=["composer-2.5"])
            return ActionSelection(cancelled=True)

        monkeypatch.setattr(init_config, "select_with_actions", swa)
        # quick: decline proposal, then select none, then confirm disable
        monkeypatch.setattr(init_config, "select_multi", lambda opts, prompt: [])

        def yes_no(prompt, default=False):
            if "Use these as your --quick panel" in prompt:
                return False
            if "Disable --quick entirely" in prompt:
                return True
            if "Write this config" in prompt:
                return True
            return default

        monkeypatch.setattr(init_config, "prompt_yes_no", yes_no)
        monkeypatch.setattr(init_config, "select_one", lambda rows, p, default=None: default)
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        cfg = yaml.safe_load((repo / ".multi-llm" / "providers.yaml").read_text())
        assert cfg["defaults"]["quick_models"] == []

    def test_edit_loop_reenters_then_writes(self, tmp_path, monkeypatch):
        _all_available(monkeypatch)
        # First confirm declined → edit → second confirm accepted.
        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]},
               confirm=False, edit=True)
        # Override: confirm should be True on the SECOND pass. Simplest: make confirm
        # a counter via prompt_yes_no replacement.
        state = {"confirm_calls": 0}

        def yes_no(prompt, default=False):
            if "Use these as your --quick panel" in prompt:
                return True
            if "Write this config" in prompt:
                state["confirm_calls"] += 1
                return state["confirm_calls"] >= 2      # decline first, accept second
            if "Edit your selections" in prompt:
                return True
            return default

        monkeypatch.setattr(init_config, "prompt_yes_no", yes_no)
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        assert state["confirm_calls"] == 2
        assert (repo / ".multi-llm" / "providers.yaml").exists()

    def test_reinit_preserves_outside_marker_edits(self, tmp_path, monkeypatch):
        # A re-init (--force) over an existing marked config must splice into the
        # EXISTING file, preserving hand-maintained content outside the markers
        # verbatim (template contract), while regenerating the managed block.
        _all_available(monkeypatch)
        repo = _git_repo(tmp_path)
        out = repo / ".multi-llm" / "providers.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Seed an existing config: template + a hand-added comment/key ABOVE the
        # managed region and a comment BELOW it.
        template = init_config.TEMPLATE_PATH.read_text()
        seeded = template.replace(
            init_config.MARKER_START,
            "# HAND-MAINTAINED above the markers\n" + init_config.MARKER_START,
            1,
        )
        seeded = seeded.rstrip("\n") + "\n# HAND-MAINTAINED below the markers\n"
        out.write_text(seeded)

        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]})
        _reset_cache()
        assert init_config.main(["--dir", str(repo), "--force"]) == 0

        result = out.read_text()
        # Outside-the-markers hand edits survive.
        assert "# HAND-MAINTAINED above the markers" in result
        assert "# HAND-MAINTAINED below the markers" in result
        # The managed block was regenerated with the new selection.
        cfg = yaml.safe_load(result)
        assert cfg["defaults"]["models"] == ["cursor-agent:composer-2.5"]

    def test_first_write_uses_bundled_template(self, tmp_path, monkeypatch):
        # First-time write (no existing file) splices into the bundled template so
        # the header comments / MODE SHADOWING note are present.
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]})
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        text = (repo / ".multi-llm" / "providers.yaml").read_text()
        assert "MODE SHADOWING" in text                      # template header preserved

    def test_reinit_markerless_file_resets_to_template(self, tmp_path, monkeypatch):
        # An existing file missing the markers has no managed region to preserve;
        # the bundled template is used (reset), restoring the marker region.
        _all_available(monkeypatch)
        repo = _git_repo(tmp_path)
        out = repo / ".multi-llm" / "providers.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("default_provider: gemini\n# legacy file, no markers\n")

        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]})
        _reset_cache()
        assert init_config.main(["--dir", str(repo), "--force"]) == 0
        text = out.read_text()
        assert init_config.MARKER_START in text and init_config.MARKER_END in text
        assert "MODE SHADOWING" in text                      # template header restored
        assert "# legacy file, no markers" not in text       # legacy body replaced


# ---------------------------------------------------------------------------
# Non-interactive / template-only behavior
# ---------------------------------------------------------------------------


class TestTemplateOnlyAndNonTTY:
    def test_non_tty_falls_back_to_template_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(init_config, "is_tty", lambda: False)
        assert init_config.main(["--dir", str(tmp_path)]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert out.read_text() == init_config.TEMPLATE_PATH.read_text()

    def test_template_only_flag_byte_identical(self, tmp_path):
        assert init_config.main(["--dir", str(tmp_path), "--template-only"]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        # Inert marker comments must not alter the verbatim stub.
        assert out.read_text() == init_config.TEMPLATE_PATH.read_text()

    def test_non_interactive_flag_is_template_only(self, tmp_path, monkeypatch):
        # Even on a 'TTY', --non-interactive must not prompt.
        monkeypatch.setattr(init_config, "is_tty", lambda: True)
        assert init_config.main(["--dir", str(tmp_path), "--non-interactive"]) == 0
        out = tmp_path / ".multi-llm" / "providers.yaml"
        assert out.read_text() == init_config.TEMPLATE_PATH.read_text()

    def test_no_providers_interactive_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(init_config, "is_tty", lambda: True)
        monkeypatch.setattr("shutil.which", lambda name: None)   # nothing installed
        assert init_config.main(["--dir", str(tmp_path)]) == 1
        assert not (tmp_path / ".multi-llm" / "providers.yaml").exists()

    def test_template_contains_marker_region(self):
        text = init_config.TEMPLATE_PATH.read_text()
        assert init_config.MARKER_START in text and init_config.MARKER_END in text
        assert "MODE SHADOWING" in text                     # note lives outside region


# ---------------------------------------------------------------------------
# Mode shadowing: warning + resolution
# ---------------------------------------------------------------------------

# A base config carrying a mode-specific list for `code-review`, used to prove the
# mode list shadows the freshly-written globals.
_BASE_WITH_MODE = {
    "providers": {
        "claude-code": {"command": "claude", "models": ["opus", "sonnet"]},
        "cursor-agent": {"command": "cursor-agent", "models": ["composer-2.5", "gpt-5.2-high"]},
    },
    "default_provider": "cursor-agent",
    "defaults": {
        "models": ["cursor-agent:composer-2.5"],
        "quick_models": ["cursor-agent:composer-2.5"],
        "modes": {"code-review": {"models": ["claude-code:opus"]}},
    },
}


@pytest.mark.config_override
class TestModeShadowing:
    def test_confirm_lists_shadowed_modes(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(init_config, "load_base_config", lambda: copy.deepcopy(_BASE_WITH_MODE))
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ["composer-2.5"]})
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0
        out = capsys.readouterr().out
        assert "code-review" in out and "still WIN" in out

    def test_globals_apply_only_where_no_mode_block(self, tmp_path, monkeypatch):
        # init writes globals using the mode-bearing base...
        monkeypatch.setattr(init_config, "load_base_config", lambda: copy.deepcopy(_BASE_WITH_MODE))
        _all_available(monkeypatch)
        _drive(monkeypatch, picks={"cursor-agent": ["gpt-5.2-high"]})
        repo = _git_repo(tmp_path)
        _reset_cache()
        assert init_config.main(["--dir", str(repo)]) == 0

        # ...and the *runtime* base also carries the mode block. Resolve via the
        # real registry to prove the mode list still wins.
        monkeypatch.setattr(registry, "_load_base_config", lambda: copy.deepcopy(_BASE_WITH_MODE))
        _reset_cache()
        # A mode WITHOUT its own block → freshly-written global.
        assert registry.get_default_models(mode="review-plan", anchor=str(repo)) == [
            "cursor-agent:gpt-5.2-high"
        ]
        # The mode WITH a block → still the mode-specific list (globals don't win).
        _reset_cache()
        assert registry.get_default_models(mode="code-review", anchor=str(repo)) == [
            "claude-code:opus"
        ]


# ---------------------------------------------------------------------------
# Dynamically configured model reaches invocation (warn-and-proceed path)
# ---------------------------------------------------------------------------


@pytest.mark.config_override
class TestConfiguredModelReachesInvocation:
    def test_noncatalog_default_survives_resolution(self, tmp_path, monkeypatch):
        repo = _git_repo(tmp_path)
        (repo / ".multi-llm").mkdir()
        (repo / ".multi-llm" / "providers.yaml").write_text(
            "default_provider: cursor-agent\n"
            "defaults:\n"
            "  models:\n"
            "    - cursor-agent:made-up-not-in-catalog\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        _reset_cache()
        from utils.interactive import resolve_models
        resolved = resolve_models(mode=None, anchor=str(repo / "plan.md"))
        # The non-catalog configured spec is dispatched (warn-and-proceed).
        assert "cursor-agent:made-up-not-in-catalog" in resolved
