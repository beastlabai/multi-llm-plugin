"""The runner catalog warning exempts the operator's own roster (plan §6.1,
2026-09-24).

`hosts/capabilities.yaml` pins codex's `default_subagent_runner` to
`codex:gpt-6-astra`, and its comment says the pin "follows the roster". A
project whose `providers.yaml` lists `codex:gpt-6-astra:xhigh` in its roster
but not `gpt-6-astra` in `codex.models` used to see "not in codex's configured
catalog" on every preflight. The exemption is keyed on the model with the
effort suffix stripped from both sides, reads the **effective** (project-layer
resolved) roster, and applies under every key `preflight_runner` checks; the
warning, when it stays, names the providers.yaml to edit and never
`hosts/capabilities.yaml`.
"""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.providers.codex import CodexProvider  # noqa: E402

#: Project-layer discovery is the subject here, so the suite's default
#: neutralization of it is opted out of.
pytestmark = pytest.mark.config_override

PIN_KEY = "driver.subagent_runner (host capabilities)"
PIN = "codex:gpt-6-astra"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A git repo whose `.multi-llm/providers.yaml` is the project layer, and
    a plan inside it to anchor discovery on."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.delenv("MULTI_LLM_PROVIDERS_CONFIG", raising=False)
    monkeypatch.setattr(CodexProvider, "is_available", lambda self: True)
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n", encoding="utf-8")
    layer = tmp_path / ".multi-llm" / "providers.yaml"
    layer.parent.mkdir()
    registry._config = None
    registry._config_key = None

    def write(quick, catalog=("gpt-5.5", "gpt-5.6-sol")):
        layer.write_text(
            "defaults:\n  quick_models:\n"
            + "".join(f"    - {spec}\n" for spec in quick)
            + "  models:\n    - codex:gpt-5.5\n"
            + "providers:\n  codex:\n    models:\n"
            + "".join(f"      - {model}\n" for model in catalog),
            encoding="utf-8",
        )
        registry._config = None
        registry._config_key = None
        return str(plan)

    write.layer = layer
    yield write
    registry._config = None
    registry._config_key = None


def _preflight(spec, anchor, key=PIN_KEY):
    return executor.preflight_runner(
        spec, key=key, anchor=anchor, layer="host",
        check_version=False, check_auth=False, check_nested=False,
    )


class TestTheRosterExemptsThePin:
    def test_the_pin_is_quiet_when_its_stripped_model_is_in_the_project_roster(
        self, project, capsys
    ):
        anchor = project(["cursor-agent:grok-4.7-high", "codex:gpt-6-astra:xhigh"])
        capsys.readouterr()
        _preflight(PIN, anchor)
        assert "configured catalog" not in capsys.readouterr().err

    @pytest.mark.parametrize("key", [
        "driver.subagent_runner",
        "driver.subagent_runner (project)",
        "tasks[T001].runner (recorded from a previous attempt)",
    ])
    def test_every_key_is_exempt(self, project, capsys, key):
        anchor = project(["codex:gpt-6-astra:xhigh"])
        capsys.readouterr()
        _preflight(PIN, anchor, key=key)
        assert "configured catalog" not in capsys.readouterr().err

    def test_a_suffixed_spec_of_a_roster_model_is_exempt_too(self, project, capsys):
        anchor = project(["codex:gpt-6-astra"])
        capsys.readouterr()
        _preflight("codex:gpt-6-astra:xhigh", anchor)
        assert "configured catalog" not in capsys.readouterr().err

    def test_it_warns_when_the_model_is_only_in_the_base_roster(
        self, project, monkeypatch, capsys
    ):
        """The project layer replaced the roster and left the model out; the
        base file's roster does not count."""
        real = registry._load_base_config

        def base_with_the_pin():
            data = copy.deepcopy(real())
            data.setdefault("defaults", {})["quick_models"] = ["codex:gpt-6-astra:xhigh"]
            return data

        monkeypatch.setattr(registry, "_load_base_config", base_with_the_pin)
        anchor = project(["cursor-agent:grok-4.7-high"])
        assert "codex:gpt-6-astra:xhigh" in base_with_the_pin()["defaults"]["quick_models"]
        capsys.readouterr()
        _preflight(PIN, anchor)
        assert "not in codex's configured catalog" in capsys.readouterr().err

    def test_a_typod_id_with_an_effort_suffix_still_warns(self, project, capsys):
        anchor = project(["codex:gpt-6-astra:xhigh"])
        capsys.readouterr()
        _preflight("codex:gpt-6-astrx:xhigh", anchor)
        assert "not in codex's configured catalog" in capsys.readouterr().err

    def test_the_warning_names_the_resolved_providers_yaml_and_codex_models(
        self, project, capsys
    ):
        anchor = project(["cursor-agent:grok-4.7-high"])
        capsys.readouterr()
        _preflight(PIN, anchor)
        err = capsys.readouterr().err
        assert str(project.layer.resolve()) in err
        assert "`codex.models`" in err
        assert "capabilities.yaml" not in err
