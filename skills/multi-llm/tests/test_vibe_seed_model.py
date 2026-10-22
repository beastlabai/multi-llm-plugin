"""A run's Vibe selection must resolve to its seeded provider catalog."""
import importlib
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from .container import wrapper

RIG = Path(__file__).parent / "container"


@pytest.mark.parametrize("model", ["openai/gpt-6-luna", "minimax/minimax-m3.1-flash:free", "deepseek/deepseek-v4.1-flash"])
@pytest.mark.parametrize("live", [False, True])
def test_selected_alias_reaches_the_configured_provider(tmp_path, monkeypatch, model, live):
    home = tmp_path / "home"
    if live:
        monkeypatch.syspath_prepend(str(RIG))
        monkeypatch.setattr(sys, "path", list(sys.path))
        runner = importlib.import_module("live_provider")
        real_path = Path
        monkeypatch.setattr(runner, "Path", lambda value: RIG / "home_seed" if value == "/opt/mllm/rig/home_seed" else real_path(value))
        runner.render_seed("vibe", home, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", model)
    else:
        homes = tmp_path / "homes"
        shutil.copytree(RIG / "home_seed" / "vibe", homes / "vibe")
        monkeypatch.setattr(wrapper, "HOMES_ROOT", homes)
        monkeypatch.setenv("MLLM_STUB_BASE_URL", "http://127.0.0.1:1234/v1")
        wrapper.seed_invocation_root("vibe", home, {"model_arg": model})
    body = (home / ".vibe/config.toml").read_text()
    assert "{{" not in body
    config = tomllib.loads(body)
    assert config["active_model"] == model
    chosen = next(row for row in config["models"] if row["alias"] == model)
    assert chosen["name"] == (model if live else "stub-model")
    assert chosen["provider"] == "openrouter"
    provider = next(row for row in config["providers"] if row["name"] == chosen["provider"])
    assert provider["api_key_env_var"] == ("OPENROUTER_API_KEY" if live else "STUB_API_KEY")
