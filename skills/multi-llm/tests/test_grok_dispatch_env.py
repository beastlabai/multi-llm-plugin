"""grok's two dispatch-time switches, and the retry envelope they make true.

Review 24, pins #1/#2, measured 2026-10-08 against grok 1.0.46 in the pinned
image (``--network none``, the probe's own stub and failserver, $0):

* ``GROK_TURN_SUMMARY=0`` removes the tool-less second chat call grok 1.0.x makes
  after every answered turn to write itself a one-line summary of the reply: an
  answered exec is four requests, not five, with the answer's bytes unchanged.
* ``GROK_CONFIG`` ``{"models": {"max_retries": 2}}`` bounds the ladder a
  fail-forever 5xx ran for ~1,341 s (60 chat requests, zero bytes on either
  channel) — past providers.yaml's 1200 s grok slot — to 8 chat requests over
  54 s (four runs: 53.97-54.07 s; a zero-byte response 47.4 s).

``GROK_CONFIG`` replaces a ``GROK_CONFIG_PATH`` overlay outright (measured: a
path overlay alone was honoured, beside a ``GROK_CONFIG`` it was not), so the
user's own overlay is carried into the one the adapter sets.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.grok import (  # noqa: E402
    GROK_MAX_RETRIES,
    GrokProvider,
)
from utils.retry_ladder import (  # noqa: E402
    STATUS_TIMEOUT,
    STOP_BUDGET,
    RetryLadder,
)


@pytest.fixture(autouse=True)
def _no_user_overlay(monkeypatch):
    monkeypatch.delenv("GROK_CONFIG", raising=False)
    monkeypatch.delenv("GROK_CONFIG_PATH", raising=False)


def _overlay(env):
    return json.loads(env["GROK_CONFIG"])


def test_every_dispatch_switches_the_turn_summary_off():
    """The billed summary call is off on every lane: get_env is what both
    spawn sites compose (``resolve_env``), whatever the profile."""
    provider = GrokProvider()
    assert provider.get_env("grok-4.5")["GROK_TURN_SUMMARY"] == "0"
    assert provider.resolve_env("grok-4.5:high")["GROK_TURN_SUMMARY"] == "0"


def test_every_dispatch_bounds_grok_s_retry_ladder():
    env = GrokProvider().get_env("grok-4.5")
    assert _overlay(env) == {"models": {"max_retries": GROK_MAX_RETRIES}}
    assert GROK_MAX_RETRIES == 2


def test_the_envelope_is_the_bounded_ladder_and_fits_grok_s_slot():
    """Declared at the measured 11 requests / 55 s, and inside the slot.

    providers.yaml's ``grok.default_timeout`` (1200 s) is the slot that runs;
    the class's own ``default_timeout`` (600 s) is the fallback for a config
    without the row. The unbounded ladder (~1,341 s) fit neither.
    """
    from utils.provider_registry import get_provider_timeout

    envelope = GrokProvider().internal_retries
    assert envelope is not None
    assert (envelope.max_requests, envelope.max_wall_s) == (11, 55.0)
    assert envelope.retries_rate_limit is True
    assert envelope.max_wall_s < get_provider_timeout("grok")
    assert envelope.max_wall_s < GrokProvider().default_timeout


def test_rule_4_refuses_a_grok_attempt_the_slot_cannot_hold_and_says_why():
    """A slot with 50 s left cannot hold one 55 s chain: refused, named."""
    clock = [0.0]
    ladder = RetryLadder(
        timeout=1200.0, retries=2, envelope=GrokProvider().internal_retries,
        now=lambda: clock[0], jitter=lambda: 0.0,
    )
    assert ladder.next_attempt() == pytest.approx(1200.0)
    clock[0] += 1150.0
    action = ladder.record(STATUS_TIMEOUT)
    assert action.retry is False
    assert ladder.stop_reason == STOP_BUDGET
    assert "55s one dispatch of this runner declares" in action.reason


def test_a_users_own_grok_config_is_merged_not_replaced(monkeypatch):
    monkeypatch.setenv("GROK_CONFIG", json.dumps({
        "models": {"default_reasoning_effort": "high", "max_retries": 9},
        "features": {"web_fetch": False},
    }))
    overlay = _overlay(GrokProvider().get_env("grok-4.5"))
    assert overlay == {
        "models": {"default_reasoning_effort": "high", "max_retries": 2},
        "features": {"web_fetch": False},
    }


@pytest.mark.parametrize("name,body", [
    ("overlay.json", '{"features": {"web_fetch": false}}'),
    ("overlay.toml", "[features]\nweb_fetch = false\n"),
])
def test_a_users_grok_config_path_is_carried_into_the_overlay(
    monkeypatch, tmp_path, name, body
):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("GROK_CONFIG_PATH", str(path))
    overlay = _overlay(GrokProvider().get_env("grok-4.5"))
    assert overlay == {
        "features": {"web_fetch": False},
        "models": {"max_retries": 2},
    }


def test_a_malformed_grok_config_falls_through_to_the_path(monkeypatch, tmp_path):
    """grok's own rule: a malformed ``GROK_CONFIG`` is ignored and the path read."""
    path = tmp_path / "overlay.json"
    path.write_text('{"features": {"web_fetch": false}}', encoding="utf-8")
    monkeypatch.setenv("GROK_CONFIG", "{not json")
    monkeypatch.setenv("GROK_CONFIG_PATH", str(path))
    overlay = _overlay(GrokProvider().get_env("grok-4.5"))
    assert overlay["features"] == {"web_fetch": False}


def test_a_relative_overlay_path_is_left_in_force(monkeypatch, tmp_path):
    """grok resolves it against the child's cwd, not the dispatcher's."""
    (tmp_path / "overlay.json").write_text('{"features": {}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROK_CONFIG_PATH", "overlay.json")
    assert "GROK_CONFIG" not in GrokProvider().get_env("grok-4.5")


def test_an_overlay_file_the_adapter_cannot_read_is_left_in_force(
    monkeypatch, tmp_path
):
    """Setting ``GROK_CONFIG`` would drop it, so nothing is pinned — and the
    envelope then says no bound can be stated rather than claiming 55 s."""
    path = tmp_path / "overlay.toml"
    path.write_text("this is = = not toml\n", encoding="utf-8")
    monkeypatch.setenv("GROK_CONFIG_PATH", str(path))
    provider = GrokProvider()
    env = provider.get_env("grok-4.5")
    assert "GROK_CONFIG" not in env
    assert env["GROK_TURN_SUMMARY"] == "0"
    assert provider.internal_retries.max_wall_s is None


def test_the_container_row_sets_what_every_dispatch_sets():
    """The wrapper rebuilds the child env from scratch, so the grok row's
    ``env_extra`` repeats get_env's switches; the two cannot drift apart."""
    import yaml

    redirect = Path(__file__).parent / "container" / "redirect.yaml"
    rows = yaml.safe_load(redirect.read_text(encoding="utf-8"))["clis"]
    row = next(r for r in rows if r.get("cli") == "grok")
    env_extra = {k: str(v) for k, v in (row.get("env_extra") or {}).items()}
    env = GrokProvider().get_env("stub-model")
    assert env_extra["GROK_TURN_SUMMARY"] == env["GROK_TURN_SUMMARY"]
    assert json.loads(env_extra["GROK_CONFIG"]) == json.loads(env["GROK_CONFIG"])
