"""Subscription-first credentials for the container lanes, pinned offline.

Operator decision 2026-10-02 (``tests/container/host_matrix.yaml``'s MODEL POLICY
block): codex and claude-code run on the developer's OWN subscription whenever
the credential file ``creds.yaml`` allowlists for them is present, and on the
OpenRouter key otherwise; opencode always runs on the key. codex drives
``gpt-6-luna`` at reasoning effort ``high``, claude-code drives ``sonnet``.

Every one of those facts is only ever exercised by a credentialed Docker run, so
a regression would be found by an invoice or by a subscription token in a run
artifact. This module grades the resolver, the model pin, the codex config the
entrypoint writes and the two by-value scrubs with no daemon, no network and no
real credential.

**The filename is deliberately not ``test_container_*``**, for the reason
``test_host_model_pin.py`` gives: that glob is what the image ships and what the
marker-gating check enumerates.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import tomllib
from pathlib import Path

import pytest

from .container import entrypoint
from .container import harness

#: Retyped on purpose: this is the assertion, so reading it from the matrix it
#: guards would make the module agree with any edit at all.
CODEX_SUBSCRIPTION = {"slug": "gpt-6-luna", "effort": "high"}
CLAUDE_SUBSCRIPTION = {"slug": "sonnet"}


def _fake_jwt(claims: dict) -> str:
    """An unsigned three-part token whose payload is CLAIMS (never verified)."""
    def part(obj: dict) -> str:
        raw = json.dumps(obj).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


@pytest.fixture
def creds_policy(monkeypatch, tmp_path):
    """Point `creds.yaml`'s allowlist at files under TMP_PATH, none created yet."""
    files = {host: tmp_path / host / "credential.json"
             for host in ("codex", "claude-code", "opencode")}
    policy = {"allowlist": {host: [{"source": str(path), "dest": f"/run/creds/{host}",
                                    "mode": "ro"}]
                            for host, path in files.items()}}
    monkeypatch.setattr(harness, "load_creds_policy", lambda: policy)
    monkeypatch.delenv(harness.HOST_CREDS_ENV, raising=False)

    def present(host: str) -> Path:
        files[host].parent.mkdir(parents=True, exist_ok=True)
        files[host].write_text("{}", encoding="utf-8")
        return files[host]

    return present


class TestTheSubscriptionPins:
    def test_codex_drives_luna_at_high(self):
        pin = harness.subscription_pin("codex")
        assert {k: pin.get(k) for k in CODEX_SUBSCRIPTION} == CODEX_SUBSCRIPTION
        assert harness.subscription_model_spec("codex") == "gpt-6-luna:high"

    def test_claude_code_drives_sonnet(self):
        pin = harness.subscription_pin("claude-code")
        assert pin.get("slug") == CLAUDE_SUBSCRIPTION["slug"]
        assert not pin.get("effort")

    def test_opencode_has_no_subscription_pin(self):
        """Operator decision: opencode always runs on the OpenRouter key."""
        assert harness.subscription_pin("opencode") == {}

    def test_the_openrouter_pins_are_untouched(self):
        """Subscription-first adds a path; it does not move the key path's pins."""
        hosts = harness.load_host_matrix()
        assert hosts["codex"]["model"]["slug"] == "deepseek/deepseek-v4.1-flash"
        assert hosts["claude-code"]["model"]["slug"] == "openai/gpt-6-luna"


class TestWhichCredentialARowTakes:
    def test_a_present_codex_login_selects_the_subscription(self, creds_policy):
        creds_policy("codex")
        assert harness.default_host_creds("codex") == "fallback"

    def test_a_present_claude_login_selects_the_subscription(self, creds_policy):
        creds_policy("claude-code")
        assert harness.default_host_creds("claude-code") == "fallback"

    def test_an_absent_login_falls_back_to_openrouter(self, creds_policy):
        assert harness.default_host_creds("codex") == "openrouter"
        assert harness.default_host_creds("claude-code") == "openrouter"

    def test_opencode_takes_the_key_even_with_a_stored_login(self, creds_policy):
        creds_policy("opencode")
        assert harness.default_host_creds("opencode") == "openrouter"

    def test_the_env_still_forces_one_mode_on_every_row(self, creds_policy,
                                                         monkeypatch):
        creds_policy("codex")
        monkeypatch.setenv(harness.HOST_CREDS_ENV, "openrouter")
        assert harness.default_host_creds("codex") == "openrouter"
        monkeypatch.setenv(harness.HOST_CREDS_ENV, "fallback")
        assert harness.default_host_creds("opencode") == "fallback"


class TestTheModelTheEntrypointDrives:
    @pytest.fixture(autouse=True)
    def _no_override(self, monkeypatch):
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)

    def _entry(self, host: str) -> dict:
        return dict(harness.load_host_matrix()[host])

    def test_a_subscription_row_drives_the_subscription_pin(self):
        assert entrypoint.pinned_model(self._entry("codex"), None,
                                       "fallback") == "gpt-6-luna"
        assert entrypoint.pinned_model(self._entry("claude-code"), None,
                                       "fallback") == "sonnet"

    def test_a_row_pin_does_not_apply_on_the_subscription(self):
        """Row pins are OpenRouter slugs; a vendor catalog may not serve them."""
        entry = self._entry("claude-code")
        row = {"model": {"slug": "openai/gpt-6-luna"}}
        assert entrypoint.pinned_model(entry, row, "fallback") == "sonnet"
        assert entrypoint.pinned_model(entry, row) == "openai/gpt-6-luna"

    def test_the_key_path_is_unchanged(self):
        assert entrypoint.pinned_model(self._entry("codex")) == \
            "deepseek/deepseek-v4.1-flash"

    def test_a_host_without_a_pin_keeps_its_openrouter_slug(self):
        entry = self._entry("opencode")
        assert entrypoint.pinned_model(entry, None, "fallback") == \
            entry["model"]["slug"]

    def test_the_operator_override_still_wins(self, monkeypatch):
        monkeypatch.setenv("MLLM_HOST_MODEL", "gpt-6-sol")
        assert entrypoint.pinned_model(self._entry("codex"), None,
                                       "fallback") == "gpt-6-sol"

    def test_the_effort_rides_with_the_pin(self):
        assert entrypoint.subscription_model(self._entry("codex")) == \
            ("gpt-6-luna", "high")

    def test_a_malformed_effort_is_refused(self):
        entry = {"subscription": {"model": {"slug": "x", "effort": 'high"\nevil'}}}
        with pytest.raises(SystemExit):
            entrypoint.subscription_model(entry)

    def test_the_host_argv_names_the_subscription_model(self):
        entry = self._entry("codex")
        row = next(r for r in entry["rows"] if r.get("args"))
        argv = entrypoint.build_host_argv(entry, row, "fallback")
        assert argv[argv.index("--model") + 1] == "gpt-6-luna"
        argv = entrypoint.build_host_argv(entry, row)
        assert argv[argv.index("--model") + 1] == "deepseek/deepseek-v4.1-flash"


class TestTheCodexConfigCarriesTheEffort:
    def test_the_effort_is_a_root_key(self):
        """After the `[projects."…"]` header it would belong to that table."""
        config = tomllib.loads(entrypoint._codex_fallback_config("high"))
        assert config["model_reasoning_effort"] == "high"
        assert config["projects"][str(entrypoint.WORKSPACE)]["trust_level"] == "trusted"

    def test_no_effort_writes_no_key(self):
        config = tomllib.loads(entrypoint._codex_fallback_config(""))
        assert "model_reasoning_effort" not in config


class TestTheRecordNamesTheSubscriptionModel:
    @pytest.fixture(autouse=True)
    def _no_daemon(self, monkeypatch):
        monkeypatch.setattr(harness, "cli_version",
                            lambda cli, image=harness.DEFAULT_IMAGE: None)

    def _record(self, host: str, row_id: str, creds: str, out_dir: Path) -> dict:
        row = harness.Row(row_id=row_id, lane="host", host=host, creds=creds)
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        return harness.row_record(row, result)

    def test_a_codex_subscription_row_records_luna_high(self, tmp_path):
        record = self._record("codex", "codex-ask", "fallback", tmp_path)
        assert record.get("model") == "gpt-6-luna"
        assert record.get("effort") == "high"

    def test_the_record_schema_admits_the_effort(self):
        """`rows[]` is `additionalProperties: false`; an unlisted key fails it."""
        rows = harness.load_record_schema()["properties"]["rows"]["items"]
        assert rows["additionalProperties"] is False
        assert "effort" in rows["properties"]

    def test_a_key_row_records_the_openrouter_pin(self, tmp_path):
        record = self._record("codex", "codex-ask", "openrouter", tmp_path)
        assert record.get("model") == "deepseek/deepseek-v4.1-flash"
        assert "effort" not in record


class TestNoSubscriptionValueSurvivesTheScrub:
    """Both by-value scrubs reach nested tokens and the claims inside a JWT."""

    EMAIL = "dev@ex.io"          # under 20 characters, on purpose
    ACCOUNT = "acct-0123456789abcdef0123"

    def _codex_auth(self) -> dict:
        return {"OPENAI_API_KEY": None,
                "tokens": {"id_token": _fake_jwt({"email": self.EMAIL,
                                                  "sub": self.ACCOUNT}),
                           "access_token": "at-" + "x" * 40,
                           "refresh_token": "rt-" + "y" * 40,
                           "account_id": self.ACCOUNT},
                "last_refresh": "2026-10-02T00:00:00Z"}

    def test_the_host_lane_scrub(self, tmp_path):
        path = tmp_path / "auth.json"
        path.write_text(json.dumps(self._codex_auth()), encoding="utf-8")
        secrets = entrypoint.fallback_secret_values([path])
        for value in (self.EMAIL, self.ACCOUNT, "at-" + "x" * 40, "rt-" + "y" * 40):
            assert value in secrets
        assert secrets == sorted(secrets, key=len, reverse=True)

    def test_the_live_lane_scrub(self, monkeypatch):
        monkeypatch.syspath_prepend(str(harness.HERE))
        monkeypatch.setattr(sys, "path", list(sys.path))  # undo the module's insert
        live_provider = importlib.import_module("live_provider")
        material = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-" + "a" * 40,
                                                 "refreshToken": "sk-ant-" + "b" * 40,
                                                 "subscriptionType": "max"}})
        secrets = live_provider.credential_strings(material)
        assert "sk-ant-" + "a" * 40 in secrets
        assert "sk-ant-" + "b" * 40 in secrets
        codex = live_provider.credential_strings(json.dumps(self._codex_auth()))
        assert self.EMAIL in codex and self.ACCOUNT in codex
        text = f"logged in as {self.EMAIL} ({self.ACCOUNT})"
        redacted = live_provider.redact(text, [material, *codex])
        assert self.EMAIL not in redacted and self.ACCOUNT not in redacted


class TestTheLiveLaneUsesTheSubscription:
    @pytest.fixture
    def live(self, monkeypatch):
        monkeypatch.delenv(harness.LIVE_CREDS_ENV, raising=False)
        return importlib.import_module("tests.test_container_live_provider")

    def test_a_present_login_selects_the_subscription_variant(self, live,
                                                              monkeypatch):
        monkeypatch.setattr(harness, "subscription_available", lambda cli: True)
        monkeypatch.setattr(harness, "subscription_sources",
                            lambda cli: [Path(f"/home/dev/{cli}.json")])
        codex = live._subscription_variant("codex")
        assert codex["model"] == "gpt-6-luna:high"
        assert codex["credential_file"] == ".codex/auth.json"
        claude = live._subscription_variant("claude-code")
        assert claude["model"] == "sonnet"
        assert "ANTHROPIC_BASE_URL" in claude["env_drop"]

    def test_opencode_never_takes_one(self, live, monkeypatch):
        monkeypatch.setattr(harness, "subscription_available", lambda cli: True)
        assert live._subscription_variant("opencode") is None

    def test_the_key_can_be_forced(self, live, monkeypatch):
        monkeypatch.setattr(harness, "subscription_available", lambda cli: True)
        monkeypatch.setenv(harness.LIVE_CREDS_ENV, "openrouter")
        assert live._subscription_variant("codex") is None

    def test_the_openrouter_fallbacks_follow_the_policy(self, live):
        """Never Haiku through OpenRouter; v4.1-flash rather than the 0731 build."""
        assert live.LIVE_MODEL_OVERRIDES["claude-code"]["slug"] == "openai/gpt-6-luna"
        assert live.LIVE_MODEL_OVERRIDES["codex"]["slug"] == \
            "deepseek/deepseek-v4.1-flash"


class TestAKeylessSubscriptionRunIsNotRefusedAsEmpty:
    """`make e2e-container-host` no longer demands the key (2026-10-02).

    opencode is the first scheduled host and has no subscription, so it skips
    BEFORE any codex row has run; the old `_refuse_empty_lane`, called at that
    skip, read it as an empty lane and killed the run. Its replacement, a
    prediction over the declared tier, then let a run that executed nothing
    exit 0 (adversarial review, 2026-10-02). The verdict is now taken once,
    after the last row, from what ran. Offline: the real module, a fake config.
    """

    class _Config:
        def __init__(self, expression, keyword=""):
            self.expression = expression
            self.keyword = keyword

        def getoption(self, name, default=None):
            return {"markexpr": self.expression, "keyword": self.keyword,
                    "allow_empty_host_lane": False}.get(name, default)

    @pytest.fixture
    def lane(self, monkeypatch, tmp_path):
        lane = importlib.import_module("tests.test_container_host_lane")
        monkeypatch.setattr(harness, "RUNS", tmp_path)
        monkeypatch.setattr(lane._Census, "ran", [])
        monkeypatch.setattr(lane._Census, "skipped", [])
        monkeypatch.delenv(harness.HOST_CREDS_ENV, raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        return lane

    PROVIDER = "container and live and not live_native_phase and not live_next_command"
    UTC = "20261002T000000Z"

    def test_a_skip_never_refuses_the_lane(self, lane, monkeypatch, tmp_path):
        """The f4d2382 regression: opencode skips first, a codex row runs after."""
        monkeypatch.setattr(harness, "subscription_available",
                            lambda host: host in ("codex", "claude-code"))
        config = self._Config(self.PROVIDER)
        with pytest.raises(pytest.skip.Exception):
            lane._skip_unkeyed_row(config, self.UTC, "opencode-ask", "opencode")
        census = json.loads(
            (tmp_path / self.UTC / "host-lane-census.json").read_text(encoding="utf-8"))
        assert [row["row"] for row in census["skipped"]] == ["opencode-ask"]
        lane._Census.record_ran("codex-ask", "codex", "fallback")
        lane._refuse_empty_lane(config, self.UTC)                 # must not exit

    def test_a_per_row_selector_that_ran_nothing_is_not_run(self, lane, monkeypatch):
        """`-k '[opencode-ask]'` keyless with codex's subscription present.

        The tier's codex rows were never collected, so the opencode skip was the
        whole lane. The tier-wide prediction said a codex row could still run and
        the lane exited 0; before f4d2382 the same run exited 1.
        """
        monkeypatch.setattr(harness, "subscription_available",
                            lambda host: host in ("codex", "claude-code"))
        config = self._Config("container and live", keyword="[opencode-ask]")
        with pytest.raises(pytest.skip.Exception):
            lane._skip_unkeyed_row(config, self.UTC, "opencode-ask", "opencode")
        with pytest.raises(pytest.exit.Exception) as exited:
            lane._refuse_empty_lane(config, self.UTC)
        assert exited.value.returncode == 1
        assert "NOT-RUN" in str(exited.value)

    def test_a_forced_fallback_is_no_credential(self, lane, monkeypatch):
        """`MLLM_E2E_HOST_CREDS=fallback` with no credential file is not a row that
        ran: the verdict reads the census, never the credential mode."""
        monkeypatch.setattr(harness, "subscription_available", lambda host: False)
        monkeypatch.setenv(harness.HOST_CREDS_ENV, "fallback")
        lane._Census.record_skipped("codex-ask", "codex", "refused for the test")
        with pytest.raises(pytest.exit.Exception):
            lane._refuse_empty_lane(self._Config(self.PROVIDER), self.UTC)

    def test_no_subscription_and_no_key_is_still_not_run(self, lane, monkeypatch):
        monkeypatch.setattr(harness, "subscription_available", lambda host: False)
        config = self._Config(self.PROVIDER)
        for host, row in (("opencode", "opencode-ask"), ("codex", "codex-ask")):
            with pytest.raises(pytest.skip.Exception):
                lane._skip_unkeyed_row(config, self.UTC, row, host)
        with pytest.raises(pytest.exit.Exception):
            lane._refuse_empty_lane(config, self.UTC)

    def test_the_operator_can_still_accept_an_empty_lane(self, lane):
        config = self._Config(self.PROVIDER)
        config.getoption = lambda name, default=None: (
            True if name == "allow_empty_host_lane" else
            self._Config(self.PROVIDER).getoption(name, default))
        lane._Census.record_skipped("opencode-ask", "opencode", "OPENROUTER_API_KEY unset")
        lane._refuse_empty_lane(config, self.UTC)                 # must not exit

    def test_a_run_with_no_credentialed_row_selected_is_not_judged(self, lane, tmp_path):
        """The free tier (`make e2e-container-host-discovery`) selects none."""
        lane._refuse_empty_lane(self._Config("container and not live"), self.UTC)
        assert not (tmp_path / self.UTC).exists()

    def test_the_verdict_runs_at_the_modules_teardown(self, lane, monkeypatch):
        """Wired as a module-scoped autouse fixture whose teardown refuses."""
        fixture = lane._host_lane_verdict
        marker = fixture._fixture_function_marker
        assert (marker.scope, marker.autouse) == ("module", True)

        class _Request:
            config = self._Config(self.PROVIDER)

        verdict = fixture._get_wrapped_function()(_Request(), self.UTC)
        next(verdict)
        lane._Census.record_skipped("opencode-ask", "opencode", "OPENROUTER_API_KEY unset")
        with pytest.raises(pytest.exit.Exception):
            next(verdict, None)

    def test_the_makefile_gate_no_longer_exits(self):
        makefile = (Path(__file__).resolve().parents[3] / "Makefile").read_text(
            encoding="utf-8")
        body = makefile.split("\ne2e-container-host: e2e-image\n", 1)[1].split(
            "\n\n", 1)[0]
        assert "exit 1" not in body, "a subscription-only run must be allowed"
        assert "OPENROUTER_API_KEY" in body
