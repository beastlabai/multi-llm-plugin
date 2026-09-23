"""The host lane's spend guard, offline: no Docker, no key, no network, $0.

`tests/container/host_spend.py` is what `test_container_host_lane.py`'s
credentialed fixture enforces spend with. Every test here drives it with a fake
usage source, so the three behaviours the guard exists for are proven in plain
CI rather than discovered on a paid run:

* a row whose spend crosses its ceiling is STOPPED mid-row, as `budget-exceeded`;
* a usage readback that fails is a refusal, never a $0 credit (fail closed);
* the between-rows lane total still refuses the next row.

Plus the matrix guard: every credentialed row declares a numeric
`cost_budget_usd`, because a row without one has nothing to stop it but the lane
total — which is what "documentation, not a cap" meant.
"""

from __future__ import annotations

import functools
import threading
import urllib.error
from pathlib import Path

import pytest
import yaml

from .container import host_spend

HOST_MATRIX = Path(host_spend.__file__).parent / "host_matrix.yaml"

#: Readbacks in these tests never sleep between retries.
no_sleep_read = functools.partial(host_spend.read_usage, sleep=lambda _s: None)


def _rows():
    data = yaml.safe_load(HOST_MATRIX.read_text(encoding="utf-8"))
    return [(host, row) for host, entry in sorted(data["hosts"].items())
            for row in (entry.get("rows") or [])]


class _Source:
    """A fake key-usage endpoint: a script of values and exceptions."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


class _Stopper:
    def __init__(self):
        self.calls = 0
        self.event = threading.Event()

    def __call__(self):
        self.calls += 1
        self.event.set()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("MLLM_E2E_BUDGET_USD", "MLLM_E2E_ROW_BUDGET_HEADROOM",
                 "MLLM_E2E_SPEND_POLL_S"):
        monkeypatch.delenv(name, raising=False)


class TestEveryCredentialedRowDeclaresABudget:
    """The plain-CI half of "make `cost_budget_usd` real"."""

    def test_the_matrix_has_credentialed_rows(self):
        assert _rows(), "host_matrix.yaml declares no rows; this guard guards nothing"

    @pytest.mark.parametrize("host,row", _rows(),
                             ids=[row["id"] for _h, row in _rows()])
    def test_row_declares_a_positive_numeric_cost_budget_usd(self, host, row):
        # Every row under `hosts.*.rows` runs with the OpenRouter key: the
        # credential-free tier is the separate `discovery` fixture, which builds
        # its own rows and is not in this table. So there is no exempt row.
        assert host_spend.declared_budget(row) is not None, (
            f"{host}/{row['id']}: `cost_budget_usd` is "
            f"{row.get('cost_budget_usd')!r}. Every credentialed row must declare "
            f"a positive number — the host lane stops the row mid-run at "
            f"{host_spend.DEFAULT_ROW_HEADROOM}x it, and refuses to start a row "
            f"with none. For an unmeasured row say so beside a provisional value.")
        assert host_spend.row_ceiling(row) >= host_spend.MIN_CEILING_USD


class TestDeclaredBudget:
    @pytest.mark.parametrize("value", [None, "0.60", True, 0, -1.0])
    def test_non_numeric_or_non_positive_is_no_budget(self, value):
        assert host_spend.declared_budget({"id": "r", "cost_budget_usd": value}) is None
        with pytest.raises(ValueError, match="cost_budget_usd"):
            host_spend.row_ceiling({"id": "r", "cost_budget_usd": value})

    def test_ceiling_is_budget_times_headroom(self, monkeypatch):
        assert host_spend.row_ceiling({"cost_budget_usd": 0.92}) == pytest.approx(1.84)
        monkeypatch.setenv("MLLM_E2E_ROW_BUDGET_HEADROOM", "1.5")
        assert host_spend.row_ceiling({"cost_budget_usd": 1.0}) == pytest.approx(1.5)

    def test_ceiling_never_drops_below_the_settling_floor(self):
        assert host_spend.row_ceiling({"cost_budget_usd": 0.01}) == host_spend.MIN_CEILING_USD

    def test_headroom_below_one_is_refused(self, monkeypatch):
        monkeypatch.setenv("MLLM_E2E_ROW_BUDGET_HEADROOM", "0.5")
        with pytest.raises(ValueError, match="below 1"):
            host_spend.row_ceiling({"cost_budget_usd": 1.0})


class TestTheReadbackFailsClosed:
    def test_a_transient_error_is_retried_then_read(self):
        source = _Source(urllib.error.URLError("reset"), 12.5)
        slept = []
        assert host_spend.read_usage(source, sleep=slept.append) == 12.5
        assert source.calls == 2
        assert slept == [host_spend.READ_BACKOFF_S[0]]

    def test_a_persistent_network_error_raises_after_the_bounded_retry(self):
        source = _Source(urllib.error.URLError("no route to host"))
        with pytest.raises(host_spend.SpendUnobservable, match="3 attempt"):
            no_sleep_read(source)
        assert source.calls == host_spend.READ_ATTEMPTS

    @pytest.mark.parametrize("bad", [None, "12.5x", KeyError("usage")])
    def test_a_parse_error_is_a_failed_read_not_zero(self, bad):
        source = _Source(bad)
        with pytest.raises(host_spend.SpendUnobservable):
            no_sleep_read(source)

    def test_the_real_source_fails_closed_on_a_network_error(self, monkeypatch):
        monkeypatch.setenv(host_spend.KEY_ENV, "sk-or-v1-" + "f" * 48)

        def refuse(*_a, **_k):
            raise urllib.error.URLError("Temporary failure in name resolution")

        monkeypatch.setattr("urllib.request.urlopen", refuse)
        with pytest.raises(host_spend.SpendUnobservable, match="URLError"):
            no_sleep_read()

    def test_the_real_source_refuses_a_null_usage(self, monkeypatch):
        import io
        import json

        monkeypatch.setenv(host_spend.KEY_ENV, "sk-or-v1-" + "f" * 48)

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *_a, **_k: _Response(json.dumps({"data": {"usage": None}}).encode()))
        with pytest.raises(host_spend.SpendUnobservable, match="not a number"):
            no_sleep_read()

    def test_the_key_never_reaches_the_message(self, monkeypatch):
        key = "sk-or-v1-" + "a1" * 24
        monkeypatch.setenv(host_spend.KEY_ENV, key)
        source = _Source(RuntimeError(f"echoed Authorization: Bearer {key}"))
        with pytest.raises(host_spend.SpendUnobservable) as caught:
            no_sleep_read(source)
        assert key not in str(caught.value)
        assert "<redacted>" in str(caught.value)


class TestTheMidRowStop:
    def _watch(self, source, stopper, ceiling=1.0, bound="row", before=10.0):
        return host_spend.RowWatch(before=before, ceiling_usd=ceiling, bound=bound,
                                   stop_row=stopper, source=source,
                                   interval_s=0.01, read=no_sleep_read)

    def test_spend_under_the_ceiling_does_not_stop_the_row(self):
        stopper = _Stopper()
        watch = self._watch(_Source(10.4), stopper)
        assert watch.poll_once() is False
        assert watch.tripped is None and stopper.calls == 0
        assert watch.peak_usd == pytest.approx(0.4)

    def test_an_overshoot_stops_the_row_as_budget_exceeded(self):
        stopper = _Stopper()
        watch = self._watch(_Source(10.5, 11.2), stopper)
        assert watch.poll_once() is False
        assert watch.poll_once() is True
        assert stopper.calls == 1
        assert watch.tripped["outcome"] == host_spend.BUDGET_EXCEEDED
        assert watch.tripped["bound"] == "row"
        assert watch.tripped["spent_at_trip_usd"] == pytest.approx(1.2)

    def test_the_background_poller_stops_a_running_row_once(self):
        stopper = _Stopper()
        source = _Source(10.1, 10.3, 10.9, 12.0, 13.0)
        with self._watch(source, stopper) as watch:
            assert stopper.event.wait(5), "the poller never stopped the row"
        assert stopper.calls == 1
        assert watch.tripped["outcome"] == host_spend.BUDGET_EXCEEDED

    def test_a_readback_that_goes_dark_mid_row_stops_it_too(self):
        stopper = _Stopper()
        watch = self._watch(_Source(urllib.error.URLError("down")), stopper)
        assert watch.poll_once() is True
        assert stopper.calls == 1
        assert watch.tripped["outcome"] == host_spend.SPEND_UNOBSERVABLE
        assert watch.tripped["spent_at_trip_usd"] is None

    def test_the_ceiling_is_the_lower_of_row_and_lane(self):
        row = {"id": "r", "cost_budget_usd": 0.92}  # row ceiling $1.84
        roomy = host_spend.Ledger(budget_usd=6.0)
        watch = host_spend.RowWatch.for_row(row, roomy, 0.0, stop_row=_Stopper())
        assert watch.bound == "row" and watch.ceiling_usd == pytest.approx(1.84)

        tight = host_spend.Ledger(budget_usd=2.0)
        tight.total = 1.5
        stopper = _Stopper()
        watch = host_spend.RowWatch.for_row(row, tight, 40.0, stop_row=stopper,
                                            source=_Source(40.6), read=no_sleep_read,
                                            interval_s=0.01)
        assert watch.bound == "lane" and watch.ceiling_usd == pytest.approx(0.5)
        assert watch.poll_once() is True
        assert watch.tripped["bound"] == "lane" and stopper.calls == 1

    def test_a_row_with_no_budget_is_refused_before_it_starts(self):
        with pytest.raises(ValueError):
            host_spend.RowWatch.for_row({"id": "r"}, host_spend.Ledger(2.0), 0.0,
                                        stop_row=_Stopper())


class TestTheBetweenRowsLaneCap:
    def test_under_budget_the_next_row_may_start(self):
        ledger = host_spend.Ledger(budget_usd=2.0)
        ledger.charge(10.0, 11.2)
        assert ledger.total == pytest.approx(1.2)
        assert ledger.refusal() is None

    def test_reaching_the_lane_total_refuses_the_next_row(self):
        ledger = host_spend.Ledger(budget_usd=2.0)
        ledger.charge(10.0, 11.2)
        ledger.charge(11.2, 12.3)
        assert "MLLM_E2E_BUDGET_USD=$2.00" in ledger.refusal()

    def test_an_unobservable_closing_read_refuses_the_next_row(self):
        ledger = host_spend.Ledger(budget_usd=2.0)
        ledger.unobservable = "codex-ask's closing readback: URLError"
        assert "UNOBSERVABLE" in ledger.refusal()

    def test_a_row_stopped_on_the_lane_halts_it_whatever_the_lagging_total_says(self):
        ledger = host_spend.Ledger(budget_usd=2.0)
        ledger.charge(10.0, 10.3)  # the endpoint has not settled the overshoot yet
        ledger.halted = "opencode-full was stopped mid-row"
        assert "HALTED" in ledger.refusal()

    def test_the_lane_budget_is_read_from_the_environment(self, monkeypatch):
        assert host_spend.Ledger().budget_usd == host_spend.DEFAULT_LANE_BUDGET_USD
        monkeypatch.setenv("MLLM_E2E_BUDGET_USD", "6.00")
        assert host_spend.Ledger().budget_usd == 6.0
