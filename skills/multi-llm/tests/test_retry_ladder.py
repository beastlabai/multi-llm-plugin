"""M9 (§7.1.b, §7.5) — the one retry policy both lanes obey.

The ladder is pure, so every rule is testable against an injected clock with no
subprocess, no sleeping and no event loop. §14.2's M9 row names four checks and
they are the four classes below:

* the budget is **monotonic** — no fresh full timeout per retry (today's Lane A
  recursion resets it);
* a slot whose declared vendor envelope exceeds the remaining budget is
  ``budget_exhausted`` with **no** spawn;
* rule 6(b) — a ``retries_rate_limit: true`` adapter returning ``rate_limited``
  yields **no** identical re-dispatch;
* a forever-429 provider finishes inside ``resolved_timeout``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.retry_ladder import (  # noqa: E402
    MAX_RATE_LIMIT_BACKOFFS,
    MIN_ATTEMPT_BUDGET_S,
    REPAIR_SCHEMA_INVALID,
    RetryLadder,
    STATUS_AUTH_FAILED,
    STATUS_INVALID_MODEL,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_REJECTED,
    STATUS_RUNNER_ERROR,
    STATUS_SANDBOX_UNAVAILABLE,
    STATUS_SCHEMA_INVALID,
    STATUS_SPAWN_FAILED,
    STATUS_TIMEOUT,
    STATUS_UNPARSEABLE,
    STOP_BUDGET,
    STOP_DEADLINE,
    STOP_EXHAUSTED_ATTEMPTS,
    STOP_FATAL,
    STOP_TERMINAL,
    STOP_VENDOR_ALREADY_RETRIED,
    backoff_delay,
    budget_exhausted_reason,
)


class Clock:
    """An injected monotonic clock the test advances by hand."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _envelope(max_requests=1, max_wall_s=None, retries_rate_limit=False):
    """A stand-in for an adapter's ``internal_retries`` (§7.6(b)).

    Structural on purpose — the ladder must stay free of provider imports, and
    a test that constructed the real dataclass would hide a coupling the design
    deliberately does not have.
    """
    return type(
        "Envelope",
        (),
        {
            "max_requests": max_requests,
            "max_wall_s": max_wall_s,
            "retries_rate_limit": retries_rate_limit,
        },
    )()


def _ladder(clock, timeout=1200.0, retries=1, envelope=None):
    return RetryLadder(
        timeout=timeout,
        retries=retries,
        envelope=envelope,
        now=clock,
        jitter=lambda: 0.0,  # deterministic: equal-jitter's random half is 0
    )


class TestTheBudgetIsMonotonic:
    """§7.5 rule 1 — one deadline per slot, taken once.

    This is the defect Lane A's recursion has today: ``run_single_model``
    re-enters itself with the *same* ``timeout`` argument, so a claude-code slot
    at ``default_timeout: 1800`` could run 4 × 1800 s + 35 s of backoff ≈ 2 h
    against a configured ceiling of 1800 s.
    """

    def test_each_attempt_is_spawned_with_what_is_left(self):
        clock = Clock()
        ladder = _ladder(clock, timeout=100.0, retries=2)

        slices = []
        while (slice_s := ladder.next_attempt()) is not None:
            slices.append(round(slice_s, 3))
            clock.advance(30.0)  # the attempt really burned 30 s
            action = ladder.record(STATUS_TIMEOUT)
            clock.advance(action.delay)

        # Not [100, 100, 100] — which is exactly what the recursion produces.
        assert slices == [100.0, 70.0, 40.0]

    def test_the_slot_never_outlives_its_timeout(self):
        clock = Clock()
        ladder = _ladder(clock, timeout=60.0, retries=5)
        while (slice_s := ladder.next_attempt()) is not None:
            clock.advance(slice_s)  # every attempt runs to its full slice
            action = ladder.record(STATUS_TIMEOUT)
            clock.advance(action.delay)
        assert ladder.elapsed() <= 60.0
        assert ladder.stop_reason == STOP_DEADLINE

    def test_backoff_sleeps_are_charged_to_the_same_deadline(self):
        """§7.5 rule 3 — the sleeps are inside the budget, not beside it."""
        clock = Clock()
        ladder = _ladder(clock, timeout=10.0, retries=0)
        total_slept = 0.0
        while ladder.next_attempt() is not None:
            clock.advance(0.1)
            action = ladder.record(STATUS_RATE_LIMITED)
            total_slept += action.delay
            clock.advance(action.delay)
        assert ladder.elapsed() <= 10.0
        assert total_slept > 0.0


class TestBudgetExhaustedRefusesToSpawn:
    """§7.5 rule 4 — vendor-internal retries are declared, not discovered."""

    def test_a_declared_envelope_larger_than_the_budget_yields_no_spawn(self):
        clock = Clock()
        # claude-code at a 600 s slot: 4 × API_TIMEOUT_MS = 480 s… but give it a
        # 1200 s envelope against a 600 s slot, which cannot fit even once.
        ladder = _ladder(
            clock, timeout=600.0, retries=1, envelope=_envelope(max_wall_s=1200.0)
        )
        assert ladder.next_attempt() is None
        assert ladder.stop_reason == STOP_BUDGET
        assert ladder.spawns == 0, "rule 4 is a pre-dispatch admission test"

    def test_muses_real_985_s_envelope_refuses_a_slot_that_cannot_hold_it(self):
        """Rule 4 re-run against the **shipped** envelope, not the synthetic one.

        ``_envelope`` above is structural on purpose — the ladder must stay free
        of provider imports — but that makes it the wrong witness for one
        question: does the number an adapter really declares compose with this
        rule? muse's does. It declares ``max_wall_s = 985.0``, the measured
        wall of four rounds of its ten-attempt provider ladder at 1.4.3
        (``tests/fixtures/muse/``'s ``truncated`` row, 40 provider requests,
        985.32 s), and that is larger than a 900 s slot. So the admission test
        refuses before anything is spawned, rather than a child being killed
        part-way through a sixteen-minute chain the harness cannot see.
        """
        from utils.providers.muse import MuseProvider

        envelope = MuseProvider().internal_retries
        assert envelope is not None and envelope.max_wall_s == 985.0

        clock = Clock()
        ladder = _ladder(clock, timeout=900.0, retries=1, envelope=envelope)

        assert ladder.next_attempt() is None
        assert ladder.stop_reason == STOP_BUDGET
        assert ladder.spawns == 0, "rule 4 is a pre-dispatch admission test"

    def test_muses_second_attempt_after_a_985_s_chain_is_refused_with_a_reason(self):
        """The review-24 repro: providers.yaml's 1800 s muse slot, two retries.

        Attempt 1 runs muse's whole measured chain (985 s at 1.4.3) and comes
        back a timeout. 815 s are left. With the stale 736 s envelope the ladder
        admitted attempt 2 into those 815 s — a child certain to be killed
        part-way through the same 985 s chain, which is exactly what rule 4
        exists to refuse. Declared at the measured 985 s, the retry is refused
        and the reason says why.
        """
        from utils.providers.muse import MuseProvider

        clock = Clock()
        ladder = _ladder(
            clock, timeout=1800.0, retries=2,
            envelope=MuseProvider().internal_retries,
        )
        assert ladder.next_attempt() == pytest.approx(1800.0)
        clock.advance(985.0)
        action = ladder.record(STATUS_TIMEOUT)

        assert action.retry is False
        assert ladder.stop_reason == STOP_BUDGET
        assert "985s one dispatch of this runner declares" in action.reason
        assert "2 retries unused" in action.reason
        assert ladder.next_attempt() is None
        assert ladder.spawns == 1

    def test_muses_real_envelope_still_admits_a_slot_that_can_hold_it(self):
        """The other direction, so 985 is a bound and not a blanket refusal."""
        from utils.providers.muse import MuseProvider

        clock = Clock()
        ladder = _ladder(
            clock,
            timeout=1800.0,
            retries=1,
            envelope=MuseProvider().internal_retries,
        )

        assert ladder.next_attempt() == pytest.approx(1800.0)
        assert ladder.spawns == 1

    def test_gooses_real_envelope_declares_no_wall_and_restricts_nothing(self):
        """goose's ~7 s is transport against a stub, so it declares ``None``.

        Rule 6(c) then performs no admission test at all — the honest posture
        the dataclass documents, because inventing a bound from a localhost
        round trip "would refuse attempts that had budget for themselves".
        """
        from utils.providers.goose import GooseProvider

        envelope = GooseProvider().internal_retries
        assert envelope is not None and envelope.max_wall_s is None

        clock = Clock()
        ladder = _ladder(clock, timeout=10.0, retries=1, envelope=envelope)

        assert ladder.next_attempt() == pytest.approx(10.0)

    def test_it_stops_before_a_LATER_attempt_that_would_not_fit(self):
        clock = Clock()
        ladder = _ladder(
            clock, timeout=1000.0, retries=3, envelope=_envelope(max_wall_s=400.0)
        )
        spawns = 0
        while ladder.next_attempt() is not None:
            spawns += 1
            clock.advance(350.0)
            ladder.record(STATUS_TIMEOUT)
        # 1000 → 650 → 300: the third attempt cannot hold a 400 s worst case.
        assert spawns == 2
        assert ladder.stop_reason == STOP_BUDGET

    def test_an_undeclared_envelope_does_not_restrict_anything(self):
        """Rule 6(c) — ``None`` means "assume one request", never "assume the worst"."""
        clock = Clock()
        ladder = _ladder(clock, timeout=10.0, retries=1, envelope=None)
        assert ladder.next_attempt() == pytest.approx(10.0)

    def test_rule_4_is_an_admission_test_not_a_second_debit(self):
        """Rule 6(a) — the inner wall is already counted; do not count it twice."""
        clock = Clock()
        ladder = _ladder(
            clock, timeout=100.0, retries=1, envelope=_envelope(max_wall_s=10.0)
        )
        assert ladder.next_attempt() == pytest.approx(100.0)
        clock.advance(30.0)  # the attempt spent 30 s, vendor retries included
        ladder.record(STATUS_TIMEOUT)
        # 70 left, not 70 − 10: the envelope is never subtracted from the clock.
        assert ladder.next_attempt() == pytest.approx(70.0)


class TestRule6bNoIdenticalRateLimitRedispatch:
    """§7.5 rule 6(b) — the two mechanisms compose; they must not multiply."""

    def test_unknown_request_bound_still_prevents_identical_redispatch(self):
        ladder = _ladder(
            Clock(), envelope=_envelope(max_requests=None, retries_rate_limit=True)
        )
        assert ladder.next_attempt() is not None
        action = ladder.record(STATUS_RATE_LIMITED)
        assert not action.retry
        assert action.delay == 0
        assert ladder.stop_reason == STOP_VENDOR_ALREADY_RETRIED
        assert "no finite request bound is known" in action.reason
        assert "None requests" not in action.reason
        assert ladder.next_attempt() is None
        assert ladder.spawns == 1

    def test_an_adapter_that_retries_429s_internally_is_not_re_dispatched(self):
        clock = Clock()
        ladder = _ladder(
            clock,
            timeout=1800.0,
            retries=1,
            envelope=_envelope(max_requests=4, retries_rate_limit=True),
        )
        assert ladder.next_attempt() is not None
        action = ladder.record(STATUS_RATE_LIMITED)

        assert action.retry is False
        assert action.delay == 0.0
        assert ladder.stop_reason == STOP_VENDOR_ALREADY_RETRIED
        # The reason names the declared envelope, per §7.5 rule 6(b).
        assert "4" in action.reason
        assert ladder.next_attempt() is None
        assert ladder.spawns == 1, "≤ 3 backoffs would have made this 4 requests → 20"

    @pytest.mark.parametrize(
        "provider_module,provider_class,max_requests",
        [
            ("utils.providers.muse", "MuseProvider", 40),
            ("utils.providers.gemini", "GeminiProvider", 10),
            # goose 1.53.0: 9 requests of 429 over 7.5 s (8 at 1.41.0).
            ("utils.providers.goose", "GooseProvider", 9),
            # T7.14: pi 0.87.1, 4 requests of 429 over 14.4 s
            # (tests/fixtures/pi/'s ``rate-limit`` row, three auto_retry_start
            # events). omp declared its envelope before this table grew it a
            # row; six attempts against a 429 (tests/fixtures/omp/'s
            # ``rate-limit-responses``).
            ("utils.providers.pi", "PiProvider", 4),
            ("utils.providers.omp", "OmpProvider", 6),
            # grok 1.0.46 with `models.max_retries` = 2 pinned (review 24): one
            # 429 retry, 11 requests on its worst (5xx) chain.
            ("utils.providers.grok", "GrokProvider", 11),
        ],
    )
    def test_the_real_declared_envelopes_get_no_second_dispatch(
        self, provider_module, provider_class, max_requests
    ):
        """Rule 6(b) re-run against the **shipped** envelopes (C12).

        All three CLIs were measured retrying a fail-forever 429 themselves —
        muse ten provider attempts over 248 s (`tests/fixtures/muse/`'s
        ``rate-limit`` row, the 5xx row's envelope to two decimals), gemini ten
        requests over 243 s (``tests/fixtures/gemini/``'s ``rate-limit`` row:
        nine ``Attempt N failed with status 429`` lines and a tenth that gives
        up), goose four attempts at the real turn inside 9 requests / 7.5 s
        (``tests/fixtures/goose/``'s ``rate-limit`` row, 1.53.0; 8 / 7.2 s at
        1.41.0). Declaring that removes
        the harness's own rate-limit backoff for them: a **smaller** retry
        budget, not a larger one. Before the declaration a rate-limited slot got
        up to ``MAX_RATE_LIMIT_BACKOFFS`` further identical dispatches of a
        chain the CLI had already run — goose's 8 requests turned into 32,
        gemini's four-minute chain and muse's twelve-minute one re-dispatched
        until the slot's deadline ran out — which is exactly the multiplication
        this rule exists to prevent.
        """
        import importlib

        envelope = getattr(
            importlib.import_module(provider_module), provider_class
        )().internal_retries
        assert envelope is not None
        assert envelope.retries_rate_limit is True
        assert envelope.max_requests == max_requests

        clock = Clock()
        ladder = _ladder(clock, timeout=1800.0, retries=1, envelope=envelope)

        assert ladder.next_attempt() is not None
        action = ladder.record(STATUS_RATE_LIMITED)

        assert action.retry is False
        assert action.delay == 0.0
        assert ladder.stop_reason == STOP_VENDOR_ALREADY_RETRIED
        # The reason names the declared envelope, per §7.5 rule 6(b).
        assert str(max_requests) in action.reason
        assert ladder.next_attempt() is None
        assert ladder.spawns == 1, "one dispatch, and no identical re-dispatch"

    def test_an_adapter_that_does_NOT_retry_429s_keeps_the_harness_backoff(self):
        """codex: one request, then ``exceeded retry limit`` — so ≤ 3 applies."""
        clock = Clock()
        ladder = _ladder(
            clock,
            timeout=1800.0,
            retries=1,
            envelope=_envelope(max_requests=30, retries_rate_limit=False),
        )
        spawns = 0
        while ladder.next_attempt() is not None:
            spawns += 1
            clock.advance(1.0)
            action = ladder.record(STATUS_RATE_LIMITED)
            clock.advance(action.delay)
        assert spawns == MAX_RATE_LIMIT_BACKOFFS + 1
        assert ladder.stop_reason == STOP_EXHAUSTED_ATTEMPTS

    def test_rate_limiting_does_not_consume_the_transport_budget(self):
        clock = Clock()
        ladder = _ladder(clock, timeout=1800.0, retries=1)
        ladder.next_attempt()
        ladder.record(STATUS_RATE_LIMITED)
        assert ladder.transport_retries_left == 1, "rung 3 is not a transport failure"


class TestForever429FinishesInsideTheTimeout:
    """§14.2's M9 row: the fake provider that 429s forever."""

    def test_it_terminates_inside_resolved_timeout(self):
        clock = Clock()
        resolved_timeout = 1200.0
        ladder = _ladder(clock, timeout=resolved_timeout, retries=1)

        spawns = 0
        while (slice_s := ladder.next_attempt()) is not None:
            spawns += 1
            assert slice_s <= resolved_timeout
            clock.advance(1.0)
            action = ladder.record(STATUS_RATE_LIMITED)
            clock.advance(action.delay)

        assert spawns <= 5, "§7.5 rule 2's ceiling across both counters"
        assert ladder.elapsed() <= resolved_timeout + MIN_ATTEMPT_BUDGET_S


class TestTerminalAndFatalStatuses:
    """The rungs that are never retried, and the three that fail the batch.

    ``invalid_model`` joined rung 1 with D4: a rejected model id is a property
    of the spec, so retrying it — here or on any sibling resolving the same
    spec — cannot do anything but fail again. ``sandbox_unavailable`` joined
    with D6 on the same reasoning one level down: bubblewrap failing to start
    is a property of the machine, and no sibling on that runner fares better.
    """

    @pytest.mark.parametrize(
        "status", [
            STATUS_SPAWN_FAILED, STATUS_AUTH_FAILED, STATUS_INVALID_MODEL,
            STATUS_SANDBOX_UNAVAILABLE,
        ]
    )
    def test_rung_1_is_fatal(self, status):
        ladder = _ladder(Clock())
        ladder.next_attempt()
        action = ladder.record(status)
        assert action.retry is False
        assert ladder.stop_reason == STOP_FATAL

    @pytest.mark.parametrize("status", [STATUS_SCHEMA_INVALID, STATUS_REJECTED])
    def test_rungs_6_and_7_are_never_retried(self, status):
        ladder = _ladder(Clock(), retries=5)
        ladder.next_attempt()
        action = ladder.record(status)
        assert action.retry is False
        assert ladder.stop_reason == STOP_TERMINAL

    def test_ok_stops_the_ladder(self):
        ladder = _ladder(Clock(), retries=5)
        ladder.next_attempt()
        assert ladder.record(STATUS_OK).retry is False
        assert ladder.next_attempt() is None

    @pytest.mark.parametrize(
        "status", [STATUS_TIMEOUT, STATUS_RUNNER_ERROR, STATUS_UNPARSEABLE]
    )
    def test_transport_statuses_consume_the_retry_budget(self, status):
        ladder = _ladder(Clock(), retries=1)
        ladder.next_attempt()
        assert ladder.record(status).retry is True
        assert ladder.transport_retries_left == 0
        ladder.next_attempt()
        assert ladder.record(status).retry is False
        assert ladder.stop_reason == STOP_EXHAUSTED_ATTEMPTS


class TestBackoffDelay:
    """Equal jitter: never immediate, never longer than the step."""

    def test_it_is_bounded_by_the_step(self):
        for attempt in range(1, 8):
            step = min(2.0 ** attempt, 30.0)
            assert backoff_delay(attempt, jitter=lambda: 0.0) == step / 2.0
            assert backoff_delay(attempt, jitter=lambda: 1.0) == step
            assert 0 < backoff_delay(attempt) <= step

    def test_it_is_capped(self):
        assert backoff_delay(50, jitter=lambda: 1.0) == 30.0


class TestSchemaRepairAllowance:
    """Rule 7 (2026-10-05): a granted schema-repair re-ask, and only that.

    Rung 6 stays terminal for every caller that grants nothing — which is every
    caller but Lane B's ``task_generation`` — and a grant is spent once, on the
    same deadline, without touching the transport or rate-limit counters.
    """

    def test_no_grant_leaves_rung_6_terminal(self):
        ladder = _ladder(Clock(), retries=3)
        ladder.next_attempt()
        action = ladder.record(STATUS_SCHEMA_INVALID)
        assert action.retry is False and action.repair is False
        assert ladder.stop_reason == STOP_TERMINAL
        assert ladder.next_attempt() is None

    def test_one_grant_is_one_re_ask_then_terminal(self):
        ladder = RetryLadder(timeout=1200.0, retries=2, now=Clock(), schema_repairs=1)
        ladder.next_attempt()
        first = ladder.record(STATUS_SCHEMA_INVALID)
        assert first.retry is True and first.repair is True
        assert first.reason == REPAIR_SCHEMA_INVALID
        assert first.delay == 0.0
        assert ladder.schema_repairs_used == 1
        assert ladder.transport_retries_left == 2, "the re-ask spends no retry"
        assert ladder.next_attempt() is not None
        second = ladder.record(STATUS_SCHEMA_INVALID)
        assert second.retry is False and second.repair is False
        assert ladder.stop_reason == STOP_TERMINAL
        assert ladder.spawns == 2

    def test_a_reply_with_no_document_is_not_repairable(self):
        ladder = RetryLadder(timeout=1200.0, retries=2, now=Clock(), schema_repairs=1)
        ladder.next_attempt()
        action = ladder.record(STATUS_SCHEMA_INVALID, repairable=False)
        assert action.retry is False
        assert ladder.schema_repairs_used == 0

    def test_the_re_ask_runs_on_what_is_left_of_the_same_deadline(self):
        clock = Clock()
        ladder = RetryLadder(timeout=100.0, retries=0, now=clock, schema_repairs=1)
        assert ladder.next_attempt() == 100.0
        clock.advance(70.0)
        assert ladder.record(STATUS_SCHEMA_INVALID).repair is True
        assert ladder.next_attempt() == pytest.approx(30.0)

    def test_no_re_ask_when_the_deadline_is_spent(self):
        clock = Clock()
        ladder = RetryLadder(timeout=100.0, retries=0, now=clock, schema_repairs=1)
        ladder.next_attempt()
        clock.advance(100.0 - MIN_ATTEMPT_BUDGET_S / 2)
        action = ladder.record(STATUS_SCHEMA_INVALID)
        assert action.retry is False
        assert ladder.stop_reason == STOP_DEADLINE
        assert "no schema-repair re-ask" in action.reason

    @pytest.mark.parametrize("status", [STATUS_REJECTED, STATUS_UNPARSEABLE])
    def test_the_grant_is_for_rung_6_only(self, status):
        ladder = RetryLadder(timeout=1200.0, retries=0, now=Clock(), schema_repairs=1)
        ladder.next_attempt()
        assert ladder.record(status).repair is False
        assert ladder.schema_repairs_used == 0


class TestAGrantIsNeverTakenBackByAdmission:
    """Fix round 6: rule 4's envelope bound decides a grant, not the next start.

    ``record()`` used to grant a retry, a backoff or the schema-repair re-ask
    on rule 3 alone; ``next_attempt()`` then applied rule 4 and refused, and
    Lane B recorded the slot ``budget_exhausted`` — "Nothing was dispatched" —
    although attempt 1 had run. A grant is now taken against both admission
    rules, so a granted retry is always admitted, and a refusal is the
    attempt's own stop with the reason as a note.
    """

    def _codex_like(self, clock, *, timeout=60.0, retries=2, repairs=1):
        return RetryLadder(
            timeout=timeout, retries=retries, now=clock, schema_repairs=repairs,
            envelope=_envelope(max_requests=4, max_wall_s=25.0),
        )

    def test_the_re_ask_is_refused_when_the_rest_cannot_hold_the_envelope(self):
        clock = Clock()
        ladder = self._codex_like(clock)
        assert ladder.next_attempt() == pytest.approx(60.0)
        clock.advance(40.0)  # 20 s left, under the declared 25 s
        action = ladder.record(STATUS_SCHEMA_INVALID)
        assert action.retry is False and action.repair is False
        assert ladder.schema_repairs_used == 0
        assert ladder.stop_reason == STOP_BUDGET
        assert action.note.startswith("no schema-repair re-ask: ")
        assert "20.0s" in action.note and "25s" in action.note
        assert action.reason == f"{STATUS_SCHEMA_INVALID} ({action.note})"
        assert ladder.next_attempt() is None
        assert ladder.spawns == 1

    def test_the_transport_retry_is_refused_the_same_way(self):
        clock = Clock()
        ladder = self._codex_like(clock)
        ladder.next_attempt()
        clock.advance(40.0)
        action = ladder.record(STATUS_TIMEOUT)
        assert action.retry is False
        assert ladder.stop_reason == STOP_BUDGET
        assert ladder.transport_retries_left == 2, "a refused retry is not spent"
        assert action.note.startswith("no retry: ")
        assert "25s" in action.note and "2 retries unused" in action.note

    def test_the_backoff_is_refused_when_what_follows_it_cannot_hold_it(self):
        clock = Clock()
        ladder = RetryLadder(
            timeout=60.0, retries=0, now=clock, jitter=lambda: 0.0,
            envelope=_envelope(max_requests=4, max_wall_s=25.0),
        )
        ladder.next_attempt()
        clock.advance(34.5)  # 25.5 s left, but not after the 1 s backoff
        action = ladder.record(STATUS_RATE_LIMITED)
        assert action.retry is False and action.delay == 0.0
        assert ladder.stop_reason == STOP_BUDGET
        assert "25s" in action.note

    def test_the_deadline_refusal_carries_its_note_too(self):
        clock = Clock()
        ladder = RetryLadder(timeout=100.0, retries=0, now=clock, schema_repairs=1)
        ladder.next_attempt()
        clock.advance(100.0 - MIN_ATTEMPT_BUDGET_S / 2)
        action = ladder.record(STATUS_SCHEMA_INVALID)
        assert ladder.stop_reason == STOP_DEADLINE
        assert action.note == (
            "no schema-repair re-ask: the per-slot wall-clock budget of 100.0s "
            "is exhausted"
        )

    @pytest.mark.parametrize("status", [
        STATUS_SCHEMA_INVALID, STATUS_TIMEOUT, STATUS_RUNNER_ERROR,
        STATUS_UNPARSEABLE, STATUS_RATE_LIMITED,
    ])
    @pytest.mark.parametrize("spent", [0.0, 10.0, 34.0, 35.5, 40.0, 59.5])
    def test_a_granted_retry_is_always_admitted(self, status, spent):
        clock = Clock()
        ladder = RetryLadder(
            timeout=60.0, retries=2, now=clock, schema_repairs=1,
            jitter=lambda: 0.5, envelope=_envelope(max_requests=4, max_wall_s=25.0),
        )
        ladder.next_attempt()
        clock.advance(spent)
        action = ladder.record(status)
        if action.retry:
            clock.advance(action.delay)
            assert ladder.next_attempt() is not None, (status, spent, action)
        else:
            assert action.note, (status, spent, action)
            assert ladder.next_attempt() is None

    def test_a_refusal_after_attempts_ran_never_says_nothing_was_dispatched(self):
        """The admission refusal can still land after a grant if the clock
        moves in between (a backoff the lane overslept); its sentence must not
        claim a slot that ran attempts dispatched nothing."""
        clock = Clock()
        ladder = self._codex_like(clock)
        ladder.next_attempt()
        clock.advance(10.0)
        assert ladder.record(STATUS_TIMEOUT).retry is True
        clock.advance(30.0)  # overslept: 20 s left
        assert ladder.next_attempt() is None
        assert ladder.stop_reason == STOP_BUDGET
        reason = budget_exhausted_reason(ladder, target="codex:gpt-5.5", timeout=60.0)
        assert "Nothing was dispatched" not in reason
        assert "1 attempt(s) ran" in reason
        assert "attempt 2 was not started" in reason

    def test_a_refusal_before_any_attempt_still_says_nothing_was_dispatched(self):
        ladder = RetryLadder(
            timeout=10.0, retries=0, now=Clock(),
            envelope=_envelope(max_requests=4, max_wall_s=25.0),
        )
        assert ladder.next_attempt() is None
        reason = budget_exhausted_reason(ladder, target="codex:gpt-5.5", timeout=10.0)
        assert "Nothing was dispatched" in reason



class TestRule8AnAttemptThatChangedFilesIsNotSentAgain:
    """Fix round 8 (review 7, r5): a write batch that replied in prose after
    editing was granted a transport retry, and the executor re-sent the whole
    batch on top of its own edits. ``repeatable=False`` is the caller's word
    that the identical dispatch must not go out again: no retry and no backoff,
    only the rule-7 re-ask when the attempt is repairable and the allowance
    has one."""

    @pytest.mark.parametrize("status", [
        STATUS_TIMEOUT, STATUS_RUNNER_ERROR, STATUS_UNPARSEABLE, STATUS_RATE_LIMITED,
    ])
    def test_no_retry_and_no_backoff(self, status):
        from utils.retry_ladder import STOP_NOT_REPEATABLE

        ladder = _ladder(Clock(), retries=3)
        ladder.next_attempt()
        action = ladder.record(status, repairable=False, repeatable=False)
        assert action.retry is False and action.delay == 0.0
        assert ladder.stop_reason == STOP_NOT_REPEATABLE
        assert ladder.transport_retries_left == 3
        assert ladder.rate_limit_attempts == 0
        assert "not sent again" in action.reason
        assert ladder.next_attempt() is None

    def test_a_repairable_reply_gets_the_re_ask_instead(self):
        ladder = RetryLadder(timeout=1200.0, retries=1, now=Clock(), schema_repairs=1)
        ladder.next_attempt()
        action = ladder.record(STATUS_UNPARSEABLE, repairable=True, repeatable=False)
        assert action.retry is True and action.repair is True
        assert ladder.schema_repairs_used == 1
        assert ladder.transport_retries_left == 1, "the re-ask spends no retry"

    def test_without_an_allowance_it_stops(self):
        ladder = _ladder(Clock(), retries=1)
        ladder.next_attempt()
        action = ladder.record(STATUS_UNPARSEABLE, repairable=True, repeatable=False)
        assert action.retry is False

    def test_the_re_ask_still_takes_the_admission_test(self):
        clock = Clock()
        ladder = RetryLadder(timeout=100.0, retries=1, now=clock, schema_repairs=1)
        ladder.next_attempt()
        clock.advance(100.0 - MIN_ATTEMPT_BUDGET_S / 2)
        action = ladder.record(STATUS_UNPARSEABLE, repairable=True, repeatable=False)
        assert action.retry is False
        assert action.note.startswith("no schema-repair re-ask: ")
        assert action.reason.startswith(STATUS_UNPARSEABLE)

    def test_fatal_and_terminal_statuses_are_unchanged(self):
        ladder = _ladder(Clock(), retries=1)
        ladder.next_attempt()
        assert ladder.record(STATUS_AUTH_FAILED, repeatable=False).retry is False
        assert ladder.stop_reason == STOP_FATAL

    def test_the_default_is_repeatable(self):
        ladder = _ladder(Clock(), retries=1)
        ladder.next_attempt()
        assert ladder.record(STATUS_UNPARSEABLE).retry is True
        assert ladder.schema_repairs_left == 0
