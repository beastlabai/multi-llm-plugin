"""Provider lane: does the real CLI still accept our argv and emit a stream we parse?

This module **is** the driver. Each test builds one `docker run` through
`tests/container/harness.py`, waits for it, and then asserts over the files that
run left on the bind mount. Nothing inside the container decides whether a row
passed — it produces evidence, and this reads it.

Why the driver stays outside the image: the marker-plus-`skipif` gating has to be
evaluatable on a machine with no container runtime at all; the golden comparison
writes into the repo tree, which is deliberately never mounted; and keeping the
test suite out of the image keeps it out of `context_manifest.txt`, so what ships
inside is the product and nothing else.

**The lane's contract, and the reason it can run for free:** zero secrets and zero
network *at run time*. Every row runs `--network none` and no credential variable
is passed into the container at all — enforced by the flag, not by the CLIs' good
behaviour. A contributor without any API key can run this half of the suite.

What this proves: *argv accepted* + *stream shape parses*, against the real binary.
What it does not prove: behaviour under rate limits or long generations. The pinned
`tests/fixtures/*/*.jsonl` captures from real runs stay as the complement.
"""

from __future__ import annotations

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, requires_docker()]

REDIRECT = harness.HERE / "redirect.yaml"

#: In-image path. The image ships tests/fixtures/e2e/ because the stub resolves
#: a scenario's `fixture:` paths relative to the scenario file.
SCENARIO = "/opt/mllm/skill/tests/fixtures/e2e/scenarios/full_workflow.yaml"


def _rows():
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return [r for r in data["clis"] if r.get("lane") == "provider"]


VERIFIED = _rows()
IDS = [r["cli"] for r in VERIFIED]


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    """One timestamped run directory for the whole module."""
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@pytest.fixture(scope="module", params=VERIFIED, ids=IDS)
def row_result(request, posture, utc):
    """Run one CLI's row once and share its evidence across the assertions below.

    One `docker run` per row, never a long-lived shared container: that is what
    makes "no row inherits another's stub session table" hold by construction
    rather than by discipline.
    """
    entry = request.param
    cli = entry["cli"]
    spec = f"{cli}:{entry.get('model_arg', 'stub-model')}"
    row = harness.Row(
        row_id=f"provider-{cli}",
        lane="provider",
        cli=cli,
        spec=spec,
        runner=spec,
        helpers=entry.get("allowed_helpers", []),
        command="review-plan plans/auth-feature.md --yes --quick",
        # Without a scenario the stub answers a bare "OK", which is not a valid
        # suggestion document — the phase then fails at rung 6 and the row exits 3.
        # A scenario is what makes this a test of the CLI rather than of the stub's
        # default. (Phase 3 replaces this with container_full_chain.yaml, which
        # covers the turns full_workflow.yaml has never heard of.)
        scenario=SCENARIO,
        row_timeout_s=entry.get("row_timeout_s", 300),
    )
    return harness.run_row(row, posture=posture, utc=utc)


class TestTheRowReachedTheStub:
    """The assertions that make 'redirection worked' evidence, not inference."""

    def test_the_row_did_not_hit_its_cap(self, row_result):
        # A cap hit is a diagnosis, not a hang: the reason is classified from the
        # row's stub log before reporting, so the failure names a cause rather
        # than a duration. A capped row is never graded against a golden.
        assert not row_result.timed_out, (
            f"{row_result.row.cli}: {row_result.cap_reason} after "
            f"{row_result.wall_clock_s}s (cap {row_result.row.row_timeout_s}s)"
        )

    def test_the_row_exited_zero(self, row_result):
        """The assertion whose absence made the first green run meaningless.

        Every other check here is satisfied by a row that dispatched, failed to
        parse, and exited 3: the stub still logged a request, the wrapper still
        wrote a call record, and the entrypoint's `finally` still exported a phase
        directory. "Artifacts exist and something ran" is not the bar — a phase
        directory is created *before* the work inside it happens, so its existence
        is not evidence that it finished.
        """
        assert row_result.exit_code == 0, (
            f"{row_result.row.cli}: exited {row_result.exit_code}\n"
            f"--- container stderr (tail) ---\n{row_result.stderr[-2000:]}"
        )

    def test_the_cli_actually_reached_the_stub(self, row_result):
        """The evidence that a CLI did not silently fall back to a real endpoint.

        Under `--network none` a fallback cannot succeed, but it can still *fail*
        in a way that looks like a redirect problem — so the positive record is
        what distinguishes "we redirected it" from "it never called anything".
        """
        calls = [r for r in row_result.stub_log if r.get("surface") != "health"]
        assert calls, (
            f"{row_result.row.cli}: zero requests reached the stub — a resolver "
            f"or socket wait, an unredirected surface, or a prompt nobody answered"
        )

    def test_no_route_went_unserved(self, row_result):
        """An unserved route is a failed row, not a skip.

        Under `--network none` an SDK's catalog fetch or token count cannot fall
        through to a real endpoint: the CLI either aborts or retries until it
        times out. Failing on the record rather than on the CLI's exit code is
        what keeps that from surfacing as an unexplained timeout.
        """
        unhandled = [(r["method"], r["path"]) for r in row_result.stub_log
                     if r.get("unhandled")]
        assert not unhandled, (
            f"{row_result.row.cli}: the stub does not serve {unhandled} — add a "
            f"handler and the route to this CLI's redirect.yaml http_log"
        )

    def test_no_prompt_fell_through_to_a_default(self, row_result):
        """An unmatched prompt fails the row; it never resolves to a default.

        A generic fallback answer is shaped enough to look like a passing phase
        while proving nothing — on this lane it would mean a CLI never exercising
        the fixture its row claims to cover.
        """
        unmatched = [r["prompt_sha"] for r in row_result.stub_log
                     if r.get("unmatched")]
        assert not unmatched, (
            f"{row_result.row.cli}: {len(unmatched)} prompt(s) matched no pattern: "
            f"{unmatched}"
        )


class TestTheRowRanExactlyOneCli:
    """"One full chain per CLI" made true rather than nominal (C7).

    The project config layer *deep-merges* onto the shipped base and never
    replaces it, so every selection key a row leaves unset keeps base's value —
    and base's values name eight other CLIs. These two assertions are what catch a
    row that quietly fanned out.
    """

    def test_the_call_log_names_only_this_row_s_binaries(self, row_result):
        entry = next(r for r in VERIFIED if r["cli"] == row_result.row.cli)
        expected = {row_result.row.cli} | set(entry.get("allowed_helpers", []))
        seen = {record["cli"] for record in row_result.call_log}
        seen.discard("uv")
        # Both directions: an unexpected binary fails the row naming it, and a
        # declared helper that never ran fails it too — the same stale-entry
        # discipline the goldens carry, for the same reason.
        assert seen == expected, (
            f"{row_result.row.cli}: call log names {sorted(seen)}, "
            f"expected exactly {sorted(expected)}"
        )

    def test_the_stub_saw_only_this_row_s_model(self, row_result):
        """An exact allowed set, not a count.

        A count is satisfied by a row whose selection override failed outright
        and ran the whole phase on base's default — one model id, the wrong one.
        `side_model_ids` is what keeps this an assertion rather than a budget:
        grok's session-title call hardcodes `grok-build` instead of the `-m`
        value, and gemini puts the model in the PATH rather than the body, so the
        stub records its own `stub-model` default for it. Both are declared;
        anything else is base's selection leaking into the row.
        """
        entry = next(r for r in VERIFIED if r["cli"] == row_result.row.cli)
        allowed = {
            # Every model this row pinned — the same strings the fixture built
            # both driver keys from, so there is no second source to drift
            # from. `Row.spec` is a LIST now (a provider-lane row still renders
            # exactly one entry; the host lane's fan-out is what needed the
            # width), so it is read through `harness.spec_list` rather than
            # `.split(":")`d as a scalar.
            *(s.split(":", 1)[1] for s in harness.spec_list(row_result.row)),
            "stub-model",  # the stub's own default for a body carrying no model
            *entry.get("side_model_ids", []),
        }
        models = {r.get("model") for r in row_result.stub_log
                  if r.get("surface") not in (None, "health")}
        models.discard(None)
        assert models <= allowed, (
            f"{row_result.row.cli}: stub saw model ids {sorted(models - allowed)} "
            f"that this row did not pin ({sorted(allowed)}), so the selection "
            f"override leaked. If the CLI legitimately sends one, declare it in "
            f"redirect.yaml's side_model_ids."
        )
        assert models, f"{row_result.row.cli}: the stub recorded no model id at all"


class TestPiAdvertisesExactlyItsAllowList:
    """T3.1's behavioural pin (review-run-scope-salvage-and-failure-visibility).

    ``PiProvider`` renders ``--tools read,grep,find,bash,write``, and pi drops an
    unknown or mis-separated name without a word — so the argv table and the
    help contract can both be green while pi reviews with a different toolset,
    or none. Only the request pi really sent says which tools the model was
    offered. (The stub logs chat-completions tool names since the same task;
    before it, a chat host's ``tool_names`` was always empty.)
    """

    def test_the_first_request_offers_exactly_the_allow_list(self, row_result):
        from utils.providers.pi import PI_TOOLS

        if row_result.row.cli != "pi":
            pytest.skip("pi's --tools allow-list only")
        calls = [r for r in row_result.stub_log
                 if r.get("method") == "POST" and r.get("has_tools")]
        assert calls, "pi: no request carried tools at all"
        assert sorted(calls[0].get("tool_names") or []) == sorted(PI_TOOLS), (
            f"pi advertised {sorted(calls[0].get('tool_names') or [])}, not the "
            f"allow-list {sorted(PI_TOOLS)} — a dropped or renamed tool name"
        )


class TestTheEvidenceSurvivedTheContainer:
    def test_artifacts_were_exported(self, row_result):
        """The entrypoint's `finally` export ran.

        The orchestrator computes its output dir from the plan path, which sits
        inside the tmpfs workspace and dies with the container. A row whose export
        step did not run is a failed row, not a row with an empty golden diff.
        """
        artifacts = row_result.out_dir / "artifacts"
        assert artifacts.is_dir(), (
            f"{row_result.row.cli}: no artifacts/ under {row_result.out_dir} — "
            f"the entrypoint's finally export did not run"
        )

    def test_the_logs_are_readable_after_the_container_is_gone(self, row_result):
        assert (row_result.out_dir / "stub.jsonl").is_file()
        assert (row_result.out_dir / "calls.jsonl").is_file()

    def test_the_wrapper_scrubbed_the_child_environment(self, row_result):
        """The scrub receipt: variable NAMES, never values (C2).

        Redirection is not containment. The wrapper builds the child's environment
        from an allowlisted empty base rather than overlaying an inherited one, so
        a CLI that falls back past the stub carries a string that buys nothing.
        This asserts the receipt shows that, which is what makes "the wrapper
        scrubs" a test rather than a comment.
        """
        for record in row_result.call_log:
            names = set(record.get("child_env_names", []))
            leaked = {n for n in names
                      if n.endswith(("_API_KEY", "_AUTH_TOKEN"))
                      and not n.startswith(("STUB_", "MLLM_"))}
            # The fake per-exec stub key is expected where a vendor variable is;
            # a REAL vendor variable name is not.
            assert "OPENROUTER_API_KEY" not in names, (
                f"{record['cli']}: the live key name reached the child environment"
            )
            assert not (leaked - {"OPENAI_API_KEY", "META_API_KEY", "XAI_API_KEY",
                                  "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                                  # kimi's row declares this in credential_vars,
                                  # so the wrapper fills it with the fake
                                  # per-exec stub key -- the name is expected
                                  # here, the host's value never is (the
                                  # OPENROUTER_API_KEY assertion above is what
                                  # pins that, and it covers kimi too).
                                  "KIMI_MODEL_API_KEY",
                                  # copilot's row declares this in
                                  # credential_vars for the same reason, and the
                                  # variable is also its BYOK ACTIVATION switch's
                                  # partner: without a provider key beside the
                                  # provider base URL the CLI falls through to
                                  # the GitHub seat path.
                                  "COPILOT_PROVIDER_API_KEY"}), (
                f"{record['cli']}: unexpected credential-shaped variable {leaked}"
            )

    def test_no_credential_shaped_value_was_logged(self, row_result):
        """Redaction happens at write time, so the log can never carry a value."""
        import re

        shapes = re.compile(r"sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}")
        for path in ("stub.jsonl", "calls.jsonl"):
            text = (row_result.out_dir / path).read_text(encoding="utf-8")
            assert not shapes.search(text), f"{path} carries a credential-shaped value"
