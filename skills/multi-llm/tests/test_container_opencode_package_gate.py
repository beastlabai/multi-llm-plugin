"""Row 8 of plans/opencode-plugin-publish.md: the installed opencode package, offline.

The release gate's package container coverage. Five rows, one per scenario the
plan's step 0 makes the row-8 minimum, each its own `docker run`, its own run
record and its own grader:

====================  ======================================
scenario              row id
====================  ======================================
init and status       ``opencode-package-gate-init-status``
detached launch       ``opencode-package-gate-detach``
supervision           ``opencode-package-gate-supervision``
completion            ``opencode-package-gate-completion``
cancellation          ``opencode-package-gate-cancellation``
====================  ======================================

**Every row runs the INSTALLED package** — `route: package`, opencode's own
config declaring the payload Dockerfile §10b staged at
`/opt/mllm/opencode-plugin`, no file shim — **from a real `opencode run`**,
typed the way a user types it (`--command multi-llm -- <args>`), with the host's
model a loopback stub scripted to make ONE `multi_llm` call. No credential, no
route out (`--creds none` runs under `--network none`), $0. The in-container
driver is `container/opencode_package_gate.py`; what it measured lands in the
bundle as `opencode-gate.json` beside the process samples, the host's own
`--format json` streams, the uv wrapper's receipts and the exported plan dir.

**Each grader writes `gate-grader.json` into its bundle** — the shape
plans/opencode-plugin-publish.impl-contract.md fixes, which `release_gate.py
check-row8` reads — with every check it ran, and `gate-observed.json` beside it
with the values those checks read. The record is re-emitted (and its bundle
re-digested) at session end by `harness.GRADES`, so both files are inside the
digest `make e2e-verify-record` re-hashes. Every grader also checks that the
dispatcher ran from the installed payload's skill dir: a file-route shim, or the
checkout copy at `/opt/mllm/skill`, fails it.

**The graders live in `container/opencode_package_gate_graders.py`**, pure
functions over a bundle, so they import without Docker:
`test_opencode_package_gate_graders.py` runs each of their checks in the plain
suite against a synthetic bundle and one targeted mutation of it, which is what
keeps a check from turning vacuous while these rows stay green.

Rows 2-4 run the same detached flow (`ask --quick`, the fan-out held so the run
outlives several `status --wait` polls) and each grades its own aspect of it; a
row is never credited with another row's run.

`container`-marked and NOT `live` or `live_provider`: `make e2e-container`'s
selection runs it, and nothing in it can spend.
"""

from __future__ import annotations

import pytest

from .container.docker_probe import requires_docker
from .container import harness
from .container.opencode_package_gate_graders import (
    Bundle,
    assert_graded,
    gate_row,
    grade_cancellation,
    grade_completion,
    grade_detach,
    grade_init_status,
    grade_supervision,
    write_grader,
)

pytestmark = [pytest.mark.container, requires_docker()]

#: The fan-out the dispatcher runs: the provider lane's own opencode slug, read
#: from redirect.yaml rather than typed (a bare `stub-model` does not resolve).
_OPENCODE_ROW = next(r for r in harness.load_redirect()["clis"]
                     if r.get("cli") == "opencode")
FAN_OUT = f"opencode:{_OPENCODE_ROW.get('model_arg', 'stub/stub-model')}"
#: Answers the fan-out's `ask` prompt (and every other phase's) on the row's
#: main stub. The host's own turn is answered by a second, scripted stub.
CHAIN_SCENARIO = ("/opt/mllm/skill/tests/fixtures/e2e/scenarios/"
                  "container_full_chain.yaml")


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _drive(scenario: str, posture, utc) -> harness.RowResult:
    row = gate_row(scenario, spec=[FAN_OUT], runner=FAN_OUT,
                   scenario=CHAIN_SCENARIO)
    return harness.run_row(row, posture=posture, utc=utc)


@pytest.fixture(scope="module")
def init_status(posture, utc):
    return _drive("init-status", posture, utc)


@pytest.fixture(scope="module")
def detach(posture, utc):
    return _drive("detach", posture, utc)


@pytest.fixture(scope="module")
def supervision(posture, utc):
    return _drive("supervision", posture, utc)


@pytest.fixture(scope="module")
def completion(posture, utc):
    return _drive("completion", posture, utc)


@pytest.fixture(scope="module")
def cancellation(posture, utc):
    return _drive("cancellation", posture, utc)


# --------------------------------------------------------------------------- #
# The five rows
# --------------------------------------------------------------------------- #

def _grade(result, scenario, grader) -> None:
    checks, observed, abort_method = grader(result)
    bundle = Bundle(result)
    document = write_grader(bundle, scenario, grader.__name__, checks, observed,
                            abort_method)
    assert_graded(document, observed)


def test_init_and_status(init_status):
    """Scenario 1: `/multi-llm` is the payload's, `init` creates its files,
    `status` answers its documented result."""
    _grade(init_status, "init-status", grade_init_status)


def test_detached_launch(detach):
    """Scenario 2: the tool composes `--detach`, claims the banner's run, and
    the worker runs in its own process group, outside opencode's tree."""
    _grade(detach, "detach", grade_detach)


def test_supervision(supervision):
    """Scenario 3: `status --wait` on the claimed run, non-terminal polls first,
    and the reported poll count equals the polls observed."""
    _grade(supervision, "supervision", grade_supervision)


def test_completion(completion):
    """Scenario 4: one terminal result naming the run; its per-model outputs
    under the plan's dir; nothing written outside it."""
    _grade(completion, "completion", grade_completion)


def test_cancellation(cancellation):
    """Scenario 5: abort the session mid-run; the tool aborts the run it
    claimed, its group is gone, and its report names the run and its log."""
    _grade(cancellation, "cancellation", grade_cancellation)
