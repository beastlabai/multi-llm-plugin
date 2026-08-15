"""What `tests/container/redirect.yaml` claims, checked against the product — in CI.

The container matrix is data, and data drifts. Two of its claims are checkable
without a daemon, and both of them **must** be, because the modules that consume
them are `container`-marked and therefore never run on a PR:

* **`chain_write_runner`** — the executor runner a chain row dispatches every
  subagent through. It is present exactly for the CLIs whose own adapter declares
  no `agent-write` profile, and it names one that does. Asserted here rather than
  inside `test_container_full_chain.py`, because a drift guard that only runs
  where Docker runs is a drift guard that does not run: the first sign of a
  mismatch would otherwise be a chain failing mid-write with a
  `ProfileUnsupported` refusal, which reads as a redirection problem and is
  nothing of the kind.

* **the gating itself** — every `test_container_*.py` module carries the marker
  *and* the no-Docker `skipif`, and the expression CI actually types selects
  nothing from them. Three separate mechanisms have to agree for that
  (`addopts`, the workflow's own `-m`, the module-level `skipif`), and the cost of
  them disagreeing is a red PR gate on three OS runners with no daemon.

Plain CI: no container, no Docker, no network. One subprocess, for the collection
check, because "what does pytest select" cannot be asserted from inside the run
that pytest already selected.

**The filename is deliberately not `test_container_*`.** That glob is what the
build context ships into the image and what the gating check below enumerates;
a plain-CI module answering to it would be shipped for nothing and would fail its
own marker assertion.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import get_provider  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REDIRECT = TESTS_DIR / "container" / "redirect.yaml"
CONTAINER_MODULES = tuple(sorted(TESTS_DIR.glob("test_container_*.py")))


def _provider_rows() -> list[dict]:
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8"))
    return [row for row in data["clis"] if row.get("lane") == "provider"]


PROVIDER_ROWS = _provider_rows()
PROVIDER_IDS = [row["cli"] for row in PROVIDER_ROWS]


def _declares_write(cli: str) -> bool:
    provider = get_provider(cli)
    assert provider is not None, f"{cli} has no built-in adapter"
    return "agent-write" in (getattr(provider, "agent_profiles", None) or ())


class TestTheChainWriteRunner:
    """`chain_write_runner` against the adapters, in both directions."""

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_it_is_declared_exactly_where_the_cli_has_no_agent_write(self, row):
        declared = row.get("chain_write_runner")
        if _declares_write(row["cli"]):
            assert declared is None, (
                f"{row['cli']} declares agent-write itself, so redirect.yaml must "
                f"not name a chain_write_runner ({declared}) — that would add a "
                f"second binary to the row for nothing, and the per-row "
                f"single-CLI assertion would then require it to have run"
            )
        else:
            assert declared is not None, (
                f"{row['cli']} declares no agent-write profile, so its chain "
                f"row's executor needs a different binary. Declare it as "
                f"chain_write_runner or the row fails at its first write "
                f"dispatch with a ProfileUnsupported refusal."
            )

    @pytest.mark.parametrize("row", PROVIDER_ROWS, ids=PROVIDER_IDS)
    def test_the_named_runner_can_actually_write(self, row):
        declared = row.get("chain_write_runner")
        if declared is None:
            pytest.skip(f"{row['cli']} is its own write runner")
        assert _declares_write(declared), (
            f"{declared} is named as {row['cli']}'s write runner but declares no "
            f"agent-write profile of its own"
        )

    def test_the_named_runner_is_itself_a_redirected_row(self):
        """A helper nobody redirects would reach for a real endpoint."""
        redirected = set(PROVIDER_IDS)
        for row in PROVIDER_ROWS:
            declared = row.get("chain_write_runner")
            if declared is not None:
                assert declared in redirected, (
                    f"{row['cli']}'s write runner {declared} is not a "
                    f"provider-lane row, so nothing redirects it at the stub"
                )


class TestTheContainerModulesAreGated:
    """The marker, the skipif, and the expression CI actually types."""

    def test_there_is_at_least_one_container_module(self):
        assert CONTAINER_MODULES, "no test_container_*.py modules found"

    @pytest.mark.parametrize("module", CONTAINER_MODULES,
                             ids=[m.stem for m in CONTAINER_MODULES])
    def test_each_module_carries_the_marker_and_the_skipif(self, module):
        """Source-level, so it holds on a machine that has no daemon to probe."""
        source = module.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "pytestmark" for t in node.targets)
        ]
        assert assignments, f"{module.name} declares no module-level pytestmark"
        rendered = ast.unparse(assignments[-1].value)
        assert "pytest.mark.container" in rendered, (
            f"{module.name}'s pytestmark does not carry the container marker: "
            f"{rendered}"
        )
        assert "requires_docker()" in rendered, (
            f"{module.name}'s pytestmark does not carry the no-Docker skipif: "
            f"{rendered}. Without it the module fails rather than skips on a "
            f"machine with no daemon."
        )

    def test_the_ci_selection_collects_nothing_from_them(self):
        """The one assertion the other two cannot make.

        An explicit `-m` on the command line *replaces* `addopts` rather than
        composing with it, which is exactly how a container-marked module ends up
        collected on a runner with no daemon. This types the expression the
        workflow types.
        """
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-m", "not live and not container",
             *[str(m) for m in CONTAINER_MODULES]],
            cwd=str(TESTS_DIR.parent), capture_output=True, text=True, timeout=300,
        )
        # Exit 5 is "no tests collected", which is the passing outcome here.
        assert completed.returncode in (0, 5), (
            f"collection failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
        selected = [
            line for line in completed.stdout.splitlines()
            if "::" in line and "test_container_" in line
        ]
        assert not selected, (
            "the default CI selection collects container-marked tests:\n"
            + "\n".join(selected[:10])
        )
