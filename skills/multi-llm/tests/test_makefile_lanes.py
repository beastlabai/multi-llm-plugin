#!/usr/bin/env python3
"""The Makefile's lane wiring, asserted as text — in plain CI, with no Docker.

``test_dockerfile_pins.py`` proves what the Dockerfile *says*; this proves what
the Makefile *says*. Both are static reads for the same reason: a guard that only
runs where the container lane runs is a guard that does not run.

The property that matters here is a money one. ``e2e-live-provider`` drives real
CLIs against ``https://openrouter.ai`` on the developer's own key
(harness-expansion.md §7.4), and the whole design of that lane assumes nobody
reaches it by accident. "It is opt-in" is a claim about a graph — one edge added
to ``check`` or ``e2e-container`` in a hurry and every future suite run bills
somebody. So the edge count is asserted rather than remembered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[3] / "Makefile"

#: The target every assertion below is about.
LIVE = "e2e-live-provider"

#: `target: prereq prereq` — the only construct that can create an edge. Recipe
#: lines start with a tab and are excluded by the leading-character check.
RULE = re.compile(r"^(?P<targets>[A-Za-z0-9_.\-/ ]+):(?!=)(?P<prereqs>[^=]*)$")


@pytest.fixture(scope="module")
def makefile_text() -> str:
    assert MAKEFILE.is_file(), f"no Makefile at {MAKEFILE}"
    return MAKEFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rules(makefile_text: str) -> list[tuple[list[str], list[str]]]:
    """`[(targets, prereqs)]` for every rule line, comments and recipes excluded."""
    parsed = []
    for line in makefile_text.splitlines():
        if not line or line[0] in ("\t", "#", " "):
            continue
        match = RULE.match(line)
        if not match:
            continue
        targets = match.group("targets").split()
        if targets == [".PHONY"]:
            continue
        parsed.append((targets, match.group("prereqs").split()))
    return parsed


def test_the_live_lane_target_exists(rules):
    assert any(LIVE in targets for targets, _ in rules), (
        f"{LIVE} is not defined — §7.4's lane is unreachable without it")


def test_it_is_declared_phony(makefile_text):
    phony = " ".join(
        line for line in makefile_text.splitlines() if ".PHONY" in line
        or (line.startswith(" ") and "e2e-" in line))
    assert LIVE in phony, f"{LIVE} is missing from .PHONY"


def test_no_other_goal_reaches_it(rules):
    """The assertion the target's own comment block promises.

    A credit-spending lane reachable from ``check``, ``e2e-container`` or any
    other goal is a bill nobody typed. This is the graph edge, checked.
    """
    offenders = [targets for targets, prereqs in rules
                 if LIVE in prereqs and LIVE not in targets]
    assert not offenders, (
        f"{LIVE} is a prerequisite of {offenders} — it must be reachable only "
        f"when typed directly, because it spends real credits")


def test_no_recipe_invokes_it_recursively(makefile_text):
    """`$(MAKE) e2e-live-provider` inside another recipe is the same edge.

    Prerequisites are not the only way to reach a target: a recursive `$(MAKE)`
    in some other target's recipe would do it too, and would not show up in the
    dependency graph at all.
    """
    for number, line in enumerate(makefile_text.splitlines(), start=1):
        if line.startswith("\t") and "$(MAKE)" in line and LIVE in line:
            pytest.fail(f"Makefile:{number} invokes {LIVE} recursively: {line.strip()}")


def test_it_does_not_build_the_image_implicitly(rules, makefile_text):
    """A stale image is a visible failure, never a silent rebuild.

    Rebuilding under the target would mean the credits get spent on whatever
    binary the rebuild happened to install, which is the opposite of what a
    pinned smoke is for.
    """
    for targets, prereqs in rules:
        if LIVE in targets:
            assert "e2e-image" not in prereqs, (
                f"{LIVE} must not depend on e2e-image — it would spend credits "
                f"on a freshly built, possibly different binary")

    body = makefile_text.split(f"{LIVE}:", 1)[1].split("\n\n", 1)[0]
    assert "e2e-image" not in body, (
        f"{LIVE}'s recipe must not build the image implicitly")


def test_the_recipe_sources_the_key_from_the_documented_path(makefile_text):
    """One credential path, no other (§7.4's credential-injection rule)."""
    body = makefile_text.split(f"{LIVE}:", 1)[1].split("\n\n", 1)[0]
    assert "MULTI_LLM_E2E_ENV" in body and "e2e.env" in body, (
        "the recipe must read the key from ~/.config/multi-llm/e2e.env "
        "(overridable by MULTI_LLM_E2E_ENV), never from an ambient export")
    assert "live_provider" in body, (
        "the recipe must select the lane by its marker, so a bare pytest and CI "
        "both deselect it")


def test_the_marker_is_registered():
    """An unregistered marker is a warning, not a filter — and warnings get muted."""
    pyproject = MAKEFILE.parent / "skills" / "multi-llm" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "live_provider:" in text, (
        "pyproject.toml must register the live_provider marker, or -m selection "
        "silently matches nothing")


def test_the_host_lane_target_does_not_demand_the_key(makefile_text):
    """codex and claude-code run on the subscription; only opencode needs the key.

    Operator decision 2026-10-02. The census (`_refuse_empty_lane`) is what stops
    a run with no credential at all from passing, so the Makefile need not.
    """
    body = makefile_text.split("\ne2e-container-host: e2e-image\n", 1)[1].split(
        "\n\n", 1)[0]
    assert "exit 1" not in body
    assert "subscription" in body


def test_help_says_the_host_lane_takes_the_subscription(makefile_text):
    line = next(l for l in makefile_text.splitlines()
                if "make e2e-container-host " in l and "@echo" in l)
    assert "subscription" in line and "OPENROUTER_API_KEY" in line


# ---------------------------------------------------------------------------
# The unpinned host lane's credential guards, executed
# ---------------------------------------------------------------------------
#
# `e2e-container-latest-host` hands a live key to `@latest` binaries, so its two
# guards are run here rather than read: in process for every rule, and through
# make for the wiring. The make runs skip the docker pre-flight with `-o
# e2e-preflight` (here only, never in a recipe) and put a `docker` on PATH that
# records any call and fails, so a guard that wrongly let the run through could
# not build an image.

import importlib.util  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

LATEST_HOST = "e2e-container-latest-host"
_GUARD_PATH = MAKEFILE.parent / "skills" / "multi-llm" / "tests" / "container" / "lane_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("lane_guard_under_test", _GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _recipe(makefile_text: str, target: str) -> list:
    """The recipe lines of ``target``, continuation lines joined."""
    lines, inside = [], False
    for line in makefile_text.replace("\\\n", " ").splitlines():
        if inside and line.startswith("\t"):
            lines.append(line.strip())
            continue
        inside = re.match(rf"^{re.escape(target)}\s*:(?!=)", line) is not None
    return lines


class TestTheUnpinnedHostRule:
    BASE = {"PATH": "/usr/bin"}

    def test_no_key_is_refused_first(self):
        refusal = guard.unpinned_host_refusal({**self.BASE, guard.OPT_IN: "1"})
        assert refusal and "needs OPENROUTER_API_KEY" in refusal[0]

    def test_an_empty_key_is_no_key(self):
        refusal = guard.unpinned_host_refusal({guard.LANE_KEY: "", guard.OPT_IN: "1"})
        assert refusal and "needs OPENROUTER_API_KEY" in refusal[0]

    @pytest.mark.parametrize("opt_in", [None, "", "0", "true", "yes", "1 ", " 1", "11"])
    def test_only_the_exact_opt_in_opens_the_lane(self, opt_in):
        env = {guard.LANE_KEY: "dummy-not-a-key"}
        if opt_in is not None:
            env[guard.OPT_IN] = opt_in
        refusal = guard.unpinned_host_refusal(env)
        assert refusal and "UNPINNED" in refusal[0]
        assert any(f"{guard.OPT_IN}=1 deliberately" in line for line in refusal)

    def test_the_key_and_the_opt_in_open_it(self):
        assert guard.unpinned_host_refusal(
            {guard.LANE_KEY: "dummy-not-a-key", guard.OPT_IN: "1"}) is None

    def test_a_refusal_names_every_exported_vendor_key(self):
        env = {guard.LANE_KEY: "dummy-not-a-key", "OPENAI_API_KEY": "x", "GEMINI_API_KEY": "y",
               "ANTHROPIC_API_KEY": ""}
        named = [line for line in guard.unpinned_host_refusal(env) if "also exported" in line]
        assert [line.split()[2] for line in named] == ["OPENAI_API_KEY", "GEMINI_API_KEY"]

    def test_the_lane_key_is_not_a_vendor_key(self):
        assert guard.LANE_KEY not in guard.VENDOR_KEYS


class TestTheRowsRanRule:
    def _stamp(self, tmp_path):
        stamp = tmp_path / "stamp"
        stamp.write_text("")
        past = time.time() - 60
        os.utime(stamp, (past, past))
        return stamp

    def test_no_record_is_not_run(self, tmp_path):
        records = tmp_path / "records"
        records.mkdir()
        old = records / "20200101T000000Z-row.json"
        old.write_text("{}")
        os.utime(old, (0, 0))
        refusal = guard.rows_ran_refusal(records, self._stamp(tmp_path))
        assert refusal and "NOT-RUN" in refusal[0]

    def test_a_record_newer_than_the_stamp_is_a_run(self, tmp_path):
        records = tmp_path / "records" / "nested"
        records.mkdir(parents=True)
        stamp = self._stamp(tmp_path)
        (records / "20991231T000000Z-row.json").write_text("{}")
        assert guard.rows_ran_refusal(tmp_path / "records", stamp) is None

    def test_a_missing_records_directory_is_not_run(self, tmp_path):
        refusal = guard.rows_ran_refusal(tmp_path / "absent", self._stamp(tmp_path))
        assert refusal and "NOT-RUN" in refusal[0]

    def test_an_unreadable_stamp_refuses(self, tmp_path):
        records = tmp_path / "records"
        records.mkdir()
        (records / "x.json").write_text("{}")
        refusal = guard.rows_ran_refusal(records, tmp_path / "no-stamp")
        assert refusal and "NOT-RUN" in refusal[0]

    def test_a_directory_named_like_a_record_is_not_one(self, tmp_path):
        records = tmp_path / "records"
        (records / "fake.json").mkdir(parents=True)
        refusal = guard.rows_ran_refusal(records, self._stamp(tmp_path))
        assert refusal and "NOT-RUN" in refusal[0]


class TestTheGuardWiring:
    def test_the_guard_runs_before_the_image_is_built(self, makefile_text, rules):
        prereqs = next(p for targets, p in rules if LATEST_HOST in targets)
        assert prereqs == ["e2e-preflight"], (
            "a prerequisite runs before the guard; the image must not be one")
        lines = _recipe(makefile_text, LATEST_HOST)
        assert lines[0] == "@$(LANE_GUARD) unpinned-host", lines[0]
        build = next(i for i, line in enumerate(lines) if "e2e-image-latest" in line)
        assert build > 0

    def test_the_lane_guard_is_this_module(self, makefile_text):
        line = next(l for l in makefile_text.splitlines() if l.startswith("LANE_GUARD :="))
        assert "$(CONTAINER_DIR)/lane_guard.py" in line
        assert "CONTAINER_DIR := skills/multi-llm/tests/container" in makefile_text

    def test_the_scrub_list_is_the_guard_s_and_spares_the_lane_key(self, makefile_text):
        """Every vendor key is unset for the run; the lane's own key never is."""
        run = next(line for line in _recipe(makefile_text, LATEST_HOST)
                   if "$(PYTEST)" in line)
        unset = re.findall(r"-u\s+([A-Z_]+)", run)
        assert unset == list(guard.VENDOR_KEYS)
        assert guard.LANE_KEY not in unset

    def test_the_not_run_check_follows_the_pytest_in_the_same_shell(self, makefile_text):
        run = next(line for line in _recipe(makefile_text, LATEST_HOST)
                   if "$(PYTEST)" in line)
        assert run.index("$(PYTEST)") < run.index("$(LANE_GUARD) rows-ran")
        assert '--since "$$stamp"' in run


@pytest.mark.skipif(shutil.which("make") is None or os.name == "nt"
                    or shutil.which("uv") is None,
                    reason="needs make, a POSIX shell and uv to run the recipe")
class TestTheGuardThroughMake:
    def _make(self, tmp_path, **env_overrides):
        stubs = tmp_path / "bin"
        stubs.mkdir()
        calls = tmp_path / "docker-calls"
        docker = stubs / "docker"
        docker.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\nexit 99\n')
        docker.chmod(0o755)
        env = {k: v for k, v in os.environ.items()
               if k not in (guard.LANE_KEY, guard.OPT_IN, *guard.VENDOR_KEYS,
                            "MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
        env["PATH"] = f"{stubs}{os.pathsep}{env.get('PATH', '')}"
        env.update(env_overrides)
        done = subprocess.run(["make", "--no-print-directory", "-o", "e2e-preflight", LATEST_HOST],
                              cwd=MAKEFILE.parent, env=env, capture_output=True, text=True,
                              timeout=300)
        return done, (calls.read_text() if calls.exists() else "")

    def test_without_the_key_it_refuses_before_docker(self, tmp_path):
        done, docker = self._make(tmp_path)
        assert done.returncode != 0
        assert "needs OPENROUTER_API_KEY" in done.stdout + done.stderr
        assert docker == "", f"docker ran before the refusal: {docker}"

    def test_without_the_opt_in_it_refuses_before_docker(self, tmp_path):
        done, docker = self._make(tmp_path, OPENROUTER_API_KEY="dummy-not-a-key",
                                  OPENAI_API_KEY="dummy-vendor")
        output = done.stdout + done.stderr
        assert done.returncode != 0
        assert "refusing to run the host lane against an UNPINNED image" in output
        assert "also exported: OPENAI_API_KEY" in output
        assert "dummy-not-a-key" not in output and "dummy-vendor" not in output
        assert docker == "", f"docker ran before the refusal: {docker}"

    def test_a_loose_opt_in_is_refused(self, tmp_path):
        done, docker = self._make(tmp_path, OPENROUTER_API_KEY="dummy-not-a-key",
                                  MLLM_ALLOW_UNPINNED_HOST_LANE="yes")
        assert done.returncode != 0
        assert "UNPINNED" in done.stdout + done.stderr
        assert docker == ""


# ---------------------------------------------------------------------------
# PLAIN_ROWS: which plain rows the three `-ci` tiers run
# ---------------------------------------------------------------------------

PLAIN = '-m "not live and not container and not package"'
PACKAGE_ROWS = '-m "package and not container"'


@pytest.mark.skipif(shutil.which("make") is None or os.name == "nt",
                    reason="dry-runs the recipes, which needs make and a POSIX shell")
class TestPlainRows:
    """``graded`` (the PR ``package`` job) keeps only the plain rows a grader
    reads; ``all`` (every other caller) keeps every one. Dry runs only: the
    builders are replaced by ``true``, and no pytest starts."""

    def _dry(self, target: str, *assignments: str):
        env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
        return subprocess.run(["make", "--no-print-directory", "-n", target,
                               "CODEX_BUILD=true", "CLAUDE_BUILD=true", *assignments],
                              cwd=MAKEFILE.parent, env=env, capture_output=True, text=True,
                              timeout=120)

    def test_an_unknown_value_is_refused(self):
        done = self._dry("help", "PLAIN_ROWS=yes")
        assert done.returncode != 0
        assert "PLAIN_ROWS must be `all` or `graded`" in done.stderr

    @pytest.mark.parametrize("target", ["codex-package-check-ci", "claude-package-check-ci"])
    def test_graded_leaves_the_plain_rows_and_keeps_the_package_rows(self, target):
        graded = self._dry(target, "PLAIN_ROWS=graded")
        assert graded.returncode == 0, graded.stderr
        assert PLAIN not in graded.stdout and "plain rows left to the test job" in graded.stdout
        assert "MLLM_REQUIRE_PACKAGE_ROWS=1" in graded.stdout and PACKAGE_ROWS in graded.stdout
        whole = self._dry(target)
        assert whole.returncode == 0, whole.stderr
        assert PLAIN in whole.stdout and PACKAGE_ROWS in whole.stdout

    def test_graded_opencode_runs_exactly_the_nodes_its_rows_name(self):
        graded = self._dry("opencode-package-check-ci", "PLAIN_ROWS=graded")
        assert graded.returncode == 0, graded.stderr
        assert 'plain="$(' in graded.stdout and 'release_gate.py ci-nodes)"; test -n "$plain"' \
            in graded.stdout
        assert "pytest $plain" in graded.stdout
        assert "grade-ci" in graded.stdout
        whole = self._dry("opencode-package-check-ci")
        assert "ci-nodes" not in whole.stdout
        assert "skills/multi-llm/tests/test_opencode_package.py" in whole.stdout

    def test_the_full_gates_force_every_plain_row(self, makefile_text):
        assert "-- $(MAKE) --no-print-directory opencode-package-check-ci PLAIN_ROWS=all" \
            in makefile_text
        assert "\t$(MAKE) --no-print-directory codex-package-check-ci PLAIN_ROWS=all\n" \
            in makefile_text


def test_the_graded_nodes_are_the_ci_rows_nodes():
    from tests.test_opencode_release_gate import gate

    nodes = gate.ci_node_ids()
    declared = [node for row in gate.ROWS for node in row.nodes]
    assert nodes == [f"skills/multi-llm/{node}" for node in dict.fromkeys(declared)]
    for node in nodes:
        assert (MAKEFILE.parent / node.split("::")[0]).is_file(), node
