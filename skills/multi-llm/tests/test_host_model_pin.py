"""Which model a host-lane row drives, pinned in CI (R1).

`tests/container/host_matrix.yaml` pins one model per HOST. Since 2026-08-21 it
also pins one per ROW — exactly one row, `claude-code-apply-native`, whose four
measured attempts on the host's `~deepseek/deepseek-v4-flash-latest` pin never
once reached R1's `apply-code-fixes … --mark-completed` finalization (bundle ids,
turn counts and spend in the row's own comment). The remedy the matrix had
already named is the verified Anthropic fallback, `anthropic/claude-haiku-4.5`.

That override is a SPEND decision in both directions and neither direction is
detectable without money: too narrow and the row keeps failing on the model
rather than on our shim; too wide and every other row silently moves from
~$0.02 to ~$0.24 a run. The only place either mistake shows up is a credentialed
Docker run — `test_container_host_lane.py` is `live`-marked beneath a module-level
`container` gate — so a PR that re-pins a row, or that "tidies" the block up to
the host, would sail through CI and be discovered by an invoice.

So this module pins the resolver and the declaration offline, in both
directions: no credential, no daemon, no network. `entrypoint` is imported for
two pure functions, exactly as `test_stray_write_paths.py` imports `harness` for
one.

**The filename is deliberately not `test_container_*`**, for the reason
`test_stray_write_paths.py` and `test_capability_promotions.py` both give: that
glob is what `tests/container/context_manifest.txt` ships into the image and what
the marker-gating check enumerates, and a plain-CI module answering to it would
be shipped for nothing and would fail that assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from .container import harness
from .container import entrypoint

#: The one row carrying a pin of its own, and the slug it carries. Retyped here
#: on purpose — this is the assertion, so reading it from the file it is meant to
#: guard would make the module agree with any edit at all.
ROW_ID = "claude-code-apply-native"
ROW_PIN = "anthropic/claude-haiku-4.5"
HOST_PIN = "~deepseek/deepseek-v4-flash-latest"

#: The repo's own skill tree, so `build_host_argv` can read the shipped
#: SKILL.md's `allowed-tools` the way it does inside the image (where the same
#: file sits at /opt/mllm/skill).
SKILL_DIR = harness.HERE.parents[1]


def _matrix() -> dict:
    return harness.load_host_matrix()


def _row(host: str, row_id: str) -> dict:
    for row in (_matrix().get(host) or {}).get("rows") or []:
        if row.get("id") == row_id:
            return row
    raise AssertionError(f"host_matrix.yaml names no row {row_id!r} under {host!r}")


class TestTheResolverPrefersTheNarrowestPin:
    """`pinned_model` resolves env > row > host, and each step is a real case."""

    def test_a_row_without_a_pin_takes_the_hosts(self, monkeypatch):
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)
        assert entrypoint.pinned_model({"model": {"slug": "host/m"}}, {"id": "r"}) \
            == "host/m"

    def test_a_row_with_a_pin_takes_its_own(self, monkeypatch):
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)
        assert entrypoint.pinned_model(
            {"model": {"slug": "host/m"}},
            {"id": "r", "model": {"slug": "row/m"}}) == "row/m"

    def test_an_empty_row_pin_is_not_a_pin(self, monkeypatch):
        """A blank `slug:` must fall through, not drive the host at ``''``."""
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)
        assert entrypoint.pinned_model(
            {"model": {"slug": "host/m"}},
            {"id": "r", "model": {"slug": "   "}}) == "host/m"

    def test_the_env_override_still_outranks_a_row_pin(self, monkeypatch):
        """The escape hatch a red row is debugged with cannot be checked in shut."""
        monkeypatch.setenv("MLLM_HOST_MODEL", "operator/typed")
        assert entrypoint.pinned_model(
            {"model": {"slug": "host/m"}},
            {"id": "r", "model": {"slug": "row/m"}}) == "operator/typed"

    def test_no_row_at_all_is_the_old_behaviour(self, monkeypatch):
        """`row=None` is every caller that predates the row pin."""
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)
        assert entrypoint.pinned_model({"model": {"slug": "host/m"}}) == "host/m"


class TestTheShippedMatrixDeclaresExactlyOneRowPin:
    """The declaration, in both directions: this row has one, no other row does."""

    def test_the_apply_native_row_carries_the_anthropic_fallback(self):
        assert (_row("claude-code", ROW_ID).get("model") or {}).get("slug") == ROW_PIN

    def test_it_is_the_only_row_pin_in_the_matrix(self):
        """The cost decision of 2026-08-14 stands for every other row.

        A second row pin is not forbidden — it is a spend decision that has to be
        made deliberately, with its own measurement beside it, which is what
        editing this list is.
        """
        pinned = sorted(
            (host, row.get("id"))
            for host, entry in _matrix().items()
            for row in (entry.get("rows") or [])
            if (row.get("model") or {}).get("slug"))
        assert pinned == [("claude-code", ROW_ID)], (
            f"rows carrying their own model pin: {pinned}. Each one costs "
            f"Anthropic-tier money on every run of that row; add it here with "
            f"the measurement that justified it.")

    def test_every_host_still_carries_the_cheap_pin(self):
        """The override is per row and must not have crept up to a host block."""
        assert {host: (entry.get("model") or {}).get("slug")
                for host, entry in _matrix().items()} == {
            "claude-code": HOST_PIN, "codex": HOST_PIN, "opencode": HOST_PIN}


class TestTheArgvTheRowReallyLaunches:
    """The resolver is not the payload — the rendered `--model` token is.

    `invoke_argv` substitutes `<model>`, so this is the byte that decides which
    account gets billed, and it is asserted rather than inferred.
    """

    @pytest.fixture(autouse=True)
    def _skill_dir(self, monkeypatch):
        monkeypatch.setattr(entrypoint, "SKILL_DIR", SKILL_DIR)
        monkeypatch.delenv("MLLM_HOST_MODEL", raising=False)

    def _argv(self, row_id: str) -> list[str]:
        entry = _matrix()["claude-code"]
        return entrypoint.build_host_argv(entry, _row("claude-code", row_id))

    def test_the_apply_native_row_launches_on_haiku(self):
        argv = self._argv(ROW_ID)
        assert ["--model", ROW_PIN] == argv[argv.index("--model"):
                                            argv.index("--model") + 2]

    @pytest.mark.parametrize("row_id", ["claude-code-ask", "claude-code-review-plan"])
    def test_its_sibling_rows_are_untouched(self, row_id):
        argv = self._argv(row_id)
        assert ["--model", HOST_PIN] == argv[argv.index("--model"):
                                             argv.index("--model") + 2]


class TestTheRecordSaysWhichSlugRan:
    """C10's cost table reads the row pin too, or it records a model that did not run."""

    @pytest.fixture(autouse=True)
    def _no_daemon(self, monkeypatch):
        """`row_record` reaches for the image's version report; nothing else does.

        Stubbed rather than skipped, so this module keeps its no-daemon promise
        instead of quietly depending on a built image being present.
        """
        monkeypatch.setattr(harness, "cli_version",
                            lambda cli, image=harness.DEFAULT_IMAGE: None)

    def _record(self, row_id: str, out_dir: Path,
                creds: str = "openrouter") -> dict:
        """The record `row_record` derives for one host row of the real matrix.

        The bundle is fabricated and empty — no stub log, no usage, no census —
        because the readback under test is the matrix lookup, and an empty
        directory is what every `_read_json` here is written to survive. `creds`
        defaults to `openrouter` because a pin is only recorded beside a row that
        really took a credential path.
        """
        row = harness.Row(row_id=row_id, lane="host", host="claude-code",
                          creds=creds)
        result = harness.RowResult(row=row, exit_code=0, wall_clock_s=1.0,
                                   out_dir=out_dir)
        return harness.row_record(row, result)

    def test_the_record_helper_resolves_the_row_pin(self, tmp_path):
        """The row's own slug, not the host's, against the row's own cost figure."""
        assert self._record(ROW_ID, tmp_path).get("model") == ROW_PIN

    def test_a_sibling_row_still_records_the_host_pin(self, tmp_path):
        """The other direction: reading the row pin must not leak across rows."""
        assert self._record("claude-code-ask", tmp_path).get("model") == HOST_PIN

    def test_a_row_that_took_no_credential_records_no_model(self, tmp_path):
        """A discovery-only row made no model call; a slug reads as one that ran."""
        assert "model" not in self._record(ROW_ID, tmp_path, creds="none")

    def test_the_matrix_is_loadable_yaml_with_the_pin_in_it(self):
        """Cheap, and it is what a hand-edited 1700-line YAML file gets wrong."""
        document = yaml.safe_load(
            (harness.HERE / "host_matrix.yaml").read_text(encoding="utf-8"))
        assert document["hosts"]["claude-code"]["rows"]
