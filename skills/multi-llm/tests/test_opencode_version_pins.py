"""opencode supports its newest release only, and every pin of it agrees.

The policy (2026-10-02): the newest opencode release is the adapter's
``min_version`` **and** ``verified_version``, and that one version is also the
package's ``engines.opencode`` floor and README, the container's ``opencode-ai``
pin (``package.json``, ``package-lock.json``, ``cli-versions.lock``), the mock
CLI's ``--version`` and the name of every committed opencode capture.
``make opencode-bump VERSION=<x.y.z>`` (``packaging/opencode/bump.py``) moves
them all at once; this file fails the build when any one did not move, so a
bump cannot half-land. ``host_matrix.yaml`` carries no opencode pin (its
``measured.cli_version`` cells record what past runs ran), so it is not a site.

Runtime, after a bump: on the pin no version line; above it (a release newer
than the commit) the one ``above-verified`` warning; below it the exit-3
refusal naming the pin and ``opencode upgrade <pin>``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests import opencode_fixtures as oc_pin  # noqa: E402
from utils.providers.base import (  # noqa: E402
    VERSION_ABOVE_VERIFIED,
    VERSION_OK,
    VERSION_TOO_LOW,
    version_verdict,
)
from utils.providers.opencode import OpenCodeProvider  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BUMP_PATH = REPO / "packaging" / "opencode" / "bump.py"


def _load_bump():
    spec = importlib.util.spec_from_file_location("opencode_bump", BUMP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump = _load_bump()


class TestEveryPinNamesOneRelease:
    def test_every_pin_agrees(self):
        pins = bump.read_pins(REPO)
        assert len(set(pins.values())) == 1, (
            "the opencode pins disagree; move them together with "
            "`make opencode-bump VERSION=<x.y.z>`:\n"
            + "\n".join(f"  {name}: {value}" for name, value in sorted(pins.items()))
        )

    def test_the_floor_is_the_capture_release(self):
        assert OpenCodeProvider.min_version == OpenCodeProvider.verified_version
        assert OpenCodeProvider.CONTINUE_LOOP_ON_DENY_RANGE == (
            OpenCodeProvider.min_version, OpenCodeProvider.verified_version,
        )

    def test_the_reader_reads_the_class_the_runtime_imports(self):
        """The bump's text reader and the imported class are one fact."""
        pins = bump.read_pins(REPO)
        assert pins["adapter min_version"] == OpenCodeProvider.min_version
        assert pins["adapter verified_version"] == OpenCodeProvider.verified_version

    def test_the_committed_captures_are_the_pins_and_only_the_pins(self):
        pin = OpenCodeProvider.verified_version
        capture_dirs = sorted(
            p.name for p in (oc_pin.ROOT / "capture").iterdir()
            if p.is_dir() and bump.VERSION_RE.fullmatch(p.name)
        )
        assert capture_dirs == [pin]
        assert (oc_pin.CAPTURE / "summary.tsv").is_file()
        recorded = json.loads(oc_pin.capture("run_review", ".config.json").read_text("utf-8"))
        assert recorded["opencode_version"] == pin
        for path in (oc_pin.AUTH_LIST, oc_pin.DEBUG_CONFIG):
            assert json.loads(path.read_text("utf-8"))["opencode_version"] == pin, path.name
        probes = sorted(p.name for p in oc_pin.ROOT.glob("*.json"))
        assert probes == sorted([oc_pin.AUTH_LIST.name, oc_pin.DEBUG_CONFIG.name])


class TestTheRuntimeVerdictAroundThePin:
    PIN = OpenCodeProvider.verified_version

    def _bumped(self, delta: int) -> str:
        major, minor, patch = (int(part) for part in self.PIN.split("."))
        return f"{major}.{minor}.{patch + delta}" if patch + delta >= 0 else f"{major}.{minor - 1}.99"

    def test_the_release_every_pin_names_says_nothing(self):
        """The container installs its pin; on it no version line prints."""
        for name, version in bump.read_pins(REPO).items():
            assert version_verdict(OpenCodeProvider(), version).state == VERSION_OK, name

    def test_a_newer_release_warns_and_still_dispatches(self):
        verdict = version_verdict(OpenCodeProvider(), self._bumped(1))
        assert verdict.state == VERSION_ABOVE_VERIFIED
        assert not verdict.refuses
        assert "best-effort" in verdict.detail and self.PIN in verdict.detail

    def test_an_older_release_is_refused_naming_the_pin_and_the_upgrade(self):
        older = self._bumped(-1)
        verdict = version_verdict(OpenCodeProvider(), older)
        assert verdict.state == VERSION_TOO_LOW
        assert verdict.refuses
        assert f"opencode {older} is older than the minimum" in verdict.detail
        assert f"Upgrade opencode to >= {self.PIN} (run: opencode upgrade {self.PIN})." in (
            verdict.detail
        )

    def test_the_availability_label_names_the_upgrade_too(self, monkeypatch):
        provider = OpenCodeProvider()
        monkeypatch.setattr(provider, "is_available", lambda: True)
        report = provider.describe_availability(version=self._bumped(-1))
        assert f"opencode upgrade {self.PIN}" in report.detail


def _scratch_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {rel for _name, rel, _pattern in bump.PINS} | {bump.LOCKFILE}
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    return root


def _fake_lock(root: Path, version: str):
    manifest = json.loads((root / bump.CONTAINER / "package.json").read_text("utf-8"))
    manifest["dependencies"][bump.NPM_PACKAGE] = version
    lock = json.loads((root / bump.LOCKFILE).read_text("utf-8"))
    lock["packages"][""]["dependencies"][bump.NPM_PACKAGE] = version
    lock["packages"][f"node_modules/{bump.NPM_PACKAGE}"]["version"] = version
    return json.dumps(manifest, indent=2) + "\n", json.dumps(lock, indent=2) + "\n"


class TestTheBumpMovesEveryPin:
    def test_set_moves_every_site_the_check_reads(self, tmp_path, capsys):
        """The writer's table is the reader's: nothing the test checks is left behind."""
        root = _scratch_repo(tmp_path)
        captured, rendered = [], []
        code = bump.cmd_set(
            "9.9.9", None, root=root,
            lock_resolver=_fake_lock,
            capture=lambda r, binary, version, scratch: captured.append((binary, version)),
            finder=lambda version, scratch, given: "/fake/opencode",
            render=rendered.append,
        )
        assert code == 0
        assert rendered == [root]
        assert captured == [("/fake/opencode", "9.9.9")]
        assert set(bump.read_pins(root).values()) == {"9.9.9"}

    def test_nothing_is_written_when_a_network_step_fails(self, tmp_path):
        root = _scratch_repo(tmp_path)
        before = bump.read_pins(root)

        def refuse(root, version):
            raise bump.BumpError("npm moved packages other than opencode's own")

        with pytest.raises(bump.BumpError):
            bump.cmd_set(
                "9.9.9", None, root=root, lock_resolver=refuse,
                capture=lambda *a: pytest.fail("captured before the lock resolved"),
                finder=lambda version, scratch, given: "/fake/opencode",
                render=lambda root: pytest.fail("rendered before the lock resolved"),
            )
        assert bump.read_pins(root) == before

    def test_a_malformed_version_is_refused(self, tmp_path):
        with pytest.raises(bump.BumpError):
            bump.cmd_set("1.18", None, root=_scratch_repo(tmp_path))

    def test_the_lock_resolution_refuses_a_foreign_move(self):
        assert bump._own_package("node_modules/opencode-ai")
        assert bump._own_package("node_modules/opencode-linux-x64")
        assert not bump._own_package("node_modules/@opencode-ai/sdk")
        assert not bump._own_package("node_modules/cline")


class TestTheLatestCheck:
    @pytest.fixture
    def npm(self, monkeypatch):
        answer = {}

        def fake_run(argv, **kwargs):
            assert argv[:3] == ["npm", "view", bump.NPM_PACKAGE]
            return subprocess.CompletedProcess(
                argv, answer.get("code", 0), answer.get("stdout", ""), answer.get("stderr", "")
            )

        monkeypatch.setattr(bump.subprocess, "run", fake_run)
        return answer

    def test_the_pin_is_latest(self, npm, capsys):
        npm["stdout"] = OpenCodeProvider.verified_version + "\n"
        assert bump.cmd_check(REPO) == 0

    def test_a_newer_release_fails_and_names_the_bump(self, npm, capsys):
        npm["stdout"] = "99.0.0\n"
        assert bump.cmd_check(REPO) == 1
        assert "make opencode-bump VERSION=99.0.0" in capsys.readouterr().out

    def test_an_unanswered_npm_is_neither(self, npm, capsys):
        npm.update(code=1, stdout="", stderr="npm ERR! network")
        assert bump.cmd_check(REPO) == 2
