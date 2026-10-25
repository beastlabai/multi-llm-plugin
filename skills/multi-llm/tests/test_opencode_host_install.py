"""``bump.py install``: the pinned opencode the release gate's row 7 probes.

The gate must probe the host the repository claims to support, so the binary
comes under the bump discipline rather than beside it
(plans/opencode-plugin-publish.md, step 0, "Give the CI probe a pinned OpenCode
binary"): the version is read through ``read_pins()`` and refused unless every
pin agrees; the install is ``opencode-ai`` and its platform packages alone,
copied out of the container lockfile with their ``resolved`` and ``integrity``
values, put down by ``npm ci`` under a scratch prefix against the single
registry; and the installed ``--version`` must equal the pin before anything is
handed to the probe.

Offline: npm and the binary are fakes, except one POSIX case that drives the
real ``_read_version`` against a fake ``opencode`` script.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGING = REPO / "packaging" / "opencode"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bump = _load("opencode_bump_install", PACKAGING / "bump.py")
registry = _load("opencode_registry_install", PACKAGING / "registry.py")

LOCK = json.loads((REPO / bump.LOCKFILE).read_text(encoding="utf-8"))
WRAPPER_KEY = f"node_modules/{bump.NPM_PACKAGE}"
PIN = bump.pinned_version(REPO)


def _platform_keys(lock=LOCK):
    wrapper = lock["packages"][WRAPPER_KEY]
    return {f"node_modules/{name}" for name in wrapper["optionalDependencies"]}


@pytest.fixture
def repo_copy(tmp_path):
    """Every file :func:`read_pins` reads, copied: a repo root to break pins in."""
    root = tmp_path / "repo"
    for rel in {rel for _, rel, _ in bump.PINS} | {bump.LOCKFILE}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    return root


def _fake_npm(calls):
    """An ``npm ci`` stand-in: records the call, leaves an ``opencode`` behind."""
    def npm_ci(prefix, url):
        calls.append({"prefix": prefix, "registry": url,
                      "manifest": json.loads((prefix / "package.json").read_text()),
                      "lock": json.loads((prefix / "package-lock.json").read_text())})
        binary = bump.installed_binary(prefix)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("fake", encoding="utf-8")
    return npm_ci


def _never(*_args, **_kwargs):
    raise AssertionError("must not be reached")


class TestTheSubset:
    def test_it_holds_exactly_the_wrapper_and_its_platform_packages(self):
        _, scratch = bump.install_subset(LOCK, PIN)
        keys = set(scratch["packages"]) - {""}
        assert keys == {WRAPPER_KEY} | _platform_keys()
        assert len(keys) > 2, "opencode-ai ships per-platform binaries"
        # ...and the container lockfile carries much more, which stays out.
        assert len(set(LOCK["packages"]) - {""}) > len(keys)

    def test_every_entry_is_the_committed_one_with_its_integrity(self):
        _, scratch = bump.install_subset(LOCK, PIN)
        single = registry.single_registry()
        for key in set(scratch["packages"]) - {""}:
            entry = scratch["packages"][key]
            assert entry == LOCK["packages"][key], key
            assert entry["integrity"].startswith("sha512-"), key
            assert entry["resolved"].startswith(single + "/"), key

    def test_the_manifest_names_the_pin_and_nothing_else(self):
        manifest, scratch = bump.install_subset(LOCK, PIN)
        assert manifest["dependencies"] == {bump.NPM_PACKAGE: PIN}
        assert scratch["packages"][""]["dependencies"] == {bump.NPM_PACKAGE: PIN}
        assert scratch["lockfileVersion"] == LOCK["lockfileVersion"]

    def test_a_wrapper_off_the_pin_refuses(self):
        with pytest.raises(bump.BumpError, match="not the pin"):
            bump.install_subset(LOCK, "0.0.0")

    def test_a_platform_package_the_lockfile_does_not_carry_refuses(self):
        lock = json.loads(json.dumps(LOCK))
        missing = sorted(_platform_keys(lock))[0]
        del lock["packages"][missing]
        with pytest.raises(bump.BumpError, match=re.escape(missing)):
            bump.install_subset(lock, PIN)

    def test_an_entry_without_integrity_refuses(self):
        lock = json.loads(json.dumps(LOCK))
        lock["packages"][WRAPPER_KEY].pop("integrity")
        with pytest.raises(bump.BumpError, match="integrity"):
            bump.install_subset(lock, PIN)

    def test_a_wrapper_with_a_hard_dependency_refuses(self):
        lock = json.loads(json.dumps(LOCK))
        lock["packages"][WRAPPER_KEY]["dependencies"] = {"left-pad": "1.0.0"}
        with pytest.raises(bump.BumpError, match="hard dependencies"):
            bump.install_subset(lock, PIN)


class TestEveryPinMustAgree:
    def test_agreeing_pins_name_one_release(self, repo_copy):
        assert bump.pinned_version(repo_copy) == PIN

    def test_a_disagreeing_pin_refuses_before_npm_runs(self, repo_copy, tmp_path):
        name, rel, pattern = next(p for p in bump.PINS if p[0] == "mock --version")
        text = (repo_copy / rel).read_text(encoding="utf-8")
        match = re.search(pattern, text, re.M)
        (repo_copy / rel).write_text(text[:match.start(1)] + "0.0.1" + text[match.end(1):],
                                     encoding="utf-8")
        prefix = tmp_path / "prefix"
        with pytest.raises(bump.BumpError, match="disagree") as caught:
            bump.cmd_install(prefix, root=repo_copy, npm_ci=_never, version_reader=_never)
        assert name in str(caught.value)
        assert not prefix.exists()


class TestTheVersionAssertion:
    def test_the_pin_prints_the_path_and_only_the_path(self, tmp_path, capsys):
        calls = []
        prefix = tmp_path / "prefix"
        assert bump.cmd_install(prefix, npm_ci=_fake_npm(calls),
                                version_reader=lambda _binary: PIN) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines == [str(bump.installed_binary(prefix))]
        assert Path(lines[-1]).is_absolute()
        assert len(calls) == 1

    def test_a_mismatched_version_refuses_and_prints_no_path(self, tmp_path, capsys):
        prefix = tmp_path / "prefix"
        with pytest.raises(bump.BumpError, match="not the pin"):
            bump.cmd_install(prefix, npm_ci=_fake_npm([]), version_reader=lambda _b: "0.0.1")
        assert capsys.readouterr().out == ""

    @pytest.mark.skipif(os.name == "nt", reason="a POSIX shell script stands in for the binary")
    def test_the_real_reader_runs_the_binary_in_an_isolated_home(self, tmp_path, capsys):
        seen = tmp_path / "seen.json"

        def npm_ci(prefix, _url):
            binary = bump.installed_binary(prefix)
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(
                "#!/bin/sh\n"
                f"printf '{{\"home\": \"%s\", \"autoupdate\": \"%s\", \"arg\": \"%s\"}}' "
                f"\"$HOME\" \"$OPENCODE_DISABLE_AUTOUPDATE\" \"$1\" > '{seen}'\n"
                "echo 0.0.1\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

        with pytest.raises(bump.BumpError, match="'0.0.1'"):
            bump.cmd_install(tmp_path / "prefix", npm_ci=npm_ci)
        observed = json.loads(seen.read_text(encoding="utf-8"))
        assert observed["arg"] == "--version"
        assert observed["autoupdate"] == "1"
        assert os.path.realpath(observed["home"]) != os.path.realpath(os.path.expanduser("~"))
        assert capsys.readouterr().out == ""


class TestTheScratchPrefix:
    def test_npm_sees_the_subset_and_the_single_registry(self, tmp_path):
        calls = []
        bump.cmd_install(tmp_path / "prefix", npm_ci=_fake_npm(calls), version_reader=lambda _b: PIN)
        manifest, scratch = bump.install_subset(LOCK, PIN)
        assert calls[0]["manifest"] == manifest
        assert calls[0]["lock"] == scratch
        assert calls[0]["registry"] == registry.single_registry()

    def test_a_registry_other_than_the_single_one_refuses(self, tmp_path):
        with pytest.raises(bump.BumpError, match="single registry"):
            bump.cmd_install(tmp_path / "prefix", registry="https://registry.example.invalid",
                             npm_ci=_never, version_reader=_never)

    def test_a_prefix_holding_someone_elses_project_refuses(self, tmp_path):
        prefix = tmp_path / "project"
        prefix.mkdir()
        (prefix / "package.json").write_text('{"name": "theirs"}', encoding="utf-8")
        with pytest.raises(bump.BumpError, match="did not write"):
            bump.cmd_install(prefix, npm_ci=_never, version_reader=_never)
        assert json.loads((prefix / "package.json").read_text()) == {"name": "theirs"}

    def test_a_prefix_this_command_used_before_is_reused(self, tmp_path):
        prefix = tmp_path / "prefix"
        for _ in range(2):
            assert bump.cmd_install(prefix, npm_ci=_fake_npm([]), version_reader=lambda _b: PIN) == 0

    def test_npm_ci_gets_nothing_of_the_callers_npm_configuration(self, tmp_path, monkeypatch):
        seen = {}

        def run(argv, **kwargs):
            seen["argv"] = argv
            seen.update(kwargs)

        monkeypatch.setenv("npm_config_registry", "http://127.0.0.1:9/")
        monkeypatch.setenv("NPM_CONFIG_IGNORE_SCRIPTS", "true")
        monkeypatch.setattr(bump.subprocess, "run", run)
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        bump._npm_ci(prefix, registry.single_registry())
        assert seen["argv"][:2] == ["npm", "ci"]
        assert f"--registry={registry.single_registry()}" in seen["argv"]
        userconfig = next(a for a in seen["argv"] if a.startswith("--userconfig="))
        assert Path(userconfig.split("=", 1)[1]).read_text() == ""
        assert not [k for k in seen["env"] if k.lower().startswith("npm_config_")]
        assert seen["cwd"] == str(prefix)
        assert "--ignore-scripts" not in seen["argv"], "the install script links the binary"
        assert seen["timeout"] == bump.NPM_CI_BUDGET_S


class TestNoVersionOfItsOwn:
    def test_the_install_path_names_no_release(self):
        """Every opencode version lives in PINS; this path only reads them."""
        for function in (bump.pinned_version, bump.install_subset, bump.installed_binary,
                         bump._npm_ci, bump.cmd_install):
            source = inspect.getsource(function)
            assert not re.search(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])", source), function.__name__


class TestTheCommandLine:
    def test_prefix_is_required(self):
        with pytest.raises(SystemExit):
            bump.main(["install"])

    def test_a_refusal_is_exit_1_with_no_stdout(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(bump, "_npm_ci", _never)
        assert bump.main(["install", "--prefix", str(tmp_path / "p"),
                          "--registry", "https://registry.example.invalid"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "single registry" in captured.err
