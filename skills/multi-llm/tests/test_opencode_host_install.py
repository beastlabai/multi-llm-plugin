"""``bump.py install``: the pinned opencode the release gate's row 7 probes.

The gate must probe the host the repository claims to support, so the binary
comes under the bump discipline rather than beside it
(plans/opencode-plugin-publish.md, step 0, "Give the CI probe a pinned OpenCode
binary"): the version is read through ``read_pins()`` and refused unless every
pin agrees; the install is ``opencode-ai`` and its platform packages alone,
copied out of the container lockfile with their ``resolved`` and ``integrity``
values, put down by ``npm ci --ignore-scripts`` under a scratch prefix against
the single registry; and the installed ``--version`` must equal the pin before
anything is handed to the probe.

No install script runs: ``opencode-ai``'s ``postinstall`` executes the binary it
links, and falls back to fetching and executing an unverified one. After ``npm
ci`` and before anything runs, what npm left is checked (``verify_installed``):
npm's own record of the install lists only the subset's packages, from the
single registry with the committed integrity, nothing else is in
``node_modules``, and the binary is a verified platform package's own
``bin/opencode``, the first in the order the ``postinstall`` would have tried
(``platform_candidates``). The wrapper's ``bin/opencode.exe`` and the
``.bin/opencode`` link to it stay the placeholder script and are never used.

Offline: npm and the binary are fakes (the fake ``npm ci`` lays down the tree
the real ``npm ci --ignore-scripts`` does, measured on the pinned release:
every platform package npm keeps for this machine with its ``bin/opencode``,
the wrapper with its placeholder ``bin/opencode.exe``, ``.bin/opencode`` a
symlink to that placeholder, ``node_modules/.package-lock.json`` beside them),
except one POSIX case that drives the real ``_read_version`` against a fake
``opencode`` script.
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


#: What ``opencode-ai``'s ``bin/opencode.exe`` is when its install script did
#: not run (measured): a shell script that says so and exits 1.
PLACEHOLDER = (b'echo "Error: opencode-ai\'s postinstall script was not run." >&2\n'
               b"exit 1\n")


def _platform_binary(prefix, key):
    return (prefix / key / "bin" / bump._platform_binary_name()).absolute()


def _lay_out(prefix, content=b"fake opencode\n", platforms=4):
    """What ``npm ci --ignore-scripts`` of the subset leaves under ``prefix``, as
    the real one does.

    The platform packages npm keeps for this machine (the subset's entries this
    machine's install-script order names, up to ``platforms``), each with its
    own ``bin/opencode`` (only the first one's bytes are ``content``), the
    wrapper with its manifest and its placeholder ``bin/opencode.exe``, the
    ``.bin`` command linking to that placeholder, and npm's own record of the
    install. Returns the first platform package's key: the one to be chosen.
    """
    subset = json.loads((prefix / "package-lock.json").read_text())["packages"]
    modules = prefix / "node_modules"
    shutil.rmtree(modules, ignore_errors=True)  # npm ci starts from no node_modules
    keys = [f"node_modules/{name}" for name in bump.platform_candidates()
            if f"node_modules/{name}" in subset][:platforms]
    assert keys, "the subset carries no platform package for this machine"
    for number, key in enumerate(keys):
        package = prefix / key
        (package / "bin").mkdir(parents=True, exist_ok=True)
        (package / "package.json").write_text(json.dumps({"name": key.split("/", 1)[1]}))
        binary = package / "bin" / bump._platform_binary_name()
        binary.write_bytes(content if number == 0 else content + key.encode())
        binary.chmod(0o755)
    wrapper = prefix / WRAPPER_KEY
    (wrapper / "bin").mkdir(parents=True, exist_ok=True)
    (wrapper / "package.json").write_text(json.dumps(
        {"name": bump.NPM_PACKAGE, "bin": {"opencode": "./bin/opencode.exe"}}))
    (wrapper / "bin" / "opencode.exe").write_bytes(PLACEHOLDER)
    command = modules / ".bin" / ("opencode.cmd" if os.name == "nt" else "opencode")
    command.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        command.write_text("@ECHO off\r\n", encoding="utf-8")
    else:
        command.symlink_to(Path("..") / bump.NPM_PACKAGE / "bin" / "opencode.exe")
    hidden = {"name": bump.INSTALL_NAME, "lockfileVersion": 3,
              "packages": {key: subset[key] for key in [WRAPPER_KEY, *keys]}}
    (modules / ".package-lock.json").write_text(json.dumps(hidden, indent=2))
    return keys[0]


def _fake_npm(calls, **layout):
    """An ``npm ci`` stand-in: records the call, lays the real tree down."""
    def npm_ci(prefix, url):
        calls.append({"prefix": prefix, "registry": url,
                      "manifest": json.loads((prefix / "package.json").read_text()),
                      "lock": json.loads((prefix / "package-lock.json").read_text())})
        _lay_out(prefix, **layout)
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
        calls, read = [], []
        prefix = tmp_path / "prefix"
        assert bump.cmd_install(prefix, npm_ci=_fake_npm(calls),
                                version_reader=lambda binary: read.append(binary) or PIN) == 0
        lines = capsys.readouterr().out.splitlines()
        key = f"node_modules/{next(n for n in bump.platform_candidates() if f'node_modules/{n}' in calls[0]['lock']['packages'])}"
        assert lines == [str(_platform_binary(prefix.absolute(), key))]
        assert Path(lines[-1]).is_absolute()
        # The version is read from the binary handed to the gate, nothing else.
        assert read == lines
        assert len(calls) == 1

    def test_a_mismatched_version_refuses_and_prints_no_path(self, tmp_path, capsys):
        prefix = tmp_path / "prefix"
        with pytest.raises(bump.BumpError, match="not the pin"):
            bump.cmd_install(prefix, npm_ci=_fake_npm([]), version_reader=lambda _b: "0.0.1")
        assert capsys.readouterr().out == ""

    @pytest.mark.skipif(os.name == "nt", reason="a POSIX shell script stands in for the binary")
    def test_the_real_reader_runs_the_binary_in_an_isolated_home(self, tmp_path, capsys):
        seen = tmp_path / "seen.json"
        script = (
            "#!/bin/sh\n"
            f"printf '{{\"home\": \"%s\", \"autoupdate\": \"%s\", \"arg\": \"%s\"}}' "
            f"\"$HOME\" \"$OPENCODE_DISABLE_AUTOUPDATE\" \"$1\" > '{seen}'\n"
            "echo 0.0.1\n").encode()

        def npm_ci(prefix, _url):
            _lay_out(prefix, content=script)

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
        # The project root is the prefix: its cwd (which holds the scratch
        # package.json) and, outright, the command-line prefix.
        assert seen["cwd"] == str(prefix)
        assert f"--prefix={prefix}" in seen["argv"]
        assert "--ignore-scripts" in seen["argv"], (
            "opencode-ai's install script would run a binary before verify_installed")
        assert seen["timeout"] == bump.NPM_CI_BUDGET_S


class TestWhatNpmCiLeft:
    """``verify_installed``: everything installed came from the single registry
    with the committed integrity, and the binary handed to the gate is a
    verified platform package's own file — chosen before anything runs."""

    @staticmethod
    def _prefix(tmp_path, **layout):
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        manifest, scratch = bump.install_subset(LOCK, PIN)
        (prefix / "package-lock.json").write_text(json.dumps(scratch))
        key = _lay_out(prefix, **layout)
        return prefix, scratch, key

    @staticmethod
    def _verify(prefix, scratch, **kwargs):
        return bump.verify_installed(prefix, scratch, registry.single_registry(), **kwargs)

    def test_the_first_candidate_npm_installed_is_the_binary(self, tmp_path):
        prefix, scratch, key = self._prefix(tmp_path)
        matched = self._verify(prefix, scratch)
        assert matched["binary"] == str(_platform_binary(prefix, key))
        assert matched["platform_package"] == key
        assert matched["chosen_from"] == bump.platform_candidates()
        assert len(matched["sha256"]) == 64

    def test_the_wrappers_placeholder_and_its_link_are_never_the_binary(self, tmp_path):
        """With the install script off the wrapper's ``bin/opencode.exe`` is the
        placeholder, and ``.bin/opencode`` links to it; whatever is put there,
        the choice is a platform package's own file."""
        prefix, scratch, key = self._prefix(tmp_path)
        (prefix / WRAPPER_KEY / "bin" / "opencode.exe").write_bytes(b"anything at all\n")
        matched = self._verify(prefix, scratch)
        assert matched["binary"] == str(_platform_binary(prefix, key))
        assert "/.bin/" not in matched["binary"] and WRAPPER_KEY not in matched["binary"]

    def test_the_install_scripts_order_decides(self, tmp_path):
        prefix, scratch, first = self._prefix(tmp_path)
        verified = [k for k in json.loads((prefix / bump.HIDDEN_LOCKFILE).read_text())["packages"]
                    if k != WRAPPER_KEY]
        if len(verified) < 2:
            pytest.skip("this machine's npm keeps one platform package")
        order = [k.split("/", 1)[1] for k in reversed(verified)]
        matched = self._verify(prefix, scratch, candidates=order)
        assert matched["platform_package"] == verified[-1]
        # A candidate without its binary is passed over for the next one.
        _platform_binary(prefix, verified[-1]).unlink()
        assert self._verify(prefix, scratch, candidates=order)["platform_package"] == verified[-2]

    def test_no_installed_candidate_refuses_naming_the_order(self, tmp_path):
        prefix, scratch, _ = self._prefix(tmp_path)
        with pytest.raises(bump.BumpError, match="opencode-plan9-mips") as caught:
            self._verify(prefix, scratch, candidates=["opencode-plan9-mips"])
        assert "Refusing to run anything" in str(caught.value)

    @pytest.mark.skipif(os.name == "nt", reason="a POSIX symlink stands in for the binary")
    def test_a_binary_resolving_outside_its_package_is_refused(self, tmp_path):
        prefix, scratch, key = self._prefix(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.write_bytes(b"not from the registry\n")
        binary = _platform_binary(prefix, key)
        binary.unlink()
        binary.symlink_to(outside)
        with pytest.raises(bump.BumpError, match="outside"):
            self._verify(prefix, scratch)

    def _hidden(self, prefix):
        path = prefix / bump.HIDDEN_LOCKFILE
        return path, json.loads(path.read_text())

    def test_a_package_from_another_registry_is_refused_by_file(self, tmp_path):
        prefix, scratch, key = self._prefix(tmp_path)
        path, hidden = self._hidden(prefix)
        hidden["packages"][key]["resolved"] = "https://registry.example.invalid/x/-/x.tgz"
        path.write_text(json.dumps(hidden))
        with pytest.raises(bump.BumpError, match="not the single registry") as caught:
            self._verify(prefix, scratch)
        assert str(path) in str(caught.value) and key in str(caught.value)

    def test_an_integrity_that_is_not_the_committed_one_is_refused(self, tmp_path):
        prefix, scratch, key = self._prefix(tmp_path)
        path, hidden = self._hidden(prefix)
        hidden["packages"][key]["integrity"] = "sha512-" + "A" * 86 + "=="
        path.write_text(json.dumps(hidden))
        with pytest.raises(bump.BumpError, match="not the committed"):
            self._verify(prefix, scratch)

    def test_a_package_outside_the_subset_is_refused(self, tmp_path):
        prefix, scratch, _ = self._prefix(tmp_path)
        path, hidden = self._hidden(prefix)
        hidden["packages"]["node_modules/left-pad"] = {
            "resolved": f"{registry.single_registry()}/left-pad/-/left-pad-1.0.0.tgz",
            "integrity": "sha512-" + "B" * 86 + "=="}
        path.write_text(json.dumps(hidden))
        with pytest.raises(bump.BumpError, match="not in the committed subset"):
            self._verify(prefix, scratch)

    def test_a_directory_npm_did_not_record_is_refused(self, tmp_path):
        prefix, scratch, _ = self._prefix(tmp_path)
        (prefix / "node_modules" / "opencode-linux-x64-dropped").mkdir()
        with pytest.raises(bump.BumpError, match="does not list"):
            self._verify(prefix, scratch)

    def test_a_platform_package_npm_did_not_record_is_never_chosen(self, tmp_path):
        """The first candidate's directory is there, but npm's record does not
        list it: refused as a stray, not run."""
        prefix, scratch, key = self._prefix(tmp_path)
        path, hidden = self._hidden(prefix)
        del hidden["packages"][key]
        path.write_text(json.dumps(hidden))
        with pytest.raises(bump.BumpError, match="does not list"):
            self._verify(prefix, scratch)

    def test_no_record_of_the_install_is_refused(self, tmp_path):
        prefix, scratch, _ = self._prefix(tmp_path)
        (prefix / bump.HIDDEN_LOCKFILE).unlink()
        with pytest.raises(bump.BumpError, match=r"\.package-lock\.json"):
            self._verify(prefix, scratch)

    def test_install_refuses_before_running_the_binary(self, tmp_path, capsys):
        def npm_ci(prefix, _url):
            key = _lay_out(prefix)
            path = prefix / bump.HIDDEN_LOCKFILE
            hidden = json.loads(path.read_text())
            hidden["packages"][key]["integrity"] = "sha512-" + "C" * 86 + "=="
            path.write_text(json.dumps(hidden))

        with pytest.raises(bump.BumpError, match="not the committed"):
            bump.cmd_install(tmp_path / "prefix", npm_ci=npm_ci, version_reader=_never)
        assert capsys.readouterr().out == ""

    def test_a_committed_url_off_the_registry_refuses_before_npm_runs(self, repo_copy, tmp_path):
        """The static half: ``npm ci`` fetches a lockfile URL on another host
        from that host, ``--registry`` notwithstanding."""
        lock = json.loads((repo_copy / bump.LOCKFILE).read_text())
        lock["packages"][WRAPPER_KEY]["resolved"] = "https://mirror.example.invalid/o.tgz"
        (repo_copy / bump.LOCKFILE).write_text(json.dumps(lock))
        with pytest.raises(bump.BumpError, match="not under the single registry"):
            bump.cmd_install(tmp_path / "prefix", root=repo_copy, npm_ci=_never,
                             version_reader=_never)
        with pytest.raises(bump.BumpError, match="mirror.example.invalid"):
            bump.install_subset(lock, PIN, registry=registry.single_registry())
        assert bump.install_subset(lock, PIN)  # without a registry, the old static check only

    def test_verify_install_checks_a_prefix_npm_ci_already_ran_in(self, tmp_path, capsys):
        prefix, _, key = self._prefix(tmp_path)
        assert bump.main(["verify-install", "--prefix", str(prefix)]) == 0
        out = capsys.readouterr().out.splitlines()
        assert key in out[0] and out[-1] == str(_platform_binary(prefix, key))
        (prefix / "node_modules" / "stray-package").mkdir()
        assert bump.main(["verify-install", "--prefix", str(prefix)]) == 1
        assert "does not list" in capsys.readouterr().err


class TestTheInstallScriptsOrder:
    """``platform_candidates`` restates ``postinstall.mjs``'s ``packageNames()``
    (read from the pinned release's tarball): with the script off, its choice
    is made here."""

    @pytest.mark.parametrize("system, arch, avx2, musl, expected", [
        ("linux", "x86_64", True, False, ["opencode-linux-x64", "opencode-linux-x64-baseline",
                                          "opencode-linux-x64-musl",
                                          "opencode-linux-x64-baseline-musl"]),
        ("linux", "x86_64", False, False, ["opencode-linux-x64-baseline", "opencode-linux-x64",
                                           "opencode-linux-x64-baseline-musl",
                                           "opencode-linux-x64-musl"]),
        ("linux", "x86_64", True, True, ["opencode-linux-x64-musl",
                                         "opencode-linux-x64-baseline-musl",
                                         "opencode-linux-x64", "opencode-linux-x64-baseline"]),
        ("linux", "x86_64", False, True, ["opencode-linux-x64-baseline-musl",
                                          "opencode-linux-x64-musl",
                                          "opencode-linux-x64-baseline", "opencode-linux-x64"]),
        ("linux", "aarch64", False, False, ["opencode-linux-arm64", "opencode-linux-arm64-musl"]),
        ("linux", "aarch64", False, True, ["opencode-linux-arm64-musl", "opencode-linux-arm64"]),
        ("darwin", "x86_64", True, False, ["opencode-darwin-x64", "opencode-darwin-x64-baseline"]),
        ("darwin", "x86_64", False, False, ["opencode-darwin-x64-baseline",
                                            "opencode-darwin-x64"]),
        ("darwin", "arm64", False, False, ["opencode-darwin-arm64"]),
        ("win32", "AMD64", True, False, ["opencode-windows-x64", "opencode-windows-x64-baseline"]),
        ("win32", "ARM64", False, False, ["opencode-windows-arm64"]),
    ])
    def test_the_order_per_machine(self, system, arch, avx2, musl, expected):
        assert bump.platform_candidates(system, arch, avx2=avx2, musl=musl) == expected

    def test_every_candidate_is_a_package_the_lockfile_carries(self):
        """Each machine's first choice exists in the committed subset, so a
        renamed platform package upstream fails here, not on release day."""
        names = {key.split("/", 1)[1] for key in _platform_keys()}
        for system, arch in (("linux", "x86_64"), ("linux", "aarch64"), ("darwin", "x86_64"),
                             ("darwin", "arm64"), ("win32", "AMD64"), ("win32", "ARM64")):
            for avx2 in (True, False):
                for musl in (True, False):
                    order = bump.platform_candidates(system, arch, avx2=avx2, musl=musl)
                    assert set(order) <= names, (system, arch, sorted(set(order) - names))

    def test_this_machine_has_a_choice(self):
        assert bump.platform_candidates()[0].startswith("opencode-")


class TestNoVersionOfItsOwn:
    def test_the_install_path_names_no_release(self):
        """Every opencode version lives in PINS; this path only reads them."""
        for function in (bump.pinned_version, bump.install_subset, bump.platform_candidates,
                         bump._npm_ci, bump.cmd_install, bump.verify_installed,
                         bump.cmd_verify_install):
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
