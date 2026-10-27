"""Row 7's probe (``packaging/opencode/probe.py``), offline.

The probe itself needs the pinned ``opencode``, the network and a packed tarball,
so it runs in the release gate and never here. What this file grades is
everything that decides whether a green row 7 means what it says:

* isolation: no inherited ``XDG_*``, ``OPENCODE_*``, npm or uv setting reaches a
  child, and the home every child sees is under ``probe_root``, never the real
  one;
* the refusals: a registry other than the single one, and an ``opencode`` that
  answers another ``--version`` than the pin, both refuse before the probe runs,
  and both still write the evidence;
* the evidence's shape on a pass and on a failure, and a no-commit project that
  fails as strictly as a committed one would;
* ``plugin[]`` only ever names the extracted directory, never a ``.tgz``;
* the readers: ``debug config``'s command check, the two-sided tool-call
  assessment, the lockfile origin read, the bounded runner's session kill, and
  the generated stub scenario against the real stub server on loopback.

Fakes stand in for OpenCode throughout (a fake ``opencode`` script for the
version refusal on POSIX). No model, no network beyond 127.0.0.1, no spend.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE_PATH = REPO / "packaging" / "opencode" / "probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("opencode_probe_row7", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()
build = probe._sibling("build")
registry = probe._sibling("registry")
SINGLE = registry.single_registry()
SURFACE = build.cli_surface()
TOOL = SURFACE["tool_name"]

#: The keys the contract's row-7 evidence requires
#: (plans/opencode-plugin-publish.impl-contract.md, "Evidence shapes").
TOP_KEYS = {"row", "pass", "opencode_version", "registry", "tarball", "probe_root",
            "isolation", "projects", "failures"}
PROJECT_KEYS = {"kind", "command_registered", "command_skill_dir", "command_inside_payload",
                "tool_registered", "init", "status"}
CALL_KEYS = {"ok", "exit", "result"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _fake_package(root: Path) -> Path:
    """An extracted package directory carrying the real generated surface and body."""
    package = root / "package"
    (package / "dist").mkdir(parents=True)
    (package / "skills" / "multi-llm").mkdir(parents=True)
    (package / "package.json").write_text('{"name": "opencode-multi-llm"}\n', encoding="utf-8")
    (package / "dist" / "command.md").write_text(build.command_body(), encoding="utf-8")
    (package / "dist" / "cli-surface.json").write_text(json.dumps(SURFACE), encoding="utf-8")
    return package


def _tarball(tmp_path: Path) -> Path:
    """Files only, every one under ``package/``: what ``npm pack`` writes."""
    source = _fake_package(tmp_path / "src")
    path = tmp_path / "opencode-multi-llm-0.0.0.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for file in sorted(p for p in source.rglob("*") if p.is_file()):
            archive.add(file, arcname="package/" + file.relative_to(source).as_posix())
    return path


def _fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "opencode"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("not run\n", encoding="utf-8")
    return binary


def _never(*_args, **_kwargs):
    raise AssertionError("the probe ran past a refusal")


def _passing_call(kind: str, skill_dir: str, mode: str) -> dict:
    result = {"mode": mode, "phase": None, "state": "done" if mode == "init" else "nothing-to-do",
              "exit": 0 if mode == "init" else 2, "artifacts": [], "next": None}
    call = {"ok": True, "exit": result["exit"], "state": result["state"], "result": result,
            "skill_dir": skill_dir, "script": str(Path(skill_dir) / "multi_llm.py")}
    if mode == "init":
        call["files_created"] = [probe.PROVIDERS_FILE]
    return call


def _fake_project(fail_no_commit: bool = False):
    def project(kind, root, iso, package_dir, binary, surface, diag):
        skill = probe.skill_dir_of(package_dir)
        entry = {"kind": kind, "project": str(Path(root) / f"project-{kind}"),
                 "command_registered": True, "command_skill_dir": skill,
                 "command_inside_payload": True, "tool_registered": True,
                 "init": _passing_call(kind, skill, "init"),
                 "status": _passing_call(kind, skill, "status"), "ok": True}
        failures = []
        if fail_no_commit and kind == "no-commit":
            entry["status"] = {"ok": False, "exit": 1, "state": "error", "result": None,
                               "error": "GitError: git rev-parse HEAD failed"}
            entry["ok"] = False
            failures = ["no-commit: status: the dispatcher exited 1, not 2"]
        return entry, failures
    return project


def _steps(**overrides):
    pin = "9.8.7"
    base = dict(
        read_version=lambda _binary: pin,
        pinned_version=lambda: pin,
        tarball_facts=lambda path: {"path": str(path), "integrity": "sha512-fake",
                                    "digest": "d" * 64, "sha256": "s" * 64},
        discover=lambda binary, root: [{"setting": "npm_config_registry", "honoured": True,
                                        "requests": 1, "fell_back_to": [], "elapsed_s": 0.1}],
        project=_fake_project(),
        installed_from=lambda dirs, cache, reg: {"lockfiles": ["config/opencode/package-lock.json"],
                                                 "packages": 3, "origins": [reg], "foreign": []},
    )
    base.update(overrides)
    return probe.Steps(**base)


def _run(tmp_path, *, registry_url=SINGLE, steps=None, keep=False):
    out = tmp_path / "evidence" / "row-7.json"
    code = probe.row7(_tarball(tmp_path), str(_fake_binary(tmp_path)), registry_url, out,
                      keep=keep, steps=steps or _steps(), tmp_parent=tmp_path)
    return code, json.loads(out.read_text(encoding="utf-8"))


#: The builder's virtualenv, in the spelling this OS gives its scripts dir.
VENV_BIN = os.path.join("/venv", "Scripts" if os.name == "nt" else "bin")
MAINTAINER = {
    "PATH": os.pathsep.join([VENV_BIN, "/usr/bin", "/home/maint/.opencode/bin"]),
    "HOME": "/home/maint",
    "VIRTUAL_ENV": "/venv",
    "XDG_CONFIG_HOME": "/home/maint/.config",
    "XDG_DATA_HOME": "/home/maint/.local/share",
    "XDG_CACHE_HOME": "/home/maint/.cache",
    "XDG_STATE_HOME": "/home/maint/.local/state",
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "OPENCODE_CONFIG": "/home/maint/opencode.json",
    "OPENCODE_CONFIG_DIR": "/home/maint/.opencode-alt",
    "OPENCODE_CONFIG_CONTENT": '{"plugin": ["/home/maint/other-plugin"]}',
    "OPENCODE_DISABLE_AUTOUPDATE": "0",
    "npm_config_registry": "http://127.0.0.1:9/",
    "NPM_CONFIG_USERCONFIG": "/home/maint/.npmrc",
    "UV_CACHE_DIR": "/home/maint/.cache/uv",
    "UV_INDEX_URL": "http://127.0.0.1:9/simple",
    "MULTI_LLM_PROVIDERS_CONFIG": "/home/maint/providers.yaml",
    "LANG": "C.UTF-8",
    "SOME_UNRELATED_TOKEN": "keep-me-out",
}


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #

class TestIsolation:
    def _iso(self, tmp_path, **kwargs):
        return probe.isolated_env(tmp_path / "env", binary=str(tmp_path / "pin" / "opencode"),
                                  registry=kwargs.pop("registry", SINGLE),
                                  inherited=kwargs.pop("inherited", MAINTAINER))

    def test_no_inherited_xdg_or_opencode_value_reaches_a_child(self, tmp_path):
        env = self._iso(tmp_path).env
        for name, value in MAINTAINER.items():
            if name in ("PATH", "LANG"):
                continue
            assert env.get(name) != value, name
        for name in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT",
                     "VIRTUAL_ENV", "UV_INDEX_URL", "MULTI_LLM_PROVIDERS_CONFIG",
                     "NPM_CONFIG_USERCONFIG", "SOME_UNRELATED_TOKEN"):
            assert name not in env, name
        everything = json.dumps({k: v for k, v in env.items() if k != "PATH"})
        assert "/home/maint" not in everything
        assert "127.0.0.1:9" not in everything

    def test_the_only_opencode_settings_are_the_probes_own(self, tmp_path):
        env = self._iso(tmp_path).env
        assert {k for k in env if k.upper().startswith("OPENCODE")} == {
            "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_MODELS_FETCH"}
        assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"

    def test_every_config_selecting_name_is_recorded_and_no_value_is(self, tmp_path):
        iso = self._iso(tmp_path)
        cleared = set(iso.cleared)
        assert {name for name in MAINTAINER if name.startswith(("XDG_", "OPENCODE_"))} <= cleared
        assert {"HOME", "VIRTUAL_ENV", "npm_config_registry", "NPM_CONFIG_USERCONFIG",
                "UV_CACHE_DIR", "UV_INDEX_URL", "MULTI_LLM_PROVIDERS_CONFIG"} <= cleared
        assert "SOME_UNRELATED_TOKEN" not in cleared
        assert all("/" not in name for name in cleared)

    def test_home_and_every_xdg_dir_are_under_the_probe_root(self, tmp_path):
        iso = self._iso(tmp_path)
        root = tmp_path / "env"
        assert iso.home == root / "home"
        assert os.path.realpath(iso.env["HOME"]) != os.path.realpath(probe.real_home())
        for name, value in iso.env.items():
            if name.startswith("XDG_") or name in ("HOME", "TMPDIR", "UV_CACHE_DIR") or (
                    name.startswith("npm_config_") and name not in (
                        "npm_config_registry", "npm_config_update_notifier")):
                assert Path(value).is_relative_to(root), name

    def test_the_real_home_is_never_used_even_unset(self, tmp_path):
        env = self._iso(tmp_path, inherited={"PATH": "/usr/bin"}).env
        assert env["HOME"] == str(tmp_path / "env" / "home")

    def test_the_pinned_binary_leads_path_and_the_builders_venv_is_gone(self, tmp_path):
        env = self._iso(tmp_path).env
        entries = env["PATH"].split(os.pathsep)
        assert entries[0] == os.path.abspath(str(tmp_path / "pin"))
        assert os.path.abspath(VENV_BIN) not in [os.path.abspath(e) for e in entries]
        assert "/usr/bin" in entries

    def test_the_registry_is_carried_only_when_named(self, tmp_path):
        assert self._iso(tmp_path).env["npm_config_registry"] == SINGLE
        assert "npm_config_registry" not in self._iso(tmp_path, registry=None).env

    def test_npm_and_uv_state_is_private_and_npm_config_files_are_empty(self, tmp_path):
        iso = self._iso(tmp_path)
        assert not (tmp_path / "env").exists(), "building the environment creates nothing"
        probe.materialize(iso)
        for name in ("npm_config_userconfig", "npm_config_globalconfig"):
            assert Path(iso.env[name]).read_text() == ""
        assert Path(iso.env["UV_CACHE_DIR"]).is_dir()
        assert not any(Path(iso.env["UV_CACHE_DIR"]).iterdir()), "uv starts cold"

    def test_a_proxy_is_never_asked_for_the_loopback_stub(self, tmp_path):
        env = self._iso(tmp_path, inherited={"PATH": "/usr/bin",
                                             "HTTPS_PROXY": "http://proxy:3128"}).env
        assert env["HTTPS_PROXY"] == "http://proxy:3128"
        assert "127.0.0.1" in env["NO_PROXY"].split(",")
        assert "127.0.0.1" in env["no_proxy"].split(",")


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

class TestTheRegistryRefusal:
    def test_a_registry_other_than_the_single_one_refuses_before_anything(self, tmp_path):
        code, evidence = _run(tmp_path, registry_url="https://registry.example.invalid",
                              steps=_steps(read_version=_never, pinned_version=_never,
                                           tarball_facts=_never, discover=_never,
                                           project=_never, installed_from=_never))
        assert code == 1
        assert evidence["pass"] is False
        assert len(evidence["failures"]) == 1
        assert SINGLE in evidence["failures"][0]
        assert "registry.example.invalid" in evidence["failures"][0]
        assert evidence["projects"] == [] and evidence["probe_root"] is None

    def test_the_trailing_slash_spelling_is_the_same_registry(self, tmp_path):
        code, evidence = _run(tmp_path, registry_url=SINGLE + "/")
        assert code == 0
        assert evidence["registry"] == SINGLE


class TestTheVersionRefusal:
    def test_a_mismatched_version_refuses_before_the_probe_runs(self, tmp_path):
        code, evidence = _run(tmp_path, steps=_steps(
            read_version=lambda _b: "0.0.1", pinned_version=lambda: "9.8.7",
            tarball_facts=_never, discover=_never, project=_never, installed_from=_never))
        assert code == 1
        assert evidence["opencode_version"] == "0.0.1"
        assert "'0.0.1'" in evidence["failures"][0] and "9.8.7" in evidence["failures"][0]
        assert "before the probe runs" in evidence["failures"][0]
        assert evidence["probe_root"] is None and evidence["projects"] == []

    @pytest.mark.skipif(os.name == "nt", reason="a POSIX shell script stands in for the binary")
    def test_a_fake_opencode_answering_another_version_is_refused(self, tmp_path):
        """The real reader (``bump._read_version``) against a fake binary."""
        binary = tmp_path / "fake" / "opencode"
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\necho 0.0.1\n", encoding="utf-8")
        binary.chmod(0o755)
        out = tmp_path / "row-7.json"
        steps = probe.Steps(pinned_version=lambda: "9.8.7", tarball_facts=_never,
                            discover=_never, project=_never, installed_from=_never)
        assert probe.row7(_tarball(tmp_path), str(binary), SINGLE, out, steps=steps,
                          tmp_parent=tmp_path) == 1
        evidence = json.loads(out.read_text())
        assert evidence["opencode_version"] == "0.0.1"
        assert "not the pin 9.8.7" in evidence["failures"][0]

    def test_disagreeing_pins_refuse(self, tmp_path):
        def disagree():
            raise RuntimeError("the opencode pins disagree")
        code, evidence = _run(tmp_path, steps=_steps(pinned_version=disagree, read_version=_never))
        assert code == 1
        assert "disagree" in evidence["failures"][0]

    def test_a_missing_binary_refuses(self, tmp_path):
        out = tmp_path / "row-7.json"
        assert probe.row7(_tarball(tmp_path), str(tmp_path / "nowhere" / "opencode"), SINGLE, out,
                          steps=_steps(read_version=_never), tmp_parent=tmp_path) == 1
        assert "is not a file" in json.loads(out.read_text())["failures"][0]


# --------------------------------------------------------------------------- #
# The evidence
# --------------------------------------------------------------------------- #

class TestTheEvidence:
    def test_the_pass_shape(self, tmp_path):
        code, evidence = _run(tmp_path)
        assert code == 0
        assert TOP_KEYS <= set(evidence)
        assert evidence["row"] == "7" and evidence["pass"] is True and evidence["failures"] == []
        assert evidence["opencode_version"] == "9.8.7"
        assert evidence["registry"] == SINGLE
        assert set(evidence["tarball"]) >= {"path", "integrity", "digest"}
        assert {"env_cleared", "home"} <= set(evidence["isolation"])
        assert [p["kind"] for p in evidence["projects"]] == list(probe.PROJECT_KINDS)
        for project in evidence["projects"]:
            assert PROJECT_KEYS <= set(project)
            for call in ("init", "status"):
                assert CALL_KEYS <= set(project[call])
            assert "files_created" in project["init"]
        assert evidence["isolation"]["registry_setting"] == "npm_config_registry"
        assert evidence["installed_from"]["foreign"] == []
        assert not Path(evidence["probe_root"]).exists(), "a pass without --keep cleans up"

    def test_keep_keeps_the_probe_root(self, tmp_path):
        _, evidence = _run(tmp_path, keep=True)
        assert Path(evidence["probe_root"]).is_dir()

    def test_the_failure_shape_and_the_no_commit_project_stays_strict(self, tmp_path, capsys):
        code, evidence = _run(tmp_path, steps=_steps(project=_fake_project(fail_no_commit=True)))
        assert code == 1
        assert TOP_KEYS <= set(evidence)
        assert evidence["pass"] is False
        assert evidence["failures"] == ["no-commit: status: the dispatcher exited 1, not 2"]
        assert [p["ok"] for p in evidence["projects"]] == [True, False]
        assert Path(evidence["probe_root"]).is_dir(), "diagnostics survive a failure"
        assert str(Path(evidence["probe_root"]) / "diagnostics") in capsys.readouterr().err

    def test_an_installer_that_ignores_npm_config_registry_fails_the_row(self, tmp_path):
        code, evidence = _run(tmp_path, steps=_steps(discover=lambda b, r: [
            {"setting": "npm_config_registry", "honoured": False, "requests": 0,
             "fell_back_to": ["https://elsewhere.invalid"], "elapsed_s": 1.0}]))
        assert code == 1
        assert any("did not honour npm_config_registry" in f for f in evidence["failures"])

    def test_a_package_from_another_registry_fails_the_row(self, tmp_path):
        code, evidence = _run(tmp_path, steps=_steps(installed_from=lambda d, c, r: {
            "lockfiles": ["x"], "packages": 1, "origins": ["https://elsewhere.invalid"],
            "foreign": ["https://elsewhere.invalid/pkg/-/pkg-1.0.0.tgz"]}))
        assert code == 1
        assert any("outside the single registry" in f for f in evidence["failures"])

    def test_an_unexpected_error_still_writes_failing_evidence(self, tmp_path):
        def explode(*_a, **_k):
            raise KeyError("boom")
        out = tmp_path / "row-7.json"
        with pytest.raises(KeyError):
            probe.row7(_tarball(tmp_path), str(_fake_binary(tmp_path)), SINGLE, out,
                       steps=_steps(discover=explode), tmp_parent=tmp_path)
        evidence = json.loads(out.read_text())
        assert evidence["pass"] is False
        assert "boom" in evidence["failures"][-1]


# --------------------------------------------------------------------------- #
# One project, with OpenCode faked out
# --------------------------------------------------------------------------- #

class _FakeStub:
    def __init__(self, scenario, diag, label, **_kwargs):
        self.scenario = scenario
        self.base_url = "http://127.0.0.1:1/v1"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


def _fake_drive(no_commit_status_fails: bool):
    def drive(binary, env, project, stub, *, tool_name, call_id, mode, diag, label, **_kw):
        args = stub.scenario["scripted_tool_call"]["arguments"]
        assert args["mode"] == mode
        config = json.loads((project / "opencode.json").read_text())
        skill = os.path.join(config["plugin"][0], "skills", "multi-llm")
        argv = ["uv", "run", "--frozen", "--no-dev", "--project", skill, "--", "python",
                os.path.join(skill, "multi_llm.py"), mode]
        if mode == "init":
            (project / ".multi-llm").mkdir(exist_ok=True)
            (project / probe.PROVIDERS_FILE).write_text("providers: {}\n")
            outcome = {"ok": True, "exit": 0, "state": "done", "argv": argv,
                       "result": {"mode": "init", "state": "done", "exit": 0}}
        elif no_commit_status_fails and project.name.endswith("no-commit"):
            outcome = {"ok": False, "exit": 1, "state": "error", "argv": argv,
                       "message": "the dispatcher exited 1",
                       "output": '{"error": "GitError: git rev-parse HEAD failed"}'}
        else:
            outcome = {"ok": True, "exit": 2, "state": "nothing-to-do", "argv": argv,
                       "output": "runs: no run recorded for this plan",
                       "result": {"mode": "status", "state": "nothing-to-do", "exit": 2}}
        return {"advertised": True, "served_call": True, "returned": json.dumps(outcome),
                "returned_identical": True, "outcome": outcome, "problems": [],
                "opencode_exit": 0}
    return drive


@pytest.fixture
def faked_opencode(monkeypatch, tmp_path):
    package = _fake_package(tmp_path / "payload")

    def debug_config(binary, env, cwd, diag, **_kw):
        config = json.loads((cwd / "opencode.json").read_text())
        body = probe.expected_template(Path(config["plugin"][0]), build.SKILL_DIR_TOKEN)
        return probe.DebugConfig(None, {
            "plugin": [f"file://{config['plugin'][0]}"],
            "command": {"multi-llm": {"template": body, "description": "d"}}}, None)

    monkeypatch.setattr(probe, "StubModel", _FakeStub)
    monkeypatch.setattr(probe, "read_debug_config", debug_config)
    iso = probe.isolated_env(tmp_path / "env", binary=str(tmp_path / "pin" / "opencode"),
                             registry=SINGLE, inherited={"PATH": os.environ.get("PATH", "")})
    probe.materialize(iso)
    return package, iso


class TestOneProject:
    def test_a_passing_committed_project_has_the_contract_shape(self, faked_opencode, tmp_path,
                                                                 monkeypatch):
        package, iso = faked_opencode
        monkeypatch.setattr(probe, "drive_scripted_call", _fake_drive(False))
        entry, failures = probe.run_project("committed", tmp_path, iso, package, "opencode",
                                            SURFACE, tmp_path / "diag")
        assert failures == []
        assert PROJECT_KEYS <= set(entry) and entry["ok"] is True
        assert entry["command_skill_dir"] == probe.skill_dir_of(package)
        assert entry["command_inside_payload"] is True
        assert entry["init"]["files_created"] == [probe.PROVIDERS_FILE]
        assert entry["status"]["exit"] == SURFACE["dispatcher"]["exit_codes"]["nothing_to_do"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=entry["project"],
                              capture_output=True)
        assert head.returncode == 0, "the committed project has its one commit"

    def test_the_no_commit_project_fails_on_the_no_head_error_and_gets_no_commit(
            self, faked_opencode, tmp_path, monkeypatch):
        package, iso = faked_opencode
        monkeypatch.setattr(probe, "drive_scripted_call", _fake_drive(True))
        entry, failures = probe.run_project("no-commit", tmp_path, iso, package, "opencode",
                                            SURFACE, tmp_path / "diag")
        assert entry["ok"] is False and entry["status"]["ok"] is False
        assert any(f.startswith("no-commit: status:") for f in failures)
        assert "GitError" in entry["status"]["error"]
        head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"],
                              cwd=entry["project"], capture_output=True)
        assert head.returncode != 0, "the fixture is never given a commit"

    def test_the_project_config_names_the_extracted_directory(self, faked_opencode, tmp_path,
                                                              monkeypatch):
        package, iso = faked_opencode
        monkeypatch.setattr(probe, "drive_scripted_call", _fake_drive(False))
        entry, _ = probe.run_project("committed", tmp_path, iso, package, "opencode",
                                     SURFACE, tmp_path / "diag")
        config = json.loads((Path(entry["project"]) / "opencode.json").read_text())
        assert config["plugin"] == [str(package)]
        assert not any(spec.endswith(".tgz") for spec in config["plugin"])

    def test_a_tool_that_ran_another_skill_dir_fails(self, faked_opencode, tmp_path, monkeypatch):
        package, iso = faked_opencode
        drive = _fake_drive(False)

        def elsewhere(*args, **kwargs):
            verdict = drive(*args, **kwargs)
            verdict["outcome"]["argv"] = ["uv", "run", "--project", "/elsewhere/skills/multi-llm"]
            return verdict

        monkeypatch.setattr(probe, "drive_scripted_call", elsewhere)
        entry, failures = probe.run_project("committed", tmp_path, iso, package, "opencode",
                                            SURFACE, tmp_path / "diag")
        assert entry["command_inside_payload"] is False
        assert any("/elsewhere/skills/multi-llm" in f for f in failures)

    @pytest.mark.parametrize("kind", ["committed", "no-commit"])
    @pytest.mark.parametrize("broken", ["project", "script", "missing-script", "null-result"])
    def test_a_good_init_cannot_hide_a_broken_status(self, faked_opencode, tmp_path,
                                                   monkeypatch, kind, broken):
        package, iso = faked_opencode
        drive = _fake_drive(False)

        def wrong_status(*args, **kwargs):
            verdict = drive(*args, **kwargs)
            if kwargs["mode"] == "status":
                argv = verdict["outcome"]["argv"]
                if broken == "project":
                    argv[argv.index("--project") + 1] = "/elsewhere/skills/multi-llm"
                elif broken == "script":
                    argv[argv.index("--") + 2] = "/elsewhere/multi_llm.py"
                elif broken == "missing-script":
                    del argv[argv.index("--") + 2:]
                else:
                    verdict.update(outcome=None, returned="null", output="null")
            return verdict

        monkeypatch.setattr(probe, "drive_scripted_call", wrong_status)
        entry, failures = probe.run_project(kind, tmp_path, iso, package, "opencode",
                                            SURFACE, tmp_path / "diag")
        assert entry["init"]["ok"] is True
        assert entry["status"]["ok"] is False
        assert entry["ok"] is False
        assert any(f.startswith(f"{kind}: status:") for f in failures), failures


# --------------------------------------------------------------------------- #
# plugin[] and the debug-config check
# --------------------------------------------------------------------------- #

class TestNeverATarballInPlugin:
    @pytest.mark.parametrize("name", ["opencode-multi-llm-0.0.0.tgz", "pkg.tar.gz", "pkg.TGZ"])
    def test_an_archive_is_refused_even_when_it_exists(self, tmp_path, name):
        archive = tmp_path / name
        archive.write_bytes(b"")
        with pytest.raises(probe.ProbeError, match="archive"):
            probe.opencode_config(archive, "http://127.0.0.1:1/v1")

    def test_a_path_that_is_not_a_directory_is_refused(self, tmp_path):
        with pytest.raises(probe.ProbeError, match="directory"):
            probe.opencode_config(tmp_path / "missing", "http://127.0.0.1:1/v1")

    def test_the_directory_is_the_one_entry(self, tmp_path):
        config = probe.opencode_config(tmp_path, "http://127.0.0.1:1/v1")
        assert config["plugin"] == [str(tmp_path)]
        assert config["provider"]["stub"]["options"]["baseURL"] == "http://127.0.0.1:1/v1"

    def test_a_tgz_that_opencode_resolved_is_not_the_payload(self, tmp_path):
        package = _fake_package(tmp_path)
        verdict = probe.check_command({"plugin": [f"file://{package}/x.tgz"]}, package)
        assert verdict["plugin_is_payload"] is False


class TestTheCommandCheck:
    def test_the_payloads_body_resolved_against_the_payload_passes(self, tmp_path):
        package = _fake_package(tmp_path)
        template = probe.expected_template(package, build.SKILL_DIR_TOKEN)
        verdict = probe.check_command({"plugin": [f"file://{package}"],
                                       "command": {"multi-llm": {"template": template}}}, package)
        assert verdict["registered"] and verdict["matches_payload"]
        assert not verdict["token_left"] and verdict["plugin_is_payload"]
        assert verdict["skill_paths_inside"]

    def test_an_unresolved_token_is_reported(self, tmp_path):
        package = _fake_package(tmp_path)
        verdict = probe.check_command(
            {"command": {"multi-llm": {"template": "run {SKILL_DIR}/multi_llm.py"}}}, package)
        assert verdict["token_left"] and not verdict["matches_payload"]

    def test_a_path_outside_the_payload_is_reported(self, tmp_path):
        package = _fake_package(tmp_path)
        verdict = probe.check_command(
            {"command": {"multi-llm": {"template": "uv run --project /opt/other/skills/multi-llm"}}},
            package)
        assert verdict["skill_paths"] == ["/opt/other/skills/multi-llm"]
        assert not verdict["skill_paths_inside"]

    def test_no_command_is_not_registered(self, tmp_path):
        verdict = probe.check_command({"command": {}}, _fake_package(tmp_path))
        assert not verdict["registered"]


# --------------------------------------------------------------------------- #
# The two-sided tool-call assessment
# --------------------------------------------------------------------------- #

def _tool_event(output, status="completed", call_id="c1"):
    return {"type": "tool_use", "part": {"tool": TOOL, "callID": call_id,
                                         "state": {"status": status, "output": output,
                                                   "input": {"mode": "init"}}}}


class TestAssessCall:
    OUTCOME = json.dumps({"ok": True, "exit": 0, "result": {"mode": "init"}})

    def _entries(self, returned=OUTCOME, advertised=True, truncated=False):
        names = [TOOL, "read"] if advertised else ["read"]
        entries = [{"tool_names": names, "scripted": {
            "phase": "call", "declaration": {"type": "function"}} if advertised else {
            "phase": "not-advertised"}}]
        if returned is not None:
            entries.append({"tool_names": names, "scripted": {
                "phase": "terminal", "output_sha": probe._sha256(returned),
                "output": returned, "output_truncated": truncated}})
        return entries

    def test_both_sides_agree(self):
        verdict = probe.assess_call([_tool_event(self.OUTCOME)], self._entries(),
                                    tool_name=TOOL, call_id="c1")
        assert verdict["problems"] == []
        assert verdict["advertised"] and verdict["served_call"] and verdict["returned_identical"]
        assert verdict["outcome"]["result"]["mode"] == "init"

    def test_a_tool_never_advertised_is_unregistered(self):
        verdict = probe.assess_call([], self._entries(returned=None, advertised=False),
                                    tool_name=TOOL, call_id="c1")
        assert not verdict["advertised"]
        assert any("not registered" in p for p in verdict["problems"])

    def test_a_result_that_never_came_back(self):
        verdict = probe.assess_call([_tool_event(self.OUTCOME)], self._entries(returned=None),
                                    tool_name=TOOL, call_id="c1")
        assert any("never reached the model" in p for p in verdict["problems"])

    def test_a_different_text_at_the_model_is_a_problem(self):
        verdict = probe.assess_call([_tool_event(self.OUTCOME)], self._entries(returned='{"x": 1}'),
                                    tool_name=TOOL, call_id="c1")
        assert not verdict["returned_identical"]
        assert any("differs" in p for p in verdict["problems"])

    def test_a_failed_tool_part_is_a_problem(self):
        verdict = probe.assess_call([_tool_event(None, status="error")], self._entries(),
                                    tool_name=TOOL, call_id="c1")
        assert any("ended 'error'" in p for p in verdict["problems"])

    def test_json_null_is_not_a_structured_tool_result(self):
        verdict = probe.assess_call([_tool_event("null")], self._entries(returned="null"),
                                    tool_name=TOOL, call_id="c1")
        assert any("not a JSON object" in p for p in verdict["problems"])

    def test_events_parse_from_ndjson_with_noise(self):
        text = "noise\n" + json.dumps(_tool_event("x")) + "\n{broken\n"
        assert probe.parse_events(text) == [_tool_event("x")]


class TestGradeCall:
    def _call(self, outcome):
        return {"problems": [], "outcome": outcome, "advertised": True,
                "returned": json.dumps(outcome), "returned_identical": True, "opencode_exit": 0}

    def test_status_exit_2_nothing_to_do_is_its_documented_success(self):
        outcome = {"ok": True, "exit": 2, "state": "nothing-to-do", "argv": [],
                   "result": {"mode": "status", "state": "nothing-to-do", "exit": 2}}
        evidence, problems = probe.grade_call(self._call(outcome), mode="status", want_exit=2,
                                              want_state="nothing-to-do", surface=SURFACE)
        assert problems == [] and evidence["ok"] and evidence["exit"] == 2

    def test_an_exit_1_names_the_dispatchers_error(self):
        outcome = {"ok": False, "exit": 1, "state": "error", "message": "the dispatcher exited 1",
                   "output": '{"error": "GitError: git rev-parse HEAD failed"}'}
        evidence, problems = probe.grade_call(self._call(outcome), mode="status", want_exit=2,
                                              want_state="nothing-to-do", surface=SURFACE)
        assert not evidence["ok"]
        assert "GitError" in evidence["error"]
        assert any("exited 1, not 2" in p for p in problems)

    @pytest.mark.parametrize("outcome", [None, {}, [], "null"])
    def test_a_missing_result_never_grades_green(self, outcome):
        evidence, problems = probe.grade_call(self._call(outcome), mode="status", want_exit=2,
                                              want_state="nothing-to-do", surface=SURFACE)
        assert problems and evidence["ok"] is False


# --------------------------------------------------------------------------- #
# The registry readers
# --------------------------------------------------------------------------- #

class TestInstalledFrom:
    def test_every_resolved_url_is_read_and_foreign_ones_named(self, tmp_path):
        config = tmp_path / "config" / "opencode"
        config.mkdir(parents=True)
        (config / "package-lock.json").write_text(json.dumps({"packages": {
            "": {"dependencies": {}},
            "node_modules/@opencode-ai/plugin": {"resolved": f"{SINGLE}/@opencode-ai/plugin/-/p.tgz"},
            "node_modules/zod": {"resolved": "https://mirror.invalid/zod/-/zod.tgz"},
        }}))
        cache = tmp_path / "cache" / "opencode"
        spec_dir = cache / "packages" / "some-spec"
        spec_dir.mkdir(parents=True)
        (spec_dir / "package-lock.json").write_text(json.dumps({"packages": {
            "node_modules/x": {"resolved": f"{SINGLE}/x/-/x.tgz"}}}))
        found = probe.installed_from([config], cache, SINGLE)
        assert found["packages"] == 3 and len(found["lockfiles"]) == 2
        assert found["origins"] == sorted([SINGLE, "https://mirror.invalid"])
        assert found["foreign"] == ["https://mirror.invalid/zod/-/zod.tgz"]

    def test_no_lockfile_reads_as_none(self, tmp_path):
        assert probe.installed_from([tmp_path], None, SINGLE)["lockfiles"] == []


class TestRegistryCandidates:
    def _iso(self, tmp_path):
        iso = probe.isolated_env(tmp_path / "env", binary=str(tmp_path / "opencode"),
                                 registry=None, inherited={"PATH": "/usr/bin"})
        probe.materialize(iso)
        return iso

    def test_each_candidate_names_the_recorder_alone(self, tmp_path):
        iso = self._iso(tmp_path)
        url = "http://127.0.0.1:5"
        env = probe._apply_registry_candidate(iso, "npm_config_registry", url)
        assert env["npm_config_registry"] == url + "/"
        assert not (iso.home / ".npmrc").exists()

        env = probe._apply_registry_candidate(iso, "home_npmrc", url)
        assert "npm_config_registry" not in env and "npm_config_userconfig" not in env
        assert (iso.home / ".npmrc").read_text() == f"registry={url}/\n"

        env = probe._apply_registry_candidate(iso, "bunfig", url)
        assert "npm_config_registry" not in env
        assert url in (iso.home / ".bunfig.toml").read_text()

    def test_the_recorder_records_package_gets_but_not_model_posts(self):
        with probe.RecordingRegistry() as recorder:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(recorder.url + "/@opencode-ai%2fplugin", timeout=5)
            assert recorder.package_requests() == ["/@opencode-ai%2fplugin"]
            assert recorder.package_requests(since=1) == []


# --------------------------------------------------------------------------- #
# The bounded runner
# --------------------------------------------------------------------------- #

class TestRunBounded:
    def test_an_exit_code_is_reported(self, tmp_path):
        bounded = probe.run_bounded([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"],
                                    cwd=tmp_path, env=dict(os.environ), budget_s=60,
                                    stdout=tmp_path / "o", stderr=tmp_path / "e")
        assert bounded.returncode == 3 and not bounded.killed
        assert bounded.stdout.read_text().strip() == "hi"

    def test_stop_ends_the_run(self, tmp_path):
        flag = tmp_path / "flag"
        flag.write_text("")
        bounded = probe.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"],
                                    cwd=tmp_path, env=dict(os.environ), budget_s=60,
                                    stdout=tmp_path / "o", stderr=tmp_path / "e",
                                    stop=flag.exists)
        assert bounded.killed and bounded.reason == "stopped" and bounded.returncode is None
        assert bounded.elapsed_s < 30

    @pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="session sweep reads /proc")
    def test_the_budget_kills_the_whole_session_other_process_groups_included(self, tmp_path):
        """The plugin starts uv in its own process group; one kill must reach it."""
        pid_file = tmp_path / "grandchild.pid"
        script = (
            "import os, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
            " preexec_fn=os.setpgrp)\n"
            f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(120)\n")
        bounded = probe.run_bounded([sys.executable, "-c", script], cwd=tmp_path,
                                    env=dict(os.environ), budget_s=3,
                                    stdout=tmp_path / "o", stderr=tmp_path / "e")
        assert bounded.killed and bounded.reason == "budget"
        grandchild = int(pid_file.read_text())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and Path(f"/proc/{grandchild}").exists():
            state = Path(f"/proc/{grandchild}/stat").read_text().rsplit(")", 1)[-1].split()[0]
            if state == "Z":
                break
            time.sleep(0.1)
        alive = Path(f"/proc/{grandchild}").exists() and Path(
            f"/proc/{grandchild}/stat").read_text().rsplit(")", 1)[-1].split()[0] != "Z"
        assert not alive, "a process in another group of the session outlived the kill"


# --------------------------------------------------------------------------- #
# The generated scenario, against the real stub server
# --------------------------------------------------------------------------- #

def _chat(base_url, body):
    request = urllib.request.Request(base_url + "/chat/completions",
                                     data=json.dumps(dict(body, stream=False)).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


class TestTheScenarioArmsTheRealStub:
    def test_one_call_then_the_result_comes_back_then_the_terminal(self, tmp_path):
        tool = {"type": "function", "function": {"name": TOOL, "parameters": {"type": "object"}}}
        scenario = probe.scripted_call_scenario(TOOL, probe.INIT_ARGS, "call_row7_test")
        result = json.dumps({"ok": True, "exit": 0, "result": {"mode": "init"}})
        with probe.StubModel(scenario, tmp_path, "t") as stub:
            first = _chat(stub.base_url, {"model": "m", "tools": [tool], "messages": [
                {"role": "user", "content": "Run the tool."}]})
            call = first["choices"][0]["message"]["tool_calls"][0]
            assert call["function"]["name"] == TOOL
            assert json.loads(call["function"]["arguments"]) == probe.INIT_ARGS
            second = _chat(stub.base_url, {"model": "m", "tools": [tool], "messages": [
                {"role": "user", "content": "Run the tool."},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": result}]})
            assert "returned" in second["choices"][0]["message"]["content"]
            title = _chat(stub.base_url, {"model": "m", "messages": [
                {"role": "user", "content": "Generate a short title (four words or less)"}]})
            assert title["choices"][0]["message"]["content"] == "Row seven probe"
            entries = stub.entries()
        verdict = probe.assess_call([_tool_event(result, call_id="call_row7_test")], entries,
                                    tool_name=TOOL, call_id="call_row7_test")
        assert verdict["problems"] == []
        assert verdict["outcome"]["result"]["mode"] == "init"
        assert not any(e.get("unmatched") or e.get("unhandled") for e in entries)

    def test_the_two_calls_carry_every_property_the_tool_requires(self):
        for args in (probe.INIT_ARGS, probe.STATUS_ARGS):
            assert set(args) == set(SURFACE["argument_descriptions"])
            assert args["question"] == "" and args["revs"] == "" and args["extra_flags"] == []
        assert probe.INIT_ARGS["plan"] == ""
        assert probe.STATUS_ARGS["plan"] == probe.PLAN


class TestTheProbeNamesNoRelease:
    def test_no_version_literal_in_the_probe(self):
        """The release lives in PINS; the probe reads it through bump.py."""
        # Exactly three numeric parts, so 127.0.0.1 (four) is not one.
        literal = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])")
        assert not literal.search(PROBE_PATH.read_text(encoding="utf-8"))

    def test_the_default_steps_are_the_real_ones(self):
        steps = probe.Steps()
        assert steps.discover is probe.discover_registry_settings
        assert steps.project is probe.run_project
        assert "bump" in inspect.getsource(probe.Steps.__post_init__)
