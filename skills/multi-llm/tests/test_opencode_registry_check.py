"""The registry-install probe, ``packaging/opencode/registry_check.py``, offline.

The probe itself needs the pinned ``opencode``, npm, a registry (or the
localhost stub) and, for the container environment, Docker; it runs in the
bootstrap job, in step 5 and in rehearsals, never here. What this file grades
is everything that decides whether a green record means what it says:

* the refusals: a registry other than the single one, and a rehearsal registry
  that is not loopback, both refuse before anything runs; so does a rehearsal
  record aimed at the committed records directory;
* a rehearsal record is never release evidence (the plan's own regression
  case), and the validator's structure-only option checks its shape without
  ever saying otherwise;
* each kind of record carries exactly its profile's criteria, in order;
* the candidate is rebuilt from the TAG's commit, not the working tree (a git
  fixture whose working tree differs);
* a digest mismatch between the registry files, the installed payload and the
  candidate fails BEFORE the first tool call (fakes for OpenCode, which never
  call the tool);
* the network proof refuses an ``npm view`` that failed with ``E404`` or an
  auth error, which are answers from a registry rather than a cut network;
* the container kit runs without the repository, and the container driver
  starts a stock image, mounts nothing, commits, cuts the network and deletes
  what it made (a fake Docker).

No model, no network beyond 127.0.0.1, no Docker, no spend.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO / "packaging" / "opencode"
CHECK_PATH = PACKAGE_DIR / "registry_check.py"
RECORD_SCRIPT = PACKAGE_DIR / "registry_record.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load("opencode_registry_check_under_test", CHECK_PATH)
probe = rc._probe()
rr = rc._record_module()
build = rc._build()
SINGLE = rc._registry().single_registry()
SURFACE = build.cli_surface()
NAME = "opencode-multi-llm"
VERSION = "0.1.1"
SPEC = f"{NAME}@{VERSION}"
STUB = "http://127.0.0.1:43210"


# --------------------------------------------------------------------------- #
# Fixtures: a candidate, its tarball, a tag
# --------------------------------------------------------------------------- #

def _package_tree(root: Path, version: str = VERSION, dispatcher: bytes = b"") -> Path:
    """A built package directory: every file npm packs, and the build's digest."""
    files = {
        "package.json": json.dumps({"name": NAME, "version": version, "main": "./dist/index.mjs",
                                    "license": "MIT"}),
        "README.md": "# opencode-multi-llm\n",
        "LICENSE": "MIT\n",
        "dist/index.mjs": "export default {}\n",
        "dist/command.md": build.command_body(),
        "dist/cli-surface.json": json.dumps(SURFACE),
        "skills/multi-llm/pyproject.toml": textwrap.dedent("""\
            [project]
            name = "multi-llm"
            version = "0.1.0"
            dependencies = ["pyyaml>=6.0"]
            """),
        "skills/multi-llm/multi_llm.py": "# the dispatcher\n" + dispatcher.decode(),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    digest = rc.inventory_digest(root, sorted(files))
    (root / rc.DIGEST_FILE).write_text(digest + "\n", encoding="utf-8")
    return root


def _candidate(root: Path, sha: str = "c" * 40) -> dict:
    files, _odd = rc.file_inventory(root)
    digest_files = [f for f in files if f != rc.DIGEST_FILE]
    return {"sha": sha, "digest": (root / rc.DIGEST_FILE).read_text().strip(), "inventory": files,
            "digest_files": digest_files}


def _pack(source: Path, path: Path) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for file in sorted(p for p in source.rglob("*") if p.is_file()):
            archive.add(file, arcname="package/" + file.relative_to(source).as_posix())
    return path


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
                           "-c", "init.defaultBranch=main", "-C", str(repo), *args], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _tagged_repo(root: Path, version: str = VERSION, public_version=None) -> dict:
    repo = root / "repo"
    (repo / "packaging" / "opencode").mkdir(parents=True)
    (repo / "skills" / "multi-llm" / "hosts").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / rr.TAG_MANIFEST).write_text(json.dumps({"name": NAME, "version": version}))
    package = {"registry": "npm", "name": NAME, "version": version}
    if public_version:
        package["public_version"] = public_version
    (repo / rr.CAPABILITIES).write_text(json.dumps({"version": 1, "hosts": {"opencode": {
        "package": package}}}))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "release")
    tag = f"opencode-v{version}"
    _git(repo, "tag", tag)
    return {"repo": repo, "tag": tag, "sha": _git(repo, "rev-parse", "HEAD")}


# --------------------------------------------------------------------------- #
# A fake host: npm, OpenCode's installer and the tool, for the assembly
# --------------------------------------------------------------------------- #

class FakeHost:
    """Stands in for every child the probe starts in one environment.

    ``served`` is the tarball npm "serves"; ``installed`` is what OpenCode's
    installer "unpacks" (the same by default); ``tamper`` edits the installed
    package directory after the install. ``calls_allowed=False`` makes any
    tool call an assertion failure: the mismatch cases must never reach one.
    """

    def __init__(self, served: Path, *, installed: Path = None, registry: str = STUB,
                 tamper=None, calls_allowed: bool = True, others=None, stale_after_upgrade=False):
        self.served = served
        self.installed = installed or served
        #: Other specs the "registry" serves (a later release's previous pin).
        self.others = dict(others or {})
        #: The upgraded installation keeps running the previous payload.
        self.stale_after_upgrade = stale_after_upgrade
        self.registry = registry
        self.tamper = tamper
        self.calls_allowed = calls_allowed
        self.tool_calls = []
        self.integrity = rc._load("opencode_registry_stub",
                                  PACKAGE_DIR / "registry_stub.py").integrity_of(served.read_bytes())

    def steps(self) -> "rc.Steps":
        return rc.Steps(run=self.run, debug_paths=self.debug_paths, debug_config=self.debug_config,
                        tool_call=self.tool_call, clock=lambda: 0.0, sleep=lambda _s: None)

    def run(self, argv, *, cwd, env, budget_s, stdout, stderr):
        argv = [str(a) for a in argv]
        out, code = "", 0
        if argv[-1] == "--version":
            out = {"opencode": "1.19.4", "node": "v22.19.0", "npm": "10.9.3"}.get(
                Path(argv[0]).name, "uv 0.13.5 (x86_64-unknown-linux-gnu)")
        elif argv[:2] == ["npm", "view"]:
            out = json.dumps({"version": VERSION, "dist": {
                "tarball": f"{self.registry}/{NAME}/-/{NAME}-{VERSION}.tgz",
                "integrity": self.integrity}})
        elif argv[:2] == ["npm", "pack"]:
            dest = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--pack-destination=")))
            shutil.copyfile(self.served, dest / f"{NAME}-{VERSION}.tgz")
            out = json.dumps([{"filename": f"{NAME}-{VERSION}.tgz", "integrity": self.integrity}])
        elif "-c" in argv and Path(argv[0]).name == "python":
            out = json.dumps({"python": "3.11.2", "dists": ["pyyaml"]})
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(out, encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return probe.Bounded(argv=tuple(argv), returncode=code, killed=False, reason=None,
                             elapsed_s=0.0, stdout=stdout, stderr=stderr)

    @staticmethod
    def _paths(env):
        return {"cache": f"{env['XDG_CACHE_HOME']}/opencode", "data": f"{env['XDG_DATA_HOME']}/opencode",
                "config": f"{env['XDG_CONFIG_HOME']}/opencode", "state": f"{env['XDG_STATE_HOME']}/opencode"}

    def debug_paths(self, binary, env, cwd, diag, label):
        return {"bounded": {}, "paths": self._paths(env)}

    def location(self, env, spec=SPEC):
        return Path(self._paths(env)["cache"]) / "packages" / spec / "node_modules" / NAME

    def debug_config(self, binary, env, cwd, diag, label):
        spec = json.loads((Path(cwd) / "opencode.json").read_text())["plugin"][0]
        location = self.location(env, spec)
        if not location.exists():
            root = location.parent.parent
            with tarfile.open(self.others.get(spec, self.installed)) as archive:
                for member in archive.getmembers():
                    target = location / member.name[len("package/"):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
            entry = {"version": VERSION, "integrity": self.integrity,
                     "resolved": f"{self.registry}/{NAME}/-/{NAME}-{VERSION}.tgz"}
            (root / "package.json").write_text(json.dumps({"dependencies": {NAME: VERSION}}))
            (root / "package-lock.json").write_text(json.dumps({"packages": {
                "": {}, f"node_modules/{NAME}": entry}}))
            (root / "node_modules" / ".package-lock.json").write_text("{}")
            if self.tamper:
                self.tamper(location)
        config = {"command": {"multi-llm": {"template": probe.expected_template(location, "{SKILL_DIR}")}},
                  "plugin": [spec], "plugin_origins": [{"spec": spec}]}
        return probe.DebugConfig(bounded=None, config=config, error=None)

    def tool_call(self, binary, env, project, spec, tool, arguments, mode, call_id, diag, label):
        if not self.calls_allowed:
            raise AssertionError(f"a tool call ({mode}) ran on bytes that are not the candidate")
        self.tool_calls.append(f"{mode} {spec}" if spec != SPEC else mode)
        if self.stale_after_upgrade and spec == SPEC and self.others:
            spec = next(iter(self.others))
        skill = self.location(env, spec) / "skills" / "multi-llm"
        if mode == "init":
            (Path(project) / ".multi-llm").mkdir(exist_ok=True)
            (Path(project) / ".multi-llm" / "providers.yaml").write_text("x: 1\n")
            (skill / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (skill / ".venv" / "bin" / "python").write_text("")
            (Path(env["UV_CACHE_DIR"]) / "wheels-v6").mkdir(parents=True, exist_ok=True)
            (skill / "__pycache__").mkdir(exist_ok=True)
            (skill / "__pycache__" / "multi_llm.cpython-311.pyc").write_bytes(b"runtime state")
        result = {"mode": mode, "phase": None, "state": "done" if mode == "init" else "nothing-to-do",
                  "exit": 0 if mode == "init" else 2, "artifacts": [], "next": None}
        outcome = {"ok": True, "state": result["state"], "exit": result["exit"], "result": result,
                   "argv": ["/usr/local/bin/uv", "run", "--frozen", "--no-dev", "--project",
                            str(skill), "--", "python", str(skill / "multi_llm.py"), mode],
                   "output": "Wrote providers" if mode == "init" else "plans/probe.md: no run recorded"}
        text = json.dumps(outcome)
        return {"advertised": True, "served_call": True, "tool_status": "completed", "output": text,
                "returned": text, "returned_identical": True, "outcome": outcome,
                "error_events": [], "unmatched_requests": 0, "problems": [], "opencode_exit": 0}


def _context(tmp_path, candidate, environment="ci-runner", previous_pin=None, rehearsal=True) -> "rc.Context":
    binary = tmp_path / "bin" / "opencode"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("")
    return rc.Context(spec=SPEC, name=NAME, version=VERSION,
                      read_registry=STUB if rehearsal else SINGLE, single_registry=SINGLE,
                      rehearsal=rehearsal, environment=environment, candidate=candidate,
                      binary=str(binary), opencode_pin="1.19.4", probe_root=tmp_path / "root",
                      previous_pin=previous_pin, poll_s=0.0, interval_s=0.0)


@pytest.fixture()
def candidate_and_tarball(tmp_path):
    tree = _package_tree(tmp_path / "candidate")
    return _candidate(tree), _pack(tree, tmp_path / f"{NAME}-{VERSION}.tgz")


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #

class TestTheRegistryRefusals:
    @pytest.mark.parametrize("url", ["https://registry.yarnpkg.com", "https://registry.npmjs.org.evil",
                                     "http://registry.npmjs.org", STUB])
    def test_a_registry_other_than_the_single_one_is_refused(self, url):
        with pytest.raises(rc.CheckError, match="single registry"):
            rc.resolve_registries(url)

    @pytest.mark.parametrize("url", [
        "http://localhost:43210", "http://10.0.0.7:43210", "https://127.0.0.1:43210",
        "http://127.0.0.1", "http://127.0.0.1:0", "http://127.0.0.1:99999",
        "http://127.0.0.1:43210/path", "http://127.0.0.2:43210", "http://[::1]:43210",
        "http://host.docker.internal:43210", SINGLE,
    ])
    def test_a_rehearsal_registry_must_be_loopback(self, url):
        with pytest.raises(rc.CheckError, match="loopback"):
            rc.resolve_registries(SINGLE, url)

    def test_a_loopback_stub_makes_a_rehearsal(self):
        assert rc.resolve_registries(SINGLE) == (SINGLE, SINGLE, False)
        assert rc.resolve_registries(SINGLE + "/", STUB + "/") == (SINGLE, STUB, True)

    def test_the_cli_refuses_before_anything_runs(self, tmp_path, capsys):
        out = tmp_path / "record.json"
        for extra in (["--registry", "https://registry.yarnpkg.com"],
                      ["--registry", SINGLE, "--rehearsal-registry", "http://10.0.0.7:1"]):
            args = rc.build_parser().parse_args(["--spec", SPEC, "--tag", f"opencode-v{VERSION}",
                                                 "--environment", "ci-runner", "--out", str(out),
                                                 *extra])
            assert rc.check(args) == 1
            assert not out.exists()
        assert "REFUSING" in capsys.readouterr().err

    def test_a_rehearsal_record_never_lands_in_the_records_directory(self, capsys):
        out = rc.RECORDS_DIR / "20261025T000000Z-opencode-release-network-check-rehearsal.json"
        args = rc.build_parser().parse_args([
            "--spec", SPEC, "--registry", SINGLE, "--rehearsal-registry", STUB,
            "--tag", f"opencode-v{VERSION}", "--environment", "container", "--out", str(out)])
        assert rc.check(args) == 1
        assert not out.exists()
        assert "committable" in capsys.readouterr().err

    @pytest.mark.parametrize("spec", [f"{NAME}@^0.1.1", f"{NAME}@latest", NAME, f"{NAME}@0.1",
                                      f"@{NAME}"])
    def test_only_an_exact_pin(self, spec):
        with pytest.raises(rc.CheckError, match="exact"):
            rc.parse_spec(spec)
        assert rc.parse_spec("@scope/pkg@1.2.3-rc.1") == ("@scope/pkg", "1.2.3-rc.1")

    def _args(self, fixture, tmp_path, *extra, environment="ci-runner"):
        return rc.build_parser().parse_args([
            "--spec", SPEC, "--registry", SINGLE, "--tag", fixture["tag"],
            "--environment", environment, "--out", str(tmp_path / "out" / "record.json"),
            "--repo", str(fixture["repo"]), *extra])

    def test_the_tag_decides_the_kind_and_the_previous_pin(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        first = _tagged_repo(tmp_path / "first")
        later = _tagged_repo(tmp_path / "later", public_version="0.1.0")
        cases = [
            (self._args(first, tmp_path), "names its workflow run"),
            (self._args(first, tmp_path, "--workflow-run-id", "5", environment="container"),
             "belongs to the ci-runner"),
            (self._args(first, tmp_path, "--workflow-run-id", "5", "--previous-pin",
                        f"{NAME}@0.1.0"), "has no previous pin"),
            (self._args(later, tmp_path, "--workflow-run-id", "5"), "first release only"),
            (self._args(later, tmp_path, "--previous-pin", f"{NAME}@0.0.9",
                        environment="container"), "is not opencode-multi-llm@0.1.0"),
        ]
        for args, message in cases:
            assert rc.check(args) == 1
            assert message in capsys.readouterr().err
        wrong = self._args(first, tmp_path, "--workflow-run-id", "5")
        wrong.spec = f"{NAME}@0.1.2"
        assert rc.check(wrong) == 1
        assert "is not opencode-v0.1.1's package.json" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Rehearsals are never release evidence (the plan's regression case)
# --------------------------------------------------------------------------- #

def _passing_criteria(names):
    criteria = {}
    for name in names:
        criteria[name] = {"pass": True, "evidence": {"output": f"{name}: ok"}}
        if name == rc.OFFLINE:
            payload = {"location": "/work/p", "digest": "d" * 64}
            side = {"taken_at": "t", "interfaces": ["lo"],
                    "npm_view": {"registry": SINGLE, "exit": 1, "code": "EAI_AGAIN"}}
            criteria[name]["network_proof"] = {"network_mode": "none", "before": side, "after": side,
                                               "payload": {"before": payload, "after": payload}}
    return criteria


def _record(fixture, *, environment="ci-runner", rehearsal=True, kind="first", previous=None,
            run_id=None, criteria=None):
    registry_url = STUB if rehearsal else SINGLE
    return rc.build_record(
        environment=environment, rehearsal=rehearsal, workflow_run_id=run_id, spec=SPEC, name=NAME,
        version=VERSION, read_registry=registry_url, installed_registry=registry_url,
        integrity="sha512-" + "Q" * 86 + "==", tag=fixture["tag"], sha=fixture["sha"],
        candidate_digest="d" * 64, kind=kind, previous_pin=previous,
        measured={"opencode": "1.19.4", "python": "3.11.2", "uv": "0.13.5"},
        variables_set={"OPENCODE_DISABLE_MODELS_FETCH": "1"},
        criteria=criteria if criteria is not None else _passing_criteria(rc.criteria_for(environment, kind)),
        payload_location="/work/p", environment_details={"probe_root": "/work"})


class TestARehearsalIsNeverReleaseEvidence:
    @pytest.fixture()
    def first(self, tmp_path):
        return _tagged_repo(tmp_path)

    @pytest.mark.parametrize("environment, profile", [("ci-runner", "bootstrap"),
                                                      ("container", "release"), ("vm", "release")])
    def test_the_validator_refuses_it_under_its_own_profile(self, first, environment, profile):
        record = _record(first, environment=environment,
                         run_id=12345 if environment == "ci-runner" else None)
        errors = rr.validate_record(record, profile=profile, tag=first["tag"], repo=first["repo"])
        assert any("never release evidence" in e for e in errors), errors
        assert any("is not the single registry" in e for e in errors)

    def test_even_with_the_single_registry_and_a_run_id(self, first):
        """The flag alone refuses it: nothing else about the record is wrong."""
        record = _record(first, rehearsal=False, run_id=12345)
        assert rr.validate_record(record, profile="bootstrap", tag=first["tag"], repo=first["repo"]) == []
        record["rehearsal"] = True
        errors = rr.validate_record(record, profile="bootstrap", tag=first["tag"], repo=first["repo"])
        assert errors and all("rehearsal" in e or "True is not False" in e for e in errors), errors

    def test_structure_only_accepts_its_shape_and_nothing_more(self, first):
        record = _record(first)
        assert rr.validate_record(record, profile="bootstrap", tag=first["tag"], repo=first["repo"],
                                  rehearsal_structure_only=True) == []
        # The probe's own helper picks structure-only exactly when the record is a rehearsal.
        assert rc.validate_record(record, environment="ci-runner", tag=first["tag"],
                                  repo=first["repo"], digest="d" * 64) == []
        # The shape still matters: a bootstrap rehearsal carrying the offline criterion,
        # a registry that is neither the single one nor loopback, a non-rehearsal record.
        offline = _record(first, criteria=_passing_criteria(rr.FIRST_RELEASE_CRITERIA))
        offline["criteria"].append({"name": rc.OFFLINE, "pass": True, "evidence": {"x": 1}})
        assert any(rc.OFFLINE in e for e in rr.validate_record(
            offline, profile="bootstrap", tag=first["tag"], repo=first["repo"],
            rehearsal_structure_only=True))
        foreign = _record(first)
        foreign["registry"] = "http://10.0.0.7:4873"
        assert any("not the single registry" in e for e in rr.validate_record(
            foreign, profile="bootstrap", tag=first["tag"], repo=first["repo"],
            rehearsal_structure_only=True))
        real = _record(first, rehearsal=False, run_id=7)
        assert any("checks rehearsal records only" in e for e in rr.validate_record(
            real, profile="bootstrap", tag=first["tag"], repo=first["repo"],
            rehearsal_structure_only=True))

    def test_the_cli_says_so_every_time(self, first, tmp_path):
        record = _record(first)
        good = tmp_path / "20261025T000000Z-opencode-bootstrap-check-rehearsal.json"
        canonical = tmp_path / "20261025T000000Z-opencode-bootstrap-check.json"
        for path in (good, canonical):
            path.write_text(json.dumps(record))

        def cli(*args):
            return subprocess.run([sys.executable, str(RECORD_SCRIPT), "validate", "--profile",
                                   "bootstrap", "--repo", str(first["repo"]), *map(str, args)],
                                  capture_output=True, text=True, timeout=120)

        done = cli(good, "--rehearsal-structure-only")
        assert done.returncode == 0, done.stderr
        assert rr.REHEARSAL_NOTICE in done.stdout and rr.REHEARSAL_NOTICE in done.stderr
        done = cli(canonical, "--rehearsal-structure-only")
        assert done.returncode == 1 and "release record's name" in done.stderr
        assert rr.REHEARSAL_NOTICE in done.stderr
        done = cli(good)
        assert done.returncode == 1 and "never release evidence" in done.stderr


# --------------------------------------------------------------------------- #
# The criteria each kind of record carries
# --------------------------------------------------------------------------- #

class TestTheCriteriaSets:
    @pytest.fixture()
    def first(self, tmp_path):
        return _tagged_repo(tmp_path)

    @pytest.mark.parametrize("environment, kind, expected", [
        ("ci-runner", "first", "BOOTSTRAP_CRITERIA"),
        ("container", "first", "FIRST_RELEASE_CRITERIA"),
        ("vm", "first", "FIRST_RELEASE_CRITERIA"),
        ("container", "later", "LATER_RELEASE_CRITERIA"),
        ("vm", "later", "LATER_RELEASE_CRITERIA"),
    ])
    def test_exactly_the_profiles_set_in_order(self, first, environment, kind, expected):
        record = _record(first, environment=environment, kind=kind,
                         previous=f"{NAME}@0.1.0" if kind == "later" else None)
        names = [c["name"] for c in record["criteria"]]
        assert tuple(names) == getattr(rr, expected)

    def test_the_bootstrap_record_never_carries_the_offline_criterion(self, first):
        record = _record(first, criteria=_passing_criteria(rr.LATER_RELEASE_CRITERIA))
        names = [c["name"] for c in record["criteria"]]
        assert len(names) == 7 and rc.OFFLINE not in names and rc.UPGRADE not in names
        assert record["previous_pin"] is None and record["workflow_run_id"] is None

    def test_a_criterion_never_reached_is_present_and_failing(self, first):
        record = _record(first, environment="container", criteria={rc.NPM: {"pass": True,
                                                                             "evidence": {"x": 1}}})
        entries = {c["name"]: c for c in record["criteria"]}
        assert len(entries) == 8 and entries[rc.NPM]["pass"] is True
        assert all(not entries[n]["pass"] and entries[n]["evidence"]["not_attempted"]
                   for n in entries if n != rc.NPM)

    def test_only_the_offline_criterion_carries_a_network_proof(self, first):
        record = _record(first, environment="container")
        for entry in record["criteria"]:
            assert ("network_proof" in entry) == (entry["name"] == rc.OFFLINE)
        assert rr.validate_record(record, profile="release", tag=first["tag"], repo=first["repo"],
                                  rehearsal_structure_only=True) == []


# --------------------------------------------------------------------------- #
# The candidate comes from the tag's commit
# --------------------------------------------------------------------------- #

FAKE_BUILD = textwrap.dedent('''\
    """A stand-in for the tag's build.py: stage the payload, record the digest."""
    import hashlib, subprocess, sys
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    dest = Path(sys.argv[sys.argv.index("--destination") + 1])
    files = {
        "package.json": (here / "packaging/opencode/package.json").read_bytes(),
        "skills/multi-llm/payload.txt": (here / "skills/multi-llm/payload.txt").read_bytes(),
        "dist/built-from": subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                                          capture_output=True).stdout,
    }
    # Every file the checkout holds there: an untracked one would be staged too,
    # so its absence from the candidate is the worktree's doing.
    for extra in sorted(p for p in (here / "skills/multi-llm").iterdir() if p.is_file()):
        files.setdefault("skills/multi-llm/" + extra.name, extra.read_bytes())
    acc = hashlib.sha256()
    for rel in sorted(files):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[rel])
        acc.update(rel.encode() + b"\\0" + hashlib.sha256(files[rel]).digest())
    (dest / "dist/tree.sha256").write_text(acc.hexdigest() + "\\n")
    ''')


class TestTheCandidateComesFromTheTagsCommit:
    @pytest.fixture()
    def dirty_repo(self, tmp_path):
        fixture = _tagged_repo(tmp_path)
        repo = fixture["repo"]
        (repo / "packaging" / "opencode" / "build.py").write_text(FAKE_BUILD)
        (repo / "skills" / "multi-llm" / "payload.txt").write_text("committed bytes\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "the build and its payload")
        _git(repo, "tag", "-f", fixture["tag"])
        fixture["sha"] = _git(repo, "rev-parse", "HEAD")
        # The working tree now differs from the tag: an edit and an untracked file.
        (repo / "skills" / "multi-llm" / "payload.txt").write_text("UNCOMMITTED EDIT\n")
        (repo / "skills" / "multi-llm" / "untracked.txt").write_text("never committed\n")
        return fixture

    @staticmethod
    def _argv(worktree, destination):
        return [sys.executable, str(worktree / "packaging" / "opencode" / "build.py"),
                "--destination", str(destination)]

    def test_the_working_tree_never_reaches_the_candidate(self, dirty_repo, tmp_path):
        repo = dirty_repo["repo"]
        candidate = rc.rebuild_candidate(repo, dirty_repo["sha"], tmp_path / "scratch",
                                         build_argv=self._argv)
        built = Path(candidate["destination"])
        assert (built / "skills/multi-llm/payload.txt").read_text() == "committed bytes\n"
        assert "skills/multi-llm/untracked.txt" not in candidate["inventory"]
        assert (built / "dist/built-from").read_text().strip() == dirty_repo["sha"]
        assert candidate["sha"] == dirty_repo["sha"]
        assert candidate["digest"] == (built / "dist/tree.sha256").read_text().strip()
        assert candidate["build"]["worktree_status_after"] == ""
        # The temporary worktree is gone, and the working tree is exactly as it was.
        assert _git(repo, "worktree", "list").count("\n") == 0
        assert (repo / "skills/multi-llm/payload.txt").read_text() == "UNCOMMITTED EDIT\n"
        assert (repo / "skills/multi-llm/untracked.txt").exists()

    def test_a_digest_it_cannot_account_for_is_refused(self, dirty_repo, tmp_path):
        """A build whose recorded digest is not its own inventory's is refused, so
        the probe never compares against a number it cannot reproduce."""
        def lying(worktree, destination):
            (worktree / "packaging" / "opencode" / "build.py").write_text(
                FAKE_BUILD.replace("acc.hexdigest()", '"0" * 64'))
            return self._argv(worktree, destination)
        with pytest.raises(rc.CheckError, match="cannot account for"):
            rc.rebuild_candidate(dirty_repo["repo"], dirty_repo["sha"], tmp_path / "s2",
                                 build_argv=lying)
        assert _git(dirty_repo["repo"], "worktree", "list").count("\n") == 0

    def test_the_real_build_runs_frozen_without_dev(self, tmp_path):
        argv = rc.default_build_argv(tmp_path / "wt", tmp_path / "dest")
        assert argv[:4] == ["uv", "run", "--frozen", "--no-dev"]
        assert argv[argv.index("--project") + 1] == str(tmp_path / "wt" / "skills" / "multi-llm")
        assert str(tmp_path / "wt" / "packaging" / "opencode" / "build.py") in argv
        assert argv[-2:] == ["--destination", str(tmp_path / "dest")]


# --------------------------------------------------------------------------- #
# The comparison happens before the first tool call
# --------------------------------------------------------------------------- #

class TestAMismatchFailsBeforeTheFirstToolCall:
    def _run(self, tmp_path, candidate, host):
        return rc.run_criteria(_context(tmp_path, candidate), host.steps())

    def test_a_tampered_installed_payload(self, tmp_path, candidate_and_tarball):
        candidate, tarball = candidate_and_tarball
        host = FakeHost(tarball, calls_allowed=False,
                        tamper=lambda p: (p / "skills/multi-llm/multi_llm.py").write_text("evil\n"))
        result = self._run(tmp_path, candidate, host)
        tree = result["criteria"][rc.TREE]
        assert tree["pass"] is False
        assert tree["evidence"]["registry_files"]["matches"] is True
        assert tree["evidence"]["installed"]["matches"] is False
        assert any("installed payload is not the candidate" in p for p in tree["evidence"]["problems"])
        for name in (rc.INIT, rc.COLD, rc.RESTART):
            assert result["criteria"][name]["evidence"]["not_attempted"].startswith(
                "the installed payload is not the candidate")
        assert host.tool_calls == []

    def test_an_extra_file_in_the_installed_package(self, tmp_path, candidate_and_tarball):
        candidate, tarball = candidate_and_tarball
        host = FakeHost(tarball, calls_allowed=False,
                        tamper=lambda p: (p / "dist" / "postinstall.sh").write_text("#!/bin/sh\n"))
        result = self._run(tmp_path, candidate, host)
        installed = result["criteria"][rc.TREE]["evidence"]["installed"]
        assert installed["unexpected"] == ["dist/postinstall.sh"]
        # Its digest still equals the candidate's: the inventory is what catches it.
        assert installed["digest"] == candidate["digest"] and installed["matches"] is False
        assert result["criteria"][rc.INIT]["pass"] is False and host.tool_calls == []

    def test_registry_files_that_are_not_the_candidate(self, tmp_path, candidate_and_tarball):
        candidate, _ = candidate_and_tarball
        other = _package_tree(tmp_path / "other", dispatcher=b"# a different build\n")
        served = _pack(other, tmp_path / "served.tgz")
        host = FakeHost(served, calls_allowed=False)
        result = self._run(tmp_path, candidate, host)
        evidence = result["criteria"][rc.TREE]["evidence"]
        assert result["criteria"][rc.NPM]["pass"] is True  # npm served its own bytes faithfully
        assert evidence["registry_files"]["matches"] is False
        assert evidence["installed"]["matches"] is False
        assert host.tool_calls == []

    def test_installer_metadata_that_is_not_identified(self, tmp_path, candidate_and_tarball):
        candidate, tarball = candidate_and_tarball
        host = FakeHost(tarball, calls_allowed=False, tamper=lambda p: (
            p.parent.parent / "bun.lock").write_text("lock"))
        result = self._run(tmp_path, candidate, host)
        metadata = result["criteria"][rc.TREE]["evidence"]["installer_metadata"]
        assert metadata["unexpected"] == ["bun.lock"]
        assert sorted(metadata["identified"]) == sorted(rc.INSTALLER_METADATA)
        assert result["criteria"][rc.TREE]["pass"] is False and host.tool_calls == []

    def test_the_identical_install_goes_on_and_every_criterion_passes(self, tmp_path,
                                                                     candidate_and_tarball):
        """The same assembly, untampered: the seven bootstrap criteria pass, the
        calls run in order, and the record validates as a rehearsal's structure."""
        candidate, tarball = candidate_and_tarball
        fixture = _tagged_repo(tmp_path / "tag")
        candidate = dict(candidate, sha=fixture["sha"])
        host = FakeHost(tarball)
        result = self._run(tmp_path, candidate, host)
        failing = {n: c["evidence"].get("problems") for n, c in result["criteria"].items()
                   if not c["pass"]}
        assert failing == {}
        assert host.tool_calls == ["init", "status", "status"]
        assert result["state"]["installed_registry"] == STUB
        restart = result["criteria"][rc.RESTART]["evidence"]
        assert restart["before"]["payload"]["digest"] == candidate["digest"]  # .venv and .pyc ignored
        record = rc.build_record(
            environment="ci-runner", rehearsal=True, workflow_run_id=None, spec=SPEC, name=NAME,
            version=VERSION, read_registry=STUB, installed_registry=result["state"]["installed_registry"],
            integrity=result["state"]["integrity"], tag=fixture["tag"], sha=fixture["sha"],
            candidate_digest=candidate["digest"], kind="first", previous_pin=None,
            measured=result["measured"], variables_set=result["variables_set"],
            criteria=result["criteria"], payload_location=result["state"]["location"],
            environment_details=result["details"])
        assert record["measured"]["uv"] == "0.13.5" and record["measured"]["python"] == "3.11.2"
        assert rc.validate_record(record, environment="ci-runner", tag=fixture["tag"],
                                  repo=fixture["repo"], digest=candidate["digest"]) == []

    def test_a_payload_outside_the_probe_root_fails(self, tmp_path, candidate_and_tarball):
        """An installer that put the package anywhere but the isolated home fails,
        wherever the binary says its cache is."""
        candidate, tarball = candidate_and_tarball
        elsewhere = tmp_path / "elsewhere"

        class Elsewhere(FakeHost):
            def debug_paths(self, binary, env, cwd, diag, label):
                return {"bounded": {}, "paths": {"cache": str(elsewhere / "opencode")}}

            def location(self, env, spec=SPEC):
                return elsewhere / "opencode" / "packages" / spec / "node_modules" / NAME

        host = Elsewhere(tarball, calls_allowed=False)
        result = self._run(tmp_path, candidate, host)
        assert any("outside the probe's root" in p for p in
                   result["criteria"][rc.TREE]["evidence"]["problems"])
        assert host.tool_calls == []


# --------------------------------------------------------------------------- #
# A later release: the upgrade of an existing installation
# --------------------------------------------------------------------------- #

class TestTheUpgrade:
    PREVIOUS = f"{NAME}@0.1.0"

    def _host(self, tmp_path, tarball, **kwargs):
        previous = _pack(_package_tree(tmp_path / "previous", version="0.1.0"), tmp_path / "prev.tgz")
        return FakeHost(tarball, others={self.PREVIOUS: previous}, **kwargs)

    def test_the_new_payload_is_the_one_executing(self, tmp_path, candidate_and_tarball):
        candidate, tarball = candidate_and_tarball
        host = self._host(tmp_path, tarball)
        result = rc.run_criteria(_context(tmp_path, candidate, environment="container",
                                          previous_pin=self.PREVIOUS), host.steps())
        upgrade = result["criteria"][rc.UPGRADE]
        assert upgrade["pass"] is True, upgrade["evidence"].get("problems")
        evidence = upgrade["evidence"]
        assert evidence["before"]["version"] == "0.1.0"
        assert evidence["before"]["location"].endswith(f"packages/{self.PREVIOUS}/node_modules/{NAME}")
        assert evidence["after"]["version"] == VERSION
        assert evidence["after"]["digest"] == candidate["digest"]
        assert evidence["ran_from"] == evidence["after"]["location"] + "/skills/multi-llm"
        # Its own home: the upgrade never touches the first installation's.
        assert "/upgrade/env/" in evidence["before"]["location"]
        assert host.tool_calls[-2:] == [f"status {self.PREVIOUS}", "status"]
        assert [n for n in rc.criteria_for("container", "later") if n not in result["criteria"]] == \
            [rc.OFFLINE]

    def test_an_upgrade_still_running_the_old_payload_fails(self, tmp_path, candidate_and_tarball):
        candidate, tarball = candidate_and_tarball
        host = self._host(tmp_path, tarball, stale_after_upgrade=True)
        result = rc.run_criteria(_context(tmp_path, candidate, environment="container",
                                          previous_pin=self.PREVIOUS), host.steps())
        upgrade = result["criteria"][rc.UPGRADE]
        assert upgrade["pass"] is False
        assert any("new pin: the tool ran skill dir" in p for p in upgrade["evidence"]["problems"])


# --------------------------------------------------------------------------- #
# The network proof
# --------------------------------------------------------------------------- #

def _side(code="EAI_AGAIN", exit_code=1, up=("lo",), addresses=("127.0.0.1",)):
    return {"taken_at": "2026-10-25T00:00:00Z", "interfaces": list(up),
            "npm_view": {"registry": SINGLE, "exit": exit_code, "code": code, "argv": ["npm", "view"]},
            "_detail": {"interfaces": [{"name": n, "up": True} for n in up],
                        "addresses": list(addresses), "npm_view_output": ""}}


def _offline(before=None, after=None, digest_after="d" * 64, problems=()):
    return {"before": before or _side(), "after": after or _side(),
            "payload": {"before": {"location": "/work/p", "digest": "d" * 64},
                        "after": {"location": "/work/p", "digest": digest_after}},
            "status": {"ok": not problems}, "problems": list(problems),
            "variables_set": {"OPENCODE_DISABLE_MODELS_FETCH": "1"},
            "left_unset": ["OPENCODE_DISABLE_AUTOUPDATE", "UV_OFFLINE"], "network_attempts": []}


class TestTheNetworkProof:
    def test_a_network_error_on_both_sides_proves_the_cut(self):
        verdict = rc.offline_criterion(_offline(), "none", "none", SINGLE, "test")
        assert verdict["pass"] is True, verdict["evidence"].get("problems")
        assert set(verdict["network_proof"]) == {"network_mode", "before", "after", "payload"}
        assert set(verdict["network_proof"]["before"]) == {"taken_at", "interfaces", "npm_view"}

    @pytest.mark.parametrize("code", ["E404", "E401", "E403", "ENEEDAUTH", "ENOTCACHED", None])
    def test_an_answer_from_a_registry_is_not_a_cut_network(self, code):
        for side in ("before", "after"):
            offline = _offline(**{side: _side(code=code)})
            verdict = rc.offline_criterion(offline, "none", "none", SINGLE, "test")
            assert verdict["pass"] is False
            assert any("only a network error code proves the cut" in p
                       for p in verdict["evidence"]["problems"]), verdict["evidence"]["problems"]
            assert rc.assess_network_proof(verdict["network_proof"], SINGLE)

    def test_npm_view_that_succeeded_is_not_a_cut_network(self):
        verdict = rc.offline_criterion(_offline(after=_side(code=None, exit_code=0)), "none", "none",
                                       SINGLE, "test")
        assert verdict["pass"] is False

    @pytest.mark.parametrize("kwargs, message", [
        ({"mode": ("bridge", "bridge")}, "NetworkMode is 'bridge'"),
        ({"mode": ("none", "bridge")}, "changed during the run"),
        ({"before": _side(up=("lo", "eth0"))}, "not loopback only"),
        ({"after": _side(addresses=("127.0.0.1", "172.17.0.2"))}, "non-loopback addresses"),
        ({"digest_after": "e" * 64}, "moved or changed the installed payload"),
        ({"problems": ["status exited 1"]}, "status exited 1"),
    ])
    def test_every_other_part_of_the_proof(self, kwargs, message):
        mode = kwargs.pop("mode", ("none", "none"))
        verdict = rc.offline_criterion(_offline(**kwargs), *mode, SINGLE, "test")
        assert verdict["pass"] is False
        assert any(message in p for p in verdict["evidence"]["problems"]), verdict["evidence"]["problems"]

    def test_the_interfaces_that_are_up(self, tmp_path):
        """Measured on linuxkit: --network none still lists the kernel's fallback
        tunnels, all down; only `lo` is up. `bonding_masters` is a file."""
        net = tmp_path / "net"
        for name, flags in (("lo", "0x9"), ("tunl0", "0x80"), ("gretap0", "0x1002"),
                            ("eth0", "0x1003")):
            (net / name).mkdir(parents=True)
            (net / name / "flags").write_text(flags + "\n")
            (net / name / "operstate").write_text("unknown\n" if name == "lo" else "down\n")
        (net / "bonding_masters").write_text("\n")
        snapshot = rc.interfaces_snapshot(net)
        assert snapshot["up"] == ["eth0", "lo"]
        assert [d["name"] for d in snapshot["all"]] == ["eth0", "gretap0", "lo", "tunl0"]
        (net / "eth0" / "flags").write_text("0x1002\n")
        assert rc.interfaces_snapshot(net)["up"] == ["lo"]

    def test_local_addresses(self, tmp_path):
        (tmp_path / "fib_trie").write_text(textwrap.dedent("""\
            Main:
              +-- 127.0.0.0/8 2 0 2
                 |-- 127.0.0.1
                    /32 host LOCAL
                 |-- 127.255.255.255
                    /32 link BROADCAST
            """))
        (tmp_path / "if_inet6").write_text("00000000000000000000000000000001 01 80 10 80       lo\n")
        assert rc.local_addresses(tmp_path) == ["0000:0000:0000:0000:0000:0000:0000:0001",
                                                "127.0.0.1"]

    def test_failed_network_attempts_are_recorded(self):
        log = "\n".join([
            'timestamp=1 level=INFO message=loading path=/x',
            'timestamp=2 level=WARN message="background dependency install failed" '
            'error="FetchError: request to http://127.0.0.1:1/@opencode-ai%2fplugin failed, '
            'reason: connect ECONNREFUSED 127.0.0.1:1"',
            'timestamp=3 level=ERROR message="upgrade check failed" error=EAI_AGAIN',
            'timestamp=4 level=WARN message="slow tool"'])
        assert [line.split()[1] for line in rc.network_attempts(log)] == ["level=WARN", "level=ERROR"]

    def test_a_vm_waits_for_its_interface_to_be_detached(self, tmp_path):
        net = tmp_path / "net"
        for name, flags in (("lo", "0x9"), ("eth0", "0x1003")):
            (net / name).mkdir(parents=True)
            (net / name / "flags").write_text(flags)
        ticks = iter(range(100))

        def detach(_seconds):
            (net / "eth0" / "flags").write_text("0x1002")

        steps = rc.Steps(clock=lambda: next(ticks), sleep=detach, sys_class_net=net)
        assert rc.wait_for_cut(steps, budget_s=50) == {"cut": True, "up": ["lo"]}
        (net / "eth0" / "flags").write_text("0x1003")
        steps = rc.Steps(clock=lambda: next(ticks), sleep=lambda _s: None, sys_class_net=net)
        assert rc.wait_for_cut(steps, budget_s=3) == {"cut": False, "up": ["eth0", "lo"]}

    def test_the_offline_run_sets_only_the_models_fetch_switch(self):
        env, unset = rc.offline_env({"HOME": "/h", "OPENCODE_DISABLE_AUTOUPDATE": "1",
                                     "UV_OFFLINE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1"})
        assert env == {"HOME": "/h", "OPENCODE_DISABLE_MODELS_FETCH": "1"}
        assert unset == ["OPENCODE_DISABLE_AUTOUPDATE", "UV_OFFLINE"]


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

class TestTheReaders:
    def test_npm_error_codes(self):
        assert rc.npm_error_code('{"error": {"code": "EAI_AGAIN"}}', "") == "EAI_AGAIN"
        assert rc.npm_error_code("", "npm error code E404\nnpm error 404 Not Found") == "E404"
        assert rc.npm_error_code("", "npm ERR! code ENOTFOUND\n") == "ENOTFOUND"
        assert rc.npm_error_code("not json", "nothing") is None

    def test_debug_paths(self):
        text = "home       /h\ncache      /h/.cache/opencode\nnot a path line\n"
        assert rc.parse_debug_paths(text) == {"home": "/h", "cache": "/h/.cache/opencode"}

    def test_find_installed_walks_node_modules_only(self, tmp_path):
        def package(where, version):
            where.mkdir(parents=True)
            (where / "package.json").write_text(json.dumps({"name": NAME, "version": version}))
        package(tmp_path / f"packages/{SPEC}/node_modules/{NAME}", VERSION)
        package(tmp_path / f"packages/{NAME}@0.1.0/node_modules/{NAME}", "0.1.0")
        package(tmp_path / f"stray/{NAME}", VERSION)  # not in a node_modules
        package(tmp_path / f"packages/{SPEC}/node_modules/{NAME}/skills/multi-llm/.venv/"
                           f"node_modules/{NAME}", VERSION)  # runtime state is never descended into
        assert rc.find_installed([tmp_path], NAME, VERSION) == [
            tmp_path / f"packages/{SPEC}/node_modules/{NAME}"]
        assert rc.install_root_of(tmp_path / f"packages/{SPEC}/node_modules/{NAME}") == \
            tmp_path / f"packages/{SPEC}"

    def test_the_skill_dir_token_is_the_builds(self):
        assert rc.SKILL_DIR_TOKEN == build.SKILL_DIR_TOKEN

    def test_runtime_requirements(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["PyYAML>=6.0", '
                                                 '"typing_extensions; python_version < \'3.11\'"]\n')
        assert rc.runtime_requirements(tmp_path) == ["pyyaml", "typing-extensions"]

    def test_the_payload_digest_ignores_runtime_state(self, tmp_path):
        tree = _package_tree(tmp_path / "p")
        candidate = _candidate(tree)
        assert rc.compare_tree(tree, candidate)["matches"] is True
        (tree / "skills/multi-llm/.venv").mkdir()
        (tree / "skills/multi-llm/.venv/pyvenv.cfg").write_text("x")
        assert rc.payload_state(tree, candidate)["digest"] == candidate["digest"]
        assert rc.compare_tree(tree, candidate)["matches"] is False  # before a call, nothing extra


# --------------------------------------------------------------------------- #
# The container: the kit stands alone, and the driver's docker calls
# --------------------------------------------------------------------------- #

class TestTheKit:
    def test_the_kit_runs_without_the_repository(self, tmp_path):
        kit = tmp_path / "kit"
        manifest = rc.stage_kit(kit)
        assert sorted(manifest) == sorted(rc.KIT_FILES)
        assert not (kit / "packaging/opencode/build.py").exists()
        assert not (kit / "skills/multi-llm/multi_llm.py").exists()
        code = textwrap.dedent(f"""\
            import importlib.util, json, sys, tempfile
            from pathlib import Path
            spec = importlib.util.spec_from_file_location("rc", "{kit}/packaging/opencode/registry_check.py")
            rc = importlib.util.module_from_spec(spec); sys.modules["rc"] = rc; spec.loader.exec_module(rc)
            assert str(rc.REPO_ROOT) == "{kit}", rc.REPO_ROOT
            assert rc.criteria_for("container", "first")[-1] == "same_pin_restart_matches"
            probe = rc._probe()
            scenario = probe.scripted_call_scenario("multi_llm", probe.STATUS_ARGS, "c1")
            diag = Path(tempfile.mkdtemp())
            with probe.StubModel(scenario, diag, "kit") as stub:
                assert stub.port
            (diag / "f").write_text("x")
            print(json.dumps({{"digest": rc.inventory_digest(diag, ["f"]),
                               "build_loaded": "opencode_build" in sys.modules}}))
            """)
        done = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), capture_output=True,
                              text=True, timeout=120, env={**os.environ, "PYTHONPATH": ""})
        assert done.returncode == 0, done.stderr
        result = json.loads(done.stdout.strip().splitlines()[-1])
        assert result["build_loaded"] is False and len(result["digest"]) == 64


class FakeDocker:
    """Records every docker call; plays a stock container whose phases succeed."""

    def __init__(self, support: Path, operating_system: str = "Docker Desktop", phase=None,
                 offline=None):
        self.support = support
        self.operating_system = operating_system
        self.phase = phase if phase is not None else {
            "criteria": {}, "failures": [], "state": {"payload": {"location": "/work/p"}},
            "details": {}, "measured": {}, "variables_set": {}}
        self.offline = offline if offline is not None else _offline()
        self.calls = []
        self.diag = support / "docker"

    def __call__(self, args, *, label, budget_s=0, check=True):
        args = [str(a) for a in args]
        self.calls.append(args)
        out, code = "", 0
        if args[0] == "info":
            out = json.dumps({"OperatingSystem": self.operating_system, "ServerVersion": "29.9.2"})
        elif args[0] == "inspect":
            out = "none" if args[-1].endswith("-offline") else "bridge"
        elif args[0] == "commit":
            out = "sha256:" + "f" * 64
        elif args[:2] == ["image", "inspect"]:
            code = 1 if args[-1].startswith("sha256:ff") else 0
            out = "sha256:" + "b" * 64
        elif args[0] == "cp" and args[1].endswith(f"{rc.WORK}/out/."):
            dest = Path(args[2])
            dest.mkdir(parents=True, exist_ok=True)
            if "-offline:" in args[1]:
                (dest / "offline.json").write_text(json.dumps(self.offline))
            else:
                (dest / "install.json").write_text(json.dumps(self.phase))
        return SimpleNamespace(returncode=code, describe=lambda: f"exit {code}"), out


def _drive(tmp_path, docker, rehearsal=True):
    candidate = {"sha": "c" * 40, "digest": "d" * 64, "inventory": ["package.json"],
                 "digest_files": ["package.json"]}
    ctx = rc.Context(spec=SPEC, name=NAME, version=VERSION, read_registry=STUB if rehearsal else SINGLE,
                     single_registry=SINGLE, rehearsal=rehearsal, environment="container",
                     candidate=candidate, binary=f"{rc.PIN_PREFIX}/node_modules/.bin/opencode",
                     opencode_pin="1.19.4", probe_root=Path(rc.WORK) / "probe-root")
    base = rc.container_base()
    return rc.drive_container(ctx=ctx, base=base, pin_files=({"name": "pin"}, {"packages": {}}),
                              support=tmp_path / "support", docker=docker), base


class TestTheContainerDriver:
    def test_a_stock_image_nothing_mounted_commit_then_no_network(self, tmp_path):
        docker = FakeDocker(tmp_path / "support")
        driven, base = _drive(tmp_path, docker)
        assert "@sha256:" in base["image"] and "mllm-e2e" not in base["image"]
        runs = [c for c in docker.calls if c[0] == "run"]
        assert len(runs) == 2
        networked, offline = runs
        assert base["image"] in networked
        for call in docker.calls:
            assert not {"-v", "--volume", "--mount"} & set(call), call
            assert not any(str(REPO) in arg for arg in call if not arg.startswith(str(tmp_path))), call
        assert "host.docker.internal:host-gateway" in networked and "--network" not in networked
        assert offline[offline.index("--network") + 1] == "none"
        assert offline[-3] == "mllm-registry-check-offline:" + driven["details"]["run_id"]
        order = [c[0] for c in docker.calls]
        assert order.index("stop") < order.index("commit") < len(order) - 1 - order[::-1].index("run")
        assert docker.calls[-2][:2] == ["rmi", "-f"] and driven["details"]["image_deleted"] is True
        removed = [c[-1] for c in docker.calls if c[:2] == ["rm", "-f"]]
        assert len(removed) == 2
        relay = next(c for c in docker.calls if "relay" in c)
        assert relay[relay.index("--target") + 1] == "host.docker.internal:43210"
        assert driven["offline"]["pass"] is True
        assert driven["offline"]["network_proof"]["network_mode"] == "none"
        assert driven["details"]["networked_phase"]["transport"] == "host-gateway"

    def test_the_phases_run_unprivileged_from_the_kit(self, tmp_path):
        docker = FakeDocker(tmp_path / "support")
        _drive(tmp_path, docker)
        phases = [c for c in docker.calls if c[:1] == ["exec"] and "container-phase" in c]
        assert [c[c.index("--phase") + 1] for c in phases] == ["install", "offline"]
        for call in phases:
            assert call[call.index("-u") + 1] == f"{rc.CONTAINER_UID}:{rc.CONTAINER_UID}"
            assert f"{rc.KIT_ROOT}/packaging/opencode/registry_check.py" in call

    def test_docker_engine_shares_the_hosts_loopback_instead(self, tmp_path):
        docker = FakeDocker(tmp_path / "support", operating_system="Ubuntu 24.04 LTS")
        driven, _ = _drive(tmp_path, docker)
        networked = next(c for c in docker.calls if c[0] == "run")
        assert networked[networked.index("--network") + 1] == "host"
        assert not any("relay" in c for c in docker.calls)
        assert driven["details"]["networked_phase"]["transport"] == "host"

    def test_a_release_run_uses_the_default_bridge_and_no_relay(self, tmp_path):
        docker = FakeDocker(tmp_path / "support")
        driven, _ = _drive(tmp_path, docker, rehearsal=False)
        networked = next(c for c in docker.calls if c[0] == "run")
        assert "--network" not in networked and "--add-host" not in networked
        assert driven["details"]["networked_phase"]["transport"] == "default-bridge"

    def test_no_offline_run_without_an_installed_payload(self, tmp_path):
        phase = {"criteria": {}, "failures": ["x"], "state": {}, "details": {}}
        docker = FakeDocker(tmp_path / "support", phase=phase)
        driven, _ = _drive(tmp_path, docker)
        assert driven["offline"] is None
        assert not any(c[0] == "commit" for c in docker.calls)
        assert sum(1 for c in docker.calls if c[:2] == ["rm", "-f"]) == 2

    def test_a_failed_offline_proof_fails_the_criterion(self, tmp_path):
        docker = FakeDocker(tmp_path / "support", offline=_offline(after=_side(code="E404")))
        driven, _ = _drive(tmp_path, docker)
        assert driven["offline"]["pass"] is False
