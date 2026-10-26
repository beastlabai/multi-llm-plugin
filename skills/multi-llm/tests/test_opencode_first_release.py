"""The laptop first-release target, ``packaging/opencode/first_release.py``.

**How it is driven.** Every subprocess the target spawns goes through one
injectable runner, and here that runner is :class:`World`: a scripted ``git``,
``gh``, ``npm``, ``node`` (row 9) and ``build.py`` whose answers each row
changes one at a time. Nothing reaches GitHub or npm, and nothing is ever
uploaded: "publish" is a call the world records.

**What is real.** The gated tarball is a real ``.tgz`` in a temporary directory,
and the target recomputes it through the real ``build.tarball_facts``, so the
integrity, inventory and extracted-digest refusals grade the code that will run
on release day. The required check-run list is derived from the real
``.github/workflows/test.yml``.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
TEST_YML = REPO_ROOT / ".github" / "workflows" / "test.yml"

sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))
sys.path.insert(0, str(PACKAGE_DIR))
import build as pkg_build  # noqa: E402
import first_release  # noqa: E402
import registry  # noqa: E402

SINGLE = registry.single_registry()
NAME, VERSION = "opencode-multi-llm", "0.1.1"
SPEC, TAG = f"{NAME}@{VERSION}", f"opencode-v{VERSION}"
HEAD, OTHER, TAG_OBJECT, MASTER = "a" * 40, "b" * 40, "c" * 40, "d" * 40
RUN_ID = 40189009375


def _required_from_yaml() -> list:
    """The required list, read from test.yml independently of the code under test."""
    workflow = yaml.safe_load(TEST_YML.read_text(encoding="utf-8"))
    legs = workflow["jobs"]["test"]["strategy"]["matrix"]["os"]
    return [f"test ({leg})" for leg in legs] + ["package"]


def _check_run(name: str, index: int, *, status="completed", conclusion="success",
               app="github-actions", run_id=RUN_ID, sha=HEAD) -> dict:
    return {"id": 121155800000 + index, "name": name, "status": status,
            "conclusion": conclusion if status == "completed" else None,
            "head_sha": sha, "app": app,
            "details_url": f"https://github.com/beastlabai/multi-llm-plugin/actions/runs/"
                           f"{run_id}/job/{121155800000 + index}"}


def _npm_error(code: str) -> str:
    return json.dumps({"error": {"code": code, "summary": f"{code} - GET {SINGLE}/{NAME}"}})


#: ``npm config list --json`` under the upload's flags, nothing configured: the
#: shape npm 10.9.3 and 11.5.1 print (measured), the keys the preflight reads.
CLEAN_CONFIG = {"access": "public", "registry": SINGLE + "/", "prefix": "<scratch>",
                "proxy": None, "https-proxy": None, "strict-ssl": True, "ca": None,
                "cafile": None, "auth-type": "web"}


def _config_text(layers: dict) -> str:
    """``npm config list``'s text: ``{'"user" config from /h/.npmrc': {key: value}}``,
    each value already in npm's display form."""
    lines = []
    for header, settings in layers.items():
        lines += [f"; {header}", ""]
        lines += [f"{key} = {value}" for key, value in settings.items()]
        lines.append("")
    lines += ["; node bin location = /usr/bin/node", "; npm version = 10.9.3",
              "; Run `npm config ls -l` to show all defaults."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The gated tarball and its stamp
# ---------------------------------------------------------------------------

def _pack(path: Path, files: dict) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative, text in sorted(files.items()):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"package/{relative}")
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))


PAYLOAD = {
    "package.json": json.dumps({"name": NAME, "version": VERSION}),
    "README.md": "# opencode-multi-llm\n",
    "LICENSE": "MIT\n",
    "dist/index.mjs": "export const MultiLlmPlugin = async () => ({});\n",
    "dist/command.md": "run {SKILL_DIR}\n",
    "skills/multi-llm/SKILL.md": "the skill\n",
}


@pytest.fixture
def gated(tmp_path):
    """A packed tarball and the schema-2 stamp binding it, as the full gate writes."""
    tarball = tmp_path / f"{NAME}-{VERSION}.tgz"
    _pack(tarball, PAYLOAD)
    facts = pkg_build.tarball_facts(tarball)
    stamp = {
        "schema": 2, "tier": "full", "digest": facts["digest"], "version": VERSION,
        "stamped_at": "2026-10-25T00:00:00Z", "session": "s-1", "candidate_sha": HEAD,
        "host_version": "1.19.4", "image_id": "sha256:" + "e" * 64, "registry": SINGLE,
        "tarball": {k: facts[k] for k in ("path", "integrity", "sha256", "files", "digest")},
        "rows": {row: "pass" for row in ("1", "2", "3", "4", "5", "5b", "6", "7", "8", "9")},
    }
    stamp_path = tmp_path / ".package-check.stamp"
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    manifest = tmp_path / "package.json"
    manifest.write_text(json.dumps({"name": NAME, "version": VERSION}), encoding="utf-8")
    return {"tarball": tarball, "stamp": stamp, "stamp_path": stamp_path,
            "manifest": manifest, "facts": facts}


def _restamp(gated, **changes):
    stamp = json.loads(json.dumps(gated["stamp"]))
    tarball_changes = changes.pop("tarball", {})
    stamp.update(changes)
    stamp["tarball"].update(tarball_changes)
    gated["stamp_path"].write_text(json.dumps(stamp), encoding="utf-8")


# ---------------------------------------------------------------------------
# The world: git, gh, npm, node and build.py, scripted
# ---------------------------------------------------------------------------

class World:
    """The runner. Each attribute is one answer; a row changes the one it grades."""

    def __init__(self):
        self.calls: list = []
        self.status = ""
        self.check_stamp = (0, "the full gate passed on these exact bytes\n", "")
        self.tag_type = "tag"            # "tag" annotated, "commit" lightweight, None missing
        self.tag_commit = HEAD
        self.fetch = 0
        self.ancestor = 0
        self.test_yml = TEST_YML.read_text(encoding="utf-8")
        self.check_runs = [_check_run(n, i) for i, n in enumerate(_required_from_yaml())]
        self.check_runs_exit = 0
        self.workflow_runs = {RUN_ID: {"id": RUN_ID, "path": ".github/workflows/test.yml",
                                       "head_sha": HEAD, "event": "push", "run_attempt": 1}}
        self.bump = (0, "shared version 0.1.1 is new for both families\n", "")
        self.whoami = (0, "beastlabai\n", "")
        self.scan = (0, "sweep clean\n", "")
        self.row9 = "absent"
        self.publish = 0
        self.views = [(1, _npm_error("E404"))]     # then the last entry repeats
        self.versions = (1, _npm_error("E404"))     # `npm view <name> versions`
        self.capabilities = yaml.safe_dump({"hosts": {"opencode": {"package": {
            "registry": "npm", "name": NAME, "version": VERSION}}}})
        self.view_count = 0
        self.config = dict(CLEAN_CONFIG)
        self.config_text = _config_text({
            '"user" config from /home/maintainer/.npmrc': {
                "//registry.npmjs.org/:_authToken": "(protected)"},
            '"cli" config from command line options': {
                "access": '"public"', "ignore-scripts": "true",
                "registry": f'"{SINGLE}/"'}})

    def __call__(self, argv, *, capture=True, timeout=None, cwd=None, inherit_npm_config=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "capture": capture, "cwd": cwd,
                           "inherit_npm_config": inherit_npm_config})
        done = self._answer(argv, capture)
        code, out, err = done if len(done) == 3 else (*done, "")
        return first_release.Completed(argv, code, out, err)

    def _answer(self, argv, capture):
        tool, rest = argv[0], argv[1:]
        if tool == "git":
            return self._git(rest)
        if tool == "gh":
            return self._gh(rest)
        if tool == "node":
            assert rest[0].endswith("registry_absent.mjs")
            assert rest[1:3] == ["--registry", SINGLE]
            out = Path(rest[rest.index("--out") + 1])
            outcome = self.row9
            out.write_text(json.dumps({
                "row": "9", "pass": outcome == "absent", "registry": SINGLE, "name": NAME,
                "version": VERSION, "npm_exit": 0 if outcome == "published" else 1,
                "code": {"absent": "E404", "published": None}.get(outcome, "E401"),
                "outcome": outcome,
            }), encoding="utf-8")
            return (0 if outcome == "absent" else 1, f"row 9: {outcome}\n")
        if tool == sys.executable:
            flag = rest[1]
            if flag == "--check-stamp":
                return self.check_stamp
            if flag == "--check-bump":
                return self.bump
            if flag == "--scan-tarball":
                return self.scan
        if tool == "npm":
            if rest[0] == "whoami":
                return self.whoami
            if rest[0] == "publish":
                assert capture is False, "the upload must keep the terminal for npm's 2FA"
                return (self.publish, "")
            if rest[:3] == ["view", NAME, "versions"]:
                return self.versions
            if rest[0] == "view":
                answer = self.views[min(self.view_count, len(self.views) - 1)]
                self.view_count += 1
                return answer
            if rest[:2] == ["config", "list"]:
                if "--json" in rest:
                    return (0, json.dumps(self.config, indent=2) + "\n")
                return (0, self.config_text)
        raise AssertionError(f"unexpected call {argv}")

    def _git(self, rest):
        if rest == ["rev-parse", "--verify", "HEAD"]:
            return (0, HEAD + "\n")
        if rest == ["status", "--porcelain"]:
            return (0, self.status)
        if rest[:2] == ["cat-file", "-t"]:
            assert rest[2] == f"refs/tags/{TAG}"
            if self.tag_type is None:
                return (128, "", f"fatal: Not a valid object name refs/tags/{TAG}")
            return (0, self.tag_type + "\n")
        if rest == ["rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"]:
            return (0, self.tag_commit + "\n")
        if rest == ["rev-parse", "--verify", f"refs/tags/{TAG}"]:
            return (0, TAG_OBJECT + "\n")
        if rest == ["fetch", "origin"]:
            return (self.fetch, "", "" if self.fetch == 0 else "fatal: unable to access")
        if rest == ["merge-base", "--is-ancestor", "HEAD", "origin/master"]:
            return (self.ancestor, "", "" if self.ancestor in (0, 1) else "fatal: bad revision")
        if rest == ["rev-parse", "origin/master"]:
            return (0, MASTER + "\n")
        if rest == ["show", f"{HEAD}:.github/workflows/test.yml"]:
            return (0, self.test_yml)
        if rest == ["show", f"{HEAD}:skills/multi-llm/hosts/capabilities.yaml"]:
            return (0, self.capabilities)
        raise AssertionError(f"unexpected git call {rest}")

    def _gh(self, rest):
        assert rest[0] == "api"
        assert "-f" not in rest and "-F" not in rest and "--method" not in rest and "-X" not in rest, (
            "a gh call with fields defaults to POST; the target only reads"
        )
        endpoint = rest[1]
        if endpoint == (f"repos/beastlabai/multi-llm-plugin/commits/{HEAD}/check-runs"
                        "?filter=latest&per_page=100"):
            assert "--paginate" in rest
            if self.check_runs_exit:
                return (self.check_runs_exit, "", "gh: HTTP 401: Bad credentials")
            return (0, "".join(json.dumps(r) + "\n" for r in self.check_runs))
        match = re.fullmatch(r"repos/beastlabai/multi-llm-plugin/actions/runs/(\d+)", endpoint)
        if match:
            info = self.workflow_runs.get(int(match.group(1)))
            return (0, json.dumps(info) + "\n") if info else (1, "", "gh: Not Found (HTTP 404)")
        raise AssertionError(f"unexpected gh call {rest}")

    # -- reading the record ---------------------------------------------------

    def argvs(self, prefix):
        return [c["argv"] for c in self.calls if c["argv"][:len(prefix)] == prefix]

    def npm_calls(self, *words):
        return [c for c in self.calls if c["argv"][:1 + len(words)] == ["npm", *words]]

    @property
    def published(self) -> bool:
        return bool(self.argvs(["npm", "publish"]))


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def world():
    return World()


@pytest.fixture
def release(gated, world, tmp_path):
    clock = Clock()
    out, err = [], []
    made = first_release.Release(
        runner=world, sleep=clock.sleep, clock=clock, out=out.append, err=err.append,
        stamp_path=gated["stamp_path"], manifest_path=gated["manifest"],
        evidence_dir=tmp_path / "evidence", poll_seconds=60.0, poll_interval=10.0,
    )
    made.lines, made.errors, made.clock_ = out, err, clock
    return made


def _refused(release, condition):
    """The run refused under *condition*, before any upload."""
    assert [r.condition for r in release.refusals].count(condition) == 1, release.errors
    assert any(line.startswith(f"REFUSING ({condition}):") for line in release.errors)


def _good_views(gated):
    return [(1, _npm_error("E404")), (0, json.dumps(gated["facts"]["integrity"]) + "\n")]


# ---------------------------------------------------------------------------
# The green path
# ---------------------------------------------------------------------------

class TestGreenPath:

    def test_it_uploads_the_gated_file_and_confirms_the_integrity(self, release, world, gated):
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == first_release.EXIT_OK, release.errors
        (call,) = world.npm_calls("publish")
        publish = call["argv"]
        assert publish[:-1] == ["npm", "publish", str(gated["tarball"]), "--ignore-scripts",
                                "--access", "public", f"--registry={SINGLE}"]
        # npm's project root pinned to the empty scratch directory it runs in.
        assert publish[-1] == f"--prefix={call['cwd']}"
        assert not Path(call["cwd"]).exists(), "the upload's scratch directory was left behind"
        # Row 9 is the last thing before the upload.
        order = [c["argv"][0] if c["argv"][0] != "npm" else f"npm {c['argv'][1]}"
                 for c in world.calls]
        assert order[order.index("npm publish") - 1] == "node"
        assert any("git push origin " + TAG in line for line in release.lines)

    def test_the_evidence_binds_tag_commit_checks_and_integrity(self, release, world, gated):
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0
        (path,) = list((release.evidence_dir).glob(f"*-{TAG}.json"))
        evidence = json.loads(path.read_text(encoding="utf-8"))
        assert evidence["tag"] == {"name": TAG, "object": TAG_OBJECT}
        assert evidence["commit"] == HEAD and evidence["origin_master"] == MASTER
        assert evidence["provenance"] is False
        assert evidence["gated_integrity"] == gated["facts"]["integrity"]
        assert evidence["required_checks"] == _required_from_yaml()
        assert {(r["name"], r["workflow_run_id"]) for r in evidence["check_runs"]} == {
            (name, RUN_ID) for name in _required_from_yaml()}
        assert all(isinstance(r["id"], int) for r in evidence["check_runs"])
        assert evidence["publish"]["exit"] == 0 and evidence["publish"]["started_at"]
        confirmation = evidence["confirmation"]
        assert confirmation["state"] == "confirmed" and confirmation["attempts"] == 2
        assert confirmation["output"] == json.dumps(gated["facts"]["integrity"])
        assert confirmation["read_at"]
        assert evidence["row_9"]["outcome"] == "absent"

    def test_every_confirming_read_uses_a_fresh_cache_and_the_single_registry(
        self, release, world, gated,
    ):
        world.views = [(1, _npm_error("E404"))] * 3 + [_good_views(gated)[1]]
        assert release.run(dry_run=False) == 0
        reads = world.argvs(["npm", "view", SPEC])
        assert len(reads) == 4
        caches = [r[r.index("--cache") + 1] for r in reads]
        assert len(set(caches)) == 4 and not any(Path(c).exists() for c in caches)
        assert all(r[:4] == ["npm", "view", SPEC, "dist.integrity"] for r in reads)
        assert all(f"--registry={SINGLE}" in r for r in reads)
        assert release.clock_.sleeps == [10.0] * 3

    def test_a_dry_run_runs_every_preflight_and_uploads_nothing(self, release, world, gated):
        assert release.run(dry_run=True) == 0
        assert not world.published
        assert not release.evidence_dir.exists()
        # As ``shlex.join`` prints it: a Windows path's backslashes get quoted.
        upload = f"npm publish {shlex.quote(str(gated['tarball']))}"
        assert any(upload in line and "--ignore-scripts" in line
                   for line in release.lines)
        ran = {c["argv"][0] if c["argv"][0] != sys.executable else c["argv"][2]
               for c in world.calls}
        assert ran == {"git", "gh", "npm", "node", "--check-stamp", "--check-bump",
                       "--scan-tarball"}
        assert {f"  ok  {condition}" for condition, _ in first_release.Release.PREFLIGHTS} \
            <= set(release.lines)

    def test_every_subprocess_goes_through_the_runner(self, release, world, gated, monkeypatch):
        """The suite's stubs see every spawn: a real one would fail this row."""
        def no_spawn(*args, **kwargs):
            raise AssertionError(f"spawned outside the runner: {args}")

        monkeypatch.setattr(subprocess, "run", no_spawn)
        monkeypatch.setattr(subprocess, "Popen", no_spawn)
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0


# ---------------------------------------------------------------------------
# Git: the tree, the tag, origin/master
# ---------------------------------------------------------------------------

class TestGitBinding:

    def test_a_dirty_tree_refuses(self, release, world):
        world.status = " M Makefile\n?? stray.txt\n"
        assert release.run(dry_run=False) == first_release.EXIT_REFUSED
        _refused(release, "clean-tree")
        assert not world.published
        refusal = next(line for line in release.errors if line.startswith("REFUSING (clean-tree)"))
        assert "release-gate record" not in refusal

    @pytest.mark.parametrize("status", [
        "?? skills/multi-llm/tests/container/records/20261025T120000Z-opencode-release-gate.json\n",
        " M Makefile\n M skills/multi-llm/tests/container/records/"
        "20261025T120000Z-opencode-release-gate.json\n",
    ])
    def test_a_record_in_the_tree_says_where_it_belongs(self, release, world, status):
        """The gate exports to the gitignored .gate/export/; a record copied into
        records/ belongs in the default branch's commit after the fast-forward,
        and the refusal says so, and how to come back to the tag."""
        world.status = status
        assert release.run(dry_run=False) == first_release.EXIT_REFUSED
        _refused(release, "clean-tree")
        refusal = next(line for line in release.errors if line.startswith("REFUSING (clean-tree)"))
        assert "default branch's own commit after the fast-forward" in refusal
        tag = release.facts["tag"]
        assert f"check {tag}'s commit out again" in refusal
        assert "make opencode-publish-first" in refusal
        assert not world.published

    def test_a_lightweight_tag_refuses(self, release, world):
        world.tag_type = "commit"
        assert release.run(dry_run=False) == 1
        _refused(release, "tag")
        assert "lightweight" in release.errors[0] and not world.published

    def test_a_missing_tag_refuses(self, release, world):
        world.tag_type = None
        assert release.run(dry_run=False) == 1
        _refused(release, "tag")
        assert f"git tag -a {TAG}" in release.errors[0]

    def test_a_tag_naming_another_commit_refuses(self, release, world):
        world.tag_commit = OTHER
        assert release.run(dry_run=False) == 1
        _refused(release, "tag")
        assert OTHER in release.errors[0] and not world.published

    def test_head_absent_from_origin_master_refuses(self, release, world):
        world.ancestor = 1
        assert release.run(dry_run=False) == 1
        _refused(release, "origin-master")
        assert "not on origin/master" in release.errors[0]
        # Fetched first, so the answer is about origin as it stands.
        order = [c["argv"][1:3] for c in world.calls if c["argv"][0] == "git"]
        assert order.index(["fetch", "origin"]) < order.index(["merge-base", "--is-ancestor"])

    @pytest.mark.parametrize("attribute, value", [("fetch", 128), ("ancestor", 128)])
    def test_a_failed_git_call_refuses(self, release, world, attribute, value):
        setattr(world, attribute, value)
        assert release.run(dry_run=False) == 1
        _refused(release, "origin-master")
        assert not world.published


# ---------------------------------------------------------------------------
# The required check runs
# ---------------------------------------------------------------------------

def _set(world, name, **changes):
    world.check_runs = [
        _check_run(r["name"], i, **{**{"status": r["status"], "conclusion": r["conclusion"]},
                                    **changes})
        if r["name"] == name else r
        for i, r in enumerate(world.check_runs)
    ]


class TestRequiredChecks:

    def test_the_required_list_is_derived_from_test_yml(self):
        assert first_release.required_check_names(TEST_YML.read_text(encoding="utf-8")) \
            == _required_from_yaml()
        # And it is the plan's three legs plus package today, so a reader of the
        # plan and of this file are reading the same list.
        assert set(_required_from_yaml()) >= {"test (ubuntu-latest)", "package"}

    def test_a_new_matrix_leg_becomes_required(self, release, world):
        """test.yml is read at HEAD: a leg added there is required here, unedited."""
        world.test_yml = world.test_yml.replace(
            "os: [ubuntu-latest, macos-latest, windows-latest]",
            "os: [ubuntu-latest, macos-latest, windows-latest, ubuntu-24.04-arm]")
        assert "ubuntu-24.04-arm" in world.test_yml
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert "test (ubuntu-24.04-arm): missing" in release.errors[0]

    def test_a_dropped_leg_is_no_longer_required(self, release, world, gated):
        world.test_yml = world.test_yml.replace(
            "os: [ubuntu-latest, macos-latest, windows-latest]", "os: [ubuntu-latest]")
        world.check_runs = [r for r in world.check_runs
                            if r["name"] in ("test (ubuntu-latest)", "package")]
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0, release.errors

    @pytest.mark.parametrize("edit", [
        lambda y: y.replace("  test:\n", "  test:\n    name: Tests\n", 1),
        lambda y: y.replace("os: [ubuntu-latest, macos-latest, windows-latest]",
                            "os: [ubuntu-latest]\n        python: ['3.10']"),
        lambda y: y.replace("  package:\n", "  pkg:\n", 1),
    ], ids=["job-name", "second-dimension", "no-package-job"])
    def test_a_shape_it_cannot_derive_refuses(self, edit):
        with pytest.raises(first_release.Refusal) as caught:
            first_release.required_check_names(edit(TEST_YML.read_text(encoding="utf-8")))
        assert caught.value.condition == "required-checks"

    @pytest.mark.parametrize("name, changes, expected", [
        ("test (ubuntu-latest)", None, "test (ubuntu-latest): missing"),
        ("package", None, "package: missing"),
        ("test (ubuntu-latest)", {"status": "in_progress"}, "test (ubuntu-latest): in_progress"),
        ("package", {"status": "queued"}, "package: queued"),
        ("test (ubuntu-latest)", {"conclusion": "cancelled"}, "test (ubuntu-latest): cancelled"),
        ("package", {"conclusion": "skipped"}, "package: skipped"),
        ("package", {"conclusion": "failure"}, "package: failure"),
        ("test (macos-latest)", {"conclusion": "failure"}, "test (macos-latest): failure"),
        ("test (windows-latest)", {"conclusion": "timed_out"},
         "test (windows-latest): timed_out"),
        ("test (windows-latest)", {"conclusion": "failure"}, "test (windows-latest): failure"),
    ])
    def test_anything_but_success_refuses(self, release, world, name, changes, expected):
        if changes is None:
            world.check_runs = [r for r in world.check_runs if r["name"] != name]
        else:
            _set(world, name, **changes)
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert expected in release.errors[0]
        assert not world.published

    @pytest.mark.parametrize("leg", ["test (macos-latest)", "test (windows-latest)"])
    def test_a_red_leg_beside_a_green_ubuntu_one_refuses(self, release, world, leg):
        """The legs gate equally: Ubuntu green does not carry a red macOS or Windows."""
        _set(world, leg, conclusion="failure")
        ubuntu = next(r for r in world.check_runs if r["name"] == "test (ubuntu-latest)")
        assert ubuntu["conclusion"] == "success"
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert f"{leg}: failure" in release.errors[0]

    def test_a_check_from_another_app_is_not_the_tests_workflow(self, release, world):
        world.check_runs = [r for r in world.check_runs if r["name"] != "package"]
        world.check_runs.append(_check_run("package", 99, app="some-other-app"))
        assert release.run(dry_run=False) == 1
        assert "package: missing" in release.errors[0]

    def test_a_check_from_another_workflow_refuses(self, release, world):
        world.check_runs = [r for r in world.check_runs if r["name"] != "package"]
        world.check_runs.append(_check_run("package", 99, run_id=777))
        world.workflow_runs[777] = {"id": 777, "path": ".github/workflows/other.yml",
                                    "head_sha": HEAD, "event": "push", "run_attempt": 1}
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert "other.yml" in release.errors[0]

    def test_every_run_of_a_name_must_be_green(self, release, world):
        """A push run and a pull-request run on one SHA both report each job;
        a red one is not hidden by a green one."""
        world.check_runs.append(_check_run("test (macos-latest)", 50, conclusion="failure",
                                           run_id=555))
        world.workflow_runs[555] = {**world.workflow_runs[RUN_ID], "id": 555,
                                    "event": "pull_request"}
        assert release.run(dry_run=False) == 1
        assert "test (macos-latest): failure" in release.errors[0]

    @pytest.mark.parametrize("failure", ["listing", "workflow-run"])
    def test_a_failed_gh_call_refuses(self, release, world, failure):
        if failure == "listing":
            world.check_runs_exit = 1
        else:
            world.workflow_runs.clear()
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert not world.published


# ---------------------------------------------------------------------------
# The stamp, the bump check, npm auth, row 9
# ---------------------------------------------------------------------------

class TestOtherPreflights:

    def test_a_failed_check_stamp_refuses(self, release, world):
        world.check_stamp = (1, "", "REFUSING: the staged tree has changed since the gate ran")
        assert release.run(dry_run=False) == 1
        _refused(release, "stamp")
        _refused(release, "gated-tarball")   # nothing names a gated tarball now
        assert not world.published

    @pytest.mark.parametrize("changes", [
        {"schema": 1}, {"tier": "ci"}, {"tarball": {"integrity": ""}},
        {"version": "0.1.0"}, {"candidate_sha": OTHER}, {"registry": "https://registry.yarnpkg.com"},
    ], ids=["schema-1", "ci-tier", "no-integrity", "version", "candidate-sha", "registry"])
    def test_a_stamp_that_does_not_bind_this_release_refuses(self, release, world, gated, changes):
        _restamp(gated, **changes)
        assert release.run(dry_run=False) == 1
        _refused(release, "stamp")
        assert not world.published

    def test_a_stamp_that_is_not_an_object_refuses_by_name(self, release, world, gated):
        gated["stamp_path"].write_text("[]", encoding="utf-8")
        assert release.run(dry_run=False) == 1
        _refused(release, "stamp")

    def test_a_preflight_that_crashes_is_a_named_refusal(self, release, world, monkeypatch):
        """Fail closed AND legibly: an exception is never a pass, nor a traceback."""
        def boom(_text):
            raise RuntimeError("boom")

        monkeypatch.setattr(first_release, "required_check_names", boom)
        assert release.run(dry_run=False) == 1
        _refused(release, "required-checks")
        assert "could not be checked (RuntimeError: boom)" in release.errors[0]
        assert not world.published

    def test_gh_output_that_is_not_check_runs_reads_as_missing(self, release, world):
        world.check_runs = ["oops", 3]
        assert release.run(dry_run=False) == 1
        assert "test (ubuntu-latest): missing" in release.errors[0]

    def test_a_missing_bump_check_flag_refuses(self, release, world):
        world.bump = (2, "", "usage: build.py ...\nbuild.py: error: unrecognized arguments: "
                             "--check-bump\n")
        assert release.run(dry_run=False) == 1
        _refused(release, "bump-check")
        assert "has no --check-bump" in next(e for e in release.errors if "bump-check" in e)

    def test_a_failing_bump_check_refuses(self, release, world):
        world.bump = (1, "", "REFUSING: 0.1.1 is already a codex-v tag\n")
        assert release.run(dry_run=False) == 1
        _refused(release, "bump-check")

    def test_no_npm_login_refuses_before_the_upload(self, release, world):
        world.whoami = (1, "", "npm error code ENEEDAUTH\nnpm error need auth\n"
                               "npm error A complete log of this run can be found in: /x.log\n")
        assert release.run(dry_run=False) == 1
        _refused(release, "npm-auth")
        assert "ENEEDAUTH" in next(e for e in release.errors if "npm-auth" in e)
        whoami = world.argvs(["npm", "whoami"])[0]
        assert f"--registry={SINGLE}" in whoami

    @pytest.mark.parametrize("outcome", ["published", "unknown"])
    def test_row_9_refuses_unless_absent(self, release, world, outcome):
        world.row9 = outcome
        assert release.run(dry_run=False) == 1
        _refused(release, "row-9")
        assert not world.published

    def test_each_refusal_is_named_once(self, release, world):
        world.status = " M x\n"
        world.tag_type = "commit"
        world.ancestor = 1
        world.row9 = "published"
        assert release.run(dry_run=True) == 1
        assert [r.condition for r in release.refusals] == [
            "clean-tree", "tag", "origin-master", "row-9"]


# ---------------------------------------------------------------------------
# The gated tarball, recomputed immediately before the upload
# ---------------------------------------------------------------------------

class TestFirstReleaseOnly:
    """The laptop path is the first release and nothing else. Found by the Docker
    dress rehearsal (2026-10-25): with 0.1.1 published and promoted, a 0.1.2
    dry run refused only what a gate run and a CI run supply, because row 9
    reads "this version is missing from an existing package" as absent too."""

    @pytest.mark.parametrize("listing", [json.dumps(["0.1.1"]) + "\n", '"0.1.1"\n'])
    def test_a_package_npm_already_lists_refuses(self, release, world, listing):
        world.versions = (0, listing)
        assert release.run(dry_run=False) == 1
        _refused(release, "first-release")
        message = next(e for e in release.errors if e.startswith("REFUSING (first-release)"))
        assert "0.1.1" in message and "workflow dispatch" in message
        assert not world.published

    def test_a_promoted_public_version_refuses_even_with_nothing_on_npm(self, release, world):
        world.capabilities = yaml.safe_dump({"hosts": {"opencode": {"package": {
            "name": NAME, "version": VERSION, "public_version": "0.1.0"}}}})
        assert release.run(dry_run=False) == 1
        _refused(release, "first-release")
        assert any("public_version 0.1.0" in e and "later release" in e for e in release.errors)
        assert not world.published

    @pytest.mark.parametrize("answer", [
        (1, _npm_error("E403")),
        (1, _npm_error("ECONNRESET")),
        (1, json.dumps({"error": {"code": "E404", "summary": f"Not Found - GET {SINGLE}/-/whoami"}})),
        (0, "not json\n"),
        (124, "", "npm: no answer within 90 s"),
    ])
    def test_anything_but_e404_on_the_package_is_an_unknown_and_refuses(
        self, release, world, answer,
    ):
        world.versions = answer
        assert release.run(dry_run=False) == 1
        _refused(release, "first-release")
        assert not world.published

    def test_unreadable_capabilities_refuse(self, release, world):
        world.capabilities = "hosts: [unclosed\n"
        assert release.run(dry_run=False) == 1
        _refused(release, "first-release")

    def test_the_read_runs_on_none_of_the_callers_configuration(self, release, world, gated):
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0
        (read,) = world.npm_calls("view", NAME, "versions")
        argv, fresh = read["argv"], str(read["cwd"])
        assert argv[argv.index("--registry") + 1] == SINGLE
        assert read["inherit_npm_config"] is False
        assert f"--prefix={fresh}" in argv
        assert f"--userconfig={Path(fresh) / 'npmrc-user'}" in argv
        assert f"--globalconfig={Path(fresh) / 'npmrc-global'}" in argv
        assert "--prefer-online" in argv and not Path(fresh).exists()


class TestGatedTarball:

    def test_a_missing_gated_tarball_refuses(self, release, world, gated):
        gated["tarball"].unlink()
        assert release.run(dry_run=False) == 1
        _refused(release, "gated-tarball")
        assert "missing" in next(e for e in release.errors if "gated-tarball" in e)
        assert not world.published

    def test_a_repacked_tarball_refuses_on_integrity_inventory_and_digest(
        self, release, world, gated,
    ):
        _pack(gated["tarball"], {**PAYLOAD, "dist/extra.js": "surplus\n"})
        assert release.run(dry_run=False) == 1
        message = next(e for e in release.errors if "gated-tarball" in e)
        for word in ("integrity", "sha256", "digest", "inventory differs"):
            assert word in message
        assert "dist/extra.js" in message and not world.published

    def test_an_integrity_the_stamp_does_not_carry_refuses(self, release, world, gated):
        _restamp(gated, tarball={"integrity": "sha512-" + "B" * 86 + "=="})
        assert release.run(dry_run=False) == 1
        message = next(e for e in release.errors if "gated-tarball" in e)
        assert "integrity" in message and "inventory" not in message
        assert not world.published

    def test_an_inventory_the_stamp_does_not_carry_refuses(self, release, world, gated):
        _restamp(gated, tarball={"files": gated["stamp"]["tarball"]["files"] + ["dist/gone.js"]})
        assert release.run(dry_run=False) == 1
        message = next(e for e in release.errors if "gated-tarball" in e)
        assert "inventory differs" in message and "dist/gone.js" in message
        assert not world.published

    def test_an_extracted_digest_the_stamp_does_not_carry_refuses(self, release, world, gated):
        """Integrity and inventory agree; only the builder's tree digest over the
        extracted package differs. Still refused."""
        _restamp(gated, digest="f" * 64, tarball={"digest": "f" * 64})
        assert release.run(dry_run=False) == 1
        message = next(e for e in release.errors if "gated-tarball" in e)
        assert "digest" in message and "integrity" not in message
        assert not world.published

    def test_a_failed_secret_sweep_refuses(self, release, world):
        world.scan = (1, "", "ERROR: a credential-shaped string in package/dist/index.mjs")
        assert release.run(dry_run=False) == 1
        _refused(release, "gated-tarball")
        scans = world.argvs([sys.executable, str(first_release.BUILD), "--scan-tarball"])
        assert scans and not world.published


# ---------------------------------------------------------------------------
# The upload's argv and the provenance choice
# ---------------------------------------------------------------------------

class TestPublishArgv:

    @pytest.mark.parametrize("mutate, why", [
        (lambda a, g: [a[0], a[1], str(PACKAGE_DIR)] + a[3:], "not the gated tarball"),
        (lambda a, g: [x for x in a if x != "--ignore-scripts"], "--ignore-scripts"),
        (lambda a, g: a + ["--provenance"], "provenance"),
        (lambda a, g: a + ["--provenance=true"], "provenance"),
        (lambda a, g: [x if not x.startswith("--registry") else "--registry=https://r.example"
                       for x in a], "single registry"),
        (lambda a, g: a + ["--registry=https://r.example"], "single registry"),
        (lambda a, g: [x for x in a if x not in ("--access", "public")], "--access public"),
        (lambda a, g: ["npm", "pack"] + a[2:], "not an `npm publish"),
        (lambda a, g: [x for x in a if not x.startswith("--prefix")], "project root"),
        (lambda a, g: a + ["--prefix=/home/maintainer/project"], "project root"),
        (lambda a, g: [a[0], a[1], os.path.relpath(a[2])] + a[3:], "relative path"),
    ], ids=["directory", "scripts", "provenance", "provenance-value", "other-registry",
            "two-registries", "access", "not-publish", "no-prefix", "two-prefixes",
            "relative-tarball"])
    def test_an_unguarded_argv_is_refused_before_npm_runs(
        self, release, world, gated, monkeypatch, mutate, why,
    ):
        honest = release.publish_argv
        monkeypatch.setattr(release, "publish_argv", lambda: mutate(honest(), gated))
        assert release.run(dry_run=False) == 1
        assert any(e.startswith("REFUSING (publish-argv):") and why in e for e in release.errors)
        assert not world.published

    def test_the_laptop_path_never_asks_for_provenance(self, release, world, gated):
        """The local half of the provenance choice: the laptop cannot sign one,
        so neither its argv nor its environment may ask. The CI half (the
        workflow's publish keeps `--provenance`) is the workflow's own suite's."""
        assert first_release.LOCAL_PROVENANCE is False
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0
        (publish,) = world.argvs(["npm", "publish"])
        assert not any(token.startswith("--provenance") for token in publish)
        assert "provenance" not in " ".join(publish)

    def test_the_runner_strips_provenance_and_the_dirty_opt_in(self, monkeypatch):
        monkeypatch.setenv("npm_config_provenance", "true")
        monkeypatch.setenv("NPM_CONFIG_PROVENANCE", "true")
        monkeypatch.setenv("MULTI_LLM_ALLOW_DIRTY", "1")
        done = first_release.Runner()(
            [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"])
        names = {name.lower() for name in json.loads(done.stdout)}
        assert not names & {"npm_config_provenance", "multi_llm_allow_dirty"}


# ---------------------------------------------------------------------------
# npm's configuration: the upload's, held to the defaults; the reads', emptied
# ---------------------------------------------------------------------------

USER_RC = '"user" config from /home/maintainer/.npmrc'
PROJECT_RC = '"project" config from /home/maintainer/src/.npmrc'
ENV_RC = '"env" config from environment'


class TestNpmConfiguration:
    """The upload must keep the maintainer's user config (the token lives
    there), so it cannot be emptied as row 9 and the confirming read empty
    theirs. Instead every npm call runs in one empty scratch directory pinned as
    npm's project root, and the ``npm-config`` preflight reads the configuration
    the upload will run under and refuses anything that can reach or trust a
    server other than the registry, naming the setting and the layer."""

    def test_the_preflight_reads_exactly_what_the_upload_runs_under(self, release, world, gated):
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0
        (listing,) = [c for c in world.npm_calls("config", "list") if "--json" in c["argv"]]
        (whoami,) = world.npm_calls("whoami")
        (publish,) = world.npm_calls("publish")
        scratch = publish["cwd"]
        assert listing["cwd"] == whoami["cwd"] == scratch
        assert listing["argv"][4:] == publish["argv"][3:], "not the upload's own flags"
        assert f"--prefix={scratch}" in whoami["argv"] and f"--registry={SINGLE}" in whoami["argv"]
        # Clean: the text listing (sources) is only read to name a refusal.
        assert not [c for c in world.npm_calls("config", "list") if "--json" not in c["argv"]]
        # The token-bearing user config is the maintainer's own: never emptied here.
        assert not any(a.startswith(("--userconfig", "--globalconfig"))
                       for c in (listing, whoami, publish) for a in c["argv"])
        assert listing["inherit_npm_config"] and publish["inherit_npm_config"]
        order = [c for c, _ in first_release.Release.PREFLIGHTS]
        assert order.index("npm-config") < order.index("npm-auth")

    @pytest.mark.parametrize("key, value, display, layer, why", [
        ("https-proxy", "http://127.0.0.1:8443/", '"http://127.0.0.1:8443/"', PROJECT_RC,
         "must be unset"),
        ("proxy", "http://proxy.corp:3128/", '"http://proxy.corp:3128/"', USER_RC,
         "must be unset"),
        ("strict-ssl", False, "false", ENV_RC, "must stay true"),
        ("cafile", "/etc/ssl/mitm.pem", '"/etc/ssl/mitm.pem"', USER_RC, "must be unset"),
        ("ca", ["-----BEGIN CERTIFICATE-----"], '["-----BEGIN CERTIFICATE-----"]', PROJECT_RC,
         "must be unset"),
        ("registry", "http://127.0.0.1:4873/", '"http://127.0.0.1:4873/"', ENV_RC,
         "must be the single registry"),
        ("@beastlabai:registry", "https://npm.pkg.github.com/", '"https://npm.pkg.github.com/"',
         USER_RC, "must be the single registry"),
    ], ids=["https-proxy", "proxy", "strict-ssl", "cafile", "ca", "registry", "scoped-registry"])
    def test_a_setting_that_can_steer_the_upload_refuses_and_names_its_layer(
        self, release, world, key, value, display, layer, why,
    ):
        world.config[key] = value
        world.config_text = _config_text({layer: {key: display}})
        assert release.run(dry_run=False) == first_release.EXIT_REFUSED
        _refused(release, "npm-config")
        message = next(e for e in release.errors if e.startswith("REFUSING (npm-config)"))
        assert f"{key} = {json.dumps(value)}" in message and why in message
        assert f"set in the {layer}" in message
        assert not world.published

    def test_a_layer_npm_did_not_name_is_still_a_refusal(self, release, world):
        world.config["https-proxy"] = "http://127.0.0.1:8443/"
        assert release.run(dry_run=False) == 1
        message = next(e for e in release.errors if e.startswith("REFUSING (npm-config)"))
        assert "from a layer `npm config list` did not name" in message

    @pytest.mark.parametrize("value", [SINGLE, SINGLE + "/"])
    def test_a_scoped_registry_naming_the_single_one_passes(self, release, world, gated, value):
        world.config["@beastlabai:registry"] = value
        world.views = _good_views(gated)
        assert release.run(dry_run=False) == 0, release.errors

    @pytest.mark.parametrize("stdout", ["", "npm error code E401", "[]"])
    def test_an_unreadable_listing_refuses(self, release, world, stdout, monkeypatch):
        world.config = None
        honest = world._answer

        def answer(argv, capture):
            if argv[:3] == ["npm", "config", "list"] and "--json" in argv:
                return (0, stdout)
            return honest(argv, capture)

        monkeypatch.setattr(world, "_answer", answer)
        assert release.run(dry_run=False) == 1
        _refused(release, "npm-config")

    def test_the_confirming_read_runs_on_none_of_the_callers_configuration(
        self, release, world, gated,
    ):
        world.views = [(1, _npm_error("E404"))] + [_good_views(gated)[1]]
        assert release.run(dry_run=False) == 0
        reads = world.npm_calls("view", SPEC)
        assert len(reads) == 2
        for read in reads:
            argv, fresh = read["argv"], str(read["cwd"])
            assert read["inherit_npm_config"] is False
            assert f"--prefix={fresh}" in argv
            assert f"--userconfig={Path(fresh) / 'npmrc-user'}" in argv
            assert f"--globalconfig={Path(fresh) / 'npmrc-global'}" in argv
            assert argv[argv.index("--cache") + 1] == str(Path(fresh) / "cache")
            assert not Path(fresh).exists()
        assert reads[0]["cwd"] != reads[1]["cwd"]
        (path,) = list(release.evidence_dir.glob("*.json"))
        recorded = json.loads(path.read_text(encoding="utf-8"))["confirmation"]["argv"]
        assert "--prefix=<fresh tmp>" in recorded and str(reads[-1]["cwd"]) not in " ".join(recorded)

    def test_the_runner_can_drop_every_inherited_npm_setting(self, monkeypatch, tmp_path):
        monkeypatch.setenv("npm_config_https_proxy", "http://127.0.0.1:9/")
        monkeypatch.setenv("NPM_CONFIG_CAFILE", "/x.pem")
        probe = [sys.executable, "-c",
                 "import json, os; print(json.dumps([sorted(os.environ), os.getcwd()]))"]
        kept, cwd = json.loads(first_release.Runner()(probe, cwd=tmp_path).stdout)
        # Windows folds environment names to upper case; POSIX keeps both spellings.
        fold = str.upper if os.name == "nt" else str
        assert {fold("npm_config_https_proxy"), fold("NPM_CONFIG_CAFILE")} <= {fold(k) for k in kept}
        assert os.path.realpath(cwd) == os.path.realpath(tmp_path)
        dropped, _ = json.loads(first_release.Runner()(probe, inherit_npm_config=False).stdout)
        assert not [k for k in dropped if k.lower().startswith("npm_config_")]


needs_npm = pytest.mark.skipif(shutil.which("npm") is None or os.name == "nt",
                               reason="needs the real npm (POSIX) to read its own layers")


@needs_npm
class TestNpmConfigAgainstTheRealNpm:
    """The preflight on the real npm's own answer, offline (``npm config list``
    makes no request). The home holds a token and maybe a proxy; above the
    scratch directory (``TMPDIR``) sits a planted project whose ``.npmrc``
    names a different proxy and turns ``strict-ssl`` off. The pinned project
    root must keep that file out, and the user layer's proxy must be refused
    with its file named."""

    PLANTED = "https-proxy=http://127.0.0.1:9/\nstrict-ssl=false\n"
    TOKEN = "//registry.npmjs.org/:_authToken=npm_notARealTokenForTheSuite\n"

    def _release(self, tmp_path, monkeypatch, home_rc):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".npmrc").write_text(home_rc, encoding="utf-8")
        project = tmp_path / "project"
        (project / "tmp").mkdir(parents=True)
        (project / "package.json").write_text("{}", encoding="utf-8")
        (project / ".npmrc").write_text(self.PLANTED, encoding="utf-8")
        monkeypatch.setattr(tempfile, "tempdir", str(project / "tmp"))
        monkeypatch.setenv("HOME", str(home))
        for key in list(os.environ):
            if key.lower().startswith("npm_config_"):
                monkeypatch.delenv(key)
        errors: list = []
        release = first_release.Release(out=lambda line: None, err=errors.append)
        release.facts["registry"] = SINGLE
        return release, home

    def test_a_proxy_in_the_users_npmrc_is_refused_and_its_file_named(self, tmp_path, monkeypatch):
        release, home = self._release(tmp_path, monkeypatch,
                                      self.TOKEN + "https-proxy=http://127.0.0.1:8/\n")
        try:
            with pytest.raises(first_release.Refusal) as caught:
                release.check_npm_config()
        finally:
            release.close()
        message = caught.value.message
        assert caught.value.condition == "npm-config"
        assert 'https-proxy = "http://127.0.0.1:8/" (must be unset)' in message
        assert f'set in the "user" config from {home / ".npmrc"}' in message
        assert "127.0.0.1:9" not in message and "strict-ssl" not in message, (
            "the planted project file above the scratch directory was read")
        assert "notARealToken" not in message

    def test_a_clean_configuration_passes_although_a_project_above_plants_one(
        self, tmp_path, monkeypatch,
    ):
        release, _ = self._release(tmp_path, monkeypatch, self.TOKEN)
        try:
            # Control: from the same scratch directory, without the pin, npm
            # walks up to the planted project and reads its .npmrc.
            scratch = release.npm_scratch()
            control = subprocess.run(["npm", "config", "get", "https-proxy"], cwd=scratch,
                                     capture_output=True, text=True, timeout=120)
            assert control.stdout.strip() == "http://127.0.0.1:9/", control.stdout + control.stderr
            release.check_npm_config()
        finally:
            release.close()
        assert release.facts["npm_config"]["https-proxy"] is None
        assert release.facts["npm_config"]["strict-ssl"] is True
        assert registry.normalize(release.facts["npm_config"]["registry"]) == SINGLE


# ---------------------------------------------------------------------------
# After the upload
# ---------------------------------------------------------------------------

class TestAfterTheUpload:

    def test_a_different_integrity_is_a_mismatch_after_publication(self, release, world, gated):
        world.views = [(0, json.dumps("sha512-" + "Z" * 86 + "==") + "\n")]
        assert release.run(dry_run=False) == first_release.EXIT_MISMATCH
        message = release.errors[-1]
        assert message.startswith("MISMATCH AFTER PUBLICATION") and "Do NOT push" in message
        assert "Recovery" in message
        (path,) = list(release.evidence_dir.glob("*.json"))
        assert json.loads(path.read_text(encoding="utf-8"))["confirmation"]["state"] == "mismatch"

    def test_an_expired_poll_is_not_a_mismatch(self, release, world, gated):
        world.views = [(1, _npm_error("E404"))]
        assert release.run(dry_run=False) == first_release.EXIT_UNCONFIRMED
        message = release.errors[-1]
        assert message.startswith("NOT YET CONFIRMED") and "Repeat the read by hand" in message
        assert f'd="$(mktemp -d)" && cd "$d" && npm view {SPEC} dist.integrity ' in message
        assert f'--registry={SINGLE} --cache "$d/c" --prefix "$d"' in message
        assert gated["facts"]["integrity"] in message
        # Bounded: 60 s at a 10 s interval.
        assert len(world.argvs(["npm", "view", SPEC])) == 7
        (path,) = list(release.evidence_dir.glob("*.json"))
        assert json.loads(path.read_text(encoding="utf-8"))["confirmation"]["state"] == "expired"

    def test_a_network_error_while_polling_keeps_polling(self, release, world, gated):
        world.views = [(1, _npm_error("ECONNRESET")), (1, _npm_error("E404")),
                       _good_views(gated)[1]]
        assert release.run(dry_run=False) == 0

    def test_a_failed_upload_is_reported_with_its_evidence_and_never_polled(
        self, release, world,
    ):
        world.publish = 1
        assert release.run(dry_run=False) == first_release.EXIT_PUBLISH_FAILED
        assert "inspect the registry before retrying" in release.errors[-1]
        assert world.argvs(["npm", "view", SPEC]) == []
        (path,) = list(release.evidence_dir.glob("*.json"))
        evidence = json.loads(path.read_text(encoding="utf-8"))
        assert evidence["publish"]["exit"] == 1 and evidence["confirmation"] is None


# ---------------------------------------------------------------------------
# The Makefile
# ---------------------------------------------------------------------------

def _recipe(target: str) -> list:
    lines, inside = [], False
    for line in (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if inside and line.startswith("\t"):
            lines.append(line.strip())
            continue
        inside = re.match(rf"^{re.escape(target)}\s*:(?!=)", line) is not None
    return lines


class TestMakefile:

    def test_opencode_publish_no_longer_uploads(self):
        body = " ".join(_recipe("opencode-publish"))
        assert body, "opencode-publish has no recipe"
        assert not re.search(r"\bnpm\s+publish\b", body) and "--provenance" not in body
        assert "opencode-publish-first" in body and "workflow" in body
        assert body.rstrip().endswith("exit 1")

    @pytest.mark.skipif(shutil.which("make") is None or os.name == "nt",
                        reason="needs make and a POSIX shell for the recipe")
    def test_opencode_publish_refuses_and_names_both_paths(self):
        done = subprocess.run(["make", "-s", "opencode-publish"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode != 0
        assert "make opencode-publish-first" in done.stderr
        assert "workflow dispatch" in done.stderr

    def test_the_first_release_target_runs_the_script_and_never_a_dirty_build(self):
        body = _recipe("opencode-publish-first")
        assert any("first_release.py" in line for line in body)
        assert any("--dry-run" in line and "DRY_RUN" in line for line in body)
        assert not any("--allow-dirty" in line or "ALLOW_DIRTY" in line for line in body)

    def test_the_old_comment_is_gone(self):
        text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        assert "publishes the DIRECTORY, never a pre-built .tgz" not in text
