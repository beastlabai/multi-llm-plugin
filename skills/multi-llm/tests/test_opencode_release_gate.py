"""The opencode release gate (``packaging/opencode/release_gate.py``), offline.

The gate is the one executable contract between ``make opencode-package-check``,
``build.py --write-stamp`` and the publishing workflow's ``verify`` job, so these
rows drive it the way those three do: the static rows graded out of junit XML,
rows 7 and 9 checked from the files the probes write, row 8 graded from run
records and bundles shaped the way the container harness writes them, the image
bound through a fake ``docker`` that never starts a container, the stamp written
and checked through ``build.py``'s own functions, and the release-gate record
exported and then validated **as a CLI**, with no site-packages at all — the
workflow runs it under ``uv run --frozen --no-dev``, where ``jsonschema`` is
absent. Nothing here needs Docker, npm, a network or an ``opencode`` binary.

:class:`GateRun` is one complete passing session built through the gate's own
writers, in a temporary tree; ``test_opencode_package.py``'s stamp rows use it
too. Each refusal row starts from it and breaks exactly one thing.
"""

from __future__ import annotations

import ast
import copy
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
GATE_PATH = PACKAGE_DIR / "release_gate.py"
MAKEFILE = REPO_ROOT / "Makefile"

for _path in (str(SKILL_ROOT), str(REPO_ROOT / "packaging")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_gate():
    """Under the name ``build.py`` loads it by, so both hold one module."""
    name = "opencode_release_gate"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, GATE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


gate = _load_gate()
#: opencode's build.py — through the gate, which reuses an already-imported
#: ``build`` only when it is this file (the codex route ships one too).
pkg_build = gate._builder()

IMAGE_ID = "sha256:" + "ab" * 32
OTHER_IMAGE_ID = "sha256:" + "cd" * 32
TAG = "mllm-e2e:opencode-gate"
CANCEL = gate.ROW8_ABORT_SCENARIO


# ---------------------------------------------------------------------------
# Builders: every file shaped the way its real writer shapes it
# ---------------------------------------------------------------------------

def make_tarball(package: Path, out: Path) -> Path:
    """``package`` packed the way ``npm pack`` packs it: every shipped file
    (``dist/tree.sha256`` included) under ``package/``, regular files only."""
    with tarfile.open(out, "w:gz") as archive:
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package).as_posix()
            if pkg_build._not_shipped(relative) and relative != "dist/tree.sha256":
                continue
            archive.add(path, arcname="package/" + relative, recursive=False)
    return out


def junit_xml(path: Path, *, outcomes=None, drop=(), extra=(), prefix="") -> Path:
    """A junit file holding every declared CI node, ``passed`` unless told
    otherwise — written in pytest's own spelling (classname + name)."""
    outcomes = outcomes or {}
    bodies = {"passed": "", "skipped": '<skipped type="pytest.skip" message="s">s</skipped>',
              "failed": '<failure message="f">f</failure>',
              "error": '<error message="teardown">e</error>'}
    cases = []
    nodes = [node for row in gate.ROWS for node in row.nodes if node not in drop]
    for node in [*nodes, *extra]:
        classname, name = gate._junit_key(node)
        for outcome in outcomes.get(node, "passed").split("+"):
            cases.append(f'<testcase classname="{prefix}{classname}" name="{name}" '
                         f'time="0.01">{bodies[outcome]}</testcase>')
    path.write_text('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
                    f'<testsuite name="pytest" tests="{len(cases)}">{"".join(cases)}'
                    "</testsuite></testsuites>", encoding="utf-8")
    return path


def installed_report(host: str) -> dict:
    """The shape of the image's ``/opt/mllm/lock/cli-versions.installed.json``."""
    return {"base_image": {"ref": "docker.io/library/node:22-bookworm-slim"},
            "clis": [{"id": "codex", "installed": "0.163.0", "status": "ok"},
                     {"id": "opencode", "binary": "opencode", "installed": host,
                      "expected": host, "status": "ok", "enforced": True}],
            "mode": "pinned"}


def fake_docker(images: dict, contents: dict, calls: list | None = None):
    """A ``docker`` that answers ``image inspect``, ``create``, ``cp`` and
    ``rm`` and refuses anything else — ``run``/``start`` included, which is
    how the rows prove the image is read without being started."""
    calls = calls if calls is not None else []
    created: dict = {}

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        assert argv[0] == "docker", argv
        verb = list(argv[1:])

        def done(code=0, out="", err=""):
            return subprocess.CompletedProcess(argv, code, out, err)

        if verb[:2] == ["image", "inspect"]:
            ref = verb[-1]
            if ref in images:
                return done(out=images[ref] + "\n")
            if ref in images.values():
                return done(out=ref + "\n")
            return done(1, err=f"Error: No such image: {ref}")
        if verb[0] == "create":
            container = f"c{len(created)}"
            created[container] = verb[1]
            return done(out=container + "\n")
        if verb[0] == "cp":
            container, path = verb[1].split(":", 1)
            data = contents.get(created[container], {}).get(path)
            if data is None:
                return done(1, err=f"Error: Could not find the file {path}")
            Path(verb[2]).write_bytes(data)
            return done()
        if verb[:2] == ["rm", "-f"]:
            return done(out=verb[2] + "\n")
        raise AssertionError(f"the gate ran `docker {' '.join(verb)}`")
    return runner


def _project(kind: str) -> dict:
    return {"kind": kind, "command_registered": True,
            "command_skill_dir": "/tmp/probe/extracted/package/skills/multi-llm",
            "command_inside_payload": True, "tool_registered": True,
            "init": {"ok": True, "exit": 0, "result": {"mode": "init", "state": "done"},
                     "files_created": [".multi-llm/config.yaml"]},
            # What status answers in a project whose only run was init, measured
            # on the real probe: exit 2, "no run recorded".
            "status": {"ok": True, "exit": 2,
                       "result": {"mode": "status", "state": "nothing-to-do"}}}


def row7_raw(facts: dict, *, host: str, registry: str) -> dict:
    """What ``probe.py row7 --out`` writes (impl contract, evidence shapes)."""
    return {"row": "7", "pass": True, "opencode_version": host, "registry": registry,
            "tarball": {"path": facts["path"], "integrity": facts["integrity"],
                        "digest": facts["digest"]},
            "probe_root": "/tmp/probe", "isolation": {"env_cleared": ["OPENCODE_CONFIG"],
                                                      "home": "/tmp/probe/home"},
            "projects": [_project("committed"), _project("no-commit")], "failures": []}


def row9_raw(*, registry: str, **overrides) -> dict:
    """What ``registry_absent.mjs --out`` writes for an absent version."""
    data = gate.manifest()
    raw = {"row": "9", "pass": True, "registry": registry, "name": data["name"],
           "version": data["version"], "npm_exit": 1, "code": "E404", "outcome": "absent"}
    raw.update(overrides)
    return raw


def _later(stamp: str, seconds: int) -> str:
    moment = datetime.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
    return (moment + datetime.timedelta(seconds=seconds)).strftime("%Y%m%dT%H%M%SZ")


def write_row8(root: Path, *, stamp: str, host: str, image_id: str = IMAGE_ID,
               skip=(), **changes) -> Path:
    """One record and bundle per scenario, as ``harness.emit_record`` and the
    scenario graders leave them. ``changes[<row id>]`` breaks one scenario:
    ``stamp``, ``outcome``, ``entry`` (merged into the row entry), ``image``,
    ``grader`` (merged into gate-grader.json), ``no_grader``, ``no_inside``,
    ``mismatches``, ``tamper``, ``no_bundle``, ``prune``."""
    from tests.container.verify_record import hash_bundle

    records = root / "records"
    records.mkdir(exist_ok=True)
    for row_id in gate.ROW8_SCENARIOS:
        if row_id in skip:
            continue
        change = changes.get(row_id, {})
        run = change.get("stamp", stamp)
        bundle = root / "runs" / run / row_id
        bundle.mkdir(parents=True, exist_ok=True)
        grader = {"scenario": row_id.split("gate-", 1)[1], "row": row_id,
                  "grader": "grade_" + row_id.split("gate-", 1)[1].replace("-", "_"),
                  "pass": True, "route": "package", "installed_root": gate.INSTALLED_ROOT,
                  "dispatcher_skill_dir": gate.INSTALLED_ROOT + "/skills/multi-llm",
                  "checks": {"tool_called": True, "result_terminal": True}}
        if row_id == CANCEL:
            grader["abort_method"] = "client.session.abort"
        grader.update(change.get("grader", {}))
        if not change.get("no_grader"):
            (bundle / "gate-grader.json").write_text(json.dumps(grader), "utf-8")
        if not change.get("no_inside"):
            (bundle / "posture-inside.json").write_text('{"uid": 10001}', "utf-8")
        (bundle / "posture-host.json").write_text(json.dumps(
            {"captured": True, "mismatches": change.get("mismatches", [])}), "utf-8")
        files, aggregate = hash_bundle(bundle)
        entry = {"row_id": row_id, "outcome": "pass", "route": "package",
                 "cli_version": host, **change.get("entry", {})}
        record = {"schema_version": 1, "started_at": run, "lane": "host",
                  "outcome": change.get("outcome", "pass"), "rows": [entry],
                  "image": {"tag": image_id, "digest": change.get("image", image_id)}}
        if not change.get("no_bundle"):
            record["bundle"] = {"root": f"runs/{run}/{row_id}", "files": files,
                                "aggregate": aggregate}
        (records / f"{run}-{row_id}.json").write_text(json.dumps(record), "utf-8")
        if change.get("tamper"):
            (bundle / "rewritten.txt").write_text("after the digest", "utf-8")
        if change.get("prune"):
            shutil.rmtree(bundle)
    return records


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One staged package for the module; every :class:`GateRun` copies it."""
    destination = tmp_path_factory.mktemp("opencode-gate") / "pkg"
    pkg_build.build(destination=destination, allow_dirty=True)
    return destination


class GateRun:
    """A complete passing gate session in ``tmp_path``.

    Every row is written through the gate's own graders, in the recipe's
    order — :func:`grade_ci` over junit, ``record 7``, ``image``,
    :func:`check_row8` over records and bundles, ``record 9`` — so the
    evidence on disk is exactly what a green ``make opencode-package-check``
    leaves. The whole-tree dirtiness check is stubbed clean (this checkout is
    under development); everything else is real: HEAD, the pins, the registry.
    """

    def __init__(self, tmp_path: Path, monkeypatch, built: Path | None = None):
        monkeypatch.setattr(gate, "dirty_paths", lambda root=gate.REPO_ROOT: [])
        self.root = tmp_path
        self.dir = tmp_path / "gate"
        self.package = tmp_path / "pkg"
        if built is not None:
            shutil.copytree(built, self.package)
        else:
            pkg_build.build(destination=self.package, allow_dirty=True)
        self.digest = (self.package / "dist" / "tree.sha256").read_text("utf-8").strip()
        self.tarball = make_tarball(self.package, tmp_path / "opencode-multi-llm-gated.tgz")
        self.facts = pkg_build.tarball_facts(self.tarball)
        self.host = gate.host_pin()
        self.registry = gate.single_registry()
        self.calls: list = []
        self.contents = {IMAGE_ID: {
            gate.IMAGE_PAYLOAD_DIGEST: (self.digest + "\n").encode(),
            gate.IMAGE_INSTALLED_REPORT: json.dumps(installed_report(self.host)).encode()}}
        self.session: dict = {}

    def begin(self, **kwargs):
        self.session = gate.begin_session(self.dir, **kwargs)
        return self.session

    def ci_rows(self, **junit):
        verdicts = gate.grade_ci([junit_xml(self.root / "ci.xml", **junit)])
        for row_id, verdict in verdicts.items():
            gate.write_row(row_id, verdict, self.dir)
        return verdicts

    def row7(self, **overrides):
        raw = row7_raw(self.facts, host=self.host, registry=self.registry)
        raw.update(overrides)
        source = self.root / "row7.json"
        source.write_text(json.dumps(raw), "utf-8")
        return gate.record_row("7", source, self.dir, tarball=self.tarball)

    def image(self, *, environ=None, contents=None, images=None):
        return gate.bind_image(
            TAG, runner=fake_docker(images or {TAG: IMAGE_ID}, contents or self.contents,
                                    self.calls),
            environ=environ or {}, directory=self.dir, digest=self.digest,
            host_version=self.host)

    def check8(self, since=None, **kwargs):
        return gate.check_row8(since or self.session["started_stamp"], directory=self.dir,
                               records_dir=self.root / "records", container_dir=self.root,
                               posture_head=kwargs.pop("posture_head", lambda: ("0" * 64, True)),
                               **kwargs)

    def row8(self, **changes):
        self.run_stamp = gate._compact(gate._now())
        write_row8(self.root, stamp=self.run_stamp, host=self.host, **changes)
        evidence = self.check8()
        gate.write_row("8", evidence, self.dir)
        return evidence

    def row9(self, **overrides):
        source = self.root / "row9.json"
        source.write_text(json.dumps(row9_raw(**{"registry": self.registry, **overrides})),
                          "utf-8")
        return gate.record_row("9", source, self.dir)

    def full(self, **begin):
        self.begin(**begin)
        self.ci_rows()
        self.row7()
        self.image()
        self.row8()
        self.row9()
        return self

    def problems(self, **override):
        args = dict(digest=self.digest, version=gate.package_version(),
                    head=gate.head_sha(), host_version=self.host, registry=self.registry,
                    tarball=pkg_build.tarball_facts(self.tarball), dirty=())
        args.update(override)
        return gate.evidence_problems(gate.load_evidence(self.dir), **args)

    def stamp(self):
        return pkg_build.write_stamp(self.package, timestamp="2026-10-25T12:00:00Z",
                                     allow_dirty=True, evidence_dir=self.dir,
                                     tarball=self.tarball)

    def check(self):
        return pkg_build.check_stamp(self.package, allow_dirty=True)

    def export(self, records=None, **kwargs):
        return gate.export_record(directory=self.dir, records_dir=records or self.root / "out",
                                  stamp_path=self.package / pkg_build.STAMP_NAME,
                                  tarball=self.tarball, **kwargs)

    def rewrite_row(self, row_id, **fields):
        path = self.dir / f"row-{row_id}.json"
        data = json.loads(path.read_text("utf-8"))
        data.update(fields)
        path.write_text(json.dumps(data), "utf-8")


@pytest.fixture
def run(tmp_path, monkeypatch, built):
    return GateRun(tmp_path, monkeypatch, built)


def _joined(problems) -> str:
    return "\n".join(problems)


# ===========================================================================
# The rows: exactly the plan's table, each static row naming real tests
# ===========================================================================

class TestTheRows:
    def test_the_row_ids_are_the_plans(self):
        assert gate.ROW_IDS == ("1", "2", "3", "4", "5", "5b", "6", "7", "8", "9")
        assert "8b" not in gate.ROW_IDS, "8b is CI's three test legs, not a stamp row"
        assert gate.CI_ROWS == ("1", "2", "3", "4", "5", "5b", "6")
        for row in gate.ROWS:
            assert row.proves and row.grader and row.evidence, row.id
        assert gate.ROW8_SCENARIOS == (
            "opencode-package-gate-init-status", "opencode-package-gate-detach",
            "opencode-package-gate-supervision", "opencode-package-gate-completion",
            "opencode-package-gate-cancellation")

    def test_every_declared_node_is_a_real_test(self):
        """A renamed test would otherwise fail its row as ``absent`` forever —
        loudly, but only at gate time. This catches it in the ordinary suite."""
        defined = {}
        for node in (n for row in gate.ROWS for n in row.nodes):
            path, *parts = node.split("::")
            if path not in defined:
                tree = ast.parse((SKILL_ROOT / path).read_text(encoding="utf-8"))
                defined[path] = {
                    f"{cls.name}::{fn.name}" for cls in tree.body
                    if isinstance(cls, ast.ClassDef) for fn in cls.body
                    if isinstance(fn, ast.FunctionDef)}
            assert "::".join(parts) in defined[path], f"{node} does not exist"

    def test_each_node_is_selected_by_the_ci_tier(self):
        """Every declared node is collected by one of the two selections
        ``opencode-package-check-ci`` runs — never deselected by the recipe's own
        marker expressions, which is the empty-selection hole the grader exists
        for."""
        def collect(marks):
            done = subprocess.run(
                [sys.executable, "-m", "pytest", "skills/multi-llm/tests/test_opencode_package.py",
                 "--collect-only", "-q", "-p", "no:cacheprovider", "-m", marks],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
            assert done.returncode == 0, done.stdout + done.stderr
            return {line.strip() for line in done.stdout.splitlines() if "::" in line}

        selected = collect("package and not container") | collect(
            "not live and not container and not package")
        missing = [node for row in gate.ROWS for node in row.nodes if node not in selected]
        assert not missing, f"the CI tier's selections deselect {missing}"


# ===========================================================================
# Rows 1-6 and 5b: graded node by node, never by exit code
# ===========================================================================

class TestGradeCi:
    def test_every_node_passed_passes_every_row(self, tmp_path):
        verdicts = gate.grade_ci([junit_xml(tmp_path / "j.xml")])
        assert set(verdicts) == set(gate.CI_ROWS)
        assert all(v["pass"] for v in verdicts.values())

    @pytest.mark.parametrize("outcome, word", [("skipped", "skipped"), ("failed", "failed"),
                                               ("passed+error", "failed")])
    def test_a_skipped_or_failed_node_fails_its_row(self, tmp_path, outcome, word):
        node = gate._BY_ID["5b"].nodes[1]
        verdicts = gate.grade_ci([junit_xml(tmp_path / "j.xml", outcomes={node: outcome})])
        assert not verdicts["5b"]["pass"]
        assert verdicts["5b"]["nodes"][node] == word
        assert all(verdicts[r]["pass"] for r in gate.CI_ROWS if r != "5b")
        assert f"row 5b: {node} {word}" in _joined(gate._ci_row_problems("5b", verdicts["5b"]))

    def test_a_deselected_node_fails_its_row_as_absent(self, tmp_path):
        node = gate._BY_ID["1"].nodes[0]
        verdicts = gate.grade_ci([junit_xml(tmp_path / "j.xml", drop=(node,))])
        assert verdicts["1"]["nodes"][node] == "absent" and not verdicts["1"]["pass"]
        assert "deselected" in _joined(gate._ci_row_problems("1", verdicts["1"]))

    def test_an_empty_selection_fails_every_row(self, tmp_path):
        nodes = tuple(n for row in gate.ROWS for n in row.nodes)
        verdicts = gate.grade_ci([junit_xml(tmp_path / "j.xml", drop=nodes)])
        assert not any(v["pass"] for v in verdicts.values())
        assert "empty selection" in _joined(gate._ci_row_problems("2", verdicts["2"]))

    def test_a_parametrized_node_needs_every_case(self, tmp_path):
        node = gate._BY_ID["2"].nodes[0]
        path = junit_xml(tmp_path / "j.xml", drop=(node,),
                         extra=(node + "[a]", node + "[b]"),
                         outcomes={node + "[b]": "skipped"})
        assert gate.grade_ci([path])["2"]["nodes"][node] == "skipped"

    def test_a_repo_rooted_classname_still_matches(self, tmp_path):
        """pytest prefixes the classname with the path from its rootdir; run from
        the repository root it would read ``skills.multi-llm.tests...``."""
        path = junit_xml(tmp_path / "j.xml", prefix="skills.multi-llm.")
        assert all(v["pass"] for v in gate.grade_ci([path]).values())

    def test_a_missing_or_torn_junit_file_refuses(self, tmp_path):
        torn = tmp_path / "torn.xml"
        torn.write_text("<testsuites><testsuite>", "utf-8")
        for path in (tmp_path / "absent.xml", torn):
            with pytest.raises(gate.GateError, match="not a readable junit file"):
                gate.grade_ci([path])

    def test_the_cli_grades_only_outside_a_session(self, tmp_path):
        evidence = tmp_path / "gate"
        node = gate._BY_ID["4"].nodes[0]
        bad = junit_xml(tmp_path / "bad.xml", outcomes={node: "skipped"})
        done = subprocess.run([sys.executable, str(GATE_PATH), "--evidence-dir", str(evidence),
                               "grade-ci", "--junit", str(bad)],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 1
        assert "row(s) 4 did not pass" in done.stderr and "skipped" in done.stderr
        assert "no gate session for HEAD; graded only" in done.stdout
        assert not list(evidence.glob("row-*.json"))

    def test_the_cli_writes_each_row_inside_a_session_for_head(self, run, tmp_path):
        run.begin()
        done = subprocess.run([sys.executable, str(GATE_PATH), "--evidence-dir", str(run.dir),
                               "grade-ci", "--junit", str(junit_xml(tmp_path / "ok.xml"))],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        for row_id in gate.CI_ROWS:
            row = json.loads((run.dir / f"row-{row_id}.json").read_text("utf-8"))
            assert row["pass"] is True and row["session"] == run.session["id"]

    def test_a_session_for_another_commit_is_not_written_into(self, run, monkeypatch):
        run.begin()
        monkeypatch.setattr(gate, "head_sha", lambda root=None: "f" * 40)
        assert gate.session_for_grading(run.dir) is None


# ===========================================================================
# The session
# ===========================================================================

class TestSession:
    def test_begin_writes_the_contract_fields_and_clears_everything(self, run):
        run.full()
        for name in ("raw", "logs", "junit"):
            (run.dir / name / "old").write_text("x", "utf-8")
        first = run.session["id"]
        session = run.begin()
        assert session["id"] != first
        assert {"id", "started_at", "candidate_sha", "version", "dirty"} <= set(session)
        assert session["candidate_sha"] == gate.head_sha()
        assert session["version"] == gate.package_version() and session["dirty"] is False
        assert sorted(p.name for p in run.dir.glob("*.json")) == ["session.json"]
        for name in ("raw", "logs", "junit"):
            assert not list((run.dir / name).iterdir()), name

    def test_stale_evidence_after_a_reset_authorizes_nothing(self, run):
        run.full()
        assert run.problems() == []
        run.begin()
        problems = _joined(run.problems())
        for row_id in gate.ROW_IDS:
            assert f"row {row_id}: no evidence in this session" in problems
        assert "no image-build entry in this session" in problems

    def test_evidence_from_another_session_is_refused(self, run):
        run.full()
        run.rewrite_row("7", session="0" * 32)
        assert "row 7: its evidence belongs to another gate session" in _joined(run.problems())

    def test_a_commit_mid_gate_is_refused(self, run):
        run.full()
        problems = _joined(run.problems(head="e" * 40))
        assert "a commit landed mid-gate" in problems

    @pytest.mark.parametrize("kwargs, why", [({"allow_dirty": True}, "ALLOW_DIRTY=1"),
                                             ({}, "a dirty tree at `begin`")])
    def test_a_rehearsal_is_named(self, run, monkeypatch, kwargs, why):
        if not kwargs:
            monkeypatch.setattr(gate, "dirty_paths", lambda root=None: [" M Makefile"])
        run.begin(**kwargs)
        problems = gate.session_problems(gate.current_session(run.dir))
        assert len(problems) == 1 and "REHEARSAL" in problems[0] and why in problems[0]

    def test_no_session_is_named(self, tmp_path):
        problems = gate.session_problems(gate.current_session(tmp_path / "none"))
        assert problems and problems[0].startswith("no gate session")


# ===========================================================================
# Row 7: the probe's file, re-read
# ===========================================================================

class TestRow7:
    def test_a_green_probe_is_recorded_against_the_gated_tarball(self, run):
        run.begin()
        path = run.row7()
        row = json.loads(path.read_text("utf-8"))
        assert row["pass"] is True and row["session"] == run.session["id"]
        assert row["tarball"]["integrity"] == run.facts["integrity"]

    @pytest.mark.parametrize("change, expected", [
        ({"pass": False, "failures": ["tool multi_llm not registered"]},
         "the probe did not pass"),
        ({"opencode_version": "1.19.3"}, "ran opencode '1.19.3'"),
        ({"registry": "http://127.0.0.1:4873"}, "the single registry is"),
        ({"failures": ["x"]}, "pass is true beside failures"),
        ({"projects": [_project("committed")]}, "exactly a committed and a no-commit"),
        ({"isolation": {}}, "no isolation block"),
    ])
    def test_a_probe_that_did_not_prove_row_7_is_refused(self, run, change, expected):
        run.begin()
        with pytest.raises(gate.GateError, match=re.escape(expected)):
            run.row7(**change)
        assert not (run.dir / "row-7.json").exists()

    def test_a_project_whose_init_failed_is_refused(self, run):
        run.begin()
        broken = _project("no-commit")
        broken["init"] = {"ok": False, "exit": 1, "result": {"error": "GitError"},
                          "files_created": []}
        with pytest.raises(gate.GateError,
                           match="no-commit project: init did not return its documented result"):
            run.row7(projects=[_project("committed"), broken])

    def test_a_status_that_found_a_finished_run_is_documented_too(self, run):
        run.begin()
        done = _project("committed")
        done["status"] = {"ok": True, "exit": 0, "result": {"mode": "status", "state": "done"}}
        run.row7(projects=[done, _project("no-commit")])
        assert (run.dir / "row-7.json").exists()

    @pytest.mark.parametrize("status", [
        {"ok": True, "exit": 1, "result": {"mode": "status", "state": "error"}},
        {"ok": True, "exit": 3, "result": {"mode": "status", "state": "needs-input"}},
        {"ok": True, "exit": 2, "result": {"mode": "review-plan", "state": "nothing-to-do"}},
        {"ok": True, "exit": 0, "result": {"mode": "status"}},
    ], ids=["exit-1", "exit-3", "another-mode", "no-state"])
    def test_a_status_outside_its_documented_results_is_refused(self, run, status):
        run.begin()
        broken = _project("no-commit")
        broken["status"] = status
        with pytest.raises(gate.GateError,
                           match="no-commit project: status did not return its documented result"):
            run.row7(projects=[_project("committed"), broken])

    def test_a_skill_dir_outside_the_probe_root_is_refused(self, run):
        run.begin()
        elsewhere = _project("committed")
        elsewhere["command_skill_dir"] = "/home/someone/checkout/skills/multi-llm"
        with pytest.raises(gate.GateError, match="outside the probe's own root"):
            run.row7(projects=[elsewhere, _project("no-commit")])

    def test_another_tarball_is_refused(self, run, tmp_path):
        run.begin()
        other = dict(run.facts, integrity="sha512-AAAA", digest="0" * 64,
                     path=str(tmp_path / "elsewhere.tgz"))
        with pytest.raises(gate.GateError) as caught:
            run.row7(tarball={k: other[k] for k in ("path", "integrity", "digest")})
        for words in ("the probe loaded", "tarball integrity", "tarball digest"):
            assert words in str(caught.value)

    def test_a_probe_failure_prevents_the_stamp(self, run):
        run.begin()
        run.ci_rows()
        with pytest.raises(gate.GateError):
            run.row7(**{"pass": False, "failures": ["init exited 1"]})
        run.image()
        run.row8()
        run.row9()
        with pytest.raises(pkg_build.PayloadError, match="row 7: no evidence in this session"):
            run.stamp()
        assert not (run.package / pkg_build.STAMP_NAME).exists()


# ===========================================================================
# Row 9: E404 and nothing else
# ===========================================================================

class TestRow9:
    def test_absent_is_recorded(self, run):
        run.begin()
        assert json.loads(run.row9().read_text("utf-8"))["code"] == "E404"

    @pytest.mark.parametrize("change, expected", [
        ({"outcome": "published", "pass": False, "npm_exit": 0, "code": None},
         "already on"),
        ({"outcome": "unknown", "pass": False, "code": "E429"}, "UNKNOWN"),
        ({"outcome": "unknown", "pass": True, "code": "ENEEDAUTH"}, "disagrees with outcome"),
        ({"code": "E401"}, "only E404 is absence"),
        ({"registry": "https://registry.example.com"}, "another server's E404"),
        ({"version": "9.9.9"}, "the candidate is"),
    ])
    def test_anything_but_e404_from_the_single_registry_refuses(self, run, change, expected):
        run.begin()
        with pytest.raises(gate.GateError, match=re.escape(expected)):
            run.row9(**change)


# ===========================================================================
# Row 8's image binding
# ===========================================================================

class TestImageBinding:
    def test_the_image_id_and_payload_are_recorded_without_starting_it(self, run):
        run.begin()
        evidence = run.image()
        assert evidence["image_id"] == IMAGE_ID and evidence["payload_digest"] == run.digest
        assert evidence["opencode_installed"] == run.host
        saved = json.loads((run.dir / "image.json").read_text("utf-8"))
        assert saved["session"] == run.session["id"]
        verbs = [call[1] for call in run.calls]
        assert "run" not in verbs and "start" not in verbs
        assert ["docker", "rm", "-f", "c0"] in run.calls
        # The payload is read from the immutable id, never from the mutable tag.
        assert any(call[:2] == ["docker", "create"] and call[2] == IMAGE_ID
                   for call in run.calls)

    def test_a_payload_digest_that_differs_is_refused_before_row_8(self, run):
        run.begin()
        contents = copy.deepcopy(run.contents)
        contents[IMAGE_ID][gate.IMAGE_PAYLOAD_DIGEST] = b"f" * 64 + b"\n"
        with pytest.raises(gate.GateError, match="built from other bytes"):
            run.image(contents=contents)
        assert not (run.dir / "image.json").exists()
        # ...so row 8 has nothing to bind to.
        write_row8(run.root, stamp=gate._compact(gate._now()), host=run.host)
        with pytest.raises(gate.GateError, match="no image-build entry in this gate session"):
            run.check8()

    def test_an_inherited_image_naming_another_id_is_refused(self, run):
        run.begin()
        images = {TAG: IMAGE_ID, "mllm-e2e:local": OTHER_IMAGE_ID}
        with pytest.raises(gate.GateError) as caught:
            run.image(environ={"MLLM_E2E_IMAGE": "mllm-e2e:local"}, images=images)
        assert OTHER_IMAGE_ID in str(caught.value) and IMAGE_ID in str(caught.value)
        with pytest.raises(gate.GateError, match="resolves to no image"):
            run.image(environ={"MLLM_E2E_IMAGE": "mllm-e2e:gone"})
        assert run.image(environ={"MLLM_E2E_IMAGE": IMAGE_ID})["image_id"] == IMAGE_ID

    def test_an_image_with_another_opencode_is_refused(self, run):
        run.begin()
        contents = copy.deepcopy(run.contents)
        contents[IMAGE_ID][gate.IMAGE_INSTALLED_REPORT] = json.dumps(
            installed_report("1.18.0")).encode()
        with pytest.raises(gate.GateError, match="installed opencode '1.18.0'"):
            run.image(contents=contents)

    def test_no_session_no_binding(self, run):
        with pytest.raises(gate.GateError, match="no gate session"):
            run.image()


# ===========================================================================
# Row 8: five graded scenarios from this session's run, on the gate's image
# ===========================================================================

class TestRow8:
    @pytest.fixture
    def ready(self, run):
        run.begin()
        run.image()
        run.stamp_now = gate._compact(gate._now())
        return run

    def _refused(self, run, expected, **changes):
        write_row8(run.root, stamp=run.stamp_now, host=run.host, **changes)
        with pytest.raises(gate.GateError) as caught:
            run.check8()
        assert expected in str(caught.value), str(caught.value)
        return str(caught.value)

    def test_a_green_run_is_graded(self, ready):
        write_row8(ready.root, stamp=ready.stamp_now, host=ready.host)
        evidence = ready.check8()
        assert sorted(evidence["scenarios"]) == sorted(gate.ROW8_SCENARIOS)
        assert evidence["image_id"] == IMAGE_ID and evidence["run"] == ready.stamp_now
        assert all(s["verify_record"]["verdict"] == "MATCH"
                   for s in evidence["scenarios"].values())
        assert evidence["scenarios"][CANCEL]["abort_method"] == "client.session.abort"
        assert evidence["posture"]["pass"] is True

    def test_a_missing_scenario_is_named(self, ready):
        self._refused(ready, "opencode-package-gate-supervision: no run record",
                      skip=("opencode-package-gate-supervision",))

    def test_a_record_from_before_the_run_is_missing(self, ready):
        self._refused(ready, "opencode-package-gate-detach: no run record",
                      **{"opencode-package-gate-detach": {"stamp": "20200101T000000Z"}})

    def test_a_skipped_scenario_is_refused(self, ready):
        self._refused(ready, "skipped (not run)",
                      **{"opencode-package-gate-detach": {"outcome": "not-run",
                                                          "entry": {"outcome": "skip"}}})

    def test_a_failing_scenario_is_refused(self, ready):
        self._refused(ready, "its grader did not pass",
                      **{"opencode-package-gate-completion": {"grader": {"pass": False}}})
        self._refused(ready, "checks not passed: ['result_terminal']",
                      **{"opencode-package-gate-completion": {
                          "grader": {"checks": {"tool_called": True, "result_terminal": False}}}})

    def test_an_ungraded_scenario_is_refused(self, ready):
        from tests.container import harness

        self._refused(ready, "ungraded — its bundle carries no readable gate-grader.json",
                      **{"opencode-package-gate-detach": {"no_grader": True}})
        for marker in (harness.PROVISIONAL_REASON, harness.UNGRADED_REASON):
            self._refused(ready, "ungraded —",
                          **{"opencode-package-gate-detach": {"entry": {"reason": marker}}})

    @pytest.mark.parametrize("where", [
        "/work/sample-project/.opencode/skills/multi-llm",
        "/opt/mllm/opencode-plugin", "/opt/mllm/opencode-plugin/../skill",
        "opt/mllm/opencode-plugin/skills/multi-llm", None])
    def test_a_dispatcher_outside_the_installed_payload_is_refused(self, ready, where):
        self._refused(ready, "not from inside /opt/mllm/opencode-plugin",
                      **{"opencode-package-gate-init-status": {
                          "grader": {"dispatcher_skill_dir": where}}})

    def test_the_file_route_is_refused(self, ready):
        self._refused(ready, "ran route 'file'",
                      **{"opencode-package-gate-detach": {"entry": {"route": "file"}}})
        self._refused(ready, "graded route 'file'",
                      **{"opencode-package-gate-detach": {"grader": {"route": "file"}}})

    def test_a_record_naming_a_different_image_is_refused(self, ready):
        self._refused(ready, f"ran on image '{OTHER_IMAGE_ID}', the gate built {IMAGE_ID}",
                      **{"opencode-package-gate-completion": {"image": OTHER_IMAGE_ID}})

    def test_another_host_version_is_refused(self, ready):
        self._refused(ready, "ran opencode '1.18.0'",
                      **{"opencode-package-gate-detach": {"entry": {"cli_version": "1.18.0"}}})

    def test_records_from_two_runs_are_refused(self, ready):
        self._refused(ready, "come from 2 different runs",
                      **{"opencode-package-gate-cancellation": {
                          "stamp": _later(ready.stamp_now, 5)}})

    def test_a_bundle_that_does_not_match_is_refused(self, ready):
        self._refused(ready, "says MISMATCH",
                      **{"opencode-package-gate-detach": {"tamper": True}})

    def test_a_pruned_bundle_is_refused(self, ready):
        """``verify_record.py`` exits 0 for MISSING; row 8 does not."""
        text = self._refused(ready, "says MISSING",
                             **{"opencode-package-gate-detach": {"prune": True}})
        assert "ungraded" in text

    def test_a_record_without_a_bundle_block_is_refused(self, ready):
        self._refused(ready, "carries no bundle block",
                      **{"opencode-package-gate-detach": {"no_bundle": True}})

    def test_cancellation_must_say_how_the_abort_was_sent(self, ready):
        self._refused(ready, "does not say how the abort was sent",
                      **{CANCEL: {"grader": {"abort_method": ""}}})

    def test_the_posture_assertions_must_hold(self, ready):
        self._refused(ready, "no posture-inside.json",
                      **{"opencode-package-gate-detach": {"no_inside": True}})
        self._refused(ready, "disagrees with posture.yaml",
                      **{"opencode-package-gate-detach": {
                          "mismatches": ["HostConfig.Privileged: True"]}})
        write_row8(ready.root, stamp=ready.stamp_now, host=ready.host)
        with pytest.raises(gate.GateError, match="posture.yaml differs from HEAD's"):
            ready.check8(posture_head=lambda: ("1" * 64, False))

    def test_since_is_held_to_the_session(self, ready):
        write_row8(ready.root, stamp=ready.stamp_now, host=ready.host)
        with pytest.raises(gate.GateError, match="reaches back before this gate session"):
            ready.check8(since="20200101T000000Z")
        with pytest.raises(gate.GateError, match="is not a YYYYmmddTHHMMSSZ"):
            ready.check8(since="2026-10-25")


# ===========================================================================
# What the stamp requires, and what it binds
# ===========================================================================

class TestStampEvidence:
    def test_a_full_session_authorizes_a_schema_2_stamp(self, run):
        run.full()
        assert run.problems() == []
        stamp = json.loads(run.stamp().read_text("utf-8"))
        assert stamp["schema"] == 2 and stamp["tier"] == "full"
        assert stamp["rows"] == {row_id: "pass" for row_id in gate.ROW_IDS}
        assert stamp["session"] == run.session["id"]
        assert stamp["candidate_sha"] == gate.head_sha()
        assert stamp["host_version"] == run.host and stamp["registry"] == run.registry
        assert stamp["image_id"] == IMAGE_ID and stamp["digest"] == run.digest
        assert stamp["tarball"] == {key: run.facts[key] for key in
                                    ("path", "integrity", "sha256", "files", "digest")}
        assert stamp["gate"]["row8"]["ran"] == "local"
        assert sorted(stamp["gate"]["row8"]["scenarios"]) == sorted(gate.ROW8_SCENARIOS)
        assert run.check() is None

    def test_write_stamp_without_any_evidence_refuses(self, tmp_path):
        """Called directly, with no gate run: refused before anything is built."""
        package = tmp_path / "pkg"
        with pytest.raises(pkg_build.PayloadError, match="no gate session"):
            pkg_build.write_stamp(package, timestamp="x", allow_dirty=True,
                                  evidence_dir=tmp_path / "empty")
        assert not package.exists(), "the refusal came after a build"

    def test_a_rehearsal_refuses_at_the_stamp_and_the_export(self, run):
        run.full(allow_dirty=True)
        with pytest.raises(pkg_build.PayloadError, match="REHEARSAL"):
            run.stamp()
        with pytest.raises(gate.GateError, match="REHEARSAL"):
            run.export()

    def test_a_missing_row_is_named(self, run):
        run.full()
        for row_id in gate.ROW_IDS:
            path = run.dir / f"row-{row_id}.json"
            kept = path.read_text("utf-8")
            path.unlink()
            problems = run.problems()
            assert f"row {row_id}: no evidence in this session" in _joined(problems)
            path.write_text(kept, "utf-8")
        assert run.problems() == []

    def test_a_skipped_static_row_refuses(self, run):
        run.full()
        node = gate._BY_ID["3"].nodes[0]
        run.ci_rows(outcomes={node: "skipped"})
        problems = _joined(run.problems())
        assert "row 3: did not pass" in problems and f"{node} skipped" in problems

    def test_an_ungraded_or_failing_row_8_refuses(self, run):
        run.full()
        row = json.loads((run.dir / "row-8.json").read_text("utf-8"))
        row["scenarios"]["opencode-package-gate-detach"]["pass"] = False
        run.rewrite_row("8", scenarios=row["scenarios"])
        assert "scenario opencode-package-gate-detach has no passing, graded record" in \
            _joined(run.problems())
        del row["scenarios"]["opencode-package-gate-detach"]
        run.rewrite_row("8", scenarios=row["scenarios"])
        assert "scenario opencode-package-gate-detach" in _joined(run.problems())

    def test_a_wrong_host_refuses(self, run):
        run.full()
        problems = _joined(run.problems(host_version="1.20.0"))
        assert "row 7: ran opencode" in problems and "the image installed opencode" in problems

    def test_a_wrong_image_or_digest_refuses(self, run):
        run.full()
        assert "is not the candidate's" in _joined(run.problems(digest="0" * 64))
        run.rewrite_row("8", image_id=OTHER_IMAGE_ID)
        assert f"row 8: its records name image '{OTHER_IMAGE_ID}'" in _joined(run.problems())

    def test_a_session_with_no_image_entry_refuses(self, run):
        run.full()
        (run.dir / "image.json").unlink()
        assert "no image-build entry in this session" in _joined(run.problems())

    def test_mixed_runs_in_row_8_refuse(self, run):
        run.full()
        row = json.loads((run.dir / "row-8.json").read_text("utf-8"))
        row["scenarios"][CANCEL]["run_id"] = "20990101T000000Z"
        run.rewrite_row("8", scenarios=row["scenarios"])
        assert "row 8: its scenarios come from different runs" in _joined(run.problems())

    def test_a_repacked_tarball_refuses(self, run, tmp_path):
        """Row 7 loaded one file; a later pack under the same name is another."""
        run.full()
        with tarfile.open(run.tarball, "w:gz", compresslevel=1) as archive:
            for path in sorted(run.package.rglob("*")):
                relative = path.relative_to(run.package).as_posix()
                if path.is_file() and (not pkg_build._not_shipped(relative)
                                       or relative == "dist/tree.sha256"):
                    archive.add(path, arcname="package/" + relative, recursive=False)
        problems = _joined(run.problems())
        assert "row 7 loaded" in problems and "it was repacked" in problems

    def test_a_tarball_of_other_bytes_refuses(self, run, tmp_path):
        run.full()
        other = tmp_path / "other"
        shutil.copytree(run.package, other)
        (other / "README.md").write_text("changed\n", "utf-8")
        facts = pkg_build.tarball_facts(make_tarball(other, tmp_path / "other.tgz"))
        assert "the gated tarball's tree digest" in _joined(run.problems(tarball=facts))
        assert "the gated tarball is missing" in _joined(run.problems(tarball=None))

    def test_a_dirty_tree_at_stamp_time_refuses(self, run):
        run.full()
        assert "uncommitted changes now" in _joined(run.problems(dirty=[" M Makefile"]))


class TestCheckStamp:
    """``--check-stamp``: each refusal its own message, against what is true now."""

    @pytest.fixture(scope="class")
    def stamped_once(self, tmp_path_factory, built):
        with pytest.MonkeyPatch.context() as monkeypatch:
            run = GateRun(tmp_path_factory.mktemp("stamped"), monkeypatch, built)
            run.full()
            run.stamp()
            yield run

    @pytest.fixture
    def stamped(self, stamped_once):
        """The class's one stamped session, put back as it was after each row."""
        stamp = stamped_once.package / pkg_build.STAMP_NAME
        kept = stamp.read_bytes(), stamped_once.tarball.read_bytes()
        yield stamped_once
        stamp.write_bytes(kept[0])
        stamped_once.tarball.write_bytes(kept[1])

    def _rewrite(self, run, **fields):
        path = run.package / pkg_build.STAMP_NAME
        data = json.loads(path.read_text("utf-8"))
        data.update(fields)
        for key, value in list(data.items()):
            if value is None:
                del data[key]
        path.write_text(json.dumps(data), "utf-8")
        return run.check()

    def test_a_schema_1_stamp_is_refused(self, stamped):
        legacy = {"tier": "full", "digest": stamped.digest, "version": gate.package_version(),
                  "stamped_at": ""}
        (stamped.package / pkg_build.STAMP_NAME).write_text(json.dumps(legacy), "utf-8")
        assert "digest-only stamp" in stamped.check()

    def test_incomplete_rows_are_refused(self, stamped):
        rows = {row_id: "pass" for row_id in gate.ROW_IDS if row_id != "5b"}
        assert "rows 5b" in self._rewrite(stamped, rows=rows)

    @pytest.mark.parametrize("key", ["session", "candidate_sha", "host_version", "image_id",
                                     "registry", "tarball"])
    def test_a_missing_binding_is_refused(self, stamped, key):
        assert f"lacks {key}" in self._rewrite(stamped, **{key: None})

    def test_each_binding_that_moved_is_named(self, stamped, monkeypatch):
        messages = set()
        for fields, expected in (
                ({"digest": "0" * 64}, "changed since the gate ran"),
                ({"version": "99.99.99"}, "99.99.99"),
                ({"candidate_sha": "e" * 40}, "the gate ran on another candidate"),
                ({"host_version": "1.0.0"}, "the pinned host is now"),
                ({"registry": "https://registry.example.com"}, "the single registry is")):
            stamp = json.loads((stamped.package / pkg_build.STAMP_NAME).read_text("utf-8"))
            reason = self._rewrite(stamped, **fields)
            assert reason and expected in reason, (fields, reason)
            messages.add(reason)
            (stamped.package / pkg_build.STAMP_NAME).write_text(json.dumps(stamp), "utf-8")
        assert len(messages) == 5, "two refusals share a message"
        assert stamped.check() is None
        monkeypatch.setattr(gate, "head_sha", lambda root=None: "a" * 40)
        assert "HEAD is " + "a" * 40 in stamped.check()

    def test_the_gated_tarball_is_bound(self, stamped, tmp_path):
        stamp = json.loads((stamped.package / pkg_build.STAMP_NAME).read_text("utf-8"))
        tarball = dict(stamp["tarball"])
        reason = self._rewrite(stamped, tarball={**tarball, "integrity": "sha512-AAAA"})
        assert "no longer has the stamped integrity" in reason
        reason = self._rewrite(stamped, tarball={**tarball, "files": tarball["files"][:-1]})
        assert "no longer has the stamped file inventory" in reason
        reason = self._rewrite(stamped, tarball={**tarball, "digest": "0" * 64})
        assert "tree digest" in reason
        self._rewrite(stamped, tarball=tarball)
        assert stamped.check() is None
        stamped.tarball.unlink()
        assert "is missing" in stamped.check()


# ===========================================================================
# The release-gate record: exported from the session, validated as CI runs it
# ===========================================================================

@pytest.fixture(scope="module")
def exported(tmp_path_factory, built):
    """One exported record, made by a full passing session, for the module."""
    root = tmp_path_factory.mktemp("exported")
    with pytest.MonkeyPatch.context() as monkeypatch:
        run = GateRun(root, monkeypatch, built)
        run.full()
        run.stamp()
        path = run.export()
    return json.loads(path.read_text("utf-8")), path.name


def _validate(tmp_path: Path, records: dict, record: dict, **override) -> subprocess.CompletedProcess:
    """``validate-record`` exactly as the ``verify`` job runs it — a CLI over a
    records directory — under ``-I -S``: no site-packages at all, which is
    stricter than the job's ``--no-dev`` and proves the same thing."""
    target = tmp_path / "records"
    target.mkdir(parents=True, exist_ok=True)
    for name, data in records.items():
        (target / name).write_text(data if isinstance(data, str) else json.dumps(data), "utf-8")
    args = {"tag": record["tag"], "sha": record["sha"], "version": record["version"],
            "digest": record["candidate_digest"], "host-version": record["host_version"],
            "registry": record["registry"], **override}
    argv = [sys.executable, "-I", "-S", str(GATE_PATH), "validate-record",
            "--records-dir", str(target)]
    for key, value in args.items():
        argv += [f"--{key}", value]
    return subprocess.run(argv, capture_output=True, text=True, timeout=120)


class TestTheDirtyTree:
    """``dirty_paths`` exempts nothing. It used to skip every
    ``records/*-opencode-release-gate.json``, the gate's own export; since the
    export lands in the gitignored ``.gate/export/``, that exemption only hid a
    hand-edited committed record."""

    RECORDS = "skills/multi-llm/tests/container/records"

    @staticmethod
    def _git(repo, *args):
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent),
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
        done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main",
                               "-C", str(repo), *args], env=env, capture_output=True, text=True,
                              timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout

    @pytest.fixture()
    def repo(self, tmp_path):
        repo = tmp_path / "repo"
        records = repo / self.RECORDS
        records.mkdir(parents=True)
        (records / f"20261025T000000Z{gate.RECORD_SUFFIX}").write_text('{"tag": "x"}\n')
        self._git(repo, "init", "-q")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "a committed record")
        return repo

    def test_a_clean_tree_is_clean(self, repo):
        assert gate.dirty_paths(repo) == []

    def test_a_hand_edited_committed_record_is_dirty(self, repo):
        name = f"20261025T000000Z{gate.RECORD_SUFFIX}"
        (repo / self.RECORDS / name).write_text('{"tag": "edited"}\n')
        assert gate.dirty_paths(repo) == [f" M {self.RECORDS}/{name}"]

    def test_an_untracked_record_is_dirty(self, repo):
        name = f"20261026T000000Z{gate.RECORD_SUFFIX}"
        (repo / self.RECORDS / name).write_text('{"tag": "new"}\n')
        assert gate.dirty_paths(repo) == [f"?? {self.RECORDS}/{name}"]

    def test_a_deleted_record_is_dirty(self, repo):
        name = f"20261025T000000Z{gate.RECORD_SUFFIX}"
        (repo / self.RECORDS / name).unlink()
        assert gate.dirty_paths(repo) == [f" D {self.RECORDS}/{name}"]

    def test_no_exemption_is_left_in_the_module(self):
        assert not hasattr(gate, "_own_export")


class TestExport:
    def test_the_default_export_stays_out_of_the_payload_prefix(self, tmp_path, monkeypatch, built):
        """records/ lies under skills/multi-llm/, the prefix the payload's
        dirty-tree refusal covers: an untracked record written there made
        `--check-stamp` and `opencode-publish-first` refuse the candidate the
        gate had just stamped (measured on the first green full gate). So the
        default lands in the gitignored evidence directory, and the record
        reaches records/ only in the post-fast-forward commit."""
        run = GateRun(tmp_path, monkeypatch, built)
        run.full()
        run.stamp()
        path = gate.export_record(directory=run.dir,
                                  stamp_path=run.package / pkg_build.STAMP_NAME,
                                  tarball=run.tarball)
        assert path.parent == run.dir / gate.EXPORT_SUBDIR
        assert gate.EXPORT_DIR == gate.EVIDENCE_DIR / gate.EXPORT_SUBDIR
        assert not gate.EXPORT_DIR.resolve().is_relative_to(
            (gate.REPO_ROOT / "skills" / "multi-llm").resolve())
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(gate.EXPORT_DIR / "x-opencode-release-gate.json")],
            cwd=gate.REPO_ROOT)
        assert ignored.returncode == 0, "the export directory must be gitignored"

    def test_the_exported_record_validates(self, exported, tmp_path):
        record, name = exported
        assert re.fullmatch(r"\d{8}T\d{6}Z-opencode-release-gate\.json", name)
        assert record["rehearsal"] is False and record["row8"]["ran"] == "local"
        assert record["tag"] == gate.release_tag()
        assert [s["row"] for s in record["scenarios"]] == list(gate.ROW8_SCENARIOS)
        assert record["image"]["installed_versions"]["clis"]
        assert record["posture"]["pass"] is True
        done = _validate(tmp_path, {name: record}, record)
        assert done.returncode == 0, done.stderr
        assert "validates" in done.stdout and "ran locally" in done.stdout

    def test_the_schema_is_draft_2020_12_and_within_the_validators_keywords(self, exported):
        jsonschema = pytest.importorskip("jsonschema")
        schema = gate.load_record_schema()
        jsonschema.Draft202012Validator.check_schema(schema)
        record, _name = exported
        assert list(jsonschema.Draft202012Validator(schema).iter_errors(record)) == []

        def keywords(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    if key == "properties":
                        for child in value.values():
                            yield from keywords(child)
                    elif isinstance(value, (dict, list)) and key not in ("enum", "required"):
                        yield from keywords(value)
            elif isinstance(node, list):
                for item in node:
                    yield from keywords(item)

        assert set(keywords(schema)) <= gate.SCHEMA_KEYWORDS

    def test_the_two_validators_agree(self, exported):
        """The stdlib engine and jsonschema refuse the same broken records."""
        jsonschema = pytest.importorskip("jsonschema")
        record, _name = exported
        schema = gate.load_record_schema()
        for broken in ({**record, "extra": 1}, {**record, "rows": {**record["rows"], "7": "fail"}},
                       {**record, "image": {**record["image"], "id": "latest"}},
                       {**record, "scenarios": [{**record["scenarios"][0], "pass": "true"}]},
                       {k: v for k, v in record.items() if k != "session"},
                       {**record, "schema": True}):
            ours = gate.schema_errors(broken, schema)
            theirs = list(jsonschema.Draft202012Validator(schema).iter_errors(broken))
            assert bool(ours) == bool(theirs), (ours, theirs)

    def test_the_record_carries_no_local_path(self, exported):
        record, _name = exported
        text = json.dumps(record)
        assert str(REPO_ROOT) not in text and str(Path.home()) not in text

    def test_export_needs_the_stamp_of_this_session(self, run):
        run.full()
        with pytest.raises(gate.GateError, match="no schema-2 stamp"):
            run.export()
        run.stamp()
        path = run.package / pkg_build.STAMP_NAME
        stamp = json.loads(path.read_text("utf-8"))
        path.write_text(json.dumps({**stamp, "session": "0" * 32}), "utf-8")
        with pytest.raises(gate.GateError, match="the evidence is session"):
            run.export()


def _named(record: dict, stamp: str = "20991231T000000Z") -> tuple:
    return f"{stamp}-opencode-release-gate.json", {**copy.deepcopy(record), "exported_at": stamp}


class TestValidateRecord:
    """The ``verify`` job's validator, run as a CLI against fixture records."""

    def _refused(self, tmp_path, exported, mutate=None, expected="", records=None, **override):
        record, _ = exported
        name, data = _named(record)
        if mutate:
            mutate(data)
        done = _validate(tmp_path, records if records is not None else {name: data},
                         record, **override)
        assert done.returncode != 0, done.stdout
        assert expected in done.stderr, done.stderr
        return done.stderr

    def test_a_genuine_record_validates(self, tmp_path, exported):
        record, _ = exported
        name, data = _named(record)
        assert _validate(tmp_path, {name: data}, record).returncode == 0

    def test_an_absent_record_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, records={}, expected="no committed release-gate "
                                                             "record names")

    @pytest.mark.parametrize("flag", [True, "true", 1, None, "false"])
    def test_a_rehearsal_is_refused(self, tmp_path, exported, flag):
        self._refused(tmp_path, exported, lambda d: d.update(rehearsal=flag),
                      "rehearsal: ")

    def test_a_missing_rehearsal_flag_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, lambda d: d.pop("rehearsal"), "rehearsal: ")

    def test_a_wrong_tag_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, expected="no committed release-gate record names "
                      "opencode-v9.9.9", tag="opencode-v9.9.9")

    @pytest.mark.parametrize("flag, value, field", [
        ("sha", "0" * 40, "sha: "), ("version", "9.9.9", "version: "),
        ("digest", "1" * 64, "candidate_digest: "), ("host-version", "1.18.0", "host_version: "),
    ])
    def test_a_binding_that_differs_is_refused_by_field(self, tmp_path, exported, flag,
                                                        value, field):
        text = self._refused(tmp_path, exported, expected=field, **{flag: value})
        if flag == "digest":
            assert "image.payload_digest" in text

    def test_another_registry_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, expected="registry: --registry",
                      registry="https://registry.example.com")
        self._refused(tmp_path / "b", exported,
                      lambda d: d.update(registry="https://registry.example.com"),
                      "registry: the record says")

    def test_a_missing_image_id_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, lambda d: d["image"].pop("id"), "image/id: missing")
        self._refused(tmp_path / "b", exported, lambda d: d["image"].update(id=""),
                      "image/id")

    def test_an_image_payload_from_other_bytes_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported,
                      lambda d: d["image"].update(payload_digest="2" * 64),
                      "image.payload_digest")

    def test_mixed_sessions_or_images_are_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported,
                      lambda d: d["scenarios"][2].update(session="f" * 32),
                      "scenarios from different sessions")
        self._refused(tmp_path / "b", exported,
                      lambda d: d["scenarios"][3].update(image_id=OTHER_IMAGE_ID),
                      "scenarios from different images")
        self._refused(tmp_path / "c", exported,
                      lambda d: d["scenarios"][4].update(run_id="20990101T000000Z"),
                      "the scenarios come from 2 runs")

    def test_a_missing_or_duplicated_scenario_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, lambda d: d["scenarios"].pop(1),
                      "scenarios: opencode-package-gate-detach is missing")
        self._refused(tmp_path / "b", exported,
                      lambda d: d["scenarios"].append(copy.deepcopy(d["scenarios"][0])),
                      "opencode-package-gate-init-status appears 2 times")

    @pytest.mark.parametrize("change, expected", [
        ({"outcome": "not-run"}, "skipped"),
        ({"graded": False}, "ungraded"),
        ({"grader": ""}, "ungraded"),
        ({"pass": False}, "is not boolean true"),
        ({"pass": "true"}, "pass"),
        ({"verify_record": {"exit": 0, "verdict": "MISSING", "output": ""}}, "must MATCH"),
    ])
    def test_a_skipped_ungraded_or_failing_scenario_is_refused(self, tmp_path, exported,
                                                               change, expected):
        self._refused(tmp_path, exported, lambda d: d["scenarios"][1].update(change), expected)

    def test_a_failed_posture_is_refused(self, tmp_path, exported):
        self._refused(tmp_path, exported, lambda d: d["posture"].update({"pass": False}),
                      "posture.pass")

    def test_the_schema_is_enforced(self, tmp_path, exported):
        self._refused(tmp_path, exported, lambda d: d.update(note="by hand"),
                      "schema: note: not a field this schema allows")
        self._refused(tmp_path / "b", exported, lambda d: d["rows"].update({"8": "skip"}),
                      "schema: rows/8")

    def test_a_renamed_record_is_refused(self, tmp_path, exported):
        record, _ = exported
        name, data = _named(record)
        self._refused(tmp_path, exported, expected="exported_at:",
                      records={"20991231T235959Z-opencode-release-gate.json": data})

    def test_an_older_valid_record_is_never_a_fallback(self, tmp_path, exported):
        record, _ = exported
        old_name, old = _named(record, "20990101T000000Z")
        new_name, new = _named(record, "20991231T000000Z")
        new["sha"] = "0" * 40
        text = self._refused(tmp_path, exported, expected="sha: the record says",
                             records={old_name: old, new_name: new})
        assert new_name in text and "never count as a fallback" in text
        # The older one alone is accepted, which is what makes the case real.
        assert _validate(tmp_path / "b", {old_name: old}, record).returncode == 0

    def test_an_unorderable_record_name_is_refused(self, tmp_path, exported):
        """A record whose name carries no stamp cannot be placed against the
        newest, so it is refused rather than skipped."""
        record, _ = exported
        name, data = _named(record)
        self._refused(tmp_path, exported, expected="cannot be ordered",
                      records={name: data, "latest-opencode-release-gate.json": data})

    def test_an_unreadable_record_is_refused(self, tmp_path, exported):
        record, _ = exported
        name, data = _named(record)
        self._refused(tmp_path, exported, expected="unreadable",
                      records={name: data, "20991231T000001Z-opencode-release-gate.json": "{"})


# ===========================================================================
# One bounded step, its log kept
# ===========================================================================

class TestStep:
    def _step(self, tmp_path, *command, timeout=30, capture=None):
        argv = [sys.executable, str(GATE_PATH), "--evidence-dir", str(tmp_path), "step",
                "demo", "--timeout", str(timeout)]
        if capture:
            argv += ["--capture", str(capture)]
        return subprocess.run([*argv, "--", *command], capture_output=True, text=True,
                              timeout=120)

    def test_output_is_tee_d_and_the_last_line_captured(self, tmp_path):
        done = self._step(tmp_path, sys.executable, "-c",
                          "import sys; print('one'); print('noise', file=sys.stderr); "
                          "print('/abs/bin/opencode')", capture=tmp_path / "cap.txt")
        assert done.returncode == 0, done.stderr
        assert "one" in done.stdout and "noise" in done.stderr
        assert (tmp_path / "cap.txt").read_text("utf-8") == "/abs/bin/opencode\n"
        log = (tmp_path / "logs" / "demo.log").read_text("utf-8")
        assert "one" in log and "noise" in log

    def test_a_failure_keeps_its_exit_and_names_its_log(self, tmp_path):
        done = self._step(tmp_path, sys.executable, "-c", "raise SystemExit(7)")
        assert done.returncode == 7
        assert "step demo failed (exit 7)" in done.stderr and "demo.log" in done.stderr

    @pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX; the gate "
                        "runs on Linux")
    def test_the_bound_kills_the_whole_group(self, tmp_path):
        """A grandchild dies with the step: the child leads its own session."""
        marker = tmp_path / "grandchild.pid"
        script = textwrap.dedent(f"""
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            open({str(marker)!r}, "w").write(str(child.pid))
            time.sleep(60)
        """)
        started = time.monotonic()
        done = self._step(tmp_path, sys.executable, "-c", script, timeout=2)
        assert done.returncode == 124 and "exceeded its 2 s bound" in done.stderr
        assert time.monotonic() - started < 60
        pid = int(marker.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the grandchild outlived the step's bound")

    def test_the_grace_is_the_probes_sweep_bound_plus_a_margin(self):
        """Imported from the probe, never re-typed: a sweep that grows moves
        the grace with it."""
        import inspect

        probe = gate._sibling("probe", "opencode_probe")
        assert probe.SWEEP_BOUND_S == probe.SWEEP_STAGES * probe.KILL_GRACE_S
        assert probe.SWEEP_STAGES >= 2  # the bounded child's session, then the stub model
        assert gate.STEP_GRACE_MARGIN_S > 0
        assert gate.step_grace_s() == probe.SWEEP_BOUND_S + gate.STEP_GRACE_MARGIN_S
        assert inspect.signature(gate.run_step).parameters["grace"].default is None
        source = inspect.getsource(gate.run_step)
        assert "step_grace_s()" in source and "10.0" not in source

    @pytest.mark.skipif(os.name != "posix", reason="sessions and SIGTERM dispositions are "
                        "POSIX; the gate runs on Linux")
    def test_a_probe_interrupted_at_its_bound_finishes_its_whole_sweep(self, tmp_path):
        """The probe's worst case, with its real constants: ``SWEEP_STAGES``
        children, each in a session of its own and each ignoring SIGTERM for
        longer than ``KILL_GRACE_S``, swept one after the other by the probe's
        own code (``run_bounded``'s sweep first, then the next, as the stub
        model's comes after the ``opencode run`` it served). The step's SIGKILL
        must not land before the last one is killed: the sweeper would die and
        that child, outside the step's group, would outlive the gate."""
        script = textwrap.dedent(f"""
            import os, subprocess, sys
            sys.path.insert(0, {str(PACKAGE_DIR)!r})
            import probe
            probe.install_exit_handlers()
            STUBBORN = ("import os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(600)")
            root = {str(tmp_path)!r}
            held = []
            try:
                for index in range(1, probe.SWEEP_STAGES):
                    held.append(subprocess.Popen(
                        [sys.executable, "-c", STUBBORN, os.path.join(root, f"held-{{index}}.pid")],
                        start_new_session=True))
                probe.run_bounded(
                    [sys.executable, "-c", STUBBORN, os.path.join(root, "bounded.pid")],
                    cwd=root, env=dict(os.environ), budget_s=600,
                    stdout=os.path.join(root, "bounded.out"), stderr=os.path.join(root, "bounded.err"))
            finally:
                with probe.held_exit_signals():
                    for proc in reversed(held):
                        probe._stop(proc)
                open(os.path.join(root, "swept"), "w").write("done")
        """)
        probe = gate._sibling("probe", "opencode_probe")
        try:
            started = time.monotonic()
            done = self._step(tmp_path, sys.executable, "-c", script, timeout=3)
            elapsed = time.monotonic() - started
            assert done.returncode == 124, done.stdout + done.stderr
            pids = [int(p.read_text()) for p in sorted(tmp_path.glob("*.pid"))]
            assert len(pids) == probe.SWEEP_STAGES, "not every child started before the bound"
            # The sweep really took the stages' graces: SIGTERM was ignored.
            assert elapsed >= 3 + probe.SWEEP_BOUND_S - 1, elapsed
            assert (tmp_path / "swept").exists(), (
                "the step's SIGKILL landed mid-sweep: the probe never finished it")
            for pid in pids:
                for _ in range(50):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                else:
                    pytest.fail(f"pid {pid} ignored SIGTERM and outlived the step")
        finally:
            for path in tmp_path.glob("*.pid"):
                try:
                    os.kill(int(path.read_text()), 9)
                except (ProcessLookupError, ValueError):
                    pass

    def test_the_child_reads_no_terminal(self, tmp_path):
        done = self._step(tmp_path, sys.executable, "-c",
                          "import sys; print(repr(sys.stdin.read()))")
        assert done.returncode == 0 and "''" in done.stdout


# ===========================================================================
# The Makefile: the recipe is the order, the gate is the authority
# ===========================================================================

def _raw_recipes() -> dict:
    """``{target: [logical recipe lines]}``, continuations joined, unexpanded."""
    text = MAKEFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    recipes: dict = {}
    current = None
    for line in text.splitlines():
        if line.startswith("\t"):
            if current:
                recipes[current].append(re.sub(r"\s+", " ", line.strip()))
            continue
        match = re.match(r"^([A-Za-z0-9_.\-/]+)\s*:(?!=)(.*)$", line)
        current = match.group(1) if match else None
        if current:
            recipes.setdefault(current, [])
    return recipes


#: The full gate's steps, in the order the plan fixes (step 0's ordering bullet
#: and the work package's recipe): reset, CI rows, the clean candidate, host
#: install, row 7, the clean-staged image and its binding, row 8, bump
#: discipline, row 9, the stamp, the record.
GATE_STEPS = ["ci-rows", "candidate", "host-install", "row7-probe", "row7-record",
              "image-build", "image-bind", "row8-run", "row8-check", "check-bump",
              "row9-probe", "row9-record", "stamp", "export"]


def _clean_make_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "ALLOW_DIRTY",
                        "MULTI_LLM_ALLOW_DIRTY", "OPENCODE_STAGE_DIRTY")}
    return env


def _make_n(*args) -> str:
    if not shutil.which("make") or not sys.platform.startswith("linux"):
        pytest.skip("needs GNU make and bash on Linux, where the full gate runs")
    done = subprocess.run(["make", "-n", "-s", "--no-print-directory", *args], cwd=REPO_ROOT,
                          env=_clean_make_env(), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


class TestTheGateRecipe:
    def test_the_steps_are_sequential_recipe_lines_in_order(self):
        makefile = MAKEFILE.read_text(encoding="utf-8")
        header = re.search(r"^opencode-package-check:(.*)$", makefile, re.M)
        assert header and not header.group(1).strip(), (
            "the steps are recipe lines, never prerequisites: `make -j` would reorder them")
        lines = _raw_recipes()["opencode-package-check"]
        assert lines[0].startswith("@$(OPENCODE_GATE) begin"), lines[0]
        steps = [re.match(r"\$\(OPENCODE_GATE_STEP\) (\S+)", line).group(1)
                 for line in lines if line.startswith("$(OPENCODE_GATE_STEP)")]
        assert steps == GATE_STEPS
        for line in lines:
            if line.startswith("$(OPENCODE_GATE_STEP)"):
                assert re.search(r" --timeout \d+ ", line), f"unbounded step: {line}"
        others = [line for line in lines if not line.startswith("$(OPENCODE_GATE_STEP)")]
        assert len(others) == 4, others  # begin, row 8's since, summary, the final echo
        since_at = next(i for i, line in enumerate(lines) if "row8-since.txt" in line
                        and line.startswith("@date"))
        run_at = next(i for i, line in enumerate(lines) if "row8-run" in line)
        assert since_at == run_at - 1

    def test_each_step_runs_what_the_contract_names(self):
        lines = {re.match(r"\$\(OPENCODE_GATE_STEP\) (\S+)", line).group(1): line
                 for line in _raw_recipes()["opencode-package-check"]
                 if line.startswith("$(OPENCODE_GATE_STEP)")}
        assert "$(MAKE) --no-print-directory opencode-package-check-ci" in lines["ci-rows"]
        assert "$(MAKE) --no-print-directory opencode-package" in lines["candidate"]
        assert "bump.py install --prefix" in lines["host-install"]
        assert "--capture $(OPENCODE_GATE_DIR)/raw/opencode-bin.txt" in lines["host-install"]
        row7 = lines["row7-probe"]
        for part in ("probe.py row7", '--tarball "$(OPENCODE_TARBALL)"',
                     '--opencode "$$(cat $(OPENCODE_GATE_DIR)/raw/opencode-bin.txt)"',
                     "--registry $(OPENCODE_REGISTRY)", "--out $(OPENCODE_GATE_DIR)/raw/row7.json"):
            assert part in row7, part
        assert "record 7 --from $(OPENCODE_GATE_DIR)/raw/row7.json" in lines["row7-record"]
        assert "image --image $(OPENCODE_GATE_IMAGE)" in lines["image-bind"]
        assert 'MLLM_E2E_IMAGE="$$(cat $(OPENCODE_GATE_DIR)/raw/image-id.txt)"' in lines["row8-run"]
        assert "$(OPENCODE_GATE_PYTEST)" in lines["row8-run"]
        assert 'check-row8 --since "$$(cat' in lines["row8-check"]
        assert "$(MAKE) --no-print-directory opencode-check-bump" in lines["check-bump"]
        assert "opencode-registry-absent OUT=$(OPENCODE_GATE_DIR)/raw/row9.json" in lines["row9-probe"]
        assert "record 9 --from $(OPENCODE_GATE_DIR)/raw/row9.json" in lines["row9-record"]
        assert "$(OPENCODE_BUILD_CLEAN) --write-stamp" in lines["stamp"]
        assert "export-record" in lines["export"]

    def test_row_8_runs_offline_on_the_gate_s_image(self):
        makefile = MAKEFILE.read_text(encoding="utf-8")
        block = re.search(r"^OPENCODE_GATE_PYTEST :=(.*?)(?:\n\n|\n#)", makefile,
                          re.M | re.S).group(1)
        for name in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                     "OPENAI_API_KEY"):
            assert f"-u {name}" in block, name
        assert "MLLM_E2E_RUNS_KEEP=0" in block
        assert "test_container_opencode_package_gate.py" in block
        assert '-m "container and not live and not live_provider"' in block
        assert "OPENCODE_GATE_IMAGE := mllm-e2e:opencode-gate" in makefile

    def test_the_gate_s_image_staging_never_passes_allow_dirty(self):
        """Statically, then as make expands it with ALLOW_DIRTY unset."""
        lines = _raw_recipes()["opencode-package-check"]
        build = next(line for line in lines if " image-build " in line)
        assert "e2e-image IMAGE=$(OPENCODE_GATE_IMAGE) OPENCODE_STAGE_DIRTY=$(OPENCODE_DIRTY)" \
            in build
        assert "--allow-dirty" not in build
        # ...and OPENCODE_DIRTY is empty unless ALLOW_DIRTY is set (a rehearsal).
        assert "OPENCODE_DIRTY := $(if $(ALLOW_DIRTY),--allow-dirty,)" in \
            MAKEFILE.read_text(encoding="utf-8")
        gate_stage = _make_n("e2e-image", "IMAGE=mllm-e2e:opencode-gate", "OPENCODE_STAGE_DIRTY=")
        assert "build.py" in gate_stage and "--allow-dirty" not in gate_stage
        assert "MULTI_LLM_ALLOW_DIRTY= " in gate_stage
        # ...while a plain `make e2e-image` keeps the development path.
        assert "build.py --allow-dirty" in _make_n("e2e-image")

    def test_allow_dirty_reaches_the_session_as_a_rehearsal(self):
        lines = _raw_recipes()["opencode-package-check"]
        assert lines[0] == "@$(OPENCODE_GATE) begin $(if $(ALLOW_DIRTY),--allow-dirty,)"

    def test_the_ci_tier_grades_both_selections(self):
        recipe = "\n".join(_raw_recipes()["opencode-package-check-ci"])
        package_xml = "$(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml"
        plain_xml = "$(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml"
        assert f"rm -f {package_xml} {plain_xml}" in recipe, "a stale junit file would grade"
        assert f'-m "package and not container" -q --junitxml={package_xml}' in recipe
        assert f'-m "not live and not container and not package" -q --junitxml={plain_xml}' \
            in recipe
        assert f"grade-ci --junit {package_xml} --junit {plain_xml}" in recipe
        assert recipe.index("--junitxml") < recipe.index("grade-ci")
        assert "exit $$rc" in recipe

    def test_the_package_target_scans_the_exact_tarball(self):
        recipe = "\n".join(_raw_recipes()["opencode-package"])
        assert '--scan-tarball "$(OPENCODE_TARBALL)"' in recipe
        assert "ls -t" not in recipe

    def test_the_stale_comments_are_gone(self):
        makefile = MAKEFILE.read_text(encoding="utf-8")
        for stale in ("REFUSING (row 7)", "REFUSING (row 8)", "Two of the three rows are not",
                      "cannot share one run, only its reasoning",
                      "every one of them today is a file-shim row",
                      "no clean-\\$$HOME install probe is wired"):
            assert stale not in makefile, stale
        assert "they share one implementation" in makefile
