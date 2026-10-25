"""The OpenCode publishing workflow, ``.github/workflows/publish-opencode-plugin.yml``, offline.

**What is graded.** The gate/verify/publish split the plan's step 0 requires
(plans/opencode-plugin-publish.md, the workflow-split, ``fetch-depth`` and
bootstrap bullets, and the regression list after them), in three ways:

* **Statically**, from the parsed YAML, the way
  ``test_the_publish_workflow_is_split_and_unprivileged`` grades the Codex
  workflow: per-job permissions, which job holds ``id-token``, what each job may
  run, step order, the pins, the shared npm setup, the one row-9 entry point.
* **By evaluating the workflow's own ``if:`` and ``env`` expressions** under a
  tag push, a ``mode: rehearse`` dispatch and a ``mode: publish`` dispatch, with
  a small evaluator of exactly the expression subset the file uses (it refuses
  anything else, so a new construct fails here loudly instead of being guessed
  at). That is what shows which jobs a run reaches and which path the gate takes.
* **By running the jobs' own shell steps**, extracted from the YAML and never
  retyped, the way ``TestPromoteGate`` runs the Codex promote job's heredoc:
  the route selection, the default-branch refusal, the tag resolution, the
  committed-record read, the release-gate record check through
  ``release_gate.py validate-record``, publish's digest comparison and the
  bootstrap record's validation through ``registry_record.py``. ``uv`` is a
  shim that runs this interpreter, so the validators run unchanged.

**Offline.** Nothing here reaches GitHub, npm or a registry, or starts an
OpenCode binary. The shell rows need ``bash`` and ``git`` and skip, by name, on
Windows, where the workflow never runs.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_opencode_registry_absent import (
    ABSENCE_QUERY, CODE_PARSE, ENTRY_POINT, SINGLE, _strip_shell_comments,
)

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-opencode-plugin.yml"
TESTS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
RECORDS = "skills/multi-llm/tests/container/records"

TEXT = WORKFLOW.read_text(encoding="utf-8")
DOCUMENT = yaml.safe_load(TEXT)
JOBS = DOCUMENT["jobs"]
#: PyYAML reads the bare key ``on`` as the boolean ``True``.
TRIGGERS = DOCUMENT[True] if True in DOCUMENT else DOCUMENT["on"]
WORKFLOW_ENV = {key: str(value) for key, value in DOCUMENT["env"].items()}
BOOTSTRAP_TAG = WORKFLOW_ENV["BOOTSTRAP_TAG"]

needs_shell = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None or shutil.which("git") is None,
    reason="runs the workflow's bash steps, which need bash and git; the workflow "
           "runs on ubuntu-latest only",
)


# ---------------------------------------------------------------------------
# The workflow's expressions, evaluated
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<op>==|!=|&&|\|\||!|\(|\))"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_.-]*))")
#: The status functions the file uses on steps. Jobs must use none of them:
#: ``always()`` on ``publish`` would let it run after a failed ``verify``.
_FUNCTIONS = {"always": True, "success": True}


def _tokens(text: str) -> list:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise AssertionError(f"the evaluator does not know {text[pos:]!r} in {text!r}")
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


def _truthy(value) -> bool:
    return value not in (None, False, "", 0)


def _equal(left, right) -> bool:
    left = "" if left is None else left
    right = "" if right is None else right
    if isinstance(left, str) and isinstance(right, str):
        return left.lower() == right.lower()   # GitHub compares strings case-insensitively
    return left == right


def evaluate(expression: str, context: dict):
    """The value of one Actions expression, with ``&&``/``||`` returning an
    operand as GitHub's do. ``${{ }}`` is optional around it."""
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    tokens = _tokens(text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def take(value=None):
        nonlocal pos
        token = peek()
        if value is not None and token[1] != value:
            raise AssertionError(f"expected {value!r} in {expression!r}, got {token[1]!r}")
        pos += 1
        return token

    def primary():
        kind, value = take()
        if value == "(":
            inner = either()
            take(")")
            return inner
        if kind == "str":
            return value[1:-1].replace("''", "'")
        if kind == "name":
            if value in ("true", "false", "null"):
                return {"true": True, "false": False, "null": None}[value]
            if peek()[1] == "(":
                take("(")
                take(")")
                if value not in _FUNCTIONS:
                    raise AssertionError(f"the evaluator does not know {value}() in {expression!r}")
                return _FUNCTIONS[value]
            node = context
            for part in value.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            return node
        raise AssertionError(f"unexpected {value!r} in {expression!r}")

    def comparison():
        left = primary()
        if peek()[1] in ("==", "!="):
            op = take()[1]
            right = primary()
            return _equal(left, right) == (op == "==")
        return left

    def negation():
        if peek()[1] == "!":
            take("!")
            return not _truthy(negation())
        return comparison()

    def both():
        left = negation()
        while peek()[1] == "&&":
            take("&&")
            right = negation()
            left = right if _truthy(left) else left
        return left

    def either():
        left = both()
        while peek()[1] == "||":
            take("||")
            right = both()
            left = left if _truthy(left) else right
        return left

    value = either()
    if pos != len(tokens):
        raise AssertionError(f"trailing tokens in {expression!r}: {tokens[pos:]}")
    return value


def _push(tag: str) -> dict:
    return {"github": {"event_name": "push", "ref_type": "tag", "ref_name": tag}, "inputs": {}}


def _dispatch(**inputs) -> dict:
    return {"github": {"event_name": "workflow_dispatch", "ref_type": "branch",
                       "ref_name": "master"}, "inputs": inputs}


#: Every input filled in, junk where the mode does not read it, so a reader that
#: ignored the mode guard would be caught taking the wrong value.
EVENTS = {
    "tag push": _push("opencode-v0.1.2"),
    "bootstrap tag push": _push(BOOTSTRAP_TAG),
    "rehearse": _dispatch(mode="rehearse", as_tag="opencode-v0.1.2", path="normal",
                          tag="opencode-v9.9.9"),
    "rehearse bootstrap": _dispatch(mode="rehearse", as_tag=BOOTSTRAP_TAG, path="bootstrap",
                                    tag="opencode-v9.9.9"),
    "publish": _dispatch(mode="publish", tag="opencode-v0.1.2", as_tag=BOOTSTRAP_TAG,
                         path="bootstrap"),
}


def reached(context: dict) -> set:
    """The jobs a run with this event reaches: its ``if`` holds and every job it
    needs was reached too."""
    out: set = set()
    pending = dict(JOBS)
    while pending:
        progressed = False
        for job_id, job in list(pending.items()):
            needs = job.get("needs") or []
            needs = [needs] if isinstance(needs, str) else list(needs)
            if any(need in pending for need in needs):
                continue
            del pending[job_id]
            progressed = True
            if all(need in out for need in needs) and _truthy(evaluate(job.get("if", "true"),
                                                                       context)):
                out.add(job_id)
        assert progressed, "the jobs' needs form a cycle"
    return out


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def steps(job_id: str) -> list:
    return JOBS[job_id]["steps"]


def step(job_id: str, name: str) -> dict:
    found = [s for s in steps(job_id) if s.get("name") == name]
    assert len(found) == 1, f"{job_id} has {len(found)} steps named {name!r}"
    return found[0]


def index(job_id: str, predicate) -> int:
    found = [i for i, s in enumerate(steps(job_id)) if predicate(s)]
    assert len(found) == 1, f"{job_id}: {len(found)} steps match"
    return found[0]


def code(s: dict) -> str:
    return _strip_shell_comments(s.get("run") or "")


def _uses(s: dict) -> str:
    return str(s.get("uses") or "").split("@")[0]


def _all_steps():
    for job_id in JOBS:
        for s in steps(job_id):
            yield job_id, s


ROW9_JOBS = ("gate", "verify", "publish")
#: The gate's canary: row 9's entry point, asked about a version never published.
CANARY_NAME = "Row 9's canary: a version never published answers E404"


def _row9(job_id: str) -> dict:
    """The job's row 9 proper: its one call of the entry point other than the
    gate's canary, which asks about another version on purpose."""
    found = [s for s in steps(job_id) if ENTRY_POINT.search(code(s))
             and s.get("name") != CANARY_NAME]
    assert len(found) == 1, f"{job_id} runs row 9's entry point {len(found)} times"
    return found[0]


def _npm_setup(job_id: str) -> tuple:
    """The job's setup-node step and its pinned-npm step, as written."""
    node = [s for s in steps(job_id) if _uses(s) == "actions/setup-node"]
    npm = [s for s in steps(job_id) if s.get("name") == "Install the pinned npm"]
    assert len(node) == 1 and len(npm) == 1, job_id
    return node[0], npm[0]


# ---------------------------------------------------------------------------
# Running a step's own shell
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
                           "-C", str(repo), *args], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


class Shell:
    """Runs one step's ``run`` text the way the runner does —
    ``bash --noprofile --norc -eo pipefail <file>`` — with the workflow-level
    env, ``GITHUB_OUTPUT``/``GITHUB_STEP_SUMMARY``/``RUNNER_TEMP`` in a temporary
    tree, and ``uv``/``python``/``python3`` shims first on ``PATH``: ``uv run
    ... -- CMD`` runs ``CMD`` with this interpreter, and logs its arguments."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "shell"
        self.bin = self.root / "bin"
        self.bin.mkdir(parents=True)
        self.runner_temp = self.root / "runner-temp"
        self.runner_temp.mkdir()
        self.uv_log = self.root / "uv-calls.log"
        (self.bin / "uv").write_text(
            "#!/bin/sh\n"
            'printf \'%s\\n\' "$*" >> "$UV_SHIM_LOG"\n'
            'while [ $# -gt 0 ]; do arg="$1"; shift; [ "$arg" = "--" ] && break; done\n'
            'exec "$@"\n', encoding="utf-8")
        for name in ("python", "python3"):
            (self.bin / name).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
                                         encoding="utf-8")
        for path in self.bin.iterdir():
            path.chmod(0o755)
        self.count = 0

    def run(self, s: dict, *, env: dict, cwd: Path, context: dict | None = None):
        """``(completed, outputs)``. The step's own ``env`` expressions are
        evaluated under *context* when given; *env* then fills or overrides."""
        self.count += 1
        script = self.root / f"step-{self.count}.sh"
        script.write_text(s["run"], encoding="utf-8")
        output = self.root / f"output-{self.count}"
        output.write_text("", encoding="utf-8")
        environ = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(self.root), "LANG": "C.UTF-8",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(self.root / "summary.md"),
            "RUNNER_TEMP": str(self.runner_temp),
            "GITHUB_WORKSPACE": str(cwd),
            "GITHUB_REF": "refs/heads/master", "GITHUB_SHA": "0" * 40,
            "UV_SHIM_LOG": str(self.uv_log),
            "GIT_CONFIG_NOSYSTEM": "1",
            **WORKFLOW_ENV,
        }
        if context is not None:
            for key, value in (s.get("env") or {}).items():
                result = evaluate(str(value), context)
                environ[key] = "" if result is None else str(result)
        environ.update(env)
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                              cwd=cwd, env=environ, capture_output=True, text=True, timeout=300)
        outputs = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            outputs[key] = value
        return done, outputs

    def uv_calls(self) -> list:
        if not self.uv_log.exists():
            return []
        return self.uv_log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def shell(tmp_path):
    return Shell(tmp_path)


# ===========================================================================
# Triggers, inputs and which jobs a run reaches
# ===========================================================================

class TestTriggers:
    def test_a_tag_pattern_of_its_own(self):
        assert TRIGGERS["push"] == {"tags": ["opencode-v*"]}

    def test_rehearse_is_the_default_mode(self):
        inputs = TRIGGERS["workflow_dispatch"]["inputs"]
        assert inputs["mode"]["type"] == "choice"
        assert inputs["mode"]["options"] == ["rehearse", "publish"]
        assert inputs["mode"]["default"] == "rehearse"
        assert inputs["path"]["options"] == ["normal", "bootstrap"]
        assert inputs["path"]["default"] == "normal"
        assert set(inputs) == {"mode", "tag", "as_tag", "path"}

    @pytest.mark.parametrize("event, jobs", [
        ("tag push", {"gate"}),
        ("bootstrap tag push", {"gate"}),
        ("rehearse", {"gate"}),
        ("rehearse bootstrap", {"gate"}),
        ("publish", {"verify", "publish"}),
    ])
    def test_each_event_reaches_exactly_its_jobs(self, event, jobs):
        assert reached(EVENTS[event]) == jobs

    def test_a_dispatch_without_a_mode_rehearses(self):
        """The default ``rehearse`` is what an unedited dispatch form sends."""
        default = TRIGGERS["workflow_dispatch"]["inputs"]["mode"]["default"]
        assert reached(_dispatch(mode=default)) == {"gate"}

    @pytest.mark.parametrize("event", ["tag push", "bootstrap tag push", "rehearse",
                                       "rehearse bootstrap"])
    def test_no_tag_push_or_rehearsal_holds_id_token_or_publishes(self, event):
        for job_id in reached(EVENTS[event]):
            assert "id-token" not in (JOBS[job_id].get("permissions") or {}), job_id
            assert "environment" not in JOBS[job_id], job_id
            for s in steps(job_id):
                assert not re.search(r"\bnpm\s+publish\b", code(s)), (job_id, s.get("name"))

    def test_no_job_condition_uses_a_status_function(self):
        """``always()`` on ``publish`` would run it after a failed ``verify``."""
        for job_id, job in JOBS.items():
            assert not re.search(r"\b(always|failure|cancelled|success)\s*\(", job.get("if", "")), \
                job_id


# ===========================================================================
# Permissions
# ===========================================================================

class TestPermissions:
    def test_the_workflow_level_block_grants_nothing(self):
        assert DOCUMENT["permissions"] == {}

    def test_each_job_declares_its_own(self):
        assert JOBS["gate"]["permissions"] == {"contents": "read"}
        assert JOBS["verify"]["permissions"] == {"contents": "read"}
        assert JOBS["publish"]["permissions"] == {"contents": "read", "id-token": "write"}

    def test_publish_is_the_only_holder_of_id_token(self):
        def holders(node, where=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "id-token":
                        yield where
                    yield from holders(value, f"{where}/{key}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    yield from holders(item, f"{where}[{i}]")

        assert list(holders(DOCUMENT)) == ["/jobs/publish/permissions"]
        uncommented = "\n".join(line for line in TEXT.splitlines()
                                if not line.lstrip().startswith("#"))
        assert uncommented.count("id-token") == 1

    def test_publish_needs_verify_and_declares_the_environment(self):
        assert JOBS["publish"]["needs"] == "verify"
        assert JOBS["publish"]["environment"] == "npm-publish"
        assert JOBS["publish"]["if"] == JOBS["verify"]["if"]


# ===========================================================================
# The rehearsal inputs, and the bootstrap tag
# ===========================================================================

GUARD = re.compile(r"\$\{\{\s*inputs\.mode == 'rehearse' && inputs\.(as_tag|path) \|\| ''\s*\}\}")


class TestRehearsalInputs:
    def test_as_tag_and_path_are_read_only_through_the_rehearse_guard(self):
        """Every read of ``as_tag`` or ``path`` is the guarded form, in the gate's
        route step and nowhere else; nothing reads ``github.event.inputs``."""
        assert "github.event.inputs" not in TEXT
        expressions = re.findall(r"\$\{\{.*?\}\}", TEXT)
        readers = [e for e in expressions if re.search(r"\binputs\.(as_tag|path)\b", e)]
        assert readers and all(GUARD.fullmatch(e) for e in readers), readers
        route = step("gate", "Which tag, and which path after row 7")
        route_values = [str(v) for v in route["env"].values()]
        assert sorted(e for e in readers) == sorted(v for v in route_values if GUARD.fullmatch(v))
        for job_id in ("verify", "publish"):
            assert not re.search(r"\binputs\.(as_tag|path)\b", yaml.safe_dump(JOBS[job_id]))

    @pytest.mark.parametrize("event", list(EVENTS))
    def test_the_guard_hides_them_outside_a_rehearsal(self, event):
        route = step("gate", "Which tag, and which path after row 7")
        context = EVENTS[event]
        values = {k: evaluate(str(v), context) for k, v in route["env"].items()
                  if GUARD.fullmatch(str(v))}
        if context["inputs"].get("mode") == "rehearse":
            assert values == {"AS_TAG": context["inputs"]["as_tag"],
                              "REHEARSE_PATH": context["inputs"]["path"]}
        else:
            assert values == {"AS_TAG": "", "REHEARSE_PATH": ""}

    def test_only_verify_and_publish_read_the_tag_input(self):
        assert not re.search(r"\binputs\.tag\b", yaml.safe_dump(JOBS["gate"]))
        for job_id in ("verify", "publish"):
            assert re.search(r"\binputs\.tag\b", yaml.safe_dump(JOBS[job_id]))

    def test_the_release_tag_pattern_is_the_record_schemas(self):
        """One shape of release tag everywhere: plain x.y.z. The workflow's
        pattern admitting a prerelease the release-gate record's schema (and
        so ``validate-record``) refuses would only defer the refusal."""
        schema = json.loads((PACKAGE_DIR / "release-gate-record.schema.json").read_text("utf-8"))
        assert WORKFLOW_ENV["RELEASE_TAG_PATTERN"] == schema["properties"]["tag"]["pattern"]
        for tag in ("opencode-v0.1.1-rc.1", "opencode-v1.0.0-beta", "opencode-v1.0"):
            assert not re.fullmatch(WORKFLOW_ENV["RELEASE_TAG_PATTERN"], tag), tag

    def test_the_bootstrap_tag_is_written_once(self):
        assert re.fullmatch(WORKFLOW_ENV["RELEASE_TAG_PATTERN"], BOOTSTRAP_TAG)
        assert TEXT.count(BOOTSTRAP_TAG) == 1
        readers = [(job_id, s.get("name")) for job_id, s in _all_steps()
                   if "BOOTSTRAP_TAG" in code(s)]
        assert readers == [("gate", "Which tag, and which path after row 7")]

    def test_no_shell_interpolates_an_expression(self):
        """Every input reaches a shell through ``env``, never spliced into the
        script, so a crafted tag or input cannot become shell code."""
        for job_id, s in _all_steps():
            assert "${{" not in (s.get("run") or ""), (job_id, s.get("name"))


@needs_shell
class TestRoute:
    """The gate's route step, run under each event."""

    ROUTE = ("gate", "Which tag, and which path after row 7")

    def _route(self, shell, context, **env):
        done, outputs = shell.run(step(*self.ROUTE), env=env, cwd=shell.root, context=context)
        return done, outputs

    @pytest.mark.parametrize("tag, path", [
        (BOOTSTRAP_TAG, "bootstrap"),
        ("opencode-v0.1.2", "normal"),
        ("opencode-v0.1.10", "normal"),
        ("opencode-v1.0.0", "normal"),
        ("opencode-v0.1.11", "normal"),
    ])
    def test_a_real_push_routes_only_the_bootstrap_tag_to_bootstrap(self, shell, tag, path):
        done, outputs = self._route(shell, _push(tag))
        assert done.returncode == 0, done.stderr
        assert outputs == {"tag": tag, "path": path, "rehearsal": "false"}

    @pytest.mark.parametrize("smuggled", ["bootstrap", "normal"])
    def test_a_push_ignores_any_path_value(self, shell, smuggled):
        """A push has no inputs; even values forced into the step's env do not
        move a non-bootstrap tag off row 9, or the bootstrap tag onto it."""
        done, outputs = self._route(shell, _push("opencode-v0.1.2"), REHEARSE_PATH=smuggled,
                                    AS_TAG=BOOTSTRAP_TAG)
        assert done.returncode == 0 and outputs["path"] == "normal"
        assert outputs["tag"] == "opencode-v0.1.2"
        done, outputs = self._route(shell, _push(BOOTSTRAP_TAG), REHEARSE_PATH="normal")
        assert done.returncode == 0 and outputs["path"] == "bootstrap"

    @pytest.mark.parametrize("path", ["normal", "bootstrap"])
    def test_a_rehearsal_takes_its_explicit_path_and_as_tag(self, shell, path):
        done, outputs = self._route(shell, _dispatch(mode="rehearse", as_tag="opencode-v0.1.2",
                                                     path=path))
        assert done.returncode == 0, done.stderr
        assert outputs == {"tag": "opencode-v0.1.2", "path": path, "rehearsal": "true"}
        assert "rehearsal true, on refs/heads/master" in (shell.root / "summary.md").read_text()

    @pytest.mark.parametrize("context, expected", [
        (_dispatch(mode="rehearse", path="normal"), "needs as_tag"),
        (_dispatch(mode="rehearse", as_tag="opencode-v0.1.2", path="sideways"),
         "neither normal nor bootstrap"),
        (_dispatch(mode="publish", tag="opencode-v0.1.2"), "not mode: publish"),
        (_dispatch(mode="rehearse", as_tag="v0.1.2", path="normal"), "is not an opencode-v"),
        (_dispatch(mode="rehearse", as_tag="opencode-v0.1.2\nx", path="normal"),
         "is not an opencode-v"),
        (_push("opencode-vfoo"), "is not an opencode-v"),
        # Plain x.y.z only, as the record schema and release_gate admit: a
        # prerelease tag is refused here, not routed on to be refused later.
        (_push(BOOTSTRAP_TAG + "-rc.1"), "is not an opencode-v"),
        (_push("opencode-v1.0.0-beta"), "is not an opencode-v"),
        ({"github": {"event_name": "push", "ref_type": "branch", "ref_name": "master"},
          "inputs": {}}, "this job gates tags"),
    ])
    def test_everything_else_refuses(self, shell, context, expected):
        done, outputs = self._route(shell, context)
        assert done.returncode != 0 and outputs == {}
        assert expected in done.stderr, done.stderr


# ===========================================================================
# The gate job: its order, and the path each run takes after row 7
# ===========================================================================

def _gate_runs(path: str, rehearsal: str) -> list:
    context = {**_push("x"), "steps": {"route": {"outputs": {"path": path,
                                                             "rehearsal": rehearsal}}}}
    return [s.get("name") or _uses(s) for s in steps("gate")
            if _truthy(evaluate(str(s.get("if", "true")), context))]


BOOTSTRAP_STEPS = [
    "Bootstrap rehearsal: serve the packed candidate from a localhost registry stub",
    "Bootstrap: a fresh probe root outside the checkout",
    "Bootstrap: wait until the registry serves the version",
    "Bootstrap: rebuild the candidate from the tag's commit (--frozen --no-dev)",
    "Bootstrap: the registry-install probe",
    "Bootstrap: validate the record (bootstrap profile)",
    "Keep the bootstrap record",
]
ROW9_NAME = "Version must be absent from the registry (row 9)"


class TestGate:
    def test_the_gate_runs_in_the_plans_order(self):
        names = [s.get("name") or _uses(s) for s in steps("gate")]
        order = ["actions/checkout", "Install the pinned npm",
                 "Which tag, and which path after row 7",
                 "Tag must equal package.json's version",
                 "Shared-version check", "Package gate (rows 1-6 and 5b)",
                 "Build, pack and sweep the candidate",
                 "The pinned OpenCode, from the container lockfile",
                 "Clean-install probe of the candidate tarball (row 7)", ROW9_NAME,
                 BOOTSTRAP_STEPS[0]]
        positions = [names.index(name) for name in order]
        assert positions == sorted(positions), names

    def test_a_normal_path_runs_row_9_and_no_bootstrap_step(self):
        for rehearsal in ("true", "false"):
            ran = _gate_runs("normal", rehearsal)
            assert ROW9_NAME in ran
            assert not set(BOOTSTRAP_STEPS) & set(ran), ran

    def test_the_bootstrap_path_runs_the_probe_in_place_of_row_9(self):
        ran = _gate_runs("bootstrap", "false")
        assert ROW9_NAME not in ran
        assert [n for n in ran if n in BOOTSTRAP_STEPS] == BOOTSTRAP_STEPS[1:]
        rehearsed = _gate_runs("bootstrap", "true")
        assert [n for n in rehearsed if n in BOOTSTRAP_STEPS] == BOOTSTRAP_STEPS
        assert ROW9_NAME not in rehearsed

    def test_only_row_9_and_the_bootstrap_steps_read_the_path(self):
        readers = {s.get("name") for s in steps("gate") if "outputs.path" in str(s.get("if", ""))}
        assert readers == {ROW9_NAME, *BOOTSTRAP_STEPS}

    def test_every_other_gate_step_runs_on_every_path(self):
        always = [n for n in _gate_runs("normal", "false") if n != ROW9_NAME]
        assert always == [n for n in _gate_runs("bootstrap", "false") if n not in BOOTSTRAP_STEPS]
        assert "Clean-install probe of the candidate tarball (row 7)" in always

    def test_the_stand_in_tag_is_a_rehearsals_alone_and_never_pushed(self):
        stand_in = step("gate", "Rehearsal: the dispatched commit stands in for the tag")
        assert stand_in["if"] == "steps.route.outputs.rehearsal == 'true'"
        assert 'git tag "${TAG}" "${head}"' in code(stand_in)
        for job_id, s in _all_steps():
            assert not re.search(r"\bgit\s+(push|commit)\b", code(s)), (job_id, s.get("name"))


class TestBootstrapBranch:
    def test_the_probe_is_the_one_registry_install_entry_point(self):
        probe = code(step("gate", "Bootstrap: the registry-install probe"))
        assert "packaging/opencode/registry_check.py" in probe
        for flag in ('--spec "${NAME}@${VERSION}"', '--registry "${REGISTRY}"', '--tag "${TAG}"',
                     "--environment ci-runner", '--out "${record}"'):
            assert flag in probe, flag
        # The stub's URL reaches the probe in a rehearsal, and only there.
        assert re.search(r'if \[ "\$\{REHEARSAL\}" = "true" \]; then '
                         r'rehearsal_flags=\(--rehearsal-registry "\$\{STUB_URL\}"\)', probe)
        assert probe.count("--rehearsal-registry") == 1
        assert 'record="${records}/$(date -u +%Y%m%dT%H%M%SZ)-opencode-bootstrap-check${suffix}.json"' in probe
        assert re.search(r'if \[ "\$\{REHEARSAL\}" = "true" \]; then suffix="-rehearsal"; fi', probe)
        # The probe makes its root inside the fresh, checked-empty probe root,
        # runs the gate's pinned binary, and names this run.
        assert '--tmp-parent "${PROBE_ROOT}"' in probe
        assert '--opencode "${HOST_BIN}"' in probe
        assert '--workflow-run-id "${GITHUB_RUN_ID}"' in probe

    def test_the_probe_root_is_fresh_and_outside_the_checkout(self):
        root = code(step("gate", "Bootstrap: a fresh probe root outside the checkout"))
        assert 'mktemp -d "${RUNNER_TEMP}/' in root
        assert "lies inside the checkout" in root and "is not empty" in root
        assert "empty_at_start" in root

    def test_the_visibility_poll_is_bounded_and_reads_the_explicit_registry(self):
        poll = step("gate", "Bootstrap: wait until the registry serves the version")
        assert isinstance(poll["timeout-minutes"], int) and poll["timeout-minutes"] <= 30
        assert "deadline=" in code(poll)
        assert re.search(r'npm view "\$\{spec\}" dist\.integrity --registry "\$\{source\}" '
                         r'--cache "\$\{cache\}"', code(poll))

    def test_the_rebuild_is_frozen_no_dev_from_the_tags_commit(self):
        rebuild = code(step("gate", BOOTSTRAP_STEPS[3]))
        assert "uv run --project skills/multi-llm --frozen --no-dev -- python " \
               "packaging/opencode/build.py" in rebuild
        assert "refs/tags/${TAG}^{commit}" in rebuild

    def test_the_record_is_validated_with_the_bootstrap_profile_and_uploaded(self):
        check = code(step("gate", "Bootstrap: validate the record (bootstrap profile)"))
        assert re.search(r"--frozen --no-dev -- python packaging/opencode/registry_record\.py"
                         r" \\\n\s+validate --profile bootstrap", check)
        assert '--integrity "${INTEGRITY}"' in check and '--digest "${DIGEST}"' in check
        upload = step("gate", "Keep the bootstrap record")
        assert _uses(upload) == "actions/upload-artifact"
        assert upload["with"]["path"].endswith("/opencode-bootstrap/")
        assert "github.run_attempt" in upload["with"]["name"]


# ===========================================================================
# verify and publish
# ===========================================================================

DISPATCH_NAME = "Dispatched from the default branch, for an opencode tag"


class TestVerify:
    def test_the_default_branch_refusal_comes_first(self):
        assert steps("verify")[0]["name"] == DISPATCH_NAME
        assert steps("publish")[0] == steps("verify")[0]

    def test_it_reruns_the_gate_rows(self):
        names = [s.get("name") for s in steps("verify")]
        for name in ("Shared-version check", "Package gate (rows 1-6 and 5b)",
                     "Build, pack and sweep the candidate",
                     "The pinned OpenCode, from the container lockfile",
                     "Clean-install probe of the candidate tarball (row 7)", ROW9_NAME):
            assert name in names, name
            assert step("verify", name)["run"] == step("gate", name)["run"], name

    def test_its_outputs_are_set_after_every_check(self):
        outputs = JOBS["verify"]["outputs"]
        assert outputs == {"sha": "${{ steps.handoff.outputs.sha }}",
                           "digest": "${{ steps.handoff.outputs.digest }}"}
        names = [s.get("name") for s in steps("verify")]
        handoff = next(i for i, s in enumerate(steps("verify")) if s.get("id") == "handoff")
        for name in ("The release-gate record validates against this rebuild",
                     "Clean-install probe of the candidate tarball (row 7)", ROW9_NAME):
            assert names.index(name) < handoff, name

    def test_the_record_check_is_the_one_validator(self):
        check = code(step("verify", "The release-gate record validates against this rebuild"))
        assert re.search(r"packaging/opencode/release_gate\.py \\\n\s+validate-record", check)
        for flag in ('--tag "${TAG}"', '--sha "${SHA}"', '--version "${VERSION}"',
                     '--digest "${DIGEST}"', '--host-version "${host}"',
                     '--registry "${REGISTRY}"', '--records-dir "${RECORDS}"'):
            assert flag in check, flag
        assert "--frozen --no-dev" in check
        for private in ("glob", "json.load", "<<'PY'", "jq "):
            assert private not in check, private

    def test_the_records_come_from_the_dispatched_commit(self):
        read = code(step("verify", "Read the release-gate records committed on the default branch"))
        assert 'git archive --format=tar "${GITHUB_SHA}" -- "${records}"' in read
        assert "origin/" not in read


class TestPublish:
    ALLOWED_ACTIONS = {"actions/checkout", "astral-sh/setup-uv", "actions/setup-node"}

    def _names(self):
        return [s.get("name") or _uses(s) for s in steps("publish")]

    def test_it_runs_nothing_but_the_builder_npm_and_row_9(self):
        """No OpenCode install, probe, test, stub or plugin step; no other action;
        every `uv run` is the builder's locked runtime set."""
        forbidden = re.compile(
            r"bump\.py|probe\.py|registry_check\.py|registry_stub|pytest|\bopencode-package\b|"
            r"opencode-package-check|e2e|release_gate\.py|\bopencode\s+(run|debug|serve|--version)|"
            r"\bnpm\s+ci\b|\bnpx\b|upload-artifact")
        for s in steps("publish"):
            if s.get("uses"):
                assert _uses(s) in self.ALLOWED_ACTIONS, s["uses"]
            text = code(s)
            assert not forbidden.search(text), (s.get("name"), forbidden.search(text))
            for line in re.findall(r"\buv run\b[^\n]*", text):
                assert "--frozen --no-dev" in line, line
            for target in re.findall(r"\bmake\s+(?:-s\s+)?([\w-]+)", text):
                assert target in {"opencode-registry-absent", "opencode-print-registry"}, target
            installs = re.findall(r"\bnpm\s+install\b[^\n]*", text)
            assert installs in ([], ['npm install -g --ignore-scripts "npm@${NPM_VERSION}"']), installs

    def test_the_order_binds_the_digest_before_anything_else(self):
        names = self._names()
        positions = [names.index(n) for n in (
            DISPATCH_NAME, "actions/checkout", "The tag still names verify's commit",
            "Rebuild the candidate from this commit (--frozen --no-dev)",
            "The rebuilt digest is verify's", ROW9_NAME,
            "Pack once, with --ignore-scripts, and bind the tarball to the digest",
            "Publish the packed tarball, with provenance")]
        assert positions == sorted(positions), names
        rebuild = names.index("Rebuild the candidate from this commit (--frozen --no-dev)")
        assert names[rebuild + 1] == "The rebuilt digest is verify's"
        assert step("publish", "Rebuild the candidate from this commit (--frozen --no-dev)") == \
            step("verify", "Rebuild the candidate from this commit (--frozen --no-dev)")

    def test_the_publish_command_names_a_tarball_never_the_directory(self):
        commands = [(s.get("name"), line) for job_id, s in _all_steps()
                    for line in code(s).splitlines() if re.search(r"\bnpm\s+publish\b", line)]
        assert len(commands) == 1 and commands[0][0] == "Publish the packed tarball, with provenance"
        line = commands[0][1].strip()
        assert re.match(r'npm publish "\./\$\{NAME\}-\$\{VERSION\}\.tgz" ', line), line
        for flag in ("--provenance", "--ignore-scripts", "--access public",
                     '--registry "${REGISTRY}"'):
            assert flag in line, flag
        pack = code(step("publish", "Pack once, with --ignore-scripts, and bind the tarball to the digest"))
        assert "npm pack --ignore-scripts --pack-destination" in pack
        assert re.search(r"build\.py \\\n\s+--scan-tarball", pack) and "tarball_facts" in pack

    def test_the_old_directory_publish_is_gone(self):
        assert "publishes the DIRECTORY" not in TEXT
        assert "cd packaging/opencode && npm publish" not in TEXT
        assert "THE GATE RAN IN THIS RUN, ON THIS SHA, BOUND BY THE DIGEST" in TEXT
        assert "in this run, on this SHA, bound by the digest" in TEXT
        assert "THE GATE RUNS IN THIS JOB" not in TEXT


# ===========================================================================
# Row 9, the npm setup, the OpenCode pin
# ===========================================================================

class TestRow9AndNpm:
    def test_every_row_9_site_is_the_entry_point_and_the_same_step(self):
        sites = [_row9(job_id) for job_id in ROW9_JOBS]
        assert {s["name"] for s in sites} == {ROW9_NAME}
        assert len({s["run"] for s in sites}) == 1
        assert "make opencode-registry-absent" in sites[0]["run"]
        for job_id, s in _all_steps():
            assert not ABSENCE_QUERY.search(code(s)), (job_id, s.get("name"))
            assert not CODE_PARSE.search(code(s)), (job_id, s.get("name"))

    def test_every_row_9_job_has_the_same_npm_setup(self):
        setups = []
        for job_id in ROW9_JOBS:
            assert index(job_id, lambda s: _uses(s) == "actions/setup-node") < \
                index(job_id, lambda s: s is _row9(job_id)), job_id
            setups.append(_npm_setup(job_id))
        assert all(pair == setups[0] for pair in setups)


class TestTheRow9Canary:
    """Plan step 0's `.npmrc` case, recorded on every gate run: under the job's
    exact npm setup, a version never published must answer ``E404``, or row 9
    and the bootstrap probe would both fail closed on the real tag."""

    def test_it_runs_on_every_path_rehearsal_or_not(self):
        canary = step("gate", CANARY_NAME)
        assert "if" not in canary
        for path in ("normal", "bootstrap"):
            for rehearsal in ("true", "false"):
                assert CANARY_NAME in _gate_runs(path, rehearsal), (path, rehearsal)

    def test_it_runs_before_row_9_and_the_bootstrap_probe(self):
        at = index("gate", lambda s: s.get("name") == CANARY_NAME)
        assert at < index("gate", lambda s: s is _row9("gate"))
        assert at < index("gate", lambda s: s.get("name") == "Bootstrap: the registry-install probe")

    def test_its_npm_setup_is_the_row_9_jobs(self):
        """The same setup-node and pinned-npm steps as every row-9 job, run
        before it in its own job, and nothing of its own that npm would read:
        no env but the package's name, no npm flag or config of its own, row
        9's entry point for the question."""
        assert _npm_setup("gate") == _npm_setup("verify") == _npm_setup("publish")
        at = index("gate", lambda s: s.get("name") == CANARY_NAME)
        assert index("gate", lambda s: _uses(s) == "actions/setup-node") < at
        assert index("gate", lambda s: s.get("name") == "Install the pinned npm") < at
        canary = step("gate", CANARY_NAME)
        assert set(canary.get("env") or {}) == {"NAME"}
        assert canary["env"]["NAME"] == "${{ steps.facts.outputs.name }}"
        body = code(canary)
        assert not re.search(r"(?i)npm_config|NODE_AUTH_TOKEN|\.npmrc|--userconfig|--registry",
                             body)
        assert not re.search(r"(?m)(?:^|[;&|(]\s*)npm\s", body), "an npm call of its own"
        assert 'make opencode-registry-absent PACKAGE_JSON="${canary}/package.json" OUT="${out}"' \
            in body
        assert _row9("gate")["run"].startswith("make opencode-registry-absent ")

    def test_it_asks_about_version_0_0_0_and_keeps_the_answer(self):
        body = code(step("gate", CANARY_NAME))
        assert '"version": "0.0.0"' in body
        assert 'out="${RUNNER_TEMP}/opencode-evidence/row9-canary.json"' in body
        assert 'tee -a "${GITHUB_STEP_SUMMARY}"' in body
        assert isinstance(step("gate", CANARY_NAME)["timeout-minutes"], int)

    @needs_shell
    @pytest.mark.parametrize("answer, status, passes", [
        ("E404", 0, True),
        ("E401", 1, False),        # a placeholder token's answer
        ("ECONNREFUSED", 1, False),
        ("E401", 0, False),        # an exit 0 that disagrees with the code still refuses
        (None, 1, False),          # no evidence at all
    ])
    def test_its_own_shell_requires_e404(self, shell, answer, status, passes):
        """The step's own bash, with ``make`` a stand-in that writes the
        evidence the entry point would and exits as it would."""
        seen = shell.root / "make-args.json"
        (shell.bin / "make").write_text(
            "#!/bin/sh\n"
            f'exec "{sys.executable}" - "$@" <<\'PY\'\n'
            "import json, sys\n"
            "args = dict(a.split('=', 1) for a in sys.argv[2:])\n"
            f"json.dump({{'argv': sys.argv[1:], 'manifest': json.load(open(args['PACKAGE_JSON']))}}, "
            f"open({str(seen)!r}, 'w'))\n"
            + (f"json.dump({{'row': '9', 'code': {answer!r}}}, open(args['OUT'], 'w'))\n"
               if answer else "")
            + f"sys.exit({status})\n"
            "PY\n", encoding="utf-8")
        (shell.bin / "make").chmod(0o755)
        done, _ = shell.run(step("gate", CANARY_NAME), env={"NAME": "opencode-multi-llm"},
                            cwd=shell.root)
        made = json.loads(seen.read_text(encoding="utf-8"))
        assert made["argv"][0] == "opencode-registry-absent"
        assert made["manifest"] == {"name": "opencode-multi-llm", "version": "0.0.0"}
        summary = (shell.root / "summary.md").read_text(encoding="utf-8")
        assert f"answered {answer or '<no evidence>'}" in summary
        if passes:
            assert done.returncode == 0, done.stderr
        else:
            assert done.returncode != 0
            assert "not E404" in done.stderr

    def test_no_job_writes_an_auth_npmrc(self):
        """No `registry-url`, so setup-node writes no `.npmrc` and exports no
        placeholder `NODE_AUTH_TOKEN` for row 9 to send (the plan's `.npmrc`
        case); and nothing else sets a token either."""
        for job_id, s in _all_steps():
            if _uses(s) == "actions/setup-node":
                assert set(s["with"]) == {"node-version"}, (job_id, s["with"])
        uncommented = "\n".join(line for line in TEXT.splitlines()
                                if not line.lstrip().startswith("#"))
        for needle in ("NODE_AUTH_TOKEN", "_authToken", ".npmrc", "registry-url", "NPM_TOKEN",
                       "secrets."):
            assert needle not in uncommented, needle

    def test_npm_and_node_are_pinned_at_or_above_the_trusted_publishing_floor(self):
        def key(version):
            return tuple(int(part) for part in version.split("."))

        assert re.fullmatch(r"\d+\.\d+\.\d+", WORKFLOW_ENV["NPM_VERSION"])
        assert key(WORKFLOW_ENV["NPM_VERSION"]) >= (11, 5, 1)
        assert key(WORKFLOW_ENV["NODE_FLOOR"]) >= (22, 14, 0)
        assert WORKFLOW_ENV["NODE_FLOOR"].split(".")[0] == WORKFLOW_ENV["NODE_VERSION"]
        assert "latest" not in WORKFLOW_ENV["NPM_VERSION"]
        npm = code(step("gate", "Install the pinned npm"))
        assert 'npm install -g --ignore-scripts "npm@${NPM_VERSION}"' in npm
        assert 'sort -V -C' in npm

    def test_the_install_step_calls_the_entry_point_and_names_no_version(self):
        from tests.test_opencode_release_gate import gate

        for job_id in ("gate", "verify"):
            install = step(job_id, "The pinned OpenCode, from the container lockfile")
            assert "packaging/opencode/bump.py install --prefix" in code(install)
        installs = [(job_id, s.get("name")) for job_id, s in _all_steps()
                    if "bump.py install" in code(s)]
        assert [j for j, _ in installs] == ["gate", "verify"]
        assert "opencode-ai" not in TEXT
        assert gate.host_pin() not in TEXT


# ===========================================================================
# Checkouts, pins and bounds
# ===========================================================================

class TestHygiene:
    def test_every_checkout_has_full_history_and_no_persisted_token(self):
        checkouts = [(job_id, s) for job_id, s in _all_steps() if _uses(s) == "actions/checkout"]
        assert [job_id for job_id, _ in checkouts] == ["gate", "verify", "publish"]
        for job_id, s in checkouts:
            assert s["with"]["fetch-depth"] == 0, job_id
            assert s["with"]["persist-credentials"] is False, job_id
        assert step_ref("verify") == "refs/tags/${{ inputs.tag }}"
        assert step_ref("publish") == "${{ needs.verify.outputs.sha }}"

    def test_actions_are_pinned_by_sha_like_the_tests_workflow(self):
        pinned = {}
        for line in TESTS_WORKFLOW.read_text(encoding="utf-8").splitlines():
            match = re.search(r"uses:\s*([\w./-]+)@([0-9a-f]{40})", line)
            if match:
                pinned[match.group(1)] = match.group(2)
        uses = re.findall(r"uses:\s*(\S+)(.*)", TEXT)
        assert uses
        for ref, rest in uses:
            action, _, sha = ref.partition("@")
            assert re.fullmatch(r"[0-9a-f]{40}", sha), ref
            assert re.fullmatch(r"\s*# v\d+\.\d+\.\d+", rest), f"{ref}: no `# vX.Y.Z` label"
            if action in pinned:
                assert sha == pinned[action], action

    def test_the_uv_cache_is_off_everywhere(self):
        for job_id, s in _all_steps():
            if _uses(s) == "astral-sh/setup-uv":
                assert s["with"] == {"enable-cache": False}, job_id

    def test_every_job_and_every_long_step_is_bounded(self):
        for job_id, job in JOBS.items():
            assert isinstance(job["timeout-minutes"], int), job_id
        for job_id, s in _all_steps():
            if re.search(r"make opencode-package-check-ci|probe\.py row7|registry_check\.py|"
                         r"bump\.py install|make opencode-registry-absent|build\.py$",
                         code(s), re.M):
                assert isinstance(s.get("timeout-minutes"), int), (job_id, s.get("name"))
        assert JOBS["publish"]["concurrency"] == {"group": "opencode-npm-publish",
                                                  "cancel-in-progress": False}


def step_ref(job_id: str) -> str:
    return next(s for s in steps(job_id) if _uses(s) == "actions/checkout")["with"]["ref"]


# ===========================================================================
# The jobs' own shell, against fixture repositories and records
# ===========================================================================

def _repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "master")
    return repo


def _commit(repo: Path, files: dict, message: str) -> str:
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content),
                        encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@needs_shell
class TestDispatchRefusal:
    @pytest.mark.parametrize("ref, tag, ok", [
        ("refs/heads/master", "opencode-v0.1.2", True),
        ("refs/heads/feature", "opencode-v0.1.2", False),
        ("refs/tags/opencode-v0.1.2", "opencode-v0.1.2", False),
        ("refs/heads/master-copy", "opencode-v0.1.2", False),
        ("refs/heads/master", "", False),
        ("refs/heads/master", "codex-v0.1.0", False),
        ("refs/heads/master", "opencode-v0.1.2; rm -rf /", False),
    ])
    def test_publish_refuses_a_non_default_ref_or_a_foreign_tag(self, shell, ref, tag, ok):
        for job_id in ("verify", "publish"):
            done, _ = shell.run(steps(job_id)[0], cwd=shell.root, env={
                "GITHUB_REF": ref, "DEFAULT_BRANCH": "master", "TAG": tag})
            assert (done.returncode == 0) is ok, (job_id, done.stderr)

    def test_an_unknown_default_branch_refuses(self, shell):
        done, _ = shell.run(steps("verify")[0], cwd=shell.root, env={
            "GITHUB_REF": "refs/heads/", "DEFAULT_BRANCH": "", "TAG": "opencode-v0.1.2"})
        assert done.returncode != 0 and "REFUSING: dispatched from" in done.stderr


@needs_shell
class TestTagResolution:
    @pytest.fixture
    def history(self, tmp_path):
        repo = _repo(tmp_path)
        tagged = _commit(repo, {"packaging/opencode/package.json": {"version": "0.1.2"}}, "candidate")
        _git(repo, "tag", "-a", "opencode-v0.1.2", "-m", "OpenCode plugin 0.1.2")
        record = _commit(repo, {f"{RECORDS}/README": "records"}, "record")
        _git(repo, "checkout", "-q", "--orphan", "unrelated")
        unrelated = _commit(repo, {"x": "y"}, "unrelated")
        _git(repo, "checkout", "-q", "--detach", tagged)
        return repo, tagged, record, unrelated

    def _resolve(self, shell, repo, **env):
        return shell.run(step("verify", "Resolve the tag to its commit"), cwd=repo, env={
            "TAG": "opencode-v0.1.2", "VERSION": "0.1.2", **env})

    def test_the_tags_commit_on_the_default_branch_resolves(self, shell, history):
        repo, tagged, record, _ = history
        done, outputs = self._resolve(shell, repo, GITHUB_SHA=record)
        assert done.returncode == 0, done.stderr
        assert outputs == {"sha": tagged}

    def test_a_commit_off_the_default_branch_refuses(self, shell, history):
        repo, _, _, unrelated = history
        done, _ = self._resolve(shell, repo, GITHUB_SHA=unrelated)
        assert done.returncode != 0 and "is not on the dispatched default-branch commit" in done.stderr

    def test_a_tag_naming_another_version_refuses(self, shell, history):
        repo, _, record, _ = history
        done, _ = self._resolve(shell, repo, GITHUB_SHA=record, VERSION="0.1.3")
        assert done.returncode != 0 and "its package.json says 0.1.3" in done.stderr

    def test_a_checkout_that_is_not_the_tag_refuses(self, shell, history):
        repo, _, record, _ = history
        _git(repo, "checkout", "-q", "--detach", record)
        done, _ = self._resolve(shell, repo, GITHUB_SHA=record)
        assert done.returncode != 0 and "the checkout is not" in done.stderr

    def test_publish_refuses_a_tag_that_moved_after_verify(self, shell, history):
        repo, tagged, record, _ = history
        moved = step("publish", "The tag still names verify's commit")
        env = {"TAG": "opencode-v0.1.2", "SHA": tagged}
        done, _ = shell.run(moved, cwd=repo, env=env)
        assert done.returncode == 0, done.stderr
        _git(repo, "tag", "-f", "-a", "opencode-v0.1.2", "-m", "moved", record)
        done, _ = shell.run(moved, cwd=repo, env=env)
        assert done.returncode != 0 and f"now names {record}" in done.stderr
        done, _ = shell.run(moved, cwd=repo, env={**env, "SHA": ""})
        assert done.returncode != 0 and "not a commit" in done.stderr


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A release-gate record exported by a full passing gate session, through
    the gate's own writers (``GateRun``), for the module."""
    from tests.test_opencode_release_gate import GateRun, pkg_build

    built = tmp_path_factory.mktemp("exported-build") / "pkg"
    pkg_build.build(destination=built, allow_dirty=True)
    root = tmp_path_factory.mktemp("exported")
    with pytest.MonkeyPatch.context() as monkeypatch:
        run = GateRun(root, monkeypatch, built)
        run.full()
        run.stamp()
        path = run.export()
    return json.loads(path.read_text("utf-8"))


def _named(record: dict, stamp: str = "20991231T000000Z", **changes) -> tuple:
    data = copy.deepcopy(record)
    data["exported_at"] = stamp
    data.update(changes)
    return f"{RECORDS}/{stamp}-opencode-release-gate.json", data


@needs_shell
class TestVerifyRecordCheck:
    """verify's two record steps, run as the job runs them: the committed
    records read out of the dispatched commit with ``git archive``, then
    ``release_gate.py validate-record`` over them through the ``uv`` shim, from
    this checkout. The validator is the one the suite runs; nothing is retyped."""

    READ = "Read the release-gate records committed on the default branch"
    CHECK = "The release-gate record validates against this rebuild"

    def _check(self, shell, tmp_path, record, committed: dict, uncommitted: dict = (), **env):
        repo = _repo(tmp_path)
        dispatched = _commit(repo, {f"{RECORDS}/README": "records", **committed}, "records")
        for relative, content in dict(uncommitted).items():
            (repo / relative).write_text(json.dumps(content), encoding="utf-8")
        done, outputs = shell.run(step("verify", self.READ), cwd=repo,
                                  env={"GITHUB_SHA": dispatched})
        assert done.returncode == 0, done.stderr
        from tests.test_opencode_release_gate import gate

        return shell.run(step("verify", self.CHECK), cwd=REPO_ROOT, env={
            "TAG": record["tag"], "SHA": record["sha"], "VERSION": gate.package_version(),
            "DIGEST": record["candidate_digest"], "REGISTRY": gate.single_registry(),
            "RECORDS": outputs["dir"], **env})

    def test_a_genuine_committed_record_validates(self, shell, tmp_path, exported):
        name, data = _named(exported)
        done, outputs = self._check(shell, tmp_path, exported, {name: data})
        assert done.returncode == 0, done.stderr
        assert "validates" in done.stdout and "ran locally" in done.stdout
        assert outputs == {"digest": exported["candidate_digest"]}
        validator = [c for c in shell.uv_calls() if "validate-record" in c]
        assert validator and all("--frozen --no-dev" in c for c in validator)

    def test_only_committed_records_count(self, shell, tmp_path, exported):
        name, data = _named(exported)
        done, _ = self._check(shell, tmp_path, exported, {}, uncommitted={name: data})
        assert done.returncode != 0 and "no committed release-gate record names" in done.stderr

    def test_a_dispatched_commit_without_the_records_directory_refuses(self, shell, tmp_path):
        repo = _repo(tmp_path)
        dispatched = _commit(repo, {"README": "no records"}, "bare")
        done, _ = shell.run(step("verify", self.READ), cwd=repo, env={"GITHUB_SHA": dispatched})
        assert done.returncode != 0 and "has no skills/multi-llm/tests/container/records/" \
            in done.stderr

    def test_a_rehearsal_record_refuses(self, shell, tmp_path, exported):
        name, data = _named(exported, rehearsal=True)
        done, _ = self._check(shell, tmp_path, exported, {name: data})
        assert done.returncode != 0 and "never release evidence" in done.stderr

    @pytest.mark.parametrize("env, field", [
        ({"SHA": "0" * 40}, "sha: "),
        ({"DIGEST": "1" * 64}, "candidate_digest: "),
        ({"VERSION": "9.9.9"}, "version: "),
        ({"REGISTRY": "https://registry.example.com"}, "registry: --registry"),
        ({"TAG": "opencode-v9.9.9"}, "no committed release-gate record names opencode-v9.9.9"),
    ])
    def test_a_record_that_does_not_bind_this_rebuild_refuses(self, shell, tmp_path, exported,
                                                               env, field):
        name, data = _named(exported)
        done, outputs = self._check(shell, tmp_path, exported, {name: data}, **env)
        assert done.returncode != 0 and field in done.stderr, done.stderr
        assert outputs == {}

    def test_an_older_valid_record_is_no_fallback_for_an_invalid_newest(self, shell, tmp_path,
                                                                         exported):
        older, good = _named(exported, "20990101T000000Z")
        newest, bad = _named(exported, "20991231T000000Z")
        bad["posture"] = {**bad["posture"], "pass": False}
        done, _ = self._check(shell, tmp_path, exported, {older: good, newest: bad})
        assert done.returncode != 0
        assert "posture.pass" in done.stderr and "never count as a fallback" in done.stderr

    def test_a_missing_or_skipped_row_8_scenario_refuses(self, shell, tmp_path, exported):
        name, data = _named(exported)
        data["scenarios"] = data["scenarios"][1:]
        done, _ = self._check(shell, tmp_path, exported, {name: data})
        assert done.returncode != 0 and "is missing" in done.stderr


@needs_shell
class TestPublishDigest:
    COMPARE = "The rebuilt digest is verify's"

    def _compare(self, shell, tmp_path, rebuilt: str, verified: str):
        tree = tmp_path / "checkout" / "packaging" / "opencode" / "dist"
        tree.mkdir(parents=True)
        (tree / "tree.sha256").write_text(rebuilt + "\n", encoding="utf-8")
        return shell.run(step("publish", self.COMPARE), cwd=tmp_path / "checkout",
                         env={"VERIFIED": verified})

    def test_verifys_digest_passes(self, shell, tmp_path):
        done, _ = self._compare(shell, tmp_path, "a" * 64, "a" * 64)
        assert done.returncode == 0, done.stderr

    def test_a_rebuild_that_differs_from_verify_refuses_before_publishing(self, shell, tmp_path):
        done, _ = self._compare(shell, tmp_path, "b" * 64, "a" * 64)
        assert done.returncode != 0 and "Nothing was published" in done.stderr

    @pytest.mark.parametrize("verified", ["", "a" * 63, "A" * 64])
    def test_a_handover_that_is_not_a_digest_refuses(self, shell, tmp_path, verified):
        done, _ = self._compare(shell, tmp_path, "a" * 64, verified)
        assert done.returncode != 0 and "not a digest" in done.stderr


@needs_shell
class TestFacts:
    def test_the_facts_step_reads_the_owners(self, shell):
        if shutil.which("make") is None:
            pytest.skip("needs make: the single registry is the Makefile's")
        done, outputs = shell.run(step("gate", "The single registry, and the package's name and version"),
                                  cwd=REPO_ROOT, env={})
        assert done.returncode == 0, done.stderr
        manifest = json.loads((PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
        assert outputs == {"registry": SINGLE, "name": manifest["name"],
                           "version": manifest["version"]}
        for job_id in ("verify", "publish"):
            assert step(job_id, "The single registry, and the package's name and version") == \
                step("gate", "The single registry, and the package's name and version")


@needs_shell
class TestBootstrapValidation:
    """The bootstrap branch's validation step, run against fixture records in a
    fixture repository whose annotated tag decides the release kind, with the
    one registry-record validator copied beside it."""

    VALIDATE = "Bootstrap: validate the record (bootstrap profile)"

    @pytest.fixture
    def world(self, tmp_path):
        from tests.test_opencode_registry_record import _make_repo

        fixture = _make_repo(tmp_path / "fixture", "0.1.1")
        repo = fixture["repo"]
        for name in ("registry_record.py", "registry.py", "registry-record.schema.json"):
            shutil.copy2(PACKAGE_DIR / name, repo / "packaging" / "opencode" / name)
        shutil.copy2(REPO_ROOT / "Makefile", repo / "Makefile")
        probe_root = tmp_path / "probe-root"
        probe_root.mkdir()
        return fixture, probe_root

    def _record(self, fixture, probe_root, **changes):
        from tests.test_opencode_registry_record import make

        record = make(fixture, "bootstrap")
        record["payload_location"] = str(
            probe_root / "home" / ".cache" / "opencode" / "packages" / "node_modules"
            / "opencode-multi-llm")
        record.update(changes)
        # Named as the workflow names it: a rehearsal's record says so.
        suffix = "-rehearsal" if record.get("rehearsal") is True else ""
        path = probe_root.parent / "records" / f"20261026T100000Z-opencode-bootstrap-check{suffix}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def _validate(self, shell, world, path, rehearsal: bool, **env):
        from tests.test_opencode_registry_record import DIGEST, INTEGRITY

        fixture, probe_root = world
        return shell.run(step("gate", self.VALIDATE), cwd=fixture["repo"], env={
            "TAG": fixture["tag"], "RECORD": str(path), "INTEGRITY": INTEGRITY,
            "DIGEST": DIGEST, "REHEARSAL": "true" if rehearsal else "false",
            "PROBE_ROOT": str(probe_root), **env})

    def test_a_genuine_bootstrap_record_validates(self, shell, world):
        done, _ = self._validate(shell, world, self._record(*world), False)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "valid bootstrap record" in done.stdout
        calls = [c for c in shell.uv_calls() if "registry_record.py" in c]
        assert calls and all("--frozen --no-dev" in c for c in calls)

    def test_a_record_that_fails_the_profile_fails_the_step(self, shell, world):
        done, _ = self._validate(shell, world, self._record(*world), False,
                                 INTEGRITY="sha512-other")
        assert done.returncode != 0 and "is not the gated" in done.stdout

    def test_a_rehearsal_record_must_be_refused_as_rehearsal_evidence(self, shell, world):
        fixture, probe_root = world
        path = self._record(fixture, probe_root, rehearsal=True,
                            installed_registry="http://127.0.0.1:4873/")
        done, _ = self._validate(shell, world, path, True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "rehearsal evidence" in done.stdout
        assert "refused as release evidence, as it must be" in done.stdout

    def test_a_validator_that_accepts_a_rehearsal_fails_the_step(self, shell, world):
        done, _ = self._validate(shell, world, self._record(*world), True)
        assert done.returncode != 0
        assert "accepted a rehearsal record as release evidence" in done.stderr

    def test_a_rehearsal_refused_for_another_reason_fails_the_step(self, shell, world):
        """Refused, but not because it is a rehearsal: the stub's record must
        fail as one, or the regression case proves nothing."""
        fixture, probe_root = world
        path = self._record(fixture, probe_root)
        done, _ = self._validate(shell, world, path, True, INTEGRITY="sha512-other")
        assert done.returncode != 0 and "not as rehearsal evidence" in done.stderr

    @pytest.mark.parametrize("location", ["/opt/elsewhere/node_modules/opencode-multi-llm", None])
    def test_a_payload_outside_the_probe_root_refuses(self, shell, world, location):
        fixture, probe_root = world
        path = self._record(fixture, probe_root, payload_location=location)
        done, _ = self._validate(shell, world, path, False)
        assert done.returncode != 0
        assert ("outside the probe root" if location else "names no payload_location") in done.stderr


@needs_shell
class TestProbeRoot:
    ROOT = "Bootstrap: a fresh probe root outside the checkout"

    def test_a_fresh_root_under_the_runner_temp_is_recorded(self, shell, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        done, outputs = shell.run(step("gate", self.ROOT), cwd=workspace, env={})
        assert done.returncode == 0, done.stderr
        root = Path(outputs["root"])
        assert root.parent == shell.runner_temp and root.is_dir()
        evidence = json.loads((shell.runner_temp / "opencode-evidence"
                               / "bootstrap-probe-root.json").read_text(encoding="utf-8"))
        assert evidence == {"probe_root": os.path.realpath(root), "outside_checkout": True,
                            "empty_at_start": True}

    def test_a_root_inside_the_checkout_refuses(self, shell, tmp_path):
        done, outputs = shell.run(step("gate", self.ROOT), cwd=shell.root,
                                  env={"GITHUB_WORKSPACE": str(shell.root)})
        assert done.returncode != 0 and outputs == {}
        assert "lies inside the checkout" in done.stderr
