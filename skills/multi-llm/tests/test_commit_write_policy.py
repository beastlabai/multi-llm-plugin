"""The commit write policy's write legs (plans/review-commits.md §2.7, task T6b).

After a commit review, apply-code-fixes runs under an apply guard
(``<pinned record>.guard``, written by ``utils/commit_apply.py``): it saves the
content every fix file must still have (a ``head`` blob id or a ``batch``
sha256) and the *protected set*, every path that differed from the reviewed
``head`` when the guard ran. These tests pin the two legs that enforce it
batch by batch:

* **The executor** (``WriteBatchScheduler``, driven here by an in-process fake
  runner, never a model): ``_dispatch_one`` re-checks every fix file a batch
  declares before its snapshot, skips the batch (``[COMMIT_FIXES_STALE]``) and
  goes on, holds back the groups that named no failing file, re-pins a
  completed batch's files, and carries the protected set as an exact-path deny
  layer of ``WritePolicy`` — so a batch that writes a protected file fails, the
  file is restored and the loop stops.
* **The native leg**: ``util write-prompt`` writes no prompt for a batch that
  fails the re-check, and ``util batch-done --status completed`` refuses a batch
  that wrote a protected path with ``[COMMIT_PROTECTED_WRITE]``.

``utils/commit_apply.py`` is the T6a agent's module and is not in this tree, so
the tests install a small fake with the contracted signatures
(``CONTRACTS.md``, "The apply guard record") and straightforward semantics.
Without a guard a tripwire stands in its place: touching it fails the test,
which is how "byte-identical without a guard" is pinned.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import multi_llm  # noqa: E402
import utils  # noqa: E402
from tests.test_write_executor import FakeState, env  # noqa: E402,F401
from utils import git_utils  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.commit_range import pinned_record  # noqa: E402
from utils.metrics import state_file_lock  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.write_batch_scheduler import (  # noqa: E402
    batch_fixes_for,
    commit_group_files,
    dispatched_prompt_path,
    load_orchestrator_batches,
)
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)
from utils.write_safety import (  # noqa: E402
    PolicyHit,
    PolicyViolation,
    WritePolicy,
    policy_allow_clause,
    policy_outside_note,
    policy_way_forward,
)

STALE = "[COMMIT_FIXES_STALE]"
PROTECTED = "[COMMIT_PROTECTED_WRITE]"


# ---------------------------------------------------------------------------
# git helpers (test-only: the runtime modules spell git through git_argv)
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(root: Path, *paths: str) -> str:
    _git(root, "add", "--", *paths)
    _git(root, "commit", "-q", "-m", "reviewed")
    return _git(root, "rev-parse", "HEAD")


def _head_blob(root: Path, head: str, path: str) -> str:
    return _git(root, "rev-parse", f"{head}:{path}")


def _disk_blob(root: Path, path: str) -> Optional[str]:
    if not (root / path).exists():
        return None
    return _git(root, "hash-object", "--", path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The fake utils.commit_apply (CONTRACTS.md signatures, plain semantics)
# ---------------------------------------------------------------------------

def _blob_now(cwd: str, path: str) -> Optional[str]:
    if not os.path.lexists(os.path.join(cwd, path)):
        return None
    out, _err, rc = git_utils._run_git("hash-object", "--", path, cwd=cwd, check=False)
    return out.strip() if rc == 0 else None


def _fake_commit_apply() -> types.ModuleType:
    import importlib

    _real = importlib.import_module("utils.commit_apply")
    module = types.ModuleType("utils.commit_apply")
    module.STALE_MARKER = STALE
    module.PROTECTED_MARKER = PROTECTED
    module.calls = []

    def guard_record(state: Mapping[str, Any]) -> Optional[dict]:
        record = pinned_record(state)
        return record.get("guard") if record else None

    def recheck_files(guard, files, *, cwd, file_sha256):
        module.calls.append(("recheck_files", list(files)))
        expect = guard.get("expect") or {}
        protected = guard.get("protected") or {}
        head_kind = [f for f in files if f not in protected
                     and (expect.get(f) or {}).get("kind") != "batch"]
        changed = (
            git_utils.paths_changed_since(guard["head"], head_kind, cwd=cwd)
            if head_kind else set()
        )
        failing = []
        for path in files:
            want = expect.get(path) or {}
            if path in protected:
                failing.append(path)
            elif want.get("kind") == "batch":
                # Called with the absolute path, as the real one calls it.
                if file_sha256(os.path.join(cwd, path)) != want.get("sha256"):
                    failing.append(path)
            elif path in changed:
                failing.append(path)
        return failing

    def repin_after_batch(guard, task_id, sha256_by_path, *, cwd=None):
        module.calls.append(("repin_after_batch", task_id, dict(sha256_by_path), cwd))
        expect = guard.setdefault("expect", {})
        changed = False
        heads = []
        for path, digest in sha256_by_path.items():
            current = expect.get(path)
            if not isinstance(current, dict):
                continue
            if current.get("kind") == "batch":
                if current.get("sha256") != digest:
                    expect[path] = {"kind": "batch", "sha256": digest, "task_id": task_id}
                    changed = True
            else:
                heads.append((path, digest))
        moved = (
            git_utils.paths_changed_since(guard["head"], [p for p, _ in heads], cwd=cwd)
            if heads and cwd is not None else {p for p, _ in heads}
        )
        for path, digest in heads:
            if path in moved:
                expect[path] = {"kind": "batch", "sha256": digest, "task_id": task_id}
                changed = True
        return changed

    def protected_violations(guard, *, cwd, baseline=None):
        module.calls.append(("protected_violations",))
        if baseline is not None:
            # The executor's post-run check (fix 3, item 1): the real
            # window comparison, whose "before" the real disk_blob_ids read.
            return _real.protected_violations(guard, cwd=cwd, baseline=baseline)
        return sorted(
            path for path, blob in (guard.get("protected") or {}).items()
            if _blob_now(cwd, path) != blob
        )

    # The executor's post-run protected check (review fix 3, item 1) reads
    # the watched set and its ids, and restores what the snapshot holds no
    # bytes for, through the real module's functions.
    module.watched_ids = _real.watched_ids
    module.disk_blob_ids = _real.disk_blob_ids
    module.restore_watched = _real.restore_watched
    module.RESTORED_FROM_COPY = _real.RESTORED_FROM_COPY
    module.DELETED_ABSENT_BEFORE = _real.DELETED_ABSENT_BEFORE
    # The native leg's per-batch window (review R2, item 1): `util
    # write-prompt` records the watched ids as it binds a prompt, and `util
    # batch-done` compares with them and names the guard's copy only when it
    # holds what the path held before the batch.
    module.BOUND_KEY = _real.BOUND_KEY
    module.bind_watched = _real.bind_watched
    module.bound_baseline = _real.bound_baseline
    module.drop_bound = _real.drop_bound
    module.copy_holding = _real.copy_holding

    def mark_stale(guard, groups, failing, *, cwd):
        module.calls.append(("mark_stale", {g: list(f) for g, f in groups.items()},
                             list(failing)))
        bad = set(failing)
        skipped = guard.setdefault("skipped", {})
        for group_id, files in groups.items():
            hit = [path for path in files if path in bad]
            if hit:
                skipped[group_id] = {"reason": "changed", "files": hit}
        protected = guard.setdefault("protected", {})
        copies = guard.get("protected_copies")
        for path in failing:
            if path in protected:
                continue
            protected[path] = _blob_now(cwd, path)
            if copies and os.path.isfile(os.path.join(cwd, path)):
                target = os.path.join(cwd, copies, path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(cwd, path), target)
        for path in failing:
            (guard.get("expect") or {}).pop(path, None)

    def save_guard(state_path, guard):
        module.calls.append(("save_guard", str(state_path)))
        path = Path(state_path)
        with state_file_lock(path):
            state = json.loads(path.read_text(encoding="utf-8"))
            pinned_record(state)["guard"] = json.loads(json.dumps(guard))
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    for function in (guard_record, recheck_files, repin_after_batch,
                     protected_violations, mark_stale, save_guard):
        setattr(module, function.__name__, function)
    return module


class _Tripwire(types.ModuleType):
    """Stands in for ``utils.commit_apply`` where no guard exists."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"utils.commit_apply.{name} was used without a guard")


def _install(monkeypatch, module) -> None:
    monkeypatch.setitem(sys.modules, "utils.commit_apply", module)
    monkeypatch.setattr(utils, "commit_apply", module, raising=False)


@pytest.fixture
def fake_apply(monkeypatch):
    module = _fake_commit_apply()
    _install(monkeypatch, module)
    return module


@pytest.fixture
def tripwire(monkeypatch):
    module = _Tripwire("utils.commit_apply")
    _install(monkeypatch, module)
    return module


def _guard(head: str, *, expect: Dict[str, Any], protected: Optional[Dict[str, Any]] = None,
           copies: str = "plans/feature/apply-fixes/protected") -> Dict[str, Any]:
    return {
        "head": head,
        "decided_at": "2026-10-26T10:00:00Z",
        "skipped": {},
        "expect": expect,
        "protected": dict(protected or {}),
        "protected_copies": copies,
        "context_plan": None,
    }


def _head_expect(root: Path, head: str, *paths: str) -> Dict[str, Any]:
    return {path: {"kind": "head", "blob": _head_blob(root, head, path)} for path in paths}


# ===========================================================================
# write_safety: the exact-path deny layer
# ===========================================================================

class TestTheExactPathDenyLayer:

    def _policy(self, tmp_path, **over):
        kwargs = dict(root=str(tmp_path), allow=["**"], deny=["**/.env"])
        kwargs.update(over)
        return WritePolicy(**kwargs)

    def test_an_empty_layer_answers_as_before(self, tmp_path):
        policy = self._policy(tmp_path)
        assert policy.exact_deny == {}
        assert policy.violation(str(tmp_path / "src/b.py")) is None
        assert policy.hit("src/b.py") is None
        assert policy.to_dict() == {"root": str(tmp_path), "allow": ["**"],
                                    "deny": ["**/.env"]}

    def test_a_protected_path_is_refused_with_its_rule_text(self, tmp_path):
        policy = self._policy(tmp_path)
        rule = "changed since reviewed commit abc1234"
        assert policy.deny_exact(["src/b.py", "./src/c.py"], rule) == ["src/b.py", "src/c.py"]
        # Added once: a second call keeps the first rule.
        assert policy.deny_exact(["src/b.py"], "other") == []
        for spelling in ("src/b.py", str(tmp_path / "src" / "b.py")):
            assert policy.violation(spelling) == rule
            assert policy.denied(spelling) == rule
        assert policy.violation("src/a.py") is None
        # `exact=False` asks driver.write_paths alone (the sweep's retention).
        assert policy.violation("src/b.py", exact=False) is None
        hit = policy.hit("src/b.py")
        assert hit == PolicyHit(path="src/b.py", deny=(), allowed=True, protected=rule)
        assert PolicyHit.from_dict(hit.to_dict()) == hit
        assert policy.to_dict()["protected"] == {"src/b.py": rule, "src/c.py": rule}

    def test_the_check_names_the_path_and_offers_no_policy_change(self, tmp_path):
        policy = self._policy(tmp_path)
        policy.deny_exact(["src/b.py"], "changed since reviewed commit abc1234")
        with pytest.raises(PolicyViolation) as caught:
            policy.check(["src/a.py", "src/b.py"], when="post-run")
        message = caught.value.message
        assert "Write policy violation (post-run): src/b.py (changed since reviewed commit abc1234)." in message
        assert "driver.write_paths and the commit review's protected paths are checked" in message
        assert "The commit review protects src/b.py, which changed since reviewed commit abc1234" in message
        assert "driver.write_paths.allow" not in message
        assert caught.value.paths == ["src/b.py"]
        assert policy_allow_clause(caught.value.hits) is None

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
    def test_a_protected_symlink_is_named_by_its_own_path_not_its_target(self, tmp_path):
        """Review R2, item 3: the hit used to be named by the resolved path,
        so a protected link re-pointed at the batch's own fix file was
        "restored" as that file. It is named by the entry it matched."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
        os.symlink("a.py", tmp_path / "src" / "link.py")
        policy = self._policy(tmp_path)
        rule = "changed since reviewed commit abc1234"
        policy.deny_exact(["src/link.py"], rule)
        assert policy.exact_match("src/link.py") == ("src/link.py", rule)
        assert policy.hit("src/link.py") == PolicyHit(
            path="src/link.py", deny=(), allowed=True, protected=rule)
        assert policy.hit("src/a.py") is None
        with pytest.raises(PolicyViolation) as caught:
            policy.check(["src/a.py", "src/link.py"], when="post-run")
        assert caught.value.paths == ["src/link.py"]
        assert f"src/link.py ({rule})" in caught.value.message
        # A path reaching a protected file through a link not itself
        # protected is named by that file, as before.
        (tmp_path / "src" / "b.py").write_text("b = 1\n", encoding="utf-8")
        policy.deny_exact(["src/b.py"], rule)
        os.symlink("b.py", tmp_path / "src" / "alias.py")
        assert policy.hit("src/alias.py").path == "src/b.py"

    def test_the_way_forward_wording_is_unchanged_without_a_protected_hit(self, tmp_path):
        outside = [PolicyHit(path="../x", outside=True)]
        assert policy_outside_note(outside) == (
            "../x is outside the repository, so no driver.write_paths rule can allow it")
        denied = [PolicyHit(path=".env", deny=("**/.env",), allowed=True)]
        assert policy_way_forward(denied).startswith("Your call: if you want .env written")
        mixed = outside + [PolicyHit(path="src/b.py", protected="changed since reviewed commit abc1234")]
        assert policy_outside_note(mixed) == (
            "../x is outside the repository, so no driver.write_paths rule can allow it; "
            "the commit review protects src/b.py, which changed since reviewed commit "
            "abc1234: this apply never writes it, so make that change by hand, or review "
            "a range that ends at HEAD so the review reads it as it is now")


# ===========================================================================
# The executor leg
# ===========================================================================

def _fix_task(env, index: int, path: str, *,
              ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return make_write_task(
        kind="apply_code_fixes",
        task_id=f"apply-fix-batch-{index}",
        prompt=f"Apply fix {index} to {env.root / path}\n\n{PRIOR_CHANGES_TOKEN}\n",
        inputs=build_inputs(plan_file=str(env.plan)),
        batch_index=index,
        prior_changes_id=f"{path}#{index:08d}",
        phase_dir=str(env.phase_dir),
        declared_files=[str(env.root / path)],
        expected_ids=list(ids or [f"f{index}"]),
        output_name=write_output_name("apply_code_fixes", index),
    )


def _fixes(groups: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """A batch's fixes as a commit review's apply formats them: each with
    ``files``, every file its group names, primary first."""
    return [{"group_id": gid, "title": f"Fix {gid}", "file": files[0], "files": list(files)}
            for gid, files in groups.items()]


def _write_output(phase_dir: Path, batches: Mapping[int, Mapping[str, Sequence[str]]]) -> None:
    """``orchestrator_output.json`` with one batch per index, as the apply
    phase writes it before the scheduler runs (or the host reads it)."""
    rows = []
    for index in range(max(batches) + 1):
        groups = batches.get(index, {})
        first = next(iter(groups.values()), ["?"])
        rows.append({"file_key": first[0], "fixes": _fixes(groups)})
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "orchestrator_output.json").write_text(
        json.dumps({"batches": rows}, indent=2), encoding="utf-8")


def _runner(env, edits: Dict[str, List[tuple]]):
    """An in-process fake runner: writes *edits* for each task and answers
    with the write-batch result contract."""
    calls: List[str] = []

    def respond(task, prompt):
        task_id = task["task_id"]
        calls.append(task_id)
        touched = []
        for relative, content in edits.get(task_id, []):
            target = env.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            touched.append(relative)
        return {"status": "applied", "ids": list(task["expected_ids"]),
                "files_touched": touched, "summary": f"{task_id} applied",
                "check": {"passed": True, "ran": ["unit tests"], "findings": []}}

    respond.calls = calls  # type: ignore[attr-defined]
    return respond


def _guarded_state(env, guard: Dict[str, Any]) -> FakeState:
    state = FakeState(env.tmp_path / "state.json")
    state.state["commit_range"] = {"base": None, "head": guard["head"], "guard": guard}
    state.save()
    return state


def _load_state(path: Path) -> FakeState:
    """A FakeState over what is on disk (a resume's fresh read)."""
    state = FakeState.__new__(FakeState)
    state.state_file = str(path)
    state._path = path
    state.state = json.loads(path.read_text(encoding="utf-8"))
    return state


def _disk_guard(env) -> Dict[str, Any]:
    return json.loads((env.tmp_path / "state.json").read_text())["commit_range"]["guard"]


def _committed_repo(env, *extra: str) -> str:
    # The run's own git reads (tracking, the guard's) stay in the fixture repo.
    env.monkeypatch.chdir(env.root)
    for name in extra:
        (env.root / name).write_text(f"{Path(name).stem} = 1\n", encoding="utf-8")
    return _commit(env.root, "src")


class TestABatchsFixFiles:
    """The re-check's targets: the batch's primary file plus every file its
    fixes name in ``orchestrator_output.json``."""

    def test_the_batch_is_found_by_its_member_ids(self, env):
        task = _fix_task(env, 1, "src/a.py", ids=["g1", "g2"])
        batches = [
            {"fixes": _fixes({"x": ["src/x.py"]})},
            {"fixes": _fixes({"g1": ["src/a.py"], "g2": ["src/a.py", "src/b.py"]})},
        ]
        fixes = batch_fixes_for(task, batches)
        assert [fix["group_id"] for fix in fixes] == ["g1", "g2"]
        # Found by its ids wherever it sits, never by position alone.
        assert batch_fixes_for(task, list(reversed(batches))) == fixes
        assert batch_fixes_for(task, [batches[0]]) is None
        assert batch_fixes_for(task, None) is None

    def test_without_files_a_fix_names_its_own_file_and_without_fixes_the_primary(self, env):
        task = _fix_task(env, 0, "src/a.py", ids=["g1", "g2"])
        fixes = [{"group_id": "g1", "file": "src/a.py"}, {"group_id": "g2", "file": "src/c.py"}]
        assert commit_group_files(task, repo_root=str(env.root), fixes=fixes) == {
            "g1": ["src/a.py"], "g2": ["src/a.py", "src/c.py"]}
        assert commit_group_files(task, repo_root=str(env.root)) == {
            "g1": ["src/a.py"], "g2": ["src/a.py"]}


class TestAProtectedWriteOnTheExecutor:
    """Drifted secondary file: a fix in unchanged ``a.py`` whose batch also
    edits drifted ``b.py`` (outside the range, no finding of its own)."""

    def test_the_write_is_restored_and_the_loop_stops_naming_it(self, env, fake_apply):
        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard(head, expect=_head_expect(env.root, head, "src/a.py"),
                       protected={"src/b.py": _disk_blob(env.root, "src/b.py")})
        state = _guarded_state(env, guard)
        respond = _runner(env, {"apply-fix-batch-0": [
            ("src/a.py", "a = 2\n"), ("src/b.py", "b = 'clobbered'\n")]})
        lines: List[str] = []
        scheduler = env.scheduler(
            [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/a.py")],
            kind="apply_code_fixes", state=state, respond=respond, log=lines.append,
        )
        rule = f"changed since reviewed commit {head[:7]}"
        assert scheduler.policy.exact_deny == {"src/b.py": rule}

        result = scheduler.run()

        assert result.exit_code == 3
        assert respond.calls == ["apply-fix-batch-0"]
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
        assert f"src/b.py ({rule})" in outcome.reason
        assert "Restored from the pre-batch snapshot: src/b.py." in outcome.reason
        assert "The commit review protects src/b.py" in outcome.reason
        assert "driver.write_paths.allow" not in outcome.reason
        # b.py is back as the user left it; nothing else is undone.
        assert (env.root / "src/b.py").read_text() == "b = 'drifted'\n"
        assert (env.root / "src/a.py").read_text() == "a = 2\n"
        assert "src/b.py" in "\n".join(result.messages)
        assert state.get_task_status("apply-fix-batch-0") == "failed"
        assert not any(line.startswith(STALE) for line in lines)

    def test_a_fix_that_declares_a_protected_path_is_skipped_and_the_run_goes_on(
        self, env, fake_apply,
    ):
        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard(head, expect=_head_expect(env.root, head, "src/a.py"),
                       protected={"src/b.py": _disk_blob(env.root, "src/b.py")})
        state = _guarded_state(env, guard)
        respond = _runner(env, {"apply-fix-batch-1": [("src/a.py", "a = 2\n")]})
        lines: List[str] = []
        result = env.scheduler(
            [_fix_task(env, 0, "src/b.py"), _fix_task(env, 1, "src/a.py")],
            kind="apply_code_fixes", state=state, respond=respond, log=lines.append,
        ).run()

        assert result.exit_code == 0, result.messages
        assert respond.calls == ["apply-fix-batch-1"]
        assert any(
            line.startswith(f"{STALE} apply-fix-batch-0: src/b.py changed since "
                            f"reviewed commit {head[:7]}") for line in lines)
        assert _disk_guard(env)["skipped"] == {"f0": {"reason": "changed", "files": ["src/b.py"]}}
        assert (env.root / "src/b.py").read_text() == "b = 'drifted'\n"


def _guard_with_copies(env, head: str, protected: Dict[str, Optional[str]], *,
                       context_plan: Optional[str] = None) -> Dict[str, Any]:
    """A guard as the real ``take_guard`` leaves it: the protected set's
    bytes (and an unprotected context plan's) copied under
    ``protected_copies``, through the real ``utils.commit_apply``."""
    from utils import commit_apply as real

    guard = _guard(head, expect=_head_expect(env.root, head, "src/a.py"),
                   protected=protected)
    copies = env.root / guard["protected_copies"]
    copied = list(protected)
    if context_plan is not None:
        guard["context_plan"] = context_plan
        guard["context_plan_id"] = real.disk_blob_ids(
            [context_plan], cwd=str(env.root))[context_plan]
        copied.append(context_plan)
    real.copy_protected(copied, cwd=str(env.root), dest=str(copies))
    return guard


class TestTheProtectedCheckDoesNotNeedTheSweep:
    """Review fix 3, item 1: the executor's protected set used to be enforced
    only through the post-run undeclared-write sweep. On a tree over
    ``SWEEP_MAX_FILES`` (this repository has over 100k files) the census is
    abandoned and a protected write landed silently; with no retained bytes
    (``SWEEP_RETAIN_MAX_BYTES``) it was detected but left on disk. Now every
    batch under a guard compares the protected set's disk ids before and
    after it ran (``commit_apply.protected_violations``) and restores each
    offender from the snapshot, else the guard's copy, else by deleting a
    file that did not exist. These run the REAL ``utils.commit_apply``."""

    def _run(self, env, guard, edits, tasks=("src/a.py",)):
        state = _guarded_state(env, guard)
        respond = _runner(env, edits)
        lines: List[str] = []
        scheduler = env.scheduler(
            [_fix_task(env, index, path) for index, path in enumerate(tasks)],
            kind="apply_code_fixes", state=state, respond=respond, log=lines.append,
        )
        return scheduler.run(), respond, state, lines

    @pytest.mark.parametrize("budget", ["SWEEP_MAX_FILES", "SWEEP_RETAIN_MAX_BYTES"])
    def test_a_protected_write_is_caught_and_restored_from_the_guards_copy(
        self, env, monkeypatch, budget,
    ):
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard_with_copies(env, head, {"src/b.py": _disk_blob(env.root, "src/b.py")})
        monkeypatch.setattr(wbs, budget, 1 if budget == "SWEEP_MAX_FILES" else 0)

        result, respond, state, _lines = self._run(
            env, guard,
            {"apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/b.py", "b = 'clobbered'\n")]},
            tasks=("src/a.py", "src/a.py"),
        )

        assert result.exit_code == 3
        # The loop stops: the second batch is never dispatched.
        assert respond.calls == ["apply-fix-batch-0"]
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
        rule = f"changed since reviewed commit {head[:7]}"
        assert f"src/b.py ({rule})" in outcome.reason
        assert "from what the commit review's guard kept" in outcome.reason
        assert "Not restored" not in outcome.reason
        assert (env.root / "src/b.py").read_text() == "b = 'drifted'\n"
        assert (env.root / "src/a.py").read_text() == "a = 2\n"
        assert state.get_task_status("apply-fix-batch-0") == "failed"

    def test_a_protected_path_absent_before_the_batch_is_deleted(self, env, monkeypatch):
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env, "src/gone.py")
        (env.root / "src/gone.py").unlink()          # deleted since head: protected, null
        guard = _guard_with_copies(env, head, {"src/gone.py": None})
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)

        result, _respond, _state, _lines = self._run(
            env, guard,
            {"apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/gone.py", "back\n")]},
        )

        assert result.exit_code == 3
        assert "src/gone.py" in result.outcomes[0].reason
        assert not (env.root / "src/gone.py").exists()

    def test_without_a_copy_the_write_is_named_and_left_for_a_human(self, env, monkeypatch):
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard(head, expect=_head_expect(env.root, head, "src/a.py"),
                       protected={"src/b.py": _disk_blob(env.root, "src/b.py")})
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)

        result, _respond, _state, _lines = self._run(
            env, guard,
            {"apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/b.py", "b = 'clobbered'\n")]},
        )

        assert result.exit_code == 3
        reason = result.outcomes[0].reason
        assert "Not restored, because no rollback point exists (src/b.py)" in reason

    def test_a_copy_older_than_a_later_edit_is_never_written_over_it(self, env, monkeypatch):
        """The user edited protected ``b.py`` after the guard took its copy,
        then a batch clobbered it: the copy is not what ``b.py`` held before
        the batch, so it is not restored over the user's edit."""
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard_with_copies(env, head, {"src/b.py": _disk_blob(env.root, "src/b.py")})
        (env.root / "src/b.py").write_text("b = 'the user, later'\n", encoding="utf-8")
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)

        result, _respond, _state, _lines = self._run(
            env, guard,
            {"apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/b.py", "b = 'clobbered'\n")]},
        )

        assert result.exit_code == 3
        assert "Not restored, because no rollback point exists (src/b.py)" in \
            result.outcomes[0].reason
        assert (env.root / "src/b.py").read_text() == "b = 'clobbered'\n"

    def test_an_edit_between_batches_is_the_users_not_a_violation(self, env, monkeypatch):
        """Only a change inside a batch's own window counts: a protected file
        the user edits before a batch that never touches it stays as they
        left it, and the batch completes."""
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard_with_copies(env, head, {"src/b.py": _disk_blob(env.root, "src/b.py")})
        (env.root / "src/b.py").write_text("b = 'the user, later'\n", encoding="utf-8")
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)

        result, respond, _state, _lines = self._run(
            env, guard, {"apply-fix-batch-0": [("src/a.py", "a = 2\n")]},
        )

        assert result.exit_code == 0, result.messages
        assert respond.calls == ["apply-fix-batch-0"]
        assert (env.root / "src/b.py").read_text() == "b = 'the user, later'\n"

    def test_a_batch_that_never_reached_classification_is_restored_too(self, env, monkeypatch):
        """A reply that never parses ends the dispatch before ``_classify``
        (``_audit_terminal``): the protected write is put back all the same."""
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env)
        (env.root / "src/b.py").write_text("b = 'drifted'\n", encoding="utf-8")
        guard = _guard_with_copies(env, head, {"src/b.py": _disk_blob(env.root, "src/b.py")})
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)
        state = _guarded_state(env, guard)

        def respond(task, prompt):
            (env.root / "src/b.py").write_text("b = 'clobbered'\n", encoding="utf-8")
            return "no json here at all"

        result = env.scheduler([_fix_task(env, 0, "src/a.py")], kind="apply_code_fixes",
                               state=state, respond=respond, log=lambda line: None).run()

        assert result.exit_code != 0
        outcome = result.outcomes[0]
        assert not outcome.ok
        assert "It also changed src/b.py, which the commit review protects" in outcome.reason
        assert (env.root / "src/b.py").read_text() == "b = 'drifted'\n"

    def test_an_undeclared_write_to_the_context_plan_fails_like_a_protected_write(
        self, env, monkeypatch,
    ):
        """Item 4: the context plan joins the exact-path deny layer and the
        watched set even when it never changed since ``head`` (so it is not
        in the protected set)."""
        from utils import write_batch_scheduler as wbs
        from utils.write_safety import CONTEXT_PLAN_RULE

        (env.root / "docs").mkdir()
        (env.root / "docs" / "x.md").write_text("# the plan\n", encoding="utf-8")
        env.monkeypatch.chdir(env.root)
        _git(env.root, "add", "--", "src", "docs")
        _git(env.root, "commit", "-q", "-m", "reviewed")
        head = _git(env.root, "rev-parse", "HEAD")
        guard = _guard_with_copies(env, head, {}, context_plan="docs/x.md")
        monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)
        state = _guarded_state(env, guard)
        respond = _runner(env, {"apply-fix-batch-0": [
            ("src/a.py", "a = 2\n"), ("docs/x.md", "# rewritten by the batch\n")]})
        scheduler = env.scheduler([_fix_task(env, 0, "src/a.py")], kind="apply_code_fixes",
                                  state=state, respond=respond, log=lambda line: None)
        assert scheduler.policy.exact_deny == {"docs/x.md": CONTEXT_PLAN_RULE}

        result = scheduler.run()

        assert result.exit_code == 3
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
        assert f"docs/x.md ({CONTEXT_PLAN_RULE})" in outcome.reason
        assert "--context-plan \"\"" in outcome.reason
        assert (env.root / "docs" / "x.md").read_text() == "# the plan\n"

    def test_a_batch_declaring_the_context_plan_is_refused_before_dispatch(self, env):
        from utils.write_safety import CONTEXT_PLAN_RULE

        (env.root / "docs").mkdir()
        (env.root / "docs" / "x.md").write_text("# the plan\n", encoding="utf-8")
        env.monkeypatch.chdir(env.root)
        _git(env.root, "add", "--", "src", "docs")
        _git(env.root, "commit", "-q", "-m", "reviewed")
        head = _git(env.root, "rev-parse", "HEAD")
        guard = _guard_with_copies(env, head, {}, context_plan="docs/x.md")
        guard["expect"].update(_head_expect(env.root, head, "docs/x.md"))
        result, respond, _state, _lines = self._run(
            env, guard, {"apply-fix-batch-0": [("docs/x.md", "# no\n")]}, tasks=("docs/x.md",),
        )
        assert respond.calls == []
        assert CONTEXT_PLAN_RULE in result.outcomes[0].reason
        assert (env.root / "docs" / "x.md").read_text() == "# the plan\n"


class TestTheSecondReviewOnTheExecutor:
    """Review R2, items 2 and 3, on the REAL ``utils.commit_apply``."""

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
    @pytest.mark.parametrize("sweep", ["available", "unavailable"])
    def test_a_repointed_protected_symlink_is_restored_as_the_link(
        self, env, monkeypatch, sweep,
    ):
        """The batch fixes ``a.py`` and re-points protected ``link.py`` (it
        pointed at ``b.py``) at ``a.py``. The violation used to name the
        link's target, so ``a.py`` was "restored" (the batch's own fix
        undone) and the link left re-pointed; it is now named and restored
        as the link, from the guard's copy, and ``a.py`` is left alone."""
        from utils import commit_apply as real
        from utils import write_batch_scheduler as wbs

        head = _committed_repo(env, "src/b.py")
        link = env.root / "src" / "link.py"
        os.symlink("b.py", link)                       # untracked since head: protected
        guard = _guard_with_copies(
            env, head, real.disk_blob_ids(["src/link.py"], cwd=str(env.root)))
        assert os.path.islink(env.root / guard["protected_copies"] / "src/link.py")
        if sweep == "unavailable":
            monkeypatch.setattr(wbs, "SWEEP_MAX_FILES", 1)
        state = _guarded_state(env, guard)

        def respond(task, prompt):
            (env.root / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            os.unlink(link)
            os.symlink("a.py", link)
            return {"status": "applied", "ids": list(task["expected_ids"]),
                    "files_touched": ["src/a.py"], "summary": "applied",
                    "check": {"passed": True, "ran": ["unit tests"], "findings": []}}

        result = env.scheduler([_fix_task(env, 0, "src/a.py")], kind="apply_code_fixes",
                               state=state, respond=respond, log=lambda line: None).run()

        assert result.exit_code == 3
        outcome = result.outcomes[0]
        assert outcome.write_outcome == executor.WRITE_POLICY_VIOLATION
        rule = f"changed since reviewed commit {head[:7]}"
        assert f"src/link.py ({rule})" in outcome.reason
        assert f"src/a.py ({rule})" not in outcome.reason
        assert os.readlink(link) == "b.py"
        assert (env.root / "src/a.py").read_text() == "a = 2\n"

    @pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                        reason="needs POSIX modes and a non-root user")
    def test_an_unreadable_protected_file_does_not_fail_every_batch(self, env):
        """It used to read as changed in every batch's window (``unreadable``
        before, ``unreadable`` after): the first batch failed with a reason
        that was not true and the loop stopped. Both batches now run; a
        batch that makes it readable is still caught."""
        from utils import commit_apply as real

        head = _committed_repo(env)
        secret = env.root / "secret.env"
        secret.write_text("TOKEN=1\n", encoding="utf-8")          # untracked: protected
        os.chmod(secret, 0)
        try:
            ids = real.disk_blob_ids(["secret.env"], cwd=str(env.root))
            assert ids == {"secret.env": "unreadable"}
            guard = _guard(head, expect=_head_expect(env.root, head, "src/a.py"),
                           protected=dict(ids))
            state = _guarded_state(env, guard)
            respond = _runner(env, {"apply-fix-batch-0": [("src/a.py", "a = 2\n")],
                                    "apply-fix-batch-1": [("src/a.py", "a = 3\n")]})
            result = env.scheduler(
                [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/a.py")],
                kind="apply_code_fixes", state=state, respond=respond, log=lambda line: None,
            ).run()
            assert result.exit_code == 0, [o.reason for o in result.outcomes]
            assert respond.calls == ["apply-fix-batch-0", "apply-fix-batch-1"]
            assert (env.root / "src/a.py").read_text() == "a = 3\n"

            def unlock(task, prompt):
                os.chmod(secret, 0o644)
                secret.write_text("TOKEN=2\n", encoding="utf-8")
                return {"status": "applied", "ids": list(task["expected_ids"]),
                        "files_touched": [], "summary": "applied",
                        "check": {"passed": True, "ran": ["unit tests"], "findings": []}}

            again = env.scheduler([_fix_task(env, 2, "src/a.py")], kind="apply_code_fixes",
                                  state=_guarded_state(env, guard), respond=unlock,
                                  log=lambda line: None).run()
            assert again.exit_code == 3
            assert again.outcomes[0].write_outcome == executor.WRITE_POLICY_VIOLATION
            assert f"secret.env (changed since reviewed commit {head[:7]})" in \
                again.outcomes[0].reason
        finally:
            os.chmod(secret, 0o644)


class TestANewFileOnTheExecutor:
    """A fix whose file is absent at head and on disk creates it; resume
    accepts it by its saved sha256, while an edited untouched fix file is
    skipped and protected."""

    def test_created_then_resumed(self, env, fake_apply):
        head = _committed_repo(env)
        expect = _head_expect(env.root, head, "src/a.py")
        expect["tests/test_new.py"] = {"kind": "head", "blob": None}
        state = _guarded_state(env, _guard(head, expect=expect))
        first = _runner(env, {"apply-fix-batch-0": [("tests/test_new.py", "def test(): pass\n")]})
        tasks = [
            _fix_task(env, 0, "tests/test_new.py"),
            _fix_task(env, 1, "tests/test_new.py"),
            _fix_task(env, 2, "src/a.py"),
        ]
        # The apply is cut off after batch 0.
        result = env.scheduler(tasks[:1], kind="apply_code_fixes", state=state,
                               respond=first).run()
        assert result.exit_code == 0, result.messages
        created = env.root / "tests/test_new.py"
        assert first.calls == ["apply-fix-batch-0"]
        assert _disk_guard(env)["expect"]["tests/test_new.py"] == {
            "kind": "batch", "sha256": _sha256(created), "task_id": "apply-fix-batch-0"}

        # Between the runs the user edits a.py, which no batch has touched.
        (env.root / "src/a.py").write_text("a = 'by hand'\n", encoding="utf-8")
        resumed = _load_state(env.tmp_path / "state.json")
        second = _runner(env, {
            "apply-fix-batch-1": [("tests/test_new.py", "def test(): assert True\n")],
            "apply-fix-batch-2": [("src/a.py", "a = 2\n")],
        })
        lines: List[str] = []
        result = env.scheduler(tasks, kind="apply_code_fixes", state=resumed,
                               respond=second, resume=True, log=lines.append).run()

        assert result.exit_code == 0, result.messages
        # The later fix to the created file runs; the edited file's does not.
        assert second.calls == ["apply-fix-batch-1"]
        assert (env.root / "src/a.py").read_text() == "a = 'by hand'\n"
        assert any(line.startswith(f"{STALE} apply-fix-batch-2: src/a.py changed since "
                                   f"reviewed commit {head[:7]}") for line in lines)
        guard = _disk_guard(env)
        assert guard["skipped"]["f2"] == {"reason": "changed", "files": ["src/a.py"]}
        assert guard["protected"]["src/a.py"] == _disk_blob(env.root, "src/a.py")
        assert guard["expect"]["tests/test_new.py"] == {
            "kind": "batch", "sha256": _sha256(created), "task_id": "apply-fix-batch-1"}
        assert resumed.get_task_status("apply-fix-batch-2") == "failed"
        assert resumed.get_task_status_reason("apply-fix-batch-2") == (
            f"skipped: src/a.py changed since reviewed commit {head[:7]}")
        # Its item stays in a resume's work set; the guard's skip decides.
        assert resumed.state["processed_apply-suggestions"]["f2"]["status"] == "failed"
        repins = [call for call in fake_apply.calls if call[0] == "repin_after_batch"]
        assert repins and all(call[3] == os.path.realpath(env.root) for call in repins)
        assert {"tests/test_new.py"} >= {p for call in repins for p in call[2]}


class TestTheReCheckBetweenBatches:

    def test_an_edit_while_batch_1_runs_stops_batch_2_and_batch_3_runs(self, env, fake_apply):
        head = _committed_repo(env, "src/c.py")
        state = _guarded_state(env, _guard(
            head, expect=_head_expect(env.root, head, "src/a.py", "src/b.py", "src/c.py")))
        respond = _runner(env, {
            "apply-fix-batch-0": [("src/a.py", "a = 2\n")],
            "apply-fix-batch-2": [("src/c.py", "c = 2\n")],
        })
        lines: List[str] = []

        def log(line: str) -> None:
            lines.append(line)
            if "apply-fix-batch-0 applied" in line:
                # The user edits batch 2's untouched primary file meanwhile.
                (env.root / "src/b.py").write_text("b = 'by hand'\n", encoding="utf-8")

        result = env.scheduler(
            [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/b.py"),
             _fix_task(env, 2, "src/c.py")],
            kind="apply_code_fixes", state=state, respond=respond, log=log,
        ).run()

        assert result.exit_code == 0, result.messages
        assert respond.calls == ["apply-fix-batch-0", "apply-fix-batch-2"]
        stale = [line for line in lines if line.startswith(STALE)]
        assert len(stale) == 1
        assert stale[0].startswith(
            f"{STALE} apply-fix-batch-1: src/b.py changed since reviewed commit {head[:7]}")
        guard = _disk_guard(env)
        assert guard["skipped"] == {"f1": {"reason": "changed", "files": ["src/b.py"]}}
        assert "src/b.py" in guard["protected"]
        # Saved shape: a head or batch identity per file, never a pass flag.
        assert guard["expect"]["src/a.py"]["kind"] == "batch"
        assert guard["expect"]["src/c.py"]["kind"] == "batch"
        # Skipped and protected now: nothing may write it, so it has no
        # expectation left.
        assert "src/b.py" not in guard["expect"]
        assert all(set(v) <= {"kind", "blob", "sha256", "task_id"}
                   for v in guard["expect"].values())
        assert (env.root / "src/b.py").read_text() == "b = 'by hand'\n"

    def test_a_file_that_moved_inside_batch_1s_window_is_not_pinned_as_its(
        self, env, fake_apply,
    ):
        """The sweep sees batch 2's file move during batch 1 and cannot tell
        who moved it; only batch 1's own fix files are re-pinned, so the
        edit is not accepted as batch 1's and batch 2 is still stopped."""
        head = _committed_repo(env, "src/c.py")
        state = _guarded_state(env, _guard(
            head, expect=_head_expect(env.root, head, "src/a.py", "src/b.py", "src/c.py")))
        respond = _runner(env, {
            "apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/b.py", "b = 'meanwhile'\n")],
            "apply-fix-batch-2": [("src/c.py", "c = 2\n")],
        })
        result = env.scheduler(
            [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/b.py"),
             _fix_task(env, 2, "src/c.py")],
            kind="apply_code_fixes", state=state, respond=respond,
        ).run()
        assert result.exit_code == 0, result.messages
        assert "src/b.py" in result.outcomes[0].sha256_after
        assert respond.calls == ["apply-fix-batch-0", "apply-fix-batch-2"]
        repinned = [call[2] for call in fake_apply.calls
                    if call[0] == "repin_after_batch" and call[1] == "apply-fix-batch-0"]
        assert repinned and all(set(paths) == {"src/a.py"} for paths in repinned)
        assert _disk_guard(env)["skipped"]["f1"]["files"] == ["src/b.py"]

    def test_an_edit_made_while_awaiting_a_decision_is_caught_before_dispatch(
        self, env, fake_apply,
    ):
        """The guard was saved before the human decisions; the file changed
        since. As far as T6b goes: the approved batch is not dispatched, its
        group is skipped and the file protected, and the run completes."""
        head = _committed_repo(env)
        state = _guarded_state(env, _guard(head, expect=_head_expect(env.root, head, "src/a.py")))
        (env.root / "src/a.py").write_text("a = 'edited during the ask'\n", encoding="utf-8")
        respond = _runner(env, {"apply-fix-batch-0": [("src/a.py", "a = 2\n")]})
        result = env.scheduler([_fix_task(env, 0, "src/a.py")], kind="apply_code_fixes",
                               state=state, respond=respond).run()
        assert result.exit_code == 0, result.messages
        assert respond.calls == []
        guard = _disk_guard(env)
        assert guard["skipped"]["f0"]["files"] == ["src/a.py"]
        assert guard["protected"]["src/a.py"] == _disk_blob(env.root, "src/a.py")

    @pytest.mark.parametrize("edited", [True, False])
    def test_an_external_edit_between_resumes_to_a_file_batch_1_edited(
        self, env, fake_apply, edited,
    ):
        head = _committed_repo(env)
        state = _guarded_state(env, _guard(head, expect=_head_expect(env.root, head, "src/a.py")))
        tasks = [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/a.py")]
        first = _runner(env, {"apply-fix-batch-0": [("src/a.py", "a = 2\n")]})
        env.scheduler(tasks[:1], kind="apply_code_fixes", state=state, respond=first).run()
        assert _disk_guard(env)["expect"]["src/a.py"]["kind"] == "batch"
        if edited:
            (env.root / "src/a.py").write_text("a = 'by hand'\n", encoding="utf-8")
        second = _runner(env, {"apply-fix-batch-1": [("src/a.py", "a = 3\n")]})
        lines: List[str] = []
        result = env.scheduler(tasks, kind="apply_code_fixes",
                               state=_load_state(env.tmp_path / "state.json"),
                               respond=second, resume=True, log=lines.append).run()
        assert result.exit_code == 0, result.messages
        if edited:
            assert second.calls == []
            assert any(line.startswith(f"{STALE} apply-fix-batch-1: src/a.py") for line in lines)
            assert (env.root / "src/a.py").read_text() == "a = 'by hand'\n"
        else:
            # Passed by its saved sha256: the batch runs.
            assert second.calls == ["apply-fix-batch-1"]
            assert not any(line.startswith(STALE) for line in lines)


class TestAHeldBackGroup:

    def test_the_other_group_is_held_back_and_a_resume_applies_it(self, env, fake_apply):
        head = _committed_repo(env, "src/c.py")
        state = _guarded_state(env, _guard(
            head, expect=_head_expect(env.root, head, "src/a.py", "src/b.py", "src/c.py")))
        # The second group's secondary file changes after the guard.
        (env.root / "src/b.py").write_text("b = 'by hand'\n", encoding="utf-8")
        batch = _fix_task(env, 0, "src/a.py", ids=["g1", "g2"])
        _write_output(env.phase_dir, {0: {"g1": ["src/a.py"], "g2": ["src/a.py", "src/b.py"]},
                                      1: {"f1": ["src/c.py"]}})
        fixes = batch_fixes_for(batch, load_orchestrator_batches(env.phase_dir))
        assert commit_group_files(batch, repo_root=str(env.root), fixes=fixes) == {
            "g1": ["src/a.py"], "g2": ["src/a.py", "src/b.py"]}
        respond = _runner(env, {"apply-fix-batch-1": [("src/c.py", "c = 2\n")]})
        lines: List[str] = []
        result = env.scheduler([batch, _fix_task(env, 1, "src/c.py")],
                               kind="apply_code_fixes", state=state, respond=respond,
                               log=lines.append).run()

        assert result.exit_code == 0, result.messages
        assert respond.calls == ["apply-fix-batch-1"]
        assert ("recheck_files", ["src/a.py", "src/b.py"]) in fake_apply.calls
        guard = _disk_guard(env)
        assert guard["skipped"] == {"g2": {"reason": "changed", "files": ["src/b.py"]}}
        assert state.get_task_status("apply-fix-batch-0") == "failed"
        assert state.get_task_status_reason("apply-fix-batch-0") == (
            "held back: batched with src/b.py")
        processed = state.state["processed_apply-suggestions"]
        assert processed["g1"]["status"] == "failed"
        assert processed["g2"]["status"] == "failed"
        line = next(line for line in lines if line.startswith(STALE))
        assert "1 fix(es) skipped, 1 held back, recorded failed" in line

        # The resume re-batches g1 alone; the guard's protected set is now
        # the policy's deny layer from the start.
        rebatched = _fix_task(env, 0, "src/a.py", ids=["g1"])
        _write_output(env.phase_dir, {0: {"g1": ["src/a.py"]}})
        again = _runner(env, {"apply-fix-batch-0": [("src/a.py", "a = 2\n")]})
        resumed = _load_state(env.tmp_path / "state.json")
        scheduler = env.scheduler([rebatched], kind="apply_code_fixes", state=resumed,
                                  respond=again, resume=True)
        assert scheduler.policy.exact_deny == {
            "src/b.py": f"changed since reviewed commit {head[:7]}"}
        result = scheduler.run()
        assert result.exit_code == 0, result.messages
        assert again.calls == ["apply-fix-batch-0"]
        assert resumed.get_task_status("apply-fix-batch-0") == "completed"
        assert (env.root / "src/a.py").read_text() == "a = 2\n"


class TestNoGuardRunsUnchanged:

    def test_a_cross_file_write_and_a_mid_run_edit_pass_as_before(self, env, tripwire):
        _committed_repo(env, "src/c.py")
        state = FakeState(env.tmp_path / "state.json")
        respond = _runner(env, {
            "apply-fix-batch-0": [("src/a.py", "a = 2\n"), ("src/b.py", "b = 2\n")],
            "apply-fix-batch-1": [("src/c.py", "c = 2\n")],
        })
        lines: List[str] = []

        def log(line: str) -> None:
            lines.append(line)
            if "apply-fix-batch-0 applied" in line:
                (env.root / "src/c.py").write_text("c = 'by hand'\n", encoding="utf-8")

        scheduler = env.scheduler(
            [_fix_task(env, 0, "src/a.py"), _fix_task(env, 1, "src/c.py")],
            kind="apply_code_fixes", state=state, respond=respond, log=log,
        )
        assert scheduler.policy.exact_deny == {}
        assert "protected" not in scheduler.policy.to_dict()
        result = scheduler.run()
        assert result.exit_code == 0, result.messages
        assert respond.calls == ["apply-fix-batch-0", "apply-fix-batch-1"]
        assert (env.root / "src/b.py").read_text() == "b = 2\n"
        assert not any(STALE in line for line in lines)
        assert "commit_range" not in state.state


class TestTheResumeBindingUnderAGuard:
    """D2g's resume binding (``_task_fingerprint``) leaves the prompt's
    protected-path clause out under a guard. A ``--resume`` grows the
    protected set whenever its re-check protects a file edited by hand, and a
    completed batch must not read as made from changed inputs and be applied
    twice. It leaves out nothing else, and nothing at all without a guard."""

    @staticmethod
    def _task(env, rule: str, title: str = "a") -> Dict[str, Any]:
        task = _fix_task(env, 0, "src/a.py")
        task["prompt"] = (f"Fix the following code issue: {title}\n\n{PRIOR_CHANGES_TOKEN}\n\n"
                          f"## Instructions\n6. {rule}\n\nReturn: Brief summary.")
        return task

    def test_only_the_protected_clause_is_left_out(self, env, fake_apply):
        from utils.code_fix_batcher import code_only_rule

        head = _committed_repo(env)
        rules = [code_only_rule("plans/b.md", protected=paths, protected_head=head)
                 for paths in ((), ["src/b.py"], ["src/b.py", "src/c.py"])]
        guarded = env.scheduler(
            [self._task(env, rules[0])], kind="apply_code_fixes",
            state=_guarded_state(env, _guard(head, expect={})), respond=_runner(env, {}),
        )
        bound = [guarded._task_fingerprint(self._task(env, rule)) for rule in rules]
        assert bound[0] == bound[1] == bound[2]
        assert guarded._task_fingerprint(self._task(env, rules[1], title="b")) != bound[1]
        other_plan = code_only_rule("plans/other.md", protected=["src/b.py"],
                                    protected_head=head)
        assert guarded._task_fingerprint(self._task(env, other_plan)) != bound[1]

        plain = env.scheduler(
            [self._task(env, rules[0])], kind="apply_code_fixes",
            state=FakeState(env.tmp_path / "plain-state.json"), respond=_runner(env, {}),
        )
        assert plain._task_fingerprint(self._task(env, rules[0])) != plain._task_fingerprint(
            self._task(env, rules[1]))


# ===========================================================================
# The native leg: util write-prompt and util batch-done, called as functions
# ===========================================================================

class NativeRepo:
    """A real repository holding a commit brief's output dir and an
    apply-fixes phase materialized for the native leg."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "repo"
        (self.root / "src").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        for name in ("a", "b", "c"):
            (self.root / "src" / f"{name}.py").write_text(f"{name} = 1\n", encoding="utf-8")
        self.head = _commit(self.root, "src")
        self.plan = self.root / "plans" / "feature.md"
        self.plan.parent.mkdir(parents=True)
        self.plan.write_text("# Commit brief\n", encoding="utf-8")
        self.out = self.root / "plans" / "feature"
        self.out.mkdir()
        self.phase_dir = self.out / "apply-fixes"

    def init_state(self) -> None:
        StateManager(self.plan).save()

    def materialize(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        tasks = materialize_write_tasks(
            kind="apply_code_fixes", batches=batches, phase_dir=str(self.phase_dir),
            plan_file=str(self.plan),
        )
        write_task_file_for_phase(
            kind="apply_code_fixes", tasks=tasks, phase_dir=str(self.phase_dir),
            plan_file=str(self.plan),
        )
        # The host's copy of the batches, fixes and their files included.
        (self.phase_dir / "orchestrator_output.json").write_text(
            json.dumps({"batches": batches}, indent=2), encoding="utf-8")
        return tasks

    @property
    def state_path(self) -> Path:
        return self.out / "state.json"

    def state(self) -> Dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def guard(self) -> Dict[str, Any]:
        return self.state()["commit_range"]["guard"]

    def install_guard(self, guard: Dict[str, Any]) -> None:
        state = self.state()
        state["commit_range"] = {"base": None, "head": guard["head"], "guard": guard}
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def write(self, relative: str, text: str) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _batch(file_key: str, *group_ids: str,
           files: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    files = files or {}
    return {
        "file_key": file_key,
        "prompt": f"Fix {file_key}.\n\n{PRIOR_CHANGES_TOKEN}\n",
        "fixes": _fixes({gid: files.get(gid, [file_key]) for gid in group_ids}),
    }


def _util(capsys, *argv: str):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _write_prompt(repo: NativeRepo, capsys, task_id: str):
    return _util(capsys, "write-prompt", "--phase-dir", str(repo.phase_dir),
                 "--task-id", task_id)


def _batch_done(repo: NativeRepo, capsys, task_id: str, *extra: str):
    return _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                 "--task-id", task_id, "--status", "completed",
                 "--own-check", "passed", *extra)


def _prompt_file(repo: NativeRepo, task_id: str) -> Path:
    return Path(dispatched_prompt_path(str(repo.phase_dir), task_id))


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    repo = NativeRepo(tmp_path)
    monkeypatch.chdir(repo.root)
    repo.init_state()
    return repo


class TestAProtectedWriteOnTheNativeLeg:

    def test_batch_done_refuses_with_the_copy_as_the_way_back(self, native, capsys, fake_apply):
        native.materialize([_batch("src/a.py", "g0")])
        native.write("src/b.py", "b = 'drifted'\n")
        copies = "plans/feature/apply-fixes/protected"
        native.write(f"{copies}/src/b.py", "b = 'drifted'\n")
        native.install_guard(_guard(
            native.head, expect=_head_expect(native.root, native.head, "src/a.py"),
            protected={"src/b.py": _disk_blob(native.root, "src/b.py")}, copies=copies))

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-0")
        assert code == 0, err
        assert out.startswith("[WRITE_PROMPT] ")

        # The subagent fixes a.py and also edits protected b.py.
        native.write("src/a.py", "a = 2\n")
        native.write("src/b.py", "b = 'clobbered'\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")

        assert code == 3
        first, way_back = out.splitlines()[:2]
        assert first == f"{PROTECTED} apply-fix-batch-0: src/b.py"
        assert f"changed since reviewed commit {native.head[:7]}" in way_back
        assert f"copy {copies}/src/b.py back over src/b.py" in way_back
        assert "--status failed --partial" in way_back
        assert "apply-fix-batch-0" not in (native.state().get("task_status") or {})

    def test_batch_done_refuses_a_write_to_the_context_plan(self, native, capsys):
        """Review fix 3, item 4, on the REAL ``utils.commit_apply``: the
        context plan never changed since ``head`` (so it is not protected),
        but its id was saved with the guard, and a batch that rewrites it is
        refused like a protected write, naming the guard's copy."""
        from utils import commit_apply as real

        native.write("docs/x.md", "# the plan\n")
        _git(native.root, "add", "--", "docs")
        _git(native.root, "commit", "-q", "-m", "plan")
        head = _git(native.root, "rev-parse", "HEAD")
        native.materialize([_batch("src/a.py", "g0")])
        copies = "plans/feature/apply-fixes/protected"
        guard = _guard(head, expect=_head_expect(native.root, head, "src/a.py"), copies=copies)
        guard["context_plan"] = "docs/x.md"
        guard["context_plan_id"] = real.disk_blob_ids(
            ["docs/x.md"], cwd=str(native.root))["docs/x.md"]
        native.write(f"{copies}/docs/x.md", "# the plan\n")
        native.install_guard(guard)

        assert _write_prompt(native, capsys, "apply-fix-batch-0")[0] == 0
        native.write("src/a.py", "a = 2\n")
        native.write("docs/x.md", "# rewritten by the batch\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")

        assert code == 3, err
        first, way_back = out.splitlines()[:2]
        assert first == f"{PROTECTED} apply-fix-batch-0: docs/x.md"
        assert "docs/x.md is the context plan this review checks the commits against" in way_back
        assert f"copy {copies}/docs/x.md back over docs/x.md" in way_back
        assert '--context-plan ""' in way_back
        assert "apply-fix-batch-0" not in (native.state().get("task_status") or {})


def _real_guard(native: NativeRepo, *, protect: Sequence[str] = ("src/b.py",),
                context_plan: Optional[str] = None, drift: bool = True) -> Dict[str, Any]:
    """Install a guard as the real ``take_guard`` leaves it, through the REAL
    ``utils.commit_apply``: *protect* drifted since ``head`` (written here
    unless not *drift*) and saved at their disk ids, the context plan
    (committed at ``head``) watched under ``context_plan_id``, and every one
    copied under ``protected_copies``."""
    from utils import commit_apply as real

    if context_plan is not None:
        native.write(context_plan, "# the plan\n")
        _git(native.root, "add", "--", context_plan)
        _git(native.root, "commit", "-q", "-m", "plan")
        native.head = _git(native.root, "rev-parse", "HEAD")
    for path in protect if drift else ():
        native.write(path, f"{path} drifted\n")
    copies = "plans/feature/apply-fixes/protected"
    guard = _guard(native.head,
                   expect=_head_expect(native.root, native.head, "src/a.py", "src/c.py"),
                   protected=real.disk_blob_ids(list(protect), cwd=str(native.root)),
                   copies=copies)
    watched = list(protect)
    if context_plan is not None:
        guard["context_plan"] = context_plan
        guard["context_plan_id"] = real.disk_blob_ids(
            [context_plan], cwd=str(native.root))[context_plan]
        watched.append(context_plan)
    real.copy_protected(watched, cwd=str(native.root), dest=str(native.root / copies))
    native.install_guard(guard)
    return guard


class TestTheNativeBatchWindow:
    """Review R2, item 1, on the REAL ``utils.commit_apply``: ``util
    batch-done`` compared the watched set with the ids the guard saved, so a
    user's edit to a protected file or to their own context plan between
    batches (or between a stop and its ``--resume``) refused every later
    batch, and the way back undid their edit. ``util write-prompt`` now
    records the watched ids as it binds a batch's prompt, and ``util
    batch-done`` compares with them: the executor's per-batch window."""

    @pytest.mark.parametrize("edited", ["src/b.py", "docs/x.md"],
                             ids=["a protected file", "the context plan"])
    def test_an_edit_between_batches_is_the_users_not_a_violation(
        self, native, capsys, edited,
    ):
        native.materialize([_batch("src/a.py", "g0"), _batch("src/c.py", "g1")])
        _real_guard(native, context_plan="docs/x.md")
        assert _write_prompt(native, capsys, "apply-fix-batch-0")[0] == 0
        assert "apply-fix-batch-0" in native.guard()["bound"]
        native.write("src/a.py", "a = 2\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 0, out + err
        assert "bound" not in native.guard()

        native.write(edited, "the user's edit\n")
        assert _write_prompt(native, capsys, "apply-fix-batch-1")[0] == 0
        native.write("src/c.py", "c = 2\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-1")

        assert code == 0, out + err
        assert out.startswith("[BATCH_DONE] apply-fix-batch-1 completed")
        assert (native.root / edited).read_text(encoding="utf-8") == "the user's edit\n"
        assert native.state()["task_status"]["apply-fix-batch-1"] == "completed"

    def test_a_write_during_the_batch_is_still_refused_and_no_stale_copy_is_offered(
        self, native, capsys,
    ):
        native.materialize([_batch("src/a.py", "g0")])
        copies = _real_guard(native)["protected_copies"]
        # The user edits protected b.py after the guard copied it ...
        native.write("src/b.py", "b = 'the user, later'\n")
        assert _write_prompt(native, capsys, "apply-fix-batch-0")[0] == 0
        # ... and the batch then clobbers it.
        native.write("src/a.py", "a = 2\n")
        native.write("src/b.py", "b = 'clobbered'\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")

        assert code == 3, out + err
        first, way_back = out.splitlines()[:2]
        assert first == f"{PROTECTED} apply-fix-batch-0: src/b.py"
        # The copy holds the guard-time bytes, not the user's edit.
        assert f"copy {copies}/src/b.py" not in way_back
        assert "put src/b.py back as it was before the batch, by hand" in way_back
        state = native.state()
        assert "apply-fix-batch-0" not in (state.get("task_status") or {})
        # Nothing recorded, so the bind record stands for a retry; the failed
        # record the refusal asks for consumes it.
        assert "apply-fix-batch-0" in state["commit_range"]["guard"]["bound"]
        code, out, err = _util(capsys, "batch-done", "--phase-dir", str(native.phase_dir),
                               "--task-id", "apply-fix-batch-0", "--status", "failed",
                               "--partial", "--reason", "it wrote src/b.py")
        assert code == 0, out + err
        assert "bound" not in native.guard()

    def test_without_a_bind_the_saved_ids_stand(self, native, capsys):
        """A prompt not bound through ``util write-prompt`` has no window:
        the comparison is with the ids the guard saved, never a guess."""
        native.materialize([_batch("src/a.py", "g0")])
        _real_guard(native)
        native.write("src/b.py", "b = 'the user, later'\n")
        native.write("src/a.py", "a = 2\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 3, out + err
        assert out.startswith(f"{PROTECTED} apply-fix-batch-0: src/b.py")

    @pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                        reason="needs POSIX modes and a non-root user")
    def test_an_unreadable_protected_file_does_not_refuse_every_batch(self, native, capsys):
        """Review R2, item 2, natively: unreadable when the guard was taken
        and still unreadable is no batch's write; a batch that makes it
        readable is refused."""
        native.materialize([_batch("src/a.py", "g0"), _batch("src/c.py", "g1")])
        secret = native.root / "secret.env"
        secret.write_text("TOKEN=1\n", encoding="utf-8")
        os.chmod(secret, 0)
        try:
            _real_guard(native, protect=("secret.env",), drift=False)
            assert native.guard()["protected"] == {"secret.env": "unreadable"}
            assert _write_prompt(native, capsys, "apply-fix-batch-0")[0] == 0
            native.write("src/a.py", "a = 2\n")
            code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
            assert code == 0, out + err

            assert _write_prompt(native, capsys, "apply-fix-batch-1")[0] == 0
            native.write("src/c.py", "c = 2\n")
            os.chmod(secret, 0o644)
            secret.write_text("TOKEN=2\n", encoding="utf-8")
            code, out, err = _batch_done(native, capsys, "apply-fix-batch-1")
            assert code == 3, out + err
            first, way_back = out.splitlines()[:2]
            assert first == f"{PROTECTED} apply-fix-batch-1: secret.env"
            assert "put secret.env back as it was before the batch, by hand" in way_back
        finally:
            os.chmod(secret, 0o644)


class TestANewFileOnTheNativeLeg:

    def test_created_then_resumed(self, native, capsys, fake_apply):
        tasks = native.materialize([
            _batch("tests/test_new.py", "g0"),
            _batch("tests/test_new.py", "g1"),
            _batch("src/a.py", "g2"),
        ])
        assert [t["expected_ids"] for t in tasks] == [["g0"], ["g1"], ["g2"]]
        expect = _head_expect(native.root, native.head, "src/a.py")
        expect["tests/test_new.py"] = {"kind": "head", "blob": None}
        native.install_guard(_guard(native.head, expect=expect))

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-0")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), err
        native.write("tests/test_new.py", "def test(): pass\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 0, out + err
        created = native.root / "tests/test_new.py"
        assert native.guard()["expect"]["tests/test_new.py"] == {
            "kind": "batch", "sha256": _sha256(created), "task_id": "apply-fix-batch-0"}

        # Cut off here; the user edits a.py, which no batch has touched.
        native.write("src/a.py", "a = 'by hand'\n")
        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-1")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), err

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-2")
        assert code == 0, err
        assert out.startswith(f"{STALE} apply-fix-batch-2: src/a.py changed since "
                              f"reviewed commit {native.head[:7]}")
        assert "[WRITE_PROMPT]" not in out
        assert not _prompt_file(native, "apply-fix-batch-2").exists()
        state = native.state()
        guard = state["commit_range"]["guard"]
        assert guard["skipped"]["g2"] == {"reason": "changed", "files": ["src/a.py"]}
        assert guard["protected"]["src/a.py"] == _disk_blob(native.root, "src/a.py")
        assert state["task_status"]["apply-fix-batch-2"] == "failed"
        assert state["task_status_reasons"]["apply-fix-batch-2"] == (
            f"skipped: src/a.py changed since reviewed commit {native.head[:7]}")
        assert state["processed_apply-fixes"]["g2"]["status"] == "failed"


class TestTheNativeReCheck:

    def test_an_edit_while_batch_1_runs_gets_batch_2_no_prompt(self, native, capsys, fake_apply):
        native.materialize([_batch("src/a.py", "g0"), _batch("src/b.py", "g1"),
                            _batch("src/c.py", "g2")])
        native.install_guard(_guard(native.head, expect=_head_expect(
            native.root, native.head, "src/a.py", "src/b.py", "src/c.py")))

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-0")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), err
        native.write("src/a.py", "a = 2\n")
        # While batch 1 runs, the user edits batch 2's untouched primary file.
        native.write("src/b.py", "b = 'by hand'\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 0, out + err
        guard = native.guard()
        assert guard["expect"]["src/a.py"]["kind"] == "batch"
        # Not adopted by batch 1's record: b.py is not one of its files.
        assert guard["expect"]["src/b.py"]["kind"] == "head"

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-1")
        assert code == 0, err
        assert out.splitlines()[0].startswith(
            f"{STALE} apply-fix-batch-1: src/b.py changed since reviewed commit "
            f"{native.head[:7]} — not dispatched")
        assert not _prompt_file(native, "apply-fix-batch-1").exists()

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-2")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), err

    @pytest.mark.parametrize("edited", [True, False])
    def test_an_external_edit_between_resumes(self, native, capsys, fake_apply, edited):
        native.materialize([_batch("src/a.py", "g0"), _batch("src/a.py", "g1")])
        native.install_guard(_guard(native.head, expect=_head_expect(
            native.root, native.head, "src/a.py")))
        _write_prompt(native, capsys, "apply-fix-batch-0")
        native.write("src/a.py", "a = 2\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 0, out + err
        if edited:
            native.write("src/a.py", "a = 'by hand'\n")
        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-1")
        assert code == 0, err
        if edited:
            assert out.startswith(f"{STALE} apply-fix-batch-1: src/a.py")
        else:
            assert out.startswith("[WRITE_PROMPT] ")

    def test_a_held_back_group_is_recorded_with_the_executors_keys(
        self, native, capsys, fake_apply,
    ):
        native.materialize([
            _batch("src/a.py", "g1", "g2", files={"g2": ["src/a.py", "src/b.py"]}),
            _batch("src/c.py", "g3"),
        ])
        native.install_guard(_guard(native.head, expect=_head_expect(
            native.root, native.head, "src/a.py", "src/b.py", "src/c.py")))
        native.write("src/b.py", "b = 'by hand'\n")

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-0")
        assert code == 0, err
        assert out.startswith(f"{STALE} apply-fix-batch-0: src/b.py changed since")
        assert "1 fix(es) skipped, 1 held back for a --resume" in out
        state = native.state()
        assert state["task_status"]["apply-fix-batch-0"] == "failed"
        assert state["task_status_reasons"]["apply-fix-batch-0"] == (
            "held back: batched with src/b.py")
        # Both items stay in a native resume's work set (`failed`); the
        # guard's saved skip takes g2 out there.
        assert state["processed_apply-fixes"]["g1"]["status"] == "failed"
        assert state["processed_apply-fixes"]["g2"]["status"] == "failed"
        assert state["commit_range"]["guard"]["skipped"] == {
            "g2": {"reason": "changed", "files": ["src/b.py"]}}

        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-1")
        assert code == 0 and out.startswith("[WRITE_PROMPT] "), err


class TestNoGuardOnTheNativeLeg:

    def test_write_prompt_and_batch_done_are_unchanged(self, native, capsys, tripwire):
        native.materialize([_batch("src/a.py", "g0")])
        native.write("src/b.py", "b = 'by hand'\n")
        state_before = native.state_path.read_bytes()
        code, out, err = _write_prompt(native, capsys, "apply-fix-batch-0")
        assert code == 0, err
        # Binding a prompt writes nothing to state.json without a guard.
        assert native.state_path.read_bytes() == state_before
        assert [line.split(" ", 1)[0] for line in out.splitlines()
                if not line.startswith("MULTI_LLM_RESULT")] == [
            "[WRITE_PROMPT]", "[WRITE_PROMPT_LINES]", "[WRITE_PROMPT_CONTEXT]"]
        native.write("src/a.py", "a = 2\n")
        native.write("src/b.py", "b = 2\n")
        code, out, err = _batch_done(native, capsys, "apply-fix-batch-0")
        assert code == 0, out + err
        assert out.startswith("[BATCH_DONE] apply-fix-batch-0 completed")
        assert native.state()["task_status"]["apply-fix-batch-0"] == "completed"
