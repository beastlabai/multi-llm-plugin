"""The contract for the names `utils.git_utils` re-exports from `utils.git_diff`
and `utils.review_diff` (stated in git_utils' module docstring):

* **Reads work.** `git_utils.X`, `from utils.git_utils import X`, `dir()`,
  `inspect.getmembers` and `create_autospec(git_utils)` see every moved name,
  as its owner's current object.
* **Writes do not.** A patch or assignment of a moved name on `git_utils` only
  shadows it there and reaches no moved code; `del` raises. The guard at the
  end of this file fails any test that writes one, naming the owner to patch.
* **The runner seam holds.** The owners run every git command through the
  `git_utils` now in `sys.modules`, looked up at each call, so a refusal
  injected on `git_utils._run_git_process` reaches the moved code, also after
  a failed import and its retry replaced `git_utils`.
* **No import waits on another.** None of the three modules imports another
  while it initialises, so no order of importing them, in one thread or
  several, can deadlock on their import locks.

What stays: Python takes a submodule's import lock before it imports the
parent package, and `utils/__init__.py` imports `git_utils` (and, through
`provider_registry`, `review_diff`). Python 3.14 also makes that submodule
import wait for a parent that is still initialising. So on 3.14 two threads
importing two *different* `utils` submodules into a cold process can deadlock
in `utils/__init__`, with or without the split: measured on the code before
it, threads importing `utils.git_utils`, `utils.next_phase` and
`utils.html_report_generator` deadlock in 37 of 40 fresh processes on 3.14
(none in 40 on 3.10-3.13), and with `utils.llm_client` and
`utils.provider_registry` instead in 40 of 40 on every version. The
concurrency tests below cover what these three modules decide; the package's
eager imports are not theirs to change.
"""

import ast
import inspect
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils import git_diff, git_utils, review_diff  # noqa: E402

OWNERS = {"git_diff": git_diff, "review_diff": review_diff}
MODULES = ("git_utils", "git_diff", "review_diff")

#: The git_utils names the owners use, all read through ``_git_utils()``.
KEPT_HELPERS = frozenset({
    "_NUMSTAT_CHUNK", "_run_git", "_run_git_process", "attribute_source_args",
    "git_argv", "git_path", "is_empty_tree", "is_empty_tree_id",
})


def _source(module_name):
    return (SKILL_ROOT / "utils" / f"{module_name}.py").read_text()


def _top_level_definitions(module_name):
    names = set()
    for node in ast.parse(_source(module_name)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("hello\n")
    return str(path)


#: Each owner's own definitions, as this file found them.
_DEFINITIONS = {
    name: vars(OWNERS[owner])[name] for name, owner in git_utils._MOVED_TO.items()
}


def _assert_intact(name):
    """Both modules as they were: the owner binds its own definition, git_utils
    binds nothing of its own and reads the owner's."""
    owner = OWNERS[git_utils._MOVED_TO[name]]
    assert vars(owner)[name] is _DEFINITIONS[name], name
    assert name not in vars(git_utils), name
    assert getattr(git_utils, name) is _DEFINITIONS[name], name


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTheMovedNamesTable:
    @pytest.mark.parametrize("owner", sorted(OWNERS))
    def test_it_lists_exactly_what_each_owner_defines(self, owner):
        assert set(git_utils._MOVED[owner]) == _top_level_definitions(owner) - {"_git_utils"}

    def test_git_utils_defines_none_of_them(self):
        assert not _top_level_definitions("git_utils") & set(git_utils._MOVED_TO)
        assert len(git_utils._MOVED_TO) == 81

    def test_the_helpers_the_owners_use_stay_in_git_utils(self):
        assert not KEPT_HELPERS & set(git_utils._MOVED_TO)
        assert KEPT_HELPERS <= set(vars(git_utils))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_every_moved_name_reads_as_its_owners_object(self):
        for name in git_utils._MOVED_TO:
            _assert_intact(name)

    def test_from_import(self):
        namespace = {}
        exec("from utils.git_utils import capture_diff_hunks, write_review_diff", namespace)
        assert namespace["capture_diff_hunks"] is git_diff.capture_diff_hunks
        assert namespace["write_review_diff"] is review_diff.write_review_diff

    def test_a_patch_on_the_owner_shows_through_git_utils(self):
        with mock.patch.object(review_diff, "write_review_diff") as fake:
            assert git_utils.write_review_diff is fake
        with mock.patch("utils.git_diff._is_binary_file") as fake:
            assert git_utils._is_binary_file is fake
        _assert_intact("write_review_diff")
        _assert_intact("_is_binary_file")

    def test_an_unknown_name_is_an_attribute_error_naming_git_utils(self):
        with pytest.raises(AttributeError, match=r"'utils\.git_utils' has no attribute '_u61a_absent'"):
            git_utils._u61a_absent
        assert not hasattr(git_utils, "_u61a_absent")

    def test_a_moved_name_deleted_from_its_owner_is_gone_from_git_utils_too(self):
        original = git_diff._is_binary_file
        try:
            del git_diff._is_binary_file
            with pytest.raises(AttributeError, match=r"'utils\.git_utils' has no attribute"):
                git_utils._is_binary_file
        finally:
            git_diff._is_binary_file = original
        _assert_intact("_is_binary_file")


class TestMovedNamesAreListed:
    def test_dir(self):
        assert set(git_utils._MOVED_TO) <= set(dir(git_utils))
        assert set(vars(git_utils)) <= set(dir(git_utils))

    def test_inspect_getmembers(self):
        members = dict(inspect.getmembers(git_utils))
        for name, owner in git_utils._MOVED_TO.items():
            assert members[name] is getattr(OWNERS[owner], name), name

    def test_create_autospec_of_the_module(self, text_file):
        spec = mock.create_autospec(git_utils)
        for name in git_utils._MOVED_TO:
            assert hasattr(spec, name), name
        spec.capture_file_context(text_file, [1, 1])
        spec.capture_file_context.assert_called_once_with(text_file, [1, 1])
        with pytest.raises(TypeError):
            spec.capture_file_context()

    def test_a_mock_specced_on_the_module(self):
        fake = mock.Mock(spec=git_utils)
        for name in git_utils._MOVED_TO:
            assert hasattr(fake, name), name


# ---------------------------------------------------------------------------
# Writes through git_utils: shadows that reach no moved code (unsupported)
# ---------------------------------------------------------------------------


class TestAWriteOnGitUtilsStaysThere:
    """The forms the guard below refuses, pinned so their effect stays the
    documented one: nothing reaches the moved code and both modules end as they
    started."""

    def test_patch_object_shadows_the_name_only_in_git_utils(self, text_file):
        with mock.patch.object(git_utils, "_is_binary_file", return_value=True) as fake:
            assert git_utils._is_binary_file is fake
            assert git_diff._is_binary_file is not fake
            assert git_utils.capture_file_context(text_file, [1, 1]) == [
                {"line_number": 1, "content": "hello"}
            ]
        assert fake.call_count == 0
        _assert_intact("_is_binary_file")

    @pytest.mark.parametrize("by_path", [False, True], ids=["patch.object", "patch"])
    def test_create_true_leaves_both_modules_intact(self, by_path, text_file):
        def patcher():
            if by_path:
                return mock.patch("utils.git_utils._is_binary_file", create=True)
            return mock.patch.object(git_utils, "_is_binary_file", create=True)

        with patcher() as outer:
            with patcher() as inner:
                assert git_utils._is_binary_file is inner
            assert git_utils._is_binary_file is outer
        _assert_intact("_is_binary_file")
        assert git_utils.capture_file_context(text_file, [1, 1]) == [
            {"line_number": 1, "content": "hello"}
        ]

    def test_create_true_inside_an_owner_patch(self):
        with mock.patch.object(git_diff, "_is_binary_file") as on_owner:
            with mock.patch.object(git_utils, "_is_binary_file", create=True) as shadow:
                assert git_utils._is_binary_file is shadow
                assert git_diff._is_binary_file is on_owner
            assert git_utils._is_binary_file is on_owner
        _assert_intact("_is_binary_file")

    def test_del_and_monkeypatch_delattr_raise(self):
        """`monkeypatch.delattr` records the value it read before deleting, so
        its undo, after the raise, leaves a copy behind as `setattr`'s does."""
        with pytest.raises(AttributeError):
            del git_utils.capture_diff_hunks
        _assert_intact("capture_diff_hunks")
        patcher = pytest.MonkeyPatch()
        try:
            with pytest.raises(AttributeError):
                patcher.delattr(git_utils, "capture_diff_hunks")
            assert git_diff.capture_diff_hunks is _DEFINITIONS["capture_diff_hunks"]
            patcher.undo()
            assert vars(git_utils)["capture_diff_hunks"] is _DEFINITIONS["capture_diff_hunks"]
        finally:
            vars(git_utils).pop("capture_diff_hunks", None)
        _assert_intact("capture_diff_hunks")

    def test_monkeypatch_setattr_leaves_a_copy_behind(self):
        """pytest's undo sets the value it read back, a binding of git_utils'
        own that later owner patches no longer show through."""
        original = git_diff.capture_diff_hunks
        patcher = pytest.MonkeyPatch()
        try:
            patcher.setattr(git_utils, "capture_diff_hunks", None)
            assert git_diff.capture_diff_hunks is original
            patcher.undo()
            assert vars(git_utils)["capture_diff_hunks"] is original
        finally:
            vars(git_utils).pop("capture_diff_hunks", None)
        _assert_intact("capture_diff_hunks")


# ---------------------------------------------------------------------------
# The runner seam
# ---------------------------------------------------------------------------


class TestARunnerPatchOnGitUtilsReachesTheMovedCode:
    @pytest.mark.parametrize("by_path", [False, True], ids=["patch.object", "patch"])
    def test_an_injected_refusal_reaches_review_diff(self, by_path, tmp_path):
        patcher = (
            mock.patch("utils.git_utils._run_git_process", side_effect=OSError("refused"))
            if by_path
            else mock.patch.object(git_utils, "_run_git_process", side_effect=OSError("refused"))
        )
        with patcher as fake:
            assert review_diff._git_diff_output(["--version"], str(tmp_path)) == (
                None, "git could not run (refused)"
            )
            assert git_utils._git_diff_output(["--version"], str(tmp_path)) == (
                None, "git could not run (refused)"
            )
        assert fake.call_count == 2

    def test_an_injected_refusal_reaches_git_diff(self, tmp_path):
        with mock.patch.object(
            git_utils, "_run_git_process", side_effect=OSError("refused")
        ) as fake:
            result = git_diff.capture_diff_hunks(
                "HEAD~1", ["a.py"], head_ref="HEAD", cwd=str(tmp_path)
            )
        assert fake.call_count == 1
        assert result["_notices"] == [
            "Git not available -- code context shown from file content only."
        ]

    def test_run_git_and_git_argv_reach_the_moved_code(self, tmp_path):
        with mock.patch.object(git_utils, "_run_git", return_value=("git version fake", "", 0)) as fake:
            assert review_diff._git_version_text(str(tmp_path)) == "git version fake"
        assert fake.call_count == 1
        with mock.patch.object(git_utils, "git_argv", side_effect=OSError("refused")) as fake:
            assert review_diff._linguist_generated(["a.py"], cwd=str(tmp_path)) == set()
            assert git_diff.capture_diff_hunks("HEAD~1", ["a.py"], cwd=str(tmp_path))["_notices"]
        assert fake.call_count == 2

    def test_runner_names_planted_in_an_owner_divert_nothing(self, tmp_path):
        def stale(*args, **kwargs):
            raise AssertionError("a runner planted in an owner ran")

        with mock.patch.object(git_diff, "_run_git_process", stale, create=True), \
                mock.patch.object(review_diff, "_run_git_process", stale, create=True), \
                mock.patch.object(git_utils, "_run_git_process", side_effect=OSError("refused")) as fake:
            assert review_diff._git_diff_output(["--version"], str(tmp_path))[0] is None
            git_diff.capture_diff_hunks("HEAD~1", ["a.py"], head_ref="HEAD", cwd=str(tmp_path))
        assert fake.call_count == 2

    @pytest.mark.parametrize("owner", sorted(OWNERS))
    def test_the_owner_reads_git_utils_only_through_the_helper(self, owner):
        """No module-level `git_utils` binding (it would go stale when a failed
        import and its retry replace the module) and no copied name (a patch of
        the original would never reach it): every use is `_git_utils().X`."""
        tree = ast.parse(_source(owner))
        imports = [
            ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.Import) and any(a.name.endswith("git_utils") for a in node.names)
            or isinstance(node, ast.ImportFrom) and (
                (node.module or "").endswith("git_utils")
                or any(a.name == "git_utils" for a in node.names)
            )
        ]
        assert imports == []
        assert "git_utils" not in vars(OWNERS[owner])
        used = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "_git_utils"
        }
        assert used and used <= KEPT_HELPERS
        loads = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assert not KEPT_HELPERS & loads


# ---------------------------------------------------------------------------
# Imports, threads, failed imports and reloads, each in a fresh process
# ---------------------------------------------------------------------------


def _import_time_nodes(tree):
    """The statements a module runs while it is imported: its body, class
    bodies and compound statements in it, not function bodies."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            pending.extend(
                child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)
            )


def test_no_module_imports_another_while_it_initialises():
    """Import-time code of the three imports only the standard library; the
    one import of an owner is git_utils' `__getattr__`, which runs on use."""
    for name in MODULES:
        package_imports = [
            ast.unparse(node) for node in _import_time_nodes(ast.parse(_source(name)))
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("utils"))
            or isinstance(node, ast.Import) and any(a.name.startswith("utils") for a in node.names)
        ]
        assert package_imports == [], name
    tree = ast.parse(_source("git_utils"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "importlib.import_module"
    ]
    getattr_def = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )
    assert [ast.literal_eval(call.args[0]) for call in calls] == [".git_diff", ".review_diff"]
    assert all(call in list(ast.walk(getattr_def)) for call in calls)


def _run(script, *args, timeout=60):
    return subprocess.run(
        [sys.executable, "-I", "-c", script, str(SKILL_ROOT), *map(str, args)],
        capture_output=True, text=True, timeout=timeout,
    )


def _run_many(script, args, attempts):
    with ThreadPoolExecutor(max_workers=8) as pool:
        procs = list(pool.map(lambda _: _run(script, *args), range(attempts)))
    failures = [p.stdout + p.stderr for p in procs if p.returncode != 0 or p.stdout.strip() != "ok"]
    assert failures == [], f"{len(failures)} of {attempts} failed; first:\n{failures[0]}"


#: argv: skill root, import order, "package" or "bare", a text file.
_ORDER_PROBE = r"""
import importlib, sys, types
from unittest import mock
skill_root, order, package, text_file = sys.argv[1:5]
sys.path.insert(0, skill_root)
if package == "bare":
    # utils/__init__.py imports git_utils first (and review_diff through
    # provider_registry); a bare package lets the order below be the order
    # the three modules really load in.
    bare = types.ModuleType("utils")
    bare.__path__ = [skill_root + "/utils"]
    sys.modules["utils"] = bare
order = order.split(",")
for position, name in enumerate(order):
    importlib.import_module("utils." + name)
    if package == "bare":
        loaded = {n for n in ("git_utils", "git_diff", "review_diff") if "utils." + n in sys.modules}
        assert loaded == set(order[:position + 1]), (order, loaded)
from utils import git_utils as g, git_diff as d, review_diff as r
owners = {"git_diff": d, "review_diff": r}
for name, owner in g._MOVED_TO.items():
    assert getattr(g, name) is vars(owners[owner])[name] and name not in vars(g), name
assert set(g._MOVED_TO) <= set(dir(g))
assert "git_utils" not in vars(d) and "git_utils" not in vars(r)
with mock.patch.object(g, "_run_git_process", side_effect=OSError("refused")) as fake:
    assert r._git_diff_output(["--version"], None) == (None, "git could not run (refused)")
    assert d.capture_diff_hunks("HEAD~1", ["a.py"], head_ref="HEAD")["_notices"]
assert fake.call_count == 2
with mock.patch.object(g, "_is_binary_file", return_value=True, create=True):
    pass
assert "_is_binary_file" not in vars(g) and g._is_binary_file is vars(d)["_is_binary_file"]
assert d.capture_file_context(text_file, [1, 1]) == [{"line_number": 1, "content": "hello"}]
print("ok")
"""


@pytest.mark.parametrize("package", ["package", "bare"])
@pytest.mark.parametrize("order", [",".join(p) for p in itertools.permutations(MODULES)])
def test_every_import_order_in_a_fresh_process(order, package, text_file):
    proc = _run(_ORDER_PROBE, order, package, text_file)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


#: argv: skill root, "package" or "bare", the modules the threads import.
_THREADS_PROBE = r"""
import importlib, sys, threading, traceback, types
skill_root, package, modules = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sys.path.insert(0, skill_root)
if package == "bare":
    bare = types.ModuleType("utils")
    bare.__path__ = [skill_root + "/utils"]
    sys.modules["utils"] = bare
start = threading.Barrier(len(modules))
errors = []
def worker(name):
    try:
        start.wait(timeout=10)
        importlib.import_module("utils." + name)
        g = importlib.import_module("utils.git_utils")
        for moved in g._MOVED_TO:
            getattr(g, moved)
    except BaseException:
        errors.append(traceback.format_exc())
threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in modules]
for t in threads:
    t.start()
for t in threads:
    t.join(20)
assert not any(t.is_alive() for t in threads), "hung"
assert not errors, errors[0]
print("ok")
"""


def test_concurrent_cold_imports_of_git_utils_read_every_moved_name():
    """Three threads importing the old module and reading all 81 names while
    neither owner is loaded: the scenario before the split had, and passes."""
    _run_many(_THREADS_PROBE, ["package", "git_utils,git_utils,git_utils"], attempts=40)


def test_concurrent_cold_imports_of_the_three_modules():
    """The three modules' own import graph, with no package `__init__` to take
    part."""
    _run_many(_THREADS_PROBE, ["bare", "git_utils,git_diff,review_diff"], attempts=40)


@pytest.mark.skipif(sys.version_info >= (3, 14), reason=(
    "Python 3.14 makes a submodule import wait for its still-initialising parent "
    "package while it holds its own import lock: threads importing different utils "
    "submodules into a cold process deadlock in utils/__init__.py, before the split "
    "too (module docstring); the lock-window test covers 3.14"
))
def test_concurrent_cold_imports_of_the_three_modules_with_the_real_package():
    _run_many(_THREADS_PROBE, ["package", "git_utils,git_diff,review_diff"], attempts=40)


#: argv: skill root. git_utils' thread is held while it owns the `utils`
#: package lock, before `utils` is in sys.modules, until git_diff's thread is
#: waiting for that lock while it holds git_diff's own: git_utils must then
#: finish initialising without needing git_diff.
_LOCK_WINDOW_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, sys, threading, time, traceback
skill_root = sys.argv[1]
sys.path.insert(0, skill_root)
holding, release = threading.Event(), threading.Event()

class HoldThePackageLock(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "utils" or threading.current_thread().name != "git_utils":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        real = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):  # runs under the module lock only
                holding.set()
                release.wait(20)
                return None
            def exec_module(self, module):
                real.exec_module(module)
        spec.loader = Loader()
        return spec

sys.meta_path.insert(0, HoldThePackageLock())
errors = []
def worker(name):
    try:
        importlib.import_module("utils." + name)
    except BaseException:
        errors.append(traceback.format_exc())

def waiting_on_a_lock(thread):
    seen, deadline = 0, time.monotonic() + 10
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(thread.ident)
        code = frame.f_code if frame is not None else None
        seen = seen + 1 if code and code.co_name == "acquire" and "_bootstrap" in code.co_filename else 0
        if seen >= 3:
            return True
        time.sleep(0.005)
    return False

first = threading.Thread(target=worker, args=("git_utils",), name="git_utils", daemon=True)
first.start()
assert holding.wait(20)
second = threading.Thread(target=worker, args=("git_diff",), name="git_diff", daemon=True)
second.start()
staged = waiting_on_a_lock(second)
release.set()
for t in (first, second):
    t.join(20)
assert not first.is_alive() and not second.is_alive(), "hung"
assert staged, "git_diff's import never waited on the package lock"
assert not errors, errors[0]
print("ok")
"""


def test_git_utils_initialising_the_package_never_waits_on_git_diff():
    _run_many(_LOCK_WINDOW_PROBE, [], attempts=5)


#: argv: skill root, "package" or "bare", "owner" (the first import of
#: review_diff fails) or "replaced" (git_utils is re-imported after both
#: owners loaded, and that import fails once after running the module).
_RETRY_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, sys, types
from unittest import mock
skill_root, package, scenario = sys.argv[1:4]
sys.path.insert(0, skill_root)
if package == "bare":
    bare = types.ModuleType("utils")
    bare.__path__ = [skill_root + "/utils"]
    sys.modules["utils"] = bare

class FailOnce(importlib.abc.MetaPathFinder):
    def __init__(self, name, after_exec):
        self.name, self.after_exec, self.made = name, after_exec, []
    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        if not self.after_exec:
            raise RuntimeError("injected import failure")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        real, made = spec.loader, self.made
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                real.exec_module(module)
                made.append(module)
                raise RuntimeError("injected import failure")
        spec.loader = Loader()
        return spec

obsolete = None
if scenario == "owner":
    sys.meta_path.insert(0, FailOnce("utils.review_diff", after_exec=False))
    try:
        importlib.import_module("utils.git_utils").write_review_diff
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected failure never happened")
else:
    for name in ("git_utils", "git_diff", "review_diff"):
        importlib.import_module("utils." + name)
    del sys.modules["utils.git_utils"]
    failing = FailOnce("utils.git_utils", after_exec=True)
    sys.meta_path.insert(0, failing)
    try:
        importlib.import_module("utils.git_utils")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected failure never happened")
    (obsolete,) = failing.made
    assert "utils.git_utils" not in sys.modules

g = importlib.import_module("utils.git_utils")
assert g is not obsolete
d = importlib.import_module("utils.git_diff")
r = importlib.import_module("utils.review_diff")
owners = {"git_diff": d, "review_diff": r}
for name, owner in g._MOVED_TO.items():
    assert getattr(g, name) is vars(owners[owner])[name], name
stale = mock.Mock(side_effect=AssertionError("the obsolete git_utils ran a command"))
with mock.patch.object(g, "_run_git_process", side_effect=OSError("refused")) as current:
    if obsolete is not None:
        obsolete._run_git_process = stale
    assert r._git_diff_output(["--version"], None) == (None, "git could not run (refused)")
    assert d.capture_diff_hunks("HEAD~1", ["a.py"], head_ref="HEAD")["_notices"]
assert (current.call_count, stale.call_count) == (2, 0)
print("ok")
"""


@pytest.mark.parametrize("scenario", ["owner", "replaced"])
@pytest.mark.parametrize("package", ["package", "bare"])
def test_a_failed_import_and_its_retry_leave_the_runner_seam_on_the_current_module(
    package, scenario
):
    proc = _run(_RETRY_PROBE, package, scenario)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


#: argv: skill root. The real package.
_RELOAD_PROBE = r"""
import importlib, sys
from unittest import mock
sys.path.insert(0, sys.argv[1])
from utils import git_utils as g, git_diff as d, review_diff as r
owners = {"git_diff": d, "review_diff": r}

def coherent():
    for name, owner in g._MOVED_TO.items():
        assert name not in vars(g) and getattr(g, name) is getattr(owners[owner], name), name
    assert set(g._MOVED_TO) <= set(dir(g))

def refusal_reaches_both():
    with mock.patch.object(g, "_run_git_process", side_effect=OSError("refused")) as fake:
        assert r._git_diff_output(["--version"], None) == (None, "git could not run (refused)")
        assert d.capture_diff_hunks("HEAD~1", ["a.py"], head_ref="HEAD")["_notices"]
    assert fake.call_count == 2

# A reload of any of the three, in any order, keeps reads and the runner seam.
for module in (g, d, r, d, g, r):
    assert importlib.reload(module) is module
    coherent()
    refusal_reaches_both()

# As before the split, with the reload of the module that defines the name:
# a deleted definition comes back; an active patch is overwritten by the
# fresh definition and its exit puts the original back.
for name, owner in (("_is_binary_file", d), ("_git_diff_output", r)):
    delattr(owner, name)
    assert not hasattr(g, name)
    importlib.reload(owner)
    coherent()
    original = getattr(owner, name)
    with mock.patch.object(owner, name) as fake:
        importlib.reload(owner)
        assert getattr(owner, name) is not fake
        assert getattr(g, name) is getattr(owner, name)
    assert getattr(owner, name) is original is getattr(g, name)
    importlib.reload(owner)
    coherent()

# A runner patch on git_utils: an owner's reload keeps it reaching the moved
# code; git_utils' reload overwrites it with the fresh definition and its exit
# puts the original back, as before the split.
original = g._run_git_process
with mock.patch.object(g, "_run_git_process", side_effect=OSError("refused")) as fake:
    importlib.reload(d)
    importlib.reload(r)
    assert r._git_diff_output(["--version"], None) == (None, "git could not run (refused)")
    assert fake.call_count == 1
    importlib.reload(g)
    assert g._run_git_process not in (fake, original)
assert g._run_git_process is original
importlib.reload(g)
coherent()
refusal_reaches_both()

# Different from before the split, in the forms the guard refuses: `del` of a
# moved name on git_utils raises (it deleted the definition, which git_utils'
# reload brought back), and a patch of one on git_utils is a shadow there that
# git_utils' reload does not re-define (it used to be overwritten); both
# modules end intact either way.
try:
    del g._is_binary_file
except AttributeError:
    pass
else:
    raise AssertionError("del of a moved name on git_utils succeeded")
with mock.patch.object(g, "_is_binary_file") as fake:
    importlib.reload(g)
    assert g._is_binary_file is fake and d._is_binary_file is not fake
coherent()
refusal_reaches_both()
print("ok")
"""


def test_reloads_in_a_fresh_process():
    proc = _run(_RELOAD_PROBE)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on git_utils
# ---------------------------------------------------------------------------

def _guard_message(path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.git_utils, but {name} lives in "
        f"utils/{owner}.py and git_utils only re-exports it for reading: the code that "
        f"uses it never sees this patch. Patch utils.{owner}.{name} instead."
    )


def test_no_test_writes_a_moved_name_on_git_utils():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(git_utils._MOVED_TO, "git_utils")
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path, hit in scanner.scan_dir(Path(__file__).parent, skip=Path(__file__))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom unittest.mock import patch\nfrom utils import git_utils\n"


class TestTheGuardSeesEveryForm:
    @pytest.mark.parametrize("snippet, name", [
        ('@patch("utils.git_utils.capture_file_context")\ndef test(m): pass', "capture_file_context"),
        ('with mock.patch("utils.git_utils.write_review_diff"): pass', "write_review_diff"),
        ('mocker.patch("utils.git_utils.write_review_diff")', "write_review_diff"),
        ('mock.patch.object(git_utils, "_is_binary_file", create=True)', "_is_binary_file"),
        ('from utils import git_utils as gu\npatch.object(gu, "classify_review_paths")', "classify_review_paths"),
        ('import utils.git_utils\npatch.object(utils.git_utils, "classify_review_paths")', "classify_review_paths"),
        ('import utils.git_utils as g\nmonkeypatch.setattr(g, "write_review_diff", None)', "write_review_diff"),
        ('monkeypatch.setattr("utils.git_utils.write_review_diff", None)', "write_review_diff"),
        ('monkeypatch.delattr(git_utils, "capture_diff_hunks")', "capture_diff_hunks"),
        ('monkeypatch.delattr("utils.git_utils.capture_diff_hunks")', "capture_diff_hunks"),
        ('git_utils._is_binary_file = lambda path: True', "_is_binary_file"),
        ('a, git_utils._is_binary_file = 1, 2', "_is_binary_file"),
        ('setattr(git_utils, "_is_binary_file", None)', "_is_binary_file"),
        ('del git_utils.capture_diff_hunks', "capture_diff_hunks"),
        ('patch.multiple(git_utils, capture_diff_hunks=mock.DEFAULT)', "capture_diff_hunks"),
        ('patch.multiple("utils.git_utils", capture_diff_hunks=mock.DEFAULT)', "capture_diff_hunks"),
        ('patch.dict(vars(git_utils), {"capture_diff_hunks": None})', "capture_diff_hunks"),
        ('vars(git_utils)["capture_diff_hunks"] = None', "capture_diff_hunks"),
        ('import importlib\ng = importlib.import_module("utils.git_utils")\ng.capture_diff_hunks = None',
         "capture_diff_hunks"),
        ('import sys\nsys.modules["utils.git_utils"]._is_binary_file = None', "_is_binary_file"),
    ])
    def test_a_write(self, snippet, name):
        hits = MovedNameWriteScanner(git_utils._MOVED_TO, "git_utils").scan(_IMPORTS + snippet)
        assert [hit[2] for hit in hits] == [name]
        owner = git_utils._MOVED_TO[name]
        assert hits[0][3] == owner
        assert f"Patch utils.{owner}.{name} instead" in _guard_message("t.py", *hits[0])

    @pytest.mark.parametrize("snippet", [
        'patch.object(git_utils, "_run_git_process")',
        'monkeypatch.setattr(git_utils, "git_argv", None)',
        'patch("utils.git_utils.subprocess.run")',
        'patch("utils.git_utils.ReviewDiffSet.write")',
        'patch("utils.review_diff.write_review_diff")',
        'patch("utils.git_diff._is_binary_file", create=True)',
        'from utils import review_diff\nmonkeypatch.setattr(review_diff, "write_review_diff", None)',
        'found = git_utils.capture_diff_hunks("HEAD", [])',
        'git_utils.ReviewDiffSet.attribute = 1',
        'other.setattr(thing, "capture_diff_hunks", 1)',
    ])
    def test_not_a_write(self, snippet):
        assert MovedNameWriteScanner(git_utils._MOVED_TO, "git_utils").scan(_IMPORTS + snippet) == []
