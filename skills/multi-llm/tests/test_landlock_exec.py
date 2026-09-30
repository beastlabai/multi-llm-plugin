#!/usr/bin/env python3
"""The ``plugin-landlock`` wrapper, ``utils/landlock_exec.py`` (plan
``codex-sandbox-unavailable-findings-and-fixes`` §8.1–8.2).

Two tiers, because Landlock is ENOSYS in the Docker image and may be missing, or
below ABI 3, on a CI kernel, while the wrapper's decisions must be tested
everywhere:

* **Tier 1, kernel-independent.** :func:`landlock_exec.main` driven through its
  seam with :class:`FakeKernel`, which returns a chosen ABI or errno and records
  every ruleset call. This is where the decisions live: exit 126 when Landlock
  is absent, the ABI is below the floor or the machine is unknown; the
  required-entry exit *before any ruleset exists*; optional entries skipped with
  a note; dir vs file rights; the mask derived from the ABI; no path ever
  created. The rows that open real entries need Linux's ``O_PATH`` and are
  skipped elsewhere: the wrapper only ever runs on Linux.
* **Tier 2, the real kernel** (``landlock`` marker). The script run as the parent
  runs it, ``sys.executable -I -S landlock_exec.py …``, around ``sh`` and
  ``python`` commands in a ``tmp_path`` scratch tree. Each denial is proven by
  the target staying byte-identical, on the same inode, with no new entry. The
  rows skip unless the ABI query returns >= 3, naming the errno or ABI found,
  and ``MULTI_LLM_REQUIRE_LANDLOCK=1`` turns that skip into a failure so a
  kernel regression cannot hide behind it.
"""

import ast
import errno
import functools
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import landlock_exec as lx  # noqa: E402

WRAPPER = SKILL_ROOT / "utils" / "landlock_exec.py"

#: Set on a host that must run tier 2: a skip there fails instead.
REQUIRE_ENV = "MULTI_LLM_REQUIRE_LANDLOCK"

#: The entry rows open real paths with ``O_PATH``, which only Linux has.
linux_only = pytest.mark.skipif(
    not hasattr(os, "O_PATH") or not os.path.isdir("/proc/self/fd"),
    reason="the wrapper's entry check is Linux-only (O_PATH, /proc/self/fd)",
)


# ===========================================================================
# Tier 1: the seam
# ===========================================================================

def _fd_facts(fd: int):
    """``(path, open flags)`` of a live descriptor, read back from ``/proc``."""
    target = os.readlink(f"/proc/self/fd/{fd}")
    flags = 0
    with open(f"/proc/self/fdinfo/{fd}", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("flags:"):
                flags = int(line.split()[1], 8)
    return target, flags


class FakeKernel:
    """Stands in for :class:`landlock_exec.Kernel`: a scripted ABI (or errno),
    a scripted failure per call, and a log of every call in order."""

    #: Never a real descriptor; nothing but this fake ever sees it.
    RULESET_FD = 7001

    def __init__(self, *, machine="x86_64", abi=7, fail=None):
        self._machine = machine
        self._abi = abi
        self._fail = dict(fail or {})
        self.calls = []

    def machine(self):
        self.calls.append(("machine",))
        return self._machine

    def abi(self):
        self.calls.append(("abi",))
        if isinstance(self._abi, BaseException):
            raise self._abi
        return self._abi

    def create_ruleset(self, mask):
        self.calls.append(("create_ruleset", mask))
        self._maybe_fail("create_ruleset")
        return self.RULESET_FD

    def add_rule(self, ruleset_fd, rights, path_fd):
        target, flags = _fd_facts(path_fd)
        self.calls.append(("add_rule", ruleset_fd, rights, target, flags))
        self._maybe_fail("add_rule")

    def no_new_privs(self):
        self.calls.append(("no_new_privs",))
        self._maybe_fail("no_new_privs")

    def restrict(self, ruleset_fd):
        self.calls.append(("restrict", ruleset_fd))
        self._maybe_fail("restrict")

    def _maybe_fail(self, name):
        if name in self._fail:
            raise self._fail[name]

    # -- views ---------------------------------------------------------------

    @property
    def names(self):
        return [call[0] for call in self.calls]

    @property
    def jail_calls(self):
        """Every call that builds or applies a ruleset."""
        return [
            call for call in self.calls
            if call[0] in ("create_ruleset", "add_rule", "no_new_privs", "restrict")
        ]

    @property
    def rules(self):
        """``{path: rights}`` for every rule added."""
        return {call[3]: call[2] for call in self.calls if call[0] == "add_rule"}


class Run:
    """One :func:`landlock_exec.main` call through the seam."""

    def __init__(self, argv, *, kernel=None, exec_error=None):
        self.kernel = kernel or FakeKernel()
        self.exec_calls = []
        stderr = io.StringIO()

        def execvp(file, args):
            self.kernel.calls.append(("execvp", file, list(args)))
            self.exec_calls.append((file, list(args)))
            if exec_error is not None:
                raise exec_error

        self.code = lx.main(argv, kernel=self.kernel, execvp=execvp, stderr=stderr)
        self.stderr = stderr.getvalue()

    @property
    def lines(self):
        return self.stderr.splitlines()


def _allow(*entries, optional=()):
    argv = []
    for spec in entries:
        argv += ["--allow", str(spec)]
    for spec in optional:
        argv += ["--allow-optional", str(spec)]
    return argv


def _snapshot(root: Path):
    """Every entry under *root*: kind, bytes and inode, symlinks not followed."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            info = path.lstat()
            if path.is_symlink():
                found[str(path)] = ("link", os.readlink(path), info.st_ino)
            elif path.is_dir():
                found[str(path)] = ("dir", None, info.st_ino)
            else:
                found[str(path)] = ("file", path.read_bytes(), info.st_ino)
    return found


class TestTheKernelDecides:
    """Absent Landlock, an ABI below the floor, an unknown machine: exit 126,
    one note, and no ruleset call. Runs on every platform (no entries)."""

    @pytest.mark.parametrize("code", ["ENOSYS", "EOPNOTSUPP", "EPERM"])
    def test_a_failed_abi_query_is_unavailable(self, code):
        number = getattr(errno, code)
        run = Run(["--", "true"], kernel=FakeKernel(abi=OSError(number, os.strerror(number))))
        assert run.code == lx.EXIT_UNAVAILABLE == 126
        assert len(run.lines) == 1
        line = run.lines[0]
        assert line.startswith("landlock_exec: unavailable: Landlock ABI none ("), line
        assert code in line and "the floor is 3" in line, line
        assert run.kernel.jail_calls == [] and run.exec_calls == []

    @pytest.mark.parametrize("abi", [1, 2])
    def test_an_abi_below_the_floor_is_unavailable_with_no_reduced_mode(self, abi):
        run = Run(["--", "true"], kernel=FakeKernel(abi=abi))
        assert run.code == lx.EXIT_UNAVAILABLE
        assert run.lines == [
            f"landlock_exec: unavailable: Landlock ABI {abi} found; the floor is 3 "
            "(below it truncation cannot be denied)"
        ]
        # Never a ruleset with a mask that lacks TRUNCATE.
        assert run.kernel.jail_calls == [] and run.exec_calls == []

    def test_an_unknown_machine_is_unavailable_and_named(self):
        run = Run(["--", "true"], kernel=FakeKernel(machine="riscv64"))
        assert run.code == lx.EXIT_UNAVAILABLE
        assert len(run.lines) == 1
        assert run.lines[0].startswith(
            "landlock_exec: unavailable: machine 'riscv64' is not in the syscall table"
        ), run.lines
        # The ABI is never even queried with numbers the table does not vouch for.
        assert run.kernel.names == ["machine"]

    def test_the_syscall_table_is_asserted_per_machine(self):
        assert lx.SYSCALLS == {"x86_64": (444, 445, 446), "aarch64": (444, 445, 446)}

    def test_the_real_kernel_names_a_non_linux_platform(self, monkeypatch):
        monkeypatch.setattr(lx.sys, "platform", "darwin")
        monkeypatch.setattr(lx.platform, "machine", lambda: "x86_64")
        assert lx.Kernel().machine() == "x86_64 (darwin)"
        assert lx.Kernel().machine() not in lx.SYSCALLS

    @pytest.mark.parametrize("failing,code", [
        ("create_ruleset", errno.EINVAL),
        ("no_new_privs", errno.EPERM),
        ("restrict", errno.EPERM),
    ])
    def test_a_failed_landlock_call_never_runs_the_command(self, failing, code):
        kernel = FakeKernel(fail={failing: OSError(code, os.strerror(code))})
        run = Run(["--", "true"], kernel=kernel)
        assert run.code == lx.EXIT_UNAVAILABLE
        assert len(run.lines) == 1 and run.lines[0].startswith(
            "landlock_exec: unavailable: "), run.lines
        assert errno.errorcode[code] in run.lines[0]
        assert run.exec_calls == []
        if failing != "restrict":
            assert "restrict" not in kernel.names


class TestTheMask:
    def test_the_right_bits_are_appendix_a(self):
        assert (lx.EXECUTE, lx.WRITE_FILE, lx.READ_FILE, lx.READ_DIR) == (1, 2, 4, 8)
        assert [
            lx.REMOVE_DIR, lx.REMOVE_FILE, lx.MAKE_CHAR, lx.MAKE_DIR, lx.MAKE_REG,
            lx.MAKE_SOCK, lx.MAKE_FIFO, lx.MAKE_BLOCK, lx.MAKE_SYM, lx.REFER,
            lx.TRUNCATE, lx.IOCTL_DEV,
        ] == [1 << i for i in range(4, 16)]
        assert lx.FILE_RIGHTS == lx.WRITE_FILE | lx.TRUNCATE

    def test_abi_3_handles_truncate_and_refer_and_never_ioctl_dev(self):
        mask = lx.handled_mask(3)
        assert mask & lx.TRUNCATE and mask & lx.REFER
        assert not mask & lx.IOCTL_DEV
        # Reads and execute stay unhandled, so they stay unrestricted.
        assert not mask & (lx.EXECUTE | lx.READ_FILE | lx.READ_DIR)

    def test_every_accepted_abi_handles_the_same_set(self):
        expected = lx.handled_mask(3)
        for abi in range(3, 12):
            assert lx.handled_mask(abi) == expected, abi
            assert not lx.handled_mask(abi) & lx.IOCTL_DEV, abi

    def test_the_derivation_below_the_floor_is_what_the_floor_refuses(self):
        assert not lx.handled_mask(2) & lx.TRUNCATE and lx.handled_mask(2) & lx.REFER
        assert not lx.handled_mask(1) & (lx.TRUNCATE | lx.REFER)

    def test_main_creates_the_ruleset_with_the_abi_mask(self):
        run = Run(["--", "true"], kernel=FakeKernel(abi=3))
        assert run.code == 0
        assert ("create_ruleset", lx.handled_mask(3)) in run.kernel.calls

    def test_the_attr_structs_are_packed_as_the_kernel_reads_them(self):
        assert len(lx.ruleset_attr(lx.handled_mask(3))) == lx.RULESET_ATTR_SIZE == 8
        packed = lx.path_beneath_attr(lx.FILE_RIGHTS, 5)
        assert len(packed) == lx.PATH_BENEATH_ATTR_SIZE == 12
        assert packed == lx.FILE_RIGHTS.to_bytes(8, sys.byteorder) + (5).to_bytes(
            4, sys.byteorder)


class TestArguments:
    @pytest.mark.parametrize("argv", [
        pytest.param([], id="nothing"),
        pytest.param(["--allow", "dir:/x", "true"], id="no-double-dash"),
        pytest.param(["--"], id="empty-command"),
        pytest.param(["--allow", "dir:/x", "--"], id="empty-command-after-entries"),
        pytest.param(["--allow", "socket:/x", "--", "true"], id="unknown-kind"),
        pytest.param(["--allow", "/x", "--", "true"], id="no-kind"),
        pytest.param(["--allow", "dir:", "--", "true"], id="empty-path"),
        pytest.param(["--allow", "--", "true"], id="flag-without-value"),
        pytest.param(["--allow-optional", "--", "true"], id="optional-without-value"),
        pytest.param(["--bogus", "--", "true"], id="unknown-option"),
        pytest.param(["--allow=dir:/x", "--", "true"], id="equals-form"),
    ])
    def test_a_usage_error_exits_122_before_the_kernel_is_asked(self, argv):
        run = Run(argv)
        assert run.code == lx.EXIT_USAGE == 122
        assert len(run.lines) == 1 and run.lines[0].startswith("landlock_exec: usage: ")
        assert run.kernel.calls == [] and run.exec_calls == []

    def test_a_colon_inside_a_path_survives(self):
        entries, command = lx.parse_args([
            "--allow", "dir:/a:b:c", "--allow-optional", "file:/x:y", "--", "cmd",
        ])
        assert entries == [lx.Entry("dir", "/a:b:c", True), lx.Entry("file", "/x:y", False)]
        assert command == ["cmd"]

    def test_execvp_receives_exactly_the_command_after_the_first_double_dash(self):
        command = ["/opt/codex", "sandbox", "-c", "x=1", "--", "sh", "-c", "echo --"]
        run = Run(["--"] + command)
        assert run.code == 0
        assert run.exec_calls == [("/opt/codex", command)]

    def test_the_jail_is_applied_before_the_exec(self):
        run = Run(["--", "true"])
        assert run.kernel.names == [
            "machine", "abi", "create_ruleset", "no_new_privs", "restrict", "execvp",
        ]

    def test_an_exec_failure_exits_127_after_the_jail(self):
        error = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT))
        run = Run(["--", "no-such-command"], exec_error=error)
        assert run.code == lx.EXIT_EXEC_FAILED == 127
        assert run.lines == [
            f"landlock_exec: exec-failed: no-such-command: {os.strerror(errno.ENOENT)}"
        ]
        assert "restrict" in run.kernel.names


@linux_only
class TestEntries:
    """The entry contract, on real paths under ``tmp_path`` with the fake kernel."""

    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "slot").mkdir()
        (tmp_path / "plans").mkdir()
        (tmp_path / "plans" / "plan.md").write_bytes(b"seed\n")
        return tmp_path

    def _assert_no_jail(self, run):
        assert run.kernel.jail_calls == [], run.kernel.calls
        assert run.exec_calls == []

    def test_a_missing_required_entry_exits_125_before_any_ruleset(self, tree):
        missing = tree / "gone"
        run = Run(_allow(f"dir:{missing}") + ["--", "true"])
        assert run.code == lx.EXIT_REQUIRED_ENTRY == 125
        assert run.lines == [f"landlock_exec: required-entry: dir:{missing}: missing"]
        self._assert_no_jail(run)
        assert not missing.exists()

    def test_a_required_symlink_exits_125(self, tree):
        link = tree / "link"
        link.symlink_to(tree / "slot")
        run = Run(_allow(f"dir:{link}") + ["--", "true"])
        assert run.code == lx.EXIT_REQUIRED_ENTRY
        assert run.lines == [f"landlock_exec: required-entry: dir:{link}: symlink"]
        self._assert_no_jail(run)

    def test_a_dangling_required_symlink_is_a_symlink_not_missing(self, tree):
        link = tree / "dangling"
        link.symlink_to(tree / "nowhere")
        run = Run(_allow(f"file:{link}") + ["--", "true"])
        assert run.lines == [f"landlock_exec: required-entry: file:{link}: symlink"]
        self._assert_no_jail(run)
        assert not (tree / "nowhere").exists()

    def test_a_dir_entry_that_is_a_file_exits_125(self, tree):
        plan = tree / "plans" / "plan.md"
        run = Run(_allow(f"dir:{plan}") + ["--", "true"])
        assert run.code == lx.EXIT_REQUIRED_ENTRY
        assert run.lines == [f"landlock_exec: required-entry: dir:{plan}: wrong-kind"]
        self._assert_no_jail(run)

    def test_a_file_entry_that_is_a_dir_exits_125(self, tree):
        slot = tree / "slot"
        run = Run(_allow(f"file:{slot}") + ["--", "true"])
        assert run.lines == [f"landlock_exec: required-entry: file:{slot}: wrong-kind"]
        self._assert_no_jail(run)

    def test_every_failing_required_entry_is_named_and_nothing_else(self, tree):
        gone, plan = tree / "gone", tree / "plans" / "plan.md"
        run = Run(
            _allow(f"dir:{gone}", f"dir:{tree / 'slot'}", f"dir:{plan}",
                   optional=[f"dir:{tree / 'also-gone'}"])
            + ["--", "true"]
        )
        assert run.code == lx.EXIT_REQUIRED_ENTRY
        assert run.lines == [
            f"landlock_exec: required-entry: dir:{gone}: missing",
            f"landlock_exec: required-entry: dir:{plan}: wrong-kind",
        ]
        self._assert_no_jail(run)

    def test_a_missing_optional_ptmx_is_skipped_and_the_ruleset_still_applied(
            self, tree, monkeypatch):
        # The real /dev/ptmx cannot be made to vanish, so a stand-in is named a
        # tty entry; the wrong-kind rows below use the real paths.
        ptmx = tree / "dev" / "ptmx"
        monkeypatch.setattr(lx, "TTY_ENTRIES", lx.TTY_ENTRIES | {str(ptmx)})
        run = Run(_allow(f"dir:{tree / 'slot'}", optional=[f"file:{ptmx}"])
                  + ["--", "true"])
        assert run.code == 0
        assert run.lines == [
            f"landlock_exec: optional-entry-skipped: file:{ptmx}: missing; "
            "tty: true exec sessions will fail (failed to openpty)"
        ]
        assert run.kernel.names[-3:] == ["no_new_privs", "restrict", "execvp"]
        assert set(run.kernel.rules) == {str(tree / "slot")}
        assert not ptmx.exists() and not ptmx.parent.exists()

    @pytest.mark.parametrize("spec", ["dir:/dev/ptmx", "file:/dev/pts"])
    def test_the_real_tty_entries_carry_the_tty_note(self, spec):
        if not os.path.exists(spec.split(":", 1)[1]):
            pytest.skip(f"{spec} does not exist here")
        run = Run(_allow(optional=[spec]) + ["--", "true"])
        assert run.code == 0
        assert len(run.lines) == 1
        line = run.lines[0]
        assert line.startswith(f"landlock_exec: optional-entry-skipped: {spec}: "), line
        assert line.endswith("; tty: true exec sessions will fail (failed to openpty)")
        assert "restrict" in run.kernel.names

    def test_another_optional_entry_is_skipped_without_the_tty_note(self, tree):
        gone = tree / "gone"
        run = Run(_allow(optional=[f"dir:{gone}"]) + ["--", "true"])
        assert run.lines == [f"landlock_exec: optional-entry-skipped: dir:{gone}: missing"]
        assert run.code == 0 and "restrict" in run.kernel.names

    def test_dir_and_file_rules_get_their_rights_on_the_checked_inode(self, tree):
        slot, plan = tree / "slot", tree / "plans" / "plan.md"
        run = Run(_allow(f"dir:{slot}", f"file:{plan}", "file:/dev/null") + ["--", "true"])
        assert run.code == 0
        mask = lx.handled_mask(7)
        assert run.kernel.rules == {
            str(slot): mask,
            str(plan): lx.WRITE_FILE | lx.TRUNCATE,
            "/dev/null": lx.WRITE_FILE | lx.TRUNCATE,
        }
        for call in run.kernel.calls:
            if call[0] == "add_rule":
                assert call[1] == FakeKernel.RULESET_FD
                flags = call[4]
                assert flags & os.O_PATH and flags & os.O_NOFOLLOW and flags & os.O_CLOEXEC

    def test_a_colon_inside_a_path_reaches_the_rule(self, tree):
        odd = tree / "a:b"
        odd.mkdir()
        run = Run(_allow(f"dir:{odd}") + ["--", "true"])
        assert run.code == 0
        assert run.kernel.rules == {str(odd): lx.handled_mask(7)}

    def test_no_path_is_ever_created(self, tree):
        before = _snapshot(tree)
        Run(_allow(f"dir:{tree / 'new-slot'}", f"file:{tree / 'plans' / 'new.md'}")
            + ["--", "true"])
        Run(_allow(f"dir:{tree / 'slot'}",
                   optional=[f"dir:{tree / 'dev' / 'pts'}", f"file:{tree / 'x' / 'y'}"])
            + ["--", "true"])
        assert _snapshot(tree) == before

    def test_a_note_is_one_line_even_for_a_path_with_a_newline(self, tree):
        odd = tree / "a\nb"
        run = Run(_allow(f"dir:{odd}") + ["--", "true"])
        assert run.code == lx.EXIT_REQUIRED_ENTRY
        assert run.stderr == f"landlock_exec: required-entry: dir:{tree}/a\\nb: missing\n"

    def test_a_failed_add_rule_names_the_entry_and_never_runs_the_command(self, tree):
        slot = tree / "slot"
        kernel = FakeKernel(fail={"add_rule": OSError(errno.EINVAL, "Invalid argument")})
        run = Run(_allow(f"dir:{slot}") + ["--", "true"], kernel=kernel)
        assert run.code == lx.EXIT_UNAVAILABLE
        assert run.lines == [
            f"landlock_exec: unavailable: landlock_add_rule failed for dir:{slot} "
            "(EINVAL, Invalid argument)"
        ]
        assert "restrict" not in kernel.names and run.exec_calls == []

    def test_entry_descriptors_do_not_outlive_main(self, tree):
        before = sorted(os.listdir("/proc/self/fd"))
        Run(_allow(f"dir:{tree / 'slot'}", f"file:{tree / 'plans' / 'plan.md'}")
            + ["--", "true"])
        Run(_allow(f"dir:{tree / 'slot'}", f"dir:{tree / 'gone'}") + ["--", "true"])
        assert sorted(os.listdir("/proc/self/fd")) == before


class TestTheScriptStandsAlone:
    def test_it_imports_nothing_but_the_stdlib(self):
        tree = ast.parse(WRAPPER.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a relative import pulls in the utils package"
                imported.add((node.module or "").split(".")[0])
        assert imported <= {"ctypes", "errno", "os", "platform", "stat", "struct", "sys"}, (
            imported)

    def test_it_runs_as_a_script_under_isolated_mode(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(WRAPPER)],
            capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        )
        assert proc.returncode == lx.EXIT_USAGE, proc.stderr
        assert proc.stdout == ""
        assert proc.stderr.startswith("landlock_exec: usage: no '--' before the command")


# ===========================================================================
# Tier 2: the real kernel
# ===========================================================================

@functools.lru_cache(maxsize=None)
def _landlock_unusable_reason():
    """``None`` when this kernel can run the jail, else why not."""
    kernel = lx.Kernel()
    machine = kernel.machine()
    if machine not in lx.SYSCALLS:
        return f"machine {machine!r} is not in the wrapper's syscall table"
    try:
        abi = kernel.abi()
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno))
        return f"Landlock unavailable: {name} ({exc.strerror})"
    if abi < lx.ABI_FLOOR:
        return f"Landlock ABI {abi} found; the floor is {lx.ABI_FLOOR}"
    return None


def require_landlock():
    """Skip unless the ABI query returns >= 3; with ``MULTI_LLM_REQUIRE_LANDLOCK=1``
    the same condition fails instead, so a skip cannot hide a regression."""
    reason = _landlock_unusable_reason()
    if reason is None:
        return
    if os.environ.get(REQUIRE_ENV, "") not in ("", "0"):
        pytest.fail(f"{REQUIRE_ENV} is set but the tier-2 Landlock rows cannot run: {reason}")
    pytest.skip(reason)


class TestTheSkipHelper:
    """Tier 1: the tier-2 gate itself, against a faked kernel."""

    @pytest.fixture
    def faked(self, monkeypatch):
        def install(**kwargs):
            monkeypatch.setattr(lx, "Kernel", lambda: FakeKernel(**kwargs))
            _landlock_unusable_reason.cache_clear()
        yield install
        _landlock_unusable_reason.cache_clear()

    @pytest.mark.parametrize("kwargs,reason", [
        ({"abi": OSError(errno.ENOSYS, "Function not implemented")},
         "Landlock unavailable: ENOSYS (Function not implemented)"),
        ({"abi": 2}, "Landlock ABI 2 found; the floor is 3"),
        ({"machine": "riscv64"}, "machine 'riscv64' is not in the wrapper's syscall table"),
    ])
    def test_it_skips_naming_the_errno_or_abi(self, faked, monkeypatch, kwargs, reason):
        monkeypatch.delenv(REQUIRE_ENV, raising=False)
        faked(**kwargs)
        with pytest.raises(pytest.skip.Exception, match=f"^{re.escape(reason)}$"):
            require_landlock()

    def test_the_require_env_turns_the_skip_into_a_failure(self, faked, monkeypatch):
        monkeypatch.setenv(REQUIRE_ENV, "1")
        faked(abi=2)
        with pytest.raises(pytest.fail.Exception, match="Landlock ABI 2 found"):
            require_landlock()

    def test_a_usable_kernel_neither_skips_nor_fails(self, faked, monkeypatch):
        monkeypatch.setenv(REQUIRE_ENV, "1")
        faked(abi=3)
        require_landlock()


def _jail(allow, command, *, optional=(), env=None, cwd=None):
    """The wrapper exactly as the parent launches it."""
    argv = [sys.executable, "-I", "-S", str(WRAPPER), *_allow(*allow, optional=optional),
            "--", *command]
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=60, env=env,
        cwd=None if cwd is None else str(cwd),
    )


def _sh(script, *args):
    """``sh -c`` with the paths as positional parameters, never spliced in."""
    return ["sh", "-c", script, "sh", *[str(a) for a in args]]


def _py(script, *args):
    return [sys.executable, "-I", "-S", "-B", "-c", script, *[str(a) for a in args]]


_TRUNCATE = "import os, sys; os.truncate(sys.argv[1], 0)"

#: Write attempts aimed at a non-allowed directory ``$1`` holding ``$1/existing.txt``.
DENIED_WRITES = {
    "echo-new": _sh('echo x > "$1/new.txt"', "{dir}"),
    "echo-existing": _sh('echo x > "$1/existing.txt"', "{dir}"),
    "rm": _sh('rm -f "$1/existing.txt"', "{dir}"),
    "mv": _sh('mv "$1/existing.txt" "$1/moved.txt"', "{dir}"),
    "mkdir": _sh('mkdir "$1/newdir"', "{dir}"),
    "ln-s": _sh('ln -s "$1/existing.txt" "$1/link"', "{dir}"),
    "truncate": _py(_TRUNCATE, "{dir}/existing.txt"),
}


def _fill(command, **values):
    return [part.format(**values) for part in command]


@pytest.mark.landlock
class TestTheRealJail:
    @pytest.fixture(autouse=True)
    def _landlock(self):
        require_landlock()

    @pytest.fixture
    def repo(self, tmp_path):
        """A scratch repo, a slot TMPDIR and a sibling slot, all under tmp_path."""
        root = tmp_path / "repo"
        for sub in ("", "sub", ".git", "plans/p"):
            (root / sub).mkdir(parents=True, exist_ok=True)
            (root / sub / "existing.txt").write_bytes(b"seeded bytes\n")
        (tmp_path / "slot-a").mkdir()
        (tmp_path / "slot-b").mkdir()
        return tmp_path

    def _base(self, repo):
        return [f"dir:{repo / 'slot-a'}", "file:/dev/null"]

    @pytest.mark.parametrize("target", ["", "sub", ".git", "plans/p"],
                             ids=["repo-root", "subdir", "dot-git", "output-folder"])
    @pytest.mark.parametrize("action", sorted(DENIED_WRITES))
    def test_a_write_outside_the_allow_list_is_denied_and_changes_nothing(
            self, repo, action, target):
        directory = repo / "repo" / target
        before = _snapshot(repo / "repo")
        proc = _jail(self._base(repo), _fill(DENIED_WRITES[action], dir=directory))
        assert proc.returncode != 0, (action, proc.stdout, proc.stderr)
        assert "Permission denied" in proc.stderr or "PermissionError" in proc.stderr, (
            proc.stderr)
        assert _snapshot(repo / "repo") == before

    def test_a_write_inside_an_allowed_dir_succeeds(self, repo):
        slot = repo / "slot-a"
        proc = _jail(self._base(repo),
                     _sh('echo x > "$1/ok.txt" && mkdir "$1/d" && rm "$1/ok.txt"', slot))
        assert proc.returncode == 0, proc.stderr
        assert (slot / "d").is_dir() and not (slot / "ok.txt").exists()

    def test_a_write_into_a_sibling_slot_is_denied(self, repo):
        before = _snapshot(repo / "slot-b")
        proc = _jail(self._base(repo), _sh('echo x > "$1/stray.txt"', repo / "slot-b"))
        assert proc.returncode != 0 and "Permission denied" in proc.stderr
        assert _snapshot(repo / "slot-b") == before == {}

    def test_dev_null_stays_writable(self, repo):
        proc = _jail(self._base(repo), _sh("echo x > /dev/null && echo ok 2>/dev/null"))
        assert proc.returncode == 0 and proc.stdout == "ok\n", proc.stderr


@pytest.mark.landlock
class TestAFileOnlyGrant:
    """A ``file:`` rule allows the in-place write shapes and nothing else."""

    @pytest.fixture(autouse=True)
    def _landlock(self):
        require_landlock()

    @pytest.fixture
    def plan(self, tmp_path):
        (tmp_path / "slot").mkdir()
        (tmp_path / "plans").mkdir()
        path = tmp_path / "plans" / "plan.md"
        path.write_bytes(b"seed line\n")
        return path

    def _allow_for(self, plan):
        return [f"dir:{plan.parent.parent / 'slot'}", "file:/dev/null", f"file:{plan}"]

    @pytest.mark.parametrize("shape,command,expected", [
        ("open-w", _py("open(__import__('sys').argv[1], 'w').write('new\\n')", "{p}"),
         b"new\n"),
        ("redirect", _sh('echo y > "$1"', "{p}"), b"y\n"),
        ("append", _sh('echo z >> "$1"', "{p}"), b"seed line\nz\n"),
    ])
    def test_an_in_place_write_succeeds_and_keeps_the_inode(
            self, plan, shape, command, expected):
        identity = (plan.stat().st_dev, plan.stat().st_ino)
        proc = _jail(self._allow_for(plan), _fill(command, p=plan))
        assert proc.returncode == 0, (shape, proc.stderr)
        assert plan.read_bytes() == expected
        assert (plan.stat().st_dev, plan.stat().st_ino) == identity

    @pytest.mark.parametrize("shape,command", [
        ("sed-i", _sh('sed -i "s/seed/SED/" "$1"', "{p}")),
        ("tempfile-replace-beside", _py(
            "import os, sys, tempfile\n"
            "fd, tmp = tempfile.mkstemp(dir=os.path.dirname(sys.argv[1]))\n"
            "os.write(fd, b'replaced'); os.close(fd); os.replace(tmp, sys.argv[1])",
            "{p}")),
        ("tempfile-replace-from-tmpdir", _py(
            "import os, sys, tempfile\n"
            "fd, tmp = tempfile.mkstemp(dir=sys.argv[2])\n"
            "os.write(fd, b'replaced'); os.close(fd); os.replace(tmp, sys.argv[1])",
            "{p}", "{slot}")),
        ("delete", _sh('rm -f "$1"', "{p}")),
        ("sibling", _sh('echo x > "$(dirname "$1")/other.md"', "{p}")),
    ])
    def test_a_replacement_delete_or_sibling_is_denied(self, plan, shape, command):
        identity = (plan.stat().st_dev, plan.stat().st_ino)
        before = _snapshot(plan.parent)
        proc = _jail(self._allow_for(plan),
                     _fill(command, p=plan, slot=plan.parent.parent / "slot"))
        assert proc.returncode != 0, (shape, proc.stdout, proc.stderr)
        assert plan.read_bytes() == b"seed line\n"
        assert (plan.stat().st_dev, plan.stat().st_ino) == identity
        assert _snapshot(plan.parent) == before


@pytest.mark.landlock
class TestTheLaunch:
    @pytest.fixture(autouse=True)
    def _landlock(self):
        require_landlock()

    def test_a_required_entry_swapped_for_a_symlink_never_runs_the_command(self, tmp_path):
        slot, witness = tmp_path / "slot", tmp_path / "witness"
        slot.mkdir()
        witness.mkdir()
        # Resolved by the parent, then swapped before the launch.
        resolved = os.path.realpath(slot)
        slot.rename(tmp_path / "elsewhere")
        slot.symlink_to(tmp_path / "elsewhere")
        marker = witness / "ran"
        proc = _jail([f"dir:{resolved}", f"dir:{witness}"],
                     _sh('echo ran > "$1"', marker))
        assert proc.returncode == lx.EXIT_REQUIRED_ENTRY, proc.stderr
        assert proc.stderr == f"landlock_exec: required-entry: dir:{resolved}: symlink\n"
        assert not marker.exists()

    def test_isolated_mode_ignores_a_shadowing_pythonpath(self, tmp_path):
        # The `.pth`-in-site-packages half of this row (plan §8.2) needs a write
        # into the interpreter's own site-packages, i.e. the skill venv; it is
        # not done here. `-S` is what stops it, and the argv above pins `-S`.
        evil, markers, slot, repo = (tmp_path / n for n in ("evil", "markers", "slot", "repo"))
        for directory in (evil, markers, slot, repo):
            directory.mkdir()
        for module in ("struct", "platform"):
            (evil / f"{module}.py").write_text(
                f"open({str(markers / module)!r}, 'w').write('ran')\n", encoding="utf-8")
        (evil / "ctypes").mkdir()
        (evil / "ctypes" / "__init__.py").write_text(
            f"open({str(markers / 'ctypes')!r}, 'w').write('ran')\n", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(evil))

        # Control: without -I the shadow does run, so the sabotage is live.
        control = subprocess.run([sys.executable, "-c", "import struct"], env=env,
                                 capture_output=True, text=True, timeout=60)
        assert (markers / "struct").exists(), control.stderr
        for leftover in markers.iterdir():
            leftover.unlink()

        proc = _jail([f"dir:{slot}", "file:/dev/null"],
                     _sh('echo x > "$1/new.txt"', repo), env=env)
        assert sorted(os.listdir(markers)) == [], proc.stderr
        # The ruleset was still applied: the write outside the allow-list failed.
        assert proc.returncode != 0 and "Permission denied" in proc.stderr
        assert not (repo / "new.txt").exists()
