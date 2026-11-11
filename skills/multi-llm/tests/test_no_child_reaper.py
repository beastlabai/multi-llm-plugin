"""No plugin source reaps a child it does not name, or has children reaped at exit.

``kill_tree(child=proc)`` signals a timed-out child's group by number because,
while the child is unreaped, its pid and the number of the group it leads cannot
be recycled (docs/development/runtime.md, "An owned child's kill"). Two things in
the process would end that early, behind the kill's back: a wait that reaps any
child rather than one it names, and a SIGCHLD disposition that has the kernel
reap every child at its exit (``SIG_IGN``, ``SA_NOCLDWAIT``). This test fails a
runtime source (``skills/multi-llm/`` and ``packaging/``, tests excluded) that
spells either. Any SIGCHLD disposition set counts, bar ``signal.signal`` back to
``SIG_DFL``, which is what a process entry does with one inherited ignored.

SIGCHLD is recognised by name (``signal.SIGCHLD``, ``signal.Signals.SIGCHLD``,
an imported or renamed ``SIGCHLD``, a name assigned one of those), passed
positionally or by keyword, and by number (:data:`_SIGCHLD_NUMBERS`) when it
goes to a call that sets a disposition. It reads the syntax, so it is best
effort: a signal held in a variable assigned a plain number, computed, unpacked
from a tuple or passed through a function is outside it, and the doc's contract
still holds there.
"""

import ast
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent

#: ``os`` waits that reap whichever child exits first: always forbidden.
#: (``wait4`` takes a pid, but is held to the same rule.)
_ANY_CHILD_WAITS = frozenset({"wait", "wait3", "wait4"})

#: ``waitid`` id types that name one process.
_ONE_PROCESS = frozenset({"P_PID", "P_PIDFD"})

#: Calls that take SIGCHLD first without changing its disposition.
_SIGCHLD_READERS = frozenset({"getsignal", "Signals", "int", "strsignal"})

#: Calls that set a signal's disposition: SIGCHLD's number counts there too.
_DISPOSITION_SETTERS = frozenset({"signal", "sigaction", "add_signal_handler"})

#: SIGCHLD's number on the kernels the plugin may run on: 17 on Linux (x86,
#: Arm, RISC-V, PowerPC), 18 on Linux MIPS and PA-RISC, 20 on macOS, the BSDs,
#: Linux Alpha and SPARC. A number another kernel gives another signal (18 is
#: SIGCONT on x86 Linux) is flagged too: the guard errs towards a finding.
_SIGCHLD_NUMBERS = frozenset({17, 18, 20})

#: Keyword names a signal goes by in the calls that take one.
_SIGNAL_KEYWORDS = frozenset({"signalnum", "signum", "sig", "signal"})

#: (source path, finding) -> why it cannot recycle an owned child's numbers.
#: Empty: the one wait in the sources, the detach double-fork's
#: ``os.waitpid(intermediate, 0)``, names its own child, which the rule allows.
_ALLOWED: Dict[Tuple[str, str], str] = {}


def _sources() -> Iterator[Path]:
    """Every runtime Python source: the skill, and the packaging scripts. A
    ``multi-llm`` directory under ``packaging/`` is a built copy of the skill,
    scanned already at its source."""
    skipped = {"tests", "__pycache__", ".venv", "node_modules"}
    for root, copies in ((SKILL_ROOT, False), (REPO_ROOT / "packaging", True)):
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(root).parts[:-1]
            if any(p in skipped or p.startswith(".") for p in parts):
                continue
            if copies and "multi-llm" in parts:
                continue
            yield path


def _names_sigchld(node: ast.AST, sigchld: Set[str], *, numeric: bool = False) -> bool:
    """*node* is SIGCHLD: ``signal.SIGCHLD``, an imported ``SIGCHLD``, a name
    in *sigchld* (an alias of either), any of them wrapped in a call
    (``int(signal.SIGCHLD)``), and with *numeric* one of
    :data:`_SIGCHLD_NUMBERS` (``signal.Signals(17)`` too)."""
    if isinstance(node, ast.Call) and node.args:
        return _names_sigchld(node.args[0], sigchld, numeric=numeric)
    if isinstance(node, ast.Attribute):
        return node.attr == "SIGCHLD"
    if numeric and isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value in _SIGCHLD_NUMBERS
    return isinstance(node, ast.Name) and (node.id == "SIGCHLD" or node.id in sigchld)


def _signal_argument(node: ast.Call) -> Optional[ast.AST]:
    """The signal a call is handed: its first positional argument, else a
    keyword a signal goes by (:data:`_SIGNAL_KEYWORDS`)."""
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg in _SIGNAL_KEYWORDS:
            return keyword.value
    return None


def _second_argument(node: ast.Call) -> Optional[ast.AST]:
    """What a disposition call is handed after the signal: the handler, or
    ``sigaction``'s new action (``None`` there only reads)."""
    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg in ("handler", "act", "action"):
            return keyword.value
    return None


def _callee(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def findings(source: str) -> List[Tuple[int, str]]:
    """Each ``(line, finding)`` in *source* that could reap an owned child
    behind its ``Popen`` or have the kernel reap it at exit."""
    tree = ast.parse(source)
    os_modules: Set[str] = set()
    #: local name -> the ``os`` function it is (``from os import``, an alias).
    os_functions: Dict[str, str] = {}
    signal_modules: Set[str] = set()
    #: local name -> the ``signal`` function it is (``from signal import``, an alias).
    signal_functions: Dict[str, str] = {}
    sigchld: Set[str] = set()
    found: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "posix"):
                    os_modules.add(alias.asname or alias.name)
                elif alias.name == "signal":
                    signal_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in ("os", "posix"):
            for alias in node.names:
                os_functions[alias.asname or alias.name] = alias.name
                if alias.name in _ANY_CHILD_WAITS:
                    found.append((node.lineno, f"os.{alias.name} imported"))
        elif isinstance(node, ast.ImportFrom) and node.module == "signal":
            for alias in node.names:
                if alias.name == "SIGCHLD":
                    sigchld.add(alias.asname or "SIGCHLD")
                else:
                    signal_functions[alias.asname or alias.name] = alias.name

    # Names assigned SIGCHLD (``S = signal.SIGCHLD``, ``T = S``) or a ``signal``
    # function (``install = signal.signal``), followed until nothing changes.
    assignments = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            assignments.append((node.targets[0].id, node.value))
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)):
            assignments.append((node.target.id, node.value))
    changed = True
    while changed:
        changed = False
        for name, value in assignments:
            if name not in sigchld and _names_sigchld(value, sigchld):
                sigchld.add(name)
                changed = True
            function = None
            if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                    and value.value.id in signal_modules):
                function = value.attr
            elif isinstance(value, ast.Name) and value.id in signal_functions:
                function = signal_functions[value.id]
            if function is not None and signal_functions.get(name) != function:
                signal_functions[name] = function
                changed = True

    def disposition_setter(func: ast.AST) -> bool:
        """*func* sets a disposition: ``signal.signal``, an imported or renamed
        ``signal``, any ``sigaction`` or ``add_signal_handler``."""
        if isinstance(func, ast.Name):
            name = signal_functions.get(func.id, func.id)
            return name in _DISPOSITION_SETTERS and (
                func.id in signal_functions or name != "signal")
        if isinstance(func, ast.Attribute):
            if func.attr == "signal":
                return isinstance(func.value, ast.Name) and func.value.id in signal_modules
            return func.attr in _DISPOSITION_SETTERS
        return False

    def restores_default(node: ast.Call) -> bool:
        """``signal.signal(SIGCHLD, SIG_DFL)``: the default disposition, under
        which every child stays a zombie until it is waited for, the contract's
        own (``process_control.restore_sigchld_default``). Only through
        ``signal.signal``, whose ``SIG_DFL`` is the kernel's default."""
        func, handler = node.func, _second_argument(node)
        if isinstance(func, ast.Attribute):
            through_signal = (func.attr == "signal" and isinstance(func.value, ast.Name)
                              and func.value.id in signal_modules)
        else:
            through_signal = (isinstance(func, ast.Name)
                              and signal_functions.get(func.id) == "signal")
        if isinstance(handler, ast.Attribute):
            default = (handler.attr == "SIG_DFL" and isinstance(handler.value, ast.Name)
                       and handler.value.id in signal_modules)
        else:
            default = (isinstance(handler, ast.Name)
                       and signal_functions.get(handler.id) == "SIG_DFL")
        return through_signal and default

    def os_attribute(node: ast.AST) -> Optional[str]:
        """The ``os`` function *node* names: ``os.x``, ``getattr(os, "x")``."""
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in os_modules):
            return node.attr
        if (isinstance(node, ast.Call) and _callee(node.func) == "getattr"
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)
                and node.args[0].id in os_modules
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            return node.args[1].value
        return None

    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name = os_attribute(node.value)
            if name is not None:
                os_functions[node.targets[0].id] = name

    for node in ast.walk(tree):
        name = os_attribute(node)
        if name in _ANY_CHILD_WAITS:
            found.append((node.lineno, f"os.{name}: reaps any child"))
        if not isinstance(node, ast.Call):
            continue
        function = os_attribute(node.func)
        if function is None and isinstance(node.func, ast.Name):
            function = os_functions.get(node.func.id)
        if function in _ANY_CHILD_WAITS and isinstance(node.func, ast.Name):
            found.append((node.lineno, f"os.{function}: reaps any child"))
        first = node.args[0] if node.args else None
        if function == "waitpid" and first is not None:
            if (isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub)) or (
                isinstance(first, ast.Constant) and isinstance(first.value, int)
                and first.value <= 0
            ):
                found.append((node.lineno, "os.waitpid(pid <= 0): reaps any child of a group"))
        if function == "waitid" and first is not None and _callee(first) not in _ONE_PROCESS:
            found.append((node.lineno, "os.waitid not by P_PID: reaps any child of a set"))
        given = [_signal_argument(node)] + [
            keyword.value for keyword in node.keywords if keyword.arg not in _SIGNAL_KEYWORDS]
        setter = disposition_setter(node.func)
        if any(arg is not None and _names_sigchld(arg, sigchld, numeric=setter)
               for arg in given):
            second = _second_argument(node)
            reads = _callee(node.func) in _SIGCHLD_READERS or (
                isinstance(second, ast.Constant) and second.value is None)
            if not reads and not restores_default(node):
                found.append((node.lineno, "SIGCHLD disposition set"))
    return sorted(set(found))


def _scan() -> List[Tuple[str, int, str]]:
    sites = []
    for path in _sources():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line, finding in findings(path.read_text(encoding="utf-8")):
            sites.append((rel, line, finding))
    return sites


def test_no_source_reaps_an_unnamed_child_or_sets_sigchld():
    offenders = [f"{rel}:{line}: {finding}" for rel, line, finding in _scan()
                 if (rel, finding) not in _ALLOWED]
    assert not offenders, (
        "kill_tree(child=...) relies on owned children staying unreaped until their "
        "Popen waits (docs/development/runtime.md, \"An owned child's kill\"): "
        + "; ".join(offenders)
    )


def test_every_allowance_still_has_its_site():
    sites = {(rel, finding) for rel, _line, finding in _scan()}
    assert set(_ALLOWED) <= sites, set(_ALLOWED) - sites


def test_the_scan_covers_the_skill_and_the_packaging_scripts():
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _sources()}
    assert "skills/multi-llm/utils/process_control.py" in scanned
    assert "skills/multi-llm/utils/subagent_executor.py" in scanned
    assert any(rel.startswith("packaging/common/") for rel in scanned)
    assert not any("/tests/" in rel for rel in scanned)


@pytest.mark.parametrize("source", [
    "import os\nos.wait()\n",
    "import os as o\no.wait()\n",
    "from os import wait\nwait()\n",
    "import os\nreap = getattr(os, 'wait')\n",
    "import os\nos.wait3(0)\n",
    "import os\nos.wait4(pid, 0)\n",
    "import os\nos.waitpid(-1, 0)\n",
    "import os\nos.waitpid(0, os.WNOHANG)\n",
    "import os\nos.waitpid(-pgid, 0)\n",
    "from os import waitpid as wp\nwp(-1, 0)\n",
    "import os\nos.waitid(os.P_ALL, 0, os.WEXITED)\n",
    "import os\nos.waitid(os.P_PGID, group, os.WEXITED)\n",
    "import os\nw = getattr(os, 'waitid', None)\nw(os.P_ALL, 0, os.WEXITED)\n",
    "import signal\nsignal.signal(signal.SIGCHLD, signal.SIG_IGN)\n",
    "import signal\nsignal.signal(signal.SIGCHLD, on_child)\n",
    "from signal import SIGCHLD, SIG_IGN, signal as install\ninstall(SIGCHLD, SIG_IGN)\n",
    "import signal\nloop.add_signal_handler(signal.SIGCHLD, on_child)\n",
    "import signal\nlibc.sigaction(int(signal.SIGCHLD), byref(new), None)\n",
    # By keyword.
    "import signal\nsignal.signal(signalnum=signal.SIGCHLD, handler=signal.SIG_IGN)\n",
    "import signal\nsignal.signal(handler=signal.SIG_IGN, signalnum=signal.SIGCHLD)\n",
    "import signal\nloop.add_signal_handler(sig=signal.SIGCHLD, callback=on_child)\n",
    # By number, to a call that sets a disposition: Linux, MIPS, macOS and the BSDs.
    "import signal\nsignal.signal(17, signal.SIG_IGN)\n",
    "import signal\nsignal.signal(18, on_child)\n",
    "import signal\nsignal.signal(20, signal.SIG_IGN)\n",
    "import signal\nsignal.signal(signalnum=17, handler=signal.SIG_IGN)\n",
    "import signal\nsignal.signal(signal.Signals(17), on_child)\n",
    "import signal as sg\nsg.signal(17, sg.SIG_IGN)\n",
    "from signal import signal as install\ninstall(17, on_child)\n",
    "import signal\ninstall = signal.signal\ninstall(17, on_child)\n",
    "libc.sigaction(17, byref(new), None)\n",
    "loop.add_signal_handler(20, on_child)\n",
    # Through a constant alias.
    "import signal\nsignal.signal(signal.Signals.SIGCHLD, on_child)\n",
    "import signal\nfrom signal import SIGCHLD as CHILD\nsignal.signal(CHILD, on_child)\n",
    "import signal\nS = signal.SIGCHLD\nsignal.signal(S, signal.SIG_IGN)\n",
    "import signal\nS = signal.SIGCHLD\nT = S\nsignal.signal(T, signal.SIG_IGN)\n",
    "import signal\nS: int = signal.SIGCHLD\nsignal.signal(S, on_child)\n",
    "import signal\nS = int(signal.SIGCHLD)\nsignal.signal(S, on_child)\n",
    "from signal import SIGCHLD\nS = SIGCHLD\nloop.add_signal_handler(S, on_child)\n",
    # SIG_DFL only through signal.signal: asyncio installs a handler of its own.
    "import signal\nloop.add_signal_handler(signal.SIGCHLD, signal.SIG_DFL)\n",
    "import signal\nsignal.signal(signal.SIGCHLD, other.SIG_DFL)\n",
])
def test_the_scan_finds(source):
    assert findings(source), source


@pytest.mark.parametrize("source", [
    "import signal\nn = 17\nsignal.signal(n, signal.SIG_IGN)\n",
    "import signal\nsignal.signal(signal.SIGCHLD + 0, signal.SIG_IGN)\n",
    "import signal\nS, T = signal.SIGCHLD, signal.SIGTERM\nsignal.signal(S, signal.SIG_IGN)\n",
])
def test_the_scan_is_best_effort(source):
    """Outside the scan, as its docstring says: a number in a variable, a
    computed signal, a tuple unpacked. The doc's contract still holds there."""
    assert findings(source) == [], source


@pytest.mark.parametrize("source", [
    "import os\nos.waitpid(intermediate, 0)\n",
    "import os\nos.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)\n",
    "import os\n_waitid = getattr(os, 'waitid', None)\n_waitid(os.P_PID, pid, os.WEXITED)\n",
    "import signal\nsignal.getsignal(signal.SIGCHLD)\n",
    "import signal\ncall(int(signal.SIGCHLD), None, old)\n",
    "import signal\nsignal.signal(signal.SIGTERM, on_term)\n",
    # Back to the default: children stay zombies until waited for.
    "import signal\nsignal.signal(signal.SIGCHLD, signal.SIG_DFL)\n",
    "from signal import SIGCHLD, SIG_DFL, signal as install\ninstall(SIGCHLD, SIG_DFL)\n",
    "import signal\nsignal.signal(signalnum=signal.SIGCHLD, handler=signal.SIG_DFL)\n",
    "proc.wait()\nproc.wait(timeout=5)\n",
    "import os\nos.waitstatus_to_exitcode(status)\n",
    "import signal\nsignal.signal(15, on_term)\n",
    "import signal\nsignal.signal(signalnum=signal.SIGTERM, handler=on_term)\n",
    "import signal\nsignal.getsignal(17)\n",
    "import signal\nS = signal.SIGCHLD\nsignal.getsignal(S)\n",
    "import signal\nsignal.getsignal(signalnum=signal.SIGCHLD)\n",
    "libc.sigaction(17, None, byref(old))\n",
    "width = pad(17, 20)\n",
    "signal(17, on_child)\n",  # no ``signal`` imported: nothing known to call
])
def test_the_scan_allows(source):
    assert findings(source) == [], source
