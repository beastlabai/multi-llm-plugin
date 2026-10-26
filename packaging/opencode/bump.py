#!/usr/bin/env python3
"""Move every opencode pin to one release and re-capture its fixtures.

opencode supports its newest release only: that release is the adapter's
``min_version`` and its ``verified_version``, and every other pin of it follows.
This script is the one way to move them, so a bump cannot half-land:

    make opencode-bump VERSION=<x.y.z>   # bump.py set <x.y.z>
    make opencode-latest-check           # bump.py check

``set`` refuses a release older than the current pin unless
``--allow-downgrade`` (``make opencode-bump VERSION=<x.y.z> ALLOW_DOWNGRADE=1``):
the newest release is the floor, so moving below it is a deliberate act. It
does, in order, and writes nothing until the network steps succeeded:

1. finds an ``opencode`` binary of that version: the installed one when it
   already is that version, else ``npm install opencode-ai@<v>`` under a
   scratch prefix (never a global install);
2. resolves the container's ``package-lock.json`` for it with
   ``npm install --package-lock-only`` on a scratch copy, refusing a resolution
   that moved any package but opencode's own;
3. re-captures the fixtures offline from that binary ($0, no model call, no
   provider: a localhost stub): ``tests/fixtures/opencode/capture/run.sh``,
   ``auth_list_capture.py`` and ``debug_config_capture.py``, and deletes the
   previous release's captures (git history keeps them);
4. rewrites every pin in :data:`PINS` plus the lockfile, and re-renders the
   documents generated from ``hosts/capabilities.yaml``.

Then run the suite: the parser, classifier and permission tests grade the new
captures, so a red test is the release's change to review, not noise. The
container image and the recorded CLI surface are rebuilt and re-seeded by hand
(printed at the end), as ``make e2e-lock-update`` leaves them.

``check`` asks npm for ``opencode-ai``'s latest version: exit 0 when it is the
pin, 1 when npm has a newer one (printing the bump command), 2 when npm could not
be asked.

``install --prefix <dir>`` is how the release gate gets **the** opencode it
probes (row 7), on the laptop and in CI alike, and it is under this file's
discipline on purpose: it reads the version through :func:`read_pins` and names
none of its own, so a bump moves what the gate tests with everything else. It
refuses unless every pin agrees, writes a scratch manifest and lockfile holding
only ``opencode-ai`` and its platform packages, copied from the container
lockfile's entries with their ``resolved`` and ``integrity`` values, and runs
``npm ci --ignore-scripts`` there against the single registry (``Makefile``
``OPENCODE_REGISTRY``), so npm refuses any tarball whose hash differs from the
committed one. Never a global install, and no install script: ``opencode-ai``'s
``postinstall`` links a platform binary into the wrapper and runs it
(``--version``), and when that fails it fetches a platform package of its own
with no committed integrity and runs THAT, all before anything here could look
at it. So nothing runs until what ``npm ci`` left is checked
(:func:`verify_installed`): every package npm's own record of the install
(``node_modules/.package-lock.json``) lists came from the single registry with
the committed integrity, and nothing else sits in ``node_modules``. The binary
is then a verified platform package's own ``bin/opencode``, chosen in the
order the ``postinstall`` would have tried them (:func:`platform_candidates`);
the wrapper's ``bin/opencode.exe``, and the ``.bin/opencode`` link to it, stay
``opencode-ai``'s placeholder script ("postinstall was not run") and are never
used. That binary's ``--version``, read in an isolated home with autoupdate
disabled exactly as :func:`find_binary` reads it, must equal the pin, or nothing
is printed but the refusal. Its absolute path is the last stdout line, and the
only one: npm's own output goes to stderr.

``verify-install --prefix <dir>`` is that check alone, against the subset
lockfile the prefix holds, and prints the chosen binary's path last: the
registry-install probe's container installs the pin with ``npm ci
--ignore-scripts`` itself and runs it there.

``tests/test_opencode_version_pins.py`` reads :func:`read_pins` and fails the
build when any two pins disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine as _machine
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
SKILL = Path("skills") / "multi-llm"
FIXTURES = SKILL / "tests" / "fixtures" / "opencode"
CONTAINER = SKILL / "tests" / "container"
NPM_PACKAGE = "opencode-ai"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")

#: Every text pin of the supported opencode release: (name, file relative to
#: the repo root, a pattern whose one group is the version). Each pattern must
#: match exactly once. The lockfile is read as JSON and written by npm
#: (:func:`resolve_lock`), never by a pattern. NOT a pin: the opencode row's
#: ``version`` in ``tests/container/cli-surface.yaml``. It names the release its
#: ``help_sha256`` was captured from, and the drift report reads it against the
#: installed version to tell "the help moved with this bump" from "it moved
#: under a fixed version", so it moves with its digest at the re-seed (step 3
#: below), never alone here (``tests/container/lock_update.py`` says the same).
PINS: Tuple[Tuple[str, Path, str], ...] = (
    ("adapter min_version", SKILL / "utils" / "providers" / "opencode.py",
     r'^    min_version = "(\d+\.\d+\.\d+)"$'),
    ("adapter verified_version", SKILL / "utils" / "providers" / "opencode.py",
     r'^    verified_version = "(\d+\.\d+\.\d+)"$'),
    ("mock --version", SKILL / "tests" / "mocks" / "mock_llm.py",
     r'^    "opencode": "(\d+\.\d+\.\d+)",$'),
    ("engines.opencode", Path("packaging") / "opencode" / "package.json",
     r'^    "opencode": ">=(\d+\.\d+\.\d+) <2\.0\.0"$'),
    # The plugin SDK's types, the reference for the plugin API the source's
    # comments rely on (`ToolContext`'s fields, the toast variant enum). Nothing
    # compiles against it and consumers never install devDependencies, but
    # package.json ships in the tarball, so a stale value publishes a plugin
    # API version older than the `engines` floor. OpenCode publishes
    # `@opencode-ai/plugin` at every host version, so the host pin owns it,
    # exact. Moving it is half the review: diff the SDK's published types
    # between the old and new versions and recheck the source's uses of them.
    ("plugin SDK devDependency", Path("packaging") / "opencode" / "package.json",
     r'^    "@opencode-ai/plugin": "(\d+\.\d+\.\d+)"$'),
    ("package README floor", Path("packaging") / "opencode" / "README.md",
     r"Requires \*\*OpenCode (\d+\.\d+\.\d+) or newer\*\*"),
    ("package README best-effort line", Path("packaging") / "opencode" / "README.md",
     r"\*\*Stream parsing is best-effort above opencode (\d+\.\d+\.\d+)\.\*\*"),
    ("host capabilities package floor", SKILL / "hosts" / "capabilities.yaml",
     r'^      name: opencode-multi-llm\n(?:      [^\n]*\n)*?      min_host_version: "(\d+\.\d+\.\d+)"'),
    ("README install line", Path("README.md"),
     r"Needs opencode (\d+\.\d+\.\d+) or newer"),
    ("docs/hosts.md install line (generated)", Path("docs") / "hosts.md",
     r"Needs opencode (\d+\.\d+\.\d+) or newer"),
    ("container package.json", CONTAINER / "package.json",
     r'^    "opencode-ai": "(\d+\.\d+\.\d+)"$'),
    ("cli-versions.lock row", CONTAINER / "cli-versions.lock",
     r'^  - id: opencode\n(?:    [^\n]*\n)*?    version: "(\d+\.\d+\.\d+)"$'),
)

LOCKFILE = CONTAINER / "package-lock.json"

#: ``check``'s question to npm. Offline, a bare ``npm view`` retries for ~70 s
#: (measured 2026-10-03 against a refused registry: two retries with backoff)
#: and a blackholed registry never answers; ``--fetch-retries=0`` ends the first
#: at once, ``--fetch-timeout`` bounds a slow response, and the overall budget
#: is the backstop for a connect that hangs (npm's timeout does not cover it).
#: :func:`cmd_check` runs it from an empty scratch directory named as
#: ``--prefix`` as well, so no project ``.npmrc`` steers it (the release path's
#: rule, ``registry_absent.mjs``; this one only advises).
NPM_VIEW_ARGV = ("npm", "view", NPM_PACKAGE, "version",
                 "--fetch-retries=0", "--fetch-timeout=10000")
NPM_VIEW_BUDGET_S = 20


class BumpError(RuntimeError):
    """A step refused; nothing after it ran."""


def _match(name: str, text: str, pattern: str) -> "re.Match[str]":
    found = list(re.finditer(pattern, text, re.M))
    if len(found) != 1:
        raise BumpError(f"{name}: expected exactly one match of {pattern!r}, found {len(found)}")
    return found[0]


def read_pins(root: Path = REPO) -> Dict[str, str]:
    """Every pin's current value, the lockfile's two included."""
    pins: Dict[str, str] = {}
    for name, rel, pattern in PINS:
        pins[name] = _match(name, (root / rel).read_text(encoding="utf-8"), pattern).group(1)
    lock = json.loads((root / LOCKFILE).read_text(encoding="utf-8"))
    pins["package-lock.json root dependency"] = lock["packages"][""]["dependencies"][NPM_PACKAGE]
    pins["package-lock.json resolved version"] = lock["packages"][f"node_modules/{NPM_PACKAGE}"]["version"]
    return pins


def write_pins(root: Path, version: str) -> List[str]:
    """Rewrite every :data:`PINS` site to ``version``; the files touched."""
    texts: Dict[Path, str] = {}
    for name, rel, pattern in PINS:
        text = texts.get(rel, (root / rel).read_text(encoding="utf-8"))
        match = _match(name, text, pattern)
        texts[rel] = text[: match.start(1)] + version + text[match.end(1):]
    for rel, text in texts.items():
        (root / rel).write_text(text, encoding="utf-8")
    return sorted(str(rel) for rel in texts)


def _own_package(key: str) -> bool:
    """``opencode-ai`` and the per-platform binaries it pulls in."""
    name = key.rsplit("node_modules/", 1)[-1]
    return name == NPM_PACKAGE or name.startswith("opencode-")


def resolve_lock(root: Path, version: str) -> Tuple[str, str]:
    """The container's ``(package.json, package-lock.json)`` texts for ``version``.

    npm resolves on a scratch copy (``--package-lock-only``: no install, no
    scripts), so the registry's integrity hashes are what land in the lock. A
    resolution that moved any other package is refused: that is
    ``make e2e-lock-update``'s job, on evidence from an image.
    """
    before = json.loads((root / LOCKFILE).read_text(encoding="utf-8"))["packages"]
    with tempfile.TemporaryDirectory(prefix="opencode-bump-lock-") as scratch:
        for name in ("package.json", "package-lock.json"):
            shutil.copy2(root / CONTAINER / name, Path(scratch) / name)
        subprocess.run(
            ["npm", "install", "--package-lock-only", "--ignore-scripts", "--no-audit",
             "--no-fund", "--save-exact", f"{NPM_PACKAGE}@{version}"],
            cwd=scratch, check=True, stdin=subprocess.DEVNULL,
        )
        manifest = (Path(scratch) / "package.json").read_text(encoding="utf-8")
        lock_text = (Path(scratch) / "package-lock.json").read_text(encoding="utf-8")
    after = json.loads(lock_text)["packages"]
    moved = sorted(
        key for key in set(before) | set(after)
        if key and not _own_package(key)
        and (before.get(key) or {}).get("version") != (after.get(key) or {}).get("version")
    )
    if moved:
        raise BumpError(
            "npm moved packages other than opencode's own: "
            + ", ".join(moved) + ". Move those with `make e2e-lock-update`, not here."
        )
    if after[f"node_modules/{NPM_PACKAGE}"]["version"] != version:
        raise BumpError(f"npm resolved {after[f'node_modules/{NPM_PACKAGE}']['version']}, not {version}")
    return manifest, lock_text


def _read_version(binary: str) -> str:
    with tempfile.TemporaryDirectory(prefix="opencode-bump-home-") as home:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "HOME": home, "OPENCODE_DISABLE_AUTOUPDATE": "1"},
            timeout=60,
        )
    return (done.stdout or "").strip()


def find_binary(version: str, scratch: Path, given: Optional[str]) -> str:
    """An ``opencode`` that answers ``--version`` with ``version``."""
    candidates = [given] if given else [
        shutil.which("opencode"), os.path.expanduser("~/.opencode/bin/opencode"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and _read_version(candidate) == version:
            return candidate
    if given:
        raise BumpError(f"{given} is not opencode {version}")
    prefix = scratch / "npm-prefix"
    print(f"opencode-bump: installing {NPM_PACKAGE}@{version} under {prefix} (not global)")
    subprocess.run(
        ["npm", "install", "--prefix", str(prefix), "--no-save", "--no-audit", "--no-fund",
         f"{NPM_PACKAGE}@{version}"],
        check=True, stdin=subprocess.DEVNULL,
    )
    binary = str(prefix / "node_modules" / ".bin" / "opencode")
    if _read_version(binary) != version:
        raise BumpError(f"the scratch install answers {_read_version(binary)!r}, not {version}")
    return binary


def recapture(root: Path, binary: str, version: str, scratch: Path) -> None:
    """Re-capture the pin's fixtures from ``binary``; drop every other release's."""
    fixtures = root / FIXTURES
    env = {**os.environ, "TMPDIR": str(scratch)}
    for argv in (
        [str(fixtures / "capture" / "run.sh"), "--bin", binary],
        [sys.executable, str(fixtures / "auth_list_capture.py"), "--bin", binary],
        [sys.executable, str(fixtures / "debug_config_capture.py"), "--bin", binary],
    ):
        print("opencode-bump: " + " ".join(Path(a).name if a.startswith(str(root)) else a for a in argv))
        subprocess.run(argv, check=True, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    for path in (fixtures / "capture").iterdir():
        if path.is_dir() and VERSION_RE.fullmatch(path.name) and path.name != version:
            shutil.rmtree(path)
    for path in fixtures.glob("*.json"):
        stem = re.fullmatch(r"(auth_list|debug_config)\.(\d+\.\d+\.\d+)\.json", path.name)
        if stem and stem.group(2) != version:
            path.unlink()


def render_capabilities(root: Path) -> None:
    """Re-render the two documents generated from ``hosts/capabilities.yaml``."""
    subprocess.run(
        [sys.executable, str(root / SKILL / "hosts" / "render_capabilities.py")],
        check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )


def _version_key(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def cmd_set(version: str, binary: Optional[str], root: Path = REPO,
            lock_resolver: Callable[[Path, str], Tuple[str, str]] = resolve_lock,
            capture: Callable[[Path, str, str, Path], None] = recapture,
            finder: Callable[[str, Path, Optional[str]], str] = find_binary,
            render: Callable[[Path], None] = None,
            allow_downgrade: bool = False) -> int:
    render = render or render_capabilities
    if not VERSION_RE.fullmatch(version):
        raise BumpError(f"{version!r} is not an x.y.z release")
    old = read_pins(root)
    current = old["adapter verified_version"]
    if _version_key(version) < _version_key(current) and not allow_downgrade:
        raise BumpError(
            f"{version} is older than the current pin {current}; opencode supports "
            f"its newest release only. Pass --allow-downgrade (make opencode-bump "
            f"VERSION={version} ALLOW_DOWNGRADE=1) if that is really the intent."
        )
    with tempfile.TemporaryDirectory(prefix="opencode-bump-") as tmp:
        scratch = Path(tmp)
        found = finder(version, scratch, binary)
        manifest, lock_text = lock_resolver(root, version)
        capture(root, found, version, scratch)
    touched = write_pins(root, version)
    (root / CONTAINER / "package.json").write_text(manifest, encoding="utf-8")
    (root / LOCKFILE).write_text(lock_text, encoding="utf-8")
    touched.append(str(LOCKFILE))
    render(root)
    print(f"opencode-bump: every pin is now {version} (was: "
          + ", ".join(sorted(set(old.values()))) + ")")
    print("  rewrote: " + ", ".join(sorted(set(touched))))
    print(f"  re-captured: {FIXTURES}/capture/{version}/, auth_list.{version}.json, "
          f"debug_config.{version}.json")
    print("Next, by hand:")
    print("  1. the suite: uv run --project skills/multi-llm -- pytest skills/multi-llm/tests "
          '-m "not live and not container and not package" -p no:xdist -q')
    print("     A red parser/permission test is this release's change: read it, never re-pin around it.")
    print("  2. make e2e-image: the pinned build is what verifies the moved lockfile")
    print("  3. re-seed the opencode row of tests/container/cli-surface.yaml against that image")
    print("     (the command `make e2e-lock-update` prints), then the container rows")
    return 0


def cmd_check(root: Path = REPO) -> int:
    pin = read_pins(root)["adapter verified_version"]
    try:
        with tempfile.TemporaryDirectory(prefix="opencode-latest-check-") as scratch:
            done = subprocess.run(
                [*NPM_VIEW_ARGV, f"--prefix={scratch}"], capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=NPM_VIEW_BUDGET_S, cwd=scratch,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"opencode-latest-check: could not ask npm: {exc}", file=sys.stderr)
        return 2
    latest = (done.stdout or "").strip()
    if done.returncode != 0 or not VERSION_RE.fullmatch(latest):
        print(f"opencode-latest-check: npm answered {done.returncode}: "
              f"{(done.stderr or done.stdout).strip()[:300]}", file=sys.stderr)
        return 2
    if _version_key(latest) > _version_key(pin):
        print(f"opencode-latest-check: npm's latest {NPM_PACKAGE} is {latest}, newer than the "
              f"pin {pin}. Run: make opencode-bump VERSION={latest}")
        return 1
    print(f"opencode-latest-check: the pin {pin} is npm's latest {NPM_PACKAGE} ({latest}).")
    return 0


# ---------------------------------------------------------------------------
# install: the pinned binary the release gate probes
# ---------------------------------------------------------------------------

#: The scratch manifest's name, and the marker ``install`` leaves in its prefix.
#: A prefix that already holds a ``package.json`` without the marker is somebody
#: else's project, and ``npm ci`` would replace its ``node_modules``: refused.
INSTALL_NAME = "mllm-opencode-pin"
INSTALL_MARKER = ".mllm-opencode-pin"
#: The scratch manifest's own version: a private package nobody publishes, so
#: any value npm accepts. Not an opencode release (those live in PINS alone).
INSTALL_MANIFEST_VERSION = "0.0.0"

#: ``npm ci``'s budget. The wrapper and every platform tarball npm keeps for
#: this machine (four on linux-x64, measured, each a whole ~200 MB binary
#: unpacked); a hung registry connection is the case this bounds, which npm's
#: own fetch timeout does not.
NPM_CI_BUDGET_S = 900


def pinned_version(root: Path = REPO) -> str:
    """The one opencode release every pin names, or :class:`BumpError` naming each pin.

    The gate installs what the repository claims to support, so a half-landed
    bump is refused here rather than resolved by picking one of its values.
    """
    pins = read_pins(root)
    values = sorted(set(pins.values()))
    if len(values) != 1:
        detail = "; ".join(f"{name} = {value}" for name, value in sorted(pins.items()))
        raise BumpError(
            f"the opencode pins disagree ({', '.join(values)}), so there is no one "
            f"release to install: {detail}. Finish the bump first "
            f"(make opencode-bump VERSION=<x.y.z>)."
        )
    return values[0]


def install_subset(lock: Dict, version: str, registry: Optional[str] = None) -> Tuple[Dict, Dict]:
    """The scratch ``(package.json, package-lock.json)`` for ``opencode-ai@version`` alone.

    Copied out of the container lockfile, never resolved: the wrapper's entry
    and the entry of every platform package its ``optionalDependencies`` name,
    each verbatim, ``resolved`` and ``integrity`` included. ``npm ci`` installs
    exactly a lockfile, verifies every tarball against its ``integrity`` and
    skips the platform entries whose ``os``/``cpu`` do not match, so what lands
    is the committed bytes for this machine or nothing. Every other CLI in that
    lockfile stays out: the gate needs one binary, not the image's toolchain.

    Refused, by name: no wrapper entry, a wrapper whose version is not the pin,
    a wrapper with a hard dependency (a subset without its entry could not
    install), a platform package the lockfile names but does not carry, or any
    entry without a ``sha512-`` integrity and a ``resolved`` URL to check. With
    ``registry``, also an entry whose ``resolved`` is not under it: ``npm ci``
    fetches a lockfile URL on another host from that host, ``--registry``
    notwithstanding. (:func:`verify_installed` checks the same of what npm
    actually recorded, after the install.)
    """
    packages = lock.get("packages") or {}
    root_key = f"node_modules/{NPM_PACKAGE}"
    wrapper = packages.get(root_key)
    if not isinstance(wrapper, dict):
        raise BumpError(f"{LOCKFILE} has no {root_key} entry")
    if wrapper.get("version") != version:
        raise BumpError(f"{LOCKFILE} {root_key} is {wrapper.get('version')!r}, not the pin {version}")
    if wrapper.get("dependencies"):
        raise BumpError(
            f"{LOCKFILE} {root_key} now has hard dependencies "
            f"({', '.join(sorted(wrapper['dependencies']))}); this subset carries "
            "only its platform packages, so extend install_subset deliberately"
        )
    subset: Dict[str, Dict] = {root_key: wrapper}
    for name, wanted in sorted((wrapper.get("optionalDependencies") or {}).items()):
        key = f"node_modules/{name}"
        platform = packages.get(key)
        if not isinstance(platform, dict):
            raise BumpError(f"{LOCKFILE} names the platform package {name} but carries no {key} entry")
        if platform.get("version") != wanted:
            raise BumpError(f"{LOCKFILE} {key} is {platform.get('version')!r}, not {wanted}")
        subset[key] = platform
    for key, entry in subset.items():
        if not str(entry.get("integrity") or "").startswith("sha512-") or not entry.get("resolved"):
            raise BumpError(
                f"{LOCKFILE} {key} carries no sha512 integrity and resolved URL, "
                "so npm could not verify what it downloads"
            )
        if registry is not None and not str(entry["resolved"]).startswith(registry.rstrip("/") + "/"):
            raise BumpError(
                f"{LOCKFILE} {key} resolves to {entry['resolved']}, not under the single "
                f"registry {registry}: npm ci would fetch it from there"
            )
    dependencies = {NPM_PACKAGE: version}
    own = {"name": INSTALL_NAME, "version": INSTALL_MANIFEST_VERSION}
    manifest = {**own, "private": True, "dependencies": dependencies}
    scratch_lock = {
        **own, "lockfileVersion": lock.get("lockfileVersion", 3), "requires": True,
        "packages": {"": {**own, "dependencies": dependencies}, **subset},
    }
    return manifest, scratch_lock


#: npm's own record of what ``npm ci`` installed, under the prefix.
HIDDEN_LOCKFILE = Path("node_modules") / ".package-lock.json"


def _platform_binary_name() -> str:
    """What a platform package's ``bin/`` holds (``postinstall.mjs``'s ``sourceBinary``)."""
    return "opencode.exe" if os.name == "nt" else "opencode"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: ``postinstall.mjs``'s ``platformMap`` and ``archMap``, keyed by what Python
#: reports (``sys.platform``, ``platform.machine()``) rather than Node.
_SYSTEMS = {"linux": "linux", "darwin": "darwin", "win32": "windows"}
_ARCHES = {"x86_64": "x64", "amd64": "x64", "x64": "x64",
           "aarch64": "arm64", "arm64": "arm64", "armv7l": "arm", "arm": "arm"}


def _supports_avx2(system: str, arch: str) -> bool:
    """``postinstall.mjs``'s ``supportsAvx2()``: ``/proc/cpuinfo`` on Linux,
    ``sysctl -n hw.optional.avx2_0`` on macOS. On Windows it asks PowerShell;
    here the answer is no, which puts the ``-baseline`` build first, the one
    every x64 CPU runs (the release path is Linux)."""
    if arch != "x64":
        return False
    if system == "linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return re.search(r"(^|\s)avx2(\s|$)", cpuinfo, re.I) is not None
    if system == "darwin":
        try:
            done = subprocess.run(["sysctl", "-n", "hw.optional.avx2_0"], capture_output=True,
                                  text=True, stdin=subprocess.DEVNULL, timeout=1.5)
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0 and done.stdout.strip() == "1"
    return False


def _is_musl(system: str) -> bool:
    """``postinstall.mjs``'s ``isMusl()``: ``/etc/alpine-release``, else
    ``ldd --version`` naming musl on either stream (musl's ``ldd`` prints its
    banner on stderr)."""
    if system != "linux":
        return False
    if Path("/etc/alpine-release").exists():
        return True
    try:
        done = subprocess.run(["ldd", "--version"], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return "musl" in f"{done.stdout or ''}{done.stderr or ''}".lower()


def platform_candidates(system: Optional[str] = None, arch: Optional[str] = None, *,
                        avx2: Optional[bool] = None, musl: Optional[bool] = None) -> List[str]:
    """The platform packages ``opencode-ai``'s ``postinstall.mjs`` tries, in its
    order (``packageNames()``, read from the pinned release): ``opencode-<os>-<arch>``,
    the ``-baseline`` build first on an x64 CPU without AVX2, the ``-musl``
    builds first under musl. The install script never runs here (``npm ci
    --ignore-scripts``), so this makes its choice without executing anything it
    would have fetched. Each argument defaults to this machine; the suite
    passes them.
    """
    raw_system = sys.platform if system is None else system
    system = _SYSTEMS.get(raw_system, raw_system)
    raw_arch = (_machine() if arch is None else arch).lower()
    arch = _ARCHES.get(raw_arch, raw_arch)
    base = f"opencode-{system}-{arch}"
    baseline = arch == "x64" and not (_supports_avx2(system, arch) if avx2 is None else avx2)
    if system == "linux":
        if _is_musl(system) if musl is None else musl:
            if arch == "x64":
                return ([f"{base}-baseline-musl", f"{base}-musl", f"{base}-baseline", base]
                        if baseline else
                        [f"{base}-musl", f"{base}-baseline-musl", base, f"{base}-baseline"])
            return [f"{base}-musl", base]
        if arch == "x64":
            return ([f"{base}-baseline", base, f"{base}-baseline-musl", f"{base}-musl"]
                    if baseline else
                    [base, f"{base}-baseline", f"{base}-musl", f"{base}-baseline-musl"])
        return [base, f"{base}-musl"]
    if arch == "x64":
        return [f"{base}-baseline", base] if baseline else [base, f"{base}-baseline"]
    return [base]


def verify_installed(prefix: Path, subset_lock: Dict, registry: str,
                     candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    """What ``npm ci --ignore-scripts`` left under ``prefix`` is the committed
    subset; returns the binary to run, a verified platform package's own file.

    ``npm ci`` verifies each tarball it downloads against the lockfile's
    ``integrity``, and that is all it vouches for. So, refusing by file:

    * npm's record of the install, ``node_modules/.package-lock.json``, lists
      only packages of the subset, each ``resolved`` under ``registry`` and
      with the committed ``integrity`` (npm 10 lists what it installed; the
      workflow's pinned npm 11 lists the platform packages it skipped too,
      measured: each entry is held to the same);
    * every package directory in ``node_modules`` is one that record lists;
    * the binary is ``bin/opencode`` (``opencode.exe`` on Windows) of the first
      package in ``candidates`` (:func:`platform_candidates` by default: the
      install script's own order) that the record verified and that holds one,
      a regular file whose symlinks, if any, stay inside that package.

    Nothing of the wrapper's is used. With its install script off, its
    ``bin/opencode.exe``, and the ``.bin/opencode`` link npm makes to it, is
    the placeholder script it ships ("postinstall was not run"); what the script
    would have linked there, or fetched with no committed integrity when the
    link failed its ``--version``, is exactly what this choice replaces, and
    nothing has run yet when it is made.

    Returns the binary's absolute path, its package, the order it was chosen
    from, and its sha256.
    """
    prefix = Path(prefix).absolute()
    hidden_path = prefix / HIDDEN_LOCKFILE
    try:
        hidden = json.loads(hidden_path.read_text(encoding="utf-8")).get("packages") or {}
    except (OSError, ValueError, AttributeError) as exc:
        raise BumpError(f"{hidden_path}: npm's record of the install is unreadable ({exc})") from exc
    committed = subset_lock.get("packages") or {}
    prefix_url = registry.rstrip("/") + "/"
    verified: List[str] = []
    for key, entry in sorted(hidden.items()):
        if not key:
            continue
        if not isinstance(entry, dict):
            raise BumpError(f"{hidden_path}: {key} is not an entry")
        if key not in committed:
            raise BumpError(f"{hidden_path}: {key} is installed but is not in the committed "
                            "subset npm ci was given")
        resolved = str(entry.get("resolved") or "")
        if not resolved.startswith(prefix_url):
            raise BumpError(f"{hidden_path}: {key} came from {resolved or '<nowhere recorded>'}, "
                            f"not the single registry {registry}")
        if entry.get("integrity") != committed[key].get("integrity"):
            raise BumpError(f"{hidden_path}: {key} was installed with integrity "
                            f"{entry.get('integrity')!r}, not the committed "
                            f"{committed[key].get('integrity')!r}")
        verified.append(key)
    modules = prefix / "node_modules"
    present: List[str] = []
    for child in sorted(modules.iterdir()) if modules.is_dir() else []:
        # `.bin`, npm's record itself, any cache npm keeps there: no package
        # name starts with a dot, so nothing there is a package Node loads.
        if child.name.startswith("."):
            continue
        if child.name.startswith("@") and child.is_dir():
            present += [f"node_modules/{child.name}/{inner.name}" for inner in sorted(child.iterdir())]
        else:
            present.append(f"node_modules/{child.name}")
    stray = [key for key in present if key not in verified]
    if stray:
        raise BumpError(f"{modules} holds {', '.join(stray)}, which {hidden_path} does not list: "
                        "npm ci did not put it there")

    order = list(platform_candidates() if candidates is None else candidates)
    wrapper_key = f"node_modules/{NPM_PACKAGE}"
    platforms = [key for key in verified if key != wrapper_key]
    for name in order:
        key = f"node_modules/{name}"
        if key not in platforms:
            continue
        package_dir = prefix / key
        binary = package_dir / "bin" / _platform_binary_name()
        if not binary.is_file():
            continue
        real = os.path.realpath(binary)
        if os.path.commonpath([real, os.path.realpath(package_dir)]) != os.path.realpath(package_dir):
            raise BumpError(f"{binary} resolves to {real}, outside {key}: refusing to run it")
        return {"binary": str(binary), "platform_package": key, "chosen_from": order,
                "sha256": _sha256_file(Path(real))}
    raise BumpError(
        f"no platform package {NPM_PACKAGE}'s install script would try on this machine "
        f"({', '.join(order) or 'none'}) is installed, verified and holds "
        f"bin/{_platform_binary_name()} under {modules} (verified: "
        f"{', '.join(key.split('/', 1)[1] for key in platforms) or 'none'}). Refusing to "
        "run anything."
    )


def _single_registry(url: Optional[str]) -> str:
    """``url`` checked against ``Makefile`` ``OPENCODE_REGISTRY`` (the default)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "opencode_registry", Path(__file__).resolve().parent / "registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.require_single(url) if url else module.single_registry()
    except module.RegistryError as exc:
        raise BumpError(str(exc)) from exc


def _npm_ci(prefix: Path, registry: str) -> None:
    """``npm ci --ignore-scripts`` in ``prefix``, with nothing of the caller's
    npm configuration.

    The registry is a command-line flag, which outranks every other layer, and
    the user config is an empty file: a maintainer's ``.npmrc`` could otherwise
    carry a scoped registry or a proxy setting this gate never chose. Inherited
    ``npm_config_*`` variables are dropped for the same reason, since npm's
    environment layer outranks the user file.

    The project layer is the prefix's own, and empty. ``npm ci`` runs in the
    prefix, which holds the scratch ``package.json``, so npm's walk up from its
    cwd stops there at once and the project file is ``<prefix>/.npmrc``, which
    nothing writes. ``--prefix`` names it on the command line too, because the
    walk does not always stop at the first ``package.json``: an ancestor whose
    ``package.json`` lists the prefix among its ``workspaces`` moves the
    project root up to itself, its ``.npmrc`` with it (measured on this
    machine's npm 10 and the publish job's pinned npm 11), while a command-line
    prefix is the root outright.

    ``--ignore-scripts``: ``opencode-ai``'s ``postinstall`` runs the binary it
    links, and when that fails fetches an unverified one and runs that, before
    :func:`verify_installed` could refuse anything. The binary is chosen
    afterwards, from the verified platform packages, instead.
    """
    empty_rc = prefix / ".npmrc-empty"
    empty_rc.write_text("", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.lower().startswith("npm_config_")}
    subprocess.run(
        ["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund", f"--registry={registry}",
         f"--userconfig={empty_rc}", f"--prefix={prefix}"],
        cwd=str(prefix), env=env, check=True, stdin=subprocess.DEVNULL,
        stdout=sys.stderr, timeout=NPM_CI_BUDGET_S,
    )


def cmd_install(prefix: Path, registry: Optional[str] = None, root: Path = REPO,
                npm_ci: Callable[[Path, str], None] = _npm_ci,
                version_reader: Callable[[str], str] = None) -> int:
    """Install the pinned opencode under ``prefix``; print its path last."""
    version_reader = version_reader or _read_version
    version = pinned_version(root)
    url = _single_registry(registry)
    lock = json.loads((root / LOCKFILE).read_text(encoding="utf-8"))
    manifest, scratch_lock = install_subset(lock, version, registry=url)

    prefix = Path(prefix).absolute()
    if (prefix / "package.json").exists() and not (prefix / INSTALL_MARKER).exists():
        raise BumpError(
            f"{prefix} already holds a package.json that this command did not "
            "write; npm ci would replace its node_modules. Name an empty or "
            "previously used --prefix."
        )
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / INSTALL_MARKER).write_text(f"{NPM_PACKAGE}@{version}\n", encoding="utf-8")
    (prefix / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (prefix / "package-lock.json").write_text(json.dumps(scratch_lock, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"opencode-install: npm ci --ignore-scripts {NPM_PACKAGE}@{version} under {prefix} "
          f"from {url} (not global)", file=sys.stderr)
    npm_ci(prefix, url)

    # Before anything runs: the subset as committed, and the binary a verified
    # platform package's own.
    matched = verify_installed(prefix, scratch_lock, url)
    binary = matched["binary"]
    print(f"opencode-install: {binary} is {matched['platform_package']}'s binary, first of "
          f"{', '.join(matched['chosen_from'])} installed (sha256 {matched['sha256']})",
          file=sys.stderr)
    answered = version_reader(binary)
    if answered != version:
        raise BumpError(
            f"{binary} answers --version {answered!r}, not the pin {version}: "
            "refusing to hand the gate a different host than the one the "
            "repository claims"
        )
    print(binary)
    return 0


def cmd_verify_install(prefix: Path, registry: Optional[str] = None) -> int:
    """:func:`verify_installed` on a prefix ``npm ci`` already ran in, against
    the subset lockfile it holds (``npm ci`` never rewrites that file). The
    chosen binary's path is the last stdout line."""
    url = _single_registry(registry)
    prefix = Path(prefix).absolute()
    lock_path = prefix / "package-lock.json"
    try:
        scratch_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BumpError(f"{lock_path}: the subset lockfile is unreadable ({exc})") from exc
    matched = verify_installed(prefix, scratch_lock, url)
    print(f"opencode-install: {matched['binary']} is {matched['platform_package']}'s binary "
          f"(sha256 {matched['sha256']})")
    print(matched["binary"])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    set_parser = sub.add_parser("set", help="move every pin to VERSION and re-capture")
    set_parser.add_argument("version")
    set_parser.add_argument("--bin", help="an opencode binary of that version")
    set_parser.add_argument("--allow-downgrade", action="store_true",
                            help="accept a VERSION older than the current pin")
    sub.add_parser("check", help="exit 1 when npm has a newer opencode than the pin")
    install_parser = sub.add_parser(
        "install", help="npm ci the pinned opencode under PREFIX; print its path last")
    install_parser.add_argument("--prefix", required=True, type=Path,
                                help="scratch directory (never a global install)")
    install_parser.add_argument("--registry", default=None,
                                help="must be Makefile OPENCODE_REGISTRY (the default)")
    verify_parser = sub.add_parser(
        "verify-install", help="check what npm ci left under PREFIX (install's own check)")
    verify_parser.add_argument("--prefix", required=True, type=Path,
                               help="a prefix npm ci ran in with the subset lockfile")
    verify_parser.add_argument("--registry", default=None,
                               help="must be Makefile OPENCODE_REGISTRY (the default)")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return cmd_check()
        if args.command == "install":
            return cmd_install(args.prefix, args.registry)
        if args.command == "verify-install":
            return cmd_verify_install(args.prefix, args.registry)
        return cmd_set(args.version, args.bin, allow_downgrade=args.allow_downgrade)
    except (BumpError, subprocess.CalledProcessError) as exc:
        print(f"opencode-bump: {exc}", file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError) as exc:
        # `install` only: npm missing, `npm ci` past its budget, a binary that
        # cannot be executed or never answers `--version`. A refusal, not a trace.
        print(f"opencode-bump: {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
