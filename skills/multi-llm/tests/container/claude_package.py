"""The Claude Code package route, inside the container: install, load, dispatch, upgrade.

Every other claude-code row stages the source tree as a PLAIN SKILL, a symlink under
``$HOME/.claude/skills/multi-llm`` (`entrypoint.stage_host`). That route loads no plugin
manifest and no plugin hooks, and it runs ``/opt/mllm/skill``, which is not what anyone
installs. The published install is the generated tree at ``packaging/claude/multi-llm``,
named by ``.claude-plugin/marketplace.json``, fetched by ``claude plugin marketplace add``.
This module runs that.

THE INSTALL IS ALWAYS THROUGH A GIT MARKETPLACE, never ``marketplace add <directory>``.
MEASURED (claude 2.1.318): a directory marketplace records ``readFromFolder`` and every
session loads the plugin straight from that folder, so the cache copy is never read and
the row would not exercise what a user gets. A Git marketplace loads from
``~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/``, as on a real machine.

The Git source is offline. The CLI refuses ``file://`` and ``ssh://``, and accepts the
scp-like ``git@host:/path``, whose transport goes through ``GIT_SSH_COMMAND``. So a shim
runs the remote command locally, against a bare repository this module builds from the
image's mirror. That is also the transport on which the CLI asks for a partial clone
(measured: plain ``http://`` gets no blob filter), so the published sparse form behaves
here as it does against GitHub, with no route out of the container.

Everything here RECORDS and nothing here judges, as in the codex probes:
``tests/test_container_claude_package.py`` grades the JSON this leaves in the bundle.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

#: Dockerfile §10d. The two paths inside it are the ones they have in the repository.
MARKETPLACE_DIR = Path("/opt/mllm/claude-marketplace")
MARKETPLACE_NAME = "beastlabai"
PLUGIN_NAME = "multi-llm"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
PLUGIN_PATH = "packaging/claude/multi-llm"
CATALOG_PATH = ".claude-plugin/marketplace.json"
MANIFEST_PATH = f"{PLUGIN_PATH}/.claude-plugin/plugin.json"
#: The published `--sparse` paths, in the order docs/installation.md prints them.
SPARSE_PATHS = (".claude-plugin", PLUGIN_PATH)
#: The skill's name as a session advertises it: namespaced by the plugin.
SKILL_NAME = f"{PLUGIN_NAME}:{PLUGIN_NAME}"

#: Outside both sparse paths. None may reach the disk, and on a partial clone the first
#: is the claim that matters most: a test file's blob is not even downloaded.
DECOY_TEST = "skills/multi-llm/tests/test_decoy.py"
DECOYS = {
    DECOY_TEST: "# Stands in for the suite a sparse registration must not fetch.\n",
    "plans/decoy-plan.md": "# Stands in for repository text no install should carry.\n",
}
#: Commit B's marker: present only once the upgrade reached the next version.
NEXT_MARKER = "NEXT-RELEASE-MARKER"
#: Commit C's marker: a shipped file moved under an unmoved version.
SAME_VERSION_MARKER = "SAME-VERSION-MARKER"

#: Things Claude Code or a run leaves in a plugin directory that are not the plugin.
RUNTIME_ENTRIES = frozenset({".in_use", ".orphaned_at", ".venv", "__pycache__"})

PROBE_SEED = "claude-seed"
PROBE_SESSION = "claude-session"
PROBE_INVOCATION = "claude-invocation"
PROBE_DISPATCH = "claude-dispatch"
PROBE_UPGRADE = "claude-upgrade"
PROBES = (PROBE_SEED, PROBE_SESSION, PROBE_INVOCATION, PROBE_DISPATCH, PROBE_UPGRADE)

OUTPUT_CAP = 20_000
PLUGIN_INSTALL_RECORD = "plugin-install.json"
PLUGIN_LIST_RECORD = "claude-plugin-list.json"

#: A sentence only the skill's BODY carries, never its frontmatter description, which
#: every session is told about whether or not the skill was invoked.
SKILL_BODY_SENTINEL = "how this host answers each one"

# A local stand-in for ssh: run the remote command, which is the last argument. Run as
# `sh <file>` because the tmpfs $HOME is noexec.
_SSH_SHIM = (
    "# A local stand-in for ssh: run the remote command (the last argument).\n"
    'eval "remote=\\${$#}"\n'
    'exec sh -c "$remote"\n'
)


@dataclass(frozen=True)
class Context:
    """What the entrypoint knows and this module needs, passed rather than imported:
    the entrypoint imports this module, and is `__main__` inside the image."""

    home: Path
    out: Path
    workspace: Path
    real_dir: Path
    log: Callable[[str], None] = print
    mirror: Path = MARKETPLACE_DIR


# --------------------------------------------------------------------------- #
# Running things
# --------------------------------------------------------------------------- #

def claude_env(home: Path, overlay: Optional[dict] = None) -> dict:
    env = dict(os.environ)
    env.update({"HOME": str(home), "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
    env.update(overlay or {})
    return env


def _capture(argv: list[str], public_argv: list[str], env: dict, cwd: Path,
             timeout: float) -> dict:
    started = time.time()
    try:
        done = subprocess.run(argv, env=env, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        code, out, err, timed_out = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, timed_out = None, True
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError as exc:
        code, out, err, timed_out = None, "", str(exc), False
    return {
        "argv": public_argv,
        "exit": code,
        "timed_out": timed_out,
        "wall_clock_s": round(time.time() - started, 2),
        "stdout": out[:OUTPUT_CAP],
        "stdout_truncated": len(out) > OUTPUT_CAP,
        "stderr": err[:OUTPUT_CAP],
        "stderr_truncated": len(err) > OUTPUT_CAP,
        "_stdout_full": out,
    }


def run_claude(ctx: Context, args: list[str], env: dict, timeout: float = 180.0) -> dict:
    """One claude command by its REAL path, stdin closed.

    By absolute path because the PATH wrapper rebuilds a child's environment for a
    dispatched provider, and every question here is about the HOST's own $HOME. Stdin
    is closed because `claude -p` reads it, and an inherited pipe is a hang.
    """
    return _capture([str(ctx.real_dir / "claude"), *args], ["claude", *args], env,
                    ctx.workspace, timeout)


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def json_or_none(text: Optional[str]):
    try:
        return json.loads(text or "")
    except ValueError:
        return None


def last_json_line(text: Optional[str]):
    """`--json` prints its one result as the LAST line, under any progress lines."""
    for line in reversed((text or "").splitlines()):
        value = json_or_none(line)
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------- #
# The offline Git marketplace
# --------------------------------------------------------------------------- #

def _fixture_env(home: Path) -> dict:
    """No system or global git config, a fixed identity and a fixed clock."""
    stamp = f"@{os.environ.get('SOURCE_DATE_EPOCH', '1767225600')} +0000"
    env = dict(os.environ)
    env.update({
        "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "mllm-fixture", "GIT_AUTHOR_EMAIL": "fixture@invalid",
        "GIT_COMMITTER_NAME": "mllm-fixture", "GIT_COMMITTER_EMAIL": "fixture@invalid",
        "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp,
    })
    return env


def _git(args: list[str], cwd: Path, env: dict, check: bool = True) -> dict:
    """One git command. `check` raises: every fixture step is a precondition, and a
    half-built fixture graded as a Claude failure would be the rig's bug reported as
    the product's."""
    git = shutil.which("git") or "/usr/bin/git"
    done = subprocess.run([git, *args], cwd=str(cwd), env=env, capture_output=True,
                          text=True, timeout=180, stdin=subprocess.DEVNULL)
    result = {"argv": ["git", *args], "exit": done.returncode, "stdout": done.stdout,
              "stderr": done.stderr[:OUTPUT_CAP]}
    if check and done.returncode != 0:
        raise SystemExit(f"claude_package: git {' '.join(args)} failed in {cwd}: "
                         f"{done.stderr[:2000]}")
    return result


def next_patch(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(.*)", version or "")
    if not match:
        raise SystemExit(f"claude_package: the manifest version {version!r} is not semver")
    major, minor, patch, _rest = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def build_fixture(home: Path, mirror: Path = MARKETPLACE_DIR) -> dict:
    """Commit A: the mirror unedited, plus decoys. Returns how to reach it.

    Served from a bare clone configured as GitHub behaves. Without `allowFilter` the
    partial clone would silently fetch every blob. Without `allowAnySHA1InWant` the
    checkout's on-demand fetch of each blob by id is refused as unadvertised: the shim's
    `simple` ssh variant cannot carry protocol v2, which allows that on its own.
    """
    root = home / "git-marketplace"
    if root.exists():
        shutil.rmtree(root)
    source = root / "src"
    source.mkdir(parents=True)
    env = _fixture_env(home)
    _git(["init", "-q", "-b", "master", str(source)], root, env)
    catalog = source / CATALOG_PATH
    catalog.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mirror / CATALOG_PATH, catalog)
    shutil.copytree(mirror / PLUGIN_PATH, source / PLUGIN_PATH, symlinks=True)
    for rel, text in DECOYS.items():
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(["add", "-A"], source, env)
    _git(["commit", "-q", "-m", "the release"], source, env)
    release_sha = _git(["rev-parse", "HEAD"], source, env)["stdout"].strip()
    bare = root / "beastlabai-multi-llm-plugin.git"
    _git(["clone", "-q", "--bare", str(source), str(bare)], root, env)
    _git(["config", "uploadpack.allowFilter", "true"], bare, env)
    _git(["config", "uploadpack.allowAnySHA1InWant", "true"], bare, env)
    _git(["remote", "add", "bare", str(bare)], source, env)
    shim = root / "ssh-shim.sh"
    shim.write_text(_SSH_SHIM, encoding="utf-8")
    manifest = json.loads((source / MANIFEST_PATH).read_text(encoding="utf-8"))
    return {
        "root": str(root), "source": str(source), "bare": str(bare),
        "url": f"git@localhost:{bare}",
        "sparse": list(SPARSE_PATHS),
        "release_sha": release_sha,
        "release_version": str(manifest.get("version")),
        "decoys": sorted(DECOYS),
        # `sh <file>` because the tmpfs $HOME is noexec. `simple` because git would
        # otherwise probe this command for ssh's option dialect.
        "env": {"GIT_SSH_COMMAND": f"sh {shim}", "GIT_SSH_VARIANT": "simple"},
    }


def advance_fixture(home: Path, fixture: dict, *, version: Optional[str], marker: str,
                    message: str) -> dict:
    """A later commit on the marketplace's default branch: a marker in the plugin, and
    a new manifest version unless `version` is None."""
    source = Path(fixture["source"])
    env = _fixture_env(home)
    if version is not None:
        path = source / MANIFEST_PATH
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["version"] = version
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (source / PLUGIN_PATH / marker).write_text(f"{message}\n", encoding="utf-8")
    _git(["add", "-A"], source, env)
    _git(["commit", "-q", "-m", message], source, env)
    _git(["push", "-q", "bare", "master"], source, env)
    return {"sha": _git(["rev-parse", "HEAD"], source, env)["stdout"].strip(),
            "version": version, "marker": marker}


# --------------------------------------------------------------------------- #
# What the install left
# --------------------------------------------------------------------------- #

def inventory(root: Optional[Path]) -> dict:
    """Files and bytes of the plugin itself, with run-time by-products set aside."""
    if not root or not Path(root).is_dir():
        return {"files": 0, "bytes": 0, "tests": [], "symlinks": []}
    files = bytes_ = 0
    tests: list[str] = []
    symlinks: list[str] = []
    for path, names, filenames in os.walk(root):
        names[:] = [n for n in names if n not in RUNTIME_ENTRIES]
        for name in filenames:
            if name in RUNTIME_ENTRIES:
                continue
            full = Path(path) / name
            rel = full.relative_to(root).as_posix()
            if full.is_symlink():
                symlinks.append(rel)
                continue
            files += 1
            bytes_ += full.stat().st_size
            if "/tests/" in f"/{rel}" or name.startswith("test_"):
                tests.append(rel)
    return {"files": files, "bytes": bytes_, "tests": sorted(tests),
            "symlinks": sorted(symlinks)}


def compare_trees(installed: Optional[Path], expected: Path) -> dict:
    """The installed plugin against the tree it should equal: bytes and executable bit.

    The bit is compared because a hook script that lost it is a hook that never runs,
    and a byte comparison alone would call that install perfect. It is read from the
    file's mode and never through `os.access`, which also answers for the mount: the two
    trees sit on differently mounted tmpfs, so identical files would be reported apart.
    """
    def listing(root: Path) -> dict:
        found = {}
        for path, names, filenames in os.walk(root):
            names[:] = [n for n in names if n not in RUNTIME_ENTRIES]
            for name in filenames:
                if name not in RUNTIME_ENTRIES:
                    full = Path(path) / name
                    found[full.relative_to(root).as_posix()] = full
        return found

    if not installed or not Path(installed).is_dir():
        return {"matches": False, "reason": f"{installed} is not a directory",
                "only_installed": [], "only_expected": [], "differ": [], "mode_differs": []}
    have, want = listing(Path(installed)), listing(expected)
    differ, mode_differs = [], []
    for rel in sorted(set(have) & set(want)):
        if have[rel].read_bytes() != want[rel].read_bytes():
            differ.append(rel)
        if (have[rel].stat().st_mode & stat.S_IXUSR) != (want[rel].stat().st_mode & stat.S_IXUSR):
            mode_differs.append(rel)
    only_installed = sorted(set(have) - set(want))
    only_expected = sorted(set(want) - set(have))
    return {"matches": not (only_installed or only_expected or differ or mode_differs),
            "only_installed": only_installed, "only_expected": only_expected,
            "differ": differ, "mode_differs": mode_differs, "compared": len(want)}


def checkout_facts(home: Path) -> dict:
    """The marketplace's own clone: what sparse and partial really did on disk."""
    checkout = home / ".claude" / "plugins" / "marketplaces" / MARKETPLACE_NAME
    if not (checkout / ".git").exists():
        return {"exists": False}
    env = _fixture_env(home)
    config = lambda key: _git(["config", "--get", key], checkout, env,  # noqa: E731
                              check=False)["stdout"].strip()
    blob = _git(["cat-file", "-e", f"HEAD:{DECOY_TEST}"], checkout,
                {**env, "GIT_NO_LAZY_FETCH": "1"}, check=False)
    return {
        "exists": True,
        "path": str(checkout),
        "sparse_list": _git(["sparse-checkout", "list"], checkout, env,
                            check=False)["stdout"].split(),
        "promisor": config("remote.origin.promisor"),
        "partial_clone_filter": config("remote.origin.partialclonefilter"),
        "decoys_on_disk": sorted(rel for rel in DECOYS if (checkout / rel).exists()),
        # True would mean the suite's blobs came down the wire after all.
        "decoy_blob_fetched": blob["exit"] == 0,
    }


def registration_facts(home: Path) -> dict:
    """Both places the CLI recorded the marketplace: the settings entry a later `add`
    must agree with, and the known-marketplaces record `update` works from."""
    plugins = home / ".claude" / "plugins"
    known = json_or_none(_read(plugins / "known_marketplaces.json")) or {}
    settings = json_or_none(_read(home / ".claude" / "settings.json")) or {}
    declared = (settings.get("extraKnownMarketplaces") or {}).get(MARKETPLACE_NAME) or {}
    return {"known": known.get(MARKETPLACE_NAME),
            "settings_entry": declared,
            "enabled": (settings.get("enabledPlugins") or {}).get(PLUGIN_ID)}


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def list_entry(document) -> Optional[dict]:
    for entry in document if isinstance(document, list) else []:
        if isinstance(entry, dict) and entry.get("id") == PLUGIN_ID:
            return entry
    return None


# --------------------------------------------------------------------------- #
# The seed
# --------------------------------------------------------------------------- #

def seed(ctx: Context) -> dict:
    """Register, install and list, as docs/installation.md prints them.

    Writes `plugin-install.json` and returns it. Raises when no installed directory
    resulted: every row below stands on it, so that is a stopped row rather than one
    grading an empty home.

    Run after the settings file is written: `marketplace add` and `plugin install` merge
    their entries into it, and a later overwrite would silently unregister the plugin.
    """
    fixture = build_fixture(ctx.home, ctx.mirror)
    env = claude_env(ctx.home, fixture["env"])
    add = run_claude(ctx, ["plugin", "marketplace", "add", fixture["url"],
                           "--sparse", *SPARSE_PATHS, "--json"], env)
    install = run_claude(ctx, ["plugin", "install", PLUGIN_ID, "--json"], env)
    listing = run_claude(ctx, ["plugin", "list", "--json"], env)
    (ctx.out / PLUGIN_LIST_RECORD).write_text(listing["_stdout_full"], encoding="utf-8")
    document = json_or_none(listing["_stdout_full"])
    entry = list_entry(document)
    root = Path(entry["installPath"]) if entry and entry.get("installPath") else None
    record = {
        "plugin_id": PLUGIN_ID,
        # WHERE the install came from: what `harness.promotable_fields` keys `package`
        # on. Always `git` on this host, for the reason in the module docstring.
        "install_source": "git",
        "marketplace_url": fixture["url"],
        "sparse_paths": list(SPARSE_PATHS),
        "commands": [public(add), public(install), public(listing)],
        "add_result": last_json_line(add["_stdout_full"]),
        "install_result": last_json_line(install["_stdout_full"]),
        "list_entry": entry,
        "version": (entry or {}).get("version"),
        "installed_root": str(root) if root else None,
        "installed_root_exists": bool(root and root.is_dir()),
        # The plain-skill route beside the plugin would be two advertised copies of one
        # skill, and a grade that could belong to either.
        "file_shim_present": (ctx.home / ".claude" / "skills" / PLUGIN_NAME).exists(),
        "installed": inventory(root),
        # Compared now, before any session or dispatch has run from it.
        "tree": compare_trees(root, Path(fixture["source"]) / PLUGIN_PATH),
        "marketplace_checkout": checkout_facts(ctx.home),
        "registration": registration_facts(ctx.home),
        "fixture": {k: v for k, v in fixture.items() if k != "env"},
        "overlay": dict(fixture["env"]),
    }
    (ctx.out / PLUGIN_INSTALL_RECORD).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    ctx.log(f"route=package: claude plugin {PLUGIN_ID} marketplace add -> {add['exit']}, "
            f"install -> {install['exit']}, list -> {listing['exit']}; installed at {root}")
    if not record["installed_root_exists"]:
        raise SystemExit(
            f"claude_package: `claude plugin install {PLUGIN_ID}` left no installed "
            f"directory (exit {install['exit']}); {PLUGIN_INSTALL_RECORD} has all three "
            f"commands. Every package row stands on this install.")
    return record


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def session_args(prompt: str, model: str) -> list[str]:
    """A headless turn that reports what it loaded.

    `claude plugin details` is not asked, because it reported 0 skills for directories
    a session loaded perfectly well. The session's own `init` event is what loaded.
    """
    return ["-p", prompt, "--output-format", "stream-json", "--verbose",
            "--include-hook-events", "--max-turns", "1", "--model", model]


def parse_session(stdout: str) -> dict:
    """The facts a `stream-json --verbose --include-hook-events` session reported."""
    init: dict = {}
    hooks: list[dict] = []
    result: dict = {}
    for line in (stdout or "").splitlines():
        event = json_or_none(line)
        if not isinstance(event, dict):
            continue
        kind, subtype = event.get("type"), event.get("subtype")
        if kind == "system" and subtype == "init":
            init = event
        elif kind == "system" and subtype == "hook_response":
            hooks.append({
                "hook_name": event.get("hook_name"),
                "hook_event": event.get("hook_event"),
                "outcome": event.get("outcome"),
                "exit_code": event.get("exit_code"),
                "output_head": str(event.get("output") or "")[:600],
                "stderr_head": str(event.get("stderr") or "")[:600],
            })
        elif kind == "result":
            result = {k: event.get(k) for k in ("subtype", "is_error", "num_turns")}
    plugin = next((p for p in init.get("plugins") or []
                   if isinstance(p, dict) and p.get("name") == PLUGIN_NAME), None)
    return {
        "init_seen": bool(init),
        "plugin": plugin,
        "skills": [s for s in init.get("skills") or [] if PLUGIN_NAME in str(s)],
        "slash_commands": [s for s in init.get("slash_commands") or []
                           if PLUGIN_NAME in str(s)],
        "plugin_errors": init.get("plugin_errors") or [],
        "hooks": hooks,
        "result": result,
    }


def session_leg(ctx: Context, overlay: dict, prompt: str = "Say hello.",
                model: str = "stub-model", timeout: float = 240.0) -> dict:
    env = claude_env(ctx.home, overlay)
    run = run_claude(ctx, session_args(prompt, model), env, timeout=timeout)
    return {"command": public(run), "session": parse_session(run["_stdout_full"])}


def invocation_facts(installed_root: str, marker: str) -> dict:
    """`{fact: regex}` for the stub's prompt probe, on the published spelling.

    ``skill_body``           the host expanded the skill into the turn. The discriminator:
                             the frontmatter description reaches every session, this
                             sentence only reaches one that invoked the skill.
    ``skill_dir_expanded``   `${CLAUDE_SKILL_DIR}` became the CACHE path, so every command
                             the skill prints will run the installed copy.
    ``argument_marker``      the typed argument reached the turn.
    ``skill_dir_unexpanded`` the NEGATIVE: the variable surviving means the shim's commands
                             are unusable as written.
    """
    return {
        "skill_body": re.escape(SKILL_BODY_SENTINEL),
        "skill_dir_expanded": re.escape(f"{installed_root}/skills/{PLUGIN_NAME}"),
        "argument_marker": re.escape(marker),
        "skill_dir_unexpanded": re.escape("${CLAUDE_SKILL_DIR}"),
    }


def stub_facts_seen(stub_log: Path) -> tuple[list[str], int]:
    """Which facts any request carried, and how many requests the host sent."""
    seen: set[str] = set()
    requests = 0
    for line in (_read(stub_log) or "").splitlines():
        entry = json_or_none(line)
        if not isinstance(entry, dict) or "facts" not in entry:
            continue
        requests += 1
        seen.update(str(name) for name in entry.get("facts") or [])
    return sorted(seen), requests


def invocation_leg(ctx: Context, overlay: dict, stub_log: Path, marker: str) -> dict:
    """Type the published `/multi-llm:multi-llm …` string at a fresh session."""
    typed = f"/{SKILL_NAME} --ask plans/auth-feature.md {marker} --quick"
    leg = session_leg(ctx, overlay, prompt=typed)
    facts, requests = stub_facts_seen(stub_log)
    leg.update({"typed": typed, "marker": marker, "facts_seen": facts,
                "stub_requests": requests})
    return leg


# --------------------------------------------------------------------------- #
# Dispatch from the installed path
# --------------------------------------------------------------------------- #

def dispatch_leg(ctx: Context, root: Path) -> dict:
    """The dispatcher, by the same `uv run` the skill prints, from the CACHE copy.

    The commands a user's first workflow depends on before any model is involved. An
    import of a module the generator left out, or an instruction file it dropped,
    fails here, while the source tree beside it would pass.
    """
    skill = root / "skills" / PLUGIN_NAME
    env = dict(os.environ)
    env["HOME"] = str(ctx.home)
    uv = ctx.real_dir / "uv"

    def dispatch(*args: str, **extra: str) -> dict:
        tail = ["--frozen", "--no-dev", "--project", str(skill), "--", "python",
                str(skill / "multi_llm.py"), *args]
        return public(_capture([str(uv), "run", *tail], ["uv", "run", *tail],
                               {**env, **extra}, ctx.workspace, 180.0))

    return {
        "skill_dir": str(skill),
        "help": dispatch("--help"),
        "instructions": dispatch("util", "instructions", "ask"),
        "doctor": dispatch("doctor", "--warn-only"),
        "install_host_check": dispatch("install-host", "claude-code", "--check",
                                       "--project-root", str(ctx.workspace)),
        "status": dispatch("status", "plans/auth-feature.md"),
    }


# --------------------------------------------------------------------------- #
# Upgrade
# --------------------------------------------------------------------------- #

def _installed_state(ctx: Context, env: dict) -> dict:
    listing = run_claude(ctx, ["plugin", "list", "--json"], env)
    entry = list_entry(json_or_none(listing["_stdout_full"]))
    root = Path(entry["installPath"]) if entry and entry.get("installPath") else None
    return {"version": (entry or {}).get("version"),
            "installed_root": str(root) if root else None,
            "markers": sorted(m for m in (NEXT_MARKER, SAME_VERSION_MARKER)
                              if root and (root / m).exists())}


def upgrade_leg(ctx: Context, overlay: dict, install: dict) -> dict:
    """The documented update, and the case the release rule exists for.

    Step by step: a new release with no `marketplace update` is invisible; with it, the
    next version installs beside the old one and sessions load it; then a shipped byte
    moves UNDER that version, and existing installs must not receive it.

    That last reading is why `packaging/claude/build.py --check-release` refuses such a
    commit. If a Claude Code release ever starts to deliver it, the rule can relax,
    and this row is what will say so.
    """
    env = claude_env(ctx.home, overlay)
    fixture = dict(install.get("fixture") or {})
    before = _installed_state(ctx, env)
    next_version = next_patch(str(fixture.get("release_version")))

    commit_b = advance_fixture(ctx.home, fixture, version=next_version, marker=NEXT_MARKER,
                               message=f"release {next_version}")
    plugin_only = run_claude(ctx, ["plugin", "update", PLUGIN_ID], env)
    after_plugin_only = _installed_state(ctx, env)
    marketplace = run_claude(ctx, ["plugin", "marketplace", "update", MARKETPLACE_NAME], env)
    update = run_claude(ctx, ["plugin", "update", PLUGIN_ID], env)
    after_update = _installed_state(ctx, env)
    source = Path(fixture["source"]) / PLUGIN_PATH
    tree = compare_trees(Path(after_update["installed_root"] or ""), source)
    session = session_leg(ctx, overlay)

    commit_c = advance_fixture(ctx.home, fixture, version=None, marker=SAME_VERSION_MARKER,
                               message="a shipped byte moved under an unmoved version")
    marketplace_c = run_claude(ctx, ["plugin", "marketplace", "update", MARKETPLACE_NAME], env)
    update_c = run_claude(ctx, ["plugin", "update", PLUGIN_ID], env)
    after_same_version = _installed_state(ctx, env)

    return {
        "before": before,
        "next_version": next_version,
        "commit_b": commit_b,
        "plugin_update_alone": {"command": public(plugin_only), "state": after_plugin_only},
        "marketplace_update": public(marketplace),
        "plugin_update": public(update),
        "after_update": after_update,
        "tree_after_update": tree,
        "old_version_kept": bool(before["installed_root"]
                                 and Path(before["installed_root"]).is_dir()),
        "session_after_update": session,
        "commit_c": commit_c,
        "marketplace_update_same_version": public(marketplace_c),
        "plugin_update_same_version": public(update_c),
        "after_same_version": after_same_version,
    }


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #

def run_probe(probe: str, ctx: Context, overlay: dict, stub_log: Path,
              marker: str = "") -> dict:
    """Run one zero-quota row and write `probe-<name>.json`. Recorded, not judged."""
    if probe not in PROBES:
        raise SystemExit(f"claude_package: unknown probe {probe!r}; known: {', '.join(PROBES)}")
    install = json_or_none(_read(ctx.out / PLUGIN_INSTALL_RECORD)) or {}
    root = Path(install["installed_root"]) if install.get("installed_root") else None
    record: dict = {"probe": probe, "installed_root": str(root) if root else None}
    if probe == PROBE_SESSION:
        record.update(session_leg(ctx, overlay))
    elif probe == PROBE_INVOCATION:
        record.update(invocation_leg(ctx, overlay, stub_log, marker))
    elif probe == PROBE_DISPATCH:
        record.update(dispatch_leg(ctx, root) if root else {"reason": "no installed root"})
    elif probe == PROBE_UPGRADE:
        record.update(upgrade_leg(ctx, overlay, install))
    (ctx.out / f"probe-{probe}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    ctx.log(f"claude package probe {probe}: recorded")
    return record
