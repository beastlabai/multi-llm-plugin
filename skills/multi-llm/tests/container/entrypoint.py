#!/usr/bin/env python3
"""In-container driver: one row, start to artifacts (C1a).

This runs **inside** the container and evaluates no assertion. It imports no
pytest. Its whole job is to produce evidence and put it where the host-side driver
can read it after the container exits:

    stub lifecycle  ->  seed homes  ->  seed workspace  ->  stage/install
                    ->  run the row  ->  ALWAYS export artifacts

**Nothing here decides whether a row passed.** `harness.py`, on the host, reads
`/work/out` and grades it. That split is what lets the marker-plus-skipif gating be
evaluatable on a machine with no container runtime at all, keeps the golden
comparison writing into a repo tree the container deliberately cannot see, and
keeps the test suite out of `context_manifest.txt` so what ships inside the image
is the product and nothing else.

The export is in a ``finally`` on purpose. The orchestrator computes its output
directory from the plan path (``<plan-parent>/<plan-stem>/``), which sits inside
the tmpfs ``/work/sample-project`` and dies with the container. A row whose export
step did not run is a **failed row**, not a row with an empty golden diff.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_DIR = Path(os.environ.get("MLLM_SKILL_DIR", "/opt/mllm/skill"))
RIG_DIR = Path(os.environ.get("MLLM_RIG_DIR", "/opt/mllm/rig"))
#: The genuine binaries, behind the PATH wrappers. Used only where a probe must
#: see what the HOST sees rather than what a dispatched provider sees.
REAL_DIR = Path(os.environ.get("MLLM_REAL_DIR", "/opt/mllm/real"))
#: The opencode PLUGIN package, unpacked into the image by the Dockerfile's §10b.
#: Declaring THIS path in `opencode.json`'s `plugin[]` is the whole package route
#: in the container, and it is a path rather than the registry spec for one
#: reason: `opencode-multi-llm` is not published, so `opencode-multi-llm@0.1.0`
#: has nothing to resolve against and `--network none` could not fetch it if it
#: had. MEASURED 2026-09-19 (opencode 1.18.31) that a directory spec loads and a
#: `.tgz` spec does not — see §10b for the four-form table.
OPENCODE_PLUGIN_DIR = Path(
    os.environ.get("MLLM_OPENCODE_PLUGIN_DIR", "/opt/mllm/opencode-plugin"))
OUT = Path("/work/out")
HOMES = Path("/work/homes")
WORKSPACE = Path("/work/sample-project")
#: Where the fallback mode's `:ro` credential mounts land (C6a rules 2 + 3). Absent
#: entirely on the default OpenRouter path and on the provider lane.
FALLBACK_CREDS_ROOT = Path("/run/creds")

# Every credential-shaped variable that must be absent on a zero-secret row. The
# provider lane and the `--creds none` host tier both refuse to start if any is
# present, so a run that was supposed to cost nothing cannot silently become a
# cheap credentialed one.
CREDENTIAL_VARS = (
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY",
    "DEEPSEEK_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
)

# Never mounted, asserted at start (C6a rule 4). The first three matter most: this
# repo's remote is public GitHub, and a container holding a gh token plus a network
# route is one stray command away from pushing.
DENYLIST = (
    "~/.gitconfig", "~/.git-credentials", "~/.config/gh/hosts.yml",
    "~/.ssh", "~/.netrc", "~/.aws",
)


def log(message: str) -> None:
    print(f"[entrypoint] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def assert_no_credentials() -> None:
    present = [v for v in CREDENTIAL_VARS if os.environ.get(v)]
    if present:
        raise SystemExit(
            "entrypoint: refusing to run a zero-secret row with a credential "
            f"present: {', '.join(present)}"
        )


def assert_denylist_absent() -> None:
    found = [p for p in DENYLIST if Path(p).expanduser().exists()]
    if found:
        raise SystemExit(
            "entrypoint: deny-listed path mounted into the container: "
            + ", ".join(found)
        )


def _rig_policy(name: str) -> dict:
    """One of the rig's checked-in policy files, read from inside the image.

    A missing file is a **refusal**, never an empty policy: "nothing to compare
    against" is a different verdict from "compared and matched", and an allowlist
    that silently degrades to ``{}`` would let every mount through.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - the image ships pyyaml
        raise SystemExit(
            f"entrypoint: PyYAML is missing from the rig interpreter, so {name} "
            "cannot be read and the mount/posture policy cannot be enforced"
        ) from exc
    path = RIG_DIR / name
    if not path.is_file():
        raise SystemExit(
            f"entrypoint: {path} is missing — an unverifiable posture is a "
            "refusal, not a pass")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


#: Mount points every container carries whatever the run command said: the kernel
#: and docker pseudo-filesystems, plus the three files docker always injects. Only
#: what remains after these is compared against the declared set.
MOUNT_IGNORED_TREES = ("/proc", "/sys", "/dev")
MOUNT_IGNORED = frozenset({"/", "/etc/resolv.conf", "/etc/hostname", "/etc/hosts"})


def _unescape_mountinfo(field: str) -> str:
    """mountinfo octal-escapes space, tab, newline and backslash."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def read_mounts() -> list[dict]:
    """`/proc/self/mountinfo` as ``{target, options, fstype}`` records.

    The mount *source* is deliberately dropped rather than recorded: on a bind it
    is a path on the developer's machine, and this record is exported into the
    row's artifact bundle.
    """
    mounts: list[dict] = []
    text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    for line in text.splitlines():
        fields = line.split(" ")
        # ``… <mount point> <options> <optional fields> - <fstype> <source> <super>``
        if len(fields) < 10 or "-" not in fields:
            continue
        separator = fields.index("-")
        mounts.append({
            "target": _unescape_mountinfo(fields[4]),
            "options": fields[5].split(","),
            "fstype": fields[separator + 1],
        })
    return mounts


def declared_mount_targets(lane: str, creds: str,
                           host: str | None = None) -> set[str]:
    """Every mount point this run is allowed to have, from the checked-in policy.

    `posture.yaml` names the run-time list and `creds.yaml` names the per-host
    credential files — and the credential ones count **only** in fallback mode,
    and then only for the host this row runs: the default OpenRouter path mounts
    nothing at all and the provider lane mounts nothing ever (C1, C6a), so a
    `/run/creds` bind on either is exactly the "enabled later without re-reading
    this section" failure the rule exists for, and a *second* host's credential on
    a row that cannot use it is the home-directory-shaped mount the rule replaced.
    """
    targets: set[str] = set()
    fallback = lane == "host" and creds == "fallback"
    for mount in _rig_policy("posture.yaml").get("mounts") or []:
        target = str(mount.get("target") or "")
        if not target:
            continue
        if target == str(FALLBACK_CREDS_ROOT) and not fallback:
            continue
        targets.add(target)
    if fallback:
        allowlist = _rig_policy("creds.yaml").get("allowlist") or {}
        for entry in (allowlist.get(host) or []) if host else [
                e for entries in allowlist.values() for e in entries or []]:
            dest = str(entry.get("dest") or "")
            if dest:
                targets.add(dest)
    return targets


def assert_mount_allowlist(lane: str, creds: str,
                           host: str | None = None) -> list[dict]:
    """C6a rule 2: refuse to start if a mount exists that the policy does not name.

    Stat-ing a handful of `~/` paths (`assert_denylist_absent`) answers a narrower
    question than the rule asks: a bind of `~/.ssh` at a path nobody expected, or
    of the repo working tree "just this once" for a debugging session, passes every
    one of those stats. This compares `/proc/self/mountinfo` against
    `posture.yaml` + `creds.yaml` instead, so an undeclared bind fails **here**,
    loudly, before a third-party CLI is anywhere near it.
    """
    declared = declared_mount_targets(lane, creds, host)
    mounts = read_mounts()
    extra = []
    for mount in mounts:
        target = mount["target"]
        if target in MOUNT_IGNORED or target in declared:
            continue
        if any(target == tree or target.startswith(tree + "/")
               for tree in MOUNT_IGNORED_TREES):
            continue
        extra.append(f"{target} ({mount['fstype']})")
    if extra:
        raise SystemExit(
            "entrypoint: mount(s) this run never declared: " + ", ".join(extra)
            + " — posture.yaml's `mounts:` plus creds.yaml's `allowlist:` are the "
              "COMPLETE set for this lane. A bind of the repo working tree, or of "
              "a credential at a path nobody named, fails here rather than "
              "surfacing three phases later as an artifact nobody expected.")
    log(f"mounts ok: {len(mounts)} mount point(s), none outside the allowlist")
    return mounts


def assert_posture(lane: str, creds: str = "openrouter") -> dict:
    """Check the posture from INSIDE, where most of it is only visible.

    Returns the record it checked so the row's evidence carries what was
    *verified*, not only what failed. `posture.yaml` declares that both ends are
    inspected; the host end is `harness.host_posture_mismatches`, which reads the
    same file against `docker inspect`'s `HostConfig`.
    """
    posture = _rig_policy("posture.yaml")
    identity = posture.get("identity") or {}
    uid, gid = os.getuid(), os.getgid()
    if uid == 0 or gid == 0:
        raise SystemExit(f"entrypoint: running as root ({uid}:{gid})")
    # Baked (image `USER`), re-stated (`--user`) and repeated on every tmpfs
    # (`uid=`/`gid=`) are three separate failures, so the declared pair is
    # compared rather than assumed to follow from "not root".
    if (uid, gid) != (int(identity.get("uid", uid)), int(identity.get("gid", gid))):
        raise SystemExit(
            f"entrypoint: running as {uid}:{gid}, posture.yaml declares "
            f"{identity.get('uid')}:{identity.get('gid')} — the image's USER, the "
            f"run's --user and the tmpfs uid=/gid= must all carry the same pair")
    try:
        import pwd

        entry = pwd.getpwuid(uid)
    except KeyError as exc:  # noqa: PERF203
        raise SystemExit(
            f"entrypoint: uid {uid} has no /etc/passwd entry — several agent CLIs "
            "mishandle a process whose ~ cannot be resolved through the user "
            "database"
        ) from exc
    home = Path(entry.pw_dir)
    if not home.is_dir() or home.stat().st_uid != uid:
        raise SystemExit(f"entrypoint: $HOME {home} is missing or not owned by {uid}")

    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields = dict(
        line.split(":", 1) for line in status.splitlines() if ":" in line
    )
    if fields.get("NoNewPrivs", "").strip() != "1":
        raise SystemExit("entrypoint: NoNewPrivs is not 1 (--security-opt missing?)")
    if fields.get("Seccomp", "").strip() != "2":
        raise SystemExit(
            "entrypoint: Seccomp is not 2 — the default profile was replaced, "
            "probably by `seccomp=unconfined` added to unblock one CLI"
        )
    # An empty effective set is all zeros ("0000000000000000"); anything left
    # after stripping them is a capability --cap-drop=ALL did not drop.
    cap_eff = fields.get("CapEff", "").strip()
    if cap_eff.strip("0") != "":
        raise SystemExit(f"entrypoint: capability set is not empty: CapEff={cap_eff}")

    mounts = {mount["target"]: mount for mount in read_mounts()}
    record = {
        "lane": lane,
        "uid": uid, "gid": gid, "home": str(home),
        "no_new_privs": fields.get("NoNewPrivs", "").strip(),
        "seccomp": fields.get("Seccomp", "").strip(),
        "cap_eff": cap_eff,
        "readonly_rootfs": _assert_readonly_root(posture, mounts),
        "mounts": _assert_declared_mounts(posture, mounts),
        "limits": _assert_limits(posture),
        "network": _assert_network(posture, lane, creds),
    }
    log(f"posture ok: uid={uid} gid={gid} home={home} "
        f"readonly-root={record['readonly_rootfs']} "
        f"network={record['network']['declared']} limits={record['limits']}")
    return record


def _assert_readonly_root(posture: dict, mounts: dict) -> bool:
    """`--read-only`, read off the root mount's own options.

    Writes are supposed to be confined to the named mounts below; a root that
    quietly became writable is how a CLI's scribble ends up in an image layer's
    upperdir instead of failing the row that produced it.
    """
    if not (posture.get("security") or {}).get("readonly_rootfs"):
        return False
    root = mounts.get("/")
    if root is None or "ro" not in root["options"]:
        raise SystemExit(
            "entrypoint: posture.yaml declares a read-only root filesystem and "
            "this container's `/` is writable — `--read-only` was dropped from "
            "the run command")
    return True


def _assert_declared_mounts(posture: dict, mounts: dict) -> dict:
    """Every non-optional declared mount is present, of the declared type, writable.

    The other direction from `assert_mount_allowlist`: a **dropped** `--tmpfs` does
    not widen the container, it moves those writes onto the read-only root, where
    the row dies with an EACCES that names a path rather than the flag.

    Ownership and mode come from `stat()`, not from the option string: `/tmp`'s
    `mode=1777` is the tmpfs default and docker does not restate it in mountinfo,
    while "can this uid write here" — the thing the declaration is actually for —
    is exactly what `stat` answers.
    """
    checked: dict = {}
    for declared in posture.get("mounts") or []:
        target = str(declared.get("target") or "")
        if not target or declared.get("optional"):
            continue
        mount = mounts.get(target)
        if mount is None:
            raise SystemExit(
                f"entrypoint: posture.yaml declares a {declared.get('type')} at "
                f"{target} and nothing is mounted there — the writes it exists for "
                f"would land on the read-only root")
        expected_type = str(declared.get("type") or "")
        if expected_type == "tmpfs" and mount["fstype"] != "tmpfs":
            raise SystemExit(
                f"entrypoint: {target} is declared tmpfs and is a "
                f"{mount['fstype']} mount — its contents would outlive the row")
        if "rw" not in mount["options"]:
            raise SystemExit(f"entrypoint: {target} is mounted read-only")
        options = dict(
            pair.split("=", 1)
            for pair in str(declared.get("mode") or "").split(",") if "=" in pair)
        info = Path(target).stat()
        for key, actual in (("uid", info.st_uid), ("gid", info.st_gid)):
            if key in options and int(options[key]) != actual:
                raise SystemExit(
                    f"entrypoint: {target} is owned by {key}={actual}, declared "
                    f"{key}={options[key]} — a tmpfs mounted for the wrong owner "
                    f"fails on the first write into it, far from this flag")
        if "mode" in options and (info.st_mode & 0o7777) != int(options["mode"], 8):
            raise SystemExit(
                f"entrypoint: {target} is mode {info.st_mode & 0o7777:04o}, "
                f"declared {int(options['mode'], 8):04o}")
        checked[target] = {"fstype": mount["fstype"],
                           "mode": f"{info.st_mode & 0o7777:04o}",
                           "uid": info.st_uid, "gid": info.st_gid}
    return checked


def _cgroup_value(*candidates: str) -> str | None:
    """The first readable cgroup file among v2-then-v1 spellings."""
    for candidate in candidates:
        try:
            return (Path("/sys/fs/cgroup") / candidate).read_text(
                encoding="utf-8").strip()
        except OSError:
            continue
    return None


def _parse_bytes(value) -> int:
    """`"4g"` -> bytes, the way docker's own `--memory` reads it."""
    text = str(value).strip().lower()
    units = {"b": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(float(text))


def _assert_limits(posture: dict) -> dict:
    """`--pids-limit`, `--memory` and `--cpus`, read back off this cgroup.

    A runaway CLI is a failed row, not a wedged workstation — and these are the
    flags nobody notices are gone until the workstation is the thing that stopped
    responding. A mismatch is a refusal; a cgroup interface that is not mounted at
    all is recorded `unverified` and left to the host-side `docker inspect` half,
    because "cannot read the limit" is not evidence the limit is wrong.
    """
    declared = posture.get("limits") or {}
    checked: dict = {}

    memory = _cgroup_value("memory.max", "memory/memory.limit_in_bytes")
    want_memory = _parse_bytes(declared.get("memory", 0))
    if memory is None:
        checked["memory"] = "unverified: no cgroup memory interface"
    elif memory == "max" or int(memory) != want_memory:
        raise SystemExit(
            f"entrypoint: cgroup memory limit is {memory}, posture.yaml declares "
            f"{declared.get('memory')} ({want_memory}) — `--memory` was dropped")
    else:
        checked["memory"] = int(memory)

    pids = _cgroup_value("pids.max", "pids/pids.max")
    want_pids = int(declared.get("pids_limit", 0))
    if pids is None:
        checked["pids"] = "unverified: no cgroup pids interface"
    elif pids == "max" or int(pids) != want_pids:
        raise SystemExit(
            f"entrypoint: cgroup pids limit is {pids}, posture.yaml declares "
            f"{want_pids} — `--pids-limit` was dropped")
    else:
        checked["pids"] = int(pids)

    want_cpus = float(declared.get("cpus", 0))
    quota_period = _cgroup_value("cpu.max")
    if quota_period is None:                                   # cgroup v1
        quota = _cgroup_value("cpu/cpu.cfs_quota_us")
        period = _cgroup_value("cpu/cpu.cfs_period_us")
    else:
        parts = quota_period.split()
        quota, period = parts[0], (parts[1] if len(parts) > 1 else "100000")
    if quota is None or period is None:
        checked["cpus"] = "unverified: no cgroup cpu interface"
    elif quota in ("max", "-1") or abs(int(quota) / int(period) - want_cpus) > 0.01:
        raise SystemExit(
            f"entrypoint: cgroup cpu quota is {quota}/{period}, posture.yaml "
            f"declares {want_cpus} cpus — `--cpus` was dropped")
    else:
        checked["cpus"] = round(int(quota) / int(period), 3)
    return checked


def _assert_network(posture: dict, lane: str, creds: str = "openrouter") -> dict:
    """The EFFECTIVE network mode, from the routing table rather than from a flag.

    The provider lane's guarantee is structural — a mis-redirected CLI cannot reach
    a real endpoint because there is no route to one — so "there is no default
    route" is the fact worth asserting, not "`--network none` appeared in an argv
    somewhere. The CREDENTIALED host modes assert the opposite direction for a
    duller reason: a row that cannot reach openrouter.ai fails deep inside a CLI's
    retry loop, and naming it here costs one read of `/proc/net/route`.

    `--creds none` is a host row on the provider lane's terms: no credential and no
    route, so it asserts routelessness like the provider lane. Which key is read
    must match `harness.network_mode`, or the tier fails on its own posture check.
    """
    declared = str((posture.get("network") or {}).get(
        "provider_lane" if lane == "provider" or creds == "none"
        else "host_lane") or "")
    routes = [line.split() for line in
              Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]]
    routed_via = sorted({fields[0] for fields in routes if fields})
    # Destination 00000000 is 0.0.0.0 — the default route, and the only entry that
    # says anything about reaching a real endpoint.
    default_route = any(len(fields) > 1 and fields[1] == "00000000"
                        for fields in routes)
    if declared == "none" and default_route:
        raise SystemExit(
            "entrypoint: this lane declares `--network none` and the container "
            f"has a DEFAULT ROUTE via {routed_via} — the lane's guarantee is that "
            "a fallback past the stub is impossible, not merely detected")
    if declared != "none" and not default_route:
        raise SystemExit(
            f"entrypoint: this lane declares network {declared!r} and the "
            "container has no default route — every request the row makes would "
            "fail deep inside a CLI's own retry loop")
    return {"declared": declared, "default_route": default_route,
            "routed_via": routed_via}


# --------------------------------------------------------------------------- #
# Stub lifecycle (C4)
# --------------------------------------------------------------------------- #

class Stub:
    """Start, wait for readiness, and reap the stub server.

    The readiness wait is not optional: without it a CLI that wins the race sees a
    connection refused and reports it as an auth or network error, which is the
    least legible failure this rig can produce. The teardown is not optional
    either — a stub still listening after a row would let the next row's requests
    land in the previous row's session table and silently corrupt both.
    """

    def __init__(self, port: int, log_path: Path, scenario: Path | None,
                 facts: Path | None = None):
        self.port = port
        self.log_path = log_path
        self.scenario = scenario
        #: The documented-invocation probe's `{fact: regex}` file, or None. The
        #: stub then records which of those patterns each prompt matched — names
        #: only, never the prompt — which is the one question a sha256 cannot
        #: answer. See `invocation_facts`.
        self.facts = facts
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Stub":
        argv = [sys.executable, str(RIG_DIR / "stub_server.py"),
                "--port", str(self.port), "--log", str(self.log_path)]
        if self.scenario:
            argv += ["--scenario", str(self.scenario)]
        if self.facts:
            argv += ["--facts", str(self.facts)]
        self.proc = subprocess.Popen(argv)
        self._wait_healthy()
        log(f"stub ready on 127.0.0.1:{self.port}")
        return self

    def _wait_healthy(self, timeout: float = 20.0) -> None:
        url = f"http://127.0.0.1:{self.port}/__health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise SystemExit(
                    f"entrypoint: stub exited {self.proc.returncode} before serving"
                )
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        raise SystemExit("entrypoint: stub never answered /__health")

    def __exit__(self, *_exc) -> None:
        if not self.proc:
            return
        self.proc.send_signal(signal.SIGINT)   # drain in-flight, flush the log
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


# --------------------------------------------------------------------------- #
# Row setup
# --------------------------------------------------------------------------- #

#: The substitution tokens a ``home_seed/<id>/`` template carries, and the reason
#: a `--creds none` host row cannot copy one wholesale. Kept byte-identical to the
#: keys of `wrapper.seed_invocation_root`'s `replacements` map — the code that
#: binds them on the provider lane — by `tests/test_host_home_seed.py`, because a
#: token added there and not here would be copied into the host's `$HOME` as a
#: literal `{{…}}` string rather than skipped.
SEED_TOKENS = ("{{base_url}}", "{{base_url_root}}", "{{stub_session_key}}",
               "{{api_key_env}}", "{{model_name}}")


def seed_host_home(host: str, home: Path) -> list[str]:
    """Put the HOST's own ``$HOME`` in the posture a real machine is in, on the
    one host-lane tier that had no wiring step at all (`--creds none`).

    **`seed_homes()` is not the function that does this and never was.** It writes
    ``/work/homes/<identity>``, which nothing reads as a ``$HOME`` except
    `wrapper.seed_invocation_root`, and only for the provider subprocesses a
    fan-out dispatches. The host process's ``$HOME`` is `Path.home()` —
    ``/home/e2e``, which `posture.yaml` mounts **tmpfs**, so whatever the image
    baked there is masked and the directory is EMPTY at row start. Both
    credentialed tiers fill it (`wire_credentials`, `wire_fallback_credentials`)
    and so does the documented-invocation probe (`wire_stub_host`); the plain
    discovery tier was the fourth path and it filled nothing, which is why it ran
    the host against a barer ``$HOME`` than any user will ever have.

    What that costs is not hypothetical. `Dockerfile` step 9 bakes opencode's
    cold-start tree — 3661 files, 59 MB of ``node_modules`` plus the models.dev
    catalog — into ``home_seed/opencode/`` precisely because a first `opencode
    run` otherwise npm-installs it, which `--network none` makes impossible rather
    than slow. Under an unseeded ``$HOME`` the free tier never had one byte of it,
    so nothing that needs that tree could ever have been measured here.

    **A templated file is SKIPPED, never copied.** The seeds are value-free by
    design: every credential and endpoint field is a ``{{token}}`` bound per exec
    by the wrapper, or per run by `live_provider.py`. This tier has neither — no
    exec id, no stub base URL bound to the host, no credential — so a copied
    ``opencode.json`` would hand the host a provider whose ``baseURL`` is the
    literal string ``{{base_url}}``, which is the same shadowing `wire_stub_host`
    already refuses by name.

    **And it matches the tokens, never a bare ``{{``.** Measured in this image:
    six files inside opencode's own ``node_modules`` carry ``{{``
    (``effect/dist/Graph.js``, ``effect/src/Graph.ts``, two ``httpApiScalar``
    bundles, ``zod/src/v4/core/errors.ts``), none of them a seed template. A
    brace-shaped heuristic would drop them out of the very tree this function
    exists to deliver, silently and only in the offline posture.

    Returns the skipped paths, seed-relative and sorted, so the row's log names
    what it declined to place rather than leaving a gap to be inferred.
    """
    seed = RIG_DIR / "home_seed" / host
    if not seed.is_dir():
        log(f"host home {home}: nothing to seed — no {seed}")
        return []

    skipped: list[str] = []

    def _skip_templated(directory, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if any(token.encode() in data for token in SEED_TOKENS):
                ignored.add(name)
                skipped.append(str(path.relative_to(seed)))
        return ignored

    home.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed, home, dirs_exist_ok=True, ignore=_skip_templated)
    skipped.sort()
    log(f"host home {home} seeded from {seed}"
        + (f"; skipped {len(skipped)} templated file(s): {', '.join(skipped)}"
           if skipped else ""))
    return skipped


def seed_homes(identities: list[str]) -> None:
    """One $HOME per CLI identity the row will run (C1a).

    A single shared $HOME would put two CLIs in the same config root, which a
    provider row hits every single time: the row's own `cli` and the
    agent-write-capable helper its executor needs are two different binaries in
    one container.
    """
    for identity in identities:
        home = HOMES / identity
        home.mkdir(parents=True, exist_ok=True)
        seed = RIG_DIR / "home_seed" / identity
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        log(f"home {home} seeded from {seed if seed.is_dir() else '<nothing>'}")


def drop_seeded_credential(identities: list[str], relative: str | None) -> None:
    """A negative auth row's ``remove_seed``, actually removed.

    Until this existed `remove_seed` was a **dead cell**: `host_matrix.yaml`
    declared it on both negative rows, and its only readers anywhere were in
    `test_container_host_lane.py`, which used its mere PRESENCE to classify a row
    as negative and never acted on its value. Nothing deleted the file, so both
    rows would have run with the seeded credential in place and could not have
    been refused for the reason they declare.

    **`/work/homes/<identity>`, never the host's own `$HOME`**, and the
    distinction is the row rather than a detail:

    * the gate a negative row trips is `preflight_runner`'s `auth_check` of the
      CONFIGURED RUNNER — a *provider* dispatch — and `wrapper.py`'s
      `seed_invocation_root` copies every per-invocation provider root from
      `/work/homes/<cli>`. Removing it there removes it from every dispatch of
      that CLI, including two concurrent ones.
    * the HOST must still be able to run, or the row proves nothing about the
      gate: a host that cannot authenticate never reaches our dispatcher and
      leaves no receipt to grade the refusal on. On opencode the two files even
      share a spelling — `wire_credentials` writes the same relative path into
      the host's `$HOME` — and that copy is the host's credential, not the
      runner's.

    **A removal that matched nothing raises.** That silence is precisely the
    failure this cell already had for its whole life, and a no-op deletion would
    reproduce it one layer down: the row would run green having been graded as a
    refusal it never made.
    """
    if not relative:
        return
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SystemExit(
            f"entrypoint: remove_seed {relative!r} does not name a path inside a "
            f"seeded home. It must be relative to `/work/homes/<identity>` and "
            f"may not climb out of it.")
    removed: list[Path] = []
    for identity in identities:
        target = HOMES / identity / candidate
        if target.is_file() or target.is_symlink():
            target.unlink()
            removed.append(target)
            log(f"remove_seed: deleted {target}")
    if not removed:
        searched = [str(HOMES / identity) for identity in identities]
        raise SystemExit(
            f"entrypoint: remove_seed names {relative!r} and no seeded home "
            f"carries it — searched {searched or '<no seeded identity>'}. A "
            f"removal that deletes nothing is the silent failure this cell "
            f"exists to prevent: the row would dispatch with the credential "
            f"still in place and be graded as a refusal it never made.")


def run_pre_commands(row_id: str, commands: list[str]) -> int:
    """The host lane's PRE-COMMAND LEG: ordered dispatcher invocations that run
    before the host process starts.

    `harness.Row.commands` and the `--command` forwarding have existed since the
    provider lane's chain rows, but the host branch of :func:`main` returns
    before the dispatch loop that consumes them — so on this lane the cell had
    never executed. That is the rig gap behind `host_matrix.yaml`'s header
    finding (1): the lane seeds one workspace and launches exactly ONE host
    process against it, and `apply-suggestions` exits 1 in `_setup` against a
    workspace with no review-plan output, before any batch, runner or auth gate.

    **The receipts land in a SIBLING directory**, `receipts/<row>-pre/`, and that
    is the whole reason this is a separate function rather than three lines
    inline. Every host-lane assertion about "the agent really ran our dispatcher"
    reads `receipts/<row>/`; a leg that wrote there would let a row pass
    `test_a_receipt_exists_for_the_row` — and even
    `test_the_payload_names_the_expected_mode_and_phase` — on **the rig's own
    invocation**, which is the exact false positive the receipt anchor exists to
    prevent. The evidence is still kept, one directory across, where a human can
    read it and no grader can mistake it for the host's work.

    Runs in the row's workspace with PATH wrapper-first, exactly like the
    entrypoint's own `install-host` dispatch, so the leg's own fan-out lands on
    the stub. Stops at the first non-zero exit and returns it: a host launched
    against a half-prepared workspace would fail for the wrong reason.
    """
    if not commands:
        return 0
    receipts = OUT / "receipts" / f"{row_id}-pre"
    results: list[dict] = []
    failed = 0
    for raw in commands:
        # A leading `NAME=value` applies to THIS invocation only, the same rule
        # the provider lane's chain rows are written under.
        command_env, argv_one = split_command_env(shlex.split(raw))
        code = dispatch(argv_one, env={**os.environ,
                                       "MLLM_RECEIPT_DIR": str(receipts),
                                       **command_env})
        results.append({"command": raw, "exit": code,
                        "env": sorted(command_env)})
        log(f"pre-command {argv_one!r} -> exit {code}"
            + (f" (env: {sorted(command_env)})" if command_env else ""))
        if code != 0:
            failed = code
            break
    (OUT / "pre-commands.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return failed


def record_stub_watermark(row_id: str, removed: str | None) -> None:
    """How many model calls the RIG had already made when the host started.

    A negative auth row declares ``stub_requests: 0`` and means *zero after the
    credential was removed* — "the refusal lands BEFORE the first model call".
    That cell was written for a lane with no pre-command leg. The leg fans out to
    the stub **by design** (it runs while the seed is still in place, through the
    very runner the row exists to have refused), so from the day it landed the
    container's stub log carries the rig's own calls and a grader counting raw
    would measure the leg instead of the row.

    So the boundary is written down at the one moment it is unambiguous: after
    :func:`run_pre_commands` has returned and after
    :func:`drop_seeded_credential`, immediately before the host process starts.
    Every record already in ``stub.jsonl`` is the rig's; everything after it is
    the host's. Health checks are excluded here for the same reason the grader
    excludes them: they are the stub proving it is up, not a model call.

    Written on EVERY credentialed row, not only the negative ones — a watermark
    that exists only where an assertion currently reads it is a watermark the
    next assertion cannot trust.
    """
    def _records(path: Path) -> list[dict]:
        # A torn last line is evidence, never a failure path: the watermark is a
        # floor and one unparsed record cannot move it in the direction that
        # hides a call. Same rule as the reader on the other side
        # (`harness._read_jsonl`), so an index taken here indexes the same list
        # the grader slices.
        rows: list[dict] = []
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    calls = sum(1 for record in _records(OUT / "stub.jsonl")
                if record.get("surface") != "health")
    # THE SAME LINE, DRAWN ON THE CALL LOG. The leg fans out through the very
    # same `/opt/mllm/bin` shim, so `calls.jsonl` carries its invocations too and
    # a per-provider grader counting the whole file credits the rig's own
    # fan-out to the host. One index, written where the boundary is unambiguous,
    # is what lets that grader slice both logs at the same moment.
    invocations = len(_records(OUT / "calls.jsonl"))
    (OUT / "stub-watermark.json").write_text(json.dumps({
        "row": row_id,
        "removed_seed": removed,
        "non_health_calls_before_host": calls,
        "call_records_before_host": invocations,
        "meaning": ("every stub request logged at or below "
                    "`non_health_calls_before_host`, and every `calls.jsonl` "
                    "record at or below `call_records_before_host`, was made "
                    "by the rig's pre-command leg, before the host process "
                    "started and before any declared credential was removed"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"stub watermark before host launch: {calls} non-health request(s), "
        f"{invocations} wrapped CLI invocation(s)")


def substitute_placeholders(root: Path, mapping: dict[str, str]) -> None:
    """Fill {{placeholder}}s in the value-free seed templates."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "{{" not in text:
            continue
        for key, value in mapping.items():
            text = text.replace("{{" + key + "}}", value)
        path.write_text(text, encoding="utf-8")


def stage_host(host: str, home: Path, route: str = "file") -> int:
    """Make the skill discoverable by ``host`` (C6c), and verify it.

    This is NOT the same operation on all three hosts, which is the whole reason
    it is a table rather than one command. `install-host <host> --global` renders
    the shim for the two installable hosts. `claude-code` has no such path:
    `installable: false` makes `install` a refusal in those words, and its
    `--check` reads the plugin's OWN SKILL.md — proving the shipped shim's shape
    and this machine's prerequisites, and nothing at all about whether Claude Code
    can *find* the skill. Left implicit, every claude-code row would fail the
    discovery assertion behind a passing install step.

    **And it is not the same operation on both ROUTES either**, which is why the
    row's `route` reaches this far. A package row's staging is not a command at
    all: it is the `plugin[]` declaration the caller has already written into the
    host's own config, resolved against the payload the image pre-extracted. So
    on that route this function installs nothing — see the guard below.
    """
    if route == "package":
        # NOTHING TO STAGE, and staging anyway is the bug rather than a
        # belt-and-braces extra. `install-host <host> --global` writes the FILE
        # shim, so a package row carrying one has the two command sources 4b.11
        # forbids: on opencode the plugin's `config` hook then overwrites the
        # file entry wholesale and silently, and the row would grade a command it
        # did not install while crediting the file route's evidence to the
        # package route's — the precise false promotion `Row.route` exists to
        # stop.
        log(f"route=package: no shim staged for {host} — the host's own config "
            f"declares the plugin and the host's installer resolves the payload")
        return 0

    env = dict(os.environ, HOME=str(home))
    if host == "claude-code":
        # Stage the WHOLE skill tree, never SKILL.md alone: every command in that
        # shim is `uv run --project "${CLAUDE_SKILL_DIR}" -- python
        # "${CLAUDE_SKILL_DIR}/multi_llm.py" …`, so a lone file leaves the host
        # expanding the variable at a directory with no dispatcher in it.
        target = home / ".claude" / "skills" / "multi-llm"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # A directory symlink: the mechanism this repo already runs the skill
            # through from outside the plugin tree, and the one cli_common's
            # skill-dir resolution is written for (abspath, symlinks preserved,
            # never realpath, so `uv run --project` stays on the pyproject.toml
            # the host loaded). It also keeps the check honest: under a link the
            # file --check reads and the file the host loads are one inode.
            target.symlink_to(SKILL_DIR, target_is_directory=True)
        log(f"staged claude-code skill tree at {target} -> {SKILL_DIR}")
        argv = ["install-host", "claude-code", "--check",
                "--project-root", str(WORKSPACE)]
    else:
        argv = ["install-host", host, "--global"]
    return dispatch(argv, env=env, cwd=WORKSPACE)


def discovery_step(host: str, home: Path) -> dict:
    """Is the staged command actually DISCOVERABLE by the host? (C6, `--creds none`)

    This is the assertion the free tier exists for: the shim landing where the
    host no longer scans is the failure that reaches users first, and it is the
    one host-lane check available at zero cost and zero secrets.

    **A green `install-host --check` is not discovery** and must never be read as
    it: that reads our own bytes at a path we chose. Discovery means the *host*
    enumerating the command by name. Phase 1 measured which hosts can be asked
    that without a model:

    * ``opencode`` — yes. ``opencode debug config`` emits resolved configuration
      as JSON, and an installed command appears under its ``command`` key.
      (``opencode debug skill`` is the trap: it lists *skills*, our shim installs
      as a *command*, and it returns exit 0 listing only a built-in.)
    * ``codex``, ``claude-code`` — no such surface exists. Both were probed,
      including each one's ``doctor``. Discovery there needs a prompted turn,
      which needs a credential, so the free tier reports them
      ``unavailable-without-credential`` rather than passing them.
    """
    env = dict(os.environ, HOME=str(home))
    matrix = _host_matrix().get(host) or {}
    step = matrix.get("list_step") or {}
    form = step.get("form")

    if form != "debug-config":
        return {"host": host, "discovery": "unavailable-without-credential",
                "reason": step.get("note", "no credential-free listing surface"),
                "listed": None}

    # THE REAL BINARY, BY ABSOLUTE PATH — never through /opt/mllm/bin. This is the
    # same exemption C2 grants the executor's credential probes, for the same
    # reason and with a sharper consequence. The wrapper rebuilds every child's
    # environment from scratch and points $HOME at a per-invocation provider root
    # seeded from /work/homes/<cli>. That is exactly right for a dispatched
    # provider and exactly wrong here: this probe is asking what the HOST can see,
    # and the host's shim lives in the host's own $HOME. Routed through the
    # wrapper it reported zero commands — a false "the shim landed somewhere this
    # host no longer scans", which is the one verdict this tier must never
    # fabricate. The host itself is launched by absolute path for the same reason.
    completed = subprocess.run(
        [str(REAL_DIR / "opencode"), "debug", "config"],
        env=env, cwd=str(WORKSPACE),
        capture_output=True, text=True, timeout=120)
    try:
        config = json.loads(completed.stdout)
    except ValueError:
        return {"host": host, "discovery": "failed", "listed": None,
                "reason": f"`opencode debug config` emitted no JSON "
                          f"(exit {completed.returncode})"}
    commands = list((config.get("command") or {}))
    listed = "multi-llm" in commands
    return {
        "host": host,
        "discovery": "listed" if listed else "not-listed",
        "listed": listed,
        "commands": commands,
        "reason": None if listed else (
            "the host enumerated its commands and ours was NOT among them — the "
            "shim landed somewhere this host no longer scans"),
    }


# --------------------------------------------------------------------------- #
# The DOCUMENTED-INVOCATION probe (--documented-invocation, `--creds none`)
#
# Follow-up 7 of plans/readme-trim-and-multi-host-install-docs.md, and the one
# acceptance row that lane had no way to answer: *does the string the README
# tells a reader to type actually reach our dispatcher on this host?*
#
# WHAT WAS ALREADY COVERED, AND WHY IT IS NOT THIS. The credentialed rows prompt
# each host with `Use the multi-llm skill with exactly these arguments: …`
# (`prompt_template`), which proves the WRITER — the shim landed somewhere the
# host reads, and an argument string reaches the dispatcher. It says nothing
# about the INVOCATION, because the two installable hosts do not install the
# same kind of object: codex gets a **skill** (`.codex/skills/multi-llm/SKILL.md`)
# and opencode gets a **command** (`.opencode/commands/multi-llm.md`), and each
# host does its own resolution, namespacing and word-splitting on its own format.
# `/multi-llm …` is therefore a claim about the host's command parser, and only
# typing it into a session can settle it.
#
# THREE PROPERTIES OF THIS ROW, each one deliberate:
#
# * **It is free, and structurally so.** It runs at `--creds none`, i.e. under
#   `--network none` with every credential variable asserted absent. The host's
#   own model is the container stub on loopback, which is enough because what is
#   under test is ARGUMENT DELIVERY, not model output: the question is answered
#   by the bytes the host sends, not by what comes back.
# * **The evidence is the prompt the host composed**, read through the stub's
#   opt-in fact probe (`stub_server.install_facts`) — a list of the rig's own
#   fact names per request, never the prompt body. A host that resolved the
#   command sends its shim's text; a host that did not sends the literal
#   characters the user typed.
# * **The verdict is written, not decided here.** `invocation.json` records what
#   happened; `tests/test_container_documented_invocation.py` grades it against
#   the matrix's `documented_invocation.resolves` cell. The entrypoint exits 0
#   either way, exactly as `run_host` does, because "the host refused the
#   documented spelling" is a finding rather than a rig failure.
# --------------------------------------------------------------------------- #

#: A no-credential placeholder for hosts that insist on *some* API key being set
#: before they will talk to a base_url. It is not a secret and is not one of
#: `CREDENTIAL_VARS`; the stub accepts anything.
STUB_PLACEHOLDER_KEY = "stub-not-a-credential"

#: Escape sequences a full-screen TUI writes, stripped before anything is matched
#: or written to the bundle. A TUI transcript is a repaint stream, so this makes
#: it readable rather than parseable: nothing in this module ever asserts on the
#: transcript's *content* except a host's own refusal line, which both hosts emit
#: as plain text.
_ANSI = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"    # OSC … BEL / ST
    r"|\x1bP[^\x1b]*\x1b\\"                 # DCS … ST
    r"|\x1b\[[0-9;?<>=]*[ -/]*[@-~]"        # CSI
    r"|\x1b[()][A-Za-z0-9]"                 # designate charset
    r"|\x1b[=>NOMc78]"                      # single-character escapes
)


def _plain(raw: bytes) -> str:
    return _ANSI.sub("", raw.decode("utf-8", "replace"))


def invocation_facts(host: str, typed: str, plan_rel: str,
                     marker: str) -> dict[str, str]:
    """The graded ladder, as `{fact: regex}` for the stub's `--facts` probe.

    Every entry is a question the acceptance row asks in as many words, and the
    ladder is ordered so that a partial result still names how far the host got
    rather than collapsing to one boolean:

    ``typed_string_verbatim``
        the characters the user typed are in the prompt at all — true on a host
        that resolved the command *and* on one that passed the line through.
    ``shim_body`` / ``shim_dispatcher_block``
        the host expanded its own installed shim into the turn. This is the
        discriminator: a host that never resolved `/multi-llm` has no reason to
        have read the file.
    ``skill_dir_placeholder_unfilled``
        the NEGATIVE. `{SKILL_DIR}` surviving into the prompt means the shim was
        rendered without substitution, which is a shipped-artifact bug wearing a
        pass.
    ``unattended_default_baked``
        D5's claim, at the point of use: `--host <host> --yes --exec-subagents`.
    ``mode_flag_with_argument``
        the flag and the plan path still adjacent, so a host parser that dropped,
        reordered or word-split the argument cannot satisfy it.
    ``argument_marker``
        the distinguishable argument survived at all — the weaker half of the
        row above, kept so "arrived mangled" and "did not arrive" stay apart.
    """
    return {
        "typed_string_verbatim": re.escape(typed),
        "shim_body": r"Multi-LLM . (?:Codex|OpenCode) entry shim",
        "shim_dispatcher_block": re.escape(f'python "{SKILL_DIR}/multi_llm.py"'),
        "skill_dir_placeholder_unfilled": r"\{SKILL_DIR\}",
        "unattended_default_baked": re.escape(
            f"--host {host} --yes --exec-subagents"),
        "mode_flag_with_argument": r"--review-plan\s+" + re.escape(plan_rel),
        "argument_marker": re.escape(marker),
    }


#: The facts whose presence means the host really resolved the documented string,
#: plus the one whose presence means it resolved it badly.
RESOLVED_FACTS = ("shim_body", "shim_dispatcher_block",
                  "unattended_default_baked", "mode_flag_with_argument")
UNRESOLVED_FACT = "skill_dir_placeholder_unfilled"


def wire_stub_host(host: str, home: Path, base_url: str) -> dict:
    """Point the HOST's own model at the container stub. No credential, no route.

    The credentialed twin of this is `wire_credentials`, and the differences are
    the point: there is no key here, the base URL is loopback, and the row runs
    under `--network none`, so the host has no endpoint to reach except ours.
    Everything else — the trust entry, the permission gates, opencode's baked
    cold-start tree — is the same wiring for the same reasons, and is shared with
    that function rather than re-derived.
    """
    home.mkdir(parents=True, exist_ok=True)
    seed = RIG_DIR / "home_seed" / host
    overlay: dict[str, str] = {}

    if host == "codex":
        codex_home = home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text(
            'model_provider = "stub"\n'
            + _codex_first_run_config()
            + "\n"
            "[model_providers.stub]\n"
            'name = "stub"\n'
            f'base_url = "{base_url}"\n'
            'wire_api = "responses"\n'
            # `env_key` rather than the credentialed path's command-based auth:
            # there is no key to keep out of the file here, and the warning that
            # form exists to suppress is about OpenRouter's catalog, which this
            # row never reaches.
            'env_key = "MLLM_STUB_API_KEY"\n',
            encoding="utf-8")
        overlay["CODEX_HOME"] = str(codex_home)
        overlay["MLLM_STUB_API_KEY"] = STUB_PLACEHOLDER_KEY

    elif host == "opencode":
        # The image's baked cold-start tree, for the same reason the credentialed
        # row copies it: without it the first launch npm-installs ~63 MB, which
        # `--network none` makes impossible rather than slow.
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        config_dir = home / ".config" / "opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "opencode.json").write_text(json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "provider": {"stub": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Stub",
                "options": {"baseURL": base_url, "apiKey": STUB_PLACEHOLDER_KEY},
                "models": {"stub-model": {"name": "Stub Model"}},
            }},
            "permission": {"bash": "allow", "edit": "allow", "webfetch": "deny"},
        }, indent=2), encoding="utf-8")
    else:
        raise SystemExit(
            f"entrypoint: no stub wiring for host {host!r} — the "
            f"documented-invocation probe runs on the INSTALLABLE hosts only")

    log(f"wired {host} at the stub in {home} (env overlay: {sorted(overlay)})")
    return overlay


def _stub_facts_seen(log_path: Path) -> tuple[set[str], int]:
    """`(fact names seen, request count)` from the stub's log, health excluded."""
    seen: set[str] = set()
    requests = 0
    if not log_path.is_file():
        return seen, requests
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:      # a torn last line while the stub is still writing
            continue
        if entry.get("surface") == "health" or entry.get("path") == "/__health":
            continue
        requests += 1
        seen.update(str(name) for name in (entry.get("facts") or []))
    return seen, requests


def drive_documented_invocation(host: str, entry: dict, home: Path,
                                overlay: dict, marker: str,
                                stub_log: Path) -> dict:
    """Launch a FRESH session, type the documented string, and record what reached
    the model. Never asserts — the grader does that over `invocation.json`.

    **A fresh session is satisfied by construction, not by a sleep.** Both
    installable hosts are `reload: new-session`, and this row launches exactly one
    host process, started after `install-host` has already written the shim: there
    is nothing to reload into, because the process IS the new session.

    **Why a pty.** Both hosts' documented surface is the interactive one — the
    README says *start a new session in that host and invoke it exactly as in
    Claude Code* — and a full-screen TUI that finds no terminal either refuses to
    start or falls back to a different parser, which would make the row answer a
    question nobody asked. The transcript is captured with its escape sequences
    stripped, and only a host's own refusal line is ever matched in it.

    **Why TMPDIR moves.** `posture.yaml` mounts /tmp `noexec` (docker's default
    for `--tmpfs`), and opencode's TUI is a bun single-file binary that extracts
    its native render library to $TMPDIR and dlopen()s it — under noexec it exits
    before painting with `failed to map segment from shared object`. The bundle
    bind is the one writable exec-capable path in this posture, so the probe puts
    $TMPDIR there and deletes it again afterwards. This is a RIG accommodation and
    is recorded as one: it changes where a temporary file lands, never what the
    host is asked or what it does with it.
    """
    cell = dict(entry.get("documented_invocation") or {})
    if not cell:
        raise SystemExit(
            f"entrypoint: host_matrix.yaml carries no `documented_invocation` "
            f"cell for {host!r}. The row cannot improvise the string a document "
            f"tells users to type — that is precisely the thing under test.")

    plan_rel = f"plans/{marker}.md"
    typed = str(cell["typed"]).replace("{plan}", plan_rel)
    plan_path = WORKSPACE / plan_rel
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        f"# {marker}\n\nA one-line plan whose only job is to be a "
        f"distinguishable argument.\n", encoding="utf-8")

    argv = [str(token).replace("<stub-model>",
                               str(cell.get("stub_model") or "stub-model"))
            for token in (cell.get("launch_argv") or [])]
    if not argv:
        raise SystemExit(
            f"entrypoint: {host}: `documented_invocation.launch_argv` is empty")
    binary = argv[0]
    argv[0] = str(REAL_DIR / binary)
    if not Path(argv[0]).exists():
        raise SystemExit(f"entrypoint: no real host binary at {argv[0]}")

    ready_rx = re.compile(str(cell["ready"])) if cell.get("ready") else None
    refusal_rx = (re.compile(str(cell["refusal"]))
                  if cell.get("refusal") else None)
    ready_timeout = float(cell.get("ready_timeout_s") or 120)
    settle_s = float(cell.get("settle_s") or 2.0)
    submit_gap_s = float(cell.get("submit_gap_s") or 1.5)
    enter_grace_s = float(cell.get("enter_grace_s") or 15.0)
    max_enters = int(cell.get("max_enters") or 3)
    after_first_request_s = float(cell.get("after_first_request_s") or 30.0)
    wall = float(cell.get("wall_clock_s") or 300)

    tmpdir = OUT / "tui-tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["TERM"] = str(cell.get("term") or "xterm-256color")
    env["TMPDIR"] = str(tmpdir)
    env.update(overlay)
    for var, value in (entry.get("env_extra") or {}).items():
        env[str(var)] = str(value)

    log(f"documented-invocation: launching {binary} in a pty "
        f"(fresh session; marker {marker})")
    log(f"documented-invocation: will type {typed!r}")

    started = time.time()
    pid, fd = pty.fork()
    if pid == 0:                                    # pragma: no cover - child
        try:
            os.chdir(str(WORKSPACE))
            os.execve(argv[0], argv, env)
        finally:
            os._exit(127)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))

    captured = bytearray()
    ready_at: float | None = None
    typed_at: float | None = None
    enters: list[float] = []
    refused_at: float | None = None
    first_request_at: float | None = None
    facts: set[str] = set()
    requests = 0
    stop_reason = "wall-clock"

    def _elapsed() -> float:
        return time.time() - started

    while _elapsed() < wall:
        readable, _, _ = select.select([fd], [], [], 0.25)
        if readable:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                stop_reason = "host exited"
                break
            captured += chunk
        text = _plain(bytes(captured))

        if ready_at is None:
            if ready_rx is not None and ready_rx.search(text):
                ready_at = _elapsed()
                log(f"documented-invocation: session ready after {ready_at:.1f}s")
            elif _elapsed() > ready_timeout:
                # Not a failure HERE: the row records that the session never
                # announced itself and types anyway, so the bundle distinguishes
                # "the host never started" from "the host rejected the string".
                ready_at = -1.0
                log(f"documented-invocation: no ready marker within "
                    f"{ready_timeout}s — typing anyway, and recording it")
            else:
                continue

        if typed_at is None:
            if _elapsed() < (max(ready_at, 0.0) + settle_s):
                continue
            os.write(fd, typed.encode("utf-8"))
            typed_at = _elapsed()
            continue

        if not enters:
            if _elapsed() >= typed_at + submit_gap_s:
                os.write(fd, b"\r")
                enters.append(_elapsed())
            continue

        facts, requests = _stub_facts_seen(stub_log)
        if requests and first_request_at is None:
            first_request_at = _elapsed()
        if refusal_rx is not None and refused_at is None and refusal_rx.search(text):
            refused_at = _elapsed()
            stop_reason = "host refused the spelling"
            break
        if "shim_body" in facts:
            stop_reason = "the shim reached the model"
            break
        if (first_request_at is not None
                and _elapsed() >= first_request_at + after_first_request_s):
            stop_reason = "a turn was sent and settled"
            break
        # A composer that opened an autocomplete popup eats the first Enter, so
        # the recipe that was MEASURED to work sends a second one. Bounded, and
        # counted into the record: an answer that needed three Enters is a
        # different fact from one that needed one.
        if (len(enters) < max_enters
                and _elapsed() >= enters[-1] + enter_grace_s
                and not requests):
            os.write(fd, b"\r")
            enters.append(_elapsed())

    facts, requests = _stub_facts_seen(stub_log)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)

    transcript = _plain(bytes(captured))
    (OUT / "invocation-transcript.txt").write_text(transcript, encoding="utf-8")

    resolved = (all(name in facts for name in RESOLVED_FACTS)
                and UNRESOLVED_FACT not in facts)
    record = {
        "host": host,
        "surface": str(cell.get("surface") or "tui"),
        "launch_argv": [binary, *argv[1:]],
        "typed": typed,
        "marker": marker,
        "plan": plan_rel,
        "resolved": resolved,
        "facts_seen": sorted(facts),
        "facts_missing": sorted(name for name in RESOLVED_FACTS
                                if name not in facts),
        "stub_requests": requests,
        "session_ready": ready_at is not None and ready_at >= 0,
        "ready_after_s": None if ready_at is None or ready_at < 0 else round(ready_at, 2),
        "typed_after_s": None if typed_at is None else round(typed_at, 2),
        "enters_sent": len(enters),
        "refusal_observed": refused_at is not None,
        "refusal_pattern": str(cell.get("refusal") or "") or None,
        "declared_resolves": bool(cell.get("resolves")),
        "stop_reason": stop_reason,
        "wall_clock_s": round(_elapsed(), 2),
        "transcript_bytes": len(transcript),
    }
    (OUT / "invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"documented-invocation: {host} -> resolved={resolved} "
        f"({stop_reason}; facts {sorted(facts)})")
    return record


def _host_matrix() -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - the image ships pyyaml
        return {}
    path = RIG_DIR / "host_matrix.yaml"
    if not path.is_file():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("hosts", {})


def _matrix_row(host: str | None, row_id: str) -> dict:
    """One row out of `host_matrix.yaml`, or `{}` — the single spelling of that
    lookup, because two of them read the row at two different points in the run
    (the seeding step and the launch step) and must never disagree about it."""
    entry = _host_matrix().get(host) or {}
    return next((dict(r) for r in (entry.get("rows") or [])
                 if r.get("id") == row_id), {})


# --------------------------------------------------------------------------- #
# Credentialed host lane (C6, C6a, C6c)
# --------------------------------------------------------------------------- #

OPENROUTER_API = "https://openrouter.ai/api/v1"
#: The Anthropic-compatible "skin". Claude Code appends `/v1/messages` itself, so
#: this is the bare origin + `/api`, NOT the `/api/v1` the other two hosts take.
OPENROUTER_ANTHROPIC_BASE = "https://openrouter.ai/api"


def _openrouter_get(path: str, key: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        f"{OPENROUTER_API}{path}", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def key_usage(key: str) -> dict | None:
    """The key's own spend readback — `usage`, `limit`, `limit_remaining`.

    This is the fallback actual the plan names: attributable to one row precisely
    because the key is dedicated and rows run one at a time.
    """
    try:
        return _openrouter_get("/key", key).get("data") or None
    except (urllib.error.URLError, OSError, ValueError) as exc:  # noqa: BLE001
        log(f"usage readback unavailable: {type(exc).__name__}")
        return None


def credential_preflight(host: str, entry: dict, key: str,
                         row: dict | None = None) -> dict:
    """Refuse before a token is spent, and record what the refusal was based on.

    Two checks, in the order the plan puts them:

    * **the key's own spend limit.** `MLLM_E2E_BUDGET_USD` is a pre-flight refusal
      and cannot stop a runaway mid-run, so the only limit that holds is the one
      enforced by the party doing the billing. A key reporting no `limit` is
      refused — `MLLM_ALLOW_UNLIMITED_KEY=1` is the deliberate, recorded override
      for a bounded operator-supervised run, and it is never a default;
    * **the pinned model resolves.** A pin the catalog does not name aborts the
      row with the pin quoted rather than substituting a neighbour: a retired slug
      is a diff, not a run-time fallback. A catalog that cannot be READ is the
      same refusal — "we could not check" is not evidence the pin is good, and
      failing open here would spend a paid turn on an unproven model.
      `MLLM_ALLOW_UNVERIFIED_MODEL=1` is the deliberate, recorded override, on
      the same terms as the spend-limit one above.
    """
    census: dict = {"host": host, "checks": {}}

    data = key_usage(key) or {}
    limit = data.get("limit")
    census["key"] = {"limit": limit, "usage": data.get("usage"),
                     "limit_remaining": data.get("limit_remaining")}
    if limit is None:
        if os.environ.get("MLLM_ALLOW_UNLIMITED_KEY") != "1":
            raise SystemExit(
                "entrypoint: this key reports NO platform spend limit "
                "(`GET /api/v1/key` -> limit: null). The local budget guard is a "
                "pre-flight refusal and cannot stop a runaway mid-run, so the host "
                "lane refuses. Use a key carrying a key-level spend limit, or set "
                "MLLM_ALLOW_UNLIMITED_KEY=1 deliberately for a bounded, supervised "
                "run (never from a shell profile).")
        census["checks"]["spend_limit"] = "overridden: MLLM_ALLOW_UNLIMITED_KEY=1"
        log("WARNING: key reports no platform spend limit; running under an "
            "explicit MLLM_ALLOW_UNLIMITED_KEY override")
    else:
        census["checks"]["spend_limit"] = f"limit={limit}"

    slug = pinned_model(entry, row)
    census["model"] = slug
    host_pin = ((entry.get("model") or {}).get("slug") or "").strip()
    if slug != host_pin:
        # WHICH override, never just "an override": the two have different
        # authority. The env one is one variable typed by an operator for one
        # run and leaves no diff; the row one is checked into host_matrix.yaml
        # with its provenance and is what the next run will drive again.
        census["model_override"] = (
            "MLLM_HOST_MODEL — deliberate, recorded"
            if os.environ.get("MLLM_HOST_MODEL", "").strip() else
            f"host_matrix.yaml row pin on {(row or {}).get('id')!r} "
            f"(host pin is {host_pin!r})")
    try:
        catalog = {m.get("id") for m in (_openrouter_get("/models", key).get("data") or [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:  # noqa: BLE001
        if os.environ.get("MLLM_ALLOW_UNVERIFIED_MODEL") != "1":
            raise SystemExit(
                f"entrypoint: the OpenRouter model catalog could not be read "
                f"({type(exc).__name__}: {exc}), so the pin {slug!r} for host "
                f"{host!r} is UNPROVEN. This lane spends real tokens, and the "
                f"fail-before-spend contract does not allow a row to start on a "
                f"pin nothing checked: fix the catalog request, or set "
                f"MLLM_ALLOW_UNVERIFIED_MODEL=1 deliberately for a bounded, "
                f"supervised run (never from a shell profile).") from exc
        census["checks"]["model_pin"] = (
            f"overridden: MLLM_ALLOW_UNVERIFIED_MODEL=1 after "
            f"{type(exc).__name__}")
        log(f"WARNING: OpenRouter catalog unreadable ({type(exc).__name__}); "
            f"running {slug!r} under an explicit MLLM_ALLOW_UNVERIFIED_MODEL "
            f"override")
    else:
        # A leading `~` marks an OpenRouter ALIAS (`~deepseek/…-latest`). The
        # catalog lists the ids an alias RESOLVES TO, without the prefix, so the
        # membership check runs on the unprefixed form while `invoke_argv` keeps
        # the raw slug — the alias is what the host is meant to send, and
        # rewriting it here would silently re-pin the row.
        canonical = slug[1:] if slug.startswith("~") else slug
        if slug and not ({slug, canonical} & catalog):
            raise SystemExit(
                f"entrypoint: the pinned model {slug!r} for host {host!r} is not in "
                f"the OpenRouter catalog (checked as {canonical!r} too). A retired "
                f"or renamed slug is a DIFF — a new pin checked into "
                f"host_matrix.yaml with its provenance — never a run-time "
                f"substitution.")
        census["checks"]["model_pin"] = (
            f"resolved: alias {slug} -> {canonical}"
            if slug and slug not in catalog else "resolved")
    return census


def pinned_model(entry: dict, row: dict | None = None) -> str:
    """The row's model: the checked-in pin, or a deliberately recorded override.

    Three sources, most specific first, and each of them is a PIN — none of them
    is a run-time "cheapest" lookup, which is the thing that would make a cost
    table meaningless:

    * ``MLLM_HOST_MODEL`` — one variable, typed by an operator, for one run. It
      still wins, because it is the escape hatch a red row is debugged with and
      a checked-in row pin must not be able to trap an operator inside it.
    * the ROW's own ``model.slug`` — added 2026-08-21 for the one case
      ``host_matrix.yaml``'s claude-code host block had already anticipated in
      prose and then measured four times over: the host pin is non-Anthropic,
      and ONE row on that host fails in a way that is the MODEL rather than our
      shim. Per-row, because the alternative is re-pinning the whole host and
      paying the Anthropic price on every row that does not need it.
    * the HOST's ``model.slug`` — the default, and still the only pin most rows
      have.

    Whichever slug ran is recorded in the row's credential census, together with
    WHICH of the three it came from, so a run is never ambiguous about what it
    drove.
    """
    override = os.environ.get("MLLM_HOST_MODEL", "").strip()
    if override:
        return override
    row_pin = ((row or {}).get("model") or {}).get("slug") or ""
    return row_pin.strip() or ((entry.get("model") or {}).get("slug") or "").strip()


def wire_credentials(host: str, home: Path, key: str, slug: str,
                     route: str = "file") -> dict:
    """Per-host OpenRouter wiring, written into the tmpfs $HOME (C6a rule 3).

    Returns the environment overlay the HOST process needs. Nothing here touches
    the provider identities under ``/work/homes``: those carry the stub's fake
    session key and must keep carrying it, which is the whole role separation the
    wrapper exists for.
    """
    home.mkdir(parents=True, exist_ok=True)
    overlay: dict[str, str] = {}
    seed = RIG_DIR / "home_seed" / host

    if host == "claude-code":
        # The synthesized ~/.claude.json: onboarding done plus a TRUST entry for
        # the workspace. ~/.claude.json is never mounted from the developer's home
        # (87 KB of per-project history and MCP definitions that routinely embed
        # third-party keys); a minimal one is synthesized instead.
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        _write_claude_settings(home)
        overlay["ANTHROPIC_BASE_URL"] = OPENROUTER_ANTHROPIC_BASE
        overlay["ANTHROPIC_AUTH_TOKEN"] = key
        # PRESENT AND EMPTY, not unset. An unset variable lets Claude Code fall
        # back to authenticating against Anthropic directly, silently spending the
        # developer's own subscription and producing a green run that proves the
        # wrong thing.
        overlay["ANTHROPIC_API_KEY"] = ""
        overlay["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    elif host == "codex":
        # COMMAND-BASED auth rather than `env_key`: a plain env_key authenticates
        # but makes Codex skip its model-catalog refresh and emit
        # fallback-metadata "Unknown model" warnings for non-OpenAI ids. The key
        # itself never lands in the file — the command reads it from the
        # environment at request time.
        codex_home = home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text(
            'model_provider = "openrouter"\n'
            + _codex_first_run_config()
            + "\n"
            "[model_providers.openrouter]\n"
            'name = "OpenRouter"\n'
            f'base_url = "{OPENROUTER_API}"\n'
            # MEASURED 2026-08-14 on codex-cli 0.146.0: `wire_api = "chat"` is
            # REFUSED at config load ("no longer supported. How to fix: set
            # wire_api = responses"), so codex exits 1 with an empty stdout
            # before a token is spent. OpenRouter serves POST /api/v1/responses,
            # so `responses` is the value that works.
            'wire_api = "responses"\n'
            "\n"
            "[model_providers.openrouter.auth]\n"
            'command = "sh"\n'
            'args = ["-c", "echo $OPENROUTER_API_KEY"]\n',
            encoding="utf-8")
        overlay["CODEX_HOME"] = str(codex_home)

    elif host == "opencode":
        # The baked cold-start tree (node_modules + the models.dev catalog) comes
        # from the image's own warm-up, folded into home_seed/opencode. Without it
        # the first `opencode run` npm-installs ~63 MB and fetches ~3.6 MB of
        # catalog before it answers anything.
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        # ...but NOT its stub provider block: that file carries the provider
        # lane's `{{base_url}}` template and would shadow the host wiring.
        _write_opencode_config(home, slug, route)
        # The stored credential — the form `providers.yaml:116` documents
        # (`opencode auth login --provider openrouter --method api`), written
        # straight into the tmpfs $HOME and never into the repo.
        auth = home / ".local" / "share" / "opencode" / "auth.json"
        auth.parent.mkdir(parents=True, exist_ok=True)
        auth.write_text(
            json.dumps({"openrouter": {"type": "api", "key": key}}, indent=2),
            encoding="utf-8")
        os.chmod(auth, 0o600)
    else:
        raise SystemExit(f"entrypoint: no credential wiring for host {host!r}")

    log(f"wired {host} credentials into {home} "
        f"(env overlay: {sorted(overlay)})")
    return overlay


def _write_claude_settings(home: Path) -> None:
    """The permission form: the narrowest non-interactive grant that still runs
    the shim's own commands, and never a `--dangerously-*` bypass.

    `claude -p` auto-denies anything not pre-granted, and MEASURED 2026-08-14 that
    is not a theoretical risk: with only a hand-written allow list the very first
    thing the host did was call the **Skill** tool —
    `{"skill": "multi-llm", "args": "--ask …"}` — and be denied, then burn its
    whole turn budget hunting for the file by hand.

    So the list is DERIVED, never invented: it is SKILL.md's own `allowed-tools`
    frontmatter (the grants the shipped shim declares it needs) plus the one grant
    that frontmatter cannot carry — permission to invoke the skill at all.
    """
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "permissions": {
            "allow": ["Skill", "Skill(multi-llm)", *_skill_allowed_tools()],
            "deny": [],
            "defaultMode": "acceptEdits",
        },
        "includeCoAuthoredBy": False,
    }, indent=2), encoding="utf-8")


def _codex_first_run_config() -> str:
    """The part of codex's config.toml that has nothing to do with WHICH auth.

    Shared by the OpenRouter wiring and the fallback (vendor-auth) one, because
    both start from a fresh tmpfs $HOME and both die the same way without it.
    """
    return (
        'approval_policy = "never"\n'
        # MEASURED: codex's own sandbox is bubblewrap, which needs unprivileged
        # user namespaces this posture does not grant. See the `sandbox_finding`
        # cell in host_matrix.yaml — the container IS the sandbox here, and a
        # second one that cannot initialize only kills the row.
        'sandbox_mode = "danger-full-access"\n'
        "\n"
        # The trust surface: a fresh tmpfs $HOME starts codex at its first-run
        # posture, and a host that stops to ask for trust fails the row on its
        # wall-clock bound with no useful signal.
        f'[projects."{WORKSPACE}"]\n'
        'trust_level = "trusted"\n'
    )


def _opencode_package_spec() -> str | None:
    """The npm spec this image's seed was warmed with, or ``None``.

    Read from the capability data rather than typed here, so the runtime
    declaration and the seeded cache directory name are the **same normalized
    spec** — a mismatch is not an error on this host, it is a silent drop.

    **The path insert is what makes that read possible at all.** This file runs as
    ``/usr/bin/python3 /opt/mllm/rig/entrypoint.py`` with ``/work/sample-project``
    as its cwd, so ``sys.path[0]`` is the *rig* directory and nothing on the path
    is the skill: a bare ``from utils import ...`` raises ``ImportError`` every
    time. The old blanket ``except`` then swallowed that into ``None`` and the
    package row died on a SystemExit blaming an absent ``package:`` block that was
    in fact sitting right there in ``hosts/capabilities.yaml`` — a misdirection
    worth more than the two lines it costs to prevent. Same idiom as
    ``wrapper.py``; the ``except`` stays only for a rig genuinely running outside
    an installed skill, which is what ``ImportError`` alone names.
    """
    # THE STAGED DIRECTORY WINS WHEN IT IS THERE, and it is there in every image
    # built since §10b landed. The registry spec below stays as the fallback for
    # a rig driving an image that predates that layer (or a future published
    # package), so this function still answers the question its name asks -- what
    # do we put in `plugin[]`? -- rather than two different questions by build.
    if (OPENCODE_PLUGIN_DIR / "package.json").is_file():
        return str(OPENCODE_PLUGIN_DIR)
    if str(SKILL_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_DIR))
    try:
        from utils import host_capabilities as caps
    except ImportError:  # pragma: no cover - the rig runs outside the skill too
        return None
    try:
        package = caps.entry("opencode").get("package")
    except Exception:  # pragma: no cover - unknown host
        return None
    if not package or package.get("registry") != "npm":
        return None
    version = package.get("version")
    return f"{package['name']}@{version}" if version else package["name"]


def _write_opencode_config(home: Path, slug: str,
                           route: str = "file") -> None:
    """opencode.json: the model declaration, the permission gates, and — on the
    package row only — the plugin declaration.

    **The plugin write is gated on the row, and the gating is the load-bearing
    part.** This function serves *every* opencode row, so an unconditional
    ``plugin[]`` would reach the file-shim rows this work measures against:
    changing their provenance, invalidating the improvisation reading taken
    beside them, and handing those rows **two command sources at once** (the
    shim's file and the plugin's injected command). So the switch is the row's
    own ``route`` field, and a guard pins that every other row's rendered config
    stays byte-identical to what it was.

    **And the runtime write is not optional on the package row.** The image
    UNPACKS the plugin at /opt/mllm/opencode-plugin (Dockerfile §10b) and declares
    it nowhere — a declaration is per-$HOME and this one is a tmpfs seeded per
    row. Without this write the package sits in the image and is never loaded, at
    exit 0, with nothing in the logs.

    This docstring used to describe a *warm-up step* that pre-extracted the
    package from the registry into the seed. No such step was ever written, the
    package is unpublished, and `--network none` could not have fetched it; §10b
    is what actually puts the payload in the image.
    """
    config_dir = home / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    external: dict[str, str] = {
        f"{SKILL_DIR}/**": "allow",
        f"{SKILL_DIR}/*": "allow",
    }
    document: dict = {
        "$schema": "https://opencode.ai/config.json",
        # The model is declared under the built-in `openrouter` provider so a
        # slug the models.dev catalog does not name still resolves.
        "provider": {"openrouter": {"models": {slug: {"name": slug}}}},
    }

    if route == "package":
        spec = _opencode_package_spec()
        if not spec:
            raise SystemExit(
                "entrypoint: the plugin row needs opencode's package: block in "
                "hosts/capabilities.yaml, and it is absent or not an npm route."
            )
        document["plugin"] = [spec]
        # The package payload is read from WHEREVER THE SPEC POINTS, never from
        # /opt/mllm/skill, and `external_directory` has to match the path opencode
        # actually reads — not the path the bytes came from. Without these entries
        # the payload matches only `"*": "deny"` and every read is auto-rejected,
        # producing the documented signature (`! permission requested:
        # external_directory … auto-rejecting`) and then a run that ends at exit 0
        # with no dispatch. A row that "passes" because the gate silently
        # swallowed the read is the worst outcome available, which is why
        # restaging the payload does not fix this on its own.
        #
        # BOTH roots are granted, because the spec decides which one is live: a
        # local directory spec (this image, §10b) is read in place, while a
        # registry spec is extracted under $HOME/.cache/opencode first. Granting
        # the one that is not in use costs nothing — it names a path no read will
        # reach — and granting only the wrong one is the silent-rejection failure
        # above.
        if spec.startswith("/"):
            external[f"{spec}/**"] = "allow"
            external[f"{spec}/*"] = "allow"
        cache = home / ".cache" / "opencode"
        external[f"{cache}/**"] = "allow"
        external[f"{cache}/*"] = "allow"

    document["permission"] = {
            "bash": "allow", "edit": "allow", "webfetch": "deny",
            # THE SKILL DIR IS AN "EXTERNAL DIRECTORY" TO THIS HOST, and that is
            # a separate gate from `bash`. MEASURED 2026-08-14: the review-plan
            # row ran `uv run --project /opt/mllm/skill …` happily (bash: allow)
            # and was then auto-rejected the moment it tried to READ a file under
            # the same tree —
            #   ! permission requested: external_directory
            #     (/opt/mllm/skill/references/*); auto-rejecting
            # — and the run simply ended there, exit 0, no dispatch. The
            # instruction core legitimately tells an agent to read
            # `references/resume-detection.md`, so any mode that resumes hits
            # this; `ask` does not, which is why the cheapest row missed it.
            # `"*": "deny"` stays last and stays: the allow entries are
            # additive, never a widening of the default.
            "external_directory": {**external, "*": "deny"},
    }
    (config_dir / "opencode.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )


def wire_fallback_credentials(host: str, home: Path, slug: str,
                              route: str = "file") -> tuple[dict, list[Path]]:
    """C6a's documented fallback: the host's OWN subscription / vendor auth.

    The default path injects one `OPENROUTER_API_KEY` and mounts nothing. This is
    the other one — for a developer with no OpenRouter account, or for testing a
    host under its native auth — and it is the ONLY path on which anything from
    the developer's home directory enters the container.

    Order is load-bearing and is the reason the copy-in lives at the bottom of
    this function rather than beside its caller: `home_seed/opencode` ships a
    value-free `auth.json` template, so a seed copied *after* the credential would
    silently replace it with the provider lane's stub session key — a row that
    then fails as "not authenticated" with the real file sitting one directory
    away, mounted and unread.

    Returns the host's environment overlay and the files it copied in (the second
    is what `run_host` redacts by value, since on this path the secret is whatever
    the vendor wrote rather than one key this process was handed).
    """
    home.mkdir(parents=True, exist_ok=True)
    overlay: dict[str, str] = {}
    seed = RIG_DIR / "home_seed" / host

    if host == "claude-code":
        # The synthesized ~/.claude.json again — it is never mounted, on ANY path
        # (87 KB of per-project history and MCP definitions that routinely embed
        # third-party keys). Only `.credentials.json` is allowlisted.
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        _write_claude_settings(home)
        # No ANTHROPIC_* overlay at all: the copied `.credentials.json` is the
        # auth, and a present-and-empty ANTHROPIC_API_KEY here would be pointing
        # the host away from the very file this mode exists to use.
        overlay["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    elif host == "codex":
        codex_home = home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        # No `model_provider` and no `[model_providers.*]`: codex reads the copied
        # ~/.codex/auth.json and talks to its own vendor.
        (codex_home / "config.toml").write_text(
            _codex_first_run_config(), encoding="utf-8")
        overlay["CODEX_HOME"] = str(codex_home)

    elif host == "opencode":
        if seed.is_dir():
            shutil.copytree(seed, home, dirs_exist_ok=True)
        _write_opencode_config(home, slug, route)

    else:
        raise SystemExit(
            f"entrypoint: no fallback credential wiring for host {host!r}")

    copied = copy_in_fallback_credentials(host, home)
    log(f"wired {host} fallback credentials into {home} "
        f"(env overlay: {sorted(overlay)}, {len(copied)} file(s) copied in)")
    return overlay, copied


def copy_in_fallback_credentials(host: str, home: Path) -> list[Path]:
    """C6a rule 3: copy the allowlisted `:ro` mounts into the tmpfs $HOME.

    A CLI that refreshes an OAuth token needs to write: a `:ro` mount would make
    that fail, and a `:rw` one would let a container full of third-party binaries
    write back to the developer's real credential file. Copy-on-start resolves
    both — every write then lives and dies in RAM with the container.

    The copy TARGET is `creds.yaml`'s own `source:` path re-read against the
    container's `$HOME`, so the allowlist stays the single statement of where a
    credential comes from and where it lands.
    """
    policy = _rig_policy("creds.yaml")
    entries = (policy.get("allowlist") or {}).get(host) or []
    if not entries:
        raise SystemExit(
            f"entrypoint: --creds fallback, but creds.yaml's allowlist names no "
            f"credential file for host {host!r}. That list is the complete set "
            f"(C6a rule 2), so this is a diff to review — never a path this "
            f"process picks at run time.")
    mode = int(str((policy.get("tmpfs_home") or {}).get("copy_mode") or "0600"), 8)
    copied: list[Path] = []
    for entry in entries:
        declared = str(entry.get("source") or "")
        source = Path(str(entry.get("dest") or ""))
        if not declared.startswith("~/"):
            raise SystemExit(
                f"entrypoint: creds.yaml entry {declared!r} is not written "
                f"relative to `~`, so it names no place under the container's "
                f"tmpfs $HOME to copy into")
        target = home / declared[2:]
        if not source.is_file():
            raise SystemExit(
                f"entrypoint: --creds fallback expects {source} (the mount of "
                f"{declared}) and it is not there. Mount it `:ro` exactly as "
                f"creds.yaml names it, or run the default `--creds openrouter` "
                f"path, which mounts nothing at all.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, mode)
        copied.append(target)
        log(f"fallback credential: {source} -> {target} (mode {mode:04o})")
    return copied


def fallback_secret_values(paths) -> list[str]:
    """Every long string inside the copied credential files, for write-time scrubbing.

    The OpenRouter path knows its one secret by value and redacts it. This path
    does not: the secret is whatever the vendor wrote into the file, so it is read
    back out of the files that were just copied. Values are used only as
    replacement needles — nothing here is logged, recorded or returned to a caller
    that writes it out.
    """
    secrets: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            if len(node) >= 20:
                secrets.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in paths:
        try:
            walk(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    # Longest first: a token that contains another as a prefix must be replaced
    # before the shorter one turns it into `<redacted:…>xyz`.
    return sorted(secrets, key=len, reverse=True)


def _skill_allowed_tools() -> list[str]:
    """`allowed-tools` from the shipped SKILL.md — the product's own declaration.

    Read out of the frontmatter rather than restated here, so the rig cannot
    become a second source of truth for what the shim needs to run.
    """
    # A LINE SCAN, not yaml.safe_load. MEASURED: the frontmatter also carries
    # `argument-hint: [--review-plan|--apply-suggestions|…]`, which is not valid
    # YAML flow syntax, so a whole-frontmatter parse raises — and an
    # `except: return []` turned that into an EMPTY grant list, i.e. a
    # claude-code row that loads the skill and is then denied every command in
    # it. That failure looked exactly like the permission bug it followed.
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    tools: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("allowed-tools:"):
            inside = True
            continue
        if inside:
            if line.startswith("  - "):
                tools.append(line[4:].strip())
                continue
            break
    if not tools:
        raise SystemExit(
            "entrypoint: SKILL.md declares no parseable `allowed-tools` — "
            "refusing to launch a claude-code row with an empty grant list, "
            "which denies every command in the shim it just loaded")
    return tools


def assert_claude_key_is_empty(env: dict) -> None:
    """`ANTHROPIC_API_KEY` present-and-EMPTY — the one that silently costs money."""
    if "ANTHROPIC_API_KEY" not in env:
        raise SystemExit(
            "entrypoint: ANTHROPIC_API_KEY is UNSET for the claude-code host. It "
            "must be present and empty: unset lets Claude Code fall back to "
            "authenticating against Anthropic directly, spending the developer's "
            "own subscription and producing a green run that proves the wrong "
            "thing.")
    if env["ANTHROPIC_API_KEY"] != "":
        raise SystemExit(
            "entrypoint: ANTHROPIC_API_KEY is present but NOT empty — this row "
            "would not be running on the OpenRouter key it claims to.")


def build_host_argv(entry: dict, row: dict) -> list[str]:
    """Render the row's `invoke_argv` from the matrix. No argv is improvised."""
    slug = pinned_model(entry, row)
    bound = entry.get("turn_bound") or {}
    prompt = render_prompt(entry, row)
    out: list[str] = []
    for token in entry.get("invoke_argv") or []:
        if token in ("<prompt>", "<args>"):
            out.append(prompt)
            continue
        if token == "<allowed-tools>":
            # SEPARATE argv tokens, and this flag must therefore be LAST —
            # `--allowedTools` is variadic and would otherwise swallow what
            # followed. MEASURED 2026-08-14: `--help` says "Comma or
            # space-separated", and a single comma-joined token IS accepted, but
            # every rule containing a space (`Bash(command -v uv)`,
            # `Bash(PYTHONUNBUFFERED=1 uv:*)`) is then silently useless — and it
            # takes the rest of the list down with it, `Read` included. The row
            # that found this loaded the skill (space-free `Skill` survived) and
            # had every single command in it denied. Separate tokens ran first
            # time.
            out.extend(["Skill", "Skill(multi-llm)", *_skill_allowed_tools()])
            continue
        # Substring substitution, not equality: `--model openrouter/<model>` is a
        # real cell (opencode namespaces a slug by its provider id).
        out.append(token.replace("<model>", slug)
                        .replace("<N>", str(bound.get("value", 12))))
    if not out:
        raise SystemExit("entrypoint: the matrix carries no invoke_argv for this host")
    return out


def render_prompt(entry: dict, row: dict) -> str:
    """The prompt (or, on opencode, the argument string) — from the matrix.

    An improvised prompt is an untracked variable: a row that fails must be
    re-runnable byte-for-byte, and a prompt edit must show up as a diff.
    """
    args = row.get("args")
    if not args:
        raise SystemExit(f"entrypoint: row {row.get('id')!r} declares no `args`")
    template = entry.get("prompt_template")
    return args if not template else template.replace("{args}", args).strip()


# --------------------------------------------------------------------------- #
# The host agent's OWN session transcript
#
# `--output-format json` prints the final envelope and nothing else. A row where
# the agent loaded the skill and stopped before its first `uv run` produces zero
# dispatcher receipts, `result: ""`, `subtype: "success"` and ~100 output tokens
# — and no evidence whatsoever of WHERE it stopped. Everything the agent does
# before its first dispatcher call (`Skill(multi-llm)`, `Read`,
# `Bash(command -v uv)`, and any denial of those) leaves no receipt, no
# `calls.jsonl` entry and no line on stdout. Four claude-code rows have flaked
# across four live runs and two of them had exactly that shape; the stopping
# point was unrecoverable, so the cause was guessed at rather than read.
#
# Claude Code writes that missing evidence to
# `$HOME/.claude/projects/<slugified cwd>/<session_id>.jsonl`, and the envelope
# already carries `session_id`. The tmpfs `$HOME` dies with the container, so the
# file is lifted into the bundle at teardown or it is gone.
#
# MEASURED 2026-09-22, claude-code 2.1.278 — the version cli-versions.lock pins —
# driven against a localhost Anthropic stub with `HOME` pointed at an empty
# directory and `ANTHROPIC_API_KEY` present-and-empty, i.e. the host lane's own
# wiring at zero spend:
#
#   * the envelope's `session_id` is exactly the file's stem;
#   * the directory is the cwd with every non-alphanumeric run replaced by `-`;
#   * a ONE-TURN session is already 191 KB, 168 KB of which is two
#     `prompt_snapshot` attachments.
#
# WHAT IS VERIFIED OFFLINE IS THE LAYOUT, NOT THE CONTAINER. That a tmpfs `$HOME`
# behaves like any other directory here is inference, and a CLI upgrade may move
# the path — which is why every branch below ends in a logged reason instead of
# an exception, and why the digest names the file it actually read.
#
# WHAT LANDS IN THE BUNDLE IS A DIGEST, NOT THE FILE. See `capture_host_session`.
# --------------------------------------------------------------------------- #

#: Free text kept per field before it becomes a character count. A stalled turn
#: is ~100 output tokens, so the whole of one fits; so does a permission-denial
#: message, which is the most likely cause of an agent that stops early.
SESSION_TEXT_CAP = 2000

#: The smaller cap for a tool RESULT that is not an error. Tool results are the
#: bulk of a transcript (a file the agent read) and the least diagnostic part of
#: it; an error one is kept at the full cap because that is where a denial lands.
SESSION_RESULT_CAP = 400

#: Total digest budget. A green `apply-native` row touched 2.3M cache-read
#: tokens; the source JSONL is unbounded and a bundle is not.
SESSION_DIGEST_CAP = 2 * 1024 * 1024

#: Source lines read, as a second bound that does not depend on line length.
SESSION_LINE_CAP = 20_000

#: Wall-clock budget. This runs at teardown, and on the `docker stop` path the
#: grace period is ten seconds — a capture that ate it would cost the artifact
#: export it is filed beside, which is the more important artifact. A digest
#: that stopped on this cap still says where the agent got to.
SESSION_SECONDS_CAP = 5.0

#: `{host: ($HOME-relative path of the transcript root)}`. Only claude-code is
#: measured; a host with no entry is a logged skip, never a failure. Written as a
#: table so adding a host is one line with its layout beside it, rather than a
#: second `if host == ...` somewhere else in this file.
SESSION_LAYOUTS = {"claude-code": (".claude", "projects")}


#: The LEFT-BOUNDARY backstop, and the one thing this file redacts that the
#: product's table does not.
#:
#: MEASURED 2026-09-22 against a real 5.7 MB claude-code transcript on this
#: machine: an `OPENAI_API_KEY` survived `write_safety.redact_text` intact
#: because the tool result it sat in read
#:
#:     OPENAI_API_KEY: SETsk-proj-XXXXXXXXXXXX…
#:
#: — a probe script that printed `SET` and the value with nothing between them.
#: Every vendor-prefix pattern in `_REDACTIONS` is anchored with a leading `\b`,
#: and `T` followed by `s` is not a word boundary, so none of them fired. The
#: shape is not exotic: agents concatenate, and a transcript is full of other
#: programs' output.
#:
#: Only the prefixes whose INFIX is itself distinctive are swept without a left
#: boundary. `sk-` plus sixteen word characters is deliberately NOT here — with
#: the boundary dropped it redacts `task-oriented-development` — but `sk-proj-`,
#: `ghp_`, `glpat-` and the rest cannot occur inside an English word.
#:
#: This is a backstop, not a second source of truth: the product's table stays
#: the redactor, and the gap above belongs in `utils/write_safety.py`, where
#: fixing it would protect every other consumer too.
_SESSION_PREFIX_SWEEP = re.compile(
    r"sk-(?:ant|proj|or-v1|live)-[A-Za-z0-9_\-]{16,}"
    r"|xox[baprse]-[A-Za-z0-9\-]{16,}"
    r"|gh[opusr]_[A-Za-z0-9]{16,}"
    r"|AIza[0-9A-Za-z_\-]{35,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
)


def _sweep_prefixes(text: str) -> str:
    """`_SESSION_PREFIX_SWEEP`, applied after the product redactor has run."""
    return _SESSION_PREFIX_SWEEP.sub("<redacted:prefix-sweep>", text or "")


def _session_redactor():
    """The PRODUCT's credential/identity redactor, or `None` — and then we stop.

    `utils.write_safety.redact_text` is the shipped one: vendor key prefixes
    (`sk-ant-`/`sk-proj-`/`sk-or-v1-`, `ghp_`, `xox…`, `AIza…`, `glpat-`, `hf_`,
    AWS ids), JWTs, basic-auth URLs, PEM private keys, `Authorization:` headers,
    the predicate-gated `<secret-named key>: <value>` branch, and the identity
    fields. Reached through the same `sys.path` insert `opencode_plugin_spec()`
    already uses, and for the same reason: `sys.path[0]` is the rig directory,
    so a bare `from utils import ...` cannot work here.

    IT FAILS CLOSED. If the import does not work — a rig driving an image that
    predates the skill install, a future layout change — no digest is written at
    all. A bundle is a shareable artifact, and MEASURED on this machine's own
    session files: real transcripts do contain `sk-or-v1-…` and `sk-proj-…`
    values that the agent read out of a file or printed from an environment. The
    per-row `_scrub` covers the one key THIS row was handed; it cannot cover a
    credential the transcript picked up from somewhere else, and duplicating the
    product's pattern table into the rig would make this repo hold two of them.
    """
    if str(SKILL_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_DIR))
    try:
        from utils.write_safety import redact_text
    except ImportError as exc:  # pragma: no cover - the rig runs outside the skill too
        log(f"session transcript: the skill's redactor is unreachable ({exc})")
        return None
    return redact_text


def session_id_from_envelope(stdout_text: str) -> str:
    """`session_id` out of `--output-format json`'s final envelope, or `""`.

    Tried whole-document first and then line by line from the end, so a
    pretty-printed envelope and a stream of JSON lines both resolve, and a host
    that printed prose before its envelope does not defeat it.
    """
    text = (stdout_text or "").strip()
    if not text:
        return ""
    for chunk in [text] + [line.strip() for line in reversed(text.splitlines())]:
        if not chunk.startswith("{"):
            continue
        try:
            payload = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("session_id"):
            return str(payload["session_id"])
    return ""


def find_host_session(host: str, home: Path, session_id: str = "") -> tuple:
    """`(path or None, siblings)` — this row's session JSONL under its `$HOME`.

    The id is the exact answer and is what a completed row has. A row that was
    killed before its envelope reached stdout has no id, and then the NEWEST
    transcript is the best available guess — recorded as such in the metadata,
    because "we guessed" and "we matched" are different claims. `siblings` is how
    many other transcripts were sitting there, so a bundle never silently drops
    one this function did not pick.
    """
    parts = SESSION_LAYOUTS.get(host)
    if not parts:
        return None, 0
    root = home.joinpath(*parts)
    if not root.is_dir():
        return None, 0
    found = [p for p in sorted(root.glob("*/*.jsonl")) if p.is_file()]
    if not found:
        return None, 0
    named = [p for p in found if session_id and p.stem == session_id]
    if named:
        return named[0], len(found) - 1
    return max(found, key=lambda p: p.stat().st_mtime), len(found) - 1


def _clip(value, cap: int = SESSION_TEXT_CAP) -> str:
    """A string kept whole, or its head plus a count of what was dropped."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= cap:
        return text
    return text[:cap] + f"...<+{len(text) - cap} chars>"


def _digest_message(message: dict) -> dict:
    """The parts of one API message that say where a turn went.

    Structure is always kept; free text is kept up to a cap and otherwise
    becomes a length. The one asymmetry is deliberate: a `tool_use` input is
    kept generously because it IS the diagnosis — `Skill(multi-llm)` followed by
    nothing is the "never started" shape, and `Bash(command -v uv)` followed by
    an error is a different row entirely — while a non-error `tool_result` is a
    file the agent read, which is most of the bytes and none of the answer.
    """
    out: dict = {"role": message.get("role"),
                 "model": message.get("model"),
                 "stop_reason": message.get("stop_reason")}
    usage = message.get("usage")
    if isinstance(usage, dict):
        out["usage"] = {name: usage[name] for name in (
            "input_tokens", "output_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens") if name in usage}
    content = message.get("content")
    if isinstance(content, str):
        out["chars"] = len(content)
        out["preview"] = _clip(content)
        return out
    if not isinstance(content, list):
        return out
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": type(block).__name__})
            continue
        kind = block.get("type")
        item: dict = {"type": kind}
        if kind in ("text", "thinking", "redacted_thinking"):
            body = block.get("text") or block.get("thinking") or ""
            item["chars"] = len(body)
            item["preview"] = _clip(body)
        elif kind == "tool_use":
            item["name"] = block.get("name")
            item["id"] = block.get("id")
            item["input"] = _clip(block.get("input"))
        elif kind == "tool_result":
            item["tool_use_id"] = block.get("tool_use_id")
            item["is_error"] = bool(block.get("is_error"))
            body = block.get("content")
            text = body if isinstance(body, str) else json.dumps(body, default=str)
            item["chars"] = len(text)
            if item["is_error"] or len(text) <= SESSION_RESULT_CAP:
                item["preview"] = _clip(text, SESSION_TEXT_CAP)
        else:
            item["chars"] = len(json.dumps(block, default=str))
        blocks.append(item)
    out["blocks"] = blocks
    return out


def _digest_record(index: int, record: dict, size: int) -> dict:
    """One transcript line, reduced to what diagnoses a stall."""
    out: dict = {"i": index, "bytes": size, "type": record.get("type")}
    for name in ("timestamp", "uuid", "parentUuid", "isSidechain", "subtype",
                 "level", "toolUseID", "stopReason", "isMeta", "requestId",
                 "durationMs", "preventedContinuation", "hookErrors"):
        if name in record:
            out[name] = record[name]
    known = False
    message = record.get("message")
    if isinstance(message, dict):
        out["message"] = _digest_message(message)
        known = True
    attachment = record.get("attachment")
    if isinstance(attachment, dict):
        # The bulk of the file and the least diagnostic part of it: two
        # `prompt_snapshot` attachments were 168 KB of the 191 KB one-turn
        # session measured above. Type and size always; the payload only while
        # it is small enough to be read rather than skimmed, which keeps the
        # `environment`, `model`, `date` and `session_context` ones whole and
        # drops the snapshots and the listings.
        body = json.dumps(attachment, default=str)
        out["attachment"] = {"type": attachment.get("type"), "chars": len(body)}
        if len(body) <= SESSION_TEXT_CAP:
            out["attachment"]["payload"] = attachment
        known = True
    content = record.get("content")
    if isinstance(content, str):
        out["content"] = _clip(content)
        known = True
    if record.get("type") == "cost-state":
        for name in ("totalCostUSD", "totalDuration", "totalAPIDuration",
                     "totalToolDuration", "totalLinesAdded", "totalLinesRemoved",
                     "messageCount"):
            if name in record:
                out[name] = record[name]
        known = True
    if not known:
        # A shape this rig does not model — a new record type after a CLI
        # upgrade — is still NAMED, so the next reader knows something was there
        # and what its fields were called.
        out["keys"] = sorted(record)
    return out


def capture_host_session(host: str, home: Path, stdout_text: str,
                         scrub, session_id: str = "") -> dict:
    """Lift the host agent's session transcript into the bundle, as a digest.

    Returns the metadata that goes into `host-run.json` under `session`, and
    **never raises**: it is called from a `finally` that a `docker stop` reaches
    with a short grace period left, and the artifact export runs after it. A
    missing file, an unreadable one, a CLI that moved the path, a transcript
    whose schema changed — each is a logged `reason` and a `captured: false`,
    and the row goes on.

    A DIGEST, NOT THE FILE, AND NOT A GZIP OF IT. Both were on the table:

    * `host-session-digest.jsonl` is newline-delimited JSON, one line per source
      record, headed by a `_capture` line naming the source, its size, the
      session id and whether that id MATCHED or was guessed. It is greppable and
      diffable by hand next to the rest of the bundle. MEASURED: 191 KB -> 7 KB
      on the one-turn session above (the two `prompt_snapshot` attachments
      become a type and a length), and 5.7 MB / 2 548 records -> 1.5 MB in 0.33 s
      on a real long agentic session.
    * A gzip of the raw file would lose nothing, and that is the problem: it
      would carry the whole system prompt, every file the agent read and every
      command's full output into an artifact that gets handed to other people,
      as a binary blob nobody can eyeball before sharing it.

    WHAT IS LOST: the tail of any free-text field past `SESSION_TEXT_CAP`, the
    body of every non-error tool result over `SESSION_RESULT_CAP` (so "what did
    that `Read` actually return" is answerable only by its length), the payload
    of every large attachment, and the whole of the transcript past whichever cap
    fired first — which the metadata names. The stall shape this exists for —
    which tool the agent reached, what it passed, whether the result was an
    error, what it then said, and the `stop_reason` it stopped on — survives all
    of them.
    """
    meta: dict = {"captured": False, "host": host}
    try:
        wanted = session_id or session_id_from_envelope(stdout_text)
        meta["session_id"] = wanted
        source, siblings = find_host_session(host, home, wanted)
        if source is None:
            meta["reason"] = (
                f"no transcript under {home.joinpath(*SESSION_LAYOUTS[host])}"
                if host in SESSION_LAYOUTS
                else f"no known session layout for host {host!r}")
            log(f"session transcript: {meta['reason']}")
            return meta
        meta.update({
            "source": str(source),
            "source_bytes": source.stat().st_size,
            "siblings": siblings,
            "matched_by": "session_id" if source.stem == wanted else "newest",
        })
        redact = _session_redactor()
        if redact is None:
            meta["reason"] = ("no redactor: an unredacted transcript does not "
                              "go into a shareable bundle")
            return meta

        target = OUT / "host-session-digest.jsonl"
        deadline = time.time() + SESSION_SECONDS_CAP
        written = records = unparsed = 0
        stopped = ""

        def _emit(sink, payload: dict) -> int:
            # Scrub by VALUE first (this row's own key, and on the fallback path
            # the copied credential's strings), then by PATTERN (the product's
            # own table), then by the left-boundary backstop above. All three
            # run on the SERIALIZED line, so nothing reaches the file
            # unscrubbed, and no replacement contains a quote or a backslash, so
            # the line stays valid JSON.
            line = _sweep_prefixes(
                redact(scrub(json.dumps(payload, default=str, sort_keys=True))))
            sink.write(line + "\n")
            return len(line) + 1

        with source.open("r", encoding="utf-8", errors="replace") as handle, \
                target.open("w", encoding="utf-8") as sink:
            written += _emit(sink, {
                "i": -1, "type": "_capture", "source": str(source),
                "source_bytes": meta["source_bytes"], "session_id": wanted,
                "matched_by": meta["matched_by"], "siblings": siblings,
                "caps": {"text": SESSION_TEXT_CAP, "result": SESSION_RESULT_CAP,
                         "digest_bytes": SESSION_DIGEST_CAP,
                         "lines": SESSION_LINE_CAP,
                         "seconds": SESSION_SECONDS_CAP},
            })
            for index, raw in enumerate(handle):
                if index >= SESSION_LINE_CAP:
                    stopped = "line-cap"
                    break
                if written >= SESSION_DIGEST_CAP:
                    stopped = "byte-cap"
                    break
                if time.time() > deadline:
                    stopped = "time-cap"
                    break
                # VALUE-scrubbed before the parse, not only on the way out:
                # `_clip` cuts free text at a fixed offset, and a secret that
                # straddles that offset leaves a head an exact-value replace can
                # no longer match. A `str.replace` over the raw line is two
                # passes and closes it; the pattern layer still runs on the
                # (much smaller) serialized digest line, where a clipped vendor
                # prefix is still long enough to match.
                raw = scrub(raw.strip())
                if not raw:
                    continue
                records += 1
                try:
                    record = json.loads(raw)
                except ValueError:
                    # A transcript read while the host is still writing it — the
                    # `docker stop` path — ends in half a line. That is one
                    # record, not the whole capture.
                    unparsed += 1
                    record = None
                written += _emit(sink, _digest_record(index, record, len(raw))
                                 if isinstance(record, dict)
                                 else {"i": index, "bytes": len(raw),
                                       "type": "_unparsed",
                                       "preview": _clip(raw)})
        meta.update({"captured": True, "digest": target.name,
                     "digest_bytes": written, "records": records,
                     "unparsed": unparsed, "stopped_early": stopped or None})
        log(f"session transcript: {records} records of {meta['source_bytes']}B "
            f"from {source.name} -> {target.name} ({written}B"
            + (f", stopped on {stopped}" if stopped else "") + ")")
    except Exception as exc:  # evidence, never a failure path
        meta["reason"] = f"{type(exc).__name__}: {exc}"
        log(f"could not capture the host session transcript: {exc}")
    return meta


def run_host(host: str, entry: dict, row: dict, home: Path, overlay: dict,
             key: str, creds: str = "openrouter", secrets=()) -> dict:
    """Launch the host itself and capture everything it said.

    **By absolute path, never through `/opt/mllm/bin`.** `PATH` stays
    wrapper-first for the whole process tree — that is what keeps every provider
    subprocess the orchestrator spawns on the stub — and the host is exempted by
    being exec'd at its real path rather than by anything in its environment.

    `creds` names which credential mode wired this row, because two things differ
    by it and nothing else does: the ANTHROPIC_API_KEY assertion (an
    OpenRouter-only guard — on the fallback path the variable must be *absent*,
    since the copied `.credentials.json` is the auth) and which values the
    transcript is scrubbed of.

    **"Everything it said" now includes what it said to itself.** `host-stdout`
    is one envelope; the agent's own session transcript — every tool it reached
    for and every result it got back, including the ones before its first
    dispatcher call — is lifted into `host-session-digest.jsonl` by
    `capture_host_session`, in a `finally` so the unfinished row is the one it
    serves.
    """
    argv = build_host_argv(entry, row)
    binary = argv[0]
    argv[0] = str(REAL_DIR / binary)
    if not Path(argv[0]).exists():
        raise SystemExit(f"entrypoint: no real host binary at {argv[0]}")

    env = dict(os.environ)
    env["HOME"] = str(home)
    env.update(overlay)
    for var, value in (entry.get("env_extra") or {}).items():
        env[str(var)] = str(value)
    if host == "claude-code" and creds == "openrouter":
        assert_claude_key_is_empty(env)
    elif host == "claude-code" and env.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "entrypoint: --creds fallback runs claude-code on the copied "
            "~/.claude/.credentials.json, and ANTHROPIC_API_KEY is set — this row "
            "would authenticate against whatever that key belongs to instead.")

    bound = entry.get("turn_bound") or {}
    wall = int(row.get("wall_clock_s") or bound.get("wall_clock_s") or
               (bound.get("value") if bound.get("kind") == "wall-clock" else 0) or 900)

    log(f"host launch: {' '.join(shlex.quote(a) for a in [binary, *argv[1:]])}")
    log(f"host wall-clock bound: {wall}s")

    # Redaction at WRITE time, on a stream nobody audited: the host's own
    # transcript is third-party output and this lane's secrets are in its
    # environment (openrouter) or in the files copied into its $HOME (fallback).
    # Cheap insurance against a CLI that echoes its config.
    #
    # DEFINED BEFORE THE LAUNCH because the session capture below runs in a
    # `finally` that is reached on the path where the host never returns, and a
    # scrub defined after the wait would not exist there.
    def _scrub(text: str) -> str:
        text = text or ""
        for secret in ([key] if key else []) + list(secrets):
            text = text.replace(secret, f"<redacted:{creds}-credential>")
        return text

    started = time.time()
    timed_out = False
    out = err = ""
    code = None
    proc = subprocess.Popen(
        argv, env=env, cwd=str(WORKSPACE), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)          # its own group, so a cap hit kills the tree
    try:
        try:
            out, err = proc.communicate(timeout=wall)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                out, err = proc.communicate(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                out, err = proc.communicate()
            code = proc.returncode
    finally:
        # IN A `finally`, for the same reason the artifact export is: the row
        # whose evidence matters most is the one that did not finish. A
        # `docker stop` unwinds PID 1 through `SystemExit` raised from the
        # SIGTERM handler — `out` is never assigned, so the envelope carrying
        # `session_id` is lost with it and `host-stdout.txt` below is never
        # written — and the transcript on disk is then the only thing left that
        # says where the agent stopped. It is time-capped so it cannot eat the
        # grace period the export still needs.
        session = capture_host_session(host, home, out, _scrub)

    (OUT / "host-stdout.txt").write_text(_scrub(out), encoding="utf-8")
    (OUT / "host-stderr.txt").write_text(_scrub(err), encoding="utf-8")
    record = {
        "host": host,
        "row": row.get("id"),
        "argv": [binary, *argv[1:]],
        "cwd": str(WORKSPACE),
        "credential_mode": creds,
        "exit": code,
        "timed_out": timed_out,
        "wall_clock_s": round(time.time() - started, 2),
        "bound_s": wall,
        # Where the agent's own transcript landed, or why it did not. A
        # `captured: false` with a reason is itself a finding on a row that
        # produced no receipts: it separates "the agent stopped early" from
        # "we could not see what the agent did".
        "session": session,
    }
    (OUT / "host-run.json").write_text(json.dumps(record, indent=2, sort_keys=True),
                                       encoding="utf-8")
    log(f"host exited {code} after {record['wall_clock_s']}s"
        + (" (WALL-CLOCK KILL)" if timed_out else ""))
    return record


def poll_terminal_run(plan: Path, timeout: float = 180.0) -> dict:
    """Wait for a detached worker's `.status.json` to reach a terminal run state.

    A `--detach` invocation returns 0 with `state: done` **before the workflow has
    run**, so on a detaching row the verdict comes from the worker's own `run`
    block. Grading before it settles would grade a launch.

    ONLY the status files that carry a `run` block are runs. Every phase writes
    `{phase}/.status.json` whether or not it detached (`status_file.py` is the
    one writer, and `detach.py`: "a foreground run's `.status.json` gains no new
    keys"), so a `--full` row's plan dir holds the detached `full/` run's file
    PLUS one checkpoint per in-process phase. Counting the checkpoints as runs
    made this loop wait out its whole timeout for a `state` that is structurally
    `None`, and then hand the grader phantom rows to fail a passing row on.
    """
    root = plan.parent / plan.stem
    deadline = time.time() + timeout
    seen: dict = {}
    while time.time() < deadline:
        seen = {}
        # rglob, not glob: `ask` nests one level deeper than every other phase
        # ({plan}/ask/<question-slug>-<hash8>/.status.json), so a one-level glob
        # reports "no run recorded" on precisely the cheapest row in the matrix.
        for status in sorted(root.rglob(".status.json")):
            try:
                document = json.loads(status.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            block = document.get("run") if isinstance(document, dict) else None
            if not isinstance(block, dict):
                # A phase checkpoint, not a run. Skipped rather than recorded as
                # `state: None`: absence of a run block here is not evidence of a
                # run that failed to reach a terminal state, and a row whose
                # detached worker never claimed one still ends with `seen` empty,
                # which the grader already reports as "graded on a launch".
                continue
            key = str(status.parent.relative_to(root))
            seen[key] = {k: block.get(k) for k in
                         ("run_id", "state", "exit_code")}
        states = [v.get("state") for v in seen.values()]
        if states and all(s in ("done", "failed", "aborted") for s in states):
            break
        time.sleep(2)
    (OUT / "run-states.json").write_text(
        json.dumps(seen, indent=2, sort_keys=True), encoding="utf-8")
    log(f"run states after poll: {seen}")
    return seen


def dispatch(argv: list[str], env: dict | None = None, cwd: Path | None = None) -> int:
    """Run one dispatcher command through the `uv` wrapper, which writes receipts."""
    command = ["uv", "run", "--project", str(SKILL_DIR), "--",
               "python", str(SKILL_DIR / "multi_llm.py"), *argv]
    log("dispatch: " + " ".join(argv))
    return subprocess.call(command, env=env or os.environ.copy(),
                           cwd=str(cwd or WORKSPACE))


# --------------------------------------------------------------------------- #
# The edit-capable fake (C4 option (a)) — the applied-bytes row only
# --------------------------------------------------------------------------- #

#: Where the shadowing launcher lands. A tmpfs (`posture.yaml`), because the root
#: filesystem is read-only and because nothing about this row may outlive it.
MOCK_BIN = Path("/tmp/mllm-mock-bin")
MOCK_LLM = SKILL_DIR / "tests" / "mocks" / "mock_llm.py"

#: Per-command environment: a leading `NAME=value` token, and nothing looser. A
#: dispatcher argv is always `<mode> …` with the mode word as argv[1], so a
#: leading assignment can never be a command of ours.
ENV_ASSIGNMENT = re.compile(r"^[A-Z][A-Z0-9_]*=")


def install_mock_runner(binary: str, scenario: Path | None) -> Path:
    """Shadow one binary name with `tests/mocks/mock_llm.py`, FIRST on PATH.

    This is the whole mechanism behind the applied-bytes row, and it is
    deliberately the narrowest one available:

    * **a symlink, so `argv[0]` still says `claude`.** The mock resolves which
      provider it is emulating from `Path(sys.argv[0]).name`, and its own
      `__file__` resolution then finds `scenario_matcher.py` two directories up
      from the *real* file — which is why a symlink works and a copy would not.
    * **PATH-first rather than replacing anything.** `/opt/mllm/bin` stays where
      it is, so every other binary the row runs is still the genuine CLI behind
      C2's wrapper; only the one name this row names is shadowed. It also means
      the mock is reached *without* the wrapper, which is required rather than
      incidental: the wrapper rebuilds its child's environment from scratch and
      carries over only `MLLM_*`, so a `MOCK_LLM_EDITS` routed through it would
      arrive unset and the row would record a clean failure it did not mean.
    * **one row, one name.** Nothing else in the image can move a byte under the
      executor, so nothing else can turn a chain row's declared `clean-failure`
      into an accidental `applied`.
    """
    if not MOCK_LLM.is_file():
        raise SystemExit(
            f"entrypoint: --mock-runner needs the edit-capable fake at {MOCK_LLM} "
            f"and it is not in this image. It is the only thing here that can "
            f"produce a real byte change under the executor, so the applied-bytes "
            f"row cannot fall back to a redirected CLI: that would silently "
            f"become a second clean-failure row.")
    MOCK_BIN.mkdir(parents=True, exist_ok=True)
    launcher = MOCK_BIN / binary
    if launcher.is_symlink() or launcher.exists():
        launcher.unlink()
    launcher.symlink_to(MOCK_LLM)
    os.environ["PATH"] = f"{MOCK_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    # The mock refuses to run without it, and the product never reads it.
    os.environ["MULTI_LLM_TEST_MODE"] = "1"
    if scenario:
        # The SAME scenario the stub resolves, through the same matcher: the read
        # dispatches this row makes must be answered from the fixtures the chain
        # already pins, not from the mock's generic default.
        os.environ["MOCK_LLM_SCENARIO"] = str(scenario)
    log(f"mock runner installed: {launcher} -> {MOCK_LLM} (PATH-first)")
    return launcher


def split_command_env(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    """Peel leading `NAME=value` assignments off one shell-split command.

    The applied-bytes row needs `MOCK_LLM_EDITS` set for its *apply* invocation
    and unset for the review invocation that precedes it — the mock applies its
    edits on **every** dispatch, so a read task run with the variable set would
    both move bytes outside a batch and answer a validation prompt with a write
    result. One process-wide export cannot express that; a per-command prefix
    can, and it reads exactly like the shell form it imitates.
    """
    env: dict[str, str] = {}
    index = 0
    for index, token in enumerate(tokens):  # noqa: B007 - index is the result
        if not ENV_ASSIGNMENT.match(token):
            return env, tokens[index:]
        key, _, value = token.partition("=")
        env[key] = value
    return env, tokens[index + 1:] if tokens else []


def export_artifacts(plan: Path) -> None:
    """Copy the plan's output dir to /work/out (the LAST step, in a finally)."""
    source = plan.parent / plan.stem
    target = OUT / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target / source.name, dirs_exist_ok=True)
        log(f"exported {source} -> {target / source.name}")
    else:
        log(f"nothing to export: {source} does not exist")
    export_receipt_named_plan_files(plan, target)
    (OUT / "workspace-census-after.json").write_text(
        json.dumps(_workspace_census(), indent=2, sort_keys=True),
        encoding="utf-8")


def export_receipt_named_plan_files(plan: Path, target: Path) -> None:
    """Also export the `plans/` files this row's own RECEIPTS named.

    The copy above takes the plan's output DIRECTORY, which is where every
    phase artifact lands — but not every artifact a receipt names lives there.
    `apply-suggestions` rewrites the plan FILE and then names it, so a receipt
    carrying `/work/sample-project/plans/auth-feature.md` is the row's own
    claim that it edited the plan. That file sits one level ABOVE the exported
    directory and therefore never reached the bundle at all, which is how
    `test_container_host_lane.py`'s `TestTheAgentReallyRanOurDispatcher::
    test_every_named_artifact_exists_and_is_non_empty` came to fail three rows
    on the export rather than on the artifact. It only
    became reachable once those rows stopped dying at `container exit 3`: a row
    that never reached `apply-suggestions` never named the plan.

    THE RULE IS "WHAT THE RECEIPTS NAMED", never "everything under plans/",
    and the narrowness is the point:

    * A bundle is read and diffed by hand, and it already carries one secret
      (`docker-inspect.json` has the OpenRouter key), so what goes into it is
      chosen rather than swept. Exporting the plans directory wholesale would
      pick up files nobody declared — `plans/auth-feature.ask.log`, a host
      CLI's own scratch log, is in a recorded census and no receipt names it.
    * `artifacts` is OUR dispatcher's declaration, not the host's, and this
      reads the very directory the grader reads: `$MLLM_RECEIPT_DIR` is
      `OUT/receipts/<row>`, which is `harness.RowResult.receipts` exactly. The
      exported set and the graded set cannot drift apart.
    * It is transport, never a verdict. A named file that does not exist is
      not copied and the row still fails on the absent artifact — which is the
      failure that assertion is for.

    Placement mirrors the test's `_exported` map (`…/plans/<tail>` ->
    `artifacts/<tail>`), so the plan file lands as a FILE at the artifacts
    root. Every existing reader of that root selects directories — golden
    `container_bundle.output_root`, the test's `_exported_before_host`, and
    `harness.phase_status`/`workflow_state`'s `*/…` globs — so no bundle a
    consumer already walks changes shape.

    Nothing here may raise: the caller runs in `main`'s `finally`, one
    statement before the after-census is written, and on a capped row it is
    reached from the SIGTERM handler with a short grace period left.
    """
    receipt_dir = os.environ.get("MLLM_RECEIPT_DIR", "")
    if not receipt_dir or not Path(receipt_dir).is_dir():
        log("no receipt directory: nothing named to export beside the output dir")
        return
    plans_root = plan.parent
    try:
        resolved_root = plans_root.resolve()
    except OSError as exc:  # evidence, never a failure path
        log(f"could not resolve {plans_root} for the named export: {exc}")
        return

    named: list[str] = []
    for receipt in sorted(Path(receipt_dir).glob("*.receipt.json")):
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8")).get("result")
        except (OSError, ValueError) as exc:
            # One unreadable receipt — a row killed mid-write — costs its own
            # artifacts and never the whole export.
            log(f"could not read {receipt.name} for the named export: {exc}")
            continue
        if isinstance(payload, dict):
            named.extend(str(a) for a in (payload.get("artifacts") or []))

    for name in sorted(set(named)):
        artifact = Path(name)
        # The LEXICAL test picks the ones the bundle has a place for and yields
        # the tail `_exported` expects; `$HOME` and `/opt/mllm` paths — the two
        # places a credential could be — are dropped here and never read.
        if not artifact.is_absolute() or not artifact.is_relative_to(plans_root):
            continue
        destination = target / artifact.relative_to(plans_root)
        if destination.exists():
            continue          # the directory copy above already carries it
        try:
            # And the RESOLVED test is containment: a `..` segment or a symlink
            # pointing out of the workspace passes the lexical one.
            if not artifact.resolve().is_relative_to(resolved_root):
                log(f"named artifact {name} resolves outside {plans_root}: not exported")
                continue
            if not artifact.is_file():
                continue      # absent is the grader's finding, not ours to hide
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, destination)
        except OSError as exc:
            log(f"could not export the named artifact {name}: {exc}")
            continue
        log(f"exported named artifact {artifact} -> {destination}")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_census() -> dict:
    """`{relative path: sha256}` for every file in the workspace, `.git` aside."""
    return {
        str(p.relative_to(WORKSPACE)): _sha(p)
        for p in sorted(WORKSPACE.rglob("*"))
        if p.is_file() and ".git/" not in str(p.relative_to(WORKSPACE))
    }


def write_host_census(when: str) -> None:
    """Bracket the HOST's own execution, in addition to the existing pair.

    `workspace-census-before.json` is written right after `seed_workspace.py`
    and **before** `run_pre_commands`. On a row whose earlier phases are seeded
    by a pre-command leg — six of the eight claude-code phase rows, the two
    exceptions being `review-plan` and `code-review`, which have no
    prerequisites — every byte the leg writes therefore lands inside the
    measured delta. Those are precisely the rows where it decides the verdict:
    the four bytes-graded write rows.

    So this pair brackets the host process alone. The existing pair is left
    exactly where it is, so nothing already graded moves; a row that wants the
    host's own credit reads these two instead.
    """
    try:
        (OUT / f"workspace-census-host-{when}.json").write_text(
            json.dumps(_workspace_census(), indent=2, sort_keys=True),
            encoding="utf-8")
    except OSError as exc:  # evidence, never a failure path
        log(f"could not write the host-bracketed census ({when}): {exc}")


def install_termination_handler() -> None:
    """Make the harness's `docker stop` unwind this process (C1's cap design).

    This script is PID 1 — exec-form ENTRYPOINT, no `--init` — and the kernel
    gives PID 1 no default disposition: a signal whose handler is not
    *explicitly installed* is discarded, not defaulted. Without this, the
    `docker stop --time 10` the harness sends on a row timeout was ignored
    outright, docker escalated to SIGKILL when the grace period ran out, and the
    two teardown paths that make a capped row readable never ran at all: the
    ``finally: export_artifacts(plan)`` below, and the stub's context-manager
    exit, which is what drains in-flight requests and flushes `stub.jsonl`. A
    capped row is exactly the row whose evidence has to survive — the grace
    period buys nothing if nobody is listening for the signal that starts it.

    Raising `SystemExit` from the handler unwinds through both. The handler
    disarms itself on entry: the grace window is short and a second SIGTERM
    landing mid-export would cut the copy in half, which is the failure this is
    here to prevent. Whatever is still running below us dies with the container
    the moment PID 1 exits, so the export races nothing that outlives it.
    """

    def _terminate(signum: int, _frame) -> None:
        signal.signal(signum, signal.SIG_IGN)
        log(f"signal {signum} received — unwinding so the artifact export runs")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    # First, before anything can block: as PID 1 an uninstalled SIGTERM handler
    # means `docker stop` is a no-op and every teardown below is lost to SIGKILL.
    install_termination_handler()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["provider", "host"], required=True)
    parser.add_argument("--row", required=True, help="row id, also $MLLM_ROW_ID")
    parser.add_argument("--cli", default=None, help="provider lane: the row's CLI")
    parser.add_argument("--host", default=None, help="host lane: the host id")
    parser.add_argument("--creds", choices=["openrouter", "fallback", "none"],
                        default="none",
                        help="host lane credential tier: `openrouter` (the "
                             "default path — one env var, no mounts), `fallback` "
                             "(C6a's subscription/vendor auth: the allowlisted "
                             "`:ro` mounts copied into the tmpfs $HOME), or "
                             "`none` (the free discovery tier)")
    parser.add_argument("--helpers", default="",
                        help="comma-separated allowed_helpers for this row")
    # Repeatable: the row's fan-out is a LIST of provider:model specs, and the
    # seeder writes every one of them into every selection key. One `--spec`
    # is a fan-out of one; no `--spec` at all is an error, not a default (see
    # the seeder call below).
    parser.add_argument("--spec", action="append", default=None,
                        help="one provider:model spec in the row's fan-out; "
                             "repeatable, order preserved")
    # Singular by construction, unlike `--spec`: `driver.subagent_runner` is
    # one slug. REQUIRED, like `--spec` and like the seeder's own `--runner` —
    # there is no defensible default (see the seeder call below).
    parser.add_argument("--runner", default=None,
                        help="executor runner spec (single; required — the "
                             "row's write runner, named explicitly)")
    # Repeatable, and shell-split rather than whitespace-split: a chain row is
    # more than one dispatcher invocation (`full`, then a standalone `ask`, then
    # a `status` read-back), and `ask` carries a quoted question that a bare
    # `str.split()` would tear into five argv words.
    parser.add_argument("--command", action="append", default=[],
                        dest="commands", help="dispatcher argv, shell-split; repeatable")
    parser.add_argument("--mock-runner", default=None, metavar="BINARY",
                        help="provider lane: shadow BINARY (e.g. `claude`) with "
                             "tests/mocks/mock_llm.py, first on PATH, for the "
                             "applied-bytes row (C4 option (a)). Every other "
                             "binary stays the genuine CLI behind the wrapper.")
    parser.add_argument("--dry-run-host", action="store_true",
                        help="host lane: wire, stage and render the argv, then "
                             "stop before the host process — the whole "
                             "credentialed path at zero cost")
    parser.add_argument("--route", choices=["file", "package"], default="file",
                        help="host lane: which install route this row exercises. "
                             "`file` stages a shim through install-host (the "
                             "default, and what every row did before a package "
                             "route existed); `package` declares the npm spec in "
                             "the host's own config and lets the host's installer "
                             "resolve the pre-seeded payload. It is NOT cosmetic: "
                             "the two routes must not credit each other's "
                             "evidence, so it is what keys capability promotion, "
                             "and it also decides whether the rendered host "
                             "config carries a `plugin[]` entry at all.")
    parser.add_argument("--documented-invocation", action="store_true",
                        help="host lane, --creds none: install the shim, start a "
                             "FRESH session with the host's own model pointed at "
                             "the loopback stub, type the exact `/multi-llm …` "
                             "string README.md and docs/hosts.md publish, and "
                             "record whether it reached our dispatcher "
                             "(invocation.json). Credential-free by construction "
                             "— what is under test is argument delivery, not "
                             "model output.")
    parser.add_argument("--stub-port", type=int,
                        default=int(os.environ.get("MLLM_STUB_PORT", "8099")))
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["MLLM_ROW_ID"] = args.row
    os.environ["MLLM_STUB_PORT"] = str(args.stub_port)
    os.environ["MLLM_CALL_LOG"] = str(OUT / "calls.jsonl")
    os.environ["MLLM_RECEIPT_DIR"] = str(OUT / "receipts" / args.row)

    assert_denylist_absent()
    assert_mount_allowlist(args.lane, args.creds, args.host)
    if args.lane == "provider" or args.creds == "none":
        assert_no_credentials()
    # Written before anything else runs: a row that dies later still carries the
    # posture it actually had, and the host-side `docker inspect` half has
    # something to be compared against.
    (OUT / "posture-inside.json").write_text(
        json.dumps(assert_posture(args.lane, args.creds), indent=2, sort_keys=True),
        encoding="utf-8")

    identities = [i for i in ([args.cli] if args.cli else [])
                  + [h for h in args.helpers.split(",") if h] if i]
    if args.lane == "host":
        identities = sorted(set(identities))
    seed_homes(identities)

    plan = WORKSPACE / "plans" / "auth-feature.md"
    exit_code = 1

    # The documented-invocation probe's marker and fact ladder are minted BEFORE
    # the stub starts, because the stub is the thing that has to carry them: a
    # probe armed after the first request would silently grade a turn it never
    # inspected.
    facts_path: Path | None = None
    marker = ""
    if args.documented_invocation:
        if args.lane != "host" or not args.host:
            raise SystemExit("entrypoint: --documented-invocation is a host-lane "
                             "flag and needs --host")
        if args.creds != "none":
            raise SystemExit(
                "entrypoint: --documented-invocation runs at --creds none. It "
                "asks whether an argument reaches the dispatcher, which the stub "
                "answers for free; a credential here would buy nothing and spend "
                "something.")
        if args.route != "file":
            # The probe is FILE-route by construction, and refusing is the only
            # honest disposition. `wire_stub_host` rewrites the host's own config
            # to point at the loopback stub, taking any `plugin[]` declaration
            # with it, so a package row here would either type the published
            # string at a host with no command at all — a fabricated "the
            # documented spelling does not reach the dispatcher" — or, if the
            # shim were staged anyway, credit the file route's evidence to the
            # package route.
            raise SystemExit(
                f"entrypoint: --documented-invocation is a file-route probe and "
                f"--route {args.route!r} was requested. It rewires the host at "
                f"the stub, which drops the package route's own declaration, so "
                f"whatever it recorded would be about the other route.")
        marker = f"mllm-invocation-{uuid.uuid4().hex[:8]}"
        cell = dict((_host_matrix().get(args.host) or {})
                    .get("documented_invocation") or {})
        typed_preview = str(cell.get("typed") or "").replace(
            "{plan}", f"plans/{marker}.md")
        facts_path = OUT / "prompt-facts.json"
        facts_path.write_text(json.dumps(
            invocation_facts(args.host, typed_preview,
                             f"plans/{marker}.md", marker),
            indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with Stub(args.stub_port, OUT / "stub.jsonl",
              Path(args.scenario) if args.scenario else None,
              facts=facts_path):
        # NOTE: the seed templates are deliberately left UNSUBSTITUTED here.
        # `{{base_url}}` and `{{stub_session_key}}` are filled by the wrapper, per
        # `exec`, because the exec id that keys the stub's turn counter is minted
        # per invocation and the entrypoint knows only the stub's origin.
        # Substituting here first would win — the wrapper copies this tree and
        # would find no `{{` left — and every config-file CLI would silently fall
        # back to the coarse `(model, surface)` session key. Measured: it did
        # exactly that, logging `session=model:gpt-5.5` instead of the exec id.
        try:
            seeder = RIG_DIR / "seed_workspace.py"
            # The fan-out is forwarded WHOLE, one `--spec` flag per member, in
            # the order the row named them.
            #
            # Both of the defaults that used to stand here — one behind
            # `--spec`, one behind `--runner` — named the same hard-coded codex
            # stub slug, which was the single-provider fan-out this lane is
            # getting rid of. They failed in the worst direction available: a
            # row that forgot `--spec` still ran, still produced a bundle, and
            # measured codex while claiming to measure whatever the row was
            # named after. So the absence of a spec is an error now, and the
            # slug itself is gone from this file entirely — grepping this file
            # for that stub slug is the cheap guard that it stays gone, which
            # is also why it is not spelled out anywhere in these comments.
            specs = list(args.spec or [])
            if not specs:
                raise SystemExit(
                    "entrypoint: --spec is required (and repeatable). It "
                    "used to fall back to a hard-coded codex stub slug, so a "
                    "row that named no fan-out silently measured codex; name "
                    "the row's fan-out explicitly instead.")
            # `--runner` stays SINGULAR — `driver.subagent_runner` is one slug
            # — and it is REQUIRED, exactly as `--spec` is here and as
            # `seed_workspace.py --runner` already is.
            #
            # It used to fall back to `specs[0]`. With `--spec` now a LIST that
            # is not "the row's runner" at all: it is whichever provider the
            # row's fan-out happens to name first — `codex` on the claude-code
            # host, `claude-code` on the opencode host — which need not declare
            # `agent-write`, and which on one host is precisely the sandboxed
            # runner T0a exists to point away from (codex's `agent-write`
            # renders `--sandbox workspace-write`, i.e. bubblewrap, which needs
            # a user namespace Docker's default seccomp profile denies).
            #
            # The write runner is a MATRIX decision — `harness.host_runner_cli`,
            # defaulting per host to the write-capable shared helper, and the
            # row's own CLI on a `remove_seed` row — so the honest disposition
            # for a row that named none is to stop, loudly, rather than to
            # dispatch every write batch through an arbitrary fan-out member
            # and file the result under the row's name.
            runner = args.runner
            if not runner:
                raise SystemExit(
                    f"entrypoint: --runner is required. It used to fall back "
                    f"to the first --spec ({specs[0]!r} for this row), which "
                    f"is only whichever provider this row's fan-out happens to "
                    f"list first — it need not declare agent-write, and it may "
                    f"be the sandboxed runner the matrix points away from. "
                    f"Name the row's write runner explicitly (the matrix's "
                    f"`runner` key, read by harness.host_runner_cli).")
            spec_argv = [arg for spec in specs for arg in ("--spec", spec)]
            subprocess.check_call(
                [sys.executable, str(seeder), "--root", str(WORKSPACE),
                 *spec_argv, "--runner", runner])
            # The WHOLE-ROW pair. Taken here, before `run_pre_commands`, so it
            # brackets everything the container did; `write_host_census` takes
            # the narrower pair that brackets the host process alone.
            (OUT / "workspace-census-before.json").write_text(
                json.dumps(_workspace_census(), indent=2, sort_keys=True),
                encoding="utf-8")

            if args.lane == "host":
                entry = dict(_host_matrix().get(args.host) or {})
                row = _matrix_row(args.host, args.row)
                key = os.environ.get("OPENROUTER_API_KEY", "")
                overlay: dict = {}
                secrets: list[str] = []

                if args.creds in ("openrouter", "fallback") and not row:
                    raise SystemExit(
                        f"entrypoint: host_matrix.yaml names no row "
                        f"{args.row!r} under host {args.host!r}")

                if args.creds == "openrouter":
                    if not key:
                        raise SystemExit(
                            "entrypoint: --creds openrouter with no "
                            "OPENROUTER_API_KEY in the environment")
                    census = credential_preflight(args.host, entry, key, row)
                    census["row"] = args.row
                    census["credential_mode"] = "openrouter"
                    (OUT / "credential-census.json").write_text(
                        json.dumps(census, indent=2, sort_keys=True),
                        encoding="utf-8")
                    overlay = wire_credentials(
                        args.host, Path.home(), key, pinned_model(entry, row),
                        route=args.route)

                elif args.creds == "fallback":
                    # The documented subscription / vendor-key path (C6a). It runs
                    # the SAME row the OpenRouter path does — the difference is
                    # where the credential comes from, which is why it joins the
                    # common stage-and-launch flow below rather than having one.
                    slug = pinned_model(entry, row)
                    overlay, copied = wire_fallback_credentials(
                        args.host, Path.home(), slug, route=args.route)
                    secrets = fallback_secret_values(copied)
                    census = {
                        "host": args.host,
                        "row": args.row,
                        "credential_mode": "fallback",
                        "model": slug,
                        # PATHS, never values — the same rule creds.yaml itself is
                        # written under.
                        "copied": sorted(str(p) for p in copied),
                        "checks": {
                            "spend_limit": "not applicable: the vendor's own "
                                           "subscription or quota is the ceiling, "
                                           "and this rig cannot read it",
                            "model_pin": "unchecked: no OpenRouter catalog on "
                                         "this path",
                        },
                    }
                    if not os.environ.get("MLLM_HOST_MODEL", "").strip():
                        # host_matrix.yaml's pins are OpenRouter slugs, verified
                        # against that catalog. On vendor auth the host resolves
                        # ids against its OWN catalog, so an unset override is a
                        # row that may fail on the model rather than on the shim —
                        # said once, here, instead of being diagnosed later from a
                        # provider's error string.
                        log(f"WARNING: --creds fallback with no MLLM_HOST_MODEL; "
                            f"driving {args.host} at the checked-in OpenRouter "
                            f"pin {slug!r}, which its own account may not serve")
                        census["checks"]["model_pin"] += (
                            " — and the pin is an OpenRouter slug, unset "
                            "MLLM_HOST_MODEL")
                    (OUT / "credential-census.json").write_text(
                        json.dumps(census, indent=2, sort_keys=True),
                        encoding="utf-8")

                if args.creds == "none":
                    # THE FREE TIER'S WIRING STEP, and the only one it has.
                    # Both credentialed tiers reach `stage_host` through a
                    # `wire_*` function that seeds `home_seed/<host>/` on its way
                    # past; this tier reached it through nothing, so
                    # `install-host --check`'s row 5 and `discovery_step` both
                    # answered about an empty `$HOME`. Seeded HERE rather than
                    # beside `discovery_step` for the reason the
                    # documented-invocation branch already gives a few lines
                    # further down: a half-wired `$HOME` makes the install check
                    # answer about the wrong thing.
                    seed_host_home(args.host, Path.home())

                    if args.route == "package":
                        # THE PACKAGE ROUTE'S WIRING STEP, and this tier is the
                        # one that has to carry it. `route` reaches
                        # `_write_opencode_config` only through `wire_credentials`
                        # / `wire_fallback_credentials`, and neither of them runs
                        # here — so the offline, model-free container row of
                        # 4b.12, which is exactly this tier, got no `plugin[]`
                        # entry and no cache `allow` entries. That is H1's silent
                        # drop reproduced by the very row that exists to catch it:
                        # the image pre-extracts the package, the host never loads
                        # it, and the run ends at exit 0 with nothing in the logs.
                        #
                        # The slug is the row's ordinary pin. Nothing here calls a
                        # model, so it buys no capability — it keeps the rendered
                        # config the same shape the credentialed opencode rows
                        # write, which is what makes the two comparable at all.
                        if args.host != "opencode":
                            raise SystemExit(
                                f"entrypoint: --route package has no host config "
                                f"writer for host {args.host!r}. Only opencode "
                                f"declares a `package:` block today, and a row "
                                f"that quietly wrote nothing would stage no "
                                f"route and then grade one.")
                        _write_opencode_config(
                            Path.home(), pinned_model(entry, row), args.route)
                        log(f"route=package: wrote {args.host}'s own config "
                            f"with the plugin declared; no shim will be staged")

                code = stage_host(args.host, Path.home(), args.route)
                if code != 0:
                    log(f"stage step exited {code}")
                    return code

                if args.creds in ("openrouter", "fallback"):
                    # The usage delta is the OpenRouter key's own readback, so it
                    # exists on that path only; the fallback path's spend lives
                    # with whoever bills the subscription.
                    usage_before = ((key_usage(key) or {})
                                    if args.creds == "openrouter" else {})
                    if args.dry_run_host:
                        # Everything except the model turn: the wiring, the
                        # staged shim and the exact argv, at zero cost.
                        (OUT / "host-run.json").write_text(json.dumps({
                            "host": args.host, "row": args.row,
                            "argv": build_host_argv(entry, row),
                            "credential_mode": args.creds,
                            "dry_run": True, "exit": None,
                        }, indent=2, sort_keys=True), encoding="utf-8")
                        log("--dry-run-host: wiring and argv rendered, no host "
                            "process started")
                        return 0

                    # THE ORDER OF THE NEXT FOUR STEPS IS THE DESIGN, and each
                    # one is where it is because the others put it there.
                    #
                    # 1. Render the argv FIRST. A row that declares no `args`
                    #    aborts in `render_prompt` — and it must keep doing that
                    #    in ~1.5 s having spent nothing, exactly as it does
                    #    today. Rendering after the leg would turn a fast, honest
                    #    abort into a slow one that prepared a workspace for a
                    #    host process that was never going to launch.
                    build_host_argv(entry, row)
                    # 2. Then the pre-command leg, while the seeded credential is
                    #    still in place: the prerequisite read phase (`review-plan`
                    #    on four rows, `review-code` on the native write row) fans
                    #    out through the CONFIGURED RUNNER and would be refused by
                    #    the very gate a negative row exists to trip.
                    code = run_pre_commands(args.row, args.commands)
                    if code != 0:
                        log(f"pre-command leg exited {code}; not launching the "
                            f"host — it would run against a workspace whose "
                            f"prerequisite phase never completed")
                        return code
                    # 3. Only then remove it. Not at seeding time, and not before
                    #    `stage_host`: `install-host codex --global`'s row 5
                    #    probes the configured runner's auth and exits 3 without
                    #    it (host_matrix.yaml, codex `stage.measured`), so an
                    #    earlier removal returns above with no `apply-suggestions`
                    #    receipt at all — and a refusal with no receipt is a row
                    #    that proves nothing about the gate.
                    drop_seeded_credential(identities, row.get("remove_seed"))
                    # 4. And mark the line the host's own calls start after.
                    #    Everything logged up to here belongs to the rig.
                    record_stub_watermark(args.row, row.get("remove_seed"))
                    # 5. And the byte-level equivalent of that watermark: the
                    #    census taken here credits the host with what the host
                    #    did, not with what the leg prepared for it.
                    write_host_census("before")

                    try:
                        record = run_host(args.host, entry, row, Path.home(),
                                          overlay, key, creds=args.creds,
                                          secrets=secrets)
                    finally:
                        # In a `finally`, for the same reason the export is: a
                        # host that died mid-phase is exactly the case where
                        # "what moved" is the question, and a census only
                        # written on the happy path answers it never.
                        write_host_census("after")
                    if ((row.get("expect") or {}).get("run")) == "done":
                        poll_terminal_run(plan)
                    if args.creds != "openrouter":
                        log(f"host row complete (host exit {record['exit']}, "
                            f"credential mode {args.creds})")
                        return 0
                    usage_after = key_usage(key) or {}
                    before = usage_before.get("usage")
                    after = usage_after.get("usage")
                    (OUT / "usage.json").write_text(json.dumps({
                        "row": args.row,
                        "usage_before": before,
                        "usage_after": after,
                        "delta_usd": (None if before is None or after is None
                                      else round(after - before, 6)),
                    }, indent=2, sort_keys=True), encoding="utf-8")
                    log(f"row spend delta: {None if before is None or after is None else round(after - before, 6)} USD")
                    # The container exits 0 because the ROW's verdict is not the
                    # entrypoint's to give: the host's own exit is recorded in
                    # host-run.json and reported separately by the grader,
                    # because "the host did not terminate" and "the command never
                    # ran" are different failures and only the receipt answers
                    # the second one.
                    log(f"host row complete (host exit {record['exit']})")
                    return 0

                if args.creds == "none" and args.documented_invocation:
                    # The FOURTH thing a `--creds none` row can be, and the only
                    # one that types anything: wire the host at the stub, install
                    # the shim, start a fresh session and type the published
                    # string. Ordered exactly as the credentialed path is —
                    # wiring first, staging second — because `install-host`'s
                    # own environment row probes the configured runner and a
                    # half-wired $HOME makes that row answer about the wrong
                    # thing.
                    overlay = wire_stub_host(
                        args.host, Path.home(),
                        f"http://127.0.0.1:{args.stub_port}/v1")
                    # `args.route` is pinned to `file` for this probe by the
                    # refusal at parse time; passed rather than defaulted so the
                    # two staging calls cannot drift apart.
                    code = stage_host(args.host, Path.home(), args.route)
                    if code != 0:
                        log(f"stage step exited {code}; not typing anything — a "
                            f"shim that was never installed cannot be invoked")
                        return code
                    drive_documented_invocation(
                        args.host, entry, Path.home(), overlay, marker,
                        OUT / "stub.jsonl")
                    # 0 either way: whether the host takes the documented
                    # spelling is the GRADER's verdict over invocation.json, and
                    # "it does not" is a finding rather than a broken row.
                    return 0

                if args.creds == "none":
                    # The --creds none tier stops here by design: stage, reload
                    # boundary, listed-and-invocable — no workflow row, no model
                    # call, no credential. It buys the single most likely
                    # real-world break at zero cost.
                    #
                    # One host process per row means the `reload: new-session`
                    # boundary is satisfied by construction: there is nothing to
                    # reload into, because the next process IS the new session.
                    census = discovery_step(args.host, Path.home())
                    (OUT / "discovery.json").write_text(
                        json.dumps(census, indent=2, sort_keys=True),
                        encoding="utf-8")
                    log(f"--creds none: {args.host} -> {census['discovery']}"
                        + (f" ({census['reason']})" if census.get("reason") else ""))
                    # `not-listed` and `failed` are row failures; the honest
                    # "this host has no free surface" is not, because reporting
                    # it as one would push an operator toward ignoring the tier.
                    return 0 if census["discovery"] in (
                        "listed", "unavailable-without-credential") else 1

                # Every host-lane credential mode returns above. FALLING THROUGH
                # to the provider lane's command loop is what a `--creds` value
                # with no branch used to do: it staged the shim and then
                # dispatched `--help` (or the row's leftover `--command` values),
                # producing a green row that never launched a host.
                raise SystemExit(
                    f"entrypoint: the host lane has no path for "
                    f"--creds {args.creds!r}")

            # Every command runs in order and every exit code is recorded, so a
            # row that failed at its second dispatcher invocation says which one
            # rather than reporting one number for three commands. The container
            # still exits on the FIRST non-zero: a chain that stopped has
            # nothing meaningful left to read back.
            if args.mock_runner:
                install_mock_runner(
                    args.mock_runner,
                    Path(args.scenario) if args.scenario else None)

            commands = args.commands or ["--help"]
            results = []
            exit_code = 0
            for raw in commands:
                # A leading `NAME=value` applies to THIS invocation only — see
                # `split_command_env`; the applied-bytes row is the reason.
                command_env, argv_one = split_command_env(shlex.split(raw))
                code = dispatch(argv_one, env={**os.environ, **command_env})
                results.append({"command": raw, "exit": code,
                                "env": sorted(command_env)})
                log(f"command {argv_one!r} -> exit {code}"
                    + (f" (env: {sorted(command_env)})" if command_env else ""))
                if code != 0:
                    exit_code = code
                    break
            (OUT / "commands.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
        finally:
            export_artifacts(plan)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
