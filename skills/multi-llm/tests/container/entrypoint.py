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
import base64
import binascii
import fcntl
import hashlib
import json
import os
import pty
import queue
import re
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Row 8's five package-gate scenarios (plans/opencode-plugin-publish.md). Its own
# module, beside this one in /opt/mllm/rig, where the rig dir is `sys.path[0]`;
# imported as a package sibling when the host-side suite imports this file.
try:
    import opencode_package_gate
except ImportError:  # pragma: no cover - the host-side package import
    from . import opencode_package_gate

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
               "{{api_key_env}}", "{{model_name}}", "{{model_alias}}")


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

#: The pty window size `drive_documented_invocation` sets (`TIOCSWINSZ`, below)
#: and the grid `_render_screen` renders onto — one pair of numbers, not two,
#: so the emulator can never disagree with the terminal it is reading.
_TUI_ROWS = 50
_TUI_COLS = 200

#: One token per iteration of `_render_screen`'s walk: an OSC/DCS/charset/
#: single-char escape (no grid effect — title changes, charset designation,
#: DEC private modes), a CSI sequence (`_apply_csi` decides whether it moves
#: the cursor or is inert), a bare control character, or a run of plain text.
#: `finditer` tries each alternative in order and reports which one matched via
#: `lastgroup`, which is what lets `_render_screen` stay a single dispatch
#: table instead of six separate regexes each re-scanning the same bytes.
_TUI_TOKEN_RE = re.compile(
    r"(?P<osc>\x1b\][^\x07\x1b]*(?:\x07|\x1b\\))"    # OSC … BEL / ST
    r"|(?P<dcs>\x1bP[^\x1b]*\x1b\\)"                 # DCS … ST
    r"|(?P<csi>\x1b\[[0-9;?<>=]*[ -/]*[@-~])"        # CSI
    r"|(?P<charset>\x1b[()][A-Za-z0-9])"             # designate charset
    r"|(?P<single>\x1b[=>NOMc78])"                   # single-character escapes
    r"|(?P<cr>\r)"
    r"|(?P<lf>\n)"
    r"|(?P<bs>\x08)"
    r"|(?P<bel>\x07)"
    r"|(?P<text>[^\x1b\r\n\x08\x07]+)"
)

#: Pulls the parameter string and final byte back out of one matched CSI token
#: (`_TUI_TOKEN_RE`'s `csi` group), so `_apply_csi` never re-derives what the
#: outer regex already found.
_CSI_PARSE_RE = re.compile(r"\x1b\[([0-9;?<>=]*)[ -/]*([@-~])")


def _csi_params(token: str) -> tuple[list[int], str]:
    """``("\\x1b[8;16H", ...)`` -> ``([8, 16], "H")``.

    Non-numeric leaders (``?``, ``<``, ``>``, ``=`` — DEC private-mode markers)
    are dropped rather than rejected, so ``\\x1b[?25h`` parses to ``([25], "h")``
    instead of raising: harmless, because ``h``/``l`` are not one of the finals
    `_apply_csi` interprets, so the (unused) numbers never matter.
    """
    match = _CSI_PARSE_RE.match(token)
    if not match:
        return [], ""
    params, final = match.group(1), match.group(2)
    numbers = [int(digits) if (digits := "".join(c for c in part if c.isdigit()))
               else 0
               for part in params.split(";")]
    return numbers, final


def _apply_csi(grid: list, cursor: list, token: str) -> None:
    """Interpret one CSI token against ``grid``, moving or clearing through
    ``cursor`` (``[row, col]``, 0-indexed, mutated in place).

    Only the finals that move the cursor or erase cells are interpreted —
    cursor position/HVP (`H`/`f`), the four relative moves (`A`/`B`/`C`/`D`),
    column-absolute (`G`), erase-in-line (`K`) and erase-in-display (`J`).
    Everything else this full-screen TUI sends — SGR colour, DEC private
    mode toggles, synchronized-update markers — has no textual effect and is
    consumed with no change to the grid, exactly as the old blanket strip
    dropped it.
    """
    numbers, final = _csi_params(token)
    rows, cols = len(grid), (len(grid[0]) if grid else 0)

    def clamp_row(r: int) -> int:
        return max(0, min(rows - 1, r))

    def clamp_col(c: int) -> int:
        return max(0, min(cols - 1, c))

    def param(index: int, default: int = 1) -> int:
        return numbers[index] if len(numbers) > index and numbers[index] else default

    if final in ("H", "f"):
        cursor[0] = clamp_row(param(0) - 1)
        cursor[1] = clamp_col(param(1) - 1)
    elif final in ("A", "B", "C", "D"):
        n = param(0)
        if final == "A":
            cursor[0] = clamp_row(cursor[0] - n)
        elif final == "B":
            cursor[0] = clamp_row(cursor[0] + n)
        elif final == "C":
            cursor[1] = clamp_col(cursor[1] + n)
        else:
            cursor[1] = clamp_col(cursor[1] - n)
    elif final == "G":
        cursor[1] = clamp_col(param(0) - 1)
    elif final == "K":
        mode = numbers[0] if numbers else 0
        row = grid[cursor[0]]
        if mode == 0:
            for c in range(cursor[1], cols):
                row[c] = " "
        elif mode == 1:
            for c in range(0, cursor[1] + 1):
                row[c] = " "
        elif mode == 2:
            grid[cursor[0]] = [" "] * cols
    elif final == "J":
        mode = numbers[0] if numbers else 0
        if mode == 0:
            for c in range(cursor[1], cols):
                grid[cursor[0]][c] = " "
            for r in range(cursor[0] + 1, rows):
                grid[r] = [" "] * cols
        elif mode == 1:
            for r in range(0, cursor[0]):
                grid[r] = [" "] * cols
            for c in range(0, cursor[1] + 1):
                grid[cursor[0]][c] = " "
        else:
            for r in range(rows):
                grid[r] = [" "] * cols
    # else: SGR, DEC private modes, synchronized-update markers, … — inert.


def _render_screen(text: str) -> str:
    """Render a full-screen TUI's repaint stream into readable text.

    A modern ratatui/ink-style TUI redraws by **positioning the cursor and
    overwriting only the cells that changed** — it does not reprint a whole
    line with real space characters between words. Measured on codex 0.159.2's
    refusal line: the six words arrive as six separate `text` runs, each
    preceded by its own absolute `CSI row;colH`, and the one-column gap between
    each pair of words is never a byte on the wire — it is the *distance*
    between where one run ends and the next one's `H` places the cursor. The
    old approach — `_ANSI.sub("", …)`, deleting every escape sequence with no
    memory of what it did — deleted that distance along with the sequence that
    encoded it, so `Unrecognized` and `command` landed adjacent with nothing
    between them. Concatenating bytes in wire order can never recover a
    position-encoded gap; only tracking where each run actually lands can.

    So this walks the stream on a `_TUI_ROWS` x `_TUI_COLS` grid — the same
    dimensions the pty is opened with — moving a cursor for CR/LF/backspace and
    for the CSI moves `_apply_csi` interprets, and writing each run of plain
    text at the cursor's current cell, advancing one column per character. The
    result is column-accurate for exactly the case that matters here (words
    placed apart on one row), and degrades no worse than the old strip for
    everything else: a CLI that writes contiguous runs with real space
    characters (opencode's ready banner, measured 2026-08-24) renders
    identically either way, because contiguous same-row text with no
    repositioning is the trivial case both approaches already agreed on.
    """
    grid = [[" "] * _TUI_COLS for _ in range(_TUI_ROWS)]
    cursor = [0, 0]  # [row, col], 0-indexed

    for match in _TUI_TOKEN_RE.finditer(text):
        kind = match.lastgroup
        token = match.group()
        if kind == "text":
            for ch in token:
                if cursor[1] >= _TUI_COLS:
                    cursor[1] = 0
                    cursor[0] = min(_TUI_ROWS - 1, cursor[0] + 1)
                grid[cursor[0]][cursor[1]] = ch
                cursor[1] += 1
        elif kind == "cr":
            cursor[1] = 0
        elif kind == "lf":
            cursor[0] = min(_TUI_ROWS - 1, cursor[0] + 1)
        elif kind == "bs":
            cursor[1] = max(0, cursor[1] - 1)
        elif kind == "csi":
            _apply_csi(grid, cursor, token)
        # osc / dcs / charset / single / bel: no grid effect.

    return "\n".join("".join(row).rstrip() for row in grid).strip("\n")


def _plain(raw: bytes) -> str:
    return _render_screen(raw.decode("utf-8", "replace"))


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


def _declare_opencode_config_provider(base_url: str) -> None:
    """Declare the container's `stub` provider the way a hermetic opencode
    dispatch can actually see it: `providers.opencode.config_providers` in
    the row's own `.multi-llm/providers.yaml` (the project config layer
    `provider_registry.load_config` deep-merges onto base), never opencode's
    own `opencode.json` — which a hermetic run replaces before the CLI ever
    reads it (`utils/providers/opencode.py`'s `_hermetic_config_content`).

    No credential here, matching `providers.yaml`'s own documentation of the
    key ("credentials excluded ... they stay in auth.json"): the container's
    stub takes none at all, so there is nothing to carry and nothing for
    `wrapper.py`'s `carried_config_violation` to have to catch later.
    """
    path = WORKSPACE / ".multi-llm" / "providers.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(
        text
        + "providers:\n"
        + "  opencode:\n"
        + "    config_providers:\n"
        + "      stub:\n"
        + '        npm: "@ai-sdk/openai-compatible"\n'
        + '        name: "Stub"\n'
        + "        options:\n"
        + f"          baseURL: {json.dumps(base_url)}\n"
        + "        models:\n"
        + '          stub-model: {"name": "Stub Model"}\n',
        encoding="utf-8")


def wire_stub_host(host: str, home: Path, base_url: str,
                   route: str = "file", install_source: str = "local-path") -> dict:
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
        # The package route's install, into the config just written (which
        # stays quiet — `_seed_codex_plugin` asserts it). A no-op on `file`.
        # A Git install hands back the one variable every later codex call
        # needs to reach its repository (`GIT_SSH_COMMAND`).
        seeded = _seed_codex_plugin(home, route, install_source)
        overlay.update((seeded or {}).get("overlay") or {})

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
        # The block above is what opencode's OWN interactive session reads for
        # its top-level chat model — it is NOT what the plugin's own hermetic
        # opencode dispatch reads. `providers.opencode.hermetic` defaults to
        # `on`, and this row's `driver.subagent_runner` names
        # `opencode:stub/stub-model` (host_matrix.yaml), so `install-host
        # opencode --check`'s row 5 resolves that runner through
        # `OpenCodeProvider.describe_unresolvable_model`, which runs opencode
        # under an empty `XDG_CONFIG_HOME` and rebuilds `OPENCODE_CONFIG_CONTENT`
        # from `providers.opencode.config_providers` alone. The moment the file
        # above exists, the CLI's OWN resolved config declares a `stub`
        # provider that config_providers does not — "defined only in your own
        # opencode config, which a hermetic run replaces" — and the row is
        # refused before the shim can be typed into. The refusal names its own
        # fix: declare the same provider under `config_providers`, credentials
        # excluded (this stub takes none at all). Appended text, not a YAML
        # round-trip, matching `_workspace_config_with`'s idiom elsewhere in
        # this file — `.multi-llm/providers.yaml` is `seed_workspace.py`'s
        # freshly rendered `providers.tmpl.yaml`, a `str.replace` render with
        # no `providers:` top-level key of its own to merge into.
        _declare_opencode_config_provider(base_url)
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
    fcntl.ioctl(fd, termios.TIOCSWINSZ,
                struct.pack("HHHH", _TUI_ROWS, _TUI_COLS, 0, 0))

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

    * **the key's own spend limit.** `MLLM_E2E_BUDGET_USD` and each row's
      `cost_budget_usd` are enforced by polling (`host_spend.py`), which stops a
      runaway only a poll interval plus the endpoint's settling lag late, so the
      only limit that holds is the one enforced by the party doing the billing. A key reporting no `limit` is
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


def pinned_model(entry: dict, row: dict | None = None,
                 creds: str = "openrouter") -> str:
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

    On a ``fallback`` row (the developer's own subscription) a fourth source sits
    between the override and the row pin: the host's ``subscription.model``. The
    row and host pins are OpenRouter slugs, which a vendor's own catalog may not
    serve, so they apply there only when the host declares no subscription pin.
    """
    override = os.environ.get("MLLM_HOST_MODEL", "").strip()
    if override:
        return override
    if creds == "fallback":
        subscription, _ = subscription_model(entry)
        if subscription:
            return subscription
    row_pin = ((row or {}).get("model") or {}).get("slug") or ""
    return row_pin.strip() or ((entry.get("model") or {}).get("slug") or "").strip()


#: What a `subscription.model.effort` may say: codex's own reasoning levels are
#: bare lowercase words, and the value is written into a TOML string, so nothing
#: else is let through.
_EFFORT_TOKEN = re.compile(r"^[a-z]+$")


def subscription_model(entry: dict) -> tuple[str, str]:
    """``(slug, effort)`` from the host's ``subscription.model`` block, or ``("", "")``.

    Operator decision 2026-10-02 (host_matrix.yaml's MODEL POLICY block): codex
    and claude-code run on the developer's own subscription when it is present,
    each on the model pinned here. Pure: no environment read — the
    ``MLLM_HOST_MODEL`` override is ``pinned_model``'s, and the effort is applied
    only when the slug that runs IS this pin.
    """
    pin = (entry.get("subscription") or {}).get("model") or {}
    slug = str(pin.get("slug") or "").strip()
    effort = str(pin.get("effort") or "").strip()
    if effort and not _EFFORT_TOKEN.match(effort):
        raise SystemExit(f"entrypoint: subscription.model.effort {effort!r} is not a "
                         f"bare reasoning level (lowercase letters only)")
    return slug, effort


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
        if route == "package":
            # The plugin install, then the key's SECOND carrier (D3a): the
            # session keeps reading it from its own environment through the
            # `sh -c` auth command above, and the MCP server's login-shell child
            # re-derives it from `.profile`, which codex's `env_vars` allowlist
            # never carries. The precedence assertion runs inside the write.
            # File-route rows are untouched: they spawn no MCP server, and their
            # host environment stays byte-identical to every recorded run.
            _seed_codex_plugin(home, route)
            write_profile_credential(home, key)
            export_harness_names_into_profile(home)
            wrap_profile_with_read_probe(home)

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


def _codex_fallback_config(effort: str = "") -> str:
    """The fallback (subscription) path's codex config.toml.

    `_codex_first_run_config` plus, when the host's `subscription.model` pins
    one, `model_reasoning_effort` — written FIRST, because a TOML key after the
    `[projects."…"]` table header would belong to that table rather than to the
    root and codex would ignore it.
    """
    if effort and not _EFFORT_TOKEN.match(effort):
        raise SystemExit(f"entrypoint: refusing reasoning effort {effort!r}")
    head = f'model_reasoning_effort = "{effort}"\n' if effort else ""
    return head + _codex_first_run_config()


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


# --------------------------------------------------------------------------- #
# The codex PACKAGE route (plans/codex-native-plugin.md, Phase 1 items 1–14)
#
# `install-host codex` writes a FILE shim a model reads and retypes. The package
# route installs the codex PLUGIN instead — a marketplace entry whose directory
# carries a manifest, a `.mcp.json` declaring a stdio MCP server, and the skill
# payload — so the model calls a `multi_llm` TOOL. Dockerfile §10c mirrors the
# repository's catalog and plugin tree into /opt/mllm/codex-marketplace/ at
# their own relative paths; everything below installs from there at RUN time,
# because an install writes into $CODEX_HOME and $CODEX_HOME is the row's tmpfs
# home, never the image.
#
# WHERE $CODEX_HOME IS. The plan's item 5 spells it `/work/homes/<host>/.codex`;
# on this lane the HOST's home is `Path.home()` (posture.yaml: the tmpfs
# /home/e2e), and every existing codex host row already runs with
# `CODEX_HOME=$HOME/.codex` (the three config writers' overlays). `/work/homes/
# codex` is the PROVIDER identity the wrapper copies into every nested codex
# dispatch — installing the plugin there would enable the plugin's MCP server
# inside every nested provider codex, the sandbox-posture row's grandchild
# included. So the install goes where the host reads: `<home>/.codex`, with the
# same argument that picks the host's config. Neither is under /tmp, which is
# the refusal the plan's placement note was guarding against.
# --------------------------------------------------------------------------- #

#: Dockerfile §10c's mirror root: `.agents/plugins/marketplace.json` and
#: `packaging/codex/multi-llm/` under it, at the repository's own relative paths.
CODEX_MARKETPLACE_DIR = Path(
    os.environ.get("MLLM_CODEX_MARKETPLACE_DIR", "/opt/mllm/codex-marketplace"))
#: The marketplace name the catalog declares and the plugin id it installs as.
CODEX_MARKETPLACE = "beastlabai"
CODEX_PLUGIN_ID = f"multi-llm@{CODEX_MARKETPLACE}"
#: The table `codex plugin add` writes into config.toml (R15). The generated
#: config is QUIET — it never carries this or any `[marketplaces.*]` table — so
#: the table's presence after the install is evidence the install wrote it.
CODEX_PLUGIN_TABLE = f'[plugins."{CODEX_PLUGIN_ID}"]'
#: The MCP server's name in `.mcp.json`, which is what `codex mcp get` takes
#: and what codex namespaces the tool under (`mcp__multi_llm`).
CODEX_MCP_SERVER = "multi_llm"
#: What a shim-route install would leave behind. A package row must never have
#: one: two advertised copies of the skill is the state D5's note warns about,
#: and grading one route on the other's file is the false promotion `Row.route`
#: exists to stop.
CODEX_FILE_SHIM = Path(".codex") / "skills" / "multi-llm" / "SKILL.md"
#: Bytes of any one command's stdout/stderr kept in a probe record.
CODEX_OUTPUT_CAP = 65536


def _codex_env(home: Path, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["CODEX_HOME"] = str(home / ".codex")
    env.update(extra or {})
    return env


def _run_codex(args: list[str], env: dict, timeout: float = 120.0,
               cwd: Path | None = None) -> dict:
    """One codex subcommand by its REAL path, recorded whole.

    By absolute path for `discovery_step`'s reason: the wrapper rebuilds the
    child's environment and points `$HOME` at a provider root, and every
    question asked here is about the HOST's own `$CODEX_HOME`.
    """
    argv = [str(REAL_DIR / "codex"), *args]
    started = time.time()
    try:
        done = subprocess.run(argv, env=env, cwd=str(cwd or WORKSPACE),
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        code, out, err, timed_out = done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, timed_out = None, True
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError as exc:
        code, out, err, timed_out = None, "", str(exc), False
    return {
        "argv": ["codex", *args],
        "exit": code,
        "timed_out": timed_out,
        "wall_clock_s": round(time.time() - started, 2),
        "stdout": out[:CODEX_OUTPUT_CAP],
        "stdout_truncated": len(out) > CODEX_OUTPUT_CAP,
        "stderr": err[:CODEX_OUTPUT_CAP],
        "stderr_truncated": len(err) > CODEX_OUTPUT_CAP,
        "_stdout_full": out,
    }


def _public(result: dict) -> dict:
    """A `_run_codex` result without its uncapped copy of stdout."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _json_or_none(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _find_key(node, key: str):
    """The first value stored under ``key`` anywhere in a JSON document."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _config_toml_tables(text: str) -> list[str]:
    """Every table header line in a config.toml, verbatim, comments excluded."""
    headers = []
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if body.startswith("["):
            headers.append(body)
    return headers


def _plugin_table_text(text: str) -> str | None:
    """The `[plugins."multi-llm@beastlabai"]` table, header to the next header."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.split("#", 1)[0].strip() == CODEX_PLUGIN_TABLE:
            block = [line]
            for follow in lines[index + 1:]:
                if follow.split("#", 1)[0].strip().startswith("["):
                    break
                block.append(follow)
            return "\n".join(block).rstrip() + "\n"
    return None


def assert_codex_config_quiet(home: Path) -> None:
    """The generated config.toml carries no `[plugins.*]` / `[marketplaces.*]`.

    Asserted rather than assumed: the install's own table is the evidence the
    install ran, and a seed that pre-wrote one would make that evidence the
    rig's rather than codex's.
    """
    config = home / ".codex" / "config.toml"
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    loud = [h for h in _config_toml_tables(text)
            if h.startswith(("[plugins", "[marketplaces"))]
    if loud:
        raise SystemExit(
            f"entrypoint: {config} already carries {loud} before the plugin "
            f"install ran. The generated codex config must stay quiet, or the "
            f"install's own table stops being evidence the install wrote it.")


def codex_installed_root(home: Path, install: dict | None = None) -> Path | None:
    """The installed plugin directory: `installedPath` off `plugin add --json`
    (R40 — the one document that carries it), else the one version directory
    under the cache root, else ``None``."""
    recorded = (install or {}).get("installedPath")
    if recorded and Path(recorded).is_dir():
        return Path(recorded)
    cache = home / ".codex" / "plugins" / "cache" / CODEX_MARKETPLACE / "multi-llm"
    versions = sorted(p for p in cache.glob("*") if p.is_dir()) if cache.is_dir() else []
    return versions[-1] if len(versions) == 1 else None


def _seed_codex_plugin(home: Path, route: str,
                       install_source: str = "local-path") -> dict | None:
    """Install the codex plugin into ``<home>/.codex`` on a `route: package` row.

    ``install_source="git"`` is A4 row 6b's install (the `git-route` probe):
    the same three commands and the same record, except that step 1 adds an
    OFFLINE Git marketplace — ``ssh://localhost<bare repo>`` with ``--ref
    release`` and the two published ``--sparse`` paths — built from this same
    mirror by :func:`build_git_marketplace`. The record then also carries a
    ``git`` block (the fixture, the snapshot codex checked out, and the
    installed tree compared blob for blob with the release ref's tree, before
    anything ran from it) and an ``overlay`` naming ``GIT_SSH_COMMAND``, which
    every later codex call on the row needs to reach the repository.

    Three commands, in the order the plan fixes (item 5), each recorded whole
    in ``plugin-install.json``:

    1. ``codex plugin marketplace add /opt/mllm/codex-marketplace`` — the
       mirror §10c built, whose ``.agents/plugins/marketplace.json`` is the
       catalog file users fetch, unedited;
    2. ``codex plugin add multi-llm@beastlabai --json`` — whose document is the
       one that carries ``installedPath`` (R40);
    3. ``codex plugin list --json`` — its stdout committed VERBATIM as
       ``codex-plugin-list.json``, because it is the only measurement of what a
       *list* entry carries, as against the ``add`` entry. D5's resolver and the
       ``--check`` fixtures replay that document, so the two payloads are never
       conflated again.

    The record also carries the config.toml table the install wrote (quiet
    before, asserted), whether a file shim is present (it must not be), and
    ``divergence: null`` — the slot D3a reserves for the one line by which an
    image-side edit of the installed ``.mcp.json`` would make the tested tree
    differ from the shipped one. Nothing here edits it.

    Returns the record, or ``None`` on a file-route row (nothing to install).
    Raises when the install did not produce an installed directory: every row
    below the seed stands on it, so a missing install is a stopped row rather
    than a row that grades an empty home.
    """
    if route != "package":
        return None
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    assert_codex_config_quiet(home)
    config = codex_home / "config.toml"
    before = config.read_text(encoding="utf-8") if config.is_file() else ""
    if install_source not in ("local-path", "git"):
        raise SystemExit(f"entrypoint: unknown codex install source {install_source!r}")
    fixture = build_git_marketplace(home) if install_source == "git" else None
    env = _codex_env(home, fixture["env"] if fixture else None)

    if fixture:
        marketplace = _run_codex(
            ["plugin", "marketplace", "add", fixture["url"], *_git_marketplace_flags(),
             "--json"], env)
    else:
        marketplace = _run_codex(
            ["plugin", "marketplace", "add", str(CODEX_MARKETPLACE_DIR)], env)
    add = _run_codex(["plugin", "add", CODEX_PLUGIN_ID, "--json"], env)
    listing = _run_codex(["plugin", "list", "--json"], env)
    # VERBATIM — the whole stdout, uncapped, as codex printed it.
    (OUT / "codex-plugin-list.json").write_text(listing["_stdout_full"],
                                                encoding="utf-8")

    add_doc = _json_or_none(add["_stdout_full"])
    installed_path = _find_key(add_doc, "installedPath")
    after = config.read_text(encoding="utf-8") if config.is_file() else ""
    root = codex_installed_root(home, {"installedPath": installed_path})
    record = {
        "plugin_id": CODEX_PLUGIN_ID,
        # WHERE the install came from — the datum `harness.promotable_fields`
        # keys `package` on (rule j). This seed is a local-path marketplace,
        # so it earns the behavioural cells only; `package` moves solely on a
        # row whose install source is recorded as `git` (Phase 4's route).
        "install_source": "local-path",
        "marketplace_dir": str(CODEX_MARKETPLACE_DIR),
        "codex_home": str(codex_home),
        "commands": [_public(marketplace), _public(add), _public(listing)],
        "add_document": add_doc,
        "installedPath": installed_path,
        "installed_root": str(root) if root else None,
        "installed_root_exists": bool(root and root.is_dir()),
        "config_toml": {
            "tables_before": _config_toml_tables(before),
            "tables_after": _config_toml_tables(after),
            "plugin_table": _plugin_table_text(after),
        },
        "list_document_file": "codex-plugin-list.json",
        "list_entry_has_installedPath": (
            _find_key(_json_or_none(listing["_stdout_full"]), "installedPath")
            is not None),
        "file_shim_present": (home / CODEX_FILE_SHIM).exists(),
        "divergence": None,
        # The digest the installed tree says it was built as. The release gate
        # requires it to equal the checkout's, so a row graded green on an image
        # built from other bytes cannot vouch for these (packaging/codex/release_gate.py).
        "installed_digest": _installed_digest(root),
        "list_document": _json_or_none(listing["_stdout_full"]),
    }
    if fixture:
        record["install_source"] = "git"
        record["marketplace_dir"] = fixture["url"]
        record["overlay"] = dict(fixture["env"])
        record["git"] = _git_install_facts(home, fixture, root)
    (OUT / "plugin-install.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    log(f"route=package: codex plugin {CODEX_PLUGIN_ID} "
        f"marketplace add -> {marketplace['exit']}, add -> {add['exit']}, "
        f"list -> {listing['exit']}; installed at {record['installed_root']}")
    if not record["installed_root_exists"]:
        raise SystemExit(
            f"entrypoint: `codex plugin add {CODEX_PLUGIN_ID} --json` left no "
            f"installed directory (exit {add['exit']}, installedPath "
            f"{installed_path!r}); plugin-install.json has all three commands. "
            f"Every package row stands on this install.")
    return record


# --------------------------------------------------------------------------- #
# A4 row 6b — the OFFLINE Git marketplace (plans/codex-native-plugin.md,
# Phase 4's "offline Git-route row").
#
# The published install is a Git marketplace pinned to a ref with two sparse
# paths; the local-path seed above lands in a different marketplace root and
# never runs that code path in codex. This fixture runs it with no network:
#
#   * a repository built from this image's own mirror (the catalog users fetch
#     and the checked-in plugin tree, byte for byte), plus decoys outside the
#     two sparse paths — `plans/`, a root README and the shadowed Claude
#     catalog — that must never reach the install;
#   * `release` holds commit A; the default branch holds commit B, which bumps
#     the manifest version and adds a marker file, the shape every real next
#     release has (bump discipline: digest moved => version moved);
#   * a bare clone served as `ssh://localhost<abs path>`, through a
#     `GIT_SSH_COMMAND` that runs the remote command locally. codex accepts
#     only `owner/repo`, HTTPS and SSH URLs (measured 0.164.0: `file://` is
#     "invalid marketplace source format"), so SSH is the one offline spelling.
#
# Measured on the host 2026-10-23 (codex 0.164.0, isolated CODEX_HOME) before
# this row existed: the install takes `release`'s tree and not the default
# branch's; `marketplace upgrade` re-resolves the pinned ref and, when the
# version moved, replaces the cache's version directory itself; and re-adding
# the marketplace with a different `--ref` is REFUSED ("already added from a
# different source"), not ignored. The row records each of those again here.
# --------------------------------------------------------------------------- #

#: The published flags, in the order the README prints them.
GIT_RELEASE_REF = "release"
GIT_DEFAULT_BRANCH = "master"
GIT_SPARSE_PATHS = (".agents", "packaging/codex")
#: Where the plugin lives inside the repository (the catalog's `source.path`).
GIT_PLUGIN_PATH = "packaging/codex/multi-llm"
#: Commit B's marker: present on the default branch, absent from `release`.
GIT_NEXT_MARKER = "NEXT-RELEASE-MARKER"
#: Outside both sparse paths. None of them may appear in the install.
GIT_DECOYS = {
    "plans/decoy-plan.md": "# A gitignored-scratch stand-in; never installed\n",
    "README.md": "# The repository root; never installed\n",
    # The real repository ships this file, with a root source. The `.agents`
    # catalog shadows it; the sparse checkout must not even fetch it.
    ".claude-plugin/marketplace.json": json.dumps({
        "name": "beastlabai", "plugins": [{"name": "multi-llm", "source": "./"}],
    }, indent=2) + "\n",
}
#: git's "simple" ssh variant passes `<host> <command>`; the command is last.
_GIT_SSH_SHIM = (
    "# A local stand-in for ssh: run the remote command (the last argument).\n"
    'eval "remote=\\${$#}"\n'
    'exec sh -c "$remote"\n'
)


def _git_marketplace_flags(ref: str = GIT_RELEASE_REF) -> list[str]:
    flags = ["--ref", ref]
    for path in GIT_SPARSE_PATHS:
        flags += ["--sparse", path]
    return flags


def _git_binary() -> str:
    return shutil.which("git") or "/usr/bin/git"


def _git(args: list[str], cwd: Path, env: dict, check: bool = True) -> dict:
    """One git command, recorded. ``check`` raises on a non-zero exit: every
    fixture step is a precondition, and a half-built fixture graded as a codex
    failure would be the rig's bug reported as the product's."""
    argv = [_git_binary(), *args]
    done = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=120)
    result = {"argv": ["git", *args], "exit": done.returncode,
              "stdout": done.stdout[:CODEX_OUTPUT_CAP],
              "stderr": done.stderr[:CODEX_OUTPUT_CAP]}
    if check and done.returncode != 0:
        raise SystemExit(f"entrypoint: git fixture step failed: {result}")
    return result


def _git_fixture_env(home: Path) -> dict:
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


def _next_patch(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(.*)", version or "")
    if not match:
        raise SystemExit(f"entrypoint: the manifest version {version!r} is not semver")
    major, minor, patch, _rest = match.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def build_git_marketplace(home: Path) -> dict:
    """Build the offline Git marketplace and return how to reach it.

    Returns ``url``, ``bare``, ``release_sha``, ``default_sha``, both manifest
    versions, and ``env`` — the one variable codex needs to fetch it
    (``GIT_SSH_COMMAND``), which the row carries into every later codex call.
    """
    root = home / "git-marketplace"
    if root.exists():
        shutil.rmtree(root)
    source = root / "src"
    source.mkdir(parents=True)
    env = _git_fixture_env(home)
    _git(["init", "-q", "-b", GIT_DEFAULT_BRANCH, str(source)], root, env)
    # Commit A: the mirror, unedited, plus the decoys.
    for rel in (".agents/plugins/marketplace.json",):
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CODEX_MARKETPLACE_DIR / rel, target)
    shutil.copytree(CODEX_MARKETPLACE_DIR / GIT_PLUGIN_PATH, source / GIT_PLUGIN_PATH,
                    symlinks=True)
    for rel, text in GIT_DECOYS.items():
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(["add", "-A"], source, env)
    _git(["commit", "-q", "-m", "release N (the tagged commit)"], source, env)
    release_sha = _git(["rev-parse", "HEAD"], source, env)["stdout"].strip()
    _git(["branch", GIT_RELEASE_REF], source, env)
    # Commit B: the next release, on the default branch only.
    manifest_path = source / GIT_PLUGIN_PATH / ".codex-plugin" / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_version = str(manifest.get("version"))
    next_version = _next_patch(release_version)
    manifest["version"] = next_version
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (source / GIT_PLUGIN_PATH / GIT_NEXT_MARKER).write_text(
        f"commit B: {next_version}\n", encoding="utf-8")
    _git(["add", "-A"], source, env)
    _git(["commit", "-q", "-m", "release N+1 (not yet promoted)"], source, env)
    default_sha = _git(["rev-parse", "HEAD"], source, env)["stdout"].strip()
    bare = root / "beastlabai-multi-llm-plugin.git"
    _git(["clone", "-q", "--bare", str(source), str(bare)], root, env)
    shim = root / "ssh-shim.sh"
    shim.write_text(_GIT_SSH_SHIM, encoding="utf-8")
    return {
        "root": str(root), "bare": str(bare), "url": f"ssh://localhost{bare}",
        "release_ref": GIT_RELEASE_REF, "default_branch": GIT_DEFAULT_BRANCH,
        "sparse": list(GIT_SPARSE_PATHS),
        "release_sha": release_sha, "default_sha": default_sha,
        "release_version": release_version, "next_version": next_version,
        "decoys": sorted(GIT_DECOYS),
        # `sh <file>` rather than an executable shim: /home/e2e is noexec.
        "env": {"GIT_SSH_COMMAND": f"sh {shim}"},
    }


def _git_tree_blobs(bare: Path, commit: str, path: str, env: dict) -> dict:
    """``{relative path: [mode, blob id]}`` for every file under ``path`` at
    ``commit``, read out of the bare repository (never a checkout)."""
    listed = _git(["ls-tree", "-r", "--full-tree", f"{commit}:{path}"], bare, env)
    blobs = {}
    for line in listed["stdout"].splitlines():
        meta, _, rel = line.partition("\t")
        mode, kind, blob = meta.split()
        if kind == "blob":
            blobs[rel] = [mode, blob]
    return blobs


def _disk_tree_blobs(root: Path, env: dict) -> dict:
    """The same mapping for a directory on disk: git's blob id of each file's
    bytes (``hash-object --no-filters``) and its mode as git would record it."""
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    if not files:
        return {}
    rels = [p.relative_to(root).as_posix() for p in files]
    done = subprocess.run([_git_binary(), "hash-object", "--no-filters", "--stdin-paths"],
                          cwd=str(root), env=env, input="\n".join(rels) + "\n",
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise SystemExit(f"entrypoint: git hash-object failed: {done.stderr[:2000]}")
    blobs = {}
    for rel, path, blob in zip(rels, files, done.stdout.split()):
        if path.is_symlink():
            mode = "120000"
        else:
            mode = "100755" if os.access(path, os.X_OK) else "100644"
        blobs[rel] = [mode, blob]
    return blobs


def compare_tree_to_commit(installed: Path | None, bare: Path, commit: str,
                           env: dict) -> dict:
    """The installed plugin directory against the plugin tree at ``commit``."""
    if not installed or not Path(installed).is_dir():
        return {"compared": False, "reason": f"no installed directory at {installed}"}
    expected = _git_tree_blobs(bare, commit, GIT_PLUGIN_PATH, env)
    actual = _disk_tree_blobs(Path(installed), env)
    differing = sorted(rel for rel in set(expected) & set(actual)
                       if expected[rel][1] != actual[rel][1])
    mode_differs = sorted(rel for rel in set(expected) & set(actual)
                          if expected[rel][1] == actual[rel][1]
                          and expected[rel][0] != actual[rel][0])
    parts = {part for rel in actual for part in Path(rel).parts}
    return {
        "compared": True, "commit": commit,
        "expected_files": len(expected), "installed_files": len(actual),
        "missing": sorted(set(expected) - set(actual))[:50],
        "extra": sorted(set(actual) - set(expected))[:50],
        "differing": differing[:50], "mode_differs": mode_differs[:50],
        "matches": not (set(expected) ^ set(actual)) and not differing,
        "has_git_dir": ".git" in parts,
        "has_plans": "plans" in parts,
        "next_marker_present": GIT_NEXT_MARKER in actual,
    }


def _git_snapshot(home: Path) -> dict:
    """What codex checked out as the marketplace root, read off its own clone."""
    snapshot = home / ".codex" / ".tmp" / "marketplaces" / CODEX_MARKETPLACE
    env = _git_fixture_env(home)
    if not (snapshot / ".git").exists():
        return {"root": str(snapshot), "exists": snapshot.is_dir(), "head": None}
    head = _git(["rev-parse", "HEAD"], snapshot, env, check=False)
    sparse = _git(["sparse-checkout", "list"], snapshot, env, check=False)
    top = sorted(p.name for p in snapshot.iterdir() if p.name != ".git")
    return {"root": str(snapshot), "exists": True,
            "head": head["stdout"].strip() or None,
            "sparse_list": sparse["stdout"].split(),
            "top_level": top,
            "decoys_present": sorted(rel for rel in GIT_DECOYS
                                     if (snapshot / rel).exists())}


def _marketplace_table_text(text: str) -> str | None:
    """The `[marketplaces.beastlabai]` table, header to the next header."""
    lines = text.splitlines()
    header = f"[marketplaces.{CODEX_MARKETPLACE}]"
    for index, line in enumerate(lines):
        if line.split("#", 1)[0].strip() == header:
            block = [line]
            for follow in lines[index + 1:]:
                if follow.split("#", 1)[0].strip().startswith("["):
                    break
                block.append(follow)
            return "\n".join(block).rstrip() + "\n"
    return None


def _list_entry(document) -> dict | None:
    """The multi-llm entry of a `codex plugin list --json` document."""
    entries = (document if isinstance(document, list)
               else (document or {}).get("installed") if isinstance(document, dict)
               else None)
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("pluginId") == CODEX_PLUGIN_ID:
            return entry
    return None


def _installed_digest(root: Path | None) -> str | None:
    path = Path(root) / "dist" / "tree.sha256" if root else None
    if path and path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def _git_install_facts(home: Path, fixture: dict, root: Path | None) -> dict:
    """The Git half of plugin-install.json, taken before anything ran."""
    env = _git_fixture_env(home)
    config = home / ".codex" / "config.toml"
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    public = {k: v for k, v in fixture.items() if k != "env"}
    return {
        "fixture": public,
        "snapshot": _git_snapshot(home),
        "marketplace_table": _marketplace_table_text(text),
        "tree": compare_tree_to_commit(root, Path(fixture["bare"]),
                                       fixture["release_sha"], env),
    }


def git_route_legs(home: Path, root: Path | None, overlay: dict, stub_log: Path,
                   install: dict | None, status_call: dict) -> dict:
    """A4 row 6b after the Git install: discovery, the tool, then a promotion.

    In order — the loader rows and the tool call run against the RELEASE
    install, before anything moves; then `release` is fast-forwarded to the
    default branch exactly as the promote job moves it, and the documented
    upgrade sequence (`marketplace upgrade`, `plugin add`) is graded on the
    tree it leaves. Every leg is recorded; the grader decides.
    """
    git = (install or {}).get("git") or {}
    fixture = git.get("fixture") or {}
    env = _codex_env(home, overlay)
    git_env = _git_fixture_env(home)
    record: dict = {"fixture": fixture}

    mcp_get = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
    record["mcp_get"] = {"command": _public(mcp_get),
                         "document": _json_or_none(mcp_get["_stdout_full"])}
    prompt = _run_codex(["debug", "prompt-input"], env)
    record["prompt_input"] = {"command": _public(prompt),
                              "skill": _parse_prompt_input(prompt["_stdout_full"])}
    record["exec"] = codex_exec_leg(home, overlay, stub_log, "git-route")
    stdio_env = {**os.environ, "PWD": str(WORKSPACE)}
    record["stdio"] = stdio_session(root, [status_call], env=stdio_env) if root else None
    listed = _run_codex(["plugin", "marketplace", "list", "--json"], env)
    record["marketplace_list"] = {"command": _public(listed),
                                  "document": _json_or_none(listed["_stdout_full"])}

    # Re-adding the registered marketplace at another ref: the publish plan's
    # step 5 assumption, measured rather than assumed.
    config = home / ".codex" / "config.toml"
    re_add = _run_codex(["plugin", "marketplace", "add", fixture.get("url", ""),
                         *_git_marketplace_flags(GIT_DEFAULT_BRANCH), "--json"], env)
    record["re_add_other_ref"] = {
        "command": _public(re_add),
        "marketplace_table_after": _marketplace_table_text(
            config.read_text(encoding="utf-8") if config.is_file() else ""),
    }

    # The promotion: a fast-forward of `release`, as the promote job pushes it.
    bare = Path(fixture.get("bare", ""))
    if not bare.is_dir():
        record["legs_not_driven"] = {"refresh": "no fixture bare repository recorded"}
        return record
    ancestor = _git(["merge-base", "--is-ancestor", fixture["release_sha"],
                     fixture["default_sha"]], bare, git_env, check=False)
    moved = _git(["update-ref", f"refs/heads/{GIT_RELEASE_REF}", fixture["default_sha"],
                  fixture["release_sha"]], bare, git_env, check=False)
    upgrade = _run_codex(["plugin", "marketplace", "upgrade", CODEX_MARKETPLACE,
                          "--json"], env)
    cache = home / ".codex" / "plugins" / "cache" / CODEX_MARKETPLACE / "multi-llm"
    versions_after_upgrade = sorted(p.name for p in cache.glob("*") if p.is_dir())
    add = _run_codex(["plugin", "add", CODEX_PLUGIN_ID, "--json"], env)
    add_doc = _json_or_none(add["_stdout_full"])
    refreshed = codex_installed_root(home, {"installedPath": _find_key(add_doc,
                                                                       "installedPath")})
    after_list = _run_codex(["plugin", "list", "--json"], env)
    record["refresh"] = {
        "fast_forward": ancestor["exit"] == 0,
        "update_ref": moved,
        "upgrade": {"command": _public(upgrade),
                    "document": _json_or_none(upgrade["_stdout_full"])},
        "snapshot_after_upgrade": _git_snapshot(home),
        "versions_after_upgrade": versions_after_upgrade,
        "add": {"command": _public(add), "document": add_doc},
        "installed_root": str(refreshed) if refreshed else None,
        "versions_after_add": sorted(p.name for p in cache.glob("*") if p.is_dir()),
        "tree": compare_tree_to_commit(refreshed, bare, fixture["default_sha"], git_env),
        "list_entry": _list_entry(_json_or_none(after_list["_stdout_full"])),
        "stdio": (stdio_session(refreshed, [status_call], env=stdio_env)
                  if refreshed else None),
    }
    return record


def write_profile_credential(home: Path, key: str) -> Path:
    """`export OPENROUTER_API_KEY=…` into ``$HOME/.profile``, at run time (D3a).

    The SECOND carrier of the key, never a replacement for the first: the codex
    session still reads it from its own process environment (`run_host`'s
    `dict(os.environ)`), and without that no tool call is ever emitted. This one
    is for the MCP server's login-shell child, which codex's `env_vars`
    allowlist does not carry — so a credentialed package row is a live
    measurement of D3a rather than a bypass of it.

    `export`, not a bare assignment: the dispatcher's own children (a provider
    CLI, a nested codex whose auth is a non-login `sh -c`) read no startup file
    and inherit through the environment or not at all.

    THEN THE PRECEDENCE ASSERTION — D3a's second harness trap. `bash -l` reads
    only the FIRST that exists of `~/.bash_profile`, `~/.bash_login`,
    `~/.profile`; a higher-precedence file anywhere in this home makes the
    export silently unread and every credentialed row fail on a missing key, a
    reading indistinguishable from a genuine D3a regression. So the row fails
    here, naming the file, instead.

    The tmpfs home, never `home_seed/codex/.codex/.profile` and never a
    committed file. Mode 0600. The value is never logged.

    NOT THE CARRIER FOR `MULTI_LLM_CODEX_UNSANDBOXED_WRITES` (D3b). The MCP
    server reads that lever from its OWN environment (D9 layer 1) at call entry,
    before any spawn, so a `.profile` export — read only by the login-shell
    child, layer 4 — can never lift the write refusal. An apply row (Phase 2's
    `codex-package-apply-executor`) must put the name in codex's own process
    environment — `run_host`'s env — from which the `.mcp.json` allowlist
    forwards it to the server. Nothing exports it today; a row that relies on
    this file for it is refused and reads as measured-red.
    """
    profile = home / ".profile"
    existing = profile.read_text(encoding="utf-8") if profile.is_file() else ""
    line = f"export OPENROUTER_API_KEY={shlex.quote(key)}\n"
    profile.write_text(existing + ("" if not existing or existing.endswith("\n")
                                   else "\n") + line, encoding="utf-8")
    os.chmod(profile, 0o600)
    assert_profile_is_read(home)
    # The NAME and the placement, never the value: what the grader needs to see
    # is that the export was written as an `export`, into the file a login shell
    # reads, with nothing shadowing it.
    (OUT / "profile-export.json").write_text(json.dumps({
        "path": str(profile),
        "exported": ["OPENROUTER_API_KEY"],
        "form": "export",
        "precedence_checked": [str(home / ".bash_profile"),
                               str(home / ".bash_login")],
        "shadowed_by": None,
    }, indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote the credential export into {profile} (value not logged); "
        f"no higher-precedence bash startup file in {home}")
    return profile


def assert_profile_is_read(home: Path) -> None:
    """Fail the row if a login shell would read something other than .profile."""
    for name in (".bash_profile", ".bash_login"):
        shadow = home / name
        if shadow.exists():
            raise SystemExit(
                f"entrypoint: {shadow} exists, so a `bash -l` login shell reads "
                f"it and never reaches {home / '.profile'} — the credential "
                f"export (and the environment probe's markers) would be silently "
                f"unread (plans/codex-native-plugin.md D3a, the startup-file "
                f"precedence trap). Remove it from the seed.")


#: The rig's own instrumentation names `wrapper.py` reads at dispatch time.
#: `main` sets them in the row's process environment, which a FILE-route host
#: passes down whole; codex's `.mcp.json` `env_vars` allowlist does not, so on
#: the package route the dispatcher ran with no receipt dir, no call log and no
#: row id — measured on `runs/20260923T231001Z`: the ask reached `done`, 5/5
#: stub answers, and zero receipts or call records to grade it by.
HARNESS_PROFILE_NAMES = ("MLLM_ROW_ID", "MLLM_STUB_PORT", "MLLM_CALL_LOG",
                         "MLLM_RECEIPT_DIR")


def export_harness_names_into_profile(home: Path) -> None:
    """Carry the rig's instrumentation to the MCP server's login-shell child.

    The same second carrier the key uses (D3a): the child's `bash -l` reads
    `.profile`. Harness plumbing only — no product name is exported here, so
    what the server itself receives stays exactly what codex's allowlist gives it.
    """
    profile = home / ".profile"
    body = profile.read_text(encoding="utf-8") if profile.is_file() else ""
    lines = "".join(f"export {name}={shlex.quote(os.environ[name])}\n"
                    for name in HARNESS_PROFILE_NAMES if os.environ.get(name))
    profile.write_text(body + ("" if not body or body.endswith("\n") else "\n")
                       + lines, encoding="utf-8")
    os.chmod(profile, 0o600)
    assert_profile_is_read(home)


#: Where every login shell that reads the seeded `.profile` files one line.
PROFILE_READS = "profile-reads.jsonl"


def wrap_profile_with_read_probe(home: Path) -> None:
    """Bracket the `.profile` export with a presence probe (item 16, D3a).

    Item 16 asks whether the MCP server's login-shell child derived the key
    FROM THE PROFILE rather than from the session's environment. So every
    shell that reads `.profile` records — as booleans, never the value —
    whether `OPENROUTER_API_KEY` was already set before the file ran and
    whether it is set after, plus its parent's argv, so the grader can pick
    out the shell whose parent is `mcp/server.py`. Other login shells (codex's
    own `exec_command`) land here too and are told apart by that parent.
    A separate step from `write_profile_credential` so that function's file
    stays exactly the one-line export its plain-CI tests pin.
    """
    profile = home / ".profile"
    body = profile.read_text(encoding="utf-8") if profile.is_file() else ""
    log_path = shlex.quote(str(OUT / PROFILE_READS))
    pre = '_mllm_pre=false; [ -n "${OPENROUTER_API_KEY:-}" ] && _mllm_pre=true\n'
    post = (
        '_mllm_post=false; [ -n "${OPENROUTER_API_KEY:-}" ] && _mllm_post=true\n'
        "_mllm_parent=$(tr '\\000' ' ' < /proc/$PPID/cmdline 2>/dev/null "
        "| tr -d '\"\\\\[:cntrl:]' | cut -c1-200)\n"
        "printf '{\"pid\":%s,\"ppid\":%s,\"key_before_profile\":%s,"
        "\"key_after_profile\":%s,\"parent\":\"%s\"}\\n' "
        '"$$" "$PPID" "$_mllm_pre" "$_mllm_post" "$_mllm_parent" '
        f">> {log_path} 2>/dev/null\n"
        "unset _mllm_pre _mllm_post _mllm_parent\n")
    profile.write_text(pre + body + post, encoding="utf-8")
    os.chmod(profile, 0o600)


# -- the probes ------------------------------------------------------------- #

#: A prompt for the stub-driven exec legs. Its words do not matter — the stub's
#: scripted scenario answers whatever the prompt says — and it names the tool so
#: a transcript read by a human says what the turn was for.
CODEX_PROBE_PROMPT = "Call the multi_llm tool."


def _stub_entries(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def codex_exec_leg(home: Path, overlay: dict, stub_log: Path, label: str,
                   extra_args: tuple[str, ...] = (), env_extra: dict | None = None,
                   interrupt_after_call: bool = False, timeout: float = 300.0,
                   cwd: Path | None = None, env_drop: tuple[str, ...] = (),
                   tap: bool = False) -> dict:
    """One `codex exec --json` turn against the stub, bracketed in its log.

    The stub's scripted scenario (`scripted_codex_plugin_tool_call.yaml`) serves the
    call; this records the event stream and the stub entries the turn made —
    the `tools[]` declaration codex built, the frames it was handed, and the
    output it sent back. `interrupt_after_call` sends SIGINT once the stub has
    served the call, for the cancellation leg (row 10(b)).
    """
    mark = len(_stub_entries(stub_log))
    env = _codex_env(home, overlay)
    # The launching shell's own `PWD`, as any interactive shell exports it: the
    # entrypoint is exec'd by docker with no shell, so it has none to inherit,
    # and D3f's env rung (measured: codex advertises no `roots`) reads exactly
    # this name. `env_drop=("PWD",)` is the leg that measures its absence.
    env["PWD"] = str(cwd or WORKSPACE)
    env.update(env_extra or {})
    for name in env_drop:
        env.pop(name, None)
    tap_log = None
    if tap:
        tap_dir, tap_log = install_mcp_tap(home, label)
        env["PATH"] = f"{tap_dir}:{env.get('PATH') or os.defpath}"
    argv = [str(REAL_DIR / "codex"), "exec", "--json", "--skip-git-repo-check",
            *extra_args, "--model", "stub-model", CODEX_PROBE_PROMPT]
    installed = codex_installed_root(home)
    pycache_before = _pycache_state(installed)
    started = time.time()
    proc = subprocess.Popen(argv, env=env, cwd=str(cwd or WORKSPACE), text=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    interrupted = None
    if interrupt_after_call:
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            served = [e for e in _stub_entries(stub_log)[mark:]
                      if (e.get("scripted") or {}).get("phase") == "call"]
            if served:
                time.sleep(0.5)
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                interrupted = round(time.time() - started, 2)
                break
            time.sleep(0.2)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    events_file = OUT / f"codex-exec-{label}.jsonl"
    events_file.write_text(out, encoding="utf-8")
    events = [e for e in (_json_or_none(line) for line in out.splitlines())
              if isinstance(e, dict)]
    entries = _stub_entries(stub_log)[mark:]
    scripted = [e.get("scripted") for e in entries if e.get("scripted")]
    return {
        "label": label,
        "argv": ["codex", *argv[1:]],
        "cwd": str(cwd or WORKSPACE),
        "env_dropped": list(env_drop),
        "exit": proc.returncode,
        "timed_out": timed_out,
        "interrupted_at_s": interrupted,
        "wall_clock_s": round(time.time() - started, 2),
        "events_file": events_file.name,
        "event_types": sorted({str(e.get("type")) for e in events}),
        "item_types": sorted({str((e.get("item") or {}).get("type"))
                              for e in events if e.get("item")}),
        "stderr": err[:CODEX_OUTPUT_CAP],
        "stub_requests": len(entries),
        "scripted_phases": [s.get("phase") for s in scripted],
        "declarations": [s.get("declaration") for s in scripted
                         if s.get("declaration") is not None][:1],
        "returned_output": next((s.get("output") for s in scripted
                                 if s.get("phase") == "terminal"), None),
        "envelope": _envelope_summary(next((s.get("output") for s in scripted
                                            if s.get("phase") == "terminal"),
                                           None)),
        "tap": summarize_mcp_tap(tap_log) if tap_log else None,
        # PYTHONDONTWRITEBYTECODE (fix 4, 2026-09-24): every `__pycache__`
        # entry under the installed plugin that this turn added or rewrote.
        "plugin_pycache_writes": _pycache_writes(pycache_before,
                                                 _pycache_state(installed)),
    }


def _pycache_state(root: Path | None) -> dict:
    """``{relative path: (size, mtime_ns)}`` for every file under a
    ``__pycache__`` directory of the installed plugin tree."""
    out: dict = {}
    if not root or not root.is_dir():
        return out
    for path in root.rglob("__pycache__/*"):
        try:
            stat = path.stat()
        except OSError:
            continue
        out[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return out


def _pycache_writes(before: dict, after: dict) -> list[str]:
    return sorted(k for k in after if before.get(k) != after[k])


# -- the stdio tap ----------------------------------------------------------- #
#
# The installed `.mcp.json` launches `python3 ./mcp/server.py`, resolved on
# codex's PATH. A tap leg puts a `python3` shim first on that PATH: for the
# server's own argv it records the server's launch cwd, environment NAMES (plus
# the two path-valued names the anchor question is about, `PWD` and
# `CODEX_HOME`), and every JSON-RPC line in each direction, then runs the real
# interpreter as its child in the same cwd; for any other argv it `exec`s the
# real interpreter untouched. So the installed tree stays byte-identical
# (`divergence: null` holds) and what codex actually sent — its `initialize`
# capabilities, its answer to `roots/list`, a `notifications/cancelled` or a
# bare EOF — is read off the wire instead of inferred from the server's reply.

_MCP_TAP = """\
import json, os, signal, subprocess, sys, threading, time
REAL = {real!r}
LOG = {log!r}
args = sys.argv[1:]
if not (args and args[0].replace(os.sep, "/").endswith("mcp/server.py")):
    os.execv(REAL, [REAL] + args)
lock = threading.Lock()
def rec(**row):
    row["t"] = round(time.time(), 3)
    with lock:
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\\n")
def parse(line):
    try:
        return json.loads(line)
    except ValueError:
        return {{"unparsed": line[:500].decode("utf-8", "replace")}}
rec(event="start", argv=args, cwd=os.getcwd(), env_names=sorted(os.environ),
    PWD=os.environ.get("PWD"), CODEX_HOME=os.environ.get("CODEX_HOME"),
    # Values only for the planted probe names (row 11(b)'s `MLLM_PD_*`) and for
    # any name or value spelling PLUGIN_DATA: none of them is a credential.
    env_values={{k: v for k, v in os.environ.items()
                 if k.startswith("MLLM_PD") or "PLUGIN_DATA" in k
                 or "PLUGIN_DATA" in v or k == "PYTHONDONTWRITEBYTECODE"}})
child = subprocess.Popen([REAL] + args, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, bufsize=0)
def on_signal(number, _frame):
    rec(event="signal", signal=number)
    try:
        child.send_signal(number)
    except OSError:
        pass
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(number, on_signal)
def pump_in():
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        rec(event="c2s", msg=parse(line))
        try:
            child.stdin.write(line)
            child.stdin.flush()
        except OSError:
            break
    rec(event="stdin-eof")
    try:
        child.stdin.close()
    except OSError:
        pass
def pump_out():
    while True:
        line = child.stdout.readline()
        if not line:
            break
        rec(event="s2c", msg=parse(line))
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
threading.Thread(target=pump_in, daemon=True).start()
out = threading.Thread(target=pump_out, daemon=True)
out.start()
code = child.wait()
out.join(5)
rec(event="exit", code=code)
sys.exit(code)
"""


def install_mcp_tap(_home: Path, label: str) -> tuple[Path, Path]:
    """Write the `python3` shim for one leg; return (its dir, its log)."""
    real = shutil.which("python3") or sys.executable
    # Under the bundle, not the home: the container's tmpfs mounts (the home
    # included) are `noexec`, and a shim codex cannot execute is skipped for
    # the next `python3` on PATH — measured: the server ran, the tap never did.
    tap_dir = OUT / "mcp-tap" / label
    tap_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"mcp-tap-{label}.jsonl"
    shim = tap_dir / "python3"
    shim.write_text(f"#!{real}\n" + _MCP_TAP.format(real=real, log=str(log_path)),
                    encoding="utf-8")
    shim.chmod(0o755)
    return tap_dir, log_path


def summarize_mcp_tap(log_path: Path) -> dict:
    """What crossed the wire, read off one tap log."""
    rows = _stub_entries(log_path)
    starts = [r for r in rows if r.get("event") == "start"]
    c2s = [r.get("msg") or {} for r in rows if r.get("event") == "c2s"]
    s2c = [r.get("msg") or {} for r in rows if r.get("event") == "s2c"]
    initialize = next((m for m in c2s if m.get("method") == "initialize"), None)
    capabilities = ((initialize or {}).get("params") or {}).get("capabilities")
    asked = [m for m in s2c if "method" in m and "id" in m]
    answers = {m.get("id"): m for m in c2s if "id" in m and "method" not in m}
    calls = [m for m in c2s if m.get("method") == "tools/call"]
    eof = next((r for r in rows if r.get("event") == "stdin-eof"), None)
    return {
        "log": log_path.name,
        "tapped": bool(starts),
        "launches": [{k: s.get(k) for k in ("argv", "cwd", "PWD", "CODEX_HOME",
                                            "env_names", "env_values")}
                     for s in starts],
        # What codex put in each `tools/call`'s `_meta["x-codex-turn-metadata"]`:
        # the workspaces (D3f's rung 2) and the sandbox posture (D3b).
        "turn_metadata": [
            {k: (((m.get("params") or {}).get("_meta") or {})
                 .get("x-codex-turn-metadata") or {}).get(k)
             for k in ("workspaces", "sandbox_mode", "sandbox")}
            for m in calls],
        "progress_tokens": [((m.get("params") or {}).get("_meta") or {})
                            .get("progressToken") for m in calls],
        "log_notification_texts": [
            ((m.get("params") or {}).get("data")) for m in s2c
            if m.get("method") == "notifications/message"][:50],
        "initialize_params": (initialize or {}).get("params"),
        "roots_advertised": isinstance(capabilities, dict)
        and isinstance(capabilities.get("roots"), dict),
        "server_requests": [{"method": m.get("method"), "id": m.get("id"),
                             "answer": answers.get(m.get("id"))} for m in asked],
        "client_methods": [m.get("method") for m in c2s if m.get("method")],
        "tool_call_ids": [m.get("id") for m in calls],
        "cancelled": [m.get("params") for m in c2s
                      if m.get("method") == "notifications/cancelled"],
        "progress_notifications": sum(
            1 for m in s2c if m.get("method") == "notifications/progress"),
        "log_notifications": sum(
            1 for m in s2c if m.get("method") == "notifications/message"),
        "first_call_t": next((r.get("t") for r in rows if r.get("event") == "c2s"
                              and (r.get("msg") or {}).get("method")
                              == "tools/call"), None),
        "stdin_eof_t": (eof or {}).get("t"),
        "signals": [r.get("signal") for r in rows if r.get("event") == "signal"],
        "exit": next((r.get("code") for r in rows if r.get("event") == "exit"),
                     None),
        "call_answered": any("method" not in m and m.get("id") in
                             {c.get("id") for c in calls} for m in s2c),
    }


def _envelope_summary(output) -> dict | None:
    envelope = _envelope_of(output)
    if not envelope:
        return None
    return {k: envelope.get(k) for k in ("ok", "state", "exit", "anchor",
                                         "anchor_source", "message",
                                         "environment", "relays")}


def _tree_hashes(root: Path) -> dict:
    """``{relative path: sha256}`` for every regular file under ``root``."""
    import hashlib
    out = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                out[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
            except OSError:
                continue
    return out


def _tree_diff(before: dict, after: dict) -> dict:
    return {"added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after)
                              if before[k] != after[k])}


class McpStdioClient:
    """A minimal JSON-RPC-over-stdio client for the installed server.

    Launches the server exactly as the installed `.mcp.json` declares it
    (`command`, `args`, `cwd` relative to the installed root), in the
    HARNESS's environment — which is why every reading taken through it is the
    CONTROL and never the graded one (item 9: the stdio client hands the
    server the harness environment, so both D3a populations can come back green
    with no forwarding and no login shell). Server-initiated requests
    (`roots/list`, …) are recorded, never answered unless `roots` is given.
    """

    def __init__(self, root: Path, env: dict, roots: list[str] | None = None,
                 command: str | None = None):
        declared = _mcp_declaration(root) or {}
        command = command or str(declared.get("command") or "python3")
        args = [str(a) for a in (declared.get("args") or ["./mcp/server.py"])]
        cwd = root / str(declared.get("cwd") or ".")
        self.argv = [command, *args]
        self.roots = roots
        self.server_requests: list[dict] = []
        self.notifications: list[dict] = []
        self.proc = subprocess.Popen(
            self.argv, cwd=str(cwd), env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._next = 0
        # A reader thread owns stdout, never select() on its fd: a text-mode
        # readline() pulls a whole kernel chunk into Python's buffer, so a
        # response that arrives in the same read as a progress/log
        # notification sits buffered while the fd never reports readable
        # again, and a select-gated loop spins to its deadline with the answer
        # already in hand. `None` on the queue is EOF.
        self._lines: queue.Queue = queue.Queue()
        self._eof = False
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def request(self, method: str, params: dict | None = None,
                timeout: float = 180.0) -> dict:
        self._next += 1
        ident = self._next
        self._send({"jsonrpc": "2.0", "id": ident, "method": method,
                    **({"params": params} if params is not None else {})})
        deadline = time.time() + timeout
        while not self._eof and time.time() < deadline:
            try:
                line = self._lines.get(
                    timeout=max(0.0, min(1.0, deadline - time.time())))
            except queue.Empty:
                continue
            if line is None:
                self._eof = True
                break
            message = _json_or_none(line)
            if not isinstance(message, dict):
                continue
            if message.get("id") == ident and "method" not in message:
                return message
            if "method" in message and "id" in message:
                self.server_requests.append(message)
                if message["method"] == "roots/list" and self.roots is not None:
                    self._send({"jsonrpc": "2.0", "id": message["id"],
                                "result": {"roots": [
                                    {"uri": f"file://{r}", "name": Path(r).name}
                                    for r in self.roots]}})
            elif "method" in message:
                self.notifications.append(message)
        return {"id": ident, "error": {"code": "harness-timeout",
                                       "message": f"no answer to {method} "
                                                  f"within {timeout}s"}}

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method,
                    **({"params": params} if params is not None else {})})

    def _send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self) -> dict:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._reader.join(timeout=5)
        err = self.proc.stderr.read() if self.proc.stderr else ""
        return {"argv": self.argv, "exit": self.proc.returncode,
                "stderr": (err or "")[:CODEX_OUTPUT_CAP],
                "server_requests": self.server_requests,
                "notifications": self.notifications[:200]}


def _mcp_declaration(root: Path) -> dict | None:
    path = root / ".mcp.json"
    doc = _json_or_none(path.read_text(encoding="utf-8")) if path.is_file() else None
    servers = (doc or {}).get("mcpServers") or {}
    entry = servers.get(CODEX_MCP_SERVER)
    return entry if isinstance(entry, dict) else None


def stdio_session(root: Path, calls: list[dict], env: dict | None = None,
                  roots: list[str] | None = None,
                  capabilities: dict | None = None,
                  command: str | None = None) -> dict:
    """initialize → initialized → tools/list → each `tools/call`, recorded.

    ``command`` replaces the declared interpreter (row 9(b) launches the server
    by absolute path because its PATH deliberately resolves nothing)."""
    client = McpStdioClient(root, env or dict(os.environ), roots=roots,
                            command=command)
    initialize = client.request("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": capabilities or {},
        "clientInfo": {"name": "mllm-container-probe", "version": "0"}})
    client.notify("notifications/initialized")
    tools = client.request("tools/list")
    results = [client.request("tools/call",
                              {"name": CODEX_MCP_SERVER, "arguments": call})
               for call in calls]
    return {"initialize": initialize, "tools_list": tools,
            "calls": [{"arguments": c, "response": r}
                      for c, r in zip(calls, results)],
            "server": client.close()}


#: The probe table: `--probe <name>` → what the row records. Each writes
#: `probe-<name>.json`; the ones whose answer is only RECORDED (7(a), 9, 10,
#: 11, 12(a), 12(d), 13, 14 — the plan's item 15) are done when the file
#: exists, and the grader never asserts a particular answer on them.
CODEX_PACKAGE_PROBES = (
    "seed", "floor", "features", "scripted-call",
    "loader-mcp-get", "loader-prompt-input", "loader-tools",
    "anchor", "runnability", "environment", "progress-cancellation",
    "plugin-data", "sandbox-posture", "overlay", "timeout",
    # The release gate's two (plans/codex-native-plugin.md A4): row 6b's
    # offline Git install, whose wiring step installs through Git instead of
    # the local path, and row 1's search for an upstream validator.
    "git-route", "upstream-validator",
)

#: The one probe whose wiring step installs through the Git route.
GIT_ROUTE_PROBE = "git-route"

#: Markers for the environment probe (D3a, row 9(a)). Presence is asserted,
#: never a value — the dummy credential is a NAME whose arrival is the reading.
PROBE_LAUNCH_MARKER = "MULTI_LLM_PROBE"
PROBE_OVERRIDE_MARKER = "MULTI_LLM_PROBE_OVERRIDE"
PROBE_PROFILE_CREDENTIAL = "MLLM_PROBE_DUMMY_API_KEY"
PROBE_PROFILE_PATH = "/opt/mllm-probe-profile-marker/bin"
PROBE_BASHRC_MARKER = "MLLM_PROBE_BASHRC_ONLY"
#: How long the cancellation leg holds the tool call in flight (row 10(b)).
PROBE_STALL_S = 30

#: Every planted NAME the child observations report on (never a value). The
#: profile's PATH entry is reported as the pseudo-name ``PATH``.
PROBE_MARKER_NAMES = (PROBE_LAUNCH_MARKER, PROBE_OVERRIDE_MARKER,
                      PROBE_PROFILE_CREDENTIAL, PROBE_BASHRC_MARKER)
PROBE_LAUNCH_POPULATION = (PROBE_LAUNCH_MARKER, PROBE_OVERRIDE_MARKER)
PROBE_PROFILE_POPULATION = (PROBE_PROFILE_CREDENTIAL, "PATH")

#: Where the login shell's DEBUG-trap observer appends one JSON line per
#: `exec` it sees: marker presence and cwd at the login-shell depth and at a
#: shell-less child of it (the depth a provider CLI's auth command runs at).
PROBE_CHILDREN_FILE = "probe-environment-children.jsonl"

#: The shell-less child. argv: out, exec target basename, login_shell,
#: the shell's $PWD, the comma-joined names set in the shell (plus `+PATH`
#: when the profile's PATH entry is on the shell's PATH). It reads names
#: only and writes names only — no environment value is ever recorded.
_PROBE_OBSERVER = """\
import json, os, sys
NAMES = {names!r}
PATH_ENTRY = {path_entry!r}
out, target, login, shell_pwd, seen = sys.argv[1:6]
seen = set(seen.split(","))
child = [n for n in NAMES if n in os.environ]
if PATH_ENTRY in os.environ.get("PATH", "").split(":"):
    child.append("PATH")
row = {{"target": target, "login_shell": login == "true",
        "shell": {{"present": sorted(
                       [n for n in NAMES if n in seen]
                       + (["PATH"] if "+PATH" in seen else [])),
                   "cwd": shell_pwd}},
        "child": {{"present": sorted(child), "cwd": os.getcwd()}}}}
with open(out, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\\n")
"""


def _process_table() -> list[str]:
    """`pid ppid pgid cmdline` for every process in /proc (no `ps` needed)."""
    rows = []
    for entry in sorted(Path("/proc").glob("[0-9]*"), key=lambda p: int(p.name)):
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            args = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
        except (OSError, IndexError):
            continue
        rows.append(f"{entry.name} {stat[1]} {stat[2]} {args}")
    return rows


def _probe_profile_hook(observer: Path, python: str, out: Path) -> str:
    """The `.profile` tail that reports what the login-shell child received.

    The server's constant `-c` string is `cd -- "$1" && shift && exec "$@"`,
    so a DEBUG trap set in the profile fires before that `exec` with the cd
    already done: `$PWD` is the anchor the shell moved to, the names set in
    the shell are the login-shell population, and a child spawned from the
    trap (no shell in between) sees only what was EXPORTED — the grandchild
    depth D3a's missing-`export` red lives at. The shell hands the child the
    NAMES it has set (`compgen -v`, names only, never a value) and the child
    keeps only the planted ones — so no marker name, the `.bashrc`-only one
    included, is ever spelled in the profile. bash-only and guarded, so a
    POSIX `sh -l` reading the same file parses nothing it cannot run; every
    other command is filtered out on `$BASH_COMMAND`.
    """
    argv = " ".join(shlex.quote(str(a)) for a in (python, observer, out))
    return (
        'if [ -n "${BASH_VERSION-}" ]; then\n'
        "_mllm_probe_observe() {\n"
        '  case "$BASH_COMMAND" in exec\\ *) ;; *) return 0 ;; esac\n'
        '  _mllm_seen=$(compgen -v | tr "\\n" ,)\n'
        f'  case ":$PATH:" in *":{PROBE_PROFILE_PATH}:"*) '
        '_mllm_seen="$_mllm_seen,+PATH" ;; esac\n'
        "  _mllm_login=false; shopt -q login_shell && _mllm_login=true\n"
        f'  {argv} "${{1##*/}}" "$_mllm_login" "$PWD" "$_mllm_seen" '
        ">/dev/null 2>&1 || true\n"
        "}\n"
        "trap '_mllm_probe_observe \"$@\"' DEBUG\n"
        "fi\n")


def plant_environment_markers(home: Path) -> dict:
    """Plant D3a's markers: launch-env only, `.profile` (export), `.bashrc`.

    The `.bashrc` marker is planted to be ABSENT — a non-interactive login
    shell does not read it, and a green there would mean the startup chain is
    not the one D3a names. The `.profile` PATH entry doubles as evidence the
    file was read (the precedence assertion runs straight after). The profile
    also carries the observer hook (`_probe_profile_hook`), so the probe reads
    which markers the dispatcher's login shell and its shell-less child
    actually received, and their cwd, into `PROBE_CHILDREN_FILE`.
    """
    observer = home / ".mllm-probe-observer.py"
    observer.write_text(_PROBE_OBSERVER.format(
        names=list(PROBE_MARKER_NAMES), path_entry=PROBE_PROFILE_PATH),
        encoding="utf-8")
    python = shutil.which("python3") or sys.executable
    profile = home / ".profile"
    existing = profile.read_text(encoding="utf-8") if profile.is_file() else ""
    profile.write_text(
        existing
        + f"export {PROBE_PROFILE_CREDENTIAL}=planted-profile-credential\n"
        + f"export PATH={PROBE_PROFILE_PATH}:$PATH\n"
        + _probe_profile_hook(observer, python, OUT / PROBE_CHILDREN_FILE),
        encoding="utf-8")
    (home / ".bashrc").write_text(
        f"export {PROBE_BASHRC_MARKER}=planted-bashrc-only\n", encoding="utf-8")
    assert_profile_is_read(home)
    return {PROBE_LAUNCH_MARKER: "planted-launch-value",
            PROBE_OVERRIDE_MARKER: "planted-override-value"}


def _child_observations() -> list[dict]:
    return _stub_entries(OUT / PROBE_CHILDREN_FILE)


def _envelope_of(value):
    """The server's envelope (``ok``/``state``/``exit``/…) under any wrapping.

    Codex hands a tool result back to the model as text headed
    ``Wall time: <s> seconds\nOutput:\n`` (measured at 0.156.1), so a string
    that is not JSON is retried from its first ``{``.
    """
    if isinstance(value, str):
        text = value
        value = _json_or_none(text)
        if value is None and "{" in text:
            value = _json_or_none(text[text.index("{"):])
    if isinstance(value, list):
        for item in value:
            found = _envelope_of(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    if "ok" in value and "state" in value and "exit" in value:
        return value
    for key in ("result", "content", "text", "output"):
        if key in value:
            found = _envelope_of(value[key])
            if found is not None:
                return found
    return None


def grade_child_observations(rows: list[dict], anchor) -> dict:
    """Grade what the dispatcher's two depths received (D3a's red ladder).

    Only the row whose `exec` target is `uv` is the dispatcher; any other
    login shell reading the same profile is kept but not graded. Names and
    paths only — the observer never carries a value.
    """
    dispatcher = [r for r in rows
                  if str(r.get("target") or "").startswith("uv")]
    graded: dict = {"observations": rows, "anchor": anchor,
                    "dispatcher_observed": bool(dispatcher)}
    if not dispatcher:
        graded["reading"] = "not-observed"
        graded["why"] = ("no login shell exec'd the dispatcher, so no child "
                         "received anything to observe")
        return graded
    row = dispatcher[0]
    shell = set((row.get("shell") or {}).get("present") or [])
    child = set((row.get("child") or {}).get("present") or [])
    profile = set(PROBE_PROFILE_POPULATION)
    graded["login_shell"] = bool(row.get("login_shell"))
    graded["launch_population"] = {
        "shell": sorted(shell & set(PROBE_LAUNCH_POPULATION)),
        "child": sorted(child & set(PROBE_LAUNCH_POPULATION))}
    graded["profile_population"] = {"shell": sorted(shell & profile),
                                    "child": sorted(child & profile)}
    graded["bashrc_marker_seen"] = PROBE_BASHRC_MARKER in (shell | child)
    cwds = {"shell": (row.get("shell") or {}).get("cwd"),
            "child": (row.get("child") or {}).get("cwd")}
    graded["cwd"] = cwds
    graded["cwd_is_anchor"] = (
        None if not anchor else
        all(c and os.path.realpath(c) == os.path.realpath(str(anchor))
            for c in cwds.values()))
    if graded["bashrc_marker_seen"]:
        reading = "red-bashrc-read"
    elif profile <= shell and not profile <= child:
        reading = "red-grandchild-only"
    elif not profile <= shell:
        reading = ("red-profile-population" if graded["login_shell"]
                   else "red-no-login-shell")
    elif graded["cwd_is_anchor"] is False:
        reading = "red-cwd-not-anchor"
    else:
        reading = "green"
    graded["reading"] = reading
    return graded


def _codex_version(home: Path) -> dict:
    result = _run_codex(["--version"], _codex_env(home), timeout=60)
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", result["stdout"] or "")
    return {"raw": (result["stdout"] or "").strip(), "exit": result["exit"],
            "version": ".".join(match.groups()) if match else None}


def _parse_features(text: str) -> dict:
    """`codex features list` → `{name: [stage, enabled]}`, one row per line."""
    table = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] in ("true", "false"):
            table[parts[0]] = [" ".join(parts[1:-1]), parts[-1] == "true"]
    return table


def _parse_prompt_input(text: str) -> dict:
    """The `### Skill roots` table and the `multi-llm:multi-llm` locator (R8′).

    At 0.156.1 `codex debug prompt-input` prints a JSON document whose `text`
    fields carry the prompt, so every `text` string is joined back into lines
    first; a plain-text printout is read as it is.
    """
    document = _json_or_none(text or "")
    if document is not None:
        texts: list[str] = []

        def collect(node) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "text" and isinstance(value, str):
                        texts.append(value)
                    else:
                        collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(document)
        text = "\n".join(texts)
    roots: dict[str, str] = {}
    in_roots = False
    for line in (text or "").splitlines():
        if line.strip().lower().startswith("### skill roots"):
            in_roots = True
            continue
        if in_roots:
            if line.startswith("#"):
                in_roots = False
                continue
            match = re.match(r"\s*[-|*]?\s*`?([A-Za-z0-9_.\-]+)`?\s*[:=|]\s*`?([^`|]+?)`?\s*\|?\s*$",
                             line)
            if match:
                roots[match.group(1)] = match.group(2).strip()
    locator = None
    for line in (text or "").splitlines():
        if "multi-llm:multi-llm" in line:
            found = re.search(r"\(file:\s*([^)]+)\)", line)
            locator = found.group(1).strip() if found else line.strip()
            break
    resolved = None
    if locator:
        if os.path.isabs(locator):
            resolved = locator
        else:
            head, _, rest = locator.partition("/")
            if head in roots:
                resolved = os.path.join(roots[head], rest)
    return {"roots": roots, "advertised": locator is not None,
            "locator": locator, "resolved": resolved,
            "resolved_exists": bool(resolved and Path(resolved).is_file())}


def _tree_census(root: Path, limit: int = 5000) -> list[dict]:
    """Relative paths and sizes under ``root`` — names only, never contents."""
    rows = []
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*")):
        if len(rows) >= limit:
            break
        rel = path.relative_to(root).as_posix()
        if "/node_modules/" in f"/{rel}/" or "/__pycache__/" in f"/{rel}/":
            continue
        rows.append({"path": rel, "dir": path.is_dir(),
                     "size": path.stat().st_size if path.is_file() else None})
    return rows


#: Port of the second stub the anchor row's `mode: init` leg talks to (the row's
#: own stub serves the `status` scenario and cannot change its arguments).
ANCHOR_INIT_STUB_PORT = 8098


def _init_scenario(path: Path) -> Path:
    """The scripted scenario with `mode: init` arguments (JSON is YAML)."""
    path.write_text(json.dumps({
        "name": "scripted_codex_plugin_init_call",
        "description": "Phase 1 row 7(c): one scripted mode:init call",
        "scripted_tool_call": {
            "surfaces": ["responses"], "namespace": "mcp__multi_llm",
            "name": CODEX_MCP_SERVER, "call_id": "call_mllm_scripted_init",
            "arguments": {"mode": "init", "plan": "", "question": "",
                          "extra_flags": []},
            "terminal": "multi-llm init call returned."},
        "prompts": [{"pattern": "call.*multi_llm tool", "content": "OK",
                     "description": "turn without the tool"}],
        "fallback": {"content": "OK"}}, indent=2), encoding="utf-8")
    return path


def _anchor_legs(home: Path, root: Path | None, overlay: dict,
                 stub_log: Path) -> dict:
    """Row 7 (D3f): which rung answers, whether the anchor holds, and init.

    (a) tapped `codex exec` legs from the seeded workspace — with `PWD` set as
        a launching shell would, and with `PWD` removed from codex's own
        environment (does codex supply one itself?) — reading codex's
        `initialize` capabilities, any `roots/list` exchange, the server's
        launch cwd and the `PWD`/`CODEX_HOME` it received;
    (b) the same from a `git worktree` checkout, with the login-shell child's
        and its shell-less grandchild's cwd observed through the profile hook;
    (c) `mode: init` from a fresh worktree through codex (a second stub
        serving init arguments), then through the stdio client with a `roots`
        answer: `.multi-llm/providers.yaml` lands under the workspace, and
        `$CODEX_HOME` is diffed across each call.
    """
    out: dict = {}
    plant_environment_markers(home)   # the child-cwd observer (profile hook)
    codex_home = home / ".codex"

    def leg(label, cwd, env_extra=None, env_drop=(), extra_args=(), log=None):
        mark = len(_child_observations())
        result = codex_exec_leg(home, overlay, log or stub_log, label,
                                extra_args=extra_args, env_extra=env_extra,
                                env_drop=env_drop, cwd=cwd, tap=True)
        anchor = (result.get("envelope") or {}).get("anchor")
        result["children"] = grade_child_observations(
            _child_observations()[mark:], anchor)
        return result

    out["exec"] = leg("anchor", WORKSPACE, env_extra={"PWD": str(WORKSPACE)})
    out["exec_without_pwd"] = leg("anchor-no-pwd", WORKSPACE, env_drop=("PWD",))

    def worktree(name: str) -> tuple[Path, dict]:
        path = home / name
        done = subprocess.run(
            ["git", "-C", str(WORKSPACE), "worktree", "add", "-q", "--detach",
             str(path), "HEAD"], capture_output=True, text=True)
        return path, {"path": str(path), "exit": done.returncode,
                      "stderr": done.stderr[-2000:]}

    wt, out["worktree_add"] = worktree("anchor-worktree")
    out["exec_worktree"] = leg("anchor-worktree", wt, env_extra={"PWD": str(wt)})
    # The session's workspace moved by codex's own `-C` while the launching
    # shell's PWD still names the main checkout: which one does the anchor get?
    out["exec_cd_flag"] = leg("anchor-cd", WORKSPACE,
                              env_extra={"PWD": str(WORKSPACE)},
                              extra_args=("-C", str(wt)))

    init_wt, out["init_worktree_add"] = worktree("anchor-init-worktree")
    before = _tree_hashes(codex_home)
    scenario = _init_scenario(OUT / "scripted-init-scenario.yaml")
    init_log = OUT / "stub-init.jsonl"
    try:
        with Stub(ANCHOR_INIT_STUB_PORT, init_log, scenario):
            out["init_exec"] = leg(
                "anchor-init", init_wt, env_extra={"PWD": str(init_wt)},
                extra_args=("-c", "model_providers.stub.base_url="
                            f'"http://127.0.0.1:{ANCHOR_INIT_STUB_PORT}/v1"'),
                log=init_log)
    except SystemExit as exc:   # a stub that never came up is a recorded leg
        out["init_exec"] = {"error": str(exc)}
    after = _tree_hashes(codex_home)
    diff = _tree_diff(before, after)
    out["init_exec_effects"] = {
        "workspace_providers_yaml": (init_wt / ".multi-llm" /
                                     "providers.yaml").is_file(),
        "codex_home_diff": diff,
        "codex_home_byte_identical": not any(diff.values()),
        "plugins_subtree_changed": [p for v in diff.values() for p in v
                                    if p.startswith("plugins/")],
        "multi_llm_under_codex_home": sorted(
            str(p) for p in codex_home.rglob(".multi-llm")),
    }

    stdio_wt, out["stdio_init_worktree_add"] = worktree("anchor-stdio-init")
    if root:
        before = _tree_hashes(codex_home)
        out["init_stdio"] = stdio_session(
            root, [{"mode": "init", "plan": "", "question": "",
                    "extra_flags": []}],
            env={**os.environ, **overlay, "HOME": str(home)},
            roots=[str(stdio_wt)], capabilities={"roots": {"listChanged": False}})
        diff = _tree_diff(before, _tree_hashes(codex_home))
        out["init_stdio_effects"] = {
            "workspace_providers_yaml": (stdio_wt / ".multi-llm" /
                                         "providers.yaml").is_file(),
            "codex_home_diff": diff,
            "codex_home_byte_identical": not any(diff.values()),
        }
    def reading(key: str, expected: Path) -> dict:
        leg_out = out.get(key) or {}
        envelope = leg_out.get("envelope") or {}
        tap = leg_out.get("tap") or {}
        meta = (tap.get("turn_metadata") or [{}])[0] or {}
        return {"anchor": envelope.get("anchor"),
                "anchor_source": envelope.get("anchor_source"),
                "state": envelope.get("state"),
                "expected_anchor": os.path.realpath(str(expected)),
                "anchor_is_expected": bool(envelope.get("anchor")) and (
                    os.path.realpath(str(envelope.get("anchor")))
                    == os.path.realpath(str(expected))),
                "child_cwd": (leg_out.get("children") or {}).get("cwd"),
                "turn_metadata_workspaces": sorted(
                    (meta.get("workspaces") or {}).keys()),
                "pwd_at_launch": [launch.get("PWD")
                                  for launch in tap.get("launches") or []]}

    # The 2026-09-24 fix (D3f rung 2): `-C` with PWD elsewhere anchors on the
    # `-C` workspace via turn metadata, and a PWD-less launch anchors via the
    # metadata instead of refusing.
    out["reverified"] = {"cd_flag": reading("exec_cd_flag", wt),
                         "without_pwd": reading("exec_without_pwd", WORKSPACE),
                         "with_pwd": reading("exec", WORKSPACE),
                         "worktree": reading("exec_worktree", wt)}
    out["legs_not_driven"] = {}
    return out


def _tool_call_item(events_file: Path) -> dict | None:
    """The last `mcp_tool_call` item codex reported, trimmed to its verdict."""
    found = None
    path = OUT / events_file
    for line in (path.read_text(encoding="utf-8").splitlines()
                 if path.is_file() else []):
        event = _json_or_none(line)
        item = (event or {}).get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "mcp_tool_call":
            found = {"status": item.get("status"), "error": item.get("error"),
                     "has_result": item.get("result") is not None}
    return found


# -- the 2026-09-24 re-verification and the legs not driven before ---------- #
#
# Every leg below is zero quota: the host's model is a loopback stub serving a
# scripted scenario, and the one provider any leg dispatches (a nested codex,
# rows 10/12(d)/14) is the wrapper's redirected codex, answered by the row's own
# stub on its text path. Harness instrumentation that reaches a login-shell
# child (a PATH shim, the wrapper's call log) is written into `.profile` and
# removed again, because codex's `env_vars` allowlist forwards neither.

#: The secondary stub each custom scripted leg starts on its own port (the
#: row's own stub keeps serving the nested providers on `MLLM_STUB_PORT`).
SCRIPTED_LEG_STUB_PORT = 8097
#: The apply-mode call rows 12(b)/(c) and the workspace-write leg send. Zero
#: quota by construction: the seeded plan has no review, so a dispatch that is
#: let through stops at the prerequisite gate before any provider runs.
APPLY_MODE_CALL = {"mode": "apply-suggestions", "plan": "plans/auth-feature.md",
                   "question": "", "extra_flags": []}
#: The nested-codex model spec the ask legs fan out to. The wrapper redirects it
#: to the row's stub, whose text path answers it.
NESTED_CODEX_SPEC = "codex:gpt-5.5"
#: What the 12(c) session runs through its OWN shell tool, beside the tool call.
LIFT_SHELL_PROBE = ".mllm-12c-session-shell-probe"


def _call_scenario(path: Path, name: str, arguments: dict,
                   then_shell: str | None = None) -> Path:
    """A one-call scripted scenario (JSON is YAML), like `_init_scenario`."""
    block = {"surfaces": ["responses"], "namespace": "mcp__multi_llm",
             "name": CODEX_MCP_SERVER, "call_id": f"call_mllm_{name}",
             "arguments": arguments, "terminal": f"{name} returned."}
    if then_shell:
        block["then_shell"] = then_shell
    path.write_text(json.dumps({
        "name": name, "description": f"2026-09-24 re-verification leg {name}",
        "scripted_tool_call": block,
        "prompts": [{"pattern": "call.*multi_llm tool", "content": "OK",
                     "description": "turn without the tool"}],
        "fallback": {"content": "OK"}}, indent=2), encoding="utf-8")
    return path


class _ProcSampler:
    """Samples /proc while a leg runs, keeping every dispatcher-shaped process.

    What "no child on the process table" is asserted on: a refusal that
    spawned nothing leaves this empty, a dispatch leaves its `uv run …
    multi_llm.py` (and the login shell before its `exec`) here.
    """

    PATTERN = re.compile(r"multi_llm\.py|cd -- \"\$1\"|--internal-worker")

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.seen: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            for row in _process_table():
                if self.PATTERN.search(row):
                    self.seen.setdefault(row.split(" ", 1)[0], row[:600])
            self._stop.wait(self.interval)

    def __enter__(self) -> "_ProcSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(5)

    def rows(self) -> list[str]:
        return [self.seen[k] for k in sorted(self.seen, key=int)]


def _porcelain(path: Path = WORKSPACE) -> str:
    done = subprocess.run(["git", "status", "--porcelain"], cwd=str(path),
                          capture_output=True, text=True)
    return done.stdout


class _ProfileExtra:
    """Append to `$HOME/.profile` for one leg and restore it byte for byte."""

    def __init__(self, home: Path, text: str):
        self.profile = home / ".profile"
        self.text = text
        self.original: str | None = None

    def __enter__(self) -> "_ProfileExtra":
        self.original = (self.profile.read_text(encoding="utf-8")
                         if self.profile.is_file() else None)
        base = self.original or ""
        if base and not base.endswith("\n"):
            base += "\n"
        self.profile.write_text(base + self.text, encoding="utf-8")
        assert_profile_is_read(self.profile.parent)
        return self

    def __exit__(self, *_exc) -> None:
        if self.original is None:
            self.profile.unlink(missing_ok=True)
        else:
            self.profile.write_text(self.original, encoding="utf-8")


class _FileEdit:
    """Replace one file's text for a leg and restore it byte for byte."""

    def __init__(self, path: Path, text: str):
        self.path, self.text, self.original = path, text, None

    def __enter__(self) -> "_FileEdit":
        self.original = (self.path.read_text(encoding="utf-8")
                         if self.path.is_file() else None)
        self.path.write_text(self.text, encoding="utf-8")
        return self

    def __exit__(self, *_exc) -> None:
        if self.original is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.write_text(self.original, encoding="utf-8")


def _slow_codex_profile(label: str, sleep_s: int, call_log: Path | None) -> str:
    """`.profile` text putting a `codex` shim first on the login shell's PATH.

    The shim sleeps before every nested `codex exec` (never before `codex
    sandbox`, the D6 probe), so a detached ask run lives long enough for relay
    points (row 10) or for a cap to fire (row 14). Under the bundle, not the
    home: the tmpfs mounts are `noexec`. The wrapper's call log is exported
    beside it when asked, since `env_vars` forwards no `MLLM_*` name.
    """
    shim_dir = OUT / "slow-bin" / label
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "codex"
    shim.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "exec" ]; then sleep {int(sleep_s)}; fi\n'
        'exec /opt/mllm/bin/codex "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    text = f"export PATH={shlex.quote(str(shim_dir))}:$PATH\n" if sleep_s else ""
    if call_log is not None:
        text += f"export MLLM_CALL_LOG={shlex.quote(str(call_log))}\n"
    return text


def _workspace_config_with(extra: str) -> _FileEdit:
    """The seeded `.multi-llm/providers.yaml` plus a top-level block."""
    path = WORKSPACE / ".multi-llm" / "providers.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    return _FileEdit(path, text + extra)


def scripted_leg(home: Path, overlay: dict, label: str, arguments: dict, *,
                 extra_args: tuple[str, ...] = (), env_extra: dict | None = None,
                 env_drop: tuple[str, ...] = (), then_shell: str | None = None,
                 cwd: Path | None = None, timeout: float = 300.0,
                 port: int = SCRIPTED_LEG_STUB_PORT) -> dict:
    """One tapped `codex exec` turn whose scripted call carries ``arguments``.

    A secondary stub on its own port serves the call (`-c
    model_providers.stub.base_url=…` points this turn at it); the process table
    is sampled for the whole turn; the porcelain of the workspace is taken on
    both sides; and the call's verdict is read three ways — codex's own
    `mcp_tool_call` item, whether the server received `tools/call` at all, and
    the envelope the model was handed.
    """
    scenario = _call_scenario(OUT / f"scenario-{label}.json",
                              label.replace("-", "_"), arguments, then_shell)
    log_path = OUT / f"stub-{label}.jsonl"
    porcelain_before = _porcelain()
    try:
        with Stub(port, log_path, scenario), _ProcSampler() as sampler:
            result = codex_exec_leg(
                home, overlay, log_path, label,
                extra_args=("-c", "model_providers.stub.base_url="
                            f'"http://127.0.0.1:{port}/v1"', *extra_args),
                env_extra=env_extra, env_drop=env_drop, cwd=cwd, tap=True,
                timeout=timeout)
    except SystemExit as exc:   # a stub that never came up is a recorded leg
        return {"label": label, "error": str(exc)}
    porcelain_after = _porcelain()
    tap = result.get("tap") or {}
    entries = _stub_entries(log_path)
    terminal = next((e.get("scripted") for e in entries
                     if (e.get("scripted") or {}).get("phase") == "terminal"), {})
    result.update({
        "arguments": arguments,
        "tool_call_item": _tool_call_item(Path(result["events_file"])),
        "server_got_tools_call": "tools/call" in (tap.get("client_methods") or []),
        "requires_approval_refusal": "requires approval" in (
            (OUT / result["events_file"]).read_text(encoding="utf-8")
            + result.get("stderr", "")),
        "dispatcher_processes": sampler.rows(),
        "spawned_dispatcher": any("multi_llm.py" in row for row in sampler.rows()),
        "porcelain_byte_identical": porcelain_before == porcelain_after,
        "porcelain_before": porcelain_before[-4000:],
        "porcelain_after": porcelain_after[-4000:],
        "advertised_tool_names": sorted({n for e in entries
                                         for n in (e.get("tool_names") or [])}),
    })
    if then_shell:
        result["session_shell"] = {
            "command": then_shell,
            "tool": terminal.get("shell_tool"),
            "unavailable": terminal.get("shell_unavailable"),
            "output": terminal.get("shell_output"),
            "bwrap_in_output": "bwrap" in str(terminal.get("shell_output") or ""),
        }
    return result


def _ask_runs() -> list[dict]:
    """Every ask run dir under the workspace: its run record and log evidence."""
    runs = []
    for status in sorted(WORKSPACE.rglob("ask/*/.status.json")):
        doc = _json_or_none(status.read_text(encoding="utf-8")) or {}
        log_path = status.parent / "orchestrator-run.log"
        lines = (log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if log_path.is_file() else [])
        runs.append({
            "dir": str(status.parent),
            "run": {k: (doc.get("run") or {}).get(k)
                    for k in ("run_id", "state", "exit_code", "pid", "error",
                              "started_at", "ended_at")},
            "state": doc.get("state"),
            "models_served": doc.get("models_served"),
            "sandbox_lines": [line[:400] for line in lines
                              if re.search(r"SANDBOX|sandbox_fallback|landlock"
                                           r"|bypass", line)][:40],
            "log_tail": lines[-15:],
        })
    return runs


def _nested_codex_calls(call_log: Path) -> list[dict]:
    """The wrapper's call records for nested codex, argv and sandbox tokens."""
    rows = []
    for row in _stub_entries(call_log):
        if row.get("cli") != "codex":
            continue
        argv = [str(a) for a in (row.get("argv") or [])]
        joined = " ".join(argv)
        rows.append({"argv": argv[:40], "subcommand": argv[1] if len(argv) > 1 else None,
                     "danger_full_access": "danger-full-access" in joined,
                     "legacy_landlock": "use_legacy_landlock" in joined,
                     "read_only": "read-only" in joined})
    return rows


def _worker_processes() -> list[str]:
    return [row[:400] for row in _process_table() if "--internal-worker" in row]


def _ask_call(question: str) -> dict:
    return {"mode": "ask", "plan": "plans/auth-feature.md", "question": question,
            "extra_flags": ["--models", NESTED_CODEX_SPEC]}


def _nested_ask_leg(home: Path, overlay: dict, label: str, question: str, *,
                    sleep_s: int = 0, poll_s: int | None = None,
                    extra_args: tuple[str, ...] = (), timeout: float = 600.0,
                    wait_worker_s: float = 0.0) -> dict:
    """An `ask` call that fans out to ONE nested codex provider (zero quota).

    Rows 10 (relay points: a slow nested exec plus a 5 s `poll_s`), 12(d)
    (which sandbox rung the grandchild ran under, off the wrapper's call log
    and the run log) and 14 (a cap firing on a detached run).
    """
    # The nested codex's provider identity: the wrapper copies
    # `/work/homes/codex` into each invocation root, and a probe row names no
    # `--cli`/`--helpers`, so nothing seeded it (codex then exits 1 on a
    # CODEX_HOME that does not exist).
    seed_homes(["codex"])
    call_log = OUT / f"calls-{label}.jsonl"
    profile = _slow_codex_profile(label, sleep_s, call_log)
    config = (_workspace_config_with(f"progress:\n  poll_s: {int(poll_s)}\n")
              if poll_s else None)
    before_runs = {r["dir"] for r in _ask_runs()}
    with _ProfileExtra(home, profile):
        if config:
            config.__enter__()
        try:
            leg = scripted_leg(home, overlay, label, _ask_call(question),
                               extra_args=extra_args, timeout=timeout)
            leg["workers_right_after"] = _worker_processes()
            runs_after = [r for r in _ask_runs() if r["dir"] not in before_runs]
            leg["ask_runs_right_after"] = runs_after
            if wait_worker_s:
                deadline = time.time() + wait_worker_s
                while time.time() < deadline and _worker_processes():
                    time.sleep(2)
                leg["workers_after_wait"] = _worker_processes()
                leg["ask_runs_after_wait"] = [r for r in _ask_runs()
                                              if r["dir"] not in before_runs]
        finally:
            if config:
                config.__exit__(None, None, None)
    leg["nested_codex_calls"] = _nested_codex_calls(call_log)
    leg["slow_codex_exec_s"] = sleep_s
    leg["poll_s"] = poll_s
    return leg


def _nested_rung(leg: dict) -> dict:
    """Which posture the nested codex exec ran under (row 12(d))."""
    execs = [c for c in leg.get("nested_codex_calls") or []
             if c.get("subcommand") == "exec"]
    lines = [line for run in leg.get("ask_runs_after_wait")
             or leg.get("ask_runs_right_after") or []
             for line in run.get("sandbox_lines") or []]
    if any(c["danger_full_access"] for c in execs):
        rung = "bypassed"
    elif any(c["legacy_landlock"] for c in execs):
        rung = "fallback-landlock"
    elif any(c["read_only"] for c in execs):
        rung = "sandboxed (read-only, no fallback)"
    else:
        rung = None
    return {"rung": rung, "nested_exec_argvs": [c["argv"] for c in execs],
            "sandbox_lines": lines,
            "session_posture": [(m or {}).get("sandbox_mode") for m in
                                ((leg.get("tap") or {}).get("turn_metadata") or [])]}


def _no_python_path_dir() -> Path:
    """A PATH directory with every command of this PATH except `python*`.

    Wrapped CLIs (`/opt/mllm/bin/*`, python scripts themselves) point at their
    real binaries instead, so nothing on it needs an interpreter to start.
    """
    target = OUT / "nopy-bin"
    target.mkdir(parents=True, exist_ok=True)
    for entry in (os.environ.get("PATH") or "").split(":"):
        base = Path(entry)
        if not base.is_dir():
            continue
        for item in sorted(base.iterdir()):
            name = item.name
            if name.startswith("python") or (target / name).exists() \
                    or (target / name).is_symlink():
                continue
            source = item
            if str(base) == "/opt/mllm/bin":
                real = REAL_DIR / name
                if not real.exists():
                    continue
                source = real
            try:
                (target / name).symlink_to(source)
            except OSError:
                continue
    return target


def _codex_log_lines(codex_home: Path, pattern: str, limit: int = 30) -> list[str]:
    """Rows of codex's own log stores (`*.sqlite`, `log/*.log`) matching
    ``pattern``, newest last, each capped. Read-only; never a value we planted."""
    import sqlite3
    regex = re.compile(pattern, re.I)
    found: list[str] = []
    # The LOG stores only: the thread/state stores carry the probe's own
    # prompt ("Call the multi_llm tool.") and would match every pattern.
    for db in sorted(codex_home.glob("logs_*.sqlite")):
        try:
            conn = sqlite3.connect(str(db))
            tables = [r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'")]
            for table in tables:
                for row in conn.execute(f'select * from "{table}"'):
                    text = " | ".join(str(c) for c in row)
                    if regex.search(text):
                        found.append(f"{db.name}:{table}: {text[:700]}")
            conn.close()
        except sqlite3.Error as exc:
            found.append(f"{db.name}: unreadable ({exc})")
    for path in sorted((codex_home / "log").glob("*.log")) \
            if (codex_home / "log").is_dir() else []:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if regex.search(line):
                found.append(f"{path.name}: {line[:700]}")
    return found[-limit:]


def _interpreter_absent_legs(home: Path, root: Path | None, overlay: dict,
                             stub_log: Path) -> dict:
    """Row 9(c): a home whose PATH carries no `python3`, recorded whole."""
    nopy = _no_python_path_dir()
    env = _codex_env(home, overlay)
    env["PATH"] = str(nopy)
    out: dict = {"path": str(nopy),
                 "python3_resolves": shutil.which("python3", path=str(nopy))}
    listing = _run_codex(["plugin", "list", "--json"], env)
    doc = _json_or_none(listing["_stdout_full"])
    entry = next((p for p in ((doc or {}).get("installed") or [])
                  if CODEX_PLUGIN_ID in json.dumps(p)), None)
    out["plugin_list"] = {"command": _public(listing), "entry": entry}
    prompt = _run_codex(["debug", "prompt-input"], env)
    out["prompt_input"] = {"exit": prompt["exit"],
                           "skill": _parse_prompt_input(prompt["_stdout_full"]),
                           "stderr": prompt["stderr"][-2000:]}
    mcp_get = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
    out["mcp_get"] = {"exit": mcp_get["exit"],
                      "document": _json_or_none(mcp_get["_stdout_full"]),
                      "stderr": mcp_get["stderr"][-2000:]}
    mark = len(_stub_entries(stub_log))
    exec_leg = codex_exec_leg(home, overlay, stub_log, "interpreter-absent",
                              env_extra={"PATH": str(nopy)})
    events = (OUT / exec_leg["events_file"]).read_text(encoding="utf-8")
    entries = _stub_entries(stub_log)[mark:]
    out["exec"] = {
        "exit": exec_leg["exit"],
        "declarations": exec_leg["declarations"],
        "scripted_phases": exec_leg["scripted_phases"],
        "tool_names_advertised": sorted({n for e in entries
                                         for n in (e.get("tool_names") or [])}),
        "multi_llm_in_tools": any("multi_llm" in n for e in entries
                                  for n in (e.get("tool_names") or [])),
        # Codex's own words about the failed spawn, VERBATIM: that text is all
        # a user gets.
        "stderr": exec_leg["stderr"],
        "error_events": [line for line in events.splitlines()
                         if re.search(r"error|mcp|multi_llm|spawn|python",
                                      line, re.I)][:20],
        # `exec --json` prints nothing about the failed spawn on stdout or
        # stderr, so codex's own log store is read for what it DID write.
        "codex_log_lines": _codex_log_lines(home / ".codex",
                                            r"mcp|python3|multi_llm"),
    }
    # The same turn once more with codex's own tracing on stderr
    # (`RUST_LOG=info`), keeping only the lines about MCP servers.
    verbose = codex_exec_leg(home, overlay, stub_log, "interpreter-absent-log",
                             env_extra={"PATH": str(nopy), "RUST_LOG": "info"})
    out["exec"]["rust_log_mcp_lines"] = [
        (line[line.index("codex_mcp"):] if "codex_mcp" in line else line)[:600]
        for line in verbose["stderr"].splitlines()
        if re.search(r"codex_mcp|multi_llm|python3", line)][:30]
    # The blocking half: `install-host codex --check` from this same PATH. Run
    # twice — through `uv run` (which prepends the venv's bin, where a
    # `python3` lives) and with the venv interpreter directly — so the answer
    # does not depend on how the check was launched without saying so.
    skill = (root / "skills" / "multi-llm") if root else None
    uv = str(REAL_DIR / "uv") if (REAL_DIR / "uv").exists() else shutil.which("uv")
    venv_python = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", "")) / "bin" / "python"
    checks = {}
    if skill and skill.is_dir():
        argv_tail = [str(skill / "multi_llm.py"), "install-host", "codex", "--check"]
        variants = {"via_uv_run": [uv, "run", "--frozen", "--no-dev", "--project",
                                   str(skill), "--", "python", *argv_tail]}
        if venv_python.exists():
            variants["venv_interpreter"] = [str(venv_python), *argv_tail]
        for name, argv in variants.items():
            done = subprocess.run(argv, env=env, cwd=str(WORKSPACE),
                                  capture_output=True, text=True, timeout=300,
                                  stdin=subprocess.DEVNULL)
            text = done.stdout + done.stderr
            checks[name] = {"argv": argv, "exit": done.returncode,
                            "output": text[-6000:],
                            "names_missing_interpreter":
                                "`python3` is not on this PATH" in text}
    out["install_host_check"] = checks
    return out


def _shell_ladder_refusal(home: Path, root: Path | None) -> dict:
    """Row 9(b): the child-spawn lookup on a PATH where no `bash` resolves.

    Driven through the stdio client, not codex: the server's fallback
    (`/bin/bash`) exists in this image, and codex forwards no `MLLM_TEST_*`
    name, so only the stdio launch can point both the PATH and the fallback
    away. The server's interpreter is named absolutely for the same reason.
    """
    if not root:
        return {"error": "no installed root"}
    empty = OUT / "no-bash-bin"
    empty.mkdir(parents=True, exist_ok=True)
    python = shutil.which("python3") or sys.executable
    env = {**os.environ, "HOME": str(home), "PWD": str(WORKSPACE),
           "PATH": str(empty), "MLLM_TEST_FALLBACK_SHELL": "/nonexistent/bash"}
    status_call = {"mode": "status", "plan": "plans/auth-feature.md",
                   "question": "", "extra_flags": []}
    with _ProcSampler() as sampler:
        session = stdio_session(root, [status_call], env=env, command=python)
    envelope = _envelope_of((session["calls"] or [{}])[0].get("response"))
    environment = (envelope or {}).get("environment") or {}
    return {
        "path_searched": str(empty),
        "fallback_pointed_at": "/nonexistent/bash",
        "session": session,
        "envelope": {k: (envelope or {}).get(k)
                     for k in ("ok", "state", "exit", "message", "anchor",
                               "anchor_source")},
        "is_error": ((session["calls"] or [{}])[0].get("response") or {})
                    .get("result", {}).get("isError"),
        "environment": environment,
        "environment_keys": sorted(environment),
        "login_shell": environment.get("login_shell"),
        "shell": environment.get("shell"),
        "message_names_bash_and_path": bool(envelope) and "bash" in str(
            envelope.get("message")) and str(empty) in str(envelope.get("message")),
        "dispatcher_processes": sampler.rows(),
    }


def _plugin_data_delivery(home: Path, root: Path | None, overlay: dict,
                          stub_log: Path) -> dict:
    """Row 11(b): can a running server name `${PLUGIN_DATA}` at all?

    Two edits of the INSTALLED `.mcp.json` (row 13 found no overlay, so the
    installed copy is the only place to spell it; each is restored after its
    leg and recorded as a divergence): (1) an `env` value and an `env_vars`
    entry spelled `${PLUGIN_DATA}`, plus `${PLUGIN_ROOT}` beside it as a
    control; (2) `cwd: "${PLUGIN_DATA}"`. The tap records the server's launch
    cwd and the planted names' values; whether `./mcp/server.py` resolved is
    read off the server's exit and whether `tools[]` carried the tool.
    """
    mcp = (root / ".mcp.json") if root else None
    if not mcp or not mcp.is_file():
        return {"error": "no installed .mcp.json"}
    original = json.loads(mcp.read_text(encoding="utf-8"))
    env = _codex_env(home, overlay)
    out: dict = {"divergence": f"{mcp}: edited per leg, restored after each"}

    def leg(label: str, mutate) -> dict:
        doc = json.loads(json.dumps(original))
        mutate(doc["mcpServers"][CODEX_MCP_SERVER])
        with _FileEdit(mcp, json.dumps(doc, indent=2)):
            mcp_get = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
            result = codex_exec_leg(home, overlay, stub_log, label, tap=True,
                                    env_extra={"MLLM_PD_FORWARDED": "${PLUGIN_DATA}"})
        tap = result.get("tap") or {}
        launches = tap.get("launches") or []
        return {
            "edit": doc["mcpServers"][CODEX_MCP_SERVER],
            "mcp_get": _json_or_none(mcp_get["_stdout_full"]),
            "mcp_get_stderr": mcp_get["stderr"][-1500:],
            "server_launched": bool(launches),
            "server_cwd": [launch.get("cwd") for launch in launches],
            "server_env_values": [launch.get("env_values") for launch in launches],
            "server_exit": tap.get("exit"),
            "tool_advertised": bool(result.get("declarations")),
            "scripted_phases": result.get("scripted_phases"),
            "envelope_state": (result.get("envelope") or {}).get("state"),
            "stderr_tail": (result.get("stderr") or "")[-3000:],
            "plugin_data_dirs_after": sorted(
                str(p) for p in (home / ".codex").rglob("*plugin*data*")),
            "codex_log_lines": _codex_log_lines(
                home / ".codex", r"multi_llm.*(fail|error|spawn|cwd|No such)"
                                 r"|PLUGIN_DATA", limit=10),
        }

    def env_edit(server: dict) -> None:
        server["env"] = {"MLLM_PD_ENV": "${PLUGIN_DATA}",
                         "MLLM_PD_ROOT_CONTROL": "${PLUGIN_ROOT}"}
        server["env_vars"] = list(server.get("env_vars") or []) + [
            "${PLUGIN_DATA}", "MLLM_PD_FORWARDED"]

    def cwd_edit(server: dict) -> None:
        server["cwd"] = "${PLUGIN_DATA}"

    out["env_and_env_vars"] = leg("plugin-data-env", env_edit)
    out["cwd"] = leg("plugin-data-cwd", cwd_edit)
    out["restored"] = json.loads(mcp.read_text(encoding="utf-8")) == original
    return out


#: The few-second cap row 14 fires, and how long the nested exec it cuts
#: sleeps (long enough that the detached run is still alive when codex's turn
#: and the session both end).
CAP_FIRES_S = 8
CAP_FIRES_NESTED_SLEEP_S = 45


def _cap_leg_reading(leg: dict) -> dict:
    envelope = leg.get("envelope") or {}
    runs = leg.get("ask_runs_right_after") or []
    after = leg.get("ask_runs_after_wait") or []
    return {
        "wall_clock_s": leg.get("wall_clock_s"),
        "exit": leg.get("exit"),
        "tool_call_item": leg.get("tool_call_item"),
        # What the model was handed, verbatim (capped by the stub).
        "model_received": leg.get("returned_output"),
        "envelope_state": envelope.get("state"),
        "envelope_message": envelope.get("message"),
        "server_handoff": envelope.get("state") == "running"
        and "time budget ran out" in str(envelope.get("message")),
        "server_exit": (leg.get("tap") or {}).get("exit"),
        "server_signals": (leg.get("tap") or {}).get("signals"),
        "workers_right_after": leg.get("workers_right_after"),
        "run_right_after": [r.get("run") for r in runs],
        "workers_after_wait": leg.get("workers_after_wait"),
        "run_after_wait": [r.get("run") for r in after],
        "run_log_tail_after_wait": [r.get("log_tail") for r in after],
    }


def _cap_fires_legs(home: Path, root: Path | None, overlay: dict) -> dict:
    """Row 14's cap-fires half.

    The server sizes its own budget from the same installed `.mcp.json` codex
    reads its cap from, so the host's cap can fire FIRST only if codex takes a
    smaller cap from somewhere else. Two such routes are tried (each only
    re-read through `codex mcp get` first, and driven only when codex reports
    the smaller cap): (1) `tool_timeout_sec` in config.toml's plugin server
    table; (2) a top-level `mcp_servers.multi_llm.tool_timeout_sec` passed as
    `-c`. Then (3) the installed `.mcp.json` itself set to a few seconds, which
    reads whether the server's own handoff beats codex's cap. Every driven leg
    fires on a detached `ask` whose nested codex exec sleeps well past the cap;
    the worker's liveness and its run record are read at once and again once
    it has had time to finish.
    """
    out: dict = {"cap_s": CAP_FIRES_S, "nested_sleep_s": CAP_FIRES_NESTED_SLEEP_S}
    config = home / ".codex" / "config.toml"
    env = _codex_env(home, overlay)

    def mcp_get(extra: tuple[str, ...] = ()) -> dict | None:
        done = _run_codex([*extra, "mcp", "get", CODEX_MCP_SERVER, "--json"], env)
        return _json_or_none(done["_stdout_full"]) or {
            "exit": done["exit"], "stderr": done["stderr"][-1500:]}

    def drive(label: str, question: str, extra: tuple[str, ...] = ()) -> dict:
        leg = _nested_ask_leg(home, overlay, label, question,
                              sleep_s=CAP_FIRES_NESTED_SLEEP_S, timeout=300,
                              wait_worker_s=150, extra_args=extra)
        return {"reading": _cap_leg_reading(leg), "leg": leg}

    table = (f'\n[plugins."{CODEX_PLUGIN_ID}".mcp_servers.{CODEX_MCP_SERVER}]\n'
             f"tool_timeout_sec = {CAP_FIRES_S}\n")
    with _FileEdit(config, config.read_text(encoding="utf-8") + table):
        seen = mcp_get()
        out["config_plugin_table"] = {"edit": table.strip(), "mcp_get": seen,
                                      "honored": (seen or {}).get(
                                          "tool_timeout_sec") == CAP_FIRES_S}
        if out["config_plugin_table"]["honored"]:
            out["config_plugin_table"].update(
                drive("cap-config", "How is auth tested?"))
    flag = ("-c", f"mcp_servers.{CODEX_MCP_SERVER}.tool_timeout_sec={CAP_FIRES_S}")
    seen = mcp_get(flag)
    out["config_top_level_flag"] = {"flag": list(flag), "mcp_get": seen,
                                    "honored": (seen or {}).get(
                                        "tool_timeout_sec") == CAP_FIRES_S}
    if out["config_top_level_flag"]["honored"]:
        out["config_top_level_flag"].update(
            drive("cap-flag", "Which tokens does auth issue?", flag))
    mcp = (root / ".mcp.json") if root else None
    if mcp and mcp.is_file():
        doc = json.loads(mcp.read_text(encoding="utf-8"))
        doc["mcpServers"][CODEX_MCP_SERVER]["tool_timeout_sec"] = CAP_FIRES_S
        with _FileEdit(mcp, json.dumps(doc, indent=2)):
            out["mcp_json"] = {"edit": f"{mcp}: tool_timeout_sec = {CAP_FIRES_S}",
                               "mcp_get": mcp_get()}
            out["mcp_json"].update(drive("cap-mcp-json",
                                         "What does the login flow return?"))
    return out


def upstream_validator_legs(home: Path, root: Path | None, env: dict) -> dict:
    """A4 row 1: look for upstream's own plugin validator, and run it if found.

    ``codex debug prompt-input`` unpacks codex's bundled system skills into
    ``$CODEX_HOME/skills/.system/`` (measured 0.164.0); ``validate_plugin.py``
    lived in the ``plugin-creator`` skill there up to 0.156.1 and has shipped
    nowhere since. So the row searches the whole home for the file rather than
    one path, runs every copy it finds against the installed tree, and hashes
    each. ``status: absent`` is a RECORDED absence: the gate reports it as
    such, never as a validator pass (packaging/codex/ingestion_rules.py).
    """
    prompt = _run_codex(["debug", "prompt-input"], env)
    codex_home = home / ".codex"
    system = codex_home / "skills" / ".system"
    skills = (sorted(p.name for p in system.iterdir() if p.is_dir())
              if system.is_dir() else [])
    cache = codex_home / "plugins" / "cache"
    found = sorted(path for path in codex_home.rglob("validate_plugin.py")
                   if cache not in path.parents)
    python = shutil.which("python3") or "/usr/bin/python3"
    runs = []
    for path in found:
        data = path.read_bytes()
        argv = [python, str(path), str(root)] if root else None
        done = (subprocess.run(argv, env=env, capture_output=True, text=True,
                               timeout=120, stdin=subprocess.DEVNULL)
                if argv else None)
        runs.append({
            "path": str(path.relative_to(codex_home)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "argv": ["python3", str(path.relative_to(codex_home)), "<installed root>"],
            "exit": done.returncode if done else None,
            "stdout": (done.stdout if done else "")[:CODEX_OUTPUT_CAP],
            "stderr": (done.stderr if done else "")[:CODEX_OUTPUT_CAP],
        })
    return {
        "codex_version": _codex_version(home),
        "prompt_input": _public(prompt),
        "system_skills": skills,
        "validators": runs,
        "status": "present" if runs else "absent",
    }


def run_codex_package_probe(probe: str, home: Path, overlay: dict,
                            stub_log: Path, install: dict | None) -> dict:
    """Run one package-route probe row and write `probe-<name>.json`.

    Every leg is RECORDED, not judged: the entrypoint exits 0 on a probe whose
    answer is merely recorded, and the grader in test_container_host_lane.py
    decides which readings are blocking (the plan's item 15). Legs this driver
    does not drive yet are named in `legs_not_driven` with the reason, so an
    absent reading is never mistaken for a green one.
    """
    if probe not in CODEX_PACKAGE_PROBES:
        raise SystemExit(f"entrypoint: unknown codex package probe {probe!r}; "
                         f"known: {', '.join(CODEX_PACKAGE_PROBES)}")
    root = codex_installed_root(home, install)
    env = _codex_env(home, overlay)
    record: dict = {"probe": probe, "installed_root": str(root) if root else None,
                    "legs_not_driven": {}}
    # All four properties: the tool schema requires every one, and a call that
    # omits `question`/`extra_flags` is refused at argument validation before
    # any dispatcher starts, so the leg would never reach `status`.
    status_call = {"mode": "status", "plan": "plans/auth-feature.md",
                   "question": "", "extra_flags": []}

    if probe == "seed":
        record["plugin_install"] = "plugin-install.json"
    elif probe == "floor":
        record["codex_version"] = _codex_version(home)
    elif probe == "features":
        result = _run_codex(["features", "list"], env, timeout=60)
        (OUT / "codex-features.txt").write_text(result["_stdout_full"],
                                                encoding="utf-8")
        record["codex_version"] = _codex_version(home)
        record["command"] = _public(result)
        record["features"] = _parse_features(result["_stdout_full"])
    elif probe == "loader-mcp-get":
        result = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
        (OUT / "codex-mcp-get.json").write_text(result["_stdout_full"],
                                                encoding="utf-8")
        record["command"] = _public(result)
        record["document"] = _json_or_none(result["_stdout_full"])
    elif probe == "loader-prompt-input":
        result = _run_codex(["debug", "prompt-input"], env)
        (OUT / "codex-prompt-input.txt").write_text(result["_stdout_full"],
                                                    encoding="utf-8")
        record["command"] = _public(result)
        record["skill"] = _parse_prompt_input(result["_stdout_full"])
    elif probe in ("scripted-call", "loader-tools"):
        record["exec"] = codex_exec_leg(home, overlay, stub_log, probe)
    elif probe == GIT_ROUTE_PROBE:
        record.update(git_route_legs(home, root, overlay, stub_log, install,
                                     status_call))
    elif probe == "upstream-validator":
        record.update(upstream_validator_legs(home, root, env))
    elif probe == "runnability":
        # The anchor row 7 settled is `env:PWD` (codex advertises no `roots`),
        # so the stdio driver hands the server the workspace as `PWD`, exactly
        # as codex forwards it from the launching shell.
        stdio_env = {**os.environ, "PWD": str(WORKSPACE)}
        record["stdio_env_pwd"] = str(WORKSPACE)
        record["stdio"] = (stdio_session(root, [status_call], env=stdio_env)
                           if root else None)
        lock = (root / "skills" / "multi-llm" / "uv.lock") if root else None
        if lock and lock.is_file():
            lock.unlink()
            record["partial_payload"] = {
                "removed": str(lock),
                "stdio": stdio_session(root, [status_call], env=stdio_env)}
        else:
            record["legs_not_driven"]["partial_payload"] = (
                "no installed uv.lock to remove")
    elif probe == "anchor":
        pycache_start = _pycache_state(root)
        record["stdio_with_roots"] = (stdio_session(
            root, [status_call], roots=[str(WORKSPACE)],
            capabilities={"roots": {"listChanged": False}}) if root else None)
        record["stdio_without_roots"] = (stdio_session(root, [status_call])
                                         if root else None)
        record.update(_anchor_legs(home, root, overlay, stub_log))
        # Fix 4 across the whole row (seven codex turns, two stdio sessions):
        # nothing under a `__pycache__` of the installed plugin was written.
        record["plugin_pycache_writes"] = _pycache_writes(
            pycache_start, _pycache_state(root))
    elif probe == "environment":
        launch = plant_environment_markers(home)
        record["planted"] = {"launch_only": sorted(launch),
                             "profile": [PROBE_PROFILE_CREDENTIAL, "PATH"],
                             "bashrc_only": [PROBE_BASHRC_MARKER]}
        record["children_file"] = PROBE_CHILDREN_FILE
        mark = len(_child_observations())
        record["control_stdio"] = (stdio_session(
            root, [status_call], env={**os.environ, **launch,
                                      "HOME": str(home),
                                      "PWD": str(WORKSPACE)}) if root else None)
        control_rows = _child_observations()[mark:]
        control_env = _envelope_of(
            ((record["control_stdio"] or {}).get("calls") or [{}])[0]
            .get("response"))
        record["control_children"] = grade_child_observations(
            control_rows, (control_env or {}).get("anchor"))
        mark = len(_child_observations())
        record["graded_exec"] = codex_exec_leg(
            home, overlay, stub_log, "environment", env_extra=launch, tap=True)
        graded_env = _envelope_of(record["graded_exec"].get("returned_output"))
        record["graded_children"] = grade_child_observations(
            _child_observations()[mark:], (graded_env or {}).get("anchor"))
        record["shell_ladder_refusal"] = _shell_ladder_refusal(home, root)
        record["interpreter_absent"] = _interpreter_absent_legs(
            home, root, overlay, stub_log)
    elif probe == "progress-cancellation":
        record["progress_exec"] = codex_exec_leg(home, overlay, stub_log,
                                                 "progress", tap=True)
        # Row 10(a)'s three-relay-point leg: a detached ask whose nested codex
        # exec sleeps 20 s, polled every 5 s, so the launch and each exit-2
        # poll are relay points. Counted on the wire and in what came back.
        relay = _nested_ask_leg(home, overlay, "relay-points",
                                "Which components does the plan add?",
                                sleep_s=20, poll_s=5, timeout=400,
                                wait_worker_s=60)
        tap = relay.get("tap") or {}
        events = [e for e in (_json_or_none(line) for line in (
            (OUT / relay["events_file"]).read_text(encoding="utf-8").splitlines()
            if relay.get("events_file") else [])) if isinstance(e, dict)]
        # Everything codex emitted OTHER than the tool call's own item (whose
        # completed result is the envelope, `progress_lines` included).
        outside = json.dumps([e for e in events if ((e.get("item") or {})
                              .get("type") != "mcp_tool_call")])
        types = [f"{e.get('type')}:{(e.get('item') or {}).get('type')}"
                 for e in events]
        started = next((i for i, t in enumerate(types)
                        if t == "item.started:mcp_tool_call"), None)
        done = next((i for i, t in enumerate(types)
                     if t == "item.completed:mcp_tool_call"), None)
        texts = [t for t in tap.get("log_notification_texts") or []
                 if isinstance(t, str) and t.strip()]
        envelope = relay.get("envelope") or {}
        relay["relay_reading"] = {
            "relays": envelope.get("relays"),
            "progress_token_sent": tap.get("progress_tokens"),
            "server_sent_progress": tap.get("progress_notifications"),
            "server_sent_log": tap.get("log_notifications"),
            "notification_texts": texts,
            # Did any notification surface on its own, outside the result?
            "texts_outside_tool_result": sum(1 for t in texts
                                             if t[:80] in outside),
            "events_while_call_in_flight": (
                types[started + 1:done] if started is not None
                and done is not None else None),
            # What the model was handed at the end: the envelope's own lines.
            "result_progress_lines": envelope.get("progress_lines"),
            "event_types": relay.get("event_types"),
            "item_types": relay.get("item_types"),
        }
        record["relay_exec"] = relay
        # `status` answers in ~0.2 s, so an interrupt would land after the
        # reply. The call is held in flight instead by a flag-gated `sleep` in
        # the login-shell child's `.profile`, and codex's process group is then
        # sent SIGINT, a terminal's Ctrl-C. (Codex's own shell-snapshot
        # `bash -lc` reads `.profile` too and stalls beside it — measured; it
        # does not hold up the tool call.)
        stall = home / ".mllm-probe-stall"
        profile = home / ".profile"
        existing = profile.read_text(encoding="utf-8") if profile.is_file() else ""
        profile.write_text(existing + f"[ -f {shlex.quote(str(stall))} ] && "
                           f"sleep {PROBE_STALL_S}\n", encoding="utf-8")
        assert_profile_is_read(home)
        stall.touch()
        try:
            record["cancel_exec"] = codex_exec_leg(home, overlay, stub_log,
                                                   "cancel",
                                                   interrupt_after_call=True,
                                                   tap=True, timeout=120)
        finally:
            stall.unlink(missing_ok=True)
        time.sleep(2)
        record["cancel_exec"]["process_table_after_2s"] = [
            line for line in _process_table()
            if "sleep" in line or "server.py" in line or "multi_llm" in line
            or "codex" in line]
        record["cancel_exec"]["stall_s"] = PROBE_STALL_S
        # The server-side receipt is read off the stdio tap: a
        # `notifications/cancelled` naming the in-flight id, a bare stdin EOF,
        # a signal, or neither (`cancel_exec.tap`).
    elif probe == "plugin-data":
        cache_home = home / ".codex"
        first = sorted(str(p) for p in cache_home.rglob("*plugin-data*"))
        again = _run_codex(["plugin", "add", CODEX_PLUGIN_ID, "--json"], env)
        second = sorted(str(p) for p in cache_home.rglob("*plugin-data*"))
        record["placement"] = {"before_reinstall": first,
                               "reinstall": _public(again),
                               "after_reinstall": second,
                               "moved": first != second}
        record["delivery"] = _plugin_data_delivery(home, root, overlay, stub_log)
    elif probe == "sandbox-posture":
        # 12(a) re-verified under the SHIPPED approve default: both sandboxed
        # postures must now reach the server with no "requires approval".
        record["read_only"] = codex_exec_leg(
            home, overlay, stub_log, "sandbox-read-only",
            extra_args=("--sandbox", "read-only"), tap=True)
        record["workspace_write"] = codex_exec_leg(
            home, overlay, stub_log, "sandbox-workspace-write",
            extra_args=("--sandbox", "workspace-write"), tap=True)
        for key in ("read_only", "workspace_write"):
            leg = record[key]
            leg["tool_call_item"] = _tool_call_item(Path(leg["events_file"]))
            leg["server_got_tools_call"] = "tools/call" in (
                (leg.get("tap") or {}).get("client_methods") or [])
            leg["requires_approval_refusal"] = "requires approval" in (
                (OUT / leg["events_file"]).read_text(encoding="utf-8")
                + leg.get("stderr", ""))
        # 12(b): read-only, an apply-mode call, the lever unset.
        record["refusal_read_only"] = scripted_leg(
            home, overlay, "refusal-read-only", APPLY_MODE_CALL,
            extra_args=("--sandbox", "read-only"))
        # 12(c): the same call with the lever set in codex's own environment,
        # and the session's OWN shell tool driven in the same turn.
        record["lift_read_only"] = scripted_leg(
            home, overlay, "lift-read-only", APPLY_MODE_CALL,
            extra_args=("--sandbox", "read-only"),
            env_extra={"MULTI_LLM_CODEX_UNSANDBOXED_WRITES": "1"},
            then_shell=f"touch {LIFT_SHELL_PROBE} && echo wrote-{LIFT_SHELL_PROBE}")
        record["lift_read_only"].setdefault("session_shell", {})["file_written"] = (
            WORKSPACE / LIFT_SHELL_PROBE).exists()
        # Workspace-write: an apply-mode call is not refused.
        record["apply_workspace_write"] = scripted_leg(
            home, overlay, "apply-workspace-write", APPLY_MODE_CALL,
            extra_args=("--sandbox", "workspace-write"))
        # The documented opt-out, measured: `default_tools_approval_mode =
        # "prompt"` in config.toml's plugin server table, approval_policy never.
        config = home / ".codex" / "config.toml"
        opt_out = (config.read_text(encoding="utf-8")
                   + f'\n[plugins."{CODEX_PLUGIN_ID}".mcp_servers.'
                     f'{CODEX_MCP_SERVER}]\n'
                     'default_tools_approval_mode = "prompt"\n')
        record["approval_opt_out"] = {"edit": opt_out.splitlines()[-2:]}
        with _FileEdit(config, opt_out):
            for posture in ("read-only", "workspace-write"):
                leg = scripted_leg(
                    home, overlay, f"opt-out-prompt-{posture}",
                    {"mode": "status", "plan": "plans/auth-feature.md",
                     "question": "", "extra_flags": []},
                    extra_args=("--sandbox", posture), timeout=120)
                record["approval_opt_out"][posture] = {
                    k: leg.get(k) for k in (
                        "exit", "timed_out", "wall_clock_s", "tool_call_item",
                        "server_got_tools_call", "requires_approval_refusal",
                        "stderr", "event_types", "item_types")}
                record["approval_opt_out"][posture]["envelope_state"] = (
                    leg.get("envelope") or {}).get("state")
        # 12(d): the tool dispatches a nested codex provider against the stub
        # from a workspace-write session; which posture did the grandchild run
        # under? Landlock is ENOSYS in-image, so rung 2 is the expected reading.
        nested = _nested_ask_leg(home, overlay, "nested-posture",
                                 "What does the plan change?",
                                 extra_args=("--sandbox", "workspace-write"),
                                 wait_worker_s=120)
        nested["posture_reading"] = _nested_rung(nested)
        record["nested_posture"] = nested
    elif probe == "overlay":
        record["codex_home_census"] = _tree_census(home / ".codex")
        record["overlay_candidates"] = [
            row["path"] for row in record["codex_home_census"]
            if "overlay" in row["path"].lower()]
    elif probe == "timeout":
        mcp = (root / ".mcp.json") if root else None
        before = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
        record["before"] = _json_or_none(before["_stdout_full"])
        if mcp and mcp.is_file():
            doc = json.loads(mcp.read_text(encoding="utf-8"))
            doc["mcpServers"][CODEX_MCP_SERVER]["tool_timeout_sec"] = 3600
            mcp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            after = _run_codex(["mcp", "get", CODEX_MCP_SERVER, "--json"], env)
            record["divergence"] = (
                f"{mcp}: tool_timeout_sec set to 3600 in the INSTALLED copy "
                f"(row 14's own edit; the shipped file is untouched)")
            record["after"] = _json_or_none(after["_stdout_full"])
            record["after_command"] = _public(after)
        # Row 14's second half: the cap FIRING on a detaching mode. Restore the
        # installed copy first (the 3600 above was this row's own edit).
        if mcp and mcp.is_file():
            doc = json.loads(mcp.read_text(encoding="utf-8"))
            doc["mcpServers"][CODEX_MCP_SERVER]["tool_timeout_sec"] = 900
            mcp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        record["cap_fires"] = _cap_fires_legs(home, root, overlay)

    path = OUT / f"probe-{probe}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    log(f"codex package probe {probe!r} recorded -> {path.name}"
        + (f" (not driven: {sorted(record['legs_not_driven'])})"
           if record["legs_not_driven"] else ""))
    return record


def wire_fallback_credentials(host: str, home: Path, slug: str,
                              route: str = "file",
                              effort: str = "") -> tuple[dict, list[Path]]:
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
            _codex_fallback_config(effort), encoding="utf-8")
        overlay["CODEX_HOME"] = str(codex_home)
        # Subscription auth stays in auth.json. The plugin's login-shell child
        # still needs the harness locations: Codex's MCP environment allowlist
        # strips them just as it does on the OpenRouter route.
        _seed_codex_plugin(home, route)
        if route == "package":
            export_harness_names_into_profile(home)
            wrap_profile_with_read_probe(home)

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


def jwt_claim_values(token: str) -> set[str]:
    """The identifying string claims inside a JWT's payload, or an empty set.

    Only what a by-value scrub needs: claims of 20+ characters (account and user
    ids) and anything carrying an ``@`` (emails). Not a verifier — the signature
    is never checked, and anything that does not decode is simply not a JWT.
    """
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return set()
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeError):
        return set()
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            if len(node) >= 20 or ("@" in node and len(node) >= 5):
                found.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(claims)
    return found


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
            # SHORT IDENTIFIERS TOO, not only long tokens. An email is under 20
            # characters often enough, and a codex `id_token` carries the
            # account's email and ids base64-encoded in its payload, where a
            # CLI that prints "logged in as …" would emit them in clear and no
            # by-value match on the token itself could catch them.
            elif "@" in node and len(node) >= 5:
                secrets.add(node)
            secrets.update(jwt_claim_values(node))
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


def build_host_argv(entry: dict, row: dict, creds: str = "openrouter") -> list[str]:
    """Render the row's `invoke_argv` from the matrix. No argv is improvised.

    `creds` picks the model the same way the credential wiring does: a
    `fallback` row drives the host's subscription pin, never an OpenRouter slug.
    """
    slug = pinned_model(entry, row, creds)
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
    # A ROW may carry its own template, and only the codex package row does:
    # the plugin advertises the skill namespaced (`multi-llm:multi-llm`, R8′),
    # so its prompt is the plan's row-16 sentence rather than the host's
    # file-route wording. Every other row renders exactly as before.
    template = row.get("prompt_template") or entry.get("prompt_template")
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
    argv = build_host_argv(entry, row, creds)
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


# --------------------------------------------------------------------------- #
# T10 of plans/next-phase-command.md: the fixture plan and the typed follow-up
# --------------------------------------------------------------------------- #

def install_fixture_plan(fixture: dict) -> Path:
    """Copy the row's fixture plan into place and return its path.

    A `next_command_check` row runs its phase on a plan whose path holds a
    SPACE, so the command the dispatcher prints carries the double-quoted form
    (§2.1) and the typed check can see whether the quoting survived the route.
    The plan is a byte copy of the seeded one (`source`), so the stub scenario
    that answers it is the one every other row already uses. Copied before the
    whole-row census is taken, so it is part of the workspace the row starts
    from rather than a write the row made.
    """
    source = WORKSPACE / str(fixture.get("source") or "")
    target = WORKSPACE / str(fixture.get("path") or "")
    if not str(fixture.get("path") or "").strip() or target == WORKSPACE:
        raise SystemExit("entrypoint: `fixture_plan.path` is empty")
    if not source.is_file():
        raise SystemExit(f"entrypoint: `fixture_plan.source` {source} is not a "
                         f"file in the seeded workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    log(f"fixture plan: {source.relative_to(WORKSPACE)} -> "
        f"{target.relative_to(WORKSPACE)!s}")
    return target


def _printed_next_step(receipt_dir: Path) -> dict | None:
    """The last `user_next_step` a receipt in *receipt_dir* carried, or ``None``.

    The receipts are the dispatcher's own result lines (`wrapper.run_uv` parses
    them with the product's parser), so this is the command the dispatcher
    PRINTED in that session — what a user would copy — and never one the rig
    composes from a template.
    """
    found = None
    receipts = []
    for path in sorted(receipt_dir.glob("*.receipt.json")) if receipt_dir.is_dir() else []:
        try:
            receipts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    for receipt in sorted(receipts, key=lambda r: int(r.get("seq") or 0)):
        payload = receipt.get("result") if isinstance(receipt, dict) else None
        if isinstance(payload, dict) and isinstance(payload.get("user_next_step"), dict):
            found = payload["user_next_step"]
    return found


def _receipt_mode(receipt: dict) -> str | None:
    argv = [str(t) for t in (receipt.get("argv") or [])]
    for index, token in enumerate(argv):
        if token.endswith("multi_llm.py") and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _typed_prompt(host: str, entry: dict, command: str) -> tuple[str, str]:
    """`(the string handed to the host, the surface it went through)`.

    The printed command is passed VERBATIM — no template around it — to the
    host's own headless surface, which is the measured way each host is
    launched (`invoke_argv`, `invoke_provenance: verified`):

    * codex and claude-code take a prompt, so the command IS the prompt — the
      exact line a user pastes into the composer.
    * opencode's headless form is `opencode run --command multi-llm -- <args>`
      (capabilities.yaml's verified `headless_invocation`): a bare
      `opencode run -- "/multi-llm …"` sends the line as literal text and never
      expands the command (measured on the documented-invocation row). So the
      command's own `/multi-llm ` prefix is what `--command multi-llm` names,
      and the rest is the argument string, byte for byte. A printed command
      without that prefix cannot be typed on this surface, and says so.
    """
    if host == "opencode":
        prefix = "/multi-llm "
        if not command.startswith(prefix):
            raise ValueError(
                f"the printed command {command!r} does not start with "
                f"{prefix!r}, the command name `--command multi-llm` types")
        return command[len(prefix):], "headless: opencode run --command multi-llm -- <args>"
    return command, "headless: the command as the whole prompt"


def run_typed_followup(host: str, entry: dict, row: dict, home: Path,
                       overlay: dict, key: str, creds: str = "openrouter",
                       secrets=()) -> dict:
    """T10's typed-template check: a FRESH session handed the printed command.

    After the row's own session has run its phase, the command that phase's
    dispatcher printed (`user_next_step.command`) is handed to a brand-new host
    process on the same route, in the same workspace — so the phase the command
    names really is the one that comes next. Its dispatcher calls land in their
    own receipt directory, `receipts/<row>-typed/`, so nothing graded about the
    first session can be satisfied (or spoiled) by the second.

    **It stops at the evidence, not at the end of the phase.** The question is
    whether the typed command reaches the dispatcher as the named mode with the
    plan argument intact — answered by the first receipt for that mode. The
    session is stopped a short grace after that receipt lands, so a write phase
    named by the command (`implement`, here) is dispatched and never driven to
    the end on the paid host model. A detached worker it launched keeps running
    on the stub, which costs nothing.

    Output goes to a file under /tmp (the container's own tmpfs) while the host
    runs, so polling the receipt directory can never fill a pipe and stall the
    host; it is scrubbed of the row's secrets before it reaches the bundle.
    """
    check = dict(row.get("next_command_check") or {})
    row_id = str(row.get("id"))
    receipts = OUT / "receipts" / f"{row_id}-typed"
    record: dict = {"host": host, "row": row_id, "exercised": False}
    step = _printed_next_step(OUT / "receipts" / row_id)
    if not step or not step.get("command") or not step.get("phase"):
        record["reason"] = ("the row's session printed no user_next_step with a "
                            "command, so there was nothing to type")
        (OUT / "typed-run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"typed follow-up: not run — {record['reason']}")
        return record
    command = str(step["command"])
    phase = str(step["phase"])
    try:
        typed, surface = _typed_prompt(host, entry, command)
    except ValueError as exc:
        record["reason"] = str(exc)
        (OUT / "typed-run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"typed follow-up: not run — {exc}")
        return record

    # The host's own launch cell, with the typed string as its whole prompt:
    # `prompt_template: "{args}"` renders it verbatim on the prompt hosts, and
    # opencode's host-level `null` template takes `args` verbatim already.
    wall = int(check.get("typed_wall_clock_s") or 300)
    typed_row = dict(row, args=typed, wall_clock_s=wall,
                     prompt_template=(None if host == "opencode" else "{args}"))
    argv = build_host_argv(entry, typed_row, creds)
    binary = argv[0]
    argv[0] = str(REAL_DIR / binary)

    receipts.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env.update(overlay)
    for var, value in (entry.get("env_extra") or {}).items():
        env[str(var)] = str(value)
    env["MLLM_RECEIPT_DIR"] = str(receipts)
    # The codex plugin's MCP server reaches the dispatcher through a login
    # shell that reads $HOME/.profile, never this environment
    # (`export_harness_names_into_profile`); a later export wins there.
    profile = home / ".profile"
    if profile.is_file():
        body = profile.read_text(encoding="utf-8")
        profile.write_text(body + ("" if body.endswith("\n") else "\n")
                           + f"export MLLM_RECEIPT_DIR={shlex.quote(str(receipts))}\n",
                           encoding="utf-8")
    if host == "claude-code" and creds == "openrouter":
        assert_claude_key_is_empty(env)

    def _scrub(text: str) -> str:
        text = text or ""
        for secret in ([key] if key else []) + list(secrets):
            text = text.replace(secret, f"<redacted:{creds}-credential>")
        return text

    grace = float(check.get("typed_grace_s") or 3.0)
    raw_out = Path("/tmp") / f"mllm-typed-{row_id}.out"
    raw_err = Path("/tmp") / f"mllm-typed-{row_id}.err"
    log(f"typed follow-up: {surface}; typing {typed!r} (printed command "
        f"{command!r}, named phase {phase!r}); wall-clock bound {wall}s")
    started = time.time()
    stop_reason = "host exited"
    dispatched_at = None
    timed_out = False
    with raw_out.open("w", encoding="utf-8") as out_handle, \
            raw_err.open("w", encoding="utf-8") as err_handle:
        proc = subprocess.Popen(argv, env=env, cwd=str(WORKSPACE), text=True,
                                stdout=out_handle, stderr=err_handle,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)
        while proc.poll() is None:
            elapsed = time.time() - started
            if dispatched_at is None:
                for path in sorted(receipts.glob("*.receipt.json")):
                    try:
                        receipt = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if _receipt_mode(receipt) == phase:
                        dispatched_at = elapsed
                        log(f"typed follow-up: {phase} dispatched after "
                            f"{elapsed:.1f}s ({path.name}); stopping in {grace}s")
                        break
            if dispatched_at is not None and elapsed >= dispatched_at + grace:
                stop_reason = "the named mode dispatched; session stopped"
                break
            if elapsed >= wall:
                stop_reason = "wall-clock"
                timed_out = True
                break
            time.sleep(1.0)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
    (OUT / "host-typed-stdout.txt").write_text(
        _scrub(raw_out.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    (OUT / "host-typed-stderr.txt").write_text(
        _scrub(raw_err.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    for path in (raw_out, raw_err):
        path.unlink(missing_ok=True)
    record.update({
        "exercised": True,
        "phase": phase,
        "printed_command": command,
        "typed": typed,
        "surface": surface,
        "argv": [binary, *argv[1:]],
        "exit": proc.returncode,
        "timed_out": timed_out,
        "stop_reason": stop_reason,
        "dispatched_after_s": None if dispatched_at is None else round(dispatched_at, 2),
        "wall_clock_s": round(time.time() - started, 2),
        "bound_s": wall,
        "receipts": str(receipts.relative_to(OUT)),
    })
    (OUT / "typed-run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"typed follow-up: {stop_reason} after {record['wall_clock_s']}s")
    return record


def plan_output_dir(plan: Path) -> Path:
    """Where the product writes *plan*'s output: its own `output_dir_path`.

    `<plans>/<stem>` only while the stem is already a safe prefix: the product
    sanitizes it (`output_handler.sanitize_prefix` — every character outside
    `[A-Za-z0-9_-]` becomes `_`), so T10's fixture `plans/next phase.md` writes
    under `plans/next_phase/`. Read from the product rather than retyped, the
    way `wrapper._write_receipt` reads the result line; the bare stem is only
    the fallback for a skill tree that cannot be imported, and it is exactly
    what every pre-T10 row's `auth-feature` resolves to either way.
    """
    try:
        if str(SKILL_DIR) not in sys.path:
            sys.path.insert(0, str(SKILL_DIR))
        from utils.output_handler import output_dir_path  # type: ignore

        return Path(output_dir_path(plan))
    except Exception:  # noqa: BLE001 - the export must never fail on this
        return plan.parent / plan.stem


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
    root = plan_output_dir(plan)
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
    source = plan_output_dir(plan)
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
    parser.add_argument("--probe", default=None,
                        choices=[*CODEX_PACKAGE_PROBES,
                                 *opencode_package_gate.GATE_PROBES],
                        help="host lane, --creds none --route package: which "
                             "zero-quota package-route probe this row runs. "
                             "--host codex: plans/codex-native-plugin.md Phase "
                             "1, writing probe-<name>.json. --host opencode: "
                             "row 8's `gate-*` scenarios "
                             "(plans/opencode-plugin-publish.md), writing "
                             "opencode-gate.json. The answer is recorded, never "
                             "judged here.")
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

    probe_host = ("opencode" if args.probe in opencode_package_gate.GATE_PROBES
                  else "codex")
    if args.probe and not (args.lane == "host" and args.host == probe_host
                           and args.route == "package" and args.creds == "none"):
        # A probe is a zero-quota package-route row by definition. Refused
        # rather than ignored: a `--probe` that silently fell through to the
        # discovery tier would record a green row that probed nothing.
        raise SystemExit(
            f"entrypoint: --probe {args.probe!r} runs only on the host lane with "
            f"--host {probe_host} --route package --creds none")

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
            # T10: a row that runs its phase on a fixture plan — a path with a
            # space — moves `plan` with it, so the terminal poll and the
            # artifact export read that plan's output directory.
            fixture = (_matrix_row(args.host, args.row).get("fixture_plan")
                       if args.lane == "host" and args.host else None)
            if fixture:
                plan = install_fixture_plan(dict(fixture))
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
                probe_overlay: dict = {}
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
                    # THE SUBSCRIPTION PIN FIRST (operator decision 2026-10-02):
                    # MLLM_HOST_MODEL, else the host's `subscription.model`, else
                    # the OpenRouter pin with the warning below. The pinned
                    # effort rides along only when the slug that runs IS the pin.
                    slug = pinned_model(entry, row, "fallback")
                    pinned_slug, pinned_effort = subscription_model(entry)
                    effort = pinned_effort if slug == pinned_slug else ""
                    overlay, copied = wire_fallback_credentials(
                        args.host, Path.home(), slug, route=args.route,
                        effort=effort)
                    secrets = fallback_secret_values(copied)
                    census = {
                        "host": args.host,
                        "row": args.row,
                        "credential_mode": "fallback",
                        "model": slug,
                        "model_source": (
                            "MLLM_HOST_MODEL"
                            if os.environ.get("MLLM_HOST_MODEL", "").strip()
                            else "host_matrix.yaml subscription.model"
                            if pinned_slug else
                            "host_matrix.yaml OpenRouter pin (no subscription pin)"),
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
                    if effort:
                        census["effort"] = effort
                    if (not os.environ.get("MLLM_HOST_MODEL", "").strip()
                            and not pinned_slug):
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
                        if args.host == "codex":
                            # The codex package route at zero quota: the host's
                            # own model pointed at the loopback stub (the
                            # scripted-call scenario answers the exec legs),
                            # then the plugin installed into that home. The
                            # config writer is the one the documented-invocation
                            # probe uses, with the route threaded through.
                            probe_overlay = wire_stub_host(
                                args.host, Path.home(),
                                f"http://127.0.0.1:{args.stub_port}/v1",
                                route=args.route,
                                install_source=("git" if args.probe == GIT_ROUTE_PROBE
                                                else "local-path"))
                        elif args.host != "opencode":
                            raise SystemExit(
                                f"entrypoint: --route package has no host config "
                                f"writer for host {args.host!r}. Only opencode "
                                f"and codex declare a package route, and a row "
                                f"that quietly wrote nothing would stage no "
                                f"route and then grade one.")
                        else:
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
                            "argv": build_host_argv(entry, row, args.creds),
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
                    build_host_argv(entry, row, args.creds)
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
                    # T10's typed-template check: a FRESH session of the same
                    # host and route, handed the command this session's
                    # dispatcher printed. Inside the row's spend bracket, so
                    # `usage.json` below charges both sessions to the row.
                    if (row.get("next_command_check") or {}).get("typed_followup"):
                        run_typed_followup(args.host, entry, row, Path.home(),
                                           overlay, key, creds=args.creds,
                                           secrets=secrets)
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

                if (args.creds == "none" and args.route == "package"
                        and args.host == "opencode" and args.probe):
                    # Row 8's package-gate scenarios. The wiring step above
                    # wrote the package declaration; the leg points the host's
                    # model at its own scripted stub and drives a real
                    # `opencode run` through the installed plugin. 0 unless the
                    # leg could not be driven at all — the verdict is
                    # test_container_opencode_package_gate.py's.
                    return opencode_package_gate.run(
                        args.probe, opencode_package_gate.GateContext(
                            out=OUT, home=Path.home(), workspace=WORKSPACE,
                            real_dir=REAL_DIR, plugin_dir=OPENCODE_PLUGIN_DIR,
                            plan=plan, stub_cls=Stub, log=log,
                            census=_workspace_census,
                            plan_output_dir=plan_output_dir))

                if (args.creds == "none" and args.route == "package"
                        and args.host == "codex"):
                    # The zero-quota package rows (Phase 1 items 1–14). One
                    # probe per row, named by the matrix; the seed itself ran in
                    # the wiring step above and wrote plugin-install.json. 0
                    # either way — a probe's answer is RECORDED and graded by
                    # test_container_host_lane.py, never decided here.
                    install_doc = _json_or_none(
                        (OUT / "plugin-install.json").read_text(encoding="utf-8")
                    ) if (OUT / "plugin-install.json").is_file() else None
                    run_codex_package_probe(
                        args.probe or "seed", Path.home(), probe_overlay,
                        OUT / "stub.jsonl", install_doc)
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
