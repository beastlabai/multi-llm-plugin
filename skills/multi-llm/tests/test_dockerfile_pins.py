#!/usr/bin/env python3
"""The Dockerfile's pins, asserted as text -- in plain CI, with no Docker (C9).

``tests/test_build_context.py`` proves what the build context *contains*; this
module proves what the Dockerfile *says*. They are deliberately separate files
because that one needs a daemon to export a context and skips itself without one,
and a pin guard that only runs where the container lane runs is a pin guard that
does not run. Nothing here shells out: every assertion is a read of
``tests/container/Dockerfile`` and ``tests/container/cli-versions.lock``.

Four properties, each of which has a specific way of rotting quietly:

1. **No ``FROM`` by tag.** ``node:22`` becomes a new Node minor on a new Debian
   without a single byte of this repository changing, and then a golden diff is
   unattributable. Only ``@sha256:`` is a pin.
2. **No ``@latest`` on the pinned build path.** The drift probe resolves the CLI
   layer at ``@latest`` on purpose; the gate must not, and the difference has to
   be a guard on ``MLLM_PIN_MODE`` rather than a comment.
3. **No ``curl``-to-shell, and no download without a checksum.** ``curl … | sh``
   makes an installer a live remote input running as root. A download whose
   ``RUN`` never mentions ``sha256sum --check`` is the same hole with an extra
   step.
4. **The Dockerfile's ARGs and cli-versions.lock agree.** The URL and sha256 of
   every artifact exist in two files; two copies of a checksum that can disagree
   are worse than one, so their agreement is asserted rather than maintained.

Plus the two structural claims Phase 2's "Done when" turns on: the image's final
``USER`` is the numeric non-root identity ``posture.yaml`` declares, and every
CLI the lock names has a wrapper entry, so a CLI added to the lock without a
``/opt/mllm/bin`` link fails here rather than as a stray unredirected row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

#: Repo root: tests/ -> multi-llm/ -> skills/ -> <repo>.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTAINER_DIR = REPO_ROOT / "skills/multi-llm/tests/container"
DOCKERFILE = CONTAINER_DIR / "Dockerfile"
LOCK = CONTAINER_DIR / "cli-versions.lock"
PACKAGE_JSON = CONTAINER_DIR / "package.json"
PACKAGE_LOCK = CONTAINER_DIR / "package-lock.json"

#: The build-arg that selects the pinned gate or the drift probe. An instruction
#: that mentions @latest has to be guarded by it.
PIN_MODE_ARG = "MLLM_PIN_MODE"


def _logical_lines(text: str) -> list[str]:
    """Join continuations and drop comment lines, preserving instruction order.

    A Dockerfile assertion that reads raw lines is trivially defeated by a
    backslash: the interesting instructions here are all multi-line.
    """
    logical: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if buffer and stripped.startswith("#"):
            # A comment inside a continuation is not a thing Docker allows, but
            # be explicit rather than silently folding it into the command.
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1].rstrip() + " "
            continue
        logical.append(buffer + stripped)
        buffer = ""
    if buffer:
        logical.append(buffer)
    return logical


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.is_file(), f"{DOCKERFILE} is missing"
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def instructions(dockerfile_text) -> list[str]:
    return _logical_lines(dockerfile_text)


@pytest.fixture(scope="module")
def lock() -> dict:
    assert LOCK.is_file(), f"{LOCK} is missing"
    return yaml.safe_load(LOCK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def build_args(instructions) -> dict[str, str]:
    args: dict[str, str] = {}
    for line in instructions:
        match = re.match(r"ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if match:
            args[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return args


# --------------------------------------------------------------------------- #
# 1. Base image
# --------------------------------------------------------------------------- #


def test_every_from_is_pinned_by_digest(instructions):
    froms = [line for line in instructions if re.match(r"FROM\s", line, re.I)]
    assert froms, "the Dockerfile declares no FROM"
    for line in froms:
        reference = line.split()[1]
        assert "@sha256:" in reference, (
            f"{line!r} pins by tag. `node:22` silently becomes a new Node minor "
            "on a new Debian, which moves the runtime under every CLI at once "
            "and makes a golden diff unattributable. Only a digest is a pin."
        )
        assert re.search(r"@sha256:[0-9a-f]{64}$", reference), (
            f"{reference!r} does not end in a 64-hex digest"
        )


def test_base_digest_matches_the_lock(instructions, lock):
    reference = next(line.split()[1] for line in instructions
                     if re.match(r"FROM\s", line, re.I))
    repo, _, digest = reference.partition("@")
    assert digest == lock["base_image"]["digest"], (
        "the Dockerfile's FROM digest and cli-versions.lock disagree; the lock is "
        "what the run record and the drift report cite, so they cannot differ"
    )
    assert repo == lock["base_image"]["ref"], (
        f"FROM names {repo!r}, cli-versions.lock names "
        f"{lock['base_image']['ref']!r}"
    )


def test_the_pinned_platform_is_stated_and_enforced(dockerfile_text, lock):
    assert lock["platform"] == "linux/amd64"
    # The Makefile passes --platform, but a hand-typed `docker build` does not,
    # and an arm64 machine would then silently produce a different image.
    assert "uname -m" in dockerfile_text, (
        "nothing in the Dockerfile checks the build architecture, so a build "
        "without --platform on an arm64 machine would produce goldens from "
        "different binaries and never say so"
    )
    assert lock["arch"] in dockerfile_text


# --------------------------------------------------------------------------- #
# 2. @latest belongs to the drift probe alone
# --------------------------------------------------------------------------- #


def test_no_unguarded_latest(instructions):
    offenders = [
        line for line in instructions
        if "@latest" in line and PIN_MODE_ARG not in line
    ]
    assert not offenders, (
        "these instructions resolve @latest without a "
        f"{PIN_MODE_ARG} guard, so the PINNED build -- the gate you run before a "
        "release -- would install unreviewed, unhashed binaries:\n  "
        + "\n  ".join(o[:160] for o in offenders)
    )


def test_the_latest_branch_is_a_real_conditional(instructions):
    latest = [line for line in instructions if "@latest" in line]
    assert latest, (
        "no instruction resolves @latest at all -- MLLM_PIN_MODE=latest would "
        "then build the same image as the gate and the drift probe would report "
        "nothing"
    )
    for line in latest:
        assert re.search(rf'\[\s*"?\$\{{{PIN_MODE_ARG}\}}"?\s*=\s*"?latest"?\s*\]', line), (
            f"{line[:160]!r} mentions {PIN_MODE_ARG} but not as a `[ \"$"
            f"{{{PIN_MODE_ARG}}}\" = \"latest\" ]` test; a guard that is not a "
            "conditional is a comment"
        )


# --------------------------------------------------------------------------- #
# 3. Downloads
# --------------------------------------------------------------------------- #

_SHELL_PIPE = re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b")


def test_no_curl_to_shell(instructions):
    offenders = [line for line in instructions if _SHELL_PIPE.search(line)]
    assert not offenders, (
        "a download piped into a shell is a live remote input executing as root "
        "on every build, which is the exact opposite of this file's premise:\n  "
        + "\n  ".join(o[:160] for o in offenders)
    )


def test_every_download_verifies_a_checksum(instructions):
    """A `curl` whose RUN never checks a hash is `curl | sh` with an extra step.

    The exception is a fetch that only *resolves a version* -- reading a channel
    pointer in the drift probe -- which by construction has nothing to pin.
    """
    downloads = [
        line for line in instructions
        if re.match(r"RUN\s", line, re.I) and re.search(r"\b(curl|wget)\b", line)
    ]
    assert downloads, "no download instruction found; did the artifact steps move?"

    for line in downloads:
        if "sha256sum" in line:
            continue
        # A version-resolution fetch writes to stdout and never to a file the
        # build then executes.
        writes_a_file = re.search(r"-o\s+\S+|--output\s+\S+", line)
        assert not writes_a_file, (
            "this RUN downloads a file and never verifies a checksum:\n  "
            + line[:300]
        )


@pytest.mark.parametrize(
    "artifact",
    ["uv", "goose", "grok", "muse", "cursor-agent", "agy", "omp", "fx", "python312"])
def test_artifact_args_match_the_lock(artifact, build_args, lock):
    """Two copies of a checksum that can disagree are worse than one."""
    entry = lock["artifacts"][artifact]
    # A lock artifact key is the CLI's id, which may carry a hyphen
    # (`cursor-agent`); a Dockerfile ARG may not, so the two spellings differ by
    # exactly this substitution and by nothing else.
    prefix = artifact.upper().replace("-", "_")
    assert build_args[f"{prefix}_VERSION"] == str(entry["version"]), (
        f"ARG {prefix}_VERSION and cli-versions.lock disagree"
    )
    assert build_args[f"{prefix}_SHA256"] == entry["sha256"], (
        f"ARG {prefix}_SHA256 and cli-versions.lock disagree -- one of them is "
        "verifying bytes the other one does not name"
    )
    if f"{prefix}_URL" in build_args:
        assert build_args[f"{prefix}_URL"] == entry["url"]
    else:
        # uv's URL is assembled from its version in the RUN itself.
        assert str(entry["version"]) in entry["url"]


def test_fx_checks_identity_and_version_before_discarding_the_archive(instructions, lock):
    from utils.providers.fx import FX_REQUIRED_FLAGS

    [layer] = [line for line in instructions
               if line.startswith("RUN ") and "install -m 0755 /tmp/fx " in line]
    assert layer.index("sha256sum --check --strict") < layer.index("tar -xzf")
    assert '/opt/mllm/real/fx --version)' in layer and '= "${fx_version}"' in layer
    assert "/opt/mllm/real/fx ask --help" in layer
    assert "grep -Fq 'fx ask'" in layer
    for flag in FX_REQUIRED_FLAGS:
        assert flag in layer
    assert 'grep -Fq -- "$flag" /tmp/fx-help || exit 1' in layer
    assert "FX_AUTO_UPGRADE=0" in layer
    artifact = lock["artifacts"]["fx"]
    assert artifact["url"].count(artifact["version"]) == 1
    [row] = [row for row in lock["clis"] if row["id"] == "fx"]
    assert row["install"] == "archive"


# --------------------------------------------------------------------------- #
# 4. The CLI set, in three files that must agree
# --------------------------------------------------------------------------- #


def test_every_lock_cli_is_wrapped(dockerfile_text, lock):
    """A CLI in the lock with no /opt/mllm/bin entry is an unredirected row.

    C2's whole mechanism is that the wrapper is first on PATH. A CLI installed
    into /opt/mllm/real and never linked into /opt/mllm/bin would run against
    whatever backend it defaults to, which on the host lane means a real one.

    This is every lock row, with no exemption for a row that reaches no stub.
    ``cursor-agent`` is the case that will tempt one: it has no redirect.yaml
    row, so linking it buys no redirection. It is still required here, because
    ``verify_versions.py``'s ``check_path_shape`` walks this same ``clis`` block
    and fails the build on any binary that does not resolve to /opt/mllm/bin --
    so an "obviously pointless" link dropped from the loop is a red build, and
    this assertion is where that gets said in plain CI rather than after a
    twenty-minute image build.
    """
    wrapper_loop = re.search(
        r"for b in ([^;]+); do\s*\\?\s*ln -sf /opt/mllm/rig/wrapper\.py",
        dockerfile_text,
    )
    assert wrapper_loop, "the wrapper symlink loop is not where this test expects it"
    wrapped = set(wrapper_loop.group(1).split())

    missing = [row["binary"] for row in lock["clis"] if row["binary"] not in wrapped]
    assert not missing, (
        f"cli-versions.lock names {missing}, which the Dockerfile never links "
        "into /opt/mllm/bin -- those rows would run unredirected"
    )
    # `uv` is not a CLI row but must be wrapped: it is what writes the receipts.
    assert "uv" in wrapped, (
        "/opt/mllm/bin/uv is what produces the run-id-correlated receipts every "
        "host-lane assertion anchors on (C6c)"
    )


def test_npm_rows_match_package_json(lock):
    manifest = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    declared = manifest["dependencies"]
    npm_rows = {row["package"]: str(row["version"])
                for row in lock["clis"] if row["install"] == "npm"}

    assert set(declared) == set(npm_rows), (
        "tests/container/package.json and the lock's npm rows name different "
        f"packages: {sorted(set(declared) ^ set(npm_rows))}"
    )
    for package, version in npm_rows.items():
        assert declared[package] == version, (
            f"package.json pins {package}@{declared[package]}, the lock says "
            f"{version}"
        )


#: The one place `npm ci` is NOT a byte-level pin, named rather than tolerated.
#:
#: `@earendil-works/pi-coding-agent` publishes with `hasShrinkwrap: true`, and its
#: own npm-shrinkwrap.json carries no `integrity` for these siblings -- so npm
#: records them with a resolved URL and no hash, even though the registry does
#: publish an integrity for each. They are installed by URL alone.
#:
#: The exposure is bounded: all of them are pinned to an exact version, come from
#: the same publisher as the parent (which IS hashed), and reach only the `pi`
#: row of the provider lane, which runs with --network none and no credential.
#: The list is asserted EXACTLY -- an unlisted entry appearing here fails, and so
#: does an entry that upstream later fixes -- because a "known gap" nobody
#: re-reads is just a gap.
#:
#: WIDENED 3 -> 5 on 2026-09-19, for the pi 0.80.6 -> 0.85.1 bump. `chord` and
#: `pi-telemetry` are two more siblings of the same vendor, nested in the same
#: shrinkwrap, unhashed for the same reason -- no new publisher and no new
#: mechanism. `lock_update.py` refuses to move the npm layer while they are
#: unlisted, which is what surfaced the decision rather than letting the pin move
#: carry it silently; it is recorded here as a decision taken, with a date.
#:
#: WIDENED 5 -> 7 on 2026-09-30, for the pi 0.87.1 -> 0.99.1 bump. `pi-codemode`
#: and `pi-mcp` are two more siblings of the same vendor, nested in the same
#: shrinkwrap at the same 0.99.1 version as the five already listed -- no new
#: publisher (same `@earendil-works` maintainers as the hashed parent) and no new
#: mechanism (`npm view @earendil-works/pi-codemode@0.99.1 dist.integrity` and the
#: `pi-mcp` equivalent both return a sha512, confirming the registry publishes an
#: integrity for each; the shrinkwrap just omits it). Checked before widening, per
#: this file's own rule: the registry-hash and same-publisher facts were verified
#: against the real `@latest` image before this change, not assumed from the
#: pattern.
#:
#: EMPTIED 7 -> 0 on 2026-10-08, for the pi 0.99.1 -> 1.1.0 bump. 1.1.0 no longer
#: publishes with `hasShrinkwrap: true`: npm hoists the seven siblings to the top
#: level of the tree and records an `integrity` for every one of them, so the
#: lockfile has no unhashed entry left anywhere (the test below found all seven
#: stale). The set stays, empty, so the next unhashed entry is a decision again.
#: (`set()` is not a literal; `lock_update._adjudicated_unhashed` reads that as
#: the empty set, its documented safe direction.)
UNHASHED_BY_UPSTREAM: set = set()


def test_package_lock_carries_integrity_hashes():
    """`npm ci` is only a pin because the lockfile hashes every tarball."""
    data = json.loads(PACKAGE_LOCK.read_text(encoding="utf-8"))
    assert data.get("lockfileVersion", 0) >= 2, (
        "lockfileVersion 1 has no integrity for transitive deps"
    )
    packages = data["packages"]
    resolved = [name for name in packages if name]
    assert len(resolved) > 100, "the CLI lockfile looks truncated"

    missing = {
        name for name, entry in packages.items()
        if name and entry.get("resolved") and not entry.get("integrity")
    }
    unexpected = missing - UNHASHED_BY_UPSTREAM
    assert not unexpected, (
        f"{len(unexpected)} lockfile entries have a resolved URL and no integrity "
        "hash, so a re-published tarball behind an unchanged version would "
        f"install unnoticed: {sorted(unexpected)[:10]}"
    )
    stale = UNHASHED_BY_UPSTREAM - missing
    assert not stale, (
        "UNHASHED_BY_UPSTREAM names entries that are now hashed; drop them from "
        f"the allowlist rather than carrying a permanent excuse: {sorted(stale)}"
    )


def test_the_npm_layer_makes_qwens_vendored_ripgrep_executable(instructions):
    """qwen-code ships vendor/ripgrep/<platform>/rg mode 0644 and chmods it on
    first use, which fails in this root-owned read-only install: every qwen run
    as uid 10001 printed `Ripgrep not available: spawn .../rg EACCES` on stderr
    (2026-10-08, qwen-code 0.25.0). The chmod must sit in the same RUN as
    `npm ci`, the layer that writes the file, or it is a second copy of 4 MB."""
    [layer] = [line for line in instructions
               if line.startswith("RUN ") and "npm ci" in line]
    target = "@qwen-code/qwen-code/vendor/ripgrep/x64-linux/rg"
    assert target in layer and re.search(r"chmod 0755 \S*qwen_rg", layer), (
        "the npm layer no longer makes qwen's vendored x64-linux ripgrep "
        "executable"
    )


def test_pypi_rows_are_hash_pinned(lock):
    for row in lock["clis"]:
        if row["install"] != "pypi":
            continue
        requirements = CONTAINER_DIR / row["requirements"]
        assert requirements.is_file(), f"{requirements} is missing"
        text = requirements.read_text(encoding="utf-8")
        assert f"{row['package']}=={row['version']}" in text, (
            f"{requirements.name} does not pin {row['package']} to "
            f"{row['version']}"
        )
        assert "--hash=sha256:" in text, (
            f"{requirements.name} carries no hashes, so --require-hashes is "
            "asserting nothing"
        )


# --------------------------------------------------------------------------- #
# 5. Posture and the two rules the build context depends on
# --------------------------------------------------------------------------- #


def test_final_user_is_the_declared_non_root_identity(instructions):
    posture = yaml.safe_load(
        (CONTAINER_DIR / "posture.yaml").read_text(encoding="utf-8"))
    identity = posture["identity"]

    users = [line for line in instructions if re.match(r"USER\s", line, re.I)]
    assert users, "the image never leaves root"
    assert users[-1].split(None, 1)[1].strip() == f"{identity['uid']}:{identity['gid']}", (
        f"the final USER is {users[-1]!r}; posture.yaml declares "
        f"{identity['uid']}:{identity['gid']}, and the numeric form is the point "
        "-- `USER e2e` would silently follow a renamed account"
    )

    if identity.get("requires_passwd_entry"):
        assert any("useradd" in line for line in instructions), (
            "no /etc/passwd entry is created; several agent CLIs mishandle a uid "
            "with no passwd entry, which is exactly the shape `--user $(id -u)` "
            "produces"
        )
    if identity.get("requires_owned_home"):
        assert any("chown" in line and "/home/e2e" in line for line in instructions), (
            "/home/e2e is never chowned to the run identity; gemini crashes "
            "outright when $HOME does not exist or cannot be written"
        )


def test_no_bare_copy_of_the_context(instructions):
    for line in instructions:
        if not re.match(r"COPY\s", line, re.I):
            continue
        args = line.split()[1:]
        args = [a for a in args if not a.startswith("--")]
        assert args[:1] != ["."], (
            "`COPY .` drags the repo root's .env, dogfooded .multi-llm/ and any "
            "scratch credential file into a layer; named paths only (C6b rule 2)"
        )


def test_the_skill_env_is_materialized_at_build_time(dockerfile_text, instructions):
    """The read-only root plus --network none makes a run-time sync unfixable.

    `--frozen` alone uses the lockfile without checking it, so a uv.lock out of
    step with pyproject.toml would sail through the build and die four phases
    later inside a container nobody can attach to. `--locked` is the half that
    has to be a build failure.
    """
    assert re.search(r"uv sync[^\n]*--locked", dockerfile_text), (
        "the skill's environment is not materialized with `uv sync --locked` at "
        "build time"
    )
    # Instructions, not the file text: the Dockerfile's own comment explains why
    # UV_OFFLINE is absent, and a guard that its own rationale trips is useless.
    assert not [line for line in instructions if "UV_OFFLINE" in line], (
        "UV_OFFLINE must not be baked into the image: the BUILD is the one point "
        "in the pipeline with an index to reach. harness.py passes it per run, "
        "from posture.yaml"
    )
