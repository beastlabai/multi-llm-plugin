#!/usr/bin/env python3
"""No committed secrets — the guard that runs in ordinary CI on every PR.

``beastlabai/multi-llm-plugin`` is a **public** repository: anything committed is
published. ``.gitignore`` keeps credentials out by *location* and prevents the
accident; this module catches the one that slips through, by walking
``git ls-files`` and running gitleaks over exactly the files git tracks. No
Docker, no ``container`` marker — an unconditional gate is the only kind that is
still there the day it matters.

**Why `git ls-files` and not the working tree.** A scanner that walks the
filesystem reports on ``.venv/``, on this repo's own ignored ``plans/`` scratch
and on whatever a developer left lying around — noise that is not a leak,
because none of it is published. The tracked set *is* the published set, so it is
the set that gets scanned. The scan runs against a copy of that set mirrored into
``tmp_path`` (paths preserved), which keeps the tracked/untracked distinction a
property of the scan rather than a post-hoc filter over its output.

**Why gitleaks' stock pack and not our own regexes.** Two false-positive families
would make a hand-rolled scanner red on its first run against this tree, and a
scanner that is red on day one gets silenced on day two:

* *Prefixes.* ``mixed_importance.json`` ships an ``API_KEY = "sk-…"`` assignment
  as the anchor text of a review finding, and ``uv.lock`` is hundreds of 64-char
  sha256 digests that a bare entropy floor would hit.
* *Field names.* ``tests/fixtures/e2e/plans/auth-feature.md`` specifies a
  ``sessions`` table with a ``refresh_token VARCHAR(255)`` column and a
  ``refresh(refresh_token)`` endpoint — schema and API field names with no value
  attached.

The stock pack qualifies every vendor prefix with length, charset and (where the
vendor defines one) a checksum, and its ``generic-api-key`` shape needs an
identifier *plus* an adjacent value clearing length, charset and entropy floors.
So those decoys are non-matches **by construction** rather than by exception —
which ``TestDecoysStayUnmatched`` pins, so a later "improvement" that keys on a
prefix or a field name fails here instead of being absorbed into an exclusion.

**What is asserted, and why each assertion exists:**

* ``TestConfigShape`` — the committed ``.gitleaks.toml`` parses, extends the
  stock pack, and carries no file-level suppression: ``paths`` may appear only
  under a ``[[rules]]`` block, only beside ``regexes``, and only with
  ``condition = "AND"``. A blind spot hides the *next* key pasted there, not just
  today's benign match. Every entry must also carry a rule ID, a rationale, a
  date and its match fingerprints.
* ``TestGitleaksAllowlistSemantics`` — *why* the rule above has that exact shape,
  measured rather than assumed. The obvious spelling of "this value in this file"
  (a top-level ``[[allowlists]]`` with ``paths`` and ``regexes``) makes gitleaks
  skip the whole file before any rule runs, and the ``targetRules`` form the
  design note reaches for first is a silent no-op in 8.28.0. Both facts are
  pinned here so a gitleaks upgrade that changes them fails loudly instead of
  quietly widening every exception the config carries.
* ``TestTrackedTreeIsClean`` — the scan itself, run with an explicit
  ``--config`` whose path is asserted to resolve *and* whose loading is read back
  out of gitleaks' own debug log. A renamed or typo'd config would otherwise send
  gitleaks to its stock defaults and the suite would go green for the wrong
  reason.
* ``TestAllowlistEntriesAreLive`` — each entry names a path that still exists and
  a regex that still matches inside it, and **removing any single entry turns the
  scanner red**. An entry that changes nothing is decoration, and decoration is
  how a ``paths``-shaped exclusion sneaks back in later. The set is also asserted
  to be no larger than what actually fires, so "empty" is a measurement rather
  than an omission.
* ``TestPlantedCredentialsAreDetected`` — genuine-shape credentials **synthesized
  at runtime** (prefix + body + the vendor's checksum, assembled in code) are
  planted into a JSON fixture, a YAML config and Markdown, and separately into
  every file carrying an allowlist entry, where each must *still* be detected.
  That second half is what proves an exception is scoped to its one known value
  rather than granting its file a pass. Vectors are never committed: the
  scanner's own test data must not become the one committed secret nobody scans.
* ``TestGitignoreDirections`` — ``git check-ignore -v`` in **both** directions
  over a fixed list. Every committed value-free path comes back un-ignored; every
  real credential location comes back ignored, with the rule that matched it
  named. Most of those paths do not exist yet, which is the point:
  ``check-ignore`` answers for hypothetical paths, so the rules are proven before
  the harness that writes them lands.

**Plants never touch this repo's working tree.** Each case builds a scratch git
repo under ``tmp_path`` — ``git init``, this repo's ``.gitignore`` and
``.gitleaks.toml`` copied in verbatim so the same ignore set and the same rules
are exercised — writes the plant there, ``git add -A``, and scans that tree. An
untracked plant in the real tree would be invisible to a ``git ls-files`` walk; a
tracked one would be a real-shape credential one commit from being the leak this
module exists to prevent. Because the copied ignore set is the real one, every
plant asserts ``git check-ignore`` on its own path *before* asserting detection,
so a pattern that shadows the plant fails by name as an ignore-rule collision
rather than being "fixed" with a path allowlist — the one exclusion shape the
config forbids.

The gitleaks binary is deliberately **not vendored**; when it is absent the
module skips with a reason that names the fix.
"""

import hashlib
import json
import os
import random
import re
import shutil
import string
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest
import yaml

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


# tests/ -> multi-llm/ -> skills/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".gitleaks.toml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

GITLEAKS = shutil.which("gitleaks")

# The skill dir is symlinked into consuming repos by `link-skill.sh`, and the
# Claude plugin cache holds an unpacked copy. Neither is a checkout of this
# repository, and "which files are published" is not a question that has an
# answer there — so the walk is skipped rather than failed. CI always checks out
# a real work tree, so this can never hide a red scan on a PR.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists(),
    reason=f"{REPO_ROOT} is not a git work tree; there is no tracked set to scan",
)

requires_gitleaks = pytest.mark.skipif(
    GITLEAKS is None,
    reason=(
        "gitleaks is not on PATH, so the committed-secret scan cannot run. "
        "The binary is deliberately not vendored into this repo — install it "
        "(https://github.com/gitleaks/gitleaks/releases) locally, and add an "
        "install step to .github/workflows/test.yml so CI runs this guard "
        "rather than skipping it."
    ),
)

requires_tomllib = pytest.mark.skipif(
    tomllib is None,
    reason="no TOML parser available (Python <3.11 without `tomli`)",
)


# ---------------------------------------------------------------------------
# git / gitleaks plumbing
# ---------------------------------------------------------------------------


# Git reads no global/system config when these point at a path that does not
# exist — the portable spelling of "neutralized" (`/dev/null` is not one on
# Windows, and this suite runs there too).
_NO_GITCONFIG = str(Path(tempfile.gettempdir()) / "multi-llm-tests-absent-gitconfig")


def _git(args, cwd, check=True):
    """Run git with the ambient user/system config neutralized.

    A scratch repo must answer for *this repo's* committed ignore set and
    nothing else. A developer's ``core.excludesFile`` or a system gitattributes
    would otherwise make these assertions machine-dependent — green here, red on
    someone else's laptop, for a reason that has nothing to do with the tree.
    """
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = _NO_GITCONFIG
    env["GIT_CONFIG_SYSTEM"] = _NO_GITCONFIG
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def _tracked_files(root=REPO_ROOT):
    """The published set: exactly what ``git ls-files`` reports, POSIX-separated."""
    out = _git(["ls-files", "-z"], cwd=root).stdout
    return [p for p in out.split("\0") if p]


def _mirror_tracked(root, dest):
    """Copy the tracked set into ``dest``, repo-relative paths preserved.

    Preserving the relative paths is not cosmetic: the ``paths`` regexes in an
    allowlist entry are written against repo-relative paths, so a mirror that
    flattened or re-rooted them would exercise different rules than CI does.
    """
    copied = []
    for rel in _tracked_files(root):
        src = root / rel
        if not src.is_file():  # staged-but-deleted; nothing to scan
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    return copied


def _run_gitleaks(target, config, report_dir, log_level="error"):
    """Scan ``target`` with ``config`` in force; return ``(findings, stderr)``.

    ``--ignore-gitleaks-allow`` and an empty ``--gitleaks-ignore-path`` close the
    two suppression channels that live outside the config file: an inline
    ``gitleaks:allow`` comment and a ``.gitleaksignore``. Both are exceptions
    nobody reviews; ``.gitleaks.toml`` is the single exception channel, and
    ``TestConfigShape`` asserts no ``.gitleaksignore`` is tracked.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "gitleaks-report.json"
    ignore_dir = report_dir / "no-gitleaksignore"
    ignore_dir.mkdir(exist_ok=True)

    proc = subprocess.run(
        [
            GITLEAKS,
            "dir",
            ".",
            "--no-banner",
            "--no-color",
            "--exit-code",
            "0",
            "--ignore-gitleaks-allow",
            "--gitleaks-ignore-path",
            str(ignore_dir),
            "--config",
            str(config),
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--log-level",
            log_level,
        ],
        cwd=str(target),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"gitleaks failed to run (exit {proc.returncode}).\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    findings = json.loads(report.read_text()) if report.exists() else []
    return findings or [], proc.stderr


def _describe(findings):
    return "\n".join(
        f"  {f['RuleID']:<24} {f['File']}:{f['StartLine']}  ({f['Fingerprint']})"
        for f in findings
    )


@pytest.fixture(scope="module")
def tracked_scan(tmp_path_factory):
    """Scan the tracked tree once, with the committed config explicitly in force."""
    if GITLEAKS is None:
        pytest.skip(requires_gitleaks.kwargs["reason"])
    root = tmp_path_factory.mktemp("tracked")
    mirror = root / "tree"
    mirror.mkdir()
    files = _mirror_tracked(REPO_ROOT, mirror)
    findings, stderr = _run_gitleaks(mirror, CONFIG_PATH, root / "out", log_level="debug")
    return {"findings": findings, "stderr": stderr, "files": files, "mirror": mirror}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _config_text():
    return CONFIG_PATH.read_text(encoding="utf-8")


def _config_data():
    return tomllib.loads(_config_text())


def _entries():
    """Every allowlist table in the config that can suppress a finding.

    Returns ``(label, scope, entry)`` where ``scope`` is ``"global"`` for a
    top-level ``[allowlist]``/``[[allowlists]]`` and ``"rule"`` for one hanging
    off a ``[[rules]]`` block. The distinction is not cosmetic — see
    ``TestGitleaksAllowlistSemantics``: a ``paths`` entry means something
    categorically different in each position.
    """
    if tomllib is None:
        return []
    data = _config_data()
    out = []
    if isinstance(data.get("allowlist"), dict):
        out.append(("[allowlist]", "global", data["allowlist"]))
    for i, entry in enumerate(data.get("allowlists") or []):
        out.append((f"[[allowlists]][{i}]", "global", entry))
    for rule in data.get("rules") or []:
        rid = rule.get("id", "<unnamed rule>")
        if isinstance(rule.get("allowlist"), dict):
            out.append((f"[[rules]] {rid} [rules.allowlist]", "rule", rule["allowlist"]))
        for i, entry in enumerate(rule.get("allowlists") or []):
            out.append((f"[[rules]] {rid} [[rules.allowlists]][{i}]", "rule", entry))
    return out


def _entry_block(entry):
    """The raw source lines of one allowlist entry, located by its description.

    The date and the match fingerprints an entry must carry are TOML *comments*
    (``# fingerprints: …``), which a parser drops. Reading the raw source keeps
    them assertable, so the provenance of an exception cannot quietly rot away
    while the exception itself stays in force. It is also what
    ``test_removing_any_entry_turns_the_scanner_red`` deletes.
    """
    description = (entry.get("description") or "").strip()
    assert description, "every allowlist entry needs a `description`; see the config"
    lines = _config_text().splitlines()

    def is_header(line):
        stripped = line.strip()
        return stripped.startswith("[") and not stripped.startswith("#")

    matches = [
        i
        for i, line in enumerate(lines)
        if line.lstrip().startswith("description") and description in line
    ]
    assert len(matches) == 1, (
        f"allowlist descriptions must be unique and single-line; {description!r} "
        f"matched {len(matches)} source lines"
    )
    start = matches[0]
    while start > 0 and not is_header(lines[start]):
        start -= 1
    end = matches[0] + 1
    while end < len(lines) and not is_header(lines[end]):
        end += 1
    return "\n".join(lines[start:end])


def _entry_files(entry):
    """Tracked files an entry's ``paths`` regexes actually select."""
    patterns = [re.compile(p) for p in entry.get("paths") or []]
    if not patterns:
        return []
    return [rel for rel in _tracked_files() if any(p.search(rel) for p in patterns)]


def _matches_somewhere(entry, files):
    regexes = [re.compile(r) for r in entry.get("regexes") or []]
    return any(
        rx.search((REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        for rel in files
        for rx in regexes
    )


# ---------------------------------------------------------------------------
# Runtime-synthesized credential vectors
# ---------------------------------------------------------------------------

_B62 = string.digits + string.ascii_lowercase + string.ascii_uppercase


def _b62(value, width):
    out = ""
    while value:
        value, rem = divmod(value, 62)
        out = _B62[rem] + out
    return out.rjust(width, "0")


def _body(rng, n, alphabet=string.ascii_letters + string.digits):
    return "".join(rng.choice(alphabet) for _ in range(n))


def _github_pat(rng):
    """A genuine-shape GitHub PAT: ``ghp_`` + 30-char body + base62 CRC32 checksum.

    The checksum is what makes this a *value* rather than a prefix — the point
    the whole rule set turns on. It is assembled here and never written to this
    repository.
    """
    body = _body(rng, 30)
    return "ghp_" + body + _b62(zlib.crc32(body.encode()) % (62**6), 6)


def _anthropic_key(rng):
    """A genuine-shape Anthropic key: ``sk-ant-api03-`` + 93 chars + the ``AA`` tail.

    Anthropic publishes no checksum, so length and charset carry the whole
    burden; that is exactly why the bare ``sk-`` prefix must never be a rule.
    """
    return "sk-ant-api03-" + _body(rng, 93, string.ascii_letters + string.digits + "-_") + "AA"


VECTORS = {
    "github-pat": _github_pat,
    "anthropic-api-key": _anthropic_key,
}


def _synthesize(vector, salt):
    """Deterministic per (vector, salt) so a failure message is reproducible."""
    return VECTORS[vector](random.Random(f"{vector}:{salt}"))


# ---------------------------------------------------------------------------
# Scratch repos and plants
# ---------------------------------------------------------------------------


def _scratch_repo(tmp_path, name="scan"):
    """A git repo carrying this repo's ignore set and rules, verbatim."""
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], cwd=repo)
    shutil.copy2(GITIGNORE_PATH, repo / ".gitignore")
    shutil.copy2(CONFIG_PATH, repo / ".gitleaks.toml")
    return repo


def _check_ignore(repo, rel_path):
    """``(ignored, source, pattern)`` for ``rel_path`` under ``repo``'s ignore set.

    ``--no-index`` makes this a question about the *patterns* rather than about
    what happens to be tracked today, which is what lets the fixed list below
    cover paths the harness has not written yet. Note ``-v`` reports a match for
    negated patterns too and still exits 0, so a leading ``!`` — not the exit
    code — is what says "re-included".
    """
    proc = _git(["check-ignore", "-v", "--no-index", "--", rel_path], cwd=repo, check=False)
    if proc.returncode == 1 or not proc.stdout.strip():
        return False, None, None
    assert proc.returncode == 0, f"git check-ignore failed: {proc.stderr}"
    location, pattern = proc.stdout.splitlines()[0].split("\t")[0].rsplit(":", 1)
    return not pattern.startswith("!"), location, pattern


def _plant(path, secret):
    """Write ``secret`` into ``path`` in a way that keeps the file's format valid.

    A plant that corrupts its host file would be a weaker test: the realistic
    leak is a well-formed config or fixture that happens to carry a live key.

    The plant carries **no identifier** — no ``api_key =``, no ``GITHUB_TOKEN=``.
    That is deliberate. An identifier hands ``generic-api-key`` a second chance
    at the value, and a plant two rules can catch cannot tell us whether the
    *vendor* rule — the one an exception targets — still fires. The bare value on
    a line is also how a key is really pasted into a README.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data.append({"note": secret})
        elif isinstance(data, dict):
            data["note"] = secret
        else:
            raise AssertionError(f"cannot plant into JSON scalar: {path}")
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return
    if suffix in (".yaml", ".yml"):
        text = path.read_text(encoding="utf-8")
        candidate = text.rstrip("\n") + f'\nnote: "{secret}"\n'
        assert isinstance(yaml.safe_load(candidate), dict), f"YAML plant broke {path}"
        path.write_text(candidate, encoding="utf-8")
        return
    if suffix == ".md":
        text = path.read_text(encoding="utf-8")
        path.write_text(text.rstrip("\n") + f"\n\n    {secret}\n", encoding="utf-8")
        return
    # Lockfiles, TOML, INI, plain text: a trailing comment is inert everywhere
    # this repo actually uses.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# {secret}\n")


def _plant_case(tmp_path, rel_path, vector):
    """Copy ``rel_path`` into a scratch repo, plant a vector, stage it, scan it."""
    repo = _scratch_repo(tmp_path)
    src = REPO_ROOT / rel_path
    assert src.is_file(), f"plant target no longer exists: {rel_path}"
    dst = repo / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    secret = _synthesize(vector, rel_path)
    _plant(dst, secret)

    ignored, location, pattern = _check_ignore(repo, rel_path)
    assert not ignored, (
        f"ignore-rule collision: {rel_path} is ignored by {location}:{pattern}, so the "
        "plant would never be walked. Fix the .gitignore pattern — never paper over "
        "this with a path allowlist."
    )
    _git(["add", "-A"], cwd=repo)
    tracked = {p for p in _git(["ls-files", "-z"], cwd=repo).stdout.split("\0") if p}
    assert rel_path in tracked, f"{rel_path} did not stage into the scratch repo"

    findings, _ = _run_gitleaks(repo, repo / ".gitleaks.toml", tmp_path / "out")
    return rel_path, secret, findings


# Representative published surfaces: a JSON fixture (the one whose anchor text is
# *about* a hardcoded key), a YAML config (the shape a real key would land in),
# two Markdown files — the README, where a pasted key really goes, and the plan
# fixture whose `refresh_token` column is the field-name decoy — and `uv.lock`,
# which the design note predicted would need an exception. It does not, and this
# row is what keeps that answer honest: gitleaks does not skip lockfiles, so a
# key pasted there is caught like any other.
PLANT_TARGETS = [
    "skills/multi-llm/tests/fixtures/e2e/responses/code_review/mixed_importance.json",
    "skills/multi-llm/providers.yaml",
    "README.md",
    "skills/multi-llm/tests/fixtures/e2e/plans/auth-feature.md",
    "skills/multi-llm/uv.lock",
]

# Committed, value-free, and therefore required to stay walkable.
MUST_NOT_BE_IGNORED = [
    # The seed layout is one directory per provider, each holding that provider's
    # own home-relative path — so these are the paths as committed, not the
    # home-relative tails. A negation that matched only the tail would leave the
    # real seeds ignored while a test written against the tail stayed green.
    "skills/multi-llm/tests/container/home_seed/codex/.codex/auth.json",
    "skills/multi-llm/tests/container/home_seed/opencode/.local/share/opencode/auth.json",
    "skills/multi-llm/tests/container/home_seed/opencode/.config/opencode/opencode.json",
    "skills/multi-llm/tests/container/creds.yaml",
    "skills/multi-llm/tests/container/redirect.yaml",
    "skills/multi-llm/tests/container/host_matrix.yaml",
    "skills/multi-llm/tests/container/expected_argv.yaml",
    "skills/multi-llm/tests/container/cli-surface.yaml",
    "skills/multi-llm/tests/container/cli-versions.lock",
    # C10's provenance record: tracked, though it sits one directory from the
    # ignored runs/ bundles. The trailing slash on that rule is what keeps them
    # apart, and this row is what proves it.
    "skills/multi-llm/tests/container/records/2026-08-14T09-31-07Z-codex.json",
]

# Real credential locations: every one must be ignored, by a named rule.
MUST_BE_IGNORED = [
    "skills/multi-llm/tests/container/runs/x",
    "skills/multi-llm/tests/container/runs/2026-08-14T09-31-07Z-codex/auth.json",
    ".env",
    ".env.local",
    "skills/multi-llm/tests/container/.env.codex",
    ".codex/auth.json",
    "skills/multi-llm/tests/container/home/.codex/auth.json",
    "skills/multi-llm/tests/container/home/.local/share/opencode/auth.json",
    "skills/multi-llm/tests/container/home/creds/token",
    "skills/multi-llm/tests/container/host.credentials.json",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_tomllib
class TestConfigShape:
    """The committed config is the contract; these are its invariants."""

    def test_config_exists_and_parses(self):
        assert CONFIG_PATH.is_file(), (
            f"{CONFIG_PATH} is missing. It is not optional: the scan below runs with "
            "an explicit --config, and the rules it carries are what make this repo's "
            "prefix and field-name decoys non-matches."
        )
        _config_data()

    def test_extends_the_stock_rule_pack(self):
        """Vendor key formats change without telling us; that pack is somebody's job."""
        data = _config_data()
        assert data.get("extend", {}).get("useDefault") is True, (
            "the config must set [extend] useDefault = true. Hand-rolled prefix "
            "regexes are the thing this design rule exists to avoid."
        )

    def test_no_file_level_suppression(self):
        """``paths`` must never buy a file a pass — in either of its two positions.

        A ``paths`` glob hides the *next* key pasted into that file, not just
        today's benign match, and fixture directories are exactly where a real
        capture gets dropped "just to reproduce something".

        Which spelling is safe is a measured fact (``TestGitleaksAllowlistSemantics``):
        a **global** ``paths`` skips the whole file before any rule runs — the
        ``regexes`` beside it is inert, and so is ``condition = "AND"`` — so it is
        rejected however it is dressed up. Only the rule-level form with
        ``regexes`` and an explicit ``AND`` really means "this value in this file".
        """
        for label, scope, entry in _entries():
            if not entry.get("paths"):
                continue
            assert scope == "rule", (
                f"{label} carries `paths` in a global allowlist. gitleaks skips the "
                "entire file before any rule runs, so this is a blind spot no matter "
                "what sits beside it. Move it under a [[rules]] block."
            )
            assert entry.get("regexes"), (
                f"{label} carries `paths` with no `regexes` beside it — a per-rule "
                "blind spot. Scope it to the value, or delete it."
            )
            assert str(entry.get("condition", "")).upper() == "AND", (
                f'{label} carries `paths` and `regexes` without `condition = "AND"`. '
                "gitleaks defaults to OR, so `paths` alone would suppress and the "
                "value scoping would be decoration."
            )

    def test_no_entry_uses_targetrules(self):
        """``targetRules`` is a silent no-op in gitleaks 8.28.0 — measured, not assumed.

        It suppresses nothing, for stock and locally-defined rules alike, and an
        unknown rule id raises no error. An exception that looks scoped and does
        nothing is worse than none, because the reviewer stops looking and the
        author moves on to a broader exclusion when the scan stays red.
        """
        for label, _, entry in _entries():
            assert not entry.get("targetRules"), (
                f"{label} uses `targetRules`, which gitleaks silently ignores. Hang "
                "the allowlist off a `[[rules]] id = \"<stock-rule-id>\"` block instead "
                "— a bare id attaches to the stock rule without restating its regex."
            )

    def test_every_entry_carries_its_provenance(self):
        """Rule ID, rationale, date and fingerprints — or the exception is unreviewable."""
        rule_ids = {rule.get("id") for rule in (_config_data().get("rules") or [])}
        for label, scope, entry in _entries():
            if scope == "rule":
                assert any(rid and rid in label for rid in rule_ids), (
                    f"{label}: the [[rules]] block it hangs on names no rule id"
                )
            description = (entry.get("description") or "").strip()
            assert len(description) > 20, f"{label}: no one-line rationale in `description`"
            assert re.search(r"\d{4}-\d{2}-\d{2}", description), (
                f"{label}: `description` carries no ISO date saying when this was added"
            )
            assert re.search(r"^\s*#\s*fingerprints:\s*\S", _entry_block(entry), re.M), (
                f"{label}: no `# fingerprints:` comment naming the matches it covers"
            )

    def test_no_gitleaksignore_is_committed(self):
        """One exception channel. A ``.gitleaksignore`` is an exception nobody reviews."""
        strays = [p for p in _tracked_files() if Path(p).name == ".gitleaksignore"]
        assert not strays, (
            f"tracked .gitleaksignore file(s): {strays}. Fingerprint suppression outside "
            ".gitleaks.toml carries no rule ID, no rationale and no date — move it."
        )


@requires_gitleaks
class TestTrackedTreeIsClean:
    """The scan itself, plus the proof that it ran with our rules."""

    def test_explicit_config_is_the_one_gitleaks_loaded(self, tracked_scan):
        """A typo'd --config path is the way this suite goes green for the wrong reason."""
        assert CONFIG_PATH.is_file()
        stderr = tracked_scan["stderr"]
        assert "from `--config`" in stderr, (
            "gitleaks did not report loading a config from --config; it may have "
            f"fallen back to its stock defaults.\n{stderr}"
        )
        assert str(CONFIG_PATH) in stderr, (
            f"gitleaks loaded a config other than {CONFIG_PATH}.\n{stderr}"
        )
        assert "extending config with default config" in stderr, (
            "the loaded config did not extend the stock rule pack.\n" + stderr
        )

    def test_the_walk_covers_the_published_set(self, tracked_scan):
        """Sanity: a mirror that silently came up empty would scan clean forever."""
        files = tracked_scan["files"]
        assert len(files) > 100, f"only {len(files)} tracked files mirrored; the walk is broken"
        assert "skills/multi-llm/uv.lock" in files
        assert ".gitignore" in files

    def test_the_rules_are_themselves_committed(self, tracked_scan):
        """An uncommitted config protects nobody, and is not the one CI loads.

        It also has to be *inside* the walk: this file names the shapes of the
        decoys it excuses, and a rules file nothing scans is as good a hiding
        place as any other.
        """
        assert ".gitleaks.toml" in tracked_scan["files"], (
            ".gitleaks.toml is not tracked. `git add .gitleaks.toml` — the config and "
            "this test land in the same commit, config first: it is a precondition of "
            "the first green run, never a follow-up fix."
        )

    def test_no_secrets_in_the_tracked_tree(self, tracked_scan):
        findings = tracked_scan["findings"]
        assert not findings, (
            "gitleaks found credentials in files this repo publishes:\n"
            + _describe(findings)
            + "\n\nDo not allowlist these without reading them first. If a finding is "
            "genuinely benign, add a match-level [[allowlists]] entry to .gitleaks.toml "
            "(targetRules + regexes + paths + fingerprints + rationale + date). If it is "
            "a real key: rotate it, then remove it from history."
        )


@requires_gitleaks
class TestDecoysStayUnmatched:
    """The false-positive families that would get this scanner switched off.

    These assert the decoys are still *present* before asserting they do not
    match, so the guard cannot go quietly vacuous if a fixture is rewritten.
    """

    def test_prefix_decoy_is_present_and_unmatched(self, tracked_scan):
        rel = "skills/multi-llm/tests/fixtures/e2e/responses/code_review/mixed_importance.json"
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        # Split so this file is not itself a carrier of the literal.
        assert "sk-" + "1234567890" in text, f"{rel} no longer carries the prefix decoy"
        assert not [f for f in tracked_scan["findings"] if f["File"] == rel]

    def test_field_name_decoy_is_present_and_unmatched(self, tracked_scan):
        rel = "skills/multi-llm/tests/fixtures/e2e/plans/auth-feature.md"
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "refresh_token" in text, f"{rel} no longer carries the field-name decoy"
        assert not [f for f in tracked_scan["findings"] if f["File"] == rel]

    def test_lockfile_digests_are_present_and_unmatched(self, tracked_scan):
        rel = "skills/multi-llm/uv.lock"
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert len(re.findall(r"sha256:[a-f0-9]{64}", text)) > 50, (
            f"{rel} no longer carries the high-entropy digest decoy"
        )
        assert not [f for f in tracked_scan["findings"] if f["File"] == rel]


@requires_gitleaks
@requires_tomllib
class TestAllowlistEntriesAreLive:
    """A dead entry is an unmaintained blind spot; a needless one is decoration."""

    def test_each_entry_still_names_a_real_path_and_a_matching_value(self):
        for label, _, entry in _entries():
            files = _entry_files(entry) if entry.get("paths") else _tracked_files()
            if entry.get("paths"):
                assert files, (
                    f"{label}: its `paths` regexes select no tracked file any more. "
                    "Delete the entry rather than carrying it."
                )
            assert entry.get("regexes"), (
                f"{label}: no `regexes` — there is no value for this exception to be "
                "scoped to (see the file-level suppression rule)"
            )
            assert _matches_somewhere(entry, files), (
                f"{label}: none of its `regexes` still match inside "
                f"{files if entry.get('paths') else 'the tracked tree'}. The value it "
                "was written for is gone; the entry should go with it."
            )

    def test_removing_any_entry_turns_the_scanner_red(self, tmp_path):
        """An entry that changes nothing is how a `paths`-shaped exclusion sneaks back."""
        entries = _entries()
        mirror = tmp_path / "tree"
        mirror.mkdir()
        _mirror_tracked(REPO_ROOT, mirror)
        for index, (label, _, entry) in enumerate(entries):
            reduced = _config_text().replace(_entry_block(entry), "")
            path = tmp_path / f"without-{index}.toml"
            path.write_text(reduced, encoding="utf-8")
            findings, _ = _run_gitleaks(mirror, path, tmp_path / f"out-{index}")
            assert findings, (
                f"{label} suppresses nothing: the tree is still clean without it. "
                "Either it is decoration, or it is one of the shapes gitleaks silently "
                "ignores — delete it."
            )

    def test_the_entry_set_is_no_larger_than_what_actually_fires(self, tmp_path):
        """"Empty" must be a measurement, not an omission.

        Stripping every entry has to leave exactly the findings the entries were
        written for — so when there are no entries, this asserts the tracked tree
        is clean under the *unmodified* stock pack, which is the only honest
        reason for an empty set.
        """
        entries = _entries()
        stripped = _config_text()
        for _, _, entry in entries:
            stripped = stripped.replace(_entry_block(entry), "")
        path = tmp_path / "no-allowlists.toml"
        path.write_text(stripped, encoding="utf-8")

        mirror = tmp_path / "tree"
        mirror.mkdir()
        _mirror_tracked(REPO_ROOT, mirror)
        findings, _ = _run_gitleaks(mirror, path, tmp_path / "out")

        if not entries:
            assert not findings, (
                "the config carries no allowlist entries, but the stock pack fires on:\n"
                + _describe(findings)
                + "\n\nEither those are real, or each needs a match-level entry."
            )
            return
        covered = {f"{f['File']}::{f['RuleID']}" for f in findings}
        assert covered, "entries exist but nothing fires without them; see the test above"


@requires_gitleaks
class TestPlantedCredentialsAreDetected:
    """The negative half: an exclusion broad enough to swallow a real key is a failure."""

    @pytest.mark.parametrize("rel_path", PLANT_TARGETS)
    @pytest.mark.parametrize("vector", sorted(VECTORS))
    def test_planted_vector_is_detected(self, tmp_path, rel_path, vector):
        rel, secret, findings = _plant_case(tmp_path, rel_path, vector)
        hits = [f for f in findings if f["File"] == rel and secret in f["Match"]]
        assert hits, (
            f"a genuine-shape {vector} planted into {rel} was NOT detected. Everything "
            "downstream of this scanner assumes such a value cannot survive a PR.\n"
            f"all findings:\n{_describe(findings)}"
        )

    @requires_tomllib
    @pytest.mark.parametrize("vector", sorted(VECTORS))
    def test_allowlisted_files_still_detect_a_real_key(self, tmp_path, vector):
        """Each exception must be scoped to its one known value, not to its file."""
        targets = [
            (index, label, rel_path)
            for index, (label, _, entry) in enumerate(_entries())
            for rel_path in _entry_files(entry)
        ]
        if not targets:
            pytest.skip(
                "no allowlist entry in .gitleaks.toml names a file, so no file carries "
                "an exception to test — see TestAllowlistEntriesAreLive, which asserts "
                "that empty set is a measurement rather than an omission"
            )
        for index, label, rel_path in targets:
            rel, secret, findings = _plant_case(
                tmp_path / f"e{index}-{Path(rel_path).name}", rel_path, vector
            )
            hits = [f for f in findings if f["File"] == rel and secret in f["Match"]]
            assert hits, (
                f"{label} is broad enough to swallow a real {vector} planted into "
                f"{rel}. Narrow its `regexes` to the exact benign value."
            )


@requires_gitleaks
class TestGitleaksAllowlistSemantics:
    """Pin the three gitleaks behaviours the config's exclusion rules are built on.

    ``.gitleaks.toml`` forbids a ``paths`` entry in a *global* allowlist and
    permits it only under a ``[[rules]]`` block alongside ``regexes`` and
    ``condition = "AND"``. That asymmetry is not stylistic — it is what the
    scanner actually does, and the obvious spelling of "this value in this file"
    is a whole-file blind spot. Measured here rather than assumed, so a gitleaks
    upgrade that changes either behaviour fails on this test instead of quietly
    widening every exception the config carries.

    The third is ``regexTarget``, which the run-record entry depends on: an
    allowlist regex is tested against the *secret* unless told otherwise, so a
    regex describing the context that makes a value benign matches nothing and
    the entry is a no-op — and ``"line"`` has to really mean one line, or an
    exception written for a benign row would cover its neighbours too.

    The probe fixture is a lockfile-shaped file holding a benign digest *and* a
    real-shape token, plus a second file holding the same token: the pair is what
    distinguishes "the file was skipped" from "the value was excused".
    """

    @pytest.fixture
    def probe(self, tmp_path):
        secret = _synthesize("github-pat", "allowlist-semantics")
        tree = tmp_path / "tree"
        (tree / "skills/multi-llm").mkdir(parents=True)
        (tree / "skills/multi-llm/uv.lock").write_text(
            f"sha256:{'a' * 64}\n# {secret}\n", encoding="utf-8"
        )
        (tree / "elsewhere.txt").write_text(f"# {secret}\n", encoding="utf-8")
        return tree, secret

    def _scan(self, probe, tmp_path, name, config_body):
        tree, secret = probe
        config = tmp_path / f"{name}.toml"
        config.write_text("[extend]\nuseDefault = true\n" + config_body, encoding="utf-8")
        findings, _ = _run_gitleaks(tree, config, tmp_path / f"out-{name}")
        return {(f["RuleID"], f["File"]) for f in findings if secret in f["Match"]}

    def test_a_global_paths_entry_skips_the_whole_file(self, probe, tmp_path):
        """Why global ``paths`` is forbidden: ``regexes`` and ``AND`` beside it are inert."""
        hits = self._scan(
            probe,
            tmp_path,
            "global-paths",
            "\n[[allowlists]]\n"
            'description = "narrow value, but globally scoped by path"\n'
            'condition   = "AND"\n'
            "regexes     = ['''sha(?:256|512):[a-f0-9]{64,128}''']\n"
            "paths       = ['''skills/multi-llm/uv\\.lock''']\n",
        )
        assert ("github-pat", "elsewhere.txt") in hits, "the probe itself is broken"
        assert ("github-pat", "skills/multi-llm/uv.lock") not in hits, (
            "gitleaks no longer skips a whole file on a global `paths` allowlist. That "
            "was the entire reason .gitleaks.toml rejects the shape — re-read the "
            "'How gitleaks really evaluates an allowlist' section before relaxing it."
        )

    def test_a_rule_level_and_entry_excuses_only_that_value_in_that_file(self, probe, tmp_path):
        """Why the rule-level ``AND`` form is the permitted one: it is genuinely narrow."""
        _, secret = probe
        hits = self._scan(
            probe,
            tmp_path,
            "rule-and",
            '\n[[rules]]\nid = "github-pat"\n\n[[rules.allowlists]]\n'
            'description = "the one known value, in the one known file"\n'
            'condition   = "AND"\n'
            f"regexes     = ['''{re.escape(secret)}''']\n"
            "paths       = ['''skills/multi-llm/uv\\.lock''']\n",
        )
        assert ("github-pat", "skills/multi-llm/uv.lock") not in hits, (
            "a rule-level paths+regexes+AND entry did not excuse the value it names; "
            "the exception shape .gitleaks.toml documents does not work on this gitleaks"
        )
        assert ("github-pat", "elsewhere.txt") in hits, (
            "a rule-level paths+regexes+AND entry leaked past its `paths` scope — the "
            "same value elsewhere must still be a finding, or the exception is global"
        )

    # The shape the committed run-record entry uses, restated here so this test
    # measures the *mechanism* rather than importing the entry it protects: a
    # config edit that widened the real regex would still have to pass this.
    _MANIFEST_LINE = r'''(?m)^[ \t]*"[A-Za-z0-9_./-]+\.[a-z0-9]{1,8}":[ \t]*"[a-f0-9]{64}",?[ \t]*\r?$'''
    _RECORDS_PATH = r"""^skills/multi-llm/tests/container/records/[^/]+\.json$"""

    @pytest.fixture
    def manifest_probe(self, tmp_path):
        """A record-shaped file: a benign digest row, then a real key on the next line.

        The two lines are the whole point. A ``regexTarget`` that matched a
        *region* rather than a line would excuse the key along with the row it
        neighbours, and the exception would be the blind spot it was written to
        avoid.
        """
        secret = _synthesize("anthropic-api-key", "regex-target")
        digest = hashlib.sha256(b"multi-llm manifest probe").hexdigest()
        rel = "skills/multi-llm/tests/container/records/probe.json"
        tree = tmp_path / "tree"
        (tree / rel).parent.mkdir(parents=True)
        (tree / rel).write_text(
            "{\n"
            '  "bundle": {\n'
            '    "files": {\n'
            f'      "credential-census.json": "{digest}",\n'
            f'      "token": "{secret}"\n'
            "    }\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        return tree, rel, digest, secret

    def _manifest_scan(self, manifest_probe, tmp_path, name, entry_lines):
        tree, _, _, _ = manifest_probe
        config = tmp_path / f"{name}.toml"
        config.write_text(
            "[extend]\nuseDefault = true\n"
            '\n[[rules]]\nid = "generic-api-key"\n\n[[rules.allowlists]]\n'
            'description = "manifest digest rows, measured against a planted key"\n'
            'condition   = "AND"\n' + entry_lines,
            encoding="utf-8",
        )
        findings, _ = _run_gitleaks(tree, config, tmp_path / f"out-{name}")
        return {f["Secret"] for f in findings}

    def test_the_probe_fires_on_both_lines_unallowlisted(self, manifest_probe, tmp_path):
        """Sanity: a probe whose digest is below the entropy floor proves nothing."""
        _, _, digest, secret = manifest_probe
        secrets = self._manifest_scan(
            manifest_probe,
            tmp_path,
            "manifest-baseline",
            f"regexes     = ['''{re.escape('no-such-value')}''']\n"
            f"paths       = ['''{self._RECORDS_PATH}''']\n",
        )
        assert digest in secrets, (
            "the benign manifest digest does not fire unallowlisted, so nothing below "
            "can show it being excused — the probe is broken, not the config"
        )
        assert secret in secrets, "the planted key does not fire; the probe is broken"

    def test_a_regex_without_regex_target_is_tested_against_the_secret(
        self, manifest_probe, tmp_path
    ):
        """Why the run-record entry carries ``regexTarget``: without it, it is a no-op.

        This is the third silent-failure shape, alongside global ``paths`` and
        ``targetRules``. A regex describing the *line* a benign value sits on is
        never compared against that line by default — it is compared against the
        capture group — so the entry looks precise, suppresses nothing, and the
        author reaches for something broader when the scan stays red.
        """
        _, _, digest, _ = manifest_probe
        secrets = self._manifest_scan(
            manifest_probe,
            tmp_path,
            "manifest-no-target",
            f"regexes     = ['''{self._MANIFEST_LINE}''']\n"
            f"paths       = ['''{self._RECORDS_PATH}''']\n",
        )
        assert digest in secrets, (
            "gitleaks now applies an allowlist regex to the line by default. The "
            "run-record entry's `regexTarget = \"line\"` was written because it did "
            "not — re-read 'How gitleaks really evaluates an allowlist' before "
            "trusting either spelling."
        )

    def test_regex_target_line_excuses_one_line_and_not_its_neighbour(
        self, manifest_probe, tmp_path
    ):
        """The property the run-record entry rests on: ``line`` is one line."""
        _, _, digest, secret = manifest_probe
        secrets = self._manifest_scan(
            manifest_probe,
            tmp_path,
            "manifest-line",
            'regexTarget = "line"\n'
            f"regexes     = ['''{self._MANIFEST_LINE}''']\n"
            f"paths       = ['''{self._RECORDS_PATH}''']\n",
        )
        assert digest not in secrets, (
            "a rule-level paths+regexes+AND entry with `regexTarget = \"line\"` did not "
            "excuse the manifest row it describes; the shape .gitleaks.toml's run-record "
            "entry uses does not work on this gitleaks"
        )
        assert secret in secrets, (
            "the manifest-row exception swallowed a real key on the *adjacent* line, so "
            "`regexTarget = \"line\"` is matching a region rather than a line. That turns "
            "the run-record entry into the blind spot it was written to avoid."
        )


class TestGitignoreDirections:
    """``git check-ignore -v``, both ways, over a fixed list.

    Most of these paths do not exist yet — that is the point. ``check-ignore``
    answers for hypothetical paths, so the rules that will protect the
    containerized harness are proven before the harness that writes them lands.
    """

    @pytest.fixture(scope="class")
    def ignore_repo(self, tmp_path_factory):
        return _scratch_repo(tmp_path_factory.mktemp("ignore"), name="ignoreset")

    @pytest.mark.parametrize("rel_path", MUST_NOT_BE_IGNORED)
    def test_committed_value_free_paths_stay_walkable(self, ignore_repo, rel_path):
        ignored, location, pattern = _check_ignore(ignore_repo, rel_path)
        assert not ignored, (
            f"{rel_path} carries no value and must stay tracked, but it is ignored by "
            f"{location}:{pattern}. A gitignored template is also a template the scanner "
            "never walks and the image build never receives."
        )

    @pytest.mark.parametrize("rel_path", MUST_BE_IGNORED)
    def test_credential_locations_are_ignored_by_a_named_rule(self, ignore_repo, rel_path):
        ignored, location, pattern = _check_ignore(ignore_repo, rel_path)
        assert ignored, (
            f"{rel_path} is a real credential location and is NOT ignored. Nothing but "
            "this pattern set stands between a captured token and a public repository."
        )
        assert pattern and location, f"{rel_path}: ignored, but by no nameable rule"

    def test_the_seed_templates_are_re_included_by_a_negation(self, ignore_repo):
        """The one subtle rule: credential *paths* are ignored, seed templates negated back.

        ``auth.json`` is literally opencode's credential filename, so the ignore
        rules are anchored by location and the seed set is re-included after
        them. Order matters — a negation before its rule does nothing — so this
        asserts the matching pattern really is the negation.
        """
        rel = "skills/multi-llm/tests/container/home_seed/codex/.codex/auth.json"
        ignored, location, pattern = _check_ignore(ignore_repo, rel)
        assert not ignored
        assert pattern and pattern.startswith("!"), (
            f"{rel} is un-ignored, but not by the home_seed negation "
            f"({location}:{pattern}). If the credential rule stops matching it, the "
            "negation is no longer load-bearing and the pair should be re-read."
        )
