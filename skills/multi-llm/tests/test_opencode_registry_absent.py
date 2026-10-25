"""Row 9's one entry point, ``packaging/opencode/registry_absent.mjs``.

**What is graded.** The three-way rule (exit 0 published, exactly ``E404``
absent, everything else unknown) against a stub ``npm`` first on ``PATH`` that
replays what npm really prints; the single-registry refusal; the alternate
configured registry that must not turn the row green; and, statically, that
the Makefile and the publishing workflow reach row 9 only through this script.

**Where the fixtures come from.** Measured 2026-10-25 on npm 10.9.3 (and the
publish job's pin, 11.5.1, which prints the same ``--json`` body on stdout):
``npm view opencode-multi-llm@0.1.0 version --json`` against npmjs for ``E404``,
``npm view opencode-ai@1.19.4`` for a published version, a localhost stub
answering 401 for ``E401``, port 9 for ``ECONNREFUSED`` and ``.invalid`` for
``ENOTFOUND``. Only stdout is replayed: the script never reads stderr.

**Offline.** Nothing here reaches a real registry. The one row that runs the
real ``npm`` points it at two localhost stubs and skips, by name, without npm.
POSIX only for the runtime rows: the stub is a shebang script, and the script
spawns ``npm`` without a shell, which on Windows cannot start npm's ``.cmd``
shim. The release path runs on Linux.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
SCRIPT = PACKAGE_DIR / "registry_absent.mjs"
MAKEFILE = REPO_ROOT / "Makefile"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-opencode-plugin.yml"

sys.path.insert(0, str(PACKAGE_DIR))
import registry as registry_module  # noqa: E402

NODE = shutil.which("node")
SINGLE = registry_module.single_registry()
MANIFEST = json.loads((PACKAGE_DIR / "package.json").read_text(encoding="utf-8"))
SPEC = f"{MANIFEST['name']}@{MANIFEST['version']}"

needs_node = pytest.mark.skipif(NODE is None, reason="needs node: the entry point is a .mjs")
posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX stub npm; the script spawns npm without a shell, which cannot "
           "start npm's .cmd shim on Windows (the release path is Linux)",
)


def _error(code: str, summary: str, detail: str = "") -> str:
    return json.dumps({"error": {"code": code, "summary": summary, "detail": detail}}, indent=2) + "\n"


#: The four measured outcomes, as ``(exit, stdout)``.
PUBLISHED = (0, f'"{MANIFEST["version"]}"\n')
E404 = (1, _error(
    "E404", f"Not Found - GET {SINGLE}/{MANIFEST['name']} - Not found",
    f"'{SPEC}' is not in this registry.\n\nNote that you can also install from a\n"
    "tarball, folder, http url, or git url.",
))
AUTH = (1, _error(
    "E401", "Unable to authenticate, your authentication token seems to be invalid.",
    "To correct this please try logging in again with:\n  npm login",
))
NEEDAUTH = (1, _error("ENEEDAUTH", "This command requires you to be logged in to "
                      f"{SINGLE}/", "You need to authorize this machine using `npm adduser`"))
REFUSED = (1, _error(
    "ECONNREFUSED",
    f"FetchError: request to {SINGLE}/{MANIFEST['name']} failed, reason: connect "
    "ECONNREFUSED 104.16.0.35:443",
    "If you are behind a proxy, please make sure that the\n'proxy' config is set properly.",
))
NOTFOUND = (1, _error(
    "ENOTFOUND",
    f"request to {SINGLE}/{MANIFEST['name']} failed, reason: getaddrinfo ENOTFOUND "
    "registry.npmjs.org",
    "This is a problem related to network connectivity.",
))


# ---------------------------------------------------------------------------
# The stub npm and the runner
# ---------------------------------------------------------------------------

STUB_SOURCE = """#!{python}
import json, os, sys
log = os.environ["STUB_NPM_LOG"]
with open(log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "argv": sys.argv[1:],
        "npm_config_registry": os.environ.get("npm_config_registry"),
        "npm_config_userconfig": os.environ.get("npm_config_userconfig"),
    }}) + "\\n")
with open(os.environ["STUB_NPM_FIXTURE"], encoding="utf-8") as handle:
    fixture = json.load(handle)
sys.stdout.write(fixture["stdout"])
sys.stderr.write(fixture.get("stderr", ""))
sys.exit(fixture["exit"])
"""


@pytest.fixture
def stub_npm(tmp_path):
    """A ``bin/npm`` replaying one fixture; returns ``run(fixture, *args, env=)``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    npm.write_text(STUB_SOURCE.format(python=sys.executable), encoding="utf-8")
    npm.chmod(0o755)
    log = tmp_path / "npm-calls.jsonl"
    fixture_path = tmp_path / "fixture.json"

    class Stub:
        calls_log = log

        @staticmethod
        def run(fixture, *args, env=None, registry=SINGLE):
            status, stdout = fixture
            fixture_path.write_text(json.dumps({"exit": status, "stdout": stdout,
                                                "stderr": "npm error noise\n"}),
                                    encoding="utf-8")
            environ = dict(os.environ)
            for key in list(environ):
                if key.lower().startswith("npm_config_"):
                    del environ[key]
            environ.update({
                "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
                "STUB_NPM_LOG": str(log),
                "STUB_NPM_FIXTURE": str(fixture_path),
            })
            environ.update(env or {})
            argv = [NODE, str(SCRIPT)]
            if registry is not None:
                argv += ["--registry", registry]
            return subprocess.run(argv + list(args), cwd=tmp_path, env=environ,
                                  capture_output=True, text=True, timeout=120)

        @staticmethod
        def calls():
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    return Stub


def _evidence(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The three outcomes
# ---------------------------------------------------------------------------

@needs_node
@posix_only
class TestThreeOutcomes:
    """Only ``E404`` passes. Each row also checks the evidence the script writes,
    because the gate records that file, not the exit status."""

    def test_e404_is_the_only_green_answer(self, stub_npm, tmp_path):
        out = tmp_path / "row9.json"
        done = stub_npm.run(E404, "--out", str(out))
        assert done.returncode == 0, done.stderr
        assert f"registry {SINGLE} answered code E404" in done.stdout
        assert _evidence(out) == {
            "row": "9", "pass": True, "registry": SINGLE,
            "name": MANIFEST["name"], "version": MANIFEST["version"],
            "npm_exit": 1, "code": "E404", "outcome": "absent",
        }

    def test_an_already_published_version_refuses(self, stub_npm, tmp_path):
        out = tmp_path / "row9.json"
        done = stub_npm.run(PUBLISHED, "--out", str(out))
        assert done.returncode == 1
        assert "already on" in done.stderr and "bump" in done.stderr
        evidence = _evidence(out)
        assert evidence["outcome"] == "published" and evidence["pass"] is False
        assert evidence["npm_exit"] == 0 and evidence["code"] is None

    @pytest.mark.parametrize("fixture, code", [
        (AUTH, "E401"), (NEEDAUTH, "ENEEDAUTH"),
        (REFUSED, "ECONNREFUSED"), (NOTFOUND, "ENOTFOUND"),
        ((1, _error("E429", "Too Many Requests - GET x")), "E429"),
        ((1, _error("E500", "Internal Server Error - GET x")), "E500"),
    ], ids=["E401", "ENEEDAUTH", "ECONNREFUSED", "ENOTFOUND", "E429", "E500"])
    def test_every_other_code_is_an_unknown(self, stub_npm, tmp_path, fixture, code):
        out = tmp_path / "row9.json"
        done = stub_npm.run(fixture, "--out", str(out))
        assert done.returncode == 1
        assert "UNKNOWN, not an absence" in done.stderr
        assert f"answered code {code}" in done.stdout
        evidence = _evidence(out)
        assert (evidence["outcome"], evidence["code"], evidence["pass"]) == ("unknown", code, False)

    @pytest.mark.parametrize("stdout", [
        "", "<html>502 Bad Gateway</html>", '{"error": {}}', '{"error": {"code": 404}}',
        "E404", '"E404"', '{"error": {"code": "e404"}}', '{"error": {"code": "E404 "}}',
    ])
    def test_output_that_is_not_npms_json_is_an_unknown(self, stub_npm, tmp_path, stdout):
        """Not a substring match: a body that merely quotes the code is no answer."""
        out = tmp_path / "row9.json"
        done = stub_npm.run((1, stdout), "--out", str(out))
        assert done.returncode == 1
        assert _evidence(out)["outcome"] == "unknown"

    def test_an_e404_from_another_server_is_an_unknown(self, stub_npm, tmp_path):
        """npm's E404 names the URL it fetched; it must be the single registry."""
        status, _ = E404
        stray = _error("E404", f"Not Found - GET http://127.0.0.1:4873/{MANIFEST['name']} - Not found")
        out = tmp_path / "row9.json"
        done = stub_npm.run((status, stray), "--out", str(out))
        assert done.returncode == 1
        assert "not from the single registry" in done.stderr
        assert _evidence(out)["outcome"] == "unknown"

    def test_a_killed_or_missing_npm_is_an_unknown(self, tmp_path):
        """No npm on PATH at all: an unknown, never an absence."""
        out = tmp_path / "row9.json"
        environ = {k: v for k, v in os.environ.items() if k != "PATH"}
        environ["PATH"] = str(tmp_path / "empty")
        done = subprocess.run([NODE, str(SCRIPT), "--registry", SINGLE, "--out", str(out)],
                              env=environ, capture_output=True, text=True, timeout=60)
        assert done.returncode == 1
        assert _evidence(out)["outcome"] == "unknown"


# ---------------------------------------------------------------------------
# The single registry
# ---------------------------------------------------------------------------

@needs_node
@posix_only
class TestSingleRegistry:

    def test_npm_is_asked_the_single_registry_explicitly(self, stub_npm):
        done = stub_npm.run(E404)
        assert done.returncode == 0, done.stderr
        (call,) = stub_npm.calls()
        argv = call["argv"]
        assert argv[:4] == ["view", SPEC, "version", "--json"]
        assert f"--registry={SINGLE}" in argv
        # Bounded: retries, back-off and a fetch timeout, so an offline registry
        # ends in an unknown instead of a hang.
        assert {"--fetch-retries=2", "--fetch-timeout=30000"} <= set(argv)
        assert "--no-update-notifier" in argv
        caches = [a for a in argv if a.startswith("--cache=")]
        assert len(caches) == 1
        assert not Path(caches[0].split("=", 1)[1]).exists(), "the fresh cache was not removed"

    def test_a_trailing_slash_is_the_same_registry(self, stub_npm):
        assert stub_npm.run(E404, registry=SINGLE + "/").returncode == 0

    @pytest.mark.parametrize("other", [
        "https://registry.yarnpkg.com", "http://registry.npmjs.org",
        "http://127.0.0.1:4873", "https://registry.npmjs.org.evil.example",
        "https://registry.npmjs.org/sub",
    ])
    def test_any_other_registry_is_refused_before_npm_runs(self, stub_npm, tmp_path, other):
        out = tmp_path / "row9.json"
        done = stub_npm.run(E404, "--out", str(out), registry=other)
        assert done.returncode == 1
        assert other in done.stderr and SINGLE in done.stderr
        assert stub_npm.calls() == [], "npm ran although the registry was refused"
        assert _evidence(out)["pass"] is False

    def test_registry_is_required(self, stub_npm):
        done = stub_npm.run(E404, registry=None)
        assert done.returncode == 2 and "--registry is required" in done.stderr
        assert stub_npm.calls() == []

    def test_a_stale_green_evidence_file_never_outlives_a_refusal(self, stub_npm, tmp_path):
        out = tmp_path / "row9.json"
        out.write_text(json.dumps({"row": "9", "pass": True, "outcome": "absent"}), encoding="utf-8")
        stub_npm.run(AUTH, "--out", str(out))
        assert _evidence(out)["pass"] is False

    def test_the_package_json_is_read_not_assumed(self, stub_npm, tmp_path):
        manifest = tmp_path / "package.json"
        manifest.write_text(json.dumps({"name": MANIFEST["name"], "version": "9.8.7"}), encoding="utf-8")
        assert stub_npm.run(E404, "--package-json", str(manifest)).returncode == 0
        assert stub_npm.calls()[0]["argv"][1] == f"{MANIFEST['name']}@9.8.7"

    @pytest.mark.parametrize("name, version", [
        ("@beastlabai/opencode-multi-llm", "0.1.0"),   # @scope:registry outranks --registry
        ("--registry=http://127.0.0.1:1", "0.1.0"),
        ("opencode-multi-llm", "--registry=http://127.0.0.1:1"),
        ("opencode-multi-llm", "latest"),
        ("opencode-multi-llm", None),
    ])
    def test_a_manifest_that_could_steer_npm_is_refused(self, stub_npm, tmp_path, name, version):
        manifest = tmp_path / "package.json"
        manifest.write_text(json.dumps({"name": name, "version": version}), encoding="utf-8")
        done = stub_npm.run(E404, "--package-json", str(manifest))
        assert done.returncode == 1
        assert stub_npm.calls() == []

    @pytest.mark.parametrize("environment", [
        "userconfig", "env", "both",
    ])
    def test_an_alternate_configured_registry_is_overridden_in_the_argv(
        self, stub_npm, tmp_path, environment,
    ):
        """The script-level half of the regression: whatever the user's config
        says, the argv names the single registry. The real-npm row below proves
        npm honours that flag over the same config."""
        npmrc = tmp_path / "user.npmrc"
        npmrc.write_text("registry=http://127.0.0.1:4873/\n", encoding="utf-8")
        env = {}
        if environment in ("userconfig", "both"):
            env["npm_config_userconfig"] = str(npmrc)
        if environment in ("env", "both"):
            env["npm_config_registry"] = "http://127.0.0.1:4873/"
        stub_npm.run(E404, env=env)
        (call,) = stub_npm.calls()
        assert f"--registry={SINGLE}" in call["argv"]


def test_both_parsers_read_the_makefile_the_same_way(tmp_path):
    """registry.py and the .mjs restate one rule; they must agree on every case."""
    if NODE is None:
        pytest.skip("needs node: the entry point is a .mjs")
    cases = {
        "real": MAKEFILE.read_text(encoding="utf-8"),
        "slash": "OPENCODE_REGISTRY := https://registry.npmjs.org/\n",
        "port": "X := 1\nOPENCODE_REGISTRY   :=   https://r.example:8443\n",
        "none": "OTHER := https://registry.npmjs.org\n",
        "two": "OPENCODE_REGISTRY := https://a.example\nOPENCODE_REGISTRY := https://b.example\n",
        "http": "OPENCODE_REGISTRY := http://registry.npmjs.org\n",
        "path": "OPENCODE_REGISTRY := https://registry.npmjs.org/npm\n",
    }
    # Not compared: `OPENCODE_REGISTRY :=` with the URL on the NEXT line.
    # registry.py's `\s*` crosses the newline and accepts it; the .mjs refuses,
    # which is make's own reading: the variable is empty and the next line
    # parses as a rule (measured, GNU make). A known divergence in registry.py,
    # reported rather than copied.
    for label, text in cases.items():
        makefile = tmp_path / f"{label}.mk"
        makefile.write_text(text, encoding="utf-8")
        try:
            python = registry_module.single_registry(makefile)
        except registry_module.RegistryError:
            python = None
        probe = (
            f"import({json.dumps(SCRIPT.as_uri())}).then((m) => {{"
            f" try {{ process.stdout.write(m.singleRegistry({json.dumps(str(makefile))})); }}"
            " catch { process.stdout.write('<refused>'); } })"
        )
        done = subprocess.run([NODE, "--input-type=module", "-e", probe],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        node = None if done.stdout == "<refused>" else done.stdout
        assert node == python, f"{label}: registry.py says {python!r}, the .mjs says {node!r}"


# ---------------------------------------------------------------------------
# The alternate-registry regression, against the real npm
# ---------------------------------------------------------------------------

class _Registry(http.server.ThreadingHTTPServer):
    """A localhost registry answering every packument read one way."""

    def __init__(self, published: bool):
        self.published = published
        self.hits: list = []
        super().__init__(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server's spelling)
        self.server.hits.append(self.path)
        name = self.path.strip("/").split("/")[0]
        if self.server.published:
            version = MANIFEST["version"]
            body = json.dumps({
                "name": name, "dist-tags": {"latest": version},
                "versions": {version: {"name": name, "version": version, "dist": {
                    "tarball": f"{self.server.url}/{name}/-/{name}-{version}.tgz",
                    "integrity": "sha512-" + "A" * 86 + "==",
                }}},
            }).encode()
            self.send_response(200)
        else:
            body = b'{"error":"Not found"}'
            self.send_response(404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


PASSTHROUGH_SOURCE = """#!{python}
import os, sys
# Rewrites ONLY the value of the explicit --registry flag the script passed,
# then runs the real npm with everything else untouched.
argv = [a if not a.startswith("--registry=") else "--registry=" + os.environ["PASSTHROUGH_REGISTRY"]
        for a in sys.argv[1:]]
os.execv({npm!r}, [{npm!r}] + argv)
"""


@needs_node
@posix_only
@pytest.mark.parametrize("environment", ["userconfig", "env"])
def test_a_configured_e404_registry_cannot_turn_row_9_green(tmp_path, environment):
    """The regression the plan names: a user ``.npmrc`` or ``npm_config_registry``
    pointing at a localhost stub that answers ``E404`` must not make row 9 pass.

    The real npm, two stubs. ``npm`` on PATH is a pass-through that swaps only
    the explicit flag's value for stub A (which says *published*) and leaves the
    rest of the script's argv alone, while the configuration points at stub B
    (which says ``E404``). If the configuration won, the row would read B's
    absence; it must read A's answer, and B must see no request at all.
    """
    real_npm = shutil.which("npm")
    if real_npm is None:
        pytest.skip("needs the real npm to prove its precedence")
    published, absent = _Registry(published=True), _Registry(published=False)
    try:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        shim = bin_dir / "npm"
        shim.write_text(PASSTHROUGH_SOURCE.format(python=sys.executable, npm=real_npm),
                        encoding="utf-8")
        shim.chmod(0o755)
        npmrc = tmp_path / "user.npmrc"
        npmrc.write_text(f"registry={absent.url}/\n", encoding="utf-8")
        environ = {k: v for k, v in os.environ.items() if not k.lower().startswith("npm_config_")}
        environ.update({
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "PASSTHROUGH_REGISTRY": published.url,
            "HOME": str(tmp_path / "home"),
        })
        if environment == "userconfig":
            environ["npm_config_userconfig"] = str(npmrc)
        else:
            environ["npm_config_registry"] = absent.url + "/"
            environ["npm_config_userconfig"] = str(tmp_path / "no.npmrc")
        out = tmp_path / "row9.json"
        done = subprocess.run([NODE, str(SCRIPT), "--registry", SINGLE, "--out", str(out)],
                              env=environ, capture_output=True, text=True, timeout=180)
        assert done.returncode == 1, done.stdout + done.stderr
        assert _evidence(out)["outcome"] == "published", done.stdout + done.stderr
        assert any(hit.startswith(f"/{MANIFEST['name']}") for hit in published.hits)
        assert absent.hits == [], f"the configured registry was consulted: {absent.hits}"

        # Control: without the explicit flag, the same configuration does reach
        # stub B and reads its E404 — so the row above proved the flag, not a
        # configuration npm happened to ignore.
        control = subprocess.run(
            [real_npm, "view", SPEC, "version", "--json", "--no-update-notifier",
             f"--cache={tmp_path / 'control-cache'}"],
            env=environ, capture_output=True, text=True, timeout=120,
        )
        assert json.loads(control.stdout)["error"]["code"] == "E404"
        assert absent.hits, "the control never reached the configured registry"
    finally:
        published.shutdown()
        absent.shutdown()


# ---------------------------------------------------------------------------
# Static: one implementation, every site calls it
# ---------------------------------------------------------------------------

#: `npm view|info|show|v <spec> ... version`: asking npm whether a version exists.
#: `dist.integrity` reads (the post-publish confirmation) do not match.
ABSENCE_QUERY = re.compile(r"\bnpm\s+(?:view|info|show|v)\b[^\n;|&]*?\sversion\b(?![.\w-])")
#: Deciding on npm's error code by hand is the other half of a private copy.
CODE_PARSE = re.compile(r"\.error\.code|\berror\.code\b|\[[\"']error[\"']\]\[[\"']code[\"']\]")
ENTRY_POINT = re.compile(
    r"registry_absent\.mjs|(?:\bmake\b|\$\(MAKE\))[^\n]*\bopencode-registry-absent\b")


def _strip_shell_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _makefile_recipes() -> dict:
    """``{target: [recipe lines]}``. No variable expansion is needed here: the
    absence check would be spelled with a literal ``npm``."""
    recipes: dict = {}
    current = None
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("\t"):
            if current:
                recipes[current].append(line.strip())
            continue
        match = re.match(r"^([A-Za-z0-9_.\-/]+)\s*:(?!=)", line)
        current = match.group(1) if match else None
        if current:
            recipes.setdefault(current, [])
    return recipes


def test_the_static_patterns_catch_the_copies_they_replace():
    """The scan is only worth something if it recognizes the old copy."""
    old_recipe = 'probe="$$(npm view "$$name@$$version" version --json 2>/dev/null)"'
    old_parse = 'try { code = JSON.parse(raw).error.code || ""; } catch { code = ""; }'
    assert ABSENCE_QUERY.search(old_recipe)
    assert ABSENCE_QUERY.search('npm info "${spec}" version')
    assert CODE_PARSE.search(old_parse)
    assert not ABSENCE_QUERY.search('npm view "${spec}" dist.integrity --registry=x')
    assert not ABSENCE_QUERY.search('npm view "${spec}" versions --json')
    assert ENTRY_POINT.search("node packaging/opencode/registry_absent.mjs --registry x")
    assert ENTRY_POINT.search("make -s opencode-registry-absent OUT=x")
    assert ENTRY_POINT.search("$(MAKE) --no-print-directory opencode-registry-absent")


def test_the_makefile_reaches_row_9_only_through_the_entry_point():
    recipes = _makefile_recipes()
    entry = " ".join(recipes["opencode-registry-absent"])
    assert "registry_absent.mjs" in entry
    assert "--registry $(OPENCODE_REGISTRY)" in entry
    assert any(ENTRY_POINT.search(line) for line in recipes["opencode-package-check"]), (
        "the full gate no longer runs row 9 through its entry point"
    )
    for target, body in recipes.items():
        for line in body:
            assert not ABSENCE_QUERY.search(line), f"{target} keeps its own absence probe: {line}"
            assert not CODE_PARSE.search(line), f"{target} parses npm's error code itself: {line}"


def _workflow_runs(path: Path):
    """``(job id, step name, run text)`` for every ``run`` step."""
    workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for job_id, job in (workflow.get("jobs") or {}).items():
        for step in (job or {}).get("steps") or []:
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                yield job_id, str(step.get("name") or ""), step["run"]


def _row9_violations(path: Path) -> list:
    """Every way a workflow file departs from "row 9 only through its entry
    point": a step deciding absence itself, a step naming row 9 that does not
    call the script, or no job calling it at all."""
    problems, callers = [], set()
    for job_id, name, text in _workflow_runs(path):
        code = _strip_shell_comments(text)
        label = f"{job_id} / {name or '<unnamed>'}"
        if ENTRY_POINT.search(code):
            callers.add(job_id)
        if ABSENCE_QUERY.search(code) or CODE_PARSE.search(code):
            problems.append(f"{label}: carries its own row-9 absence logic")
        if re.search(r"\brow\s*9\b", name, re.I) and not ENTRY_POINT.search(code):
            problems.append(f"{label}: names row 9 but does not call registry_absent.mjs")
    if not callers:
        problems.append("no job runs row 9's entry point")
    return problems


@pytest.mark.xfail(
    strict=True,
    reason="the workflow's publish job still carries its own row-9 copy until the "
           "gate/verify/publish split lands; strict, so that commit must drop this mark")
def test_the_publish_workflow_reaches_row_9_only_through_the_entry_point():
    """A generic scan, because another package rewrites this workflow: whatever
    its jobs become, no step may decide absence itself, every step that names
    row 9 must call the entry point, and at least one job must run it.

    Red until that rewrite lands: today's `publish` job still carries the
    hand-written copy this script replaces."""
    assert _row9_violations(PUBLISH_WORKFLOW) == []


GOOD_WORKFLOW = """
jobs:
  gate:
    steps:
      - name: Version must be absent from the registry (row 9)
        run: make -s opencode-registry-absent
  verify:
    steps:
      - name: Row 9
        run: |
          # npm view "$spec" version would be a copy; this comment is not one.
          node packaging/opencode/registry_absent.mjs --registry "$(make -s opencode-print-registry)"
  publish:
    steps:
      - name: Absent immediately before upload (row 9)
        run: make opencode-registry-absent OUT="$RUNNER_TEMP/row9.json"
      - name: Confirm the integrity
        run: npm view "$spec" dist.integrity --registry="$registry" --cache "$(mktemp -d)"
"""


@pytest.mark.parametrize("mutation, expected", [
    (None, []),
    (("make -s opencode-registry-absent",
      'npm view "${name}@${version}" version --json || true'),
     ["gate / Version must be absent from the registry (row 9): carries its own row-9 absence logic",
      "gate / Version must be absent from the registry (row 9): names row 9 but does not "
      "call registry_absent.mjs"]),
    (("npm view \"$spec\" dist.integrity",
      "code=$(node -e 'process.stdout.write(JSON.parse(s).error.code)'); npm view \"$spec\" dist.integrity"),
     ["publish / Confirm the integrity: carries its own row-9 absence logic"]),
])
def test_the_workflow_scan_grades_a_rewritten_workflow(tmp_path, mutation, expected):
    """The scan, run on the shape the rewritten workflow is planned to have, and
    on that shape with a private copy slipped back in."""
    text = GOOD_WORKFLOW
    if mutation:
        assert mutation[0] in text
        text = text.replace(mutation[0], mutation[1])
    path = tmp_path / "workflow.yml"
    path.write_text(text, encoding="utf-8")
    assert _row9_violations(path) == expected


def test_no_other_workflow_keeps_an_absence_probe():
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if path == PUBLISH_WORKFLOW:
            continue
        for job_id, name, text in _workflow_runs(path):
            code = _strip_shell_comments(text)
            assert not ABSENCE_QUERY.search(code), f"{path.name}: {job_id} / {name}"
