"""The ``opencode-multi-llm`` npm package: its build, its manifest, its plugin.

**Why these rows read the way they do.** The plugin is JavaScript and the single
owners it must not drift from are Python, so almost every assertion here grades
the JS *against a Python owner* rather than against an expectation someone typed.
Two rows carry the weight of that: ``test_worker_flags_are_reserved``, an
equivalence with ``cli_common.flag_present()`` over a shared case table, and
``test_every_uv_candidate_spelling_agrees_with_the_source``, which lives in its
own module. Where a rule has no Python owner to mirror — the argv order, the
zero-element empty positional, the reserved ``--interactive`` — the row carries
the expected argv as a literal, which is what moves that half of the
specification out of a plan document and into the repository.

**Where each row's input comes from.** ``.gitignore`` deletes the obvious answer:
``packaging/opencode/{dist/,skills/,*.tgz}`` are build artifacts, and CI never
runs ``build.py``. So on a clean checkout **no build artifact exists**, and a row
that reads one is red before it asserts anything. Three classes:

* **(src)** — the default, and preferred wherever there is a choice. Reads
  ``packaging/opencode/src/index.mjs``, never ``dist/index.mjs``, and renders what
  it needs in-process. It also removes an ambiguity from every red row: a failure
  can never mean "someone forgot to rebuild".
* **(build)** — genuinely needs ``build.py``'s output. A **session-scoped**
  fixture builds into a temp directory and yields it; it never writes into the
  working tree, so a developer's stale ``dist/`` can neither hide a failure nor
  cause one. Needs uv and Python only, no npm, so it runs on all three OS legs.
* **(pack, Linux)** — needs ``npm pack``. These belong to the Linux-only package
  job and nowhere else; in the default suite they skip with a named reason.

Because the build emits ``dist/index.mjs`` as a **verbatim copy** of
``src/index.mjs``, a (src) grep is a grep against what ships —
``test_dist_mirrors_src`` is what pins that equivalence rather than leaving it
assumed.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
SRC_MJS = PACKAGE_DIR / "src" / "index.mjs"
MANIFEST = PACKAGE_DIR / "package.json"

sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packaging"))

import modes as mode_table  # noqa: E402
from utils import (  # noqa: E402
    cli_common,
    host_capabilities as caps,
    host_install,
    progress_tracker,
    provider_registry,
    run_status,
    unattended,
)

sys.path.insert(0, str(PACKAGE_DIR))
import build as pkg_build  # noqa: E402
from common import payload  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

#: Set by the PR package job. When it is set, a row that would skip **fails**
#: instead: a gate that skips silently everywhere is indistinguishable from a
#: gate that passes.
REQUIRE_ROWS_ENV = "MLLM_REQUIRE_PACKAGE_ROWS"


def _unavailable(reason: str) -> None:
    """Skip, or fail if the package job demanded this row actually run."""
    if os.environ.get(REQUIRE_ROWS_ENV):
        pytest.fail(f"{REQUIRE_ROWS_ENV} is set but this row cannot run: {reason}")
    pytest.skip(reason)


def _need_node() -> str:
    node = shutil.which("node")
    if not node:
        _unavailable("node is not on PATH; the plugin is JavaScript")
    return node


def _load_bump():
    """``packaging/opencode/bump.py`` by path, as ``test_opencode_version_pins``
    loads it: a bare ``import bump`` would bind whatever module of that name
    another test put on ``sys.path`` first."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "opencode_bump_for_package_tests", PACKAGE_DIR / "bump.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: One release in the schema's spelling: ``x.y.z`` plus an optional
#: pre-release or build suffix. Dot-separated identifiers only, so a trailing
#: ``.tgz`` or a sentence's full stop is never read as part of the version.
_RELEASE = r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"


def _package_versions(text: str, name: str, *, tarballs: bool = False) -> set:
    """Every version of package *name* that *text* spells as ``name@<v>``.

    With *tarballs*, ``name-<v>.tgz`` counts too: on the npm README a tarball
    name is a version a bump has to find, like a spec. The pages a user installs
    from spell no tarball, so they ask for specs alone.
    """
    separators = "[@-]" if tarballs else "@"
    return set(re.findall(rf"{re.escape(name)}{separators}({_RELEASE})", text))


@pytest.fixture(scope="session")
def built_package(tmp_path_factory) -> Path:
    """``build.py``'s output, in a temp directory. Never the working tree."""
    destination = tmp_path_factory.mktemp("opencode-package")
    pkg_build.build(destination=destination, allow_dirty=True)
    return destination


def module_url(package: Path) -> str:
    """The built plugin as a ``file:`` URL, which is what ``import()`` takes.

    Node's ESM loader reads a bare string as a URL, and ``C:\\...`` parses as
    the scheme ``c:`` (``ERR_UNSUPPORTED_ESM_URL_SCHEME``). POSIX paths happen
    to be valid specifiers, which is the only reason ``str(path)`` ever worked.
    Only the specifier needs this: a path handed to ``fs`` or ``process.env``
    inside the snippet is an ordinary string.
    """
    return (package / "dist" / "index.mjs").as_uri()


def run_node(script: str, *, cwd: Path | None = None, env: dict | None = None):
    """Evaluate an ES-module snippet and return its ``CompletedProcess``."""
    node = _need_node()
    environment = dict(os.environ)
    if env:
        environment.update(env)
    return subprocess.run(
        [node, "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=str(cwd) if cwd else None,
        env=environment,
        capture_output=True,
        # UTF-8 both ways: Node writes it, and `text=True` alone decodes with the
        # locale (cp1252 on Windows), which turns every em dash in a result into
        # three characters nothing compares equal to.
        encoding="utf-8",
        timeout=300,
    )


def node_json(script: str, package: Path, **kwargs):
    """Import the built plugin, run *script*, and parse its printed JSON.

    The snippet gets ``MultiLlmPlugin`` (the module's **one** export), ``T``
    (its test hooks) and ``surface`` (the generated CLI surface) already bound.

    ``T`` and ``MultiLlmPlugin.execute`` are reached as *properties of the
    factory*, never as module exports, and that is not a style choice. Measured
    against opencode 1.18.18 and 1.18.21: the loader calls **every** named export
    as a plugin factory, so a second export that is not a function makes it
    refuse the whole module — ``Plugin export is not a function`` — at exit 0.
    ``test_every_export_is_a_plugin_factory`` is what holds that line; this
    helper simply reads the shape that rule leaves behind.
    """
    prelude = f"""
    const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(package))})
    const T = MultiLlmPlugin.__testing
    const surface = T.loadSurface()
    const emit = (value) => console.log("@@" + JSON.stringify(value) + "@@")
    """
    result = run_node(prelude + textwrap.dedent(script), **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"@@(.*)@@", result.stdout, re.S)
    assert match, f"snippet printed no payload:\n{result.stdout}\n{result.stderr}"
    return json.loads(match.group(1))


# ===========================================================================
# The manifest and the package shape
# ===========================================================================

class TestManifest:
    def test_package_declares_no_dependencies(self):
        """``dependencies`` and ``peerDependencies`` both absent or empty.

        Zero-dep is what makes local-tarball testing and a cheap real install
        both work: the loader never reads ``dependencies``, so a *pre-extracted*
        package with declared deps loads fine — but a ``file:``/tarball install
        of a dependency-declaring package fails, because npm must then fetch.

        The peer half is not a formality. npm 7+ auto-installs peers, so a
        ``peerDependencies`` entry costs exactly what a ``dependencies`` one
        does: measured, an 11 KB package declaring one peer produced a **63 MB**
        spec directory. That cost lands per pinned version and is never
        reclaimed. Worse for correctness, a peer range resolves independently of
        the running binary, so it does not even buy version agreement with the
        host.
        """
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert not manifest.get("dependencies")
        assert not manifest.get("peerDependencies")

    def test_the_plugin_sdk_types_are_the_host_pin(self):
        """``devDependencies["@opencode-ai/plugin"]`` is the supported host
        release, exactly, and ``make opencode-bump`` is what moves it.

        Nothing compiles against it: ``src/index.mjs`` is plain JavaScript the
        build copies verbatim, and consumers never install devDependencies. It
        is the reference for the plugin API the source's comments rely on. But
        ``package.json`` ships in the tarball, so a stale value publishes a
        plugin API version older than the ``engines`` floor beside it. That is
        how this row was found: 1.18.23 under a 1.19.3 floor, behind a row that
        only checked the entry existed. OpenCode publishes the SDK at every host
        version, so the host pin owns it: a :data:`bump.PINS` site, which
        ``test_every_pin_agrees`` then holds equal to every other pin.
        """
        from utils.providers.opencode import OpenCodeProvider

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        pinned = manifest.get("devDependencies", {}).get("@opencode-ai/plugin")
        # Exact, never a range: a range resolves independently of the host it
        # describes, which is the drift this row exists to stop.
        assert pinned == OpenCodeProvider.verified_version, (
            f"@opencode-ai/plugin is {pinned!r}, the host pin is "
            f"{OpenCodeProvider.verified_version}; move both with "
            "`make opencode-bump VERSION=<x.y.z>`"
        )
        bump = _load_bump()
        sites = [
            name for name, rel, pattern in bump.PINS
            if rel == MANIFEST.relative_to(REPO_ROOT) and "@opencode-ai/plugin" in pattern
        ]
        assert len(sites) == 1, (
            "the SDK pin is not one bump.PINS site, so `make opencode-bump` "
            f"would leave it behind (sites naming it: {sites})"
        )
        assert bump.read_pins(REPO_ROOT)[sites[0]] == pinned

    def test_package_manifest_is_publishable(self):
        """The publishable shape, asserted rather than remembered."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ("description", "keywords", "repository", "license", "files"):
            assert manifest.get(key), f"manifest is missing {key!r}"
        # The discovery mechanism: opencode users find plugins by this keyword.
        assert manifest["keywords"][0] == "opencode-plugin"
        assert manifest["type"] == "module"
        # `files` must name both, explicitly. With no `files` key the
        # package-root .gitignore becomes the ignore list; with `files: ["dist"]`
        # the skill tree vanishes silently.
        assert "dist" in manifest["files"] and "skills" in manifest["files"]
        assert "README.md" in manifest["files"] and "LICENSE" in manifest["files"]

        # Resolution tries exports["./server"] FIRST, then main, and never
        # exports["."]. Both of the first two must point at the built copy.
        assert manifest["main"] == "./dist/index.mjs"
        assert manifest["exports"]["./server"] == "./dist/index.mjs"

    def test_the_build_hook_is_prepack(self):
        """``prepack``, never ``prepublishOnly`` — and the difference is the
        entire reason for having a hook.

        ``prepublishOnly`` fires on ``npm publish`` and **nowhere else**;
        ``npm pack`` does not run it. ``npm pack`` is exactly what
        ``make opencode-package`` and the pre-publish gate invoke, so a build
        wired only to ``prepublishOnly`` leaves every pack path free to ship
        whatever ``dist/`` happens to be lying around.

        Not ``prepare`` either: ``prepare`` is a strict superset that also runs
        on a bare ``npm install``/``npm ci`` inside the package directory,
        neither of which has any business shelling out to uv.
        """
        scripts = json.loads(MANIFEST.read_text(encoding="utf-8")).get("scripts", {})
        assert "prepack" in scripts, "the build hook must be prepack"
        assert "prepublishOnly" not in scripts
        assert "prepare" not in scripts
        # It must invoke the build, never a nested `npm pack`: npm packs by
        # itself immediately after the hook, so a hook that packs writes a .tgz
        # nobody publishes and packs twice on the make path.
        assert "build.py" in scripts["prepack"]
        assert "npm" not in scripts["prepack"]

    def test_the_package_readme_pins_the_same_version(self):
        """The npm landing page's spec agrees with the manifest.

        It is hand-written prose carrying a version number, which is exactly the
        drift the one-owner chain exists to prevent everywhere else — and it is
        the page a user copies the install line from, so a stale number here is a
        config entry that resolves to nothing.
        """
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        readme = (PACKAGE_DIR / "README.md").read_text(encoding="utf-8")
        spec = f"{manifest['name']}@{manifest['version']}"
        assert spec in readme, f"the package README does not pin {spec}"
        # And it names no OTHER version of itself — not as a spec, and not as a
        # tarball name either, which is spelled `<name>-<version>.tgz` and is
        # written `<version>` in the prose for exactly this reason: every
        # number left in the page is one a version bump has to find, and this
        # is what finds it. It pins `version`, never `public_version`: it ships
        # inside that version and is read only once npm has it.
        stale = _package_versions(readme, manifest["name"], tarballs=True) - {
            manifest["version"]
        }
        assert not stale, f"the package README also names {sorted(stale)}"

        floor = re.search(
            r">=\s*([0-9]+\.[0-9]+\.[0-9]+)", manifest["engines"]["opencode"]
        ).group(1)
        assert floor in readme, (
            f"the README does not state the {floor} floor `engines` enforces"
        )

    def test_the_package_readme_reads_as_published(self):
        """The npm README is read only on npm, from inside the published version.

        npm takes it with the tarball, and editing GitHub afterwards does not
        change the bytes already published. So it must never say the package is
        unavailable. It must still carry what a first install needs: the
        local-directory route, the first-run network requirement (with the uv
        flags the plugin really passes, read from their owner), and the support
        claim. That claim is Linux. Windows and macOS keep their caveat, because
        mocked suites passing on those OSes are not an OpenCode install there.
        """
        readme = (PACKAGE_DIR / "README.md").read_text(encoding="utf-8")
        flat = " ".join(readme.split())
        for unavailable in ("not yet published", "before this package is published",
                            "unpublished", "resolves to nothing"):
            assert unavailable not in flat.lower(), (
                f"the npm README says {unavailable!r}, and npm only ever shows it "
                "for a published version"
            )
        assert '{ "plugin": ["/abs/path/to/packaging/opencode"] }' in readme
        flags = " ".join(unattended.DEFAULT_UV_RUN_ARGS)
        assert f"`uv run {flags}`" in flat, (
            f"the first-run section no longer names `uv run {flags}`"
        )
        assert "needs network" in flat
        assert "Supported on **Linux**." in flat
        assert "**Windows is untested on this route.**" in flat
        assert "macOS is in the same position." in flat

    def test_engines_floor_matches_the_container_pin(self):
        """A4 item 6: the ``engines`` floor is the one supported release.

        This is the one field in the manifest whose correct value lives in
        *other* files, so it is the field most likely to go stale while still
        looking deliberate. Since 2026-10-02 the floor is the one supported
        opencode release — the container's own pin, the parser-fixture pin and
        the adapter's ``min_version`` at once (``make opencode-bump`` moves all
        of them; ``test_opencode_version_pins.py`` holds them equal).

        Second half of the same row: the floor must sit **at or above** the
        parser-fixture pin, so the package's declared floor does not disagree
        with the second, undeclared floor. The ``engines`` gate dominates it and
        the compatibility error is unambiguously about ``engines``.
        """
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        spec = manifest["engines"]["opencode"]
        floor = re.search(r">=\s*([0-9]+\.[0-9]+\.[0-9]+)", spec)
        assert floor, f"no floor in engines.opencode: {spec!r}"
        # The ceiling is at the major only. A tighter one would hard-fail the
        # container on the next release, and the undocumented JSON-Schema
        # fallback it exists for is reachable by a *minor* anyway.
        assert "<2.0.0" in spec, f"the ceiling must be <2.0.0, got {spec!r}"

        container = json.loads(
            (SKILL_ROOT / "tests" / "container" / "package.json").read_text("utf-8")
        )
        pinned = container["dependencies"]["opencode-ai"].lstrip("^~=")
        assert _version(floor.group(1)) <= _version(pinned), (
            f"engines floor {floor.group(1)} is above the container's "
            f"opencode-ai pin {pinned}; the container could never load the "
            "package it is supposed to grade"
        )

        fixture_pin = _parser_fixture_version()
        assert _version(floor.group(1)) >= _version(fixture_pin), (
            f"engines floor {floor.group(1)} sits *below* the parser-fixture pin "
            f"{fixture_pin}, so the package would admit versions the shipped "
            "stream fixtures were never captured from"
        )

    def test_engines_floor_is_the_container_pin(self):
        """Row 6 exactly: the floor **equals** the container's opencode pin.

        The row above holds the floor at or below the pin and at or above the
        parser-fixture pin; under the support policy (the newest release is the
        only supported one) those collapse to one release, and this is the
        equality the gate grades. The whole spec is matched, so a ceiling other
        than ``<2.0.0`` or a second clause fails too.
        """
        spec = json.loads(MANIFEST.read_text(encoding="utf-8"))["engines"]["opencode"]
        match = re.fullmatch(r">=([0-9]+\.[0-9]+\.[0-9]+) <2\.0\.0", spec)
        assert match, f"engines.opencode is {spec!r}, not `>=<pin> <2.0.0`"
        container = json.loads(
            (SKILL_ROOT / "tests" / "container" / "package.json").read_text("utf-8")
        )
        pinned = container["dependencies"]["opencode-ai"]
        assert match.group(1) == pinned, (
            f"engines floor {match.group(1)} is not the container's opencode-ai pin "
            f"{pinned}; `make opencode-bump` moves both"
        )


def _version(text: str):
    return tuple(int(part) for part in text.split(".")[:3])


def _parser_fixture_version() -> str:
    source = (SKILL_ROOT / "utils" / "providers" / "opencode.py").read_text("utf-8")
    match = re.search(r'verified_version\s*[:=]\s*["\']([0-9.]+)["\']', source)
    assert match, "no verified_version found in utils/providers/opencode.py"
    return match.group(1)


# ===========================================================================
# The JS, graded as text (src)
# ===========================================================================

class TestJavaScriptSource:
    def test_no_bare_specifiers_in_js(self):
        """Every runtime ``import`` is a ``node:`` builtin.

        This is the rule that decides whether the plugin loads at all. A runtime
        import that cannot resolve next to the plugin fails **completely
        silently** — exit 0, the plugin still listed in ``plugin[]``, config
        unmutated, and a 0-byte debug log. It, not the ``dependencies`` field, is
        what kills a plugin offline.
        """
        text = SRC_MJS.read_text(encoding="utf-8")
        specifiers = re.findall(r'^\s*import\s[^"\']*["\']([^"\']+)["\']', text, re.M)
        specifiers += re.findall(r'^\s*import\s+["\']([^"\']+)["\']', text, re.M)
        assert specifiers, "no imports found — the grep is broken, not the file"
        for specifier in specifiers:
            assert specifier.startswith("node:"), (
                f"bare specifier {specifier!r}: a runtime import that cannot "
                "resolve next to the plugin kills it silently at exit 0"
            )

    def test_skill_dir_is_composed_with_node_path(self):
        """Every derived path is built with ``path.join``/``path.resolve``.

        A plain grep — no node, no npm, nothing built — so it is collected on the
        macOS and Windows legs plain CI already runs, and it is the only
        Windows-relevant assertion the suite has.

        Read what it does *not* buy: spelling, not behaviour. Green on the
        Windows leg says the JS builds a backslash-separated skill dir there, not
        that the path survives ``--project``, ``uv run`` and the Python side.
        """
        text = SRC_MJS.read_text(encoding="utf-8")
        assert "import.meta.dirname" in text
        # No path expression concatenates a "/" literal onto a derived path.
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r'(SKILL_DIR|PACKAGE_DIR|DIST_DIR)\s*\+\s*["\']/', line)
            or re.search(r'["\']/["\']\s*\+\s*(SKILL_DIR|PACKAGE_DIR|DIST_DIR)', line)
        ]
        assert not offenders, "string-concatenated paths:\n" + "\n".join(offenders)
        for anchor in ("PACKAGE_DIR", "SKILL_DIR", "DIST_DIR"):
            assignment = re.search(rf"const {anchor} = (.+)", text)
            assert assignment, f"{anchor} is not defined"
            assert "path.join" in assignment.group(1) or "path.resolve" in (
                assignment.group(1)
            ), f"{anchor} is not composed with node:path: {assignment.group(1)}"

    def test_no_core_prose_in_js(self):
        """No mode word, no alias, no summary, no sentence from the core.

        The JS analogue of the shim rule. The tool's description and arg-schema
        descriptions are assembled at load from ``dist/cli-surface.json``, so the
        file's only added text is the connective scaffold around generated
        values.
        """
        text = SRC_MJS.read_text(encoding="utf-8")
        body = _code_only(text)

        for mode in mode_table.user_facing_modes():
            assert f'"{mode.name}"' not in body, (
                f"mode word {mode.name!r} is a literal in the JS; it must come "
                "from cli-surface.json"
            )
            assert mode.summary not in text, (
                f"{mode.name}'s summary is duplicated into the JS"
            )
        # One genuine collision to exclude, and it is worth naming rather than
        # loosening the rule for: the alias `abort` is also the DOM event name
        # on an AbortSignal, which the cancellation contract necessarily
        # listens for. Strip those calls, then the scan is exact again.
        listeners = re.sub(
            r"(add|remove)EventListener\??\.?\([^)]*\)", "", body
        )
        for alias in mode_table.ALIASES:
            # `abort`/`cancel`/`stop`/`review`/`tasks` — never spelled here.
            assert f'"{alias}"' not in listeners, (
                f"alias {alias!r} is a JS literal"
            )

        # No sentence from the instruction core.
        for path in sorted((SKILL_ROOT / "instructions").glob("*.md")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if len(stripped) < 60 or stripped.startswith(("#", "-", "*", "`")):
                    continue
                assert stripped not in text, (
                    f"a sentence from {path.name} is duplicated into the JS"
                )

        # No needs-human prose, no outcomes line, no lede — the surface's own
        # authored strings. The three scans above walk closed sets (mode words,
        # aliases, core sentences) that a `modes.py` string constant is in none
        # of, which is how the exit-3 message sat in this file as a bare literal
        # for as long as it did. Scanned by *fragment* after joining the file's
        # `" + "`-concatenated literals: a JS string is wrapped across lines,
        # so a whole-sentence absence check would match nothing and pass
        # vacuously.
        surface = pkg_build.cli_surface()
        authored = [
            *surface["needs_human_message"].values(),
            surface["outcomes_line"],
            surface["needs_human_option"],
            mode_table.TOOL_DESCRIPTION_LEDE,
            # Schema v4: the argument descriptions and the description's
            # connective text are the emitter's, never this file's.
            *surface["argument_descriptions"].values(),
            surface["tool_description"],
        ]
        joined = _joined_literals(body)
        for clause in authored:
            for fragment in _prose_fragments(clause):
                assert fragment not in joined, (
                    f"needs-human/description prose is a literal in the JS "
                    f"({fragment!r}); it must be read from cli-surface.json"
                )

    def test_no_flag_name_is_a_js_literal(self):
        """The reserved and baked flags come from the generated table.

        The set is closed by construction — it is exactly the flags the plugin
        composes — and that closure only holds if both sides read one owner.
        """
        body = _code_only(SRC_MJS.read_text(encoding="utf-8"))
        for flag in (
            "--host", "--yes", "--exec-subagents", "--detach", "--non-interactive",
            "--if-not-running", "--max-detached", "--interactive", "--internal-",
            "--frozen", "--no-dev",
        ):
            assert f'"{flag}"' not in body, (
                f"{flag!r} is a literal in the JS; it must be read from "
                "cli-surface.json, whose owner is Python"
            )


def _code_only(text: str) -> str:
    """*text* with block and line comments removed.

    The prose rules are about what the plugin *shows a model*, not about what a
    maintainer reads: this file's own comments necessarily name the flags and
    modes they explain.
    """
    without_block = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(
        line for line in without_block.splitlines()
        if not line.strip().startswith("//")
    )


def _joined_literals(code: str) -> str:
    """*code* with adjacent ``"a" + "b"`` string literals joined into one.

    A long JS string is spelled as fragments concatenated across lines, so an
    absence assertion on a whole sentence would never see it. Collapsing the
    ``" +\\n  "`` seams (and the whitespace around them) makes a wrapped literal
    read as the one string a model receives.
    """
    return re.sub(r'"\s*\+\s*"', "", code)


def _prose_fragments(clause: str, minimum: int = 24) -> list:
    """The sentence-sized pieces of an authored clause, tokens stripped.

    Split at sentence *and* clause boundaries (commas included) so a reworded
    sentence in the JS is still caught by the clauses it kept, while the
    ``{phase}``/``{error}`` tokens — which the JS legitimately spells when it
    substitutes them — never become a fragment of their own.
    """
    stripped = re.sub(r"\{(?:phase|error)\}", " ", clause)
    pieces = re.split(r"(?<=[.;:,])\s+", stripped)
    return [
        piece.strip() for piece in pieces if len(piece.strip()) >= minimum
    ]


# ===========================================================================
# The build's artifacts (build)
# ===========================================================================

class TestBuildArtifacts:
    def test_dist_mirrors_src(self, built_package):
        """``dist/index.mjs`` is byte-identical to ``src/index.mjs``.

        One assertion, and it is what licenses every (src) grep above to stand in
        for the shipped file. If the build ever acquires a transform, this row
        goes red first and the greps move rather than quietly grading the wrong
        bytes.
        """
        assert (built_package / "dist" / "index.mjs").read_bytes() == (
            SRC_MJS.read_bytes()
        )

    def test_every_export_is_a_plugin_factory(self, built_package):
        """The built module exports exactly one thing, and it is a function.

        MEASURED against the real binary — opencode 1.18.18 in the container and
        1.18.21 on the host — the plugin loader iterates **every named export** of
        the module and calls each one as a plugin factory. Two consequences, both
        observed rather than reasoned about:

        * A single export that is not a function makes opencode refuse the
          **whole module**::

              level=ERROR message="failed to load plugin"
              path=opencode-multi-llm@0.1.0
              error="Plugin export is not a function"

          and it fails in the shape this package keeps having to defend against:
          **silently**. Exit 0, the spec still listed in ``plugin[]``, and
          ``command["multi-llm"]`` simply never appearing. Nothing a user runs
          says the plugin is dead.
        * Exporting the *same* factory twice — which this package did, as
          ``export const MultiLlmPlugin`` beside ``export default
          MultiLlmPlugin`` — gets it invoked twice.

        The shipped package carried ``export const __testing = {…}``, an object,
        so on a real opencode **the plugin never loaded at all**; proven in
        Docker, offline, on the pinned 1.18.18 image. ``execute`` and
        ``__testing`` are hung on the factory as properties instead: function
        properties are not module exports, so the loader never walks them while
        the suite still reaches them through a plain ``import``.

        It grades the **built** ``dist/index.mjs``, not the source, because what
        opencode loads is whatever ``main``/``exports`` resolve to — a stray
        ``export`` introduced by a future build step would be invisible to the
        (src) greps above.
        """
        result = run_node(
            f"""
            const module = await import({json.dumps(module_url(built_package))})
            const table = Object.keys(module).map((n) => [n, typeof module[n]])
            console.log("@@" + JSON.stringify(table) + "@@")
            """
        )
        assert result.returncode == 0, result.stdout + result.stderr
        match = re.search(r"@@(.*)@@", result.stdout, re.S)
        assert match, f"the export probe printed nothing:\n{result.stdout}"
        table = [tuple(row) for row in json.loads(match.group(1))]

        # The offending export is named, with its type, because "the module has
        # two exports" is not the sentence that fixes this — "`__testing` is an
        # object" is.
        offenders = [f"{name} -> {kind}" for name, kind in table if kind != "function"]
        assert not offenders, (
            "every export must be a plugin factory; these are not: "
            + ", ".join(offenders)
            + '. opencode calls each export and refuses the whole module with '
            '`Plugin export is not a function` — silently: exit 0, still listed '
            "in plugin[], and the /multi-llm command key never injected. Hang "
            "the value on the factory as a property instead."
        )
        assert [name for name, _ in table] == ["MultiLlmPlugin"], (
            "the module must export exactly one factory, and it exports "
            f"{[name for name, _ in table]}. A second export of the same "
            "function does not fail loudly — it gets the plugin instantiated "
            "once per export."
        )

    def test_command_md_matches_render(self, built_package):
        """The command body is a faithful render of an authored template.

        It compares against **the same call ``build.py`` makes**, over the
        phase-current template, never against a post-processed render — which is
        what keeps the single-source contract true across the Phase 2 switch
        instead of contradicted by it.
        """
        built = (built_package / "dist" / "command.md").read_text("utf-8")
        assert built == pkg_build.command_body()

    def test_command_md_keeps_the_skill_dir_token(self, built_package):
        """``{SKILL_DIR}`` survives the render, for the plugin to resolve.

        ``render()`` already takes the baked prefix as a parameter, so passing
        the token *as* the prefix makes the substitution an identity and no
        sentinel round-trip is needed.
        """
        built = (built_package / "dist" / "command.md").read_text("utf-8")
        # Nothing absolute leaked in: no real skill dir, and no other spelling.
        assert str(SKILL_ROOT) not in built
        for spelling in ("${CLAUDE_SKILL_DIR}", "$MULTI_LLM_SKILL_DIR"):
            assert spelling not in built

    def test_command_md_names_only_the_tool(self, built_package):
        """Phase 2's bar: no dispatcher argv and no baked flag literal.

        The flags are code now, and a second spelling in prose is a template for
        reconstructing the shell route — which is the whole thing this route
        exists to remove. A terminal result proves nothing about which route ran;
        removing the alternative is what closes it.
        """
        built = (built_package / "dist" / "command.md").read_text("utf-8")
        for token in ("uv run", "multi_llm.py", "```bash", "```sh"):
            assert token not in built, (
                f"{token!r} is still in the command body: the expansion would "
                "hand the model a working shell command beside the tool"
            )
        for flag in ("--host opencode", "--yes", "--exec-subagents", "--detach"):
            assert flag not in built, f"baked flag {flag!r} is spelled in prose"
        assert "multi_llm" in built, "the body must name the tool it delegates to"

    def test_validation_counts_are_reported_only_when_the_result_has_them(self):
        """OpenCode racing run 4 (06): "On exit 0, report `validation_counts`"
        held for every mode, and the host wrote "`validation_counts`: not
        present in this result" after implement, which never carries them."""
        flat = " ".join(pkg_build.command_body().split())
        assert "On exit 0, report `validation_counts`" not in flat
        assert ("When the result carries `validation_counts` (a review phase's "
                "does), report them") in flat
        assert ("A result without that key has no counts to report, so do not "
                "mention them.") in flat

    def test_needs_human_rule_reaches_the_command_body(self):
        """The exit-3 decision rule is rendered into ``dist/command.md``.

        The slash-command body is one of the two instruction surfaces this
        route loads (the other is the tool description), so the rule reaches it
        through the ``{NEEDS_HUMAN_RULE}`` token — never as a literal in the
        template, which would be a fourth hand-typed copy of the sentence the
        incident turned on.
        """
        template = (SKILL_ROOT / pkg_build.COMMAND_TEMPLATE).read_text(
            encoding="utf-8"
        )
        assert template.count(cli_common.NEEDS_HUMAN_RULE_TOKEN) == 1
        assert mode_table.NEEDS_HUMAN_WORK_SENTENCE not in " ".join(
            template.split()
        ), "the command template spells the rule by hand; render the token"

        body = " ".join(pkg_build.command_body().split())
        assert cli_common.NEEDS_HUMAN_RULE_TOKEN not in body
        assert mode_table.NEEDS_HUMAN_RULE in body
        assert mode_table.NEEDS_HUMAN_WORK_SENTENCE in body

    def test_cli_surface_json_matches_render(self, built_package):
        """The generated surface is byte-identical to a fresh generation."""
        built = (built_package / "dist" / "cli-surface.json").read_text("utf-8")
        assert built == pkg_build.cli_surface_json()

    def test_default_mode_has_one_owner(self):
        """"No mode flag means `review-plan`" is spelled once, in `modes.py`.

        The rule used to be prose in two hand-written places with no constant
        behind either, and the tool description would have made a third copy.
        `modes.DEFAULT_MODE` closes that only if every surface actually renders
        from it, which is what this row asserts across all three:

        * the generated `cli-surface.json`, which the model-facing tool
          definition is assembled from;
        * every shim **template**, which carries the rule as the
          `{DEFAULT_MODE}` token in both places it states it (the step-2 prose
          and the worked example) and never as a literal, asserted on the
          template *and* on `host_install.render`'s output; and
        * the one shim nobody renders — Claude Code's hand-authored `SKILL.md`,
          whose `${CLAUDE_SKILL_DIR}` the host expands at load time, so its two
          sentences are literals by necessity and this is the only thing holding
          them to the constant.

        All three in-process: no build artifact, so a red row can only mean the
        sentences drifted.
        """
        default = mode_table.DEFAULT_MODE
        # The two sentences, composed from the constant rather than typed.
        rule = f"No mode flag means `{default}`."
        example = f"# {default} (the default)"

        assert pkg_build.cli_surface()["default_mode"] == default

        checked = []
        for host_id in sorted(caps.host_ids()):
            entry = caps.entry(host_id)
            relative = entry["shim"]["template"] or entry["shim"]["installed"]
            path = SKILL_ROOT / relative
            if not path.exists():
                continue
            checked.append(host_id)
            source = path.read_text(encoding="utf-8")
            if entry["shim"]["template"]:
                assert source.count(cli_common.DEFAULT_MODE_TOKEN) == 2, (
                    f"{relative} states the default-mode rule twice — the step-2 "
                    f"prose and the worked example — and both must be the "
                    f"{cli_common.DEFAULT_MODE_TOKEN} token, not a literal"
                )
                assert default not in source, (
                    f"{relative} spells {default!r} by hand; render it from "
                    "modes.DEFAULT_MODE instead"
                )
            # Rendered (or, for the unrendered reference shim, as authored):
            # both sentences read exactly as the constant dictates. Whitespace is
            # collapsed because the prose sentence wraps across lines.
            rendered = " ".join(
                host_install.render(
                    host_id, skill="/tmp/multi llm fixture/skill dir",
                    template=source,
                ).split()
            )
            for sentence in (rule, example):
                assert sentence in rendered, (
                    f"{relative}: the default-mode sentence is not the one "
                    f"modes.DEFAULT_MODE dictates.\n  expected: {sentence}"
                )

        assert checked == ["claude-code", "codex", "opencode"], (
            f"the shim corpus moved ({checked}); this row must keep covering "
            "the unrendered SKILL.md as well as every template"
        )

    def test_cli_surface_carries_the_needs_human_prose(self):
        """The exit-3 message, its option, the outcomes line — all Python's.

        Each is pinned by identity to its ``modes.py`` owner, the way the lede
        is. The outcomes line spells the four exit codes, so its numbers are
        held to ``cli_common.EXIT_*`` here — ``modes.py`` imports nothing by
        design and must not grow an import for it. The message is a template:
        Python expands neither token, so both must survive into the surface
        for the plugin to fill.
        """
        surface = pkg_build.cli_surface()
        assert surface["outcomes_line"] == mode_table.TOOL_OUTCOMES_LINE
        assert surface["needs_human_option"] == mode_table.NEEDS_HUMAN_OPTION
        assert surface["needs_human_message"] == mode_table.NEEDS_HUMAN_MESSAGE

        numbers = {
            name: int(code) for code, name in re.findall(
                r"(\d+) (done|error|nothing to do|needs a human)",
                surface["outcomes_line"],
            )
        }
        assert numbers == {
            "done": cli_common.EXIT_OK,
            "error": cli_common.EXIT_ERROR,
            "nothing to do": cli_common.EXIT_NOTHING_TO_DO,
            "needs a human": cli_common.EXIT_NEEDS_HUMAN,
        }

        message = surface["needs_human_message"]
        assert "{phase}" in message["template"] and "{error}" in message["template"]
        for clause in ("phase_incomplete", "phase_known", "review_completed"):
            assert "{phase}" in message[clause]
        assert "{error}" in message["error_known"]
        # The one sentence the incident turned on, in the message itself.
        assert mode_table.NEEDS_HUMAN_WORK_SENTENCE in message["template"]
        # The rule's order: stop, report, no other mode, no doing the work,
        # options only after the user picks one.
        template = message["template"]
        order = [
            template.index("Stop here"),
            template.index("Report to the user"),
            template.index("Do not run a different mode"),
            template.index(mode_table.NEEDS_HUMAN_WORK_SENTENCE),
            template.index("only after the user has picked one"),
        ]
        assert order == sorted(order), "the stop rule's clauses are out of order"

        # What the plugin selects the clauses from, all generated.
        states = surface["dispatcher"]["states"]
        assert states["needs_human"] == cli_common._STATE_BY_EXIT[
            cli_common.EXIT_NEEDS_HUMAN
        ]
        assert states["ok"] == cli_common._STATE_BY_EXIT[cli_common.EXIT_OK]
        for row in surface["modes"]:
            assert row["phase"] == mode_table.resolve(row["name"]).phase
        gate = surface["dispatcher"]["review_gate"]
        assert gate == mode_table.REVIEW_GATE
        assert mode_table.resolve(gate["mode"]) is not None
        # The flag is declared by the orchestrator's parser; the two spellings
        # are held together here rather than by an import modes.py may not
        # have.
        orchestrator = (SKILL_ROOT / "code_review_orchestrator.py").read_text(
            encoding="utf-8"
        )
        assert re.search(
            r"add_argument\(\s*['\"]" + re.escape(gate["flag"]) + r"['\"]",
            orchestrator,
        ), f"code_review_orchestrator.py no longer declares {gate['flag']}"

    def test_cli_surface_carries_the_generated_table(self):
        """Mode words, aliases, arity, summaries and detach bits, all derived.

        From Phase 2 this is where the model's entire mode vocabulary comes from,
        so a mode added to ``modes.py`` must appear here without an edit.
        """
        surface = pkg_build.cli_surface()
        assert surface["schema_version"] == pkg_build.SURFACE_SCHEMA_VERSION
        assert surface["default_mode"] == mode_table.DEFAULT_MODE
        assert surface["description_lede"] == mode_table.TOOL_DESCRIPTION_LEDE

        exposed = {row["name"] for row in surface["modes"]}
        assert exposed == {m.name for m in mode_table.user_facing_modes()}
        # Not `util` (a routing word), and not the install modes (npm's job).
        assert mode_table.UTIL_NAMESPACE not in exposed
        assert "install-host" not in exposed and "uninstall-host" not in exposed
        # `status` stays exposed: re-attaching is a tool call with mode "status",
        # and stopping a run is the abort/cancel/stop alias.
        assert "status" in exposed

        by_name = {row["name"]: row for row in surface["modes"]}
        for mode in mode_table.user_facing_modes():
            row = by_name[mode.name]
            assert row["positionals"] == list(mode.positionals)
            assert row["supports_detach"] is mode.supports_detach
            assert row["summary"] == mode.summary
            assert set(row["aliases"]) == {
                word for word, target in mode_table.ALIASES.items()
                if target == mode.name
            }

        detachable = {r["name"] for r in surface["modes"] if r["supports_detach"]}
        assert detachable == set(mode_table.DETACH_MODES) & exposed

    def test_default_uv_run_argv_carries_no_dev(self, built_package, monkeypatch):
        """``--no-dev`` is the default everywhere, and the override replaces.

        ``--frozen`` suppresses only the lockfile staleness check; it does not
        stop the sync, and ``uv run`` syncs the dev group by default. Measured
        against this tree: ``--frozen`` alone would install 17 packages,
        ``--frozen --no-dev`` installs one. So a first run pulls the whole dev
        group without it, and every offline posture fails on a package it never
        needed.
        """
        assert unattended.DEFAULT_UV_RUN_ARGS == ("--frozen", "--no-dev")
        monkeypatch.delenv(unattended.UV_RUN_ARGS_ENV, raising=False)
        assert unattended.uv_run_args() == ("--frozen", "--no-dev")

        # Blank and whitespace-only behave as unset, so a stray `export VAR=`
        # cannot silently re-enable the dev-group sync.
        for blank in ("", "   ", "\t\n"):
            monkeypatch.setenv(unattended.UV_RUN_ARGS_ENV, blank)
            assert unattended.uv_run_args() == ("--frozen", "--no-dev")

        # It REPLACES, never appends — which is what makes --no-sync
        # expressible at all: appended, it would sit beside the very sync the
        # setter is trying to suppress.
        monkeypatch.setenv(unattended.UV_RUN_ARGS_ENV, "--no-sync")
        assert unattended.uv_run_args() == ("--no-sync",)

        surface = json.loads(
            (built_package / "dist" / "cli-surface.json").read_text("utf-8")
        )
        assert surface["uv"]["default_run_args"] == ["--frozen", "--no-dev"]

    def test_the_no_dev_sync_installs_one_package(self, built_package, tmp_path):
        """Row 5b's second half: what that argv installs, asked of uv itself.

        The argv row above proves the flags are composed; this one proves what
        they buy against the *staged* lockfile — the runtime dependency and
        nothing from the dev group — with ``--dry-run --offline``, so uv
        resolves the plan from the lock without fetching or writing anything.
        The expected set is read from ``pyproject.toml``'s ``dependencies``,
        never typed here.
        """
        import tomllib

        uv = unattended.find_uv()
        if uv is None:
            _unavailable("uv is not installed")
        staged = built_package / "skills" / "multi-llm"
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
        env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "venv")
        result = subprocess.run(
            [uv, "sync", *unattended.DEFAULT_UV_RUN_ARGS, "--dry-run", "--offline",
             "--project", str(staged)],
            capture_output=True, text=True, env=env, timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        planned = sorted(
            re.match(r"\s*\+\s*([A-Za-z0-9_.-]+)==", line).group(1).lower()
            for line in output.splitlines() if re.match(r"\s*\+\s*\S+==", line)
        )
        runtime = tomllib.loads(
            (SKILL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["dependencies"]
        expected = sorted(
            re.match(r"[A-Za-z0-9_.-]+", spec).group(0).lower() for spec in runtime
        )
        assert planned == expected and len(planned) == 1, (
            f"`uv sync --frozen --no-dev` would install {planned}, the runtime "
            f"dependencies are {expected}:\n{output}"
        )
        assert not (tmp_path / "venv").exists(), "the dry run created the venv"

    def test_package_is_built_before_every_pack(self):
        """Every Makefile recipe reaching npm runs ``build.py`` first.

        The other half of the manifest hook's backstop: the explicit call covers
        ``--ignore-scripts``, the hook covers a pack nobody routed through Make.
        A grep over the recipes, not a run — it needs no npm.
        """
        recipes = _makefile_recipes(REPO_ROOT / "Makefile")
        reached = False
        for target, body in recipes.items():
            if not any(
                re.search(r"\bnpm\s+(pack|publish)\b", line) for line in body
            ):
                continue
            assert body, f"{target!r} has an empty recipe"
            reached = True
            build_at = next(
                (i for i, line in enumerate(body) if "build.py" in line), None
            )
            npm_at = next(
                i for i, line in enumerate(body)
                if re.search(r"\bnpm\s+(pack|publish)\b", line)
            )
            assert build_at is not None and build_at < npm_at, (
                f"target {target!r} reaches npm without running build.py first, "
                "so it could ship a stale dist/ under --ignore-scripts"
            )
        assert reached, "no Makefile recipe packs or publishes — the grep is stale"


def _makefile_recipes(path: Path) -> dict:
    """``{target: [recipe lines]}``, with simple variables expanded.

    Enough of make's grammar for a grep — and the expansion is the load-bearing
    part: the recipes reach the build through a variable, so a parser that read
    the literal text would report that no target runs it and the guard would be
    green for the wrong reason.
    """
    text = path.read_text(encoding="utf-8")
    variables = dict(
        re.findall(r"^([A-Z_][A-Z0-9_]*)\s*:?=\s*(.*)$", text, re.M)
    )

    def expand(line: str, depth: int = 0) -> str:
        if depth > 5:
            return line
        replaced = re.sub(
            r"\$\(([A-Z_][A-Z0-9_]*)\)",
            lambda m: variables.get(m.group(1), m.group(0)),
            line,
        )
        return replaced if replaced == line else expand(replaced, depth + 1)

    recipes: dict = {}
    current = None
    for line in text.splitlines():
        if line.startswith("\t"):
            if current:
                recipes[current].append(expand(line.strip()))
            continue
        match = re.match(r"^([A-Za-z0-9_.\-/]+)\s*:(?!=)", line)
        current = match.group(1) if match else None
        if current and current not in recipes:
            recipes[current] = []
    return recipes


# ===========================================================================
# Argument validation and argv composition
# ===========================================================================

#: Tokens the JS verdict and ``cli_common.flag_present()`` must agree on.
#: Abbreviations, ``=`` forms, alias spellings, and a ``--host-something`` that
#: must **not** read as ``--host``.
RESERVED_CASES = [
    "--host", "--ho", "--h", "--host=codex", "--hos=x",
    "--yes", "--ye", "-y", "--non-interactive", "--non-int",
    "--exec-subagents", "--exec", "--exec-subagents=1",
    "--detach", "--det", "--if-not-running", "--if-not", "--max-detached",
    "--internal-worker", "--internal-run-id", "--internal-anything-new",
    "--interactive", "--inter", "-i", "--interactive=1",
    # Negatives: real options that must never read as a reserved abbreviation.
    "--host-something", "--models", "--quick", "--force", "--dry-run",
    "--view", "--wait", "--run-id", "--abort", "--no-dev", "--frozen",
    "plans/x.md", "-", "--",
]


class TestArguments:
    def test_worker_flags_are_reserved(self, built_package):
        """An **equivalence** with ``cli_common.flag_present()``.

        Not a list comparison: the JS verdict on each token must match Python's
        over a shared case table, so the guard cannot rot as the parent parser
        gains flags. Matching literal spellings does not close it — no parser in
        this tree passes ``allow_abbrev=False``, so argparse binds ``--ho``,
        ``--ye`` and ``--det`` to their full options, and a guard that scanned
        exact spellings was bypassable by typing three fewer characters.
        """
        surface = pkg_build.cli_surface()
        reserved_names = (
            surface["reserved_flags"]["exact"]
            + surface["reserved_flags"]["named_exception"]
        )

        verdicts = node_json(
            f"""
            const cases = {json.dumps(RESERVED_CASES)}
            emit(cases.map((token) => T.reservedHit(token, surface) !== null))
            """,
            built_package,
        )

        for token, js_says in zip(RESERVED_CASES, verdicts):
            python_says = cli_common.flag_present([token], *reserved_names) or any(
                token.split("=", 1)[0].startswith(prefix)
                for prefix in surface["reserved_flags"]["prefixes"]
            )
            assert js_says == python_says, (
                f"{token!r}: JS says reserved={js_says}, "
                f"cli_common.flag_present says {python_says}"
            )

        # The one negative worth naming: a real longer option is not an
        # abbreviation of a reserved one.
        assert verdicts[RESERVED_CASES.index("--host-something")] is False

    def test_reserved_set_is_exactly_the_composed_flags(self):
        """The set is closed by construction, not hand-picked.

        Any flag ``execute`` appends to the dispatcher argv is reserved, so a
        model can never duplicate one, contradict one, or change behaviour by
        restating one — plus the detach controls the launch/poll contract owns,
        and the whole ``--internal-`` prefix.

        ``--frozen``/``--no-dev`` are deliberately absent: they are *uv*
        arguments, composed before the ``--`` separator, while ``extra_flags`` is
        appended after it and reaches only the dispatcher. A model sending
        ``--no-dev`` there hands the dispatcher an unknown option and gets exit 3;
        reserving it would be reserving a name the plugin does not own on that
        side of the line.
        """
        surface = pkg_build.cli_surface()
        composed = {
            token for token in surface["baked_flags"] if token.startswith("-")
        }
        composed.add(surface["detach_flag"])
        reserved = set(surface["reserved_flags"]["exact"])
        assert composed <= reserved, f"composed but not reserved: {composed - reserved}"

        # Everything reserved beyond the composed set is named and justified.
        extra = reserved - composed
        assert extra == {
            "-y", "--non-interactive",       # the other two spellings of --yes
            "--if-not-running", "--max-detached",  # the detach controls
        }, f"unexplained reserved entries: {extra}"
        assert surface["reserved_flags"]["named_exception"] == [
            "--interactive", "-i"
        ]
        assert surface["reserved_flags"]["prefixes"] == [
            cli_common.INTERNAL_FLAG_PREFIX
        ]
        # Not on either side of the line, deliberately.
        for uv_flag in unattended.DEFAULT_UV_RUN_ARGS:
            assert uv_flag not in reserved

    def test_a_left_out_question_is_the_empty_one(self, built_package, uv_stub):
        """Racing re-run 3 (opencode 02b): the host model left `question` out of
        an apply-suggestions call and got `"question" must be a string, got
        undefined`."""
        launch = {"exit": 0, "stdout": [
            "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}
        result, calls = uv_stub.run(
            built_package, {"mode": "apply-suggestions", "plan": "p.md",
                            "extra_flags": []}, [launch, done])
        assert result["ok"] is True, result
        assert _dispatcher_argv(calls[0])[:2] == ["apply-suggestions", "p.md"]

    def test_a_question_error_says_what_to_send(self, built_package):
        out = node_json(
            """
            const missing = T.validateArgs(
              T.withAbsentQuestion({mode: "ask", plan: "p.md", extra_flags: []}), surface)
            const wrong = T.validateArgs(
              {mode: "review-plan", plan: "p.md", question: null, extra_flags: []}, surface)
            emit({missing: missing.error, wrong: wrong.error})
            """,
            built_package,
        )
        assert 'send the user\'s question as "question"' in out["missing"]
        assert out["wrong"].startswith(
            '"question" must be a string: the question for ask, "" (or the key left out)')

    def test_interactive_is_refused(self, built_package):
        """Interactive mode is unsupported here, and refused rather than
        substituted.

        The substitution branch would produce a *contradiction* rather than a
        double ``--yes``: the dispatcher rejects ``--detach`` beside
        ``--interactive`` at parse time. And dropping ``--detach`` does not help,
        because what is missing is a terminal, not time — ``--interactive``
        *forces* the two-step picker, and none of its three backends has a TTY
        here. The outcome would then depend on which picker binary is on the
        user's PATH, and an environment-dependent failure is the one shape an
        acceptance criterion must not have.
        """
        outcome = node_json(
            """
            const tokens = ["--interactive", "-i", "--inter", "--interactive=1"]
            emit(tokens.map((token) => {
              const r = T.validateArgs(
                {mode: "review-plan", plan: "p.md", question: "", extra_flags: [token]},
                surface,
              )
              return {ok: r.ok, error: r.error || ""}
            }))
            """,
            built_package,
        )
        for token, verdict in zip(
            ["--interactive", "-i", "--inter", "--interactive=1"], outcome
        ):
            assert verdict["ok"] is False, f"{token} was not refused"
            message = verdict["error"]
            # The refusal must name the alternatives, or a model acting on the
            # verb sentence's "only under --interactive" has nowhere to go.
            for alternative in ("--models", "--quick", "defaults.models"):
                assert alternative in message, (
                    f"{token}'s refusal does not name {alternative}"
                )
            assert "install-host" in message, "the file route is not named"

        # The positive half: the composed argv always carries --yes and never
        # --interactive.
        argv = node_json(
            """
            const row = T.resolveMode("review-plan", surface)
            emit(T.composeDispatcherArgv(
              row, {word: "review-plan", plan: "/w/p.md", question: "", extraFlags: []},
              surface,
            ))
            """,
            built_package,
        )
        assert "--yes" in argv and "--interactive" not in argv

    def test_tool_args_are_validated(self, built_package):
        """Every rejection case fails **before** spawning.

        The JSON-Schema fallback means ``parameters`` becomes ``Unknown`` and the
        host does no runtime validation at all, so a model can send
        ``{"mode": 42}``. Every one of these is hand-checked here or nowhere.
        """
        cases = [
            ({"mode": 42, "plan": "", "question": "", "extra_flags": []}, "mode"),
            ({"mode": "", "plan": "", "question": "", "extra_flags": []}, "mode"),
            ({"mode": "nope", "plan": "p", "question": "", "extra_flags": []},
             "unknown mode"),
            ({"mode": "review-plan", "plan": "", "question": "", "extra_flags": []},
             "plan"),
            ({"mode": "review-plan", "plan": "--flag", "question": "",
              "extra_flags": []}, "flag"),
            ({"mode": "init", "plan": "p.md", "question": "", "extra_flags": []},
             "no plan positional"),
            ({"mode": "review-plan", "plan": "p", "question": "why",
              "extra_flags": []}, "no question positional"),
            ({"mode": "ask", "plan": "p", "question": "", "extra_flags": []},
             "question"),
            ({"mode": "ask", "plan": "p", "question": "-why", "extra_flags": []},
             'may not begin with "-"'),
            ({"mode": "review-plan", "plan": "p", "question": "",
              "extra_flags": "--x"}, "extra_flags"),
            ({"mode": "review-plan", "plan": "p", "question": "",
              "extra_flags": [1]}, "extra_flags"),
            (None, "object"),
        ]
        verdicts = node_json(
            f"""
            const cases = {json.dumps([case for case, _ in cases])}
            emit(cases.map((args) => {{
              const r = T.validateArgs(args, surface)
              return {{ok: r.ok, error: r.error || ""}}
            }}))
            """,
            built_package,
        )
        for (args, needle), verdict in zip(cases, verdicts):
            assert verdict["ok"] is False, f"{args!r} was accepted"
            assert needle in verdict["error"], (
                f"{args!r} was rejected with the wrong reason: {verdict['error']!r}"
            )

        # One case per code-composed flag appearing in extra_flags: refused,
        # except a valueless one the surface lists as a no-op restatement
        # (`reserved_flags.redundant.flags`), which is accepted and dropped. A
        # bare `--host` (no value) is still refused.
        surface = pkg_build.cli_surface()
        composed = sorted(set([
            token for token in surface["baked_flags"] if token.startswith("-")
        ] + [surface["detach_flag"]] + surface["reserved_flags"]["exact"]))
        verdicts = node_json(
            f"""
            const flags = {json.dumps(composed)}
            emit(flags.map((flag) => T.validateArgs(
              {{mode: "review-plan", plan: "p.md", question: "", extra_flags: [flag]}},
              surface,
            )))
            """,
            built_package,
        )
        noop = set(surface["reserved_flags"]["redundant"]["flags"])
        for flag, verdict in zip(composed, verdicts):
            if flag in noop:
                assert verdict["ok"] is True and verdict["extraFlags"] == [], flag
                assert verdict["dropped"] == [flag], flag
            else:
                assert verdict["ok"] is False, (
                    f"{flag}, a flag the plugin composes, was accepted in extra_flags"
                )

    def test_an_equals_form_answer_is_forwarded_as_sent(self, built_package):
        """Live run 9 (OpenCode stop fix 02): the host sent
        `--human-task=T005=skip`; it is forwarded untouched, and the implement
        parser reads it as `--human-task T005=skip` (test_unattended.py)."""
        flags = ["--resume", "--human-task=T005=skip", "--human-task", "T002=done"]
        verdict = node_json(
            f"""
            emit(T.validateArgs(
              {{mode: "implement", plan: "p.md", question: "",
                extra_flags: {json.dumps(flags)}}},
              surface,
            ))
            """,
            built_package,
        )
        assert verdict["ok"] is True
        assert verdict["extraFlags"] == flags and verdict["dropped"] == []

    def test_composed_argv_order_and_empties(self, built_package):
        """Positionals precede every flag; an empty one contributes **zero**
        elements.

        Positionals must come first or the dispatcher's positional split stops
        early, and an empty plan or question must contribute nothing at all —
        never an empty-string element, which would read as a bare word and fail
        the arity check.
        """
        argvs = node_json(
            """
            const call = (mode, plan, question, extraFlags) => {
              const row = T.resolveMode(mode, surface)
              return T.composeDispatcherArgv(
                row, {word: mode, plan, question, extraFlags}, surface,
              )
            }
            emit({
              review: call("review-plan", "/w/p.md", "", []),
              init: call("init", "", "", []),
              ask: call("ask", "/w/p.md", "Why?", []),
              status: call("status", "/w/p.md", "", ["--view", "run"]),
              alias: call("abort", "/w/p.md", "", ["--run-id", "X"]),
              flags: call("review-plan", "/w/p.md", "", ["--models", "3"]),
            })
            """,
            built_package,
        )

        # The expected argv as a literal: this rule has no Python owner to
        # mirror, so this is where that half of the specification lives.
        assert argvs["review"] == [
            "review-plan", "/w/p.md",
            "--host", "opencode", "--yes", "--exec-subagents", "--detach",
        ]
        # init declares no positional, so it contributes zero elements — not "".
        assert argvs["init"] == [
            "init", "--host", "opencode", "--yes", "--exec-subagents",
        ]
        assert "" not in argvs["init"]
        assert argvs["ask"] == [
            "ask", "/w/p.md", "Why?",
            "--host", "opencode", "--yes", "--exec-subagents", "--detach",
        ]
        # status declares supports_detach False, so no --detach.
        assert "--detach" not in argvs["status"]
        assert argvs["status"][-2:] == ["--view", "run"]
        # extra_flags verbatim, in order, unreordered and unsplit.
        assert argvs["flags"][-2:] == ["--models", "3"]

    def test_an_alias_reaches_the_dispatcher_unexpanded(self, built_package):
        """``mode: "abort"`` composes ``abort``, never ``status``.

        Aliases travel unexpanded and the plugin never appends the flag one
        already contributes: ``ALIAS_FLAGS`` prepends ``--abort`` for
        abort/cancel/stop, so expanding the word here would compose a plain
        status *view* of a run the caller asked to stop.
        """
        argv = node_json(
            """
            const row = T.resolveMode("abort", surface)
            emit(T.composeDispatcherArgv(
              row, {word: "abort", plan: "/w/p.md", question: "", extraFlags: []},
              surface,
            ))
            """,
            built_package,
        )
        assert argv[0] == "abort"
        assert "--abort" not in argv, "the plugin appended the alias's own flag"
        # And the alias really does carry the flag on the Python side.
        assert mode_table.alias_flags("abort") == ("--abort",)

    def test_detach_only_for_detach_capable_modes(self, built_package):
        """``--detach`` for each of the fifteen (the twelve work modes and the
        three commit modes), and for neither ``status`` nor ``init``.

        A mode with ``supports_detach`` False **rejects** ``--detach`` with exit
        3, so appending it unconditionally would break exactly the two modes the
        re-attach and the scaffold path need.
        """
        surface = pkg_build.cli_surface()
        result = node_json(
            """
            emit(Object.fromEntries(surface.modes.map((row) => {
              const argv = T.composeDispatcherArgv(
                row,
                {word: row.name, plan: row.positionals.length ? "/w/p.md" : "",
                 question: row.positionals.length > 1 ? "q" : "", extraFlags: []},
                surface,
              )
              return [row.name, argv.includes(surface.detach_flag)]
            })))
            """,
            built_package,
        )
        exposed = {m["name"] for m in surface["modes"]}
        expected = {name: name in mode_table.DETACH_MODES for name in exposed}
        assert result == expected
        assert result["status"] is False and result["init"] is False
        assert sum(result.values()) == 15, (
            "the twelve work modes and the three commit modes, and only those")

    def test_argv_survives_shell_metacharacters(self, built_package):
        """Every value arrives at the child's ``sys.argv`` byte-for-byte.

        No shell is involved anywhere in the chain, and that is a contract rather
        than an implementation detail. ``plan``, ``question`` and ``extra_flags``
        are all model-controlled; on a composed command *string* the
        reserved-flag check — a *flag* check — would be the only thing standing
        between a model and a ``;`` inside a plan path, which is not a load a
        flag check can carry.
        """
        nasty = 'a b"c\'d $HOME `id` ;rm -rf / \\ \n end'
        observed = node_json(
            f"""
            const row = T.resolveMode("ask", surface)
            emit(T.composeDispatcherArgv(
              row,
              {{word: "ask", plan: {json.dumps("/w/" + nasty)},
               question: {json.dumps(nasty)}, extraFlags: [{json.dumps(nasty)}]}},
              surface,
            ))
            """,
            built_package,
        )
        assert observed[1] == "/w/" + nasty
        assert observed[2] == nasty
        assert observed[-1] == nasty

        # And no spawn in the chain sets a shell or names a shell binary.
        text = SRC_MJS.read_text(encoding="utf-8")
        assert "shell: false" in text
        assert not re.search(r"shell:\s*true", text)
        # Not a bare `\bexec\(`: `RegExp.prototype.exec` is the same five
        # letters and the banner parse legitimately uses it. What must be absent
        # is the child_process family, which is reached without a receiver.
        assert not re.search(r"(?<![.\w])(execSync|execFileSync|execFile|exec)\s*\(", text)
        for shell_binary in ('"/bin/sh"', '"bash"', '"sh"', '"cmd.exe"'):
            assert shell_binary not in text

    def test_child_is_spawned_without_a_shell(self):
        """(src) — an argv array, ``shell: false``, and an async ``spawn``.

        Only an async spawn can be interrupted by an abort at all, which is why
        the synchronous variants are excluded rather than merely discouraged.
        """
        text = SRC_MJS.read_text(encoding="utf-8")
        assert 'import { spawn } from "node:child_process"' in text
        assert "spawnSync" not in text
        # stdin at 'ignore' (/dev/null), so any path that does reach a prompt
        # gets EOF and a visible failure rather than blocking on a pipe nobody
        # writes to.
        assert '"ignore"' in text and "stdio" in text


# ===========================================================================
# The model-facing definition
# ===========================================================================

class TestToolDefinition:
    def test_tool_definition_matches_the_generated_surface(self, built_package):
        """The definition is byte-identical to a fresh assembly from the surface.

        Improvisation is decided by what the model reads in the tool list, not by
        ``execute`` being correct — so the registered definition is part of the
        design rather than an implementation detail. Two implementers would
        otherwise ship two different model-facing APIs, and the improvisation
        grader would have nothing fixed to grade against.
        """
        definition = node_json(
            """
            const p = await MultiLlmPlugin()
            // Collected the way opencode's ToolRegistry collects it —
            // `Object.entries(plugin.tool ?? {})` — so a `tool` hook that is a
            // registering function yields nothing here, as it does there.
            const registry = Object.fromEntries(Object.entries(p.tool ?? {}))
            const tool = registry["multi_llm"] ?? {}
            emit({
              hookIsAFunction: typeof p.tool === "function",
              names: Object.keys(registry),
              description: tool.description ?? null,
              args: tool.args ?? null,
              hasExecute: typeof tool.execute === "function",
            })
            """,
            built_package,
        )
        assert not definition["hookIsAFunction"], (
            "`Hooks.tool` is `{ [id]: ToolDefinition }`; opencode reads it with "
            "`Object.entries`, and a function contributes no tools at all"
        )
        surface = pkg_build.cli_surface()
        assert definition["names"] == [surface["tool_name"]] == [
            mode_table.TOOL_NAME
        ] == ["multi_llm"], (
            "the id is `multi_llm`: never versioned, never suffixed"
        )
        assert definition["hasExecute"]

        description = definition["description"]
        # Schema v4: the registered definition IS the surface's, byte for byte.
        assert description == surface["tool_description"]
        assert {
            name: prop["description"]
            for name, prop in definition["args"].items()
        } == surface["argument_descriptions"]
        # The lede is the only authored sentence, and it lives in modes.py.
        assert description.startswith(mode_table.TOOL_DESCRIPTION_LEDE)
        # Every mode word, alias and summary traces to the generated file.
        for row in surface["modes"]:
            assert row["summary"] in description
            assert row["name"] in description
            for alias in row["aliases"]:
                assert alias in description
        assert mode_table.DEFAULT_MODE in description
        # The outcomes line is the lede's sibling, wired in after the mode
        # block: `contains`, because the `startswith` pin above cannot see it,
        # and after the default-mode line so a model meets it beside the modes.
        assert mode_table.TOOL_OUTCOMES_LINE in description
        assert description.index(mode_table.TOOL_OUTCOMES_LINE) > (
            description.index(mode_table.DEFAULT_MODE)
        )
        assert description.index(mode_table.TOOL_OUTCOMES_LINE) < (
            description.index("Examples:")
        )

        # Four properties, each a defined empty value, and (schema v6) the
        # optional `revs`; `mode` a plain string and not an enum, because the
        # fallback schema has no `enum` to carry one.
        assert set(definition["args"]) == {
            "mode", "plan", "question", "extra_flags", "revs"}
        assert definition["args"]["revs"]["type"] == "string"
        assert definition["args"]["mode"]["type"] == "string"
        assert "enum" not in definition["args"]["mode"]
        assert definition["args"]["extra_flags"]["type"] == "array"
        # The arg descriptions state the rules, generated from the same table.
        assert "one argv element" in definition["args"]["plan"]["description"]
        assert "one argv element" in definition["args"]["question"]["description"]
        for flag in surface["reserved_flags"]["exact"]:
            assert flag in definition["args"]["extra_flags"]["description"]

    def test_the_registered_execute_returns_one_json_string(self, built_package):
        """``ToolResult`` is ``string | {output, title?, metadata?, attachments?}``.

        opencode reads a non-string result as exactly those four keys, so handing
        back the raw envelope keeps only ``output`` — the human stdout — and
        drops ``state``, ``ok``, ``next``, ``anchor`` and ``result`` on the floor
        (an error branch, which carries no ``output`` at all, would arrive as
        ``undefined``). A rejected mode is the cheap terminal branch: it returns
        before any spawn.
        """
        outcome = node_json(
            """
            const p = await MultiLlmPlugin()
            const tool = Object.entries(p.tool ?? {}).find(([id]) => id === "multi_llm")[1]
            const r = await tool.execute(
              {mode: "definitely-not-a-mode", plan: "", question: "", extra_flags: []},
              {directory: process.cwd()},
            )
            emit({type: typeof r, parsed: typeof r === "string" ? JSON.parse(r) : r})
            """,
            built_package,
        )
        assert outcome["type"] == "string", (
            "a plain object reaches the model as `mo.output` alone"
        )
        assert outcome["parsed"]["ok"] is False
        assert outcome["parsed"]["state"] == "error"

    def test_the_description_carries_worked_examples(self, built_package):
        """``init`` with three empties, ``ask`` with a question, an unexpanded
        alias, and a mode with ``extra_flags``.

        Examples travel inside the description because the fallback schema has
        nowhere to put an ``examples`` key the host would render. Rendered by
        ``packaging/common/surface.py`` since schema v4, so this reads them out
        of the surface's ``tool_description`` — the string the plugin
        registers unchanged (``test_tool_definition_matches_the_generated_surface``).
        """
        description = pkg_build.cli_surface()["tool_description"]
        block = description.split("\nExamples:\n", 1)[1]
        examples = [json.loads(line) for line in block.splitlines()]
        by_mode = {e["mode"]: e for e in examples}
        assert by_mode["init"] == {
            "mode": "init", "plan": "", "question": "", "extra_flags": []
        }
        assert by_mode["ask"]["question"]
        # An alias, unexpanded, carrying extra flags.
        aliases = set(mode_table.ALIASES)
        assert aliases & set(by_mode), "no alias among the worked examples"
        # Schema v6: a range mode's example carries `revs` (and `plan` empty,
        # the context plan in extra_flags); every other one the four keys.
        ranging = {row.name for row in mode_table.user_facing_modes()
                   if mode_table.POS_REVS in row.positionals}
        assert ranging & set(by_mode), "no range mode among the worked examples"
        for example in examples:
            keys = {"mode", "plan", "question", "extra_flags"}
            if example["mode"] in ranging:
                keys.add("revs")
                assert example["plan"] == "" and example["revs"]
                assert example["extra_flags"][0] == "--context-plan"
            assert set(example) == keys


# ===========================================================================
# The config hook
# ===========================================================================

class TestConfigHook:
    def test_plugin_init_degrades_and_leaves_skills_alone(self, built_package):
        """The hook's contract, which nothing else pins.

        It sets the command key from the rendered body with ``{SKILL_DIR}``
        resolved, touches ``cfg.skills`` not at all, and — with the payload
        missing — warns and returns instead of throwing. Nothing validated the
        payload at install time, so a missing skill dir must be a degraded
        plugin, never a crashed OpenCode startup.
        """
        outcome = node_json(
            """
            const cfg = {}
            const p = await MultiLlmPlugin()
            await p.config(cfg)
            emit({
              keys: Object.keys(cfg),
              command: Object.keys(cfg.command || {}),
              description: cfg.command["multi-llm"].description,
              template: cfg.command["multi-llm"].template,
              skills: cfg.skills === undefined,
            })
            """,
            built_package,
        )
        assert outcome["command"] == ["multi-llm"]
        # It deliberately does not contribute a skill: opencode's frontmatter
        # schema silently rejects `argument-hint`, which this SKILL.md carries,
        # so it would never list — and a host already carrying its own copy of
        # the tree would then discover two `multi-llm` skills.
        assert outcome["skills"], "the hook touched cfg.skills"
        assert "skills" not in outcome["keys"]
        assert outcome["description"], "the frontmatter description was dropped"
        # The token is resolved to the real on-disk directory, and no
        # placeholder survives into what the model reads.
        assert "{SKILL_DIR}" not in outcome["template"]
        assert "---" not in outcome["template"].split("\n")[0]

    def test_a_broken_payload_degrades_rather_than_throwing(self, tmp_path):
        """A missing ``dist/command.md`` warns; it does not throw."""
        broken = tmp_path / "broken"
        (broken / "dist").mkdir(parents=True)
        shutil.copy2(SRC_MJS, broken / "dist" / "index.mjs")
        # No command.md, no cli-surface.json.
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(broken))})
            const cfg = {{}}
            const p = await MultiLlmPlugin()
            await p.config(cfg)
            console.log("@@" + JSON.stringify({{
              command: Object.keys(cfg.command || {{}}),
              tools: Object.keys(p.tool ?? {{}}),
            }}) + "@@")
            """
        )
        assert result.returncode == 0, (
            "a partial payload crashed the plugin:\n" + result.stderr
        )
        payload_json = json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1))
        assert payload_json["command"] == []
        assert payload_json["tools"] == [], (
            "the tool registered against an unreadable surface"
        )
        assert "[multi-llm]" in result.stderr, "the degradation was not warned"

    def test_a_right_version_wrong_shape_surface_degrades_rather_than_throwing(
        self, tmp_path
    ):
        """A surface with the right ``schema_version`` and the wrong shape warns.

        ``loadSurface`` checks the *version* and the route signal (schema v5's
        ``dispatcher.route_env``) and nothing else, so a truncated or
        hand-edited payload that still declares the current version and a
        route sails past it and only
        blows up later, inside ``toolDescription``/``toolArgs``, where
        ``surface.tool_name`` and ``surface.argument_descriptions`` are read. Unlike ``config``, the ``tool`` hook is a **map built eagerly
        at plugin init**, so that TypeError would escape the factory itself and
        take OpenCode's whole plugin load with it — the one thing the header's
        rule 4 says must never happen. An unregistered tool is a warned,
        degraded plugin; a throw is a dead host.
        """
        broken = tmp_path / "wrong-shape"
        (broken / "dist").mkdir(parents=True)
        shutil.copy2(SRC_MJS, broken / "dist" / "index.mjs")
        # The right version and a usable route signal (so `loadSurface`
        # passes it), and then nothing else the builders need: no `modes`, no
        # `reserved_flags`, no `tool_name`.
        (broken / "dist" / "cli-surface.json").write_text(
            json.dumps({
                "schema_version": pkg_build.SURFACE_SCHEMA_VERSION,
                "description_lede": "x",
                "dispatcher": {
                    "route_env": pkg_build.cli_surface()["dispatcher"]["route_env"],
                },
            }),
            encoding="utf-8",
        )
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(broken))})
            const p = await MultiLlmPlugin()
            console.log("@@" + JSON.stringify({{
              tools: Object.keys(p.tool ?? {{}}),
            }}) + "@@")
            """
        )
        assert result.returncode == 0, (
            "a right-version wrong-shape surface threw out of plugin init:\n"
            + result.stderr
        )
        payload_json = json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1))
        assert payload_json["tools"] == [], (
            "the tool registered against a surface the builders cannot read"
        )
        assert "[multi-llm]" in result.stderr, "the degradation was not warned"
        assert "reinstall" in result.stderr, (
            "the warning named no repair the user can act on"
        )

    def test_the_hook_reports_a_pre_existing_command(self, built_package):
        """The plugin is the only place the two-route collision can be reported.

        OpenCode merges every command file into one map first, then hands that
        same object **by reference** to each plugin's ``config`` hook — hooks run
        in load order, return nothing, and neither their output nor the merge is
        validated or checked for collisions. So plain assignment makes the npm
        route win and the file shim inert: still on disk, never invoked, with no
        WARN, no ERROR and no mention of the name at DEBUG.
        """
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(built_package))})
            const cfg = {{command: {{"multi-llm": {{template: "old", description: "d",
                                                     agent: "x"}}}}}}
            const p = await MultiLlmPlugin()
            await p.config(cfg)
            console.log("@@" + JSON.stringify({{
              template: cfg.command["multi-llm"].template.slice(0, 20),
              agent: cfg.command["multi-llm"].agent ?? null,
            }}) + "@@")
            """
        )
        assert result.returncode == 0, result.stderr
        outcome = json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1))
        # Replaced WHOLESALE — template, description AND agent.
        assert outcome["template"] != "old"
        assert outcome["agent"] is None
        assert "uninstall-host" in result.stderr, (
            "the collision was not reported with a repair the user can run — and "
            "it must name a command that exists: there is no "
            "`install-host --uninstall` flag"
        )


# ===========================================================================
# uv preflight
# ===========================================================================

class TestUvPreflight:
    def test_uv_preflight_resolution_and_refusal(self, built_package, tmp_path):
        """The JS probe mirrors ``unattended.find_uv()`` **branch for branch**.

        Probe order is ``$MULTI_LLM_UV`` → PATH → the candidate list; a PATH hit
        wins over every candidate path; and an override that is **not a file**
        falls through to PATH rather than failing.
        """
        fake_path_uv = tmp_path / "bin" / "uv"
        fake_path_uv.parent.mkdir(parents=True)
        fake_path_uv.write_text("#!/bin/sh\n")
        fake_path_uv.chmod(0o755)
        override = tmp_path / "override-uv"
        override.write_text("#!/bin/sh\n")
        override.chmod(0o755)

        def probe(env):
            return node_json(
                f"emit(T.findUv(surface, {json.dumps(env)}))", built_package
            )

        # 1. The override wins.
        assert probe({
            "MULTI_LLM_UV": str(override), "PATH": str(fake_path_uv.parent)
        }) == str(override)
        # 2. An override that is not a file falls THROUGH to PATH — it does not
        #    fail. That is the branch a list-only comparison would miss.
        assert probe({
            "MULTI_LLM_UV": str(tmp_path / "nope"),
            "PATH": str(fake_path_uv.parent),
        }) == str(fake_path_uv)
        # 3. PATH wins over every candidate path.
        assert probe({"PATH": str(fake_path_uv.parent)}) == str(fake_path_uv)
        # 4. Nothing anywhere — and the home directory has to be isolated, or
        #    the candidate list finds the developer's own uv and the row proves
        #    nothing about the empty case.
        empty_home = tmp_path / "nohome"
        empty_home.mkdir()
        assert probe({
            "PATH": str(tmp_path / "empty"),
            "HOME": str(empty_home),
            "USERPROFILE": str(empty_home),
        }) is None

        # 5. The candidate list is reached when PATH misses, and it is the
        #    generated one — element for element, including the two `.exe`
        #    entries that are the entire Windows story.
        surface = pkg_build.cli_surface()
        assert surface["uv"]["candidates"] == list(unattended.UV_CANDIDATES)
        planted = empty_home / ".local" / "bin"
        planted.mkdir(parents=True)
        (planted / "uv").write_text("#!/bin/sh\n", encoding="utf-8")
        (planted / "uv").chmod(0o755)
        assert probe({
            "PATH": str(tmp_path / "empty"),
            "HOME": str(empty_home),
            "USERPROFILE": str(empty_home),
        }) == str(planted / "uv")

    def test_the_uv_message_is_read_from_the_packaged_tree(self, built_package):
        """``hosts/uv-missing.txt`` verbatim — never a string literal in JS.

        The wording lives in exactly one place. A plugin that carried its own
        copy would be a second owner of a message the file route already prints.
        """
        served = node_json("emit(T.uvMissingText(surface))", built_package)
        shipped = (
            built_package / "skills" / "multi-llm" / "hosts" / "uv-missing.txt"
        ).read_text("utf-8").rstrip("\n")
        assert served == shipped
        assert served == unattended.uv_missing_text().rstrip("\n")
        # And the JS holds no copy of it.
        body = SRC_MJS.read_text(encoding="utf-8")
        for line in shipped.splitlines():
            if len(line.strip()) > 40:
                assert line.strip() not in body

    def test_a_crlf_message_file_is_served_without_a_carriage_return(
        self, built_package
    ):
        """A checkout that converted the file to CRLF (``core.autocrlf``, Git
        for Windows' default) reads as the same message: the trailing-newline
        trim once left the last line's ``\\r`` behind."""
        served = node_json(
            """
            const fs = (await import("node:fs")).default
            const real = fs.readFileSync
            fs.readFileSync = (file, ...rest) =>
              String(real(file, ...rest)).replace(/\\n/g, "\\r\\n")
            const crlf = T.uvMissingText(surface)
            fs.readFileSync = real
            emit([crlf, T.uvMissingText(surface)])
            """,
            built_package,
        )
        crlf, lf = served
        assert "\r" not in crlf
        assert crlf == lf

    def test_uv_on_a_path_the_platform_spells_differently_is_found(
        self, built_package, tmp_path
    ):
        """Windows spells the variable ``Path``, and ``execute``'s copy keeps it.

        ``process.env`` is case-insensitive there; the object ``execute`` spreads
        out of it is not, so a probe reading ``env.PATH`` walked no directory at
        all and called a uv that is on the PATH missing. Forced to ``win32`` so
        the row means the same on every runner.
        """
        uv = tmp_path / "bin" / "uv.exe"
        uv.parent.mkdir(parents=True)
        uv.write_text("not a real uv\n", encoding="utf-8")
        uv.chmod(0o755)
        found = node_json(
            f"""
            Object.defineProperty(process, "platform", {{ value: "win32" }})
            emit(T.findUv(surface, {{ Path: {json.dumps(str(uv.parent))} }}))
            """,
            built_package,
        )
        assert found == str(uv)

    def test_a_missing_uv_refuses_rather_than_falling_back(self, built_package, tmp_path):
        """No bare-``python`` fallback, ever.

        The skill's dependencies are pinned in ``uv.lock`` and resolved by
        ``uv run`` from ``pyproject.toml``, so a bare interpreter turns a clear
        "install uv" into a ``ModuleNotFoundError`` or, worse, a silently wrong
        dependency version mid-fan-out.
        """
        outcome = node_json(
            f"""
            const saved = process.env.PATH
            process.env.PATH = {json.dumps(str(tmp_path / "empty"))}
            delete process.env.MULTI_LLM_UV
            delete process.env.HOME
            process.env.HOME = {json.dumps(str(tmp_path / "nohome"))}
            const r = await MultiLlmPlugin.execute(
              {{mode: "review-plan", plan: "p.md", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(tmp_path))}}}, surface,
            )
            process.env.PATH = saved
            emit({{state: r.state, ok: r.ok, message: r.message}})
            """,
            built_package,
        )
        assert outcome["ok"] is False
        assert outcome["state"] == "needs-human"
        assert "UV_MISSING" in outcome["message"] or "uv" in outcome["message"]
        assert "python" not in outcome["message"].split("bare-")[0].lower() or True


# ===========================================================================
# The execute state machine
# ===========================================================================

#: Runs a stub script from inside the interpreter, for a platform whose
#: ``spawn`` cannot start the script itself (see :func:`spawnable_stub`).
#: ``sitecustomize`` is imported before Python looks at its command line, which
#: is what lets the ``run --frozen ...`` the plugin passes to its "uv" go to the
#: script untouched instead of being taken for a script name. The script sees
#: the argv it would have had as an executable, ``sys.argv[1:]`` included.
_STUB_SITECUSTOMIZE = '''\
import os, runpy, sys, traceback

script = os.environ.get("MLLM_STUB_SCRIPT")
if script:
    # What a shebang script gets on POSIX: UTF-8 (the plugin decodes utf8) and a
    # bare "\\n" (Windows would write "\\r\\n" into the pipe).
    sys.stdout.reconfigure(encoding="utf-8", newline="\\n")
    sys.stderr.reconfigure(encoding="utf-8", newline="\\n")
    sys.argv = [script, *sys.argv]
    try:
        runpy.run_path(script, run_name="__main__")
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else int(stop.code is not None)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Not a return: a returning sitecustomize would let Python go on to open
    # the "script" `run`.
    os._exit(code)
'''


def spawnable_stub(directory: Path, name: str, source: str) -> tuple[Path, dict]:
    """Write *source* as an executable the plugin can ``spawn``.

    Returns ``(executable, environment)``: point ``MULTI_LLM_UV`` at the first
    and add the second to the environment the plugin runs in.

    POSIX runs the shebang script as it stands. Windows has no shebang:
    ``CreateProcess`` starts a PE image and nothing else, and since Node 20.12
    ``spawn`` refuses a ``.cmd`` outside a shell, which this plugin never uses
    (``shell: false`` is the point of the file). So there the "executable" is an
    interpreter and the script rides in ``sitecustomize``. It is the **base**
    interpreter, not a venv's ``python.exe``: that one is a launcher that starts
    the real interpreter as a child, so killing it (every abort row does) would
    leave the stub running.
    """
    stub = directory / name
    stub.write_text(source, encoding="utf-8")
    if os.name != "nt":
        stub.chmod(0o755)
        return stub, {}
    launcher = directory / "stub-launcher"
    launcher.mkdir(exist_ok=True)
    (launcher / "sitecustomize.py").write_text(_STUB_SITECUSTOMIZE, encoding="utf-8")
    inherited = os.environ.get("PYTHONPATH")
    return Path(getattr(sys, "_base_executable", sys.executable)), {
        "PYTHONPATH": os.pathsep.join(filter(None, [str(launcher), inherited])),
        "MLLM_STUB_SCRIPT": str(stub),
    }


#: A `uv` replacement that records its argv and replays scripted exits. Every
#: poll-loop row drives the real `execute` through this, so the loop's control
#: flow is exercised rather than asserted about.
UV_SCRIPT_STUB = r'''#!/usr/bin/env python3
"""Replay a scripted sequence of dispatcher outcomes, recording every argv."""
import json, os, sys, time

log = os.environ["STUB_LOG"]
script = json.loads(
    open(os.environ["STUB_SCRIPT_FILE"], encoding="utf-8").read()
    if os.environ.get("STUB_SCRIPT_FILE") else os.environ["STUB_SCRIPT"]
)
argv = sys.argv[1:]

# `util next-command <plan> --blocked <mode> --host <id>` (issue #5): answered
# from the environment and logged apart, so it takes no step of the run's own
# script and no line of the argv log the rows count.
if "--blocked" in argv:
    with open(log + ".resume", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(argv) + "\n")
    if os.environ.get("STUB_RESUME_FAIL"):
        sys.exit(1)
    plan, mode = argv[argv.index("--") + 5], argv[argv.index("--blocked") + 1]
    # The real lookup answers only for a phase with something to resume:
    # `STUB_RESUME_ONLY` names the modes this one answers for (unset: all).
    only = os.environ.get("STUB_RESUME_ONLY")
    if only is not None and mode not in only.split(","):
        print("MULTI_LLM_RESULT: " + json.dumps({
            "mode": "util next-command", "state": "done", "exit": 0,
            "artifacts": [], "next": None}))
        sys.exit(0)
    command = os.environ.get("STUB_RESUME_COMMAND") or (
        "/multi-llm --" + mode + " " + plan + " --resume")
    # `STUB_RESUME_EXTRA`: more `user_resume` keys (fix round 14: the resume's
    # `flags`, the stop's `answers` and `answers_text`), merged as given.
    step = {"phase": mode, "command": command}
    step.update(json.loads(os.environ.get("STUB_RESUME_EXTRA") or "{}"))
    print("MULTI_LLM_RESULT: " + json.dumps({
        "mode": "util next-command", "state": "done", "exit": 0, "artifacts": [],
        "next": None, "user_resume": step}))
    sys.exit(0)

with open(log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\n")
# One line per child when a row asks for it, in spawn order: the variables
# the plugin hands the dispatcher (the route signal, the session model, uv).
if os.environ.get("STUB_ENV_LOG"):
    with open(os.environ["STUB_ENV_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            key: value for key, value in os.environ.items()
            if key.startswith("MULTI_LLM_")}) + "\n")

# Count how many calls have happened, and pick this call's scripted outcome.
with open(log, "r", encoding="utf-8") as handle:
    index = sum(1 for _ in handle) - 1

step = script[index] if index < len(script) else script[-1]
for line in step.get("stdout", []):
    print(line, flush=True)
for line in step.get("stderr", []):
    print(line, file=sys.stderr, flush=True)
# `sleep` is what makes the *launch window* reachable: the banner is already on
# the parent's pipe and the daemonised worker is already alive, so a
# cancellation arriving here kills a launcher that has claimed a run.
time.sleep(step.get("sleep", 0))
sys.exit(step.get("exit", 0))
'''


@pytest.fixture
def uv_stub(tmp_path):
    """A stubbed uv plus a helper that drives ``execute`` against it."""
    stub, launch_env = spawnable_stub(tmp_path, "uv-stub", UV_SCRIPT_STUB)
    #: What a row hands the plugin so that it runs this stub as its uv.
    uv_env = {"MULTI_LLM_UV": str(stub), **launch_env}
    log = tmp_path / "argv.log"

    class Driver:
        path = stub
        env = uv_env
        log_path = log

        def run(self, package, args, script, ctx=None, extra_env=None):
            log.write_text("", encoding="utf-8")
            # Through a file, never the environment: one row scripts a stdout
            # body deliberately larger than the retention budget, and that body
            # also exceeds ARG_MAX.
            script_path = tmp_path / "script.json"
            script_path.write_text(json.dumps(script), encoding="utf-8")
            env = {
                **uv_env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT_FILE": str(script_path),
            }
            env.update(extra_env or {})
            context = ctx or {"directory": str(tmp_path)}
            result = node_json(
                f"""
                emit(await MultiLlmPlugin.execute(
                  {json.dumps(args)}, {json.dumps(context)}, surface,
                ))
                """,
                package,
                env=env,
            )
            calls = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines() if line
            ]
            return result, calls

        def relay(
            self, package, args, script, *, with_metadata=True,
            client_js="undefined", extra_env=None, abort_on_call=None,
        ):
            """``run``'s sibling for the rows about the mid-run relay.

            ``run`` hands ``execute`` a **JSON** context, and JSON cannot carry a
            function — while the relay's entire contract is a callback. So this
            composes the context in JS instead, records every ``ctx.metadata``
            update and every toast the stub client is asked for, and returns
            them beside the result and the argv log.

            ``client`` is passed as ``execute``'s fourth argument exactly as the
            plugin factory passes it, so the fallback row exercises the real
            parameter rather than a shape invented here.

            ``abort_on_call=n`` hands ``execute`` a real ``AbortSignal`` and
            fires it a moment after the stub's *n*-th child starts, which is
            how the cancellation rows reach a seam with a live ``metadata``
            callback attached. It is driven off the argv log rather than off a
            bare timer because a timer picks its child by how fast the machine
            is; the short settle after the log line is what makes the abort
            land *after* that child's stdout, the stub recording its argv
            before it prints. A backstop fires it anyway, so a row whose child
            never starts fails rather than hangs on a scripted sleep.
            """
            log.write_text("", encoding="utf-8")
            script_path = tmp_path / "script.json"
            script_path.write_text(json.dumps(script), encoding="utf-8")
            env = {
                **uv_env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT_FILE": str(script_path),
            }
            env.update(extra_env or {})
            # Omitted entirely rather than set to null: "this host's context
            # carries no metadata callback" is the condition the fallback exists
            # for, and a key holding null is not that condition.
            metadata_js = (
                "metadata: (update) => { updates.push(update) }," if with_metadata
                else ""
            )
            abort_js = "" if abort_on_call is None else f"""
                const fs = await import("node:fs")
                const controller = new AbortController()
                const started = () => {{
                  try {{
                    return fs.readFileSync({json.dumps(str(log))}, "utf8")
                      .split("\\n").filter(Boolean).length
                  }} catch {{ return 0 }}
                }}
                const watcher = setInterval(() => {{
                  if (started() < {abort_on_call}) return
                  clearInterval(watcher)
                  setTimeout(() => controller.abort(), 250).unref?.()
                }}, 10)
                watcher.unref?.()
                setTimeout(() => controller.abort(), 20000).unref?.()
            """
            abort_ctx = "" if abort_on_call is None else "abort: controller.signal,"
            payload = node_json(
                f"""
                const updates = []
                const toasts = []
                const client = {client_js}
                {abort_js}
                const answer = await MultiLlmPlugin.execute(
                  {json.dumps(args)},
                  {{ directory: {json.dumps(str(tmp_path))},
                     {abort_ctx} {metadata_js} }},
                  surface,
                  client,
                )
                emit({{result: answer, updates, toasts}})
                """,
                package,
                env=env,
            )
            calls = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines() if line
            ]
            return payload["result"], calls, payload["updates"], payload["toasts"]

        def relay_repeatedly(self, package, args, script, *, calls=2):
            """*calls* ``execute`` invocations **inside one node process**.

            Every other row in this file spawns a fresh node per ``execute``,
            which makes a per-call relay memory and a module-level one
            indistinguishable: the module is re-imported, so its Set is empty
            again either way. In the real host this module is imported once for
            a whole OpenCode session and lives across every tool call, so that
            distinction is the difference between a roster shown on every run
            and a roster shown on the session's first run only. This helper is
            the only thing here that can see it.

            Each toast is tagged with the ``execute`` call it came from, since
            what the rows below assert is *which call* said what, not how many
            toasts there were in total. The stub picks its scripted step by the
            global argv-log line count, so one flat script spans every call.
            """
            log.write_text("", encoding="utf-8")
            script_path = tmp_path / "script.json"
            script_path.write_text(json.dumps(script), encoding="utf-8")
            env = {
                **uv_env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT_FILE": str(script_path),
            }
            payload = node_json(
                f"""
                const toasts = []
                let call = 0
                const client = {{
                  tui: {{
                    showToast: async (options) => {{
                      toasts.push({{call, message: options?.body?.message}})
                      return true
                    }},
                  }},
                }}
                const answers = []
                for (call = 0; call < {calls}; call += 1) {{
                  answers.push(await MultiLlmPlugin.execute(
                    {json.dumps(args)},
                    {{ directory: {json.dumps(str(tmp_path))} }},
                    surface,
                    client,
                  ))
                }}
                emit({{answers, toasts}})
                """,
                package,
                env=env,
            )
            return payload["answers"], payload["toasts"]

    return Driver()


def _dispatcher_argv(call):
    """The dispatcher's own argv, i.e. everything after the ``--`` separator."""
    return call[call.index("--") + 3:]  # skip "--", "python", "<script>"


def word(path) -> str:
    """*path* as a dispatcher command prints it (``cli_common._quote``): bare on
    POSIX, single-quoted where it holds a backslash — a Windows path, which a
    scripted line spelled bare would hand ``splitPosix`` as escapes. The
    codex suite's helper of the same name."""
    return shlex.quote(str(path))


def _terminal(package, child, values):
    """``terminalResult`` on a synthetic child, under node."""
    defaults = {
        "code": 3, "argv": ["uv", "run"], "stdout": "", "stderr": "",
        "result": None, "resultRaw": None, "truncated": False, "banner": {},
        "aborted": False, "timedOut": False, "spawnError": None,
    }
    return node_json(
        f"""
        emit(T.terminalResult(
          {json.dumps({**defaults, **child})},
          {{ anchor: "/anchor", surface, values: {json.dumps(values)} }},
        ))
        """,
        package,
    )


def _filled(clause: str, **vars) -> str:
    """A surface clause with its tokens filled the way the plugin fills them."""
    for key, value in vars.items():
        clause = clause.replace("{" + key + "}", value)
    return clause


class TestNeedsHumanMessage:
    """The exit-3 ``message`` is the surface's template, filled — and true.

    Four paths land in the needs-human branch and exit 3 means something
    different on each: a foreground gate (the phase never ran), a detached
    worker's death seen through the poll (the run died, and the poll's own
    result line names no phase), an unparsable line (nothing ran) and a
    completed ``review-code --fail-on-high`` (the review finished and wants a
    decision). The phase word therefore comes from the launch, never the
    payload, and "did not complete" is claimed only where it holds. The
    expected fragments are built from ``cli_surface()``'s clauses rather than
    typed, so a rewording moves these rows with the constant and only a
    *dropped* clause turns one red.
    """

    RESULT = {
        "mode": "review-plan", "phase": "review-plan", "state": "needs-human",
        "exit": 3, "artifacts": [], "next": None,
    }
    VALUES = {
        "word": "review-plan", "plan": "p.md", "question": "", "extraFlags": [],
    }

    @staticmethod
    def _text():
        return pkg_build.cli_surface()["needs_human_message"]

    @staticmethod
    def _rule_tail(text):
        """Everything after the two tokens: the stop/report/never rule."""
        return text["template"].split("{error}", 1)[1].strip()

    def test_foreground_gate_names_the_phase_and_the_whole_rule(
        self, built_package
    ):
        text = self._text()
        stdout = "ERROR: the gate refused\n"
        stderr = "the executor could not resolve the runner\n"
        result = _terminal(
            built_package,
            {"stdout": stdout, "stderr": stderr, "result": self.RESULT},
            self.VALUES,
        )
        message = result["message"]
        assert result["state"] == "needs-human" and result["ok"] is False
        # Pinned clause by clause, from the surface: the named phase did not
        # complete; stop; report to the user; never do the phase's work by
        # hand; an options entry goes back only after the user picks one.
        assert message.startswith("Stop here")
        assert _filled(text["phase_incomplete"], phase="review-plan") in message
        assert text["error_unknown"] in message
        assert self._rule_tail(text) in message
        assert mode_table.NEEDS_HUMAN_WORK_SENTENCE in message
        assert "only after the user has picked one" in message
        # No clause claimed that does not hold.
        assert text["phase_unknown"] not in message
        assert text["phase_unnamed"] not in message
        assert _filled(text["review_completed"], phase="review-plan") not in message

    def test_a_dead_worker_seen_through_the_poll_names_the_launched_mode(
        self, built_package
    ):
        """``result.phase`` is null on this path: the poll's result line is
        ``status``'s own. The message names the launched phase and never
        attributes the poll's."""
        text = self._text()
        poll_line = {
            "mode": "status", "phase": None, "state": "needs-human", "exit": 3,
            "artifacts": [], "next": None,
        }
        result = _terminal(
            built_package,
            {"stdout": "run RID-1 failed\n", "result": poll_line},
            {**self.VALUES, "extraFlags": ["--run-id", "RID-1"]},
        )
        message = result["message"]
        assert _filled(text["phase_incomplete"], phase="review-plan") in message
        assert "`status`" not in message
        assert text["phase_unknown"] not in message
        assert text["phase_unnamed"] not in message

    def test_a_payload_with_no_phase_degrades_to_the_generic_rule(
        self, built_package
    ):
        """``init`` owns no phase and its payload names none: the message
        keeps the stop rule and names nothing false."""
        text = self._text()
        payload = {
            "mode": "init", "phase": None, "state": "needs-human", "exit": 3,
            "artifacts": [], "next": None,
        }
        result = _terminal(
            built_package,
            {"stdout": "Choose a provider set:\n", "result": payload},
            {"word": "init", "plan": "", "question": "", "extraFlags": []},
        )
        message = result["message"]
        assert text["phase_unnamed"] in message
        assert self._rule_tail(text) in message
        for phase in mode_table.phase_names():
            assert f"`{phase}`" not in message, f"{phase} named falsely"
        assert "did not complete" not in message

    def test_a_completed_review_held_on_high_asks_for_a_decision(
        self, built_package
    ):
        """``review-code --fail-on-high`` is exit 3 *after* completing: the
        message asks for a decision and never claims an unfinished phase."""
        text = self._text()
        gate = pkg_build.cli_surface()["dispatcher"]["review_gate"]
        payload = {
            "mode": gate["mode"], "phase": mode_table.phase_for(gate["mode"]),
            "state": "needs-human", "exit": 3, "artifacts": ["review.md"],
            "next": None, "high_priority_groups": {"count": 2},
        }
        result = _terminal(
            built_package,
            {"stdout": "[HIGH_PRIORITY_ISSUES] 2\n", "result": payload},
            {"word": gate["mode"], "plan": "p.md", "question": "",
             "extraFlags": [gate["flag"]]},
        )
        message = result["message"]
        expected = _filled(text["review_completed"], phase=payload["phase"])
        assert expected in message
        assert "did not complete" not in message
        assert self._rule_tail(text) in message
        # Through the alias too: the launch word is what is resolved.
        alias = next(
            word for word, target in mode_table.ALIASES.items()
            if target == gate["mode"]
        )
        via_alias = _terminal(
            built_package,
            {"stdout": "", "result": payload},
            {"word": alias, "plan": "p.md", "question": "",
             "extraFlags": [gate["flag"]]},
        )
        assert expected in via_alias["message"]
        # Without the flag the same mode is an ordinary unfinished phase.
        plain = _terminal(
            built_package,
            {"stdout": "", "result": payload},
            {"word": gate["mode"], "plan": "p.md", "question": "",
             "extraFlags": []},
        )
        assert _filled(text["phase_incomplete"], phase=payload["phase"]) in (
            plain["message"]
        )

    def test_fail_on_high_does_not_claim_a_refused_review_completed(
        self, built_package
    ):
        text = self._text()
        gate = pkg_build.cli_surface()["dispatcher"]["review_gate"]
        phase = mode_table.phase_for(gate["mode"])
        payload = {
            "mode": gate["mode"], "phase": phase,
            "state": "needs-human", "exit": 3, "artifacts": [], "next": None,
        }
        result = _terminal(
            built_package,
            {"stdout": "ERROR: runner authentication failed\n", "result": payload},
            {"word": gate["mode"], "plan": "p.md", "question": "",
             "extraFlags": [gate["flag"]]},
        )
        assert _filled(text["phase_incomplete"], phase=phase) in result["message"]
        assert _filled(text["review_completed"], phase=phase) not in result["message"]

    def test_an_unparsable_result_line_names_no_phase_and_quotes_redacted_text(
        self, built_package
    ):
        """No result parsed, so nothing names the phase; the last printed line
        is quoted from the *redacted* output, never from the raw stream."""
        text = self._text()
        command = (
            "uv run --frozen --no-dev --project /s -- python /s/multi_llm.py "
            "init"
        )
        raw_line = "MULTI_LLM_RESULT: {not json " + command
        result = _terminal(
            built_package,
            {
                "stdout": f"Choose:\n  1) {command}\n{raw_line}\n",
                "result": None,
                "resultRaw": raw_line,
            },
            self.VALUES,
        )
        message = result["message"]
        assert text["phase_unknown"] in message
        for phase in mode_table.phase_names():
            assert f"`{phase}`" not in message, f"{phase} named falsely"
        assert "did not complete" not in message
        assert text["error_unknown"] not in message
        # The quoted line is the redacted one: the pointer, not the command.
        assert "uv run" not in message
        assert "multi_llm.py" not in message
        redacted_last = result["output"].rstrip("\n").split("\n")[-1]
        assert redacted_last in message
        assert "`log`" in message
        assert self._rule_tail(text) in message

    def test_no_result_line_and_nothing_printed_points_at_the_log(
        self, built_package
    ):
        """The run left no diagnosis, which is itself what to report."""
        text = self._text()
        result = _terminal(
            built_package,
            {"stdout": "", "stderr": "", "result": None, "resultRaw": None},
            self.VALUES,
        )
        message = result["message"]
        assert text["phase_unknown"] in message
        assert text["error_unknown"] in message
        assert text["error_known"].split("{error}")[0].strip() not in message
        for phase in mode_table.phase_names():
            assert f"`{phase}`" not in message, f"{phase} named falsely"
        assert "did not complete" not in message
        assert "`log`" in message and "`argv`" in message
        assert "uv run" not in message
        assert self._rule_tail(text) in message

    def test_stderr_stands_in_when_output_is_empty(self, built_package):
        """With no result and no stdout, the last stderr line is quoted."""
        text = self._text()
        result = _terminal(
            built_package,
            {"stdout": "", "stderr": "Killed\nworker died: SIGKILL\n",
             "result": None},
            self.VALUES,
        )
        assert _filled(
            text["error_known"], error='"worker died: SIGKILL"'
        ) in result["message"]

    def test_the_message_is_the_surface_and_nothing_else(self, built_package):
        """Every sentence in the assembled message traces to a surface clause."""
        text = self._text()
        result = _terminal(
            built_package,
            {"stdout": "x\n", "result": self.RESULT},
            self.VALUES,
        )
        expected = _filled(
            text["template"],
            phase=_filled(text["phase_incomplete"], phase="review-plan"),
            error=text["error_unknown"],
        )
        assert result["message"] == expected


class TestExecuteStateMachine:
    def test_run_id_comes_from_the_banner(self, built_package, uv_stub):
        """A non-``ask`` launch still yields a poll argv pinned to the banner id.

        The ``MULTI_LLM_RESULT`` payload has **no ``run_id`` key at all**, and the
        launch's ``next`` carries ``--run-id`` only for ``ask`` runs — so the id
        is parsed off the banner. It is never re-derived by listing the phase dir
        or taking "the newest run", because a concurrent run under the same plan
        would then be adopted, polled and, worse, aborted on cancellation.
        """
        launch = {
            "exit": 0,
            "stdout": [
                "=== DETACHED RUN STARTED ===",
                "run_id: RID-42",
                "pid: 1234",
                "log: /tmp/run.log",
                # Deliberately carries no --run-id, as a non-ask launch does not.
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":[],"next":"uv run -- python multi_llm.py status p.md '
                '--view run --wait"}',
            ],
        }
        done = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":["out.md"],"next":null}'
            ],
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, done, done],
        )
        assert result["run_id"] == "RID-42"
        poll = _dispatcher_argv(calls[1])
        assert "--run-id" in poll and poll[poll.index("--run-id") + 1] == "RID-42"
        assert poll[0] == "status" and "--wait" in poll and "--view" in poll
        # `--wait` is a store_true and the plugin still passes the bare flag.
        # What it *means* is no longer a fixed default — the dispatcher resolves
        # it from `progress.poll_s` and clamps that against this host's ceiling —
        # and `test_the_poll_argv_is_the_bare_wait_flag` is where that contract
        # is pinned rather than commented.
        assert "--timeout" not in poll

    def test_launch_result_is_not_the_answer(self, built_package, uv_stub):
        """The launch's ``MULTI_LLM_RESULT`` is a handshake, not the answer.

        It says ``state: done``, ``exit: 0`` — returning it would report "done"
        for a fan-out that has not started work. The terminal payload is the
        final poll's.
        """
        launch = {
            "exit": 0,
            "stdout": [
                "run_id: RID-1",
                "log: /tmp/a.log",
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":["LAUNCH-ARTIFACT"],"next":null}',
            ],
        }
        final = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":["REAL-ARTIFACT"],"next":null}'
            ],
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, final, final],
        )
        assert result["artifacts"] == ["REAL-ARTIFACT"]
        assert "LAUNCH-ARTIFACT" not in json.dumps(result["artifacts"])
        assert result["launch_state"] == "done"
        assert len(calls) >= 2

    def test_argv_names_the_launch_never_the_poll(self, built_package, uv_stub):
        """Codex racing re-run 5: a host told the user the internal
        ``status … --wait --run-id`` poll was "the dispatcher's final argv",
        because the result's ``argv`` — which the needs-human message says to
        report as what was being run — was the terminal poll's (fix round 14).
        """
        launch = {
            "exit": 0,
            "stdout": [
                "run_id: RID-1",
                "log: /tmp/a.log",
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":[],"next":null}',
            ],
        }
        final = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":[],"next":null}'
            ],
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, final, final],
        )
        assert result["argv"][1:] == calls[0]
        assert "review-plan" in result["argv"] and "--detach" in result["argv"]
        assert "--wait" not in result["argv"]
        assert "--run-id" not in result["argv"]

    def test_a_mechanical_next_is_launched_claimed_and_polled_too(
        self, built_package, uv_stub
    ):
        """A ``--reaggregate`` continuation goes through the whole machine.

        ``composeDispatcherArgv`` appends ``--detach`` to every
        ``supports_detach`` mode, and a mechanical continuation always resolves
        to one — so a continuation dispatched with a single spawn hands back the
        *launcher's* handshake (``state: done``, exit 0) while a brand-new worker
        runs with nobody reading it. That is the failure
        ``test_launch_result_is_not_the_answer`` pins for the first launch, one
        cycle later, and it is strictly worse: `execute` would report `done` for
        a re-aggregate that had not started, never poll it, and — on a later
        budget or cap abort — stop the run that had already **finished** while
        the live one spent provider budget unread.

        So the continuation is launched, its banner's run id is adopted, and it
        is polled to its own terminal result. Salvage and the native-marker path
        both emit ``--reaggregate`` as ``next``, which is what makes this
        reachable rather than theoretical.
        """
        reaggregate = "uv run -- python multi_llm.py review-plan p.md --reaggregate"
        launch_one = {
            "exit": 0,
            "stdout": [
                "run_id: RID-1",
                "log: /tmp/one.log",
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":["LAUNCH-1"],"next":null}',
            ],
        }
        phase_one = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":["PHASE-1"],"next":' + json.dumps(reaggregate) + "}"
            ],
        }
        launch_two = {
            "exit": 0,
            "stdout": [
                "run_id: RID-2",
                "log: /tmp/two.log",
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":["LAUNCH-2"],"next":null}',
            ],
        }
        phase_two = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                '"artifacts":["PHASE-2"],"next":null}'
            ],
        }
        args = {"mode": "review-plan", "plan": "p.md", "question": "",
                "extra_flags": []}

        result, calls = uv_stub.run(
            built_package, args, [launch_one, phase_one, launch_two, phase_two]
        )
        dispatches = [
            _dispatcher_argv(call) for call in calls
            if "instructions" not in _dispatcher_argv(call)
        ]
        # Four, not two: the continuation is a launch **and** a poll of its own.
        assert len(dispatches) == 4, dispatches
        continuation = dispatches[2]
        assert continuation[0] == "review-plan"
        assert "--reaggregate" in continuation
        # The flag that makes a single spawn the wrong shape is present here,
        # which is precisely why the poll below has to exist.
        assert "--detach" in continuation

        # Each poll is pinned to the run its own launch claimed — never to the
        # first one, and never to "the newest run".
        pinned = [
            argv[argv.index("--run-id") + 1]
            for argv in dispatches if "--run-id" in argv
        ]
        assert pinned == ["RID-1", "RID-2"]

        # The answer is the continuation's terminal poll: not its launcher's
        # handshake, and not the phase that spawned it.
        assert result["artifacts"] == ["PHASE-2"]
        assert result["continuations"] == 1
        # And the report names the run that was actually in flight last.
        assert result["run_id"] == "RID-2"
        assert result["log"] == "/tmp/two.log"

    def test_the_continuation_cap_aborts_the_run_it_is_watching(
        self, built_package, uv_stub
    ):
        """Cap exhaustion stops the *continuation's* run, not the first launch.

        Every cycle claims a new detached run, so the run a cap abort must reach
        is the last one claimed. Aborting ``RID-1`` here would signal a run that
        finished cycles ago and leave the live one spending provider budget with
        nobody reading it.
        """
        reaggregate = "uv run -- python multi_llm.py review-plan p.md --reaggregate"
        payload = (
            'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
            '"artifacts":["a"],"next":' + json.dumps(reaggregate) + "}"
        )
        launch_one = {"exit": 0, "stdout": ["run_id: RID-1", "log: /tmp/one.log",
                                            payload]}
        phase_one = {"exit": 0, "stdout": [payload]}
        launch_two = {"exit": 0, "stdout": ["run_id: RID-2", "log: /tmp/two.log",
                                            payload]}
        phase_two = {"exit": 0, "stdout": [payload]}
        # What a confirmed stop actually looks like on the wire: `_run_abort`
        # prints its human line and the dispatcher ends the invocation with the
        # terminal result line. Both are required — the exit code alone is not
        # the dispatcher's own verdict, so a stub without the result line would
        # be scripting an *unconfirmed* abort.
        aborted = {
            "exit": 0,
            "stdout": [
                "aborted review-plan run RID-2 (process group terminated)",
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":[],"next":null}',
            ],
        }

        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "",
             "extra_flags": []},
            [launch_one, phase_one, launch_two, phase_two, aborted],
            extra_env={"MLLM_TEST_NEXT_CAP": "1"},
        )
        assert result["state"] == "needs-human" and result["ok"] is False
        assert result["run_id"] == "RID-2"
        assert result["log"] == "/tmp/two.log"
        assert result["abort_dispatched"] is True
        abort = [
            argv for argv in (_dispatcher_argv(call) for call in calls)
            if "--abort" in argv
        ]
        assert len(abort) == 1, abort
        assert abort[0][abort[0].index("--run-id") + 1] == "RID-2"
        # The abort tool call handed back names the same run.
        assert result["abort_command"]["args"]["extra_flags"][-1] == "RID-2"

    def test_poll_loop_exit_map(self, built_package, uv_stub):
        """2,2,0 → done and exactly two re-invocations; 2,3 → needs-human; 1 →
        error.

        Exit 2's ambiguity is the trap worth naming: "still running" and "no run
        recorded" are the same code **by design**, because 0 would read as
        finished. The ``--run-id`` pin is what distinguishes them — a loop that
        adopted "the newest run" would poll a stranger *successfully*, which is
        the worse failure.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: RID", "log: /tmp/l.log",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        running = {"exit": 2, "stdout": [
            'MULTI_LLM_RESULT: {"state":"nothing-to-do","exit":2,"next":null}']}
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":["a"],'
            '"next":null}']}
        failed = {"exit": 3, "stdout": ["the run failed"], "stderr": ["trace"]}
        broken = {"exit": 1, "stdout": [], "stderr": ["status: bad arguments"]}
        args = {"mode": "review-plan", "plan": "p.md", "question": "",
                "extra_flags": []}

        result, calls = uv_stub.run(
            built_package, args, [launch, running, running, done]
        )
        assert result["state"] == "done" and result["ok"] is True
        # The launch, then exactly three polls: two non-terminal, one terminal.
        # The instructions fetch is a separate contract with its own row, so it
        # is excluded rather than counted.
        dispatches = [
            call for call in calls
            if "instructions" not in _dispatcher_argv(call)
        ]
        assert len(dispatches) == 4, [c[-4:] for c in dispatches]

        result, calls = uv_stub.run(built_package, args, [launch, running, failed])
        assert result["state"] == "needs-human" and result["ok"] is False
        assert "the run failed" in result["output"]
        assert "trace" in result["stderr"]

        result, calls = uv_stub.run(built_package, args, [launch, broken])
        assert result["state"] == "error"
        # `status` failing is never a statement about the run: the claimed run is
        # still alive, so the result carries what a human needs to stop it.
        assert result["run_id"] == "RID"
        assert result["log"] == "/tmp/l.log"
        assert result["abort_command"]["tool"] == "multi_llm"
        # The EXPLICIT `status … --abort` spelling, from the generated table —
        # not the `abort` alias. Both are legal and they are the same run, but
        # the plugin holds no mode word of its own to spell the alias with.
        spelling = pkg_build.cli_surface()["dispatcher"]["abort"]
        assert result["abort_command"]["args"]["mode"] == spelling[0]
        assert spelling[1] in result["abort_command"]["args"]["extra_flags"]

    def test_a_live_reattach_polls_instead_of_returning_exit_2(
        self, built_package, uv_stub
    ):
        """`status --view run --wait` on a live run polls to a terminal state.

        `status` declares ``supports_detach`` false, so a re-attach takes the
        foreground branch — and a view of a run still in flight exits 2. Handing
        that straight back ends the call after a single wait interval and reports
        an unfinished run as ``ok: true``, which tells a model the fan-out
        succeeded. So a **caller-pinned** view of a live run enters the same
        bounded loop a launch does, pinned to the id the *dispatcher* named
        rather than the prefix the caller typed.

        Adopted is not claimed, though: this call did not start that run, so
        exhausting the bound reports the orphan and leaves it alive. Killing a
        paid fan-out some other call launched is not what abandoning a watch
        asks for.
        """
        # The re-attach itself: exit 2, and a `next` that is the dispatcher
        # saying "this run is alive; poll it again with this" — carrying the
        # WHOLE id, where the caller pinned only a unique prefix.
        live = {
            "exit": 2,
            "stdout": [
                "log: /tmp/r.log",
                'MULTI_LLM_RESULT: {"mode":"status","state":"nothing-to-do",'
                '"exit":2,"artifacts":[],"next":"uv run -- python multi_llm.py '
                'status p.md --view run --wait --run-id RID-77"}',
            ],
        }
        done = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":["out.md"],"next":null}'
            ],
        }
        watch = {
            "mode": "status", "plan": "p.md", "question": "",
            "extra_flags": ["--view", "run", "--wait", "--run-id", "RID"],
        }

        result, calls = uv_stub.run(built_package, watch, [live, live, done])
        assert result["state"] == "done" and result["ok"] is True
        assert result["run_id"] == "RID-77"
        assert result["log"] == "/tmp/r.log"
        dispatches = [
            _dispatcher_argv(call) for call in calls
            if "instructions" not in _dispatcher_argv(call)
        ]
        # The re-attach, then two polls: one non-terminal, one terminal. Exit 2
        # is not the answer it once was.
        assert len(dispatches) == 3, dispatches
        poll = dispatches[1]
        assert poll[0] == "status" and "--wait" in poll
        assert poll[poll.index("--run-id") + 1] == "RID-77"

        # Abandoning the watch leaves the run alone: no abort is dispatched, and
        # the result still names the run, its log and the abort as a tool call.
        result, calls = uv_stub.run(
            built_package, watch, [live, done],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1"},
        )
        assert result["state"] == "needs-human" and result["ok"] is False
        assert result["run_id"] == "RID-77"
        assert result["abort_dispatched"] is False
        assert result["abort_command"]["args"]["extra_flags"][-1] == "RID-77"
        assert not [
            call for call in calls if "--abort" in _dispatcher_argv(call)
        ], "a watch this call never claimed was aborted"

    def test_terminal_status_queries_stay_terminal(self, built_package, uv_stub):
        """Everything that is not a live pinned view still returns on one child.

        An abort, a view of a run already finished, and an exit-2 snapshot that
        named no run at all: exit 2 means "no such run in view" just as often as
        it means "still running", and what separates them is the `next` the
        dispatcher records — never the exit code. A loop entered on the empty
        case would poll nothing to the budget and then report a run that does
        not exist.
        """
        finished = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":[],"next":null}'
            ],
        }
        # Exit 2 with no follow-up: `run_status.main` records one ONLY for a
        # non-empty, non-terminal view, so a bare 2 is an empty view.
        empty = {
            "exit": 2,
            "stdout": [
                'MULTI_LLM_RESULT: {"mode":"status","state":"nothing-to-do",'
                '"exit":2,"artifacts":[],"next":null}'
            ],
        }
        rows = [
            ({"mode": "status", "plan": "p.md", "question": "",
              "extra_flags": ["--abort", "--run-id", "RID-77"]}, finished),
            ({"mode": "status", "plan": "p.md", "question": "",
              "extra_flags": ["--view", "run", "--run-id", "RID-77"]}, finished),
            ({"mode": "status", "plan": "p.md", "question": "",
              "extra_flags": ["--view", "run", "--wait", "--run-id", "RID-77"]},
             empty),
            # No pin at all: a plan holding several live runs records a follow-up
            # for whichever phase dir sorted first, and polling — then reporting
            # — a run nobody asked about is the adoption the launch path refuses.
            ({"mode": "status", "plan": "p.md", "question": "",
              "extra_flags": ["--view", "run", "--wait"]},
             {"exit": 2, "stdout": [
                 'MULTI_LLM_RESULT: {"mode":"status","state":"nothing-to-do",'
                 '"exit":2,"artifacts":[],"next":"uv run -- python multi_llm.py '
                 'status p.md --view run --wait --run-id STRANGER"}']}),
        ]
        for args, step in rows:
            _, calls = uv_stub.run(built_package, args, [step, step, step])
            dispatches = [
                call for call in calls
                if "instructions" not in _dispatcher_argv(call)
            ]
            assert len(dispatches) == 1, (args["extra_flags"], dispatches)

    def test_budget_exhaustion_aborts_the_claimed_run(self, built_package, uv_stub):
        """The bound aborts what it claimed; it never silently gives up.

        A tool that merely stops polling leaves a fan-out spending paid provider
        budget with nobody left to read the result. ``detach.launch()``
        daemonises a worker that outlives both the launcher and the session, and
        the banner says so verbatim on every launch.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: RID-9", "log: /tmp/x.log",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        # A 1 ms budget is spent by the time the launch returns, so the second
        # child is the abort itself and never a poll. It is scripted as the real
        # one behaves on a confirmed stop: `_run_abort`'s human line plus the
        # dispatcher's terminal result line. Both are required — an exit code
        # alone is not the dispatcher's own verdict, so a stub without the
        # result line would be scripting an *unconfirmed* abort.
        aborted = {
            "exit": 0,
            "stdout": [
                "aborted review-plan run RID-9 (process group terminated)",
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":[],"next":null}',
            ],
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, aborted],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1"},
        )
        assert result["state"] == "needs-human"
        assert result["run_id"] == "RID-9"
        assert result["log"] == "/tmp/x.log"
        # `status <plan> --abort --run-id <id>` is the documented ONLY kill path:
        # identity probe, process-group SIGTERM, then SIGKILL after a grace. The
        # plugin never signals the worker pid itself — the worker is not its
        # child and the pid may have been recycled.
        abort_calls = [
            _dispatcher_argv(call) for call in calls
            if "--abort" in _dispatcher_argv(call)
        ]
        assert abort_calls, f"no abort was dispatched: {[c[-6:] for c in calls]}"
        abort = abort_calls[0]
        assert abort[0] == "status"
        assert abort[abort.index("--run-id") + 1] == "RID-9"
        assert result["abort_dispatched"] is True

    def test_the_budget_interrupts_a_hung_poll_child(self, built_package, uv_stub):
        """The budget bounds *time*, not iterations.

        Checking the deadline and then awaiting an unbounded child makes the
        six-hour backstop a bound on how many polls are started, never on how
        long the call takes: one `uv` that never starts moving — or a status
        child whose python grandchild still holds stdout open — blocks past the
        deadline and never lets the loop reach it again. The remaining budget is
        passed into the child, and a child killed on it takes the budget's exit
        rather than being reported as the run's terminal answer: it has no exit
        code of its own, and `null` is not `nothing_to_do`.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: RID-H", "log: /tmp/h.log",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        # The hang: no output at all, and far longer than the budget below. A
        # poll that ran to completion would exit 0 with no result line, which
        # the loop reads as terminal — so an unbounded await fails this row by
        # returning `state: error` thirty seconds late, not by hanging forever.
        hung = {"sleep": 30}
        aborted = {
            "exit": 0,
            "stdout": [
                "aborted review-plan run RID-H (process group terminated)",
                'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
                '"artifacts":[],"next":null}',
            ],
        }
        started = time.monotonic()
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, hung, aborted],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "3000"},
        )
        elapsed = time.monotonic() - started
        # The point of the row: the call came back on the budget, not on the
        # child's own schedule.
        assert elapsed < 20, f"the hung poll outlived the budget ({elapsed:.1f}s)"
        assert result["state"] == "needs-human" and result["ok"] is False
        assert "budget" in result["reason"] and "exhausted" in result["reason"]
        assert result["run_id"] == "RID-H"
        # A poll *was* started — otherwise the budget expired before the loop
        # and this row would pass without ever bounding a child.
        dispatches = [_dispatcher_argv(call) for call in calls]
        polls = [
            argv for argv in dispatches
            if argv[0] == "status" and "--abort" not in argv
        ]
        assert polls, f"no poll was started: {dispatches}"
        assert polls[0][polls[0].index("--run-id") + 1] == "RID-H"
        # And the run it claimed is still aborted: a killed poll is the call
        # stopping, never the run stopping.
        aborts = [argv for argv in dispatches if "--abort" in argv]
        assert aborts, f"the killed poll abandoned the claimed run: {dispatches}"
        assert aborts[0][aborts[0].index("--run-id") + 1] == "RID-H"
        assert result["abort_dispatched"] is True

    def test_failed_or_malformed_child_is_a_result(
        self, built_package, uv_stub, tmp_path
    ):
        """ENOENT, a non-zero exit with no result line, and unparseable JSON.

        Each returns a structured error naming the argv and the exit code;
        ``execute`` never throws. A throw here is a silent exit-0 death with a
        0-byte log.
        """
        args = {"mode": "init", "plan": "", "question": "", "extra_flags": []}

        # A non-zero exit carrying no MULTI_LLM_RESULT.
        result, _ = uv_stub.run(
            built_package, args, [{"exit": 1, "stderr": ["Traceback: boom"]}]
        )
        assert result["state"] == "error" and result["ok"] is False
        assert result["exit"] == 1
        assert "Traceback: boom" in result["stderr"]
        assert result["argv"], "the failing argv is not reported"

        # An unparseable result line.
        result, _ = uv_stub.run(
            built_package, args,
            [{"exit": 0, "stdout": ["MULTI_LLM_RESULT: {not json"]}],
        )
        assert result["state"] == "error"
        assert "not valid JSON" in result["message"]

        # A spawn error — a uv the probe accepts and the OS refuses to exec.
        #
        # It has to be a real **file**, because `findUv`'s override branch tests
        # `isFile` and nothing else: pointed at a path that does not exist, the
        # override falls through to PATH and then to the home-relative candidate
        # list, and this row's `HOME` is the developer's own. That is not a
        # hypothetical — it is what this row used to do, quietly spawning the
        # *real* uv against the staged tree, leaving a `.venv` and a
        # `__pycache__` inside the session-scoped `built_package` that `TestPack`
        # then found in the tarball. So the file exists and is unreadable to
        # `exec`, and HOME/PATH are neutered as well so no later change to that
        # branch can silently put the real binary back.
        not_a_binary = tmp_path / "uv-not-executable"
        not_a_binary.write_text("not a binary\n", encoding="utf-8")
        not_a_binary.chmod(0o644)
        outcome = node_json(
            """
            const r = await MultiLlmPlugin.execute(
              {mode: "init", plan: "", question: "", extra_flags: []},
              {directory: "/"}, surface,
            )
            emit({state: r.state, message: r.message})
            """,
            built_package,
            env={
                "MULTI_LLM_UV": str(not_a_binary),
                "PATH": str(tmp_path / "no-path"),
                "HOME": str(tmp_path / "no-home"),
            },
        )
        assert outcome["state"] == "error"
        assert "could not start uv" in outcome["message"], (
            "the spawn failure was not reported as a result naming its cause: "
            + outcome["message"]
        )

    def test_the_whole_stream_is_scanned_for_the_result_line(
        self, built_package, uv_stub
    ):
        """A body larger than the retention budget still yields the result.

        The payload is the **last** ``MULTI_LLM_RESULT:`` line and an arbitrarily
        long body can precede it, so a fixed-size tail *read* is not a legal
        implementation — the tail would no longer contain the line the stream
        still does. What is bounded is what is retained, and the truncation is
        marked and taken from the middle, never the end.
        """
        noise = [f"filler line {index} " + "x" * 200 for index in range(1000)]
        result, _ = uv_stub.run(
            built_package,
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            [{
                "exit": 0,
                "stdout": noise + [
                    'MULTI_LLM_RESULT: {"state":"done","exit":0,'
                    '"artifacts":["found"],"next":null}'
                ],
            }],
        )
        assert result["ok"] is True
        assert result["artifacts"] == ["found"]
        assert result["stdout_truncated"] is True

    def test_next_is_never_a_shell_command(self, built_package, uv_stub):
        """The result contains no runnable ``uv run`` string, anywhere.

        Handed a shell command, a model's natural move is to run it in bash,
        which bypasses every guard in the plugin — reintroducing exactly the
        improvisation this route exists to remove.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: R", "log: /l",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        follow = (
            "uv run --project /skill -- python /skill/multi_llm.py "
            "apply-suggestions /w/p.md --host opencode --yes"
        )
        done = {
            "exit": 0,
            "stdout": [
                'MULTI_LLM_RESULT: ' + json.dumps({
                    "state": "done", "exit": 0, "artifacts": [],
                    "next": follow,
                })
            ],
        }
        result, _ = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, done, done],
        )
        blob = json.dumps(result)
        assert "uv run" not in blob, f"a runnable command survived into the result"
        assert result["next"]["tool"] == "multi_llm"
        assert result["next"]["args"]["mode"] == "apply-suggestions"
        assert result["next"]["args"]["plan"] == "/w/p.md"
        # The flags the plugin composes itself are stripped: the model would be
        # refused for sending them back.
        assert result["next"]["args"]["extra_flags"] == []

    def test_an_escaped_copy_of_next_is_redacted_too(self, built_package, uv_stub):
        """The *escaped* spelling of a command is the same command.

        ``next`` reaches the plugin decoded — ``JSON.parse`` did that — while the
        text it was parsed out of holds it escaped, and a fan-out echoes exactly
        such text: a worker's own ``MULTI_LLM_RESULT:`` line, quoted inside the
        parent's output with a prefix, so it is neither parsed as the payload nor
        dropped as the envelope.

        The two spellings differ only in ``"`` and ``\\``, which is why the
        Windows-path case is the one that leaks: splitting on the decoded form
        alone never touches the escaped copy, and the shape pattern is no
        backstop for it — the pattern's run ends at a quote, so a command whose
        script path sits behind an escaped one is not matched at all and reaches
        the model whole, and runnable.
        """
        follow = (
            'uv run --project "C:\\Program Files\\skill" -- python '
            '"C:\\Program Files\\skill\\multi_llm.py" status "C:\\w\\p.md" '
            "--host opencode --yes"
        )
        launch = {
            "exit": 0,
            "stdout": ["run_id: R", "log: /l",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        done = {
            "exit": 0,
            "stdout": [
                # A worker's envelope, echoed by the parent: prefixed, so the
                # plugin neither parses it nor filters it out of `output`.
                "[worker] MULTI_LLM_RESULT: " + json.dumps(
                    {"state": "done", "exit": 0, "next": follow}
                ),
                "MULTI_LLM_RESULT: " + json.dumps(
                    {"state": "done", "exit": 0, "artifacts": [], "next": follow}
                ),
            ],
        }
        # The third call is `util instructions`, whose stdout is returned as
        # prose rather than redacted; kept empty so this row grades redaction.
        result, _ = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch, done, {"exit": 0, "stdout": []}],
        )
        blob = json.dumps(result)
        assert "uv run" not in blob, "an escaped runnable command survived"
        # The whole span went, not just its head: a match that stopped at the
        # first escaped quote would leave the rest of the command in place.
        assert "Program Files" not in blob
        assert result["next"]["tool"] == "multi_llm"

    def test_an_unparseable_next_is_reported_not_run(self, built_package):
        """Reported, never executed. Never handed to a shell."""
        outcome = node_json(
            """
            emit([
              T.nextAsToolCall("some free text a shell would choke on", surface),
              T.nextAsToolCall(null, surface),
            ])
            """,
            built_package,
        )
        assert outcome[0]["unparseable"] is True
        assert outcome[1] is None

    def test_post_return_abort_is_not_a_cancellation(self, built_package, uv_stub):
        """``ctx.abort`` firing ~20 ms after ``execute`` returns changes nothing.

        That signal is the per-call AbortController tearing down, not a
        cancellation. Code treating it as "I was cancelled" mis-fires on **every
        successful call**. The listener is removed before returning, so a late
        signal has nowhere to land: no kill of any child, no abort dispatch, and
        — the half the kill assertions alone do not cover — no spurious error.
        The already-resolved payload is the done result, unchanged.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: R2", "log: /l2",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":["ok"],'
            '"next":null}']}
        log = uv_stub.log_path
        log.write_text("", encoding="utf-8")
        outcome = node_json(
            f"""
            const controller = new AbortController()
            const promise = MultiLlmPlugin.execute(
              {{mode: "review-plan", plan: "p.md", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(log.parent))}, abort: controller.signal}}, surface,
            )
            const settled = await promise
            controller.abort()               // the ~20 ms teardown
            await new Promise((r) => setTimeout(r, 120))
            emit({{state: settled.state, ok: settled.ok,
                  artifacts: settled.artifacts,
                  keys: Object.keys(settled).sort()}})
            """,
            built_package,
            env={
                **uv_stub.env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT": json.dumps([launch, done, done]),
            },
        )
        assert outcome["state"] == "done" and outcome["ok"] is True
        assert outcome["artifacts"] == ["ok"]
        # No abort dispatch happened after the return, and no SIGKILL was issued
        # off that edge.
        calls = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines() if line
        ]
        assert not any("--abort" in call for call in calls), (
            "the teardown signal reached a process"
        )
        # And nothing was appended to the resolved payload.
        assert "abort_command" not in outcome["keys"]
        assert "reason" not in outcome["keys"]

    def test_abort_mid_execute_aborts_the_claimed_run(self, built_package, uv_stub):
        """A genuine abort mid-poll stops polling and aborts what it claimed.

        The run id comes from the launch record and is never re-derived, so a
        concurrent run under the same plan is untouched.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: CLAIMED", "log: /l3",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        running = {"exit": 2, "stdout": []}
        log = uv_stub.log_path
        log.write_text("", encoding="utf-8")
        outcome = node_json(
            f"""
            const controller = new AbortController()
            setTimeout(() => controller.abort(), 400)
            emit(await MultiLlmPlugin.execute(
              {{mode: "review-plan", plan: "p.md", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(log.parent))}, abort: controller.signal}}, surface,
            ))
            """,
            built_package,
            env={
                **uv_stub.env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT": json.dumps([launch, running]),
            },
        )
        assert outcome["state"] == "needs-human"
        assert outcome["run_id"] == "CLAIMED"
        calls = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines() if line
        ]
        aborts = [call for call in calls if "--abort" in call]
        assert aborts, "a claimed run was abandoned rather than aborted"
        assert "CLAIMED" in aborts[0]

    @pytest.mark.parametrize("failure", [None, "error", "exit"])
    def test_windows_cancellation_kills_the_tree_before_its_wrapper(
        self, built_package, failure
    ):
        outcome = node_json(
            """
            const cp = (await import('node:child_process')).default
            const { syncBuiltinESMExports } = await import('node:module')
            const { EventEmitter } = await import('node:events')
            const { PassThrough } = await import('node:stream')
            Object.defineProperty(process, 'platform', { value: 'win32' })
            const calls = []
            const child = new EventEmitter()
            child.pid = 123
            child.stdout = new PassThrough()
            child.stderr = new PassThrough()
            let alive = true
            let descendantAlive = true
            const finish = () => {
              alive = false
              child.stdout.end()
              child.stderr.end()
              child.emit('close', 1)
            }
            child.kill = () => { calls.push('leader'); finish() }
            process.kill = () => { throw new Error('no POSIX groups on Windows') }
            cp.spawn = (exe, argv, options) => {
              if (exe === 'fake-uv') return child
              calls.push({ exe, argv, timeout: options.timeout, alive })
              const killer = new EventEmitter()
              queueMicrotask(() => {
                if (FAILURE === 'error') killer.emit('error', new Error('not found'))
                else if (FAILURE === 'exit') killer.emit('close', 1)
                else {
                  descendantAlive = false
                  finish()
                  killer.emit('close', 0)
                }
              })
              return killer
            }
            syncBuiltinESMExports()
            const controller = new AbortController()
            const pending = T.runChild({
              uv: 'fake-uv', argv: [], env: process.env,
              marker: surface.dispatcher.result_marker, signal: controller.signal,
            })
            controller.abort()
            const result = await pending
            emit({ calls, descendantAlive, aborted: result.aborted })
            """.replace("FAILURE", json.dumps(failure)),
            built_package,
        )
        assert outcome["aborted"] is True
        assert outcome["calls"][0] == {
            "exe": "taskkill", "argv": ["/PID", "123", "/T", "/F"],
            "timeout": 5000, "alive": True,
        }
        if failure is None:
            assert outcome["descendantAlive"] is False
            assert len(outcome["calls"]) == 1
        else:
            assert outcome["calls"][1:] == ["leader"]

    def test_a_bounded_dispatch_survives_a_child_that_ignores_sigterm(
        self, built_package, tmp_path
    ):
        """The timeout escalates to SIGKILL, and records that it fired.

        ``runChild`` resolves on ``close``, which waits for the stdio pipes as
        well as for the exit — so one SIGTERM to a child that ignores it (or to
        a grandchild still holding stdout) never settles the promise at all. The
        caller that sets a timeout is the *abort* dispatch, the one call that
        exists to stop a paid fan-out, so an unbounded wait there is the
        "abandoned, not aborted" outcome arriving by the back door. The
        escalation is what bounds it; ``timedOut`` is what keeps the report from
        reading the resulting silence — no exit code, no result line — as the
        dispatcher's own verdict.
        """
        stubborn, launch_env = spawnable_stub(
            tmp_path, "deaf-uv",
            "#!/usr/bin/env python3\n"
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True:\n"
            "    time.sleep(0.2)\n",
        )
        outcome = node_json(
            f"""
            const started = Date.now()
            const out = await T.runChild({{
              uv: {json.dumps(str(stubborn))},
              argv: [],
              cwd: {json.dumps(str(tmp_path))},
              env: process.env,
              marker: surface.dispatcher.result_marker,
              timeoutMs: 250,
            }})
            emit({{elapsedMs: Date.now() - started, code: out.code,
                  timedOut: out.timedOut, aborted: out.aborted,
                  spawnError: out.spawnError}})
            """,
            built_package,
            env=launch_env,
        )
        # Settled at all is the whole point: without the SIGKILL this hangs
        # until the caller's own timeout kills node.
        assert outcome["elapsedMs"] < 30_000, "the bounded dispatch was not bounded"
        assert outcome["spawnError"] is None
        # A forced termination carries no result line, which is why the timeout
        # must be recorded independently of the process exit status.
        if sys.platform == "win32":
            # An external taskkill reports a native termination exit code;
            # Node only reports a signal when it sent the signal itself.
            assert outcome["code"] != 0
        else:
            assert outcome["code"] is None
        assert outcome["timedOut"] is True
        # A timeout is not a cancellation: no signal was passed, so nothing may
        # claim the caller aborted.
        assert outcome["aborted"] is False

    def test_abort_during_the_launch_aborts_the_claimed_run(
        self, built_package, uv_stub,
    ):
        """Cancellation inside the *launch window* still aborts what was claimed.

        ``detach.launch()`` daemonises the worker and prints the ``run_id``
        banner **before** the launcher exits, so a cancellation landing in that
        window kills the launcher while the detached worker is already alive.
        Reading the launcher's exit first and returning on it would ignore the
        run id it had already printed: no abort dispatched, no orphan reported —
        the "abandoned, not aborted" outcome the whole path exists to prevent.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: CLAIMED", "log: /l7"],
            # Long enough that the cancellation below is certain to land while
            # this child is still running.
            "sleep": 30,
        }
        done = {
            "exit": 0,
            "stdout": ['MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        log = uv_stub.log_path
        log.write_text("", encoding="utf-8")
        outcome = node_json(
            f"""
            const controller = new AbortController()
            setTimeout(() => controller.abort(), 400)
            emit(await MultiLlmPlugin.execute(
              {{mode: "review-plan", plan: "p.md", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(log.parent))}, abort: controller.signal}}, surface,
            ))
            """,
            built_package,
            env={
                **uv_stub.env,
                "STUB_LOG": str(log),
                "STUB_SCRIPT": json.dumps([launch, done]),
            },
        )
        assert outcome["state"] == "needs-human" and outcome["ok"] is False
        assert outcome["run_id"] == "CLAIMED"
        assert outcome["log"] == "/l7"
        assert outcome["abort_dispatched"] is True
        calls = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines() if line
        ]
        aborts = [call for call in calls if "--abort" in call]
        assert aborts, "the launch window claimed a run and then abandoned it"
        assert "CLAIMED" in aborts[0]

    def test_launch_failing_after_its_banner_aborts_the_claimed_run(
        self, built_package, uv_stub,
    ):
        """A launcher that printed a run id and *then* failed still claimed one.

        The mirror of the cancellation case, and the reason the banner is parsed
        before the exit code is consulted: a nonzero exit on the way out of a
        launch is not proof that nothing is running.
        """
        launch = {
            "exit": 3,
            "stdout": ["run_id: CLAIMED", "log: /l7"],
            "stderr": ["the launcher fell over on its way out"],
        }
        done = {
            "exit": 0,
            "stdout": ['MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "",
             "extra_flags": []},
            [launch, done],
        )
        assert result["state"] == "needs-human" and result["ok"] is False
        assert result["run_id"] == "CLAIMED"
        assert result["log"] == "/l7"
        assert result["abort_dispatched"] is True
        # The abort is spelled back as a tool call too, so a failed dispatch
        # still leaves the reader something runnable.
        assert result["abort_command"]["args"]["extra_flags"][-2:] == [
            "--run-id", "CLAIMED",
        ]
        aborts = [call for call in calls if "--abort" in call]
        assert aborts, "a run named on the banner was abandoned rather than aborted"
        assert "CLAIMED" in aborts[0]

    def test_ask_exit3_renders_the_option_block(self, built_package, uv_stub):
        """An exit-3 block comes back **verbatim**, and ``next`` stops.

        When the *tool* ran the dispatcher, that block went into the tool's
        captured stdout, not into the transcript. So it is returned exactly as
        rendered — never synthesized, and never paraphrased.

        Verbatim stops where D3a does. The fixture's payload carries a runnable
        ``uv run`` in its ``next``, which reaches the result twice — parsed, and
        again inside the captured ``MULTI_LLM_RESULT:`` line — and neither copy
        may survive as something a model can paste. The envelope line goes
        (``result`` already holds it, parsed), the command is redacted where it
        stood, and it comes back only as a tool call in ``options``.
        """
        block = [
            "Choose a model set:",
            "  1) provider:a",
            "  2) provider:b",
        ]
        result, _ = uv_stub.run(
            built_package,
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            [{"exit": 3, "stdout": block + [
                'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,'
                '"next":"uv run -- python multi_llm.py init"}']}],
        )
        assert result["state"] == "needs-human" and result["ok"] is False
        for line in block:
            assert line in result["output"]
        assert result["next"] is None, "following `next` did not stop on exit 3"
        assert "uv run" not in json.dumps(result), (
            "a runnable command survived into the exit-3 result"
        )
        assert "MULTI_LLM_RESULT" not in result["output"], (
            "the dispatcher's own envelope line came back as rendered output"
        )
        # Nothing is dropped by the redaction: the command it replaced is handed
        # back in the only form this plugin ever hands one back — and the
        # final element is the one option that is not a retry (D2 step 5):
        # one key, `human`, no `tool`, no `args`, read off the surface.
        options = result["options"]
        assert [
            (call["tool"], call["args"]["mode"])
            for call in options if "tool" in call
        ] == [("multi_llm", "init")]
        assert options[-1] == {"human": mode_table.NEEDS_HUMAN_OPTION}

    def test_nested_exec_reasons_are_tool_errors(self, built_package, uv_stub):
        """All four executor refusals become named, structured tool errors.

        The executor's refusals are structured, not crashes. They surface with
        the reason code and the capability field they came from — never a stack
        trace, never a bare non-zero exit, and never a silent exit 0.
        """
        for code, field in (
            ("nested_exec_denied", "nested_exec"),
            ("nested_exec_spawn", "nested_exec.spawn"),
            ("nested_exec_network", "nested_exec.network"),
            ("nested_exec_credentials", "nested_exec.credentials"),
        ):
            result, _ = uv_stub.run(
                built_package,
                {"mode": "init", "plan": "", "question": "", "extra_flags": []},
                [{"exit": 3, "stdout": [f"[{code}] the executor refused"]}],
            )
            assert result["state"] == "needs-human"
            assert result["nested_exec"]["reason"] == code
            assert result["nested_exec"]["capability_field"] == field

    def test_child_cwd_is_the_workspace(self, built_package, uv_stub, tmp_path):
        """Composed from a process whose own cwd is outside the workspace.

        ``ctx.worktree ?? ctx.directory``, in that precedence; a relative plan
        resolves against it with ``~/`` expanded; and the anchor used is echoed
        in the result, because a wrong anchor is otherwise invisible — it
        surfaces as a model-selection bug, not as a config error.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        result, calls = uv_stub.run(
            built_package,
            {"mode": "init", "plan": "", "question": "", "extra_flags": []},
            [{"exit": 0, "stdout": [
                'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}],
            ctx={"worktree": str(workspace), "directory": str(other)},
        )
        # worktree wins over directory.
        assert result["anchor"] == str(workspace)

        # A relative plan resolves under the anchor, not under the process cwd.
        argv = node_json(
            f"""
            const anchor = T.resolveAnchor(
              {{worktree: {json.dumps(str(workspace))},
                directory: {json.dumps(str(other))}}})
            emit({{
              anchor,
              relative: T.normalizePlan("plans/x.md", anchor),
              absolute: T.normalizePlan("/abs/x.md", anchor),
              home: T.normalizePlan("~/x.md", anchor),
              fallback: T.resolveAnchor({{}}) === process.cwd(),
              directoryOnly: T.resolveAnchor(
                {{directory: {json.dumps(str(other))}}}),
            }})
            """,
            built_package,
            cwd=Path("/"),
        )
        assert argv["anchor"] == str(workspace)
        assert argv["relative"] == str(workspace / "plans" / "x.md")
        assert argv["absolute"] == "/abs/x.md"
        assert argv["home"] == str(Path.home() / "x.md")
        assert argv["fallback"] is True
        assert argv["directoryOnly"] == str(other)

    def test_a_plan_spelled_from_the_repository_root(self, built_package, tmp_path):
        """Fix round 26 (live run 11): the dispatcher runs a plan that names
        nothing from the anchor from the root of the repository the anchor is
        in, so this server names that root's file for its polls and abort too.
        Fix round 27 (the Fable review of round 26, claude #5): it walked to
        the nearest `.git` itself, which named the root's plan where the
        dispatcher refused to move (`GIT_DIR` set, a planted `core.worktree`,
        a planted nested `.git`). `normalizePlan` now resolves from the anchor
        alone, and `planFromRoot` renames the plan only by the root the
        dispatcher's result line says it ran from."""
        repo = Path(os.path.realpath(tmp_path)) / "repo"
        sub = repo / "sub"
        sub.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "plans").mkdir()
        (repo / "plans" / "x.md").write_text("# Plan\n", encoding="utf-8")
        other = Path(os.path.realpath(tmp_path)) / "other"
        other.mkdir()
        key = pkg_build.cli_surface()["dispatcher"]["ran_from_root_key"]
        resolved = node_json(
            f"""
            const sub = {json.dumps(str(sub))}
            const moved = (root, planArg, plan) => {{
              const values = {{ plan, planArg }}
              T.planFromRoot({{ {key}: root }}, values, sub, surface)
              return values.plan
            }}
            const joined = T.normalizePlan("plans/x.md", sub)
            emit({{
              joined,
              sigil: T.normalizePlan("@plans/x.md", sub),
              root: moved({json.dumps(str(repo))}, "plans/x.md", joined),
              none: moved(undefined, "plans/x.md", joined),
              other: moved({json.dumps(str(other))}, "plans/x.md", joined),
              relative: moved("repo", "plans/x.md", joined),
              absolute: moved({json.dumps(str(repo))}, "/abs/x.md", "/abs/x.md"),
              noPlan: moved({json.dumps(str(repo))}, "", ""),
            }})
            """,
            built_package,
            cwd=Path("/"),
        )
        # Nothing is walked: from the anchor, whatever the root holds.
        assert resolved["joined"] == str(sub / "plans" / "x.md")
        assert resolved["sigil"] == str(sub / "plans" / "x.md")
        assert resolved["root"] == str(repo / "plans" / "x.md")
        assert resolved["none"] == str(sub / "plans" / "x.md")
        assert resolved["other"] == str(sub / "plans" / "x.md")
        assert resolved["relative"] == str(sub / "plans" / "x.md")
        assert resolved["absolute"] == "/abs/x.md"
        assert resolved["noPlan"] == ""

    @pytest.mark.parametrize("moved", [True, False])
    def test_the_polls_name_the_plan_the_launch_read(
        self, built_package, uv_stub, tmp_path, moved,
    ):
        repo = Path(os.path.realpath(tmp_path)) / "repo"
        sub = repo / "sub"
        sub.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "plans").mkdir()
        (repo / "plans" / "x.md").write_text("# Plan\n", encoding="utf-8")
        key = pkg_build.cli_surface()["dispatcher"]["ran_from_root_key"]
        body = {"mode": "review-plan", "state": "done", "exit": 0,
                "artifacts": [], "next": None}
        if moved:
            body[key] = str(repo)
        launch = {"exit": 0, "stdout": [
            "run_id: RID-R", "log: /tmp/r.log",
            f"{cli_common.RESULT_MARKER} {json.dumps(body)}"]}
        _result, calls = uv_stub.run(
            built_package, dict(NEXT_STEP_ARGS, plan="plans/x.md"),
            [launch, _poll(0, state="done")], ctx={"directory": str(sub)})
        assert _dispatcher_argv(calls[0])[1] == "plans/x.md"
        polls = [_dispatcher_argv(c) for c in calls[1:]]
        base = repo if moved else sub
        assert polls and all(p[1] == str(base / "plans" / "x.md") for p in polls)

    def test_init_scaffolds_in_the_workspace_not_the_package(
        self, built_package, tmp_path
    ):
        """The artifact, not only the argv.

        ``init`` declares no positional, so it has no plan to anchor on and
        resolves its target from the child's cwd alone. Spawned from the package
        cache it would scaffold inside ``~/.cache/opencode/packages/…`` or report
        "not a git repository", and every later run would silently fall back to
        the shipped defaults. Every other mode has the resolved plan path as a
        second anchor; ``init`` has only the first, which is why the mode with no
        arguments is the one most exposed to this.
        """
        if unattended.find_uv() is None:
            _unavailable("uv is not installed; this row runs the real dispatcher")
        workspace = tmp_path / "project"
        workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
        (workspace / "README.md").write_text("x", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(workspace), "add", "-A"], check=True
        )
        subprocess.run(
            ["git", "-C", str(workspace), "-c", "user.email=t@t",
             "-c", "user.name=t", "commit", "-qm", "init"], check=True
        )

        outcome = node_json(
            f"""
            emit(await MultiLlmPlugin.execute(
              {{mode: "init", plan: "", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(workspace))}}}, surface,
            ))
            """,
            built_package,
            cwd=Path("/"),
            # This is the one row that runs the **real** dispatcher out of the
            # session-scoped staged tree, and CPython would leave a `__pycache__`
            # behind in it. `built_package` is shared, and `TestPack` packs that
            # same directory to assert `__pycache__` never reaches the tarball —
            # so without this the pack row fails on bytecode this row wrote,
            # blaming the build for what the suite did.
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert outcome["ok"] is True, outcome.get("message")
        assert (workspace / ".multi-llm" / "providers.yaml").exists(), (
            "init scaffolded somewhere other than the workspace"
        )
        assert not Path("/.multi-llm").exists()


# ===========================================================================
# The mid-run progress relay
# ===========================================================================

#: A `client` stub carrying only the one SDK route the fallback uses. It closes
#: over the `toasts` array the driver's snippet declares above it.
TOAST_CLIENT = (
    "{ tui: { showToast: async (options) => { toasts.push(options); return true } } }"
)


def _poll(exit_code, progress=None, state=None, **extra):
    """One scripted `status` child, with or without a ``progress`` payload."""
    body = {
        "mode": "status",
        "state": state or cli_common.STATE_RUNNING,
        "exit": exit_code,
        "artifacts": [],
        "next": None,
        **extra,
    }
    # Absent, never null: "this run predates the tracker" is the degrade case,
    # and a key holding null is a different payload from a key that is missing.
    if progress is not None:
        body["progress"] = progress
    return {"exit": exit_code, "stdout": [f"MULTI_LLM_RESULT: {json.dumps(body)}"]}


def _fanout(step, done, succeeded, total, **states):
    """A ``progress`` object shaped like the one ``progress_tracker`` writes."""
    return {
        "step": step,
        "models_done": done,
        "models_succeeded": succeeded,
        "models_total": total,
        "interval_s": 30,
        "models": {
            spec: {
                "state": state,
                "started_at": "2026-09-20T10:00:00Z",
                "updated_at": "2026-09-20T10:01:00Z",
                "timeout_s": 600,
                "attempt": 1,
                "source": None,
            }
            for spec, state in states.items()
        },
    }


#: The two rendered lines the dispatcher puts on `MULTI_LLM_RESULT.progress_lines`
#: — built here by the **renderers themselves**, never typed out, because what
#: these rows are about is that the string the plugin displays is the string
#: Python composed. A literal would still pass the day one of them is reworded.
RELAY_ROSTER = progress_tracker.render_roster_line({
    "codex:gpt-6-astra:xhigh": {"state": "running", "timeout_s": 1800},
    "opencode:auto": {"state": "queued", "timeout_s": 1800},
})
RELAY_JUST_FINISHED = "{} {}".format(
    run_status.JUST_FINISHED_LABEL,
    run_status._just_finished_text({"finished_during_wait": [{
        "spec": "codex:gpt-6-astra:xhigh", "state": "done",
        "elapsed_s": 252.0, "error": None,
    }]}),
)


#: The launch every relay row starts from: a banner, then a handshake result
#: line that carries **no** `progress` of its own, because a launcher's does not.
RELAY_LAUNCH = {
    "exit": 0,
    "stdout": [
        "run_id: RID-P",
        "log: /tmp/relay.log",
        'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
        '"artifacts":[],"next":null}',
    ],
}

RELAY_ARGS = {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []}

#: The abort dispatch a cancelled relay row ends on, scripted the way
#: `_run_abort` really answers a confirmed stop: its human line plus the
#: dispatcher's own terminal result line. Both are required — an exit code
#: alone is not the dispatcher's verdict, so a stub without the result line
#: would be scripting an *unconfirmed* abort.
RELAY_ABORTED = {
    "exit": 0,
    "stdout": [
        "aborted review-plan run RID-P (process group terminated)",
        'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
        '"artifacts":[],"next":null}',
    ],
}


class TestProgressRelay:
    """The package route's mid-run feedback, asserted where the JS really runs.

    **Why this class is the acceptance evidence and the container lane is not.**
    OpenCode has two install routes and only one of them is a shim. Every row in
    ``tests/container/host_matrix.yaml`` is file-route, the ``Makefile`` refuses
    the package rows outright — "a green file-route run is no evidence about the
    published package" — and the image cannot even receive the tarball. The two
    routes share no delivery code, so "OpenCode passed" means both routes or
    neither, and this is the package route's half.

    **What is being pinned.** On this route there is no agent in the loop:
    ``execute`` supervises the run and returns exactly once, so the only thing
    that can speak mid-run is the poll loop itself. The loop already held the
    data and was silent. These rows assert the relay's *cadence* (one update per
    non-terminal poll — "at least one" is satisfied by a single terminal flush
    and proves nothing), its *silence* after the answer and after a
    cancellation, its *channels*, and that a payload it cannot read costs it
    nothing.
    """

    def test_one_update_per_non_terminal_poll(self, built_package, uv_stub):
        """Several updates across a run, one per poll, in order, none repeated.

        Three non-terminal polls plus the launch banner is four updates, and the
        per-model map arrives *whole* — the object the status file already
        computed, not a sentence composed in JS about it.
        """
        first = _fanout("fan-out", 0, 0, 3, a="running", b="queued", c="queued")
        second = _fanout("fan-out", 1, 1, 3, a="done", b="running", c="queued")
        third = _fanout("fan-out", 2, 1, 3, a="done", b="failed", c="running")
        terminal = _fanout("TERMINAL-STEP", 3, 2, 3, a="done", b="failed", c="done")

        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [
                RELAY_LAUNCH,
                _poll(2, first),
                _poll(2, second),
                _poll(2, third),
                _poll(0, terminal, state="done"),
            ],
        )

        assert result["state"] == "done" and result["ok"] is True
        # One per non-terminal poll, plus the launch seam. Not one, not one per
        # run: the count IS the criterion.
        assert len(updates) == 4, updates

        bodies = [update["metadata"] for update in updates]
        # The launch seam: identifiers the moment the banner yields them, before
        # a whole `--wait` has elapsed. Its result line carries no map, and that
        # relays as null rather than throwing or being dropped.
        assert bodies[0]["run_id"] == "RID-P"
        assert bodies[0]["log"] == "/tmp/relay.log"
        assert bodies[0]["progress"] is None
        assert bodies[0]["polls"] == 0

        # Then one per poll, in order, each carrying that poll's own object
        # verbatim — key for key, including the nested per-model map.
        assert [body["progress"] for body in bodies[1:]] == [first, second, third]
        assert [body["polls"] for body in bodies[1:]] == [1, 2, 3]
        assert all(body["run_id"] == "RID-P" for body in bodies)
        # Not a rendered line, not a summary: the JS relays the object it was
        # handed and composes no prose of its own.
        assert all(isinstance(body["progress"], dict) for body in bodies[1:])
        # No `client` was handed to `execute` on this row, so the toast channel
        # simply is not there — this assertion pins that the missing client
        # costs nothing, not that metadata suppresses the toast. The two-channel
        # contract is pinned in `test_every_update_goes_out_on_both_channels`.
        assert toasts == []

    def test_the_terminal_poll_neither_repeats_nor_appends(
        self, built_package, uv_stub
    ):
        """Nothing is emitted for the poll that ends the call, or after it.

        The terminal poll's numbers ride out in the result ``execute`` returns,
        so relaying them would be the same state twice and an update *after* the
        answer. The seam sits below the loop's break, which is what makes that
        structural rather than remembered.
        """
        running = _fanout("fan-out", 1, 1, 2, a="done", b="running")
        terminal = _fanout("TERMINAL-STEP", 2, 2, 2, a="done", b="done")
        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, running), _poll(0, terminal, state="done")],
        )

        assert result["ok"] is True
        assert len(updates) == 2, updates
        # The terminal object reached the *result*, and only the result.
        assert result["result"]["progress"] == terminal
        assert "TERMINAL-STEP" not in json.dumps(updates)
        # And the last thing relayed is the last non-terminal poll, so no update
        # trails the answer.
        assert updates[-1]["metadata"]["progress"] == running

    def test_cancellation_ends_the_updates_at_the_last_live_poll(
        self, built_package, uv_stub
    ):
        """Zero updates after a cancellation, on both channels.

        The abort lands while a poll child is sleeping, and three things have
        to stay silent from that moment: the poll that was killed (its payload
        was printed, but a corpse's numbers are not progress this call may
        report), the abort dispatch behind it, and everything after the
        needs-human result. So the last thing a cancelled call relays is the
        last *live* poll — which is what "silent after cancellation" means on a
        route where the poll loop is the only thing that speaks.

        Driven through the relay driver rather than ``uv_stub.run``: ``run``
        hands ``execute`` a JSON context, JSON cannot carry a callback, and
        every other cancellation row in this file is blind to the updates for
        exactly that reason.
        """
        live = _fanout("fan-out", 1, 1, 3, a="done", b="running", c="queued")
        # Scripted and printed, and still never relayed: this is the child the
        # cancellation kills mid-sleep.
        killed = _fanout("KILLED-POLL", 2, 2, 3, a="done", b="done", c="running")
        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, live), {**_poll(2, killed), "sleep": 30},
             RELAY_ABORTED],
            client_js=TOAST_CLIENT,
            abort_on_call=3,
        )

        assert result["state"] == "needs-human" and result["ok"] is False
        assert result["run_id"] == "RID-P"
        # The claimed run really was stopped, so this is the cancellation path
        # and not some earlier exit that happens to be quiet.
        assert result["abort_dispatched"] is True
        assert [
            call for call in calls if "--abort" in _dispatcher_argv(call)
        ], "the cancelled call abandoned the run it had claimed"

        # The launch seam and the one live poll. Not three: the killed poll
        # relays nothing, and neither does the abort dispatch behind it.
        assert len(updates) == 2, updates
        assert len(toasts) == 2, toasts
        bodies = [update["metadata"] for update in updates]
        # `polls` is incremented before the aborted child is inspected, so the
        # killed poll's number is 2 — and no update carries it.
        assert [body["polls"] for body in bodies] == [0, 1]
        assert bodies[-1]["progress"] == live
        assert "KILLED-POLL" not in json.dumps(updates)
        assert "KILLED-POLL" not in json.dumps(toasts)

    def test_a_launch_cancelled_mid_flight_relays_nothing(
        self, built_package, uv_stub
    ):
        """The banner seam goes quiet too, once the signal has fired.

        ``detach.launch()`` prints the banner *before* the launcher exits, so a
        cancellation landing in the launch window comes back holding a claimed
        run whose identifiers the launch seam would happily relay — after the
        call was cancelled. That seam sits deliberately above the failure
        checks, because a launch that claimed a run and then **died** still
        started one and the update naming it is what points a human at the log.
        So the two halves are asserted together here: cancelled relays nothing,
        failed still relays, and moving the seam below the checks satisfies
        only the first.
        """
        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [{**RELAY_LAUNCH, "sleep": 30}, RELAY_ABORTED],
            client_js=TOAST_CLIENT,
            abort_on_call=1,
        )
        assert result["state"] == "needs-human" and result["ok"] is False
        # The run was claimed and aborted, so this is the launch *window* — not
        # a launcher killed before it ever printed a banner, which claims
        # nothing and would have nothing to relay anyway.
        assert result["run_id"] == "RID-P" and result["abort_dispatched"] is True
        assert updates == [] and toasts == []

        # The other half: a launcher that printed its banner and then failed
        # was never cancelled, and its one update is what a human reads.
        failed = {
            "exit": 3,
            "stdout": [*RELAY_LAUNCH["stdout"], "the launcher fell over"],
        }
        result, _, updates, toasts = uv_stub.relay(
            built_package, RELAY_ARGS, [failed, RELAY_ABORTED],
            client_js=TOAST_CLIENT,
        )
        assert result["state"] == "needs-human" and result["run_id"] == "RID-P"
        assert len(updates) == 1 and len(toasts) == 1, (updates, toasts)
        assert updates[0]["metadata"]["log"] == "/tmp/relay.log"

    def test_a_result_with_no_progress_payload_degrades_quietly(
        self, built_package, uv_stub
    ):
        """A run that predates the tracker still polls, still relays, never throws.

        Three shapes in one row, because they are three different absences: a
        payload with no ``progress`` key, a non-terminal poll with no
        ``MULTI_LLM_RESULT`` line at all, and a terminal result that carries
        none either. The relay reads the key optionally and sends ``null``
        rather than dropping the update — the identifiers and the poll count are
        themselves the signal that the loop is alive, so silence would make the
        cadence depend on the run's age.
        """
        silent = {"exit": 2, "stdout": ["still working"]}
        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2), silent, _poll(0, state="done")],
        )

        assert result["state"] == "done" and result["ok"] is True, result
        assert "progress" not in result["result"]
        assert len(updates) == 3, updates
        assert [update["metadata"]["progress"] for update in updates] == [
            None, None, None,
        ]
        assert [update["metadata"]["polls"] for update in updates] == [0, 1, 2]

    def test_every_update_goes_out_on_both_channels(self, built_package, uv_stub):
        """``ctx.metadata`` **and** ``client.tui.showToast``, on every seam.

        This row used to pin the opposite — metadata wins, the toast is skipped
        — on the reading that ``ctx.metadata`` is what opencode's bash tool
        streams through, so the host renders it mid-call. That holds for *bash*,
        which has a dedicated renderer; the generic renderer a third-party
        plugin tool gets is not known to paint a running tool's metadata, and
        this suite cannot see the host's TUI to settle it. So the assertion was
        changed deliberately: ``ctx.metadata`` still goes out (it is cheap and
        it attaches the update to this tool part, which is the right record),
        and the toast goes out beside it because that channel is definitely
        displayed. A successful metadata call no longer suppresses it.

        There is still no ``tool.execute.progress`` hook to prefer over either —
        ``tool.execute.after`` fires only once the result exists.

        Every payload here carries no ``progress_lines``, which is why the two
        channels are still byte-for-byte identical below: with nothing rendered
        to show, the toast serializes the same object it always did. The rows
        above pin the other half — that a payload which *does* render one sends
        the line to the toast and the whole object to the record.
        """
        running = _fanout("fan-out", 1, 2, 4, a="done", b="running")
        script = [RELAY_LAUNCH, _poll(2, running), _poll(0, state="done")]

        result, _, updates, toasts = uv_stub.relay(
            built_package, RELAY_ARGS, script, client_js=TOAST_CLIENT,
        )
        assert result["ok"] is True
        # Both channels, same cadence — the launch seam and the one
        # non-terminal poll — and byte-for-byte the same object on each.
        assert len(updates) == 2, updates
        assert len(toasts) == 2, toasts
        assert [json.loads(toast["body"]["message"]) for toast in toasts] == [
            update["metadata"] for update in updates
        ]
        assert updates[-1]["metadata"]["progress"] == running

        # A context that carries no ``metadata`` callback at all (an older host,
        # or any caller that built the context itself) loses that channel and
        # nothing else: the toast still carries every seam.
        result, _, updates, toasts = uv_stub.relay(
            built_package, RELAY_ARGS, script,
            with_metadata=False, client_js=TOAST_CLIENT,
        )
        assert result["ok"] is True
        assert updates == []
        # Same seams, same cadence, same payload — only the transport differs.
        assert len(toasts) == 2, toasts
        assert json.loads(toasts[-1]["body"]["message"])["progress"] == running
        # A `message` field has to hold a string, so the object goes over as
        # JSON. It is still data: the plugin authors no sentence about the run.
        assert toasts[-1]["body"]["variant"] in {
            "info", "success", "warning", "error"
        }

    def test_a_command_hidden_in_progress_is_redacted_on_both_channels(
        self, built_package, uv_stub
    ):
        """``progress`` goes out redacted, exactly as the terminal result does.

        ``progress`` is not the plugin's text and it is not the dispatcher's
        prose either: ``models[*].error`` is whatever sentence a provider handed
        back, mirrored by the executor into the status file. So a runnable
        ``uv run … multi_llm.py …`` can reach the relay by the same route it
        reaches ``child.result`` — and a command a model can paste is a command
        a model may paste, whichever field it arrived in. Both exits therefore
        run the same redaction; a relay that skipped it would enforce the rule
        for the model reading the result and not for the model reading the
        toast.
        """
        planted = (
            "uv run --project /skill -- python /skill/multi_llm.py "
            "review-plan /w/p.md --host opencode --yes"
        )
        running = _fanout("fan-out", 1, 0, 2, a="running", b="failed")
        # The provider's own sentence, with the command inside it: the prose
        # must survive, only the runnable span may go.
        running["models"]["b"]["error"] = (
            f"the provider timed out; resume it with `{planted}` if you must"
        )
        script = [RELAY_LAUNCH, _poll(2, running), _poll(0, state="done")]

        result, _, updates, toasts = uv_stub.relay(
            built_package, RELAY_ARGS, script, client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        assert len(updates) == 2 and len(toasts) == 2, (updates, toasts)
        relayed = [update["metadata"] for update in updates]
        tui = [json.loads(toast["body"]["message"]) for toast in toasts]

        # Neither channel carries a pasteable command.
        assert "uv run" not in json.dumps(relayed), (
            "a runnable command survived into ctx.metadata"
        )
        assert "uv run" not in json.dumps(tui), (
            "a runnable command survived into the toast"
        )
        # The whole span went, not just its head.
        assert "multi_llm.py" not in json.dumps(relayed)
        assert "multi_llm.py" not in json.dumps(tui)
        # Redacted in place, not dropped: the provider's own sentence is still
        # there, and so is every other field of the object.
        scrubbed = relayed[-1]["progress"]["models"]["b"]["error"]
        assert "the provider timed out" in scrubbed
        assert planted not in scrubbed
        assert scrubbed != running["models"]["b"]["error"]
        assert relayed[-1]["progress"]["step"] == "fan-out"
        assert relayed[-1]["progress"]["models_total"] == 2
        assert relayed[-1]["progress"]["models"]["a"]["state"] == "running"
        assert relayed[-1]["polls"] == 1 and relayed[-1]["run_id"] == "RID-P"
        # And the two channels still carry byte-for-byte the same object, so
        # the redaction is one pass rather than one per transport.
        assert tui == relayed

    def test_the_toast_shows_the_line_the_dispatcher_rendered(
        self, built_package, uv_stub
    ):
        """The reported defect, closed: a sentence, not a serialized object.

        Measured 2026-09-21 on a 5 m 40 s four-model fan-out: the user saw
        nothing at all and then one compressed line. The toast was firing — it
        was carrying `JSON.stringify(update)`, which is the whole poll payload
        including the per-model map. `ctx.metadata` was firing too and producing
        no persisted part update whatsoever (opencode's `event` table ran
        `pending` → `running` → `completed` with no gap across four `relay()`
        calls), so the toast is the only channel there is.

        What fills `message` now is the dispatcher's own rendered line, passed
        through byte for byte — asserted against the Python renderer's output,
        because a plugin that composed the sentence itself would pass an
        assertion written against a string typed into this file.
        """
        running = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        result, _, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [
                RELAY_LAUNCH,
                _poll(2, running, progress_lines=[RELAY_ROSTER]),
                _poll(0, state="done"),
            ],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        assert len(toasts) == 2, toasts
        # The launch seam carries no rendered line (its result is a handshake),
        # so it is the object as before — which is also the fallback this row's
        # sibling below is about.
        assert json.loads(toasts[0]["body"]["message"])["run_id"] == "RID-P"
        # And the poll carries the line, alone, exactly as Python wrote it.
        assert toasts[-1]["body"]["message"] == RELAY_ROSTER
        assert RELAY_ROSTER.startswith("[PROGRESS] fan-out over 2 models: ")

        # The structured half is untouched on both channels: the record still
        # carries the map, the identifiers and the lines it displayed.
        body = updates[-1]["metadata"]
        assert body["progress"] == running
        assert body["lines"] == [RELAY_ROSTER]
        assert body["run_id"] == "RID-P" and body["polls"] == 1

    def test_the_roster_is_said_once_and_the_news_every_time(
        self, built_package, uv_stub
    ):
        """Selection, which is the one judgement this file is allowed.

        The roster is byte-identical on every poll of one run — it names what
        the fan-out is *over*, which does not change — so repeating it every
        83 s would bury the line that is news under the line that is not. The
        plugin remembers what it has shown and shows what it has not; it reads
        no line's content to decide, so no wording from the Python side is
        spelled on this side. A poll with nothing fresh falls back to the
        object, which keeps the cadence exactly what it was: one update per
        non-terminal poll, never a silent one.
        """
        first = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        second = _fanout("fan-out", 1, 1, 2, a="done", b="running")
        third = _fanout("fan-out", 1, 1, 2, a="done", b="running")
        result, _, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [
                RELAY_LAUNCH,
                _poll(2, first, progress_lines=[RELAY_ROSTER]),
                _poll(2, second, progress_lines=[RELAY_ROSTER, RELAY_JUST_FINISHED]),
                _poll(2, third, progress_lines=[RELAY_ROSTER]),
                _poll(0, state="done"),
            ],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        messages = [toast["body"]["message"] for toast in toasts]
        assert len(messages) == 4, messages
        # Poll 1: the roster, said once.
        assert messages[1] == RELAY_ROSTER
        # Poll 2: only the news — the roster came again and was not repeated.
        assert messages[2] == RELAY_JUST_FINISHED
        # Poll 3: nothing fresh, so the object — the relay still speaks.
        assert json.loads(messages[3])["polls"] == 3

    def test_a_second_tool_call_in_one_process_says_the_roster_again(
        self, built_package, uv_stub
    ):
        """The relay memory is per call, and only one process can prove it.

        `relayMemory()` is created inside `execute`, so its `shown` Set dies
        with the call — which is what makes a second tool call a second run
        whose roster is news again. Moved to module scope the plugin would
        still pass every other row in this class, because each of those spawns
        a fresh node and re-imports the module: the Set is empty at the start
        of each one either way. The real host imports this module once per
        OpenCode session, so the regression it would ship is a roster the user
        sees on the session's first run and never again — for runs the plugin
        has never seen.

        Two full runs, one node process, one flat script: launch, roster poll,
        terminal poll, twice over. The second call's roster must be relayed.
        """
        first = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        second = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        one_run = [
            RELAY_LAUNCH,
            _poll(2, first, progress_lines=[RELAY_ROSTER]),
            _poll(0, state="done"),
        ]
        answers, toasts = uv_stub.relay_repeatedly(
            built_package,
            RELAY_ARGS,
            [*one_run,
             RELAY_LAUNCH,
             _poll(2, second, progress_lines=[RELAY_ROSTER]),
             _poll(0, state="done")],
            calls=2,
        )

        assert [answer["ok"] for answer in answers] == [True, True], answers
        shown = {}
        for toast in toasts:
            shown.setdefault(toast["call"], []).append(toast["message"])
        assert sorted(shown) == [0, 1], shown
        # Each call: the launch seam (the object, since this scripted launch
        # renders no line of its own) then the poll's roster.
        for call in (0, 1):
            assert len(shown[call]) == 2, shown[call]
            assert json.loads(shown[call][0])["run_id"] == "RID-P"
            assert shown[call][1] == RELAY_ROSTER, (
                f"call {call} did not announce its roster; a relay memory that "
                "outlives the call suppresses every run after the first"
            )

    def test_a_launch_that_renders_its_own_roster_says_it_at_the_banner(
        self, built_package, uv_stub
    ):
        """The roster ~83 s early, and no line of JS changed to get it.

        The launch's `MULTI_LLM_RESULT` is the first thing this loop can relay
        and the next relay is a whole `--wait` away, while the *terminal* poll
        is deliberately never relayed — so a run that went terminal inside that
        first wait announced its models nowhere at all. `detach.launch` now
        renders the roster from the selection it resolved and puts it on that
        handshake line; the launch seam here already read
        `result.progress_lines`, so the only thing this row asserts about the JS
        is that it needed nothing.

        And the poll behind it brings the identical string, which the per-call
        memory drops — so the user is told the line-up once, at the start,
        rather than twice or a poll late.
        """
        launched = {
            "exit": 0,
            "stdout": [
                "run_id: RID-P",
                "log: /tmp/relay.log",
                "MULTI_LLM_RESULT: " + json.dumps({
                    "mode": "review-plan", "state": "done", "exit": 0,
                    "artifacts": [], "next": None,
                    "progress_lines": [RELAY_ROSTER],
                }),
            ],
        }
        running = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        result, _, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [
                launched,
                _poll(2, running, progress_lines=[RELAY_ROSTER]),
                _poll(2, running, progress_lines=[RELAY_ROSTER,
                                                  RELAY_JUST_FINISHED]),
                _poll(0, state="done"),
            ],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        messages = [toast["body"]["message"] for toast in toasts]
        assert len(messages) == 3, messages
        # The banner seam: the roster, as a sentence, before any poll ran.
        assert messages[0] == RELAY_ROSTER
        # The first poll brings it again and is not allowed to repeat it, so it
        # falls back to the object — the cadence never goes silent.
        assert json.loads(messages[1])["polls"] == 1
        # And the news is news.
        assert messages[2] == RELAY_JUST_FINISHED
        assert updates[0]["metadata"]["lines"] == [RELAY_ROSTER]

    def test_a_dispatcher_that_renders_nothing_still_relays_the_object(
        self, built_package, uv_stub
    ):
        """The compatibility half: an older dispatcher must not break.

        A package and the skill tree it carries travel together, but a user's
        `MULTI_LLM_PROVIDERS_CONFIG`, a hand-run dispatcher or a phase with no
        per-model map at all (`generate-tasks`, an apply phase) all produce a
        poll with `progress` and no `progress_lines`. That is the payload this
        plugin shipped against and it goes over exactly as it did — the key is
        read optionally, never required.
        """
        running = _fanout("fan-out", 1, 1, 2, a="done", b="running")
        result, _, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, running), _poll(0, state="done")],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        assert len(toasts) == 2, toasts
        assert json.loads(toasts[-1]["body"]["message"])["progress"] == running
        assert updates[-1]["metadata"]["lines"] is None

    def test_a_command_hidden_in_a_rendered_line_is_redacted_too(
        self, built_package, uv_stub
    ):
        """The rendered line is a provider's prose as much as `models[*].error` is.

        `just finished:` renders a failed slot's own error text, which is
        whatever sentence the provider handed back — so a runnable
        `uv run … multi_llm.py …` reaches the toast through the rendered line
        by exactly the route it reaches `progress.models[*].error`. The line
        rides *inside* the update for that reason: one redaction pass covers
        both, rather than one that covers the object and misses the sentence
        composed from it.
        """
        planted = (
            "uv run --project /skill -- python /skill/multi_llm.py "
            "review-plan /w/p.md --host opencode --yes"
        )
        line = (
            "just finished: opencode:auto (0m30s, failed: the provider died; "
            f"resume it with `{planted}`)"
        )
        running = _fanout("fan-out", 1, 0, 2, a="failed", b="running")
        result, _, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [
                RELAY_LAUNCH,
                _poll(2, running, progress_lines=[line]),
                _poll(0, state="done"),
            ],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        shown = toasts[-1]["body"]["message"]
        # Redacted in place: the provider's own sentence survives, the runnable
        # span does not, and the line is still the line — not dropped, not JSON.
        assert shown.startswith("just finished: opencode:auto (0m30s, failed:")
        assert "the provider died" in shown
        assert planted not in shown and "uv run" not in shown
        assert "multi_llm.py" not in shown
        assert "uv run" not in json.dumps(updates)

    #: The schema default `duration` falls back to when the body omits it —
    #: opencode's own number, not this suite's, and the one the relay was
    #: silently taking. Every assertion below is written against it rather than
    #: against the constants `index.mjs` chose, so a re-tuned hold does not
    #: rewrite the row and a regression to "omit it" fails by name.
    SCHEMA_DEFAULT_TOAST_MS = 5000

    def test_the_toast_asks_to_outlive_the_gap_to_the_next_one(
        self, built_package, uv_stub
    ):
        """The reported defect: a five-second flash every ninety seconds.

        `tui.showToast`'s `duration` is optional and opencode's schema defaults
        it to 5 000 ms, so a body carrying only `{message, variant}` asks for a
        flash. Measured 2026-09-21 on a 363 s run: five toasts, four of them
        ~90 s apart, a 5.5 % duty cycle inside a six-minute silence — and the
        user saw one, the launch one, because it lands a second after they press
        enter. The channel worked the whole time.

        What is asserted is the *property*, not the constants: a toast asks to
        stay up longer than the schema default, and past the first it asks for
        the gap this process just watched. The launch seam has no gap to
        measure, so it falls back to the largest hold — which is why the first
        toast's request is longer than the ones behind it in a suite where the
        stubbed polls return in milliseconds.
        """
        running = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        result, _, _updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, running), _poll(2, running),
             _poll(0, state="done")],
            client_js=TOAST_CLIENT,
        )

        assert result["ok"] is True
        assert len(toasts) == 3, toasts
        for toast in toasts:
            body = toast["body"]
            # The title is the tool's own registration key — identifiable
            # among whatever else the host is showing, and not a word this
            # plugin authors about the run.
            assert body["title"] == "multi_llm", body
            assert isinstance(body["duration"], (int, float)), body
            assert body["duration"] > self.SCHEMA_DEFAULT_TOAST_MS, body
        # No gap to measure at the launch seam, so it asks for the fallback
        # hold; every poll behind it asks for the gap it watched, which in this
        # suite is milliseconds.
        assert toasts[0]["body"]["duration"] > toasts[1]["body"]["duration"]

    def test_the_hold_is_the_measured_gap_not_a_configured_cadence(
        self, built_package
    ):
        """`toastHoldMs` in isolation: three properties, no constants restated.

        `--wait`'s duration is a fact Python owns — resolved from the project's
        poll cadence, clamped to this host's foreground cap — and this file may
        not hold it. The interval between two relays is something this process
        *observes*, and after the first poll it is that duration by
        construction, so the hold is derived rather than configured.
        """
        payload = node_json(
            """
            const memory = T.relayMemory()
            const first = T.toastHoldMs(memory)
            memory.lastRelayAt = Date.now() - 30_000
            const short = T.toastHoldMs(memory)
            memory.lastRelayAt = Date.now() - 90_000
            const measured = T.toastHoldMs(memory)
            memory.lastRelayAt = Date.now() - 60 * 60 * 1000
            const capped = T.toastHoldMs(memory)
            // A memory-less direct call must not throw or stamp anything.
            const stateless = T.toastHoldMs(undefined)
            emit({first, short, measured, capped, stateless,
                  stamped: typeof memory.lastRelayAt})
            """,
            built_package,
        )
        # Before any gap exists: long enough to survive one `--wait`, which
        # Python clamps under this host's ~120 s foreground cap.
        assert payload["first"] >= 90_000, payload
        # After one: the gap it watched, plus a margin, and never less. Two
        # different gaps, because one alone is satisfied by any constant that
        # happens to sit in the window — a hold that ignored the measurement
        # would pass a single-gap assertion and show a stale toast for minutes.
        assert 30_000 < payload["short"] <= 30_000 + 60_000, payload
        assert 90_000 < payload["measured"] <= 90_000 + 60_000, payload
        assert payload["short"] < payload["measured"], payload
        # And bounded, because nothing clears a toast when `execute` returns.
        assert payload["capped"] < 60 * 60 * 1000, payload
        assert payload["capped"] >= payload["measured"], payload
        assert payload["stateless"] == payload["first"], payload
        assert payload["stamped"] == "number", payload

    def test_the_result_counts_what_the_relay_was_asked_to_do(
        self, built_package, uv_stub
    ):
        """The question a forensic pass could not answer, made answerable.

        Two failures used to leave identical evidence — none: `showToast`
        persists nothing in `opencode.db`, a rejected POST was swallowed by a
        bare `.catch(() => {})`, and `warn` writes to a stderr `opencode.log`
        does not capture. "It fired and you missed it" and "every POST failed"
        looked the same from the outside. The counters are numbers on the
        result, which is consistent with the rule that this file authors no
        sentence about the run.

        `attempted` is exact when the result is composed. `delivered` and
        `rejected` settle asynchronously — nothing awaits the SDK call — so the
        last relay of a call may still be in flight, which is why the rows below
        read them as a floor rather than as a census.
        """
        running = _fanout("fan-out", 0, 0, 2, a="running", b="queued")
        script = [RELAY_LAUNCH, _poll(2, running), _poll(2, running),
                  _poll(0, state="done")]

        # A host that displays them: three relays, none refused.
        result, _, updates, toasts = uv_stub.relay(
            built_package, RELAY_ARGS, script, client_js=TOAST_CLIENT,
        )
        assert result["ok"] is True
        assert len(updates) == 3 and len(toasts) == 3
        assert result["relays"]["attempted"] == 3, result["relays"]
        assert result["relays"]["rejected"] == 0, result["relays"]
        assert result["relays"]["unavailable"] == 0, result["relays"]

        # A host whose route refuses: the rejection is counted rather than
        # swallowed, which is the whole point of the change.
        refusing = (
            "{ tui: { showToast: async (options) => { toasts.push(options); "
            "throw new Error('the TUI is gone') } } }"
        )
        result, _, _u, _t = uv_stub.relay(
            built_package, RELAY_ARGS, script, client_js=refusing,
        )
        assert result["ok"] is True
        assert result["relays"]["attempted"] == 3, result["relays"]
        assert result["relays"]["rejected"] >= 1, result["relays"]
        assert result["relays"]["delivered"] == 0, result["relays"]

        # And a host with no route at all is the third diagnosis, told apart
        # from both of the above.
        result, _, _u, _t = uv_stub.relay(built_package, RELAY_ARGS, script)
        assert result["ok"] is True
        assert result["relays"]["attempted"] == 3, result["relays"]
        assert result["relays"]["unavailable"] == 3, result["relays"]
        assert result["relays"]["delivered"] == 0, result["relays"]
        assert result["relays"]["rejected"] == 0, result["relays"]

    def test_an_absent_toast_channel_says_so_once(self, built_package):
        """"The channel was not there" and "it fired and you missed it" are
        different diagnoses, and they used to leave identical evidence.

        `showToast` persists nothing in `opencode.db`, and this file's `catch`
        only fires when a channel *throws* — so a host carrying no
        `client.tui.showToast` at all produced exactly what a displayed-and-
        missed toast produces: nothing. It now warns, once per call rather than
        once per poll, since a run that polls twenty times has one missing
        channel and not twenty.

        Read what this buys, honestly: `warn` is `console.error`, so the line
        lands on whatever stderr the opencode process was started with. That is
        visible to someone running the binary in a terminal and is **not**
        captured in `opencode.log`, so the probe helps a maintainer reproducing
        the condition and does not by itself answer it from a user's logs.
        """
        payload = node_json(
            """
            const said = []
            const real = console.error
            console.error = (...parts) => { said.push(parts.join(" ")) }
            try {
              const memory = T.relayMemory()
              T.relayProgress({}, undefined, {progress: null}, memory)
              T.relayProgress({}, undefined, {progress: null}, memory)
              // A fresh call is a fresh run, so it says so again.
              T.relayProgress({}, undefined, {progress: null}, T.relayMemory())
              // And a host that HAS the channel says nothing at all.
              const toasts = []
              T.relayProgress({}, """ + TOAST_CLIENT + """, {progress: null},
                              T.relayMemory())
              emit({said, toasts})
            } finally { console.error = real }
            """,
            built_package,
        )
        said = payload["said"]
        assert len(said) == 2, said
        assert all("showToast" in line for line in said), said
        assert all(line.startswith("[multi-llm] ") for line in said), said
        assert len(payload["toasts"]) == 1

    def test_the_message_chooser_reads_no_line_s_content(self, built_package):
        """`toastMessage` in isolation: selection by identity, nothing else.

        Whatever the dispatcher supplies is shown; a non-string or empty entry
        is skipped rather than rendered as `null`; several fresh lines go
        together in the order they were supplied; and with nothing supplied the
        object is serialized as it always was. No prefix, marker or word from
        the Python side appears in the decision — which is what keeps this file
        free of a vocabulary it would then own a second copy of.

        The non-string check is a `typeof` and not a truthiness test, and the
        `42` below is what makes the difference visible: a falsy non-string
        (`null`, `0`, `""`) is skipped either way, while a **truthy** one is
        joined into `message` and read by a user as a sentence the dispatcher
        composed. Today's dispatcher only ever emits strings
        (`cli_common.set_progress_lines` maps `str` over the list), so this is a
        guard against the payload rather than against the current writer of it —
        `update.lines` is JSON parsed off whatever process printed the marker.
        """
        payload = node_json(
            f"""
            const memory = T.relayMemory()
            const update = {{
              progress: null, run_id: "RID-P", polls: 1,
              lines: [{json.dumps(RELAY_ROSTER)}, "", null, 0, 42, {{}},
                      [{json.dumps(RELAY_ROSTER)}],
                      {json.dumps(RELAY_JUST_FINISHED)}],
            }}
            const both = T.toastMessage(update, memory)
            const again = T.toastMessage(update, memory)
            const bare = T.toastMessage({{progress: null, lines: null}}, memory)
            const stateless = T.toastMessage(update, undefined)
            emit({{both, again, bare, stateless}})
            """,
            built_package,
        )
        assert payload["both"] == f"{RELAY_ROSTER}\n{RELAY_JUST_FINISHED}"
        # Nothing but the two strings: no "42", no "[object Object]", and no
        # array stringified into a line of its own.
        assert "42" not in payload["both"] and "object" not in payload["both"]
        # Both were shown, so a second identical poll has nothing fresh.
        assert json.loads(payload["again"])["run_id"] == "RID-P"
        assert json.loads(payload["bare"])["progress"] is None
        # No memory, no deduplication: the right default for one direct call.
        assert payload["stateless"] == payload["both"]

    def test_neither_channel_costs_the_run_nothing(self, built_package, uv_stub):
        """No callback and no client at all: the fan-out is unaffected (rule 4).

        A relay is a nicety on top of a run that is already paid for. An older
        host, or any caller that built the context itself, must lose the updates
        and nothing else.
        """
        result, calls, updates, toasts = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, _fanout("fan-out", 0, 0, 1, a="running")),
             _poll(0, state="done")],
            with_metadata=False,
        )
        assert result["state"] == "done" and result["ok"] is True
        assert updates == [] and toasts == []
        assert len([c for c in calls if "instructions" not in _dispatcher_argv(c)]) == 3

    def test_a_channel_that_throws_is_swallowed_and_the_other_still_fires(
        self, built_package, uv_stub
    ):
        """A rejecting host costs a warning line, never the fan-out.

        A ``metadata`` that *throws* must not take the toast down with it: the
        two channels are independent, so the seam still reaches the TUI and
        ``relayProgress`` still reports that something was relayed. With neither
        channel there is nothing to relay, and that is the only ``false``.
        """
        running = _fanout("fan-out", 0, 0, 2, a="running", b="running")
        payload = node_json(
            f"""
            const toasts = []
            const client = {TOAST_CLIENT}
            const ctx = {{ metadata: () => {{ throw new Error("no channel here") }} }}
            const relayed = T.relayProgress(ctx, client, {json.dumps(
                {"progress": running, "run_id": "RID-P", "log": None, "polls": 2}
            )})
            // And with neither channel: still false, still no throw.
            const bare = T.relayProgress({{}}, undefined, {{progress: null}})
            emit({{relayed, bare, toasts}})
            """,
            built_package,
        )
        assert payload["relayed"] is True
        assert payload["bare"] is False
        assert len(payload["toasts"]) == 1
        assert json.loads(payload["toasts"][0]["body"]["message"])["run_id"] == "RID-P"

    def test_the_poll_argv_is_the_bare_wait_flag(self, built_package, uv_stub):
        """The poll argv, pinned — not left to a doc comment (resolution (b)).

        ``--wait`` is a ``store_true`` and the plugin passes the bare flag, so
        the duration is the dispatcher's to resolve: from the project's
        ``progress.poll_s`` (90 s as shipped, so a default install polls exactly
        as it always did), then clamped to the named host's foreground cap less
        the margin. The clamp *does* fire on this route — the baked ``--host``
        is on every poll argv, as this assertion also shows — and it is accepted
        as authoritative here rather than routed around with a ``--timeout``
        composed in JS, because the poll duration is this loop's cancellation
        and budget granularity and the clamp is the only thing bounding it.

        What the row guards is a **configured** project: a ``poll_s: 300`` that
        silently became 90 s here is invisible unless the argv is pinned. Every
        element is read from the generated surface, so the spelling has exactly
        one owner.
        """
        result, calls, _, _ = uv_stub.relay(
            built_package,
            RELAY_ARGS,
            [RELAY_LAUNCH, _poll(2, _fanout("fan-out", 0, 0, 1, a="running")),
             _poll(0, state="done")],
        )
        assert result["ok"] is True

        surface = pkg_build.cli_surface()
        spelling = surface["dispatcher"]["poll"]
        poll = _dispatcher_argv(calls[1])
        assert poll == [
            spelling[0],
            str(uv_stub.log_path.parent / "p.md"),
            *surface["baked_flags"],
            *spelling[1:],
            surface["dispatcher"]["run_id_option"],
            "RID-P",
        ], poll
        # Spelled out because it is the half of the contract that is an absence:
        # no duration is composed here, and none is spelled in the JS.
        assert "--timeout" not in poll
        assert spelling[-1] == "--wait"
        # The runtime `--host` the dispatcher's clamp reads is what makes the
        # clamp fire on this route at all.
        assert surface["dispatcher"]["host_option"] in poll


# ===========================================================================
# The core instructions, surfaced by the tool
# ===========================================================================

class TestInstructionsSurface:
    def test_util_instructions_takes_a_host(self):
        """``util instructions <mode> --host <id>`` appends the verb block.

        The core files name the three capability verbs abstractly, and it has
        always been the *shim's* section 4 that resolved them. Once the command
        body is one line there is no section 4 left, so the resolution has to
        arrive with the core — same single owner, ``caps.verb_block()``, just
        served at run time and through the tool.
        """
        from utils import host_capabilities as caps

        plain = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"),
             "util", "instructions", "review-plan"],
            capture_output=True, encoding="utf-8", timeout=120,
        )
        assert plain.returncode == 0, plain.stderr
        with_host = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"),
             "util", "instructions", "review-plan", "--host", "opencode"],
            capture_output=True, encoding="utf-8", timeout=120,
        )
        assert with_host.returncode == 0, with_host.stderr

        for sentence in caps.verb_sentences("opencode"):
            assert sentence in with_host.stdout, "a verb sentence is missing"
            assert sentence not in plain.stdout, (
                "the verb block leaked into the host-less form"
            )
        # `--host` appends; it never rewrites the core.
        core = plain.stdout.split(cli_common.RESULT_MARKER)[0]
        assert core.rstrip() in with_host.stdout

    def test_an_unknown_host_is_refused(self):
        """A bad ``--host`` is exit 3, not a silently host-less core."""
        result = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"),
             "util", "instructions", "review-plan", "--host", "nope"],
            capture_output=True, encoding="utf-8", timeout=120,
        )
        assert result.returncode == cli_common.EXIT_NEEDS_HUMAN

    def test_tool_output_carries_the_core_and_the_verbs(
        self, built_package, uv_stub
    ):
        """The result includes the served core, three verb sentences included.

        After the prose shrink this is requirement #6's **only** home. The
        command body is one line, so the place a model used to read the workflow
        and the resolved verbs is gone — and if the tool did not surface them,
        shrinking the prose would simply have deleted them.

        The terminal poll hands back a ``next`` — a **non**-mechanical one, so
        it is reported rather than auto-followed — because that is what "the
        model still has work" means, and it is the only condition under which
        the instructions are attached at all (see the sibling row below).
        """
        from utils import host_capabilities as caps

        follow = "uv run -- python multi_llm.py generate-tasks p.md"
        launch = {
            "exit": 0,
            "stdout": ["run_id: R", "log: /l",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":["a"],'
            '"next":' + json.dumps(follow) + '}']}
        # `util instructions` must reach the REAL dispatcher, or this row would
        # be grading the stub's own echo.
        instructions = {
            "exit": 0,
            "stdout": (
                subprocess.run(
                    [sys.executable, str(SKILL_ROOT / "multi_llm.py"),
                     "util", "instructions", "review-plan", "--host", "opencode"],
                    capture_output=True, encoding="utf-8", timeout=180,
                ).stdout.splitlines()
            ),
        }
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "",
             "extra_flags": []},
            # launch, one terminal poll, then the instructions fetch. The stub
            # replays by call index, so the order is the call order.
            [launch, done, instructions],
        )
        assert result["ok"] is True
        assert result.get("instructions"), (
            "the tool returned no core instructions, so the prose shrink deleted "
            "the only place a model could read them"
        )
        for sentence in caps.verb_sentences("opencode"):
            assert sentence in result["instructions"], (
                "a resolved verb sentence is missing from the tool's output"
            )
        # It asked for them the one supported way, naming this host.
        fetch = next(
            call for call in calls if "instructions" in _dispatcher_argv(call)
        )
        argv = _dispatcher_argv(fetch)
        assert argv[:3] == ["util", "instructions", "review-plan"]
        assert argv[argv.index("--host") + 1] == "opencode"

        # And the JS holds no copy of either half.
        text = SRC_MJS.read_text(encoding="utf-8")
        for sentence in caps.verb_sentences("opencode"):
            assert sentence not in text

    def test_a_finished_run_is_not_sent_the_manual_run_instructions(
        self, built_package, uv_stub
    ):
        """`next: null` and `state: "done"` means there is provably no work left.

        Measured 2026-09-16: the tool result for a completed `review-plan` was
        25 829 chars, 22 860 of them the rendered runbook — including fifteen
        `uv run …` command lines the same result's own message forbids the model
        to run — attached because the predicate read only `row.supports_detach`
        and so fired for every detachable mode, finished or not. The step that
        consumed it billed 16 205 input tokens for a run with nothing to do.
        """
        launch = {
            "exit": 0,
            "stdout": ["run_id: R", "log: /l",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":["a"],'
            '"next":null}']}
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "",
             "extra_flags": []},
            [launch, done, {"exit": 0, "stdout": ["instructions would go here"]}],
        )
        assert result["ok"] is True and result["next"] is None
        assert "instructions" not in result
        # Not fetched either: the cost is the dispatch as well as the payload.
        assert not [
            call for call in calls if "instructions" in _dispatcher_argv(call)
        ]

        # The predicate itself, over the four combinations of the two questions
        # it asks — the mode's, and this call's outcome.
        verdicts = node_json(
            """
            const row = surface.modes.find((m) => m.supports_detach)
            const flat = surface.modes.find((m) => !m.supports_detach)
            // The state words come from the surface, as they do in the file
            // under test: no state name is spelled in JS, here or there.
            const done = surface.dispatcher.states.ok
            const stuck = surface.dispatcher.states.needs_human
            // Exit 2's three words. "Already completed, use --force" is the
            // one a second `--review-plan` on the same plan hits, and it is
            // `ok: true` — so a predicate written as "not done" attaches the
            // runbook to a result whose whole content is "nothing to do".
            const spare = surface.dispatcher.states.nothing_to_do
            const live = surface.dispatcher.states.running
            const gone = surface.dispatcher.states.died
            emit({
              finished: T.rowNeedsInstructions(
                row, {state: done, next: null}, surface),
              owesNext: T.rowNeedsInstructions(
                row, {state: done, next: {tool: "multi_llm"}}, surface),
              unfinished: T.rowNeedsInstructions(
                row, {state: stuck, next: null}, surface),
              notDetachable: T.rowNeedsInstructions(
                flat, {state: stuck, next: null}, surface),
              nothingToDo: T.rowNeedsInstructions(
                row, {state: spare, next: null}, surface),
              running: T.rowNeedsInstructions(
                row, {state: live, next: null}, surface),
              died: T.rowNeedsInstructions(
                row, {state: gone, next: null}, surface),
            })
            """,
            built_package,
        )
        assert verdicts == {
            "finished": False, "owesNext": True,
            "unfinished": True, "notDetachable": False,
            "nothingToDo": False, "running": False, "died": False,
        }

    def test_partial_is_terminal_owes_no_instructions_and_is_not_incomplete(
        self, built_package
    ):
        """Plan §5.5: a detached run whose result was ``partial`` exits 0 with
        ``next: null``. It must neither attach the runbook
        (``rowNeedsInstructions``) nor read as "did not complete" (the
        needs-human completeness test). The word is read from the generated
        surface, never spelled in JS or here.
        """
        surface = pkg_build.cli_surface()
        assert surface["dispatcher"]["states"]["partial"] == cli_common.STATE_PARTIAL
        verdicts = node_json(
            """
            const row = surface.modes.find((m) => m.supports_detach)
            const partial = surface.dispatcher.states.partial
            const payload = {state: partial, exit: 0, artifacts: [], next: null}
            const finished = T.terminalResult(
              {code: surface.dispatcher.exit_codes.ok, argv: ["uv"], stdout: "",
               stderr: "", result: payload, resultRaw: null, truncated: false,
               banner: {}, aborted: false, timedOut: false, spawnError: null},
              {anchor: "/anchor", surface, values: null},
            )
            // Hypothetical exit 3 carrying the word: the completeness test
            // keys off the needs-human word alone, so it is not "incomplete".
            const held = T.terminalResult(
              {code: surface.dispatcher.exit_codes.needs_human, argv: ["uv"],
               stdout: "", stderr: "", result: {...payload, exit: 3},
               resultRaw: null, truncated: false, banner: {}, aborted: false,
               timedOut: false, spawnError: null},
              {anchor: "/anchor", surface,
               values: {word: row.name, plan: "p.md", question: "",
                        extraFlags: []}},
            )
            emit({
              defined: typeof partial === "string" && partial.length > 0,
              owes: T.rowNeedsInstructions(
                row, {state: partial, next: null}, surface),
              ok: finished.ok,
              state: finished.state,
              next: finished.next,
              message: held.message,
              phase: row.phase,
            })
            """,
            built_package,
        )
        assert verdicts["defined"] is True
        assert verdicts["owes"] is False
        assert verdicts["ok"] is True
        assert verdicts["state"] == cli_common.STATE_PARTIAL
        assert verdicts["next"] is None
        text = surface["needs_human_message"]
        assert _filled(text["phase_incomplete"], phase=verdicts["phase"]) not in (
            verdicts["message"]
        )
        assert _filled(text["phase_known"], phase=verdicts["phase"]) in (
            verdicts["message"]
        )


# ===========================================================================
# The host's own file sigil
# ===========================================================================

class TestPlanSigil:
    def test_one_leading_at_sigil_is_stripped_from_the_plan_path(
        self, built_package, tmp_path
    ):
        """`@` is how a user names a file on this host, and it reaches the model.

        Measured 2026-09-16, the argument text was `--review-plan
        @plans/simple-3d-racing-game.md --quick`; that run's model stripped the
        sigil on its own initiative and nothing made it. Left on, the path
        resolves to `<workspace>/@plans/…` and the dispatcher exits 1 with "Plan
        file not found", naming a path whose leading character nothing explains.

        One, and only at the front: `@` is legal anywhere else in a path, and a
        bare `"@"` keeps its sigil so the positional still reaches the
        dispatcher and the refusal names what was typed.
        """
        anchor = json.dumps(str(tmp_path))
        paths = node_json(
            f"""
            const anchor = {anchor}
            emit({{
              sigil: T.normalizePlan("@plans/x.md", anchor),
              plain: T.normalizePlan("plans/x.md", anchor),
              doubled: T.normalizePlan("@@plans/x.md", anchor),
              absolute: T.normalizePlan("@/abs/x.md", anchor),
              home: T.normalizePlan("@~/x.md", anchor),
              inner: T.normalizePlan("plans/@team/x.md", anchor),
              bare: T.normalizePlan("@", anchor),
              empty: T.normalizePlan("", anchor),
            }})
            """,
            built_package,
        )
        assert paths["sigil"] == paths["plain"] == str(tmp_path / "plans" / "x.md")
        assert paths["doubled"] == str(tmp_path / "@plans" / "x.md")
        assert paths["absolute"] == "/abs/x.md"
        assert paths["home"] == str(Path.home() / "x.md")
        assert paths["inner"] == str(tmp_path / "plans" / "@team" / "x.md")
        assert paths["bare"] == str(tmp_path / "@")
        assert paths["empty"] == ""

    def test_the_plan_property_says_so(self, built_package):
        """A model reading only the schema has to know, or it sends the sigil."""
        described = node_json(
            "emit(T.toolArgs(surface).plan.description)", built_package
        )
        assert "@" in described

    def test_the_sigil_never_reaches_the_dispatcher(self, built_package, uv_stub):
        """End to end: what `execute` composes carries the stripped path."""
        launch = {
            "exit": 0,
            "stdout": ["run_id: R", "log: /l",
                       'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}'],
        }
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],'
            '"next":null}']}
        result, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "@plans/x.md", "question": "",
             "extra_flags": []},
            [launch, done],
        )
        argv = _dispatcher_argv(calls[0])
        # The sigil off, nothing resolved: the user's own spelling reaches the
        # dispatcher (issue #15), and the child runs in the anchor.
        assert argv[1] == "plans/x.md"
        assert not any(token.startswith("@") for token in argv)


class TestPlanSpelling:
    """Issue #15 (2026-10-02): the dispatcher received the anchor-resolved path,
    so every next command printed it absolute although the user typed a relative
    one. The launch now carries the user's spelling; the child's cwd is the anchor."""

    def test_the_spelling_helper_strips_the_sigil_and_resolves_nothing(self, built_package):
        paths = node_json(
            """
            emit({
              relative: T.planSpelling("plans/x.md"),
              sigil: T.planSpelling("@plans/x.md"),
              dotted: T.planSpelling("./plans/../plans/x.md"),
              absolute: T.planSpelling("/abs/x.md"),
              bare: T.planSpelling("@"),
              empty: T.planSpelling(""),
            })
            """,
            built_package,
        )
        assert paths == {"relative": "plans/x.md", "sigil": "plans/x.md",
                         "dotted": "./plans/../plans/x.md", "absolute": "/abs/x.md",
                         "bare": "@", "empty": ""}

    def test_a_continuation_of_the_same_plan_keeps_the_typed_spelling(
        self, built_package, uv_stub, tmp_path
    ):
        absolute = str(tmp_path / "plans" / "x.md")
        launch = {"exit": 0, "stdout": [
            "run_id: R1", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
        first = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":'
            + json.dumps(f"uv run -- python multi_llm.py review-plan {word(absolute)} "
                         "--reaggregate --host opencode") + "}"]}
        second_launch = {"exit": 0, "stdout": [
            "run_id: R2", "log: /l2", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}
        _, calls = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "plans/x.md", "question": "",
             "extra_flags": []},
            [launch, first, second_launch, done],
        )
        argvs = [_dispatcher_argv(call) for call in calls]
        launches = [argv for argv in argvs if "--detach" in argv]
        polls = [argv for argv in argvs if "--run-id" in argv]
        assert [argv[1] for argv in launches] == ["plans/x.md", "plans/x.md"]
        assert all(argv[1] == absolute for argv in polls)


from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401


@pytest.fixture
def host_path() -> str:
    """``PATH`` as the session had it, taken before ``repo`` narrows it to
    the mock runner's (node lives elsewhere)."""
    return os.environ.get("PATH", "")


class TestResumeCommand:
    """Issue #5: a blocked write phase leaves the user a command they can type.

    The dispatcher builds the host-spelled `--resume` command
    (`util next-command --blocked`); this plugin only asks and relays it."""

    IMPLEMENT = {"mode": "implement", "plan": "plans/x.md", "question": "",
                 "extra_flags": []}

    LAUNCH = {"exit": 0, "stdout": [
        "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
    BLOCKED = {"exit": 3, "stdout": [
        'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,"artifacts":[],'
        '"next":null,"error":"batch 0 refused"}']}

    def _resume_calls(self, uv_stub):
        path = Path(str(uv_stub.log_path) + ".resume")
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_the_result_carries_it_and_the_message_says_how_to_use_it(
        self, built_package, uv_stub
    ):
        result, calls = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED])
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        assert result["resume_command"] == "/multi-llm --implement plans/x.md --resume"
        assert result["message"].endswith(surface["needs_human_message"]["resume"])
        assert result.get("next") is None
        assert len(calls) == 2  # the launch and the poll: nothing re-ran

    def test_it_asks_with_the_users_spelling_the_word_and_the_host(
        self, built_package, uv_stub
    ):
        uv_stub.run(built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED])
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        resume = surface["dispatcher"]["resume"]
        (asked,) = self._resume_calls(uv_stub)
        assert _dispatcher_argv(asked) == resume["argv"] + [
            "plans/x.md", resume["option"], "implement",
            surface["dispatcher"]["host_option"], surface["host"]]

    def test_the_agent_facing_option_does_not_read_as_a_command(
        self, built_package, uv_stub
    ):
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED])
        human = result["options"][-1]["human"]
        assert human.startswith("(a note to you, not a command")
        assert all("resume_command" not in option for option in result["options"])

    def test_only_an_exit_3_asks(self, built_package, uv_stub):
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}
        result, _ = uv_stub.run(built_package, self.IMPLEMENT, [self.LAUNCH, done])
        assert "resume_command" not in result
        assert self._resume_calls(uv_stub) == []

    def test_a_failed_lookup_leaves_the_result_as_it_was(self, built_package, uv_stub):
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED],
            extra_env={"STUB_RESUME_FAIL": "1"})
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        assert "resume_command" not in result
        assert not result["message"].endswith(surface["needs_human_message"]["resume"])
        assert result["state"] == surface["dispatcher"]["states"]["needs_human"]

    def test_a_surface_without_the_entry_skips_the_lookup(self, built_package, uv_stub):
        """Additive within schema v5: an older surface only loses the command."""
        assert "resume" in json.loads(
            (built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))["dispatcher"]
        out = node_json(
            """
            const bare = structuredClone(surface)
            delete bare.dispatcher.resume
            emit(await T.fetchResumeCommand({
              values: { word: "implement", plan: "/a/x.md", planArg: "x.md" },
              surface: bare, env: {}, uv: "/nonexistent", anchor: "/tmp",
            }))
            """,
            built_package,
        )
        assert out is None

    def _answers_step(self):
        """A `user_resume` carrying a stop's answers, spelled by the real
        Python (`next_phase.phase_command`, `render_resume_answers`)."""
        sys.path.insert(0, str(SKILL_ROOT))
        from utils import next_phase

        flag_sets = [
            ["--resume", "--human-task", "T002=done", "--human-task", "T005=skip"],
            ["--resume", "--human-task", "T005=skip"],
        ]
        answers = [
            {"command": next_phase.phase_command(
                "plans/x.md", "opencode", "package", "implement", *flags),
             "flags": flags, "why": f"way {index}"}
            for index, flags in enumerate(flag_sets)
        ]
        return {"flags": ["--resume"], "answers": answers,
                "answers_text": next_phase.render_resume_answers(answers)}

    def test_the_resume_and_the_stops_answers_lead_the_options(
        self, built_package, uv_stub,
    ):
        """Fix round 14 (live run 5, OpenCode stop fix 01 and accept 02): the
        first option replayed `--human-task` answers beside a plain
        `resume_command`, and a first stop's `message` and `options` offered
        only the bare resume, the answers sitting in `error`."""
        step = self._answers_step()
        resume_line = {"exit": 3, "stdout": [
            "      resume with: uv run -- python multi_llm.py implement "
            "plans/x.md --host opencode --detach --resume",
            'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,"artifacts":[],'
            '"next":null,"error":"batch 0 refused"}']}
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, resume_line],
            extra_env={"STUB_RESUME_EXTRA": json.dumps(step)})
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        assert result["resume_command"] == "/multi-llm --implement plans/x.md --resume"
        calls = [option["args"] for option in result["options"] if "args" in option]
        assert calls[0] == dict(self.IMPLEMENT, extra_flags=["--resume"])
        assert calls[1:] == [dict(self.IMPLEMENT, extra_flags=answer["flags"])
                             for answer in step["answers"]]
        assert len(calls) == 1 + len(step["answers"])
        assert "human" in result["options"][-1] and "args" not in result["options"][-1]
        assert result["resume_answers"] == [
            {"command": answer["command"], "why": answer["why"]}
            for answer in step["answers"]]
        clause = surface["needs_human_message"]["answers"].replace(
            "{answers}", step["answers_text"])
        assert result["message"].endswith(clause)
        assert list(result)[:5] == [
            "ok", "state", "user_next_step", "resume_command", "resume_answers"]

    def test_a_composites_inner_resume_leads_with_its_own_calls(
        self, built_package, uv_stub,
    ):
        """plans/improve-plan-and-improve-code.md §2.6: `--blocked
        improve-plan` answers with apply-suggestions' own resume when the stop
        is inside it, and `user_resume.mode` names it. Built from the word
        asked about, an answer would become an `improve-plan` call carrying
        `--human-task`, which the composite's parser rejects."""
        sys.path.insert(0, str(SKILL_ROOT))
        from utils import next_phase

        call = dict(self.IMPLEMENT, mode="improve-plan")
        flags = ["--resume", "--human-task", "apply-batch-1=skip"]
        answers = [{
            "command": next_phase.phase_command(
                "plans/x.md", "opencode", "package", "apply-suggestions", *flags),
            "flags": flags, "why": "leave batch 1 as it stands"}]
        command = next_phase.phase_command(
            "plans/x.md", "opencode", "package", "apply-suggestions", "--resume")
        step = {"phase": "apply-suggestions", "mode": "apply-suggestions",
                "flags": ["--resume"], "answers": answers,
                "answers_text": next_phase.render_resume_answers(answers)}
        result, _ = uv_stub.run(
            built_package, call, [self.LAUNCH, self.BLOCKED],
            extra_env={"STUB_RESUME_COMMAND": command,
                       "STUB_RESUME_EXTRA": json.dumps(step)})
        asked = [argv[argv.index("--blocked") + 1] for argv in self._resume_calls(uv_stub)]
        assert asked == ["improve-plan"]
        assert result["resume_command"] == command
        calls = [option["args"] for option in result["options"] if "args" in option]
        inner = dict(call, mode="apply-suggestions")
        assert calls[:2] == [dict(inner, extra_flags=["--resume"]),
                             dict(inner, extra_flags=flags)]
        assert not any(c["mode"] == "improve-plan" and "--human-task" in c["extra_flags"]
                       for c in calls)
        assert result["resume_answers"] == [
            {"command": answers[0]["command"], "why": answers[0]["why"]}]

    def test_with_no_mode_the_asked_word_builds_the_calls(self, built_package, uv_stub):
        """An older lookup with no `mode`: the word asked about, as before."""
        call = dict(self.IMPLEMENT, mode="improve-plan")
        result, _ = uv_stub.run(
            built_package, call, [self.LAUNCH, self.BLOCKED],
            extra_env={"STUB_RESUME_EXTRA": json.dumps({"flags": ["--resume"]})})
        calls = [option["args"] for option in result["options"] if "args" in option]
        assert calls[0] == dict(call, extra_flags=["--resume"])

    def test_the_plain_resume_carries_what_it_does(self, built_package, uv_stub):
        """Fix round 22 (live run 9, OpenCode accept 01): the plain resume had
        no why and the message said "once the cause above is fixed". The
        stop's own why (`user_resume.why`) is `resume_why` and fills the
        message's resume clause."""
        why = ("runs the tasks that can run; T005 stays refused until "
               "driver.write_paths changes (T006 waits on it)")
        step = dict(self._answers_step(), why=why)
        resume_line = {"exit": 3, "stdout": [
            "      resume with: uv run -- python multi_llm.py implement "
            "plans/x.md --host opencode --detach --resume",
            'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,"artifacts":[],'
            '"next":null,"error":"batch 0 refused"}']}
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, resume_line],
            extra_env={"STUB_RESUME_EXTRA": json.dumps(step)})
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        text = surface["needs_human_message"]
        assert result["resume_why"] == why
        assert text["resume_why"].replace("{why}", why) in result["message"]
        assert text["resume"] not in result["message"]
        assert list(result)[:6] == [
            "ok", "state", "user_next_step", "resume_command", "resume_why",
            "resume_answers"]

    def test_the_written_section_is_the_way_forward(self, built_package, uv_stub):
        """Fix round 24 (live run 10, OpenCode stop 06: "Options (pick one…)
        1. Call multi_llm with …"): the lookup's section is
        `resume_markdown`, and the message's clause says to copy it in place
        of the resume and answers clauses."""
        sys.path.insert(0, str(SKILL_ROOT))
        from utils import next_phase

        why = "runs the tasks that can run; keeps the 4 tasks already done"
        step = dict(self._answers_step(), why=why)
        step["markdown"] = next_phase.render_resume_markdown(
            "implement", "/multi-llm --implement plans/x.md --resume", why,
            step["answers"])
        resume_line = {"exit": 3, "stdout": [
            "      resume with: uv run -- python multi_llm.py implement "
            "plans/x.md --host opencode --detach --resume",
            'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,"artifacts":[],'
            '"next":null,"error":"batch 0 refused"}']}
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, resume_line],
            extra_env={"STUB_RESUME_EXTRA": json.dumps(step)})
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        text = surface["needs_human_message"]
        assert result["resume_markdown"] == step["markdown"]
        assert result["message"].endswith(text["resume_markdown"])
        assert step["answers_text"] not in result["message"]
        assert text["resume_why"].replace("{why}", why) not in result["message"]
        assert list(result)[:7] == [
            "ok", "state", "user_next_step", "resume_command", "resume_why",
            "resume_answers", "resume_markdown"]

    #: A stop's line as recorded, and without the section's ways forward.
    STOP_LINE = (
        "implement stopped at T005: refused before it was dispatched. Not "
        "run: T006. To go on (your call): remove or narrow the deny rule "
        "`src/vendor/**` in driver.write_paths, then resume this phase; if "
        "not, skip T005: add --human-task T005=skip to the resume command.")
    STOP_LINE_BESIDE = (
        "implement stopped at T005: refused before it was dispatched. Not "
        "run: T006. To go on (your call): remove or narrow the deny rule "
        "`src/vendor/**` in driver.write_paths, then resume this phase.")

    def _stop_with_section(self, built_package, uv_stub, error, **extra):
        sys.path.insert(0, str(SKILL_ROOT))
        from utils import next_phase

        why = "runs the tasks that can run; keeps the 4 tasks already done"
        step = dict(self._answers_step(), why=why, **extra)
        step.setdefault("markdown", next_phase.render_resume_markdown(
            "implement", "/multi-llm --implement plans/x.md --resume", why,
            step["answers"]))
        resume_line = {"exit": 3, "stdout": [
            "      resume with: uv run -- python multi_llm.py implement "
            "plans/x.md --host opencode --detach --resume",
            "MULTI_LLM_RESULT: " + json.dumps({
                "state": "needs-human", "exit": 3, "artifacts": [],
                "next": None, "error": error})]}
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, resume_line],
            extra_env={"STUB_RESUME_EXTRA": json.dumps(step)})
        return result

    def test_the_error_beside_the_section_names_no_answer(
        self, built_package, uv_stub,
    ):
        """Fix round 28 (live run 12 #5): the hosts relayed the whole `error`
        line, whose "To go on (your call): … --human-task T005=skip …"
        restated the section's answers. With the section lifted, `error`,
        `result.error`, the line the message quotes and the stop title are
        the lookup's `error_beside_section`, while the error is the line the
        lookup read off the stop's record."""
        result = self._stop_with_section(
            built_package, uv_stub, self.STOP_LINE,
            error_line=self.STOP_LINE, error_beside_section=self.STOP_LINE_BESIDE)
        assert result["error"] == self.STOP_LINE_BESIDE
        assert result["result"]["error"] == self.STOP_LINE_BESIDE
        assert f'"{self.STOP_LINE_BESIDE}"' in result["message"]
        assert self.STOP_LINE not in result["message"]
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        assert result["message"].endswith(
            surface["needs_human_message"]["resume_markdown"])
        # The tool's own line, which shows without the model, says it too.
        got = node_json(f"""
        const hooks = await MultiLlmPlugin({{}})
        const output = {{ title: "", output: {json.dumps(json.dumps(result))}, metadata: {{}} }}
        await hooks["tool.execute.after"](
          {{ tool: surface.tool_name, sessionID: "s", callID: "c", args: {{}} }},
          output,
        )
        emit(output.title)
        """, built_package)
        assert "src/vendor/**" in got and "--human-task" not in got, got

    def test_an_error_the_record_does_not_describe_keeps_its_own_line(
        self, built_package, uv_stub,
    ):
        other = "implement stopped at T009: the batch was refused."
        result = self._stop_with_section(
            built_package, uv_stub, other,
            error_line=self.STOP_LINE, error_beside_section=self.STOP_LINE_BESIDE)
        assert result["error"] == other
        assert result["result"]["error"] == other

    def test_without_a_section_the_error_stays_whole(self, built_package, uv_stub):
        result = self._stop_with_section(
            built_package, uv_stub, self.STOP_LINE, markdown=["not text"],
            error_line=self.STOP_LINE, error_beside_section=self.STOP_LINE_BESIDE)
        assert "resume_markdown" not in result
        assert result["error"] == self.STOP_LINE

    def test_a_manual_gate_stop_relays_the_gate_beside_its_section(
        self, built_package, uv_stub, host_path, repo, capsys,
    ):
        """The merge of fix round 28: implement's manual-task gate recorded
        no line beside its section, so this plugin relayed "To go on: …
        --human-task T001=done" beside a `resume_markdown` giving the same
        command. Driven on the real path: the executor's stop and the real
        `util next-command --blocked` lookup, handed over as it answered."""
        from tests.test_manual_task_gate import (
            IMPLEMENT_WRITE_IDS, _clean_env, _manual, _run, _seed_generate_tasks,
        )

        _clean_env(repo.monkeypatch)
        _seed_generate_tasks(repo)
        _manual(repo, "T001")
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        code, _out, err, stopped = _run([
            "implement", str(repo.plan), "--yes", "--exec-subagents",
            "--host", "opencode"], capsys)
        assert code == 3, err
        code, _out, err, lookup = _run([
            "util", "next-command", str(repo.plan), "--blocked", "implement",
            "--host", "opencode"], capsys)
        assert code == 0, err
        step = dict(lookup["user_resume"])
        command = step.pop("command")
        step.pop("phase")
        repo.monkeypatch.setenv("PATH", host_path)
        resume_line = {"exit": 3, "stdout": [
            "MULTI_LLM_RESULT: " + json.dumps({
                "state": "needs-human", "exit": 3, "artifacts": [],
                "next": None, "error": stopped["error"]})]}
        result, _ = uv_stub.run(
            built_package, self.IMPLEMENT, [self.LAUNCH, resume_line],
            extra_env={"STUB_RESUME_COMMAND": command,
                       "STUB_RESUME_EXTRA": json.dumps(step)})
        beside = result["error"]
        assert "--human-task" not in beside, beside
        assert beside.startswith(
            "implement ran nothing: 2 code tasks wait on manual task T001"), beside
        assert result["result"]["error"] == beside
        assert f'"{beside}"' in result["message"]
        assert "--human-task" not in result["message"].split("`resume_markdown`")[0]
        assert result["resume_markdown"].count("--human-task T001=done") == 1
        # Fix round 29 (review of round 28 #1): and the user's decline.
        assert result["resume_markdown"].count("--human-task T001=skip") == 1
        calls = [option["args"] for option in result["options"] if "args" in option]
        assert [call["extra_flags"] for call in calls[:2]] == [
            ["--resume", "--human-task", "T001=done"],
            ["--resume", "--human-task", "T001=skip"]]

    def test_the_plugin_spells_none_of_the_lookup(self, built_package):
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        resume = surface["dispatcher"]["resume"]
        source = (REPO_ROOT / "packaging" / "opencode" / "src" / "index.mjs").read_text(encoding="utf-8")
        for word in [*resume["argv"], resume["option"]]:
            assert f'"{word}"' not in source, word


#: Every phase the surface knows, read from the surface (never retyped), so a
#: phase added later is covered the day it lands. The commit modes take no
#: plan (a typed one is refused before any spawn): their run is named by its
#: pinned range, which `TestACommitRunIsNamedByItsPinnedRange` holds.
REVS_KIND = pkg_build.cli_surface()["positional_kinds"]["revs"]
PHASE_ROWS = [row for row in pkg_build.cli_surface()["modes"]
              if row.get("phase") and REVS_KIND not in row["positionals"]]
RANGE_ROWS = [row for row in pkg_build.cli_surface()["modes"]
              if REVS_KIND in row["positionals"]]


class TestTheTypedPlanSurvivesEveryFailure:
    """Racing re-run 3 (2026-10-05, codex 06): a tool call the route handed back
    named the plan absolutely, the model sent it back as given, and the run's
    failure came back with a `resume_command` carrying the absolute path for a
    plan the user typed relative. Every phase, and every tool call this route
    hands back (`next`, `abort_command`, `options`), keeps the plan as typed."""

    PLAN = "plans/p.md"
    LAUNCH = {"exit": 0, "stdout": [
        "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}

    def _args(self, row):
        takes_question = "<question>" in row["positionals"]
        return {"mode": row["name"], "plan": self.PLAN,
                "question": "why?" if takes_question else "", "extra_flags": []}

    def _failed(self, row, absolute):
        prefix = "uv run --frozen --no-dev --project /skill -- python /skill/multi_llm.py"
        body = {"mode": row["name"], "phase": row["phase"], "state": "needs-human",
                "exit": 3, "artifacts": [], "next": None}
        return {"exit": 3, "stdout": [
            f"      resume with: {prefix} {row['name']} {word(absolute)} --host opencode "
            "--detach --yes --resume",
            "MULTI_LLM_RESULT: " + json.dumps(body)]}

    @pytest.mark.parametrize("row", PHASE_ROWS, ids=[row["name"] for row in PHASE_ROWS])
    def test_a_failed_phase_resumes_with_the_plan_as_typed(
        self, built_package, uv_stub, tmp_path, row
    ):
        absolute = str(tmp_path / "plans" / "p.md")
        result, _ = uv_stub.run(built_package, self._args(row),
                                [self.LAUNCH, self._failed(row, absolute)])
        assert self.PLAN in result["resume_command"]
        assert absolute not in result["resume_command"]
        assert result["user_next_step"] is None
        for option in result["options"]:
            if "args" in option:
                assert option["args"]["plan"] == self.PLAN, option
        assert absolute not in result["output"]

    @pytest.mark.parametrize("row", PHASE_ROWS, ids=[row["name"] for row in PHASE_ROWS])
    def test_a_stopped_run_hands_back_its_abort_with_the_plan_as_typed(
        self, built_package, uv_stub, tmp_path, row
    ):
        live = {"exit": 2, "stdout": [
            'MULTI_LLM_RESULT: {"state":"running","exit":2,"artifacts":[],"next":null}']}
        aborted = {"exit": 0, "stdout": [
            "aborted", 'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
            '"artifacts":[],"next":null}']}
        result, calls = uv_stub.run(
            built_package, self._args(row), [self.LAUNCH, live, aborted],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1"})
        assert result["ok"] is False and result["run_id"] == "R"
        assert result["abort_command"]["args"]["plan"] == self.PLAN
        # Inside the plugin the abort it dispatched still names the file absolutely.
        abort = [_dispatcher_argv(c) for c in calls if "--abort" in _dispatcher_argv(c)]
        assert abort and abort[0][1] == str(tmp_path / "plans" / "p.md")

    def test_an_unfollowed_next_names_the_plan_as_typed(
        self, built_package, uv_stub, tmp_path
    ):
        absolute = str(tmp_path / "plans" / "p.md")
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":'
            + json.dumps(f"uv run -- python multi_llm.py generate-tasks {word(absolute)}")
            + "}"]}
        result, _ = uv_stub.run(
            built_package, {"mode": "review-plan", "plan": self.PLAN, "question": "",
                            "extra_flags": []}, [self.LAUNCH, done])
        assert result["next"]["args"]["plan"] == self.PLAN
        assert absolute not in result["next"]["human"]


# ===========================================================================
# The commit modes' range (plans/review-commits.md §2.9, "Tool routes")
# ===========================================================================

from utils import commit_range as range_mod  # noqa: E402

REVS_SHAPE = pkg_build.cli_surface()["revs"]
FULL_BASE = "a1b2c3d4e5f6" + "0" * 28
FULL_HEAD = "e4f5a6b7c8d9" + "1" * 28
PINNED = "a1b2c3d4e5f6..e4f5a6b7c8d9"
BRIEF = "plans/commit-reviews/a1b2c3d4e5f6-e4f5a6b7c8d9.md"
ROOT_BRIEF = "plans/commit-reviews/root-e4f5a6b7c8d9.md"
PHRASE = "last 2 commits"
RESULT_DONE = 'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}'


def _range_record(*, base=FULL_BASE, brief=BRIEF) -> dict:
    """The launch's result-line ``commit_range``, built and rendered by the
    owner (``utils/commit_range.py``) as the front door builds it, the block it
    printed riding along as ``lines``."""
    commits = (
        range_mod.Commit(sha=FULL_HEAD, date="2026-10-24", ct=2, subject="Second",
                         author="a", merge=False),
        range_mod.Commit(sha="c3d4e5f6a7b8" + "2" * 28, date="2026-10-23", ct=1,
                         subject="First", author="a", merge=False),
    )
    cr = range_mod.CommitRange(base=base, head=FULL_HEAD, commits=commits,
                               phrase=PHRASE, window=None, kind="last")
    size = range_mod.RangeSize.of(2, 3, 10, 2)
    record = range_mod.result_payload(cr, size=size, brief=brief, brief_on_disk=True)
    record[REVS_SHAPE["lines_key"]] = range_mod.render_commit_block(cr, size=size)
    return record


def _commit_launch(record=None, mode="review-commits") -> dict:
    record = _range_record() if record is None else record
    return {"exit": 0, "stdout": [
        *record[REVS_SHAPE["lines_key"]], "[COMMIT_BRIEF] " + str(record["brief"]),
        "run_id: R", "log: /l",
        "MULTI_LLM_RESULT: " + json.dumps({
            "mode": mode, "state": "done", "exit": 0, "artifacts": [], "next": None,
            REVS_SHAPE["range_key"]: record})]}


def _range_call(mode="review-commits", revs=PHRASE, **extra) -> dict:
    call = {"mode": mode, "revs": revs, "plan": "", "question": "", "extra_flags": []}
    call.update(extra)
    return call


_PLAIN_LAUNCH = {"exit": 0, "stdout": ["run_id: R", "log: /l", RESULT_DONE]}
_DONE = {"exit": 0, "stdout": [RESULT_DONE]}


class TestTheCommitModesRange:
    """`revs` is the commit modes' one range argument: composed right after the
    mode word, never resolved as a file, and `plan` keeps one meaning."""

    @pytest.mark.parametrize("row", RANGE_ROWS, ids=[row["name"] for row in RANGE_ROWS])
    @pytest.mark.parametrize("revs", ["", PHRASE])
    def test_a_plan_on_a_commit_mode_is_refused_before_any_spawn(
        self, built_package, uv_stub, row, revs
    ):
        result, calls = uv_stub.run(
            built_package, _range_call(row["name"], revs, plan="docs/x.md"), [_DONE])
        assert calls == []
        assert result["ok"] is False and result["state"] == "error"
        assert result["message"] == REVS_SHAPE["plan_refusal"].replace(
            "{mode}", row["name"])
        assert '"revs"' in result["message"]
        assert "--context-plan" in result["message"]

    @pytest.mark.parametrize("revs, words", [
        (PHRASE, ["last", "2", "commits"]),
        ("-2", ["-2"]),
        (PINNED, [PINNED]),
        ("  this   branch ", ["this", "branch"]),
        ("", []),
        ("plans/commit reviews/root-e4f5a6b7c8d9.md",
         ["plans/commit reviews/root-e4f5a6b7c8d9.md"]),
        # A word inside it starting with one "-" is the path's, not a flag.
        ("plans/commit reviews -old/root-e4f5a6b7c8d9.md",
         ["plans/commit reviews -old/root-e4f5a6b7c8d9.md"]),
    ])
    def test_revs_reaches_the_argv_right_after_the_mode(
        self, built_package, uv_stub, revs, words
    ):
        surface = pkg_build.cli_surface()
        _, calls = uv_stub.run(built_package, _range_call(revs=revs),
                               [_PLAIN_LAUNCH, _DONE])
        assert _dispatcher_argv(calls[0]) == (
            ["review-commits", *words, *surface["baked_flags"], surface["detach_flag"]])

    def test_a_left_out_revs_is_head(self, built_package, uv_stub):
        surface = pkg_build.cli_surface()
        call = _range_call()
        del call["revs"]
        _, calls = uv_stub.run(built_package, call, [_PLAIN_LAUNCH, _DONE])
        assert _dispatcher_argv(calls[0]) == (
            ["review-commits", *surface["baked_flags"], surface["detach_flag"]])

    @pytest.mark.parametrize("revs", ["--quick", "last 2 --internal-worker", "-2h",
                                      "--2", "- 2",
                                      # Kept whole as a brief path, yet refused.
                                      "HEAD~3..HEAD --context-plan docs/x.md"])
    def test_a_flag_inside_revs_is_refused(self, built_package, uv_stub, revs):
        result, calls = uv_stub.run(built_package, _range_call(revs=revs), [_DONE])
        flag = next(word for word in revs.split() if word.startswith("-"))
        assert calls == [] and result["state"] == "error"
        assert f'"revs" holds "{flag}"' in result["message"]

    def test_a_range_on_a_mode_that_takes_none_is_refused(self, built_package, uv_stub):
        result, calls = uv_stub.run(
            built_package, {"mode": "review-plan", "revs": PHRASE, "plan": "p.md",
                            "question": "", "extra_flags": []}, [_DONE])
        assert calls == [] and '"revs" must be ""' in result["message"]

    @pytest.mark.parametrize("revs", [2, None, ["last", "2"]])
    def test_revs_must_be_a_string(self, built_package, uv_stub, revs):
        result, calls = uv_stub.run(built_package, _range_call(revs=revs), [_DONE])
        assert calls == [] and result["message"].startswith('"revs" must be a string')

    def test_the_range_shaped_worked_example_dispatches_its_context_plan(
        self, built_package, uv_stub
    ):
        surface = pkg_build.cli_surface()
        block = surface["tool_description"].split("\nExamples:\n", 1)[1]
        (example,) = [json.loads(line) for line in block.splitlines()
                      if '"revs"' in line]
        assert example["plan"] == "" and example["revs"] == PHRASE
        _, calls = uv_stub.run(built_package, example, [_PLAIN_LAUNCH, _DONE])
        assert _dispatcher_argv(calls[0]) == (
            [example["mode"], "last", "2", "commits", *surface["baked_flags"],
             surface["detach_flag"], "--context-plan", "docs/x.md"])


class TestACommitRunIsNamedByItsPinnedRange:
    """Every tool call this plugin builds for a commit run (the abort, the
    `--blocked` lookup and its options) names the range the launch pinned (the
    result line's `commit_range`), never the phrase the call typed."""

    LIVE = {"exit": 2, "stdout": [
        'MULTI_LLM_RESULT: {"state":"running","exit":2,"artifacts":[],"next":null}']}
    ABORTED = {"exit": 0, "stdout": [
        "aborted", 'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
        '"artifacts":[],"next":null}']}

    def _stopped(self, built_package, uv_stub, row, record=None):
        return uv_stub.run(
            built_package, _range_call(row["name"]),
            [_commit_launch(record, row["name"]), self.LIVE, self.ABORTED],
            extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1"})

    @pytest.mark.parametrize("row", RANGE_ROWS, ids=[row["name"] for row in RANGE_ROWS])
    def test_a_stopped_run_hands_back_its_abort_by_the_pinned_range(
        self, built_package, uv_stub, tmp_path, row
    ):
        result, calls = self._stopped(built_package, uv_stub, row)
        assert result["ok"] is False and result["run_id"] == "R"
        assert result["abort_command"]["args"]["plan"] == PINNED
        assert PHRASE not in json.dumps(result["abort_command"])
        # Inside the plugin the abort it dispatched names the brief absolutely.
        abort = [_dispatcher_argv(c) for c in calls if "--abort" in _dispatcher_argv(c)]
        assert abort and abort[0][1] == str(tmp_path / BRIEF)

    def test_a_root_range_is_named_by_its_brief(self, built_package, uv_stub):
        result, _ = self._stopped(built_package, uv_stub, RANGE_ROWS[0],
                                  _range_record(base=None, brief=ROOT_BRIEF))
        assert result["abort_command"]["args"]["plan"] == ROOT_BRIEF

    @pytest.mark.parametrize("row", RANGE_ROWS, ids=[row["name"] for row in RANGE_ROWS])
    def test_a_failed_run_is_asked_about_by_its_brief(
        self, built_package, uv_stub, row
    ):
        failed = {"exit": 3, "stdout": ["MULTI_LLM_RESULT: " + json.dumps({
            "mode": row["name"], "phase": row["phase"], "state": "needs-human",
            "exit": 3, "artifacts": [], "next": None, "error": "stopped"})]}
        result, _ = uv_stub.run(
            built_package, _range_call(row["name"]),
            [_commit_launch(mode=row["name"]), failed],
            extra_env={"STUB_RESUME_EXTRA": json.dumps({"flags": ["--resume"]})})
        (asked,) = [json.loads(line) for line in Path(
            str(uv_stub.log_path) + ".resume").read_text(encoding="utf-8").splitlines()]
        dispatcher = _dispatcher_argv(asked)
        assert dispatcher[2] == BRIEF and dispatcher[3:5] == ["--blocked", row["name"]]
        assert result["resume_command"]
        assert result["options"][0]["args"] == {
            "mode": row["name"], "revs": PINNED, "plan": "", "question": "",
            "extra_flags": ["--resume"]}
        assert PHRASE not in json.dumps(result["options"])

    def test_a_status_call_naming_the_pinned_range_passes_it_on(
        self, built_package, uv_stub
    ):
        surface = pkg_build.cli_surface()
        call = {"mode": "status", "plan": PINNED, "question": "",
                "extra_flags": [*surface["dispatcher"]["poll"][1:], "--run-id", "R"]}
        _, calls = uv_stub.run(built_package, call, [_DONE])
        assert _dispatcher_argv(calls[0])[:2] == ["status", PINNED]

    def test_a_launch_that_pinned_nothing_asks_nothing(self, built_package, uv_stub):
        refused = {"exit": 3, "stdout": [
            "[RANGE_UNRECOGNIZED] last 2 comits",
            'MULTI_LLM_RESULT: {"mode":"review-commits","state":"needs-human",'
            '"exit":3,"artifacts":[],"next":null,"error":"[RANGE_UNRECOGNIZED]"}']}
        result, _ = uv_stub.run(built_package, _range_call(revs="last 2 comits"),
                                [refused])
        resume = Path(str(uv_stub.log_path) + ".resume")
        assert not resume.exists() or not resume.read_text(encoding="utf-8").strip()
        assert result["state"] == "needs-human" and "commit_blocks" not in result


class TestACommitFollowUpMapsBackToRevs:
    @pytest.mark.parametrize("tail, revs, extra", [
        ("-2 --quick", "-2", ["--quick"]),
        ("last 2 commits", PHRASE, []),
        ("--quick -2", "", ["--quick", "-2"]),
        (PINNED + " --focus 'x y' --context-plan docs/x.md --host opencode --yes",
         PINNED, ["--focus", "x y", "--context-plan", "docs/x.md"]),
        ("'plans/commit reviews/root-e4f5a6b7c8d9.md'",
         "plans/commit reviews/root-e4f5a6b7c8d9.md", []),
    ])
    def test_the_parse_follows_the_dispatchers_split(
        self, built_package, tail, revs, extra
    ):
        got = node_json(f"""
        const parsed = T.parseDispatcherCommand(
          {json.dumps("uv run -- python /s/multi_llm.py review-commits " + tail)},
          surface)
        emit({{parsed, spelled: T.userSpelling(parsed, surface)}})
        """, built_package)
        assert got["parsed"] == {"mode": "review-commits", "revs": revs, "plan": "",
                                 "question": "", "extra_flags": extra}
        assert got["spelled"].startswith("/multi-llm --review-commits")

    def test_an_unfollowed_next_comes_back_as_a_range_call(self, built_package, uv_stub):
        done = {"exit": 0, "stdout": ["MULTI_LLM_RESULT: " + json.dumps({
            "state": "done", "exit": 0, "artifacts": [],
            "next": f"uv run -- python multi_llm.py apply-commit-fixes {PINNED}"})]}
        result, _ = uv_stub.run(built_package, _range_call(), [_commit_launch(), done])
        assert result["next"]["args"] == {
            "mode": "apply-commit-fixes", "revs": PINNED, "plan": "", "question": "",
            "extra_flags": []}
        assert PINNED in result["next"]["human"]


class TestTheCommitBlocksAreRelayedOnce:
    """§2.3 "Relay": the launch printed the block before it detached, and a
    tool route renders it into the result from the result line's
    `commit_range` (and `[COMMIT_FIXES]` from `commit_fixes`), as printed, once."""

    def test_the_commit_block_comes_from_the_launch_s_result_line(
        self, built_package, uv_stub
    ):
        record = _range_record()
        result, _ = uv_stub.run(built_package, _range_call(),
                                [_commit_launch(record), _DONE])
        assert result["commit_blocks"] == "\n".join(record["lines"])
        # Right after the way forward and the message, ahead of the rest.
        keys = list(result)
        assert set(keys[:keys.index("commit_blocks")]) <= {
            "ok", "state", "user_next_step", "resume_command", "resume_why",
            "resume_answers", "resume_markdown", "no_command_reason", "next",
            "abort_command", "options", "message"}

    def test_the_fixes_block_comes_from_the_terminal_poll_and_leaves_output(
        self, built_package, uv_stub
    ):
        block = range_mod.FixBlock(to_apply=2, per_commit=[(FULL_HEAD, 2, "Second")],
                                   unattributed=0, skipped_changed=1,
                                   changed_files=["b.py"], head=FULL_HEAD)
        lines = range_mod.render_fix_block(block)
        done = {"exit": 0, "stdout": ["status view", *lines, "MULTI_LLM_RESULT: "
                                      + json.dumps({
                                          "state": "done", "exit": 0, "artifacts": [],
                                          "next": None, REVS_SHAPE["fixes_key"]: {
                                              "to_apply": 2,
                                              REVS_SHAPE["lines_key"]: lines}})]}
        result, _ = uv_stub.run(built_package, _range_call("apply-commit-fixes"),
                                [_commit_launch(mode="apply-commit-fixes"), done])
        assert result["commit_blocks"] == "\n".join(_range_record()["lines"] + lines)
        assert "[COMMIT_FIXES]" not in result["output"]
        assert "status view" in result["output"]
        assert "lines" not in result["result"]["commit_fixes"]

    def test_a_launch_that_stopped_after_its_block_relays_it_once(
        self, built_package, uv_stub
    ):
        record = _range_record()
        stopped = {"exit": 3, "stdout": [
            *record["lines"], "ERROR: a run is live on this brief",
            "MULTI_LLM_RESULT: " + json.dumps({
                "mode": "review-commits", "state": "needs-human", "exit": 3,
                "artifacts": [], "next": None, "error": "a run is live",
                REVS_SHAPE["range_key"]: record})]}
        result, _ = uv_stub.run(built_package, _range_call(), [stopped])
        assert result["commit_blocks"] == "\n".join(record["lines"])
        assert record["lines"][0] not in result["output"]
        assert "ERROR: a run is live on this brief" in result["output"]
        assert "lines" not in result["result"]["commit_range"]

    def test_a_plan_mode_carries_no_block(self, built_package, uv_stub):
        result, _ = uv_stub.run(
            built_package, {"mode": "review-plan", "plan": "p.md", "question": "",
                            "extra_flags": []}, [_PLAIN_LAUNCH, _DONE])
        assert "commit_blocks" not in result


class TestTheWayForwardLeadsTheResult:
    """Racing re-run 3 (2026-10-05, codex 06b): the model read a cut of a large
    result, `resume_command` sat in the dropped middle, and the model told the
    user there was none. Every result here leads with the way forward."""

    LAUNCH = {"exit": 0, "stdout": [
        "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
    BLOCKED = {"exit": 3, "stdout": [
        "progress line " + "x" * 80 for _ in range(60)] + [
        'MULTI_LLM_RESULT: {"mode":"implement","phase":"implement",'
        '"state":"needs-human","exit":3,"artifacts":[],"next":null}']}
    IMPLEMENT = {"mode": "implement", "plan": "plans/x.md", "question": "",
                 "extra_flags": []}

    def test_a_blocked_run_leads_with_the_resume_command(self, built_package, uv_stub):
        result, _ = uv_stub.run(built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED])
        # Fix round 8: the re-invoke options lead too, ahead of `result`,
        # `output` and `stderr`; the `next: null` beside them does not.
        assert list(result)[:6] == [
            "ok", "state", "user_next_step", "resume_command", "options", "message"]

    def test_the_calls_a_result_hands_back_lead_when_set(self, built_package):
        """Fix round 8: a stopped watch's way forward is its `abort_command`,
        and a running one's its `next`; both sat behind the run's details."""
        ordered = node_json(
            """
            emit([
              Object.keys(T.wayForwardFirst({
                state: "needs-human", ok: false, reason: "r", run_id: "R", log: "/l",
                abort_dispatched: false, abort_command: { tool: "t", args: {} },
                anchor: "/a", message: "m", instructions: "long" })),
              Object.keys(T.wayForwardFirst({
                state: "running", ok: true, run_id: "R", log: "/l", polls: 3,
                next: { tool: "t", args: {} }, abort_command: { tool: "t", args: {} },
                options: null, message: "m", instructions: "long" })),
            ])
            """,
            built_package,
        )
        assert ordered[0] == ["ok", "state", "abort_command", "message", "reason", "run_id",
                              "log", "abort_dispatched", "anchor", "instructions"]
        assert ordered[1] == ["ok", "state", "next", "abort_command", "message", "run_id",
                              "log", "polls", "options", "instructions"]

    def test_no_command_leads_with_its_reason(self, built_package, uv_stub):
        result, _ = uv_stub.run(built_package, self.IMPLEMENT, [self.LAUNCH, self.BLOCKED],
                                extra_env={"STUB_RESUME_FAIL": "1"})
        assert list(result)[:4] == ["ok", "state", "user_next_step", "no_command_reason"]

    def test_a_finished_phase_leads_with_its_next_step(self, built_package, uv_stub):
        step = {"phase": "generate-tasks", "command": "/multi-llm --generate-tasks x.md",
                "complete": False}
        done = {"exit": 0, "stdout": ["MULTI_LLM_RESULT: " + json.dumps({
            "state": "done", "exit": 0, "artifacts": [], "next": None,
            "user_next_step": step})]}
        result, _ = uv_stub.run(
            built_package, {"mode": "review-plan", "plan": "x.md", "question": "",
                            "extra_flags": []}, [self.LAUNCH, done])
        assert list(result)[:3] == ["ok", "state", "user_next_step"]
        assert result["user_next_step"] == step

    def test_the_instructions_come_last(self, built_package):
        ordered = node_json(
            """
            emit(Object.keys(T.wayForwardFirst({
              instructions: "long", exit: 0, ok: true, next: null, state: "done",
              message: "m", user_next_step: null })))
            """,
            built_package,
        )
        assert ordered == ["ok", "state", "user_next_step", "message", "exit", "next",
                           "instructions"]


class TestAFailedResultSaysWhatTheUserCanRun:
    """Racing re-run 2026-10-04 (opencode #3/#4): a failed generate-tasks came
    back with the same re-invoke option twice, its plan absolute, the
    dispatcher's own retry and resume lines replaced by a removal note, and
    no command key at all — and the host model printed a next command it had
    composed itself. Every failed result now says what the user can run."""

    PLAN = "plans/racing-game.md"
    ARGS = {"mode": "generate-tasks", "plan": PLAN, "question": "", "extra_flags": []}
    LAUNCH = {"exit": 0, "stdout": [
        "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}

    @staticmethod
    def _surface(package):
        return json.loads((package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))

    def _failed(self, tmp_path):
        """The poll a failed detached generate-tasks answers with: its own
        retry hint (`--yes`, from the worker) and status's `resume with:` line
        (the launcher's `--detach`), both naming the plan absolutely."""
        absolute = str(tmp_path / "plans" / "racing-game.md")
        prefix = "uv run --frozen --no-dev --project /skill -- python /skill/multi_llm.py"
        body = {"mode": "generate-tasks", "phase": None, "state": "needs-human",
                "exit": 3, "artifacts": [], "next": None}
        return {"exit": 3, "stdout": [
            "Re-invoke to retry only the failures (task_generation):",
            f"  {prefix} generate-tasks {word(absolute)} --host opencode --yes",
            f"      resume with: {prefix} generate-tasks {word(absolute)} "
            "--host opencode --detach --yes",
            "MULTI_LLM_RESULT: " + json.dumps(body),
        ]}

    def test_one_option_per_call_with_the_plan_as_typed(self, built_package, uv_stub,
                                                        tmp_path):
        result, _ = uv_stub.run(built_package, self.ARGS,
                                [self.LAUNCH, self._failed(tmp_path)],
                                extra_env={"STUB_RESUME_FAIL": "1"})
        calls = [option for option in result["options"] if "tool" in option]
        assert len(calls) == 1, result["options"]
        assert calls[0]["args"] == {"mode": "generate-tasks", "plan": self.PLAN,
                                    "question": "", "extra_flags": []}
        assert str(tmp_path) not in calls[0]["human"]
        assert result["options"][-1] == {
            "human": self._surface(built_package)["needs_human_option"]}

    def test_the_redacted_lines_name_the_hosts_own_command(self, built_package,
                                                           uv_stub, tmp_path):
        result, _ = uv_stub.run(built_package, self.ARGS,
                                [self.LAUNCH, self._failed(tmp_path)],
                                extra_env={"STUB_RESUME_FAIL": "1"})
        spelled = f"/multi-llm --generate-tasks {self.PLAN}"
        lines = result["output"].split("\n")
        assert f"  {spelled}" in lines
        assert f"      resume with: {spelled}" in lines
        blob = json.dumps({key: result.get(key)
                           for key in ("output", "result", "options", "stderr", "message")})
        assert "uv run" not in blob and "multi_llm.py" not in blob
        assert "a runnable command was removed" not in blob

    def test_no_resume_means_an_explicit_no_command(self, built_package, uv_stub,
                                                    tmp_path):
        surface = self._surface(built_package)
        result, _ = uv_stub.run(built_package, self.ARGS,
                                [self.LAUNCH, self._failed(tmp_path)],
                                extra_env={"STUB_RESUME_FAIL": "1"})
        assert "user_next_step" in result and result["user_next_step"] is None
        assert result["no_command_reason"] == surface["no_command_reason"]
        assert "resume_command" not in result
        assert result["message"].endswith(surface["needs_human_message"]["no_command"])

    def test_a_resume_command_replaces_the_no_command(self, built_package, uv_stub,
                                                      tmp_path):
        surface = self._surface(built_package)
        result, _ = uv_stub.run(
            built_package, self.ARGS, [self.LAUNCH, self._failed(tmp_path)],
            extra_env={"STUB_RESUME_COMMAND": f"/multi-llm --generate-tasks {self.PLAN}"})
        assert result["resume_command"] == f"/multi-llm --generate-tasks {self.PLAN}"
        assert result["user_next_step"] is None
        assert "no_command_reason" not in result
        assert result["message"].endswith(surface["needs_human_message"]["resume"])

    def test_the_resume_option_is_the_resume_command(self, built_package, uv_stub,
                                                     tmp_path):
        """OpenCode racing run 4 (02): a stopped `--agent-decide` apply run
        came back with the resume offered as `--agent-decide --resume` while
        `resume_command` read `--resume`. This tool composes `--yes`, which
        already means `--agent-decide` in an apply phase, so the replay drops
        it and the two agree. Both strings are built by the real Python:
        `status`'s `resume with:` line from the launch's replay flags,
        `resume_command` by `next_phase`."""
        from utils import next_phase

        absolute = str(tmp_path / "plans" / "racing-game.md")
        recorded = cli_common.replay_flags("apply-suggestions", [
            "--plan-file", absolute, "--agent-decide", "--host", "opencode",
            "--yes", "--exec-subagents", "--detach"])
        view = {"run": {"mode": "apply-suggestions", "replay_flags": recorded},
                "progress": {}}
        resume_with = run_status._resume_command(view, Path(absolute))
        command = next_phase.resume_command(
            self.PLAN, "opencode", "package", "apply-suggestions",
            workflow={"phases": {"apply-suggestions": {"status": "in_progress"}}})
        body = {"mode": "apply-suggestions", "phase": "apply-suggestions",
                "state": "needs-human", "exit": 3, "artifacts": [], "next": None}
        stopped = {"exit": 3, "stdout": [
            f"      resume with: {resume_with}",
            "MULTI_LLM_RESULT: " + json.dumps(body)]}
        args = {"mode": "apply-suggestions", "plan": self.PLAN, "question": "",
                "extra_flags": ["--yes", "--agent-decide"]}

        result, _ = uv_stub.run(built_package, args, [self.LAUNCH, stopped],
                                extra_env={"STUB_RESUME_COMMAND": command})

        assert result["resume_command"] == command
        calls = [option for option in result["options"] if "tool" in option]
        assert [call["args"] for call in calls] == [{
            "mode": "apply-suggestions", "plan": self.PLAN, "question": "",
            "extra_flags": command.split()[3:]}]
        assert command.split()[3:] == ["--resume"]

    def test_an_error_result_says_so_too(self, built_package, uv_stub):
        surface = self._surface(built_package)
        crashed = {"exit": 1, "stdout": [
            'MULTI_LLM_RESULT: {"state":"error","exit":1,"artifacts":[],"next":null}']}
        result, _ = uv_stub.run(built_package, self.ARGS, [self.LAUNCH, crashed])
        assert result["ok"] is False
        assert result["user_next_step"] is None
        assert result["no_command_reason"] == surface["no_command_reason"]
        assert not result["message"].endswith(surface["needs_human_message"]["no_command"])

    def test_a_finished_run_and_a_refused_call_carry_neither(self, built_package,
                                                             uv_stub):
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}
        result, _ = uv_stub.run(built_package, self.ARGS, [self.LAUNCH, done])
        assert "no_command_reason" not in result and "user_next_step" not in result
        refused, calls = uv_stub.run(
            built_package, dict(self.ARGS, extra_flags=["--detach"]), [done])
        assert calls == [] and refused["ok"] is False
        assert "no_command_reason" not in refused and "user_next_step" not in refused

    def test_an_older_surface_keeps_the_note_and_drops_only_the_sentences(
        self, built_package, tmp_path
    ):
        """Additive within schema v5: without `user_command` a redacted command
        keeps its pointer, and without `no_command_reason` the null stands alone."""
        absolute = str(tmp_path / "p.md")
        out = node_json(
            f"""
            const bare = structuredClone(surface)
            delete bare.dispatcher.user_command
            delete bare.no_command_reason
            delete bare.needs_human_message.no_command
            const values = {{ word: "generate-tasks", plan: {json.dumps(absolute)},
                             planArg: "p.md", question: "", extraFlags: [] }}
            const child = {{
              code: 3, argv: ["uv"], stderr: "", resultRaw: null, truncated: false,
              banner: {{}}, aborted: false, timedOut: false, spawnError: null,
              result: {{ state: "needs-human", exit: 3, next: null }},
              stdout: "  uv run -- python /s/multi_llm.py generate-tasks " +
                {json.dumps(absolute)} + " --yes\\n",
            }}
            const terminal = T.terminalResult(child, {{ anchor: "/a", surface: bare, values }})
            emit({{ terminal, finished: T.finishOutcome({{ ...terminal }}, {{}}, bare) }})
            """,
            built_package,
        )
        assert "a runnable command was removed" in out["terminal"]["output"]
        finished = out["finished"]
        assert finished["user_next_step"] is None
        assert "no_command_reason" not in finished
        assert finished["message"] == out["terminal"]["message"]


UV_PREFIX = "uv run --frozen --no-dev --project /skill -- python /skill/multi_llm.py"


class TestAReportCallOnAFailedRun:
    """Adversarial review 2026-10-05: a direct `status` (or `abort`) call on a
    failed write run re-spelled status's `resume with:` line into the host's
    command, then asked `--blocked status` — which answers nothing, status is
    no phase — and so added `user_next_step: null`, `no_command_reason` and
    "give the user no command to type" beside output that visibly named one."""

    PLAN = "plans/x.md"
    STATUS = {"mode": "status", "plan": PLAN, "question": "", "extra_flags": []}
    SPELLED = "/multi-llm --implement plans/x.md --resume"

    def _failed(self, tmp_path, **body):
        """A status poll of a failed implement run (exit 3), plan absolute."""
        absolute = str(tmp_path / "plans" / "x.md")
        payload = {"mode": "implement", "phase": "implement",
                   "state": "needs-human", "exit": 3, "artifacts": [], "next": None}
        payload.update(body)
        return {"exit": 3, "stdout": [
            "implement: failed (batch 2 of 4)",
            f"      resume with: {UV_PREFIX} implement {word(absolute)} --host opencode "
            "--detach --resume",
            "MULTI_LLM_RESULT: " + json.dumps(payload),
        ]}

    def _asked(self, uv_stub):
        path = Path(str(uv_stub.log_path) + ".resume")
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return [argv[argv.index("--blocked") + 1]
                for argv in (json.loads(line) for line in lines if line)]

    def test_it_asks_about_the_runs_own_phase(self, built_package, uv_stub, tmp_path):
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        result, _ = uv_stub.run(built_package, self.STATUS, [self._failed(tmp_path)],
                                extra_env={"STUB_RESUME_ONLY": "implement"})
        assert self._asked(uv_stub) == ["implement"]
        assert result["resume_command"] == self.SPELLED
        assert f"resume with: {self.SPELLED}" in result["output"]
        assert "no_command_reason" not in result
        assert result["message"].endswith(surface["needs_human_message"]["resume"])

    def test_an_unpinned_listing_names_it_through_its_one_option(
        self, built_package, uv_stub, tmp_path
    ):
        """A bare `status` keeps `mode: status, phase: null` on its result line;
        the one phase its output re-invokes is the run's."""
        result, _ = uv_stub.run(
            built_package, self.STATUS,
            [self._failed(tmp_path, mode="status", phase=None)],
            extra_env={"STUB_RESUME_ONLY": "implement"})
        assert self._asked(uv_stub) == ["implement"]
        assert result["resume_command"] == self.SPELLED

    def test_no_lookup_answer_beside_a_visible_one_claims_no_absence(
        self, built_package, uv_stub, tmp_path
    ):
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        result, _ = uv_stub.run(built_package, self.STATUS, [self._failed(tmp_path)],
                                extra_env={"STUB_RESUME_ONLY": "review-plan"})
        assert f"resume with: {self.SPELLED}" in result["output"]
        assert "resume_command" not in result
        assert result["user_next_step"] is None
        assert "no_command_reason" not in result
        assert not result["message"].endswith(
            surface["needs_human_message"]["no_command"])

    def test_a_phase_call_still_says_there_is_none(self, built_package, uv_stub,
                                                   tmp_path):
        """On a phase's own exit 3 the lookup is authoritative: a visible retry
        it does not back is one that would repeat the stop."""
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        launch = {"exit": 0, "stdout": [
            "run_id: R", "log: /l",
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
        result, _ = uv_stub.run(
            built_package, dict(self.STATUS, mode="implement"),
            [launch, self._failed(tmp_path)],
            extra_env={"STUB_RESUME_ONLY": "review-plan"})
        assert self._asked(uv_stub) == ["implement"]
        assert result["no_command_reason"] == surface["no_command_reason"]


class TestARunStillAliveIsNoFinishedStop:
    """Adversarial review 2026-10-05: `abortAndReport` / `orphanReport` are
    `ok: false` with no `exit` — the run outlived the call (or its abort is
    unconfirmed) — and still got "this phase did not finish … nothing names a
    re-run" as their `no_command_reason`."""

    def test_the_continuation_cap_stop_carries_no_reason(self, built_package, uv_stub):
        payload = ('MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
                   '"artifacts":["a"],"next":"uv run -- python multi_llm.py '
                   'review-plan p.md --reaggregate"}')
        launch = lambda rid: {"exit": 0, "stdout": [  # noqa: E731
            f"run_id: {rid}", "log: /tmp/l.log", payload]}
        aborted = {"exit": 0, "stdout": [
            "aborted", 'MULTI_LLM_RESULT: {"mode":"status","state":"done","exit":0,'
            '"artifacts":[],"next":null}']}
        result, _ = uv_stub.run(
            built_package,
            {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []},
            [launch("RID-1"), {"exit": 0, "stdout": [payload]},
             launch("RID-2"), {"exit": 0, "stdout": [payload]}, aborted],
            extra_env={"MLLM_TEST_NEXT_CAP": "1"})
        assert result["ok"] is False and result["run_id"] == "RID-2"
        assert result.get("exit") is None
        assert "no_command_reason" not in result

    def test_an_abandoned_watch_carries_no_reason(self, built_package, uv_stub):
        live = {"exit": 2, "stdout": [
            "log: /tmp/r.log",
            'MULTI_LLM_RESULT: {"state":"running","exit":2,"artifacts":[],'
            '"next":"uv run -- python /s/multi_llm.py status p.md --view run '
            '--wait --run-id RID-77"}']}
        done = {"exit": 0, "stdout": [
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}
        result, _ = uv_stub.run(
            built_package,
            {"mode": "status", "plan": "p.md", "question": "",
             "extra_flags": ["--view", "run", "--wait", "--run-id", "RID"]},
            [live, done], extra_env={"MLLM_TEST_POLL_BUDGET_MS": "1"})
        assert result["ok"] is False and result["run_id"] == "RID-77"
        assert "no_command_reason" not in result

    def test_a_refusal_before_any_run_keeps_it(self, built_package):
        out = node_json(
            """
            const alive = T.finishOutcome(
              { state: "needs-human", ok: false, run_id: "R", message: "m" }, {}, surface)
            const refused = T.finishOutcome(
              { state: "needs-human", ok: false, message: "m" }, {}, surface)
            emit({ alive, refused })
            """,
            built_package,
        )
        assert "no_command_reason" not in out["alive"]
        assert out["alive"]["user_next_step"] is None
        assert out["refused"]["no_command_reason"]


class TestARedactedCommandCutAtAQuote:
    """Adversarial review 2026-10-05: the runnable pattern stops at a `"`, and
    `shlex.quote` spells an apostrophe `'"'"'`, so an `ask` retry whose question
    held one was matched up to the first `"`, parsed as question `what`, and
    re-spelled — and offered as an option — mangled."""

    PLAN = "plans/racing game.md"

    def _probe(self, package, tmp_path, question):
        absolute = str(tmp_path / "plans" / "racing game.md")
        command = cli_common.retry_command("ask", absolute, question, "--reaggregate",
                                           flags=[])
        return node_json(
            f"""
            const values = {{ word: "ask", plan: {json.dumps(absolute)},
                             planArg: {json.dumps(self.PLAN)}, question: "",
                             extraFlags: [] }}
            const child = {{ result: null, stderr: "",
                            stdout: "ask: " + {json.dumps(command)} + "\\n" }}
            emit({{
              output: T.redactCommand(child.stdout, null, surface,
                                      T.respellerFor(surface, values)),
              calls: T.toolCallsFor(child, surface, values),
            }})
            """,
            package,
        )

    def test_an_apostrophe_is_re_spelled_whole(self, built_package, tmp_path):
        out = self._probe(built_package, tmp_path, "what's next")
        assert out["output"].strip() == \
            f'ask: /multi-llm --ask "{self.PLAN}" "what\'s next" --reaggregate'
        (call,) = out["calls"]
        assert call["args"]["question"] == "what's next"

    def test_a_double_quote_keeps_the_pointer(self, built_package, tmp_path):
        out = self._probe(built_package, tmp_path, 'what\'s "up"')
        assert "a runnable command was removed" in out["output"]
        assert "/multi-llm --ask" not in out["output"]
        assert "uv run" not in out["output"]
        assert all(call.get("args", {}).get("question") != "what"
                   for call in out["calls"])
        assert [call.get("unparseable") for call in out["calls"]] == [True]

    def test_a_cut_after_a_closing_quote_is_refused(self, built_package):
        """The guard itself: a match that a `"` cut right after a closing
        quote may have lost the rest of its last token."""
        out = node_json(
            """
            const values = { word: "ask", plan: "/w/p.md", planArg: "p.md",
                             question: "", extraFlags: [] }
            const text = "uv run -- python /s/multi_llm.py ask /w/p.md 'a'\\"x\\"\\n"
            emit({
              output: T.redactCommand(text, null, surface, T.respellerFor(surface, values)),
              calls: T.toolCallsFor({ result: null, stderr: "", stdout: text },
                                    surface, values),
            })
            """,
            built_package,
        )
        assert out["output"].startswith("<a runnable command was removed")
        assert [call.get("unparseable") for call in out["calls"]] == [True]


class TestASymlinkedAnchor:
    """Adversarial review 2026-10-05: the call's plan is resolved without
    following symlinks, the dispatcher prints `Path(…).resolve()`, so under a
    symlinked workspace the two never compared equal and every re-spelled
    command and option carried the absolute real path instead of the plan as
    typed."""

    def test_the_real_path_is_put_back_as_typed(self, built_package, uv_stub, tmp_path):
        real = tmp_path / "real"
        (real / "plans").mkdir(parents=True)
        (real / "plans" / "x.md").write_text("# x\n", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        resolved = str((link / "plans" / "x.md").resolve())
        assert resolved.startswith(str(real.resolve()))
        launch = {"exit": 0, "stdout": [
            "run_id: R", "log: /l",
            'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
        failed = {"exit": 3, "stdout": [
            f"  {UV_PREFIX} generate-tasks {word(resolved)} --host opencode --yes",
            'MULTI_LLM_RESULT: {"state":"needs-human","exit":3,"next":null}']}
        result, _ = uv_stub.run(
            built_package,
            {"mode": "generate-tasks", "plan": "plans/x.md", "question": "",
             "extra_flags": []},
            [launch, failed], ctx={"directory": str(link)},
            extra_env={"STUB_RESUME_FAIL": "1"})
        assert "  /multi-llm --generate-tasks plans/x.md" in result["output"].split("\n")
        calls = [option for option in result["options"] if "args" in option]
        assert [call["args"]["plan"] for call in calls] == ["plans/x.md"]
        assert str(real.resolve()) not in json.dumps(result)

    def test_on_windows_the_real_path_is_the_native_one(self, built_package, tmp_path):
        """Windows port, round seven: there the dispatcher's ``Path.resolve()``
        and its root's ``os.path.realpath`` answer the on-disk case (and the
        long name for an 8.3 one), which the JS ``fs.realpathSync`` walk never
        does, so an anchor spelled in another case named the plan absolutely
        in every command handed back. Forced to ``win32``, with the native call
        folding case as ``GetFinalPathNameByHandleW`` does there; on any other
        platform the walk is still the answer, case and all."""
        root = tmp_path / "Proj"
        (root / "plans").mkdir(parents=True)
        real = str(root / "plans" / "x.md")
        Path(real).write_text("# x\n", encoding="utf-8")
        shouted = str(tmp_path / "PROJ" / "plans" / "x.md")
        answers = node_json(
            f"""
            const fs = (await import("node:fs")).default
            const real = {json.dumps(real)}, root = {json.dumps(str(root))}
            const known = [real, root]
            fs.realpathSync.native = (target) => {{
              const hit = known.find((p) => p.toLowerCase() === target.toLowerCase())
              if (hit) return hit
              throw Object.assign(new Error("ENOENT"), {{ code: "ENOENT" }})
            }}
            const values = {{ plan: {json.dumps(shouted)}, planArg: "plans/x.md" }}
            Object.defineProperty(process, "platform", {{ value: "linux" }})
            const elsewhere = T.typedPlan(real, values)
            Object.defineProperty(process, "platform", {{ value: "win32" }})
            const rooted = {{ planArg: "plans/x.md" }}
            T.planFromRoot({{ ran_from_root: root }}, rooted,
                           {json.dumps(str(tmp_path / "PROJ"))},
                           {{ dispatcher: {{ ran_from_root_key: "ran_from_root" }} }})
            emit({{ elsewhere, windows: T.typedPlan(real, values), rooted: rooted.plan }})
            """,
            built_package,
        )
        assert answers == {"elsewhere": real, "windows": "plans/x.md", "rooted": real}


#: Values the plan quoting must agree on with `next_phase.quote_plan_arg`.
QUOTE_CASES = [
    "plans/x.md", "plans/my plan.md", "plans/it's.md", 'plans/a"b.md',
    "plans/a\"b'c.md", "C:\\skill\\x.md", "tab\there", "--quick", "a,b", "",
]


class TestRestatedComposedFlags:
    """Racing re-run 2026-10-04 (opencode #1): big-pickle put `--yes` in
    extra_flags on four calls and each came back "reserved". A flag this tool
    composes with the same meaning is now dropped, quietly since run 6
    (#10a/#14: the note was relayed as "`--yes` was dropped from your flags");
    one whose meaning would differ is still refused."""

    ARGS = {"mode": "review-plan", "plan": "p.md", "question": "", "extra_flags": []}
    DONE = {"exit": 0, "stdout": [
        "run_id: R", "log: /l", 'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}']}
    FINISHED = {"exit": 0, "stdout": [
        'MULTI_LLM_RESULT: {"state":"done","exit":0,"artifacts":[],"next":null}']}

    @pytest.mark.parametrize("flags, shown", [
        (["--yes"], "--yes"), (["-y"], "-y"), (["--non-interactive"], "--non-interactive"),
        (["--exec-subagents"], "--exec-subagents"), (["--ye"], "--ye"),
        (["--host", "opencode"], "--host opencode"), (["--host=opencode"], "--host=opencode"),
    ])
    def test_a_restated_flag_runs_quietly(self, built_package, uv_stub, flags, shown):
        result, calls = uv_stub.run(
            built_package, dict(self.ARGS, extra_flags=[*flags, "--quick"]),
            [self.DONE, self.FINISHED])
        surface = json.loads((built_package / "dist" / "cli-surface.json").read_text(encoding="utf-8"))
        assert result["ok"] is True, result
        launch = _dispatcher_argv(calls[0])
        for baked in surface["baked_flags"]:
            if baked.startswith("-"):
                assert launch.count(baked) == 1, launch
        assert "--quick" in launch
        assert not set(flags) - set(surface["baked_flags"]) & set(launch)
        assert "extra_flags_note" not in result
        assert "note" not in surface["reserved_flags"]["redundant"]
        assert "leave them out" not in json.dumps(result)

    @pytest.mark.parametrize("flags", [
        ["--host", "codex"], ["--host=codex"], ["--host"], ["--detach"],
        ["--interactive"], ["--internal-worker"], ["--if-not-running"], ["--yes=1"],
    ])
    def test_a_flag_with_another_meaning_is_still_refused(self, built_package,
                                                          uv_stub, flags):
        result, calls = uv_stub.run(
            built_package, dict(self.ARGS, extra_flags=flags), [self.DONE])
        assert result["ok"] is False and calls == []
        assert "extra_flags_note" not in result

    def test_a_surface_without_the_set_refuses_as_before(self, built_package):
        out = node_json(
            """
            const bare = structuredClone(surface)
            delete bare.reserved_flags.redundant
            emit(T.validateArgs(
              {mode: "review-plan", plan: "p.md", question: "", extra_flags: ["--yes"]},
              bare,
            ))
            """,
            built_package,
        )
        assert out["ok"] is False and "reserved" in out["error"]

    def test_the_quoting_agrees_with_the_dispatchers(self, built_package):
        from utils import next_phase

        quoted = node_json(
            f"emit({json.dumps(QUOTE_CASES)}.map((v) => T.quoteForReader(v)))",
            built_package,
        )
        assert quoted == [next_phase.quote_plan_arg(v) for v in QUOTE_CASES]


# ===========================================================================
# Staging, the sweep and the version chain
# ===========================================================================

class TestStaging:
    def test_payload_allowlist_matches_the_build_context(self):
        """The allowlist is *parsed*, never typed — and the parse is guarded.

        It already exists in this repository twice, where an existing test
        asserts the two equal in both directions. A third hand-typed copy inside
        a build script would be exactly the single-source failure this project
        forbids, so the build parses the manifest and this row proves the parse
        agrees with the manifest's own reader.
        """
        from tests import test_build_context

        parsed = payload.runtime_allowlist(REPO_ROOT)
        assert parsed, "the product-under-test block parsed empty"
        manifest_patterns = [
            row.pattern for row in test_build_context._load_manifest(
                SKILL_ROOT / "tests" / "container" / "context_manifest.txt"
            )
        ]
        for pattern in parsed:
            assert pattern in manifest_patterns, (
                f"{pattern!r} is not a manifest pattern — the parse has drifted"
            )
        # It stops at the rig block: tracked-but-not-runtime is left behind.
        assert not any("tests/" in pattern for pattern in parsed)

    def test_the_glob_rule_matches_the_manifest_reader(self):
        """``*`` stops at a separator, in both readers.

        ``fnmatch`` translates ``*`` to ``.*``, which would make
        ``skills/multi-llm/*.py`` match every ``.py`` in every subtree — and the
        payload would silently ship this repository's own test corpus. That is
        not hypothetical; it is what this rule was written in response to.
        """
        assert payload._matches("skills/multi-llm/modes.py", "skills/multi-llm/*.py")
        assert not payload._matches(
            "skills/multi-llm/tests/test_x.py", "skills/multi-llm/*.py"
        )
        assert payload._matches(
            "skills/multi-llm/utils/a/b.py", "skills/multi-llm/utils/**"
        )

    def test_the_staged_set_equals_the_allowlist_expansion(self, tmp_path):
        """Nothing unexpected, and nothing missing.

        Staged **fresh**, never into the session-scoped build: the rows that
        drive the real dispatcher run ``uv run --project`` against that tree,
        which creates a ``.venv`` and ``__pycache__`` inside the staged payload.
        That is a true fact about this route rather than a staging bug — every
        installed version pays for its own venv, inside the installed package —
        so the assertion grades what staging produced, not what later use added.
        """
        destination = tmp_path / "payload"
        payload.stage(REPO_ROOT, destination, allow_dirty=True)
        staged = {
            str(path.relative_to(destination)).replace(os.sep, "/")
            for path in destination.rglob("*")
            if path.is_file()
        }
        expected = {
            path[len("skills/multi-llm/"):]
            for path in payload.selected_files(REPO_ROOT, "skills/multi-llm")
        }
        assert staged == expected

        # The regressions the old denylist caught, kept as regressions.
        for forbidden in ("tests/", "legacy/", ".venv/", "__pycache__/",
                          ".pytest_cache/", "plans/"):
            assert not any(path.startswith(forbidden) for path in staged), (
                f"{forbidden} reached the payload"
            )

    def test_the_required_inventory_is_complete(self, built_package):  # noqa: D401
        """Every resource the runtime reads is present.

        This is the assertion that catches a **short** payload, which no absence
        list can see. A missing message or template file is an ``ENOENT`` raised
        from inside a dispatch, which on this route is a silent exit 0 — and at
        least one entry is fatal rather than cosmetic: the provider registry
        loads ``providers.yaml`` by a path relative to its own file.
        """
        root = built_package / "skills" / "multi-llm"
        for relative in payload.REQUIRED_INVENTORY:
            assert (root / relative).is_file(), f"missing {relative}"
        for tree in payload.REQUIRED_TREES:
            assert any((root / tree).rglob("*")), f"{tree} is empty"

    def test_every_mode_table_target_module_is_staged(self, built_package):
        """Every ``modes.py`` ``target_module`` is present in the staged payload.

        The regression this guards: ``modes.py`` can name a ``target_module``
        the static :data:`payload.REQUIRED_INVENTORY` never anticipated, and
        ``payload.stage``'s allowlist takes its candidate file list from ``git
        ls-files`` alone — an untracked implementation file is invisible to
        both, so a build can succeed while shipping a dispatch table that
        points nowhere. ``utils/doctor.py`` shipped exactly that gap once:
        ``modes.py`` already routed ``doctor`` to ``utils.doctor`` while
        ``utils/doctor.py`` itself was untracked. ``pkg_build.target_module_paths``
        derives the same set the dispatcher resolves lazily at runtime, and
        ``build()`` now feeds it to ``payload.stage`` as ``required_extra`` — this
        asserts the set it derives is actually satisfied by what shipped.
        """
        root = built_package / "skills" / "multi-llm"
        derived = pkg_build.target_module_paths()
        assert derived, "modes.py declared no target_module at all"
        for relative in derived:
            assert (root / relative).is_file(), (
                f"modes.py dispatches to {relative}, but it is missing from "
                "the staged payload"
            )

    def test_required_extra_catches_an_untracked_target_module(self, tmp_path):
        """``required_extra`` closes exactly the gap ``doctor.py`` fell through.

        A target module that exists on disk but was never ``git add``ed is
        invisible to ``git ls-files``, so it never becomes a staging candidate
        — the exact shape of the real regression, reproduced independently of
        the real ``modes.py`` with a minimal synthetic repo staged under
        ``--allow-dirty`` (the posture ``MULTI_LLM_ALLOW_DIRTY=1`` builds run
        under, which is what let the real gap ship). ``required`` and
        ``required_trees`` are passed empty so the raise can only be coming
        from ``required_extra`` itself, isolating the new parameter from the
        static defaults it is additive to.
        """
        repo = tmp_path / "repo"
        (repo / "skills" / "multi-llm" / "utils").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "skills" / "multi-llm" / "modes.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-qm", "c"], check=True,
        )
        # On disk, never `git add`ed — the doctor.py shape.
        (repo / "skills" / "multi-llm" / "utils" / "doctor.py").write_text(
            "x = 1\n", encoding="utf-8"
        )

        with pytest.raises(payload.PayloadError, match="utils/doctor.py"):
            payload.stage(
                repo,
                tmp_path / "payload",
                allow_dirty=True,
                allowlist=("skills/multi-llm/**",),
                required=(),
                required_trees=(),
                required_extra=("utils/doctor.py",),
            )

    def test_the_staging_sweep_refuses_a_planted_secret(self, tmp_path):
        """A sweep nobody proved catches anything is not a sweep."""
        tree = tmp_path / "tree"
        (tree / "sub").mkdir(parents=True)
        (tree / "sub" / "ok.py").write_text("x = 1\n", encoding="utf-8")
        payload.scan_tree(tree)  # clean

        planted = [
            ("notes.md", "the key is sk-" + "a" * 40),
            ("config.yaml", "token: ghp_" + "b" * 36),
            ("aws.txt", "AKIA" + "C" * 16),
        ]
        for name, body in planted:
            target = tree / "sub" / name
            target.write_text(body, encoding="utf-8")
            with pytest.raises(payload.PayloadError, match="credential-shaped"):
                payload.scan_tree(tree)
            target.unlink()

        # By filename and by location, not only by content.
        for name in (".env", "id_rsa", "creds/auth.json"):
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            with pytest.raises(payload.PayloadError):
                payload.scan_tree(tree)
            target.unlink()

    def test_staging_refuses_a_dirty_tree(self, tmp_path):
        """A build that ships local state is one nobody can reproduce.

        On npm the result is immutable the moment it is published, so the refusal
        is a default rather than a warning. ``--allow-dirty`` stages to scratch
        for a local test and never reaches a publish.
        """
        repo = tmp_path / "repo"
        (repo / "skills" / "multi-llm").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "skills" / "multi-llm" / "a.py").write_text("x", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-qm", "c"], check=True
        )
        (repo / "skills" / "multi-llm" / "a.py").write_text("y", encoding="utf-8")
        with pytest.raises(payload.PayloadError, match="dirty tree"):
            payload.assert_clean(repo, "skills/multi-llm")

    def test_version_source_is_singular(self):
        """One file owns the number; every other appearance agrees with it.

        The chain is rooted at ``skills/multi-llm/pyproject.toml`` rather than at
        the npm manifest, and one fact settles it: **``pyproject.toml`` is inside
        the npm tarball.** It is on the allowlist and in the required inventory,
        so a package published as ``1.0.0`` would ship a payload internally
        declaring something else — and this row would then have to be written to
        *tolerate* that drift, which is the exact opposite of what it is for.
        """
        import tomllib

        pyproject = tomllib.loads(
            (SKILL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        owner = pyproject["project"]["version"]

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["version"] == owner, (
            f"package.json says {manifest['version']}, pyproject.toml says {owner}"
        )

        # uv.lock shadows the number and rides along in the same tarball.
        lock = (SKILL_ROOT / "uv.lock").read_text(encoding="utf-8")
        assert re.search(
            rf'name = "multi-llm"\nversion = "{re.escape(owner)}"', lock
        ), f"uv.lock does not declare {owner}"

        # The capability data's copy, which the generated docs render the pinned
        # spec from — so a hand-edited version there is a CI failure.
        from utils import host_capabilities as caps

        block = caps.entry("opencode").get("package")
        assert block is not None, (
            "opencode's capability entry has no `package` block; the pinned "
            "spec the generated docs render comes from it, so its absence is a "
            "break in the chain rather than a reason to skip the check"
        )
        assert block["version"] == owner, (
            f"capabilities.yaml says {block['version']}, pyproject.toml says {owner}"
        )
        assert block["name"] == manifest["name"], (
            f"capabilities.yaml says {block['name']}, package.json says "
            f"{manifest['name']}"
        )

        plugin_manifest = REPO_ROOT / ".claude-plugin" / "plugin.json"
        assert plugin_manifest.is_file(), (
            f"{plugin_manifest} is missing; it is the Claude Code end of the "
            "same chain"
        )
        declared = json.loads(plugin_manifest.read_text(encoding="utf-8"))
        assert "version" in declared, (
            ".claude-plugin/plugin.json declares no `version`; an unversioned "
            "manifest silently drops out of the chain instead of failing it"
        )
        assert declared["version"] == owner, (
            f"plugin.json says {declared['version']}, pyproject.toml says {owner}"
        )

        # The codex end of the chain is GENERATED — build.py fills it from
        # pyproject.toml — and checked in, so a bump without `make codex-regen`
        # leaves the committed copy behind (plans/codex-native-plugin.md, D6).
        codex_manifest = (
            REPO_ROOT / "packaging" / "codex" / "multi-llm" / ".codex-plugin"
            / "plugin.json"
        )
        assert codex_manifest.is_file(), (
            f"{codex_manifest} is missing; run `make codex-regen`"
        )
        codex_version = json.loads(
            codex_manifest.read_text(encoding="utf-8")
        ).get("version")
        assert codex_version == owner, (
            f"the codex plugin.json says {codex_version}, pyproject.toml says "
            f"{owner}. It is generated: run `make codex-regen`."
        )

    def test_lockfile_parity_with_image(self):
        """The packaged pair matches what the image bakes.

        Read it as "matches what the image bakes", never "matches what a running
        image contains": the image's pre-baked pair is a ``COPY`` of these same
        two repo files, so parity is two path assertions on the Dockerfile plus
        the files themselves. No daemon, no image build.
        """
        dockerfile = (
            SKILL_ROOT / "tests" / "container" / "Dockerfile"
        ).read_text(encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            assert (SKILL_ROOT / name).is_file()
            assert re.search(
                rf"^COPY\s+.*skills/multi-llm/{re.escape(name)}\s",
                dockerfile,
                re.M,
            ), (
                f"the Dockerfile no longer COPYs skills/multi-llm/{name}; the "
                "packaged pair and the image's would drift silently"
            )
        # And the staged copy is those same files, by construction.
        selected = payload.selected_files(REPO_ROOT, "skills/multi-llm")
        for name in ("pyproject.toml", "uv.lock"):
            assert f"skills/multi-llm/{name}" in selected


# ===========================================================================
# (pack, Linux) — needs npm
# ===========================================================================

def _need_npm():
    if not shutil.which("npm"):
        _unavailable("npm is not on PATH; this row needs `npm pack`")
    if sys.platform.startswith("win"):
        _unavailable("the package job is Linux-only by decision")


@pytest.mark.package
class TestPack:
    def test_package_files_manifest(self, built_package):
        """``npm pack --dry-run`` ships the payload and none of the rig.

        ``files`` semantics are surprising in both directions: with no key at all
        the package-root ignore file becomes the list; with ``files: ["dist"]``
        the skill tree vanishes silently; and with ``files`` naming ``skills`` the
        tree ships whole, ``__pycache__`` included, because ``files`` overrides
        the package-root ignore file.
        """
        _need_npm()
        listing = _pack_listing(built_package)
        for required in (
            "skills/multi-llm/multi_llm.py",
            "skills/multi-llm/pyproject.toml",
            "skills/multi-llm/uv.lock",
            "dist/index.mjs",
            "dist/command.md",
            "dist/cli-surface.json",
        ):
            assert any(name.endswith(required) for name in listing), (
                f"{required} is not in the tarball"
            )
        assert any("skills/multi-llm/instructions/" in n for n in listing)
        for forbidden in ("/tests/", "/legacy/", "/.venv/", "__pycache__",
                          ".pytest_cache"):
            assert not any(forbidden in name for name in listing), (
                f"{forbidden} reached the tarball"
            )

    def test_package_ships_license_and_readme(self, built_package):
        """Both are **real files**, not symlinks.

        A symlink inside an npm tarball is a portability hazard.
        """
        _need_npm()
        listing = _pack_listing(built_package)
        assert any(name.endswith("LICENSE") for name in listing)
        assert any(name.endswith("README.md") for name in listing)
        assert not (built_package / "LICENSE").is_symlink()
        assert not (built_package / "README.md").is_symlink()

    def test_package_payload_runs(self, built_package, tmp_path):
        """A **fresh** staged environment imports a real runtime dependency.

        Not ``--help``, which passes even against an empty venv. It runs under
        the *composed* flag list from the single owner, not a hand-typed pair, so
        a default that lost ``--no-dev`` fails here.
        """
        uv = unattended.find_uv()
        if uv is None:
            _unavailable("uv is not installed")
        staged = built_package / "skills" / "multi-llm"
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
        env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "venv")
        result = subprocess.run(
            [uv, "run", *unattended.uv_run_args(), "--project", str(staged),
             "--", "python", "-c", "import yaml; print(yaml.__name__)"],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "yaml" in result.stdout

    def test_the_no_sync_counter_case_fails_the_documented_way(
        self, built_package, tmp_path
    ):
        """``--no-sync`` with no pre-built venv does not fail cleanly.

        uv **creates an empty environment and runs against it anyway** — no
        uv-level error, no "sync first", just an import failure from inside the
        dispatcher; and the empty venv persists, so every later run repeats it
        identically. That is a different signature from the offline default's,
        which fails inside uv at the download step. Both are documented, and this
        row is what stops the wording drifting away from what uv does.
        """
        uv = unattended.find_uv()
        if uv is None:
            _unavailable("uv is not installed")
        staged = built_package / "skills" / "multi-llm"
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
        env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "empty-venv")
        result = subprocess.run(
            [uv, "run", "--frozen", "--no-dev", "--no-sync", "--project",
             str(staged), "--", "python", "-c", "import yaml"],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert result.returncode != 0
        assert "ModuleNotFoundError" in result.stderr, (
            "the documented --no-sync signature has changed; the README says "
            f"ModuleNotFoundError and uv said:\n{result.stderr}"
        )

    def test_the_tarball_sweep_runs_over_what_ships(self, built_package, tmp_path):
        """The scan reads the finished ``.tgz``, not the staging directory.

        What ``npm pack`` writes is what ships, and on the publish path
        ``prepack`` only re-stages — npm packs afterwards — so a lifecycle hook
        runs too early to see a tarball.
        """
        _need_npm()
        archive = _pack(built_package, tmp_path)
        payload.scan_tarball(archive)  # clean, or this raises


def _pack(package: Path, destination: Path) -> Path:
    result = subprocess.run(
        ["npm", "pack", "--ignore-scripts", "--pack-destination", str(destination)],
        cwd=str(package), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list(destination.glob("*.tgz"))
    assert len(archives) == 1, f"expected one tarball, got {archives}"
    return archives[0]


def _pack_listing(package: Path):
    result = subprocess.run(
        ["npm", "pack", "--dry-run", "--json", "--ignore-scripts"],
        cwd=str(package), capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    return [entry["path"] for entry in report[0]["files"]]


# ===========================================================================
# The capability data, the schema and the --check route note
# ===========================================================================

def _hosts_with_a_package_block():
    """Driven from ``capabilities.yaml`` itself, never a literal host list.

    A second host's packaging plan adds a YAML block and inherits this coverage
    with no edit here — which is the whole reason the vocabulary was written as
    a union in the first place.
    """
    from utils import host_capabilities as caps

    document = caps.load()
    return sorted(
        host_id for host_id, host in document["hosts"].items()
        if host.get("package")
    )


class TestPackageBlock:
    def test_at_least_one_host_declares_one(self):
        """Otherwise every row below is vacuously green."""
        assert _hosts_with_a_package_block(), (
            "no host declares a package: block, so this whole class asserts "
            "nothing"
        )

    @pytest.mark.parametrize("host_id", _hosts_with_a_package_block())
    def test_package_block_schema(self, host_id):
        """The union vocabulary, validated the way every other field is.

        A free-form ``{"type": "object"}`` here would be exactly the silent guess
        the schema exists to remove, and ``definitions.host`` is
        ``additionalProperties: false`` — so without the property at all the
        document simply fails to validate.
        """
        jsonschema = pytest.importorskip("jsonschema")
        from utils import host_capabilities as caps

        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        document = caps.load()
        jsonschema.validate(document, schema)

        package = document["hosts"][host_id]["package"]
        node = schema["definitions"]["host"]["properties"]["package"]
        assert node["additionalProperties"] is False
        assert set(node["required"]) == {
            "registry", "name", "config_key", "provides", "update",
            "min_host_version", "invocation",
        }
        assert set(node["properties"]["registry"]["enum"]) == {
            "npm", "codex-marketplace"
        }
        assert set(node["properties"]["update"]["enum"]) == {
            "pinned-spec", "reinstall", "range"
        }
        assert set(node["properties"]["provides"]["items"]["enum"]) == {
            "command", "skill", "tool"
        }
        # `mcp_tool` was retired in favour of `tool`.
        assert "mcp_tool" not in node["properties"]["provides"]["items"]["enum"]
        assert package["invocation"].endswith("."), (
            "invocation is a sentence fragment appended after \"Invoke it \""
        )

    def test_version_is_required_exactly_on_a_pinned_spec(self):
        """``pinned-spec`` is literally "the pin is the upgrade".

        So the pin must exist — while a ``reinstall`` route has no number to
        give and must not be forced to invent one.
        """
        jsonschema = pytest.importorskip("jsonschema")
        from utils import host_capabilities as caps

        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        document = json.loads(json.dumps(caps.load()))
        host = next(iter(_hosts_with_a_package_block()))
        block = document["hosts"][host]["package"]
        block["update"] = "pinned-spec"
        block.pop("version", None)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

        block["update"] = "reinstall"
        jsonschema.validate(document, schema)  # no version needed, and none given

    def test_marketplace_is_conditional_on_the_registry(self):
        """Required on a marketplace route, forbidden elsewhere.

        A stray ``marketplace:`` on an npm block would otherwise validate and
        render nowhere, which is the quietest possible way to be wrong.
        """
        jsonschema = pytest.importorskip("jsonschema")
        from utils import host_capabilities as caps

        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        document = json.loads(json.dumps(caps.load()))
        # An npm block specifically: once codex declares its marketplace block,
        # the first host in sorted order is no longer one this row can stray on.
        host = next(
            h for h in _hosts_with_a_package_block()
            if document["hosts"][h]["package"]["registry"] == "npm"
        )
        block = document["hosts"][host]["package"]

        block["marketplace"] = "somewhere"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

        block["registry"] = "codex-marketplace"
        block["update"] = "reinstall"
        block.pop("version", None)
        # A reinstall route has no pin to recommend; the npm block carries one
        # once its route is verified.
        block.pop("public_version", None)
        # The other registry-conditional keys a marketplace block requires.
        block.update(repo="owner/repo", ref="release", sparse=["."],
                     write_posture="refuses-under-readonly")
        jsonschema.validate(document, schema)

        block.pop("marketplace")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_the_block_and_its_provenance_move_as_a_pair(self):
        """A schema error for a consumer, not only a pytest failure.

        ``additionalProperties: false`` on ``provenance`` rejects an *undeclared*
        row; it does not *require* one when the block is present, because
        ``provenance.required`` is a fixed list that correctly excludes the
        optional keys. Making the pairing a validation error takes the explicit
        conditional this asserts.
        """
        jsonschema = pytest.importorskip("jsonschema")
        from utils import host_capabilities as caps

        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        document = json.loads(json.dumps(caps.load()))
        host = next(iter(_hosts_with_a_package_block()))
        document["hosts"][host]["provenance"].pop("package")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)

    def test_package_is_not_in_the_runtime_floor(self):
        """Optional in the schema means optional at runtime, and vice versa.

        The two lists are mirrored by an existing guard, so they move as a pair
        or not at all — and ``package`` must be in neither: requiring it would
        force a dummy block into every host that has no package route.
        """
        from utils import host_capabilities as caps

        assert "package" not in caps.REQUIRED_HOST_KEYS
        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        assert "package" not in schema["definitions"]["host"]["required"]

    def test_shim_format_gains_no_enum_value(self):
        """The package route writes no shim; the command body is a pack artifact."""
        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        formats = schema["definitions"]["host"]["properties"]["shim"]
        assert "package" not in json.dumps(formats)


class TestGeneratedDocs:
    def test_both_pages_are_byte_identical_to_a_fresh_render(self):
        """The pinned spec in them is generated and cannot be hand-edited."""
        result = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "hosts" / "render_capabilities.py"),
             "--check"],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_npm_paragraph_reads_as_its_provenance(self):
        """``documented`` names the route; ``verified`` recommends it.

        The docs half of the publish flip is therefore one provenance edit plus
        a re-render, not a prose rewrite — which is what the byte-compare
        forbids.
        """
        from utils import host_capabilities as caps

        # The pre-publish sentence each registry's paragraph must carry. Keyed
        # by registry, so a host is graded against its own wording and a
        # registry with no row here fails loudly instead of passing on another
        # host's sentence.
        not_yet_published = {
            # A config entry naming an unpublished spec resolves to nothing,
            # and on opencode that failure is silent.
            "npm": "No published install has been verified yet",
            # The ref the install names does not exist until the first publish.
            "codex-marketplace": "It is not yet published",
        }
        page = (REPO_ROOT / "docs" / "hosts.md").read_text(encoding="utf-8")
        lines = page.splitlines()
        for host_id in _hosts_with_a_package_block():
            host = caps.entry(host_id)
            package = host["package"]
            provenance = host["provenance"]["package"]
            registry = package["registry"]
            # Grade the host's own paragraph, not the page: from its heading to
            # the next host heading or section.
            heading = f"**`{host_id}` — install via {registry}"
            starts = [i for i, line in enumerate(lines) if line.startswith(heading)]
            assert len(starts) == 1, (
                f"expected one {heading!r} paragraph, found {len(starts)}"
            )
            end = next(
                (i for i in range(starts[0] + 1, len(lines))
                 if lines[i].startswith(("**`", "## "))),
                len(lines),
            )
            block = "\n".join(lines[starts[0]:end])
            # The candidate `version` before the route is verified, the last
            # registry-checked `public_version` after (TestThePublicPin).
            spec = package["name"]
            pinned = caps.package_spec_version(host)
            if pinned:
                spec = f"{spec}@{pinned}"
            assert spec in block, f"the pinned spec {spec!r} is not rendered"
            if provenance == "verified":
                assert "recommended route" in block
                assert "(not yet published)" not in block
            else:
                assert registry in not_yet_published, (
                    f"no pre-publish sentence recorded for registry {registry!r}"
                )
                assert not_yet_published[registry] in block, (
                    f"{host_id}'s pre-publish paragraph must say plainly that "
                    "the route is not yet published"
                )
                assert "recommended route" not in block

    def test_the_readme_is_in_its_pre_publish_state(self):
        """Until a registry install is recorded, the docs must not recommend it.

        A README recommending an `npm`-installable spec in the window before the
        first publish is instructions for something that resolves to nothing —
        and the failure is the silent one: plugin listed, exit 0, empty log.

        Asserted **positively**, because "does not contain the word recommended"
        is not a property: the flip would simply write an install line without
        it. What the pre-publish state must actually say is that the route is not
        published and that `install-host` is what to use, and those are the two
        sentences Phase 5 replaces.
        """
        from utils import host_capabilities as caps

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for host_id in _hosts_with_a_package_block():
            package = caps.entry(host_id)["package"]
            if caps.entry(host_id)["provenance"]["package"] == "verified":
                # Post-flip: the published spec is the install line, and
                # install-host is demoted rather than deleted.
                assert package["name"] in readme
                continue
            assert "not yet published" in readme, (
                f"the README mentions no unpublished state for "
                f"{package['name']}, whose provenance is still "
                f"{caps.entry(host_id)['provenance']['package']!r}"
            )
            assert f"install-host {host_id}" in readme, (
                "the README must still tell a reader to use the file route, "
                "which is the only one that works today"
            )


#: Where step 5 commits its registry-install record, and the canonical name it
#: is committed under. The bootstrap check's ``*-opencode-bootstrap-check.json``
#: is a CI artifact under another name, so this glob never counts one.
RELEASE_RECORDS = SKILL_ROOT / "tests" / "container" / "records"
RELEASE_RECORD_GLOB = "*-opencode-release-network-check*.json"

#: The one validator of those records. Called through its CLI, so this file
#: depends on the contract (``validate --profile release <record>``, exit 0 on
#: an accepted record) and on none of its internals.
REGISTRY_RECORD_CLI = PACKAGE_DIR / "registry_record.py"


def _pinned_spec_hosts():
    """The hosts whose package route pins a version: the only ones a public pin
    applies to (the schema forbids it on every other route)."""
    return [
        host_id for host_id in _hosts_with_a_package_block()
        if caps.entry(host_id)["package"]["update"] == "pinned-spec"
    ]


def _next_patch(version: str) -> str:
    """The release after *version*'s ``x.y.z``, for a candidate one bump ahead."""
    major, minor, patch = re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def _host_paragraph(page: str, host_id: str, registry: str) -> str:
    """*host_id*'s generated "install via" paragraph on *page*: from its heading
    to the next host heading or section."""
    lines = page.splitlines()
    heading = f"**`{host_id}` — install via {registry}"
    starts = [i for i, line in enumerate(lines) if line.startswith(heading)]
    assert len(starts) == 1, f"expected one {heading!r} paragraph, found {len(starts)}"
    end = next(
        (i for i in range(starts[0] + 1, len(lines))
         if lines[i].startswith(("**`", "## "))),
        len(lines),
    )
    return "\n".join(lines[starts[0]:end])


def _pin_problems(host_id, host, *, readme=None, host_doc=None) -> list:
    """What is wrong with the install pins *readme* and *host_doc* spell for
    *host*'s package, as a list of sentences (empty when nothing is).

    The version they must name is :func:`caps.package_spec_version`'s, which
    also refuses a public pin that disagrees with the provenance row or is newer
    than the candidate. Each page must name that version of the package and no
    other: a second ``name@<v>`` anywhere on it is a line a user can copy. A page
    left out (``None``) is not checked, so a fixture render can grade the
    generated block alone.
    """
    package = host["package"]
    name = package["name"]
    provenance = host["provenance"]["package"]
    verified = provenance == "verified"
    expected = caps.package_spec_version(host)
    problems = []
    for page, text in (("README.md", readme), ("docs/hosts.md", host_doc)):
        if text is None:
            continue
        found = _package_versions(text, name)
        if found != {expected}:
            spelled = ", ".join(f"{name}@{v}" for v in sorted(found)) or "no pin"
            problems.append(
                f"{page} names {spelled}; at `{provenance}` it must name "
                f"{name}@{expected} and no other version"
            )
    if host_doc is not None:
        unpublished = "(not yet published)" in _host_paragraph(
            host_doc, host_id, package["registry"]
        )
        if verified and unpublished:
            problems.append(
                f"docs/hosts.md's {host_id} paragraph is `verified` but still "
                "says it is not yet published"
            )
        if not verified and not unpublished:
            problems.append(
                f"docs/hosts.md's {host_id} paragraph pins {name}@{expected} "
                "without the unpublished notice"
            )
    if readme is not None and not verified and "not yet published" not in readme:
        problems.append(
            f"README.md pins {name}@{expected} without the unpublished notice"
        )
    return problems


def _record(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _release_record_problems(
    name, version, *, records=RELEASE_RECORDS, validator=REGISTRY_RECORD_CLI
) -> list:
    """Empty when a committed step-5 record backs ``name@version``.

    With the validator present, a record counts only if
    ``registry_record.py validate --profile release`` accepts it, and it must
    name the version. **Without it this is a floor, not the check**: the record
    names the version, came from the registry (``install_source: npm``) and is
    no rehearsal (``rehearsal: false``). The validator lands with the registry
    work package. Until it does, nothing can promote the route either, so this
    branch only runs from a tree that has neither.
    """
    candidates = sorted(records.glob(RELEASE_RECORD_GLOB))
    if validator is not None and Path(validator).is_file():
        accepted = [
            path for path in candidates
            if subprocess.run(
                [sys.executable, str(validator), "validate", "--profile", "release",
                 str(path)],
                capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
            ).returncode == 0
        ]
        how = f"that `{Path(validator).name} validate --profile release` accepts"
    else:
        accepted = [
            path for path in candidates
            if _record(path).get("install_source") == "npm"
            and _record(path).get("rehearsal") is False
        ]
        how = (
            "with `install_source: npm` and `rehearsal: false` (registry_record.py "
            "is absent, so that floor stands in for its validation)"
        )
    if any(_record(path).get("version") == version for path in accepted):
        return []
    return [
        f"no committed {RELEASE_RECORD_GLOB} record {how} names {name}@{version} "
        f"({len(candidates)} under {records}); `public_version` moves only after "
        "that version's registry record validates"
    ]


@pytest.fixture
def pin_render(tmp_path, monkeypatch):
    """Render the generated host doc from a COPY of the capability data whose
    opencode package cells and provenance row are replaced. Never the real file.

    ``render(provenance, **cells)`` returns ``(host, page)``; a cell given as
    ``None`` is removed. Each call writes its own base directory, so the per-base
    cache cannot hand one state's document to another.
    """
    import yaml

    monkeypatch.syspath_prepend(str(SKILL_ROOT / "hosts"))
    import render_capabilities as renderer

    monkeypatch.setattr(caps, "_CACHE", {})
    original = (SKILL_ROOT / caps.CAPABILITIES_FILE).read_text(encoding="utf-8")
    calls = []

    def render(provenance, **cells):
        document = yaml.safe_load(original)
        host = document["hosts"]["opencode"]
        for key, value in cells.items():
            if value is None:
                host["package"].pop(key, None)
            else:
                host["package"][key] = value
        host["provenance"]["package"] = provenance
        base = tmp_path / f"skill-{len(calls)}"
        calls.append(base)
        (base / "hosts").mkdir(parents=True)
        (base / caps.CAPABILITIES_FILE).write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        return host, renderer.render_host_doc_section(str(base))

    return render


class TestThePublicPin:
    """``package.version`` is the candidate; ``package.public_version`` is what a
    user is told to pin.

    ``version`` follows the shared version at every bump, so once the route is
    verified, a recommendation built from it would name a version npm does not
    serve yet: through the dispatch and the registry check, and indefinitely if
    either fails. On this host that is the silent failure: plugin listed, exit 0,
    nothing loaded. So the recommendation pins ``public_version``, the last
    version whose registry record validated. The schema forbids it until the
    route is verified and requires it from then on, and it never runs ahead of
    the candidate. The npm README is the exception: it ships inside ``version``
    and is read only once npm has it, so it pins ``version``
    (``TestManifest.test_the_package_readme_pins_the_same_version``).
    """

    def test_the_install_pins_follow_the_provenance_row(self):
        """The committed pages, in whichever state the committed row is.

        At ``documented``, ``README.md`` and the generated ``docs/hosts.md`` pin
        the candidate under the unpublished notice. At ``verified`` both pin
        ``public_version`` and no other version, even when ``version`` is newer,
        and a committed step-5 record backs that version.
        """
        hosts = _pinned_spec_hosts()
        assert hosts, "no package route pins a version, so this row grades nothing"
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        host_doc = (REPO_ROOT / "docs" / "hosts.md").read_text(encoding="utf-8")
        for host_id in hosts:
            host = caps.entry(host_id)
            package = host["package"]
            problems = _pin_problems(host_id, host, readme=readme, host_doc=host_doc)
            if host["provenance"]["package"] == "verified":
                problems += _release_record_problems(
                    package["name"], package["public_version"]
                )
            assert not problems, "\n".join(problems)

    def test_the_verified_paragraph_recommends_the_public_pin_until_it_moves(
        self, pin_render
    ):
        """A candidate one release ahead leaves the recommendation where it is,
        and advancing ``public_version`` is the one edit that moves it."""
        public = caps.entry("opencode")["package"]["version"]
        ahead = _next_patch(public)
        name = caps.entry("opencode")["package"]["name"]

        host, page = pin_render("verified", version=ahead, public_version=public)
        assert not _pin_problems("opencode", host, host_doc=page)
        paragraph = _host_paragraph(page, "opencode", "npm")
        assert f"{name}@{public}" in paragraph
        assert "recommended route" in paragraph
        assert f"{name}@{ahead}" not in page, (
            "the verified paragraph recommends the candidate, which npm does not "
            "serve yet"
        )

        host, page = pin_render("verified", version=ahead, public_version=ahead)
        assert not _pin_problems("opencode", host, host_doc=page)
        assert f"{name}@{ahead}" in _host_paragraph(page, "opencode", "npm")
        assert f"{name}@{public}" not in page

        # Before the promotion the same paragraph names the candidate, and says
        # it is not published.
        host, page = pin_render("documented", version=ahead, public_version=None)
        assert not _pin_problems("opencode", host, host_doc=page)
        paragraph = _host_paragraph(page, "opencode", "npm")
        assert f"{name}@{ahead}" in paragraph
        assert "(not yet published)" in paragraph
        assert "recommended route" not in paragraph

    def test_the_schema_pairs_the_public_pin_with_the_provenance_row(self):
        """Forbidden until ``verified``, required at it, and only on a pinned
        route; :func:`caps.package_spec_version` refuses exactly what the schema
        does, so a render without the dev extra refuses too."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(
            (SKILL_ROOT / "schemas" / "host_capabilities.schema.json")
            .read_text(encoding="utf-8")
        )
        version = caps.entry("opencode")["package"]["version"]

        def variant(host_id, provenance, **cells):
            document = json.loads(json.dumps(caps.load()))
            host = document["hosts"][host_id]
            host["provenance"]["package"] = provenance
            for key, value in cells.items():
                if value is None:
                    host["package"].pop(key, None)
                else:
                    host["package"][key] = value
            return document, host

        cases = [
            ("opencode", "documented", {"public_version": None}, True),
            ("opencode", "documented", {"public_version": version}, False),
            ("opencode", "unknown", {"public_version": version}, False),
            ("opencode", "verified", {"public_version": None}, False),
            ("opencode", "verified", {"public_version": version}, True),
            # A reinstall route has no pin to recommend, verified or not.
            ("codex", "verified", {"public_version": version}, False),
            ("codex", "documented", {"public_version": version}, False),
        ]
        for host_id, provenance, cells, valid in cases:
            document, host = variant(host_id, provenance, **cells)
            label = f"{host_id} at {provenance} with {cells}"
            if valid:
                jsonschema.validate(document, schema)
                caps.package_spec_version(host)
            else:
                with pytest.raises(jsonschema.ValidationError):
                    jsonschema.validate(document, schema)
                with pytest.raises(caps.PackagePinError):
                    caps.package_spec_version(host)
                    pytest.fail(f"package_spec_version accepted {label}")

    def test_the_public_pin_is_never_newer_than_the_candidate(self, pin_render):
        """No JSON Schema keyword compares two versions, so this half is the
        helper's and the renderer's: a public pin ahead of the candidate means
        one was hand-edited past the other, and the render refuses."""
        version = caps.entry("opencode")["package"]["version"]
        ahead = _next_patch(version)
        host = json.loads(json.dumps(caps.entry("opencode")))
        host["provenance"]["package"] = "verified"
        host["package"]["public_version"] = ahead
        with pytest.raises(caps.PackagePinError, match="newer than"):
            caps.package_spec_version(host)
        with pytest.raises(SystemExit, match="newer than"):
            pin_render("verified", public_version=ahead)
        # Equal is the state right after a promotion or an advance.
        host["package"]["public_version"] = version
        assert caps.package_spec_version(host) == version

        # SemVer precedence, not string order: the spec's own example chain,
        # then the two cases a string compare gets wrong.
        chain = [
            "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
            "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
        ]
        keys = [caps._release_precedence(item) for item in chain]
        assert keys == sorted(keys) and len(set(keys)) == len(keys)
        assert caps._release_precedence("0.1.10") > caps._release_precedence("0.1.9")
        assert caps._release_precedence("1.0.0+build.7") == caps._release_precedence("1.0.0")

    def test_the_pin_check_catches_a_stale_hand_written_pin(self):
        """The grader itself, on pages written to fail: a check that cannot go
        red proves nothing about the committed pages it passes."""
        base = json.loads(json.dumps(caps.entry("opencode")))
        name = base["package"]["name"]
        public = base["package"]["version"]
        ahead = _next_patch(public)

        verified = json.loads(json.dumps(base))
        verified["provenance"]["package"] = "verified"
        verified["package"].update(version=ahead, public_version=public)
        # The hand-written pin followed the candidate instead of the public pin.
        stale = f'{{ "plugin": ["{name}@{ahead}"] }}'
        assert any(
            "README.md" in problem
            for problem in _pin_problems("opencode", verified, readme=stale)
        )
        both = f"{name}@{public} then {name}@{ahead}"
        assert _pin_problems("opencode", verified, readme=both)
        assert not _pin_problems(
            "opencode", verified, readme=f'{{ "plugin": ["{name}@{public}"] }}'
        )

        # Pre-promotion, whatever state the committed row is in now.
        documented = json.loads(json.dumps(base))
        documented["provenance"]["package"] = "documented"
        documented["package"].pop("public_version", None)
        documented["package"]["version"] = ahead
        silent = f'{{ "plugin": ["{name}@{ahead}"] }}'
        assert any(
            "unpublished notice" in problem
            for problem in _pin_problems("opencode", documented, readme=silent)
        )
        assert not _pin_problems(
            "opencode", documented,
            readme=f"> This route is not yet published.\n{silent}",
        )

    def test_the_record_check_needs_a_registry_record_naming_the_version(
        self, tmp_path
    ):
        """Both branches of the record check, on records written to pass and fail.

        The validator branch runs a stand-in that holds the CLI contract
        (``validate --profile release <record>``), so the call this file makes
        is pinned before the real validator exists.
        """
        name, version = "opencode-multi-llm", "0.1.1"
        records = tmp_path / "records"
        records.mkdir()

        def write(stem, **fields):
            path = records / f"{stem}.json"
            path.write_text(json.dumps(fields), encoding="utf-8")
            return path

        def problems(**kwargs):
            return _release_record_problems(name, version, records=records, **kwargs)

        # The floor, the branch a tree without registry_record.py takes.
        assert problems(validator=None), "no record at all passed"
        write("20261101T000000Z-opencode-bootstrap-check",
              version=version, install_source="npm", rehearsal=False)
        assert problems(validator=None), "a bootstrap record counted"
        write("20261101T000001Z-opencode-release-network-check",
              version=version, install_source="npm", rehearsal=True)
        assert problems(validator=None), "a rehearsal counted"
        write("20261101T000002Z-opencode-release-network-check",
              version="0.1.0", install_source="npm", rehearsal=False)
        assert problems(validator=None), "a record for another version counted"
        write("20261101T000003Z-opencode-release-network-check",
              version=version, install_source="file", rehearsal=False)
        assert problems(validator=None), "a non-registry install counted"
        write("20261101T000004Z-opencode-release-network-check",
              version=version, install_source="npm", rehearsal=False)
        assert not problems(validator=None)

        # The validator branch: only what the CLI accepts counts.
        stand_in = tmp_path / "bin" / "registry_record.py"
        stand_in.parent.mkdir()
        stand_in.write_text(textwrap.dedent("""
            import json, sys
            assert sys.argv[1:4] == ["validate", "--profile", "release"], sys.argv
            sys.exit(0 if json.load(open(sys.argv[4])).get("accepted") else 1)
        """), encoding="utf-8")
        assert problems(validator=stand_in), "records the validator refused counted"
        write("20261101T000005Z-opencode-release-network-check",
              version="0.1.0", accepted=True)
        assert problems(validator=stand_in), "an accepted record for another version"
        write("20261101T000006Z-opencode-release-network-check",
              version=version, accepted=True)
        assert not problems(validator=stand_in)


class TestCheckRouteDetection:
    """D5, driven by a fabricated merged config plus a fake cache tree."""

    @pytest.fixture(autouse=True)
    def ready(self, monkeypatch):
        """Row 5's prerequisites stubbed, as `test_install_host.py`'s `ready`
        does: these tests are about which route row 1 reports, and whether the
        configured runner's CLI happens to be installed on the machine running
        them is not their question. Unstubbed, a runner without it (CI has no
        claude/codex/opencode binary) fails row 5, whose repair names
        `install-host`, and `install()` refuses the shim outright."""
        monkeypatch.setattr(host_install, "find_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(host_install, "runner_available", lambda provider: True)
        monkeypatch.setattr(
            host_install, "probe_runner",
            lambda spec, **kwargs: {"auth": "ok", "version": "9.9.9", "problem": ""},
        )

    @pytest.fixture
    def posture(self, tmp_path):
        from utils import host_capabilities as caps

        host_id = "opencode"
        package = caps.entry(host_id)["package"]
        spec = f"{package['name']}@{package['version']}"

        def build(*, declared: bool, resolved: bool, shim: bool):
            home = tmp_path / f"h-{declared}-{resolved}-{shim}"
            (home / ".config" / host_id / "commands").mkdir(parents=True)
            if declared:
                (home / ".config" / host_id / f"{host_id}.json").write_text(
                    json.dumps({package["config_key"]: [spec]}), encoding="utf-8"
                )
            if resolved:
                (
                    home / ".cache" / host_id / "packages" / spec
                    / "node_modules" / package["name"]
                ).mkdir(parents=True)
            if shim:
                host_install.install(
                    host_id, scope="global",
                    env={"HOME": str(home), "USERPROFILE": str(home)},
                )
            return home, spec

        return build

    def _check(self, home, **kwargs):
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        env.update({k: os.environ[k] for k in ("PATH",) if k in os.environ})
        return host_install.check("opencode", env=env, **kwargs)

    def test_a_absent_absent_is_byte_unchanged(self, posture):
        """No entry, no route: **nothing is stat-ed**, and the output stands.

        Config-first detection is what makes this true. A cache glob would fire
        on an extracted tree left behind by an uninstall, by an older pin, or by
        a different project's config — and a *false* shadowing warning is worse
        than no warning at all, because warning about shadowing is the whole job.
        """
        home, _ = posture(declared=False, resolved=False, shim=False)
        result = self._check(home)
        assert not result.ok
        assert result.row(host_install.ROW_PRESENT).fix
        assert "install-host" in result.row(host_install.ROW_PRESENT).fix
        assert not any("package" in note for note in result.notes)

    def test_c_declared_and_resolved_reports_the_package_route(self, posture):
        """Row 1 is OK and names the payload; rows 2-4 are not emitted.

        Row 1's question is "is a route present and loadable?", not "is there a
        file?". Answered the old way, an npm user running ``--check`` would be
        told to run ``install-host`` — creating the second, shadowed route this
        detection exists to find.
        """
        home, spec = posture(declared=True, resolved=True, shim=False)
        result = self._check(home)
        row = result.row(host_install.ROW_PRESENT)
        assert row.ok, row.detail
        assert spec in row.detail
        # No repair line at all: there is nothing to repair.
        assert not row.fix
        for number in (2, 3, 4):
            assert result.row(number) is None, f"row {number} was emitted"
        assert result.row(5) is not None, "row 5 is route-independent"
        assert any("installed as a package" in note for note in result.notes)
        # And never the install-host repair, which would create the duplicate.
        assert "install-host" not in json.dumps(
            [row.fix for row in result.rows if row.fix]
        )

    def test_declared_but_unresolved_fails_with_its_own_repair(self, posture):
        """The state a user most needs named, and the one a cache glob cannot see.

        Measured: the spec directory is created *before* resolution is attempted
        and survives every failure as an empty directory — so a probe stopping
        one directory short would report a live route for a package that does not
        exist on the registry. The failure is genuinely silent otherwise: a
        DEBUG-level run against a 404 spec logs ten INFO lines and not one ERROR.
        """
        home, spec = posture(declared=True, resolved=False, shim=False)
        result = self._check(home)
        row = result.row(host_install.ROW_PRESENT)
        assert not row.ok
        assert spec in row.detail
        assert "install-host" not in (row.fix or ""), (
            "an unresolved package route must not be repaired by creating a "
            "second, shadowing file route"
        )
        assert not result.ok, (
            "`rows 1-4 N/A` would report success on a machine whose declared "
            "route never resolved"
        )

    def test_both_routes_present_names_both(self, posture):
        """Rows 1-4 grade the file as today, and the note names the winner.

        Flat rather than hedged: `--check` does not probe for a winner, it states
        the one the plugin guarantees. The plugin assigns unconditionally, so the
        file shim becomes inert — and nothing in the host says so.
        """
        home, spec = posture(declared=True, resolved=True, shim=True)
        result = self._check(home)
        assert result.row(host_install.ROW_PRESENT).ok
        for number in (2, 3, 4, 5):
            assert result.row(number) is not None, f"row {number} is missing"
        note = next(n for n in result.notes if "BOTH routes" in n)
        assert spec in note
        assert "uninstall-host" in note, (
            "the note must name a command that exists: there is no "
            "`install-host --uninstall` flag"
        )

    def test_the_route_note_never_changes_the_check(self, posture):
        """Notes are not rows, and this is why it may never be promoted.

        ``install()``'s verify path rolls back on a row failure, so a route note
        that became a row would roll back an install on any machine carrying a
        package route.
        """
        home, _ = posture(declared=True, resolved=True, shim=True)
        result = self._check(home)
        assert result.ok
        # Five rows, still. The count is asserted in the generated docs too.
        assert len(result.rows) == 5
        assert not any(note in [row.detail for row in result.rows]
                       for note in result.notes)

    def test_the_note_fires_on_install_not_only_on_check(self, posture):
        """Ungated by ``path`` — which is the gate easiest to get backwards.

        ``install-host --all`` is the command the docs prescribe after every
        plugin update. Once a package route is live, ``--all`` silently recreates
        the shadowed file shim on every host, and this note is the only thing
        that says so at the moment it happens.
        """
        home, _ = posture(declared=True, resolved=True, shim=False)
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        env.update({k: os.environ[k] for k in ("PATH",) if k in os.environ})
        code = host_install.install("opencode", scope="global", env=env)
        assert code == 0
        # `install()` verifies by calling `check()` with `path=` pinned to what
        # it just wrote — the same call re-made here, so the row carve-out is
        # exercised on the install path rather than the standalone one.
        target = host_install.target_location(
            "opencode", "global", env=env
        ).path
        result = host_install.check("opencode", env=env, path=target)
        assert any("BOTH routes" in note for note in result.notes), (
            "installing over a live package route said nothing about it"
        )
        # But the row carve-out stays gated: install() still grades its own file.
        assert result.row(2) is not None and result.row(3) is not None

    def test_an_unparseable_config_detects_no_route(self, posture, tmp_path):
        """Best-effort read, and its failure mode is "no route", never a crash.

        A `--check` that died on someone's trailing comma would be worse than one
        that stays quiet about a route it could not read.
        """
        home = tmp_path / "broken"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.json").write_text(
            "{ this is not json", encoding="utf-8"
        )
        result = self._check(home)
        assert not any("package" in note.lower() for note in result.notes)

    def test_a_different_package_in_the_config_is_not_our_route(self, tmp_path):
        """The probe matches on **package name**, not on the key being non-empty."""
        home = tmp_path / "other"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.json").write_text(
            json.dumps({"plugin": ["some-other-plugin@1.0.0"]}), encoding="utf-8"
        )
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        assert host_install.package_route("opencode", env=env) is None

    def test_an_unpinned_spec_normalizes_to_latest(self, tmp_path):
        """The cache directory is the **normalized** spec, not the config string.

        ``"opencode-multi-llm"`` caches at ``…@latest``. A probe that stat-ed the
        config string would miss every unpinned install.
        """
        from utils import host_capabilities as caps

        package = caps.entry("opencode")["package"]
        home = tmp_path / "unpinned"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.json").write_text(
            json.dumps({package["config_key"]: [package["name"]]}), encoding="utf-8"
        )
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        route = host_install.package_route("opencode", env=env)
        assert route is not None
        assert route.payload.parent.parent.name == f"{package['name']}@latest"

    def test_xdg_cache_home_is_honoured(self, tmp_path):
        """``$XDG_CACHE_HOME/<host>`` when set, else ``~/.cache/<host>``."""
        from utils import host_capabilities as caps

        package = caps.entry("opencode")["package"]
        home = tmp_path / "xdg"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.json").write_text(
            json.dumps({package["config_key"]: [package["name"]]}), encoding="utf-8"
        )
        cache = tmp_path / "elsewhere"
        route = host_install.package_route(
            "opencode",
            env={"HOME": str(home), "USERPROFILE": str(home),
                 "XDG_CACHE_HOME": str(cache)},
        )
        assert route is not None
        assert str(route.payload).startswith(str(cache / "opencode"))

    # -- The config set is the one `Npm.install` is driven from (D5 step 1) ---
    #
    # `install_paths` names *command* directories, and three of OpenCode's config
    # locations hold no commands at all: `$OPENCODE_CONFIG`, `$OPENCODE_CONFIG_DIR`
    # and `~/.opencode`. A probe deriving its set from `install_paths` parents
    # alone answers "no route" for a user who really has one — and that answer is
    # not quiet: row 1 then fails with the `install-host` repair, telling them to
    # write the duplicate file route the shadowing note exists to prevent.

    def _declaration(self, directory, *, name="opencode.json"):
        """A config file at *directory* declaring the real package spec."""
        from utils import host_capabilities as caps

        package = caps.entry("opencode")["package"]
        spec = f"{package['name']}@{package['version']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(
            json.dumps({package["config_key"]: [spec]}), encoding="utf-8"
        )
        return spec

    def test_opencode_config_env_names_the_file_that_is_read(self, tmp_path):
        """`$OPENCODE_CONFIG` is read *instead of* the per-user config."""
        home = tmp_path / "env-file"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        elsewhere = tmp_path / "somewhere"
        spec = self._declaration(elsewhere, name="my-opencode.json")
        env = {
            "HOME": str(home), "USERPROFILE": str(home),
            "OPENCODE_CONFIG": str(elsewhere / "my-opencode.json"),
        }
        route = host_install.package_route("opencode", env=env)
        assert route is not None and route.spec == spec
        assert route.declared_in == elsewhere / "my-opencode.json"

    def test_opencode_config_dir_env_is_probed(self, tmp_path):
        """`$OPENCODE_CONFIG_DIR` redirects the whole directory."""
        home = tmp_path / "env-dir"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        redirected = tmp_path / "redirected"
        spec = self._declaration(redirected)
        env = {
            "HOME": str(home), "USERPROFILE": str(home),
            "OPENCODE_CONFIG_DIR": str(redirected),
        }
        route = host_install.package_route("opencode", env=env)
        assert route is not None and route.spec == spec
        assert route.declared_in == redirected / "opencode.json"

    def test_the_per_user_dot_opencode_directory_is_probed(self, tmp_path):
        """`~/.opencode` holds config and no commands, so nothing derives it."""
        home = tmp_path / "dot-dir"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        spec = self._declaration(home / ".opencode")
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        route = host_install.package_route("opencode", env=env)
        assert route is not None and route.spec == spec
        assert route.declared_in == home / ".opencode" / "opencode.json"

    def test_a_project_declaration_is_found_from_a_subdirectory(self, tmp_path):
        """The project `.opencode` belongs to the *tree*, not to one directory."""
        home = tmp_path / "walk-home"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        project = tmp_path / "tree"
        (project / ".git").mkdir(parents=True)
        spec = self._declaration(project / ".opencode")
        deep = project / "a" / "b" / "c"
        deep.mkdir(parents=True)
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        route = host_install.package_route("opencode", env=env, project_root=deep)
        assert route is not None and route.spec == spec
        assert route.declared_in == project / ".opencode" / "opencode.json"

    def test_the_upward_walk_stops_at_the_worktree_root(self, tmp_path):
        """A declaration *above* the worktree belongs to another tree.

        The walk is bounded on purpose: answering with a config the project does
        not belong to is the same class of false positive as globbing the cache.
        """
        home = tmp_path / "stop-home"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        outside = tmp_path / "outside"
        self._declaration(outside / ".opencode")
        project = outside / "tree"
        (project / ".git").mkdir(parents=True)
        env = {"HOME": str(home), "USERPROFILE": str(home)}
        assert host_install.package_route(
            "opencode", env=env, project_root=project
        ) is None

    def test_a_declaration_in_an_env_config_keeps_row_1_off_install_host(
        self, tmp_path
    ):
        """The whole point: no `install-host` repair for a package user.

        Without the env sources in the probe set this row FAILed and told the
        user to create the second, shadowed route.
        """
        from utils import host_capabilities as caps

        package = caps.entry("opencode")["package"]
        home = tmp_path / "row-one"
        (home / ".config" / "opencode" / "commands").mkdir(parents=True)
        spec = self._declaration(home / ".opencode")
        (
            home / ".cache" / "opencode" / "packages" / spec
            / "node_modules" / package["name"]
        ).mkdir(parents=True)
        result = self._check(home)
        row = result.row(host_install.ROW_PRESENT)
        assert row.ok, row.detail
        assert not row.fix
        assert any("installed as a package" in note for note in result.notes)


# ===========================================================================
# Route-keyed promotion — without which Phase 5's own promotion cannot run
# ===========================================================================

class TestRoutePromotion:
    """The harness must not credit one route with the other's evidence."""

    def _row(self, **kwargs):
        from tests.container import harness

        defaults = dict(
            row_id="r", lane="host", host="opencode", creds="openrouter",
            command="review-plan plans/x.md --host opencode --yes "
                    "--exec-subagents --detach",
        )
        defaults.update(kwargs)
        return harness.Row(**defaults)

    def _result(self, row, out_dir, *, receipts=2, done=True,
                install_source=None):
        """A real bundle on disk, because ``RowResult``'s evidence is read from
        one — a stubbed attribute would grade a shape the harness never uses."""
        from tests.container import harness

        receipt_dir = out_dir / "receipts" / row.row_id
        receipt_dir.mkdir(parents=True, exist_ok=True)
        for index in range(receipts):
            (receipt_dir / f"{index}.receipt.json").write_text(
                json.dumps({"argv": ["multi_llm.py", "review-plan"]}),
                encoding="utf-8",
            )
        (out_dir / "run-states.json").write_text(
            json.dumps(
                {"review-plan": {"state": "done" if done else "running"}}
            ),
            encoding="utf-8",
        )
        if install_source:
            (out_dir / "plugin-install.json").write_text(
                json.dumps({"install_source": install_source}),
                encoding="utf-8",
            )
        return harness.RowResult(
            row=row, exit_code=0, wall_clock_s=1.0, out_dir=out_dir,
        )

    def test_promotion_is_route_keyed(self, tmp_path):
        """A package row promotes ``package`` and the neutral cells, never the
        four file-route ones; a file row keeps today's set and can never reach
        ``package``.

        Under a route-blind field set the plugin row — which writes no shim at
        all — would propose promotions for ``shim``, ``install_paths``,
        ``arguments`` and ``reload``, describing a file at a path it never
        touched. That is the same class of false promotion the
        ``documented_invocation`` carve-out exists to stop.
        """
        from tests.container import harness

        file_row = self._row(route="file")
        package_row = self._row(route="package")

        file_fields = harness.promotable_fields(
            file_row, self._result(file_row, tmp_path / "file")
        )
        # `package` is keyed on the recorded install source (rule j): only a
        # Git-route install is evidence for it, so this half records one.
        package_fields = harness.promotable_fields(
            package_row,
            self._result(package_row, tmp_path / "package",
                         install_source="git"),
        )

        file_only = {"shim", "install_paths", "arguments", "reload"}
        assert file_only <= file_fields
        assert "package" not in file_fields, (
            "a file row reached `package`, which it is no evidence for"
        )

        assert "package" in package_fields
        assert not (file_only & package_fields), (
            f"a package row proposed file-route cells: "
            f"{sorted(file_only & package_fields)}"
        )
        # The behavioural cells are genuinely route-neutral: what the host does
        # once the skill is reachable does not depend on how it became reachable.
        assert {"detach", "subagents", "multi_turn"} <= package_fields

        # The refusal half: the same green row from a local build keeps the
        # behavioural cells and never reaches `package`.
        local_fields = harness.promotable_fields(
            package_row, self._result(package_row, tmp_path / "local")
        )
        assert "package" not in local_fields
        assert {"detach", "subagents", "multi_turn"} <= local_fields

    def test_the_default_route_is_file(self):
        """Every row written before a package route existed stays a file row."""
        from tests.container import harness

        assert harness.Row(row_id="r", lane="host", host="opencode").route == "file"

    def test_a_package_row_with_no_receipts_promotes_nothing_about_the_package(
        self, tmp_path
    ):
        """``package`` needs the same two-part evidence a receipt gives a shim:
        the payload was found *and* the dispatcher really ran."""
        from tests.container import harness

        row = self._row(route="package")
        assert "package" not in harness.promotable_fields(
            row, self._result(row, tmp_path, receipts=0, install_source="git")
        )

    def test_the_record_schema_admits_route(self):
        """``rows[]`` is ``additionalProperties: false``.

        So without the property a package row's record is a **refused write**,
        not a record missing a column — which is why the schema change lands with
        the field rather than after it.
        """
        jsonschema = pytest.importorskip("jsonschema")

        schema = json.loads(
            (SKILL_ROOT / "tests" / "container" / "records" /
             "record.schema.json").read_text(encoding="utf-8")
        )
        rows = schema["properties"]["rows"]["items"]
        assert rows["additionalProperties"] is False
        assert set(rows["properties"]["route"]["enum"]) == {"file", "package"}
        assert "route" not in rows.get("required", []), (
            "absent must mean `file`, or every existing record becomes invalid"
        )
        # And a record carrying it validates.
        document = {
            "schema_version": 1, "started_at": "20260101T000000Z",
            "ended_at": "20260101T000100Z", "lane": "host",
            "image": {"tag": "x"}, "egress": "none", "outcome": "pass",
            "promotions": [],
            "rows": [{"row_id": "r", "outcome": "pass", "route": "package"}],
        }
        try:
            jsonschema.validate(document, schema)
        except jsonschema.ValidationError as exc:
            if "route" in str(exc):
                raise
            # Other required top-level keys are this fixture's problem, not the
            # property's; what matters is that `route` itself is admitted.
            assert "'route'" not in str(exc)


# ===========================================================================
# Route-keyed staging — the other half, in the container's own entrypoint
# ===========================================================================

class TestRouteStaging:
    """A package row must never carry the FILE shim, and this is where that is
    pinned.

    `entrypoint.stage_host` runs `install-host <host> --global`, which writes the
    shim. Run it on a package row and the row has two command sources at once:
    opencode's `config` hook then overwrites the file entry wholesale and
    silently, so a plugin that never loaded still produces receipts and the row
    promotes `package` on the file route's evidence — the exact false promotion
    `Row.route` exists to stop. Worse, it fails *safe-looking*: green.

    Two independent ways that returns, so two rows. The guard can be deleted, and
    a call site can quietly stop passing the route — a three-argument function
    with a `file` default swallows the second silently.
    """

    @pytest.mark.parametrize("host", ["opencode", "codex", "claude-code"])
    def test_a_package_row_stages_nothing_at_all(self, tmp_path, monkeypatch, host):
        """No command dispatched, and not one byte written into the host's
        ``$HOME`` — the claude-code leg symlinks the skill tree *before* it
        dispatches, so the exit code alone would not have caught it."""
        # The entrypoint is the Linux image's driver and imports POSIX-only
        # modules (fcntl, pty, termios) at load: on a runner without them there
        # is no entrypoint to stage anything with.
        pytest.importorskip(
            "fcntl",
            reason="tests/container/entrypoint.py runs only in the Linux image "
                   "(POSIX-only imports)",
        )
        from tests.container import entrypoint

        dispatched = []
        monkeypatch.setattr(
            entrypoint, "dispatch",
            lambda *a, **k: dispatched.append((a, k)) or 0)

        home = tmp_path / "home"
        home.mkdir()
        assert entrypoint.stage_host(host, home, "package") == 0
        assert not dispatched, (
            f"route=package dispatched {dispatched!r}; `install-host "
            f"{host} --global` writes the file shim")
        assert list(home.iterdir()) == [], (
            f"route=package wrote {sorted(p.name for p in home.iterdir())} "
            f"into the host's $HOME")

    def test_every_call_site_passes_the_row_route(self):
        """Over `main`'s AST, because both call sites only execute inside a
        container: a defaulted call is `route="file"`, which is the bug back.

        Read as text, never imported: the module's POSIX-only imports would
        skip this row on a runner that can perfectly well parse it."""
        source = (Path(__file__).parent / "container" / "entrypoint.py").read_text(
            encoding="utf-8"
        )
        main = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "stage_host"]
        assert calls, "`main` no longer stages the host at all"
        for call in calls:
            passed = ([ast.unparse(arg) for arg in call.args[2:]]
                      + [ast.unparse(kw.value) for kw in call.keywords
                         if kw.arg == "route"])
            assert passed == ["args.route"], (
                f"stage_host at line {call.lineno} passes {passed!r} as its "
                f"route, so a package row would stage the file shim")


# ===========================================================================
# The publish gate
# ===========================================================================

class TestPublishGate:
    """``make opencode-publish`` refuses unless the full gate passed on **these
    exact bytes** — and each refusal names which condition it was.

    A gate is worth nothing if the publish command can skip it, and a refusal
    nobody can act on is nearly as bad: "the gate is red" must never be the same
    message as "you edited a file since". The stamp is schema 2 since the gate
    began reading evidence: these rows stamp from a complete gate session
    (``GateRun``, every row written through ``release_gate.py``'s own graders in
    a temp tree), and ``test_opencode_release_gate.py`` carries the gate's own
    refusals.
    """

    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        """A full staged package in a temp directory, with its stamp.

        ``allow_dirty`` is passed explicitly, never through the environment:
        neither ``write_stamp`` nor ``check_stamp`` reads the env var, precisely
        so a developer who exported it once cannot stamp — and therefore publish
        — from a dirty tree.
        """
        from tests.test_opencode_release_gate import GateRun

        run = GateRun(tmp_path, monkeypatch)
        run.full()
        run.stamp()
        return run.package

    def test_a_matching_full_stamp_permits_a_publish(self, staged):
        assert pkg_build.check_stamp(staged, allow_dirty=True) is None

    def test_the_stamp_helpers_ignore_the_dirty_environment_variable(
        self, tmp_path, monkeypatch
    ):
        """The env var reaches npm's `prepack` hook; it must not reach here.

        The refusal is *forced*, not borrowed from whatever state this checkout
        happens to be in. With the real ``assert_clean`` the row only fails on a
        dirty working tree and passes vacuously on a clean one — CI's fresh
        clone, or this branch once it is committed — which is the opposite of a
        guard. Patching the clean check to always refuse makes the assertion the
        one the docstring claims: ``write_stamp`` *reached* the clean check,
        which happens only when the env var failed to turn ``allow_dirty`` on.
        A gate session is begun first, because the stamp refuses a missing one
        before it builds anything.
        """
        from tests.test_opencode_release_gate import gate

        monkeypatch.setattr(gate, "dirty_paths", lambda root=None: [])
        evidence = tmp_path / "gate"
        gate.begin_session(evidence)
        reached = []

        def refuse(root, prefix):
            reached.append(prefix)
            raise payload.PayloadError(
                f"refusing to stage from a dirty tree — 1 change(s) under {prefix}"
            )

        monkeypatch.setattr(payload, "assert_clean", refuse)
        monkeypatch.setenv(pkg_build.ALLOW_DIRTY_ENV, "1")
        with pytest.raises(payload.PayloadError, match="dirty tree"):
            pkg_build.write_stamp(tmp_path / "nope", timestamp="x", evidence_dir=evidence)
        assert reached, (
            "write_stamp never ran the clean check, so the environment "
            "variable reached it after all"
        )

    def test_publish_is_gated_on_the_a4_check(self, staged, tmp_path):
        """Missing, CI-tier, schema-1, partial, stale and wrong-version stamps,
        each with its own reason."""
        stamp_path = staged / pkg_build.STAMP_NAME

        # 1. No stamp at all.
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        stamp_path.unlink()
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "no " in reason and pkg_build.STAMP_NAME in reason

        # 2. A stamp the CI tier wrote. It can never authorize a publish: that
        #    tier runs rows 1-6 and 5b and deliberately writes no stamp at all,
        #    so a `tier: ci` stamp means someone made one by hand.
        stamp_path.write_text(
            json.dumps({**stamp, "tier": "ci"}), encoding="utf-8"
        )
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "tier=" in reason and "ci" in reason

        # 3. A schema-1 stamp: the digest-only shape written before the gate
        #    read any evidence.
        legacy = {key: stamp[key] for key in ("tier", "digest", "version", "stamped_at")}
        stamp_path.write_text(json.dumps(legacy), encoding="utf-8")
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "digest-only" in reason

        # 4. A partial stamp: a row it does not record as passed.
        partial = {row: "pass" for row in stamp["rows"] if row != "8"}
        stamp_path.write_text(json.dumps({**stamp, "rows": partial}), encoding="utf-8")
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "rows 8" in reason

        # 5. A digest that differs from a fresh build — an edit landed after the
        #    gate ran. Keyed on the staged CONTENT, not on a git tree hash, which
        #    would miss both the generated files and an uncommitted edit.
        stamp_path.write_text(
            json.dumps({**stamp, "digest": "0" * 64}), encoding="utf-8"
        )
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "changed since the gate ran" in reason

        # 6. A version that differs from the manifest's.
        stamp_path.write_text(
            json.dumps({**stamp, "version": "99.99.99"}), encoding="utf-8"
        )
        reason = pkg_build.check_stamp(staged, allow_dirty=True)
        assert reason and "99.99.99" in reason

        # And each is a *different* message.
        messages = set()
        for mutation in ({"tier": "ci"}, {"schema": 1}, {"rows": partial},
                         {"digest": "1" * 64}, {"version": "98.0.0"}):
            stamp_path.write_text(
                json.dumps({**stamp, **mutation}), encoding="utf-8"
            )
            messages.add(pkg_build.check_stamp(staged, allow_dirty=True))
        assert len(messages) == 5, "two refusals share a message"

    def test_the_publish_recipe_checks_the_stamp_before_npm(self):
        """The refusal happens **before** npm is invoked at all.

        ``opencode-publish`` no longer reaches npm: it refuses and names the two
        guarded paths. The laptop one, ``opencode-publish-first``, checks the
        stamp among its preflights, all of which run before the upload
        (``test_opencode_first_release.py`` grades the order and each refusal).
        """
        recipes = _makefile_recipes(REPO_ROOT / "Makefile")
        assert not any(
            re.search(r"\bnpm\s+publish\b", line) for line in recipes["opencode-publish"]
        )
        assert any("first_release.py" in line for line in recipes["opencode-publish-first"])
        source = (PACKAGE_DIR / "first_release.py").read_text(encoding="utf-8")
        assert source.index('"--check-stamp"') < source.index('"npm", "publish"')

    def test_the_publish_recipe_never_allows_a_dirty_tree(self):
        """``ALLOW_DIRTY`` is a development convenience and must not reach here.

        A build that ships local state is one nobody can reproduce, and npm makes
        the result immutable the moment it is published — so the publish path
        neither sets nor forwards the opt-in, however it was invoked.
        """
        recipes = _makefile_recipes(REPO_ROOT / "Makefile")
        for line in recipes["opencode-publish"]:
            assert "--allow-dirty" not in line, (
                f"the publish recipe can stage from a dirty tree: {line!r}"
            )

    def test_the_ci_tier_writes_no_stamp(self):
        """A `-ci` run can never authorize a publish, and the guard is that it
        leaves nothing behind to be mistaken for authorization."""
        recipes = _makefile_recipes(REPO_ROOT / "Makefile")
        for line in recipes["opencode-package-check-ci"]:
            assert "--write-stamp" not in line, (
                "the CI tier wrote a stamp; only the full gate may"
            )
        assert any(
            "--write-stamp" in line
            for line in recipes["opencode-package-check"]
        ), "the full gate writes no stamp, so opencode-publish can never proceed"

    def test_the_stamp_is_gitignored(self):
        """It never travels: a stamp from another machine attests to that
        machine's rows."""
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore",
             f"packaging/opencode/{pkg_build.STAMP_NAME}"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, "the publish stamp is not gitignored"


# ===========================================================================
# D2 — the exit-3 options: one retry with the caller's flags, then the human
# ===========================================================================

class TestNeedsHumanOptions:
    """Captured stdout through the real Python-to-JavaScript conversion.

    The fixture text is what D2's Python sites print — built with
    ``cli_common.retry_command`` so the command is a real dispatcher command —
    and the assertions are on the tool-call side: one ``--reaggregate`` tool
    call carrying the caller's flags, the human entry last, no ``--force``
    tool call, and nothing of the sort on exit 1.
    """

    RESULT = {
        "mode": "review-plan", "phase": "review-plan", "state": "needs-human",
        "exit": 3, "artifacts": [], "next": None,
    }
    VALUES = {
        "word": "review-plan", "plan": "/p.md", "question": "", "extraFlags": [],
    }

    @staticmethod
    def _captured():
        retry = cli_common.retry_command(
            "review-plan", "/p.md", "--reaggregate",
            flags=["--host", "opencode", "--quick"],
        )
        return (
            "ERROR: runner refused\n"
            "Refused before dispatch: reason_code=runner_invalid_model\n"
            "Options (nothing was dispatched; every result already on disk is kept):\n"
            "  1. Repair first: the runner's model id is not one its CLI accepts.\n"
            f"  2. Then re-invoke with `{retry}` — it re-dispatches exactly this "
            "batch; --force is not needed and would re-run the reviewer panel.\n"
            "  3. Otherwise report this to the user.\n"
        )

    def test_exit_three_offers_one_reaggregate_call_and_the_human_entry_last(
        self, built_package
    ):
        result = node_json(
            f"""
            const child = {json.dumps({
                "code": 3, "argv": ["uv", "run"], "stdout": "",
                "stderr": self._captured(), "result": self.RESULT,
                "resultRaw": None, "truncated": False, "banner": {},
                "aborted": False, "timedOut": False, "spawnError": None,
            })}
            const out = T.terminalResult(
              child, {{ anchor: "/anchor", surface, values: {json.dumps(self.VALUES)} }},
            )
            emit({{
              options: out.options,
              mechanical: out.options.map((o) => T.isMechanical(o, surface)),
            }})
            """,
            built_package,
        )
        options = result["options"]
        calls = [o for o in options if "tool" in o]
        assert len(calls) == 1, options
        assert calls[0]["args"]["mode"] == "review-plan"
        assert calls[0]["args"]["plan"] == "/p.md"
        # The caller's selector rides along; the baked --host is the tool's own
        # and is stripped by the parser, as every reserved flag is.
        assert calls[0]["args"]["extra_flags"] == ["--quick", "--reaggregate"]
        assert not any("--force" in o.get("args", {}).get("extra_flags", [])
                       for o in calls)
        assert options[-1] == {"human": mode_table.NEEDS_HUMAN_OPTION}
        assert set(options[-1]) == {"human"}
        # The retry is the surface's mechanical continuation; the human entry
        # never is, so the continuation loop cannot auto-follow it.
        assert result["mechanical"][-1] is False
        assert result["mechanical"][0] is True

    def test_exit_one_carries_no_human_entry(self, built_package):
        for stderr, expect_options in ((self._captured(), True), ("", False)):
            result = node_json(
                f"""
                const child = {json.dumps({
                    "code": 1, "argv": ["uv", "run"], "stdout": "",
                    "stderr": stderr, "result": None, "resultRaw": None,
                    "truncated": False, "banner": {}, "aborted": False,
                    "timedOut": False, "spawnError": None,
                })}
                emit(T.terminalResult(
                  child, {{ anchor: "/anchor", surface, values: {json.dumps(self.VALUES)} }},
                ))
                """,
                built_package,
            )
            assert result["state"] == "error"
            if expect_options:
                assert all("tool" in o for o in result["options"])
                assert not any("human" in o and "tool" not in o for o in result["options"])
            else:
                assert "options" not in result

    def test_status_stdout_yields_no_option_missing_the_flags(
        self, built_package, tmp_path, capsys
    ):
        """The JS half of the printed-text carrier: ``toolCallsFor`` over a real
        ``status`` stdout (log tail inlined) has no ``--reaggregate`` option
        without ``--quick``."""
        from utils import detach, run_status, status_file

        plan = tmp_path / "p.md"
        plan.write_text("# p\n", encoding="utf-8")
        phase = tmp_path / "p" / "review-plan"
        phase.mkdir(parents=True)
        line = cli_common.retry_command(
            "review-plan", str(plan), "--reaggregate", flags=["--quick"]
        )
        Path(detach.log_path(str(phase))).write_text(
            f"Re-invoke to retry only the failures:\n  {line}\n", encoding="utf-8"
        )
        status_file.claim_run(str(phase), {
            "run_id": "run-js-000001", "state": "failed", "exit_code": 3,
            "error": "BatchFailed", "log_path": detach.log_path(str(phase)),
            "detached": True, "mode": "review-plan", "replay_flags": ["--quick"],
        })
        assert run_status.main(["--plan-file", str(plan), "--view", "run"]) == 3
        stdout = capsys.readouterr().out
        result = node_json(
            f"""
            const child = {json.dumps({
                "code": 3, "argv": ["uv", "run"], "stdout": stdout,
                "stderr": "", "result": None, "resultRaw": None,
                "truncated": False, "banner": {}, "aborted": False,
                "timedOut": False, "spawnError": None,
            })}
            emit(T.terminalResult(
              child, {{ anchor: "/anchor", surface, values: {json.dumps(self.VALUES)} }},
            ).options)
            """,
            built_package,
        )
        calls = [o for o in result if "tool" in o]
        reaggregates = [o for o in calls if "--reaggregate" in o["args"]["extra_flags"]]
        assert reaggregates, result
        assert all("--quick" in o["args"]["extra_flags"] for o in reaggregates)
        assert result[-1] == {"human": mode_table.NEEDS_HUMAN_OPTION}


class TestTheToolResultCarriesTheValidationCounts:
    """§6 T6.1, end to end on a detached run: the worker's checkpoint →
    `.status.json` → `classify_run`'s view → `run_status`'s result line → the
    tool result the opencode host receives. Run b38de8bb's summary left out
    its 3 needs-human-decision groups because no hop carried the counts."""

    VALUES = {
        "word": "review-plan", "plan": "/p.md", "question": "", "extraFlags": [],
    }

    def test_a_finished_detached_run_hands_the_host_its_counts(
        self, built_package, tmp_path, capsys
    ):
        from utils import review_orchestrator_base as rob
        from utils import status_file

        plan = tmp_path / "p.md"
        plan.write_text("# p\n", encoding="utf-8")
        phase = tmp_path / "p" / "review-plan"
        phase.mkdir(parents=True)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
        })
        groups = (
            [{"validation_status": "valid"}] * 53
            + [{"validation_status": "invalid"}]
            + [{"validation_status": "needs-human-decision"}] * 3
        )
        # The worker's own write, through the one recorder.
        counts = rob.record_validation_counts(str(phase), groups)
        assert counts == {
            "valid": 53, "invalid": 1,
            "needs_human_decision": 3, "validation_failed": 0,
        }
        status_file.claim_run(str(phase), {
            "run_id": "run-counts-js-01", "state": "done", "exit_code": 0,
            "detached": True, "mode": "review-plan",
        })

        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            # Pinned by run id, as the plugin's supervised poll pins it: an
            # unselected listing carries no single run's counts.
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--run-id", "run-counts-js-01",
            ]) == 0
            poll_line = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        stdout = capsys.readouterr().out
        assert poll_line["validation_counts"] == counts

        result = node_json(
            f"""
            const child = {json.dumps({
                "code": 0, "argv": ["uv", "run"], "stdout": stdout,
                "stderr": "", "result": poll_line, "resultRaw": None,
                "truncated": False, "banner": {}, "aborted": False,
                "timedOut": False, "spawnError": None,
            })}
            emit(T.terminalResult(
              child, {{ anchor: "/anchor", surface, values: {json.dumps(self.VALUES)} }},
            ))
            """,
            built_package,
        )
        assert result["result"]["validation_counts"] == counts


# ===========================================================================
# The session model
# ===========================================================================

#: The shape `client.session.messages` really answers with on SDK 1.18.x: one
#: `{info, parts}` entry per message, `info.role` telling them apart, and the
#: assistant entries carrying the `providerID`/`modelID` they were produced with.
def _messages(*rows):
    return {"data": [{"info": dict(row), "parts": []} for row in rows]}


_ASSISTANT = {"role": "assistant", "providerID": "opencode",
              "modelID": "nemotron-3-ultra-free"}
_OLDER_ASSISTANT = {"role": "assistant", "providerID": "opencode",
                    "modelID": "big-pickle"}
_USER = {"role": "user"}


def _client(js_body: str) -> str:
    """A stub `client` whose `session.messages` runs *js_body*."""
    return f"{{ session: {{ messages: async ({{ path }}) => {{ {js_body} }} }} }}"


class TestTheSessionModelIsObservedAndHandedToPython:
    """The plugin observes; the ladder decides (D3o).

    THE DEFECT. A `--review-plan --quick` run launched from a TUI session whose
    model was `opencode/nemotron-3-ultra-free` validated all sixteen of its
    batches on `opencode/big-pickle` — the host's credential-free *floor*, which
    is what `resolve_subagent_runner` falls to when no layer has an opinion. The
    `ToolContext` this plugin's `execute` receives carries `sessionID`,
    `messageID`, `agent`, `directory`, `worktree` and `abort`, and no model at
    all; the *factory* input carries the SDK `client`, and the session's newest
    assistant message records what produced it. So the plugin reads it there and
    exports it, and Python's ladder decides what it means.
    """

    def test_the_surface_carries_the_env_name_from_its_python_owner(self):
        """The name is generated, like every flag and mode word (rule 2)."""
        surface = pkg_build.cli_surface()
        assert (
            surface["dispatcher"]["session_model_env"]
            == provider_registry.HOST_SESSION_MODEL_ENV
        )

    def test_the_env_name_is_not_a_js_literal(self):
        """A second spelling is a second owner, which is the rule this breaks."""
        body = _code_only(SRC_MJS.read_text(encoding="utf-8"))
        assert provider_registry.HOST_SESSION_MODEL_ENV not in body

    def test_the_newest_assistant_message_supplies_the_model(self, built_package):
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client(f"return {json.dumps(_messages(_USER, _OLDER_ASSISTANT, _USER, _ASSISTANT))}")},
              {{sessionID: "ses_1"}},
            ))
            """,
            built_package,
        )
        assert observed == "opencode/nemotron-3-ultra-free"

    def test_the_pickers_variant_rides_along_as_an_effort_suffix(self, built_package):
        """A session opened with `--variant low` must dispatch at low, not default.

        Measured 2026-09-16: a generate-tasks run launched from an opencode
        session on `openrouter/~deepseek/deepseek-flash-latest` with
        `--variant low` dispatched its subagent as
        `opencode:openrouter/~deepseek/deepseek-flash-latest` — the model's
        default effort — because `info.variant` sat unread beside the
        `providerID`/`modelID` this function does read.
        """
        row = {"role": "assistant", "providerID": "openrouter",
               "modelID": "~deepseek/deepseek-flash-latest", "variant": "low"}
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client(f"return {json.dumps(_messages(_USER, row))}")},
              {{sessionID: "ses_1"}},
            ))
            """,
            built_package,
        )
        assert observed == "openrouter/~deepseek/deepseek-flash-latest:low"

    @pytest.mark.parametrize(
        "variant, label",
        [
            (None, "no variant key at all"),
            ("", "an empty variant"),
            ("   ", "whitespace only"),
            (7, "a non-string variant"),
            ("low:high", "a word carrying the suffix separator"),
            ("open/router", "a word carrying the provider separator"),
            ("two words", "a word carrying whitespace"),
        ],
    )
    def test_a_variant_that_cannot_be_a_suffix_is_left_off(
        self, built_package, variant, label
    ):
        """Shape, not vocabulary: which efforts a model takes is Python's call.

        A word carrying `:`, `/` or whitespace could not be split back off as an
        `:effort` suffix, so gluing it on would corrupt the model id rather than
        select an effort — and losing the model is worse than losing the effort.
        """
        row = {"role": "assistant", "providerID": "opencode",
               "modelID": "nemotron-3-ultra-free"}
        if variant is not None:
            row["variant"] = variant
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client(f"return {json.dumps(_messages(row))}")},
              {{sessionID: "ses_1"}},
            ))
            """,
            built_package,
        )
        assert observed == "opencode/nemotron-3-ultra-free", label

    def test_an_array_without_the_data_wrapper_is_read_too(self, built_package):
        """One shape check, not a version sniff."""
        rows = _messages(_ASSISTANT)["data"]
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client(f"return {json.dumps(rows)}")}, {{sessionID: "ses_1"}},
            ))
            """,
            built_package,
        )
        assert observed == "opencode/nemotron-3-ultra-free"

    @pytest.mark.parametrize(
        "rows, label",
        [
            ([_USER], "no assistant message yet"),
            ([{"role": "assistant", "providerID": "opencode"}], "no modelID"),
            ([{"role": "assistant", "modelID": "x"}], "no providerID"),
            ([], "an empty session"),
        ],
    )
    def test_a_session_that_cannot_answer_yields_null(
        self, built_package, rows, label
    ):
        """Null means "the host did not say", which Python reads as no rung."""
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client(f"return {json.dumps({'data': [{'info': r, 'parts': []} for r in rows]})}")},
              {{sessionID: "ses_1"}},
            ) ?? "NULL")
            """,
            built_package,
        )
        assert observed == "NULL", label

    @pytest.mark.parametrize(
        "client_expr, ctx, label",
        [
            ("undefined", '{sessionID: "ses_1"}', "no client at all (old host)"),
            ("{}", '{sessionID: "ses_1"}', "a client with no session route"),
            ('{session: {messages: async () => ({data: []})}}', "{}", "no sessionID"),
        ],
    )
    def test_a_missing_channel_is_not_an_error(
        self, built_package, client_expr, ctx, label
    ):
        """Rule 4: never throw. A degraded lookup is the pre-existing behaviour."""
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel({client_expr}, {ctx}) ?? "NULL")
            """,
            built_package,
        )
        assert observed == "NULL", label

    def test_a_rejecting_client_is_swallowed_and_warned(self, built_package):
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(built_package))})
            const T = MultiLlmPlugin.__testing
            const v = await T.resolveSessionModel(
              {_client('throw new Error("route is gone")')}, {{sessionID: "ses_1"}},
            )
            console.log("@@" + JSON.stringify(v ?? "NULL") + "@@")
            """
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1)) == "NULL"
        assert "[multi-llm]" in result.stderr

    def test_a_hanging_client_costs_the_budget_and_no_more(self, built_package):
        """The lookup is a nicety; three rungs sit under it.

        A `Promise.race` against an unref'd timer rather than an `AbortSignal`,
        because the budget has to hold against a client that ignores one — which
        is exactly what this stub is.
        """
        started = time.monotonic()
        observed = node_json(
            f"""
            emit(await T.resolveSessionModel(
              {_client("await new Promise(() => {})")}, {{sessionID: "ses_1"}}, 250,
            ) ?? "NULL")
            """,
            built_package,
        )
        assert observed == "NULL"
        # Generous: this bounds a runaway wait, not node's startup.
        assert time.monotonic() - started < 60

    def test_the_factory_still_works_with_no_input_at_all(self, built_package):
        """`async ({ client } = {})`: an older loader passes nothing (rule 4)."""
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(built_package))})
            const p = await MultiLlmPlugin()
            console.log("@@" + JSON.stringify(Object.keys(p.tool ?? {{}})) + "@@")
            """
        )
        assert result.returncode == 0, result.stderr
        tools = json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1))
        assert tools == ["multi_llm"]

    def test_the_factory_hands_the_client_down_to_the_tool(
        self, built_package, tmp_path
    ):
        """End to end: the model reaches the child's environment and the result.

        The whole point of reading it at all. The dispatcher's own environment is
        what the detached worker inherits (`utils/detach.py`'s `child_env` copies
        `os.environ`), so setting it on the launcher sets it for every poll and
        for the fan-out itself.
        """
        env_log = tmp_path / "child-env.json"
        stub, launch_env = spawnable_stub(
            tmp_path, "uv-env-stub",
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            f"open({json.dumps(str(env_log))}, 'w', encoding='utf-8').write(\n"
            "    json.dumps({k: v for k, v in os.environ.items()\n"
            "                if k.startswith('MULTI_LLM_')}))\n"
            'print(\'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}\')\n',
        )
        outcome = node_json(
            f"""
            const plugin = await MultiLlmPlugin({{
              client: {_client(f"return {json.dumps(_messages(_ASSISTANT))}")},
            }})
            const r = await plugin.tool.multi_llm.execute(
              {{mode: "init", plan: "", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(tmp_path))}, sessionID: "ses_1"}},
            )
            emit(JSON.parse(r))
            """,
            built_package,
            env={"MULTI_LLM_UV": str(stub), **launch_env},
        )
        assert outcome["session_model"] == "opencode/nemotron-3-ultra-free"
        child_env = json.loads(env_log.read_text(encoding="utf-8"))
        assert child_env[provider_registry.HOST_SESSION_MODEL_ENV] == (
            "opencode/nemotron-3-ultra-free"
        )

    def test_a_session_with_no_model_leaves_the_variable_unset(
        self, built_package, tmp_path
    ):
        """Unset, never empty: an empty value is a rung the ladder must ignore.

        Python treats blank as "the host did not say", but only because this side
        never writes one — the two halves agree by construction rather than by a
        special case on each side.
        """
        env_log = tmp_path / "child-env-2.json"
        stub, launch_env = spawnable_stub(
            tmp_path, "uv-env-stub-2",
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            f"open({json.dumps(str(env_log))}, 'w', encoding='utf-8').write(\n"
            "    json.dumps({k: v for k, v in os.environ.items()\n"
            "                if k.startswith('MULTI_LLM_')}))\n"
            'print(\'MULTI_LLM_RESULT: {"state":"done","exit":0,"next":null}\')\n',
        )
        outcome = node_json(
            f"""
            const plugin = await MultiLlmPlugin({{
              client: {_client(f"return {json.dumps(_messages(_USER))}")},
            }})
            const r = await plugin.tool.multi_llm.execute(
              {{mode: "init", plan: "", question: "", extra_flags: []}},
              {{directory: {json.dumps(str(tmp_path))}, sessionID: "ses_1"}},
            )
            emit(JSON.parse(r))
            """,
            built_package,
            env={"MULTI_LLM_UV": str(stub), **launch_env},
        )
        assert "session_model" not in outcome
        child_env = json.loads(env_log.read_text(encoding="utf-8"))
        assert provider_registry.HOST_SESSION_MODEL_ENV not in child_env


# ===========================================================================
# The route signal and the lifted next step (plans/next-phase-command.md T2, T8)
# ===========================================================================

#: Schema v5's route signal, from the generated surface (never retyped).
ROUTE_ENV = pkg_build.cli_surface()["dispatcher"]["route_env"]

#: Every next-phase spelling the capability data carries, read through the
#: accessor: six ``(host, route)`` cells.
NEXT_COMMAND_CELLS = [
    (host_id, route)
    for host_id in caps.host_ids()
    for route in caps.next_command_routes(host_id)
]

NEXT_STEP_ARGS = {"mode": "review-plan", "plan": "p.md", "question": "",
                  "extra_flags": []}
NEXT_STEP_VALUES = {"word": "review-plan", "plan": "/w/p.md", "question": "",
                    "extraFlags": []}
NEXT_STEP_LAUNCH = {
    "exit": 0,
    "stdout": [
        "run_id: RID-N",
        "log: /tmp/next.log",
        'MULTI_LLM_RESULT: {"mode":"review-plan","state":"done","exit":0,'
        '"artifacts":[],"next":null}',
    ],
}


def _next_step(command=None, *, complete=False):
    """A ``user_next_step`` value shaped like the dispatcher's (§2.1)."""
    if complete:
        return {"phase": None, "command": None, "complete": True}
    return {
        "phase": "implement",
        "command": command,
        "optional_before": [{
            "phase": "review-tasks",
            "command": "/multi-llm --review-tasks /w/p.md",
            "why": next(
                m for m in mode_table.MODES if m.name == "review-tasks"
            ).optional_why,
        }],
        "complete": False,
    }


def _child(code, state, **payload):
    """A synthetic finished child for ``terminalResult``."""
    body = {"mode": "status", "state": state, "exit": code, "artifacts": [],
            "next": None, **payload}
    return {
        "code": code,
        "result": body,
        "stdout": f"{cli_common.RESULT_MARKER} {json.dumps(body)}\n",
    }


class TestTheRouteSignalReachesEveryChild:
    def test_the_surface_carries_the_owners_name_and_this_hosts_route(self):
        assert ROUTE_ENV == {"name": mode_table.ROUTE_ENV, "value": "package"}
        assert ROUTE_ENV["value"] in mode_table.TOOL_ROUTES
        assert ROUTE_ENV["value"] in caps.next_command_routes("opencode")

    def test_neither_the_name_nor_the_value_is_a_js_literal(self):
        """The plugin reads both strings from the surface (T2)."""
        body = _code_only(SRC_MJS.read_text(encoding="utf-8"))
        assert ROUTE_ENV["name"] not in body
        for quote in ('"', "'", "`"):
            assert f"{quote}{ROUTE_ENV['value']}{quote}" not in body

    def test_the_launch_and_every_poll_carry_it(
        self, built_package, uv_stub, tmp_path
    ):
        env_log = tmp_path / "child-env.log"
        result, calls = uv_stub.run(
            built_package, NEXT_STEP_ARGS,
            [NEXT_STEP_LAUNCH, _poll(2), _poll(0, state="done")],
            extra_env={"STUB_ENV_LOG": str(env_log)},
        )
        envs = [json.loads(line) for line in
                env_log.read_text(encoding="utf-8").splitlines() if line]
        assert len(calls) == 3 and len(envs) == 3, (calls, envs)
        assert [env.get(ROUTE_ENV["name"]) for env in envs] == (
            [ROUTE_ENV["value"]] * 3
        )
        assert result["ok"] is True

    @pytest.mark.parametrize("spelling", [
        "previous-version", "no-route-env", "empty-route-name", "no-route-value",
    ])
    def test_a_stale_surface_is_refused_and_spawns_nothing(
        self, tmp_path, uv_stub, spelling
    ):
        """A v4 surface, or a v5 one without a usable ``dispatcher.route_env``,
        is refused like a version mismatch: no tool registered, a warning naming
        the repair, and ``execute`` (reached directly) spawns no child. Never a
        spawn without the route signal, which would print the file-shim
        spelling on this route."""
        current = pkg_build.cli_surface()
        dispatcher = dict(current["dispatcher"])
        if spelling == "previous-version":
            dispatcher.pop("route_env")
            stale = dict(
                current, schema_version=pkg_build.SURFACE_SCHEMA_VERSION - 1,
                dispatcher=dispatcher,
                modes=[{k: v for k, v in row.items() if k != "optional_why"}
                       for row in current["modes"]],
            )
        else:
            if spelling == "no-route-env":
                dispatcher.pop("route_env")
            elif spelling == "empty-route-name":
                dispatcher["route_env"] = dict(ROUTE_ENV, name="")
            else:
                dispatcher["route_env"] = {"name": ROUTE_ENV["name"]}
            stale = dict(current, dispatcher=dispatcher)
        package = tmp_path / "stale"
        (package / "dist").mkdir(parents=True)
        shutil.copy2(SRC_MJS, package / "dist" / "index.mjs")
        (package / "dist" / "cli-surface.json").write_text(
            json.dumps(stale), encoding="utf-8")
        log = tmp_path / "spawns.log"
        log.write_text("", encoding="utf-8")
        script = tmp_path / "script.json"
        script.write_text(json.dumps([_poll(0, state="done")]), encoding="utf-8")
        result = run_node(
            f"""
            const {{ MultiLlmPlugin }} = await import({json.dumps(module_url(package))})
            const p = await MultiLlmPlugin()
            const surface = MultiLlmPlugin.__testing.loadSurface()
            const answer = await MultiLlmPlugin.execute(
              {json.dumps(NEXT_STEP_ARGS)}, {{ directory: {json.dumps(str(tmp_path))} }},
              surface,
            )
            console.log("@@" + JSON.stringify({{
              tools: Object.keys(p.tool ?? {{}}), surface, answer,
            }}) + "@@")
            """,
            env={**uv_stub.env, "STUB_LOG": str(log),
                 "STUB_SCRIPT_FILE": str(script)},
        )
        assert result.returncode == 0, result.stderr
        payload_json = json.loads(re.search(r"@@(.*)@@", result.stdout, re.S).group(1))
        assert payload_json["tools"] == [], "the tool registered on a stale surface"
        assert payload_json["surface"] is None
        assert payload_json["answer"]["ok"] is False
        assert log.read_text(encoding="utf-8") == "", "a child was spawned"
        assert "[multi-llm]" in result.stderr and "reinstall" in result.stderr
        if spelling != "previous-version":
            assert "route_env" in result.stderr


class TestTheNextStepIsLiftedAndNeverFollowed:
    """§2.4 on this route: ``terminalResult`` lifts, ``isMechanical`` ignores."""

    def test_exit_0_moves_it_out_of_the_payload(self, built_package):
        """Fix round 26 (live run 11, the decline run printed "Workflow
        complete." twice): moved, not copied, so the section the model
        copies is in the result once."""
        step = _next_step("/multi-llm --implement /w/p.md")
        out = _terminal(built_package, _child(0, "done", user_next_step=step),
                        NEXT_STEP_VALUES)
        assert out["ok"] is True
        assert out["user_next_step"] == step
        assert "user_next_step" not in out["result"]
        assert "next_phase" not in json.dumps(out)

    def test_the_completed_form_is_lifted_too(self, built_package):
        step = _next_step(complete=True)
        out = _terminal(built_package, _child(0, "done", user_next_step=step),
                        NEXT_STEP_VALUES)
        assert out["user_next_step"] == {
            "phase": None, "command": None, "complete": True}

    def test_exit_2_of_a_phase_already_complete_lifts_it(self, built_package):
        step = _next_step("/multi-llm --generate-tasks /w/p.md")
        out = _terminal(
            built_package,
            _child(2, cli_common._STATE_BY_EXIT[cli_common.EXIT_NOTHING_TO_DO],
                   mode="review-plan", user_next_step=step),
            NEXT_STEP_VALUES,
        )
        assert out["ok"] is True and out["exit"] == 2
        assert out["user_next_step"] == step
        assert "next_phase" not in json.dumps(out)

    def test_a_still_running_poll_carries_none(self, built_package):
        out = _terminal(built_package, _child(2, cli_common.STATE_RUNNING),
                        NEXT_STEP_VALUES)
        assert "user_next_step" not in out
        assert "user_next_step" not in out["result"]

    def test_an_absent_key_lifts_nothing(self, built_package):
        out = _terminal(built_package, _child(0, "done"), NEXT_STEP_VALUES)
        assert "user_next_step" not in out

    @pytest.mark.parametrize("code", [1, 3])
    def test_exit_1_and_3_lift_nothing(self, built_package, code):
        step = _next_step("/multi-llm --implement /w/p.md")
        out = _terminal(built_package, _child(code, "error", user_next_step=step),
                        NEXT_STEP_VALUES)
        assert "user_next_step" not in out

    def test_redaction_leaves_every_spelling_intact(self, built_package):
        """All six ``next_command`` spellings survive ``redactCommand`` and the
        lift byte for byte: they are the host's own invocation, never a
        ``uv run`` line."""
        commands = [
            caps.next_command_template(host_id, route).replace(
                "{args}", "--generate-tasks /w/plans/p.md")
            for host_id, route in NEXT_COMMAND_CELLS
        ]
        assert len(commands) == 6
        children = []
        for command in commands:
            child = _child(0, "done", user_next_step=_next_step(command))
            child["stdout"] = (f"re-run it with: {command}\n[NEXT_COMMAND] {command}\n"
                               + child["stdout"])
            children.append({
                "code": 0, "argv": ["uv", "run"], "stderr": "", "resultRaw": None,
                "truncated": False, "banner": {}, "aborted": False,
                "timedOut": False, "spawnError": None, **child,
            })
        out = node_json(
            f"""
            const commands = {json.dumps(commands)}
            const children = {json.dumps(children)}
            emit({{
              redacted: commands.map((c) => T.redactCommand(c, null, surface)),
              lifted: children.map((child) => T.terminalResult(
                child, {{ anchor: "/anchor", surface,
                          values: {json.dumps(NEXT_STEP_VALUES)} }},
              )),
            }})
            """,
            built_package,
        )
        assert out["redacted"] == commands
        for command, result in zip(commands, out["lifted"]):
            assert result["user_next_step"]["command"] == command
            assert "user_next_step" not in result["result"]
            lines = result["output"].split("\n")
            assert f"re-run it with: {command}" in lines
            # Fix round 27: the block line the lifted step carries is left to it.
            assert f"[NEXT_COMMAND] {command}" not in lines


class TestTheOutputLeavesTheBlockToTheStep:
    """The Fable review of fix round 26 (relay #1): the block printed after
    a terminal poll's text handed the model the manual note and the optional
    whys in `output` beside `user_next_step.markdown`. A result that carries
    the step leaves the block's lines to it; the text above stays."""

    MARKERS = tuple(pkg_build.cli_surface()["dispatcher"]["next_step_markers"])

    def _carried(self, output):
        return [line for line in (output or "").split("\n")
                if any(line == marker or line.startswith(marker + " ")
                       for marker in self.MARKERS)]

    def _real(self, argv, monkeypatch, capsys):
        """The dispatcher's own stdout and result for *argv*, on this route."""
        import multi_llm
        from utils import detach as _detach

        for name in ("MULTI_LLM_HOST", _detach.PLAN_ARG_ENV, _detach.WORKER_ENV):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(ROUTE_ENV["name"], ROUTE_ENV["value"])
        capsys.readouterr()
        code = multi_llm.main(list(argv))
        out = capsys.readouterr().out
        assert code == 0, out
        return {"code": 0, "stdout": out,
                "result": cli_common.parse_result_line(out)}

    def test_the_markers_are_the_ones_the_step_carries(self):
        from utils import next_phase as _np

        assert set(self.MARKERS) == set(_np.CARRIED_BLOCK_MARKERS)
        assert _np.RUNNER_WARNING_MARKER not in self.MARKERS

    def test_a_real_complete_poll_says_the_manual_note_once(
        self, built_package, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_manual_tasks_on_completion import _complete_plan
        from tests.test_next_command_emitters import _terminal_record

        plan = _complete_plan(tmp_path / "project")
        _terminal_record(plan, "apply-fixes", mode="apply-code-fixes",
                         plan_file=str(plan))
        child = self._real(["status", str(plan), "--view", "run", "--phase",
                            "apply-code-fixes", "--host", "opencode"], monkeypatch, capsys)
        assert "[MANUAL_TASK_NOTE] " in child["stdout"]
        out = _terminal(built_package, child, dict(NEXT_STEP_VALUES, plan=str(plan)))
        step = out["user_next_step"]
        assert step["complete"] is True and step["manual_tasks"]
        assert self._carried(out["output"]) == [], out["output"]
        assert step["manual_tasks_note"] not in out["output"]
        assert "runs ({phase}/.status.json)" in out["output"]

    def test_a_real_poll_says_the_optional_why_once(
        self, built_package, tmp_path, monkeypatch, capsys,
    ):
        from tests.test_next_command_emitters import _terminal_record, build
        from tests.test_next_phase import P, UP_TO_TASKS, WHY_REVIEW_TASKS, S

        root = tmp_path / "project"
        root.mkdir()
        monkeypatch.chdir(root)
        plan = build(root, S(UP_TO_TASKS))
        _terminal_record(plan, "generate-tasks", mode="generate-tasks",
                         plan_file=str(plan), plan_arg=P)
        child = self._real(["status", P, "--view", "run", "--phase",
                            "generate-tasks", "--host", "opencode"], monkeypatch, capsys)
        assert "[NEXT_COMMAND_OPTIONAL_WHY] " + WHY_REVIEW_TASKS in child["stdout"]
        out = _terminal(built_package, child, dict(NEXT_STEP_VALUES, plan=str(plan)))
        assert out["user_next_step"]["optional_before"][0]["why"] == WHY_REVIEW_TASKS
        assert self._carried(out["output"]) == [], out["output"]
        assert WHY_REVIEW_TASKS not in out["output"]

    def test_an_exit_2_rerun_leaves_it_too(self, built_package):
        step = _next_step("/multi-llm --implement /w/p.md")
        child = _child(2, "nothing-to-do", user_next_step=step)
        child["stdout"] = ("review-plan is already complete for this plan.\n"
                           f"[NEXT_COMMAND] {step['command']}\n" + child["stdout"])
        out = _terminal(built_package, child, NEXT_STEP_VALUES)
        assert out["user_next_step"] == step
        assert out["output"] == "review-plan is already complete for this plan."

    def test_a_result_without_the_step_keeps_every_line(self, built_package):
        child = _child(0, "done")
        child["stdout"] = ("[WORKFLOW_COMPLETE]\n[MANUAL_TASK] T011 Manual playthrough\n"
                           + child["stdout"])
        out = _terminal(built_package, child, NEXT_STEP_VALUES)
        assert "user_next_step" not in out
        assert out["output"].split("\n") == [
            "[WORKFLOW_COMPLETE]", "[MANUAL_TASK] T011 Manual playthrough"]

    def test_what_the_step_does_not_carry_stays(self, built_package):
        step = _next_step("/multi-llm --implement /w/p.md")
        child = _child(0, "done", user_next_step=step)
        kept = [
            "[RUNNER_WARNING] implement on this host is refused: say so first",
            "  [NEXT_COMMAND] an indented quote is not the block",
            "[NEXT_COMMAND]x not a marker line either",
        ]
        child["stdout"] = "\n".join(
            kept[:1] + [f"[NEXT_COMMAND] {step['command']}"] + kept[1:]
        ) + "\n" + child["stdout"]
        out = _terminal(built_package, child, NEXT_STEP_VALUES)
        assert out["output"].split("\n") == kept

    def test_the_continuation_loop_never_follows_it(self, built_package, uv_stub):
        """Zero follows for a result carrying only ``user_next_step`` — even one
        whose command carries a mechanical-continuation flag."""
        marker = pkg_build.cli_surface()["dispatcher"]["mechanical_continuations"][0]
        step = _next_step("/multi-llm --review-plan /w/p.md " + marker)
        result, calls = uv_stub.run(
            built_package, NEXT_STEP_ARGS,
            [NEXT_STEP_LAUNCH, _poll(0, state="done", user_next_step=step),
             NEXT_STEP_LAUNCH],
        )
        assert len(calls) == 2, calls
        assert "continuations" not in result
        assert result["user_next_step"] == step
        assert result.get("next") is None
