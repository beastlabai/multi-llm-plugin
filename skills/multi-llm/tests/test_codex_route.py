"""The codex plugin route, as the host layer and the docs see it (D4, D5, A3).

Plain CI: nothing here needs the built plugin or a real ``codex``. The resolver
rows run over fixture ``CODEX_HOME`` trees with ``PATH`` pinned — either to no
``codex`` at all (step 2, the ``config.toml`` scan) or to a fake ``codex`` that
replays ``fixtures/codex/plugin-route/plugin-list.json`` (step 1). That fixture
is the document Phase 1's container row 5 recorded on codex 0.156.1 (run
``20260923T191255Z``); see the README beside it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "hosts"))

from utils import host_capabilities as caps  # noqa: E402
from utils import host_install  # noqa: E402
import render_capabilities as renderer  # noqa: E402

PLUGIN_LIST = SKILL_ROOT / "tests" / "fixtures" / "codex" / "plugin-route" / "plugin-list.json"
PACKAGE = caps.entry("codex")["package"]
PLUGIN_ID = f"{PACKAGE['name']}@{PACKAGE['marketplace']}"
INGESTION_RULES = REPO_ROOT / "packaging" / "codex" / "ingestion_rules.py"


# ---------------------------------------------------------------------------
# Fixture homes
# ---------------------------------------------------------------------------

def _payload(codex_home: Path, marketplace: str, version: str) -> Path:
    """A version directory that answers step 3's two stats."""
    root = codex_home / "plugins" / "cache" / marketplace / PACKAGE["name"] / version
    (root / "skills" / "multi-llm").mkdir(parents=True)
    (root / "skills" / "multi-llm" / "SKILL.md").write_text("---\n---\n")
    (root / ".mcp.json").write_text("{}\n")
    return root


def _table(plugin_id: str, enabled: bool) -> str:
    return f'[plugins."{plugin_id}"]\nenabled = {"true" if enabled else "false"}\n'


def _script(path: Path, body: str) -> Path:
    """An executable Python script, shebang'd to this interpreter by absolute path
    so it runs whatever ``PATH`` the test pins."""
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


class Home:
    """One fixture machine: a ``HOME``, a ``CODEX_HOME`` and a pinned ``PATH``."""

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.codex_home = root / "codex-home"
        self.bin = root / "bin"
        for directory in (self.home, self.codex_home, self.bin):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Path:
        return self.codex_home / "config.toml"

    def env(self) -> dict:
        return {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "CODEX_HOME": str(self.codex_home),
            "PATH": str(self.bin),
        }

    def python3(self, version: Optional[str] = "3.12.1") -> None:
        """``None`` leaves ``python3`` off ``PATH``; a string fakes that version."""
        if version is not None:
            _script(self.bin / "python3", f"print('Python {version}')\n")

    def codex_replaying(self, document: object) -> None:
        payload = self.root / "plugin-list.out.json"
        payload.write_text(json.dumps(document), encoding="utf-8")
        _script(
            self.bin / "codex",
            "import sys\n"
            "assert sys.argv[1:] == ['plugin', 'list', '--json'], sys.argv\n"
            f"sys.stdout.write(open({str(payload)!r}).read())\n",
        )

    def codex_failing(self, body: str) -> None:
        _script(self.bin / "codex", body)

    def route(self):
        return host_install.package_route("codex", env=self.env())

    def note(self, *, file_present: bool = False) -> Optional[str]:
        route = self.route()
        if route is None:
            return None
        return host_install.package_route_note(
            "codex", route, file_present=file_present, env=self.env()
        )


@pytest.fixture
def machine(tmp_path):
    made = Home(tmp_path)
    made.python3()
    return made


def _recorded(**overrides) -> dict:
    """The fixture document, with the multi-llm entry's fields overridden."""
    document = json.loads(PLUGIN_LIST.read_text(encoding="utf-8"))
    for entry in document["installed"]:
        if entry["pluginId"] == PLUGIN_ID:
            for key, value in overrides.items():
                if value is _DROP:
                    entry.pop(key, None)
                else:
                    entry[key] = value
    return document


_DROP = object()


# ---------------------------------------------------------------------------
# D5: the five states, the scan grammar, the list half
# ---------------------------------------------------------------------------

class TestCheckReportsPluginRoute:
    """``test_check_reports_plugin_route``, split into the rows it names."""

    # -- step 2 alone: no `codex` on PATH ---------------------------------

    def test_enabled_from_the_table_alone_says_what_it_does_not_know(self, machine):
        machine.config.write_text(_table(PLUGIN_ID, True))
        _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        note = machine.note(file_present=True)
        assert note is not None
        assert PLUGIN_ID in note
        assert "version and install directory unknown" in note
        assert "codex plugin list --json" in note
        assert str(machine.config) in note, "the note names its evidence"
        # Activation evidence is not path evidence: the cache is not consulted.
        assert "plugins/cache" not in note.replace(os.sep, "/")

    def test_enabled_with_the_shim_present_warns_of_two_skills(self, machine):
        machine.config.write_text(_table(PLUGIN_ID, True))
        note = machine.note(file_present=True)
        assert "Two copies of this skill are advertised" in note
        assert "`multi-llm:multi-llm`" in note and "`multi-llm`" in note
        assert f"codex plugin remove {PLUGIN_ID}" in note
        assert "uninstall-host codex --global" in note

    def test_enabled_without_a_shim_warns_of_nothing(self, machine):
        machine.config.write_text(_table(PLUGIN_ID, True))
        note = machine.note(file_present=False)
        assert "Two copies" not in note

    def test_unread_activation_is_not_called_enabled(self, machine):
        # A list entry without `enabled` and a config.toml with no table:
        # nobody read the activation, so the note must not assert one.
        machine.codex_replaying(_recorded(enabled=_DROP))
        note = machine.note(file_present=True)
        assert note is not None and PLUGIN_ID in note
        assert "enabled" not in note
        assert "activation not recorded" in note
        assert "Two copies" not in note

    def test_disabled_says_so_and_warns_of_nothing(self, machine):
        machine.config.write_text(_table(PLUGIN_ID, False))
        note = machine.note(file_present=True)
        assert "disabled" in note and PLUGIN_ID in note
        assert "Two copies" not in note
        assert "python3" not in note

    def test_removed_is_silence_even_with_an_empty_cache_dir(self, machine):
        # A remove leaves `cache/<marketplace>/` behind, empty (R25).
        (machine.codex_home / "plugins" / "cache" / PACKAGE["marketplace"]).mkdir(
            parents=True
        )
        machine.config.write_text('model = "gpt-6-astra"\n')
        assert machine.route() is None

    def test_no_config_file_at_all_is_silence(self, machine):
        assert machine.route() is None

    def test_two_marketplaces_name_every_plugin_id(self, machine):
        other = f"{PACKAGE['name']}@a-fork"
        machine.config.write_text(_table(PLUGIN_ID, True) + _table(other, True))
        note = machine.note(file_present=True)
        assert PLUGIN_ID in note and other in note
        assert "codex plugin remove" in note

    def test_step_two_never_nominates_a_sibling_directory(self, machine):
        machine.config.write_text(_table(PLUGIN_ID, True))
        older = _payload(machine.codex_home, PACKAGE["marketplace"], "0.0.9")
        newer = _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        note = machine.note(file_present=True)
        assert "unknown" in note
        assert str(older) not in note and str(newer) not in note

    def test_keyed_on_codex_home_never_home(self, machine):
        # A table under $HOME/.codex is not codex's when CODEX_HOME points elsewhere.
        (machine.home / ".codex").mkdir()
        (machine.home / ".codex" / "config.toml").write_text(_table(PLUGIN_ID, True))
        assert machine.route() is None

    def test_codex_home_falls_back_to_dot_codex(self, machine):
        (machine.home / ".codex").mkdir()
        (machine.home / ".codex" / "config.toml").write_text(_table(PLUGIN_ID, True))
        env = machine.env()
        env.pop("CODEX_HOME")
        route = host_install.package_route("codex", env=env)
        assert route is not None and route.spec == PLUGIN_ID

    # -- python3 -------------------------------------------------------------

    def test_python3_missing_is_a_second_line(self, tmp_path):
        bare = Home(tmp_path)
        bare.python3(None)
        bare.config.write_text(_table(PLUGIN_ID, True))
        note = bare.note(file_present=True)
        first, second = note.split("\n")
        assert "Two copies" in first
        assert "python3" in second and "cannot start" in second
        floor = ".".join(map(str, host_install.CODEX_SERVER_MIN_PYTHON))
        assert f">= {floor}" in second

    def test_the_activated_venv_is_not_the_users_python3(self, tmp_path):
        # `uv run` puts the skill venv's bin first on PATH and sets VIRTUAL_ENV;
        # codex launches the server from the user's PATH, which has no such
        # entry. Measured red in-container (Phase 1 row 9(c), 2026-09-24).
        bare = Home(tmp_path)
        bare.python3(None)
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        _script(venv / "bin" / "python3", "print('Python 3.12.1')\n")
        env = {**bare.env(), "VIRTUAL_ENV": str(venv),
               "PATH": os.pathsep.join([str(venv / "bin"), str(bare.bin)])}
        assert host_install._python3_problem(env) is not None
        assert "python3" in host_install._python3_problem(env)
        # Without the activation marker the same PATH entry is the user's own.
        env.pop("VIRTUAL_ENV")
        assert host_install._python3_problem(env) is None

    def test_python3_below_the_floor_is_named(self, tmp_path):
        old = Home(tmp_path)
        old.python3("3.8.10")
        old.config.write_text(_table(PLUGIN_ID, True))
        note = old.note(file_present=False)
        assert "3.8" in note.split("\n")[1] and "below" in note

    def test_python3_at_the_floor_adds_nothing(self, tmp_path):
        ok = Home(tmp_path)
        ok.python3("3.9.0")
        ok.config.write_text(_table(PLUGIN_ID, True))
        assert "\n" not in ok.note(file_present=True)

    # -- the check itself ----------------------------------------------------

    @pytest.fixture
    def quiet_environment_row(self, monkeypatch):
        # Row 5 probes the runner CLI on the real machine; it is route-independent
        # and not what these rows are about.
        monkeypatch.setattr(
            host_install, "_environment_row",
            lambda *a, **k: host_install.Row(host_install.ROW_ENVIRONMENT, True, "stub"),
        )

    def test_the_note_never_moves_the_check(self, machine, quiet_environment_row):
        before = host_install.check("codex", env=machine.env())
        machine.config.write_text(_table(PLUGIN_ID, True))
        after = host_install.check("codex", env=machine.env())
        assert [(r.number, r.ok) for r in before.rows] == [
            (r.number, r.ok) for r in after.rows
        ]
        assert before.ok == after.ok
        assert any(PLUGIN_ID in note for note in after.notes)
        assert not any(PLUGIN_ID in note for note in before.notes)
        # On this route row 1 is NOT answered by the plugin: the file route's
        # repair stays in place, since both skills are advertised side by side.
        assert not after.row(host_install.ROW_PRESENT).ok

    def test_the_note_is_printed_on_the_install_path(
        self, machine, quiet_environment_row, capsys
    ):
        machine.config.write_text(_table(PLUGIN_ID, True))
        host_install.install("codex", scope="global", env=machine.env())
        out = capsys.readouterr().out
        assert "Two copies of this skill are advertised" in out
        result = host_install.check("codex", env=machine.env())
        assert result.row(host_install.ROW_PRESENT).ok

    # -- step 1: the list half ----------------------------------------------

    def test_the_recorded_list_resolves_the_computed_path(self, machine):
        machine.codex_replaying(_recorded())
        payload = _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        route = machine.route()
        assert route.spec == PLUGIN_ID and route.version == "0.1.0"
        assert route.enabled is True and route.payload == payload
        note = machine.note(file_present=True)
        assert str(payload) in note and "0.1.0" in note
        assert "codex plugin list --json" in note
        assert "Two copies" in note

    def test_a_dropped_version_is_unknown_and_names_no_sibling(self, machine):
        machine.codex_replaying(_recorded(version=_DROP))
        older = _payload(machine.codex_home, PACKAGE["marketplace"], "0.0.9")
        newer = _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        note = machine.note(file_present=True)
        assert "unknown" in note
        assert str(older) not in note and str(newer) not in note

    def test_a_path_that_does_not_answer_is_unknown(self, machine):
        machine.codex_replaying(_recorded(version="0.2.0"))
        _payload(machine.codex_home, PACKAGE["marketplace"], "0.0.9")
        _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        # An empty husk at the named version is not a payload either.
        (
            machine.codex_home / "plugins" / "cache" / PACKAGE["marketplace"]
            / PACKAGE["name"] / "0.2.0"
        ).mkdir()
        note = machine.note(file_present=True)
        assert "unknown" in note and "0.0.9" not in note and "0.1.0" not in note

    def test_a_sibling_version_is_named_only_as_a_leftover(self, machine):
        machine.codex_replaying(_recorded())
        older = _payload(machine.codex_home, PACKAGE["marketplace"], "0.0.9")
        current = _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        note = machine.note(file_present=False)
        assert f"at {current}" in note
        assert f"{older} is also on disk" in note and "leftover" in note

    def test_a_second_marketplace_entry_fires_the_several_row(self, machine):
        document = _recorded()
        extra = dict(
            next(e for e in document["installed"] if e["pluginId"] == PLUGIN_ID)
        )
        extra.update(pluginId=f"{PACKAGE['name']}@a-fork", marketplaceName="a-fork")
        document["installed"].append(extra)
        machine.codex_replaying(document)
        note = machine.note(file_present=True)
        assert PLUGIN_ID in note and f"{PACKAGE['name']}@a-fork" in note

    def test_a_disabled_list_entry_is_disabled(self, machine):
        machine.codex_replaying(_recorded(enabled=False))
        assert "disabled" in machine.note(file_present=True)

    def test_the_recorded_shape_carries_no_installed_path(self):
        """Row 5's document: both source objects, no ``installedPath``."""
        document = _recorded()
        assert document["available"] == []
        entry = next(e for e in document["installed"] if e["pluginId"] == PLUGIN_ID)
        assert "installedPath" not in entry
        assert entry["source"]["source"] == "local" and "path" in entry["source"]
        assert set(entry["marketplaceSource"]) == {"sourceType", "source"}

    def test_an_available_row_is_not_an_install(self, machine):
        document = _recorded()
        entry = next(e for e in document["installed"] if e["pluginId"] == PLUGIN_ID)
        document["installed"] = [
            e for e in document["installed"] if e["pluginId"] != PLUGIN_ID
        ]
        # The recorded `available` is empty; a catalog row is added here.
        document["available"].append(dict(entry, installed=False, enabled=False))
        machine.codex_replaying(document)
        # The table says enabled, but the host answered: nothing installed.
        machine.config.write_text(_table(PLUGIN_ID, True))
        assert machine.route() is None

    def test_no_row_reads_installed_path_off_a_list(self, machine):
        machine.codex_replaying(_recorded(installedPath="/somewhere/else"))
        payload = _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        assert machine.route().payload == payload

    @pytest.mark.parametrize("body", [
        "import sys\nsys.exit(2)\n",
        "print('this is not json')\n",
        "import time\ntime.sleep(30)\n",
        "print('{\"unexpected\": 1}')\n",
    ], ids=["non-zero", "unparsable", "timeout", "unknown-shape"])
    def test_a_list_that_does_not_answer_falls_through_to_the_table(
        self, machine, monkeypatch, body
    ):
        monkeypatch.setattr(host_install, "CODEX_LIST_TIMEOUT_S", 1)
        machine.codex_failing(body)
        machine.config.write_text(_table(PLUGIN_ID, True))
        _payload(machine.codex_home, PACKAGE["marketplace"], "0.1.0")
        note = machine.note(file_present=True)
        assert "version and install directory unknown" in note
        assert str(machine.config) in note


#: D5 step 2's grammar, as ``config.toml`` bodies and what the scan must return:
#: ``None`` abstains, ``[]`` is "no table", otherwise the (pluginId, enabled) pairs.
SCAN_GRAMMAR = [
    ("commented-out table",
     f'# [plugins."{PLUGIN_ID}"]\n# enabled = true\n', []),
    ("no spaces around =",
     f'[plugins."{PLUGIN_ID}"]\nenabled=true\n', [(PLUGIN_ID, True)]),
    ("tabs around =",
     f'[plugins."{PLUGIN_ID}"]\nenabled\t=\tfalse\n', [(PLUGIN_ID, False)]),
    ("trailing comment",
     f'[plugins."{PLUGIN_ID}"]\nenabled = false  # was true\n', [(PLUGIN_ID, False)]),
    ("hash inside a string is not a comment",
     f'note = "a # b"\n[plugins."{PLUGIN_ID}"]\nenabled = true\n', [(PLUGIN_ID, True)]),
    ("a later header ends the block",
     f'[plugins."{PLUGIN_ID}"]\nenabled = true\n[other]\nenabled = maybe\n',
     [(PLUGIN_ID, True)]),
    ("Enabled is a different key",
     f'[plugins."{PLUGIN_ID}"]\nEnabled = false\nenabled = true\n', [(PLUGIN_ID, True)]),
    ("different case is a different plugin",
     '[plugins."Multi-LLM@beastlabai"]\nenabled = true\n', []),
    ("other plugins only",
     '[plugins."sites@openai-curated-remote"]\nenabled = true\n', []),
    ("enabled commented out", f'[plugins."{PLUGIN_ID}"]\n# enabled = true\n', None),
    ("no enabled in the block", f'[plugins."{PLUGIN_ID}"]\n', None),
    ("duplicate enabled",
     f'[plugins."{PLUGIN_ID}"]\nenabled = true\nenabled = false\n', None),
    ("non-boolean value", f'[plugins."{PLUGIN_ID}"]\nenabled = "true"\n', None),
    ("capitalised boolean", f'[plugins."{PLUGIN_ID}"]\nenabled = True\n', None),
    ("single-quoted header", f"[plugins.'{PLUGIN_ID}']\nenabled = true\n", None),
    ("padded header", f'[ plugins."{PLUGIN_ID}" ]\nenabled = true\n', None),
    ("inline entry under [plugins]",
     f'[plugins]\n"{PLUGIN_ID}" = {{ enabled = true }}\n', None),
    ("dotted entry under [plugins]",
     f'[plugins]\n"{PLUGIN_ID}".enabled = true\n', None),
    ("top-level plugins assignment",
     f'plugins = {{ "{PLUGIN_ID}" = {{ enabled = true }} }}\n', None),
    ("a duplicate table", _table(PLUGIN_ID, True) * 2, None),
    ("another plugin whose name ends in ours",
     '[plugins]\n"x-multi-llm@m".enabled = true\n' + _table(PLUGIN_ID, True),
     [(PLUGIN_ID, True)]),
    ("our id named outside the plugins tables",
     '[mcp_servers.x]\nargs = ["' + PLUGIN_ID + '"]\n' + _table(PLUGIN_ID, True),
     [(PLUGIN_ID, True)]),
    ("a UTF-8 BOM before the first header",
     "\ufeff" + _table(PLUGIN_ID, True), [(PLUGIN_ID, True)]),
    ("a top-level dotted entry", f'plugins."{PLUGIN_ID}".enabled = true\n', None),
    ("two marketplaces",
     _table(PLUGIN_ID, True) + _table("multi-llm@fork", False),
     [(PLUGIN_ID, True), ("multi-llm@fork", False)]),
]


class TestScanGrammar:
    @pytest.mark.parametrize(
        "body,expected", [(b, e) for _, b, e in SCAN_GRAMMAR],
        ids=[name for name, _, _ in SCAN_GRAMMAR],
    )
    def test_the_scan_grammar(self, tmp_path, body, expected):
        config = tmp_path / "config.toml"
        config.write_text(body, encoding="utf-8")
        assert host_install._scan_codex_config(config, PACKAGE["name"]) == expected

    def test_an_abstaining_file_emits_no_note(self, machine):
        machine.config.write_text(f'[plugins."{PLUGIN_ID}"]\nenabled = "yes"\n')
        assert machine.route() is None

    def test_undecodable_bytes_abstain(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_bytes(b"\xff\xfe[plugins]\n")
        assert host_install._scan_codex_config(config, PACKAGE["name"]) is None


# ---------------------------------------------------------------------------
# D4: the generated pages
# ---------------------------------------------------------------------------

def _codex_block(page: str) -> str:
    start = page.index("**`codex` — install via codex-marketplace")
    end = page.index("**`opencode`**", start)
    return page[start:end]


def _fences(text: str) -> List[str]:
    return re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S)


class TestGeneratedPagesSplitTheTwoRoutes:
    """``test_generated_pages_split_the_two_routes``, on a fresh render."""

    @pytest.fixture(scope="class")
    def block(self) -> str:
        return _codex_block(renderer.render_host_doc_section())

    def test_install_is_two_commands_in_order(self, block):
        install = _fences(block)[0].splitlines()
        sparse = " ".join(f"--sparse {path}" for path in PACKAGE["sparse"])
        assert install == [
            f"codex plugin marketplace add {PACKAGE['repo']} --ref "
            f"{PACKAGE['ref']} {sparse}",
            f"codex plugin add {PLUGIN_ID}",
        ]
        assert "--ref release" in install[0], "the default branch is not a release"

    def test_upgrade_is_its_own_sequence(self, block):
        start = block.index("Upgrading")
        upgrade = block[start:]
        assert upgrade.index("marketplace upgrade") < upgrade.index("plugin add")
        assert "new" in upgrade.lower() and "session" in upgrade
        assert "marketplace add" not in upgrade
        assert _fences(upgrade)[0].splitlines() == [
            "codex plugin marketplace upgrade", f"codex plugin add {PLUGIN_ID}",
        ]

    def test_no_npm_word_and_no_config_file_fence(self, block):
        for word in ("npm", "tarball", ".tgz", "config entry", "opencode.json"):
            assert word not in block, word
        assert "```json" not in block

    def test_the_two_names_are_not_swapped(self, block):
        assert "multi-llm:multi-llm" in block
        assert "Use the multi-llm skill" not in block
        page = renderer.render_host_doc_section()
        file_route = page[page.index("**`codex`**"):page.index("**`codex` — install via")]
        assert "multi-llm:multi-llm" not in file_route

    def test_the_block_reads_its_provenance(self, block):
        if caps.entry("codex")["provenance"]["package"] == "verified":
            assert "recommended route" in block
        else:
            assert "not yet published" in block
            assert "install-host codex" in block
            assert "fails **loudly**" in block

    def test_the_write_posture_is_met_before_the_install(self, block):
        assert "MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1" in block
        assert block.index("UNSANDBOXED_WRITES") < block.index("```bash")

    def test_the_npm_row_carries_no_upgrade_of_its_own(self):
        # So opencode's `pinned-spec` sentence is the one it keeps rendering.
        assert "upgrade" not in renderer._REGISTRY_FORMS["npm"]
        assert "Upgrading is bumping the pin" in renderer.render_host_doc_section()

    def test_the_reinstall_sentence_does_not_say_re_run_the_install(self):
        assert "re-running the install" not in renderer._UPDATE_FORMS["reinstall"]


# ---------------------------------------------------------------------------
# A3: the README, docs/hosts.md and the plugin's own page
# ---------------------------------------------------------------------------

def _readme_codex_section() -> str:
    """The README's short install block: commands, floors, the one warning."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # `### 1.2 Codex`: the README numbers its headings, so match the title.
    start = re.search(r"^### (?:[\d.]+\s+)?Codex$", readme, re.M).start()
    end = readme.index("\n### ", start + 1)
    return readme[start:end]


def _hosts_codex_section() -> str:
    """The hand-written details the README links to, below the generated block."""
    page = (REPO_ROOT / "docs" / "hosts.md").read_text(encoding="utf-8")
    start = page.index("## Codex plugin route\n")
    end = page.index("\n## ", start + 1)
    return page[start:end]


def _plugin_page() -> str:
    return (REPO_ROOT / "packaging" / "codex" / "README.md").read_text(encoding="utf-8")


def _codex_docs() -> dict:
    return {
        "README.md (codex section)": _readme_codex_section(),
        "docs/hosts.md (codex plugin route)": _hosts_codex_section(),
        "packaging/codex/README.md": _plugin_page(),
        "docs/hosts.md (codex block)": _codex_block(
            (REPO_ROOT / "docs" / "hosts.md").read_text(encoding="utf-8")
        ),
    }


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]


_EDIT_VERB = re.compile(r"\b(edit|open|modify|change)\b", re.I)
_PACKAGED_PLACES = ("plugins/cache", "$CODEX_HOME", "skill directory", "installed inside")


def _floors():
    """(codex floor, python floor) — from ingestion_rules when it exists."""
    if INGESTION_RULES.is_file():
        spec = importlib.util.spec_from_file_location(
            "_codex_ingestion_rules", INGESTION_RULES
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SOURCE_CODEX_VERSION, tuple(module.MIN_PYTHON)
    return PACKAGE["min_host_version"], host_install.CODEX_SERVER_MIN_PYTHON


class TestDocs:
    def test_docs_never_point_at_the_packaged_config(self):
        # Positive half first, so the row cannot pass by saying nothing.
        for name, text in _codex_docs().items():
            assert "--init" in text, name
            assert "<git-root>/.multi-llm/providers.yaml" in text, name
        # Then the negatives.
        for name, text in _codex_docs().items():
            for sentence in _sentences(text):
                if "providers.yaml" in sentence and _EDIT_VERB.search(sentence):
                    assert not any(place in sentence for place in _PACKAGED_PLACES), (
                        f"{name} tells a reader to edit the packaged config: "
                        f"{sentence!r}"
                    )
                if ".mcp.json" in sentence and _EDIT_VERB.search(sentence):
                    assert re.search(r"\b(not|never)\b", sentence, re.I), (
                        f"{name} tells a reader to edit the installed .mcp.json: "
                        f"{sentence!r}"
                    )
                assert "landlock" not in sentence.lower(), (
                    f"{name} suggests the landlock flag: {sentence!r}"
                )

    def test_docs_declare_platform_support(self):
        section = _hosts_codex_section()
        rows = {
            line.split("|")[1].strip(): line
            for line in section.splitlines()
            if re.match(r"\|\s*(Linux|macOS|Windows)\s*\|", line)
        }
        assert set(rows) == {"Linux", "macOS", "Windows"}
        assert "`verified`" in rows["Linux"]
        assert "`unmeasured`" in rows["macOS"]
        assert "zprofile" in rows["macOS"], "the zsh-profile caveat"
        assert "unsupported for the tool route" in rows["Windows"]
        assert "Git Bash" in rows["Windows"] and "hard prerequisite" in rows["Windows"]
        assert "refuses" in rows["Windows"], "the refusal is named"
        assert "`verified`" not in rows["macOS"] + rows["Windows"]
        for name, text in _codex_docs().items():
            lowered = text.lower()
            for claim in ("cross-platform", "any platform", "all platforms",
                          "works on windows", "works on macos"):
                assert claim not in lowered, f"{name}: {claim!r}"

    def test_docs_declare_the_version_floors(self):
        codex_floor, python_floor = _floors()
        python = ".".join(str(part) for part in python_floor)
        for name, text in (
            ("README.md", _readme_codex_section()),
            ("docs/hosts.md", _hosts_codex_section()),
            ("packaging/codex/README.md", _plugin_page()),
        ):
            assert f"codex >= {codex_floor}" in text, name
            assert f"python3 >= {python}" in text, name
            assert re.search(r"\buv\b", text), name
        assert f"Needs codex {PACKAGE['min_host_version']} or newer" in (
            renderer.render_host_doc_section()
        )
        # The host layer's copy of the python floor agrees with its owner.
        assert host_install.CODEX_SERVER_MIN_PYTHON == python_floor
        assert PACKAGE["min_host_version"] == codex_floor

    def test_the_docs_publish_the_disclaimer_and_the_file_route(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        section = _hosts_codex_section()
        assert "--ref release" in _readme_codex_section()
        assert "MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1" in _readme_codex_section()
        assert "docs/hosts.md#codex-plugin-route" in _readme_codex_section()
        if caps.entry("codex")["provenance"]["package"] != "verified":
            for text in (readme, section):
                assert "not yet published" in text
                assert "install-host codex" in text
            assert "--ref release" in section
        assert "codex plugin marketplace upgrade" in section
        assert "uninstall-host codex --global" in section
        assert "codex plugin list" in section
        assert "MULTI_LLM_CODEX_UNSANDBOXED_WRITES=1" in section


def test_plugin_skill_description_budget_is_declared():
    """D3e's constant lives beside the renderer, for TestPluginSkill to import."""
    assert host_install.PLUGIN_SKILL_DESCRIPTION_MAX == 400
