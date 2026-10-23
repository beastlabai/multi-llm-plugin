"""The codex package route's container plumbing, offline (plans/codex-native-plugin.md
Phase 1 items 1, 1a, 5–16).

The container rows themselves are `container`-marked and need the plugin tree,
the image and Docker. What they STAND ON does not, and is pinned here in plain
CI:

* ``entrypoint._seed_codex_plugin`` — the three commands in their fixed order,
  ``plugin-install.json``'s fields, the list document committed verbatim, the
  quiet-config assertion, and the refusal when the install left nothing —
  driven against a FAKE ``codex`` that behaves like the measured one (R15/R40);
* ``entrypoint.write_profile_credential`` — ``export``, 0600, the value never
  in the bundle, and D3a's startup-file precedence assertion failing the row by
  name;
* the matrix: the probe rows live OUTSIDE ``rows:`` (which is the credentialed
  schedule), every probe is one the entrypoint runs, every cell is
  ``unmeasured``, and ``codex-package-ask`` mirrors ``opencode-package-ask``
  argument for argument;
* the harness forwards ``--probe`` only on a row that names one, so no existing
  row's argv moves.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from pathlib import Path

import pytest
import yaml

from .container import entrypoint, harness

TESTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TESTS_DIR.parent
MATRIX = yaml.safe_load((harness.HERE / "host_matrix.yaml").read_text(encoding="utf-8"))
CODEX = MATRIX["hosts"]["codex"]
OPENCODE = MATRIX["hosts"]["opencode"]

#: A fake `codex` with the measured behaviour of the three commands the seed
#: runs: `marketplace add` succeeds, `plugin add --json` creates the cache
#: directory, appends the `[plugins."…"]` table and prints `installedPath`
#: (R40), and `plugin list --json` prints a list entry WITHOUT `installedPath`
#: (R25). `FAKE_CODEX_FAIL_ADD=1` makes `add` fail and install nothing.
FAKE_CODEX = r'''#!/usr/bin/env python3
import json, os, sys
home = os.environ["CODEX_HOME"]
args = sys.argv[1:]
with open(os.path.join(os.environ["FAKE_CODEX_LOG"]), "a") as log:
    log.write(json.dumps(args) + "\n")
if args[:3] == ["plugin", "marketplace", "add"]:
    print("Added marketplace beastlabai")
    sys.exit(0)
if args[:2] == ["plugin", "add"]:
    if os.environ.get("FAKE_CODEX_FAIL_ADD") == "1":
        print("plugin `multi-llm` was not found in marketplace `beastlabai`",
              file=sys.stderr)
        sys.exit(1)
    root = os.path.join(home, "plugins", "cache", "beastlabai", "multi-llm", "0.1.0")
    os.makedirs(os.path.join(root, "skills", "multi-llm"), exist_ok=True)
    with open(os.path.join(home, "config.toml"), "a") as config:
        config.write('\n[plugins."multi-llm@beastlabai"]\nenabled = true\n')
    print(json.dumps({"pluginId": "multi-llm@beastlabai", "installedPath": root}))
    sys.exit(0)
if args[:2] == ["plugin", "list"]:
    print(json.dumps([{"pluginId": "multi-llm@beastlabai", "name": "multi-llm",
                       "marketplaceName": "beastlabai", "version": "0.1.0",
                       "installed": True, "enabled": True}]))
    sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    fake = real / "codex"
    fake.write_text(FAKE_CODEX, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "out"
    out.mkdir()
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text(
        'model_provider = "stub"\napproval_policy = "never"\n'
        '[model_providers.stub]\nname = "stub"\n', encoding="utf-8")
    log = tmp_path / "codex-argv.jsonl"
    monkeypatch.setattr(entrypoint, "REAL_DIR", real)
    monkeypatch.setattr(entrypoint, "OUT", out)
    monkeypatch.setattr(entrypoint, "WORKSPACE", tmp_path)
    monkeypatch.setattr(entrypoint, "CODEX_MARKETPLACE_DIR", tmp_path / "mirror")
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.delenv("FAKE_CODEX_FAIL_ADD", raising=False)
    return home, out, log


class TestTheSeed:
    def test_a_file_route_row_installs_nothing(self, rig):
        home, out, log = rig
        assert entrypoint._seed_codex_plugin(home, "file") is None
        assert not log.exists()
        assert not (out / "plugin-install.json").exists()

    def test_three_commands_in_the_plans_order(self, rig):
        home, _out, log = rig
        entrypoint._seed_codex_plugin(home, "package")
        argvs = [json.loads(line) for line in log.read_text().splitlines()]
        assert argvs == [
            ["plugin", "marketplace", "add", str(entrypoint.CODEX_MARKETPLACE_DIR)],
            ["plugin", "add", "multi-llm@beastlabai", "--json"],
            ["plugin", "list", "--json"],
        ]

    def test_the_record_carries_what_the_graders_read(self, rig):
        home, out, _log = rig
        entrypoint._seed_codex_plugin(home, "package")
        record = json.loads((out / "plugin-install.json").read_text())
        assert [c["exit"] for c in record["commands"]] == [0, 0, 0]
        assert record["installedPath"].endswith(
            "plugins/cache/beastlabai/multi-llm/0.1.0")
        assert record["installed_root_exists"] is True
        assert record["config_toml"]["plugin_table"].startswith(
            '[plugins."multi-llm@beastlabai"]')
        assert "enabled = true" in record["config_toml"]["plugin_table"]
        assert not any(t.startswith("[plugins")
                       for t in record["config_toml"]["tables_before"])
        assert record["file_shim_present"] is False
        assert record["divergence"] is None
        # R25/R40: `installedPath` belongs to `add`, and the record says
        # whether the LIST entry carried it too rather than assuming.
        assert record["list_entry_has_installedPath"] is False
        assert all("_stdout_full" not in c for c in record["commands"])

    def test_the_list_document_is_committed_verbatim(self, rig):
        home, out, log = rig
        entrypoint._seed_codex_plugin(home, "package")
        listing = (out / "codex-plugin-list.json").read_text()
        assert json.loads(listing)[0]["pluginId"] == "multi-llm@beastlabai"
        assert listing.endswith("\n"), "verbatim: codex's own trailing newline"

    def test_a_loud_config_is_refused_before_the_install(self, rig):
        home, _out, log = rig
        config = home / ".codex" / "config.toml"
        config.write_text(config.read_text()
                          + '[plugins."multi-llm@beastlabai"]\nenabled = true\n')
        with pytest.raises(SystemExit, match="quiet"):
            entrypoint._seed_codex_plugin(home, "package")
        assert not log.exists(), "nothing may run once the config is loud"

    def test_a_commented_table_is_still_quiet(self, rig):
        home, _out, _log = rig
        config = home / ".codex" / "config.toml"
        config.write_text(config.read_text()
                          + '# [plugins."multi-llm@beastlabai"]\n')
        entrypoint._seed_codex_plugin(home, "package")

    def test_an_install_that_left_nothing_stops_the_row(self, rig, monkeypatch):
        home, out, _log = rig
        monkeypatch.setenv("FAKE_CODEX_FAIL_ADD", "1")
        with pytest.raises(SystemExit, match="left no installed directory"):
            entrypoint._seed_codex_plugin(home, "package")
        record = json.loads((out / "plugin-install.json").read_text())
        assert record["commands"][1]["exit"] == 1, (
            "the record is written BEFORE the refusal, so the failed install "
            "is evidence rather than a missing file")

    def test_a_file_shim_beside_the_plugin_is_recorded(self, rig):
        home, out, _log = rig
        shim = home / entrypoint.CODEX_FILE_SHIM
        shim.parent.mkdir(parents=True)
        shim.write_text("---\nname: multi-llm\n---\n")
        entrypoint._seed_codex_plugin(home, "package")
        record = json.loads((out / "plugin-install.json").read_text())
        assert record["file_shim_present"] is True


class TestTheProfileExport:
    KEY = "sk-or-v1-not-a-real-key'with a quote"

    def test_it_exports_and_never_records_the_value(self, rig):
        home, out, _log = rig
        profile = entrypoint.write_profile_credential(home, self.KEY)
        text = profile.read_text()
        assert text.startswith("export OPENROUTER_API_KEY=")
        assert stat.S_IMODE(profile.stat().st_mode) == 0o600
        record = (out / "profile-export.json").read_text()
        assert self.KEY not in record
        assert json.loads(record)["form"] == "export"

    def test_the_export_survives_a_shell_round_trip(self, rig):
        import subprocess

        home, _out, _log = rig
        entrypoint.write_profile_credential(home, self.KEY)
        seen = subprocess.run(
            ["sh", "-c", '. "$1"; sh -c \'printf %s "$OPENROUTER_API_KEY"\'',
             "sh", str(home / ".profile")],
            capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
        assert seen.stdout == self.KEY, (
            "a grandchild spawned WITHOUT a shell must inherit it — `export`, "
            "not a bare assignment (D3a)")

    def test_an_existing_profile_is_appended_to(self, rig):
        home, _out, _log = rig
        (home / ".profile").write_text("umask 022")
        entrypoint.write_profile_credential(home, "k")
        assert (home / ".profile").read_text().splitlines() == [
            "umask 022", "export OPENROUTER_API_KEY=k"]

    @pytest.mark.parametrize("shadow", [".bash_profile", ".bash_login"])
    def test_a_higher_precedence_startup_file_fails_the_row_by_name(
            self, rig, shadow):
        home, _out, _log = rig
        (home / shadow).write_text("# a seed step dropped this\n")
        with pytest.raises(SystemExit, match=shadow.replace(".", r"\.")):
            entrypoint.write_profile_credential(home, "k")

    def test_the_environment_markers_keep_bashrc_out_of_the_profile(self, rig):
        home, _out, _log = rig
        launch = entrypoint.plant_environment_markers(home)
        assert entrypoint.PROBE_LAUNCH_MARKER in launch
        profile = (home / ".profile").read_text()
        assert f"export {entrypoint.PROBE_PROFILE_CREDENTIAL}=" in profile
        assert entrypoint.PROBE_BASHRC_MARKER not in profile
        assert entrypoint.PROBE_BASHRC_MARKER in (home / ".bashrc").read_text()


class TestThePromptInputParse:
    def test_a_root_relative_locator_resolves_through_the_table(self, tmp_path):
        skill = tmp_path / "cache" / "multi-llm" / "0.1.0" / "skills" / "multi-llm"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("x")
        text = (
            "### Skill roots\n"
            f"- r3: {tmp_path / 'cache'}\n"
            "\n## Skills\n"
            "- multi-llm:multi-llm: Orchestrate reviews "
            "(file: r3/multi-llm/0.1.0/skills/multi-llm/SKILL.md)\n")
        parsed = entrypoint._parse_prompt_input(text)
        assert parsed["advertised"] is True
        assert parsed["resolved_exists"] is True

    def test_an_absent_skill_is_not_advertised(self):
        parsed = entrypoint._parse_prompt_input("### Skill roots\n- r1: /x\n")
        assert parsed["advertised"] is False
        assert parsed["resolved"] is None


def _mirror(root: Path) -> Path:
    """A miniature of Dockerfile §10c's mirror: the catalog and a plugin tree
    with a manifest, an ordinary file and an executable one."""
    (root / ".agents" / "plugins").mkdir(parents=True, exist_ok=True)
    (root / ".agents" / "plugins" / "marketplace.json").write_text(
        json.dumps({"name": "beastlabai", "plugins": [{
            "name": "multi-llm",
            "source": {"source": "local", "path": "./packaging/codex/multi-llm"}}]}),
        encoding="utf-8")
    plugin = root / "packaging" / "codex" / "multi-llm"
    (plugin / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (plugin / ".codex-plugin" / "plugin.json").write_text(
        json.dumps({"name": "multi-llm", "version": "0.1.0"}), encoding="utf-8")
    (plugin / "mcp").mkdir(exist_ok=True)
    server = plugin / "mcp" / "server.py"
    server.write_text("print('server')\n", encoding="utf-8")
    server.chmod(0o755)
    (plugin / "LICENSE").write_text("MIT\n", encoding="utf-8")
    return plugin


class TestTheGitFixture:
    """A4 row 6b's offline Git marketplace, with real git and no codex."""

    @pytest.fixture
    def fixture(self, rig):
        home, _out, _log = rig
        _mirror(entrypoint.CODEX_MARKETPLACE_DIR)
        return home, entrypoint.build_git_marketplace(home)

    def test_release_is_held_one_release_behind_the_default_branch(self, fixture):
        home, fx = fixture
        env = entrypoint._git_fixture_env(home)
        bare = Path(fx["bare"])
        assert fx["release_sha"] != fx["default_sha"]
        assert entrypoint._git(["rev-parse", "refs/heads/release"], bare, env)[
            "stdout"].strip() == fx["release_sha"]
        assert entrypoint._git(["rev-parse", "refs/heads/master"], bare, env)[
            "stdout"].strip() == fx["default_sha"]
        assert (fx["release_version"], fx["next_version"]) == ("0.1.0", "0.1.1")
        assert entrypoint._git(["merge-base", "--is-ancestor", fx["release_sha"],
                                fx["default_sha"]], bare, env, check=False)["exit"] == 0
        assert fx["url"] == f"ssh://localhost{bare}"
        assert fx["env"]["GIT_SSH_COMMAND"].startswith("sh "), (
            "/home/e2e is noexec: the shim runs through sh, never as a program")

    def test_the_ssh_url_resolves_offline_through_the_shim(self, fixture):
        """What codex's own clone does first, done by plain git."""
        home, fx = fixture
        env = {**entrypoint._git_fixture_env(home), **fx["env"]}
        listed = entrypoint._git(["ls-remote", fx["url"]], home, env)
        assert f"{fx['release_sha']}\trefs/heads/release" in listed["stdout"]

    def test_the_decoys_sit_outside_both_sparse_paths(self, fixture):
        _home, fx = fixture
        for decoy in fx["decoys"]:
            assert not decoy.startswith(tuple(f"{p}/" for p in fx["sparse"])), decoy

    def test_the_tree_comparison_reads_blobs_not_names(self, fixture, tmp_path):
        home, fx = fixture
        env = entrypoint._git_fixture_env(home)
        bare = Path(fx["bare"])
        install = tmp_path / "install"
        shutil.copytree(entrypoint.CODEX_MARKETPLACE_DIR / entrypoint.GIT_PLUGIN_PATH,
                        install)
        same = entrypoint.compare_tree_to_commit(install, bare, fx["release_sha"], env)
        assert same["matches"] and not same["next_marker_present"], same
        assert same["mode_differs"] == [], same
        newer = entrypoint.compare_tree_to_commit(install, bare, fx["default_sha"], env)
        assert not newer["matches"]
        assert newer["differing"] == [".codex-plugin/plugin.json"], newer
        assert newer["missing"] == [entrypoint.GIT_NEXT_MARKER], newer
        (install / ".git").mkdir()
        (install / ".git" / "HEAD").write_text("ref: x\n", encoding="utf-8")
        (install / "plans").mkdir()
        (install / "plans" / "p.md").write_text("x\n", encoding="utf-8")
        polluted = entrypoint.compare_tree_to_commit(install, bare, fx["release_sha"], env)
        assert not polluted["matches"]
        assert polluted["has_git_dir"] and polluted["has_plans"], polluted
        assert entrypoint.compare_tree_to_commit(None, bare, fx["release_sha"], env)[
            "compared"] is False

    def test_a_git_seed_runs_the_published_flags(self, rig):
        home, out, log = rig
        _mirror(entrypoint.CODEX_MARKETPLACE_DIR)
        entrypoint._seed_codex_plugin(home, "package", install_source="git")
        argvs = [json.loads(line) for line in log.read_text().splitlines()]
        record = json.loads((out / "plugin-install.json").read_text())
        url = record["git"]["fixture"]["url"]
        assert argvs[0] == ["plugin", "marketplace", "add", url, "--ref", "release",
                            "--sparse", ".agents", "--sparse", "packaging/codex",
                            "--json"]
        assert argvs[1:] == [["plugin", "add", "multi-llm@beastlabai", "--json"],
                             ["plugin", "list", "--json"]]
        assert record["install_source"] == "git"
        assert record["marketplace_dir"] == url
        assert record["overlay"]["GIT_SSH_COMMAND"].startswith("sh ")
        assert record["git"]["tree"]["compared"] is True
        assert record["git"]["snapshot"]["exists"] is False, (
            "the fake codex clones nothing; the record says so rather than "
            "inventing a snapshot")

    def test_an_unknown_install_source_is_refused(self, rig):
        home, _out, log = rig
        with pytest.raises(SystemExit, match="unknown codex install source"):
            entrypoint._seed_codex_plugin(home, "package", install_source="npm")
        assert not log.exists()

    def test_only_the_git_route_probe_installs_through_git(self):
        source = (TESTS_DIR / "container" / "entrypoint.py").read_text(encoding="utf-8")
        assert ('install_source=("git" if args.probe == GIT_ROUTE_PROBE\n'
                in source)
        assert entrypoint.GIT_ROUTE_PROBE in entrypoint.CODEX_PACKAGE_PROBES


def _release_gate():
    import importlib.util

    path = SKILL_DIR.parent.parent / "packaging" / "codex" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("_codex_release_gate_seed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEveryGateRowHasItsOwnGrader:
    """A run record is graded `pass` when ANY test that read the row passed,
    and the generic seed checks pass on every package row. So a gate row whose
    own grader stopped matching it (a renamed probe word, a dropped class)
    would still come back green. This holds the grader text to the matrix."""

    def test_every_required_probe_is_named_by_an_only_guard(self):
        gate = _release_gate()
        source = (TESTS_DIR / "test_container_host_lane.py").read_text(encoding="utf-8")
        named = set()
        for args in re.findall(r"_only\(package_probe,\s*([^)]*)\)", source):
            named |= set(re.findall(r'"([a-z-]+)"', args))
        # `seed` is graded on every row by TestTheCodexPackageSeed by design.
        required = set(gate.EVERY_TIER) - {"seed"}
        assert required <= named, sorted(required - named)
        assert "class TestTheCodexPackageSeed" in source
        assert set(gate.EVERY_TIER) <= set(entrypoint.CODEX_PACKAGE_PROBES)


class TestTheMatrix:
    def test_probe_rows_are_not_credentialed_rows(self):
        credentialed = {row["id"] for row in CODEX["rows"]}
        probes = {row["id"] for row in CODEX["package_probes"]}
        assert not credentialed & probes, (
            "a zero-quota probe under `rows:` would be scheduled with the "
            "OpenRouter key and required to carry a cost budget")
        assert "codex-package-ask" in credentialed

    def test_every_entrypoint_probe_has_exactly_one_row(self):
        named = [row["probe"] for row in CODEX["package_probes"]]
        assert sorted(named) == sorted(entrypoint.CODEX_PACKAGE_PROBES)
        assert len(named) == len(set(named))

    def test_every_probe_cell_is_unmeasured_until_a_run_dates_it(self):
        for row in CODEX["package_probes"]:
            assert row["provenance"] == "unmeasured", row["id"]
            assert isinstance(row["blocking"], bool), row["id"]
            assert row["records"], row["id"]

    def test_every_probe_scenario_is_the_scripted_one_and_exists(self):
        for row in CODEX["package_probes"]:
            scenario = row.get("scenario")
            if scenario is None:
                continue
            local = SKILL_DIR / scenario.removeprefix("/opt/mllm/skill/")
            assert local.name == "scripted_codex_plugin_tool_call.yaml", row["id"]
            assert local.is_file(), row["id"]

    def test_the_exec_driven_probes_name_the_scenario(self):
        exec_driven = {"scripted-call", "loader-tools", "anchor", "environment",
                       "progress-cancellation", "sandbox-posture", "git-route"}
        for row in CODEX["package_probes"]:
            if row["probe"] in exec_driven:
                assert row.get("scenario"), (
                    f"{row['id']} drives `codex exec` against the stub and "
                    f"names no scenario, so the stub would answer text and the "
                    f"tool would never be called")

    def test_codex_package_ask_mirrors_opencode_package_ask(self):
        ours = next(r for r in CODEX["rows"] if r["id"] == "codex-package-ask")
        twin = next(r for r in OPENCODE["rows"] if r["id"] == "opencode-package-ask")
        for key in ("route", "args", "command", "expect", "wall_clock_s",
                    "cost_budget_usd"):
            assert ours[key] == twin[key], key
        # First executed 2026-09-24 (runs/20260923T231950Z, verified); any of
        # the header's three words is legal, a free-text one is not.
        assert ours["provenance"] in ("unmeasured", "verified", "measured-red")

    def test_the_package_ask_prompt_names_the_namespaced_skill(self):
        row = next(r for r in CODEX["rows"] if r["id"] == "codex-package-ask")
        prompt = entrypoint.render_prompt(CODEX, row)
        assert prompt.startswith("Use the `multi-llm:multi-llm` skill")
        assert row["args"] in prompt

    def test_every_other_codex_row_keeps_the_host_template(self):
        # A package-route row is the exception by construction: the plugin
        # advertises the skill namespaced (`multi-llm:multi-llm`, R8′), so each
        # one (`codex-package-ask`, and T10's `codex-package-next-command`)
        # carries the row-16 sentence — graded just below — and no file-route
        # row may borrow it.
        for row in CODEX["rows"]:
            if row.get("route") == "package" or not row.get("args"):
                continue
            assert "prompt_template" not in row, row["id"]
            assert entrypoint.render_prompt(CODEX, row).startswith(
                "Use the multi-llm skill with exactly these arguments")

    def test_every_package_row_names_the_namespaced_skill(self):
        rows = [r for r in CODEX["rows"] if r.get("route") == "package"]
        assert rows
        for row in rows:
            assert entrypoint.render_prompt(CODEX, row).startswith(
                "Use the `multi-llm:multi-llm` skill"), row["id"]


class TestTheHarnessForwardsTheProbe:
    def _argv(self, tmp_path, **kwargs):
        row = harness.Row(row_id="r", lane="host", host="codex", **kwargs)
        return harness.build_run_argv(row, tmp_path, harness.load_posture())

    def test_a_probe_row_forwards_probe_and_route(self, tmp_path):
        argv = self._argv(tmp_path, route="package", probe="seed")
        assert argv[argv.index("--probe") + 1] == "seed"
        assert argv[argv.index("--route") + 1] == "package"

    def test_no_other_row_grows_the_flag(self, tmp_path):
        assert "--probe" not in self._argv(tmp_path)
        assert "--probe" not in self._argv(tmp_path, route="package")


@pytest.mark.parametrize("route", ["file", "package"])
def test_subscription_package_keeps_instrumentation_without_a_key(rig, monkeypatch, route):
    import subprocess

    home, out, _log = rig
    monkeypatch.setattr(entrypoint, "copy_in_fallback_credentials", lambda *args: [])
    expected = {name: f"test-{name}" for name in entrypoint.HARNESS_PROFILE_NAMES}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    entrypoint.wire_fallback_credentials("codex", home, "gpt-6-luna", route=route)
    profile = home / ".profile"
    assert profile.exists() is (route == "package")
    assert not (out / "profile-export.json").exists()
    if route == "file":
        return
    shell_env = {"PATH": os.environ["PATH"]}
    command = '. "$1"; env'
    result = subprocess.run(["/bin/sh", "-c", command, "sh", str(profile)],
                            env=shell_env, text=True, capture_output=True, check=True)
    actual = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert {name: actual[name] for name in expected} == expected
    assert "OPENROUTER_API_KEY" not in actual
    reads = [json.loads(line) for line in (out / entrypoint.PROFILE_READS).read_text().splitlines()]
    assert reads[-1]["key_before_profile"] is False
    assert reads[-1]["key_after_profile"] is False
