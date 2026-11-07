"""The Claude package route's container plumbing, with no Docker and no claude binary.

The rows in test_container_claude_package.py need the image. What they stand on does not,
and is pinned here in plain CI: the offline Git marketplace, the install record against a
fake `claude` that behaves as measured, the session parser, the tree comparison, the
entrypoint's wiring order, and the image inputs that decide whether the rows can run.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

# entrypoint.py is the Linux image's entrypoint and imports POSIX-only modules at load.
pytest.importorskip("fcntl", reason="tests/container/entrypoint.py runs only in the Linux image (POSIX-only imports)")
from .container import claude_package, entrypoint, harness, next_command_check

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
CONTAINER = TESTS_DIR / "container"
MATRIX = yaml.safe_load((CONTAINER / "host_matrix.yaml").read_text(encoding="utf-8"))
CLAUDE = MATRIX["hosts"]["claude-code"]

#: Measured behaviour (2.1.318) of the three commands the seed runs: `add` merges into
#: settings and prints a result as its last line under progress lines; `install` copies
#: the plugin into the cache; `list --json` names that path.
FAKE_CLAUDE = r"""#!/bin/sh
printf '%s\n' "$*" >> "$HOME/claude-calls.log"
case "$1 $2" in
  "plugin marketplace")
    python3 - "$HOME/.claude/settings.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
data["extraKnownMarketplaces"] = {"beastlabai": {"source": {"sparsePaths": ["a"]}}}
json.dump(data, open(path, "w"))
PY
    echo "Cloning repository"
    echo '{"command":"marketplace-add","outcome":"ok"}'
    ;;
  "plugin install")
    if [ -z "$FAKE_INSTALL_NOTHING" ]; then
      mkdir -p "$HOME/.claude/plugins/cache/beastlabai/multi-llm/1.2.3"
      cp -r "$FAKE_PLUGIN/." "$HOME/.claude/plugins/cache/beastlabai/multi-llm/1.2.3/"
    fi
    echo '{"command":"install","outcome":"ok"}'
    ;;
  "plugin list")
    if [ -n "$FAKE_INSTALL_NOTHING" ]; then echo '[]'; exit 0; fi
    printf '[{"id":"multi-llm@beastlabai","version":"1.2.3","enabled":true,"installPath":"%s"}]\n' \
      "$HOME/.claude/plugins/cache/beastlabai/multi-llm/1.2.3"
    ;;
esac
"""


def _make_mirror(root: Path) -> Path:
    mirror = root / "mirror"
    plugin = mirror / claude_package.PLUGIN_PATH
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / "hooks").mkdir()
    (plugin / "skills" / "multi-llm").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "multi-llm", "version": "1.2.3"}), encoding="utf-8")
    hook = plugin / "hooks" / "doctor.sh"
    hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hook.chmod(0o755)
    (plugin / "skills" / "multi-llm" / "SKILL.md").write_text("body\n", encoding="utf-8")
    (mirror / ".claude-plugin").mkdir()
    (mirror / claude_package.CATALOG_PATH).write_text(
        json.dumps({"name": "beastlabai", "plugins": [
            {"name": "multi-llm", "source": "./packaging/claude/multi-llm"}]}), encoding="utf-8")
    return mirror


@pytest.fixture
def rig(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    real = tmp_path / "real"
    real.mkdir()
    fake = real / "claude"
    fake.write_text(FAKE_CLAUDE, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "out"
    out.mkdir()
    workspace = tmp_path / "work"
    workspace.mkdir()
    mirror = _make_mirror(tmp_path)
    monkeypatch.setenv("FAKE_PLUGIN", str(mirror / claude_package.PLUGIN_PATH))
    monkeypatch.delenv("FAKE_INSTALL_NOTHING", raising=False)
    return claude_package.Context(home=home, out=out, workspace=workspace, real_dir=real,
                                  log=lambda _line: None, mirror=mirror)


def _git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True)


class TestTheGitMarketplace:
    def test_commit_a_holds_the_mirror_and_the_decoys(self, rig):
        fixture = claude_package.build_fixture(rig.home, rig.mirror)
        listed = _git(fixture["bare"], "ls-tree", "-r", "--name-only", "master").stdout.split()
        assert claude_package.CATALOG_PATH in listed
        assert f"{claude_package.PLUGIN_PATH}/hooks/doctor.sh" in listed
        assert claude_package.DECOY_TEST in listed
        assert fixture["release_version"] == "1.2.3"

    def test_the_url_is_a_form_the_cli_accepts(self, rig):
        """MEASURED: `file://` and `ssh://` are refused as an invalid source."""
        fixture = claude_package.build_fixture(rig.home, rig.mirror)
        assert fixture["url"].startswith("git@localhost:/")

    def test_a_published_style_clone_works_offline_and_skips_the_decoy_blob(self, rig, tmp_path):
        """The clone the CLI performs, through the shim, without the CLI. This is the
        promise the sparse rows rest on: filtered, sparse, and no test blob on disk."""
        fixture = claude_package.build_fixture(rig.home, rig.mirror)
        env = {**os.environ, **fixture["env"], "HOME": str(rig.home)}
        target = tmp_path / "clone"
        done = _git(tmp_path, "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                    fixture["url"], str(target), env=env)
        assert done.returncode == 0, done.stderr
        assert _git(target, "sparse-checkout", "set", "--cone", "--",
                    *claude_package.SPARSE_PATHS, env=env).returncode == 0
        assert _git(target, "read-tree", "-u", "--reset", "HEAD", env=env).returncode == 0
        assert (target / claude_package.PLUGIN_PATH / "hooks" / "doctor.sh").is_file()
        assert not (target / claude_package.DECOY_TEST).exists()
        probe = _git(target, "cat-file", "-e", f"HEAD:{claude_package.DECOY_TEST}",
                     env={**env, "GIT_NO_LAZY_FETCH": "1"})
        assert probe.returncode != 0, "the decoy test blob was fetched despite the filter"

    def test_advancing_moves_the_served_branch_and_the_version(self, rig):
        fixture = claude_package.build_fixture(rig.home, rig.mirror)
        moved = claude_package.advance_fixture(rig.home, fixture, version="1.2.4",
                                               marker=claude_package.NEXT_MARKER, message="next")
        assert _git(fixture["bare"], "rev-parse", "master").stdout.strip() == moved["sha"]
        served = json.loads(_git(fixture["bare"], "show",
                                 f"master:{claude_package.MANIFEST_PATH}").stdout)
        assert served["version"] == "1.2.4"

    def test_advancing_without_a_version_leaves_the_manifest_alone(self, rig):
        fixture = claude_package.build_fixture(rig.home, rig.mirror)
        claude_package.advance_fixture(rig.home, fixture, version=None,
                                       marker=claude_package.SAME_VERSION_MARKER, message="c")
        served = json.loads(_git(fixture["bare"], "show",
                                 f"master:{claude_package.MANIFEST_PATH}").stdout)
        assert served["version"] == "1.2.3"
        assert claude_package.SAME_VERSION_MARKER in _git(
            fixture["bare"], "ls-tree", "-r", "--name-only", "master").stdout

    def test_a_failing_git_step_stops_the_row_by_name(self, tmp_path):
        with pytest.raises(SystemExit, match="git rev-parse"):
            claude_package._git(["rev-parse", "no-such-ref^{commit}"], tmp_path,
                                dict(os.environ))

    @pytest.mark.parametrize("version,want", [("0.1.6", "0.1.7"), ("1.2.3-rc1", "1.2.4")])
    def test_next_patch(self, version, want):
        assert claude_package.next_patch(version) == want

    def test_a_version_that_is_not_semver_stops_the_row(self):
        with pytest.raises(SystemExit, match="not semver"):
            claude_package.next_patch("one.two")


class TestTheSeed:
    def test_the_three_commands_run_in_the_published_order(self, rig):
        claude_package.seed(rig)
        calls = (rig.home / "claude-calls.log").read_text(encoding="utf-8").splitlines()
        assert len(calls) == 3
        assert calls[0].startswith("plugin marketplace add git@localhost:")
        assert calls[0].endswith(f"--sparse {' '.join(claude_package.SPARSE_PATHS)} --json")
        assert calls[1] == f"plugin install {claude_package.PLUGIN_ID} --json"
        assert calls[2] == "plugin list --json"

    def test_the_record_says_what_landed(self, rig):
        record = claude_package.seed(rig)
        on_disk = json.loads((rig.out / claude_package.PLUGIN_INSTALL_RECORD).read_text())
        assert on_disk == json.loads(json.dumps(record))
        assert record["install_source"] == "git"
        assert record["installed_root_exists"] is True
        assert record["version"] == "1.2.3"
        assert record["tree"]["matches"] is True, record["tree"]
        assert record["installed"]["files"] == 3
        assert record["file_shim_present"] is False
        assert record["add_result"] == {"command": "marketplace-add", "outcome": "ok"}
        assert "GIT_SSH_COMMAND" in record["overlay"]

    def test_the_list_document_is_kept_verbatim(self, rig):
        claude_package.seed(rig)
        document = json.loads((rig.out / claude_package.PLUGIN_LIST_RECORD).read_text())
        assert document[0]["id"] == claude_package.PLUGIN_ID

    def test_uncapped_output_never_reaches_the_record(self, rig):
        record = claude_package.seed(rig)
        assert all("_stdout_full" not in command for command in record["commands"])

    def test_an_install_that_left_nothing_stops_the_row(self, rig, monkeypatch):
        monkeypatch.setenv("FAKE_INSTALL_NOTHING", "1")
        with pytest.raises(SystemExit, match="left no installed"):
            claude_package.seed(rig)
        assert (rig.out / claude_package.PLUGIN_INSTALL_RECORD).is_file(), (
            "the commands are the only evidence of why, so the record is written first")

    def test_a_tree_edited_in_the_cache_is_reported(self, rig, monkeypatch):
        tampered = rig.mirror.parent / "tampered"
        shutil.copytree(rig.mirror / claude_package.PLUGIN_PATH, tampered)
        (tampered / "skills" / "multi-llm" / "SKILL.md").write_text("other\n", encoding="utf-8")
        monkeypatch.setenv("FAKE_PLUGIN", str(tampered))
        assert claude_package.seed(rig)["tree"]["differ"] == ["skills/multi-llm/SKILL.md"]


class TestCompareTrees:
    @pytest.fixture
    def pair(self, tmp_path):
        left, right = tmp_path / "a", tmp_path / "b"
        for root in (left, right):
            (root / "hooks").mkdir(parents=True)
            (root / "SKILL.md").write_text("same\n", encoding="utf-8")
            (root / "hooks" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (root / "hooks" / "run.sh").chmod(0o755)
        return left, right

    def test_equal_trees_match(self, pair):
        assert claude_package.compare_trees(*pair)["matches"] is True

    def test_a_changed_byte_is_named(self, pair):
        (pair[0] / "SKILL.md").write_text("Same\n", encoding="utf-8")
        result = claude_package.compare_trees(*pair)
        assert result["matches"] is False and result["differ"] == ["SKILL.md"]

    def test_a_lost_executable_bit_is_named(self, pair):
        """Identical bytes, and a hook that can never run."""
        (pair[0] / "hooks" / "run.sh").chmod(0o644)
        result = claude_package.compare_trees(*pair)
        assert result["matches"] is False and result["mode_differs"] == ["hooks/run.sh"]

    def test_the_bit_comes_from_the_mode_and_not_the_mount(self, pair, monkeypatch):
        """`os.access` also answers for noexec, and the two trees are mounted differently."""
        monkeypatch.setattr(os, "access", lambda path, mode: "/a/" in str(path))
        assert claude_package.compare_trees(*pair)["matches"] is True

    def test_extra_and_missing_files_are_named(self, pair):
        (pair[0] / "extra").write_text("x", encoding="utf-8")
        (pair[1] / "gone").write_text("x", encoding="utf-8")
        result = claude_package.compare_trees(*pair)
        assert result["only_installed"] == ["extra"] and result["only_expected"] == ["gone"]

    def test_run_time_by_products_are_not_the_plugin(self, pair):
        for name in (".venv", "__pycache__", ".in_use"):
            (pair[0] / name).mkdir()
            (pair[0] / name / "x").write_text("x", encoding="utf-8")
        (pair[0] / ".orphaned_at").write_text("1", encoding="utf-8")
        assert claude_package.compare_trees(*pair)["matches"] is True

    def test_a_missing_install_does_not_match(self, pair):
        assert claude_package.compare_trees(pair[0] / "nope", pair[1])["matches"] is False


class TestInventory:
    def test_counts_exclude_run_time_by_products(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "big").write_text("x" * 100, encoding="utf-8")
        (tmp_path / "f").write_text("12345", encoding="utf-8")
        assert claude_package.inventory(tmp_path) == {
            "files": 1, "bytes": 5, "tests": [], "symlinks": []}

    @pytest.mark.parametrize("rel", ["skills/multi-llm/tests/x.py", "test_y.py", "tests/z"])
    def test_a_test_file_is_named(self, tmp_path, rel):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
        assert claude_package.inventory(tmp_path)["tests"] == [rel]

    def test_a_symlink_is_named_because_the_host_refuses_it(self, tmp_path):
        (tmp_path / "link").symlink_to("/etc/hostname")
        assert claude_package.inventory(tmp_path)["symlinks"] == ["link"]

    def test_a_missing_directory_is_an_empty_inventory(self, tmp_path):
        assert claude_package.inventory(tmp_path / "nope")["files"] == 0


class TestParseSession:
    STREAM = "\n".join(json.dumps(event) for event in [
        {"type": "system", "subtype": "hook_started", "hook_name": "SessionStart:startup"},
        {"type": "system", "subtype": "hook_response", "hook_name": "SessionStart:startup",
         "hook_event": "SessionStart", "exit_code": 0, "outcome": "success",
         "output": "multi-llm doctor — host claude-code", "stderr": ""},
        {"type": "system", "subtype": "init",
         "plugins": [{"name": "other", "path": "builtin"},
                     {"name": "multi-llm", "path": "/c/1.2.3", "version": "1.2.3"}],
         "skills": ["multi-llm:multi-llm", "unrelated"],
         "slash_commands": ["multi-llm:multi-llm", "help"]},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
         "extra": "dropped"},
    ])

    def test_it_reads_what_loaded(self):
        session = claude_package.parse_session(self.STREAM)
        assert session["init_seen"] is True
        assert session["plugin"]["path"] == "/c/1.2.3"
        assert session["skills"] == ["multi-llm:multi-llm"]
        assert session["slash_commands"] == ["multi-llm:multi-llm"]
        assert session["plugin_errors"] == []
        assert session["result"] == {"subtype": "success", "is_error": False, "num_turns": 1}

    def test_it_reads_only_hook_responses(self):
        hooks = claude_package.parse_session(self.STREAM)["hooks"]
        assert len(hooks) == 1
        assert hooks[0]["exit_code"] == 0
        assert hooks[0]["output_head"].startswith("multi-llm doctor")

    def test_it_surfaces_plugin_errors(self):
        stream = json.dumps({"type": "system", "subtype": "init", "plugins": [],
                             "plugin_errors": [{"error": "Path escapes plugin directory"}]})
        assert claude_package.parse_session(stream)["plugin_errors"] == [
            {"error": "Path escapes plugin directory"}]

    def test_a_noexec_home_reads_as_a_failed_hook(self):
        stream = json.dumps({"type": "system", "subtype": "hook_response",
                             "hook_name": "SessionStart:startup", "exit_code": 126,
                             "outcome": "error", "output": "Permission denied\n"})
        assert claude_package.parse_session(stream)["hooks"][0]["exit_code"] == 126

    @pytest.mark.parametrize("stdout", ["", "not json\n{broken", "[1, 2]\n"])
    def test_garbage_is_no_init_rather_than_a_crash(self, stdout):
        session = claude_package.parse_session(stdout)
        assert session["init_seen"] is False and session["plugin"] is None


class TestInvocationFacts:
    def test_the_sentinel_is_in_the_body_and_not_the_description(self):
        """The frontmatter reaches every session, so a sentinel there proves nothing."""
        text = (REPO_ROOT / "skills" / "multi-llm" / "SKILL.md").read_text(encoding="utf-8")
        _, frontmatter, body = text.split("---\n", 2)
        assert claude_package.SKILL_BODY_SENTINEL in body
        assert claude_package.SKILL_BODY_SENTINEL not in frontmatter

    def test_the_patterns_tell_the_cases_apart(self):
        import re

        facts = {name: re.compile(pattern) for name, pattern in
                 claude_package.invocation_facts("/h/cache/1.2.3", "mark-1").items()}
        good = ("how this host answers each one\n"
                'uv run --project "/h/cache/1.2.3/skills/multi-llm" -- mark-1')
        hit = {name for name, rx in facts.items() if rx.search(good)}
        assert hit == {"skill_body", "skill_dir_expanded", "argument_marker"}
        bad = 'how this host answers each one\nuv run --project "${CLAUDE_SKILL_DIR}"'
        assert facts["skill_dir_unexpanded"].search(bad)
        assert not facts["skill_dir_expanded"].search(bad)

    def test_stub_facts_are_read_from_requests_that_carry_them(self, tmp_path):
        log = tmp_path / "stub.jsonl"
        log.write_text("\n".join([
            json.dumps({"facts": ["skill_body"]}),
            json.dumps({"facts": []}),
            json.dumps({"no_facts_key": True}),
            "not json",
        ]), encoding="utf-8")
        assert claude_package.stub_facts_seen(log) == (["skill_body"], 2)


class TestTheModelDidNotRunTheNextPhase:
    """Criterion 5, read from receipts because claude-code's stream shows no tool calls."""

    STEP = {"phase": "implement", "command": "/multi-llm:multi-llm --implement x"}

    @staticmethod
    def _receipts(*modes):
        skill = "/c/skills/multi-llm"
        return [{"argv": ["run", "--project", skill, "--", "python", f"{skill}/multi_llm.py",
                          *mode.split()]} for mode in modes]

    def _check(self, *modes, terminal):
        return next_command_check.receipts_no_call_check(
            self.STEP, self._receipts(*modes), terminal)

    def test_the_measured_session_holds(self):
        """Run 20261107T154825Z, whose last receipt is the plugin's own Stop hook."""
        result = self._check("doctor", "util instructions", "generate-tasks", "generate-tasks",
                             "util next-command", "util report-check", terminal=4)
        assert result["observable"] is True and result["holds"] is True, result

    def test_running_the_next_phase_is_a_violation_wherever_it_lands(self):
        result = self._check("implement", "generate-tasks", terminal=1)
        assert result["holds"] is False
        assert result["receipts_naming_next_phase"] == ["implement"]

    def test_any_dispatch_by_the_model_after_the_end_is_a_violation(self):
        result = self._check("generate-tasks", "status", terminal=0)
        assert result["holds"] is False
        assert result["receipts_after_terminal"] == ["status"]

    def test_the_hooks_are_not_the_model(self):
        result = self._check("generate-tasks", "util report-check", "doctor", terminal=0)
        assert result["holds"] is True, result

    def test_with_no_terminal_receipt_it_is_unobserved_rather_than_green(self):
        result = self._check("generate-tasks", terminal=None)
        assert result["observable"] is False and "holds" not in result

    def test_with_nothing_printed_it_is_unobserved_rather_than_green(self):
        result = next_command_check.receipts_no_call_check(None, [], None)
        assert result["observable"] is False and "holds" not in result


class TestTheEntrypointWiring:
    @pytest.fixture
    def seeded(self, monkeypatch, tmp_path):
        calls = []

        def fake_seed(ctx):
            calls.append(json.loads((ctx.home / ".claude" / "settings.json").read_text()))
            return {"overlay": {"GIT_SSH_COMMAND": "sh shim"}}

        monkeypatch.setattr(entrypoint.claude_package, "seed", fake_seed)
        monkeypatch.setattr(entrypoint, "RIG_DIR", tmp_path / "no-rig")
        monkeypatch.setattr(entrypoint, "_skill_allowed_tools", lambda: ["Read"])
        return calls

    def test_the_file_route_installs_nothing(self, seeded, tmp_path):
        overlay = entrypoint.wire_stub_host("claude-code", tmp_path / "h",
                                            "http://127.0.0.1:9/v1", route="file")
        assert seeded == []
        assert "GIT_SSH_COMMAND" not in overlay

    def test_the_package_route_installs_after_the_settings_exist(self, seeded, tmp_path):
        """`marketplace add` merges into that file. Written afterwards, it would silently
        drop the registration, and the row would grade a host with no plugin."""
        overlay = entrypoint.wire_stub_host("claude-code", tmp_path / "h",
                                            "http://127.0.0.1:9/v1", route="package")
        assert len(seeded) == 1 and "permissions" in seeded[0]
        assert overlay["GIT_SSH_COMMAND"] == "sh shim"

    def test_the_stub_is_reached_without_a_doubled_path(self, seeded, tmp_path):
        """The Anthropic SDK appends `/v1/messages` to its base URL."""
        overlay = entrypoint.wire_stub_host("claude-code", tmp_path / "h",
                                            "http://127.0.0.1:9/v1", route="package")
        assert overlay["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9"

    def test_no_real_key_can_be_read_on_the_stub_path(self, seeded, tmp_path):
        overlay = entrypoint.wire_stub_host("claude-code", tmp_path / "h",
                                            "http://127.0.0.1:9/v1", route="package")
        assert overlay["ANTHROPIC_API_KEY"] == ""
        assert overlay["ANTHROPIC_AUTH_TOKEN"] == entrypoint.STUB_PLACEHOLDER_KEY

    def test_every_probe_is_a_parser_choice(self):
        text = (CONTAINER / "entrypoint.py").read_text(encoding="utf-8")
        assert "*claude_package.PROBES" in text

    def test_probe_names_never_collide_with_another_hosts(self):
        """The host a probe needs is looked up from its name."""
        others = set(entrypoint.CODEX_PACKAGE_PROBES) | set(
            entrypoint.opencode_package_gate.GATE_PROBES)
        assert not others & set(claude_package.PROBES)


class TestTheMatrix:
    def _row(self, row_id):
        return next(r for r in CLAUDE["rows"] if r["id"] == row_id)

    def test_every_probe_row_is_one_the_module_runs(self):
        assert {p["probe"] for p in CLAUDE["package_probes"]} == set(claude_package.PROBES)

    def test_probe_rows_stay_out_of_the_credentialed_schedule(self):
        ids = {r["id"] for r in CLAUDE["rows"]}
        assert not ids & {p["id"] for p in CLAUDE["package_probes"]}

    @pytest.mark.parametrize("package,file_route", [
        ("claude-code-package-ask", "claude-code-ask"),
        ("claude-code-package-next-command", "claude-code-next-command"),
    ])
    def test_a_package_row_mirrors_its_twin(self, package, file_route):
        """So that a difference between the two is the route and nothing else."""
        ours, twin = self._row(package), self._row(file_route)
        assert ours["route"] == "package" and "route" not in twin
        for key in ("args", "command", "expect", "wall_clock_s", "cost_budget_usd"):
            assert ours[key] == twin[key], key
        assert "multi-llm:multi-llm" in ours["prompt_template"]


class TestTheImageInputs:
    """The rows cannot run on an image these lines do not produce, and nothing but a
    failed build would otherwise say so."""

    def test_the_context_admits_the_tree_and_the_catalog(self):
        text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "!.claude-plugin/marketplace.json\n" in text
        assert "!packaging/claude/multi-llm/**\n" in text

    def test_a_local_venv_cannot_ride_in(self):
        """Below the negation, since the last match wins."""
        text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert (text.index("\npackaging/claude/multi-llm/skills/multi-llm/.venv/**\n")
                > text.index("!packaging/claude/multi-llm/**"))

    def test_the_manifest_lists_them_too(self):
        text = (CONTAINER / "context_manifest.txt").read_text(encoding="utf-8")
        assert "\n.claude-plugin/marketplace.json\n" in text
        assert "\npackaging/claude/multi-llm/**\n" in text

    def test_the_image_copies_the_mirror_and_the_driver(self):
        text = (CONTAINER / "Dockerfile").read_text(encoding="utf-8")
        assert ("COPY .claude-plugin/marketplace.json "
                "/opt/mllm/claude-marketplace/.claude-plugin/marketplace.json") in text
        assert ("COPY packaging/claude/multi-llm      "
                "/opt/mllm/claude-marketplace/packaging/claude/multi-llm") in text
        assert "/opt/mllm/rig/claude_package.py" in text

    def test_the_mirror_path_is_the_one_the_driver_reads(self):
        assert str(claude_package.MARKETPLACE_DIR) == "/opt/mllm/claude-marketplace"

    def test_hooks_can_execute_under_the_cache(self):
        """MEASURED: under a noexec home each hook exits 126, and the session carries on."""
        mounts = {m["target"]: m for m in yaml.safe_load(
            (CONTAINER / "posture.yaml").read_text(encoding="utf-8"))["mounts"]}
        assert "exec" in mounts["/home/e2e/.claude"]["mode"].split(",")
        assert "exec" not in mounts["/home/e2e"]["mode"].split(","), (
            "widening all of $HOME was never the point")

    def test_the_catalog_names_the_tree_in_the_sparse_paths(self):
        catalog = json.loads((REPO_ROOT / claude_package.CATALOG_PATH).read_text(encoding="utf-8"))
        entry = next(p for p in catalog["plugins"] if p["name"] == claude_package.PLUGIN_NAME)
        assert entry["source"] == f"./{claude_package.PLUGIN_PATH}"
        assert claude_package.PLUGIN_PATH in claude_package.SPARSE_PATHS

    def test_the_published_sparse_paths_are_the_ones_the_rows_use(self):
        text = (REPO_ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
        assert f"--sparse {' '.join(claude_package.SPARSE_PATHS)}" in text

    def test_the_make_target_runs_no_live_row(self):
        text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        recipe = text[text.index("\nclaude-package-container:"):]
        recipe = recipe[:recipe.index("\n\n")]
        assert 'not live' in recipe
        for variable in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            assert f"-u {variable}" in recipe, variable
        assert "test_container_claude_package.py" in recipe
