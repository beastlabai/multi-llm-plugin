"""The Claude Code package route in the container, graded. No key, no route out, $0.

Every other claude-code row stages the source tree as a plain skill, so until these rows
nothing in the Docker suite had loaded the generated plugin, its manifest or its hooks.
Each row here is one `docker run` at `--creds none --route package`, whose wiring step
performs the published sparse install (`container/claude_package.py`), with the host's
model at the loopback stub. The answers are about the install and the load. Whether a
real model follows the skill is `claude-code-package-*` in test_container_host_lane.py.

THE RULE THE GRADERS KEEP, from the codex rows: assert on what was advertised and what
moved, never on an exit status or on a file existing. `plugin list` calls a plugin
enabled while a session refuses to load a thing from it, which is the failure that made
this route worth building, so loading is graded on the session's own `init` event.
"""

from __future__ import annotations

import json
import sys

import pytest
import yaml

from .container import claude_package, harness
from .container.docker_probe import requires_docker

pytestmark = [pytest.mark.container, requires_docker()]

REPO_ROOT = harness.HERE.parents[3]
ENTRY = yaml.safe_load((harness.HERE / "host_matrix.yaml").read_text(encoding="utf-8"))[
    "hosts"]["claude-code"]
PROBE_ROWS = {str(row["probe"]): dict(row) for row in ENTRY["package_probes"]}
SHIPPED_VERSION = json.loads(
    (REPO_ROOT / claude_package.MANIFEST_PATH).read_text(encoding="utf-8"))["version"]


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _run(probe: str, posture, utc):
    row = PROBE_ROWS[probe]
    return harness.run_row(harness.Row(
        row_id=row["id"], lane="host", host="claude-code", creds="none",
        # Inert here, since nothing dispatches a provider, and named rather than
        # defaulted because the entrypoint refuses a row that names no fan-out.
        spec=["opencode:stub/stub-model"], runner="opencode:stub/stub-model",
        route="package", probe=probe,
        row_timeout_s=int(row.get("wall_clock_s") or 600),
    ), posture=posture, utc=utc)


def _bundle(result, name: str):
    path = result.out_dir / name
    assert path.is_file(), (
        f"{result.row.row_id}: no {name} in {result.out_dir}: the row stopped before it "
        f"(see the container's stdout/stderr there)")
    return json.loads(path.read_text(encoding="utf-8"))


def _record(result):
    return _bundle(result, f"probe-{result.row.probe}.json")


@pytest.fixture(scope="module")
def seed_row(posture, utc):
    return _run(claude_package.PROBE_SEED, posture, utc)


@pytest.fixture(scope="module")
def session_row(posture, utc):
    return _run(claude_package.PROBE_SESSION, posture, utc)


@pytest.fixture(scope="module")
def invocation_row(posture, utc):
    return _run(claude_package.PROBE_INVOCATION, posture, utc)


@pytest.fixture(scope="module")
def dispatch_row(posture, utc):
    return _run(claude_package.PROBE_DISPATCH, posture, utc)


@pytest.fixture(scope="module")
def upgrade_row(posture, utc):
    return _run(claude_package.PROBE_UPGRADE, posture, utc)


def test_every_row_the_matrix_declares_has_a_fixture():
    assert set(PROBE_ROWS) == set(claude_package.PROBES)


class TestTheInstall:
    def test_the_row_ran_offline_with_no_credential(self, seed_row):
        assert not seed_row.timed_out, seed_row.cap_reason
        inside = _bundle(seed_row, "posture-inside.json")
        assert not inside["network"]["default_route"], "a zero-quota row had a route out"

    def test_all_three_commands_succeeded(self, seed_row):
        install = _bundle(seed_row, "plugin-install.json")
        failed = [c for c in install["commands"] if c["exit"] != 0]
        assert not failed, json.dumps(failed, indent=2)[:3000]
        assert install["add_result"]["outcome"] == "ok", install["add_result"]
        assert install["install_result"]["outcome"] == "ok", install["install_result"]

    def test_it_landed_in_the_cache_at_the_shipped_version(self, seed_row):
        install = _bundle(seed_row, "plugin-install.json")
        assert install["version"] == SHIPPED_VERSION
        assert install["installed_root"].endswith(
            f"/.claude/plugins/cache/beastlabai/multi-llm/{SHIPPED_VERSION}")
        assert install["list_entry"]["enabled"] is True

    def test_the_install_is_the_generated_tree_byte_for_byte(self, seed_row):
        tree = _bundle(seed_row, "plugin-install.json")["tree"]
        assert tree["matches"], json.dumps(tree, indent=2)[:3000]
        assert tree["compared"] > 100, f"compared only {tree['compared']} files"

    def test_no_test_file_and_no_symlink_was_installed(self, seed_row):
        installed = _bundle(seed_row, "plugin-install.json")["installed"]
        assert installed["tests"] == [], installed["tests"][:20]
        assert installed["symlinks"] == [], (
            f"Claude Code refuses a plugin file resolving outside the plugin directory: "
            f"{installed['symlinks'][:20]}")

    def test_the_install_is_within_its_budget(self, seed_row):
        """The figure the whole payload work exists to hold, measured on what landed."""
        sys.path.insert(0, str(REPO_ROOT / "packaging"))
        from common import payload

        max_files, max_bytes = payload.DISTRIBUTION_BUDGETS["claude-code"]
        installed = _bundle(seed_row, "plugin-install.json")["installed"]
        assert installed["files"] <= max_files, installed
        assert installed["bytes"] <= max_bytes, installed

    def test_no_plain_skill_rides_beside_the_plugin(self, seed_row):
        assert _bundle(seed_row, "plugin-install.json")["file_shim_present"] is False

    def test_the_registration_is_sparse_and_partial(self, seed_row):
        checkout = _bundle(seed_row, "plugin-install.json")["marketplace_checkout"]
        assert checkout["exists"], checkout
        assert checkout["sparse_list"] == list(claude_package.SPARSE_PATHS), checkout
        assert checkout["promisor"] == "true", checkout
        assert checkout["partial_clone_filter"] == "blob:none", checkout

    def test_nothing_outside_the_sparse_paths_arrived(self, seed_row):
        checkout = _bundle(seed_row, "plugin-install.json")["marketplace_checkout"]
        assert checkout["decoys_on_disk"] == [], checkout
        assert checkout["decoy_blob_fetched"] is False, (
            "a test file's blob came down the wire: sparse registration is not saving "
            "what docs/installation.md says it saves")

    def test_the_settings_keep_the_sparse_paths(self, seed_row):
        """The entry a later `marketplace add` must agree with. Without it a re-add is
        refused, and the marketplace vanishes from `list`."""
        registration = _bundle(seed_row, "plugin-install.json")["registration"]
        source = registration["settings_entry"]["source"]
        assert source["sparsePaths"] == list(claude_package.SPARSE_PATHS), registration
        assert registration["known"]["source"]["sparsePaths"] == list(
            claude_package.SPARSE_PATHS), registration
        assert registration["enabled"] is True, registration


class TestAFreshSessionLoadsIt:
    def test_the_session_reported_what_it_loaded(self, session_row):
        record = _record(session_row)
        assert record["session"]["init_seen"], (
            f"no init event: {json.dumps(record['command'])[:2000]}")

    def test_it_loaded_the_plugin_from_the_cache(self, session_row):
        record = _record(session_row)
        plugin = record["session"]["plugin"]
        assert plugin, record["session"]
        assert plugin["path"] == record["installed_root"], (
            "the session is not reading the installed copy, so nothing here grades what "
            f"a user runs: {plugin}")
        assert plugin["version"] == SHIPPED_VERSION

    def test_it_advertises_the_namespaced_skill_and_no_plugin_error(self, session_row):
        """The silent failure this route exists to catch: `plugin list` says enabled, and
        the session refuses every file, with `Path escapes plugin directory`."""
        session = _record(session_row)["session"]
        assert claude_package.SKILL_NAME in session["skills"], session
        assert session["plugin_errors"] == [], session["plugin_errors"]

    def test_the_doctor_hook_really_ran(self, session_row):
        """On its output, because doctor.sh exits 0 on every path it controls. Exit 126
        means the script could not even be executed, as on a noexec home."""
        hooks = _record(session_row)["session"]["hooks"]
        doctor = [h for h in hooks if str(h.get("hook_name")).startswith("SessionStart")]
        assert doctor, f"no SessionStart hook event at all: {hooks}"
        assert doctor[0]["exit_code"] == 0, doctor[0]
        assert doctor[0]["output_head"].startswith("multi-llm doctor"), doctor[0]

    def test_the_stop_hooks_did_not_error(self, session_row):
        hooks = _record(session_row)["session"]["hooks"]
        stopped = [h for h in hooks if str(h.get("hook_name")).startswith("Stop")]
        assert stopped, f"no Stop hook event: the plugin's Stop hooks did not fire: {hooks}"
        bad = [h for h in stopped if h["exit_code"] != 0]
        assert not bad, bad


class TestThePublishedSpellingResolves:
    def test_the_host_sent_a_request(self, invocation_row):
        record = _record(invocation_row)
        assert record["stub_requests"] > 0, (
            f"the host never reached the model: {json.dumps(record['command'])[:2000]}")

    def test_the_skill_body_reached_the_turn(self, invocation_row):
        record = _record(invocation_row)
        assert "skill_body" in record["facts_seen"], (
            f"{record['typed']!r} did not expand the skill: {record['facts_seen']}")

    def test_the_skill_directory_expanded_to_the_cache(self, invocation_row):
        record = _record(invocation_row)
        assert "skill_dir_expanded" in record["facts_seen"], record["facts_seen"]
        assert "skill_dir_unexpanded" not in record["facts_seen"], (
            "the variable survived into the prompt, so every command the skill prints "
            "is unusable as written")

    def test_the_argument_arrived(self, invocation_row):
        record = _record(invocation_row)
        assert "argument_marker" in record["facts_seen"], record["facts_seen"]


class TestTheDispatcherRunsFromTheCache:
    @pytest.mark.parametrize("leg", ["help", "instructions", "doctor", "install_host_check"])
    def test_the_leg_succeeded(self, dispatch_row, leg):
        result = _record(dispatch_row)[leg]
        assert result["exit"] == 0, json.dumps(result, indent=2)[:3000]
        assert "Traceback" not in result["stderr"] + result["stdout"], (
            json.dumps(result, indent=2)[:3000])

    def test_the_instructions_name_the_cache_path(self, dispatch_row):
        record = _record(dispatch_row)
        text = record["instructions"]["stdout"]
        assert record["skill_dir"] in text, text[:1000]
        assert "/opt/mllm/skill" not in text, "the core resolved to the image's source tree"

    def test_status_reads_the_workflow(self, dispatch_row):
        """Exit 2 is the documented "nothing to do": no phase of this plan has run yet."""
        result = _record(dispatch_row)["status"]
        assert result["exit"] == 2, json.dumps(result, indent=2)[:3000]
        assert "workflow (state.json)" in result["stdout"], result["stdout"][:1000]
        assert "Traceback" not in result["stderr"], result["stderr"][:2000]


class TestTheUpgrade:
    def test_a_new_release_is_invisible_until_the_marketplace_updates(self, upgrade_row):
        """Why the documented update has two steps."""
        record = _record(upgrade_row)
        state = record["plugin_update_alone"]["state"]
        assert state["version"] == record["before"]["version"], state
        assert claude_package.NEXT_MARKER not in state["markers"], state

    def test_the_documented_update_reaches_the_next_version(self, upgrade_row):
        record = _record(upgrade_row)
        for step in ("marketplace_update", "plugin_update"):
            assert record[step]["exit"] == 0, json.dumps(record[step])[:2000]
        after = record["after_update"]
        assert after["version"] == record["next_version"], after
        assert claude_package.NEXT_MARKER in after["markers"], after
        assert record["tree_after_update"]["matches"], record["tree_after_update"]

    def test_a_new_session_loads_the_new_version(self, upgrade_row):
        record = _record(upgrade_row)
        plugin = record["session_after_update"]["session"]["plugin"]
        assert plugin["version"] == record["next_version"], plugin
        assert plugin["path"] == record["after_update"]["installed_root"], plugin

    def test_a_byte_moved_under_an_unmoved_version_reaches_nobody(self, upgrade_row):
        """The fact `packaging/claude/build.py --check-release` stands on. If this starts
        failing, Claude Code has begun delivering such a commit, and the rule that every
        shipped change needs a version bump can be relaxed. Read the release notes before
        changing either."""
        record = _record(upgrade_row)
        for step in ("marketplace_update_same_version", "plugin_update_same_version"):
            assert record[step]["exit"] == 0, json.dumps(record[step])[:2000]
        after = record["after_same_version"]
        assert after["version"] == record["next_version"], after
        assert claude_package.SAME_VERSION_MARKER not in after["markers"], after
