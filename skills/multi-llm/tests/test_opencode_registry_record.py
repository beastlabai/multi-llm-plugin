"""The registry-install record's validator, ``packaging/opencode/registry_record.py``.

**What is graded.** Both profiles (``release`` for the canonical step-5 record,
``bootstrap`` for the laptop path's tag-push check), every refusal the plan
lists, and the first-release exception decided from the tag's commit: each
fixture repository below is a real git repository with an annotated tag whose
``capabilities.yaml`` either has ``hosts.opencode.package.public_version`` (a
later release) or has none (a first release).

**What is not.** The probe that produces these records is a later package; the
records here are built by hand in the shape the schema fixes.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGE_DIR = REPO_ROOT / "packaging" / "opencode"
SCRIPT = PACKAGE_DIR / "registry_record.py"

sys.path.insert(0, str(PACKAGE_DIR))
import registry  # noqa: E402
import registry_record as rr  # noqa: E402

SINGLE = registry.single_registry()
NAME = "opencode-multi-llm"
INTEGRITY = "sha512-" + "Q" * 86 + "=="
DIGEST = "d" * 64
RUN_ID = 40189009375


# ---------------------------------------------------------------------------
# Fixture repositories: the tag's commit decides the release kind
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
                           "-C", str(repo), *args], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _make_repo(root: Path, version: str, public_version=None, *, extra_tag=None) -> dict:
    repo = root / f"repo-{version}-{public_version}"
    (repo / "packaging" / "opencode").mkdir(parents=True)
    (repo / "skills" / "multi-llm" / "hosts").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / rr.TAG_MANIFEST).write_text(json.dumps({"name": NAME, "version": version}),
                                        encoding="utf-8")
    package = {"registry": "npm", "name": NAME, "version": version}
    if public_version is not None:
        package["public_version"] = public_version
    capabilities = {"version": 1, "hosts": {"opencode": {"package": package}}}
    (repo / rr.CAPABILITIES).write_text(json.dumps(capabilities), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"release {version}")
    tag = f"opencode-v{version}"
    _git(repo, "tag", "-a", tag, "-m", f"OpenCode plugin {version}")
    return {"repo": repo, "tag": tag, "version": version, "sha": _git(repo, "rev-parse", "HEAD"),
            "public_version": public_version}


@pytest.fixture(scope="module")
def first(tmp_path_factory):
    """The first release: the tag's capabilities file has no public_version."""
    return _make_repo(tmp_path_factory.mktemp("first"), "0.1.1")


@pytest.fixture(scope="module")
def later(tmp_path_factory):
    """A later release: 0.1.1 was verified, so the tag's file says so."""
    return _make_repo(tmp_path_factory.mktemp("later"), "0.1.2", "0.1.1")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _probe(code="EAI_AGAIN", interfaces=("lo",), registry_url=SINGLE, exit_code=1):
    return {"taken_at": "2026-10-26T10:00:00Z", "interfaces": list(interfaces),
            "npm_view": {"registry": registry_url, "exit": exit_code, "code": code,
                         "argv": ["npm", "view", f"{NAME}@0.1.1", "version",
                                  f"--registry={registry_url}"]}}


def _proof():
    payload = {"location": f"/home/u/.cache/opencode/packages/{NAME}@0.1.1/node_modules/{NAME}",
               "digest": DIGEST}
    return {"network_mode": "none", "before": _probe(), "after": _probe(),
            "payload": {"before": dict(payload), "after": dict(payload)}}


def _criteria(names):
    out = []
    for name in names:
        criterion = {"name": name, "pass": True, "evidence": {"output": f"{name}: ok"}}
        if name == rr.OFFLINE:
            criterion["network_proof"] = _proof()
        out.append(criterion)
    return out


def make(fixture: dict, profile: str = "release") -> dict:
    later_kind = fixture["public_version"] is not None
    if profile == "bootstrap":
        names = rr.BOOTSTRAP_CRITERIA
    else:
        names = rr.LATER_RELEASE_CRITERIA if later_kind else rr.FIRST_RELEASE_CRITERIA
    version = fixture["version"]
    return {
        "schema_version": 1, "kind": "opencode-registry-install-check",
        "recorded_at": "2026-10-26T10:00:00Z",
        "environment": "container" if profile == "release" else "ci-runner",
        "workflow_run_id": None if profile == "release" else RUN_ID,
        "rehearsal": False, "install_source": "npm",
        "spec": f"{NAME}@{version}", "name": NAME, "version": version,
        "registry": SINGLE, "installed_registry": SINGLE + "/",
        "integrity": INTEGRITY, "tag": fixture["tag"], "sha": fixture["sha"],
        "candidate_digest": DIGEST,
        "release_kind": "later" if later_kind else "first",
        "previous_pin": f"{NAME}@{fixture['public_version']}" if later_kind else None,
        "measured": {"opencode": "1.19.4", "python": "3.12.3", "uv": "0.13.5"},
        "variables_set": {"OPENCODE_DISABLE_MODELS_FETCH": "1"},
        "criteria": _criteria(names),
    }


def check(record, fixture, profile="release", **kwargs):
    kwargs.setdefault("tag", fixture["tag"])
    return rr.validate_record(record, profile=profile, repo=fixture["repo"], **kwargs)


def _criterion(record, name):
    return next(c for c in record["criteria"] if c["name"] == name)


def _has(errors, *needles):
    assert any(all(n in e for n in needles) for e in errors), errors


# ---------------------------------------------------------------------------
# Accepted records
# ---------------------------------------------------------------------------

class TestAccepted:

    def test_a_first_release_record_is_accepted(self, first):
        record = make(first)
        assert record["previous_pin"] is None and len(record["criteria"]) == 8
        assert check(record, first) == []

    def test_a_vm_record_is_accepted(self, first):
        record = make(first)
        record["environment"] = "vm"
        assert check(record, first) == []

    def test_a_later_release_record_is_accepted(self, later):
        record = make(later)
        assert record["previous_pin"] == f"{NAME}@0.1.1" and len(record["criteria"]) == 9
        assert check(record, later) == []

    def test_a_bootstrap_record_is_accepted_by_its_profile(self, first):
        record = make(first, "bootstrap")
        assert len(record["criteria"]) == 7
        assert check(record, first, "bootstrap") == []

    def test_the_cli_accepts_a_committed_record(self, first, tmp_path):
        path = tmp_path / "20261026T100000Z-opencode-release-network-check.json"
        path.write_text(json.dumps(make(first)), encoding="utf-8")
        done = subprocess.run([sys.executable, str(SCRIPT), "validate", "--profile", "release",
                               str(path), "--tag", first["tag"], "--repo", str(first["repo"])],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert "valid release record" in done.stdout and "first release" in done.stdout

    def test_the_cli_refuses_and_lists_every_reason(self, first, tmp_path):
        record = make(first)
        record["rehearsal"] = True
        record["registry"] = "https://registry.yarnpkg.com"
        path = tmp_path / "20261026T100000Z-opencode-release-network-check.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        done = subprocess.run([sys.executable, str(SCRIPT), "validate", "--profile", "release",
                               str(path), "--repo", str(first["repo"])],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 1
        assert "rehearsal evidence" in done.stderr and "yarnpkg" in done.stderr

    def test_the_cli_needs_a_profile(self, tmp_path):
        done = subprocess.run([sys.executable, str(SCRIPT), "validate", str(tmp_path / "x.json")],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 2


# ---------------------------------------------------------------------------
# The two profiles
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_a_bootstrap_record_never_passes_the_release_profile(self, first):
        errors = check(make(first, "bootstrap"), first, "release")
        _has(errors, "environment 'ci-runner'")
        _has(errors, "workflow_run_id")
        _has(errors, "missing criteria", rr.OFFLINE)

    def test_a_release_record_never_passes_the_bootstrap_profile(self, first):
        errors = check(make(first), first, "bootstrap")
        _has(errors, "environment 'container'")
        _has(errors, f"cannot carry {rr.OFFLINE}")

    @pytest.mark.parametrize("environment", ["ci-runner", "laptop", None])
    def test_the_release_profile_needs_a_container_or_vm(self, first, environment):
        record = make(first)
        record["environment"] = environment
        _has(check(record, first), "environment")

    def test_the_release_profile_names_no_workflow_run(self, first):
        record = make(first)
        record["workflow_run_id"] = RUN_ID
        _has(check(record, first), "not a workflow run's")

    @pytest.mark.parametrize("run_id", [None, 0, "40189009375", True])
    def test_the_bootstrap_profile_names_its_workflow_run(self, first, run_id):
        record = make(first, "bootstrap")
        record["workflow_run_id"] = run_id
        _has(check(record, first, "bootstrap"), "workflow_run_id")

    @pytest.mark.parametrize("environment", ["container", "vm"])
    def test_the_bootstrap_profile_needs_the_ci_runner(self, first, environment):
        record = make(first, "bootstrap")
        record["environment"] = environment
        _has(check(record, first, "bootstrap"), "environment")

    def test_a_bootstrap_record_rejects_the_offline_criterion(self, first):
        record = make(first, "bootstrap")
        record["criteria"] = _criteria(rr.FIRST_RELEASE_CRITERIA)
        _has(check(record, first, "bootstrap"), f"cannot carry {rr.OFFLINE}")

    def test_a_bootstrap_record_rejects_an_unrun_offline_criterion_too(self, first):
        """Omitted, never reported unrun: a `pass: false` entry is refused as well."""
        record = make(first, "bootstrap")
        record["criteria"].append({"name": rr.OFFLINE, "pass": False, "evidence": "not attempted"})
        _has(check(record, first, "bootstrap"), f"cannot carry {rr.OFFLINE}")

    def test_the_bootstrap_profile_covers_the_first_release_only(self, later):
        record = make(later, "bootstrap")
        record["release_kind"], record["previous_pin"] = "first", None
        errors = check(record, later, "bootstrap")
        _has(errors, "first release only")

    def test_criterion_sets(self):
        assert len(rr.FIRST_RELEASE_CRITERIA) == 8 == len(set(rr.FIRST_RELEASE_CRITERIA))
        assert rr.LATER_RELEASE_CRITERIA == rr.FIRST_RELEASE_CRITERIA + (rr.UPGRADE,)
        assert set(rr.BOOTSTRAP_CRITERIA) == set(rr.FIRST_RELEASE_CRITERIA) - {rr.OFFLINE}
        assert len(rr.BOOTSTRAP_CRITERIA) == 7
        schema = rr.load_schema()
        assert set(schema["$defs"]["criterion"]["properties"]["name"]["enum"]) \
            == set(rr.LATER_RELEASE_CRITERIA)


# ---------------------------------------------------------------------------
# The binding to the release
# ---------------------------------------------------------------------------

class TestBinding:

    def test_a_wrong_tag_is_rejected(self, first):
        _has(check(make(first), first, tag="opencode-v0.1.9"), "is not the release's")

    def test_a_tag_that_does_not_resolve_is_rejected(self, first):
        record = make(first)
        record.update(tag="opencode-v0.1.9", version="0.1.9", spec=f"{NAME}@0.1.9")
        _has(check(record, first, tag=None), "does not resolve")

    def test_a_wrong_sha_is_rejected(self, first):
        record = make(first)
        record["sha"] = "0" * 40
        _has(check(record, first), "is not opencode-v0.1.1's commit")

    def test_a_wrong_version_is_rejected(self, first):
        record = make(first)
        record.update(version="0.1.2", spec=f"{NAME}@0.1.2")
        errors = check(record, first)
        _has(errors, "does not name version '0.1.2'")
        _has(errors, "is not the tag's package.json")

    def test_a_spec_that_is_not_name_at_version_is_rejected(self, first):
        record = make(first)
        record["spec"] = f"{NAME}@latest"
        _has(check(record, first), "spec")

    def test_a_wrong_name_is_rejected(self, first):
        record = make(first)
        record.update(name="opencode-multi-llm-fork", spec="opencode-multi-llm-fork@0.1.1")
        _has(check(record, first), "is not the tag's package.json")

    def test_a_wrong_integrity_is_rejected(self, first):
        errors = check(make(first), first, integrity="sha512-" + "R" * 86 + "==")
        _has(errors, "is not the gated")

    @pytest.mark.parametrize("value", ["sha1-abc", "sha512-short==", None])
    def test_a_malformed_integrity_is_rejected(self, first, value):
        record = make(first)
        record["integrity"] = value
        _has(check(record, first), "integrity")

    def test_a_wrong_candidate_digest_is_rejected(self, first):
        _has(check(make(first), first, digest="e" * 64), "candidate_digest")

    @pytest.mark.parametrize("key", ["registry", "installed_registry"])
    @pytest.mark.parametrize("url", ["https://registry.yarnpkg.com", "http://127.0.0.1:4873",
                                     None])
    def test_another_registry_is_rejected(self, first, key, url):
        record = make(first)
        record[key] = url
        _has(check(record, first), key, "is not the single registry")

    def test_an_unknown_key_is_refused_by_the_closed_schema(self, first):
        record = make(first)
        record["transcript"] = "everything the probe printed"
        _has(check(record, first), "'transcript' is not a key this schema allows")

    def test_a_non_npm_install_source_is_rejected(self, first):
        record = make(first)
        record["install_source"] = "file"
        _has(check(record, first), "install_source")

    def test_measured_versions_are_required(self, first):
        record = make(first)
        del record["measured"]["uv"]
        _has(check(record, first), "measured", "'uv'")


# ---------------------------------------------------------------------------
# Rehearsals
# ---------------------------------------------------------------------------

class TestRehearsal:

    @pytest.mark.parametrize("flag", [True, "false", 0, None])
    def test_anything_but_false_is_rehearsal_evidence(self, first, flag):
        record = make(first)
        record["rehearsal"] = flag
        _has(check(record, first), "rehearsal evidence")

    def test_a_record_without_the_flag_is_rejected(self, first):
        record = make(first)
        del record["rehearsal"]
        _has(check(record, first), "rehearsal evidence")

    def test_a_rehearsal_file_name_is_rejected(self, first, tmp_path):
        path = tmp_path / "20261026T100000Z-opencode-release-network-check-rehearsal.json"
        path.write_text(json.dumps(make(first)), encoding="utf-8")
        _has(rr.validate(path, profile="release", repo=first["repo"]), "rehearsal record by name")

    @pytest.mark.parametrize("profile, name", [
        ("release", "20261026T100000Z-opencode-bootstrap-check.json"),
        ("bootstrap", "20261026T100000Z-opencode-release-network-check.json"),
        ("release", "opencode-release-network-check.json"),
    ])
    def test_a_file_named_for_the_other_profile_is_rejected(self, first, tmp_path, profile, name):
        path = tmp_path / name
        path.write_text(json.dumps(make(first, profile)), encoding="utf-8")
        _has(rr.validate(path, profile=profile, repo=first["repo"]), "is not a <UTC>")


# ---------------------------------------------------------------------------
# The criteria
# ---------------------------------------------------------------------------

class TestCriteria:

    @pytest.mark.parametrize("name", rr.FIRST_RELEASE_CRITERIA)
    def test_each_missing_criterion_is_rejected(self, first, name):
        record = make(first)
        record["criteria"] = [c for c in record["criteria"] if c["name"] != name]
        _has(check(record, first), "missing criteria", name)

    def test_a_record_without_the_same_pin_restart_is_rejected_by_name(self, first):
        record = make(first)
        record["criteria"] = [c for c in record["criteria"] if c["name"] != rr.RESTART]
        _has(check(record, first), f"no {rr.RESTART}")

    def test_a_duplicate_criterion_is_rejected(self, first):
        record = make(first)
        record["criteria"].append(copy.deepcopy(record["criteria"][0]))
        _has(check(record, first), "appears more than once")

    def test_an_unknown_criterion_is_rejected(self, first):
        record = make(first)
        record["criteria"].append({"name": "marketplace_resolves_beastlabai", "pass": True,
                                   "evidence": "x"})
        errors = check(record, first)
        _has(errors, "are not this release record's")

    @pytest.mark.parametrize("verdict", ["true", 1, None, "yes"])
    def test_a_non_boolean_pass_is_rejected(self, first, verdict):
        record = make(first)
        record["criteria"][1]["pass"] = verdict
        _has(check(record, first), "is not a boolean")

    def test_a_red_criterion_is_rejected(self, first):
        record = make(first)
        record["criteria"][2]["pass"] = False
        _has(check(record, first), f"{rr.FIRST_RELEASE_CRITERIA[2]} did not pass")

    @pytest.mark.parametrize("evidence", [None, "", {}, []])
    def test_a_criterion_without_its_evidence_is_rejected(self, first, evidence):
        record = make(first)
        record["criteria"][0]["evidence"] = evidence
        _has(check(record, first), "no supporting evidence")

    def test_a_criterion_list_that_is_not_one_is_rejected(self, first):
        record = make(first)
        record["criteria"] = {"npm_spec_resolves": True}
        _has(check(record, first), "missing criteria")


# ---------------------------------------------------------------------------
# warm_offline_operation's network proof
# ---------------------------------------------------------------------------

def _offline(record):
    return _criterion(record, rr.OFFLINE)


class TestNetworkProof:

    def test_a_passing_offline_criterion_without_its_proof_is_rejected(self, first):
        record = make(first)
        del _offline(record)["network_proof"]
        _has(check(record, first), "without its network proof")

    @pytest.mark.parametrize("mode", ["bridge", "host", "default", None])
    def test_the_container_must_have_had_no_network(self, first, mode):
        record = make(first)
        _offline(record)["network_proof"]["network_mode"] = mode
        _has(check(record, first), "NetworkMode")

    @pytest.mark.parametrize("moment", ["before", "after"])
    @pytest.mark.parametrize("interfaces", [["lo", "eth0"], [], ["eth0"], ["lo", "docker0"]])
    def test_the_interface_list_must_be_loopback_only(self, first, moment, interfaces):
        record = make(first)
        _offline(record)["network_proof"][moment]["interfaces"] = interfaces
        _has(check(record, first), moment, "loopback only")

    @pytest.mark.parametrize("moment", ["before", "after"])
    @pytest.mark.parametrize("code", ["E404", "E401", "E403", "ENEEDAUTH", "ENOTCACHED", "E500",
                                      None])
    def test_npm_view_must_fail_with_a_network_error_code(self, first, moment, code):
        """An E404 or an auth error is an ANSWER from the registry, so the network
        was up; ENOTCACHED is npm's own offline mode, a setting rather than a cut."""
        record = make(first)
        _offline(record)["network_proof"][moment] = _probe(code=code)
        _has(check(record, first), moment, "only a network error code proves the cut")

    @pytest.mark.parametrize("code", sorted(rr.NETWORK_ERROR_CODES))
    def test_every_network_error_code_is_proof(self, first, code):
        record = make(first)
        _offline(record)["network_proof"]["before"] = _probe(code=code)
        assert check(record, first) == []

    def test_a_succeeding_npm_view_is_no_proof(self, first):
        record = make(first)
        _offline(record)["network_proof"]["after"] = _probe(code=None, exit_code=0)
        _has(check(record, first), "after", "network error code")

    def test_the_proof_must_ask_the_single_registry(self, first):
        record = make(first)
        _offline(record)["network_proof"]["before"] = _probe(registry_url="http://127.0.0.1:9")
        _has(check(record, first), "not the single registry")

    @pytest.mark.parametrize("moment", ["before", "after"])
    def test_both_moments_are_required(self, first, moment):
        record = make(first)
        del _offline(record)["network_proof"][moment]
        _has(check(record, first), moment)

    @pytest.mark.parametrize("field, value", [("location", "/elsewhere"), ("digest", "e" * 64)])
    def test_the_payload_must_be_unchanged(self, first, field, value):
        record = make(first)
        _offline(record)["network_proof"]["payload"]["after"][field] = value
        _has(check(record, first), "moved or changed the installed payload")

    def test_the_payload_must_be_recorded(self, first):
        record = make(first)
        del _offline(record)["network_proof"]["payload"]
        _has(check(record, first), "location and digest are not recorded")

    def test_a_proof_on_another_criterion_is_rejected(self, first):
        record = make(first)
        record["criteria"][0]["network_proof"] = _proof()
        _has(check(record, first), "belongs to warm_offline_operation only")


# ---------------------------------------------------------------------------
# The first-release exception, from the tag's commit
# ---------------------------------------------------------------------------

class TestFirstReleaseException:

    def test_a_later_release_record_without_previous_pin_is_rejected(self, later):
        record = make(later)
        record["previous_pin"] = None
        _has(check(record, later), "names its previous_pin")

    def test_a_later_release_record_without_the_upgrade_criterion_is_rejected(self, later):
        record = make(later)
        record["criteria"] = [c for c in record["criteria"] if c["name"] != rr.UPGRADE]
        errors = check(record, later)
        _has(errors, f"must pass {rr.UPGRADE}")

    def test_a_red_upgrade_criterion_is_rejected(self, later):
        record = make(later)
        _criterion(record, rr.UPGRADE)["pass"] = False
        _has(check(record, later), f"{rr.UPGRADE} did not pass")

    @pytest.mark.parametrize("pin", [f"{NAME}@0.1.0", f"{NAME}@0.1.2", "other-package@0.1.1",
                                     "0.1.1"])
    def test_a_previous_pin_other_than_the_public_version_is_rejected(self, later, pin):
        record = make(later)
        record["previous_pin"] = pin
        _has(check(record, later), "is not opencode-multi-llm@0.1.1")

    def test_a_first_release_record_naming_a_previous_pin_is_rejected(self, first):
        record = make(first)
        record["previous_pin"] = f"{NAME}@0.1.0"
        _has(check(record, first), "has no previous pin")

    def test_a_first_release_record_carrying_the_upgrade_criterion_is_rejected(self, first):
        record = make(first)
        record["criteria"] = _criteria(rr.LATER_RELEASE_CRITERIA)
        _has(check(record, first), f"carries no {rr.UPGRADE}")

    def test_the_kind_comes_from_the_tag_not_from_the_record(self, first, later):
        """A record that calls itself `later` on a first-release tag, or `first`
        on a later one, is judged by the tag's commit."""
        record = make(first)
        record["release_kind"] = "later"
        _has(check(record, first), "has no public_version: a first release")
        record = make(later)
        record.update(release_kind="first", previous_pin=None)
        record["criteria"] = _criteria(rr.FIRST_RELEASE_CRITERIA)
        errors = check(record, later)
        _has(errors, "has public_version 0.1.1: a later release")
        _has(errors, "names its previous_pin")

    def test_a_public_version_not_older_than_the_release_is_rejected(self, tmp_path):
        stale = _make_repo(tmp_path, "0.1.2", "0.1.2")
        _has(check(make(stale), stale), "is not older than 0.1.2")

    def test_a_malformed_public_version_is_rejected(self, tmp_path):
        broken = _make_repo(tmp_path, "0.1.2", "soon")
        _has(check(make({**broken, "public_version": "0.1.1"}), broken), "malformed public_version")

    def test_the_exception_reads_the_tag_not_the_working_tree(self, first, tmp_path):
        """A public_version added on a later commit does not make the tag's
        release a later one: the exception is decided at the tag."""
        repo = _make_repo(tmp_path, "0.1.1")
        capabilities = repo["repo"] / rr.CAPABILITIES
        data = json.loads(capabilities.read_text(encoding="utf-8"))
        data["hosts"]["opencode"]["package"]["public_version"] = "0.1.1"
        capabilities.write_text(json.dumps(data), encoding="utf-8")
        _git(repo["repo"], "commit", "-q", "-am", "promote 0.1.1")
        assert check(make(repo), repo) == []


# ---------------------------------------------------------------------------
# The schema, applied without jsonschema
# ---------------------------------------------------------------------------

class TestSchema:

    def _cases(self, first, later):
        valid = [make(first), make(later), make(first, "bootstrap")]
        broken = []
        for mutate in (
            lambda r: r.update(schema_version=2),
            lambda r: r.update(extra=1),
            lambda r: r.update(workflow_run_id=0),
            lambda r: r.update(workflow_run_id=True),
            lambda r: r.update(rehearsal=0),
            lambda r: r.update(sha="ABC"),
            lambda r: r.update(integrity="sha512-x"),
            lambda r: r["criteria"][0].update(pass_="x"),
            lambda r: r["criteria"][0].update({"pass": 1}),
            lambda r: r["criteria"][0].update(name="nope"),
            lambda r: r["measured"].update(uv=3),
            lambda r: r.pop("measured"),
            lambda r: _criterion(r, rr.OFFLINE)["network_proof"].update(extra=1),
            lambda r: _criterion(r, rr.OFFLINE)["network_proof"]["before"]["npm_view"].update(exit="1"),
        ):
            record = make(first)
            mutate(record)
            broken.append(record)
        return valid, broken

    def test_the_interpreter_agrees_with_jsonschema(self, first, later):
        jsonschema = pytest.importorskip("jsonschema")
        schema = rr.load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        valid, broken = self._cases(first, later)
        for record in valid:
            assert rr.schema_errors(record, schema) == []
            assert list(validator.iter_errors(record)) == []
        for record in broken:
            ours = rr.schema_errors(record, schema)
            theirs = list(validator.iter_errors(record))
            assert bool(ours) and bool(theirs), (ours, [e.message for e in theirs])

    def test_the_schema_is_well_formed(self):
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.Draft202012Validator.check_schema(rr.load_schema())

    def test_a_keyword_the_interpreter_does_not_apply_is_refused(self):
        schema = rr.load_schema()
        schema["properties"]["notes"]["maxLength"] = 10
        with pytest.raises(rr.RecordError, match="maxLength"):
            rr._assert_supported(schema)


# ---------------------------------------------------------------------------
# Where the records live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, ignored", [
    ("20990101T000000Z-opencode-release-network-check.json", False),
    ("20990101T000000Z-opencode-bootstrap-check.json", True),
])
def test_only_the_canonical_record_is_committable(name, ignored):
    """The step-5 record is committed (a `.gitignore` exception); a bootstrap
    record stays a workflow artifact and matches none. The validator and the
    schema live in packaging/opencode/, so they need no exception at all."""
    path = f"skills/multi-llm/tests/container/records/{name}"
    done = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path],
                          capture_output=True, text=True, timeout=60)
    assert (done.returncode == 0) is ignored
    for tracked in ("packaging/opencode/registry_record.py",
                    "packaging/opencode/registry-record.schema.json"):
        done = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", tracked],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 1, f"{tracked} is gitignored"
