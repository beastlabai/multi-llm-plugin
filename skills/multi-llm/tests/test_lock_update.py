"""The pin writer, graded on evidence it is handed rather than on a live image.

``tests/container/lock_update.py`` is what ``make e2e-lock-update`` runs. Its
inputs are files a ``MLLM_PIN_MODE=latest`` build carries inside itself, so they
are trivially fabricable here — which is the point: the interesting cases are the
ones a developer's machine happens not to be in today. The two that matter most
are the *refusals*, since a pin writer that moves a pin it cannot substantiate is
worse than no pin writer at all.

No Docker, no marker: every assertion is a read and a write of temp files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from .container import lock_update

CONTAINER_DIR = Path(lock_update.__file__).resolve().parent
NPM_ROWS = ("codex", "claude-code", "gemini", "opencode", "cline", "kilocode", "pi")


# --------------------------------------------------------------------------- #
# Fixtures: a container dir to rewrite, and an evidence dir to rewrite it from
# --------------------------------------------------------------------------- #
@pytest.fixture
def container_dir(tmp_path):
    """A copy of the four files the writer is allowed to touch."""
    target = tmp_path / "container"
    target.mkdir()
    for name in lock_update.WRITABLE:
        shutil.copy2(CONTAINER_DIR / name, target / name)
    return target


@pytest.fixture
def lock(container_dir):
    return yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))


def _census(lock, moves: dict[str, str], mode: str = "latest") -> dict:
    """The census the build's last layer writes, with `moves` applied."""
    return {
        "mode": mode,
        "platform": lock["platform"],
        "base_image": lock["base_image"],
        "toolchain": [{"id": row["id"], "expected": row["version"],
                       "installed": row["version"], "status": "ok",
                       "enforced": bool(row.get("enforce", True))}
                      for row in lock["toolchain"]],
        "clis": [{"id": row["id"], "binary": row["binary"],
                  "install": row["install"], "expected": row["version"],
                  "installed": moves.get(row["id"], row["version"]),
                  "status": "mismatch" if row["id"] in moves else "ok",
                  "enforced": True}
                 for row in lock["clis"]],
        "path_shape": [{"id": row["id"], "binary": row["binary"], "status": "ok"}
                       for row in lock["clis"]],
    }


@pytest.fixture
def evidence(tmp_path, container_dir, lock):
    """A factory: build an evidence tree for a given set of upstream moves."""
    def build(moves: dict[str, str], *, npm_versions: dict[str, str] | None = None,
              extra_unhashed: list[str] | None = None,
              census: dict | None = None) -> Path:
        root = tmp_path / "evidence"
        shutil.rmtree(root, ignore_errors=True)
        (root / "npm").mkdir(parents=True)
        (root / "artifacts").mkdir(parents=True)

        (root / lock_update.CENSUS_NAME).write_text(
            json.dumps(census if census is not None else _census(lock, moves)),
            encoding="utf-8")

        # The npm lockfile the image resolved: the committed one, moved.
        resolved = json.loads(
            (container_dir / "package-lock.json").read_text("utf-8"))
        packages = {row["id"]: row["package"] for row in lock["clis"]
                    if row["install"] == "npm"}
        for row_id, version in (npm_versions or {}).items():
            package = packages[row_id]
            resolved["packages"][f"node_modules/{package}"]["version"] = version
            resolved["packages"][""]["dependencies"][package] = f"^{version}"
        for name in extra_unhashed or []:
            resolved["packages"][name] = {"version": "1.0.0",
                                          "resolved": f"https://example.invalid/{name}"}
        (root / "npm" / "package-lock.json").write_text(json.dumps(resolved),
                                                        encoding="utf-8")
        (root / "npm" / "package.json").write_text("{}", encoding="utf-8")

        for row in lock["clis"]:
            if row["install"] == "download":
                (root / "artifacts" / row["binary"]).write_bytes(
                    f"{row['id']}-{moves.get(row['id'], row['version'])}".encode())
        return root
    return build


def _apply(container_dir, evidence_root, tmp_path, **kwargs):
    argv = ["apply", "--evidence", str(evidence_root),
            "--container-dir", str(container_dir),
            "--record", str(tmp_path / "record.json"), "--allow-dirty"]
    for key, value in kwargs.items():
        if value is True:
            argv.append(f"--{key.replace('_', '-')}")
    return lock_update.main(argv)


# --------------------------------------------------------------------------- #
# What it moves
# --------------------------------------------------------------------------- #
class TestTheNpmLayerMovesAsOneTree:
    def test_the_lock_manifest_and_lockfile_all_land_on_the_new_version(
            self, container_dir, evidence, tmp_path):
        moves = {"codex": "0.147.0", "gemini": "0.55.1"}
        assert _apply(container_dir, evidence(moves, npm_versions=moves),
                      tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        rows = {row["id"]: row["version"] for row in lock["clis"]}
        assert rows["codex"] == "0.147.0" and rows["gemini"] == "0.55.1"
        # Untouched rows keep their pin: the drift probe said nothing about them.
        # RESTATED, not read from the lock — the point is to disagree with a bad
        # edit, so it moves by hand when the pin does (2026-09-30: 3.0.64 -> 3.0.66,
        # 2026-10-08: 3.0.66 -> 3.0.69).
        assert rows["cline"] == "3.0.69"

        manifest = json.loads((container_dir / "package.json").read_text("utf-8"))
        assert manifest["dependencies"]["@openai/codex"] == "0.147.0"

        resolved = json.loads((container_dir / "package-lock.json").read_text("utf-8"))
        root = resolved["packages"][""]["dependencies"]
        assert root["@openai/codex"] == "0.147.0", (
            "the root requirement must be exact and equal to package.json, or "
            "`npm ci` refuses the pair as out of sync"
        )
        assert resolved["packages"]["node_modules/@openai/codex"]["version"] == "0.147.0"

    def test_the_manifest_and_the_lock_never_disagree(
            self, container_dir, evidence, tmp_path):
        """The invariant tests/test_dockerfile_pins.py asserts, kept by the writer."""
        moves = {"kilocode": "7.4.22", "pi": "0.84.2"}
        _apply(container_dir, evidence(moves, npm_versions=moves), tmp_path)

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        manifest = json.loads((container_dir / "package.json").read_text("utf-8"))
        for row in lock["clis"]:
            if row["install"] == "npm":
                assert manifest["dependencies"][row["package"]] == row["version"]

    def test_a_row_the_two_witnesses_disagree_about_stops_the_layer(
            self, container_dir, evidence, tmp_path):
        """`--version` said one thing, the lockfile another: that is not evidence."""
        root = evidence({"codex": "0.147.0"}, npm_versions={"codex": "0.148.0"})
        assert _apply(container_dir, root, tmp_path) == 2
        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        assert {row["id"]: row["version"] for row in lock["clis"]}["codex"] == "0.161.0"

    def test_a_new_unhashed_dependency_stops_the_layer_but_not_the_run(
            self, container_dir, evidence, tmp_path):
        """A tarball installed by URL alone is a trust decision, not bookkeeping."""
        root = evidence({"codex": "0.147.0", "grok": "1.0.3"},
                        npm_versions={"codex": "0.147.0"},
                        extra_unhashed=["node_modules/brand-new-unhashed"])
        assert _apply(container_dir, root, tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        rows = {row["id"]: row["version"] for row in lock["clis"]}
        assert rows["codex"] == "0.161.0", "the npm layer must not have moved"
        assert rows["grok"] == "1.0.3", (
            "the artifact rows are independent files and must still move"
        )

    def test_an_adjudicated_unhashed_dependency_lets_the_layer_move(
            self, container_dir, evidence, tmp_path):
        """The decision, once taken and committed, is honoured.

        The refusal above tells the reader to widen `UNHASHED_BY_UPSTREAM` in
        `tests/test_dockerfile_pins.py` — and until 2026-09-19 doing exactly that
        changed nothing, because the comparison was against the committed
        lockfile alone and nothing in `lock_update` ever read that allowlist. A
        remedy a tool names and then ignores is worse than no remedy: it reads as
        the tool being broken, and the only way past it was to hand-write the
        lockfile this whole evidence path exists to derive.

        `container_dir` is a temp copy of the four writable files, so the
        allowlist is read from a sibling that is not there and the strict
        behaviour is what the test above still sees. Here the sibling is written,
        naming the very dependency the evidence adds.
        """
        (container_dir.parent / "test_dockerfile_pins.py").write_text(
            'UNHASHED_BY_UPSTREAM = {\n'
            '    "node_modules/brand-new-unhashed",\n'
            '}\n', encoding="utf-8")
        root = evidence({"codex": "0.147.0"},
                        npm_versions={"codex": "0.147.0"},
                        extra_unhashed=["node_modules/brand-new-unhashed"])
        assert _apply(container_dir, root, tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        rows = {row["id"]: row["version"] for row in lock["clis"]}
        assert rows["codex"] == "0.147.0", (
            "the adjudicated entry is named in the allowlist, so the npm layer "
            "must move — this is the assertion that makes the refusal's own "
            "instructions true"
        )

    def test_an_unadjudicated_entry_still_stops_it_when_a_sibling_exists(
            self, container_dir, evidence, tmp_path):
        """Reading the allowlist must not become "allow everything".

        The sibling is present and names a DIFFERENT dependency, which is the
        case that separates "the allowlist is consulted" from "the allowlist is
        assumed to cover whatever turned up".
        """
        (container_dir.parent / "test_dockerfile_pins.py").write_text(
            'UNHASHED_BY_UPSTREAM = {"node_modules/something-else"}\n',
            encoding="utf-8")
        # `grok` rides along for the same reason it does in the sibling test: an
        # artifact row that really moves keeps the RUN alive (exit 0), so the
        # assertion below is "the npm layer was held back" rather than the much
        # weaker "nothing happened at all", which exit 2 would also satisfy.
        root = evidence({"codex": "0.147.0", "grok": "1.0.3"},
                        npm_versions={"codex": "0.147.0"},
                        extra_unhashed=["node_modules/brand-new-unhashed"])
        assert _apply(container_dir, root, tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        rows = {row["id"]: row["version"] for row in lock["clis"]}
        assert rows["codex"] == "0.161.0", "the npm layer must not have moved"
        assert rows["grok"] == "1.0.3", "and the artifact rows must still move"
        record = json.loads((tmp_path / "record.json").read_text("utf-8"))
        assert any(entry["row"] == "npm layer" for entry in record["not_moved"])


class TestTheArtifactRowsMoveOnlyWhenTheBytesSurvive:
    def test_a_download_row_gets_a_recomputed_checksum_and_a_retargeted_url(
            self, container_dir, evidence, tmp_path):
        import hashlib

        root = evidence({"grok": "1.0.3"})
        expected = hashlib.sha256(b"grok-1.0.3").hexdigest()
        assert _apply(container_dir, root, tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        artifact = lock["artifacts"]["grok"]
        assert artifact["version"] == "1.0.3"
        assert artifact["sha256"] == expected
        assert artifact["url"].endswith("grok-1.0.3-linux-x86_64")
        assert "0.2.93" not in artifact["url"]

        dockerfile = (container_dir / "Dockerfile").read_text("utf-8")
        assert f"ARG GROK_SHA256={expected}" in dockerfile, (
            "the ARG and the lock are two copies of one checksum; "
            "tests/test_dockerfile_pins.py asserts they agree"
        )
        assert "ARG GROK_VERSION=1.0.3" in dockerfile

    def test_the_omp_row_moves_its_three_args_and_its_url_together(
            self, container_dir, evidence, tmp_path):
        """The second download row, and the one whose URL carries a `v` prefix.

        `retarget_url` substitutes the bare version inside the recorded URL and
        refuses anything but an exactly-once match, so a tag spelled `v18.2.2`
        against a row version of `18.2.2` is the case worth pinning: the tag is
        where the only occurrence lives.
        """
        import hashlib

        root = evidence({"omp": "19.0.1"})
        expected = hashlib.sha256(b"omp-19.0.1").hexdigest()
        assert _apply(container_dir, root, tmp_path) == 0

        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        artifact = lock["artifacts"]["omp"]
        assert artifact["version"] == "19.0.1"
        assert artifact["sha256"] == expected, (
            "the digest must be recomputed from /opt/mllm/real/omp, which is the "
            "artifact itself only while `install: download` copies it unchanged"
        )
        assert artifact["url"].endswith("/v19.0.1/omp-linux-x64")
        assert "18.2.2" not in artifact["url"]
        assert artifact["releases_url"].endswith("/releases/latest"), (
            "the channel pointer carries no version and must not be rewritten"
        )

        dockerfile = (container_dir / "Dockerfile").read_text("utf-8")
        assert "ARG OMP_VERSION=19.0.1" in dockerfile
        assert f"ARG OMP_SHA256={expected}" in dockerfile
        assert f"ARG OMP_URL={artifact['url']}" in dockerfile, (
            "all three ARGs move together or the build verifies bytes the lock "
            "does not name; tests/test_dockerfile_pins.py asserts they agree"
        )

    def test_muses_manifest_url_follows_its_version(
            self, container_dir, evidence, tmp_path):
        _apply(container_dir, evidence({"muse": "0.1.0-R900.2"}), tmp_path)
        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        artifact = lock["artifacts"]["muse"]
        assert "0.1.0-R900.2" in artifact["url"]
        assert "0.1.0-R900.2" in artifact["manifest_url"]
        assert artifact["channel_url"].endswith("muse-stable"), (
            "the channel pointer carries no version and must not be rewritten"
        )

    def test_an_archive_row_is_refused_with_the_command(
            self, container_dir, evidence, tmp_path):
        """goose is unpacked and the tarball discarded: the checksum is gone."""
        assert _apply(container_dir, evidence({"goose": "1.46.0"}), tmp_path) == 2
        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        assert lock["artifacts"]["goose"]["version"] == "1.53.0"

        record = json.loads((tmp_path / "record.json").read_text("utf-8"))
        [refused] = [entry for entry in record["not_moved"] if entry["row"] == "goose"]
        assert "v1.46.0" in refused["command"] and "sha256sum" in refused["command"]

    def test_a_pypi_row_is_refused_with_the_command(
            self, container_dir, evidence, tmp_path):
        assert _apply(container_dir, evidence({"aider": "0.90.0"}), tmp_path) == 2
        record = json.loads((tmp_path / "record.json").read_text("utf-8"))
        [refused] = [entry for entry in record["not_moved"] if entry["row"] == "aider"]
        assert "--generate-hashes" in refused["command"]


# --------------------------------------------------------------------------- #
# What it refuses outright
# --------------------------------------------------------------------------- #
class TestEvidenceItRefusesToActOn:
    def test_a_pinned_census_proves_nothing_about_what_moved(
            self, container_dir, evidence, tmp_path, lock, capsys):
        root = evidence({}, census=_census(lock, {"codex": "0.147.0"}, mode="pinned"))
        assert _apply(container_dir, root, tmp_path) == 1
        assert "mode 'pinned'" in capsys.readouterr().err

    def test_a_census_from_another_platform_cannot_move_this_pin(
            self, container_dir, evidence, tmp_path, lock, capsys):
        census = _census(lock, {"codex": "0.147.0"})
        census["platform"] = "linux/arm64"
        assert _apply(container_dir, evidence({}, census=census), tmp_path) == 1
        assert "arm64" in capsys.readouterr().err

    def test_a_drifted_toolchain_invalidates_every_version_under_it(
            self, container_dir, evidence, tmp_path, lock, capsys):
        census = _census(lock, {"codex": "0.147.0"})
        census["toolchain"][0].update(installed="23.0.0", status="mismatch")
        assert _apply(container_dir, evidence({}, census=census), tmp_path) == 1
        assert "toolchain" in capsys.readouterr().err

    def test_a_broken_wrapper_tree_is_not_evidence_about_anything(
            self, container_dir, evidence, tmp_path, lock, capsys):
        census = _census(lock, {"codex": "0.147.0"})
        census["path_shape"][0]["status"] = "not-wrapped"
        assert _apply(container_dir, evidence({}, census=census), tmp_path) == 1
        assert "wrapper tree" in capsys.readouterr().err

    def test_an_ambiguous_version_inside_a_url_is_a_guess_not_a_pin(self):
        with pytest.raises(lock_update.Refusal):
            lock_update.retarget_url("https://x/1.0/pkg-1.0.tar", "1.0", "2.0", "x")

    def test_nothing_is_written_when_nothing_can_be_substantiated(
            self, container_dir, evidence, tmp_path):
        before = {name: (container_dir / name).read_bytes()
                  for name in lock_update.WRITABLE}
        assert _apply(container_dir, evidence({"goose": "1.46.0"}), tmp_path) == 2
        assert {name: (container_dir / name).read_bytes()
                for name in lock_update.WRITABLE} == before


class TestDryRunWritesNothing:
    def test_the_plan_prints_and_the_files_stay_put(
            self, container_dir, evidence, tmp_path, capsys):
        before = (container_dir / "cli-versions.lock").read_bytes()
        assert _apply(container_dir, evidence({"grok": "1.0.3"}), tmp_path,
                      dry_run=True) == 0
        assert (container_dir / "cli-versions.lock").read_bytes() == before
        assert "MOVE" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The record, and the half that grades it
# --------------------------------------------------------------------------- #
class TestTheRecordIsVerifiedAgainstTheRebuiltImage:
    @pytest.fixture
    def rewritten(self, container_dir, evidence, tmp_path):
        moves = {"codex": "0.147.0", "grok": "1.0.3"}
        assert _apply(container_dir, evidence(moves, npm_versions={"codex": "0.147.0"}),
                      tmp_path) == 0
        return tmp_path / "record.json"

    def _verify(self, container_dir, record, census, tmp_path):
        path = tmp_path / "pinned.json"
        path.write_text(json.dumps(census), encoding="utf-8")
        return lock_update.main(["verify", "--record", str(record),
                                 "--installed", str(path),
                                 "--container-dir", str(container_dir)])

    def _pinned_census(self, container_dir):
        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        return _census(lock, {}, mode="pinned")

    def test_a_rebuild_that_installed_the_new_pins_verifies(
            self, container_dir, rewritten, tmp_path):
        assert self._verify(container_dir, rewritten,
                            self._pinned_census(container_dir), tmp_path) == 0

    def test_a_rebuild_that_installed_something_else_fails(
            self, container_dir, rewritten, tmp_path, capsys):
        census = self._pinned_census(container_dir)
        for row in census["clis"]:
            if row["id"] == "codex":
                row.update(installed="0.146.0", status="mismatch")
        assert self._verify(container_dir, rewritten, census, tmp_path) == 1
        assert "git checkout --" in capsys.readouterr().err

    def test_a_file_edited_between_apply_and_verify_fails(
            self, container_dir, rewritten, tmp_path, capsys):
        census = self._pinned_census(container_dir)
        (container_dir / "cli-versions.lock").write_text("version: 1\n", "utf-8")
        assert self._verify(container_dir, rewritten, census, tmp_path) == 1
        assert "changed since the rewrite" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The line editor
# --------------------------------------------------------------------------- #
class TestTheRewriteIsSurgical:
    def test_the_prose_survives(self, container_dir, evidence, tmp_path):
        """~150 of the lock's 270 lines are the reasoning behind the pins."""
        before = (container_dir / "cli-versions.lock").read_text("utf-8")
        _apply(container_dir, evidence({"grok": "1.0.3"}), tmp_path)
        after = (container_dir / "cli-versions.lock").read_text("utf-8")

        comments = [line for line in before.splitlines() if line.lstrip().startswith("#")]
        assert comments and all(line in after for line in comments)
        assert len(before.splitlines()) == len(after.splitlines())

    def test_the_evidence_list_is_derived_from_the_lock(self, container_dir, capsys):
        assert lock_update.main(
            ["evidence", "--container-dir", str(container_dir)]) == 0
        printed = dict(line.split("\t")
                       for line in capsys.readouterr().out.splitlines())
        lock = yaml.safe_load((container_dir / "cli-versions.lock").read_text("utf-8"))
        for row in lock["clis"]:
            path = f"/opt/mllm/real/{row['binary']}"
            assert (path in printed) == (row["install"] == "download"), (
                f"{row['id']} is install:{row['install']} — only a download row's "
                f"installed file IS the artifact it was fetched as"
            )
