"""The row-capture kit (tests/fixtures/row_capture/) still renders what the
adapters render, and the manifests it fed grade only its own captures.

The kit's argv.json is the argv every capture under
``tests/fixtures/<provider>/rows/`` was taken with; a changed ``build_command``
fails here first, which is the cue to re-run ``batch.sh`` for that CLI.

The two shell drivers run here with ``docker`` replaced by a stub first on
PATH and their output under ``tmp_path``: no container starts.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.row_capture import make_argv  # noqa: E402
from tests.test_provider_fixture_rows import MANIFESTS  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402

KIT = Path(__file__).parent / "fixtures" / "row_capture"

#: Rows that grade an older release's capture kept beside the manifest, because
#: the pinned release no longer produces the shape (each row's notes say so).
KEPT_OLDER = {
    ("qwen-code", "exit-zero-failure"),  # 0.21.13, the adapter's min_version
    ("qwen-code", "stdout-error"),  # 0.25.0 with --json-schema
}


def test_argv_json_is_what_the_adapters_render():
    recorded = json.loads((KIT / "argv.json").read_text(encoding="utf-8"))
    assert recorded == make_argv.render(), (
        "an adapter's build_command moved: run "
        "tests/fixtures/row_capture/make_argv.py, then re-capture that CLI's rows "
        "with batch.sh <cli>"
    )


def _which_finding(*names):
    """A ``shutil.which`` that resolves exactly ``names``, as a host PATH would."""
    def which(name, *_args, **_kwargs):
        return f"/usr/local/bin/{name}" if name in names else None
    return which


@pytest.mark.parametrize(
    "host_path",
    [(), ("kilo",), ("kilo", "codex", "kimi", "opencode", "qwen")],
    ids=["nothing-installed", "kilo-only", "kilo-and-the-rest"],
)
def test_argv_json_does_not_depend_on_the_hosts_path(host_path):
    """kilocode's adapter names whichever of ``kilocode``/``kilo`` resolves; the
    kit renders the image's binaries on any host, so a valid ``kilo``-only
    machine renders the committed argv and never writes ``kilo`` into it."""
    recorded = json.loads((KIT / "argv.json").read_text(encoding="utf-8"))
    with mock.patch("shutil.which", _which_finding(*host_path)):
        if "kilo" in host_path:
            # The stub reaches the adapter: outside render(), argv[0] moves.
            assert _PROVIDERS["kilocode"].build_command("p", "m")[0] == "kilo"
        assert make_argv.render() == recorded


def test_each_argv_names_the_image_binary():
    rendered = make_argv.render()
    assert {cli: argv[0] for cli, argv in rendered.items()} == make_argv.image_binaries()


def test_every_kit_cli_has_a_script_row():
    scripts = json.loads((KIT / "scripts.json").read_text(encoding="utf-8"))
    assert set(scripts) == set(make_argv.CLIS)
    for cli, script in scripts.items():
        assert isinstance(json.loads(script["args"]), dict), cli


@pytest.mark.parametrize("provider", make_argv.CLIS)
def test_the_manifest_grades_the_kits_captures(provider):
    """Each kit CLI has a manifest, and every graded row names files under its
    ``rows/`` directory, which nothing but the kit writes, unless it is one
    of the :data:`KEPT_OLDER` rows."""
    manifest = MANIFESTS.get(provider)
    assert manifest, f"{provider} has no tests/fixtures/{provider}/manifest.yaml"
    for name, row in manifest["rows"].items():
        if "not_applicable" in row:
            continue
        in_rows = all(
            str(row.get(key, "rows/")).startswith("rows/") for key in ("stdout", "stderr")
        )
        assert in_rows != ((provider, name) in KEPT_OLDER), (provider, name)


# --------------------------------------------------------------------------- #
# run_row.sh and batch.sh, with the launcher stubbed
# --------------------------------------------------------------------------- #

SHELL_DRIVERS = pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("timeout")),
    reason="POSIX shell scripts run under bash and coreutils timeout",
)

#: A ``docker`` that starts nothing: it logs its argv, writes the row's meta
#: when ``FAKE_DOCKER_WRITES_META`` is set (as incontainer.py would), and exits
#: ``FAKE_DOCKER_EXIT``, or 125 for the row named by ``FAKE_DOCKER_FAIL_ROW``.
_FAKE_DOCKER = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
cli=; row=; prev=
for arg in "$@"; do
  if [ "$prev" = -e ]; then
    case $arg in CLI=*) cli=${arg#CLI=};; ROW=*) row=${arg#ROW=};; esac
  fi
  prev=$arg
done
if [ -n "${FAKE_DOCKER_WRITES_META:-}" ]; then
  mkdir -p "$CAPTURE_OUT/rows/$cli"
  echo '{"new": true}' > "$CAPTURE_OUT/rows/$cli/$row.meta.json"
  echo '{"new": true}'
fi
if [ -n "${FAKE_DOCKER_FAIL_ROW:-}" ] && [ "$row" = "$FAKE_DOCKER_FAIL_ROW" ]; then
  exit 125
fi
exit "${FAKE_DOCKER_EXIT:-0}"
"""


@pytest.fixture
def stub_launcher(tmp_path):
    """The environment the drivers run in: the stub docker first on PATH, the
    capture output under ``tmp_path``, and an image name nothing can pull."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        CAPTURE_OUT=str(tmp_path / "out"),
        FAKE_DOCKER_LOG=str(tmp_path / "docker.log"),
        MLLM_E2E_IMAGE="row-capture-kit-test.invalid/never:pulled",
    )
    for key in ("FAKE_DOCKER_EXIT", "FAKE_DOCKER_FAIL_ROW", "FAKE_DOCKER_WRITES_META"):
        env.pop(key, None)
    return env


def _run(script, args, env):
    return subprocess.run(
        ["bash", str(KIT / script), *args], env=env, capture_output=True, text=True,
        timeout=120,
    )


def _launches(env):
    log = Path(env["FAKE_DOCKER_LOG"])
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@SHELL_DRIVERS
@pytest.mark.parametrize("code", [125, 137, 2],
                         ids=["docker-cannot-start", "outer-timeout-kill", "driver-failed"])
def test_run_row_returns_the_launch_status(stub_launcher, code):
    stub_launcher["FAKE_DOCKER_EXIT"] = str(code)
    result = _run("run_row.sh", ["kimi", "auth-error", "fail", "401"], stub_launcher)

    assert len(_launches(stub_launcher)) == 1
    assert f"docker exit: {code}" in result.stdout
    assert result.returncode == code, result.stderr


@SHELL_DRIVERS
def test_run_row_keeps_a_successful_runs_meta(stub_launcher):
    stub_launcher["FAKE_DOCKER_WRITES_META"] = "1"
    result = _run("run_row.sh", ["kimi", "success", "stub"], stub_launcher)

    assert result.returncode == 0, result.stderr
    meta = Path(stub_launcher["CAPTURE_OUT"]) / "rows" / "kimi" / "success.meta.json"
    assert json.loads(meta.read_text(encoding="utf-8")) == {"new": True}


@SHELL_DRIVERS
def test_a_failed_recapture_leaves_no_previous_evidence(stub_launcher):
    """The previous run's streams and meta at the reused location are gone
    before the launch, so a launch that never starts presents nothing."""
    rows = Path(stub_launcher["CAPTURE_OUT"]) / "rows" / "kimi"
    rows.mkdir(parents=True)
    old = [rows / f"auth-error.{suffix}"
           for suffix in ("stdout.txt", "stderr.txt", "meta.json", "server.jsonl")]
    for path in old:
        path.write_text("previous run\n", encoding="utf-8")
    (rows / "rate-limit.meta.json").write_text("another row\n", encoding="utf-8")
    stub_launcher["FAKE_DOCKER_EXIT"] = "125"

    result = _run("run_row.sh", ["kimi", "auth-error", "fail", "401"], stub_launcher)

    assert result.returncode == 125
    assert [path.name for path in old if path.exists()] == []
    assert (rows / "rate-limit.meta.json").exists(), "only the recaptured row is cleared"


@SHELL_DRIVERS
def test_a_run_that_fails_after_its_meta_keeps_no_meta(stub_launcher):
    """A meta written before the outer deadline killed the run is removed:
    only a capture that exited 0 leaves one."""
    stub_launcher.update(FAKE_DOCKER_WRITES_META="1", FAKE_DOCKER_EXIT="137")
    result = _run("run_row.sh", ["kimi", "success", "stub"], stub_launcher)

    assert result.returncode == 137
    rows = Path(stub_launcher["CAPTURE_OUT"]) / "rows" / "kimi"
    assert not (rows / "success.meta.json").exists()


@SHELL_DRIVERS
def test_run_row_stops_on_a_preparation_failure(stub_launcher, tmp_path):
    """An output directory that cannot be made stops the run before docker."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    stub_launcher["CAPTURE_OUT"] = str(blocker)

    result = _run("run_row.sh", ["kimi", "success", "stub"], stub_launcher)

    assert result.returncode != 0
    assert _launches(stub_launcher) == []


@SHELL_DRIVERS
@pytest.mark.parametrize("args", [["kimi"], ["kimi", "../up", "stub"], ["../up", "x", "stub"]],
                         ids=["too-few-arguments", "row-is-a-path", "cli-is-a-path"])
def test_run_row_refuses_bad_arguments_before_any_change(stub_launcher, args):
    result = _run("run_row.sh", args, stub_launcher)

    assert result.returncode == 2
    assert _launches(stub_launcher) == []
    assert not Path(stub_launcher["CAPTURE_OUT"]).exists()


@SHELL_DRIVERS
def test_batch_fails_when_any_child_fails(stub_launcher):
    stub_launcher["FAKE_DOCKER_FAIL_ROW"] = "auth-error"
    result = _run("batch.sh", ["kimi", "codex"], stub_launcher)

    assert result.returncode == 1
    assert len(_launches(stub_launcher)) == 22, "every row still ran"
    assert "(exit 125)" in result.stdout
    assert "2 run(s) failed: kimi auth-error,codex auth-error" in result.stderr


@SHELL_DRIVERS
def test_batch_succeeds_when_every_child_does(stub_launcher):
    result = _run("batch.sh", ["kimi"], stub_launcher)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("=== kimi ") == 11
    assert "(exit 0)" in result.stdout
