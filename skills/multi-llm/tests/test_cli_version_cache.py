#!/usr/bin/env python3
"""Doctor's version cache (`utils/cli_version_cache.py`, PERF-5 / PROV-12).

A parsed `--version` string is remembered per provider, keyed on the resolved
executable's path, size, mtime, ctime and inode plus the parser's source file,
so a SessionStart doctor skips the spawn. The executor never reads it, auth
and sandbox probes are never cached, a failed read is never recorded, and a
corrupt or concurrently rewritten file fails open to a fresh probe.

Every binary here is a `#!/bin/sh` script under tmp_path that appends to a log
on each run, so "probed" and "not probed" are observed, not assumed.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cli_version_cache as cvc  # noqa: E402
from utils import doctor  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from tests.source_cache import python_files, source_tree

SKILL_ROOT = Path(__file__).resolve().parents[1]

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="the fake CLIs are `#!/bin/sh` scripts on a replaced PATH"
)


def _cli(bin_dir: Path, name: str, version: str, log: Path) -> Path:
    target = bin_dir / name
    target.write_text(
        f'#!/bin/sh\necho run >> "{log}"\necho "{name} {version}"\n', encoding="utf-8"
    )
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return target


def _runs(log: Path) -> int:
    return len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0


class FakeProvider:
    """The three attributes `_probe_version` and `version_verdict` read."""

    name = "fake-cache-cli"
    verified_version = ""

    def __init__(self, command: str, min_version: str = ""):
        self.version_cmd = (command, "--version")
        self.min_version = min_version

    def parse_version(self, stdout, stderr):
        match = re.search(r"\b(\d+\.\d+(?:\.\d+)*)\b", stdout or "")
        return match.group(1) if match else None


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("the fake CLIs are `#!/bin/sh` scripts on a replaced PATH")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "runs.log"
    path = _cli(bin_dir, "fakecli", "1.2.3", log)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]))
    return path, log


def _probe(provider, cache_path, **cache_kwargs):
    cache = cvc.VersionCache(str(cache_path), **cache_kwargs)
    row = doctor._probe_version(provider, "fakecli", 10.0, cache)
    cache.save()
    return row


# ---------------------------------------------------------------------------
# Hits and misses
# ---------------------------------------------------------------------------

class TestHitAndMiss:

    def test_a_second_run_reads_the_cache_and_spawns_nothing(self, fake_cli, tmp_path):
        path, log = fake_cli
        cache_path = tmp_path / "cache" / "cli-versions.json"
        provider = FakeProvider("fakecli")
        first = _probe(provider, cache_path)
        assert (first["version"], first["source"], _runs(log)) == ("1.2.3", "probe", 1)
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        assert stored["schema"] == cvc.SCHEMA
        entry = stored["entries"]["fakecli"]
        # Only the parsed string is stored, never the probe's raw output.
        assert entry["version"] == "1.2.3"
        assert "fakecli 1.2.3" not in cache_path.read_text(encoding="utf-8")
        assert entry["key"]["files"][0]["path"] == os.path.realpath(path)

        second = _probe(provider, cache_path)
        assert (second["version"], second["source"], _runs(log)) == ("1.2.3", "cache", 1)
        assert second["state"] == first["state"] and second["ok"] is True

    def test_the_floor_verdict_is_taken_live_on_a_cached_version(self, fake_cli, tmp_path):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        _probe(FakeProvider("fakecli"), cache_path)
        row = _probe(FakeProvider("fakecli", min_version="2.0.0"), cache_path)
        assert row["source"] == "cache" and _runs(log) == 1
        assert row["ok"] is False and "older than the minimum" in row["detail"]

    def test_same_size_new_mtime_is_a_miss(self, fake_cli, tmp_path):
        """A launcher can keep its size, or its exact bytes, across versions:
        the mtime carries the key (roadmap section 9)."""
        path, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        before = path.stat()
        path.write_text(path.read_text(encoding="utf-8").replace("1.2.3", "1.2.4"),
                        encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
        assert path.stat().st_size == before.st_size
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"], _runs(log)) == ("1.2.4", "probe", 2)
        assert _probe(provider, cache_path)["source"] == "cache"

    def test_identical_bytes_with_only_the_mtime_moved_is_a_miss(self, fake_cli, tmp_path):
        path, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        assert _probe(provider, cache_path)["source"] == "probe"
        assert _runs(log) == 2

    def test_a_replaced_file_with_the_old_mtime_is_a_miss(self, fake_cli, tmp_path):
        """`cp -p`/`install -p` keep the mtime; the inode and ctime move."""
        path, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        st = path.stat()
        replacement = path.with_name("fakecli.new")
        replacement.write_text(path.read_text(encoding="utf-8").replace("1.2.3", "1.2.9"),
                               encoding="utf-8")
        replacement.chmod(st.st_mode)
        os.utime(replacement, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(replacement, path)
        assert path.stat().st_mtime_ns == st.st_mtime_ns
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"]) == ("1.2.9", "probe")

    def test_a_retargeted_symlink_is_a_miss(self, fake_cli, tmp_path, monkeypatch):
        path, log = fake_cli
        other = _cli(tmp_path, "fakecli-2", "3.0.0", log)
        link_dir = tmp_path / "linkbin"
        link_dir.mkdir()
        link = link_dir / "linkcli"
        link.symlink_to(path)
        monkeypatch.setenv("PATH", os.pathsep.join([str(link_dir), os.environ["PATH"]]))
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("linkcli")
        assert _probe(provider, cache_path)["version"] == "1.2.3"
        link.unlink()
        link.symlink_to(other)
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"]) == ("3.0.0", "probe")

    def test_a_changed_parser_source_is_a_miss(self, fake_cli, tmp_path, monkeypatch):
        """The cached value is the parser's reading, so its file is keyed too."""
        _, log = fake_cli
        parser_file = tmp_path / "parser.py"
        parser_file.write_text("# parser\n", encoding="utf-8")
        monkeypatch.setattr(cvc, "_parser_source", lambda provider: str(parser_file))
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        assert _probe(provider, cache_path)["source"] == "cache"
        st = parser_file.stat()
        os.utime(parser_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        assert _probe(provider, cache_path)["source"] == "probe"

    def test_the_real_adapters_parser_file_is_in_the_key(self):
        source = Path(cvc._parser_source(registry.get_provider("grok")))
        assert source.suffix == ".py" and (SKILL_ROOT / "utils" / "providers") in source.parents

    def test_an_expired_or_future_entry_is_a_miss(self, fake_cli, tmp_path):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        now = time.time()
        _probe(provider, cache_path, now=now - cvc.TTL_S - 60)
        assert _probe(provider, cache_path, now=now)["source"] == "probe"
        _probe(provider, cache_path, now=now + 3600, read=False)
        assert _probe(provider, cache_path, now=now)["source"] == "probe"
        assert _runs(log) == 4

    def test_fresh_reads_nothing_and_still_writes(self, fake_cli, tmp_path):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        assert _probe(provider, cache_path, read=False)["source"] == "probe"
        assert _runs(log) == 2
        assert _probe(provider, cache_path)["source"] == "cache"


class TestNothingUncertainIsCached:

    def test_a_failed_read_is_never_recorded(self, fake_cli, tmp_path):
        path, log = fake_cli
        path.write_text(f'#!/bin/sh\necho run >> "{log}"\necho "no digits"\n', encoding="utf-8")
        cache_path = tmp_path / "cli-versions.json"
        row = _probe(FakeProvider("fakecli"), cache_path)
        assert row["state"] == "unread"
        assert not cache_path.exists()

    def test_a_probe_that_ran_out_of_budget_is_never_recorded(self, fake_cli, tmp_path):
        path, log = fake_cli
        path.write_text('#!/bin/sh\nsleep 5\necho "fakecli 1.2.3"\n', encoding="utf-8")
        cache = cvc.VersionCache(str(tmp_path / "cli-versions.json"))
        row = doctor._probe_version(FakeProvider("fakecli"), "fakecli", 0.2, cache)
        assert row["detail"].startswith("slow")
        assert cache.save() is False

    def test_an_unresolved_command_is_never_cached(self, tmp_path):
        cache = cvc.VersionCache(str(tmp_path / "cli-versions.json"))
        row = doctor._probe_version(FakeProvider("no-such-cli-xyz"), "x", 1.0, cache)
        assert row["source"] == "probe"
        assert doctor._version_cache_key(FakeProvider("no-such-cli-xyz"),
                                         ("no-such-cli-xyz", "--version")) is None
        assert cache.save() is False


# ---------------------------------------------------------------------------
# A file the cache cannot trust fails open
# ---------------------------------------------------------------------------

class TestCorruptCache:

    @pytest.mark.parametrize("content", [
        "{not json",
        "",
        "[]",
        json.dumps({"schema": cvc.SCHEMA + 1, "entries": {}}),
        json.dumps({"schema": cvc.SCHEMA, "entries": []}),
        json.dumps({"schema": cvc.SCHEMA, "entries": {"fakecli": "1.2.3"}}),
        json.dumps({"schema": cvc.SCHEMA, "entries": {"fakecli": {
            "key": None, "version": 7, "probed_at": "yesterday"}}}),
        "\x00\xff garbage",
        b"\xff\xfe\x00 not utf-8",
    ])
    def test_a_corrupt_file_is_an_empty_cache_and_is_replaced(
        self, fake_cli, tmp_path, content
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        if isinstance(content, bytes):
            cache_path.write_bytes(content)
        else:
            cache_path.write_text(content, encoding="utf-8")
        provider = FakeProvider("fakecli")
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"], _runs(log)) == ("1.2.3", "probe", 1)
        assert json.loads(cache_path.read_text(encoding="utf-8"))["schema"] == cvc.SCHEMA
        assert _probe(provider, cache_path)["source"] == "cache"

    def test_a_tampered_entry_with_the_right_key_but_a_bad_version_is_a_miss(
        self, fake_cli, tmp_path
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        data["entries"]["fakecli"]["version"] = "   "
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        assert _probe(provider, cache_path)["source"] == "probe"

    def test_an_unwritable_cache_dir_fails_open(self, fake_cli, tmp_path):
        _, log = fake_cli
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where the directory should be", encoding="utf-8")
        cache = cvc.VersionCache(str(blocker / "multi-llm" / "cli-versions.json"))
        row = doctor._probe_version(FakeProvider("fakecli"), "fakecli", 10.0, cache)
        assert row["version"] == "1.2.3"
        assert cache.save() is False


def _plant(cache_path: Path, name: str = "fakecli", **fields) -> None:
    """Rewrite fields of an entry doctor wrote, keeping its matching key (and
    the file's own mode: ``write_text`` truncates in place)."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    data["entries"][name].update(fields)
    cache_path.write_text(json.dumps(data), encoding="utf-8")


class TestOnlyWhatDoctorWroteIsRead:
    """An entry is read only when this module could have written it: a finite
    `probed_at` inside the window, a version the adapter's parser reads back
    unchanged, in a file and directory this user owns, that are not symlinks
    and not group- or world-writable (Codex review of U28, points 1 and 2)."""

    @pytest.mark.parametrize("probed_at", [
        10 ** 400, float("nan"), float("inf"), float("-inf"), 1e308, -1e308,
        True, "1700000000", None,
    ], ids=lambda value: repr(value)[:24])
    def test_a_huge_or_non_finite_probed_at_is_a_miss_not_a_crash(
        self, fake_cli, tmp_path, probed_at
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        _plant(cache_path, probed_at=probed_at)
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"], _runs(log)) == ("1.2.3", "probe", 2)

    @pytest.mark.parametrize("version", [
        "unknown", "Unknown", "v1.2.3", "1.2.3 ", "fakecli 1.2.3", "1.2.3\n", "1",
        "abc", "", "9" * 70 + ".0", 123, None, ["1.2.3"],
    ], ids=lambda value: repr(value)[:24])
    def test_a_version_the_parser_could_not_have_returned_is_a_miss(
        self, fake_cli, tmp_path, version
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        _plant(cache_path, version=version)
        row = _probe(provider, cache_path)
        assert (row["version"], row["source"], _runs(log)) == ("1.2.3", "probe", 2)

    def test_a_version_the_parser_would_not_read_is_never_recorded(self, fake_cli, tmp_path):
        """A parser whose output does not read back as itself is never cached,
        so its entries could not pass a lookup either."""
        _, log = fake_cli

        class Prefixed(FakeProvider):
            def parse_version(self, stdout, stderr):
                found = super().parse_version(stdout, stderr)
                return f"v{found}" if found else None

        cache_path = tmp_path / "cli-versions.json"
        assert _probe(Prefixed("fakecli"), cache_path)["version"] == "v1.2.3"
        assert not cache_path.exists()

    def test_a_planted_entry_cannot_turn_a_floor_failure_into_ok(self, fake_cli, tmp_path):
        """Codex's reproduction: a CLI below its floor stays below it whatever
        a matching entry says — `unknown` in a private file, or a passing
        version in a world-writable one."""
        path, log = fake_cli
        path.write_text(path.read_text(encoding="utf-8").replace("1.2.3", "0.1.0"),
                        encoding="utf-8")
        provider = FakeProvider("fakecli", min_version="0.144.0")
        cache_path = tmp_path / "cache" / "cli-versions.json"
        first = _probe(provider, cache_path)
        assert (first["version"], first["ok"]) == ("0.1.0", False)
        _plant(cache_path, version="unknown")
        row = _probe(provider, cache_path)
        assert (row["version"], row["ok"], row["source"]) == ("0.1.0", False, "probe")
        _plant(cache_path, version="0.146.0")
        cache_path.chmod(0o666)
        row = _probe(provider, cache_path)
        assert (row["version"], row["ok"], row["source"]) == ("0.1.0", False, "probe")
        assert _runs(log) == 3

    @posix_only
    @pytest.mark.parametrize("mode", [0o666, 0o660, 0o620, 0o602])
    def test_a_group_or_world_writable_file_is_not_read_and_is_replaced(
        self, fake_cli, tmp_path, mode
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cache" / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        cache_path.chmod(mode)
        assert cvc.VersionCache(str(cache_path)).refused == (
            "the cache file is group- or world-writable"
        )
        assert _probe(provider, cache_path)["source"] == "probe" and _runs(log) == 2
        # The directory is trusted, so the write puts a private file in its place.
        assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
        assert _probe(provider, cache_path)["source"] == "cache"

    @posix_only
    def test_a_symlinked_file_is_not_read_and_its_target_is_not_written(
        self, fake_cli, tmp_path
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cache" / "cli-versions.json"
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        planted = tmp_path / "planted.json"
        os.replace(cache_path, planted)
        cache_path.symlink_to(planted)
        before = planted.read_bytes()
        assert cvc.VersionCache(str(cache_path)).refused == "the cache file is a symlink"
        assert _probe(provider, cache_path)["source"] == "probe" and _runs(log) == 2
        assert not cache_path.is_symlink() and planted.read_bytes() == before

    @posix_only
    def test_a_symlinked_directory_is_neither_read_nor_written(self, fake_cli, tmp_path):
        _, log = fake_cli
        real_dir = tmp_path / "elsewhere"
        provider = FakeProvider("fakecli")
        _probe(provider, real_dir / cvc.FILE_NAME)
        link_dir = tmp_path / "cache"
        link_dir.symlink_to(real_dir, target_is_directory=True)
        before = (real_dir / cvc.FILE_NAME).read_bytes()
        cache = cvc.VersionCache(str(link_dir / cvc.FILE_NAME))
        assert cache.refused == "the cache directory is a symlink"
        row = doctor._probe_version(provider, "fakecli", 10.0, cache)
        assert row["source"] == "probe" and _runs(log) == 2
        assert cache.save() is False
        assert os.listdir(real_dir) == [cvc.FILE_NAME]
        assert (real_dir / cvc.FILE_NAME).read_bytes() == before

    @posix_only
    @pytest.mark.parametrize("mode", [0o770, 0o730, 0o707, 0o777])
    def test_a_group_or_world_writable_directory_is_neither_read_nor_written(
        self, fake_cli, tmp_path, mode
    ):
        _, log = fake_cli
        cache_dir = tmp_path / "cache"
        cache_path = cache_dir / cvc.FILE_NAME
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        before = cache_path.read_bytes()
        cache_dir.chmod(mode)
        try:
            # --fresh reads nothing, and still writes nothing into it.
            for read in (True, False):
                cache = cvc.VersionCache(str(cache_path), read=read)
                row = doctor._probe_version(provider, "fakecli", 10.0, cache)
                assert row["source"] == "probe"
                assert cache.save() is False
                assert cache.refused == "the cache directory is group- or world-writable"
            assert _runs(log) == 3
            assert os.listdir(cache_dir) == [cvc.FILE_NAME]
            assert cache_path.read_bytes() == before
        finally:
            cache_dir.chmod(0o700)

    @posix_only
    def test_a_directory_owned_by_another_user_is_neither_read_nor_written(
        self, fake_cli, tmp_path, monkeypatch
    ):
        _, log = fake_cli
        cache_path = tmp_path / "cache" / cvc.FILE_NAME
        provider = FakeProvider("fakecli")
        _probe(provider, cache_path)
        before = cache_path.read_bytes()
        mine = os.getuid()
        monkeypatch.setattr(cvc, "_my_uid", lambda: mine + 1)
        cache = cvc.VersionCache(str(cache_path))
        assert cache.refused == f"the cache directory is owned by uid {mine}, not this user"
        row = doctor._probe_version(provider, "fakecli", 10.0, cache)
        assert row["source"] == "probe" and _runs(log) == 2
        assert cache.save() is False and cache_path.read_bytes() == before

    @posix_only
    def test_a_file_owned_by_another_user_is_refused(self, tmp_path):
        st = os.stat(tmp_path)
        foreign = os.stat_result((
            stat.S_IFREG | 0o600, st.st_ino, st.st_dev, 1, os.getuid() + 1, st.st_gid,
            10, 0, 0, 0,
        ))
        assert cvc._refusal(foreign, "file", stat.S_ISREG) == (
            f"the cache file is owned by uid {os.getuid() + 1}, not this user"
        )

    @posix_only
    def test_a_fifo_is_refused_without_waiting_for_a_writer(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(mode=0o700)
        fifo = cache_dir / cvc.FILE_NAME
        os.mkfifo(fifo, 0o600)
        made = []
        worker = threading.Thread(target=lambda: made.append(cvc.VersionCache(str(fifo))),
                                  daemon=True)
        worker.start()
        worker.join(5)
        if worker.is_alive():
            # Release the blocked reader before failing.
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            worker.join(5)
            pytest.fail("loading the cache waited on a FIFO")
        assert made[0].refused == "the cache file is not a regular file"


# ---------------------------------------------------------------------------
# Concurrent sessions
# ---------------------------------------------------------------------------

_WRITER = r"""
import sys
sys.path.insert(0, {skill!r})
from utils import cli_version_cache as cvc
path, ident, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
for n in range(rounds):
    cache = cvc.VersionCache(path)
    cache.record(f"{{ident}}-{{n % 3}}", {{"argv": [ident], "files": [], "parser": None}}, f"1.{{n}}.0")
    assert cache.save() is True
"""


@posix_only
class TestConcurrentWriters:

    def test_parallel_sessions_never_tear_the_file(self, tmp_path):
        cache_path = tmp_path / "cache" / "cli-versions.json"
        script = tmp_path / "writer.py"
        script.write_text(_WRITER.format(skill=str(SKILL_ROOT)), encoding="utf-8")
        writers = [
            subprocess.Popen(
                [sys.executable, str(script), str(cache_path), f"w{i}", "60"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for i in range(6)
        ]
        reads = 0
        while any(w.poll() is None for w in writers):
            if cache_path.exists():
                text = cache_path.read_text(encoding="utf-8")
                data = json.loads(text)  # a torn file raises here
                assert data["schema"] == cvc.SCHEMA
                reads += 1
        for writer in writers:
            out, err = writer.communicate()
            assert writer.returncode == 0, err
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["schema"] == cvc.SCHEMA
        # Each writer's own last round survives at least in its own write;
        # merging keeps most entries, and a lost one costs only a probe.
        assert data["entries"] and all(
            isinstance(e, dict) and e["version"].startswith("1.") for e in data["entries"].values()
        )
        assert sorted(os.listdir(cache_path.parent)) == ["cli-versions.json"], "temp file left"
        assert reads > 0

    def test_entries_written_by_another_session_are_merged_not_dropped(self, tmp_path):
        cache_path = tmp_path / "cli-versions.json"
        key = {"argv": ["a"], "files": [], "parser": None}
        mine = cvc.VersionCache(str(cache_path))
        theirs = cvc.VersionCache(str(cache_path))
        theirs.record("other", key, "9.9.9")
        assert theirs.save()
        mine.record("mine", key, "1.0.0")
        assert mine.save()
        entries = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]
        assert set(entries) == {"mine", "other"}


# ---------------------------------------------------------------------------
# Doctor end to end: versions cached, auth and sandbox never
# ---------------------------------------------------------------------------

@pytest.fixture
def two_cli_clinic(tmp_path, monkeypatch):
    """`grok` and `goose` as logging scripts, and a config naming only them
    (plus one name with no adapter)."""
    if os.name == "nt":
        pytest.skip("the fake CLIs are `#!/bin/sh` scripts on a replaced PATH")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "runs.log"
    _cli(bin_dir, "grok", "0.2.93", log)
    _cli(bin_dir, "goose", "1.57.0", log)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    )
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        registry, "load_config",
        lambda **kwargs: {"providers": {"grok": {}, "goose": {}, "not-a-cli": {}}},
    )
    monkeypatch.setattr(doctor, "_bwrap_self_test", lambda timeout: ("ok", ""))
    executor.reset_version_cache()
    yield log
    executor.reset_version_cache()


class TestDoctorUsesTheCache:

    def test_the_second_report_spawns_no_version_probe(self, two_cli_clinic):
        log = two_cli_clinic
        first = doctor.collect(budget_s=10)
        assert _runs(log) == 2
        assert first["version_cache"]["hits"] == 0 and first["version_cache"]["written"]
        second = doctor.collect(budget_s=10)
        assert _runs(log) == 2
        assert second["version_cache"]["hits"] == 2
        assert not second["version_cache"]["written"]
        for name in ("grok", "goose"):
            row = next(r for r in second["providers"] if r["provider"] == name)
            assert row["version"]["source"] == "cache"
        assert doctor.collect(budget_s=10, fresh=True)["version_cache"]["hits"] == 0
        assert _runs(log) == 4

    def test_auth_and_sandbox_probes_run_every_time(self, two_cli_clinic, monkeypatch):
        calls = []
        real = doctor._run

        def recording(argv, **kwargs):
            calls.append(list(argv))
            return real(argv, **kwargs)

        monkeypatch.setattr(doctor, "_run", recording)
        auths, sandboxes = [], []
        monkeypatch.setattr(
            doctor, "_probe_auth",
            lambda provider, *a: auths.append(provider) or {"state": "ok", "detail": "", "ok": True},
        )
        real_sandbox = doctor._probe_sandbox
        monkeypatch.setattr(
            doctor, "_probe_sandbox",
            lambda provider, *a: sandboxes.append(provider) or real_sandbox(provider, *a),
        )
        doctor.collect(budget_s=10)
        doctor.collect(budget_s=10)
        assert len(auths) == 4 and len(sandboxes) == 4
        version_calls = [c for c in calls if c[-1:] == ["--version"]]
        assert len(version_calls) == 2

    def test_the_cache_holds_no_auth_or_sandbox_answer(self, two_cli_clinic):
        doctor.collect(budget_s=10)
        text = Path(cvc.cache_file()).read_text(encoding="utf-8")
        data = json.loads(text)
        assert set(data["entries"]) == {"grok", "goose"}
        for entry in data["entries"].values():
            assert set(entry) == {"key", "version", "probed_at"}
            assert entry["key"]["argv"][-1] == "--version"

    def test_a_malformed_entry_never_stops_the_report(self, two_cli_clinic):
        doctor.collect(budget_s=10)
        cache_path = Path(cvc.cache_file())
        _plant(cache_path, "grok", probed_at=10 ** 400)
        _plant(cache_path, "goose", probed_at=float("nan"))
        report = doctor.collect(budget_s=10)
        assert report["version_cache"]["hits"] == 0
        for name in ("grok", "goose"):
            row = next(r for r in report["providers"] if r["provider"] == name)
            assert row["version"]["source"] == "probe"

    def test_a_refused_cache_is_named_in_json_only(self, two_cli_clinic, capsys, monkeypatch):
        monkeypatch.setattr(
            registry, "load_config", lambda **kwargs: {"providers": {"grok": {}, "goose": {}}},
        )
        doctor.collect(budget_s=10)
        assert doctor.collect(budget_s=10)["version_cache"]["refused"] == ""
        os.chmod(cvc.cache_dir(), 0o777)
        try:
            assert doctor.main(["--warn-only", "--budget", "10"]) == 0
            assert capsys.readouterr().out.splitlines()[0].endswith("; no problems")
            assert doctor.main(["--budget", "10"]) == 0
            out = capsys.readouterr().out
            assert "cache" not in out and "writable" not in out
            assert doctor.main(["--json", "--budget", "10"]) == 0
            cache = json.loads(capsys.readouterr().out)["version_cache"]
            assert cache["refused"] == "the cache directory is group- or world-writable"
            assert (cache["hits"], cache["written"]) == (0, False)
        finally:
            os.chmod(cvc.cache_dir(), 0o700)

    def test_the_full_table_says_where_its_versions_came_from(self, two_cli_clinic, capsys):
        assert doctor.main(["--budget", "10"]) == 0
        assert "read from doctor's cache" not in capsys.readouterr().out
        assert doctor.main(["--budget", "10"]) == 0
        out = capsys.readouterr().out
        assert "  versions: 2 read from doctor's cache" in out and "`--fresh`" in out
        assert doctor.main(["--budget", "10", "--fresh"]) == 0
        assert "read from doctor's cache" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The executor never reads it
# ---------------------------------------------------------------------------

class TestTheExecutorNeverReadsTheCache:

    def test_no_module_but_doctor_imports_the_cache(self):
        importers = []
        for path in python_files(SKILL_ROOT):
            if {"tests", ".venv", "__pycache__"} & set(path.relative_to(SKILL_ROOT).parts):
                continue
            tree = source_tree(path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [f"{node.module or ''}.{a.name}" for a in node.names]
                    names += [a.name for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                if any(n.split(".")[-1] == "cli_version_cache" for n in names):
                    importers.append(str(path.relative_to(SKILL_ROOT)))
        assert sorted(set(importers)) == [os.path.join("utils", "doctor.py")]

    def test_the_runner_version_check_reads_the_cli_live(self, two_cli_clinic, monkeypatch):
        """Seed the cache with a version doctor would trust for this exact
        binary, then show the executor's `min_version` read ignores it."""
        log = two_cli_clinic
        doctor.collect(budget_s=10)
        data = json.loads(Path(cvc.cache_file()).read_text(encoding="utf-8"))
        data["entries"]["grok"]["version"] = "99.0.0"
        Path(cvc.cache_file()).write_text(json.dumps(data), encoding="utf-8")
        hits = doctor.collect(budget_s=10)
        grok = next(r for r in hits["providers"] if r["provider"] == "grok")
        assert grok["version"]["version"] == "99.0.0"

        def forbidden(*args, **kwargs):
            raise AssertionError("the executor read doctor's version cache")

        monkeypatch.setattr(cvc.VersionCache, "__init__", forbidden)
        monkeypatch.setattr(cvc.VersionCache, "lookup", forbidden)
        monkeypatch.setattr(cvc, "fingerprint", forbidden)
        before = _runs(log)
        executor.reset_version_cache()
        assert executor._detect_version("grok", key="grok", spec="grok", profile="generate") == "0.2.93"
        assert _runs(log) == before + 1


# ---------------------------------------------------------------------------
# --warn-only: one line per problem
# ---------------------------------------------------------------------------

class TestProblemLines:

    def test_a_healthy_machine_is_one_line(self, two_cli_clinic, capsys, monkeypatch):
        monkeypatch.setattr(
            registry, "load_config", lambda **kwargs: {"providers": {"grok": {}, "goose": {}}},
        )
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("multi-llm doctor: 2 of 2 configured provider(s) present")
        assert lines[0].endswith("; no problems")

    def test_a_configured_name_without_an_adapter_is_a_problem(self, two_cli_clinic, capsys):
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "1 problem(s)" in lines[0]
        assert lines[1:] == [
            "WARN: not-a-cli: no built-in adapter (config name is ignored at runtime)"
        ]

    def test_slow_probes_share_one_note_are_not_problems_and_make_checking_incomplete(
        self, two_cli_clinic, capsys, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            registry, "load_config", lambda **kwargs: {"providers": {"grok": {}, "goose": {}}},
        )
        for name in ("grok", "goose"):
            script = tmp_path / "bin" / name
            script.write_text('#!/bin/sh\nsleep 5\necho "x 1.0.0"\n', encoding="utf-8")
        assert doctor.main(["--warn-only", "--budget", "0.3"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].endswith(
            "; no problems found, but checking was incomplete (NOTE lines below)"
        )
        assert lines[1:] == [
            "NOTE: no answer within the 0.3 s budget, so not checked: grok version, "
            "goose version (`multi_llm.py doctor --budget 10` waits longer)"
        ]

    def test_a_failure_line_per_problem_and_the_json_stays_complete(
        self, two_cli_clinic, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            doctor, "_probe_auth",
            lambda provider, *a: {"state": "not-logged-in", "ok": False,
                                  "detail": f"run `{provider.name} login`"},
        )
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "3 problem(s)" in lines[0]
        assert lines[1:] == [
            "WARN: not-a-cli: no built-in adapter (config name is ignored at runtime)",
            "WARN: grok: auth — not logged in (run `grok login`)",
            "WARN: goose: auth — not logged in (run `goose login`)",
        ]
        assert doctor.main(["--warn-only", "--json", "--budget", "10"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert len(report["failures"]) == 2 and report["exit"] == 0
        assert {"version", "auth", "sandbox"} <= set(
            next(r for r in report["providers"] if r["provider"] == "grok")
        )
        assert set(report["version_cache"]) == {"path", "read", "hits", "written", "refused"}

    def test_each_inconclusive_check_gets_a_note_and_the_summary_says_incomplete(
        self, two_cli_clinic, capsys, monkeypatch, tmp_path
    ):
        """Codex review of U28, point 3: an unread version, an auth answer no
        verdict matched and a bwrap self-test with no result each get a line;
        probes the adapter never declared stay silent."""
        monkeypatch.setattr(
            registry, "load_config", lambda **kwargs: {"providers": {"grok": {}, "goose": {}}},
        )
        bin_dir = tmp_path / "bin"
        (bin_dir / "goose").write_text('#!/bin/sh\necho "no digits here"\n', encoding="utf-8")
        _cli(bin_dir, "grokstatus", "", tmp_path / "status.log").write_text(
            "#!/bin/sh\nexit 2\n", encoding="utf-8"
        )
        grok = type(registry.get_provider("grok"))
        monkeypatch.setattr(grok, "auth_check", ("grokstatus",), raising=False)
        monkeypatch.setattr(grok, "sandbox_mechanism", "bubblewrap", raising=False)
        monkeypatch.setattr(
            doctor, "_bwrap_self_test",
            lambda timeout: ("unknown", "the bwrap self-test could not be read in time"),
        )
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].endswith(
            "; no problems found, but checking was incomplete (NOTE lines below)"
        )
        assert lines[1:] == [
            "NOTE: grok auth inconclusive: `grokstatus` exited 2 without a recognised verdict",
            "NOTE: grok sandbox inconclusive: the bwrap self-test could not be read in time",
            "NOTE: goose version inconclusive: `goose --version` printed nothing this "
            "adapter reads as a version",
        ]

    def test_a_problem_beside_an_inconclusive_check_says_both(
        self, two_cli_clinic, capsys, monkeypatch, tmp_path
    ):
        (tmp_path / "bin" / "goose").write_text('#!/bin/sh\necho "no digits"\n', encoding="utf-8")
        assert doctor.main(["--warn-only", "--budget", "10"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "1 problem(s), one per line below, and checking was incomplete" in lines[0]
        assert lines[1].startswith("WARN: not-a-cli:")
        assert lines[2].startswith("NOTE: goose version inconclusive:")


# ---------------------------------------------------------------------------
# The suite never reaches the developer's cache, in children either
# ---------------------------------------------------------------------------

@posix_only
class TestSuiteIsolation:

    def test_a_subprocess_doctor_writes_under_the_redirect(self, tmp_path):
        """`tests/conftest.py` redirects `XDG_CACHE_HOME`, which a child
        interpreter inherits, and pins uv's own cache where it was."""
        redirect = Path(os.environ["XDG_CACHE_HOME"])
        assert tmp_path.parent in redirect.parents
        assert os.environ["UV_CACHE_DIR"]
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "runs.log"
        _cli(bin_dir, "grok", "0.2.93", log)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(
            [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
        )
        env["MULTI_LLM_TEST_MODE"] = "1"
        where = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from utils import cli_version_cache as c; print(c.cache_file())",
             str(SKILL_ROOT)],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
        )
        expected = redirect / "multi-llm" / cvc.FILE_NAME
        assert where.stdout.strip() == str(expected)
        done = subprocess.run(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"), "doctor", "--warn-only",
             "--budget", "10"],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert _runs(log) >= 1
        entries = json.loads(expected.read_text(encoding="utf-8"))["entries"]
        assert entries["grok"]["version"] == "0.2.93"
        assert entries["grok"]["key"]["files"][0]["path"] == str((bin_dir / "grok").resolve())
