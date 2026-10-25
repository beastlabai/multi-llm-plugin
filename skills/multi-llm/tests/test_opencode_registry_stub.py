"""The localhost npm registry stub, ``packaging/opencode/registry_stub.py``, offline.

**What is graded.** That the stub answers in the shapes npm itself reads to
resolve and verify a version: the packument at ``/<name>`` (``dist.tarball``
pointing back at the stub, ``dist.integrity``/``shasum`` computed from the
file, ``dist-tags.latest``), the version manifest, the tarball's exact bytes,
and the registry's own ``404 {"error":"Not found"}`` for anything else, which
npm reports as ``E404``. Then the real clients: ``npm view`` and ``npm pack
<spec>`` resolve the package through it, when npm is installed (a named skip
otherwise). Then the two CLIs package E and the probe run: ``serve`` (binds
``127.0.0.1:0``, publishes the port, stops cleanly on SIGTERM) and ``relay``.

**What is not.** OpenCode's installer resolving through it is measured by the
registry-install probe's rehearsals (``registry_check.py``), not here: it needs
the pinned binary.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STUB_PATH = REPO / "packaging" / "opencode" / "registry_stub.py"
NAME = "opencode-multi-llm"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stub = _load("opencode_registry_stub_test", STUB_PATH)

NPM = shutil.which("npm")
needs_npm = pytest.mark.skipif(NPM is None, reason="npm is not installed: the real-client rows "
                                                   "need npm on PATH")


def _tarball(root: Path, version: str = "0.1.0", name: str = NAME, extra: bytes = b"") -> Path:
    """A packed tarball the way npm writes one: files only, under ``package/``."""
    path = root / f"{name.replace('/', '-').lstrip('@')}-{version}.tgz"
    manifest = json.dumps({"name": name, "version": version, "main": "./dist/index.mjs",
                           "license": "MIT"}).encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        for member, data in (("package/package.json", manifest),
                             ("package/dist/index.mjs", b"export default {}\n" + extra)):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return path


def _sri(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


@pytest.fixture()
def tgz(tmp_path):
    return _tarball(tmp_path)


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# --------------------------------------------------------------------------- #
# The shapes, without a socket
# --------------------------------------------------------------------------- #

class TestTheShapes:
    def test_the_packument_points_back_at_the_stub_and_hashes_the_file(self, tgz):
        with stub.RegistryStub(tgz) as server:
            status, kind, body = server.route("GET", f"/{NAME}")
            document = json.loads(body)
            url = server.url
        data = tgz.read_bytes()
        assert (status, kind) == (200, "application/json")
        assert document["name"] == NAME and document["dist-tags"] == {"latest": "0.1.0"}
        dist = document["versions"]["0.1.0"]["dist"]
        assert dist["tarball"] == f"{url}/{NAME}/-/{NAME}-0.1.0.tgz"
        assert dist["integrity"] == _sri(data)
        assert dist["shasum"] == hashlib.sha1(data).hexdigest()
        # The version document is the tarball's own manifest, plus what npm adds.
        assert document["versions"]["0.1.0"]["main"] == "./dist/index.mjs"
        assert document["versions"]["0.1.0"]["_id"] == f"{NAME}@0.1.0"

    def test_its_integrity_is_the_builders_spelling(self, tgz):
        """``build.integrity_of`` is the publish path's formula; the stub's must agree."""
        sys.path.insert(0, str(REPO / "packaging" / "opencode"))
        build = _load("opencode_build_for_stub_test", REPO / "packaging" / "opencode" / "build.py")
        assert stub.integrity_of(tgz.read_bytes()) == build.integrity_of(tgz.read_bytes())

    def test_the_tarball_and_the_version_manifest(self, tgz):
        with stub.RegistryStub(tgz) as server:
            status, kind, body = server.route("GET", f"/{NAME}/-/{NAME}-0.1.0.tgz")
            assert (status, kind, body) == (200, "application/octet-stream", tgz.read_bytes())
            for path in (f"/{NAME}/0.1.0", f"/{NAME}/latest"):
                status, _kind, body = server.route("GET", path)
                assert status == 200 and json.loads(body)["version"] == "0.1.0"
            assert server.route("GET", "/-/ping")[0] == 200

    @pytest.mark.parametrize("method, path", [
        ("GET", "/@opencode-ai%2fplugin"),
        ("GET", "/another-package"),
        ("GET", f"/{NAME}/9.9.9"),
        ("GET", f"/{NAME}/-/{NAME}-9.9.9.tgz"),
        ("POST", "/-/npm/v1/security/advisories/bulk"),
        ("PUT", f"/{NAME}"),
    ])
    def test_anything_else_is_the_registrys_404(self, tgz, method, path):
        with stub.RegistryStub(tgz) as server:
            assert server.route(method, path) == (404, "application/json", stub.NOT_FOUND)
        assert json.loads(stub.NOT_FOUND) == {"error": "Not found"}

    def test_a_scoped_name_is_matched_url_decoded(self, tmp_path):
        scoped = _tarball(tmp_path, name="@beast/plugin")
        with stub.RegistryStub(scoped) as server:
            status, _kind, body = server.route("GET", "/@beast%2fplugin")
            document = json.loads(body)
            assert status == 200
            tarball = document["versions"]["0.1.0"]["dist"]["tarball"]
            assert tarball.endswith("/@beast/plugin/-/plugin-0.1.0.tgz")
            assert server.route("GET", tarball[len(server.url):])[0] == 200

    def test_several_versions_of_one_package(self, tmp_path):
        for sub in ("a", "b"):
            (tmp_path / sub).mkdir()
        old = _tarball(tmp_path / "a", version="0.1.1")
        new = _tarball(tmp_path / "b", version="0.1.2")
        with stub.RegistryStub([new, old]) as server:
            document = json.loads(server.route("GET", f"/{NAME}")[2])
            assert document["dist-tags"] == {"latest": "0.1.2"}
            assert sorted(document["versions"]) == ["0.1.1", "0.1.2"]
            assert server.route("GET", f"/{NAME}/-/{NAME}-0.1.1.tgz")[2] == old.read_bytes()

    def test_refusals(self, tmp_path):
        one = _tarball(tmp_path)
        (tmp_path / "other").mkdir()
        other = _tarball(tmp_path / "other", name="other-package")
        with pytest.raises(stub.StubError, match="one package per stub"):
            stub.RegistryStub([one, other])
        with pytest.raises(stub.StubError, match="given twice"):
            stub.RegistryStub([one, one])
        bare = tmp_path / "bare.tgz"
        with tarfile.open(bare, "w:gz") as archive:
            info = tarfile.TarInfo("package/README.md")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"hi"))
        with pytest.raises(stub.StubError, match="no package/package.json"):
            stub.RegistryStub(bare)


# --------------------------------------------------------------------------- #
# Over a socket, and through the real clients
# --------------------------------------------------------------------------- #

class TestOnLoopback:
    def test_it_binds_loopback_and_serves(self, tgz, tmp_path):
        log = tmp_path / "requests.jsonl"
        with stub.RegistryStub(tgz, log=log) as server:
            assert server.url.startswith("http://127.0.0.1:")
            status, body = _get(f"{server.url}/{NAME}")
            assert status == 200 and json.loads(body)["name"] == NAME
            status, body = _get(f"{server.url}/{NAME}/-/{NAME}-0.1.0.tgz")
            assert status == 200 and body == tgz.read_bytes()
            assert _get(f"{server.url}/nope")[0] == 404
            assert server.fetched("tarball") == [f"/{NAME}/-/{NAME}-0.1.0.tgz"]
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert [(e["path"], e["status"]) for e in lines] == [
            (f"/{NAME}", 200), (f"/{NAME}/-/{NAME}-0.1.0.tgz", 200), ("/nope", 404)]


def _npm(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    rc = cwd / "npmrc-empty"
    rc.write_text("", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith("npm_config_")}
    return subprocess.run([NPM, *args, f"--cache={cwd / 'npm-cache'}", f"--userconfig={rc}",
                           "--no-update-notifier", "--fetch-retries=0"],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
                          stdin=subprocess.DEVNULL)


@needs_npm
class TestTheRealNpmClient:
    def test_npm_view_resolves_the_spec(self, tgz, tmp_path):
        with stub.RegistryStub(tgz) as server:
            url = server.url
            done = _npm("view", f"{NAME}@0.1.0", "version", "dist", "--json",
                        f"--registry={url}", cwd=tmp_path)
        assert done.returncode == 0, done.stderr
        view = json.loads(done.stdout)
        assert view["version"] == "0.1.0"
        assert view["dist"]["integrity"] == _sri(tgz.read_bytes())
        assert view["dist"]["tarball"].startswith(url + "/")

    def test_npm_reports_E404_for_a_version_it_does_not_serve(self, tgz, tmp_path):
        with stub.RegistryStub(tgz) as server:
            missing_version = _npm("view", f"{NAME}@9.9.9", "version", "--json",
                                   f"--registry={server.url}", cwd=tmp_path)
            missing_package = _npm("view", "some-other-package", "version", "--json",
                                   f"--registry={server.url}", cwd=tmp_path)
        for done in (missing_version, missing_package):
            assert done.returncode != 0
            assert json.loads(done.stdout)["error"]["code"] == "E404"

    def test_npm_pack_downloads_the_exact_bytes(self, tgz, tmp_path):
        dest = tmp_path / "dest"
        dest.mkdir()
        with stub.RegistryStub(tgz) as server:
            done = _npm("pack", f"{NAME}@0.1.0", "--json", f"--pack-destination={dest}",
                        f"--registry={server.url}", cwd=tmp_path)
            downloads = server.fetched("tarball")
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout)[0]["integrity"] == _sri(tgz.read_bytes())
        assert (dest / f"{NAME}-0.1.0.tgz").read_bytes() == tgz.read_bytes()
        assert downloads == [f"/{NAME}/-/{NAME}-0.1.0.tgz"]


# --------------------------------------------------------------------------- #
# The CLIs: `serve` (package E runs it in the background) and `relay`
# --------------------------------------------------------------------------- #

def _wait_for(path: Path, budget_s: float = 15.0) -> str:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return path.read_text(encoding="utf-8").strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM delivery: the CLI's stop is POSIX-only here")
class TestTheCommandLines:
    def test_serve_publishes_its_port_and_stops_on_sigterm(self, tgz, tmp_path):
        port_file = tmp_path / "port"
        proc = subprocess.Popen([sys.executable, str(STUB_PATH), "serve", "--tarball", str(tgz),
                                 "--port-file", str(port_file)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        try:
            port = int(_wait_for(port_file))
            status, body = _get(f"http://127.0.0.1:{port}/{NAME}")
            assert status == 200 and json.loads(body)["dist-tags"]["latest"] == "0.1.0"
        finally:
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=15)
        assert proc.returncode == 0, err
        assert f"{NAME}@0.1.0 at http://127.0.0.1:{port}" in out.decode()
        assert not port_file.exists()

    def test_serve_refuses_a_tarball_it_cannot_read(self, tmp_path):
        done = subprocess.run([sys.executable, str(STUB_PATH), "serve", "--tarball",
                               str(tmp_path / "absent.tgz"), "--port-file", str(tmp_path / "p")],
                              capture_output=True, text=True, timeout=30)
        assert done.returncode == 1 and "REFUSING" in done.stderr

    def test_relay_forwards_a_loopback_port_to_the_target(self, tgz, tmp_path):
        port_file = tmp_path / "relay.port"
        with stub.RegistryStub(tgz) as server:
            proc = subprocess.Popen([sys.executable, str(STUB_PATH), "relay", "--listen-port", "0",
                                     "--target", f"127.0.0.1:{server.port}",
                                     "--port-file", str(port_file)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                relayed = int(_wait_for(port_file))
                assert relayed != server.port
                status, body = _get(f"http://127.0.0.1:{relayed}/{NAME}/-/{NAME}-0.1.0.tgz")
                assert status == 200 and body == tgz.read_bytes()
            finally:
                proc.send_signal(signal.SIGTERM)
                proc.communicate(timeout=15)
        assert proc.returncode == 0
