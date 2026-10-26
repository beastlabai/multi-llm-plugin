#!/usr/bin/env python3
"""A localhost npm registry that serves one packed tarball, and nothing else.

Run it as::

    python3 packaging/opencode/registry_stub.py serve \\
        --tarball packaging/opencode/opencode-multi-llm-<version>.tgz \\
        --port-file <file> [--log <requests.jsonl>]

It binds ``127.0.0.1:0``, writes the port it got to ``--port-file`` (no
pick-then-bind race), and stays in the foreground until SIGTERM or SIGINT. The
registry URL is then ``http://127.0.0.1:<port>``.

**What it is for.** Rehearsing the registry-install probe
(``registry_check.py``) before anything is published: the gated candidate
tarball, served under its own name and version, so that real ``npm view``, real
``npm pack <spec>`` and OpenCode's bundled installer (npm's own config loader
and Arborist, measured on 1.19.4) all resolve the spec through it exactly as
they would through the registry. A record made against it says
``rehearsal: true`` and is never release evidence; ``registry_check.py``
accepts this URL only as ``--rehearsal-registry``, only on loopback.

**What it serves**, read from the tarball and never from a checkout:

* ``GET /<name>`` — the packument: ``name``, ``dist-tags.latest``, one entry
  under ``versions`` (the tarball's own ``package/package.json`` plus ``_id``
  and ``dist``) and ``time``. ``dist.tarball`` points back at this server;
  ``dist.integrity`` (``sha512-<base64>``, the spelling npm and
  ``build.integrity_of`` use) and ``dist.shasum`` (sha1, hex) are computed from
  the file's bytes when the server starts. Both the full document and npm's
  abbreviated ``install-v1`` form are this one document: the abbreviated form
  is a subset of the keys npm reads, so a client that asked for it is served
  correctly by a superset.
* ``GET /<name>/<version>`` and ``GET /<name>/latest`` — that version's manifest.
* ``GET /<name>/-/<basename>-<version>.tgz`` — the tarball's bytes.
* ``GET /-/ping`` — ``{}``, for ``npm ping``.

Several ``--tarball`` arguments serve several versions of ONE package (a later
release's upgrade rehearsal needs the previous pin too); ``dist-tags.latest`` is
the highest version.

Anything else, any other package or version included, is answered exactly as
the registry answers a missing package: ``404`` with ``{"error":"Not found"}``,
which npm reports as code ``E404``. Scoped names arrive URL-encoded
(``/@scope%2fname``) and are decoded before matching.

**The relay** (``relay --listen-port P --target HOST:P``) forwards every TCP
connection made to ``127.0.0.1:P`` to ``HOST:P``. The registry-install probe's
container driver runs it inside a networked container on Docker Desktop, where
``--network host`` is the VM's network rather than the host's: the container
reaches the host's loopback stub as ``host.docker.internal`` (``host-gateway``),
and the relay keeps the URL the packument names, ``http://127.0.0.1:P``, valid
inside the container too.

**Why the packument is generated rather than copied.** The registry's own
documents carry fields this stub cannot vouch for (``_npmUser``, signatures,
provenance attestations); the probe reads only what npm itself reads to resolve
and verify a version, and a generated document makes every one of those fields
a computation over the file the stub was handed.

Stdlib only, importable without side effects: the probe and the suite use
:class:`RegistryStub` in-process, and the workflow's ``path: bootstrap``
rehearsal runs ``serve`` in the background.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import socket
import socketserver
import sys
import tarfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

HOST = "127.0.0.1"
#: npm's prefix for every member of a packed tarball.
TARBALL_PREFIX = "package/"
#: The body the registry sends for a package or version it does not have; npm
#: turns the 404 into ``E404`` whatever the body says, but this is the real one.
NOT_FOUND = b'{"error":"Not found"}'


class StubError(RuntimeError):
    """The tarball cannot be served: unreadable, or no ``package/package.json``."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def integrity_of(data: bytes) -> str:
    """npm's ``dist.integrity`` spelling: ``sha512-<base64>``.

    The same formula as ``build.integrity_of``; spelled again here only because
    this module is stdlib-only and importing ``build`` imports the skill tree.
    The suite asserts the two agree on the same bytes.
    """
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def read_manifest(tarball: Path) -> Dict[str, Any]:
    """The tarball's own ``package/package.json``."""
    try:
        with tarfile.open(tarball, mode="r:gz") as archive:
            member = archive.getmember(TARBALL_PREFIX + "package.json")
            source = archive.extractfile(member)
            if source is None:
                raise StubError(f"{tarball}: {TARBALL_PREFIX}package.json has no content")
            with source:
                manifest = json.loads(source.read().decode("utf-8"))
    except KeyError as exc:
        raise StubError(f"{tarball}: no {TARBALL_PREFIX}package.json") from exc
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise StubError(f"{tarball}: cannot read its manifest ({exc})") from exc
    if not isinstance(manifest, dict) or not manifest.get("name") or not manifest.get("version"):
        raise StubError(f"{tarball}: its manifest names no package name and version")
    return manifest


def tarball_basename(name: str) -> str:
    """The file part npm's own tarball URLs use: the name without its scope."""
    return name.split("/", 1)[1] if name.startswith("@") and "/" in name else name


class Package:
    """One packed tarball and everything the stub says about it."""

    def __init__(self, tarball: Path):
        self.path = Path(tarball).absolute()
        try:
            self.data = self.path.read_bytes()
        except OSError as exc:
            raise StubError(f"{self.path}: cannot read the tarball ({exc})") from exc
        self.manifest = read_manifest(self.path)
        self.name: str = str(self.manifest["name"])
        self.version: str = str(self.manifest["version"])
        self.integrity = integrity_of(self.data)
        self.shasum = hashlib.sha1(self.data).hexdigest()
        self.published = _utc_now()

    @property
    def tarball_path(self) -> str:
        """The URL path the packument's ``dist.tarball`` names."""
        return f"/{self.name}/-/{tarball_basename(self.name)}-{self.version}.tgz"

    def version_document(self, base_url: str) -> Dict[str, Any]:
        document = dict(self.manifest)
        document["_id"] = f"{self.name}@{self.version}"
        document["dist"] = {
            "tarball": base_url + self.tarball_path,
            "integrity": self.integrity,
            "shasum": self.shasum,
        }
        return document

    def packument(self, base_url: str) -> Dict[str, Any]:
        return {
            "_id": self.name,
            "name": self.name,
            "dist-tags": {"latest": self.version},
            "versions": {self.version: self.version_document(base_url)},
            "time": {"created": self.published, "modified": self.published,
                     self.version: self.published},
            "modified": self.published,
        }


class _Loopback(ThreadingHTTPServer):
    """No reverse lookup of 127.0.0.1 at bind (``probe._Loopback``'s reason:
    it hung the macOS runner), and quiet about a client that hung up."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


def _version_key(version: str) -> Tuple:
    core, _, pre = version.partition("-")
    numbers = tuple(int(part) if part.isdigit() else 0 for part in core.split("."))
    return numbers + ((0, pre) if pre else (1, ""))


class RegistryStub:
    """One package's :class:`Package` versions on ``127.0.0.1:<port>``; a context manager.

    ``port=0`` (the default) lets the kernel pick. Every request is kept in
    :attr:`requests` as ``(method, path, status)``, and appended to ``log`` as
    one JSON line when a log path is given, so a probe can say afterwards which
    paths a client actually fetched from it.
    """

    def __init__(self, tarball: "Path | Sequence[Path]", *, port: int = 0,
                 log: Optional[Path] = None):
        tarballs = [tarball] if isinstance(tarball, (str, Path)) else list(tarball)
        if not tarballs:
            raise StubError("no tarball to serve")
        packages = [Package(t) for t in tarballs]
        names = sorted({p.name for p in packages})
        if len(names) != 1:
            raise StubError(f"one package per stub; the tarballs name {names}")
        self.versions: Dict[str, Package] = {}
        for package in packages:
            if package.version in self.versions:
                raise StubError(f"{package.name}@{package.version} is given twice")
            self.versions[package.version] = package
        #: The latest version: the one the packument's ``dist-tags.latest`` names.
        self.package = max(packages, key=lambda p: _version_key(p.version))
        self.requested_port = port
        self.log = Path(log) if log else None
        self.requests: List[Tuple[str, str, int]] = []
        self._lock = threading.Lock()
        self._server: Optional[_Loopback] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        assert self._server is not None, "the stub is not serving"
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        """The registry URL, without a trailing slash."""
        return f"http://{HOST}:{self.port}"

    def packument(self) -> Dict[str, Any]:
        latest = self.package
        document = latest.packument(self.url)
        for version, package in self.versions.items():
            document["versions"][version] = package.version_document(self.url)
            document["time"][version] = package.published
        return document

    def route(self, method: str, raw_path: str) -> Tuple[int, str, bytes]:
        """``(status, content type, body)`` for one request. Pure, apart from
        reading the bound port, so the suite can grade it without a socket."""
        path = unquote(urlsplit(raw_path).path)
        name = self.package.name
        if method not in ("GET", "HEAD"):
            return 404, "application/json", NOT_FOUND
        if path == "/-/ping":
            return 200, "application/json", b"{}"
        for package in self.versions.values():
            if path == package.tarball_path:
                return 200, "application/octet-stream", package.data
        if path == f"/{name}":
            return 200, "application/json", json.dumps(self.packument()).encode("utf-8")
        version = path[len(f"/{name}/"):] if path.startswith(f"/{name}/") else None
        if version == "latest":
            version = self.package.version
        if version in self.versions:
            body = json.dumps(self.versions[version].version_document(self.url)).encode("utf-8")
            return 200, "application/json", body
        return 404, "application/json", NOT_FOUND

    def _record(self, method: str, path: str, status: int) -> None:
        with self._lock:
            self.requests.append((method, path, status))
            if self.log is not None:
                entry = {"at": _utc_now(), "method": method, "path": path, "status": status}
                with open(self.log, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")

    def __enter__(self) -> "RegistryStub":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                status, content_type, body = stub.route(self.command, self.path)
                stub._record(self.command, self.path, status)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(body)
                except OSError:
                    pass  # the client hung up

            do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = _answer

            def log_message(self, *_args: Any) -> None:
                pass

        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
        self._server = _Loopback((HOST, self.requested_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def fetched(self, kind: str = "tarball") -> List[str]:
        """Paths served with 200: ``tarball`` downloads, or ``packument`` reads."""
        wanted = {p.tarball_path for p in self.versions.values()}
        with self._lock:
            served = [(m, p) for m, p, s in self.requests if s == 200]
        if kind == "tarball":
            return [p for _m, p in served if unquote(urlsplit(p).path) in wanted]
        return [p for _m, p in served if unquote(urlsplit(p).path) not in wanted]


def _write_port(port_file: Path, port: int) -> None:
    """Atomically, so a reader polling for the file never sees it half-written."""
    port_file = Path(port_file)
    port_file.parent.mkdir(parents=True, exist_ok=True)
    partial = port_file.with_name(port_file.name + ".partial")
    partial.write_text(f"{port}\n", encoding="utf-8")
    os.replace(partial, port_file)


def _until_signalled() -> threading.Event:
    stopping = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return stopping


def serve(tarball: "Path | Sequence[Path]", port_file: Path, *, log: Optional[Path] = None) -> int:
    """Serve until SIGTERM or SIGINT; 0 on a clean stop."""
    stopping = _until_signalled()
    with RegistryStub(tarball, log=log) as stub:
        _write_port(port_file, stub.port)
        for package in sorted(stub.versions.values(), key=lambda p: _version_key(p.version)):
            print(f"registry stub: {package.name}@{package.version} at {stub.url} "
                  f"(integrity {package.integrity})", flush=True)
        print(f"registry stub: port in {port_file}", flush=True)
        while not stopping.wait(0.5):
            pass
    try:
        Path(port_file).unlink()
    except OSError:
        pass
    return 0


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while True:
            chunk = source.recv(65536)
            if not chunk:
                break
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        for end in (sink, source):
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def relay(listen_port: int, target: str, port_file: Optional[Path] = None,
          connect_timeout_s: float = 10.0) -> int:
    """Forward ``127.0.0.1:<listen_port>`` to ``target`` (``host:port``) until signalled."""
    host, _, port_text = target.rpartition(":")
    if not host or not port_text.isdigit():
        raise StubError(f"--target {target!r} is not host:port")
    stopping = _until_signalled()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, int(listen_port)))
    listener.listen(64)
    listener.settimeout(0.5)
    if port_file is not None:
        _write_port(port_file, listener.getsockname()[1])
    print(f"registry stub relay: {HOST}:{listener.getsockname()[1]} -> {target}", flush=True)
    try:
        while not stopping.is_set():
            try:
                client, _address = listener.accept()
            except socket.timeout:
                continue
            try:
                upstream = socket.create_connection((host, int(port_text)), timeout=connect_timeout_s)
                upstream.settimeout(None)
            except OSError:
                client.close()
                continue
            client.settimeout(None)
            for source, sink in ((client, upstream), (upstream, client)):
                threading.Thread(target=_pipe, args=(source, sink), daemon=True).start()
    finally:
        listener.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="registry_stub.py",
        description="Serve one packed tarball as a localhost npm registry (rehearsals only).")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve", help="serve until signalled")
    run.add_argument("--tarball", required=True, type=Path, action="append",
                     help="the packed .tgz to serve (repeat for more versions of the same package)")
    run.add_argument("--port-file", required=True, type=Path,
                     help="where the bound port is written once the server listens")
    run.add_argument("--log", type=Path, default=None,
                     help="append one JSON line per request here")
    forward = commands.add_parser("relay", help="forward a loopback port to host:port")
    forward.add_argument("--listen-port", required=True, type=int)
    forward.add_argument("--target", required=True, help="host:port to forward to")
    forward.add_argument("--port-file", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "relay":
            return relay(args.listen_port, args.target, args.port_file)
        return serve(args.tarball, args.port_file, log=args.log)
    except (StubError, OSError) as exc:
        print(f"registry stub: REFUSING: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
