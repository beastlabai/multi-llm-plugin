"""The host lane's own ``$HOME``, seeded on the tier that had no wiring step (H2).

`plans/opencode-native-plugin.md`'s finding H2 records the symptom exactly — the
`--creds none` discovery tier runs the host against a barer ``$HOME`` than any
real machine — and names the wrong cause. `entrypoint.seed_homes()` was never the
function that could have filled it: it writes ``/work/homes/<identity>``, which
only `wrapper.seed_invocation_root` reads, and only for the provider subprocesses
a fan-out dispatches. The host process's ``$HOME`` is `Path.home()`
(``/home/e2e``, tmpfs per `posture.yaml`, therefore empty at row start), so adding
the host id to `identities` would have seeded a directory no host ever opens — a
fix that looks like one and measures nothing.

`entrypoint.seed_host_home` is the real one, and this module is where it is
pinned, offline: no daemon, no credential, no network. Three things are asserted
because each fails independently:

* the **copy rule** — the value-free templates are skipped and everything else
  lands, matched on the seed's own tokens rather than on a bare ``{{``;
* the **token list** — it is `wrapper.py`'s, read out of `wrapper.py`, so a token
  added to the provider lane's substitution map cannot silently start arriving in
  a host's ``$HOME`` as the literal string ``{{…}}``;
* the **call site** — over `main`'s AST, because the branch that calls it only
  executes inside a container and a deleted call would otherwise go quiet for a
  release, exactly as `tests/test_golden_container_bundle.py` argues for the
  golden gate.

**The filename is deliberately not `test_container_*`**, for the reason
`tests/test_host_model_pin.py` and `tests/test_stray_write_paths.py` both give:
that glob is what `tests/container/context_manifest.txt` ships into the image and
what the marker-gating check enumerates, and a plain-CI module answering to it
would be shipped for nothing and would fail that assertion.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

# tests/container/entrypoint.py is the Linux image's entrypoint and imports
# POSIX-only modules (fcntl, pty, termios) at load; on a runner without them
# there is nothing of it to test, and the module says so instead of erroring.
pytest.importorskip("fcntl", reason="tests/container/entrypoint.py runs only in the Linux image (POSIX-only imports)")
from .container import entrypoint

CONTAINER_DIR = Path(entrypoint.__file__).resolve().parent
HOME_SEED = CONTAINER_DIR / "home_seed"

#: The host ids the host lane runs. Retyped rather than read from
#: `host_matrix.yaml`, so this module cannot agree with an edit to the file it is
#: meant to guard.
HOSTS = ("claude-code", "codex", "opencode")


def _seed(tmp_path: Path, host: str, files: dict[str, str]) -> Path:
    """A fixture rig dir holding ``home_seed/<host>/`` with `files` in it."""
    rig = tmp_path / "rig"
    for rel, text in files.items():
        target = rig / "home_seed" / host / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return rig


class TestWhatItCopiesAndWhatItRefusesTo:
    def test_a_value_free_file_lands(self, tmp_path, monkeypatch):
        rig = _seed(tmp_path, "claude-code",
                    {".claude.json": '{"hasCompletedOnboarding": true}\n'})
        monkeypatch.setattr(entrypoint, "RIG_DIR", rig)
        home = tmp_path / "home"
        assert entrypoint.seed_host_home("claude-code", home) == []
        assert json.loads((home / ".claude.json").read_text(encoding="utf-8")) \
            == {"hasCompletedOnboarding": True}

    def test_a_templated_file_is_skipped_not_copied(self, tmp_path, monkeypatch):
        """The tier has no exec id, no bound base URL and no key to substitute.

        Copying the template anyway is not a smaller failure than skipping it: it
        hands the host a provider whose `baseURL` is the seven-character string
        `{{base_url}}`, which is precisely the shadowing `wire_stub_host` already
        refuses by name on the documented-invocation path.
        """
        rig = _seed(tmp_path, "opencode", {
            ".config/opencode/opencode.json":
                '{"provider": {"stub": {"options": '
                '{"baseURL": "{{base_url}}", "apiKey": "{{stub_session_key}}"}}}}',
            ".local/share/opencode/auth.json":
                '{"openrouter": {"type": "api", "key": "{{stub_session_key}}"}}',
            ".config/opencode/node_modules/pkg/index.js": "module.exports = 1;\n",
        })
        monkeypatch.setattr(entrypoint, "RIG_DIR", rig)
        home = tmp_path / "home"
        skipped = entrypoint.seed_host_home("opencode", home)
        assert skipped == [".config/opencode/opencode.json",
                           ".local/share/opencode/auth.json"]
        assert not (home / ".config" / "opencode" / "opencode.json").exists()
        assert not (home / ".local" / "share" / "opencode" / "auth.json").exists()
        # ...and the tree the skip exists to protect still arrived whole.
        assert (home / ".config" / "opencode" / "node_modules" / "pkg"
                / "index.js").is_file()

    def test_a_brace_that_is_not_a_seed_token_is_not_a_skip(self, tmp_path,
                                                            monkeypatch):
        """The node_modules regression, in one file.

        MEASURED in `mllm-e2e:local`: six files inside opencode's own baked
        ``node_modules`` carry `{{` and none of them is a seed template —
        `effect/dist/Graph.js`, `effect/src/Graph.ts`, two `httpApiScalar`
        bundles and `zod/src/v4/core/errors.ts`. A brace-shaped heuristic drops
        them out of the very tree the seeding exists to deliver, silently and
        only in the offline posture, which is the hardest place to notice it.
        """
        rig = _seed(tmp_path, "opencode", {
            ".config/opencode/node_modules/zod/src/v4/core/errors.ts":
                'const tpl = "{{issue}} at {{path}}";\n',
        })
        monkeypatch.setattr(entrypoint, "RIG_DIR", rig)
        home = tmp_path / "home"
        assert entrypoint.seed_host_home("opencode", home) == []
        assert (home / ".config" / "opencode" / "node_modules" / "zod" / "src"
                / "v4" / "core" / "errors.ts").is_file()

    def test_it_merges_into_an_existing_home(self, tmp_path, monkeypatch):
        """`stage_host` and the two `wire_*` paths all write into the same tmpfs
        `$HOME`; a seeding step that could not merge would be ordered-only-one-way
        forever."""
        rig = _seed(tmp_path, "codex", {".codex/notes.txt": "seeded\n"})
        monkeypatch.setattr(entrypoint, "RIG_DIR", rig)
        home = tmp_path / "home"
        (home / ".codex").mkdir(parents=True)
        (home / ".codex" / "already.txt").write_text("kept\n", encoding="utf-8")
        entrypoint.seed_host_home("codex", home)
        assert (home / ".codex" / "already.txt").read_text() == "kept\n"
        assert (home / ".codex" / "notes.txt").read_text() == "seeded\n"

    def test_a_host_with_no_seed_is_not_an_error(self, tmp_path, monkeypatch):
        """Only two hosts are installable and a third may join with no seed at
        all; a missing directory is "nothing to place", not a failed row."""
        monkeypatch.setattr(entrypoint, "RIG_DIR", tmp_path / "rig")
        home = tmp_path / "home"
        assert entrypoint.seed_host_home("nosuchhost", home) == []


class TestNoCredentialCanRideInOnTheFreeTier:
    """`--creds none` asserts every credential VARIABLE is absent
    (`assert_no_credentials`) and says nothing about files, so the seeds are the
    one place a value could reach a tier that is supposed to prove things for
    free. It cannot, and this is where that stays true."""

    @pytest.mark.parametrize("host", HOSTS)
    def test_every_committed_seed_file_is_value_free_or_skipped(self, host):
        seed = HOME_SEED / host
        if not seed.is_dir():
            pytest.skip(f"no committed home_seed/{host}/")
        for path in sorted(seed.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            fields = ("api_key", "apiKey", "API_KEY", '"key"', "auth_token")
            if not any(field in text for field in fields):
                continue
            assert any(token in text for token in entrypoint.SEED_TOKENS), (
                f"{path.relative_to(HOME_SEED)} names a credential field with no "
                f"{'/'.join(entrypoint.SEED_TOKENS)} placeholder in it, so "
                f"seed_host_home would COPY it into the host's $HOME on a tier "
                f"that is defined to hold no credential at all")


class TestTheTokenListIsWrappersOwn:
    def test_it_matches_the_substitution_map_that_binds_them(self):
        """Read out of `wrapper.py`, never retyped.

        The two lists are one contract seen from both ends: the wrapper BINDS
        these tokens per exec on the provider lane, and `seed_host_home` SKIPS
        the files carrying them on the host lane's free tier. A token added to
        one side only is a file that starts arriving in a host's `$HOME` with a
        literal `{{…}}` in it, which no assertion here or in the container would
        otherwise catch.
        """
        tree = ast.parse((CONTAINER_DIR / "wrapper.py").read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name == "seed_invocation_root"):
                continue
            for stmt in ast.walk(node):
                if not (isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Dict)
                        and any(isinstance(t, ast.Name)
                                and t.id == "replacements" for t in stmt.targets)):
                    continue
                found = {key.value for key in stmt.value.keys
                         if isinstance(key, ast.Constant)
                         and isinstance(key.value, str)}
        assert found, ("wrapper.seed_invocation_root no longer assigns a literal "
                       "`replacements` dict — this guard reads it by AST and "
                       "cannot follow it anywhere else")
        assert found == set(entrypoint.SEED_TOKENS)


class TestTheDiscoveryTierReallyCallsIt:
    """Over the AST, because the branch runs only inside a container.

    `tests/test_container_host_lane.py` owns the behaviour and is `container`-
    marked, so it never runs on a PR. A call deleted or moved out from under its
    guard would go unnoticed for exactly as long as nobody runs Docker, and the
    row would go back to reporting on an empty `$HOME` while still passing.
    """

    @staticmethod
    def _main() -> ast.FunctionDef:
        tree = ast.parse((CONTAINER_DIR / "entrypoint.py")
                         .read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return node
        raise AssertionError("entrypoint.py defines no module-level `main`")

    @staticmethod
    def _calls(node: ast.AST, name: str) -> list[ast.Call]:
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == name]

    def test_main_calls_it_exactly_once(self):
        assert len(self._calls(self._main(), "seed_host_home")) == 1

    def test_the_call_is_guarded_by_the_free_tier(self):
        """Guarded, because the credentialed tiers seed through
        `wire_credentials` and a second unconditional copy there would put the
        provider lane's templated `opencode.json` back on top of the host's own
        wiring.

        The INNERMOST enclosing `if` is the guard — every outer one
        (`args.lane == "host"`, and the `with`/`try` above it) contains the call
        too, and reading any of those would let the free-tier condition be
        deleted while this still passed.
        """
        main = self._main()
        call = self._calls(main, "seed_host_home")[0]
        enclosing = [node for node in ast.walk(main)
                     if isinstance(node, ast.If)
                     and node.lineno <= call.lineno <= node.end_lineno]
        assert enclosing, "seed_host_home is called under no `if` at all"
        guard = min(enclosing, key=lambda node: node.end_lineno - node.lineno)
        assert '"none"' in ast.unparse(guard.test).replace("'", '"'), (
            f"seed_host_home is guarded by {ast.unparse(guard.test)!r}, "
            f"which does not name the `--creds none` tier")

    def test_it_runs_before_the_host_is_staged(self):
        """`install-host --check`'s row 5 probes this machine, so it has to see
        the same `$HOME` the discovery step will. Staging first would have the
        check answer about a posture the row then changes underneath it."""
        blocks = []
        for node in ast.walk(self._main()):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            seeded = [i for i, stmt in enumerate(body)
                      if self._calls(stmt, "seed_host_home")]
            staged = [i for i, stmt in enumerate(body)
                      if self._calls(stmt, "stage_host")]
            # An outer block holds both inside ONE statement and says nothing
            # about their order; only a block that separates them does.
            if seeded and staged and set(seeded) != set(staged):
                blocks.append((seeded, staged))
        assert blocks, (
            "no block in `main` holds the seeding step and a `stage_host` call "
            "as separate statements, so their order is no longer pinned")
        for seeded, staged in blocks:
            assert max(seeded) < min(staged)
