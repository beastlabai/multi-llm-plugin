#!/usr/bin/env python3
"""In-container runner for the credentialed provider lane (harness-expansion.md §7.4).

THIS IS THE THIRD LANE, and it exists because neither of the other two can answer
its question. The provider lane runs ``--network none`` against ``stub_server.py``
— there is no openrouter.ai to reach. The credentialed host lane proves the
*host's* brain is live while its PATH wrapper does the exact opposite for
providers: ``/opt/mllm/bin/<cli>`` **is** ``wrapper.py``, and that wrapper rebuilds
the child environment from scratch, substituting a fake per-exec stub key for any
vendor credential and pointing the base URL at the stub ("step 2 is a scrub, not
an overlay"). Relaxing the scrub to let a live key through would leave
``OPENROUTER_API_KEY`` in a third-party binary's environment on an open network,
which is precisely what it exists to prevent. So this lane bypasses the wrapper
and invokes the real binary at ``/opt/mllm/real/<binary>`` directly.

WHAT IT MUST NOT BECOME. Not a hand-written command. Argv and stdin come from the
shipped adapter's ``build_command_for_transport(..., "argv")`` (aider's decline
supply rides stdin) and the environment from its
``resolve_env``/``get_env``/``get_remove_env``, so the smoke exercises what a user
actually runs. A bespoke command line here would green a lane while the adapter
was broken.

THE CONTAINMENT SET IS ALL-OR-NOTHING (§7.4's containment table). This process
refuses to spend a token unless every control it owns is in place, and it
signals that refusal with :data:`EXIT_CONTROL_UNAVAILABLE` so the host driver can
**skip with the reason recorded** rather than run weakened. "Run it anyway with
weaker controls" is never the fallback.

Stdlib only. The image's system ``python3`` runs it, with the product importable
from ``/opt/mllm/skill``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import shutil
from urllib.parse import urlsplit
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/mllm/skill")

from egress_resolver import DEFAULT_ALLOWLIST, serve  # noqa: E402

#: The lane's DEFAULT production endpoint — every row that reaches its models
#: through OpenRouter. It stopped being "the lane's one endpoint" when
#: cursor-agent joined: cursor has no OpenRouter path and no stub is possible for
#: it (binary protobuf over Connect-RPC), so a row may override this, its
#: credential variable and its DNS allowlist together. What did **not** become
#: per-row is any control: the resolver still runs, the credential is still
#: fail-closed, and redaction still happens at write time.
LIVE_BASE_URL = "https://openrouter.ai/api/v1"

#: The default credential variable, overridable per row for the same reason.
LIVE_CREDENTIAL_ENV = "OPENROUTER_API_KEY"

#: A row whose credential is a FILE rather than a variable names the destination
#: here and receives the file's bytes through this variable. cursor-agent is the
#: case: its credential is `~/.config/cursor/auth.json`, and a `:ro` bind of that
#: file cannot be read inside the container at all — it is mode 0600 owned by the
#: developer's uid, and this lane runs as uid 10001 with `--user $(id -u)`
#: explicitly rejected by `posture.yaml` ("it produces a process with no passwd
#: entry and an unowned $HOME"). So the host-side driver, which *can* read it,
#: passes the bytes in the environment and this process writes them into the
#: run's own disposable HOME at 0600. Same credential, same source file, one
#: transport that works — and, unlike a bind, the value is in `os.environ`, so
#: `redact` scrubs it from every artifact by value with no extra machinery.
CREDENTIAL_MATERIAL_ENV = "MLLM_LIVE_CREDENTIAL_MATERIAL"

#: A control could not be applied — the host driver turns this into a SKIP with
#: the reason, per §7.4's skip policy. Distinct from any CLI exit code, and
#: distinct from a real failure, because a lane that quietly downgrades its own
#: containment is worse than a lane that does not run.
EXIT_CONTROL_UNAVAILABLE = 20

#: Vendor homes a shipped adapter has no business reaching on this lane. Recorded
#: here as documentation; the host-side assertion 5 owns the actual grading, and
#: the resolver's own allowlist is what enforces it.
FORBIDDEN_EGRESS = (
    "github.com", "githubusercontent.com", "ampcode.com", "moonshot.cn",
    "kimi.com", "dashscope", "mistral.ai", "deepseek.com",
)

#: Which of the child's variables reach ``evidence["env"]`` — assertion 5's
#: belt-and-braces half reads that dict BY NAME, so a variable this filter drops
#: is a variable the assertion can never see. It used to be a bare prefix tuple
#: inline at the capture site, and it silently bounded which opt-outs could be
#: graded: every name a Wave-3.1 row wants asserted (``CLINE_NO_AUTO_UPDATE``,
#: ``DO_NOT_TRACK``) starts with a prefix that was not on it, so declaring them
#: in ``OPT_OUTS`` would have failed on the CAPTURE rather than on the CLI.
#:
#: Prefixes are per-CLI namespaces; :data:`ENV_CAPTURE_NAMES` carries the
#: cross-vendor opt-outs the image sets once for everybody, which have no prefix
#: to match on. Values still go through the KEY/TOKEN/SECRET redaction at the
#: capture site — widening what is *named* never widens what is *printed*.
#:
#: WAVE 3.2-3.4 adds eight namespaces for the same reason: every variable the
#: five new rows are wired with (``ANTHROPIC_BASE_URL``, ``ANTHROPIC_API_KEY``,
#: ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC``, ``CODEX_HOME``,
#: ``GROK_MODELS_BASE_URL``, ``XAI_API_KEY``, ``GEMINI_API_KEY``) starts with a
#: prefix that was not on this tuple, so the resolved env the evidence carried
#: could not have shown whether the wiring landed at all — and a5's opt-out half
#: would have failed on the CAPTURE rather than on the CLI. The two credential
#: aliases among them (``XAI_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``) are captured by
#: NAME and redacted by the KEY/TOKEN rule below, exactly like every other key.
ENV_CAPTURE_PREFIXES = (
    "AIDER_", "ANTHROPIC_", "CLAUDE_", "CLINE_", "CODEX_", "COPILOT_",
    "CURSOR_", "DEEPSEEK_", "DSH_", "GEMINI_", "GOOGLE_", "GOOSE_", "GROK_",
    "KILO_", "KILOCODE_", "KIMI_", "MISTRAL_", "OPENAI_", "OPENCODE_",
    "OPENROUTER_", "PI_", "QWEN_", "VIBE_", "XAI_",
)

#: The image-wide opt-outs from the Dockerfile's step-8 ENV block, which carry no
#: vendor prefix and so cannot be matched by :data:`ENV_CAPTURE_PREFIXES`.
ENV_CAPTURE_NAMES = (
    "DO_NOT_TRACK", "NO_UPDATE_NOTIFIER", "NPM_CONFIG_UPDATE_NOTIFIER",
)


def _jwt_claims(token: str) -> list[str]:
    """Identifying string claims in a JWT payload (ids, emails); [] when not a JWT."""
    parts = token.split(".")
    if len(parts) != 3 or not parts[1]:
        return []
    try:
        claims = json.loads(base64.urlsafe_b64decode(
            (parts[1] + "=" * (-len(parts[1]) % 4)).encode("ascii")))
    except (ValueError, binascii.Error, UnicodeError):
        return []
    found: list[str] = []
    _walk_strings(claims, found, decode_jwt=False)
    return found


def _walk_strings(node, found: list, decode_jwt: bool = True) -> None:
    if isinstance(node, str):
        if len(node) >= 20 or ("@" in node and len(node) >= 5):
            found.append(node)
        if decode_jwt:
            found.extend(_jwt_claims(node))
    elif isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, found, decode_jwt)
    elif isinstance(node, list):
        for value in node:
            _walk_strings(value, found, decode_jwt)


def credential_strings(material: str) -> list[str]:
    """Every value a file credential's bytes could leak, for by-value redaction.

    NESTED, not only top-level: a subscription file keeps its tokens one level
    down (codex `tokens.{id_token,access_token,refresh_token,account_id}`,
    claude-code `claudeAiOauth.{accessToken,refreshToken}`), so reading only
    top-level strings scrubbed none of them. Plus the claims inside any JWT
    (codex's `id_token` carries the account email base64-encoded) and any
    email-shaped value, because a CLI that prints "logged in as …" emits those
    in clear. Not JSON: the whole blob, already in the caller's list, is all
    there is.
    """
    try:
        document = json.loads(material)
    except ValueError:
        return []
    found: list[str] = []
    _walk_strings(document, found)
    # Longest first, so a value containing another is replaced before the
    # shorter one splits it.
    return sorted(set(found), key=len, reverse=True)


def redact(text: str, secrets) -> str:
    """Remove every credential value before anything reaches an artifact.

    Several of these CLIs echo their configuration on startup, so a captured
    transcript is exactly where a key leaks. Redaction happens at write time —
    the artifact can then never carry a value, rather than carrying one until
    somebody remembers to scrub it.
    """
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "<redacted>")
    return text


#: Directories inside a seed tree that are UPSTREAM ARTIFACTS rather than our
#: templates, skipped by both :func:`render_seed` and :func:`seed_guards`.
#:
#: MEASURED, and it is opencode that forces it. Its seed is not a config file: the
#: image's warm-up bakes the whole cold-start tree into it — 3,654 files, 67 MB of
#: ``node_modules`` plus the 4 MB models.dev catalogue — because a fresh
#: ``opencode run`` otherwise npm-installs ~63 MB before it answers anything. Two
#: consequences, both of which turn a green row red for no reason:
#:
#: * 68 of those files contain the substring ``localhost`` (``zod``'s test
#:   fixtures, ``kubernetes-types``' declarations, the opencode SDK's own dev
#:   server). ``seed_guards``' stub-address check would fail the row on every one
#:   of them, and none is an endpoint this row could POST to;
#: * rendering means READING and rewriting all 3,654, to substitute tokens that
#:   appear in exactly three of them.
#:
#: Excluding these two names narrows nothing that matters: they hold no
#: ``{{token}}``, and the endpoint a row really reached is graded off the DNS log
#: by assertion 1, not off a grep of a vendored tree.
VENDOR_SUBTREES = ("node_modules", ".cache")


def _is_vendor_artifact(path: Path, home: Path) -> bool:
    """True for a file inside a :data:`VENDOR_SUBTREES` directory."""
    try:
        parts = path.relative_to(home).parts
    except ValueError:
        parts = path.parts
    return any(part in VENDOR_SUBTREES for part in parts)


def render_seed(cli: str, home: Path, base_url: str, api_key_env: str,
                model_name: str = "", drop: tuple[str, ...] = ()) -> list[str]:
    """Render ``home_seed/<cli>/`` into this run's HOME, bound to the LIVE endpoint.

    **One template, two bindings — never a second tree.** The provider lane binds
    ``{{base_url}}`` to the stub's ``/s/<exec-id>/v1`` and ``{{api_key_env}}`` to
    ``STUB_API_KEY``; this lane binds them to openrouter.ai and
    ``OPENROUTER_API_KEY``. A parallel ``home_seed-live/`` is the artifact that
    rots the first time the stub row's config changes, which is why it does not
    exist.

    Only the key's *variable name* ever enters the file. Its value reaches the CLI
    through the process environment and nowhere else (§3.3 rule 4: a rendered
    secret inside a seed tree is a committed credential).

    ``drop`` removes named files from the COPY, for a tree that is part live
    configuration and part stub artifact. Wave 3.2 is what needs it: opencode's
    seed carries both the baked cold-start tree this lane cannot run without AND
    a stored credential (``.local/share/opencode/auth.json``) whose committed
    value is the stub's session key — which on this lane would SHADOW the real
    ``OPENROUTER_API_KEY`` with a string worth nothing. ``--no-seed`` is
    all-or-nothing and would throw away the 67 MB warm tree with it; a per-file
    drop says the same thing at the resolution the tree actually has. Every
    dropped path is returned in the evidence, so "the file was removed" and "the
    file was never there" stay distinguishable after the fact.

    Returns the list of rendered files, for the caller's pre-flight guards.
    """
    source = Path("/opt/mllm/rig/home_seed") / cli
    home.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return []
    shutil.copytree(source, home, dirs_exist_ok=True)
    for rel in drop:
        target = home / rel
        if target.is_file():
            target.unlink()

    replacements = {
        "{{base_url}}": base_url,
        "{{base_url_root}}": base_url.rsplit("/v1", 1)[0],
        "{{api_key_env}}": api_key_env,
        # The wire model. On the provider lane wrapper.py binds this to
        # `stub-model`; here it is the real slug, so a seed whose catalog maps an
        # alias to a wire name (vibe) reaches OpenRouter with a model it knows.
        # Without it the smoke would POST `stub-model` and take a 400 that reads
        # like an adapter bug.
        "{{model_name}}": model_name,
    }
    rendered = []
    for path in home.rglob("*"):
        if not path.is_file() or _is_vendor_artifact(path, home):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "{{" not in text:
            continue
        for token, value in replacements.items():
            text = text.replace(token, value)
        path.write_text(text, encoding="utf-8")
        rendered.append(str(path))
    return rendered


def seed_guards(home: Path, base_url: str,
                require_base_url: bool = True) -> list[str]:
    """The two cheap pre-flight guards §7.4 asks for, run BEFORE the CLI spawns.

    (i) no ``{{`` survives anywhere in the rendered tree — an unsubstituted
    placeholder is a hard failure at seed time, not a TOML parse error or a 404 to
    diagnose after the fact; (ii) the rendered endpoint really is the live one —
    because reusing the provider lane's rendered seed would have the smoke quietly
    POST to 127.0.0.1 while assertion 1 looked green.

    ``require_base_url`` turns (ii) off for a row whose endpoint DOES NOT TRAVEL
    IN THE SEED — the CLI routes itself and the seed is there for other first-run
    facts (opencode's baked cold-start tree; claude-code's onboarding and trust
    entry). That is a real relaxation and it is per-row, declared host-side with
    its measurement, never a default: guard (i) and the stub-address half of (ii)
    still run over the whole tree, and the endpoint a row REACHED is graded by
    assertion 1 off the DNS log — evidence about the wire rather than about a
    string in a file, and strictly the stronger of the two.
    """
    problems = []
    saw_base_url = False
    for path in home.rglob("*"):
        if not path.is_file() or _is_vendor_artifact(path, home):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "{{" in text:
            problems.append(f"unsubstituted placeholder in {path}")
        if base_url in text:
            saw_base_url = True
        if "127.0.0.1" in text or "localhost" in text:
            problems.append(f"stub address survived into {path}")
    if require_base_url and not saw_base_url:
        problems.append(f"no seeded file names the live endpoint {base_url}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, help="provider key, e.g. copilot")
    parser.add_argument("--row-env", default="{}",
                        help="the row's `env_extra` as JSON. §7.4 drops a live "
                             "row's STUB overrides (base_url_vars, "
                             "credential_vars, http_log) and keeps the rest, so "
                             "these are image-layout facts the adapter has no "
                             "business hardcoding -- copilot's pre-warmed "
                             "XDG_CACHE_HOME is the case that forced this. "
                             "Applied BENEATH resolve_env so the adapter's own "
                             "production env still stands.")
    parser.add_argument("--binary", default=None,
                        help="binary name when it differs from the provider key")
    parser.add_argument("--config-root", action="append", default=[],
                        metavar="VAR=REL",
                        help="the row's `config_roots`, exported into THIS "
                             "process's environment before resolve_env runs. An "
                             "adapter that respects a pinned root (copilot's "
                             "COPILOT_HOME, vibe's VIBE_HOME) then points the CLI "
                             "at the rendered seed instead of minting a fresh "
                             "empty one -- which for a CLI whose catalog lives in "
                             "a config file is the difference between a working "
                             "row and one with no providers configured at all.")
    parser.add_argument("--model", required=True, help="the models[0] slug")
    parser.add_argument("--prompt", required=True,
                        help="the fixed short fixture prompt (a few hundred bytes)")
    parser.add_argument("--out", type=Path, default=Path("/work/out"))
    parser.add_argument("--timeout-s", type=int, default=180,
                        help="the CLI's hard inner bound, strictly under the outer")
    parser.add_argument("--upstream-dns", default=os.environ.get(
        "MLLM_EGRESS_UPSTREAM", "1.1.1.1"))
    parser.add_argument("--live-base-url", default=LIVE_BASE_URL,
                        help="the row's production endpoint. Defaults to "
                             "OpenRouter; a row whose vendor is its own backend "
                             "(cursor-agent) overrides it, and the value is what "
                             "the seed's {{base_url}} binds to.")
    parser.add_argument("--credential-env", default=LIVE_CREDENTIAL_ENV,
                        help="the variable carrying this row's credential. "
                             "Fail-closed on absence either way -- only the NAME "
                             "is per-row, never the requirement.")
    parser.add_argument("--credential-file", default=None, metavar="REL",
                        help="destination, relative to this run's HOME, for a "
                             "row whose credential is a FILE. The bytes arrive "
                             "in " + CREDENTIAL_MATERIAL_ENV + " (see its "
                             "docstring for why a bind mount cannot work) and "
                             "are written here at 0600.")
    parser.add_argument("--no-seed", action="store_true",
                        help="do not render home_seed/<cli>/ into this run's "
                             "HOME. For a row whose seed exists ONLY to point "
                             "the CLI at the stub -- pi, cline and kilocode all "
                             "spell a stub session key into theirs -- there is "
                             "nothing on this lane for it to configure: the CLI "
                             "routes itself from its own models[0] spec and "
                             "reads OPENROUTER_API_KEY from the environment. "
                             "The alternative was teaching render_seed a "
                             "credential token, which its own docstring forbids "
                             "(a rendered secret inside a seed tree is a "
                             "committed credential). The host driver decides; "
                             "see STUB_ONLY_SEED in the test module for the "
                             "per-row measurement.")
    parser.add_argument("--credential-alias", action="append", default=[],
                        metavar="NAME",
                        help="an ADDITIONAL variable name this row's credential "
                             "value is exported under, inside the container "
                             "only (repeatable). For a CLI whose production "
                             "auth variable is its own vendor's "
                             "(claude-code's ANTHROPIC_AUTH_TOKEN, grok's "
                             "XAI_API_KEY) while the key on this lane is the "
                             "OpenRouter one, which is exactly what the HOST "
                             "lane's `wire_credentials` does for the same two "
                             "surfaces. The value is read from "
                             "--credential-env inside this process and never "
                             "appears in argv, in a tracked file or in a run "
                             "record: only the NAME crosses the boundary, and "
                             "`redact` already scrubs the value from every "
                             "artifact because it is the same string.")
    parser.add_argument("--seed-drop", action="append", default=[],
                        metavar="REL",
                        help="a seed file to remove after the copy and before "
                             "rendering (repeatable), for a tree that is part "
                             "live configuration and part stub artifact. See "
                             "`render_seed`'s docstring: opencode's seed carries "
                             "both the baked cold-start tree this lane needs and "
                             "a stored credential whose committed value would "
                             "shadow the real key. The host driver decides; the "
                             "per-row measurement lives in SEED_DROP in the test "
                             "module.")
    parser.add_argument("--seed-link", action="append", default=[],
                        metavar="REL=TARGET",
                        help="a symlink to create inside this row's HOME before "
                             "the CLI runs (repeatable), from the row's own "
                             "`seed_links` — the SAME declaration `wrapper.py` "
                             "reads on the provider lane, so one row says it "
                             "once and both lanes obey it. It exists for a "
                             "vendor tree the image already carries and the "
                             "run must not re-create: omp extracts 346 MB of "
                             "Bun addons into $HOME/.omp/natives on its first "
                             "agent run, and this lane's HOME is a **noexec** "
                             "tmpfs (`--tmpfs /work/live:rw,...`, to which "
                             "docker adds nosuid,nodev,noexec), so the addon "
                             "lands and then cannot be mapped: `failed to map "
                             "segment from shared object`, exit 1 in 1.05 s, "
                             "before a single request. Pointing HOME at the "
                             "image's read-only extract fixes the cost and the "
                             "failure at once. NOT a seed file: nothing here is "
                             "rendered, nothing is templated, and `seed_guards` "
                             "never sees it — which is why it survives "
                             "`--no-seed`.")
    parser.add_argument("--endpoint-in-env", action="store_true",
                        help="this row's endpoint does not travel in its seed — "
                             "the CLI routes itself, or the base URL is an "
                             "environment variable — so `seed_guards` must not "
                             "require a seeded file to name it. The other two "
                             "guards still run, and assertion 1 still grades the "
                             "endpoint the row actually REACHED off the DNS log.")
    parser.add_argument("--allow", action="append", default=None, metavar="NAME",
                        help="a DNS name this row may resolve (repeatable). "
                             "Defaults to the resolver's own allowlist. This is "
                             "the lane's only egress enforcement, so a row that "
                             "widens it is widening the containment set.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get(args.credential_env, "")
    material = os.environ.get(CREDENTIAL_MATERIAL_ENV, "")
    evidence: dict = {
        "cli": args.cli,
        "model": args.model,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def bail(reason: str, code: int) -> int:
        evidence["control_failure"] = reason
        (args.out / "live-provider.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"live_provider: {reason}", file=sys.stderr)
        return code

    # Fail-closed, whichever shape this row's credential takes. A row declaring
    # `--credential-file` is satisfied by the material; every other row by the
    # variable. Neither is ever waved through: "run it anyway with weaker
    # controls" is not this lane's fallback, and an unauthenticated CLI would
    # spend the row proving only that it is logged out.
    if args.credential_file:
        if not material:
            return bail(f"{CREDENTIAL_MATERIAL_ENV} is unset in the container — "
                        f"{args.cli}'s credential is a file and the host-side "
                        f"driver passed no bytes for it",
                        EXIT_CONTROL_UNAVAILABLE)
    elif not key:
        return bail(f"{args.credential_env} is unset in the container",
                    EXIT_CONTROL_UNAVAILABLE)
    # An alias with nothing to alias is fail-closed too, and for the same reason
    # the checks above are: a row that exported an EMPTY ANTHROPIC_AUTH_TOKEN
    # would not fail loudly, it would fall through to whatever other auth the CLI
    # can find — which for claude-code is the developer's own subscription.
    if args.credential_alias and not key:
        return bail(f"{args.cli} declares the credential aliases "
                    f"{sorted(args.credential_alias)} but {args.credential_env} "
                    f"carries no value to alias — a present-and-empty auth "
                    f"variable is how a row silently authenticates somewhere "
                    f"else", EXIT_CONTROL_UNAVAILABLE)

    #: Every value redaction must scrub. For a file credential that is the whole
    #: blob AND each long string inside it, because a CLI that echoes its own
    #: config prints the token, not the JSON.
    secrets = [value for value in (key, material) if value]
    if material:
        secrets.extend(credential_strings(material))

    # ---- Egress: enforced, or the row does not run ------------------------ #
    #
    # THE ROW'S OWN ENDPOINT IS ALWAYS ALLOWED, and that is a fix, not a
    # convenience. `--allow` used to REPLACE the default allowlist rather than
    # extend it, so the first row to name any extra host silently lost
    # `openrouter.ai` and could no longer reach the endpoint it exists to test.
    # Measured 2026-08-19: permitting one name for aider turned a 6 s green row
    # into a 67 s run where raw.githubusercontent.com resolved, openrouter.ai
    # went NXDOMAIN, and litellm retried the model call to exhaustion. Nothing
    # said "your allowlist dropped the endpoint" — it surfaced as a parse
    # failure two assertions away.
    #
    # A row can still narrow or widen everything else; what it cannot do any
    # more is cut itself off from the endpoint under test, which was never a
    # containment property worth having.
    endpoint_host = urlsplit(args.live_base_url).hostname or ""
    allowlist = tuple(
        h for h in dict.fromkeys(
            [endpoint_host] + list(args.allow or DEFAULT_ALLOWLIST))
        if h
    )
    try:
        udp, tcp, _log = serve(allowlist,
                               args.upstream_dns, args.out / "dns.jsonl")
    except OSError as exc:
        # Almost always the privileged-port bind: the lane's `docker run` passes
        # `--sysctl net.ipv4.ip_unprivileged_port_start=53`, and a daemon that
        # refuses that sysctl leaves this control unavailable.
        return bail(f"the egress resolver could not bind port 53 ({exc}) — "
                    f"the lane's containment cannot be applied",
                    EXIT_CONTROL_UNAVAILABLE)

    try:
        # ---- Workspace: empty, disposable, no checkout in sight ----------- #
        # §7.4's containment table: no repository mount, no `-v $(pwd)`, and
        # copilot's `--add-dir` (which resolves to the spawn's own cwd) therefore
        # points at an empty directory rather than a source tree.
        workdir = Path("/work/live") / args.cli
        home = workdir / "home"
        workdir.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)

        # The row's `seed_links`, before anything else touches HOME: a config
        # root or a credential under a linked subtree has to resolve THROUGH the
        # link, not land beside it. Same declaration, same failure mode and the
        # same hard refusal as `wrapper.py`'s copy — a declared target missing
        # from the image is the image being stale, and running anyway would
        # reproduce the very extraction the link exists to avoid.
        for spec in args.seed_link:
            rel, _, target = spec.partition("=")
            if not rel or not target:
                return bail(f"--seed-link {spec!r} is not REL=TARGET",
                            EXIT_CONTROL_UNAVAILABLE)
            if not Path(target).exists():
                return bail(
                    f"{args.cli} declares seed_links[{rel}] -> {target}, which "
                    f"does not exist in this image. Rebuild it "
                    f"(`make e2e-image`): the layer that installs this CLI is "
                    f"what produces that tree.",
                    EXIT_CONTROL_UNAVAILABLE)
            link = home / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if not (link.is_symlink() or link.exists()):
                os.symlink(target, link)
        evidence["seed_links"] = list(args.seed_link)

        # The row's config roots, BEFORE resolve_env: `get_env` reads os.environ
        # to decide whether a root is already pinned, so this has to land here
        # rather than in `child`. It is this process's own environment, not an
        # adapter mutating a shared one -- §3.3 rule 1 binds adapters, and the
        # rule exists so one slot cannot re-point another's spawn.
        config_roots: dict[str, str] = {}
        for spec in args.config_root:
            var, _, rel = spec.partition("=")
            if var and rel:
                target = home / rel
                target.mkdir(parents=True, exist_ok=True)
                os.environ[var] = str(target)
                config_roots[var] = str(target)
        evidence["config_roots"] = config_roots

        # A SEED THIS LANE MUST NOT RENDER IS NOT THE SAME AS A CLI THAT NEEDS
        # NO SEED, and the evidence says which one this row is. `--no-seed`
        # means the tree exists and is a stub artifact; the absent-tree case
        # below still reports `seeded` False on its own.
        seeded = (not args.no_seed
                  and (Path("/opt/mllm/rig/home_seed") / args.cli).is_dir())
        rendered = ([] if args.no_seed
                    else render_seed(args.cli, home, args.live_base_url,
                                     args.credential_env, args.model,
                                     tuple(args.seed_drop)))
        evidence["seed_files_rendered"] = rendered
        evidence["seed_skipped"] = bool(args.no_seed)
        # A DROPPED FILE IS RECORDED, not silently absent: "this lane removed the
        # stub's stored credential" and "this CLI never had one" are different
        # facts, and only one of them is a decision somebody made.
        evidence["seed_files_dropped"] = list(args.seed_drop)
        evidence["endpoint_in_env"] = bool(args.endpoint_in_env)
        # Gated on the tree EXISTING, not on a token having been substituted: a
        # seed whose endpoint was hardcoded rather than templated substitutes
        # nothing, and that is precisely the case the guards must catch — it is
        # how a live smoke ends up POSTing to 127.0.0.1 while assertion 1 looks
        # green. An env-only CLI seeds no tree and skips the guards legitimately.
        problems = (seed_guards(home, args.live_base_url,
                                require_base_url=not args.endpoint_in_env)
                    if seeded else [])
        if problems:
            return bail("seed pre-flight failed: " + "; ".join(problems),
                        EXIT_CONTROL_UNAVAILABLE)

        # A file credential lands in this run's disposable HOME, at the path the
        # CLI's own resolution rule looks in, owned by this process and mode
        # 0600. STAGED LAST, after the seed is rendered, its drops applied and
        # its guards run: codex's seed ships a stub `.codex/auth.json` at the
        # very path a subscription credential lands on, so a credential staged
        # first was overwritten by the copy and then deleted by the drop. It is
        # never the developer's file: the bytes came through the environment
        # precisely because that file cannot be mounted readable here, and
        # nothing in this container can write back to it.
        if args.credential_file:
            target = home / args.credential_file
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(material, encoding="utf-8")
                target.chmod(0o600)
            except OSError as exc:
                return bail(f"could not stage {args.cli}'s credential at "
                            f"{target} ({exc})", EXIT_CONTROL_UNAVAILABLE)
            evidence["credential_staged_at"] = str(target)

        # ---- Argv and env come from the SHIPPED adapter ------------------- #
        from utils.provider_registry import get_provider

        provider = get_provider(args.cli)
        if provider is None:
            return bail(f"no built-in adapter for {args.cli!r} — the lane's row "
                        f"list and the registry disagree", EXIT_CONTROL_UNAVAILABLE)

        # `timeout_s=` is threaded, and that is a CONTAINMENT control rather
        # than tidiness. §7.4's Time row requires "a hard outer `row_timeout_s`
        # on the `docker run` AND the CLI's own cap wherever one exists", with
        # the inner strictly under the outer. Omitting it would leave an adapter
        # that HAS a wall-clock flag (qwen-code's `--max-wall-time`) running
        # unbounded inside the child on a lane that spends real credit, with
        # only the outer kill to stop it -- and a killed child reports a bare
        # TIMEOUT instead of the CLI's own structured budget error.
        #
        # Adapters with no such flag are unaffected: §1's guard makes
        # `timeout_s=None` and a threaded value render identical argv for them.
        # THE DECLARED OUTPUT CHANNEL, GRADED. `supports_file_output` says
        # whether this adapter's CLI can write the per-model result file the Lane
        # A prompts name, and until now nothing anywhere could check it: the
        # container stub never emits a tool call, so on every offline lane an
        # adapter that cannot write a file is indistinguishable from one that
        # can. That is precisely how cursor-agent shipped a release answering
        # every review slot with "I can't write the requested file while Ask mode
        # is active".
        #
        # This lane is the only place the question can be asked, so it asks it:
        # the smoke prompt ends in the block PRODUCTION renders for this very
        # adapter (`llm_client.render_output_location`), not a hand-written one.
        # A `True` adapter is asked for a file and the file's existence is the
        # verdict; a `False` adapter is asked for stdout and the file's ABSENCE
        # is. No second run and no extra spend — the same one-shot, with the tail
        # production would have put on it.
        from utils.llm_client import render_output_location

        declared = bool(getattr(provider, "supports_file_output", True))
        channel_path = workdir / "declared-channel.json"
        # A FULL `provider:model` spec, never the bare CLI id. `parse_model_spec`
        # reads a bare token as a MODEL and resolves the provider from
        # `default_provider` — so passing `args.cli` here silently rendered every
        # row's block from cursor-agent's declaration instead of its own, and the
        # probe graded each CLI against a different adapter's capability. Caught
        # on copilot's first run, which was asked for stdout and then failed for
        # not producing a file.
        spec = f"{args.cli}:{args.model}"
        rendered = render_output_location(channel_path, spec)
        prompt = (
            f"{args.prompt}\n\n## Output Location\n"
            f"{rendered}\n"
        )
        # The block and the declaration come from the same place, so a mismatch
        # is a bug in this probe rather than a finding about the CLI. Refuse
        # rather than spend a row producing a verdict about the wrong adapter.
        asks_for_file = "Write your JSON output to this file" in rendered
        if asks_for_file != declared:
            return bail(
                f"the rendered output-location block disagrees with "
                f"{args.cli}'s own supports_file_output={declared} (spec "
                f"{spec!r} resolved elsewhere) — the channel probe would grade "
                f"the wrong adapter", EXIT_CONTROL_UNAVAILABLE)
        evidence["declared_supports_file_output"] = declared
        evidence["channel_path"] = str(channel_path)

        # The ARGV TRANSPORT's pair, never `build_command` alone: for aider the
        # read-only invocation is argv PLUS the decline supply it reads on stdin
        # (plans/ask-answer-recovery-and-false-success.md D4.2). Spawned from
        # `build_command` with no input, aider's "Add file to the chat?" prompts
        # read EOF, EOF means yes, and the mention auto-add this lane must not
        # pay for comes back silently. Every other adapter returns
        # `stdin_text=None`, and `input=None` below leaves its spawn as it was.
        from utils.providers.base import ReadOnlyInvocationRefused

        try:
            argv, stdin_text = provider.build_command_for_transport(
                prompt, args.model, "argv", timeout_s=args.timeout_s,
                cwd=str(workdir))
        except ReadOnlyInvocationRefused as exc:
            return bail(f"{args.cli} refused before spawn: {exc}",
                        EXIT_CONTROL_UNAVAILABLE)
        argv = list(argv)
        binary = args.binary or args.cli
        real = Path("/opt/mllm/real") / binary
        if not real.exists():
            return bail(f"{real} is not in this image — rebuild with the CLI's "
                        f"Dockerfile install (§7 step 2)", EXIT_CONTROL_UNAVAILABLE)
        argv[0] = str(real)

        child = dict(os.environ)
        # The staged bytes have done their job; the CLI reads the FILE, never
        # this variable. Dropping it here keeps the credential out of the child's
        # environment entirely, so a CLI that echoes its own env on startup —
        # several of these do, which is why `redact` exists at all — cannot print
        # it. Redaction still covers it if some other path reaches it.
        child.pop(CREDENTIAL_MATERIAL_ENV, None)
        child["HOME"] = str(home)
        child["XDG_CONFIG_HOME"] = str(home / ".config")
        child["XDG_DATA_HOME"] = str(home / ".local" / "share")
        child["XDG_CACHE_HOME"] = str(home / ".cache")
        child["XDG_STATE_HOME"] = str(home / ".local" / "state")
        child["TMPDIR"] = str(workdir / "tmp")
        Path(child["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        for name in provider.get_remove_env() or []:
            child.pop(name, None)
        # The row's `env_extra`, applied BEFORE the adapter so the adapter wins
        # every name it sets (§7.4: "the adapter's own production env stands").
        # What survives is exactly what the adapter leaves unset -- for copilot
        # that is XDG_CACHE_HOME, which points at the 209 MB package cache the
        # image pre-warms at build time. Without it the CLI re-extracts into this
        # row's throwaway HOME and dies on a partial tree; with it the row costs
        # nothing extra. A per-slot value in the adapter would pay 209 MB on every
        # spawn, and a hardcoded /opt path would be wrong on a host, so this is
        # the right owner for the fact.
        row_env = json.loads(args.row_env or "{}")
        adapter_env = provider.resolve_env(args.model)
        from_row = sorted(set(row_env) - set(adapter_env))
        child.update({k: str(v) for k, v in row_env.items()})
        child.update(adapter_env)
        evidence["env_from_row"] = from_row
        # THE SAME CREDENTIAL, UNDER THE NAME THIS CLI'S PRODUCTION AUTH PATH
        # READS. claude-code authenticates through ANTHROPIC_AUTH_TOKEN and grok
        # through XAI_API_KEY; neither product routes through OpenRouter, so
        # neither adapter emits an OpenRouter variable and neither ever will —
        # this lane's key still has to arrive under the name the binary looks
        # for. It is a port of the host lane's own wiring
        # (`entrypoint.py::wire_credentials`, which sets ANTHROPIC_AUTH_TOKEN =
        # key for exactly this reason), applied one layer down.
        #
        # AFTER the adapter, deliberately: an adapter that DID set one of these
        # names would be setting it from the user's own environment, and on this
        # lane the row's declared credential is the only one that may reach the
        # binary. Recorded by name in the evidence, redacted by value everywhere.
        for alias in args.credential_alias:
            child[alias] = key
        evidence["credential_aliases"] = list(args.credential_alias)

        evidence["argv"] = [redact(a, secrets) for a in argv]
        # NAMES with credential values redacted — never a value, on the same rule
        # the provider lane's credential allowlist already encodes.
        evidence["env"] = {
            name: ("<redacted>" if ("KEY" in name or "TOKEN" in name
                                    or "SECRET" in name) else redact(value, secrets))
            for name, value in sorted(child.items())
            if name.startswith(ENV_CAPTURE_PREFIXES) or name in ENV_CAPTURE_NAMES
        }

        # ---- One request, one attempt, a hard inner bound ----------------- #
        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(argv, cwd=str(workdir), env=child,
                                  capture_output=True, text=True,
                                  input=stdin_text,
                                  timeout=args.timeout_s)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out, code = True, None
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        evidence["wall_clock_s"] = round(time.time() - started, 2)
        evidence["exit_code"] = code
        evidence["timed_out"] = timed_out
        # Scrubbed before it is written, not after.
        evidence["stdout"] = redact(out or "", secrets)
        evidence["stderr"] = redact(err or "", secrets)

        # WHICH CHANNEL ACTUALLY DELIVERED. File first, exactly as
        # `invoke_with_file_output` orders it in production, so an adapter that
        # writes one is served from it here too.
        file_text = ""
        if channel_path.is_file():
            try:
                file_text = channel_path.read_text(encoding="utf-8")
            except OSError:
                file_text = ""
        evidence["observed_output_channel"] = "file" if file_text.strip() else "stdout"
        evidence["channel_file_bytes"] = len(file_text)

        # ---- The adapter's OWN parser and classifier ---------------------- #
        # Assertion 3 is about the shipped parser surviving the vendor's real
        # bytes, so it has to be the shipped parser that reads them.
        try:
            parsed = provider.parse_output(out or "", err or "")
        except Exception as exc:  # noqa: BLE001
            parsed = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        evidence["parsed"] = json.loads(redact(json.dumps(parsed, default=str), secrets))
        # The file's own contents, parsed the way production parses them: the
        # result file holds the raw JSON document, never a CLI envelope, so it is
        # `json.loads` and not `provider.parse_output`. Assertion 3 accepts
        # either channel, because production does.
        file_parsed = None
        if file_text.strip():
            try:
                file_parsed = json.loads(file_text)
            except ValueError as exc:
                file_parsed = {"error": f"file is not JSON: {exc}"}
        evidence["file_parsed"] = json.loads(
            redact(json.dumps(file_parsed, default=str), secrets))
        evidence["classification"] = provider.classify_failure(out or "", err or "",
                                                               code or 0)
        # `.pattern`, not the object: these are COMPILED regexes (base.py types
        # them `Tuple[Pattern[str], ...]`), and `json.dumps` cannot serialize
        # one. kimi declares an empty tuple so this never fired; qwen-code is the
        # first adapter on this lane with a non-empty marker set, and the whole
        # evidence file -- every assertion's input -- was lost to the TypeError.
        # THE SPEND RECORD FOR A VENDOR WITH NO USAGE ENDPOINT. Assertion 6
        # reads OpenRouter's `GET /api/v1/key` for a before/after credit window;
        # cursor publishes no such surface (`status`/`about --format json` carry
        # auth state and a subscription tier, never usage). What it does carry is
        # a per-run `usage` block in its own envelope — which `parse_output`
        # discards — so the adapter's own `parse_usage` is read here and the row
        # records real token counts instead of a fabricated cost. `usage_support`
        # says how far to trust it; a price is never derived from token counts,
        # because that needs a per-model price table this repo does not own.
        try:
            usage = provider.parse_usage(out or "", err or "")
        except Exception as exc:  # noqa: BLE001
            usage = {"error": f"{type(exc).__name__}: {exc}"}
        evidence["usage"] = json.loads(redact(json.dumps(usage, default=str), secrets))
        evidence["usage_support"] = getattr(provider, "usage_support", None)

        evidence["exit_zero_markers"] = [
            getattr(marker, "pattern", str(marker))
            for marker in (getattr(provider, "exit_zero_failure_markers", ()) or ())
        ]
    finally:
        udp.shutdown()
        tcp.shutdown()

    # `default=str` is a backstop, not laziness: this file IS every assertion's
    # input, so one unserializable field must degrade to a string rather than
    # raise and leave the lane with no evidence at all to grade -- which is
    # exactly what a compiled-regex marker did before the fix above.
    (args.out / "live-provider.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return 0 if evidence.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
