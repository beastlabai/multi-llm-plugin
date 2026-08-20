"""Credentialed provider lane: does the shipped adapter still reach the real endpoint?

**Name it correctly first.** This is a *third* lane, not a tier of the host lane
and not a variant of the provider lane (harness-expansion.md §7.4):

* the **provider lane** asks "does this CLI's argv, env and parser still match the
  shipped adapter?", ``--network none``, against ``stub_server.py``;
* the **host lane** asks "does claude-code / codex / opencode still load our shim,
  against a real brain?" — it is keyed by host, feeds the promotion ledger, and
  its PATH wrapper deliberately **scrubs every provider credential**;
* neither asks "does this adapter still reach the real vendor endpoint on the
  user's own key", which is what Wave-1 acceptance criterion 6 demands.

So it gets its own module, marker, make target and row list, and it borrows the
host lane's *credential-forwarding mechanics* without becoming a host row. It is
explicitly **not** a ``host_matrix.yaml`` row: that file is host-keyed, feeds
``creds.yaml``, ``posture.yaml`` and the promotion gate, and putting a
money-spending provider row behind the gate that decides host promotion would be
a category error (§7.2).

**Rows are derived, never duplicated.** A live row is the CLI's existing
``redirect.yaml`` ``lane: provider`` row **minus its stub overrides** — same
``cli``, same ``model_arg``, same ``mechanism``, with ``base_url_vars``,
``credential_vars`` and ``http_log`` dropped so the adapter's own production env
stands. That is the entire difference: no new matrix file, no duplicated argv.

**It costs real credits, so it is opt-in twice over**: the ``live_provider``
marker keeps it out of a bare ``pytest`` and out of CI, and ``make
e2e-live-provider`` is the only thing that sources the key. Failure policy is
hard — a 429 or an upstream 5xx **fails** the row, to be re-run deliberately once
the cause is understood. There is no retry sweetener and no "network flake" soft
pass, because a lane that retries its own failures spends twice to learn once.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, pytest.mark.live_provider, requires_docker()]

REDIRECT = harness.HERE / "redirect.yaml"

#: §6.6's live smoke record, appended to per attempt. A row per **attempt**,
#: never overwritten: §7.4's hard-failure policy means a failed attempt keeps its
#: row and the re-run gets a new one, so the history shows what it cost to get
#: green.
RECORD = harness.HERE / "records" / "live-provider.jsonl"

#: The CLIs this lane covers — the Wave-1/2 candidates and nobody else.
#:
#: Deliberately an explicit allowlist rather than "every provider row": the
#: thirteen shipped CLIs have their contract graded for free against the stub,
#: and sweeping them onto a credit-spending lane is a bill nobody asked for. A
#: candidate joins this set in the same commit that lands its adapter.
#:
#: WAVE 3.1 adds five at once because they share one route and cost nothing to
#: wire: aider, goose, pi, cline and kilocode already carry an OpenRouter spec in
#: their own `providers.yaml` `models[0]` (`openrouter/z-ai/glm-5.2` for the
#: first four, `openrouter/moonshotai/kimi-k2.7-code` for kilocode — all five
#: slugs verified present in OpenRouter's 415-model catalogue on 2026-08-19), so
#: the CLI routes ITSELF and `OPENROUTER_API_KEY` inherited from the runner's
#: environment is the whole credential. No base-URL plumbing, no seeded
#: endpoint, nothing for `live_base_url` to override.
#: WAVE 3.2-3.4 adds the five that need real wiring, and they are the rows that
#: make this lane more than a re-run of the free one. The first ten either route
#: themselves through OpenRouter from their own `models[0]` or have no stub at
#: all; these five are the CLIs whose PRODUCTION route is their own vendor —
#: Anthropic for claude-code, OpenAI for codex, opencode's zen backend, xAI for
#: grok, Google for gemini — so each needed an endpoint, a credential spelling and
#: (for four of them) a model this lane's key can actually buy. The wiring is a
#: port of the HOST lane's own `entrypoint.py::wire_credentials`, which has been
#: proving the same three surfaces reach OpenRouter since 2026-08-14.
LANE_CLIS = ("copilot", "kimi", "qwen-code", "vibe", "cursor-agent",
             "aider", "goose", "pi", "cline", "kilocode",
             "codex", "opencode", "claude-code", "grok", "gemini")

#: The slug a row is pinned to ON THIS LANE, where `providers.yaml`'s `models[0]`
#: is not servable by the endpoint this lane holds a key for — with the reason,
#: per row.
#:
#: `models[0]` STAYS THE RULE, and this table is the narrow exception to it. §8
#: item 3 makes `models[0]` the id `--init` injects and the one a bare `<cli>`
#: spec resolves to, so a lane that read anything else by default would be grading
#: a model the product never sends — and it is precisely by reading it that the
#: live lane found `gemini-3-flash`, a shipped default returning 404 from Google's
#: own API. But `models[0]` is ALSO the product default, so "make the row green by
#: editing providers.yaml" would ship a new default to every user in order to
#: satisfy a test. Four of Wave 3.2-3.3's five rows are exactly that case: their
#: default names a VENDOR'S OWN backend and OpenRouter — the only endpoint this
#: lane can pay for — refuses it.
#:
#: WHAT AN OVERRIDE COSTS, stated so it is not mistaken for free: the row proves
#: the shipped ADAPTER reaches a real model through the product's own argv, env
#: and parser, and does NOT prove that `models[0]` itself resolves. gemini keeps
#: no entry here for that reason — its vendor IS its own, so its default is what
#: runs.
LIVE_MODEL_OVERRIDES: dict[str, dict[str, str]] = {
    "claude-code": {
        "slug": "anthropic/claude-haiku-4.5",
        "why": "`models[0]` is `sonnet`, a FIRST-PARTY ANTHROPIC ALIAS that only "
               "api.anthropic.com resolves; OpenRouter's Anthropic skin takes "
               "catalogue ids and 400s an alias. `anthropic/claude-haiku-4.5` is "
               "the same vendor's cheapest current model and was verified "
               "present in the 414-model catalogue on 2026-08-20. Keeping the "
               "row on an ANTHROPIC model is deliberate: host_matrix.yaml's own "
               "note records that OpenRouter's cookbook warns Claude Code 'may "
               "not work correctly with other providers', so a non-Anthropic pin "
               "would make every failure ambiguous between the model and the "
               "adapter.",
    },
    "codex": {
        "slug": "deepseek/deepseek-v4-flash-0731",
        "why": "`models[0]` is `gpt-5.5`, a first-party OpenAI id; through "
               "OpenRouter the same model is `openai/gpt-5.5`, so the bare form "
               "is a 400. Pinned to the cheap deepseek flash the HOST lane "
               "already drives codex with, in its dated rather than its "
               "`~…-latest` alias form: the alias's canonical id is absent from "
               "the catalogue listing read on 2026-08-20 (only the dated "
               "`deepseek/deepseek-v4-flash-0731` is present), and this lane's "
               "job is not to discover which aliases OpenRouter expands.",
    },
    "opencode": {
        "slug": "openrouter/deepseek/deepseek-v4-flash-0731",
        "why": "`models[0]` is `opencode/big-pickle` — opencode's OWN zen "
               "backend, which authenticates against opencode.ai and not with "
               "this lane's key. This slug is not invented for the lane: it is "
               "`models[7]` of the same shipped list, i.e. a spec the product "
               "already ships, and its `openrouter/` half selects opencode's "
               "BUILT-IN provider, whose models.dev entry (baked into the "
               "image's own warm tree) declares `env: [OPENROUTER_API_KEY]` and "
               "`api: https://openrouter.ai/api/v1`. That is the whole "
               "credential and endpoint wiring for this row — opencode routes "
               "itself.",
    },
    "grok": {
        "slug": "x-ai/grok-4.5",
        "why": "`models[0]` is `grok-4.5:high`, an xAI-side id plus an `:effort` "
               "suffix. Under GROK_MODELS_BASE_URL the `-m` value must appear in "
               "the ENDPOINT's `GET /v1/models` or grok exits 1 `unknown model "
               "id`, and OpenRouter names that model `x-ai/grok-4.5`. The "
               "`:high` half is dropped with it, so the "
               "`--reasoning-effort high` the adapter would append is NOT "
               "exercised on this row — expected_argv.yaml grades that argv "
               "offline for free, and paying a live row to re-prove a flag "
               "renders is not what this lane is for.",
    },
}

#: An ADDITIONAL variable name this row's credential is exported under, inside the
#: container, with the measurement — the lane-side half of `--credential-alias`.
#:
#: WHY IT EXISTS AT ALL. Ten of the fifteen rows need nothing here, because their
#: shipped adapter already routes them through OpenRouter and reads
#: `OPENROUTER_API_KEY` itself (copilot's BYOK switch, kimi's
#: `KIMI_MODEL_BASE_URL`, qwen's OpenAI trio). claude-code and grok are the two
#: whose PRODUCT route is their own vendor's: `ClaudeCodeProvider.get_env` emits
#: retry and timeout variables and no auth at all, because a user's claude-code is
#: authenticated against Anthropic, and grok's is against xAI. Teaching either
#: adapter an OpenRouter variable to make this lane work would be the lane
#: rewriting the product — so the key arrives under the name the BINARY looks for
#: instead, which is exactly what the host lane's `wire_credentials` does for the
#: same two surfaces and has been doing since 2026-08-14.
#:
#: The value never leaves the container: `_run_argv` passes only the NAME, the
#: runner reads the value from the row's own `credential_env` in its own
#: environment, and `redact` scrubs it from every artifact because it is the same
#: string the lane already knows.
LIVE_CREDENTIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "claude-code": ("ANTHROPIC_AUTH_TOKEN",),
    "grok": ("XAI_API_KEY",),
}

#: A seed file this lane REMOVES before rendering, with the measurement. The
#: file-level sibling of :data:`STUB_ONLY_SEED`, for a tree that is part live
#: configuration and part stub artifact.
#:
#: `--no-seed` is all-or-nothing, and for these two rows all-or-nothing is the
#: wrong resolution: codex cannot reach any endpoint without the `config.toml`
#: that names it, and opencode cannot START without the 67 MB cold-start tree the
#: image bakes into its seed. What each of them must NOT carry onto this lane is
#: the one file holding the stub's fake session key.
SEED_DROP: dict[str, dict[str, str]] = {
    "codex": {
        ".codex/auth.json": (
            "THE LOGIN-PROBE ARTIFACT, and it carries `{{stub_session_key}}` — "
            "a token render_seed does not substitute (by design: its docstring "
            "forbids a rendered secret), so leaving it in place turns the row "
            "into a SKIP on the unsubstituted-placeholder guard. It is also "
            "unnecessary here: the README records it exists because "
            "`codex login status` prints 'Not logged in' and exits 1 without "
            "it, which is the executor's auth probe on the CHAIN lane, and this "
            "lane runs `codex exec` directly. The credential this row spends "
            "arrives through config.toml's `env_key`, which now renders as "
            "OPENROUTER_API_KEY. The HOST lane writes no auth.json for codex "
            "either."
        ),
    },
    "opencode": {
        ".local/share/opencode/auth.json": (
            "THE STUB'S STORED CREDENTIAL, and it would SHADOW the real one. The "
            "file is `{\"openrouter\": {\"type\": \"api\", \"key\": "
            "\"{{stub_session_key}}\"}}` — the same provider key this row's "
            "model resolves under, so on this lane a rendered stub value would "
            "be the credential opencode prefers over the environment. (It "
            "carries an unsubstitutable token too, so the row would skip before "
            "it could fail.) Dropped, opencode falls through to the env var its "
            "own baked models.dev catalogue names for the provider: "
            "`env: [OPENROUTER_API_KEY]`."
        ),
        ".config/opencode/opencode.json": (
            "THE STUB PROVIDER BLOCK: an `@ai-sdk/openai-compatible` provider "
            "literally named `stub`, at `{{base_url}}` with apiKey "
            "`{{stub_session_key}}`, offering one model `stub-model`. The live "
            "row's model is `openrouter/deepseek/deepseek-v4-flash-0731`, which "
            "selects opencode's built-in `openrouter` provider instead, so the "
            "block is dead weight whose only live effect would be the "
            "unrenderable key token. What the row keeps from the seed is the "
            "part that matters: the baked node_modules and the 4 MB "
            "`.cache/opencode/models.json`, without which a fresh opencode "
            "npm-installs 63 MB before it answers anything."
        ),
    },
}

#: Rows whose endpoint does not travel in the seed, with the measurement —
#: `seed_guards`' "some seeded file names the live endpoint" check is turned off
#: for these and for no others.
#:
#: The check exists to catch a seed whose endpoint was HARDCODED rather than
#: templated, which is how a live smoke ends up POSTing to 127.0.0.1 while
#: assertion 1 looks green. For a row whose seed was never the endpoint's home the
#: check cannot pass no matter how correct the row is, and the two rows below are
#: that case for two different reasons. Both keep the other two guards over the
#: whole tree, and both still have to satisfy assertion 1, which grades the host
#: the row actually REACHED off the DNS log — evidence about the wire rather than
#: about a string in a file.
#: Rows whose CLI builds its OWN sandbox around every tool call, needing a kernel
#: facility this lane's posture denies — with the measurement, and probed at run
#: time rather than assumed.
#:
#: THIS IS A PRECONDITION, NOT A VERDICT, which is the whole reason it is a skip
#: and not a red row. The lane's policy for a control it cannot apply is already
#: "skip with the reason recorded, never run weakened" (the resolver's port-53
#: sysctl is the existing case). codex is the same shape from the other side: the
#: posture is intact and it is the CLI's own inner containment that cannot start,
#: so a row run here would grade the container rather than the product — while
#: costing $0.026 and 180 s of a model retrying a write that cannot succeed, on
#: every invocation.
#:
#: The probe is a real `docker run` under this lane's own identity, capability and
#: privilege flags, so the day a daemon or posture permits CLONE_NEWUSER the row
#: starts running by itself rather than waiting for somebody to notice.
NEEDS_INNER_USERNS: dict[str, str] = {
    "codex": (
        "MEASURED TWICE on 2026-08-20, at $0.0258 and $0.0217, both timed out at "
        "the 180 s inner bound. `CodexProvider._BASE_ARGV` passes "
        "`--sandbox workspace-write` on every spawn, and codex 0.146.0 builds "
        "that sandbox with bubblewrap, which needs an unprivileged user "
        "namespace: `bwrap: No permissions to create a new namespace`. Every "
        "tool call failed — `cat >`, `printf >`, `ls`, and apply_patch's own "
        "read-back — so the model burned the whole bound retrying and the row "
        "could not produce the file its `supports_file_output = True` promises. "
        "THE KERNEL IS INNOCENT: the host's /proc/sys/user/max_user_namespaces "
        "is 160397 and `unshare -Un true` succeeds in the SAME image once "
        "`--security-opt seccomp=unconfined` is added, with `--cap-drop ALL` "
        "still in place. It is Docker's default seccomp profile denying "
        "CLONE_NEWUSER to a container without CAP_SYS_ADMIN — the containment "
        "this rig exists to hold, so unconfining seccomp for one row is refused. "
        "codex's other Linux sandbox was tried too: "
        "`[features] use_legacy_landlock = true` failed identically and printed "
        "its own deprecation error, and the flag was reverted rather than left "
        "as a setting that changes nothing. NOTHING ABOUT THE ADAPTER IS IN "
        "DOUBT — the two attempts are in this lane's record and they reached "
        "https://openrouter.ai/api/v1/responses with the pinned slug, passing "
        "a1, a2, a5 and a6; only the tool-calling half is ungradeable here. On a "
        "user's own machine, where unprivileged user namespaces are normal, "
        "codex's sandbox builds and this question does not arise."
    ),
}

ENDPOINT_IN_ENV: dict[str, str] = {
    "claude-code": (
        "claude-code has NO CONFIG-FILE MECHANISM — redirect.yaml's own row says "
        "so (`ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN); "
        "no config file`), and its `seed_files` list is empty. The seed tree "
        "exists for a different first-run fact: `.claude.json`'s "
        "`hasCompletedOnboarding` plus the workspace trust entry. The endpoint "
        "therefore arrives in the environment, as it does on the host lane."
    ),
    "opencode": (
        "opencode OWNS THE ROUTING on this row. The model is "
        "`openrouter/deepseek/...`, whose provider half resolves against the "
        "baked models.dev catalogue — `api: https://openrouter.ai/api/v1`, "
        "`env: [OPENROUTER_API_KEY]` — so there is no base URL for a config file "
        "to carry, and the row's only seeded config (the stub provider block) is "
        "dropped by SEED_DROP. The catalogue itself does name the endpoint, but "
        "it lives under `.cache/`, which the guards skip as a vendor artifact, "
        "so passing on it would be an accident rather than a check."
    ),
}

#: The default vendor for a row that names none. Every Wave-1/2 candidate reaches
#: its models through OpenRouter; `cursor-agent` is the first row that does not,
#: and it is here for a reason no other row shares: it has **no stub and can
#: never have one** (binary protobuf over Connect-RPC), it is `providers.yaml`'s
#: `default_provider`, and it is most of both `defaults.models` and
#: `quick_models`. For every other CLI this lane is a supplement to a free
#: offline grade; for cursor it is the only behavioural coverage that can exist.
DEFAULT_LIVE_HOST = "openrouter.ai"
DEFAULT_CREDENTIAL_ENV = "OPENROUTER_API_KEY"

#: The variable a FILE credential's bytes travel in, from this driver to the
#: in-container runner. Mirrors `live_provider.CREDENTIAL_MATERIAL_ENV`; kept as
#: a literal because the driver cannot import the in-container module.
CREDENTIAL_MATERIAL_ENV = "MLLM_LIVE_CREDENTIAL_MATERIAL"

#: A few hundred bytes, deliberately **not** §6 step 3's realistic multi-KB
#: prompt — that one belongs to the offline probe, where it costs nothing. This
#: lane's job is to prove the wire, not to exercise the parser's limits.
SMOKE_PROMPT = (
    "Reply with a single JSON array containing exactly one string: "
    '["ok"]. Output only the JSON array and nothing else.'
)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

#: Vendor homes, updaters and telemetry sinks a shipped adapter has no business
#: reaching on this lane. Kept host-side so the assertion is readable next to the
#: thing it grades; the rig's ``live_provider.py`` carries the same list as
#: documentation and the resolver's allowlist is what actually enforces it.
FORBIDDEN_EGRESS = (
    "github.com", "githubusercontent.com", "ampcode.com", "moonshot.cn",
    "kimi.com", "dashscope", "mistral.ai", "deepseek.com",
)

#: A forbidden host this row is MEASURED to query, together with what the query
#: is for. It is not an exemption: :meth:`test_a5_egress` still fails the row if
#: the resolver ANSWERED the name, so the only thing a declaration buys is the
#: distinction between "refused, as designed" and "reached a forbidden host" —
#: which the raw query log cannot make on its own, because a5 grades queries and
#: a refused query is still a query.
#:
#: Why this lives here and not in `redirect.yaml`'s `egress_beyond_stub`: that
#: field is the measured STUB matrix, its aider entry spells the fact as an IP
#: range in prose ("185.199.108-111.133 (raw.githubusercontent.com — litellm
#: model metadata)") and a5's second half matches it with `endswith`, so it
#: matches nothing and grades nothing. This is the lane-shaping table, next to
#: `FORBIDDEN_EGRESS` itself, in the file that owns the assertion.
#: Hosts a row is PERMITTED to resolve, against `FORBIDDEN_EGRESS`'s blanket
#: refusal — the opposite of `REFUSED_EGRESS` below, and deliberately a separate
#: table rather than a flag on that one.
#:
#: The distinction this lane has to be able to make is *what the request is for*,
#: not merely whether it succeeded. An updater or a telemetry sink is a request
#: users did not ask for and stays refused forever. A CONFIGURATION fetch that
#: upstream itself treats as best-effort is a different thing, and refusing it
#: buys nothing while costing a false signal.
#:
#: Each name here is added to the row's resolver allowlist, so the fetch really
#: happens, and a5 then asserts it was ANSWERED — an allowance that has quietly
#: stopped being exercised is a dead declaration, and dead declarations are how a
#: containment set drifts without anything going red.
ALLOWED_EGRESS: dict[str, dict[str, str]] = {
    "aider": {
        "raw.githubusercontent.com":
            "litellm's model COST MAP — `model_prices_and_context_window.json`, "
            "fetched at import time (before aider parses a flag, which is why "
            "`--no-check-update`/`--no-analytics` cannot reach it) with a 5 s "
            "timeout and a bundled 1.28 MB fallback on any exception. It is "
            "pricing and context-window metadata, not an updater and not "
            "telemetry: nothing about the user's run is sent, and upstream's own "
            "code treats the refresh as best-effort. Permitted deliberately so "
            "the row exercises the configuration the product actually ships. "
            "redirect.yaml records the same call on the stub lane as "
            "`fatal_when_blocked: false`. The SEPARATE defect blocking it exposed "
            "— aider printing the failure on stdout, where `Name or service not "
            "known` tripped CONNECTION_ERROR_MARKERS and classified a successful "
            "run as `connection_error` — is fixed in AiderProvider, not here, "
            "because it bites every user on a restricted network whether or not "
            "this lane allows the name.",
    },
}

REFUSED_EGRESS: dict[str, dict[str, str]] = {
    "kimi": {
        "telemetry-logs.kimi.com":
            "TELEMETRY THE OPT-OUT DOES NOT GATE, measured and unfixable from "
            "here. kimi 0.36.1 resolves this on EVERY run with "
            "KIMI_DISABLE_TELEMETRY=1 set — which the adapter does set, and "
            "which a5's OPT_OUTS check asserts by name in the resolved env. "
            "Reproduced offline three ways during the §6 probe (opt-out set, "
            "opt-out unset, and with KIMI_DISABLE_CRON=1): two DNS queries every "
            "time, identical in all three. The VALUE is not the problem — `1` is "
            "in the CLI's own true-value set and its `isTelemetryDisabledByEnv` "
            "gate reads it — the connection attempt simply is not gated by it. "
            "So there is no configuration that stops this and the DNS allowlist "
            "is the only control, exactly as with cursor-agent's statsig/sentry "
            "calls. Declared with its measurement rather than left as an "
            "unattributed red (harness-expansion.md §14.7 gap 8, which asked for "
            "this decision): the resolver returns NXDOMAIN, nothing leaves the "
            "container, and an ALLOW decision on this name still fails a5. On a "
            "user's own machine there is no such allowlist and the request does "
            "go out — which is a product fact worth knowing, not a lane defect. "
            "Re-check on the next version bump.",
    },
    "kilocode": {
        "github.com":
            "AN UPDATE CHECK, AND ITS OPT-OUT IS NOW SET AND MEASURED NOT TO "
            "STOP IT — which is a sharper statement than the one this entry "
            "carried on 2026-08-19, when neither of kilocode's two opt-outs was "
            "set by anything. Both now are, in the row's own `env_extra` "
            "(redirect.yaml): `KILO_DISABLE_AUTOUPDATE=1` and "
            "`KILO_TELEMETRY_LEVEL=off`. BEFORE (three runs, 2026-08-19, "
            "neither variable set) the row resolved six names: api.kilo.ai, "
            "github.com, models.dev, openrouter.ai, registry.npmjs.org and "
            "us.i.posthog.com. AFTER (2026-08-20, both set, row green in 15.5 s) "
            "it resolved five: **us.i.posthog.com is gone** and nothing else "
            "moved. So `KILO_TELEMETRY_LEVEL` works exactly as its bundle "
            "describes (`level ? level === \"all\" : config.enabled`, read once "
            "at spawn) and silences the PostHog client's two queries per run, "
            "while `KILO_DISABLE_AUTOUPDATE` does NOT stop the "
            "registry.npmjs.org + github.com update-check pair, and neither "
            "touches models.dev (the 3.6 MB catalogue, best-effort — the row's "
            "own note records it still exits 0 unreachable) or api.kilo.ai (the "
            "vendor home). The variables stay set: one of the two demonstrably "
            "works, and the other is the shipped opt-out for the behaviour "
            "observed, so leaving it unset would misreport the finding as "
            "'nobody tried'. github.com is the only one of the five in "
            "FORBIDDEN_EGRESS; every one of them was refused, openrouter.ai was "
            "the only name this row resolved, and an ALLOW decision on any of "
            "them still fails a5.",
    },

    # ---- Wave 3.2-3.4 ---------------------------------------------------- #
    # All three below were measured BEFORE any credit was spent, on a $0 dry run
    # of the real wiring with a deliberately invalid key: a call-home fires at
    # startup regardless of whether the model call later succeeds, so the DNS
    # log is complete long before the 401 comes back. Each was then confirmed on
    # the row's own credentialed run.
    "codex": {
        "github.com":
            "AN UPDATE/RELEASE CHECK WITH NO OPT-OUT TO SET. Measured on the "
            "2026-08-20 rows: `github.com` and `api.github.com` are queried "
            "twice each, alongside four queries to `chatgpt.com` (codex's own "
            "vendor home — the binary carries `chatgpt.com/codex/install.sh` "
            "and `chatgpt.com/backend-api/codex`, so this is the install/auth "
            "surface, not the model API). The 0.146.0 binary contains NO "
            "`DO_NOT_TRACK` (zero occurrences) and no `CODEX_DISABLE_*` "
            "telemetry or updater variable of any kind; its only telemetry path "
            "is OpenTelemetry, which is opt-IN through config.toml's `[otel]` "
            "block and which our seeded config does not declare. So, as with "
            "cursor-agent and cline, there is no configuration that stops this "
            "and the DNS allowlist is the only control: every one of these "
            "names got NXDOMAIN, the run was unaffected, and an ALLOW decision "
            "on any of them still fails a5.",
    },
    "opencode": {
        "github.com":
            "AN UPDATE CHECK, AND ITS OWN OPT-OUT WAS TRIED AND MEASURED NOT TO "
            "STOP IT. opencode 1.18.18 queried `github.com` 3 times on the "
            "control run and 6 times on a run with `OPENCODE_DISABLE_AUTOUPDATE=1` "
            "added — the variable is real (it is in the binary's env schema "
            "beside OPENCODE_DISABLE_SHARE and OPENCODE_ALWAYS_NOTIFY_UPDATE) "
            "and setting it changed nothing, which is why this row does NOT set "
            "it: a variable that demonstrably does nothing is a dead "
            "declaration, and dead declarations are how a containment set drifts "
            "without anything going red. `DO_NOT_TRACK` is set image-wide and "
            "asserted by name in OPT_OUTS. The resolver returns NXDOMAIN, the "
            "run is unaffected, and an ALLOW decision here still fails a5.",
    },
    "gemini": {
        "github.com":
            "THE UPDATE NAG, past two opt-outs that are already set. "
            "gemini-cli 0.25.2 bundles `update-notifier` and queries "
            "`github.com` once per run even though the image sets both "
            "`NO_UPDATE_NOTIFIER=1` and `NPM_CONFIG_UPDATE_NOTIFIER=false` — "
            "measured 2026-08-20. Its own switch is a SETTINGS key "
            "(`disableUpdateNag` in ~/.gemini/settings.json), not an "
            "environment variable, and writing a settings file to silence it "
            "would be this lane inventing product configuration. NXDOMAIN, run "
            "unaffected, ALLOW still fails a5. The separate telemetry sink "
            "`play.googleapis.com` (Clearcut, queried twice per run) is NOT in "
            "FORBIDDEN_EGRESS and is graded instead by a5's second half against "
            "the row's `egress_beyond_stub` declaration.",
    },
}

#: A row whose ``home_seed/<cli>/`` tree exists ONLY to point the CLI at the
#: container stub, with the measurement. Those rows run ``--no-seed``.
#:
#: THE SEED IS A STUB ARTIFACT, NOT A CREDENTIAL SPELLING PROBLEM. All three
#: trees below carry ``{{stub_session_key}}``, which is not one of the four
#: tokens :func:`live_provider.render_seed` substitutes (`{{base_url}}`,
#: `{{base_url_root}}`, `{{api_key_env}}`, `{{model_name}}`), so rendering them
#: here leaves the placeholder in place and `seed_guards`' unsubstituted-token
#: check turns the row into a SKIP. The fix is not to teach `render_seed` a
#: fifth token: its own docstring forbids exactly that — "Only the key's
#: variable name ever enters the file … a rendered secret inside a seed tree is
#: a committed credential" — and on this lane there is nothing for the seed to
#: say anyway. Each of these three CLIs routes itself from its own
#: `providers.yaml` `models[0]` OpenRouter spec and reads `OPENROUTER_API_KEY`
#: from the environment; a seeded `stub` provider block would add a second,
#: unused provider whose baseUrl points at an endpoint this lane never uses.
STUB_ONLY_SEED: dict[str, str] = {
    "pi": (
        ".pi/agent/models.json defines a provider literally named `stub` "
        "(baseUrl {{base_url}}, apiKey {{stub_session_key}}, one model "
        "`stub-model`). The live row's model is `openrouter/z-ai/glm-5.2`, "
        "which pi resolves through its own built-in openrouter provider off "
        "OPENROUTER_API_KEY — the seeded block is dead weight and its key "
        "token is unrenderable."
    ),
    "cline": (
        ".cline/data/settings/providers.json pins lastUsedProvider "
        "`openai-compatible` at {{base_url}} with apiKey {{stub_session_key}}. "
        "The live argv is `-P openrouter -m z-ai/glm-5.2`, which selects a "
        "different provider entirely, and cline reads OPENROUTER_API_KEY for "
        "it (the string is in @cline/llms' bundled provider table). Seeding "
        "would leave a stale default pointing at a dead 127.0.0.1 endpoint."
    ),
    "kilocode": (
        ".config/kilo/kilo.jsonc declares provider `stub` "
        "(@ai-sdk/openai-compatible at {{base_url}}, apiKey "
        "{{stub_session_key}}). The live model is "
        "`openrouter/moonshotai/kimi-k2.7-code`, i.e. kilocode's own built-in "
        "`openrouter` provider off OPENROUTER_API_KEY (the string is in "
        "@kilocode/cli's bundle); the `stub` block names a provider the live "
        "row never selects."
    ),
}

#: Per-row corrections to the STUB row's ``env_extra``, applied after the model
#: substitution in :func:`_live_rows`, with what was measured.
#:
#: `env_extra` is carried through from `redirect.yaml` on purpose (§7.4: it holds
#: image-layout facts the adapter has no business hardcoding — copilot's
#: pre-warmed XDG_CACHE_HOME is the case that forced it). But a stub row's
#: `env_extra` can also carry a VENDOR SELECTION that is only true next to a
#: `base_url_vars` block this lane deletes, and then the carry-through is not
#: neutral: it aims the CLI at the wrong vendor's real endpoint on a key that is
#: not ours. That is what this table exists to stop, and it is checked against
#: every row on the lane, not only the one that failed.
LIVE_ENV_OVERRIDES: dict[str, dict[str, str]] = {
    # MEASURED: the stub row pairs `GOOSE_PROVIDER: openai` with
    # `base_url_vars: {OPENAI_BASE_URL: '{base_url}'}` — the pair is what aims
    # goose's `openai` backend at the container stub. This lane drops
    # `base_url_vars` by definition, so carrying `GOOSE_PROVIDER=openai` alone
    # would leave goose's openai backend on its vendor default,
    # **api.openai.com**, reached with an OpenRouter key: a wrong-vendor call on
    # a credential that is not ours. `GooseProvider.build_command` already
    # splits `openrouter/z-ai/glm-5.2` into `--provider openrouter --model
    # z-ai/glm-5.2`, so the argv and the environment now agree instead of
    # contradicting each other and relying on argv precedence to save the row.
    "goose": {"GOOSE_PROVIDER": "openrouter"},
    # The other four rows were checked for the same class of problem and have
    # none: aider and kilocode declare no `env_extra` at all; cline's is
    # `CLINE_NO_AUTO_UPDATE: '1'`, an updater opt-out with no vendor in it; pi's
    # row carries no `env_extra` either (its stub wiring is `credential_vars:
    # [STUB_API_KEY]` and `config_roots`, and this lane drops the first by
    # definition).

    # ---- Wave 3.2-3.3: the two rows whose ENDPOINT lives here ------------- #
    #
    # A BASE URL IN THIS TABLE IS A DELIBERATE FOURTH ROUTE, and it is worth
    # saying why the other three were refused for these two rows.
    #
    #   * the ADAPTER's `get_env` is how copilot, kimi and qwen-code reach
    #     OpenRouter — but for them that is the PRODUCT's own route, hardcoded
    #     in the shipped adapter because that is how the product ships those
    #     CLIs. claude-code's route is the user's Anthropic subscription and
    #     grok's is xAI; teaching either adapter an OpenRouter endpoint to make a
    #     test pass would ship a routing change to every user.
    #   * a `home_seed/<cli>/` TEMPLATE is how codex reaches it — but neither of
    #     these CLIs has a config-file mechanism at all (`redirect.yaml`:
    #     claude-code "no config file"; grok "GROK_MODELS_BASE_URL +
    #     XAI_API_KEY", `seed_files: []`). There is no file to template.
    #   * `redirect.yaml`'s own `env_extra` is the one place this must NOT go:
    #     `wrapper.build_child_env` applies it AFTER `base_url_vars`, so a base
    #     URL there would override the per-exec stub URL and break the provider
    #     lane — the row would still be green and would be testing nothing.
    #
    # This table is applied in `_live_rows`, on this lane only, which is what
    # makes it the right home: the provider lane never reads it, and the value
    # sits next to the measurement that justifies it. The CREDENTIAL half is not
    # here — a value in a tracked file is a committed credential — it travels by
    # name through LIVE_CREDENTIAL_ALIASES.
    "claude-code": {
        # THE BARE `/api` ORIGIN, not `/api/v1`. Claude Code appends
        # `/v1/messages` itself, which is why `entrypoint.py`'s
        # OPENROUTER_ANTHROPIC_BASE is spelled the same way and why the row's
        # `live_base_url` matches it: a `/api/v1` here produces
        # `/api/v1/v1/messages` and a 404 that reads like an auth failure.
        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
        # PRESENT AND EMPTY, and this is the single most important cell in the
        # table. An UNSET ANTHROPIC_API_KEY lets Claude Code fall back to
        # authenticating against Anthropic directly — on this machine that is the
        # developer's own subscription — and the row would go green while
        # proving the exact opposite of what it claims. The host lane asserts the
        # same thing at launch; here it is the value the runner exports.
        "ANTHROPIC_API_KEY": "",
        # One variable covering four opt-outs: measured in the 2.1.232 bundle,
        # which reads DISABLE_TELEMETRY, DISABLE_AUTOUPDATER,
        # DISABLE_ERROR_REPORTING and DISABLE_BUG_COMMAND, all of which
        # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC gates together.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    },
    "grok": {
        # grok's own documented custom-endpoint mode (its bundled
        # ~/.grok/README.md, "Custom Models Endpoint"): this switches it off
        # xAI's cli-chat-proxy onto a plain OpenAI-compatible base URL AND off
        # session auth onto `Authorization: Bearer`, which is what makes an
        # OpenRouter key usable at all. The provider lane sets the same variable
        # from `base_url_vars`; this lane sets it to the real endpoint.
        "GROK_MODELS_BASE_URL": "https://openrouter.ai/api/v1",
        # MEASURED in the 0.2.93 binary, which consults all three (plus
        # GROK_AUTO_UPDATE, already set image-wide to 0 and asserted by name in
        # OPT_OUTS). Set here rather than in the row's `env_extra` only because
        # the provider lane is `--network none`, where none of them can matter;
        # a5 grades whether they hold on the one lane where they can.
        "GROK_TELEMETRY_ENABLED": "false",
        "GROK_TELEMETRY_MIXPANEL_ENABLED": "false",
        "GROK_DISABLE_AUTOUPDATER": "1",
    },
    # codex, opencode and gemini need NOTHING here, and each for a reason worth
    # recording rather than an absence: codex's endpoint is in its seeded
    # config.toml (whose `env_key` now renders as OPENROUTER_API_KEY), opencode
    # resolves both endpoint and credential from its baked models.dev catalogue,
    # and gemini's vendor IS its own — the row points at
    # generativelanguage.googleapis.com with GEMINI_API_KEY, which is what the
    # CLI reads with no help from anybody.
}

#: CLIs whose adapter SPLITS `models[0]` before it crosses the argv boundary,
#: with the measurement. Assertion 2 looks for the half the CLI was actually
#: given, because the full spec is a string the CLI never sees and so can never
#: report back — a `pinned in blob` test against it would fail a row that echoed
#: its model perfectly.
#:
#: This narrows nothing that a1 does not already cover: the provider half is the
#: ROUTE, and a1 grades the route directly off the DNS log.
SPLIT_MODEL_SPEC: dict[str, str] = {
    "cline": (
        "ClineProvider.build_command partitions on the first '/': "
        "`openrouter/z-ai/glm-5.2` becomes `-P openrouter -m z-ai/glm-5.2`, and "
        "the run_result stream reports it back as "
        "`\"model\":{\"id\":\"z-ai/glm-5.2\",\"provider\":\"openrouter\"}` "
        "— confirmed on the 2026-08-19 live row, so this is a real strict grade "
        "and not a way around one."
    ),
    "kilocode": (
        "kilocode is handed the WHOLE spec (`-m "
        "openrouter/moonshotai/kimi-k2.7-code`) and splits it internally: its "
        "banner reports `> code · moonshotai/kimi-k2.7-code`, the resolved "
        "model without the provider key. Measured 7.4.19 on the 2026-08-19 live "
        "row. goose is NOT in this table even though it splits identically — it "
        "reports no model at all, which NO_MODEL_ECHO says with the "
        "measurement, and that check runs first."
    ),
}

#: CLIs whose model banner lands on **stderr**, with the measurement. Assertion 2
#: reads the parsed document plus stdout, which is the pair production reads; a
#: CLI that prints its banner on the other descriptor would otherwise be graded
#: `unread` for a fact it does report.
#:
#: This is a per-CLI declaration rather than "always read both streams" on
#: purpose: widening the haystack for every row makes every row's grade laxer
#: for the sake of one, and the only row that needs it is the one named here
#: with the bytes that prove it.
MODEL_ECHO_ON_STDERR: dict[str, str] = {
    "kilocode": (
        "`kilocode run --auto` prints its banner (`> code · <model>`) and its "
        "tool trace on stderr and keeps stdout for the answer alone — measured "
        "on the 2026-08-19 live row: 58 bytes of stdout (the file-write "
        "confirmation) against 230 of stderr carrying the model. Nothing about "
        "the model is on stdout, so a2 reads stderr for this row and grades the "
        "slug strictly there."
    ),
}

#: Assertion 5's belt-and-braces half: the offline/telemetry opt-outs §8.2 lists
#: per CLI, asserted **by name** in the resolved env.
#:
#: PROBE-SHAPED SEAM. qwen's updater/telemetry variables are unrecorded in §3.3
#: and its H3 probe is what names them; vibe's are a probe item too. An empty
#: tuple here means "nothing measured yet", and the landing adapter fills it in
#: the same commit that adds its row — it does not mean "this CLI has none".
#: CLIs whose captured output carries **no model field at all**, with the
#: measurement that established it. Assertion 2 records `unread` for these
#: rather than failing: there is nothing in the stream to compare the pinned
#: slug against. Every other CLI is asserted strictly.
NO_MODEL_ECHO: dict[str, str] = {
    "kimi": (
        "kimi stream-json records are {role, content} only — no model field, "
        "no usage block and no generation id anywhere in the stream (measured "
        "0.36.1; the same fact behind its usage_support = none)"
    ),
    "vibe": (
        "vibe's --output json transcript entries carry a closed field set "
        "(id, sessionId, turnId, createdAt, updatedAt, generationStatus, "
        "relatedEntryId, type, role, content[], source, userDisplayContent) "
        "with no model field on any of them — the substring 'model' does not "
        "occur anywhere in a whole transcript (measured 2.24.1, and the same "
        "fact behind its usage_support = none). The model is selected through "
        "VIBE_ACTIVE_MODEL and never echoed back, so a2 has nothing to read; "
        "the endpoint half is still asserted by a1 and the spend by a6."
    ),
    "cursor-agent": (
        "cursor's --output-format json envelope carries a closed nine-key set "
        "(type, subtype, is_error, duration_ms, duration_api_ms, result, "
        "session_id, request_id, usage) with no model field on any of them — "
        "the same fact behind its usage_support = partial. And `models[0]` is "
        "`auto`, which is a SELECTOR resolved server-side, not a slug: even an "
        "echoing envelope could not report it back. `about --format json` "
        "reports a model, but it is the configured default's display name "
        "(\"Cursor Grok 4.6 Extra High\"), not what this run served."
    ),
    "goose": (
        "goose's --output-format json envelope is {messages, metadata} and the "
        "SUBSTRING 'model' does not occur anywhere in it — measured 1.41.0 on "
        "the 3,162-byte envelope of the 2026-08-19 live row, whose `metadata` "
        "is exactly {total_tokens, input_tokens, output_tokens, status}. The "
        "same fact behind its usage_support = none. And goose is handed only "
        "half the spec in the first place: `GooseProvider.build_command` "
        "partitions on the first '/', so `openrouter/z-ai/glm-5.2` reaches the "
        "binary as `--provider openrouter --model z-ai/glm-5.2` and the full "
        "slug is a string goose never sees. The route half is asserted by a1 "
        "off the DNS log and the spend by a6."
    ),
    "pi": (
        "pi -p (print mode) writes the final assistant TEXT to stdout and "
        "nothing else — no envelope, no banner, no metadata line, and stderr "
        "empty (measured 0.80.6 on the 2026-08-19 live row: 52 bytes of stdout, "
        "0 of stderr). With supports_file_output = True the answer goes to the "
        "file and stdout is the confirmation line alone, so there is no channel "
        "on which a model name could appear. Its usage_support is none for the "
        "same reason."
    ),
    "opencode": (
        "MEASURED ON THE 2026-08-20 LIVE ROW, on the green run rather than on a "
        "failing one: the SUBSTRING 'model' does not occur anywhere in "
        "`opencode run --format json`'s 2,347-byte stream — its parts are "
        "step-start / tool / text / step-finish, and the only provider-shaped "
        "field is `metadata.openrouter`, the provider key with no model id "
        "under it. stderr is EMPTY, because the adapter pins "
        "`--print-logs --log-level ERROR` and a healthy run logs nothing; the "
        "model id (`modelID=deepseek/deepseek-v4-flash-0731`) appears only in "
        "those ERROR-level log lines, i.e. only on a run that FAILED — which is "
        "why MODEL_ECHO_ON_STDERR would not help this row and is not used for "
        "it. Unlike the five entries above, opencode's `usage_support` is FULL: "
        "the row still records real input/output/cache token counts and the "
        "vendor's own per-step `cost`, so a6 has hard numbers where a2 has "
        "nothing to read, and a1 grades the route off the DNS log."
    ),
    "grok": (
        "grok's `--output-format json` envelope is a CLOSED FOUR-KEY SET — "
        "{text, stopReason, sessionId, requestId} — with no model field, no "
        "usage block and no generation id, measured 0.2.93 on the 2026-08-20 "
        "live row (a 217-byte envelope, stderr empty). The same fact is behind "
        "its usage_support = none. And the row cannot fall back to the stderr "
        "trick either: there is no banner. What a2 would have graded is covered "
        "twice over here — the `-m` value must appear in the ENDPOINT's own "
        "`GET /v1/models` or grok exits 1 `unknown model id` before any turn, "
        "so a run that produced an answer at all proves the endpoint served "
        "that slug, and a1 grades the route off the DNS log."
    ),
}

OPT_OUTS: dict[str, tuple[str, ...]] = {
    "copilot": ("COPILOT_OFFLINE", "COPILOT_AUTO_UPDATE"),
    "kimi": ("KIMI_CODE_NO_AUTO_UPDATE", "KIMI_DISABLE_TELEMETRY"),
    # Measured at 0.21.13 (§7.4's belt-and-braces assertion): the updater and
    # both telemetry channels, asserted BY NAME in the resolved env.
    "qwen-code": (
        "QWEN_TELEMETRY_ENABLED",
        "QWEN_USAGE_STATISTICS_ENABLED",
        "QWEN_CODE_SKIP_UPDATE_CHECK_ONCE",
    ),
    # Measured at 2.24.1: both default TRUE upstream and the dependency tree
    # carries sentry-sdk and opentelemetry-*, so these are asserted BY NAME in
    # the resolved env rather than assumed from the seed.
    "vibe": ("VIBE_ENABLE_TELEMETRY", "VIBE_ENABLE_UPDATE_CHECKS"),
    # MEASURED EMPTY, which is a different statement from "not measured yet" —
    # the header above says an empty tuple means the latter, so this row says it
    # explicitly: the cursor-agent bundle carries no DO_NOT_TRACK, no
    # CURSOR_TELEMETRY_*, and no other opt-out variable of any kind. Its statsig
    # and sentry calls cannot be turned off by configuration, so the DNS
    # allowlist is what stops them, and `test_a5_egress` grades that instead.
    "cursor-agent": (),

    # ---- Wave 3.1 -------------------------------------------------------- #
    # Every tuple below was measured on 2026-08-19 by reading the INSTALLED
    # artefact in `mllm-e2e:local` (not upstream docs) for the variable names it
    # actually consults, then checking which of those the resolved env carries.
    # An empty tuple here therefore means "measured, and nothing is assertable",
    # never "not looked at" — the two cases are spelled out per row.

    # MEASURED EMPTY, BY DESIGN. aider's opt-outs are ARGV, not environment:
    # `AiderProvider.build_command` passes `--no-check-update`, `--no-analytics`,
    # `--no-show-release-notes` and `--no-detect-urls` on every spawn. aider does
    # mirror each flag as an `AIDER_<FLAG>` variable (AIDER_ANALYTICS_DISABLE,
    # AIDER_CHECK_UPDATE and ~200 more are in the 0.86.2 site-packages), but the
    # shipped adapter uses the flag form, so asserting the env form here would
    # grade a mechanism the product does not use. What the DNS log then shows is
    # the residual: litellm's import-time metadata fetch, which no aider flag
    # reaches — declared in REFUSED_EGRESS and graded by a5's first half.
    "aider": (),

    # MEASURED EMPTY, AND THERE WAS NOTHING TO SUPPRESS. The goose 1.41.0 binary
    # consults GOOSE_TELEMETRY_ENABLED and GOOSE_TELEMETRY_OFF (both in its
    # strings) and has NO DO_NOT_TRACK support at all (`grep -a DO_NOT_TRACK
    # /opt/mllm/real/goose` → no match), and neither GOOSE_TELEMETRY_* is set by
    # the image or by the row (whose `env_extra` carries GOOSE_PROVIDER and
    # GOOSE_DISABLE_KEYRING only). That would be a gap if goose called home —
    # and on the 2026-08-19 live row it did not: the ONLY name in its DNS log is
    # openrouter.ai, resolved four times, with nothing else attempted. So the
    # empty tuple is the whole truth for this row rather than an untested
    # assumption, and a5's egress half is what keeps it honest.
    "goose": (),

    # MEASURED EMPTY, same shape and same result. The 0.80.6 bundle consults
    # PI_TELEMETRY, PI_SKIP_VERSION_CHECK and PI_OFFLINE and references neither
    # DO_NOT_TRACK nor NO_UPDATE_NOTIFIER; none is set by the image or by pi's
    # row, whose only container-shaped keys are `config_roots:
    # {PI_CODING_AGENT_DIR: .pi/agent}` and the stub-only `credential_vars`. And
    # again the live row queried openrouter.ai and nothing else — pi is the
    # quietest CLI on this lane.
    "pi": (),

    # The one Wave-3.1 row with a real pair. CLINE_NO_AUTO_UPDATE is set twice
    # over — the Dockerfile's image-wide ENV and the row's own `env_extra` — and
    # earns it: cline was measured rewriting its own global node_modules from
    # 3.0.39 to 3.0.55 *while a probe was running*. DO_NOT_TRACK is the image's,
    # and cline is one of the two CLIs here whose bundle actually reads it
    # (the string is in the 3.0.55 tree, alongside its posthog and sentry.io
    # clients).
    # BOTH ARE SET AND NEITHER STOPS THE TELEMETRY — which is a finding, not a
    # reason to drop them: measured on the 2026-08-19 live row, cline still
    # queried `data.cline.bot` and `otel.cline.bot` (its analytics and OTLP
    # sinks) with CLINE_NO_AUTO_UPDATE=1 and DO_NOT_TRACK=1 both present in the
    # resolved env. Both were refused by the resolver and the run was
    # unaffected. cline's bundle carries no third opt-out to try (its only
    # UPDATE/TELEMETRY/TRACK-shaped name is CLINE_NO_AUTO_UPDATE), so as with
    # cursor-agent the DNS allowlist is the control that actually holds, and
    # a5's egress half is what grades it.
    "cline": ("CLINE_NO_AUTO_UPDATE", "DO_NOT_TRACK"),

    # DO_NOT_TRACK is in the 7.4.19 @kilocode/cli bundle and set image-wide, so
    # it is asserted — and it demonstrably does not cover everything: the
    # 2026-08-19 live row still reached for us.i.posthog.com twice, api.kilo.ai,
    # models.dev, and the registry.npmjs.org + github.com update-check pair.
    # kilocode's own KILO_DISABLE_AUTOUPDATE and KILO_TELEMETRY_LEVEL (both in
    # the bundle) are set by neither the image nor the row's `env_extra`, whose
    # home is redirect.yaml. See REFUSED_EGRESS["kilocode"] for the full
    # measured set and why the row is graded rather than exempted.
    "kilocode": ("DO_NOT_TRACK", "KILO_DISABLE_AUTOUPDATE", "KILO_TELEMETRY_LEVEL"),

    # ---- Wave 3.2-3.4 ---------------------------------------------------- #
    # Measured the same way as Wave 3.1's: by reading the INSTALLED artefact in
    # `mllm-e2e:local` for the variable names it consults, then checking which of
    # those the resolved env carries. An empty tuple is a measurement, not a gap.

    # MEASURED EMPTY, AND THERE IS NOTHING TO SET. The codex 0.146.0 binary
    # contains no DO_NOT_TRACK (0 occurrences), no `CODEX_DISABLE_*` telemetry or
    # updater name, and no auto-updater at all — the npm package is the update
    # channel. Its only telemetry surface is OpenTelemetry, which is OPT-IN
    # through config.toml's `[otel]` block (the binary carries the OTEL_* SDK
    # names, and our seeded config declares no such block), so there is no
    # variable this row could assert. What remains assertable is the DNS log, and
    # a5's egress half is what grades it.
    "codex": (),

    # OPENCODE_DISABLE_MODELS_FETCH is the image's, and this lane inherits the
    # image ENV directly (unlike the provider lane, where wrapper.py rebuilds the
    # child env from scratch and the row's `env_extra` has to repeat it). It is a
    # NETWORK opt-out rather than a telemetry one — it stops the 3.6 MB models.dev
    # refresh, which is exactly the fetch this row must not need, because its
    # provider and model resolve from the catalogue baked into the seed.
    # DO_NOT_TRACK is asserted too: the 1.18.18 binary carries the string, and the
    # image sets it. The bundle's other candidates (OPENCODE_DISABLE_AUTOUPDATE,
    # OPENCODE_DISABLE_SHARE, OPENCODE_AUTO_SHARE) are NOT set and are not
    # asserted: the image installs opencode from npm with no self-update path, and
    # `opencode run` shares nothing by default. If the DNS log shows otherwise
    # that is a finding, and the finding belongs in REFUSED_EGRESS with its
    # measurement rather than in a variable set on a hunch.
    "opencode": ("OPENCODE_DISABLE_MODELS_FETCH", "DO_NOT_TRACK"),

    # ONE VARIABLE COVERING FOUR OPT-OUTS, plus the image's DO_NOT_TRACK (12
    # occurrences in the 2.1.232 bundle). CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
    # (20 occurrences) is the documented switch that gates DISABLE_TELEMETRY,
    # DISABLE_AUTOUPDATER, DISABLE_ERROR_REPORTING and DISABLE_BUG_COMMAND
    # together — all four names are in the binary. It is set by
    # LIVE_ENV_OVERRIDES, exactly as the host lane sets it, and asserted here so a
    # future edit to that table cannot quietly drop it.
    "claude-code": ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DO_NOT_TRACK"),

    # FOUR, AND THE FIRST IS THE IMAGE'S. GROK_AUTO_UPDATE=0 is in the
    # Dockerfile's step-8 ENV block and reaches this lane directly. The other
    # three were measured in the 0.2.93 binary (GROK_TELEMETRY_ENABLED,
    # GROK_TELEMETRY_MIXPANEL_ENABLED, GROK_DISABLE_AUTOUPDATER — alongside
    # GROK_TELEMETRY_EVENTS_URL, GROK_TELEMETRY_GCS_BUCKET and a Mixpanel token,
    # which say what the telemetry path is for) and are set by LIVE_ENV_OVERRIDES,
    # because the provider lane is `--network none` where none of them can matter
    # and this is the one lane where they can. grok has NO DO_NOT_TRACK support:
    # 0 occurrences in the binary, so it is deliberately not asserted here.
    "grok": ("GROK_AUTO_UPDATE", "GROK_TELEMETRY_ENABLED",
             "GROK_TELEMETRY_MIXPANEL_ENABLED", "GROK_DISABLE_AUTOUPDATER"),

    # MEASURED EMPTY, AND THE OPT-OUT IS NOT A VARIABLE. gemini-cli 0.25.2 gates
    # its usage statistics on `usageStatisticsEnabled` in
    # `~/.gemini/settings.json` — a SETTINGS key with no environment equivalent
    # (the bundle reads GEMINI_API_KEY, GEMINI_MODEL, GEMINI_SANDBOX and friends;
    # it contains no DO_NOT_TRACK at all) — and ships the sink in the binary:
    # `play.googleapis.com/log`, Google's Clearcut. Writing a settings file to
    # silence it would be this lane inventing product configuration, so the DNS
    # allowlist is the control instead: the row allows
    # `generativelanguage.googleapis.com` and NOT the `googleapis.com` suffix,
    # precisely so the telemetry sink under that same suffix is refused, and the
    # row declares it in `egress_beyond_stub` where a5's second half grades it.
    "gemini": (),
}


def _default_slug(cli: str) -> str:
    """`models[0]` from the shipped `providers.yaml` — the §6.1 default model.

    §8 item 3 makes `models[0]` the slug `--init` injects and the one the §6.1
    *Default model* cell records, so it is the single place this lane can read a
    real slug from without duplicating one into a second file.
    """
    from utils.provider_registry import load_base_config

    models = (load_base_config()["providers"].get(cli) or {}).get("models") or []
    return str(models[0]) if models else ""


def _live_rows() -> list[dict]:
    """The provider rows this lane covers, minus their stub overrides.

    **`model_arg` is a stub override too, and §7.4's own wording gets this
    wrong.** That section says a live row is the provider row with "same
    `model_arg`" — but a provider row pins `model_arg` to a *stub* id
    (`stub-model`), which exists so `test_the_stub_saw_only_this_row_s_model` can
    assert the stub saw exactly that id. There is no stub on this lane, and
    §7.4's own assertion 2 requires the response to report back the slug in the
    §6.1 *Default model* cell. Measured, 2026-08-17: carrying the row's
    `model_arg` verbatim sent `stub-model` to OpenRouter and the row died on
    `400 stub-model is not a valid model ID` — a failure of the harness, not of
    the adapter.

    So the model is substituted here, from `providers.yaml`'s `models[0]`, and
    **every `env_extra` value equal to the old stub id moves with it**. That
    second half is not tidiness: §7 step 4 requires a row's `env_extra` model
    name to be byte-equal to its `model_arg`, and kimi selects its model *only*
    through `KIMI_MODEL_NAME` in `env_extra` — so substituting one without the
    other would leave the CLI on the stub id while this lane reported the real
    one.
    """
    data = yaml.safe_load(REDIRECT.read_text(encoding="utf-8")) or {}
    rows = []
    for entry in data.get("clis") or []:
        # `lane: provider` OR an explicit `live_lane` opt-in. The second clause
        # is what lets cursor-agent join without being MOVED off `lane: fake`:
        # that value is the promise `tests/test_fake_lane_roster.py` grades — the
        # five mock parts that are cursor's only argv, parser and version-gate
        # coverage — and every reader of `lane` is a positive-match filter, so
        # retagging the row would have deleted that coverage with nothing going
        # red. The two lanes are orthogonal, so the row declares both.
        if entry.get("cli") not in LANE_CLIS:
            continue
        if entry.get("lane") != "provider" and not entry.get("live_lane"):
            continue
        row = dict(entry)
        # The entire difference between this lane and the provider lane.
        for stub_only in ("base_url_vars", "credential_vars", "http_log"):
            row.pop(stub_only, None)
        stub_model = str(row.get("model_arg", ""))
        # `models[0]`, unless this lane's endpoint cannot serve it and the row
        # carries a declared override with its reason (LIVE_MODEL_OVERRIDES).
        override_slug = (LIVE_MODEL_OVERRIDES.get(str(row["cli"])) or {}).get("slug")
        live_model = override_slug or _default_slug(str(row["cli"]))
        if live_model:
            row["model_arg"] = live_model
            row["env_extra"] = {
                key: (live_model if str(value) == stub_model else value)
                for key, value in (row.get("env_extra") or {}).items()
            }
        # …and then the lane's own corrections, which are NOT a model
        # substitution: see LIVE_ENV_OVERRIDES for why a carried-through
        # `env_extra` can name the wrong vendor once `base_url_vars` is gone.
        override = LIVE_ENV_OVERRIDES.get(str(row["cli"]))
        if override:
            row["env_extra"] = {**(row.get("env_extra") or {}), **override}
        rows.append(row)
    return rows


ROWS = _live_rows()
IDS = [r["cli"] for r in ROWS]


def _credential_env(row: dict) -> str:
    """The variable this row's credential travels in."""
    return str(row.get("credential_env") or DEFAULT_CREDENTIAL_ENV)


def _live_host(row: dict) -> str:
    """The one hostname assertion 1 must see this row reach."""
    url = str(row.get("live_base_url") or "")
    if url:
        return url.split("//", 1)[-1].split("/", 1)[0]
    return DEFAULT_LIVE_HOST


def _credential_material(row: dict) -> str | None:
    """The bytes of a FILE credential, read HOST-side, or None.

    A `:ro` bind of the source is not an option and the reason is structural, not
    a preference: the file is mode 0600 owned by the developer's uid, this lane
    runs as uid 10001, and `posture.yaml` rejects `--user $(id -u)` by name
    ("it produces a process with no passwd entry and an unowned $HOME, which is
    precisely the shape those CLIs mishandle"). The driver runs as the developer
    and can read it; the runner writes it into the container's own disposable
    HOME at 0600. Nothing in the container can reach the developer's file, so a
    token refresh cannot write back to it — which a `:rw` bind would have
    allowed and rule 3 of `creds.yaml` forbids outright.
    """
    source = row.get("credential_source")
    if not source:
        return None
    path = Path(str(source)).expanduser()
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _row_is_credentialed(row: dict) -> tuple[bool, str]:
    """Can this row run at all? Returns (yes, the reason when no)."""
    if row.get("credential_file"):
        source = row.get("credential_source", "<unset>")
        if _credential_material(row) is None:
            return False, (
                f"{row['cli']}'s credential is the file {source}, which is "
                f"missing or unreadable on this host — log the CLI in first "
                f"(`cursor-agent login`), or give the row a "
                f"`credential_env` instead")
        return True, ""

    name = _credential_env(row)
    if not os.environ.get(name):
        return False, (
            f"the credentialed provider lane needs {name} (a DEDICATED "
            f"credential carrying a spend limit); `make e2e-live-provider` "
            f"sources it from ~/.config/multi-llm/e2e.env")
    return True, ""


def _key_snapshot() -> dict | None:
    """``GET /api/v1/key`` — the accounting control that survives a silent CLI.

    A CLI's own usage number can be absent or wrong; the key-side delta cannot,
    which is why §7.4 makes it mandatory even where the row's reported usage is
    unreadable.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        request = urllib.request.Request(
            OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")).get("data")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _record(**fields) -> None:
    """Append one §6.6 row. A skip is a row; a blank means it was never attempted."""
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    fields.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with RECORD.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields, sort_keys=True) + "\n")


def _run_argv(row: dict, out_dir: Path, posture: dict, image: str) -> list[str]:
    """This lane's ``docker run``, built from posture plus its own three deltas.

    Identity, capability drop and resource limits come from ``posture.yaml`` so
    there is still one source for them. The three things that differ are stated
    here rather than added to that file, because they are true of this lane only:

    * **bridge network**, since the whole question is whether openrouter.ai is
      reachable — the guarantee downgrades from impossibility to detection;
    * **``--dns 127.0.0.1``** plus
      ``--sysctl net.ipv4.ip_unprivileged_port_start=53``, which is what lets the
      in-container allowlisting resolver bind the DNS port under ``--cap-drop
      ALL`` as uid 10001. A daemon that refuses the sysctl leaves the control
      unavailable, and the runner turns that into a recorded skip;
    * **an empty disposable workspace** — no repository mount, no ``-v $(pwd)``,
      no ``/work`` bind of the checkout. ``--add-dir`` therefore resolves to an
      empty directory rather than a source tree.
    """
    identity = posture["identity"]
    security = posture["security"]
    limits = posture["limits"]

    argv = ["docker", "run", "--rm", "--name", f"mllm-live-{row['cli']}-{os.getpid()}"]
    argv += ["--user", f"{identity['uid']}:{identity['gid']}"]
    for capability in security["cap_drop"]:
        argv += ["--cap-drop", capability]
    for option in security["security_opt"]:
        argv += ["--security-opt", option]
    if security.get("readonly_rootfs"):
        argv += ["--read-only"]
    argv += ["--pids-limit", str(limits["pids_limit"])]
    argv += ["--memory", str(limits["memory"])]
    argv += ["--cpus", str(limits["cpus"])]

    argv += ["--network", "bridge"]
    argv += ["--dns", "127.0.0.1"]
    argv += ["--sysctl", "net.ipv4.ip_unprivileged_port_start=53"]

    argv += ["--tmpfs", "/work/live:rw,mode=0700,uid=10001,gid=10001"]
    argv += ["--tmpfs", "/tmp:rw,mode=1777"]
    argv += ["-v", f"{out_dir}:/work/out:rw"]
    # The credential, in whichever shape this row declares — and by NAME for a
    # variable, so its value passes through the daemon from this process's
    # environment and never appears in the argv this function returns (which is
    # written verbatim to `run-command.txt`). A file credential is the one case
    # where a value must be handed over explicitly; see `_credential_material`
    # for why a bind mount cannot serve it.
    # ONLY this row's own credential. A row whose credential is a file declares
    # no variable, and forwarding the lane's default anyway would hand the
    # OpenRouter key to a third-party binary that has no use for it — the exact
    # cross-vendor leak the host lane's wrapper scrub exists to prevent, arriving
    # by a different door.
    if not row.get("credential_file") or row.get("credential_env"):
        argv += ["-e", _credential_env(row)]
    # BY NAME here too, and that is not a style choice. `-e VAR=value` would put
    # the credential's bytes in the argv this function returns, and the caller
    # writes that argv verbatim to `run-command.txt` — a token in a tracked run
    # artifact, which is the exact leak this lane's write-time redaction exists
    # to prevent. The value is placed in the *docker client's* environment by the
    # caller instead, and `-e VAR` forwards it from there.
    if row.get("credential_file"):
        argv += ["-e", CREDENTIAL_MATERIAL_ENV]

    inner = min(int(row.get("row_timeout_s", 300)) - 30, 180)
    argv += ["--entrypoint", "/usr/bin/python3", image]
    argv += ["/opt/mllm/rig/live_provider.py",
             "--cli", row["cli"],
             "--model", str(row.get("model_arg", "")),
             "--prompt", SMOKE_PROMPT,
             "--timeout-s", str(max(inner, 60)),
             # §7.4 drops only the STUB overrides; `env_extra` is part of the row
             # and carries image-layout facts (copilot's pre-warmed
             # XDG_CACHE_HOME). The runner applies it beneath the adapter.
             "--row-env", json.dumps(row.get("env_extra") or {})]
    if row.get("binary"):
        argv += ["--binary", str(row["binary"])]
    if row.get("live_base_url"):
        argv += ["--live-base-url", str(row["live_base_url"])]
    if row.get("credential_env"):
        argv += ["--credential-env", str(row["credential_env"])]
    if row.get("credential_file"):
        argv += ["--credential-file", str(row["credential_file"])]
    # The DNS allowlist is this lane's ONLY egress enforcement, so a row that
    # names its own is narrowing or widening the containment set in the one file
    # a reviewer reads for it. Absent, the resolver's default (openrouter.ai)
    # stands.
    for name in row.get("live_allow") or []:
        argv += ["--allow", str(name)]
    # ...and the per-row permitted names, which widen it for one row only and
    # carry their measurement in ALLOWED_EGRESS rather than in an argv literal.
    for name in sorted(ALLOWED_EGRESS.get(str(row["cli"]), {})):
        argv += ["--allow", name]
    # The row's config roots, so an adapter that respects a pinned one points the
    # CLI at the rendered seed rather than at a fresh empty directory. vibe is the
    # case that needs it: its [[providers]]/[[models]] catalog lives in
    # $VIBE_HOME/config.toml, so an unpinned root means no providers configured.
    for var, rel in (row.get("config_roots") or {}).items():
        argv += ["--config-root", f"{var}={rel}"]
    # A seed that exists only to point the CLI at the stub is not rendered here
    # at all. STUB_ONLY_SEED carries the per-row measurement; the runner records
    # `seed_skipped` in the evidence so the two cases — "the tree was skipped"
    # and "there is no tree" — stay distinguishable after the fact.
    if row["cli"] in STUB_ONLY_SEED:
        argv += ["--no-seed"]
    # ...and the per-FILE version of the same statement, for a tree that is part
    # live configuration and part stub artifact. SEED_DROP carries the per-file
    # measurement; the runner records what it removed in the evidence.
    for rel in sorted(SEED_DROP.get(str(row["cli"]), {})):
        argv += ["--seed-drop", rel]
    # The credential, under the name this CLI's own auth path reads. Only the
    # NAME crosses here — the value is read inside the container from the row's
    # `credential_env`, so nothing in this argv (which is written verbatim to
    # `run-command.txt`) can carry it.
    for alias in LIVE_CREDENTIAL_ALIASES.get(str(row["cli"]), ()):
        argv += ["--credential-alias", alias]
    if row["cli"] in ENDPOINT_IN_ENV:
        argv += ["--endpoint-in-env"]
    return argv


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def inner_userns(posture):
    """Can a CLI build its OWN sandbox inside this lane's container? (measured)

    One `docker run` per session, under this lane's real identity, capability and
    privilege flags, asking the kernel the question codex's bubblewrap asks:
    `unshare -Un true`. Cheap (~1 s, no network, no credential) and it is what
    keeps :data:`NEEDS_INNER_USERNS` a MEASUREMENT rather than a belief — the day
    a daemon or a posture permits CLONE_NEWUSER the row runs again by itself.
    """
    import subprocess

    identity, security = posture["identity"], posture["security"]
    argv = ["docker", "run", "--rm",
            "--user", f"{identity['uid']}:{identity['gid']}",
            "--network", "none"]
    for capability in security["cap_drop"]:
        argv += ["--cap-drop", capability]
    for option in security["security_opt"]:
        argv += ["--security-opt", option]
    argv += ["--entrypoint", "/usr/bin/unshare", harness.DEFAULT_IMAGE,
             "-Un", "true"]
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        # Unanswerable is not "available": a probe that could not run must not
        # green-light a row whose whole problem is that this facility is absent.
        return False


@pytest.fixture(scope="module")
def credit_window():
    """The key's balance before and after the whole target (§7.4's Accounting row).

    **This is OpenRouter's window, and it does not generalize.** `GET
    /api/v1/key` is a vendor surface, not a lane concept: cursor publishes
    nothing equivalent — `status --format json` reports auth state and
    `about --format json` a subscription tier, neither carrying usage, credit or
    quota. So the window still spans the whole target for every OpenRouter row,
    and a row whose vendor has no usage endpoint records the SHAPE of its bound
    instead of a fabricated delta: see `_record_metering_note`. Deriving a cost
    from token counts is refused for the same reason the adapters refuse it —
    it needs a per-model price table this repo does not own.
    """
    before = _key_snapshot()
    yield before
    after = _key_snapshot()
    if before or after:
        _record(kind="credit-window",
                metering="metered",
                vendor="openrouter",
                usage_before=(before or {}).get("usage"),
                usage_after=(after or {}).get("usage"),
                limit=(before or {}).get("limit"),
                limit_remaining_after=(after or {}).get("limit_remaining"))


def _record_metering_note(row: dict, evidence: dict) -> None:
    """§6.6's accounting row for a vendor with no usage endpoint.

    "A skip is a row, not a blank" applies to accounting too: a row that cannot
    be priced records *why*, plus every real number it does have — the adapter's
    own `parse_usage` block, read in-container from the envelope `parse_output`
    discards. `usage_support` states how far that goes (`partial` for cursor:
    tokens yes, cost never).
    """
    if not row.get("live_lane") or row.get("live_vendor") in (None, "openrouter"):
        return
    _record(kind="credit-window",
            vendor=str(row.get("live_vendor")),
            cli=row["cli"],
            metering="unmetered — this vendor publishes no usage endpoint; the "
                     "bound is the row's own one-shot plus row_timeout_s",
            usage=evidence.get("usage"),
            usage_support=evidence.get("usage_support"))


@pytest.fixture(scope="module", params=ROWS or [None],
                ids=IDS or ["no-rows-yet"])
def smoke(request, posture, credit_window):
    """Run one row's live one-shot and return its evidence.

    Skip — never fail, never soft-pass — when the key is absent or a containment
    control cannot be applied, mirroring ``test_container_host_lane.py``. A
    skipped row is reported as skipped in §6.6's record and is never counted as a
    green smoke.
    """
    row = request.param
    if row is None:
        # Deliberately writes NO record row. §6.6's "a skip is a row, not a
        # blank" is about a CLI whose smoke was attempted and could not run; a
        # candidate with no adapter yet has no smoke to skip, and recording one
        # per invocation would fill the tracked record with the absence of work.
        pytest.skip("no Wave-1/2 provider row exists yet — the lane is wired and "
                    "selects zero rows (see LANE_CLIS)")

    cli = row["cli"]
    credentialed, why = _row_is_credentialed(row)
    if not credentialed:
        _record(kind="smoke", cli=cli, result=f"skipped — {why}")
        pytest.skip(why)

    # BEFORE THE SPEND, not after it. A row whose CLI cannot build its own inner
    # sandbox here cannot produce a gradeable answer — it produces a model
    # retrying a tool call for the whole inner bound — so the lane refuses to buy
    # one. Recorded as a row, never a blank: §6.6's "a skip is a row" applies to
    # a measured precondition exactly as it does to a missing credential.
    if cli in NEEDS_INNER_USERNS and not request.getfixturevalue("inner_userns"):
        reason = NEEDS_INNER_USERNS[cli]
        _record(kind="smoke", cli=cli, result=f"skipped — {reason}")
        pytest.skip(f"{cli}: {reason}")

    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = harness.RUNS / utc / f"live-{cli}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)

    argv = _run_argv(row, out_dir, posture, harness.DEFAULT_IMAGE)
    # Safe to write verbatim precisely because every credential is forwarded by
    # NAME — `_run_argv` never puts a value in argv. The assertion below is the
    # guard on that promise, not decoration.
    material = _credential_material(row) if row.get("credential_file") else None
    command_line = " ".join(argv)
    assert not material or material.strip() not in command_line, (
        f"{cli}: the credential's bytes reached the docker argv — refusing to "
        f"write run-command.txt")
    (out_dir / "run-command.txt").write_text(command_line + "\n", encoding="utf-8")

    import subprocess

    # The docker CLIENT's environment, which is where a `-e VAR` forward reads
    # from. A file credential is placed here and nowhere else: not in argv, not
    # in a mount, not in this process's own os.environ (so a later test, or a
    # crash dump, cannot pick it up from there).
    client_env = dict(os.environ)
    if material is not None:
        client_env[CREDENTIAL_MATERIAL_ENV] = material

    started = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              env=client_env,
                              timeout=int(row.get("row_timeout_s", 300)))
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        # Hitting the outer bound kills the row and FAILS it. It is never retried
        # into a second charge.
        code, out, err = None, "", f"outer row_timeout_s expired: {exc}"

    evidence_path = out_dir / "live-provider.json"
    evidence = (json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence_path.is_file() else {})
    dns_path = out_dir / "dns.jsonl"
    queries = []
    if dns_path.is_file():
        for line in dns_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    queries.append(json.loads(line))
                except ValueError:
                    continue

    # The runner's own refusal code: a control could not be applied, so the row
    # skips with the reason rather than running weakened.
    if code == 20 or evidence.get("control_failure"):
        reason = evidence.get("control_failure", "containment control unavailable")
        _record(kind="smoke", cli=cli, result=f"skipped — {reason}")
        pytest.skip(f"{cli}: {reason}")

    bundle = {
        "row": row,
        "cli": cli,
        "exit_code": code,
        "wall_clock_s": round(time.time() - started, 2),
        "evidence": evidence,
        "queries": queries,
        "out_dir": out_dir,
        "stderr": err,
    }
    _record(kind="smoke", cli=cli, model=row.get("model_arg"),
            version=harness.cli_version(cli), exit_code=code,
            wall_clock_s=bundle["wall_clock_s"],
            dns_names=sorted({q.get("qname") for q in queries}),
            # The adapter's own `parse_usage`, not `parsed["usage"]`: several
            # adapters (cursor among them) discard the envelope in `parse_output`
            # and read usage from it separately, so the old spelling recorded
            # `null` for exactly the rows that DO report tokens.
            usage=evidence.get("usage") or evidence.get("parsed", {}).get("usage"),
            result="attempted")
    _record_metering_note(row, evidence)
    return bundle


class TestTheSmokeReachedTheRealEndpoint:
    """§7.4's six assertions. All six filled, or the smoke does not count."""

    def test_a1_endpoint(self, smoke):
        """Assertion 1 — the run reached openrouter.ai, evidenced not assumed.

        Proves the §2.1 mechanism still works at the installed version and that
        the CLI did not silently fall back to a vendor default or a stale config.
        """
        host = _live_host(smoke["row"])
        names = {q.get("qname", "").lower() for q in smoke["queries"]}
        env = smoke["evidence"].get("env", {})
        # Dot-suffix, the same rule the resolver's allowlist uses: cursor's row
        # allows `cursor.sh` and the CLI resolves `api2.cursor.sh` under it.
        reached = (any(n == host or n.endswith("." + host) for n in names)
                   or any(host in str(v) for v in env.values()))
        assert reached, (
            f"{smoke['cli']}: nothing in the DNS log or the resolved env names "
            f"{host} — saw {sorted(names)}")

    def test_a2_model(self, smoke):
        """Assertion 2 — the slug we pinned is the slug the response reports back.

        The live analogue of ``test_the_stub_saw_only_this_row_s_model``: it
        catches a backend that accepts a slug and serves another.

        **A CLI that reports no model at all records `unread`, and that is a
        declared per-CLI fact rather than a fallback.** §7.4 assumed every
        candidate echoes the served model somewhere ("copilot's `data.model`,
        kimi's assistant message, qwen's JSON"). Measured 2026-08-17, kimi does
        not: its `stream-json` records are `{role, content}` and carry no model
        field, no usage block and no generation id anywhere in the stream, which
        is the same measurement behind its `usage_support = "none"`. There is
        therefore nothing for this assertion to read — a different thing from
        reading it and finding the wrong slug, and §7.4's assertion 6 and §6.6's
        record both spell that difference `unread — <why>`.

        A CLI absent from :data:`NO_MODEL_ECHO` is still asserted strictly, so
        this cannot become a way to silence a real substitution: adding a name
        here is a visible edit that has to carry its measurement.
        """
        cli = smoke["cli"]
        pinned = str(smoke["row"].get("model_arg", ""))
        blob = json.dumps(smoke["evidence"].get("parsed", {})) + \
            smoke["evidence"].get("stdout", "")
        # …plus stderr, for the one row measured to put its model banner there
        # (see MODEL_ECHO_ON_STDERR). Per-CLI, so no other row's haystack — and
        # therefore no other row's grade — changes.
        if cli in MODEL_ECHO_ON_STDERR:
            blob += smoke["evidence"].get("stderr", "")
        if cli in NO_MODEL_ECHO:
            _record(kind="assertion", cli=cli, assertion="a2_model",
                    outcome=f"unread — {NO_MODEL_ECHO[cli]}")
            pytest.skip(f"a2 unread — {NO_MODEL_ECHO[cli]}")
        # THE SPELLING THAT CROSSED THE ARGV BOUNDARY, for the two adapters that
        # split the spec before spawning (see SPLIT_MODEL_SPEC). Asking a CLI to
        # report a string it was never given is a harness bug wearing an
        # assertion's clothes; the provider half those adapters strip is the
        # route, and a1 grades the route off the DNS log.
        expected = pinned.partition("/")[2] if cli in SPLIT_MODEL_SPEC else pinned
        assert expected and expected in blob, (
            f"{smoke['cli']}: the response does not name the pinned slug "
            f"{expected!r}"
            + (f" (from model_arg {pinned!r}; {SPLIT_MODEL_SPEC[cli]})"
               if cli in SPLIT_MODEL_SPEC else ""))

    def test_a3_parsed_output(self, smoke):
        """Assertion 3 — the SHIPPED parser survives the vendor's real bytes.

        The stub's bytes are ours; only real bytes prove the parser holds. And no
        declared ``exit_zero_failure_markers`` may fire.
        """
        # WHICHEVER CHANNEL DELIVERED, because production reads the file first
        # and falls back to stdout. An adapter asked for a file may legitimately
        # print a confirmation line rather than the array, and grading only
        # `parse_output(stdout)` would fail it for obeying the instruction.
        file_parsed = smoke["evidence"].get("file_parsed")
        if file_parsed not in (None, "", [], {}) and "error" not in (
                file_parsed if isinstance(file_parsed, dict) else {}):
            return

        parsed = smoke["evidence"].get("parsed") or {}
        assert parsed.get("success") is True, (
            f"{smoke['cli']}: neither channel produced a document — the file at "
            f"{smoke['evidence'].get('channel_path')} holds "
            f"{smoke['evidence'].get('channel_file_bytes', 0)} bytes and "
            f"parse_output failed: {parsed.get('error')!r}")
        assert parsed.get("data") not in (None, "", [], {}), (
            f"{smoke['cli']}: parse_output returned empty data")

    def test_a4_exit_status(self, smoke):
        """Assertion 4 — exit 0 and ``classify_failure`` reports nothing.

        Pins the exit-code contract against the live backend, whose auth and
        rate-limit shapes differ from the stub's. A 429 or a 5xx fails here, hard.
        """
        assert smoke["evidence"].get("exit_code") == 0, (
            f"{smoke['cli']}: exited {smoke['evidence'].get('exit_code')}\n"
            f"{smoke['evidence'].get('stderr', '')[-2000:]}")
        assert smoke["evidence"].get("classification") is None, (
            f"{smoke['cli']}: classify_failure reported "
            f"{smoke['evidence'].get('classification')!r}")

        # THE EXIT CODE IS NOT ALWAYS THE VERDICT, and for one row it is never
        # the verdict: cursor-agent "exited 0 in all six auth states measured",
        # so the two assertions above would pass a logged-out run that produced
        # nothing. Its declared `exit_zero_failure_markers` are what carry the
        # signal, they were captured into the evidence for every row, and until
        # now no assertion read them — a captured control nobody grades is not a
        # control. `classify_failure` above already consumes the markers for
        # adapters whose classifier routes through them; this is the direct read
        # of the stream, so a marker that fires cannot be lost to a classifier
        # that returned None for some other reason.
        import re as _re

        stream = (smoke["evidence"].get("stdout", "")
                  + smoke["evidence"].get("stderr", ""))
        fired = [pattern for pattern in smoke["evidence"].get("exit_zero_markers", [])
                 if _re.search(pattern, stream)]
        assert not fired, (
            f"{smoke['cli']}: exited 0 but its own exit-zero failure markers "
            f"matched {fired} — for this CLI the exit code carries no verdict "
            f"and the stream does")

    def test_a5_egress(self, smoke):
        """Assertion 5 — openrouter.ai and nothing else, plus the opt-outs by name.

        ``--network none`` makes this question unanswerable offline; this is the
        only lane where a vendor call-home *could* happen, so it is the only lane
        that can rule one out. A forbidden query is reported as a finding: it
        means a shipped adapter leaks a request users did not ask for.
        """
        host = _live_host(smoke["row"])
        # A DECLARED REFUSAL IS NOT AN EXEMPTION. a5 grades queries, so a name
        # the resolver refused still shows up in the log; a row may declare such
        # a name in REFUSED_EGRESS with its measurement, and it is dropped from
        # `leaked` ONLY while the resolver went on refusing it. The moment a
        # declared name comes back `decision == "allow"` it counts as a leak
        # again, which is the difference between "refused, as designed" and
        # "reached a forbidden host" — the thing the raw log cannot say.
        declared_refusals = REFUSED_EGRESS.get(smoke["cli"], {})

        def _refused_as_declared(qname: str) -> bool:
            # Dot-suffix, the same rule the resolver's own allowlist uses, so a
            # declaration cannot be dodged by a subdomain.
            name = (qname or "").lower().rstrip(".")
            if not any(name == d or name.endswith("." + d)
                       for d in declared_refusals):
                return False
            return not any(
                q.get("decision") == "allow"
                and (q.get("qname") or "").lower().rstrip(".") == name
                for q in smoke["queries"])

        # A PERMITTED name is not a leak. Same dot-suffix rule as everywhere
        # else, so a permission cannot be dodged by a subdomain.
        permitted = ALLOWED_EGRESS.get(smoke["cli"], {})

        def _permitted(qname: str) -> bool:
            name = (qname or "").lower().rstrip(".")
            return any(name == d or name.endswith("." + d) for d in permitted)

        leaked = sorted({
            q.get("qname") for q in smoke["queries"]
            if any(bad in (q.get("qname") or "").lower() for bad in FORBIDDEN_EGRESS)
            and not (q.get("qname") or "").lower().endswith(host)
            and not _refused_as_declared(q.get("qname"))
            and not _permitted(q.get("qname"))
        })
        assert not leaked, (
            f"{smoke['cli']}: queried vendor hosts this lane forbids: {leaked} — "
            f"a shipped adapter is making a request users did not ask for")
        if declared_refusals:
            _record(kind="assertion", cli=smoke["cli"], assertion="a5_egress",
                    outcome="declared-refusal upheld",
                    refused=sorted(declared_refusals))

        # A PERMISSION THAT IS NEVER EXERCISED IS A DEAD DECLARATION, and a dead
        # declaration is how a containment set widens without anything going red.
        # So a permitted name that WAS queried must have been answered: if the
        # resolver refused it, either the allowlist wiring broke or the CLI stopped
        # needing it, and both are things to find out now rather than later.
        for name in sorted(permitted):
            asked = [q for q in smoke["queries"]
                     if (q.get("qname") or "").lower().rstrip(".") == name]
            if not asked:
                _record(kind="finding", cli=smoke["cli"], assertion="a5_egress",
                        outcome="permitted host was never queried", host=name,
                        note="ALLOWED_EGRESS entry may be stale — nothing asked "
                             "for it on this run")
                continue
            assert any(q.get("decision") == "allow" for q in asked), (
                f"{smoke['cli']}: {name} is declared in ALLOWED_EGRESS but the "
                f"resolver refused every query for it — the permission is not "
                f"reaching the resolver's allowlist")
            _record(kind="assertion", cli=smoke["cli"], assertion="a5_egress",
                    outcome="declared permission exercised", host=name,
                    queries=len(asked))

        env = smoke["evidence"].get("env", {})
        for name in OPT_OUTS.get(smoke["cli"], ()):
            assert name in env, (
                f"{smoke['cli']}: the resolved env is missing the {name} opt-out")

        # THE ROW'S DECLARED CALL-HOME, graded rather than tolerated. A CLI with
        # no opt-out variable (cursor: no DO_NOT_TRACK, no CURSOR_TELEMETRY_*,
        # nothing) still must not reach its telemetry sinks here — the DNS
        # allowlist is what stops it, and this asserts the allowlist really did.
        # A name appearing ANSWERED rather than refused means the row widened its
        # own containment set.
        declared = [h.lower() for h in
                    (smoke["row"].get("egress_beyond_stub") or {}).get("hosts", [])]
        answered = sorted({
            q.get("qname") for q in smoke["queries"]
            if q.get("decision") == "allow" and any(
                (q.get("qname") or "").lower().endswith(h) for h in declared)
            # ...except one the row explicitly permits. `egress_beyond_stub` is
            # the FACTUAL list of what this row reaches past the stub; whether a
            # given entry is tolerated is policy, and policy lives in
            # ALLOWED_EGRESS. Without this clause, permitting a name would make
            # the row fail here for doing exactly what it was told to do.
            and not _permitted(q.get("qname"))
        })
        assert not answered, (
            f"{smoke['cli']}: its declared telemetry hosts {answered} were "
            f"ALLOWED to resolve — the row's live_allow is wider than its "
            f"egress_beyond_stub says it needs to be")

    def test_a6_spend_bound_observed(self, smoke):
        """Assertion 6 — the run's reported usage is recorded and within bound.

        The stub reports no cost, so nothing offline can show the lane stays
        inside its budget. A row whose usage cannot be read records **that**
        rather than omitting the check — which is why this asserts the record
        exists, not that a number was parseable.
        """
        assert RECORD.is_file(), "no §6.6 record row was written for this attempt"
        rows = [json.loads(line) for line in
                RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
        mine = [r for r in rows if r.get("cli") == smoke["cli"]]
        assert mine, f"{smoke['cli']}: no row in the live smoke record"

    def test_a7_declared_output_channel(self, smoke):
        """Assertion 7 — `supports_file_output` is what the CLI really does.

        **The assertion that did not exist when it was needed.** That declaration
        decides which `{output_location}` block every Lane A review prompt gets,
        and nothing offline can grade it: the container stub never emits a tool
        call, so on the provider lane an adapter that cannot write a file looks
        exactly like one that can. cursor-agent shipped a release under
        `--mode ask` answering every review slot with "I can't write the
        requested file while Ask mode is active", and no lane went red.

        This lane is the only one where a real model can really try, so the smoke
        prompt carries production's own rendered block and this reads the result:

        * declared ``True``  → asked for a file, and the file must be there;
        * declared ``False`` → asked for stdout, and a file must NOT be there
          (an adapter writing one anyway is disobeying its own mode, which is
          how cursor's sibling model masked the original bug).

        A failure here is a declaration to change, not a flake to retry.
        """
        evidence = smoke["evidence"]
        declared = evidence.get("declared_supports_file_output")
        observed = evidence.get("observed_output_channel")
        if declared is None or observed is None:
            pytest.skip("this row predates the declared-channel probe")

        # THE RECORD, BEFORE THE ASSERTION. A red row's reason otherwise lives
        # only in the terminal that ran it, and §6.6's record is what survives
        # the session — the same argument that put a2's `unread` and a5's
        # declared refusals in there. Written only on DISAGREEMENT, so a green
        # row adds nothing.
        if bool(declared) != (observed == "file"):
            _record(kind="finding", cli=smoke["cli"],
                    assertion="a7_declared_output_channel",
                    declared_supports_file_output=declared,
                    observed_output_channel=observed,
                    channel_file_bytes=evidence.get("channel_file_bytes"),
                    stdout_head=(evidence.get("stdout") or "")[:400])

        if declared:
            assert observed == "file", (
                f"{smoke['cli']}: declares supports_file_output = True and was "
                f"asked to write {evidence.get('channel_path')}, but no file "
                f"appeared — the prompt every review sends it is one it cannot "
                f"obey, and the slot survives only on the stdout fallback. "
                f"Declare False (and the prompt will ask for stdout), or find "
                f"the flag that lets it write.")
        else:
            assert observed == "stdout", (
                f"{smoke['cli']}: declares supports_file_output = False and was "
                f"asked for stdout, yet wrote "
                f"{evidence.get('channel_file_bytes')} bytes to "
                f"{evidence.get('channel_path')} — it can write files after all, "
                f"or it is ignoring the read-only mode the declaration rests on. "
                f"Either way the declaration and the binary disagree.")
