"""Does the string the documents tell a reader to type actually dispatch?

Follow-up 7 of `plans/readme-trim-and-multi-host-install-docs.md`, and the
acceptance row *"Documented invocation syntax dispatches on each host"* that had
never been answered because **nothing had ever typed it**.

## The gap this module closes

`README.md` and `docs/hosts.md` both said, until this row ran, that the shim is
invoked as `/multi-llm <the same arguments the Claude Code skill takes>` on *both*
installable hosts. The two hosts do not install the same kind of object, though:

* `codex` gets a **skill** — `$HOME/.codex/skills/multi-llm/SKILL.md`
* `opencode` gets a **command** — `$XDG_CONFIG_HOME/opencode/commands/multi-llm.md`

so each host does its own resolution, namespacing and word-splitting, on its own
format. `/multi-llm …` is therefore a claim about a **host's command parser**, and
that claim had no provenance at all: it was a constant inside the renderer, beside
a `hosts/capabilities.yaml` comment calling the shared-skills-format claim "a
hypothesis, not a premise".

The credentialed host lane does not settle it. Its rows prompt each host with
`Use the multi-llm skill with exactly these arguments: …` (`prompt_template` in
`tests/container/host_matrix.yaml`), which proves the **writer** — the shim landed
somewhere the host reads, and an argument string reaches our dispatcher — and
proves nothing at all about the published spelling.

## What this row does instead

Per installable host, at `--creds none` (so: `--network none`, every credential
variable asserted absent, zero spend):

1. install the shim (`install-host <host> --global`) into the row's tmpfs `$HOME`;
2. point the **host's own model** at the container stub on loopback — enough,
   because what is under test is *argument delivery*, not model output;
3. start a **fresh session** in a pty. Both hosts are `reload: new-session`, and
   one host process per row satisfies that boundary by construction: there is
   nothing to reload into, because the process *is* the new session;
4. type the exact published string, with a per-run marker (`plans/mllm-invocation-
   <hex>.md`) so no assertion can be satisfied by stale text, and press Enter;
5. read back **what the host sent to the model**, through the stub's opt-in
   prompt-fact probe — a list of the rig's own fact names per request, never the
   prompt body (`stub_server.install_facts`, `entrypoint.invocation_facts`).

A host that resolved the command sends its own shim's text: the shim title, the
dispatcher block with `{SKILL_DIR}` already substituted, the baked
`--host <host> --yes --exec-subagents`, and `--review-plan plans/<marker>.md`
still adjacent. A host that did not sends the characters the user typed.

## The measured answer, 2026-08-24

| Host | `/multi-llm …` in a fresh session |
| --- | --- |
| `opencode` 1.18.18 | **resolves.** The TUI expands the installed command; every fact above was present in the turn the host sent. |
| `codex` 0.146.0 | **refused.** `• Unrecognized command '/multi-llm'. Type "/" for a list of supported commands.` — and nothing is sent at all. |

Codex's shim is discovered — the skill is advertised to the model by name inside
`<skills_instructions>`, with its `SKILL.md` locator — but a Codex *skill* is a
model-invoked instruction source, not a composer command. Its slash surface lists
`/model`, `/ide`, `/permissions`, `/keymap`, `/vim`, `/experimental`, `/approve`,
`/memories` and no skill; `/multi` offers no completion; and the deprecated
`install_paths.legacy` prompts directory does not rescue it either (a file at
`$CODEX_HOME/prompts/<name>.md` gets the same "Unrecognized command" on this
version). Headless, `codex exec "/multi-llm …"` sends the line verbatim.

**That makes this a documentation defect, per D5 of the plan**, and
`test_the_documents_state_the_syntax_each_host_actually_takes` is where it lands.
`README.md`, `docs/hosts.md` and `hosts/render_capabilities.py` are owned
elsewhere, so this module measured and named the correction rather than making it.

**The correction landed on 2026-08-24** (follow-up 7): the renderer's hard-coded
"Invoke it as /multi-llm …" sentence is gone, replaced by a required per-host
`invocation:` field in `hosts/capabilities.yaml` — `verified` on both installable
hosts, on the strength of this row — which the per-host quickstart renders in its
place, and README's `### Codex and OpenCode` block now splits the two hosts by
hand. The assertion below stays exactly as it was: it is what keeps that split
present, and it is what fails the day a host's verdict flips.

## Why the assertions are shaped like this

Every test grades `invocation.json` against the `documented_invocation` cell in
`host_matrix.yaml`, which carries the measurement and its date. So the module
fails in **both** directions, which is the whole point of committing a negative:
a host that stops taking the spelling fails, and a host that *starts* taking it
fails too — and the second failure is the one that says "go update the document,
the correction below is now stale".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from .container.docker_probe import requires_docker
from .container import harness

pytestmark = [pytest.mark.container, requires_docker()]

#: The repo root, from this file rather than from any generated artifact's depth
#: — the same anchoring defect D2 of the plan calls out in `test_install_host.py`.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Where the published spelling has to appear for the coupling below to hold. Both
#: are read, never written: they are owned by other work.
PUBLISHED_IN = ("README.md", "docs/hosts.md")


def _installable_hosts() -> list[str]:
    """`installable: true` in `hosts/capabilities.yaml` — the source of truth.

    Read from the capability data rather than from `host_matrix.yaml` on purpose:
    a fourth installable host must arrive here *unmeasured*, so the schedule guard
    below fails and somebody has to type its documented string once before the
    documents can claim anything about it.
    """
    return sorted(host_id for host_id, host in harness.load_capabilities().items()
                  if host.get("installable"))


INSTALLABLE = _installable_hosts()


def _cell(host: str) -> dict:
    entry = harness.load_host_matrix().get(host) or {}
    return dict(entry.get("documented_invocation") or {})


class TestEveryInstallableHostIsMeasured:
    """The schedule guard: no installable host may go unmeasured and undocumented.

    Without it, adding a fourth `installable: true` host extends `docs/hosts.md`'s
    `/multi-llm` claim to a host nobody ever typed it into — which is exactly the
    state this module was written to end, one host later.
    """

    @pytest.mark.parametrize("host", INSTALLABLE)
    def test_it_carries_a_documented_invocation_cell(self, host):
        cell = _cell(host)
        assert cell, (
            f"{host}: `installable: true` in hosts/capabilities.yaml and no "
            f"`documented_invocation` cell in tests/container/host_matrix.yaml. "
            f"docs/hosts.md will render the `/multi-llm …` sentence for this "
            f"host regardless, so the claim would ship unmeasured. Add the cell "
            f"(typed / launch_argv / ready / refusal / resolves) and run this "
            f"module.")

    @pytest.mark.parametrize("host", INSTALLABLE)
    def test_its_verdict_is_measured_rather_than_assumed(self, host):
        cell = _cell(host)
        assert cell.get("provenance") == "verified", (
            f"{host}: `documented_invocation.provenance` is "
            f"{cell.get('provenance')!r}. This cell exists precisely because the "
            f"capability data's own `documented` provenance was not evidence; a "
            f"second `documented` cell would repeat the problem it fixes.")
        assert isinstance(cell.get("resolves"), bool), (
            f"{host}: `documented_invocation.resolves` must be a bool — it is "
            f"the row's expected verdict, and an absent one makes the grader "
            f"assert nothing.")

    def test_the_published_spelling_is_the_one_being_typed(self):
        """The matrix's `typed` really is the string the documents publish.

        A cell that drifted from the documents would test a spelling nobody is
        told to use, which is the same failure as not testing at all.
        """
        published = {name: (REPO_ROOT / name) for name in PUBLISHED_IN}
        missing = sorted(name for name, path in published.items()
                         if not path.is_file())
        assert not missing, (
            f"the documents that publish the invocation are missing: {missing}. "
            f"This guard fails on ABSENCE deliberately — a tree with no "
            f"docs/hosts.md has no install page for two of three hosts, and that "
            f"must never render as 'up to date'.")
        text = "\n".join(path.read_text(encoding="utf-8")
                         for path in published.values())
        for host in INSTALLABLE:
            typed = str(_cell(host).get("typed") or "")
            head = typed.split()[0] if typed else ""
            assert head and head in text, (
                f"{host}: host_matrix.yaml types {typed!r}, whose command word "
                f"{head!r} appears in neither {' nor '.join(PUBLISHED_IN)}. The "
                f"row would then be measuring a spelling this repository does "
                f"not publish.")


@pytest.fixture(scope="module", params=INSTALLABLE, ids=INSTALLABLE)
def probe(request, posture, utc):
    """Run one host's documented-invocation row and return its evidence."""
    host = request.param
    cell = _cell(host)
    row = harness.Row(
        row_id=f"documented-invocation-{host}",
        lane="host",
        host=host,
        creds="none",
        documented_invocation=True,
        # The container's own bound is the outer one: the entrypoint stops the
        # host at `documented_invocation.wall_clock_s` and still writes its
        # record, so a row that hits THIS cap has failed the rig rather than the
        # host, and says so.
        row_timeout_s=int(cell.get("wall_clock_s") or 300) + 300,
    )
    result = harness.run_row(row, posture=posture, utc=utc)
    return result, result.invocation, cell


@pytest.fixture(scope="module")
def posture():
    return harness.load_posture()


@pytest.fixture(scope="module")
def utc():
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class TestTheRowIsFreeAndReallyRan:
    """The tier's own claims, asserted rather than assumed — as the free
    discovery tier next door asserts its own."""

    def test_the_row_completed(self, probe):
        result, _record, _cell = probe
        assert not result.timed_out, (
            f"{result.row.host}: {result.cap_reason} after "
            f"{result.wall_clock_s}s — the container hit the DRIVER's cap, which "
            f"means the entrypoint never reached its own bound and wrote no "
            f"verdict. That is a rig failure, not a host finding.\n"
            f"--- container stderr (tail) ---\n{result.stderr[-2000:]}")
        assert result.exit_code == 0, (
            f"{result.row.host}: exited {result.exit_code}\n"
            f"--- container stderr (tail) ---\n{result.stderr[-2000:]}")

    def test_it_had_no_route_out(self, probe):
        """`--network none`, from both sides of the container wall.

        The row drives a third-party binary with a model endpoint configured. An
        empty external-call log would prove only that nothing reached a host we
        watched; the absent default route is what makes it impossible.
        """
        result, _record, _cell = probe
        inside = json.loads((result.out_dir / "posture-inside.json")
                            .read_text(encoding="utf-8"))
        assert inside["network"]["declared"] == "none", (
            f"{result.row.host}: this row declared network "
            f"{inside['network']['declared']!r}")
        assert not inside["network"]["default_route"], (
            f"{result.row.host}: the documented-invocation row had a DEFAULT "
            f"ROUTE via {inside['network']['routed_via']}")
        host_side = json.loads((result.out_dir / "posture-host.json")
                               .read_text(encoding="utf-8"))
        if host_side["captured"]:
            assert not host_side["mismatches"], (
                f"{result.row.host}: docker's own record of this run disagrees "
                f"with posture.yaml: {host_side['mismatches']}")

    def test_it_took_no_credential(self, probe):
        result, _record, _cell = probe
        assert not result.credential_census, (
            f"{result.row.host}: a credential census exists on a row defined to "
            f"take no credential: {result.credential_census}")
        assert harness.credential_mode(result.row, result) == "discovery-only", (
            f"{result.row.host}: the row's credential mode reads "
            f"{harness.credential_mode(result.row, result)!r}. A credential-free "
            f"row that really exercised the host is `discovery-only`; anything "
            f"else means the census will count this host as covered by a mode it "
            f"did not take.")

    def test_the_shim_was_installed_before_anything_was_typed(self, probe):
        """Order matters: typing into a session with no shim proves nothing.

        `install-host` prints `[SHIM_INSTALLED] <path>`, and the entrypoint
        refuses to type at all if the stage step exits non-zero — so this reads
        the receipt rather than trusting the ordering.
        """
        result, _record, _cell = probe
        assert "[SHIM_INSTALLED]" in result.stdout, (
            f"{result.row.host}: no `[SHIM_INSTALLED]` line in the container's "
            f"stdout, so nothing proves the shim was on disk when the session "
            f"opened.\n--- container stdout (tail) ---\n{result.stdout[-2000:]}")


class TestTheSessionWasRealAndFresh:
    def test_a_verdict_was_written(self, probe):
        result, record, _cell = probe
        assert record, (
            f"{result.row.host}: no invocation.json — the row produced no "
            f"verdict, which is worse than a negative one.\n"
            f"--- container stdout (tail) ---\n{result.stdout[-3000:]}")

    def test_the_session_announced_itself(self, probe):
        """The host really started, rather than dying before its first paint.

        Without this, "the string was refused" and "the host never ran" look
        identical from the outside: both produce zero model requests.
        """
        result, record, cell = probe
        transcript = (result.out_dir / "invocation-transcript.txt")
        assert record.get("session_ready"), (
            f"{result.row.host}: the session never printed its ready marker "
            f"{cell.get('ready')!r} within {cell.get('ready_timeout_s')}s, so "
            f"nothing here can distinguish a refused command from a host that "
            f"never started.\n"
            f"--- transcript (tail) ---\n"
            f"{transcript.read_text(encoding='utf-8')[-2000:] if transcript.is_file() else '<none>'}")
        assert record.get("enters_sent"), (
            f"{result.row.host}: the row typed the string and never submitted it")

    def test_it_typed_the_published_string_and_not_a_paraphrase(self, probe):
        result, record, cell = probe
        expected = str(cell["typed"]).replace("{plan}",
                                              f"plans/{record.get('marker')}.md")
        assert record.get("typed") == expected, (
            f"{result.row.host}: the row typed {record.get('typed')!r}, not the "
            f"published {expected!r}")
        assert str(record.get("typed", "")).startswith("/multi-llm "), (
            f"{result.row.host}: the whole question is whether the leading "
            f"`/multi-llm` resolves, and this row did not type it")
        assert record.get("marker") and record["marker"] in record["typed"], (
            f"{result.row.host}: the argument carries no per-run marker, so a "
            f"fact could be satisfied by text left over from an earlier row")


class TestTheHostsVerdictMatchesWhatWasMeasured:
    """The row this module exists for, and it fails in both directions."""

    def test_the_documented_string_dispatches_exactly_where_it_is_measured_to(
            self, probe):
        result, record, cell = probe
        host = result.row.host
        declared = bool(cell.get("resolves"))
        observed = bool(record.get("resolved"))
        assert observed == declared, (
            f"{host}: host_matrix.yaml records that typing "
            f"{record.get('typed')!r} into a fresh session "
            f"{'DOES' if declared else 'does NOT'} reach our dispatcher, and this "
            f"run observed the opposite.\n"
            f"  facts seen    : {record.get('facts_seen')}\n"
            f"  facts missing : {record.get('facts_missing')}\n"
            f"  stub requests : {record.get('stub_requests')}\n"
            f"  refusal seen  : {record.get('refusal_observed')}\n"
            f"  stop reason   : {record.get('stop_reason')}\n"
            f"  bundle        : {result.out_dir}\n"
            + ("\nA host that has GAINED the documented spelling is good news "
               "with a chore attached: flip `resolves` to true, and revisit the "
               "documentation correction this module's docstring records — it is "
               "now stale."
               if observed else
               "\nA host that has LOST it is a shipped-artifact regression: "
               "README.md and docs/hosts.md tell every reader to type this."))

    def test_a_resolving_host_carried_every_fact_into_the_turn(self, probe):
        """Not "something arrived" — each of D5's four claims, one at a time."""
        result, record, cell = probe
        if not cell.get("resolves"):
            pytest.skip(
                f"{result.row.host}: measured NOT to take the documented "
                f"spelling — graded by "
                f"test_a_non_resolving_host_refused_it_visibly_and_sent_nothing")
        seen = set(record.get("facts_seen") or [])
        host = result.row.host
        assert "shim_body" in seen, (
            f"{host}: the turn carried no shim text, so the host never expanded "
            f"its own installed command — seen: {sorted(seen)}")
        assert "shim_dispatcher_block" in seen, (
            f"{host}: the shim's dispatcher block did not reach the turn with "
            f"the skill directory substituted — seen: {sorted(seen)}")
        assert "skill_dir_placeholder_unfilled" not in seen, (
            f"{host}: `{{SKILL_DIR}}` survived into the prompt UNSUBSTITUTED. "
            f"The command resolved and the shim it resolved to is broken, which "
            f"is a worse outcome than not resolving at all.")
        assert "unattended_default_baked" in seen, (
            f"{host}: the turn does not carry `--host {host} --yes "
            f"--exec-subagents`. D5's 'unattended is the default' bullet on the "
            f"README is a claim about this exact string — seen: {sorted(seen)}")
        assert "mode_flag_with_argument" in seen, (
            f"{host}: `--review-plan` and the plan path did not arrive adjacent, "
            f"so this host's parser dropped, reordered or word-split the "
            f"argument — seen: {sorted(seen)}"
            + ("" if "argument_marker" in seen else
               " (and the marker did not arrive at all)"))
        assert "argument_marker" in seen, (
            f"{host}: the distinguishable argument never reached the model")

    def test_a_non_resolving_host_refused_it_visibly_and_sent_nothing(self, probe):
        """A refusal has to be LOUD and total, or the finding is a different one.

        A host that swallowed `/multi-llm …` into a normal turn would be far
        worse than one that rejects it: the user would get a plausible-looking
        answer with no workflow behind it. So both halves are asserted — the
        host said so, and it spent nothing saying it.
        """
        result, record, cell = probe
        if cell.get("resolves"):
            pytest.skip(
                f"{result.row.host}: measured to take the documented spelling — "
                f"graded by test_a_resolving_host_carried_every_fact_into_the_turn")
        host = result.row.host
        assert cell.get("refusal"), (
            f"{host}: declared `resolves: false` with no `refusal` pattern, so "
            f"the row cannot tell a rejection from a host that never started")
        assert record.get("refusal_observed"), (
            f"{host}: the documented spelling did not resolve AND the host never "
            f"printed {cell['refusal']!r}. Silent non-resolution is the worse "
            f"failure — the user gets an ordinary answer with no workflow behind "
            f"it — so this is graded as a finding of its own.\n"
            f"  stop reason: {record.get('stop_reason')}\n"
            f"  bundle     : {result.out_dir}")
        assert record.get("stub_requests") == 0, (
            f"{host}: refused the command and still sent "
            f"{record.get('stub_requests')} model request(s). A refusal that "
            f"spends a turn is not a refusal.")
        assert "shim_body" not in set(record.get("facts_seen") or []), (
            f"{host}: refused the command and the shim text reached the model "
            f"anyway — the two cannot both be true, so the probe is wrong")


class TestTheDocumentsStateWhatEachHostActuallyTakes:
    """D5's deliverable: where a host will not take the published spelling, the
    fix is the document.

    This is the one assertion in the module that grades a **shipped artifact**
    rather than a run, and it is deliberately not silent about a host measured
    `resolves: false`: README's `### Codex and OpenCode` block and the sentence
    `render_capabilities.py` renders into every per-host quickstart in
    `docs/hosts.md` both tell every reader of every installable host to type
    `/multi-llm …`, and for codex that is measurably false.

    It is graded here rather than in a plain-CI module because the measurement
    that justifies it lives here — but it reads only committed text, so it is
    reproducible from the bundle without Docker once the row has run.
    """

    @pytest.mark.parametrize("host", INSTALLABLE)
    def test_the_documents_state_the_syntax_each_host_actually_takes(self, host):
        cell = _cell(host)
        if cell.get("resolves"):
            pytest.skip(f"{host}: takes the published spelling, so the documents "
                        f"are already correct for it")
        alternative = cell.get("alternative")
        assert alternative, (
            f"{host}: measured not to take `/multi-llm …` and the matrix records "
            f"no `alternative` spelling, so the correction owed to the documents "
            f"cannot even be stated")
        qualifier = str(cell.get("doc_qualifier") or "")
        assert qualifier, (
            f"{host}: no `doc_qualifier` in the matrix cell, so this guard has no "
            f"way to recognise a page that HAS been corrected and would stay red "
            f"forever")
        pages = {name: (REPO_ROOT / name) for name in PUBLISHED_IN}
        offenders = []
        for name, path in pages.items():
            if not path.is_file():
                offenders.append(f"{name}: missing")
                continue
            text = path.read_text(encoding="utf-8")
            # An unqualified claim is one that publishes the slash spelling and
            # nowhere says this host does not take it. The qualifier is
            # deliberately loose — the wording belongs to whoever owns the page.
            claims_slash = "/multi-llm " in text or "/multi-llm <" in text
            if claims_slash and not re.search(qualifier, text, re.IGNORECASE):
                offenders.append(name)
        assert not offenders, (
            f"{host} does NOT accept `/multi-llm …` (measured — see this "
            f"module's docstring and host_matrix.yaml), and {offenders} still "
            f"publish that spelling for every installable host without "
            f"qualifying it.\n\n"
            f"THE CORRECTION IS THE DOCUMENT, NOT THIS TEST (D5).\n"
            f"  * docs/hosts.md is GENERATED: the sentence comes from this "
            f"host's `invocation:` field in hosts/capabilities.yaml, rendered by "
            f"skills/multi-llm/hosts/render_capabilities.py's per-host "
            f"quickstart. Fix the FIELD (and its provenance) and re-render; a "
            f"hand-edit inside the markers cannot ship.\n"
            f"  * for {host} that field should say: it installs as a SKILL, which "
            f"is model-invoked rather than typed, so ask for it by name — "
            f"`{alternative}` — because the composer answers `/multi-llm` with "
            f"\"Unrecognized command\".\n"
            f"  * README.md's `### Codex and OpenCode` block carries the same "
            f"claim by hand (the `/multi-llm plans/my-feature.md` code block, and "
            f"the `Invoke with …` line in Modes) and needs the same split.\n"
            f"  * this guard goes green again once EVERY page listed above "
            f"matches /{qualifier}/i — it grades the substance, not a sentence, "
            f"but it grades each page separately.\n"
            f"Both files and the renderer are owned outside this module, which "
            f"is why this fails loudly instead of editing them.")


def _yaml_cell(host: str) -> dict:      # pragma: no cover - diagnostic helper
    """The raw cell, for reading a failure back out of the tree by hand."""
    path = harness.HERE / "host_matrix.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ((data.get("hosts") or {}).get(host) or {}).get(
        "documented_invocation") or {}
