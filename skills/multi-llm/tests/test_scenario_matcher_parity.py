#!/usr/bin/env python3
"""Every invoker of a scenario resolves it to the same bytes (C4).

``tests/scenario_matcher.py`` exists so that the two processes which have to
agree about what a scenario means cannot drift apart. This file is the assertion
that they haven't. Plain CI: no Docker, no ``container`` marker, no network — a
parity guard that only runs where the container lane runs is a parity guard that
does not run.

**Half of the claim is assertable today, and half is not, and this file is
explicit about which is which.**

*Assertable now* — ``tests/mocks/mock_llm.py``, the fake provider CLI, driven as
a **subprocess** through a launcher named after the provider, exactly as the rest
of the suite drives it. That matters: the parity claim is about the real entry
point (argv parsing, provider detection, the env-to-parameter adapter, the wire
format wrapper), not about an internal function call that happens to share a
module. For every shipped scenario and a representative prompt per pattern, the
content the binary prints must be *byte-identical* to what
``scenario_matcher`` resolves for the same inputs — and where the matcher says
"this scenario has nothing to say", the binary must land on its documented
generic default rather than on some third answer.

*Assertable now too* — ``tests/container/stub_server.py``, the HTTP stub the real
CLIs are redirected at inside the container. It is driven the way the fake binary
is: as the **real process**, started as a subprocess on loopback and asked over
HTTP, on each of its three protocol surfaces, streamed and not. No Docker and no
``container`` marker — the stub is a stdlib server and needs neither, and a
parity guard that only runs where the container lane runs is a parity guard that
does not run.

The third thing asserted here is the strictness both invokers inherit together.
``on_unmatched: fail`` makes an unanswered prompt raise in the matcher, exit 8 in
the fake binary and answer a machine-readable error with an ``unmatched: true``
log record in the stub — the two invokers refusing the same set. Absent the key,
behaviour is exactly what it always was, which every scenario that predates
``container_full_chain.yaml`` is the control for.

The fourth is the container scenario's own non-overlap: **no pattern may match a
turn it does not own**, cross-matched over a corpus of recorded prompts
(``tests/golden/container_turns/``) rather than generated probes. A cross-match
is not a benign miss — it hands one phase's answer to another phase's validator,
under a schema that phase never asked for.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

# The suite's own convention for reaching sibling test packages.
sys.path.insert(0, str(Path(__file__).parent))

import scenario_matcher  # noqa: E402
from mocks.mock_llm import default_response_content  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
MOCK_LLM = TESTS_DIR / "mocks" / "mock_llm.py"
SCENARIO_DIR = TESTS_DIR / "fixtures" / "e2e" / "scenarios"
STUB_SERVER = TESTS_DIR / "container" / "stub_server.py"

#: The identity the parity sweep runs under. cursor-agent's wire format is the
#: thinnest of the six (``{"type": "result", "result": <content>}``), so the
#: unwrap that gets us back to the *content* adds nothing of its own to compare.
PROBE_PROVIDER = "cursor-agent"

#: Prompts no shipped pattern answers — the fall-through half of the parity
#: claim. Asserted to match nothing, so a future scenario that starts answering
#: one of them fails here rather than silently weakening the case.
UNMATCHED_PROBES = (
    "zzz qqq unmatchable probe 4f3a",
    "",
)

#: Every scenario the repo ships, in a stable order.
SHIPPED_SCENARIOS = tuple(sorted(SCENARIO_DIR.glob("*.yaml")))

#: The container lane's own scenario, and the one place ``on_unmatched: fail``
#: is declared by a shipped file. It is held out of the generic sweeps below
#: because both of their assumptions stop holding for it: :func:`probe_for_branch`
#: derives a probe by collapsing ``.*`` to a space, which cannot survive a
#: negative lookahead, and the fake binary answers an unmatched probe by exiting
#: 8 rather than by printing its documented default. It gets its own sweep — a
#: stricter one, over recorded prompts rather than generated ones — further down.
CONTAINER_SCENARIO = SCENARIO_DIR / "container_full_chain.yaml"

#: The recorded corpus that sweep runs over. See its manifest for what each file
#: is and why the side turns are the interesting half.
TURNS_DIR = TESTS_DIR / "golden" / "container_turns"
TURNS_MANIFEST = TURNS_DIR / "manifest.yaml"

#: T4.9's deny rows' scenario (plans/opencode-validator-false-rate-limit-and-
#: denied-turn.md): `container_full_chain.yaml`'s rules VERBATIM plus one
#: `scripted_tool_call:` block. Held out for the same two reasons, and covered
#: by equality instead: :func:`test_the_deny_scenario_is_the_chain_scenario`
#: pins its `prompts:` and `on_unmatched` to the chain scenario's, so every
#: recorded-turn assertion :class:`TestTheContainerScenario` makes of the chain
#: rules is a statement about this file's rules too.
DENY_SCENARIO = SCENARIO_DIR / "scripted_opencode_read_deny.yaml"
STRICT_SCENARIOS = (CONTAINER_SCENARIO, DENY_SCENARIO)

GENERIC_SCENARIOS = tuple(p for p in SHIPPED_SCENARIOS
                          if p not in STRICT_SCENARIOS)


# --------------------------------------------------------------------------- #
# Driving the real binary
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def launcher(tmp_path_factory) -> list[str]:
    """A provider-named launcher for mock_llm.py, and the argv prefix to run it.

    Module-scoped because this file spawns one subprocess per probe and the
    launcher is immutable; installing it per test would triple the runtime for
    nothing.

    The launcher's *name* carries the provider identity (mock_llm.py reads
    ``sys.argv[0]``), which is the whole reason this parity test drives a
    subprocess rather than calling in: a provider the binary misdetects resolves
    a different default, and an in-process call would never see it.
    """
    bin_dir = tmp_path_factory.mktemp("scenario-parity-bin")
    if os.name == "nt":  # pragma: no cover - POSIX CI
        target = bin_dir / f"{PROBE_PROVIDER}.py"
        shutil.copyfile(MOCK_LLM, target)
        shutil.copyfile(TESTS_DIR / "scenario_matcher.py", bin_dir / "scenario_matcher.py")
        return [sys.executable, str(target)]

    target = bin_dir / PROBE_PROVIDER
    try:
        target.symlink_to(MOCK_LLM)
    except (OSError, NotImplementedError):  # pragma: no cover
        shutil.copyfile(MOCK_LLM, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        shutil.copyfile(TESTS_DIR / "scenario_matcher.py", bin_dir / "scenario_matcher.py")
    return [str(target)]


def run_mock(
    launcher: list[str],
    prompt: str,
    scenario: Path | None = None,
    **extra_env: str,
) -> subprocess.CompletedProcess:
    """Run the fake CLI once, the way production would."""
    env = os.environ.copy()
    env["MULTI_LLM_TEST_MODE"] = "1"
    # A developer's own scenario/fixture exports must not reach the probe.
    for leaked in ("MOCK_LLM_FIXTURE", "MOCK_LLM_ECHO", "MOCK_LLM_CONFIG",
                   "MOCK_LLM_SCENARIO", "MOCK_LLM_TURN_INDEX"):
        env.pop(leaked, None)
    if scenario is not None:
        env["MOCK_LLM_SCENARIO"] = str(scenario)
    env.update(extra_env)
    return subprocess.run(
        launcher + [prompt],
        capture_output=True,
        # The empty probe leaves argv with no prompt, at which point the mock
        # falls back to its stdin transport; an inherited stdin would make that
        # probe depend on whatever pytest was launched with.
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=30,
    )


def printed_content(result: subprocess.CompletedProcess) -> str:
    """Unwrap cursor-agent's envelope back to the content the mock resolved."""
    return json.loads(result.stdout)["result"]


# --------------------------------------------------------------------------- #
# Representative prompts, derived from the patterns themselves
# --------------------------------------------------------------------------- #


def probe_for_branch(branch: str) -> str:
    """A prompt that should match one alternation branch of a scenario pattern.

    Every shipped pattern is an alternation of literal words joined by ``.*``
    (``review.*implementation plan|reviewing.*plan|...``), so collapsing ``.*``
    to a space yields the prompt the branch was written for. The generator is
    deliberately dumb; :func:`probes_for_scenario` *checks* its output against
    the real pattern, so a future pattern it cannot handle fails loudly instead
    of quietly producing a probe that exercises nothing.
    """
    return re.sub(r"\.\*", " ", branch).strip()


def probes_for_scenario(scenario: scenario_matcher.Scenario) -> list[str]:
    """One representative prompt per alternation branch of every rule."""
    probes: list[str] = []
    for rule in scenario.rules:
        for branch in rule.pattern.split("|"):
            probe = probe_for_branch(branch)
            assert re.search(rule.pattern, probe, re.IGNORECASE), (
                f"{scenario.path.name}: the generated probe {probe!r} does not "
                f"match the pattern {rule.pattern!r} it was derived from. The "
                f"pattern has outgrown probe_for_branch(); give it a "
                f"hand-written representative prompt rather than leaving it "
                f"unexercised."
            )
            probes.append(probe)
    return probes


def parity_cases() -> list[tuple[Path, str]]:
    """(scenario, prompt) for every pattern branch, plus the fall-through probes."""
    cases: list[tuple[Path, str]] = []
    for scenario_path in GENERIC_SCENARIOS:
        scenario = scenario_matcher.load_scenario(scenario_path)
        assert scenario is not None, f"{scenario_path} did not load"
        probes = probes_for_scenario(scenario)
        assert probes, (
            f"{scenario_path.name} contributed no probes: it declares no rule "
            f"that can answer anything, so this sweep would cover it vacuously."
        )
        for probe in probes + list(UNMATCHED_PROBES):
            cases.append((scenario_path, probe))
    return cases


PARITY_CASES = parity_cases()


# --------------------------------------------------------------------------- #
# The half that exists: the fake binary against the shared matcher
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("scenario_path", "prompt"),
    PARITY_CASES,
    ids=[f"{path.stem}-{index}" for index, (path, _) in enumerate(PARITY_CASES)],
)
def test_mock_llm_resolves_what_the_matcher_resolves(launcher, scenario_path, prompt):
    """The binary's answer is the matcher's answer, byte for byte."""
    scenario = scenario_matcher.load_scenario(scenario_path)
    resolution = scenario.resolve(prompt, provider=PROBE_PROVIDER)

    result = run_mock(launcher, prompt, scenario=scenario_path)
    assert result.returncode == 0, result.stderr
    content = printed_content(result)

    if resolution is None:
        # "This scenario has nothing to say" has to mean the same thing on both
        # sides too, or the fall-through cases would be free passes.
        assert content == default_response_content(PROBE_PROVIDER), (
            f"the matcher resolved nothing for {prompt!r} against "
            f"{scenario_path.name}, but the binary answered something other "
            f"than its documented default"
        )
    else:
        assert content == resolution.content


def test_every_shipped_scenario_is_covered():
    """The sweep really did touch every scenario file the repo ships.

    The container scenario is covered by :class:`TestTheContainerScenario`
    instead, so the two together still account for every file in the directory —
    which this asserts, rather than letting a new scenario land uncovered.
    """
    covered = {path for path, _ in PARITY_CASES}
    assert covered == set(GENERIC_SCENARIOS)
    assert covered | set(STRICT_SCENARIOS) == set(SHIPPED_SCENARIOS)
    assert all(path.is_file() for path in STRICT_SCENARIOS)
    assert len(SHIPPED_SCENARIOS) >= 6


def test_unmatched_probes_really_match_nothing():
    """The fall-through half is only meaningful while these match no rule."""
    for scenario_path in SHIPPED_SCENARIOS:
        scenario = scenario_matcher.load_scenario(scenario_path)
        for probe in UNMATCHED_PROBES:
            assert scenario.matching_rules(probe, PROBE_PROVIDER) == [], (
                f"{scenario_path.name} now answers the probe {probe!r}, which "
                f"this file relies on being unanswered"
            )


def test_mock_llm_keeps_no_matcher_of_its_own():
    """The seam most likely to rot: a second copy growing back in the binary.

    ``mock_llm.py`` used to own ``load_scenario``/``match_prompt_pattern``. If
    either name reappears there, someone has re-implemented the scenario
    semantics next to the shared definition instead of extending it.
    """
    from mocks import mock_llm

    for name in ("load_scenario", "match_prompt_pattern"):
        assert not hasattr(mock_llm, name), (
            f"mock_llm.{name} is back. Scenario semantics live in "
            f"tests/scenario_matcher.py, which the HTTP stub imports too; a "
            f"copy here is how the two lanes start disagreeing about what a "
            f"scenario means."
        )


# --------------------------------------------------------------------------- #
# on_unmatched: fail — inherited by both invokers, together
# --------------------------------------------------------------------------- #


PROMPT_BODY = "a distinctive prompt body no scenario answers: kumquat-72"


def write_scenario(tmp_path: Path, body: str) -> Path:
    """Write a scenario next to a fixture the strict cases can match against."""
    (tmp_path / "answer.json").write_text('{"answered": true}', encoding="utf-8")
    path = tmp_path / "scenario.yaml"
    path.write_text(body, encoding="utf-8")
    return path


STRICT_SCENARIO = """
name: "Strict Scenario"
on_unmatched: fail
prompts:
  - pattern: "hello.*world"
    fixture: "answer.json"
"""

LENIENT_SCENARIO = STRICT_SCENARIO.replace("on_unmatched: fail\n", "")


def test_only_the_container_scenario_declares_on_unmatched():
    """The control for "this refactor changed nothing that already existed".

    ``on_unmatched`` is purely additive, and every scenario that predates it
    still resolves exactly as it always did. The container lane's own scenario
    is the one opt-in, and it brings its own tests
    (:class:`TestTheContainerScenario`) — which is the deal the key was added
    under: a scenario that declares it must assert what it buys.
    """
    for scenario_path in GENERIC_SCENARIOS:
        scenario = scenario_matcher.load_scenario(scenario_path)
        assert scenario.on_unmatched == scenario_matcher.ON_UNMATCHED_DEFAULT
        assert not scenario.strict

    for path in STRICT_SCENARIOS:
        container = scenario_matcher.load_scenario(path)
        assert container.on_unmatched == scenario_matcher.ON_UNMATCHED_FAIL
        assert container.strict


def test_the_deny_scenario_is_the_chain_scenario():
    """The one other strict scenario is the chain's rules plus a script.

    Compared as parsed YAML, so a comment may differ and a rule may not. This
    is what lets it skip the generic sweep: it is covered by the chain
    scenario's recorded-turn sweep, rule for rule.
    """
    import yaml

    deny = yaml.safe_load(DENY_SCENARIO.read_text(encoding="utf-8"))
    chain = yaml.safe_load(CONTAINER_SCENARIO.read_text(encoding="utf-8"))
    assert deny["prompts"] == chain["prompts"], (
        f"{DENY_SCENARIO.name} drifted from {CONTAINER_SCENARIO.name}: copy the "
        f"chain scenario's `prompts:` list across verbatim")
    assert deny["on_unmatched"] == chain["on_unmatched"]
    assert set(deny) - set(chain) == {"scripted_tool_call"}


def test_matcher_raises_on_unmatched_prompt(tmp_path):
    """The matcher refuses rather than inventing an answer."""
    path = write_scenario(tmp_path, STRICT_SCENARIO)
    scenario = scenario_matcher.load_scenario(path)

    with pytest.raises(scenario_matcher.UnmatchedPrompt) as caught:
        scenario.resolve(PROMPT_BODY, provider="codex")

    error = caught.value
    assert error.scenario_name == "Strict Scenario"
    assert error.prompt_sha256 == scenario_matcher.prompt_digest(PROMPT_BODY)
    assert error.prompt_bytes == len(PROMPT_BODY.encode("utf-8"))
    # The whole point of carrying a digest is that the body never travels.
    assert PROMPT_BODY not in str(error)
    assert error.prompt_sha256 in str(error)
    assert "Strict Scenario" in str(error)


def test_matcher_still_answers_a_match_under_on_unmatched_fail(tmp_path):
    """Strictness is about the *miss*; a hit resolves exactly as before."""
    path = write_scenario(tmp_path, STRICT_SCENARIO)
    scenario = scenario_matcher.load_scenario(path)
    resolution = scenario.resolve("hello there world", provider="codex")
    assert resolution is not None
    assert resolution.content == '{"answered": true}'
    assert resolution.source == scenario_matcher.SOURCE_PATTERN


def test_matcher_returns_none_without_the_key(tmp_path):
    """Absent the key, an unmatched prompt is still the historical ``None``."""
    path = write_scenario(tmp_path, LENIENT_SCENARIO)
    scenario = scenario_matcher.load_scenario(path)
    assert scenario.resolve(PROMPT_BODY, provider="codex") is None


def test_matcher_raises_on_a_missing_fixture_only_when_strict(tmp_path):
    """A matched rule naming a fixture that is not on disk.

    Lenient: falls through exactly as it always has (several shipped scenarios
    name fixtures that do not resolve). Strict: refuses, because answering a
    generic default while claiming to cover a named fixture is the leniency the
    key exists to remove.
    """
    body = STRICT_SCENARIO.replace('fixture: "answer.json"', 'fixture: "absent.json"')
    strict = scenario_matcher.load_scenario(write_scenario(tmp_path, body))
    with pytest.raises(scenario_matcher.MissingFixture) as caught:
        strict.resolve("hello world")
    assert "absent.json" in str(caught.value)
    assert PROMPT_BODY not in str(caught.value)

    lenient_dir = tmp_path / "lenient"
    lenient_dir.mkdir()
    lenient = scenario_matcher.load_scenario(
        write_scenario(lenient_dir, body.replace("on_unmatched: fail\n", ""))
    )
    assert lenient.resolve("hello world") is None


def test_mock_llm_exits_non_zero_on_an_unmatched_prompt(launcher, tmp_path):
    """The fake binary inherits the strictness rather than staying lenient."""
    path = write_scenario(tmp_path, STRICT_SCENARIO)

    result = run_mock(launcher, PROMPT_BODY, scenario=path)

    assert result.returncode != 0
    assert result.returncode == 8, "the unmatched-prompt code, not a generic failure"
    assert scenario_matcher.prompt_digest(PROMPT_BODY) in result.stderr
    assert "Strict Scenario" in result.stderr
    assert PROMPT_BODY not in result.stderr
    # Nothing shaped like an answer may reach stdout: a row that read one would
    # grade a fixture that never ran.
    assert result.stdout.strip() == ""


def test_mock_llm_is_unchanged_without_the_key(launcher, tmp_path):
    """The same prompt, the same scenario, minus the key: exit 0 and the default."""
    path = write_scenario(tmp_path, LENIENT_SCENARIO)

    result = run_mock(launcher, PROMPT_BODY, scenario=path)

    assert result.returncode == 0, result.stderr
    assert printed_content(result) == default_response_content(PROBE_PROVIDER)


def test_mock_llm_answers_a_match_under_on_unmatched_fail(launcher, tmp_path):
    """A strict scenario that *does* match still answers normally, exit 0."""
    path = write_scenario(tmp_path, STRICT_SCENARIO)

    result = run_mock(launcher, "hello there world", scenario=path)

    assert result.returncode == 0, result.stderr
    assert printed_content(result) == '{"answered": true}'


def test_turn_index_is_the_second_selection_stage(launcher, tmp_path):
    """pattern -> turn -> fallback, in both invokers.

    The fake binary has no session of its own, so its turn index arrives in
    ``MOCK_LLM_TURN_INDEX``; the stub will derive the same number from the
    per-exec id in its base URL. Both hand it to the same matcher.
    """
    (tmp_path / "turn0.json").write_text('{"turn": 0}', encoding="utf-8")
    (tmp_path / "fallback.json").write_text('{"fallback": true}', encoding="utf-8")
    path = write_scenario(
        tmp_path,
        STRICT_SCENARIO
        + '\nturns:\n  - fixture: "turn0.json"\nfallback:\n  fixture: "fallback.json"\n',
    )
    scenario = scenario_matcher.load_scenario(path)

    turned = scenario.resolve(PROMPT_BODY, turn_index=0)
    assert turned.source == scenario_matcher.SOURCE_TURN
    assert turned.content == '{"turn": 0}'

    # Past the end of the turn list, the declared fallback answers -- so even a
    # strict scenario never raises while it declares one.
    fell_back = scenario.resolve(PROMPT_BODY, turn_index=9)
    assert fell_back.source == scenario_matcher.SOURCE_FALLBACK
    assert fell_back.content == '{"fallback": true}'

    result = run_mock(launcher, PROMPT_BODY, scenario=path, MOCK_LLM_TURN_INDEX="0")
    assert result.returncode == 0, result.stderr
    assert printed_content(result) == turned.content

    result = run_mock(launcher, PROMPT_BODY, scenario=path, MOCK_LLM_TURN_INDEX="9")
    assert result.returncode == 0, result.stderr
    assert printed_content(result) == fell_back.content


def test_a_bad_on_unmatched_value_is_refused_at_load(tmp_path):
    """A key this module owns, spelled wrong, fails where it was written."""
    body = STRICT_SCENARIO.replace("on_unmatched: fail", "on_unmatched: maybe")
    with pytest.raises(scenario_matcher.ScenarioConfigError):
        scenario_matcher.load_scenario(write_scenario(tmp_path, body))


def test_an_unavailable_scenario_is_still_just_none(tmp_path):
    """Unreadable is not the same as invalid, and has always degraded quietly."""
    assert scenario_matcher.load_scenario(tmp_path / "nope.yaml") is None
    broken = tmp_path / "broken.yaml"
    broken.write_text("::: not: yaml: [", encoding="utf-8")
    assert scenario_matcher.load_scenario(broken) is None


# --------------------------------------------------------------------------- #
# The stub server: the second invoker, driven as the server it is
# --------------------------------------------------------------------------- #


def test_stub_server_reaches_for_the_shared_matcher():
    """Whatever ``stub_server.py`` exists must *import* the matcher, not copy it.

    A source-level check, not a behavioural one, and deliberately the weakest
    assertion that is still worth making: it cannot say the stub resolves the
    same bytes (that is the owed test below), but it does catch the one failure
    this whole extraction exists to prevent — a second pattern loop growing up
    inside the stub. Skips while no stub exists, since there is then nothing to
    constrain.
    """
    if not STUB_SERVER.exists():
        pytest.skip("tests/container/stub_server.py has not landed yet")

    source = STUB_SERVER.read_text(encoding="utf-8")
    assert "scenario_matcher" in source, (
        "stub_server.py resolves prompts without naming tests/scenario_matcher, "
        "so it is answering from a second definition of what a scenario means. "
        "The fake binary and the stub disagreeing is the seam this module was "
        "extracted to close."
    )


@pytest.fixture(scope="module")
def stub_factory(tmp_path_factory):
    """Start ``stub_server.py`` per scenario, on loopback, and reap them all.

    The real process, not an import: the parity claim is about the server the
    container runs, whose resolver is wired at startup by ``install_resolver``
    and whose answers are rendered by its own three surface renderers. An
    in-process call to :func:`scenario_matcher.resolve` would compare this module
    against itself.
    """
    import socket
    import urllib.error
    import urllib.request

    root = tmp_path_factory.mktemp("stub-parity")
    started: dict[Path, tuple[str, Path]] = {}
    procs: list[subprocess.Popen] = []

    def _free_port() -> int:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def factory(scenario: Path) -> tuple[str, Path]:
        if scenario in started:
            return started[scenario]
        port = _free_port()
        log = root / f"{scenario.stem}.jsonl"
        env = os.environ.copy()
        # The stub reaches for `tests.scenario_matcher`; inside the image that is
        # under /opt/mllm/skill, which it puts on sys.path itself. Here the skill
        # dir is the parent of this tests package.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(TESTS_DIR.parent), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        proc = subprocess.Popen(
            [sys.executable, str(STUB_SERVER), "--port", str(port),
             "--log", str(log), "--scenario", str(scenario)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        procs.append(proc)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if proc.poll() is not None:  # pragma: no cover - startup failure
                raise AssertionError(
                    f"stub exited {proc.returncode} before serving:\n"
                    f"{proc.stderr.read() if proc.stderr else ''}"
                )
            try:
                with urllib.request.urlopen(f"{base}/__health", timeout=1) as answer:
                    if answer.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.05)
        else:  # pragma: no cover - startup failure
            raise AssertionError("stub never answered /__health")
        started[scenario] = (base, log)
        return started[scenario]

    yield factory

    for proc in procs:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


def _post(base: str, path: str, payload: dict) -> tuple[int, str]:
    """One request, returning ``(status, body)`` and never raising on 4xx/5xx."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"{base}{path}", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            return answer.status, answer.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


#: ``(surface path, request builder, answer extractor)`` per protocol surface.
#: One resolved answer, rendered two ways per surface, so the streamed and
#: non-streamed shapes cannot disagree about what the scenario said.
def _chat_request(prompt: str, stream: bool) -> dict:
    return {"model": "stub-model", "stream": stream,
            "messages": [{"role": "user", "content": prompt}]}


def _responses_request(prompt: str, stream: bool) -> dict:
    return {"model": "stub-model", "stream": stream,
            "input": [{"role": "user",
                       "content": [{"type": "input_text", "text": prompt}]}]}


def _messages_request(prompt: str, stream: bool) -> dict:
    return {"model": "stub-model", "stream": stream,
            "messages": [{"role": "user", "content": prompt}]}


def _sse_payloads(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        chunk = line[len("data: "):]
        if chunk == "[DONE]":
            continue
        out.append(json.loads(chunk))
    return out


def _chat_answer(body: str, stream: bool) -> str:
    if not stream:
        return json.loads(body)["choices"][0]["message"]["content"]
    return "".join(
        frame["choices"][0]["delta"].get("content", "")
        for frame in _sse_payloads(body)
    )


def _responses_answer(body: str, stream: bool) -> str:
    if not stream:
        return json.loads(body)["output"][0]["content"][0]["text"]
    for frame in _sse_payloads(body):
        if frame.get("type") == "response.output_text.done":
            return frame["text"]
    raise AssertionError("no response.output_text.done frame")  # pragma: no cover


def _messages_answer(body: str, stream: bool) -> str:
    if not stream:
        return json.loads(body)["content"][0]["text"]
    return "".join(
        frame["delta"]["text"] for frame in _sse_payloads(body)
        if frame.get("type") == "content_block_delta"
    )


SURFACES = (
    ("chat", "/v1/chat/completions", _chat_request, _chat_answer),
    ("responses", "/v1/responses", _responses_request, _responses_answer),
    ("anthropic", "/v1/messages", _messages_request, _messages_answer),
)


def _ask_every_surface(base: str, prompt: str, exec_id: str) -> dict[str, str]:
    """The same prompt on all three surfaces, streamed and not.

    Each request gets its own per-exec id in the path — the session key the
    wrapper bakes into the base URL inside the container — so the turn counter
    starts at 1 for every probe and no two probes perturb each other.
    """
    answers = {}
    for index, (name, path, build, extract) in enumerate(SURFACES):
        for stream in (False, True):
            status, body = _post(
                base, f"/s/{exec_id}-{index}-{int(stream)}{path}",
                build(prompt, stream))
            assert status == 200, f"{name} stream={stream}: HTTP {status}: {body[:400]}"
            answers[f"{name}-{'stream' if stream else 'body'}"] = extract(body, stream)
    return answers


@pytest.mark.parametrize(
    ("scenario_path", "prompt"),
    PARITY_CASES,
    ids=[f"{path.stem}-{index}" for index, (path, _) in enumerate(PARITY_CASES)],
)
def test_stub_server_resolves_identically_to_the_matcher(
    stub_factory, scenario_path, prompt
):
    """The stub's answer is the matcher's answer, byte for byte, on every surface.

    Including the shape axis: a resolved response is rendered streamed and
    whole from one value, so both renderings must carry the same text. A stub
    that agreed with the matcher only in its non-streaming form would pass a
    self-test and hang every real CLI, which ask for a stream by default.
    """
    base, _log = stub_factory(scenario_path)
    scenario = scenario_matcher.load_scenario(scenario_path)
    # The stub knows no provider identity — it has a request, not an argv[0] —
    # so the matcher is asked the same way, with turn 1: the number the stub's
    # per-session counter hands it for a fresh exec id.
    resolution = scenario.resolve(prompt, turn_index=1)
    expected = resolution.content if resolution is not None else "OK"

    answers = _ask_every_surface(base, prompt, f"probe-{abs(hash(prompt)) % 10**8}")
    for shape, answer in answers.items():
        assert answer == expected, f"{scenario_path.name}/{shape} disagrees"


def _omp_shaped_request(prompt: str, *, system: str, reminder: str) -> dict:
    """A chat/completions body in the shape omp 18.2.2 really sends.

    MEASURED 2026-09-17 against the real binary, prompt on stdin: one
    `POST /v1/chat/completions`, `stream: true` with
    `stream_options.include_usage`, `store: false`, `max_completion_tokens`, six
    tool schemas, a 10,179-byte `system` message, and a user message whose
    `content` is a LIST OF TWO text parts — a `<system-reminder>` block carrying
    the date and the cwd (237 B), then the prompt itself. Every other redirected
    CLI on this lane sends the user prompt as one string.
    """
    return {
        "model": "stub-model", "stream": True, "store": False,
        "max_completion_tokens": 16384,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": reminder},
                                         {"type": "text", "text": prompt}]},
        ],
        "tools": [{"type": "function",
                   "function": {"name": "read", "description": "Read a file."}}],
    }


def test_the_stub_reads_omps_two_part_user_message_as_one_prompt(stub_factory):
    """omp is the first row whose user message is two parts, and it still resolves.

    Three claims, and each one is a way a container row could go quietly wrong:

    * the prompt is read AT ALL — `extract_prompt` walks `messages` and joins
      every text part it finds, so a list-valued `content` must not fall through
      to the unmatched branch (a 422 the CLI reports as a model error);
    * the prompt is read ONCE. omp takes the prompt on stdin *or* on argv and
      both together become TWO user messages, so the sentinel appearing twice in
      one request is the regression that would mean the adapter had started
      rendering both. The transport is declared (`prompt_transport = "stdin"`),
      and this is what the assertion looks like on the wire;
    * a ~10 KB vendor system prompt in front of the product's own text does not
      change which fixture answers. The system message is part of the joined
      prompt by design (claude-code's arrives as a top-level `system` key and is
      joined the same way), so a pattern that matched only by accident of
      position would break here first.
    """
    scenario_path = GENERIC_SCENARIOS[0]
    scenario = scenario_matcher.load_scenario(scenario_path)
    rule = next(r for r in scenario.rules if r.response.fixture)
    sentinel = "MLLM-OMP-STDIN-SENTINEL-7f3a91"
    prompt = f"{probe_for_branch(rule.pattern.split('|')[0])} Marker: {sentinel}"

    # Stand-ins for the vendor bytes, at the measured sizes. The real system
    # prompt is not committed here: it is 10 KB of another project's text, it
    # moves with every omp release, and what this test needs from it is its
    # SHAPE and its SIZE, not its wording.
    system = ("You are a coding agent. " * 460)[:10_179]
    reminder = ("<system-reminder>\nToday: 2026-09-17; current working "
                "directory: '/work/sample-project'\n</system-reminder>")

    base, log = stub_factory(scenario_path)
    status, body = _post(base, "/s/omp-two-part/v1/chat/completions",
                         _omp_shaped_request(prompt, system=system,
                                             reminder=reminder))
    assert status == 200, f"HTTP {status}: {body[:400]}"
    answer = _chat_answer(body, True)

    resolution = scenario.resolve(prompt, turn_index=1)
    assert resolution is not None
    assert answer == resolution.content, (
        "the two-part user message resolved to a different fixture than the "
        "same prompt sent as one string"
    )

    records = [json.loads(line) for line in
               log.read_text(encoding="utf-8").splitlines()]
    mine = [r for r in records if r.get("raw_path", "").startswith("/s/omp-two-part/")]
    assert len(mine) == 1, f"expected exactly one request, got {len(mine)}"
    assert not mine[0]["unmatched"], "the stub could not read the prompt at all"
    assert mine[0]["has_tools"] is True

    # The sentinel, counted over the whole joined prompt — every message, both
    # parts. `prompt_len` is the stub's own measure of what it read, and the log
    # deliberately never carries the body, so the count is asserted here on the
    # bytes this test built and cross-checked against that length.
    from .container import stub_server

    joined = stub_server.extract_prompt(
        _omp_shaped_request(prompt, system=system, reminder=reminder))
    assert joined.count(sentinel) == 1, (
        "the prompt reached the model twice — an argv+stdin regression"
    )
    assert mine[0]["prompt_len"] == len(joined)
    assert len(system) < mine[0]["prompt_len"], (
        "the system prompt is part of what the matcher sees, and it must be"
    )


def test_stub_server_records_the_fixture_that_answered(stub_factory):
    """The log names the fixture, which is what a row's evidence is graded on."""
    scenario_path = GENERIC_SCENARIOS[0]
    scenario = scenario_matcher.load_scenario(scenario_path)
    rule = next(r for r in scenario.rules if r.response.fixture)
    prompt = probe_for_branch(rule.pattern.split("|")[0])

    base, log = stub_factory(scenario_path)
    _ask_every_surface(base, prompt, "fixture-probe")

    # The stub is shared with the sweep above, so the log carries every probe
    # this module has sent it; the prompt digest is what selects ours.
    digest = scenario_matcher.prompt_digest(prompt)
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    answered = [r for r in records
                if r.get("prompt_sha") == digest and r.get("fixture")]
    assert answered, f"no record names a fixture for {digest}"
    resolution = scenario.resolve(prompt, turn_index=1)
    assert {r["fixture"] for r in answered} == {resolution.fixture}


# --------------------------------------------------------------------------- #
# The container lane's scenario: recorded turns, and non-overlap asserted
# --------------------------------------------------------------------------- #


def _load_turns_manifest() -> dict:
    import yaml

    return yaml.safe_load(TURNS_MANIFEST.read_text(encoding="utf-8"))


TURNS = _load_turns_manifest()
RECORDED_TURNS = tuple(
    (name, TURNS_DIR / f"{name}.txt", spec["fixture"])
    for name, spec in TURNS["turns"].items()
)
RECORDED_SIDE_TURNS = tuple(
    (name, TURNS_DIR / "side_turns" / f"{name}.txt", spec["fixture"])
    for name, spec in TURNS["side_turns"].items()
)
ALL_RECORDED = RECORDED_TURNS + RECORDED_SIDE_TURNS


class TestTheContainerScenario:
    """`container_full_chain.yaml`, against the turns a real chain really sent.

    The generic sweep above cannot cover this file: its probes are *derived*
    from the patterns by collapsing ``.*`` to a space, which a negative
    lookahead does not survive, and under ``on_unmatched: fail`` the fake binary
    answers an unmatched probe by exiting 8 rather than by printing a default.
    Both of those are the scenario working as designed, so it gets a sweep whose
    inputs are recorded prompts instead of generated ones — which is the
    stronger test anyway.
    """

    @pytest.mark.parametrize(("name", "path", "fixture"), ALL_RECORDED,
                             ids=[n for n, _, _ in ALL_RECORDED])
    def test_no_pattern_matches_a_turn_it_does_not_own(self, name, path, fixture):
        """Exactly one rule matches each recorded turn, and it is the owner.

        Both halves matter. A **second** match is the failure C4 names: it hands
        one phase's answer to another phase's validator, under a schema that
        phase never asked for, and list order is the only thing standing between
        that and a green run. **Zero** matches is the other half — with
        ``on_unmatched: fail`` the row would fail loudly in the container, but
        failing here instead costs a second rather than a chain.
        """
        assert path.is_file(), f"recorded turn {name} is missing at {path}"
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        prompt = path.read_text(encoding="utf-8")

        matches = scenario.matching_rules(prompt)
        assert len(matches) == 1, (
            f"{name}: {len(matches)} rules match this prompt "
            f"({[m.response.fixture for m in matches]}). Exactly one must: a "
            f"cross-match answers the turn with another phase's document."
        )
        assert matches[0].response.fixture == fixture, (
            f"{name} is answered by {matches[0].response.fixture}, not {fixture}"
        )

    @pytest.mark.parametrize(("side_name", "side_path", "side_fixture"),
                             RECORDED_SIDE_TURNS,
                             ids=[n for n, _, _ in RECORDED_SIDE_TURNS])
    def test_a_side_turn_quoting_any_product_prompt_still_resolves_to_the_side_rule(
        self, side_name, side_path, side_fixture
    ):
        """The property the negative lookahead exists for, over every turn.

        A session-title call quotes the prompt it is titling — goose's carries
        ours in full — so the composite carries both the title marker and a
        product anchor. The title rule has to keep winning for *every* product
        turn, not only for the one a recording happened to catch, because which
        prompt a CLI is titling depends on which phase dispatched it.
        """
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        side = side_path.read_text(encoding="utf-8")
        for name, path, _fixture in RECORDED_TURNS:
            composite = f"{side}\n{path.read_text(encoding='utf-8')}"
            matches = scenario.matching_rules(composite)
            assert len(matches) == 1 and matches[0].response.fixture == side_fixture, (
                f"{side_name} quoting {name} resolved to "
                f"{[m.response.fixture for m in matches]} — the side-turn "
                f"lookahead does not beat that product anchor"
            )

    def test_every_rule_owns_at_least_one_recorded_turn(self):
        """No rule is unexercised, in either direction.

        A pattern with no recorded turn is either dead (the phase stopped
        sending it) or untested (nobody recorded it) — both are worth failing
        for, because `on_unmatched: fail` cannot see a rule that never fires.
        """
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        declared = {rule.response.fixture for rule in scenario.rules}
        recorded = {fixture for _, _, fixture in ALL_RECORDED}
        assert declared == recorded, (
            f"rules with no recorded turn: {sorted(declared - recorded)}; "
            f"recorded turns with no rule: {sorted(recorded - declared)}"
        )

    def test_every_named_fixture_is_on_disk(self):
        """Under `on_unmatched: fail` a missing fixture is a refusal, not a miss."""
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        for rule in scenario.rules:
            path = rule.response.path(scenario.dir)
            assert path is not None and path.is_file(), (
                f"{rule.response.fixture} is named by a rule but is not on disk"
            )

    def test_an_unrecognized_prompt_raises_rather_than_resolving(self):
        """The matcher refuses; it never invents an answer for this scenario."""
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        with pytest.raises(scenario_matcher.UnmatchedPrompt) as caught:
            scenario.resolve(PROMPT_BODY)
        assert PROMPT_BODY not in str(caught.value)
        assert scenario_matcher.prompt_digest(PROMPT_BODY) in str(caught.value)

    def test_the_fake_binary_refuses_the_same_prompt(self, launcher):
        """Invoker one: exit 8, the digest on stderr, nothing on stdout."""
        result = run_mock(launcher, PROMPT_BODY, scenario=CONTAINER_SCENARIO)
        assert result.returncode == 8, result.stderr
        assert scenario_matcher.prompt_digest(PROMPT_BODY) in result.stderr
        assert PROMPT_BODY not in result.stderr
        assert result.stdout.strip() == ""

    def test_the_stub_refuses_the_same_prompt(self, stub_factory):
        """Invoker two: a machine-readable error and an ``unmatched: true`` record.

        The record is what the row is graded on, independently of the CLI's exit
        code — a client that shrugs off a 422 and retries would otherwise surface
        only as an unexplained timeout.
        """
        base, log = stub_factory(CONTAINER_SCENARIO)
        status, body = _post(base, "/s/refusal-probe/v1/chat/completions",
                             _chat_request(PROMPT_BODY, False))
        assert status == 422, f"HTTP {status}: {body[:300]}"
        payload = json.loads(body)
        assert payload["error"]["type"] == "unmatched_prompt"
        assert payload["error"]["prompt_sha"] == \
            scenario_matcher.prompt_digest(PROMPT_BODY)
        assert PROMPT_BODY not in body

        records = [json.loads(line) for line
                   in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        unmatched = [r for r in records if r.get("unmatched")]
        assert unmatched, "the stub answered 422 without recording unmatched: true"
        assert scenario_matcher.prompt_digest(PROMPT_BODY) in \
            {r["prompt_sha"] for r in unmatched}

    @pytest.mark.parametrize(("name", "path", "fixture"), RECORDED_TURNS,
                             ids=[n for n, _, _ in RECORDED_TURNS])
    def test_the_stub_answers_each_recorded_turn_with_its_fixture(
        self, stub_factory, name, path, fixture
    ):
        """End to end, over the wire, for the scenario the container lane runs."""
        base, _log = stub_factory(CONTAINER_SCENARIO)
        scenario = scenario_matcher.load_scenario(CONTAINER_SCENARIO)
        prompt = path.read_text(encoding="utf-8")
        expected = scenario.resolve(prompt, turn_index=1)
        assert expected.fixture == fixture

        answers = _ask_every_surface(base, prompt, f"turn-{name}")
        for shape, answer in answers.items():
            assert answer == expected.content, f"{name}/{shape} disagrees"
