"""aider: exit-0 failures, the answer reducer, and the declined mentions.

Plan: plans/ask-answer-recovery-and-false-success.md §4 — T4.1 (D4.1) and the
provider side of T4.2 (D4.2). The argv pins, the fixture rows and the salvage
sweep live elsewhere (``TestAiderProvider`` in ``test_providers.py``,
``tests/fixtures/aider/manifest.yaml``, ``test_salvage_text.py``); this module
holds what those cannot say:

- **D4.1.** aider exits 0 on every failure, and its error text lands on stdout.
  An exit-0 failure on one of aider's *diagnostic* lines now wins before any
  JSON is extracted — the 2026-09-26 incident's 400 body was JSON and parsed as
  ``success: True``. Which lines are diagnostic is decided by structure (the
  prologue, and whatever follows the trailing ``Tokens:`` line), because an
  ordinary answer has no ANSWER marker and may quote any of that wording.
- **The hook.** ``salvage_text`` is the answer between the prologue and the
  trailing accounting line, the confirmation prompts removed. The salvage
  gate is a JSON gate, so the prose answers are pinned here, not by labels.
- **D4.2.** No ``--yes-always``: the argv transport hands aider ``"s\\n" * 64``
  on stdin, whose bound is pinned below, and an inherited yes-always setting
  is refused before spawn.
"""

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client  # noqa: E402
from utils.llm_client import invoke_with_provider  # noqa: E402
from utils.providers.aider import (  # noqa: E402
    AIDER_DECLINE_ANSWER,
    AIDER_DECLINE_LINES,
    AIDER_YES_ALWAYS_ENV,
    ANSWER_MARKER,
    AiderProvider,
    _CONFIRM_PROMPT_RE,
    _GROUP_PREFERENCE_ECHOES,
    _drop_confirmation_pairs,
)
from utils.providers.base import (  # noqa: E402
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    ProfileUnsupported,
    ReadOnlyInvocationRefused,
)

FIXTURES = Path(__file__).parent / "fixtures" / "aider"
SKILL_ROOT = Path(__file__).parent.parent

MODEL = "openrouter/z-ai/glm-5.2"
OVERFLOW_FALLBACK = "aider: prompt exceeds the model's context window (token count not reported)"
INCIDENT_ERROR = "aider: prompt exceeds the model's context window (1,083,501 > 1,048,576 tokens)"


def _read(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("utf-8")


def _parse(stdout: str) -> dict:
    return AiderProvider().parse_output(stdout, "")


def _salvage(stdout: str):
    return AiderProvider().salvage_text(stdout)


#: The committed `success` row: litellm's cost-map line, the banner, a blank
#: line, the answer, a blank line, `Tokens: 1 sent, 1 received.`.
SUCCESS = _read("success.stdout.txt")
SUCCESS_ANSWER = next(line for line in SUCCESS.splitlines() if line.startswith("[{"))


def _success_shape(answer: str) -> str:
    """The `success` row's bytes with ``answer`` in place of its JSON line."""
    assert SUCCESS.count(SUCCESS_ANSWER) == 1
    return SUCCESS.replace(SUCCESS_ANSWER, answer)


def _marker_shape(answer: str) -> str:
    """The same row as a reasoning model prints it: THINKING, then the marker."""
    return _success_shape(
        "--------------\n► **THINKING**\n\nDraft: Model: none. Tokens: 9 sent, 9 received.\n\n"
        f"------------\n{ANSWER_MARKER}\n\n{answer}"
    )


# --------------------------------------------------------------------------- #
# D4.1: the incident, and aider's own overflow
# --------------------------------------------------------------------------- #

class TestTheIncidentRow:
    """The synthetic `incident-400` row: banner, path list, context warning and
    a `litellm.BadRequestError` carrying OpenRouter's 400 body, exit 0."""

    def test_fails_with_the_body_counts_not_the_estimate(self):
        stdout = _read("incident-400.stdout.txt")
        # The row's two sources really disagree, or the precedence is untested.
        assert "Your estimated chat context of 1,159,634 tokens" in stdout
        assert "you requested about 1083501 tokens" in stdout

        result = _parse(stdout)

        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] == INCIDENT_ERROR, result["error"]
        assert "1,159,634" not in result["error"]

    def test_salvage_is_final_empty(self):
        assert _salvage(_read("incident-400.stdout.txt")) == ""

    def test_a_body_carrying_an_array_still_fails(self):
        """The review-plan `[]` case: an array in the body used to reach
        `save_model_result`, and every element failing validation wrote `[]`."""
        stdout = _read("incident-400-array.stdout.txt")
        assert '"providers":["a"]' in stdout

        result = _parse(stdout)

        assert result["success"] is False
        assert result["error"] == INCIDENT_ERROR
        assert _salvage(stdout) == ""

    def test_the_estimate_supplies_the_counts_only_when_no_body_states_both(self):
        body_without_counts = _read("incident-400.stdout.txt").replace(
            "This endpoint's maximum context length is 1048576 tokens. However, "
            "you requested about 1083501 tokens (1083501 of text input). ",
            "",
        )
        assert "maximum context length" not in body_without_counts

        result = _parse(body_without_counts)

        assert result["error"] == (
            "aider: prompt exceeds the model's context window (1,159,634 > 1,048,576 tokens)"
        )

    def test_the_body_alone_is_an_overflow(self):
        """No aider warning (litellm did not know the model's limit), only the
        endpoint's 400: still the overflow sentence, with the body's counts."""
        lines = [
            line for line in _read("incident-400.stdout.txt").splitlines()
            if not line.startswith(("Your estimated", "To reduce", "- Use", "- Break",
                                    "It's probably safe"))
        ]
        result = _parse("\n".join(lines))

        assert result["error"] == INCIDENT_ERROR

    def test_the_incident_has_no_accounting_line(self):
        """So no request completed and every line is scanned."""
        assert "Tokens:" not in _read("incident-400.stdout.txt")


class TestAidersOwnOverflowBlock:
    """`context-overflow`: `has hit a token limit!` after `Tokens:`, no litellm."""

    def test_error_is_exactly_the_fallback_wording(self):
        stdout = _read("context-overflow.stdout.txt")
        assert "litellm." not in stdout
        assert "Input tokens: ~103 of 0" in stdout

        result = _parse(stdout)

        assert result["success"] is False
        assert result["error"] == OVERFLOW_FALLBACK
        assert "103" not in result["error"]

    def test_salvage_is_final_empty(self):
        assert _salvage(_read("context-overflow.stdout.txt")) == ""

    def test_the_block_comes_after_the_accounting_line(self):
        lines = _read("context-overflow.stdout.txt").splitlines()
        tokens = next(i for i, line in enumerate(lines) if line.startswith("Tokens:"))
        limit = next(i for i, line in enumerate(lines) if "has hit a token limit!" in line)
        assert tokens < limit


class TestTheOverflowWordingStillClassifies:
    """The review lanes classify a failed slot from its ``error`` string alone
    (``classify_streams(error, "")``). Before D4.1 aider's overflow error carried
    its raw block and classified ``context_length_exceeded`` (terminal, with the
    remedy message); the new sentence must keep that classification.
    """

    @pytest.mark.parametrize("error", [OVERFLOW_FALLBACK, INCIDENT_ERROR])
    def test_the_sentence_classifies_as_a_context_overflow(self, error):
        from utils.providers.base import classify_streams

        assert classify_streams(error, "") == "context_length_exceeded"


class TestTheOtherFailureRows:
    @pytest.mark.parametrize("row,prefix", [
        ("auth-error", "litellm.AuthenticationError: "),
        ("error-envelope", "litellm.APIError: "),
        ("exit-zero-failure", "litellm.InternalServerError: "),
        ("invalid-model", "litellm.NotFoundError: "),
        ("rate-limit", "litellm.RateLimitError: "),
        ("truncated", "litellm.InternalServerError: "),
    ])
    def test_the_error_is_the_litellm_line(self, row, prefix):
        stdout = _read(f"{row}.stdout.txt")
        result = _parse(stdout)

        assert result["success"] is False
        assert result["error"].startswith(prefix), result["error"]
        assert result["error"] in stdout
        assert _salvage(stdout) == ""

    def test_an_auth_error_still_classifies_as_auth(self):
        stdout = _read("auth-error.stdout.txt")
        assert AiderProvider().classify_failure(stdout, "", 0) == "auth_failed"


# --------------------------------------------------------------------------- #
# The hook on a real answer
# --------------------------------------------------------------------------- #

class TestTheAnswerAlone:
    def test_the_success_row_is_the_answer_and_nothing_else(self):
        text = _salvage(SUCCESS)

        assert text == SUCCESS_ANSWER
        assert "HTTPSConnectionPool" not in text
        assert "Aider v" not in text
        assert "Tokens:" not in text

    def test_the_multi_assistant_row_is_the_last_answer(self):
        text = _salvage(_read("multi-assistant.stdout.txt"))

        assert text.startswith('[{"id": "S001"')
        assert "FIRST ANSWER" not in text and "Tokens:" not in text

    def test_an_empty_answer_after_the_marker_is_empty(self):
        assert _salvage(_read("ephemeral-only.stdout.txt")) == ""

    @pytest.mark.parametrize("stdout", ["", "   \n", "I could not finish.", "[1, 2]"])
    def test_a_stdout_with_no_banner_is_not_an_answer(self, stdout):
        """aider prints its banner before any request, so a stdout without one
        comes from a run that never reached a model."""
        assert _salvage(stdout) == ""

    def test_the_large_repo_banner_lines_are_prologue(self):
        stdout = _success_shape(SUCCESS_ANSWER).replace(
            "Git repo: .git with 0 files\n",
            "Git repo: .git with 14,420 files\n"
            "Warning: For large repos, consider using --subtree-only and .aiderignore\n"
            "See: https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo\n",
        )
        assert _salvage(stdout) == SUCCESS_ANSWER


class TestAMarkerAnswerQuotingTheFailure:
    """The failure check covers only the no-marker branch."""

    ANSWER = (
        "The run failed with litellm.BadRequestError, as its log shows:\n"
        'litellm.BadRequestError: OpenrouterException - {"error":…}\n'
        '[{"title": "Trim the prompt", "desc": "d", "importance": "HIGH"}]'
    )

    def test_parses_as_success(self):
        result = _parse(_marker_shape(self.ANSWER))

        assert result["success"] is True
        assert result["data"] == [
            {"title": "Trim the prompt", "desc": "d", "importance": "HIGH"}
        ]

    def test_salvage_is_the_text_after_the_marker(self):
        assert _salvage(_marker_shape(self.ANSWER)) == self.ANSWER


#: Markerless answers that quote the failure wording — the ordinary shape, and
#: the one a whole-stdout search would have recorded as a 400.
QUOTING_ANSWERS = {
    "mid-sentence": (
        "The endpoint answered with a litellm.BadRequestError when the context "
        "overflowed, so the fix is to stop inlining the plan."
    ),
    "quoted-log-line": (
        "The log shows:\n"
        'litellm.BadRequestError: OpenrouterException - {"error":…}\n'
        "which means the request was rejected before any model ran."
    ),
    "aiders-own-lines": (
        "aider printed these lines:\n"
        "Model x has hit a token limit!\n"
        "Total tokens: ~103 of 0 -- possibly exhausted context window!\n"
        "Your estimated chat context of 1,159,634 tokens exceeds the 1,048,576 "
        "token limit for x!\n"
        "So the prompt was too large for the model."
    ),
}


class TestMarkerlessAnswersQuotingTheFailure:
    @pytest.mark.parametrize("name", sorted(QUOTING_ANSWERS))
    def test_is_the_ordinary_no_json_parse_error(self, name):
        result = _parse(_success_shape(QUOTING_ANSWERS[name]))

        assert result["success"] is False
        assert result["error"].startswith("No JSON found in aider output"), result["error"]
        assert "prompt exceeds the model's context" not in result["error"]

    @pytest.mark.parametrize("name", sorted(QUOTING_ANSWERS))
    def test_salvage_is_the_prose_whole(self, name):
        text = _salvage(_success_shape(QUOTING_ANSWERS[name]))

        assert text == QUOTING_ANSWERS[name]
        assert text


#: Answers with lines that start like aider's banner and accounting lines.
PREFIX_ANSWERS = {
    "first-line": (
        "Model: glm-5.2 is the one to keep.\n"
        "Git repo: .git with 3 files is what it saw.\n"
        "Nothing else changes."
    ),
    "mid-answer": (
        "The banner reads, in order:\n"
        "Model: x with whole edit format\n"
        "Git repo: .git with 3 files\n"
        "Tokens: 5 sent, 3 received.\n"
        "and then the reply."
    ),
    "last-line": (
        "Each request ends with one accounting line, for example:\n"
        "Tokens: 5 sent, 3 received."
    ),
    "fenced-transcript": (
        "A quoted aider transcript:\n"
        "```\n"
        "Aider v0.86.2\n"
        "Model: x with whole edit format\n"
        "Git repo: .git with 3 files\n"
        "Repo-map: using 1024 tokens, auto refresh\n"
        "\n"
        "Tokens: 5 sent, 3 received.\n"
        "```\n"
        "That is the whole shape."
    ),
}


class TestBannerAndAccountingPrefixesInTheAnswer:
    @pytest.mark.parametrize("name", sorted(PREFIX_ANSWERS))
    def test_every_line_is_kept(self, name):
        answer = PREFIX_ANSWERS[name]
        text = _salvage(_success_shape(answer))

        assert text == answer
        assert "Tokens: 1 sent, 1 received." not in text
        assert "HTTPSConnectionPool" not in text

    @pytest.mark.parametrize("name", sorted(PREFIX_ANSWERS))
    def test_never_the_d41_failure(self, name):
        result = _parse(_success_shape(PREFIX_ANSWERS[name]))

        assert result["error"].startswith("No JSON found in aider output")

    @pytest.mark.parametrize("name", sorted(PREFIX_ANSWERS))
    def test_the_marker_variant_keeps_every_line_too(self, name):
        answer = PREFIX_ANSWERS[name]
        text = _salvage(_marker_shape(answer))

        assert text == answer
        assert "Tokens: 1 sent, 1 received." not in text


# --------------------------------------------------------------------------- #
# The pair rule (D4.1 step 1)
# --------------------------------------------------------------------------- #

class TestDropConfirmationPairs:
    PROMPT = "Add file to the chat? (Y)es/(N)o/(A)ll/(S)kip all/(D)on't ask again [Yes]: "
    PROCEED = "Try to proceed anyway? (Y)es/(N)o [Yes]: "

    def test_a_mixed_sequence(self):
        lines = [
            "Answer line one.",
            "src/core/mod_2.py",
            self.PROMPT,
            "Answer line two.",
            "It's probably safe to try and send the request.",
            self.PROCEED + "Relevant files:",
            "docs/mod_4.py",
            "Answer line three.",
        ]

        assert _drop_confirmation_pairs(lines) == [
            "Answer line one.",
            "Answer line two.",
            "Relevant files:",
            "docs/mod_4.py",
            "Answer line three.",
        ]

    def test_a_group_preference_echo_is_not_answer_text(self):
        """T4.2's e5 capture: after one `s`, later prompts print `…]: skip`."""
        assert _drop_confirmation_pairs(["a.py", self.PROMPT + "skip", "Answer."]) == [
            "Answer."
        ]
        assert _GROUP_PREFERENCE_ECHOES == {"skip", "all"}

    def test_a_tail_after_a_tail_keeps_both(self):
        """Text an earlier prompt shared is never taken for the next subject."""
        assert _drop_confirmation_pairs(
            ["a.py", self.PROMPT + "First.", self.PROCEED + "Second."]
        ) == ["First.", "Second."]

    def test_the_declined_mentions_capture_loses_every_prompt(self):
        stdout = _read("declined-mentions.stdout.txt")
        kept = _drop_confirmation_pairs(stdout.splitlines())

        assert not any("Add file to the chat?" in line for line in kept)
        assert "skip" not in [line.strip() for line in kept]
        assert "uv.lock" not in kept  # the reply scan's last subject
        assert "- uv.lock" in kept  # the answer's line naming it


# --------------------------------------------------------------------------- #
# D4.2: the decline supply
# --------------------------------------------------------------------------- #

DECLINED_ANSWER = (
    "Relevant files:\n- docs/mod_10.md\n- docs/mod_16.py\n- docs/mod_22.py\n"
    "- docs/mod_28.py\n- docs/mod_4.py\n- plans/myplan.md\n- plans/z_done/x/mod_1.py\n"
    "- plans/z_done/x/mod_13.py\n- plans/z_done/x/mod_19.py\n- uv.lock\nThat is all."
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake, empty ``$HOME``: aider reads ``~/.aider.conf.yml`` and ``~/.env``.

    An exported ``AIDER_ENV_FILE`` would name one more dotenv, so it goes too.
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    # `Path.home()` — aider's and the adapter's — reads USERPROFILE on Windows.
    monkeypatch.setenv("USERPROFILE", str(fake))
    monkeypatch.delenv("AIDER_ENV_FILE", raising=False)
    return fake


@pytest.fixture
def repo(tmp_path):
    """A git root (``.git`` is all aider's root search looks for) with a subdir."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    return root


def _transport(cwd, prompt="Analyze this"):
    return AiderProvider().build_command_for_transport(prompt, MODEL, "argv", cwd=str(cwd))


class TestTheArgv:
    @pytest.mark.parametrize("model", [MODEL, f"{MODEL}:high", "openrouter/x/y:free"])
    def test_there_is_no_yes_always(self, model):
        assert "--yes-always" not in AiderProvider().build_command("Analyze this", model)

    def test_the_transport_argv_is_build_commands(self, home, repo):
        argv, _stdin = _transport(repo)
        assert argv == AiderProvider().build_command("Analyze this", MODEL)

    @pytest.mark.parametrize("transport", ["stdin", "file"])
    def test_no_transport_was_added(self, transport, home, repo):
        """A supply-less path reads EOF, and EOF means yes."""
        with pytest.raises(ValueError):
            AiderProvider().build_command_for_transport(
                "Analyze this", MODEL, transport, prompt_file=str(repo / "p.txt"),
                cwd=str(repo),
            )

    def test_agent_profiles_stay_unsupported(self):
        provider = AiderProvider()
        assert tuple(provider.agent_profiles) == (PROFILE_GENERATE,)
        assert provider.supports_file_edits is False
        for profile in (PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
            with pytest.raises(ProfileUnsupported):
                provider.build_agent_command_for_transport("p", MODEL, profile, "argv")

    def test_the_env_var_is_stripped(self):
        assert AIDER_YES_ALWAYS_ENV == "AIDER_YES_ALWAYS"
        assert "AIDER_YES_ALWAYS" in AiderProvider().get_remove_env()


class TestTheSupplyAndItsBound:
    """The failure the supply guards against is silent — aider outruns it, the
    next prompt reads EOF, and EOF is yes — so every pin here fails loudly."""

    def test_the_supply_is_s_times_the_constant(self, home, repo):
        _argv, stdin_text = _transport(repo)

        assert stdin_text == "s\n" * AIDER_DECLINE_LINES, (
            f"stdin_text is {stdin_text[:40]!r}… ({len(stdin_text or '')} bytes); "
            f"expected 's\\n' x {AIDER_DECLINE_LINES}"
        )
        lines = stdin_text.split("\n")[:-1]
        assert len(lines) == AIDER_DECLINE_LINES
        assert "" not in lines, "a blank answer takes aider's default, which is yes"
        assert AIDER_DECLINE_ANSWER == "s"

    def test_the_bound_and_its_inputs(self):
        prompt_scans = 1
        max_reflections = 3  # aider 0.86.2, base_coder.py:101
        requests = 1 + max_reflections
        reply_scans = requests
        proceed_prompts = requests
        startup_prompts = 2  # "create a git repo?" and the OpenRouter login offer
        bound = prompt_scans + reply_scans + proceed_prompts + startup_prompts

        assert AiderProvider.verified_version == "0.86.2", (
            "aider's pin moved: re-run the T4.2 probe and recompute the bound "
            "(max_reflections and the confirm sites reachable in /ask) first"
        )
        assert bound == 11, f"bound is {bound} prompts, the D4.2 reading says 11"
        assert AIDER_DECLINE_LINES == 64, f"AIDER_DECLINE_LINES is {AIDER_DECLINE_LINES}"
        assert AIDER_DECLINE_LINES >= 5 * bound, (
            f"{AIDER_DECLINE_LINES} lines against a bound of {bound} prompts: "
            f"under {5 * bound}, the margin D4.2 sized is gone"
        )

    def test_the_declined_mentions_capture_stays_within_it(self):
        stdout = _read("declined-mentions.stdout.txt")
        prompts = [line for line in stdout.splitlines() if _CONFIRM_PROMPT_RE.match(line)]
        # A prompt answered from its group's preference prints the echo and
        # reads nothing; every other one read a line (the e5 shape).
        read_a_line = [
            line for line in prompts
            if line[_CONFIRM_PROMPT_RE.match(line).end():].strip()
            not in _GROUP_PREFERENCE_ECHOES
        ]

        assert len(prompts) == 10, f"{len(prompts)} prompt lines in the capture"
        assert len(read_a_line) == 2, f"{len(read_a_line)} prompts read a line"
        assert len(read_a_line) <= 11 and len(read_a_line) < AIDER_DECLINE_LINES
        added = [line for line in stdout.splitlines()
                 if line.startswith("Added ") and "to the chat" in line]
        assert added and all(line.endswith("(read-only).") for line in added), added
        # One request: under yes-always aider prints subjects and no `Added`
        # line for a mention-add, so the request count is the real proof.
        assert sum(line.startswith("Tokens:") for line in stdout.splitlines()) == 1

    def test_declined_mentions_parses_to_its_answer(self):
        stdout = _read("declined-mentions.stdout.txt")

        assert _salvage(stdout) == DECLINED_ANSWER
        result = _parse(stdout)
        assert result["success"] is False  # prose: the ordinary no-JSON failure
        assert "Relevant files:" in result["error"]
        assert "Add file to the chat?" not in result["error"]


class TestInheritedYesAlwaysIsRefused:
    """A file that turns yes-always on makes aider answer "y" before it reads
    the supply (measured: uv.lock re-added from each source), and argv cannot
    turn it off. So the slot is refused before spawn, naming the file."""

    @staticmethod
    def _refused(cwd) -> ReadOnlyInvocationRefused:
        with pytest.raises(ReadOnlyInvocationRefused) as caught:
            _transport(cwd)
        return caught.value

    def test_conf_at_the_cwd(self, home, repo):
        conf = repo / "src" / ".aider.conf.yml"
        conf.write_text("yes-always: true\n", encoding="utf-8")

        refused = self._refused(repo / "src")

        assert refused.path == str(conf)
        assert str(conf) in str(refused) and "yes-always" in str(refused)

    def test_conf_at_the_git_root(self, home, repo):
        conf = repo / ".aider.conf.yml"
        conf.write_text("model: x\nyes-always: yes\n", encoding="utf-8")

        assert self._refused(repo / "src").path == str(conf)

    def test_conf_in_home(self, home, repo):
        conf = home / ".aider.conf.yml"
        conf.write_text("yes-always: true\n", encoding="utf-8")

        assert self._refused(repo).path == str(conf)

    def test_dotenv_at_the_repo(self, home, repo):
        dotenv = repo / ".env"
        dotenv.write_text("OTHER=1\nexport AIDER_YES_ALWAYS=true # on\n", encoding="utf-8")

        refused = self._refused(repo)

        assert refused.path == str(dotenv)
        assert "AIDER_YES_ALWAYS=true" in str(refused)

    def test_dotenv_named_by_aider_env_file(self, home, repo, tmp_path, monkeypatch):
        # aider loads the --env-file last with override=True, so it restores
        # the variable get_remove_env stripped.
        custom = tmp_path / "elsewhere" / "keys.env"
        custom.parent.mkdir()
        custom.write_text("AIDER_YES_ALWAYS=1\n", encoding="utf-8")
        monkeypatch.setenv("AIDER_ENV_FILE", str(custom))

        refused = self._refused(repo)

        assert refused.path == str(custom)
        assert "AIDER_YES_ALWAYS=1" in str(refused)

    def test_dotenv_named_by_a_conf_env_file_key(self, home, repo):
        # A relative `env-file` resolves against the cwd, not the conf's dir.
        (home / ".aider.conf.yml").write_text("env-file: keys.env\n", encoding="utf-8")
        custom = repo / "src" / "keys.env"
        custom.write_text("AIDER_YES_ALWAYS=true\n", encoding="utf-8")

        assert self._refused(repo / "src").path == str(custom)

    def test_a_custom_dotenv_without_the_key_does_not_refuse(self, home, repo, monkeypatch):
        (repo / ".aider.conf.yml").write_text("--env-file: keys.env\n", encoding="utf-8")
        (repo / "keys.env").write_text("OPENROUTER_API_KEY=x\n", encoding="utf-8")
        monkeypatch.setenv("AIDER_ENV_FILE", str(repo / "missing.env"))

        assert _transport(repo)[1] == "s\n" * AIDER_DECLINE_LINES

    def test_an_unparseable_conf_is_still_read(self, home, repo):
        (repo / ".aider.conf.yml").write_text("yes-always: true\n: [broken\n", encoding="utf-8")

        assert self._refused(repo).path == str(repo / ".aider.conf.yml")

    @pytest.mark.parametrize("value", ["false", "no", "off", "0", "False"])
    def test_a_falsy_setting_does_not_refuse(self, home, repo, value):
        (repo / ".aider.conf.yml").write_text(f"yes-always: {value}\n", encoding="utf-8")
        (repo / ".env").write_text(f"AIDER_YES_ALWAYS={value}\n", encoding="utf-8")

        _argv, stdin_text = _transport(repo)

        assert stdin_text == "s\n" * AIDER_DECLINE_LINES

    def test_no_config_at_all_does_not_refuse(self, home, repo):
        assert _transport(repo)[1] == "s\n" * AIDER_DECLINE_LINES


class TestThroughInvokeWithProvider:
    """The spawn site: the refusal is a non-retried config error with no child,
    and an allowed run writes the supply to the child's stdin."""

    def test_a_refusal_is_a_config_error_and_nothing_spawns(self, home, repo):
        conf = repo / ".aider.conf.yml"
        conf.write_text("yes-always: true\n", encoding="utf-8")

        def must_not_spawn(*_args, **_kwargs):
            raise AssertionError("aider was spawned despite the refusal")

        with patch.object(llm_client, "_run_provider_cli", side_effect=must_not_spawn), \
                patch("shutil.which", return_value="/usr/bin/aider"):
            result = invoke_with_provider("Q", f"aider:{MODEL}", cwd=str(repo))

        assert result["success"] is False
        assert str(conf) in result["error"]
        assert result["details"]["retryable"] is False
        assert result["details"]["failure_reason"] == "config_error"

    def test_an_allowed_run_feeds_the_supply(self, home, repo, monkeypatch):
        # An exported AIDER_YES_ALWAYS is stripped from the child, not refused.
        monkeypatch.setenv("AIDER_YES_ALWAYS", "1")
        seen = {}

        def spawn(cmd, **kwargs):
            seen.update(kwargs, cmd=cmd)
            return subprocess.CompletedProcess(cmd, 0, SUCCESS, "")

        with patch.object(llm_client, "_run_provider_cli", side_effect=spawn), \
                patch("shutil.which", return_value="/usr/bin/aider"):
            result = invoke_with_provider("Q", f"aider:{MODEL}", cwd=str(repo))

        assert result["success"] is True
        assert seen["input"] == "s\n" * AIDER_DECLINE_LINES
        assert "stdin" not in seen
        assert "--yes-always" not in seen["cmd"]
        assert "AIDER_YES_ALWAYS" not in seen["env"]


def test_the_container_live_probe_spawns_aider_with_the_supply():
    """`tests/container/live_provider.py` spawns the real binary itself, so it
    must take the argv transport's pair, not `build_command` alone (T4.2)."""
    tree = ast.parse((SKILL_ROOT / "tests" / "container" / "live_provider.py").read_text(
        encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def attr(node):
        return node.func.attr if isinstance(node.func, ast.Attribute) else None

    transport = [call for call in calls if attr(call) == "build_command_for_transport"]
    assert transport, "live_provider.py no longer calls build_command_for_transport"
    assert any(
        isinstance(arg, ast.Constant) and arg.value == "argv"
        for call in transport for arg in call.args
    )
    assert not [call for call in calls if attr(call) == "build_command"], (
        "live_provider.py spawns from build_command: aider would read EOF, and EOF is yes"
    )
    spawns = [call for call in calls if attr(call) == "run"
              and isinstance(call.func.value, ast.Name) and call.func.value.id == "subprocess"]
    assert any(
        keyword.arg == "input" and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "stdin_text"
        for call in spawns for keyword in call.keywords
    ), "live_provider.py's spawn does not pass input=stdin_text"
