"""Tests for the KiloCode adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_extractor import _ANSI_ESCAPE
from utils.providers.base import LLMProvider, split_reasoning_effort
from utils.providers.kilocode import KiloCodeProvider


class TestKiloCodeProvider:
    """Tests for the KiloCodeProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a KiloCodeProvider instance."""
        return KiloCodeProvider()

    def test_name_property(self, provider):
        """Test that name property returns correct identifier."""
        assert provider.name == "kilocode"

    def test_default_timeout(self, provider):
        """No per-adapter timeout: docs/configuration.md#provider-blocks."""
        from utils.providers.base import DEFAULT_TIMEOUT_S

        # The shared fallback; the timeout a run uses is providers.yaml's row.
        assert provider.default_timeout == DEFAULT_TIMEOUT_S

    # -- Dual binary name (`kilocode` / `kilo`) -----------------------------
    #
    # Upstream ships the CLI under both spellings. This adapter knew only
    # `kilocode`, so on a `kilo`-only machine `is_available()` was False and the
    # provider was invisible to the fan-out — not degraded, absent. The three
    # cases below are the whole contract: canonical present (today's argv,
    # byte-identical), only the alias present (argv[0] moves, nothing else
    # does), and neither present (not available).

    @staticmethod
    def _which_finding(*names):
        """A `shutil.which` stub that resolves exactly `names`."""
        found = set(names)
        return lambda name: f"/usr/local/bin/{name}" if name in found else None

    @patch("shutil.which")
    def test_is_available_true(self, mock_which, provider):
        """Test is_available returns True when kilocode is found."""
        mock_which.return_value = "/usr/local/bin/kilocode"

        assert provider.is_available() is True
        mock_which.assert_called_once_with("kilocode")

    @patch("shutil.which")
    def test_is_available_true_for_the_kilo_spelling(self, mock_which, provider):
        """A `kilo`-only install is available — the defect this pins."""
        mock_which.side_effect = self._which_finding("kilo")

        assert provider.is_available() is True
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_is_available_false(self, mock_which, provider):
        """is_available is False only when NEITHER spelling resolves."""
        mock_which.return_value = None

        assert provider.is_available() is False
        assert [c.args[0] for c in mock_which.call_args_list] == ["kilocode", "kilo"]

    @patch("shutil.which")
    def test_build_command_uses_the_kilocode_spelling_when_present(self, mock_which, provider):
        """Canonical install: argv is byte-identical to what shipped before."""
        mock_which.side_effect = self._which_finding("kilocode")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd == [
            "kilocode", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_prefers_kilocode_when_both_resolve(self, mock_which, provider):
        """Both installed ⇒ `kilocode` wins, so argv does not move under the user."""
        mock_which.side_effect = self._which_finding("kilocode", "kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    @patch("shutil.which")
    def test_build_command_moves_only_argv0_for_the_kilo_spelling(self, mock_which, provider):
        """`kilo`-only: argv[0] moves and every other element is identical.

        Asserted against the canonical argv element-by-element rather than as a
        literal, so this cannot drift from the row above.
        """
        mock_which.side_effect = self._which_finding("kilocode")
        canonical = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        mock_which.side_effect = self._which_finding("kilo")
        aliased = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert aliased[0] == "kilo"
        assert aliased[1:] == canonical[1:]

    @patch("shutil.which")
    def test_build_command_variant_branch_also_resolves_the_binary(self, mock_which, provider):
        """The `--variant` branch is the second hardcoded literal §2.3 names."""
        mock_which.side_effect = self._which_finding("kilo")

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilo", "run", "--auto", "-m", "openrouter/z-ai/glm-5.2",
            "--variant", "xhigh", "Review this code",
        ]

    @patch("shutil.which")
    def test_build_command_falls_back_to_the_canonical_name_with_neither_installed(
        self, mock_which, provider
    ):
        """Neither installed ⇒ the canonical spelling, so the argv contract holds.

        `is_available()` is False here, so this argv never reaches a subprocess;
        it is what the argv-contract tests, the housekeeping MODELS table and
        `seed_expected_argv.py` render on a binary-free host.
        """
        mock_which.return_value = None

        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2")

        assert cmd[0] == "kilocode"

    def test_build_command(self, provider):
        """Test command building for Kilo Code CLI."""
        prompt = "Review this code"
        model = "openrouter/moonshotai/kimi-k2.5"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "openrouter/moonshotai/kimi-k2.5", "Review this code"]

    def test_build_command_different_model(self, provider):
        """Test command building with a different model."""
        prompt = "Analyze this"
        model = "kilo/z-ai/glm-5:free"

        cmd = provider.build_command(prompt, model)

        assert cmd == ["kilocode", "run", "--auto", "-m", "kilo/z-ai/glm-5:free", "Analyze this"]

    def test_reasoning_efforts_constant(self):
        """Module-level REASONING_EFFORTS holds exactly the nine valid efforts."""
        from utils.providers.kilocode import REASONING_EFFORTS

        assert isinstance(REASONING_EFFORTS, frozenset)
        assert REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant",
        }

    def test_build_command_effort_suffix_exact(self, provider):
        """Exact argv for an :effort suffix: --variant pair after -m <base>, prompt last."""
        cmd = provider.build_command("Review this code", "openrouter/z-ai/glm-5.2:xhigh")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/z-ai/glm-5.2",
            "--variant",
            "xhigh",
            "Review this code",
        ]

    def test_build_command_thinking_variant_suffix(self, provider):
        """The kilocode-specific :thinking suffix maps to --variant thinking."""
        cmd = provider.build_command("Review this code", "openrouter/moonshotai/kimi-k2.7-code:thinking")

        assert cmd == [
            "kilocode",
            "run",
            "--auto",
            "-m",
            "openrouter/moonshotai/kimi-k2.7-code",
            "--variant",
            "thinking",
            "Review this code",
        ]

    def test_build_command_all_reasoning_efforts(self, provider):
        """Every valid effort suffix produces the --variant form."""
        efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "thinking", "instant")
        for effort in efforts:
            cmd = provider.build_command("Review this code", f"openrouter/z-ai/glm-5.2:{effort}")

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                "openrouter/z-ai/glm-5.2",
                "--variant",
                effort,
                "Review this code",
            ], f"effort {effort!r} did not build the expected command"

    def test_build_command_effort_passthrough(self, provider):
        """Non-effort models pass through verbatim with no --variant flag.

        Critically, openrouter ":free" variants are NOT effort suffixes:
        the whole string stays the model.
        """
        for model in ("openrouter/z-ai/glm-5.2", "openrouter/deepseek/deepseek-r1:free"):
            cmd = provider.build_command("Review this code", model)

            assert cmd == [
                "kilocode",
                "run",
                "--auto",
                "-m",
                model,
                "Review this code",
            ], f"model {model!r} was not passed through verbatim"

    def test_parse_output_direct_json_array(self, provider):
        """Parse direct JSON array output."""
        data = [{"item": 1}, {"item": 2}]
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_direct_json_object(self, provider):
        """Parse direct JSON object output."""
        data = {"status": "ok", "items": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data

    def test_parse_output_code_block(self, provider):
        """Extract JSON from markdown code block."""
        inner_json = [{"title": "Suggestion 1", "importance": "high"}]
        stdout = f"""Here is my analysis:

```json
{json.dumps(inner_json, indent=2)}
```

This concludes the review."""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_code_block_without_json_label(self, provider):
        """Extract JSON from code block without json language label."""
        inner_json = {"status": "ok", "items": [1, 2, 3]}
        stdout = f"""Output:
```
{json.dumps(inner_json)}
```
"""

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_embedded_json(self, provider):
        """Extract embedded JSON from plain text."""
        inner_json = [{"id": 1}, {"id": 2}]
        stdout = f"The results are: {json.dumps(inner_json)} end of results."

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == inner_json

    def test_parse_output_empty(self, provider):
        """Handle empty output."""
        result = provider.parse_output("", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_whitespace_only(self, provider):
        """Handle whitespace-only output."""
        result = provider.parse_output("   \n\t  ", "")

        assert result["success"] is False
        assert "Empty output" in result["error"]

    def test_parse_output_no_json(self, provider):
        """Handle output with no extractable JSON."""
        stdout = "Just plain text with no JSON at all."

        result = provider.parse_output(stdout, "")

        assert result["success"] is False
        assert "error" in result

    def test_parse_output_whitespace_handling(self, provider):
        """Test that whitespace is handled correctly."""
        data = [1, 2, 3]
        stdout = f"  \n  {json.dumps(data)}  \n  "

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == data


class TestKiloCodeReadsOnlyItsPaintedErrorLines:
    """kilocode's ``vendor_text`` gives the classifier and its quote only the
    stderr lines kilocode paints as its own ``Error:``, colour off: never
    stdout (the model's answer), the banner or the tool trace. The captured
    rows themselves are graded in test_provider_fixture_rows.py.
    """

    ROWS = Path(__file__).parent / "fixtures" / "kilocode" / "rows"
    BANNER = "\x1b[0m\n> code · stub-model\n\x1b[0m\n"
    PAINT = BANNER + "\x1b[91m\x1b[1mError: \x1b[0m{cause}\n"
    #: The tool trace line exactly as ``multi-assistant`` captured it, with
    #: what the model chose to read or run after the arrow.
    TRACE = "\x1b[0m→ \x1b[0m{tool}\n"

    #: The vendor sentences the tables read as a verdict, as a model's answer
    #: or a tool's argument may quote them.
    PHRASES = {
        "401": "Incorrect API key provided",
        "429": "Rate limit reached for gpt-4o in organization org-x on requests per min",
        "500": "The server had an error while processing your request.",
        "500-gloss": "currently experiencing high demand, which may cause temporary errors",
    }
    NEUTRAL = "The plan needs a focused regression test."

    @pytest.fixture
    def provider(self):
        return KiloCodeProvider()

    @staticmethod
    def _lanes(stdout, stderr):
        from tests.test_provider_fixture_rows import _lane_a_status, _lane_b_status
        from utils.provider_outcome import Capture

        capture = Capture(stdout=stdout, stderr=stderr, exit_code=1)
        return (
            _lane_a_status("kilocode", capture),
            _lane_b_status("kilocode", capture),
        )

    @staticmethod
    def _quote(provider, stdout, stderr):
        from utils.attempt_classifier import _classify_failed_run
        from utils.provider_outcome import Capture

        return _classify_failed_run(
            provider, Capture(stdout=stdout, stderr=stderr, exit_code=1), "m"
        )

    def test_the_hook_keeps_only_the_painted_lines_of_every_capture(self, provider):
        """Every committed pair: no stdout, and of stderr exactly the lines that
        open on the paint, each a piece of the capture with its escapes off."""
        pairs = sorted(self.ROWS.glob("*.stderr.txt"))
        assert pairs
        for stderr_path in pairs:
            stdout_path = stderr_path.with_name(
                stderr_path.name.replace(".stderr.", ".stdout.")
            )
            stdout = stdout_path.read_bytes().decode("utf-8")
            stderr = stderr_path.read_bytes().decode("utf-8")
            reduced_out, reduced_err = provider.vendor_text(stdout, stderr)
            assert reduced_out == "", stderr_path.name
            painted = [
                line for line in stderr.split("\n")
                if line.startswith("\x1b[91m\x1b[1mError: \x1b[0m")
            ]
            kept = reduced_err.split("\n") if reduced_err else []
            assert kept == [_ANSI_ESCAPE.sub("", line) for line in painted]
            assert all(line.startswith("Error: ") for line in kept), stderr_path.name
            assert "\x1b" not in reduced_err and "> code" not in reduced_err
        multi = (self.ROWS / "multi-assistant.stderr.txt").read_text(encoding="utf-8")
        assert "Read plans/auth-feature.md" in multi
        assert provider.vendor_text("", multi) == ("", "")

    def test_a_failed_reduction_returns_its_input(self, provider):
        assert provider.vendor_text("answer", b"\x1b[0m") == ("answer", b"\x1b[0m")

    @pytest.mark.parametrize("cause,expected", [
        ("Rate limit reached for stub-model in organization org-stub.", "rate_limited"),
        ("Incorrect API key provided: sk-stub***.", "auth_failed"),
        ("The server had an error while processing your request. Sorry about that!",
         "server_error"),
        ("The model `stub-model` does not exist or you do not have access to it.",
         "invalid_model"),
        ("This model's maximum context length is 8192 tokens.", "context_length_exceeded"),
        ("Connection reset by server", "connection_error"),
        # Deliberately no verdict: see `KiloCodeProvider.describe_failure`.
        ("User not found.", None),
        ("run ended without an assistant message; the model returned no output", None),
    ])
    def test_the_painted_cause_is_classified(self, provider, cause, expected):
        assert provider.classify_failure("", self.PAINT.format(cause=cause), 1) == expected

    def test_the_paint_is_what_hid_the_rate_limit(self, provider):
        """The same bytes through the base hook (identity): no boundary, no verdict."""
        stderr = self.PAINT.format(cause="Rate limit reached for stub-model.")
        with patch.object(KiloCodeProvider, "vendor_text", LLMProvider.vendor_text):
            assert provider.classify_failure("", stderr, 1) is None
        assert provider.classify_failure("", stderr, 1) == "rate_limited"

    def test_the_auth_wording_is_the_whole_clause(self, provider):
        assert "incorrect api key provided" in provider.auth_error_patterns
        for line in ("Find your API key at the dashboard.", "an incorrect API key"):
            assert provider.classify_failure("", self.PAINT.format(cause=line), 1) is None

    @pytest.mark.parametrize("stderr", ["", "banner", "trace"])
    @pytest.mark.parametrize("phrase", sorted(PHRASES))
    def test_a_failed_runs_answer_quoting_a_vendor_sentence_decides_nothing(
        self, provider, phrase, stderr
    ):
        """A non-zero exit after a partial answer that quotes the 401, 429 or
        500 sentence: no verdict, and both lanes as for a neutral answer."""
        stderr = {
            "": "",
            "banner": self.BANNER,
            "trace": self.BANNER + self.TRACE.format(tool="Read plans/auth-feature.md"),
        }[stderr]
        answer = (
            "Finding: preserve the vendor message \"" + self.PHRASES[phrase]
            + "\" in the error log.\n"
        )
        assert provider.classify_failure(answer, stderr, 1) is None
        assert self._lanes(answer, stderr) == self._lanes(self.NEUTRAL, stderr)

    @pytest.mark.parametrize("phrase", sorted(PHRASES))
    @pytest.mark.parametrize("tool", [
        "Read docs/{phrase}.md",
        "Grep \"{phrase}\" docs/",
        "Bash echo '{phrase}'",
    ])
    def test_a_tool_trace_naming_a_vendor_sentence_decides_nothing(
        self, provider, phrase, tool
    ):
        """The trace names what the model chose to read or run, so its words are
        the model's: no verdict, and both lanes as for a neutral trace."""
        stderr = self.BANNER + self.TRACE.format(
            tool=tool.format(phrase=self.PHRASES[phrase])
        )
        neutral = self.BANNER + self.TRACE.format(tool="Read plans/auth-feature.md")
        assert provider.classify_failure("", stderr, 1) is None
        assert self._lanes("", stderr) == self._lanes("", neutral)

    @pytest.mark.parametrize("phrase", sorted(PHRASES))
    def test_an_unpainted_line_is_no_diagnosis(self, provider, phrase):
        """Only the captured framing is kilocode's: the same sentence on a line
        without it (a tool's output, a wrapped continuation) gives no verdict."""
        for line in ("Error: " + self.PHRASES[phrase], self.PHRASES[phrase]):
            assert provider.classify_failure("", self.BANNER + line + "\n", 1) is None

    @pytest.mark.parametrize("row,expected", [
        ("auth-error", "auth_failed"),
        ("rate-limit", "rate_limited"),
        ("server-error", "server_error"),
    ])
    def test_the_captured_401_429_and_500_keep_their_verdicts(self, provider, row, expected):
        stdout = (self.ROWS / f"{row}.stdout.txt").read_text(encoding="utf-8")
        stderr = (self.ROWS / f"{row}.stderr.txt").read_text(encoding="utf-8")
        assert provider.classify_failure(stdout, stderr, 1) == expected

    @pytest.mark.parametrize("answer_phrase,cause,expected", [
        ("401", "Connection reset by server", "connection_error"),
        ("401", "The server had an error while processing your request.", "server_error"),
        ("429", "Incorrect API key provided: sk-stub***.", "auth_failed"),
        ("500", "Rate limit reached for stub-model.", "rate_limited"),
    ])
    def test_an_answer_never_outranks_the_painted_cause(
        self, provider, answer_phrase, cause, expected
    ):
        """The table order would let a quoted 401 beat a real dropped connection;
        with the answer out of the haystack the painted cause decides."""
        answer = "The log should keep: " + self.PHRASES[answer_phrase]
        assert provider.classify_failure(answer, self.PAINT.format(cause=cause), 1) == expected

    @pytest.mark.parametrize("row,answer_phrase", [
        ("auth-error", "429"),
        ("rate-limit", "401"),
    ])
    def test_the_verdict_quotes_the_painted_line_never_the_answer(
        self, provider, row, answer_phrase
    ):
        """The executor's failure reason is cut from the same reduction: the
        captured error line, colour off, and none of the answer."""
        stderr = (self.ROWS / f"{row}.stderr.txt").read_text(encoding="utf-8")
        answer = "ANSWER-TEXT: the plan quotes " + self.PHRASES[answer_phrase] + ".\n"
        found = self._quote(provider, answer, stderr)
        assert found is not None
        status, reason = found
        assert status == {"auth-error": "auth_failed", "rate-limit": "rate_limited"}[row]
        assert "ANSWER-TEXT" not in reason and "\\x1b" not in reason
        sentence = {"auth-error": "Incorrect API key provided", "rate-limit": "Rate limit reached"}
        assert sentence[row] in reason


class TestKiloCodeRetryEnvelope:
    """kilocode retries a 429 itself (6 requests over 70.48 s against a
    fail-forever stub, tests/fixtures/kilocode/manifest.yaml ``rate-limit``),
    so the ladder must not re-dispatch the same chain."""

    def test_the_declaration_is_the_measured_one(self):
        envelope = KiloCodeProvider.internal_retries
        assert envelope is not None
        assert envelope.retries_rate_limit is True
        # Unproven bounds stay unstated: a chain per model turn, a stub's wall.
        assert envelope.max_requests is None and envelope.max_wall_s is None
        assert envelope.configurable_by == ()

    def test_the_capture_behind_it_is_a_chain(self):
        import yaml

        manifest = yaml.safe_load(
            (Path(__file__).parent / "fixtures" / "kilocode" / "manifest.yaml")
            .read_text(encoding="utf-8")
        )
        row = manifest["rows"]["rate-limit"]
        assert row["requests"] == 6 and row["wall_s"] == pytest.approx(70.48)
        assert row["expect"]["classify_failure"] == "rate_limited"

    @staticmethod
    def _dispatches(envelope, *, timeout=600.0, chain_s=70.5):
        """How many times one slot dispatches a run that ends rate-limited
        after a ``chain_s`` chain: the real ladder, a fake clock, no jitter."""
        from utils.retry_ladder import STATUS_RATE_LIMITED, RetryLadder

        clock = {"t": 0.0}
        ladder = RetryLadder(
            timeout=timeout, retries=1, envelope=envelope,
            now=lambda: clock["t"], jitter=lambda: 0.0,
        )
        dispatched = 0
        while ladder.next_attempt() is not None:
            dispatched += 1
            clock["t"] += chain_s
            action = ladder.record(STATUS_RATE_LIMITED)
            clock["t"] += action.delay
            if not action.retry:
                break
        return dispatched, ladder

    def test_a_rate_limited_slot_runs_one_chain(self):
        from utils.retry_ladder import STOP_VENDOR_ALREADY_RETRIED

        dispatched, ladder = self._dispatches(KiloCodeProvider().internal_retries)
        assert dispatched == 1
        assert ladder.stop_reason == STOP_VENDOR_ALREADY_RETRIED

    def test_without_the_declaration_the_chain_repeats(self):
        """The baseline the declaration removes: the ladder's own backoffs, each
        one re-running kilocode's whole chain inside the same 600 s slot."""
        dispatched, _ = self._dispatches(None)
        assert dispatched == 4


class TestKiloCodeDefaultModelSpecs:
    """Every kilocode `:variant` in `defaults.models` must be one the model publishes.

    **This is the only place the mistake can be caught.** `kilocode run
    --variant <name>` SILENTLY IGNORES an unsupported name — no warning, no
    non-zero exit, no marker in the stream — so a fan-out row asking for a
    variant the model does not have runs at the model's default reasoning level
    and reports success. Nothing at run time distinguishes it from a row that
    got what it asked for.

    Four of the six kilocode rows of `defaults.models` were in exactly that
    state (§13.4 #2): `kimi-k3:thinking`, `nemotron-3-ultra-550b-a55b:xhigh`,
    `minimax-m3:xhigh`, `mimo-v2.5-pro:xhigh`.

    The published set is a **captured** fixture, not a live probe: `kilocode
    models <backend> --verbose` is credential-free but it is still a network
    call, and a test that reaches the internet is a test that fails for reasons
    that are not about this repository. `tests/fixtures/kilocode/
    model_variants.json` carries the capture plus its own provenance, and this
    test refuses to pass over a model the capture does not cover — a stale
    fixture must be a failure, not a silent skip, or the guard evaporates the
    first time somebody adds a row.
    """

    VARIANTS_FIXTURE = (Path(__file__).parent / "fixtures" / "kilocode"
                        / "model_variants.json")

    @staticmethod
    def _kilocode_specs():
        """The `kilocode:<model>` rows of the BASE `defaults.models`.

        `load_base_config` and not `load_config`: a developer's own
        `<git-root>/.multi-llm/providers.yaml` deep-merges over base and would
        otherwise decide whether this test passes.
        """
        from utils.provider_registry import load_base_config

        models = (load_base_config().get("defaults") or {}).get("models") or []
        return [spec[len("kilocode:"):] for spec in models
                if isinstance(spec, str) and spec.startswith("kilocode:")]

    def _published(self):
        return json.loads(
            self.VARIANTS_FIXTURE.read_text(encoding="utf-8"))["variants"]

    def test_fixture_covers_every_kilocode_default(self):
        """A spec the capture does not know about fails loudly, naming the refresh."""
        published = self._published()
        specs = self._kilocode_specs()

        assert specs, "defaults.models lists no kilocode rows — did the key move?"
        for spec in specs:
            from utils.providers.kilocode import REASONING_EFFORTS

            base_model, _variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            assert base_model in published, (
                f"no captured `variants` for kilocode model {base_model!r}. "
                f"Refresh {self.VARIANTS_FIXTURE.name} with the commands in its "
                "`_provenance` block (`kilocode models <backend> --verbose`, "
                "credential-free) — an uncovered model is an unchecked variant, "
                "which the CLI would ignore in silence."
            )

    def test_every_default_variant_is_published_by_its_model(self):
        """The L11 #2 assertion: red on four of six before the catalog was corrected."""
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        for spec in self._kilocode_specs():
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is not None and variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not offenders, (
            "defaults.models asks kilocode for a variant the model does not "
            "publish. `kilocode run --variant` ignores an unsupported name "
            "SILENTLY, so these rows run at the model's default effort and "
            "still report success:\n" + "\n".join(offenders)
        )

    @staticmethod
    def _catalog_specs():
        """The `providers.kilocode.models` rows — the *selectable* catalog.

        Separate from `defaults.models` because they fail differently, not
        because the rule differs: a defaults row is silently mis-run on every
        fan-out, while a catalog row is silently mis-run only once a user or
        `--models` picks it. Same silent-ignore mechanism, same guard.
        """
        from utils.provider_registry import load_base_config

        providers = load_base_config().get("providers") or {}
        return list((providers.get("kilocode") or {}).get("models") or [])

    def test_every_catalog_variant_is_published_by_its_model(self):
        """Five catalog rows carried the same defect the defaults did.

        Found while L11 was correcting `defaults.models` and fixed with the
        same rule (the top variant the model actually publishes, preserving the
        row's maximum-reasoning intent): `kimi-k3:thinking` → `:max`,
        `minimax-m3:xhigh` → `:thinking`, `qwen3.7-max:xhigh` → `:max`,
        `hy3:xhigh` → `:high`, `step-3.7-flash:xhigh` → `:high`.
        """
        published = self._published()
        from utils.providers.kilocode import REASONING_EFFORTS

        offenders = []
        uncovered = []
        for spec in self._catalog_specs():
            if not isinstance(spec, str):
                continue
            base_model, variant = split_reasoning_effort(spec, REASONING_EFFORTS)
            if variant is None:
                continue
            allowed = published.get(base_model)
            if allowed is None:
                uncovered.append(f"  {base_model}")
                continue
            if variant not in allowed:
                offenders.append(
                    f"  kilocode:{spec} — {base_model} publishes {sorted(allowed)}")

        assert not uncovered, (
            "providers.kilocode.models carries a `:variant` on a model the "
            f"capture does not cover. Refresh {self.VARIANTS_FIXTURE.name} "
            "with the commands in its `_provenance` block:\n"
            + "\n".join(uncovered)
        )
        assert not offenders, (
            "providers.kilocode.models asks for a variant the model does not "
            "publish; `kilocode run --variant` ignores it SILENTLY:\n"
            + "\n".join(offenders)
        )
