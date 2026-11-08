"""P12, the PEM private-key row of ``write_safety._REDACTIONS``, redacts exactly
what its regex redacted, in linear time.

The regex ``-----BEGIN [A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END [A-Z ]*PRIVATE
KEY-----`` is quadratic in headers with no footer after them, and
``redact_text`` runs on a provider's whole stream before any cap. The row is now
``_PemPrivateKeyBlocks``, a scan that yields the regex's own matches. Secret
scrubbing must never shrink, so the claim checked here is equality, not
"close": every input's redacted output, and every match the row yields, equals
what the regex gave, over the redaction suite's own inputs, its benign corpora,
its differential corpus, and PEM documents built to stress the block rules
(multi-line, CRLF, nested and stray headers, truncated blocks, near-miss
headers, other secrets inside and around a block). A megabyte of unterminated
headers is then redacted in a child process under a timeout.
"""

import json
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_redaction import (  # noqa: E402
    _GLUED_CASES,
    _PEM,
    _PEM_BODY,
    BENIGN_CORPUS,
    LINEARITY_PAYLOADS,
    PROSE_THE_LEFT_ANCHOR_PROTECTS,
    SECRET_KV_KEY_SHAPES,
    SHAPES,
    _differential_corpus,
)
from utils import write_safety  # noqa: E402
from utils.write_safety import redact_text  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent

#: The row as it was.
_OLD_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)


def _old_rows():
    rows = list(write_safety._REDACTIONS)
    assert rows[0][0] == "pem-private-key", "P12 runs first; see the _REDACTIONS comment"
    rows[0] = ("pem-private-key", _OLD_PEM, 0, None)
    return rows


_OLD_ROWS = _old_rows()
_NEW_PEM = write_safety._REDACTIONS[0][1]


def _old_redact_text(text):
    """``redact_text``'s loop, verbatim, over the table with the old P12."""
    if not text:
        return text or ""
    out = text
    for name, pattern, group, predicate in _OLD_ROWS:
        def _sub(match, _name=name, _group=group, _predicate=predicate):
            if _predicate is not None and not _predicate(match.group(_group)):
                return match.group(0)
            return write_safety._replace_group(match, _group, _name)

        out = pattern.sub(_sub, out)
    return out


def _matches(pattern, text):
    return [(m.span(), m.group(0)) for m in pattern.finditer(text)]


def _assert_same(text):
    assert _matches(_NEW_PEM, text) == _matches(_OLD_PEM, text), f"text={text[:300]!r}"
    assert redact_text(text) == _old_redact_text(text), f"text={text[:300]!r}"


# --- Generated PEM documents. -----------------------------------------------------

_KINDS = ["RSA ", "EC ", "DSA ", "OPENSSH ", "ENCRYPTED ", "", "RSA  ", "A B C "]
_NEAR_MISSES = [
    "-----BEGIN PUBLIC KEY-----", "-----BEGIN CERTIFICATE-----",
    "-----BEGIN rsa PRIVATE KEY-----", "-----BEGIN RSA PRIVATE  KEY-----",
    "----BEGIN RSA PRIVATE KEY-----", "-----BEGIN RSA PRIVATE KEY----",
    "-----BEGIN RSA-PRIVATE KEY-----", "-----BEGINRSA PRIVATE KEY-----",
    "-----END PUBLIC KEY-----", "-----END RSA PRIVATE KEY----",
    "------BEGIN RSA PRIVATE KEY------", "-----BEGIN RSA PRIVATE KEY-----BEGIN",
]
_FILLER = [
    "provider said:", "Traceback (most recent call last):", "  File \"x.py\"",
    "token=Xy9-_.z01234567", "sk-proj-" + "a1B2" * 8, "", " ", "\t", "-----", "---",
    "BEGIN", "END", "PRIVATE KEY", "KEY-----", "```", "é漢",
]


def _body_lines(rng):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 64)))
            for _ in range(rng.randint(0, 6))]


def _header(rng, word):
    kind = rng.choice(_KINDS)
    return f"-----{word} {kind}PRIVATE KEY-----"


def _block(rng):
    """One block, often well-formed, sometimes nested, stray or cut short."""
    roll = rng.random()
    lines = [_header(rng, "BEGIN"), *_body_lines(rng)]
    if roll < 0.15:  # nested: a second header inside the body
        lines += [_header(rng, "BEGIN"), *_body_lines(rng)]
    if roll < 0.75 or rng.random() < 0.3:
        lines.append(_header(rng, "END"))
    if 0.75 <= roll < 0.85:  # a stray footer after the block
        lines += _body_lines(rng) + [_header(rng, "END")]
    if rng.random() < 0.1:  # glued header and footer, no newline at all
        return "".join(lines)
    return "\n".join(lines)


def _pem_document(rng):
    parts = []
    for _ in range(rng.randint(1, 6)):
        roll = rng.random()
        if roll < 0.5:
            parts.append(_block(rng))
        elif roll < 0.65:
            parts.append(rng.choice(_NEAR_MISSES))
        else:
            parts.append(rng.choice(_FILLER))
    newline = rng.choice(["\n", "\n", "\r\n", "", " "])
    text = newline.join(parts)
    if newline == "\n" and rng.random() < 0.3:
        text = text.replace("\n", "\r\n")
    if rng.random() < 0.25:  # truncated anywhere: a cap applied by someone else
        text = text[:rng.randint(0, len(text))]
    return text


def _pem_fragment_soup(rng):
    fragments = [
        "-----BEGIN ", "-----END ", "RSA ", "PRIVATE KEY", "-----", "-", "\n", "\r\n",
        " ", "A", "BEGIN", "END", "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----", "-----BEGIN PRIVATE KEY-----",
        "-----END PRIVATE KEY-----", "PRIVATE KEY-----", "x", "token=abcdefgh12",
    ]
    return "".join(rng.choice(fragments) for _ in range(rng.randint(0, 30)))


class TestThePemRowRedactsWhatItsRegexDid:

    def test_the_seventeen_shapes_and_the_planted_block(self):
        for _shape, line, _planted in SHAPES:
            _assert_same(line)
        redacted = redact_text(_PEM)
        assert redacted == "<redacted:pem-private-key>"
        assert not any(line in redacted for line in _PEM_BODY)

    def test_the_suite_s_other_inputs(self):
        texts = [line for _case, line, _key in _GLUED_CASES]
        texts += [line for _shape, line in SECRET_KV_KEY_SHAPES]
        texts += [line for _shape, line in PROSE_THE_LEFT_ANCHOR_PROTECTS]
        texts += [unit * 50 for _label, unit in LINEARITY_PAYLOADS]
        for text in texts:
            _assert_same(text)

    def test_the_benign_corpora(self):
        for _label, path in BENIGN_CORPUS:
            text = path.read_text(encoding="utf-8", errors="replace")
            _assert_same(text)
            # Each corpus file with a planted block in the middle, CRLF too.
            middle = len(text) // 2
            _assert_same(text[:middle] + "\n" + _PEM + "\n" + text[middle:])
            _assert_same((text[:middle] + "\n" + _PEM).replace("\n", "\r\n"))

    def test_the_differential_corpus(self):
        """The secret-kv rewrite's corpus: the row's matches only (no line of
        it holds a header, so the full pass would only repeat that suite)."""
        for line in _differential_corpus():
            assert _matches(_NEW_PEM, line) == _matches(_OLD_PEM, line)

    def test_generated_pem_documents(self):
        rng = random.Random("U69 PEM documents")
        for _ in range(6000):
            _assert_same(_pem_document(rng))

    def test_pem_fragment_soup(self):
        rng = random.Random("U69 PEM soup")
        for _ in range(20000):
            _assert_same(_pem_fragment_soup(rng))

    @pytest.mark.parametrize("text", [
        "",
        _PEM,
        _PEM.replace("\n", "\r\n"),
        _PEM + "\n" + _PEM,
        _PEM[:-10],
        "-----BEGIN RSA PRIVATE KEY-----\n" * 20 + "-----END RSA PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\n" * 20,
        "-----END RSA PRIVATE KEY-----\n" + _PEM,
        "-----BEGIN PRIVATE KEY----------END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY--------END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----BEGIN PRIVATE KEY-----x-----END PRIVATE KEY-----",
        "-----BEGIN " + "A" * 500 + " PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----",
        "-----BEGIN " + "A " * 300 + "-----END PRIVATE KEY-----",
        "a-----BEGIN EC PRIVATE KEY-----b-----END EC PRIVATE KEY-----c" * 3,
        "-----BEGIN RSA PRIVATE KEY-----\n" + "token=Xy9-_.z01234567\n"
        + "-----END RSA PRIVATE KEY-----\ntoken=Xy9-_.z01234567",
    ], ids=lambda text: repr(text[:40]))
    def test_hand_written_edges(self, text):
        _assert_same(text)

    def test_the_executor_replay_reads_the_same_matches(self):
        """``subagent_executor._redact_tracking`` replays the table with
        ``finditer`` and the matches' spans and groups."""
        from utils import subagent_executor

        text = "lead " + _PEM + " middle -----BEGIN EC PRIVATE KEY----- tail"
        offset = text.index("middle")
        redacted, position = subagent_executor._redact_tracking(text, offset)
        assert redacted == _old_redact_text(text)
        assert redacted[position:].startswith("middle")

    def test_a_string_replacement_and_a_count(self):
        text = _PEM + " and " + _PEM
        assert _NEW_PEM.sub("[k]", text) == _OLD_PEM.sub("[k]", text)
        assert _NEW_PEM.sub("[k]", text, count=1) == _OLD_PEM.sub("[k]", text, count=1)
        assert _NEW_PEM.pattern == _OLD_PEM.pattern


# --- Bounded time, in a child process. --------------------------------------------


def _run_timed(body, what):
    script = textwrap.dedent(f"""
        import json, sys, time
        sys.path.insert(0, {str(SKILL_ROOT)!r})
        size = 1024 * 1024
    """) + textwrap.dedent(body) + textwrap.dedent("""
        print(json.dumps({"elapsed": elapsed}))
    """)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{what} took over 60 s on 1 MiB: the PEM row is quadratic again "
                    "(see write_safety._PemPrivateKeyBlocks)")
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1])["elapsed"] < 20


def test_redact_text_takes_a_megabyte_of_unterminated_headers_in_bounded_time():
    """About 0.1 s at 32 KB and four times that per doubling with the regex
    (minutes at a megabyte); milliseconds now."""
    _run_timed("""
        from utils.write_safety import redact_text
        header = "-----BEGIN RSA PRIVATE KEY-----"
        texts = [
            (header + "\\n") * (size // 32),
            (header + "\\r\\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\\r\\n")
            * (size // 100),
            header * (size // 31),
            "-----BEGIN " + "A" * size,
            ("-----BEGIN EC PRIVATE KEY-----x-----END EC PRIVATE KEY-----" * (size // 120))
            + (header + "\\n") * (size // 64),
        ]
        started = time.perf_counter()
        for text in texts:
            redact_text(text)
        elapsed = time.perf_counter() - started
    """, "redact_text")
