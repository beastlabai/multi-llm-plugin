"""Corpora for the JSON extractor: recorded provider outputs, fenced and prose-wrapped documents.

``USAGE_CAP_ENVELOPE`` is a recorded usage-cap envelope and ``FIXTURES`` is
``tests/fixtures``; the ``_*_mix`` builders generate seeded mixes of fenced and
prose-wrapped documents for the property tests. No fixtures.
"""

import json
from pathlib import Path


#: The 2026-09-17 envelope, re-keyed. ``claude-code:claude-fable-5-1:high``
#: reviewed for 491 s, spawned four subagents, spent $25.05 and 137 348 output
#: tokens, and then hit the account's usage cap: ``api_error_status: 429`` with
#: the vendor's sentence in ``result`` and **no** ``API Error:`` prefix, no
#: ``429`` and no "rate limit" anywhere in the text — the text ladder cannot
#: read it and only the status rung can. The session and event ids are fresh;
#: nothing else is changed, because the bookkeeping shape (``iterations: []``,
#: ``permission_denials: []``, the four-entry ``subagent_stats``) is what the
#: gate misread. The top-level ``usage`` block being all zeros beside a
#: populated ``modelUsage`` is measured too, not an edit.
USAGE_CAP_ENVELOPE = (
    '{"duration_api_ms":1738676,"stop_reason":"stop_sequence",'
    '"session_id":"00000000-0000-4000-8000-000000000001",'
    '"total_cost_usd":25.05318824999999,'
    '"usage":{"output_tokens_details":{"thinking_tokens":0},"input_tokens":0,'
    '"cache_creation_input_tokens":0,"cache_read_input_tokens":0,'
    '"output_tokens":0,"server_tool_use":{"web_search_requests":0,'
    '"web_fetch_requests":0},"service_tier":"standard",'
    '"cache_creation":{"ephemeral_1h_input_tokens":0,'
    '"ephemeral_5m_input_tokens":0},"inference_geo":"","iterations":[],'
    '"speed":"standard"},'
    '"modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":1944,'
    '"outputTokens":13,"cacheReadInputTokens":0,"cacheCreationInputTokens":0,'
    '"webSearchRequests":0,"costUSD":0.002009,"contextWindow":200000,'
    '"maxOutputTokens":32000,"thinkingTokens":0,'
    '"canonicalModel":"claude-haiku-4-5","provider":"firstParty",'
    '"costBasis":"list"},"claude-fable-5-1":{"inputTokens":1936,'
    '"outputTokens":137348,"cacheReadInputTokens":37205114,'
    '"cacheCreationInputTokens":1065254,"webSearchRequests":0,'
    '"costUSD":25.051179249999993,"contextWindow":1000000,'
    '"maxOutputTokens":64000,"thinkingTokens":75679,'
    '"canonicalModel":"claude-fable-5-1","provider":"firstParty",'
    '"costBasis":"list"}},"permission_denials":[],'
    '"terminal_reason":"api_error","fast_mode_state":"off",'
    '"fast_mode_disabled_reason":"sdk_opt_in_required",'
    '"origin":{"kind":"task-notification"},'
    '"subagent_stats":{"spawned":4,"requested":{"background":4,'
    '"foreground":0,"unset":0},"started_in_background":4,"max_depth":1,'
    '"spawned_by_subagents":0,"completed":2,"failed":2,'
    '"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,'
    '"concurrency_limit":0,"budget":0},"by_type":{"general-purpose":4}},'
    '"is_error":true,"num_turns":1,"subtype":"success","api_error_status":429,'
    '"result":"You\'ve reached your Fable limit. Switch to another model, or '
    'manage usage credits at claude.ai/settings/usage?from=cc_cli_limit_message, '
    'to continue.","type":"result","duration_ms":409,'
    '"uuid":"00000000-0000-4000-8000-000000000002","queued_turn_count":0,'
    '"result_index":3}'
)


#: Token alphabets for random strings. A token lands whole, so multi-character
#: noise (an ANSI colour code, an escaped quote or backslash) stays intact.
_WALK_ALPHABETS = (
    ("[", "]", "{", "}", '"', "\\", " ", "a", "x", ",", ":", "\n"),
    ("[", "]", '"', "\\"),
    ("{", "}", '"', "\\"),
    ("[", "]", "{", "}", '"', "\\", '\\"', "\\\\", '""'),
    ("[", "]", "{", "}", '"', "\\", "\x1b[32m", "\x1b[0m", "é", "\U0001F600", "x"),
)

_TRICKY_STRINGS = (
    'he said "hi"', "back\\slash", "] [ } {", '\\"', "\\", '"',
    "\x1b[31mred\x1b[0m", "é\U0001F600", "",
)

_PROSE_NOISE = (
    "\n", " prose [not json ", ' x "unterminated ', "\\",
    "\x1b[33mWARN\x1b[0m step [3/9] ", "{ and then } ", "",
)


def _random_value(rng, depth=0):
    roll = rng.random()
    if depth >= 4 or roll < 0.35:
        return rng.choice(
            [rng.randint(-5, 500), rng.choice(_TRICKY_STRINGS), True, None, 1.5])
    if roll < 0.7:
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {
        rng.choice(_TRICKY_STRINGS) + str(i): _random_value(rng, depth + 1)
        for i in range(rng.randint(0, 4))
    }


def _document_mix(rng):
    """Prose, ANSI noise and several JSON documents, some fenced and some cut
    short anywhere, the last one included."""
    parts = []
    for _ in range(rng.randint(1, 5)):
        doc = json.dumps(
            _random_value(rng),
            ensure_ascii=rng.random() < 0.5,
            indent=rng.choice([None, 2]),
        )
        shape = rng.random()
        if shape < 0.25:
            doc = doc[: rng.randint(0, len(doc))]
        elif shape < 0.4:
            doc = "```json\n" + doc + "\n```"
        parts.append(doc)
        parts.append(rng.choice(_PROSE_NOISE))
    return "".join(parts)


FIXTURES = Path(__file__).parent.parent / "fixtures"


#: Hand-written edges: language tags, unclosed fences, fences inside a block,
#: backticks that are not three, closing brackets not followed by a fence,
#: whitespace the pattern's ``\s`` counts (CRLF, NBSP, U+2028, U+001C).
_FENCE_EDGE_CASES = [
    "", "```", "``````", "```[", "```[]", "```[]```", "```{}```", "```json[1]```",
    "```json\n[1]\n```", "```json\n{\"a\": 1}\n```", "```JSON\n[1]\n```",
    "```jsonc\n[1]\n```", "```python\n[1]\n```", "```js\n[1]\n```", "```json json[1]```",
    "````[1]````", "````json\n[1]\n````", "``[1]``", "~~~json\n[1]\n~~~",
    "```[1]````", "```[1] ```x```", "```[1] x ```[2]```", "```[1]\n```\n```[2]\n```",
    "```python\nprint(1)\n```\n```json\n[2]\n```", "```[``` ]```", "```[ ```[1]``` ]```",
    "```[1]``````[2]```", "```[1]```[2]```", "```[1}```", "```{1]```", "```[[1],[2]]```",
    "```[\"]```\"]```", "```json\r\n[1]\r\n```", "```json [1] ```",
    "```json\x1c[1]\x1f```", "```\t\n [1] \t\n```", "] ``` [1] ```", "```json\n[1]\n",
    "```json\n[1]", "prose ```json\n[{\"a\": [1, 2]}]\n``` more prose ``` tail",
    "```json\n[1]\n``` ```json\n[\n``` ```json\n[3]\n```", "```" + "json" * 3 + "[1]```",
    "x```[]```y```{}```z", "```json\n[1]\n```json\n[2]\n```", "```[]　```",
]

#: Token alphabets for random strings, each heavy in what the fence rule reads.
_FENCE_ALPHABETS = (
    ("```", "`", "json", "[", "]", "{", "}", " ", "\n", "x"),
    ("```", "``", "````", "[", "]", " "),
    ("```json\n", "```", "\n```", "[1]", "{}", "]", "}", "[", "{", "python\n", "\r\n"),
    ("```", "json", "JSON", "[", "}", "\t", " ", " ", "\x1c", "\x85", "é", "a"),
)


def _fence_mix(rng):
    """Prose and documents fenced every way a model fences one: tagged and not,
    with another language, unclosed, cut short, closing on the same line."""
    parts = []
    for _ in range(rng.randint(1, 6)):
        doc = json.dumps(_random_value(rng), indent=rng.choice([None, 2]))
        opening = rng.choice(["```json\n", "```\n", "```", "```json ", "```python\n",
                              "````json\n", "``` json\n", "```JSON\n"])
        closing = rng.choice(["\n```", "```", "\n````", "\n", " ``` ", ""])
        if rng.random() < 0.2:
            doc = doc[: rng.randint(0, len(doc))]
        parts.append(opening + doc + closing)
        parts.append(rng.choice(_PROSE_NOISE + ("\n\n", " see ```x``` ", "`[1]`")))
    return "".join(parts)


def _fenced_fixture_texts():
    """Every committed fixture holding a fence, decoded leniently: captured
    provider output is the corpus fences come from."""
    texts = []
    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", errors="replace")
            if "```" in text:
                texts.append(text)
    return texts


#: Edges where the array-only block and the greedy span differ from the
#: either-kind block: mismatched closers, an object fenced before an array,
#: a closer only before the first opener.
_FIRST_READER_EDGE_CASES = [
    "```[1}```", "```[1}``` ```[2]```", "```{\"a\": [1]}```", "```{\"a\": 1}``` ```[1]```",
    "```[1} ] ```", "```[{]```", "``` {}```[1]```", "] [", "} {", "][", "}{", "[", "]",
    "x [1] y [2] z", "x {1} y {2} z", "[ { ] }", "{ [ } ]", "[\n]", "{\n}",
    "```json\n{\"k\": [1, 2]}\n```\n[3]", "[1] ```json\n[2]\n```",
]
