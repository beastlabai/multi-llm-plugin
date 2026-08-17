"""The one result-envelope parser (§9.5).

``cursor-agent --output-format json`` and ``claude -p --output-format json``
both wrap the model's answer in the same envelope::

    {"type": "result", "result": "<the answer, usually JSON as a string>", ...}

``cursor_agent.parse_output`` and ``claude_code.parse_output`` each carried a
34-line copy of the unwrapping logic, differing in **one comment**, with 0/9
behavioural differences measured over the envelope shapes in
``tests/test_result_envelope.py``. This module is that logic, once.

Not folded in here: ``llm_client.parse_subagent_response``, the legacy third
copy. It uses a weaker extractor and returns bare parsed JSON instead of a
``{"success": ..., "data": ...}`` result, and it is **live** behind the shipped
``--internal-validation`` flag (``review_orchestrator_base`` →
``validation.validate_groups`` → ``invoke_for_json`` → ``invoke_subagent`` +
``parse_subagent_response``). Unifying it would change what that flag returns,
so it stays until the flag is retired.
"""
import json
from typing import Any, Dict

from ..json_extractor import extract_json_from_text


def parse_result_envelope(stdout: str) -> Dict[str, Any]:
    """Unwrap a ``{"type": "result", "result": ...}`` envelope.

    The envelope itself is discarded — adapters that need what it carries
    (``usage``, ``total_cost_usd``, ``modelUsage``) read it separately in
    ``parse_usage``, before this drops it.

    Args:
        stdout: The child's captured stdout.

    Returns:
        ``{"success": True, "data": ...}`` or ``{"success": False, "error":
        ..., "data": None}``, the shape every ``parse_output`` returns.
    """
    stdout = stdout.strip()

    if stdout.startswith("{"):
        try:
            wrapper = json.loads(stdout)
            if isinstance(wrapper, dict) and "result" in wrapper:
                inner_result = wrapper["result"]
                if isinstance(inner_result, str):
                    inner_result = inner_result.strip()
                    if inner_result.startswith(("[", "{")):
                        try:
                            return {"success": True, "data": json.loads(inner_result)}
                        except json.JSONDecodeError:
                            pass
                    # Try to extract JSON from text
                    return extract_json_from_text(inner_result, prefer_arrays=True)
                return (
                    {"success": True, "data": inner_result}
                    if inner_result
                    else {"success": False, "error": "Empty result", "data": None}
                )
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError:
            pass

    # Try direct JSON parsing for arrays
    if stdout.startswith("["):
        try:
            return {"success": True, "data": json.loads(stdout)}
        except json.JSONDecodeError:
            pass

    return extract_json_from_text(stdout, prefer_arrays=True)
