#!/usr/bin/env python3
"""Unified mock LLM binary for testing.

This mock serves as a unified replacement for all provider-specific mocks.
It detects which provider is being emulated via sys.argv[0] (symlinked names)
and outputs in the correct format for each provider.

Supported Providers:
    - cursor-agent: {"type": "result", "result": "..."}
    - gemini: {"session_id": "...", "response": "...", "stats": {...}}
    - opencode: NDJSON events (step_start, text, step_finish)
    - codex: NDJSON events (text, message, content)
    - kilocode: Direct JSON output

Environment Variables (Legacy - for backward compatibility):
    MULTI_LLM_TEST_MODE: Must be set to "1" for the mock to run (safety check)
    MOCK_LLM_FIXTURE: Path to a fixture file to use as the response
    MOCK_LLM_FAIL: If set to "1", exit with an error
    MOCK_LLM_TIMEOUT: If set to "1", simulate timeout (quick exit with error)
    MOCK_OUTPUT_PATH: Path to write output file to (alternative to parsing from prompt)

Environment Variables (New - for scenario-based testing):
    MOCK_LLM_SCENARIO: Path to scenario YAML file for pattern-based responses
    MOCK_LLM_CONFIG: JSON string with dynamic configuration
    MOCK_LLM_CALL_LOG: Path to JSONL file for logging all invocations

Precedence:
    1. Legacy variables take precedence when set (backward compatibility)
    2. Scenario-based variables used when legacy variables are not set
    3. MOCK_LLM_CALL_LOG is always respected regardless of mode
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_provider_name() -> str:
    """Detect provider name from sys.argv[0] (symlink name).

    When this script is symlinked as 'cursor-agent', 'gemini', etc.,
    sys.argv[0] will contain that name (or path ending with that name).

    Returns:
        The provider name (cursor-agent, gemini, opencode, codex, kilocode)
        or 'unknown' if not detected.
    """
    invoked_name = Path(sys.argv[0]).name

    # Handle both direct name and any suffix/prefix variations
    known_providers = ["cursor-agent", "gemini", "opencode", "codex", "kilocode"]

    for provider in known_providers:
        if invoked_name == provider or invoked_name.endswith(provider):
            return provider

    # Fallback: if invoked directly as mock_llm.py, check for provider env var
    if os.environ.get("MOCK_LLM_PROVIDER"):
        return os.environ["MOCK_LLM_PROVIDER"]

    return "unknown"


def load_scenario(scenario_path: str) -> Optional[Dict[str, Any]]:
    """Load a scenario YAML file.

    Args:
        scenario_path: Path to the scenario YAML file.

    Returns:
        Parsed scenario dict or None if loading fails.
    """
    try:
        import yaml
    except ImportError:
        # yaml not available, return None
        return None

    try:
        with open(scenario_path, "r") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def match_prompt_pattern(prompt: str, scenario: Dict[str, Any]) -> Optional[str]:
    """Match prompt against scenario patterns and return fixture path.

    Args:
        prompt: The prompt text to match.
        scenario: The loaded scenario dict.

    Returns:
        Path to fixture file or None if no match.
    """
    prompts = scenario.get("prompts", [])
    for entry in prompts:
        pattern = entry.get("pattern", "")
        fixture = entry.get("fixture", "")
        if pattern and fixture:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return fixture
            except re.error:
                # Invalid regex, skip
                continue
    return None


def get_response_content(prompt: str) -> str:
    """Determine response content based on environment and prompt.

    Precedence:
    1. MOCK_LLM_FIXTURE (legacy)
    2. MOCK_LLM_SCENARIO pattern matching
    3. MOCK_LLM_CONFIG
    4. Default response

    Args:
        prompt: The prompt text (used for pattern matching).

    Returns:
        Response content string.
    """
    # Legacy: MOCK_LLM_FIXTURE takes highest precedence
    fixture_path = os.environ.get("MOCK_LLM_FIXTURE")
    if fixture_path and Path(fixture_path).exists():
        with open(fixture_path, "r") as f:
            return f.read()

    # New: MOCK_LLM_SCENARIO for pattern-based responses
    scenario_path = os.environ.get("MOCK_LLM_SCENARIO")
    if scenario_path:
        scenario = load_scenario(scenario_path)
        if scenario:
            fixture_rel = match_prompt_pattern(prompt, scenario)
            if fixture_rel:
                # Resolve fixture path relative to scenario file location
                scenario_dir = Path(scenario_path).parent
                fixture_full = scenario_dir / fixture_rel
                if fixture_full.exists():
                    with open(fixture_full, "r") as f:
                        return f.read()

    # New: MOCK_LLM_CONFIG for dynamic configuration
    config_json = os.environ.get("MOCK_LLM_CONFIG")
    if config_json:
        try:
            config = json.loads(config_json)
            if "response" in config:
                response = config["response"]
                if isinstance(response, str):
                    return response
                return json.dumps(response)
        except json.JSONDecodeError:
            pass

    # Default response based on provider
    provider = get_provider_name()
    return json.dumps([
        {
            "id": f"mock-{provider}-1",
            "type": "suggestion",
            "description": f"Mock suggestion from {provider}",
            "importance": "MEDIUM",
            "location": "mock_file.py:10"
        }
    ])


def write_output_file(prompt: str, content: str) -> None:
    """Write response content to output file if specified.

    Args:
        prompt: The prompt text (may contain OUTPUT_FILE directive).
        content: The response content to write.
    """
    output_path = None

    # Check environment variable first
    if os.environ.get("MOCK_OUTPUT_PATH"):
        output_path = Path(os.environ["MOCK_OUTPUT_PATH"])
    elif "OUTPUT_FILE:" in prompt:
        # Extract output file path from prompt
        for line in prompt.split("\n"):
            if "OUTPUT_FILE:" in line:
                output_path = Path(line.split("OUTPUT_FILE:")[1].strip())
                break

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)


def log_call(provider: str, args: List[str], prompt: str) -> None:
    """Log the mock call to JSONL file if configured.

    Args:
        provider: The detected provider name.
        args: Command line arguments.
        prompt: The prompt text.
    """
    log_path = os.environ.get("MOCK_LLM_CALL_LOG")
    if not log_path:
        return

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provider": provider,
        "argv": sys.argv,
        "args": args,
        "prompt": prompt,
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("MOCK_LLM_") or k == "MULTI_LLM_TEST_MODE"
        }
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def format_cursor_agent_output(content: str) -> str:
    """Format output for cursor-agent provider.

    Format: {"type": "result", "result": "<content>"}
    """
    return json.dumps({"type": "result", "result": content})


def format_gemini_output(content: str) -> str:
    """Format output for gemini provider.

    Format: {"session_id": "...", "response": "<content>", "stats": {...}}
    """
    return json.dumps({
        "session_id": "mock-session-12345",
        "response": content,
        "stats": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
    })


def format_opencode_output(content: str) -> str:
    """Format output for opencode provider.

    Format: NDJSON events (step_start, text, step_finish)
    """
    events = [
        {
            "type": "step_start",
            "step_id": "mock-step-1",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": content,
            },
        },
        {
            "type": "step_finish",
            "step_id": "mock-step-1",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def format_codex_output(content: str) -> str:
    """Format output for codex provider.

    Format: NDJSON events (text type events)
    The codex parser looks for:
    - event.type == "text" with event.text or event.part.text
    - event.type == "message" with event.content
    - event.type == "content" with event.text
    """
    events = [
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": content,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def format_kilocode_output(content: str) -> str:
    """Format output for kilocode provider.

    Format: Direct JSON output (the content itself, or wrapped if needed)
    The kilocode parser expects direct JSON that can be parsed.
    """
    # kilocode just outputs the content directly
    # If content is already valid JSON, return as-is
    # Otherwise, wrap it
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        return json.dumps({"output": content})


def format_output(provider: str, content: str) -> str:
    """Format output according to provider's expected wire format.

    Args:
        provider: The provider name.
        content: The response content.

    Returns:
        Formatted output string.
    """
    formatters = {
        "cursor-agent": format_cursor_agent_output,
        "gemini": format_gemini_output,
        "opencode": format_opencode_output,
        "codex": format_codex_output,
        "kilocode": format_kilocode_output,
    }

    formatter = formatters.get(provider, format_cursor_agent_output)
    return formatter(content)


def parse_provider_args(provider: str) -> tuple[argparse.Namespace, str]:
    """Parse command-line arguments based on provider.

    Each provider has slightly different argument patterns.

    Args:
        provider: The provider name.

    Returns:
        Tuple of (parsed args namespace, prompt string)
    """
    if provider == "cursor-agent":
        parser = argparse.ArgumentParser(description="Mock cursor-agent CLI")
        parser.add_argument("--print", dest="print_mode", action="store_true")
        parser.add_argument("-f", "--force", action="store_true")
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--model", default="auto")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "gemini":
        parser = argparse.ArgumentParser(description="Mock gemini CLI")
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--model", default="gemini-2.0-flash")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "opencode":
        parser = argparse.ArgumentParser(description="Mock opencode CLI")
        subparsers = parser.add_subparsers(dest="command")
        run_parser = subparsers.add_parser("run")
        run_parser.add_argument("--format", dest="output_format", default="json")
        run_parser.add_argument("--model", default="claude-sonnet")
        run_parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        if args.command != "run":
            print("ERROR: Only 'run' subcommand is supported", file=sys.stderr)
            sys.exit(1)
        return args, args.prompt

    elif provider == "codex":
        parser = argparse.ArgumentParser(description="Mock codex CLI")
        subparsers = parser.add_subparsers(dest="command")
        exec_parser = subparsers.add_parser("exec")
        exec_parser.add_argument("--full-auto", action="store_true")
        exec_parser.add_argument("--json", action="store_true")
        exec_parser.add_argument("--model", default="gpt-4")
        exec_parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        if args.command != "exec":
            print("ERROR: Only 'exec' subcommand is supported", file=sys.stderr)
            sys.exit(1)
        return args, args.prompt

    elif provider == "kilocode":
        parser = argparse.ArgumentParser(description="Mock kilocode CLI")
        parser.add_argument("--auto", action="store_true")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    else:
        # Unknown provider - basic argument parsing
        parser = argparse.ArgumentParser(description="Mock LLM CLI")
        parser.add_argument("--model", default="default")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt


def main() -> int:
    """Main entry point for the unified mock LLM binary.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Safety check: only run in test mode
    if os.environ.get("MULTI_LLM_TEST_MODE") != "1":
        print(
            "ERROR: mock_llm.py should only be used in test mode. "
            "Set MULTI_LLM_TEST_MODE=1 to enable.",
            file=sys.stderr,
        )
        return 1

    # Detect provider from invocation name
    provider = get_provider_name()

    # Parse arguments based on provider
    args, prompt = parse_provider_args(provider)

    # Log the call (always, regardless of mode)
    log_call(provider, sys.argv[1:], prompt)

    # Check for simulated failures (legacy)
    if os.environ.get("MOCK_LLM_FAIL") == "1":
        print("ERROR: Simulated failure", file=sys.stderr)
        return 1

    # Check for simulated timeout (legacy)
    # Instead of sleeping forever (which would hang tests), we exit with
    # a recognizable timeout error pattern after a brief delay
    if os.environ.get("MOCK_LLM_TIMEOUT") == "1":
        # Small delay to simulate some processing, then fail with timeout error
        time.sleep(0.05)  # 50ms
        print("ERROR: Simulated timeout - operation timed out", file=sys.stderr)
        return 124  # Standard timeout exit code

    # Get response content
    content = get_response_content(prompt)

    # Write to output file if specified
    write_output_file(prompt, content)

    # Format and print output according to provider's wire format
    output = format_output(provider, content)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
