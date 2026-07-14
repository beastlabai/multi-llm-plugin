#!/usr/bin/env python3
"""Mock opencode CLI for testing.

This mock simulates the opencode CLI behavior for integration tests.

Environment variables:
    MULTI_LLM_TEST_MODE: Must be set to "1" for the mock to run (safety check)
    MOCK_LLM_FIXTURE: Path to a fixture file to use as the response
    MOCK_LLM_FAIL: If set to "1", exit with an error
    MOCK_LLM_TIMEOUT: If set to "1", sleep forever (simulate timeout)
    MOCK_OUTPUT_PATH: Path to write output file to (alternative to parsing from prompt)

Output format:
    NDJSON (newline-delimited JSON) events:
    {"type": "step_start", ...}
    {"type": "text", "part": {"type": "text", "text": "..."}}
    {"type": "step_finish", ...}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    # Safety check: only run in test mode
    if os.environ.get("MULTI_LLM_TEST_MODE") != "1":
        print(
            "ERROR: mock_opencode.py should only be used in test mode. "
            "Set MULTI_LLM_TEST_MODE=1 to enable.",
            file=sys.stderr,
        )
        return 1

    # Check for simulated failures
    if os.environ.get("MOCK_LLM_FAIL") == "1":
        print("ERROR: Simulated failure", file=sys.stderr)
        return 1

    # Check for simulated timeout
    if os.environ.get("MOCK_LLM_TIMEOUT") == "1":
        while True:
            time.sleep(1)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Mock opencode CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--format", dest="output_format", default="json")
    run_parser.add_argument("--model", default="claude-sonnet")
    run_parser.add_argument("prompt", nargs="?", default="")

    args = parser.parse_args()

    # Handle non-run commands
    if args.command != "run":
        print("ERROR: Only 'run' subcommand is supported", file=sys.stderr)
        return 1

    prompt = args.prompt

    # Determine output file path
    output_path = None
    if os.environ.get("MOCK_OUTPUT_PATH"):
        output_path = Path(os.environ["MOCK_OUTPUT_PATH"])
    elif "OUTPUT_FILE:" in prompt:
        # Extract output file path from prompt
        for line in prompt.split("\n"):
            if "OUTPUT_FILE:" in line:
                output_path = Path(line.split("OUTPUT_FILE:")[1].strip())
                break

    # Get response content
    fixture_path = os.environ.get("MOCK_LLM_FIXTURE")
    if fixture_path and Path(fixture_path).exists():
        with open(fixture_path, "r", encoding="utf-8") as f:
            response_content = f.read()
    else:
        # Default response
        response_content = json.dumps([
            {
                "id": "mock-opencode-1",
                "type": "fix",
                "description": "Mock fix from opencode",
                "importance": "HIGH",
                "location": "mock_file.py:30"
            }
        ])

    # Write to output file if specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response_content)

    # Output in opencode NDJSON format
    events = [
        {
            "type": "step_start",
            "step_id": "mock-step-1",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": response_content,
            },
        },
        {
            "type": "step_finish",
            "step_id": "mock-step-1",
            "timestamp": "2024-01-01T00:00:01Z",
        },
    ]

    # Print one JSON object per line (NDJSON format)
    for event in events:
        print(json.dumps(event))

    return 0


if __name__ == "__main__":
    sys.exit(main())
