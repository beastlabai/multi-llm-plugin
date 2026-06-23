#!/usr/bin/env python3
"""Mock cursor-agent CLI for testing.

This mock simulates the cursor-agent CLI behavior for integration tests.

Environment variables:
    MULTI_LLM_TEST_MODE: Must be set to "1" for the mock to run (safety check)
    MOCK_LLM_FIXTURE: Path to a fixture file to use as the response
    MOCK_LLM_FAIL: If set to "1", exit with an error
    MOCK_LLM_TIMEOUT: If set to "1", sleep forever (simulate timeout)
    MOCK_OUTPUT_PATH: Path to write output file to (alternative to parsing from prompt)

Output format:
    {"type": "result", "result": "..."}
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
            "ERROR: mock_cursor_agent.py should only be used in test mode. "
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
    parser = argparse.ArgumentParser(description="Mock cursor-agent CLI")
    parser.add_argument("--print", dest="print_mode", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--output-format", dest="output_format", default="json")
    parser.add_argument("--model", default="auto")
    parser.add_argument("prompt", nargs="?", default="")

    args = parser.parse_args()
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
        with open(fixture_path, "r") as f:
            response_content = f.read()
    else:
        # Default response
        response_content = json.dumps([
            {
                "id": "mock-1",
                "type": "enhancement",
                "description": "Mock suggestion from cursor-agent",
                "importance": "MEDIUM",
                "location": "mock_file.py:10"
            }
        ])

    # Write to output file if specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(response_content)

    # Output in cursor-agent format
    output = {
        "type": "result",
        "result": response_content,
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
