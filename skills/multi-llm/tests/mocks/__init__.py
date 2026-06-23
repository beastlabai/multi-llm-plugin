# Mock LLM CLI binaries for testing
from pathlib import Path

# Path to the unified mock LLM binary
MOCK_LLM_PATH = Path(__file__).parent / "mock_llm.py"

# Import mock_llm module for direct access to mock functions
from . import mock_llm

__all__ = [
    "MOCK_LLM_PATH",
    "mock_llm",
]
