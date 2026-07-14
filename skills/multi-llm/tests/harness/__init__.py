"""Test harness package for end-to-end integration tests.

This package provides test infrastructure for running multi-llm skill
orchestrators with mock LLM providers in isolated test environments.

Main components:
- SkillRunner: Runs orchestrators via subprocess with PATH manipulation
- FixtureManager: Creates isolated test directories and loads fixtures
- MockProvider: Configures mock LLM behavior
- AssertionHelpers: Custom assertion methods for test verification
"""

from .skill_runner import SkillRunner, SkillResult, PERF_SCALE
from .fixture_manager import FixtureManager, FixturePlan
from .mock_provider import MockProvider, MockLLMCall
from .assertion_helpers import AssertionHelpers

# Backward compatibility aliases
PlanFixture = FixturePlan
TestPlanFixture = FixturePlan

__all__ = [
    "SkillRunner",
    "SkillResult",
    "PERF_SCALE",
    "FixtureManager",
    "FixturePlan",
    "PlanFixture",  # backward compat alias
    "TestPlanFixture",  # backward compat alias
    "MockProvider",
    "MockLLMCall",
    "AssertionHelpers",
]
