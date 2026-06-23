#!/usr/bin/env python3
"""Tests for fast backoff support in review_plan_orchestrator."""

import os
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from review_plan_orchestrator import get_backoff_delay


class TestGetBackoffDelay:
    """Tests for get_backoff_delay function."""

    def test_normal_mode_returns_base_delay(self):
        """Without env var, returns the base delay unchanged."""
        # Ensure env var is not set
        os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)

        assert get_backoff_delay(5) == 5
        assert get_backoff_delay(10) == 10
        assert get_backoff_delay(20) == 20
        assert get_backoff_delay(2.5) == 2.5

    def test_fast_mode_returns_10ms(self):
        """With MULTI_LLM_TEST_FAST_BACKOFF=1, returns 10ms (0.01s)."""
        os.environ["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"
        try:
            assert get_backoff_delay(5) == 0.01
            assert get_backoff_delay(10) == 0.01
            assert get_backoff_delay(20) == 0.01
            assert get_backoff_delay(2.5) == 0.01
        finally:
            os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)

    def test_env_var_must_be_exactly_one(self):
        """Only the value '1' enables fast mode."""
        # Test various values that should NOT enable fast mode
        for value in ["0", "true", "True", "yes", "enabled", ""]:
            os.environ["MULTI_LLM_TEST_FAST_BACKOFF"] = value
            try:
                assert get_backoff_delay(5) == 5, f"Value '{value}' should not enable fast mode"
            finally:
                os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)

    def test_env_var_unset_returns_base_delay(self):
        """When env var is completely unset, returns base delay."""
        # Ensure it's definitely unset
        os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)

        assert get_backoff_delay(15) == 15

    def test_fast_mode_with_stagger_delay(self):
        """Fast mode works correctly with provider stagger delay values."""
        os.environ["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"
        try:
            # Typical stagger delays: index * 2.0s
            assert get_backoff_delay(2.0) == 0.01  # 1 * 2.0
            assert get_backoff_delay(4.0) == 0.01  # 2 * 2.0
            assert get_backoff_delay(6.0) == 0.01  # 3 * 2.0
        finally:
            os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)

    def test_fast_mode_with_rate_limit_backoffs(self):
        """Fast mode works correctly with rate limit retry backoff values."""
        os.environ["MULTI_LLM_TEST_FAST_BACKOFF"] = "1"
        try:
            # Rate limit backoffs: [5, 10, 20]
            assert get_backoff_delay(5) == 0.01
            assert get_backoff_delay(10) == 0.01
            assert get_backoff_delay(20) == 0.01
        finally:
            os.environ.pop("MULTI_LLM_TEST_FAST_BACKOFF", None)
