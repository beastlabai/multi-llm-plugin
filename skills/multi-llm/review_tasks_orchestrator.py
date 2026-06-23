#!/usr/bin/env python3
"""
Orchestrator for reviewing generated implementation tasks against the original plan.

This orchestrator sends the task decomposition to multiple LLMs for quality review,
with plan coverage analysis as the primary concern. It follows the review-plan
pattern as a thin specialization -- all shared review-orchestration logic
(model invocation, result saving, validation, grouping, HTML reports, state
management) lives in utils/review_orchestrator_base.py.

The phase is optional: `--implement` does NOT require `--review-tasks`.

Supports multiple providers (cursor-agent, gemini, opencode) via the provider
registry. Models can be specified as 'provider:model' or bare 'model' names
(which use the default provider from providers.yaml).

Usage:
    # Use YAML defaults (no prompting if defaults.models is set)
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py --plan-file plans/my-plan.md

    # Override defaults with specific models
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py --plan-file plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

    # Force interactive selection even if defaults exist
    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/review_tasks_orchestrator.py --plan-file plans/my-plan.md --interactive
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_loader import load_prompt
from utils.provider_registry import get_all_model_specs
from utils.review_orchestrator_base import (
    derive_prefix,
    run_review_orchestration,
)
from utils.tasks_file import find_tasks_file


# Phase configuration constants
PHASE_NAME = 'review-tasks'
INVOKE_PHASE = 'task_review'
SALVAGE_PHASE_NAME = 'review_tasks'
ORCHESTRATOR_SCRIPT = 'review_tasks_orchestrator.py'
TASK_REVIEW_PROMPT_FILE = "task_review.txt"


def load_task_review_prompt_template() -> str:
    """Load the task review prompt template.

    Returns:
        Prompt template string with placeholders for substitution
    """
    return load_prompt(TASK_REVIEW_PROMPT_FILE)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for task review."""
    try:
        available_models = get_all_model_specs()
    except FileNotFoundError:
        available_models = ["(providers.yaml not found - run with --models to specify)"]
    except Exception as e:
        available_models = [f"(config error: {type(e).__name__} - check providers.yaml)"]

    parser = argparse.ArgumentParser(
        description="Review generated implementation tasks against the original plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models (provider:model format):
  {', '.join(available_models[:10])}{'...' if len(available_models) > 10 else ''}

Model selection priority:
  1. --models flag       -> Use specified models
  2. --interactive flag  -> Force two-step interactive selection
  3. --quick flag        -> Use quick_models from providers.yaml (2 models)
  4. YAML defaults       -> Use configured defaults from providers.yaml
  5. Interactive         -> Two-step selection (fallback if no defaults)

Examples:
  # Use YAML defaults (no prompting)
  uv run --project ${{CLAUDE_SKILL_DIR}} -- python ${{CLAUDE_SKILL_DIR}}/review_tasks_orchestrator.py --plan-file plans/my-plan.md

  # Specify models explicitly
  uv run --project ${{CLAUDE_SKILL_DIR}} -- python ${{CLAUDE_SKILL_DIR}}/review_tasks_orchestrator.py --plan-file plans/my-plan.md --models cursor-agent:auto gemini:gemini-2.5-flash

  # Force interactive selection
  uv run --project ${{CLAUDE_SKILL_DIR}} -- python ${{CLAUDE_SKILL_DIR}}/review_tasks_orchestrator.py --plan-file plans/my-plan.md --interactive
        """
    )

    parser.add_argument(
        '--plan-file',
        required=True,
        help='Path to the implementation plan markdown file'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='List of models in provider:model format (e.g., cursor-agent:auto gemini:gemini-2.5-flash). '
             'Bare model names use default provider from providers.yaml.'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Force interactive model selection (ignores YAML defaults)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Use quick_models from providers.yaml for lightweight reviews (2 models)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Override timeout per model in seconds (default: use per-provider timeout from providers.yaml)'
    )

    parser.add_argument(
        '--max-parallel',
        type=int,
        default=5,
        help='Maximum number of parallel model invocations (default: 5)'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip the validation step (faster but no false-positive filtering)'
    )

    parser.add_argument(
        '--validation-model',
        type=str,
        default='auto',
        help='Model to use for validation (default: auto)'
    )

    parser.add_argument(
        '--reaggregate',
        action='store_true',
        help='Re-aggregate existing model results (use after salvage operations complete)'
    )

    parser.add_argument(
        '--internal-validation',
        action='store_true',
        help='Run validation inside orchestrator (legacy mode) instead of delegating to Claude Code subagent'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if phase was previously completed'
    )

    parser.add_argument(
        '--report-style',
        choices=['pr', 'flat'],
        default='pr',
        help='HTML report template style: "pr" for PR-style contextual view, "flat" for classic card layout (default: pr)'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Load the prompt template
    prompt_template = load_task_review_prompt_template()

    # Resolve plan path early for find_tasks_file
    plan_path_abs = os.path.abspath(args.plan_file)

    # Pre-run existence check: validate tasks file before any model invocation.
    # This is done here rather than in the pre_run_hook because we need the
    # resolved tasks_path for the prompt context factory.
    # In reaggregate mode, we don't need the tasks file (just reprocessing results).
    tasks_path = None
    if not args.reaggregate:
        try:
            tasks_path = find_tasks_file(plan_path_abs)
            print(f"Tasks file: {tasks_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    effective_prefix = derive_prefix(plan_path_abs)

    def prompt_context_factory(model_spec: str) -> Dict:
        return {
            "plan_path": plan_path_abs,
            "tasks_path": tasks_path or "",
            "prefix": effective_prefix,
        }

    # Pass tasks_file as extra validation metadata so validation subagents
    # can access the tasks file for coverage analysis
    extra_validation_metadata = {}
    if tasks_path:
        extra_validation_metadata["tasks_file"] = tasks_path

    await run_review_orchestration(
        args=args,
        phase_name=PHASE_NAME,
        invoke_phase=INVOKE_PHASE,
        salvage_phase_name=SALVAGE_PHASE_NAME,
        prompt_template=prompt_template,
        prompt_context_factory=prompt_context_factory,
        orchestrator_script=ORCHESTRATOR_SCRIPT,
        supports_consolidation=False,
        extra_validation_metadata=extra_validation_metadata or None,
        base_ref=None,
        tasks_path=Path(tasks_path) if tasks_path else None,
        template_style=args.report_style,
    )


if __name__ == "__main__":
    asyncio.run(main())
