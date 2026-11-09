"""The concurrency both executor lanes share: launch spacing, backoff, the task gather.

:func:`plan_launch_limiter` spaces two dispatches of one provider by its
stagger (:func:`get_provider_launch_stagger`, :data:`PROVIDER_STAGGER_DELAY`
overriding every adapter); :func:`get_backoff_delay` is the test switch for a
retry's sleep; :func:`run_tasks_concurrent` is the read/analyze executor's
gather, keyed by task id, with the fan-out's semaphores, spacing and overall
timeout. ``review_orchestrator_base.run_models_concurrent`` (the model
fan-out) and the executor (``task_batch``, ``task_dispatch``) read them here,
below the review stack (docs/development/dispatcher.md#import-layers);
``review_orchestrator_base`` re-exports the functions for reading.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .launch_limiter import LaunchLimiter
from .provider_registry import (
    get_provider,
    get_provider_max_concurrent,
    get_provider_timeout,
    parse_model_spec,
)


# Launch spacing. The minimum interval between two dispatches of the *same*
# provider is the adapter's own ``LAUNCH_STAGGER_S`` (``utils/providers/base.py``,
# enforced by ``utils/launch_limiter.py``); this module attribute is only a
# uniform override of every adapter's value, ``None`` in production. Tests set
# it (``0.0`` to switch the spacing off, a positive number to force it on
# providers that declare none).
PROVIDER_STAGGER_DELAY: Optional[float] = None


def get_backoff_delay(base_delay: float) -> float:
    """Get backoff delay, respecting test mode for fast retries.

    When MULTI_LLM_TEST_FAST_BACKOFF=1 is set, returns minimal delay (10ms).
    """
    if os.environ.get("MULTI_LLM_TEST_FAST_BACKOFF") == "1":
        return 0.01
    return base_delay


def get_provider_launch_stagger(provider_name: str) -> float:
    """Seconds between two launches of one provider (0 means none).

    Declared by the adapter (``LLMProvider.LAUNCH_STAGGER_S``), not by
    ``providers.yaml``; :data:`PROVIDER_STAGGER_DELAY` overrides every adapter
    when set. A name with no adapter staggers nothing.
    """
    if PROVIDER_STAGGER_DELAY is not None:
        return float(PROVIDER_STAGGER_DELAY)
    provider = get_provider(provider_name)
    if provider is None:
        return 0.0
    return float(getattr(provider, "LAUNCH_STAGGER_S", 0.0) or 0.0)


def plan_launch_limiter(
    provider_names: Sequence[str],
) -> Tuple[LaunchLimiter, float]:
    """One fan-out's launch limiter, and the margin its overall deadline adds.

    The limiter spaces two dispatches of one provider by that provider's
    stagger (:func:`get_provider_launch_stagger`, through
    :func:`get_backoff_delay`); a provider with none is never throttled. Every
    slot of the fan-out shares it, retries included. The margin is the largest
    provider's slot count times its stagger. See
    docs/development/delegation.md#read-and-analyze-executor.
    """
    counts: Dict[str, int] = {}
    for name in provider_names:
        counts[name] = counts.get(name, 0) + 1
    staggers = {name: get_provider_launch_stagger(name) for name in counts}
    limiter = LaunchLimiter({
        name: get_backoff_delay(stagger)
        for name, stagger in staggers.items() if stagger > 0
    })
    overhead = max(
        (counts[name] * stagger for name, stagger in staggers.items()), default=0.0
    )
    return limiter, overhead


async def run_tasks_concurrent(
    tasks: List[Dict[str, Any]],
    per_task_coro_factory: Callable[[Dict[str, Any], str, int], Any],
    *,
    runner_for_task: Callable[[Dict[str, Any]], str],
    timeout: Optional[float] = None,
    max_parallel: int = 4,
    skip_predicate: Optional[Callable[[Dict[str, Any], str, int], Optional[Any]]] = None,
    provider_limits: Optional[Dict[str, int]] = None,
    anchor: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """``review_orchestrator_base.run_models_concurrent``'s mechanics, keyed by ``task_id`` (D3).

    ``run_models_concurrent`` returns a dict keyed by ``model_spec`` — correct for
    model fan-out, **wrong** for a subagent batch, where every task normally
    shares the one configured runner, so completed tasks would overwrite one
    another. This is the same machinery keyed by each task's stable ``task_id``:

    - the same global ``asyncio.Semaphore(max_parallel)``
    - the same per-provider semaphores from ``get_provider_max_concurrent``,
      acquired *before* the global one, and **shared in shape** with the fan-out
      path so a mixed-tier batch on one provider contends on ONE semaphore
      (the rate limit belongs to the provider, not the model)
    - the same launch spacing, per resolved provider
    - the same overall timeout wrapping ``asyncio.gather``

    ``model_spec`` is used *only* to pick the provider semaphore and launch
    spacing; it is a column on each row, never the key. ``run_models_concurrent``
    keeps its current signature and callers.

    Args:
        tasks: The materialized task objects (not model specs — each task carries
            its own prompt and its runner is a *derived* attribute).
        per_task_coro_factory: ``(task, runner_spec, index) -> awaitable``.
        runner_for_task: Resolves a task's runner spec once, at batch start.
        timeout: Per-task timeout override, also driving the overall timeout.
        max_parallel: The global gate (``driver.max_concurrent_subagents``,
            already reduced to D3i's three-way minimum by the caller).
        skip_predicate: Cheap synchronous ``(task, runner, index) -> result|None``
            checked *before* any semaphore acquisition and launch turn, so a
            resume of already-complete tasks stays fast.
        provider_limits: Optional per-provider in-flight overrides (D3i
            backpressure narrows a throttled provider's slots for the rest of the
            batch); defaults to ``get_provider_max_concurrent``.
        anchor: Optional discovery anchor (the run's plan-file path), the Lane B
            twin of ``run_models_concurrent``'s. ``subagent_executor``
            already holds the phase's plan path and passes it.

    Returns:
        ``{task_id: whatever the callback returned}``.
    """
    results: Dict[str, Any] = {}
    if not tasks:
        return results

    runners = {
        str(task.get("task_id", index)): runner_for_task(task)
        for index, task in enumerate(tasks)
    }

    semaphore = asyncio.Semaphore(max_parallel)

    provider_semaphores: Dict[str, asyncio.Semaphore] = {}
    for spec in set(runners.values()):
        prov, _ = parse_model_spec(spec, anchor=anchor)
        if prov in provider_semaphores:
            continue
        limit = (provider_limits or {}).get(
            prov, get_provider_max_concurrent(prov, anchor=anchor)
        )
        if limit is not None:
            provider_semaphores[prov] = asyncio.Semaphore(limit)

    launcher, stagger_overhead = plan_launch_limiter(
        [
            parse_model_spec(runners[str(task.get("task_id", index))], anchor=anchor)[0]
            for index, task in enumerate(tasks)
        ]
    )

    async def run_with_semaphore(task: Dict[str, Any], index: int) -> Tuple[str, Any]:
        task_id = str(task.get("task_id", index))
        runner = runners[task_id]
        # Anchored for the same reason the fan-out's display resolution is: the
        # semaphore this picks must be the one the batch was sized for.
        provider_name, _ = parse_model_spec(runner, anchor=anchor)

        if skip_predicate is not None:
            skipped = skip_predicate(task, runner, index)
            if skipped is not None:
                return (task_id, skipped)

        # The launch turn is taken inside both semaphores, as on the fan-out;
        # `dispatch_task` confirms or renews it before every attempt.
        prov_sem = provider_semaphores.get(provider_name)
        turn = launcher.slot(provider_name, label=f"{task_id} {runner}")
        if prov_sem:
            async with prov_sem:
                async with semaphore, turn:
                    return (task_id, await per_task_coro_factory(task, runner, index))
        else:
            async with semaphore, turn:
                return (task_id, await per_task_coro_factory(task, runner, index))

    coros = [
        asyncio.create_task(run_with_semaphore(task, idx))
        for idx, task in enumerate(tasks)
    ]

    if timeout is not None:
        base_timeout = timeout
    else:
        base_timeout = max(
            get_provider_timeout(parse_model_spec(spec, anchor=anchor)[0], anchor=anchor)
            for spec in runners.values()
        )
    total_timeout = base_timeout * 2 + stagger_overhead

    try:
        completed = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=True), timeout=total_timeout
        )
        for item in completed:
            if isinstance(item, Exception):
                print(f"Subagent task exception: {item}")
                continue
            task_id, result = item
            results[task_id] = result
    except asyncio.TimeoutError:
        print(f"Total subagent batch timeout ({total_timeout}s) exceeded")
        for coro in coros:
            if not coro.done():
                coro.cancel()

    return results
