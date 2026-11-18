"""Greedy Best-First Search for deterministic state-space problems."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Hashable
from itertools import count
from numbers import Real
from time import perf_counter
from typing import TypeVar

from study_planner.search.core import (
    SearchProblem,
    SearchResult,
    SearchStatistics,
    _reconstruct_path,
    _validate_cost,
)

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


def _validate_heuristic(value: Real) -> float:
    """Convert a heuristic value to float and reject unsupported values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Heuristic values must be finite, non-negative real numbers")
    try:
        estimate = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "Heuristic values must be finite, non-negative real numbers"
        ) from error
    if not math.isfinite(estimate) or estimate < 0:
        raise ValueError("Heuristic values must be finite and non-negative")
    return estimate


def greedy_best_first_search(
    problem: SearchProblem[StateT, ActionT],
    heuristic: Callable[[StateT], float],
) -> SearchResult[StateT, ActionT]:
    """Find a goal path by prioritizing heuristic value alone.

    Queue entries are ordered by ``h(state)`` and then insertion order; path
    cost is accumulated only to report the returned solution's actual cost and
    never contributes to queue priority. The heuristic must be deterministic
    for each immutable, hash-stable state. States are expanded at most once;
    duplicate successors are ignored. As with standard Greedy Best-First
    Search, the returned solution is not guaranteed to have minimum cost and
    may be more expensive than a UCS solution.

    Action costs are still validated and accumulated for result reporting.
    ``nodes_generated`` counts every successor produced before duplicate
    filtering; ``nodes_expanded`` counts non-goal states whose actions are
    enumerated. The initial state is not counted as generated.
    """
    started = perf_counter()
    initial_state = problem.initial_state
    sequence = count()
    nodes_expanded = 0
    nodes_generated = 0
    expanded_order: list[StateT] = []
    parents: dict[StateT, tuple[StateT, ActionT]] = {}
    path_costs: dict[StateT, float] = {initial_state: 0.0}

    if problem.is_goal(initial_state):
        state_path, action_path = _reconstruct_path(
            initial_state, initial_state, parents
        )
        return SearchResult(
            state_path=state_path,
            action_path=action_path,
            solution_cost=0.0,
            statistics=SearchStatistics(
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                execution_time_seconds=perf_counter() - started,
            ),
        )

    initial_heuristic = _validate_heuristic(heuristic(initial_state))
    frontier: list[tuple[float, int, StateT]] = [
        (initial_heuristic, next(sequence), initial_state)
    ]
    best_heuristic: dict[StateT, float] = {initial_state: initial_heuristic}
    expanded: set[StateT] = set()

    while frontier:
        estimate, _, state = heapq.heappop(frontier)
        if estimate != best_heuristic.get(state) or state in expanded:
            continue

        if problem.is_goal(state):
            state_path, action_path = _reconstruct_path(
                initial_state, state, parents
            )
            return SearchResult(
                state_path=state_path,
                action_path=action_path,
                solution_cost=path_costs[state],
                statistics=SearchStatistics(
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    execution_time_seconds=perf_counter() - started,
                    expanded_order=tuple(expanded_order),
                ),
            )

        expanded.add(state)
        nodes_expanded += 1
        expanded_order.append(state)
        for action in problem.actions(state):
            next_state = problem.transition(state, action)
            nodes_generated += 1
            step_cost = _validate_cost(
                problem.action_cost(state, action, next_state)
            )
            next_cost = path_costs[state] + step_cost
            if not math.isfinite(next_cost):
                raise ValueError("Cumulative path costs must remain finite")

            if next_state in best_heuristic:
                continue

            next_heuristic = _validate_heuristic(heuristic(next_state))
            best_heuristic[next_state] = next_heuristic
            path_costs[next_state] = next_cost
            parents[next_state] = (state, action)
            heapq.heappush(
                frontier, (next_heuristic, next(sequence), next_state)
            )

    return SearchResult(
        state_path=(),
        action_path=(),
        solution_cost=None,
        statistics=SearchStatistics(
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            execution_time_seconds=perf_counter() - started,
            expanded_order=tuple(expanded_order),
        ),
    )
