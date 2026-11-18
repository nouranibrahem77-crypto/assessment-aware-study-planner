"""A* search for finite, non-negative-cost state spaces."""

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
    """Convert one heuristic value and reject invalid estimates."""
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


def a_star_search(
    problem: SearchProblem[StateT, ActionT],
    heuristic: Callable[[StateT], float],
) -> SearchResult[StateT, ActionT]:
    """Find an optimal path using ``g(state) + h(state)`` priority.

    The heuristic must be finite, non-negative, deterministic, zero at goals,
    and consistent on each generated transition. Consistency is checked while
    expanding the graph. With a consistent heuristic, the first goal removed
    from the priority queue has minimum path cost.

    Generated-node counts include each successor before duplicate filtering,
    matching the project's UCS and Greedy conventions.
    """
    started = perf_counter()
    initial_state = problem.initial_state
    nodes_expanded = 0
    nodes_generated = 0
    expanded_order: list[StateT] = []
    if problem.is_goal(initial_state):
        state_path, action_path = _reconstruct_path(
            initial_state, initial_state, {}
        )
        return SearchResult(
            state_path=state_path,
            action_path=action_path,
            solution_cost=0.0,
            statistics=SearchStatistics(
                nodes_expanded,
                nodes_generated,
                perf_counter() - started,
                tuple(expanded_order),
            ),
        )

    sequence = count()
    heuristic_values = {initial_state: _validate_heuristic(heuristic(initial_state))}
    frontier: list[tuple[float, int, float, StateT]] = [
        (heuristic_values[initial_state], next(sequence), 0.0, initial_state)
    ]
    best_cost: dict[StateT, float] = {initial_state: 0.0}
    parents: dict[StateT, tuple[StateT, ActionT]] = {}

    while frontier:
        _, _, path_cost, state = heapq.heappop(frontier)
        if path_cost != best_cost.get(state):
            continue

        if problem.is_goal(state):
            if heuristic_values[state] != 0:
                raise ValueError("Heuristic values must be zero at goal states")
            state_path, action_path = _reconstruct_path(
                initial_state, state, parents
            )
            return SearchResult(
                state_path=state_path,
                action_path=action_path,
                solution_cost=path_cost,
                statistics=SearchStatistics(
                    nodes_expanded,
                    nodes_generated,
                    perf_counter() - started,
                    tuple(expanded_order),
                ),
            )

        nodes_expanded += 1
        expanded_order.append(state)
        state_heuristic = heuristic_values[state]
        for action in problem.actions(state):
            next_state = problem.transition(state, action)
            nodes_generated += 1
            step_cost = _validate_cost(
                problem.action_cost(state, action, next_state)
            )
            next_cost = path_cost + step_cost
            if not math.isfinite(next_cost):
                raise ValueError("Cumulative path costs must remain finite")

            next_heuristic = heuristic_values.get(next_state)
            if next_heuristic is None:
                next_heuristic = _validate_heuristic(heuristic(next_state))
                heuristic_values[next_state] = next_heuristic
            if state_heuristic > step_cost + next_heuristic:
                raise ValueError("Heuristic must be consistent on every transition")

            previous_best = best_cost.get(next_state)
            if previous_best is not None and next_cost >= previous_best:
                continue

            best_cost[next_state] = next_cost
            parents[next_state] = (state, action)
            heapq.heappush(
                frontier,
                (next_cost + next_heuristic, next(sequence), next_cost, next_state),
            )

    return SearchResult(
        state_path=(),
        action_path=(),
        solution_cost=None,
        statistics=SearchStatistics(
            nodes_expanded,
            nodes_generated,
            perf_counter() - started,
            tuple(expanded_order),
        ),
    )
