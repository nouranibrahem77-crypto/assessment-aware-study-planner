"""Uniform Cost Search for finite, non-negative-cost state spaces."""

from __future__ import annotations

import heapq
import math
from collections.abc import Hashable
from itertools import count
from numbers import Real
from time import perf_counter
from typing import TypeVar

from study_planner.search.core import SearchProblem, SearchResult, SearchStatistics

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


def _validated_cost(value: Real) -> float:
    """Convert one action cost to float and reject unsupported costs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Action costs must be finite, non-negative real numbers")
    try:
        cost = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "Action costs must be finite, non-negative real numbers"
        ) from error
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Action costs must be finite and non-negative")
    return cost


def _reconstruct_path(
    initial_state: StateT,
    goal_state: StateT,
    parents: dict[StateT, tuple[StateT, ActionT]],
) -> tuple[tuple[StateT, ...], tuple[ActionT, ...]]:
    """Reconstruct states and actions from the predecessor map."""
    states = [goal_state]
    actions: list[ActionT] = []
    current = goal_state
    while current != initial_state:
        previous, action = parents[current]
        states.append(previous)
        actions.append(action)
        current = previous
    states.reverse()
    actions.reverse()
    return tuple(states), tuple(actions)


def uniform_cost_search(
    problem: SearchProblem[StateT, ActionT],
) -> SearchResult[StateT, ActionT]:
    """Find a minimum-cost goal path using Uniform Cost Search.

    The problem must expose immutable, hashable states and deterministic action
    ordering. Each action cost must be finite and non-negative. For a finite
    reachable state space these conditions make UCS complete and optimal; the
    first goal removed from the priority queue has minimum path cost. Zero-cost
    actions are supported: equal-cost duplicate paths are ignored, while a
    strictly cheaper path updates the best-cost record and can reopen a state.

    ``nodes_generated`` counts every successor produced by the transition
    interface before duplicate/stale-path filtering. ``nodes_expanded`` counts
    non-goal states whose actions are enumerated. The initial state is not
    counted as generated.
    """
    started = perf_counter()
    initial_state = problem.initial_state
    sequence = count()
    frontier: list[tuple[float, int, StateT]] = [
        (0.0, next(sequence), initial_state)
    ]
    best_cost: dict[StateT, float] = {initial_state: 0.0}
    parents: dict[StateT, tuple[StateT, ActionT]] = {}
    nodes_expanded = 0
    nodes_generated = 0

    while frontier:
        path_cost, _, state = heapq.heappop(frontier)
        if path_cost != best_cost.get(state):
            continue

        if problem.is_goal(state):
            state_path, action_path = _reconstruct_path(
                initial_state, state, parents
            )
            return SearchResult(
                state_path=state_path,
                action_path=action_path,
                solution_cost=path_cost,
                statistics=SearchStatistics(
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    execution_time_seconds=perf_counter() - started,
                ),
            )

        nodes_expanded += 1
        for action in problem.actions(state):
            next_state = problem.transition(state, action)
            nodes_generated += 1
            step_cost = _validated_cost(
                problem.action_cost(state, action, next_state)
            )
            next_cost = path_cost + step_cost
            if not math.isfinite(next_cost):
                raise ValueError("Cumulative path costs must remain finite")
            previous_best = best_cost.get(next_state)
            if previous_best is not None and next_cost >= previous_best:
                continue

            best_cost[next_state] = next_cost
            parents[next_state] = (state, action)
            heapq.heappush(
                frontier, (next_cost, next(sequence), next_state)
            )

    return SearchResult(
        state_path=(),
        action_path=(),
        solution_cost=None,
        statistics=SearchStatistics(
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            execution_time_seconds=perf_counter() - started,
        ),
    )
