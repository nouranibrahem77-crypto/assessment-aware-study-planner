"""Typed contracts and immutable result records for graph search."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


def _validate_cost(value: Real) -> float:
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
    """Reconstruct states and actions from a predecessor map."""
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


class SearchProblem(Protocol[StateT, ActionT]):
    """A deterministic state-space problem consumed by generic search."""

    @property
    def initial_state(self) -> StateT:
        """Return the immutable, hashable initial state."""

    def is_goal(self, state: StateT) -> bool:
        """Return whether ``state`` satisfies the problem goal."""

    def actions(self, state: StateT) -> Iterable[ActionT]:
        """Return legal actions in deterministic order."""

    def transition(self, state: StateT, action: ActionT) -> StateT:
        """Return the successor state without mutating ``state``."""

    def action_cost(
        self, state: StateT, action: ActionT, next_state: StateT
    ) -> float:
        """Return the finite, non-negative cost of one transition."""


@dataclass(frozen=True, slots=True)
class SearchStatistics:
    """Counters and elapsed wall time for one search invocation."""

    nodes_expanded: int
    nodes_generated: int
    execution_time_seconds: float


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[StateT, ActionT]):
    """A found path or an exhausted-search result with its statistics."""

    state_path: tuple[StateT, ...]
    action_path: tuple[ActionT, ...]
    solution_cost: float | None
    statistics: SearchStatistics

    @property
    def found(self) -> bool:
        """Whether the search found a goal state."""
        return self.solution_cost is not None
