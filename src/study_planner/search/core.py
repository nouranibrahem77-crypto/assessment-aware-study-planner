"""Typed contracts and immutable result records for graph search."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


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
