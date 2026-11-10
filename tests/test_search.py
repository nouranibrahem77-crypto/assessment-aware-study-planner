"""Correctness tests for generic Uniform Cost Search."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from study_planner.search.ucs import uniform_cost_search


@dataclass(frozen=True)
class Edge:
    action: str
    destination: str
    cost: float


@dataclass(frozen=True)
class GraphProblem:
    start: str
    goals: frozenset[str]
    graph: dict[str, tuple[Edge, ...]]

    @property
    def initial_state(self) -> str:
        return self.start

    def is_goal(self, state: str) -> bool:
        return state in self.goals

    def actions(self, state: str) -> tuple[Edge, ...]:
        return self.graph.get(state, ())

    def transition(self, state: str, action: Edge) -> str:
        assert action in self.graph.get(state, ())
        return action.destination

    def action_cost(self, state: str, action: Edge, next_state: str) -> float:
        assert action.destination == next_state
        return action.cost


def test_finds_known_minimum_cost_solution() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("direct-expensive", "goal", 8),
                Edge("to-a", "a", 2),
                Edge("to-b", "b", 1),
            ),
            "a": (Edge("a-to-goal", "goal", 2),),
            "b": (Edge("b-to-goal", "goal", 5),),
        },
    )

    result = uniform_cost_search(problem)

    assert result.found
    assert result.state_path == ("start", "a", "goal")
    assert tuple(edge.action for edge in result.action_path) == ("to-a", "a-to-goal")
    assert result.solution_cost == 4


def test_updates_duplicate_state_when_a_cheaper_path_is_found() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("expensive-x", "x", 9),
                Edge("to-y", "y", 1),
                Edge("to-x", "x", 2),
            ),
            "y": (Edge("cheap-x", "x", 0),),
            "x": (Edge("finish", "goal", 1),),
        },
    )

    result = uniform_cost_search(problem)

    assert result.state_path == ("start", "y", "x", "goal")
    assert tuple(edge.action for edge in result.action_path) == (
        "to-y",
        "cheap-x",
        "finish",
    )
    assert result.solution_cost == 2


def test_supports_zero_cost_transitions_and_ignores_zero_cost_cycle() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (Edge("free", "free", 0),),
            "free": (
                Edge("zero-cycle", "start", 0),
                Edge("finish", "goal", 2),
            ),
        },
    )

    result = uniform_cost_search(problem)

    assert result.state_path == ("start", "free", "goal")
    assert result.solution_cost == 2


def test_returns_clean_no_solution_when_frontier_is_exhausted() -> None:
    result = uniform_cost_search(
        GraphProblem(
            start="start",
            goals=frozenset({"unreachable"}),
            graph={"start": (Edge("dead-end", "dead", 1),)},
        )
    )

    assert not result.found
    assert result.state_path == ()
    assert result.action_path == ()
    assert result.solution_cost is None
    assert result.statistics.nodes_expanded == 2
    assert result.statistics.nodes_generated == 1
    assert result.statistics.execution_time_seconds >= 0


def test_initial_goal_returns_zero_cost_empty_action_path() -> None:
    result = uniform_cost_search(
        GraphProblem(start="goal", goals=frozenset({"goal"}), graph={})
    )

    assert result.found
    assert result.state_path == ("goal",)
    assert result.action_path == ()
    assert result.solution_cost == 0
    assert result.statistics.nodes_expanded == 0
    assert result.statistics.nodes_generated == 0


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_rejects_negative_or_non_finite_action_cost(cost: float) -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={"start": (Edge("invalid", "goal", cost),)},
    )

    with pytest.raises(ValueError, match="cost"):
        uniform_cost_search(problem)


def test_equal_cost_ties_follow_deterministic_action_order() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal-a", "goal-b"}),
        graph={
            "start": (
                Edge("first", "goal-a", 1),
                Edge("second", "goal-b", 1),
            )
        },
    )

    first_result = uniform_cost_search(problem)
    second_result = uniform_cost_search(problem)

    assert first_result.state_path == second_result.state_path == ("start", "goal-a")
    assert first_result.action_path == second_result.action_path == (
        problem.graph["start"][0],
    )


def test_statistics_count_generated_successors_before_duplicate_filtering() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("to-a", "a", 1),
                Edge("duplicate-a", "a", 1),
            ),
            "a": (Edge("finish", "goal", 1),),
        },
    )

    result = uniform_cost_search(problem)

    assert result.statistics.nodes_generated == 3
