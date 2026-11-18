"""Correctness tests for generic Uniform Cost Search."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from study_planner.search.astar import a_star_search
from study_planner.search.greedy import greedy_best_first_search
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
    assert result.statistics.expanded_order == ("start", "b", "a")
    assert result.statistics.nodes_expanded == 3
    assert result.statistics.nodes_generated == 5
    assert "goal" not in result.statistics.expanded_order
    assert isinstance(result.statistics.expanded_order, tuple)


def test_a_star_finds_optimal_path_with_consistent_heuristic() -> None:
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
    heuristic = {"start": 4, "a": 2, "b": 5, "goal": 0}.__getitem__

    result = a_star_search(problem, heuristic)

    assert result.found
    assert result.state_path == ("start", "a", "goal")
    assert tuple(edge.action for edge in result.action_path) == ("to-a", "a-to-goal")
    assert result.solution_cost == 4
    assert result.statistics.nodes_expanded == 2
    assert result.statistics.nodes_generated == 4
    assert result.statistics.expanded_order == ("start", "a")
    assert "goal" not in result.statistics.expanded_order


def test_a_star_rejects_inconsistent_heuristic() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={"start": (Edge("finish", "goal", 1),)},
    )

    with pytest.raises(ValueError, match="consistent"):
        a_star_search(problem, {"start": 2, "goal": 0}.__getitem__)


def test_a_star_requires_zero_heuristic_at_goal() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={"start": (Edge("finish", "goal", 1),)},
    )

    with pytest.raises(ValueError, match="zero at goal"):
        a_star_search(problem, {"start": 0, "goal": 1}.__getitem__)


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
    assert result.statistics.expanded_order == ("start", "dead")
    assert "unreachable" not in result.statistics.expanded_order


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
    assert result.statistics.expanded_order == ()


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


def test_greedy_follows_low_heuristic_path_and_reconstructs_solution() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("to-a", "a", 4),
                Edge("to-b", "b", 1),
            ),
            "a": (Edge("finish-a", "goal", 1),),
            "b": (Edge("finish-b", "goal", 1),),
        },
    )
    heuristic_values = {"start": 5, "a": 3, "b": 1, "goal": 0}

    result = greedy_best_first_search(problem, heuristic_values.__getitem__)

    assert result.state_path == ("start", "b", "goal")
    assert tuple(edge.action for edge in result.action_path) == ("to-b", "finish-b")
    assert result.solution_cost == 2
    assert result.statistics.expanded_order == ("start", "b")
    assert result.statistics.nodes_expanded == 2
    assert result.statistics.nodes_generated == 3
    assert "goal" not in result.statistics.expanded_order


def test_greedy_can_return_a_higher_cost_solution_than_ucs() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("toward-expensive", "expensive", 10),
                Edge("toward-cheap", "cheap", 1),
            ),
            "expensive": (Edge("expensive-finish", "goal", 10),),
            "cheap": (Edge("cheap-finish", "goal", 1),),
        },
    )
    heuristic_values = {"start": 5, "expensive": 1, "cheap": 2, "goal": 0}

    greedy_result = greedy_best_first_search(
        problem, heuristic_values.__getitem__
    )
    ucs_result = uniform_cost_search(problem)

    assert greedy_result.state_path == ("start", "expensive", "goal")
    assert greedy_result.solution_cost == 20
    assert ucs_result.solution_cost == 2


def test_greedy_ignores_duplicate_states_after_first_discovery() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"goal"}),
        graph={
            "start": (
                Edge("to-a", "a", 1),
                Edge("to-b", "b", 1),
            ),
            "a": (Edge("a-to-shared", "shared", 1),),
            "b": (Edge("b-to-shared", "shared", 1),),
            "shared": (Edge("finish", "goal", 1),),
        },
    )
    heuristic_values = {
        "start": 0,
        "a": 1,
        "b": 2,
        "shared": 3,
        "goal": 0,
    }

    result = greedy_best_first_search(problem, heuristic_values.__getitem__)

    assert result.state_path == ("start", "a", "shared", "goal")
    assert result.statistics.nodes_expanded == 4
    assert result.statistics.nodes_generated == 5
    assert result.statistics.expanded_order == ("start", "a", "b", "shared")
    assert "goal" not in result.statistics.expanded_order


def test_greedy_equal_heuristics_follow_deterministic_insertion_order() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"first-goal", "second-goal"}),
        graph={
            "start": (
                Edge("first", "first-goal", 1),
                Edge("second", "second-goal", 1),
            )
        },
    )
    heuristic_values = {"start": 1, "first-goal": 0, "second-goal": 0}

    first = greedy_best_first_search(problem, heuristic_values.__getitem__)
    second = greedy_best_first_search(problem, heuristic_values.__getitem__)

    assert first.state_path == second.state_path == ("start", "first-goal")


def test_greedy_returns_no_solution_when_frontier_is_exhausted() -> None:
    problem = GraphProblem(
        start="start",
        goals=frozenset({"missing"}),
        graph={"start": (Edge("dead-end", "dead", 1),)},
    )

    result = greedy_best_first_search(problem, lambda _: 0)

    assert not result.found
    assert result.state_path == ()
    assert result.action_path == ()
    assert result.solution_cost is None
    assert result.statistics.nodes_expanded == 2
    assert result.statistics.nodes_generated == 1
    assert result.statistics.execution_time_seconds >= 0
    assert result.statistics.expanded_order == ("start", "dead")
    assert "missing" not in result.statistics.expanded_order


def test_greedy_initial_goal_does_not_need_heuristic_evaluation() -> None:
    problem = GraphProblem(start="goal", goals=frozenset({"goal"}), graph={})

    def fail_if_called(_: str) -> float:
        raise AssertionError("An initial goal does not need a heuristic")

    result = greedy_best_first_search(problem, fail_if_called)

    assert result.found
    assert result.state_path == ("goal",)
    assert result.action_path == ()
    assert result.solution_cost == 0
    assert result.statistics.nodes_expanded == 0
    assert result.statistics.nodes_generated == 0
    assert result.statistics.expanded_order == ()


def test_a_star_initial_goal_has_empty_expansion_order() -> None:
    result = a_star_search(
        GraphProblem(start="goal", goals=frozenset({"goal"}), graph={}),
        lambda _: 0,
    )

    assert result.found
    assert result.state_path == ("goal",)
    assert result.solution_cost == 0
    assert result.statistics.nodes_expanded == 0
    assert result.statistics.nodes_generated == 0
    assert result.statistics.expanded_order == ()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_greedy_rejects_invalid_heuristic_values(value: float) -> None:
    problem = GraphProblem(start="start", goals=frozenset(), graph={})

    with pytest.raises(ValueError, match="(?i)heuristic"):
        greedy_best_first_search(problem, lambda _: value)
