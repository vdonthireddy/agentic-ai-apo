"""
Pareto Multi-Objective Frontier Engine for GEPA.
Implements Pareto dominance, non-dominated ranking, and diversity preservation
across multiple competing prompt objectives (e.g., accuracy, schema validity, token efficiency).
"""

from typing import Dict, List, Set, Tuple
from app.core.evaluator import PromptCandidate

def dominates(candidate_a: PromptCandidate, candidate_b: PromptCandidate, objectives: List[str]) -> bool:
    """
    Check if candidate_a Pareto-dominates candidate_b.
    a dominates b if:
      - a is at least as good as b in all objectives (a[obj] >= b[obj])
      - a is strictly better than b in at least one objective (a[obj] > b[obj])
    """
    at_least_as_good = True
    strictly_better = False

    for obj in objectives:
        score_a = candidate_a.scores.get(obj, 0.0)
        score_b = candidate_b.scores.get(obj, 0.0)

        if score_a < score_b - 1e-6:
            at_least_as_good = False
            break
        if score_a > score_b + 1e-6:
            strictly_better = True

    return at_least_as_good and strictly_better

def compute_pareto_frontiers(
    candidates: List[PromptCandidate],
    objectives: List[str]
) -> List[List[PromptCandidate]]:
    """
    Fast non-dominated sorting algorithm (NSGA-II style).
    Returns list of Pareto frontiers [Frontier_0 (Rank 1), Frontier_1 (Rank 2), ...].
    """
    if not candidates:
        return []

    # Map each candidate to the set of candidates it dominates
    dominates_map: Dict[str, List[PromptCandidate]] = {c.candidate_id: [] for c in candidates}
    # Count of candidates that dominate this candidate
    domination_count: Dict[str, int] = {c.candidate_id: 0 for c in candidates}

    frontiers: List[List[PromptCandidate]] = [[]]

    for i, p in enumerate(candidates):
        for j, q in enumerate(candidates):
            if i == j:
                continue
            if dominates(p, q, objectives):
                dominates_map[p.candidate_id].append(q)
            elif dominates(q, p, objectives):
                domination_count[p.candidate_id] += 1

        if domination_count[p.candidate_id] == 0:
            p.pareto_rank = 1
            p.is_frontier = True
            frontiers[0].append(p)
        else:
            p.is_frontier = False

    current_rank = 0
    candidate_lookup = {c.candidate_id: c for c in candidates}

    while current_rank < len(frontiers) and frontiers[current_rank]:
        next_frontier: List[PromptCandidate] = []
        for p in frontiers[current_rank]:
            for q in dominates_map[p.candidate_id]:
                domination_count[q.candidate_id] -= 1
                if domination_count[q.candidate_id] == 0:
                    q.pareto_rank = current_rank + 2
                    q.is_frontier = False
                    next_frontier.append(q)
        current_rank += 1
        if next_frontier:
            frontiers.append(next_frontier)

    return frontiers

def compute_crowding_distance(front: List[PromptCandidate], objectives: List[str]) -> None:
    """
    Assign crowding distance to each candidate in a frontier to preserve diversity.
    Boundary candidates receive infinity so extreme solutions are never lost.
    """
    n = len(front)
    if n == 0:
        return
    for c in front:
        c.crowding_distance = 0.0

    if n <= 2:
        for c in front:
            c.crowding_distance = float("inf")
        return

    for obj in objectives:
        # Sort by objective value
        front.sort(key=lambda c: c.scores.get(obj, 0.0))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        min_val = front[0].scores.get(obj, 0.0)
        max_val = front[-1].scores.get(obj, 0.0)
        denom = max_val - min_val
        if denom < 1e-9:
            continue

        for i in range(1, n - 1):
            next_score = front[i + 1].scores.get(obj, 0.0)
            prev_score = front[i - 1].scores.get(obj, 0.0)
            front[i].crowding_distance += (next_score - prev_score) / denom

def get_pareto_frontier(candidates: List[PromptCandidate], objectives: List[str]) -> List[PromptCandidate]:
    """Return the set of non-dominated (Rank 1) candidates."""
    frontiers = compute_pareto_frontiers(candidates, objectives)
    if not frontiers or not frontiers[0]:
        return []
    frontier = frontiers[0]
    compute_crowding_distance(frontier, objectives)
    return frontier
