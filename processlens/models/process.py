from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class Step(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    actor: str = Field(..., min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    next_step_ids: list[str] = Field(default_factory=list)
    is_decision: bool = Field(default=False)
    conditions: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_loop: bool = Field(default=False)


class ProcessModel(BaseModel):
    id: Optional[str] = Field(default=None)
    title: str = Field(..., min_length=1)
    version: str = Field(default="1.0")
    description: Optional[str] = Field(default="")
    actors: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph_semantics(self) -> "ProcessModel":
        if not self.steps:
            return self

        step_ids = [step.id for step in self.steps]
        unique_ids = set(step_ids)
        if len(step_ids) != len(unique_ids):
            seen = set()
            duplicates = set()
            for s_id in step_ids:
                if s_id in seen:
                    duplicates.add(s_id)
                seen.add(s_id)
            raise ValueError(f"Duplicate step IDs found: {sorted(list(duplicates))}")

        for step in self.steps:
            if step.actor not in self.actors:
                self.actors.append(step.actor)

        for step in self.steps:
            for next_id in step.next_step_ids:
                if next_id not in unique_ids:
                    raise ValueError(
                        f"Step '{step.id}' references unknown next step '{next_id}'"
                    )

        if len(self.steps) > 1:
            in_degrees = {s_id: 0 for s_id in unique_ids}
            out_degrees = {s.id: len(s.next_step_ids) for s in self.steps}
            for step in self.steps:
                for next_id in step.next_step_ids:
                    in_degrees[next_id] += 1

            for s_id in unique_ids:
                if in_degrees[s_id] == 0 and out_degrees[s_id] == 0:
                    raise ValueError(
                        f"Orphan step detected: '{s_id}' has no incoming or outgoing connections"
                    )

            undirected_adj: dict[str, set[str]] = {s_id: set() for s_id in unique_ids}
            for step in self.steps:
                for next_id in step.next_step_ids:
                    undirected_adj[step.id].add(next_id)
                    undirected_adj[next_id].add(step.id)

            visited_undirected = set()
            queue = [self.steps[0].id]
            visited_undirected.add(self.steps[0].id)
            while queue:
                current = queue.pop(0)
                for neighbor in undirected_adj[current]:
                    if neighbor not in visited_undirected:
                        visited_undirected.add(neighbor)
                        queue.append(neighbor)

            disconnected = unique_ids - visited_undirected
            if disconnected:
                raise ValueError(
                    f"Orphan steps detected in disconnected component: {sorted(list(disconnected))}"
                )

            entry_points = [s_id for s_id, deg in in_degrees.items() if deg == 0]
            if not entry_points:
                entry_points = [self.steps[0].id]

            reachable_from_entries = set()
            bfs_queue = list(entry_points)
            reachable_from_entries.update(entry_points)
            step_dict = {s.id: s for s in self.steps}

            while bfs_queue:
                current_id = bfs_queue.pop(0)
                current_step = step_dict[current_id]
                for next_id in current_step.next_step_ids:
                    if next_id not in reachable_from_entries:
                        reachable_from_entries.add(next_id)
                        bfs_queue.append(next_id)

            unreachable = unique_ids - reachable_from_entries
            if unreachable:
                raise ValueError(
                    f"Orphan steps detected: steps unreachable from entry points: {sorted(list(unreachable))}"
                )

        adjacency: dict[str, list[str]] = {step.id: step.next_step_ids for step in self.steps}
        step_map = {step.id: step for step in self.steps}

        state: dict[str, int] = {s_id: 0 for s_id in unique_ids}
        stack: list[str] = []

        def dfs_detect_cycles(node: str) -> None:
            state[node] = 1
            stack.append(node)

            for neighbor in adjacency[node]:
                if neighbor == node:
                    if not step_map[node].is_loop:
                        raise ValueError(
                            f"Cycle detected without explicit loop marker: {node} -> {node}"
                        )
                elif state[neighbor] == 1:
                    cycle_start_index = stack.index(neighbor)
                    cycle_nodes = stack[cycle_start_index:]
                    has_loop_marker = any(step_map[c_id].is_loop for c_id in cycle_nodes)
                    if not has_loop_marker:
                        cycle_repr = " -> ".join(cycle_nodes + [neighbor])
                        raise ValueError(
                            f"Cycle detected without explicit loop marker: {cycle_repr}"
                        )
                elif state[neighbor] == 0:
                    dfs_detect_cycles(neighbor)

            stack.pop()
            state[node] = 2

        for s_id in unique_ids:
            if state[s_id] == 0:
                dfs_detect_cycles(s_id)

        return self
