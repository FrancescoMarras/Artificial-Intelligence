from heapq import heappush, heappop
from collections import deque

from utils import (
    reconstruct_path,
    compute_path_cost,
    round_cost,
    heuristic
)

def _bfs_nodes_in_memory(queue, visited, parent):
    """
    Estimate the number of distinct nodes currently stored in memory by BFS.

    The function considers the main BFS data structures:
    - queue: frontier nodes waiting to be expanded
    - visited: nodes already discovered
    - parent: nodes for which a predecessor has been stored

    Since the same node may appear in more than one structure, the union of
    these sets is used to count each stored node only once.

    Parameters
    ----------
    queue : collections.deque[str]
        BFS frontier.
    visited : set[str]
        Set of discovered nodes.
    parent : dict[str, str | None]
        Parent dictionary used for path reconstruction.

    Returns
    -------
    int
        Number of distinct nodes currently stored in memory.
    """
    return len(set(queue) | visited | set(parent.keys()))


def _ucs_nodes_in_memory(priority_queue, expanded, cost_so_far, parent):
    """
    Estimate the number of distinct nodes currently stored in memory by UCS.

    The function considers the main UCS data structures:
    - priority_queue: frontier nodes with their cumulative costs
    - expanded: nodes already expanded
    - cost_so_far: best known cost for each discovered node
    - parent: predecessor dictionary used for path reconstruction

    Since a node may be present in multiple structures at the same time,
    the union of the corresponding node sets is used to avoid double counting.

    Parameters
    ----------
    priority_queue : list[tuple[float, str]]
        Priority queue used by Uniform Cost Search.
    expanded : set[str]
        Set of nodes already expanded.
    cost_so_far : dict[str, float]
        Best known cumulative cost for each discovered node.
    parent : dict[str, str | None]
        Parent dictionary used for path reconstruction.

    Returns
    -------
    int
        Number of distinct nodes currently stored in memory.
    """
    frontier_nodes = {node for _, node in priority_queue}
    return len(frontier_nodes | expanded | set(cost_so_far.keys()) | set(parent.keys()))


def bfs(graph, start, goal, verbose=False, weighted_graph=None, data=None):
    if start not in graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    if goal not in graph:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    queue = deque([start])
    visited = {start}
    parent = {start: None}
    visited_order = []
    step = 0

    expanded_nodes = 0
    max_frontier_size = len(queue)
    max_nodes_in_memory = _bfs_nodes_in_memory(queue, visited, parent)

    if verbose:
        print(f"Start node: {start}")
        print(f"Goal node: {goal}")
        print(f"Initial frontier: {list(queue)}")

    while queue:
        if verbose:
            print(f"Step {step}")
            print(f"Frontier before extraction: {list(queue)}")

        current = queue.popleft()
        visited_order.append(current)
        expanded_nodes += 1

        if verbose:
            print(f"Extracted node: {current}")

        if current == goal:
            if verbose:
                print(f"Goal reached: {goal}")
            break

        newly_added = []

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                newly_added.append(neighbor)

        max_frontier_size = max(max_frontier_size, len(queue))
        max_nodes_in_memory = max(
            max_nodes_in_memory,
            _bfs_nodes_in_memory(queue, visited, parent)
        )

        if verbose:
            print(f"Newly added nodes: {newly_added}")
            print(f"Frontier after expansion: {list(queue)}")
            print(f"Visited order so far: {visited_order}")

        step += 1

    if goal not in parent:
        metrics = {
            "expanded_nodes": expanded_nodes,
            "max_frontier_size": max_frontier_size,
            "max_nodes_in_memory": max_nodes_in_memory,
            "path_cost": None
        }
        return None, visited_order, metrics

    path = reconstruct_path(parent, goal)

    metrics = {
        "expanded_nodes": expanded_nodes,
        "max_frontier_size": max_frontier_size,
        "max_nodes_in_memory": max_nodes_in_memory,
        "path_cost": compute_path_cost(path, weighted_graph)
    }

    return path, visited_order, metrics


def uniform_cost_search(graph, start, goal, verbose=False, weighted_graph=None, data=None):
    if weighted_graph is None:
        weighted_graph = graph

    if start not in weighted_graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    if goal not in weighted_graph:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    priority_queue = [(0.0, start)]

    parent = {start: None}
    cost_so_far = {start: 0.0}
    visited_order = []
    expanded = set()
    step = 0

    expanded_nodes = 0
    max_frontier_size = len(priority_queue)
    max_nodes_in_memory = _ucs_nodes_in_memory(
        priority_queue, expanded, cost_so_far, parent
    )

    if verbose:
        print(f"Start node: {start}")
        print(f"Goal node: {goal}")
        print(f"Initial frontier: {priority_queue}")

    while priority_queue:
        if verbose:
            print(f"Step {step}")
            print(f"Frontier before extraction: {priority_queue}")

        current_cost, current = heappop(priority_queue)

        if current in expanded:
            continue

        expanded.add(current)
        visited_order.append(current)
        expanded_nodes += 1

        if verbose:
            print(f"Extracted node: {current} with cumulative cost {round(current_cost, 2)}")

        if current == goal:
            if verbose:
                print(f"Goal reached: {goal}")
            break

        newly_added = []

        for neighbor, edge_cost in weighted_graph[current].items():
            new_cost = current_cost + edge_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = current
                heappush(priority_queue, (new_cost, neighbor))
                newly_added.append((neighbor, round(new_cost, 2)))

        max_frontier_size = max(max_frontier_size, len(priority_queue))
        max_nodes_in_memory = max(
            max_nodes_in_memory,
            _ucs_nodes_in_memory(priority_queue, expanded, cost_so_far, parent)
        )

        if verbose:
            print(f"Newly added/updated nodes: {newly_added}")
            print(f"Frontier after expansion: {priority_queue}")
            print(f"Visited order so far: {visited_order}")

        step += 1

    if goal not in cost_so_far:
        metrics = {
            "expanded_nodes": expanded_nodes,
            "max_frontier_size": max_frontier_size,
            "max_nodes_in_memory": max_nodes_in_memory,
            "path_cost": None
        }
        return None, visited_order, metrics

    path = reconstruct_path(parent, goal)

    metrics = {
        "expanded_nodes": expanded_nodes,
        "max_frontier_size": max_frontier_size,
        "max_nodes_in_memory": max_nodes_in_memory,
        "path_cost": round_cost(cost_so_far[goal])
    }

    return path, visited_order, metrics

def a_star_search(graph, start, goal, verbose=False, weighted_graph=None, data=None):
    """
    Perform A* Search on a weighted graph.

    A* expands the node with the lowest estimated total cost:
        f(n) = g(n) + h(n)
    where:
    - g(n) is the exact cost from the start node to n
    - h(n) is a heuristic estimate from n to the goal


    This implementation also collects empirical metrics:
    - expanded_nodes: number of expanded nodes
    - max_frontier_size: maximum size of the priority queue
    - max_nodes_in_memory: maximum number of distinct nodes stored
    - path_cost: total cost of the final path

    Parameters
    ----------
    graph : dict[str, list[str]]
        Unweighted adjacency list. It is included to keep the same interface
        used by the other search algorithms, although A* uses `weighted_graph`.
    start : str
        Starting node.
    goal : str
        Target node.
    verbose : bool, optional
        If True, prints the frontier evolution and intermediate steps.
    weighted_graph : dict[str, dict[str, float]], optional
        Weighted graph representation used for path costs.
    data : dict, optional
        Full map data, including city coordinates, used by the heuristic.

    Returns
    -------
    tuple[list[str] | None, list[str], dict]
        Returns:
        - path: optimal path from start to goal, or None if unreachable
        - visited_order: order of node expansion
        - metrics: dictionary containing empirical complexity information

    Notes
    -----
    - With a suitable heuristic, A* is typically more efficient than UCS.
    - If the heuristic is admissible and consistent, A* returns an optimal path.
    """
    if weighted_graph is None:
        raise ValueError("A* requires a weighted graph.")

    if data is None:
        raise ValueError("A* requires the map data with city coordinates.")

    if start not in weighted_graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    if goal not in weighted_graph:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    if start not in data or goal not in data:
        raise ValueError("Start or goal node is missing from the map data.")

    initial_heuristic = heuristic(start, goal, data)

    # Each entry is:
    # (estimated_total_cost, cumulative_cost, current_node)
    priority_queue = [(initial_heuristic, 0.0, start)]

    parent = {start: None}
    cost_so_far = {start: 0.0}
    visited_order = []
    expanded = set()
    step = 0

    expanded_nodes = 0
    max_frontier_size = len(priority_queue)
    max_nodes_in_memory = _ucs_nodes_in_memory(
        [(f, node) for f, g, node in priority_queue],
        expanded,
        cost_so_far,
        parent
    )

    if verbose:
        print(f"Start node: {start}")
        print(f"Goal node: {goal}")
        print(f"Initial frontier: {priority_queue}")

    while priority_queue:
        if verbose:
            print(f"Step {step}")
            print(f"Frontier before extraction: {priority_queue}")

        current_f, current_cost, current = heappop(priority_queue)

        # Ignore obsolete entries extracted from the heap.
        if current in expanded:
            continue

        expanded.add(current)
        visited_order.append(current)
        expanded_nodes += 1

        if verbose:
            print(
                f"Extracted node: {current} "
                f"with g={round_cost(current_cost)}, "
                f"h={round_cost(heuristic(current, goal, data))}, "
                f"f={round_cost(current_f)}"
            )

        if current == goal:
            if verbose:
                print(f"Goal reached: {goal}")
            break

        newly_added = []

        for neighbor, edge_cost in weighted_graph[current].items():
            new_cost = current_cost + edge_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                heuristic_value = heuristic(neighbor, goal, data)
                estimated_total_cost = new_cost + heuristic_value

                cost_so_far[neighbor] = new_cost
                parent[neighbor] = current
                heappush(priority_queue, (estimated_total_cost, new_cost, neighbor))

                newly_added.append(
                    (
                        neighbor,
                        round_cost(new_cost),
                        round_cost(heuristic_value),
                        round_cost(estimated_total_cost)
                    )
                )

        max_frontier_size = max(max_frontier_size, len(priority_queue))
        max_nodes_in_memory = max(
            max_nodes_in_memory,
            _ucs_nodes_in_memory(
                [(f, node) for f, g, node in priority_queue],
                expanded,
                cost_so_far,
                parent
            )
        )

        if verbose:
            print(f"Newly added/updated nodes: {newly_added}")
            print(f"Frontier after expansion: {priority_queue}")
            print(f"Visited order so far: {visited_order}")

        step += 1

    if goal not in cost_so_far:
        metrics = {
            "expanded_nodes": expanded_nodes,
            "max_frontier_size": max_frontier_size,
            "max_nodes_in_memory": max_nodes_in_memory,
            "path_cost": None
        }
        return None, visited_order, metrics

    path = reconstruct_path(parent, goal)

    metrics = {
        "expanded_nodes": expanded_nodes,
        "max_frontier_size": max_frontier_size,
        "max_nodes_in_memory": max_nodes_in_memory,
        "path_cost": round_cost(cost_so_far[goal])
    }

    return path, visited_order, metrics