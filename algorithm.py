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

    """
    frontier_nodes = {node for _, node in priority_queue}
    return len(frontier_nodes | expanded | set(cost_so_far.keys()) | set(parent.keys()))

def _a_star_nodes_in_memory(priority_queue, expanded, cost_so_far, parent):
    """
    Estimate the number of distinct nodes currently stored in memory by A*.

    The function considers the main A* data structures:
    - priority_queue: frontier nodes stored as (f, g, node)
    - expanded: nodes already expanded
    - cost_so_far: best known g(n) for each discovered node
    - parent: predecessor dictionary used for path reconstruction

    Since the same node may appear in multiple structures at the same time,
    the union of the corresponding node sets is used to avoid double counting.
    """
    frontier_nodes = {node for _, _, node in priority_queue}
    return len(frontier_nodes | expanded | set(cost_so_far.keys()) | set(parent.keys()))

def _bidirectional_bfs_nodes_in_memory(
    queue_start, queue_goal, visited_start, visited_goal, parent_start, parent_goal
):
    """
    Estimate the number of distinct nodes currently stored in memory by
    Bidirectional BFS.
    """
    return len(
        set(queue_start)
        | set(queue_goal)
        | visited_start
        | visited_goal
        | set(parent_start.keys())
        | set(parent_goal.keys())
    )


def _reconstruct_bidirectional_path(parent_start, parent_goal, meeting_node):
    """
    Reconstruct the final path once the two BFS frontiers meet.
    """
    path_from_start = reconstruct_path(parent_start, meeting_node)

    path_to_goal = []
    node = parent_goal[meeting_node]

    while node is not None:
        path_to_goal.append(node)
        node = parent_goal[node]

    return path_from_start + path_to_goal


def _expand_bidirectional_layer(
    graph, queue, visited_this, visited_other, parent_this, visited_order
):
    """
    Expand exactly one BFS layer from one side of a bidirectional search.

    Returns:
    - meeting_node: the first node where the two searches meet, or None
    - expanded_in_layer: number of nodes expanded in this layer
    - newly_added: nodes discovered while expanding the layer
    - expanded_nodes_list: nodes extracted from the frontier in this layer
    """
    layer_size = len(queue)
    expanded_in_layer = 0
    newly_added = []
    expanded_nodes_list = []

    for _ in range(layer_size):
        current = queue.popleft()
        visited_order.append(current)
        expanded_nodes_list.append(current)
        expanded_in_layer += 1

        for neighbor in graph[current]:
            if neighbor not in visited_this:
                visited_this.add(neighbor)
                parent_this[neighbor] = current
                queue.append(neighbor)
                newly_added.append(neighbor)

                if neighbor in visited_other:
                    return neighbor, expanded_in_layer, newly_added, expanded_nodes_list

    return None, expanded_in_layer, newly_added, expanded_nodes_list

def bfs(graph, start, goal, verbose=False, weighted_graph=None, data=None):
    """
        Breadth-First Search on an unweighted graph.

        BFS explores the graph level by level, so if all edges are considered to
        have the same cost, it returns a path with the minimum number of edges.
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    if goal not in graph:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    # FIFO queue
    queue = deque([start])

    #Nodes already discovered.
    visited = {start}

    # Parent dictionary
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
    """
       Uniform Cost Search on a weighted graph.

       UCS always expands the node with the smallest cumulative path cost g(n).
       With non-negative edge costs, it returns an optimal path in terms of
       total cost.
    """
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
    max_nodes_in_memory = _a_star_nodes_in_memory(
        priority_queue,
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
            _a_star_nodes_in_memory(
                priority_queue,
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

def bidirectional_bfs(graph, start, goal, verbose=False, weighted_graph=None, data=None):
    """
    Bidirectional Breadth-First Search on an unweighted graph.

    The algorithm runs two BFS searches simultaneously:
    one forward from the start node and one backward from the goal node.
    When the two frontiers meet, the corresponding partial paths are joined.
    """
    if start not in graph:
        raise ValueError(f"Start node '{start}' is not present in the graph.")

    if goal not in graph:
        raise ValueError(f"Goal node '{goal}' is not present in the graph.")

    if start == goal:
        path = [start]
        metrics = {
            "expanded_nodes": 1,
            "max_frontier_size": 1,
            "max_nodes_in_memory": 1,
            "path_cost": compute_path_cost(path, weighted_graph)
        }
        return path, [start], metrics

    queue_start = deque([start])
    queue_goal = deque([goal])

    visited_start = {start}
    visited_goal = {goal}

    parent_start = {start: None}
    parent_goal = {goal: None}

    visited_order = []
    step = 0
    meeting_node = None

    expanded_nodes = 0
    max_frontier_size = len(queue_start) + len(queue_goal)
    max_nodes_in_memory = _bidirectional_bfs_nodes_in_memory(
        queue_start, queue_goal, visited_start, visited_goal, parent_start, parent_goal
    )

    if verbose:
        print(f"Start node: {start}")
        print(f"Goal node: {goal}")
        print(f"Initial forward frontier: {list(queue_start)}")
        print(f"Initial backward frontier: {list(queue_goal)}")

    while queue_start and queue_goal and meeting_node is None:
        if verbose:
            print(f"Step {step} (forward)")
            print(f"Forward frontier before extraction: {list(queue_start)}")
            print(f"Backward frontier before extraction: {list(queue_goal)}")

        meeting_node, expanded_in_layer, newly_added, expanded_layer_nodes = (
            _expand_bidirectional_layer(
                graph,
                queue_start,
                visited_start,
                visited_goal,
                parent_start,
                visited_order
            )
        )
        expanded_nodes += expanded_in_layer

        if verbose:
            print(f"Expanded nodes from start side: {expanded_layer_nodes}")
            print(f"Newly added forward nodes: {newly_added}")

        max_frontier_size = max(max_frontier_size, len(queue_start) + len(queue_goal))
        max_nodes_in_memory = max(
            max_nodes_in_memory,
            _bidirectional_bfs_nodes_in_memory(
                queue_start, queue_goal,
                visited_start, visited_goal,
                parent_start, parent_goal
            )
        )

        if meeting_node is not None:
            if verbose:
                print(f"Search frontiers met at node: {meeting_node}")
            break

        if verbose:
            print(f"Forward frontier after expansion: {list(queue_start)}")
            print(f"Backward frontier after expansion: {list(queue_goal)}")
            print(f"Visited order so far: {visited_order}")
            print(f"Step {step} (backward)")
            print(f"Forward frontier before extraction: {list(queue_start)}")
            print(f"Backward frontier before extraction: {list(queue_goal)}")

        meeting_node, expanded_in_layer, newly_added, expanded_layer_nodes = (
            _expand_bidirectional_layer(
                graph,
                queue_goal,
                visited_goal,
                visited_start,
                parent_goal,
                visited_order
            )
        )
        expanded_nodes += expanded_in_layer

        if verbose:
            print(f"Expanded nodes from goal side: {expanded_layer_nodes}")
            print(f"Newly added backward nodes: {newly_added}")

        max_frontier_size = max(max_frontier_size, len(queue_start) + len(queue_goal))
        max_nodes_in_memory = max(
            max_nodes_in_memory,
            _bidirectional_bfs_nodes_in_memory(
                queue_start, queue_goal,
                visited_start, visited_goal,
                parent_start, parent_goal
            )
        )

        if verbose:
            if meeting_node is not None:
                print(f"Search frontiers met at node: {meeting_node}")
            print(f"Forward frontier after expansion: {list(queue_start)}")
            print(f"Backward frontier after expansion: {list(queue_goal)}")
            print(f"Visited order so far: {visited_order}")

        step += 1

    if meeting_node is None:
        metrics = {
            "expanded_nodes": expanded_nodes,
            "max_frontier_size": max_frontier_size,
            "max_nodes_in_memory": max_nodes_in_memory,
            "path_cost": None
        }
        return None, visited_order, metrics

    path = _reconstruct_bidirectional_path(parent_start, parent_goal, meeting_node)

    metrics = {
        "expanded_nodes": expanded_nodes,
        "max_frontier_size": max_frontier_size,
        "max_nodes_in_memory": max_nodes_in_memory,
        "path_cost": compute_path_cost(path, weighted_graph)
    }

    return path, visited_order, metrics