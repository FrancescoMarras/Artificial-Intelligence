import json
import matplotlib.pyplot as plt


def load_map_from_json(json_path):
    """
    Load a graph-based map from a JSON file.

    The JSON file is expected to contain, for each city:
    - its 2D coordinates (`x`, `y`)
    - its neighboring cities together with the edge distances

    This function returns:
    1. the full raw data structure, useful for plotting and heuristics
    2. an unweighted adjacency-list graph, useful for BFS and DFS

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing:
        - data: the full JSON data
        - graph: a simplified adjacency-list representation
                 in the form {city: [neighbor1, neighbor2, ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = {
        city: list(info["neighbours"].keys())
        for city, info in data.items()
    }

    return data, graph


def load_weighted_graph_from_json(json_path):
    """
    Load a weighted graph from a JSON file.

    Unlike `load_map_from_json`, this function preserves the edge weights
    (distances between neighboring cities). This representation is useful
    for informed or cost-based search algorithms such as Uniform Cost Search
    or A*.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing:
        - data: the full JSON data
        - weighted_graph: a weighted adjacency representation in the form
                          {city: {neighbor1: distance1, neighbor2: distance2, ...}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weighted_graph = {
        city: info["neighbours"]
        for city, info in data.items()
    }

    return data, weighted_graph


def draw_cities(data, show_names=True):
    """
    Plot only the city positions based on their 2D coordinates.

    Each city is represented as a point in the Cartesian plane.
    This function is useful for visually inspecting the spatial distribution
    of the nodes before plotting the graph connections.

    Parameters
    ----------
    data : dict
        Dictionary containing city information, including coordinates.
    show_names : bool, optional
        If True, city names are displayed next to the points.
        Default is True.

    Returns
    -------
    None
        The function displays a matplotlib figure.
    """
    plt.figure(figsize=(12, 8))

    for city, info in data.items():
        x = info["x"]
        y = info["y"]
        plt.scatter(x, y)

        if show_names:
            plt.text(x, y, city, fontsize=8)

    plt.title("City positions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def draw_graph(data, show_names=True):
    """
    Plot the full graph, including nodes and edges.

    The function first draws the edges between neighboring cities and then
    draws the city nodes. To avoid drawing the same undirected edge twice,
    a set of already drawn edges is maintained.

    Parameters
    ----------
    data : dict
        Dictionary containing city coordinates and neighbors.
    show_names : bool, optional
        If True, city names are displayed next to the nodes.
        Default is True.

    Returns
    -------
    None
        The function displays a matplotlib figure.
    """
    plt.figure(figsize=(12, 8))

    drawn_edges = set()

    for city, info in data.items():
        x1, y1 = info["x"], info["y"]

        for neighbor in info["neighbours"]:
            # Since the graph is undirected, the edge (A, B) is equivalent
            # to the edge (B, A). Sorting the pair ensures a unique form.
            edge = tuple(sorted((city, neighbor)))

            if edge not in drawn_edges:
                x2 = data[neighbor]["x"]
                y2 = data[neighbor]["y"]
                plt.plot([x1, x2], [y1, y2], linewidth=0.8)
                drawn_edges.add(edge)

    for city, info in data.items():
        x = info["x"]
        y = info["y"]
        plt.scatter(x, y)

        if show_names:
            plt.text(x, y, city, fontsize=8)

    plt.title("City graph")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def draw_path(data, path, show_names=True):
    """
    Plot the graph and highlight a specific path.

    The full graph is first drawn in the background with lighter edges.
    Then the given path is highlighted using thicker lines. This is useful
    to visually display the solution returned by a search algorithm.

    Parameters
    ----------
    data : dict
        Dictionary containing city coordinates and neighbors.
    path : list[str] or None
        Sequence of cities representing the path to highlight.
        If None, no path is highlighted.
    show_names : bool, optional
        If True, city names are displayed next to the nodes.
        Default is True.

    Returns
    -------
    None
        The function displays a matplotlib figure.
    """
    plt.figure(figsize=(12, 8))

    drawn_edges = set()

    # Draw the full graph in the background.
    for city, info in data.items():
        x1, y1 = info["x"], info["y"]

        for neighbor in info["neighbours"]:
            edge = tuple(sorted((city, neighbor)))

            if edge not in drawn_edges:
                x2 = data[neighbor]["x"]
                y2 = data[neighbor]["y"]
                plt.plot([x1, x2], [y1, y2], alpha=0.3, linewidth=1)
                drawn_edges.add(edge)

    # Draw all nodes.
    for city, info in data.items():
        x = info["x"]
        y = info["y"]
        plt.scatter(x, y)

        if show_names:
            plt.text(x, y, city, fontsize=8)

    # Highlight the final path, if available.
    if path is not None and len(path) > 1:
        for i in range(len(path) - 1):
            city_1 = path[i]
            city_2 = path[i + 1]

            x1, y1 = data[city_1]["x"], data[city_1]["y"]
            x2, y2 = data[city_2]["x"], data[city_2]["y"]

            plt.plot([x1, x2], [y1, y2], linewidth=3)

    # Emphasize the start and goal nodes.
    if path is not None and len(path) > 0:
        start = path[0]
        goal = path[-1]

        plt.scatter(data[start]["x"], data[start]["y"], s=120)
        plt.scatter(data[goal]["x"], data[goal]["y"], s=120)

    plt.title("Highlighted path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def round_cost(value, digits=2):
    """
    Round a path cost to a fixed number of decimal digits.

    This helper is used to present path costs in a cleaner and more readable
    way inside the project output. If the value is None, the function returns
    None unchanged.

    Parameters
    ----------
    value : float or None
        Numeric value to round.
    digits : int, optional
        Number of decimal digits to keep. Default is 2.

    Returns
    -------
    float or None
        Rounded numeric value, or None if the input was None.
    """
    return None if value is None else round(value, digits)


def reconstruct_path(parent, goal):
    """
    Reconstruct a path from the parent dictionary produced by a search.

    The function starts from the goal node and walks backward through the
    parent links until the start node is reached. The resulting sequence is
    then reversed so that the path is returned from start to goal.

    Parameters
    ----------
    parent : dict[str, str | None]
        Dictionary mapping each discovered node to its predecessor in the path.
    goal : str
        Target node from which the reconstruction starts.

    Returns
    -------
    list[str]
        Ordered list of nodes representing the path from start to goal.
    """
    path = []
    node = goal

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()
    return path


def compute_path_cost(path, weighted_graph, digits=2):
    """
    Compute the total cost of a path on a weighted graph.

    The path cost is obtained by summing the weights of all consecutive edges
    in the path. If the path is None, or if no weighted graph is provided,
    the function returns None.

    Parameters
    ----------
    path : list[str] or None
        Sequence of nodes representing a path.
    weighted_graph : dict[str, dict[str, float]] or None
        Weighted adjacency representation of the graph.
    digits : int, optional
        Number of decimal digits used to round the final cost. Default is 2.

    Returns
    -------
    float or None
        Total rounded path cost, or None if it cannot be computed.
    """
    if path is None or weighted_graph is None:
        return None

    total = 0.0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        total += weighted_graph[current_node][next_node]

    return round_cost(total, digits)

import math

def heuristic(city_1, city_2, data):
    """
    Compute the straight-line distance in kilometers between two cities
    using their longitude/latitude coordinates and the Haversine formula.

    Parameters
    ----------
    city_1 : str
        Name of the first city.
    city_2 : str
        Name of the second city.
    data : dict
        Full map data loaded from JSON, including coordinates:
        x = longitude, y = latitude.

    Returns
    -------
    float
        Great-circle distance between the two cities in kilometers.
    """
    lon1, lat1 = data[city_1]["x"], data[city_1]["y"]
    lon2, lat2 = data[city_2]["x"], data[city_2]["y"]

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    R = 6371.0  # Earth's radius in km
    return R * c