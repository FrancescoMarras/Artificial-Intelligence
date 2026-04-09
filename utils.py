import json
import matplotlib.pyplot as plt


def load_map_from_json(json_path):
    """
    Load a graph-based map from a JSON file.
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
    Plot only the city positions based on their coordinates.
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
    """
    return None if value is None else round(value, digits)


def reconstruct_path(parent, goal):
    """
    Reconstruct a path from the parent dictionary produced by a search.
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
    """
    R = 6371.0  # Earth's radius in km
    lon1, lat1 = data[city_1]["x"], data[city_1]["y"]
    lon2, lat2 = data[city_2]["x"], data[city_2]["y"]

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * R * math.asin(math.sqrt(a))

    return c