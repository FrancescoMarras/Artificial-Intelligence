# Pathfinding Algorithms on a Sardinian City Graph

This repository contains a **university project** developed for the **Artificial Intelligence course**.  
The project was carried out by:

- **Francesco Marras (70/90/00651)**
- **Giulia Melis (70/90/00642)**

## Project Overview

The goal of this project is to study, implement, and compare different **search algorithms** for pathfinding on a graph of cities in **Sardinia**.

Each city is represented as a node, while roads between cities are represented as edges with associated distances.  
The project focuses on how different search strategies behave in terms of:

- Path found
- Total path cost
- Space and Time complexity

## Implemented Algorithms

The following algorithms are included in the project:

- **Breadth-First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **A\* Search**
- **Bidirectional Breadth-First Search**

These algorithms are compared on the same graph in order to evaluate their performance and behavior.

## Project Structure

```text
.
├── algorithm.py       # Implementation of search algorithms
├── utils.py           # Utility functions for loading data, plotting, and metrics
├── cities.json        # Graph data: cities, coordinates, and distances
├── main.ipynb         # Notebook used to run experiments and visualize results
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation