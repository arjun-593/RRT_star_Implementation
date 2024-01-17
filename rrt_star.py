# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:59:54 2024

@author: arjun

The following is an implementation of the RRT* algorithm.
"""


import numpy as np
import matplotlib.pyplot as plt

# Node class definition
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


def node_2_node_distance(node1, node2):
# This function [Assumes nodes are in 2D plane] defenition is to compute the euclidean distance between two nodes.
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def check_collision_free(new_node, obstacles, obstacle_radius):
# This is a collision checking function. By default it assumes True, ie. no collision.
    for obstacle in obstacles:
        distance = node_2_node_distance(new_node, obstacle)
        if distance < obstacle_radius:
            return False  # Collision
    return True  # Default -> collision free

def move_node_2_node(from_node, to_node, max_distance):
# Moves between nodes within a maximum distance.
    distance = node_2_node_distance(from_node, to_node)
    if distance < max_distance:
        return to_node
    else:
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + max_distance * np.cos(theta)
        new_y = from_node.y + max_distance * np.sin(theta)
        return Node(new_x, new_y)

def rewire_tree(tree, new_node, max_distance, obstacles, obstacle_radius):
# Function that rewires the tree to update the parent of nodes if a shorter path is found.
    for node in tree:
        if node == new_node or not check_collision_free(node, obstacles, obstacle_radius):
            continue
        new_cost = node.cost + node_2_node_distance(node, new_node)
        if new_cost < new_node.cost and node_2_node_distance(node, new_node) < max_distance:
            new_node.parent = node
            new_node.cost = new_cost

# Main RRT* algorithm
def rrt_star(start, goal, x_range, y_range, obstacles, max_iter=1000, max_distance=0.4, obstacle_radius=0.2):
    tree = [start]
    for _ in range(max_iter):
        random_node = Node(np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1]))
        nearest_node = min(tree, key=lambda node: node_2_node_distance(node, random_node))
        new_node = move_node_2_node(nearest_node, random_node, max_distance)

        if check_collision_free(new_node, obstacles, obstacle_radius):
            near_nodes = [node for node in tree if node_2_node_distance(node, new_node) < max_distance]
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + node_2_node_distance(nearest_node, new_node)

            for near_node in near_nodes:
                if near_node.cost + node_2_node_distance(near_node, new_node) < new_node.cost and check_collision_free(new_node, obstacles, obstacle_radius):
                    new_node.cost = near_node.cost + node_2_node_distance(near_node, new_node)
                    new_node.parent = near_node

            tree.append(new_node)
            rewire_tree(tree, new_node, max_distance, obstacles, obstacle_radius)

        # Checking if the new node is close to the goal. -> breaks the loop if true.
        if node_2_node_distance(new_node, goal) < max_distance:
            break

    # Backtrack to find the path
    path = []
    current_node = tree[-1]
    while current_node is not None:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    path.reverse()
    return path


# Here I have set up the start and goal nodes, state space, obstacles and radius of obstacle(assumed circular).
start_node = Node(0, 0)
goal_node = Node(5, 5)
x_range = (-1, 6)
y_range = (-1, 6)
obstacle1 = Node(1, 1)
obstacle2 = Node(2, 0.5)
obstacle3 = Node(2, 2)
obstacle4 = Node(3, 4)
obstacle5 = Node(3, 0)
obstacle6 = Node(4, 1)
obstacle7 = Node(3, 3)
obstacle8 = Node(1.5, 3)
obstacle9 = Node(4, 4)
obstacle10 = Node(0, 1)
obstacle11 = Node(1.3, 2)
obstacle12 = Node(2.5, 1.3)
obstacle13 = Node(3.5, 1.5)
obstacle14 = Node(4, 2)
obstacle15 = Node(4.5, 3)
obstacle16 = Node(5, 4)
obstacles = [obstacle1, obstacle2, obstacle3, obstacle4, obstacle5, obstacle6, obstacle7, obstacle8, obstacle9, obstacle10, obstacle11, obstacle12, obstacle13, obstacle14, obstacle15, obstacle16]
obstacle_radius = 0.2

# Running the RRT* algorithm.
path = rrt_star(start_node, goal_node, x_range, y_range, obstacles)

# Plotting results for Visualization.
plt.scatter(start_node.x, start_node.y, color='green', marker='o', label='Start')
plt.scatter(goal_node.x, goal_node.y, color='red', marker='o', label='Goal')
plt.scatter(*zip(*[(obstacle.x, obstacle.y) for obstacle in obstacles]), color='black', marker='x', label='Obstacle')
plt.plot(*zip(*path), linestyle='-', marker='.', color='blue', label='Path')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('RRT* Algorithm')
plt.show()