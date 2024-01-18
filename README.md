# RRT* Path Planning Algorithm

## Overview
This repository contains a Python implementation of the RRT* (Rapidly-exploring Random Tree Star) algorithm for path planning in a 2D state space on Ubuntu 20. The algorithm is designed to find a collision-free path from a start point to a goal point considering circular obstacles in the environment.

## Getting Started 
1. Install Python 3:
Ubuntu 20.04 comes with Python 3 pre-installed, but you can ensure you have the latest version by running:

       sudo apt update
       sudo apt install python3
2. Install pip (Python package installer):

       sudo apt install python3-pip
3. Install NumPy and Matplotlib:
Use pip3 to install NumPy and Matplotlib:

       pip3 install numpy matplotlib

4. Test the installations:

       python3 -V
       pip3 show numpy
       pip3 show matplotlib

## Usage
Create a new wrokspace :

    mkdir RRTstar_ws
Go into that workspace :

    cd RRTstar_ws
Clone the repo :

    git clone <repository_url>
    cd RRT_star_Implementation
Run the RRTstar.py node :

    python rrt_star.py


## Input
Modify the following parameters in the rrt_star.py file:

• start_node: Specify the starting point (x, y) coordinates.
• goal_node: Specify the goal point (x, y) coordinates.
• x_range: Specify the range of x-axis values.
• y_range: Specify the range of y-axis values.
• obstacles: Specify the circular obstacles in the environment by providing their (x, y) coordinates.
• obstacle_radius: Specify the radius of the circular obstacles.

## Output 
The output will be a plot showing the start point, goal point, obstacles, and the path found by the RRT* algorithm.

## Example 

### Input

    start_node = Node(0, 0)
    goal_node = Node(5, 5)
    x_range = (-1, 6)
    y_range = (-1, 6)
    obstacle1 = Node(2, 2)
    obstacle2 = Node(3, 3)
    obstacle3 = Node(4, 4)
    obstacles = [obstacle1, obstacle2, obstacle3]
    obstacle_radius = 0.2
    
    # Run RRT* algorithm
    path = rrt_star(start_node, goal_node, x_range, y_range, obstacles)
    
    # Plotting
    plt.scatter(*start_node.x, color='green', marker='o', label='Start')
    plt.scatter(*goal_node.x, color='red', marker='o', label='Goal')
    plt.scatter(*zip(*[obstacle.x for obstacle in obstacles]), color='black', marker='x', label='Obstacle')
    plt.plot(*zip(*path), linestyle='-', marker='.', color='blue', label='Path')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('RRT* Algorithm')
    plt.show()

### Output
<img src="https://github.com/arjun-593/RRT_star_Implementation/blob/main/data/example_output.png" width="400" height="400" />

  ## License

  This project is licensed under the MIT License - see the [LICENSE](https://github.com/arjun-593/RRT_star_Implementation/blob/main/LICENSE) file for details.
