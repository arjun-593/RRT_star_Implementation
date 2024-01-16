'''#!/usr/bin/env python

# rrtstar.py
# This program generates a 
# asymptotically optimal rapidly exploring random tree (RRT* proposed by Sertac Keraman, MIT) in a rectangular region.
#
# Originally written by Steve LaValle, UIUC for simple RRT in
# May 2011
# Modified by Md Mahbubur Rahman, FIU for RRT* in
# January 2016

import sys, random, math, pygame
from pygame.locals import *
from math import sqrt,cos,sin,atan2

#constants
XDIM = 640
YDIM = 480
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000
RADIUS=15

def dist(p1,p2):
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def step_from_to(p1,p2):
    if dist(p1,p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
        return p1[0] + EPSILON*cos(theta), p1[1] + EPSILON*sin(theta)

def chooseParent(nn,newnode,nodes):
 	for p in nodes:
	   if dist([p.x,p.y],[newnode.x,newnode.y]) <RADIUS and p.cost+dist([p.x,p.y],[newnode.x,newnode.y]) < nn.cost+dist([nn.x,nn.y],[newnode.x,newnode.y]):
	      nn = p
        newnode.cost=nn.cost+dist([nn.x,nn.y],[newnode.x,newnode.y])
	newnode.parent=nn
        return newnode,nn

def reWire(nodes,newnode,pygame,screen):
 	white = 255, 240, 200
        black = 20, 20, 40
	for i in xrange(len(nodes)):
    	   p = nodes[i]
	   if p!=newnode.parent and dist([p.x,p.y],[newnode.x,newnode.y]) <RADIUS and newnode.cost+dist([p.x,p.y],[newnode.x,newnode.y]) < p.cost:
	      pygame.draw.line(screen,white,[p.x,p.y],[p.parent.x,p.parent.y])  
	      p.parent = newnode
              p.cost=newnode.cost+dist([p.x,p.y],[newnode.x,newnode.y]) 
              nodes[i]=p  
              pygame.draw.line(screen,black,[p.x,p.y],[newnode.x,newnode.y])                    
	return nodes

def drawSolutionPath(start,goal,nodes,pygame,screen):
	pink = 200, 20, 240
	nn = nodes[0]
	for p in nodes:
	   if dist([p.x,p.y],[goal.x,goal.y]) < dist([nn.x,nn.y],[goal.x,goal.y]):
	      nn = p
	while nn!=start:
		pygame.draw.line(screen,pink,[nn.x,nn.y],[nn.parent.x,nn.parent.y],5)  
		nn=nn.parent



class Node:
    x = 0
    y = 0
    cost=0  
    parent=None
    def __init__(self,xcoord, ycoord):
         self.x = xcoord
         self.y = ycoord
	
def main():
    #initialize and prepare screen
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(white)

    nodes = []
    
    #nodes.append(Node(XDIM/2.0,YDIM/2.0)) # Start in the center
    nodes.append(Node(0.0,0.0)) # Start in the corner
    start=nodes[0]
    goal=Node(630.0,470.0)
    for i in range(NUMNODES):
	rand = Node(random.random()*XDIM, random.random()*YDIM)
	nn = nodes[0]
        for p in nodes:
	   if dist([p.x,p.y],[rand.x,rand.y]) < dist([nn.x,nn.y],[rand.x,rand.y]):
	      nn = p
        interpolatedNode= step_from_to([nn.x,nn.y],[rand.x,rand.y])
	
	newnode = Node(interpolatedNode[0],interpolatedNode[1])
 	[newnode,nn]=chooseParent(nn,newnode,nodes);
       
	nodes.append(newnode)
	pygame.draw.line(screen,black,[nn.x,nn.y],[newnode.x,newnode.y])
        nodes=reWire(nodes,newnode,pygame,screen)
        pygame.display.update()
        #print i, "    ", nodes

        for e in pygame.event.get():
	   if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
	      sys.exit("Leaving because you requested it.")
    drawSolutionPath(start,goal,nodes,pygame,screen)
    pygame.display.update()
# if python says run, then we should run
if __name__ == '__main__':
    main()
    running = True
    while running:
       for event in pygame.event.get():
	    if event.type == pygame.QUIT:
               running = False'''
               
               
               
               
'''              
#!/usr/bin/env python

# rrtstar.py
# This program generates an asymptotically optimal rapidly exploring random tree (RRT* proposed by Sertac Keraman, MIT) in a rectangular region.
#
# Originally written by Steve LaValle, UIUC for simple RRT in May 2011
# Modified by Md Mahbubur Rahman, FIU for RRT* in January 2016
# Modified again for visualization using pygame

import sys
import random
import math
import pygame
from pygame.locals import *
from math import sqrt, cos, sin, atan2

# Constants
XDIM = 640
YDIM = 480
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 5000
RADIUS = 15

# Node class definition
class Node:
    x = 0
    y = 0
    cost = 0
    parent = None

    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord

# Euclidean distance function
def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Stepping from one node to another
def step_from_to(p1, p2):
    if dist(p1, p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
        return p1[0] + EPSILON * cos(theta), p1[1] + EPSILON * sin(theta)

# Choosing parent node based on cost
def chooseParent(nn, newnode, nodes):
    for p in nodes:
        if dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and p.cost + dist([p.x, p.y], [newnode.x, newnode.y]) < nn.cost + dist(
                [nn.x, nn.y], [newnode.x, newnode.y]):
            nn = p
    newnode.cost = nn.cost + dist([nn.x, nn.y], [newnode.x, newnode.y])
    newnode.parent = nn
    return newnode, nn

# Rewiring the tree
def reWire(nodes, newnode, pygame, screen):
    white = 255, 240, 200
    black = 20, 20, 40
    for i in range(len(nodes)):
        p = nodes[i]
        if p != newnode.parent and dist([p.x, p.y], [newnode.x, newnode.y]) < RADIUS and newnode.cost + dist(
                [p.x, p.y], [newnode.x, newnode.y]) < p.cost:
            pygame.draw.line(screen, white, [p.x, p.y], [p.parent.x, p.parent.y])
            p.parent = newnode
            p.cost = newnode.cost + dist([p.x, p.y], [newnode.x, newnode.y])
            nodes[i] = p
            pygame.draw.line(screen, black, [p.x, p.y], [newnode.x, newnode.y])
    return nodes

# Drawing the solution path
def drawSolutionPath(start, goal, nodes, pygame, screen):
    pink = 200, 20, 240
    nn = nodes[0]
    for p in nodes:
        if dist([p.x, p.y], [goal.x, goal.y]) < dist([nn.x, nn.y], [goal.x, goal.y]):
            nn = p
    while nn != start:
        pygame.draw.line(screen, pink, [nn.x, nn.y], [nn.parent.x, nn.parent.y], 5)
        nn = nn.parent


def main():
    # Initialize and prepare screen
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(white)

    nodes = []

    nodes.append(Node(0.0, 0.0))  # Start in the corner
    start = nodes[0]
    goal = Node(630.0, 470.0)
    for i in range(NUMNODES):
        rand = Node(random.random() * XDIM, random.random() * YDIM)
        nn = nodes[0]
        for p in nodes:
            if dist([p.x, p.y], [rand.x, rand.y]) < dist([nn.x, nn.y], [rand.x, rand.y]):
                nn = p
        interpolatedNode = step_from_to([nn.x, nn.y], [rand.x, rand.y])

        newnode = Node(interpolatedNode[0], interpolatedNode[1])
        [newnode, nn] = chooseParent(nn, newnode, nodes)

        nodes.append(newnode)
        pygame.draw.line(screen, black, [nn.x, nn.y], [newnode.x, newnode.y])
        nodes = reWire(nodes, newnode, pygame, screen)
        pygame.display.update()

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
    drawSolutionPath(start, goal, nodes, pygame, screen)
    pygame.display.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == '__main__':
    main()
'''





import numpy as np
import matplotlib.pyplot as plt

# Node class definition
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# Euclidean distance function
def euclidean_distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

# Collision checking function (modify as needed)
def is_collision_free(new_node, obstacles, obstacle_radius):
    for obstacle in obstacles:
        distance = euclidean_distance(new_node, obstacle)
        if distance < obstacle_radius:
            return False  # Collision
    return True  # Collision-free

# Steer from one node towards another within a maximum distance
def steer(from_node, to_node, max_distance):
    distance = euclidean_distance(from_node, to_node)
    if distance < max_distance:
        return to_node
    else:
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + max_distance * np.cos(theta)
        new_y = from_node.y + max_distance * np.sin(theta)
        return Node(new_x, new_y)

# Rewiring the tree to update the parent of nodes if a shorter path is found
def rewire(tree, new_node, max_distance, obstacles, obstacle_radius):
    for node in tree:
        if node == new_node or not is_collision_free(node, obstacles, obstacle_radius):
            continue
        new_cost = node.cost + euclidean_distance(node, new_node)
        if new_cost < new_node.cost and euclidean_distance(node, new_node) < max_distance:
            new_node.parent = node
            new_node.cost = new_cost

# RRT* algorithm
def rrt_star(start, goal, x_range, y_range, obstacles, max_iter=1000, max_distance=0.4, obstacle_radius=0.2):
    tree = [start]
    for _ in range(max_iter):
        random_node = Node(np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1]))
        nearest_node = min(tree, key=lambda node: euclidean_distance(node, random_node))
        new_node = steer(nearest_node, random_node, max_distance)

        if is_collision_free(new_node, obstacles, obstacle_radius):
            near_nodes = [node for node in tree if euclidean_distance(node, new_node) < max_distance]
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + euclidean_distance(nearest_node, new_node)

            for near_node in near_nodes:
                if near_node.cost + euclidean_distance(near_node, new_node) < new_node.cost and is_collision_free(new_node, obstacles, obstacle_radius):
                    new_node.cost = near_node.cost + euclidean_distance(near_node, new_node)
                    new_node.parent = near_node

            tree.append(new_node)
            rewire(tree, new_node, max_distance, obstacles, obstacle_radius)

        # Check if the new node is close to the goal, break if true
        if euclidean_distance(new_node, goal) < max_distance:
            break

    # Backtrack to find the path
    path = []
    current_node = tree[-1]
    while current_node is not None:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    path.reverse()
    return path


# Set up the start and goal nodes, state space, and obstacles
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
obstacle_radius = 0.2  # Radius of circular obstacles

# Run RRT* algorithm
path = rrt_star(start_node, goal_node, x_range, y_range, obstacles)

# Plotting
plt.scatter(start_node.x, start_node.y, color='green', marker='o', label='Start')
plt.scatter(goal_node.x, goal_node.y, color='red', marker='o', label='Goal')
plt.scatter(*zip(*[(obstacle.x, obstacle.y) for obstacle in obstacles]), color='black', marker='x', label='Obstacle')
plt.plot(*zip(*path), linestyle='-', marker='.', color='blue', label='Path')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('RRT* Algorithm')
plt.show()

