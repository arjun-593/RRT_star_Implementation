#!/usr/bin/env python

# rrtstar.py
# This program generates an asymptotically optimal rapidly exploring random tree (RRT*)in a rectangular region.
# Modified for visualization using pygame

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







