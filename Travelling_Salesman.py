from attr import s
import numpy as np
import pygame as pg
import sys
import random

import shader as sh
import controller as ct
import shapes as sp
import optimiser as op
import ui

pg.init()

clock = pg.time.Clock()

# Screen
width, height = 800, 600

screen = pg.display.set_mode((width, height))

pg.display.set_caption("Travelling Salesman Problem")

# Shader
shader = sh.Shader(width, height)

# User Interface
graph1 = ui.Graph((200,200), "scatter")

points = []
demands = []

with open("data//vrp9.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, d = line.split()
        points.append(np.array([float(x), float(y), 0]))
        demands.append(int(d))



"""
optimiser = op.FerromoneOptimiser(points, 5)
ferromone_network = sp.WeightedNetwork(points, optimiser.global_ferromones)
best_route = sp.Path(points, route)
shader.scene.add_objects(ferromone_network)
shader.scene.add_objects(best_route)
"""

optimiser = op.MultiObjectiveOptimiser(points, demands)
current_route = sp.Path(points, optimiser.route)
shader.scene.add_objects(current_route)
run_optimiser = False

graph_data = [[],[]]


camControl = ct.controller()

running = True
speed = 8
dt = 0
while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LSHIFT:
                speed = 20
            if event.key == pg.K_e:
                camControl.translation[1] = 1
            if event.key == pg.K_q:
                camControl.translation[1] = -1
            if event.key == pg.K_a:
                camControl.translation[2] = 1
            if event.key == pg.K_d:
                camControl.translation[2] = -1
            if event.key == pg.K_w:
                camControl.translation[0] = -1
            if event.key == pg.K_s: 
                camControl.translation[0] = 1
            if event.key == pg.K_UP:
                camControl.rotation[0] = 1
            if event.key == pg.K_DOWN:
                camControl.rotation[0] = -1
            if event.key == pg.K_LEFT:
                camControl.rotation[2] = -1
            if event.key == pg.K_RIGHT:
                camControl.rotation[2] = 1
            if event.key == pg.K_SPACE:
                run_optimiser = not run_optimiser

        if event.type == pg.KEYUP:
            if event.key == pg.K_LSHIFT:
                speed = 8
            if event.key == pg.K_e:
                camControl.translation[1] = 0
            if event.key == pg.K_q:
                camControl.translation[1] = 0
            if event.key == pg.K_a:
                camControl.translation[2] = 0
            if event.key == pg.K_d:
                camControl.translation[2] = 0
            if event.key == pg.K_w:
                camControl.translation[0] = 0
            if event.key == pg.K_s: 
                camControl.translation[0] = 0
            if event.key == pg.K_UP:
                camControl.rotation[0] = 0
            if event.key == pg.K_DOWN:
                camControl.rotation[0] = 0
            if event.key == pg.K_LEFT:
                camControl.rotation[2] = 0
            if event.key == pg.K_RIGHT:
                camControl.rotation[2] = 0
    
    if run_optimiser:
        optimiser.HillClimbSwapperWithExplorationOptimise()
        current_route.update_route(optimiser.route)
        graph_data[0] = optimiser.customer_satisfaction_history
        graph_data[1] = optimiser.distance_history

        graph1.update_ui(np.array(graph_data), "customer satisfaction", "distance", pareto_front=optimiser.pareto_front)
        
    camControl.transform(shader.camera, speed, 0.7, dt)
    screen.blit(shader.rasterize(), (0, 0))

    screen.blit(graph1.surface, (0,0))
    pg.display.flip()
    dt = clock.tick(30) / 1000

sys.exit()