import numpy as np
import pygame as pg
import sys
import random

import optimiser as op
import Render2D as r


pg.init()
clock = pg.time.Clock()

# Screen
width, height = 1200, 700
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Travelling Salesman Problem")

points = []
demands = []
initial_connections = []
with open("data//vrp8.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, d = line.split()
        points.append(np.array([float(x), float(y)]))
        demands.append(int(d))
        initial_connections.append(int(n)-2)
points = np.array(points)

optimiser = op.MOGA_PTSP(points, demands, 10)
graph = r.GraphObj(np.array([0,0]), np.array([optimiser.demand_history, optimiser.distance_history]), np.array([300,300]), "random walk", "remaining demand", "total distance")
   
camera = r.Camera2D((width, height))
camera.add_to_scene(graph)

running = True
speed = 8

horr = 0
vert = 0

dt = 0
while running:
    screen.fill((0, 0, 0))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                vert = 1
            elif event.key == pg.K_DOWN:
                vert = -1
            elif event.key == pg.K_LEFT:
                horr = 1
            elif event.key == pg.K_RIGHT:
                horr = -1

        if event.type == pg.KEYUP:
            if event.key == pg.K_UP:
                vert = 0
            elif event.key == pg.K_DOWN:
                vert = 0
            elif event.key == pg.K_LEFT:
                horr = 0
            elif event.key == pg.K_RIGHT:
                horr = 0


    arrow_vector = speed * np.array([horr, vert], dtype=np.float64) / np.linalg.norm(np.array([horr, vert])) if np.linalg.norm(np.array([horr, vert])) != 0 else np.array([0, 0])
    camera.position -= arrow_vector

    if arrow_vector[0] == 0 and arrow_vector[1] == 0:
        optimiser.random_walk_optimise()
        graph.update(np.array([optimiser.demand_history, optimiser.distance_history]), optimiser.pareto_front)
        graph.draw_surface()
    camera.render()
    screen.blit(camera.surface, (0,0))

    pg.display.flip()
    dt = clock.tick(30) / 1000

sys.exit()