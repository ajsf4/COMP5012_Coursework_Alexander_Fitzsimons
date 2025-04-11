from attr import s
import numpy as np
import pygame as pg
import sys
import random

import optimiser as op


pg.init()
clock = pg.time.Clock()

# Screen
width, height = 1200, 700
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Travelling Salesman Problem")


points = []
demands = []
with open("data//vrp8.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, d = line.split()
        points.append(np.array([float(x), float(y), 0]))
        demands.append(int(d))


optimiser_A = op.MultiObjectiveOptimiser(points, demands)
run_optimiser_A = False

graph_data_A = [[],[]]


running = True
speed = 8
dt = 0
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    optimiser_A.HillClimbSwapperWithExplorationOptimise()

    pg.display.flip()
    dt = clock.tick(30) / 1000


sys.exit()