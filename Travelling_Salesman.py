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
    
camera = r.Camera2D((width, height))


ant_optimiser_variants = []
moga_optimiser = op.MultiObjectiveGeneticAlgorithm(points, demands, 10)
moga_optimiser.next_generation()
moga_graph = r.GraphObj(np.array([0, 330]), np.array((moga_optimiser.cSat_history, moga_optimiser.dist_history)), (330, 300), "MOGA Optimiser", "X Axis", "Y Axis")
camera.add_to_scene(moga_graph)

Map_Objects = []
Graph_Objects = []
for i in range(0, 11):
    # Genetic Algorithm
    Map_Objects.append(r.MapObj(np.array([330*i, 0]), np.array((75, 75)), (255, 0, 0), points, initial_connections))
    camera.add_to_scene(Map_Objects[-1])


    
    # Ant Colony Optimiser
    Map_Objects.append(r.MapObj(np.array([330*i, 660]), np.array((75, 75)), (255, 255, 0), points, initial_connections))
    camera.add_to_scene(Map_Objects[-1])
    
    ant_optimiser_variants.append(op.MultiObjectiveAntColonyOptimiser(points, demands, 10))
    ant_optimiser_variants[-1].objective_preference = i/10
    ant_optimiser_variants[-1].optimise()

    Graph_Objects.append(r.GraphObj(np.array([330*i, 990]), np.array([ant_optimiser_variants[-1].customer_satisfaction_history, ant_optimiser_variants[-1].distance_history]), (330, 300), "Ant Colony Optimiser", "X Axis", "Y Axis"))
    camera.add_to_scene(Graph_Objects[-1])
 

running = True
speed = 8

# input managers
horr = 0
vert = 0

dt = 0
count = 0
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

        moga_optimiser.next_generation()
        moga_graph.update([moga_optimiser.cSat_history,
                           moga_optimiser.dist_history], 
                           moga_optimiser.pareto_front)

        moga_graph.draw_surface()

        for i in range(0, 10):
            # genetic algorithm
            Map_Objects[i].update(moga_optimiser.population[i])

            # ant colony optimiser
            ant_optimiser_variants[i].optimise()
            Map_Objects[i*2].update(ant_optimiser_variants[i].route)
            Graph_Objects[i].update([ant_optimiser_variants[i].customer_satisfaction_history,
                                    ant_optimiser_variants[i].distance_history],
                                    ant_optimiser_variants[i].pareto_front)
            Graph_Objects[i].draw_surface() 


    
    
    
    camera.render()
    screen.blit(camera.surface, (0,0))

    pg.display.flip()
    dt = clock.tick(30) / 1000
    count += 1

sys.exit()