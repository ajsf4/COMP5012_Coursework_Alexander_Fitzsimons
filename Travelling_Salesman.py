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

"""
map_A = r.MapObj(np.array([0, 0]), np.array((75, 75)), (255, 0, 0), points, initial_connections)
camera.add_to_scene(map_A)
optimiser_A = op.MultiObjectiveOptimiser(points, demands)
graph_A = r.graph(np.array([0, 330]), np.array([optimiser_A.customer_satisfaction_history,optimiser_A.distance_history]), (330, 300), "Graph A", "X Axis", "Y Axis")
camera.add_to_scene(graph_A)

map_B = r.MapObj(np.array([330, 0]), np.array((75, 75)), (0, 255, 0), points, initial_connections)
camera.add_to_scene(map_B)
optimiser_B = op.MultiObjectiveOptimiser(points, demands)
graph_B = r.graph(np.array([330, 330]), np.array([optimiser_B.customer_satisfaction_history,optimiser_B.distance_history]), (330, 300), "Graph B", "X Axis", "Y Axis")
camera.add_to_scene(graph_B)

map_C = r.MapObj(np.array([0, 330]), np.array((75, 75)), (0, 0, 255), points, initial_connections)
camera.add_to_scene(map_C)
optimiser_C = op.MultiObjectiveOptimiser(points, demands)
optimiser_C.HillClimbSwapperOptimise()

map_D = r.MapObj(np.array([330, 330]), np.array((75, 75)), (255, 0, 255), points, initial_connections)
camera.add_to_scene(map_D)
optimiser_D = op.MultiObjectiveOptimiser(points, demands)
optimiser_D.HillClimbSwapperOptimise()
"""

optimiser_E_variants = []
Map_Objects = []
Graph_Objects = []
for i in range(0, 11):
    Map_Objects.append(r.MapObj(np.array([330*i, 0]), np.array((75, 75)), (255, 255, 0), points, initial_connections))
    camera.add_to_scene(Map_Objects[-1])
    
    optimiser_E_variants.append(op.MultiObjectiveOptimiser(points, demands, percentile=(i+1)*5))
    
    Graph_Objects.append(r.graph(np.array([330*i, 330]), np.array([optimiser_E_variants[-1].customer_satisfaction_history,optimiser_E_variants[-1].distance_history]), (330, 300), "Graph E", "X Axis", "Y Axis"))
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

        """
        optimiser_A.HillClimbSwapperOptimise()
        map_A.update(optimiser_A.route)
    
        optimiser_B.HillClimbSwapperWithExplorationOptimise()
        map_B.update(optimiser_B.route)
        
        optimiser_C.MutateOnlyParetoFrontOptimise()
        map_C.update(optimiser_C.route)
    
        optimiser_D.MutateParetoAndExploreOptimise()
        map_D.update(optimiser_D.route)
        """
        for i in range(0, 11):
            optimiser_E_variants[i].ReducerOptimiser()
            Map_Objects[i].update(optimiser_E_variants[i].route)

            Graph_Objects[i].update(np.array([optimiser_E_variants[i].customer_satisfaction_history, optimiser_E_variants[i].distance_history]))
            Graph_Objects[i].draw_surface() 

    # update graphs
    """
    graph_A.update(np.array([optimiser_A.customer_satisfaction_history, optimiser_A.distance_history]))
    graph_A.draw_surface() 

    graph_B.update(np.array([optimiser_B.customer_satisfaction_history, optimiser_B.distance_history]))
    graph_B.draw_surface() 
    """

    
    
    
    camera.render()
    screen.blit(camera.surface, (0,0))

    pg.display.flip()
    dt = clock.tick(30) / 1000
    count += 1

sys.exit()