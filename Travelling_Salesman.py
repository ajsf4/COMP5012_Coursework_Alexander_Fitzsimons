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
width, height = 1200, 700
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Travelling Salesman Problem")

# Shader
shader = sh.Shader(width, height)



points = []
demands = []
with open("data//vrp8.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, d = line.split()
        points.append(np.array([float(x), float(y), 0]))
        demands.append(int(d))


optimiser_A = op.MultiObjectiveOptimiser(points, demands)
current_route_A = sp.Path(points, optimiser_A.route)
shader.scene.add_objects(current_route_A)
run_optimiser_A = False

optimiser_B = op.MultiObjectiveOptimiser(points, demands)
shifted_points = (np.array(points)+np.array([0, -60, 0])).tolist()
current_route_B = sp.Path(shifted_points, optimiser_B.route, default_colour=(255,0,255))
shader.scene.add_objects(current_route_B)
run_optimiser_B = False

shifted_points = (np.array(points)+np.array([60, 0, 0])).tolist()
optimiser_C = op.MultiObjectiveOptimiser(points, demands)
current_route_C = sp.Path(shifted_points, optimiser_C.route, default_colour=(255,255,0))
shader.scene.add_objects(current_route_C)
run_optimiser_C = False
# initial solutions needed to produce a pareto front that the new optimiser can use
optimiser_C.HillClimbSwapperOptimise()
current_route_C.update_route(optimiser_C.route)

shifted_points = (np.array(points)+np.array([60, -60, 0])).tolist()
optimiser_D = op.MultiObjectiveOptimiser(points, demands)
current_route_D = sp.Path(shifted_points, optimiser_D.route, default_colour=(255,0,0))
shader.scene.add_objects(current_route_D)
run_optimiser_D = False
# initial solutions needed to produce a pareto front that the new optimiser can use
optimiser_D.HillClimbSwapperOptimise()
current_route_D.update_route(optimiser_D.route)

# User Interface
graph_A = ui.Graph((150,150), "scatter")
graph_B = ui.Graph((150,150), "scatter")
graph_C = ui.Graph((150,150), "scatter")
graph_D = ui.Graph((150,150), "scatter")



graph_data_A = [[],[]]
graph_data_B = [[],[]]
graph_data_C = [[],[]]
graph_data_D = [[],[]]

# testing the efficeincy of an optimiser:
time_allowed_per_optimiser = 60
timer = 0

camControl = ct.controller()

running = True
speed = 8
dt = 0
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_LSHIFT:
                speed = 20
            elif event.key == pg.K_e:
                camControl.translation[1] = 1
            elif event.key == pg.K_q:
                camControl.translation[1] = -1
            elif event.key == pg.K_a:
                camControl.translation[2] = 1
            elif event.key == pg.K_d:
                camControl.translation[2] = -1
            elif event.key == pg.K_w:
                camControl.translation[0] = -1
            elif event.key == pg.K_s: 
                camControl.translation[0] = 1
            elif event.key == pg.K_UP:
                camControl.rotation[0] = 1
            elif event.key == pg.K_DOWN:
                camControl.rotation[0] = -1
            elif event.key == pg.K_LEFT:
                camControl.rotation[2] = -1
            elif event.key == pg.K_RIGHT:
                camControl.rotation[2] = 1
            elif event.key == pg.K_1:
                run_optimiser_A = not run_optimiser_A
            elif event.key == pg.K_2:
                run_optimiser_B = not run_optimiser_B
            elif event.key == pg.K_3:
                run_optimiser_C = not run_optimiser_C
            elif event.key == pg.K_4:
                run_optimiser_D = not run_optimiser_D
        elif event.type == pg.KEYUP:
            if event.key == pg.K_LSHIFT:
                speed = 8
            elif event.key == pg.K_e:
                camControl.translation[1] = 0
            elif event.key == pg.K_q:
                camControl.translation[1] = 0
            elif event.key == pg.K_a:
                camControl.translation[2] = 0
            elif event.key == pg.K_d:
                camControl.translation[2] = 0
            elif event.key == pg.K_w:
                camControl.translation[0] = 0
            elif event.key == pg.K_s: 
                camControl.translation[0] = 0
            elif event.key == pg.K_UP:
                camControl.rotation[0] = 0
            elif event.key == pg.K_DOWN:
                camControl.rotation[0] = 0
            elif event.key == pg.K_LEFT:
                camControl.rotation[2] = 0
            elif event.key == pg.K_RIGHT:
                camControl.rotation[2] = 0
    
    if timer < 60:
        optimiser_A.HillClimbSwapperWithExplorationOptimise()
        current_route_A.update_route(optimiser_A.route)

        graph_data_A[0] = optimiser_A.customer_satisfaction_history
        graph_data_A[1] = optimiser_A.distance_history
        graph_A.update_ui(np.array(graph_data_A), "customer satisfaction", "distance", pareto_front=optimiser_A.pareto_front)
    
    elif timer < 120:
        optimiser_B.HillClimbSwapperOptimise()
        current_route_B.update_route(optimiser_B.route)

        graph_data_B[0] = optimiser_B.customer_satisfaction_history
        graph_data_B[1] = optimiser_B.distance_history
        graph_B.update_ui(np.array(graph_data_B), "customer satisfaction", "distance", pareto_front=optimiser_B.pareto_front)
        
    elif timer < 180:    
        optimiser_C.MutateOnlyParetoFrontOptimise()
        current_route_C.update_route(optimiser_C.route)

        graph_data_C[0] = optimiser_C.customer_satisfaction_history
        graph_data_C[1] = optimiser_C.distance_history
        graph_C.update_ui(np.array(graph_data_C), "customer satisfaction", "distance", pareto_front=optimiser_C.pareto_front)#
    
    elif timer < 240:    
        optimiser_D.MutateParetoAndExploreOptimise()
        current_route_D.update_route(optimiser_D.route)

        graph_data_D[0] = optimiser_D.customer_satisfaction_history
        graph_data_D[1] = optimiser_D.distance_history
        graph_D.update_ui(np.array(graph_data_D), "customer satisfaction", "distance", pareto_front=optimiser_D.pareto_front)

    camControl.transform(shader.camera, speed, 0.7, dt)
    screen.blit(shader.rasterize(), (0, 0))

    screen.blit(graph_A.surface, (0,0))
    screen.blit(graph_B.surface, (0,graph_A.size[1]+5))
    screen.blit(graph_C.surface, (0,graph_A.size[1]+graph_B.size[1]+10))
    screen.blit(graph_D.surface, (0,graph_A.size[1]+graph_B.size[1]+graph_C.size[1]+15))

    pg.display.flip()
    dt = clock.tick(30) / 1000
    timer += dt


sys.exit()