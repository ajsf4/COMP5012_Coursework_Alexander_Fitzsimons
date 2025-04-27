from matplotlib.cbook import safe_first_element
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
graph = r.GraphObj(np.array([0,0]), np.array(op.extract_objectives(optimiser.solution_history)), np.array([500,500]), "MOGA", "total distance", "remaining demand")

camera = r.Camera2D((width, height))
camera.add_to_scene(graph)

loading_font = pg.font.SysFont("Consolas", 40)
loading_text = loading_font.render("Loading next generation...", True, (255, 0, 0))

mapsOfPopulation = []
for y, solution_index in enumerate(optimiser.population):
    mapsOfSolution = []
    for x, route in enumerate(optimiser.solution_history[solution_index].solution):
        mapsOfSolution.append(r.MapObj(np.array((x*200, y*200 + 500)), np.array((200, 200)), (255, 0, 0), points, route, f"solution:{y}, period:{x}"))
        camera.add_to_scene(mapsOfSolution[-1])
    mapsOfPopulation.append(mapsOfSolution.copy())


running = True
speed = 8
select = False
selected_solution = 0

horr = 0
vert = 0
space = False

screen.fill((0, 0, 0))
graph_data = np.array(op.extract_objectives(optimiser.solution_history))
graph.highlight_population = optimiser.population
graph.update(graph_data, optimiser.pareto_front)
graph.draw_surface()

for s, mapsOfSolution in enumerate(mapsOfPopulation):
    for r, routeMap in enumerate(mapsOfSolution):
        routeMap.update(optimiser.solution_history[optimiser.population[s]].solution[r])

instruction_font = pg.font.SysFont("Consolas", 20)
instruction1 = instruction_font.render("Press space to create the next generation", True, (255,0,0))
instruction2 = instruction_font.render("Use the arrow keys to scroll around the screen", True, (255, 0,0))
instruction3 = instruction_font.render("Press S to enter selection mode", True, (255,0,0))
instruction4 = instruction_font.render("Use arrow keys to choose a solution on the pareto front", True, (255,0,0))
instruction5 = instruction_font.render("Press S to exit selection mode", True, (255,0,0))

dt = 0
while running:
    space = False
    select_arrows = 0
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
                select_arrows = -1
            elif event.key == pg.K_RIGHT:
                horr = 0
                select_arrows = 1
            elif event.key == pg.K_SPACE:
                space = True
            elif event.key == pg.K_s:
                select = not select

            elif event.key == pg.K_RETURN:
                # debug:

                for a, solution_a in enumerate(optimiser.solution_history):
                    for b, solution_b in enumerate(optimiser.solution_history):
                        if solution_a.solution == solution_b.solution:
                            if a!=b:
                                if solution_b.demand != solution_a.demand:
                                    print(solution_b.demand, solution_a.demand)



    arrow_vector = speed * np.array([horr, vert], dtype=np.float64) / np.linalg.norm(np.array([horr, vert])) if np.linalg.norm(np.array([horr, vert])) != 0 else np.array([0, 0])
    if not select:
        camera.position -= arrow_vector

    if arrow_vector[0] == 0 and arrow_vector[1] == 0 and space and not select:
        screen.blit(loading_text, (width - loading_text.get_width(), 70))
        pg.display.flip()
        screen.fill((0, 0, 0))
        optimiser.genetic_optimiser()
        graph_data = np.array(op.extract_objectives(optimiser.solution_history))
        graph.highlight_population = optimiser.population
        graph.update(graph_data, optimiser.pareto_front)
        graph.draw_surface()

        for s, mapsOfSolution in enumerate(mapsOfPopulation):
            for r, routeMap in enumerate(mapsOfSolution):
                routeMap.colour = np.array((255, 0, 0))
                routeMap.update(optimiser.solution_history[optimiser.population[s]].solution[r])

    if select and len(graph.pareto_front) > 0:
        if select_arrows != 0:
            selected_solution += select_arrows
        if selected_solution < 0:
            selected_solution = len(graph.pareto_front)-1
        if selected_solution >= len(graph.pareto_front):
            selected_solution = 0
        
        graph.draw_surface()
        graph.select_solution(selected_solution)

        for r, routeMap in enumerate(mapsOfPopulation[0]):
            routeMap.colour = np.array((0, 0, 255))
            routeMap.update(optimiser.solution_history[graph.sorted_pareto_front[selected_solution]].solution[r])

    else:
        graph.draw_surface()                

    camera.render()
    screen.blit(camera.surface, (0,0))

    if not select:
        screen.blit(instruction1, (width-instruction1.get_width(), 0))
        screen.blit(instruction2, (width-instruction2.get_width(), 25))
        screen.blit(instruction3, (width-instruction3.get_width(), 50))
    else:
        screen.blit(instruction4, (width-instruction4.get_width(), 0))
        screen.blit(instruction5, (width-instruction5.get_width(), 25))

    pg.display.flip()
    dt = clock.tick(30) / 1000


sys.exit()