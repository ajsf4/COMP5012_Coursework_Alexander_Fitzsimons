import numpy as np
# installed module imports
import pygame as pg
import sys
# custom module imports
import optimiser as op
import Render2D as r

# gets user input for number of generations to automatically run
try:
    starting_gens = int(input("Enter the number of generations to start with (e.g: 10):"))
except:
    print("invalid input. Starting with a single generation")
    starting_gens = 1

# pygame general initialisations
pg.init()
clock = pg.time.Clock()
width, height = 1200, 700
screen = pg.display.set_mode((width, height), pg.RESIZABLE)
pg.display.set_caption("Travelling Salesman Problem")
pg.display.set_icon(pg.image.load("icon.png"))

# loading data
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

# initialises the optimiser
optimiser = op.MOGA_PTSP(points, demands, 30)
# initialises the objective space graph
graph = r.GraphObj(np.array([0,0]), np.array(op.extract_objectives(optimiser.solution_history)), np.array([500,500]), "MOGA", "total distance", "remaining demand")

# initialises the camera for viewing the scene
camera = r.Camera2D((width, height))
camera.add_to_scene(graph)

# initialises the display maps
mapsOfPopulation = []
for y, solution_index in enumerate(optimiser.population):
    mapsOfSolution = []
    for x, route in enumerate(optimiser.solution_history[solution_index].solution):
        mapsOfSolution.append(r.MapObj(np.array((x*200, y*200 + 500)), np.array((200, 200)), (255, 0, 0), points, route, f"solution:{y}, period:{x}"))
        camera.add_to_scene(mapsOfSolution[-1])
    mapsOfPopulation.append(mapsOfSolution.copy())


# adds data to the graph
graph_data = np.array(op.extract_objectives(optimiser.solution_history))
graph.highlight_population = optimiser.population
graph.update(graph_data, optimiser.pareto_front)
graph.draw_surface()

# adding solutions to the display maps
for s, mapsOfSolution in enumerate(mapsOfPopulation):
    for r, routeMap in enumerate(mapsOfSolution):
        routeMap.update(optimiser.solution_history[optimiser.population[s]].solution[r])


# initialises fonts
loading_font = pg.font.SysFont("Consolas", 40)
instruction_font = pg.font.SysFont("Consolas", 20)

# test surface definitions
loading_text = loading_font.render("Loading next generation...", True, (255, 0, 0))
instruction1 = instruction_font.render("Press space to create the next generation", True, (255, 0 ,0))
instruction2 = instruction_font.render("Use the arrow keys to scroll around the screen", True, (255, 0, 0))
instruction2_5 = instruction_font.render("Press enter to run pareto distance optimisation", True, (255, 0, 0))
instruction3 = instruction_font.render("Press S to enter selection mode", True, (255, 0, 0))
instruction4 = instruction_font.render("Use arrow keys to choose a solution on the pareto front", True, (255, 0, 0))
instruction4_5 = instruction_font.render("Press enter to optimise the distance of selected solution", True, (255, 0, 0))
instruction5 = instruction_font.render("Press S to exit selection mode", True, (255,0 , 0))
instruction6 = instruction_font.render("Generation: 0", True, (255,0,0))

#keeps count of the number of MOGA generations
gen = 0
# delta time
dt = 0

# keeps track of user inputs
horr = 0
vert = 0
space = False
speed = 15
select = False
selected_solution = 0
optimise_selected = False
running = True

#main program loop
while running:
    #variable resets
    space = False
    select_arrows = 0
    optimise_selected = False

    #event handling loop
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        elif event.type == pg.VIDEORESIZE:
            width = screen.get_width()
            height = screen.get_height()
            camera.surface = pg.Surface((width, height))

        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                vert = 1
            elif event.key == pg.K_DOWN:
                vert = -1
            elif event.key == pg.K_LEFT:
                horr = 1
            elif event.key == pg.K_RIGHT:
                horr = -1

        elif event.type == pg.KEYUP:
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
                optimise_selected = True
    
    #view control
    arrow_vector = speed * np.array([horr, vert], dtype=np.float64) / np.linalg.norm(np.array([horr, vert])) if np.linalg.norm(np.array([horr, vert])) != 0 else np.array([0, 0])
    if not select:
        camera.position -= arrow_vector

    # runs a generation if the following condition
    if arrow_vector[0] == 0 and arrow_vector[1] == 0 and (space or starting_gens > 0) and not select:
        screen.blit(loading_text, (width - loading_text.get_width(), 120))
        gen +=1
        instruction6 = instruction_font.render(f"Generation: {gen}", True, (255,0,0))
        pg.display.flip()
        screen.fill((0, 0, 0))
        optimiser.optimise()
        graph_data = np.array(op.extract_objectives(optimiser.solution_history))
        graph.highlight_population = optimiser.population
        graph.update(graph_data, optimiser.pareto_front)
        graph.draw_surface()

        for s, mapsOfSolution in enumerate(mapsOfPopulation):
            for r, routeMap in enumerate(mapsOfSolution):
                routeMap.colour = np.array((255, 0, 0))
                routeMap.update(optimiser.solution_history[optimiser.population[s]].solution[r])

        if starting_gens > 0:
            starting_gens += -1

    if select and len(graph.pareto_front) > 0:

        if optimise_selected:
            optimiser.optimise_distance_of_selected_solution(graph.sorted_pareto_front[selected_solution])

            graph_data = np.array(op.extract_objectives(optimiser.solution_history))
            graph.update(graph_data, optimiser.pareto_front)

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
        if optimise_selected:
            screen.blit(loading_text, (width - loading_text.get_width(), 120))
            pg.display.flip()
            for i in optimiser.pareto_front:
                optimiser.optimise_distance_of_selected_solution(i)
            
            optimiser.pareto_front = op.findParetoFront(optimiser.solution_history)
            optimiser.population = op.findBestSolutions(optimiser.pareto_front, optimiser.population_size, optimiser.solution_history)
            optimiser.clusterer(0.02)
            if len(optimiser.population) < optimiser.population_size:
                number_of_solutions = optimiser.population_size - len(optimiser.population)
                optimiser.select_diverse_solutions_from_history(number_of_solutions)
            graph_data = np.array(op.extract_objectives(optimiser.solution_history))
            graph.update(graph_data, optimiser.pareto_front)
            graph.highlight_population = optimiser.population
        graph.draw_surface()                

    camera.render()
    screen.blit(camera.surface, (0,0))

    if not select:
        screen.blit(instruction1, (width-instruction1.get_width(), 0))
        screen.blit(instruction2, (width-instruction2.get_width(), 25))
        screen.blit(instruction2_5, (width-instruction2_5.get_width(), 50))
        screen.blit(instruction3, (width-instruction3.get_width(), 75))
        screen.blit(instruction6, (width-instruction6.get_width(), 100))
    else:
        screen.blit(instruction4, (width-instruction4.get_width(), 0))
        screen.blit(instruction4_5, (width-instruction4_5.get_width(), 25))
        screen.blit(instruction5, (width-instruction5.get_width(), 50))

    pg.display.flip()
    dt = clock.tick(30) / 1000


sys.exit()