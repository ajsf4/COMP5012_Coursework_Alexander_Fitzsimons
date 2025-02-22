import numpy as np


class SwapperOptimiser:
    def __init__(self, points, route):
        self.points = points
        self.route = route
        self.route_history = []
        self.best_route = self.route
        self.best_distance = self.calculate_distance()

    def calculate_distance(self):
        distance = 0
        for i in range(len(self.route)):
            distance += np.linalg.norm(self.points[self.route[i]] - self.points[self.route[i-1]])
        return distance

    def swap(self, i, j):
        self.route[i], self.route[j] = self.route[j], self.route[i]

    def optimise(self):
        self.route_history.append(self.route.copy())
        route_tested = True
        i=0
        j=0
        while route_tested:
            i = np.random.randint(0, len(self.route))
            j = np.random.randint(0, len(self.route))
            
            self.swap(i, j)
            if self.route not in self.route_history:
                route_tested = False
            else:
                self.swap(i, j)
        distance = self.calculate_distance()
        if distance < self.best_distance:
            self.best_distance = distance
            self.best_route = self.route
        else:
            self.swap(i, j)
        
class FerromoneOptimiser:
    def __init__(self, points, colony_size):
        self.points = points
        self.colony_size = colony_size
        self.global_ferromones = np.ones((len(self.points), len(self.points)))
        self.ant_ferromones = np.zeros((self.colony_size, len(self.points), len(self.points)))
        self.new_generation()
        self.dist_pow = 2
        self.ferr_pow = 2
        self.best_routes = []
        self.best_scores = []
        self.iterations = 0


    def new_generation(self):
        
        self.ant_current_positions = []
        self.ant_distances = [0 for i in range(self.colony_size)]
        self.ant_available_positions = [[j for j in range(len(self.points))] for i in range(self.colony_size)]
        self.select_starting_positions()
        self.ant_routes = [[self.ant_current_positions[i]] for i in range(self.colony_size)]

    def select_starting_positions(self):
        for ant in range(self.colony_size):
            point_position = np.random.randint(0, len(self.points))
            self.ant_current_positions.append(point_position)

    def calculate_distance(self, point1, point2):
        distance = np.linalg.norm(point1 - point2)
        return distance

    def select_new_positions(self):
        new_positions = []
        for ant, ant_position in enumerate(self.ant_current_positions):
            self.ant_available_positions[ant].remove(ant_position)
            weights = []
            lengths = []
            if len(self.ant_available_positions[ant]) > 0:
                for i, pos in enumerate(self.ant_available_positions[ant]):
                    dist = self.calculate_distance(self.points[pos], self.points[ant_position])
                    ferromone = self.global_ferromones[ant_position][pos]
                    weight = ((1/dist) ** self.dist_pow) * (ferromone ** self.ferr_pow)
                    weights.append(weight)
                    lengths.append(dist)
                # normalise to a probability ditribution
                total = sum(weights)
                p = np.array(weights)/total
                # ant makes his decision
                selected_point = np.random.choice([i for i in range(len(p))], 1, p=p)[0]
                new_pos = self.ant_available_positions[ant][selected_point]
            else:
                selected_point = 0
                new_pos = self.ant_routes[ant][selected_point]
                lengths = [self.calculate_distance(self.points[new_pos], self.points[ant_position])]
            new_positions.append(new_pos)
            self.ant_distances[ant] += lengths[selected_point]
            self.ant_ferromones[ant][ant_position][new_pos] = 1
            self.ant_routes[ant].append(new_pos)
        self.ant_current_positions = new_positions.copy()

    def traverse_network(self):
        for i in range(len(self.points)):
            self.select_new_positions()
        ant_route_scores = 1 / np.array(self.ant_distances)
        normalised_ant_route_scores = ant_route_scores / np.sum(ant_route_scores)
        self.best_scores.append(max(ant_route_scores))
        self.best_routes.append(self.ant_routes[ant_route_scores.tolist().index(max(ant_route_scores))])
        self.new_generation()
        for ant, score in enumerate(normalised_ant_route_scores):
            self.global_ferromones += score * self.ant_ferromones[ant]
        self.ant_ferromones = np.zeros((self.colony_size, len(self.points), len(self.points)))
        self.global_ferromones /= np.amax(self.global_ferromones)
        self.iterations += 1
        
            
                

"""
# test optimiser
points = []

with open("data//vrp_test.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, z = line.split()
        points.append(np.array([float(x), float(y), float(z)]))

colony = FerromoneOptimiser(points, 1)
colony.traverse_network()
"""
