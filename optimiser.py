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
        print("optimising")
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

        print(distance, self.best_distance)

        
class FerromoneOptimiser:
    def __init__(self, points, colony_size):
        self.points = points
        self.colony_size = colony_size
        self.ant_current_positions = []
        self.ant_distances = [0 for i in range(colony_size)]
        self.ant_available_positions = [[j for j in range(len(points))] for i in range(colony_size)]
        self.global_ferromones = np.zeros((len(points), len(points)))
        self.ant_ferromones = np.zeros((colony_size, len(points), len(points)))
        self.select_starting_positions()

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
            for i, pos in enumerate(self.ant_available_positions[ant]):
                # add weight depending on the distance travelled
                dist = self.calculate_distance(self.points[pos], self.points[ant_position])
                
                # add weight from ferromones
                ferromone = self.global_ferromones[ant_position][pos]

                # ferromone and distance weightings
                weight = ferromone + (1/dist)
                
                weights.append(weight)
                lengths.append(dist)

            # normalise to a probability ditribution
            total = sum(weights)
            p = np.array(weights)/total
            # ant makes his decision
            selected_point = np.random.choice([i for i in range(len(p))], 1, p=p)[0]
            new_pos = self.ant_available_positions[ant][selected_point]
            new_positions.append(new_pos)
            self.ant_distances[ant] += lengths[selected_point]
            self.ant_ferromones[ant][ant_position][new_pos] = 1/lengths[selected_point]
        self.ant_current_positions = new_positions.copy()

    def traverse_network(self):
        for i in range(len(self.points)-1):
            self.select_new_positions()
        ant_route_score = 1 / np.array(self.ant_distances)
        normalised_ant_route_score = ant_route_score / np.sum(ant_route_score)
        for ant, score in enumerate(normalised_ant_route_score):
            self.global_ferromones += score * self.ant_ferromones[ant]
            
                


# test optimiser
points = []

with open("data//vrp_test.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        n, x, y, z = line.split()
        points.append(np.array([float(x), float(y), float(z)]))

colony = FerromoneOptimiser(points, 1)
colony.traverse_network()


