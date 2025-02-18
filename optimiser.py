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
    def __init__(self, points, numberOfAnts):
        self.points = points
        self.N = numberOfAnts



