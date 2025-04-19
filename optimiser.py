import numpy as np


def findParetoFront(distance_history, demand_history):
    pareto_front = []
    de = np.array(demand_history.copy())
    di = np.array(distance_history.copy())
    for i in range(len(distance_history)):

        i_is_better_than_these_di = di[i] <= di
        i_is_better_than_these_de = de[i] <= de

        if all(np.logical_or(i_is_better_than_these_de, i_is_better_than_these_di)):
            pareto_front.append(i)

    return pareto_front

class MOGA_PTSP:
    def __init__(self, points, demands, population):
        self.points = points
        self.starting_demand = demands
        self.population_size = population
        self.depot = 0 #the index of the starting point
        self.all_distances = [[ np.linalg.norm(point_a - point_b) for point_b in self.points ] for point_a in self.points ]

        # number of periods is equal to the maximum demand as this allows every node to be visited at least that many times within the number of period
        # in this variation a node doesn't have to be visited in every period, but each node has a demand that is reduced by 1 every time it is visited in a period
        # this continues until the end of the number of periods and the remaining demand is summed up
        # the sum total demand at the end should be minimized as well as the total distance travelled       
        self.periods = max(demands)
        
        # population consists of a series of solutions which will be mutated each generation to try and optimise the objectives
        # each solution has n routes where n is the number of periods
        # each route consists of a series of indexes referring to the self.points list indicating the movement from one node to the next
        self.population = [[self.generate_random_route() for j in range(self.periods)] for i in range(self.population_size)]
        self.pop_distance, self.pop_demand = self.evaluate_objectives()

        self.solution_history = self.population.copy()
        self.demand_history = self.pop_demand.copy()
        self.distance_history = self.pop_distance.copy()

        self.pareto_front = []

    def generate_random_route(self):
        route_length = np.random.randint(1, len(self.points))

        default_route = np.arange(1, len(self.points))
        np.random.shuffle(default_route)

        random_route = np.append(self.depot, default_route[0:route_length])
        # when evaluating each route, the distance between the last and first points is accounted for
        # this means we only need to state the depot at the start of the route and not the end#

        return random_route

    def evaluate_objectives(self):
        #we need to evaluate the total distances travelled and remaining demands throughout each set of periods for every solution in the population
        population_distances = []
        population_demands = []
        for solution in self.population:
            solution_distance = 0
            current_demands = self.starting_demand.copy()
            for route in solution:
                for point in route:
                    if current_demands[point] > 0: # if there is no demand in a visited node, then there is no reduction in the total demand
                        current_demands[point] -= 1
                    solution_distance += self.all_distances[point][point-1]
            solution_demand = sum(current_demands)

            #solution_demand/distance is the respective objective values of each solution
            population_demands.append(solution_demand)
            population_distances.append(solution_distance)

        return population_distances, population_demands

    def random_walk_optimise(self):
        #generates random routes and adds it to the history to see if any solutions are better by chance
        self.population = [[self.generate_random_route() for j in range(self.periods)] for i in range(self.population_size)]
        self.pop_distance, self.pop_demand = self.evaluate_objectives()

        for solution in self.population:
            self.solution_history.append(solution)

        self.demand_history = np.append(self.demand_history, self.pop_demand).tolist()
        self.distance_history = np.append(self.distance_history, self.pop_distance).tolist()
        
        self.pareto_front = findParetoFront(self.distance_history, self.demand_history)


