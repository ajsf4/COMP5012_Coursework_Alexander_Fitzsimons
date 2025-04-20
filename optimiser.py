import numpy as np


def findParetoFront(solution_history):
    pareto_front = []
    distances, demands = extract_objectives(solution_history)
    for i in range(len(solution_history)):
        i_is_better_than_these_di = distances[i] <= distances
        i_is_better_than_these_de = demands[i] <= demands

        bool_list = np.logical_or(i_is_better_than_these_de, i_is_better_than_these_di)
        if all(bool_list):
            pareto_front.append(i)

    return pareto_front

def extract_objectives(solution_history):
    distances = []
    demands = []
    for solution in solution_history:
        distances.append(solution.distance)
        demands.append(solution.demand)
    distances = np.array(distances)
    demands = np.array(demands)
    return distances, demands

def findBestSolutions(distances, demands, number_of_solutions):
    de = np.array(demands.copy())
    di = np.array(distances.copy())

    dominated_by = [] # a list of numbers indicating how many solutions dominate the solution at the given index
    for i in range(len(distances)):
        i_is_better_than_these_di = di[i] <= di
        i_is_better_than_these_de = de[i] <= de

        dominated_by.append(sum(np.logical_or(i_is_better_than_these_de, i_is_better_than_these_di)))

    dominated_by = np.array(dominated_by)
    best_solutions = np.array([])
    acceptance_level = 0 # how many solutions can dominate this solution to allow it to be accepted
    while len(best_solutions) < number_of_solutions:
        best_solutions = np.append(best_solutions, np.where(dominated_by==acceptance_level)[0])
        acceptance_level+=1
    best_solutions = best_solutions[0:number_of_solutions]
    return best_solutions.astype(int).tolist()

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
        self.population = []
        for i in range(self.population_size):
            solution = [self.generate_random_route() for j in range(self.periods)]
            self.population.append(Solution(i, solution, self.all_distances, self.starting_demand))
        self.solution_history = self.population.copy()

        self.combined_population = []
        self.current_population = []
        self.pareto_front = []

    def generate_random_route(self):
        route_length = np.random.randint(1, len(self.points))

        default_route = np.arange(1, len(self.points))
        np.random.shuffle(default_route)

        random_route = np.append(self.depot, default_route[0:route_length]).tolist()
        # when evaluating each route, the distance between the last and first points is accounted for
        # this means we only need to state the depot at the start of the route and not the end#

        return random_route

    def random_walk_optimise(self):
        #generates random routes and adds it to the history to see if any solutions are better by chance
        self.population = []
        for i in range(self.population_size):
            solution = [self.generate_random_route() for j in range(self.periods)]
            new_solution = Solution(len(self.solution_history), solution, self.all_distances, self.starting_demand)
            self.population.append(new_solution)
            self.solution_history.append(new_solution)
        
        self.pareto_front = findParetoFront(self.solution_history)

    """
    def mutate(self, mutation_rate):
        new_population = self.population.copy()
        for i_s, solution in enumerate(self.population):
            for i_r, route in enumerate(solution):
                for mutation in range(mutation_rate):
                    point_to_mutate = np.random.randint(1, len(self.points)) # assumes depot point is zero
                    if point_to_mutate in route:
                        new_population[i_s][i_r].remove(point_to_mutate)
                    else:
                        new_population[i_s][i_r].insert(np.random.randint(1, len(new_population[i_s][i_r])), point_to_mutate) # also assume depot is zero
        return new_population 

    def genetic_optimiser(self):
        new_population = self.mutate(3)

        # combine the new with the old
        self.combined_population = self.population.copy()
        for solution in new_population:
            self.combined_population.append(solution)
        new_pop_distance, new_pop_demand = self.evaluate_objectives(new_population)
        combined_pop_distance = np.append(self.pop_distance, new_pop_distance)
        combined_pop_demand = np.append(self.pop_demand, new_pop_demand)

        # eliminate the worst solutions
        solutions = findBestSolutions(combined_pop_distance, combined_pop_distance, self.population_size)      
        self.current_population = solutions.copy()
        self.population = [self.combined_population[i] for i in solutions]

        # add the new to the history
        for solution in new_population:
            self.solution_history.append(solution)
        
        self.demand_history = np.append(self.demand_history, new_pop_demand).tolist()
        self.distance_history = np.append(self.distance_history, new_pop_distance).tolist()

        self.pareto_front = findParetoFront(self.distance_history, self.demand_history)
    """

# a class which keeps track of each solution and their objective values
class Solution:
    def __init__(self, solution_number, solution, distance_array, demands):
        self.solution_number = solution_number
        self.solution = solution
        self.demands = demands.copy()
        self.distance, self.demand = self.evaluate_objectives(distance_array)

    
    def evaluate_objectives(self, all_distances):
        total_distance = 0
        for route in self.solution:
            for point in route:
                total_distance += all_distances[point][point-1]
                if self.demands[point] > 0:# if there is no demand in a visited node, then there is no reduction in the total demand
                    self.demands[point] -= 1

        return total_distance, sum(self.demands)


