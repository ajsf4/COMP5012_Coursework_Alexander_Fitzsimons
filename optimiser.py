from random import shuffle
import numpy as np

# sets the random seed meaning the same inputs should yield the same results every time
# comment out or change the seed parameter to see different results
np.random.seed(1)

def findParetoFront(solution_history):
    distances, demands = extract_objectives(solution_history)
    num = len(distances)
    dominated = np.zeros(num, dtype=bool)
    
    for i in range(num):
        # Compare i against all j
        dominated_by_i = (distances <= distances[i]) & (demands <= demands[i]) 
        dominated_by_i &= ((distances < distances[i]) | (demands < demands[i]))
        dominated[i] = np.any(dominated_by_i)
    
    pareto_front = np.where(np.bitwise_not(dominated))[0].tolist()
    return pareto_front

def extract_objectives(solution_history):
    distances = []
    demands = []
    for solution in solution_history:
        distances.append(solution.distance+0)
        demands.append(solution.demand+0)
    distances = np.array(distances)
    demands = np.array(demands)
    return distances, demands

def normalise(value, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def findBestSolutions(solution_indexes, number_of_solutions, solution_history):
    history_distances, history_demands = extract_objectives(solution_history)
    distances = [history_distances[i] for i in solution_indexes]
    demands = [history_demands[i] for i in solution_indexes]

    if len(solution_indexes) < number_of_solutions:
        number_of_solutions = len(solution_indexes)

    dominated_by = []
    for i in range(len(solution_indexes)):
        i_is_better_than_these_di = distances[i] < history_distances
        i_is_better_than_these_de = demands[i] < history_demands

        #indicates for each solution in the list, how many other solutions dominate it
        dominated_by.append(sum(np.logical_not(np.logical_or(i_is_better_than_these_de, i_is_better_than_these_di))))

    dominated_by = np.array(dominated_by)


    let_through = []
    threshold = 0
    while len(let_through) < number_of_solutions:
        for c, i in enumerate(dominated_by):
            if i == threshold:
                let_through.append(solution_indexes[c])
            if len(let_through) == number_of_solutions:
                break
        threshold += 1

    return let_through


# Main Multi objective genetic algorithm class (also contains the single objective distance optimiser)
class MOGA_PTSP:
    def __init__(self, points, demands, population):
        self.points = points
        self.starting_demands = demands
        self.population_size = population
        self.depot = 0 #the index of the starting point
        self.all_distances = [[ np.linalg.norm(point_a - point_b) for point_b in self.points ] for point_a in self.points ]

        # number of periods is equal to the maximum demand as this allows every node to be visited at least that many times within the number of period
        # in this variation a node doesn't have to be visited in every period, but each node has a demand that is reduced by 1 every time it is visited in a period
        # this continues until the end of the number of periods and the remaining demand is summed up  
        self.periods = 28

        
        # population consists of a series of solutions which will be mutated each generation to try and optimise the objectives
        # each solution has n routes where n is the number of periods
        # each route consists of a series of indexes referring to the self.points list indicating the movement from one node to the next
        self.population = []
        self.solution_history = []
        for i in range(self.population_size-1):
            solution = [self.generate_random_route() for j in range(self.periods)]
            self.solution_history.append(Solution(solution, self.all_distances, self.starting_demands))
            self.solution_history[-1].evaluate_objectives(self.all_distances)
            self.population.append(i)

        # add in the null solution of empty routes:
        null_solution = [[self.depot] for i in range(self.periods)]
        self.solution_history.append(Solution(null_solution, self.all_distances, self.starting_demands))
        self.population.append(len(self.solution_history) - 1)

        self.pareto_front = []

        self.generations = 0

    def generate_random_route(self):
        route_length = np.random.randint(1, len(self.points))

        default_route = np.arange(1, len(self.points))
        np.random.shuffle(default_route)

        random_route = np.append(self.depot, default_route[0:route_length]).tolist()
        # when evaluating each route, the distance between the last and first points is accounted for
        # this means we only need to state the depot at the start of the route and not the end#

        return random_route
    
    def remove_distance_heuristic(self, route):
        route_distance_scores = []
        for i in range(1, len(route)):
            p1 = route[i-1]
            p2 = route[i]
            if i < len(route)-1:
                p3 = route[i+1]
            else:
                p3 = route[0]

            # a point has two distances associated with it corresponding to the connections to the points before and after it
            route_distance_scores.append(self.all_distances[p1][p2] + self.all_distances[p2][p3])
        # get weights:
        weights = np.array(route_distance_scores)
        normalised = (weights / np.sum(weights)).tolist()
        return np.random.choice(route[1:], 1, normalised)[0] # p distribution biased towards high distances as we want to remove those

    def add_distance_heuristic(self, route):
        not_in_route = []
        for i in range(len(self.points)):
            if i not in route:
                not_in_route.append(i)
        
        new_points_and_positions = [(new_point, position) for new_point in not_in_route for position in range(len(route))]        
        indexes = [i for i in range(len(new_points_and_positions))]
        scores = []
        for i in range(len(route)):
            p1 = route[i-1]
            p3 = route[i]
            for p2 in not_in_route:
                scores.append(1/(self.all_distances[p1][p2] + self.all_distances[p2][p3]))
                # score is 1/distance as lower distances are preferred
        weights = np.array(scores)
        normalised = (weights / np.sum(weights)).tolist()
        new_point, position = new_points_and_positions[np.random.choice(indexes, 1, normalised)[0]]
        if position == 0:
            position = len(route)

        return new_point, position
                
    def mutate(self, mutation_rate):
        new_population = []
        
        for s, solution_index in enumerate(self.population):
            # determine removal probability by the solution in the population:
            prob = s / len(self.population)
            original = self.solution_history[solution_index].solution
            altered_solution = [route.copy() for route in original]
            for r, route in enumerate(original):
                for _ in range(mutation_rate):
                    if np.random.rand() < prob and len(altered_solution[r]) > 1:
                        # remove points with bad distance scores:
                        point_to_mutate = self.remove_distance_heuristic(altered_solution[r])
                        altered_solution[r].remove(point_to_mutate)
                    elif len(route) < len(self.points): # 70% chance to add
                        new_point, position = self.add_distance_heuristic(route)
                        altered_solution[r].insert(position, new_point) # also assume depot is zero

            self.solution_history.append(Solution(altered_solution, self.all_distances, self.starting_demands))
            self.solution_history[-1].evaluate_objectives(self.all_distances)
            new_population.append(len(self.solution_history)-1)
        return new_population 

    def crossover_routes(self, crossover_rate):

        shuffleSolutions = self.population.copy()
        np.random.shuffle(shuffleSolutions)
        number_of_parents = int(len(shuffleSolutions)/2)
        
        dominant_parents = shuffleSolutions[0:number_of_parents]
        recessive_parents = shuffleSolutions[-number_of_parents:]

        for dom_sol, rec_sol in zip(dominant_parents, recessive_parents):
            new_solution = []
            for dom_route, rec_route in zip(self.solution_history[dom_sol].solution, self.solution_history[rec_sol].solution):
                # crossover should be set to > 0.5 in order for dominant to be actually dominant
                # if it is less it shouldn't matter too much but variable labels would be the wrong way around
                if np.random.random() < crossover_rate:
                    new_solution.append(dom_route)
                else:
                    new_solution.append(rec_route)
            self.solution_history.append(Solution(new_solution, self.all_distances, self.starting_demands))
            self.population.append(len(self.solution_history)-1)


        #for solution in self.population:

    def optimise(self):
        self.generations+=1

        self.crossover_routes(0.95)

        new_populations = []
        for i in range(1, 10):
            new_populations.append(self.mutate(4))
        # combine the new mutated solutions with the old population
        for population in new_populations:
            for solution_index in population:
                self.population.append(solution_index)

        # eliminate the worst solutions:
        self.population = findBestSolutions(self.population, self.population_size, self.solution_history)

        # apply clusterer to reduce solutions which are too crowded:
        self.clusterer(0.02)

        if len(self.population) < self.population_size:
            number_of_solutions = self.population_size - len(self.population)
            self.select_diverse_solutions_from_history(number_of_solutions)



        # calculate pareto front from the entire history
        self.pareto_front = findParetoFront(self.solution_history)

    def clusterer(self, min_dist_threshold):
        #removes points from the current population that are too close together to increase diversity when eliminating bad points
        distances, demands = extract_objectives(self.solution_history)
        distances = [distances[i] for i in self.population]
        demands = [demands[i] for i in self.population]

        min_dist, max_dist = np.min(distances), np.max(distances)
        min_demand, max_demand = np.min(demands), np.max(demands)
        remaining = []
        selected_indices = []

        for i in range(len(self.population)):
            is_too_close = False
            # first in population is always added to remaining as selected indices is empty
            # subsequent solutions are compared against any that are allowed through to see if they are too close
            # if they are too close to anything then they do not get added
            for j in selected_indices:
                #normalisation of objectives to ensure balanced comparison
                pos1 = np.array([normalise(distances[i], min_dist, max_dist),
                                 normalise(demands[i], min_demand, max_demand)])
                pos2 = np.array([normalise(distances[j], min_dist, max_dist),
                                 normalise(demands[j], min_demand, max_demand)])

                objective_distance = np.linalg.norm(pos1 - pos2)
                if objective_distance < min_dist_threshold:
                    is_too_close = True
                    break
            if not is_too_close:
                selected_indices.append(i)
                remaining.append(self.population[i])
        self.population = remaining

    def select_diverse_solutions_from_history(self, num_sols, computation_cutoff=200):
        distances, demands = extract_objectives(self.solution_history)
        
        effective_cutoff = min(len(self.solution_history), computation_cutoff)
        distances = distances[-effective_cutoff:]
        demands = demands[-effective_cutoff:]

        distances = normalise(np.array(distances), np.min(distances), np.max(distances)).tolist()
        demands = normalise(np.array(demands), np.min(demands), np.max(demands)).tolist()
  
        solution_sparsity = []
        for dist1, dmnd1 in zip(distances, demands):
            objective_summed_distance = np.array([0, 0], dtype=np.float64)
            for dist2, dmnd2 in zip(distances, demands):
                if not(dist1==dist2 and dmnd1==dmnd2):
                    objective_summed_distance += np.array([abs(dist1-dist2), abs(dmnd1-dmnd2)])
            solution_sparsity.append(np.linalg.norm(objective_summed_distance))


        weighting = (solution_sparsity / np.sum(solution_sparsity)).tolist()

        offset = max(0, len(self.solution_history) - effective_cutoff) # re-indexing required
        to_append = np.random.choice(range(effective_cutoff), num_sols, replace=False, p=weighting)
        for i in to_append:
            self.population.append(i+offset) # use this to ensure that index is consistent with solution history
      
    def dist_of_route(self, route):
        summation = 0
        for i in range(len(route)):
            summation += self.all_distances[route[i]][route[i-1]]
        return summation

    def optimise_distance_of_selected_solution(self, selected_solution, dont_update=False):
        solution = [route.copy() for route in  self.solution_history[selected_solution].solution] 

        for route in solution:
            if len(route) >= 2:
                for iteration in range(50):

                    a = np.random.randint(1, len(route))
                    b = np.random.randint(1, len(route))
                    if a!=b:
                        original_dist = self.dist_of_route(route)
                        route[a], route[b] = route[b], route[a]
                        new_dist = self.dist_of_route(route)

                        if new_dist > original_dist:
                            route[a], route[b] = route[b], route[a]

        self.solution_history.append(Solution(solution, self.all_distances, self.starting_demands))
        #new population is all the new found solution which will all get mutated when there is another generation
        self.population = [len(self.solution_history)-1 for i in range(self.population_size)]

        if not dont_update:
            self.pareto_front = findParetoFront(self.solution_history)
                

# a class which keeps track of each solution and their objective values
class Solution:
    def __init__(self, solution, distance_array, demands):
        self.solution = [route.copy() for route in solution]      # deep copy routes
        self.initial_demands  = demands.copy()  
        self.first_solution = solution.copy()
        self.demands = demands.copy()
        self.distance, self.demand = self.evaluate_objectives(distance_array)
       
    def evaluate_objectives(self, all_distances):
        demands = self.initial_demands.copy()
        total_distance = 0
        for route in self.solution:
            for p, point in enumerate(route):
                prev_point = route[p-1] if p > 0 else route[-1]
                total_distance += all_distances[prev_point][point]
                if demands[point] > 0:# if there is no demand in a visited node, then there is no reduction in the total demand
                    demands[point] -= 1

        return total_distance, sum(demands)

