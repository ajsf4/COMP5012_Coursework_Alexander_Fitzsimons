import numpy as np


def findParetoFront(route_history, distance_history, customer_satisfaction_history):
    pareto_front = []
    c = np.array(customer_satisfaction_history.copy())
    d = np.array(distance_history.copy())
    for i in range(len(route_history)):

        i_is_better_than_these_d = d[i] <= d
        i_is_better_than_these_c = c[i] >= c

        if all(np.logical_or(i_is_better_than_these_c, i_is_better_than_these_d)):
            pareto_front.append(i)

    return pareto_front

# this is the base of the exponent used for calculating customer satisfaction
exp_base = 1.1
def calculateCustomerSatisfaction(time_taken, demand):
    # formula determined by requiring that zero demand has maximum satisfaction no matter the time taken ...
    #  ... and that zero time taken has maximum satisfaction no matter the demand ...
    #  ... and that for non zero demand and non zero time taken, satisfaction decreases exponentially according to a given time taken ...
    #  ... and satisfaction decreases exponentially as time taken increases according to a given demand.
    exponent = (- time_taken * demand)/10000
    cs = np.power(exp_base, exponent)
    return cs
        

# a simple optimiser that also accounts for the demand of each city 
class MultiObjectiveGeneticAlgorithm:
    def __init__(self, points, demands, population_size):
        self.points = np.array(points)
        self.demands = np.array(demands)
        self.route = np.arange(len(points))

        self.population_size = population_size
        self.population = [self.route.copy() for i in range(population_size)]
        
        
        self.all_distances = [[np.linalg.norm(i - j) for j in self.points] for i in self.points]
        
        #cSat is short for customer satisfaction, an objective to be maximised while dist is short for distance which should be minimised
        dist, cSat, dists, cSats = self.calculate_objectives(self.route)
        self.route_history = [self.route.copy()]
        self.dist_history = [dist]
        self.cSat_history = [cSat]

        self.dists_for_population = []
        self.cSats_for_population = []
        for route in self.population:
            np.random.shuffle(route)
            self.route_history.append(route.copy())
            dist, cSat, dists, cSats = self.calculate_objectives(route)
            self.dist_history.append(dist)
            self.cSat_history.append(cSat)
            self.dists_for_population.append(dists)
            self.cSats_for_population.append(cSats)

        self.pareto_front = []

    def calculate_objectives(self, route):
        dists = []
        cSats = []
        total_dist = 0
        total_cSat = 0
        for i in range(len(route)):
            cSats.append(calculateCustomerSatisfaction(total_dist, self.demands[route[i]]))
            dists.append(self.all_distances[route[i]][route[i-1]])
            total_dist += dists[-1]
            total_cSat += cSats[-1]
        return total_dist, total_cSat, dists, cSats

    def mutate(self, parent_population, mutation_rate):
        next_generation = []
        for i in range(self.population_size):
            next_generation.append(parent_population[i].copy())

            #Swap two points in the route
            for _ in range(mutation_rate):
                a = np.random.randint(0, len(next_generation))
                b = np.random.randint(0, len(next_generation))

                next_generation[i][a], next_generation[i][b] = next_generation[i][b], next_generation[i][a]

        return next_generation

    def next_generation(self):
        parents = self.population.copy()
        #children = self.crossover(parents)
        children = self.mutate(parents, 1)

        self.dists_for_population = []
        self.cSats_for_population = []

        for i in range(self.population_size):
            dist, cSat, dists, cSats = self.calculate_objectives(children[i])
            self.dist_history.append(dist)
            self.cSat_history.append(cSat)
            self.route_history.append(children[i].copy())

            self.dists_for_population.append(dists)
            self.cSats_for_population.append(cSats)

            self.population.append(children[i].copy())

        #calculate the weights to determine which routes to keep from the population
        weights_customer_satisfaction = self.cSat_history[-self.population_size*2-1:-1] # select objective values from the history for the parent and child populations
        weights_customer_satisfaction = np.array(weights_customer_satisfaction)/sum(weights_customer_satisfaction)

        weights_distance = self.dist_history[-self.population_size*2-1:-1]
        weights_distance = np.array(weights_distance)/sum(weights_distance)

        chosen_ones = np.random.choice(np.arange(len(self.population)), self.population_size, p=(weights_customer_satisfaction+weights_distance)/2)      
        self.population = [self.population[i] for i in chosen_ones]

        self.pareto_front = findParetoFront(self.route_history, self.dist_history, self.cSat_history)





class MultiObjectiveAntColonyOptimiser:
    def __init__(self, points, demands, colony_size):
        self.points = points
        self.route = []
        self.colony_size = colony_size
        self.demands = demands

        self.current_colony_routes = [[] for i in range(colony_size)]

        self.distances = [[np.linalg.norm(i - j) for j in self.points] for i in self.points]
        self.global_ferromones = np.ones((len(self.points), len(self.points)))

        self.distance_history = []
        self.customer_satisfaction_history = []
        self.route_history = []

        self.objective_preference = 0.5
        
        self.pareto_front = []

    def get_probabilities(self, ant_position, available_positions, time_taken):
        weights = []
        for pos in available_positions:
            dist = self.distances[ant_position][pos]
            cSat = calculateCustomerSatisfaction(time_taken+dist, self.demands[pos])
            ferromone = self.global_ferromones[ant_position][pos]
            weight = ((1/dist) * self.objective_preference + cSat * (1-self.objective_preference)) * ferromone
            weights.append(weight)
        total = sum(weights)
        if total == 0:
            p = np.ones(len(weights)) / len(weights)
            return p
        p = np.array(weights)/total
        return p

    def optimise(self):
        best_ferromone = 0
        self.current_colony_routes = [[] for i in range(self.colony_size)]
        for route in range(self.colony_size):
            available_positions = list(range(len(self.points)))
            current_position = np.random.randint(0, len(self.points))
            available_positions.remove(current_position)
            total_dist = 0
            total_cSat = 0
            while len(available_positions) > 0:
                p = self.get_probabilities(current_position, available_positions, total_dist)
                selected_point = np.random.choice(available_positions, 1, p=p)[0]
                self.current_colony_routes[route].append(selected_point)
                total_dist += self.distances[current_position][selected_point]
                total_cSat += calculateCustomerSatisfaction(total_dist, self.demands[selected_point])
                current_position = selected_point
                available_positions.remove(current_position)

            # lay down ferromones
            route_ferromone = ((1/total_dist) * self.objective_preference + total_cSat * (1-self.objective_preference))
            for i in range(len(self.current_colony_routes[route])):
                self.global_ferromones[self.current_colony_routes[route][i]][self.current_colony_routes[route][i-1]] += route_ferromone

            if route_ferromone > best_ferromone:
                best_ferromone = route_ferromone
                self.route = self.current_colony_routes[route].copy()

            self.distance_history.append(total_dist)
            self.customer_satisfaction_history.append(total_cSat)

            self.route_history.append(self.current_colony_routes[route].copy())

        # normalise ferromones:
        self.global_ferromones /= np.amax(self.global_ferromones)
        self.pareto_front = findParetoFront(self.route_history, self.distance_history, self.customer_satisfaction_history)



