from hmac import new
import numpy as np
from numpy.random import pareto
from pkg_resources import dist_factory

# this is the base of the exponent used for calculating customer satisfaction
exp_base = 1.1

# obsolete:
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
       
    
# an ant colony optimiser that only uses lowest distance travelled as an objective
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
        

# a simple optimiser that also accounts for the demand of each city 
class MultiObjectiveOptimiser:
    def __init__(self, points, demands, percentile=0.5):
        self.points = np.array(points)
        self.demands = np.array(demands)
        self.route = np.arange(len(points))
        np.random.shuffle(self.route)
        self.route_history = [self.route.copy()]
        self.all_distances = [[np.linalg.norm(i - j) for j in self.points] for i in self.points]

        # exploration variables
        self.explore_threshold = 5 #int(len(points) / 4)
        self.attempts_since_change = 0
        self.swaps_per_iteration = 1

        # variables for mutate paretos + explore
        self.tested = []
        self.number_of_swaps = 1

        # reducer variables
        self.dist_per_point_history = [[]]
        self.cSat_per_point_history = [[]]
        self.percentile = percentile

        cSat, dist = self.CalculateObjectives()
        self.distance_history = [dist]
        self.customer_satisfaction_history = [cSat]


        #pareto front variables
        self.pareto_front = []

    def CalculateObjectives(self):
        total_customer_satisfaction = 0
        total_distance_of_route = 0
        time_taken = 0

        time_per_unit_of_demand = 50
        time_per_unit_of_distance = 10

        prev_i = self.route[-1]
        for i in self.route:
            #finding the route distance:
            d = self.all_distances[prev_i][i]
            total_distance_of_route += d
            time_taken += d*time_per_unit_of_distance

            #finding the customer satisfaction:
            #formula determined by requiring that zero demand has maximum satisfaction no matter the time taken ...
            #  ... and that zero time taken has maximum satisfaction no matter the demand ...
            #  ... and that for non zero demand and non zero time taken, satisfaction decreases exponentially according to a given time taken ...
            #  ... and satisfaction decreases exponentially as time taken increases according to a given demand.
            exponent = (- time_taken * self.demands[i])/10000
            cs = np.pow(exp_base, exponent)

            self.dist_per_point_history[-1].append(d)
            self.cSat_per_point_history[-1].append(cs)
            total_customer_satisfaction += cs
            time_taken += time_per_unit_of_demand*self.demands[i]

            prev_i = i.copy()
        self.dist_per_point_history.append([])
        self.cSat_per_point_history.append([])
        return total_customer_satisfaction, total_distance_of_route

    def Undominated(self, d, c):
        if c > max(self.customer_satisfaction_history) or d < min(self.distance_history):
            return True
        else:
            return False

    def FindParetoFront(self):
        self.pareto_front = []
        for solution in range(len(self.route_history)):
            betterDistanceThanThese = self.distance_history[solution] <= self.distance_history
            betterCustomerSatisfactionThanThese = self.customer_satisfaction_history[solution] >= self.customer_satisfaction_history

            if (all(np.logical_or(betterCustomerSatisfactionThanThese, betterDistanceThanThese))):
                self.pareto_front.append(solution)

    def RandomWalkOptimise(self):
        while any(np.array_equal(self.route, route) for route in self.route_history):
            np.random.shuffle(self.route)
        self.route_history.append(self.route.copy())
        custSat, routeDist = self.CalculateObjectives()
        self.distance_history.append(routeDist)
        self.customer_satisfaction_history.append(custSat)

    def HillClimbSwapperOptimise(self):
        i=0
        j=0
        while any(np.array_equal(self.route, route) for route in self.route_history):
            i = np.random.randint(0, len(self.route))
            j = np.random.randint(0, len(self.route))
            self.route[i], self.route[j] = self.route[j], self.route[i]

        custSat, routeDist  = self.CalculateObjectives()

        # condition to revert back if we got worse
        if self.Undominated(routeDist, custSat):
            pass
        else:
            self.route[i], self.route[j] = self.route[j], self.route[i]

        self.route_history.append(self.route.copy())
        self.distance_history.append(routeDist)
        self.customer_satisfaction_history.append(custSat)

        self.FindParetoFront()

    def HillClimbSwapperWithExplorationOptimise(self):
        old_route = self.route.copy()
        while any(np.array_equal(self.route, route) for route in self.route_history):
            for n in range(self.swaps_per_iteration):
                i = np.random.randint(0, len(self.route))
                j = np.random.randint(0, len(self.route))
                self.route[i], self.route[j] = self.route[j], self.route[i]

        route_diff = np.array(old_route) - np.array(self.route)
        custSat, routeDist = self.CalculateObjectives()

        # condition to revert back if we got worse
        if self.Undominated(routeDist, custSat):
            self.attempts_since_change = 0
            self.swaps_per_iteration = 1

        else:
            self.attempts_since_change += 1
            self.route = old_route
            if self.attempts_since_change > self.explore_threshold:
                self.attempts_since_change = 0
                self.swaps_per_iteration += 1


        self.route_history.append(self.route.copy())
        self.distance_history.append(routeDist.copy())
        self.customer_satisfaction_history.append(custSat.copy())

        self.FindParetoFront()

    def MutateOnlyParetoFrontOptimise(self):
        self.route = self.route_history[np.random.choice(self.pareto_front)]
        original = self.route.copy()
        random_point = np.random.randint(0, len(self.route))
        for route_index, point_index in enumerate(self.route):
            self.route[random_point], self.route[route_index] = self.route[route_index], self.route[random_point]
            if not any(np.array_equal(self.route, route) for route in self.route_history):
                customer_satisfaction, total_distance = self.CalculateObjectives()
                self.route_history.append(self.route.copy())
                self.customer_satisfaction_history.append(customer_satisfaction)
                self.distance_history.append(total_distance)
            self.route = original.copy()

        self.FindParetoFront()

    def MutateParetoAndExploreOptimise(self):
        test_route = np.random.choice(self.pareto_front)
        available_test_routes = list(set(self.pareto_front) - set(self.tested)) # gets the intesection of routes that were mutated and tested with the current pareto front
        if len(available_test_routes) > 0: # if there are available routes to test, test them
            test_route = np.random.choice(available_test_routes)
        else: # if there are no available routes to test, then reset the tested list and explore each route more deeply by having more starting points
            test_route = np.random.choice(self.pareto_front)
            self.tested = []
            self.number_of_swaps += 1
        self.tested.append(test_route)

        original = self.route.copy()

        random_points = [np.random.randint(0, len(self.route)) for i in range(self.number_of_swaps)]
        for route_index, point_index in enumerate(self.route):
            for swap in range(self.number_of_swaps):
                additional_point = np.random.randint(0, len(self.route))
                for random_point in random_points:
                    self.route[random_point], self.route[route_index] = self.route[route_index], self.route[random_point]
            if not any(np.array_equal(self.route, route) for route in self.route_history):
                customer_satisfaction, total_distance = self.CalculateObjectives()
                self.route_history.append(self.route.copy())
                self.customer_satisfaction_history.append(customer_satisfaction)
                self.distance_history.append(total_distance)
            self.route = original.copy()

        self.FindParetoFront()

    def ReducerOptimiser(self):
        max_cSat_per_point = max(self.cSat_per_point_history[-2])
        min_cSat_per_point = min(self.cSat_per_point_history[-2])
        range_cSat_per_point = max_cSat_per_point - min_cSat_per_point
        cSat_threshold = self.percentile * range_cSat_per_point + min_cSat_per_point

        max_dist_per_point = max(self.dist_per_point_history[-2])
        min_dist_per_point = min(self.dist_per_point_history[-2])
        range_dist_per_point = max_dist_per_point - min_dist_per_point
        dist_threshold = (1-self.percentile) * range_dist_per_point + min_dist_per_point


        bad_points = []
        for i in range(len(self.route)):
            if self.cSat_per_point_history[-2][i] < cSat_threshold or self.dist_per_point_history[-2][i] > dist_threshold:
                bad_points.append(i)
        np.random.shuffle(bad_points)


        # reduce bad points:
        original_route = self.route.copy()
        for i in range(0, len(bad_points), 2):
            self.route[bad_points[i]], self.route[bad_points[i-1]] = self.route[bad_points[i-1]], self.route[bad_points[i]]

        cSat, dist = self.CalculateObjectives()
        self.route_history.append(self.route.copy())
        if self.Undominated(dist, cSat):
            pass
        else:
            self.route = original_route.copy()  
            
        self.distance_history.append(dist)
        self.customer_satisfaction_history.append(cSat)

        self.FindParetoFront()


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

    def calculate_customer_satisfaction(self, time_taken, demand):
        # formula determined by requiring that zero demand has maximum satisfaction no matter the time taken ...
        #  ... and that zero time taken has maximum satisfaction no matter the demand ...
        #  ... and that for non zero demand and non zero time taken, satisfaction decreases exponentially according to a given time taken ...
        #  ... and satisfaction decreases exponentially as time taken increases according to a given demand.
        exponent = (- time_taken * demand)/10000
        cs = np.power(exp_base, exponent)
        return cs

    def get_probabilities(self, ant_position, available_positions, time_taken):
        weights = []
        for pos in available_positions:
            dist = self.distances[ant_position][pos]
            cSat = self.calculate_customer_satisfaction(time_taken+dist, self.demands[pos])
            ferromone = self.global_ferromones[ant_position][pos]
            weight = ((1/dist) * self.objective_preference + cSat * (1-self.objective_preference)) * ferromone
            weights.append(weight)
        total = sum(weights)
        p = np.array(weights)/total
        return p

    def optimise(self):
        best_ferromone = 0
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
                total_cSat += self.calculate_customer_satisfaction(total_dist, self.demands[selected_point])
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

