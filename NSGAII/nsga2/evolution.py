# from NSGAII_copy.nsga2.utils import NSGA2Utils
# from NSGAII_copy.nsga2.population import Population

from nsga2.utils import NSGA2Utils
from nsga2.population import Population
import copy
class Evolution:

    def __init__(self,
                 problem,
                 global_best,
                 global_error,
                 population: list,
                 num_of_iterations: int,
                 population_size:int,
                 offspring_population_size: int,
                 prob_mutation: float,
                 prob_crossover: float,
                 finalerr: list):

        self.utils = NSGA2Utils(problem,population_size= population_size, num_of_tour_particips = 2, tournament_prob = 1, crossover_param = 2, mutation_param = 5) # num_of_tour_particips = 2, tournament_prob = 0.9, crossover_param = 2, mutation_param = 5):
        self.num_of_iterations = num_of_iterations
        self.on_generation_finished = []
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.finalerr = finalerr
        self.global_best = global_best
        self.global_error = global_error
        self.population = population

    def evolve(self):
        if len(self.population)==0:
            self.population = self.utils.create_initial_population() # Create initial
        # else:
            # Automatically Inject it
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front,None)
        children = self.utils.create_children(self.population)
        returned_population = None
        for i in range(self.num_of_iterations):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)

            local_best = self.utils.find_best(self.population.fronts[0])
            local_error = self.utils.calc_error(local_best)
            if local_error < self.global_error:
                self.global_best = copy.deepcopy(local_best)
                self.global_error = local_error
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.population_size:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num],None)
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num],local_best)
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.population_size - len(new_population)])
            returned_population = self.population
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front,local_best)
            children = self.utils.create_children(self.population)

            # if len(self.finalerr) > 0:
                # if local_error > self.finalerr[-1]:
                    # print('stop')

            self.finalerr.append(local_error)
            # try:
            #     if not self.global_best in self.population:
            #         del(self.population.population[-1])
            #         self.population.population.append(self.global_best)
                    # print('stop')
            # except:
            #     x


        # return returned_population.fronts[0]
        return returned_population

