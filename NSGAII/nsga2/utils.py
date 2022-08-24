# from NSGAII_copy.nsga2.population import Population
from nsga2.population import Population
import copy
import numpy as np
from numba import jit
import random

class NSGA2Utils:

    def __init__(self, problem, population_size, num_of_tour_particips, tournament_prob, crossover_param, mutation_param):

    # def __init__(self, problem, num_of_individuals=100,
    #              num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):

        self.problem = problem
        self.population_size = population_size
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        # self.mutation_param = mutation_param

    def create_initial_population(self):
        population = Population()
        for _ in range(self.population_size):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front,local):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0
            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10**9
                front[solutions_num-1].crowding_distance = 10**9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num-1):
                    front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m])/scale
                    if front[i] is local:
                        front[i].crowding_distance=10**9


    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            # parent2 = parent1
            # while parent1 == parent2:
            parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        child1 = copy.deepcopy(individual1)
        child2 = copy.deepcopy(individual2)
        for i in range(len(child1.features)):
            if np.random.rand() <= 0.5:

                child1.features[i]=individual2.features[i]
                child2.features[i]=individual1.features[i]
            else:
                child1.features[i] = individual1.features[i]
                child2.features[i] = individual2.features[i]

        return child1, child2

    def __mutate(self, child):
        num_of_features = len(child.features)
        for i in range(num_of_features):
            if np.random.rand() <= (1 / (num_of_features)):
                child.features[i]= 1 - child.features[i]

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant
        return best

    def __choose_with_prob(self, prob):
        if np.random.rand() <= prob:
            return True
        return False

    def calc_prob(self, solution):
        W = np.sum(np.multiply(solution.features, self.problem.w))
        X = sum(solution.features)
        myinf = float(np.finfo(np.float128).max)
        myexpmax = int(np.finfo(np.float128).maxexp)
        if W >= self.problem.capacity or solution.objectives[1] > (self.problem.maxp * len(solution.features)):
            # return 0.5
            return np.abs(W-self.problem.capacity) + 1
        else:
            if self.problem.cond == 1:
                return mycheby(X, self.problem.capacity, W, self.problem.delta, myinf)
            elif self.problem.cond == 2:
                return mychern(X, self.problem.capacity, W, self.problem.delta, myinf, myexpmax)

    def find_best(self,population):
        # this function finds the best solution for our single objective problem
        # it finds the best solution
        # it takes best nPop solutions in output population and if the best solution is not one of the nPop solutions
        # it eliminates the worst and appends our best solution
        Sminus = []
        Splus = []

        for i in population:
            if i.objectives[0] <= self.problem.capacity:
                Sminus.append(i)
            else:
                Splus.append(i)

        if len(Sminus) > 0:  # the solution with highest profit should be chosen
            best_solution = Sminus[np.argmax([np.abs(i.objectives[1]) for i in Sminus])]  # maximize the profit

        else:
            best_solution = Splus[np.argmin([i.objectives[0] for i in Splus])]  # minimize the profit

        return best_solution

    def calc_error(self, best_solution):
        prob = self.calc_prob(best_solution)

        if prob <= self.problem.rho:
            err = self.problem.pstar - np.abs(best_solution.objectives[1])
        else:
            err = (1 + prob) * self.problem.pstar

        if err < 0:
            print('stop')
        return err

@jit
def mycheby(X, C, W, delta, myinf):
    if X == 0:
        return 0
    if W > C:
        return myinf
    else:
        num = (delta ** 2) * X
        den = ((delta ** 2) * X) + 3 * ((C - W) ** 2)
        prob_cheby = num / den
        return prob_cheby

def mychern(X, C, W, delta, myinf, myexpmax):
    if X == 0:
        return 0
    else:
        t = (C - W) / (delta * X)
        if t < 0:
            return myinf
        else:
            num = (-t ** 2) * 0.5 * X
            den = (2 + (2 / 3) * t)
            nd = num / den
            if nd > myexpmax:
                nd = myexpmax
            prob_chern = np.exp(nd)
        return prob_chern
