import numpy as np
import copy
import numexpr as ne
from numba import jit
# class POSDC():
#     def __init__(self):
#         self.__init__()

# @jit
def mycheby(X, C, W, delta,myinf):
    if X ==0:
        return 0
    if W > C:
        return myinf
    else:
        num = (delta ** 2) * X
        den = ((delta ** 2) * X) + 3 * ((C - W) ** 2)
        prob_cheby = num / den
        return prob_cheby

def mychern(X, C, W, delta,myinf,myexpmax):
    if X == 0:
        return 0
    else:
        t = (C - W) / (delta * X)
        if t<0:
            return myinf
        else:
            num = (-t ** 2) * 0.5 * X
            den = (2 + (2 / 3) * t)
            nd=num/den
            if nd>myexpmax:
                nd=myexpmax
            prob_chern = np.exp(nd)
        return prob_chern
# def sbound_cheby():

class posdc():
    def __init__(self, solution, sminus: list, splus: list, capacity: int, w: list, p: list, maxp: int, delta: int, rho: float, cond: int,
                     r: int, eta:int, pstar: int):  # number_of_items: int):
            self.solution = solution # best available solution (single-objective)
            self.splus = splus
            self.sminus = sminus
            self.finalerr = []
            self.capacity = capacity
            self.w = w
            self.p = p
            self.maxp = maxp
            self.delta = delta
            self.rho = rho
            self.cond = cond
            self.r = r
            self.eta = eta
            self.pstar = pstar
            # self.global_best=None
            # self.global_error = 1e9

    def calc_ranking(self,solutions):
        dominating_ith = [0 for _ in range(len(solutions))] # number of solutions dominating solution ith
        # ith_dominated = [[] for _ in range(len(solutions))] # list of solutions dominated by solution ith
        # front[i] contains the list of solutions belonging to front i
        # front = [[] for _ in range(len(solutions))]
        front=[]
        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                # dominance test result
                if solutions[p].profit >= solutions[q].profit and solutions[p].sbound <= solutions[q].sbound:
                    dominance_test_result = -1
                elif solutions[q].profit >= solutions[p].profit and solutions[q].sbound <= solutions[p].sbound:
                    dominance_test_result = 1
                else:
                    dominance_test_result = 0

                if dominance_test_result == -1:
                    # ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result == 1:
                    # ith_dominated[q].append(p)
                    dominating_ith[p] += 1

        for i in range(len(solutions)):
            if dominating_ith[i] == 0:
                front.append(i)

        return [solutions[i] for i in front]

    def mutate(self):
        if (self.splus == [] and self.sminus == []):
            if self.solution.sbound <= self.capacity:
                self.sminus.append(self.solution)
            else:
                self.splus.append(self.solution)

    # choose an individual random from splus+sminus
    #     S=self.sminus+self.splus
        parent = np.random.choice(self.sminus+self.splus)
        number_of_items = len(parent.variables)
        offspring = copy.deepcopy(parent)
        # Flipping a bit
        for i in range(number_of_items):
            if np.random.rand() <= (1 / (number_of_items)):
                offspring.variables[i] = 1 - offspring.variables[i]

        offspring.evaluate(self.capacity,self.w,self.rho,self.delta,self.cond,self.p)
        # Dominance:
        if offspring.sbound <= self.capacity+self.eta and offspring.sbound >= self.capacity-self.eta:
            # first check where the offspring belongs to: splus or sminus?
            if offspring.sbound <= self.capacity:
                self.sminus.append(offspring)
                self.sminus = self.calc_ranking(self.sminus)
            else:
                self.splus.append(offspring)
                self.splus = self.calc_ranking(self.splus)


    def calc_err(self):
        # find best solution:
        if len(self.sminus) > 0:  # the solution with highest profit should be chosen
            best_solution = self.sminus[np.argmax([np.abs(i.profit) for i in self.sminus])]  # maximize the profit

        else:
            best_solution = self.splus[np.argmin([i.sbound for i in self.splus])]  # minimize the profit

        self.solution = best_solution

        self.solution.calc_prob(self.capacity, self.w, self.delta, self.cond)

        prob = self.solution.prob

        if (self.solution.weight >= self.capacity):
        #     prob = 0.5
            prob = np.abs(self.solution.weight - self.capacity) + 1


        if prob <= self.rho:
            err = self.pstar - self.solution.profit
        else:
            err = (1 + prob) * self.pstar

        # # This need to be changed according to final discussions
        # if self.solution.sbound <= self.capacity:
        #     err = self.pstar - self.solution.profit
        # else:
        #     penalty = np.abs(self.solution.sbound - self.capacity)
        #     err = self.pstar + penalty

        self.finalerr.append(err)




class individual_multi():
    # def __init__(self, population_size: int)
    def __init__(self, number_items: int):
        self.variables = np.random.choice(range(2), number_items)
        self.prob= np.inf
        self.profit = np.inf
        self.sbound = np.inf

    def calc_prob(self, C, w, delta, cond):
        W = int(np.sum(np.multiply(self.variables, w)))
        X = sum(self.variables)
        myinf = float(np.finfo(np.float128).max)
        myexpmax = int(np.finfo(np.float128).maxexp)

        if cond == 1:
            return mycheby(X, C, W, delta, myinf)
        elif cond == 2:
            return mychern(X, C, W, delta, myinf, myexpmax)
        else:
            return min(mycheby(X, C, W, delta, myinf), mychern(X, C, W, delta, myinf, myexpmax))

    # def calc_sbound(self, w, rho, delta, cond):
    #     W = int(np.sum(np.multiply(self.variables, w)))  # total nominal weight of a given solution
    #     S = sum(self.variables)
    #     if cond == 1:
    #         sigma = np.sqrt(S / 3) * delta
    #         lam = sigma * np.sqrt(rho * (1 - rho)) / rho
    #         # return W + lam
    #         self.sbound = W + lam
    #     elif cond == 2:
    #         X = 0.5 * S
    #         if X !=0:
    #             lam = -.66*self.delta*(np.log(self.rho)-np.sqrt((np.log(self.rho)**2)-9*np.log(self.rho)*S))
    #             self.sbound = W + lam
    #         elif X==0:
    #             t=0
    #             self.sbound = W

    def calc_sbound(self, w, rho, delta, cond):
        W = int(np.sum(np.multiply(self.variables, w)))  # total nominal weight of a given solution
        S = sum(self.variables)
        if cond == 1:
            sigma = np.sqrt(S / 3) * delta
            lam = sigma * np.sqrt(rho * (1 - rho)) / rho
            # return W + lam
            self.sbound = W + lam
        elif cond == 2:
            X = 0.5 * S
            if X != 0:
                t = -(np.log(rho) - np.sqrt(np.log(rho) ** 2 - 18 * X * np.log(rho))) / (3 * X)
            elif X == 0:
                t = 0
            # return W + t * delta * S
            # lam=-.66*delta*(np.log(rho)-np.sqrt((np.log(rho)**2)-9*np.log(rho)*S))
            # self.sbound = W + lam
            self.sbound = W + t * delta * S
            
    def evaluate(self,capacity,w,rho,delta,cond,p):
        self.profit= np.sum(np.multiply(self.variables, p))
        self.weight=np.sum(np.multiply(self.variables, w))
        self.calc_sbound(w,rho,delta,cond)
        self.prob = self.calc_prob(capacity, w, delta, cond)

class oneplusone():
    def __init__(self, solution,capacity:int,w:list,p:list,maxp:int,delta:int,rho:float,cond:int,r:int,pstar:int): # number_of_items: int):
        # self.individual =
        self.individual = copy.deepcopy(solution)
        self.finalerr=[]
        self.capacity = capacity
        self.w = w
        self.p = p
        self.maxp = maxp
        self.delta = delta
        self.rho = rho
        self.cond = cond
        self.r = r
        self.pstar = pstar

    def mutate(self,number_of_items):
        # flip each bit and return the one who won the tournament
        offspring = copy.deepcopy(self.individual)
        for i in range(number_of_items):
            if np.random.rand()<=(1/(number_of_items)):
                offspring.variables[i]=1-offspring.variables[i]

        offspring.evaluate(self.capacity,self.w,self.rho,self.delta,self.cond,self.p)

        tournament=[self.individual,offspring]


        weight_tournament=np.array([self.individual.weight,offspring.weight])
        prob_tournament=np.array([self.individual.prob,offspring.prob])
        profit_tournamet=np.array([self.individual.profit,offspring.profit])

        if (np.all(weight_tournament< self.capacity) and np.all(prob_tournament <= self.rho)):
            winner = tournament[np.argmax(profit_tournamet)]
        elif not np.all(weight_tournament< self.capacity):
            winner = tournament[np.argmin(weight_tournament - self.capacity)]
        elif not np.all(prob_tournament <= self.rho):
            winner = tournament[np.argmin(prob_tournament - self.rho)]
        else:
            # find which one is feasible:
            for item in tournament:
                if ((item.weight < self.capacity) and (item.prob <= self.rho)):
                    winner = item

        self.individual=winner

    def evaluate(self):
        self.individual.evaluate(self.capacity,self.w,self.rho,self.delta,self.cond,self.p)
            
            















