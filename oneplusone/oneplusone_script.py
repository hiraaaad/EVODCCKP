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
            # prob_chern = (np.exp(t)/((1+t)**(1+t)))**W
        return prob_chern

# def sbound_cheby():

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

    # def mutate(self,number_of_items):
    #     # flip each bit and return the one who won the tournament
    #     offspring = copy.deepcopy(self.individual)
    #     for i in range(number_of_items):
    #         if np.random.rand()<=(1/(number_of_items)):
    #             offspring.variables[i]=1-offspring.variables[i]
    #
    #     offspring.evaluate(self.capacity,self.w,self.delta,self.cond,self.p)
    #
    #     tournament=[self.individual,offspring]
    #
    #
    #     weight_tournament=np.array([self.individual.weight,offspring.weight])
    #     prob_tournament=np.array([self.individual.prob,offspring.prob])
    #     profit_tournamet=np.array([self.individual.profit,offspring.profit])
    #
    #     if (np.all(weight_tournament< self.capacity) and np.all(prob_tournament <= self.rho)):
    #         winner = tournament[np.argmax(profit_tournamet)]
    #     elif not np.all(weight_tournament< self.capacity):
    #         winner = tournament[np.argmin(weight_tournament - self.capacity)]
    #     elif not np.all(prob_tournament <= self.rho):
    #         winner = tournament[np.argmin(prob_tournament - self.rho)]
    #     else:
    #         # find which one is feasible:
    #         for item in tournament:
    #             if ((item.weight < self.capacity) and (item.prob <= self.rho)):
    #                 winner = item
    #
    #     self.individual=winner
    #
    #
    # def calc_err(self):
    #     prob = self.individual.prob
    #     if (self.individual.weight >= self.capacity):
    #         prob = 0.5
    #
    #     if prob < self.rho:
    #         err = self.pstar - self.individual.profit
    #     else:
    #         err = (1 + prob) * self.pstar
    #
    #     if err > 100000:
    #         print('stop')
    #     self.finalerr.append(err)

    def evaluate(self):
        self.individual.evaluate(self.w,self.rho,self.delta,self.cond,self.p)


class individual():
    # def __init__(self, population_size: int)
    def __init__(self, number_items: int):
        self.variables = np.random.choice(range(2), number_items)
        self.prob= np.inf
        self.profit = np.inf

    def calc_prob(self, C, w, delta,cond):
        W = int(np.sum(np.multiply(self.variables, w)))
        X=sum(self.variables)
        myinf=float(np.finfo(np.float128).max)
        myexpmax=int(np.finfo(np.float128).maxexp)

        if cond == 1:
            return mycheby(X, C, W, delta,myinf)
        elif cond == 2:
            return mychern(X, C, W, delta,myinf,myexpmax)
        else:
            return min(mycheby(X, C, W, delta,myinf),mychern(X, C, W, delta,myinf,myexpmax))

    # def calc_prob(self, C, w, delta,cond):
    #     W = np.sum(np.multiply(self.variables, w))
    #     X = sum(self.variables)
    #     # Cheby
    #     if cond == 1:
    #         if W>C:
    #             prob_cheby = np.inf
    #         else:
    #             num = (delta ** 2) * X
    #             den = ((delta ** 2) * X) + 3 * ((C - W) ** 2)
    #             prob_cheby = num / den
    #         self.prob = prob_cheby
    #     elif cond == 2:
    #     # Chernoff
    #         if X == 0:
    #             prob_chern = 0
    #         else:
    #             t = (C - W) / (delta * X)
    #             num = (-t ** 2) * 0.5 * X
    #             den = (2 + (2 / 3) * t)
    #             if den ==0:
    #                 prob_chern = np.inf
    #             else:
    #                 nd=num/den
    #                 prob_chern = np.exp(nd)
    #             #     prob_chern = ne.evaluate('exp(nd)')
    #         self.prob = prob_chern

    def evaluate(self,C,w,delta,cond,p):
        self.profit= np.sum(np.multiply(self.variables, p))
        self.weight=np.sum(np.multiply(self.variables, w))
        self.prob=self.calc_prob(C, w, delta ,cond)

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

    def mutate(self):
        number_of_items=len(self.individual.variables)
        # flip each bit and return the one who won the tournament
        offspring = copy.deepcopy(self.individual)
        for i in range(number_of_items):
            if np.random.rand()<=(1/(number_of_items)):
                offspring.variables[i]=1-offspring.variables[i]

        offspring.evaluate(self.capacity,self.w,self.delta,self.cond,self.p)

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


    def calc_err(self):
        prob = self.individual.calc_prob(self.capacity,self.w,self.delta, self.cond)

        if (self.individual.weight >= self.capacity):
            prob = np.abs(self.individual.weight - self.capacity) + 1

        if prob <= self.rho:
            err = self.pstar - self.individual.profit
        else:
            err = (1 + prob) * self.pstar

        # if err > 100000:lv
            # print('stop')
        self.finalerr.append(err)

    def evaluate(self):
        self.individual.evaluate(self.capacity,self.w,self.delta,self.cond,self.p)
            
            















