# from NSGAII_copy.nsga2.individual import Individual
from nsga2.individual import Individual
import numpy as np

class Problem:

    def __init__(self, num_of_variables,new_capacity,w,p,maxp,delta,rho,cond,r,pstar):
        self.num_of_objectives = 2
        self.num_of_variables = num_of_variables
        self.capacity = new_capacity
        self.w = w
        self.p = p
        self.maxp = maxp
        self.delta = delta
        self.rho = rho
        self.cond = cond
        self.r = r
        self.pstar = pstar

    def generate_individual(self): # Binary solution
        individual = Individual(length=self.num_of_variables)
        individual.features = [1 if np.random.randint(0, 2) == 0 else 0 for _ in range(self.num_of_variables)]
        individual.features = np.array(individual.features)
        return individual

    def calculate_objectives(self, individual):
        def sbound(x):
            W=np.sum(np.multiply(x,self.w)) # total nominal weight of a given solution
            S=np.sum(x)
            if self.cond == 1:
                sigma = np.sqrt(S / 3) * self.delta
                lam = sigma * np.sqrt(self.rho * (1 - self.rho)) / self.rho
                return W + lam
            elif self.cond == 2:
                X = 0.5 * S
                if X !=0:
                    t = -(np.log(self.rho) - np.sqrt(np.log(self.rho) ** 2 - 18 * X * np.log(self.rho))) / (3 * X)
                elif X==0:
                    t=0
                # lam = -.66*self.delta*(np.log(self.rho)-np.sqrt((np.log(self.rho)**2)-9*np.log(self.rho)*S))
                return W + t*self.delta*S

        individual.objectives[0]=sbound(individual.features)

        if (individual.objectives[0] <= self.capacity + self.r ) & (individual.objectives[0] >= self.capacity - self.r):
            individual.objectives[1] = np.sum(-np.multiply(individual.features,self.p))
        else:
            if individual.objectives[0] > self.capacity + self.r:
                violation=np.abs(individual.objectives[0] - (self.capacity + self.r))
            else:
                violation=np.abs((self.capacity - self.r) - individual.objectives[0])

            individual.objectives[0]=(1+violation)*(self.maxp*self.num_of_variables) # penalization
            individual.objectives[1]=(1+violation)*(self.maxp*self.num_of_variables)
