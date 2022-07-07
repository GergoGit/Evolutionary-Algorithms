# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:28:37 2022

Fruit Fly (FOA) optimization algorithm

https://www.hindawi.com/journals/mpe/2015/492195/
https://www.hindawi.com/journals/mpe/2013/108768/
https://github.com/zixuanweeei/fruit-fly-optimization-algorithm/blob/master/ffoa.m
https://sci-hub.se/10.1016/j.knosys.2011.07.001
https://braininformatics.springeropen.com/track/pdf/10.1186/s40708-020-0102-9.pdf
https://www.youtube.com/watch?v=xPR5aLv9Ylw

D: distance
S: smell

Note:
    easily stuck in a local optima
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class FOA(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 alfa=0.5, 
                 decay_factor=0.9,
                 n_generation=300):
                        
        if stopping_criterion is not None:
            self.termination = StoppingCriterion.criteria_fn_map(stopping_criterion)()
        else:
            self.termination = None
        
        self.objective = objective
        self.obj_fn = objective.evaluate
        self.search_space = self.objective.search_space
        
        self.min_bound = np.asarray([min(dim) for dim in self.search_space])
        self.max_bound = np.asarray([max(dim) for dim in self.search_space])
        self.dim_range = np.fabs(self.min_bound - self.max_bound)
        self.n_dim = objective.dim_num
        
        self.population_size = population_size
        self.n_generation = n_generation
        self.alfa = alfa
        self.decay_factor = decay_factor
        
        
    def initialize_population(self):
        X_axis = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        Y_axis = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range        
        fitness = np.asarray([self.obj_fn(individual) for individual in X_axis])
        best_idx = np.argmin(fitness)
        best = X_axis[best_idx]
        return X_axis, Y_axis, fitness, best_idx, best
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        X_axis, Y_axis, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        for nth_gen in range(self.n_generation):
            
            X = X_axis + self.alfa * np.random.uniform(low=-1, high=1, size=(self.population_size, self.n_dim)) * self.dim_range
            Y = Y_axis + self.alfa * np.random.uniform(low=-1, high=1, size=(self.population_size, self.n_dim)) * self.dim_range               
            X = self.check_search_space(X)
            Y = self.check_search_space(Y)
            Distances = [(x**2 + y**2)**0.5 for x, y in zip(X, Y)]
            Smells = [1/dist for dist in Distances]
            fitness = np.asarray([self.obj_fn(smell) for smell in Smells])
            best_idx = np.argmin(fitness)
            best = X[best_idx]
            X_axis = np.tile(A=X[best_idx], reps=(self.population_size, 1))
            Y_axis = np.tile(A=Y[best_idx], reps=(self.population_size, 1))

            self.alfa *= self.decay_factor
                                
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(X, fitness, best_idx, nth_gen):
                    break


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin
    from StoppingCriterion import ImpBestObj
    fn = Rastrigin(2)
    optimiser = FOA(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    population_size=50,
                    alfa=0.6, 
                    decay_factor=0.99,
                    n_generation=300)
    optimiser.termination.from_nth_gen = 50
    optimiser.termination.patience = 20
    optimiser.run()
    
    plt.yscale('log', base=2) 
    plt.plot(optimiser.best_fitness)
    plt.legend()    
    
    # optimiser.n_dim
    # optimiser.best_fitness
    # optimiser.termination.metric_list
    # optimiser.termination.check_list
    # fn.minima_loc
    # fn.minima