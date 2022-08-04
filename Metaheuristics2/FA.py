# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:08:15 2022

Firefly algorithm

https://sci-hub.se/10.1002/9780470640425.ch17

Note:
   slow
   decay_factor has to be tuned
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class FA(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 alfa=0.5, 
                 decay_factor=0.9,
                 beta_null=1,
                 gamma=0.5,
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
        self.n_dim = objective.n_dim
        
        self.population_size = population_size
        self.n_generation = n_generation
        self.alfa = alfa
        self.decay_factor = decay_factor
        self.beta_null = beta_null
        self.gamma = gamma

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        
        for nth_gen in range(self.n_generation):
            
            for i in range(self.population_size):    
                for j in [j for j in range(self.population_size) if j != i]:
                    if fitness[j] >= fitness[i]:
                        r = np.sum(np.square(population[i] - population[j]), axis=-1)
                        beta = self.beta_null / (1 + self.gamma * r)
                        random_step = self.alfa * (np.random.rand(self.n_dim) - 0.5) * self.dim_range
                        offspring = population[i] + beta * (population[j] - population[i]) + random_step
                        offspring = self.check_search_space(offspring)
                        offspring_fitness = self.obj_fn(offspring)
                        if offspring_fitness < fitness[i]:
                            population[i] = offspring
                            fitness[i] = offspring_fitness
                            if offspring_fitness < fitness[best_idx]:
                                best_idx = i
                                best = population[best_idx]
                                
            self.alfa *= self.decay_factor
                                
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin
    from StoppingCriterion import ImpBestObj
    fn = Rastrigin(2)
    optimiser = FA(objective=fn, 
                   stopping_criterion='imp_avg_obj',
                   population_size=30,
                   alfa=0.5, 
                   decay_factor=0.9,
                   beta_null=1,
                   gamma=0.5,
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