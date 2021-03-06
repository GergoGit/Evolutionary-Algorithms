# -*- coding: utf-8 -*-
"""
Created on Sun May  1 16:57:00 2022

@author: bonnyaigergo

Whale optimization algorithm (WOA)

inspired by the humpback whale attacking prey

https://www.geeksforgeeks.org/whale-optimization-algorithm-woa/
https://www.geeksforgeeks.org/implementation-of-whale-optimization-algorithm/
https://sci-hub.se/10.1016/j.advengsoft.2016.01.008
https://github.com/docwza/woa/blob/master/src/whale_optimization.py

phases: search, encircle, hunt

exploration and exploitation phases

A, C coefficients defines the movement of encircling the prey
b constant defining the logarithmic spiral
p = switch between phases

"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class WOA(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 b=1,
                 strategy_switch_probability=0.5,
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
        self.b = b
        self.strategy_switch_probability = strategy_switch_probability

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    def initialize_parameters(self, a):
        A = 2 * a * np.random.rand() - a
        C = 2 * np.random.rand()
        l = np.random.uniform(-1, 1)
        return A, C, l
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        
        for nth_gen in range(self.n_generation):
            
            # linearly decreased from 2 to 0
            a = 2 * (1 - nth_gen / self.n_generation)
     
            for indiv_idx in range(self.population_size):
                A, C, l = self.initialize_parameters(a)
                
                if np.random.rand() < self.strategy_switch_probability:
                    # Encircling prey
                    if abs(A) < 1:
                        # Moving towards prey (exploitation)
                        D = np.abs(C * best - population[indiv_idx])
                        offspring = best - A * D
                    else:
                        # Global search (exploration)
                        idxs = [i for i in range(self.population_size) if i != indiv_idx]  
                        random_partner = population[np.random.choice(a=idxs, size=1)].ravel()
                        D = np.abs(C * random_partner - population[indiv_idx])
                        offspring = random_partner - A * D
                else:
                    D = np.abs(best - population[indiv_idx])
                    # Bubble-net attacking method (exploitation)
                    offspring = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best
     
                offspring = self.check_search_space(offspring)          
                offspring_fitness = self.obj_fn(offspring)
                if offspring_fitness < fitness[indiv_idx]:
                    population[indiv_idx] = offspring
                    fitness[indiv_idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best = population[indiv_idx]
                        best_idx = indiv_idx
                                
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
    optimiser = WOA(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    population_size=30,
                    b=1,
                    strategy_switch_probability=0.5,
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