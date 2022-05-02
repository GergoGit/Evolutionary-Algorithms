# -*- coding: utf-8 -*-
"""
Created on Mon May  2 23:54:48 2022

Black Hole Algorithm (BHA)

https://github.com/mMarzeta/BlackHole_Swarm_Alghorithm/blob/master/BH.py
https://www.researchgate.net/publication/281786410_Black_Hole_Algorithm_and_Its_Applications/link/570df45108ae2b772e43305a/download
https://www.sciencepubco.com/index.php/JACST/article/view/4094

Best = Black Hole
event horizon
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class BHA(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
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

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    def random_initialization(self):
        individual = self.min_bound + np.random.rand(self.n_dim) * self.dim_range
        return individual
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
                
        for nth_gen in range(self.n_generation):
            
            for indiv_idx in range(self.population_size):                    
                population[indiv_idx] += np.random.rand() * (best - population[indiv_idx])
                population[indiv_idx] = self.check_search_space(population[indiv_idx])
            event_horizon = fitness[best_idx] / sum(fitness)
            
            for indiv_idx in range(self.population_size):
                if np.linalg.norm(best - population[indiv_idx]) < event_horizon and indiv_idx != best_idx:
                    population[indiv_idx] = self.random_initialization()
                    
            fitness = np.asarray([self.obj_fn(individual) for individual in population])
            best_idx = np.argmin(fitness)
            best = population[best_idx]
                                
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
    optimiser = BHA(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    population_size=30,
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