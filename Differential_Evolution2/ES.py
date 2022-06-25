# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:49:23 2022

@author: bonnyaigergo

Evolutionary Strategy (ES)

"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class ES(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 apply_elitism=True,
                 population_size=30,
                 n_offspring=10,
                 sigma=0.1,
                 decay=0.99,
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
        self.n_offspring = n_offspring
        self.sigma = sigma
        self.decay = decay
        self.apply_elitism = apply_elitism

        
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
            
            all_offspring = np.empty(shape=(0, self.n_dim))
            
            for indiv_idx in range(self.population_size):
                
                for _ in range(self.n_offspring):  
                
                    offspring = population[indiv_idx] + np.random.normal(size=self.n_dim) * self.sigma * self.dim_range
                    offspring = self.check_search_space(offspring)
                    all_offspring = np.vstack((all_offspring, offspring))
                    
            if self.apply_elitism:
                fitness_offspring = np.asarray([self.obj_fn(individual) for individual in all_offspring])
                population = np.concatenate((population, all_offspring), axis=0)
                fitness = np.concatenate((fitness, fitness_offspring), axis=0)
            else:
                population = all_offspring
                fitness = np.asarray([self.obj_fn(individual) for individual in population])
                
            n_best_idx = fitness.argsort()[:self.population_size]
            population = population[n_best_idx]
            fitness = fitness[n_best_idx]
            best_idx = np.argmin(fitness)
            best = population[best_idx]
            
            self.sigma *= self.decay
                                
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
    optimiser = ES(objective=fn, 
                   stopping_criterion='imp_avg_obj',
                   apply_elitism=True,
                   population_size=30,
                   n_offspring=10,
                   sigma=0.1,
                   decay=0.7,
                   n_generation=300)
    optimiser.termination.from_nth_gen = 50
    optimiser.termination.patience = 20
    optimiser.run()
    
    plt.yscale('log', base=2) 
    plt.plot(optimiser.best_fitness)
    plt.legend()    
    
    # optimiser.n_dim
    # optimiser.best_fitness
    # optimiser.best_par
    # optimiser.termination.metric_list
    # optimiser.termination.check_list
    # fn.minima_loc
    # fn.minima