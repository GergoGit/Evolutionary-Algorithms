# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:54:28 2022

Grey Wolf Optimization (GWO)

https://sci-hub.se/10.1016/j.advengsoft.2013.12.007
https://www.youtube.com/watch?v=CQquzq24BPc&t=1s

"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class GWO(object):
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
    
    def mutation(self, a, abc_wolf, other_wolf):
        A = 2 * a * np.random.rand() - a
        C = 2 * np.random.rand()    
        D = np.abs(C * abc_wolf - other_wolf)
        X = abc_wolf - A * D
        return X
            
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
            
            top3_idx = fitness.argsort()[:3]            
            second_best = population[top3_idx[1]]
            third_best = population[top3_idx[2]]
            
            for indiv_idx in range(self.population_size):
                
                X1 = self.mutation(a, abc_wolf=best,         other_wolf=population[indiv_idx])
                X2 = self.mutation(a, abc_wolf=second_best,  other_wolf=population[indiv_idx])
                X3 = self.mutation(a, abc_wolf=third_best,   other_wolf=population[indiv_idx])
                offspring = (X1 + X2 + X3) / 3
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
    optimiser = GWO(objective=fn, 
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