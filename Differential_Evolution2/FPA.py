# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:10:07 2022

Flower Pollination Algorithm

https://sci-hub.se/10.1007/978-3-319-67669-2_5

search type is chosen randomly by swotch probability:
1, Global pollination or global (biotic) search
2, Local pollination or local (abiotic) search 

"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class FPA(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 switch_probability = 0.8,
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
        self.switch_probability = switch_probability

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    def LevyFlight(self, n_dim):
        beta = 1.5
        u = np.random.normal(loc=0, scale=0.6966, size=n_dim)
        v = np.random.normal(loc=0, scale=1, size=n_dim)
        s = u/(abs(v)**(1/beta))
        return s
    
    def global_search(self, population, indiv_idx, best):
        offspring = population[indiv_idx] + self.LevyFlight(self.n_dim)*(best - population[indiv_idx])
        return offspring
    
    def local_search(self, population, indiv_idx):
        idxs = [i for i in range(self.population_size) if i != indiv_idx]
        random_partners = population[np.random.choice(idxs, 2, replace=False)].ravel()
        offspring = population[indiv_idx] + np.random.rand()*(random_partners[0] - random_partners[1])
        return offspring
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        
        for nth_gen in range(self.n_generation):
            
            for indiv_idx in range(self.population_size):    
                if np.random.rand() < self.switch_probability:
                    offspring = self.global_search(population, indiv_idx, best)
                else:
                    offspring = self.local_search(population, indiv_idx)
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
    optimiser = FPA(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    population_size=30,
                    switch_probability=0.8,
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
        