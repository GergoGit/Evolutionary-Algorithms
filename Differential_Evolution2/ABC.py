# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 23:56:21 2021

@author: bonnyaigergo

Artificial Bee Colony (ABC) algorithm
https://github.com/ntocampos/artificial-bee-colony/blob/master/main.py
https://github.com/renard162/BeeColPy/blob/master/beecolpy/beecolpy.py
https://www.youtube.com/watch?v=OPWCTs0d7vA

"""

import numpy as np
import StoppingCriterion

class ABC(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 limit = 40,
                 population_size=30,
                 worker_bee_prop = 0.5,
                 generation_num=300):
        
                
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
        self.dim_num = objective.dim_num
        
        self.limit = limit
        self.n_bees = round(worker_bee_prop * population_size)
        self.population_size = population_size
        self.generation_num = generation_num

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.dim_num) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    def search_rand_place(self):
        new_place = self.min_bound + np.random.rand(self.dim_num) * self.dim_range
        return new_place
    
    
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
    
        
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.dim_num))
        self.best_fitness = np.empty(shape=(0, 1))
        
        bad_trials = np.zeros(shape=(self.population_size))
        
        for nth_gen in range(self.generation_num):
            
            # Employed bees
            for indiv_idx in range(self.n_bees):                          
                idxs = [i for i in range(self.population_size) if i != indiv_idx]
                random_partner = population[np.random.choice(idxs, 1)].ravel()
                phi = np.random.uniform(-1, 1, self.dim_num)
                new_place = population[indiv_idx] + phi * (population[indiv_idx] - random_partner)
                new_place = self.check_search_space(new_place)
                new_place_fitness = self.obj_fn(new_place)
                if new_place_fitness < fitness[indiv_idx]:
                    population[indiv_idx] = new_place
                    fitness[indiv_idx] = new_place_fitness
                    if new_place_fitness < fitness[best_idx]:
                        best = population[indiv_idx]
                        best_idx = indiv_idx
                else:
                    bad_trials[indiv_idx] += 1
            
            P = 0.9 * (fitness / fitness[best_idx]) + 0.1
            
            # Onlooker bees
            for indiv_idx in range(self.n_bees):
                if np.random.rand() < P[indiv_idx]:
                    idxs = [i for i in range(self.population_size) if i != indiv_idx]
                    random_partner = population[np.random.choice(idxs, 1)].ravel()
                    phi = np.random.uniform(-1, 1, self.dim_num)
                    new_place = population[indiv_idx] + phi * (population[indiv_idx] - random_partner)
                    new_place = self.check_search_space(new_place)
                    new_place_fitness = self.obj_fn(new_place)
                    if new_place_fitness < fitness[indiv_idx]:
                        population[indiv_idx] = new_place
                        fitness[indiv_idx] = new_place_fitness
                        if new_place_fitness < fitness[best_idx]:
                            best = population[indiv_idx]
                            best_idx = indiv_idx
                    else:
                        bad_trials[indiv_idx] += 1
            
            # Scout bees
            for indiv_idx in range(self.n_bees):
                if bad_trials[indiv_idx] > self.limit:
                    new_place = self.search_rand_place()
                    new_place_fitness = self.obj_fn(new_place)
                    if new_place_fitness < fitness[indiv_idx]:
                        population[indiv_idx] = new_place
                        fitness[indiv_idx] = new_place_fitness
                        bad_trials[indiv_idx] = 0
                        if fitness[indiv_idx] < fitness[best_idx]:
                                best = population[indiv_idx]
                                best_idx = indiv_idx

                        
            self.best_par = np.append(self.best_par, best)
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin
    from StoppingCriterion import ImpBestObj
    fn = Rastrigin(2)  
    optimiser = ABC(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    limit=40,
                    population_size=30,
                    worker_bee_prop=0.5)
    optimiser.termination.from_nth_gen = 50
    optimiser.termination.patience = 20
    optimiser.run()
    
    plt.yscale('log', base=2) 
    plt.plot(optimiser.best_fitness)
    plt.legend()    
    
    # optimiser.dim_num
    # optimiser.best_fitness
    # optimiser.termination.metric_list
    # optimiser.termination.check_list
    # fn.minima_loc
    # fn.minima
        