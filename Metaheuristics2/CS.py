# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:22:36 2022

@author: bonnyaigergo

Cuckoo Search algorithm (CS)

https://sci-hub.se/10.1109/nabic.2009.5393690

https://www.researchgate.net/publication/45904981_Cuckoo_Search_via_Levy_Flights
https://www.youtube.com/watch?v=8sHkQ8kGEr8
https://github.com/YutaUme/CS/blob/master/cs.py

https://www.randomservices.org/random/special/Levy.html
https://www.vosesoftware.com/riskwiki/Levydistribution.php

alfa = step size
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import math
import StoppingCriterion

class CS0(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 n_cuckoo=5,
                 worst_nest_proportion=0.25,
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
        self.n_cuckoo = n_cuckoo
        self.n_worst = math.ceil(population_size * worst_nest_proportion)

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    def LevyFlight(self):
        beta = 1.5
        u = np.random.normal(loc=0, scale=0.6966, size=self.n_dim)
        v = np.random.normal(loc=0, scale=1, size=self.n_dim)
        s = u/(abs(v)**(1/beta))
        return s
    
    def new_cuckoo_via_levyflight(self, population):
        idxs = [i for i in range(self.population_size)]
        random_nests = population[np.random.choice(idxs, 3, replace=False)].ravel()
        offspring = random_nests[0] + self.LevyFlight()*(random_nests[1] - random_nests[2])
        return offspring
    
    def leave_worst_nests_and_create_new_ones(self, population, fitness):
        idxs_leave = fitness.argsort()[-self.n_worst:]  
        idxs_keep = [i for i in range(self.population_size) if i not in idxs_leave]  
        
        for indiv_idx in idxs_leave:
            random_nests = np.random.choice(idxs_keep, 2, replace=False)
            population[indiv_idx] = population[indiv_idx] + \
                self.LevyFlight()*(population[random_nests[0]] - population[random_nests[1]])
            population[indiv_idx] = self.check_search_space(population[indiv_idx])
            fitness[indiv_idx] = self.obj_fn(population[indiv_idx])
        return population, fitness
            
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        
        for nth_gen in range(self.n_generation):
            
            # Cuckoo
            for _ in range(self.n_cuckoo):  
                
                offspring = self.new_cuckoo_via_levyflight(population)
                offspring = self.check_search_space(offspring)
                offspring_fitness = self.obj_fn(offspring)
                
                # Choose random nest
                random_nest_idx = np.random.choice(range(self.population_size), 1)
                
                if offspring_fitness < fitness[random_nest_idx]:
                    population[random_nest_idx] = offspring
                    fitness[random_nest_idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best = population[random_nest_idx]
                        best_idx = random_nest_idx
            
            # Elitism
            population, fitness = self.leave_worst_nests_and_create_new_ones(population, fitness)
            best_idx = np.argmin(fitness)
            best = population[best_idx]
                                
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break
                
                

class CS1(object):
    """
    2 phase:
    1) from every nest search towards the best solution with Levy flight
    2) secondary random search towards random nests
    """
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 population_size=30,
                 abandon_probability=0.25,
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
        self.abandon_probability = abandon_probability

        
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
    
    def search_around_best(self, population, indiv_idx, best):
        offspring = population[indiv_idx] + self.LevyFlight(self.n_dim)*(best - population[indiv_idx])
        return offspring
    
    def abandon_and_create_new_nest(self, population, indiv_idx):
        offspring = population[indiv_idx].copy()
        idxs = [i for i in range(self.population_size) if i != indiv_idx]
        for dim in range(self.n_dim):
            if np.random.rand() < self.abandon_probability:                        
                random_partner = population[np.random.choice(a=idxs, size=1)].ravel()[dim]
                offspring[dim] = offspring[dim] + np.random.rand() * (offspring[dim] - random_partner)
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
                
                offspring = self.search_around_best(population, indiv_idx, best)
                offspring = self.check_search_space(offspring)
                offspring_fitness = self.obj_fn(offspring)
                if offspring_fitness < fitness[indiv_idx]:
                    population[indiv_idx] = offspring
                    fitness[indiv_idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best = population[indiv_idx]
                        best_idx = indiv_idx
                        
                offspring = self.abandon_and_create_new_nest(population, indiv_idx)
                offspring = self.check_search_space(offspring)
                offspring_fitness = self.obj_fn(offspring)
                if offspring_fitness < fitness[indiv_idx]:
                    population[indiv_idx] = offspring
                    fitness[indiv_idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best = population[indiv_idx]
                        best_idx = indiv_idx
                                
            self.best_par =np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin
    from StoppingCriterion import ImpBestObj
    fn = Rastrigin(2)  
    optimiser = CS0(objective=fn, 
                    stopping_criterion='imp_avg_obj',
                    population_size=30,
                    n_cuckoo=40,
                    worst_nest_proportion=0.25,
                    n_generation=300)
    # optimiser = CS1(objective=fn, 
    #                 stopping_criterion='imp_avg_obj',
    #                 population_size=30,
    #                 abandon_probability=0.25,
    #                 n_generation=300)
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