# -*- coding: utf-8 -*-
"""
Created on Sat Dec 4 21:50:47 2021

@author: bonnyaigergo

Cuckoo Search algorithm

https://www.researchgate.net/publication/45904981_Cuckoo_Search_via_Levy_Flights
https://www.youtube.com/watch?v=8sHkQ8kGEr8

https://www.randomservices.org/random/special/Levy.html
https://www.vosesoftware.com/riskwiki/Levydistribution.php

alfa = step size

"""

import numpy as np


def LevyFlight(n):
    beta = 1.5
    u = np.random.normal(loc=0, scale=0.6966, size=n)
    v = np.random.normal(loc=0, scale=1, size=n)
    s = u/(abs(v)**(1/beta))
    return s

def CS(objective_func, 
       func_bounds, 
       population_size, 
       alfa,
       decay=True,
       decay_linear=False,
       decay_factor=0.95,
       alfa_min=0,
       alfa_max=0.01,
       abandon_probability=0.25,
       runs=1, 
       iterations=50,
       patience=10,
       epsilon=1E-10,
       verbose=0):

    dimensions = len(func_bounds)
    min_bound = np.asarray([min(dim) for dim in func_bounds])
    max_bound = np.asarray([max(dim) for dim in func_bounds])
    dimension_range = np.fabs(min_bound - max_bound)
    
    for run_num in range(runs):
        population = min_bound + np.random.rand(population_size, dimensions) * dimension_range
        fitness = np.asarray([objective_func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_list = [best]
        for ite_num in range(iterations):   
            
            for idx in range(population_size):
                offspring = population[idx] + alfa * LevyFlight(dimensions) * (population[idx] - best)
                offspring = np.clip(a=offspring, a_min=min_bound, a_max=max_bound)
                offspring_fitness = objective_func(offspring)
                if offspring_fitness < fitness[idx]:
                    population[idx] = offspring
                    fitness[idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = idx
                        best = population[best_idx]
                        
            for idx in range(population_size):
                offspring = population[idx]*1
                idxs = [i for i in range(population_size) if i != idx]
                for dim in range(dimensions):
                    if np.random.rand() < abandon_probability:                        
                        random_partner = population[np.random.choice(a=idxs, size=1)].ravel()[dim]
                        offspring[dim] = offspring[dim] + np.random.rand() * (offspring[dim] - random_partner)
                offspring = np.clip(a=offspring, a_min=min_bound, a_max=max_bound)
                offspring_fitness = objective_func(offspring)
                if offspring_fitness < fitness[idx]:
                    population[idx] = offspring
                    fitness[idx] = offspring_fitness
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = idx
                        best = population[best_idx]
                        
            if patience != None and ite_num >= patience:
                if (np.asarray([abs(element-best) for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            if decay:
                if decay_linear:
                    alfa = (alfa_max - alfa_min) * ((ite_num + 1) / iterations)
                else:
                    alfa *= decay_factor
            best_list.append(best)
            yield run_num, ite_num, best, fitness[best_idx] 
            
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(CS(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     alfa=0.01, 
                     decay=True,
                     decay_linear=True,
                     decay_factor=0.95,
                     alfa_min=0,
                     alfa_max=1,                     
                     abandon_probability=0.25,
                     runs=runs, 
                     iterations=300, 
                     patience=50))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()