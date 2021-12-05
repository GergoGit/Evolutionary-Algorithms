# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:47:09 2021

@author: bonnyaigergo

Whale optimization algorithm (WOA)

https://www.geeksforgeeks.org/whale-optimization-algorithm-woa/
https://www.geeksforgeeks.org/implementation-of-whale-optimization-algorithm/
https://sci-hub.se/10.1016/j.advengsoft.2016.01.008

phases: search, encircle, hunt

exploration and exploitation phases

A, C coefficients
b constant defining the logarithmic spiral

"""

import numpy as np

def WOA(objective_func, 
       func_bounds, 
       population_size, 
       b=1,
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
            # linearly decreased from 2 to 0
            a = 2 * (1 - ite_num / iterations)
     
            for idx in range(population_size):
                A = 2 * a * np.random.rand() - a
                C = 2 * np.random.rand()
                l = np.random.uniform(-1, 1)
                p = np.random.rand()
                D = np.zeros(shape=dimensions)

                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best - population[idx])
                        offspring = best - A * D
                    else:
                        idxs = [i for i in range(population_size) if i != idx]  
                        random_partner = population[np.random.choice(a=idxs, size=1)].ravel()
                        D = np.abs(C * random_partner - population[idx])
                        offspring = random_partner - A * D
                else:
                    D = np.abs(best - population[idx])
                    offspring = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
     
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
            best_list.append(best)
            yield run_num, ite_num, best, fitness[best_idx]
            
            
            
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(WOA(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     b=1,
                     runs=runs, 
                     iterations=300, 
                     patience=50))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()