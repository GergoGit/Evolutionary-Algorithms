# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:39:03 2021

@author: bonnyaigergo

Firefly algorithm

https://github.com/HaaLeo/swarmlib/blob/master/swarmlib/fireflyalgorithm/firefly_problem.py
https://github.com/firefly-cpp/FireflyAlgorithm/blob/master/FireflyAlgorithm.py
https://www.youtube.com/watch?v=7bxn14n57Qk

intensity of brightness: fitness

exponential decay vs linear decay
"""

import numpy as np

def FA(objective_func, 
       func_bounds, 
       population_size, 
       alfa=0.5, 
       decay=True,
       decay_linear=True,
       alfa_min=0,
       alfa_max=0.1,
       beta_null=1,
       gamma=0.5, 
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
            if decay:
                alfa = (alfa_max - alfa_min) * ((ite_num + 1) / iterations)
                # alfa *= 0.9
            for i in range(population_size):
                for j in [j for j in range(population_size) if j != i]:
                    if fitness[j] >= fitness[i]:
                        r = np.sum(np.square(population[i] - population[j]), axis=-1)
                        # beta = beta_null * np.exp(-gamma * r)
                        beta = beta_null / (1 + gamma * r)
                        # random_step = alfa * np.random.uniform(low=-1, high=1, size=dimensions) * dimension_range
                        random_step = alfa * (np.random.rand(dimensions) - 0.5) * dimension_range
                        offspring = population[i] + beta * (population[j] - population[i]) + random_step
                        offspring = np.clip(a=offspring, a_min=min_bound, a_max=max_bound)
                        offspring_fitness = objective_func(offspring)
                        if offspring_fitness < fitness[i]:
                            population[i] = offspring
                            fitness[i] = offspring_fitness
                            if offspring_fitness < fitness[best_idx]:
                                best_idx = i
                                best = population[best_idx]
            if patience != None and ite_num >= patience:
                if (np.asarray([element-best for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, ite_num, best, fitness[best_idx]   
                        
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(FA(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     alfa=0.01, 
                     decay=True,
                     alfa_min=0,
                     alfa_max=0.01,
                     beta_null=1,
                     gamma=0.1,
                     runs=runs, 
                     iterations=300, 
                     patience=50))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()