# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:26:19 2021

@author: bonnyaigergo

Grey Wolf Optimization (GWO)

https://sci-hub.se/10.1016/j.advengsoft.2013.12.007
https://www.youtube.com/watch?v=CQquzq24BPc&t=1s

"""

import numpy as np

def mutation(a, abc_wolf, other_wolf):
    A = 2 * a * np.random.rand() - a
    C = 2 * np.random.rand()    
    D = np.abs(C * abc_wolf - other_wolf)
    X = abc_wolf - A * D
    return X

def GWO(objective_func, 
        func_bounds, 
        population_size, 
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

        top3_idx = fitness.argsort()[:3]            
        best = population[top3_idx[0]]
        second_best = population[top3_idx[1]]
        third_best = population[top3_idx[2]]
        best_list = [best]
        for ite_num in range(iterations):
            # linearly decreased from 2 to 0
            a = 2 * (1 - ite_num / iterations)            
     
            for idx in range(population_size):
                X1 = mutation(a, abc_wolf=best, other_wolf=population[idx])
                X2 = mutation(a, abc_wolf=second_best, other_wolf=population[idx])
                X3 = mutation(a, abc_wolf=third_best, other_wolf=population[idx])
                
                offspring = (X1 + X2 + X3) / 3
                offspring_fitness = objective_func(offspring)
                # Greedy selection
                if offspring_fitness < fitness[idx]:
                    population[idx] = offspring
                    fitness[idx] = offspring_fitness
                
            top3_idx = fitness.argsort()[:3]            
            best = population[top3_idx[0]]
            second_best = population[top3_idx[1]]
            third_best = population[top3_idx[2]]
            
            if patience != None and ite_num >= patience:
                if (np.asarray([abs(element-best) for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, ite_num, best, fitness[top3_idx[0]]


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(GWO(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     runs=runs, 
                     iterations=300, 
                     patience=50))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()                