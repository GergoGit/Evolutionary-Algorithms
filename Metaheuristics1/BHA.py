# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:33:26 2021

@author: bonnyaigergo

Black Hole Algorithm (BHA)

https://github.com/mMarzeta/BlackHole_Swarm_Alghorithm/blob/master/BH.py
https://www.researchgate.net/publication/281786410_Black_Hole_Algorithm_and_Its_Applications/link/570df45108ae2b772e43305a/download
https://www.sciencepubco.com/index.php/JACST/article/view/4094

Best = Black Hole
event horizon

maybe an inertia can help here as well
"""

import numpy as np

    
def BHA(objective_func, 
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
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_list = [best]
        for ite_num in range(iterations):
            for idx in range(population_size):
                population[idx] += np.random.rand() * (best - population[idx])
                population[idx] = np.clip(a=population[idx], a_min=min_bound, a_max=max_bound)
            event_horizon = fitness[best_idx] / sum(fitness)
            
            for idx in range(population_size):
                if np.linalg.norm(best - population[idx]) < event_horizon and idx != best_idx:
                    population[idx] = min_bound + np.random.rand(dimensions) * dimension_range
            fitness = np.asarray([objective_func(individual) for individual in population])
            best_idx = np.argmin(fitness)
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
    result = list(BHA(objective_func=obj_func, 
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