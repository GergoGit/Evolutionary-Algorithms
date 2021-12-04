# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 23:56:21 2021

@author: bonnyaigergo

Artificial Bee Colony (ABC) algorithm
https://github.com/ntocampos/artificial-bee-colony/blob/master/main.py
https://www.youtube.com/watch?v=OPWCTs0d7vA

"""

import numpy as np

def ABC(objective_func, 
        func_bounds, 
        population_size, 
        limit,
        runs=1, 
        iterations=50,
        patience=10,
        epsilon=1E-10,
        verbose=0):
    dimensions = len(func_bounds)
    min_bound = np.asarray([min(dim) for dim in func_bounds])
    max_bound = np.asarray([max(dim) for dim in func_bounds])
    dimension_range = np.fabs(min_bound - max_bound)
    n_bees = round(population_size/2)
    for run_num in range(runs):
        population = min_bound + np.random.rand(population_size, dimensions) * dimension_range
        fitness = np.asarray([objective_func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_list = [best]
        bad_trials = np.zeros(shape=(population_size))
        
        for ite_num in range(iterations):
            
            # Employed bees
            for idx in range(n_bees):                                
                idxs = [i for i in range(population_size) if i != idx]
                random_partner = population[np.random.choice(idxs, 1)].ravel()
                phi = np.random.uniform(-1, 1, dimensions)
                new_place = np.clip(population[idx] + phi * (population[idx] - random_partner), a_min=min_bound, a_max=max_bound)
                new_place_fitness = objective_func(new_place)
                if new_place_fitness < fitness[idx]:
                    population[idx] = new_place
                    fitness[idx] = new_place_fitness
                    if new_place_fitness < fitness[best_idx]:
                        best = population[idx]
                        best_idx = idx
                else:
                    bad_trials[idx] += 1
            
            P = 0.9 * (fitness / fitness[best_idx]) + 0.1
            
            # Onlooker bees
            for idx in range(n_bees):
                if np.random.rand() < P[idx]:
                    idxs = [i for i in range(population_size) if i != idx]
                    random_partner = population[np.random.choice(idxs, 1)].ravel()
                    phi = np.random.uniform(-1, 1, dimensions)
                    new_place = np.clip(population[idx] + phi * (population[idx] - random_partner), a_min=min_bound, a_max=max_bound)
                    new_place_fitness = objective_func(new_place)
                    if new_place_fitness < fitness[idx]:
                        population[idx] = new_place
                        fitness[idx] = new_place_fitness
                        if new_place_fitness < fitness[best_idx]:
                            best = population[idx]
                            best_idx = idx
                    else:
                        bad_trials[idx] += 1
            
            # Scout bees
            for idx in range(n_bees):
                if bad_trials[idx] > limit:
                    new_place = np.clip(min_bound + np.random.rand(dimensions) * dimension_range, a_min=min_bound, a_max=max_bound)
                    new_place_fitness = objective_func(new_place)
                    if new_place_fitness < fitness[idx]:
                        population[idx] = new_place
                        fitness[idx] = new_place_fitness
                        bad_trials[idx] = 0
                        if fitness[idx] < fitness[best_idx]:
                                best = population[idx]
                                best_idx = idx
            
            if patience != None and ite_num >= patience:
                if (np.asarray([element-best for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, ite_num, best, fitness[best_idx]


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-50,50)]*2
    runs = 5
    result = list(ABC(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=30, 
                     limit=40,
                     runs=runs, 
                     iterations=300, 
                     patience=30))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()
        