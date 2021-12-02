# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:42:13 2021

@author: bonnyaigergo
"""

import numpy as np

def PSO(objective_func, 
        func_bounds, 
        population_size, 
        c1=2,
        c2=2,
        alfa=0.5, 
        runs=1, 
        generations=50,
        patience=10,
        epsilon=1E-10,
        verbose=0):
    dimensions = len(func_bounds)
    min_bound = np.asarray([min(dim) for dim in func_bounds])
    max_bound = np.asarray([max(dim) for dim in func_bounds])
    dimension_range = np.fabs(min_bound - max_bound)
    for run_num in range(runs):
        population_normalized = np.random.rand(population_size, dimensions)
        population_denormalized = min_bound + population_normalized * dimension_range
        velocity = np.zeros(shape=(population_size, dimensions))
        fitness = np.asarray([objective_func(individual) for individual in population_denormalized])
        global_best_idx = np.argmin(fitness)
        global_best = population_denormalized[global_best_idx]
        global_best_list = [global_best]
        particle_best = population_denormalized*1
        best_fitness = objective_func(global_best)
        
        for gen_num in range(generations):
            for idx in range(population_size):
                r1, r2 = np.random.rand(2)
                velocity[idx] = alfa*velocity[idx] + c1*r1*(particle_best[idx] - population_denormalized[idx]) + c2*r2*((global_best - population_denormalized[idx]))
                population_denormalized[idx] = np.clip(population_denormalized[idx] + velocity[idx], a_min=min_bound, a_max=max_bound)
                offspring_fitness = objective_func(population_denormalized[idx])
                if offspring_fitness < objective_func(particle_best[idx]):
                    particle_best[idx] = population_denormalized[idx]
                    if offspring_fitness < best_fitness:
                        global_best = population_denormalized[idx]
                        best_fitness = objective_func(global_best)
                        
            
            if patience != None and gen_num >= patience:
                if (np.asarray([element-global_best for element in global_best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            global_best_list.append(global_best)
            yield run_num, gen_num, global_best, best_fitness
                
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-50,50)]*2
    runs = 5
    result = list(PSO(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     runs=runs, 
                     generations=300, 
                     patience=30))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()