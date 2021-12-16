# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:47:24 2021

@author: bonnyaigergo

Simulated Annealing (SA)

https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/

"""

import numpy as np

def SA(objective_func, 
       func_bounds,
       temperature,
       alfa,
       step_size,
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
        individual = min_bound + np.random.rand(dimensions) * dimension_range
        individual_fitness = objective_func(individual)
        best = individual.copy()
        best_fitness = individual_fitness.copy()
        best_list = [best]        
        # alfa_decay = alfa * (1 / iterations)
        
        for ite_num in range(iterations):
            
            mutant = individual + alfa * np.random.uniform(-1, 1, dimensions) * step_size * dimension_range
            mutant = np.clip(a=mutant, a_min=min_bound, a_max=max_bound)
            mutant_fitness = objective_func(mutant)
            
            if mutant_fitness < best_fitness:
                best, best_fitness = mutant, mutant_fitness
                
            diff = mutant_fitness - individual_fitness
            T = temperature / (ite_num + 1)
            prob = np.exp(-diff / T)
            # alfa *= 0.99
            # alfa -= alfa_decay
            
            if diff < 0 or np.random.rand() < prob:
                individual, individual_fitness = mutant, mutant_fitness
            if patience != None and ite_num >= patience:
                if (np.asarray([abs(element-best) for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, ite_num, best, best_fitness
                
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(SA(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     alfa=1,
                     temperature=10,
                     step_size=0.025,
                     runs=runs, 
                     iterations=300, 
                     patience=120))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()