# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:31:46 2021

@author: bonnyaigergo

Radial Movement Optimization

PSO + RMPO hybrid
https://journals.sagepub.com/doi/pdf/10.1177/0020294019842597

https://www.sciencedirect.com/science/article/pii/S2314717217300223
https://sci-hub.se/10.1002/cplx.21766

inertia = alfa
in the original paper k is a parameter to scale the random step, here beta = 1/k

"""

import numpy as np

def RMO(objective_func, 
        func_bounds, 
        population_size, 
        alfa,
        decay=True,
        decay_linear=False,
        decay_factor=0.95,
        alfa_min=0.1,
        alfa_max=1,
        beta=0.2,
        C1=0.1,
        C2=0.05,
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
        central_point = np.mean(a=population, axis=0)
        fitness = np.asarray([objective_func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        Global_best = population[best_idx]
        Global_best_fitness = min(fitness)
        best_list = [Global_best]
        V = np.empty(shape=(population_size, dimensions))
        alfa = alfa_max

        for ite_num in range(iterations):
            for idx in range(population_size):
                V[idx] = alfa * beta * np.random.normal(loc=0, scale=1, size=dimensions) * dimension_range
                population[idx] = np.clip(a=central_point + V[idx], a_min=min_bound, a_max=max_bound)
                fitness[idx] = objective_func(population[idx])
            
            Radial_best_fitness = min(fitness)
            Radial_best = population[np.argmin(fitness)]
            
            if Radial_best_fitness < Global_best_fitness:
                Global_best_fitness = Radial_best_fitness
                Global_best = Radial_best
                step = C1 * (Global_best - central_point)
            else:
                step = C1 * (Global_best - central_point) + C2 * (Radial_best - central_point)
            
            central_point += step 
                    
            if patience != None and ite_num >= patience:
                if (np.asarray([abs(element-Global_best) for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            if decay:
                if decay_linear:
                    alfa = (alfa_max - alfa_min) * ((ite_num + 1) / iterations)
                else:
                    alfa *= decay_factor
            best_list.append(Global_best)
            yield run_num, ite_num, Global_best, Global_best_fitness
            

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-100,100)]*2
    runs = 5
    result = list(RMO(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=20, 
                     alfa=1,
                    decay=True,
                    decay_linear=False,
                    decay_factor=0.95,
                    alfa_min=0.01,
                    alfa_max=1,
                    beta=0.1,
                    C1=0.3,
                    C2=0.2,
                     runs=runs, 
                     iterations=300, 
                     patience=100))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()