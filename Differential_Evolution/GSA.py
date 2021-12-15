# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:25:37 2021

@author: bonnyaigergo

Gravitational Search Algorithm (GSA)

https://sci-hub.se/10.1016/j.ins.2009.03.004
https://www.degruyter.com/document/doi/10.1515/math-2018-0132/html
https://downloads.hindawi.com/journals/jam/2015/894758.pdf
https://github.com/himanshuRepo/Gravitational-Search-Algorithm/blob/master/GSA.py

G = Gravitational constant
M = inertial Mass
m = mass
F = gravitational force
A = acceleration
V = velocity

"""

import numpy as np

def GSA(objective_func, 
        func_bounds, 
        population_size,
        G0,
        alfa,
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
        A = np.empty(shape=(population_size, dimensions))
        V = np.zeros(shape=(population_size, dimensions))
        mass = np.empty(shape=population_size)
        M = np.empty(shape=population_size)
        F = np.empty(shape=(population_size, dimensions))
        Force = np.empty(shape=(population_size, dimensions))
        
        for ite_num in range(iterations):
            G = G0 * ((1 / (ite_num + 1))**alfa)
            best_fitness = np.min(fitness)
            worst_fitness = np.max(fitness)
            # sum_fitness = np.sum(fitness)
            
            for idx in range(population_size):
                mass[idx] = (fitness[idx] - best_fitness)/(best_fitness - worst_fitness)
            
            sum_mass = np.sum(mass)
            M = mass / sum_mass
            
            for i in range(population_size):
                for j in range(population_size):
                    Force[j] = G * (M[i] * M[j]) / (np.linalg.norm(population[i] - population[j])**2 + 0.001) * (population[j] - population[i])
                    Force[j] *= np.random.uniform(0, 1, size=dimensions)
                F[i] = np.sum(Force, axis=0)
                if M[i] != 0:
                    A[i] = F[i] / M[i]
                else:
                    A[i] = 0
                
            V = G * np.random.uniform(-1, 1, size=(population_size, dimensions)) * V + A * dimension_range
            mutant = population + V
            mutant = np.clip(a=mutant, a_min=min_bound, a_max=max_bound)
            mutant_fitness = np.asarray([obj_func(individual) for individual in mutant])
            is_improved = np.reshape(np.asarray(mutant_fitness < fitness), newshape=(population_size,1))
            population = np.where(is_improved, mutant, population)         
            # population += V
            # population = np.clip(a=population, a_min=min_bound, a_max=max_bound)
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
    result = list(GSA(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     G0=10,
                     alfa=0.5,
                     population_size=20, 
                     runs=runs, 
                     iterations=300, 
                     patience=120))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()