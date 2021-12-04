# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 21:46:28 2021

@author: bonnyaigergo

Fruit Fly (FOA) optimization algorithm

https://www.hindawi.com/journals/mpe/2015/492195/
https://www.hindawi.com/journals/mpe/2013/108768/
https://github.com/zixuanweeei/fruit-fly-optimization-algorithm/blob/master/ffoa.m

D: distance
S: smell

"""

import numpy as np

def FOA(objective_func, 
        func_bounds, 
        population_size, 
        alfa=0.5, 
        decay=False,
        alfa_min=0,
        alfa_max=0.5,
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
        X_axis = min_bound + np.random.rand(population_size, dimensions) * dimension_range
        Y_axis = min_bound + np.random.rand(population_size, dimensions) * dimension_range
        fitness = np.asarray([objective_func(individual) for individual in X_axis])
        best = X_axis[np.argmin(fitness)]
        best_list = [best]
        for ite_num in range(iterations):
            if decay:
                alfa = (alfa_max - alfa_min) * ((ite_num + 1) / iterations)
            X = np.clip(X_axis + alfa * np.random.uniform(low=-1, high=1, size=(population_size, dimensions)) * dimension_range, a_min=min_bound, a_max=max_bound)
            Y = np.clip(Y_axis + alfa * np.random.uniform(low=-1, high=1, size=(population_size, dimensions)) * dimension_range, a_min=min_bound, a_max=max_bound)                
            D = [(x**2 + y**2)**0.5 for x, y in zip(X, Y)]
            S = [1/dist for dist in D]
            fitness = np.asarray([objective_func(smell) for smell in S])
            best_idx = np.argmin(fitness)
            X_axis = np.tile(A=X[best_idx], reps=(population_size, 1))
            Y_axis = np.tile(A=Y[best_idx], reps=(population_size, 1))
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
    result = list(FOA(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     population_size=30, 
                     alfa=0.5, 
                     decay=True,
                     alfa_min=0,
                     alfa_max=0.5,
                     runs=runs, 
                     iterations=300, 
                     patience=30))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()
