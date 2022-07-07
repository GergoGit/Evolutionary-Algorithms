# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:22:23 2021

@author: bonnyaigergo


https://machinelearningmastery.com/evolution-strategies-from-scratch-in-python/
https://github.com/anyoptimization/pymoo/blob/master/pymoo/algorithms/soo/nonconvex/es.py
https://github.com/alirezamika/evostra/blob/master/evostra/algorithms/evolution_strategy.py
http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaes_inmatlab.html
https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py
https://cma-es.github.io/
https://www.researchgate.net/publication/254462106_Tutorial_CMA-ES_evolution_strategies_and_covariance_matrix_adaptation

"""

import numpy as np

def create_new_generation(elitism, offsprings_normalized, population_normalized, population_denormalized, fitness, objective_func, min_bound, dimension_range, population_size, dimensions):
    if elitism:
        offsprings_denormalized = min_bound + offsprings_normalized * dimension_range
        fitness_offspring = np.asarray([objective_func(individual) for individual in offsprings_denormalized])
        population_normalized = np.concatenate((population_normalized, offsprings_normalized), axis=0)
        population_denormalized = np.concatenate((population_denormalized, offsprings_denormalized), axis=0)
        fitness = np.concatenate((fitness, fitness_offspring), axis=0)
    else:
        population_normalized = offsprings_normalized
        population_denormalized = min_bound + population_normalized * dimension_range
        fitness = np.asarray([objective_func(individual) for individual in population_denormalized])
    n_best_idx = fitness.argsort()[:population_size]
    new_population_normalized = population_normalized[n_best_idx]
    new_population_denormalized = population_denormalized[n_best_idx]
    new_fitness = fitness[n_best_idx]
    best_idx = np.argmin(new_fitness)
    return new_population_normalized, new_population_denormalized, new_fitness, new_population_denormalized[best_idx], best_idx


def ES(objective_func, 
         func_bounds,
         elitism=True,
         population_size=20, 
         n_offspring=10,
         sigma=0.1,
         decay=0.99,
         runs=1, 
         generations=10,
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
        fitness = np.asarray([objective_func(individual) for individual in population_denormalized])
        best_idx = np.argmin(fitness)
        best = population_denormalized[best_idx]
        best_list = [best]
        for gen_num in range(generations):
            # keep n best
            offsprings_normalized = np.empty(shape=(0, dimensions))
            for idx in range(population_size):
                for _ in range(n_offspring):
                    offspring = np.clip(population_normalized[idx] + np.random.normal(size=dimensions) * sigma * decay, a_min=0, a_max=1) 
                    offsprings_normalized = np.vstack((offsprings_normalized, offspring))
            population_normalized, population_denormalized, fitness, best, best_idx = create_new_generation(elitism, offsprings_normalized, population_normalized, population_denormalized, fitness, objective_func, min_bound, dimension_range, population_size, dimensions)

            if patience != None and gen_num >= patience:
                if (np.asarray([element-best for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, gen_num, best, fitness[best_idx]
            
                    
        
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
        
    obj_func = lambda x: sum(x**2)/len(x)
    func_bounds = [(-50,50)]*2
    runs = 5
    result = list(ES(objective_func=obj_func, 
                     func_bounds=func_bounds, 
                     elitism=True,
                     population_size=20, 
                     n_offspring=10,
                     sigma=0.04,
                     decay=0.91,
                     runs=runs, 
                     generations=300, 
                     patience=30))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()
    
    