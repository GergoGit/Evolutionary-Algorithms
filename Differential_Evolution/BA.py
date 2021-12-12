# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:00:20 2021

@author: bonnyaigergo

Bat Algorithm (BA)

https://github.com/buma/BatAlgorithm/blob/master/BatAlgorithm.py
https://github.com/rahuldjoshi28/batOptimization/blob/master/bat.py
https://sci-hub.se/10.1007/978-3-642-12538-6_6
https://www.youtube.com/watch?v=4OfJa3SfU84

loudness = A
pulse rate = r

"""

import numpy as np

def BA(objective_func, 
       func_bounds, 
       population_size,
       loudness,       
       starting_pulse_rate,
       Fmin,
       Fmax,
       alfa=0.95,
       gamma=0.95,
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
        velocity = np.zeros(shape=(population_size, dimensions))
        loudness = np.tile(A=loudness, reps=population_size)
        pulse_rate = np.tile(A=starting_pulse_rate, reps=population_size)
        
        for ite_num in range(iterations):
            
            for bat in range(population_size):
                # update position
                # F = Fmin + (Fmax - Fmin) * np.random.rand()
                # velocity[bat] += F * (population[bat] - best)
                velocity[bat] += (alfa ** ite_num) * 0.05 * np.random.uniform(-1, 1, dimensions) * (population[bat] - best)
                offspring = population[bat] + velocity[bat]
                offspring = np.clip(a=offspring, a_min=min_bound, a_max=max_bound)
                
                if np.random.rand() < pulse_rate[bat]:
                    offspring = best + (alfa ** ite_num) * 0.05 * np.random.uniform(-1, 1, dimensions) * dimension_range
                    offspring = np.clip(a=offspring, a_min=min_bound, a_max=max_bound)
                                
                offspring_fitness = objective_func(offspring)
                if offspring_fitness < fitness[bat]:
                    # np.random.rand() < loudness[bat] and 
                    population[bat] = offspring
                    fitness[bat] = offspring_fitness
                    loudness[bat] *= alfa
                    pulse_rate[bat] = starting_pulse_rate * (1 - np.exp(-gamma * ite_num))
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = bat
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
    result = list(BA(objective_func=obj_func, 
                       func_bounds=func_bounds, 
                       population_size=30,
                       loudness=1,       
                       starting_pulse_rate=0.15,
                       Fmin=0,
                       Fmax=1,
                       alfa=0.95,
                       gamma=0.95,
                       runs=runs, 
                       iterations=300,
                       patience=300,
                       epsilon=1E-10,
                       verbose=0))
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()  
                    
                
        