# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:59:33 2021

@author: bonnyaigergo

article:
https://arxiv.org/pdf/2101.06599.pdf
https://www.researchgate.net/publication/220403311_A_comparative_study_of_crossover_in_differential_evolution
file:///C:/Users/BONNYA~1/AppData/Local/Temp/mathematics-09-00427-v2.pdf
https://sci-hub.se/10.1016/j.swevo.2012.09.004
https://www.sciencedirect.com/science/article/pii/S111001682100613X
http://metahack.org/PPSN2014-Tanabe-Fukunaga.pdf
https://www.researchgate.net/figure/Exponential-crossover-pseudo-code_fig1_220176034
    
code:
https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/
https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python/
https://pymoo.org/algorithms/soo/de.html

DE/x/y/z

x: individual selection type for mutation, vector to be mutated like randomly selected member of the population or the one with best fitness
    rand: u = x1 + F(x2 - x3)
    best: u = xbest + F(x2 - x3)
    where F is the mutation factor (constant)
y: number of difference vectors used for perturbation like 
    y=1 case: u = x1 + F(x2 - x3)
    y=2 case: u = x1 + F(x2 - x3) + F(x4 - x5)
z: type of crossover (binomial, exponential, uniform)

F, C can be different random uniform numbers in the range of (0.5, 1) for each generation.

Best practices:
    DE/rand/1/bin
    1, set the population size to 10 times the number of parameters (dimensions)
    2, mutation factor = 0.8
    3, crossover probability = 0.9

verbose
generation count

at the beginning select n fittest individual

termination:
    x_best_new - x_best_old < epsilon

types:
     DE/rand/1:
         u = x1 + F(x2 - x3)
     DE/rand/2:
         u = x1 + F(x2 - x3) + F(x4 - x5)
     DE/best/1:
         u = xbest + F(x2 - x3)
     DE/best/2:
         u = xbest + F(x1 - x2) + F(x3 - x4)
     DE/current-to-best/1:
         u = x + F(xbest - x1) + F(x2 - x3)
     DE/current-to-pbest/1:
         u = x + F(xp - x1) + F(x2 - x3)
         where xp is a random individual from the top 100*p % of the population based on fitness 
         (p in (0, 1])

https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC05/quin05sad.pdf
SaDE - Self Adaptive DE:
    2 strategies applied for the individuals
    after a learning period (x generations) the application of strategies 
    becomes adaptive through the probability

https://sci-hub.se/10.1109/tevc.2007.894200
ODE - OPPOSITION-BASED DE
    
"""


import numpy as np

obj_func = lambda x: sum(x**2)/len(x)

func_bounds = [(-5,5)]

def crossover(crossover_type, mutant, individual, dimensions, crossover_probability):
    if crossover_type == 'exp':
        L = 0
        j = np.random.randint(low=0, high=dimensions)
        random_number = np.random.rand()
        indexes = []
        while random_number < crossover_probability and L < dimensions:
            L += 1
            j = dimensions % (j+1)
            random_number = np.random.rand()
            indexes.append(j)
    return indexes
            

def mutation(de_type):
    _, individual_selection, n_diff_vectors, crossover_type, = de_type.split("/")
    n_diff_vectors = int(n_diff_vectors)
    


def differential_evolution(obj_func, 
                           func_bounds, 
                           de_type, 
                           mutation_factor=0.8, 
                           crossover_probability=0.7, 
                           population_size=20, 
                           iterations=1000, 
                           runs=1, 
                           verbose=0):
    dimensions = len(func_bounds)
    population_normalized = np.random.rand(d0=population_size, d1=dimensions)
    min_bound, max_bound = np.asarray(func_bounds).T
    dimension_range = np.fabs(min_bound - max_bound)
    population_denormalized = min_bound + population_normalized * dimension_range
    fitness = np.asarray([obj_func(individual) for individual in population_denormalized])
    best_idx = np.argmin(fitness)
    best = population_denormalized[best_idx]
    for i in range(iterations):
        for j in range(population_size):
            idxs = [idx for idx in range(population_size) if idx != j]
            a, b, c = population_normalized[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mutation_factor * (b - c), a_min=0, a_max=1)
            # Crossover
            crossover_vector = np.random.rand(dimensions) < crossover_probability
            offspring_normalized = np.where(condition=crossover_vector, x=mutant, y=population_normalized[j])
            offspring_denormalized = min_bound + offspring_normalized * dimension_range
            offspring_fitness = obj_func(offspring_denormalized)
            if offspring_fitness < fitness[j]:
                fitness[j] = offspring_fitness
                population_normalized[j] = offspring_normalized
                if offspring_fitness < fitness[best_idx]:
                    best_idx = j
                    best = offspring_denormalized
        yield best, fitness[best_idx]
        
        
