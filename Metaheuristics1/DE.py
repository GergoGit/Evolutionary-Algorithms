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
https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC05/quin05sad.pdf
    
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
     DE/current-to-rand/2:
         u = x + F(x1 - x2) + F(x3 - x4)
     DE/current-to-best/2:
         u = x + F(xbest - x1) + F(x2 - x3)
     DE/current-to-pbest/2:
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


def sampling(sampling, obj_func, min_bound, dimension_range, population_size, dimensions, n_best):
    if sampling == None:
        population_normalized = np.random.rand(population_size, dimensions)
        population_denormalized = min_bound + population_normalized * dimension_range
        fitness = np.asarray([obj_func(individual) for individual in population_denormalized])
        best_idx = np.argmin(fitness)
        best = population_denormalized[best_idx]
    if sampling == 'opposition-based':
        ob_sampling(population_denormalized, obj_func, min_bound, max_bound, population_size, dimensions, n_best)
    return population_normalized, population_denormalized, fitness
                

def variant_func(individual_selection_type, n_difference_vectors):

    def rand_1(random_individuals, mutation_factor):
        return np.clip(random_individuals[0] + mutation_factor * (random_individuals[1] - random_individuals[2]), a_min=0, a_max=1)
    def rand_2(random_individuals, mutation_factor):
        return np.clip(random_individuals[0] + mutation_factor * (random_individuals[1] - random_individuals[2]) + mutation_factor * (random_individuals[3] - random_individuals[4]), a_min=0, a_max=1)    
    def best_1(random_individuals, best_normalized, mutation_factor):
        return np.clip(best_normalized + mutation_factor * (random_individuals[0] - random_individuals[1]), a_min=0, a_max=1)
    def best_2(random_individuals, best_normalized, mutation_factor):
        return np.clip(best_normalized + mutation_factor * (random_individuals[0] - random_individuals[1]) + mutation_factor * (random_individuals[2] - random_individuals[3]), a_min=0, a_max=1)
    def current_to_best_1(random_individuals, best_normalized, current, mutation_factor):
        return np.clip(current + mutation_factor * (best_normalized - random_individuals[0]), a_min=0, a_max=1)
    def current_to_best_2(random_individuals, best_normalized, current, mutation_factor):
        return np.clip(current + mutation_factor * (best_normalized - random_individuals[0]) + mutation_factor * (random_individuals[1] - random_individuals[2]), a_min=0, a_max=1)    

    if individual_selection_type == 'rand':
        if n_difference_vectors == 1:
            return rand_1
        if n_difference_vectors == 2:
            return rand_2
    if individual_selection_type == 'best':
        if n_difference_vectors == 1:
            return best_1
        if n_difference_vectors == 2:
            return best_2        
    if individual_selection_type == 'current-to-best':
        if n_difference_vectors == 1:
            return current_to_best_1
        if n_difference_vectors == 2:
            return current_to_best_2
        
def input_func(individual_selection_type, n_difference_vectors, population_size, mutation_factor, population_normalized, idx, best_idx):
    idxs = [i for i in range(len(population_normalized)) if i != idx]
    if individual_selection_type == 'rand':
        n_rand = 2 * n_difference_vectors + 1
        random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals, mutation_factor
    if individual_selection_type == 'best':
        n_rand = 2 * n_difference_vectors
        random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals, population_normalized[best_idx], mutation_factor
    if individual_selection_type == 'current-to-best':
        n_rand = 2 * n_difference_vectors - 1
        random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals, population_normalized[best_idx], population_normalized[idx], mutation_factor
    # if individual_selection_type == 'current-to-pbest':
    #     pass


def crossover(crossover_type, dimensions, crossover_probability):
    if crossover_type == 'bin':
        crossover_vector = np.random.rand(dimensions) < crossover_probability
    if crossover_type == 'exp':
        L = 0
        j = np.random.randint(low=0, high=dimensions)
        random_number = np.random.rand()
        crossover_vector = np.asarray([False] * dimensions)
        while random_number < crossover_probability and L < dimensions:
            L += 1
            j = (j+1) % dimensions
            random_number = np.random.rand()
            crossover_vector[j] = True
    return crossover_vector


def differential_evolution(objective_func, 
                           func_bounds, 
                           de_type='DE/best/1/bin', 
                           mutation_factor=0.8, 
                           crossover_probability=0.7, 
                           population_size=30, 
                           generations=1000, 
                           runs=1, 
                           patience=10,
                           epsilon=1E-10,
                           verbose=0):
    
    _, individual_selection_type, n_difference_vectors, crossover_type, = de_type.split("/")
    n_difference_vectors = int(n_difference_vectors)
    
    dimensions = len(func_bounds)
    min_bound = np.asarray([min(dim) for dim in func_bounds])
    max_bound = np.asarray([max(dim) for dim in func_bounds])
    dimension_range = np.fabs(min_bound - max_bound)
    mutation = variant_func(individual_selection_type, n_difference_vectors)
    for run_num in range(runs):
        population_normalized = np.random.rand(population_size, dimensions)
        population_denormalized = min_bound + population_normalized * dimension_range
        fitness = np.asarray([objective_func(individual) for individual in population_denormalized])
        best_idx = np.argmin(fitness)
        best = population_denormalized[best_idx]
        best_list = [best]
        for gen_num in range(generations):
            for idx in range(population_size):
                # Mutation
                mutant = mutation(*input_func(individual_selection_type, n_difference_vectors, mutation_factor, population_normalized, idx, best_idx))
                # Crossover
                crossover_vector = crossover(crossover_type, dimensions, crossover_probability)
                offspring_normalized = np.where(crossover_vector, mutant, population_normalized[idx])                
                offspring_denormalized = min_bound + offspring_normalized * dimension_range
                offspring_fitness = objective_func(offspring_denormalized)
                if offspring_fitness < fitness[idx]:
                    fitness[idx] = offspring_fitness
                    population_normalized[idx] = offspring_normalized
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = idx
                        best = offspring_denormalized            
            
            if patience != None and gen_num >= patience:
                if (np.asarray([element-best for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, gen_num, best, fitness[best_idx]    


def ob_sampling(population_normalized, population_denormalized, objective_func, min_bound, dimension_range, population_size, dimensions):
    opposition_normalized = np.zeros_like(population_normalized)
    for i in range(population_size):
        for j in range(dimensions):
            opposition_normalized[i,j] = 1 - population_normalized[i,j]
    opposition_denormalized = min_bound + opposition_normalized * dimension_range
    population_and_opposition_denormalized = np.concatenate((population_denormalized, opposition_denormalized), axis=0)
    fitness_all = np.asarray([objective_func(individual) for individual in population_and_opposition_denormalized])
    n_best_idx = fitness_all.argsort()[:population_size]
    new_population_normalized = np.concatenate((population_normalized, opposition_normalized), axis=0)[n_best_idx]
    new_population_denormalized = population_and_opposition_denormalized[n_best_idx]
    new_fitness = np.asarray([objective_func(individual) for individual in new_population_denormalized])
    best_idx = np.argmin(new_fitness)
    return new_population_normalized, new_population_denormalized, new_fitness, best_idx, new_population_denormalized[best_idx]


def ob_de(objective_func, 
            func_bounds, 
            de_type='DE/rand/1/bin', 
            mutation_factor=0.8, 
            crossover_probability=0.7, 
            jumping_rate=0.3,
            population_size=30, 
            generations=1000, 
            runs=1, 
            patience=20,
            epsilon=1E-10,
            verbose=0):
    """
    We expect it is a minimization problem.

    Parameters
    ----------
    objection_func : TYPE
        DESCRIPTION.
    func_bounds : TYPE
        DESCRIPTION.
    de_type : TYPE, optional
        DESCRIPTION. The default is 'DE/rand/1/exp'.
    mutation_factor : TYPE, optional
        DESCRIPTION. The default is 0.8.
    crossover_probability : TYPE, optional
        DESCRIPTION. The default is 0.7.
    jumping_rate : TYPE, optional
        DESCRIPTION. The default is 0.3.
    population_size : TYPE, optional
        DESCRIPTION. The default is 30.
    generations : TYPE, optional
        DESCRIPTION. The default is 1000.
    runs : TYPE, optional
        DESCRIPTION. The default is 1.
    patience : TYPE, optional
        DESCRIPTION. The default is 20.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1E-10.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.

    Yields
    ------
    run_num : TYPE
        DESCRIPTION.
    gen_num : TYPE
        DESCRIPTION.
    best : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    _, individual_selection_type, n_difference_vectors, crossover_type, = de_type.split("/")
    n_difference_vectors = int(n_difference_vectors)
    
    dimensions = len(func_bounds)
    min_bound = np.asarray([min(dim) for dim in func_bounds])
    max_bound = np.asarray([max(dim) for dim in func_bounds])
    dimension_range = np.fabs(min_bound - max_bound)
    mutation = variant_func(individual_selection_type, n_difference_vectors)
    for run_num in range(runs):
        population_normalized = np.random.rand(population_size, dimensions)
        population_denormalized = min_bound + population_normalized * dimension_range
        population_normalized, population_denormalized, fitness, best_idx, best = ob_sampling(population_normalized, population_denormalized, obj_func, min_bound, dimension_range, population_size, dimensions)
        best_list = [best]
        for gen_num in range(generations):
            for idx in range(population_size):
                # Mutation
                mutant = mutation(*input_func(individual_selection_type, n_difference_vectors, mutation_factor, population_normalized, idx, best_idx))
                # Crossover
                crossover_vector = crossover(crossover_type, dimensions, crossover_probability)
                offspring_normalized = np.where(crossover_vector, mutant, population_normalized[idx])                
                offspring_denormalized = min_bound + offspring_normalized * dimension_range
                offspring_fitness = obj_func(offspring_denormalized)
                if offspring_fitness < fitness[idx]:
                    fitness[idx] = offspring_fitness
                    population_normalized[idx] = offspring_normalized
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = idx
                        best = offspring_denormalized
            
            if np.random.rand() < jumping_rate:
                population_normalized, population_denormalized, fitness, best_idx, best = ob_sampling(population_normalized, population_denormalized, obj_func, min_bound, dimension_range, population_size, dimensions)
            
            if patience != None and gen_num >= patience:
                if (np.asarray([element-best for element in best_list[-patience:]]) < [epsilon]*dimensions).all():
                    break
            best_list.append(best)
            yield run_num, gen_num, best, fitness[best_idx]
            
def sa_de(objective_func, 
            func_bounds, 
            de_strategy_1='DE/rand/1/bin',
            de_strategy_2='DE/best/1/bin',
            learning_period = 50,
            mutation_factor=0.8, 
            crossover_probability=0.7, 
            population_size=30, 
            generations=1000, 
            runs=1, 
            patience=10,
            epsilon=1E-10,
            verbose=0):
    
    _, individual_selection_type_1, n_difference_vectors_1, crossover_type_1, = de_strategy_1.split("/")
    n_difference_vectors_1 = int(n_difference_vectors_1)
    _, individual_selection_type_2, n_difference_vectors_2, crossover_type_2, = de_strategy_2.split("/")
    n_difference_vectors_2 = int(n_difference_vectors_2)
    
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
        strategy_1_success_num, strategy_2_success_num, strategy_1_failure_num, strategy_2_failure_num = 0, 0, 0, 0
        for gen_num in range(generations):
            if gen_num > learning_period:
                p1 = strategy_1_success_num * (strategy_2_success_num + strategy_2_failure_num)/(strategy_2_success_num * (strategy_1_success_num + strategy_1_failure_num) + strategy_1_success_num * (strategy_2_success_num + strategy_2_failure_num))
            else:
                p1 = 0.5
            for idx in range(population_size):
                if np.random.rand() <= p1:
                    strategy_1_active = True
                    mutation = variant_func(individual_selection_type_1, n_difference_vectors_1)
                    mutant = mutation(*input_func(individual_selection_type_1, n_difference_vectors_1, mutation_factor, population_normalized, idx, best_idx))
                    crossover_vector = crossover(crossover_type_1, dimensions, crossover_probability)
                else:
                    strategy_1_active = False
                    mutation = variant_func(individual_selection_type_2, n_difference_vectors_2)
                    mutant = mutation(*input_func(individual_selection_type_2, n_difference_vectors_2, mutation_factor, population_normalized, idx, best_idx))
                    crossover_vector = crossover(crossover_type_2, dimensions, crossover_probability)
                offspring_normalized = np.where(crossover_vector, mutant, population_normalized[idx])                
                offspring_denormalized = min_bound + offspring_normalized * dimension_range
                offspring_fitness = objective_func(offspring_denormalized)
                if offspring_fitness < fitness[idx]:
                    fitness[idx] = offspring_fitness
                    population_normalized[idx] = offspring_normalized
                    if strategy_1_active:
                        strategy_1_success_num =+ 1
                    else:
                        strategy_2_success_num =+ 1
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = idx
                        best = offspring_denormalized
                else:
                    if strategy_1_active:
                        strategy_1_failure_num =+ 1
                    else:
                        strategy_2_failure_num =+ 1
            
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
    result = list(differential_evolution(objective_func=obj_func, func_bounds=func_bounds, de_type='DE/rand/1/bin', runs=runs, generations=300, patience=20))
    result = list(ob_de(objective_func=obj_func, func_bounds=func_bounds, runs=runs, generations=300))
    result = list(sa_de(objective_func=obj_func, func_bounds=func_bounds, runs=runs, generations=300,
                        de_strategy_1='DE/rand/1/bin',
                        de_strategy_2='DE/best/1/bin',
                        learning_period = 50,
                        patience=20)
                  )    
    
    for run_num in range(runs):
        run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
        plt.yscale('log', base=2) 
        plt.plot(f, label='run_num={}'.format(run_num))
    plt.legend()
    
    result[-1]
    result[-1][0]
    
    run_num = 0

    run, gen, x, f = zip(*result)
    run, gen, x, f = zip(*[element for element in result if element[0]==run_num])
    plt.yscale('log', base=2) 
    plt.plot(f)
    

    
    
    # class mutation:
    # def __init__(self,
    #              individual_selection_type,
    #              n_difference_vectors,
    #              mutation_factor
    #              ):
    #     self.individual_selection_type = individual_selection_type
    #     self.n_difference_vectors = n_difference_vectors
    #     self.mutation_factor = mutation_factor
        
    # def rand_1(self, random_individuals):
    #     return np.clip(random_individuals[0] + self.mutation_factor * (random_individuals[1] - random_individuals[2]), a_min=0, a_max=1)
    
    # def rand_2(self, random_individuals):
    #     return np.clip(random_individuals[0] + self.mutation_factor * (random_individuals[1] - random_individuals[2]) + self.mutation_factor * (random_individuals[3] - random_individuals[4]), a_min=0, a_max=1)
    
    # def best_1(self, random_individuals, best):
    #     return np.clip(best + self.mutation_factor * (random_individuals[0] - random_individuals[1]), a_min=0, a_max=1)
    
    # def best_2(self, random_individuals, best):
    #     return np.clip(best + self.mutation_factor * (random_individuals[0] - random_individuals[1]) + self.mutation_factor * (random_individuals[2] - random_individuals[3]), a_min=0, a_max=1)
    
    # def current_to_best_1(self, random_individuals, best, current):
    #     return np.clip(current + self.mutation_factor * (best - random_individuals[0]), a_min=0, a_max=1)

    # def current_to_best_2(self, random_individuals, best, current):
    #     return np.clip(current + self.mutation_factor * (best - random_individuals[0]) + self.mutation_factor * (random_individuals[1] - random_individuals[2]), a_min=0, a_max=1)    
    
    # def variant_func(self, population_normalized, idxs, best, current):
    #     if self.individual_selection_type == 'rand':
    #         n_rand = 2 * self.n_difference_vectors + 1
    #         random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
    #         if self.n_difference_vectors == 1:
    #             return rand_1(self, random_individuals)
    #         if self.n_difference_vectors == 2:
    #             return rand_2
    #     if self.individual_selection_type == 'best':
    #         n_rand = 2 * self.n_difference_vectors
    #         random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
    #         if self.n_difference_vectors == 1:
    #             return best_1
    #         if self.n_difference_vectors == 2:
    #             return best_2
    #     if self.individual_selection_type == 'current-to-best':
    #         n_rand = 2 * self.n_difference_vectors - 1
    #         random_individuals = population_normalized[np.random.choice(idxs, n_rand, replace = False)]
    #         if self.n_difference_vectors == 1:
    #             return current_to_best_1
    #         if self.n_difference_vectors == 2:
    #             return current_to_best_2