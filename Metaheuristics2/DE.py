# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:59:33 2021

@author: bonnyaigergo

article:
https://arxiv.org/pdf/2101.06599.pdf
https://www.researchgate.net/publication/220403311_A_comparative_study_of_crossover_in_differential_evolution
https://sci-hub.se/10.1016/j.swevo.2012.09.004
https://sci-hub.se/10.1109/nafips.1996.534790
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

# TODO: stopping criterion, Visualization, mutation factor decay, parallelization

import numpy as np
import StoppingCriterion

class VariantRandom:
    
    def __init__(self, n_difference_vectors, mutation_factor):
        self.n_difference_vectors = n_difference_vectors
        self.mutation_factor = mutation_factor
    
    def rand_1(self, random_individuals):
        mutant = random_individuals[0] + self.mutation_factor * (random_individuals[1] - random_individuals[2])
        return mutant
    def rand_2(self, random_individuals):
        mutant = random_individuals[0] + self.mutation_factor * (random_individuals[1] - random_individuals[2]) + self.mutation_factor * (random_individuals[3] - random_individuals[4])   
        return mutant
    
    def subvariant_fn(self):
        if self.n_difference_vectors == 1:
            return self.rand_1
        if self.n_difference_vectors == 2:
            return self.rand_2
    
    def subvariant_fn_args(self, population, indiv_idx, best_idx):
        idxs = [i for i in range(len(population)) if i != indiv_idx]
        n_rand = 2 * self.n_difference_vectors + 1
        random_individuals = population[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals
        
class VariantBest:
    
    def __init__(self, n_difference_vectors, mutation_factor):
        self.n_difference_vectors = n_difference_vectors
        self.mutation_factor = mutation_factor
    
    def best_1(self, random_individuals, best):
        mutant = best + self.mutation_factor * (random_individuals[0] - random_individuals[1])
        return mutant
    def best_2(self, random_individuals, best):
        mutant =  best + self.mutation_factor * (random_individuals[0] - random_individuals[1]) + self.mutation_factor * (random_individuals[2] - random_individuals[3])
        return mutant
    
    def subvariant_fn(self):
        if self.n_difference_vectors == 1:
            return self.best_1
        if self.n_difference_vectors == 2:
            return self.best_2
    
    def subvariant_fn_args(self, population, indiv_idx, best_idx):
        idxs = [i for i in range(len(population)) if i != indiv_idx]
        n_rand = 2 * self.n_difference_vectors
        random_individuals = population[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals, population[best_idx]
    
class VariantCurrentToBest:
    
    def __init__(self, n_difference_vectors: int, mutation_factor: float):
        self.n_difference_vectors = n_difference_vectors
        self.mutation_factor = mutation_factor
    
    def current_to_best_1(self, random_individuals: float, best: int, current: int):
        mutant = current + self.mutation_factor * (best - random_individuals[0])
        return mutant
    def current_to_best_2(self, random_individuals, best, current):
        mutant = current + self.mutation_factor * (best - random_individuals[0]) + self.mutation_factor * (random_individuals[1] - random_individuals[2])
        return mutant
    
    def subvariant_fn(self):
        if self.n_difference_vectors == 1:
            return self.current_to_best_1
        if self.n_difference_vectors == 2:
            return self.current_to_best_2
    
    def subvariant_fn_args(self, population: float, indiv_idx: int, best_idx: int):
        idxs = [i for i in range(len(population)) if i != indiv_idx]
        n_rand = 2 * self.n_difference_vectors - 1
        random_individuals = population[np.random.choice(idxs, n_rand, replace = False)]
        return random_individuals, population[best_idx], population[indiv_idx]
            

def select_mutation_variant(individual_selection_type: str, 
                            n_difference_vectors: int, 
                            mutation_factor: float):
    
    if individual_selection_type == 'rand':
        return VariantRandom(n_difference_vectors, mutation_factor)
    if individual_selection_type == 'best':
        return VariantBest(n_difference_vectors, mutation_factor)
    if individual_selection_type == 'current-to-best':
        return VariantCurrentToBest(n_difference_vectors, mutation_factor)




class DE(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 de_type='DE/best/1/bin', 
                 mutation_factor=0.8, 
                 crossover_prob=0.7, 
                 population_size=30, 
                 n_generation=300):
        
                
        if stopping_criterion is not None:
            self.termination = StoppingCriterion.criteria_fn_map(stopping_criterion)()
        else:
            self.termination = None
        
        self.objective = objective
        self.obj_fn = objective.evaluate
        self.search_space = self.objective.search_space
        
        self.min_bound = np.asarray([min(dim) for dim in self.search_space])
        self.max_bound = np.asarray([max(dim) for dim in self.search_space])
        self.dim_range = np.fabs(self.min_bound - self.max_bound)
        self.n_dim = objective.n_dim
        
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.population_size = population_size
        self.n_generation = n_generation
        
        _, self.individual_selection_type, n_difference_vectors, self.crossover_type, = de_type.split("/")
        self.n_difference_vectors = int(n_difference_vectors)
        
        self.M = select_mutation_variant(self.individual_selection_type,
                                        self.n_difference_vectors,
                                        self.mutation_factor)
        
        self.mutation = self.M.subvariant_fn()
               
        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
        
    def crossover(self, mutant: float, individual: float):
        if self.crossover_type == 'bin':
            crossover_vector = np.random.rand(self.n_dim) < self.crossover_prob
        if self.crossover_type == 'exp':
            L = 0
            j = np.random.randint(low=0, high=self.n_dim)
            random_number = np.random.rand()
            crossover_vector = np.asarray([False] * self.n_dim)
            while random_number < self.crossover_prob and L < self.n_dim:
                L += 1
                j = (j+1) % self.n_dim
                random_number = np.random.rand()
                crossover_vector[j] = True
        offspring = np.where(crossover_vector, mutant, individual)
        return offspring
    
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
    
        
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        for nth_gen in range(self.n_generation):
            
            for indiv_idx in range(self.population_size): 
                
                mutant = self.mutation(*self.M.subvariant_fn_args(population, indiv_idx, best_idx))
                mutant = self.check_search_space(mutant)
                offspring = self.crossover(mutant, population[indiv_idx])
                offspring_fitness = self.obj_fn(offspring)
                if offspring_fitness < fitness[indiv_idx]:
                    fitness[indiv_idx] = offspring_fitness
                    population[indiv_idx] = offspring
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = indiv_idx
                        best = offspring
                        
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break


class OBDE(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 de_type='DE/best/1/bin', 
                 mutation_factor=0.8, 
                 crossover_prob=0.7,
                 jumping_rate=0.3,
                 population_size=30, 
                 n_generation=300):
        
                
        if stopping_criterion is not None:
            self.termination = StoppingCriterion.criteria_fn_map(stopping_criterion)()
        else:
            self.termination = None
        
        self.objective = objective
        self.obj_fn = objective.evaluate
        self.search_space = self.objective.search_space
        
        self.min_bound = np.asarray([min(dim) for dim in self.search_space])
        self.max_bound = np.asarray([max(dim) for dim in self.search_space])
        self.dim_range = np.fabs(self.min_bound - self.max_bound)
        self.n_dim = objective.n_dim
        
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.jumping_rate = jumping_rate
        self.population_size = population_size
        self.n_generation = n_generation
        
        _, self.individual_selection_type, n_difference_vectors, self.crossover_type, = de_type.split("/")
        self.n_difference_vectors = int(n_difference_vectors)
        
        self.M = select_mutation_variant(self.individual_selection_type,
                                        self.n_difference_vectors,
                                        self.mutation_factor)
        
        self.mutation = self.M.subvariant_fn()
               
        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.n_dim) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
        
    def crossover(self, mutant: float, individual: float):
        if self.crossover_type == 'bin':
            crossover_vector = np.random.rand(self.n_dim) < self.crossover_prob
        if self.crossover_type == 'exp':
            L = 0
            j = np.random.randint(low=0, high=self.n_dim)
            random_number = np.random.rand()
            crossover_vector = np.asarray([False] * self.n_dim)
            while random_number < self.crossover_prob and L < self.n_dim:
                L += 1
                j = (j+1) % self.n_dim
                random_number = np.random.rand()
                crossover_vector[j] = True
        offspring = np.where(crossover_vector, mutant, individual)
        return offspring
    
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
    
    def ob_sampling(self, population):
        """
        Opposition-Based (OB) sampling
    
        """
        opposition = np.zeros_like(population)
        for idx in range(self.population_size):
            for dim in range(self.n_dim):
                opposition[idx,dim] = self.max_bound[dim] - np.abs(self.min_bound[dim] - population[idx,dim])
        population_and_opposition = np.concatenate((population, opposition), axis=0)
        fitness_all = np.asarray([self.obj_fn(individual) for individual in population_and_opposition])
        n_best_idx = fitness_all.argsort()[:self.population_size]
        new_population = population_and_opposition[n_best_idx]
        new_fitness = np.asarray([self.obj_fn(individual) for individual in new_population])
        best_idx = np.argmin(new_fitness)
        return new_population, new_fitness, best_idx, new_population[best_idx]
        
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        self.best_par = np.empty(shape=(0, self.n_dim))
        self.best_fitness = np.empty(shape=(0, 1))
        
        for nth_gen in range(self.n_generation):
            
            for indiv_idx in range(self.population_size): 
                
                mutant = self.mutation(*self.M.subvariant_fn_args(population, indiv_idx, best_idx))
                mutant = self.check_search_space(mutant)
                offspring = self.crossover(mutant, population[indiv_idx])
                offspring_fitness = self.obj_fn(offspring)
                if offspring_fitness < fitness[indiv_idx]:
                    fitness[indiv_idx] = offspring_fitness
                    population[indiv_idx] = offspring
                    if offspring_fitness < fitness[best_idx]:
                        best_idx = indiv_idx
                        best = offspring
                        
            if np.random.rand() < self.jumping_rate:
                population, fitness, best_idx, best = self.ob_sampling(population)
                
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break
            


if __name__ == "__main__":
    
    # M = select_mutation_variant(individual_selection_type = 'rand', 
    #                                     n_difference_vectors = 1, 
    #                                     mutation_factor = 0.8)
    
    # mutation = M.subvariant_fn()
    # mutation(*M.subvariant_fn_args(population, indiv_idx, best_idx))
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin, two_equations_two_unknown
    from StoppingCriterion import ImpBestObj
    # fn = Rastrigin(2)
    fn = two_equations_two_unknown()
    # optimiser = OBDE(objective=fn, stopping_criterion='imp_best_obj', de_type='DE/best/2/bin')
    optimiser = DE(objective=fn, stopping_criterion='imp_best_obj', de_type='DE/best/1/bin')
    optimiser.termination.from_nth_gen = 50
    optimiser.termination.patience = 30
    optimiser.run()
    
    plt.yscale('log', base=2) 
    plt.plot(optimiser.best_fitness)
    plt.legend()
    
    # optimiser.n_dim
    print(optimiser.best_par[-1])
    # optimiser.best_fitness
    # optimiser.termination.metric_list
    # optimiser.termination.check_list
    # fn.minima_loc
    # fn.minima
    
    

