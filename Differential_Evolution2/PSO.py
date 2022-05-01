# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:42:13 2021

@author: bonnyaigergo

Particle Swarm Optimization

https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
https://github.com/anyoptimization/pymoo/blob/master/pymoo/algorithms/soo/nonconvex/pso.py

alfa = inertia
c1 and c2: cognitive and social parameters

NOTE: works, but with not continous improvement
"""

import numpy as np
import StoppingCriterion

class PSO(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
                 c1=2,
                 c2=2,
                 alfa=0.5, 
                 decay=False,
                 alfa_min=0,
                 alfa_max=1,
                 population_size=30, 
                 generation_num=300):
        
                
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
        self.dim_num = objective.dim_num
        
        self.c1=c1
        self.c2=c2
        self.alfa=alfa
        self.decay=decay
        self.alfa_min=alfa_min
        self.alfa_max=alfa_max
        self.population_size = population_size
        self.generation_num = generation_num

        
    def initialize_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.dim_num) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
    
    
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
    
        
    def run(self):
        population, fitness, best_idx, best = self.initialize_population()
        particle_indiv_best = population.copy()
        particle_fitness = fitness.copy()
        velocity = np.zeros(shape=(self.population_size, self.dim_num))

        self.best_par = np.empty(shape=(0, self.dim_num))
        self.best_fitness = np.empty(shape=(0, 1))
        
        for nth_gen in range(self.generation_num):
            for indiv_idx in range(self.population_size): 
                r1, r2 = np.random.rand(2)
                velocity_cognitive = self.c1*r1*(particle_indiv_best[indiv_idx] - population[indiv_idx])
                velocity_social = self.c2*r2*(best - population[indiv_idx])
                velocity[indiv_idx] = self.alfa*velocity[indiv_idx] + velocity_cognitive + velocity_social
                population[indiv_idx] = population[indiv_idx] + velocity[indiv_idx]
                population[indiv_idx] = self.check_search_space(population[indiv_idx])                
                fitness[indiv_idx] = self.obj_fn(population[indiv_idx])
                
                if fitness[indiv_idx] < particle_fitness[indiv_idx]:
                    particle_indiv_best[indiv_idx] = population[indiv_idx]
                    if fitness[indiv_idx] < fitness[best_idx]:
                        best = population[indiv_idx]
                        best_idx = indiv_idx
                        
            self.best_par = np.vstack((self.best_par, best))
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break
            if self.decay:
                self.alfa = (self.alfa_max - self.alfa_min) * ((nth_gen + 1) / self.generation_num)


                
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ObjectiveFunctions import Beale, Rastrigin
    from StoppingCriterion import ImpBestObj
    fn = Rastrigin(2)  
    optimiser = PSO(objective=fn, stopping_criterion='imp_avg_par')
    optimiser.termination.from_nth_gen = 50
    optimiser.termination.patience = 20
    optimiser.run()
    
    plt.yscale('log', base=2) 
    plt.plot(optimiser.best_fitness)
    plt.legend()    
    
    # optimiser.dim_num
    # optimiser.best_fitness
    # optimiser.termination.metric_list
    # optimiser.termination.check_list
    # fn.minima_loc
    # fn.minima