# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:17:15 2022

@author: bonnyaigergo

Electro-Search Optimization (ESO)

https://www.researchgate.net/profile/Mdfadil-Mdesa/publication/329103221_A_Hybrid_Algorithm_Based_on_Flower_Pollination_Algorithm_and_Electro_Search_for_Global_Optimization/links/5dc8eab392851c81804360e8/A-Hybrid-Algorithm-Based-on-Flower-Pollination-Algorithm-and-Electro-Search-for-Global-Optimization.pdf
https://sci-hub.se/10.1016/j.compchemeng.2017.01.046
https://sci-hub.se/10.1109/access.2019.2894857

population members are nucleuses of atoms
D is the relocation distance
Re is Rydberg’s energy constant-coefficient
Ac is the accelerator coefficient
n is the energy level (vicinity in which electrons can be positioned) n ∈ {2, 3, 4, 5}
r is the orbital radius (average distance)
"""
# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Evolutionary-Algorithms\Differential_Evolution2')

import numpy as np
import StoppingCriterion

class ESO(object):
    def __init__(self, 
                 objective, 
                 stopping_criterion=None,
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
        
        self.population_size = population_size
        self.generation_num = generation_num

        
    def initialize_nucleus_population(self):
        population = self.min_bound + np.random.rand(self.population_size, self.dim_num) * self.dim_range
        fitness = np.asarray([self.obj_fn(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        return population, fitness, best_idx, best
        
    def initialize_electrons(self, nucleus):
        electron = nucleus + np.random.rand(self.population_size, self.dim_num)/10
        electron = self.check_search_space(electron) # TODO check
        electron_fitness = np.asarray([self.obj_fn(individual) for individual in electron])
        best_electron_idx = np.argmin(electron_fitness)
        best_electron = electron[best_electron_idx]
        return electron, electron_fitness, best_electron_idx, best_electron
    
    def check_search_space(self, mutant: float):
        mutant = np.clip(mutant, a_min=self.min_bound, a_max=self.max_bound)
        return mutant
            
    def run(self):
        nucleus, nucleus_fitness, best_nucleus_idx, best_nucleus = self.initialize_population()

        Ac
        Re
        orbital_radius


        self.best_par = np.empty(shape=(0, self.dim_num))
        self.best_fitness = np.empty(shape=(0, 1))
        
        for nth_gen in range(self.generation_num):
            
            for indiv_idx in range(self.population_size): 
                r1 = np.random.rand(1)
                
                electron = population[indiv_idx] + (2*r1 - 1)(1 - 1/n**2)*r
                D = [indiv_idx]
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
                        
            self.best_par = np.append(self.best_par, best)
            self.best_fitness = np.append(self.best_fitness, fitness[best_idx])
            if self.termination is not None:
                if self.termination.meet_criterion(population, fitness, best_idx, nth_gen):
                    break
            orbital_radius = np.mean(np.linalg.norm(nucleus - electron))
            AC
            Re
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