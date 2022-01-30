# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:14:02 2022

@author: bonnyaigergo

https://www.researchgate.net/post/What-is-stopping-criteria-of-any-optimization-algorithm
https://www.researchgate.net/post/What-is-stopping-criteria-of-any-optimization-algorithm/588c67c0ed99e1ac8e10fe93/citation/download
https://pymoo.org/interface/termination.html

metrics often used:
    
Improvement-based:
    ImpBestObj:     best_fitness change < tol in time_window(patience)
    ImpBestPar:     best_individual change < tol in time_window(patience)
    ImpAvObj:       avg_fitness change < tol in time_window(patience)
Distribution-based:
    MaxDist:        max distance of all individual from best individual < tol
    MaxDistQuick:   max distance of n_best individual from best individual < tol
    StdDev:         stdev of pbest (or all) individual < tol
    Diff:           difference of best and worst objective
    
    
function map
metric(data)
condition

TODO: combine criteria
"""

import numpy as np
# import typing
from abc import ABC, abstractmethod

class StoppingCriterion(ABC):
    
    @abstractmethod
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int):
        pass
    


########################################
# Improvement-based using Time Window
########################################

class ImpBestObj(StoppingCriterion):
    """
    Checks whether the improvement of the best objective function value is below 
    a threshold ('tolerance') for a number of generations ('patience'). 
    'from_nth_gen' parameter means an initial number of
    generations without checking the improvement.
    """    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool: 
        
        if nth_gen >= self.from_nth_gen:
            self.metric_list = np.append(self.metric_list, fitness[best_idx])
            check = np.abs(fitness[best_idx] - self.metric_list[-1]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
            if nth_gen >= (self.from_nth_gen + self.patience - 1):
                is_all_under_threshold = self.check_list[-self.patience:].all()            
                return is_all_under_threshold
            else:
                return False
        else:
            return False
        

        
# fitness=np.array([10])     
# es = EarlyStopping() 
# estop = ImpBestObj(from_nth_gen=0, patience=5, tolerance=1E-10, metric=fitness)

# estop = ImpBest()
# for i in range(10):
#     print(estop.MeetStoppingCriterion(nth_gen=i, metric=np.random.rand()))

# estop.metric_list
# estop.check_list
# estop.MeetStoppingCriterion(nth_gen=8, metric=fitness)

# estop.metric_list
# dir(estop)

class ImpAvgObj(StoppingCriterion):
    """
    Checks whether the improvement of the average objective function value is below 
    a threshold ('tolerance') for a number of generations ('patience'). 
    'from_nth_gen' parameter means an initial number of
    generations without checking the improvement.
    """  
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
        
        if nth_gen > self.from_nth_gen:
            metric = np.mean(fitness)
            self.metric_list = np.append(self.metric_list, metric)
            check = np.abs(self.metric_list[-1] - self.metric_list[-2]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
            if nth_gen >= (self.from_nth_gen + self.patience):
                is_all_under_threshold = self.check_list[-self.patience:].all()            
                return is_all_under_threshold
            else:
                return False
        else:
            return False
        
# estop = ImpAv()
# for i in range(10):
#     print(estop.MeetStoppingCriterion(nth_gen=i, fitness=np.random.rand()))       
    
# estop.metric_list
# estop.check_list

class ImpAvgPar(StoppingCriterion):
    """
    Checks whether the improvement of the average objective function value is below 
    a threshold ('tolerance') for a number of generations ('patience'). 
    'from_nth_gen' parameter means an initial number of
    generations without checking the improvement.
    """ 
    
    def __init__(self):
        
        # super().__init__(from_nth_gen=0, patience=10, tolerance=1E-10, n_best=5)
        
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
        
        if nth_gen > self.from_nth_gen:
            # Container
            diff = np.empty(shape=(0, population.ndim))
            
            for idx in range(len(population)):
                diff[idx] = np.linalg.norm(population[idx] - population[best_idx])
                
            metric = np.mean(diff)
            self.metric_list = np.append(self.metric_list, metric)
            check = np.abs(self.metric_list[-1] - self.metric_list[-2]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
            if nth_gen >= (self.from_nth_gen + self.patience):
                is_all_under_threshold = self.check_list[-self.patience:].all()            
                return is_all_under_threshold
            else:
                return False
        else:
            return False

##########################################################
# Distribusion-based using metric related to 1 generation
##########################################################

class MaxDistObj(StoppingCriterion):    
    """
    Checks whether the max distance (objective space) of the individuals within the population 
    is under a given threshold ('tolerance') from the individual of best fitness in a given generation.
    'from_nth_gen' parameter means an initial number of generations without checking the improvement.
    """
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10

        # Container
        self.metric_list = np.empty(shape=(0,1))
                   
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
        
        if nth_gen >= self.from_nth_gen:
            best_fitness = np.min(fitness)
            worst_fitness = np.max(fitness)
            metric = worst_fitness - best_fitness
            self.metric_list = np.append(self.metric_list, metric)
            return metric < self.tolerance        
        else:
            return False
        
# estop = MaxDist()
# for i in range(10):
#     print(estop.MeetStoppingCriterion(nth_gen=i, fitness=np.random.uniform(size=10)))       
    
# estop.metric_list

class MaxDistQuickObj(StoppingCriterion):    
    """
    Checks whether the max distance (objective space) of the 'p_best' % (min 3) best individuals 
    within the population is under a given threshold ('tolerance') from the individual of best fitness 
    in a given generation.
    'from_nth_gen' parameter means an initial number of generations without checking the improvement.
    """
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10
        self.p_best = 0.2

        # Container
        self.metric_list = np.empty(shape=(0,1))
                   
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
        
        if nth_gen >= self.from_nth_gen:
            n_individuals = len(fitness)
            if n_individuals < 3:
                raise ValueError('Number of individuals in a generation should be higher than or equal to 3')
            # assert n_individuals >= 3, 'Number of individuals in a generation should be higher than 3'
            k = np.round(n_individuals * self.p_best)
            k = max(k, 3)
            k_best_idx = np.argsort(fitness)[:k]
            # Container
            diff = np.empty(shape=(0,1))
            for idx in k_best_idx:
                diff = np.append(diff, fitness[idx] - fitness[best_idx])
            metric = np.max(diff)
            self.metric_list = np.append(self.metric_list, metric)
            return metric < self.tolerance        
        else:
            return False


class AvgDistObj(StoppingCriterion):    
    """
    Checks whether the average distance (objective space) of the individuals within the population  
    from the individual of best fitness in a given generation is under a given threshold ('tolerance') 
    'from_nth_gen' parameter means an initial number of generations without checking the improvement.
    """
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10

        # Container
        self.metric_list = np.empty(shape=(0,1))
                   
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
                
        if nth_gen >= self.from_nth_gen:
            # Container
            diff = np.empty(shape=(0,1))
            
            for idx in range(len(fitness)):
                diff = np.append(diff, fitness[idx] - fitness[best_idx])
                
            metric = np.mean(diff)
            self.metric_list = np.append(self.metric_list, metric)
            return metric < self.tolerance        
        else:
            return False
    
    

class AvgDistPar(StoppingCriterion):
    """
    Checks whether the max (Euclidean) distance (objective space) of the individuals within the population 
    is under a given threshold ('tolerance') from the individual of best fitness in a given generation.
    'from_nth_gen' parameter means an initial number of generations without checking the improvement.
    """
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))        
                   
    def meet_criterion(self, population: float, fitness: float, best_idx: int, nth_gen: int) -> bool:
        
        if nth_gen >= self.from_nth_gen:
            # Container
            diff = np.empty(shape=(0, population.ndim))
            
            for idx in range(len(population)):
                diff = np.append(diff, np.linalg.norm(population[idx] - population[best_idx]))
                
            metric = np.mean(diff)
            self.metric_list = np.append(self.metric_list, metric)
            return metric < self.tolerance
        else:
            return False
    
# estop = MaxDistPar()
# for i in range(10):
#     print(estop.MeetStoppingCriterion(nth_gen=i, best_idx=1, population=np.random.uniform(size=(4,5))))       
    
# estop.metric_list


# def StoppingCriterion(criterion):
        
#     criteria_fn_map = {
#         'imp_best_obj': ImpBestObj,
#         'imp_avg_obj': ImpAvgObj,
#         'imp_avg_par': ImpAvgPar,
#         'max_dist_obj': MaxDistObj,
#         'max_distquick_obj': MaxDistQuickObj,
#         'avg_dist_obj': AvgDistObj,
#         'avg_dist_par': AvgDistPar
#     }
    
#     assert criterion in criteria_fn_map.keys(), 'Invalid criterion'
        
#     return criteria_fn_map[criterion]


# sc = StoppingCriterion(criterion='impbest')




    
    
def criteria_fn_map(criterion):
    
    stopping_criterion_fn_map = {
        'imp_best_obj': ImpBestObj,
        'imp_avg_obj': ImpAvgObj,
        'imp_avg_par': ImpAvgPar,
        'max_dist_obj': MaxDistObj,
        'max_distquick_obj': MaxDistQuickObj,
        'avg_dist_obj': AvgDistObj,
        'avg_dist_par': AvgDistPar
    }
            
    assert criterion in stopping_criterion_fn_map.keys(), 'Invalid criterion'
        
    return stopping_criterion_fn_map[criterion]



        
if __name__ == '__main__':
    # m_list = np.asarray([1]*18)
    # stopping = SingleObjectiveEarlyStopping()
    # stopping.patience = 10
    # stopping.MeetStoppingCriterion(metric_list=m_list, nth_gen=12)
    termination = criteria_fn_map('imp_best_obj')()
    termination.meet_criterion(population, fitness, best_idx, nth_gen)
    pass

import json
import os

stopping_criterion_fn_dictionary = {
        'imp_best_obj': 
            {'fn': ImpBestObj, 'args': [fitness, best_idx, nth_gen]},
        'imp_avg_obj': 
            {'fn': ImpAvgObj, 'args': [fitness, nth_gen]},
        'imp_avg_par': 
            {'fn': ImpAvgPar, 'args': [population, best_idx, nth_gen]},
        'max_dist_obj': 
            {'fn': MaxDistObj, 'args': [fitness, nth_gen]},
        'max_distquick_obj': 
            {'fn': MaxDistQuickObj, 'args': [fitness, best_idx, nth_gen]},
        'avg_dist_obj': 
            {'fn': AvgDistObj, 'args': [fitness, best_idx, nth_gen]},
        'avg_dist_par': 
            {'fn': AvgDistPar, 'args': [population, best_idx, nth_gen]}
    }
    
    
source_loc = os.getcwd() + '/'

with open(source_loc + 'stopping_criterion_fn_dictionary.json', "w") as outfile:
    json.dump(stopping_criterion_fn_dictionary, outfile)
