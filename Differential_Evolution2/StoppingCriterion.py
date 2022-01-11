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
"""

import numpy as np

class EarlyStopping(object):
    """
    
    """
    
    def __init__(self):
        self.from_nth_gen = 0
        self.patience = 10
        self.tolerance = 1E-10
        self.n_best = 5
        

class ImpBest(object):
    
    def __init__(self):
        
        # super().__init__(from_nth_gen=0, patience=10, tolerance=1E-10, n_best=5)
        
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def MeetStoppingCriterion(self, fitness, best_idx, nth_gen):
        self.metric_list = np.append(self.metric_list, fitness[best_idx])
        
        if nth_gen > self.from_nth_gen:
            check = np.abs(fitness[best_idx] - self.metric_list[-1]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
        if nth_gen >= (self.from_nth_gen + self.patience):
            is_all_under_threshold = self.check_list[-self.patience:].all()            
            return is_all_under_threshold
        else:
            return False
        

        
fitness=np.array([10])     
es = EarlyStopping() 
estop = ImpBestObj(from_nth_gen=0, patience=5, tolerance=1E-10, metric=fitness)

estop = ImpBestObj()
for i in range(10):
    print(estop.MeetStoppingCriterion(nth_gen=i, metric=np.random.rand()))

estop.metric_list
estop.check_list
estop.MeetStoppingCriterion(nth_gen=8, metric=fitness)

estop.metric_list
dir(estop)

class ImpAv(object):
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def MeetStoppingCriterion(self, fitness, nth_gen):
        metric = np.mean(fitness)
        self.metric_list = np.append(self.metric_list, metric)
        
        if nth_gen > self.from_nth_gen:
            check = np.abs(metric - self.metric_list[-1]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
        if nth_gen >= (self.from_nth_gen + self.patience):
            is_all_under_threshold = self.check_list[-self.patience:].all()            
            return is_all_under_threshold
        else:
            return False
        
        

class MovPar(object):
    
    def __init__(self):
        
        # super().__init__(from_nth_gen=0, patience=10, tolerance=1E-10, n_best=5)
        
        self.from_nth_gen = 0
        self.patience = 5
        self.tolerance = 1E-10
        
        # Container
        self.metric_list = np.empty(shape=(0,1))
        self.check_list = np.empty(shape=(0,1))
            
        
    def MeetStoppingCriterion(self, population, best_idx, nth_gen):
        self.metric_list = np.append(self.metric_list, metric)
        
        if nth_gen > self.from_nth_gen:
            check = (metric - self.metric_list[-1]) < self.tolerance
            self.check_list = np.append(self.check_list, check)
        if nth_gen >= (self.from_nth_gen + self.patience):
            is_all_under_threshold = self.check_list[-self.patience:].all()            
            return is_all_under_threshold
        else:
            return False


class MaxDist(object):
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10
                   
    def MeetStoppingCriterion(self, fitness, nth_gen):
        
        if nth_gen >= self.from_nth_gen:
            best_fitness = np.min(fitness)
            worst_fitness = np.max(fitness)
            self.metric = worst_fitness - best_fitness
            return self.metric < self.tolerance        
        else:
            return False
            
class MaxDistPar(object):
    """
    max distance from the vector of best fitness
    """
    
    def __init__(self):
                
        self.from_nth_gen = 0
        self.tolerance = 1E-10
                   
    def MeetStoppingCriterion(self, population, best_idx, nth_gen):
        
        # Container
        diff = np.empty(shape=(0, population.ndim))
        
        for idx in range(len(population)):
            diff[idx] = np.linalg.norm(population[idx] - population[best_idx])
            
        self.metric = np.max(diff)       
        return self.metric < self.tolerance 
    

def StoppingCriterion(criterion):
        
    criteria_func_map = {
        'impbest': ImpBest,
        'impav': ImpAv,
        'movpar': MovPar,
        'maxdist': MaxDist,
        'maxdistpar': MaxDistPar
    }
    
    assert criterion in criteria_func_map.keys(), 'Invalid criterion'
        
    return criteria_func_map[criterion]

StoppingCriterion(criterion='impbest')      
        
        
if __name__ == '__main__':
    m_list = np.asarray([1]*18)
    stopping = SingleObjectiveEarlyStopping()
    stopping.patience = 10
    stopping.MeetStoppingCriterion(metric_list=m_list, nth_gen=12)
    
    
    if nth_gen >= from_nth_gen + patience:
        is_insig_imp = (np.asarray([np.linalg.norm(element-metric_list[-1]) for element in metric_list[-(from_nth_gen + patience):]]) < tolerance).all()            
        return is_insig_imp
    else:
        return False
    
    class SingleObjectiveEarlyStopping(object):
    """
    
    """
    # # Container
    # MetricList = np.empty(shape=(dim_num))
    
    # Store()
    
    def __init__(self):
        self.from_nth_gen = 0
        self.patience = 10
        self.tolerance = 1E-10
        self.n_best = 5
    
        
    def Metric(data, n_best=None, aggr=None):
        """
        Parameters
        ----------
        data : TYPE NUMPY ARRAY
            DESCRIPTION.
        aggr : TYPE, optional
            DESCRIPTION. The default is None. Can be np.mean(), np.std()

        Returns
        -------
        TYPE NUMPY ARRAY
            DESCRIPTION Chosen metric with or without aggregation
        """
        if aggr is not None:
            metric = aggr(data, axis=0)
            return metric
        else:
            return data
        
        
    def Store(self, metric, metric_list):
        metric_list = np.append(metric_list, metric)
    
    def MeetStoppingCriterion(self, metric_list, nth_gen):
        """
        Parameters
        ----------
        metric_list : TYPE NUMPY ARRAY
            DESCRIPTION: list of metric 
        nth_gen : TYPE INTEGER
            DESCRIPTION: number of the actual generation (iteration) number
            
        Returns
        -------
        TYPE BOOLEAN
            DESCRIPTION: Conditions meets the stopping criterion or not.
        """
        if nth_gen >= (self.from_nth_gen + self.patience):
            is_insig_imp = (np.asarray([np.linalg.norm(element-metric_list[-1]) for element in metric_list[-(self.from_nth_gen + self.patience):]]) < self.tolerance).all()            
            return is_insig_imp
        else:
            return False