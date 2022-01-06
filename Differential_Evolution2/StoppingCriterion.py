# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:14:02 2022

@author: bonnyaigergo

https://www.researchgate.net/post/What-is-stopping-criteria-of-any-optimization-algorithm
https://www.researchgate.net/post/What-is-stopping-criteria-of-any-optimization-algorithm/588c67c0ed99e1ac8e10fe93/citation/download

metrics often used:
    
Improvement-based:
    ImpBestObj:     best_fitness change < tol in time_window(patience)
    ImpBestPar:     best_individual change < tol in time_window(patience)
    ImpAvObj:       avg_fitness change < tol in time_window(patience)
Distribution-based:
    MaxDist:        max distance of all individual from best individual < tol
    MaxDistQuick:   max distance of pbest individual from best individual < tol
    StdDev:         stdev of pbest (or all) individual < tol
    Diff:           difference of best and worst objective
"""
import numpy as np

class SingleObjectiveEarlyStopping(object):
    """
    
    """
    def __init__(self):
        self.from_nth_gen = 0
        self.patience = 10
        self.tolerance = 1E-10
        
    def Metric(data, pbest=None, aggr=None):
        """
        Parameters
        ----------
        data : TYPE NUMPY ARRAY
            DESCRIPTION.
        aggr : TYPE, optional
            DESCRIPTION. The default is None. Can be np.mean()

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
        np.append(metric_list, metric)
    
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
    