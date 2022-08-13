# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:26:24 2021

@author: bonnyaigergo

https://en.wikipedia.org/wiki/Test_functions_for_optimization
https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
https://github.com/anyoptimization/pymoo/tree/master/pymoo/problems/single
https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

"""

import numpy as np

# ackley multimodal function
class Ackley(object):
    def __init__(self):
        self.name = "Ackley function"
        self.minima = 0.0
        self.minima_loc = np.array([0.0, 0.0], dtype=np.float32)
        self.search_space = np.array([(-5, 5)]*2, dtype=np.float32)
        self.any_dim = False
        
    def evaluate(self, x, y):
        a = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (np.power(x, 2) + np.power(y, 2))))
        b = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20.0
        z = a - b
        return z

    
class Rastrigin(object):
    def __init__(self, n_dim, A=10):
        self.name = "Rastrigin function"
        self.search_space = np.array([(-5.12, 5.12)] * n_dim, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0] * n_dim, dtype=np.float32)
        self.A = A
        self.n_dim = n_dim
        self.any_dim = True
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.n_dim + np.sum(z)
        return z
    
class two_equations_two_unknown(object):
    """
    equations:
     2*x1 + 1*x2 = 7
    -6*x1 + 2*x2 = 4
    solution:
    x1=1
    x2=5
    """
    def __init__(self):
        self.search_space = np.array([(-100, 100)], dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([1, 5], dtype=np.float32)
        self.n_dim = 2
        
    def evaluate(self, x):
        coef = np.array([[2, 1], [-6, 2]])
        z = np.dot(coef, x)
        loss = np.sum(np.abs(np.array([7, 4], dtype=np.float32) - z))
        return loss
    

# TODO: correct Michalewicz    
class Michalewicz(object):
    def __init__(self, n_dim, m=10):
        """
        Minima and location is valid in case of 2 dimensions
        """
        self.name = "Michalewicz function"
        self.search_space = np.array([(0, np.pi)] * n_dim, dtype=np.float32)
        self.minima = -1.8013
        self.minima_loc = np.array([2.2, 1.57], dtype=np.float32)
        self.m = m
        self.n_dim = n_dim
        self.any_dim = True
        
    def evaluate(self, x):
        i = np.arange(1, self.n_dim+1, 1)
        z = -np.sum(np.sin(x)*np.sin(np.power(i*np.power(x, 2)/np.pi, 2*self.m)))
        return z
    
class Salomon(object):
    def __init__(self, n_dim):
        self.name = "Salomon function"
        self.search_space = np.array([(-3, 3)] * n_dim, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0.0] * n_dim, dtype=np.float32)
        self.n_dim = n_dim
        self.any_dim = True
        
    def evaluate(self, x):
        z = 1-np.cos(2*np.pi*np.sqrt(np.sum(np.power(x, 2))))+0.1*np.sqrt(np.sum(np.power(x, 2)))
        return z

class Himmelblau(object):
    def __init__(self):
        self.name = "Himmelblau function"
        self.search_space = np.array([(-5, 5)] * 2, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([(3, 2), 
                                    (-2.805118, 3.131312), 
                                    (-3.779310, -3.283186), 
                                    (3.584428, -1.848126)], dtype=np.float32)
        self.any_dim = False
        
    def evaluate(self, x, y):
        z = np.power(np.power(x, 2) + y - 11, 2) + np.power(np.power(y, 2) + x - 7, 2)
        return z

class Eggholder(object):
    def __init__(self):
        self.name = "Eggholder function"
        self.minima = -959.6407
        self.minima_loc = np.array([512, 404.2319], dtype=np.float32)
        self.search_space = np.array([(-512, 512)]*2, dtype=np.float32)
        self.any_dim = False

    def evaluate(self, x, y):
        a = np.sin(np.sqrt(np.abs(x / 2 + y + 47)))
        b = np.sin(np.sqrt(np.abs(x - y + 47)))
        z = - (y + 47) * a - x * b
        return z


class Levy(object):
    def __init__(self):
        self.name = "Levy function"
        self.minima = 0.0
        self.minima_loc = np.array([1, 1], dtype=np.float32)
        self.search_space = np.array([(-10, 10)]*2, dtype=np.float32)
        self.any_dim = False

    def evaluate(self, x, y):
        a = np.power(np.sin(3 * np.pi * x), 2) 
        b = np.power(x - 1, 2)
        c = np.power(y - 1, 2)
        d = np.power(np.sin(2 * np.pi * x), 2)
        z = a + b * (1 - a) + c * (1 + d)
        return z
    
class GoldsteinPrice(object):
    def __init__(self):
        self.name = "Goldstein - Price function"
        self.minima = 3.0
        self.minima_loc = np.array([0, -1], dtype=np.float32)
        self.search_space = np.array([(-2, 2)]*2, dtype=np.float32)
        self.any_dim = False

    def evaluate(self, x, y):
        a = 1 + np.power(x + y + 1, 2) * (19 - 14*x + 3*np.power(x, 2) - 14*y + 6*x*y + 3*np.power(y, 2))
        b = 30 + np.power(2*x - 3*y, 2) * (18 - 32*x + 12*np.power(x, 2) + 48*y - 36*x*y + 27*np.power(y, 2))
        z = a * b
        return z
    
class Beale(object):
    def __init__(self):
        self.name = "Beale function"
        self.minima = 0.0
        self.minima_loc = np.array([3.0, 0.5], dtype=np.float32)
        self.search_space = np.array([(-4.5, 4.5)]*2, dtype=np.float32)
        self.any_dim = False
        self.n_dim = 2

    def evaluate(self, x, y):
        z = np.power(1.5 - x + x*y, 2) + np.power(2.25 - x + x*np.power(y, 2), 2) + np.power(2.625 - x + x*np.power(y, 3), 2)
        return z
    
class HolderTable(object):
    def __init__(self):
        self.name = "Holder Table function"
        self.search_space = np.array([(-10, 10)] * 2, dtype=np.float32)
        self.minima = -19.2085
        self.minima_loc = np.array([(8.05502, 9.66459), 
                                    (-8.05502, 9.66459), 
                                    (-8.05502, -9.66459), 
                                    (8.05502, -9.66459)], dtype=np.float32)
        self.any_dim = False
        
    def evaluate(self, x, y):
        a = np.abs(1 - np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.pi)
        z = -np.abs(np.sin(x) * np.cos(y) * np.exp(a))
        return z

# https://www.sfu.ca/~ssurjano/levy.html
class LevyMD(object):
    def __init__(self, n_dim, A):
        self.search_space = np.array([(-5.12, 5.12)] * n_dim, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0] * n_dim, dtype=np.float32)
        self.A = A
        self.n_dim = n_dim
        self.any_dim = True
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.n_dim + np.sum(z, axis=1)
        return z
    
# rastrigin function
# https://jamesmccaffrey.wordpress.com/2018/07/27/graphing-the-rastrigin-function-using-python/
# https://gist.github.com/miku/fca6afe05d65302f14c2b6f5242458d6
# Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
#   (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
  
