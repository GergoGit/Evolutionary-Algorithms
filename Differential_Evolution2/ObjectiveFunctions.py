# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:26:24 2021

@author: bonnyaigergo

https://en.wikipedia.org/wiki/Test_functions_for_optimization
https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12

https://github.com/anyoptimization/pymoo/tree/master/pymoo/problems/single

"""

import numpy as np

# ackley multimodal function
class Ackley(object):
    def __init__(self):
        self.name = "Ackley function"
        self.minima = 0.0
        self.minima_loc = np.array([0.0, 0.0], dtype=np.float32)
        self.search_space = np.array([(-5, 5)]*2, dtype=np.float32)
        
    def evaluate(self, x, y):
        z = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
        return z

    
class Rastrigin(object):
    def __init__(self, dimensions, A=10):
        self.name = "Rastrigin function"
        self.search_space = np.array([(-5.12, 5.12)] * dimensions, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0, 0] * dimensions, dtype=np.float32)
        self.A = A
        self.dimensions = dimensions
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.dimensions + np.sum(z, axis=1)
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
        
    def evaluate(self, x, y):
        z = np.power(np.power(x, 2) + y - 11, 2) + np.power(np.power(y, 2) + x - 7, 2)
        return z

class Eggholder(object):
    def __init__(self):
        self.name = "Eggholder function"
        self.minima = -959.6407
        self.minima_loc = np.array([512, 404.2319], dtype=np.float32)
        self.search_space = np.array([(-512, 512)]*2, dtype=np.float32)

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
        
    def evaluate(self, x, y):
        a = np.abs(1 - np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.pi)
        z = -np.abs(np.sin(x) * np.cos(y) * np.exp(a))
        return z

# https://www.sfu.ca/~ssurjano/levy.html
class LevyMD(object):
    def __init__(self, dimensions, A):
        self.search_space = np.array([(-5.12, 5.12)] * dimensions, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0] * dimensions, dtype=np.float32)
        self.A = A
        self.dimensions = dimensions
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.dimensions + np.sum(z, axis=1)
        return z
    
# rastrigin function
# https://jamesmccaffrey.wordpress.com/2018/07/27/graphing-the-rastrigin-function-using-python/
# https://gist.github.com/miku/fca6afe05d65302f14c2b6f5242458d6
# Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
#   (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
  
