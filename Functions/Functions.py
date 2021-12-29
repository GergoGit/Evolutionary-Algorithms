# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:26:24 2021

@author: bonnyaigergo

https://en.wikipedia.org/wiki/Test_functions_for_optimization
https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12

https://github.com/anyoptimization/pymoo/tree/master/pymoo/problems/single

"""

# ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
 
# objective function
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
 
# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()

import numpy as np

class Ackley(object):
    def __init__(self):
        self.minima = 0.0
        self.minima_loc = np.array([0.0, 0.0], dtype=np.float32)
        self.search_space = np.array([(-5, 5)]*2, dtype=np.float32)
        
    def evaluate(self, x, y):
        z = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
        return z

    
class Rastrigin(object):
    def __init__(self, dimensions, A):
        self.search_space = np.array([(-5.12, 5.12)] * dimensions, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0, 0] * dimensions, dtype=np.float32)
        self.A = A
        self.dimensions = dimensions
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.dimensions + np.sum(z, axis=1)
        return z


class Eggholder(object):
    def __init__(self):
        self.minima = -959.6407
        self.minima_loc = np.array([512, 404.2319], dtype=np.float32)
        self.search_space = np.array([(-512, 512)], dtype=np.float32)

    def evaluate(self, x, y):
        a = np.sin(np.sqrt(np.abs(x / 2 + y + 47)))
        b = np.sin(np.sqrt(np.abs(x - y + 47)))
        z = - (y + 47) * a - x * b
        return z


class Levy(object):
    def __init__(self):
        self.minima = 0.0
        self.minima_loc = np.array([1, 1], dtype=np.float32)
        self.search_space = np.array([(-10, 10)], dtype=np.float32)

    def evaluate(self, x, y):
        a = np.power(np.sin(3 * np.pi * x), 2) 
        b = np.power(x - 1, 2)
        c = np.power(y - 1, 2)
        d = np.power(np.sin(2 * np.pi * x), 2)
        z = a + b * (1 - a) + c * (1 + d)
        return z

# https://www.sfu.ca/~ssurjano/levy.html
class LevyMD(object):
    def __init__(self, dimensions, A):
        self.search_space = np.array([(-5.12, 5.12)] * dimensions, dtype=np.float32)
        self.minima = 0.0
        self.minima_loc = np.array([0, 0] * dimensions, dtype=np.float32)
        self.A = A
        self.dimensions = dimensions
        
    def evaluate(self, x):
        z = np.power(x, 2) - self.A * np.cos(2 * np.pi * x)
        z = self.A * self.dimensions + np.sum(z, axis=1)
        return z
    
# rastrigin function
# https://jamesmccaffrey.wordpress.com/2018/07/27/graphing-the-rastrigin-function-using-python/
# https://gist.github.com/miku/fca6afe05d65302f14c2b6f5242458d6
Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
  (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
  
