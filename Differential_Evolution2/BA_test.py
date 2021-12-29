# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:23:56 2021

@author: bonnyaigergo
"""

import numpy as np
from numba.experimental import jitclass

bat_alg_spec = []

@jitclass(bat_alg_spec)
class BA(object):
    def __init__(self, obj_func, population_size, iterations):
        self.population_size = population_size
        self.iterations = iterations
        