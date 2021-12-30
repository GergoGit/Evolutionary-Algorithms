# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:36:54 2021

@author: bonnyaigergo
"""

from numpy import meshgrid, arange, asarray
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from inspect import isfunction


def func_plot_3D(objective, search_space, title=None):

    """
    Parameters
    ----------
    objective : FUNCTION
        DESCRIPTION: 
            objective function with x and y inputs, gives z as output
    search_space : ARRAY
        DESCRIPTION:
            numpy array with min, max pairs as range related to x and y
    title : STRING, optional
        DESCRIPTION:
            title name of the plot
    Returns
    -------
    
    3D surface plot with jet color map
    
    """
    # if isfunction(objective) == False:
    #     raise Exception("objective should be a function")
        
    if isinstance(title, str) == False:
        raise Exception("title should be string")
        
    if isinstance(search_space, np.ndarray):
        raise Exception("search space should be given in numpy array format")
        
    
    min_bound = asarray([min(dim) for dim in search_space])
    max_bound = asarray([max(dim) for dim in search_space])

    # sample input range uniformly at 0.1 increments
    x_axis = arange(min_bound[0], max_bound[0], 0.1)
    y_axis = arange(min_bound[1], max_bound[1], 0.1)

    x, y = meshgrid(x_axis, y_axis)
    z = objective(x, y)
    
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, z, cmap='jet')
    axis.set_xlabel('X')
    axis.set_ylabel('$Y$')
    axis.set_zlabel('$Z$')
    axis.set_title(title)
    pyplot.show()

    
if __name__ == '__main__':
    fn = Ackley()
    fn = Eggholder()
    fn = Levy()
    func_plot_3D(objective=fn.evaluate, search_space=fn.search_space, title="Ackley")
    
    
    
