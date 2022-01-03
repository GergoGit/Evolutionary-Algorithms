# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:36:54 2021

@author: bonnyaigergo

source links:
http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
https://notebook.community/ltiao/notebooks/visualizing-and-animating-optimization-algorithms-with-matplotlib
https://pyswarms.readthedocs.io/en/development/examples/visualization.html
https://pythonrepo.com/repo/logancyang-loss-landscape-anim
https://www.pyretis.org/current/examples/examples-pso.html
https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Surface.html

https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective/blob/main/pybenchfunction/util.py

Color logscale doesn't work in case of negative values in matplotlib'
"""

from numpy import meshgrid, arange, asarray
import numpy as np
# from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from inspect import isfunction, ismethod
# import types
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = "browser" # it shows plotly functions in web broser
pio.renderers.default = "svg" # it shows plotly functions in spyder

# import autograd.numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, ticker
from IPython.display import HTML

def log_scaler(vmin, vmax, num):
    r = vmax - vmin + 1
    m = np.log2(r)
    return np.logspace(start=0, stop=m, num=num, base=2.0) + vmin - 1


def Surface_3D(objective, search_space, title=None, logscale=False, rotation_xy=None, rotation_z=None):

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
        
    if ismethod(objective) == False:
        raise Exception("objective should be a function")
        
    # if isinstance(objective, types.MethodType) == False:
    #     raise Exception("objective should be a function")
        
    if (isinstance(title, str) or title is None) == False:
        raise Exception("title should be a string")
        
    if isinstance(search_space, np.ndarray) == False:
        raise Exception("search space should be given in numpy array format")
        
    
    min_bound = asarray([min(dim) for dim in search_space])
    max_bound = asarray([max(dim) for dim in search_space])
    
    x_axis = np.linspace(start=min_bound[0], stop=max_bound[0], num=1000)
    y_axis = np.linspace(start=min_bound[1], stop=max_bound[1], num=1000)

    x, y = meshgrid(x_axis, y_axis)
    z = objective(x, y)
    
    color_norm = LogNorm() if logscale else None
    
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, z, cmap='jet', norm=color_norm)
    axis.set_xlabel('X')
    axis.set_ylabel('$Y$')
    axis.set_zlabel('$Z$')
    axis.set_title(title)
    axis.view_init(azim=rotation_xy, elev=rotation_z)
    plt.show()



def Surface_3D_plotly(objective, search_space, title=None):

    """
    There is no log scale setting for z colour axis at the moment for plotly
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
        
    if ismethod(objective) == False:
        raise Exception("objective should be a function")
        
    # if isinstance(objective, types.MethodType) == False:
    #     raise Exception("objective should be a function")
        
    if (isinstance(title, str) or title is None) == False:
        raise Exception("title should be a string")
        
    if isinstance(search_space, np.ndarray) == False:
        raise Exception("search space should be given in numpy array format")
            
    min_bound = asarray([min(dim) for dim in search_space])
    max_bound = asarray([max(dim) for dim in search_space])
    
    x_axis = np.linspace(start=min_bound[0], stop=max_bound[0], num=1000)
    y_axis = np.linspace(start=min_bound[1], stop=max_bound[1], num=1000)

    x, y = meshgrid(x_axis, y_axis)
    z = objective(x, y)
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Jet')])
    fig.update_layout(title_text=title, title_x=0.5)
    fig.show()
    
    

def contour_plot(objective, search_space, minima=None, minima_loc=None, title=None, logscale=False):
    
    if ismethod(objective) == False:
        raise Exception("objective should be a function")
        
    # if isinstance(objective, types.MethodType) == False:
    #     raise Exception("objective should be a function")
        
    if (isinstance(title, str) or title is None) == False:
        raise Exception("title should be a string")
        
    if isinstance(search_space, np.ndarray) == False:
        raise Exception("search space should be given in numpy array format")
        if search_space.ndim != 2:
            raise Exception("search space should be given in 2D numpy array format")
            
    if (isinstance(minima_loc, np.ndarray) or minima_loc is None) == False:
        raise Exception("minima should be given in numpy array format")
    if minima_loc.ndim == 1:
        minima_loc = minima_loc.reshape(1, 2)
        
            
    min_bound = asarray([min(dim) for dim in search_space])
    max_bound = asarray([max(dim) for dim in search_space])

    x_axis = np.linspace(start=min_bound[0], stop=max_bound[0], num=1000)
    y_axis = np.linspace(start=min_bound[1], stop=max_bound[1], num=1000)

    x, y = meshgrid(x_axis, y_axis)
    z = objective(x, y)
    
    color_norm = LogNorm() if logscale else None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot_surface(x,y,z, cmap='jet', norm=color_norm)
    if logscale:
        ax.contour(
            x,y,z, 
            levels=log_scaler(vmin=minima, vmax=z.max(), num=50), 
            cmap=plt.cm.jet
            )
    else:
        cs = ax.contourf(x,y,z, levels=25, cmap=plt.cm.jet)
        fig.colorbar(cs, ax=ax, shrink=0.9)
    if minima_loc is not None:
        ax.plot(minima_loc[:,0], minima_loc[:,1], 'r*', markersize=18) # red stars as markers of minima        
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_title(title)    
    ax.set_xlim((min_bound[0], max_bound[0]))
    ax.set_ylim((min_bound[1], max_bound[1]))
    
class ObjectiveToPlot(object):
    def __init__(self, objective, search_space, minima, minima_loc, title):
        
        if ismethod(objective) == False:
            raise Exception("objective should be a function")
            
        if (isinstance(title, str) or title is None) == False:
            raise Exception("title should be a string")
            
        if isinstance(search_space, np.ndarray) == False:
            raise Exception("search space should be given in numpy array format")
            if search_space.ndim != 2:
                raise Exception("search space should be given in 2D numpy array format")
        
        if (isinstance(minima_loc, np.ndarray) or minima_loc is None) == False:
            raise Exception("minima_loc should be given in numpy array format")
        if minima_loc is not None and minima_loc.ndim == 1:
            minima_loc = minima_loc.reshape(1, 2)
            
        self.objective = objective
        self.search_space = search_space
        self.title= title
        self.minima = minima
        self.minima_loc = minima_loc
        
        self.min_bound = asarray([min(dim) for dim in self.search_space])
        self.max_bound = asarray([max(dim) for dim in self.search_space])
    
        self.x_axis = np.linspace(start=self.min_bound[0], stop=self.max_bound[0], num=1000)
        self.y_axis = np.linspace(start=self.min_bound[1], stop=self.max_bound[1], num=1000)
    
        self.x, self.y = meshgrid(self.x_axis, self.y_axis)
        self.z = objective(self.x, self.y)
    
    def PlotlyContour3D(self):
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=self.x_axis,
                y=self.y_axis,
                z=self.z,
                colorscale="Jet"
                )
            )
        if self.minima_loc is not None:
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=self.minima_loc[:,0],
                    y=self.minima_loc[:,1],
                    marker=dict(
                        symbol = 'star',
                        color='Red',
                        size=15
                    ),
                    showlegend=False
                )
            )
        fig.update_layout(title_text=self.title, title_x=0.5, xaxis_title='X', yaxis_title='Y')
        fig.show()
        
    def PlotlySurface3D(self):
        fig = go.Figure(data=[go.Surface(x=self.x, y=self.y, z=self.z, colorscale='Jet')])
        fig.update_layout(title_text=self.title, title_x=0.5)
        fig.show()
        
    def ContourPlot(self, logscale=False):            
        color_norm = LogNorm() if logscale else None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if logscale:
            ax.contour(
                self.x, self.y, self.z, 
                levels=log_scaler(vmin=self.minima, vmax=self.z.max(), num=50), 
                cmap=plt.cm.jet
                )
        else:
            cs = ax.contourf(self.x, self.y, self.z, levels=25, cmap=plt.cm.jet, norm=color_norm)
            fig.colorbar(cs, ax=ax, shrink=0.9)
        if self.minima_loc is not None:
            ax.plot(self.minima_loc[:,0], self.minima_loc[:,1], 'r*', markersize=18) # red stars as markers of minima        
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_title(self.title)    
        ax.set_xlim((self.min_bound[0], self.max_bound[0]))
        ax.set_ylim((self.min_bound[1], self.max_bound[1]))
        
    def Surface3D(self, logscale=False, rotation_xy=None, rotation_z=None):
        color_norm = LogNorm() if logscale else None
    
        figure = plt.figure()
        ax = figure.gca(projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='jet', norm=color_norm)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(self.title)
        ax.view_init(azim=rotation_xy, elev=rotation_z)
        plt.show()

class CreateBaseToPlot(object):
    
    def __init__(self, obj_func):
            
        self.objective =    obj_func.evaluate
        self.search_space = obj_func.search_space
        self.title =        obj_func.name
        self.minima =       obj_func.minima
        self.minima_loc =   obj_func.minima_loc
        
        if ismethod(self.objective) == False:
            raise Exception("objective should be a function")
            
        if (isinstance(self.title, str) or self.title is None) == False:
            raise Exception("title should be a string")
            
        if isinstance(self.search_space, np.ndarray) == False:
            raise Exception("search space should be given in numpy array format")
            if self.search_space.ndim != 2:
                raise Exception("search space should be given in 2D numpy array format")
        
        if (isinstance(self.minima_loc, np.ndarray) or self.minima_loc is None) == False:
            raise Exception("minima_loc should be given in numpy array format")
        if self.minima_loc is not None and self.minima_loc.ndim == 1:
            self.minima_loc = self.minima_loc.reshape(1, 2)
        
        if obj_func.any_dim and obj_func.dimensions != 2:
            raise Exception("for functions working with any dimensions set dimensions = 2")
            
        self.min_bound = np.asarray([min(dim) for dim in self.search_space])
        self.max_bound = np.asarray([max(dim) for dim in self.search_space])
    
        self.x_axis = np.linspace(start=self.min_bound[0], stop=self.max_bound[0], num=1000)
        self.y_axis = np.linspace(start=self.min_bound[1], stop=self.max_bound[1], num=1000)
    
        self.x, self.y = np.meshgrid(self.x_axis, self.y_axis)
        
        if obj_func.any_dim:
            xy = np.array([self.x, self.y])
            self.z = np.apply_along_axis(func1d=self.objective, axis=0, arr=xy)
        else:    
            self.z = self.objective(self.x, self.y)
        
    def PlotlyContour3D(self):
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=self.x_axis,
                y=self.y_axis,
                z=self.z,
                colorscale="Jet"
                )
            )
        if self.minima_loc is not None:
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=self.minima_loc[:,0],
                    y=self.minima_loc[:,1],
                    marker=dict(
                        symbol = 'star',
                        color='Red',
                        size=15
                    ),
                    showlegend=False
                )
            )
        fig.update_layout(title_text=self.title, title_x=0.5, xaxis_title='X', yaxis_title='Y')
        fig.show()
        
    def PlotlySurface3D(self):
        fig = go.Figure(data=[go.Surface(x=self.x, y=self.y, z=self.z, colorscale='Jet')])
        fig.update_layout(title_text=self.title, title_x=0.5)
        fig.show()
        
    def ContourPlot(self, logscale=False):            
        color_norm = LogNorm() if logscale else None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if logscale:
            ax.contour(
                self.x, self.y, self.z, 
                levels=log_scaler(vmin=self.minima, vmax=self.z.max(), num=100), 
                cmap=plt.cm.jet
                )
        else:
            cs = ax.contourf(self.x, self.y, self.z, levels=25, cmap=plt.cm.jet, norm=color_norm)
            fig.colorbar(cs, ax=ax, shrink=0.9)
        if self.minima_loc is not None:
            ax.plot(self.minima_loc[:,0], self.minima_loc[:,1], 'r*', markersize=18) # red stars as markers of minima        
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_title(self.title)    
        ax.set_xlim((self.min_bound[0], self.max_bound[0]))
        ax.set_ylim((self.min_bound[1], self.max_bound[1]))
        
    def Surface3D(self, logscale=False, rotation_xy=None, rotation_z=None):
        color_norm = LogNorm() if logscale else None
    
        figure = plt.figure()
        ax = figure.gca(projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='jet', norm=color_norm)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(self.title)
        ax.view_init(azim=rotation_xy, elev=rotation_z)
        plt.show()
    
    def ContourSurface3D(self, logscale=False, rotation_xy=None, rotation_z=None):
        color_norm = LogNorm() if logscale else None
    
        figure = plt.figure()
        ax = figure.gca(projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='jet', norm=color_norm)
        if logscale:
            ax.contour(
                self.x, self.y, self.z, 
                levels=log_scaler(vmin=self.minima, vmax=self.z.max(), num=50), 
                cmap=plt.cm.jet, offset=self.minima
                )
        else:
            ax.contour(self.x, self.y, self.z, zdir='z', levels=25, offset=self.minima, cmap='jet')
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title(self.title)
        ax.view_init(azim=rotation_xy, elev=rotation_z)
        plt.show()        
        
    
        
if __name__ == '__main__':
    
    from ObjectiveFunctions import Ackley, Eggholder, Levy, Himmelblau, GoldsteinPrice 
    from ObjectiveFunctions import Beale, HolderTable, Rastrigin, Michalewicz, Salomon
    
    fn = Ackley()
    fn = Eggholder()
    fn = Levy()
    fn = Himmelblau()
    fn = GoldsteinPrice()
    fn = Beale()
    fn = HolderTable()
    fn = Rastrigin(dimensions=2)
    fn = Michalewicz(dimensions=2)
    fn = Salomon(dimensions=2)
    
    Surface_3D(objective=fn.evaluate, 
               search_space=fn.search_space, 
               title=fn.name,
               logscale=True)
    contour_plot(objective=fn.evaluate, 
                 search_space=fn.search_space, 
                 minima=fn.minima_loc, 
                 title=fn.name,
                 logscale=True)
    Surface_3D_plotly(objective=fn.evaluate, 
                      search_space=fn.search_space, 
                      title=fn.name)
    
    PlotBase = ObjectiveToPlot(objective=fn.evaluate, 
                               search_space=fn.search_space,
                               minima=fn.minima,
                               minima_loc=fn.minima_loc,
                               title=fn.name)
    PlotBase.Surface3D(logscale=True, rotation_xy=None, rotation_z=None)
    PlotBase.ContourPlot(logscale=True)
    PlotBase.PlotlySurface3D()
    PlotBase.PlotlyContour3D()
    
    
    PlotBase = CreateBaseToPlot(fn)
    PlotBase.Surface3D(logscale=False, rotation_xy=None, rotation_z=None)
    PlotBase.ContourPlot(logscale=False)
    PlotBase.PlotlySurface3D()
    PlotBase.PlotlyContour3D()
    PlotBase.ContourSurface3D(logscale=True)
    
