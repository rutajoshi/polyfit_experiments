import os
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import cv2
import copy
import glob
import sys
import numpy.linalg as LA
import IPython

def set_axis_ranges(x_data, y_data):
    """ Make sure to pass in y data for all functions being plotted. """
    x_data_list = list(x_data)
    y_data_list = list(y_data)
    axes = plt.gca()
    x_width = max(x_data_list) - min(x_data_list)
    y_width = max(y_data_list) - min(y_data_list)
    axes.set_xlim([min(x_data_list) - (x_width//3), max(x_data_list) + (x_width//3)])
    axes.set_ylim([min(y_data_list) - (y_width//3), max(y_data_list) + (y_width//3)])
    plt.gca().set_autoscale_on(False)

def plot_data(x_range, y_range, pencil, color='k', label='f1'):
    x_range = np.array(list(x_range))
    y_range = np.array(list(y_range))
    if (label == None):
        plt.plot(x_range, y_range, pencil, color=color)
    else:
        plt.plot(x_range, y_range, pencil, color=color, label=label)

def plot_function(x_data, function, pencil, color='k', label='f1'):
    x_range = np.linspace(min(x_data) - 10, max(x_data) + 10, 1000)
    if (label == None):
        plt.plot(x_range, function(x_range), pencil, color=color)
    else:
        plt.plot(x_range, function(x_range), pencil, color=color, label=label)

def show_all(legend_loc='lower right'):
    plt.legend(loc=legend_loc)
    plt.show()
