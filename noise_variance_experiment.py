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
from polyfit_experiment import *
from plot_functions import *

def plot_noise_variance_experiment(sample_coefficients, model_degree, x_train, new_training_point_x, noise_std=0.5, noise_points=6):
    """
    Plots new model and old model, training points used to generate new model.
    New training point (just 1) is chosen by adding noise to old model output for that point.
    """
    sample_degree = len(sample_coefficients) - 1
    n = len(x_train)

    # Train the base model using the given training data
    y_train = evaluate_function_vec(x_train, sample_coefficients)
    base_model_coefficients = polynomial_regression(x_train, y_train, model_degree)
    y_base_model = evaluate_function_vec(x_train, base_model_coefficients)
    base_variance = sum([(y_base_model[i] - y_train[i])**2 for i in range(len(y_train))])
    print("\nBase variance from true function with n=" + str(n) + " training samples is: " + str(base_variance) + "\n")

    # Evaluate new training point using the base model
    new_training_point_y = evaluate_function(new_training_point_x, base_model_coefficients)
    # Generate deviated evaluations of the new training point using gaussian noise
    dev_inc = (3*noise_std) / (noise_points//2)
    deviations = [new_training_point_y - z*dev_inc for z in range(1, noise_points//2 + 1)] + [new_training_point_y + z*dev_inc for z in range(1, noise_points//2 + 1)]

    # Plot the ground truth, base model, and training data
    true_function = make_function(sample_coefficients)
    base_model_function = make_function(base_model_coefficients)
    set_axis_ranges(list(x_train) + [new_training_point_x], list(y_train) + [new_training_point_y] + list(y_base_model))
    plot_function(x_train, true_function, '-', color='r', label='ground truth function')
    plot_function(x_train, base_model_function, '--', color='b', label='base model fit function')
    plot_data(x_train, y_train, '.', color='k', label='training data')
    plot_data([new_training_point_x], [new_training_point_y], '.', color='c', label='new training point')

    # Go through and plot each deviated function
    x_new_train = np.array(list(x_train) + [new_training_point_x])
    for deviation in deviations:
        # Train the model including the deviated point and plot it.
        y_new_train = np.array(list(y_train) + [deviation])
        new_model_coefficients = polynomial_regression(x_new_train, y_new_train, model_degree)
        y_new_model = evaluate_function_vec(x_train, new_model_coefficients)

        # Plot this new model
        deviated_function = make_function(new_model_coefficients)
        plot_data([new_training_point_x], [deviation], '.', color='g', label='deviated point')
        plot_function(x_train, deviated_function, '-', color='m', label=None)

        # Find the variance of that model from base model function using only x_train points
        variance = sum([(y_new_model[i] - y_base_model[i])**2 for i in range(len(y_base_model))]) / len(y_base_model)
        print("Variance from base model using (" + str(new_training_point_x) + "," + str(deviation) + ") = " + str(variance))

    print("\nFinished plotting for new training input " + str(new_training_point_x) + '\n')
    show_all(legend_loc='upper right')

def main():
    w = [3, 1, 1, -2]
    model_degree = 4
    lower = -50
    upper = 50
    input_range = upper - lower
    n = 4
    k = 4
    np.random.seed(1)
    x_train = np.array([lower + i*(input_range // (n+1)) for i in range(1, n+1)])
    new_training_point_x = (min(x_train) + max(x_train)) / 2
    plot_noise_variance_experiment(w, model_degree, x_train, new_training_point_x, noise_std=50)

if __name__ == "__main__": main()
