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

def variance_experiment(sample_coefficients, model_degree, x_train, noise_mean=0, noise_std=0.5, iterations=10):
    """
    No plots, just variance testing.
    Adds random points (not well chosen).
    """
    sample_degree = len(sample_coefficients) - 1
    n = len(x_train)

    # Train the base model using the given training data
    y_train = evaluate_function_vec(x_train, sample_coefficients)
    base_model_coefficients = polynomial_regression(x_train, y_train, model_degree, noise_mean, noise_std)
    y_base_model = evaluate_function_vec(x_train, base_model_coefficients)
    base_variance = sum([(y_base_model[i] - y_train[i])**2 for i in range(len(y_train))])
    print("\nBase variance with n=" + str(n) + " training samples is: " + str(base_variance) + "\n")

    for i in range(1, iterations+1):
        # Sample i new points with a random label from the model
        x_new_points, y_new_points = sample_function(model_degree, base_model_coefficients, i)
        # Add gaussian noise to the labels of the new points
        y_new_points = [y+np.random.normal(noise_mean, noise_std) for y in y_new_points]

        # Train a new model using the new point and the old training data
        x_new_train = np.array(list(x_train) + list(x_new_points))
        y_new_train = np.array(list(y_train) + list(y_new_points))
        new_model_coefficients = polynomial_regression(x_new_train, y_new_train, model_degree, noise_mean, noise_std)
        y_new_model = evaluate_function_vec(x_train, new_model_coefficients)

        # Find the variance of that model from base model function
        variance = sum([(y_new_model[i] - y_base_model[i])**2 for i in range(len(y_base_model))]) / len(y_base_model)
        print("Variance with " + str(n + i) + " points is: " + str(variance))

    print("\nFinished " + str(iterations) + " iterations.\n")

def main():
    w = [3, 1, 1, -2]
    model_degree = 4
    lower = -50
    upper = 50
    input_range = upper - lower
    n = 3
    k = 4
    np.random.seed(1)
    x_train = np.array([lower + i*(input_range // (n+1)) for i in range(1, n+1)])
    variance_experiment(w, model_degree, x_train)

if __name__ == "__main__": main()
