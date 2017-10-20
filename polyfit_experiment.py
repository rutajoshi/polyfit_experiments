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

def make_function(w):
    def func(x):
        return evaluate_function(x, w)
    return func

def evaluate_function(x, w):
    y = 0
    for i in range(len(w)):
        y += (w[i] * (x**i))
    return y

def evaluate_function_vec(x_values, w):
    y_values = [evaluate_function(x, w) for x in x_values]
    return np.array(y_values)

def sample_function(d, w, n, lower=-100, upper=100):
    """
    d = order of the polynomial
    w = coefficients vector
    n = number of points to sample

    lower = lower bound for sample points (x)
    upper = upper bound for sample points (x)

    returns x_train, y_train
    """
    assert len(w) == d+1, "Must pass in d+1 coefficients"
    func_range = upper - lower
    x_values = np.array([lower + func_range*np.random.sample() for i in range(n)])
    y_values = evaluate_function_vec(x_values, w)
    return x_values, y_values

def polynomial_regression(x_train, y_train, d, noise_mean=0, noise_std=1):
    """
    x_train = training data input values
    y_train = training data output values (ground truth function)
    d = order of the polynomial we want to fit to this data

    Note: Unused because np.polyfit does the same thing.
    Note: Included for reference.
    """
    w = [] #the coefficients vector that should be returned
    X_mtx = []

    for x in x_train:
        evaluated_orders = [x**i for i in range(d+1)]
        X_mtx.append(np.array(evaluated_orders))

    X_mtx = np.array(X_mtx)
    y_train = np.array([i+np.random.normal(noise_mean, noise_std) for i in y_train])

    w = np.matmul(np.linalg.inv(np.matmul(X_mtx.T, X_mtx)), np.matmul(X_mtx.T, y_train))
    return w

def run_experiment(sample_coefficients, model_degree, n, noise_mean=0, noise_std=0.5):
    sample_degree = len(sample_coefficients) - 1

    x_train, y_train = sample_function(sample_degree, sample_coefficients, n, lower=-5, upper=5)
    polyfit_coefficients = polynomial_regression(x_train, y_train, model_degree)

    x_range = np.linspace(min(x_train) - 10, max(x_train) + 10, 1000)
    true_function = make_function(sample_coefficients)
    learned_function = make_function(polyfit_coefficients)

    axes = plt.gca()
    axes.set_xlim([min(x_train) - 10, max(x_train) + 10])
    axes.set_ylim([min(y_train) - 10, max(y_train) + 10])
    plt.gca().set_autoscale_on(False)

    plt.plot(x_range, true_function(x_range), '-', color='r', label='ground truth function')
    plt.plot(x_range, learned_function(x_range), '--', color='b', label='model fit function')
    plt.plot(x_train, y_train, '.', color='g', label='sample points')
    plt.legend(loc='lower right');
    plt.show()

def main():
    # true_function_degree = 2
    # w = [np.random.sample()*10 - 5 for i in range(true_function_degree + 1)]
    w = [3, 1, -2]
    run_experiment(w, 3, 6, noise_mean=0, noise_std=0.5)

if __name__ == "__main__": main()
