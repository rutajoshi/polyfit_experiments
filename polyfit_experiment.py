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

    Note: Used even though np.polyfit does the same thing.
    Note: Included for reference and for modifications as needed to basic pfit
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

def plot_noise_variance_experiment(sample_coefficients, model_degree, x_train, new_training_point_x, noise_std=0.5, noise_points=6):
    """
    Plots new model and old model, training points used to generate new model.
    New training point (just 1) is chosen by adding noise to old model output for that point.
    """
    sample_degree = len(sample_coefficients) - 1
    n = len(x_train)

    # Train the base model using the given training data
    y_train = evaluate_function_vec(x_train, sample_coefficients)
    base_model_coefficients = polynomial_regression(x_train, y_train, model_degree, noise_mean=0, noise_std=0.5)
    y_base_model = evaluate_function_vec(x_train, base_model_coefficients)
    base_variance = sum([(y_base_model[i] - y_train[i])**2 for i in range(len(y_train))])
    print("\nBase variance from true function with n=" + str(n) + " training samples is: " + str(base_variance) + "\n")

    new_training_point_y = evaluate_function(new_training_point_x, base_model_coefficients)
    dev_inc = (3*noise_std) / (noise_points//2)
    deviations = [new_training_point_y - z*dev_inc for z in range(1, noise_points//2 + 1)] + [new_training_point_y + z*dev_inc for z in range(1, noise_points//2 + 1)]

    axes = plt.gca()
    x_range = np.linspace(min(x_train) - 10, max(x_train) + 10, 1000)
    true_function = make_function(sample_coefficients)
    learned_function = make_function(base_model_coefficients)

    x_width = max(list(x_train) + [new_training_point_x]) - min(list(x_train) + [new_training_point_x])
    y_width = max(list(y_train) + [new_training_point_y] + list(y_base_model)) - min(list(y_train) + [new_training_point_y] + list(y_base_model))
    axes.set_xlim([min(x_train) - (x_width//3), max(x_train) + (x_width//3)])
    axes.set_ylim([min(list(true_function(x_range)) + list(learned_function(x_range))) - (y_width//3), max(list(true_function(x_range)) + list(learned_function(x_range))) + (y_width//3)])
    plt.gca().set_autoscale_on(False)

    plt.plot(x_range, true_function(x_range), '-', color='r', label='ground truth function')
    plt.plot(x_range, learned_function(x_range), '-', color='b', label='base model fit function')
    plt.plot(x_train, y_train, '.', color='k', label='training data')

    for deviation in deviations:
        # Train the model including the deviated point and plot it.
        x_new_train = np.array(list(x_train) + [new_training_point_x])
        y_new_train = np.array(list(y_train) + [deviation])
        new_model_coefficients = polynomial_regression(x_new_train, y_new_train, model_degree)
        y_new_model = evaluate_function_vec(x_train, new_model_coefficients)

        # Plot this new model
        deviated_function = make_function(new_model_coefficients)
        plt.plot(np.array([new_training_point_x]), np.array([new_training_point_y]), '.', color='g', label='new point')
        plt.plot(x_range, deviated_function(x_range), '-', color='m')

        # Find the variance of that model from base model function using only x_train points
        variance = sum([(y_new_model[i] - y_base_model[i])**2 for i in range(len(y_base_model))]) / len(y_base_model)
        print("Variance from base model using (" + str(new_training_point_x) + "," + str(deviation) + ") = " + str(variance))

    print("\nFinished plotting for new training input " + str(new_training_point_x) + '\n')
    plt.legend(loc='upper right');
    plt.show()


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

def run_experiment(sample_coefficients, model_degree, n, noise_mean=0, noise_std=0.5):
    sample_degree = len(sample_coefficients) - 1

    x_train, y_train = sample_function(sample_degree, sample_coefficients, n, lower=-5, upper=5)
    polyfit_coefficients = polynomial_regression(x_train, y_train, model_degree)

    x_range = np.linspace(min(x_train) - 10, max(x_train) + 10, 1000)
    true_function = make_function(sample_coefficients)
    learned_function = make_function(polyfit_coefficients)

    y_model = evaluate_function_vec(x_train, polyfit_coefficients)
    variance = sum([(y_model[i] - y_train[i])**2 for i in range(len(x_train))]) / len(x_train)
    print("\nVariance of the polyfit = " + str(variance) + "\n")

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
    w = [3, 1, 1, -2]
    # run_experiment(w, 3, 6, noise_mean=0, noise_std=0.5)
    model_degree = 4
    lower = -50
    upper = 50
    input_range = upper - lower
    n = 3
    k = 4
    np.random.seed(1)
    x_train = np.array([lower + i*(input_range // (n+1)) for i in range(1, n+1)])
    new_training_point_x = (min(x_train) + max(x_train)) / 2
    plot_noise_variance_experiment(w, 3, x_train, new_training_point_x, noise_std=10000)


if __name__ == "__main__": main()
