# polyfit_experiments
Experiments with variance on polynomial regression

polyfit_experiments.py runs vanilla polynomial regression. Modify the parameters in the main method to try fitting different functions.

variance_random_extra_points.py runs polynomial regression iteratively, by adding 1 to k randomly chosen extra points to the training set each time.

noise_variance_experiment.py runs polynomial regression on a ground truth function. It generates a base model function and then samples one new training point from that base model. It computes gaussian deviations of the base model evaluated at that new training point and fits a new model using the deviation on each iteration.

plot_functions.py is a utility file for matplotlib plotting, for readability in the other files.
