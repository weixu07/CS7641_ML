# Script to run experiments

import neuralNetwork as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset(split_percentage=0.3):

    data = pd.read_csv('data/winequality-white.csv', sep=';')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.values
    y = y.astype(int)
    y[y < 6] = 0
    y[y >= 6] = 1
    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=100)
    # scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def neural_network(x_train, x_test, y_train, y_test, random_seeds):
    """Define and experiment the Neural Network weights optimization problem.

        Training Neural Networks weights can be done using GD and backpropation, but also another RO
        optimization algorithm, like RHC, SA or GA, can be used.

        Args:
          x_train (ndarray): training data.
          x_test (ndarray): test data.
          y_train (ndarray): training labels.
          y_test (ndarray): test labels.
          random_seeds (list or ndarray): random seeds for get performances over multiple random runs.

        Returns:
          None.
        """
    # Maximum iterations to run the Neural Network for
    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])


    # Plot performances for RHC, SA, GA and GD with Neural Networks
    nn.plot_nn_performances(x_train, y_train,
                            random_seeds=random_seeds,
                            rhc_max_iters=iterations, sa_max_iters=iterations,
                            ga_max_iters=iterations, gd_max_iters=iterations,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)

    # Test performances for RHC, SA, GA and GD with Neural Networks
    nn.test_nn_performances(x_train, x_test, y_train, y_test,
                            random_seed=random_seeds[0], max_iters=200,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)

if __name__ == "__main__":

    random_seeds = [5 + 5 * i for i in range(2)]  # random seeds for get performances over multiple random runs

    # Experiment Neural Networks optimization with RHC, SA, GA and GD on the WDBC dataset
    X_train, X_test, y_train, y_test = load_dataset(split_percentage=0.3)
    neural_network(X_train, X_test, y_train, y_test, random_seeds=random_seeds)