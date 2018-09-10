# Chase Brown
# SID 106015389
# Deep Learning
# Program 1: Simple ANN model

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import Imputer

# sigmoid activation function
def sigmoid_activation(z):
    return expit(z)


# derivative of the sigmoid function for back propagation
def derivative_sigmoid(z):
    return z * (1 - z)

def thresh(x):
    if x > .5:
        return 1
    else:
        return 0

def relu_activation(x):
    return np.maximum(x, 0, x)
    #return x * (x > 0)

def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def cost_function(y_hat, y_actual):
    return float((y_hat - y_actual))


def missing_values(x):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit_transform(x)
    return imputer


def precision(true_p, false_p):
    if (true_p + false_p) == 0:
        return 0
    else:
        return true_p / (true_p + false_p)

def accuracy(true_p, false_p, true_n, false_n):
    if (true_p + false_p + true_n + false_n) == 0:
        return 0
    else:
        return (true_p + true_n) / (true_p + false_p + true_n + false_n)

def recall(true_p, false_n):
    if (true_p + false_n) == 0:
        return 0
    else:
        return true_p / (true_p + false_n)

def f1(precision, recall):
    if (precision + recall) == 0:
        return 0
    else:
        return 2 * ((precision * recall) / (precision + recall))