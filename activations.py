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

