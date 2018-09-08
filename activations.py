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
    return x * (x > 0)

def derivative_relu(x):
    return 1 * (x > 0)

def cost_function(y_hat, y_actual):
    return float((y_hat - y_actual))


def missing_values(x):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit_transform(x)
    return imputer


# for i in np.arange(10000):
#     ran = random.randint(0, 7199)
#     self.input = x_train.iloc[[0]]
#     y_actual = y_train.iloc[[0]]
#     layer1 = np.array([1])
#     layer1 = np.append(layer1, activations.relu_activation(np.dot(self.input.values, self.W1)))
#     layer2 = np.array([1])
#     layer2 = np.append(layer2, activations.sigmoid_activation(np.dot(layer1, self.W2)))
#     layer3 = activations.sigmoid_activation(np.dot(layer2, self.W3))
#
#     layer3_error = activations.cost_function(layer3[0], y_actual.iloc[0])
#     # if (i % 100) == 0:
#     #     print('Error: ' + str(np.mean(np.abs(layer3_error))))
#     #     print(i)
#
#     # dsum/dresult * (target - calculated) = delta sum
#     layer3_delta = layer3_error * activations.derivative_sigmoid(layer3)
#
#     layer2_error = layer3_delta.dot(self.W3.T)
#
#     layer2_delta = layer2_error * activations.derivative_sigmoid(layer2)
#     layer2_delta = np.delete(layer2_delta, 0)
#     layer1_error = layer2_delta.dot(self.W2.T)
#
#     layer1_delta = layer1_error * activations.derivative_relu(layer1)
#
#     self.W3 = self.W3 + layer2.T.dot(layer3_delta) * .7
#     self.W2 = self.W2 + layer1.T.dot(layer2_delta) * .7
#     self.W1 = self.W1 + self.input.T.dot(layer1_delta) * .7
