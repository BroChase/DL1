# Chase Brown
# SID 106015389
# Deep Learning
# Program 1: Simple ANN model

import numpy as np
import pandas as pd
import random
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ANN(object):

    def __init__(self):
        # Network layers
        self.input = 0
        self.hidden = 3
        self.output = 1
        # Read the data from csv file
        dataset = pd.read_csv('dataset.csv')
        # Drop Surnames and Geography from set
        dataset = dataset.drop(['Surname', 'Geography'], axis=1)

        # Split the Dataset into depend/indep x/y values
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        # Onehot enocode the Gender attribute
        one_hot = pd.get_dummies(dataset['Gender'])
        # Drop Gender from dataset and join the onehot encoded
        x = x.drop(['Gender'], axis=1)
        x = x.join(one_hot)

        x_train, x_tests, y_train, y_test = train_test_split(x, y, test_size=.20)

        SC = StandardScaler()
        x_train = SC.fit_transform(x_train)
        x_train = pd.DataFrame(x_train)

        # Randomize the synapse values from the input layer leading to hidden layer 1
        self.W1 = np.random.randn(11, 10)
        # Randomize the synapse values from the hidden layer 1 to the output layer
        self.W2 = np.random.randn(10, 1)


        for i in np.arange(10000):
            ran = random.randint(0, 7199)
            self.input = x_train.iloc[[ran]]
            y_actual = y_train.iloc[[ran]]
            layer1 = self.sigmoid_activation(np.dot(self.input, self.W1))
            layer2 = self.sigmoid_activation(np.dot(layer1, self.W2))

            layer2_error = self.cost_function(layer2[0], y_actual)
            if(i % 100) == 0:
                print('Error: ' + str(np.mean(np.abs(layer2_error))))
                print(ran)

            layer2_delta = layer2_error*self.derivative_sigmoid(layer2)

            layer1_error = layer2_delta.dot(self.W2.T)

            layer1_delta = layer1_error*self.derivative_sigmoid(layer1)


            self.W2 = self.W2 + layer1.T.dot(layer2_delta)
            self.W1 = self.W1 + self.input.T.dot(layer1_delta)
        print('test')




    # sigmoid activation function
    def sigmoid_activation(self, z):
        return expit(z)

    # derivative of the sigmoid function for back propagation
    def derivative_sigmoid(self, z):
        return z*(1-z)

    def cost_function(self, y_hat, y_actual):
        return 0.5*sum((y_actual-y_hat)**2)


if __name__ == '__main__':

    ANN = ANN()

    print('test')