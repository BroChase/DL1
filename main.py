# Chase Brown
# SID 106015389
# Deep Learning
# Program 1: Simple ANN model

import activations
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ANN(object):

    def __init__(self):
        # Read the data from csv file
        dataset = pd.read_csv('dataset.csv')
        # Dropping CustomerId, Surname, Geography from attributes
        dataset = dataset.drop(['CustomerId', 'Surname', 'Geography'], axis=1)

        # Split the Dataset into depend/indep x/y values
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        # Onehot enocode the Gender attribute
        one_hot = pd.get_dummies(dataset['Gender'])
        # Drop Gender from dataset and join the onehot encoded
        x = x.drop(['Gender'], axis=1)
        x = x.join(one_hot)
        # Check for NaN values in data.
        x = activations.missing_values(x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=12345)
        SC = StandardScaler()
        # Scale data 'standardscaler defaults
        x_train = SC.fit_transform(x_train)
        # Add Bias term
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        x_train = pd.DataFrame(x_train)
        # Fix indexing on dataframe
        y_train = pd.DataFrame(y_train)
        y_train = y_train.reset_index()
        y_train = y_train.drop(['index'], axis=1)

        x_test = SC.transform(x_test)
        x_test = np.c_[np.ones(x_test.shape[0]), x_test]
        x_test = pd.DataFrame(x_test)

        # Randomize the synapse values from the input layer leading to hidden layer 1
        self.W1 = np.random.randn(11, 5)
        self.W2 = np.random.randn(5, 3)
        self.W3 = np.random.randn(3, 1)
        self.b1 = np.full((1, 1), 1)
        self.b2 = np.full((1, 1), 1)
        self.bw1 = np.random.randn(1, 3)
        self.bw2 = np.random.randn(1, 1)
        alpha = 0.01
        for i in np.arange(10000):
            ran = random.randint(0, 7199)
            self.layer0 = x_train.iloc[[ran]]
            y_actual = y_train.iloc[[ran]]
            # Hidden layer 1 Relu 5 nodes
            self.layer1 = activations.relu_activation(np.dot(self.layer0, self.W1))
            # Hidden layer 2 sigmoid activation 3 nodes
            self.layer2 = activations.sigmoid_activation(np.dot(self.layer1, self.W2))# + activations.sigmoid_activation(np.dot(self.b1, self.bw1))
            # output layer sigmoid activation 1 node
            self.output = activations.sigmoid_activation(np.dot(self.layer2, self.W3))# + activations.sigmoid_activation(np.dot(self.b2, self.bw2))
            error = activations.cost_function(self.output[0], y_actual.iloc[0].values)

            if (i % 100) == 0:
                print(y_actual.iloc[0].values, self.output[0])
                print('Error: ' + str(np.mean(np.abs(error))))

            output_error = 2 * (y_actual - self.output)
            delta_w3 = output_error * activations.derivative_sigmoid(self.output)
            delta_w2 = np.dot(delta_w3, self.W3.T) * activations.derivative_sigmoid(self.layer2)
            delta_w1 = np.dot(delta_w2, self.W2.T) * activations.derivative_relu(self.layer1)
            #
            delta_b1 = np.dot(delta_w3, self.bw2.T) * activations.derivative_sigmoid(self.b2)
            self.bw2 += np.dot(self.b2.T, delta_w3*.01)
            self.bw1 += np.dot(self.b1.T, delta_b1*.01)
            # backpro updating the weights using the known layers and their delta values
            # Use alpha to adjust the learning rate
            self.W3 += np.dot(self.layer2.T, delta_w3)*.1
            self.W2 += np.dot(self.layer1.T, delta_w2)*.1
            self.W1 += np.dot(self.layer0.T, delta_w1)*.1

        testing = []
        for k in np.arange(x_test.shape[0]):
            self.layer0 = x_test.iloc[[k]]
            self.layer1 = activations.relu_activation(np.dot(self.layer0, self.W1))
            self.layer2 = activations.sigmoid_activation(np.dot(self.layer1, self.W2))# + activations.sigmoid_activation(np.dot(self.b1, self.bw1))
            self.output = activations.sigmoid_activation(np.dot(self.layer2, self.W3))# + activations.sigmoid_activation(np.dot(self.b2, self.bw2))
            print(self.output)
            testing.append(self.output[0])
        print('testing')



if __name__ == '__main__':

    ANN = ANN()