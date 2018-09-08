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
        # Randomize the synapse values from the input layer leading to hidden layer 1
        self.W1 = np.random.randn(11, 5)
        self.W2 = np.random.randn(5, 1)
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

        for i in np.arange(100000):
            ran = random.randint(0, 7199)
            self.layer0 = x_train.iloc[[ran]]
            y_actual = y_train.iloc[[ran]]

            self.layer1 = activations.sigmoid_activation(np.dot(self.layer0, self.W1))
            self.output = activations.sigmoid_activation(np.dot(self.layer1, self.W2))

            error = activations.cost_function(self.output[0], y_actual.iloc[0])
            if (i % 100) == 0:
                print('Error: ' + str(np.mean(np.abs(error))))
                print(i)

            delta_w2 = np.dot(self.layer1.T, (2*(y_actual - self.output) * activations.derivative_sigmoid(self.output)))
            delta_w1 = np.dot(self.layer0.T, (np.dot(2*(y_actual - self.output) *
                                                     activations.derivative_sigmoid(self.output),
                                                     self.W2.T) * activations.derivative_sigmoid(self.layer1)))

            self.W1 += delta_w1
            self.W2 += delta_w2

        testing = []
        for k in np.arange(x_test.shape[0]):
            self.layer0 = x_test.iloc[[k]]
            self.layer1 = activations.sigmoid_activation(np.dot(self.layer0, self.W1))
            self.output = activations.sigmoid_activation(np.dot(self.layer1, self.W2))
            testing.append(self.output)
        df = pd.DataFrame({'tests': testing})
        print(testing)
        print('testing')



if __name__ == '__main__':

    ANN = ANN()