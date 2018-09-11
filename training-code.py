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
from sklearn.metrics import confusion_matrix


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
        self.W1 = np.random.randn(11, 11)
        self.W2 = np.random.randn(11, 9)
        self.W3 = np.random.randn(9, 5)
        self.W4 = np.random.randn(5, 3)
        self.W5 = np.random.randn(3, 1)
        # bias term '1'
        self.bias = np.array([1])
        # Bias wieghts for each hidden layer bias for first layer 'input' alraedy added into matrix
        self.bw1 = np.random.randn(1, 9)
        self.bw2 = np.random.randn(1, 5)
        self.bw3 = np.random.randn(1, 3)
        self.bw4 = np.random.randn(1, 1)
        for i in np.arange(100000):
            ran = random.randint(0, 7199)
            self.layer0 = x_train.iloc[[ran]]
            y_actual = y_train.iloc[[ran]]
            # Hidden layer 1 Activation with layer0'input' dot with Weights 1 bias already inclused in inputs matrix
            self.layer1 = activations.relu_activation(np.dot(self.layer0, self.W1))
            # Hidden layer 2 Activation with layer 1 dot with weights 2 and bias 1 added
            self.layer2 = activations.relu_activation(np.dot(self.layer1, self.W2)+np.dot(self.bias, self.bw1))
            # Hidden layer 3 Activation with layer 2 dot with weights 3 and bias 2 added in
            self.layer3 = activations.relu_activation(np.dot(self.layer2, self.W3)+np.dot(self.bias, self.bw2))
            # Hidden layer 4 Activation with layer 3 dot with weights 4 and bias 3 added in
            self.layer4 = activations.sigmoid_activation(np.dot(self.layer3, self.W4)+np.dot(self.bias, self.bw3))
            # Output layer
            self.output = activations.sigmoid_activation(np.dot(self.layer4, self.W5)+np.dot(self.bias, self.bw4))
            error = activations.cost_function(self.output, y_actual.iloc[0].values)

            if (i % 100) == 0:
                # print(y_actual.iloc[0].values, self.output[0])
                print('Error: ' + str(np.mean(np.abs(error))))
            # calculating the error between the actual and predicted
            output_error = y_actual - self.output
            # deltas found based on the derivatives of the activations for each hidden layer
            delta_w5 = output_error * activations.derivative_sigmoid(self.output)
            delta_w4 = np.dot(delta_w5, self.W5.T) * activations.derivative_sigmoid(self.layer4)
            delta_w3 = np.dot(delta_w4, self.W4.T) * activations.derivative_relu(self.layer3)
            delta_w2 = np.dot(delta_w3, self.W3.T) * activations.derivative_relu(self.layer2)
            delta_w1 = np.dot(delta_w2, self.W2.T) * activations.derivative_relu(self.layer1)

            delta_bw4 = np.dot(delta_w5, self.bw4.T) * activations.derivative_sigmoid(self.layer4)
            delta_bw3 = np.dot(delta_bw4, self.bw3.T) * activations.derivative_relu(self.layer3)
            delta_bw2 = np.dot(delta_bw3, self.bw2.T) * activations.derivative_relu(self.layer2)
            delta_bw1 = np.dot(delta_bw2, self.bw1.T) * activations.derivative_relu(self.layer1)
            # the updated wights added to the previous weights 'adjusted by lambda of .1'
            self.bw4 += np.dot(self.layer4, delta_bw4.T)*.1
            self.bw3 += np.dot(self.layer3, delta_bw3.T)*.1
            self.bw2 += np.dot(self.layer2, delta_bw2.T)*.1
            self.bw1 += np.dot(self.layer1, delta_bw1.T)*.1
            # backpro updating the weights using the known layers and their delta values
            # Use alpha to adjust the learning rate
            self.W5 += np.dot(self.layer4.T, delta_w5)*.1
            self.W4 += np.dot(self.layer3.T, delta_w4)*.1
            self.W3 += np.dot(self.layer2.T, delta_w3)*.1
            self.W2 += np.dot(self.layer1.T, delta_w2)*.1
            self.W1 += np.dot(self.layer0.T, delta_w1)*.1

        testing1 = []
        for k in np.arange(x_train.shape[0]):
            self.layer0 = x_train.iloc[[k]]
            self.layer1 = activations.relu_activation(np.dot(self.layer0, self.W1))
            # Hidden layer 2
            self.layer2 = activations.relu_activation(np.dot(self.layer1, self.W2) + np.dot(self.bias, self.bw1))
            # Hidden layer 3
            self.layer3 = activations.relu_activation(np.dot(self.layer2, self.W3) + np.dot(self.bias, self.bw2))
            # Hidden layer 4
            self.layer4 = activations.sigmoid_activation(np.dot(self.layer3, self.W4) + np.dot(self.bias, self.bw3))
            # Output layer
            self.output = activations.sigmoid_activation(np.dot(self.layer4, self.W5) + np.dot(self.bias, self.bw4))

            testing1.append(activations.thresh(self.output[0][0]))
        y_pred_1 = np.array(testing1)
        # cm = confusion_matrix(y_test, y_pred)
        tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_train, y_pred_1).ravel()

        accuracy = activations.accuracy(tp_1, fp_1, tn_1, fn_1)
        precision = activations.precision(tp_1, fp_1)
        recall = activations.recall(tp_1, fn_1)
        f1 = activations.f1(precision, recall)
        print(accuracy)
        print(precision)
        print(recall)
        print(f1)

        testing = []
        for k in np.arange(x_test.shape[0]):
            self.layer0 = x_test.iloc[[k]]
            self.layer1 = activations.relu_activation(np.dot(self.layer0, self.W1))
            # Hidden layer 2
            self.layer2 = activations.relu_activation(np.dot(self.layer1, self.W2)+np.dot(self.bias, self.bw1))
            # Hidden layer 3
            self.layer3 = activations.relu_activation(np.dot(self.layer2, self.W3)+np.dot(self.bias, self.bw2))
            # Hidden layer 4
            self.layer4 = activations.sigmoid_activation(np.dot(self.layer3, self.W4)+np.dot(self.bias, self.bw3))
            # Output layer
            self.output = activations.sigmoid_activation(np.dot(self.layer4, self.W5)+np.dot(self.bias, self.bw4))

            testing.append(activations.thresh(self.output[0][0]))
        y_pred = np.array(testing)
        # cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuracy = activations.accuracy(tp, fp, tn, fn)
        precision = activations.precision(tp, fp)
        recall = activations.recall(tp, fn)
        f1 = activations.f1(precision, recall)
        print(accuracy)
        print(precision)
        print(recall)
        print(f1)

if __name__ == '__main__':

    ANN = ANN()
