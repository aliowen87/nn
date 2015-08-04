__author__ = 'alaso0'
import numpy as np
import scipy.stats as stats

class Layer(object):

    def __init__(self, inner, outer, activation):
        self.inner = inner
        self.outer = outer
        self.activation = activation
        # initialise weights and bias
        self.weights = np.random.randn(self.outer, self.inner) / np.sqrt(self.inner + self.outer)
        self.bias = np.random.randn(self.outer, 1).T
        self.nablaw = np.zeros(self.weights.shape)
        self.nablab = np.zeros(self.bias.shape)


class Network(object):

    def __init__(self, layers, seed=42):
        # seed random number generator
        np.random.seed(seed)
        # initialise layers
        self.layers = [Layer(l, layers[i+1], self.sigmoid)
                       for i, l in enumerate(layers[:-1])]

    # def dropout(self, layer, p=1.0):
    #     """
    #     Takes an activated layer during forward prop and applies a mask to increase sparsity
    #     of the layer using probability p. Returns dropped-out layer
    #     :param layer: activated layer
    #     :param p: dropout probability i.e. sparsity of outputted layer
    #     :return: dropped layer
    #     """
    #     mask = np.random.rand(*layer.shape) < p
    #     return layer * mask

    def inverted_dropout(self, layer, p=1.0):
        """
        Takes an activated layer during forward prop and applies a mask to increase sparsity
        of the layer using probability p. Inverts using p. Returns dropped-out layer
        :param layer: activated layer
        :param p: dropout probability i.e. sparsity of outputted layer
        :return: dropped layer
        """
        mask = (np.random.rand(*layer.shape) < p) / p
        return layer * mask


    def normalise(self, train, val=None, test=None):
        """
        Normalise split train/validation/test data by subtracting train.mean() and scaling by
        train.std()
        :param train: training set
        :param val: validation set (optional)
        :param test: test set (optional)
        :return: Normalised train/validation/test set
        """
        mean = train.mean()
        std = train.std()
        train_scaled = (train - mean) / std
        if val:
            val = (val - mean) / std
        if test:
            test = (test - mean) / std
        return train_scaled, val, test


    def pca_whiten(self):
        pass

    def feedforward(self, X):
        activations = list()
        zs = list()
        activation  = X
        for l in self.layers:
            z = np.dot(activation, l.weights.T) + l.bias
            # save z for backprop
            zs.append(z)
            activation = l.activation(z)
            # save activation for backprop
            activations.append(activation)
        return activations, zs

    def backprop(self, X, y):
        # get activations
        A, Z = self.feedforward(X)

        # start backprop
        delta = (A[-1] - y)
        self.layers[-1].nablab = delta[0]
        self.layers[-1].nablaw = np.dot(delta.T, A[-2])
        for i in range(2, len(self.layers)):
            z = Z[-i]
            activation_prime = self.sigmoid_prime(z)
            delta = np.dot(delta, self.layers[-i+1].weights) * activation_prime

            self.layers[-i].nablab = delta.mean(axis=0)
            self.layers[-i].nablaw = np.dot(delta.T, A[-i-1])
        # weights and bias are updated in place, so nothing to return
        return None

    def stochastic_gradient_descent(self):
        pass

    def update_mini_batch(self):
        pass

    def cost_function(self):
        pass

    def sigmoid(self, z):
        """
        Computes sigmoid activation
        :param z: Numpy array of the form X.W.T
        :return: Numpy array activated by sigmoid
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        Computes sigmoid gradient of numpy array
        :param z: Numpy array of the form activation - y/ grad.T.activation[-1] etc
        :return: Sigmoid gradient
        """
        y = self.sigmoid(z)
        return y * (1 - y)

    def reLU(self, z):
        """
        Rectified linear unit, returns the maximum of 0 or z(i,j)
        :param z: Numpy array of the form X.W.T
        :return:
        """
        return z * (z > 0)
