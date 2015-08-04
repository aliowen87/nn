__author__ = 'alaso0'
import numpy as np
import scipy.stats as stats

class Layer(object):

    def __init__(self, size, activation):
        pass


class Network(object):

    def __init__(self, layers):
        pass

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
        of the layer using probability p. Returns dropped-out layer
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

    def feedforward(self):
        pass

    def backprop(self):
        pass

    def stochastic_gradient_descent(self):
        pass

    def update_mini_batch(self):
        pass

    def cost_function(self):
        pass
