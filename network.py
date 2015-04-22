__author__ = 'aliowen'
import numpy as np

# Stochastic Gradient Descent
# Minibatch (data shuffled before input to network)
# How do I generalise to n-layers maintaining vectorisation
# Look at the general form of backprop/feedforward.

# List of layer sizes
# Weights and biases are packed for storage
# Unpacked for feedforward/backprop using knowledge of num layers and layer size
# Use dictionary to store values e.g. Theta_n: array()


class Network(object):

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = list()
        self.thetas = list()
        # early stopping
        self.min_bias = list()
        self.min_theta = list()
        # momentum
        self.theta_v = list()
        self.bias_v = list()
        self.min_loss = 100
        # initialise weights
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_layers - 1):
            epsilon = np.sqrt(6) / np.sqrt(self.layers[i] + self.layers[i+1])
            theta = np.random.rand(self.layers[i+1], self.layers[i]) * 2 * epsilon - epsilon
            bias = np.random.randn(self.layers[i+1], 1).T
            self.biases.append(bias)
            self.thetas.append(theta)

    def feedforward(self, a):
        """
        Return network's output based on input a. Used for testings, CV etc
        :param a: Inputs
        :return: Outputs
        """
        for b, t in zip(self.biases, self.thetas):
            a = self.sigmoid(np.dot(a, t.T) + b)
        return a

    def logloss(self, X, y, h):
        return - (1 / X.shape[0]) * np.sum(y * np.log(h))

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, eta, lmbda=0.0, alpha=0.1,
                                    X_cv=None, y_cv=None, hidden_activation='sigmoid',
                                    output_activation='sigmoid', mu=1):
        """
        Train the neural network using stochastic gradient descent (minibatch method).
        :param X: training inputs
        :param y: training labels
        :param epochs: number of times to iterate
        :param mini_batch_size: minibatch size
        :param eta: learning rate hyperparameter
        :param lmbda: L2 regularisation hyperparameter
        :param X_val: cross-validation inputs
        :param y_val: cross_validation labels
        :return:
        """
        m = X.shape[0]
        for j in range(epochs):
            Xy = np.hstack((X,y))
            np.random.shuffle(Xy)
            shuffled_X = Xy[:, :X.shape[1]]
            shuffled_y = Xy[:, -y.shape[1]:]
            for k in range(0, m, mini_batch_size):
                mini_X = shuffled_X[k:k+mini_batch_size, :]
                mini_y = shuffled_y[k:k+mini_batch_size, :]
                self.update_mini_batch(mini_X, mini_y, eta, lmbda, m, mu)
            if X_cv is not None and y_cv is not None:
                h_cv = self.feedforward(X_cv)
                logloss = self.logloss(X_cv, y_cv, h_cv)
                if logloss < self.min_loss:
                    self.min_loss = logloss
                    self.min_theta = self.thetas
                    self.min_bias = self.biases
                loss_rate = logloss / self.min_loss - 1
                # if loss_rate is > alpha , set thetas and biases to best logloss values
                # and exit function
                if j > epochs / 10 and j > 300 and loss_rate > alpha:
                    print("Stopping early, loss rate:", loss_rate, " | ", "Score:", self.min_loss)
                    self.thetas = self.min_theta
                    self.biases = self.biases
                    return None
            if not j % 10:
                h = self.feedforward(shuffled_X)
                J = self.cost_function(shuffled_y, h, m, lmbda)
                print("Epoch", j, " | ", "Cost:", J, " | ", "Logloss:", logloss)

    def update_mini_batch(self, X, y, eta, lmbda, m, mu):
        """
        Update the network's weights and biases by applying gradient descent using the minibatch
        method.
        :param X: minibatch of training examples
        :param y: minibatch of matching targets
        :param eta: learning rate hyperparameter
        :param lmbda: L2 regularisation hyperparameter
        :param m: total number of training examples
        :return: None, weights and biases updated in place
        """

        # get gradients using backprop
        nabla_b, nabla_t = self.backprop(X, y)

        # update weights
        # update bias: b - gradient + momentum
        self.biases = [b - (eta/len(X)) * (nb + mu * (b - nb)) for b, nb in zip(self.biases, nabla_b)]
        # update theta: th - regularisation - gradient + momentum
        self.thetas = [(1 - eta * (lmbda/m)) * t - (eta/len(X)) * (nt + mu * (t - nt))
                       for t, nt in zip(self.thetas, nabla_t)]

    def backprop(self, X, y):
        """
        Calculaye the gradient for weights and biases
        :param X: Training cases
        :param y: Labels
        :return: Nabla(Bias), Nabla(Theta)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_t = [np.zeros(t.shape) for t in self.thetas]
        z_list = list()
        activations = list()

        # feedforward
        activation = X
        activations.append(activation)
        for b, t in zip(self.biases, self.thetas):
            z = np.dot(activation, t.T) + b
            z_list.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backprop
        delta = (activations[-1] - y)
        nabla_b[-1] = delta[0]
        nabla_t[-1] = np.dot(delta.T, activations[-2])
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sigmoid_prime = self.sigmoid_prime(z)
            delta = np.dot(delta, self.thetas[-l+1]) * sigmoid_prime
            nabla_b[-l] = delta.mean(axis=0)
            nabla_t[-l] = (np.dot(delta.T, activations[-l-1]))
        return nabla_b, nabla_t

    def cost_function(self, y, h, m, lmbda):
        J = - 1 / m * np.sum(y * np.log(h) - (1 - y) * np.log(1 - h))
        R = 0
        for t in self.thetas:
            R += lmbda / (2 * m) * self.sumsqr(t)
        return J + R

    def sumsqr(self, a):
        return np.sum(a ** 2)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        y = self.sigmoid(z)
        return y * (1 - y)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        y = self.tanh(z)
        return 1 - np.power(y, 2)

    def softmax(self, z):
        e = np.exp(z)
        return e / np.sum(e)
