__author__ = 'alaso0'
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize

class Network(object):

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = list()
        self.thetas = list()
        # early stopping
        self.min_bias = list()
        self.min_theta = list()
        # learning curves
        self.train_accuracy = list()
        self.cv_accuracy = list()
        self.min_loss = 100
        # initialise weights
        self.init_weights()
        # activation functions
        self.activation_dict = {'sigmoid': self.sigmoid, 'sigmoid_prime': self.sigmoid_prime,
                                'tanh': self.tanh, 'tanh_prime': self.tanh_prime, 'softmax': self.softmax}
        self.hidden_activation = 'sigmoid'
        self.output_activation = 'softmax'
        # rmsprop
        self.rmsprop_theta = list()
        self.rmsprop_bias = list()

    def init_weights(self):
        for i in range(self.num_layers - 1):
            epsilon = np.sqrt(6) / np.sqrt(self.layers[i] + self.layers[i+1])
            theta = np.random.rand(self.layers[i+1], self.layers[i]) * 2 * epsilon - epsilon
            bias = np.random.randn(self.layers[i+1], 1).T
            self.biases.append(bias)
            self.thetas.append(theta)

    def feedforward(self, a, p=1):
        """
        Return network's output based on input a. Used for testings, CV etc
        :param a: Inputs
        :return: Outputs
        """
        for i, (b, t) in enumerate(zip(self.biases, self.thetas)):
            bias = b * p
            theta = t * p
            if i == len(self.biases):
                a = self.activation_dict[self.output_activation](np.dot(a, theta.T) + bias)
            else:
                a = self.activation_dict[self.hidden_activation](np.dot(a, theta.T) + bias)
        return a

    def dropout(self, X, p):
        """
        Dropout a number of elements depending on
        :param x: matrix to dropout
        :param p: probability of 1
        :return: Dropped out training matrix
        """
        # for each training examples
        # drop a unit (feature or hidden) depending on bernoulli.rvs(p)
        # r.shape = (1, 93)
        r = stats.bernoulli.rvs(p, size=X.shape)
        # for i in range(X.shape[0]):
        #     r[i, :] = stats.bernoulli.rvs(p, size=(X.shape[1]))
        return X * r

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, eta, lmbda=0.0, alpha=0.1,
                                    X_cv=None, y_cv=None, hidden_activation='sigmoid',
                                    output_activation='sigmoid', mu=0, score=False, p=0.5):
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
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        m = X.shape[0]
        for j in range(epochs):
            Xy = np.hstack((X, y))
            np.random.shuffle(Xy)
            # dropout
            shuffled_X = Xy[:, :X.shape[1]]
            shuffled_y = Xy[:, -y.shape[1]:]
            for k in range(0, m, mini_batch_size):
                mini_X = shuffled_X[k:k+mini_batch_size, :]
                mini_y = shuffled_y[k:k+mini_batch_size, :]
                # tune momentum
                if k < 25:
                    mu = 0.5
                self.update_mini_batch(mini_X, mini_y, eta, lmbda, m, mu, p)
            if score:
                # save train accuracy for each epoch
                h = self.feedforward(X, p)
                J = self.cost_function(y, h, m, lmbda, p)
                self.train_accuracy.append(J)
            if X_cv is not None and y_cv is not None:
                h_cv = self.feedforward(X_cv, p)
                logloss = self.logloss(X_cv.shape[0], y_cv, h_cv)
                # save CV accuracy for each epoch
                self.cv_accuracy.append(logloss)
                if logloss < self.min_loss:
                    self.min_loss = logloss
                    self.min_theta = self.thetas
                    self.min_bias = self.biases
                loss_rate = logloss / self.min_loss - 1
                # if loss_rate is > alpha , set thetas and biases to best logloss values
                # and exit function
                if j > epochs / 5 and loss_rate > alpha:
                    print("Stopping early, iter:", j, " |  loss rate:", loss_rate, " | ",
                          "Score:", self.min_loss)
                    self.thetas = self.min_theta
                    self.biases = self.min_bias
                    return None
            if not j % 10 and score:
                print("Epoch", j, " | ", "Cost:", J, " | ", "CV Logloss:", logloss)
        # regularise by using the min CV values
        self.biases = self.min_bias
        self.thetas = self.min_theta

    def update_mini_batch(self, X, y, eta, lmbda, m, mu, p=1):
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
        nabla_b, nabla_t = self.backprop(X, y, p)

        # # update weights
        # # update bias: b - gradient + momentum
        # self.biases = [b - (1 / len(X)) * (eta * nb - mu * (b - nb)) for b, nb in zip(self.biases, nabla_b)]
        # # update theta: th - L2 regularisation - gradient + momentum
        # self.thetas = [(1 - eta * (lmbda/m)) * t - (1 / len(X)) * (eta * nt - mu * (t - nt))
        #                for t, nt in zip(self.thetas, nabla_t)]

        # Nesterov accelerated gradient
        # update biases
        self.biases = [b - (1 / len(X)) * (eta * nb - mu * (b - nb)) for b, nb in zip(self.biases, nabla_b)]
        # update weights
        self.thetas = [(1 - eta * (lmbda/m)) * t - (1 / len(X)) * (eta * nt - mu * (t - nt))
                       for t, nt in zip(self.thetas, nabla_t)]

    def backprop(self, X, y, p=1):
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
        for i, (b, t) in enumerate(zip(self.biases, self.thetas)):
            if i > 0 and p < 1:
                activation = self.dropout(activation, p)
            z = np.dot(activation, t.T) + b
            z_list.append(z)
            if i == len(self.biases):
                # output layer activation
                activation = self.activation_dict[self.output_activation](z)
            else:
                activation = self.activation_dict[self.hidden_activation](z)
            activations.append(activation)
            i += 1
        # backprop
        delta = (activations[-1] - y)
        nabla_b[-1] = delta[0]
        nabla_t[-1] = np.dot(delta.T, activations[-2])
        for l in range(2, self.num_layers):
            z = z_list[-l]
            activation_prime = self.activation_dict[self.hidden_activation + '_prime'](z)
            delta = np.dot(delta, self.thetas[-l+1]) * activation_prime
            nabla_b[-l] = delta.mean(axis=0)
            nabla_t[-l] = (np.dot(delta.T, activations[-l-1]))
        return nabla_b, nabla_t

    def cost_function(self, y, h, m, lmbda, p=1):
        R = 0
        if self.output_activation == 'softmax':
            J = self.logloss(m, y, h)
        else:
            J = - 1 / m * np.nan_to_num(np.sum(y * np.log(h) - (1 - y) * np.log(1 - h)))
            for t in self.thetas:
                R += lmbda / (2 * m) * self.sumsqr(t * p)
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

    def logloss(self, m, y, h):
        return - (1 / m) * np.sum(y * np.log(h))

    def check_gradient(self, X, y):
        """
        Gradient checking (troubleshooting function)
        :param X: Training set
        :param y: Training targets
        :return: Error between numerical gradient and grad function
        """
        
        # initialise weights
        self.init_weights()

        error = optimize.check_grad(self.cost_function, self.backprop, self.thetas, X, y,
                                    epsilon=10**-4)
        print('Error=', error)
        return error
