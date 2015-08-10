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
        # Velocity for momentum
        self.velocity = 0.0
        # early stopping weights/bias
        self.best_weights = np.zeros(self.weights.shape)
        self.best_bias = np.zeros(self.bias.shape)
        #RMS/Adaprop cache varible
        self.cache = np.zeros(self.weights.shape)


class Network(object):

    def __init__(self, layers, seed=42):
        # seed random number generator
        np.random.seed(seed)
        # initialise layers
        self.layers = [Layer(l, layers[i+1], self.sigmoid)
                       for i, l in enumerate(layers[:-1])]
        # variables for storing errors
        self.train_error = list()
        self.val_error = list()
        self.min_val_err = 100.0

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
        if val is not None:
            val = (val - mean) / std
        if test is not None:
            test = (test - mean) / std
        return train_scaled, val, test


    def pca_whiten(self):
        pass

    def feedforward(self, X):
        activations = list()
        zs = list()
        activation = X
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

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, eta=0.01, lambda_=1.0,
                                    Xval=None, yval=None, momentum="nesterov", alpha=0.1):
        """
        Stochastic gradient descent...
        :param X: Training examples in the form M X N
        :param y: Training labels in the from C * I
        :param epochs: Number of epochs to train
        :param mini_batch_size: Mini batch size
        :param eta: Learning rate
        :param lambda_: L2 regularisation parameter
        :param momentum: Type of momentum to employ, default is 'nesterov'
        :return: None, class variables updated in-place
        """
        # TODO: Finish spec
        # save number of training examples
        m = X.shape[0]

        # primary descent loop
        for j in range(epochs):
            # annealing
            # if j > 0 and j % 5 == 0:
            #     eta *= 0.9
            # reset velocity
            for l in self.layers:
                l.velocity = 0.0

            # combine training samples and labels for minibatch sampling
            Xy = np.hstack((X,y))
            np.random.shuffle(Xy)
            # Split back into examples and labels
            X_shuffled = Xy[:, :X.shape[1]]
            y_shuffled = Xy[:, -y.shape[1]:]

            # minibatch loop
            for k in range(0, m, mini_batch_size):
                # slice minibatches
                mini_X = X_shuffled[k:k+mini_batch_size, :]
                mini_y = y_shuffled[k:k+mini_batch_size, :]
                # perform update
                self.update_mini_batch(mini_X, mini_y, eta, lambda_, m, momentum=momentum)

            # Get the validation cost
            if Xval is not None and yval is not None:
                val_activations, _ = self.feedforward(Xval)
                val_cost = self.cost_function(yval, val_activations[-1], Xval.shape[0], lambda_)
                self.val_error.append(val_cost)

                # early stopping regularisation, calculate loss rate
                loss_rate = val_cost / self.min_val_err - 1

                # save best network weights and bias
                if val_cost < self.min_val_err:
                    self.min_val_err = val_cost
                    for l in self.layers:
                        l.best_weights = l.weights
                        l.best_bias = l.bias

                if j > 10 and loss_rate > alpha:
                    print("Stopping early, epoch %d \t loss rate: %3.f \t Val Error: %.6f"
                          % (j, loss_rate, self.min_val_err))
                    for l in self.layers:
                        l.weights = l.best_weights
                        l.bias = l.best_bias
                    # exit function
                    return None
            else:
                val_cost = 1.0

            # get the training error
            activations, _ = self.feedforward(X)
            cost = self.cost_function(y, activations[-1], m, lambda_)
            self.train_error.append(cost)

            # Get the cost and print out every 10 epochs
            if j % 10 == 0:
                print("Epoch: %d \t Cost: %.6f \t Val Cost: %.6f" % (j, cost, val_cost))


    def update_mini_batch(self, X, y, eta, lambda_, m, mu=0.9, momentum="nesterov",
                          decay=0.99):
        """
        Update weights and bias based on minibatch
        :param X: Minibatch of training examples
        :param y: Minibatch of training labels
        :param eta: learning rate
        :param lambda_: L2 regularisation parameter
        :param decay: RMSProp decay hyperparameter
        :return: None
        """
        # TODO: How do I differentiate between different update strategies, e.g. Momentum, NAG, RMSPROP
        # update gradients with backprop
        self.backprop(X, y)

        # Momentum
        if momentum == "classic":
            for l in self.layers:
                l.bias -= 1 / m * eta * l.nablab
                l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw + lambda_ * l.weights)
                l.weights += l.velocity
            return None

        # Nesterov
        if momentum == "nesterov":
            for l in self.layers:
                l.bias -= 1 / m * eta * l.nablab
                vel_prev = l.velocity
                l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw + lambda_ * l.weights)
                l.weights += - mu * vel_prev + (1 + mu) * l.velocity
            return None

        # RMSProp c.f. Hinton
        # cache = decay * cache + (1 - decay) * grad ** 2
        # weights += - eta * grad / np.sqrt(cache + 1e-8)
        if momentum == 'rmsprop':
            for l in self.layers:
                l.cache = decay * l.cache + (1 - decay) * l.nablaw ** 2
                l.weights -= (eta * l.nablaw) / np.sqrt(l.cache + 1e-8)
            return None

        # default update
        # update weights and bias with simple gradient descent
        for l in self.layers:
            l.bias -= 1 / m * eta * l.nablab
            l.weights -= eta * (1 / m * l.nablaw + lambda_ * l.weights)


    def cost_function(self, y, h, m, lambda_):
        """
        Logloss cost function
        :return: Logloss cost
        """
        J = - 1 / m * np.sum(y * np.log(h) - (1 - y) * np.log(1 - h))
        J = np.sqrt(J**2)
        # TODO: L2 regularisation
        for l in self.layers:
            J += lambda_  / (2 * m) * self.sumsqr(l.weights)
        return J

    def sumsqr(self, x):
        return np.sum(x ** 2)

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

    def gradient_check(self, X, y, eps=1e-5):
        # run gradient function for a few epochs
        epochs = 30
        mini_batch_size = 50
        lambda_ = 0.0
        self.stochastic_gradient_descent(X, y, epochs, mini_batch_size, lambda_=lambda_)
        m = X.shape[0]

        for i, l in enumerate(self.layers):
            # For gradient checking I need to take each element of the gradient
            # and compare it to the numerical gradient calculated by G(W(i) + eps)) - G(W(i) - eps)) / 2*eps
            # Looks like previously I was comparing tht weights rather than gradW, which is obvs wrong
            # Diff should be ~< eps
            weights = l.weights
            gradients = l.nablaw
            plus = weights + eps
            l.weights = plus
            actplus, _ = self.feedforward(X)
            costplus = self.cost_function(y, actplus[-1], m, lambda_)
            minus = weights - eps
            l.weights = minus
            actminus, _ = self.feedforward(X)
            costminus = self.cost_function(y, actminus[-1], m, lambda_)
            grad = (costplus - costminus) / (2 * eps)
            difference = (np.sum(gradients) - grad) / (np.sum(gradients) + grad)
            print("Difference for Layer %d: %.6f" %(i, difference))


