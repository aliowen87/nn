__author__ = 'aliowen87'
import numpy as np

class Layer(object):

    def __init__(self, inner, outer, activation):
        """
        Layer class is used to create the layers of the network, initialising the weights and biases and
        setting up class variables for parameters that are updated with each minibatch (e.g. caching for
        RMSPROP or velocity for momentum).
        :param inner: Number of connections into the layer
        :param outer: Number of connections out of the layer
        :param activation: Activation function to use for the layer
        :return:
        """
        self.inner = inner
        self.outer = outer
        self.activation = activation
        # initialise weights and bias
        if activation == 'relu':
            self.weights = np.random.randn(self.outer, self.inner) * np.sqrt(2.0 / (self.inner + self.outer))
        else:
            self.weights = np.random.randn(self.outer, self.inner) / np.sqrt(self.inner + self.outer)
        # create bias numpy array
        self.bias = np.random.randn(self.outer, 1).T
        # placeholder numpy array for gradient of weights and biases
        self.nablaw = np.zeros(self.weights.shape)
        self.nablab = np.zeros(self.bias.shape)
        # Velocity parameter for momentum
        self.velocity = 0.0
        # early stopping weights/bias
        self.best_weights = np.zeros(self.weights.shape)
        self.best_bias = np.zeros(self.bias.shape)
        #RMS/Adaprop cache variable
        self.cache = np.zeros(self.weights.shape)

# TODO: Convolutional and pooling layer extensions to Layer
class ConvolutionalLayer(Layer):
    pass

class Network(object):

    def __init__(self, layers, seed=42):
        """
        The Network class contains the activation and stochastic gradient descent functions necessary to
        carry out learning.
        :param layers: list of tuples of the form (layer_size, activation type) e.g. (200, 'relu'), the output layer
        activation should be set to None.
        :param seed: seed random number generator, default=42
        """

        # seed random number generator
        np.random.seed(seed)

        # dictionary of activation functions
        self.activation_functions = {
            'sigmoid': self.sigmoid,
            'sigmoid_prime': self.sigmoid_prime,
            'relu': self.reLU,
            'relu_prime': self.reLU_prime,
            'tanh': self.tanh,
            'tanh_prime': self.tanh_prime
        }

        # initialise layers
        self.layers = [Layer(l[0], layers[i+1][0], l[1])
                       for i, l in enumerate(layers[:-1])]

        # variables for storing errors for plotting once learning is complete
        self.train_error = list()
        self.val_error = list()
        # storage variable for early stopping
        self.min_val_err = 100.0

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


    def pca_whiten(self, X, whiten=False, n_components=None, eps=1e-5):
        """
        Carries out principle component analysis with optional whitening
        :param X: Input data
        :param whiten: Whether to whiten data, True or False
        :param n_components: number of components to reduce to, defaults to X.shape[1] - 1
        :param eps: Constant to prevent division by 0, default 1e-5. Increase if data is of the same magnitude
        :return: Data with n_components parameters, whitened is whiten=True
        """
        # TODO: Automatically retain X% variance
        # if n_components is undefined or > number of parameters then set to default
        if n_components is None or n_components > X.shape[1]:
            n_components = X.shape[1] - 1
        # centre the data on zero
        X -= np.mean(X, axis=0)
        # calculate the covariance matrix
        cov = np.dot(X.T, X) / X.shape[0]
        # Singular value decomposition
        U, S, V = np.linalg.svd(cov)
        # Print variance retained
        print("Variance retained: %.1f%%" % (np.sum(S[:n_components])/np.sum(S) * 100))
        # project zero-centred data onto eigenbasis
        X_rotated = np.dot(X, U)
        # PCA
        X_rotated_reduced = np.dot(X, U[:, :n_components])
        # whitening if flag set to True
        if whiten:
            Xwhitened = X_rotated / np.sqrt(S + eps)
            return Xwhitened

        return X_rotated_reduced

    def feedforward(self, X, p=1.0):
        """
        Standard feedforward algorithm
        :param X: training examples in the shape m examples x n features
        :param p: dropout probability, default=1 i.e. no dropout
        :return: list of activated arrays and dot products
        """
        # list of activations and activated z's for passing to backprop
        activations = list()
        zs = list()
        activation = X
        # loop through each layer and apply that layer's activation function the the previous' activated output
        # e.g. z1 = sigmoid(X.w1.T + b1) -> z2 = sigmoid(z1.w2.T + b2) etc
        for l in self.layers:
            z = np.dot(activation, l.weights.T) + l.bias
            # save z for backprop
            zs.append(z)
            activation = self.activation_functions[l.activation](z)
            # dropout
            if p < 1.0:
                activation = self.inverted_dropout(activation, p=p)
            # save activation for backprop
            activations.append(activation)
        return activations, zs

    def predict(self, X, p=1.0):

        act, z = self.feedforward()
        return act[-1]


    def backprop(self, X, y, p=1.0):
        """
        Backpropogation algorithm
        :param X: training examples in the shape m examples x n features
        :param y: target values in binary array, shape m examples x num classes
        :param p: dropout probability, default = 1 i.e. no dropout
        :return: None, weights and biases updated in Layer class arrays
        """
        # get activations and z-values
        A, Z = self.feedforward(X, p=p)

        # initial error, difference between the final output layer and y.
        delta = (A[-1] - y)
        # store bias gradient as initial error
        self.layers[-1].nablab = delta[0]

        self.layers[-1].nablaw = np.dot(delta.T, A[-2])
        for i in range(2, len(self.layers)):
            z = Z[-i]
            activation_prime = self.activation_functions[self.layers[-i].activation + '_prime'](z)
            delta = np.dot(delta, self.layers[-i+1].weights) * activation_prime

            self.layers[-i].nablab = delta.mean(axis=0)
            self.layers[-i].nablaw = np.dot(delta.T, A[-i-1])


    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, eta=0.01, lambda_=0.0,
                                    Xval=None, yval=None, momentum="nesterov", alpha=0.1, p=1):
        """
        Stochastic gradient descent...
        :param X: Training examples in the form M X N
        :param y: Training labels in the from C * I
        :param epochs: Number of epochs to train
        :param mini_batch_size: Minibatch size
        :param eta: Learning rate
        :param lambda_: L2 regularisation parameter
        :param momentum: :param momentum: Type of stochastic update strategy to use, 'classic'=momentum,
        'nag'=nesterov accelerated gradient, 'rmsprop'=RMSProp, else use standard update rule. Default is 'nag'
        :param alpha: early stopping hyperparameter, will stop when validation error increasing by this factor
        :param Xval: validation array in the shape m examples by n features
        :param yval: validation array in the shape m examples by num classes
        :return: None, class variables updated in-place
        """
        # save number of training examples
        m = X.shape[0]
        # placeholder for validation error
        val_cost = 1e4

        # primary descent loop
        for j in range(epochs):
            # TODO: consider adding annealing or similar back in to speed up learning
            # if j > 0 and j % 5 == 0:
            #     eta *= 0.9

            # reset velocity each epoch
            for l in self.layers:
                l.velocity = 0.0

            # combine training samples and labels for minibatch sampling
            Xy = np.hstack((X,y))
            # shuffle t
            np.random.shuffle(Xy)

            # Split back into examples and class labels
            X_shuffled = Xy[:, :X.shape[1]]
            y_shuffled = Xy[:, -y.shape[1]:]

            # minibatch loop
            for k in range(0, m, mini_batch_size):
                # slice minibatches
                mini_X = X_shuffled[k:k+mini_batch_size, :]
                mini_y = y_shuffled[k:k+mini_batch_size, :]
                # perform update
                self.update_mini_batch(mini_X, mini_y, eta, lambda_, m, momentum=momentum, p=p)

            # If supplied with validation data, calculate validation error and perform early stopping
            if Xval is not None and yval is not None:
                val_activations, _ = self.feedforward(Xval)
                val_cost = self.cost_function(yval, val_activations[-1], Xval.shape[0], lambda_)
                self.val_error.append(val_cost)

                # early stopping regularisation, calculate loss rate
                loss_rate = float(val_cost / self.min_val_err - 1)

                # save best network weights and bias
                if val_cost < self.min_val_err:
                    self.min_val_err = val_cost
                    for l in self.layers:
                        l.best_weights = l.weights
                        l.best_bias = l.bias

                # early stopping starts tracking after first 10 epochs to allow for initial fluctuations at
                # high learning rates
                if j > 10 and loss_rate > alpha:
                    print("Stopping early, epoch %d \t loss rate: %.3f \t Val Error: %.6f"
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

            # Print the training error every 10 epochs
            if j % 10 == 0:
                print("Epoch: %d \t Cost: %.6f \t Val Cost: %.6f" % (j, cost, val_cost))

        # end loop (early stopping criterion not exceeded), print results
        print("Epoch: %d \t Cost: %.6f \t Val Cost: %.6f" % (epochs, cost, val_cost))


    def update_mini_batch(self, X, y, eta, lambda_, m, mu=0.9, momentum="nag",
                          decay=0.99, p=1.0):
        """
        Update weights and bias based on minibatch
        :param X: Minibatch of training examples
        :param y: Minibatch of training labels
        :param eta: learning rate
        :param lambda_: L2 regularisation parameter
        :param decay: RMSProp decay hyperparameter
        :param momentum: Type of stochastic update strategy to use, 'classic'=momentum, 'nag'=nesterov accelerated
        gradient, 'rmsprop'=RMSProp, else use standard update rule. Default is 'nag'
        :return: None
        """
        try:
            assert momentum is str
        except AssertionError:
            print('Invalid momentum value used, should be a string')
            momentum = 'nag'
        momentum = momentum.lower()

        # update gradients with backprop
        self.backprop(X, y, p=p)

        # Momentum c.f. http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        if momentum == "classic":
            for l in self.layers:
                l.bias -= 1 / m * eta * l.nablab
                l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw - lambda_ * l.weights)
                l.weights += l.velocity
            return None

        # Nesterov accelerated gradient (NAG) c.f http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        if momentum == "nag":
            for l in self.layers:
                l.bias -= 1 / m * eta * l.nablab
                vel_prev = l.velocity
                l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw - lambda_ * l.weights)
                l.weights += - mu * vel_prev + (1 + mu) * l.velocity
            return None

        # RMSProp c.f. Hinton http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
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
            l.weights -= eta * (1 / m * l.nablaw - lambda_ * l.weights)


    def cost_function(self, y, h, m, lambda_):
        """
        Logloss cost function
        :param y: Binary array of ground truth in the shape m x num classes
        :param h: Output layer from network (i.e. the predictions)
        :param m: Number of training examples
        :return: Logloss cost
        """
        # cross-entropy error/cost function c.f. https://en.wikipedia.org/wiki/Cross_entropy
        J = - 1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # L2 regularisation
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
        Computes sigmoid gradient of numpy array of the form sigmoid(z) * (1 - sigmoid(z))
        :param z: Numpy array of the form activation - y/ grad.T.activation[-1] etc
        :return: Sigmoid gradient (numpy array)
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


    def reLU_prime(self, z):
        """
        Computes ReLU gradient of numpy array, 1 if z(i) > 0
        :param z: Numpy array of the form activation - y/ grad.T.activation[-1] etc
        :return: ReLU gradient (numpy array)
        """
        z[z <= 0] = 0
        return z


    def tanh(self, z):
        return np.tanh(z)


    def tanh_prime(self, z):
        y = self.tanh(z)
        return 1 - np.power(y, 2)


    def gradient_check(self, X, y, eps=1e-5):
        # TODO: Complete/fix gradient check code
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

# TODO: Preprocessing and Metrics classes

class Preprocessing(object):

    def __init__(self):
        pass

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


    def pca_whiten(self, X, whiten=False, n_components=None, eps=1e-5):
        """
        Carries out principle component analysis with optional whitening
        :param X: Input data
        :param whiten: Whether to whiten data, True or False
        :param n_components: number of components to reduce to, defaults to X.shape[1] - 1
        :param eps: Constant to prevent division by 0, default 1e-5. Increase if data is of the same magnitude
        :return: Data with n_components parameters, whitened is whiten=True
        """
        # TODO: Automatically retain X% variance
        # if n_components is undefined or > number of parameters then set to default
        if n_components is None or n_components > X.shape[1]:
            n_components = X.shape[1] - 1
        # centre the data on zero
        X -= np.mean(X, axis=0)
        # calculate the covariance matrix
        cov = np.dot(X.T, X) / X.shape[0]
        # Singular value decomposition
        U, S, V = np.linalg.svd(cov)
        # Print variance retained
        print("Variance retained: %.1f%%" % (np.sum(S[:n_components])/np.sum(S) * 100))
        # project zero-centred data onto eigenbasis
        X_rotated = np.dot(X, U)
        # PCA
        X_rotated_reduced = np.dot(X, U[:, :n_components])
        # whitening if flag set to True
        if whiten:
            Xwhitened = X_rotated / np.sqrt(S + eps)
            return Xwhitened

        return X_rotated_reduced


class Metrics(object):

    def __init__(self):
        pass

    def accuracy(y, h):
        # pick the highest score in each row
        highest = np.zeros(h.shape)
        for i, x in enumerate(h):
            highest[i, x.argmax()] = 1.0
        # sum by row and then sum again
        return (sum(np.sum(y * h, axis=0)) / y.shape[0]) * 100.0

    def f1_score(y, h):
        # pick the highest score in each row
        highest = np.zeros(h.shape)
        for i, x in enumerate(h):
            highest[i, x.argmax()] = 2
        compare = (y - highest).astype(int)
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for x in compare:
            if x.sum() == -2:
                false_pos += 1
            if x.sum() == -1:
                true_pos += 1
            if x.sum() == 0:
                true_neg += 1
            if x.sum() == 1:
                false_neg =+1
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        # calculate F1 score
        return 2 * (precision * recall) / (precision + recall)


