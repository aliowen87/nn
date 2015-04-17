__author__ = 'aliowen'
import numpy as np
from scipy import optimize, stats

# TODO: Gradient checking function
# TODO: Take Otto test code into a separate .py

class EarlyStop(Exception):
    pass


class NN():

    def __init__(self,  hidden_layer, hidden_layer_2=0, reg_lambda=0.0,
                 opti_method='CG', maxiter=500, dropout=None, p=0.5, alpha=0.1, activation='sigmoid'):
        self.activation = activation
        activation_dict = {'sigmoid': self.sigmoid, 'sigmoid_prime': self.sigmoid_prime,
                           'tanh': self.tanh, 'tanh_prime': self.tanh_prime}
        self.reg_lambda = reg_lambda
        self.hidden_layer = hidden_layer
        if hidden_layer_2 == 0:
            self.hidden_layer_2 = hidden_layer
        else:
            self.hidden_layer_2 = hidden_layer_2
        self.activation_func = activation_dict[self.activation]
        self.activation_func_prime = activation_dict[self.activation + '_prime']
        self.method = opti_method
        self.maxiter = maxiter
        self.dropout = dropout
        self.p = p
        self.iters = 0
        self.min_loss = 20
        self.cross_val = list()
        self.alpha = alpha

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

    def sumsqr(self, a):
        return np.sum(a ** 2)

    def rand_init(self, layer_in, layer_out):
        epsilon = np.sqrt(6) / np.sqrt(layer_in + layer_out)
        theta = np.random.rand(layer_out, layer_in + 1) * 2 * epsilon - epsilon
        return theta

    def pack_thetas(self, t1, t2, t3):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1), t3.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer, hidden_layer, hidden_layer_2, output_layer):
        t1_start = 0
        t1_end = hidden_layer * (input_layer + 1)
        t2_end = t1_end + hidden_layer_2 * (hidden_layer + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer, input_layer + 1))
        t2 = thetas[t1_end:t2_end].reshape((hidden_layer_2, hidden_layer + 1))
        t3 = thetas[t2_end:].reshape((output_layer, hidden_layer_2 + 1))
        return t1, t2, t3

    def dropout_layer(self, input):
        """
        Apply dropout to input matrix using Bernoulli distribution, can either be the activated
        input matrix a(W x (V +b)) or the weight matrix (LeCun)
        :param input: Input matrix
        :return: Dropped out matrix
        """
        r = stats.bernoulli.rvs(p=self.p, size=input.shape)
        return r * input

    def feed_forward(self, X, t1, t2, t3):
        m = X.shape[0]
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1, )
        else:
            bias = np.ones(m).reshape(m, 1)
        # TODO: Dropout needs to be applied with a separate Bernoulli distro per training case
        # TODO: Refactor into dropout class function
        # dropout some features
        # if self.dropout is True:
        #     r = stats.bernoulli.rvs(0.9, size=X.shape)
        #     X = X * r

        # input layer
        a1 = np.hstack((bias, X))

        # dropconnect
        if self.dropout == 'connect':
            t1 = self.dropout_layer(t1)

        # hidden layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)

        # dropout
        if self.dropout == 'dropout':
            a2 = self.dropout_layer(a2)

        # add bias units
        a2 = np.hstack((bias, a2.T))

        # dropconnect
        if self.dropout == 'connect':
            t2 = self.dropout_layer(t2)

        # hidden layer 2
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)

        # dropout hidden layer 2
        if self.dropout == 'dropout':
            a3 = self.dropout_layer(a3)

        # add bias units
        a3 = np.hstack((bias, a3.T))

        # dropconnect
        if self.dropout == 'connect':
            t3 = self.dropout_layer(t3)

        # output layer
        z4 = np.dot(t3, a3.T)
        a4 = self.sigmoid(z4)
        return a1, z2, a2, z3, a3, z4, a4

    def cost_function(self, thetas, input_layer, hidden_layer, hidden_layer_2, output_layer,
                      X, y, reg_lambda):
        t1, t2, t3 = self.unpack_thetas(thetas, input_layer, hidden_layer, hidden_layer_2, output_layer)
        m = X.shape[0]
        Y = np.eye(output_layer)[y]

        _, _, _, _, _, _, h = self.feed_forward(X, t1, t2, t3)
        J = - np.sum(Y * np.log(h).T + (1 - Y) * np.log(1 - h).T) / m

        # regularisation of cost
        if reg_lambda != 0:
            t1_flat = t1[:, 1:]
            t2_flat = t2[:, 1:]
            R = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1_flat)
                                               + self.sumsqr(t2_flat))
            J = J + R

        return J

    def gradient(self, thetas, input_layer, hidden_layer, hidden_layer_2, output_layer,
                 X, y, reg_lambda):
        """
        Backprop implementation that returns the gradient of the current weights based on the output
        of the feedforward function
        :param thetas: Weight matrices
        :param input_layer: number of features
        :param hidden_layer: size of hidden layer
        :param hidden_layer_2: size of second hidden layer
        :param output_layer: number of classes
        :param X: training cases
        :param y: training targets
        :param reg_lambda: regularisation parameter
        :return: gradient
        """
        t1, t2, t3 = self.unpack_thetas(thetas, input_layer, hidden_layer, hidden_layer_2,
                                        output_layer)
        m = X.shape[0]
        t1_flat = t1[:, 1:]
        t2_flat = t2[:, 1:]
        t3_flat = t3[:, 1:]
        Y = np.eye(output_layer)[y]

        # vectorised implementation
        a1, z2, a2, z3, a3, z4, a4 = self.feed_forward(X, t1, t2, t3)
        # backprop
        d4 = a4 - Y.T
        d3 = np.dot(d4.T, t3_flat) * self.activation_func_prime(z3).T
        d2 = np.dot(d3, t2_flat) * self.activation_func_prime(z2).T

        Theta1_grad = (1 / m) * np.dot(d2.T, a1)
        Theta2_grad = (1 / m) * np.dot(d3.T, a2)
        Theta3_grad = (1 / m) * np.dot(d4, a3)

        if reg_lambda != 0:
            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (reg_lambda / m) * t1_flat
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (reg_lambda / m) * t2_flat
            Theta3_grad[:, 1:] = Theta3_grad[:, 1:] + (reg_lambda / m) * t3_flat

        return self.pack_thetas(Theta1_grad, Theta2_grad, Theta3_grad)

    def fit(self, X, y, X_cv=None, y_cv=None):
        """
        Uses scipy.optimize.minimize function e.g. CG to minimise the cost of the NN using the gradient
        from backpropagation.

        Callback function performs cross validation on each iteration and prints the error every few
        iterations, performs early stopping and updates weights for the lowest CV error.
        :param X: Training inputs
        :param y: Training targets
        :param X_cv: Cross validation inputs
        :param y_cv: Cross validation targets
        :return: None, updates class variables for weights (t1, t2, t3)
        """
        input_layer = X.shape[1]
        output_layer = len(set(y))

        theta1_0 = self.rand_init(input_layer, self.hidden_layer)
        theta2_0 = self.rand_init(self.hidden_layer, self.hidden_layer_2)
        theta3_0 = self.rand_init(self.hidden_layer_2, output_layer)
        thetas0 = self.pack_thetas(theta1_0, theta2_0, theta3_0)

        def stop(thetas):
            """
            Callback function from scipy.optimize.minimize, attempts to implement early stopping to reduce
            overfitting
            :param thetas: Weights for this iteration
            :return: None, will raise an exception if require to stop early
            """
            if X_cv is None or y_cv is None:
                pass
            else:
                self.iters += 1
                input_layer = X.shape[1]
                output_layer = len(set(y))
                t1, t2, t3 = self.unpack_thetas(thetas, input_layer, self.hidden_layer, self.hidden_layer_2,
                                                 output_layer)
                _,_,_,_,_,_,test_results = self.feed_forward(X_cv, t1, t2, t3)
                Y = np.eye(len(set(y)))[y_cv]
                logloss = - (1 / X_cv.shape[0]) * np.sum(Y.T * np.log(test_results))
                if logloss < self.min_loss:
                    self.min_loss = logloss
                    self.t1, self.t2, self.t3 = t1, t2, t3
                loss_rate = logloss / self.min_loss - 1
                if not self.iters % 20:
                    print("Iterations: ", self.iters, "; ", "Loss Rate:", loss_rate, "; ",
                          "Min Loss: ", self.min_loss)
                    # save cross_validation scores for plotting
                    self.cross_val.append(logloss)
                if self.iters > 250 and loss_rate > self.alpha:
                    print("Loss rate too high:", loss_rate)
                    raise EarlyStop

        options = {'maxiter': self.maxiter, 'disp': True}
        _res = optimize.minimize(self.cost_function, thetas0, jac=self.gradient, callback=stop,
                                 method=self.method, args=(input_layer, self.hidden_layer, self.hidden_layer_2,
                                                           output_layer, X, y, self.reg_lambda), options=options)
        self.t1, self.t2, self.t3 = self.unpack_thetas(_res.x, input_layer, self.hidden_layer,
                                                       self.hidden_layer_2, output_layer)

    def check_gradient(self, X, y):
        """
        Gradient checking (troubleshooting function)
        :param X: Training set
        :param y: Training targets
        :return: Error between numerical gradient and grad function
        """
        input_layer = X.shape[1]
        output_layer = len(set(y))

        theta1_0 = self.rand_init(input_layer, self.hidden_layer)
        theta2_0 = self.rand_init(self.hidden_layer, self.hidden_layer_2)
        theta3_0 = self.rand_init(self.hidden_layer_2, output_layer)
        thetas0 = self.pack_thetas(theta1_0, theta2_0, theta3_0)

        error = optimize.check_grad(self.cost_function, self.gradient, thetas0,
                                    input_layer, self.hidden_layer, self.hidden_layer_2, output_layer, X, y,
                                    self.reg_lambda, epsilon=10**-4)
        print('Error=', error)
        return error

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        _, _, _, _, _, _, h = self.feed_forward(X, self.t1, self.t2, self.t3)
        return h


# testing

from sklearn import cross_validation
np.random.seed(seed=15)

import pandas as pd
from sklearn import preprocessing
train = pd.read_csv('../otto/train.csv')
np.random.shuffle(np.array(train))
test = pd.read_csv('../otto/test.csv')
X = train.iloc[:, 1:94]
# attempt to scale X features to its Z-value
X_scaled = (X - X.mean()) / X.std()
y = train.iloc[:, 94]
# convert labels into integers
int_y = np.array([int(q[-1]) - 1 for i, q in enumerate(y)])
# CV split
X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X_scaled, int_y, test_size=0.2)

nn = NN(hidden_layer=50, hidden_layer_2=50, maxiter=500, reg_lambda=2, alpha=0.07, activation='tanh')

if nn.dropout:
    t1 = np.zeros((nn.hidden_layer, X_train.shape[1] + 1))
    t2 = np.zeros((nn.hidden_layer_2, nn.hidden_layer + 1))
    t3 = np.zeros((len(set(y_train)), nn.hidden_layer_2 + 1))
    # number of times to run algorithm
    n = 25
    for i in range(n):
        try:
            nn.fit(X_train, y_train)
        except EarlyStop:
            pass
        t1 += nn.t1
        t2 += nn.t2
        t3 += nn.t3
    t1 = (t1 / n) * nn.p
    t2 = (t2 / n) * nn.p
    t3 = (t3 / n) * nn.p
    _,_,_,_,_,_,test_results = nn.feed_forward(X_cv, t1, t2, t3)
else:
    try:
        nn.fit(X_train, y_train, X_cv, y_cv)
    except EarlyStop:
        pass
    _,_,_,_,_,_,test_results = nn.feed_forward(X_cv, nn.t1, nn.t2, nn.t3)
Y = np.eye(len(set(y)))[y_cv]
logloss = - (1 / X_cv.shape[0]) * np.sum(Y.T * np.log(test_results))
print("Score=", logloss)
print("hidden layer size=", nn.hidden_layer)
print("hidden layer 2 size=", nn.hidden_layer_2)
print("Lambda=", nn.reg_lambda)


# test data
X_test = test.iloc[:, 1:94]
X_test = (X_test - X_test.mean()) / X_test.std()
_,_,_,_,_,_, results = nn.feed_forward(X_test, nn.t1, nn.t2, nn.t3)
# add id back into results
id = np.array(range(X_test.shape[0])) + 1
results = results.T
results_dict = dict()
results_dict['id'] = id
for i in range(len(set(y))):
    results_dict['Class_' + str(i+1)] = results[:, i]


def write_results(results_dict):
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df['id'] = results_df['id'].astype('int32')
    results_df.to_csv('../otto/results.csv', index=False)