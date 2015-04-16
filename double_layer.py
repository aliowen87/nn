__author__ = 'aliowen'
import numpy as np
from scipy import optimize, stats

class NN():

    def __init__(self,  hidden_layer, hidden_layer_2=0, reg_lambda=0.0,
                 opti_method='sigmoid', maxiter=500, dropout=False, p=0.5):
        self.reg_lambda = reg_lambda
        self.hidden_layer = hidden_layer
        if hidden_layer_2 == 0:
            self.hidden_layer_2 = hidden_layer
        else:
            self.hidden_layer_2 = hidden_layer_2
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method
        self.maxiter = maxiter
        self.dropout = dropout
        self.p = p

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

    def feed_forward(self, X, t1, t2, t3):
        m = X.shape[0]
        if len(X.shape) == 1:
            bias = np.array(1).reshape(1, )
        else:
            bias = np.ones(m).reshape(m, 1)
        # dropout some features
        if self.dropout is True:
            r = stats.bernoulli.rvs(self.p, size=X.shape[1])
            X = X * r.T

        # input layer
        a1 = np.hstack((bias, X))

        # hidden layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)

        # dropout hidden layer
        if self.dropout is True:
            r = stats.bernoulli.rvs(self.p, size=self.hidden_layer)
            a2 = a2.T * r
            a2 = a2.T

        # add bias units
        a2 = np.hstack((bias, a2.T))

        # hidden layer 2
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)

        # dropout hidden layer 2
        if self.dropout is True:
            r = stats.bernoulli.rvs(self.p, size=self.hidden_layer_2)
            a3 = a3.T * r
            a3 = a3.T

        # add bias units
        a3 = np.hstack((bias, a3.T))

        # output layer
        z4 = np.dot(t3, a3.T)
        a4 = self.activation_func(z4)
        return a1, z2, a2, z3, a3, z4, a4

    def cost_function(self, thetas, input_layer, hidden_layer, hidden_layer_2, output_layer,
                      X, y, reg_lambda):
        t1, t2, t3 = self.unpack_thetas(thetas, input_layer, hidden_layer, hidden_layer_2, output_layer)
        m = X.shape[0]
        Y = np.eye(output_layer)[y]

        _, _, _, _, _, _, h = self.feed_forward(X, t1, t2, t3)
        J = np.sum(-Y * np.log(h).T - (1 - Y) * np.log(1 - h).T) / m

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

    def fit(self, X, y):
        input_layer = X.shape[1]
        output_layer = len(set(y))

        theta1_0 = self.rand_init(input_layer, self.hidden_layer)
        theta2_0 = self.rand_init(self.hidden_layer, self.hidden_layer_2)
        theta3_0 = self.rand_init(self.hidden_layer_2, output_layer)
        thetas0 = self.pack_thetas(theta1_0, theta2_0, theta3_0)

        options = {'maxiter': self.maxiter, 'disp': True}
        _res = optimize.minimize(self.cost_function, thetas0, jac=self.gradient,
                                 method=self.method, args=(input_layer, self.hidden_layer, self.hidden_layer_2,
                                                           output_layer, X, y, self.reg_lambda), options=options)
        self.t1, self.t2, self.t3 = self.unpack_thetas(_res.x, input_layer, self.hidden_layer,
                                                       self.hidden_layer_2, output_layer)

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

nn = NN(hidden_layer=100, hidden_layer_2=50, maxiter=500, opti_method='CG', reg_lambda=2.21, dropout=True)

if nn.dropout is True:
    t1 = np.zeros((nn.hidden_layer, X_train.shape[1] + 1))
    t2 = np.zeros((nn.hidden_layer_2, nn.hidden_layer + 1))
    t3 = np.zeros((len(set(y_train)), nn.hidden_layer_2 + 1))
    # number of times to run algorithm
    n = 25
    for i in range(n):
        nn.fit(X_train, y_train)
        t1 += nn.t1
        t2 += nn.t2
        t3 += nn.t3
    t1 = (t1 / n) * nn.p
    t2 = (t2 / n) * nn.p
    t3 = (t3 / n) * nn.p
    _,_,_,_,_,_,test_results = nn.feed_forward(X_cv, t1, t2, t3)
else:
    nn.fit(X_train, y_train)
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