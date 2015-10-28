__author__ = 'aliowen87'
import numpy as np
import random


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
        self.type = 'fc'
        eps = 1
        # initialise weights and bias
        if activation != 'sigmoid':
            # relu requires carful initialisation to prevent exploding activations
            # size = (self.outer  * np.sqrt(2.0 / (self.inner + self.outer)))
            self.weights = np.random.randn(self.outer, self.inner) * eps # * np.sqrt(2.0 / (self.inner + self.outer))
            self.bias = np.random.randn(self.outer, 1).T * eps
        else:
            self.weights = np.random.randn(self.outer, self.inner) / np.sqrt(self.inner + self.outer)
            self.bias = np.random.randn(self.outer, 1).T

        # placeholder numpy array for gradient of weights and biases
        self.nablaw = np.zeros_like(self.weights)
        self.nablab = np.zeros_like(self.bias)
        # Velocity parameter for momentum
        self.velocity = 0.0
        # early stopping weights/bias
        self.best_weights = np.zeros_like(self.weights)
        self.best_bias = np.zeros_like(self.bias)
        #RMS/Adaprop cache variable
        self.cache = np.zeros_like(self.weights)


class ConvolutionalLayer(object):
    """
    # Accepts a volume of W1 x H1 x D1
    # Hyperparameters k: no of filters, f: spatial extent of filters, s: stride, p: zero-padding
    # Produces a volume W2 x H2 x D2 where
    # W2 = (W1 - F + 2P) / S + 1
    # H2 = (H1 - F + 2P) / S + 1
    # D2 = K
    # Results in F.F.D1 weights per filter i.e. (F.F.D1).K weights and K biases
    """

    def __init__(self, channels, num_filters, filter_size, activation='relu'):
        """
        :param channels: Number of channels in the input matrix, i.e. depth of matrix
        :param num_filters: number of filters to use (K)
        :param filter_size: width and height of filter (assumed square)
        :return:
        """
        self.type = 'conv'
        self.depth = channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        # check filter size is odd
        assert self.filter_size % 2 == 1, 'Filter size must be odd, %d is even' % self.filter_size

        # # initialise weights and biases
        # if self.activation == 'relu':
        #     eps = 5e-3
        #     self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size) * eps
        #     self.bias = np.random.randn(self.num_filters) * eps
        # else:
        denom = np.sqrt(self.num_filters + self.depth + self.filter_size + self.filter_size)
        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size) / denom
        self.bias = np.random.randn(self.num_filters)

        # placeholder numpy array for gradient of weights and biases
        self.nablaw = np.zeros_like(self.weights)
        self.nablab = np.zeros_like(self.bias)

        # calculate stride and padding hyperparams
        self.stride, self.padding = self.calc_stride_padding()

        # Velocity parameter for momentum
        self.velocity = 0.0
        # early stopping weights/bias
        self.best_weights = np.zeros_like(self.weights)
        self.best_bias = np.zeros_like(self.bias)
        #RMS/Adaprop cache variable
        self.cache = np.zeros_like(self.weights)


    def calc_stride_padding(self):
        """
        Checks the filter sizes are appropriate and calculates the stride and padding for the convolution
        :return: Calculated stride and padding
        """
        filter_height, filter_width = self.filter_size, self.filter_size
        assert filter_height == filter_width, 'Filter height must equal width (i.e. square)'
        assert filter_height % 2 == 1, 'Filter 1d should be odd'
        # currently makes the assumption that stride = 1, expression for padding is more complex if this isn't the
        # case. Padding will default to 1
        stride = 1
        padding = (filter_height - 1) / 2
        return stride, padding


class MaxPoolLayer(object):
    """
    Pooling layer that performs the max pooling action, used after a ConvolutedLayer
    """
    def __init__(self, filter_width=2, filter_height=2, stride=2):
        """
        :param filter_width: Size of the filter 'width', default = 3
        :param filter_height: Size of the filter 'height', default = 3
        :param stride: Stride to take with each filter, default = 2
        """
        self.type = 'pool'
        self.filter_height, self.filter_width = filter_height, filter_width
        self.stride = stride
        # class variable to store the indices of max
        self.indices = list()
        self.weights = 0

    def pool_function(self, x, type='reshape'):
        """
        Returns the max value in x, assumed an array of shape filter width x filter height
        :param x: filtered input
        :return: Max value
        """
        if type == 'reshape':
            return x.max(axis=3).max(axis=4)
        else:
            return x.max(), x.argmax()


class Network(object):
    """
    Main neural network class, sets up the network architecture
    """
    def __init__(self, layers, seed=42):
        """
        The Network class contains the activation and stochastic gradient descent functions necessary to
        carry out learning.
        :param layers: # Class takes a list-dict specifying the
        e.g FC layer {type: 'fc', size: 512, activation:'tanh'}
        e.g. Conv Layer {type: 'conv', depth:3, num_filters:10, filter_size:3}
        e.g. Pool Layer {type: 'pool', filter_width: 2, filter_height:2, stride:2, activation:'max'}
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
            'tanh_prime': self.tanh_prime,
            'softmax': self.softmax,
            'softmax_prime': self.softmax_prime
        }

        # initialise layers
        self.layers = list()
        for layer in layers:
            keys = list(layer.keys())
            # TODO: simplify with kwargs
            if layer['type'] == 'conv':
                depth = layer['depth']
                num_filters = layer['num_filters']
                filter_size = layer['filter_size']
                self.layers.append(ConvolutionalLayer(depth, num_filters, filter_size))
            elif layer['type'] == 'pool':
                activation = layer['activation']
                filter_width = layer['filter_width']
                filter_height = layer['filter_height']
                stride = layer['stride']
                # TODO: add logic test for pool type once others are implemented
                self.layers.append(MaxPoolLayer(filter_width, filter_height, stride))
            else:
                activation = layer['activation']
                inner = layer['inner']
                outer = layer['outer']
                self.layers.append(Layer(inner, outer, activation))

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


    def feedforward(self, X, p=1.0):
        """
        Standard feedforward algorithm
        :param X: training examples in the shape m examples by n features
        :param p: dropout probability, default=1 i.e. no dropout
        :return: list of activated arrays and dot products
        """
        # list of activations and activated z's for passing to backprop
        activations = list()
        zs = list()
        activation = X
        # loop through each layer and apply that layer's activation function the the previous' activated output
        # e.g. a1 = sigmoid(X.w1.T + b1) -> a2 = sigmoid(z1.w2.T + b2) etc
        # TODO: feedforward algorithm could be abstracted into the layer classes for simplicity
        for layer in (self.layers):
            if layer.type == 'conv':
                # Convolution
                z = self.conv_feedforward(activation, layer)
                zs.append(z)
                activation = self.activation_functions[layer.activation](z)
                activations.append(activation)
            elif layer.type == 'pool':
                # Pool
                z = self.pool_feedforward(activation, layer)
                zs.append(z)
                # pooling = activation
                activation = z
                activations.append(z)
            else:
                # if the previous layer was a conv/pool layer then will need to convert from a 4D matrix to 2D
                if len(activation.shape) > 2:
                    activation = self.sparse_to_dense(activation)
                z = np.dot(activation, layer.weights.T) + layer.bias
                # save z for backprop
                zs.append(z)
                activation = self.activation_functions[layer.activation](z)
                # dropout
                if p < 1.0:
                    activation = self.inverted_dropout(activation, p=p)
                # save activation for backprop
                activations.append(activation)
        return activations, zs

    def sparse_to_dense(self, x):
        flat = x.shape[1] * x.shape[2] * x.shape[3]
        return x.reshape((x.shape[0], flat))

    def dense_to_sparse(self, x, shape):
        return x.reshape(shape)

    def predict(self, X, p=1.0):
        """
        Convenience function to get the hypothesis (final activation) of the network, use when testing/validating
        network
        :param X: training examples in the shape m examples by n features
        :param p: dropout probability, default=1 i.e. no dropout
        :return: Hypothesis of network
        """
        act, z = self.feedforward(X, p)
        return act[-1]

    def conv_feedforward(self, X, conv_layer):
        """
        Feedforward function for convolution layer
        :param X: Input examples in the shape (number of examples, depth, height, width)
        :param conv_layer: Convoluted layer
        :return: Convolved matrix in the shape (number of examples, num_filters, H2, W2)
        H2 = (h + 2 * padding - filter_height) / stride + 1
        W2 = (w + 2 * padding - filter_height) / stride + 1
        """
        n, depth, height, width = X.shape
        # cache shape in class variable
        conv_layer.shape = X.shape

        num_filters, _, filter_height, filter_width = conv_layer.weights.shape
        stride, padding = conv_layer.stride, conv_layer.padding

        # create output placeholder
        output_height = (height + 2 * padding - filter_height) / stride + 1
        output_width = (width + 2 * padding - filter_width) / stride + 1
        output = np.zeros((n, num_filters, output_height, output_width))

        # using im2col
        X_cols = self.im2col(X, filter_height, filter_width, padding, stride)
        # cache in class variable
        conv_layer.X_cols = X_cols

        # collapses weights into matrix shape k, fw * fh * d
        f = conv_layer.weights.shape[0]
        weights = conv_layer.weights.reshape((f, -1))
        weights = np.dot(weights, X_cols) + conv_layer.bias.reshape(-1, 1)
        output = weights.reshape(f, output_height, output_width, X.shape[0])
        output = output.transpose(3, 0, 1, 2)

        return output

    def im2col(self, X, filter_height, filter_width, padding=1, stride=1):
        """
        Numpy implementation of the im2col method (distinct)
        :param X: Input matrix
        :param filter_height: filter height
        :param filter_width: filter width
        :param padding: depth of zero-padding to be applied, default = 1
        :param stride: stride length, default = 1
        :return: X with padding applied as a flattened convolution
        """
        # apply zero-padding, (0,0) means no padding applied
        X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

        # get indices
        k, i, j = self.im2col_idx(X.shape, filter_height, filter_width, padding, stride)
        d = X.shape[1]
        X_cols = X_padded[:, k, i, j]
        X_cols = X_cols.transpose(1, 2, 0).reshape(filter_height * filter_width * d, -1)
        return X_cols


    def im2col_idx(self, shape, filter_height, filter_width, padding, stride):
        """
        Convenience function to get the indices for the im2col and col2im functions based on the filter
        size and output size
        :param d: depth of the input matrix
        :param filter_height: filter height
        :param filter_width: filter width
        :param stride:
        :return:
        """
        n, d, h, w = shape
        # check filters are of the right size
        assert ((h + 2 * padding - filter_height) % stride == 0)
        assert ((w + 2 * padding - filter_width) % stride == 0)
        # calculate size of the output of the convolution
        out_h = (h + 2 * padding - filter_height) / stride + 1
        out_w = (w + 2 * padding - filter_width) / stride + 1

        # calculate the indices
        i0 = np.tile(np.repeat(np.arange(filter_height), filter_width), d)
        i1 = stride * np.repeat(np.arange(out_h), out_w)
        # combine, -1 transposes the elements in that dimension essentially i0 = rows, i1 = columns
        i = i0.reshape(-1, 1) + i1.reshape(1, -1).astype(int)

        j0 = np.tile(np.arange(filter_width), filter_height * d)
        j1 = stride * np.tile(np.arange(out_w), out_h)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1).astype(int)

        k = np.repeat(np.arange(d), filter_height * filter_width).reshape(-1, 1).astype(int)

        return k, i, j

    def col2im(self, cols, shape, filter_height, filter_width, padding=1, stride=1):
        """
        Inverse operation of im2col
        :param cols: col2im representation of original matrix
        :param shape: Shape of the input matrix
        :param filter_height:
        :param filter_width:
        :param padding:
        :param stride:
        :return: Padded matrix prior to convolution
        """
        n, d, h, w, = shape
        h_padded, w_padded = h + 2 * padding, w + 2 * padding
        X_padded = np.zeros((n, d, h_padded, w_padded), dtype=cols.dtype)
        k, i, j = self.im2col_idx(shape, filter_height, filter_width, padding, stride)

        cols_reshaped = cols.reshape((filter_width * filter_height * d), -1, n)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(X_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return X_padded
        return X_padded[:, :, padding:-padding, padding:-padding]


    def pool_feedforward(self, X, pool_layer):
        """
        Feedforward function for pooling layer
        :param X: Training examples, typically after passing through a convoluted layer
        :param pool_layer: Pooling layer
        :return: Pooled matrix in the shape (
        """
        n, depth, height, width = X.shape
        filter_height, filter_width, stride = pool_layer.filter_height, pool_layer.filter_width, pool_layer.stride

        # create output placeholders
        output_width = int((width - filter_width) / stride + 1)
        output_height = int((height - filter_height) / stride + 1)
        output = np.zeros((n, depth, output_height, output_width))

        # TODO: Less retarded way of doing it, i.e. im2col or matrix reshaping
        # reshape method, quickest method if filter is square and stride is the same size
        if filter_height == filter_width:
            output, pool_layer.reshaped = self.pool_reshape_forward(X, pool_layer)
        else:
            # slow way
            for i in range(n):
                for z in range(depth):
                    for w in range(output_width):
                        for h in range(output_height):
                            posw = w * stride
                            posh = h * stride
                            # create filter_width x filter_height filter at depth d
                            filter_ = X[i, z, posh:filter_height+posh, posw:filter_width+posw]
                            output[i, z, h, w], index = pool_layer.pool_function(filter_)
                            # save location of the value picked by the pool function, e.g. max in layer class variable
                            pool_layer.indices.append((i, z, (index + h), (index + w)))

        return output

    def pool_reshape_forward(self, X, pool_layer):
        n, d, h, w  = X.shape
        filter_height, filter_width, stride = pool_layer.filter_height, pool_layer.filter_width, pool_layer.stride
        # error checks
        assert filter_height == filter_width == stride, 'Pool parameters not equal w = h = s'
        assert h % filter_height == 0, 'Pool height doesn\'t tile with input'
        assert w % filter_width == 0, 'Pool width doesn\'t tile with input'

        # reshape input matrix
        x_reshaped = X.reshape(n, d, h / filter_height, filter_height, w / filter_width, filter_width)
        output = pool_layer.pool_function(x_reshaped)
        # cache the shape of the layer
        pool_layer.shape = output.shape

        return output, x_reshaped


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
        # TODO: Should I be using the cross-entropy/softmax function here rather than a difference?
        output = self.layers[-1]
        delta = (A[-1] - y)

        # store bias gradient as initial error
        output.nablab = delta.mean()
        output.nablaw = np.dot(delta.T, A[-2])

        for i in range(2, len(self.layers)):
            #backprop is different for pool layers
            layer = self.layers[-i]
            print('d', delta.mean(), 'a', A[-i].mean())
            # backprop for pool layer
            if layer.type == 'pool':
                a = A[-i-1]
                # TODO: check for what type of feedforward was used e.g reshape or loopception
                # reshape from fc layer, assumes that filter is square...
                if len(delta.shape) == 2:
                    delta = self.dense_to_sparse(delta, layer.shape)
                delta = self.pool_backprop_reshape(delta, a, layer)

            # perform backprop for a convolution layer
            elif layer.type == 'conv':
                delta = self.conv_backprop(delta, layer)

            # fc layer backprop
            else:
                z = Z[-i]
                activation_prime = self.activation_functions[layer.activation + '_prime'](z)
                delta = np.dot(delta, self.layers[-i+1].weights) * activation_prime

                # update gradients
                layer.nablab = delta.mean(axis=0)
                # check shape of previous activation, reshape if necessary
                if len(A[-i-1].shape) > 2:
                    a = self.sparse_to_dense(A[-i-1])
                    self.layers[-i].nablaw = np.dot(delta.T, a)
                else:
                    layer.nablaw = np.dot(delta.T, A[-i-1])

    def conv_backprop(self, delta, conv_layer):
        """
        # TODO: Write explanation
        :param delta:
        :param activation:
        :param pool_layer:
        :return:
        """
        conv_layer.nablab = np.sum(delta, axis=(0, 2, 3))
        num_filters, _, filter_width, filter_height = conv_layer.weights.shape
        padding, stride = conv_layer.padding, conv_layer.stride

        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        conv_layer.nablaw = np.dot(delta_reshaped, conv_layer.X_cols.T)
        conv_layer.nablaw = conv_layer.nablaw.reshape(conv_layer.weights.shape)

        delta_cols = np.dot(conv_layer.weights.reshape(num_filters, -1).T, delta_reshaped)
        delta = self.col2im(delta_cols, conv_layer.shape, filter_height, filter_width,
                            padding, stride)

        return delta


    def pool_backprop(self, delta, activation, pool_layer):
        """
        Pool backprop works by applying the error according to the type of pooling used, e.g. in a maxpool the
        error is applied to the max value
        :param delta:
        :param activation:
        :return:
        """
        return self.pool_backprop_reshape(delta, activation, pool_layer)

    def pool_backprop_reshape(self, delta, activation, pool_layer):
        """
        Backpropagation for a pool layer when the reshape method was used for feedforward
        :param delta:
        :param activation:
        :param pool_layer:
        :return:
        """
        # TODO: Generalise to any pool layer, currently this is for a max pool
        x_reshaped = pool_layer.reshaped
        delta_reshaped = np.zeros_like(x_reshaped)
        # act_newaxis = activation[:, :, np.newaxis, :, np.newaxis, :]
        act_newaxis = activation.reshape(x_reshaped.shape)
        mask = (x_reshaped == act_newaxis)
        delta_newaxis = delta[:, :, :, np.newaxis, :, np.newaxis]
        # delta_newaxis = delta.reshape(x_reshaped.shape)
        delta_broadcast, _ = np.broadcast_arrays(delta_newaxis, delta_reshaped)
        delta_reshaped[mask] = delta_broadcast[mask]
        delta_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = delta_reshaped.reshape(activation.shape)

        return dx

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, eta=0.01, lambda_=0.0,
                                    Xval=None, yval=None, momentum="nag", alpha=0.1, p=1):
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
        momentum = momentum.lower()
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
            # reshape X to align with target matrix (i.e. 2D)
            shape = X.shape
            if len(shape) > 2:
                dense_X = self.sparse_to_dense(X)
                Xy = np.hstack((dense_X,y))
            else:
            # TODO: Can this step be removed? Do I need to shuffle each epoch
                Xy = np.hstack((X,y))
            # shuffle t
            np.random.shuffle(Xy)

            # Split back into examples and class labels
            X_shuffled = Xy[:, :Xy.shape[1]-y.shape[1]]
            # reshape X_shuffled if necessary
            if len(shape) > 2:
                X_shuffled = self.dense_to_sparse(X_shuffled, shape)
            y_shuffled = Xy[:, -y.shape[1]:]

            # minibatch loop
            for k in range(0, m, mini_batch_size):
                # slice minibatches
                mini_X = X_shuffled[k:k+mini_batch_size,]
                mini_y = y_shuffled[k:k+mini_batch_size,]
                # perform update
                self.update_mini_batch(mini_X, mini_y, eta, lambda_, m, momentum=momentum, p=p)

            # If supplied with validation data, calculate validation error and perform early stopping
            if Xval is not None and yval is not None:
                val_h = self.predict(Xval)
                val_cost = self.cost_function(yval, val_h, Xval.shape[0], lambda_)
                self.val_error.append(val_cost)

                # early stopping regularisation, calculate loss rate
                loss_rate = float(val_cost / self.min_val_err - 1)

                # save best network weights and bias
                if val_cost < self.min_val_err:
                    self.min_val_err = val_cost
                    for l in self.layers:
                        if l.type != 'pool':
                            l.best_weights = l.weights
                            l.best_bias = l.bias

                # early stopping starts tracking after first 10 epochs to allow for initial fluctuations at
                # high learning rates
                if j > 10 and loss_rate > alpha:
                    print("Stopping early, epoch %d \t loss rate: %.3f \t Val Error: %.6f"
                          % (j, loss_rate, self.min_val_err))
                    for l in self.layers:
                        if l.type != 'pool':
                            l.weights = l.best_weights
                            l.bias = l.best_bias
                    # exit function
                    return None
            else:
                val_cost = 1.0
            # get the training error
            h = self.predict(X, p)
            cost = self.cost_function(y, h, m, lambda_)
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

        # update gradients with backprop
        self.backprop(X, y, p=p)

        for l in self.layers:
            # no updates requried for the pool layer
            if l.type != 'pool':
                # Momentum c.f. http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
                if momentum == "classic":
                    l.bias -= 1 / m * eta * l.nablab
                    l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw - lambda_ * l.weights)
                    l.weights += l.velocity

                # Nesterov accelerated gradient (NAG) c.f http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
                elif momentum == "nag":
                    l.bias -= 1 / m * eta * l.nablab
                    vel_prev = l.velocity
                    l.velocity = mu * l.velocity - eta * (1 / m * l.nablaw - lambda_ * l.weights)
                    l.weights += - mu * vel_prev + (1 + mu) * l.velocity

                # RMSProp c.f. Hinton http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
                # cache = decay * cache + (1 - decay) * grad ** 2
                # weights += - eta * grad / np.sqrt(cache + 1e-8)
                elif momentum == 'rmsprop':
                    l.cache = decay * l.cache + (1 - decay) * l.nablaw ** 2
                    l.weights -= (eta * l.nablaw) / np.sqrt(l.cache + 1e-8)

                # default update
                # update weights and bias with simple gradient descent
                else:
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
        if nn.layers[-1].activation == 'softmax':
            J = - 1 / m * np.sum(y * np.log(h + 1e-8))
        else:
            J = - 1 / m * (np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - (h + 1e-8))))

        # L2 regularisation
        for l in self.layers:
            if l.type != 'pool':
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


    def softmax(self, z):
        e = np.exp(z)
        return e / np.sum(e)


    def softmax_prime(self, z):
        y = self.softmax(z)
        return y * (1 - y)


    # def gradient_check(self, X, y, eps=1e-5):
    #     # TODO: Complete/fix gradient check code
    #     # run gradient function for a few epochs
    #     epochs = 30
    #     mini_batch_size = 50
    #     lambda_ = 0.0
    #     self.stochastic_gradient_descent(X, y, epochs, mini_batch_size, lambda_=lambda_)
    #     m = X.shape[0]
    #
    #     for i, l in enumerate(self.layers):
    #         # For gradient checking I need to take each element of the gradient
    #         # and compare it to the numerical gradient calculated by G(W(i) + eps)) - G(W(i) - eps)) / 2*eps
    #         # Looks like previously I was comparing tht weights rather than gradW, which is obvs wrong
    #         # Diff should be ~< eps
    #         weights = l.weights
    #         gradients = l.nablaw
    #         plus = weights + eps
    #         l.weights = plus
    #         actplus, _ = self.feedforward(X)
    #         costplus = self.cost_function(y, actplus[-1], m, lambda_)
    #         minus = weights - eps
    #         l.weights = minus
    #         actminus, _ = self.feedforward(X)
    #         costminus = self.cost_function(y, actminus[-1], m, lambda_)
    #         grad = (costplus - costminus) / (2 * eps)
    #         difference = (np.sum(gradients) - grad) / (np.sum(gradients) + grad)
    #         print("Difference for Layer %d: %.6f" %(i, difference))

    def gradient_check(self, X, forward, eps=1e-8):
        """
        Calculates the difference between the numerical gradient and the analytical gradient, run the
        network for 10-30 epochs before running this function to allow the weights to settle
        :param X: point or array to evaluate the gradient at
        :param forward: forward function
        :param back: backprop function
        :param eps: defualt 1e-8, small number to approximate lim->0
        :return: absolute difference between the analytical and numerical gradients
        """
        grad_an = nn.layers[0].nablaw
        # pick a random item from X
        idx = tuple([random.randrange(m) for m in grad_an.shape])
        # cache original value
        old = X[idx]

        # eval f(x + eps)
        X[idx] = old + eps
        plus, _ = forward(X)

        # eval f(x - eps)
        X[idx] = old - eps
        minus, _ = forward(X)

        grad_num = (plus[0] - minus[0]) / 2 * eps

        return np.sum(grad_num[idx] - grad_an[idx])

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

###################################################
# testing
# layers = [
#     {'type':'conv', 'depth':3, 'num_filters':32, 'filter_size':3},
#     {'type':'conv', 'depth':32, 'num_filters':32, 'filter_size':3},
#     {'type': 'pool', 'filter_width':2, 'filter_height':2, 'stride':2, 'activation':'max'},
#     {'type':'conv', 'depth':32, 'num_filters':64, 'filter_size':3},
#     {'type':'conv', 'depth':64, 'num_filters':64, 'filter_size':3},
#     {'type': 'pool', 'filter_width': 5, 'filter_height':5, 'stride':5, 'activation':'max'},
#     {'type': 'fc', 'inner':1600, 'outer':1600, 'activation':'relu'},
#     {'type': 'fc', 'inner':1600, 'outer':1, 'activation':'relu'}
# ]

# smaller net

# layers=[
#     {'type':'conv', 'depth':3, 'num_filters':2, 'filter_size':3},
#     {'type':'conv', 'depth':2, 'num_filters':2, 'filter_size':3},
#     {'type': 'pool', 'filter_width': 2, 'filter_height':2, 'stride':2, 'activation':'max'},
#     {'type': 'fc', 'inner':25*25*2, 'outer':50, 'activation':'sigmoid'},
#     {'type': 'fc', 'inner':50, 'outer':1, 'activation':'sigmoid'}
# ]
# nn = Network(layers)
# print(nn.layers)
# X = np.random.rand(10, 50, 50, 3)
# y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0]).reshape((10,1))
# a,z = nn.feedforward(X)
# print([x.shape for x in z])
# nn.backprop(X, y)
# print("Done")

# test pool backprop

# delta = a[-1].T - y
# dz = nn.activation_functions[nn.layers[-1].activation + '_prime'](z[-1])
# delta2 = (np.dot(delta.T, nn.layers[-1].weights) * dz).reshape((10, 25, 25, 2))
# delta3 = nn.pool_backprop_reshape(delta2, a[-3], nn.layers[-2])

# FC Network

layers=[
    {'type':'conv', 'depth':3, 'num_filters':10, 'filter_size':3, 'activation':'relu'},
    {'type':'conv', 'depth':10, 'num_filters':10, 'filter_size':3, 'activation':'relu'},
    {'type': 'pool', 'filter_width': 2, 'filter_height':2, 'stride':2, 'activation':'max'},
    {'type': 'fc', 'inner':6250, 'outer':6250, 'activation':'relu'},
    {'type': 'fc', 'inner':6250, 'outer':200, 'activation':'relu'},
    {'type': 'fc', 'inner':200, 'outer':1, 'activation':'softmax'}
]
nn = Network(layers)
print([l.type for l in nn.layers])
np.random.seed = 33
X = np.random.rand(10, 3, 50, 50)
X = (X - X.mean()) / X.std()
y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0]).reshape((10,1))
X_val = X[:2]
y_val = np.array([1,0]).reshape((2,1))
# a,z = nn.feedforward(X)
# print([x.shape for x in z])
# nn.backprop(X, y)
# print("Done")
nn.stochastic_gradient_descent(X, y, 5, 10, eta=1e-1, momentum='nag', Xval=X_val, yval=y_val, alpha=100)
