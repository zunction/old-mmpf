import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import os
import timeit


def load_data(dataset = '16-50K.npy', borrow = True):
    """
    Loads the dataset.
    """

    data = np.load(dataset)
    dataset = theano.shared(np.asarray(data, dtype = theano.config.floatX), borrow = borrow)

    return dataset


class mpf(object):
    """
    Minimum Probability Flow
    """
    def __init__(self, input = None, n = 16, W = None, b = None, gpu = False):
        """
        Initialize MPF class
        """

        self.n = n

        # U = np.random.rand(self.n, self.n)
        # R = 0.5 * (U + U.T)
        # np.fill_diagonal(R, 0)

        if not W:
            initial_W = np.asarray(R, dtype = theano.config.floatX)
            W = theano.shared(initial_W, name = 'W', borrow = True)

        if not b:
            initial_b = np.asarray(np.random.rand(n), dtype = theano.config.floatX)
            b = theano.shared(initial_b, name = 'b', borrow = True)


        self.W = W
        self.b = b
        self.gpu = gpu

        if input is None:
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b]

    def Kcost(self, learning_rate = 1e-2, epsilon = 1):
        """
        Returns the cost of vanilla SGD.
        """

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x, self.W) + self.b))) * epsilon
        gparams = T.grad(cost, self.params)

        updates = []

        # Wgrad = T.grad(cost, self.W)
        # bgrad = T.grad(cost, self.b)

        Wupdate = T.fill_diagonal(0.5 * ((self.W - learning_rate * Wgrad) + (self.W - learning_rate * Wgrad).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - learning_rate * bgrad )]

        return cost, updates
