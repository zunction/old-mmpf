import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import os

from pathlib import Path

def getseed(n = 16, randstate = 123):
    """
    Generates the seed for the Gibbs sampler.
    Inputs:
    - n: (int) number of neurons in the network
    """
    prng = np.random.RandomState(randstate)
    return np.random.randint(2, size = n)

def getW(n = 16, randstate = 123):
    """
    Generates the matrix to learn using MPF.
    Inputs:
    - n: (int) number of neurons in the network
    - index: if None, will be labeled by YearMonthDay-HourMinute
    """
    prng = np.random.RandomState(randstate)
    U = prng.normal(0, 1, (n, n))
    W = 0.5 * (U + U.T)
    np.fill_diagonal(W, 0)

    filename = str(n) + '-' + 'W'
    myfile = Path(filename + '.npy')

    if myfile.is_file():
        print (filename + '.npy' + ' exists')
    else:
        np.save(filename, W)
        print ('W matrix saved as ' + filename + '.npy')
    np.save(filename, W)
    return W


def getb(n = 16, randstate = 123):
    """
    Generates the bias to learn using MPF.
    Inputs:
    - n: (int) number of neurons in the network
    - index: if None, will be labeled by YearMonthDay-HourMinute
    """
    prng = np.random.RandomState(randstate)
    b = prng.normal(0, 1, (n, ))


    filename = str(n) + '-' + 'b'
    myfile = Path(filename + '.npy')

    if myfile.is_file():
        print (filename + '.npy' + ' exists.')
    else:
        np.save(filename, b)
        print ('b bias saved as ' + filename + '.npy')
    np.save(filename, b)

    return b
def sigmoid(x):
    """
    Takes in a vector x and returns its sigmoid activation.
    Input:
    - x: a numpy array
    """
    return 1/(1 + np.exp(-x))


def one_state_update(x, W, b, s):
    """
    Does a single update of the sth neuron of the network.
    Inputs:
    - x: current state of the network to produce a new state
    - W: numpy array of weights
    - b: numpy array of biases
    """
    p = sigmoid(np.dot(W[s, :], x) + b)
    new_x = np.zeros(x.shape) + x
    new_x[s] = np.random.binomial(1, p[s], 1)
    return new_x


def burn_in(x, W, b, n = 10000):
    """
    Performs the burning in before doing the Gibbs sampling.
    """
    v = x.shape[0]
    for i in range(n * v):
        s = np.random.randint(0, v)
        x = one_state_update(x, W, b, s)
    return x


def n_updates(x, W, b, n = 100):
    """
    Performs n times of the one_state_update.
    Inputs:
    - x: current state of the network to produce a new state
    - W: numpy array of weights
    - b: numpy array of biases
    - n: (int) number of updates to be made
    """
    v = x.shape[0]
    for i in range(n):
        s = np.random.randint(0, v)
        x = one_state_update(x, W, b, s)
    return x


def mixing(x, W, b, n = 50000, m = 100):
    """
    Does mixing for m times before obtaining a single sample.
    Inputs:
    - x: current state of the network to produce a new state
    - W: numpy array of weights
    - b: numpy array of biases
    - n: (int) number of samples to be generated
    - m: (int) number of updates before a sample is saved
    """
    samples = np.zeros((n, x.shape[0]))
    one = n // 100
    p = 1
    last_x = np.zeros(x.shape)
    for i in range(n):
        # if i % one == 0:
        #     print ('%d %%' % p)
        #     p += 1
        x = n_updates(x, W, b, m)
        last_x = x
        samples[i, :] = x
    return last_x, samples


def sampling(units = 16, n = 50000, m = 100, randstate = 123):
    """
    Generate n samples from seed x.
    Input:
    - units: (int) number of units in the Boltzmann machine
    - x: current state of the network to produce a new state
    - W: numpy array of weights
    - b: numpy array of biases
    - n: (int) number of samples to be generated
    - m: (int) number of updates before a sample is saved
    - savesamples: (bool)
    """
    sess = n // 50000 + 1
    sess = int(sess)
    samples = np.zeros((sess * 50000, units))
    K = sess * 50
    filename = str(units) + '-' + str(K) + 'K'
    myfile = Path(filename + '.npy')

    if myfile.is_file():
        print (filename + '.npy' + ' exists.')
    else:
        tic = timeit.default_timer()
        print ('#' * 19 + ' Sampling ' + '#' * 19)
        print ('Number of units: %d' % units)
        print ('Samples requested: %d' % n)
        print ('Samples to be generated: %dK' % K)
        print ('=' * 48)
        # print ('Rounding to nearest 50K...')
        # print ('Generating %dK samples with %d units...' % (K, units))
        print ('#' * 20 +' Status ' + '#' * 20)
        x = getseed(n = units, randstate = randstate)
        W = getW(n = units, randstate = randstate)
        b = getb(n = units, randstate = randstate)
#         use the below for setting bias to be zeros
#         b = np.zeros(units,)
#         np.save('32-b_zeros', b)

        print ('#' * 18  + ' Burning in ' + '#' * 18)
        burnt = burn_in(x, W, b)
        print ('Burnt:', burnt)

        print ('#' * 20 + ' Mixing ' + '#' * 20)
        if sess > 1:
            last_x = burnt
            for i in range(sess):
                print ('Mixing %d/%d parts' % (i + 1, sess))
                last_x, subsamples = mixing(last_x, W, b, 50000, m)
                samples[i * 50000: ((i + 1) * 50000)] = subsamples
        else:
            last_x, samples = mixing(burnt, W, b, n, m)

        np.save(filename, samples)
        print ('samples saved as ' + filename + '.npy')

        toc = timeit.default_timer()
        print ('Time taken to create %d samples is %.2f minutes' % (n, (toc - tic)/60.))
        return samples


if __name__ == '__main__':
    sampling(units = 32, n = 2e7, m = 100, randstate = 123)
