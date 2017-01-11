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
    for i in range(n):
        # if i % one == 0:
        #     print ('%d %%' % p)
        #     p += 1
        x = n_updates(x, W, b, m)
        samples[i, :] = x
#     timestr = time.strftime('%Y%m%d-%H%M%S')
#     filename = 'sample'+timestr+'.dat'

#     if savesamples == "True":
#         np.save(filename, samples)
#         print ('Samples are saved as ' + filename)
#     elif savesamples == "False":
#         print ('Samples were not saved.')
#     else:
#         raise ValueError("savesamples must be 'True' or 'False'")
    return samples


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
    K = n // 1000
    filename = str(units) + '-' + str(K) + 'K'
    myfile = Path(filename + '.npy')

    if myfile.is_file():
        print (filename + '.npy' + ' exists.')
    else:
        tic = timeit.default_timer()

        print ('Generating %dK samples with %d units...' % (K, units))

        x = getseed(n = units, randstate = randstate)
        W = getW(n = units, randstate = randstate)
        b = getb(n = units, randstate = randstate)


        print ('Burning in...')
        burnt = burn_in(x, W, b)
        print ('Burnt:', burnt)
        print ('Mixing...')
        samples = mixing(burnt, W, b, n, m)

        np.save(filename, samples)
        print ('samples saved as ' + filename + '.npy')

        toc = timeit.default_timer()
        print ('Time taken to create %d samples is %.2f minutes' % (n, (toc - tic)/60.))
        return samples


if __name__ == "__main__":
    sampling(units = 32, n = 100000)
