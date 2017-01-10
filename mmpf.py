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
    Minimum probability flow
    """

    def __init__(self, input = None, n = 16, W = None, b = None):
        """
        Initialize mpf class
        """
        self.n = n

        U = np.random.rand(self.n, self.n)
        R = 0.5 * (U + U.T)
        np.fill_diagonal(R, 0)

        if not W:
            initial_W = np.asarray(R, dtype = theano.config.floatX)
            W = theano.shared(initial_W, name = 'W', borrow = True)

        if not b:
            initial_b = np.asarray(np.random.rand(n), dtype = theano.config.floatX)
            b = theano.shared(initial_b, name = 'b', borrow = True)


        self.W = W
        self.b = b
#         self.input = input
        if input is None:
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b]


    def Kcost(self, learning_rate = 10e-2):
        """
        Returns the cost
        """

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x, self.W) + self.b)))
#         gparams = T.grad(cost, self.params)
#         updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        Wgrad = T.grad(cost, self.W)
#         T.fill_diagonal(Wgrad, 0)
        bgrad = T.grad(cost, self.b)

        Wupdate = T.fill_diagonal(0.5 * ((self.W - learning_rate * Wgrad) + (self.W - learning_rate * Wgrad).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - learning_rate * bgrad )]
#         updates = [(self.W, self.W - learning_rate * Wgrad), (self.b, self.b - learning_rate * bgrad )]

        return cost, updates


def sgd(units = 16, learning_rate = 10e-2, n_epochs = 1000, batch_size = 16,  sample = '16-50K.npy'):
    """
    Perform stochastic gradient descent on MPF
    """
    print ('Loading '+ 'sample' + '...')

    dataset = load_data(sample)

    n_dataset_batches = dataset.get_value(borrow = True).shape[0] // batch_size

    print ('Building the model...')

    index = T.lscalar()
    x = T.matrix('x')

#     if not os.path.isdir(output_folder):
#         os.makedirs(output_folder)
#     os.chdir(output_folder)

    flow = mpf(input = x, n = units)
    cost, updates = flow.Kcost()

    train_mpf = theano.function(inputs = [index], outputs = cost, updates = updates, \
                                givens = {x: dataset[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()

    for epoch in range(n_epochs):
        c = []
        current_time = timeit.default_timer()
        for batch_index in range(n_dataset_batches):
            c.append(train_mpf(batch_index))

        print ('Training epoch %d, Cost: %f, Time Elasped: %.2f' % (epoch, np.mean(c, dtype='float64'), (current_time - start_time)/60) )

    end_time = timeit.default_timer()

    training_time = end_time - start_time

    print ('The training took %.2f minutes' % (training_time/60.))

    W_learnt = flow.W.get_value(borrow = True)
    b_learnt = flow.b.get_value(borrow = True)
    W = np.load(sample[0:2] + '-' + 'W' + '.npy')
    b = np.load(sample[0:2] + '-' + 'b' + '.npy')

    fnormW = np.linalg.norm(W - W_learnt)
    fnormb = np.linalg.norm(b - b_learnt)

    print ('Comparing the parameters learnt...')
    fig, ax = plt.subplots(2)
    ax[0].plot(W.reshape(-1,1)[0:100], 'b')
    ax[0].plot(W_learnt.reshape(-1,1)[0:100], 'r')
    ax[0].set_title('Weight matrix, W')
    ax[0].legend(['W', 'Learnt W'])
    ax[0].text(0.2, 0.1, 'F-norm(W): ' + str(fnormW), ha='center', va='center', transform = ax[0].transAxes, fontsize = 10)
    ax[1].plot(b.reshape(-1,1), 'b')
    ax[1].plot(b_learnt.reshape(-1,1),'r')
    ax[1].set_title('Bias, b')
    ax[1].legend(['b', 'Learnt b'])
    ax[1].text(0.2, 0.1, 'F-norm(b): ' + str(fnormb), ha='center', va='center', transform = ax[1].transAxes, fontsize = 10)

    E = n_epochs // 1000

    # savefilename = sample[:-4] + '-' + str(learning_rate)+ '-' + str(E) + 'K-' + str(batch_size) + '-'
    savefilename = sample[:-4] + '-' + str(E) + 'K-' + str(batch_size) + '-'


    print ('Frobenius norm (W): %f' % fnormW)
    print ('Frobenius norm (b): %f' % fnormb)

    i = 0
    while os.path.exists('{}{:d}.png'.format(savefilename, i)):
        i += 1

    plt.savefig('{}{:d}.png'.format(savefilename, i))
    print ('Saving plots to ' + '{}{:d}.png'.format(savefilename, i))

    return W_learnt, b_learnt


if __name__ == "__main__":
    sgd(units = 16, learning_rate = 1e-2, n_epochs = 1000, batch_size = 16,\
      sample = '16-50K.npy')
