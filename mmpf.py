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
        Wgrad = T.grad(cost, self.W)
        bgrad = T.grad(cost, self.b)

        Wupdate = T.fill_diagonal(0.5 * ((self.W - learning_rate * Wgrad) + (self.W - learning_rate * Wgrad).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - learning_rate * bgrad )]

        return cost, updates


    def Kcost_momentum(self, learning_rate = 1e-2, epsilon = 1, gamma = 0.9):
        """
        Returns the cost of SGD with Momentum.
        """

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x, self.W) + self.b)))
        Wgrad = T.grad(cost, self.W)
        bgrad = T.grad(cost, self.b)

        vW = theano.shared(np.zeros(self.W.eval().shape))
        vb = theano.shared(np.zeros(self.b.eval().shape))
        vW_new = gamma * vW + learning_rate * Wgrad
        vb_new = gamma * vb + learning_rate * bgrad
        Wupdate = T.fill_diagonal(0.5 * ((self.W - vW_new) + (self.W - vW_new).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - vb_new), (vW, vW_new), (vb, vb_new)]

        return cost, updates


    def Kcost_nesterov(self, learning_rate = 1e-2, epsilon = 1, gamma = 0.9):
        """
        Returns the cost of SGD with Nesterov's accelerated gradient.
        """

        vW = theano.shared(np.zeros(self.W.eval().shape))
        vb = theano.shared(np.zeros(self.b.eval().shape))

        nextW = self.W - gamma * vW
        nextb = self.b - gamma * vb

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x, nextW) + nextb)))
        Wgrad = T.grad(cost, nextW)
        bgrad = T.grad(cost, nextb)

        vW_new = gamma * vW + learning_rate * Wgrad
        vb_new = gamma * vb + learning_rate * bgrad
        Wupdate = T.fill_diagonal(0.5 * ((self.W - vW_new) + (self.W - vW_new).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - vb_new), (vW, vW_new), (vb, vb_new)]

        return cost, updates


def sgd(units = 16, learning_rate = 1e-2, epsilon = 1, n_epochs = 1000, batch_size = 16,  sample = '16-50K.npy', flavour = 'vanilla'):
    """
    Perform stochastic gradient descent on MPF, plots parameters, computes Froenius norm and time taken.
    """
    print ('Loading '+ 'sample' + '...')

    dataset = load_data(sample)

    n_dataset_batches = dataset.get_value(borrow = True).shape[0] // batch_size

    print ('Building the model with flavour ' + flavour + '...')

    index = T.lscalar()
    x = T.matrix('x')

    flow = mpf(input = x, n = units)

    if flavour == 'vanilla':
        cost, updates = flow.Kcost()
    elif flavour == 'momentum':
        cost, updates = flow.Kcost_momentum()
    elif flavour == 'nesterov':
        cost, updates = flow.Kcost_nesterov()
    else:
        raise ValueError("Flavour must be 'vanilla', 'momentum' or 'nesterov'.")

    train_mpf = theano.function(inputs = [index], outputs = cost, updates = updates, \
                                givens = {x: dataset[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()

    W = np.load(sample[0:2] + '-' + 'W' + '.npy')
    b = np.load(sample[0:2] + '-' + 'b' + '.npy')

    for epoch in range(n_epochs):
        c = []
        current_time = timeit.default_timer()
        for batch_index in range(n_dataset_batches):
            c.append(train_mpf(batch_index))
        W_learnt = flow.W.get_value(borrow = True)
        b_learnt = flow.b.get_value(borrow = True)
        fnormW = np.linalg.norm(W - W_learnt)/np.linalg.norm(W + W_learnt)
        fnormb = np.linalg.norm(b - b_learnt)/np.linalg.norm(b + b_learnt)

        # print ('Training epoch %d/%d, Cost: %f, Time Elasped: %.2f' % (epoch, n_epochs, np.mean(c, dtype='float64'), (current_time - start_time)/60) )
        print ('Training epoch %d/%d, Cost: %f, F-normW: %.2f, F-normb: %.2f, Time Elasped: %.2f'\
         % (epoch, n_epochs, np.mean(c, dtype='float64'), \
        fnormW, fnormb,  (current_time - start_time)/60) )


    end_time = timeit.default_timer()

    training_time = end_time - start_time

    print ('The training took %.2f minutes' % (training_time/60.))

    W_learnt = flow.W.get_value(borrow = True)
    b_learnt = flow.b.get_value(borrow = True)


    fnormW = np.linalg.norm(W - W_learnt)/np.linalg.norm(W + W_learnt)
    fnormb = np.linalg.norm(b - b_learnt)/np.linalg.norm(b + b_learnt)

    print ('Comparing the parameters learnt...')
    fig, ax = plt.subplots(3)
    fig.tight_layout()
    ax[0].plot(W.reshape(-1,1)[0:100], 'b')
    ax[0].plot(W_learnt.reshape(-1,1)[0:100], 'r')
    # ax[0].set_title('W')
    ax[0].set_title('W')
    ax[0].legend(['W', 'Learnt W'])
    # ax[0].text(0.2, 0.1, 'F-norm(W): ' + str(fnormW), ha='center', va='center', transform = ax[0].transAxes, fontsize = 10)
    # ax[0].text(0.8, 0.1, 'Time taken: ' + str(training_time/60.), ha='center',  va='center', transform = ax[0].transAxes, fontsize = 10)
    ax[1].plot(b.reshape(-1,1), 'b')
    ax[1].plot(b_learnt.reshape(-1,1),'r')
    # ax[1].set_title('b')
    ax[1].set_title('b')
    ax[1].legend(['b', 'Learnt b'])
    # ax[1].text(0.2, 0.1, 'F-norm(b): ' + str(fnormb), ha='center', va='center', transform = ax[1].transAxes, fontsize = 10)
    ax[2].axis('off')
    ax[2].text(0.2, 0.8, 'F-norm(W): ' + str(fnormW), ha='center', va='center', transform = ax[2].transAxes, fontsize = 10)
    ax[2].text(0.2, 0.7, 'F-norm(b): ' + str(fnormb), ha='center', va='center', transform = ax[2].transAxes, fontsize = 10)
    ax[2].text(0.2, 0.6, 'Time taken: ' + str(training_time/60.), ha='center',  va='center', transform = ax[2].transAxes, fontsize = 10)


    E = n_epochs // 1000

    savefilename = sample[:-4] + '-' + '{0:.0e}'.format(learning_rate) + '-' + \
    '{0:.0e}'.format(epsilon) + '-' + str(E) + 'K-' + str(batch_size) + '-' + \
    flavour + '-'

    print ('Frobenius norm (W): %f' % fnormW)
    print ('Frobenius norm (b): %f' % fnormb)

    i = 0
    while os.path.exists('{}{:d}.png'.format(savefilename, i)):
        i += 1

    plt.savefig('{}{:d}.png'.format(savefilename, i))
    print ('Saving plots to ' + '{}{:d}.png'.format(savefilename, i))
    print ('Naming convention: units-sample_size-learning_rate-epsilon-epochs-batchsize-runs')

    return W_learnt, b_learnt


if __name__ == "__main__":
    sgd(units = 32, learning_rate = 1e-3, epsilon = 1e-2, n_epochs = 5000, batch_size = 16,\
      sample = '32-50K.npy', flavour = 'nesterov')
