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

    def __init__(self, input = None, n = 16, W = None, b = None, gpu = False):
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
        self.gpu = gpu

        if input is None:
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b]

    def loss_forcedsymmetry(self, learning_rate = 1e-2, epsilon = 1):
        """
        Returns the cost of vanilla SGD.
        """

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x, self.W) + self.b))) * epsilon
        Wgrad = T.grad(cost, self.W)
        bgrad = T.grad(cost, self.b)

        Wupdate = T.fill_diagonal(0.5 * ((self.W - learning_rate * Wgrad) + (self.W - learning_rate * Wgrad).T), 0)
        updates = [(self.W, Wupdate), (self.b, self.b - learning_rate * bgrad )]

        return cost, updates

<<<<<<< HEAD
    def loss(self, learning_rate = 1e-2, epsilon = 1):
=======
    def Kcost(self, learning_rate = 1e-2, epsilon = 1, temperature = 100):
>>>>>>> wild
        """
        Returns the cost of vanilla SGD.
        """

        print ('Using Vanilla with  learning rate = %f, epsilon = %f, temperature = %d'\
         % (learning_rate, epsilon, temperature))

        cost = T.mean(T.exp((1 / temperature * (0.5 - self.x) * \
        (T.dot(self.x, T.fill_diagonal(self.W, 0)) + self.b)))) * epsilon
        gparams = T.grad(cost, self.params)

        updates = [(param, param - learning_rate * gparam) \
        for param, gparam in zip(self.params, gparams)]

        return cost, updates


    def Kcost_momentum(self, learning_rate = 1e-2, epsilon = 1, gamma = 0.9):
        """
        Returns the cost of SGD with Momentum.
        """
        print ('Using Momentum with gamma = %f, learning rate = %f, epsilon = %f'\
         % (gamma, learning_rate, epsilon))

        cost = T.mean(T.exp((0.5 - self.x) * \
        (T.dot(self.x, T.fill_diagonal(self.W, 0)) + self.b))) * epsilon

        gparams = T.grad(cost, self.params)

        if self.gpu:
            vW = theano.shared(np.zeros(self.W.eval().shape).astype(np.float32))
            vb = theano.shared(np.zeros(self.b.eval().shape).astype(np.float32))
        else:
            vW = theano.shared(np.zeros(self.W.eval().shape))
            vb = theano.shared(np.zeros(self.b.eval().shape))

        momentum = [vW, vb]
        momentum_updates = [(v, gamma * v + learning_rate * gparam) \
        for v, gparam in zip(momentum, gparams)]

        updates = [(param, param - v) \
        for param, v in zip(self.params, momentum)]

        updates = updates + momentum_updates

        return cost, updates


    def Kcost_nesterov(self, learning_rate = 1e-2, epsilon = 1, gamma = 0.9):
        """
        Returns the cost of SGD with Nesterov's accelerated gradient.
        """
        print ('Using Nesterov with gamma = %f, learning rate = %f, epsilon = %f'\
         % (gamma, learning_rate, epsilon))
        if self.gpu:
            vW = theano.shared(np.zeros(self.W.eval().shape).astype(np.float32))
            vb = theano.shared(np.zeros(self.b.eval().shape).astype(np.float32))
        else:
            vW = theano.shared(np.zeros(self.W.eval().shape))
            vb = theano.shared(np.zeros(self.b.eval().shape))


        nextW = self.W - gamma * vW
        nextb = self.b - gamma * vb

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x,\
                T.fill_diagonal(nextW, 0)) + nextb))) * epsilon

        Wgrad = T.grad(cost, nextW)
        bgrad = T.grad(cost, nextb)

        gparams = [Wgrad, bgrad]
        momentum = [vW, vb]
        momentum_updates = [(v, gamma * v + learning_rate * gparam)\
        for v, gparam in zip(momentum, gparams)]

        updates = [(param, param - v) \
        for param, v in zip(self.params, momentum)]

        updates = updates + momentum_updates

        return cost, updates

    def Kcost_adagrad(self, learning_rate = 1e-2, epsilon = 1, smoothingterm = 1):
        """
        Returns the cost of SGD using adagrad.
        """
        print ('Using Adagrad with smoothing term = %.9f, learning rate = %f, epsilon = %f'\
         % (smoothingterm, learning_rate, epsilon))

        param_shapes = [param.get_value().shape for param in self.params ]
        grad_hists = [theano.shared(np.zeros(param_shape,
                        dtype = theano.config.floatX),
                        borrow = True,
                        name = 'grad_hist_' + param.name)
                        for param_shape, param in zip(param_shapes, self.params)]

        cost = T.mean(T.exp((0.5 - self.x) * (T.dot(self.x,\
                T.fill_diagonal(self.W, 0)) + self.b))) * epsilon


        gparams = T.grad(cost, self.params)

        grad_hist_updates = [(g_hist, g_hist + g ** 2) for g_hist, g in zip(grad_hists, gparams)]

        updates = [(param, param - learning_rate * gparam/(T.sqrt(grad_hist + smoothingterm)))\
        for param, grad_hist, gparam in zip(self.params, grad_hists, gparams)]

        updates = updates + grad_hist_updates

        return cost, updates

def sgd(units = 16, learning_rate = 1e-2, epsilon = 1, n_epochs = 1000,\
    batch_size = 16, temperature = 100, sample = '16-50K.npy', gpu = False, flavour = 'vanilla'):
    """
    Perform stochastic gradient descent on MPF, plots parameters, computes Froenius norm and time taken.
    """
    print ('Loading '+ 'sample' + '...')

    dataset = load_data(sample)

    n_dataset_batches = dataset.get_value(borrow = True).shape[0] // batch_size

    print ('Building the model with flavour ' + flavour + '...')

    index = T.lscalar()
    x = T.matrix('x')

    flow = mpf(input = x, n = units, gpu = gpu)

    if flavour == 'vanilla':
        cost, updates = flow.Kcost(learning_rate = learning_rate, \
        epsilon = epsilon, temperature = temperature)
    elif flavour == 'momentum':
        cost, updates = flow.Kcost_momentum()
    elif flavour == 'nesterov':
        cost, updates = flow.Kcost_nesterov()
    elif flavour == 'adagrad':
        cost, updates = flow.Kcost_adagrad()
    else:
        raise ValueError("Flavour must be 'vanilla', 'momentum', 'nesterov' or 'adagrad'.")

    train_mpf = theano.function(inputs = [index], outputs = cost, updates = updates, \
                                givens = {x: dataset[index * batch_size: (index + 1) * batch_size]})

    start_time = timeit.default_timer()
    best_W = [None, np.inf]
    best_b = [None, np.inf]
    best_cost = None
    best_epoch = None
    best_mse = np.inf

    W = np.load(sample[0:2] + '-' + 'W' + '.npy')
    b = np.load(sample[0:2] + '-' + 'b' + '.npy')

    cost_history = []
    mse_history = []
    mseW_history = []
    mseb_history = []

    for epoch in range(n_epochs):
        c = []
        current_time = timeit.default_timer()
        for batch_index in range(n_dataset_batches):
            c.append(train_mpf(batch_index))

        W_learnt = flow.W.get_value(borrow = True)
        b_learnt = flow.b.get_value(borrow = True)

        mseW = np.linalg.norm(W - W_learnt)/ (units**2 - units)
        mseb = np.linalg.norm(b - b_learnt)/ units
        mse = (mseW * mseb)/(mseW + mseb)

        cost_history.append(np.mean(c, dtype='float64'))
        mse_history.append(mse)
        mseW_history.append(mseW)
        mseb_history.append(mseb)

        if mse < best_mse:
            best_mse = mse
            best_W[0] = flow.W.get_value(borrow = True)
            best_W[1] = mseW
            best_b[0] = flow.b.get_value(borrow = True)
            best_b[1] = mseb
            best_cost = np.mean(c, dtype='float64')
            best_epoch = epoch


        print ('Training epoch %d/%d, Cost: %f mseW: %.5f, mseb: %.5f, mse: %.5f Time Elasped: %.2f '\
         % (epoch, n_epochs, np.mean(c, dtype='float64'), \
         mseW, mseb, mse, (current_time - start_time)/60) )


    end_time = timeit.default_timer()

    training_time = end_time - start_time

    print ('The training took %.2f minutes' % (training_time/60.))


    W_learnt = best_W[0]
    mseW = best_W[1]
    b_learnt = best_b[0]
    mseb = best_b[1]

    print ('Comparing the parameters learnt... ')

    fig, ax = plt.subplots(2, 2, figsize=(20,10))
    fig.tight_layout()

    ax[0,0].plot(W.reshape(-1,1)[0:100], 'b')
    ax[0,0].plot(W_learnt.reshape(-1,1)[0:100], 'r')
    ax[0,0].set_title('W')
    ax[0,0].legend(['W', 'Learnt W'])

    ax[0,1].plot(b.reshape(-1,1), 'b')
    ax[0,1].plot(b_learnt.reshape(-1,1),'r')
    ax[0,1].set_title('b')
    ax[0,1].legend(['b', 'Learnt b'])

    ax[1,0].plot(cost_history, 'r.', mseW_history, 'b.', mseb_history, 'g.', mse_history, 'c.')
    ax[1,0].legend(['cost', 'mseW', 'mseb', 'mse'])
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Value')
    ax[1,0].set_ylim([-0.05,1.05])


    ax[1,1].axis('off')
    ax[1,1].text(0.5, 0.7, 'MSE(W): ' + str(mseW), ha='center', va='center', transform = ax[1,1].transAxes, fontsize = 15)
    ax[1,1].text(0.5, 0.6, 'MSE(b): ' + str(mseb), ha='center', va='center', transform = ax[1,1].transAxes, fontsize = 15)
    ax[1,1].text(0.5, 0.8, 'Best epoch: ' + str(best_epoch), ha='center', va='center', transform = ax[1,1].transAxes, fontsize = 15)
    ax[1,1].text(0.5, 0.5, 'Cost for best epoch: ' + str(best_cost), ha='center', va='center', transform = ax[1,1].transAxes, fontsize = 15)
    ax[1,1].text(0.5, 0.4, 'Time taken: ' + str(training_time/60.), ha='center',  va='center', transform = ax[1,1].transAxes, fontsize = 15)



    E = n_epochs // 1000

    if flow.gpu:
        savefilename = sample[:-4] + '-' + '{0:.0e}'.format(learning_rate) + '-' + \
        '{0:.0e}'.format(epsilon) + '-' + str(E) + 'K-' + str(batch_size) + '-' + \
        flavour + '-' + 'gpu' + '-'
    else:
        savefilename = sample[:-4] + '-' + '{0:.0e}'.format(learning_rate) + '-' + \
        '{0:.0e}'.format(epsilon) + '-' + str(E) + 'K-' + str(batch_size) + '-' + \
        flavour + '-' + 'cpu' + '-'


    print ('MSE (W): %f' % mseW)
    print ('MSE (b): %f' % mseb)

    i = 0
    while os.path.exists('{}{:d}.png'.format(savefilename, i)):
        i += 1

    plt.savefig('{}{:d}.png'.format(savefilename, i))
    print ('Saving plots to ' + '{}{:d}.png '.format(savefilename, i))
    print ('Naming convention: units-sample_size-learning_rate-epsilon-epochs-batchsize-processor-runs ')

    return W_learnt, b_learnt


if __name__ == "__main__":
    units = 32
    lr = 1e-5
    epsilon = 1
    epochs = 1000
    batch_size = 32
    samples = '32-50K.npy'
    gpu = False
    temperature = 1
    sgd(units = units, learning_rate = lr, epsilon = epsilon, n_epochs = epochs, batch_size = batch_size,\
      sample = samples, gpu = gpu, flavour = 'vanilla', temperature = temperature)
    # sgd(units = units, learning_rate = learning_rate, epsilon = epsilon, n_epochs = epochs, batch_size = batch_size,\
    #   sample = samples, gpu = gpu, flavour = 'momentum')
    # sgd(units = units, learning_rate = learning_rate, epsilon = epsilon, n_epochs = epochs, batch_size = batch_size,\
    #   sample = samples, gpu = gpu, flavour = 'nesterov')
    # sgd(units = units, learning_rate = learning_rate, epsilon = epsilon, n_epochs = epochs, batch_size = batch_size,\
    #   sample = samples, gpu = gpu, flavour = 'adagrad')
