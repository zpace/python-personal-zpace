import numpy as np
import matplotlib.pyplot as plt
import pymc
from triangle import corner


class lin_fit():

    '''
    fit a straight line with no outliers to one independent variable
        (`xi`, with zero errors) and one dependent variable
        (`yi`, with possibly heteroscedastic errors `dyi`)

    Modified from Vanderplas's code
    (found at http://www.astroml.\
        org/book_figures/chapter8/fig_outlier_rejection.html)
    '''

    def __init__(self, xi, yi, dyi, value):

        self.xi, self.yi, self.dyi, self.value = xi, yi, dyi, value

        # priors on parameters
        @pymc.stochastic
        def beta_M0(value=value):
            '''Slope and intercept parameters for a straight line.
            The likelihood corresponds to the prior probability of the
            parameters.'''
            slope, intercept = value
            prob_intercept = 1 + 0 * intercept
            # uniform prior on theta = arctan(slope)
            # d[arctan(x)]/dx = 1 / (1 + x^2)
            prob_slope = np.log(1. / (1. + slope ** 2))
            return prob_intercept + prob_slope

        # linear function
        @pymc.deterministic
        def model_M0(xi=xi, beta=beta_M0):
            slope, intercept = beta
            return slope * xi + intercept

        # probability model
        y = pymc.Normal('y', mu=model_M0, tau=dyi ** -2,
                        observed=True, value=yi)

        self.M0 = dict(beta_M0=beta_M0, model_M0=model_M0, y=y)

        self.sample_invoked = False

    def sample(self, iter, burn):
        self.S0 = pymc.MCMC(self.M0)
        self.S0.sample(iter=iter, burn=burn)
        self.trace = self.S0.trace('beta_M0')
        self.mtrace, self.btrace = self.trace[:, 0], self.trace[:, 1]

        self.sample_invoked = True

    def triangle(self):
        assert self.sample_invoked == True, \
            'Must sample first! Use sample(iter, burn)'

        corner(self.trace[:], labels=['$m$', '$b$'])

    def plot(self, xlab='$x$', ylab='$y$'):
        # plot the data points
        plt.errorbar(self.xi, self.yi, yerr=self.dyi, fmt='.k')

        # do some shimmying to get quantile bounds
        xa = np.linspace(self.xi.min(), self.xi.max(), 100)
        A = np.vander(xa, 2)
        # generate all possible lines
        lines = np.dot(self.trace[:], A.T)
        quantiles = np.percentile(lines, [16, 84], axis=0)
        plt.fill_between(xa, quantiles[0], quantiles[1],
                         color="#8d44ad", alpha=0.5)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
