import numpy as np
import matplotlib.pyplot as plt
import pymc
from triangle import corner
from astroML.plotting import hist


def models_compare(models):
    '''
    compute a bunch of model statistics

    Arguments:
     - models: a list of models (NOT pure pymc), as defined below
    '''

    # start off with information criteria
    ICs = np.array([list(model.ICs()) for model in models])
    AICs = ICs[:, 0]
    BICs = ICs[:, 1]

    print 'M{}\'s AIC is superior'.format(np.argmin(AICs) + 1)
    print 'M{}\'s BIC is superior'.format(np.argmin(BICs) + 1)

    p_rel_AIC = np.exp((AICs.min() - AICs)/2.)
    print 'Relative probability of information loss minimization:\n', p_rel_AIC

    DBIC = BICs - BICs.min()
    print 'Evidence against a higher BIC:\n', DBIC


class lin_fit(object):

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

        self.M = dict(beta_M0=beta_M0, model_M0=model_M0, y=y)

        self.sample_invoked = False

    def sample(self, iter, burn, calc_deviance=True):
        self.S0 = pymc.MCMC(self.M)
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

    def ICs(self):
        self.MAP = pymc.MAP(self.M)
        self.MAP.fit()

        self.BIC = self.MAP.BIC
        self.AIC = self.MAP.AIC
        self.logp = self.MAP.logp
        self.logp_at_max = self.MAP.logp_at_max
        return self.AIC, self.BIC


class const_fit(object):

    '''
    fit a straight line with no outliers to one independent variable
        (`xi`, with zero errors) and one dependent variable
        (`yi`, with possibly heteroscedastic errors `dyi`)

    Intended to be a complement to a straight-line fit, for model
        testing purposes

    Modified from Vanderplas's code
        (found at http://www.astroml.\
        org/book_figures/chapter8/fig_outlier_rejection.html)
    '''

    def __init__(self, xi, yi, dyi, value):

        self.xi, self.yi, self.dyi, self.value = xi, yi, dyi, value

        # priors on parameters
        @pymc.stochastic
        def beta_Mc(value=value):
            '''intercept parameter for a straight line.
            The likelihood corresponds to the prior probability of the
            parameter.'''
            intercept = value
            prob_intercept = 1 + 0 * intercept
            return prob_intercept

        # linear function
        @pymc.deterministic
        def model_Mc(xi=xi, beta=beta_Mc):
            intercept = beta
            return 0. * xi + intercept

        # probability model
        y = pymc.Normal('y', mu=model_Mc, tau=dyi ** -2,
                        observed=True, value=yi)

        self.M = dict(beta_Mc=beta_Mc, model_Mc=model_Mc, y=y)

        self.sample_invoked = False

    def sample(self, iter, burn, calc_deviance=True):
        self.S0 = pymc.MCMC(self.M)
        self.S0.sample(iter=iter, burn=burn)
        self.trace = self.S0.trace('beta_Mc')
        self.btrace = self.trace[:, 0]

        self.sample_invoked = True

    def triangle(self):
        assert self.sample_invoked == True, \
            'Must sample first! Use sample(iter, burn)'

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        hist(self.trace[:].flatten(), bins='knuth', normed=True,
             histtype='step', color='k', ax=ax)
        plt.xlabel('$b$')

    def plot(self, xlab='$x$', ylab='$y$'):
        # plot the data points
        plt.errorbar(self.xi, self.yi, yerr=self.dyi, fmt='.k')

        # do some shimmying to get quantile bounds
        xa = np.linspace(self.xi.min(), self.xi.max(), 100)
        A = np.vander(xa, 2)
        # generate all possible lines
        lines = np.dot(np.hstack((np.zeros_like(self.trace[:]),
                                  self.trace[:])), A.T)
        quantiles = np.percentile(lines, [16, 84], axis=0)
        plt.fill_between(xa, quantiles[0], quantiles[1],
                         color="#8d44ad", alpha=0.5)

        plt.xlabel(xlab)
        plt.ylabel(ylab)

    def ICs(self):
        self.MAP = pymc.MAP(self.M)
        self.MAP.fit()

        self.BIC = self.MAP.BIC
        self.AIC = self.MAP.AIC
        self.logp = self.MAP.logp
        self.logp_at_max = self.MAP.logp_at_max
        return self.AIC, self.BIC


class lin_fit_ol(object):

    '''
    fit a straight line to one independent variable
        (`xi`, with zero errors) and one dependent variable
        (`yi`, with possibly heteroscedastic errors `dyi`)
    Outliers in `yi` are permitted

    Intended to be a complement to a straight-line fit, for model
        testing purposes

    Modified from Vanderplas's code
        (found at http://www.astroml.\
        org/book_figures/chapter8/fig_outlier_rejection.html)
    '''

    def __init__(self, xi, yi, dyi, value):

        self.xi, self.yi, self.dyi, self.value = xi, yi, dyi, value

        @pymc.stochastic
        def beta(value=np.array([0.5, 1.0])):
            """Slope and intercept parameters for a straight line.
            The likelihood corresponds to the prior probability of the parameters."""
            slope, intercept = value
            prob_intercept = 1 + 0 * intercept
            # uniform prior on theta = arctan(slope)
            # d[arctan(x)]/dx = 1 / (1 + x^2)
            prob_slope = np.log(1. / (1. + slope ** 2))
            return prob_intercept + prob_slope

        @pymc.deterministic
        def model(xi=xi, beta=beta):
            slope, intercept = beta
            return slope * xi + intercept

        # uniform prior on Pb, the fraction of bad points
        Pb = pymc.Uniform('Pb', 0, 1.0, value=0.1)

        # uniform prior on Yb, the centroid of the outlier distribution
        Yb = pymc.Uniform('Yb', -10000, 10000, value=0)

        # uniform prior on log(sigmab), the spread of the outlier distribution
        log_sigmab = pymc.Uniform('log_sigmab', -10, 10, value=5)

        # qi is bernoulli distributed
        # Note: this syntax requires pymc version 2.2
        qi = pymc.Bernoulli('qi', p=1 - Pb, value=np.ones(len(xi)))

        @pymc.deterministic
        def sigmab(log_sigmab=log_sigmab):
            return np.exp(log_sigmab)

        def outlier_likelihood(yi, mu, dyi, qi, Yb, sigmab):
            """likelihood for full outlier posterior"""
            Vi = dyi ** 2
            Vb = sigmab ** 2

            root2pi = np.sqrt(2 * np.pi)

            logL_in = -0.5 * np.sum(
                qi * (np.log(2 * np.pi * Vi) + (yi - mu) ** 2 / Vi))

            logL_out = -0.5 * np.sum(
                (1 - qi) * (np.log(2 * np.pi * (Vi + Vb)) +
                            (yi - Yb) ** 2 / (Vi + Vb)))

            return logL_out + logL_in

        OutlierNormal = pymc.stochastic_from_dist(
            'outliernormal', logp=outlier_likelihood, dtype=np.float,
            mv=True)

        y_outlier = OutlierNormal(
            'y_outlier', mu=model, dyi=dyi, Yb=Yb, sigmab=sigmab, qi=qi,
            observed=True, value=yi)

        self.M = dict(y_outlier=y_outlier, beta=beta, model=model,
                      qi=qi, Pb=Pb, Yb=Yb, log_sigmab=log_sigmab,
                      sigmab=sigmab)

        self.sample_invoked = False

    def sample(self, iter, burn, calc_deviance=True):
        self.S0 = pymc.MCMC(self.M)
        self.S0.sample(iter=iter, burn=burn)
        self.trace = self.S0.trace('beta')
        self.btrace = self.trace[:, 0]
        self.mtrace = self.trace[:, 1]

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

        # plot circles around points identified as outliers
        qi = self.S0.trace('qi')[:]
        Pi = qi.astype(float).mean(0)
        outlier_x = self.xi[Pi < 0.32]
        outlier_y = self.yi[Pi < 0.32]
        plt.scatter(outlier_x, outlier_y, lw=1, s=400, alpha=0.5,
                    facecolors='none', edgecolors='red')

        plt.xlabel(xlab)
        plt.ylabel(ylab)

    def ICs(self):
        self.MAP = pymc.MAP(self.M)
        self.MAP.fit()

        self.BIC = self.MAP.BIC
        self.AIC = self.MAP.AIC
        self.logp = self.MAP.logp
        self.logp_at_max = self.MAP.logp_at_max
        return self.AIC, self.BIC
