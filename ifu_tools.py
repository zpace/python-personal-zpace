import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
import astropy.table as table
import matplotlib as mpl
from copy import copy
from gz2tools import download_sloan_im


class gp_kriging_fits(object):

    '''
    do sklearn gaussian process kriging on results from spectral fitting
        (such as SPaCT)
    assume perfect knowledge of the fiber locations

    Parameters:
        - objname
        - fiberfits_path
        - q: quantity of interest (either 'Velocity', 'Velocity Dispersion', 'Stellar Metallicity', or 'Stellar Age')
    '''

    def __init__(self, objname, fiberfits_path, qty, theta0=4.687/2,
                 thetaU=10., thetaL=1e-1, regr='linear', **gpparams):
        # first read in fiberfits table
        fiberfits_raw = table.Table.read(fiberfits_path, format='ascii')
        fiberfits = fiberfits_raw[(np.isnan(fiberfits_raw['V']) == False)]  # *
        #                          (np.abs(fiberfits_raw['V']) < 1000.)]

        self.objname = objname
        self.ra = fiberfits['ra']
        self.dec = fiberfits['dec']
        self.qty = qty

        gpparams['thetaU'] = thetaU
        gpparams['thetaL'] = thetaL
        gpparams['theta0'] = theta0
        gpparams['regr'] = regr

        self.qtys = {
            'Velocity':
                {'repr': r'$V$', 'qstr': 'V', 'u': 'km/s', 'err': 'dV',
                 'shift': True, 'extend': 'both',
                 'cmparams': {'cmap': 'RdBu_r',
                              'vmin': -600., 'vmax': 600.},
                 'fmt': '%.0f'},
            'Velocity Dispersion':
                {'repr': r'$\sigma$', 'qstr': 'sigma', 'u': 'km/s',
                 'err': 'dsigma', 'shift': False,
                 'extend': 'max',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': 0., 'vmax': 300.},
                 'fmt': '%.0f'},
            'Stellar Metallicity':
                {'repr': r'$Z$', 'qstr': 'Z', 'u': 'M/H',
                 'err': None, 'shift': False,
                 'extend': 'max',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': -1., 'vmax': 0.5},
                 'fmt': '%.2f'},
            'Stellar Age':
                {'repr': r'$\tau$', 'qstr': 't', 'u': 'Gyr', 'err': None,
                 'shift': False, 'extend': 'none',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': 0., 'vmax': 13.6},
                 'fmt': '%.0f'}
        }

        self.coords = np.column_stack((self.ra, self.dec))
        self.q0 = fiberfits[self.qtys[qty]['qstr']].ravel()
        if self.qtys[qty]['err'] != None:
            self.q0_err = fiberfits[self.qtys[self.qty]['err']].ravel()
            self.nugget0 = (self.q0_err/self.q0)**2.
        else:
            self.q0_err = None
            self.nugget0 = 2.2204460492503131e-15

        gpparams['nugget'] = self.nugget0

        # set up an initial fit to find the zeropoint
        self.gp0 = gaussian_process.GaussianProcess(**gpparams)
        self.gp0.fit(self.coords, self.q0)
        q_ctr, q_ctr_err2 = self.gp0.predict([[0., 0.]], eval_MSE=True)

        # if we're fitting a shifted quantity, then shift,
        # redefine the nugget, and fit the whole grid
        if self.qtys[qty]['shift'] == True:
            print 'Shifting... V_ctr = {0[0]:.0f} +/- {1[0]:.0f} km/s'.format(
                q_ctr, np.sqrt(q_ctr_err2))
            self.q1 = self.q0 - q_ctr[0]
            if fself.qtys[qty]['err'] != None:
                self.nugget1 = (np.sqrt(self.q0_err**2. +
                                        q_ctr_err2)/self.q1)**2.
            else:
                self.nugget1 = self.nugget0
            # q_ctr_err2 *is* squared, so don't square it again!
        else:
            self.q1 = self.q0
            self.nugget1 = self.nugget0

        gpparams['nugget'] = self.nugget1

        ragrid, decgrid = np.meshgrid(
            np.linspace(self.ra.min(), self.ra.max(), 1000),
            np.linspace(self.dec.min(), self.dec.max(), 1000))
        x_pred = np.column_stack((ragrid.ravel(), decgrid.ravel()))

        self.gp1 = gaussian_process.GaussianProcess(**gpparams)
        self.gp1.fit(self.coords, self.q1)
        self.y_pred, self.sigma2_pred = self.gp1.predict(x_pred, eval_MSE=True)
        self.sigma_pred = np.sqrt(self.sigma2_pred)
        print 'theta: {0:.2f} +/- {0:.2f}'.format(np.mean(self.gp1.theta_),
                                                  np.std(self.gp1.theta_))

    def fits_show(self):
        plt.close('all')

        ragrid, decgrid = np.meshgrid(
            np.linspace(self.ra.min(), self.ra.max(), 1000),
            np.linspace(self.dec.min(), self.dec.max(), 1000))

        fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 4.5))

        cmparams = self.qtys[self.qty]['cmparams']
        cmparams['levels'] = np.linspace(
            cmparams['vmin'], cmparams['vmax'], 13)
        extent = [ragrid.min(), ragrid.max(),
                  decgrid.min(), decgrid.max()]

        # if there's an error map for the quantity available, use it!
        if self.qtys[self.qty]['err'] != None:
            err_im = ax1.pcolormesh(ragrid, decgrid,
                                    self.sigma_pred.reshape((1000, 1000)),
                                    cmap='cubehelix')
            cb = plt.colorbar(err_im,
                              label=r'$\Delta${}[{}]'.format(
                                  self.qtys[self.qty]['qstr'],
                                  self.qtys[self.qty]['u']))
            cb.ax.tick_params(labelsize='small')
            cb.ax.yaxis.label.set_font_properties(
                mpl.font_manager.FontProperties(size='small'))
        # otherwise, just use an image of the object
        else:
            im = mpl.image.imread(self.objname + '.png')
            x = y = np.linspace(-40., 40., np.shape(im)[0])
            X, Y = np.meshgrid(x, y)
            galim = ax1.imshow(
                im, extent=[-40, 40, -40, 40], origin='lower',
                interpolation='nearest', zorder=0, aspect='equal')

        CS = ax1.contour(ragrid, decgrid,
                         self.y_pred.reshape((1000, 1000)), **cmparams)
        ax1.clabel(CS, fontsize='small', fmt=self.qtys[self.qty]['fmt'])
        ax1.set_aspect('equal')

        ax1.set_title('{}: {} fit'.format(self.objname, self.qty))
        ax1.set_xlabel(r'$\Delta$ RA [arcsec]', size='small')
        ax1.set_ylabel(r'$\Delta$ DEC [arcsec]', size='small')
        l = [tick.label.set_fontsize('small')
             for tick in ax1.xaxis.get_major_ticks()]
        l = [tick.label.set_fontsize('small')
             for tick in ax1.yaxis.get_major_ticks()]
        plt.gcf().tight_layout()
        plt.show()
        return fig
