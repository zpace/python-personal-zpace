import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
import astropy.table as table
import matplotlib as mpl
from copy import copy


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

    def __init__(self, objname, fiberfits_path, qty, **gpparams):
        # first read in fiberfits table
        fiberfits_raw = table.Table.read(fiberfits_path, format='ascii')
        fiberfits = fiberfits_raw[np.isnan(fiberfits_raw['V']) == False]

        self.ra = fiberfits['ra']
        self.dec = fiberfits['dec']
        self.qty = qty

        self.qtys = {
            'Velocity':
                {'repr': r'$V$', 'qstr': 'V', 'u': 'km/s', 'err': 'dV',
                 'shift': True, 'extend': 'both',
                 'cmparams': {'cmap': 'RdBu_r',
                              'vmin': -600., 'vmax': 600.}},
            'Velocity Dispersion':
                {'repr': r'$\sigma', 'qstr': 'sigma', 'u': 'km/s',
                 'err': 'dsigma', 'shift': False,
                 'extend': 'max',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': 0., 'vmax': 300.}},
            'Stellar Metallicity':
                {'repr': r'$Z$', 'qstr': 'Z', 'u': 'M/H',
                 'err': None, 'shift': False,
                 'extend': 'max',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': -1., 'vmax': 0.5}},
            'Stellar Age':
                {'repr': r'$\tau$', 'qstr': 't', 'u': 'Gyr', 'err': None,
                 'shift': False, 'extend': 'none',
                 'cmparams': {'cmap': 'cubehelix_r',
                              'vmin': 0., 'vmax': 13.6}}
        }

        self.coords = np.column_stack((self.ra, self.dec))
        self.q0 = fiberfits[self.qtys[qty]['qstr']].ravel()
        self.q0_err = fiberfits[self.qtys[qty]['err']].ravel()

        # set up an initial fit to find the zeropoint
        self.nugget0 = (self.q0_err/self.q0)**2.
        gpparams['nugget'] = self.nugget0
        self.gp0 = gaussian_process.GaussianProcess(**gpparams)
        self.gp0.fit(self.coords, self.q0)
        q_ctr, q_ctr_err2 = self.gp0.predict([[0., 0.]], eval_MSE=True)

        # if we're fitting a shifted quantity, then shift,
        # redefine the nugget, and fit the whole grid
        if self.qtys[qty]['shift'] == True:
            print 'Shifting... V_ctr = {0[0]:.0f} +/- {1[0]:.0f} km/s'.format(
                q_ctr, np.sqrt(q_ctr_err2))
            self.q1 = self.q0 - q_ctr[0]
            self.nugget1 = (np.sqrt(self.q0_err**2. + q_ctr_err2)/self.q1)**2.
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

    def fits_show(self):
        plt.close('all')

        ragrid, decgrid = np.meshgrid(
            np.linspace(self.ra.min(), self.ra.max(), 1000),
            np.linspace(self.dec.min(), self.dec.max(), 1000))

        fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 6.5))

        cmparams = self.qtys[self.qty]['cmparams']
        cmparams['levels'] = np.linspace(-600., 600., 13)
        extent = [ragrid.min(), ragrid.max(),
                  decgrid.min(), decgrid.max()]

        err_im = ax1.pcolormesh(ragrid, decgrid,
                                self.sigma_pred.reshape((1000, 1000)),
                                cmap='cubehelix_r')
        plt.colorbar(err_im, shrink=0.8,
                     label=r'$\Delta${}[{}]'.format(
                         self.qtys[self.qty]['qstr'],
                         self.qtys[self.qty]['u']))
        CS = ax1.contour(ragrid, decgrid,
                         self.y_pred.reshape((1000, 1000)), **cmparams)
        ax1.clabel(CS, fontsize='small', fmt='%.0f')
        ax1.set_aspect('equal')
        plt.show()
