import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
import astropy.table as table


class gp_kriging_fits(object):

    '''
    do sklearn gaussian process kriging on results from spectral fitting
        (such as SPaCT)
    assume perfect knowledge of the fiber locations

    Parameters:
        - objname
        - fiberfits_path
        - q: quantity of interest (either 'Velocity', 'Velocity Dispersion', 'Metallicity', or 'Age')
    '''
    def __init__(objname, fiberfits_path, qty, **gpparams):
        # first read in fiberfits table
        fiberfits_raw = table.Table.read(fiberfits_path, format='ascii')
        fiberfits = fiberfits_raw[np.isnan(fiberfits_raw['V']) == False]

        self.ra = fiberfits['ra']
        self.dec = fiberfits['dec']

        self.qtys = {
            'Velocity':
                {'repr': r'$V$', 'qstr': 'V', 'u': 'km/s', 'err': 'dV',
                 'eo': 'odd', 'shift': True},
            'Velocity Dispersion':
                {'repr': r'$\sigma', 'qstr': 'sigma', 'u': 'km/s',
                 'err': 'dsigma', 'eo': 'neither', 'shift': False},
            'Metallicity':
                {'repr': r'$Z$', 'qstr': 'Z', 'u': 'M/H',
                 'err': None, 'eo': 'neither', 'shift': False},
            'Age':
                {'repr': r'$\tau$', 'qstr': 't', 'u': 'Gyr', 'err': None,
                 'eo': 'neither', 'shift': False}
        }

        self.coords = np.column_stack((self.ra, self.dec))
        self.q = fiberfits[qtys['qstr']].ravel()
        self.q_err = fiberfits[qtys['err']].ravel()
        nugget = (self.q_err/self.q)**2.

        gpparams['nugget'] = nugget
        self.gp = gaussian_process.GaussianProcess(**gpparams)
        self.gp.fit(coords, q)

        ragrid, decgrid = np.meshgrid(
            np.linspace(ra.min(), ra.max(), 1000),
            np.linspace(dec.min(), dec.max(), 1000))
        x_pred = np.column_stack((ragrid.ravel(), decgrid.ravel()))

        self.y_pred, self.sigma2_pred = self.gp.predict(x_pred, eval_MSE=True)
        q_ctr, q_ctr_err2 = gp.predict([[0., 0.]], eval_MSE=True)
        q_ctr_err = np.sqrt(q_ctr_err2)
        self.ctr = (q_ctr, q_ctr_err)
        return x_pred, self.y_pred, np.sqrt(self.sigma2_pred), self.ctr

    def fits_show():
        plt.close('all')


def gp_kriging_fits_show(objname, x_pred, y_pred, sigma_pred,
                         ctr, shift=False):
    '''
    display the results of the GP kriging fits
    '''

    plt.close('all')

    ra = x_pred[:, 0].reshape((1000, 1000))
    dec = x_pred[:, 1].reshape((1000, 1000))

    # if you're looking at velocity, apply a constant shift
    # to bring things to rest-frame
    if shift == True:
        y_pred -= ctr[0]
        print '{0[0]:.0f} +/- {1[0]:.0f}'.format(*ctr)

    y_pred = y_pred.reshape((1000, 1000))

    fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 6.5))
    ax1.set_aspect('equal')

    # c = ax1.contour(
    #    ra, dec, y_pred, levels=np.linspace(-600., 600., 13), cmap='RdBu_r')
    c = ax1.imshow(y_pred, extent=[ra.min(), ra.max(), dec.min(), dec.max()],
                   interpolation='none')
    #plt.clabel(c, inline=1, fontsize=10, fmt='%.1f')
    plt.colorbar(c)
    plt.tight_layout()
    plt.show()
