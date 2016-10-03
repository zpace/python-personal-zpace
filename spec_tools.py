'''
some tools for working with astronomical spectra
'''

import numpy as np

from astropy import units as u, constants as c


def D4000_index(l, s):
    '''
    compute D4000 index

    params:
     - l: wavelength array [Angstroms], length n
     - s: spectrum [flux or flux density units], length n
    '''

    blue = [3850., 3950.]
    red = [4100., 4200.]

    blueflux = (s[(l > blue[0]) * (l < blue[-1])]).sum()
    redflux = (s[(l > red[0]) * (l < red[-1])]).sum()

    return redflux / blueflux


def HdA_index(l, s):
    '''
    compute HdA index

    params:
     - l: wavelength array [Angstroms], length n
     - s: spectrum [flux or flux density units], length n
    '''

    blue = [4041.600, 4079.750]
    red = [4128.500, 4161.000]
    line = [4083.500, 4122.250]

    assert len(l_edges) == len(s) + 1, 'invalid length of l_edges'

    lred = l[(l > red[0]) * (l < red[-1])]
    lblue = l[(l > blue[0]) * (l < blue[-1])]
    lline = l[(l > line[0]) * (l < line[-1])]
    dl = np.mean(lline[1:] - lline[:-1])
    F_c = np.append(
            s[(l > red[0]) * (l < red[-1])],
            s[(l > blue[0]) * (l < blue[-1])]).mean()

    EW = (1. - s[(l > line[0]) * (l < line[-1])] / F_c) * dl
    return EW


def shift_to_rest_roll(v, dlogl=None):
    '''
    return number of pixels of roll needed to cancel out doppler shift
    '''
    if dlogl is None:
        dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)

    z = (v / c.c).to('').value
    npix = -int(np.rint(z / dlogl))
    return npix


def shift_to_rest(logl, v):
    '''
    apply a constant velocity offset to a log-wavelength grid
    (i.e., DE-redshift the grid)
    '''
    z = (v / c.c).to('').value
    return logl - z


def determine_dlogl(logl):
    dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)
    return dlogl
