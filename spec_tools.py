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
    red = [4000., 4100.]

    bluef = (s[(l > blue[0]) * (l < blue[-1]), ...]).mean(axis=0)
    redf = (s[(l > red[0]) * (l < red[-1]), ...]).mean(axis=0)

    return redf / bluef


def Hdelta_A_index(l, s, dl=None):
    '''
    compute HdA index

    define index wrt blue continuum and red continuum

    params:
     - l: wavelength array [Angstroms], length n
     - s: spectrum [flux or flux density units], length n
    '''

    if dl is None:
        dl = determine_dl(l)

    blue = [4041.600, 4079.750]
    red = [4128.500, 4161.000]
    line = [4083.500, 4122.250]

    lred_m = (l > red[0]) * (l < red[-1])
    lblue_m = (l > blue[0]) * (l < blue[-1])
    lline_m = (l > line[0]) * (l < line[-1])

    lred = l[lred_m]
    lblue = l[lblue_m]
    lline = l[lline_m]

    sline = s[lline_m, ...]

    # calculate expected continuum flux within line
    red_avg = s[lred_m, ...].mean(axis=0)
    blue_avg = s[lblue_m, ...].mean(axis=0)
    cont_slope = (red_avg - blue_avg) / (red[0] - blue[-1])
    cont_interp = lambda l: cont_slope * (l - blue[0]) + blue_avg
    # use trapezoidal rule to approximate deficit in continuum flux-density
    EW = np.trapz(x=l[lline_m], y=(1. - (sline / cont_interp(lline))),
                  axis=0)

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

def determine_dl(l):
    logl = np.log10(l)
    dlogl = determine_dlogl(logl)
    llllim, llulim = logl - dlogl/2., logl + dlogl/2.
    lllim, lulim = np.log10(llllim), np.log10(llulim)
    dl = lulim - lllim
    return dl

def cube2rss(a, axis=0):
    '''
    transform cube-shaped data into RSS-shaped data
    '''
    if type(a) is not list:
        a = [a, ]

    a = np.row_stack([a_.reshape((-1, ) + (a_.shape[axis], )).T for a_ in a])
    return a
