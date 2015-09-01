import numpy as np
import astropy
import astropy.table as table
import astropy.io.fits as fits
import pandas as pd
import os
import re
from matplotlib import rcParams, pyplot as plt
from astropy.wcs import WCS
from astropy import units as u, constants as c

drpall_loc = '/home/zpace/Documents/MaNGA_science/'
pw_loc = drpall_loc + '.saspwd'

MPL_versions = {'MPL-3': 'v1_3_3'}

base_url = 'dtn01.sdss.org/sas/'
mangaspec_base_url = base_url + 'mangawork/manga/spectro/redux/'


def get_platelist(version, dest=drpall_loc, **kwargs):
    '''
    retrieve the drpall file for a particular version or MPL, and place
    in the indicated folder, or in the default location
    '''

    # call `get_something`
    platelist_fname = 'platelist.fits'
    get_something(version, platelist_fname, dest, **kwargs)


def get_drpall(version, dest=drpall_loc, **kwargs):
    '''
    retrieve the drpall file for a particular version or MPL, and place
    in the indicated folder, or in the default location
    '''

    # call `get_something`
    drp_fname = 'drpall-{}.fits'.format(MPL_versions[version])
    get_something(version, drp_fname, dest, **kwargs)


def get_something(version, what, dest='.', verbose=False):
    # are we working with an MPL designation or a simple version number?
    if type(version) == int:
        vtype = 'MPL'
        version = 'MPL-' + str(version)
    elif version[:3] == 'MPL':
        vtype = 'MPL'

    v_url = mangaspec_base_url + version

    full_url = v_url + '/' + what

    rsync_cmd = 'rsync -avz --password-file {0} rsync://sdss@{1} {2}'.format(
        pw_loc, full_url, dest)

    if verbose == True:
        print 'Here\'s what\'s being run...\n\n{0}\n'.format(rsync_cmd)

    os.system(rsync_cmd)


def get_datacube(version, plate, bundle, dest, **kwargs):
    '''
    retrieve a full, log-rebinned datacube of a particular galaxy
    '''

    what = '{0}/stack/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate, bundle)

    get_something(version, what, dest, **kwargs)


def get_RSS(version, plate, bundle, dest, **kwargs):
    '''
    retrieve a full, log-rebinned datacube of a particular galaxy
    '''

    what = '{0}/stack/manga-{0}-{1}-LOGRSS.fits.gz'.format(plate, bundle)

    get_something(version, what, dest, **kwargs)


def read_datacube(fname):
    hdu = fits.open(fname)
    return hdu


def good_spaxels(hdu):
    '''
    yield a binary array that is zero in spaxels where:
        - the IFU was never on that sky (ivargood)
        - there's no signal whatsoever (fluxgood)
    '''

    flux = hdu['FLUX'].data
    ivar = hdu['IVAR'].data

    ivargood = (np.sum(ivar, axis=0) != 0)
    fluxgood = (np.sum(flux, axis=0) != 0)

    return ivargood * fluxgood


def wave(hdu):
    '''
    yield a wavelength array
    '''

    return hdu['WAVE']


def target_data(hdu):
    h = hdu[0].header

    objra, objdec = h['OBJRA'], h['OBJDEC']
    cenra, cendec = h['CENRA'], h['CENDEC']
    mangaID = h['MANGAID']

    return objra, objdec, cenra, cendec, mangaID


def make_ifu_fig(hdu):
    '''
    DOES NOT WORK
    '''
    wcs = WCS(hdu[0].header)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_xlabel(r'RA')
    ax.set_ylabel(r'Dec')
    return fig


def conroy_to_fits(fname):
    '''
    translate a Conroy-style SSP (at one metallicity)
        to a bunch of fits tables

    Conroy format

    <HEADER LINES> (beginning with #)
    <NAXIS1> (number of spectral bins) <NAXIS2> (number of age bins)


    '''

    names = ['logT', 'logM', 'logLbol', 'logSFR', 'spectra']

    data = []

    with open(fname) as f:
        for i, l in enumerate(f):
            if l[0] == '#':
                pass
            else:
                data.append(
                    [float(j) for j in l.rstrip('\n').lstrip(' ').split()])

    nT, nL = data.pop(0)  # number of age bins and number of wavelengths
    nT, nL = int(nT), int(nL)
    l = table.Column(data=data.pop(0) * u.Angstrom,
                     name='lambda')  # wavelengths

    assert len(l) == nL, 'There should be {} elements in nL, \
        but there are {}'.format(len(l), len(nL))

    # now restrict the wavelength range to a usable interval
    lgood = ((1500. * u.Angstrom <= l) * (l <= 1.1 * u.micron))
    l = l[lgood]

    # log-transform
    logl = np.log(l / u.Angstrom)

    dlogl = np.mean(logl[1:] - logl[:-1])
    CRVAL1 = logl[0]
    CDELT1 = dlogl
    NAXIS1 = len(logl)
    # print CRVAL1, CDELT1, NAXIS1

    spectra_flam = data[1::2] * u.solLum/u.Hz  # spectra are every second row
    metadata = data[::2]

    print len(metadata)

    age = table.Column([10.**i[0] for i in metadata] * u.year,
                       name='age')
    mass = table.Column([10.**i[1] for i in metadata] * u.solMass,
                        name='mass')
    lbol = table.Column([10.**i[2] for i in metadata] * u.solLum,
                        name='lbol')
    SFR = table.Column([10.**i[3] for i in metadata] * u.solMass/u.year,
                       name='SFR')

    # now convert to f_lambda
    spectra_fnu = [np.asarray(s[lgood] * (c.c/l**2.)) for s in spectra_flam]

    print spectra_fnu

    return table.Table(data=[age, mass, lbol, SFR])

    '''
    These spectra are in units of Lsun/Hz.
    We want them in
    '''
