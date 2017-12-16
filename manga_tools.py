import numpy as np

import astropy.table as t
import astropy.io.fits as fits
from astropy import (units as u, constants as c,
                     coordinates as coords, table as t,
                     wcs, cosmology as cosmo)

import os, sys

from matplotlib import pyplot as plt, patches

from glob import glob

from matplotlib import gridspec, colors
import matplotlib.ticker as mtick

import gz2tools as gz2
import copy

# add manga RC location to path, and import config
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

drpall_loc = '/home/zpace/Documents/MaNGA_science/'
dap_loc = '/home/zpace/mangadap/default/'
pw_loc = drpall_loc + '.saspwd'

uwdata_loc = '/d/www/karben4/'

MPL_versions = {'MPL-3': 'v1_3_3', 'MPL-4': 'v1_5_1', 'MPL-5': 'v2_0_1',
                'MPL-6': 'v2_3_1'}

base_url = 'dtn01.sdss.org/sas/'
mangaspec_base_url = base_url + 'mangawork/manga/spectro/redux/'

CosmoModel = cosmo.WMAP9

# =====
# units
# =====

# Maggy (SDSS flux unit)
Mgy = u.def_unit(
    s='Mgy', represents=3631. * u.Jy,
    doc='SDSS flux unit', prefixes=True)


def res_over_plate(version, plate='7443', plot=False, **kwargs):
    '''
    get average resolution vs logL over all IFUs on a single plate
    '''

    # double-check that everything on a plate is downloaded
    get_whole_plate(version, plate, dest='.', **kwargs)

    fl = glob('manga-{}-*-LOGCUBE.fits.gz'.format(plate))

    # load in each file, get hdu#5 data, and average across bundles
    print('READING IN HDU LIST...')
    specres = np.array(
        [fits.open(f)['SPECRES'].data for f in fl])

    l = np.array([wave(fits.open(f)).data for f in fl])
    lp = np.percentile(l, 50, axis=0)

    p = np.percentile(specres, [14, 50, 86], axis=0)

    if plot:
        print('PLOTTING...')
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.plot(lp, p[1], color='b', linewidth=3, label=r'50$^{th}$ \%-ile')
        ax.fill_between(lp, p[0], p[2],
                        color='purple', alpha=0.5, linestyle='--',
                        label=r'14$^{th}$ & 86$^{th}$ \%-ile')
        for i in specres:
            ax.plot(lp, i, color='r', alpha=0.1, zorder=5)
        ax.set_xlabel(r'$\lambda~[\AA]$')
        ax.set_ylabel('Spectral resolution')
        ax.set_ylim([0., ax.get_ylim()[1]])
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    # calculate average percent variation
    specres_var = np.abs(specres - p[1]) / specres
    print('Average % variability of SPECRES: {0:.5f}'.format(
          specres_var.mean()))

    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(specres_var.flatten(), bins=50)
        ax.set_xlabel(r'$\frac{\Delta R}{\bar{R}}$')
        plt.tight_layout()
        plt.show()

    return p[1], lp  # return 50th percentile (median)


def read_datacube(fname):
    hdu = fits.open(fname)
    return hdu

def shuffle_table(tab):
    a = np.arange(len(tab))
    np.random.shuffle(a)
    tab = tab[a]
    return tab

def mask_from_maskbits(a, b=[10]):
    '''
    transform a maskbit array into a cube or map mask
    '''

    # assume 32-bit maskbits
    a = a.astype('uint32')
    n = 32
    m = np.bitwise_or.reduce(
        ((a[..., None] & (1 << np.arange(n))) > 0)[..., b], axis=-1)

    return m

class MaNGA_DNE_Error(Exception):
    '''
    generic file access does-not-exist
    '''

class IFU_DNE_Error(MaNGA_DNE_Error):
    '''
    generic IFU does-not-exist error, for both DRP & DAP
    '''

class DRP_IFU_DNE_Error(IFU_DNE_Error):

    '''
    DRP IFU does not exist
    '''

    def __init__(self, plate, ifu):
        self.plate = plate
        self.ifu = ifu

    def __str__(self):
        return 'reduced products for {}-{} DNE in the given location!'.format(
            self.plate, self.ifu)


class DAP_IFU_DNE_Error(IFU_DNE_Error):

    '''
    DAP IFU does not exist
    '''

    def __init__(self, plate, ifu, kind):
        self.plate = plate
        self.ifu = ifu
        self.kind = kind

    def __str__(self):
        return 'analyzed products for {}-{} ({}) \
                DNE in the given location!'.format(
            self.plate, self.ifu, self.kind)

def load_drpall(mpl_v, index=None):
    fname = os.path.join(
        mangarc.manga_data_loc[mpl_v],
        'drpall-{}.fits'.format(MPL_versions[mpl_v]))
    tab = t.Table.read(fname)

    if index is not None:
        tab.add_index(index)

    return tab

def load_drp_logcube(plate, ifu, mpl_v):
    plate, ifu = str(plate), str(ifu)
    fname = os.path.join(
        mangarc.manga_data_loc[mpl_v], 'drp/', plate, 'stack/',
        '-'.join(('manga', plate, ifu, 'LOGCUBE.fits.gz')))

    if not os.path.isfile(fname):
        raise DAP_IFU_DNE_Error(plate, ifu, kind)

    hdulist = fits.open(fname)

    return hdulist

def get_gal_bpfluxes(plate, ifu, mpl_v, bs, th=5.):
    hdulist = load_drp_logcube(plate, ifu, mpl_v)
    snr = np.median(
        hdulist['FLUX'].data * np.sqrt(hdulist['IVAR'].data), axis=0)
    fluxes = np.column_stack(
        [hdulist['{}IMG'.format(band.upper())].data[(snr >= th)]
        for band in bs])
    hdulist.close()
    return fluxes

def get_drp_hdrval(plate, ifu, mpl_v, k):
    hdulist = load_drp_logcube(plate, ifu, mpl_v)
    val = hdulist[0].header[k]
    hdulist.close()
    return val

def hdu_data_extract(hdulist, names):
    return [hdulist[n].data for n in names]

def load_dap_maps(plate, ifu, mpl_v, kind):
    plate, ifu = str(plate), str(ifu)
    fname = os.path.join(
        mangarc.manga_data_loc[mpl_v], 'dap/', kind, plate, ifu,
        '-'.join(('manga', plate, ifu, 'MAPS',
                  '{}.fits.gz'.format(kind))))

    if not os.path.isfile(fname):
        raise DRP_IFU_DNE_Error(plate, ifu)

    hdulist = fits.open(fname)

    return hdulist
