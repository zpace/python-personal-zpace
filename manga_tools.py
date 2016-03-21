import numpy as np
import astropy
import astropy.table as table
import astropy.io.fits as fits
import pandas as pd
import os
import re
from matplotlib import rcParams, pyplot as plt, patches
from astropy.wcs import WCS
from astropy import units as u, constants as c, coordinates as coords, wcs
from glob import glob
from scipy.interpolate import interp1d
from pysynphot import observation, spectrum
from matplotlib import gridspec, colors
import matplotlib.ticker as mtick
import pywcsgrid2
import itertools
import gz2tools as gz2
import copy

drpall_loc = '/home/zpace/Documents/MaNGA_science/'
dap_loc = '/home/zpace/mangadap/default/'
pw_loc = drpall_loc + '.saspwd'

uwdata_loc = '/d/www/karben4/'

MPL_versions = {'MPL-3': 'v1_3_3', 'MPL-4': 'v1_5_1'}

base_url = 'dtn01.sdss.org/sas/'
mangaspec_base_url = base_url + 'mangawork/manga/spectro/redux/'

#c = 299792.458 #km/s
H0 = 70. #km/s/Mpc

def get_drpall_val(fname, qtys, plateifu):
    drpall = table.Table.read(fname)
    obj = drpall[drpall['plateifu'] == plateifu]
    #print drpall.colnames
    return obj[qtys]

def get_cutout(version, plateifu, verbose=False):

    plate, ifudsgn = plateifu.split('-')
    dest = plateifu + '.png'

    what = '{}/stack/images/{}.png'.format(plate, ifudsgn)
    get_something(version, what, dest, verbose)

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

    if verbose == True:
        v = 'v'
    else:
        v = ''

    rsync_cmd = 'rsync -a{3}z --password-file {0} rsync://sdss@{1} {2}'.format(
        pw_loc, full_url, dest, v)

    if verbose == True:
        print 'Here\'s what\'s being run...\n\n{0}\n'.format(rsync_cmd)

    os.system(rsync_cmd)


def get_datacube(version, plate, bundle, dest, **kwargs):
    '''
    retrieve a full, log-rebinned datacube of a particular galaxy
    '''

    what = '{0}/stack/manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate, bundle)

    get_something(version, what, dest, **kwargs)


def get_whole_plate(version, plate, dest, **kwargs):
    '''
    get all MaNGA science datacubes from a certain plate
    '''

    # check if correct version of drpall exists. if not, get it
    if not os.path.isfile(
        '{0}drpall-{1}.fits'.format(
            drpall_loc, MPL_versions[version])):

        get_drpall(version)

    # now open drpall and see what IFUs are used

    drpall = table.Table.read(
        '{0}drpall-{1}.fits'.format(drpall_loc, MPL_versions[version]))

    drpall = drpall[drpall['plate'].astype(str) == plate]
    drpall = drpall[drpall['ifudsgn'].astype(int) > 712]

    for plate, bundle in zip(drpall['plate'], drpall['ifudsgn']):
        if not os.path.isfile(
                'manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate, bundle)):
            get_datacube(version, plate, bundle, dest, **kwargs)


def res_over_plate(version, plate='7443', plot=False, **kwargs):
    '''
    get average resolution vs logL over all IFUs on a single plate
    '''

    # double-check that everything on a plate is downloaded
    get_whole_plate(version, plate, dest='.', **kwargs)

    fl = glob('manga-{}-*-LOGCUBE.fits.gz'.format(plate))

    # load in each file, get hdu#5 data, and average across bundles
    print 'READING IN HDU LIST...'
    specres = np.array(
        [fits.open(f)['SPECRES'].data for f in fl])

    l = np.array([wave(fits.open(f)).data for f in fl])
    lp = np.percentile(l, 50, axis=0)

    p = np.percentile(specres, [14, 50, 86], axis=0)

    if plot == True:
        print 'PLOTTING...'
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        # print p.shape
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
    print 'Average % variability of SPECRES: {0:.5f}'.format(
        specres_var.mean())

    if plot == True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(specres_var.flatten(), bins=50)
        ax.set_xlabel(r'$\frac{\Delta R}{\bar{R}}$')
        plt.tight_layout()
        plt.show()

    return p[1], lp  # return 50th percentile (median)


def get_RSS(version, plate, bundle, dest, **kwargs):
    '''
    retrieve a full, log-rebinned row-stacked spectrum of a particular galaxy
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
    snrgood = (np.mean(flux*np.sqrt(ivar), axis=0) >= 1)

    return ivargood * fluxgood * snrgood


def wave(hdu):
    '''
    yield a wavelength array
    '''

    return hdu['WAVE']


def target_data(hdu):
    h = hdu[0].header

    objra, objdec = h['OBJRA'], h['OBJDEC']
    cenra, cendec = h['CENRA'], h['CENDEC']  # this is PLATE CENTER
    mangaID = h['MANGAID']

    return objra, objdec, cenra, cendec, mangaID

def read_dap_ifu(plate, ifu):
    hdu = fits.open('{0}{1}/mangadap-{1}-{2}-default.fits.gz'.format(
        dap_loc, plate, ifu))
    objname = '{}-{}'.format(plate, ifu)
    return hdu, objname

class DAP_elines(object):
    '''
    load emission line equivalent widths from the MAPS DAP output

        - hdu: FITS HDU of MaNGA DAP MAPS output
        - q: string identifying what quantity we're getting (e.g., 'EW')
    '''
    def __init__(self, hdu, q='EW', sn_t=3.):
        import re
        self.sn_t = sn_t
        self.qtype = q
        self.hdu = hdu

        if q not in ['EW', 'GFLUX', 'SFLUX']:
            em = 'Invalid FITS extension group: ' + \
                'choose "EW", "GFLUX", or "SFLUX"'
            raise KeyError(em)
        # Build dictionaries with the emission line names to ease selection
        self.q = 'EMLINE_{}'.format(q)
        emline = {}
        for k, v in hdu[self.q].header.items():
            if (k[0] == 'C') and (len(k) == 3):
                try:
                    i = int(k[1:])-1
                except KeyError:
                    continue
                emline[v] = i
        self.emline = emline

        # select the mask using the HDUCLASS structure
        mask_extension = hdu[self.q].header['QUALDATA']
        ivar_extension = hdu[self.q].header['ERRDATA']

        self.ivar_maps = {k: 64.*np.array(hdu[ivar_extension].data[v, :, :])
            for (k, v) in self.emline.iteritems()}

        self.qty_maps = {k: np.array(hdu[self.q].data[v, :, :])
                         for (k, v) in self.emline.iteritems()}

        self.mask_maps = {k: np.array(hdu[mask_extension].data[v, :, :])
                          for (k, v) in self.emline.iteritems()}

        #for (k, v) in self.emline.iteritems():
        #    print k, '\n', self.qty_maps[k] * np.sqrt(self.ivar_maps[k])

        self.SNR_maps = {k: np.ma.array(
            self.qty_maps[k] * np.sqrt(self.ivar_maps[k]),
            mask=self.mask_maps[k])
            for (k, v) in self.emline.iteritems()}

        self.SNR_mask_maps = {k: self.SNR_maps[k] < sn_t
                              for (k, v) in self.emline.iteritems()}

        # mask bad data and data where SNR < sn_t
        self.qty_maps = {k: np.ma.array(
            self.qty_maps[k], mask=(self.mask_maps[k]))
            for (k, v) in self.emline.iteritems()}

        self.eline_hdr = hdu[self.q].header

        self.lines_per_species = {}
        self.species_mask_maps = {}
        self.species_ivar_maps = {}
        self.species_SNR_maps = {}
        self.species_maps = {}
        # which lines belong to which species
        for k in self.qty_maps.keys():
            s, l = re.split(r'[\s-]+', k.replace('d', ''))
            if s not in self.lines_per_species:
                self.lines_per_species[s] = [k, ]
            else:
                self.lines_per_species[s].append(k)

        for s, ls in self.lines_per_species.iteritems():
            # masks
            # any is like logical_or
            self.species_mask_maps[s] = np.any([
                self.mask_maps[k] for k in ls], axis=0)
            # fluxes
            self.species_maps[s] = np.ma.array(np.sum(
                [self.qty_maps[k] for k in ls], axis=0),
                mask=self.species_mask_maps[s])
            # ivar maps
            self.species_ivar_maps[s] = np.ma.array(1./np.sum(
                [1./self.ivar_maps[k] for k in ls],
                axis=0), mask=self.species_mask_maps[s])
            # SNR maps
            self.species_SNR_maps[s] = np.ma.array(np.sqrt(
                self.species_ivar_maps[s]) * self.species_maps[s],
                mask=self.species_mask_maps[s])

    def map(self, save=False, objname=None, loc=''):
        # make quantity and SNR maps for each species
        # mostly for QA purposes

        import mpl_toolkits.axes_grid1.axes_grid as axes_grid
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cmap1 = plt.cm.cubehelix
        cmap1.set_bad('gray')

        cmap2 = plt.cm.Purples_r
        cmap2.set_bad('gray')

        n_species = len(self.emline) # number of rows of subplots
        n_cols = 2 # col 1 for qty, col 2 for SNR
        fig_dims = (3.*n_cols, 2.*n_species)

        fig = plt.figure(figsize=fig_dims, dpi=300)
        gh = pywcsgrid2.GridHelper(wcs=self.eline_hdr)
        g = axes_grid.ImageGrid(fig, 111,
                                nrows_ncols=(n_species, n_cols),
                                ngrids=None, direction='row',
                                axes_pad=[.5, .02],
                                add_all=True, share_all=True,
                                aspect=True, label_mode='L',
                                cbar_mode='each', cbar_location='right',
                                cbar_size='5%', cbar_pad='-5%',
                                axes_class=(pywcsgrid2.Axes,
                                            dict(grid_helper=gh)))


        for i, k in enumerate(sorted(self.emline.keys())):
            qpn = 2*i # quantity subplot number
            spn = 2*i + 1 # SNR subplot number

            # create arrays to represent shown quantities
            sa = np.ma.array(
                self.qty_maps[k] * np.sqrt(self.ivar_maps[k]),
                mask=(self.mask_maps[k] | (self.qty_maps[k] < 0)))
            qa = np.ma.array(
                self.qty_maps[k].data,
                mask=sa.mask)

            q_im = g[qpn].imshow(
                qa,
                interpolation=None, origin='lower',
                cmap=cmap2, norm=colors.LogNorm())
            s_im = g[spn].imshow(
                sa,
                interpolation=None,
                origin='lower', cmap=cmap1, norm=colors.LogNorm())

            s_c = g[qpn].contour(
                sa, levels=[self.sn_t,],
                colors='r', linewidths=1.,
                linestyles='-')

            g[spn].contour(
                sa, levels=[self.sn_t,],
                colors='r', linewidths=1., linestyles='-')

            g[qpn].grid()
            g[spn].grid()
            g[qpn].set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.qty_maps[k].shape))
            g[spn].set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.qty_maps[k].shape))
            g[qpn].add_inner_title('{}'.format(k.replace(
                '-','')), loc=2, frameon=False)
            # get rid of axes labels
            g[qpn].set_xlabel('')
            g[qpn].set_ylabel('')
            g[spn].set_xlabel('')
            g[spn].set_ylabel('')

            qcb = g.cbar_axes[qpn].colorbar(q_im)
            qcb.set_label_text(r'{} [{}]'.format(
                self.qtype, self.hdu[self.q].header['BUNIT'].replace(
                    '^2', '$^2$')), size=8)
            # configure quantity's colorbar
            # define max and min over range within factor of 4 of max SNR
            q_clims = [np.min(qa[sa >= 0.25*sa.max()]),
                       np.max(qa[sa >= 0.25*sa.max()])]
            dqt = int(np.log10(q_clims[-1]))
            q_ticks = np.arange(
                10.**dqt, q_clims[-1], 10.**dqt)
            if len(q_ticks) < 2:
                q_ticks = np.concatenate(
                    [np.arange(1., 10., 1.) * 10.**(dqt-1) * np.ones(9),
                     np.arange(1., 10., 1.) * 10.**dqt * np.ones(9),
                     np.arange(1., 10., 1.) * 10.**(dqt+1) * np.ones(9)])
            qcb.ax.set_yticks(q_ticks)
            qcb.ax.set_yticklabels(q_ticks, size=8)
            qcb.set_clim(q_clims)
            qcb.ax.set_ylim(q_clims)

            scb = g.cbar_axes[spn].colorbar(s_im)
            scb.set_label_text('SNR', size=8)
            # configure SNR's colorbar
            s_ticks = np.concatenate(
                [np.arange(1., 10., 1.) * 10.**(-1.) * np.ones(9),
                 np.arange(1., 10., 1.) * 10.**0. * np.ones(9),
                 np.arange(1., 10., 1.) * 10.**(1.) * np.ones(9)])
            s_clims = [0.5, scb.get_clim()[1]]
            scb.ax.set_yticks(s_ticks)
            scb.ax.set_yticklabels(s_ticks, size=8)
            scb.set_clim(s_clims)
            scb.ax.set_ylim(s_clims)

        plt.tight_layout()
        plt.subplots_adjust(top=.95)
        plt.suptitle(objname)

        if save == False:
            plt.show()
        else:
            plt.savefig('{}{}_eline_map.png'.format(loc, objname))

        plt.close()

    def species_map(self, save=False, objname=None, loc=''):
        # make quantity and SNR maps for each species
        # mostly for QA purposes

        import mpl_toolkits.axes_grid1.axes_grid as axes_grid
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cmap1 = plt.cm.cubehelix
        cmap1.set_bad('gray')

        cmap2 = plt.cm.Purples_r
        cmap2.set_bad('gray')

        n_species = len(self.lines_per_species) # number of rows of subplots
        n_cols = 2 # col 1 for qty, col 2 for SNR
        fig_dims = (3.*n_cols, 2.*n_species)

        fig = plt.figure(figsize=fig_dims, dpi=300)
        gh = pywcsgrid2.GridHelper(wcs=self.eline_hdr)
        g = axes_grid.ImageGrid(fig, 111,
                                nrows_ncols=(n_species, n_cols),
                                ngrids=None, direction='row',
                                axes_pad=[.5, .02],
                                add_all=True, share_all=True,
                                aspect=True, label_mode='L',
                                cbar_mode='each', cbar_location='right',
                                cbar_size='5%', cbar_pad='-5%',
                                axes_class=(pywcsgrid2.Axes,
                                            dict(grid_helper=gh)))

        for i, k in enumerate(sorted(self.lines_per_species.keys())):
            qpn = 2*i # quantity subplot number
            spn = 2*i + 1 # SNR subplot number

            # create arrays to represent shown quantities
            sa = np.ma.array(
                self.species_SNR_maps[k],
                mask=(self.species_SNR_maps[k].mask | \
                      (self.species_maps[k] < 0)))
            qa = np.ma.array(
                self.species_maps[k].data, mask=sa.mask)

            q_im = g[qpn].imshow(
                qa,
                interpolation=None, origin='lower',
                cmap=cmap2, norm=colors.LogNorm())
            s_im = g[spn].imshow(
                sa,
                interpolation=None,
                origin='lower', cmap=cmap1, norm=colors.LogNorm())

            s_c = g[qpn].contour(
                sa, levels=[0.5*self.sn_t, self.sn_t, 2.*self.sn_t],
                colors='r', linewidths=1.,
                linestyles=[':', '-', '-.'])

            g[spn].contour(
                sa, levels=[0.5*self.sn_t, self.sn_t, 2.*self.sn_t],
                colors='r', linewidths=1.,
                linestyles=[':', '-', '-.'])

            g[qpn].grid()
            g[spn].grid()
            g[qpn].set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.species_maps[k].shape))
            g[spn].set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.species_maps[k].shape))
            g[qpn].add_inner_title('{}'.format(k), loc=2, frameon=False)
            # get rid of axes labels
            g[qpn].set_xlabel('')
            g[qpn].set_ylabel('')
            g[spn].set_xlabel('')
            g[spn].set_ylabel('')

            qcb = g.cbar_axes[qpn].colorbar(q_im)
            qcb.set_label_text(r'{} [{}]'.format(
                self.qtype, self.hdu[self.q].header['BUNIT'].replace(
                    '^2', '$^2$')), size=8)
            # configure quantity's colorbar
            # define max and min over range within factor of 4 of max SNR
            q_clims = [np.min(qa[sa >= 0.25*sa.max()]),
                       np.max(qa[sa >= 0.25*sa.max()])]
            dqt = int(np.log10(q_clims[-1]))
            q_ticks = np.arange(
                10.**dqt, q_clims[-1], 10.**dqt)
            if len(q_ticks) < 2:
                q_ticks = np.concatenate(
                    [np.arange(1., 10., 1.) * 10.**(dqt-1) * np.ones(9),
                     np.arange(1., 10., 1.) * 10.**dqt * np.ones(9),
                     np.arange(1., 10., 1.) * 10.**(dqt+1) * np.ones(9)])
            qcb.ax.set_yticks(q_ticks)
            qcb.ax.set_yticklabels(q_ticks, size=8)
            qcb.set_clim(q_clims)
            qcb.ax.set_ylim(q_clims)

            scb = g.cbar_axes[spn].colorbar(s_im)
            scb.set_label_text('SNR', size=8)
            # configure SNR's colorbar
            s_ticks = np.concatenate(
                [np.arange(1., 10., 1.) * 10.**(-1.) * np.ones(9),
                 np.arange(1., 10., 1.) * 10.**0. * np.ones(9),
                 np.arange(1., 10., 1.) * 10.**(1.) * np.ones(9)])
            s_clims = [0.5, scb.get_clim()[1]]
            scb.ax.set_yticks(s_ticks)
            scb.ax.set_yticklabels(s_ticks, size=8)
            scb.set_clim(s_clims)
            scb.ax.set_ylim(s_clims)

        plt.tight_layout()
        plt.subplots_adjust(top=.95)
        plt.suptitle('\n'.join((objname, 'SNR: {}'.format(self.sn_t))))

        if save == False:
            plt.show()
        else:
            plt.savefig('{}{}_species_map.png'.format(loc, objname))

        plt.close()

    def to_BPT(self):
        # convenience method: returns a dict of lines, that you can
        # double-splat into BPT.__init__() below
        if not 'FLUX' in self.qtype:
            em = '{} is illegal -- try \'SLFUX\' or \'GFLUX\' (pref)'.format(
                self.qtype)
            raise ValueError(em)
        elif self.qtype != 'GFLUX':
            raise Warning('\'GFLUX\' is probably a better choice!')

        ldata = {'Ha': self.qty_maps['Ha-----6564'],
                 'Hb': self.qty_maps['Hb-----4862'],
                 'OIII': self.qty_maps['OIII---5008'],
                 'NII': self.qty_maps['NII----6585'],
                 'SII': self.qty_maps['SII----6732'] + \
                    self.qty_maps['SII----6718'],
                 'OI': self.qty_maps['OI-----6302'],
                 'qtype': self.qtype}

        return ldata

    def to_BPT_SNRmask(self):
        if not 'FLUX' in self.qtype:
            em = '{} is illegal -- try \'SLFUX\' or \'GFLUX\' (pref)'.format(
                self.qtype)
            raise ValueError(em)
        elif self.qtype != 'GFLUX':
            raise Warning('\'GFLUX\' is probably a better choice!')

        snmask = {
            'Ha_SNRm': self.species_SNR_maps['Ha'] > self.sn_t,
            'Hb_SNRm': self.species_SNR_maps['Hb'] > self.sn_t,
            'OIII_SNRm': self.SNR_maps['OIII---5008'] > self.sn_t,
            'NII_SNRm': self.SNR_maps['NII----6585'] > self.sn_t,
            'SII_SNRm': self.species_SNR_maps['SII'] > self.sn_t,
            'OI_SNRm': self.SNR_maps['OI-----6302'] > self.sn_t}

        return snmask

    def species_SNRmask(self):
        if not 'FLUX' in self.qtype:
            em = '{} is illegal -- try \'SLFUX\' or \'GFLUX\' (pref)'.format(
                self.qtype)
            raise ValueError(em)
        elif self.qtype != 'GFLUX':
            raise Warning('\'GFLUX\' is probably a better choice!')

        snmask = {
            'Ha_SNRm': self.species_SNR_maps['Ha'] > self.sn_t,
            'Hb_SNRm': self.species_SNR_maps['Hb'] > self.sn_t,
            'OIII_SNRm': self.species_SNR_maps['OIII'] > self.sn_t,
            'OII_SNRm': self.species_SNR_maps['OII'] > self.sn_t,
            'NII_SNRm': self.species_SNR_maps['NII'] > self.sn_t,
            'SII_SNRm': self.species_SNR_maps['SII'] > self.sn_t,
            'OI_SNRm': self.species_SNR_maps['OI'] > self.sn_t}

        return snmask

class BPT(object):
    '''
    BPT classification in three schemes:
        - [NII]/Ha vs [OIII]/Hb (scheme 1)
        - [SII]/Ha vs [OIII]/Ha (scheme 2)
        - [OI]/Ha vs [OIII]/Hb (scheme 3)

    (after Kewley et al., 2006, MNRAS372: 961-976)

    Doesn't care about dimensionality of input -- just that all
        inputs are same size

    In self.diag, 0 for ambiguous, 1 for star-forming,
        2 for composite, 3 for Seyfert, 4 for LI(N)ER.
    '''

    def __init__(self, Ha, Hb, OIII, NII, SII, OI, qtype, Ha_SNRm,
                 Hb_SNRm, OIII_SNRm, NII_SNRm, OI_SNRm, SII_SNRm, **kwargs):

        '''
        SF:
            - below and to left of Ka03 in [NII]/Ha vs [OIII]/Hb
            - below and to left of Ke01 in [SII]/Ha vs [OIII]/Hb
            - below and to left of Ke01 in [OI]/Ha vs [OIII]/Hb

        Composite:
            - between Ka03 and Ke01 in [NII]/Ha vs [OIII]/Hb
            - between Ka03 and Ke01 in [NII]/Ha vs [OIII]/Hb

        AGN:
            - above Ke01 in [NII]/Ha vs [OIII]/Hb
            - above Ke01 in [SII]/Ha vs [OIII]/Hb
            - above Ke01 in [OI]/Ha vs [OIII]/Hb
            - above Seyfert-LI(N)ER line in [SII]/Ha vs [OIII]/Hb
            - above Seyfert-LI(N)ER line in [OI]/Ha vs [OIII]/Hb

        LI(N)ER:
            - above Ke01 in [NII]/Ha vs [OIII]/Hb
            - above Ke01 in [SII]/Ha vs [OIII]/Hb
            - above Ke01 in [OI]/Ha vs [OIII]/Hb
            - below Seyfert-LI(N)ER line in [SII]/Ha vs [OIII]/Hb
            - below Seyfert-LI(N)ER line in [OI]/Ha vs [OIII]/Hb

        computes class where at least one of the lines considered in a
        particular scheme is known-good (this should be filtered down
        further for actual science)
        '''

        self.Ka03_cs = {'NII_Ha': {'u': 0.61, 'v': -0.05, 'w': 1.3, 'l': -.1}}
        self.Ke01_cs = {'SII_Ha': {'u': 0.72, 'v': -0.32, 'w': 1.3, 'l': .1},
                        'OI_Ha': {'u': 0.73, 'v': 0.59, 'w':1.33, 'l': .3},
                        'NII_Ha': {'u': 0.61, 'v': -0.47, 'w': 1.19, 'l': 1.7}}
        self.Sf_L_cs = {'SII_Ha': {'m': 1.89, 'b': 0.76},
                        'OI_Ha': {'m': 1.18, 'b': 1.3}}

        self.qtype = qtype

        # set mask arrays (measurements have to be bad for both to be masked)

        self.SII_Ha = np.ma.array(
            np.log10(SII/Ha),
            mask=(Ha.mask | SII.mask | (~Ha_SNRm & ~SII_SNRm)))
        self.NII_Ha = np.ma.array(
            np.log10(NII/Ha),
            mask=(Ha.mask | NII.mask | (~Ha_SNRm & ~NII_SNRm)))
        self.OI_Ha = np.ma.array(
            np.log10(OI/Ha),
            mask=(Ha.mask | OI.mask | (~Ha_SNRm)))
        self.OIII_Hb = np.ma.array(
            np.log10(OIII/Hb),
            mask=(OIII.mask | Hb.mask | (~Hb_SNRm & ~OIII_SNRm)))

        # compute over NII_Ha first
        SF_NII_Ha = ~self.Ka03_decision(self.NII_Ha, self.OIII_Hb,
                                        **self.Ka03_cs['NII_Ha'])
        comp_NII_Ha = (self.Ka03_decision(self.NII_Ha, self.OIII_Hb,
                                          **self.Ka03_cs['NII_Ha'])) * \
                      (~self.Ke01_decision(self.NII_Ha, self.OIII_Hb,
                                           **self.Ke01_cs['NII_Ha'])) * \
                      (self.NII_Ha < 0.25)
        # "nuclear"-dominated must be neither SF nor composite
        nuc_NII_Ha = ~(SF_NII_Ha | comp_NII_Ha)
        class_NII_Ha = np.argmax(
            np.stack((np.zeros_like(SF_NII_Ha), SF_NII_Ha, comp_NII_Ha, nuc_NII_Ha, np.zeros_like(SF_NII_Ha))), axis=0)
        self.class_NII_Ha = np.ma.array(
            class_NII_Ha, mask=(self.NII_Ha.mask | self.OIII_Hb.mask))

        # compute over SII_Ha
        SF_SII_Ha = ~self.Ke01_decision(self.SII_Ha, self.OIII_Hb,
                                        **self.Ke01_cs['SII_Ha'])
        AGN_SII_Ha = (self.Ke01_decision(self.SII_Ha, self.OIII_Hb,
                                        **self.Ke01_cs['SII_Ha'])) & \
                     (self.AGN_LIER_decision(self.SII_Ha, self.OIII_Hb,
                                             **self.Sf_L_cs['SII_Ha']))
        LIER_SII_Ha = ~(SF_SII_Ha | AGN_SII_Ha)
        class_SII_Ha = np.argmax(
            np.stack((np.zeros_like(SF_SII_Ha), SF_SII_Ha,
                       np.zeros_like(SF_SII_Ha), AGN_SII_Ha,
                      LIER_SII_Ha)), axis=0)
        self.class_SII_Ha = np.ma.array(
            class_SII_Ha, mask=self.SII_Ha.mask)

        # compute over OI_Ha
        SF_OI_Ha = (~self.Ke01_decision(self.OI_Ha, self.OIII_Hb,
                                        **self.Ke01_cs['OI_Ha'])) & \
                   (self.OI_Ha < -.75)
        AGN_OI_Ha = (self.Ke01_decision(self.OI_Ha, self.OIII_Hb,
                                       **self.Ke01_cs['OI_Ha'])) & \
                    (self.AGN_LIER_decision(self.OI_Ha, self.OIII_Hb,
                                            **self.Sf_L_cs['OI_Ha']))
        LIER_OI_Ha = ~(SF_OI_Ha | AGN_OI_Ha)
        class_OI_Ha = np.argmax(
            np.stack((np.zeros_like(SF_OI_Ha), SF_OI_Ha,
                      np.zeros_like(SF_OI_Ha), AGN_OI_Ha,
                      LIER_OI_Ha)), axis=0)
        self.class_OI_Ha = np.ma.array(
            class_OI_Ha, mask=self.OI_Ha.mask)

        ## decide where everything agrees

        self.SF = np.ma.array(
            SF_NII_Ha * SF_SII_Ha * SF_OI_Ha,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        self.comp = np.ma.array(
            comp_NII_Ha * SF_SII_Ha * SF_OI_Ha,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        self.AGN = np.ma.array(
            nuc_NII_Ha * AGN_SII_Ha * AGN_OI_Ha,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        self.LIER = np.ma.array(
            nuc_NII_Ha * AGN_SII_Ha * AGN_OI_Ha,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        stack_class = np.stack((self.SF, self.comp, self.AGN, self.LIER))

        self.ambig = ~(self.SF | self.comp | self.AGN | self.LIER)

        self.diag = np.argmax(
            np.stack(
                (self.ambig, self.SF, self.comp, self.AGN, self.LIER)),
            axis=0)

        self.diag = np.ma.array(
            self.diag, mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                             self.OI_Ha.mask | self.OIII_Hb.mask))

        # define color map
        self.cmap = colors.ListedColormap(
            ['gray', 'blue', 'green', 'red', 'orange'])
        self.cmap.set_bad('w')
        self.cmap_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        self.norm = colors.BoundaryNorm(self.cmap_bounds, self.cmap.N)

    def Ka03_(self, x, u, v, w, l):
        return u/(x + v) + w

    def Ke01_(self, x, u, v, w, l):
        return u/(x + v) + w

    def AGN_LIER_(self, x, m, b):
        return m*x + b

    def Ka03_decision(self, x, y, u, v, w, l):
        '''
        y is [OIII]/Hb
        x is [NII]/Ha, [SII]/Ha, or [OI]/Ha
        '''
        return (y > u/(x + v) + w) | (x > l)

    def Ke01_decision(self, x, y, u, v, w, l):
        '''
        y is [OIII]/Hb
        x is [NII]/Ha, [SII]/Ha, or [OI]/Ha
        '''
        return (y > u/(x + v) + w) | (x > l)

    def AGN_LIER_decision(self, x, y, m, b):
        '''
        y is [OIII]/Hb
        x is [NII]/Ha, [SII]/Ha, or [OI]/Ha
        '''
        return y > m * x + b

    def map_plot_ind(self, h, dep, objname, save=False, loc=''):
        '''
        make BPT plots with each of the schemes,
        accompanied by a map in each of the schemes
        '''
        qtype = self.qtype

        # should mask low-SNR spaxels, but don't want to do that
        # without thinking about how (need to get it from the DAP obj)
        plt.close('all')

        fig = plt.figure(figsize=(9, 6), dpi=300)

        # scatter plots on top
        NII_ax = plt.subplot(231)
        SII_ax = plt.subplot(232)
        OI_ax = plt.subplot(233)

        # colors and labels
        cs = ['gray', 'blue', 'green', 'red', 'orange']
        ls = ['ambig.', 'SF', 'comp.', 'AGN/Nuc.', 'LIER']

        # same concept as above
        for i in range(1, 5):
            NII_ax.scatter(
                self.NII_Ha[self.class_NII_Ha == i],
                self.OIII_Hb[self.class_NII_Ha == i],
                edgecolor='None', facecolor=cs[i], label = ls[i],
                marker='.', alpha=0.5)
            SII_ax.scatter(
                self.SII_Ha[self.class_SII_Ha == i],
                self.OIII_Hb[self.class_SII_Ha == i],
                edgecolor='None', facecolor=cs[i], label = ls[i],
                marker='.', alpha=0.5)
            OI_ax.scatter(
                self.OI_Ha[self.class_OI_Ha == i],
                self.OIII_Hb[self.class_OI_Ha == i],
                edgecolor='None', facecolor=cs[i], label = ls[i],
                marker='.', alpha=0.5)

        # set up convenience grids
        NII_Ha_grid = np.linspace(-2., 1., 200)
        SII_Ha_grid = np.linspace(-1.25, 0.75, 200)
        OI_Ha_grid = np.linspace(-2.25, 0., 200)

        # plug grids in to get Ke01 lines
        Ke01_line_NII = self.Ke01_(
            NII_Ha_grid, **self.Ke01_cs['NII_Ha'])
        Ke01_line_SII = self.Ke01_(
            SII_Ha_grid, **self.Ke01_cs['SII_Ha'])
        Ke01_line_OI = self.Ke01_(
            OI_Ha_grid, **self.Ke01_cs['OI_Ha'])

        # plot the Ke01 (extreme starburst) line on all three axes
        NII_ax.plot(
            NII_Ha_grid[NII_Ha_grid < 0.4],
            Ke01_line_NII[NII_Ha_grid < 0.4],
            linestyle='-', c='k',
            marker='')
        SII_ax.plot(
            SII_Ha_grid[SII_Ha_grid < 0.1],
            Ke01_line_SII[SII_Ha_grid < 0.1],
            linestyle='-', c='k', marker='')
        OI_ax.plot(
            OI_Ha_grid[OI_Ha_grid < -.7],
            Ke01_line_OI[OI_Ha_grid < -.7],
            linestyle='-', c='k', label='Extr. S-B \nlim. (Ke01)',
            marker='')

        # get Ka03 line for NII
        Ka03_line_NII = self.Ka03_(
            NII_Ha_grid, **self.Ka03_cs['NII_Ha'])

        # plot the Ka03 (pure SF) line on NII axes where it's less than Ka01
        NII_ax.plot(
            NII_Ha_grid[(Ka03_line_NII < Ke01_line_NII) * \
                             (NII_Ha_grid < 0.4)],
            Ka03_line_NII[(Ka03_line_NII < Ke01_line_NII) * \
                          (NII_Ha_grid < 0.4)],
            linestyle='--', c='k', marker='')

        # dummy Ka03 plot on OI, for legend
        OI_ax.plot(
            [-3., -4.], [-2., -3.],
            linestyle='--', c='k', marker='',
            label='Pure SF \nlim. (Ka03)')

        # get the AGN-LIER line for SII and OI
        Sf_L_line_SII = self.AGN_LIER_(SII_Ha_grid, **self.Sf_L_cs['SII_Ha'])
        Sf_L_line_OI = self.AGN_LIER_(OI_Ha_grid, **self.Sf_L_cs['OI_Ha'])

        # plot the AGN-LIER line on SII & OI axes where it's more than Ka01\
        SII_ax.plot(
            SII_Ha_grid[(Sf_L_line_SII > Ke01_line_SII) |
                             (SII_Ha_grid > 0.1)],
            Sf_L_line_SII[(Sf_L_line_SII > Ke01_line_SII) |
                          (SII_Ha_grid > 0.1)],
            linestyle='-.', c='k', marker='')

        OI_ax.plot(
            OI_Ha_grid[(Sf_L_line_OI > Ke01_line_OI) |
                            (OI_Ha_grid > -0.9)],
            Sf_L_line_OI[(Sf_L_line_OI > Ke01_line_OI) |
                         (OI_Ha_grid > -0.9)],
            linestyle='-.', c='k', marker='', label='AGN-LIER')

        # fix axes limits and scales
        NII_ax.set_ylim([-1.25, 1.5])
        SII_ax.set_ylim(NII_ax.get_ylim())
        OI_ax.set_ylim(NII_ax.get_ylim())
        SII_ax.set_yticklabels([])
        OI_ax.set_yticklabels([])
        NII_ax.set_xlim([-2., 1.])
        SII_ax.set_xlim([-1.25, 0.75])
        OI_ax.set_xlim([-2.25, 0.])

        # set up legend for OI axis, and move it to the empty space to right
        OI_ax.legend(loc='lower center', prop={'size': 10},
                     ncol=1, bbox_to_anchor=(1.4, 0.2, 0.4, 1.))

        # make scatter plot axes labels
        NII_ax.set_ylabel(
            r'$\log{\frac{\mathrm{%s(OIII)}}{\mathrm{%s(H\beta)}}}$' \
                % (qtype, qtype), size=14)
        NII_ax.set_xlabel(
            r'$\log{\frac{\mathrm{%s(NII)}}{\mathrm{%s(H\alpha)}}}$' \
                % (qtype, qtype), size=14)
        SII_ax.set_xlabel(
            r'$\log{\frac{\mathrm{%s(SII)}}{\mathrm{%s(H\alpha)}}}$' \
                % (qtype, qtype), size=14)
        OI_ax.set_xlabel(
            r'$\log{\frac{\mathrm{%s(OI)}}{\mathrm{%s(H\alpha)}}}$' \
                % (qtype, qtype), size=14)

        NII_ax.tick_params(axis='both', labelsize=10)
        SII_ax.tick_params(axis='both', labelsize=10)
        OI_ax.tick_params(axis='both', labelsize=10)

        # BPT maps on bottom
        NII_Ha_map = pywcsgrid2.subplot(234, header=h)
        SII_Ha_map = pywcsgrid2.subplot(235, header=h)
        OI_Ha_map = pywcsgrid2.subplot(236, header=h)

        for ax, t in zip(
            [NII_Ha_map, SII_Ha_map, OI_Ha_map],
            ['NII-Ha', 'SII-Ha', 'OI-Ha']):
            ax.set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.diag.shape))
            ax.axis['bottom'].major_ticklabels.set(fontsize=10)
            ax.axis['left'].major_ticklabels.set(fontsize=10)
            # add beam
            bs = 2.*u.arcsec/(h['CDELT1'] * h['PC1_1'] * u.deg)
            ax.add_beam_size(bs, bs, 0., loc=1)
            ax.tick_params(axis='both', colors='w')
            ax.grid()
            ax.add_inner_title(t, loc=2, frameon=False)
            ax.yaxis.label.set_size(10.)
            ax.xaxis.label.set_size(10.)

            if dep is not None:
                ctr = ax.contour(
                    dep.d, levels=[.5, 1., 2., 3.], colors='k',
                    zorder=2)
                ax[dep.w].clabel(ctr, fmt=r'%1.1f $R_{eff}$', fontsize=6,
                                 c='k')

        # make BPT maps in individual schemes
        # for the time being, they're masked above 3 Re,
        # but in the long-run this should be done with SNR
        NII_Ha_map.imshow(
            np.ma.array(
                self.class_NII_Ha, mask=dep.d > 3.),
            origin='lower', aspect='equal',
            interpolation='None', cmap=self.cmap, norm=self.norm,
            vmin=0., vmax=4., zorder=1)
        SII_Ha_map.imshow(
            np.ma.array(
                self.class_SII_Ha, mask=dep.d > 3.),
            origin='lower', aspect='equal',
            interpolation='None', cmap=self.cmap, norm=self.norm,
            vmin=0., vmax=4., zorder=1)
        im = OI_Ha_map.imshow(
            np.ma.array(
                self.class_OI_Ha, mask=dep.d > 3.),
            origin='lower', aspect='equal',
            interpolation='None', cmap=self.cmap, norm=self.norm,
            vmin=0., vmax=4., zorder=1)
        SII_Ha_map.set_ylabel('')
        OI_Ha_map.set_ylabel('')

        # colorbar axis takes up the bottom right
        cb_ax = fig.add_axes([0.775, 0.1, 0.2, 0.025])
        cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
        cbar.set_ticks([0., 1., 2., 3., 4])
        cbar.set_ticklabels(['UND.', 'SF', 'Comp.', 'AGN/\nNuc.', 'LIER'])
        cbar.ax.tick_params(labelsize=9, length=0)

        plt.subplots_adjust(
            left=.1, right=.75, bottom=.05, top=.925,
            wspace=.15, hspace=.175)

        # set up field image
        f_im_ax = fig.add_axes([0.79, 0.25, 0.20, 0.30])
        s = .05
        f_im = gz2.download_sloan_im(
            h['CRVAL1'], h['CRVAL2'], scale=s,
            width=40./s, height=40./s, verbose=False,)
        f_im_ax.imshow(
            f_im[::-1], extent=[-20., 20., -20., 20.], aspect='equal')
        f_im_ax.tick_params(labelsize=9)
        f_im_ax.set_xticks([-10., 0., 10.])
        f_im_ax.set_yticks([-10., 0., 10.])
        fmt = '%.0f"'
        tks = mtick.FormatStrFormatter(fmt)
        f_im_ax.xaxis.set_major_formatter(tks)
        f_im_ax.yaxis.set_major_formatter(tks)
        f_im_ax.grid(color='gray')

        plt.suptitle(objname)

        if save == False:
            plt.show()
        else:
            plt.savefig('{}{}_bpt_map_plot_ind.png'.format(loc, objname))


ifu_dims = {127: 32., 91: 27., 61: 22., 37: 17., 19: 12., 7: 7.}

class deproject(object):
    def __init__(self, hdu, drpall_row, plot=False, verbose=False):
        '''
        given a fits header, construct an array of wcs coordinates
        then use a given phi & i to deproject
        '''

        objcoords = coords.SkyCoord(ra=drpall_row['objra']*u.deg,
                                    dec=drpall_row['objdec']*u.deg,
                                    frame='fk5')

        ifucoords = coords.SkyCoord(ra=drpall_row['ifura']*u.deg,
                                    dec=drpall_row['ifudec']*u.deg,
                                    frame='fk5')

        ifudesignsize = drpall_row['ifudesignsize']
        ifu_r = ifu_dims[ifudesignsize] / 3600. / 2.

        ba_min = .13

        incl = np.arccos(
            np.sqrt(
                (drpall_row['nsa_ba']**2. - ba_min**2.)/(1 - ba_min**2.)))
        incl *= 180./np.pi

        phi = drpall_row['nsa_phi']
        phi, incl = phi*np.pi/180., incl*np.pi/180.
        ba = drpall_row['nsa_ba']
        Re = drpall_row['nsa_petro_th50_el']
        self.Re = Re
        self.zdist = drpall_row['nsa_zdist']
        objra = objcoords.ra.deg
        objdec = objcoords.dec.deg
        self.plateifu = drpall_row['plateifu']
        if verbose == True:
            print drpall_row['plateifu']
            print '\t', drpall_row['ifucoords'].to_string()
            print '\t', 'phi:', phi, '\n\ti:', incl, '\n\tb/a:', ba

        ifura = ifucoords.ra.deg
        ifudec = ifucoords.dec.deg

        w = wcs.WCS(hdu.header).dropaxis(2)
        self.w = w
        XX, YY = np.meshgrid(
            np.arange(hdu.header['NAXIS1']),
            np.arange(hdu.header['NAXIS2']))
        im_coords = np.stack((XX.flatten(), YY.flatten())).T
        # transform image coordinates into WCS,
        # and then divide by an eff. rad.
        # finally rotate the reference frame to align with major axis
        world = w.wcs_pix2world(im_coords, 0)
        world = world.reshape(XX.shape[0], -1, 2)
        rot_m = np.array(
            [[np.cos(phi), -np.sin(phi)],
             [np.sin(phi), np.cos(phi)]])
        rot_m_r = np.linalg.pinv(rot_m)
        dfromc = ((world - np.array([objra, objdec]))*3600./Re)

        v = 5.*np.array([[0., 0.], [0., 1.]])/3600.
        maj_a = v.dot(rot_m_r)
        min_a = maj_a.dot(
            np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2)],
                      [np.sin(-np.pi/2), np.cos(-np.pi/2)]]))

        # make two displacement matrices (both in x and y)
        # first is displacement in direction of maj axis,
        # second is displacement in direction of minor axis
        # then multiply second by cos(incl) or something
        # sum, and add result in quadrature along axes
        #
        # see http://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/dotprod/dotprod.html
        # where `a` is maj ax or min ax vector
        # and `b` is a generic, single displacement vector.

        d1 = np.inner(maj_a[1], dfromc)/np.linalg.norm(maj_a[1])
        d1 = (d1/np.linalg.norm(maj_a[1]))[:, :, np.newaxis] * maj_a[1]
        d2 = np.inner(min_a[1], dfromc)/np.linalg.norm(min_a[1])
        d2 = (d2/np.linalg.norm(min_a[1]))[:, :, np.newaxis] * min_a[1]
        d2 /= (np.cos(incl))

        d = np.sqrt(((d1 + d2)**2.).sum(axis=-1))

        a2tv = np.arccos(
            np.inner(dfromc, maj_a[1])/ \
                (np.linalg.norm(maj_a[1]) * np.linalg.norm(dfromc, axis=-1)))

        self.ifura, self.ifudec = ifura, ifudec
        self.ifu_r = ifu_r
        self.objra, self.objdec = objra, objdec
        self.world = world
        self.d = d
        self.ba = ba
        self.maj_a, self.min_a = maj_a, min_a
        self.v = v
        self.incl = incl
        self.phi = phi

    def __repr__(self):
        return 'MaNGA DAP deproject object @ {} w/ i = {}, phi = {}'.format(
            (self.objra, self.objdec), self.incl, self.phi)

    def plot(self, save=True, loc=''):
        plt.close('all')
        fig = plt.figure(figsize=(5, 4), dpi=300)

        ax = plt.subplot(111)

        wn = hn = 2400. # can't go any higher
        s = 0.02
        img = gz2.download_sloan_im(
            ra=self.ifura, dec=self.ifudec, scale=s,
            width=wn, height=hn, verbose=False)
        ax.imshow(
            img[::-1, :],
            extent=[self.ifura + wn*s/3600./2, self.ifura - wn*s/3600./2,
                    self.ifudec - hn*s/3600./2, self.ifudec + hn*s/3600./2],
            interpolation='None', origin='lower')
        ctr = ax.contour(
            self.world[:, :, 0], self.world[:, :, 1],
            self.d, levels=[0.5, 1., 2., 3.],
            vmin=0.1, vmax=3., cmap='cool_r')
        ax.clabel(ctr, fmt=r'%1.1f $R_{eff}$', fontsize=10)
        # add outline of IFU footprint
        ax.add_patch(
            patches.RegularPolygon(
                xy=(self.ifura, self.ifudec), numVertices=6,
                radius=self.ifu_r, orientation=np.pi/6.,
                edgecolor='purple', facecolor='None'))
        ax.plot(
            self.ifura + self.v[:, 0],
            self.ifudec + self.v[:, 1],
            c='b')
        ax.plot(
            self.ifura + self.maj_a[:, 0],
            self.ifudec + self.maj_a[:, 1],
            c='r')
        ax.plot(
            self.ifura + self.min_a[:, 0]*self.ba,
            self.ifudec + self.min_a[:, 1]*self.ba,
            c='c')
        ax.set_aspect('equal')
        plt.tight_layout()
        if save == True:
            plt.savefig('{}{}_deproject.png'.format(loc, self.plateifu))
        else:
            plt.show()

def Zdiag_map(hdulist, objname, diag, save=True, loc=''):
    # set up figure

    Z_cmap = copy.copy(plt.cm.cubehelix)
    Z_cmap.set_bad('gray')
    Z_cmap.set_under('k')
    Z_cmap.set_over('w')

    hdu = hdulist[diag]
    header = hdu.header
    data = hdu.data
    d14 = data[0]
    d50 = data[1]
    d86 = data[2]
    mask = data[3]

    med = np.ma.array(d50, mask=mask)
    h16_84_w = np.ma.array(0.5 * np.abs(d86 - d14), mask=mask)

    plt.close('all')
    fig = plt.figure(figsize=(7, 4), dpi=300)

    ax1 = pywcsgrid2.subplot(121, header=header)
    ax2 = pywcsgrid2.subplot(122, header=header)

    vmin = np.min(d14[~np.isnan(d14)])
    vmax = np.max(d86[~np.isnan(d86)])
    #print vmin, vmax

    Zmap = ax1.imshow(
        med, cmap=Z_cmap, vmin=vmin, vmax=vmax, aspect='equal')
    Zemap = ax2.imshow(
        h16_84_w, cmap=Z_cmap, vmin=0., vmax=np.max(h16_84_w), aspect='equal')

    ax1.set_ticklabel_type(
        'delta',
        center_pixel=tuple(t/2. for t in d14.shape))
    ax1.axis['bottom'].major_ticklabels.set(fontsize=10)
    ax1.axis['left'].major_ticklabels.set(fontsize=10)
    ax1.tick_params(axis='both', colors='w')
    ax1.grid()
    ax1.yaxis.label.set_size(10.)
    ax1.xaxis.label.set_size(10.)

    ax2.set_ticklabel_type(
        'delta',
        center_pixel=tuple(t/2. for t in d14.shape))
    ax2.axis['bottom'].major_ticklabels.set(fontsize=10)
    ax2.axis['left'].major_ticklabels.set(fontsize=10)
    ax2.tick_params(axis='both', colors='w')
    ax2.grid()
    ax2.yaxis.label.set_size(10.)
    ax2.xaxis.label.set_size(10.)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1125, right=0.95, top=0.925, bottom=.275)
    Zcax = fig.add_axes([.05, .1125, .425, .05])
    Zecax = fig.add_axes([.55, .1125, .425, .05])
    Zcax.tick_params(labelsize=8)
    Zecax.tick_params(labelsize=8)

    plt.colorbar(
        Zmap, cax=Zcax, orientation='horizontal', extend='both').set_label(
        label=r'$Z_{50}$', size=8)
    plt.colorbar(
        Zemap, cax=Zecax, orientation='horizontal', extend='both').set_label(
        label=r'$\frac{1}{2}(Z_{84}-Z_{16})$', size=8)

    plt.suptitle(r'{}: {}'.format(objname, diag.replace('_', '\_')))

    if save == True:
        plt.savefig('{}{}-Z-{}.png'.format(loc, objname, diag))
    else:
        plt.show()

class gas_surf_dens(object):
    '''
    estimate a galaxy's gas mass, optical depth, and DGR, per-spaxel
        based on Brinchmann+13 method

    Both EBV (B-V color excess) and OH (oxygen abundance) must be (4 x N x N),
        where N is the number of spaxels on each side of the MaNGA datacube.

    the 4th (N x N) slice of both EBV and OH is a mask. The result will
        combine the masks, and mask any spaxel where the
        quoted, median metallicity is < 8.6. The first three slices correspond
        to the 14th, 50th, and 86th percentile metallicities (resulting
        from pyMCZ simulations). Since all functions are monotonic,
        the resultant gas mass estimates should be identical percentiles

    results are somewhat dependent on your religious choice of
        solar (O/H) and Z. Default values are taken from
        Asplund 2009 (arXiv 0909:0948v1)
    '''
    def __init__(self, hdulist, objname, diag, R_V=3.1, OH_sol=8.69,
                 Z_sol=.0134, **kwargs):

        self.EBV = hdulist['E(B-V)'].data
        self.OH = hdulist[diag].data
        self.hdulist = hdulist
        OH = self.OH
        EBV = self.EBV
        self.diag = diag
        self.R_V = R_V
        self.objname = objname
        self.OH_sol = OH_sol
        self.OH_halfsol = self.OH_sol - 0.301
        self.Z_sol = Z_sol

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        if (OH.shape[0] != 4) or (len(OH.shape) != 3):
            raise ValueError('OH must have shape (4 x 4 x N)')
        if (EBV.shape[0] != 4) or (len(EBV.shape) != 3):
            raise ValueError('EBV must have shape (4 x 4 x N)')

        SIGMA_gas = np.empty_like(OH)
        Z = np.empty_like(OH)
        xi = np.empty_like(OH)
        tau_V = np.empty_like(OH)

        # define measured metallicity in terms of solar metallicity and
        # oxygen abundance, assuming a constant oxygen-to-other-things
        # scaling (O/Z)_sun = (O/Z)_universal
        Z[:3, :, :] = 10.**(OH[:3, :, :] - OH_sol) * Z_sol
        xi[:3, :, :] = 10.**(-4.45 + 0.43 * OH[:3, :, :])
        tau_V[:3, :, :] = R_V * EBV[:3, :, :] / 1.086
        SIGMA_gas[:3, :, :] = 0.2 * \
            (tau_V[:3, :, :] / (xi[:3, :, :] * Z[:3, :, :]))

        mask = ((EBV[3, :, :].astype(bool)) | (OH[3, :, :].astype(bool)))
        for a in [SIGMA_gas, xi, tau_V, Z]:
            a[3, :, :] = mask

        self.SIGMA_gas = SIGMA_gas
        self.Z = Z
        self.xi = xi
        self.tau_V = tau_V
        self.mask = mask

    def __repr__(self):
        return 'gas_surf_dens object ({}), {} good spaxels'.format(
            self.objname, (mask == 0).sum())

    def to_kpc(self, dep, re_min_max, ang_scale=0.5*u.arcsec):

        from astropy.cosmology import WMAP9 as cosmo

        # angular and physical (on-galaxy) spaxel sizes
        # [ang_scale] = arcsec
        phys_scale = ang_scale / cosmo.arcsec_per_kpc_proper(
            dep.zdist)

        # dep.Re is arcseconds per effective radius along major axis
        ang_min_max = (re_min_max * dep.Re) * u.arcsec
        phys_min_max = ang_min_max / cosmo.arcsec_per_kpc_proper(
            dep.zdist)
        return phys_min_max.to('kpc').value

    def make_fig(self, dep, save=True, loc=''):
        '''
        make map of gas mass surface density, and radial dependence thereof

        requires a deprojection object to be generated previously (can use
            header metadata from read in Zsample file--any hdu but 0)
        '''

        # set up colormap to use
        cmap = copy.copy(plt.cm.cubehelix_r)
        cmap.set_bad('gray', 0.)

        # size of qty array
        s = self.OH[1].shape

        diag = self.diag
        header = self.hdulist[diag].header
        plt.close('all')
        fig = plt.figure(figsize=(9, 8), dpi=300)

        # set up maps and plot axes
        gs = gridspec.GridSpec(
            4, 2, height_ratios=[2, .35, .2, 1.5], width_ratios=[1, 1])

        # axes for OH & xi map
        OH_map_ax = pywcsgrid2.subplot(gs[0, 0], header=header)
        OH_map_ax.add_inner_title('EL-derived abundance', loc=4,
                                  prop={'size': 8})
        OH_map_ax.add_inner_title('dust-to-metal mass ratio', loc=1,
                                  prop={'size': 8})
        # axes for SIGMA map
        S_map_ax = pywcsgrid2.subplot(gs[0, 1], header=header)
        S_map_ax.add_inner_title('Gas mass surface density', loc=1,
                                 prop={'size': 8})
        # axes for image of galaxy
        gal_im_ax = plt.subplot(gs[-1, 0])

        OH_map_cax = fig.add_axes([.05, .465, .4, .035])

        # fix maps axes
        for ax in [OH_map_ax, S_map_ax]:
            ax.set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.OH[1].shape))

            ax.axis['bottom'].major_ticklabels.set(fontsize=10)
            ax.axis['left'].major_ticklabels.set(fontsize=10)
            ax.tick_params(axis='both', colors='w')
            ax.grid()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_aspect('equal')
            ax.add_patch(patches.Rectangle(
                (-s[0], -s[1]), 2*s[0], 2*s[1],
                linewidth=0, fill=None, hatch=' / ', zorder=0))

        ## start working on Z/xi map axes
        #set effective lower limit at sol - 0.301 dex (50%)
        OH_med = np.ma.array(self.OH[1], mask=self.mask)
        OH_clims = np.array([self.OH_halfsol, 9.5])
        if OH_med.min() < OH_clims[0]:
            OH_clims[0] = OH_med.min()
        OH_map = OH_map_ax.imshow(
            OH_med, cmap=cmap,
            vmin=OH_clims[0], vmax=OH_clims[1])
        # OH colorbar
        OH_map_cb = plt.colorbar(OH_map, cax=OH_map_cax,
                                 orientation='horizontal')
        OH_map_cb.ax.tick_params(labelsize=12)
        OH_map_cbar_tick_locator = mtick.MaxNLocator(nbins=5)
        OH_map_cb.locator = OH_map_cbar_tick_locator
        OH_map_cb.update_ticks()
        OH_map_cb.set_label(
            label=r'$12 + \log{\frac{O}{H}}$',
            size=14)

        # xi colorbar
        xi_clims = -4.45 + 0.43 * OH_clims
        xi_map_cax = OH_map_cax.twiny()
        xi_map_cax.set_xlim(xi_clims)
        xi_map_cax.xaxis.set_tick_params(labelsize=12)
        xi_map_cbar_tick_locator = mtick.MultipleLocator(base=0.1)
        xi_map_cax.xaxis.set_major_locator(xi_map_cbar_tick_locator)
        xi_map_cax.set_xlabel(r'$\log{\xi}$', size=14)

        # axes for radial SIGMA
        S_R_ax = plt.subplot(gs[-2:, 1])
        S_R_ax.set_ylabel(
            r'$\log{\frac{\Sigma_{gas}}{\mathrm{M_{\odot}} ' + \
            '\mathrm{pc^{-2}}}}$')
        S_R_ax.set_xlabel(r'$\frac{R}{R_e}$')
        # specify plot bounds, not implemented until later
        Rmin, Rmax = -.05, np.max(np.ma.array(dep.d, mask=self.mask)) + .05
        logSmin, logSmax = -1., 2.75
        # minimum and maximum SIGMA values
        S_R_ax.set_ylim([logSmin, logSmax])
        S_R_ax.set_xlim([Rmin, Rmax])

        S_Rphys_ax = S_R_ax.twiny()
        S_Rphys_ax.set_xlim(self.to_kpc(dep, np.array([Rmin, Rmax])))
        S_Rphys_ax.set_xlabel(r'R [kpc]')

        r = np.ma.array(
            dep.d,
            mask=(self.mask | (OH_med < self.OH_halfsol)) ).flatten()

        ## plot radial abundance
        OH_R_ax = S_R_ax.twinx()
        OH = np.ma.array(self.OH[1, :, :], mask=self.mask).flatten()
        OH16 = np.ma.array(self.OH[0, :, :], mask=self.mask).flatten()
        OH84 = np.ma.array(self.OH[2, :, :], mask=self.mask).flatten()
        OH_e = np.ma.abs(np.ma.row_stack([OH16, OH84]) - OH)

        OH_R_ax.scatter(
            r, OH,
            marker='.', facecolor='g', edgecolor='None', alpha=0.8)
        OH_R_ax.set_ylim(OH_clims)
        OH_R_ax.set_ylabel(r'$12 + \log{\frac{O}{H}}$', color='g')
        OH_R_ax.tick_params(axis='y', colors='g')
        OH_R_ax.set_xlim([Rmin, Rmax])
        ## plot radial SIGMA
        S = np.ma.array(
            self.SIGMA_gas[1, :, :],
            mask=(self.mask | (OH_med < self.OH_halfsol)) ).flatten()
        S16 = np.ma.array(
            self.SIGMA_gas[0, :, :],
            mask=(self.mask | (OH_med < self.OH_halfsol)) ).flatten()
        S84 = np.ma.array(
            self.SIGMA_gas[2, :, :],
            mask=(self.mask | (OH_med < self.OH_halfsol)) ).flatten()
        Se = np.ma.abs(np.ma.row_stack([S16, S84]) - S)
        S_R_ax.scatter(
            r, np.log10(S),
            marker='.', facecolor='k', edgecolor='None', alpha=0.8)

        ## galaxy image
        imscale = np.abs(header['CDELT1'] * header['PC1_1'] * 3600.)
        wn = header['NAXIS1']
        hn = header['NAXIS2']
        f = 20
        im = gz2.download_sloan_im(
            ra=header['CRVAL1'], dec=header['CRVAL2'],
            scale=imscale/f, width=wn*f, height=hn*f, verbose=False)
        gal_im_ax.imshow(
            im[::-1, :, :],
            extent=[- wn*imscale/2., wn*imscale/2.,
                    - hn*imscale/2., hn*imscale/2.],
            aspect='equal', zorder=1)
        gal_im_ax.set_xticks([-10, 0, 10])
        gal_im_ax.set_yticks([-10, 0, 10])

        gal_im_ax.add_patch(
            patches.RegularPolygon(
                xy=(0, 0), numVertices=6,
                radius=dep.ifu_r*3600, orientation=np.pi/6.,
                edgecolor='purple', facecolor='None', zorder=3))

        ## SIGMA map axes
        S_map = S_map_ax.imshow(
            np.log10(
                np.ma.array(self.SIGMA_gas[1], mask=self.mask)),
            vmin=logSmin, vmax=logSmax)

        S_map_cax = fig.add_axes([.8875, .5175, .035, .45])
        S_map_cb = plt.colorbar(
            S_map, cax=S_map_cax,
            orientation='vertical', extend='both')
        S_map_cb.ax.tick_params(labelsize=12)
        S_map_cbar_tick_locator = mtick.MaxNLocator(nbins=5)
        S_map_cb.locator = S_map_cbar_tick_locator
        S_map_cb.update_ticks()
        S_map_cb.set_label(
            label=r'$\log{\frac{\Sigma_{gas}}{\mathrm{M_{\odot}} ' + \
                '\mathrm{pc^{-2}}}}$',
            size=12)

        #plt.tight_layout()
        plt.subplots_adjust(top=0.95, left=.1, hspace=.15, right=0.95)

        plt.suptitle('{} ({})'.format(self.objname, self.diag.replace(
            '_', '\_')))

        if save == True:
            plt.savefig('{}{}-SIGMA-{}.png'.format(
                loc, self.objname, self.diag))
        else:
            plt.show()

    def reddening(self, save=True, loc=''):

        # set up colormap to use
        cmap = copy.copy(plt.cm.cubehelix_r)
        cmap.set_bad('gray', 0.)

        # size of qty array
        s = self.OH[1].shape

        diag = self.diag
        header = self.hdulist[diag].header

        plt.close('all')

        fig = plt.figure(figsize=(4, 5))

        gs = gridspec.GridSpec(2, 1, height_ratios=[15, 1])
        ax = pywcsgrid2.subplot(gs[0, 0], header=header)
        ax.set_ticklabel_type(
            'delta',
            center_pixel=tuple(t/2. for t in self.OH[1].shape))

        ax.axis['bottom'].major_ticklabels.set(fontsize=10)
        ax.axis['left'].major_ticklabels.set(fontsize=10)
        ax.tick_params(axis='both', colors='w')
        ax.grid()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')
        ax.add_patch(patches.Rectangle(
            (-s[0], -s[1]), 2*s[0], 2*s[1],
            linewidth=0, fill=None, hatch=' / ', zorder=0))

        tau_map_cax = plt.subplot(gs[1, 0])

        tau_map = ax.imshow(
            np.ma.array(self.tau_V[1, :, :], mask=self.mask), vmin=0.,
            cmap=cmap)
        tau_map_cb = plt.colorbar(
            tau_map, orientation='horizontal', cax=tau_map_cax)
        tau_map_cb.ax.tick_params(labelsize=12)
        tau_map_cb_tick_locator = mtick.MaxNLocator(nbins=5)
        tau_map_cb.locator = tau_map_cb_tick_locator
        tau_map_cb.update_ticks()
        tau_map_cb.set_label(
            label=r'$\tau_{V}$', size=14)

        tau_clims = np.array([tau_map_cb.vmin, tau_map_cb.vmax])
        EBV_clims = tau_clims * 1.086 / self.R_V
        EBV_map_cax = tau_map_cax.twiny()
        EBV_map_cax.set_xlim(EBV_clims)
        EBV_map_cax.xaxis.set_tick_params(labelsize=12)
        EBV_map_cbar_tick_locator = mtick.MultipleLocator(base=0.25)
        EBV_map_cax.xaxis.set_major_locator(EBV_map_cbar_tick_locator)
        EBV_map_cax.set_xlabel(r'$E(B-V)$', size=12)

        ax.set_aspect('equal')
        plt.suptitle(r'{} Extinction'.format(self.objname))
        plt.subplots_adjust(
            hspace=.3, bottom=.1, top=.95, left=.1, right=.95)

        if save == True:
                plt.savefig('{}{}-reddening.png'.format(
                    loc, self.objname))
        else:
            plt.show()

    def SFR(self, Ha, drpall_row, plot=False, save=True, loc=''):
        '''
        estimate SFR from uncorrected Ha and dust extinction
        '''

        from astropy.cosmology import WMAP9 as cosmo

        Ha_mask = Ha.mask
        Ha = Ha.data
        # angular and physical (on-galaxy) spaxel sizes
        spaxel_asize = 0.25 * u.arcsec**2.
        spaxel_psize = spaxel_asize / (cosmo.arcsec_per_kpc_proper(
            drpall_row['nsa_zdist']))**2.

        Ha_f = Ha * np.exp(self.tau_V[1]) * u.Unit('1e-17 erg s^-1 cm^-2')
        Ha_f /= spaxel_psize
        dist = (drpall_row['nsa_zdist'] * c.c / cosmo.H(0)).to(u.cm)
        Ha_L = (4 * np.pi * Ha_f * dist**2.).to(u.Unit('erg s^-1 kpc^-2'))
        Ha_L_SFR_conv = 7.9e-42 * u.Unit('solMass yr^-1 s erg^-1')
        sfr = Ha_L_SFR_conv * Ha_L
        sfr, sfr_u = np.ma.array(
            sfr.value, mask=(Ha_mask | self.tau_V[-1].astype(bool))), \
            sfr.unit
        self.sfr, self.sfr_u = sfr, sfr_u

        if plot == True:
            plt.close('all')

            s = sfr.shape

            # set up colormap to use
            cmap = copy.copy(plt.cm.cubehelix_r)
            cmap.set_bad('gray', 0.)
            cmap.set_under('w')
            cmap.set_over('k')

            diag = self.diag
            header = self.hdulist[diag].header
            plt.close('all')
            fig = plt.figure(figsize=(6, 3), dpi=300)

            # axes for SFR map

            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])
            SFR_map_ax = pywcsgrid2.subplot(gs[0], header=header)
            SFR_map_ax.add_patch(patches.Rectangle(
                (-s[0], -s[1]), 2*s[0], 2*s[1],
                linewidth=0, fill=None, hatch=' / ', zorder=0))

            SFR_map_ax.set_ticklabel_type(
                'delta',
                center_pixel=tuple(t/2. for t in self.sfr.shape))
            SFR_map_ax.axis['bottom'].major_ticklabels.set(fontsize=10)
            SFR_map_ax.axis['left'].major_ticklabels.set(fontsize=10)
            SFR_map_ax.tick_params(axis='both', colors='w')
            SFR_map_ax.grid()
            SFR_map_ax.yaxis.label.set_size(10.)
            SFR_map_ax.xaxis.label.set_size(10.)
            SFR_map_ax.set_xlabel('')
            SFR_map_ax.set_ylabel('')

            vmin = -3 if (np.min(np.log10(sfr) < -3.)) else np.min(
                np.log10(sfr))
            el = True if (np.min(np.log10(sfr) < -3.)) else False
            vmax = -3 if (np.max(np.log10(sfr) > 3.)) else np.max(
                np.log10(sfr))
            eu = True if (np.max(np.log10(sfr) > 3.)) else False

            if el and eu:
                extend = 'both'
            elif el:
                extend = 'lower'
            elif eu:
                extend = 'upper'
            else:
                extend = 'neither'

            SFR_map = SFR_map_ax.imshow(
                np.log10(sfr), cmap=cmap, aspect='equal',
                vmin=vmin, vmax=vmax)
            SFR_cb = plt.colorbar(
                SFR_map, orientation='vertical', extend=extend)
            SFR_cb.set_label(
                r'$\log{\Sigma_{SFR}}$' + \
                r'[{}]'.format(sfr_u.to_string('latex')),
                fontsize=8)
            SFR_cb.ax.tick_params(labelsize=8)

            # axes for SFR-SIGMA plot
            SFR_SIGMA_ax = plt.subplot(gs[1])
            S_masked = np.ma.array(
                self.SIGMA_gas[1], mask=self.SIGMA_gas[-1])
            data  = np.row_stack(
                [self.SIGMA_gas[1].flatten(), self.sfr.flatten()])
            mask = np.empty_like(data[1], dtype=bool)
            mask = ((self.SIGMA_gas[-1].flatten().astype(bool)) | \
                          (data[0] < 10**-0.5) | (data[0] > 10.**5.) | \
                          (data[1] < 10.**-4) | (data[1] > 10.**3.))
            S_masked = data[0][~mask]
            sfr_masked = data[1][~mask]

            SFR_SIGMA_ax.set_xlabel(
                r'$\log{\Sigma_{gas}} [\frac{M_{\odot}}{\mathrm{pc}^{2}}]$',
                size=10)
            SFR_SIGMA_ax.set_ylabel(r'$\log{\Sigma_{SFR}}$ ' + \
                r'[{}]'.format(
                self.sfr_u.to_string('latex')),
                size=10)

            SFR_SIGMA_ax.scatter(
                np.log10(S_masked),
                np.log10(sfr_masked), marker='.',
                facecolor='k', edgecolor='None', label='spaxels',
                alpha=0.4)
            SFR_SIGMA_ax.tick_params(axis='both', labelsize=8.)
            xll = np.log10(S_masked).min() - .05
            if xll < -0.5: xll = -0.5
            xul = np.log10(S_masked).max() + .05
            if xul > 5: xll = 5
            yll = np.log10(sfr_masked).min() - .05
            if yll < -4: yll = -4
            yul = np.log10(sfr_masked).max() + .05
            if yul > 3: yul = 3

            SFR_SIGMA_ax.set_xlim([xll,xul])
            SFR_SIGMA_ax.set_ylim([yll,yul])

            plt.tight_layout()
            plt.suptitle(r'{} H$\alpha$ SFR'.format(self.objname))
            plt.subplots_adjust(top=0.9)

            if save == True:
                plt.savefig('{}{}-Ha_SFR.png'.format(
                    loc, self.objname))
            else:
                plt.show()

        return sfr, sfr_u

def radial_gradient(x, y, yuerr, ylerr, regr='linear',
                    corr='squared_exponential'):

    from sklearn import gaussian_process as gp

    X = np.atleast_2d(x.compressed()).T
    nugget = np.average(
        np.row_stack(
            [ylerr.compressed(), yuerr.compressed()]), axis=0)
    nugget = (nugget / y.compressed())**2.
    GP = gp.GaussianProcess(
        regr=regr, corr=corr, nugget=nugget)
    GP.fit(X, y.compressed())

    return GP
