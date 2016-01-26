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
from glob import glob
from scipy.interpolate import interp1d
from pysynphot import observation, spectrum
from matplotlib import gridspec, colors
import pywcsgrid2
import itertools

drpall_loc = '/home/zpace/Documents/MaNGA_science/'
dap_loc = '/home/zpace/mangadap/default/'
pw_loc = drpall_loc + '.saspwd'

uwdata_loc = '/d/www/karben4/'

MPL_versions = {'MPL-3': 'v1_3_3', 'MPL-4': 'v1_5_1'}

base_url = 'dtn01.sdss.org/sas/'
mangaspec_base_url = base_url + 'mangawork/manga/spectro/redux/'

c = 299792.458 #km/s
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


def conroy_to_table(fname):
    '''
    translate a Conroy-style SSP (at one metallicity)
        to a bunch of fits tables

    Conroy format

    <HEADER LINES> (beginning with #)
    <NAXIS1> (number of spectral bins) <NAXIS2> (number of age bins)


    '''

    names = ['logT', 'logM', 'logLbol', 'logSFR', 'spectra']

    data = []

    Z = None

    with open(fname) as f:
        for i, l in enumerate(f):
            if (l[0] == '#') and (i != 0):
                pass
            elif i == 0:
                for t in l.split():
                    try:
                        Z = float(t)
                        break
                    except ValueError:
                        pass
            else:
                data.append(
                    [float(j) for j in l.rstrip('\n').lstrip(' ').split()])

    if Z is None:
        print 'NO METALLICITY PARSED for file \n{}'.format(fname)
    else:
        print 'log Z/Zsol found: {}'.format(Z)

    nT, nL = data.pop(0)  # number of age bins and number of wavelengths
    nT, nL = int(nT), int(nL)
    l = table.Column(data=data.pop(0) * u.AA,
                     name='lambda')  # wavelengths

    assert len(l) == nL, 'There should be {} elements in nL, \
        but there are {}'.format(len(l), len(nL))

    # now restrict the wavelength range to a usable interval
    lgood = ((1500. * u.AA <= l) * (l <= 1.1 * u.micron))
    l = l[lgood]

    # log-transform
    logl = np.log(l / u.Angstrom)

    dlogl = np.mean(logl[1:] - logl[:-1])
    CRVAL1 = logl[0]
    CDELT1 = dlogl
    NAXIS1 = len(logl)
    # print CRVAL1, CDELT1, NAXIS1

    # spectra are every second row
    spectra_fnu = data[1::2] * u.solLum/u.Hz
    # use only good wavelength range
    spectra_fnu = [row[lgood] for row in spectra_fnu]
    metadata = data[::2]

    age = table.Column([10.**i[0] / 10**9 for i in metadata] * u.Gyr,
                       name='age')
    mass = table.Column([10.**i[1] for i in metadata] * u.solMass,
                        name='orig SSP mass')
    lbol = table.Column([10.**i[2] for i in metadata] * u.solLum,
                        name='lbol')
    SFR = table.Column([10.**i[3] for i in metadata] * u.solMass/u.year,
                       name='SFR')

    '''
    # this is how you convert one row of spectra_fnu into f_lambda units
    f_nu = spectra_fnu[0]
    f_lambda = (f_nu/u.cm**2.).to(
        u.erg/u.s/u.cm**2./u.AA,
        equivalencies=u.spectral_density(l)) * u.cm**2.
    '''

    # do the same thing as above, except normalize by SSP mass
    spectra_fl_m = [(f_nu/u.cm**2.).to(
        u.erg/u.s/u.cm**2./u.AA,
        equivalencies=u.spectral_density(l)) * u.cm**2. / (m*u.solMass)
        for f_nu, m in zip(spectra_fnu, mass)]

    SSPs = table.Table(data=[age, mass, lbol, SFR])
    SSPs.add_column(table.Column(np.ones_like(np.asarray(SFR)) * u.solMass,
                                 name='new SSP mass'))  # all 1
    SSPs.add_column(table.Column(data=spectra_fl_m * spectra_fl_m[0].unit,
                                 name='spectrum'))
    SSPs.add_column(table.Column(data=Z * np.ones_like(np.asarray(SFR)),
                                 name='Z'))

    return SSPs, l


def make_conroy_file(loc, plot=False, Zll=.05, Zul=99.,
                     Tll=.0008, Tul=13.5,
                     Lll=3250., Lul=15000.):
    fnames = glob(loc + '*.out.spec')

    print '{} metallicities spotted'.format(len(fnames))

    SSPs = table.vstack([conroy_to_table(f)[0]
                         for f in fnames])

    Zconds = (Zll <= 10**SSPs['Z']) * (10**SSPs['Z'] <= Zul)
    Tconds = (Tll <= SSPs['age']) * (SSPs['age'] <= Tul)

    SSPs = SSPs[Zconds * Tconds]

    print 'SSPs read'

    SSPs.sort(['age', 'Z'])

    Ts = np.unique(SSPs['age'])
    Ts.sort()
    Zs = np.unique(SSPs['Z'])
    Zs.sort()
    # retrieve the array of wavelengths by calling
    # conroy_to_table one last time
    Ls = conroy_to_table(fnames[0])[1]
    Lconds = (Lll <= Ls) * (Ls <= Lul)
    Ls = Ls[Lconds]

    nT, nZ, nL = len(Ts), len(Zs), len(Ls)

    if plot == True:
        plt.close('all')

        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.semilogy(Ts)
        ax2.plot(Zs)
        ax3.semilogy(Ls)
        plt.tight_layout()
        plt.show()

    # initialize an array of dimension [nT, nZ, nL]
    SSPs_cube = np.empty([nT, nZ, nL])

    # T, Z, and L are evenly spaced in log (Z is already there)

    logT = np.log(Ts)
    logL = np.log(Ls)

    # set up header keywords to make reading in & writing out easier

    NAXIS = 3

    # define all the header keywords you'll need for the fits file

    h = {'CTYPE3': 'ln age/Gyr',
         'CRVAL3': np.min(logT),
         'NAXIS3': nT,
         'CDELT3': np.abs(np.mean(logT[:-1] - logT[1:])),
         'CTYPE2': 'log10 Z/Zsol',
         'CRVAL2': np.min(Zs),
         'NAXIS2': nZ,
         'CDELT2': np.abs(np.mean(Zs[:-1] - Zs[1:])),
         'CTYPE1': 'ln lambda/AA',
         'CRVAL1': np.min(logL),
         'NAXIS1': nL,
         'CDELT1': np.abs(np.mean(logL[:-1] - logL[1:])),
         'BUNIT': SSPs['spectrum'].unit.to_string(),
         'NAXIS': NAXIS}

    for i in range(len(logT)):  # iterate over ages
        for j in range(len(Zs)):  # iterate over metallicities
            # print 'Z = {}, T = {}'.format(Zs[j], Ts[i])
            Zcond = (SSPs['Z'] == Zs[j])
            Tcond = (SSPs['age'] == Ts[i])
            spectrum = SSPs[Zcond * Tcond]['spectrum']
            SSPs_cube[i, j, :] = spectrum[0][Lconds]

    h['BSCALE'] = np.median(SSPs_cube)
    SSPs_cube /= h['BSCALE']
    print h

    print 'SSP cube constructed'
    hdu = fits.PrimaryHDU(SSPs_cube)
    for key, value in zip(h.keys(), h.values()):
        hdu.header[key] = value
    hdu.writeto('conroy_SSPs.fits', clobber=True)
    print 'FITS file written'

    # return SSPs, SSPs_cube


def models(fname):
    ssps = fits.open(fname)
    logL = ssps[0].header['CRVAL1'] + np.linspace(
        0., ssps[0].header['CDELT1'] * (ssps[0].header['NAXIS1'] - 1),
        ssps[0].header['NAXIS1'])

    return logL, ssps[0].data


def ssp_rebin(logL_ssp, spec_ssp, dlogL_new, Lll=3250.):
    '''
    rebin a GRID of model spectrum to have an identical velocity
    resolution to an input spectrum

    intended to be used on a grid of models with wavelength varying
    along final axis (in 3d array)

    DEPENDS ON pysynphot, which may not be an awesome thing, but
        it definitely preserves spectral integrity, and does not suffer
        from drawbacks of interpolation (failing to average line profiles)
    '''
    dlogL_ssp = np.median(logL_ssp[1:] - logL_ssp[:-1])
    f = dlogL_ssp/dlogL_new

    # print 'zoom factor: {}'.format(f)
    # print 'new array should have length {}'.format(logL_ssp.shape[0]*f)

    # print spec_ssp.shape

    # we want to only sample where we're sure we have data
    CRVAL1_new = logL_ssp[0] - 0.5*dlogL_ssp + 0.5*dlogL_new
    CRSTOP_new = logL_ssp[-1] + 0.5*dlogL_ssp - 0.5*dlogL_new
    NAXIS1_new = int((CRSTOP_new - CRVAL1_new) / dlogL_new)
    # start at exp(CRVAL1_new) AA, and take samples every exp(dlogL_new) AA
    logL_ssp_new = CRVAL1_new + \
        np.linspace(0., dlogL_new*(NAXIS1_new - 1), NAXIS1_new)
    L_new = np.exp(logL_ssp_new)
    L_ssp = np.exp(logL_ssp)

    # now find the desired new wavelengths

    spec = spectrum.ArraySourceSpectrum(wave=L_ssp, flux=spec_ssp)
    f = np.ones_like(L_ssp)
    filt = spectrum.ArraySpectralElement(wave=L_ssp, throughput=f,
                                         waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=L_new,
                                  force='taper')
    spec_ssp_new = obs.binflux

    # the following are previous attempts to do this rebinning

    '''
    # first, interpolate to a constant multiple of the desired resolution
    r_interm = int(1./f)
    print r_interm
    dlogL_interm = f * dlogL_new
    print dlogL_interm

    CDELT1_interm = dlogL_interm
    CRVAL1_interm = logL_ssp[0] + 0.5*CDELT1_interm
    CRSTOP_interm = logL_ssp[-1] - 0.5*CDELT1_interm
    NAXIS1_interm = int((CRSTOP_interm - CRVAL1_interm) / CDELT1_interm)
    logL_ssp_interm = CRVAL1_interm + np.linspace(
        0., CDELT1_interm * (NAXIS1_interm - 1), NAXIS1_interm)
    edges_interm = np.column_stack((logL_ssp_interm - 0.5*CDELT1_interm,
                                    logL_ssp_interm + 0.5*CDELT1_interm))

    spec_interp = interp1d(logL_ssp, spec_ssp)
    spec_interm = spec_interp(logL_ssp_interm)
    print spec_interm.shape

    spec_ssp_new = zoom(spec_interm, zoom=[1., 1., 1./r_interm])[1:-1]
    logL_ssp_new = zoom(logL_ssp_interm, zoom=1./r_interm)[1:-1]
    print logL_ssp_new.shape'''

    '''s = np.cumsum(spec_ssp, axis=-1)
    # interpolate cumulative array
    s_interpolator = interp1d(x=logL_ssp, y=s, kind='linear')
    s_interpolated_l = s_interpolator(edges[:, 0])
    s_interpolated_u = s_interpolator(edges[:, 1])
    total_in_bin = np.diff(
        np.row_stack((s_interpolated_l, s_interpolated_u)), n=1, axis=0)
    spec_ssp_new = total_in_bin * (dlogL_new/dlogL_ssp)'''

    return spec_ssp_new, logL_ssp_new

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
    def __init__(self, hdu, q='EW', sn_t=2.):
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

        self.ivar_maps = {k: np.array(hdu[ivar_extension].data[v, :, :])
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

        # mask bad data and data where SNR < sn_t
        self.qty_maps = {k: np.ma.array(
            self.qty_maps[k], mask=(self.mask_maps[k]))
            for (k, v) in self.emline.iteritems()}

        self.eline_hdr = hdu[self.q].header

    def map(self, save=False, objname=None):
        # make quantity and SNR maps for each species
        # mostly for QA purposes

        import mpl_toolkits.axes_grid1.axes_grid as axes_grid
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cmap1 = plt.cm.cubehelix
        cmap1.set_bad('gray')

        cmap2 = plt.cm.Purples_r
        cmap2.set_bad('gray')

        vr = {'GFLUX': [1., 20.], 'EW': [1., 200.], 'SFLUX': [1., 20.]}

        n_species = len(self.emline) # number of rows of subplots
        n_cols = 2 # col 1 for qty, col 2 for SNR
        fig_dims = (2.*n_cols + 1., 2.*n_species)

        fig = plt.figure(figsize=fig_dims, dpi=300)
        gh = pywcsgrid2.GridHelper(wcs=self.eline_hdr)
        g = axes_grid.ImageGrid(fig, 111,
                                nrows_ncols=(n_species, n_cols),
                                ngrids=None, direction='row', axes_pad=.02,
                                add_all=True, share_all=True,
                                aspect=True, label_mode='L', cbar_mode=None,
                                axes_class=(pywcsgrid2.Axes,
                                            dict(grid_helper=gh)))

        for i, k in enumerate(self.emline.keys()):
            qpn = 2*i # quantity subplot number
            spn = 2*i + 1 # SNR subplot number
            q_im = g[qpn].imshow(self.qty_maps[k], norm=colors.LogNorm(),
                                 interpolation=None, origin='lower',
                                 vmin=vr[self.qtype][0],
                                 vmax=vr[self.qtype][1], cmap=cmap2)
            s_im = g[spn].imshow(self.SNR_maps[k], norm=colors.LogNorm(),
                                 vmin=0.5, vmax=20., interpolation=None,
                                 origin='lower', cmap=cmap1)

            s_c = g[qpn].contour(self.qty_maps[k], levels=[self.sn_t],
                                 colors='r', linewidths=1.,
                                 linestyles='-')

            g[spn].contour(self.qty_maps[k], levels=[self.sn_t],
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

        qcb_ax = fig.add_axes([0.15, 0.935, 0.7, 0.0175])
        scb_ax = fig.add_axes([0.15, 0.035, 0.7, 0.0175])

        scb = fig.colorbar(s_im, cax=scb_ax, orientation='horizontal')
        scb.set_label('S/N')
        scb.add_lines(s_c)
        scb.set_ticks([1., 2., 5., 10.])

        qcb = fig.colorbar(q_im, cax=qcb_ax, orientation='horizontal')
        q = self.hdu[self.q].header['BUNIT'].replace('^2', '$^2$')
        qcb.set_label(r'{} [{}]'.format(self.qtype, q))

        plt.suptitle(objname)

        if save == False:
            plt.show()
        else:
            plt.savefig('eline_map.png')

        plt.close()

    def to_BPT(self):
        # convenience method: returns a dict of lines, that you can
        # double-splat into BPT.__init__() below
        ldata = {'Ha': self.qty_maps['Ha-----6564'],
                 'Hb': self.qty_maps['Hb-----4862'],
                 'OIII': self.qty_maps['OIII---4960'] + \
                    self.qty_maps['OIII---5008'],
                 'NII': self.qty_maps['NII----6549'] + \
                    self.qty_maps['NII----6585'],
                 'SII': self.qty_maps['SII----6732'] + \
                    self.qty_maps['SII----6718'],
                 'OI': self.qty_maps['OI-----6302'] + \
                    self.qty_maps['OI-----6365']}

        return ldata


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

    def __init__(self, Ha, Hb, OIII, NII, SII, OI):

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

        '''

        self.Ka03_cs = {'NII_Ha': {'u': 0.61, 'v': -0.05, 'w': 1.3, 'l': -.1}}
        self.Ke01_cs = {'SII_Ha': {'u': 0.72, 'v': -0.32, 'w': 1.3, 'l': .1},
                        'OI_Ha': {'u': 0.73, 'v': 0.59, 'w':1.33, 'l': .3},
                        'NII_Ha': {'u': 0.61, 'v': -0.47, 'w': 1.19, 'l': 1.7}}
        self.Sf_L_cs = {'SII_Ha': {'m': 1.89, 'b': 0.76},
                        'OI_Ha': {'m': 1.18, 'b': 1.3}}

        self.SII_Ha = np.ma.array(np.log10(SII/Ha),
                                  mask=(Ha.mask | SII.mask))
        self.NII_Ha = np.ma.array(np.log10(NII/Ha),
                                  mask=(Ha.mask | NII.mask))
        self.OI_Ha = np.ma.array(np.log10(OI/Ha),
                                 mask=(Ha.mask | OI.mask))
        self.OIII_Hb = np.ma.array(np.log10(OIII/Hb),
                                   mask=(OIII.mask | Hb.mask))

        # product of conditions for SF classification
        SF = (~self.Ka03_decision(self.NII_Ha, self.OIII_Hb,
            **self.Ka03_cs['NII_Ha']))
        SF *= (~self.Ke01_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Ke01_cs['SII_Ha']))
        SF *= (~self.Ke01_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Ke01_cs['OI_Ha']))
        self.SF = np.ma.array(
            SF,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        # product of conditions for composite
        comp = (self.Ka03_decision(
            self.NII_Ha, self.OIII_Hb,
            **self.Ka03_cs['NII_Ha']))
        comp *= (~self.Ke01_decision(
            self.NII_Ha, self.OIII_Hb,
            **self.Ke01_cs['NII_Ha']))
        comp *= (~self.Ke01_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Ke01_cs['SII_Ha']))
        comp *= (~self.Ke01_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Ke01_cs['OI_Ha'])) & (self.OI_Ha < -.7)
        self.comp = np.ma.array(
            comp,
            mask=(self.NII_Ha.mask | self.OIII_Hb.mask))

        # product of conditions for AGN
        AGN = (self.Ke01_decision(
            self.NII_Ha, self.OIII_Hb,
            **self.Ke01_cs['NII_Ha']))
        AGN *= (self.Ke01_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Ke01_cs['SII_Ha']))
        AGN *= (self.Ke01_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Ke01_cs['OI_Ha']))
        AGN *= self.AGN_LIER_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Sf_L_cs['SII_Ha'])
        AGN *= self.AGN_LIER_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Sf_L_cs['OI_Ha'])
        self.AGN = np.ma.array(
            AGN,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        # product of conditions for LI(N)ERs
        LIER = (self.Ke01_decision(
            self.NII_Ha, self.OIII_Hb,
            **self.Ke01_cs['NII_Ha']))
        LIER *= (self.Ke01_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Ke01_cs['SII_Ha']))
        LIER *= (self.Ke01_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Ke01_cs['OI_Ha']))
        LIER *= ~self.AGN_LIER_decision(
            self.SII_Ha, self.OIII_Hb,
            **self.Sf_L_cs['SII_Ha'])
        LIER *= ~self.AGN_LIER_decision(
            self.OI_Ha, self.OIII_Hb,
            **self.Sf_L_cs['OI_Ha'])
        self.LIER = np.ma.array(
            LIER,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        # ambiguous galaxies are zero in all of SF, comp, AGN, and LIER
        stack_class = np.stack((self.SF, self.comp, self.AGN, self.LIER))
        self.ambig = np.logical_not(np.any(stack_class, axis=0))
        self.ambig = np.ma.array(
            self.ambig,
            mask=(self.SII_Ha.mask | self.NII_Ha.mask | \
                  self.OI_Ha.mask | self.OIII_Hb.mask))

        self.stack_class = np.stack(
            (self.ambig, self.SF, self.comp, self.AGN, self.LIER))
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

    def plot(self, save=False, objname=None):
        cs = ['gray', 'blue', 'green', 'red', 'orange']
        ls = ['ambig.', 'SF', 'comp.', 'AGN', 'LIER']
        fig = plt.figure(dpi=300, figsize=(10, 5))
        gs = gridspec.GridSpec(
            nrows=1, ncols=3, left=0.1, bottom=0.25, right=0.95, top=0.95,
            wspace=0.05, hspace=0., width_ratios=[1, 1, 1])

        NII_ax = plt.subplot(gs[0])
        SII_ax = plt.subplot(gs[1])
        OI_ax = plt.subplot(gs[2])

        # plot data on each set of axes
        for (ax, qty) in zip([NII_ax, SII_ax, OI_ax],
                             [self.NII_Ha, self.SII_Ha, self.OI_Ha]):
            for i in range(5):
                ax.scatter(
                    qty[self.diag == c], self.OIII_Hb[self.diag == c],
                    edgecolor='None', facecolor=cs[i], label = ls[i],
                    marker='.', alpha=0.5)

        # plot the Ke01 (extreme starburst) line on all three axes
        NII_Ha_grid = np.linspace(-2., 1., 200)
        SII_Ha_grid = np.linspace(-1.25, 0.75, 200)
        OI_Ha_grid = np.linspace(-2.25, 0., 200)

        Ke01_line_NII = self.Ke01_(
            NII_Ha_grid, **self.Ke01_cs['NII_Ha'])
        Ke01_line_SII = self.Ke01_(
            SII_Ha_grid, **self.Ke01_cs['SII_Ha'])
        Ke01_line_OI = self.Ke01_(
            OI_Ha_grid, **self.Ke01_cs['OI_Ha'])

        NII_ax.plot(
            NII_Ha_grid[NII_Ha_grid < 0.4],
            Ke01_line_NII[NII_Ha_grid < 0.4],
            linestyle='-', c='k',
            label='Extr. S-B lim. (Ke01)', marker='')
        SII_ax.plot(
            SII_Ha_grid[SII_Ha_grid < 0.1],
            Ke01_line_SII[SII_Ha_grid < 0.1],
            linestyle='-', c='k', marker='')
        OI_ax.plot(
            OI_Ha_grid[OI_Ha_grid < -.7],
            Ke01_line_OI[OI_Ha_grid < -.7],
            linestyle='-', c='k', marker='')

        # plot the Ka03 (pure SF) line on NII axes where it's less than Ka01
        Ka03_line_NII = self.Ka03_(
            NII_Ha_grid, **self.Ka03_cs['NII_Ha'])
        NII_ax.plot(
            NII_Ha_grid[(Ka03_line_NII < Ke01_line_NII) * \
                             (NII_Ha_grid < 0.4)],
            Ka03_line_NII[(Ka03_line_NII < Ke01_line_NII) * \
                          (NII_Ha_grid < 0.4)],
            linestyle='--', c='k', marker='',
            label='Pure SF lim. (Ka03)')

        # plot the AGN-LIER line on SII & OI axes where it's more than Ka01
        NII_ax.plot([-5., -10.], [-10., -10.],
                    linestyle='-.', c='k', marker='', label='AGN-LIER')
        Sf_L_line_SII = self.AGN_LIER_(SII_Ha_grid, **self.Sf_L_cs['SII_Ha'])
        SII_ax.plot(
            SII_Ha_grid[(Sf_L_line_SII > Ke01_line_SII) |
                             (SII_Ha_grid > 0.1)],
            Sf_L_line_SII[(Sf_L_line_SII > Ke01_line_SII) |
                          (SII_Ha_grid > 0.1)],
            linestyle='-.', c='k', marker='', label='AGN-LIER')
        Sf_L_line_OI = self.AGN_LIER_(OI_Ha_grid, **self.Sf_L_cs['OI_Ha'])
        OI_ax.plot(
            OI_Ha_grid[(Sf_L_line_OI > Ke01_line_OI) |
                            (OI_Ha_grid > -0.9)],
            Sf_L_line_OI[(Sf_L_line_OI > Ke01_line_OI) |
                         (OI_Ha_grid > -0.9)],
            linestyle='-.', c='k', marker='')

        # scatter plot of spaxels
        diag_scatter = NII_ax.scatter(
            self.NII_Ha, self.OIII_Hb, c=self.diag, edgecolor='None',
            marker='.', cmap=self.cmap, norm=self.norm, vmin=0., vmax=4.)
        SII_ax.scatter(
            self.SII_Ha, self.OIII_Hb, c=self.diag, edgecolor='None',
            marker='.', cmap=self.cmap, norm=self.norm, vmin=0., vmax=4.)
        OI_ax.scatter(
            self.OI_Ha, self.OIII_Hb, c=self.diag, edgecolor='None',
            marker='.', cmap=self.cmap, norm=self.norm, vmin=0., vmax=4.)
        '''cbar = fig.colorbar(diag_scatter, cax=cbar_ax)
        cbar.set_ticks([0., 1., 2., 3., 4])
        cbar.set_ticklabels(['UND.', 'SF', 'Comp.', 'AGN', 'LIER'])
        cbar.ax.tick_params(labelsize=10, length=0)
        cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.8])'''

        # fix axes limits and scales
        NII_ax.set_ylim([-1.25, 1.5])
        SII_ax.set_ylim(NII_ax.get_ylim())
        OI_ax.set_ylim(NII_ax.get_ylim())
        SII_ax.set_yticklabels([])
        OI_ax.set_yticklabels([])
        NII_ax.set_xlim([-2., 1.])
        SII_ax.set_xlim([-1.25, 0.75])
        OI_ax.set_xlim([-2.25, 0.])

        NII_ax.legend(loc='lower center', prop={'size': 11},
                      ncol=4, bbox_to_anchor=(1.5, -0.35, 0.1, 3.))

        NII_ax.set_ylabel(
            r'$\log{\frac{\mathrm{EW(OIII)}}{\mathrm{EW(H\beta)}}}$',
            size=14)
        NII_ax.set_xlabel(
            r'$\log{\frac{\mathrm{EW(NII)}}{\mathrm{EW(H\alpha)}}}$',
            size=14)
        SII_ax.set_xlabel(
            r'$\log{\frac{\mathrm{EW(SII)}}{\mathrm{EW(H\alpha)}}}$',
            size=14)
        OI_ax.set_xlabel(
            r'$\log{\frac{\mathrm{EW(OI)}}{\mathrm{EW(H\alpha)}}}$',
            size=14)

        if objname is not None:
            plt.suptitle('{} BPT'.format(objname))

        if save == False:
            plt.show()
        else:
            plt.savefig('bpt.png')

        plt.close()

    def map(self, h, save=False, objname=None):
        '''
        make a BPT map of an ifu
        '''

        fig = plt.figure(figsize=(5, 4), dpi=300)
        ax = pywcsgrid2.subplot(111, header=h)

        im = ax.imshow(
            self.diag, origin='lower', aspect='equal', interpolation='None',
            cmap=self.cmap, norm=self.norm, vmin=0., vmax=4.)
        cbar = plt.colorbar(im)
        cbar.set_ticks([0., 1., 2., 3., 4])
        cbar.set_ticklabels(['UND.', 'SF', 'Comp.', 'AGN', 'LIER'])
        cbar.ax.tick_params(labelsize=9, length=0)
        ax.set_ticklabel_type(
            'delta',
            center_pixel=tuple(t/2. for t in self.diag.shape))
        # add beam
        bs = 2.*u.arcsec/(h['CDELT1'] * h['PC1_1'] * u.deg)
        ax.add_beam_size(bs, bs, 0., loc=1)
        ax.tick_params(axis='both', colors='w')
        ax.grid()
        plt.subplots_adjust(bottom=0.125, left=0.2, top=0.95, right=0.95)

        if objname is not None:
            plt.suptitle('{} BPT'.format(objname))

        #plt.tight_layout()

        if save == False:
            plt.show()
        else:
            plt.savefig('bpt_map.png')

        plt.close()
