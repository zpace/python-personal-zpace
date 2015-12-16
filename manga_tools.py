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

drpall_loc = '/home/zpace/Documents/MaNGA_science/'
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
