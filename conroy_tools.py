import numpy as np
import astropy
import astropy.table as table
import astropy.io.fits as fits
import pandas as pd
import os
import re
from matplotlib import rcParams, pyplot as plt, patches
from astropy.wcs import WCS
from astropy import units as u, constants as c, wcs
from glob import glob
from scipy.interpolate import interp1d
from pysynphot import observation, spectrum
from matplotlib import gridspec, colors
import matplotlib.ticker as mtick
import pywcsgrid2
import itertools
import gz2tools as gz2

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
    lgood = ((3000. * u.AA <= l) * (l <= 10500. * u.AA))
    l = l[lgood]

    # log-transform
    logl = np.log10(l / u.AA)

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

    h = {'CTYPE3': 'log10 age/Gyr',
         'CRVAL3': np.min(logT),
         'NAXIS3': nT,
         'CDELT3': np.abs(np.mean(logT[:-1] - logT[1:])),
         'CTYPE2': 'log10 Z/Zsol',
         'CRVAL2': np.min(Zs),
         'NAXIS2': nZ,
         'CDELT2': np.abs(np.mean(Zs[:-1] - Zs[1:])),
         'CTYPE1': 'log10 lambda/AA',
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
