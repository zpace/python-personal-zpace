import warnings as w
with w.catch_warnings():
    w.simplefilter('ignore')
    import manga_tools as m
from astropy import table, units as u
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import os

pipe3d_base_url = m.base_url + 'mangawork/manga/sandbox/pipe3d/'

def get_p3d_out(version, plateifu, dest='.', verbose=False):
    # are we working with an MPL designation or a simple version number?
    if type(version) == int:
        vtype = 'MPL'
        version = 'MPL-' + str(version)
    elif version[:3] == 'MPL':
        vtype = 'MPL'

    v_url = pipe3d_base_url + version

    for ftype in ['ELINES', 'SSP']:

        full_url = v_url + '/manga-' + plateifu + '.' + ftype + '.cube.fits.gz'

        if verbose == True:
            v = 'v'
        else:
            v = ''

        c = 'rsync -a{3}z --password-file {0} rsync://sdss@{1} {2} \
            > rs_log.log'.format(m.pw_loc, full_url, dest, v)

        if verbose == True:
            print 'Here\'s what\'s being run...\n\n{0}\n'.format(c)

        os.system(c)

def process_p3d_out(version, plateifu, t=None):
    '''
    add info from one galaxy into table t
    '''

    # load drpall
    drpall_fname = m.drpall_loc + 'drpall-{}.fits'.format(
        m.MPL_versions[version])

    # load galaxy derived parameters data
    eline = fits.open('manga-{}.ELINES.cube.fits.gz'.format(plateifu))[0].data
    SSP = fits.open('manga-{}.SSP.cube.fits.gz'.format(plateifu))[0].data

    # get useful metadata
    drpall_params = m.get_drpall_val(
                        drpall_fname,
                        ['nsa_zdist', 'nsa_mstar', 'nsa_ba', 'nsa_phi',
                        'nsa_sersic_n', 'nsa_sersicflux'],
                        plateifu)
    mstar = drpall_params['nsa_mstar'].quantity[0]
    zdist = drpall_params['nsa_zdist'].quantity[0]

    sersic_n = drpall_params['nsa_sersic_n'].quantity[0]
    sersic_flux = drpall_params['nsa_sersicflux'].quantity[0, 2]

    zdist *= m.c / m.H0
    platescale = zdist / 206265. # convert from arcsec to linear dist

    ba = drpall_params['nsa_ba'][0]
    phi = drpall_params['nsa_phi'][0] * np.pi/180.
    # Hubble formula for inclination in terms of axis ratio
    # http://aas.aanda.org/articles/aas/full/2000/01/ds9210/node3.html
    i = np.arccos(np.sqrt(1.024*ba**2. - .042))

    # print ba, i

    # make x & y coordinates
    coord = np.linspace(-0.5*eline.shape[1]/2+.25, 0.5*eline.shape[1]/2-.25,
                        eline.shape[1])
    XX, YY = np.meshgrid(coord, coord)
    coords = np.stack((XX, YY), axis=0)
    R = np.sqrt((coords**2.).sum(axis=0))

    a = np.arctan2(YY, XX)
    # angle to either phi or phi - pi, whichever is smaller
    a2phi = np.minimum(np.abs(a-phi), np.abs(a-(phi-np.pi)))
    a2phi = np.minimum(a2phi.T, np.abs(a-(phi-2.*np.pi)))

    factor = np.abs(np.sin(a2phi))

    m.get_cutout('MPL-4', plateifu)
    R_deproj = R * (1 + 2*factor)
    #im = plt.imshow(mpimg.imread(plateifu + '.png'),
    #                aspect='equal', origin='upper',
    #                extent=[coord.min(), coord.max(),
    #                        coord.min(), coord.max()])
    #c = plt.contour(XX, YY, R * (1 + 2. * factor), cmap='RdBu',
    #                levels=[1., 2., 4., 8., 16.])
    #c = plt.contour(XX, YY, SSP[0]/SSP[0].max(), cmap='cubehelix')
    #plt.colorbar(c)
    #plt.show()

    names_0 = ['plateifu', 'R', 'R_deproj']

    eline_names = ['OII3727', 'OIII5007', 'OIII4959', 'Hb', 'Ha', 'NII6583',
                   'NII6548', 'SII6731', 'SII6717']
    eline_idxs = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    SSP_names = ['LWA', 'LWZ', 'sig', 'M/L', 'Mdens', 'MeanFlux']
    SSP_idxs = [5, 8, 15, 17, 18, 3]

    # how big is the good data?
    good_spaxels = (eline[0, :, :] != 0) * (SSP[3, :, :] != 0)
    good_spaxels *= ((SSP[0, :, :] / SSP[0, :, :].max()) >= 0.05)
    ngood = int(good_spaxels.sum())

    t_new = table.Table()
    #    data=np.empty((ngood, len(names_0 + eline_names + SSP_names))),
    #    names=names_0 + eline_names + SSP_names)

    t_new['plateifu'] = [plateifu,]*ngood
    t_new['R'] = R[good_spaxels].flatten() * platescale
    t_new['R_deproj'] = R_deproj[good_spaxels].flatten() * platescale
    t_new['NSA_Mstar'] = [mstar,]*ngood
    t_new['Sersic_N'] = [sersic_n,]*ngood
    t_new['Sersic_Flux'] = [sersic_flux,]*ngood # nanomaggies

    for i, n in zip(eline_idxs, eline_names):
        t_new[n] = eline[i][good_spaxels].flatten()

    for i, n in zip(SSP_idxs, SSP_names):
        t_new[n] = SSP[i][good_spaxels].flatten()

    if t is None:
        t = t_new
    else:
        t = table.vstack((t, t_new))

    return t

if __name__ == '__main__':
    test = False
    version = 'MPL-4'
    try:
        os.remove('rs_log.log')
    except:
        pass
    drpall_fname = m.drpall_loc + 'drpall-{}.fits'.format(
        m.MPL_versions[version])
    drpall = table.Table.read(drpall_fname)

    large_ifu = [True if (str(s)[:3] == '127') else False
                 for s in drpall['ifudsgn']]

    drpall = drpall[((drpall['nsa_ba'] < 2.5) * large_ifu)]
    print len(drpall), 'galaxies'

    for i, obj in enumerate(drpall):
        if i%10 == 0:
            print i+1, 'of', len(drpall)
        try:
            get_p3d_out(version, obj['plateifu'])

            if i == 0:
                t = process_p3d_out(version, obj['plateifu'])
            else:
                t = process_p3d_out(version, obj['plateifu'], t)
        except KeyboardInterrupt:
            raise
        except:
            pass

        fnames_used = glob('*' + obj['plateifu'] + '*.fits.gz')
        for f in fnames_used:
            try:
                os.remove(f)
            except KeyboardInterrupt:
                raise
            except:
                pass

        if (test == True) and (i == 5):
            break

    f = '/home/zpace/Documents/classes/STAT575/project/MPL-4_127f_p3d.fits'
    t.write(f, overwrite=True)
