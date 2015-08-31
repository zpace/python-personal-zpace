import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.table as table
import pandas as pd
import os
import re

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
    if version[:3] == 'MPL':
        vtype = 'MPL'
    elif type(version) == int:
        vtype = 'MPL'
        version = 'MPL-' + str(version)

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
