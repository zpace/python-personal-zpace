import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib
import io
from scipy.misc import imsave
import os.system
import datetime


def download_sloan_im(ra, dec, scale, width=256, height=256, verbose=True):
    '''
    ra & dec: sky coords
    scale: pixel scale (in arcsec) of image
        (.02 * rPetro is recommended for galaxies)
    '''

    plt.ioff()

    width, height = int(width), int(height)

    if verbose:
        print('Requesting image...')
        print('RA: {}'.format(ra))
        print('DEC: {}'.format(dec))
        print('scale: {} arcsec/pix'.format(scale))
        print('width: {} pix'.format(width))
        print('height: {} pix'.format(height))

    base_url = \
        'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.' + \
        'aspx?opt=0'
    im_url = base_url + '&width=' + str(width) + '&height=' + \
        str(height) + '&scale=' + str(np.round(scale, 3)) + '&ra=' + \
        str(ra) + '&dec=' + str(dec)

    if verbose:
        print(im_url)

    im = urllib.urlopen(im_url)
    image_file = io.BytesIO(im.read())
    img = mpimg.imread(image_file, format='jpg')
    return img


def example():
    ra = 197.61446
    dec = 18.43817
    scale = 0.02 * 18.23
    width, height = 256, 256
    img = download_sloan_im(ra, dec, scale, width, height)
    plt.imshow(
        img, extent=[-width * scale / 2., width * scale / 2.,
                     -height * scale / 2., height * scale / 2.])
    plt.show()


def save_sloan(ra, dec, scale, width=256, height=256, imname=None,
               ext='jpg'):
    '''
    save an SDSS thumbnail as an image file
    '''

    if not imname:
        # if there's no name specified, just use the RA and Dec
        imname = str(ra) + '_' + str(dec)

    img = download_sloan_im(ra, dec, scale, width, height, verbose=False)
    imsave(imname + '.' + ext, img)


def casjobs_query(query, query_name=None, destination=None):
    '''
    run a CasJobs query using CasJobs .jar file, via a shell script wrapper

    Arguments:
        - query (str): possibly multi-line string
    '''
    # assumes that casjobs.jar is in ~/casjobs, and that CasJobs.config is
    # shipshape (this is done by the shell script in /home/bin)

    if not query_name:
        query_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    os.system(
        'casjobs run -t "dr12/1" -n {qn} {q}'.format(qn=query_name, q=query))

    """UNFINISHED"""
