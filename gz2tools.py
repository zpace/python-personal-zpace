import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib
import io
from scipy.misc import imsave

def download_sloan_im(ra, dec, scale, width = 256, height = 256):
    '''
    ra & dec: sky coords
    scale: pixel scale (in arcsec) of image (.02 * rPetro is recommended)
    '''

    plt.ioff()

    base_url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?opt=0'
    im_url = base_url + '&width=' + str(width) + '&height=' + str(height) + '&scale=' + str(np.round(scale, 3)) + '&ra=' + str(ra) + '&dec=' + str(dec)

    im = urllib.urlopen(im_url)
    image_file = io.BytesIO(im.read())
    img = mpimg.imread(image_file, format = 'jpg')
    return img

def example():
    ra = 197.61446
    dec = 18.43817
    scale = 0.02 * 18.23
    width, height = 256, 256
    img = download_sloan_im(ra, dec, scale, width, height)
    plt.imshow(img, extent = [-width*scale/2., width*scale/2., -height*scale/2., height*scale/2.])
    plt.show()

def save_sloan(ra, dec, scale, width = 256, height = 256, imname = None,
               ext = 'jpg'):
    '''
    save an SDSS thumbnail as an image file
    '''

    if imname == None:
        #if there's no name specified, just use the RA and Dec
        imname = str(ra) + '_' + str(dec)

    img = download_sloan_im(ra, dec, scale, width, height)
    imsave(imname + '.' + ext, img)
