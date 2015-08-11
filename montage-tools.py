import montage_wrapper as montage
import astropy.io.ascii as ascii
import numpy as np

def header_group(objlist, out_file):
    '''
    create a montage-style header for a list of objects (NOT ra/dec FTTB)

    calls montage-wrapper.commands.mHdr multiple times to get coordinates
    '''