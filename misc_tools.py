import sys
import os
from astropy import time as t, units as u, coordinates as c
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def element_count(p):
    elements = list(p)
    count = 0
    while elements:
        entry = elements.pop()
        count += 1
        if isinstance(entry, list):
            elements.extend(entry)
    return count

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

class site_LST(object):
    '''
    doesn't work
    '''
    def __init__(self, d, site='kpno'):
        site_c = c.EarthLocation.of_site(site)
        print site_c
        times = [t.Time(datetime(d[0], d[1], d[2], int(h)), scale='utc',
                        location=site_c, format='datetime')
                 for h in np.linspace(0, 23, 24)]

        for t_ in times:
            print t_.sidereal_time('apparent')
