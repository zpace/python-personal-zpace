import numpy as np
import manga_elines as mel


class BPT(object):
    '''
    compute BPT classes
    '''
    labels = ['None', 'SF', 'Li(N)ER', 'AGN', 'Comp.']
    class_labels = dict(enumerate(labels))
    class_numbers = {v: k for k, v in class_labels.items()}
    lines = ['Hb-4862', 'Ha-6564', 'OIII-5008', 'NII-6585',
             'SII-6718', 'SII-6732', 'OI-6302']

    def __init__(self, maps_hdulist, snr_dict):
        self.maps_hdulist = maps_hdulist
        self.snr_dict = snr_dict

        self.elines = {mel.get_emline_qty(maps_hdulist, 'SEW', key=k,
                                          sn=snr_dict.get(k, 3.))
                       for k in lines}


class KaKe(object):
    '''
    Kauffmann-Kewley-style separation line

    log(Oiii / Hb) = (a / log(forbidden / Ha) + b) + c
    '''
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def __call__(self, forbidden, Ha):
        return (self.a / np.log10(forbidden / Ha) + self.b) + self.c

    def classify(self, forbidden, Ha, Oiii, Hb):
        return np.log10(Oiii / Hb) > self.__call__(forbidden, Ha)

class LINER_AGN(object):
    '''
    power-law LINER-AGN separation line
    '''
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, forbidden, Ha):
        return (self.a * np.log10(forbidden / Ha) + self.b)

    def classify(self, forbidden, Ha, Oiii, Hb):
        return np.log10(Oiii / Hb) > self.__call__(forbidden, Ha)

