"""
defining custom IMFs to sample from
"""

import numpy as np
from scipy.stats import rv_continuous
import functools

def _plaw_eval(x, slope):
    '''
    power law evaluated at x
    '''
    return x**slope

def _plaw_integral_eval(x, slope):
    '''
    indefinite integral of a power law evaluated at x
    '''
    return x**(slope + 1) / (slope + 1)


class SalpeterInitialMassFunctionRV(rv_continuous):
    """
    Salpeter Stellar Initial Mass Function random variate
    """
    def __init__(self, mlow=.08, mhigh=150., slope=-2.35, **kwargs):

        super().__init__(a=mlow, b=mhigh)

        self.a = self.mlow = mlow
        self.b = self.mhigh = mhigh
        self.slope = slope

    @property
    def norm(self):
        return (_plaw_integral_eval(self.mhigh, self.slope) -
                _plaw_integral_eval(self.mlow, self.slope))

    def _pdf(self, x):
        return 1./self.norm * _plaw_eval(x, self.slope)


class BrokenPowerLawInitialMassFunction(rv_continuous):
    """
    Broken power-law Initial Mass Function random variate
    """
    def __init__(self, bounds, slopes, breaks, **kwargs):

        # make some idiot checks
        if not hasattr(breaks, '__len__'):
            breaks = [breaks]

        if not hasattr(slopes, '__len__'):
            raise ValueError('slopes must have a length')

        if len(bounds) != 2:
            raise ValueError('Need upper and lower mass bound')

        if len(slopes) != len(breaks) + 1:
            raise ValueError('Need one more power-law slope than breakpoint!')

        self.a = self.mlow = bounds[0]
        self.b = self.mhigh = bounds[1]
        self.bounds = np.array(bounds)
        self.slopes = np.array(slopes)
        self.breaks = np.array(breaks)

        super().__init__(a=self.mlow, b=self.mhigh, **kwargs)

    @property
    def _match(self):
        '''
        How to scale each power-law segment to maintain continuity
        '''
        # start off by evaluating each power-law at each appropriate break
        # iterate through each break,
        # and then iterate through each function that hits it
        bk_vals = np.column_stack(
            [np.array([_plaw_eval(b, self.slopes[i]),
                       _plaw_eval(b, self.slopes[i + 1])])
             for i, b in enumerate(self.breaks)])

        bk_ratios = bk_vals[1, :] / bk_vals[0, :]
        bk_ratios = np.insert(arr=bk_ratios, obj=0, values=1.)
        prod = np.cumprod(bk_ratios, axis=0)

        # so if there are three segments (and two breaks)
        # then bk_ratios will have length 3
        # segment n will be multiplied by prod[n]

        return prod

    @property
    def endpoints(self):
        endpoints = np.concatenate(
            [np.array([self.bounds[0]]),
             self.breaks,
             np.array([self.bounds[1]])])
        return endpoints

    @property
    def norm(self):
        '''
        integral of entire broken power-law, with component scalings
        '''

        segment_integrals = self._match * np.array(
            [(_plaw_integral_eval(self.endpoints[i + 1], s) -
              _plaw_integral_eval(self.endpoints[i], s))
             for i, s in enumerate(self.slopes)])

        return segment_integrals.sum()

    def _pdf(self, x):
        # choose the function branch
        begin = self.endpoints[:-1]
        end = self.endpoints[1:]
        x_ = np.asarray(x)[..., None]
        branch = np.argmax((x_ >= begin) & (x_ < end), axis=-1)

        match = np.array([self._match[b]
                          for b in np.atleast_1d(np.asarray(branch))])

        # and evaluate on that branch
        return (1. / (self.norm * match) *
                _plaw_eval(x, self.slopes[branch]))


class KroupaInitialMassFunction(BrokenPowerLawInitialMassFunction):
    """
    Kroupa (broken power-law) Initial Mass Function random variate
    """
    def __init__(self, mlow=.08, mhigh=150., mbreak1=.08, mbreak2=.5,
                 slope_low=-.03, slope_mid=-1.3, slope_high=-2.3, **kwargs):

        super().__init__(bounds=[mlow, mhigh],
                         slopes=[slope_low, slope_mid, slope_high],
                         breaks=[mbreak1, mbreak2],
                         **kwargs)

        self.a = self.mlow = mlow
        self.b = self.mhigh = mhigh
        self.mbreak1 = mbreak1
        self.mbreak2 = mbreak2
        self.slope_low = slope_low
        self.slope_mid = slope_mid
        self.slope_high = slope_high
