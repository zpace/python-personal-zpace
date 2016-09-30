"""
defining custom IMFs to sample from
"""

import numpy as np
from scipy.stats import rv_continuous
import functools

class PowerLawInitialMassFunction(object):
    """
    Contains convenience power-law functions
    """
    def __init__(self):
        self.mlow = self.mhigh = self.slope = 0.

    @functools.lru_cache()
    @property
    def norm(self):
        return np.abs(self._plaw_integral_eval(self.mlow, self.slope) -
                      self._plaw_integral_eval(self.mhigh, self.slope))

    def _plaw_eval(self, x, slope):
        '''
        power law evaluated at x
        '''
        return x**slope

    def _plaw_integral_eval(self, x, slope):
        '''
        indefinite integral of a power law evaluated at x
        '''
        return x**(slope + 1) / (slope + 1)


class SalpeterInitialMassFunctionRV(PowerLawInitialMassFunction,
                                    rv_continuous):
    """
    Salpeter Stellar Initial Mass Function random variate
    """
    def __init__(self, mlow=.08, mhigh=150., slope=-2.35, **kwargs):

        super().__init__(a=mlow, b=mhigh)
        super().__init__(**kwargs)

        self.a = self.mlow = mlow
        self.b = self.mhigh = mhigh
        self.slope = slope

    def _pdf(self, x):
        return 1./self.norm * self._plaw_eval(x, self.slope)


class BrokenPowerLawInitialMassFunction(PowerLawInitialMassFunction,
                                        rv_continuous):
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
        self.bounds = bounds
        self.slopes = slopes
        self.breaks = breaks

        super().__init__(a=mlow, b=mhigh, **kwargs)

    @functools.lru_cache()
    @property
    def _match(self):
        '''
        How to scale each power-law segment to maintain continuity
        '''
        # start off by evaluating each power-law at each appropriate break
        # iterate through each break,
        # and then iterate through each function that hits it
        bk_vals = np.column_stack(
            [np.array([self._plaw_eval(b, slopes[i]),
                       self._plaw_eval(b, slopes[i + 1])])
             for i, b in enumerate(breaks)])

        bk_ratios = bk_vals[:, 1] / bk_vals[:, 0]
        bk_ratios = np.insert(arr=bk_ratios, obj=0, values=1.)
        prod = np.prod(bk_ratios)

        # so if there are three segments (and two breaks)
        # then bk_ratios will have length 3
        # segment n will be multiplied by prod[n]

        return prod

    @functools.lru_cache()
    @property
    def endpoints(self):
        endpoints = np.concatenate(
            [np.array([self.bounds[0]]),
             self.breaks,
             np.array([self.bounds[1]])])
        return endpoints

    @functools.lru_cache()
    @property
    def norm(self):
        '''
        integral of entire broken power-law, with component scalings
        '''

        segment_integrals = self.match * np.array(
            [(self._plaw_integral_eval(self.endpoints[i], s) -
              self._plaw_integral_eval(self.endpoints[i + 1], s))
             for i, s in enumerate(self.slopes)])

        return segment_integrals.sum()

    def _pdf(self, x):
        # choose the function branch
        begin = self.endpoints[:-1]
        end = self.endpoints[1:]
        branch = np.where(((x >= begin) & (x < end)) == True)[0]

        # and evaluate on that branch
        return (self._match[branch] / self.norm *
                self._plaw_eval(x, self.slopes[branch]))


class KroupaInitialMassFunction(BrokenPowerLawInitialMassFunction):
    """
    Kroupa (broken power-law) Initial Mass Function random variate
    """
    def __init__(self, mlow=.08, mhigh=150., mbreak1=.08, mbreak2=.5
                 slope_low=-.03, slope_mid=-1.3, slope_high=-2.3, **kwargs):

        super().__init__(bounds=[mlow, mhigh],
                         slopes=[slope_low, slope_mid, slope_high],
                         **kwargs)

        self.a = self.mlow = mlow
        self.b = self.mhigh = mhigh
        self.mbreak1 = mbreak1
        self.mbreak2 = mbreak2
        self.slope_low = slope_low
        self.slope_mid = slope_mid
        self.slope_high = slope_high
