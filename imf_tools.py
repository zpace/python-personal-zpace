"""
defining custom IMFs to sample from
"""

import numpy as np
from scipy.stats import rv_continuous

class SalpeterInitialMassFunctionRV(rv_continuous):
    """
    Salpeter Stellar Initial Mass Function random variate
    """
    def __init__(self, mlow=.08, mhigh=150., slope=-2.35, **kwargs):

        super().__init__(a=mlow, b=mhigh, **kwargs)

        self.a = self.mlow = mlow
        self.b = self.mhigh = mhigh
        self.slope = slope

    def _plaw_eval(self, x):
        return x**self.slope

    def _plaw_integral_eval(self, x):
        return x**(self.slope + 1) / (self.slope + 1)

    def _pdf(self, x):
        norm = np.abs(self._plaw_integral_eval(self.mlow) -
                      self._plaw_integral_eval(self.mhigh))

        return 1./norm * self._plaw_eval(x)
