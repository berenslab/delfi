import numpy as np

from delfi.kernel.BaseKernel import BaseKernel

"""
simple hack to avoid Nans in the network if imputed=True
TODO: better solution would be (?) to calculate kernel on imputed values
"""

class GaussNan(BaseKernel):
    @staticmethod
    def kernel(u):
        if np.isnan(u):
            # not that clear what value to choose, but should not be that important
            return 0.5
        else:
            return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * u ** 2)

class HalfGaussNan(BaseKernel):
    @staticmethod
    def kernel(u):
        if np.isnan(u):
            # not that clear what value to choose, but should not be that important
            return 0.5
        if u <= 0.0 :
            return 1.0
        else:
            return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u**2)