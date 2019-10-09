import numpy as np

from delfi.kernel.BaseKernel import BaseKernel


class Gauss(BaseKernel):
    @staticmethod
    def kernel(u):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u**2)
        
class HalfGauss(BaseKernel):
    def __init__(self, obs, bandwidth=1., loss_failed_sims=np.inf, atleast=None):
        super().__init__(obs, bandwidth=bandwidth, spherical=False, atleast=atleast)
        self.loss_failed_sims = loss_failed_sims

    @staticmethod
    def kernel(u):
        if np.isnan(u):
            return 0.0
        elif u <= 0.0 :
            return 1/np.sqrt(2*np.pi)
        else:
            return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u**2)
            
    def eval(self, x):
        """Kernel for loss calibration

        Parameters
        ----------
        x : N x dim
            points at which to evaluate kernel

        Returns
        -------
        weights : N
            normalized to be 1. for x = obs
        """
        assert x.shape[0] >= 1, 'x.shape[0] needs to be >= 1'
        assert x.shape[1] == self.dim, 'x.shape[1] needs to be == self.obs'

        out = np.ones((x.shape[0],))

        for n in range(x.shape[0]):
            if np.any(x[n] >= self.loss_failed_sims):
              us = np.nan * self.obs
            else:
              us = np.dot(self.invH, np.array(x[n] - self.obs).T)

            if self.spherical:
                out[n] = self.normalizer * self.kernel(np.linalg.norm(us))
            else:
                for u in us:
                    out[n] *= self.normalizer * self.kernel(u)

        # check fraction of points accepted
        if self.atleast is not None:
            accepted = out > 0.0
            if sum(accepted) / len(accepted) < self.atleast:
                dists = np.linalg.norm(x - self.obs, axis=1)
                N = int(np.round(x.shape[0] * self.atleast))
                idx = np.argsort(dists)[:N]
                out = np.zeros((x.shape[0],))
                out[idx] = 1.
                return out

        return out
