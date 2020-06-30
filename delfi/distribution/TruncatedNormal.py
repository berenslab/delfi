import numpy as np
import scipy.stats

from delfi.distribution.BaseDistribution import BaseDistribution
from delfi.distribution.StudentsT import StudentsT


class TruncatedNormal(BaseDistribution):
    def __init__(self, m, S, a=None, b=None, seed=None):
        """Truncated normal distribution

        Initialize a gaussian pdf

        Parameters
        ----------
        m : list or np.array, 1d
            Mean
        S : list or np.array, 2d
            Covariance
        a : list or np.array, 1d, or None
            Lower bound
        b : list or np.array, 1d, or None
            Upper bounds

        seed : int or None
            If provided, random number generator will be seeded
        """
        assert np.asarray(m).ndim == 1
        assert np.asarray(S).ndim == 2
        assert a is None or np.asarray(a).ndim == 1
        assert b is None or np.asarray(b).ndim == 1

        m = np.asarray(m)
        self.m = m
        ndim = self.m.size

        S = np.asarray(S)
        self.P = np.linalg.inv(S)
        self.C = np.linalg.cholesky(S).T  # C is upper triangular here
        self.S = S
        #self.Pm = np.dot(self.P, m)
        self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

        if a is not None:
            a = np.asarray(a)
            assert a.size == m.size
            self.a = a
        else:
            self.a = None
            
        if b is not None:
            b = np.asarray(b)
            assert b.size == m.size
            self.b = b
        else:
            self.b = None

        super().__init__(ndim, seed=seed)

        self.p_normalizer_n_samples = 100_000
        self.p_normalizer = self.compute_p_normalizer(m, S, a, b)
        self.p_normalizer_dict = {}

    @property
    def mean(self):
        """Means"""
        return self.m.reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt(np.diag(self.S)).reshape(-1)

    def compute_p_normalizer(self, m, S, a, b):
        ''' Compute cumulative distribution outside valid area.
        This can then be used as a normalization factor to compute the pdf.
        pdf = pdf(Normal) * p_normalizer
        
        Returns:
        
        p_normalizer : float
            1 over fraction of probability outside [a,b]
        
        '''
        if a is None:
            a = np.full(m.size, np.NINF)
        if b is None:
            b = np.full(m.size, np.inf)
        
        if (m.size == 1) or ((m.size == 2) and np.all(m >= a) and np.all(m <= b)):
            cdb, cdna = scipy.stats.multivariate_normal.cdf(
                x=[m-a, b-m], mean=np.zeros(m.size), cov=S,
                abseps=1e-15, releps=1e-15,
            )
            cd_in_area = cdb+cdna-1
            p_normalizer = 1 / cd_in_area
            
        else:
            # The CDF approach failes if too much weight is outside of the bounds.
            n_samples = self.p_normalizer_n_samples
            n_accepted = self.gen_and_reject(n_samples).shape[0]
            
            if n_accepted < 1000:
                n_samples = int(self.p_normalizer_n_samples * 100)
                n_accepted = self.gen_and_reject(n_samples).shape[0]
            
            assert n_accepted > 0, 'Too much weight outside bounds'
                
            p_normalizer = float(n_samples / n_accepted)

        
        return p_normalizer

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):

        x = np.asarray(x)
        if ii is not None:
            ii = np.asarray(ii).astype(int)
            if self.a is not None: a = self.a[ii]
            if self.b is not None: b = self.b[ii]
        else:
            if self.a is not None: a = self.a
            if self.b is not None: b = self.b
        
        accepted_samples = np.ones(x.shape[0], dtype=bool)
        
        if self.a is not None:
            accepted_samples[np.any(x < a, axis=1)] = False
        
        if self.b is not None:
            accepted_samples[np.any(x > b, axis=1)] = False

        x = x[accepted_samples]

        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

            p_normalizer = self.p_normalizer

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            if np.linalg.matrix_rank(S)==len(S[:,0]):
                lp = scipy.stats.multivariate_normal.logpdf(x, m, S, allow_singular=True)
                lp = np.array([lp]) if x.shape[0] == 1 else lp
            else:
                raise ValueError('Rank deficiency in covariance matrix')
            
            if str(ii) not in self.p_normalizer_dict.keys():
                self.p_normalizer_dict[str(ii)] = self.compute_p_normalizer(m, S, a, b)

            p_normalizer = self.p_normalizer_dict[str(ii)]

        if log:
            res = lp + np.log(p_normalizer)
        else:
            res = np.exp(lp) * p_normalizer
            
        res_all = np.zeros(accepted_samples.size, dtype=float)
        if np.any(accepted_samples):
          res_all[accepted_samples] = res
            
        return res_all

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        
        # Resampling if outside of bounds.
        samples = self.gen_and_reject(n_samples=n_samples)
        acceptance_rate = float(1+samples.shape[0]) / n_samples # Avoid zero.

        while samples.shape[0] < n_samples:
            n_samples_remain = n_samples - samples.shape[0]
            n_samples_gen = int(np.ceil(1.5*n_samples_remain/acceptance_rate))
            new_samples = self.gen_and_reject(n_samples=n_samples_gen)
            
            samples = np.concatenate([samples, new_samples])
            acceptance_rate = float(1+new_samples.shape[0]) / n_samples_gen

        return samples[:n_samples]
        
    
    def gen_and_reject(self, n_samples):
        ''' Generate samples and return accpeted samples,
        i.e. samples that are in [a,b].
        Note, that the n_samples is therefore not equal to
        the number of returned samples.
        '''
        samples = self.gen_normal(n_samples=n_samples)
        accepted_samples = np.ones(samples.shape[0], dtype=bool)
        
        if self.a is not None:
            accepted_samples[np.any(samples < self.a, axis=1)] = False
        
        if self.b is not None:
            accepted_samples[np.any(samples > self.b, axis=1)] = False
        
        return samples[accepted_samples]
        
    
    def gen_normal(self, n_samples=1):
        ''' Generate samples from a normal distribution,
        i.e. as if a and b were None.
        '''
        z = self.rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m
        return samples
