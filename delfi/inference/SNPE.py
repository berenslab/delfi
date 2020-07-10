import numpy as np
import pickle
import time

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero

class SNPE(BaseInference):
    def __init__(self, generator, obs=None, pseudo_obs_dim=None, pseudo_obs_perc=None, pseudo_obs_n=None,
                 kernel_bandwidth_perc=None, kernel_bandwidth_n=None, kernel_bandwidth_min=None,
                 pseudo_obs_use_all_data=False, prior_norm=False, pilot_samples=100,
                 convert_to_T=3, reg_lambda=0.01, prior_mixin=0, kernel=None, seed=None, verbose=True,
                 use_doubling=False, 
                 **kwargs):
        """Sequential neural posterior estimation (SNPE)

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array or list of arrays
            Observation or list of of observations in the format the generator returns (1 x n_summary)
            If list, obs will be changed every round. In this case it should be converging to the true value.
            The different observed value can be used as guidance for the algorithm.
            Alternatively, set pseudo_obs_perc.
        pseudo_obs_dim : int or None
            Dimension with adaptive pseudo-obs.
        pseudo_obs_perc : double in [0,100]
            If set, adaptively change obs relative to percentile of best samples.
            Set to zero to use best sample only.
        pseudo_obs_n : integer in [1, np.inf]
            If set, adaptively change obs. Set obs always to the n-th best sample.
        kernel_bandwidth_perc : double in [0,100]
            If set, adaptively change kernel bandwidth as percentile of best samples.
        kernel_bandwidth_n : integer in [1, np.inf]
            If set, adaptively change kernel bandwidth relatively to the n-th best sample.
        kernel_bandwidth_min : double >= 0
            If set, bandwidth will always be at least this size.
        pseudo_obs_use_all_data : bool
            Set to True to use all training data to compute percentile obs.
            Default is False. Then only the samples of the current round are used.
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        convert_to_T : None or int
            Convert proposal distribution to Student's T? If a number if given,
            the number specifies the degrees of freedom. None for no conversion
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        prior_mixin : float
            Percentage of the prior mixed into the proposal prior. While training,
            an additional prior_mixin * N samples will be drawn from the actual prior
            in each round.
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        use_doubling: bool
            if True, will duplicate every discrepancy such to have both
            negative and positive values.
        
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_components : int
                    Number of components of the mixture density
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)
        assert obs is not None, 'SNPE needs obs'
        assert pseudo_obs_perc is None or pseudo_obs_n is None, 'Can\'t set both. Use one or none.'
        assert kernel_bandwidth_perc is None or kernel_bandwidth_n is None, 'Can\'t set both. Use one or none.'
        
        self.obs = np.asarray(obs).astype(float)
        assert pseudo_obs_dim < self.obs.size
        
        self.pseudo_obs_dim = pseudo_obs_dim
        self.pseudo_obs_perc = pseudo_obs_perc
        self.pseudo_obs_n = pseudo_obs_n
        self.pseudo_obs_use_all_data = pseudo_obs_use_all_data
        self.kernel_bandwidth_perc = kernel_bandwidth_perc
        self.kernel_bandwidth_n = kernel_bandwidth_n
        self.kernel_bandwidth_min = kernel_bandwidth_min
        self.pseudo_obs = []
        self.kernel_bandwidth = []
        
        if use_doubling:
            assert (pseudo_obs_n is None) and (pseudo_obs_perc is None)
        self.use_doubling = use_doubling

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.convert_to_T = convert_to_T

        self.prior_mixin = 0 if prior_mixin is None else prior_mixin

        self.kernel = kernel
        
        if self.kernel.vector_kernel:
            assert pseudo_obs_dim is not None, 'Define a dimension where kernel will be streched'
        else:
            assert pseudo_obs_dim is None, 'Not implemented for this kernel.'

    def loss(self, N, round_cl=1):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = self.network.get_loss()

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.lprobs'] = self.network.lprobs
        self.observables['loss.iws'] = self.network.iws
        self.observables['loss.raw_loss'] = loss

        if self.svi:
            if self.round <= round_cl:
                # weights close to zero-centered prior in the first round
                if self.reg_lambda > 0:
                    kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                           self.reg_lambda)
                else:
                    kl, imvs = 0, {}
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss


    @staticmethod
    def percentile(arr, q):
        return arr[np.argsort(arr)[int(np.round(q/100*arr.size))]]


    @staticmethod
    def abs_diff_tds_obs_dim(tds, obs, ii):
        return np.abs(tds[:,ii] - obs[0,ii]).flatten()


    def compute_pseudo_obs(self, obs, pseudo_obs_dim, tds=None, q=None, n=None):
        """ Get or compute observed value
        
        Parameters:
        -----------
        
        obs : array
            Observed data
            
        pseudo_obs_dim : int
            Dimension to compute pseudo_obs for
        
        tds: array
            Only needed when obs is computed as percentile of current samples.
            Will then be used to compute the pseudo_obs_perc percentile of the samples relative
            to the true observed value. Should eventually converge to the true value.
            
        q / n : float / int
            Define how to compute pseudo_obs, either as percentile or as n-th sample.
            
        """
        
        assert ((q is None) + (n is None)) == 1, 'Set exactly one'
        
        assert isinstance(obs, np.ndarray)
        abs_tds = self.abs_diff_tds_obs_dim(tds=tds, obs=obs, ii=pseudo_obs_dim)
        
        if q is not None:
            new_obs_i = self.percentile(arr=abs_tds, q=q)
        else:
            new_obs_i = abs_tds[np.argsort(abs_tds)[n-1]]
            
        new_obs = obs.copy()
        new_obs[0,pseudo_obs_dim] = new_obs_i
            
        return new_obs
    
    
    def adapt_kernel_bandwidth(self, pseudo_obs_tds, pseudo_obs=None):
        ''' Compute and set new kernel bandwidth using training data and distance to obs.
        '''
      
        kernel_bandwidth = self.compute_kernel_bandwidth(
            obs=self.obs, pseudo_obs_dim=self.pseudo_obs_dim,
            bandwidth=self.kernel.bandwidth, tds=pseudo_obs_tds,
            bandwidth_min=self.kernel_bandwidth_min,
            q=self.kernel_bandwidth_perc, n=self.kernel_bandwidth_n,
        )
        
        if self.pseudo_obs_dim is not None:
            assert (pseudo_obs is not None), 'pseudo_obs_dim need pseudo_obs'
            self.kernel.set_max_weight_range_ii(pseudo_obs[0,self.pseudo_obs_dim], ii=self.pseudo_obs_dim)
            self.kernel.set_bandwidth_ii(kernel_bandwidth, ii=self.pseudo_obs_dim)
        else:
            if (pseudo_obs is not None): self.kernel.obs = pseudo_obs
            self.kernel.set_bandwidth(kernel_bandwidth)
            
        self.kernel_bandwidth.append(self.kernel.bandwidth) # Save.
    
    def compute_kernel_bandwidth(self, obs, pseudo_obs_dim, bandwidth, bandwidth_min, tds=None, q=None, n=None):
        """ Update kernel_bandwidth for dimension with adaptive observed.
        Can not deal with multiple adaptive dimensions.
        
        Parameters:
        -----------
        
        obs : array
            Observed data
            
        pseudo_obs_dim : int
            Dimension to compute pseudo_obs for
        
        bandwidth : float or 1d-array
            Current bandwidth.
        
        tds: array
            Only needed when obs is computed as percentile of current samples.
            Will then be used to compute the pseudo_obs_perc percentile of the samples relative
            to the true observed value. Should eventually converge to the true value.
            
        q / n : float / int
            Define how to compute pseudo_obs, either as percentile or as n-th sample.
            
        Returns:
        --------
        
        bandwidth
            
        """
      
        assert ((q is None) + (n is None)) == 1, 'Set exactly one'
        
        abs_tds = self.abs_diff_tds_obs_dim(tds=tds, obs=obs, ii=pseudo_obs_dim)
        
        # Compute new bandwidth.
        if q is not None:
            new_bandwidth = self.percentile(arr=abs_tds, q=q) - obs[0,pseudo_obs_dim]
        else:
            new_bandwidth = abs_tds[np.argsort(abs_tds)[n-1]] - obs[0,pseudo_obs_dim]
        
        if bandwidth_min is not None:
            if not np.isfinite(new_bandwidth):
                print('\t Computed bandwidth was NaN, use minimum value {:.2g}.'.format(bandwidth_min))
                new_bandwidth = bandwidth_min
            elif new_bandwidth < bandwidth_min:
                print('\t Computed bandwidth {:.2g} smaller than minimum {:.2g}.'.format(new_bandwidth, bandwidth_min))
                new_bandwidth = bandwidth_min
        
        assert new_bandwidth > 0
        
        return new_bandwidth
    
    
    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            round_cl=1, stop_on_nan=False, proposal=None, verbose=True,
            monitor=None, initial_tds=None, **kwargs):

        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        round_cl : int
            Round after which to start continual learning
        stop_on_nan : bool
            If True, will halt if NaNs in the loss are encountered
        proposal : Distribution or None
            If given, will use this distribution as the starting proposal prior
        verbose: bool
            if True, simple print output for the progress
        initial_tds : tuple of training data
            Precomputed training data.

        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of distributions
            posterior after each round
        """
        
        logs = []
        trn_datasets = []
        posteriors = []
        
        for r in range(n_rounds):
            self.round += 1
            
            # Set proposal distribution to sample from.
            if r == 0 and proposal is not None:
                self.generator.proposal = proposal
            
            elif self.round > 1:
                # posterior becomes new proposal prior
                # choose specific observation. It is either fixed, or changes per round.
                proposal = self.predict(obs)

                # convert proposal to student's T?
                if self.convert_to_T is not None:
                    if type(self.convert_to_T) == int:
                        dofs = self.convert_to_T
                    else:
                        dofs = 10
                    proposal = proposal.convert_to_T(dofs=dofs)

                self.generator.proposal = proposal

            # Get number of samples to generate.
            if type(n_train) == list:
                try:    n_train_round = n_train[self.round-1]
                except: n_train_round = n_train[-1]
            else:
                n_train_round = n_train

            if verbose: print('\t', n_train_round, 'samples requested.')

            if initial_tds is not None:
                assert initial_tds[0].shape[0] == initial_tds[1].shape[0], 'Number of samples must be the same'
                n_loaded_samples = initial_tds[0].shape[0]
                if verbose: print('\t', n_loaded_samples, 'samples given.')
            else:
                n_loaded_samples = 0
            
            n_train_round -= n_loaded_samples
            
            if n_train_round > 0:
                t0 = time.time()
                
                if verbose: print('\tDrawing', n_train_round, 'parameter samples ... ')
                trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=False)
                if verbose: print('\tDone after {:.4g} min'.format((time.time()-t0)/60))
                  
                if initial_tds is not None:
                  trn_data = (np.concatenate((initial_tds[0], trn_data[0])),
                              np.concatenate((initial_tds[1], trn_data[1])))
            
            else:
                trn_data = initial_tds                
            
            # Update number of samples.
            n_train_round = trn_data[0].shape[0]

            # Precompute importance weights
            if self.generator.proposal is not None:
                params = self.params_std * trn_data[0] + self.params_mean
                p_prior = self.generator.prior.eval(params, log=False)
                p_proposal = self.generator.proposal.eval(params, log=False)
                iws = p_prior / (self.prior_mixin * p_prior + (1 - self.prior_mixin) * p_proposal)
            else:
                iws = np.ones((n_train_round,))
            
            # normalize weights
            iws /= np.mean(iws)
            
            # Get training values.
            use_pseudo_obs = (self.pseudo_obs_perc is not None) or (self.pseudo_obs_n is not None)
            use_adap_kernel = (self.kernel_bandwidth_perc is not None) or (self.kernel_bandwidth_n is not None)
            
            if use_pseudo_obs or use_adap_kernel:
                pseudo_obs_tds = trn_data[1]
                if self.pseudo_obs_use_all_data:
                    pseudo_obs_tds = np.concatenate([pseudo_obs_tds] + [tds_i[1] for tds_i in trn_datasets])                
            
            # Get observed or pseudo-observed value.
            if use_pseudo_obs:
                pseudo_obs = self.compute_pseudo_obs(
                    obs=self.obs, pseudo_obs_dim=self.pseudo_obs_dim, tds=pseudo_obs_tds,
                    q=self.pseudo_obs_perc, n=self.pseudo_obs_n,
                )
                self.pseudo_obs.append(pseudo_obs) # Save.
                network_eval_obs = pseudo_obs.copy()
            else:
                pseudo_obs = None
                network_eval_obs = self.obs.copy()
            

            # Update importance weights based on kernel.
            if self.kernel is not None:
                if use_adap_kernel:
                    self.adapt_kernel_bandwidth(pseudo_obs_tds=pseudo_obs_tds, pseudo_obs=pseudo_obs)
                
                # Compute importance weights with kernel.    
                iws *= self.kernel.eval(trn_data[1].reshape(n_train_round, -1))
            
            # Add importance weights to data.
            trn_data = (trn_data[0], trn_data[1], iws)
              
            if verbose:
                t0 = time.time()
                np.set_printoptions(precision=3)
                print('\tTraining network with observed:')
                print('\t', network_eval_obs.flatten())
                print('\tand kernel bandwidth:')
                print('\t', self.kernel.bandwidth)
            
            # Train network.
            trn_inputs = [self.network.params, self.network.stats, self.network.iws]

            if self.use_doubling:
                print('\tDuplicate all discrepancies (pos and neg)')
                assert np.all(trn_data[1] >= 0)
                trn_data=(np.tile(trn_data[0], (2, 1)), np.concatenate([trn_data[1], trn_data[1]*(-1)]), np.tile(iws, 2)*0.5)
                n_train_round *= 2
    
            t = Trainer(self.network,
                        self.loss(N=n_train_round, round_cl=round_cl),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        seed=self.gen_newseed(),
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
    
            logs.append(t.train(epochs=epochs, minibatch=minibatch, verbose=False, stop_on_nan=stop_on_nan))

            if verbose:
                print('\tDone after {:.4g} min'.format((time.time()-t0)/60))
                                
            trn_datasets.append(trn_data)
            
            try:
                posteriors.append(self.predict(network_eval_obs))
            except np.linalg.LinAlgError:
                posteriors.append(None)
                print("Cannot predict posterior after round {} due to NaNs".format(r))

        return logs, trn_datasets, posteriors
