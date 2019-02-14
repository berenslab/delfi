import numpy as np
import pickle

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero

class SNPE(BaseInference):
    def __init__(self, generator, obs=None, pseudo_obs_perc=None, psuedo_obs_n=None,
                 kernel_bandwidth_perc=None, kernel_bandwidth_n=None,
                 pesudo_obs_use_all_data=False, prior_norm=False, pilot_samples=100,
                 convert_to_T=3, reg_lambda=0.01, prior_mixin=0, kernel=None, seed=None, verbose=True,
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
        pseudo_obs_perc : double in [0,100]
            If set, adaptively change obs relative to percentile of best samples.
            Set to zero to use best sample only.
        psuedo_obs_n : integer in [1, np.inf]
            If set, adaptively change obs. Set obs always to the n-th best sample.
        kernel_bandwidth_perc : double in [0,100]
            If set, adaptively change kernel bandwidth as percentile of best samples.
        kernel_bandwidth_n : integer in [1, np.inf]
            If set, adaptively change kernel bandwidth relatively to the n-th best sample.
        pesudo_obs_use_all_data : bool
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
        assert pseudo_obs_perc is None or psuedo_obs_n is None, 'Can\'t set both. Use one or none.'
        assert kernel_bandwidth_perc is None or kernel_bandwidth_n is None, 'Can\'t set both. Use one or none.'
        
        self.obs = np.asarray(obs)
        self.pseudo_obs_perc = pseudo_obs_perc
        self.pesudo_obs_use_all_data = pesudo_obs_use_all_data
        self.kernel_bandwidth_perc = kernel_bandwidth_perc
        
        if pseudo_obs_perc is not None:
            self.pseudo_obs = []

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.convert_to_T = convert_to_T

        self.prior_mixin = 0 if prior_mixin is None else prior_mixin

        self.kernel = kernel

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

    def get_obs(self, tds=None):
      """ Get or compute observed value
      
      Parameters:
      -----------
      
      tds: array
          Only needed when obs is computed as percentile of current samples.
          Will then be used to compute the pseudo_obs_perc percentile of the samples relative
          to the true observed value. Should eventually converge to the true value.
      """
      
      if self.pseudo_obs_perc is None and self.psuedo_obs_n is None:
        if isinstance(self.obs, np.ndarray):
            # Take fixed value.
            return self.obs
        elif isinstance(self.obs, list):
            # Take element from round. If more rounds than elements, take last.
            return self.obs[np.min([len(self.obs)-1, r-1])]
      else:
          assert isinstance(self.obs, np.ndarray)
          abs_tds = np.abs(tds - self.obs)
          if self.pseudo_obs_perc is not None:
              # Take percentile of samples.
              obs = abs_tds[np.argsort(abs_tds.flatten())[int(np.round(self.pseudo_obs_perc/100*abs_tds.shape[0]))]]
              return np.reshape(obs, self.obs.shape)
          elif self.psuedo_obs_n is not None:
              # Take n-th best sample.
              obs = abs_tds[np.argsort(abs_tds.flatten())[self.psuedo_obs_n]]
              return np.reshape(obs, self.obs.shape)
        
    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            round_cl=1, stop_on_nan=False, proposal=None, text_verbose=True,
            monitor=None, load_trn_data=False, save_trn_data=False, append_trn_data=False,
            init_trn_data_file=None, verbose=False, **kwargs):

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
        proposal : Distribution of None
            If given, will use this distribution as the starting proposal prior
        text_verbose: bool
            if True, simple print output for the progress
        load_trn_data:bool
            If True, load tds from specified file
        save_trn_data: bool
            If True, save tds to specified file
        append_trn_data: bool
            if True draws n_train new trainingsdata and appends it to the loaded tds
        init_trn_data_file: None or filepath
            if filepath loads/saves the trainingsdata of this file


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

        if load_trn_data or save_trn_data:
            assert init_trn_data_file is not None, 'If you want to load or save data, please state a file'
        if append_trn_data and not(load_trn_data):
            print('Will not append since loading is not set to true.')
        
        for r in range(n_rounds):
            # Define what to do this round.
            # Load data only in first round, and only if flag is set.
            r_load_data = (r==0) and load_trn_data
            # Draw new samples in every round. Not in the first however if data was loaded and should not be appended.
            generate_data = (r > 0) or not(load_trn_data) or append_trn_data
        
            self.round += 1
            if text_verbose: print('Round: ' + str(r))
            
            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False
            
            if r == 0 and proposal is not None:
                self.generator.proposal = proposal
            # if round > 1, set new proposal distribution before sampling
            elif self.round > 1:
                # posterior becomes new proposal prior
                # choose specific observation. It is either fixed, or changes per round.
                proposal = self.predict(obs) # see super

                # convert proposal to student's T?
                if self.convert_to_T is not None:
                    if type(self.convert_to_T) == int:
                        dofs = self.convert_to_T
                    else:
                        dofs = 10
                    proposal = proposal.convert_to_T(dofs=dofs)

                self.generator.proposal = proposal

            # Loading trainind from previous trainings. Only samples from the prior distribution are loaded.
            if r_load_data:
                with open(init_trn_data_file + '.pkl', 'rb') as f:
                    initial_trn_data = pickle.load(f)
                assert initial_trn_data[0].shape[0] == initial_trn_data[1].shape[0], 'Number of samples must be the same'
                assert initial_trn_data[0].shape[0] == initial_trn_data[2].size, 'Number of samples must be the same'

                n_train_round = initial_trn_data[0].shape[0]
                trn_data = initial_trn_data
                if text_verbose: print('Used initial training data.')
                if append_trn_data: loaded_trn_data = trn_data
                else:               loaded_trn_data = None

            # Generate samples.
            if generate_data:            
                if type(n_train) == list:
                    try:
                        n_train_round = n_train[self.round-1]
                    except:
                        n_train_round = n_train[-1]
                else:
                    n_train_round = n_train       
          
                if text_verbose: print('\t Sampling ' + str(n_train_round) + ' samples')
                trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=verbose, from_prior=(r==0))
                n_train_round = trn_data[0].shape[0]

                # precompute importance weights
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
            perc_tds = trn_data[1]
            if self.pesudo_obs_use_all_data:
                perc_tds = np.concatenate([perc_tds] + [tds_i[1] for tds_i in trn_datasets])                
            
            # Get observed or pseudo-observed value.
            obs = self.get_obs(perc_tds)
            if self.pseudo_obs_perc is not None: self.pseudo_obs.append(obs)
                
            if self.verbose or text_verbose: print('Observed = ' + str(obs))
            
            # Continue generating data?
            if generate_data:
                
                if self.kernel is not None:
                    self.kernel.obs = obs # Update observed in kernel.
                    if self.kernel_bandwidth_perc is not None or self.kernel_bandwidth_n is not None:
                        abs_tds = np.abs(perc_tds - self.obs)
                        if self.kernel_bandwidth_perc is not None:
                            # Compute percentile.
                            bandwidth_tot = abs_tds[np.argsort(abs_tds.flatten())[int(np.round(self.kernel_bandwidth_perc/100*abs_tds.shape[0]))]]
                        elif self.kernel_bandwidth_n is not None:
                            # Compute n-th best sample.
                            bandwidth_tot = abs_tds[np.argsort(abs_tds.flatten())[self.kernel_bandwidth_n]]
                        
                        # Subtract current obs value.
                        bandwidth_rel = bandwidth_tot - obs
                        if self.verbose or text_verbose: print('New bandwidth = ' + str(bandwidth_rel))
                        self.kernel.set_bandwidth(bandwidth_rel)
                        
                    iws *= self.kernel.eval(trn_data[1].reshape(n_train_round, -1))

                trn_data = (trn_data[0], trn_data[1], iws)

                # Given data should be appended, combine old trn_data and new trn_data.
                if append_trn_data:
                    trn_data = (np.concatenate((loaded_trn_data[0], trn_data[0])),
                                np.concatenate((loaded_trn_data[1], trn_data[1])),
                                np.concatenate((loaded_trn_data[2], trn_data[2]))) 
                    n_train_round = trn_data[0].shape[0]

                # Save data sampled from prior for future use.
                if r == 0 and save_trn_data:
                    with open(init_trn_data_file + '.pkl', 'wb') as f:
                        pickle.dump(trn_data, f, pickle.HIGHEST_PROTOCOL)

            if text_verbose: print('\t Training network ... ', end='')
            trn_inputs = [self.network.params, self.network.stats,
                          self.network.iws]

            t = Trainer(self.network,
                        self.loss(N=n_train_round, round_cl=round_cl),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        seed=self.gen_newseed(),
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
            logs.append(t.train(epochs=epochs, minibatch=minibatch,
                                verbose=verbose, stop_on_nan=stop_on_nan))

            trn_datasets.append(trn_data)
            if text_verbose: print('Done!')
            
            try:
                posteriors.append(self.predict(obs))
            except np.linalg.LinAlgError:
                posteriors.append(None)
                print("Cannot predict posterior after round {} due to NaNs".format(r))
                break

        return logs, trn_datasets, posteriors
