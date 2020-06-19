# Author: Janek Groß

import numpy as np
import pickle

class evolutionary_strategies_model(object):
    def __init__(
        self, n_population, n_params, n_survival,
        n_crossover = 2, sigma_init = 1, mu_init = 0, tau = None):
        """
        Evolutionary strategies model loosely based on
        Beyer and Schwefel, 2002, Evolution strategies - A Comprehensive Introduction
        Model type (in the notation from the paper): (mu/ro, lambda) where 
        mu = n_survival
        ro = n_crossover
        lambda = n_population
        
        Parameters
        ----------
        n_population  : integer
                        number of instances that are created each generation
        n_params      : integer
                        dimension of the parameter space to optimize
        n_survival    : integer
                        number of instances to be selected each generation
        n_crossover   : integer
                        number of parent instances for each new child usually 2
        sigma_init    : integer
                        standard deviation for the normal distribution the 
                        mutation term is sampled from at the start
        mu_init       : integer
                        starting value for parameters
        tau           : float
                        learning rate like parameter
                        default (if None): tau = 1/sqrt(2*n_population)
        """
        assert sigma_init > 0
        assert n_population > n_survival
        assert n_population % n_crossover == 0
        assert n_population % n_survival == 0
        
        self.n_population = n_population
        self.n_survival = n_survival
        self.sigma_init = sigma_init
        self.n_crossover = n_crossover
        if tau == None:
            self.tau = 1/((2*n_population)**0.5)
        else: self.tau = tau
        self.n_params = n_params
        self.params = np.random.normal(mu_init, sigma_init, (n_population, n_params))
        self.sigmas = np.full((n_population, n_params), sigma_init, dtype = 'float64')
        self.fitness = np.zeros(n_population)
        self.indices_fittest = None
        
    def mutate(self):
        """
        mutate parameters          : x = N(x,sigma)
        mutate standard deviations : sigma = sigma * exp(N(0,tau))
        """
        self.params = np.random.multivariate_normal(
            self.params.reshape(self.n_population * self.n_params),
            np.diag(self.sigmas.reshape(self.n_population * self.n_params)))\
            .reshape((self.n_population, self.n_params))

        self.sigmas *= np.exp(np.random.multivariate_normal(
            np.zeros(self.n_population * self.n_params),
            self.tau * np.eye(self.n_population * self.n_params)))\
            .reshape((self.n_population, self.n_params))
            
    def select(self):
        """
        retreive the indices of the n_survival best instances
        """
        self.indices_fittest = np.argsort(self.fitness)[-self.n_survival:]

    def procreate(self):
        """
        Create n_population new instances from the fittest instances of
        the current generation.
        Parent groups are selected randomly.
        Parameters and sigmas of n_crossover parents are shuffled to create
        n_crossover children per parent group.
        """
        n_children = self.n_population // self.n_survival
            
        parent_list = np.tile(self.indices_fittest, n_children)
        np.random.shuffle(parent_list)

        next_generation_params = self.params[parent_list,:]
        next_generation_sigmas = self.sigmas[parent_list,:]
        n_groups = self.n_population // self.n_crossover
        
        for group in range(n_groups):
            for i in range(self.n_params):
                np.random.shuffle(
                next_generation_params[
                    group * self.n_crossover : (group + 1) * self.n_crossover,i])
                np.random.shuffle(
                next_generation_sigmas[
                    group * self.n_crossover : (group + 1) * self.n_crossover,i])
        self.params = next_generation_params
        self.sigmas = next_generation_sigmas
    def save(self):
        """
        create/replace an object file to store the current model.
        """
        filehandler = open("evolutionary_strategies_model", 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()
        print("### saved ###")