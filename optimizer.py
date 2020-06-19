# Author: Janek Groß

import numpy as np

class optimizer():
    def __init__(self, n_generations, fitness_function, verbose = True):
        """
        The optimizer handles the training process
        Parameters
        ----------
        n_generations    : integer
                           number of training epochs
        verbose          : boolean, default: True
                         : print status during training
        """
        self.n_generations = n_generations
        self.verbose = verbose
        self.fitness_function = fitness_function
    def train(self, model):
        """
        train a evolutionary_strategies_model model
        """
        print('### Starting training ###')
        
        for gens in range(self.n_generations):

            self._evaluate_fitness(model)    
            model.select()
            if self.verbose:
                print("### Generation %d/%d ###" % (gens+1, self.n_generations))
                print("# Best parameters #")
                print(model.params[model.indices_fittest,:])
                print("# Mutation rates instances [0 - 4] #")
                print(model.sigmas[0:5])
            elif gens % 20 == 0:
                print("### Generation %d/%d ###" % (gens+1, self.n_generations))
            model.procreate()
            model.mutate()

        print('### Training finished ###')

    def _evaluate_fitness(self, model):
        """
        Use an external function to evalute the fitness
        of the current parameters.
        """
        for i in range(model.n_population):
            model.fitness[i] = self.fitness_function(model.params[i,:])