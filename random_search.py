from unittest import result
import transformer as t
import random
from typing import List
from torch import nn
import math
import copy
import threading
from statistics import mean
from torch import cuda
import time
import pickle
from enum import Enum



class Solution:
    def __init__(self, transformer: t.Transformer, result: float):
        self.transformer = transformer
        self.result = result


class Mode(Enum):
    DEEPCOPY = 1,
    CONFIGONLY = 2

class RandomSearch:
    def __init__(self, num_threads: int, max_iters: int, configFile: str, test_mode: bool = True, csv_output: str = 'out.csv', result_output: str = 'result'):
        self.num_threads = num_threads
        self.max_iters = max_iters
        self.epochs_number = []
        self.csv_output = csv_output
        self.result_output = result_output


        self.general_params = t.ParameterProvider(configFile)


        self.v_in = t.VocabProvider(self.general_params,self.general_params.provide("language_in_file"))
        self.v_out = t.VocabProvider(self.general_params,self.general_params.provide("language_out_file"))
        self.cd = t.CustomDataSet(self.general_params.provide("language_in_file"), self.general_params.provide("language_out_file"),self.v_in,self.v_out)
        self.train_dataset, self.test_dataset = self.cd.getSets()
        if test_mode:
            self.test_dataset = self.train_dataset



    def NeighbourOperator(self, transformer: t.Transformer) -> t.Transformer:

        config = transformer.config
        params = config.getArray()
        
        for i, p in enumerate(params):
            modifyBy = max(int(2.*p),1)
            params[i] += random.randint(-modifyBy,modifyBy)
            if params[i] < 1:
                params[i] = 1

        newconfig = copy.deepcopy(config)
        newconfig.modifyWithArray(params)

        newtransformer = t.Transformer(newconfig,transformer.vocab_in,transformer.vocab_out)

        return newtransformer

    def performEpochThread(self, thread_number: int, previous_solution: Solution, new_solutions: List[Solution]):
        new_transformer = self.NeighbourOperator(previous_solution.transformer)
        res, epochs = t.train_cuda(new_transformer, self.train_dataset, cuda.current_device(), batch_size = 32, lr = new_transformer.config.provide("learning_rate"), epochs = 50)
        new_fitness = t.evaluate(new_transformer,self.test_dataset,use_cuda=True,device=cuda.current_device())
        new_solutions.append(Solution(new_transformer,new_fitness))
        
    def run(self, runs: int):

        initial_t = t.Transformer(self.general_params, self.v_in, self.v_out)
        res, epochs = t.train_cuda(initial_t, self.train_dataset, cuda.current_device(), batch_size = 32, lr = initial_t.config.provide("learning_rate"), epochs = 50)
        new_fitness = t.evaluate(initial_t,self.test_dataset,use_cuda=True,device=cuda.current_device())
        best_solution = Solution(initial_t,new_fitness)
        previous_solution = best_solution

        for epoch_number in range(runs):

            new_solutions = []
            thread_list = [threading.Thread(target=self.performEpochThread, args=(th,previous_solution,new_solutions)) for th in range(0,self.num_threads)]
            for th in thread_list:
                th.start()
            for th in thread_list:
                th.join()

            best_solution_index = min(range(len(new_solutions)), key=lambda i: new_solutions[i].result)
            best_solution = new_solutions[best_solution_index]

            bestfile = open(str(self.result_output) + '-epoch' + str(epoch_number) + '.pydump','wb')
            pickle.dump(best_solution,bestfile)
            bestfile.close()
        
        return best_solution

    