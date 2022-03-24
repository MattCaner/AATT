from statistics import mean
import transformer as t
import random
from typing import List
from numpy import average, number
from sqlalchemy import false
from sympy import numer
import torch
import torch.nn.functional as f
from torch import inner, nn
from torchtext.vocab import Vocab
from torch import Tensor
import math
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
import threading



def NeighbourOperator(transformer: t.Transformer, operationsMemory: List = None) -> t.Transformer:
    config = transformer.config

    params = [config.provide("d_model"), config.provide("d_qk"), config.provide("d_v"), config.provide("d_ff"), config.provide("n_heads"), config.provide("n_encoders"), config.provide("n_decoders"), config.provide("epochs")]

    for i in range(len(params)):
        if params[i] > 2:
            params[i] += random.randint(-2,2)
        else:
            params[i] += random.randint(params[i]-2,2)
    
    newconfig = copy.deepcopy(config)

    newconfig.change("d_model",params[0])
    newconfig.change("d_qk",params[1])
    newconfig.change("d_v",params[2])
    newconfig.change("d_ff",params[3])
    newconfig.change("n_encoders",params[4])
    newconfig.change("n_decoders",params[5])    
    newconfig.change("epochs",params[6])    

    newtransformer = t.Transformer(newconfig,transformer.vocab_in,transformer.vocab_out)
    
    return newtransformer

class Solution:
    def __init__(self, transformer: t.Transformer, result: float):
        self.transformer = transformer
        self.result = result

num_threads = 8
max_iters = 200
best_update_interval = 10
alpha = 0.9
solutions_list = []

general_params = t.ParameterProvider("benchmark.config")

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_out)
train_dataset, test_dataset = cd.getSets()
#temporary, benchmark-wise(!ONLY!)
test_dataset = train_dataset

def performAnnealing(thread_number: int, solutions: List[Solution], T: float, operationsMemory: List) -> None:
    #print("Started thread: ", thread_number)
    old_fitness = solutions[thread_number].result
    new_transformer = NeighbourOperator(solutions[thread_number].transformer, operationsMemory)
    new_fitness = t.train_until_difference(new_transformer,train_dataset,test_dataset,0.005,lr=general_params.provide("learning_rate"),max_epochs=10)
    if new_fitness < old_fitness:
        solutions[thread_number] = Solution(new_transformer,new_fitness)
    else:
        if random.uniform(0,1) < math.exp(-1.*(new_fitness - old_fitness) / T):
            solutions[thread_number] = Solution(new_transformer,new_fitness)

def generateTemperature(initial_solutions: List) -> float:
    return mean(i.result for i in initial_solutions)

#mostly placeholder
def generateInitialSolutions() -> List[Solution]:
    s = []
    for _ in range(num_threads):
        transformer = t.Transformer(general_params,v_in,v_out)
        result = t.train_until_difference(transformer,train_dataset,test_dataset,0.005,lr=general_params.provide("learning_rate"),max_epochs=10)
        #result = t.evaluate(transformer,test_dataset)
        s.append(Solution(transformer,result))
    return s

print("preparing initial solution")
solutions_list = generateInitialSolutions()
temperature = generateTemperature(solutions_list)

operations_memory = [[] for _ in range(num_threads)]

global_best_index = max(range(len(solutions_list)), key=lambda i: solutions_list[i].result)
global_best_solution = solutions_list[global_best_index]



for i in range(0,max_iters):
    print("annealing epoch: ",i, " temperature: ", temperature)
    thread_list = [threading.Thread(target=performAnnealing, args=(th,solutions_list,temperature,operations_memory[th])) for th in range(0,num_threads)]
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()
    
    best_solution_index = max(range(len(solutions_list)), key=lambda i: solutions_list[i].result)
    best_solution = solutions_list[best_solution_index]
    if best_solution.result < global_best_solution.result:
        global_best_solution = best_solution

    print("best epoch result: ",best_solution.result)

    temperature *= alpha

    if i % best_update_interval == 0:
        for j in range(num_threads):
            solutions_list[j] = global_best_solution
