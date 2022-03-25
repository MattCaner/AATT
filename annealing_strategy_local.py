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
from statistics import mean


def addRandomHead(transformer: t.Transformer) -> t.Transformer:
    total_attentions = len(transformer.encoder_stack.encoders) + 2*len(transformer.decoder_stack.decoders)
    location = random.randint(0,total_attentions-1)
    if location < len(transformer.encoder_stack.encoders):
        random_head_id = random.randint(0,len(transformer.encoder_stack.encoders[location].mha.heads)-1)
        transformer.encoder_stack.encoders[location].mha.heads.insert(random_head_id,t.AttentionHead(transformer.config,d_v_override=transformer.encoder_stack.encoders[location].mha.d_v))
        transformer.encoder_stack.encoders[location].mha.n_heads += 1
        transformer.encoder_stack.encoders[location].mha.linear = nn.Linear(transformer.encoder_stack.encoders[location].mha.d_v*transformer.encoder_stack.encoders[location].mha.n_heads,transformer.encoder_stack.encoders[location].mha.d_model)
    else:
        location -= len(transformer.encoder_stack.encoders)
        if location % 2 == 0:
            location = int(location/2)
            random_head_id = random.randint(0,len(transformer.decoder_stack.decoders[location].self_mha.heads)-1)
            transformer.decoder_stack.decoders[location].self_mha.heads.insert(random_head_id,t.AttentionHead(transformer.config,masked=True,d_v_override=transformer.decoder_stack.decoders[location].self_mha.d_v))
            transformer.decoder_stack.decoders[location].self_mha.n_heads += 1
            transformer.decoder_stack.decoders[location].self_mha.linear = nn.Linear(transformer.decoder_stack.decoders[location].self_mha.d_v*transformer.decoder_stack.decoders[location].self_mha.n_heads,transformer.decoder_stack.decoders[location].self_mha.d_model)
        else:
            location = int(location/2)
            random_head_id = random.randint(0,len(transformer.decoder_stack.decoders[location].ed_mha.heads)-1)
            transformer.decoder_stack.decoders[location].ed_mha.heads.insert(random_head_id,t.AttentionHead(transformer.config,d_v_override=transformer.decoder_stack.decoders[location].ed_mha.d_v))
            transformer.decoder_stack.decoders[location].ed_mha.n_heads += 1
            transformer.decoder_stack.decoders[location].ed_mha.linear = nn.Linear(transformer.decoder_stack.decoders[location].ed_mha.d_v*transformer.decoder_stack.decoders[location].ed_mha.n_heads,transformer.decoder_stack.decoders[location].ed_mha.d_model)

    return transformer

def removeRandomHead(transformer: t.Transformer) -> t.Transformer:
    total_attentions = len(transformer.encoder_stack.encoders) + 2*len(transformer.decoder_stack.decoders)
    location = random.randint(0,total_attentions-1)
    if location < len(transformer.encoder_stack.encoders):
        if len(transformer.encoder_stack.encoders[location].mha.heads) > 1:
            random_head_id = random.randint(0,len(transformer.encoder_stack.encoders[location].mha.heads)-1)
            transformer.encoder_stack.encoders[location].mha.heads.pop(random_head_id)
            transformer.encoder_stack.encoders[location].mha.n_heads -= 1
            transformer.encoder_stack.encoders[location].mha.linear = nn.Linear(transformer.encoder_stack.encoders[location].mha.d_v*transformer.encoder_stack.encoders[location].mha.n_heads,transformer.encoder_stack.encoders[location].mha.d_model)
    else:
        location -= len(transformer.encoder_stack.encoders)
        if location % 2 == 0:
            location = int(location/2)
            if len(transformer.decoder_stack.decoders[location].self_mha.heads) > 1:
                random_head_id = random.randint(0,len(transformer.decoder_stack.decoders[location].self_mha.heads)-1)
                transformer.decoder_stack.decoders[location].self_mha.heads.pop(random_head_id)
                transformer.decoder_stack.decoders[location].self_mha.n_heads -= 1
                transformer.decoder_stack.decoders[location].self_mha.linear = nn.Linear(transformer.decoder_stack.decoders[location].self_mha.d_v*transformer.decoder_stack.decoders[location].self_mha.n_heads,transformer.decoder_stack.decoders[location].self_mha.d_model)
        else:
            location = int(location/2)
            if len(transformer.decoder_stack.decoders[location].ed_mha.heads) > 1:
                random_head_id = random.randint(0,len(transformer.decoder_stack.decoders[location].ed_mha.heads)-1)
                transformer.decoder_stack.decoders[location].ed_mha.heads.pop(random_head_id)
                transformer.decoder_stack.decoders[location].ed_mha.n_heads -= 1
                transformer.decoder_stack.decoders[location].ed_mha.linear = nn.Linear(transformer.decoder_stack.decoders[location].ed_mha.d_v*transformer.decoder_stack.decoders[location].ed_mha.n_heads,transformer.decoder_stack.decoders[location].ed_mha.d_model)

    return transformer

def changeFFDimensions(transformer: t.Transformer) -> t.Transformer:
    total_attentions = len(transformer.encoder_stack.encoders) + len(transformer.decoder_stack.decoders)
    location = random.randint(0,total_attentions-1)
    if location < len(transformer.encoder_stack.encoders):
        old_dim = transformer.encoder_stack.encoders[location].d_ff
        dim_model = transformer.d_model
        ubound = 1
        lbound = -1
        if old_dim < 2:
            lbound = 1
        change = random.randint(lbound,ubound)
        changed = old_dim+change
        transformer.encoder_stack.encoders[location].feed_forward = nn.Sequential(nn.Linear(dim_model,changed),nn.ReLU(),nn.Linear(changed,dim_model))
    else:
        location -= len(transformer.encoder_stack.encoders)
        old_dim = transformer.decoder_stack.decoders[location].d_ff
        dim_model = transformer.d_model
        ubound = 1
        lbound = -1
        if old_dim < 2:
            lbound = 1
        change = random.randint(lbound,ubound)
        changed = old_dim+change
        transformer.decoder_stack.decoders[location].feed_forward = nn.Sequential(nn.Linear(dim_model,changed),nn.ReLU(),nn.Linear(changed,dim_model))
    return transformer

def changeDimensions(transformer: t.Transformer) -> t.Transformer:

    total_attentions = len(transformer.encoder_stack.encoders) + 2*len(transformer.decoder_stack.decoders)
    # change a single attention dimensions:
    location = random.randint(0,total_attentions-1)
    if location < len(transformer.encoder_stack.encoders):
        act_location = transformer.encoder_stack.encoders[location]
        random_head_id = random.randint(0,len(act_location.mha.heads)-1)
        dim = act_location.mha.d_v
        lbound = -1
        ubound = 1
        if dim >= transformer.config.provide("d_model"):
            ubound = 0
        if dim <= 2:
            lbound = 0
        change = random.randint(lbound,ubound)
        act_location.mha = t.MultiHeadedAttention(transformer.config,masked=act_location.mha.masked,d_v_override=dim+change)

    else:
        location -= len(transformer.encoder_stack.encoders)
        if location % 2 == 0:
            location = int(location/2)
            act_location = transformer.decoder_stack.decoders[location]
            random_head_id = random.randint(0,len(act_location.self_mha.heads)-1)
            dim = act_location.self_mha.d_v
            lbound = -1
            ubound = 1
            if dim >= transformer.config.provide("d_model"):
                ubound = 0
            if dim <= 2:
                lbound = 0
            change = random.randint(lbound,ubound)
            act_location.self_mha = t.MultiHeadedAttention(transformer.config,masked=act_location.self_mha.masked,d_v_override=dim+change)
        else:
            location = int(location/2)
            act_location = transformer.decoder_stack.decoders[location]
            random_head_id = random.randint(0,len(act_location.ed_mha.heads)-1)
            dim = act_location.ed_mha.d_v
            lbound = -1
            ubound = 1
            if dim >= transformer.config.provide("d_model"):
                ubound = 0
            if dim <= 2:
                lbound = 0
            change = random.randint(lbound,ubound)
            act_location.ed_mha = t.MultiHeadedAttention(transformer.config,masked=act_location.ed_mha.masked,d_v_override=dim+change)
    return transformer

def addEncoder(transformer: t.Transformer) -> t.Transformer:
    # add an encoder:
    pos = random.randint(0,len(transformer.encoder_stack.encoders)-1)
    transformer.encoder_stack.encoders.insert(pos,t.EncoderLayer(transformer.config))
    return transformer

def removeEncoder(transformer: t.Transformer) -> t.Transformer:
    # remove an encoder:
    if len(transformer.encoder_stack.encoders) > 1:
        encoder_to_remove = random.randint(0,len(transformer.encoder_stack.encoders)-1)
        transformer.encoder_stack.encoders.pop(encoder_to_remove)
    return transformer

def addDecoder(transformer: t.Transformer) -> t.Transformer:
    # add a decoder:
    pos = random.randint(0,len(transformer.decoder_stack.decoders)-1)
    transformer.decoder_stack.decoders.insert(pos,t.DecoderLayer(transformer.config))
    return transformer

def removeDecoder(transformer: t.Transformer) -> t.Transformer:
    # remove a decoder:
    if len(transformer.decoder_stack.decoders) > 1:
        decoder_to_remove = random.randint(0,len(transformer.decoder_stack.decoders)-1)
        transformer.decoder_stack.decoders.pop(decoder_to_remove)
    return transformer

mutations_table = [addRandomHead, removeRandomHead, changeDimensions, changeFFDimensions, addEncoder, addDecoder, removeDecoder]
mutation_chances = [1.0/len(mutations_table) for _ in range(len(mutations_table))]

class Solution:
    def __init__(self, transformer: t.Transformer, result: float):
        self.transformer = transformer
        self.result = result

class AnnealingStrategyLocal():

    def __init__(self, num_threads: int, max_iters: int, best_update_interval: int , alpha: float, configFile: str, test_mode = False):
        self.num_threads = num_threads
        self.max_iters = max_iters
        self.best_update_interval = best_update_interval
        self.alpha = alpha
        self.solutions_list = []

        self.general_params = t.ParameterProvider(configFile)

        self.v_in = t.VocabProvider(self.general_params,self.general_params.provide("language_in_file"))
        self.v_out = t.VocabProvider(self.general_params,self.general_params.provide("language_out_file"))
        self.cd = t.CustomDataSet(self.general_params.provide("language_in_file"), self.general_params.provide("language_out_file"),self.v_out)
        self.train_dataset, self.test_dataset = self.cd.getSets()
        if test_mode:
            self.test_dataset = self.train_dataset

    def NeighbourOperator(self, transformer: t.Transformer, operationsMemory: List = None) -> t.Transformer:
        newtransformer = copy.deepcopy(transformer)
        performed_mutations = 0
        for i, func in enumerate(mutations_table):
            if random.uniform(0.,1.) < mutation_chances[i]:
                operationsMemory.append("Operation id "+str(i))
                performed_mutations += 1
                func(newtransformer)
        
        if performed_mutations == 0:
            operationId = random.randint(0,len(mutations_table)-1)
            operationsMemory.append("Operation id "+str(operationId))
            mutations_table[operationId](newtransformer)

        return newtransformer

    def performAnnealing(self, thread_number: int, solutions: List[Solution], T: float, operationsMemory: List) -> None:
        old_fitness = solutions[thread_number].result
        new_transformer = self.NeighbourOperator(solutions[thread_number].transformer, operationsMemory)
        #t.train(new_transformer,self.train_dataset,new_transformer.config.provide("learning_rate"),new_transformer.config.provide("epochs"))
        #new_fitness = t.evaluate(new_transformer,self.test_dataset)
        t.train_until_difference(new_transformer,self.train_dataset,0.005,lr=new_transformer.config.provide("learning_rate"),max_epochs=new_transformer.config.provide("epochs"))
        new_fitness = t.evaluate(new_transformer,self.test_dataset)
        if new_fitness < old_fitness:
            solutions[thread_number] = Solution(new_transformer,new_fitness)
        else:
            if random.uniform(0,1) < math.exp(-1.*(new_fitness - old_fitness) / T):
                solutions[thread_number] = Solution(new_transformer,new_fitness)

    def generateTemperature(self,initial_solutions: List) -> float:
        return mean(i.result for i in initial_solutions)

    #mostly placeholder
    def generateInitialSolutions(self) -> List[Solution]:
        s = []
        def generateSingleSolution():
            transformer = t.Transformer(self.general_params,self.v_in,self.v_out)
            #t.train(transformer,self.train_dataset,transformer.config.provide("learning_rate"),transformer.config.provide("epochs"))
            #result = t.evaluate(transformer,self.test_dataset)
            t.train_until_difference(transformer,self.train_dataset,0.005,lr=transformer.config.provide("learning_rate"),max_epochs=10)
            result = t.evaluate(transformer,self.test_dataset)
            s.append(Solution(transformer,result))
            
        thread_list = [threading.Thread(target=generateSingleSolution) for th in range(0,self.num_threads)]
        for th in thread_list:
            th.start()
        for th in thread_list:
            th.join()

        return s

    def run(self) -> None:

        print("preparing initial solution")
        solutions_list = self.generateInitialSolutions()
        temperature = self.generateTemperature(solutions_list)

        operations_memory = [[] for _ in range(self.num_threads)]

        global_best_index = max(range(len(solutions_list)), key=lambda i: solutions_list[i].result)
        global_best_solution = solutions_list[global_best_index]



        for i in range(0,self.max_iters):
            print("annealing epoch: ",i, " temperature: ", temperature)
            thread_list = [threading.Thread(target=self.performAnnealing, args=(th,solutions_list,temperature,operations_memory[th])) for th in range(0,self.num_threads)]
            for th in thread_list:
                th.start()
            for th in thread_list:
                th.join()
            
            best_solution_index = max(range(len(solutions_list)), key=lambda i: solutions_list[i].result)
            best_solution = solutions_list[best_solution_index]
            if best_solution.result < global_best_solution.result:
                global_best_solution = best_solution

            print("best epoch result: ",best_solution.result)

            temperature *= self.alpha

            if i % self.best_update_interval == 0:
                for j in range(self.num_threads):
                    solutions_list[j] = global_best_solution



strategy = AnnealingStrategyLocal(num_threads=4,max_iters=50,best_update_interval=10,alpha=0.9,configFile="benchmark.config")

strategy.run()