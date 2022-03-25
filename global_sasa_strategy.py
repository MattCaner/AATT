from statistics import mean
import transformer as t
import random
from typing import List
import math
import copy
import threading
from smt.surrogate_models import SurrogateModel, RBF

class Solution:
    def __init__(self, transformer: t.Transformer, result: float):
        self.transformer = transformer
        self.result = result

class AnnealingStrategyGlobalSasa():

    def __init__(self, num_threads: int, max_iters: int, best_update_interval: int , t_train: int, alpha: float, configFile: str, validity_threshold: float = 0.05, test_mode = False):
        self.surrogate_archive = []
        self.num_threads = num_threads
        self.max_iters = max_iters
        self.best_update_interval = best_update_interval
        self.alpha = alpha
        self.solutions_list = []
        self.t_train = t_train
        self.validity_threshold = validity_threshold

        self.general_params = t.ParameterProvider(configFile)

        self.v_in = t.VocabProvider(self.general_params,self.general_params.provide("language_in_file"))
        self.v_out = t.VocabProvider(self.general_params,self.general_params.provide("language_out_file"))
        self.cd = t.CustomDataSet(self.general_params.provide("language_in_file"), self.general_params.provide("language_out_file"),self.v_out)
        self.train_dataset, self.test_dataset = self.cd.getSets()
        if test_mode:
            self.test_dataset = self.train_dataset

    def NeighbourOperator(self, transformer: t.Transformer, operationsMemory: List = None) -> t.Transformer:
        config = transformer.config

        params = config.getArray()

        for i in range(len(params)):
            if params[i] > 1:
                params[i] += random.randint(-1,2)
            else:
                params[i] += random.randint(params[i]-1,2)
        
        newconfig = copy.deepcopy(config)

        newconfig.modifyWithArray(params)

        newtransformer = t.Transformer(newconfig,transformer.vocab_in,transformer.vocab_out)
        
        return newtransformer

    def performAnnealing(self, thread_number: int, solutions: List[Solution], T: float, operationsMemory: List) -> None:
        #print("Started thread: ", thread_number)
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

    def approximate(self, thread_number: int, surrogate: SurrogateModel, solutions: List[Solution], T: float, operationsMemory: List) -> None:
        old_fitness = solutions[thread_number].result
        new_transformer = self.NeighbourOperator(solutions[thread_number].transformer, operationsMemory)
        new_fitness = surrogate.predict_values([new_transformer.config.getArray()])[0]
        if new_fitness < old_fitness:
            solutions[thread_number] = Solution(new_transformer,new_fitness)
        else:
            if random.uniform(0,1) < math.exp(-1.*(new_fitness - old_fitness) / T):
                solutions[thread_number] = Solution(new_transformer,new_fitness)

    def generateTemperature(self,initial_solutions: List) -> float:
        return mean(i.result for i in initial_solutions)


    def generateInitialSolutions(self) -> List[Solution]:
        s = []
        for _ in range(self.num_threads):
            transformer = t.Transformer(self.general_params,self.v_in,self.v_out)
            #t.train(transformer,self.train_dataset,transformer.config.provide("learning_rate"),transformer.config.provide("epochs"))
            #result = t.evaluate(transformer,self.test_dataset)
            t.train_until_difference(transformer,self.train_dataset,0.005,lr=transformer.config.provide("learning_rate"),max_epochs=transformer.config.provide("epochs"))
            result = t.evaluate(transformer,self.test_dataset)
            s.append(Solution(transformer,result))
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
            
            surrogation_valid = False

            if i % self.t_train:
                #create samples for surogation:
                points_array = [s.transformer.config.getArray() for s in self.surrogate_archive]
                results_array = [s.result for s in self.surrogate_archive]

                surrogate = RBF()
                surrogate.set_training_values(points_array, results_array)
                surrogate.train()
                prediction = surrogate.predict_values(points_array)

                p_err = 1./(max(results_array) - min(results_array)) * math.sqrt(1./len(results_array)*sum([(prediction[i]-results_array[i]) for i in range(len(results_array))]))

                if p_err < self.validity_threshold:
                    surrogation_valid = True
                    thread_list = [threading.Thread(target=self.approximate, args=(th,surrogate,solutions_list,temperature,operations_memory[th])) for th in range(0,self.num_threads)]
                    for th in thread_list:
                        th.start()
                    for th in thread_list:
                        th.join()

            if surrogation_valid == False:
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



strategy = AnnealingStrategyGlobalSasa(num_threads=8,max_iters=50,best_update_interval=10,alpha=0.9,configFile="benchmark.config")

strategy.run()