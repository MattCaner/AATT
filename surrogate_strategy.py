import transformer as t
from base64 import encode
from hmac import trans_36
import io
from typing import List
from numpy import number
from sqlalchemy import false
from sympy import numer
import torch
import torch.nn.functional as f
from torch import inner, nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader
import copy
from pySOT.experimental_design import LatinHypercube
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
import numpy as np
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import Ackley
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import GPRegressor
from poap.controller import BasicWorkerThread, ThreadController

from pySOT.optimization_problems import OptimizationProblem

#dimensions:
 
embedding_size = 32
epochs = 3

general_params = t.ParameterProvider("benchmark.config")

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_out)
train_dataset, test_dataset = cd.getSets()

class Problem(OptimizationProblem):
    def __init__(self):

        self.dim = 7
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.array([2,2,2,1,1,1,0.1])
        self.ub = np.array([embedding_size, embedding_size, embedding_size, embedding_size, 64, 64, 5]) 
    
    def eval(self, x):
        config = t.ParameterProvider("params.config")
        config.change("d_model",embedding_size)
        config.change("d_qk",int(x[0]))
        config.change("d_v",int(x[1]))
        config.change("d_ff",int(x[2]))
        config.change("n_heads",int(x[3]))
        config.change("n_encoders",int(x[4]))
        config.change("n_decoders",int(x[5]))
        config.change("learning_rate",x[6])

        network = t.Transformer(config,v_in,v_out)

        t.train(network,train_dataset,config.provide("learning_rate"),epochs)
        print("Evaluated")
        return t.evaluate(network,test_dataset)



lb, ub = np.array([2,2,2,1,1,1,0.1]), np.array([embedding_size, embedding_size, embedding_size, embedding_size, 64, 64, 5])  # Domain is [0, 1]^5

num_threads = 4
max_evals = 50

problem = Problem()
gp = GPRegressor(dim=7, lb=lb, ub=ub)
slhd = SymmetricLatinHypercube(dim=7, num_pts=2 * (7 + 1))

# Create a strategy and a controller
controller = ThreadController()
controller.strategy = SRBFStrategy(
    max_evals=max_evals, opt_prob=problem, exp_design=slhd, surrogate=gp, asynchronous=True, batch_size=num_threads
)

print("Number of threads: {}".format(num_threads))
print("Maximum number of evaluations: {}".format(max_evals))
print("Strategy: {}".format(controller.strategy.__class__.__name__))
print("Experimental design: {}".format(slhd.__class__.__name__))
print("Surrogate: {}".format(gp.__class__.__name__))

# Launch the threads and give them access to the objective function
for _ in range(num_threads):
    print("Starting new thread")
    worker = BasicWorkerThread(controller, problem.eval)
    controller.launch_worker(worker)

# Run the optimization strategy
result = controller.run()

print("Best value found: {0}".format(result.value))
print(
    "Best solution found: {0}\n".format(
        np.array_str(result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
    )
)