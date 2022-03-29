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

general_params = t.ParameterProvider("benchmark.config")
v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_in, v_out)
train_dataset, test_dataset = cd.getSets()
#temporary, benchmark-wise(!ONLY!)
test_dataset = train_dataset

transformer = t.Transformer(general_params,v_in,v_out)

if torch.cuda.is_available():
    t.train_cuda(transformer,train_dataset,epochs=10,device=torch.cuda.current_device(),batch_size=3)

#t.train(transformer,train_dataset)

print(t.evaluate(transformer,test_dataset, use_cuda = True, device = torch.cuda.current_device()))