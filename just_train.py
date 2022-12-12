#from global_sasa_strategy import AnnealingStrategyGlobalSasa
import transformer as t
import torch
from pprint import pprint

general_params = t.ParameterProvider("params_local.config")

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_in,v_out)
train_dataset, test_dataset = cd.getSets()

transformer = t.Transformer(general_params,v_in,v_out)

lr = 0.001

for i in range(0,1):
    print("iteration: "+str(i)+"lr: "+str(lr))
    t.train_cuda(transformer, train_dataset, torch.cuda.current_device(), batch_size = 32, lr = lr, epochs = 100)
    print("Evaluation: ")
    quality = t.evaluate(transformer,test_dataset,use_cuda = True, device = torch.cuda.current_device(),batch_size=32)
    print(quality)

q = t.evaluate(transformer,test_dataset,use_cuda = True, device = torch.cuda.current_device(),batch_size=32)

bleu = t.calculate_bleu(transformer,train_dataset)
rogue = t.calculate_rogue(transformer,train_dataset)


print(bleu)
pprint(rogue)

f = open('simplepl.txt','r',encoding = 'utf-8')
f2 = open('simpleen.txt','r',encoding = 'utf-8')

lines_compare = list(map(str.lower,f2.readlines()))
lines = list(map(str.lower,f.readlines()))


for i, line in enumerate(lines):
    print(transformer.processSentence(line,autoprint=True))
    print(line)
    print(lines_compare[i])
    print("-----------------------")

f.seek(0)
f2.seek(0)

bleu = t.raw_data_bleu(transformer,f,f2)


f.seek(0)
f2.seek(0)

rogue = t.raw_data_rogue(transformer,f,f2)


print(bleu)
pprint(rogue)


