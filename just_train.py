#from global_sasa_strategy import AnnealingStrategyGlobalSasa
import transformer as t
import torch
from pprint import pprint
import os

#configpath = os.environ["SCRATCH"] + "/params2.config"
configpath = "params_local.config"

general_params = t.ParameterProvider(configpath)

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_in,v_out)
train_dataset, test_dataset = cd.getSets()

transformer = t.Transformer(general_params,v_in,v_out)

lr = 0.001

for i in range(0,1000):
    print("iteration: "+str(i)+"lr: "+str(lr))
    t.train_cuda(transformer, train_dataset, torch.cuda.current_device(), batch_size = 32, lr = lr, epochs = 20)
    print("Evaluation: ", flush=True)
    quality = t.evaluate(transformer,test_dataset,use_cuda = True, device = torch.cuda.current_device(),batch_size=32)
    print(quality,flush=True)

q = t.evaluate(transformer,test_dataset,use_cuda = True, device = torch.cuda.current_device(),batch_size=32)

bleu = t.calculate_bleu(transformer,train_dataset)
rogue = t.calculate_rogue(transformer,train_dataset)

print("----------------------------", flush=True)

f_to_translate = open(general_params.provide("language_in_file"),'r',encoding = 'utf-8')
f_to_compare = open(general_params.provide("language_out_file"),'r',encoding = 'utf-8')
toReturn =  t.raw_data_bleu(transformer,f_to_translate,f_to_compare)
f_to_translate.close()
f_to_compare.close()
print(toReturn)

print("---------------------------", flush = True)

f_to_translate = open(general_params.provide("language_in_file"),'r',encoding = 'utf-8')
f_to_compare = open(general_params.provide("language_out_file"),'r',encoding = 'utf-8')
toReturn =  t.raw_data_rogue(transformer,f_to_translate,f_to_compare)
f_to_translate.close()
f_to_compare.close()
pprint(toReturn)