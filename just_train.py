#from global_sasa_strategy import AnnealingStrategyGlobalSasa
import transformer as t
import torch

general_params = t.ParameterProvider("params.config")

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_in,v_out)
train_dataset, test_dataset = cd.getSets()

transformer = t.Transformer()

for i in range(0,50):
    t.train_cuda(transformer, train_dataset, torch.cuda.current_device(), batch_size = 32, lr = 1, epochs = 10)
    print("Evaluation: ")
    quality = t.evaluate(transformer,test_dataset,use_cuda = True, device = torch.cuda.current_device(),batch_size=32)
    print(quality)