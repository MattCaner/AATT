
import random_search as r
import os
from pprint import pprint
'''
configpath = os.environ["SCRATCH"] + "/params.config"
outpath = os.environ["SCRATCH"]
'''

configpath = "params_local.config"


strategy = r.RandomSearch(8,5,configpath,False)
strategy.run(5)
print('----finished----------------')
bleu = strategy.performBleuMetrics()
rogue = strategy.performRogueMetrics()
print(bleu)
pprint(rogue)
print('finished')