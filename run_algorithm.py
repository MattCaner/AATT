import annealing_strategy as a
import os
from pprint import pprint


configpath = "params_local.config"

modificationTable = [a.ModificationFunctions.addDecoderToEndOfStack,
                     a.ModificationFunctions.addEncoderToEndOfStack, 
                     a.ModificationFunctions.removeDecoderFromEndOfStack,
                     a.ModificationFunctions.removeEncoderFromEndOfStack,
                     a.ModificationFunctions.addRandomHead,
                     a.ModificationFunctions.removeRandomHead
                    ]

modificationChances = [0.2,
                       0.2,
                       0.2,
                       0.2,
                       0.2,
                       0.2
                    ]

strategy = a.AnnealingStrategy(8,5,2,0.9,configpath,modificationTable,modificationChances,test_mode=True)
strategy.run()
print('----finished----------------')
bleu = strategy.performBleuMetrics()
rogue = strategy.performRogueMetrics()
print(bleu)
pprint(rogue)
print('finished')