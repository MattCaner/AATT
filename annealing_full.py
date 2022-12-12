import transformer as t
from annealing_strategy_global import AnnealingStrategyGlobal
from annealing_strategy_local import AnnealingStrategyLocal
import os
'''
configpath = os.environ["SCRATCH"] + "/params.config"
outpath = os.environ["SCRATCH"]
'''
configpath = "params_local.config"
outpath = ""


def full_solution():
    firstStrategy = AnnealingStrategyGlobal(num_threads=8,max_iters=5,best_update_interval=1,alpha=0.9,test_mode=False,configFile=configpath,csv_output=outpath+"dump_global/global_out.csv", result_output=outpath+"/global_dump")
    
    firstStrategy.run()

    bestConfig = firstStrategy.global_best_solution.transformer.config.getArray()

    secondStrategy = AnnealingStrategyLocal(num_threads=8,max_iters=50,best_update_interval=10,alpha=0.9,test_mode=False,configFile=configpath,csv_output=outpath+"dump_local/local_out.csv", result_output=outpath+"/local_dump")
    secondStrategy.general_params.modifyWithArray(bestConfig)

    secondStrategy.run()


full_solution()