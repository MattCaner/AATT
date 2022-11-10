from global_sasa_strategy import AnnealingStrategyGlobalSasa
from annealing_strategy_global import AnnealingStrategyGlobal
from annealing_strategy_local import AnnealingStrategyLocal
import os

configpath = os.environ["SCRATCH"] + "/params_local.config"
outpath = os.environ["SCRATCH"]

def run_global_sasa():
    strategy = AnnealingStrategyGlobalSasa(num_threads=8,t_train=3,max_iters=50,best_update_interval=10,alpha=0.9,test_mode=False,configFile=configpath,csv_output=outpath+"/sasa_out.csv", result_output=outpath+"/local_dump/dump")
    strategy.run()

def run_global():
    strategy = AnnealingStrategyGlobal(num_threads=8,max_iters=50,best_update_interval=10,alpha=0.9,test_mode=False,configFile=configpath,csv_output=outpath+"/global_out.csv", result_output=outpath+"/local_dump/dump")
    strategy.run()

def run_local():
    strategy = AnnealingStrategyLocal(num_threads=8,max_iters=50,best_update_interval=10,alpha=0.9,test_mode=False,configFile=configpath,csv_output=outpath+"/local_out.csv", result_output=outpath+"/local_dump/dump")
    strategy.run()

#run_global()
#run_local()
#run_global_sasa()

