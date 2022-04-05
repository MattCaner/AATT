#from global_sasa_strategy import AnnealingStrategyGlobalSasa
from annealing_strategy_global import AnnealingStrategyGlobal
from annealing_strategy_local import AnnealingStrategyLocal
import os


#run_global()
#run_local()
#run_global_sasa()

strategy = AnnealingStrategyGlobal(num_threads=8,max_iters=50,best_update_interval=10,alpha=0.9,test_mode=False,configFile="benchmark.config",csv_output="global_out.csv")
strategy.run()