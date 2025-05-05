from src.data_collection.targets_calculation.recompute_returns.yfin_target_executor import YFTargetExecutor
from src.data_collection.consts import DB_PARAMS

def main(): 
    YFTargetExecutor(DB_PARAMS, pool_size=32).run()

if __name__ == "__main__":
    main()
