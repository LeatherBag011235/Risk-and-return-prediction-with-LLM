from src.data_collection.targets_calculation.target_executer_class import TargetExecutor
from src.data_collection.consts import API_KEY, SECRET_KEY, DB_PARAMS

def main(): 
    TargetExecutor(API_KEY, SECRET_KEY, DB_PARAMS, pool_size=10).run()

if __name__ == "__main__":
    main()
