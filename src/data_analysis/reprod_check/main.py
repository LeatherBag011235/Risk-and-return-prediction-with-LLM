from src.data_analysis.reprod_check.data_checker import ColumnRestorer
from src.data_collection.consts import  DB_PARAMS

def main():
    ColumnRestorer(
        db_params=DB_PARAMS,
        root_dir=r"/home/maxim-shibanov/Projects_Py/Parsim-sec/src/Analysis/data/snp_500_scores&returns"
    ).update_column(old_col='eps_surprise', new_col='eps_surprise')

if __name__ == "__main__":
    main()