import sys 
sys.path.append(r'C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.srores_computation.dictionary_sentiments.sentiment_analyzer import SentimentAnalyzer

DB_PARAMS = {
    "dbname": "reports_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

def main():
    analyzer1 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\lm.paquet"), "lm_orig_score", workers=32)
    analyzer1.run()

    analyzer2 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_lm1.paquet"), "md_lm1", workers=32)
    analyzer2.run()

    analyzer3 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_lm2.paquet"), "md_lm2", workers=32)
    analyzer3.run()

    analyzer4 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_lm3.paquet"), "md_lm3", workers=32)
    analyzer4.run()

    analyzer5 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\hv.paquet"), "hv_orig_score", workers=32)
    analyzer5.run()

    analyzer6 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_hv1.paquet"), "md_hv1", workers=32)
    analyzer6.run()

    analyzer7 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_hv2.paquet"), "md_hv2", workers=32)
    analyzer7.run()

    analyzer8 = SentimentAnalyzer(DB_PARAMS, Path(r"C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src\data_collection\srores_computation\dictionary_sentiments\dictionaries\md_hv3.paquet"), "md_hv3", workers=32)
    analyzer8.run()

if __name__ == "__main__":
    main()