from pathlib import Path

from src.data_collection.srores_computation.dictionary_sentiments.sentiment_analyzer import SentimentAnalyzer
from src.data_collection.consts import  DB_PARAMS

def main():
    dict_dir = Path(r"src/data_collection/srores_computation/dictionary_sentiments/dictionaries/ditc_to_count")
    
    for path in dict_dir.iterdir():
        name = path.name.split('.')[0]
        SentimentAnalyzer(DB_PARAMS, path, name, workers=32).run()

if __name__ == "__main__":
    main()