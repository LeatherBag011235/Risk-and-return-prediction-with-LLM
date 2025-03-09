import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.converter.dictionary_converter import DictionaryConverter

def test_main():
    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\test_dir")
    prepared_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\prepared_data\test_dir")

    converter = DictionaryConverter(raw_files_dir, prepared_files_dir)
    converter.convert_files()

if __name__ == "__main__":
    test_main()