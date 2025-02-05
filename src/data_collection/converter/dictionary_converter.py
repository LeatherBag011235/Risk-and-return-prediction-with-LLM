from pathlib import Path
import polars as pl
import re

from .converter_class import Converter
from .consts import date_pattern
from data_collection.logging_config import logger

class DictionaryConverter(Converter):

    def __init__(self, raw_files_dir: Path, prepared_files_dir: Path):
        super().__init__(raw_files_dir, prepared_files_dir)
        self.sentiment_words = pl.DataFrame()
        self.dict_with_all_companies = {}

    def set_of_sentimen_words(self) -> pl.DataFrame:
        lm_dict = pl.read_csv(r'C:\Users\310\Desktop\Progects_Py\Parsim-sec\src\converter_api\Loughran-McDonald_MasterDictionary_1993-2021.csv')
        hv_dict = pl.read_csv(r'C:\Users\310\Desktop\Progects_Py\Parsim-sec\src\converter_api\Harvard_inquirerbasic.csv')
        logger.debug(f"Len of lm_dict {len(lm_dict)} \nLen of hv_dict {len(hv_dict)}")

        positive_words_lm = lm_dict.filter(lm_dict["Positive"] > 0)
        negative_words_lm = lm_dict.filter(lm_dict["Negative"] > 0)

        lm_words = positive_words_lm.vstack(negative_words_lm)
        lm_words = lm_words.to_series().str.to_lowercase()
        lm_words = lm_words.to_frame('column_0')

        positive_words_hv = hv_dict.filter(hv_dict["Positiv"] == 'Positiv')
        negative_words_hv = hv_dict.filter(hv_dict["Negativ"] == 'Negativ')

        hv_words = positive_words_hv.vstack(negative_words_hv)
        hv_words = hv_words.to_series().str.to_lowercase()
        hv_words = hv_words.map_elements(lambda word: re.sub(r'#\d+', '', word), return_dtype=pl.Utf8)
        hv_words = hv_words.to_frame('column_0')

        self.sentiment_words = lm_words.vstack(hv_words).drop_nulls().unique()

        return self.sentiment_words
    
    @staticmethod
    def make_texts_same_len(company_dict: dict[str, list[str]]) -> dict[str, list[str]]: 
        max_length = max(len(lst) for lst in company_dict.values())

        for key in company_dict:
            additional_length = max_length - len(company_dict[key])
            company_dict[key] = company_dict[key] + ["N/A"] * additional_length

        return company_dict
    
    def extract_text(self) -> None:
        logger.debug(f"self.raw_files_dir {self.raw_files_dir}")
        for company_dir in self.raw_files_dir.iterdir():

            if company_dir.is_dir():
                company_dict: dict[str, list[str]] = {}

                for file_path in company_dir.iterdir():
                    logger.debug(f"file_path {file_path}")
                    df = pl.read_parquet(file_path)
                    logger.debug(f"{df}")
                    doc_len = df.shape[0]

                    df = df.join(self.sentiment_words, on='column_0', how='inner')
                    report_lst = df.get_column(df.columns[0]).to_list()

                    match = date_pattern.search(str(file_path))
                    report_date = match.group(1)
                    date_and_len = f'{report_date}_{doc_len}'

                    company_dict[date_and_len] = report_lst

                self.dict_with_all_companies[company_dir.name] = DictionaryConverter.make_texts_same_len(company_dict)

    def save_text(self) -> None:
        for company_name, company_dict in self.dict_with_all_companies.items():
            df = pl.DataFrame(company_dict)

            self.prepared_files_dir.mkdir(parents=True, exist_ok=True)
            file_name_new = f"{company_name}_reports.parquet"
            full_path = self.prepared_files_dir / file_name_new

            logger.debug(f"full_path: {full_path}")
            logger.debug(f"prepared df {df}")

            df.write_parquet(full_path)
            logger.info(f"{company_name} is saved successfully")

    def convert_files(self) -> None:
        self.set_of_sentimen_words()

        self.extract_text()
        logger.debug(f".extract_text() has been executed")

        self.save_text()
        logger.debug(f".save_text() has been executed")