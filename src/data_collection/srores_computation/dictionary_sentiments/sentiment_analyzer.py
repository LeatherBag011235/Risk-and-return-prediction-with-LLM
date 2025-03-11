import polars as pl
import psycopg2
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class SentimentAnalyzer:
    """
    Computes sentiment scores for quarterly reports using a single dictionary 
    and stores the results in the PostgreSQL database.
    """

    def __init__(self, db_params: dict, dictionary_path: Path, score_column: str, workers: int = 16):
        """
        Initializes the SentimentAnalyzer.

        Args:
            db_params (dict): PostgreSQL connection parameters.
            dictionary_path (Path): Path to the Parquet file containing the sentiment dictionary.
            score_column (str): Column name where the sentiment score will be stored.
            workers (int): Number of parallel workers (default: 16).
        """
        self.db_params = db_params
        self.dictionary_path = dictionary_path
        self.score_column = score_column
        self.workers = workers
        self.pos_words, self.neg_words = self.load_dictionary()

        # Ensure the score column exists in the reports table
        self.ensure_score_column_exists()

    def load_dictionary(self) -> tuple[set[str], set[str]]:
        """
        Loads the sentiment dictionary from a Parquet file.

        Returns:
            tuple[set[str], set[str]]: A set of positive words and a set of negative words.
        """
        df = pl.read_parquet(self.dictionary_path)
        pos_words = set(df.filter(pl.col("positive") == True)["word"].to_list())
        neg_words = set(df.filter(pl.col("positive") == False)["word"].to_list())
        return pos_words, neg_words

    def ensure_score_column_exists(self):
        """
        Checks if the score column exists in the reports table and adds it if missing.
        """
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'reports' AND column_name = %s;
        """, (self.score_column,))
        exists = cursor.fetchone()

        if not exists:
            # Add column if it does not exist
            cursor.execute(f"""
                ALTER TABLE reports ADD COLUMN {self.score_column} DOUBLE PRECISION;
            """)
            conn.commit()
            print(f"✅ Added missing column '{self.score_column}' to reports table.")

        conn.close()

    def fetch_reports_from_db(self) -> list[tuple[int, str]]:
        """
        Fetches quarterly reports from the PostgreSQL database.

        Returns:
            list[tuple[int, str]]: List of (id, raw_text) tuples.
        """
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        query = "SELECT id, raw_text FROM reports WHERE raw_text IS NOT NULL;"
        cursor.execute(query)
        reports = cursor.fetchall()

        conn.close()
        return reports

    @staticmethod
    def count_sentiment_words(text: str, pos_words: set[str], neg_words: set[str]) -> tuple[int, int]:
        """
        Counts the number of positive and negative words in a text.

        Args:
            text (str): The document text.
            pos_words (set[str]): Set of positive words.
            neg_words (set[str]): Set of negative words.

        Returns:
            tuple[int, int]: (Positive word count, Negative word count)
        """
        words = re.findall(r'\b\w+\b', text.lower())  # Tokenize words
        word_counts = Counter(words)

        n_pos = sum(word_counts[word] for word in pos_words if word in word_counts)
        n_neg = sum(word_counts[word] for word in neg_words if word in word_counts)

        return n_pos, n_neg

    def compute_sentiment_score(self, report_id: int, text: str) -> dict:
        """
        Computes the sentiment score for a document using the dictionary.

        Args:
            report_id (int): Unique report identifier.
            text (str): The text of the document.

        Returns:
            dict: Dictionary with id and computed sentiment score.
        """
        n_pos, n_neg = self.count_sentiment_words(text, self.pos_words, self.neg_words)

        if n_pos + n_neg == 0:
            score = 0.0  
        else:
            score = (n_pos - n_neg) / (n_pos + n_neg)

        return {"id": report_id, self.score_column: score}

    def process_reports_parallel(self, reports: list[tuple[int, str]]) -> list[dict]:
        """
        Processes multiple reports in parallel to compute sentiment scores.

        Args:
            reports (list[tuple[int, str]]): List of (id, raw_text).

        Returns:
            list[dict]: List of sentiment score dictionaries.
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor, tqdm(
            total=len(reports), desc=f"Processing {self.score_column}", unit="report"
        ) as progress:
            future_to_report = {
                executor.submit(self.compute_sentiment_score, report_id, text): report_id
                for report_id, text in reports
            }

            for future in as_completed(future_to_report):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing report {future_to_report[future]}: {e}")
                
                progress.update(1)  

        return results

    def update_database_with_sentiments(self, results: list[dict]):
        """
        Updates the PostgreSQL database with computed sentiment scores.

        Args:
            results (list[dict]): List of sentiment score dictionaries.
        """
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        with tqdm(total=len(results), desc=f"Updating DB: {self.score_column}", unit="report") as progress:
            for result in results:
                report_id = result["id"]
                score = result[self.score_column]

                update_query = f"""
                UPDATE reports
                SET {self.score_column} = %s
                WHERE id = %s;
                """
                cursor.execute(update_query, (score, report_id))

                progress.update(1)  

        conn.commit()
        conn.close()

    def run(self):
        """
        Runs the full sentiment analysis pipeline:
        1. Fetches reports from the database.
        2. Computes sentiment scores in parallel.
        3. Updates the database with computed scores.
        """
        print(f"Fetching reports from the database for {self.score_column}...")
        reports = self.fetch_reports_from_db()

        print(f"Processing {len(reports)} reports in parallel for {self.score_column}...")
        sentiment_results = self.process_reports_parallel(reports)

        print(f"Updating the database with {self.score_column} scores...")
        self.update_database_with_sentiments(sentiment_results)

        print(f"✅ Sentiment analysis completed and stored in column {self.score_column}.")
