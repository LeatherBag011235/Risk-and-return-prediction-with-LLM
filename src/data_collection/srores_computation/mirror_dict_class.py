import requests
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class MirrorDict:
    """
    A class to generate a 'mirror dictionary' using synonyms from the Merriam-Webster Thesaurus API.

    Attributes:
        api_key (str): API key for the Merriam-Webster Thesaurus API.
        set_pos (set[str]): Set of positive words.
        set_neg (set[str]): Set of negative words.
        session (requests.Session): Persistent session for API requests.
        base_url (str): Base URL for the API requests.
    """

    def __init__(self, api_key: str, set_pos: set[str], set_neg: set[str]):
        """
        Initializes the MirrorDict instance.

        Args:
            api_key (str): API key for the Merriam-Webster API.
            set_pos (set[str]): Set of positive words.
            set_neg (set[str]): Set of negative words.
        """
        self.api_key = api_key
        self.set_pos = set_pos
        self.set_neg = set_neg
        self.session = MirrorDict.create_session()  # Use a session for persistent connections
        self.base_url = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json"

    def retrieve_syns(self, word: str) -> list[str]:
        """
        Retrieves synonyms for a given word from the Merriam-Webster Thesaurus API.

        Args:
            word (str): The word to find synonyms for.

        Returns:
            list[str]: A list of synonyms if found, otherwise an empty list.
        """
        url = f"{self.base_url}/{word}?key={self.api_key}"
        try:
            response = self.session.get(url, timeout=5)  # Set timeout to prevent hanging
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return data[0].get("meta", {}).get("syns", [])[0]  # Return the first list of synonyms
            return []  # Return an empty list if no synonyms found
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []

    @staticmethod    
    def choose_rand_syn(syns: list[str]) -> str:
        """
        Randomly selects a synonym from a given list.

        Args:
            syns (list[str]): List of synonyms.

        Returns:
            str: A randomly chosen synonym, or an empty string if no synonyms are available.
        """
        return random.choice(syns) if syns else ""

    def validate_word(self, word: str) -> bool:
        """
        Checks if a word is not in set_pos or set_neg.

        Args:
            word (str): The word to validate.

        Returns:
            bool: True if the word is not in either set, False otherwise.
        """
        return word not in self.set_pos and word not in self.set_neg
    
    def process_word(self, word: str) -> str | None:
        """
        Processes a word by retrieving a valid synonym.

        Args:
            word (str): The word to process.

        Returns:
            str | None: A valid synonym if found, otherwise None.
        """
        syns = self.retrieve_syns(word)
        if syns:
            for _ in range(20):  # Attempt up to 20 times to find a valid synonym
                rand_syn = self.choose_rand_syn(syns)
                if self.validate_word(rand_syn):
                    return rand_syn
        return None  # Return None if no valid synonym found
                    
    @staticmethod
    def create_session() -> requests.Session:
        """
        Creates a requests.Session() with an increased connection pool limit.

        Returns:
            requests.Session: A configured session with retry mechanisms.
        """
        session = requests.Session()
        
        # Configure the adapter to increase the pool size and retry failed requests
        adapter = HTTPAdapter(
            pool_connections=35,  # Increase max connections
            pool_maxsize=35,  # Max simultaneous requests
            max_retries=Retry(total=5, backoff_factor=1.5)  # Retry failed requests
        )
        
        session.mount("https://", adapter)
        return session

    def create_mirror_set(self, init_set: set[str]) -> set[str]:
        """
        Creates a mirrored dictionary by replacing words with their synonyms using parallel processing.

        Args:
            init_set (set[str]): The initial set of words to process.

        Returns:
            set[str]: A set of words with synonyms replacing the original words where possible.
        """
        alter_set = set()

        with ThreadPoolExecutor(max_workers=32) as executor, tqdm(total=len(init_set), desc="MirrorDict creation", unit="word") as progress:
            futures = {executor.submit(self.process_word, word): word for word in init_set}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:  
                        alter_set.add(result)
                except Exception as e:
                    print(f"Error processing word '{futures[future]}': {e}")
                
                progress.update(1)  
        
        print(f"Processed set. Resulting length: {len(alter_set)}")
        return alter_set
    
    def create_mirror_dict(self) -> tuple[set[str], set[str]]:
        """
        Creates a mirrored dictionary for both positive and negative word sets.

        Returns:
            tuple[set[str], set[str]]: The modified positive and negative word sets.
        """
        return self.create_mirror_set(self.set_pos), self.create_mirror_set(self.set_neg)
