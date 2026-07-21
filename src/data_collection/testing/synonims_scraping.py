import requests

from src.data_collection.consts import THESAURUS_API_KEY

word = "risk"  # Change this to any word you want to test

if not THESAURUS_API_KEY:
    raise RuntimeError("Set THESAURUS_API_KEY before running this script.")

url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={THESAURUS_API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        synonyms = data[0].get("meta", {}).get("syns", [])
        if synonyms:
            print(f"Synonyms for '{word}':", synonyms[0])
        else:
            print(f"No synonyms found for '{word}'.")
    else:
        print(f"No valid data returned for '{word}'.")
else:
    print("Error:", response.status_code, response.text)
