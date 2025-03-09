import requests

API_KEY = "972efd50-c997-4c60-96f6-ffddc29dc0f1"  # Replace this with your actual key
word = "risk"  # Change this to any word you want to test

url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={API_KEY}"

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
