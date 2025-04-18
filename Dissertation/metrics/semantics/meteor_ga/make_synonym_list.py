import requests
import time
from bs4 import BeautifulSoup
from ufal.udpipe import Model, Pipeline


def lemmatize_word(word):
    # Process the word as a sentence
    processed = pipeline.process(word)
    for line in processed.split("\n"):
        if line and not line.startswith("#"):
            cols = line.split("\t")
            if len(cols) >= 3:
                return cols[2]  # Lemma column
    return word  # Fallback: return original word


def get_synonyms_from_potafocal(word):
    url = f"http://www.potafocal.com/thes/?s={word}"
    response = requests.get(url)
    if response.status_code != 200:
        return []  # Return an empty list
    soup = BeautifulSoup(response.text, "html.parser")

    syns = {a.text.strip().lower()
                for section in soup.find_all("div", class_="sense")
                for a in section.find_all("a", href=True)
    }
    return list(syns)


MODEL_PATH = "irish-idt-ud-2.5-191206.udpipe"
model = Model.load(MODEL_PATH)
if not model:
    print("Error: Could not load the UDPipe model.")
    exit(1)
pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

with open("data/irish/references.txt", "r", encoding="utf-8") as f:
    references = f.readlines()

output_file = "paraphrase-ga.txt"

visited = []

for i, reference in enumerate(references):
    pairs = []
    for word in reference.split(" "):
        word = lemmatize_word(word)
        if word in visited:
            continue
        visited.append(word)
        try:
            synonyms = get_synonyms_from_potafocal(word)
            for syn in synonyms:
                if syn != word:
                    pairs.append((word, syn))
            time.sleep(1)  # Be respectful
        except Exception as e:
            print(f"Error for {word}: {e}")

    with open(output_file, "a", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")