from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import json

nlp = spacy.blank("en")  # For new model creation
# To train the existing model with new parameters
nlp = spacy.load("en_core_web_sm")

db = DocBin()

training_data = [
    ("ABCD", {"entities": [(0, 4, "PRODUCT")]}),
    ("AEGHIGJ", {"entities": [(0, 6, "PRODUCT")]}),
    ("DimDum", {"entities": [(0, 5, "PERSON")]}),
]
# Opening JSON file
with open('training_input.json') as f:
    data = json.load(f)

    for value in data:
        if isinstance(value, dict):
            keyword = value["keyword"]
            if keyword["string"] and keyword["type"]:
                print(keyword["string"])
                print(keyword["type"])

exit(0)

for text, annot in tqdm(training_data):
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label,
                             alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy")
