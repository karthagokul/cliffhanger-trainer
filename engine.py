from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")  # For new model creation
nlp = spacy.load("en_core_web_sm")  # To train the existing model with new parameters

db = DocBin()

training_data = [
    ("ABCD", {"entities": [(0, 4, "PRODUCT")]}),
    ("AEGHIGJ", {"entities": [(0, 6, "PRODUCT")]}),
    ("DimDum", {"entities": [(0, 5, "PERSON")]}),
]

for text, annot in tqdm(training_data):
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy")
