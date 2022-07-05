import pandas as pd
import numpy as np
from wikipedia import page
import nltk
#nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from string import punctuation
from pickle import dump

stop_words = stopwords.words('english')
stop_words.append("-")
stop_words.append("president")
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

president_list = pd.read_csv("Presidents_List.csv")

president_summary = {}

for president in president_list.President:
    president_summary[president] = page(president, auto_suggest=False).content

for key, value in president_summary.items():
    president_summary[key] = value.replace("\n"," ").replace("\'", '').replace('U.S.',"usa").translate(str.maketrans('', '', punctuation)).lower()

for key, value in president_summary.items():
    president_summary[key] = [lemmatizer.lemmatize(word) for word in value.split() if word not in stop_words]

docs = [TaggedDocument(words=doc, tags=[president]) for president, doc in president_summary.items()]

model = Doc2Vec(docs, vector_size = 500, window = 10000, epochs = 100, min_count = 10, dm = 0)

with (open("model", "wb")) as openfile:
    dump(model, openfile)
