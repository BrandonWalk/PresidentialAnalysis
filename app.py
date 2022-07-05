import streamlit as st
from pickle import load
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from pandas import DataFrame

stop_words = stopwords.words('english')
stop_words.append("-")
stop_words.append("president")
lemmatizer = WordNetLemmatizer()

with (open("model", "rb")) as openfile:
    model = load(openfile)

def run_model(user_words):
    user_words = user_words.replace("\n","").replace("\'", '').replace('U.S.',"usa").translate(str.maketrans('', '', string.punctuation)).lower()
    user_words = [lemmatizer.lemmatize(word) for word in user_words.split() if word not in stop_words]
    vector = model.infer_vector(user_words)
    return model.docvecs.most_similar([vector])

st.title("Presidential Analysis")
user_words = st.text_input("Insert words")
if st.button("Press To Run"):
    data = run_model(user_words)
    df = DataFrame(data, columns=["President", "Similarity"]).set_index("President").sort_values(by="Similarity")
    st.bar_chart(data = df["Similarity"])
