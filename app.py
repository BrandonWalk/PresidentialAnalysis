import streamlit as st
import altair as alt
from pickle import load
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from pandas import DataFrame

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
st.write("Using a ML model called doc2vec trained on Wikipedia articles of US Presidents, you can enter words any number of words and see what article is most relevant.")
st.write("It is interesting to try historical events, legislative agendas, wars, political rivals, or political jargon and see what president comes up.")
user_words = st.text_input("Insert words")
if st.button("Press To Run"):
    data = run_model(user_words)
    df = DataFrame(data, columns=["President", "Similarity"]).sort_values(by="Similarity")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('President', sort=None),
        y='Similarity'
    )
    st.altair_chart(chart, use_container_width=True)
