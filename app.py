import nltk
nltk.download('punkt')
import streamlit as st
import pickle

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classification")
input_sms=st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Text Preprocessing
    transformed_sms=transform_text(input_sms)
    # 2. Vectorization
    vector_input=tfidf.transform([transformed_sms])
    # 3. Prediction
    result=model.predict(vector_input)[0]
    # 4. Display Output
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")


