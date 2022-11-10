import nltk
import sklearn
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform(email):
    email = email.lower()
    email = nltk.word_tokenize(email)
    y = []
    for i in email:
        if i.isalnum():
            y.append(i)
    email = y[:]
    y.clear()
    for i in email:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    email = y[:]
    y.clear()
    for i in email:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Email/email spam")
input_email = st.text_area("enter the message")
transform_email = transform(input_email)
vector_input = tfidf.transform([transform_email])
result = model.predict(vector_input)[0]
if result==1:
    st.header("spam")
else:
    st.header("not spam")