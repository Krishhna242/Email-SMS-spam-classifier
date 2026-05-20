import streamlit as st
import pickle
import string
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = text.split()

    # Remove stopwords and stemming
    processed_words = []

    for word in words:
        if word not in stopwords.words('english'):
            stemmed_word = ps.stem(word)
            processed_words.append(stemmed_word)

    return ' '.join(processed_words)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model1 = pickle.load(open('model1.pkl','rb'))
model3 = pickle.load(open('model3.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict  
    if model1.predict(vector_input) == model3.predict(vector_input):
        result = model1.predict(vector_input)
    else:
        result = 0
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


