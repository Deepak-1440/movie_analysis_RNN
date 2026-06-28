import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import streamlit as st

# Custom layer to ignore 'time_major' kwarg from legacy models in Keras 3
class CustomSimpleRNN(SimpleRNN):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

@st.cache_data
def load_imdb_data():
    with open('word_index.json', 'r') as f:
        word_index = json.load(f)
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

@st.cache_resource
def load_imdb_model():
    return load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CustomSimpleRNN})

## load the imdb dataset word index
word_index, reverse_word_index = load_imdb_data()

## load the pre-trained model with Relu activation
model = load_imdb_model()

## helper functions
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

### prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negative.")

## user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    ## make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score:  {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
