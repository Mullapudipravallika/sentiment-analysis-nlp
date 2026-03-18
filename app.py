import streamlit as st
import pickle

# Title
st.title("Sentiment Analysis App")

# Input
text = st.text_input("Enter text to analyze")

# Dummy prediction logic (replace later with your model)
def predict_sentiment(text):
    if "good" in text.lower() or "happy" in text.lower():
        return "Positive "
    else:
        return "Negative "

# Button
if st.button("Analyze"):
    result = predict_sentiment(text)
    st.write("Sentiment:", result)
