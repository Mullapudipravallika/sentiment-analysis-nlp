import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Sentiment Analysis App")

text = st.text_input("Enter text")

if st.button("Analyze"):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        st.write("Positive 😊")
    else:
        st.write("Negative 😞")
