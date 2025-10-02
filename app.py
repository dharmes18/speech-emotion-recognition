# app.py
import streamlit as st
import joblib
import numpy as np
from feature_extraction import extract_features

# Load models
model = joblib.load("models/optimized_ser_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV file and get the predicted emotion!")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    features = extract_features("temp.wav")
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)
    emotion = encoder.inverse_transform(pred)[0]

    st.success(f"Predicted Emotion: **{emotion}** 🎯")
