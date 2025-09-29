import joblib
from feature_extraction import extract_features  # import from your new file

# Load trained model, scaler, and encoder
model = joblib.load("models/optimized_ser_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Path to your test audio file
audio_path = r"D:\project\dataset\test_audio\03-01-04-01-01-01-02.wav" # example

# Extract and scale features
features = extract_features(audio_path)
features_scaled = scaler.transform([features])

# Predict
predicted_label = model.predict(features_scaled)
predicted_emotion = encoder.inverse_transform(predicted_label)

print("Predicted Emotion:", predicted_emotion[0])
