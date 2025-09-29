import librosa
import numpy as np

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    
    features = np.concatenate((
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel_spectrogram.T, axis=0),
        np.mean(spectral_contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0)
    ), axis=0)
    
    return features
