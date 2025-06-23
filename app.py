# streamlit_app.py (dengan prediksi probabilitas + deteksi file uji)
import streamlit as st
import numpy as np
import librosa
import pickle
import os
from keras.models import load_model

# === Fungsi Ekstraksi Fitur (sama seperti training dan prediksi) ===
def extract_features(file_path, max_len=216):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        combined = np.vstack([mfcc, zcr, rms, spec_centroid, spec_bandwidth, spec_rolloff])

        if combined.shape[1] < max_len:
            pad_width = max_len - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :max_len]

        return np.expand_dims(combined, axis=-1)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# === Load Model dan Encoder ===
model = load_model("cnn_model_best.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === Muat daftar file uji jika tersedia ===
test_filenames = set()
if os.path.exists("test_filenames.txt"):
    with open("test_filenames.txt", "r") as f:
        test_filenames = set(line.strip().lower() for line in f)

# === UI Streamlit ===
st.set_page_config(page_title="Audio Emotion Recognition", page_icon="🎤", layout="centered")
st.title("🎤 Audio Emotion Recognition\n(CNN)")
st.markdown("Unggah satu atau beberapa file audio (.wav) untuk diprediksi.")

uploaded_files = st.file_uploader("Pilih file .wav", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file, format='audio/wav')
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        feat = extract_features("temp.wav")
        if feat is not None:
            feat_input = np.expand_dims(feat, axis=0)
            pred_probs = model.predict(feat_input)[0]
            pred_class = np.argmax(pred_probs)
            pred_label = le.inverse_transform([pred_class])[0]

            st.success(f"**{uploaded_file.name}: {pred_label.upper()}**")
            st.markdown("### Probabilitas Emosi:")
            for i, label in enumerate(le.classes_):
                st.write(f"{label.capitalize()}: {pred_probs[i]*100:.2f}%")

            # ✅ Tandai jika termasuk file uji
            if uploaded_file.name.lower() in test_filenames:
                st.info("📎 File ini merupakan bagian dari data uji saat training (belum pernah dilihat saat pelatihan).")
            else:
                st.success("🆕 File ini bukan bagian dari data training/testing sebelumnya.")
