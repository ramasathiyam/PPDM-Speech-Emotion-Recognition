import streamlit as st
import numpy as np
import librosa
import pickle
import os
import matplotlib.pyplot as plt
import tempfile
from keras.models import load_model

st.set_page_config(layout="wide", page_title="Speech Emotion Recognition", page_icon="🎤")

# === Custom CSS for Styling ===
st.markdown("""
<style>
    .header {
        font-size: 40px !important;
        font-weight: bold !important;
        color: #FF4B4B !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px !important;
        color: #1F51FF !important;
        border-bottom: 2px solid #1F51FF;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .emotion-display {
        font-size: 32px;
        font-weight: bold;
        color: #FF4B4B;
        margin: 15px 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        margin: 10px 0;
        background: linear-gradient(90deg, #FF4B4B, #1F51FF);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    .file-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# === Fungsi Ekstraksi Fitur ===
def extract_features(file_path, max_len=216):
    try:
        from scipy.io import wavfile
        from scipy.fftpack import dct

        def compute_mfcc_manual(signal, sr, num_ceps=40, n_fft=512, n_filters=26):
            emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
            frame_size = 0.025
            frame_stride = 0.01
            frame_len = int(round(frame_size * sr))
            frame_step = int(round(frame_stride * sr))
            signal_length = len(emphasized)
            num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step)) + 1
            pad_signal_length = num_frames * frame_step + frame_len
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized, z)

            indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                      np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
            frames = pad_signal[indices.astype(np.int32, copy=False)]
            frames *= np.hamming(frame_len)
            mag_frames = np.absolute(np.fft.rfft(frames, n=n_fft))
            pow_frames = ((1.0 / n_fft) * ((mag_frames) ** 2))

            mel_low = 2595 * np.log10(1 + 0 / 700)
            mel_high = 2595 * np.log10(1 + (sr / 2) / 700)
            mel_points = np.linspace(mel_low, mel_high, n_filters + 2)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            bin = np.floor((n_fft + 1) * hz_points / sr).astype(int)

            fbank = np.zeros((n_filters, int(n_fft / 2 + 1)))
            for m in range(1, n_filters + 1):
                f_m_minus, f_m, f_m_plus = bin[m - 1], bin[m], bin[m + 1]
                for k in range(f_m_minus, f_m):
                    fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                for k in range(f_m, f_m_plus):
                    fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            filter_banks = np.dot(pow_frames, fbank.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
            log_fbank = np.log(filter_banks)
            mfcc = dct(log_fbank, type=2, axis=1, norm='ortho')[:, :num_ceps]
            return mfcc.T

        def compute_time_features(signal, time_len):
            zcr = np.mean(np.diff(np.sign(signal)) != 0)
            rms = np.sqrt(np.mean(signal ** 2))
            return np.full((1, time_len), zcr), np.full((1, time_len), rms)

        def compute_freq_features(signal, sr, time_len):
            spectrum = np.fft.fft(signal)
            magnitude = np.abs(spectrum[:len(spectrum) // 2])
            freqs = np.fft.fftfreq(len(signal), d=1 / sr)[:len(spectrum) // 2]
            centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-6)
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-6))
            return np.full((1, time_len), centroid), np.full((1, time_len), bandwidth)

        sr, signal = wavfile.read(file_path)
        signal = signal.astype(np.float32)

        mfcc = compute_mfcc_manual(signal, sr, num_ceps=40)  # (40, time)
        time_len = mfcc.shape[1]

        zcr, rms = compute_time_features(signal, time_len)
        centroid, bandwidth = compute_freq_features(signal, sr, time_len)

        combined = np.vstack([mfcc, zcr, rms, centroid, bandwidth])  # shape: (44, time)

        # Pad to max_len
        if combined.shape[1] < max_len:
            pad_width = max_len - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :max_len]

        return np.expand_dims(combined, axis=-1), signal, sr  # shape: (44, 216, 1)
    except Exception as e:
        st.error(f"[ERROR in extract_features] {e}")
        return None, None, None

# === Visualization Functions ===
def plot_waveform(data, sr):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(data, sr=sr, color="#1F51FF")
    plt.title('Audio Waveform', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    return plt

def plot_spectrogram(data, sr):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram', fontsize=16)
    plt.tight_layout()
    return plt

def plot_probabilities(probabilities, classes):
    colors = ['#FF4B4B', '#FF9E4B', '#FFD84B', '#4BFF5E', '#4BC0FF', '#8C4BFF']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(classes, probabilities, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Probabilities', fontsize=16)
    plt.xticks(rotation=15)
    return fig

# === Load Model and Encoder ===
@st.cache_resource
def load_resources():
    model = load_model("cnn_model_best.keras")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_resources()

# === Emotion Emoji Mapping ===
emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sadness': '😢',
    'surprise': '😲',
    'calm': '😌',
    'excited': '🤩'
}               

# Dimas 
def main():
    # Header
    st.markdown('<div class="header">Speech Emotion Recognition</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; font-size: 18px;">
        Upload audio files (.wav) to recognize emotions in speech.<br>
        The model can recognize: <strong>angry</strong>, <strong>disgust</strong>, <strong>fear</strong>, 
        <strong>happy</strong>, <strong>neutral</strong>, <strong>sadness</strong>, and more.
    </div>
    """, unsafe_allow_html=True)
    
    # Menampilkan file uploader di bagian utama
    st.markdown('<div class="subheader">Upload Audio Files</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Choose WAV audio files", 
                                      type=["wav"], 
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")
    
    # Proses hanya jika ada file yang diunggah
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.container():
                st.markdown(f'<div class="file-card"><h3>{uploaded_file.name}</h3></div>', unsafe_allow_html=True)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Process file
                with st.spinner(f'Analyzing {uploaded_file.name}...'):
                    feat, data, sr = extract_features(tmp_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    if feat is not None:
                        # Membuat kolom untuk audio player dan visualisasi
                        col_audio, col_viz = st.columns([1, 2])
                        
                        with col_audio:
                            st.audio(uploaded_file, format='audio/wav')
                        
                        with col_viz:
                            # Display visualizations
                            tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
                            
                            with tab1:
                                fig_wave = plot_waveform(data, sr)
                                st.pyplot(fig_wave)
                            
                            with tab2:
                                fig_spec = plot_spectrogram(data, sr)
                                st.pyplot(fig_spec)
                        
                        # --- Bagian Prediksi dan Hasil ---
                        # Make prediction
                        feat_input = np.expand_dims(feat, axis=0)
                        pred_probs = model.predict(feat_input, verbose=0)[0]
                        pred_class = np.argmax(pred_probs)
                        pred_label = le.inverse_transform([pred_class])[0]
                        
                        # Display results
                        emoji = emotion_emojis.get(pred_label, '❓')
                        emotion_result = f"{emoji} {pred_label.capitalize()}"
                        
                        st.markdown('<div class="subheader">Emotion Prediction</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="prediction-box"><div class="emotion-display pulse">{emotion_result}</div></div>', unsafe_allow_html=True)
                        
                        # Display confidence levels
                        st.markdown("**Confidence Levels:**")
                        classes = le.classes_
                        
                        for i, emotion_class in enumerate(classes):
                            prob = pred_probs[i] * 100
                            label = emotion_class.capitalize()
                            emoji = emotion_emojis.get(emotion_class, '❓')
                            
                            st.markdown(f"{emoji} **{label}**: {prob:.1f}%")
                            st.progress(float(pred_probs[i]))
                        
                        # Plot probabilities
                        fig_prob = plot_probabilities(pred_probs, classes)
                        st.pyplot(fig_prob)
                
                # Menambahkan garis pemisah antar file
                st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #6c757d; font-size: 14px;">
        <h3>About This Application</h3>
        <p>This speech emotion recognition system uses a CNN model trained on audio features.</p>
        <p>Features extracted include MFCCs, zero-crossing rate, spectral centroid, and more.</p>
        <p>For best results, use clear audio recordings (2-3 seconds) with a single speaker.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()