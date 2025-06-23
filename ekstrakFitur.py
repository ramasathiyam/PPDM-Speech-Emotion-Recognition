# === ekstraksiFitur.py ===
import os
import numpy as np
import pickle
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

def compute_mfcc_manual(signal, sr, num_ceps=13, n_fft=512, n_filters=26):
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
    return mfcc.T  # shape: (num_ceps, time)

def compute_time_features(signal):
    zcr = np.mean(np.diff(np.sign(signal)) != 0)
    rms = np.sqrt(np.mean(signal ** 2))
    return zcr, rms

def compute_freq_features(signal, sr):
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum[:len(spectrum) // 2])
    freqs = np.fft.fftfreq(len(signal), d=1 / sr)[:len(spectrum) // 2]
    centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-6)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-6))
    return centroid, bandwidth

def extract_features(file_path, max_len=216):
    try:
        sr, signal = wavfile.read(file_path)
        signal = signal.astype(np.float32)

        mfcc = compute_mfcc_manual(signal, sr, num_ceps=40)  # shape (40, time)
        zcr, rms = compute_time_features(signal)
        centroid, bandwidth = compute_freq_features(signal, sr)

        time_domain = np.full((1, mfcc.shape[1]), zcr)
        energy_domain = np.full((1, mfcc.shape[1]), rms)
        spectral_centroid = np.full((1, mfcc.shape[1]), centroid)
        spectral_bandwidth = np.full((1, mfcc.shape[1]), bandwidth)

        combined = np.vstack([mfcc, time_domain, energy_domain, spectral_centroid, spectral_bandwidth])

        if combined.shape[1] < max_len:
            pad_width = max_len - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :max_len]

        return np.expand_dims(combined, axis=-1)  # shape: (features, time, 1)
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

# ===== Ekstraksi dan Simpan Data =====
folder = "Tess"
features, labels, filenames = [], [], []

for root, _, files in os.walk(folder):
    for file in tqdm([f for f in files if f.endswith(".wav")], desc=f"Processing {os.path.basename(root)}", unit="file"):
        path = os.path.join(root, file)
        label = os.path.basename(root).split("_")[-1].lower()
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append(label)
            filenames.append(file)

features, labels, filenames = shuffle(features, labels, filenames, random_state=42)

X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
    features, labels, filenames, test_size=0.2, stratify=labels, random_state=42
)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

np.save("X_train.npy", np.array(X_train))
np.save("X_test.npy", np.array(X_test))
np.save("y_train.npy", y_train_enc)
np.save("y_test.npy", y_test_enc)

with open("test_filenames.txt", "w") as f:
    for name in fn_test:
        f.write(name + "\n")

print("\n✅ Ekstraksi fitur selesai dan disimpan.")
