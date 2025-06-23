# ✅ Modifikasi `ekstrakFitur.py` untuk menyimpan nama file uji
import os
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

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
        print(f"Error in {file_path}: {e}")
        return None

folder = "Tess"
features, labels, filenames = [], [], []

for root, _, files in os.walk(folder):
    wav_files = [f for f in files if f.endswith(".wav")]
    for file in tqdm(wav_files, desc=f"Processing {os.path.basename(root)}", unit="file"):
        path = os.path.join(root, file)
        label = os.path.basename(root).split("_")[-1].lower()
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            labels.append(label)
            filenames.append(file)  # ✅ Simpan nama file

features, labels, filenames = shuffle(features, labels, filenames, random_state=42)

X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
    features, labels, filenames, test_size=0.2, random_state=42, stratify=labels
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

# ✅ Simpan nama file yang masuk ke data uji
with open("test_filenames.txt", "w") as f:
    for name in fn_test:
        f.write(name + "\n")
