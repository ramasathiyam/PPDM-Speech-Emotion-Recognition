import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.keras import TqdmCallback  # ✅ Untuk progress bar

# === Load Dataset ===
X_train = np.load("X_train.npy")  # shape: (samples, features, time, 1)
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# === Load Label Encoder ===
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === One-hot Encode Labels ===
n_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_test_cat  = to_categorical(y_test, num_classes=n_classes)

# === Bangun Model CNN ===
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2], 1)),  # ✅ Input layer eksplisit
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_ckpt = ModelCheckpoint("cnn_model_best.keras", monitor='val_accuracy', save_best_only=True)

# === Training dengan Progress Bar ===
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, model_ckpt, TqdmCallback(verbose=1)],
    verbose=0
)

# === Evaluasi Akhir ===
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_pred_labels, target_names=le.classes_))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred_labels)
print("\nConfusion Matrix:")
print(cm)

# === Visualisasi Confusion Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === Plot Accuracy & Loss ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()

print("\n✅ Training selesai. Model terbaik disimpan di cnn_model_best.keras")
