import numpy as np
import pandas as pd
import wave
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras.models import load_model

# Path to the .wav file
wav_path = "C:/Users/u144572/self_development/sed_online/compare_ml_with_decision_tree/own_data_micro_pg_clear/12kmdwa.wav"

# Load the trained neural network model
nn_model = load_model("mfcc_ml_methhod/test_mfcc_model.h5")

# Function to extract MFCC features from wav files and flatten to match model input shape
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=400, flatten_size=5200):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # Padding/truncating to ensure equal lengths
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    # Flattening the MFCC features to match the expected input shape (5200)
    mfcc_flattened = mfcc.T.flatten()[:flatten_size]  # Ensure the shape is 5200
    return mfcc_flattened

# Extract MFCC features from the audio file and reshape to match model input
mfcc_features = extract_mfcc(wav_path)

# Convert MFCC features to DataFrame (compatible input for the model)
df = pd.DataFrame([mfcc_features])  # Wrap it as a 2D array for model input

# Make predictions without reshaping
predictions = nn_model.predict(df.values)
predicted_classes = np.argmax(predictions, axis=1)

# Load the audio signal for plotting
with wave.open(wav_path, 'rb') as wf:
    sample_rate = wf.getframerate()
    num_frames = wf.getnframes()
    signal = wf.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

# Generate time axis for the signal
time = np.linspace(0, num_frames / sample_rate, num=num_frames)

# Plotting the waveform with colors based on predictions
plt.figure(figsize=(12, 6))
samples_per_window = len(signal) // len(predicted_classes)  # Adjust this to match your data size

# Smooth out prediction changes
for i in range(1, len(predicted_classes) - 1):
    if predicted_classes[i] != predicted_classes[i - 1] and predicted_classes[i] != predicted_classes[i + 1]:
        predicted_classes[i] = predicted_classes[i - 1]

# Handle the first and last element separately
if len(predicted_classes) > 1 and predicted_classes[0] != predicted_classes[1]:
    predicted_classes[0] = predicted_classes[1]
if len(predicted_classes) > 1 and predicted_classes[-1] != predicted_classes[-2]:
    predicted_classes[-1] = predicted_classes[-2]

# Plot the waveform, coloring segments based on predictions
for i, prediction in enumerate(predicted_classes):
    start_idx = i * samples_per_window
    end_idx = start_idx + samples_per_window
    color = 'red' if prediction == 0 else 'blue'
    plt.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color=color)

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Audio Signal: Red = Breathing Cycle, Blue = Other')
plt.grid()
plt.tight_layout()
plt.xticks(np.arange(0, int(time[-1]), 1))
plt.show()
