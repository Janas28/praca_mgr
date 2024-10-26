import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

######################################################
# Define paths
WAV_CYCLE_DIR = 'C:/Users/u144572/self_development/sed_online/mfcc_ml_methhod/own_data_micro_pg_clear_cycles/'  # Directory containing cycle wav files
WAV_SILENCE_DIR = 'C:/Users/u144572/self_development/sed_online/mfcc_ml_methhod/own_data_micro_pg_clear_pause/'  # Directory containing silence wav files
MODEL_PATH = 'test_mfcc_model.h5'
######################################################

# Function to extract MFCC features from wav files
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=400):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # Padding/truncating to ensure equal lengths
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Load training data
def load_data_from_dir(directory):
    mfcc_features = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            mfcc = extract_mfcc(file_path)
            mfcc_features.append(mfcc)
    return np.array(mfcc_features)

X_cycle = load_data_from_dir(WAV_CYCLE_DIR)
X_silence = load_data_from_dir(WAV_SILENCE_DIR)

# Combining training data into one array
X = np.concatenate([X_cycle, X_silence], axis=0)

# Labeling training data
Y = (
    [0] * len(X_cycle) 
    + [1] * len(X_silence)
)

class_num = 2

# Reshape X to 2D (samples, features) for neural network
X = X.reshape(X.shape[0], -1)

# Convert labels to numpy array
Y = np.array(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# One-hot encoding labels
Y_train = to_categorical(Y_train, num_classes=class_num)
Y_test = to_categorical(Y_test, num_classes=class_num)

# Neural Network model
model = Sequential()

model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(class_num, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model and saving the history
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test))

# Saving the trained model
model.save(MODEL_PATH)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Plotting the learning accuracy curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid()
plt.tight_layout()
plt.show()

model.summary()
