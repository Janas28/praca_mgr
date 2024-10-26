import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model


v = "3"
######################################################
# CSV_CYCLE_PATH = 'data/check_cycles.csv'
# CSV_CYCLE_PATH = 'own_data_train_cycles_and_check_cycles.csv'
# CSV_CYCLE_PATH = 'ml_method/all_cycles.csv'
# CSV_CYCLE_PATH = 'ml_method/own_data_micro_pg_clear_cycles_without_inhale.csv'
# CSV_CYCLE_PATH = 'ml_method/test_cycle.csv'
CSV_CYCLE_PATH = f'ml_method/test_csvs/test_cycle{v}.csv'
# CSV_CYCLE_PATH = 'own_data_train_cycles_csv/own_data_train_cycles.csv'
# CSV_SILENCE_PATH = 'data/silence.csv'
# CSV_NOISE_PATH = 'data/noise.csv'
# CSV_SILENCE_PATH = 'data/noise_silence_other.csv'
# CSV_SILENCE_PATH = "ml_method/own_data_micro_pg_clear_pause_without_inhale.csv"
# CSV_SILENCE_PATH = "ml_method/test_silence.csv"
CSV_SILENCE_PATH = f"ml_method/test_csvs/test_silence{v}.csv"
MODEL_PATH = f'ml_method/test{v}.h5'
######################################################

# Load training data
X_cycle = pd.read_csv(CSV_CYCLE_PATH)
X_silence = pd.read_csv(CSV_SILENCE_PATH)
# X_noise = pd.read_csv(CSV_NOISE_PATH)
# X_CSV_NOISE_SILENCE_OTHER = pd.read_csv(CSV_NOISE_SILENCE_OTHER)

# Combining training data into one DataFrame
X = pd.concat([
    X_cycle, 
    X_silence, 
    # X_noise,
    # X_CSV_NOISE_SILENCE_OTHER
    ], ignore_index=True)

# Labeling training data
Y = (
    [0] * len(X_cycle) 
    + [1] * len(X_silence) 
    # + [2] * len(X_noise)
    # + [1] * len(X_CSV_NOISE_SILENCE_OTHER)
)

class_num = 2

# Convert to numpy arrays
X = X.values
Y = np.array(Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# One-hot encoding labels
Y_train = to_categorical(Y_train, num_classes=class_num)
Y_test = to_categorical(Y_test, num_classes=class_num)

# Neural Network model
model = Sequential()

# model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dropout(0.5))

from tensorflow.keras.layers import BatchNormalization

model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(class_num, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model and saving the history
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test))

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
