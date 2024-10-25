import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the wav file
file_path = 'own_data/18.wav'
sample_rate, data = wavfile.read(file_path)

# Calculate time axis in seconds
time = np.linspace(0, len(data) / sample_rate, num=len(data))

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, data, label="Audio Signal")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Waveform of WAV File')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
