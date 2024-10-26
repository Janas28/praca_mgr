import numpy as np
import pyaudio
import time
import librosa
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.window = 10
        self.sleep = 0.2
        self.model = load_model('9_dotrenowany_o_dane_wlasne/model.h5')
        self.model.load_weights('9_dotrenowany_o_dane_wlasne/model.weights.h5')

        # Buffer to store last 20 seconds of audio data
        self.buffer = deque(maxlen=int(self.RATE * self.window))  # Buffer for 20 seconds of audio

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(0, self.RATE * self.window)  # 10 seconds x sampling rate
        self.ax.set_ylim(-1, 1)  # Amplitude range
        self.ax.set_title('Audio Signal - Last 10 Seconds')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Amplitude')

        # Initialize time
        self.current_time = 0
    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        self.buffer.extend(numpy_array)
        return None, pyaudio.paContinue

    def plot_audio(self, pred):
        """Update the plot with the last 10 seconds of the audio signal."""
        # if len(self.buffer) >= self.RATE * 10:  # Check if we have 10 seconds of audio
        last_10s_audio = np.array(list(self.buffer)[(-self.RATE * self.window):])  # Extract last 10 seconds
        last_10s_time = np.arange(
            self.current_time * self.RATE, 
            (self.current_time * self.RATE) + len(last_10s_audio)
        )
        
        if len(last_10s_time) != len(last_10s_audio):
            last_10s_time = last_10s_time[:-1]
            print(len(last_10s_time), len(last_10s_audio))

        # Update the plot data
        self.line.set_xdata(last_10s_time)
        self.line.set_ydata(last_10s_audio)

        # Rescale the x-axis if needed
        self.ax.set_xlim(
            self.current_time * self.RATE, 
            (self.current_time * self.RATE) + len(last_10s_audio)
            )

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.current_time += self.sleep

    def mainloop(self):
        while self.stream.is_active():
            # if len(self.buffer) >= self.RATE * 20:
            # Get the last 20 seconds of data
            last_20s_audio = np.array(self.buffer)
            
            # Extract MFCC features
            features = librosa.feature.mfcc(y=last_20s_audio, sr=self.RATE, n_mfcc=50)
            features = np.mean(features.T, axis=0)

            # Reshape features to match the model's input shape (1, 50, 1)
            features = features.reshape(1, 50, 1)
            
            # Predict using the model
            predicted_value = self.model.predict(features, verbose=0)[0][0]

            # Update the plot with the last 10 seconds of the audio signal
            self.plot_audio(predicted_value)
            
            # Sleep for 2 seconds before making the next prediction
            time.sleep(self.sleep)

# Initialize and run the audio handler
audio = AudioHandler()
audio.start()     # Open the stream
audio.mainloop()  # Main loop with predictions and interactive plotting
audio.stop()      # Stop the stream
