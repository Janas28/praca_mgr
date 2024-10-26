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
        self.WINDOW = 20  # Window size to analyze in seconds.
        self.SLEEP_TIME = 0.05  # Time to sleep between each prediction is made.
        self.p = None
        self.stream = None
        self.model = load_model('9_dotrenowany_o_dane_wlasne/model.h5')
        self.model.load_weights('9_dotrenowany_o_dane_wlasne/model.weights.h5')

        # Buffer to store up to self.WINDOW seconds of last audio data
        self.buffer = deque(maxlen=int(self.RATE * self.WINDOW))

        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()

        # Primary line for the audio signal
        self.line, = self.ax.plot([], [], lw=2, label='Audio Signal')
        self.ax.set_ylim(-1, 1)  # Amplitude range
        self.ax.set_title(f'Audio Signal and Predicted Respiratory Rate - Last {self.WINDOW} Seconds')
        self.ax.set_xlabel('Seconds')
        self.ax.set_ylabel('Amplitude')

        # Secondary y-axis for predicted values
        self.ax2 = self.ax.twinx()
        self.pred_line, = self.ax2.plot([], [], 'r-', lw=2, label='Respiratory Rate')
        self.ax2.set_ylim(0, 40)  # Predicted value range
        self.ax2.set_ylabel('Respiratory Rate')

        # Deques to store time and predicted values for plotting
        self.time_data = deque(maxlen=int(self.WINDOW / self.SLEEP_TIME))
        self.predicted_data = deque(maxlen=int(self.WINDOW / self.SLEEP_TIME))
        
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

    def plot_audio(self, curr_pred, last_audio, first):
        """Plot the last self.WINDOW seconds of the audio signal and predicted respiratory rate."""
        # Convert x-axis from samples to seconds
        self.current_time = time.perf_counter() - first
        time_axis = (np.arange(
            max(0, self.current_time - self.WINDOW) * self.RATE, 
            (max(0, self.current_time - self.WINDOW) * self.RATE) + len(last_audio)
        )) / self.RATE
        if len(time_axis) != len(last_audio):
            time_axis = time_axis[1:]

        # Update the plot data for the audio signal
        self.line.set_xdata(time_axis)
        self.line.set_ydata(last_audio)
        
        self.time_data.append(self.current_time)
        self.predicted_data.append(curr_pred)

        # Ensure both x-axes are aligned for a consistent time display
        self.ax.set_xlim(max(0, self.current_time - self.WINDOW), self.current_time)
        self.ax2.set_xlim(max(0, self.current_time - self.WINDOW), self.current_time)

        # Update the plot data for the predicted values
        self.pred_line.set_xdata(self.time_data)
        self.pred_line.set_ydata(self.predicted_data)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def mainloop(self):
        first=0
        while self.stream.is_active():
            # if len(self.buffer) >= self.RATE * self.WINDOW:
            # Get the last self.WINDOW seconds of data
            last_audio = np.array(self.buffer)
            
            # Extract MFCC features
            features = librosa.feature.mfcc(y=last_audio, sr=self.RATE, n_mfcc=50)
            features = np.mean(features.T, axis=0)

            # Reshape features to match the model's input shape (1, 50, 1)
            features = features.reshape(1, 50, 1)

            # Predict using the model
            if first == 0:
                first=time.perf_counter()
            predicted_value = self.model.predict(features, verbose=0)[0][0]

            # Update the plot with the last self.WINDOW seconds of the audio signal and predicted values
            self.plot_audio(predicted_value, last_audio, first)
            
            # Sleep for the defined SLEEP_TIME before making the next prediction
            time.sleep(self.SLEEP_TIME)

audio = AudioHandler()
audio.start()     # open the stream
audio.mainloop()  # main operations with librosa
audio.stop()
