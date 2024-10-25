import numpy as np
import pyaudio
import time
import librosa
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import joblib
import pandas as pd

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.WINDOW = 20  # Window size to analyze in seconds.
        self.SLEEP_TIME = 0.1  # Time to sleep between each prediction is made.
        self.p = None
        self.stream = None

        self.vggish_checkpoint_path = 'model/vggish_model.ckpt'
        self.CLASS_MODEL_PATH = 'model/rf_for_cycle_silence_noise.pkl'
        self.VGGISH_PARAMS_PATH = 'model/vggish_pca_params.npz'
        self.pproc = vggish_postprocess.Postprocessor(self.VGGISH_PARAMS_PATH)
        self.PREVIOUS_CLASSIFIED_CLASS = None
        self.CLASSIFIES_IN_ROW_TO_COUNT = 2
        self.CYCLE_COUNTER = 0

        # Buffer to store up to self.WINDOW seconds of last audio data
        self.buffer = deque(maxlen=int(self.RATE * self.WINDOW))
      
        # Initialize the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()

        # Primary line for the audio signal
        self.line, = self.ax.plot([], [], lw=2, label='Audio Signal')
        self.ax.set_ylim(-1, 1)  # Amplitude range
        self.ax.set_title(f'Audio Signal and Respiratory Cycle - Last {self.WINDOW} Seconds. The red rectangle describes the duration of one respiratory cycle.')
        self.ax.set_xlabel('Seconds')
        self.ax.set_ylabel('Amplitude')

        # Secondary y-axis for predicted values
        self.ax2 = self.ax.twinx()
        self.pred_line, = self.ax2.plot([], [], 'r-', lw=2)
        # self.ax2.set_ylabel('Respiratory Rate')

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

    def plot_audio(self, curr_pred, last_audio, pred_tmstp):
        """Plot the last self.WINDOW seconds of the audio signal and predicted respiratory rate."""
        # Convert x-axis from samples to seconds
        time_axis = (np.arange(
            max(0, pred_tmstp - self.WINDOW) * self.RATE, 
            (max(0, pred_tmstp - self.WINDOW) * self.RATE) + len(last_audio)
        )) / self.RATE
        if len(time_axis) != len(last_audio):
            time_axis = time_axis[1:]

        # Update the plot data for the audio signal
        self.line.set_xdata(time_axis)
        self.line.set_ydata(last_audio)
        
        self.time_data.append(pred_tmstp)
        self.predicted_data.append(curr_pred)
        if curr_pred == [0]:
            counter = 0
            for x in range(2, 5):
                if self.time_data[-x] > self.time_data[-1] - 1:
                    self.predicted_data.pop()
                    counter += 1
                else:
                    break
            for x in range(counter):
                self.predicted_data.append([0])

        # Ensure both x-axes are aligned for a consistent time display
        self.ax.set_xlim(max(0, pred_tmstp - self.WINDOW), pred_tmstp)
        self.ax2.set_xlim(max(0, pred_tmstp - self.WINDOW), pred_tmstp)

        # Update the plot data for the predicted values
        self.pred_line.set_xdata(self.time_data)
        self.pred_line.set_ydata(self.predicted_data)

        # Change the color of the x-axis label based on the predicted value
        xlabel_color = 'red' if curr_pred == 1 else ('blue' if curr_pred == 2 else 'yellow')
        self.ax.xaxis.label.set_color(xlabel_color)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def mainloop(self):
        first=0    
        rf_classifier = joblib.load(self.CLASS_MODEL_PATH)
        
        tf.compat.v1.enable_eager_execution()
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define VGGish

            embeddings = vggish_slim.define_vggish_slim()

            # Initialize all variables in the model, then load the VGGish checkpoint

            sess.run(tf.global_variables_initializer())
            vggish_slim.load_vggish_slim_checkpoint(sess, self.vggish_checkpoint_path)

            # Get the input tensor

            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            while self.stream.is_active():
                # if len(self.buffer) >= self.RATE * self.WINDOW:
                # Get the last self.WINDOW seconds of data
                last_audio = np.array(self.buffer)
                if first == 0:
                    first=time.perf_counter()
                pred_tmstp = time.perf_counter() - first
                            
                input_batch = vggish_input.waveform_to_examples(last_audio[-(self.RATE):], self.RATE)

                embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: input_batch}))

                postprocessed_batch = self.pproc.postprocess(embedding_batch)

                df = pd.DataFrame(postprocessed_batch)  # 128 features vector

                # Random Forest prediction from VGGish embeddings

                # prediction = rf_classifier.predict(df)
                
                nn_model = load_model('ml_method/nn_for_cycle_silence_noise.h5')

                # print(type(df), type(df.values))
                predictions = nn_model.predict(df.values)

                predicted_classes = np.argmax(predictions, axis=1)
                print(predicted_classes)
                # Increase same class classififications in row

                # if prediction != self.PREVIOUS_CLASSIFIED_CLASS:
                #     self.SAME_CLASS_IN_ROW_COUNTER = 0
                # else:
                #     self.SAME_CLASS_IN_ROW_COUNTER += 1

                # # If we classified enough same classes in row, we can count it as a real one

                # # Update previous classified class
                # self.PREVIOUS_CLASSIFIED_CLASS = prediction

                # if not self.SAME_CLASS_IN_ROW_COUNTER == self.CLASSIFIES_IN_ROW_TO_COUNT:
                #     prediction = [1]

                self.plot_audio(prediction, last_audio, pred_tmstp)
                
                # Sleep for the defined SLEEP_TIME before making the next prediction
                # if prediction == [0]:
                #     time.sleep(1.5)
                # else:
                time.sleep(self.SLEEP_TIME)

audio = AudioHandler()
audio.start()     # open the stream
audio.mainloop()  # main operations with librosa
audio.stop()
