import wave
import math
import librosa
import numpy as np
import pandas as pd
import pyaudio
import torch
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import stft
from model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import tensorflow.compat.v1 as tf
from df.enhance import init_df
import joblib

# Constants

vggish_checkpoint_path = 'C:/Users/u144572/self_development/sed_online/model/vggish_model.ckpt'
CLASS_MODEL_PATH = 'C:/Users/u144572/self_development/sed_online/model/rf_for_cycle_silence_noise.pkl'
VGGISH_PARAMS_PATH = 'C:/Users/u144572/self_development/sed_online/model/vggish_pca_params.npz'
pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
model, df_state, _ = init_df()

REFRESH_TIME = 0.5
N_FOURIER = 2048

FORMAT = pyaudio.paInt16

CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 4

CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True

filename = '107_2b3_Al_mc_AKGC417L'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''                         Main function                             '''

if __name__ == "__main__":
    prediction_path = f'{filename}.wav'

    wf = wave.open("own_data/Jddd_7.wav", 'rb')

    signal = wf.readframes(-1)

    # Length of recording in seconds

    length_in_seconds = wf.getnframes() / float(wf.getframerate())  # 4 because of stereo + 2 bytes per sample
    print(length_in_seconds)
    # Number of 0.25s sectors

    num_frames = math.floor(length_in_seconds / REFRESH_TIME)
    # Count number of frames to read

    frames_to_read = num_frames * CHUNK_SIZE

    signal = np.frombuffer(signal[:], dtype='int16')

    time = np.linspace(0, len(signal) / wf.getframerate(), num=len(signal))

    fig, axs = plt.subplots(4, 1)  # 2 rows, 1 column



    '''                        VGGish                      '''

    rf_classifier = joblib.load(CLASS_MODEL_PATH)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish

        embeddings = vggish_slim.define_vggish_slim()

        # Initialize all variables in the model, then load the VGGish checkpoint

        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)

        # Get the input tensor

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

        # Number of samples per 0.25s

        samples_per_quarter_second = int(wf.getframerate() * 0.25 * 2)

        # Dividing recording into 0.25s parts

        split_signal = [signal[i:i + samples_per_quarter_second] for i in
                        range(0, len(signal), samples_per_quarter_second)]

        for i, recording in enumerate(split_signal):
            print(i)
            if len(recording) < samples_per_quarter_second:
                continue

            # Write 0.25s part to a file

            buffer = [recording, recording, recording, recording]

            wf = wave.open("temp.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(buffer))
            wf.close()

            # Prepare input for the model

            input_batch = vggish_input.wavfile_to_examples('temp.wav')

            # Calculate embeddings

            embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: input_batch}))

            postprocessed_batch = pproc.postprocess(embedding_batch)

            df = pd.DataFrame(postprocessed_batch)  # 128 features vector

            prediction = rf_classifier.predict(df)

            color = ''
            if prediction == 0:  # Inhale
                color = 'red'
            elif prediction == 1:  # Exhale
                color = 'green'
            else:
                color = 'blue'
            axs[2].plot(time[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], signal[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE],
                        color=color)

        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Amplitude')
        axs[2].set_title('Segmenty przewidziane przez model oparty na modelu VGGish')
    plt.tight_layout()

    plt.show()