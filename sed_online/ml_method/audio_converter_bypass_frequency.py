from scipy.signal import butter, lfilter
import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import pandas as pd
import librosa

##################################################
VGGISH_CHECKPOINT_PATH = 'model/vggish_model.ckpt'
VGGISH_PARAMS_PATH = 'model/vggish_pca_params.npz'
##################################################

CSV_PATH = 'C:/Users/u144572/self_development/sed_online/ml_method/data_bypass_frequency/'
CYCLE_DIR_PATH = 'C:/Users/u144572/self_development/sed_online/data/silence'

start_time = time.time()
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
paths = [CYCLE_DIR_PATH]
pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)

# High-pass filter function to remove frequencies below 200 Hz
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=200, fs=22050, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to set frequencies below 200 Hz to 0
def zero_low_frequencies(spectrogram, threshold_freq=200, sampling_rate=22050):
    freq_bins = np.linspace(0, sampling_rate / 2, spectrogram.shape[0])
    spectrogram[freq_bins < threshold_freq] = 0
    return spectrogram

# Updating the main code
for path in paths:
    all_embeddings = []
    print("Converting:", path)
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            one_time = time.time()
            # Full path to the sound file
            breathing_sound_file_path = os.path.join(path, filename)
            print("Processing:", breathing_sound_file_path)

            # Load the breathing sound as sound waves
            breathing_waveform, sampling_rate = librosa.load(breathing_sound_file_path, sr=22050)

            # Apply high-pass filter to remove frequencies below 200 Hz
            filtered_waveform = highpass_filter(breathing_waveform, cutoff=200, fs=sampling_rate)

            # Convert the waveform to log mel spectrogram for VGGish processing
            try:
                breathing_waveform = vggish_input.waveform_to_examples(filtered_waveform, sample_rate=sampling_rate)
            except Exception:
                continue

            with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
                # Define VGGish
                embeddings = vggish_slim.define_vggish_slim()

                # Initialize all variables in the model, then load the VGGish checkpoint
                sess.run(tf.global_variables_initializer())
                vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT_PATH)

                # Get the input tensor
                features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

                # Transform sound waves into log mel spectrogram and pass to the VGGish model to get embeddings
                try:
                    [embedding_batch] = np.array(sess.run([embeddings], feed_dict={features_tensor: breathing_waveform}))
                except Exception as e:
                    print("Error:", e)
                    continue

                # Set frequency components below 200 Hz to zero in the spectrogram
                postprocessed_batch = pproc.postprocess(embedding_batch)
                postprocessed_batch = zero_low_frequencies(postprocessed_batch, threshold_freq=200, sampling_rate=sampling_rate)

                all_embeddings.append(postprocessed_batch)
                print("Size", len(embedding_batch))
                print("Time", time.time() - one_time)

    # Convert embeddings to DataFrame and save to CSV
    df = pd.DataFrame(np.concatenate(all_embeddings))
    if path == CYCLE_DIR_PATH:
        file_path = CSV_PATH + 'silence.csv'
        df.to_csv(file_path, index=False)

print("End time:", time.time() - start_time)
