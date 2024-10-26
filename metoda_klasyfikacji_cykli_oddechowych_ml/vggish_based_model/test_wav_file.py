from model import vggish_input, vggish_slim, vggish_params, vggish_postprocess
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import joblib
import wave
import matplotlib.pyplot as plt


# Load the audio signal for plotting
wav_path = "C:/Users/u144572/self_development/final_model/own_data/Jddd_7.wav"
# wav_path = "compare_ml_with_decision_tree/own_data_micro_pg_clear/12kmdwa.wav"
# wav_path = "recorded_audio_20s.wav"
# wav_path = "recorded_audio_noise_talking.wav"
# wav_path = "recorded_audio_silence_with_background_music.wav"

# Load and initialize the model
with tf.Graph().as_default(), tf.Session() as sess:
    embeddings = vggish_slim.define_vggish_slim()
    rf_classifier = joblib.load('model/rf_for_cycle_silence_noise.pkl')

    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, 'model/vggish_model.ckpt')
    pproc = vggish_postprocess.Postprocessor('model/vggish_pca_params.npz')

    # Get the input tensor
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

    # Load audio features
    features = vggish_input.wavfile_to_examples(wav_path)
    
    # Generate embeddings and predictions
    embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: features}))
    postprocessed_batch = pproc.postprocess(embedding_batch)
    df = pd.DataFrame(postprocessed_batch)  # 128 features vector
    predictions = rf_classifier.predict(df)


with wave.open(wav_path, 'rb') as wf:
    sample_rate = wf.getframerate()
    num_frames = wf.getnframes()
    signal = wf.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

# Generate time axis for the signal
time = np.linspace(0, num_frames / sample_rate, num=num_frames)

# Plotting the waveform with colors based on predictions
plt.figure(figsize=(12, 6))
samples_per_overlap = int(sample_rate * vggish_params.EXAMPLE_HOP_SECONDS)
samples_per_window = int(sample_rate * 0.96)

# Find indices where values appear only once (changes in adjacent values)
for i in range(1, len(predictions) - 1):
    if predictions[i] != predictions[i - 1] and predictions[i] != predictions[i + 1]:
        predictions[i] = predictions[i - 1]  # Set to the previous value to make it consistent

# Handle the first and last element separately
if len(predictions) > 1 and predictions[0] != predictions[1]:
    predictions[0] = predictions[1]
if len(predictions) > 1 and predictions[-1] != predictions[-2]:
    predictions[-1] = predictions[-2]

print(predictions)

for i, prediction in enumerate(predictions):
    if i == 0:
        start_idx = 0
        end_idx = samples_per_window
    else:
        start_idx = (i - 1) * samples_per_overlap + samples_per_window - samples_per_overlap * 2
        end_idx = start_idx + samples_per_overlap
    color = 'red' if prediction == 0 else ('blue' if prediction == 1 else 'yellow')
    
    plt.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color=color)

plt.xlabel('Czas [s]')
plt.ylabel('Częstotliwość')
plt.title('Nagranie sygnału audio. Oznaczenia: czerwony = cykl oddechowy, niebieski = inne.')
plt.grid()
plt.tight_layout()
plt.show()

