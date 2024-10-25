from model import vggish_input, vggish_slim, vggish_params, vggish_postprocess
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import wave
import matplotlib.pyplot as plt

# Load the audio signal for plotting
# wav_path = "C:/Users/u144572/self_development/final_model/own_data/18(1).wav"
# wav_path = "recorded_audio_cycles_30s.wav"
# wav_path = "4okmokmk.wav"
# wav_path = "recorded_audio_noise_talking.wav"

wav_path = "C:/Users/u144572/self_development/sed_online/compare_ml_with_decision_tree/own_data_micro_pg_clear/12kmdwa.wav"
# wav_path = "C:/Users/u144572/self_development/sed_online/compare_ml_with_decision_tree/own_data_micro_pg_clear/Jddd_7.wav"
# wav_path = "recorded_audio_silence_with_background_music.wav"
# wav_path = "C:/Users/u144572/self_development/final_model/own_data_test/Jddd_7.wav"
# wav_path = "record_noise3.wav"
# wav_path = "compare_ml_with_decision_tree/own_data_test/spoczynek_ustami_9.wav"

# nn_model = load_model('ml_method/nn_for_cycle_and_any_other.h5')
# nn_model = load_model('ml_method/nn_for_cycle_silence_noise.h5')
# nn_model = load_model("ml_method/test_2137.h5")
nn_model = load_model("ml_method/test3.h5")


# Load and initialize the VGGish model
with tf.Graph().as_default(), tf.Session() as sess:
    embeddings = vggish_slim.define_vggish_slim()

    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, 'model/vggish_model.ckpt')
    pproc = vggish_postprocess.Postprocessor('model/vggish_pca_params.npz')

    # Get the input tensor
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

    # Load audio features
    features = vggish_input.wavfile_to_examples(wav_path)
    
    # Generate embeddings
    embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: features}))
    postprocessed_batch = pproc.postprocess(embedding_batch)
    df = pd.DataFrame(postprocessed_batch)  # 128 features vector
    
# Load the trained neural network model

# Make predictions without reshaping
print(df.values)
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
samples_per_overlap = int(sample_rate * vggish_params.EXAMPLE_HOP_SECONDS)
samples_per_window = int(sample_rate * 0.96)


for i in range(1, len(predicted_classes) - 1):
    if predicted_classes[i] != predicted_classes[i - 1] and predicted_classes[i] != predicted_classes[i + 1]:
        predicted_classes[i] = predicted_classes[i - 1]  # Set to the previous value to make it consistent


# def flip_value(val):
#     return 1 if val == 0 else 0
# i = 0
# while i < len(predicted_classes):
#     start = i
#     # Find the length of the consecutive group of the same value
#     while i < len(predicted_classes) - 1 and predicted_classes[i] == predicted_classes[i + 1]:
#         i += 1
    
#     group_size = i - start + 1  # Calculate the size of the group

#     # If group is smaller than 5, flip all values in the group
#     if group_size < 10:
#         for j in range(start, i + 1):
#             predicted_classes[j] = flip_value(predicted_classes[j])
    
    # i += 1  # Move to the next value


# Handle the first and last element separately
if len(predicted_classes) > 1 and predicted_classes[0] != predicted_classes[1]:
    predicted_classes[0] = predicted_classes[1]
if len(predicted_classes) > 1 and predicted_classes[-1] != predicted_classes[-2]:
    predicted_classes[-1] = predicted_classes[-2]

for i, prediction in enumerate(predicted_classes):
    if -1 < i < 300:
        if i == 0:
            start_idx = 0
            end_idx = samples_per_window
        else:
            start_idx = (i - 1) * samples_per_overlap + samples_per_window - samples_per_overlap * 2
            end_idx = start_idx + samples_per_overlap
        color = 'red' if prediction == 0 else 'blue'# if prediction == 1 else 'yellow'
        
        plt.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color=color)

plt.xlabel('Czas [s]')
plt.ylabel('Częstotliwość')
plt.title('Nagranie sygnału audio. Oznaczenia: czerwony = cykl oddechowy, niebieski = cisza.')
plt.grid()
plt.tight_layout()
# plt.yticks(np.arange(0, 500, 50)) 
# plt.xticks(np.arange(0, 30, 1)) 
plt.show()
