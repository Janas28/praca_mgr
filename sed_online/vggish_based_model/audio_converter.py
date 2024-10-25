import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import pandas as pd

##################################################
VGGISH_CHECKPOINT_PATH = 'model/vggish_model.ckpt'
VGGISH_PARAMS_PATH = 'model/vggish_pca_params.npz'
##################################################

CSV_PATH = 'C:/Users/u144572/self_development/sed_online/ml_method/'
CYCLE_DIR_PATH = 'C:/Users/u144572/self_development/sed_online/ml_method/own_data_train_cycles'

start_time = time.time()
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
paths = [CYCLE_DIR_PATH]
pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
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
            try:
                breathing_waveform = vggish_input.wavfile_to_examples(breathing_sound_file_path)
            except:
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
                    [embedding_batch] = np.array(
                        sess.run([embeddings], feed_dict={features_tensor: breathing_waveform}))
                except Exception as e:
                    print("Error:", e)
                    continue
                postprocessed_batch = pproc.postprocess(embedding_batch)
                all_embeddings.append(postprocessed_batch)
                print("Size", len(embedding_batch))
                print("Time", time.time() - one_time)

    df = pd.DataFrame(np.concatenate(all_embeddings))
    if path == CYCLE_DIR_PATH:
        file_path = CSV_PATH + 'own_data_train_cycles.csv'
        df.to_csv(file_path, index=False)

print("End time:", time.time() - start_time)
