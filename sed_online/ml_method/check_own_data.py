import re
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import vggish_input, vggish_slim, vggish_params, vggish_postprocess
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import joblib


root = 'C:/Users/u144572/self_development/final_model/own_data_test/'

filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.wav' in s]
# Load your model here
# MODEL_PATH='../model/rf_for_cycle_silence_noise.pkl'

def get_array_of_predictions(file_name):
    # Initialize all variables in the model, then load the VGGish checkpoint
    with tf.Graph().as_default(), tf.Session() as sess:
        embeddings = vggish_slim.define_vggish_slim()

        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, 'model/vggish_model.ckpt')
        pproc = vggish_postprocess.Postprocessor('model/vggish_pca_params.npz')

        # Get the input tensor

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

        feautures = vggish_input.wavfile_to_examples(root + file_name + ".wav")

        # Generate embeddings
        embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: feautures}))
        postprocessed_batch = pproc.postprocess(embedding_batch)
        df = pd.DataFrame(postprocessed_batch)  # 128 features vector
            
    # Load the trained neural network model
    nn_model = load_model('ml_method/nn_for_cycle_with_students_data.h5')
    # nn_model = load_model('ml_method/nn_for_cycle_with_students_data_niko_test.h5')

    # Make predictions without reshaping
    predictions = nn_model.predict(df.values)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return(predicted_classes)

def get_rr_from_wav(file_name):
    predictions = get_array_of_predictions(file_name)
    cycles_no = 0
    in_group = False
    for i in range(1, len(predictions)):
        if predictions[i] == 0 and predictions[i - 1] == 0:
            if not in_group:
                cycles_no += 1
                in_group = True
        else:
            in_group = False
    return cycles_no


def calculated_rr_and_predicted(file_name, root):
    respiratory_cycle_number = int(re.search(r'\d+', file_name).group())
    audio_length = librosa.get_duration(filename=f'{root}{file_name}.wav')
    rr = respiratory_cycle_number / (audio_length / 60)

    # Load audio with 22050 Hz sampling rate
    data_x, sampling_rate = librosa.load(f'{root}{file_name}.wav', res_type='kaiser_fast')
    
    # Extract features using MFCC (50 features)
    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=50).T, axis=0)

    # Reshape features to match the model's input shape (1, 50, 1)
    features = features.reshape(1, 50, 1)
    
    # Get the prediction
    predicted_value = get_rr_from_wav(file_name) / (audio_length / 60)

    return pd.DataFrame(data=[[file_name, rr, 
                            #    (predicted_value / (audio_length / 60))
                               predicted_value
                               ]], columns=['patient_id', 'rr', "rr_predicted"])

i_list = []
for s in filenames:
    i = calculated_rr_and_predicted(s, root)
    i_list.append(i)
recording_info = pd.concat(i_list, axis=0)

# Calculate metrics
mae = mean_absolute_error(recording_info['rr'], recording_info['rr_predicted'])
mse = mean_squared_error(recording_info['rr'], recording_info['rr_predicted'])
r2 = r2_score(recording_info['rr'], recording_info['rr_predicted'])

metrics_df = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R² Score'],
    'Value': [mae, mse, r2]
})


import os
if not os.path.isdir("plots_test"):
    os.makedirs('plots_test')

# Display the metrics
print(metrics_df)

# Plotting
def plot_results(df):
    # Calculate residuals
    residuals = df['rr'] - df['rr_predicted']
    
    # Residuals Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(df['rr'], residuals)
    plt.hlines(y=0, xmin=min(df['rr']), xmax=max(df['rr']), colors='r')
    plt.title('Różnica dla wartości rzeczywistych')
    plt.xlabel('Wartość rzeczywista')
    plt.ylabel('Różnica')
    plt.savefig("plots_test/roznia_dla_wartosci_rel")
    plt.show()

    # Predicted vs Actual Values with Linear Regression
    plt.figure(figsize=(12, 6))
    sns.regplot(x='rr', y='rr_predicted', data=df, scatter_kws={'s':10}, line_kws={"color": "red"})
    plt.plot([min(df['rr']), max(df['rr'])], [min(df['rr']), max(df['rr'])], color='blue', linestyle='dashed')  # Line of perfect prediction
    plt.title('Wartość przewidziana vs rzeczywista')
    plt.xlabel('Wartość rzeczywista')
    plt.ylabel('Wartość przewidziana')
    plt.savefig("plots_test/pred_vs_rel")
    plt.show()

    # Error Distribution (Histogram of residuals)
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50)
    plt.title('Rozkład błędów')
    plt.xlabel('Różnica')
    plt.ylabel('Częstotliwość')
    plt.savefig("plots_test/rozklad_bledow")
    plt.show()

# Call the plotting function
plot_results(recording_info)