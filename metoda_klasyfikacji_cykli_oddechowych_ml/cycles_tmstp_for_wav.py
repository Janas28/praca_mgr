import os
import time
from pydub import AudioSegment
from pydub.playback import play
import keyboard  # Library to detect keypress events
import pandas as pd
from time import perf_counter

# Directory containing .wav files
wav_directory = 'compare_ml_with_decision_tree/own_data_micro_pg_clear/'  # Replace with your directory path

# List to store the timestamps for each file
timestamps_list = []

# Loop through each .wav file in the directory
for filename in os.listdir(wav_directory):        
    if filename.endswith(".wav"):
        file_path = os.path.join(wav_directory, filename)
        print(f"Playing file: {filename}")
        
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)
        
        # Define lists to store start and stop timestamps for each press
        start_timestamps = []
        stop_timestamps = []

        start = perf_counter()
        # Function to handle space bar press
        def on_space_press(event):
            if len(start_timestamps) == len(stop_timestamps):  # Equal count, start new interval
                start_timestamps.append(perf_counter() - start)
                print(f"Start timestamp: {start_timestamps[-1]:.2f} seconds")
            else:  # One more start than stop, so end the interval
                stop_timestamps.append(perf_counter() - start)
                print(f"Stop timestamp: {stop_timestamps[-1]:.2f} seconds")

        # Register the space bar event handler
        keyboard.on_press_key("space", on_space_press)

        # Play the audio in a non-blocking way
        play(audio)
 
        print("Press space to record timestamps. Press 'q' to move to the next file.")

        # Wait until the user presses 'q' to move to the next file
        while True:
            if keyboard.is_pressed('q'):
                print("Moving to the next file.")
                break
            elif keyboard.is_pressed('esc'):  # Stop the entire script
                print("Stopping the script.")
                continue_script = False
                break
            time.sleep(0.1)  # Short sleep to reduce CPU usage

        # Remove space bar handler after file is done
        keyboard.unhook_all()

        # Save timestamps for this file to the list
        file_timestamps = {
            'filename': filename,
            'start_timestamps': start_timestamps,
            'stop_timestamps': stop_timestamps
        }
        timestamps_list.append(file_timestamps)

# Convert the list of timestamps to a DataFrame for easier manipulation and saving
timestamps_df = pd.DataFrame(timestamps_list)

# Save the timestamps to a CSV file
timestamps_df.to_csv('own_data_micro_pg_clear_timestamps.csv', index=False)
print("Timestamps saved to 'timestamps.csv'.")
