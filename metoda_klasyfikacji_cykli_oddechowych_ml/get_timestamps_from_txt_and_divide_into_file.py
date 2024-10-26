import os
from pydub import AudioSegment

# Define the path to the directory containing the .wav files
wav_directory = 'C:/Users/u144572/self_development/sed_online/own_data/'  # Replace with your directory path

# Define the path to the output directory where subfiles will be saved
output_directory = 'C:/Users/u144572/self_development/sed_online/ml_method/own_data_train_cycles/'  # Replace with your output directory path
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the path to the pause directory where pauses will be saved
pause_directory = 'C:/Users/u144572/self_development/sed_online/ml_method/own_data_train_pause/'  # Replace with your pause output directory path
if not os.path.exists(pause_directory):
    os.makedirs(pause_directory)

# Define the path to the text file containing the timestamps
timestamps_file_path = 'C:/Users/u144572/self_development/sed_online/ml_method/own_data_train_cycles.txt'  # Replace with your timestamps file path

# Function to parse the timestamps from the text file
def parse_timestamps(file_path):
    timestamps_dict = {}
    current_file = None

    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            print(line)
            line = line.strip()
            if line.startswith('Playing file:'):
                # Extract the file name
                current_file = line.split(': ')[1]
                timestamps_dict[current_file] = []
            elif line.startswith('Start timestamp:'):
                # Extract the start timestamp
                start_time = float(line.split(': ')[1].split()[0])
                print(start_time)
                timestamps_dict[current_file].append((start_time, None))
            elif line.startswith('Stop timestamp:'):
                # Extract the stop timestamp
                stop_time = float(line.split(': ')[1].split()[0])
                # Update the last start timestamp entry with the stop time
                if current_file and timestamps_dict[current_file]:
                    start_stop_pair = timestamps_dict[current_file].pop()
                    timestamps_dict[current_file].append((start_stop_pair[0], stop_time))
    
    return timestamps_dict

# Read and parse the timestamps from the text file
timestamps_data = parse_timestamps(timestamps_file_path)

# Iterate through each file and its corresponding timestamps
for filename, timestamps in timestamps_data.items():
    file_path = os.path.join(wav_directory, filename)
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load the audio file using pydub
    audio = AudioSegment.from_wav(file_path)

    # Extract and save each segment based on timestamps
    previous_stop = 0  # Keep track of the previous stop time for pauses
    for index, (start, stop) in enumerate(timestamps):
        segment = audio[start * 1000:stop * 1000]  # Convert seconds to milliseconds for slicing
        
        # Define output filename for the segment
        output_filename = f"{os.path.splitext(filename)[0]}_segment_{index + 1}.wav"
        output_path = os.path.join(output_directory, output_filename)
        
        # Export the segment to a new .wav file
        segment.export(output_path, format="wav")
        print(f"Saved segment {index + 1} to {output_path}")

        # Extract and save the pause (between the previous stop and current start)
        if previous_stop < start:
            pause_segment = audio[previous_stop * 1000:start * 1000]  # Convert seconds to milliseconds for slicing
            pause_filename = f"{os.path.splitext(filename)[0]}_pause_{index}.wav"
            pause_path = os.path.join(pause_directory, pause_filename)
            pause_segment.export(pause_path, format="wav")
            print(f"Saved pause {index} to {pause_path}")

        previous_stop = stop  # Update previous stop to the current stop time

# Handle any remaining audio after the last stop timestamp
    if previous_stop < len(audio) / 1000:
        pause_segment = audio[previous_stop * 1000:] 
