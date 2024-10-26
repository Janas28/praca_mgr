from pydub import AudioSegment
import os

def split_wav_by_events(wav_file, txt_file, output_dir):
    # Load the audio file
    audio = AudioSegment.from_wav(wav_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the description file
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    # Process each line in the description file
    for i, line in enumerate(lines):
        start_time, end_time, _, _ = line.strip().split()
        start_time = float(start_time) * 1000  # Convert to milliseconds
        end_time = float(end_time) * 1000      # Convert to milliseconds
        
        # Extract the segment
        segment = audio[start_time:end_time]
        
        # Define output file name
        if i < 10:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(wav_file))[0]}_segment_0{i+1}.wav")
        else:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(wav_file))[0]}_segment_{i+1}.wav")
        
        # Export the segment
        segment.export(output_file, format="wav")

def process_directory(directory, output_base_dir):
    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            # Construct the full path to the wav file and its corresponding txt file
            wav_file_path = os.path.join(directory, filename)
            txt_file_path = os.path.splitext(wav_file_path)[0] + ".txt"
            
            # Ensure the txt file exists
            if os.path.exists(txt_file_path):
                # Create an output directory for the segments of this wav file
                # output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])
                # os.makedirs(output_dir, exist_ok=True)
                
                # Call the split function
                split_wav_by_events(wav_file_path, txt_file_path, output_base_dir)
            else:
                print(f"Warning: No corresponding TXT file found for {filename}")

# Example usage
directory_path = 'C:/Users/u144572/self_development/single-Class-Audio-Classification/audio_and_txt_files/'
output_base_dir = 'C:/Users/u144572/self_development/single-Class-Audio-Classification/single_cycle/'
process_directory(directory_path, output_base_dir)

