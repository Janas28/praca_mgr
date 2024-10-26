import os
import shutil
import random

def move_files_with_matching_names(src_dir, dst_dir, percentage=10):
    # List all files in the source directory
    files = os.listdir(src_dir)

    # Create a set of base filenames (without extensions)
    base_files = set()
    for file in files:
        if file.endswith('.wav') or file.endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            base_files.add(base_name)

    # Convert the set to a list and shuffle it
    base_files = list(base_files)
    random.shuffle(base_files)

    # Calculate the number of files to move
    num_files_to_move = int(len(base_files) * (percentage / 100))

    # Select the files to move
    files_to_move = base_files[:num_files_to_move]

    # Ensure destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Move the selected files and their counterparts
    for base_name in files_to_move:
        wav_file = base_name + '.wav'
        txt_file = base_name + '.txt'

        # Move the wav file if it exists
        if wav_file in files:
            shutil.move(os.path.join(src_dir, wav_file), os.path.join(dst_dir, wav_file))

        # Move the txt file if it exists
        if txt_file in files:
            shutil.move(os.path.join(src_dir, txt_file), os.path.join(dst_dir, txt_file))

    print(f"Moved {len(files_to_move)} sets of files to {dst_dir}")

# Example usage
source_directory = 'audio_and_txt_files_train_val'  # Replace with the path to your source directory
destination_directory = 'audio_and_txt_files_test'  # Replace with the path to your destination directory

move_files_with_matching_names(source_directory, destination_directory)

