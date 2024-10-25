import os
import wave
import pyaudio
import keyboard

def play_audio(file_path):
    """Play the audio file and allow timestamp recording."""
    # Open the wave file
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Initialize variables
    start_time = None
    timestamps = []
    recording = False

    print("Press SPACE to mark start/stop timestamps.")
    print("Press 'q' to quit playback.")

    # Read data in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Start playing the audio
    while data:
        stream.write(data)

        # Check if the space bar is pressed
        if keyboard.is_pressed('space'):
            if not recording:
                # Mark start time
                start_time = wf.getnframes() / wf.getframerate()  # Convert frames to seconds
                timestamps.append(f"Start: {start_time:.2f} seconds")
                recording = True
                print(f"Start timestamp recorded: {start_time:.2f} seconds")
            else:
                # Mark stop time
                stop_time = wf.getnframes() / wf.getframerate()  # Convert frames to seconds
                timestamps.append(f"Stop: {stop_time:.2f} seconds")
                recording = False
                print(f"Stop timestamp recorded: {stop_time:.2f} seconds")

        # Check if 'q' is pressed to quit playback
        if keyboard.is_pressed('q'):
            break

        # Read the next chunk of audio data
        data = wf.readframes(chunk_size)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save timestamps to a text file
    save_timestamps(file_path, timestamps)

def save_timestamps(file_path, timestamps):
    """Save timestamps to a text file with the same name as the audio file."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    txt_file_path = os.path.join(os.path.dirname(file_path), f"{base_name}_timestamps.txt")

    with open(txt_file_path, 'w') as f:
        for timestamp in timestamps:
            f.write(timestamp + '/n')
    
    print(f"Timestamps saved to {txt_file_path}")

def main():
    # Set the directory containing the WAV files
    wav_directory = 'C:/Users/u144572/self_development/sed_online/compare_ml_with_decision_tree/own_data_test'  # Change this to your directory

    # Get a list of all WAV files in the directory
    wav_files = [f for f in os.listdir(wav_directory) if f.endswith('.wav')]

    # Play each WAV file
    for wav_file in wav_files:
        print(f"Playing {wav_file}...")
        play_audio(os.path.join(wav_directory, wav_file))

if __name__ == "__main__":
    main()   
