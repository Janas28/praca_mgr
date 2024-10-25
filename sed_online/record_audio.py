import pyaudio
import wave
import random
# Parameters for recording
FORMAT = pyaudio.paInt16  # Format of the audio (16-bit)
CHANNELS = 1              # Number of channels (1 for mono, 2 for stereo)
RATE = 44100              # Sampling rate (44.1 kHz)
CHUNK = 1024 * 2              # Buffer size (number of frames per buffer)
RECORD_SECONDS = 25       # Duration of the recording (in seconds)
OUTPUT_FILENAME = "record_noise3.wav"  # Output file name

for i in range(0, 10):
    # Create an instance of PyAudio
    audio = pyaudio.PyAudio()

    # Open a stream for audio input
    stream = audio.open(
        format=FORMAT, 
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Recording...")

    frames = []

    # Record the audio in chunks
    for _ in range(0, int(RATE / CHUNK * (RECORD_SECONDS + 
    random.randint(2, 7)))):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio instance
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(input("write file name") + ".wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {OUTPUT_FILENAME}")
