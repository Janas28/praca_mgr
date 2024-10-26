import pyaudio
import numpy as np
import time
from collections import deque

p = pyaudio.PyAudio()

CHANNELS = 2
RATE = 44100

CHUNK = 1024
RECORD_SECONDS = 3
MFCC_FEATURES = 50  # Number of MFCC features

# Create a queue to store audio data (max length for 20 seconds)
audio_queue = deque(maxlen=int(RATE / CHUNK * RECORD_SECONDS))

def callback(in_data, frame_count, time_info, flag):
    # using Numpy to convert to array for processing
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_queue.append(audio_data)
    if len(audio_queue) == audio_queue.maxlen:
        print("dupa")
        time.sleep(1)
        return in_data, pyaudio.paContinue

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(20)
    stream.stop_stream()
    print("Stream is stopped")

stream.close()

p.terminate()