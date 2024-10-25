import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Wczytaj plik WAV
sample_rate, data = wavfile.read('compare_ml_with_decision_tree/own_data_micro_pg_clear/12kmdwa.wav')

# Zapamiętaj oryginalny typ danych
original_dtype = data.dtype

# Sprawdź, czy sygnał jest mono czy stereo
if len(data.shape) == 2:
    # Jeśli stereo, wybierz jeden kanał
    data = data[:, 0]

# Konwertuj dane do typu float32
data = data.astype(np.float32)

# Normalizuj amplitudę do zakresu od -1 do 1
# if original_dtype == np.int16:
#     max_val = 2 ** 15
#     data = data / max_val
# elif original_dtype == np.int32:
#     max_val = 2 ** 31
#     data = data / max_val
# elif original_dtype == np.uint8:
#     data = data - 128  # Przesunięcie środka sygnału
#     max_val = 128
#     data = data / max_val
# else:
max_val = np.max(np.abs(data))
data = data / max_val

# Liczba próbek
N = len(data)

# Wektor czasu
t = np.linspace(0, N / sample_rate, N, endpoint=False)

# Wykres sygnału wejściowego w dziedzinie czasu
plt.figure(figsize=(12, 4))
plt.plot(t, data)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title('Sygnał wejściowy w dziedzinie czasu.')
plt.grid(True)
plt.show()

# Oblicz transformatę Fouriera
fft_data = np.fft.fft(data)
# Wektor częstotliwości
freq = np.fft.fftfreq(N, d=1/sample_rate)

# Pobierz tylko dodatnie częstotliwości
mask = freq >= 0
freq = freq[mask]
fft_data = fft_data[mask]

# Definiuj maksymalną częstotliwość do wyświetlenia
max_freq = 5100  # w Hz

# Filtruj częstotliwości i amplitudy do zakresu 0 - max_freq
freq_filtered = freq[freq <= max_freq]
fft_data_filtered = fft_data[freq <= max_freq]

# Wykres widma sygnału z ograniczonym zakresem częstotliwości
plt.figure(figsize=(12, 4))
plt.plot(freq_filtered, np.abs(fft_data_filtered))
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.title(f'Widmo sygnału po transformacji Fouriera.')
plt.grid(True)
plt.show()
