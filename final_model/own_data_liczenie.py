import os
import re
import wave

def get_wave_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration_seconds = n_frames / frame_rate
        return duration_seconds

def extract_numbers_from_name(file_name):
    # Znajduje liczby w nazwie pliku
    return [int(num) for num in re.findall(r'\d+', file_name)]

def analyze_wav_files(folder_path):
    total_duration_seconds = 0
    total_sum_of_numbers = 0
    number_of_files = 0

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            # Oblicza długość pliku WAV
            duration_seconds = get_wave_duration(file_path)
            total_duration_seconds += duration_seconds

            # Sumuje liczby w nazwie pliku
            numbers = extract_numbers_from_name(file_name)
            total_sum_of_numbers += sum(numbers)
            
            number_of_files += 1
    
    total_duration_hours = total_duration_seconds / 3600

    return total_duration_hours, total_sum_of_numbers, number_of_files

# Ustaw ścieżkę do folderu zawierającego pliki WAV
folder_path = 'own_data/'

# Analizuje pliki WAV
duration_hours, sum_of_numbers, file_count = analyze_wav_files(folder_path)

print(f"Długość wszystkich plików WAV: {duration_hours:.2f} godzin")
print(f"Suma liczb w nazwach plików: {sum_of_numbers}")
print(f"Liczba plików WAV: {file_count}")
