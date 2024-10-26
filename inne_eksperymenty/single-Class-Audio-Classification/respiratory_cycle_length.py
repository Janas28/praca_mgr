import os

def calculate_average_event_length(txt_file):
    # Read the description file
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    total_duration = 0
    event_count = len(lines)
    
    # Process each line in the description file
    for line in lines:
        start_time, end_time, _, _ = line.strip().split()
        start_time = float(start_time) * 1000  # Convert to milliseconds
        end_time = float(end_time) * 1000      # Convert to milliseconds
        
        # Calculate the duration of the event
        duration = end_time - start_time
        total_duration += duration
    
    # Calculate average event length in seconds
    average_duration = (total_duration / event_count) / 1000  # Convert back to seconds
    return average_duration

def process_directory_for_average(directory):
    total_average_duration = 0
    processed_files = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Construct the full path to the txt file
            txt_file_path = os.path.join(directory, filename)
            
            # Call the function to calculate the average event length
            average_duration = calculate_average_event_length(txt_file_path)
            total_average_duration += average_duration
            processed_files += 1
            print(f"Average event length for {filename}: {average_duration:.2f} seconds")
    
    if processed_files > 0:
        overall_average_duration = total_average_duration / processed_files
        print(f"\nOverall average event length across all files: {overall_average_duration:.2f} seconds")
    else:
        print("No TXT files processed.")

directory_path = 'C:/Users/u144572/self_development/single-Class-Audio-Classification/audio_and_txt_files/'
process_directory_for_average(directory_path)
