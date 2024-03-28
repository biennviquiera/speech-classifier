import subprocess
import os
import json
import re
import sys

if len(sys.argv) < 3:
    print("Usage: normalize_audio_in_dir.py <source dir> <output dir>")
    sys.exit(1)  


def analyze_loudness(file_path):
    """Analyze the loudness of an audio file using FFmpeg and return the loudness stats as a dictionary."""
    cmd = [
        'ffmpeg', '-i', file_path, '-filter_complex',
        'loudnorm=I=-23:LRA=7:tp=-2:print_format=json', '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)
    try:
        # Adjusted regex to match multi-line JSON
        json_str_match = re.search(r'(\{.+\})', result.stderr, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(1)
            loudness_stats = json.loads(json_str)
            return loudness_stats
        else:
            print("No JSON data found in FFmpeg output.")
            return None
    except json.JSONDecodeError:
        print("Failed to decode JSON from FFmpeg output.")
        return None
    
def normalize_audio(file_path, output_dir, filename, loudness_stats):
    """Normalize an audio file based on loudness analysis using FFmpeg and save the output."""
    output_path = os.path.join(output_dir, filename)
    filter_complex = (
        f"loudnorm=I=-23:LRA=7:tp=-2:measured_I={loudness_stats['input_i']}:"
        f"measured_LRA={loudness_stats['input_lra']}:measured_tp={loudness_stats['input_tp']}:"
        f"measured_thresh={loudness_stats['input_thresh']}:linear=true:print_format=summary"
    )
    cmd = [
        'ffmpeg', '-i', file_path, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
        '-af', filter_complex, output_path
    ]
    subprocess.run(cmd, text=True)

directory = sys.argv[1]
output_directory = sys.argv[2]
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory):
    if filename.endswith(('.wav', '.mp3', '.flac')):
        file_path = os.path.join(directory, filename)
        
        loudness_stats = analyze_loudness(file_path)
        
        # Normalize audio based on loudness analysis
        print(f"processing file: {file_path}")
        normalize_audio(file_path, output_directory, filename, loudness_stats)

        print(f"Processed: {filename}")
