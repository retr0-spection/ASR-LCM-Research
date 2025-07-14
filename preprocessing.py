# Utils for preprocessing audio files
from typing import Dict
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import os

path = "LibriTTS/train-clean-100/19/198/19_198_000000_000000.wav"

def read_audio(path: str) -> Dict:
    """
    Reads an audio file and returns a dictionary containing the sample rate, data, and duration.
    """
    sample_rate, data = wavfile.read(path)
    duration = data.shape[0] / sample_rate
    return {"sample_rate": sample_rate, "data": data, "duration": duration}


def write_audio(path: str, audio: Dict):
    """
    Writes an audio file from a dictionary containing the sample rate, data, and duration.
    """
    wavfile.write(path, audio["sample_rate"], audio["data"])

def downsample_and_save(src: str,dst:str, target_sample_rate: int) -> None:
    """
    Downsamples an audio file from a dictionary containing the sample rate, data, and duration.
    """
    audio = read_audio(src)
    audio = normalise_audio(audio)

    original_sr = audio['sample_rate']
    data = audio['data']

    if original_sr != target_sample_rate:
        num_samples = int(data.shape[0] * target_sample_rate) // original_sr
        resampled_data = resample(data, num_samples)
    else:
        resampled_data = data

    wavfile.write(dst, target_sample_rate, resampled_data.astype(np.float32))

def normalise_audio(audio: Dict) -> Dict:
    """
    Normalises an audio file from a dictionary containing the sample rate, data, and duration.
    """
    max_value = max(abs(audio["data"]))
    if max_value > 0:
        audio["data"] = audio["data"] / max_value
    else:
        print("Warning: Audio data is zero")
    return audio

def traverse_and_downsample(src_root, dst_root, target_sample_rate):
    print("Preprocessing in progress")
    for root, dirs, files in os.walk(src_root):
        relpath = os.path.relpath(root, src_root)

        for file in files:
            if file.endswith(".wav"):
                src_path = os.path.join(root, file)
                new_src_path = os.path.join(dst_root, file)
                os.rename(src_path, new_src_path)
                dst_path = os.path.join(dst_root, file.replace(".wav", f"_downsampled_{target_sample_rate}.wav"))
                downsample_and_save(new_src_path,dst_path, target_sample_rate)

source_folder = "LibriTTS/train-clean-100"
destination_folder = "training_data"
target_sr = 12000

print("Starting preprocessing")

traverse_and_downsample(source_folder, destination_folder, target_sr)

print("Preprocessing complete :)")
