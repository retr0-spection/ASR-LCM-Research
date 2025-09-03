from encodec import EncodecModel
from encodec.utils import convert_audio
from audiocraft.models import MultiBandDiffusion
import torch
from audiocraft.data.audio import audio_read, audio_write
import os
import pandas as pd

bandwidth = 3.0  # 1.5, 3.0, 6.0
mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
encodec_24khz = EncodecModel.encodec_model_24khz()

def get_degraded_codes(path='training_data/19_198_000000_000000_downsampled_12000.wav'):
    with torch.no_grad():
        wav, sr = audio_read(path)
        wav = convert_audio(wav, sr, encodec_24khz.sample_rate, encodec_24khz.channels)
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            encoded_frames = encodec_24khz.encode(wav)

        return encoded_frames[0][0], sr

def upsample_24kHzcodes_with_mbd(encoded_frames_12khz):
    with torch.no_grad():
        reconstructed_wav_24khz = mbd.tokens_to_wav(encoded_frames_12khz)
        encoded_frames_24khz = encodec_24khz.encode(reconstructed_wav_24khz)

    return reconstructed_wav_24khz, encoded_frames_24khz[0][0]

def generate_using_mbd(encoded_frames):
    with torch.no_grad():
        reconstructed_wav_24khz, encoded_frames_24khz = upsample_24kHzcodes_with_mbd(encoded_frames)

    audio_write(
        'mbd_generated_from_codes',
        reconstructed_wav_24khz.squeeze(0).cpu(),
        mbd.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )

    print(f"[INFO] Audio generated and saved to 'mbd_generated_from_codes.wav'.")

    return reconstructed_wav_24khz, encoded_frames_24khz

# traverse training_data folder
print("-"*10, "Starting coding process", "-"*10)

training_data_path = 'training_data/'
df = pd.DataFrame({'filename':[],'filepath':[]})
total_files = sum(1 for _, _, filenames in os.walk(training_data_path) for f in filenames if 'downsampled' in f)
processed = 0

for dirpath, dirnames, filenames in os.walk(training_data_path):
    for filename in filenames:
        target = f"{dirpath}{filename}"
        if 'downsampled' in filename:
            print(f"[PROCESSING] {filename} ({processed+1}/{total_files})")
            degraded_codes, sr = get_degraded_codes(target)
            reconstructed_wav, upsampled_codes = generate_using_mbd(degraded_codes)
            # save for each clip
            clip_name = ''.join([x + "_" if i < 2 else x for i, x in enumerate(filename.split('_')[:3])])
            clip_path = f"codes/{clip_name}.pt"
            torch.save({
                "degraded": degraded_codes.cpu(),
                "teacher": upsampled_codes.cpu()
            }, clip_path)
            new_entry = pd.DataFrame({'filename': [filename], 'filepath':[clip_path]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv('code_data.csv',  index=False, na_rep='N/A')
            processed += 1
            print(f"[SAVED] {clip_path} (processed {processed}/{total_files})")

print("-"*10, "Coding process completed", "-"*10)
