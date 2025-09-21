import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from audiocraft.models import MultiBandDiffusion
from audiocraft.data.audio import audio_read, audio_write
import logging

# ---- CONFIG ----
device = "cuda" if torch.cuda.is_available() else "cpu"
BANDWIDTH = 3.0
NUM_WORKERS = 4  # adjust based on GPU memory
TRAINING_DATA = '/datasets/onailana/training_data'
UPSAMPLED_AUDIO_DIR = '/datasets/onailana/upsampled_mbd_audio'
CODES_DIR = '/datasets/onailana/codes'
CSV_PATH = os.path.join(CODES_DIR, 'code_data.csv')
LOG_FILE = 'processing.log'

# ---- SETUP LOGGING ----
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("Starting audio coding pipeline.")

# ---- CREATE DIRS IF MISSING ----
os.makedirs(UPSAMPLED_AUDIO_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

# ---- LOAD MODELS ----
mbd = MultiBandDiffusion.get_mbd_24khz(bw=BANDWIDTH)  # internally handles CUDA
encodec_24khz = EncodecModel.encodec_model_24khz().to(device)

# ---- HELPER FUNCTIONS ----
def get_degraded_codes(path):
    with torch.no_grad():
        wav, sr = audio_read(path)
        wav = convert_audio(wav, sr, encodec_24khz.sample_rate, encodec_24khz.channels)
        wav = wav.unsqueeze(0).to(device)
        encoded_frames = encodec_24khz.encode(wav)
        return encoded_frames[0][0], sr

def upsample_24kHzcodes_with_mbd(encoded_frames_12khz):
    with torch.no_grad():
        device_mbd = next(mbd.parameters()).device
        encoded_frames_12khz = encoded_frames_12khz.to(device_mbd)
        reconstructed_wav_24khz = mbd.tokens_to_wav(encoded_frames_12khz)
        encoded_frames_24khz = encodec_24khz.encode(reconstructed_wav_24khz.to(device))
        return reconstructed_wav_24khz, encoded_frames_24khz[0][0]

def generate_using_mbd(encoded_frames, clip_name):
    with torch.no_grad():
        reconstructed_wav_24khz, encoded_frames_24khz = upsample_24kHzcodes_with_mbd(encoded_frames)

    filename = os.path.join(UPSAMPLED_AUDIO_DIR, f'{clip_name}_mbd_upsampled_24000.wav')
    try:
        audio_write(
            filename,
            reconstructed_wav_24khz.squeeze(0).cpu(),
            mbd.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )
    except RuntimeError as e:
        logging.warning(f"Loudness normalization failed for {clip_name}, saving raw waveform")
        audio_write(
            filename,
            reconstructed_wav_24khz.squeeze(0).cpu(),
            mbd.sample_rate,
            strategy=None,
            loudness_compressor=False
        )

    return reconstructed_wav_24khz, encoded_frames_24khz

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        clip_name = ''.join([x + "_" if i < 2 else x for i, x in enumerate(filename.split('_')[:3])])

        degraded_codes, sr = get_degraded_codes(filepath)
        reconstructed_wav, upsampled_encoded_frames = generate_using_mbd(degraded_codes, clip_name)

        clip_path = os.path.join(CODES_DIR, f"{clip_name}.pt")
        torch.save({
            "degraded": degraded_codes.cpu(),
            "teacher": upsampled_encoded_frames.cpu()
        }, clip_path)

        logging.info(f"Processed {filename} -> {clip_path}")
        return filename, clip_path

    except Exception as e:
        logging.error(f"Failed {filepath}: {e}")
        return None

# ---- LOAD PREVIOUS CSV TO RESUME ----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    processed_files = set(df['filename'].tolist())
else:
    df = pd.DataFrame(columns=['filename', 'filepath'])
    processed_files = set()

# ---- COLLECT FILES ----
all_files = []
for dirpath, _, filenames in os.walk(TRAINING_DATA):
    for f in filenames:
        if 'downsampled' in f and f not in processed_files:
            all_files.append(os.path.join(dirpath, f))

total_files = len(all_files)
logging.info(f"Total files to process: {total_files}")

# ---- PARALLEL PROCESSING ----
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_file, f): f for f in all_files}

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            filename, clip_path = result
            df = pd.concat([df, pd.DataFrame({'filename':[filename],'filepath':[clip_path]})], ignore_index=True)
            df.to_csv(CSV_PATH, index=False, na_rep='N/A')
            logging.info(f"Saved CSV progress ({i}/{total_files})")

logging.info("Audio coding pipeline completed.")
