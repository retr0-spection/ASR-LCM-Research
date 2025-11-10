import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from audiocraft.models import MultiBandDiffusion
from audiocraft.data.audio import audio_read, audio_write
import logging

# import degrade utils
from preprocessing import downsample_and_save, read_audio

# ---- CONFIG ----
device = "cuda" if torch.cuda.is_available() else "cpu"
BANDWIDTH = 3.0
NUM_WORKERS = 4
TARGET_SR = 12000

TEST_DATA = '/datasets/onailana/LibriTTS/test-clean'
UPSAMPLED_AUDIO_DIR = '/datasets/onailana/upsampled_mbd_audio_test'
CODES_DIR = '/datasets/onailana/test_codes'
DEGRADED_AUDIO_DIR = '/datasets/onailana/test_degraded_audio'
CSV_PATH = os.path.join(CODES_DIR, 'test_code_data.csv')
LOG_FILE = 'processing_test.log'

# ---- SETUP LOGGING ----
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("Starting test audio coding pipeline.")

# ---- CREATE DIRS ----
os.makedirs(UPSAMPLED_AUDIO_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)
os.makedirs(DEGRADED_AUDIO_DIR, exist_ok=True)

# ---- LOAD MODELS ----
mbd = MultiBandDiffusion.get_mbd_24khz(bw=BANDWIDTH)
encodec_24khz = EncodecModel.encodec_model_24khz().to(device)

# ---- HELPERS ----
def get_degraded_codes(path):
    with torch.no_grad():
        wav, sr = audio_read(path)
        wav = convert_audio(wav, sr, encodec_24khz.sample_rate, encodec_24khz.channels)
        wav = wav.unsqueeze(0).to(device)
        encoded_frames = encodec_24khz.encode(wav)
        return encoded_frames[0][0], sr

def upsample_24kHzcodes_with_mbd(encoded_frames_12khz):
    with torch.no_grad():
        encoded_frames_12khz = encoded_frames_12khz.to(device)
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
    except RuntimeError:
        logging.warning(f"Loudness normalization failed for {clip_name}, saving raw waveform")
        audio_write(
            filename,
            reconstructed_wav_24khz.squeeze(0).cpu(),
            mbd.sample_rate,
            strategy=None,
            loudness_compressor=False
        )

    return filename, reconstructed_wav_24khz, encoded_frames_24khz

def process_file(filepath):
    try:
        filename = os.path.basename(filepath)
        clip_name = filename.replace('.wav', '')

        # get duration from original
        audio_info = read_audio(filepath)
        duration = audio_info["duration"]

        # degraded file
        degraded_path = os.path.join(DEGRADED_AUDIO_DIR, f"{clip_name}_downsampled_{TARGET_SR}.wav")
        if not os.path.exists(degraded_path):
            downsample_and_save(filepath, degraded_path, TARGET_SR)

        degraded_codes, sr = get_degraded_codes(degraded_path)
        upsampled_path, reconstructed_wav, upsampled_encoded_frames = generate_using_mbd(degraded_codes, clip_name)

        # ensure same time length
        min_len = min(degraded_codes.shape[-1], upsampled_encoded_frames.shape[-1])
        degraded_codes = degraded_codes[..., :min_len]
        upsampled_encoded_frames = upsampled_encoded_frames[..., :min_len]

        # create mask
        mask = torch.ones(min_len, dtype=torch.bool)

        clip_path = os.path.join(CODES_DIR, f"{clip_name}.pt")
        torch.save({
            "degraded": degraded_codes.cpu(),
            "teacher": upsampled_encoded_frames.cpu(),
            "mask": mask
        }, clip_path)

        logging.info(f"Processed {filename} -> {clip_path}")
        return filename, filepath, degraded_path, upsampled_path, clip_path, duration

    except Exception as e:
        logging.error(f"Failed {filepath}: {e}")
        return None

# ---- LOAD PREVIOUS CSV ----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    processed_files = set(df['filename'].tolist())
else:
    df = pd.DataFrame(columns=[
        'filename', 'original_wav', 'degraded_wav', 'upsampled_wav', 'code_path', 'duration'
    ])
    processed_files = set()

# ---- COLLECT FILES ----
all_files = []
for dirpath, _, filenames in os.walk(TEST_DATA):
    for f in filenames:
        if f.endswith('.wav') and f not in processed_files:
            all_files.append(os.path.join(dirpath, f))

total_files = len(all_files)
logging.info(f"Total test files to process: {total_files}")

# ---- PARALLEL PROCESSING ----
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_file, f): f for f in all_files}

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            filename, original_path, degraded_path, upsampled_path, clip_path, duration = result
            df = pd.concat([df, pd.DataFrame({
                'filename':[filename],
                'original_wav':[original_path],
                'degraded_wav':[degraded_path],
                'upsampled_wav':[upsampled_path],
                'code_path':[clip_path],
                'duration':[duration]
            })], ignore_index=True)
            df.to_csv(CSV_PATH, index=False, na_rep='N/A')
            logging.info(f"Saved CSV progress ({i}/{total_files})")

logging.info("Test audio coding pipeline completed.")

