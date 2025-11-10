import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
from audiocraft.models import MultiBandDiffusion
import logging

# ---- CONFIG ----
device = "cuda" if torch.cuda.is_available() else "cpu"
BANDWIDTH = 3.0
NUM_WORKERS = 4
TARGET_SR = 12000

TEST_DATA = '/datasets/onailana/train-val'
CODES_DIR = '/datasets/onailana/val_codes'
CSV_PATH = os.path.join(CODES_DIR, 'val_code_data.csv')
LOG_FILE = 'processing_val.log'

os.makedirs(CODES_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("Starting test/val audio coding pipeline.")

# ---- LOAD MODELS ----
mbd = MultiBandDiffusion.get_mbd_24khz(bw=BANDWIDTH)
encodec_24khz = EncodecModel.encodec_model_24khz().to(device)

# ---- HELPERS ----
def get_degraded_codes(path):
    with torch.no_grad():
        wav, sr = convert_audio_from_file(path)
        wav = wav.unsqueeze(0).to(device)
        encoded_frames = encodec_24khz.encode(wav)
        return encoded_frames[0][0]

def convert_audio_from_file(path):
    # minimal version of convert_audio + read
    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = convert_audio(wav, sr, encodec_24khz.sample_rate, encodec_24khz.channels)
    return wav, encodec_24khz.sample_rate

def upsample_with_mbd(encoded_frames):
    with torch.no_grad():
        encoded_frames = encoded_frames.to(device)
        reconstructed_wav_24khz = mbd.tokens_to_wav(encoded_frames)
        upsampled_codes = encodec_24khz.encode(reconstructed_wav_24khz)[0][0]
        return upsampled_codes

def process_file(filepath):
    try:
        clip_name = os.path.basename(filepath).replace('.wav','')

        # degraded codes
        degraded_codes = get_degraded_codes(filepath)

        # upsampled codes
        upsampled_codes = upsample_with_mbd(degraded_codes)

        # ensure same length
        min_len = min(degraded_codes.shape[-1], upsampled_codes.shape[-1])
        degraded_codes = degraded_codes[..., :min_len]
        upsampled_codes = upsampled_codes[..., :min_len]

        # mask
        mask = torch.ones(min_len, dtype=torch.bool)

        # save codes
        clip_path = os.path.join(CODES_DIR, f"{clip_name}.pt")
        torch.save({
            "degraded": degraded_codes.cpu(),
            "teacher": upsampled_codes.cpu(),
            "mask": mask
        }, clip_path)

        logging.info(f"Processed {clip_name} -> {clip_path}")
        return clip_name, filepath, clip_path

    except Exception as e:
        logging.error(f"Failed {filepath}: {e}")
        return None

# ---- LOAD PREVIOUS CSV ----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    processed_files = set(df['filename'].tolist())
else:
    df = pd.DataFrame(columns=['filename','original_wav','code_path'])
    processed_files = set()

# ---- COLLECT FILES ----
all_files = [
    os.path.join(dp, f) for dp, _, files in os.walk(TEST_DATA)
    for f in files if f.endswith('.wav') and f not in processed_files
]

logging.info(f"Total files to process: {len(all_files)}")

# ---- PARALLEL PROCESSING ----
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_file, f): f for f in all_files}

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            filename, original_wav, code_path = result
            df = pd.concat([df, pd.DataFrame({
                'filename':[filename],
                'original_wav':[original_wav],
                'code_path':[code_path]
            })], ignore_index=True)
            df.to_csv(CSV_PATH, index=False)
            logging.info(f"Saved CSV progress ({i}/{len(all_files)})")

logging.info("Test/val audio coding pipeline completed.")

