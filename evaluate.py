import os
import time
import torch
from torch.utils.data import DataLoader
from lcm2 import LCM_MCB, collate_fn
from dataset import LCMDistillDataset
from encodec import EncodecModel
from audiocraft.models import MultiBandDiffusion
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from pystoi import stoi
import soundfile as sf
import logging
import pandas as pd
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchaudio

# ------------------------------
# Config
# ------------------------------
CHECKPOINT_PATH = "/datasets/onailana/checkpoints1/lcm_epoch20.pt"
DATASET_CSV = "/datasets/onailana/test_codes/test_code_data.csv"
UPSAMPLED_DIR = "/datasets/onailana/upsampled_lcm_vs_teacher"
CSV_PATH = os.path.join(UPSAMPLED_DIR, "evaluation_results.csv")
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000  # Hz

os.makedirs(UPSAMPLED_DIR, exist_ok=True)

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ------------------------------
# Helper Functions
# ------------------------------
def generate_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Generate mel spectrogram from audio"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    mel_spec = mel_transform(audio_tensor)
    mel_db = 20 * torch.log10(torch.clamp(mel_spec, min=1e-9))
    
    # Convert to image
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    
    # Convert to wandb Image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return wandb.Image(img_array)

def align_audio_length(audio, target_length):
    """Ensure audio has target length by padding or truncating"""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    else:
        return audio[:target_length]

# ------------------------------
# WandB init
# ------------------------------
wandb.init(
    project="audio-super-resolution",
    id="final-eval-1",
    name="LCM_vs_Teacher_eval_epoch20_short",
    config={
        "checkpoint": CHECKPOINT_PATH,
        "batch_size": BATCH_SIZE,
        "device": DEVICE
    }
)

# ------------------------------
# Load Dataset and CSV
# ------------------------------
dataset = LCMDistillDataset(DATASET_CSV)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
df_csv = pd.read_csv(DATASET_CSV)

# ------------------------------
# Load Models
# ------------------------------
logger.info("Loading LCM model...")
n_codebooks = 32
vocab_size = 1024
d_model = 256
n_heads = 8
num_layers = 6

lcm_model = LCM_MCB(
    vocab_size=vocab_size,
    n_codebooks=n_codebooks,
    d_model=d_model,
    n_heads=n_heads,
    num_layers=num_layers
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
lcm_model.load_state_dict(checkpoint['model_state_dict'])
lcm_model.eval()

logger.info("Loading MultiBandDiffusion teacher model...")
teacher_model = MultiBandDiffusion.get_mbd_24khz()

logger.info("Loading Encodec model...")
encodec_model = EncodecModel.encodec_model_24khz().to(DEVICE)

# Metrics
si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

# ------------------------------
# Evaluation loop
# ------------------------------
logger.info("Starting evaluation...")
results = []
lcm_inference_times = []
teacher_inference_times = []

# For continuous plotting
all_si_sdr_lcm = []
all_si_sdr_teacher = []
all_stoi_lcm = []
all_stoi_teacher = []
sample_indices = []

count = 0

with torch.no_grad():
    for batch_idx, (degraded, teacher_codes, mask) in enumerate(tqdm(dataloader, desc="Evaluating batches")):
        degraded = degraded.to(DEVICE)
        teacher_codes = teacher_codes.transpose(1, 2).to(DEVICE)
        mask = mask.to(DEVICE)

        # ===== LCM Inference =====
        start_time = time.time()
        logits = lcm_model(degraded, attention_mask=mask)
        pred_tokens = torch.argmax(logits, dim=2).transpose(1, 2).to(DEVICE)
        lcm_inference_time = time.time() - start_time
        lcm_inference_times.append(lcm_inference_time)

        # ===== Teacher Inference (timing only) =====
        # Get embeddings for teacher model
        degraded_transposed = degraded.transpose(1, 2).to(DEVICE)
        emb = encodec_model.quantizer.decode(degraded_transposed)
        
        start_time = time.time()
        _ = teacher_model.generate(emb)  # Generate only for timing
        teacher_inference_time = time.time() - start_time
        teacher_inference_times.append(teacher_inference_time)

        # Decode audio for LCM and degraded
        lcm_wav = encodec_model.decode([(pred_tokens, None)])
        deg_wav = encodec_model.decode([(degraded_transposed, None)])

        # Process each sample in batch
        for i in range(lcm_wav.shape[0]):
            row_idx = batch_idx * BATCH_SIZE + i
            if row_idx >= len(df_csv):
                continue

            original_path = df_csv.loc[row_idx, 'original_wav']
            teacher_path_csv = df_csv.loc[row_idx, 'upsampled_wav']  # Pre-generated teacher audio
            code_path = df_csv.loc[row_idx, 'code_path']
            duration = df_csv.loc[row_idx, 'duration']
            filename = os.path.basename(original_path)
            num_samples = int(duration * SAMPLE_RATE)

            # Load original
            orig_wav, sr = sf.read(original_path)
            if sr != SAMPLE_RATE:
                raise ValueError(f"Original WAV sample rate {sr} != {SAMPLE_RATE}")
            orig_wav = orig_wav[:num_samples].astype('float32')

            # Load pre-generated teacher audio
            teacher_audio, sr_teacher = sf.read(teacher_path_csv)
            if sr_teacher != SAMPLE_RATE:
                raise ValueError(f"Teacher WAV sample rate {sr_teacher} != {SAMPLE_RATE}")
            teacher_audio = teacher_audio[:num_samples].astype('float32')

            # Extract and align audio
            degraded_wav = deg_wav[i].cpu().squeeze().numpy().astype('float32')
            lcm_audio = lcm_wav[i].cpu().squeeze().numpy().astype('float32')
            
            # Align to same length
            degraded_wav = align_audio_length(degraded_wav, num_samples)
            lcm_audio = align_audio_length(lcm_audio, num_samples)
            teacher_audio = align_audio_length(teacher_audio, num_samples)

            # Compute metrics for LCM
            try:
                si_sdr_lcm = si_sdr_metric(
                    torch.tensor(lcm_audio).unsqueeze(0).to(DEVICE),
                    torch.tensor(orig_wav).unsqueeze(0).to(DEVICE)
                ).item()
                stoi_lcm = stoi(orig_wav, lcm_audio, SAMPLE_RATE, extended=False)
            except Exception as e:
                logger.warning(f"LCM metric computation failed: {e}")
                si_sdr_lcm, stoi_lcm = None, None

            # Compute metrics for Teacher
            try:
                si_sdr_teacher = si_sdr_metric(
                    torch.tensor(teacher_audio).unsqueeze(0).to(DEVICE),
                    torch.tensor(orig_wav).unsqueeze(0).to(DEVICE)
                ).item()
                stoi_teacher = stoi(orig_wav, teacher_audio, SAMPLE_RATE, extended=False)
            except Exception as e:
                logger.warning(f"Teacher metric computation failed: {e}")
                si_sdr_teacher, stoi_teacher = None, None

            # Save audio files (only LCM and degraded - teacher already exists)
            lcm_path = os.path.join(UPSAMPLED_DIR, f"sample_{row_idx}_lcm.wav")
            degraded_path = os.path.join(UPSAMPLED_DIR, f"sample_{row_idx}_degraded.wav")
            
            sf.write(lcm_path, lcm_audio, SAMPLE_RATE)
            sf.write(degraded_path, degraded_wav, SAMPLE_RATE)

            # Store for continuous plotting
            sample_indices.append(row_idx)
            all_si_sdr_lcm.append(si_sdr_lcm if si_sdr_lcm is not None else 0)
            all_si_sdr_teacher.append(si_sdr_teacher if si_sdr_teacher is not None else 0)
            all_stoi_lcm.append(stoi_lcm if stoi_lcm is not None else 0)
            all_stoi_teacher.append(stoi_teacher if stoi_teacher is not None else 0)

            # Append results
            results.append({
                "sample_idx": row_idx,
                "filename": filename,
                "si_sdr_lcm": si_sdr_lcm,
                "si_sdr_teacher": si_sdr_teacher,
                "stoi_lcm": stoi_lcm,
                "stoi_teacher": stoi_teacher,
                "lcm_inference_time": lcm_inference_time,
                "teacher_inference_time": teacher_inference_time,
                "original_wav": original_path,
                "lcm_wav_path": lcm_path,
                "teacher_wav_path": teacher_path_csv,  # Reference pre-generated file
                "degraded_wav_path": degraded_path,
                "code_path": code_path,
                "duration": duration
            })

            # Generate mel spectrograms for first 5 samples
            if row_idx < 5:
                mel_orig = generate_mel_spectrogram(orig_wav)
                mel_degraded = generate_mel_spectrogram(degraded_wav)
                mel_lcm = generate_mel_spectrogram(lcm_audio)
                mel_teacher = generate_mel_spectrogram(teacher_audio)

                wandb.log({
                    f"sample_{row_idx}/mel_original": mel_orig,
                    f"sample_{row_idx}/mel_degraded": mel_degraded,
                    f"sample_{row_idx}/mel_lcm": mel_lcm,
                    f"sample_{row_idx}/mel_teacher": mel_teacher,
                    f"sample_{row_idx}/audio_original": wandb.Audio(orig_wav, sample_rate=SAMPLE_RATE),
                    f"sample_{row_idx}/audio_degraded": wandb.Audio(degraded_wav, sample_rate=SAMPLE_RATE),
                    f"sample_{row_idx}/audio_lcm": wandb.Audio(lcm_audio, sample_rate=SAMPLE_RATE),
                    f"sample_{row_idx}/audio_teacher": wandb.Audio(teacher_audio, sample_rate=SAMPLE_RATE),
                })

        # Log batch metrics
        wandb.log({
            "batch": batch_idx,
            "lcm_batch_inference_time": lcm_inference_time,
            "teacher_batch_inference_time": teacher_inference_time,
            "inference_time_ratio": lcm_inference_time / teacher_inference_time if teacher_inference_time > 0 else 0,
        })

        count += 1
        if count > 99:
            break

# ------------------------------
# Summary metrics
# ------------------------------
valid_sdr_lcm = [r['si_sdr_lcm'] for r in results if r['si_sdr_lcm'] is not None]
valid_sdr_teacher = [r['si_sdr_teacher'] for r in results if r['si_sdr_teacher'] is not None]
valid_stoi_lcm = [r['stoi_lcm'] for r in results if r['stoi_lcm'] is not None]
valid_stoi_teacher = [r['stoi_teacher'] for r in results if r['stoi_teacher'] is not None]

si_sdr_avg_lcm = sum(valid_sdr_lcm) / len(valid_sdr_lcm) if valid_sdr_lcm else None
si_sdr_avg_teacher = sum(valid_sdr_teacher) / len(valid_sdr_teacher) if valid_sdr_teacher else None
stoi_avg_lcm = sum(valid_stoi_lcm) / len(valid_stoi_lcm) if valid_stoi_lcm else None
stoi_avg_teacher = sum(valid_stoi_teacher) / len(valid_stoi_teacher) if valid_stoi_teacher else None
avg_inference_time_lcm = sum(lcm_inference_times) / len(lcm_inference_times) if lcm_inference_times else None
avg_inference_time_teacher = sum(teacher_inference_times) / len(teacher_inference_times) if teacher_inference_times else None

logger.info(f"LCM - Average SI-SDR: {si_sdr_avg_lcm:.3f}, Average STOI: {stoi_avg_lcm:.3f}")
logger.info(f"Teacher - Average SI-SDR: {si_sdr_avg_teacher:.3f}, Average STOI: {stoi_avg_teacher:.3f}")
logger.info(f"LCM avg inference time: {avg_inference_time_lcm:.3f}s vs Teacher: {avg_inference_time_teacher:.3f}s")
logger.info(f"Speedup: {avg_inference_time_teacher / avg_inference_time_lcm:.2f}x")

# Create continuous plots
fig_sisdr, ax_sisdr = plt.subplots(figsize=(12, 6))
ax_sisdr.plot(sample_indices, all_si_sdr_lcm, label='LCM', marker='o', alpha=0.7)
ax_sisdr.plot(sample_indices, all_si_sdr_teacher, label='Teacher (MBD)', marker='s', alpha=0.7)
ax_sisdr.set_xlabel('Sample Index')
ax_sisdr.set_ylabel('SI-SDR (dB)')
ax_sisdr.set_title('SI-SDR Comparison: LCM vs Teacher')
ax_sisdr.legend()
ax_sisdr.grid(True, alpha=0.3)

fig_stoi, ax_stoi = plt.subplots(figsize=(12, 6))
ax_stoi.plot(sample_indices, all_stoi_lcm, label='LCM', marker='o', alpha=0.7)
ax_stoi.plot(sample_indices, all_stoi_teacher, label='Teacher (MBD)', marker='s', alpha=0.7)
ax_stoi.set_xlabel('Sample Index')
ax_stoi.set_ylabel('STOI')
ax_stoi.set_title('STOI Comparison: LCM vs Teacher')
ax_stoi.legend()
ax_stoi.grid(True, alpha=0.3)

# Log summary to wandb
wandb.log({
    "avg_si_sdr_lcm": si_sdr_avg_lcm,
    "avg_si_sdr_teacher": si_sdr_avg_teacher,
    "avg_stoi_lcm": stoi_avg_lcm,
    "avg_stoi_teacher": stoi_avg_teacher,
    "avg_inference_time_lcm": avg_inference_time_lcm,
    "avg_inference_time_teacher": avg_inference_time_teacher,
    "speedup_factor": avg_inference_time_teacher / avg_inference_time_lcm if avg_inference_time_lcm else None,
    "continuous_si_sdr_plot": wandb.Image(fig_sisdr),
    "continuous_stoi_plot": wandb.Image(fig_stoi),
})

plt.close('all')

# Save CSV
df_results = pd.DataFrame(results)
df_results.to_csv(CSV_PATH, index=False)
logger.info(f"Saved evaluation CSV to {CSV_PATH}")

wandb.finish()
logger.info("Evaluation complete!")
