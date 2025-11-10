import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
import json
import time
from datetime import datetime
import torch.distributed as dist
import wandb
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchaudio.transforms import Spectrogram, AmplitudeToDB

# ------------------------------
# ENV + SETTINGS
# ------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["FLASH_ATTENTION_DISABLE"] = "1"

# ------------------------------
# LOGGING
# ------------------------------
def setup_logging(checkpoint_dir, log_level=logging.INFO, ddp=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(checkpoint_dir, f"training_log_{timestamp}.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()] if ddp else [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

# ------------------------------
# COLLATE FN
# ------------------------------
def collate_fn(batch):
    degraded_list, teacher_list, mask_list = zip(*[(b[0], b[1], b[2]) for b in batch])
    degraded_list = [d.squeeze(0).transpose(0, 1) for d in degraded_list]
    teacher_list = [t.squeeze(0).transpose(0, 1) for t in teacher_list]
    mask_list = [m.squeeze() for m in mask_list]
    degraded_padded = pad_sequence(degraded_list, batch_first=True, padding_value=0)
    teacher_padded = pad_sequence(teacher_list, batch_first=True, padding_value=0)
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0)
    mask_padded = mask_padded.unsqueeze(1).expand(-1, degraded_padded.shape[2], -1)
    return degraded_padded, teacher_padded, mask_padded

# ------------------------------
# MODEL
# ------------------------------
class LCM_MCB(nn.Module):
    def __init__(self, vocab_size, n_codebooks, d_model=256, n_heads=8, num_layers=6):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)])

    def forward(self, x, attention_mask=None):
        logits_list = []
        for i in range(self.n_codebooks):
            tokens = x[..., i]
            x_emb = self.embedding(tokens)

            if attention_mask is not None:
                mask = attention_mask[:, i, :].unsqueeze(-1).to(x_emb.device)
                x_emb = x_emb * mask

            x_trans = self.transformer(x_emb)
            logits_list.append(self.output_heads[i](x_trans))
        return torch.stack(logits_list, dim=-1)

# ------------------------------
# LOSSES
# ------------------------------
class AudioLoss(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.db = AmplitudeToDB(stype="power")
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, pred_wav, target_wav):
        # Waveform L1
        l1_loss = F.l1_loss(pred_wav, target_wav)

        # STFT loss
        pred_spec = self.db(self.spec(pred_wav))
        tgt_spec = self.db(self.spec(target_wav))
        stft_loss = F.l1_loss(pred_spec, tgt_spec)

        # Combine
        total = 0.5 * l1_loss + 0.5 * stft_loss
        return total

# ------------------------------
# UTILITIES
# ------------------------------
def latest_checkpoint(checkpoint_dir):
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)

def load_checkpoint_compatible(model, ckpt_path, device, ddp):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module.") and not ddp:
            new_state_dict[k[len("module."):]] = v
        elif not k.startswith("module.") and ddp:
            new_state_dict["module." + k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    return checkpoint

# ------------------------------
# VALIDATION
# ------------------------------
def validate_lcm(dataloader, model, device, loss_fn=None):
    model.eval()
    total_loss = 0
    per_codebook_losses = [0] * 32 // model.n_codebooks
    with torch.no_grad():
        for degraded, teacher, mask in dataloader:
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)
            logits = model(degraded, attention_mask=mask)
            n_codebooks = logits.shape[-1]
            ce_loss = 0
            for i in range(n_codebooks):
                ce = F.cross_entropy(logits[..., i].transpose(1, 2), teacher[..., i], reduction="mean")
                per_codebook_losses[i] += ce.item()
                ce_loss += ce
            loss = ce_loss / n_codebooks
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    avg_per_codebook = [l / len(dataloader) for l in per_codebook_losses]
    model.train()
    return avg_loss, avg_per_codebook

# ------------------------------
# TRAINING
# ------------------------------
def train_lcm(
    dataset,
    model,
    train_sampler,
    val_dataset=None,
    val_batch_size=4,
    device="cuda",
    per_step_batch_size=4,
    effective_batch_size=16,
    max_batch_size=64,
    epochs=20,
    lr=1e-4,
    checkpoint_dir="./checkpoints",
    save_every=1,
    warmup_steps=500,
    log_level=logging.INFO,
    ddp=False,
    resume=False,
):
    logger = setup_logging(checkpoint_dir, log_level, ddp=ddp)
    train_loader = DataLoader(dataset, batch_size=per_step_batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1.0)
    )

    start_epoch = 0
    if resume:
        latest_ckpt = latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            logger.info(f"Resuming from {latest_ckpt}")
            checkpoint = load_checkpoint_compatible(model, latest_ckpt, device, ddp)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
        else:
            logger.info("No checkpoint found, starting from scratch.")

    wandb.init(project="lcm-training",id="major-voice-12", config={"epochs": epochs, "batch_size": per_step_batch_size, "lr": lr})
    wandb.watch(model, log="all")

    loss_fn = AudioLoss()

    for epoch in range(start_epoch, epochs):
        if ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        per_codebook_losses = [0] * (model.module.n_codebooks if ddp else model.n_codebooks)
        batch_times = []
        restart_epoch = False

        for batch_idx, (degraded, teacher, mask) in enumerate(train_loader):
            batch_start = time.time()
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)

            try:
                optimizer.zero_grad()
                logits = model(degraded, attention_mask=mask)
                n_codebooks = logits.shape[-1]
                loss = 0
                for i in range(n_codebooks):
                    ce = F.cross_entropy(logits[..., i].transpose(1, 2), teacher[..., i], reduction="mean")
                    per_codebook_losses[i] += ce.item()
                    loss += ce
                loss /= n_codebooks

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_times.append(time.time() - batch_start)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"OOM at batch {batch_idx}, retrying with smaller batch.")
                    if per_step_batch_size > 2:
                        per_step_batch_size //= 2
                        restart_epoch = True
                        torch.cuda.empty_cache()
                        break
                    else:
                        torch.cuda.empty_cache()
                        continue
                else:
                    raise e

        if restart_epoch:
            continue

        avg_loss = total_loss / len(train_loader)
        avg_per_codebook = [l / len(train_loader) for l in per_codebook_losses]

        val_loss, val_per_codebook = (None, None)
        if val_loader is not None:
            val_loss, val_per_codebook = validate_lcm(val_loader, model, device, loss_fn)

        is_rank0 = True
        if ddp and dist.is_initialized():
            is_rank0 = dist.get_rank() == 0

        if is_rank0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"lcm_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if ddp else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "val_loss": val_loss
            }, ckpt_path)
            logger.info(f"Checkpoint saved at {ckpt_path}")
            wandb.save(ckpt_path)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            **{f"codebook_{i}_train_loss": avg_per_codebook[i] for i in range(len(avg_per_codebook))},
        })

    logger.info("Training complete.")

