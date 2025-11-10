#!/usr/bin/env python3
"""
training_final.py

DDP-safe LCM training with:
 - token cross entropy (backprop)
 - waveform L1 + multi resolution STFT (perceptual metrics; detached)
 - Encodec decode under no_grad()
 - AMP (autocast + GradScaler)
 - gradient accumulation with DDP no_sync()
 - synchronized OOM handling across ranks (reduces batch size and restarts epoch)
 - latent-space denoising for noise suppression

Fixes applied:
 - safe_stft: force float32, pad short signals, CPU fallback on cuFFT internal errors
 - compute STFT/waveform losses with autocast disabled
 - ensure Encodec decode outputs moved to device and cast to float32
 - set PYTORCH_CUDA_ALLOC_CONF default to reduce fragmentation if unset
"""
import os
import sys
import argparse
import logging
import traceback
from datetime import datetime
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import wandb
from encodec import EncodecModel

# Reduce fragmentation defaults (honors user env if already set)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    # This is a reasonable default to reduce fragmentation (can be tuned).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ---------------------------
# Utils: DDP init & exception hook
# ---------------------------
def install_except_hook(rank):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error(f"Uncaught exception on rank {rank}", exc_info=(exc_type, exc_value, exc_traceback))
        with open(f"traceback_rank{rank}.log", "w") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    sys.excepthook = handle_exception

def ddp_setup_from_env():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    is_ddp = world_size > 1
    if is_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return world_size, rank, local_rank, device, is_ddp

# ---------------------------
# Logging
# ---------------------------
def setup_logging(checkpoint_dir, rank=0, log_level=logging.INFO):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(checkpoint_dir, f"train_{timestamp}_rank{rank}.log")
    handlers = [logging.StreamHandler()] if rank != 0 else [logging.FileHandler(logfile), logging.StreamHandler()]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    logger = logging.getLogger("lcm_train")
    logger.info(f"Logging to {logfile} (rank {rank})")
    return logger

# ---------------------------
# Collate
# ---------------------------
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

# ---------------------------
# Model
# ---------------------------
class LCM_MCB(nn.Module):
    def __init__(self, vocab_size, n_codebooks, d_model=256, n_heads=4, num_layers=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)])

    def forward(self, x, attention_mask=None):
        logits_list = []
        for i in range(self.n_codebooks):
            tokens = x[..., i]               # [B, S]
            x_emb = self.embedding(tokens)   # [B, S, D]
            src_key_padding_mask = (attention_mask[:, i, :] == 0) if attention_mask is not None else None
            x_trans = self.transformer(x_emb, src_key_padding_mask=src_key_padding_mask)
            logits_list.append(self.output_heads[i](x_trans))
        return torch.stack(logits_list, dim=-1)  # [B, S, vocab, n_codebooks]

# ---------------------------
# Losses (safe STFT)
# ---------------------------
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def safe_stft(self, x, n_fft, hop, win, device):
        """
        Run STFT in float32. If cuFFT fails, fallback to CPU.
        Pads signals shorter than n_fft.
        x: [B, L] or [L]
        """
        # ensure contiguous float32
        x = x.contiguous().to(dtype=torch.float32)
        # if last dimension is length, pad if shorter than n_fft
        if x.dim() == 1:
            length = x.size(0)
        else:
            length = x.size(-1)
        if length < n_fft:
            pad_amount = n_fft - length
            # F.pad expects (left, right) for 1D; for batch [B, L] use (0, pad)
            x = F.pad(x, (0, pad_amount))

        # create window on the same device/dtype where possible
        try:
            win_tensor = torch.hann_window(win, device=device, dtype=torch.float32)
        except Exception:
            # if device window creation fails (e.g., device bad), fallback to cpu window
            win_tensor = torch.hann_window(win, device='cpu', dtype=torch.float32).to(device)

        try:
            # Try running STFT on the same device (GPU if available)
            return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=win_tensor, return_complex=True)
        except RuntimeError as e:
            msg = str(e).lower()
            # catch CUFFT internal errors and fallback to CPU implementation
            if "cufft" in msg or "cufft_internal" in msg or "cufft_internal_error" in msg or "cufft_internal_error" in str(e):
                # fallback to cpu; make sure to move window and data to cpu
                cpu_x = x.cpu()
                cpu_win = torch.hann_window(win, device='cpu', dtype=torch.float32)
                X_cpu = torch.stft(cpu_x, n_fft=n_fft, hop_length=hop, win_length=win, window=cpu_win, return_complex=True)
                return X_cpu.to(device)
            else:
                # other runtime errors propagate
                raise

    def forward(self, x, y):
        # canonicalize shapes: expect [B, 1, L] or [B, L]
        x = x.squeeze(1) if x.dim() == 3 else x
        y = y.squeeze(1) if y.dim() == 3 else y
        # move to float32 early (safe_stft also enforces)
        x = x.contiguous().to(dtype=torch.float32)
        y = y.contiguous().to(dtype=torch.float32)

        device = x.device
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # perform stft with safe wrapper (handles padding, dtype, cpu fallback)
            X = self.safe_stft(x, n_fft=n_fft, hop=hop, win=win, device=device)
            Y = self.safe_stft(y, n_fft=n_fft, hop=hop, win=win, device=device)

            mag_x = torch.abs(X)
            mag_y = torch.abs(Y)

            sc_loss = torch.norm(mag_y - mag_x, p='fro') / (torch.norm(mag_y, p='fro') + 1e-8)
            mag_loss = F.l1_loss(torch.log1p(mag_x), torch.log1p(mag_y))
            loss += sc_loss + mag_loss
        return loss / len(self.fft_sizes)

class WaveformL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        # ensure float32
        x = x.contiguous().to(dtype=torch.float32)
        y = y.contiguous().to(dtype=torch.float32)
        return self.l1(x, y)

# ---------------------------
# Latent denoising utility
# ---------------------------
def denoise_latents(logits, temperature=0.7, min_prob=0.05):
    """
    Suppress low-confidence token logits.
    logits: [B, S, vocab, n_codebooks]
    temperature: softmax temperature to sharpen
    min_prob: minimum probability to keep a token
    """
    probs = F.softmax(logits / temperature, dim=2)
    probs[probs < min_prob] = 0.0
    probs = probs / (probs.sum(dim=2, keepdim=True) + 1e-8)
    tokens = probs.argmax(dim=2)
    # reshape to [B, n_codebooks, S]
    return tokens.transpose(1, 2)

# ---------------------------
# Checkpoint utils
# ---------------------------
def latest_checkpoint(checkpoint_dir):
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    return max(ckpts, key=os.path.getmtime) if ckpts else None

def load_checkpoint_compatible(model, ckpt_path, device, ddp):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError("No model weights found in checkpoint")
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module.") and not ddp:
            new_state[k[len("module."):]] = v
        elif not k.startswith("module.") and ddp:
            new_state["module." + k] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    return checkpoint

# ---------------------------
# Validation
# ---------------------------
def validate_one_epoch(val_loader, model, device, encodec_model, n_codebooks, lambda_waveform, lambda_stft, is_ddp):
    model.eval()
    mrstft = MultiResolutionSTFTLoss()
    wf_fn = WaveformL1Loss()
    total = total_ce = total_stft = total_wf = count = 0.0

    with torch.no_grad():
        for degraded, teacher, mask in val_loader:
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)
            logits = model(degraded, attention_mask=mask)

            ce = 0.0
            for i in range(n_codebooks):
                ce += F.cross_entropy(logits[..., i].transpose(1,2), teacher[..., i], reduction='mean')
            ce = ce / n_codebooks

            pred_tokens = denoise_latents(logits)  # <-- denoising applied
            teacher_codes = teacher.transpose(1,2)

            # ensure decode outputs are tensors on device and float32
            pred_wav = encodec_model.decode([(pred_tokens, None)])
            teacher_wav = encodec_model.decode([(teacher_codes, None)])
            if not isinstance(pred_wav, torch.Tensor):
                pred_wav = torch.tensor(pred_wav)
            if not isinstance(teacher_wav, torch.Tensor):
                teacher_wav = torch.tensor(teacher_wav)
            pred_wav = pred_wav.to(device=device, dtype=torch.float32, non_blocking=True)
            teacher_wav = teacher_wav.to(device=device, dtype=torch.float32, non_blocking=True)

            # compute stft/waveform losses in full precision (disable autocast)
            with torch.cuda.amp.autocast(enabled=False):
                stft_l = mrstft(pred_wav, teacher_wav)
                wf_l = wf_fn(pred_wav, teacher_wav)
            loss = ce + lambda_waveform * wf_l + lambda_stft * stft_l

            bsz = degraded.size(0)
            total += loss.item() * bsz
            total_ce += ce.item() * bsz
            total_stft += stft_l.item() * bsz
            total_wf += wf_l.item() * bsz
            count += bsz

    if is_ddp:
        t = torch.tensor([total, total_ce, total_stft, total_wf, count], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, total_ce, total_stft, total_wf, count = [t[i].item() for i in range(5)]

    model.train()
    if count == 0:
        return None
    return {"loss": total/count, "ce": total_ce/count, "stft": total_stft/count, "waveform": total_wf/count}

# ---------------------------
# Training (with synchronized OOM handling + AMP + accumulation)
# ---------------------------
def train(dataset, model, train_sampler, val_dataset, args, device, rank, local_rank, world_size, is_ddp):
    logger = setup_logging(args.checkpoint_dir, rank=rank)
    install_except_hook(rank)

    n_codebooks = args.n_codebooks
    if not hasattr(args, "accum_steps"):
        args.accum_steps = 4

    def make_train_loader():
        return DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)

    train_loader = make_train_loader()
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(val_dataset) if is_ddp and val_dataset is not None else None, collate_fn=collate_fn) if val_dataset is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0))
    warmup_steps = getattr(args, "warmup_steps", 500)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        total_steps = args.epochs * max(1, len(train_loader))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    encodec_model = EncodecModel.encodec_model_24khz().to(device)
    encodec_model.eval()

    stft_fn = MultiResolutionSTFTLoss()
    wf_fn = WaveformL1Loss()

    start_epoch = 0
    if getattr(args, "resume", False):
        ckpt = latest_checkpoint(args.checkpoint_dir)
        if ckpt:
            logger.info(f"Resuming from {ckpt}")
            loaded = load_checkpoint_compatible(model, ckpt, device, ddp=is_ddp)
            if "optimizer_state_dict" in loaded:
                optimizer.load_state_dict(loaded["optimizer_state_dict"])
            if "scheduler_state_dict" in loaded:
                scheduler.load_state_dict(loaded["scheduler_state_dict"])
            start_epoch = loaded.get("epoch", 0)
        else:
            logger.info("No checkpoint found; starting fresh")

    if rank == 0:
        wandb.init(project=getattr(args, "wandb_project", "lcm-distill"), id="Latent-Consistency-Distillation-Denoising", config=vars(args), resume="allow" if getattr(args, "resume", False) else None)
        wandb.watch(model, log="gradients" if getattr(args, "log_grads", False) else None)

    scaler = torch.cuda.amp.GradScaler()
    accum_steps = int(getattr(args, "accum_steps", 4))

    logger.info(f"Starting training: epochs={args.epochs}, per_step_batch_size={args.batch_size}, accum_steps={accum_steps}, ddp={is_ddp}")

    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch) if train_sampler is not None else None
        model.train()
        train_loader = make_train_loader()

        epoch_loss = epoch_ce = epoch_stft = epoch_wf = epoch_count = 0.0
        restart_epoch = False

        optimizer.zero_grad()
        for batch_idx, (degraded, teacher, mask) in enumerate(train_loader):
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)

            use_no_sync = is_ddp and ((batch_idx + 1) % accum_steps != 0)
            ddp_ctx = model.no_sync if use_no_sync else nullcontext

            try:
                with ddp_ctx():
                    # compute model + CE under autocast (fine)
                    with torch.cuda.amp.autocast():
                        logits = model(degraded, attention_mask=mask)
                        ce_loss = 0.0
                        for i in range(n_codebooks):
                            ce_loss += F.cross_entropy(logits[..., i].transpose(1,2), teacher[..., i], reduction='mean')
                        ce_loss = ce_loss / float(n_codebooks)

                        pred_tokens = denoise_latents(logits)  # <-- latent denoising
                        teacher_codes = teacher.transpose(1,2)

                        # Encodec decode done under no_grad (but could be under autocast)
                        with torch.no_grad():
                            pred_wav = encodec_model.decode([(pred_tokens, None)])
                            teacher_wav = encodec_model.decode([(teacher_codes, None)])

                    # Ensure decoded audio tensors are real torch tensors on the correct device/dtype
                    if not isinstance(pred_wav, torch.Tensor):
                        pred_wav = torch.tensor(pred_wav)
                    if not isinstance(teacher_wav, torch.Tensor):
                        teacher_wav = torch.tensor(teacher_wav)
                    pred_wav = pred_wav.to(device=device, dtype=torch.float32, non_blocking=True)
                    teacher_wav = teacher_wav.to(device=device, dtype=torch.float32, non_blocking=True)

                    # compute STFT and waveform losses in full precision (disable autocast)
                    with torch.cuda.amp.autocast(enabled=False):
                        stft_loss = stft_fn(pred_wav, teacher_wav)
                        wf_loss = wf_fn(pred_wav, teacher_wav)

                    # If you still want these perceptual losses detached (as earlier), detach here
                    stft_loss = stft_loss.detach()
                    wf_loss = wf_loss.detach()

                    loss = ce_loss + args.lambda_waveform * wf_loss + args.lambda_stft * stft_loss
                    loss_scaled = loss / accum_steps

                    # backward outside autocast (scale handles it)
                    scaler.scale(loss_scaled).backward()

                is_last_step = (batch_idx + 1 == len(train_loader))
                if ((batch_idx + 1) % accum_steps == 0) or is_last_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                bsz = degraded.size(0)
                epoch_loss += loss.item() * bsz
                epoch_ce += ce_loss.item() * bsz
                epoch_stft += stft_loss.item() * bsz
                epoch_wf += wf_loss.item() * bsz
                epoch_count += bsz

            except RuntimeError as e:
                oom = ("out of memory" in str(e).lower()) or ("cuda out of memory" in str(e).lower())
                if not oom:
                    # re-raise non OOM runtime errors (including unexpected cuFFT errors we've tried to handle)
                    raise

                logging.warning(f"OOM at epoch {epoch} batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                local_oom = torch.tensor([1], device=device, dtype=torch.int32)

                if is_ddp:
                    global_oom = local_oom.clone()
                    dist.all_reduce(global_oom, op=dist.ReduceOp.SUM)
                    any_oom = int(global_oom.item()) > 0
                else:
                    any_oom = True

                if any_oom:
                    if is_ddp:
                        if rank == 0:
                            new_bs = max(1, args.batch_size // 2)
                            decide = torch.tensor([new_bs], device=device, dtype=torch.int32)
                        else:
                            decide = torch.tensor([0], device=device, dtype=torch.int32)
                        dist.broadcast(decide, src=0)
                        new_bs = int(decide.item())
                    else:
                        new_bs = max(1, args.batch_size // 2)

                    if new_bs < args.batch_size:
                        logging.warning(f"Synchronized decision: reducing per_step_batch_size {args.batch_size} -> {new_bs}")
                        args.batch_size = new_bs
                        restart_epoch = True
                        break
                    else:
                        logging.error("OOM at per_step_batch_size==1 on all ranks; cannot recover. Aborting.")
                        raise e

        if restart_epoch:
            # when we reduced args.batch_size we restart this epoch with new loader
            continue

        if is_ddp:
            stats = torch.tensor([epoch_loss, epoch_ce, epoch_stft, epoch_wf, epoch_count], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_ce, epoch_stft, epoch_wf, epoch_count = [stats[i].item() for i in range(5)]

        avg_loss = epoch_loss / epoch_count if epoch_count > 0 else 0.0
        avg_ce = epoch_ce / epoch_count if epoch_count > 0 else 0.0
        avg_stft = epoch_stft / epoch_count if epoch_count > 0 else 0.0
        avg_wf = epoch_wf / epoch_count if epoch_count > 0 else 0.0

        val_metrics = None
        if val_loader is not None:
            val_metrics = validate_one_epoch(val_loader, model, device, encodec_model, n_codebooks=args.n_codebooks, lambda_waveform=args.lambda_waveform, lambda_stft=args.lambda_stft, is_ddp=is_ddp)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}: train loss={avg_loss:.6f} ce={avg_ce:.6f} stft={avg_stft:.6f} wf={avg_wf:.6f} | val: {val_metrics}")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"lcm_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_loss,
                "train_ce": avg_ce,
                "train_stft": avg_stft,
                "train_wf": avg_wf,
                **({f"val_{k}": v for k, v in (val_metrics or {}).items()} )
            })

    logger.info("Training finished")

