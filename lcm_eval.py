import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
import time
from datetime import datetime
import torch.distributed as dist
import wandb

# ------------------------------
# Env setup
# ------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["FLASH_ATTENTION_DISABLE"] = "1"

# ------------------------------
# Logging
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
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# ------------------------------
# Collate function
# ------------------------------
def collate_fn(batch):
    degraded_list, teacher_list, mask_list = zip(*batch)
    degraded_list = [d.squeeze(0).transpose(0, 1) for d in degraded_list]
    teacher_list = [t.squeeze(0).transpose(0, 1) for t in teacher_list]
    mask_list = [m.squeeze() for m in mask_list]
    degraded_padded = pad_sequence(degraded_list, batch_first=True, padding_value=0)
    teacher_padded = pad_sequence(teacher_list, batch_first=True, padding_value=0)
    mask_padded = pad_sequence(mask_list, batch_first=True, padding_value=0)
    mask_padded = mask_padded.unsqueeze(1).expand(-1, degraded_padded.shape[2], -1)
    return degraded_padded, teacher_padded, mask_padded

# ------------------------------
# LCM Model
# ------------------------------
class LCM_MCB(nn.Module):
    def __init__(self, vocab_size, n_codebooks, d_model=256, n_heads=4, num_layers=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.num_layers = num_layers
        self.n_heads = n_heads

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)]
        )

    def forward(self, x, attention_mask=None, return_attn=False):
        logits_list = []
        all_attn_maps = []

        for i in range(self.n_codebooks):
            tokens = x[..., i]
            x_emb = self.embedding(tokens)

            if attention_mask is not None:
                mask = attention_mask[:, i, :].unsqueeze(-1).to(x_emb.device)
                x_emb = x_emb * mask

            B, S, D = x_emb.shape
            layer_attn_maps = []

            out = x_emb
            for layer in self.layers:
                # Call the self-attention module directly to get weights
                sa = layer.self_attn
                out2, attn_weights = sa(out, out, out, need_weights=True, attn_mask=None)
                out = out2 + out  # residual
                out = layer.linear2(layer.dropout(layer.activation(layer.linear1(out)))) + out  # FFN residual
                if return_attn:
                    layer_attn_maps.append(attn_weights.detach().cpu())  # shape: [B, S, S] per head summed

            if return_attn:
                all_attn_maps.append(torch.stack(layer_attn_maps))  # [num_layers, B, S, S]

            logits_list.append(self.output_heads[i](out))

        logits = torch.stack(logits_list, dim=-1)

        if return_attn:
            return logits, all_attn_maps[-1]  # return last codebook's attention
        return logits

# ------------------------------
# Checkpoints / Utilities
# ------------------------------
def latest_checkpoint(checkpoint_dir):
    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]
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
# Training Loop
# ------------------------------
def train_lcm(
    dataset,
    model,
    sampler,
    device,
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
    dataloader = DataLoader(
        dataset,
        batch_size=per_step_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_step: (float(current_step) / float(max(1, warmup_steps)))
        if current_step < warmup_steps
        else max(
            0.0,
            float(epochs * len(dataloader) - current_step)
            / float(max(1, epochs * len(dataloader) - warmup_steps)),
        ),
    )

    start_epoch = 0

    if resume:
        latest_ckpt = latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            logger.info(f"Resuming training from {latest_ckpt}")
            checkpoint = load_checkpoint_compatible(model, latest_ckpt, device, ddp)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
        else:
            logger.info("No checkpoint found, starting fresh.")

    wandb.init(
        project="lcm-training",
        id="major-voice-12",
        config={
            "epochs": epochs,
            "batch_size": per_step_batch_size,
            "learning_rate": lr,
            "model": "LCM_MCB",
        },
        resume="allow" if resume else None,
    )
    wandb.watch(model, log="all")

    training_start_time = time.time()
    for epoch in range(start_epoch, epochs):
        if ddp:
            sampler.set_epoch(epoch)
        logger.info(
            f"Epoch {epoch+1}/{epochs} starting. Per-step batch size: {per_step_batch_size}"
        )
        total_loss = 0
        per_codebook_losses = [0] * (model.module.n_codebooks if ddp else model.n_codebooks)
        batch_times = []
        restart_epoch = False

        for batch_idx, (degraded, teacher, mask) in enumerate(dataloader):
            batch_start_time = time.time()
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)
            try:
                optimizer.zero_grad()
                logits = model(degraded, attention_mask=mask)
                loss = 0
                n_codebooks = logits.shape[-1]
                for i in range(n_codebooks):
                    ce = F.cross_entropy(
                        logits[..., i].transpose(1, 2),
                        teacher[..., i],
                        reduction="mean",
                    )
                    per_codebook_losses[i] += ce.item()
                    loss += ce
                loss = loss / n_codebooks
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_times.append(time.time() - batch_start_time)
                total_loss += loss.item()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(
                        f"CUDA OOM at epoch {epoch}, batch {batch_idx}. Attempting recovery."
                    )
                    if per_step_batch_size > 2:
                        per_step_batch_size //= 2
                        logger.warning(
                            f"Reducing batch size to {per_step_batch_size} and restarting epoch."
                        )
                        restart_epoch = True
                        torch.cuda.empty_cache()
                        break
                    else:
                        logger.error("OOM with batch size <=2. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                else:
                    raise e

        if restart_epoch:
            continue

        avg_loss = total_loss / len(dataloader)
        avg_per_codebook_losses = [l / len(dataloader) for l in per_codebook_losses]

        is_rank0 = True
        if ddp:
            is_rank0 = dist.is_initialized() and (dist.get_rank() == 0)
        if is_rank0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"lcm_epoch{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict() if ddp else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "per_codebook_losses": avg_per_codebook_losses,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            wandb.save(checkpoint_path)

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "batch_time": sum(batch_times) / len(batch_times) if batch_times else 0,
                **{f"codebook_{i}_loss": avg_per_codebook_losses[i] for i in range(len(avg_per_codebook_losses))},
            }
        )

    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time/3600:.2f} hours. Final loss: {avg_loss:.6f}")

