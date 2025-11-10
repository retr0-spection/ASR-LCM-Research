import os
import argparse
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train_lcm import train_lcm, collate_fn, LCM_MCB  # your modules
from dataset import LCMDistillDataset

def main():
    parser = argparse.ArgumentParser(description="Train LCM with DDP")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")

    args = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["FLASH_ATTENTION_DISABLE"] = "1"

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{args.local_rank}")
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load dataset
    dataset = LCMDistillDataset(args.csv_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # Determine n_codebooks from first sample
    sample_data = torch.load(dataset.df.iloc[0]["filepath"])
    n_codebooks = sample_data["degraded"].shape[1]

    model = LCM_MCB(
        vocab_size=1024,
        n_codebooks=n_codebooks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers
    ).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Train
    train_lcm(
        dataset=dataset,
        model=model,
        sampler=sampler,
        device=device,
        per_step_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        max_batch_size=args.max_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        warmup_steps=args.warmup_steps,
        log_level=getattr(logging, args.log_level.upper(), logging.INFO),
        ddp=True  # flag inside train_lcm for DDP adjustments
    )

if __name__ == "__main__":
    main()

