import os
import argparse
import torch
import torch.distributed as dist
from lcm3 import train_lcm, collate_fn, LCM_MCB
from dataset import LCMDistillDataset
from torch.utils.data import DataLoader, DistributedSampler, random_split

def main():
    parser = argparse.ArgumentParser(description="Train LCM with DDP")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--val_csv_path", type=str, default=None, help="CSV path for validation dataset")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--resume", type=int, choices=[0,1], default=0, help="Resume training from latest checkpoint (1=yes, 0=no)")
    args = parser.parse_args()

    # DDP environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Assign GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"[Rank {local_rank}/{world_size}] Using device: {device}")
    torch.cuda.synchronize()

    # Initialize DDP process group
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"[Rank {local_rank}] Distributed process group initialized")

    # Load training dataset
    dataset = LCMDistillDataset(args.csv_path)
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    # Load validation dataset if provided
    val_dataset = None
    if args.val_csv_path is not None:
        val_dataset = LCMDistillDataset(args.val_csv_path)

    # Detect number of codebooks from first sample
    sample_data = torch.load(dataset.df.iloc[0]["filepath"])
    n_codebooks = sample_data["degraded"].shape[1]

    # Initialize model
    model = LCM_MCB(
        vocab_size=1024,
        n_codebooks=n_codebooks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers
    ).to(device)

    # Wrap model with DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        print(f"[Rank {local_rank}] Model wrapped in DDP")

    # Start training
    train_lcm(
        dataset=dataset,
        model=model,
        train_sampler=sampler,
        val_dataset=val_dataset,
        val_batch_size=args.val_batch_size,
        device=device,
        per_step_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        max_batch_size=args.max_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        warmup_steps=args.warmup_steps,
        log_level=args.log_level,
        ddp=(world_size > 1),
        resume=args.resume
    )

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()
        print(f"[Rank {local_rank}] Distributed process group destroyed")


if __name__ == "__main__":
    main()

