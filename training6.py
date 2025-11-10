#!/usr/bin/env python3
import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from lcm_denoised import train, LCM_MCB
from dataset import LCMDistillDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--val_csv_path", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="/datasets/onailana/checkpoints_denoised")
    parser.add_argument("--lambda_waveform", type=float, default=0.5)
    parser.add_argument("--lambda_stft", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="lcm-training")
    parser.add_argument("--log_grads", action="store_true")
    parser.add_argument("--resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()

    # ------------------------------
    # Distributed Setup
    # ------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.distributed.barrier()
        if rank == 0:
            print(f"[DDP] Initialized with {world_size} processes")

    # ------------------------------
    # Dataset Loading
    # ------------------------------
    dataset = LCMDistillDataset(args.csv_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None

    val_dataset = LCMDistillDataset(args.val_csv_path) if args.val_csv_path else None

    # Infer number of codebooks from one sample
    first_path = dataset.df.iloc[0]["filepath"]
    sample_data = torch.load(first_path, map_location="cpu")
    n_codebooks = sample_data["degraded"].shape[1]
    args.n_codebooks = n_codebooks
    if rank == 0:
        print(f"Detected {n_codebooks} codebooks from dataset sample")

    # ------------------------------
    # Model Init
    # ------------------------------
    model = LCM_MCB(
        vocab_size=1024,
        n_codebooks=n_codebooks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers
    ).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if rank == 0:
            print("[DDP] Model wrapped successfully")

    # ------------------------------
    # Start Training
    # ------------------------------
    train(
        dataset=dataset,
        val_dataset=val_dataset,
        model=model,
        train_sampler=sampler,
        args=args,
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_ddp=is_ddp,
    )

    # ------------------------------
    # Cleanup
    # ------------------------------
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("[DDP] Cleanup complete")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

