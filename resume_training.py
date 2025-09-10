# resume_training.py
import argparse
import os
import torch
from lcm import LCMDistillDataset, LCM_MCB, collate_fn  # import necessary pieces
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

def resume_training(csv_path, checkpoint_dir='checkpoints', vocab_size=1024, d_model=256, n_heads=4,
                    num_layers=4, batch_size=8, epochs=10, lr=1e-4, device='cuda', save_every=2):

    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = LCMDistillDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    n_codebooks = torch.load(dataset.df.iloc[0]["filepath"])["degraded"].shape[1]

    # Find latest checkpoint
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if ckpts:
        latest_ckpt = max(ckpts, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        print(f"[INFO] Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = LCM_MCB(vocab_size, n_codebooks, d_model, n_heads, num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        print("[INFO] No checkpoint found, starting from scratch")
        model = LCM_MCB(vocab_size, n_codebooks, d_model, n_heads, num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        start_epoch = 0

    model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for degraded, teacher, mask in dataloader:
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(degraded, attention_mask=mask)

            # Cross-entropy per codebook
            loss = 0
            for i in range(n_codebooks):
                ce = F.cross_entropy(logits[..., i].transpose(1,2), teacher[..., i], reduction='mean')
                loss += ce
            loss = loss / n_codebooks

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"lcm_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"[Checkpoint] Saved at {checkpoint_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "lcm_final.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, final_path)
    print(f"[Checkpoint] Final model saved at {final_path}")

    return model


# --- CLI interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training LCM from last checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV containing preprocessed Encodec code filepaths")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs to train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=2)
    args = parser.parse_args()

    resume_training(
        csv_path=args.csv_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_every=args.save_every
    )
