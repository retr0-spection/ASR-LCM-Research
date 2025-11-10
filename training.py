# training.py
import argparse
from lcm2 import train_lcm  # your LCM training function

def main():
    parser = argparse.ArgumentParser(description="Train Latent Consistency Model on Encodec codes")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV containing preprocessed Encodec code filepaths")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer hidden dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="/datasets/onailana/checkpoints1", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    print(f"Starting training with CSV: {args.csv_path}")

    train_lcm(
        csv_path=args.csv_path,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every
    )

    print("Training complete!")

if __name__ == "__main__":
    main()

