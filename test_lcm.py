# test_lcm.py
import argparse
import torch
from torch.utils.data import DataLoader
from lcm import LCMDistillDataset, LCM_MCB, collate_fn
import torch.nn.functional as F

def evaluate_lcm(csv_path, checkpoint_path, vocab_size=1024, d_model=256,
                 n_heads=4, num_layers=4, batch_size=4, device='cuda'):

    dataset = LCMDistillDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    n_codebooks = torch.load(dataset.df.iloc[0]["filepath"])["degraded"].shape[1]

    # Load model
    model = LCM_MCB(vocab_size, n_codebooks, d_model, n_heads, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for degraded, teacher, mask in dataloader:
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)

            logits = model(degraded, attention_mask=mask)  # [batch, seq_len, vocab, n_codebooks]

            # compute accuracy per codebook
            for i in range(n_codebooks):
                pred = logits[..., i].argmax(dim=2)  # [batch, seq_len]
                matches = (pred == teacher[..., i]) & mask
                correct_tokens += matches.sum().item()
                total_tokens += mask.sum().item()

    accuracy = correct_tokens / total_tokens
    print(f"[EVAL] Token-level accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LCM on Encodec test set")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with test filepaths")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Trained checkpoint path")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate_lcm(
        csv_path=args.csv_path,
        checkpoint_path=args.checkpoint_path,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        device=args.device
    )
