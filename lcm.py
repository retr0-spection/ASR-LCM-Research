import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataset import LCMDistillDataset
import os
from torch.nn.utils.rnn import pad_sequence

# collate function to pad variable-length sequences
def collate_fn(batch):
    degraded_list, teacher_list, mask_list = zip(*batch)

    # Remove batch dimension from each item (currently [1, n_codebooks, seq_len])
    degraded_list = [d.squeeze(0).transpose(0, 1) for d in degraded_list]  # -> [seq_len, n_codebooks]
    teacher_list = [t.squeeze(0).transpose(0, 1) for t in teacher_list]    # -> [seq_len, n_codebooks]
    mask_list    = [m.squeeze(0).transpose(0, 1) for m in mask_list]

    # Pad along sequence dimension
    degraded_padded = pad_sequence(degraded_list, batch_first=True, padding_value=0)  # [B, max_len, n_codebooks]
    teacher_padded  = pad_sequence(teacher_list,  batch_first=True, padding_value=0)
    mask_padded     = pad_sequence(mask_list,     batch_first=True, padding_value=0)

    # Return in consistent shape (B, n_codebooks, max_len)
    return (
        degraded_padded.transpose(1, 2),  # [B, n_codebooks, max_len]
        teacher_padded.transpose(1, 2),
        mask_padded.transpose(1, 2),
    )

class LCM_MCB(nn.Module):
    def __init__(self, vocab_size, n_codebooks, d_model=256, n_heads=4, num_layers=4):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)])

    def forward(self, x, attention_mask=None):
        # x: [batch, seq_len, n_codebooks]
        logits_list = []
        for i in range(self.n_codebooks):
            tokens = x[..., i]  # [batch, seq_len]
            x_emb = self.embedding(tokens)
            if attention_mask is not None:
                x_emb = x_emb * attention_mask.unsqueeze(-1)  # mask padding
            x_trans = self.transformer(x_emb)
            logits_list.append(self.output_heads[i](x_trans))  # [batch, seq_len, vocab_size]
        return torch.stack(logits_list, dim=-1)  # [batch, seq_len, vocab_size, n_codebooks]

def train_lcm(csv_path, vocab_size=1024, d_model=256, n_heads=4, num_layers=4,
              batch_size=8, epochs=10, lr=1e-4, device='cuda', checkpoint_dir='checkpoints',
              save_every=2):

    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = LCMDistillDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    n_codebooks = torch.load(dataset.df.iloc[0]["filepath"])["degraded"].shape[1]
    model = LCM_MCB(vocab_size, n_codebooks, d_model, n_heads, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for degraded, teacher, mask in dataloader:
            degraded, teacher, mask = degraded.to(device), teacher.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(degraded, attention_mask=mask)

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

        # Save checkpoint every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"lcm_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'n_codebooks': n_codebooks
                }
            }, checkpoint_path)

            print(f"[Checkpoint] Saved at {checkpoint_path}")

    # Final save
    final_path = os.path.join(checkpoint_dir, "lcm_final.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': {
            'vocab_size': vocab_size,
            'n_codebooks': n_codebooks
        }
    }, final_path)
    print(f"[Checkpoint] Final model saved at {final_path}")

    return model
