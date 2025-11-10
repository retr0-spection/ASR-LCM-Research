import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LCMDistillDataset
from lcm2 import collate_fn
import wandb
import matplotlib.pyplot as plt
import numpy as np
import logging

# ------------------------------
# Config
# ------------------------------
CHECKPOINT_PATH = "/datasets/onailana/checkpoints1/lcm_epoch20.pt"
DATASET_CSV = "/datasets/onailana/test_codes/test_code_data.csv"
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# Attention plotting
# ------------------------------
def plot_attn_heads(attn_map):
    num_heads = attn_map.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(3*num_heads, 3))
    if num_heads == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(attn_map[i], origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"Head {i}")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return wandb.Image(img_array)

# ------------------------------
# Model
# ------------------------------
class LCM_MCB_Attn(nn.Module):
    def __init__(self, vocab_size, n_codebooks, d_model=256, n_heads=8, num_layers=6):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)])

        # Hook attention
        self.attn_maps = []
        for layer in self.transformer.layers:
            layer.self_attn.register_forward_hook(self._save_attn)

    def _save_attn(self, module, input, output):
        # PyTorch >= 2.1 does not store attn weights automatically
        if hasattr(module, 'attn_output_weights'):
            attn = module.attn_output_weights.detach().cpu()
        else:
            # Saved manually during forward
            attn = module._saved_attn.detach().cpu()
        self.attn_maps.append(attn)

    def forward(self, x, attention_mask=None, return_attn=False):
        self.attn_maps = []
        logits_list = []
        for i in range(self.n_codebooks):
            tokens = x[..., i]
            x_emb = self.embedding(tokens)
            if attention_mask is not None:
                mask = attention_mask[:, i, :].unsqueeze(-1).to(x_emb.device)
                x_emb = x_emb * mask
            x_trans = self.transformer(x_emb)
            logits_list.append(self.output_heads[i](x_trans))
        logits = torch.stack(logits_list, dim=-1)
        if return_attn:
            return logits, self.attn_maps
        return logits

# ------------------------------
# Dataset
# ------------------------------
dataset = LCMDistillDataset(DATASET_CSV)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ------------------------------
# Model init and checkpoint load
# ------------------------------
logger.info("Loading LCM model...")
n_codebooks = 32
vocab_size = 1024
d_model = 256
n_heads = 8
num_layers = 6

lcm_model = LCM_MCB_Attn(vocab_size, n_codebooks, d_model, n_heads, num_layers)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
lcm_model.load_state_dict(checkpoint['model_state_dict'])
lcm_model.to(DEVICE).eval()

# ------------------------------
# WandB
# ------------------------------
wandb.init(project="audio-super-resolution", name="attention-heads-only", config={
    "checkpoint": CHECKPOINT_PATH, "batch_size": BATCH_SIZE, "device": DEVICE
})

# ------------------------------
# Evaluation loop
# ------------------------------
with torch.no_grad():
    for batch_idx, (degraded, teacher_codes, mask) in enumerate(dataloader):
        degraded = degraded.to(DEVICE)
        mask = mask.to(DEVICE)
        logits, attn_maps = lcm_model(degraded, attention_mask=mask, return_attn=True)

        # Log first sample only
        for layer_idx, layer_attn in enumerate(attn_maps):
            wandb.log({f"batch_{batch_idx}/layer_{layer_idx}_attn": plot_attn_heads(layer_attn[0])})

