import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pesq import pesq
from pystoi import stoi

from lcm import LCM_MCB, collate_fn
from dataset import LCMDistillDataset
from encodec import EncodecModel

# ------------------------------
# Metric helpers
# ------------------------------
def si_sdr(reference, estimation, eps=1e-8):
    reference = reference - torch.mean(reference)
    estimation = estimation - torch.mean(estimation)

    reference_energy = torch.sum(reference ** 2) + eps
    optimal_scaling = torch.sum(reference * estimation) / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection

    ratio = torch.sum(projection ** 2) / (torch.sum(noise ** 2) + eps)
    return 10 * torch.log10(ratio + eps)


# ------------------------------
# EnCodec decoding helpers
# ------------------------------
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.eval()

def tokens_to_wav(tokens):
    """
    tokens: [n_codebooks, seq_len]
    """
    tokens = tokens.unsqueeze(0)  # [1, n_codebooks, seq_len]
    with torch.no_grad():
        wav = encodec_model.decode([(tokens, None)])  # decode expects (codes, scale)
    return wav.squeeze(0).cpu()



# ------------------------------
# Evaluation
# ------------------------------
def evaluate_metrics(model, dataloader, device):
    pesq_scores, stoi_scores, sisdr_scores = [], [], []

    for degraded, teacher, mask in tqdm(dataloader, desc="Evaluating"):
        degraded, teacher = degraded.to(device), teacher.to(device)

        with torch.no_grad():
            pred = model(degraded, mask)

        for i in range(pred.size(0)):
            pred_wav = tokens_to_wav(pred[i].cpu())
            teacher_wav = tokens_to_wav(teacher[i].cpu())

            min_len = min(pred_wav.shape[-1], teacher_wav.shape[-1])
            pred_wav, teacher_wav = pred_wav[..., :min_len], teacher_wav[..., :min_len]

            try:
                pesq_score = pesq(24000, teacher_wav.numpy(), pred_wav.numpy(), "wb")
            except Exception:
                pesq_score = np.nan

            try:
                stoi_score = stoi(teacher_wav.numpy(), pred_wav.numpy(), 24000, extended=False)
            except Exception:
                stoi_score = np.nan

            sisdr_score = si_sdr(torch.tensor(teacher_wav), torch.tensor(pred_wav)).item()

            pesq_scores.append(pesq_score)
            stoi_scores.append(stoi_score)
            sisdr_scores.append(sisdr_score)

    return pesq_scores, stoi_scores, sisdr_scores


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # load dataset + dataloader
    dataset = LCMDistillDataset(args.csv_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # load model
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    vocab_size = checkpoint.get("vocab_size", 1024)
    n_codebooks = checkpoint.get("n_codebooks", 32)

    model = LCM_MCB(vocab_size=vocab_size, n_codebooks=n_codebooks)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    # run evaluation
    pesq_scores, stoi_scores, sisdr_scores = evaluate_metrics(model, dataloader, args.device)

    # save + plot
    df = pd.DataFrame({
        "PESQ": pesq_scores,
        "STOI": stoi_scores,
        "SI-SDR": sisdr_scores,
    })
    df.to_csv("metrics_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.boxplot([df["PESQ"].dropna(), df["STOI"].dropna(), df["SI-SDR"].dropna()],
                labels=["PESQ", "STOI", "SI-SDR"])
    plt.title("Evaluation Metrics Distribution")
    plt.savefig("metrics_boxplot.png")
    print("Saved metrics_results.csv and metrics_boxplot.png")
