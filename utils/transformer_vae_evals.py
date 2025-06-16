import torch
from torch.utils.data import DataLoader, TensorDataset
import selfies as sf
from model.transformer_vae import TransformerVAE, kl_divergence, make_causal_mask

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Reconstruction quality ignoring pad tokens
def compute_recon_quality(true_seq, pred_seq, pad_idx):
    mask = true_seq != pad_idx
    if mask.sum().item() == 0:
        return 0.0
    correct = (true_seq == pred_seq) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return (accuracy * 100.0).item()

def compute_validation_recon(model, data_loader, pad_idx):
    model.eval()
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)

            # Split as in training
            x_input = x[:, :-1]       # [B, T]
            x_target = x[:, 1:]       # [B, T]

            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)

            S = x_input.size(1)
            causal_mask = make_causal_mask(S, x.device)
            pad_mask = (x_input == pad_idx)

            logits = model.decode(z, x_input, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
            pred = logits.argmax(dim=2)  # [B, T]

            total_acc += compute_recon_quality(x_target, pred, pad_idx)
            batches += 1
    return total_acc / batches if batches > 0 else 0.0

# Monte Carlo reconstruction accuracy at molecule level
def compute_mc_recon(model, data_tensor, n_enc, n_dec, pad_idx, batch_size):
    """
    Estimate molecule-level exact reconstruction rate via Monte Carlo:
      - encode each molecule n_enc times
      - decode each latent n_dec times
    Uses causal & pad masks like compute_validation_recon.
    Returns: percentage of exact sequence matches (ignoring pads).
    """
    model.eval()
    total_exact = 0
    total_trials = 0

    loader = DataLoader(
        TensorDataset(data_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)           # [B, T]
            # prepare teacher-forcing input/target
            x_input  = x_batch[:, :-1]             # [B, T-1]
            x_target = x_batch[:,  1:]             # [B, T-1]
            B, S = x_input.size()

            # build masks once per batch
            causal_mask = make_causal_mask(S, x_input.device)  # [S, S]
            pad_mask    = (x_input == pad_idx)                 # [B, S]

            for _ in range(n_enc):
                # encode/reparam with full input
                mu, logvar = model.encode(x_input)
                z = model.reparameterize(mu, logvar)

                for _ in range(n_dec):
                    # decode with teacher-forcing masks
                    logits = model.decode(
                        z, 
                        x_input, 
                        tgt_mask=causal_mask,
                        tgt_key_padding_mask=pad_mask
                    )  # [B, S, V]
                    preds = logits.argmax(dim=2)  # [B, S]

                    # check exact match across each sequence (ignore pads)
                    mask = x_target != pad_idx       # [B, S]
                    matched = ((preds == x_target) | ~mask).all(dim=1)
                    total_exact  += matched.sum().item()
                    total_trials += B

    return (total_exact / total_trials * 100.0) if total_trials > 0 else 0.0





