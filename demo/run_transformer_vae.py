
import numpy as np
import selfies as sf
import pandas as pd
import random 
import os
import yaml
import torch
import importlib
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pylab as plt
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs 
from rdkit.Chem import Draw, AllChem
from utils.one_hot_encoding import multiple_selfies_to_int, multiple_selfies_to_hot, multiple_smile_to_hot
from utils.one_hot_encoding import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_int

# import the TransformerVAE class from your existing module
from model.transformer_vae import TransformerVAE, kl_divergence, make_causal_mask
from utils.transformer_vae_evals import compute_recon_quality, compute_mc_recon, compute_validation_recon

import model.transformer_vae as tvae
importlib.reload(tvae)

# Load data
def load_data(file_name_smiles):
    
    print('--> Acquiring data...')
    print('Representation: SELFIES')
    encoding_list, encoding_alphabet, largest_molecule_len, smiles_list, smiles_alphabet, largest_smiles_len, properties = \
        get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
    print('--> Creating integer encoding...')
    int_data = multiple_selfies_to_int(encoding_list, largest_molecule_len, encoding_alphabet)
    print('Finished creating integer encoding.')
    return int_data, properties, encoding_alphabet

# Generate unconditional samples
def generate_unconditional(model, num_samples, max_seq_len, sos_idx, eos_idx, pad_idx, encoding_alphabet, sf):
    """
    Generate unconditional samples from the prior, decode to SELFIES and SMILES.
    Returns a list of tuples: (smiles, selfie)
    """
    model.eval()
    with torch.no_grad():
        z_prior = torch.randn(num_samples, model.fc_mu.out_features).to(next(model.parameters()).device)  # [N, z_dim]
        generated = torch.full((num_samples, 1), sos_idx, dtype=torch.long, device=z_prior.device)  # [N, 1]

        for t in range(1, max_seq_len):
            tgt_mask = make_causal_mask(generated.size(1), device=z_prior.device)
            logits = model.decode(z_prior, generated, tgt_mask=tgt_mask)  # [N, t, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [N, 1]
            generated = torch.cat([generated, next_token], dim=1)  # [N, t+1]

        # Decode to SELFIES and SMILES
        selfies_list, smiles_list = [], []
        for i in range(generated.size(0)):
            tokens = []
            for idx in generated[i].tolist():
                if idx == eos_idx:
                    break
                if idx not in (sos_idx, pad_idx):
                    tokens.append(encoding_alphabet[idx])
            selfie = ''.join(tokens)
            selfies_list.append(selfie)
            try:
                smiles = sf.decoder(selfie)
            except Exception:
                smiles = ''
            smiles_list.append(smiles)

    # Return only unique SMILES (and corresponding SELFIES)
    unique = list(dict.fromkeys(zip(smiles_list, selfies_list)))
    return unique

def save_epoch_logs(log_dir, epoch, train_recon, val_recon, final_log, samples=None):
    
    os.makedirs(log_dir, exist_ok=True)
    # 1. Append average S_gate as a new row in CSV
    #csv_path = os.path.join(log_dir, "S_avg_evolution.csv")
    #with open(csv_path, "a") as f:
    #    np.savetxt(f, S_avg.reshape(1, -1), delimiter=",")

    # 2. Append training and validation recon to log file
    with open(os.path.join(log_dir, "recon_log.txt"), "a") as f:
        f.write(f"Epoch {epoch}, Train Recon: {train_recon:.4f}, Val Recon: {val_recon:.4f}\n")

    # 3. Save final print log
    with open(os.path.join(log_dir, "final_summary.txt"), "a") as f:
        f.write(f"{final_log}\n")
    
    # 4. Save smiles sampled from the prior
    if samples is not None:
        smiles_selfies_path = os.path.join(log_dir, "smiles_selfies_log.csv")
        with open(smiles_selfies_path, "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write("smiles,selfie\n")
            for smiles, selfie in samples:
                f.write(f"{smiles},{selfie}\n")
                
def save_latents_to_disk(z_out, mu_out, logvar_out, save_dir="latents"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    if z_out is not None:
        torch.save(z_out.cpu(), os.path.join(save_dir, "z_out.pt"))
    torch.save(mu_out.cpu(), os.path.join(save_dir, "mu_out.pt"))
    torch.save(logvar_out.cpu(), os.path.join(save_dir, "logvar_out.pt"))

    print(f"Saved latents to {save_dir}/")

if __name__ == '__main__':
    
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists("utils/settings.yml"):
            settings = yaml.safe_load(open("utils/settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        
    file_name_smiles = settings['data']['merged_drugs_file']
    int_data, properties, encoding_alphabet = load_data(file_name_smiles)
    
    ##### Extracting length of largest molecule
    
    len_max_molec = int_data.shape[1]
    len_alphabet = len(encoding_alphabet)
               
    #### Train, validation and test data 
       
    train_valid_test_size = [0.8, 0.1, 0.1]
    idx_train_val = int(len(int_data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(int_data) * train_valid_test_size[1])
    
    data_train = int_data[0:idx_train_val]
    #properties_train = properties[0:idx_train_val]
    
    data_valid = int_data[idx_train_val:idx_val_test]
    #properties_valid = properties[idx_train_val:idx_val_test]
    data_test = int_data[idx_val_test:]
    #properties_test = properties[idx_val_test:]

    # Hyperparameters
    batch_size = 256
    lr = 1e-3
    epochs = 5
    vocab_size = len(encoding_alphabet)
    pad_idx = encoding_alphabet.index('[nop]')
    eos_idx = encoding_alphabet.index('[eos]')
    sos_idx = encoding_alphabet.index('[sos]')

    # Wrap inputs and targets in a TensorDataset
    train_dataset = TensorDataset(data_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_dataset = TensorDataset(data_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(data_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Tracking lists
    train_recon_evolution = []
    valid_recon_evolution = []
    recon_loss_evolution = []
    kl_loss_evolution = []

    # Instantiate model
    model = tvae.TransformerVAE(
    vocab_size=vocab_size,
    embed_size=132,
    num_layers=3,
    num_heads=4,
    hidden_dim=300,
    z_dim=256,
    max_seq_len=len_max_molec
).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training loop with causal and padding masks applied to decoder
    for epoch in range(1, epochs + 1):
    #for epoch in np.arange(97, 130):
        model.train()
        epoch_train_recons = []
        
        for batch_iteration, x in enumerate(train_dataloader, start=1):
            x = x[0].to(device)  # [B, T+1] (with SOS at position 0)
            optimizer.zero_grad()
            logits, z, mu, logvar, x_target = model(x, pad_idx)
            
            B, S = x_target.size()  # B: batch size, S: sequence length
            # logits: [B, S, V] where V = vocab size

            V = logits.size(-1)
            rec_loss = F.cross_entropy(
                logits.reshape(B * S, V),     # [B*S, V]
                x_target.reshape(B * S),      # [B*S]
                ignore_index=pad_idx,
                reduction='mean'
            )

            kl_loss = kl_divergence(mu, logvar)  # scalar (sum over batch)
                
            beta_kl = min(10.0, epoch / 40.0 * 0.2)            
            scaled_kl = beta_kl * kl_loss
            
            loss = rec_loss + scaled_kl 
            #+ scaled_jac + scaled_reg + scaled_card
            loss.backward()
            optimizer.step()

            # Logging every 20 batches     
            if batch_iteration % 100 == 0:
                preds = logits.argmax(dim=2)
                train_acc = compute_recon_quality(x_target, preds, pad_idx)
                print(
                    f"Epoch {epoch}, Batch {batch_iteration}/{len(train_dataloader)} | "
                    f"Rec: {rec_loss.item():.4f}, KL: {scaled_kl:.4f}, "
                    f"Acc: {train_acc:.4f}", flush=True)
                epoch_train_recons.append(train_acc)

        # End of epoch metrics
        avg_train_rec = float(np.mean(epoch_train_recons)) if epoch_train_recons else 0.0
        train_recon_evolution.append(avg_train_rec)

        valid_rec = compute_validation_recon(model, valid_loader, pad_idx)
        valid_recon_evolution.append(valid_rec)
        
        samples=None
        # Custom logging after 50 epochs: unconditional generation + MC reconstruction
        if epoch >= 80:
            print("\nSampled unconditional generations from prior (novel out of 100 samples):")
            samples = generate_unconditional(model, 10000, len_max_molec, sos_idx, eos_idx, pad_idx, encoding_alphabet, sf)
            for i, (smiles, selfie) in enumerate(samples):
                print(f"Sample {i+1}: {smiles}  <-- {selfie}")
                
            #mc_recon = compute_mc_recon(model, data_valid, n_enc=10, n_dec=10, pad_idx=pad_idx, batch_size=256)
            #print(f"Monte Carlo molecule-level exact match: {mc_recon:.2f}%\n")

        scheduler.step(valid_rec)  # Adjust learning rate based on validation reconstruction accuracy

        current_lr = optimizer.param_groups[0]["lr"]
        final_log = f"Epoch [{epoch}/{epochs}] | "
        final_log += f"Avg Train Recon: {avg_train_rec:.4f}, "
        final_log += f"Val Recon: {valid_rec:.4f}, "
        final_log += f"LR: {current_lr:.2e}"
        print(final_log, flush=True)
        
        log_dir = f"logs/transformer_vae_cancer_merged_drugs_128_132_4_6_256_128_60_256_100"
        save_epoch_logs(log_dir, epoch, avg_train_rec, valid_rec, final_log, samples)
        
        ##### Extract latent representations
        
        test_dataset = TensorDataset(data_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        latent_extractor = tvae.LatentExtractor(model)
        z_all, mu_all, logvar_all = latent_extractor.extract_latents(test_dataloader)
        print(z_all.shape, mu_all.shape, logvar_all.shape)
        
        # Save latent representations to disk
        save_latents_to_disk(z_all, mu_all, logvar_all)
        
        save_dir = "latents"
        z_out = torch.load(f"{save_dir}/z_out.pt")
        mu_out = torch.load(f"{save_dir}/mu_out.pt")
        logvar_out = torch.load(f"{save_dir}/logvar_out.pt")


    







