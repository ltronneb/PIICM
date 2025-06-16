#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative model embeddings for cancer drug data 

@author: vr308
"""

import numpy as np
import matplotlib.pylab as plt
import selfies as sf
import pandas as pd
import random 
import os
import yaml
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from selfies.exceptions import EncoderError, SMILESParserError
from torch import nn
from rdkit import Chem, DataStructs 
from rdkit.Chem import Draw, AllChem
from selfies.exceptions import EncoderError, SMILESParserError
from utils.one_hot_encoding import \
    multiple_selfies_to_hot, multiple_smile_to_hot, get_selfie_and_smiles_encodings_for_dataset, \
        selfies_to_integer_encoded, multiple_selfies_to_int, int_to_selfie_and_smile

# Define the RNN Encoder
class RNNEncoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, h_dim, z_dim, num_layers):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=True)
        self.gru = nn.GRU(embedding_dim, h_dim, num_layers=num_layers, batch_first=True)
        
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim))
        
        # Linear layers for mean and log variance
        self.linear_mean = nn.Linear(h_dim, z_dim)
        self.linear_logvar = nn.Linear(h_dim, z_dim)
    
    def reparameterize(self, mean, logvar):
        # Reparameterization trick: sample from N(0, 1) and scale by the standard deviation, then add the mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        
        embedded = self.embedding(x)
        embedded = F.relu(embedded)
        _, hidden = self.gru(embedded)
       
        return hidden
    
    def get_z(self, hidden):
        hidden_last = hidden[-1,:,:]
        # Map hidden state to mean and log variance
        means = self.linear_mean(hidden_last)
        logvars = self.linear_logvar(hidden_last)
        
        # Reparameterize and get the sample
        z = self.reparameterize(means, logvars)
        return z, means, logvars

# Define the RNN Decoder
class RNNDecoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, h_dim, z_dim, num_layers):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=True)
        #self.embedding = embedding
        #self.embedding.weight.requires_grad = False
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=embedding_dim + z_dim, hidden_size=h_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(h_dim, vocab_size)
        self.h_dim = h_dim
        
    def init_hidden(self, batch_size=1): ## always decoding one token at a time
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size, 
                                self.h_dim)    

    def forward(self, decoder_input, latent, hidden):
        embedded = self.embedding(decoder_input)
        embedded = F.relu(embedded)
        if latent.dim() == 2:
            latent = latent.unsqueeze(1) 
        output, hidden = self.gru(torch.cat([embedded, latent], 2), hidden)
        output = self.fc(output)
        return output, hidden
    
def compute_elbo(logits, batch, mus, log_vars, KLD_alpha):
    
    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(logits, batch)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
    return recon_loss + KLD_alpha * kld, recon_loss, kld

def get_latents(encoder, input_mols):
    
      encoder.eval()
      encoder_hidden = encoder.forward(input_mols)
      z, mean, logvars = encoder.get_z(encoder_hidden)
      #latents = latents.unsqueeze(1)
      return z, mean, logvars, encoder_hidden[1]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_recon_quality(x_indices, x_hat_indices):

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()
    return quality

def compute_mse_loss(encoder_hidden, init_decoder_hidden):
    
    mse = torch.nn.MSELoss()
    mse_loss = mse(encoder_hidden, init_decoder_hidden)
    return mse_loss

def reconstruct(input_mols, encoder, decoder, h_dim, largest_molecule_len, encoding_alphabet):

    decoder.eval()
    
    if encoder is not None:
        encoder.eval()
        encoder_hidden = encoder.forward(input_mols)
        latents, means, logvars = encoder.get_z(encoder_hidden)
        latents = latents.unsqueeze(1)
    else:
        latents = input_mols.to(device)
        
    num_mols = len(input_mols)
    
    all_mols = torch.empty(size=(num_mols,largest_molecule_len), dtype=torch.long)
    decoder_input = torch.tensor([[0]]*num_mols).to(device) # Start-of-sequence token
    #decoder_hidden = encoder_hidden[-1,:,:].unsqueeze(0)
    
    gathered_atoms = []
    #decoder_hidden = torch.zeros(decoder.num_layers, num_mols, hidden.shape[-1], device=device)
    decoder_hidden = encoder.phi_z(means).reshape(1,num_mols, h_dim)

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(largest_molecule_len):
        
            decoder_output, decoder_hidden = decoder(decoder_input, latents, decoder_hidden)

            # Get the predicted indices for the current time step
            _, predicted_indices = torch.max(decoder_output, 2)
            predicted_index = predicted_indices[:, -1]
            
            decoder_input = predicted_index.unsqueeze(1)

            gathered_atoms.append(predicted_index.cpu().tolist())
        
    all_mols = torch.Tensor(gathered_atoms).T
    return all_mols 
    
    selfie_mols = []
    
    for _ in range(num_mols):

        molecule_pre = ''
        
        for i in all_mols[_]:
            molecule_pre += encoding_alphabet[i.int().item()]
        molecule = molecule_pre.replace(' ', '')
        selfie_mols.append(molecule)
    
    one_hot_recons = multiple_selfies_to_hot(selfie_mols, largest_molecule_len, encoding_alphabet)
    
    decoder.train()
    
    if encoder is not None:
        encoder.train()
    
    return all_mols, selfie_mols, one_hot_recons

def decode(decoder, latents, hidden, largest_selfies_len):
   
   decoder.eval()
   num_mols = len(latents)
   all_mols = torch.empty(size=(num_mols,largest_selfies_len), dtype=torch.long)
   decoder_input = torch.tensor([[0]]*num_mols).to(device) # Start-of-sequence token
   
   gathered_atoms = torch.empty(size=(num_mols, largest_selfies_len), dtype=torch.int64)
   
   # runs over letters from molecules (len=size of largest molecule)
   for i in range(largest_selfies_len):
       
           decoder_output, hidden = decoder(decoder_input, latents, hidden)
           # Get the predicted indices for the current time step
           _, predicted_indices = torch.max(decoder_output, 2)
           predicted_index = predicted_indices[:, -1]
           
           decoder_input = predicted_index.unsqueeze(1)
   
           #gathered_atoms.append(predicted_index.cpu().tolist())
           gathered_atoms[:,i] = torch.Tensor(predicted_index)
   
   all_mols = torch.Tensor(gathered_atoms)
   return all_mols

def compute_mae_loss(encoder_hidden, init_decoder_hidden):
    
    mae = torch.nn.L1Loss()
    mae_loss = mae(encoder_hidden, init_decoder_hidden)
    return mae_loss

def compute_validation_recon(rnn_encoder, rnn_decoder, data_valid, batch_size, h_dim, vocab_size, num_layers, device):
    
    rnn_encoder.eval()
    rnn_decoder.eval()
    
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    valid_recons = []
    
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        # Forward pass through the encoder
        encoder_hidden = rnn_encoder(batch)
        z, means, logvars = rnn_encoder.get_z(encoder_hidden)
        
        # Initialize the output sequence tensor
        output_sequence = torch.zeros((batch_size, batch.size(1), vocab_size), device=device)
        
        z = z.unsqueeze(1)
        encoder_hidden = encoder_hidden[1].reshape(1, batch.size(0), h_dim)
        init_decoder_hidden = rnn_encoder.phi_z(z).reshape(num_layers, batch.size(0), h_dim)
        
        # Initialize the input for the first time step of the decoder
        decoder_input = torch.tensor([[0]]*batch_size).to(device) # Start-of-sequence token
        
        # Loop through each time step of the target sequence
        decoder_hidden = init_decoder_hidden

        for t in range(batch.size(1)):       
            
            # Forward pass through the decoder at each time step
            decoder_output, decoder_hidden = rnn_decoder(decoder_input, z, decoder_hidden)
    
            # Get the predicted indices for the current time step
            _, predicted_indices = torch.max(decoder_output, 2)
            predicted_index = predicted_indices[:, -1]
    
            ## Update the input for the next time step with the predicted input
            decoder_input = predicted_index.unsqueeze(1)
    
            # Update the output sequence tensor
            output_sequence[:, t, :] = decoder_output.squeeze()
       
        predicted_indices_batch = output_sequence.argmax(dim=2)
        valid_recon = compute_recon_quality(batch, predicted_indices_batch)      
        valid_recons.append(valid_recon)
       
    rnn_encoder.train()
    rnn_decoder.train()

    return np.mean(valid_recons).item()

def lr_lambda(epoch):
    factor = 1
    if epoch > 15 and epoch <= 20:    
        return 0.1
    if epoch > 20:
        return 0.01
    else:
        return factor
    
def kl_weight_at_epoch(epoch):
    if epoch <= 15:
        kl_weight = 1e-4  # Start with a very small value for the first 5 epochs
    elif epoch <= 50:
        # Linearly increase the KL weight from 1e-5 to 1 between epochs 6 to 30
        kl_weight = 1e-5 + (0.02 - 1e-5) * ((epoch - 5) / (30 - 5))
    else:
        kl_weight = 0.05  # Keep the weight at 1 after epoch 30
    return kl_weight 

def kl_annealing_schedule(num_epochs=40):
    kl_weights = []
    for epoch in range(1, num_epochs + 1):
        kl_weights.append(kl_weight_at_epoch(epoch))
    return kl_weights

def cyclic_kl_weight_at_epoch(epoch):
    if epoch <= 30:
        return 1e-5  # Start with a very small value for the first 5 epochs
    elif epoch <= 40:
       #return 1 / (1 + np.exp(-0.2 * (epoch - 15)))    
       if epoch % 2 == 0:
           return 0.5
       else: 
           return 1e-4
    else:
        return 1  # Keep the weight at 1 after epoch 30
    
def lr_at_epoch(epoch):
    if epoch <= 7:
        return 1e-3  
    elif epoch <= 20:
        return 1e-4
    else:
        return 1e-5  
    
def kl_weight_at_epoch_cyclic(epoch, total_epochs=40, num_cycles=5):
    # Calculate the cycle length
    cycle_length = total_epochs / num_cycles
    # Determine the position within the current cycle
    cycle_position = epoch % cycle_length
    # Normalize cycle position to [0, 1]
    cycle_progress = cycle_position / cycle_length
    # KL weight follows a triangular pattern within each cycle
    kl_weight = 0.5 * (1 - np.cos(np.pi * cycle_progress))  # Ranges from 0 to 1
    return kl_weight

def compute_kl_per_dim(data_train, rnn_encoder, batch_size, device='cuda'):
    
    rnn_encoder.eval()
    kl_sum = None
    num_batches = int(len(data_train)/batch_size)
    
    with torch.no_grad():
        
        for batch_iteration in range(num_batches):
            
            batch = data_train[batch_iteration * batch_size: (batch_iteration + 1) * batch_size]
            
            batch = batch.to(device)
            
            # Forward pass through the encoder
            encoder_hidden = rnn_encoder(batch)
            z, means, logvars = rnn_encoder.get_z(encoder_hidden)

            # Compute KL divergence per dim: shape = [batch_size, z_dim]
            kl = -0.5 * (1 + logvars - means.pow(2) - logvars.exp())

            if kl_sum is None:
                kl_sum = kl.sum(dim=0)
            else:
                kl_sum += kl.sum(dim=0)
            num_batches += batch.size(0)

    kl_per_dim = kl_sum / num_batches  # Average over dataset
    return kl_per_dim.cpu().numpy()

def get_token_position_accuracy(output_sequence, batch):
    # pred_logits: [batch_size, seq_len, vocab_size]
    # batch: [batch_size, seq_len]

    with torch.no_grad():
        pred_tokens = output_sequence.argmax(dim=-1)  # [batch, seq_len]
        correct = (pred_tokens == batch).float()  # [batch, seq_len]
        accuracy_by_pos = correct.mean(dim=0).cpu().numpy()  # [seq_len]
    
    return accuracy_by_pos
