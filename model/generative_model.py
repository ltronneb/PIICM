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
import cairosvg
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
        
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim))
        
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
        output, hidden = self.gru(torch.cat([embedded, latent], 2), hidden)
        output = self.fc(output)
        return output, hidden

def compute_elbo(logits, batch, mus, log_vars, KLD_alpha):

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(logits, batch)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
    #kld = _kld_gauss(mus, stds)
    
    return recon_loss + KLD_alpha * kld

def get_latents(encoder, input_mols):
    
      encoder.eval()
      encoder_hidden = encoder.forward(input_mols)
      z, mean, logvars = encoder.get_z(encoder_hidden)
      #latents = latents.unsqueeze(1)
      return z, mean, logvars

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

def lr_lambda(epoch):
    factor = 1
    if epoch > 15 and epoch <= 20:    
        return 0.1
    if epoch > 20:
        return 0.01
    else:
        return factor
    
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists("utils/settings.yml"):
        settings = yaml.safe_load(open("utils/settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
    
    drugs = pd.read_csv('data/final_drugs.csv', delimiter=',')
    
    ##### Converting to SELFIES #########
    
    drugs_ = drugs[['smiles','efficacy', 'logp', 'qed', 'sas']]
    smiles_lengths = [len(x) for x in drugs_['smiles']]
    drugs_['smiles_lengths'] = smiles_lengths
    drugs_short = drugs_[drugs_['smiles_lengths'] < 100]
    drugs_long = drugs_[drugs_['smiles_lengths'] >= 100]
    
    ###### Loading pubchem data #########
    
    df = pd.read_csv('data/pubchem_500k.csv')
    
    ## create a small training dataset
    
    index = random.sample(range(0,len(df)),100000)
    pubchem_df = df.loc[index].smiles
    final_df = pd.concat([pubchem_df, drugs_short['smiles']], axis=0)
    final_df = final_df.reset_index()
    
    ##### Filtering out molecules which cannot be parsed ######
    
    invalids = []
    invalid_index = []
    for i in final_df.index:
        try:
             sf.encoder(final_df.loc[i]['smiles'])
        except (SMILESParserError, EncoderError) as err:
            err_msg = "failed to parse input\n\tSMILES: {}".format(final_df.loc[i])
            print(err_msg)
            print(err)
            invalids.append(final_df.loc[i]['smiles'])
            invalid_index.append(i)
            continue;
    
    final_df = final_df.drop(invalid_index)
    final_df.to_csv('data/merged_drugs.csv')
    
    file_name_smiles = 'data/merged_drugs.csv'
    
    print('Representation: SELFIES')
    encoding_list, encoding_alphabet, largest_molecule_len, smiles_list, smiles_alphabet, largest_smiles_len, properties = \
        get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        
    print('--> Creating integer encoding...')
    int_data = multiple_selfies_to_int(encoding_list, largest_molecule_len, encoding_alphabet)
    
    len_max_molec = int_data.shape[1]
    len_alphabet = len(encoding_alphabet)
    
    property_decoding = False
    # properties = torch.Tensor(np.array(smiles_df[['logP','qed','SAS']]))    

    int_data = int_data.to(device)
    #properties = properties.to(device)

    #### Train and test data     
    data_train = int_data
    #properties_train = properties[0:idx_train_val]
        
    # Parameters
    
    vocab_size = len_alphabet  # Replace with the actual size of your vocabulary
    embedding_dim = 12   # 12
    h_dim = 400          # 400
    batch_size = 100     # 100
    latent_size = 50     # 50
    
    KLD_alpha = settings['training']['KLD_alpha']
    
    n_inducing = 200

    rnn_encoder = RNNEncoder(vocab_size, 12, h_dim, latent_size, 2).cuda()
    rnn_decoder = RNNDecoder(vocab_size, 12, h_dim, latent_size, 1).cuda()
    optimizer = optim.Adam(list(rnn_encoder.parameters()) + list(rnn_decoder.parameters()), lr=0.0001)
    
    # if property_decoding is True:
      
    #   gplvm_decoder = PropertyGPLVM(len(data_train), 50, n_inducing).to(device)
      
    #   # Likelihood
   
    #   likelihood_logP = GaussianLikelihood(batch_shape = gplvm_decoder.model_logP.batch_shape).to(device)
    #   #likelihood_qed = GaussianLikelihood(batch_shape = gplvm_decoder.model_sas.batch_shape).to(device)
    #   #likelihood_sas = GaussianLikelihood(batch_shape = gplvm_decoder.model_sas.batch_shape).to(device)
     
    #   #mll_qed = VariationalELBO(likelihood_qed, gplvm_decoder.model_qed, num_data=len(data_train)).to(device)
    #   #mll_sas = VariationalELBO(likelihood_sas, gplvm_decoder.model_sas, num_data=len(data_train)).to(device)
    #   mll_logP = VariationalELBO(likelihood_logP, gplvm_decoder.model_logP, num_data=len(data_train)).to(device)
          
    #   optimizer = torch.optim.Adam([
    #        dict(params=gplvm_decoder.parameters(), lr=0.001),
    #        #dict(params=likelihood_qed.parameters(), lr=0.001),
    #        dict(params=likelihood_logP.parameters(), lr=0.001),
    #        #dict(params=likelihood_sas.parameters(), lr=0.001),
    #        dict(params=rnn_encoder.parameters(), lr=0.0001),
    #        dict(params=rnn_decoder.parameters(), lr=0.0001)
    #    ])
      
    ##### Training loop #####
    
    num_epochs = 5
    num_batches_train = int(len(data_train)/batch_size)
    
    losses = []
    recon_evolution = []
    
    #optimizer = optim.Adam(list(rnn_encoder.parameters()) + list(rnn_decoder.parameters()), lr=0.001)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(num_epochs):
        
        #permute_index = torch.randperm(data_train.size()[0])
        #data_train = data_train[permute_index]
        #properties_train = properties_train[permute_index]
        
        #print('epoch ' + str(epoch) + ' ' + str(optimizer.param_groups[0]['lr']))
        #scheduler.step()
        
        #before_lr = optimizer.param_groups[0]["lr"]
        #after_lr = optimizer.param_groups[0]["lr"]
        
        for batch_iteration in range(num_batches_train):  # batch iterator
        
            gplvm_loss = torch.Tensor([0.0]).to(device)
    
            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]
            #prop_batch = properties_train[start_idx:stop_idx]

            optimizer.zero_grad()
                    
            # Forward pass through the encoder
            encoder_hidden = rnn_encoder(batch)
            z, means, logvars = rnn_encoder.get_z(encoder_hidden)
            
            # Initialize the output sequence tensor
            output_sequence = torch.zeros((batch_size, batch.size(1), vocab_size), device=device)
            
            z = z.unsqueeze(1)
            encoder_hidden = encoder_hidden[1].reshape(1, batch.size(0), h_dim)
            #decoder_hidden = encoder_hidden
            init_decoder_hidden = rnn_encoder.phi_z(means).reshape(1,batch.size(0), h_dim)
            #decoder_hidden = torch.zeros(rnn_decoder.num_layers, batch_size, hidden_size, device=device)
            
            # Initialize the input for the first time step of the decoder
            decoder_input = torch.tensor([[0]]*batch_size).to(device) # Start-of-sequence token
            
            # Loop through each time step of the target sequence
            #decoder_hidden = encoder_hidden[-1,:,:].unsqueeze(0)
            decoder_hidden = init_decoder_hidden
            
            for t in range(batch.size(1)):
                
                # Forward pass through the decoder at each time step
                decoder_output, decoder_hidden = rnn_decoder(decoder_input, z, decoder_hidden)
                #decoder_output, hidden = decoder(latent, hidden)
        
                # Get the predicted indices for the current time step
                _, predicted_indices = torch.max(decoder_output, 2)
                predicted_index = predicted_indices[:, -1]
        
                ## Update the input for the next time step with the predicted input
                decoder_input = predicted_index.unsqueeze(1)
        
                # Update the output sequence tensor
                output_sequence[:, t, :] = decoder_output.squeeze()
                
            ## property decoding part
            
            # if property_decoding is True:
                            
            #     ### Getting the output of the 3 groups of GPs
                                
            #     output_logP = gplvm_decoder.model_logP(means)
            #     #output_qed = gplvm_decoder.model_qed(means[:,0:3])
            #     #output_sas = gplvm_decoder.model_sas(means[:,0:3])
    
            #     ### Adding together the ELBO losses 
                
            #     gplvm_loss += -mll_logP(output_logP, prop_batch.T[0]).sum()
            #     #gplvm_loss += -mll_qed(output_qed, prop_batch.T[1]).sum()
            #     #gplvm_loss += -mll_sas(output_sas, prop_batch.T[2]).sum()
                
            #     #gplvm_decoder.inducing_inputs.grad = gplvm_decoder.inducing_inputs.grad.to(device)
        
            # Calculate the loss
            
            elbo_loss = compute_elbo(output_sequence.view(-1, vocab_size), batch.view(-1), means, logvars, KLD_alpha)
            mse_loss = compute_mse_loss(encoder_hidden, init_decoder_hidden) 
            #mse_loss = 0.0
            
            rnn_loss = elbo_loss + mse_loss + gplvm_loss
            losses.append(rnn_loss.item())
            
            # Backward pass and optimization
            
            rnn_loss.backward()
            optimizer.step() 
            recons = []
            
            if batch_iteration % 50 == 0:
                
                predicted_indices_batch = output_sequence.argmax(dim=2)
                recon = compute_recon_quality(batch, predicted_indices_batch)

                report = 'Epoch: %d,  Batch: %d / %d, RNN Loss: %.4f  MSE Loss: %.4f GPLVM Loss: %.4f' \
                         % (epoch, batch_iteration, num_batches_train,
                            elbo_loss.item(), mse_loss.item(), gplvm_loss.item())
                print(report)
                recons.append(recon)
                avg_recon = np.round(np.mean(recons),4)
                recon_evolution.append(avg_recon)
        
        #scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], ELBO Loss: {elbo_loss.item():.4f}, GPLVM Loss: {gplvm_loss.item(): .4f} , MSE Loss: {mse_loss.item(): .4f}, \
                  Train Recon: {avg_recon} lr: {optimizer.param_groups[0]["lr"]}')
        
        
        #### Block reconstruction for training data ######
        
        train_num = len(data_train)
        j = 0
        
        latent_mean_train = np.empty(shape=(len(int_data), latent_size))
        latent_sd_train = np.empty(shape=(len(int_data), latent_size))
        z_train = np.empty(shape=(len(int_data), latent_size))
        
        blocks = np.int64(len(int_data)/2000) 
        last_gap = len(int_data) - blocks*2000
        
        for i in range(blocks+1):
            
            start_index = j
            
            if i == range(blocks+1)[-1]:
                end_index = start_index + last_gap
            else:
                end_index = (i+1)*2000
            
            z, mean, logvars = get_latents(rnn_encoder, data_train[start_index:end_index])
            
            ## Insert into the latent_mean_train and latent_logvars_train 
            
            latent_mean_train[start_index:end_index] = mean.cpu().detach()
            latent_sd_train[start_index:end_index] = logvars.cpu().detach().exp().sqrt()
            z_train[start_index:end_index] = z.cpu().detach()
            
            ## Advance the j index to extract the latent for next block of data
            
            j = end_index
            
        ####### Attach drugs_short with the latents and save #########
        
        drugs_latent = pd.DataFrame(z_train[-7268::])
        drugs_short = drugs_short['smiles'].reset_index()
        drugs_latent['smiles'] = drugs_short['smiles']
        drugs_latent.insert(0, 'smiles', drugs_latent.pop('smiles'))