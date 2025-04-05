# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import numpy as np
import selfies as sf
import pandas as pd
import random 
import os
import yaml
import torch
import torch.optim as optim
import time
from selfies.exceptions import EncoderError, SMILESParserError
from model.generative_model import RNNEncoder, RNNDecoder, compute_elbo, get_latents, reconstruct, \
    decode, lr_at_epoch, kl_weight_at_epoch, compute_mae_loss, compute_recon_quality, compute_validation_recon
from utils.one_hot_encoding import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_int
        
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists("utils/settings.yml"):
        settings = yaml.safe_load(open("utils/settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
    
    drugs = pd.read_csv('data/final_drugs.csv', delimiter=',')
    
    #### Converting to SELFIES ####
    
    drugs_ = drugs[['smiles','efficacy', 'logp', 'qed', 'sas']]
    smiles_lengths = [len(x) for x in drugs_['smiles']]
    drugs_['smiles_lengths'] = smiles_lengths
    drugs_short = drugs_[drugs_['smiles_lengths'] < 120]
    drugs_long = drugs_[drugs_['smiles_lengths'] >= 120]
    
    #### Loading pubchem data ####
    
    df = pd.read_csv('data/pubchem_500k.csv')
    
    ## create a small training dataset
    
    index = random.sample(range(0,len(df)),400000)
    pubchem_df = df.loc[index].smiles
    final_df = pd.concat([pubchem_df, drugs_short['smiles']], axis=0)
    final_df = final_df.reset_index()
    
    #### Filtering out molecules which cannot be parsed ####
    
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
    
    #### Loading the merged drugs data ####
    final_df = pd.read_csv('data/merged_drugs.csv')
    
    print('Representation: SELFIES')
    encoding_list, encoding_alphabet, largest_molecule_len, smiles_list, smiles_alphabet, largest_smiles_len = \
        get_selfie_and_smiles_encodings_for_dataset(final_df)
        
    print('--> Creating integer encoding...')
    int_data = multiple_selfies_to_int(encoding_list, largest_molecule_len, encoding_alphabet)
    
    len_max_molec = int_data.shape[1]
    len_alphabet = len(encoding_alphabet)
    
    int_data = int_data.to(device)

    #### Train and test data  ####
      
    # Create a random permutation of the indices
    permute_index = torch.randperm(int_data.size(0))
    
    # Shuffle the data
    shuffled_data = int_data[permute_index]
    
    # Store the permutation indices for later recovery if needed
    # You can use torch.argsort(permute_index) to recover the original order
           
    train_valid_test_size = [0.8, 0.1, 0.1]
    idx_train_val = int(len(shuffled_data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(shuffled_data) * train_valid_test_size[1])
    
    data_train = shuffled_data[0:idx_train_val]
    data_valid = shuffled_data[idx_train_val:idx_val_test]
    data_test = shuffled_data[idx_val_test:]
        
    #### Parameters  ####
    
    vocab_size = len_alphabet  # Replace with the actual size of your vocabulary
    embedding_dim = 32   # 12
    h_dim = 300       # 400
    batch_size = 512     # 100
    latent_size = 128     # 50
        
    rnn_encoder = RNNEncoder(vocab_size, embedding_dim, h_dim, latent_size, 2).to(device)
    rnn_decoder = RNNDecoder(vocab_size, embedding_dim, h_dim, latent_size, 1).to(device)
          
    #### Training loop  ####
  
    num_epochs = 50
    num_batches_train = int(len(data_train)/batch_size)
    
    losses = []
    train_recon_evolution = []
    valid_recon_evolution = []
    
    recon_loss_evolution = []
    kl_loss_evolution = []
    rnn_loss_evolution = []
    
    for epoch in range(num_epochs):
        
        epoch_start_time = time.time()
        
        lr = lr_at_epoch(epoch)        
        optimizer = optim.Adam(list(rnn_encoder.parameters()) + list(rnn_decoder.parameters()), lr=lr)
                
        # Permute the data
        permute_epoch_index = torch.randperm(data_train.size()[0])
        data_shuffle = data_train[permute_epoch_index]
        
        print('epoch ' + str(epoch) + '  learning rate: ' + str(optimizer.param_groups[0]['lr']))
        
        for batch_iteration in range(num_batches_train):  # batch iterator
            
            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_shuffle[start_idx: stop_idx]
  
            optimizer.zero_grad()
                    
            # Forward pass through the encoder
            encoder_hidden = rnn_encoder(batch)
            z, means, logvars = rnn_encoder.get_z(encoder_hidden)
            
            # Initialize the output sequence tensor
            output_sequence = torch.zeros((batch_size, batch.size(1), vocab_size), device=device)
            
            #z = z.unsqueeze(1)
            encoder_hidden = encoder_hidden[1].reshape(1, batch.size(0), h_dim)
            init_decoder_hidden = rnn_encoder.phi_z(z).reshape(rnn_decoder.num_layers,batch.size(0), h_dim)
            #decoder_hidden = torch.zeros(rnn_decoder.num_layers, batch_size, hidden_size, device=device)
            
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
            
            # Calculate the loss
            
            KLD_alpha = kl_weight_at_epoch(epoch)
            KLD_alpha = 0.0001
            elbo_loss, recon_loss, kld = compute_elbo(output_sequence.view(-1, vocab_size), batch.view(-1), means, logvars, KLD_alpha)
            #mae_loss = compute_mae_loss(encoder_hidden, init_decoder_hidden) 
            
            rnn_loss = elbo_loss #+ mae_loss 
            mae_loss = 0.0
            
            losses.append(rnn_loss.item())
            
            # Backward pass and optimization
            
            rnn_loss.backward()
            
            optimizer.step() 
            train_recons = []

            if batch_iteration % 50 == 0:
                
                predicted_indices_batch = output_sequence.argmax(dim=2)
                train_recon = compute_recon_quality(batch, predicted_indices_batch) 
  
                report = 'Epoch: %d, Batch: %d / %d, RNN Loss: %.4f  MAE Loss: %.4f Recon loss: %.4f KL-Div: %.4f KL alpha: %.4f ' % (epoch, batch_iteration, num_batches_train,
                            elbo_loss.item(), mae_loss, recon_loss.item(), kld.item(), KLD_alpha)
                print(report)
                
                train_recons.append(train_recon)
                
                avg_train_recon = np.round(np.mean(train_recons),4)
        
        ## Compute validation reconstruction once every epoch
        valid_recon = compute_validation_recon(rnn_encoder, rnn_decoder, data_valid, batch_size, h_dim, vocab_size, 1, device)

        train_recon_evolution.append(avg_train_recon) ## every 50 batches
        valid_recon_evolution.append(valid_recon) ## for the whole epoch
        
        recon_loss_evolution.append(recon_loss.cpu().item()) 
        kl_loss_evolution.append(kld.cpu().item()) 
        rnn_loss_evolution.append(rnn_loss.cpu().item())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Recon: {avg_train_recon} Valid Recon: {valid_recon} lr: {optimizer.param_groups[0]["lr"]}')
        
         #### Reconstruction for test data  ####
        
        test_recon = compute_validation_recon(rnn_encoder, rnn_decoder, data_test, batch_size, h_dim, vocab_size, 1, device)
    
        #### Block reconstruction for training/test data  ####
        
        data = int_data
        
        num = len(data)
        j = 0
        
        latent_mean = np.empty(shape=(num, latent_size))
        latent_sd = np.empty(shape=(num, latent_size))
        z_data = np.empty(shape=(num, latent_size))
        hidden_data = np.empty(shape=(num, h_dim))
        
        blocks = np.int64(num/2000) 
        last_gap = num - blocks*2000
        
        for i in range(blocks+1):
            
            print('Block: ', i)
            
            start_index = j
            
            if i == range(blocks+1)[-1]:
                end_index = start_index + last_gap
            else:
                end_index = (i+1)*2000
            
            z_, mean, logvars, hidden = get_latents(rnn_encoder, data[start_index:end_index])
            
            ## Insert into the latent_mean_train and latent_logvars_train 
            
            latent_mean[start_index:end_index] = mean.cpu().detach()
            latent_sd[start_index:end_index] = logvars.cpu().detach().exp().sqrt()
            z_data[start_index:end_index] = z_.cpu().detach()
            hidden_data[start_index:end_index] = hidden.cpu().detach()
            
            ## Advance the j index to extract the latent for next block of data
            
            j = end_index
            
        #### Attach drugs_short with the latents and save  ####
        
        drugs_latent = pd.DataFrame(z_train[-7268::])
        drugs_short = drugs_short['smiles'].reset_index()
        drugs_latent['smiles'] = drugs_short['smiles']
        drugs_latent.insert(0, 'smiles', drugs_latent.pop('smiles'))
        
        
        
        
            ## property decoding part
            
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
            
            
            
            #gplvm_loss = torch.Tensor([0.0]).to(device)

            
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
        
        # Restore original ordering after epoch
