#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smiles to one-hot, one-hot to smiles and selfies to smiles

"""
import numpy as np
import selfies as sf
import pandas as pd
from rdkit import Chem
import torch

def get_smiles_df(file_path):
    
    df = pd.read_csv(file_path, index_col=0)   
    return df

def get_smiles_list(file_path):
    
    df = pd.read_csv(file_path)    
    smiles_list = np.asanyarray(df.smiles)
    return smiles_list

def get_properties_array(file_path):
    
    df = pd.read_csv(file_path)    
    properties_arr = np.asanyarray(df[['logP','qed', 'SAS']])
    return properties_arr

def get_selfie_and_smiles_encodings_for_dataset(smiles_df):
    
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """    
    smiles_list = np.asanyarray(smiles_df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding
    smiles_alphabet.sort()

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    all_selfies_symbols.add('.')
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.sort()

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len

def smile_to_hot(smile, largest_smile_len, alphabet):
    
    """Go from a single smile string to a one-hot encoding.
    """

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """

    hot_list = []
    for s in smiles_list:
        _, onehot_encoded = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def selfies_to_integer_encoded(selfie, largest_selfie_len, alphabet):
    
    """Go from a single selfies string to an integer encoding.
    """
    
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
    
    return integer_encoded

def multiple_selfies_to_int(selfies_list, largest_molecule_len, alphabet):
    
    int_list = []
    for s in selfies_list:
        integer_encoded = selfies_to_integer_encoded(s, largest_molecule_len, alphabet)
        int_list.append(integer_encoded)
    return torch.Tensor(int_list).to(dtype=torch.long)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    int_list = []
    for s in selfies_list:
        integer_encoded, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
        int_list.append(integer_encoded)
    return np.array(hot_list)

def dist_heavy_atoms(smiles_list):
    
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    ha = [m.GetNumHeavyAtoms() for m in mols]
    return ha

def int_to_selfie_and_smile(int_encoded, encoding_alphabet):
    
    smile_mols = []
    selfie_mols = []
    
    for _ in range(len(int_encoded)):

        molecule_pre = ''
        
        for i in int_encoded[_]:
            molecule_pre += encoding_alphabet[i.item()]
        
        molecule = molecule_pre.replace(' ', '')
    
        selfie_mols.append(molecule)            
        smile_mols.append(sf.decoder(molecule))
        
    return smile_mols, selfie_mols
