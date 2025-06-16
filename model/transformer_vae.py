import torch
import torch.nn as nn
import torch.nn.functional as F
import re

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seq_len = max_seq_len

    def forward(self, x):  # x: [batch, seq_len]
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x.permute(1, 0, 2)  # [seq_len, batch, embed]
        enc = self.transformer_encoder(x)  # [seq_len, batch, embed]
        enc = enc.permute(1, 0, 2)  # [batch, seq_len, embed]
        h = enc.mean(dim=1)  # mean pooling over sequence -> [batch, embed]
        return h

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_seq_len = max_seq_len

    # def forward(self, tgt, memory):  # tgt: [batch, seq_len], memory: [batch, mem_seq_len, embed]
    #     batch_size, tgt_seq_len = tgt.size()
    #     positions = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0)
    #     tgt_emb = self.token_embedding(tgt) + self.pos_embedding(positions)
    #     tgt_emb = tgt_emb.permute(1, 0, 2)  # [tgt_seq_len, batch, embed]
    #     mem = memory.permute(1, 0, 2)     # [mem_seq_len, batch, embed]
    #     out = self.transformer_decoder(tgt_emb, mem)
    #     out = out.permute(1, 0, 2)        # [batch, tgt_seq_len, embed]
    #     logits = self.fc_out(out)         # [batch, seq_len, vocab_size]
    #    return logits
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):  # added mask args
        """
        tgt: [batch, seq_len]
        memory: [batch, mem_seq_len, embed]
        tgt_mask: causal mask of shape [seq_len, seq_len]
        tgt_key_padding_mask: mask for padding tokens [batch, seq_len]
        """
        batch_size, tgt_seq_len = tgt.size()
        positions = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0)
        tgt_emb = self.token_embedding(tgt) + self.pos_embedding(positions)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [seq_len, batch, embed]
        mem = memory.permute(1, 0, 2)      # [mem_seq_len, batch, embed]
        out = self.transformer_decoder(
            tgt_emb, mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = out.permute(1, 0, 2)         # [batch, seq_len, embed]
        logits = self.fc_out(out)          # [batch, seq_len, vocab_size]
        return logits

class TransformerVAE(nn.Module):
    def __init__(
        self, vocab_size, embed_size=64, num_layers=2,
        num_heads=4, hidden_dim=128, z_dim=32,
        max_seq_len=100, dropout=0.1, sigma=1.0
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, embed_size, num_layers,
            num_heads, hidden_dim, max_seq_len, dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size, embed_size, num_layers,
            num_heads, hidden_dim, max_seq_len, dropout
        )
        self.fc_mu = nn.Linear(embed_size, z_dim)
        self.fc_logvar = nn.Linear(embed_size, z_dim)
        self.z2mem = nn.Linear(z_dim, embed_size)

    def encode(self, x):
        h = self.encoder(x)              # [batch, embed]
        mu = self.fc_mu(h)               # [batch, z_dim]
        logvar = self.fc_logvar(h)       # [batch, z_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x, tgt_mask=None, tgt_key_padding_mask=None):
        mem = self.z2mem(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.decoder(x, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            
    def forward(self, x, pad_idx):
        mu, logvar = self.encode(x) # mu, logvar: [B, z_dim]
        z = self.reparameterize(mu, logvar) # z: [B, z_dim]

        x_input = x[:, :-1] # [B, T] - decoder input (with SOS)
        x_target = x[:, 1:] # [B, T] - prediction target (ends with EOS)

        S = x_input.size(1)
        causal_mask = make_causal_mask(S, x.device)
        pad_mask = (x_input == pad_idx)

        logits = self.decode(z, x_input, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
        return logits, z, mu, logvar, x_target
    
class LatentExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder  # model.encode(x) -> mu, logvar
        self.model = model

    def extract_latents(self, dataloader, sample=True, device=device):
        """
        Compute latent representations for input batch x.
        Args:
            x (Tensor): [batch, seq_len] input token indices.
            sample (bool): whether to sample from q(z|x) or return (mu, logvar).
        Returns:
            If sample: z, mu, logvar; else: mu, logvar
        """
        all_z, all_mu, all_logvar = [], [], []
        self.encoder.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x[0].to(device)
                mu, logvar = self.model.encode(x)
                if sample:
                    z = self.model.reparameterize(mu, logvar)
                    all_z.append(z)
                all_mu.append(mu)
                all_logvar.append(logvar)

        z_out = torch.cat(all_z) if sample else None
        mu_out = torch.cat(all_mu)
        logvar_out = torch.cat(all_logvar)
        return (z_out, mu_out, logvar_out) if sample else (mu_out, logvar_out)
    
def attention_regularization(S):
    # S: [B, d]
    sparsity_loss = S.mean(dim=0).pow(2).sum()
    orthogonality_loss = ((S @ S.T) - torch.eye(S.size(0), device=S.device)).pow(2).mean()
    return sparsity_loss + orthogonality_loss

# KL divergence
def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# Make causal mask
def make_causal_mask(seq_len, device):
    """Creates an upper triangular causal mask [seq_len, seq_len]"""
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

def extract_latents(model, data_tensor, batch_size=128, device=device, return_z=True):
    """
    Encodes a dataset into latent space using the model encoder.
    Returns the concatenated mu, logvar, and optionally z (stochastic sample) tensors.
    """
    model.eval()
    all_mu, all_logvar, all_z = [], [], []
    dataloader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size)

    with torch.no_grad():
        for x_batch in dataloader:
            x = x_batch.to(device)
            mu, logvar = model.module.encode(x) if hasattr(model, 'module') else model.encode(x)
            z = model.module.reparameterize(mu, logvar) if hasattr(model, 'module') else model.reparameterize(mu, logvar)
            all_mu.append(mu)
            all_logvar.append(logvar)
            all_z.append(z)

    mu_all = torch.cat(all_mu, dim=0)
    logvar_all = torch.cat(all_logvar, dim=0)
    z_all = torch.cat(all_z, dim=0)
    return mu_all, logvar_all, z_all
    
    # if __name__ == '__main__':
       
    #     # Example SELFIES string
    #     selfie = '[C][O][C]'
    #     # Simple tokenization by bracketed tokens
    #     tokens = re.findall(r"\[[^\]]+\]", selfie)
    #     vocab = list(set(tokens + ['[PAD]']))
    #     token2idx = {tok: idx for idx, tok in enumerate(vocab)}
    #     pad_idx = token2idx['[PAD]']

    #     # Convert to tensor [batch=1, seq_len]
    #     input_ids = [token2idx[t] for t in tokens]
    #     input_tensor = torch.tensor(input_ids).unsqueeze(0)

    #     # Initialize model and run forward pass
    #     model = TransformerVAE(
    #         vocab_size=len(vocab), embed_size=64,
    #         num_layers=2, num_heads=4, hidden_dim=128,
    #         z_dim=32, max_seq_len=10
    #     )
    #     logits, mu, logvar = model(input_tensor)

    #     print(f'Logits shape: {logits.shape}')  # [1, seq_len, vocab_size]
    #     print(f'mu shape: {mu.shape}, logvar shape: {logvar.shape}')