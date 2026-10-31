import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=200):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))
  def forward(self, x):
    return x + self.pe[:, : x.size(1)]

class TransformerDecoder(nn.Module):
  def __init__(self, vocab_size, hidden_dim = 256, n_heads = 8, num_layers = 3):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, hidden_dim)
    self.pos_encoder = PositionalEncoding(hidden_dim)
    decoder_layer = nn.TransformerDecoderLayer(d_model= hidden_dim, nhead= n_heads, batch_first = True)
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    self.fc_out = nn.Linear(hidden_dim, vocab_size) 
  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
    mask = (mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)))
    return mask
  def forward(self,tgt,memory):
    sequence_len = tgt.size(1)
    tgt_mask = self.generate_square_subsequent_mask(sequence_len).to(tgt.device)
    tgt_emb = self.embedding(tgt)
    tgt_emb = self.pos_encoder(tgt_emb)
    output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
    logits = self.fc_out(output)
    return logits