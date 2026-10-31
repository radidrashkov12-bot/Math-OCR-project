import torch
import torch.nn as nn

from src.decoder import TransformerDecoder
from src.encoder import CNNEncoder

class MathOCR(nn.Module):
 def __init__(self,vocab_size, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(input_channels = 1, hidden_dim = hidden_dim)
        self.decoder = TransformerDecoder(vocab_size = vocab_size, hidden_dim = hidden_dim)
 def forward(self, images, tgt_tokens):
        memory = self.encoder(images)
        logits = self.decoder(tgt_tokens, memory)
        return logits