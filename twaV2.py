import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from twa import *

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout = 0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x 

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout = 0.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask = None):
        for layer in range(self.layers):
            x = layer(x, mask)
        return x 

