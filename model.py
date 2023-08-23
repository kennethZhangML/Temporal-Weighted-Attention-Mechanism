import torch 
import torch.nn as nn 

from twa import TransformerBlock

class ReverseSequenceModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, use_twa = False):
        super(ReverseSequenceModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff)
        self.use_twa = use_twa
        
        if self.use_twa:
            self.transformer_block.attn.lambda_ = nn.Parameter(torch.tensor(0.5))
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        return self.fc(x)
