import torch
from torch.utils.data import Dataset, DataLoader

class ReverseSequenceDataset(Dataset):
    def __init__(self, num_samples, max_length):
        self.num_samples = num_samples
        self.max_length = max_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        length = torch.randint(1, self.max_length + 1, (1, )).item()
        sequence = torch.randperm(self.max_length)[: length]
        reversed_sequence = torch.flip(sequence, [0])
        return sequence, reversed_sequence

