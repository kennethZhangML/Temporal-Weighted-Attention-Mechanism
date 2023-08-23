import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
import random

from twaV2 import *

corpus = [
    "This is a sample sentence.",
    "How does the encoder work?",
    "The decoder takes in the encoded sequence.",
    "Transformers are powerful models.",
    "The attention mechanism is crucial.",
]

tokens = [word_tokenize(sentence.lower()) for sentence in corpus]
flat_tokens = [token for sublist in tokens for token in sublist]
vocab = ["<PAD>", "<UNK>"] + [item[0] for item in Counter(flat_tokens).most_common()]

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

class SentenceReverseDataset(Dataset):
    def __init__(self, corpus, token_to_id, seq_length = 10):
        self.corpus = corpus
        self.token_to_id = token_to_id
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        tokens = word_tokenize(sentence.lower())
        sequence = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        
        while len(sequence) < self.seq_length:
            sequence.append(self.token_to_id["<PAD>"])

        sequence = sequence[:self.seq_length]
        
        reversed_sequence = list(reversed(sequence))
        return torch.tensor(sequence), torch.tensor(reversed_sequence)

def one_hot_encode(sequence, vocab_size):
    encoded = torch.zeros(len(sequence), vocab_size).to(sequence.device)
    for i, val in enumerate(sequence):
        encoded[i, val] = 1.0
    return encoded

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = [one_hot_encode(seq, 10) for seq in inputs]   
    targets = [one_hot_encode(seq, 10) for seq in targets]
    return torch.stack(inputs), torch.stack(targets)

dataset = SentenceReverseDataset(corpus, token_to_id)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1

encoder = TransformerEncoder(d_model, len(vocab), num_heads, d_ff, dropout).to(device)
decoder = TransformerDecoder(d_model, len(vocab), num_heads, d_ff, dropout).to(device)

parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr = 0.001)
loss_fn = torch.nn.CrossEntropyLoss()

def train_epoch(dataloader, encoder, decoder, optimizer, loss_fn, device):
    encoder.train()
    decoder.train()

    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        enc_outputs = encoder(inputs)
        dec_outputs = decoder(targets[:, :-1], enc_outputs)
        
        outputs_reshape = dec_outputs.view(-1, len(vocab))
        targets_reshape = targets[:, 1:].contiguous().view(-1)
        
        loss = loss_fn(outputs_reshape, targets_reshape)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

num_epochs = 10
for epoch in range(num_epochs):
    loss = train_epoch(dataloader, encoder, decoder, optimizer, loss_fn, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")