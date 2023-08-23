import torch
from torch.nn.utils.rnn import pad_sequence

from twa import *
from dataset import *
from model import *

def custom_collate_fn(batch):
    sequences, reversed_sequences = zip(*batch)

    sequences_padded = pad_sequence(sequences, batch_first = True)
    reversed_sequences_padded = pad_sequence(reversed_sequences, batch_first = True)

    return sequences_padded, reversed_sequences_padded

def train(model, dataloader, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for sequences, reversed_sequences in dataloader:
            optimizer.zero_grad()
            
            sequences, reversed_sequences = sequences.to(device), reversed_sequences.to(device)
            
            output = model(sequences)
            loss = criterion(output.view(-1, output.size(-1)), reversed_sequences.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":

    dataset = ReverseSequenceDataset(10000, 10)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, collate_fn = custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model without TWA
    model_standard = ReverseSequenceModel(vocab_size = 11, d_model = 512, num_heads = 8, d_ff = 2048, use_twa = False)
    print("Training standard transformer...")
    train(model_standard, dataloader, epochs = 10, device = device)

    # Model with TWA
    model_twa = ReverseSequenceModel(vocab_size = 11, d_model = 512, num_heads = 8, d_ff = 2048, use_twa = True)
    print("Training transformer with TWA...")
    train(model_twa, dataloader, epochs = 10, device = device)