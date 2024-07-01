import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm


# Define the custom dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)


# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_size))
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim), num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_layers(x)
        x = self.fc(x)
        return x


# Load the dataset and create dataloaders
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

seq_length = 128
dataset = ShakespeareDataset(text, seq_length)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model, loss function, and optimizer
vocab_size = dataset.vocab_size
embed_size = 256
num_heads = 8
hidden_dim = 512
num_layers = 6

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {avg_train_loss:.4f} - Validation loss: {avg_val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'transformer_shakespeare.pth')

# Text generation
model.eval()
prompt = "To be or not to be"
generated = [dataset.char2idx[ch] for ch in prompt]
input_seq = torch.tensor(generated).unsqueeze(0).to(device)

for _ in range(100):
    with torch.no_grad():
        output = model(input_seq)
        next_char = torch.argmax(output[0, -1, :]).item()
        generated.append(next_char)
        input_seq = torch.tensor(generated[-seq_length:]).unsqueeze(0).to(device)

generated_text = ''.join([dataset.idx2char[idx] for idx in generated])
print(generated_text)
