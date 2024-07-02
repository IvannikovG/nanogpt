import torch
import mmap
import random
import pickle
from variables import batch_size, block_size, max_iters, learning_rate, eval_iters, save_iters, files_to_use, \
    vocab_to_use
from LLM_Classes import GPTLanguageModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Read all files and combine their contents
def read_files(files):
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data.extend(f.readlines())
    return data

# Read and combine all data
all_data = read_files(files_to_use)

# Shuffle the combined data
random.shuffle(all_data)

# Define the split ratio
train_ratio = 0.8
train_size = int(len(all_data) * train_ratio)

# Split the combined data into training and testing sets
train_data = all_data[:train_size]
test_data = all_data[train_size:]

# Save the training and testing data into separate files
os.makedirs('project_data', exist_ok=True)
with open('project_data/train_data.txt', 'w', encoding='utf-8') as train_file:
    train_file.writelines(train_data)

with open('project_data/test_data.txt', 'w', encoding='utf-8') as test_file:
    test_file.writelines(test_data)

print("Data has been split and saved successfully.")

# Define the vocabulary based on the training data
chars = sorted(list(set(''.join(train_data))))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    encoded = [string_to_int.get(c, 777) for c in s]
    assert all(0 <= i < vocab_size for i in encoded), "Found index out of bounds in encoding"
    return torch.tensor(encoded, dtype=torch.long)

def decode(tensor):
    return ''.join([int_to_string[i.item()] for i in tensor])

# Memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "project_data/train_data.txt" if split == 'train' else "project_data/test_data.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            result = torch.tensor(encode(decoded_block), dtype=torch.long)
    return result

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = GPTLanguageModel(vocab_size)


# Debug: Check if the model can handle dummy data
try:
    dummy_input = torch.randint(0, vocab_size, (1, block_size)).to(device)
    dummy_output = model(dummy_input)
    print("Model can handle dummy data.")
except Exception as e:
    print(f"Error with dummy data: {e}")

m = model.to(device)

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, threshold=1e-5)

min_loss = float('inf')
train_losses = []
val_losses = []

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations (x' + str(eval_iters) + ')')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.show()

def save_checkpoint(mdl, optim, itr):
    torch.save({
        'iteration': itr,
        'model_state_dict': mdl.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at iteration {itr}")

def load_checkpoint(mdl, optim):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        mdl.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")
        return iteration
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

must_train = True

if __name__ == "__main__" and must_train:
    print(f'Training loop initialized with files: {files_to_use}')
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')

    start_iter = load_checkpoint(model, optimizer)

    train_losses = []
    val_losses = []
    min_loss = float('inf')

    for iter in tqdm(range(start_iter, max_iters), desc="Training Progress"):
        if iter % (eval_iters * 10) == 0 and iter != 0:
            losses = estimate_loss()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            if min_loss == float('inf'):
                min_loss = losses['train']
            loss_diff = losses['train'] - min_loss
            loss_percentage_change = abs((loss_diff / min_loss) * 100)
            print(f"Current train loss: {losses['train']:.6f}, Loss difference: {loss_diff:.6f}, Percentage change: {loss_percentage_change:.2f}%")

            if losses['train'] < min_loss:
                min_loss = losses['train']

            if loss_percentage_change <= 0.01:
                print(f"Algorithm has converged at step {iter}. Loss percentage change is {loss_percentage_change:.2f}%")

            save_checkpoint(model, optimizer, iter)
            scheduler.step(losses['val'])
            current_lr = get_current_lr(optimizer)
            print(f"Current learning rate: {current_lr:.6f}")

        xb, yb = get_batch('train')
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save final model at the end of training
    save_checkpoint(model, optimizer, max_iters)

    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')
