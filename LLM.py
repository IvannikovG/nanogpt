import torch
import mmap
import random
import pickle
from variables import batch_size, block_size, max_iters, learning_rate, eval_iters, save_iters
from LLM_Classes import GPTLanguageModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

chars = ""
with open("project_data/vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "project_data/train_split.txt" if split == 'train' else "project_data/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = GPTLanguageModel(vocab_size)
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully!')
m = model.to(device)


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


# create a PyTorch optimizer and
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

# Initialize variables for tracking the minimum loss and storing loss values
min_loss = float('inf')
train_losses = []
val_losses = []


# Function to plot loss values
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations (x' + str(eval_iters) + ')')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.show()


# Function to save checkpoint
def save_checkpoint(mdl, optim, itr):
    torch.save({
        'iteration': itr,
        'model_state_dict': mdl.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, checkpoint_path)


def load_checkpoint(mdl, optim):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        mdl.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")
        return iteration
    else:
        return 0


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


must_train = False

if __name__ == "__main__" and must_train == True:
    # Set up variables for checkpointing
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')  # .pth extension is standard for PyTorch models

    # Determine the starting iteration
    start_iter = load_checkpoint(model, optimizer)

    # Training loop with checkpointing
    train_losses = []
    val_losses = []
    min_loss = float('inf')

    for iter in tqdm(range(start_iter, max_iters), desc="Training Progress"):
        if iter % (eval_iters * 10) == 0:
            losses = estimate_loss()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            # Calculate the difference and percentage change from the minimal loss value
            if min_loss == float('inf'):
                min_loss = losses['train']
            loss_diff = losses['train'] - min_loss
            loss_percentage_change = abs((loss_diff / min_loss) * 100)
            print(
                f"Current train loss: {losses['train']:.6f}, Loss difference: {loss_diff:.6f}, Percentage change: {loss_percentage_change:.2f}%")

            # Update the minimum loss if the current loss is lower
            if losses['train'] < min_loss:
                min_loss = losses['train']

            if loss_percentage_change <= 0.01:
                print(
                    f"Algorithm has converged at step {iter}. Loss percentage change is {loss_percentage_change:.2f}%")

            # Save checkpoint
            if iter % save_iters == 0:
                save_checkpoint(model, optimizer, iter)

            # Update the scheduler
            scheduler.step(losses['val'])
            current_lr = get_current_lr(optimizer)
            print(f"Current learning rate: {current_lr:.6f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save final model at the end of training
    save_checkpoint(model, optimizer, max_iters)

    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')
