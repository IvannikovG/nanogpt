import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32  # parallel samples processing
block_size = 8  # context length in prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#
#  32 of 8-len vectors (arrays)
#  [[] [8] .. 32 ]
#       - o -
#       - o -
#       - o -
#
#

torch.manual_seed(1337)  # reproducible random

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # {a : 0}
itos = {i: ch for i, ch in enumerate(chars)}  # {0 : a}
encode = lambda s: [stoi[c] for c in s]  # 'abs' -> [1 2 3]
decode = lambda n: ''.join([itos[i] for i in n])  # [1 2 3] -> 'ass'
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
validate = data[n:]
n_embd = 32


#
# 1. Tokenization 'i like apples' -> [i like apple s]
# 2. Encoding [i like apple s] -> [8 7864 0323274 33]
# 3. Embedding 'king [1 2 3] - queen [1 2 4]' 'man [0 2 3] - woman [0 2 4]'
# Embedding is done via
# 1. random matrix init -> 2. works just as a layer, and trained too! via backpropagation and loss
#
#  Typical simplified embedding table
#  0 - 8 - i - [2 16 18] <- this is the vector in 3d vector space (usually multidim)
#  0 - 7864 - like - [43 122 1113]
#  0 - 0323274 - apple - [12 16 118]
#  0 - 33 - s - [1 3 2]
#
#  Later when trained, the LLM captures the patterns in speech or text and
#  has a grasp of this in the weights between NN layers
#


def get_batch(split):
    data = train if split == 'train' else validate
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and tagets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample of the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # print(f'Logits ::: {logits}, Probs ::: {probs}, idx_next ::: {idx_next}')
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {it}: train loss {losses}')
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # print(loss)

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
