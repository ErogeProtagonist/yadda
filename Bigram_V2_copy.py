import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embed = 512
num_heads = 8
num_blocks = 6
dropout = 0.2
# ------------------------------------------

torch.manual_seed(1337)

# Check if input file exists before opening
import os
# This ensures we find the file even if we run the script from a different folder
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, 'Karpathy_Shakespeare_input.txt')

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    print("Please make sure 'Karpathy_Shakespeare_input.txt' is in the same folder as this script.")
    import sys
    sys.exit(1)

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# vocabulary of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from character to index
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs and targets
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
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

class Head(nn.Module):
    

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        v = self.value(x)     # (B, T, head_size)

        # compute attention
        # scale by head_size (the dimension of the features being dot-producted)
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class FeedForward(nn.Module):
    """ A simple feed forward network. This means its just a linear layer followed by an activation """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):

    # multi-head attention is just a collection of heads in parallel
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """ A Block is a single block of the transformer """
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # We apply LayerNorm before the transformations (Pre-Norm)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) 
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads=num_heads) for _ in range(num_blocks)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensor of integers
        token_embeds = self.token_embedding_table(idx) # (B, T, C)
        pos_embeds = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embeds + pos_embeds # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # run the model
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == '__main__':
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)
    if device == 'cuda':
        try:
            m = torch.compile(m)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed (normal on some Windows setups): {e}")

    # PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # The new way to handle mixed precision in modern PyTorch
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    for iter in range(max_iters):
        
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # sample a batch of data and feed it to the model
        x, y = get_batch('train')
        
        # evaluate the loss with mixed precision
        # Use the unified torch.amp.autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            logits, loss = model(x, y)
        
        # backprop and optimizer step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    # Save the model weights
    save_path = os.path.join(current_dir, 'shakespeare_model_a100.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")