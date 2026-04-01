
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters (note n_embd = num_heads * head_size (which will be 32 too))
batch_size = 32
n_embd = 128
num_heads = 4
block_size = 64
max_iters = 10000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # when estimating the loss in an evaluation step, how many batches to use
dropout = 0.2
n_transformer_layers = 4
# Usually, we want the total size of all heads combined to equal our total embedding size (n_embd).
# If n_embd = 32, a common choice is num_heads = 4 and head_size = 8.
# 4 heads x 8 features = 32 total features.

# 1. In the Bigram Model (What you built first):
#       Yes, C was exactly the vocab_size. Because the model was so simple, we just mapped each character directly to a score for every other character in the vocabulary.
# 2. In a Transformer (What you are building now):
#       No, C is no longer the vocab_size. It is now a hyperparameter called n_embd (which we set to 32 at the top of your file)

class Tokenizer:
    def __init__(self, text):
        self.text = text
        chars = sorted(set(text))
        self.int_to_char = {i: c for i, c in enumerate(chars)}
        self.char_to_int = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_int[c] for c in text]
    
    def decode(self, encoded_text):
        return "".join([self.int_to_char[i] for i in encoded_text])
        
    def debug(self):
        print(self.int_to_char)
        print(self.char_to_int)

    def get_validation_training_tensors(self):
        tensor_text = torch.tensor(self.encode(self.text), dtype=torch.long)
        split_index = int(0.9 * len(tensor_text))
        train_data = tensor_text[:split_index]
        validation_data = tensor_text[split_index:]
        return train_data, validation_data


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # self.token_embedding_table.weight is randomly initialized
        # size is vocab_size x vocab_size
        # the rows are the prob of each token given this provided token (=row number)

    # idx is a tensor of shape B x T
    # logits (or token_embedding_table(idx)), is a tensor of shape B x T x vocab_size
    # targets is a B X T matrix, no extra vocab_size dimension, since it's just the precise next token (0,...1,0,0)
    def forward(self, idx, targets=None):
        # this build a prob for each of the tokens, one row  for each index in the B X T matrix
        logits = self.token_embedding_table(idx)
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample the next character (it returns a select index, not a value)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# In the Transformer class, the dimension C (over feautures or Channels) runs 0,..,n_embd-1 
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    # Head normally receives the embeddings, not the raw integers
    # x is a tensor of shape (B, T, C)
    def forward(self, x):
        B, T, C = x.shape
        # each of the following 3 are linear transformations x(B,T,C), 
        # where the C --> head_size is provided by the nn.Linear layer
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)   # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # k.transpose(-2, -1) is (B, head_size, T)
        wei = q @ k.transpose(-2, -1) # a (B, T, T) matrix showing how much tokens "relate" to each other.
        wei = wei * (self.head_size**-0.5) # normalize the scores (div by sqrt(size)) to get 'probs'
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, n_embd, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        # the following layer mixes the multiple heads into one embedding
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = [h(x) for h in self.heads] # list of n_heads entires of (B, T, head_size) dim each
        out = torch.cat(out, dim=-1) # (B, T, n_heads * head_size)
        out = self.proj(out) # (B, T, n_embd)
        out = self.dropout(out)
        return out

# Expand to 4*n_embd and contact againt
class FFWD(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        head_size = n_embd // num_heads
        self.self_att = MultiHead(n_embd, num_heads, head_size)
        self.ffwd = FFWD(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # 1. Attention + Residual Connection
        # first term is residual connection, second is attention
        # ln1: does not change dimension, just normalizes
        # x is a tensor of shape (B, T, C)
        # self_att.forwards returns (B, T, n_embd), so all good
        x = x + self.self_att(self.ln1(x))
        
        # 2. Feed-Forward + Residual Connection
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (B, T, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # syntax of the unpac operator *: 
        #   nn.Sequential(*[Block1, Block2, Block3]) = nn.Sequential(Block1, Block2, Block3)
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_transformer_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape # idx is (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # torch.arange(3) = [0, 1, 2]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_emd)
        # it uses broadcast to add these 2 tensors (copies pos_emb B times)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
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
            # Crop the context so it's never longer than block_size
            # This ensures we don't look up a position index out of bounds!
            idx_cond = idx[:, -block_size:] # <--- THIS is the key line
            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample the next character (it returns a select index, not a value)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        


# Return "hell", "ello"
def get_batch(data, block_size):
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def read_training_set():
    with open("input.txt", 'r') as f:
        text = f.read()
    return text

@torch.no_grad() # tells torch we don't need to track gradients here (saves memory)
def estimate_loss(model, train_data, validation_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else validation_data
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            x, y = get_batch(data, block_size)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[iteration] = loss
        out[split] = losses.mean()
    model.train()
    return out

def generate_sample(model, tokenizer, max_new_tokens=500):
    model.eval() # set to evaluation mode
    # Start with a single "0" (newline/start) character
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Generate and decode
    generated_chars = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    print("----- GENERATED SAMPLE -----")
    print(tokenizer.decode(generated_chars))
    print("-----------------------------")
    model.train() # set back to training mode


if __name__ == "__main__":
    text = read_training_set()
    tokenizer = Tokenizer(text)
    train_data, validation_data = tokenizer.get_validation_training_tensors()

    model = GPTLanguageModel(tokenizer.vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if (iter == 0) or (iter % eval_interval == 0):
            losses = estimate_loss(model, train_data, validation_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            generate_sample(model, tokenizer, max_new_tokens=500)

        x, y = get_batch(train_data, block_size)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()       

    print("Final:")
    generate_sample(model, tokenizer, max_new_tokens=500)
    