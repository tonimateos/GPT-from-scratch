
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
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
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        # the following layer mixes the multiple heads into one embedding
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        
    def forward(self, x):
        out = [h(x) for h in self.heads] # list of n_heads entires of (B, T, head_size) dim each
        out = torch.cat(out, dim=-1) # (B, T, n_heads * head_size)
        out = self.proj(out) # (B, T, n_embd)
        return out

# Expand to 4*n_embd and contact againt
class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)
        



# Return "hell", "ello"
def get_batch(data, block_size):
    first_index = torch.randint(0, len(data)-block_size, (1,))
    return data[first_index:first_index+block_size], data[first_index+1:first_index+block_size+1]
    

def read_training_set():
    with open("input.txt", 'r') as f:
        text = f.read()
    return text

if __name__ == "__main__":
    text = read_training_set()
    tokenizer = Tokenizer(text)
    # train_data, validation_data = tokenizer.get_validation_training_tensors()
    bigram = BigramModel(tokenizer.vocab_size)
    idx = torch.zeros((1, 1), dtype=torch.long)
    print(tokenizer.decode(bigram.generate(idx, max_new_tokens=100)[0].tolist()))
