
import torch

class Tokenizer:
    def __init__(self, text):
        self.text = text
        chars = sorted(set(text))
        self.int_to_char = {i: c for i, c in enumerate(chars)}
        self.char_to_int = {c: i for i, c in enumerate(chars)}

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


class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)
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
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss
        

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
    bigram = BigramModel(3)
    print(bigram.token_embedding_table.weight)
    a = torch.tensor([[1,1,2,2,2],[1,1,2,2,2]])
    table = bigram.token_embedding_table(a)
    print(table)
    print(f"Shape of a: {a.shape}, shape of table: {table.shape}")