
import torch

class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_int[c] for c in text]
    
    def decode(self, encoded_text):
        return "".join([self.int_to_char[i] for i in encoded_text])
        
    def printme(self):
        print(self.int_to_char)
        print(self.char_to_int)


def read_training_set():
    with open("input.txt", 'r') as f:
        text = f.read()
    return text

if __name__ == "__main__":
    text = read_training_set()
    tokenizer = Tokenizer(text)
    text_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Shape of tensor: {text_tensor.shape}")
    print(f"First 10 chars: {text[:10]}")
    print(f"First 10 encoded chars: {text_tensor[:10]}")
    print(f"First 10 decoded chars: {tokenizer.decode(text_tensor[:10].tolist())}")
    