
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
    train_data, validation_data = tokenizer.get_validation_training_tensors()
    