from train import *

def test_int_and_char():
    tokenizer = Tokenizer('abc')
    assert tokenizer.int_to_char[2] == 'c'
    assert tokenizer.char_to_int['c'] == 2

def test_endode_decode():
    tokenizer = Tokenizer('I wanted to say hello to you')
    encoded_text = tokenizer.encode("hello")
    decoded_text = tokenizer.decode(encoded_text)
    assert decoded_text == "hello"
    
def test_split():
    tokenizer = Tokenizer('0123456789')
    train_data, validation_data = tokenizer.get_validation_training_tensors()
    assert len(train_data) == 9
    assert len(validation_data) == 1
    assert train_data.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert validation_data.tolist() == [9]
    
def test_batching():
    tokenizer = Tokenizer('0123456789')
    train_data, validation_data = tokenizer.get_validation_training_tensors()
    batch_x, batch_y = get_batch(train_data, 3)
    assert batch_x.shape == (3,)
    assert batch_y.shape == (3,)
    assert batch_x[1] == batch_y[0]
    assert batch_x[2] == batch_y[1]
    
def test_bigram():
    bigram = BigramModel(3)
    a = torch.tensor([[1,1,2,2,2],[1,1,2,2,2]])
    table = bigram.token_embedding_table(a)
    assert a.shape == (2, 5)
    assert table.shape == (2, 5, 3)
    
def test_generate():
    tokenizer = Tokenizer('0123456789')
    bigram = BigramModel(tokenizer.vocab_size)
    B, T = 4, 6 
    idx = torch.zeros((B, T), dtype=torch.long)
    generated_idx = bigram.generate(idx, max_new_tokens=5)
    assert generated_idx.shape == (B, T + 5) # e.g. tensor([[0, 1, 5, 2, 0, 9]])

def test_head():
    B, T, C = 4, 8, 32
    head_size = 16
    x = torch.randn(B, T, C)
    head = Head(C, head_size)
    out = head(x)
    
    assert out.shape == (B, T, head_size)
    
    # Test causal property: changing the last token shouldn't affect previous outputs
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    out2 = head(x2)
    
    # Everything except the last time step should be identical
    assert torch.allclose(out[:, :-1, :], out2[:, :-1, :])
    # The last time step should be different
    assert not torch.allclose(out[:, -1, :], out2[:, -1, :])
    
    