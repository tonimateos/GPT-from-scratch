from train import Tokenizer

def test_int_and_char():
    tokenizer = Tokenizer('abc')
    assert tokenizer.int_to_char[2] == 'c'
    assert tokenizer.char_to_int['c'] == 2

def test_endode_decode():
    tokenizer = Tokenizer('I wanted to say hello to you')
    encoded_text = tokenizer.encode("hello")
    decoded_text = tokenizer.decode(encoded_text)
    assert decoded_text == "hello"
    