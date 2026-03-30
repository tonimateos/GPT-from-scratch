
def read_training_set():
    with open("input.txt", 'r') as f:
        text = f.read()
    return text

def get_chars(text):
    return sorted(set(text))

def int_to_char(integer, sorted_chars):
    return sorted_chars[integer]

def char_to_int(char, sorted_chars):
    return sorted_chars.index(char)

def encode(text, sorted_chars):
    return [char_to_int(c, sorted_chars) for c in text]
    
def decode(encoded_text, sorted_chars):
    return "".join([int_to_char(i, sorted_chars) for i in encoded_text])


if __name__ == "__main__":
    chars = get_chars(read_training_set())
    int_for_c = char_to_int('c', chars)
    char_for_int_for_c = int_to_char(int_for_c, chars)

    print(f"The int for c is {int_for_c} and the char for {int_for_c} is {char_for_int_for_c}")
    
    encoded_text = encode("hello", chars)
    print(encoded_text)
    decoded_text = decode(encoded_text, chars)
    print(decoded_text)