
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


if __name__ == "__main__":
    chars = get_chars(read_training_set())
    int_for_c = char_to_int('c', chars)
    char_for_int_for_c = int_to_char(int_for_c, chars)

    print(f"The int for c is {int_for_c} and the char for {int_for_c} is {char_for_int_for_c}")