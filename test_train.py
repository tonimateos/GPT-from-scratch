from train import *

def test_int_to_char():
    chars = ['a', 'b', 'c']
    assert int_to_char(2, chars) == 'c'
    assert char_to_int('c', chars) == 2
    