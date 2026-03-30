with open("input.txt", "r") as f:
    text = f.read()

# print(text[:200])


theset = set(text)
print(f"Total chars: {len(text)}")
print("Vocabulary: ", sorted(theset))
print("Vocabulary Length: ", len(theset))

print(theset)
print(list(theset))

# Because a set is unordered, you cannot access its elements by an index (number). If you try theset[0], Python will throw a TypeError.
# To "get at" the elements in a set, you have three main ways:

# 1. Loop through them (Most Common)
# You use a for loop. This is the only way to "visit" every item in the set:

# for char in theset:
#     print(char)
# 2. Check for membership
# If you want to know if a specific character is in your vocabulary, use in:

# python
# if 'q' in theset:
#     print("Yes, 'q' is in the vocabulary!")