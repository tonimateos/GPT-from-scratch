# Python AI: Transformer from Scratch

Welcome! This project is designed to help you practice building a Transformer model from the ground up using PyTorch. Your goal is to train a character-level language model on the Tiny Shakespeare dataset to generate text in the style of the Bard.

## 🚀 Roadmap

Follow these steps to build your model. Try to implement each part yourself before looking at tutorials!

Suggestion to use python 3.12:
- 3.14 (Bleeding Edge): It’s so new that heavy libraries (like PyTorch) haven’t built their "pre-compiled binary wheels" for it yet. pip won't find them, and you'd have to compile them from source (which is a headache).
- 3.12 (The Sweet Spot): It’s the "goldilocks" version right now. It has all the modern features (better error messages, faster execution) but it's been out long enough that every library is guaranteed to work out of the box with pip install.

### 1. Environment Setup
- [ ] Create a virtual environment (`python3.12 -m venv venv`).
- [ ] Activate: `source venv/bin/activate`.
- [ ] Deps: `pip install torch numpy matplotlib`.
- [ ] Settle requirements: `pip freeze > requirements.txt`.
- [ ] If you need: `deactivate` to exit the virtual environment.
- [ ] If you had a `requirements.txt`before, just `pip install -r requirements.txt`.



### 2. Data Acquisition & Exploration
- [ ] Download the `input.txt` (Tiny Shakespeare) dataset: `curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- [ ] Read the file and inspect its contents (total characters, unique characters/alphabet).
- [ ] **Challenge:** Print the first 200 characters and the size of your "vocabulary".


### 3. Character-Level Tokenization
- [ ] Create a mapping from characters to integers (`char_to_int`) and vice versa (`int_to_char`).
- [ ] Write an `encode` function (string -> list of ints) and a `decode` function (list of ints -> string).
- [ ] Convert the entire dataset into a `torch.Tensor` of type `long`.

### 4. Data Splitting & Batching
- [ ] Split your data into Training (90%) and Validation (10%) sets.
- [ ] Implement a `get_batch(split)` function that returns a random batch of inputs `x` and targets `y`.
    - `x` should be a sequence of length `block_size`.
    - `y` should be the same sequence shifted by one (the "next token").

### 5. The Baseline: Bigram Model
- [ ] Start simple! Create a `BigramLanguageModel` class that just uses an embedding table to predict the next token based *only* on the current token.
- [ ] Implement the `forward` pass and calculate `cross_entropy` loss.
- [ ] Implement a `generate` function to see what "unintelligent" text looks like.

### 6. Building the Transformer Blocks
Now for the real deal. Implement these components in order:
- [ ] **Self-Attention:** A single head of attention (Query, Key, Value). Remember the "mask" to prevent looking into the future!
- [ ] **Multi-Head Attention:** Multiple Single-Head Attentions running in parallel and concatenated.
- [ ] **Feed-Forward Network (FFN):** A simple linear-ReLU-linear layer.
- [ ] **Transformer Block:** Combine Multi-Head Attention, FFN, and **LayerNorm**. Don't forget the **Residual Connections**!

### 7. The Full Model
- [ ] Implement **Positional Encodings** so the model knows where tokens are in the sequence.
- [ ] Assemble the full `GPT` model with multiple Transformer Blocks.
- [ ] Calculate the final loss against targets.

### 8. The Training Loop
- [ ] Choose an optimizer (AdamW is recommended).
- [ ] Write a loop that:
    1. Gets a batch.
    2. Evaluates the loss.
    3. Performs backpropagation (`loss.backward()`).
    4. Evaluates the validation loss periodically to check for overfitting.

### 9. Generation & Sampling
- [ ] Use your trained model to generate a long sequence of text (e.g., 1000 characters).
- [ ] Experiment with "Temperature" to control the randomness of your Shakespearean gibberish.

---

## 📚 Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Let's build GPT: from scratch, in code, spelled out. (Andrej Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention is All You Need (Paper)](https://arxiv.org/abs/1706.03762)

Good luck! You'll learn more by debugging a single `RuntimeError` than by reading ten tutorials.
