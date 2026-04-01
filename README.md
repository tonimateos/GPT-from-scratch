# GPT from Scratch 🎭

Welcome to **GPT from Scratch**, a simple, open-source educational project designed to build and train a character-level Generative Pre-trained Transformer (GPT) from the ground up using PyTorch.

This project follows the architecture described in the "Attention is All You Need" paper and was inspired by Andrej Karpathy's "Let's build GPT" series. It trains on the **Tiny Shakespeare** dataset to generate text that mimics the Bard's style.

---

## 🚀 Features

- **Character-Level Tokenization**: Simple and efficient encoding for text generation.
- **Full Transformer Architecture**:
  - Multi-Head Self-Attention
  - Positional Encodings
  - Feed-Forward Networks
  - Residual Connections & Layer Normalization
- **Scalable Design**: Easily adjust hyperparameters like embedding size, number of layers, and number of heads.
- **Real-Time Monitoring**: Periodic text generation samples during training to visualize the model's learning progress.
- **Robust Testing**: Comprehensive unit tests for every individual component using `pytest`.

---

## 🛠️ Quick Start

### 1. Environment Setup
Clone the repository and set up a virtual environment:

```bash
# Create and activate venv
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model
Run the primary training script to start the optimization process:

```bash
python train.py
```

By default, the model will output the training and validation loss every 200 iterations, followed by a live text generation sample.

### 3. Running Tests
To ensure all architectural components are functioning correctly:

```bash
pytest
```

---

## 📊 Hyperparameters

You can tweak the constants at the top of `train.py` to experiment with model scale:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `n_embd` | 128 | The number of features in each character's vector. |
| `n_transformer_layers` | 4 | The number of blocks to stack. |
| `num_heads` | 4 | The number of attention heads in each block. |
| `block_size` | 64 | The maximum context length the model can "see". |
| `max_iters` | 10000 | Total number of training steps. |

---

## 📚 Resources & Credits

- [Andrej Karpathy's "Let's build GPT" Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

---

## 🗺️ Roadmap
For the original step-by-step development checklist, refer to [roadmap.md](file:///Users/toni/gits/python101/roadmap.md).

Happy coding! 🚀
