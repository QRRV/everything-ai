# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Models the course trains itself and ships in data/, so that a lesson can use a trained model without a download.
data/tools/ holds the scripts that train them; data/README.md says how each was made. Import this module only where
a model is needed: it imports PyTorch."""
import torch
from torch import nn

from everything_ai import paths

DIGIT_CNN = paths.ROOT / "data" / "course-qmnist-cnn.pt"


class DigitCNN(nn.Module):
    """The course's digit reader: two convolutions with max pooling, a hidden layer of 64 neurons, an embedding of
    2 numbers, and 10 logits. Two numbers are few for a digit reader, but they can be drawn: the embedding of every
    image is a point on a map."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.hidden = nn.Linear(16 * 7 * 7, 64)
        self.embed = nn.Linear(64, 2)
        self.out = nn.Linear(2, 10)

    def embedding(self, x):
        """The 2 numbers of each image: x has shape (n, 784), with pixels from 0 to 1, or (n, 1, 28, 28)."""
        x = x.reshape(-1, 1, 28, 28)
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        return self.embed(torch.relu(self.hidden(x.flatten(1))))

    def forward(self, x):
        return self.out(self.embedding(x))


def digit_cnn() -> DigitCNN:
    """The trained digit reader from data/, in eval mode."""
    model = DigitCNN()
    model.load_state_dict(torch.load(DIGIT_CNN, weights_only=True))
    model.eval()
    return model


# --- GPT-2 small, from OpenAI's own weights on Hugging Face ----------------------------------------------------------
# The course does not ship it (548 MB): everything_ai.hub downloads it once, at a fixed commit. The code below is
# GPT-2 as the paper and OpenAI's code describe it, written out so that nothing beyond PyTorch is needed.

GPT2 = ("openai-community/gpt2", "model.safetensors")


class _Dense(nn.Module):
    """A linear layer whose weight is stored as (in, out), as in GPT-2's file: x @ weight + bias."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * 0.02)   # as GPT-2 starts, before training
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return x @ self.weight + self.bias


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.c_attn = _Dense(d, 3 * d)   # queries, keys and values in one
        self.c_proj = _Dense(d, d)

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=2)
        q, k, v = (a.reshape(n, t, self.heads, d // self.heads).transpose(1, 2) for a in (q, k, v))
        scores = q @ k.transpose(2, 3) / (d // self.heads) ** 0.5
        scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.c_proj(out.transpose(1, 2).reshape(n, t, d))


class _MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.c_fc = _Dense(d, 4 * d)
        self.c_proj = _Dense(4 * d, d)

    def forward(self, x):
        return self.c_proj(nn.functional.gelu(self.c_fc(x), approximate="tanh"))


class _Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = _Attention(d, heads)
        self.ln_2 = nn.LayerNorm(d, eps=1e-5)
        self.mlp = _MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    """GPT-2: token and position embeddings, transformer blocks with pre-layer norm, a last layer norm, and the token
    embedding again as the output layer (weight tying). The parameter names are those of GPT-2's file. Its input is a
    tensor of token ids, shape (n, t), or (t,) for one text; its output the logits of the next token at every
    position, shape (n, t, vocab) or (t, vocab)."""

    def __init__(self, vocab: int = 50257, context: int = 1024, d: int = 768, layers: int = 12, heads: int = 12):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(context, d)
        self.h = nn.ModuleList(_Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d, eps=1e-5)

    def forward(self, ids):
        one = ids.dim() == 1
        ids = ids.reshape(1, -1) if one else ids
        x = self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device))
        for block in self.h:
            x = block(x)
        logits = self.ln_f(x) @ self.wte.weight.T
        return logits[0] if one else logits


def read_safetensors(path, skip=lambda name: False) -> dict:
    """The tensors of a .safetensors file, each read straight from the file into its own tensor: 8 bytes with the
    length of a JSON header, the header with each tensor's type, shape and place, then the numbers."""
    import json
    import numpy as np
    types = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__" or skip(name):
                continue
            start, end = info["data_offsets"]
            t = torch.empty(info["shape"], dtype=types[info["dtype"]])
            f.seek(8 + n + start)
            if f.readinto(memoryview(t.view(torch.uint8).numpy() if t.numel() else np.empty(0, np.uint8)).cast("B")) != end - start:
                raise OSError(f"{path} ends in the middle of {name}")
            tensors[name] = t
    return tensors


def gpt2() -> GPT2Model:
    """GPT-2 small, 124 million parameters, in eval mode, from the file everything_ai.hub downloaded. It raises
    FileNotFoundError when that file is not there yet: hub.download(*GPT2) gets it."""
    from everything_ai import hub
    path = hub.pinned(*GPT2)
    if path is None:
        raise FileNotFoundError("GPT-2 is not downloaded yet: everything_ai.hub.download(*everything_ai.models.GPT2)")
    state = read_safetensors(path, skip=lambda name: name.endswith(".attn.bias"))   # the causal mask, as numbers
    with torch.device("meta"):
        model = GPT2Model()
    model.load_state_dict(state, assign=True)
    model.requires_grad_(False)
    return model.eval()
