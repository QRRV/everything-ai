# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""DINOv2-small, a vision foundation model of Meta (Oquab and others, 2023, arXiv:2304.07193; Apache-2.0), written out
so that nothing beyond PyTorch is needed. It is a Vision Transformer: a photo is cut into squares of 14 by 14 pixels,
each square becomes a token of 384 numbers, a class token is put in front, and 12 transformer blocks let every token
look at every other. What the class token holds at the end is the photo's features. The model learned without any
labels, on 142 million photos (LVD-142M), at 518 by 518 pixels: 37 by 37 squares.

The course does not ship it (88 MB): everything_ai.hub downloads the file of facebook/dinov2-small once, at a fixed
commit. The model has short names of the course's own (`blocks`, `query`, `scale1` and so on); the file has the names
of Hugging Face's code, and `load` renames them. Import this module only where the model is needed: it imports
PyTorch."""
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from everything_ai import hub, models

FILE = ("facebook/dinov2-small", "model.safetensors")
D, HEADS, LAYERS, PATCH = 384, 6, 12, 14
GRID = 37   # the squares per side it learned on: 518 / 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet's numbers, from the model's preprocessor_config.json


class Block(nn.Module):
    """A transformer block as in the GPT of the course, with two differences: every token sees every other (no
    causal mask), and each branch is multiplied by a learned scale per number before it is added (layer scale)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D, eps=1e-6)
        self.query, self.key, self.value = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.out = nn.Linear(D, D)
        self.scale1 = nn.Parameter(torch.ones(D))
        self.norm2 = nn.LayerNorm(D, eps=1e-6)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.scale2 = nn.Parameter(torch.ones(D))

    def attention(self, x):
        n, t, _ = x.shape
        q, k, v = (f(x).view(n, t, HEADS, D // HEADS).transpose(1, 2) for f in (self.query, self.key, self.value))
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // HEADS), dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(n, t, D))

    def forward(self, x):
        x = x + self.scale1 * self.attention(self.norm1(x))
        return x + self.scale2 * self.fc2(F.gelu(self.fc1(self.norm2(x))))


class DINOv2(nn.Module):
    """DINOv2-small: 22 million parameters. `forward` gives the features of a batch of photos, 384 numbers each."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, D, PATCH, PATCH)   # each square of 14 by 14 pixels becomes 384 numbers
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, 1 + GRID * GRID, D))   # a learned position per square, and the class token's
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D, eps=1e-6)

    def places(self, rows: int, cols: int):
        """The position embeddings for rows by cols squares: the 37 by 37 the model learned, made to fit by bicubic
        interpolation, as a picture is resized."""
        if (rows, cols) == (GRID, GRID):
            return self.pos
        grid = self.pos[:, 1:].reshape(1, GRID, GRID, D).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(rows, cols), mode="bicubic", align_corners=False)
        return torch.cat([self.pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, rows * cols, D)], dim=1)

    def tokens(self, x):
        """Every token after the last block: the class token first, then one per square, row after row."""
        squares = self.patch(x)   # (n, 384, rows, cols)
        n, _, rows, cols = squares.shape
        t = torch.cat([self.cls.expand(n, -1, -1), squares.flatten(2).transpose(1, 2)], dim=1) + self.places(rows, cols)
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward(self, x):
        """x: (n, 3, h, w), prepared as `prepare` does, h and w multiples of 14. The class token: (n, 384)."""
        return self.tokens(x)[:, 0]


def _rename(name: str) -> str | None:
    """The course's name for a tensor in the file of facebook/dinov2-small, or None for one the model does not use."""
    if name == "embeddings.mask_token":   # for training, where squares are hidden
        return None
    fixed = {"embeddings.cls_token": "cls", "embeddings.position_embeddings": "pos",
             "embeddings.patch_embeddings.projection.weight": "patch.weight",
             "embeddings.patch_embeddings.projection.bias": "patch.bias",
             "layernorm.weight": "norm.weight", "layernorm.bias": "norm.bias"}
    if name in fixed:
        return fixed[name]
    found = re.fullmatch(r"encoder\.layer\.(\d+)\.(.+)\.(weight|bias|lambda1)", name)
    if found is None:
        raise KeyError(f"an unknown tensor in the file: {name}")
    i, part, kind = found.groups()
    part = {"attention.attention.query": "query", "attention.attention.key": "key", "attention.attention.value": "value",
            "attention.output.dense": "out", "layer_scale1": "scale1", "layer_scale2": "scale2",
            "mlp.fc1": "fc1", "mlp.fc2": "fc2"}.get(part, part)
    return f"blocks.{i}.{part}" + ("" if kind == "lambda1" else f".{kind}")


def downloaded() -> bool:
    return hub.pinned(*FILE) is not None


def load() -> DINOv2:
    """DINOv2-small in eval mode, from the file everything_ai.hub downloaded. It raises FileNotFoundError when that
    file is not there yet: hub.download(*FILE) gets it."""
    path = hub.pinned(*FILE)
    if path is None:
        raise FileNotFoundError("DINOv2-small is not downloaded yet: everything_ai.hub.download(*everything_ai.dinov2.FILE)")
    state = {}
    for name, t in models.read_safetensors(path).items():
        new = _rename(name)
        if new is not None:
            state[new] = t
    model = DINOv2()
    model.load_state_dict(state)
    model.requires_grad_(False)
    return model.eval()


def prepare(photo) -> torch.Tensor:
    """A photo (a path or a PIL image) as the model takes it, as in cv-prepare-photo but resized bicubically, as the
    model's preprocessor_config.json on Hugging Face says: the shorter side to 256, the middle 224 by 224, every color scaled with ImageNet's numbers, and
    a float32 tensor of shape (1, 3, 224, 224). A gray photo gets three equal colors."""
    img = (photo if isinstance(photo, Image.Image) else Image.open(photo)).convert("RGB")
    w, h = img.size
    size = (int(256 * w / h), 256) if w >= h else (256, int(256 * h / w))
    small = np.array(img.resize(size, Image.BICUBIC))
    top, left = int((small.shape[0] - 224) / 2), int((small.shape[1] - 224) / 2)
    crop = small[top:top + 224, left:left + 224]
    x = (crop / 255 - np.array(MEAN)) / np.array(STD)
    return torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0)
