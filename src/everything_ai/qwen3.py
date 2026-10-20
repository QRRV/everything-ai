# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Qwen3-0.6B-Base, an open model of Alibaba's Qwen team (Apache-2.0), written out so that nothing beyond PyTorch and
tiktoken is needed. The course does not ship it (1.2 GB): everything_ai.hub downloads its files once, at a fixed
commit (FILES). The model has the names of the learner's ModernGPT of llm-modern-gpt (`blocks`, `attention.query`,
`mlp.gate` and so on), so that the learner's own add_lora, dpo_step and the rest work on it unchanged; its file has
the names of Qwen's own code, and a function that renames them loads it (`load`). The numbers of the layers are
those of Qwen3's config.json; data/qwen3-0.6b-base.json holds that config and the names and shapes in the file.
Import this module only where the model is needed: it imports PyTorch."""
import json
import unicodedata

import torch
import torch.nn.functional as F
from torch import nn

from everything_ai import hub, paths

REPO = "Qwen/Qwen3-0.6B-Base"
FILES = ["model.safetensors", "vocab.json", "tokenizer_config.json"]   # the weights, and the tokenizer's two files
HEADER = paths.ROOT / "data" / "qwen3-0.6b-base.json"   # its config, and the name, type and shape of every tensor

# Qwen3's config.json, in the names the course uses
CONFIG = {"vocab": 151936, "d": 1024, "heads": 16, "kv_heads": 8, "head_dim": 128, "hidden": 3072, "layers": 28,
          "rope_base": 1_000_000.0, "eps": 1e-6}

# How Qwen2 and Qwen3 split text into pieces before BPE (tokenizer.json): as GPT-4's pattern, but digits one by one.
PATTERN = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"""
           r"""|\s+(?!\S)|\s+""")


class RMSNorm(nn.Module):
    """As in llm-modern-gpt, computed in float32 also when the model runs in bfloat16, as Qwen's code does."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_half(x, base: float, start: int = 0):
    """RoPE as Qwen lays it out: dimension i of a head turns together with dimension i + d/2, where the rope of
    llm-modern-gpt turns neighbours 2i and 2i + 1 together. x has shape (..., t, d)."""
    t, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = (start + torch.arange(t, device=x.device, dtype=torch.float32))[:, None] * theta[None, :]
    cos, sin = torch.cos(angles).to(x.dtype), torch.sin(angles).to(x.dtype)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def rope_pairs(x, base: float, start: int = 0):
    """RoPE as llm-modern-gpt lays it out, neighbours 2i and 2i + 1 together: the wrong layout for Qwen's weights."""
    t, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = (start + torch.arange(t, device=x.device, dtype=torch.float32))[:, None] * theta[None, :]
    cos, sin = torch.cos(angles).to(x.dtype), torch.sin(angles).to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


class Attention(nn.Module):
    """Grouped-query attention with RoPE, as in llm-modern-gpt, and two things more: every head's queries and keys
    get an RMSNorm of their own before RoPE (q_norm, k_norm), and a head is head_dim wide, so that heads times
    head_dim (2048) is not d (1024)."""

    def __init__(self, c: dict):
        super().__init__()
        self.h, self.kv, self.dh, self.base = c["heads"], c["kv_heads"], c["head_dim"], c["rope_base"]
        self.query = nn.Linear(c["d"], c["heads"] * c["head_dim"], bias=False)
        self.key = nn.Linear(c["d"], c["kv_heads"] * c["head_dim"], bias=False)
        self.value = nn.Linear(c["d"], c["kv_heads"] * c["head_dim"], bias=False)
        self.out = nn.Linear(c["heads"] * c["head_dim"], c["d"], bias=False)
        self.q_norm = RMSNorm(c["head_dim"], c["eps"])
        self.k_norm = RMSNorm(c["head_dim"], c["eps"])
        self.rope = rope_half

    def forward(self, x, cache=None):
        n, t, _ = x.shape
        start = 0 if cache is None or "k" not in cache else cache["k"].shape[2]
        q = self.q_norm(self.query(x).view(n, t, self.h, self.dh)).transpose(1, 2)
        k = self.k_norm(self.key(x).view(n, t, self.kv, self.dh)).transpose(1, 2)
        v = self.value(x).view(n, t, self.kv, self.dh).transpose(1, 2)
        q, k = self.rope(q, self.base, start), self.rope(k, self.base, start)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        k = k.repeat_interleave(self.h // self.kv, dim=1)   # every key and value head serves h / kv query heads
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=start == 0 and t > 1)
        return self.out(out.transpose(1, 2).reshape(n, t, self.h * self.dh))


class MLP(nn.Module):
    """SwiGLU, as in llm-modern-gpt."""

    def __init__(self, c: dict):
        super().__init__()
        self.gate = nn.Linear(c["d"], c["hidden"], bias=False)
        self.up = nn.Linear(c["d"], c["hidden"], bias=False)
        self.down = nn.Linear(c["hidden"], c["d"], bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.norm1 = RMSNorm(c["d"], c["eps"])
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c["d"], c["eps"])
        self.mlp = MLP(c)

    def forward(self, x, cache=None):
        x = x + self.attention(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class Qwen3(nn.Module):
    """The blocks of llm-modern-gpt with Qwen3's numbers, and the token embedding again as the output layer (weight
    tying). Its input is a tensor of token ids, shape (n, t), or (t,) for one text, on any device: they go to the
    model's. Its output: the logits of the next token at every position, shape (n, t, vocab) or (t, vocab). With
    `caches`, one dict per block, it keeps the keys and values, as the KV cache of llm-modern-gpt."""

    def __init__(self, c: dict = CONFIG):
        super().__init__()
        self.tok = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["layers"]))
        self.norm = RMSNorm(c["d"], c["eps"])

    def forward(self, ids, caches=None):
        one = ids.dim() == 1
        ids = (ids.reshape(1, -1) if one else ids).to(self.tok.weight.device)
        x = self.tok(ids)
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        logits = self.norm(x) @ self.tok.weight.T
        return logits[0] if one else logits


def header() -> dict:
    """Qwen3's config.json, and the type and shape of every tensor in its file: {"config": ..., "tensors": {name:
    {"dtype": ..., "shape": [...]}}}, from data/qwen3-0.6b-base.json, so that a lesson can show them offline."""
    return json.loads(HEADER.read_text(encoding="utf-8"))


def downloaded() -> bool:
    return all(hub.pinned(REPO, f) is not None for f in FILES)


def download(progress=None) -> None:
    """Download the model's files once; `progress(done, total)` hears about every megabyte of all of them."""
    total = sum(hub.PINNED[(REPO, f)]["size"] for f in FILES)
    done = 0
    for f in FILES:
        hub.download(REPO, f, None if progress is None else lambda d, t, before=done: progress(before + d, total))
        done += hub.PINNED[(REPO, f)]["size"]


def model_names() -> dict:
    """The name and shape of every tensor of the course's Qwen3, built without its numbers."""
    with torch.device("meta"):
        model = Qwen3()
    return {k: list(v.shape) for k, v in model.state_dict().items()}


class NamesDoNotFit(ValueError):
    """The renamed tensors of the file are not those of the model: `missing` are in the model and not among the new
    names, `unknown` among the new names and not in the model, `shapes` in both with another shape."""

    def __init__(self, missing: list, unknown: list, shapes: list):
        super().__init__(f"{len(missing)} missing, {len(unknown)} unknown, {len(shapes)} of another shape")
        self.missing, self.unknown, self.shapes = missing, unknown, shapes


def check_names(rename) -> None:
    """Raise NamesDoNotFit when `rename` does not turn the names of Qwen3's file into those of the course's model,
    with the same shapes; from the header alone, without the file."""
    ours = model_names()
    theirs = {rename(k): v["shape"] for k, v in header()["tensors"].items()}
    missing = sorted(k for k in ours if k not in theirs)
    unknown = sorted(k for k in theirs if k not in ours)
    shapes = sorted(k for k in theirs if k in ours and list(theirs[k]) != ours[k])
    if missing or unknown or shapes:
        raise NamesDoNotFit(missing, unknown, shapes)


def load(rename, dtype=torch.float32, rope=rope_half) -> Qwen3:
    """Qwen3-0.6B-Base from its downloaded file, with every name of the file turned into the model's by `rename`, in
    `dtype` and in eval mode, nothing trainable. It raises NamesDoNotFit when the names do not fit, and
    FileNotFoundError when the file is not downloaded yet (download() gets it). `rope` is for the lesson only: with
    rope_pairs, the model computes with the wrong layout of RoPE."""
    from everything_ai.models import read_safetensors
    check_names(rename)
    path = hub.pinned(REPO, "model.safetensors")
    if path is None:
        raise FileNotFoundError("Qwen3-0.6B-Base is not downloaded yet: everything_ai.qwen3.download()")
    state = {rename(k): v.to(dtype) for k, v in read_safetensors(path).items()}
    with torch.device("meta"):
        model = Qwen3()
    model.load_state_dict(state, assign=True, strict=True)
    for block in model.blocks:
        block.attention.rope = rope
    model.requires_grad_(False)
    return model.eval()


def tokenizer():
    """Qwen3's tokenizer as a tiktoken Encoding, from the downloaded vocab.json (the bytes of every token, in the
    characters of GPT-2's byte-level alphabet, with its rank) and tokenizer_config.json (the special tokens, such as
    <|im_start|> and <|im_end|>). encode() below normalizes the text first, as Qwen's tokenizer does (NFC)."""
    import tiktoken
    vocab_path, config_path = hub.pinned(REPO, "vocab.json"), hub.pinned(REPO, "tokenizer_config.json")
    if vocab_path is None or config_path is None:
        raise FileNotFoundError("Qwen3's tokenizer is not downloaded yet: everything_ai.qwen3.download()")
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    byte_of, extra = {chr(b): b for b in printable}, 0
    for b in range(256):   # GPT-2's alphabet: the bytes without a printable character get 256, 257, ... in order
        if b not in printable:
            byte_of[chr(256 + extra)] = b
            extra += 1
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    ranks = {bytes(byte_of[ch] for ch in token): rank for token, rank in vocab.items()}
    added = json.loads(config_path.read_text(encoding="utf-8"))["added_tokens_decoder"]
    special = {v["content"]: int(k) for k, v in added.items()}
    return tiktoken.Encoding("qwen3", pat_str=PATTERN, mergeable_ranks=ranks, special_tokens=special)


def encode(enc, text: str) -> list:
    """The token ids of `text`, special tokens such as <|im_end|> included, as Qwen's tokenizer gives them."""
    return enc.encode(unicodedata.normalize("NFC", text), allowed_special="all")
