# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""DETR, a detector of Meta (Carion and others, 2020, "End-to-End Object Detection with Transformers",
arXiv:2005.12872; Apache-2.0), with a ResNet-50 backbone, written out so that nothing beyond PyTorch is needed. It
learned on COCO 2017: 118,000 photos with a box and a kind for every object of 80 kinds.

How it works: ResNet-50 turns a photo into 2,048 numbers for every square of 32 by 32 pixels, a 1 by 1 convolution
makes that 256, and six transformer blocks let every square look at every other (the encoder). Then 100 object
queries, 100 learned vectors of 256 numbers, each ask the squares what is there, in six more blocks (the decoder).
Every query ends in a box, (center x, center y, width, height) as fractions of the photo, and 92 logits: one for
each of the numbers 0 to 90 that COCO gives its kinds, and a last one for "no object". COCO's 80 kinds have numbers
from 1 to 90; the 10 unused numbers and 0 never had a box, so the model has hardly learned them. Training matched each
true object to exactly one query (the Hungarian algorithm), so the model learned to give every object mostly one box
and send the other queries to "no object": DETR needs no non-maximum suppression.

The course does not ship it (167 MB): everything_ai.hub downloads the file of facebook/detr-resnet-50 once, at a
fixed commit. The model has short names of the course's own; the file has the names of Hugging Face's code, and
`load` renames them. On the photo of two cats from COCO that Hugging Face's own tests use, this code gives their
numbers to within their tolerance of 0.0002 (tests/test_detr.py). Import this module only where the model is needed: it imports
PyTorch."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from everything_ai import hub, models

FILE = ("facebook/detr-resnet-50", "model.safetensors")
D, HEADS, FFN, LAYERS, QUERIES = 256, 8, 2048, 6, 100
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet's numbers, from preprocessor_config.json
SHORT, LONG = 800, 1333   # the shorter side of a photo becomes 800 pixels, the longer at most 1333

# The names for the numbers 0 to 90, in the order of the logits, from the model's config.json: "N/A" is the number 0,
# which COCO does not use, and 10 more numbers, such as 12, "street sign", were left without boxes in COCO; the last
# logit is "no object".
KINDS = (
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "no object",
)


class FrozenNorm(nn.Module):
    """Batch normalization with numbers that never change: DETR kept those of ResNet-50 on ImageNet."""

    def __init__(self, n: int):
        super().__init__()
        for name in ("weight", "bias", "running_mean", "running_var"):
            self.register_buffer(name, torch.zeros(n))

    def forward(self, x):
        scale = self.weight / torch.sqrt(self.running_var + 1e-5)
        shift = self.bias - self.running_mean * scale
        return x * scale[None, :, None, None] + shift[None, :, None, None]


class Bottleneck(nn.Module):
    """ResNet-50's residual block: a 1 by 1 convolution to `width` channels, a 3 by 3 one, and a 1 by 1 one to four
    times `width`, each with batch normalization, and the block's input added to their outcome."""

    def __init__(self, n_in: int, width: int, stride: int = 1):
        super().__init__()
        n_out = 4 * width
        self.conv1, self.bn1 = nn.Conv2d(n_in, width, 1, bias=False), FrozenNorm(width)
        self.conv2, self.bn2 = nn.Conv2d(width, width, 3, stride, 1, bias=False), FrozenNorm(width)
        self.conv3, self.bn3 = nn.Conv2d(width, n_out, 1, bias=False), FrozenNorm(n_out)
        self.down = None
        if stride != 1 or n_in != n_out:
            self.down = nn.Sequential(nn.Conv2d(n_in, n_out, 1, stride, bias=False), FrozenNorm(n_out))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return F.relu(y + (x if self.down is None else self.down(x)))


class ResNet50(nn.Module):
    """ResNet-50 without its last layer: four stages of 3, 4, 6 and 3 blocks, 2,048 channels at the end, for every
    square of 32 by 32 pixels."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False), FrozenNorm(64)
        stages, n_in = [], 64
        for width, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
            stage = [Bottleneck(n_in, width, stride)] + [Bottleneck(4 * width, width) for _ in range(blocks - 1)]
            stages.append(nn.Sequential(*stage))
            n_in = 4 * width
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class Attention(nn.Module):
    """Attention with 8 heads, as in the GPT of the course, without a causal mask: every query sees every key."""

    def __init__(self):
        super().__init__()
        self.query, self.key, self.value, self.out = (nn.Linear(D, D) for _ in range(4))

    def forward(self, q, k, v):
        n, tq, tk, size = q.shape[0], q.shape[1], k.shape[1], D // HEADS
        q = self.query(q).view(n, tq, HEADS, size).transpose(1, 2)
        k = self.key(k).view(n, tk, HEADS, size).transpose(1, 2)
        v = self.value(v).view(n, tk, HEADS, size).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(size), dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(n, tq, D))


class EncoderBlock(nn.Module):
    """Every square looks at every other; the position is added to the queries and keys, not to the values. The
    layer normalization comes after each branch is added, as in the original transformer."""

    def __init__(self):
        super().__init__()
        self.attn, self.norm1 = Attention(), nn.LayerNorm(D)
        self.fc1, self.fc2, self.norm2 = nn.Linear(D, FFN), nn.Linear(FFN, D), nn.LayerNorm(D)

    def forward(self, x, pos):
        x = self.norm1(x + self.attn(x + pos, x + pos, x))
        return self.norm2(x + self.fc2(F.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    """The 100 queries look at each other, then at the squares the encoder made, then each goes through a small
    network of its own."""

    def __init__(self):
        super().__init__()
        self.attn, self.norm1 = Attention(), nn.LayerNorm(D)
        self.cross, self.norm2 = Attention(), nn.LayerNorm(D)
        self.fc1, self.fc2, self.norm3 = nn.Linear(D, FFN), nn.Linear(FFN, D), nn.LayerNorm(D)

    def forward(self, x, queries, memory, pos):
        x = self.norm1(x + self.attn(x + queries, x + queries, x))
        x = self.norm2(x + self.cross(x + queries, memory + pos, memory))
        return self.norm3(x + self.fc2(F.relu(self.fc1(x))))


def positions(rows: int, cols: int) -> torch.Tensor:
    """The place of every square, as 256 numbers: 128 sines and cosines of its row and 128 of its column, at
    wavelengths from 2 pi to 10,000 times that, with the rows and columns scaled to run up to 2 pi. Shape
    (rows * cols, 256), row after row."""
    y = torch.arange(1, rows + 1, dtype=torch.float32)[:, None].expand(rows, cols) / (rows + 1e-6) * 2 * math.pi
    x = torch.arange(1, cols + 1, dtype=torch.float32)[None, :].expand(rows, cols) / (cols + 1e-6) * 2 * math.pi
    i = torch.arange(D // 2, dtype=torch.float32)
    waves = 10000 ** (2 * torch.div(i, 2, rounding_mode="floor") / (D // 2))
    out = []
    for place in (y, x):
        a = place[:, :, None] / waves
        out.append(torch.stack((a[:, :, 0::2].sin(), a[:, :, 1::2].cos()), dim=3).flatten(2))
    return torch.cat(out, dim=2).reshape(rows * cols, D)


class DETR(nn.Module):
    """DETR with ResNet-50: 41.5 million parameters. `forward` gives the 100 queries' logits and boxes."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50()
        self.project = nn.Conv2d(2048, D, 1)
        self.encoder = nn.ModuleList(EncoderBlock() for _ in range(LAYERS))
        self.decoder = nn.ModuleList(DecoderBlock() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.queries = nn.Parameter(torch.zeros(QUERIES, D))   # the object queries
        self.classes = nn.Linear(D, len(KINDS))
        self.box = nn.ModuleList([nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, 4)])

    def forward(self, x):
        """x: (n, 3, h, w), prepared as `prepare` does; n photos of one size. Gives the logits, (n, 100, 92), and the
        boxes, (n, 100, 4): (center x, center y, width, height), each a fraction of the photo's width or height."""
        squares = self.project(self.backbone(x))   # (n, 256, rows, cols)
        n, _, rows, cols = squares.shape
        memory = squares.flatten(2).transpose(1, 2)   # (n, rows * cols, 256)
        pos = positions(rows, cols)[None]
        for block in self.encoder:
            memory = block(memory, pos)
        queries = self.queries[None].expand(n, -1, -1)
        h = torch.zeros(n, QUERIES, D)
        for block in self.decoder:
            h = block(h, queries, memory, pos)
        h = self.norm(h)
        b = F.relu(self.box[1](F.relu(self.box[0](h))))
        return self.classes(h), torch.sigmoid(self.box[2](b))


def _rename(name: str) -> str | None:
    """The course's name for a tensor in the file of facebook/detr-resnet-50, or None for one the model does not use."""
    if name.endswith("num_batches_tracked"):   # a count batch normalization keeps while it trains
        return None
    for old, new in (("model.backbone.conv_encoder.model.", "backbone."), (".downsample.", ".down."),
                     ("model.input_projection", "project"), ("model.query_position_embeddings.weight", "queries"),
                     ("model.encoder.layers", "encoder"), ("model.decoder.layers", "decoder"),
                     ("model.decoder.layernorm", "norm"), ("class_labels_classifier", "classes"),
                     ("bbox_predictor.layers", "box"), (".self_attn_layer_norm.", ".norm1."),
                     (".encoder_attn_layer_norm.", ".norm2."), (".self_attn.", ".attn."), (".encoder_attn.", ".cross."),
                     (".q_proj.", ".query."), (".k_proj.", ".key."), (".v_proj.", ".value."), (".out_proj.", ".out.")):
        name = name.replace(old, new)
    if ".final_layer_norm." in name:   # the last norm of a block: the second in the encoder, the third in the decoder
        name = name.replace(".final_layer_norm.", ".norm3." if name.startswith("decoder") else ".norm2.")
    return name


def downloaded() -> bool:
    return hub.pinned(*FILE) is not None


def load() -> DETR:
    """DETR in eval mode, from the file everything_ai.hub downloaded. It raises FileNotFoundError when that file is not
    there yet: hub.download(*FILE) gets it."""
    path = hub.pinned(*FILE)
    if path is None:
        raise FileNotFoundError("DETR is not downloaded yet: everything_ai.hub.download(*everything_ai.detr.FILE)")
    state = {}
    for name, t in models.read_safetensors(path).items():
        new = _rename(name)
        if new is not None:
            state[new] = t
    model = DETR()
    model.load_state_dict(state)
    model.requires_grad_(False)
    return model.eval()


def size(height: int, width: int) -> tuple[int, int]:
    """The (height, width) a photo gets, as Hugging Face's DetrImageProcessor computes it: the shorter side 800
    pixels and the other in proportion, rounded down, unless the longer side would then be over 1333 pixels; then the
    longer side becomes about 1333."""
    short, raw = SHORT, None
    if max(height, width) / min(height, width) * SHORT > LONG:
        raw = LONG * min(height, width) / max(height, width)   # the shorter side that makes the longer one 1333
        short = int(round(raw))
    if min(height, width) == short:
        return height, width
    if width < height:
        return int((raw or short) * height / width), short
    return short, int((raw or short) * width / height)


def prepare(photo) -> torch.Tensor:
    """A photo (a path or a PIL image) as the model takes it: resized bilinearly to `size`, every color scaled with
    ImageNet's numbers, and a float32 tensor of shape (1, 3, h, w). A gray photo gets three equal colors."""
    img = (photo if isinstance(photo, Image.Image) else Image.open(photo)).convert("RGB")
    h, w = size(img.height, img.width)
    x = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
