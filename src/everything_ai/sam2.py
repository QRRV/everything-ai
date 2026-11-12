# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""SAM 2.1 tiny, the smallest Segment Anything Model 2.1 of Meta (Ravi and others, 2024, arXiv:2408.00714;
Apache-2.0), written out for photos, so that nothing beyond PyTorch is needed. You point at something in a photo with a
click or a box, and it draws that thing's mask, also of things it has no name for. It learned on SA-1B, 11 million
photos with 1.1 billion masks, and on SA-V, about 51,000 videos.

It has three parts. The image encoder, a Hiera of 27 million parameters with a small FPN after it, turns the photo, made
1024 by 1024 pixels, into features: 256 numbers for each square of 16 by 16 pixels, 64 by 64 squares, and two finer
maps for the edges. It runs once per photo. The prompt encoder turns every click and the corners of a box into 256
numbers. The mask decoder, a small transformer, lets those numbers and the features look at each other, and gives
four masks of 256 by 256 logits, each with the IoU it expects that mask to have with the thing you meant. The
decoder takes a fraction of a second, so you can click again and again on one photo. For video, SAM 2.1 also keeps a
memory of earlier frames; the course leaves that part out.

The course does not ship it (156 MB): everything_ai.hub downloads the file of facebook/sam2.1-hiera-tiny once, at a
fixed commit. The model has short names of the course's own; the file has the names of Hugging Face's code, and
`load` renames them. Import this module only where the model is needed: it imports PyTorch."""
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from everything_ai import hub, models

FILE = ("facebook/sam2.1-hiera-tiny", "model.safetensors")
SIZE = 1024   # the photo is made SIZE by SIZE pixels
D = 256       # numbers per square of the features, per click and per token of the decoder
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet's numbers, from the model's preprocessor_config.json
# Hiera tiny: 12 blocks in 4 stages; from stage to stage the squares halve per side and the numbers per square double
STAGES, WIDTHS, HEADS = (1, 2, 7, 2), (96, 192, 384, 768), (1, 2, 4, 8)
WINDOWS = (8, 4, 14, 7)    # attention within windows of this many squares per side, per stage
EVERYWHERE = (5, 7, 9)     # blocks whose attention looks at the whole photo instead of a window
POINT, NEGATIVE, TOP_LEFT, BOTTOM_RIGHT, NOT_A_POINT = 1, 0, 2, 3, -1   # the labels of the prompt's points


def _windows(x, w):
    """Cut a map (n, h, w', c) into windows of w by w squares, after padding it with zeros to a whole number of them."""
    n, h, wd, c = x.shape
    ph, pw = (w - h % w) % w, (w - wd % w) % w
    x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, wd + pw
    x = x.view(n, hp // w, w, wp // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w, w, c)
    return x, (hp, wp)


def _unwindow(x, w, padded, size):
    """Put windows of w by w squares back into one map of the padded size, and cut it to size."""
    hp, wp = padded
    n = x.shape[0] // (hp * wp // w // w)
    x = x.reshape(n, hp // w, wp // w, w, w, -1).permute(0, 1, 3, 2, 4, 5).reshape(n, hp, wp, -1)
    return x[:, :size[0], :size[1], :]


def _pool(x):
    """Halve a map (n, h, w, c) per side, with the maximum of every 2 by 2 squares."""
    return F.max_pool2d(x.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)


class Unit(nn.Module):
    """A block of Hiera: attention within windows of squares, or over the whole map, then an MLP, each added to what
    came in. The first block of a stage halves the squares per side by pooling the queries, and widens the numbers."""

    def __init__(self, width, width_out, heads, window, pool):
        super().__init__()
        self.heads, self.window, self.pool = heads, window, pool
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width_out)
        self.out = nn.Linear(width_out, width_out)
        self.norm2 = nn.LayerNorm(width_out, eps=1e-6)
        self.fc1, self.fc2 = nn.Linear(width_out, 4 * width_out), nn.Linear(4 * width_out, width_out)
        self.proj = nn.Linear(width, width_out) if width != width_out else None

    def attention(self, x):
        n, h, w, _ = x.shape
        q, k, v = torch.unbind(self.qkv(x).reshape(n, h * w, 3, self.heads, -1), 2)
        if self.pool:
            q = _pool(q.reshape(n, h, w, -1))
            h, w = q.shape[1:3]
            q = q.reshape(n, h * w, self.heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(x.transpose(1, 2).reshape(n, h, w, -1))

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.proj is not None:
            shortcut = self.proj(x)
        if self.pool:
            shortcut = _pool(shortcut)
        size = shortcut.shape[1:3]
        if self.window:
            x, padded = _windows(x, self.window)
        x = self.attention(x)
        if self.window:
            w = self.window // 2 if self.pool else self.window
            padded = tuple(s + (w - s % w) % w for s in size)
            x = _unwindow(x, w, padded, size)
        x = shortcut + x
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Hiera(nn.Module):
    """The image encoder's backbone: a photo of 1024 by 1024 pixels becomes squares of 4 by 4 pixels, 256 by 256 of
    them, with 96 numbers each; every stage halves the squares per side. `forward` gives the map after every stage."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, WIDTHS[0], 7, 4, 3)
        self.pos = nn.Parameter(torch.zeros(1, WIDTHS[0], 7, 7))            # a position, stretched over the map
        self.pos_window = nn.Parameter(torch.zeros(1, WIDTHS[0], 8, 8))     # and one repeated in every window
        self.blocks = nn.ModuleList()
        ends = list(np.cumsum(STAGES) - 1)   # the last block of every stage: 0, 2, 9 and 11
        stage, width = 0, WIDTHS[0]
        for i in range(sum(STAGES)):
            # the first block of a stage still uses the window of the stage before
            window = 0 if i in EVERYWHERE else WINDOWS[stage]
            first = i - 1 in ends
            if first:
                stage += 1
            self.blocks.append(Unit(width, WIDTHS[stage], HEADS[stage], window, first))
            width = WIDTHS[stage]
        self.ends = ends

    def forward(self, x):
        x = self.patch(x).permute(0, 2, 3, 1)
        h, w = x.shape[1:3]
        pos = F.interpolate(self.pos, size=(h, w), mode="bicubic")
        pos = pos + self.pos_window.tile(1, 1, h // 8, w // 8)
        x = x + pos.permute(0, 2, 3, 1)
        maps = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.ends:
                maps.append(x.permute(0, 3, 1, 2))
        return maps


class LayerNorm2d(nn.Module):
    """Layer normalization over the channels of every square of a map."""

    def __init__(self, channels):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - mean) / torch.sqrt(var + 1e-6) + self.bias[:, None, None]


class MLP(nn.Module):
    """Three linear layers with ReLU between them."""

    def __init__(self, width_out):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, width_out)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class Attention(nn.Module):
    """Attention with 8 heads, in which a query, a key and a value can come from different places; the cross
    attentions work with half the numbers."""

    def __init__(self, inner):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(D, inner), nn.Linear(D, inner), nn.Linear(D, inner)
        self.out = nn.Linear(inner, D)

    def forward(self, q, k, v):
        def heads(x):
            n, t, c = x.shape
            return x.reshape(n, t, 8, c // 8).transpose(1, 2)
        x = F.scaled_dot_product_attention(heads(self.query(q)), heads(self.key(k)), heads(self.value(v)))
        n, _, t, c = x.shape
        return self.out(x.transpose(1, 2).reshape(n, t, 8 * c))


class TwoWay(nn.Module):
    """A layer of the decoder's transformer: the tokens look at each other, then at the squares of the photo; an MLP
    for every token; then every square looks at the tokens."""

    def __init__(self, first):
        super().__init__()
        self.first = first
        self.self_attn = Attention(D)
        self.to_image, self.to_tokens = Attention(D // 2), Attention(D // 2)
        self.norm1, self.norm2, self.norm3, self.norm4 = (nn.LayerNorm(D) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(D, 2048), nn.Linear(2048, D)

    def forward(self, tokens, squares, tokens_pe, squares_pe):
        if self.first:
            tokens = self.self_attn(tokens, tokens, tokens)
        else:
            q = tokens + tokens_pe
            tokens = tokens + self.self_attn(q, q, tokens)
        tokens = self.norm1(tokens)
        tokens = self.norm2(tokens + self.to_image(tokens + tokens_pe, squares + squares_pe, squares))
        tokens = self.norm3(tokens + self.fc2(F.relu(self.fc1(tokens))))
        squares = self.norm4(squares + self.to_tokens(squares + squares_pe, tokens + tokens_pe, tokens))
        return tokens, squares


class SAM2(nn.Module):
    """SAM 2.1 tiny for photos: `features` once per photo, then `decode` for every prompt."""

    def __init__(self):
        super().__init__()
        self.hiera = Hiera()
        # the FPN: a 1 by 1 convolution to 256 numbers per stage, from the last stage to the first
        self.lateral = nn.ModuleList(nn.Conv2d(c, D, 1) for c in reversed(WIDTHS))
        self.fine0, self.fine1 = nn.Conv2d(D, 32, 1), nn.Conv2d(D, 64, 1)   # the finer maps, for the edges
        self.no_memory = nn.Parameter(torch.zeros(1, D, 1, 1))   # added to the features of a photo without a video
        # the prompt encoder
        self.gauss = nn.Parameter(torch.zeros(2, D // 2))        # random directions for the positions
        self.labels = nn.Parameter(torch.zeros(4, D))            # one per label: negative, point, top-left, bottom-right
        self.not_a_point = nn.Parameter(torch.zeros(1, D))
        self.no_mask = nn.Parameter(torch.zeros(1, D))
        # the mask decoder
        self.iou_token, self.mask_tokens = nn.Parameter(torch.zeros(1, D)), nn.Parameter(torch.zeros(4, D))
        self.object_token = nn.Parameter(torch.zeros(1, D))
        self.layers = nn.ModuleList(TwoWay(i == 0) for i in range(2))
        self.final, self.final_norm = Attention(D // 2), nn.LayerNorm(D)
        self.up1, self.up_norm, self.up2 = nn.ConvTranspose2d(D, 64, 2, 2), LayerNorm2d(64), nn.ConvTranspose2d(64, 32, 2, 2)
        self.hyper = nn.ModuleList(MLP(32) for _ in range(4))
        self.iou_head = MLP(4)

    def features(self, x):
        """The features of a batch of prepared photos (n, 3, 1024, 1024): the map of 64 by 64 squares with 256
        numbers, and the finer maps of 128 by 128 with 64 numbers and of 256 by 256 with 32."""
        maps = self.hiera(x)
        coarse = self.lateral[0](maps[3])
        middle = self.lateral[1](maps[2]) + F.interpolate(coarse, scale_factor=2.0, mode="nearest")
        return middle + self.no_memory, self.fine1(self.lateral[2](maps[1])), self.fine0(self.lateral[3](maps[0]))

    def encode(self, xy):
        """Points (..., 2) in pixels of the 1024 by 1024 photo, x across and y down, as 256 numbers each: sines and
        cosines of the point, from the top-left corner at -1 to the bottom-right at 1, along 128 random directions."""
        xy = 2 * (xy / SIZE) - 1
        angle = 2 * math.pi * (xy @ self.gauss)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def prompt(self, points, labels):
        """The tokens of a prompt: points (n, k, 2) in pixels of the 1024 by 1024 photo, with their labels (n, k): 1 for a
        click on the thing, 0 for a click on what is not, 2 and 3 for the corners of a box. A point that is not a point
        is added at the end, as SAM 2.1 learned with."""
        points = torch.cat([points + 0.5, torch.zeros_like(points[:, :1])], dim=1)   # + 0.5: the pixel's middle
        labels = torch.cat([labels, torch.full_like(labels[:, :1], NOT_A_POINT)], dim=1)
        tokens = self.encode(points)
        tokens = torch.where(labels[..., None] == NOT_A_POINT, self.not_a_point, tokens + self.labels[labels.clamp(min=0)])
        return tokens

    def decode(self, features, points, labels):
        """For every prompt: four masks of 256 by 256 logits, the first for when the prompt says exactly what it means,
        the other three for a click that could mean more than one thing, such as a part or the whole, in no fixed
        order, and the IoU the decoder expects of each."""
        image, fine1, fine0 = features
        sparse = self.prompt(points, labels)
        n = sparse.shape[0]
        out = torch.cat([self.object_token, self.iou_token, self.mask_tokens]).expand(n, -1, -1)
        tokens = torch.cat([out, sparse], dim=1)
        squares = (image + self.no_mask[:, :, None, None]).expand(n, -1, -1, -1)
        _, c, h, w = squares.shape
        rows, cols = torch.meshgrid((torch.arange(h) + 0.5) * SIZE / h, (torch.arange(w) + 0.5) * SIZE / w, indexing="ij")
        squares_pe = self.encode(torch.stack([cols, rows], dim=-1)).reshape(1, h * w, c)
        squares = squares.flatten(2).transpose(1, 2)
        t = tokens
        for layer in self.layers:
            t, squares = layer(t, squares, tokens, squares_pe)
        t = self.final_norm(t + self.final(t + tokens, squares + squares_pe, squares))
        squares = squares.transpose(1, 2).reshape(n, c, h, w)
        up = F.gelu(self.up_norm(self.up1(squares) + fine1))
        up = F.gelu(self.up2(up) + fine0)
        hyper = torch.stack([self.hyper[i](t[:, 2 + i]) for i in range(4)], dim=1)   # (n, 4, 32)
        masks = (hyper @ up.flatten(2)).reshape(n, 4, *up.shape[2:])
        return masks, torch.sigmoid(self.iou_head(t[:, 1]))


def prepare(photo: Image.Image) -> torch.Tensor:
    """A photo as SAM 2.1 takes it, of shape (1, 3, 1024, 1024): made 1024 by 1024 pixels, whatever its shape, with
    bilinear interpolation on its bytes, as Hugging Face's Sam2ImageProcessor does, and normalized with ImageNet's
    mean and std."""
    x = torch.from_numpy(np.array(photo.convert("RGB"))).permute(2, 0, 1)[None].contiguous()
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True).float() / 255
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def to_photo(logits, width: int, height: int) -> np.ndarray:
    """Masks of logits, (..., 256, 256), made as large as the photo, (..., height, width), with bilinear
    interpolation, as a NumPy array. A pixel belongs to a mask where its logit is above 0."""
    t = torch.as_tensor(logits, dtype=torch.float32)
    shape = t.shape[:-2]
    t = F.interpolate(t.reshape(-1, 1, *t.shape[-2:]), size=(height, width), mode="bilinear", align_corners=False)
    return t.reshape(*shape, height, width).numpy()


def deepest(region) -> tuple[int, int] | None:
    """The pixel deepest inside a region, a binary image of True and False, as (x, y): x across and y down, from the
    top-left pixel. The region is eroded with a square of 3 by 3, again and again, until one more erosion would leave
    nothing; pixels beyond the edge of the image count as False. Of the pixels that are left, it gives the one nearest
    to their middle. None for a region without any pixel."""
    r = np.asarray(region, dtype=bool)
    if not r.any():
        return None
    h, w = r.shape
    while True:
        padded = np.pad(r, 1)
        eroded = np.ones_like(r)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + h, dx:dx + w]
        if not eroded.any():
            break
        r = eroded
    ys, xs = np.nonzero(r)
    i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    return int(xs[i]), int(ys[i])


_NAMES = [
    (r"vision_encoder\.backbone\.patch_embed\.projection\.", "hiera.patch."),
    (r"vision_encoder\.backbone\.pos_embed$", "hiera.pos"),
    (r"vision_encoder\.backbone\.pos_embed_window$", "hiera.pos_window"),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.", r"hiera.blocks.\1.qkv."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.", r"hiera.blocks.\1.out."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.layer_norm(\d)\.", r"hiera.blocks.\1.norm\2."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.proj_in\.", r"hiera.blocks.\1.fc1."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.proj_out\.", r"hiera.blocks.\1.fc2."),
    (r"vision_encoder\.backbone\.blocks\.(\d+)\.proj\.", r"hiera.blocks.\1.proj."),
    (r"vision_encoder\.neck\.convs\.", "lateral."),
    (r"mask_decoder\.conv_s0\.", "fine0."),
    (r"mask_decoder\.conv_s1\.", "fine1."),
    (r"no_memory_embedding$", "no_memory"),
    (r"prompt_encoder\.shared_embedding\.positional_embedding$", "gauss"),
    (r"prompt_encoder\.point_embed\.weight$", "labels"),
    (r"prompt_encoder\.not_a_point_embed\.weight$", "not_a_point"),
    (r"prompt_encoder\.no_mask_embed\.weight$", "no_mask"),
    (r"mask_decoder\.iou_token\.weight$", "iou_token"),
    (r"mask_decoder\.mask_tokens\.weight$", "mask_tokens"),
    (r"mask_decoder\.obj_score_token\.weight$", "object_token"),
    (r"mask_decoder\.transformer\.layers\.(\d)\.self_attn\.", r"layers.\1.self_attn."),
    (r"mask_decoder\.transformer\.layers\.(\d)\.cross_attn_token_to_image\.", r"layers.\1.to_image."),
    (r"mask_decoder\.transformer\.layers\.(\d)\.cross_attn_image_to_token\.", r"layers.\1.to_tokens."),
    (r"mask_decoder\.transformer\.layers\.(\d)\.layer_norm(\d)\.", r"layers.\1.norm\2."),
    (r"mask_decoder\.transformer\.layers\.(\d)\.mlp\.proj_in\.", r"layers.\1.fc1."),
    (r"mask_decoder\.transformer\.layers\.(\d)\.mlp\.proj_out\.", r"layers.\1.fc2."),
    (r"mask_decoder\.transformer\.final_attn_token_to_image\.", "final."),
    (r"mask_decoder\.transformer\.layer_norm_final_attn\.", "final_norm."),
    (r"mask_decoder\.upscale_conv1\.", "up1."),
    (r"mask_decoder\.upscale_layer_norm\.", "up_norm."),
    (r"mask_decoder\.upscale_conv2\.", "up2."),
    (r"mask_decoder\.output_hypernetworks_mlps\.(\d)\.", r"hyper.\1."),
    (r"mask_decoder\.iou_prediction_head\.", "iou_head."),
]
_INSIDE = [(r"\.q_proj\.", ".query."), (r"\.k_proj\.", ".key."), (r"\.v_proj\.", ".value."), (r"\.o_proj\.", ".out."),
           (r"\.proj_in\.", ".fc1."), (r"\.layers\.0\.", ".fc2."), (r"\.proj_out\.", ".fc3.")]


def _rename(name: str) -> str | None:
    """The course's name for a tensor of the file, or None for the parts the course leaves out: the memory for video,
    the encoder of a mask as a prompt, and the head that says whether an object is there at all."""
    for old, new in _NAMES:
        if re.match(old, name):
            name = re.sub(old, new, name)
            if not name.startswith("hiera."):
                for o, n in _INSIDE:
                    name = re.sub(o, n, name)
            return name
    return None


def load(path=None) -> SAM2:
    """SAM 2.1 tiny with the parameters Meta published, from the pinned file (downloaded once by everything_ai.hub)."""
    path = path or hub.pinned(*FILE)
    model = SAM2()
    state = {}
    for name, t in models.read_safetensors(path).items():
        new = _rename(name)
        if new is None:
            continue
        if new == "no_memory":
            t = t.reshape(1, D, 1, 1)
        state[new] = t
    model.load_state_dict(state)
    return model.eval()
