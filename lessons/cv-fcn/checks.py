# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-fcn: class_map, the class of every pixel of a scene from the masks of its shapes;
transposed, a transposed convolution by hand; upsampler, the transposed convolution of the course's FCN, started as
bilinear upsampling; and kind_iou, the IoU of the pixels of one kind. class_map, upsampler and kind_iou are saved in
the learner's project (work/projects/cnn/classes.py, upsample.py and kind_iou.py) when right. The demo trains the
course's FCN with them on the course's scenes and measures it on new ones. And the pictures: what the FCN makes of
the scene of the worked example, step by step, and for the slider, what the course's FCN gives that scene at every
output stride. Everything is computed from the course's own scenes (scenes.py): no download."""
import base64
import functools
import html
import io
import math
import re
import zlib

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn

from everything_ai import chart, projects, scenes
from everything_ai.check import Mistake, ScenarioBox, Timeout, run_code, with_time_limit
from everything_ai import yours

SEED = 18                     # the scene of the worked example: scenes.scene(18), the scene of cv-annotations and cv-iou
KINDS = scenes.KINDS          # circle, square, triangle: classes 1 to 3; 0 is background
CLASSES = 1 + len(KINDS)
N_TRAIN, N_TEST = 2000, 200
TRAIN, TEST = range(1000, 1000 + N_TRAIN), range(N_TEST)   # the seeds of the scenes, as in the one-stage detector
EPOCHS, LR, BATCH = 8, 0.01, 32
CHANNELS = (16, 32, 64, 64, 64)   # the channels of the blocks, one block for every halving of the maps
STRIDE = 8                    # the output stride of the course's FCN: its maps are 8 times smaller than the scene
STRIDES = (4, 8, 16, 32)      # the output strides the slider shows, each an FCN the course trained
SIZES = (96, 192)             # the other scene sizes the trained FCN is tried on
SHOWN = (18, 3, 23)           # the test scenes of the rewind
TRAIN_LIMIT = 600             # seconds the demo gives the training: the note says ten minutes

# The FCNs the course compares, each trained with the seeds 0 to 4: the output stride, how its transposed
# convolution starts and whether it learns, and the epochs. "learned": started as bilinear upsampling, and learned;
# "random": started as PyTorch starts every layer, and learned; "bilinear" and "nearest": fixed to that upsampling.
VARIANTS = {
    "s4": (4, "learned", EPOCHS),
    "s8": (8, "learned", EPOCHS),
    "s16": (16, "learned", EPOCHS),
    "s32": (32, "learned", EPOCHS),
    "s8-random": (8, "random", EPOCHS),
    "s8-bilinear": (8, "bilinear", EPOCHS),
    "s8-nearest": (8, "nearest", EPOCHS),
    "s16-long": (16, "learned", 2 * EPOCHS),
    "s32-long": (32, "learned", 2 * EPOCHS),
}
BY_STRIDE = {4: "s4", 8: "s8", 16: "s16", 32: "s32"}

# The worked example's numbers, measured again by `measure()`.
REFERENCE = {
    "counts": [14622, 694, 653, 415],
    "acc": 0.8925,
    "steps": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 25.0, 75.0, 75.0, 25.0, 0.0, 0.0],
              [0.0, 0.0, 25.0, 75.0, 125.0, 175.0, 150.0, 50.0]],
    "row": [0.0, 25.0, 75.0, 125.0, 175.0, 150.0],
    "nearest": [0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
    "shapes": [(1, 16, 64, 64), (1, 32, 32, 32), (1, 64, 16, 16), (1, 4, 16, 16), (1, 4, 128, 128)],
    "loss": 1.3862,
    "background": 0.8696,
}
# The course's FCNs, trained as VARIANTS says with the seeds 0 to 4 by train_course, measured by measure_variant on the
# 200 test scenes: the share of pixels right, the IoU of every kind, the loss; for seed 0 of the strides, the class
# map of the scene of the worked example; for the course's FCN, scenes of 96 and 192 pixels, and with seed 0 its
# coarse map and its loss curves. measure_trained() measures it all again.
TRAINED = {
    "s4": {
        "acc": [0.9569, 0.9467, 0.9399, 0.9452, 0.9429],
        "ious": [[0.6605, 0.3232, 0.457], [0.5908, 0.1529, 0.3513], [0.5433, 0.0737, 0.2951], [0.5819, 0.107, 0.357], [0.5612, 0.1155, 0.3184]],
        "val": [0.1074, 0.1279, 0.1354, 0.1277, 0.1325],
        "map": "eNrt1L1twzAQBWDKhlO4YeXGExgZIQggj6AiF8RsUnkPZwiOQCBmoxGSkVQHBs78EWBH1B2DKK7C136Q+HCUToiSkpKSkkw2PM/sgfWl1cD6EVwU+ZI7E/yd8hUEtzk/3MqVL6jMlnUwY/Z48dEZIe8SPyd5y/ss+m5PuETvawDC2+DqGD2Zz/LjyvepI165gWbA896tDfPXT2m73nW4H8j5oED9zZ+TAjV+tZfzXZphP9ti534Pa1TwYcGHjAvnp+h61Bev0XW8wcSFd8G49H7/BpQL70LQLrvQA4DwvmfGq7/xhlwxP/Ptrz2sCEWvuDCAF9orvl48gNuhFf94yX8MTvO5X25MJHYT/ZRxzDlb0O3vbqKzBTH/vOCdn29dvuAb5gzygep6",
    },
    "s8": {
        "acc": [0.9802, 0.9739, 0.9744, 0.9684, 0.9704],
        "ious": [[0.8725, 0.7125, 0.7261], [0.8139, 0.6192, 0.6732], [0.8328, 0.6262, 0.652], [0.7788, 0.5268, 0.5995], [0.7892, 0.5652, 0.6282]],
        "val": [0.0604, 0.0771, 0.076, 0.0838, 0.084],
        "map": "eNrt1T0SgjAQBeAolTapPIKNl4AjUBAKTuBwCo5iyeQUnMcqB2AmxgVpzL51zFiZ134kbHbyo1ROTk5OTmLumAtrr8j3lsL6cfFB8JvgY6KzBV6AF4J7B10LXgru01wL7skt50XwCXgZXK1uui42/TY++BBzt3k7Qu9bi930Ffa2xm6gW2OayPoXN5Tmvf1r/xc38f7zHj6Yaf9yrp8eJuCc9mc4oKxrRyeAdTXRCYqvb8su0dVnXrN+EvxMXrF+4Jb/uuJweWEPwN8vHYaXLKou5z/jhHfZT9Dp8UnyGXpJt//37mWffuxJ9ZVCf6T+5qTkAX2NHLo=",
        "coarse": "eNpjYMACrA4wMGs1MPADsb7WAgZmrgUMDEwNcGkbKAYBAJtsBQE=",
        "curves": {"train": [0.5183, 0.1633, 0.1259, 0.1015, 0.076, 0.0683, 0.0633, 0.0616], "val": [0.2383, 0.1439, 0.1141, 0.0836, 0.0678, 0.064, 0.0625, 0.0604]},
        "size96": {"acc": [0.9514, 0.9391, 0.9414, 0.9341, 0.9357], "ious": [[0.7694, 0.5699, 0.5907], [0.7035, 0.4746, 0.5226], [0.7315, 0.4989, 0.5056], [0.6883, 0.4335, 0.4831], [0.6901, 0.4474, 0.4955]]},
        "size192": {"acc": [0.9936, 0.9909, 0.9911, 0.9881, 0.9893], "ious": [[0.9223, 0.8223, 0.8058], [0.8651, 0.7212, 0.7535], [0.8831, 0.7312, 0.7327], [0.8203, 0.6107, 0.675], [0.8374, 0.6634, 0.7139]]},
    },
    "s16": {
        "acc": [0.9305, 0.9582, 0.9346, 0.9538, 0.953],
        "ious": [[0.537, 0.2092, 0.267], [0.7243, 0.4745, 0.5344], [0.5612, 0.241, 0.3127], [0.6846, 0.4284, 0.495], [0.6762, 0.3948, 0.5065]],
        "val": [0.1683, 0.1132, 0.1608, 0.1196, 0.1237],
        "map": "eNrt1TEOwjAMBVBHqHOn3qdH8JB/H47CyPE4AJKhRCCEmu/QIJXBf32Na0dNKhKJRCKR/QPOEzLjAcBMl4MVSHeG8vJs/VIeZIIHkwaLK22PNNDoKw2O19d4q272jc+Oq+O54qnR4flngwez5Ymqi5UCqA1QuNnzRp9qPl64v3+/HZ46/dmgbvXBOUGpzb0T6pzw7NwQKvwFwl35FZjjJxGJ/DDnTj85fowt/uPcAAm5fR8=",
    },
    "s32": {
        "acc": [0.9285, 0.8966, 0.9284, 0.9199, 0.9255],
        "ious": [[0.5997, 0.296, 0.2696], [0.3776, 0.0262, 0.0007], [0.5596, 0.2239, 0.2613], [0.4998, 0.1591, 0.1792], [0.5439, 0.2097, 0.234]],
        "val": [0.1897, 0.2817, 0.1834, 0.2028, 0.1905],
        "map": "eNrt1cENgCAQRNHlQAn2syVw2O3MKqxSg16ZIXAymX99UUGzaKaUUur3XSfkmunI8ymI57rX7uMVHN0b8cCPH3t5PZn7otfPG/GA29v3ZO54fWmb7vj9UW9rbrMei777geqmF7I/5jbpgQdwfH1JMuFkgN8bOD7BAh5xRT8BpZSa6wbSh2A2",
    },
    "s8-random": {
        "acc": [0.9746, 0.9622, 0.9501, 0.9569, 0.9521],
        "ious": [[0.822, 0.6347, 0.6782], [0.7221, 0.4499, 0.5613], [0.6324, 0.2809, 0.4821], [0.6679, 0.3729, 0.5391], [0.6469, 0.3202, 0.4892]],
        "val": [0.0763, 0.1021, 0.1176, 0.1085, 0.1146],
    },
    "s8-bilinear": {
        "acc": [0.9797, 0.9621, 0.9721, 0.9573, 0.9572],
        "ious": [[0.87, 0.7075, 0.7171], [0.7192, 0.4436, 0.5538], [0.8168, 0.6043, 0.6156], [0.6915, 0.3765, 0.5107], [0.6766, 0.3684, 0.542]],
        "val": [0.0613, 0.1013, 0.0815, 0.1045, 0.1079],
    },
    "s8-nearest": {
        "acc": [0.9382, 0.9444, 0.9226, 0.9409, 0.9425],
        "ious": [[0.6019, 0.3197, 0.444], [0.665, 0.4208, 0.4553], [0.4955, 0.139, 0.2768], [0.6302, 0.3663, 0.4492], [0.6369, 0.3921, 0.4556]],
        "val": [0.1546, 0.1442, 0.1916, 0.1526, 0.1506],
    },
    "s16-long": {
        "acc": [0.9473, 0.975, 0.9717, 0.9737, 0.9712],
        "ious": [[0.659, 0.3301, 0.4154], [0.8614, 0.698, 0.6471], [0.8317, 0.6651, 0.63], [0.8561, 0.6824, 0.626], [0.8196, 0.6478, 0.6313]],
        "val": [0.1326, 0.0667, 0.0772, 0.071, 0.0782],
    },
    "s32-long": {
        "acc": [0.9468, 0.9231, 0.9525, 0.9465, 0.9507],
        "ious": [[0.7321, 0.509, 0.3694], [0.5205, 0.2053, 0.2182], [0.7496, 0.5173, 0.4447], [0.7138, 0.4472, 0.3615], [0.7497, 0.5372, 0.4054]],
        "val": [0.1427, 0.1971, 0.1276, 0.144, 0.1311],
    },
}


# --- the course's versions ------------------------------------------------------------------------------------------

def class_map(s) -> np.ndarray:
    """The class of every pixel of a scene, as the learner's: 0 for background, and the label of a shape plus 1 where
    its mask is."""
    classes = np.zeros(s.image.shape[:2], dtype=int)
    for shape in s.shapes:
        classes[shape.mask] = shape.label + 1
    return classes


def transposed(x, kernel, stride: int):
    """A transposed convolution by hand, as the learner's: every value of x paints a copy of the kernel, times that
    value, `stride` apart; where copies overlap, they add up."""
    k = len(kernel)
    out = torch.zeros((len(x) - 1) * stride + k, (len(x[0]) - 1) * stride + k)
    for i in range(len(x)):
        for j in range(len(x[0])):
            out[i * stride:i * stride + k, j * stride:j * stride + k] += x[i, j] * kernel
    return out


def bilinear_kernel(k: int) -> torch.Tensor:
    """A k by k kernel that is 1 in the middle and falls off in a straight line to the edges, in every direction: as
    a transposed convolution with stride k // 2 and padding k // 4, it upsamples bilinearly."""
    line = [1 - abs(i - (k - 1) / 2) / (k // 2) for i in range(k)]
    return torch.tensor([[a * b for b in line] for a in line])


def upsampler(classes: int, stride: int) -> nn.ConvTranspose2d:
    """The transposed convolution of the course's FCN, as the learner's: every one of `classes` maps `stride` times
    larger, with a kernel of 2 * stride and padding stride // 2, started as bilinear upsampling, each class on its
    own map."""
    up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride // 2, bias=False)
    nn.init.zeros_(up.weight)
    with torch.no_grad():
        for c in range(classes):
            up.weight[c, c] = bilinear_kernel(2 * stride)
    return up


def kind_iou(pred, classes, kind: int) -> float:
    """The IoU of the pixels of one kind, as the learner's: the pixels that are that kind in both, divided by the
    pixels that are that kind in either."""
    p, t = pred == kind, classes == kind
    return (p & t).sum() / (p | t).sum()


def receptive_field(stride: int) -> int:
    """How many pixels wide a part of the scene one place of the last map sees: every 3 by 3 convolution adds 2 times
    the step between neighbours of its input, every pooling of 2 adds that step once and doubles it."""
    field, step = 1, 1
    for _ in range(int(math.log2(stride))):
        field += 2 * step           # the 3 by 3 convolution
        field += step               # the pooling
        step *= 2
    return field + 2 * step         # the last 3 by 3 convolution


# --- the network ------------------------------------------------------------------------------------------------------

def make_up(up: str, stride: int, upsampler_fn=upsampler) -> nn.ConvTranspose2d:
    """The transposed convolution of one variant: each makes one nn.ConvTranspose2d, so that every variant with the
    same seed starts its convolutions the same."""
    if up == "nearest":   # every score copied into a block of stride by stride: a kernel of ones, as large as the stride
        layer = nn.ConvTranspose2d(CLASSES, CLASSES, stride, stride=stride, bias=False)
        nn.init.zeros_(layer.weight)
        with torch.no_grad():
            for c in range(CLASSES):
                layer.weight[c, c] = 1.0
    elif up == "random":
        layer = nn.ConvTranspose2d(CLASSES, CLASSES, 2 * stride, stride=stride, padding=stride // 2, bias=False)
    else:
        layer = upsampler_fn(CLASSES, stride)
    if up in ("nearest", "bilinear"):
        layer.weight.requires_grad_(False)
    return layer


class FCN(nn.Module):
    """The course's fully convolutional network: blocks of a 3 by 3 convolution, ReLU and max pooling of 2 until the
    maps are `stride` times smaller than the scene, one more 3 by 3 convolution with ReLU, a 1 by 1 convolution to a
    score for every class at every place, and a transposed convolution that makes the scores as large as the scene."""

    def __init__(self, stride: int = STRIDE, up: str = "learned", upsampler_fn=upsampler):
        super().__init__()
        chans = (3,) + CHANNELS[:int(math.log2(stride))]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans, chans[1:]))
        self.pool = nn.MaxPool2d(2)
        self.extra = nn.Conv2d(chans[-1], chans[-1], 3, padding=1)
        self.score = nn.Conv2d(chans[-1], CLASSES, 1)
        self.up = make_up(up, stride, upsampler_fn)

    def coarse(self, x):
        """The scores of every class at every place of the last map, `stride` times smaller than the scene."""
        for conv in self.convs:
            x = self.pool(torch.relu(conv(x)))
        return self.score(torch.relu(self.extra(x)))

    def forward(self, x):
        return self.up(self.coarse(x))


# --- the scenes, training and measuring ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    return scenes.scene(SEED)


@functools.lru_cache(maxsize=4)
def scene_set(which: str, size: int = scenes.SIZE) -> tuple:
    """The training or the test scenes of that size, their pictures as one tensor of bytes, shape (N, 3, size,
    size), and their class maps, shape (N, size, size)."""
    ss = tuple(scenes.scene(seed, size=size) for seed in (TRAIN if which == "train" else TEST))
    X = torch.tensor(np.stack([s.image for s in ss])).permute(0, 3, 1, 2).contiguous()
    C = torch.tensor(np.stack([class_map(s) for s in ss]), dtype=torch.int64)
    return ss, X, C


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Warmup and cosine decay, as the Trainer of nn-trainer."""
    if step < warmup:
        return (step + 1) / warmup
    if step >= total:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def outputs(model, X) -> torch.Tensor:
    """The model's scores for every pixel of every scene of X (bytes), in eval mode, without a graph."""
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X[i:i + 50].float() / 255) for i in range(0, len(X), 50)])
    model.train()
    return out


def scores(pred, C, iou_fn=kind_iou) -> tuple[float, list[float]]:
    """The share of all pixels whose class is right, and the IoU of every kind over all pixels of all scenes."""
    return float((pred == C).float().mean()), [float(iou_fn(pred, C, k)) for k in range(1, CLASSES)]


def train(model, data, test, seed: int = 0, epochs: int = EPOCHS, heard=None, iou_fn=kind_iou) -> list[dict]:
    """Train `model` on data = (X, C) with the cross-entropy of every pixel, as the course measured, and measure it
    on test after every epoch: its loss, the share of pixels right and the IoU of every kind. `heard` hears every 10
    batches how far an epoch is, with the mean loss of those 10 batches ("loss"), and after every epoch what it
    measured, with the class of every test pixel."""
    X, C = data
    Xt, Ct = test
    n = len(X)
    per = math.ceil(n / BATCH)
    total = epochs * per
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, per, total))
    order_rng = torch.Generator().manual_seed(seed + 1)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=order_rng)
        running = reported = 0.0
        for b, i in enumerate(range(0, n, BATCH), 1):
            idx = order[i:i + BATCH]
            loss = F.cross_entropy(model(X[idx].float() / 255), C[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if heard is not None and b % 10 == 0:
                heard({"epoch": epoch, "epochs": epochs, "batch": b, "batches": per, "loss": (running - reported) / 10})
                reported = running
        out = outputs(model, Xt)
        pred = out.argmax(1)
        acc, ious = scores(pred, Ct, iou_fn)
        history.append({"epoch": epoch, "train_loss": running / per, "val_loss": float(F.cross_entropy(out, Ct)),
                        "acc": acc, "ious": ious})
        if heard is not None:
            heard(history[-1] | {"epochs": epochs, "pred": pred})
    return history


def train_course(variant: str = "s8", seed: int = 0, heard=None) -> tuple:
    """An FCN of the course, trained as VARIANTS says with `seed` on one thread, as a check runs: (model, history)."""
    stride, up, epochs = VARIANTS[variant]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = FCN(stride, up)
        _, X, C = scene_set("train")
        _, Xt, Ct = scene_set("test")
        history = train(model, (X, C), (Xt, Ct), seed, epochs, heard)
    finally:
        torch.set_num_threads(threads)
    return model, history


# --- class maps as text -----------------------------------------------------------------------------------------------

def pack(classes) -> str:
    """A class map (classes 0 to 3) as short text: 2 bits a pixel, compressed, in base64."""
    flat = np.asarray(classes, dtype=np.uint8).ravel()
    flat = np.concatenate([flat, np.zeros(-len(flat) % 4, dtype=np.uint8)])
    packed = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]
    return base64.b64encode(zlib.compress(packed.tobytes(), 9)).decode("ascii")


def unpack(text: str, side: int) -> np.ndarray:
    """The class map of `pack`, side by side pixels."""
    packed = np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype=np.uint8)
    flat = np.stack([(packed >> 6) & 3, (packed >> 4) & 3, (packed >> 2) & 3, packed & 3], axis=1).ravel()
    return flat[:side * side].reshape(side, side).astype(int)


def measure_variant(variant: str, seed: int) -> dict:
    """Train one FCN and measure what TRAINED holds for it: on the test scenes the share of pixels right, the IoU of
    every kind and the loss; with seed 0 of the strides, the class map it gives the scene of the worked example; for
    the course's FCN, on scenes of other sizes too, and with seed 0 also its coarse map and its loss curves."""
    model, history = train_course(variant, seed)
    _, Xt, Ct = scene_set("test")
    out = outputs(model, Xt)
    acc, ious = scores(out.argmax(1), Ct)
    r = {"acc": round(acc, 4), "ious": [round(v, 4) for v in ious], "val": round(history[-1]["val_loss"], 4)}
    if variant in BY_STRIDE.values() and seed == 0:
        r["map"] = pack(out[TEST.index(SEED)].argmax(0))
    if variant == "s8":
        for size in SIZES:
            _, Xs, Cs = scene_set("test", size)
            a, i = scores(outputs(model, Xs).argmax(1), Cs)
            r[f"size{size}"] = {"acc": round(a, 4), "ious": [round(v, 4) for v in i]}
        if seed == 0:
            x = example().image
            with torch.no_grad():
                model.eval()
                r["coarse"] = pack(model.coarse(torch.tensor(x).permute(2, 0, 1)[None].float() / 255)[0].argmax(0))
            r["curves"] = {"train": [round(h["train_loss"], 4) for h in history],
                           "val": [round(h["val_loss"], 4) for h in history]}
    return r


def measure_trained(seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every variant of VARIANTS with every seed. Takes two to eight
    minutes per training on one thread, about three hours in all."""
    out = {}
    for variant in VARIANTS:
        runs = [measure_variant(variant, seed) for seed in seeds]
        out[variant] = {key: [r[key] for r in runs] for key in ("acc", "ious", "val")}
        for key in ("map", "coarse", "curves"):
            if key in runs[0]:
                out[variant][key] = runs[0][key]
        for size in SIZES:
            if f"size{size}" in runs[0]:
                out[variant][f"size{size}"] = {key: [r[f"size{size}"][key] for r in runs] for key in ("acc", "ious")}
    return out


# --- the worked example ---------------------------------------------------------------------------------------------

ROW = (0.0, 100.0, 200.0)          # the row of cv-resize-interpolation's guess
ROW_KERNEL = (0.25, 0.75, 0.75, 0.25)


def worked() -> dict:
    """The worked example's numbers, as its cells compute them: the pixels of every class in the scene, and what a
    network that says background everywhere gets right of it; the row painted by a transposed convolution, with a
    bilinear and a nearest-neighbor kernel; the shapes on the way through the course's FCN at the start, and its
    loss. And the share of background in all pixels of the test scenes, for the guess."""
    s = example()
    classes = class_map(s)
    counts = [int((classes == k).sum()) for k in range(CLASSES)]
    none = np.zeros_like(classes)
    acc = float((none == classes).mean())
    row, kernel = torch.tensor(ROW), torch.tensor(ROW_KERNEL)
    out = torch.zeros((len(row) - 1) * 2 + len(kernel))
    steps = []
    for i in range(len(row)):
        out[i * 2:i * 2 + len(kernel)] += row[i] * kernel
        steps.append(out.tolist())
    near = torch.zeros(len(row) * 2)
    for i in range(len(row)):
        near[i * 2:i * 2 + 2] += row[i] * torch.tensor([1.0, 1.0])
    torch.manual_seed(0)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    convs = nn.ModuleList([nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)])
    pool = nn.MaxPool2d(2)
    shapes = []
    for conv in convs:
        x = pool(torch.relu(conv(x)))
        shapes.append(tuple(x.shape))
    extra = nn.Conv2d(64, 64, 3, padding=1)
    score = nn.Conv2d(64, 4, 1)
    coarse = score(torch.relu(extra(x)))
    up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)
    o = up(coarse)
    loss = nn.CrossEntropyLoss()(o, torch.tensor(classes).unsqueeze(0))
    _, _, Ct = scene_set("test")
    return {"counts": counts, "acc": round(acc, 4), "steps": steps, "row": out[1:-1].tolist(), "nearest": near.tolist(),
            "shapes": shapes + [tuple(coarse.shape), tuple(o.shape)], "loss": round(loss.item(), 4),
            "background": round(float((Ct == 0).float().mean()), 4)}


def measure() -> dict:
    return worked()


# --- pictures ---------------------------------------------------------------------------------------------------------

KIND_MARKS = ("●", "■", "▲")   # the shape that goes with the group colour of each kind (style guide 1.7)
_JPEGS: dict = {}


def _jpeg(seed: int) -> str:
    """The picture of a scene as a JPEG in base64, to draw inside an SVG."""
    if seed not in _JPEGS:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(scenes.scene(seed).image, dtype=np.uint8)).save(buf, format="JPEG", quality=92)
        _JPEGS[seed] = base64.b64encode(buf.getvalue()).decode("ascii")
    return _JPEGS[seed]


def _svg(parts: list, width: int, height: int, alt: str | None = None) -> str:
    label = f' role="img" aria-label="{html.escape(alt, quote=True)}"' if alt else ' aria-hidden="true"'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"{label} '
            f'style="max-width:{width}px;width:100%;height:auto;display:block">' + "".join(parts) + "</svg>")


_TOP = 22   # room for a title above a picture


def _title(text: str, width: int) -> str:
    return (f'<text x="{width / 2:.1f}" y="15" style="font-size:12px;font-weight:600;fill:var(--ea-ink);'
            f'text-anchor:middle">{html.escape(text)}</text>')


def map_svg(classes, title: str, width: int = 150) -> str:
    """A class map as a picture: background in the page's colour, every kind in its group colour, in runs of pixels
    along every row; a thin frame around it."""
    classes = np.asarray(classes)
    n = len(classes)
    scale = width / n
    parts = [_title(title, width), f'<g transform="translate(0,{_TOP}) scale({scale:.5g})" shape-rendering="crispEdges">',
             f'<rect x="0" y="0" width="{n}" height="{n}" style="fill:var(--ea-page)"/>']
    for r in range(n):
        row = classes[r]
        c = 0
        while c < n:
            k = int(row[c])
            end = c + 1
            while end < n and row[end] == k:
                end += 1
            if k > 0:
                parts.append(f'<rect x="{c}" y="{r}" width="{end - c}" height="1" style="fill:var(--ea-group-{k})"/>')
            c = end
    parts.append("</g>")
    parts.append(f'<rect x="0.5" y="{_TOP + 0.5}" width="{width - 1}" height="{width - 1}" '
                 'style="fill:none;stroke:var(--ea-axis);stroke-width:1"/>')
    return _svg(parts, width, width + _TOP)


def photo_svg(seed: int, title: str, width: int = 150, squares=()) -> str:
    """The picture of a scene, with squares over it: each (top, left, side, dashed) in pixels of the scene."""
    scale = width / scenes.SIZE
    parts = [_title(title, width),
             f'<image href="data:image/jpeg;base64,{_jpeg(seed)}" x="0" y="{_TOP}" width="{width}" height="{width}" '
             'preserveAspectRatio="none"/>']
    for top, left, side, dashed in squares:
        dash = "stroke-dasharray:4 3;" if dashed else ""
        x, y, w = left * scale, _TOP + top * scale, side * scale
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{w:.1f}" style="fill:none;'
                     f'stroke:var(--ea-page);stroke-width:4;opacity:0.8"/>')   # a light edge, so the square shows on any colour
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{w:.1f}" style="fill:none;'
                     f'stroke:var(--ea-ink);stroke-width:1.8;{dash}"/>')
    return _svg(parts, width, width + _TOP)


def side_by_side(panels: list[tuple[str, int]], alt: str) -> str:
    """Pictures next to each other, that go below each other on a phone, so that their text keeps its size."""
    cells = "".join(f'<div style="flex:0 1 {basis}px;min-width:0">{svg}</div>' for svg, basis in panels)
    return (f'<div role="img" aria-label="{html.escape(alt, quote=True)}" style="display:flex;flex-wrap:wrap;gap:6px 14px;'
            f'align-items:flex-start">{cells}</div>')


def key_html(names) -> str:
    """The key of the kinds: their mark in their group colour, and their name."""
    return ('<div style="font-size:0.9rem;color:var(--ea-ink-2);margin-top:4px">' + " &nbsp; ".join(
        f'<span style="color:var(--ea-group-{k + 1})">{KIND_MARKS[k]}</span> {html.escape(name)}'
        for k, name in enumerate(names)) + "</div>")


def steps_picture(words: dict) -> str:
    """What the course's FCN (stride 8, seed 0) makes of the scene of the worked example: the scene, the class with the
    highest score at every place of its 16 by 16 map, the class map after the transposed convolution, and the true
    class map."""
    r = TRAINED["s8"]
    panels = [(photo_svg(SEED, words["scene"]), 150), (map_svg(unpack(r["coarse"], 16), words["coarse"]), 150),
              (map_svg(unpack(r["map"], scenes.SIZE), words["up"]), 150), (map_svg(class_map(example()), words["truth"]), 150)]
    return side_by_side(panels, words["alt"]) + key_html(words["kinds"])


def _centre_of(shape) -> tuple[int, int]:
    rows, cols = np.nonzero(shape.mask)
    return int(round(rows.mean())), int(round(cols.mean()))


def stride_frame(stride: int, words: dict) -> tuple[str, dict]:
    """For the slider: the scene of the worked example with one place of the last map (stride by stride pixels) and
    the part of the scene that place sees (its receptive field), around the middle of the circle; next to it, the
    class map that the course's FCN with that stride (seed 0) gives the scene."""
    r = TRAINED[BY_STRIDE[stride]]
    field = receptive_field(stride)
    circle = next(sh for sh in example().shapes if sh.kind == "circle")
    cr, cc = _centre_of(circle)
    top, left = (cr // stride) * stride, (cc // stride) * stride          # the place of the last map that holds the middle
    mid_r, mid_c = top + stride / 2, left + stride / 2
    squares = [(top, left, stride, False), (mid_r - field / 2, mid_c - field / 2, field, True)]
    photo = photo_svg(SEED, words["sees"], 200, squares)
    pred = map_svg(unpack(r["map"], scenes.SIZE), words["title"].format(stride=stride), 200)
    stop = {"places": scenes.SIZE // stride, "field": field,
            "ious": [_span([run[k] for run in r["ious"]]) for k in range(len(KINDS))], "seed0": r["ious"][0]}
    return side_by_side([(photo, 200), (pred, 200)], words["alt"].format(stride=stride, places=stop["places"], field=field)) \
        + key_html(words["kinds"]), stop


def _span(values) -> tuple:
    return min(values), max(values)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def whole(n: int, lang: str) -> str:
    """A whole number with a mark between the thousands: 14,622 in English, 14.622 in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def span_text(values, lang: str, digits: int = 2) -> str:
    low, high = min(values), max(values)
    text = number(low, digits, lang) if number(low, digits, lang) == number(high, digits, lang) else \
        f"{number(low, digits, lang)}–{number(high, digits, lang)}"
    return text


def upsampling_table(words: dict, lang: str, rows=("s8", "s8-random", "s8-bilinear", "s8-nearest")) -> str:
    """The ways of upsampling side by side, as a Markdown table: for each, the IoU of every kind on the test scenes and
    the share of pixels right, from the lowest to the highest over the seeds 0 to 4."""
    lines = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in rows:
        r = TRAINED[variant]
        cells = [words["names"][variant]] + [span_text([run[k] for run in r["ious"]], lang) for k in range(len(KINDS))] \
            + [span_text(r["acc"], lang, 3)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stride_table(words: dict, lang: str, rows=("s4", "s8", "s16", "s32", "s16-long", "s32-long")) -> str:
    """The output strides side by side, as a Markdown table, as upsampling_table."""
    lines = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in rows:
        r = TRAINED[variant]
        stride, _, epochs = VARIANTS[variant]
        cells = [words["name"].format(stride=stride, epochs=epochs), f"{receptive_field(stride)} × {receptive_field(stride)}"] + \
            [span_text([run[k] for run in r["ious"]], lang) for k in range(len(KINDS))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training scenes solid and on the test scenes dashed, as in
    torch-loss-curves. With `at`, a line at that epoch, for rewinding."""
    train_l, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train_l + val if math.isfinite(v)]
    top = max([0.2] + [1.1 * v for v in mine])
    p = chart.Panel((0.5, max(len(val), len(train_l), 1) + 0.5), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    drawn = [(ys, name) for ys, name in ((train_l, labels["train"]), (val, labels["val"])) if ys]
    for ys, dashed in ((train_l, False), (val, True)):
        if ys:
            p.line(list(range(1, len(ys) + 1)), ys, "loss", 2.2, dashed=dashed)
    drawn.sort(key=lambda d: d[0][-1], reverse=True)
    for (ys, name), y in zip(drawn, name_heights([ys[-1] for ys, _ in drawn], top)):
        p.text(len(ys) - 0.3 if len(ys) > 1 else 1, y, name, "loss", "end")   # clear of the line at the last epoch
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=276, height=200)   # 11 px text in a play box on a phone
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def name_heights(ends: list, top: float) -> list:
    """Where the names of one or two lines go at their last epoch, given where they end (the higher end first) on a
    y-axis from 0 to top: over the line that ends higher and under the other. A name is about 0.1 of the axis high.
    Where there is no room over the higher line, its name goes under it too; where there is no room under the lower
    one, its name goes just over it when the higher line is far enough above, and else both names go over both lines,
    the name of the higher line on top."""
    if len(ends) < 2:
        return [e + 0.07 * top if e + 0.07 * top <= 0.95 * top else e - 0.08 * top for e in ends]
    hi, lo = ends
    up, down = hi + 0.07 * top, lo - 0.08 * top
    if up > 0.95 * top:                          # the higher line ends at the top: both names under their lines
        return [hi - 0.08 * top, down if hi - lo > 0.2 * top else hi - 0.23 * top]
    if down >= 0.06 * top:
        return [up, down]
    if hi - lo > 0.2 * top:                      # the lower line's name fits between the two lines
        return [up, lo + 0.07 * top]
    return [hi + 0.22 * top, hi + 0.07 * top]    # both end close together near 0: both names over them


def rewind_picture(maps: list, words: dict) -> str:
    """The test scenes of the rewind, each with the class map the learner's FCN gave it at one moment."""
    panels = [(map_svg(m, words["scene"].format(seed=seed), 120), 120) for seed, m in zip(SHOWN, maps)]
    return side_by_side(panels, words["alt"]) + key_html(words["kinds"])


# --- the exercises: shared --------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    """The same value, with floats within 1 in 10,000 of each other (plus a little for numbers near 0)."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return False
        if math.isnan(want) or math.isnan(got):   # 0 / 0: the same only when both are
            return math.isnan(want) and math.isnan(got)
        return abs(got - want) <= 1e-4 * abs(want) + 1e-4
    if isinstance(want, int) and not isinstance(want, bool) and isinstance(got, float):
        return got == want
    return got == want


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


# --- exercise 1: class_map --------------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\ndef class_map(s):\n"
           "    # s: a scene of the course, with s.image (rows, columns, 3) and s.shapes\n"
           "    # give back an array as large as the scene: 0 for background, and\n"
           "    # shape.label + 1 on the pixels of shape.mask\n")
_C_START = _C_HEAD + "    ...\n"
_C_ZEROS = "    classes = np.zeros(s.image.shape[:2], dtype=int)\n"
_C_LOOP = "    for shape in s.shapes:\n"
_C_SET = "        classes[shape.mask] = shape.label + 1\n"
_C_RETURN = "    return classes\n"


def _c(zeros=_C_ZEROS, loop=_C_LOOP, put=_C_SET, ret=_C_RETURN) -> str:
    return _C_HEAD + zeros + loop + put + ret


_C = _c()
_C_SCENARIOS = [
    # the scene of the worked example: the pixels of every class
    "from everything_ai.scenes import scene\n\nclasses = class_map(scene(18))\n[int((classes == k).sum()) for k in range(4)]",
    # three more scenes: the class of some pixels in the middle of each
    "from everything_ai.scenes import scene\n\n[[int(v) for v in class_map(scene(seed))[64, 40:90:5]] for seed in (3, 23, 9)]",
    # scenes of other sizes
    "from everything_ai.scenes import scene\n\n"
    "[[int((class_map(scene(7, size=size)) == k).sum()) for k in range(4)] for size in (96, 192)]",
]
_C_WRONG = {
    "label": _c(put="        classes[shape.mask] = shape.label\n"),
    "box": _c(put="        top, left, bottom, right = shape.box\n        classes[top:bottom + 1, left:right + 1] = shape.label + 1\n"),
    "one": _c(put="        classes[shape.mask] = 1\n"),
}


def _mask_too_large(e: BaseException) -> bool:
    return isinstance(e, IndexError) and "boolean index did not match" in str(e)


def _kind_text(e: BaseException) -> bool:
    return isinstance(e, ValueError) and re.search(r"invalid literal for int|could not convert string", str(e)) is not None


_C_MISTAKES = [
    Mistake("size", error=_mask_too_large,
            en="Python stops when your function puts a mask into the array: the check also tries scenes of 96 and 192 pixels, and your array has another size than the scene. Make it as large as the scene's picture: `s.image.shape[:2]` gives its rows and columns.",
            nl="Python stopt als je functie een mask in de array zet: de check probeert ook scènes van 96 en 192 pixels, en je array heeft een andere grootte dan de scène. Maak hem zo groot als het beeld van de scène: `s.image.shape[:2]` geeft de rijen en kolommen."),
    Mistake("kind", error=_kind_text,
            en="Python stops when your function puts the kind into the array: `shape.kind` is a word, such as `'circle'`, and the array holds numbers. Use the number of the kind, `shape.label`, plus 1.",
            nl="Python stopt als je functie de soort in de array zet: `shape.kind` is een woord, zoals `'circle'`, en de array bevat getallen. Gebruik het nummer van de soort, `shape.label`, plus 1."),
    Mistake("label", wrong=_C_WRONG["label"],
            en=_S_EN + ": the circles get 0, the class of background. `shape.label` is 0 for a circle, 1 for a square and 2 for a triangle; the classes are one higher, so that 0 is left for background: `shape.label + 1`.",
            nl=_S_NL + ": de cirkels krijgen 0, de klasse van achtergrond. `shape.label` is 0 voor een cirkel, 1 voor een vierkant en 2 voor een driehoek; de klassen zijn één hoger, zodat 0 overblijft voor achtergrond: `shape.label + 1`."),
    Mistake("box", wrong=_C_WRONG["box"],
            en=_S_EN + ": you fill the box of every shape, and the corners of a box are background. `shape.mask` says exactly which pixels are the shape: `classes[shape.mask] = ...`.",
            nl=_S_NL + ": je vult de box van elke vorm, en de hoeken van een box zijn achtergrond. `shape.mask` zegt precies welke pixels de vorm zijn: `classes[shape.mask] = ...`."),
    Mistake("one", wrong=_C_WRONG["one"],
            en=_S_EN + ": every shape gets class 1, whatever its kind. The class says the kind: `shape.label + 1`.",
            nl=_S_NL + ": elke vorm krijgt klasse 1, welke soort hij ook is. De klasse zegt de soort: `shape.label + 1`."),
]

# --- exercise 2: transposed -------------------------------------------------------------------------------------------

_T_HEAD = ("import torch\n\n\ndef transposed(x, kernel, stride):\n"
           "    # x: a tensor of shape (H, W); kernel: a tensor of shape (k, k)\n"
           "    # every value x[i, j] paints kernel times that value onto the output, with its\n"
           "    # top left at row i * stride and column j * stride; where copies overlap, they add up\n"
           "    # give back the output, of shape ((H - 1) * stride + k, (W - 1) * stride + k)\n")
_T_START = _T_HEAD + "    ...\n"
_T_K = "    k = len(kernel)\n"
_T_ZEROS = "    out = torch.zeros((len(x) - 1) * stride + k, (len(x[0]) - 1) * stride + k)\n"
_T_LOOPS = "    for i in range(len(x)):\n        for j in range(len(x[0])):\n"
_T_ADD = "            out[i * stride:i * stride + k, j * stride:j * stride + k] += x[i, j] * kernel\n"
_T_RETURN = "    return out\n"


def _t(k=_T_K, zeros=_T_ZEROS, loops=_T_LOOPS, add=_T_ADD, ret=_T_RETURN) -> str:
    return _T_HEAD + k + zeros + loops + add + ret


_T = _t()
_T_SCENARIOS = [
    # the picture of the worked example's row in two directions, with the bilinear kernel of 4 by 4
    "import torch\n\nline = torch.tensor([0.25, 0.75, 0.75, 0.25])\nkernel = line.reshape(4, 1) * line.reshape(1, 4)\n"
    "x = torch.tensor([[0.0, 100.0], [100.0, 200.0]])\ntransposed(x, kernel, 2).tolist()",
    # nearest neighbor: a kernel of ones as large as the stride
    "import torch\n\nx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n"
    "transposed(x, torch.tensor([[1.0, 1.0], [1.0, 1.0]]), 2).tolist()",
    # random numbers, more columns than rows, and strides below, at and above the kernel's size
    "import torch\n\ntorch.manual_seed(1)\nx = torch.randn(3, 5)\nkernel = torch.randn(3, 3)\n"
    "[[round(v, 4) for v in transposed(x, kernel, s).reshape(-1).tolist()] for s in (1, 3, 4)]",
    # the same as PyTorch's nn.ConvTranspose2d
    "import torch\nfrom torch import nn\n\ntorch.manual_seed(2)\nup = nn.ConvTranspose2d(1, 1, 5, stride=3, bias=False)\n"
    "x = torch.randn(4, 6)\nwith torch.no_grad():\n    theirs = up(x.reshape(1, 1, 4, 6))[0, 0]\n"
    "    mine = transposed(x, up.weight[0, 0], 3)\nbool((mine - theirs).abs().max() < 1e-4)",
]
_T_WRONG = {
    "overwrite": _t(add="            out[i * stride:i * stride + k, j * stride:j * stride + k] = x[i, j] * kernel\n"),
    "unscaled": _t(add="            out[i * stride:i * stride + k, j * stride:j * stride + k] += kernel\n"),
}


def _does_not_fit(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and re.search(r"must match the (existing )?size|size of tensor a \(\d+\) must match", str(e)) is not None


_T_MISTAKES = [
    Mistake("fit", error=_does_not_fit,
            en="Python stops when your function paints a copy of the kernel: the copy does not fit where your code puts it. Does every copy start at row `i * stride` and column `j * stride`, and is the output `(H - 1) * stride + k` rows high and `(W - 1) * stride + k` columns wide?",
            nl="Python stopt als je functie een kopie van de kernel schildert: de kopie past niet waar je code hem neerzet. Begint elke kopie op rij `i * stride` en kolom `j * stride`, en is de output `(H - 1) * stride + k` rijen hoog en `(W - 1) * stride + k` kolommen breed?"),
    Mistake("overwrite", wrong=_T_WRONG["overwrite"],
            en=_S_EN + ": where two copies overlap, the second takes the place of the first. They add up: `+=`, not `=`. That is what blends two neighbors with the bilinear kernel.",
            nl=_S_NL + ": waar twee kopieën overlappen, neemt de tweede de plaats in van de eerste. Ze tellen op: `+=`, niet `=`. Dat is wat twee buren mengt met de bilineaire kernel."),
    Mistake("unscaled", wrong=_T_WRONG["unscaled"],
            en=_S_EN + ": every value paints the same copy of the kernel. The copy is the kernel times that value: `x[i, j] * kernel`.",
            nl=_S_NL + ": elke waarde schildert dezelfde kopie van de kernel. De kopie is de kernel maal die waarde: `x[i, j] * kernel`."),
]

# --- exercise 3: upsampler ---------------------------------------------------------------------------------------------

_U_HEAD = ("import torch\nfrom torch import nn\n\n\ndef bilinear_kernel(k):\n"
           "    # a k by k kernel: 1 in the middle, falling off in a straight line to the edges\n"
           "    line = torch.tensor([1 - abs(i - (k - 1) / 2) / (k // 2) for i in range(k)])\n"
           "    return line.reshape(k, 1) * line.reshape(1, k)\n\n\n"
           "def upsampler(classes, stride):\n"
           "    # give back an nn.ConvTranspose2d that makes `classes` maps stride times larger:\n"
           "    # kernel 2 * stride, padding stride // 2, no bias; started as bilinear upsampling:\n"
           "    # map c in gives map c out with bilinear_kernel(2 * stride), the other pairs 0\n")
_U_START = _U_HEAD + "    ...\n"
_U_MAKE = "    up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride // 2, bias=False)\n"
_U_ZERO = "    nn.init.zeros_(up.weight)\n"
_U_SET = "    with torch.no_grad():\n        for c in range(classes):\n            up.weight[c, c] = bilinear_kernel(2 * stride)\n"
_U_RETURN = "    return up\n"


def _u(make=_U_MAKE, zero=_U_ZERO, put=_U_SET, ret=_U_RETURN) -> str:
    return _U_HEAD + make + zero + put + ret


_U = _u()
_U_SCENARIOS = [
    # the course's FCN: 4 maps of 16 by 16, 8 times larger; with 1 everywhere for class 1, 2 for class 3 and 0 for the
    # others, class 1 stays 1 inside, class 3 stays 2, and the others stay 0
    "import torch\n\ntorch.manual_seed(0)\nup = upsampler(4, 8)\nx = torch.zeros(1, 4, 16, 16)\nx[0, 1] = 1.0\nx[0, 3] = 2.0\n"
    "with torch.no_grad():\n    y = up(x)\ninside = y[0, :, 8:120, 8:120]\n"
    "[list(y.shape)] + [[round(float(inside[c].min()), 4), round(float(inside[c].max()), 4)] for c in range(4)]",
    # random maps, other numbers of classes and other strides
    "import torch\n\ntorch.manual_seed(1)\nx = torch.randn(2, 3, 5, 7)\nwith torch.no_grad():\n"
    "    ys = [upsampler(3, s)(x) for s in (2, 4)]\n[[list(y.shape), round(float(y.abs().sum()), 2)] for y in ys]",
    # it learns: its weights take part in training, and it has no bias
    "import torch\n\ntorch.manual_seed(0)\nup = upsampler(4, 8)\n[up.weight.requires_grad, up.bias is None, round(float(up.weight[0, 1].abs().sum()), 4), "
    "round(float(up.weight[2, 2].sum()), 4)]",
]
_U_WRONG = {
    "padding": _u(make="    up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride, bias=False)\n"),
    "no-padding": _u(make="    up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, bias=False)\n"),
    "all-pairs": _u(put="    with torch.no_grad():\n        for c in range(classes):\n            for d in range(classes):\n"
                        "                up.weight[c, d] = bilinear_kernel(2 * stride)\n"),
    "no-zeros": _u(zero=""),
    "not-started": _u(zero="", put=""),
    "fixed": _u(ret="    up.weight.requires_grad_(False)\n    return up\n"),
    "bias": _u(make="    up = nn.ConvTranspose2d(classes, classes, 2 * stride, stride=stride, padding=stride // 2)\n"),
}


def _leaf_in_place(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e)


def _channels_fixed(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None


_U_MISTAKES = [
    Mistake("fixed-numbers", error=_channels_fixed,
            en="Python stops when the check gives your transposed convolution 3 maps: it was made for another number of maps. Make it for `classes` maps in and out, and with `stride` and `2 * stride`, not with the 4 and the 8 of the course's FCN.",
            nl="Python stopt als de check je transposed convolution 3 maps geeft: hij is gemaakt voor een ander aantal maps. Maak hem voor `classes` maps erin en eruit, en met `stride` en `2 * stride`, niet met de 4 en de 8 van het FCN van de course."),
    Mistake("no-grad", error=_leaf_in_place,
            en="Python stops when your function sets the kernel: PyTorch does not let you change a weight that training follows, unless you say it is not part of training. Set the kernels inside `with torch.no_grad():`.",
            nl="Python stopt als je functie de kernel zet: PyTorch laat je geen weight veranderen die het trainen volgt, tenzij je zegt dat het geen deel van het trainen is. Zet de kernels binnen `with torch.no_grad():`."),
    Mistake("padding", wrong=_U_WRONG["padding"],
            en=_S_EN + ": the maps come out too small. A transposed convolution gives (H − 1) · stride − 2 · padding + k; with k = 2 · stride, the padding that gives H · stride is stride // 2.",
            nl=_S_NL + ": de maps komen te klein uit. Een transposed convolution geeft (H − 1) · stride − 2 · padding + k; met k = 2 · stride is de padding die H · stride geeft stride // 2."),
    Mistake("no-padding", wrong=_U_WRONG["no-padding"],
            en=_S_EN + ": without padding, the maps come out too large: (H − 1) · stride + k, with k = 2 · stride. Padding cuts that many pixels off every side: `padding=stride // 2` gives H · stride.",
            nl=_S_NL + ": zonder padding komen de maps te groot uit: (H − 1) · stride + k, met k = 2 · stride. Padding snijdt zoveel pixels van elke kant af: `padding=stride // 2` geeft H · stride."),
    Mistake("all-pairs", wrong=_U_WRONG["all-pairs"],
            en=_S_EN + ": every map out gets every map in, so the scores of the classes are added together. Map c in should give only map c out: set `up.weight[c, c]`, and leave the other pairs 0.",
            nl=_S_NL + ": elke map eruit krijgt elke map erin, dus de scores van de klassen worden bij elkaar opgeteld. Map c erin hoort alleen map c eruit te geven: zet `up.weight[c, c]`, en laat de andere paren 0."),
    Mistake("no-zeros", wrong=_U_WRONG["no-zeros"],
            en=_S_EN + ": the pairs of a map in and another map out keep the random weights PyTorch starts them with, so every class gets a little of the scores of the others. Set all weights to 0 first, with `nn.init.zeros_(up.weight)`, and then the bilinear kernel for every pair of map c in and map c out.",
            nl=_S_NL + ": de paren van een map erin en een andere map eruit houden de willekeurige weights waarmee PyTorch ze laat beginnen, dus elke klasse krijgt een beetje van de scores van de andere. Zet eerst alle weights op 0, met `nn.init.zeros_(up.weight)`, en dan de bilineaire kernel voor elk paar van map c erin en map c eruit."),
    Mistake("not-started", wrong=_U_WRONG["not-started"],
            en=_S_EN + ": your transposed convolution starts with the random weights of PyTorch. The course's FCN starts it as bilinear upsampling: first all weights 0 with `nn.init.zeros_`, then `bilinear_kernel(2 * stride)` for every pair of map c in and map c out.",
            nl=_S_NL + ": je transposed convolution begint met de willekeurige weights van PyTorch. Het FCN van de course laat hem als bilineaire upsampling beginnen: eerst alle weights 0 met `nn.init.zeros_`, dan `bilinear_kernel(2 * stride)` voor elk paar van map c erin en map c eruit."),
    Mistake("fixed", wrong=_U_WRONG["fixed"],
            en=_S_EN + ": your weights do not take part in training, so the network can never learn how to upsample. The course's FCN only starts as bilinear upsampling, and lets training change it, which did better on the course's scenes than bilinear upsampling that stays fixed: leave `requires_grad` as it is.",
            nl=_S_NL + ": je weights doen niet mee aan het trainen, dus het netwerk kan nooit leren hoe het moet upsamplen. Het FCN van de course begint alleen als bilineaire upsampling, en laat het trainen hem veranderen, wat op de scènes van de course beter werkte dan vaste bilineaire upsampling: laat `requires_grad` zoals het is."),
    Mistake("bias", wrong=_U_WRONG["bias"],
            en=_S_EN + ": your transposed convolution has a bias, which PyTorch starts at random numbers, so every map gets a little added that bilinear upsampling does not add. Make it without: `bias=False`.",
            nl=_S_NL + ": je transposed convolution heeft een bias, die PyTorch met willekeurige getallen laat beginnen, dus elke map krijgt iets erbij dat bilineaire upsampling niet toevoegt. Maak hem zonder: `bias=False`."),
]

# --- exercise 4: kind_iou ---------------------------------------------------------------------------------------------

_I_HEAD = ("def kind_iou(pred, classes, kind):\n"
           "    # pred: the class the network gives every pixel; classes: the true class of every pixel;\n"
           "    # arrays or tensors of the same shape, with 0 for background and 1 to 3 for a kind\n"
           "    # give back the IoU of the pixels of `kind`: the pixels that are that kind in both,\n"
           "    # divided by the pixels that are that kind in either\n")
_I_START = _I_HEAD + "    ...\n"
_I_P = "    p = pred == kind\n"
_I_T = "    t = classes == kind\n"
_I_RETURN = "    return (p & t).sum() / (p | t).sum()\n"


def _i(p=_I_P, t=_I_T, ret=_I_RETURN) -> str:
    return _I_HEAD + p + t + ret


_I = _i()
_I_SCENE = ("import numpy as np\nfrom everything_ai.scenes import scene\n\ns = scene(18)\nclasses = np.zeros((128, 128), dtype=int)\n"
            "for shape in s.shapes:\n    classes[shape.mask] = shape.label + 1\n")
_I_SCENARIOS = [
    # the scene of the worked example, and a prediction that is every pixel 3 pixels to the right of where it should be
    _I_SCENE + "pred = np.zeros((128, 128), dtype=int)\npred[:, 3:] = classes[:, :-3]\n"
    "[round(float(kind_iou(pred, classes, k)), 4) for k in (1, 2, 3)]",
    # a network that says background everywhere
    _I_SCENE + "pred = np.zeros((128, 128), dtype=int)\n[round(float(kind_iou(pred, classes, k)), 4) for k in range(4)]",
    # random classes in tensors, of two scenes of 6 by 6 pixels
    "import torch\n\ntorch.manual_seed(2)\npred = torch.randint(0, 4, (2, 6, 6))\nclasses = torch.randint(0, 4, (2, 6, 6))\n"
    "[round(float(kind_iou(pred, classes, k)), 4) for k in range(4)]",
]
_I_WRONG = {
    "found": _i(ret="    return (p & t).sum() / t.sum()\n"),
    "right": _i(ret="    return (p & t).sum() / p.sum()\n"),
    "both": _i(ret="    return (p & t).sum() / (p.sum() + t.sum())\n"),
    "dice": _i(ret="    return 2 * (p & t).sum() / (p.sum() + t.sum())\n"),
    "not-divided": _i(ret="    return (p & t).sum()\n"),
}
_I_MISTAKES = [
    Mistake("found", wrong=_I_WRONG["found"],
            en=_S_EN + ": you divide by the pixels that are that kind in the class map: the share of them the network finds. A network that says that kind everywhere would get 1. The IoU divides by the union: `(p | t).sum()`.",
            nl=_S_NL + ": je deelt door de pixels die die soort zijn in de class map: het deel ervan dat het netwerk vindt. Een netwerk dat overal die soort zegt, zou 1 krijgen. De IoU deelt door de vereniging: `(p | t).sum()`."),
    Mistake("right", wrong=_I_WRONG["right"],
            en=_S_EN + ": you divide by the pixels the network calls that kind: the share of them that is right. A network that calls only one pixel that kind, rightly, would get 1. The IoU divides by the union: `(p | t).sum()`.",
            nl=_S_NL + ": je deelt door de pixels die het netwerk die soort noemt: het deel daarvan dat klopt. Een netwerk dat maar één pixel die soort noemt, terecht, zou 1 krijgen. De IoU deelt door de vereniging: `(p | t).sum()`."),
    Mistake("both", wrong=_I_WRONG["both"],
            en=_S_EN + ": `p.sum() + t.sum()` counts the overlap twice. The union counts every pixel once: `(p | t).sum()`, as for the masks in [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en).",
            nl=_S_NL + ": `p.sum() + t.sum()` telt de overlap twee keer. De vereniging telt elke pixel één keer: `(p | t).sum()`, zoals bij de masks in [Hoeveel twee boxes overlappen: IoU](/course/open/cv-iou?lang=nl)."),
    Mistake("dice", wrong=_I_WRONG["dice"],
            en=_S_EN + ": that is another measure, the Dice score, which is higher than the IoU. The IoU is the overlap divided by the union: `(p & t).sum() / (p | t).sum()`.",
            nl=_S_NL + ": dat is een andere maat, de Dice-score, die hoger is dan de IoU. De IoU is de overlap gedeeld door de vereniging: `(p & t).sum() / (p | t).sum()`."),
    Mistake("not-divided", wrong=_I_WRONG["not-divided"],
            en=_S_EN + ": that is the overlap, a number of pixels. The IoU divides it by the union, `(p | t).sum()`, so that it runs from 0 to 1 for a shape of any size.",
            nl=_S_NL + ": dat is de overlap, een aantal pixels. De IoU deelt die door de vereniging, `(p | t).sum()`, zodat hij van 0 tot 1 loopt voor een vorm van elke grootte."),
]


EXERCISES = {
    "class-map": ScenarioBox(
        name="class-map",
        picture=yours.image(call=0, result="classes"),
        start=_C_START,
        solution=_C,
        scenarios=_C_SCENARIOS,
        compare=_close,
        save_as=("cnn", "classes"),
        first=[_DOTS],
        mistakes=_C_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: an array of zeros as large as the scene, then for every shape its class on the pixels of its mask, and the array back.",
                      "nl": "Vervang de `...` door je code: een array met nullen zo groot als de scène, dan voor elke vorm zijn klasse op de pixels van zijn mask, en de array terug."},
            "wrong": {"en": _S_EN + ". Does every pixel start at 0, and does every shape put `shape.label + 1` on the pixels of `shape.mask`?",
                      "nl": _S_NL + ". Begint elke pixel bij 0, en zet elke vorm `shape.label + 1` op de pixels van `shape.mask`?"},
            "almost": {"en": "Right for the scene of the worked example. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for every scene, and for scenes of every size?",
                       "nl": "Goed voor de scène van het uitgewerkte voorbeeld. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke scène, en voor scènes van elke grootte?"},
            "correct": {"en": "Your `class_map` is right: every pixel has its class, and that is what the FCN learns. It is now in your project, as `{saved}`.",
                        "nl": "Je `class_map` klopt: elke pixel heeft zijn klasse, en dat is wat het FCN leert. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("class_map"),
        },
    ),
    "transposed": ScenarioBox(
        name="transposed",
        picture=yours.arrays(call=0, show=("x", "kernel"), result="transposed(x, kernel, 2)", shade=True),
        start=_T_START,
        solution=_T,
        scenarios=_T_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=_T_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the size of the kernel, an output of zeros of the right size, and for every value of `x` its copy of the kernel, added at its place.",
                      "nl": "Vervang de `...` door je code: de grootte van de kernel, een output van nullen van de goede grootte, en voor elke waarde van `x` zijn kopie van de kernel, opgeteld op zijn plek."},
            "wrong": {"en": _S_EN + ". Does every value paint the kernel times that value, with its top left at `i * stride` and `j * stride`, and do the copies add up?",
                      "nl": _S_NL + ". Schildert elke waarde de kernel maal die waarde, met de linkerbovenhoek op `i * stride` en `j * stride`, en tellen de kopieën op?"},
            "almost": {"en": "Right for the picture of 2 by 2. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any x, any kernel and any stride, also with more columns than rows?",
                       "nl": "Goed voor het beeld van 2 bij 2. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke x, elke kernel en elke stride, ook met meer kolommen dan rijen?"},
            "correct": {"en": "Your `transposed` is right: it gives what PyTorch's `nn.ConvTranspose2d` gives, copy by copy.",
                        "nl": "Je `transposed` klopt: hij geeft wat `nn.ConvTranspose2d` van PyTorch geeft, kopie voor kopie."},
        },
    ),
    "upsampler": ScenarioBox(
        name="upsampler",
        picture=yours.arrays(call=0, result="bilinear_kernel(4)", shade=True),
        start=_U_START,
        solution=_U,
        scenarios=_U_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "upsample"),
        first=[_DOTS],
        mistakes=_U_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: an `nn.ConvTranspose2d` with the right kernel, stride and padding and no bias, all its weights 0, then the bilinear kernel for every class on its own map, and the layer back.",
                      "nl": "Vervang de `...` door je code: een `nn.ConvTranspose2d` met de goede kernel, stride en padding en zonder bias, al zijn weights 0, dan de bilineaire kernel voor elke klasse op zijn eigen map, en de layer terug."},
            "wrong": {"en": _S_EN + ". Is the kernel 2 · stride and the padding stride // 2, without bias? Are all weights 0, except `up.weight[c, c]`, which is `bilinear_kernel(2 * stride)`?",
                      "nl": _S_NL + ". Is de kernel 2 · stride en de padding stride // 2, zonder bias? Zijn alle weights 0, behalve `up.weight[c, c]`, die `bilinear_kernel(2 * stride)` is?"},
            "almost": {"en": "Right for the course's FCN. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of classes and any stride?",
                       "nl": "Goed voor het FCN van de course. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal klassen en elke stride?"},
            "correct": {"en": "Your `upsampler` is right: it starts as bilinear upsampling, and training can change it. It is now in your project, as `{saved}`.",
                        "nl": "Je `upsampler` klopt: hij begint als bilineaire upsampling, en het trainen kan hem veranderen. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("upsampler"),
        },
    ),
    "kind-iou": ScenarioBox(
        name="kind-iou",
        start=_I_START,
        solution=_I,
        scenarios=_I_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "kind_iou"),
        first=[_DOTS],
        mistakes=_I_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: which pixels the network calls that kind, which pixels are that kind, and the pixels in both divided by the pixels in either.",
                      "nl": "Vervang de `...` door je code: welke pixels het netwerk die soort noemt, welke pixels die soort zijn, en de pixels in allebei gedeeld door de pixels in een van beide."},
            "wrong": {"en": _S_EN + ". Do you compare both with `kind`, and divide the pixels in both, `&`, by the pixels in either, `|`?",
                      "nl": _S_NL + ". Vergelijk je allebei met `kind`, en deel je de pixels in allebei, `&`, door de pixels in een van beide, `|`?"},
            "almost": {"en": "Right for the scene of the worked example. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any kind, and for tensors too?",
                       "nl": "Goed voor de scène van het uitgewerkte voorbeeld. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke soort, en ook voor tensors?"},
            "correct": {"en": "Your `kind_iou` is right: it counts every kind on its own, so background cannot hide a mistake. It is now in your project, as `{saved}`.",
                        "nl": "Je `kind_iou` klopt: hij telt elke soort apart, dus achtergrond kan geen fout verbergen. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("kind_iou"),
        },
    ),
}

# The lesson is done when these are correct. "why-stride" is the open question.
REQUIRED = ["class-map", "transposed", "upsampler", "kind-iou", "why-stride"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("class-map", _C_START, "empty", None),
    ("class-map", _C, "correct", None),
    # the rows and columns one by one, and the shapes in another way
    ("class-map", _c(zeros="    classes = np.zeros((len(s.image), len(s.image[0])), dtype=int)\n",
                     loop="    for i in range(len(s.shapes)):\n", put="        classes[s.shapes[i].mask] = s.shapes[i].label + 1\n"),
     "correct", None),
    # the classes added up: the masks of a scene never overlap
    ("class-map", _c(put="        classes = classes + shape.mask * (shape.label + 1)\n"), "correct", None),
    ("class-map", _c(zeros="    classes = np.zeros((128, 128), dtype=int)\n"), "error", "size"),
    ("class-map", _c(put="        classes[shape.mask] = shape.kind\n"), "error", "kind"),
    ("class-map", _c(ret="    ...\n"), "wrong", "dots-left"),
] + [("class-map", code, "wrong", key) for key, code in _C_WRONG.items()] + [
    ("transposed", _T_START, "empty", None),
    ("transposed", _T, "correct", None),
    # the size from the shapes, and the copy in two lines
    ("transposed", _t(k="    k = kernel.shape[0]\n", zeros="    out = torch.zeros((x.shape[0] - 1) * stride + k, (x.shape[1] - 1) * stride + k)\n",
                      add="            r, c = i * stride, j * stride\n            out[r:r + k, c:c + k] = out[r:r + k, c:c + k] + x[i, j] * kernel\n"),
     "correct", None),
    # the output as large as x times the stride: too small for the copies
    ("transposed", _t(zeros="    out = torch.zeros(len(x) * stride, len(x[0]) * stride)\n"), "error", "fit"),
    # every copy k apart instead of stride
    ("transposed", _t(add="            out[i * k:i * k + k, j * k:j * k + k] += x[i, j] * kernel\n"), "error", "fit"),
    ("transposed", _t(ret="    ...\n"), "wrong", "dots-left"),
] + [("transposed", code, "wrong", key) for key, code in _T_WRONG.items()] + [
    ("upsampler", _U_START, "empty", None),
    ("upsampler", _U, "correct", None),
    # the kernel made once, and the keyword names written out
    ("upsampler", _u(make="    k = 2 * stride\n    up = nn.ConvTranspose2d(classes, classes, kernel_size=k, stride=stride, padding=k // 4, bias=False)\n",
                     put="    kernel = bilinear_kernel(k)\n    with torch.no_grad():\n        for c in range(classes):\n            up.weight[c, c] = kernel\n"),
     "correct", None),
    # the weights set to 0 under no_grad too
    ("upsampler", _u(zero="", put="    with torch.no_grad():\n        up.weight[:] = 0\n        for c in range(classes):\n"
                                  "            up.weight[c, c] = bilinear_kernel(2 * stride)\n"), "correct", None),
    ("upsampler", _u(put="    for c in range(classes):\n        up.weight[c, c] = bilinear_kernel(2 * stride)\n"), "error", "no-grad"),
    # the course's 4 classes and stride 8 as fixed numbers: right for the FCN, an error for 3 classes
    ("upsampler", _u(make="    up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4, bias=False)\n",
                     put="    with torch.no_grad():\n        for c in range(4):\n            up.weight[c, c] = bilinear_kernel(16)\n"), "error", "fixed-numbers"),
    ("upsampler", _u(ret="    ...\n"), "wrong", "dots-left"),
] + [("upsampler", code, "wrong", key) for key, code in _U_WRONG.items()] + [
    ("kind-iou", _I_START, "empty", None),
    ("kind-iou", _I, "correct", None),
    # in one line
    ("kind-iou", _I_HEAD + "    return ((pred == kind) & (classes == kind)).sum() / ((pred == kind) | (classes == kind)).sum()\n", "correct", None),
    # the union as both minus the overlap
    ("kind-iou", _i(ret="    overlap = (p & t).sum()\n    return overlap / (p.sum() + t.sum() - overlap)\n"), "correct", None),
    ("kind-iou", _i(ret="    ...\n"), "wrong", "dots-left"),
] + [("kind-iou", code, "wrong", key) for key, code in _I_WRONG.items()]


# --- the demo: the course's FCN with the learner's class_map, upsampler and kind_iou ----------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


class _Stopped(Exception):
    """An error in one of the learner's functions while the demo runs, with the key of the note that names it."""

    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


def _named(fn, key: str):
    """fn, with every error it raises turned into _Stopped(key), so that the demo names the module that stopped; a
    Timeout goes on as it is."""
    def run(*args):
        try:
            return fn(*args)
        except Timeout:
            raise
        except Exception as e:
            raise _Stopped(key) from e
    return run


def _their_classes(fn, ss) -> torch.Tensor | None:
    """The learner's class maps of the scenes ss as one tensor of whole numbers, or None when one is not a map as
    large as its scene with classes 0 to 3."""
    maps = []
    for s in ss:
        m = torch.as_tensor(np.asarray(fn(s)))
        if tuple(m.shape) != s.image.shape[:2] or not bool(((m >= 0) & (m < CLASSES)).all()) or bool((m != m.round()).any()):
            return None
        maps.append(m.long())
    return torch.stack(maps)


def _demo(show=None):
    """For the demo: None when classes.py, upsample.py or kind_iou.py is not in the project yet; "classes" when the
    learner's class_map stops or does not give a class map for every scene; "upsample" when their upsampler stops or
    does not give a transposed convolution that makes the 4 maps of 16 by 16 as large as the scene and that training
    can change; "kind_iou" when their kind_iou stops or does not give a number between 0 and 1; "slow" when the
    training takes too long; else a dict with the loss after every epoch ("curves"), the class maps of the SHOWN scenes
    after every epoch ("moments"), and the share of pixels right and the IoU of every kind on the test scenes after the
    last epoch. `show` hears the run while it trains."""
    if not (_saved("classes") and _saved("upsample") and _saved("kind_iou")):
        return None
    names = {}
    for module, name, key in (("classes", "class_map", "classes"), ("upsample", "upsampler", "upsample"),
                              ("kind_iou", "kind_iou", "kind_iou")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    ss, X, _ = scene_set("train")
    test_ss, Xt, _ = scene_set("test")
    try:   # the learner's class maps: the targets of every training and test scene
        C = _their_classes(names["class_map"], ss)
        Ct = _their_classes(names["class_map"], test_ss)
    except Exception:
        return "classes"
    if C is None or Ct is None:
        return "classes"
    try:   # the learner's transposed convolution on the 4 maps of 16 by 16 of one scene
        layer = names["upsampler"](CLASSES, STRIDE)
        y = layer(torch.zeros(1, CLASSES, scenes.SIZE // STRIDE, scenes.SIZE // STRIDE))
        if not (isinstance(layer, nn.Module) and tuple(y.shape) == (1, CLASSES, scenes.SIZE, scenes.SIZE)
                and any(p.requires_grad for p in layer.parameters())):
            return "upsample"
    except Exception:
        return "upsample"
    try:   # the learner's IoU of every kind for a class map against itself: 1
        ones = [float(names["kind_iou"](Ct[:4], Ct[:4], k)) for k in range(1, CLASSES)]
        if not all(math.isfinite(v) and abs(v - 1) < 1e-6 for v in ones):
            return "kind_iou"
    except Exception:
        return "kind_iou"
    up_of, iou_of = _named(names["upsampler"], "upsample"), _named(names["kind_iou"], "kind_iou")
    torch.manual_seed(0)   # the same start as the course's run
    try:
        model = FCN(STRIDE, "learned", up_of)
    except _Stopped as e:
        return e.key
    shown_at = [TEST.index(seed) for seed in SHOWN]
    moments = []

    def heard(p):
        if "pred" in p:
            moments.append([p["pred"][i].numpy() for i in shown_at])
            p = {k: v for k, v in p.items() if k != "pred"}
        if show is not None:
            try:
                show(p)
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: train(model, (X, C), (Xt, Ct), 0, EPOCHS, heard, iou_of), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "upsample"
    last = history[-1]
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    return {"curves": curves, "moments": moments, "acc": last["acc"], "ious": last["ious"]}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("classes", "upsample", "kind_iou"):
        path = projects.folder() / "cnn" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
