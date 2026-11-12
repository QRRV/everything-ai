# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-vit: patches, a batch of photos cut into square patches, each a row of numbers;
SelfAttention, the learner's CausalSelfAttention of attn-transformer-block without its mask; and ViT, a Vision
Transformer from those and the learner's Block, each saved in the learner's project (work/projects/cnn/patches.py,
self_attention.py and vit.py) when right. All three are checked on made-up photos, so they work without CIFAR-10. The
demo trains the learner's ViT, with 11,738 parameters about as many as the ColorCNN of cv-color-cnn, on the changed
photos of cv-augment-photos, and keeps after every epoch its guesses for one test photo of each kind and how alike the
position of one patch is to those of the others, for rewinding through the run."""
import errno
import functools
import html
import math
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, Dataset, TensorDataset

from everything_ai import chart, cifar, hub, live, netview, projects, yours
from everything_ai.lesson import download_card
from everything_ai.check import ClassBox, Mistake, ScenarioBox, Timeout, run_code, with_time_limit

# The demo uses the learner's ViT of this lesson, with their CausalSelfAttention and Block of attn-transformer-block,
# their crop and jitter of cv-augment-photos, their channel_stats of cv-color-cnn, their Trainer of nn-trainer and their
# count_params of cnn-pytorch; where one is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"), ("cnn", "channels", "cv-color-cnn", "channel-stats"),
             ("cnn", "crop", "cv-augment-photos", "crop"), ("cnn", "color", "cv-augment-photos", "color"),
             ("llm", "attention", "attn-transformer-block", "causal-self-attention"),
             ("llm", "block", "attn-transformer-block", "block"))
projects.builds_on(*BUILDS_ON)

DOWNLOAD_LINE = {
    "en": "Works on every computer. CIFAR-10, 170 MB of photos, is downloaded once from the University of Toronto when you click its button in the lesson, unless you did so in Project: your CNN on color photos; after that, the lesson works without a connection. The exercises work without it too.",
    "nl": "Werkt op elke computer. CIFAR-10, 170 MB aan foto's, wordt één keer gedownload van de University of Toronto als je in de les op de knop ervoor klikt, tenzij je dat al deed in Project: je CNN op kleurenfoto's; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
NAMES = {
    "en": dict(zip(cifar.KINDS, ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"))),
    "nl": dict(zip(cifar.KINDS, ("vliegtuig", "auto", "vogel", "kat", "hert", "hond", "kikker", "paard", "schip", "vrachtwagen"))),
}
N_TRAIN, N_TEST = 6_000, 1_000   # the first photos of each set, as in cv-color-cnn and cv-augment-photos
EPOCHS, LR = 40, 0.01            # the demo, as in cv-augment-photos
D, H, DEPTH, P = 16, 2, 3, 4     # the demo's ViT: 16 numbers per token, 2 heads, 3 blocks, patches of 4 by 4
PARAMS = 11_738                  # of ViT(16, 2, 3, 4): see test_the_idea
CNN_PARAMS = 11_642              # of ColorCNN of cv-color-cnn
PAD = 4                          # the black border of a random crop, as in cv-augment-photos
LOW, HIGH = 0.8, 1.2             # the range of the factors of the color, as in cv-augment-photos
PHOTO = 82                       # the test photo of the picture of the patches and the worked example: a car
SHOWN = tuple(cifar.checked(kind, 1)[0] for kind in cifar.KINDS)   # one checked test photo of each kind, for the rewind
WATCHED = (3, 3)                 # the patch whose position the demo compares with those of the others: row 3, column 3
MOMENTS = (0, 1, 2, 5, 10, 20, 30, 40)   # the epochs to rewind to after the demo (0: before the first), with its last
EVERY = 10                       # the live chart hears the training loss of every 10 batches (live.follow)

# What the course measured, on one thread, with the course's code: ViT(16, 2, 3, 4) of this lesson after
# torch.manual_seed(seed), trained with the Trainer of nn-trainer (a learning rate of 0.01, a warmup of one epoch,
# max_norm 1.0, mini-batches of 64) on the first N_TRAIN training photos, shuffled after torch.manual_seed(seed + 1),
# for the seeds 0 to 4, as cv-augment-photos trained ColorCNN: "all" from the Dataset of that lesson's demo (mirror,
# crop and color, anew every time), "none" from the photos as they are. The photos are normalized with the mean and the
# standard deviation of each color of the training photos as they are; the test photos are only normalized. "train"
# and "test" are how many of the training photos as they are and of the first N_TEST test photos are read right after
# the last epoch; "best" the first epoch with the lowest validation loss, rounded to 3 decimals; "last_val" and
# "last_train" the validation and the training loss of the last epoch. "near" and "far" say how alike the learned
# positions of the 64 patches are after the last epoch: the mean cosine similarity of the position of each patch with
# those of its 4 neighbors (up, down, left, right), and with those 5 or more patches away (dist >= 5, counted along the
# straight line between their middles), rounded to 3 decimals. CNN: ColorCNN of cv-color-cnn, from REFERENCE of
# cv-augment-photos ("all" and "none" after 40 epochs). "curves" are seed 0 of the ViT with "all"; "cnn_val" the
# validation loss of ColorCNN with "all", seed 0 (CURVES["val"] of cv-augment-photos). POSITIONS are the 65 positions of
# seed 0 with "all" after the last epoch, rounded to 4 decimals, for the picture of how alike they are.
# tests/test_cv_vit.py measures seed 0 of "all" again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    ('vit', 'all'): {"train": [3458, 3309, 3363, 3243, 3305], "test": [524, 500, 512, 494, 489], "best": [38, 35, 36, 39, 33],
                       "last_val": [1.358, 1.397, 1.347, 1.391, 1.398], "last_train": [1.247, 1.297, 1.263, 1.322, 1.268],
                       "near": [0.541, 0.485, 0.421, 0.505, 0.543], "far": [-0.193, -0.164, -0.104, -0.202, -0.137]},
    ('vit', 'none'): {"train": [4727, 4676, 4677, 4676, 4722], "test": [442, 455, 448, 465, 453], "best": [15, 17, 16, 17, 12],
                       "last_val": [2.016, 2.062, 1.971, 2.005, 1.99], "last_train": [0.622, 0.667, 0.657, 0.655, 0.607],
                       "near": [0.252, 0.249, 0.218, 0.233, 0.334], "far": [-0.042, -0.083, -0.081, -0.013, -0.051]},
    ('cnn', 'all'): {"train": [4015, 3917, 3967, 4024, 3986], "test": [617, 593, 583, 619, 598], "best": [39, 33, 39, 36, 39],
                       "last_val": [1.072, 1.176, 1.167, 1.079, 1.162], "last_train": [1.122, 1.136, 1.122, 1.1, 1.111]},
    ('cnn', 'none'): {"train": [5992, 5960, 6000, 5998, 6000], "test": [498, 460, 502, 545, 511], "best": [6, 4, 4, 5, 3],
                       "last_val": [4.425, 4.118, 4.467, 4.193, 4.478], "last_train": [0.026, 0.065, 0.014, 0.014, 0.01]},
}
CURVES = {
    "train": [2.155, 2.035, 1.947, 1.88, 1.818, 1.78, 1.745, 1.711, 1.675, 1.667, 1.626, 1.613, 1.598, 1.583, 1.543,
              1.518, 1.508, 1.477, 1.454, 1.442, 1.43, 1.424, 1.397, 1.382, 1.37, 1.371, 1.346, 1.329, 1.316, 1.289,
              1.291, 1.282, 1.27, 1.254, 1.248, 1.24, 1.245, 1.237, 1.24, 1.247],
    "val": [2.041, 1.991, 1.926, 1.83, 1.747, 1.736, 1.74, 1.692, 1.643, 1.612, 1.616, 1.564, 1.651, 1.563, 1.532,
            1.485, 1.485, 1.472, 1.482, 1.456, 1.402, 1.42, 1.405, 1.387, 1.421, 1.399, 1.39, 1.38, 1.372, 1.376, 1.36,
            1.363, 1.371, 1.359, 1.364, 1.36, 1.361, 1.357, 1.358, 1.358],
    "cnn_val": [1.77, 1.626, 1.527, 1.546, 1.475, 1.416, 1.468, 1.457, 1.454, 1.357, 1.371, 1.332, 1.369, 1.338, 1.283,
                1.27, 1.262, 1.256, 1.224, 1.217, 1.21, 1.186, 1.146, 1.151, 1.195, 1.152, 1.148, 1.179, 1.111, 1.097,
                1.112, 1.117, 1.084, 1.088, 1.079, 1.074, 1.077, 1.074, 1.072, 1.072],
}
POSITIONS = [
    [0.6159, -0.3567, -0.3172, 0.0711, 0.494, 0.6153, -1.0811, -1.7212, -0.1335, -0.5674, 0.701, 0.1205, 1.0475, 1.0976, 0.5956, 0.6256],
    [-0.7047, -0.6533, -0.3239, 2.0377, -2.3985, 1.6188, -1.3768, 1.0275, -1.8211, 1.141, 1.6435, 1.2245, -1.4592, 0.9534, -2.1762, -1.2324],
    [-0.9905, -0.7802, -1.8861, 2.2616, -4.1286, 0.0933, 1.9447, -0.4308, 0.1622, 1.8458, -0.1454, 1.4613, -2.4774, -0.2245, -1.057, -1.5824],
    [-2.1031, 0.9194, -0.5687, 0.1921, -1.0073, 0.3639, -0.1052, 1.1336, 0.088, 0.8325, 0.0856, -0.234, -1.4864, 0.1643, -0.5323, -1.7572],
    [-1.0955, 1.2996, 0.0664, 3.0896, -1.3249, 1.1525, -0.8957, 0.1867, -0.7216, 1.5943, -0.4012, -0.3101, -1.8213, 0.8238, -0.4063, -1.086],
    [2.3242, 0.1629, -1.1085, 1.7978, -1.8251, 1.8269, 0.3011, -0.6609, -0.2195, 1.1602, -0.9842, 0.493, -1.5904, 0.3575, 1.6706, -0.5966],
    [-0.2299, 0.8468, 0.0315, 3.2045, -2.6072, 1.2255, -0.4833, 0.1037, 0.1449, 0.4281, -0.2699, -0.7382, -2.3273, -0.8432, 1.3466, -0.6933],
    [-0.6075, -0.491, 0.952, 2.1801, -3.9332, 1.8775, 1.1221, -0.5143, -0.0239, 1.5439, 0.1137, 0.7696, -1.6932, 2.8962, -1.8181, -2.5101],
    [-0.245, 1.8798, 1.8383, 2.8303, -2.1094, 2.2637, -0.0289, 1.9214, -2.1854, 0.9616, -0.0551, -0.4274, -1.1139, -1.0505, -0.7308, -1.2004],
    [1.0971, 0.3343, 1.3314, 1.7417, -2.5915, 1.763, 0.4098, -0.4604, -2.0285, 1.3042, -1.2503, 0.6641, -0.6526, 0.5272, 1.7093, -1.4075],
    [-0.1124, 0.2982, -1.4495, 1.4503, -1.9632, 0.9492, 0.1731, 0.4041, -0.4055, 1.8302, 0.7852, -0.042, -1.2437, 0.2269, -0.2616, -0.5778],
    [-0.1386, 1.2102, -2.5912, 0.1662, -1.2181, 1.4843, 0.2077, -0.1234, -0.7558, 0.2825, -0.9715, -0.202, -0.6129, -0.303, 1.3874, -0.4587],
    [-0.9979, 1.31, -0.8826, -0.3928, -0.5935, 0.6432, -0.8301, 1.3295, -1.2325, 1.0991, -0.0421, -0.4676, -0.7427, 0.5364, 0.2638, -0.2721],
    [-0.1401, 1.2251, -2.6056, 1.3637, -1.3025, 1.1057, 0.254, 0.3746, -0.0105, 1.0069, -0.0012, 0.5493, -0.966, -0.4079, 1.3566, 0.5334],
    [-0.2754, 0.0902, -0.8072, 0.2758, -0.9569, 0.4264, 0.3452, 0.6018, -0.3968, 0.7327, 0.8272, -0.4698, -2.2393, 1.1582, 0.2844, -1.0834],
    [-0.9832, -0.1679, -1.7107, 0.1525, -1.3328, 0.4663, 2.1205, 0.9052, 0.0549, 1.7542, 0.7414, -0.5039, -1.6643, 0.9048, 1.0061, -1.1064],
    [-0.4088, 0.9802, -1.95, 1.2131, -1.1925, 2.3148, 2.5379, -0.2212, -0.4806, -0.5465, 0.3109, 1.0934, -1.8916, -1.3885, 1.4774, -0.8822],
    [-0.2405, -0.8236, -1.8927, 1.9019, -0.9944, 1.4231, 0.2918, -1.9233, 0.3415, 2.2566, 0.5958, 1.1919, -0.4854, -1.4793, -1.496, 1.1315],
    [-1.351, 0.6906, -1.9492, 0.0756, -0.3596, 0.6317, -0.1281, -0.3188, -1.9988, 0.9707, -0.7559, -0.5315, 0.0367, 0.1786, -1.5965, -0.4991],
    [-1.0443, 1.3338, -1.7153, -1.1605, -0.2354, 0.129, 0.7287, 0.4508, -0.6348, 0.3672, -0.6883, -0.517, -1.0057, 1.503, -0.6277, -0.773],
    [0.4761, 1.044, -1.1776, -0.0409, -0.6255, 0.9512, 0.4305, 0.1283, -0.1733, -1.8176, -0.1003, -0.9828, -0.9556, 2.3234, -0.9963, -0.0445],
    [0.8483, 1.9167, -1.2538, -0.135, -0.093, 1.0217, -0.6315, 0.5842, -0.384, -0.625, -0.5532, -1.8464, -0.6235, 1.5597, -0.7975, 0.2148],
    [0.3605, 1.6919, -1.5393, 0.0374, -0.0427, 0.9824, -0.0085, 0.5163, -1.0745, -0.2771, -0.8001, -0.7254, -0.42, 1.3303, -1.1173, 0.2412],
    [-0.0841, -0.3433, -0.9239, 1.1773, -0.6296, 0.9753, -0.0043, 0.4424, -0.7454, 1.053, 1.4207, -0.1556, -0.6302, 1.5245, -0.3114, -0.3763],
    [-0.9259, -0.1672, -1.0777, 1.1342, -1.0993, 0.6339, 1.2597, 0.6964, -1.6358, 1.1848, 1.0968, 1.8401, 0.5386, 0.5665, -0.1019, -0.8832],
    [0.3152, -1.014, -0.3292, 1.8321, 0.553, 0.3173, 0.8985, -1.9886, -0.8406, 0.4847, 0.6401, 1.8129, -1.7666, 0.0476, -1.3087, 0.6313],
    [-1.1484, 1.2199, -0.0953, -0.5314, 0.5743, 0.4213, 0.6113, 1.3158, -0.0484, 1.609, 0.9128, -0.0575, 0.265, 1.1973, -0.0945, 0.4441],
    [-0.2804, 1.5079, -0.6471, -1.0424, 0.766, 0.4647, 0.5363, 0.2033, -0.3326, -1.2987, -0.1072, -1.5018, -1.345, 2.4998, -1.5271, -0.111],
    [1.5656, 2.582, -1.6447, -0.5285, 0.307, 1.0509, 0.5468, 0.296, 0.5864, -1.3681, -0.4291, -1.624, -0.2592, 2.2986, -0.7522, 0.589],
    [1.1735, 3.2518, -1.6528, 0.293, 0.7401, 0.595, 0.1832, 0.2483, -0.0348, 0.1957, -0.6909, -2.5141, -0.1943, 2.0137, -1.5711, 0.9288],
    [1.1787, 2.7698, -1.4685, -1.5754, 0.3074, -0.0902, 0.339, 0.388, 0.239, -0.2517, 0.4429, 0.472, -1.1044, 0.4517, -0.9368, -0.5858],
    [0.3264, 0.6873, 0.1319, -0.1646, -0.5891, 0.0942, -0.4234, 1.6498, 0.0364, 1.0534, 0.2774, -0.8884, 0.9667, 1.189, 0.2707, 0.3019],
    [0.6091, -0.5062, -0.5007, 0.5117, 0.0308, -0.3046, 1.9785, -1.3731, -1.3315, 0.1187, 1.7301, 1.3114, 0.7749, 0.068, -0.3643, 0.0512],
    [-1.1664, -0.9463, 0.0843, 1.1706, 0.2549, -0.913, -1.3091, -0.1891, -0.2851, 0.7331, 0.5537, 1.1003, 0.958, 0.0109, -0.166, 0.546],
    [-0.7424, 0.4956, -0.1378, -1.0874, 0.8713, -0.7905, -1.029, 0.1191, -0.3075, 0.6289, -0.7133, -0.9671, 0.9751, 0.2402, -0.3694, -0.4013],
    [0.3754, 1.9715, -1.3848, -0.9308, 0.4222, -1.3436, -0.4108, -1.0474, 0.3325, 0.1128, -0.8306, -0.9471, -0.4678, 2.3086, -0.8888, -0.9461],
    [0.8497, 3.7368, -0.7794, -0.5885, 0.9045, -0.7421, -1.5617, -0.8032, 1.1148, -0.8516, 0.5419, -2.1872, -0.4713, 1.5505, -0.4258, 0.1205],
    [1.4144, 3.1917, -0.9072, -1.081, 0.7472, 0.1309, -0.6348, 0.3694, 0.8824, -0.9822, -0.3255, -0.8434, 0.3333, 0.4881, -0.5431, 0.3656],
    [-0.2591, 2.5195, -0.0378, -1.2305, 0.6068, -0.4528, -0.9033, 0.1932, 0.101, 0.4213, 0.0774, -1.4333, -0.7635, 0.8369, -1.0296, 0.2954],
    [0.1038, 0.2488, 0.4088, -0.9808, 0.2799, 0.1938, -1.0987, 0.6879, -0.3547, 0.2248, 0.866, -0.3299, 0.6798, 1.0531, 1.4087, 0.5031],
    [-0.7178, -0.4055, 1.2732, -0.0927, -0.1185, -0.5448, -0.1541, 0.9344, -0.4152, 0.6911, 1.453, 1.9389, 1.0672, 0.2993, 0.7245, -0.3198],
    [-0.6682, -1.4881, 1.6377, 0.1623, 1.6727, -0.759, -0.2525, 0.5637, 0.0445, 0.5097, -0.0624, 0.5733, 0.4174, 0.0302, -0.9568, 0.0318],
    [-1.2214, -0.195, 0.8963, -1.1535, 1.9244, -0.8974, -1.4961, -0.3333, 0.3715, 0.4527, -0.4636, -0.4042, 0.7283, 0.8403, 0.3113, -0.1172],
    [1.5199, 1.7087, 0.3164, -0.5263, 1.5183, -0.4474, -1.1397, -0.0288, 0.6853, -1.4838, -0.2113, -0.2371, 0.6285, -0.0248, -0.2826, 0.3428],
    [-0.5357, 1.994, 1.1391, -1.2195, 1.918, -0.5483, -1.1592, -0.243, 0.9932, -0.1126, -0.3268, -0.2199, -0.2125, 0.7623, -0.5293, 0.4127],
    [1.0837, 1.6811, -0.3323, -0.2478, 1.3638, -1.6813, -2.3003, -0.2791, 0.7162, -0.6611, 0.4975, -0.7208, -0.4161, 0.0096, 1.2653, 0.7601],
    [-0.2554, 0.7144, 0.0416, -1.172, 1.575, -1.962, -1.5636, -0.1816, 0.6924, 1.0512, 0.2368, -0.4062, -0.7948, 0.599, -0.0687, 0.245],
    [-0.0851, 0.2205, -0.9409, -0.7835, 2.0731, -0.9354, -1.0221, -0.8627, 0.1449, -0.4989, -0.5192, 0.2463, 1.3689, -0.5722, 2.0074, 0.2556],
    [-2.5877, -2.0787, 1.0362, -0.0675, 0.6622, -0.3447, -0.488, -0.5187, -1.0342, -0.0021, 0.885, 1.0364, 0.2071, -0.4391, 0.6429, 0.1756],
    [0.4878, -2.2388, 0.4583, -0.8838, 0.0082, -1.2594, -0.226, -0.0048, -0.4921, 1.881, 0.6024, 0.3791, 1.4947, -1.8496, 0.5762, 0.098],
    [-1.1202, -2.2755, 0.6949, -1.257, 2.5022, -0.7379, -0.7863, -0.4877, 0.9332, 0.0014, 0.0314, -0.5431, 1.8441, -0.1455, -0.1297, 1.0233],
    [0.2136, -1.6626, -1.2359, -1.8985, 1.1553, -1.595, -0.4578, -0.269, 0.685, -0.9843, -0.0734, -0.6847, 1.0531, -0.4723, 0.8293, 0.3443],
    [0.7301, -0.3865, 0.1073, -1.4776, 1.6238, -1.1586, -1.5979, -0.6478, 0.1582, -1.478, 0.5193, -0.9761, -0.2284, -0.6602, 0.4959, 0.1643],
    [0.9494, 0.2902, 0.1506, -0.9284, 2.0039, -2.1259, -1.839, -0.3524, 0.5313, 0.0722, 0.2395, -0.083, -1.0743, -0.2844, 1.6396, -0.2559],
    [-0.4083, -0.2767, 0.521, -1.7511, 1.9798, -0.9377, -1.8341, 0.2091, 0.0499, 0.6458, -1.4349, -1.194, -0.3926, 0.0868, 0.3085, -0.3855],
    [-0.3032, -2.1042, 0.7429, -1.2541, 3.0516, -0.8218, 0.8679, -0.8188, 0.4977, -0.1317, -0.0816, 0.3072, 0.3282, 0.0353, 1.6544, 1.0105],
    [-0.3633, -2.0311, 1.6029, -0.4438, 1.5223, 0.171, 0.8525, -0.0709, -0.5915, -1.9205, -1.0589, 0.2446, -0.0954, -1.3554, 1.7996, 0.7041],
    [-0.607, -1.3947, 0.9014, -0.3118, 1.13, -1.5358, 1.388, -1.1163, 0.6746, 2.1023, 0.9335, 1.5178, 3.2217, -2.5709, -0.7684, 0.4913],
    [-0.1645, -2.8381, -0.3811, -2.0695, 0.9876, -1.7313, 1.0005, -0.1434, -0.3956, 0.3683, 1.3237, 0.7628, 2.9993, -2.9763, 0.6178, 0.6876],
    [-1.8478, -1.7418, 1.5281, -1.3397, 2.2817, 0.6405, -1.0521, -0.8879, -0.1065, 0.3112, -0.3093, -0.8692, 2.0514, -2.22, 0.0625, 1.3163],
    [0.2544, -0.8428, 0.3363, -1.8986, 1.3889, -1.309, -0.6163, -0.6433, 0.2509, 0.4115, -0.3906, 0.6715, 1.1318, -0.747, 1.3728, -0.3599],
    [-0.8384, -1.2528, 0.9514, -1.8597, 0.6391, -0.8658, -1.2048, 0.8848, 0.138, 0.3627, 0.5111, 0.825, 2.2276, -1.5658, 1.7624, 0.4108],
    [0.0333, -1.0011, 0.1392, 0.4278, 3.2645, -0.5461, -0.0787, -0.3385, -0.3036, 1.259, -0.4099, 0.715, 0.8043, -1.5021, -0.0478, 1.2608],
    [-1.1643, -3.0843, 1.6703, -1.364, 2.377, -0.5208, 1.0389, -0.8378, -0.8736, 0.1915, 0.0075, 0.096, 0.6399, -2.8455, 1.8869, 2.0169],
    [-0.9538, -3.3645, 1.2319, 0.0777, 0.7689, -1.9557, 0.6159, -0.6789, -0.7231, -0.6133, -1.4504, 1.1869, 1.4747, -2.7115, 0.6822, 1.8118],
]


def low_high(values) -> tuple:
    return min(values), max(values)


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 5,514 or 5.514."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(right: int, of: int) -> int:
    return round(100 * right / of)


# --- CIFAR-10 on the learner's computer ----------------------------------------------------------------------------

def ready() -> bool:
    return cifar.unpacked()


def cifar_or_stop(button, texts: dict):
    """everything_ai.cifar for the lesson's cells, once CIFAR-10 is downloaded and unpacked. When it is not there
    yet, the cell stops with `texts["ask"]` and the button; a click downloads the archive, with the progress in the
    cell, and unpacks it. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({done} and
    {total} in MB), "unpacking", "no_net", "not_pinned", "damaged", "no_room" and "failed" ({why}): why it did not
    work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            if hub.pinned(*cifar.TAR) is None:
                total = round(hub.PINNED[cifar.TAR]["size"] / 1e6)
                last = [-1]

                def progress(done, size):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
                hub.download(*cifar.TAR, progress=progress)
            mo.output.replace(mo.md(texts["unpacking"]))
            cifar.unpack()
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except cifar.Damaged:
            failed = texts["damaged"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return cifar


@functools.lru_cache(maxsize=1)
def photos():
    """The first N_TRAIN training photos and the first N_TEST test photos of CIFAR-10 as tensors of 0 to 1, of shape
    (N, 3, 32, 32), and their kinds."""
    X, y, X_test, y_test = cifar.load()
    as_floats = lambda a: torch.tensor(a, dtype=torch.float32) / 255
    return (as_floats(X[:N_TRAIN]), torch.tensor(y[:N_TRAIN], dtype=torch.int64),
            as_floats(X_test[:N_TEST]), torch.tensor(y_test[:N_TEST], dtype=torch.int64))


# --- the course's patches, and how alike the positions are ---------------------------------------------------------

def course_patches(x, p: int):
    """Photos of shape (N, C, H, W) as (N, (H // p) · (W // p), C · p · p): every patch a row, row by row of patches,
    each holding its colors one after the other, every color row by row."""
    n, c, h, w = x.shape
    x = x.reshape(n, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(n, (h // p) * (w // p), c * p * p)


def similarity(positions) -> np.ndarray:
    """How alike the learned positions of the patches are: the cosine similarity of every two, of the 64 patches
    (the first row, the class token's, left out), as an array of shape (64, 64)."""
    pos = torch.as_tensor(np.asarray(positions, dtype=np.float32))[1:]
    pos = pos / pos.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (pos @ pos.T).numpy()


def _grid_distance(side: int) -> np.ndarray:
    rows, cols = np.divmod(np.arange(side * side), side)
    return np.sqrt((rows[:, None] - rows[None]) ** 2 + (cols[:, None] - cols[None]) ** 2)


def near_far(positions) -> tuple[float, float]:
    """The mean cosine similarity of the position of every patch with those of its 4 neighbors, and with those 5 or
    more patches away, rounded to 3 decimals (REFERENCE's "near" and "far")."""
    sim = similarity(positions)
    dist = _grid_distance(int(round(math.sqrt(len(sim)))))
    return round(float(sim[dist == 1].mean()), 3), round(float(sim[dist >= 5].mean()), 3)


def watched_row(positions) -> list:
    """How alike the position of the patch WATCHED is to that of every patch, row by row: 64 numbers."""
    side = int(round(math.sqrt(len(positions) - 1)))
    return [round(float(v), 4) for v in similarity(positions)[WATCHED[0] * side + WATCHED[1]]]


# --- the pictures --------------------------------------------------------------------------------------------------

def _map_svg(row: list, side: int, x0: float, y0: float, cell: float, mark: int | None = None) -> str:
    """One map of how alike the position of one patch is to those of the others, as SVG: a square per patch, row by
    row, filled with the color of what a model learns, as strongly as the cosine similarity (nothing at 0 or below);
    a frame around the whole, and around the patch itself when `mark` gives it. Few elements: one path per strength,
    in 8 steps, so that 64 maps stay small."""
    steps: dict = {}
    for k, v in enumerate(row):
        level = min(8, int(round(max(0.0, float(v)) * 8)))
        if level:
            r, c = divmod(k, side)
            steps.setdefault(level, []).append(f"M{x0 + c * cell:.1f} {y0 + r * cell:.1f}h{cell:.1f}v{cell:.1f}h{-cell:.1f}z")
    out = [f'<path d="{"".join(d)}" fill="var(--ea-param)" fill-opacity="{level / 8:.3f}"/>' for level, d in sorted(steps.items())]
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{side * cell:.1f}" height="{side * cell:.1f}" fill="none" '
               f'stroke="var(--ea-line)" stroke-width="1"/>')
    if mark is not None:
        r, c = divmod(mark, side)
        out.append(f'<rect x="{x0 + c * cell:.1f}" y="{y0 + r * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                   f'fill="none" stroke="var(--ea-ink)" stroke-width="1.5"/>')
    return "".join(out)


def watched_picture(row: list, title: str, alt: str) -> str:
    """The map of the patch WATCHED: how alike its position is to that of every patch, as a figure for beside the
    live chart and in the rewind."""
    side = int(round(math.sqrt(len(row))))
    cell = 18
    size = side * cell
    body = (f'<svg viewBox="0 0 {size + 2} {size + 2}" width="{size + 2}" height="{size + 2}" role="img" '
            f'aria-label="{html.escape(alt)}" style="max-width:100%">'
            + _map_svg(row, side, 1, 1, cell, WATCHED[0] * side + WATCHED[1]) + "</svg>")
    return yours.figure(title, body, alt=alt)


def positions_picture(positions, labels: dict) -> str:
    """How alike the learned positions are, for all 64 patches: a small map per patch, at the place of that patch,
    each showing how alike its position is to those of all patches (the course's run, seed 0, after the last epoch),
    as Figure 7 of the paper on Vision Transformers shows them."""
    sim = similarity(positions)
    side = int(round(math.sqrt(len(sim))))
    cell, gap, top = 6, 6, 26
    size = side * (side * cell + gap) - gap
    parts = [f'<text x="0" y="16" font-size="13" font-weight="600" fill="var(--ea-ink)" '
             f'font-family="var(--ea-font-ui)">{html.escape(labels["title"])}</text>']
    for k in range(side * side):
        r, c = divmod(k, side)
        parts.append(_map_svg(sim[k].tolist(), side, c * (side * cell + gap), top + r * (side * cell + gap), cell, k))
    parts.append(f'<text x="0" y="{top + size + 18}" font-size="12" fill="var(--ea-ink-2)" '
                 f'font-family="var(--ea-font-ui)">{html.escape(labels["under"])}</text>')
    return (f'<svg viewBox="0 0 {size:.0f} {top + size + 26:.0f}" width="{size:.0f}" role="img" '
            f'aria-label="{html.escape(labels["alt"])}" style="max-width:100%;height:auto">' + "".join(parts) + "</svg>")


def patches_picture(labels: dict) -> str:
    """Test photo PHOTO, a car, with lines between its 64 patches of 4 by 4, and patch (1, 2) framed; next to it that
    patch enlarged, and its 48 numbers as a strip of colors, red, then green, then blue, each row by row."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    x = torch.tensor(X_test[PHOTO], dtype=torch.float32) / 255
    r, c = 1, 2
    patch = x[:, r * P:(r + 1) * P, c * P:(c + 1) * P]
    fig = plt.figure(figsize=(5.6, 2.5))
    whole = fig.add_axes((0.01, 0.12, 0.36, 0.76))
    big = fig.add_axes((0.43, 0.30, 0.2, 0.44))
    strip = fig.add_axes((0.43, 0.10, 0.56, 0.06))
    whole.imshow(np.transpose(x.numpy(), (1, 2, 0)), interpolation="nearest")
    for k in range(1, 32 // P):
        whole.axhline(k * P - 0.5, color=plots.PAGE, linewidth=0.8)
        whole.axvline(k * P - 0.5, color=plots.PAGE, linewidth=0.8)
    xs = [c * P - 0.5, (c + 1) * P - 0.5, (c + 1) * P - 0.5, c * P - 0.5, c * P - 0.5]
    ys = [r * P - 0.5, r * P - 0.5, (r + 1) * P - 0.5, (r + 1) * P - 0.5, r * P - 0.5]
    whole.plot(xs, ys, color=plots.INK, linestyle="-", linewidth=1.6, solid_joinstyle="miter")
    big.imshow(np.transpose(patch.numpy(), (1, 2, 0)), interpolation="nearest")
    flat = patch.reshape(3, P * P)
    colors = np.zeros((1, 3 * P * P, 3))
    for k in range(3):   # each number shown in its own color, as strong as it is
        colors[0, k * P * P:(k + 1) * P * P, k] = flat[k].numpy()
    strip.imshow(colors, interpolation="nearest", aspect="auto")
    for ax in (whole, big, strip):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("none")
    whole.set_title(labels["whole"], fontsize=9)
    big.set_title(labels["patch"], fontsize=9)
    strip.set_title(labels["numbers"], fontsize=9, loc="left")
    out = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return out


def guesses_picture(guesses: list, truth: list, lang: str) -> str:
    """The checked test photos of the rewind, one of each kind, each with the kind the network guessed at that moment
    under it: in ink when it is right, in the color of mistakes and with the right kind under it when it is not
    (yours.guesses: quick to draw again while the network trains)."""
    _, _, X_test, _ = cifar.load()
    said = [NAMES[lang][cifar.KINDS[g]] for g in guesses]
    right = [NAMES[lang][cifar.KINDS[k]] for k in truth]
    alt = {"en": "One test photo of each kind, each with the kind your ViT guessed then, and in brackets the right kind where it is wrong: ",
           "nl": "Eén testfoto van elke soort, elk met de soort die je ViT toen gokte, en tussen haakjes de goede soort waar het fout is: "}
    return yours.guesses([np.transpose(X_test[place], (1, 2, 0)) for place in SHOWN], said, right,
                         alt[lang] + "; ".join(f"{r}: {g}" for r, g in zip(right, said)) + ".")


_WATCHED_TEXT = {
    "en": {"title": "How alike the position of patch (3, 3) is to the others",
           "alt": "A grid of 8 by 8 patches; the stronger a patch is colored, the more alike its learned position is to that of the framed patch at row 3, column 3. Strongest: {near}."},
    "nl": {"title": "Hoe sterk de positie van patch (3, 3) op de andere lijkt",
           "alt": "Een rooster van 8 bij 8 patches; hoe sterker een patch gekleurd is, hoe meer zijn geleerde positie lijkt op die van de omlijste patch op rij 3, kolom 3. Het sterkst: {near}."},
}


def _strongest(row: list, lang: str) -> str:
    side = int(round(math.sqrt(len(row))))
    me = WATCHED[0] * side + WATCHED[1]
    best = sorted((k for k in range(len(row)) if k != me), key=lambda k: -row[k])[:3]
    return ", ".join(f"({k // side}, {k % side})" for k in best)


def now_pictures(seen: dict, lang: str) -> list:
    """What the learner's ViT makes at one moment of its run, as `look` or a rewind gives it: its guesses for the
    photos SHOWN, and under them how alike the position of the patch WATCHED is to those of the others."""
    out = [guesses_picture(seen["guesses"], seen["truth"], lang)] if seen.get("guesses") else []
    row = seen.get("watched")
    if row:
        texts = _WATCHED_TEXT[lang]
        out.append(watched_picture(row, texts["title"], texts["alt"].format(near=_strongest(row, lang))))
    return out


def epoch_seen(result: dict, e: int) -> dict:
    """What the rewind shows of the network after epoch e (0: before the first epoch), for now_pictures."""
    guesses, rows = result.get("guesses", []), result.get("watched", [])
    return {"guesses": guesses[e] if e < len(guesses) else [], "truth": result["truth"],
            "watched": rows[e] if e < len(rows) else None}


TOP = 2.6   # the top of the curves picture: above the loss of guessing, ln 10 ≈ 2.30


def best_epoch(val: list):
    """The first epoch with the lowest validation loss, rounded to 3 decimals as in REFERENCE."""
    return int(np.argmin(np.round(val, 3))) + 1 if len(val) else None


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the changed training photos solid and on the test photos
    dashed, with a dot where the validation loss is lowest; and in gray the validation loss of ColorCNN trained on the same
    changed photos (the course's run of cv-augment-photos, seed 0). With `at`, a line at that epoch, for rewinding."""
    train, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train + val if np.isfinite(v)]
    top = max([TOP] + [1.1 * v for v in mine])
    cnn = CURVES.get("cnn_val", [])
    p = chart.Panel((0, max(len(val), len(train), len(cnn), 1) + 0.5), (0, top), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    if cnn:
        p.line(list(range(1, len(cnn) + 1)), cnn, "ink-3", 1.6, dashed=True)
        p.text(len(cnn) - 0.3, cnn[-1] - 0.08 * top, labels["cnn"], "ink-3", "end")
    train_above = bool(train and val and train[-1] >= val[-1])   # with changed photos, the training loss can end higher
    for ys, dashed, name, above in ((train, False, labels["train"], train_above), (val, True, labels["val"], not train_above)):
        if ys:
            xs = list(range(1, len(ys) + 1))
            p.line(xs, ys, "loss", 2.2, dashed=dashed)
            p.text(len(ys) - 0.3, ys[-1] + (0.07 if above else -0.08) * top + (0.1 * top if cnn and not above else 0), name, "loss", "end")
    if val:
        best = best_epoch(val)
        p.points([best], [val[best - 1]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=380, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_frame(result: dict, e: int, labels: dict, lang: str) -> str:
    """The rewind at epoch e (0: before the first): the loss curves of the run with a line at e, then the network's
    guesses for the photos SHOWN, and under them the map of the patch WATCHED, as they were then (decision 1.17)."""
    return live.rewound(curves_picture(result["curves"], labels, lang, at=e if e else None),
                        now_pictures(epoch_seen(result, e), lang))


def moments(n: int) -> list:
    """The epochs to rewind to in a run of n epochs: 0 (before the first) and those of MOMENTS before its last, and
    its last."""
    return [m for m in MOMENTS if m < n] + ([n] if n else [])


# The rows of the picture of how many photos each network reads right, from top to bottom.
ROWS = [("cnn", "none"), ("cnn", "all"), ("vit", "none"), ("vit", "all")]


def gap_picture(labels: dict) -> str:
    """How many photos ColorCNN and the ViT read right after 40 epochs, of their training photos as they are and of
    the test photos, without changes and with mirror, crop and color: one row each, with a dot for every seed, filled
    for the training photos and open for the test photos, and a line between their means; the last row names both."""
    p = chart.Panel((40, 100), (-0.8, len(ROWS) - 0.25), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[40, 50, 60, 70, 80, 90, 100])
    for k, key in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        run = REFERENCE[key]
        train = [100 * v / N_TRAIN for v in run["train"]]
        test = [100 * v / N_TEST for v in run["test"]]
        middle = sum(test) / len(test), sum(train) / len(train)
        p.line(list(middle), [y, y], "ink-3", 1.6)
        p.points(train, [y] * len(train), "ink", r=3.5)
        p.points(test, [y] * len(test), "ink", r=3.5, hollow=True)
        p.text(41, y + 0.38, labels["_".join(key)], "ink-2", "start")
        if k == len(ROWS) - 1:   # the means lie close here: each name on its own side
            left = middle[0] <= middle[1]
            p.text(middle[0] + (-0.8 if left else 0.8), y - 0.45, labels["test"], "ink-2", "end" if left else "start")
            p.text(middle[1] + (0.8 if left else -0.8), y - 0.45, labels["train"], "ink-2", "start" if left else "end")
    svg = chart.figure([p], alt=labels["alt"], width=420, height=280)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# The lesson is done when these are correct. "learned-places" is the choice question in the lesson, and
# "inductive-bias" the question to answer in your own words.
REQUIRED = ["patches", "self-attention", "vit", "learned-places", "inductive-bias"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_K_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_K_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _yours():
    return yours


# --- exercise 1: patches --------------------------------------------------------------------------------------------

_P_HEAD = "import torch\n\n\ndef patches(x, p):\n"
_P_START = (_P_HEAD + "    # x: photos of shape (N, 3, H, W); p: the side of a patch, which divides H and W\n"
            "    # give back shape (N, (H // p) * (W // p), 3 * p * p): every patch a row of numbers,\n"
            "    # the patches row by row, and in each patch its red, its green and its blue, each row by row\n"
            "    n, c, h, w = x.shape\n    ...\n")
_P_BODY = ("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 2, 4, 1, 3, 5)\n"
           "    return x.reshape(n, (h // p) * (w // p), c * p * p)\n")


def _p(body: str = _P_BODY) -> str:
    return _P_HEAD + "    n, c, h, w = x.shape\n" + body


_P = _p()
# A made-up photo of 16 by 16 whose patches of 4 by 4 are easy to tell apart: red grows down the rows, green along the
# columns, and blue is a diagonal band; the picture under the check puts the learner's patches back, each in its place.
_P_PHOTO = ("import torch\n\nrows = torch.arange(16.0).reshape(16, 1).expand(16, 16) / 15\n"
            "cols = torch.arange(16.0).reshape(1, 16).expand(16, 16) / 15\n"
            "band = ((rows - cols).abs() < 0.25).float()\nx = torch.stack([rows, cols, band]).reshape(1, 3, 16, 16)\n"
            "photo = x[0].permute(1, 2, 0)\n")
_P_SCENARIOS = [
    # 2 photos of 4 by 4 with the numbers 0 to 95 in them, in patches of 2 by 2: the shape, and patch 1 of photo 0,
    # the top right square: its 2 by 2 in red, then in green, then in blue
    "import torch\n\nx = torch.arange(96.0).reshape(2, 3, 4, 4)\nout = patches(x, 2)\n[list(out.shape), out[0, 1].tolist()]",
    # a photo of 6 rows and 4 columns, in patches of 2 by 2: 3 rows of 2 patches; the shape, and patch 3, row 1, column 1
    "import torch\n\ntorch.manual_seed(0)\nx = torch.rand(1, 3, 6, 4)\nout = patches(x, 2)\n[list(out.shape), out[0, 3].tolist()]",
    # a photo of 32 by 32 in patches of 4 by 4, as the demo cuts it: 64 patches of 48 numbers; patch 9, row 1, column
    # 1, is pixels 4 to 7 of rows 4 to 7, and the last patch the bottom right corner
    "import torch\n\ntorch.manual_seed(1)\nx = torch.rand(2, 3, 32, 32)\nout = patches(x, 4)\n"
    "[list(out.shape), (out[1, 9] - x[1, :, 4:8, 4:8].reshape(48)).abs().max().item(), (out[0, 63] - x[0, :, 28:, 28:].reshape(48)).abs().max().item()]",
    # the made-up photo of the picture, its patches put back in their places with a black gap between them
    _P_PHOTO + "out = patches(x, 4)\n"
    "torch.nn.functional.pad(out[0].reshape(4, 4, 3, 4, 4), (0, 1, 0, 1)).permute(0, 3, 1, 4, 2).reshape(20, 20, 3)[:19, :19].tolist()",
]
_P_WRONG = {
    "no-permute": _p("    return x.reshape(n, (h // p) * (w // p), c * p * p)\n"),
    "colors-last": _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 2, 4, 3, 5, 1)\n"
                      "    return x.reshape(n, (h // p) * (w // p), c * p * p)\n"),
    "columns-first": _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 4, 2, 1, 3, 5)\n"
                        "    return x.reshape(n, (h // p) * (w // p), c * p * p)\n"),
    "rows-of-pixels": _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 2, 4, 1, 5, 3)\n"
                         "    return x.reshape(n, (h // p) * (w // p), c * p * p)\n"),
}

# --- exercise 2: SelfAttention --------------------------------------------------------------------------------------

_A_HEAD = ("import math\nimport torch\nfrom llm.attention import CausalSelfAttention\n\n\n"
           "class SelfAttention(CausalSelfAttention):\n"
           "    # the layers of your CausalSelfAttention, with a forward without the mask\n"
           "    def forward(self, x):\n        q, k, v = self.query(x), self.key(x), self.value(x)\n        outputs = []\n"
           "        for i in range(self.h):\n            a, b = i * self.d_head, (i + 1) * self.d_head\n")
_A_START = (_A_HEAD + "            # head i, for every photo at once: the scaled scores, the softmax, times the values\n"
            "            ...\n        # the outputs of the heads side by side, then the output projection\n        ...\n")
_A_SCORES = "            S = q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2) / math.sqrt(self.d_head)\n"
_A_WEIGHTS = "            outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])\n"
_A_OUT = "        return self.out(torch.cat(outputs, dim=2))\n"
_A_MASK = ("            n = x.shape[1]\n            future = torch.tril(torch.ones(n, n)) == 0\n"
           "            S = S.masked_fill(future, float(\"-inf\"))\n")


def _a(scores: str = _A_SCORES, mask: str = "", weights: str = _A_WEIGHTS, out: str = _A_OUT) -> str:
    return _A_HEAD + scores + mask + weights + out


_A = _a()
_A_SCENARIOS = [
    # 2 photos of 3 patches, 6 numbers each, in 2 heads of 3: the shape, and the new vector of the first patch of
    # photo 2, which now takes from all 3
    "import torch\n\ntorch.manual_seed(0)\nattention = SelfAttention(6, 2)\nx = 4 * torch.rand(2, 3, 6)\ny = attention(x)\n[list(y.shape), y[1, 0].tolist()]",
    # 3 heads of 2, one photo of 4 patches: what the first patch gets changes when the last patch changes
    "import torch\n\ntorch.manual_seed(1)\nattention = SelfAttention(6, 3)\nx = 4 * torch.rand(1, 4, 6)\nchanged = x.clone()\n"
    "changed[0, 3] = 0\n[attention(x)[0, 0].tolist(), (attention(x)[0, 0] - attention(changed)[0, 0]).abs().max().item() > 0]",
    # 3 photos of 5 patches, 4 numbers in 2 heads: the middle patch of the last photo
    "import torch\n\ntorch.manual_seed(2)\nattention = SelfAttention(4, 2)\nx = 4 * torch.rand(3, 5, 4)\ny = attention(x)\ny[2, 2].tolist()",
]

# --- exercise 3: ViT ------------------------------------------------------------------------------------------------

_V_HEAD = ("import torch\nfrom torch import nn\nfrom cnn.patches import patches\nfrom cnn.self_attention import SelfAttention\n"
           "from llm.block import Block\n\n\nclass ViT(nn.Module):\n    def __init__(self, d, h, depth, p):\n"
           "        super().__init__()\n        self.p = p\n        n = (32 // p) ** 2   # the patches of a photo of 32 by 32\n")
_V_INIT = ("        self.patch = nn.Linear(3 * p * p, d)\n        self.cls = nn.Parameter(torch.zeros(1, 1, d))\n"
           "        self.pos = nn.Embedding(n + 1, d)\n")
_V_REST = ("        self.blocks = nn.ModuleList([Block(d, h) for _ in range(depth)])\n        for block in self.blocks:\n"
           "            block.attention = SelfAttention(d, h)   # your Block, without the mask in its attention\n"
           "        self.norm = nn.LayerNorm(d)\n        self.out = nn.Linear(d, 10)\n\n    def forward(self, x):\n")
_V_FORWARD = ("        x = self.patch(patches(x, self.p))\n        cls = self.cls.expand(len(x), 1, -1)\n"
              "        x = torch.cat([cls, x], dim=1)\n        x = x + self.pos(torch.arange(x.shape[1]))\n")
_V_BLOCKS = "        for block in self.blocks:\n            x = block(x)\n"
_V_READ = "        return self.out(self.norm(x[:, 0]))\n"


def _v(init: str = _V_INIT, forward: str = _V_FORWARD, read: str = _V_READ) -> str:
    return _V_HEAD + init + _V_REST + forward + _V_BLOCKS + read


_V = _v()
_V_START = (_V_HEAD + "        # patch: a linear layer from the 3 * p * p numbers of a patch to d\n"
            "        # cls: the class token, a parameter of shape (1, 1, d), zeros at the start\n"
            "        # pos: a learned position for each of the n patches and the class token\n        ...\n"
            + _V_REST + "        # x: photos of shape (N, 3, 32, 32); first the patches as tokens, then the\n"
            "        # class token in front of the tokens of every photo, then the positions added\n        ...\n"
            + _V_BLOCKS + "        # the class token after the last block, normalized, through out\n        ...\n")
_V_SCENARIOS = [
    # one block of 8 numbers in 2 heads, patches of 8 by 8: 2 photos give 10 logits each; the first 4 of photo 2
    "import torch\n\ntorch.manual_seed(0)\nmodel = ViT(8, 2, 1, 8)\nx = torch.rand(2, 3, 32, 32)\ny = model(x)\n[list(y.shape), y[1, :4].tolist()]",
    # the demo's ViT: its parameters, in the order they were made, and how many numbers they hold
    "import torch\n\ntorch.manual_seed(1)\nmodel = ViT(16, 2, 3, 4)\n[[list(p.shape) for p in model.parameters()], sum(p.numel() for p in model.parameters())]",
    # the class token reads every patch: a change in the bottom right corner changes the logits; and two photos in a
    # batch give what each gives alone
    "import torch\n\ntorch.manual_seed(2)\nmodel = ViT(8, 2, 2, 4)\nx = torch.rand(2, 3, 32, 32)\nchanged = x.clone()\n"
    "changed[:, :, 28:, 28:] = 0\nboth = model(x)\n[(both - model(changed)).abs().max().item() > 0, (both[1] - model(x[1:])[0]).abs().max().item() < 1e-5, both[0, :3].tolist()]",
]


def _v_order(got, text) -> bool:
    """The three layers of the patches made in another order, and nothing else wrong."""
    lines = text.split("\n")
    rows = []
    for name in ("patch", "cls", "pos"):
        found = [i for i, line in enumerate(lines) if line.strip().startswith(f"self.{name} =")]
        if len(found) != 1:
            return False
        rows.append(found[0])
    if rows == sorted(rows):
        return False
    fixed = list(lines)
    for row, i in zip(sorted(rows), rows):
        fixed[row] = lines[i]
    return EXERCISES["vit"]._same_everywhere("\n".join(fixed), _V)


EXERCISES = {
    "patches": ScenarioBox(
        name="patches",
        picture=_yours().image(call=3, show=("photo",), size=120),
        start=_P_START,
        solution=_P,
        scenarios=_P_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "patches"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, (AttributeError, TypeError)) and "NoneType" in str(e),
                    en="The check asks for the shape and the numbers of what your function gives back, and it gives back nothing, `None`: end with `return` and the patches.",
                    nl="De check vraagt de shape en de getallen van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de patches."),
            Mistake("view", error=lambda e: isinstance(e, RuntimeError) and "view size is not compatible" in str(e),
                    en="`.view` cannot give the numbers a new shape after `permute`: `permute` only changes the order in which the axes are read, not where the numbers sit, and `.view` needs them in reading order. Use `.reshape`: it copies them into that order where it has to.",
                    nl="`.view` kan de getallen na `permute` geen nieuwe shape geven: `permute` verandert alleen de volgorde waarin de axes gelezen worden, niet waar de getallen staan, en `.view` heeft ze in leesvolgorde nodig. Gebruik `.reshape`: dat kopieert ze in die volgorde waar het moet."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and re.search(r"shape '\[.*\]' is invalid for input of size", str(e)) is not None,
                    en="One of your shapes does not hold as many numbers as the photos: a batch of N photos of shape (3, H, W) holds N · 3 · H · W numbers. Cut each axis of the photo into two: rows into `h // p` rows of patches of `p` rows each, `(n, c, h // p, p, w // p, p)`. That holds for a photo that is not square too: its columns are `w`, not `h`.",
                    nl="Een van je shapes bevat niet zoveel getallen als de foto's: een batch van N foto's met shape (3, H, W) bevat N · 3 · H · W getallen. Knip elke axis van de foto in tweeën: de rijen in `h // p` rijen patches van elk `p` rijen, `(n, c, h // p, p, w // p, p)`. Dat geldt ook voor een foto die niet vierkant is: zijn kolommen zijn `w`, niet `h`."),
            Mistake("no-permute", wrong=_P_WRONG["no-permute"],
                    en=_S_EN + ": the right shape, but your patches are not squares. A reshape reads the numbers in the order they sit, row by row, so every \"patch\" is a piece of a few rows of pixels. First cut the rows and the columns each into two axes, `(n, c, h // p, p, w // p, p)`, then put the two axes that say which patch first with `permute`, and only then the numbers of one patch.",
                    nl=_S_NL + ": de goede shape, maar je patches zijn geen vierkanten. Een reshape leest de getallen in de volgorde waarin ze staan, rij na rij, dus elke \"patch\" is een stuk van een paar rijen pixels. Knip eerst de rijen en de kolommen elk in twee axes, `(n, c, h // p, p, w // p, p)`, zet dan met `permute` de twee axes die zeggen welke patch het is vooraan, en pas daarna de getallen van één patch."),
            Mistake("colors-last", wrong=_P_WRONG["colors-last"],
                    en=_S_EN + ": your patches hold the right pixels, but in another order: every pixel with its 3 colors together. The course keeps a patch as the photo keeps it, its red, then its green, then its blue, each row by row, so that `patch.reshape(3, p, p)` gives the square back: after the axes of which patch, first the colors, then the rows and the columns of the patch, `permute(0, 2, 4, 1, 3, 5)`.",
                    nl=_S_NL + ": je patches bevatten de goede pixels, maar in een andere volgorde: elke pixel met zijn 3 kleuren bij elkaar. De course houdt een patch zoals de foto hem houdt, zijn rood, dan zijn groen, dan zijn blauw, elk rij na rij, zodat `patch.reshape(3, p, p)` het vierkant teruggeeft: na de axes van welke patch eerst de kleuren, dan de rijen en de kolommen van de patch, `permute(0, 2, 4, 1, 3, 5)`."),
            Mistake("columns-first", wrong=_P_WRONG["columns-first"],
                    en=_S_EN + ": your patches are right, but they come column by column: down the first column of patches, then the second. A text goes row by row, and so do the patches: first the axis of the rows of patches, then that of the columns, `permute(0, 2, 4, 1, 3, 5)`.",
                    nl=_S_NL + ": je patches kloppen, maar ze komen kolom na kolom: de eerste kolom patches naar beneden, dan de tweede. Een tekst gaat rij na rij, en de patches ook: eerst de axis van de rijen patches, dan die van de kolommen, `permute(0, 2, 4, 1, 3, 5)`."),
            Mistake("rows-of-pixels", wrong=_P_WRONG["rows-of-pixels"],
                    en=_S_EN + ": your patches hold the right pixels, but every color of a patch goes column by column. Keep the rows of a patch before its columns, as in the photo: `permute(0, 2, 4, 1, 3, 5)`.",
                    nl=_S_NL + ": je patches bevatten de goede pixels, maar elke kleur van een patch gaat kolom na kolom. Houd de rijen van een patch vóór zijn kolommen, zoals in de foto: `permute(0, 2, 4, 1, 3, 5)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: cut the rows and the columns each into two axes with `reshape`, put the axes in another order with `permute`, and give back one row of `3 * p * p` numbers per patch.",
                      "nl": "Vervang de `...` door je code: knip de rijen en de kolommen elk in twee axes met `reshape`, zet de axes in een andere volgorde met `permute`, en geef één rij van `3 * p * p` getallen per patch terug."},
            "wrong": {"en": _S_EN + ". The shape `(n, (h // p) * (w // p), c * p * p)`, the patches row by row, and in each patch its red, its green and its blue, each row by row: steps 1 to 3 of the recipe?",
                      "nl": _S_NL + ". De shape `(n, (h // p) * (w // p), c * p * p)`, de patches rij na rij, en in elke patch zijn rood, zijn groen en zijn blauw, elk rij na rij: stap 1 tot en met 3 van het recept?"},
            "almost": {"en": "Right for the first photos. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function use `h`, `w` and `p`, and no fixed numbers, also for a photo that is not square?",
                       "nl": "Goed voor de eerste foto's. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Gebruikt je functie `h`, `w` en `p`, en geen vaste getallen, ook voor een foto die niet vierkant is?"},
            "correct": {"en": "Your `patches` is right: every photo is now a row of patches, as a text is a row of tokens, and each patch a row of its numbers. It is now in your project, as `{saved}`.",
                        "nl": "Je `patches` klopt: elke foto is nu een rij patches, zoals een tekst een rij tokens is, en elke patch een rij van zijn getallen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `patches` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `patches` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "self-attention": ClassBox(
        name="self-attention",
        picture=netview.network("SelfAttention", lambda: 4 * torch.rand(1, 5, 6), 6, 2, scores=False),
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "self_attention"),
        first=[_DOTS],
        mistakes=[
            Mistake("dot-T", syntax=r"^[^#\n]*(?:\[\s*:\s*,\s*:\s*,[^\]\n]*\]|\b[qkv])\.T\b",
                    en="`{line}` uses `.T`, which turns a tensor of 3 axes around completely: (B, n, d_head) becomes (d_head, n, B). In a mini-batch, swap only the last two axes, so that every photo gets its own transpose: `.transpose(1, 2)`.",
                    nl="`{line}` gebruikt `.T`, dat een tensor met 3 axes helemaal omdraait: (B, n, d_head) wordt (d_head, n, B). Wissel in een mini-batch alleen de laatste twee axes, zodat elke foto zijn eigen getransponeerde krijgt: `.transpose(1, 2)`."),
            Mistake("cat-axis", syntax=r"^[^#\n]*torch\.cat\(\s*outputs\s*,\s*(?:dim\s*=\s*)?[01]\s*\)",
                    en="`{line}` puts the outputs of the heads together along another axis than the columns, so `self.out` does not get d numbers per patch. In a mini-batch of shape (B, n, d), the columns are axis 2: `dim=2`.",
                    nl="`{line}` zet de outputs van de heads aan elkaar langs een andere axis dan de kolommen, dus `self.out` krijgt geen d getallen per patch. In een mini-batch met shape (B, n, d) zijn de kolommen axis 2: `dim=2`."),
            Mistake("parent", matches=lambda got, text: re.search(r"super\(\)\.forward\(", text) is not None,
                    en=_K_EN + ": `super().forward(x)` runs the forward of your CausalSelfAttention, mask and all. Write the forward of SelfAttention itself, without the mask: the scores, the softmax and the values of each head, then the output projection.",
                    nl=_K_NL + ": `super().forward(x)` voert de forward van je CausalSelfAttention uit, met de mask. Schrijf de forward van SelfAttention zelf, zonder de mask: de scores, de softmax en de values van elke head, en dan de output-projectie."),
            Mistake("mask-kept", wrong=_a(mask=_A_MASK),
                    en=_K_EN + ": every patch still takes only from the patches before it, as a token does in your CausalSelfAttention, so the first patch sees only itself, and a change in the last patch never reaches it. A photo has no before and after: leave out the mask, and every patch takes from every patch.",
                    nl=_K_NL + ": elke patch neemt nog steeds alleen iets over van de patches ervoor, zoals een token in je CausalSelfAttention, dus de eerste patch ziet alleen zichzelf, en een verandering in de laatste patch komt nooit bij hem aan. Een foto heeft geen ervoor en erna: laat de mask weg, en elke patch neemt iets over van elke patch."),
            Mistake("softmax-axis", wrong=_a(weights=_A_WEIGHTS.replace("dim=2", "dim=1")),
                    en=_K_EN + ": your softmax goes along axis 1, down the queries of a photo, so the weights of one patch do not add up to 1. In a mini-batch, the scores of one patch are a row along axis 2: `torch.softmax(S, dim=2)`.",
                    nl=_K_NL + ": je softmax gaat langs axis 1, langs de queries van een foto, dus de weights van één patch tellen niet op tot 1. In een mini-batch zijn de scores van één patch een rij langs axis 2: `torch.softmax(S, dim=2)`."),
            Mistake("no-scale", wrong=_a(scores=_A_SCORES.replace(" / math.sqrt(self.d_head)", "")),
                    en=_K_EN + ": your scores are not scaled. Divide them by the square root of the width of a head: `/ math.sqrt(self.d_head)`.",
                    nl=_K_NL + ": je scores zijn niet geschaald. Deel ze door de wortel uit de breedte van een head: `/ math.sqrt(self.d_head)`."),
            Mistake("no-out", wrong=_a(out="        return torch.cat(outputs, dim=2)\n"),
                    en=_K_EN + ": the outputs of the heads come out side by side, without the output projection. Send them through `self.out`: `self.out(torch.cat(outputs, dim=2))`.",
                    nl=_K_NL + ": de outputs van de heads komen naast elkaar naar buiten, zonder de output-projectie. Stuur ze door `self.out`: `self.out(torch.cat(outputs, dim=2))`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in the loop, the attention of head `i` on its own columns, for every photo at once and without the mask, added to `outputs`; after the loop, the outputs side by side through the output projection.",
                      "nl": "Vervang de twee `...`: in de loop de attention van head `i` op zijn eigen kolommen, voor elke foto tegelijk en zonder de mask, toegevoegd aan `outputs`; na de loop de outputs naast elkaar door de output-projectie."},
            "wrong": {"en": _K_EN + ". Per head: the scores of its columns with `transpose(1, 2)`, scaled, the softmax along axis 2, times its values, and no mask; then the outputs side by side along axis 2, through `self.out`?",
                      "nl": _K_NL + ". Per head: de scores van zijn kolommen met `transpose(1, 2)`, geschaald, de softmax langs axis 2, maal zijn values, en geen mask; dan de outputs naast elkaar langs axis 2, door `self.out`?"},
            "almost": {"en": "Right for the first example. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your class use `self.h` and `self.d_head`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Gebruikt je class `self.h` en `self.d_head`, en geen vaste getallen?"},
            "correct": {"en": "Your attention works without a mask: every patch takes from every patch of its own photo, also from those after it. It keeps the layers of your CausalSelfAttention, and is now in your project, as `{saved}`.",
                        "nl": "Je attention werkt zonder mask: elke patch neemt iets over van elke patch van zijn eigen foto, ook van die erna. Hij houdt de layers van je CausalSelfAttention, en staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `SelfAttention` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `SelfAttention` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "vit": ClassBox(
        name="vit",
        picture=netview.network("ViT", lambda: torch.rand(1, 3, 32, 32), D, H, DEPTH, P),
        start=_V_START,
        solution=_V,
        scenarios=_V_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "vit"),
        first=[_DOTS],
        mistakes=[
            Mistake("expand", error=lambda e: isinstance(e, RuntimeError) and "Sizes of tensors must match except in dimension 1" in str(e),
                    en="`torch.cat` along axis 1 puts the class token in front of the patches only when both hold as many photos along axis 0: `self.cls` has shape (1, 1, d), one token for one photo, and the patches (N, n, d). Give every photo its own copy first: `self.cls.expand(len(x), 1, -1)` has shape (N, 1, d).",
                    nl="`torch.cat` langs axis 1 zet het class token alleen vóór de patches als ze langs axis 0 evenveel foto's hebben: `self.cls` heeft shape (1, 1, d), één token voor één foto, en de patches (N, n, d). Geef eerst elke foto zijn eigen kopie: `self.cls.expand(len(x), 1, -1)` heeft shape (N, 1, d)."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and re.search(r"cannot be multiplied|must match the size of tensor", str(e)) is not None,
                    en="The sizes do not fit: one of your layers gets another number of numbers than it was made for. `self.patch` gets the `3 * p * p` numbers of a patch and gives d, `self.pos` has a row of d numbers for each of the `n + 1` tokens, the patches and the class token, and `self.out` gets the d numbers of the class token.",
                    nl="De maten passen niet: een van je layers krijgt een ander aantal getallen dan waarvoor hij gemaakt is. `self.patch` krijgt de `3 * p * p` getallen van een patch en geeft er d, `self.pos` heeft een rij van d getallen voor elk van de `n + 1` tokens, de patches en het class token, en `self.out` krijgt de d getallen van het class token."),
            Mistake("pos-range", error=lambda e: isinstance(e, IndexError) and "index out of range" in str(e),
                    en="`self.pos` has fewer rows than there are tokens: with the class token, a photo has `n + 1` of them. Make it `nn.Embedding(n + 1, d)`, and look up `torch.arange(x.shape[1])` after the class token is in front.",
                    nl="`self.pos` heeft minder rijen dan er tokens zijn: met het class token heeft een foto er `n + 1`. Maak hem `nn.Embedding(n + 1, d)`, en zoek `torch.arange(x.shape[1])` op nadat het class token vooraan staat."),
            Mistake("no-pos", wrong=_v(forward=_V_FORWARD.replace("        x = x + self.pos(torch.arange(x.shape[1]))\n", "")),
                    en=_K_EN + ": your ViT has positions, but does not add them, so where a patch lies makes no difference to it: shuffle the patches, and the class token gets the same numbers. Add them after the class token is in front: `x = x + self.pos(torch.arange(x.shape[1]))`.",
                    nl=_K_NL + ": je ViT heeft posities, maar telt ze niet op, dus maakt het voor hem niets uit waar een patch ligt: hussel de patches door elkaar, en het class token krijgt dezelfde getallen. Tel ze erbij op nadat het class token vooraan staat: `x = x + self.pos(torch.arange(x.shape[1]))`."),
            Mistake("mean", wrong=_v(read="        return self.out(self.norm(x.mean(dim=1)))\n"),
                    en=_K_EN + ": you take the mean of all tokens after the last block. Some vision transformers do that, but then the class token is not needed; this ViT reads the class token, the first token: `x[:, 0]`.",
                    nl=_K_NL + ": je neemt het gemiddelde van alle tokens na het laatste block. Sommige vision transformers doen dat, maar dan is het class token niet nodig; deze ViT leest het class token, het eerste token: `x[:, 0]`."),
            Mistake("cls-last", wrong=_v(forward=_V_FORWARD.replace("torch.cat([cls, x], dim=1)", "torch.cat([x, cls], dim=1)"),
                                         read="        return self.out(self.norm(x[:, -1]))\n"),
                    en=_K_EN + ": your class token comes after the patches. That works as well, but then every position belongs to another token than in the course's ViT. Put it in front, as the paper does: `torch.cat([cls, x], dim=1)`, and read `x[:, 0]`.",
                    nl=_K_NL + ": je class token komt na de patches. Dat werkt net zo goed, maar dan hoort elke positie bij een ander token dan in de ViT van de course. Zet hem vooraan, zoals de paper doet: `torch.cat([cls, x], dim=1)`, en lees `x[:, 0]`."),
            Mistake("last-patch", wrong=_v(read="        return self.out(self.norm(x[:, -1]))\n"),
                    en=_K_EN + ": you read the last token, the patch in the bottom right corner. The class token is the first: `x[:, 0]`.",
                    nl=_K_NL + ": je leest het laatste token, de patch in de hoek rechtsonder. Het class token is het eerste: `x[:, 0]`."),
            Mistake("no-norm", wrong=_v(read="        return self.out(x[:, 0])\n"),
                    en=_K_EN + ": the class token goes to `self.out` without the last layer normalization. After the last block, normalize it first: `self.out(self.norm(x[:, 0]))`.",
                    nl=_K_NL + ": het class token gaat zonder de laatste layer normalization naar `self.out`. Normaliseer het na het laatste block eerst: `self.out(self.norm(x[:, 0]))`."),
            Mistake("order", matches=_v_order,
                    en=_K_EN + ": the right shape, other numbers. Make `patch`, `cls` and `pos` in that order, the order the data passes them, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_K_NL + ": de goede shape, andere getallen. Maak `patch`, `cls` en `pos` in die volgorde, de volgorde waarin de data erdoor gaat, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: in `__init__` the patch embedding, the class token and the positions; in `forward` the patches as tokens with the class token in front and the positions added; and at the end the class token, normalized, through `self.out`.",
                      "nl": "Vervang de drie `...`: in `__init__` de patch embedding, het class token en de posities; in `forward` de patches als tokens met het class token vooraan en de posities erbij; en aan het eind het class token, genormaliseerd, door `self.out`."},
            "wrong": {"en": _K_EN + ". The patches through `self.patch`, the class token in front of every photo's tokens, the positions of all `n + 1` tokens added, the blocks, and at the end the first token through `self.norm` and `self.out`: steps 4 to 6 of the recipe?",
                      "nl": _K_NL + ". De patches door `self.patch`, het class token vóór de tokens van elke foto, de posities van alle `n + 1` tokens erbij, de blocks, en aan het eind het eerste token door `self.norm` en `self.out`: stap 4 tot en met 6 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your class use `d`, `p` and `n`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Gebruikt je class `d`, `p` en `n`, en geen vaste getallen?"},
            "correct": {"en": "Your ViT gives 10 logits per photo, from the class token after the last block, and the class token reads every patch. `ViT(16, 2, 3, 4)` has 11,738 parameters, about as many as your ColorCNN. It is now in your project, as `{saved}`.",
                        "nl": "Je ViT geeft 10 logits per foto, uit het class token na het laatste block, en het class token leest elke patch. `ViT(16, 2, 3, 4)` heeft 11.738 parameters, ongeveer zoveel als je ColorCNN. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ViT` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ViT` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("patches", _P_START, "empty", None),
    ("patches", _P, "correct", None),
    # one reshape with -1, and a loop over the patches with slices: other code, the same patches
    ("patches", _p("    x = x.reshape(n, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)\n    return x.reshape(n, -1, c * p * p)\n"), "correct", None),
    ("patches", _p("    rows = []\n    for r in range(h // p):\n        for k in range(w // p):\n"
                   "            rows.append(x[:, :, r * p:(r + 1) * p, k * p:(k + 1) * p].reshape(n, 1, c * p * p))\n"
                   "    return torch.cat(rows, dim=1)\n"), "correct", None),
    ("patches", _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 2, 4, 1, 3, 5)\n    x.reshape(n, (h // p) * (w // p), c * p * p)\n"), "error", "no-return"),
    ("patches", _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    x = x.permute(0, 2, 4, 1, 3, 5)\n    return x.view(n, (h // p) * (w // p), c * p * p)\n"), "error", "view"),
    ("patches", _p("    x = x.reshape(n, c, h, p, w, p)\n    x = x.permute(0, 2, 4, 1, 3, 5)\n    return x.reshape(n, h * w, c * p * p)\n"), "error", "sizes"),
    ("patches", _p("    x = x.reshape(n, c, h // p, p, w // p, p)\n    ...\n    return x.reshape(n, (h // p) * (w // p), c * p * p)\n"), "wrong", "dots-left"),
    # a square photo only: h for w
    ("patches", _p("    x = x.reshape(n, c, h // p, p, h // p, p)\n    x = x.permute(0, 2, 4, 1, 3, 5)\n    return x.reshape(n, (h // p) * (h // p), c * p * p)\n"), "error", "sizes"),
] + [("patches", code, "wrong", key) for key, code in _P_WRONG.items()] + [
    ("self-attention", _A_START, "empty", None),
    ("self-attention", _A, "correct", None),
    ("self-attention", _a(scores="            S = q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2) / self.d_head ** 0.5\n"), "correct", None),
    ("self-attention", _a(out="        return self.out(torch.cat(outputs, 2))\n"), "correct", None),
    ("self-attention", _a(mask=_A_MASK), "wrong", "mask-kept"),
    ("self-attention", _a(scores="", weights="            return super().forward(x)\n", out=""), "wrong", "parent"),
    ("self-attention", _a(weights=_A_WEIGHTS.replace("dim=2", "dim=1")), "wrong", "softmax-axis"),
    ("self-attention", _a(scores=_A_SCORES.replace(" / math.sqrt(self.d_head)", "")), "wrong", "no-scale"),
    ("self-attention", _a(out="        return torch.cat(outputs, dim=2)\n"), "wrong", "no-out"),
    ("self-attention", _a(scores=_A_SCORES.replace(".transpose(1, 2)", ".T")), "error", "dot-T"),
    ("self-attention", _a(out="        return self.out(torch.cat(outputs, dim=1))\n"), "error", "cat-axis"),
    ("self-attention", _a(scores=_A_SCORES.replace("math.sqrt(self.d_head)", "math.sqrt(3)")), "almost", None),
    ("vit", _V_START, "empty", None),
    ("vit", _V, "correct", None),
    # the class token with a broadcast instead of expand, and the positions looked up for n + 1 tokens by name
    ("vit", _v(forward=_V_FORWARD.replace("self.cls.expand(len(x), 1, -1)", "torch.zeros(len(x), 1, 1) + self.cls")), "correct", None),
    ("vit", _v(forward=_V_FORWARD.replace("torch.arange(x.shape[1])", "torch.arange(len(self.pos.weight))")), "correct", None),
    ("vit", _v(forward=_V_FORWARD.replace("        cls = self.cls.expand(len(x), 1, -1)\n", "        cls = self.cls\n")), "error", "expand"),
    ("vit", _v(init=_V_INIT.replace("nn.Embedding(n + 1, d)", "nn.Embedding(n, d)")), "error", "pos-range"),
    ("vit", _v(init=_V_INIT.replace("nn.Linear(3 * p * p, d)", "nn.Linear(p * p, d)")), "error", "sizes"),
    ("vit", _v(forward=_V_FORWARD.replace("        x = x + self.pos(torch.arange(x.shape[1]))\n", "")), "wrong", "no-pos"),
    ("vit", _v(read="        return self.out(self.norm(x.mean(dim=1)))\n"), "wrong", "mean"),
    ("vit", _v(forward=_V_FORWARD.replace("torch.cat([cls, x], dim=1)", "torch.cat([x, cls], dim=1)"), read="        return self.out(self.norm(x[:, -1]))\n"), "wrong", "cls-last"),
    ("vit", _v(read="        return self.out(self.norm(x[:, -1]))\n"), "wrong", "last-patch"),
    ("vit", _v(read="        return self.out(x[:, 0])\n"), "wrong", "no-norm"),
    ("vit", _v(init="        self.pos = nn.Embedding(n + 1, d)\n        self.patch = nn.Linear(3 * p * p, d)\n        self.cls = nn.Parameter(torch.zeros(1, 1, d))\n"), "wrong", "order"),
    ("vit", _v(read="        return ...\n"), "wrong", "dots-left"),
]


# --- the demo: the learner's ViT on changed photos -----------------------------------------------------------------

class _Changed(Dataset):
    """The first N_TRAIN training photos, each changed anew every time it is asked for, as in the demo of
    cv-augment-photos: mirrored when the first of 6 numbers from torch.rand is below 0.5, cut out at the place the next
    two give, and changed in color with the factors of the last three; then every number is kept between 0 and 1, and
    the photo normalized."""

    def __init__(self, X, y, mean, std, crop, jitter):
        self.X, self.y, self.crop, self.jitter = X, y, crop, jitter
        self.mean, self.std = mean[:, None, None], std[:, None, None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()   # a jitter that changes its x itself then leaves the training photo as it is
        u = torch.rand(6)
        if u[0] < 0.5:
            x = x.flip(dims=[2])
        x = self.crop(x, int(u[1] * (2 * PAD + 1)), int(u[2] * (2 * PAD + 1)))
        x = torch.as_tensor(self.jitter(x, *(LOW + (HIGH - LOW) * u[3:6]).tolist())).clamp(0, 1)
        return (x - self.mean) / self.std, self.y[i]


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def _positions(model):
    """The learned positions of the ViT, as a list of rows, or None when it has none the course can read."""
    pos = getattr(model, "pos", None)
    weight = getattr(pos, "weight", pos)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or len(weight) != (32 // P) ** 2 + 1:
        return None
    return weight.detach().tolist()


def hearing(model, shown, truth: list, show, guesses: list, rows: list):
    """How the demo hears the learner's Trainer while it fits their ViT (live.follow), decision 1.17: after every
    epoch it keeps the ViT's guesses for the photos `shown` in `guesses`, and how alike the position of the patch
    WATCHED is to those of the others in `rows`, for the rewind; and it passes every report on to `show` with "look",
    what the ViT makes at that moment. Both look in eval mode without a graph and put every module back in its own
    mode (netview.looking), so the run stays the same. A look that fails never stops the run."""
    def seen() -> dict:
        with netview.looking(model):
            now = [int(g) for g in model(shown).argmax(dim=1)]
        pos = _positions(model)
        return {"guesses": now, "truth": truth, "watched": watched_row(pos) if pos else None}

    def heard(e):
        if "val_loss" in e:   # after every epoch: the guesses and the map for the rewind
            try:
                now = seen()
            except Timeout:
                raise
            except Exception:
                now = {"guesses": [], "watched": None}
            guesses.append(now["guesses"])
            rows.append(now["watched"])
        if show is not None:
            try:
                show(e | {"look": seen})
            except Timeout:   # the demo's time limit, also when it falls while the chart draws
                raise
            except Exception:
                pass
    return heard


def train_run(ViT, Trainer, crop, jitter, mean, std, seed: int = 0, changes: bool = True, show=None, epochs: int | None = None):
    """Train ViT(D, H, DEPTH, P) as the course measured it (REFERENCE): made after torch.manual_seed(seed), the training
    photos shuffled after torch.manual_seed(seed + 1), changed when `changes`, with the Trainer for `epochs` epochs.
    Gives the model, its history, the guesses and the maps after every epoch (the first before training), and the
    positions after the last epoch. `epochs` is EPOCHS unless given (read when it runs, so a test can change EPOCHS)."""
    epochs = EPOCHS if epochs is None else epochs
    X, y, X_test, y_test = photos()
    torch.manual_seed(seed)
    model = ViT(D, H, DEPTH, P)
    X_norm = (X - mean[:, None, None]) / std[:, None, None]
    X_seen = (X_test - mean[:, None, None]) / std[:, None, None]
    truth, guesses, rows = [int(y_test[place]) for place in SHOWN], [], []
    shown = X_seen[list(SHOWN)]
    heard = hearing(model, shown, truth, show, guesses, rows)
    with netview.looking(model):   # before the first epoch, for the rewind
        guesses.append([int(g) for g in model(shown).argmax(dim=1)])
    pos = _positions(model)
    rows.append(watched_row(pos) if pos else None)
    torch.manual_seed(seed + 1)
    data = _Changed(X, y, mean, std, crop, jitter) if changes else TensorDataset(X_norm, y)
    loader = DataLoader(data, batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_seen, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=epochs, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "vit", every=EVERY)
    history = trainer.fit() or []
    return model, history, guesses, rows, _positions(model), X_norm, X_seen


def _demo(show=None):
    """For the demo: None when patches.py, self_attention.py or vit.py is not in the project yet; "data" when CIFAR-10
    is not unpacked; "vit" when the learner's ViT stops with an error or gives no 10 logits for a photo; "changes" when
    their crop or jitter does; "trainer" when the Trainer, count_params or channel_stats does; "slow" when the training
    takes too long; else a dict with how many of the training photos as they are and of the test photos it reads
    right, the loss after every epoch on both ("curves"), the epoch with the lowest validation loss ("best"), the
    ViT's guesses for the photos SHOWN before and after every epoch ("guesses") and their right kinds ("truth"), how
    alike the position of the patch WATCHED is to the others then ("watched"), "near" and "far" after the last epoch,
    and the modules the course's versions stood in for. `show` hears the run while it trains (live.follow, every 10
    batches), with what it makes then (hearing)."""
    if not (_saved("patches") and _saved("self_attention") and _saved("vit")):
        return None
    if not ready():
        return "data"
    run = run_code("from cnn.vit import ViT")
    if run.error is not None:
        return "vit"
    ViT = run.names["ViT"]
    run = run_code("from cnn.crop import crop\nfrom cnn.color import jitter")
    if run.error is not None:
        return "changes"
    crop, jitter = run.names["crop"], run.names["jitter"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params\nfrom cnn.channels import channel_stats")
    if run.error is not None:
        return "trainer"
    Trainer, count_params, channel_stats = run.names["Trainer"], run.names["count_params"], run.names["channel_stats"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = photos()
    try:
        out = with_time_limit(lambda: ViT(D, H, DEPTH, P)(X[:2]), 30)
        if tuple(out.shape) != (2, 10):
            return "vit"
        params = int(count_params(ViT(D, H, DEPTH, P)))
    except Timeout:
        return "slow"
    except Exception:
        return "vit"
    for change in (lambda x: crop(x, 0, 8), lambda x: jitter(x, 1.1, 0.9, 1.2)):
        try:
            got = with_time_limit(lambda: torch.as_tensor(change(X[0].clone())), 10)
            if tuple(got.shape) != (3, 32, 32) or not torch.isfinite(got).all():
                return "changes"
        except Exception:
            return "changes"
    try:
        mean, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    except Exception:
        return "trainer"
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        return "trainer"
    try:
        model, history, guesses, rows, pos, X_norm, X_seen = with_time_limit(
            lambda: train_run(ViT, Trainer, crop, jitter, mean, std, show=show), 600)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    n = len(curves["val"]) + 1
    near, far = near_far(pos) if pos else (None, None)
    return {"params": params, "train": _accuracy(model, X_norm, y), "test": _accuracy(model, X_seen, y_test),
            "curves": curves, "best": best_epoch(curves["val"]), "guesses": guesses[:n], "watched": rows[:n],
            "truth": [int(y_test[place]) for place in SHOWN], "near": near, "far": far, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "patches"), ("cnn", "self_attention"), ("cnn", "vit"), ("cnn", "crop"),
                            ("cnn", "color"), ("cnn", "channels"), ("cnn", "count"), ("llm", "attention"),
                            ("llm", "block"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
