---
license: cc-by-4.0
pipeline_tag: image-classification
tags:
- digits
- qmnist
- everything-ai
---

# The digit reader of Everything AI

A small network that reads handwritten digits: a grey image of 28 by 28 pixels in, 10 logits out, one for each digit 0 to 9. The course Everything AI trained it for its lessons, and shares it here so that anyone can load it.

## Model

- **Class:** `DigitCNN`, in `everything_ai.models` of the course; its code is below.
- **Layers:** two convolutions of 3 by 3, with 8 and 16 channels, each with ReLU and max pooling of 2 by 2; a hidden layer of 64 neurons; an embedding of 2 numbers; and 10 logits.
- **Input:** a tensor of shape (n, 784), the pixels of each image row after row, or (n, 1, 28, 28), with pixels from 0 to 1: a white digit on black.
- **Size:** 51,648 parameters, in 10 tensors.
- **File:** `model.pt`, a `state_dict` for PyTorch. Load it with `torch.load("model.pt", weights_only=True)`.

```python
import torch
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.hidden = nn.Linear(16 * 7 * 7, 64)
        self.embed = nn.Linear(64, 2)
        self.out = nn.Linear(2, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = self.embed(torch.relu(self.hidden(x.flatten(1))))
        return self.out(x)
```

## Training data

The first 6,000 training images of QMNIST (Chhavi Yadav and Léon Bottou, 2019): handwritten digits in the form of MNIST, white on black, 28 by 28 pixels, with pixels from 0 to 1. QMNIST has the license BSD-3-Clause, Copyright (c) Facebook, Inc. and its affiliates.

## Training

Adam with a learning rate of 0.003, 30 epochs, mini-batches of 64, seed 2, on one thread, with PyTorch 2.14.1. The script `data/tools/train_course_cnn.py` of the course gives the same file again with the same PyTorch.

## How well it does

929 of the first 1,000 test images of QMNIST right: images the model never trained on.

## Limits

- Only digits as in QMNIST and MNIST: one digit per image, white on black, centred, 28 by 28 pixels. A photo of a digit, a digit on paper as it is, black on white, or a letter is still read as some digit.
- 6,000 training images is few: about 7 in every 100 test digits are read wrong.
- The embedding of 2 numbers is small for a digit reader. It is there so that the course can draw the embedding of every image as a point on a map.

## License

CC BY 4.0, by the Everything AI contributors. You may use, change and share the model, also to earn money, if you name the makers ("Everything AI contributors"), link to the license (https://creativecommons.org/licenses/by/4.0/) and say what you changed.
