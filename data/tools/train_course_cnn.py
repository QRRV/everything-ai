# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Trains the course's digit reader (everything_ai.models.DigitCNN) on the 6,000 training images of
data/qmnist-small.npz and writes its state_dict to data/course-qmnist-cnn.pt. Run from the course folder:

    uv run python data/tools/train_course_cnn.py

On one thread, with the same PyTorch, it gives the same weights every time; it takes about a minute."""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.models import DIGIT_CNN, DigitCNN

SEED, EPOCHS, LR, BATCH = 2, 30, 0.003, 64   # seed 2: every digit gets its own direction, none sits in the middle


def main():
    torch.set_num_threads(1)   # the same weights on every computer with the same PyTorch
    d = np.load(paths.ROOT / "data" / "qmnist-small.npz")
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    torch.manual_seed(SEED)
    model = DigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            right = int((model(X_test).argmax(dim=1) == y_test).sum())
        print(f"epoch {epoch + 1}: {right} of {len(y_test)} test digits right")
    torch.save(model.state_dict(), DIGIT_CNN)
    print(DIGIT_CNN.name, hashlib.sha256(DIGIT_CNN.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
