# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/fashion-mnist-small.npz from the four files of Fashion-MNIST (zalandoresearch/fashion-mnist, folder
data/fashion): the first 6,000 training images and the first 1,000 test images, as NumPy arrays of 0 to 255.

    python data/tools/make_fashion_small.py FOLDER

FOLDER holds train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz and
t10k-labels-idx1-ubyte.gz; their SHA-256 is checked first. The result is written to data/, this folder's parent."""
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np

FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
TRAIN, TEST = 6000, 1000


def _read(path: Path, offset: int) -> np.ndarray:
    return np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8, offset=offset)


def main(folder: Path) -> None:
    for name, digest in FILES.items():
        got = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if got != digest:
            sys.exit(f"{name}: SHA-256 {got}, expected {digest}")
    train_images = _read(folder / "train-images-idx3-ubyte.gz", 16).reshape(-1, 784)[:TRAIN]
    train_labels = _read(folder / "train-labels-idx1-ubyte.gz", 8)[:TRAIN]
    test_images = _read(folder / "t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784)[:TEST]
    test_labels = _read(folder / "t10k-labels-idx1-ubyte.gz", 8)[:TEST]
    out = Path(__file__).resolve().parents[1] / "fashion-mnist-small.npz"
    np.savez_compressed(out, train_images=train_images, train_labels=train_labels,
                        test_images=test_images, test_labels=test_labels)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
