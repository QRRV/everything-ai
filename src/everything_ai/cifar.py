# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""CIFAR-10: 60,000 color photos of 32 by 32 pixels in 10 kinds, 6,000 of each, 50,000 to train on and 10,000 to
test with, put together in 2009 by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton at the University of Toronto
(Krizhevsky 2009, *Learning Multiple Layers of Features from Tiny Images*). Its page states no licence, so the course
does not ship it and shares no model trained on it (course/route/bronnen.yaml, cifar-10-download-only):
everything_ai.hub downloads its binary version once, at its source, and `unpack` turns it into one file of arrays in
work/cache/cifar-10/ and then deletes the download. The binary version holds only bytes, so nothing in it is ever run. Some photos show a person
next to the object, such as a rider on a horse, so a lesson shows only test photos that were looked at one by one
(CHECKED); the course trains on all of them, as everyone who uses CIFAR-10 does."""
import functools
import os
import tarfile
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from everything_ai import hub, paths

TAR = ("cs.toronto.edu/kriz", "cifar-10-binary.tar.gz")
KINDS = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SIZE = 32
_RECORD = 1 + 3 * SIZE * SIZE   # one byte for the kind, then the red, green and blue pixels, row by row
_PER_FILE = 10_000
_TRAIN = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
_TEST = "cifar-10-batches-bin/test_batch.bin"
TRAIN, TEST = 50_000, 10_000

# Test photos, by their place in the test set, that were looked at one by one, enlarged seven times, and show no
# person and no part of one: the only photos of CIFAR-10 a lesson shows. Per kind, in the order of the test set. Of
# the first 14 to 28 photos of each kind, those with a hand, an arm, a lap, a rider or people on deck, or where that
# could not be ruled out, were left out; so were a book cover and a plane on fire. Cat 103 was left out later, after a
# second look: beside it, a blue cloth and a pale shape could be a sleeve and an arm.
CHECKED: dict[str, tuple[int, ...]] = {
    "airplane": (10, 21, 27, 44, 52, 90, 97, 98, 111, 116),
    "automobile": (6, 9, 66, 81, 82, 104, 105, 114, 131, 161),
    "bird": (35, 65, 67, 70, 75, 84, 86, 113, 118, 123),
    "cat": (53, 63, 91, 106, 115, 121, 143, 176, 184, 187),
    "deer": (22, 26, 32, 36, 40, 58, 94, 110, 117, 130),
    "dog": (24, 31, 33, 42, 85, 101, 141, 178, 181, 190),
    "frog": (4, 5, 7, 19, 29, 30, 41, 49, 62, 64),
    "horse": (13, 17, 48, 57, 60, 69, 83, 87, 99, 109),
    "ship": (1, 2, 54, 55, 72, 73, 79, 80, 92, 108),
    "truck": (11, 14, 23, 28, 34, 38, 45, 47, 76, 89),
}


class Damaged(OSError):
    """The downloaded archive is not what CIFAR-10 must be; `unpack` deleted it, so that the next try downloads it
    again."""


def folder() -> Path:
    return paths.WORK / "cache" / "cifar-10"


def _file() -> Path:
    return folder() / "cifar-10.npz"


def unpacked() -> bool:
    return _file().is_file()


def unpack() -> Path:
    """Read the downloaded archive once, as a stream, into work/cache/cifar-10/cifar-10.npz, and give that file. Only
    the six files of photos are read, each of exactly the size it must have; nothing is written outside that folder.
    Once the photos are in that file, the archive is deleted: 170 MB the lesson never needs again. It raises
    FileNotFoundError when the archive is not downloaded yet: hub.download(*TAR) gets it. An archive that turns out to
    be damaged is deleted, so that the next try downloads it again, and gives Damaged."""
    if unpacked():
        return _file()
    source = hub.pinned(*TAR)
    if source is None:
        raise FileNotFoundError("CIFAR-10 is not downloaded yet: everything_ai.hub.download(*everything_ai.cifar.TAR)")
    parts = {}
    try:
        with tarfile.open(source, mode="r|gz") as archive:
            for member in archive:
                if member.name in _TRAIN + [_TEST] and member.isfile() and member.size == _RECORD * _PER_FILE:
                    with archive.extractfile(member) as f:
                        parts[member.name] = np.frombuffer(f.read(member.size), dtype=np.uint8).reshape(_PER_FILE, _RECORD)
    except (tarfile.TarError, EOFError, zlib.error) as e:
        source.unlink(missing_ok=True)
        raise Damaged(f"the archive of CIFAR-10 is damaged, so the course deleted it: {e}") from e
    if set(parts) != set(_TRAIN + [_TEST]):
        source.unlink(missing_ok=True)
        raise Damaged(f"the archive of CIFAR-10 is damaged: it holds {len(parts)} of its 6 files of photos, "
                      "so the course deleted it")
    train = np.concatenate([parts[name] for name in _TRAIN])
    test = parts[_TEST]
    arrays = {"train_images": train[:, 1:].reshape(-1, 3, SIZE, SIZE), "train_labels": train[:, 0].copy(),
              "test_images": test[:, 1:].reshape(-1, 3, SIZE, SIZE), "test_labels": test[:, 0].copy()}
    for name in ("train_labels", "test_labels"):
        if arrays[name].max() >= len(KINDS) or (np.bincount(arrays[name]) != len(arrays[name]) // len(KINDS)).any():
            source.unlink(missing_ok=True)
            raise Damaged("the archive of CIFAR-10 is damaged: its kinds are not 10 equal groups, so the course "
                          "deleted it")
    folder().mkdir(parents=True, exist_ok=True)
    for old in folder().glob("cifar-10.*.npz.part"):   # left by a lesson that was stopped while it unpacked
        if time.time() - old.stat().st_mtime > 3600:
            old.unlink(missing_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder(), prefix="cifar-10.", suffix=".npz.part")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.chmod(tmp, 0o644)   # mkstemp makes a file only its owner may read
        os.replace(tmp, _file())   # in one step: a lesson never reads half the file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    source.unlink(missing_ok=True)
    return _file()


@functools.lru_cache(maxsize=1)
def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The training photos, their kinds, the test photos and their kinds: photos as bytes of shape (N, 3, 32, 32),
    channels first as PyTorch wants them, kinds as numbers 0 to 9 (KINDS). They are read-only: every lesson gets the
    same arrays, so a learner's code cannot change them for the next cell. It raises FileNotFoundError when the
    archive is not unpacked yet."""
    if not unpacked():
        raise FileNotFoundError("CIFAR-10 is not unpacked yet: everything_ai.cifar.unpack()")
    with np.load(_file()) as d:   # never a pickle: np.load refuses one by default
        arrays = d["train_images"], d["train_labels"], d["test_images"], d["test_labels"]
    for a in arrays:
        a.setflags(write=False)
    return arrays


def checked(kind: str, n: int = 8) -> list[int]:
    """The places in the test set of the first n checked photos of a kind."""
    if kind not in CHECKED or n > len(CHECKED[kind]):
        raise ValueError(f"only {len(CHECKED.get(kind, ()))} photos of {kind} were looked at, not {n}")
    return list(CHECKED[kind][:n])
