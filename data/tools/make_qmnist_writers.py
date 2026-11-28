# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/qmnist-writers.npz from two files of QMNIST (facebookresearch/qmnist, branch master): who wrote the
1,000 test digits of data/qmnist-small.npz, and 1,000 more test digits written by government employees.

    python data/tools/make_qmnist_writers.py FOLDER

FOLDER holds qmnist-test-images-idx3-ubyte.gz and qmnist-test-labels-idx2-int.gz; their SHA-256 is checked first.
A label row of QMNIST has the digit, the NIST series (0 and 1 written by government employees, 4 by high-school
students), the writer's number and more. The 1,000 test digits of qmnist-small.npz are the first 1,000 of the test
set, all of series 4. The employees' digits are 1,000 picked at random, with seed 0, from the test digits of series
0 and 1 from number 10,000 on, the part of QMNIST that is not in MNIST's test set; the first ones of series 0 are
from only ten writers. The result is written to data/, this folder's parent."""
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np

FILES = {
    "qmnist-test-images-idx3-ubyte.gz": "43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
    "qmnist-test-labels-idx2-int.gz": "9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
}
STUDENTS, EMPLOYEES = 1000, 1000


def main(folder: Path) -> None:
    for name, digest in FILES.items():
        got = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if got != digest:
            sys.exit(f"{name}: SHA-256 {got}, expected {digest}")
    images = np.frombuffer(gzip.decompress((folder / "qmnist-test-images-idx3-ubyte.gz").read_bytes()),
                           dtype=np.uint8, offset=16).reshape(-1, 784)
    raw = gzip.decompress((folder / "qmnist-test-labels-idx2-int.gz").read_bytes())
    rows, columns = np.frombuffer(raw[4:12], dtype=">i4")
    labels = np.frombuffer(raw[12:], dtype=">i4").reshape(rows, columns)
    assert (labels[:STUDENTS, 1] == 4).all()
    pool = np.where(((labels[:, 1] == 0) | (labels[:, 1] == 1)) & (np.arange(rows) >= 10000))[0]
    pick = np.sort(np.random.default_rng(0).choice(pool, EMPLOYEES, replace=False))
    out = Path(__file__).resolve().parents[1] / "qmnist-writers.npz"
    np.savez_compressed(out,
                        student_writers=labels[:STUDENTS, 2].astype(np.int16),
                        employee_images=images[pick],
                        employee_labels=labels[pick, 0].astype(np.uint8),
                        employee_series=labels[pick, 1].astype(np.uint8),
                        employee_writers=labels[pick, 2].astype(np.int16),
                        employee_index=pick.astype(np.int32))
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
