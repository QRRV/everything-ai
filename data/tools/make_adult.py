# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/adult.csv.gz from Adult (UCI Machine Learning Repository, dataset 2; Barry Becker and Ronny Kohavi,
1996; CC BY 4.0): 48,842 people from the 1994 US Current Population Survey, with whether they earned more than
50,000 dollars a year, for the lessons on fairness, explaining models and privacy of the domain verantwoord.

    python data/tools/make_adult.py path/to/adult.zip

The zip holds adult.data (the 32,561 rows of the training set) and adult.test (the 16,281 rows of the test set),
without a header, with a comma and a space between the values and a question mark where a value is unknown. The
test set starts with a line that is not a row, and its labels end in a full stop. This script writes both sets as
one table with a header: the names of adult.names with an underscore for a hyphen, `income` for the label (which
adult.names leaves without a name), and a last column `split` (`train` or `test`). A value is kept as it is, without
the space in front; an unknown value is an empty cell, and the full stop after a label of the test set is left out. Nothing else changes, rows keep their order. The file is
written without a time in its gzip header, so the same zip gives the same bytes."""
import csv
import gzip
import hashlib
import io
import sys
import zipfile
from pathlib import Path

ZIP_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
           "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income"]


def rows(text: str, split: str):
    for line in text.splitlines():
        if not line.strip() or line.startswith("|"):
            continue
        values = [v.strip() for v in line.split(",")]
        assert len(values) == len(COLUMNS), line
        values[-1] = values[-1].rstrip(".")
        assert values[-1] in ("<=50K", ">50K"), line
        yield ["" if v == "?" else v for v in values] + [split]


def main(source: str) -> None:
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ZIP_SHA256, "not the pinned adult.zip"
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS + ["split"])
    counts = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name, split in (("adult.data", "train"), ("adult.test", "test")):
            for row in rows(z.read(name).decode("ascii"), split):
                w.writerow(row)
                counts[split] = counts.get(split, 0) + 1
    assert counts == {"train": 32561, "test": 16281}, counts
    target = Path(__file__).resolve().parents[1] / "adult.csv.gz"
    with open(target, "wb") as f, gzip.GzipFile(filename="adult.csv", mode="wb", fileobj=f, compresslevel=9, mtime=0) as g:
        g.write(out.getvalue().encode("ascii"))
    print(f"{counts['train']} train and {counts['test']} test rows written to {target}")
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
