# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/bike-sharing-day.csv.gz and data/bike-sharing-hour.csv.gz from Bike Sharing (UCI Machine Learning
Repository, dataset 275; Hadi Fanaee-T, 2013; CC BY 4.0): the bikes rented from Capital Bikeshare in Washington, D.C.,
per day and per hour in 2011 and 2012, with the weather, for the lessons on time series.

    python data/tools/make_bike_sharing.py path/to/bike+sharing+dataset.zip

The zip holds day.csv, hour.csv and Readme.txt. The two tables are kept as they are, byte for byte, and only
compressed: the file is written without a time in its gzip header, so the same table gives the same bytes. The
source asks publications to cite Fanaee-T and Gama (2013), Event labeling combining ensemble detectors and
background knowledge, Progress in Artificial Intelligence, doi:10.1007/s13748-013-0040-3."""
import gzip
import sys
import zipfile
from pathlib import Path


def main(source: str) -> None:
    data = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(source) as z:
        for name, out in (("day.csv", "bike-sharing-day.csv.gz"), ("hour.csv", "bike-sharing-hour.csv.gz")):
            raw = z.read(name)
            with open(data / out, "wb") as f, gzip.GzipFile(filename=name, mode="wb", fileobj=f, compresslevel=9, mtime=0) as g:
                g.write(raw)
            print(f"{name}: {len(raw)} bytes, {raw.count(b'\n') - 1} rows, written to {data / out}")


if __name__ == "__main__":
    main(sys.argv[1])
