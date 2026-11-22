# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/wals-word-order.tsv.gz: three word-order features of the World Atlas of Language Structures (WALS) for
every language that has a value for at least one of them, with its family, genus, area and place on the map. Lessons
of domain taal use it to count and draw how the languages of the world order subject, verb and object.

WALS Online (Dryer and Haspelmath, eds., 2013) is released under CC BY 4.0 as a CLDF dataset on Zenodo. The course took
version v2020.6, published on September 28, 2026, DOI 10.5281/zenodo.23010827, the same version wals.info runs, and
pins its zip by the SHA-256 below: `wals-2020.6.zip` from https://zenodo.org/records/23010827. Download the zip into a
folder and run, from the course folder:

    uv run python data/tools/make_wals_word_order.py <folder>

The features, each with the name WALS gives its values:
- 81A Order of Subject, Object and Verb (Matthew S. Dryer): SOV, SVO, VSO, VOS, OVS, OSV or No dominant order;
- 81B Languages with two Dominant Orders of Subject, Object, and Verb (Dryer), such as SOV or SVO;
- 83A Order of Object and Verb (Dryer): OV, VO or No dominant order;
- 85A Order of Adposition and Noun Phrase (Dryer): Postpositions, Prepositions, Inpositions, No dominant order or No
  adpositions.

What the file holds: a header line, then one language per line, in the order of the WALS codes, with the columns
wals (the WALS code), name, family, genus, macroarea, latitude, longitude, glottocode, order (81A), two_orders (81B),
object_verb (83A) and adposition (85A), separated by tabs; a feature without a value for a language is empty. The rows
of families, subfamilies and genera in languages.csv are not languages, and stay out.
"""
import csv
import gzip
import hashlib
import io
import sys
import zipfile
from pathlib import Path

from everything_ai import paths

SOURCE = "https://zenodo.org/records/23010827/files/wals-2020.6.zip"
SHA256 = "25f12622c612dff848719b7502a2f043b4f88b7f2850db451bf15de7e4289faa"
FEATURES = {"81A": "order", "81B": "two_orders", "83A": "object_verb", "85A": "adposition"}
COLUMNS = ["wals", "name", "family", "genus", "macroarea", "latitude", "longitude", "glottocode", *FEATURES.values()]


def _table(zf: zipfile.ZipFile, name: str) -> list:
    with zf.open(f"wals-2020.6/cldf/{name}") as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")))


def main(folder: str) -> None:
    data = (Path(folder) / "wals-2020.6.zip").read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        languages = {r["ID"]: r for r in _table(zf, "languages.csv") if not r["ID"].startswith(("family-", "subfamily-", "genus-"))}
        codes = {r["ID"]: r["Name"] for r in _table(zf, "codes.csv") if r["Parameter_ID"] in FEATURES}
        values = {}
        for r in _table(zf, "values.csv"):
            if r["Parameter_ID"] in FEATURES:
                assert r["Language_ID"] in languages and r["Code_ID"] in codes
                values.setdefault(r["Language_ID"], {})[FEATURES[r["Parameter_ID"]]] = codes[r["Code_ID"]]
    rows = []
    for wals in sorted(values):
        lang = languages[wals]
        if not any(k in values[wals] for k in ("order", "object_verb", "adposition")):
            continue
        cells = [wals, lang["Name"], lang["Family"], lang["Genus"], lang["Macroarea"], lang["Latitude"],
                 lang["Longitude"], lang["Glottocode"]] + [values[wals].get(k, "") for k in FEATURES.values()]
        assert all("\t" not in c and "\n" not in c for c in cells)
        rows.append("\t".join(cells))
    out = paths.ROOT / "data" / "wals-word-order.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\t".join(COLUMNS) + "\n" + "\n".join(rows) + "\n").encode("utf-8"))
    counted = {k: sum(1 for v in values.values() if k in v) for k in FEATURES.values()}
    print(f"{len(rows):,} languages;", ", ".join(f"{k} {n:,}" for k, n in counted.items()))
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
