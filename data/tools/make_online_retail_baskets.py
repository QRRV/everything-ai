# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/online-retail-baskets.csv.gz from Online Retail (UCI Machine Learning Repository, dataset 352; Daqing
Chen, Sai Liang Sain and Kun Guo, 2012; CC BY 4.0): the orders of a British online shop from 1 December 2010 to
9 December 2011, for the lessons on association rules and on recommending.

    python data/tools/make_online_retail_baskets.py path/to/online+retail.zip

The zip holds one sheet, `Online Retail.xlsx`. It is read with the standard library alone (an .xlsx file is a zip of
XML), so the course needs no package for Excel. What is kept:

- no cancellations: an invoice number that starts with C;
- no rows with a quantity or a unit price of 0 or less: corrections and lost stock;
- only products: a stock code that starts with five digits; postage, discounts, bank charges, manual entries and the
  like have other codes;
- every product once per order: an order is a set of products (two stock codes with the same description count as
  one product);
- three columns: `invoice`, the order number; `customer`, the customer number the source already gave instead of a
  name, empty where it gave none; and `product`, the description the source gives that stock code most often, in
  capitals as it gives it, spaces at the ends removed. Date, quantity, price and country are left out.

The rows keep the order of the sheet. The file is written without a time in its gzip header, so the same sheet gives
the same bytes."""
import csv
import gzip
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def _column(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def rows(xlsx: zipfile.ZipFile):
    """Every row of the first sheet as a list of strings, the first row being the column names."""
    strings = []
    with xlsx.open("xl/sharedStrings.xml") as f:
        for _, el in ElementTree.iterparse(f):
            if el.tag == NS + "si":
                strings.append("".join(t.text or "" for t in el.iter(NS + "t")))
                el.clear()
    with xlsx.open("xl/worksheets/sheet1.xml") as f:
        for _, el in ElementTree.iterparse(f):
            if el.tag != NS + "row":
                continue
            out = [""] * len(COLUMNS)
            for c in el.iter(NS + "c"):
                v = c.find(NS + "v")
                text = c.find(NS + "is")
                if v is not None:
                    value = strings[int(v.text)] if c.get("t") == "s" else v.text
                elif text is not None:
                    value = "".join(t.text or "" for t in text.iter(NS + "t"))
                else:
                    value = ""
                i = _column(c.get("r"))
                if i < len(out):
                    out[i] = value or ""
            yield out
            el.clear()


def _number(text: str) -> str:
    """A whole number as the sheet stores it, such as 536365 or 17850.0, as text without decimals."""
    return str(int(float(text))) if re.fullmatch(r"-?\d+(\.0+)?", text.strip()) else text.strip()


def main(source: str) -> None:
    data = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(source) as outer:
        name = next(n for n in outer.namelist() if n.endswith(".xlsx"))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
    it = rows(inner)
    assert next(it) == COLUMNS, "the columns of the sheet are not the expected ones"
    kept, names = [], defaultdict(Counter)
    seen = set()
    total = 0
    for invoice, code, description, quantity, _, price, customer, _ in it:
        total += 1
        invoice, code = _number(invoice), code.strip().upper()
        if invoice.startswith("C") or float(quantity) <= 0 or float(price) <= 0 or not re.match(r"\d{5}", code):
            continue
        if description.strip():
            names[code][description.strip()] += 1
        kept.append((invoice, _number(customer) if customer.strip() else "", code))
    product = {code: c.most_common(1)[0][0] for code, c in names.items()}
    out = []
    for i, cu, co in kept:
        if co in product and (i, product[co]) not in seen:
            seen.add((i, product[co]))
            out.append((i, cu, product[co]))
    kept = out
    with open(data / "online-retail-baskets.csv.gz", "wb") as raw, \
            gzip.GzipFile(filename="online-retail-baskets.csv", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as z, \
            io.TextIOWrapper(z, encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["invoice", "customer", "product"])
        w.writerows(kept)
    print(f"{total} rows read, {len(kept)} kept, {len({i for i, _, _ in kept})} orders, {len(set(product.values()))} products,"
          f" written to {data / 'online-retail-baskets.csv.gz'}")


if __name__ == "__main__":
    main(sys.argv[1])
