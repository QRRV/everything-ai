# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/documents/: the handbook of Spoke & Chain, the made-up bike shop of the domain bouwen, as three files
of the same text, for build-documents-to-text: Markdown (spoke-and-chain.md), a web page with a menu, a cookie
banner, a style, a script and a footer around it (spoke-and-chain.html), and a PDF of one page in two columns
(spoke-and-chain.pdf). The course writes the PDF itself, without compression and with the 14 standard fonts that
every PDF reader has, so that a few lines of Python can read it; it draws the lines of both columns row by row, as
some programs do, so its drawing order is not its reading order. The texts are the course's own, under CC BY-SA 4.0.
Run it from the course folder:

    uv run python data/tools/make_documents.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "documents"

TITLE = "Spoke & Chain: the shop handbook"
SECTIONS = [
    ('Opening hours', "The shop is open from Tuesday to Friday from 9:00 to 18:00, and on Saturday from 9:00 to 17:00. It is closed on Sunday and Monday, and on public holidays. On the last Thursday of each month the shop stays open until 21:00 for the repair evening, where customers can fix their own bike with the shop's tools and a mechanic nearby."),
    ('Repairs', "A flat tire is fixed while you wait, in about twenty minutes, for 15 euros, the inner tube included. A full service costs 85 euros and takes two working days; it covers the brakes, the gears, the chain, the wheels and a safety check. Brake pads cost 12 euros per wheel, fitted. A new chain costs 35 euros, fitted. A wheel that is bent can often be trued for 20 euros; a new wheel starts at 70 euros. Electric bikes get a service for 120 euros, which includes a test of the battery and an update of the motor's software. Repairs are paid when the bike is collected. A bike that is not collected within thirty days after the repair is finished costs 2 euros a day for storage."),
    ('Returns', 'A product that has not been used can be returned within thirty days with the receipt, for the full price back on the card or account it was paid with. Used products, products made to measure, and inner tubes cannot be returned. A bike can be returned within fourteen days if it has been ridden less than 50 kilometers, minus 10 percent of the price for the inspection.'),
    ('Warranty', 'New bikes have five years of warranty on the frame and two years on all other parts, except parts that wear, such as tires, brake pads, chains and cables. Batteries of electric bikes have two years of warranty, or 500 charges, whichever comes first. Warranty only holds with the receipt and when the bike was serviced at least once a year. Damage from a fall, an accident or wrong use is not covered.'),
    ('Rentals', 'City bikes can be rented for 15 euros a day or 60 euros a week; electric bikes for 35 euros a day or 150 euros a week. Every rental includes a lock and a helmet. A deposit of 100 euros is paid when the bike is picked up and given back when it is returned undamaged. A rental bike that comes back late costs the price of an extra day. Rental bikes may not be taken abroad, and a lost lock costs 30 euros.'),
    ('Delivery', 'Bikes bought in the shop can be delivered at home within 15 kilometers for 25 euros, ready to ride. Parts ordered online arrive within three working days; orders above 50 euros are sent for free. Orders placed before 14:00 on a working day leave the same day.'),
    ('Payment', "The shop takes cards and mobile payments, and cash up to 500 euros. Bikes above 1,000 euros can be paid in twelve monthly parts without interest, after a check of the customer's identity. Gift cards are sold in the shop for any amount from 10 euros, and they are valid for two years."),
    ('Bike fitting', "A fitting takes an hour and costs 60 euros: a mechanic sets the saddle, the handlebars and the pedals to the customer's body and way of riding. Customers who buy a bike above 1,500 euros get the fitting for free within three months of the purchase."),
    ('Trade-in', 'The shop buys used bikes that are less than eight years old and in working order, for a price the mechanic sets after a check of twenty minutes. The price is paid as a voucher for the shop, which is 10 percent higher than the price in cash. Bikes without a frame number, and bikes with a cracked frame, are not bought.'),
    ("Children's bikes", "Children grow fast, so the shop takes back a children's bike that was bought new in the shop for 40 percent of its price, as a voucher, when a larger one is bought within three years. Children's helmets are fitted for free."),
    ('Safety', 'Every bike that leaves the shop after a sale or a service has working lights, a bell and reflectors. The shop does not sell or fit parts that change an electric bike so that its motor helps above 25 kilometers an hour, as that is not allowed on the road.'),
    ('Members', 'A membership costs 30 euros a year. Members get 10 percent off all parts and services, one free flat tire repair a year, and two hours earlier access to the yearly sale in March. A membership cannot be shared and ends after one year unless it is renewed.'),
]


def markdown() -> str:
    return f"# {TITLE}\n\n" + "\n\n".join(f"## {head}\n\n{text}" for head, text in SECTIONS) + "\n"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def web_page() -> str:
    body = "\n".join(f"<h2>{_escape(head)}</h2>\n<p>{_escape(text)}</p>" for head, text in SECTIONS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Handbook | Spoke &amp; Chain</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
nav a {{ margin-right: 1em; }}
</style>
<script>
window.visits = (window.visits || 0) + 1;
console.log("page view", window.visits);
</script>
</head>
<body>
<nav><a href="/">Home</a> <a href="/shop">Shop</a> <a href="/repairs">Repairs</a> <a href="/rentals">Rentals</a> <a href="/contact">Contact</a></nav>
<div class="cookies">We use cookies to make this site work. <button>Accept all</button> <button>Settings</button></div>
<main>
<h1>{_escape(TITLE)}</h1>
{body}
</main>
<footer>&copy; 2026 Spoke &amp; Chain &middot; <a href="/privacy">Privacy</a> &middot; <a href="/cookies">Cookies</a></footer>
</body>
</html>
"""


# --- the PDF ------------------------------------------------------------------------------------------------------------

WIDTH, HEIGHT = 595, 842              # A4, in points
LEFT, RIGHT_COLUMN = 50, 310          # where each column starts
TOP, LEADING, SIZE = 780, 12, 9       # the first line, the distance between lines, the size of the letters
CHARACTERS = 48                       # about what a line of the column holds at this size


def _wrap(text: str) -> list:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > CHARACTERS:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    return lines + [line] if line else lines


def pdf_lines() -> list:
    """Every line of the handbook in reading order, as (font, text): F2 bold for the title and the headings."""
    out = [("F2", TITLE), ("F1", "")]
    for head, text in SECTIONS:
        out.append(("F2", head))
        out += [("F1", line) for line in _wrap(text)]
        out.append(("F1", ""))
    return out[:-1]


def _pdf_string(text: str) -> str:
    return "(" + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def pdf() -> bytes:
    lines = pdf_lines()
    half = (len(lines) + 1) // 2
    left, right = lines[:half], lines[half:]
    ops = []
    for row in range(half):   # row by row: the left column's line, then the right column's
        y = TOP - row * LEADING
        for column, x in ((left, LEFT), (right, RIGHT_COLUMN)):
            if row < len(column) and column[row][1]:
                font, text = column[row]
                ops.append(f"BT /{font} {SIZE} Tf {x} {y} Td {_pdf_string(text)} Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] /Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>".encode(),
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spoke-and-chain.md").write_text(markdown(), encoding="utf-8")
    (OUT / "spoke-and-chain.html").write_text(web_page(), encoding="utf-8")
    (OUT / "spoke-and-chain.pdf").write_bytes(pdf())
    print("written", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
