# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-documents-to-text: a function that takes the tags out of a web page, and one that puts the pieces
of text of a PDF in two columns in reading order; the handbook of Spoke & Chain as the course hosts it in three files
(data/documents/, made by data/tools/make_documents.py): Markdown, a web page and a PDF of one page in two columns;
the course's own small reader of that PDF, what the course measured on the three files, and the pictures."""
import functools
import html
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

FOLDER = paths.ROOT / "data" / "documents"
MARKDOWN = FOLDER / "spoke-and-chain.md"
PAGE = FOLDER / "spoke-and-chain.html"
PDF = FOLDER / "spoke-and-chain.pdf"
MIDDLE = 300   # the page is 595 points wide; the left column starts at 50, the right one at 310


def read(path) -> str:
    return path.read_text(encoding="utf-8")


@functools.cache
def handbook() -> dict:
    """The handbook as the Markdown file has it: its title, and every section's heading and paragraph."""
    blocks = read(MARKDOWN).strip().split("\n\n")
    title = blocks[0].removeprefix("# ")
    sections = [(blocks[i].removeprefix("## "), blocks[i + 1]) for i in range(1, len(blocks), 2)]
    return {"title": title, "sections": sections}


def handbook_lines() -> list:
    """The title, then every heading and paragraph, in reading order: the lines of text a reader should get."""
    h = handbook()
    return [h["title"]] + [line for section in h["sections"] for line in section]


def handbook_words() -> int:
    return sum(len(line.split()) for line in handbook_lines())


def sentences() -> list:
    """Every sentence of the handbook's paragraphs: cut after each full stop that a space follows."""
    out = []
    for _, text in handbook()["sections"]:
        for part in text.split(". "):
            out.append(part if part.endswith(".") else part + ".")
    return out


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the worked example: the Markdown file ------------------------------------------------------------------------------

MARKDOWN_CODE = '''with open("data/documents/spoke-and-chain.md") as f:
    text = f.read()
lines = []
for line in text.split("\\n"):
    lines.append(line.strip("# "))
clean = "\\n".join(lines)
'''


@functools.cache
def markdown_example() -> dict:
    """What the worked example gives: the words of the file as it is, and of the clean text."""
    got = run_code(MARKDOWN_CODE.replace('"data/documents/spoke-and-chain.md"', repr(str(MARKDOWN)))).names
    clean = got["clean"]
    return {"raw": len(got["text"].split()), "clean": len(clean.split()), "marks": got["text"].count("#"),
            "lines": [line for line in clean.split("\n") if line.strip()], "text": clean}


# --- exercise 1: html_text -----------------------------------------------------------------------------------------------

START_HTML = '''def html_text(page):
    out = ""
    for ch in page:
        out = out + ch
    return out
'''
SOLUTION_HTML = '''def html_text(page):
    out = ""
    inside = False
    for ch in page:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out = out + ch
    return out
'''
KEEPS_CLOSE = '''def html_text(page):
    out = ""
    inside = False
    for ch in page:
        if ch == "<":
            inside = True
        if ch == ">":
            inside = False
        if not inside:
            out = out + ch
    return out
'''
NEVER_OUT = SOLUTION_HTML.replace('        elif ch == ">":\n            inside = False\n', "")
html_text = run_code(SOLUTION_HTML).names["html_text"]
EXAMPLE_PAGE = "<h2>Repairs</h2>\n<p>A flat tire is <b>fixed</b> while you wait.</p>"


def page_lines(text: str) -> list:
    """The lines of a text that are not empty, without spaces at their ends, each with whether it is a line of the
    handbook: the same, once its entities such as &amp; are read as the characters they stand for."""
    known = set(handbook_lines())
    return [(line.strip(), html.unescape(line.strip()) in known) for line in text.split("\n") if line.strip()]


@functools.cache
def page_findings() -> dict:
    """What taking the tags out of the whole page gives: the words of the handbook and the other words, and the lines
    that are not the handbook; then the main part alone, with its entities read, and whether its words are the
    Markdown's."""
    page = read(PAGE)
    lines = page_lines(html_text(page))
    main = page.split("<main>")[1].split("</main>")[0]
    text = html.unescape(html_text(main))
    return {"page_words": len(page.split()), "kept": sum(len(t.split()) for t, ok in lines if ok),
            "other": sum(len(t.split()) for t, ok in lines if not ok), "other_lines": [t for t, ok in lines if not ok],
            "entities": len(re.findall(r"&[a-z]+;", html_text(page))), "main": text,
            "same": text.split() == markdown_example()["text"].split()}


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"

_PAGE_WORDS = {
    "en": {"title": "The text your html_text gets out of the web page, line by line", "kept": "the handbook",
           "other": "not the handbook", "run": "{n} lines of the handbook, {w} words", "run1": "1 line of the handbook, {w} words",
           "count": "{k} words of the handbook, {o} other words", "more": "and {n} more lines"},
    "nl": {"title": "De tekst die je html_text uit de webpagina haalt, regel voor regel", "kept": "het handboek",
           "other": "niet het handboek", "run": "{n} regels van het handboek, {w} woorden", "run1": "1 regel van het handboek, {w} woorden",
           "count": "{k} woorden van het handboek, {o} andere woorden", "more": "en nog {n} regels"},
}


def _swatch(color: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{color})"></span>{html.escape(label)}</span>')


def page_picture(seen) -> str:
    """The learner's html_text on the whole web page: every line that is not the handbook as a row of its own, as a
    mistake, and every run of lines of the handbook as one row, as input data, with its lines and words."""
    w = _PAGE_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("html_text") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        got = with_time_limit(lambda: fn(read(PAGE)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, str):
        return ""
    groups = []
    for text, ok in page_lines(got):
        if ok and groups and groups[-1][0]:
            groups[-1][1].append(text)
        else:
            groups.append((ok, [text]))
    rows = []
    for ok, texts in groups[:24]:
        if ok:
            words = sum(len(t.split()) for t in texts)
            label = (w["run1"] if len(texts) == 1 else w["run"]).format(n=len(texts), w=words)
            style = "background:var(--ea-data-wash);color:var(--ea-ink);border-left:3px solid var(--ea-data)"
        else:
            label = texts[0] if len(texts[0]) <= 90 else texts[0][:88] + "…"
            style = "background:var(--ea-loss-wash);color:var(--ea-loss-text);border-left:3px solid var(--ea-loss)"
        rows.append(f'<div style="margin:2px 0;padding:0.1rem 0.4rem;border-radius:2px;font-family:var(--ea-font-code);font-size:12px;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;{style}">{html.escape(label)}</div>')
    if len(groups) > 24:
        rows.append(f'<div style="{_LABEL}">{html.escape(w["more"].format(n=sum(len(t) for _, t in groups[24:])))}</div>')
    lines = page_lines(got)
    count = w["count"].format(k=number(sum(len(t.split()) for t, ok in lines if ok), seen.lang),
                              o=number(sum(len(t.split()) for t, ok in lines if not ok), seen.lang))
    foot = f'<div style="margin-top:0.3rem">{_swatch("data", w["kept"])}{_swatch("loss", w["other"])}<span style="{_LABEL}">{html.escape(count)}</span></div>'
    return yours.figure(w["title"], "".join(rows) + foot, alt=w["title"])


# --- exercise 2: left_first ----------------------------------------------------------------------------------------------

_PIECE = re.compile(r"BT /F\d+ [\d.]+ Tf ([\d.]+) ([\d.]+) Td \(((?:[^()\\]|\\.)*)\) Tj ET")


def pdf_pieces(path=PDF) -> list:
    """The course's own small reader of PDFs like the course's handbook: every piece of text the page draws, in the
    order it draws them, as (x, y, text), x from the left edge of the page and y from the bottom, in points. It only
    reads pages that draw each piece with one BT ... Tj ET, without compression, as data/tools/make_documents.py
    writes them; for other PDFs there are libraries such as pypdf."""
    stream = path.read_bytes().decode("latin-1")
    return [(float(x) if "." in x else int(x), float(y) if "." in y else int(y),
             re.sub(r"\\(.)", r"\1", text)) for x, y, text in _PIECE.findall(stream)]


def pdf_ops(n: int) -> list:
    """The first n lines of the page that draw text, as the PDF holds them."""
    stream = PDF.read_bytes().decode("latin-1")
    return [m.group(0) for m in _PIECE.finditer(stream)][:n]


def reading_order() -> list:
    """The pieces' texts in the order a reader reads them: the left column from top to bottom, then the right one."""
    pieces = pdf_pieces()
    return [t for x, y, t in sorted(pieces, key=lambda p: (p[0] >= MIDDLE, -p[1]))]


def whole_sentences(text: str) -> int:
    """How many of the handbook's sentences a text has whole, with its lines read as one text."""
    flat = " ".join(text.split())
    return sum(1 for s in sentences() if s in flat)


START_LEFT = '''def left_first(pieces, middle):
    lines = []
    for x, y, text in pieces:
        lines.append(text)
    return "\\n".join(lines)
'''
SOLUTION_LEFT = '''def left_first(pieces, middle):
    lines = []
    for x, y, text in pieces:
        if x < middle:
            lines.append(text)
    for x, y, text in pieces:
        if x >= middle:
            lines.append(text)
    return "\\n".join(lines)
'''
RIGHT_FIRST = SOLUTION_LEFT.replace("if x < middle", "if x > middle").replace("if x >= middle", "if x <= middle")
ONLY_LEFT = '''def left_first(pieces, middle):
    lines = []
    for x, y, text in pieces:
        if x < middle:
            lines.append(text)
    return "\\n".join(lines)
'''
left_first = run_code(SOLUTION_LEFT).names["left_first"]
EXAMPLE_PIECES = [(50, 700, "The shop is open"), (310, 700, "Bikes are delivered"),
                  (50, 688, "on Saturday."), (310, 688, "within 15 kilometers.")]


@functools.cache
def pdf_findings() -> dict:
    pieces = pdf_pieces()
    drawn = "\n".join(t for x, y, t in pieces)
    ordered = left_first(pieces, MIDDLE)
    return {"pieces": len(pieces), "left": sum(1 for x, y, t in pieces if x < MIDDLE), "sentences": len(sentences()),
            "drawn": whole_sentences(drawn), "ordered": whole_sentences(ordered),
            "same": ordered.split("\n") == reading_order(), "words": len(drawn.split())}


_ORDER_WORDS = {
    "en": {"title": "Where each line of the page lands in the text your left_first gives", "right": "in reading order",
           "wrong": "out of place", "count": "{k} of {n} lines in reading order", "first": "The first lines of your text:"},
    "nl": {"title": "Waar elke regel van de pagina terechtkomt in de tekst die je left_first geeft", "right": "op volgorde van lezen",
           "wrong": "op de verkeerde plek", "count": "{k} van de {n} regels op volgorde van lezen", "first": "De eerste regels van je tekst:"},
}


def order_picture(seen) -> str:
    """The learner's left_first on the PDF's pieces: the page in small, a bar per line where the page draws it, as input
    data when the line lands where a reader reads it and as a mistake when it does not; then the first lines."""
    w = _ORDER_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("left_first") if mine is not None else None
    if not callable(fn):
        return ""
    pieces = pdf_pieces()
    try:
        got = with_time_limit(lambda: fn(list(pieces), MIDDLE))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, str):
        return ""
    lines = got.split("\n")
    order = reading_order()
    place = {t: i for i, t in enumerate(order)}
    ok = {t: i < len(lines) and lines[i] == t for t, i in place.items()}
    top = max(y for x, y, t in pieces)
    bars = []
    for x, y, t in pieces:
        left = 4 + (x - 50) * 0.5
        bars.append(f'<span title="{html.escape(t)}" style="position:absolute;left:{left:.0f}px;top:{4 + (top - y) * 0.5:.0f}px;'
                    f'width:{min(124, len(t) * 2.6):.0f}px;height:4px;border-radius:1px;'
                    f'background:var(--ea-{"data" if ok[t] else "loss"})"></span>')
    height = 8 + (top - min(y for x, y, t in pieces)) * 0.5 + 4
    page = (f'<div style="position:relative;width:268px;height:{height:.0f}px;border:1px solid var(--ea-line);'
            f'border-radius:3px;background:var(--ea-inset);flex:none">{"".join(bars)}</div>')
    first = "".join(f'<div style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
                    f'color:var(--ea-{"ink" if i < len(order) and line == order[i] else "loss-text"})">{html.escape(line) or "&nbsp;"}</div>'
                    for i, line in enumerate(lines[:8]))
    k = sum(ok.values())
    side = (f'<div style="min-width:0;flex:1 1 14rem"><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(w["first"])}</div>{first}'
            f'<div style="margin-top:0.5rem">{_swatch("data", w["right"])}{_swatch("loss", w["wrong"])}</div>'
            f'<div style="{_LABEL};margin-top:0.2rem">{html.escape(w["count"].format(k=k, n=len(order)))}</div></div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start">{page}{side}</div>'
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture of the three files ---------------------------------------------------------------------------------------

_FILES_WORDS = {
    "en": {"title": "The start of the handbook in each of the three files", "markdown": "Markdown", "page": "Web page (HTML)",
           "pdf": "PDF, the lines that draw text", "text": "text", "markup": "what tells a program how to show the text",
           "size": "{b} bytes"},
    "nl": {"title": "Het begin van het handboek in elk van de drie bestanden", "markdown": "Markdown", "page": "Webpagina (HTML)",
           "pdf": "Pdf, de regels die tekst tekenen", "text": "tekst", "markup": "wat een programma zegt hoe de tekst eruitziet",
           "size": "{b} bytes"},
}

_MARK = "color:var(--ea-ink-3)"


def _cut(text: str, n: int = 44) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _marked_markdown(line: str) -> str:
    rest = line.lstrip("#")
    mark = line[:len(line) - len(rest)]
    return f'<span style="{_MARK}">{html.escape(mark)}</span>{html.escape(_cut(rest))}' if mark else html.escape(_cut(line))


def _marked_html(line: str) -> str:
    line = _cut(line)
    out = []
    for part in re.split(r"(<[^>]*>?)", line):
        if part.startswith("<"):
            out.append(f'<span style="{_MARK}">{html.escape(part)}</span>')
        else:
            out.append(html.escape(part))
    return "".join(out)


def _marked_pdf(line: str) -> str:
    m = re.match(r"(.*? Td \()(.*)(\) Tj ET)$", line)
    head, text, tail = m.groups()
    return (f'<span style="{_MARK}">{html.escape(head)}</span>{html.escape(_cut(text, 30))}'
            f'<span style="{_MARK}">{html.escape(tail)}</span>')


def files_picture(lang: str) -> str:
    """The start of the handbook in each file, the text in ink and the markup in grey, with each file's size."""
    w = _FILES_WORDS[lang]
    md = read(MARKDOWN).split("\n")[:7]
    page = read(PAGE).split("\n")
    start = next(i for i, line in enumerate(page) if line.startswith("<nav>"))
    page = page[start:start + 7]
    ops = pdf_ops(7)
    panels = []
    for key, path, lines, mark in (("markdown", MARKDOWN, md, _marked_markdown), ("page", PAGE, page, _marked_html),
                                   ("pdf", PDF, ops, _marked_pdf)):
        body = "".join(f'<div style="white-space:nowrap;overflow:hidden">{mark(line) or "&nbsp;"}</div>' for line in lines)
        panels.append(f'<div style="flex:1 1 15rem;min-width:0"><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(w[key])}, '
                      f'{html.escape(w["size"].format(b=number(path.stat().st_size, lang)))}</div>'
                      f'<div style="font-family:var(--ea-font-code);font-size:11px;line-height:1.5;color:var(--ea-ink);background:var(--ea-inset);'
                      f'border:1px solid var(--ea-line);border-radius:3px;padding:0.3rem 0.4rem;overflow:hidden">{body}</div></div>')
    key = (f'<div style="margin-top:0.4rem">'
           f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}"><span style="font-family:var(--ea-font-code);color:var(--ea-ink)">Abc</span>{html.escape(w["text"])}</span>'
           f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}"><span style="font-family:var(--ea-font-code);{_MARK}">&lt;p&gt;</span>{html.escape(w["markup"])}</span></div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:0.8rem">{"".join(panels)}</div>{key}'
    return yours.figure(w["title"], body, alt=w["title"])


# --- the checks ---------------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "html_text": FunctionBox(
        name="html_text",
        picture=page_picture,
        function="html_text",
        start=START_HTML,
        solution=SOLUTION_HTML,
        visible=[(EXAMPLE_PAGE,)],
        hidden=[("No tags at all.",), ("",), ('<a href="/shop">Shop</a> &amp; more',), ("<br><br>",),
                ('<p class="note">one</p><p>two</p>',)],
        mistakes=[
            Mistake("keeps-close", wrong=KEEPS_CLOSE,
                    en="`{call}` gives `{got}`: every `>` stays in the text. At a `>`, your function sets `inside` back to `False` and then, in the same round, adds the character. Use `elif`, so that a round does only one of the three things.",
                    nl="`{call}` geeft `{got}`: elke `>` blijft in de tekst staan. Bij een `>` zet je functie `inside` terug op `False` en voegt ze in dezelfde ronde het teken toe. Gebruik `elif`, zodat een ronde maar één van de drie dingen doet."),
            Mistake("never-out", wrong=NEVER_OUT,
                    en="`{call}` gives `{got}`: after the first `<`, nothing comes through any more. Set `inside` back to `False` at every `>`, where a tag ends.",
                    nl="`{call}` geeft `{got}`: na de eerste `<` komt er niets meer door. Zet `inside` bij elke `>`, waar een tag eindigt, terug op `False`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`html_text` gives the page as it is, with all its tags. Keep track of whether you are inside a tag, from a `<` to the next `>`, and add only the characters outside.",
                      "nl": "`html_text` geeft de pagina zoals hij is, met al zijn tags. Houd bij of je binnen een tag bent, van een `<` tot de volgende `>`, en voeg alleen de tekens daarbuiten toe."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: every character outside the tags, and nothing of a tag, from its `<` up to and including its `>`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: elk teken buiten de tags, en niets van een tag, van zijn `<` tot en met zijn `>`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a text without tags, for an empty text, and for a tag with something inside it, such as `href`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een tekst zonder tags, voor een lege tekst, en voor een tag met iets erin, zoals `href`?"},
            "no_function": {"en": "Your box does not make a function `html_text`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `html_text`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `html_text` is right in all {n} cases: every character outside the tags, and nothing of the tags themselves.",
                        "nl": "Je `html_text` klopt in alle {n} gevallen: elk teken buiten de tags, en niets van de tags zelf."},
        },
    ),
    "left_first": FunctionBox(
        name="left_first",
        picture=order_picture,
        function="left_first",
        start=START_LEFT,
        solution=SOLUTION_LEFT,
        visible=[(EXAMPLE_PIECES, MIDDLE)],
        hidden=[([], MIDDLE), ([(50, 700, "only"), (50, 688, "left")], MIDDLE), ([(310, 700, "right"), (50, 700, "left")], MIDDLE),
                ([(300, 10, "at the middle"), (299, 10, "just left")], MIDDLE), ([(20, 5, "a"), (120, 5, "b"), (20, 1, "c")], 100)],
        mistakes=[
            Mistake("right-first", wrong=RIGHT_FIRST,
                    en="`{call}` gives `{got}`: the right column comes first. A piece is in the left column when its `x` is smaller than `middle`: take those first.",
                    nl="`{call}` geeft `{got}`: de rechterkolom komt eerst. Een stuk staat in de linkerkolom als zijn `x` kleiner is dan `middle`: neem die eerst."),
            Mistake("only-left", wrong=ONLY_LEFT,
                    en="`{call}` gives `{got}`: the left column is there, but the right column is gone. After the left column, go through the pieces once more for those with `x >= middle`.",
                    nl="`{call}` geeft `{got}`: de linkerkolom staat er, maar de rechterkolom is weg. Ga na de linkerkolom nog een keer de stukken langs, voor die met `x >= middle`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`left_first` gives the lines in the order the page draws them, left and right mixed. Take first the pieces with `x` smaller than `middle`, then the others.",
                      "nl": "`left_first` geeft de regels in de volgorde waarin de pagina ze tekent, links en rechts door elkaar. Neem eerst de stukken met een `x` kleiner dan `middle`, dan de andere."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: first the texts of the pieces with `x` smaller than `middle`, in their order, then the others, in their order, one per line.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: eerst de teksten van de stukken met een `x` kleiner dan `middle`, op hun volgorde, dan de andere, op hun volgorde, één per regel."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work without pieces, with only a left column, and for a piece right at `middle`, which belongs to the right column?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie zonder stukken, met alleen een linkerkolom, en voor een stuk precies op `middle`, dat bij de rechterkolom hoort?"},
            "no_function": {"en": "Your box does not make a function `left_first`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `left_first`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `left_first` is right in all {n} cases: the left column from top to bottom, then the right one.",
                        "nl": "Je `left_first` klopt in alle {n} gevallen: de linkerkolom van boven naar beneden, dan de rechter."},
        },
    ),
}

REQUIRED = ["html_text", "left_first", "scan"]

CASES = [
    ("html_text", START_HTML, "empty", None),
    ("html_text", "", "empty", None),
    ("html_text", SOLUTION_HTML, "correct", None),
    ("html_text", "def html_text(page):\n    out = \"\"\n    depth = 0\n    for ch in page:\n        if ch == \"<\":\n            depth = 1\n"
                  "        elif ch == \">\":\n            depth = 0\n        elif depth == 0:\n            out = out + ch\n    return out\n", "correct", None),
    ("html_text", KEEPS_CLOSE, "wrong", "keeps-close"),
    ("html_text", NEVER_OUT, "wrong", "never-out"),
    ("html_text", SOLUTION_HTML.replace("        elif not inside:\n            out = out + ch\n", "        else:\n            out = out + ch\n"), "wrong", None),
    ("html_text", SOLUTION_HTML.replace('    inside = False\n    for', '    inside = True\n    for'), "almost", None),
    ("left_first", START_LEFT, "empty", None),
    ("left_first", SOLUTION_LEFT, "correct", None),
    ("left_first", "def left_first(pieces, middle):\n    left = []\n    right = []\n    for x, y, text in pieces:\n        if x < middle:\n"
                   "            left.append(text)\n        else:\n            right.append(text)\n    return \"\\n\".join(left + right)\n", "correct", None),
    ("left_first", RIGHT_FIRST, "wrong", "right-first"),
    ("left_first", ONLY_LEFT, "wrong", "only-left"),
    ("left_first", SOLUTION_LEFT.replace("if x >= middle", "if x > middle"), "almost", None),
]
