# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-documents-to-text. Route: course/route/lessen/bouwen.yaml. Checks, the files and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-documents-to-text", "en")
    C = L.checks   # the three files, the course's PDF reader, what the course measured, and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Reading a text file](/course/open/py-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_files():
    quick_files = L.choice(
        "quick-files",
        question="The file `notes.txt` has three lines. What is in `text` after this code?\n\n```python\nwith open(\"notes.txt\") as f:\n    text = f.read()\n```",
        options={
            "lines": "A list with one string for each of the three lines",
            "string": "One string: the whole file",
            "first": "The first of the three lines, as a string",
        },
        correct="string",
        why={
            "string": "`f.read()` gives the whole file as one string, with a newline where each line ends. That is where every file in this lesson starts.",
            "lines": "`f.read()` makes no list: it gives the whole file as one string, newlines and all. Going through the lines one at a time is what `for line in f:` does. See [Reading a text file](/course/open/py-files?lang=en).",
            "first": "`f.read()` reads the whole file, not one line: one string with all three lines, and a newline where each one ends. See [Reading a text file](/course/open/py-files?lang=en).",
        },
    )
    quick_files
    return (quick_files,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Your first RAG](/course/open/build-rag-first?lang=en), the points the search went through were text already. What a company knows is in files: a handbook as a PDF, pages on its website, notes in Markdown, a scanned contract. A search goes through text and a model reads text, so the first step of every RAG system is getting the text out of each file. When that step goes wrong, nothing warns you: the search finds the cookie banner, and the model reads sentences made of two columns.

    **An everyday example.** Copy the text of a web page into an email, and the menu and *Accept all* come along. Copy it from a PDF in two columns, and you can get a line of the left column, then a line of the right one, and so on. That is only a comparison; below you see what really comes out of three files with the same text.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **document parsing** | getting the text, and what belongs together in it, out of a file such as a web page or a PDF |
    | **ingestion** | everything that takes documents into a RAG system: collecting them, parsing, cleaning, cutting into pieces and storing them for the search; again whenever a document changes |
    | **markup** | what a file holds besides its text to say how to show it, such as `#` for a heading in Markdown |
    | **tag** | markup in a web page, from a `<` to the next `>`: `<p>` starts a paragraph and `</p>` ends it |
    | **entity** | a code in a web page for a character, such as `&amp;` for `&` |
    | **boilerplate** | the text that is on a page but not part of what it is about: the menu, a cookie banner, the footer |
    | **reading order** | the order in which a person reads the text on a page |
    | **OCR** | optical character recognition: software that recognizes the letters in a picture of text |

    ### Three files, one handbook

    The course has the handbook of a made-up bike shop, Spoke & Chain, in three files in the folder `data/documents`. The text is the same in each: {words} words, a title and twelve sections, each with a heading and a paragraph. What is around the text differs:
    """.format(words=C.handbook_words()))
    return


@app.cell(hide_code=True)
def picture_files():
    mo.Html(C.files_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_files():
    mo.md(r"""
    - **Markdown** is text with a little markup: `#` before the title and `##` before a heading. The text comes out cleanly.
    - **A web page** (HTML) has its text between tags. And a page holds more than its article: a title for the browser tab, a style that says how it looks, a script that runs in the browser, a menu, a cookie banner and a footer. Taking out the tags keeps all of that text.
    - **A PDF** is a list of instructions to draw: *this piece of text, in this font, at this place on the page*. It has no paragraphs, no columns and no tables, only pieces with a place, and it may draw them in any order. A table comes out as loose pieces, and which number belongs in which column is gone, unless a tool finds the table by its lines.
    - **A scan** is a picture of each page. There is no text in it at all, until OCR recognizes the letters.

    Tools that do this (examples in November 2026): Python's own `html.parser` and the library BeautifulSoup take a web page apart into its tags; trafilatura finds the main text of a page; pypdf and pdfplumber read PDFs, and pdfplumber also finds tables; Docling and Unstructured read many kinds of files at once; Tesseract is free OCR software. In this lesson you do the core of it yourself, on the course's three files.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: from a file to text

    1. **Read the file** into one string.
    2. **Take out what is not text:** markup, tags, and the parts around the main text.
    3. **Put the pieces in reading order,** where the file stores them in another order.
    4. **Check against the original:** count the words, and read a few pieces of every kind of file yourself.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _m = C.markdown_example()
    mo.md(r"""
    ### Worked example

    The Markdown file first:

    ```python
    {code}
    ```

    1. **Read the file:** `f.read()` gives the whole file as one string, {raw} words when you split it at the spaces. `text.split("\n")` cuts it at every line break, `\n`, instead, so the loop goes through its lines.
    2. **Take out what is not text:** a `#` is a word of its own here, so {marks} of those words are only markup: one `#` for the title and twelve `##` for the headings. `strip` with characters between its parentheses takes those characters off both ends of a string, here hash signs and spaces: `"## Repairs".strip("# ")` gives `"Repairs"`. A line without them stays as it is.
    3. **Reading order:** a Markdown file stores its text in the order you read it. Nothing to do.
    4. **Check:** `clean` has {clean} words, the {words} words of the handbook, and its first lines are *{first}* and *{second}*.
    """.format(code=C.MARKDOWN_CODE.strip(), raw=_m["raw"], marks=_m["raw"] - _m["clean"], clean=_m["clean"], words=C.handbook_words(),
               first=_m["lines"][0], second=_m["lines"][1]))
    return


@app.cell(hide_code=True)
def text_html():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Taking out the tags

        Write `html_text(page)`, which gives every character of `page` that is outside the tags. A tag runs from a `<` to the next `>`, and those two belong to the tag. Keep track with `inside` whether you are in a tag: `True` from a `<`, `False` again after the `>`. `not inside` is `True` when `inside` is `False`. The box adds every character, tags and all.
        """),
        L.task("Add a character only when you are not inside a tag, and leave out the `<` and the `>` themselves."),
    ])
    return


@app.cell(hide_code=True)
def exercise_html():
    box_html = L.code_box("html_text", lines=11)
    box_html
    return (box_html,)


@app.cell(hide_code=True)
def check_html(box_html):
    result_html = L.check("html_text", box_html.value)
    result_html
    return (result_html,)


@app.cell(hide_code=True)
def hints_html():
    L.hints(
        "html_text",
        [
            "Each round of the loop does one of three things: a `<` starts a tag, a `>` ends one, and any other character is added, but only outside a tag.",
            "Start with `inside = False`. Then `if ch == \"<\":` sets `inside = True`, `elif ch == \">\":` sets `inside = False`, and the third case adds `ch`.",
            "The third case: `elif ___:`, with `out = out + ch` under it.",
        ],
        why="""```python
def html_text(page):
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
```

With `elif`, a round does only one of the three things. With three separate `if`s, the `>` would end the tag and then be added in the same round.""",
    )
    return


@app.cell(hide_code=True)
def text_main():
    _p = C.page_findings()
    mo.vstack([
        mo.md(r"""
        ### What else is on the page

        Once exercise 1 is right, the picture under it shows your `html_text` on the whole web page. The text a right `html_text` gets out of it has the {kept} words of the handbook, and {other} words that are not: the title for the browser tab, the style, the script, the menu, the cookie banner and the footer, {lines} lines. And {entities} entities, such as `&amp;` where the page means `&`. Two more steps clean that up:

        ```python
        main = page.split("<main>")[1].split("</main>")[0]
        text = html.unescape(html_text(main))
        ```

        `split` with a text between its parentheses cuts at that text instead of at the spaces, so `[1]` is everything after `<main>`, and from that, `[0]` everything before `</main>`: only the page's main part. `html.unescape`, from Python's own module `html`, turns every entity into the character it stands for. What is left has {words} words: the same words as the Markdown file, {same}.
        """.format(kept=C.number(_p["kept"], L.lang), other=_p["other"], lines=len(_p["other_lines"]), entities=_p["entities"],
                   words=len(_p["main"].split()), same="in the same order" if _p["same"] else "but not in the same order")),
        L.watch_out("Many pages have no `<main>`, or put the menu inside it. Tools such as trafilatura then guess the main text, for instance from where a page has much text and few links. That guess can be wrong, so read a few cleaned pages of every website yourself."),
    ])
    return


@app.cell(hide_code=True)
def text_pdf():
    _f = C.pdf_findings()
    _first = C.pdf_pieces()[:4]
    mo.md(r"""
    ### What a PDF holds

    The PDF has the handbook on one page, in two columns. These are the first lines of the page that draw text, as the PDF holds them:

    ```
    {ops}
    ```

    `BT` and `ET` begin and end a piece of text, `/F1 9 Tf` picks the first font at 9 points, `Td` moves to a place on the page, `x` points from the left edge and `y` from the bottom (a point is 1/72 of an inch), and `Tj` draws the text between the parentheses there. The course's own small reader, `C.pdf_pieces()`, gives every piece the page draws as `(x, y, text)`, in the order the page draws them: {n} pieces, of which these are the first four:

    ```python
    {first}
    ```

    This page draws row by row: a line of the left column, at `x` 50, then the line of the right column at the same height, at `x` 310. Read in that order, only {drawn} of the {sentences} sentences of the handbook stay whole. A reader reads the left column from top to bottom first, then the right one, and within each column the page draws from top to bottom.
    """.format(ops="\n".join(C.pdf_ops(4)), n=_f["pieces"], first="\n".join(repr(p) for p in _first), drawn=_f["drawn"], sentences=_f["sentences"]))
    return


@app.cell(hide_code=True)
def text_left():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Two columns

        Write `left_first(pieces, middle)`, which gives the texts of the pieces in reading order, one per line: first those with an `x` smaller than `middle`, in their order, then the others, in their order. `for x, y, text in pieces:` gives each value of a piece its own name, as `x, y = point` does. The box gives the texts in the order the page draws them.
        """),
        L.task("Go through the pieces twice: once for the left column, once for the right one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_left():
    box_left = L.code_box("left_first", lines=9)
    box_left
    return (box_left,)


@app.cell(hide_code=True)
def check_left(box_left):
    result_left = L.check("left_first", box_left.value)
    result_left
    return (result_left,)


@app.cell(hide_code=True)
def hints_left():
    L.hints(
        "left_first",
        [
            "One loop gives the order the page draws in. Two loops over the same pieces, each keeping only one column, give the left column before the right one.",
            "In the first loop, `append` the text only for a piece of the left column, with an `x` smaller than `middle`; in a second loop after it, only for the others.",
            "The second loop: `for x, y, text in pieces:`, then `if ___:` with `lines.append(text)` under it.",
        ],
        why="""```python
def left_first(pieces, middle):
    lines = []
    for x, y, text in pieces:
        if x < middle:
            lines.append(text)
    for x, y, text in pieces:
        if x >= middle:
            lines.append(text)
    return "\\n".join(lines)
```

Each loop keeps the order of the pieces, and on this page each column is drawn from top to bottom, so each column comes out in the order you read it.""",
    )
    return


@app.cell(hide_code=True)
def text_after_pdf():
    _f = C.pdf_findings()
    mo.vstack([
        mo.md(r"""
        In reading order, all {ordered} sentences are whole, and the {n} lines are exactly those a reader reads, in the same order.
        """.format(ordered=_f["ordered"], n=_f["pieces"])),
        L.watch_out("This page draws each column from top to bottom, and the course knows where the middle is. Other PDFs draw in any order, have one column on one page and three on the next, or a heading across both columns. Real readers sort the pieces by their place and find the columns themselves, and they can still get it wrong: read a few pages of every kind of PDF yourself."),
    ])
    return


@app.cell(hide_code=True)
def text_scan():
    mo.md(r"""
    ### Exercise 3. A scan
    """)
    return


@app.cell(hide_code=True)
def choice_scan_cell():
    choice_scan = L.choice(
        "scan",
        question="The shop also has its old rental contract, scanned as a PDF: a photo of each page. What does it need before a RAG system can search it?",
        options={
            "reader": "Nothing extra: a PDF reader such as the course's gives its pieces of text, as with the handbook",
            "ocr": "OCR, which recognizes the letters in the picture of each page",
            "tags": "Taking out the tags first, as with a web page",
        },
        correct="ocr",
        why={
            "ocr": "A scan holds only a picture of each page, no text. OCR software such as Tesseract recognizes the letters; then check a sample, because OCR makes mistakes, such as a 1 for an l.",
            "reader": "A PDF reader finds the pieces of text a page draws, and a scan draws none: it only shows a picture of each page. OCR has to recognize the letters first.",
            "tags": "Tags are markup in a web page; a scan has none, and no text either, only a picture of each page. OCR has to recognize the letters first.",
        },
    )
    choice_scan
    return (choice_scan,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The search of [Your first RAG](/course/open/build-rag-first?lang=en) needs text, and now you can get it out of three kinds of files. A whole handbook is still one long text; [Cutting documents into pieces](/course/open/build-chunking?lang=en) cuts documents into the pieces a search goes through.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scan, result_html, result_left):
    L.wrap_up(result_html, result_left, choice_scan)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
