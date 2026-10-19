# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-text-as-numbers. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-text-as-numbers", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Text: strings](/course/open/py-strings?lang=en) and [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="What does `len(\"ice tea\")` give?",
        options={
            "letters": "`6`",
            "characters": "`7`",
            "words": "`2`",
        },
        correct="characters",
        why={
            "letters": "That counts only the letters. Is a space a character too? See the table of new words in [Text: strings](/course/open/py-strings?lang=en).",
            "words": "That is the number of words. What does `len` count in a string? See the table of new words in [Text: strings](/course/open/py-strings?lang=en).",
            "characters": "Six letters and one space: seven characters. A space counts too.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="What does `[w[0] for w in [\"ice\", \"tea\"]]` give?",
        options={
            "text": "`\"it\"`",
            "words": "`[\"ice\", \"tea\"]`",
            "firsts": "`[\"i\", \"t\"]`",
        },
        correct="firsts",
        why={
            "text": "A comprehension between square brackets makes a list, not a text. What goes in it, for each `w`? See the recipe in [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).",
            "words": "`w[0]` is not the whole word. What does an index give? See the table of new words in [Text: strings](/course/open/py-strings?lang=en).",
            "firsts": "`w[0]` is the first character of each word, and each one goes into the list.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A computer stores only numbers, and a model calculates only with numbers. So before a language model can read a sentence, the sentence has to become numbers. That is the first step of every chatbot: the text you type becomes a list of whole numbers, and its answer comes out as numbers that become text again.

    **An everyday example.** The secret code from school: a = 1, b = 2, c = 3, and so on. You write a word as a row of numbers, and a friend with the same table turns them back into the word.
    """)
    return


@app.cell(hide_code=True)
def predict_codes():
    guess_codes = L.guess(
        "abc-numbers",
        question="Python has a number for every character. What do you think `[ord(ch) for ch in \"abc\"]` gives?",
        options={
            "one": "`[1, 2, 3]`",
            "zero": "`[0, 1, 2]`",
            "unicode": "`[97, 98, 99]`",
        },
        reveal="**`[97, 98, 99]`.** `ord` gives each character's number in Unicode. The small letters start at 97, because the numbers before them went to other characters first: the space is 32, the digits are 48 to 57, and the capitals 65 to 90. The letters do follow each other: a, b and c are 97, 98 and 99.",
    )
    guess_codes
    return (guess_codes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## The idea

    Unicode is a worldwide list that gives every character its own number: every letter, digit, symbol and space, in every writing system, emoji too. `ord` gives the number of a character, and `chr` gives the character back. `for` goes through a text one character at a time, so a text becomes a list of numbers, and a list of numbers becomes text again. To store a text in a file, the computer needs one more step: a character encoding turns each character into bytes. A byte only goes up to 255, and many code points are bigger: `€` is 8364. So UTF-8, the encoding nearly everyone uses, spreads a character over one to four bytes: one for a code point below 128, such as a plain letter; two below 2048, such as `é`; three below 65536, such as `€`; and four for the rest, such as emoji.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **`for` over a text** | goes through its characters one at a time | `for ch in "hey":` gives `"h"`, `"e"`, `"y"` |
    | **Unicode** | the worldwide list that gives every character its own number, in every writing system, emoji too | |
    | **code point** | the number Unicode gives one character | `97` for `"a"` |
    | **`ord`** | gives the code point of one character | `ord("a")` gives `97` |
    | **`chr`** | the other way round: gives the character of a code point | `chr(97)` gives `"a"` |
    | **byte** | a whole number from 0 to 255; a file, and the memory of a computer, hold bytes | |
    | **character encoding** | a rule for storing characters as bytes | |
    | **UTF-8** | the character encoding nearly everyone uses: one byte for a code point below 128, two to four for bigger ones | `"utf-8"` |
    | **`.encode`** | turns a text into bytes, with the encoding you give | `"café".encode("utf-8")` |
    | **`.decode`** | turns bytes back into text, with the encoding you give | `data.decode("utf-8")` |
    | **`"".join`** | sticks a list of texts together into one text; the text before `.join` goes between the parts, and with `""` nothing does | `"".join(["h", "i"])` gives `"hi"` |

    `.encode` gives a value of the type `bytes`. Python shows it as `b'...'`, with codes such as `\xc3` for the bytes above 127; `list(...)` of it gives the bytes as plain numbers.
    """)])
    return


@app.cell(hide_code=True)
def picture_cafe():
    mo.vstack([
        draw.text_numbers("café", ("character", "code point", "bytes in UTF-8"), title="One word, three ways"),
        mo.md(r"""
        The word `café` three times. Each character has one code point: `é` has 233. In UTF-8, `c`, `a` and `f` take one byte each, the same number as their code point, but `é` takes two: 195 and 169. So `café` is 4 characters and 5 bytes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Bytes read with a different encoding from the one they were written with turn into garbled text: `\"café\".encode(\"utf-8\").decode(\"latin-1\")` gives `cafÃ©`, because `latin-1`, an older encoding, reads each of the two bytes of `é` as a character of its own. When you open a text file, give the encoding: `open(path, encoding=\"utf-8\")`."),
        mo.md(r"""
        ### Recipe

        To turn a text into numbers, and back:

        1. Take the code point of each character: `codes = [ord(ch) for ch in text]`.
        2. Turn the code points back into text: a character for each number, stuck together with `"".join([chr(n) for n in codes])`.
        3. To store the text, turn it into bytes: `data = text.encode("utf-8")`; `list(data)` shows them as numbers.
        4. Read the bytes back with the same encoding: `data.decode("utf-8")`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The text `café €3`, as code points and as bytes. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cafe():
    # 1. the code point of each character
    _text = "café €3"
    _codes = [ord(ch) for ch in _text]
    print(_codes)
    # 2. back to text: a character for each code point, stuck together
    _chars = [chr(n) for n in _codes]
    print(_chars)
    print("".join(_chars))
    # 3. the bytes in UTF-8: more bytes than characters
    _data = _text.encode("utf-8")
    print(len(_text), len(_data))
    print(list(_data))
    # 4. and back from the bytes
    print(_data.decode("utf-8"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([mo.md(r"""
    Seven characters, seven code points: `é` is 233, the space 32, `€` 8364, and the digit `3` is 51, not 3. `chr` turns each number back into its character, and `"".join` sticks them together into the same text. In UTF-8 the text takes 10 bytes: `é` takes two, 195 and 169, and `€` three, 226, 130 and 172. `.decode("utf-8")` reads them back as the same text.
    """)])
    return


@app.cell(hide_code=True)
def text_exercise_bytes():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many bytes?
    """)
    return


@app.cell(hide_code=True)
def choice_bytes_cell():
    choice_bytes = L.choice(
        "bytes",
        question="`len(\"naïve\")` is `5`: five characters. What does `len(\"naïve\".encode(\"utf-8\"))` give?",
        options={
            "six": "`6`",
            "five": "`5`",
            "ten": "`10`",
        },
        correct="six",
        why={
            "five": "That is the number of characters. Does every character take one byte? Look at `é` in the picture, and at the rule for UTF-8 in *The idea*.",
            "ten": "Not every character takes two bytes. Which characters take only one, in the picture?",
            "six": "`n`, `a`, `v` and `e` are plain letters, one byte each. `ï` has the code point 239: below 2048 but not below 128, so it takes two, like `é` in the picture. 4 + 2 = 6.",
        },
    )
    choice_bytes
    return (choice_bytes,)


@app.cell(hide_code=True)
def text_exercise_codes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A word as numbers

        Make `codes`: a list with the code point of each character of `word`, in the same order. Replace the `...`. The last line is `codes`, so the check sees it. The check also tries another word.
        """),
        L.task("Make `codes`: the code point of each character of `word`, in a list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_codes():
    box_codes = L.code_box("codes", lines=5)
    box_codes
    return (box_codes,)


@app.cell(hide_code=True)
def check_codes(box_codes):
    result_codes = L.check("codes", box_codes.value)
    result_codes
    return (result_codes,)


@app.cell(hide_code=True)
def hints_codes():
    L.hints(
        "codes",
        [
            "A text is a row of characters, and `ord` takes one character. So go through the characters one by one.",
            "Step 1 of the recipe: a comprehension, between square brackets, with `for ch in word`.",
            "`codes = [___ for ch in word]`: what goes in front of the `for`, for each `ch`?",
        ],
        why="`for ch in word` gives the characters `l`, `a`, `m` and `p` one by one, and `ord(ch)` gives the code point of each: 108, 97, 109 and 112.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Numbers back to text

        Finish `decode(codes)`. It gets a list of code points, and gives back the text they make. Replace the `...`. The check calls `decode([104, 105, 33])`: that should give `"hi!"`. It also tries other lists, one of them empty.
        """),
        L.task("Make `decode(codes)` give back the text of the code points in `codes`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=6)
    box_decode
    return (box_decode,)


@app.cell(hide_code=True)
def check_decode(box_decode):
    result_decode = L.check("decode", box_decode.value)
    result_decode
    return (result_decode,)


@app.cell(hide_code=True)
def hints_decode():
    L.hints(
        "decode",
        [
            "Step 2 of the recipe: first a character for each number, then one text of those characters.",
            "`chr` turns one number into its character, so a comprehension with `chr` gives a list of characters. `\"\".join` makes one text of such a list.",
            "`chars = [___ for n in codes]`, and then `return ___`.",
        ],
        why="`[chr(n) for n in codes]` gives a list with the character of each number, and `\"\".join(chars)` sticks them together with nothing in between. An empty list gives an empty text, `\"\"`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Text: strings](/course/open/py-strings?lang=en) a text was a row of characters, each at its own place; now each character also has a number. In [Reading a text file](/course/open/py-files?lang=en) you opened a file: it holds bytes, and `open(path, encoding="utf-8")` says how to read them as text. Later in the course, a language model reads text as a list of numbers like these, but with a table of its own, like the secret code: each character, or each piece of a word, gets a number from that table. In the next lesson, *Tables with pandas*, you read a whole table of data from a file.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bytes, result_codes, result_decode):
    L.wrap_up(choice_bytes, result_codes, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
