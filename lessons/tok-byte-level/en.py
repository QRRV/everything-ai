# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson tok-byte-level. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-byte-level", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en) and [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="A BPE learned the merge rules (`l`, `o`) and then (`lo`, `w`). Into how many tokens does it cut \"lowly\"?",
        options={
            "two": "2: `low` and `ly`",
            "right": "3: `low`, `l` and `y`",
            "four": "4: `lo`, `w`, `l` and `y`",
        },
        correct="right",
        why={
            "two": "There is no rule (`l`, `y`), so `l` and `y` stay apart. See [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en).",
            "four": "The rules go in order: after (`l`, `o`) makes `lo`, the rule (`lo`, `w`) makes `low`. See [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en).",
            "right": "Today the same rules work on bytes instead of characters.",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def quick_check_byte():
    quick_byte = L.choice(
        "quick-byte",
        question="Which whole numbers can a byte be?",
        options={
            "ascii": "0 to 127: one for every code point that takes one byte",
            "code": "Any code point, such as 8364 for €",
            "right": "0 to 255: there are 256 of them",
        },
        correct="right",
        why={
            "ascii": "A code point below 128 takes one byte with the same number, but a character such as é uses bytes above 127. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "code": "A code point can be far above 255; UTF-8 then writes it as two, three or four bytes. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "right": "Only 256 of them, for every text in every language: that is what today's tokenizer builds on.",
        },
    )
    quick_byte
    return (quick_byte,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your BPE of [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en) starts from the characters it saw. A text with a character it never saw, an emoji, a Chinese character or a letter with an accent that Shakespeare never used, has no token for it. Unicode has more than 150,000 characters, too many to put in a vocabulary. GPT-2 works on the bytes of UTF-8 instead: every text is bytes, and there are only 256 of them. Put all 256 in the vocabulary from the start, and any text can be tokenized. In this project step your tokenizer moves to bytes.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           3, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_bytes():
    guess_bytes = L.guess(
        "bytes",
        question="The text \"café 🙂\" has 6 characters. Before any merge, how many tokens is this text for a byte-level BPE?",
        options={
            "six": "6: one for every character",
            "seven": "7: é needs one extra",
            "right": "10: é takes two bytes, and the emoji four",
        },
        reveal="**10.** In UTF-8, `c`, `a`, `f` and the space take one byte each; `é` takes two, 195 and 169, and 🙂 four, 240, 159, 153 and 130. A byte-level BPE starts with one id for every byte, and merges from there.",
    )
    guess_bytes
    return (guess_bytes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Byte-level BPE** is BPE on the bytes of a text in UTF-8, instead of on its characters. The first tokens are the bytes, ids 0 to 255, and all 256 are in the vocabulary from the start, whether the training text has them or not. Every merge rule gets a new id: 256 for the first, 257 for the next, and so on. So the merges are a dictionary, from a pair of ids to its new id: `{(99, 97): 256}` means that `c` followed by `a` becomes id 256. GPT-2 has the 256 bytes, 50,000 merges and one special token: 50,257 tokens.

        To turn ids back into text, every id needs its bytes. An id below 256 is that byte: `bytes([i])` makes a bytes value from a list of numbers. A new id is the bytes of the two halves of its pair, one after the other, and `+` puts two bytes values after each other. Then all bytes in a row are decoded as UTF-8. One token is not always a whole character: an id can be the first byte of é alone. `.decode("utf-8", errors="replace")` then puts in �, instead of stopping with an error.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **byte-level BPE** | BPE on the bytes of UTF-8; the vocabulary starts with all 256 bytes, so any text can be tokenized |
        | `bytes([...])` | a bytes value made from a list of numbers from 0 to 255: `bytes([104, 105])` |
        | `b""` | an empty bytes value; `+` puts bytes values after each other: `data + vocab[i]` |
        | `errors="replace"` | in `.decode`: bytes that are not a whole character become �, instead of an error |
        """),
        L.watch_out("A new id stands for a pair of ids, not of bytes: (256, 102) means the bytes of 256, then those of 102. Build the bytes in the order the rules were learned, and never add two ids: 256 + 102 is just another number."),
        mo.md(r"""
        The picture: "café 🙂" as 6 characters and as 10 bytes. `c`, `a`, `f` and the space take one byte each, the same number as their code point; `é` takes two bytes, and the emoji four.
        """),
        C.bytes_picture({"title": "Six characters, ten bytes", "chars": "{k} characters", "word": "café and the space: {k} bytes", "emoji": "the emoji: {k} bytes",
                         "alt": "café, a space and an emoji: 6 characters; café and the space are the 6 bytes 99, 97, 102, 195, 169 and 32 in UTF-8, and the emoji the 4 bytes 240, 159, 153 and 130."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Byte-level BPE with $n$ merge rules:

    1. The text as bytes in UTF-8, as numbers: `list(text.encode("utf-8"))`, the first ids.
    2. Every round, the most frequent pair of ids gets the next new id, `256 + step`: a merge rule in a dictionary, pair → new id. Merge it everywhere, and go on, $n$ rounds.
    3. To encode a new text: its bytes, with every merge rule applied in the order it was learned.
    4. To decode: the bytes of every id, the byte itself below 256 and the bytes of both halves of its pair for a new id; all of them in a row, back to text in UTF-8 with `errors="replace"`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "café café" as bytes, one merge rule, and back. `_merge_ids` is your `merge` of the last lesson, with a new id in place of two strings put together. The comments with `#` say which step of the recipe each part is. Step 4 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_bytes():
    def _merge_ids(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i = i + 2
            else:
                out.append(ids[i])
                i = i + 1
        return out

    # 1. the bytes of the text in UTF-8: the first ids
    _ids = list("café café".encode("utf-8"))
    print(_ids)
    # 2. the most frequent pair, (99, 97) for "ca", gets the new id 256
    _merges = {(99, 97): 256}
    _ids = _merge_ids(_ids, (99, 97), _merges[(99, 97)])
    print(_ids)
    # 4. back to text: the bytes of every id, all in a row
    _vocab = {}
    for _b in range(256):
        _vocab[_b] = bytes([_b])
    _vocab[256] = b""  # complete: _vocab[99] + _vocab[97]
    _data = b""
    for _id in _ids:
        _data = _data + _vocab[_id]
    print(_data.decode("utf-8", errors="replace"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    9 characters are 11 bytes, `[99, 97, 102, 195, 169, 32, 99, 97, 102, 195, 169]`: é is 195 and 169. After the merge there are 9 ids, `[256, 102, 195, 169, 32, 256, 102, 195, 169]`. As it is, id 256 has no bytes, so the text comes back as `fé fé`. Completed, 256 is the bytes of 99 and 97, and the text is `café café` again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vocab():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The size of the vocabulary
    """)
    return


@app.cell(hide_code=True)
def choice_vocab_cell():
    choice_vocab = L.choice(
        "vocab-size",
        question="A byte-level BPE learned 1,000 merge rules from Shakespeare, which uses 65 different characters. How many ids does its vocabulary have?",
        options={
            "right": "1,256: the 256 bytes and one per rule",
            "rules": "1,000: one for every merge rule",
            "seen": "1,065: the 65 characters of Shakespeare and one per rule",
        },
        correct="right",
        why={
            "rules": "The bytes are tokens too: every text starts as bytes, and a byte that is in no merge stays an id of its own.",
            "seen": "A byte-level BPE keeps all 256 bytes, also those Shakespeare never used, so it can take any text.",
            "right": "Ids 0 to 255 for the bytes, 256 to 1,255 for the rules. That is how GPT-2 gets to 50,257: 256 bytes, 50,000 rules, and one special token.",
        },
    )
    choice_vocab
    return (choice_vocab,)


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. BPE on bytes

        Complete `train_byte_bpe(text, n)`: your `train_bpe` of [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en), on bytes. The ids start as the bytes of the text, and every merge rule gets a new id from 256 on; it gives the rules as a dictionary, pair → new id. `merge_ids` and the search for the best pair are given; the first line imports `pair_counts` from your `bpe.py`. The check calls `train_byte_bpe("naïve naïve", 4)`, which should give `{(110, 97): 256, (256, 195): 257, (257, 175): 258, (258, 118): 259}`, and tries other texts. When it is right, the check saves it in your project, as `work/projects/llm/byte_bpe.py`.
        """),
        L.task("Make `train_byte_bpe` learn merge rules on the bytes of a text, with new ids from 256 on."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-byte-bpe", lines=30)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train-byte-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-byte-bpe",
        [
            "The first `...` is step 1 of the worked example, with `text`.",
            "Three lines in the loop: `new_id = ___ + step`, then `merges[___] = new_id`, then `ids = merge_ids(ids, ___, ___)`, so the next round counts the merged ids.",
        ],
        why="Ids 0 to 255 are taken by the bytes, so the first new id is 256. With the pair as the key, a tokenizer can look up any pair it sees, and find what it becomes.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Back to text

        Complete `decode(ids, merges)`: it gives the text of a list of ids, step 4 of the recipe. The 256 bytes in `vocab` and the loop that puts all bytes in a row are given. The check calls `decode([257, 256, 111], {(108, 108): 256, (104, 101): 257})`, which should give `"hello"`, and tries other ids, also a byte that is only part of a character. When it is right, the check saves it as `work/projects/llm/byte_decode.py`.
        """),
        L.task("Make `decode` give the bytes of every new id, and turn all bytes back into text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=14)
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
            "Step 4 of the recipe: the bytes of a new id are those of both halves of its pair, one after the other, and `vocab` already has the bytes of each half.",
            "`vocab[___] = vocab[___] + vocab[___]`, like the line you completed in step 4 of the worked example; at the end, `return data.decode(___)`, with the encoding and the `errors=` of the recipe.",
        ],
        why="The merges come in the order they were learned, so both halves of a pair already have their bytes in `vocab` when the pair comes by.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your project. It trains your byte-level BPE on the first 20,000 characters of Tiny Shakespeare, 100 merge rules, and sends two lines through your `merge_ids` and back through your `decode`: the line of Hamlet, and a Dutch line with an emoji, which Shakespeare never wrote. It works once exercises 2 and 3 are done, and takes about a second.
    """)
    return


@app.cell(hide_code=True)
def demo_bytes(result_train, result_decode):
    result_train, result_decode  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `train_byte_bpe` and your `decode` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your byte-level BPE takes too long here: more than a minute. Does your `train_byte_bpe` do more work than it needs, such as a loop that runs too often?*",
            "bpe": "*`train_byte_bpe`, `merge_ids` or `decode` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
        }[_result])
    else:
        _rows = []
        for _line, _r in zip(C.LINES, _result["lines"]):
            _back = "and your `decode` gives the line back" if _r["back"] else "but your `decode` does not give the line back"
            _rows.append(f"- \"{_line}\": {len(_line)} characters, {_r['bytes']} bytes, **{len(_r['ids'])} ids**, {_back}: `{_r['ids']}`")
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _out = mo.md("\n".join(_rows) + f"""

The Dutch line has letters and an emoji that are not in Shakespeare. They stay as their bytes, such as 240, 159, 141 and 166 for the ice cream, but they can be tokenized, and come back whole. On 5,000 characters it did not learn from, an id holds {C.number(_result['per_id'], 2, L.lang)} bytes on average.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The bytes of UTF-8 of [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en), with the BPE of [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en): a tokenizer that takes any text. The tokenizer of GPT-2 in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) is byte-level BPE with 50,000 merge rules. Your project `llm` now has `byte_bpe.py` and `byte_decode.py`. Next: *Pre-tokenization*, which first cuts a text into words, numbers and punctuation, so that merges stay inside a word.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_vocab, result_train, result_decode):
    L.wrap_up(choice_vocab, result_train, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
