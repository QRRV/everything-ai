# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson tok-pre-tokenization. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-pre-tokenization", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Byte-level BPE](/course/open/tok-byte-level?lang=en) and [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_decode():
    quick_decode = L.choice(
        "quick-decode",
        question="A byte-level BPE decodes the single id 195, the first byte of é, with `errors=\"replace\"`. What text comes out?",
        options={
            "e": "`é`: the tokenizer knows which character it belongs to",
            "right": "`�`: one byte of é is not a whole character",
            "a": "`Ã`: the character with code point 195",
        },
        correct="right",
        why={
            "e": "One byte cannot be é: é needs both 195 and 169. See [Byte-level BPE](/course/open/tok-byte-level?lang=en).",
            "a": "That is how Latin-1 reads the byte 195. In UTF-8, 195 only starts a character of two bytes. See [Byte-level BPE](/course/open/tok-byte-level?lang=en).",
            "right": "Today the bytes come in pieces: words, signs and spaces, each encoded on its own.",
        },
    )
    quick_decode
    return (quick_decode,)


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="What does `[w + \"!\" for w in [\"to\", \"be\"]]` give?",
        options={
            "joined": "`\"to!be!\"`",
            "added": "`[\"to\", \"be\", \"!\"]`",
            "right": "`[\"to!\", \"be!\"]`",
        },
        correct="right",
        why={
            "joined": "A comprehension gives a list, with one item for every item it goes over. See [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).",
            "added": "The `!` goes onto every item, not after the list. See [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).",
            "right": "Today a comprehension does something to every piece of a text.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your byte-level BPE of [Byte-level BPE](/course/open/tok-byte-level?lang=en) merges across words: `e␣` and `,␣` became tokens, each with the end of one word and the space before the next. The vocabulary then fills up with such combinations, `the␣`, `and␣`, `you␣`, and the same word gets different tokens depending on what follows it. GPT-2 first cuts a text into words, numbers and signs, each with the space before it, and runs BPE inside those pieces only. In this project step you add that cutting to your tokenizer.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           3, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-merge",
        question="Without pre-tokenization, the first merge on 20,000 characters of Tiny Shakespeare was `e` and a space. With pre-tokenization, the pieces are words with the space before them, signs and spaces. Which pair do you think is merged first now?",
        options={
            "e": "`e` and a space, as before",
            "right": "A space and `t`, as at the start of the, to and that",
            "th": "`t` and `h`, as in the, that and thou",
        },
        reveal="**A space and `t`**: 404 times inside the pieces, more than any other pair. `e` and a space can no longer meet: inside a piece, a space never comes after a letter. `t` and `h` come 402 times, close behind.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Pre-tokenization** cuts a text into pieces before BPE, and BPE counts and merges inside every piece only, so a merge never crosses from one piece into the next. To encode a new text, it is cut the same way, every piece is encoded on its own, and the ids are put one after the other.

        The pieces come from a **regular expression**: a short pattern for pieces of text. `re.findall(pattern, text)`, from Python's module `re`, gives every piece of `text` that fits the pattern, from left to right, without overlap. The pattern of this lesson is `r" ?\w+| ?[^\w\s]+|\s+"`; the `r` before the quotes keeps every backslash as it is. Its three parts are separated by `|`, which means *or*; at every place, the first part that fits wins:

        - `r" ?\w+"`: maybe a space (`?` makes what comes before it optional), then one or more (`+`) letters or digits (`\w`), as many as there are: a word, with the space before it.
        - `r" ?[^\w\s]+"`: maybe a space, then one or more characters that are not (`[^...]`) a letter, a digit or white space (`\s`): signs, or an emoji.
        - `r"\s+"`: white space that is left over, such as a new line.

        So `re.findall(PATTERN, "Hi, you!")` gives `["Hi", ",", " you", "!"]`. GPT-2 uses a longer pattern of the same kind, which also keeps endings such as `'s` and `'re` apart, and numbers apart from letters.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **pre-tokenization** | | cutting a text into pieces before BPE, so that no merge crosses from one piece into the next | `re.findall(PATTERN, text)` |
        | **regular expression** | | a short pattern for pieces of text | `r" ?\w+"` |
        | `re.findall` | re find all | every piece of a text that fits a pattern, from left to right, without overlap | `re.findall(r"\w+", "Hi, you!")` |
        | `\w`, `\s` | | a letter, a digit or `_`; white space, such as a space or a new line | `\s+` |
        | `+`, `?` | | one or more of what comes before, as many as there are; maybe one | `r" ?\w+"` |
        | `|` | or | either part may fit; the first that fits wins | `r"\w+|\s+"` |
        | `[^...]` | | one character that is none of the ones inside | `[^\w\s]` |
        | `r"..."` | | a raw string: every backslash in it stays as it is | `r"\w+"` |
        """),
        L.watch_out("Train and encode with the same pattern: a tokenizer that cuts different pieces when it encodes than when it learned meets pairs it never counted. And cut the text first, then turn every piece into bytes: `re.findall` with this pattern cuts a text, not bytes."),
        mo.md(r"""
        The picture: the line of Hamlet, cut into 12 pieces, and then into tokens by 100 merge rules that the course learned inside the pieces of the first 20,000 characters of Tiny Shakespeare. ␣ is a space. No token holds the end of one word and the start of another.
        """),
        C.pieces_picture({"title": "Pieces first, then merges", "pieces": "{k} pieces of the pattern", "tokens": "{k} tokens after 100 merge rules inside the pieces",
                          "alt": "To be, or not to be, that is the question in 12 pieces: To, space-be, comma, space-or, space-not, space-to, space-be, comma, space-that, space-is, space-the, space-question. After 100 merge rules inside the pieces, 20 tokens, such as space-be, space-not and space-the; question is still in small pieces."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Byte-level BPE with pre-tokenization, $n$ merge rules:

    1. Cut the text into pieces: `re.findall(PATTERN, text)`, words with the space before them, signs, and white space.
    2. Every piece as a list of its bytes in UTF-8: a list of lists of ids.
    3. Count the pairs inside every piece, never across two pieces; the most frequent pair gets the next new id.
    4. Merge that pair inside every piece, and go on, $n$ rounds. To encode a new text: cut it the same way, encode every piece, and put the ids one after the other.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "be, or not to be", cut into pieces, and the pairs counted inside them. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_pieces():
    import re

    _PATTERN = r" ?\w+| ?[^\w\s]+|\s+"
    _text = "be, or not to be"
    # 1. the pieces: words with the space before them, signs, and white space
    _pieces = re.findall(_PATTERN, _text)
    print(_pieces)
    # 2. every piece as its bytes in UTF-8
    _chunks = [list(_p.encode("utf-8")) for _p in _pieces]
    print(_chunks)
    # 3. count the pairs inside every piece
    _counts = {}
    for _ids in [list(_text.encode("utf-8"))]:  # complete: loop over _chunks instead
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _counts:
                _counts[_pair] = _counts[_pair] + 1
            else:
                _counts[_pair] = 1
    print(len(_counts), (101, 44) in _counts, _counts[(98, 101)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The pieces are `['be', ',', ' or', ' not', ' to', ' be']`, and their bytes `[[98, 101], [44], [32, 111, 114], [32, 110, 111, 116], [32, 116, 111], [32, 98, 101]]`. As it is, the loop counts the bytes of the whole text as one piece: 14 different pairs, among them (101, 44), `e` and a comma (`True`), and `be` (98, 101) twice. Completed, it counts inside the pieces: 9 different pairs, (101, 44) is gone (`False`), and `be` still comes twice.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_findall():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What the pattern cuts
    """)
    return


@app.cell(hide_code=True)
def choice_findall_cell():
    choice_findall = L.choice(
        "findall",
        question="With `PATTERN = r\" ?\\w+| ?[^\\w\\s]+|\\s+\"`, what does `re.findall(PATTERN, \"Wait... what?!\")` give?",
        options={
            "right": "`[\"Wait\", \"...\", \" what\", \"?!\"]`",
            "each": "`[\"Wait\", \".\", \".\", \".\", \" what\", \"?\", \"!\"]`",
            "stuck": "`[\"Wait...\", \" what?!\"]`",
        },
        correct="right",
        why={
            "each": "`+` takes one or more, as many as there are: the three dots are one run of signs, and so are `?!`.",
            "stuck": "`\\w` takes only letters, digits and `_`, so a word stops at the first sign; the signs are a piece of their own.",
            "right": "A word, a run of signs, a word with the space before it, and a run of signs. BPE will never merge `Wait` with its dots.",
        },
    )
    choice_findall
    return (choice_findall,)


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count inside the pieces

        Complete `chunk_pair_counts(chunks)`: it gives a dictionary with how often every pair of neighboring ids comes by inside the pieces of `chunks`, a list of lists of ids, and never across two pieces: step 3 of the recipe. The check calls `chunk_pair_counts([[104, 105], [32, 104, 105], [33]])`, the pieces "hi", " hi" and "!", which should give `{(104, 105): 2, (32, 104): 1}`, and tries other pieces. When it is right, the check saves it in your project, as `work/projects/llm/chunk_pairs.py`.
        """),
        L.task("Make `chunk_pair_counts` count the pairs inside every piece, and none across two pieces."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("chunk-pair-counts", lines=11)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("chunk-pair-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "chunk-pair-counts",
        [
            "Inside the loop over the pieces, a second loop over the pairs of one piece: `for i in range(len(ids) - 1):`, as in your `pair_counts` of [Byte pair encoding: build your own tokenizer](/course/open/tok-bpe?lang=en).",
            "Step 3 of the completed worked example, with `chunks`, `ids` and `counts` instead of `_chunks`, `_ids` and `_counts`.",
        ],
        why="The pairs are counted piece by piece, all in the one dictionary: the last id of one piece and the first id of the next are never neighbors in a loop over one piece.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_counts):
    result_counts  # made again after the check of exercise 2, which saves your chunk_pair_counts
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. BPE inside the pieces

        Complete `train_pre_bpe(text, n)`: your `train_byte_bpe` of [Byte-level BPE](/course/open/tok-byte-level?lang=en), on the pieces of `PATTERN`. Its first lines import `merge_ids` from your `byte_bpe.py` and your `chunk_pair_counts`; the search for the best pair and the new id are given. Replace the two `...` with steps 1, 2 and 4 of the recipe. The check calls `train_pre_bpe("the cat, the hat", 3)`, which should give `{(116, 104): 256, (256, 101): 257, (97, 116): 258}`, and tries other texts. When it is right, the check saves it as `work/projects/llm/pre_bpe.py`.
        """),
        L.course_versions(),
        L.task("Make `train_pre_bpe` learn merge rules inside the pieces of a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-pre-bpe", lines=22)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_counts):
    result_counts  # check again when exercise 2 saves your chunk_pair_counts
    result_train = L.check("train-pre-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-pre-bpe",
        [
            "The first `...` is steps 1 and 2 of the worked example in one comprehension: for every piece of `re.findall(PATTERN, text)`, the list of its bytes.",
            "The second `...` is a comprehension too: `merge_ids(ids, best, new_id)` for every `ids` in `chunks`.",
        ],
        why="Every piece is merged on its own, so a pair at the end of one piece and the start of the next is never merged, as it was never counted.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your project. It trains your BPE with pre-tokenization on the first 20,000 characters of Tiny Shakespeare, 100 merge rules, and encodes the line of Hamlet with them. Then it counts how many of the 100 new tokens cross from one word into the next, and compares that with the course's byte-level BPE without pre-tokenization, on the same text. It works once exercises 2 and 3 are done, and takes about a second.
    """)
    return


@app.cell(hide_code=True)
def demo_pieces(result_counts, result_train):
    result_counts, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `chunk_pair_counts` and your `train_pre_bpe` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your BPE takes too long here: more than a minute. Does your `train_pre_bpe` do more work than it needs, such as a loop that runs too often?*",
            "bpe": "*`train_pre_bpe`, `chunk_pair_counts` or `merge_ids` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
        }[_result])
    else:
        _first = ", ".join(f"`{C.shown(t)}`" for t in _result["first"])
        _line = " ".join(f"`{C.shown(t)}`" for t in _result["line"])
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note at exercise 3 says."
        _out = mo.md(f"""Your first 12 new tokens: {_first}. ␣ is a space.

The line becomes {len(_result['line'])} tokens: {_line}. Of your 100 new tokens, **{_result['crossing']}** cross from one word into the next; without pre-tokenization, {_result['plain_crossing']} of the 100 did, such as `e␣` and `,␣`. On the next 5,000 characters, an id holds {C.number(_result['per_id'], 2, L.lang)} bytes on average, about as many as without pre-tokenization: as many tokens, but tokens that are words or parts of words.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your byte-level BPE of [Byte-level BPE](/course/open/tok-byte-level?lang=en), now inside the pieces of a regular expression: the same steps as the tokenizer of GPT-2 in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en), where " surprisingly" was one token with its space in front. Your project `llm` now has `chunk_pairs.py` and `pre_bpe.py`. Next: *Pre-tokenization and special tokens*, tokens such as end-of-text that never come out of normal text, and mark where one document stops and the next begins.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_findall, result_counts, result_train):
    L.wrap_up(choice_findall, result_counts, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
