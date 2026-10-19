# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson tok-special-tokens. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-special-tokens", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en) and [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cross():
    quick_cross = L.choice(
        "quick-cross",
        question="With pre-tokenization, can BPE learn a merge rule for a comma followed by a space?",
        options={
            "right": "No: the space goes with the next piece",
            "often": "Yes: like any pair, if it comes often enough",
            "lines": "Only in a text without new lines",
        },
        correct="right",
        why={
            "often": "BPE only counts pairs inside a piece, and the comma and the space are in two pieces: `,` and ` word`. See [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en).",
            "lines": "New lines do not matter here: in any text, the space goes with the next piece. See [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en).",
            "right": "Today the cutting goes one step further: at every end-of-text, before the pieces.",
        },
    )
    quick_cross
    return (quick_cross,)


@app.cell(hide_code=True)
def quick_check_join():
    quick_join = L.choice(
        "quick-join",
        question="What does `\"-\".join([\"a\", \"b\", \"c\"])` give?",
        options={
            "around": "`\"-a-b-c-\"`",
            "right": "`\"a-b-c\"`",
            "end": "`\"abc-\"`",
        },
        correct="right",
        why={
            "around": "The text before `.join` goes between the parts, not before the first or after the last. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "end": "The text before `.join` goes between every two parts. See [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "right": "Today `.join` puts an end-of-text between documents again, when ids become text.",
        },
    )
    quick_join
    return (quick_join,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A language model trains on many documents, put after each other into one long row of tokens: plays, letters, web pages. Without a mark between them, the model learns that a new document goes on from the last one, and it never learns where a text ends. GPT-2 puts a **special token** between every two documents: end-of-text, written `<|endoftext|>`. It is one token with an id of its own, and normal text can never make it. When the model later writes end-of-text, its text is finished. In this project step your tokenizer gets end-of-text, and with that it is done.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           3, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_plain():
    guess_plain = L.guess(
        "as-text",
        question="Your tokenizer with 100 merge rules from Shakespeare gets the 13 characters `<|endoftext|>` as normal text, without any special handling. How many tokens do you think they become?",
        options={
            "one": "1: it is one word",
            "three": "3: `<|`, `endoftext` and `|>`",
            "right": "12: almost one per character",
        },
        reveal="**12.** Pre-tokenization cuts it into `<|`, `endoftext` and `|>`, and in there, only `e` and `n` merge, into `en`. So 13 characters are 12 tokens, and none of them means: a text ends here. That is why end-of-text needs an id of its own, and must not come from the letters.",
    )
    guess_plain
    return (guess_plain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **special token** is a token with an id of its own that normal text never turns into. End-of-text gets the first id that is still free, after the 256 bytes and the merge rules: `256 + len(merges)`. In GPT-2, with its 50,000 rules, that is 50,256, the last of its 50,257 ids.

        To encode, the tokenizer first cuts the text at every end-of-text. `text.split(END)` gives the parts between them, as a list: `"a<|endoftext|>b".split(END)` gives `["a", "b"]`. Every part is encoded as before, with pre-tokenization and the merge rules, and the id of end-of-text goes between the parts. To decode, it goes the other way: the ids between two ids of end-of-text become text through their bytes, and `END.join(texts)` puts end-of-text between the texts again.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **special token** | a token with an id of its own that normal text never turns into, such as end-of-text | `END` |
        | end-of-text | the special token between two documents; when a model writes it, its text is finished | `end_id` |
        | `.split(sep)` | the parts of a text between every `sep`, as a list | `"a-b".split("-")` |
        """),
        L.watch_out("End-of-text is neither a byte nor a merge rule, so it has no bytes: decode must handle its id before it looks up bytes. And a tokenizer that turns every `<|endoftext|>` in a text into the special token lets anyone who types those 13 characters end a document. That is why tiktoken of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) stops with an error on a text with `<|endoftext|>` in it, unless the program allows that special token itself."),
        mo.md(r"""
        The picture: two documents as they go into training, each cut into pieces, and end-of-text between them as one token of its own, id 356 after 100 merge rules. ␣ is a space.
        """),
        C.special_picture({"title": "Two documents, one after the other", "first": "document 1, in pieces", "end": "end-of-text: one token, id 356", "second": "document 2, in pieces",
                           "alt": "Document 1, To be, or not to be, in pieces; then end-of-text as one token with id 356; then document 2, Één ijsje with an ice cream emoji, in pieces."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A tokenizer with end-of-text, for merge rules `merges`:

    1. End-of-text gets the first free id: `end_id = 256 + len(merges)`.
    2. To encode: cut the text at every end-of-text with `text.split(END)`; every part in pieces, every piece as bytes with the merge rules applied in order; the id of end-of-text between the parts.
    3. To decode: cut the ids at every `end_id`; every part back to text through its bytes; `END.join` between the texts.
    4. In training data, end-of-text goes between every two documents.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two short documents with end-of-text between them, cut into parts, without merge rules, so that every part is just its bytes. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_end():
    _END = "<|endoftext|>"
    _merges = {}
    # 1. end-of-text gets the first free id: with no merge rules, 256
    _end_id = 256 + len(_merges)
    # 2. the parts between the end-of-text tokens, each as its bytes, and the id of end-of-text between two parts
    _parts = ("Hi." + _END + "Bye.").split(_END)
    print(_parts)
    _ids = []
    for _k in range(len(_parts)):
        # complete: when _k > 0, first add _end_id to _ids
        _ids = _ids + list(_parts[_k].encode("utf-8"))
    print(_ids)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The parts are `['Hi.', 'Bye.']`. As it is, the ids are `[72, 105, 46, 66, 121, 101, 46]`: the bytes of both documents after each other, and nobody can see that there were two. Completed, they are `[72, 105, 46, 256, 66, 121, 101, 46]`: end-of-text is id 256, the first after the bytes, as there are no merge rules here.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A free id
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "end-id",
        question="Your tokenizer has the 256 bytes and 100 merge rules, ids 0 to 355, and end-of-text gets 356. Why not 33, which no merge rule uses either?",
        options={
            "free": "It could: any number that no merge rule uses is fine",
            "order": "Ids must be given in the order the tokens were made",
            "right": "33 is already a byte, `!`: decode could not tell them apart",
        },
        correct="right",
        why={
            "free": "Every id from 0 to 255 is a byte, used or not; end-of-text needs an id that means nothing else.",
            "order": "The order alone does not matter; what matters is that every id means exactly one thing.",
            "right": "Every id must mean exactly one thing. With 33 for end-of-text, every `!` in a text would end it.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_exercise_encode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Encode

        Complete `encode(text, merges)`: it gives the ids of a text, with the id of end-of-text wherever `<|endoftext|>` stands, steps 1 and 2 of the recipe. `encode_part(part, merges)` is given: the ids of one part, cut into pieces and merged as in your `train_pre_bpe`, with `merge_ids` from your `byte_bpe.py` and `PATTERN` from your `pre_bpe.py`. Replace the two `...`: the id of end-of-text, and the parts with their ids. The check calls `encode("hi!<|endoftext|>hi", {(104, 105): 256})`, which should give `[256, 33, 257, 256]`, and tries other texts and rules. When it is right, the check saves it in your project, as `work/projects/llm/encode.py`.
        """),
        L.task("Make `encode` put the id of end-of-text between the ids of the parts of a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_encode():
    box_encode = L.code_box("encode", lines=24)
    box_encode
    return (box_encode,)


@app.cell(hide_code=True)
def check_encode(box_encode):
    result_encode = L.check("encode", box_encode.value)
    result_encode
    return (result_encode,)


@app.cell(hide_code=True)
def hints_encode():
    L.hints(
        "encode",
        [
            "The first `...` is step 1 of the worked example, with `merges`.",
            "The second `...` is step 2 of the completed worked example, with `text` and `END`, and `encode_part(parts[k], merges)` for the ids of a part.",
        ],
        why="`split` gives one part more than the text has end-of-text tokens, so the id goes before every part but the first. Two of them in a row give an empty part between them, which has no ids.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Decode

        Complete `decode(ids, merges)`: it gives the text of a list of ids, with `<|endoftext|>` wherever the id of end-of-text stands, step 3 of the recipe. The first line imports `decode` from your `byte_decode.py`, as `decode_bytes`; `import ... as ...` gives what it imports another name. The loop over the ids and the last part are given. The check calls `decode([256, 33, 257, 256], {(104, 105): 256})`, which should give `"hi!<|endoftext|>hi"`, and tries other ids. When it is right, the check saves it as `work/projects/llm/decode.py`.
        """),
        L.task("Make `decode` turn the ids between the end-of-text tokens back into text, with end-of-text between them."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=16)
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
            "An `if` on `i == end_id`: then the part so far is finished; `else`, `i` goes into the part with `part.append(i)`.",
            "A finished part goes to `texts` in the same way as the last part, in the line after the loop; then `part` starts again as an empty list.",
        ],
        why="End-of-text has no bytes, so its id never goes into a part: it only closes one. `END.join` puts it back as text, between the parts.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your whole tokenizer: your `train_pre_bpe` learns 100 merge rules from the first 20,000 characters of Tiny Shakespeare, your `encode` turns two short documents with end-of-text between them into ids, and your `decode` turns those ids back into text. It works once exercises 2 and 3 are done, and takes about a second.
    """)
    return


@app.cell(hide_code=True)
def demo_tokenizer(result_encode, result_decode):
    result_encode, result_decode  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `encode` and your `decode` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your tokenizer takes too long here: more than a minute. Does your `train_pre_bpe` or your `encode` do more work than it needs, such as a loop that runs too often?*",
            "tokenizer": "*`train_pre_bpe`, `encode` or `decode` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check its exercise again: its check then saves your code again.*",
        }[_result])
    else:
        _back = "Your `decode` gives the two documents back, with end-of-text between them." if _result["back"] else "But your `decode` does not give the two documents back."
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _out = mo.md(f"""\"{C.DOCS[0]}\" and \"{C.DOCS[1]}\" become {len(_result['ids'])} ids: `{_result['ids']}`. End-of-text is the one id {_result['end_id']}, between the two documents. {_back}

As normal text, the 13 characters of `<|endoftext|>` would be {_result['plain']} ids. Your tokenizer is done: it learns merge rules, cuts text into pieces, takes any text in any language, and marks where documents end.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The pieces of [Pre-tokenization](/course/open/tok-pre-tokenization?lang=en), the bytes and merge rules of [Byte-level BPE](/course/open/tok-byte-level?lang=en), and now end-of-text between documents: the parts of the tokenizer of GPT-2, which [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) used ready-made. Your project `llm` now has `encode.py` and `decode.py`. Next: *Assemble your GPT*, which puts embeddings, a stack of your transformer blocks and an output head together into your own GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_end, result_encode, result_decode):
    L.wrap_up(choice_end, result_encode, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
