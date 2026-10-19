# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-tokens-outside. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-tokens-outside", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="With one character per token, how many tokens does the text \"a cat\" have?",
        options={
            "words": "2: one for every word, \"a\" and \"cat\"",
            "letters": "4: one for every letter",
            "right": "5: the space is a token too",
        },
        correct="right",
        why={
            "words": "In that lesson, a token is one character, not a word. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "letters": "The space is a character as well, and so a token. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "\"a\", space, \"c\", \"a\", \"t\": five characters, five tokens. Real models use bigger pieces, and today you see which.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_examples():
    quick_examples = L.choice(
        "quick-examples",
        question="How many examples for next-token prediction does a text of 10 tokens give?",
        options={
            "right": "9: every token after the first is the label of one example",
            "ten": "10: one for every token",
            "one": "1: the whole text is one example, with its last token as the label",
        },
        correct="right",
        why={
            "ten": "The first token has no text before it, so it is never a label. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "one": "Every place gives an example: the text so far, and the token that comes next. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "A text of $n$ tokens gives $n - 1$ examples. So how a text is cut into tokens decides how many examples it gives, and how long it is for the model.",
        },
    )
    quick_examples
    return (quick_examples,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The language model of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) read one character at a time. Real language models read tokens that are whole words and pieces of words, from a fixed list of tens of thousands. Much of what you hear about LLMs is counted in those tokens: how much text a model can read at once, what it costs per million tokens, how fast it writes. And how many tokens a text costs depends on how the tokenizer cuts it.

    **An everyday example.** With only the smallest building bricks, one stud each, you can build anything, but it takes a great many bricks. With only ready-made walls, you build fast, but only what you have walls for. With a box of big pieces for what you build often, and small ones for the rest, you build anything, and what you build often with few pieces. Characters are the one-stud bricks, whole words the ready-made walls, and real tokens the mixed box.
    """)
    return


@app.cell(hide_code=True)
def predict_dutch():
    guess_dutch = L.guess(
        "dutch-tokens",
        question="GPT-2, an open language model from 2019, cuts the English sentence \"Tokenization is surprisingly important.\" into 6 tokens. About how many tokens does it make of the same sentence in Dutch, \"Tokenisatie is verrassend belangrijk.\"?",
        options={
            "same": "About 6: the same sentence, the same number of words",
            "fewer": "About 4: one token per word",
            "right": "About 15: more than twice as many",
        },
        reveal="**15.** \"Tokenization\" is 2 tokens, \"Token\" and \"ization\", and \" surprisingly\" and \" important\" are one each. The Dutch words fall apart: \" verrassend\" becomes \" ver\", \"r\", \"ass\" and \"end\", and \" belangrijk\" five pieces. GPT-2 made its list of tokens from text that was almost all English, so it has many English words whole, and of most Dutch words only pieces.",
    )
    guess_dutch
    return (guess_dutch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **tokenizer** cuts a text into tokens and gives every token its number, its **token ID**; it also turns token IDs back into text. Its **vocabulary** is the fixed list of all tokens it knows: for GPT-2, 50,257. A common word is one token, often with the space before it, and a rarer word falls into **subwords**: pieces of words that are common, such as "un", "bel", "iev" and "able". The smallest tokens are the 256 bytes of [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en), so every text fits, even a name the tokenizer has never seen, or an emoji: at worst, it falls into bytes.

        The vocabulary was made once, from a lot of text: pieces that often came together became one token. For GPT-2, that text was almost all English, so English words are often whole, and Dutch words fall into more pieces. A later lesson has you build such a tokenizer yourself.

        In Python, the library **tiktoken** has the tokenizers of OpenAI's models: `tiktoken.get_encoding("gpt2")` gives that of GPT-2. `enc.encode(text)` gives the list of token IDs, `enc.decode(ids)` the text back, and `enc.n_vocab` the size of the vocabulary. Exercise 3 also uses one new bit of Python: `text.split()` cuts a text at its spaces into a list of words.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **tokenizer** | | cuts a text into tokens, gives each its token ID, and turns token IDs back into text | `enc.encode(text)`, `enc.decode(ids)` |
        | **token ID** | | the number of a token in the vocabulary | `enc.encode(" the")` gives `[262]` |
        | **vocabulary** | | the fixed list of all tokens a tokenizer knows | `enc.n_vocab` |
        | **subword** | | a token that is a piece of a word | `"un"`, `"able"` |
        | `split` | | the words of a text, cut at its spaces | `"a cat".split()` gives `['a', 'cat']` |
        """),
        L.watch_out("A token is not a word: \" the\", with its space, is a different token from \"the\", and \"The\" is a third. The token IDs belong to one tokenizer: 262 is \" the\" for GPT-2, and can be something else for a model with another tokenizer. And a character of several bytes can fall into several tokens: 🙂 is 4 bytes and 2 tokens. Decode one of them alone, and you get � signs, the sign for bytes that are not a whole character."),
        mo.md(r"""
        The picture: the two sentences of the guess, cut into GPT-2's tokens, with the token ID of each under it. A space shows as ␣: in GPT-2, it belongs to the token after it.
        """),
        mo.Html(C.tokens_picture({"title": "One sentence, two languages", "en": "English: {n} tokens", "nl": "Dutch: {n} tokens",
                                  "alt": "English, 6 tokens: Token, ization, space is, space surprisingly, space important, full stop. Dutch, 15 tokens: Token, is, at, ie, space is, space ver, r, ass, end, space bel, ang, ri, j, k, full stop."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The tokens of a text, with a ready-made tokenizer:

    1. Load the tokenizer: `enc = tiktoken.get_encoding("gpt2")`.
    2. Encode the text: `enc.encode(text)` gives its token IDs, in a list.
    3. Count the tokens: the length of that list.
    4. Decode to see the pieces: `enc.decode([i])` gives the piece of text of token ID `i`, and `enc.decode(ids)` the whole text back.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An English and a Dutch sentence, through GPT-2's tokenizer. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tokens():
    import tiktoken

    # 1. the tokenizer of GPT-2, and the size of its vocabulary
    _enc = tiktoken.get_encoding("gpt2")
    print(_enc.n_vocab)
    # 2. the token IDs of the two sentences
    _en = _enc.encode("The cat sat on the mat.")
    _nl = _enc.encode("De kat zat op de mat.")
    print(_en)
    print(_nl)
    # 3. the number of tokens
    print(len(_en), len(_nl))
    # 4. the pieces of the Dutch sentence, and the whole text back
    print([_enc.decode([_i]) for _i in _nl])
    print(_enc.decode(_nl))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 knows 50,257 tokens. The English sentence is 7 tokens: one for every word, each but the first with the space before it, and one for the full stop. "The" at the start is 464, and " the" with its space is 262. The Dutch sentence is 9 tokens: " kat" and " zat" are not in the vocabulary, so they fall into " k" and "at", and " z" and "at". " mat" is the same token, 2603, in both sentences. And `decode` gives the whole sentence back, exactly.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_space():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two IDs for "the"
    """)
    return


@app.cell(hide_code=True)
def choice_space_cell():
    choice_space = L.choice(
        "space",
        question="`enc.encode(\"the\")` gives `[1169]`, and `enc.encode(\" the\")` gives `[262]`. Why two different token IDs?",
        options={
            "slip": "The tokenizer slips: one word should always get one token ID, whatever comes before it",
            "right": "The space belongs to the token: \"the\" and \" the\" are two tokens of the vocabulary",
            "random": "The token IDs are drawn at random each time you call `encode`",
        },
        correct="right",
        why={
            "slip": "It is no slip: the tokenizer cuts the whole text, spaces too. A word in the middle of a sentence has a space before it, so \" the\" is far more common than \"the\" and got a token of its own.",
            "random": "Call `encode` again and you get the same IDs: the vocabulary is fixed, and so is every token ID in it.",
            "right": "In the middle of a sentence, almost every word has a space before it, so GPT-2's vocabulary has \" the\", with the space, as one token. \"the\" without a space, at the start of a line, is another token.",
        },
    )
    choice_space
    return (choice_space,)


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The pieces of any text

        Make `pieces(text)` give the tokens of a text as pieces of text, in a list. The box loads the tokenizer as `enc`. Replace the two `...`: the token IDs of the text, by step 2 of the recipe, and for every token ID its piece of text, by step 4. The check calls `pieces("Hello world")`, which should give `['Hello', ' world']`, and tries other texts, including an empty one.
        """),
        L.task("Make `pieces` give the tokens of a text as pieces of text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("pieces", lines=11)
    box_pieces
    return (box_pieces,)


@app.cell(hide_code=True)
def check_pieces(box_pieces):
    result_pieces = L.check("pieces", box_pieces.value)
    result_pieces
    return (result_pieces,)


@app.cell(hide_code=True)
def hints_pieces():
    L.hints(
        "pieces",
        [
            "Two steps of the recipe: first the token IDs of the whole text, then in the loop one piece for each token ID.",
            "Step 2 gives the token IDs: `enc.encode` of the text. Step 4 turns one token ID back into text, and `decode` wants a list, also for one ID.",
            "`ids = enc.encode(___)`, and in the loop `out.append(enc.decode([___]))`.",
        ],
        why="For \"Hello world\", `enc.encode` gives `[15496, 995]`, and decoding each alone gives \"Hello\" and \" world\", with its space: the space belongs to the token after it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_per_word():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Tokens per word

        Make `tokens_per_word(text)` give how many tokens a text costs per word: its number of tokens divided by its number of words. Replace the two `...`: the number of tokens, by step 3 of the recipe, and the number of words, with `text.split()`. The check calls `tokens_per_word("The weather is nice today.")`, which should give `1.2`, and tries other texts, some in Dutch too.
        """),
        L.task("Make `tokens_per_word` give the number of tokens of a text per word."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_word():
    box_per_word = L.code_box("tokens-per-word", lines=9)
    box_per_word
    return (box_per_word,)


@app.cell(hide_code=True)
def check_per_word(box_per_word):
    result_per_word = L.check("tokens-per-word", box_per_word.value)
    result_per_word
    return (result_per_word,)


@app.cell(hide_code=True)
def hints_per_word():
    L.hints(
        "tokens-per-word",
        [
            "Two counts: the tokens of the text, and its words. Both are the length of a list.",
            "The list of tokens comes from `enc.encode(text)`, and the list of words from `text.split()`. `len` gives the length of a list.",
            "`n_tokens = len(enc.encode(___))` and `n_words = len(___.split())`.",
        ],
        why="\"The weather is nice today.\" is 6 tokens, five words and a full stop, and `split` gives 5 words, the full stop stuck to \"today.\": 6 / 5 = 1.2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The tokens of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) were characters; a real tokenizer cuts text into words and pieces of words, from a fixed vocabulary, and its smallest pieces are the bytes of [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en). A language model reads and writes only token IDs: it gives a probability for every token of the vocabulary. In the next lesson, *A bigram model by counting*, you build a language model of your own by counting, with characters as tokens again: a small vocabulary that is easy to count.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_space, result_pieces, result_per_word):
    L.wrap_up(choice_space, result_pieces, result_per_word)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
