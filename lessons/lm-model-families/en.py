# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-model-families. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-model-families", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Attention: every word looks at the others](/course/open/attn-idea?lang=en) and [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_attention():
    quick_attention = L.choice(
        "quick-attention",
        question="With attention, which words does a word take in to make its new vector?",
        options={
            "right": "All words, its own too, each by its attention weight",
            "before": "Only the word just before it",
            "same": "Only the words that are spelled exactly the same as it",
        },
        correct="right",
        why={
            "before": "That is closer to a recurrent neural network, which passes things on word by word. See [Attention: every word looks at the others](/course/open/attn-idea?lang=en).",
            "same": "Every word gets a weight, also a word that only matches a little. See [Attention: every word looks at the others](/course/open/attn-idea?lang=en).",
            "right": "Every word looks at all words, its own included. Today you see the families of transformers, and the one rule that sets them apart: which tokens a token may look at.",
        },
    )
    quick_attention
    return (quick_attention,)


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="In next-token prediction, the model gets the text so far and must predict the next token. Is that next token part of its input?",
        options={
            "yes": "Yes: it needs the token to learn from",
            "end": "Yes, but only at the end of the text",
            "right": "No: the input stops just before it",
        },
        correct="right",
        why={
            "yes": "The token is the label, which the model's prediction is compared with; in the input it would give the answer away. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "end": "At every place, the input is only the text before the label. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "The input is the text so far, without the token it must predict: otherwise the model could simply copy its answer. A model with attention, which looks at all tokens, needs a rule for that: today you see it.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    BERT, GPT and T5 are three famous models from 2018 and 2019, and all three are transformers: layers of attention, each followed by a small MLP. Yet they do different jobs. BERT reads a text and says something about it, GPT writes text, and T5 turns one text into another. When you meet a new model, its family tells you what it can do, and what it is not made for. The difference comes from one rule in the attention: which tokens a token may look at.

    **An everyday example.** When you read a letter to decide what it is about, you read all of it, and every word helps you understand the others. When you write a letter, you can only look back at what you have written so far: the next words are not there yet. An interpreter does both: they first listen to the whole sentence, then speak the translation word by word, looking back at what was said and at what they have said themselves.
    """)
    return


@app.cell(hide_code=True)
def predict_look():
    guess_look = L.guess(
        "third-token",
        question="GPT is trained to predict every next token of a text: at the place of the third token, it predicts the fourth. Which tokens may the third token look at with attention?",
        options={
            "before": "The first two: only the tokens before it",
            "right": "The first three: the two before it, and itself",
            "all": "All tokens of the text: also those after the third",
        },
        reveal="**The first three.** At its place, the third token is the last token of the text so far, and like every token with attention, it may look at itself and the tokens before it. It may not look at the fourth: that is the token it must predict, and with a look ahead it could simply copy it and learn nothing. When GPT writes, the tokens after the third are not even there yet. In BERT there is no such rule: every token looks at all tokens.",
    )
    guess_look
    return (guess_look,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **encoder** is a transformer in which every token looks at all tokens, before and after it. It reads a whole text, and gives every token a vector that knows the whole context. The best known is BERT, from Google in 2018: it learned by guessing words that were hidden in a text, from the words on both sides. Encoders are made to read: to say whether a review is positive, to find the names in a text, or to find the texts that fit a search. They are not made to write text.

        A **decoder** is a transformer in which every token looks only at itself and the tokens before it. That makes it a language model as in [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en): it predicts the next token, and writes by adding one token at a time. That is the GPT line: GPT-1 in 2018, with 117 million parameters, GPT-2 in 2019 with 1.5 billion, and GPT-3 in 2020 with 175 billion; each the same kind of decoder, but bigger, and trained on more text. From GPT-3 on, a task described in words, often with a few examples in the text, was enough: no extra training for each task. Most of today's large language models are decoders.

        An **encoder-decoder** has both: an encoder reads the input text, and a decoder writes the output text, one token at a time. Every token of the decoder looks at all tokens of the input, and at itself and the tokens before it in the output. The first transformer, from 2017, was one, made for translating; T5, from Google in 2019, does every task this way, from a text to a text.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **encoder** | | a transformer in which every token looks at all tokens; it reads a text | |
        | **decoder** | | a transformer in which every token looks only at itself and the tokens before it; it writes text, one token at a time | |
        | **encoder-decoder** | | an encoder that reads the input, and a decoder that writes the output and also looks at the whole input | |
        """),
        L.watch_out("\"Decoder\" and \"encoder\" mean something else here than in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en), where encoding turned text into bytes, and decoding bytes back into text. Here they are two kinds of transformer. And GPT is the name of OpenAI's models, but people also use it for other decoders built the same way."),
        mo.md(r"""
        The picture: who looks at whom, in an encoder and in a decoder, for a text of five tokens. Row by row, a token; a filled box is a token it looks at.
        """),
        mo.Html(C.look_picture({"title": "Who looks at whom", "encoder": "Encoder, such as BERT", "decoder": "Decoder, such as GPT",
                                "tokens": ["the", "cat", "sat", "down", "."], "caption": "Row: the token that looks. Column: a token it looks at.",
                                "alt": "Two grids of five by five boxes, for the tokens the, cat, sat, down and full stop. In the encoder, all 25 boxes are filled: every token looks at all tokens. In the decoder, only the boxes on and below the diagonal are filled: the first token looks at itself, the last at all five."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Which tokens token $i$ looks at, with places counted from 0:

    1. In an encoder: all tokens, before and after it.
    2. In a decoder: itself and the tokens before it, the places 0 up to and including $i$.
    3. In an encoder-decoder: a token of the decoder looks at all tokens of the input, and at the output as in step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    For every token of "the cat sat down .", the tokens it looks at, in an encoder and in a decoder; and for a translation, the second token the decoder has written. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_families():
    _tokens = ["the", "cat", "sat", "down", "."]
    # 1. an encoder: every token looks at all tokens
    print("encoder")
    for _i in range(len(_tokens)):
        print(_tokens[_i], "->", _tokens)
    # 2. a decoder: every token looks at itself and the tokens before it
    print("decoder")
    for _i in range(len(_tokens)):
        print(_tokens[_i], "->", _tokens[:_i + 1])
    # 3. an encoder-decoder that translates: the second output token looks at all of the input, and at output places 0 and 1
    _input = ["the", "cat"]
    _output = ["de", "kat"]
    print(_output[1], "->", _input, _output[:2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the encoder, every token gets the same list: all five tokens. In the decoder, the list grows: "the" looks only at itself, "cat" at "the" and itself, and the full stop at all five. `_tokens[:_i + 1]` stops just after place `_i`, so the token itself is in it. In the translation, "kat", the second token the decoder wrote, looks at the whole English input, and at "de" and itself in the Dutch output, to predict the next Dutch token.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_family():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The right family
    """)
    return


@app.cell(hide_code=True)
def choice_family_cell():
    choice_family = L.choice(
        "family",
        question="A program reads a whole product review and says whether it is positive or negative. Which family is made for exactly this?",
        options={
            "right": "An encoder: every word looks at the whole review",
            "decoder": "A decoder: it is the newest family, so the best at everything",
            "encdec": "An encoder-decoder: it turns the review into another text",
        },
        correct="right",
        why={
            "decoder": "A decoder can do it, by writing \"positive\" or \"negative\", but it is made to write, and every token sees only the words before it. Newer is not the same as made for this job.",
            "encdec": "The answer is a label, not a new text: there is nothing to write. A decoder on top adds work without need.",
            "right": "To judge a review you need all of it, and in an encoder every token looks at all tokens. Classifying a text is just what BERT and its family are made for.",
        },
    )
    choice_family
    return (choice_family,)


@app.cell(hide_code=True)
def text_exercise_can_see():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. May token i look at token j?

        Make `can_see(kind, i, j)` give `True` when, in a model of kind `kind`, the token at place `i` may look at the token at place `j`, and `False` when not. `kind` is `"encoder"` or `"decoder"`. Replace the two `...`, following steps 1 and 2 of the recipe: the first is one value, the second a comparison of `i` and `j`. The check calls `can_see("decoder", 1, 3)`, which should give `False`, and tries both kinds with other places.
        """),
        L.task("Make `can_see` say whether a token may look at another one, in an encoder and in a decoder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_can_see():
    box_can_see = L.code_box("can-see", lines=6)
    box_can_see
    return (box_can_see,)


@app.cell(hide_code=True)
def check_can_see(box_can_see):
    result_can_see = L.check("can-see", box_can_see.value)
    result_can_see
    return (result_can_see,)


@app.cell(hide_code=True)
def hints_can_see():
    L.hints(
        "can-see",
        [
            "In an encoder the answer does not depend on `i` and `j`. In a decoder it does: is place `j` at or before place `i`?",
            "A comparison such as `a < b` gives `True` or `False` itself. In a decoder, `j` may be smaller than `i`, or equal to it.",
            "`return True` for the encoder, and `return j ___ i` for the decoder.",
        ],
        why="In `can_see(\"decoder\", 1, 3)`, token 1 asks about token 3, which comes after it: a decoder may not look ahead, so `False`. `can_see(\"decoder\", 2, 2)` is `True`: a token may always look at itself.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_translate():
    mo.md(r"""
    ### Exercise 3. A translator writes
    """)
    return


@app.cell(hide_code=True)
def choice_translate_cell():
    choice_translate = L.choice(
        "translate",
        question="An encoder-decoder translates \"the cat sat down .\" into Dutch. Its decoder has written three Dutch tokens, and at the place of the third, it predicts the fourth. Which tokens may the third Dutch token look at?",
        options={
            "input": "Only the five English tokens: the translation comes from the input",
            "all": "All English and all Dutch tokens, also those after it",
            "right": "All five English tokens, and the first three Dutch tokens",
        },
        correct="right",
        why={
            "input": "The decoder also looks back at what it has written so far: the next Dutch word depends on the words before it, too.",
            "all": "The Dutch tokens after the third are not written yet: the decoder may only look back in its own text.",
            "right": "The decoder looks at the whole input, as step 3 of the recipe says, and in its own text, like every decoder, only at itself and what it wrote before.",
        },
    )
    choice_translate
    return (choice_translate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The attention of [Attention: every word looks at the others](/course/open/attn-idea?lang=en) lets every word look at all words; a decoder limits that to itself and the words before it, so that it can learn next-token prediction as in [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) without seeing the answer. At the intermediate level, you build that rule into attention yourself, and then a decoder of your own: a small GPT. In the next lesson, *Look inside a real LLM's next-token probabilities*, you load a real decoder, GPT-2, and see its probabilities for the next token.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_family, result_can_see, choice_translate):
    L.wrap_up(choice_family, result_can_see, choice_translate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
