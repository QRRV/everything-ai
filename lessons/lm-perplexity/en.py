# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-perplexity. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-perplexity", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How good is a language model?](/course/open/lm-nll?lang=en) and [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="On the same text, model A has an average NLL of 2.2 and model B one of 2.5. Which one predicts the text better?",
        options={
            "b": "B: a higher number means a better score",
            "right": "A: it is less surprised by the characters that really came",
            "same": "Neither: 2.2 and 2.5 are too close together to tell the two models apart",
        },
        correct="right",
        why={
            "b": "The NLL adds up surprises, minus the log of each probability: the higher the probabilities, the lower the NLL. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "same": "On the same text, a lower average NLL is simply better: 0.3 per character over a whole text is a big difference. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "right": "Lower is better: A gave the characters that came higher probabilities. Today you turn such a number into one you can picture.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Text comes from a source with entropy $H(p)$. A model $q$ is trained on it. How low can its cross-entropy $H(p, q)$ go?",
        options={
            "zero": "To 0: with enough training, the model is never surprised",
            "kl": "To the KL divergence between $p$ and $q$: training removes the entropy",
            "right": "To $H(p)$: the KL divergence on top of it is never negative",
        },
        correct="right",
        why={
            "zero": "Only a source without any uncertainty has entropy 0; text is not like that. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "kl": "The KL divergence is what the model adds on top of the entropy, and it is never negative. See [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en).",
            "right": "$H(p, q) = H(p) + D_{\\mathrm{KL}}(p \\,\\|\\, q)$, and the KL divergence is at least 0: the entropy of the text is a floor that no model gets below. Perplexity has that floor too.",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An average NLL of 2.20 is hard to picture. Is that good? How much better is 2.20 than 2.45? Papers and model cards describe language models with another number, made from the same loss: **perplexity**. It answers a question you can picture: between how many tokens is the model choosing, as if it chose evenly between them? A model with a perplexity of 9 is, on average, as surprised by each token as by the roll of a die with 9 sides.
    """)
    return


@app.cell(hide_code=True)
def predict_ratio():
    guess_ratio = L.guess(
        "lower-loss",
        question="Model A has an average NLL of 2.2 per character, model B one of 2.4: A's loss is about 8% lower. How much lower is A's perplexity?",
        options={
            "same": "Also about 8% lower",
            "right": "About 18% lower",
            "less": "About 3% lower",
        },
        reveal="**About 18% lower.** Perplexity is e to the power of the loss: $e^{2.2} \\approx 9.0$ and $e^{2.4} \\approx 11.0$. A loss that is 0.2 lower divides the perplexity by $e^{0.2} \\approx 1.22$, whatever the loss was: on the perplexity scale, every step down in loss counts as a fixed factor.",
    )
    guess_ratio
    return (guess_ratio,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Give the model a text of $n$ tokens and take the average NLL, as in [How good is a language model?](/course/open/lm-nll?lang=en): the mean of minus the log of the probability it gave each token that really came. **Perplexity** is e to the power of that:
        """),
        L.formula(r"\begin{aligned} [[loss:\mathrm{PPL}]] &= e^{[[loss:\text{average NLL}]]} \\ &= \exp\Big(-\frac{1}{n} \sum_{t=1}^{n} \log [[pred:P(x_t \mid x_{<t})]]\Big) \end{aligned}"),
        mo.md(r"""
        Put another way: give every token one and the same probability, chosen so that the average NLL stays the same; the perplexity is 1 divided by that probability, which is called the geometric mean of the probabilities. A model that gives each of 65 characters the probability $1/65$ has a perplexity of exactly 65; a model that is always sure and always right, 1. With $\log_2$ and 2 to the power, you get the same number: the base cancels out.

        Perplexity is per token, so it depends on the tokenizer. A model that counts in characters and one that counts in the tokens of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) choose between other things, and their perplexities cannot be compared as they are. To compare, spread the total NLL of the text over the same unit, such as its characters: the average NLL per token, times the number of tokens, divided by the number of characters. And perplexity has a floor: the cross-entropy of a model is at least the entropy of the text, so its perplexity is at least $e^{H(p)}$, with $H$ in nats ($2^{H(p)}$ with $H$ in bits). That floor holds on new text; on the text it was trained on, a model can get below it.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **perplexity** | | e to the power of the average NLL: the number of tokens a model is choosing between, as if it chose evenly | `np.exp(nll)` |
        | $\mathrm{PPL}$ | P P L | perplexity | `ppl` |
        | $x_{<t}$ | x before t | the tokens before token $t$: $x_1$ to $x_{t-1}$ | `tokens[:t]` |
        """),
        L.watch_out("A perplexity is only meaningful together with its text and its tokens. On the text it was trained on, a model looks better than on new text; and a perplexity per GPT-2 token is not a perplexity per character. Compare models on the same unseen text, in the same unit."),
        mo.md(r"""
        The picture: the curve from average NLL to perplexity, with three models on it. The counted bigram model of Tiny Shakespeare, on the text it counted, has an average NLL of 2.45: a perplexity of 11.6, a choice between about 12 characters. A model that gives all 65 characters the same probability is at 65, and one that is always sure and right at 1. The curve rises ever faster: at the high end, a little more loss means many more choices.
        """),
        mo.Html(C.ppl_picture({"title": "e to the power of the loss", "x": "average NLL", "y": "perplexity", "perfect": "always right: {}",
                               "bigram": "bigram model: {}", "uniform": "all 65 equal: {}", "decimal": ".",
                               "alt": "The curve of perplexity, e to the power of the average NLL, from 1 at a loss of 0 to 65 at a loss of 4.17. On it: always right, 1; the bigram model, 11.6 at a loss of 2.45; and all 65 characters equally likely, 65."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The perplexity of a model on a text:

    1. For every token of the text, the probability the model gave it, given the tokens before it.
    2. The average NLL: minus the mean of the logs of those probabilities.
    3. The perplexity: e to the power of the average NLL.
    4. Compare only on the same unseen text, in the same unit; if the tokens differ, first spread the total NLL over the characters.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The counted bigram model on "the king": the probabilities of its 7 pairs, the average NLL and the perplexity. `C.bigram_prob(a, b)` gives the probability of `b` after `a`. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_perplexity():
    import numpy as np

    _text = "the king"
    # 1. the probability of every character after the one before it
    _probs = [C.bigram_prob(_text[_i - 1], _text[_i]) for _i in range(1, len(_text))]
    print(np.round(_probs, 4))
    # 2. the average NLL
    _nll = -np.mean(np.log(_probs))
    print(round(_nll, 4))
    # 3. the perplexity
    _ppl = _nll  # complete: e to the power of _nll
    print(round(_ppl, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The probabilities run from 0.355 for "e" after "h" to 0.0094 for "k" after a space: few words start with "k". Their average NLL is 1.9525. As it is, the last line prints that again, `1.952`: the loss, not the perplexity. Completed, with `np.exp(_nll)`, it prints `7.046`: on "the king", the bigram model is as unsure as a choice between about 7 characters, fewer than its 11.6 on the whole text, because its pairs "th", "he" and "e " are likely.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Two perplexities
    """)
    return


@app.cell(hide_code=True)
def choice_compare_cell():
    choice_compare = L.choice(
        "compare",
        question="On the same unseen text, model A has a perplexity of 9 per character, and model B one of 40 per GPT-2 token. Which model is better?",
        options={
            "a": "A: 9 is far lower than 40",
            "b": "B: it chooses between more tokens at every step, so it must know more about the text",
            "right": "That does not follow from these numbers: first bring them to the same unit",
        },
        correct="right",
        why={
            "a": "Lower is better only in the same unit. A GPT-2 token is often several characters, so B makes fewer choices for the same text, each among more options.",
            "b": "A higher perplexity means more uncertainty per step, not more knowledge; but B also takes fewer steps.",
            "right": "Spread B's total NLL over the characters of the text: with 4 characters per token, a perplexity of 40 per token is $40^{1/4} \\approx 2.5$ per character, and then B is the better one.",
        },
    )
    choice_compare
    return (choice_compare,)


@app.cell(hide_code=True)
def text_exercise_perplexity():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Perplexity

        Complete `perplexity(probs)`: it gets the probabilities a model gave the tokens that really came, as a list, and gives the perplexity. Replace the two `...`, following steps 2 and 3 of the recipe. The check calls `perplexity([0.5, 0.125])`, which should give `4.0`, and tries other lists, such as one with every probability $1/65$.
        """),
        L.task("Make `perplexity` give e to the power of the average NLL of a list of probabilities."),
    ])
    return


@app.cell(hide_code=True)
def exercise_perplexity():
    box_perplexity = L.code_box("perplexity", lines=7)
    box_perplexity
    return (box_perplexity,)


@app.cell(hide_code=True)
def check_perplexity(box_perplexity):
    result_perplexity = L.check("perplexity", box_perplexity.value)
    result_perplexity
    return (result_perplexity,)


@app.cell(hide_code=True)
def hints_perplexity():
    L.hints(
        "perplexity",
        [
            "Two lines from the worked example, with `probs` instead of `_probs`: the average NLL, and the completed step 3.",
            "The NLL needs `np.log` of the probabilities, their `np.mean`, and a minus; the perplexity is `np.exp` of the NLL.",
        ],
        why="For `[0.5, 0.125]`, the logs are −0.693 and −2.079, so the average NLL is 1.386, and $e^{1.386} \\approx 4$: one divided by the geometric mean of 0.5 and 0.125, which is $\\sqrt{0.0625} = 0.25$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_char():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Per character

        Complete `char_perplexity(avg_nll, n_tokens, n_chars)`: on a text of `n_chars` characters, cut into `n_tokens` tokens, a model has an average NLL of `avg_nll` per token. The function gives its perplexity per character. Replace the `...` with the NLL per character, following step 4 of the recipe. The check calls `char_perplexity(3.0, 100, 400)`, which should give about `2.117`, and tries other texts.
        """),
        L.task("Make `char_perplexity` turn an average NLL per token into a perplexity per character."),
    ])
    return


@app.cell(hide_code=True)
def exercise_char():
    box_char = L.code_box("char-perplexity", lines=6)
    box_char
    return (box_char,)


@app.cell(hide_code=True)
def check_char(box_char):
    result_char = L.check("char-perplexity", box_char.value)
    result_char
    return (result_char,)


@app.cell(hide_code=True)
def hints_char():
    L.hints(
        "char-perplexity",
        [
            "The total NLL of the text is the same, however you count: the average per token times the number of tokens.",
            "Divide that total by the number of characters, and `np.exp` in the last line does the rest.",
        ],
        why="For `(3.0, 100, 400)`, the total NLL is 3.0 × 100 = 300, so 0.75 per character, and $e^{0.75} \\approx 2.117$. The perplexity per token, $e^{3} \\approx 20.1$, is that number to the power 4: four characters per token.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The average NLL of [How good is a language model?](/course/open/lm-nll?lang=en) is the cross-entropy of [Cross-entropy and KL divergence](/course/open/info-cross-entropy-kl?lang=en) between the text and the model; perplexity is e to the power of it, a number of choices. Its floor is $e^{H(p)}$, set by the entropy of the text itself. Every language model from here on, your GPT included, gets a perplexity on unseen text, and for models with different tokens you first bring it to characters, as in exercise 3. The next lesson, *Convolution in detail: kernel size, stride and padding*, turns back to images: what a convolutional layer does with its kernel, how far it steps, and what happens at the edges.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_compare, result_perplexity, result_char):
    L.wrap_up(choice_compare, result_perplexity, result_char)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
