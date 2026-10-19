# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-top-k-top-p. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow nucleus sampling

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-top-k-top-p", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en) and [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hot():
    quick_hot = L.choice(
        "quick-hot",
        question="At a temperature of 2, your model's text gets odder. Why?",
        options={
            "right": "The unlikely characters get a bigger share",
            "never": "Characters that never came get a chance too",
            "random": "The model stops using its table and draws at random",
        },
        correct="right",
        why={
            "never": "A character that never came keeps probability 0 at every temperature. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
            "random": "The model still draws from its table, only with flatter probabilities. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
            "right": "Above 1, the gaps between the logits shrink, so the unlikely characters come up more often.",
        },
    )
    quick_hot
    return (quick_hot,)


@app.cell(hide_code=True)
def quick_check_sorted():
    quick_sorted = L.choice(
        "quick-sorted",
        question="What does `sorted([0.2, 0.5, 0.3])[-1]` give?",
        options={
            "first": "`0.2`: the first number of the list",
            "last": "`0.3`: the last number of the list",
            "right": "`0.5`: the largest number, at the end after sorting",
        },
        correct="right",
        why={
            "first": "`sorted` puts the numbers in order first, from small to large, and `[-1]` is the last of those. See [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "last": "`[-1]` is the last number of the sorted list, not of the list you gave. See [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "right": "`sorted` gives `[0.2, 0.3, 0.5]`, from small to large, and `[-1]` is the last of those: the largest.",
        },
    )
    quick_sorted
    return (quick_sorted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With sampling, every character that ever came after the current one can be drawn, even one that came there only once in the whole of the plays. Each of those rare draws has a small chance, but your model draws again for every character, so they add up: a capital in the middle of a word, a question mark after a single letter. A temperature below 1 makes them rarer, but never impossible. Another way is to cut the unlikely characters off altogether, and draw only from the likely ones. Language models often do this when they write.

    **An everyday example.** A small restaurant with a long menu cooks only the dishes that people order most on a busy night. Now and then a guest would have liked something else, but the kitchen stays fast and nothing odd comes out of it.
    """)
    return


@app.cell(hide_code=True)
def predict_space():
    guess_space = L.guess(
        "keep-space",
        question=f"After a space, {C.possible_after(' ')} different characters ever came in the plays. You keep only the most likely ones, as few as possible, that together hold 90% of the probability. How many do you think you keep?",
        options={
            "most": "About 50: the rare ones barely add up",
            "right": "About 20: about a third of the characters",
            "few": "About 5: the top few already hold most",
        },
        reveal=f"**About 20: {C.kept_after(' ', 0.9)} of the {C.possible_after(' ')}.** After a space, the probability is spread over many letters, so it takes {C.kept_after(' ', 0.9)} of them to reach 90%. The other {C.possible_after(' ') - C.kept_after(' ', 0.9)} together hold {L.number(100 * C.tail_share(' '), 1)}% of the probability: with plain sampling, about one draw in {round(1 / C.tail_share(' '))} after a space is one of them. After \"q\", one character already holds all of it.",
    )
    guess_space
    return (guess_space,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Top-k sampling** keeps only the k most likely characters of the row. The others get probability 0, and the k that are left are divided by their sum, so that they add up to 1 again. Then you draw as before. With k = 5, the model only ever draws one of the five most likely characters.

        **Top-p sampling** keeps the most likely characters, as few as possible, that together hold at least probability p, such as 0.9. It is also called nucleus sampling. How many it keeps depends on the row: after "q", one character holds all of it, so top-p keeps one; after a space, where the probability is spread out, it keeps many. Top-k keeps k after a space just as after "e", however spread out the row is.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **top-k sampling** | top k | drawing only from the k most likely tokens: the rest get probability 0, and the kept ones are divided by their sum | `cut = sorted(p)[-k]` |
        | **top-p sampling** | top p | drawing only from the most likely tokens, as few as possible, that together hold at least probability p | `if total >= top:` |
        | `sorted(p, reverse=True)` | | the numbers of `p` from large to small | `sorted([0.2, 0.5, 0.3], reverse=True)` gives `[0.5, 0.3, 0.2]` |
        | `np.where(p >= cut, p, 0)` | | an array with each number of `p` that is at least `cut`, and 0 in the other places | `np.where(p >= 0.3, p, 0)` |
        """),
        L.watch_out("Top-k and top-p only cut; the temperature reshapes the probabilities. They are often used together: first the temperature, then top-p. Do not make k or p too small: with k = 1, or a p that the most likely character reaches on its own, it is greedy decoding again."),
        mo.md(r"""
        The picture: the most likely characters after a space, from large to small. In colour are the ones that each method keeps; the grey ones it drops.
        """),
        mo.Html(C.kept_bars(" ", {"title": "After a space: what each method keeps", "k": "top-k, k = 5: keeps {n}",
                                  "p": "top-p, p = 0.9: keeps {n}", "rest": "and {n} more characters, all very unlikely, which both drop",
                                  "alt": f"Two rows of 30 bars, the most likely characters after a space from large to small, starting with t, a, s and h. In the first row, top-k with k = 5, the first five bars are in colour. In the second row, top-p with p = 0.9, the first {C.kept_after(' ', 0.9)} bars are in colour. The other bars are grey."})),
        mo.md(r"""
        What the model of the plays writes with seed 1, first with plain sampling, then with top-p, p = 0.9:
        """),
        mo.Html(C.text_block(C.sample_text(lambda p: p))),
        mo.Html(C.text_block(C.sample_text(lambda p: C.keep_top_p(p, 0.9)))),
        mo.md(f"""
        Neither is English: a bigram model only sees the last character. But in the first text, in {C.tail_pairs(C.sample_text(lambda p: p))} pairs the second character is one that top-p cuts off, such as a line that starts with a small "r"; in the second, there are none. The capitals in a row there are not odd draws: the plays mostly write the names of the speakers in capitals.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To keep only the likely characters of a row `p`, and draw from them:

    1. For top-k, find the cut: the k-th largest probability, `cut = sorted(p)[-k]`.
    2. Keep what is at least the cut, and make the rest 0: `keep = np.where(p >= cut, p, 0)`.
    3. For top-p, go over the probabilities from large to small and add them up: the one that brings the total to at least `top` is the cut. Stop there with `break`.
    4. Keep what is at least that cut, as in step 2.
    5. Divide by the sum, `keep / keep.sum()`, and draw with these probabilities.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A row with four characters and the probabilities 0.2, 0.05, 0.45 and 0.3: top-k with k = 2, then top-p with p = 0.9. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_cut():
    import numpy as np

    _p = np.array([0.2, 0.05, 0.45, 0.3])   # a row with four characters
    # 1. top-k with k = 2: the cut is the 2nd largest
    _cut = sorted(_p)[-2]
    # 2. keep what is at least the cut, 0 for the rest
    _keep = np.where(_p >= _cut, _p, 0)
    # 5. divided by the sum
    print(_keep / _keep.sum())
    # 3. top-p with p = 0.9: from large to small, until the total is at least 0.9
    _total = 0
    for _q in sorted(_p, reverse=True):
        _total = _total + _q
        if _total >= 0.9:
            _cut = _q
            break
    # 4. keep what is at least the cut
    _keep = np.where(_p >= _cut, _p, 0)
    # 5. divided by the sum, and ten draws
    print(_keep / _keep.sum())
    _rng = np.random.default_rng(0)
    print(_rng.choice(len(_p), size=10, p=_keep / _keep.sum()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Top-k keeps 0.45 and 0.3, and they become 0.6 and 0.4. For top-p the total goes 0.45, 0.75, 0.95: at 0.2 it reaches 0.9, so 0.45, 0.3 and 0.2 are kept, and divided by their sum, 0.95, they become about 0.47, 0.32 and 0.21. Character 1, with 0.05, is never drawn. If several characters share the k-th largest probability, `p >= cut` keeps them all, so top-k can then keep a few more than k.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keeps():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What does top-p keep?
    """)
    return


@app.cell(hide_code=True)
def choice_keeps_cell():
    choice_keeps = L.choice(
        "keeps",
        question="A row has the probabilities 0.5, 0.3, 0.15 and 0.05. Which characters does top-p with p = 0.9 keep?",
        options={
            "first": "Only the first: 0.5 is the largest",
            "two": "The first two: 0.5 and 0.3",
            "all": "All four: together they make 1",
            "right": "The first three: together 0.95",
        },
        correct="right",
        why={
            "first": "0.5 on its own is less than 0.9, so top-p goes on to the next one.",
            "two": "0.5 + 0.3 = 0.8, still less than 0.9.",
            "all": "Top-p stops as soon as the total reaches 0.9: the last 0.05 is not needed.",
            "right": "0.5 + 0.3 = 0.8 is not yet 0.9; with 0.15 the total is 0.95, so those three are kept, and 0.05 is cut off.",
        },
    )
    choice_keeps
    return (choice_keeps,)


@app.cell(hide_code=True)
def text_exercise_top_k():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Top-k

        `top_k(p, k)` keeps the k most likely probabilities of `p`, and divides them by their sum. The last two lines are there. Replace the `...` with step 1 of the recipe. The check calls `top_k(np.array([0.1, 0.5, 0.15, 0.2, 0.05]), 2)`, which should give about `[0, 0.714, 0, 0.286, 0]`, and tries other rows and other values of `k`.
        """),
        L.task("Make `top_k` take the k-th largest probability of `p` as `cut`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top_k():
    box_top_k = L.code_box("top-k", lines=8)
    box_top_k
    return (box_top_k,)


@app.cell(hide_code=True)
def check_top_k(box_top_k):
    result_top_k = L.check("top-k", box_top_k.value)
    result_top_k
    return (result_top_k,)


@app.cell(hide_code=True)
def hints_top_k():
    L.hints(
        "top-k",
        [
            "Step 1 of the recipe, as in the worked example.",
            "`sorted(p)` goes from small to large, so the largest is at the end. Which negative index is the k-th from the end?",
            "`cut = sorted(p)[___]`.",
        ],
        why="Sorted, `[0.1, 0.5, 0.15, 0.2, 0.05]` is `[0.05, 0.1, 0.15, 0.2, 0.5]`. The 2nd from the end is 0.2, so 0.5 and 0.2 are kept; divided by their sum, 0.7, they become 0.714 and 0.286.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_top_p():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Top-p

        `top_p(p, top)` keeps the most likely probabilities of `p` until together they hold at least `top`; it is called `top` and not `p` because `p` is already the row. The loop is there: it goes over the probabilities from large to small and adds them up. Replace the `...` with the rest of step 3: an `if` that makes `q` the cut and stops the loop. The check calls `top_p(np.array([0.3, 0.1, 0.4, 0.15, 0.05]), 0.8)`, which should give about `[0.353, 0, 0.471, 0.176, 0]`, and tries other rows and other values of `top`.
        """),
        L.task("Make `top_p` stop at the probability that brings the total to at least `top`, with that probability as `cut`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_top_p():
    box_top_p = L.code_box("top-p", lines=11)
    box_top_p
    return (box_top_p,)


@app.cell(hide_code=True)
def check_top_p(box_top_p):
    result_top_p = L.check("top-p", box_top_p.value)
    result_top_p
    return (result_top_p,)


@app.cell(hide_code=True)
def hints_top_p():
    L.hints(
        "top-p",
        [
            "Step 3 of the recipe, as in the worked example.",
            "Inside the loop, after the total is updated: is it at least `top` yet? Then this `q` is the cut, and the loop can stop.",
            "`if total >= ___:`, and inside it `cut = ___` and `break`.",
        ],
        why="From large to small: 0.4, 0.3, 0.15 and so on. The total goes 0.4, 0.7, 0.85: at 0.15 it reaches 0.8, so 0.4, 0.3 and 0.15 are kept. Their sum is 0.85, so they become 0.471, 0.353 and 0.176.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The temperature of [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en) reshapes the probabilities; top-k and top-p cut off the unlikely ones, and the sampling of [Sampling: let the model write](/course/open/lm-sampling?lang=en) draws from what is left. Chat services often use both: a temperature, and then top-p. In the next lesson, *How good is a language model?*, you measure how surprised your model is by real text.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keeps, result_top_k, result_top_p):
    L.wrap_up(choice_keeps, result_top_k, result_top_p)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
