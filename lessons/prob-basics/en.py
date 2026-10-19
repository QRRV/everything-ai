# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson prob-basics. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-basics", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Dictionaries](/course/open/py-dicts?lang=en) and [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="A program counts the words of \"to be or not to be\" in a dictionary `counts`. What is `counts[\"to\"]`?",
        options={
            "once": "1",
            "right": "2",
            "words": "6",
        },
        correct="right",
        why={
            "once": "Each key is in a dictionary only once, but what is its value? See exercise 3 of [Dictionaries](/course/open/py-dicts?lang=en).",
            "words": "That is the number of words. What does the value under the key \"to\" count? See exercise 3 of [Dictionaries](/course/open/py-dicts?lang=en).",
            "right": "\"to\" comes by twice, so its count goes up to 2.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`A` has shape (3, 4). What is the shape of `A.sum(axis=0)`?",
        options={
            "right": "(4,)",
            "rows": "(3,)",
            "one": "One number",
        },
        correct="right",
        why={
            "rows": "That is `axis=1`. Which axis disappears with `axis=0`? See *The idea* in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "one": "That is `A.sum()` without an axis. See *The idea* in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "right": "`axis=0` adds up down the rows: one sum per column, 4 of them.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A language model does not say which word comes next: it says how likely each word is. A spam filter says how likely it is that an email is spam. Models speak in probabilities, and a probability is something you can count: how often something happens, out of all tries.

    **An everyday example.** "70% chance of rain" means: on days like this one, it rained on about 7 out of 10.
    """)
    return


@app.cell(hide_code=True)
def predict_sixes():
    guess_sixes = L.guess(
        "sixes",
        question="You throw a fair die 60 times. How often do you think a 6 comes up?",
        options={
            "exactly": "Exactly 10 times",
            "about": "About 10 times, a few more or less",
            "never-more": "Never more than 10 times",
        },
        reveal="**About 10, a few more or less.** One throw in six is a 6 on average, but every throw is chance: exactly 10 sixes comes up in only about 1 in 7 series of 60 throws, and more than 10 in about 4 in 10. See below what more throws do.",
    )
    guess_sixes
    return (guess_sixes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Count how often something happened, and divide by the number of tries: that share is its probability. It lies between 0, never, and 1, always.
        """),
        L.formula(r"P(A) = \frac{\text{how often } A \text{ happened}}{\text{number of tries}}"),
        mo.md(r"""
        A probability distribution lists every outcome with its probability. Every try has exactly one outcome, so the probabilities of all outcomes add up to 1. When every outcome is equally likely, you can also count without trying: a fair die has six faces, so each has the probability 1/6, about 0.167.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **probability** | | how likely something is, as a number from 0 to 1: how often it happens, out of all tries | `count / total` |
        | **probability distribution**, or **distribution** | | every outcome with its probability; together they add up to 1 | `{"heads": 0.7, "tails": 0.3}` |
        | $P(A)$ | P of A | the probability of $A$ | |
        """),
        L.watch_out("Divide by the number of tries, not by the number of different outcomes. 7 heads in 10 throws is 7 / 10 = 0.7, not 7 / 2."),
        mo.md(r"""
        2000 throws of a die, always the same ones. The bars show the share of each face in the first throws, as many as the slider says: 60 at the start, as in your guess. The other chart shows the smallest and the largest of the six shares for every number of throws. Move the slider, or drag across that chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _frame(n):
        n = int(n)
        shares = C.shares(n)
        return (C.picture(n, {"title": "Share of each face", "x": "face", "y": "share of the throws", "title_spread": "Smallest and largest share", "x_spread": "throws",
                              "alt": f"Six bars, one per face of the die, each as high as the share of its face in the first {n} throws: " + ", ".join(f"{_f}: {L.number(_s, 3)}" for _f, _s in zip(range(1, 7), shares)) + f". A dashed line at 1/6, about 0.167. A second chart shows the smallest and the largest share for every number of throws from 10 to 2000: from 0 and 0.3 after 10 throws, they close in on the dashed line; at {n} throws they are {L.number(min(shares), 3)} and {L.number(max(shares), 3)}."}),
                f"""After {n} throws, the six shares go from <span class="ea-num">{L.number(min(shares), 3)}</span> to <span class="ea-num">{L.number(max(shares), 3)}</span>; the dashed line is 1/6. Together they are exactly 1.""")

    explore_n = L.explore("throws", "throws", _frame, start=60, low=10, high=2000, step=10, role="", task="Go from 10 throws to 2000: what happens to the six shares?")
    explore_n
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With 10 throws, the shares jump all over the place, from 0 to 0.3; with 2000, all six lie close to 1/6, between 0.158 and 0.172. A probability you count from tries is an estimate, and more tries make it better.

    ### Recipe

    To get a distribution from tries:

    1. Count how often each outcome happened, in a dictionary.
    2. Add up all the counts: the number of tries.
    3. Divide each count by that number: the probability of each outcome.
    4. Check that the probabilities add up to 1.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The distribution of the letters in "to be or not to be": how likely each letter is, if you point at a random letter. Spaces do not count. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_letters():
    _text = "to be or not to be"
    # 1. count each letter
    _counts = {}
    for _ch in _text:
        if _ch != " ":
            if _ch in _counts:
                _counts[_ch] = _counts[_ch] + 1
            else:
                _counts[_ch] = 1
    print(_counts)
    # 2. the number of tries: all counts added up
    _total = 0
    for _ch in _counts:
        _total = _total + _counts[_ch]
    print(_total)
    # 3. each count divided by the total
    _probs = {}
    for _ch in _counts:
        _probs[_ch] = _counts[_ch] / _total
    print(_probs)
    # 4. together they make 1
    _check = 0
    for _ch in _probs:
        _check = _check + _probs[_ch]
    print(_check)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    There are 13 letters. `o` appears 4 times, so P(o) = 4 / 13, about 0.31; `r` and `n` only once, about 0.08 each. The six probabilities add up to `1.0`. The simplest language models start exactly like this: by counting how often each letter appears.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_marbles():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Marbles
    """)
    return


@app.cell(hide_code=True)
def choice_marbles_cell():
    choice_marbles = L.choice(
        "marbles",
        question="A bag has 3 red, 5 blue and 2 green marbles. You take one without looking. What is P(blue)?",
        options={
            "count": "5",
            "colours": "1/3",
            "right": "0.5",
        },
        correct="right",
        why={
            "count": "That is how many blue marbles there are. A probability is a share: of what? See the formula in *The idea*.",
            "colours": "That divides by the number of colours, but the colours are not equally likely. See the watch-out in *The idea*.",
            "right": "5 of the 10 marbles are blue: 5 / 10 = 0.5.",
        },
    )
    choice_marbles
    return (choice_marbles,)


@app.cell(hide_code=True)
def text_exercise_distribution():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. From counts to a distribution

        Finish `distribution(counts)`. It gets a dictionary with how often each outcome happened, and gives back a new dictionary with the probability of each outcome. Replace the `...`. The check calls `distribution({"heads": 7, "tails": 3})`: that should give `{"heads": 0.7, "tails": 0.3}`. It also tries other counts.
        """),
        L.task("Make `distribution` give back the probability of each outcome."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distribution():
    box_distribution = L.code_box("distribution", lines=9)
    box_distribution
    return (box_distribution,)


@app.cell(hide_code=True)
def check_distribution(box_distribution):
    result_distribution = L.check("distribution", box_distribution.value)
    result_distribution
    return (result_distribution,)


@app.cell(hide_code=True)
def hints_distribution():
    L.hints(
        "distribution",
        [
            "Steps 2 and 3 of the recipe: first the number of tries, then each count divided by it.",
            "A loop over the keys of `counts` adds up the total. A second loop fills a new dictionary. What goes under each key?",
            "`total = total + ___` in the first loop, and `probs[outcome] = ___ / total` in the second.",
        ],
        why="The first loop adds up all counts: 10 tries for the example. The second loop puts `counts[outcome] / total` under each key of a new dictionary: 7 / 10 = 0.7 and 3 / 10 = 0.3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Estimate from throws

        The box throws a die 500 times with `random.randint(1, 6)`, starting from a seed, and keeps the throws in the list `throws`. Put the share of sixes in `p_six`: an estimate of P(6). Replace the `...`. The last line is `p_six`, so the check sees it. The check also tries another seed.
        """),
        L.task("Put the share of sixes in `throws` in `p_six`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimate():
    box_estimate = L.code_box("estimate", lines=14)
    box_estimate
    return (box_estimate,)


@app.cell(hide_code=True)
def check_estimate(box_estimate):
    result_estimate = L.check("estimate", box_estimate.value)
    result_estimate
    return (result_estimate,)


@app.cell(hide_code=True)
def hints_estimate():
    L.hints(
        "estimate",
        [
            "The formula in *The idea*: how often a 6 happened, divided by the number of tries.",
            "Count the sixes with a loop over `throws` and an `if`. How many tries were there?",
            "`if t == 6:` then `sixes = sixes + 1`, and after the loop `p_six = sixes / ___`.",
        ],
        why="A loop over `throws` counts the throws that are a 6: 80 of them. Divided by `len(throws)`, 500, that gives 0.16: close to 1/6, but not exactly, as 500 throws are many, but not endlessly many.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Dictionaries](/course/open/py-dicts?lang=en) you counted words; divided by the total, counts become probabilities, and all of them together are a distribution. Later in the course, a model gives a distribution itself: a probability for every next word, or for every class. In the next lesson, *Conditional probability*, you count only among the cases where something else already happened.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_marbles, result_distribution, result_estimate):
    L.wrap_up(choice_marbles, result_distribution, result_estimate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
