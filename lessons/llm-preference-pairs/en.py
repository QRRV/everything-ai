# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-preference-pairs. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-preference-pairs", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en) and [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="The sigmoid of a logit $z$ is 0.8. What is the sigmoid of $-z$?",
        options={
            "right": "0.2",
            "minus": "−0.8",
            "same": "0.8",
        },
        correct="right",
        why={
            "minus": "A sigmoid is always between 0 and 1. A minus sign on the logit turns the probability around: $\\sigma(-z) = 1 - \\sigma(z)$. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "same": "The sigmoid grows with $z$, so $-z$ gives a smaller probability: $\\sigma(-z) = 1 - \\sigma(z)$. See [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en).",
            "right": "$\\sigma(-z) = 1 - \\sigma(z)$. Today: the probability that a is preferred to b, and that b is preferred to a, add up to 1.",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def quick_check_both():
    quick_both = L.choice(
        "quick-both",
        question="Your model before SFT and after SFT both wrote a story for *Write a story with the word: kite.*, and `words_used` says True for both. Which story does that check call better?",
        options={
            "after": "The one after SFT: that model did more instructions",
            "longer": "The longer one: it had more room for the word",
            "right": "Neither: the check only says whether each one did it",
        },
        correct="right",
        why={
            "after": "On this instruction both did what was asked; the check looks at one answer at a time. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
            "longer": "`words_used` only looks at whether the words are there, not at the length. See [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en).",
            "right": "A pass and a pass say nothing about which is better. Today you only compare answers where the check tells them apart.",
        },
    )
    quick_both
    return (quick_both,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    For SFT, someone has to write a good answer to every instruction. Often it is easier to say which of two answers is better. The people who made the data of InstructGPT did that: they ranked several answers of the model to one prompt, from best to worst (Ouyang and others, 2022). Two answers to the same prompt, with a verdict on which is better, are a **preference pair**. A program can give that verdict too, when it can check what was asked. And to turn many such verdicts into one score per answer or per model, there is a model from 1952: the **Bradley-Terry model**. Chatbot Arena ranks chat models with it, from the votes of people who compared two answers (Chiang and others, 2024).

    **An everyday example.** A football league. A team that beats a strong team shows more than one that beats a weak team. From who beat whom, you can give every team a strength, and from two strengths say how likely one team is to beat the other.
    """)
    return


@app.cell(hide_code=True)
def predict_odds():
    guess_odds = L.guess(
        "odds",
        question="Answer A is preferred to answer B 3 times out of 4. Answer B is preferred to answer C 3 times out of 4. If each answer has one score, as in the Bradley-Terry model, how often is A preferred to C?",
        options={
            "same": "3 times out of 4",
            "right": "9 times out of 10",
            "always": "Always",
        },
        reveal="**9 times out of 10.** Odds of 3 to 1 twice in a row give odds of 3 × 3 = 9 to 1: in the Bradley-Terry model, the odds of two steps multiply. Why, the next part shows.",
    )
    guess_odds
    return (guess_odds,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **A preference pair** is an instruction, an answer that is **chosen** and one that is **rejected**. A checker gives one only where it tells two answers apart: one passes and one fails. Two passes or two fails say nothing, so they give no pair.

    **The Bradley-Terry model** gives every answer, or every model, a **score** $s$. The probability that $a$ is preferred to $b$ is the sigmoid of the difference of their scores:
    """)
    return


@app.cell(hide_code=True)
def formula_bt():
    mo.vstack([L.formula(r"P(a \succ b) = \sigma(s_a - s_b)"), L.formula(r"\sigma(z) = \frac{1}{1 + e^{-z}}")])
    return


@app.cell(hide_code=True)
def picture_bt():
    mo.vstack([
        mo.md("The picture: the probability against the difference of the scores, with the three players of the worked example below: A, B and C, with the scores ln 3, 0 and −ln 3."),
        mo.Html(C.curve_picture([(math.log(3), "A − B, B − C"), (2 * math.log(3), "A − C"), (-2 * math.log(3), "C − A")],
                                {"title": "Bradley-Terry: the sigmoid of a difference", "x": "difference of the scores", "y": "probability preferred", "lang": L.lang,
                                 "alt": "An S-shaped curve from 0 to 1 that passes 0.5 at a difference of 0; points at a difference of 1.1 (0.75), 2.2 (0.9) and −2.2 (0.1)."})),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    Only the difference counts: add the same number to every score, and every probability stays as it was. Equal scores give 0.5. And $s_a - s_b$ is the logit of the probability, the log of the odds, so the differences add up along a chain and the odds multiply: that is the 9 to 1 of the guess.

    **Fitting the scores.** Give every player a score of 0, and change the scores so that what happened becomes more likely: the mean of $-\log P(\text{winner} \succ \text{loser})$ over all comparisons is the cross-entropy of the outcomes, a loss like any other. Gradient descent makes it smaller. A reward model, in the next lesson, is trained with the same loss.

    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **preference pair** | an instruction with a chosen and a rejected answer | `(chosen, rejected)` |
    | **chosen**, **rejected** | the answer that is preferred, and the one that is not | `answers[i]`, `answers[j]` |
    | **Bradley-Terry model** | a score per player; the sigmoid of the difference of two scores is the probability that the first is preferred | `bt_prob(s_a, s_b)` |
    | $a \succ b$ | "$a$ is preferred to $b$" | |
    | $s_a$ | the score of $a$ | `scores["a"]` |

    ### Recipe

    To rank answers or models from comparisons:

    1. Collect comparisons: two answers to the same instruction, and which is better, from people or from a check.
    2. Leave out the ties: a check that passes or fails both says nothing.
    3. Give every player a score, and the probability $\sigma(s_a - s_b)$ to every comparison.
    4. Fit the scores: make the mean of $-\log P(\text{winner} \succ \text{loser})$ as small as you can.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Three players with the scores $\ln 3$, 0 and $-\ln 3$, about 1.1, 0 and −1.1.
    """)
    return


@app.cell
def example_bt():
    _s = {"A": math.log(3), "B": 0.0, "C": -math.log(3)}
    for _a, _b in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")):
        print(_a, _b, round(1 / (1 + math.exp(-(_s[_a] - _s[_b]))), 4))
    _shifted = {_k: _v + 5 for _k, _v in _s.items()}   # 5 more for every player
    print(round(1 / (1 + math.exp(-(_shifted["A"] - _shifted["C"]))), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A is preferred to B 3 times out of 4, and so is B to C: their scores are $\ln 3$ apart, odds of 3 to 1. A and C are $2 \ln 3$ apart, odds of 9 to 1: 0.9. C to A is the other side, 0.1. With 5 more for every player, A to C is still 0.9.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prob():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The probability of a preference

        Complete `bt_prob(score_a, score_b)`: step 3. Give back the probability that the player with `score_a` is preferred to the one with `score_b`. The check calls `bt_prob(1.0, 0.0)`, and others.
        """),
        L.task("Make `bt_prob` give the Bradley-Terry probability of two scores."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prob():
    box_prob = L.code_box("bt-prob", lines=5)
    box_prob
    return (box_prob,)


@app.cell(hide_code=True)
def check_prob(box_prob):
    result_prob = L.check("bt-prob", box_prob.value)
    result_prob
    return (result_prob,)


@app.cell(hide_code=True)
def hints_prob():
    L.hints(
        "bt-prob",
        [
            "The sigmoid of [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en), with `math.exp` for $e$ to a power. Which number goes in? Not a score, but the difference of the two: the first minus the second.",
            "`return 1 / (1 + math.exp(-(___)))`, with the difference in the blank.",
        ],
        why="Only the difference of the scores counts, and a higher score for `a` gives a higher probability for `a`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Pairs from a checker

        Complete `make_pairs(answers, hits)`: steps 1 and 2. `answers` are answers to one instruction, `hits` says for each whether it passed the check. Give back a list of preference pairs `(chosen, rejected)`: one for every answer that passed and every answer that failed. The order of the pairs does not matter. The check calls `make_pairs(["A", "B", "C"], [True, False, False])`, and others.
        """),
        L.task("Make `make_pairs` give a pair for every answer that passed and every answer that failed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("make-pairs", lines=8)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("make-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "make-pairs",
        [
            "Answer `i` can be chosen when it passed, when `hits[i]` is True; answer `j` can be rejected when it failed. Both must hold for the same `i` and `j`.",
            "`if ___ and ___:`, and under it `pairs.append(...)` with a tuple of the two answers: the chosen one first.",
        ],
        why="A pair says which of two answers is better, so it needs one that passed and one that failed. Two passes or two fails would be pairs that say nothing.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The loss of the scores

        Complete `bt_loss(games, scores)`: step 4. `games` is a list of `(winner, loser)` pairs of names, `scores` a dict from names to scores. Give back the mean, over the games, of minus the log of the probability that the winner is preferred. The check calls `bt_loss([("lora", "base"), ("sft", "base"), ("base", "sft"), ("lora", "sft")], {"base": 0.0, "sft": 1.0, "lora": 1.5})`, and others.
        """),
        L.task("Make `bt_loss` give the mean loss of the scores on the games."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("bt-loss", lines=9)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("bt-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "bt-loss",
        [
            "For every game, look up the two scores in `scores`; the sigmoid of the score of the winner minus that of the loser is the probability `p` that the winner is preferred.",
            "The loss of one game is minus the log of `p`, with `math.log`: 0 when `p` is 1, and large when the winner was unlikely. Add it to `total`; the last line takes the mean.",
        ],
        why="The loss is small when the scores agree with who won. It is the cross-entropy of the outcomes, so it can be fitted like any loss.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _h = C.arena()
    mo.vstack([
        mo.md(f"""
        ### An arena of three models

        In [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en) and [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en), three models answered the same {len(_h['base'])} test instructions: the base model of the capstone, it after full SFT, and it with LoRA. Wherever exactly one of two models did an instruction, the check says which answer is better: a comparison. The button fits a score to every model with your `bt_prob`, by gradient descent on the mean loss, {C.number(C.STEPS, 0, L.lang)} steps with a learning rate of {C.number(C.RATE, 1, L.lang)} from scores of 0, with the score of the base model held at 0; your `bt_loss` gives the loss before and after. The verdicts are those of your own runs of both demos when you have them, on the same model of the capstone; else those of the course's runs.
        """),
    ])
    return


@app.cell(hide_code=True)
def fit_button_cell():
    fit_button = mo.ui.run_button(label="Fit the scores")
    fit_button
    return (fit_button,)


@app.cell(hide_code=True)
def demo_fit(box_loss, box_prob, fit_button, result_loss, result_prob):
    fitted = None
    _note = None
    if result_prob.state != "correct" or result_loss.state != "correct":
        _note = "*Finish exercises 1 and 3 first: the demo fits with your `bt_prob`, and measures with your `bt_loss`.*"
    elif fit_button.value:
        _hits = C.arena()
        _games = C.comparisons(_hits)
        _prob = run_code(box_prob.value).names["bt_prob"]
        _loss = run_code(box_loss.value).names["bt_loss"]
        _scores = C.fit(_games, _prob)
        if not _games:
            _note = "*In this arena, no two models differ on any instruction: there is nothing to compare, and so nothing to fit.*"
        elif _scores == "prob":
            _note = "*Your `bt_prob` stops with an error during the fit, or gives no probability between 0 and 1. Does it give a number between 0 and 1 for any two scores?*"
        else:
            try:
                fitted = {"hits": _hits, "games": _games, "scores": _scores, "start": _loss(_games, {_m: 0.0 for _m in C.MODELS}),
                          "end": _loss(_games, _scores)}
            except Exception:
                _note = "*Your `bt_loss` stops with an error on the games of the arena. Does it work for a few hundred games?*"
    mo.md(_note) if _note else None
    return (fitted,)


@app.cell(hide_code=True)
def demo_result(fitted):
    _fit = fitted if fitted is not None else C.course_fit()   # no mo.stop: the course's fit stands in meanwhile
    if not _fit["games"]:   # the learner's runs differ nowhere: the note of the cell before says so
        _fit = C.course_fit(standin=True)
    _names = {"base": "base model", "sft": "full SFT", "lora": "LoRA"}
    _short = {"base": "base", "sft": "SFT", "lora": "LoRA"}   # short, so the labels of the picture fit
    _h, _s, _w = _fit["hits"], _fit["scores"], C.wins(_fit["games"])
    _rows = [f"| {_names[_m]} | {sum(_h[_m])} of {len(_h[_m])} | {C.number(_s[_m], 2, L.lang)} |" for _m in C.MODELS]
    _pairs, _gap = [], 0.0
    for _a, _b in (("sft", "base"), ("lora", "base"), ("lora", "sft")):
        _n = _w[(_a, _b)] + _w[(_b, _a)]
        _seen = _w[(_a, _b)] / _n if _n else None
        _gap = max(_gap, abs(_seen - C.sigmoid(_s[_a] - _s[_b])) if _n else 0.0)
        _pairs.append(f"| {_names[_a]} against {_names[_b]} | {_w[(_a, _b)]} to {_w[(_b, _a)]} | {C.number(_seen, 2, L.lang) if _n else '–'} | {C.number(C.sigmoid(_s[_a] - _s[_b]), 2, L.lang)} |")
    _whose = "your runs" if _h["source"] == "yours" else "the course's runs, on its model of the capstone's demo run"
    _stand = "" if fitted is not None else " The fit of the course, with its own functions, stands in until you fit with yours."
    mo.vstack([
        mo.md(f"From {_whose}: {len(_fit['games'])} comparisons. The loss went from {C.number(_fit['start'], 3, L.lang)}, the log of 2 for scores of 0, to {C.number(_fit['end'], 3, L.lang)}.{_stand}\n\n"
              "| model | instructions done | score |\n|---|---|---|\n" + "\n".join(_rows)),
        mo.md("| comparison | wins | share won | Bradley-Terry |\n|---|---|---|---|\n" + "\n".join(_pairs)
              + ("\n\nHere the three shares happen to fit one set of scores: the odds of the two models against the base model, divided, give the odds between them." if _gap < 0.005 else
                 f"\n\nThe probabilities of the scores differ from the shares by up to {C.number(_gap, 2, L.lang)}: three scores cannot fit all three shares at once.")),
        mo.Html(C.curve_picture([(_s[_a] - _s[_b], f"{_short[_a]} − {_short[_b]}") for _a, _b in (("sft", "base"), ("lora", "base"), ("lora", "sft"))],
                                {"title": "The fitted scores on the curve", "x": "difference of the scores", "y": "probability preferred", "lang": L.lang,
                                 "alt": "The sigmoid of the difference of two scores, with a point for every two models of the arena."})),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Three scores have to fit three pairs of models at once, and the shares that were seen need not agree with each other: each is measured on a few dozen comparisons. When they do not, not every share comes out exactly. What the fitted scores always keep is every model's number of wins: the probabilities of its comparisons add up to the wins it really had.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shift():
    mo.md(r"""
    ### Exercise 4. Only differences
    """)
    return


@app.cell(hide_code=True)
def choice_shift_cell():
    _lora = C.course_fit()["scores"]["lora"]
    choice_shift = L.choice(
        "shift",
        question=f"The course's fit held the score of the base model at 0, and gave LoRA a score of {C.number(_lora, 2, L.lang)}. Had it held the base model at 10 instead, which score would LoRA have got?",
        options={
            "same": f"{C.number(_lora, 2, L.lang)}: the same",
            "times": f"{C.number(10 * _lora, 2, L.lang)}: ten times as much",
            "right": f"{C.number(10 + _lora, 2, L.lang)}",
        },
        correct="right",
        why={
            "same": "Then LoRA would be preferred to the base model with the sigmoid of a difference of almost −8: nearly never. Only differences count, so every score moves by the same 10.",
            "times": "Multiplying scores changes their differences, and so every probability. Adding the same number to every score changes nothing: LoRA moves by 10 too.",
            "right": "Every score moves by the same 10, so every difference, and every probability, stays as it was. That is why the fit may hold one score at any number.",
        },
    )
    choice_shift
    return (choice_shift,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The sigmoid of [From score to probability: the sigmoid](/course/open/ml-sigmoid?lang=en), now of a difference of two scores; and the check of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), now as a judge between two answers. Your project `llm` has `preference_pairs.py` now. Next, a reward model learns such a score for every answer with the same loss, and DPO uses the preference pairs of your `make_pairs` without a reward model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_prob, result_pairs, result_loss, choice_shift):
    L.wrap_up(result_prob, result_pairs, result_loss, choice_shift)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
