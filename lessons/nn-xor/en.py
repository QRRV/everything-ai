# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nn-xor. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-xor", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A neuron](/course/open/nn-neuron?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fire():
    quick_fire = L.choice(
        "quick-fire",
        question="A neuron with a step has the weights 2 and −1 and the bias term −1. Does it fire for the input (1, 1)?",
        options={
            "right": "No: $z$ is exactly 0, and it fires only above 0",
            "zero": "Yes: $z$ is 0, and 0 counts as firing",
            "negative": "No: one of its weights is negative, so it never fires",
        },
        correct="right",
        why={
            "zero": "Work out $z = 2 \\cdot 1 - 1 \\cdot 1 - 1$. When does a step fire: at 0, or only above it? See the table in *The idea* of [A neuron](/course/open/nn-neuron?lang=en).",
            "negative": "A negative weight only pulls $z$ down. For (1, 0), for example, $z$ is 1. See the recipe in [A neuron](/course/open/nn-neuron?lang=en).",
            "right": "$z = 2 \\cdot 1 - 1 \\cdot 1 - 1 = 0$. A step fires only when $z$ is above 0.",
        },
    )
    quick_fire
    return (quick_fire,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="You raise the bias term of a neuron and leave its weights alone. What happens to the line where $z = 0$?",
        options={
            "turn": "It turns around a corner of the square",
            "right": "It shifts, and keeps its direction",
            "nothing": "Nothing: the bias term does not touch the line",
        },
        correct="right",
        why={
            "turn": "What sets the direction of the line: the weights or the bias term? See the slider in [A neuron](/course/open/nn-neuron?lang=en).",
            "nothing": "Move the slider in [A neuron](/course/open/nn-neuron?lang=en): what does the bias term do to the line?",
            "right": "The weights set the direction of the line; the bias term only shifts it.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [A neuron](/course/open/nn-neuron?lang=en), one neuron with a step computed AND, and with another bias term it computes OR. In 1958 Frank Rosenblatt presented the perceptron: at its core such a neuron, one that finds its weights by itself, from examples. It raised high hopes. In 1969 Marvin Minsky and Seymour Papert worked out in their book *Perceptrons* what such a neuron can and cannot learn. For years after that, few people worked on neural networks. In this lesson you look for that limit yourself.

    **An everyday example.** A stair light has a switch at the bottom and one at the top. Flip either switch and the light changes, whatever the other one is set to.
    """)
    return


@app.cell(hide_code=True)
def predict_xor():
    guess_xor = L.guess(
        "xor",
        question="Can one neuron with a step fire for the inputs (0, 1) and (1, 0), but not for (0, 0) and (1, 1)?",
        options={
            "yes": "Yes, with the right weights and bias term",
            "no": "No, whatever the weights and bias term",
            "sigmoid": "Only with a sigmoid instead of a step",
        },
        reveal="**No.** A neuron fires on one side of a straight line. (0, 1) and (1, 0) lie on one diagonal of the square, (0, 0) and (1, 1) on the other, and no straight line has both of one diagonal on one side and both of the other on the other side. A sigmoid does not help: it still draws the same straight line at 0.5. This task is called XOR, and it is the stair light.",
    )
    guess_xor
    return (guess_xor,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Two classes are **linearly separable** when one straight line splits them without mistakes; with more than two inputs, a flat plane. One neuron draws one such line, so it can only learn tasks whose classes are linearly separable. **XOR** gives 1 when exactly one of its two inputs is 1. Its classes are not linearly separable, and you do not have to try every line to know it. Suppose a neuron fires for (1, 0) and (0, 1), and not for (0, 0):
        """),
        L.formula(r"[[param:w_1]] + [[param:b]] > 0 \qquad [[param:w_2]] + [[param:b]] > 0 \qquad [[param:b]] \le 0"),
        mo.md(r"""
        Add the first two: $w_1 + w_2 + 2b > 0$, so $w_1 + w_2 + b > -b \ge 0$. That is $z$ for (1, 1): above 0, so the neuron fires for (1, 1) as well. No choice of weights and bias term makes one neuron fire for exactly the inputs with one 1: it cannot compute XOR.

        The **perceptron** of Rosenblatt is a neuron with a step that learns from examples: after each example it gets wrong, it moves its weights and bias term towards the label, by a learning rate $\eta$ times $y - \hat{y}$:
        """),
        L.formula(r"[[param:\mathbf{w}]] \leftarrow [[param:\mathbf{w}]] + [[hyper:\eta]]\,(y - [[pred:\hat{y}]])\,\mathbf{x} \qquad [[param:b]] \leftarrow [[param:b]] + [[hyper:\eta]]\,(y - [[pred:\hat{y}]])"),
        mo.md(r"""
        When it is right, $y - \hat{y}$ is 0 and nothing changes. When it should have fired and did not, $y - \hat{y}$ is 1, and the weights move in the direction of the input; the other way round, against it. If one straight line can split the classes, this always stops at weights without mistakes. If not, it never stops.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **linearly separable** | | two classes that one straight line, or a flat plane, splits without mistakes | |
        | **XOR** | ex-or | 1 when exactly one of two inputs is 1, else 0 | `x1 != x2` |
        | **perceptron** | | a neuron with a step that learns from examples, by moving its weights towards the label after each mistake (Rosenblatt, 1958) | `w = w + lr * (y - pred) * x` |
        | $\leftarrow$ | becomes | the new value on the left is worked out from the old ones on the right | `w = w + ...` |
        """),
        L.watch_out("XOR is not hard for a computer: `x1 != x2` does it. It is hard for one neuron, because a neuron can only draw one straight line."),
        mo.md(r"""
        Try it yourself. In the field, the <span class="c-param">orange point</span> sets the second weight $w_2$ (left to right) and the bias term $b$ (bottom to top) of a neuron with a step; its first weight is 1. Drag the point, or use the arrow keys. In the picture, the filled dots are the inputs XOR gives 1, the open dots the ones it gives 0, and a <span class="c-loss">purple ring</span> marks each input the neuron gets wrong.
        """),
    ])
    return


@app.cell(hide_code=True)
def plane_xor():
    def _frame(_w2, _b):
        _which = ", ".join(str(_c) for _c in C.wrong_corners(_w2, _b))
        return (C.picture(_w2, _b, {"title": "XOR and one straight line",
                                     "alt": f"The four corners of the square: (0, 1) and (1, 0) filled, (0, 0) and (1, 1) open, and the line where z = 0 of a neuron with the weights 1 and {L.number(_w2, 1)} and the bias term {L.number(_b, 1)}. A ring marks each corner the neuron gets wrong: {_which}."}),
                f"""With $w_2$ = {L.number(_w2, 1)} and $b$ = {L.number(_b, 1)}, the neuron gets <span class="c-loss ea-num">{C.mistakes(_w2, _b)}</span> of the 4 inputs wrong.""")
    _plane = L.plane("w2-bias", "w₂", "b", _frame, [round(-2 + 0.1 * _i, 1) for _i in range(41)],
                     [round(-2 + 0.1 * _j, 1) for _j in range(31)], start=(1, -0.5), task="Can you get no rings at all?",
                     count=C.mistakes, count_values=[0, 1, 2, 3], labels={"show": "Colour every point by its mistakes", "hide": "Hide the colours", "count": "wrong:"})
    _plane
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Wherever you put the point, at least one ring stays. The button shows this for every point at once: no part of the field is without colour.

    ### Recipe

    The perceptron learns like this, with a learning rate `lr`:

    1. For each example, predict with a step: `pred` is 1 when `np.dot(w, x) + b > 0`, else 0.
    2. Move the weights and the bias term: `w = w + lr * (y - pred) * x` and `b = b + lr * (y - pred)`. When `pred` is right, nothing changes.
    3. Go over all examples again, round after round, until a round has no mistakes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The perceptron on AND and on XOR: ten rounds over the four inputs each, starting from zeros, with a learning rate of 0.5. After each round the cell keeps how many inputs it got wrong in that round. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_perceptron():
    import numpy as np

    _inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for _name, _labels in [("AND", [0, 0, 0, 1]), ("XOR", [0, 1, 1, 0])]:
        _w = np.zeros(2)
        _b = 0.0
        _per_round = []
        # 3. round after round
        for _round in range(10):
            _mistakes = 0
            for _i in range(4):
                _x = np.array(_inputs[_i])
                _y = _labels[_i]
                # 1. predict with a step
                if np.dot(_w, _x) + _b > 0:
                    _pred = 1
                else:
                    _pred = 0
                if _pred != _y:
                    _mistakes = _mistakes + 1
                # 2. move towards the label
                _w = _w + 0.5 * (_y - _pred) * _x
                _b = _b + 0.5 * (_y - _pred)
            _per_round.append(_mistakes)
        print(_name, _per_round, _w, _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On AND the perceptron makes 1, 3, 3, 2 and 1 mistakes in its first five rounds, and none after that: it has found the weights 1 and 0.5 and the bias term −1, a line with only (1, 1) above it. On XOR it never gets there: from the third round on it gets all four inputs wrong, round after round. Each mistake moves the line, and the move fixes one input by breaking another.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_line():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. One line or not?

    Six tasks on two inputs, each 0 or 1. Put each one in a group: one straight line can split its 1s from its 0s, or no straight line can. Draw the four corners if it helps, or use the field above.
    """)
    return


@app.cell(hide_code=True)
def sort_line_cell():
    sort_line = L.sort("one-line")
    sort_line
    return (sort_line,)


@app.cell(hide_code=True)
def check_line(sort_line):
    result_line = L.check("one-line", sort_line)
    result_line
    return (result_line,)


@app.cell(hide_code=True)
def hints_line():
    L.hints(
        "one-line",
        [
            "For each task, mark the corners that give 1. Is there a line with those on one side and the rest on the other?",
            "One corner, three corners or one side of the square can be cut off with a line. What about two corners on a diagonal?",
            "One line: AND, OR, \\_\\_\\_ and \\_\\_\\_. No line: XOR and \\_\\_\\_.",
        ],
        why="**One line:** AND (one corner), OR and NAND (three corners), and the first input (one side). **No line:** XOR and same, whose 1s sit on one diagonal and 0s on the other.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One step of the perceptron

        Finish `perceptron_step(x, y, w, b, lr)`: it gets one example, its inputs `x` and its label `y`, the weights `w`, the bias term `b` and the learning rate `lr`. The prediction `pred` is already there. Replace the `...` with step 2 of the recipe, so that the function gives back the new weights and the new bias term. The check calls `perceptron_step([1, 1], 1, [0.0, 0.0], 0.0, 0.5)`: that should give `([0.5, 0.5], 0.5)`. It also tries examples the neuron gets right, ones where it fires but should not, and one with three inputs.
        """),
        L.task("Make `perceptron_step` give back the weights and the bias term after one step of the perceptron."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("perceptron-step", lines=13)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("perceptron-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "perceptron-step",
        [
            "Step 2 of the recipe, as in the worked example, with `lr` instead of 0.5.",
            "`y - pred` is 0, 1 or −1. What does it do to the step when the prediction is right?",
            "`w = w + lr * (___) * x` and `b = b + ___`.",
        ],
        why="With `w` and `b` all 0, $z$ is 0 and the neuron does not fire, but the label is 1: `y - pred` is 1. So each weight grows by 0.5 times its input, 1, and the bias term by 0.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_follows():
    mo.md(r"""
    ### Exercise 3. What follows?
    """)
    return


@app.cell(hide_code=True)
def choice_follows_cell():
    choice_follows = L.choice(
        "follows",
        question="A neuron with a step fires for (1, 0) and (0, 1), so $w_1 + b > 0$ and $w_2 + b > 0$. It does not fire for (0, 0), so $b \\le 0$. What follows for (1, 1), where $z = w_1 + w_2 + b$?",
        options={
            "below": "$z$ is 0 or below, so it does not fire for (1, 1)",
            "depends": "Nothing: it depends on the weights you choose",
            "right": "$z$ is above 0, so it fires for (1, 1) as well",
        },
        correct="right",
        why={
            "below": "Add the first two: $w_1 + w_2 + 2b > 0$. What does that say about $w_1 + w_2 + b$, when $b \\le 0$? See *The idea*.",
            "depends": "The three facts hold for every neuron that fires for (1, 0) and (0, 1) but not for (0, 0). Add the first two. See *The idea*.",
            "right": "$w_1 + w_2 + 2b > 0$, so $w_1 + w_2 + b > -b \\ge 0$. Every such neuron fires for (1, 1) too, so none computes XOR.",
        },
    )
    choice_follows
    return (choice_follows,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A neuron](/course/open/nn-neuron?lang=en) one neuron drew one straight line. That is exactly its limit: it can only learn classes that a straight line splits, and XOR is not one of them. The way out is not a better neuron but more of them. In the next lesson, *A hidden layer and ReLU*, neurons feed into other neurons, and together they bend the boundary.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_follows, result_line, result_step):
    L.wrap_up(result_line, result_step, choice_follows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
