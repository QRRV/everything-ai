# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-complex-rotation. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-complex-rotation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Also from outside the course:** the **sine and cosine of an angle**, where the point at angle $\theta$ on the circle of radius 1 around the origin is $(\cos\theta, \sin\theta)$; and **angles in radians**, where a full turn is $2\pi$ and a quarter turn $\pi/2$. `np.cos` and `np.sin` take angles in radians, and `np.pi` is $\pi$, as `math.pi` is. If either is rusty, read [Unit circle](https://en.wikipedia.org/wiki/Unit_circle) first.
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en) and [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_matrix():
    quick_matrix = L.choice(
        "quick-matrix",
        question="$\\mathbf{A}$ is the $2 \\times 2$ matrix with rows $(0, -1)$ and $(1, 0)$. What is its transpose $\\mathbf{A}^\\top$?",
        options={
            "same": "The same matrix",
            "right": "Rows $(0, 1)$ and $(-1, 0)$",
            "minus": "Rows $(0, 1)$ and $(1, 0)$",
        },
        correct="right",
        why={
            "same": "Only a symmetric matrix equals its transpose, and here the two numbers off the diagonal differ. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "minus": "The transpose swaps rows and columns; it changes no signs. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "right": "Row $i$ of $\\mathbf{A}^\\top$ is column $i$ of $\\mathbf{A}$. $\\mathbf{A}$ turns a vector a quarter counterclockwise; step 3 shows that $\\mathbf{A}^\\top$ turns it a quarter back.",
        },
    )
    quick_matrix
    return (quick_matrix,)


@app.cell(hide_code=True)
def quick_check_exponent():
    quick_exponent = L.choice(
        "quick-exponent",
        question="Which is the same as $e^{a} \\cdot e^{b}$?",
        options={
            "times": "$e^{a \\cdot b}$",
            "right": "$e^{a + b}$",
            "sum": "$e^{a} + e^{b}$",
        },
        correct="right",
        why={
            "times": "Multiplying powers of the same base adds the exponents; it does not multiply them. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "sum": "A product of two powers is not their sum. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "right": "Multiplying powers of the same base adds the exponents. This lesson uses that rule for angles.",
        },
    )
    quick_exponent
    return (quick_exponent,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many language models mark the place of each token by turning pairs of entries of its query and key vectors by an angle that grows with the position. That is rotary position embedding, RoPE, the subject of a later lesson. Turning a vector by hand takes a rotation matrix of sines and cosines. Complex numbers do the same with one multiplication, and they make the property that matters easy to see: after the turns, the dot product of a query and a key depends on their positions only through the difference between them. The same numbers are also the language of Fourier analysis, which splits a signal into waves.
    """)
    return


@app.cell(hide_code=True)
def predict_square():
    guess_square = L.guess(
        "square",
        question="The complex number $1 + i$ is the point $(1, 1)$: at 45 degrees, at distance $\\sqrt{2}$ from the origin. Where does $(1 + i)^2$ land?",
        options={
            "same": "At 45 degrees again, at distance 2",
            "right": "At 90 degrees, at distance 2",
            "double": "At 90 degrees, at distance $2\\sqrt{2}$",
        },
        reveal="**At 90 degrees, at distance 2.** $(1 + i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2i$: the point $(0, 2)$. Multiplying two complex numbers adds their angles, $45 + 45 = 90$ degrees, and multiplies their lengths, $\\sqrt{2} \\cdot \\sqrt{2} = 2$.",
    )
    guess_square
    return (guess_square,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Step 1: the complex plane.** A **complex number** $z = a + bi$ has a real part $a$ and an imaginary part $b$, with one new rule: $i^2 = -1$. Draw it as the point $(a, b)$ in the **complex plane**: the real part to the right, the imaginary part up. Its **length** $|z| = \sqrt{a^2 + b^2}$ is its distance to the origin; its **angle** $\varphi$ is its direction, measured counterclockwise from the positive real axis. With length $r$ and angle $\varphi$, $z = r(\cos\varphi + i\sin\varphi)$: the **polar form**. The **complex conjugate** $\bar z = a - bi$ is the mirror image of $z$ in the real axis: the same length, at angle $-\varphi$. And $z\bar z = (a + bi)(a - bi) = a^2 + b^2 = |z|^2$.

        **Step 2: multiplying turns.** Multiply out with $i^2 = -1$: $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$. Now take $w = \cos\theta + i\sin\theta$, of length 1 and at angle $\theta$:
        """),
        L.formula(r"\begin{aligned} w z &= (a\cos\theta - b\sin\theta) \\ &\quad + (a\sin\theta + b\cos\theta)\,i \end{aligned}"),
        mo.md(r"""
        Its real and imaginary parts are exactly what the **rotation matrix** makes of the vector $(a, b)$:
        """),
        L.formula(r"\begin{aligned} &\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} \\ &\quad = \begin{pmatrix} a\cos\theta - b\sin\theta \\ a\sin\theta + b\cos\theta \end{pmatrix} \end{aligned}"),
        mo.md(r"""
        So multiplying by a complex number of length 1 is a **rotation** by its angle: counterclockwise for a positive angle. A number of length $r$ is $r$ times a number of length 1, so in general, multiplying two complex numbers multiplies their lengths and adds their angles. Multiplying out also shows that the conjugate of a product is the product of the conjugates: $\overline{zw} = \bar z\,\bar w$.

        **Step 3: Euler's formula.** The number of length 1 at angle $\theta$ has a short name:
        """),
        L.formula(r"e^{i\theta} = \cos\theta + i\sin\theta"),
        mo.md(r"""
        This is **Euler's formula**. Why it is a power of $e$ takes more calculus than this lesson needs; here we take it as a fact and check it in NumPy below. It makes the rule of step 2 the rule for exponents: $e^{i\alpha}\, e^{i\beta} = e^{i(\alpha + \beta)}$, so turning by $\alpha$ and then by $\beta$ is turning by $\alpha + \beta$. The conjugate of $e^{i\theta}$ is $\cos\theta - i\sin\theta = e^{-i\theta}$: it turns back by $\theta$, as the transpose of the rotation matrix does, like $\mathbf{A}^\top$ in the quick check. The $n$ numbers $e^{2\pi i k/n}$ with $k = 0, \ldots, n-1$ lie at equal steps around the circle, and each gives 1 when raised to the power $n$: the **roots of unity**. For $n = 4$ they are $1$, $i$, $-1$ and $-i$, the quarter turns. They are the building blocks of the discrete Fourier transform.

        The picture: $z = 2 + i$, and $e^{i\pi/3} z$, with the same length and turned by 60 degrees. The dashed circle has radius 1.
        """),
        mo.Html(C.plane_picture({"title": "Multiplying by e^(iθ) turns", "re": "real part", "im": "imaginary part", "z": "z = 2 + i", "w": "e^(iπ/3) z", "angle": "60°", "lang": L.lang,
                                 "alt": "The complex plane with a dashed circle of radius 1. An arrow from the origin to z = 2 + i, and a second arrow of the same length, turned 60 degrees counterclockwise, to about 0.13 + 2.23i."})),
        mo.md(r"""
        **Step 4: a vector as pairs.** A real vector of even length can be read as pairs of entries next to each other, $(x_0, x_1), (x_2, x_3), \ldots$, and each pair as one complex number: $x_0 + i\,x_1$, $x_2 + i\,x_3$, and so on. Turn each pair by its own angle, and you have turned the vector piece by piece, without changing the length of any pair. That is how RoPE turns each query and key.

        **Step 5: the dot product after a turn.** The dot product of two complex vectors $\mathbf{u}$ and $\mathbf{v}$ conjugates one of them; this course conjugates the second: $\sum_j u_j \bar v_j$. The conjugate makes the dot product of a vector with itself its squared length, $\sum_j u_j \bar u_j = \sum_j |u_j|^2$, as $z\bar z = |z|^2$ in step 1. With $\mathbf{u}$ and $\mathbf{v}$ as columns, it is $\mathbf{v}^H \mathbf{u}$: the row $\mathbf{v}^H$ is the **conjugate transpose** of $\mathbf{v}$, its transpose with every entry conjugated. Now turn $\mathbf{u}$ by $\alpha$ and $\mathbf{v}$ by $\beta$. With the rules of steps 2 and 3, each term becomes
        """),
        L.formula(r"\begin{aligned} e^{i\alpha} u_j\, \overline{e^{i\beta} v_j} &= e^{i\alpha} e^{-i\beta}\, u_j \bar v_j \\ &= e^{i(\alpha - \beta)}\, u_j \bar v_j \end{aligned}"),
        mo.md(r"""
        The turns change the dot product only through the difference $\alpha - \beta$: turn both by the same extra angle, and nothing changes. That is what RoPE uses.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **complex number** | | a number $a + bi$, with a **real part** $a$ and an **imaginary part** $b$ | `3 + 4j`; `z.real`, `z.imag` |
        | $i$ | i | the imaginary unit: $i^2 = -1$ | `1j` |
        | **complex plane** | | the plane in which $a + bi$ is the point $(a, b)$ | |
        | $\lvert z\rvert$ | absolute value of z | the **length** of $z$: its distance to the origin | `abs(z)` |
        | $\varphi$ | phi | the **angle** of $z$: its direction, counterclockwise from the positive real axis, in radians | `np.angle(z)` |
        | **polar form** | | $z$ written with its length $r$ and its angle $\varphi$: $r e^{i\varphi}$ | |
        | **complex conjugate**, $\bar z$ | z bar | $a - bi$, the mirror image of $a + bi$ in the real axis | `np.conj(z)` |
        | **rotation** | | turning vectors around the origin by an angle, without changing their length | |
        | **rotation matrix** | | the matrix that does a rotation; in 2D, by $\theta$, the rows $(\cos\theta, -\sin\theta)$ and $(\sin\theta, \cos\theta)$ | |
        | **Euler's formula** | | $e^{i\theta} = \cos\theta + i\sin\theta$ | |
        | $e^{i\theta}$ | e to the i theta | the complex number of length 1 at angle $\theta$ | `np.exp(1j * theta)` |
        | **root of unity** | | a complex number $z$ with $z^n = 1$; the $n$ of them, $e^{2\pi i k/n}$, lie at equal steps around the circle | |
        | **conjugate transpose**, $\mathbf{v}^H$ | v H | the transpose with every entry conjugated; $\mathbf{v}^H \mathbf{u} = \sum_j u_j \bar v_j$ is the dot product of complex vectors | `np.dot(u, np.conj(v))` |
        """),
        L.watch_out("Python writes the imaginary unit as `j`, not `i`: `3 + 4j`, and `1j` on its own. A plain `j` is just a name."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forms that are new: complex numbers in Python and NumPy, with the length and the angle of $3 + 4i$, Euler's formula checked against cos and sin, and a quarter turn; and a slice with a step, which takes every second entry of an array.
    """)
    return


@app.cell
def example_complex():
    import numpy as np

    _z = 3 + 4j
    print("real", _z.real, " imaginary", _z.imag, " length", abs(_z), " angle", np.angle(_z))
    _theta = np.pi / 2  # a quarter turn, in radians
    _w = np.exp(1j * _theta)
    print("e^(i theta):", _w, "  cos + i sin:", np.cos(_theta) + 1j * np.sin(_theta))
    print("turned:", _w * _z, " length", abs(_w * _z))
    _x = np.array([10, 11, 12, 13, 14, 15])
    print("every second entry, from 0:", _x[0::2], " from 1:", _x[1::2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $3 + 4i$ has length 5 and angle 0.927 radians, 53 degrees. $e^{i\pi/2}$ comes out as `(6.123233995736766e-17+1j)`, the same as cos plus $i$ sin: that is $i$, up to a rounding error of about $10^{-16}$. Turned a quarter, $3 + 4i$ becomes $-4 + 3i$, again with a tiny rounding error, and still with length 5. In a slice, a third number is the step: `_x[0::2]` starts at index 0 and takes every second entry, `[10 12 14]`, and `_x[1::2]` starts at index 1, `[11 13 15]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rotate():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Turn a vector pair by pair

        Write `rotate_pairs(x, angles)`. `x` is a real NumPy array of even length, read as in step 4: entries next to each other, $(x_0, x_1), (x_2, x_3), \ldots$, are pairs, and pair $j$ is the complex number $x_{2j} + i\,x_{2j+1}$. `angles` is a list with one angle per pair, in radians. Turn pair $j$ by `angles[j]`, counterclockwise, and give back a real array with the same layout as `x`. The check calls `rotate_pairs(np.array([1, 0, 0, 1]), [np.pi / 2, np.pi / 2])`, which should give `[0, 1, -1, 0]`, and tries other vectors and angles.
        """),
        L.task("Write `rotate_pairs(x, angles)`: turn each pair of entries next to each other in `x` by its own angle, and give back a real array with the same layout."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rotate():
    box_rotate = L.code_box("rotate-pairs", lines=10)
    box_rotate
    return (box_rotate,)


@app.cell(hide_code=True)
def check_rotate(box_rotate):
    result_rotate = L.check("rotate-pairs", box_rotate.value)
    result_rotate
    return (result_rotate,)


@app.cell(hide_code=True)
def hints_rotate():
    L.hints(
        "rotate-pairs",
        [
            "Make one complex vector of all pairs at once, with the slices of the worked example: `x[0::2]` holds the real parts, `x[1::2]` the imaginary parts. After the turn, each real part goes back to an even place, with its imaginary part right after it.",
        ],
        why="Each pair becomes one complex number; multiplying by $e^{i\\theta}$ turns it by $\\theta$ and keeps its length; the real and imaginary parts go back where the pair came from. `np.array(angles)` makes the list of angles an array, because `1j * angles` does not work on a list. A rotation matrix per pair gives the same numbers.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_relative():
    mo.md(r"""
    ### Exercise 2. Two turns
    """)
    return


@app.cell(hide_code=True)
def choice_relative_cell():
    choice_relative = L.choice(
        "relative",
        question="You turn a complex vector $\\mathbf{q}$ by $3\\theta$ and a complex vector $\\mathbf{k}$ by $5\\theta$, and take their dot product $\\sum_j q_j \\bar k_j$. Which other two turns always give the same dot product?",
        options={
            "swap": "$\\mathbf{q}$ by $5\\theta$ and $\\mathbf{k}$ by $3\\theta$",
            "right": "$\\mathbf{q}$ by $4\\theta$ and $\\mathbf{k}$ by $6\\theta$",
            "double": "$\\mathbf{q}$ by $6\\theta$ and $\\mathbf{k}$ by $10\\theta$",
        },
        correct="right",
        why={
            "swap": "Then each term gets the factor $e^{5i\\theta} e^{-3i\\theta} = e^{2i\\theta}$ instead of $e^{-2i\\theta}$: a turn the other way, which in general changes the sum. The order counts, because the second vector is conjugated. See step 5.",
            "double": "Then each term gets the factor $e^{6i\\theta} e^{-10i\\theta} = e^{-4i\\theta}$: the difference has doubled. Only the difference of the angles counts, not their ratio. See step 5.",
            "right": "Each term gets the factor $e^{4i\\theta} e^{-6i\\theta} = e^{-2i\\theta}$, as with $3\\theta$ and $5\\theta$: both turns grew by $\\theta$, and the difference stayed $-2\\theta$. In RoPE that means: a query and a key that both move one place on keep their dot product.",
        },
    )
    choice_relative
    return (choice_relative,)


@app.cell(hide_code=True)
def text_exercise_real():
    mo.md(r"""
    ### Exercise 3. The dot product of real vectors
    """)
    return


@app.cell(hide_code=True)
def explain_real():
    explain_real_cell = L.explain(
        "real-dot",
        question="RoPE works with real vectors and their ordinary dot product. Read two real vectors $\\mathbf{x}$ and $\\mathbf{y}$ pair by pair, as in exercise 1, and turn pair $j$ of $\\mathbf{x}$ by $\\alpha_j$ and pair $j$ of $\\mathbf{y}$ by $\\beta_j$. Show that their dot product $\\mathbf{x} \\cdot \\mathbf{y}$ then depends on the angles only through the differences $\\alpha_j - \\beta_j$. Name each rule you use. Start with one pair: what is the real part of $u \\bar v$, for $u = a + bi$ and $v = c + di$?",
        model_answer="**One pair.** $u \\bar v = (a + bi)(c - di) = (ac + bd) + (bc - ad)\\,i$. Its real part, $ac + bd$, is the dot product of the pairs $(a, b)$ and $(c, d)$. So $\\mathbf{x} \\cdot \\mathbf{y}$ is the real part of $\\sum_j u_j \\bar v_j$, with $u_j$ and $v_j$ pair $j$ of $\\mathbf{x}$ and of $\\mathbf{y}$. **The turns.** Term $j$ becomes $e^{i\\alpha_j} u_j\\, \\overline{e^{i\\beta_j} v_j}$. **Conjugate of a product:** that is $e^{i\\alpha_j}\\, \\overline{e^{i\\beta_j}}\\, u_j \\bar v_j$, and the conjugate of $e^{i\\beta_j}$ is $e^{-i\\beta_j}$. **Exponent rule:** $e^{i\\alpha_j} e^{-i\\beta_j} = e^{i(\\alpha_j - \\beta_j)}$. So each term depends on the angles only through $\\alpha_j - \\beta_j$, and so do the sum and its real part, $\\mathbf{x} \\cdot \\mathbf{y}$. With all $\\alpha_j = \\beta_j$, nothing changes: turning both vectors the same way keeps their dot product.",
        points=[
            "the dot product of two pairs, $ac + bd$, is the real part of $u \\bar v$, so $\\mathbf{x} \\cdot \\mathbf{y}$ is the real part of $\\sum_j u_j \\bar v_j$",
            "you write the turned term $e^{i\\alpha_j} u_j \\cdot \\overline{e^{i\\beta_j} v_j}$ as $e^{i\\alpha_j} e^{-i\\beta_j}\\, u_j \\bar v_j$, with the conjugate of a product and the conjugate of $e^{i\\beta_j}$",
            "you use $e^{i\\alpha_j} e^{-i\\beta_j} = e^{i(\\alpha_j - \\beta_j)}$, so each term, the sum and its real part depend only on the differences",
        ],
    )
    explain_real_cell
    return (explain_real_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The rotation matrix is a matrix as in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en), and its transpose turns back, as the conjugate does; the rule $e^{a} e^{b} = e^{a+b}$ comes from [Exponents and logarithms](/course/open/math-exp-log?lang=en). The lesson on rotary position embeddings, RoPE, turns the pairs of each query and key the way your `rotate_pairs` turns a vector: pair $j$ of a token at position $m$ by $m$ times a fixed angle of that pair. By exercise 3, the dot product of a query at position $m$ and a key at position $n$ then depends on their positions only through $m - n$.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_relative, explain_real_cell, result_rotate):
    L.wrap_up(result_rotate, choice_relative, explain_real_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
