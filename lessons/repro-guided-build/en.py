# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-build. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-build", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Guided reproduction 1: read the paper and pick the claim](/course/open/repro-guided-claim?lang=en) and [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_verdict():
    quick_verdict = L.choice(
        "quick-verdict",
        question="Your plan says a run grokked when validation reaches 99% at a step at least 10 times as late as training. A run has t = 400 and v = 3,000. What is its verdict?",
        options={
            "grokked": "\"grokked\": validation got there much later than training",
            "right": "\"no delay\": 3,000 is less than 10 times 400",
            "not": "\"not reached\": validation needs more steps",
        },
        correct="right",
        why={
            "grokked": "Later, but not late enough by the plan: 10 times 400 is 4,000, and 3,000 is less. See [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).",
            "not": "Validation did reach 99%, at step 3,000. See [Guided reproduction 1](/course/open/repro-guided-claim?lang=en).",
            "right": "The plan decides, not how the curves look: that is why it was written before any run.",
        },
    )
    quick_verdict
    return (quick_verdict,)


@app.cell(hide_code=True)
def quick_check_kv():
    quick_kv = L.choice(
        "quick-kv",
        question="The paper's transformer has 4 heads, each with keys and values of its own. Your ModernGPT is made as `ModernGPT(vocab, d, h, kv_heads, layers)`. With `h = 4`, which `kv_heads` gives the paper's attention?",
        options={
            "one": "1: one key head and one value head, shared by all 4 query heads",
            "right": "4: as many key and value heads as query heads",
            "two": "2: each key and value head shared by 2 query heads, as in the GPT of your project",
        },
        correct="right",
        why={
            "one": "That is multi-query attention: it keeps less in memory, but it is not the paper's transformer. See [Grouped-query attention](/course/open/llm-gqa?lang=en).",
            "two": "That is grouped-query attention, a choice of your project's GPT, not of the paper. See [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en).",
            "right": "Then every query head has keys and values of its own, as in the paper; the demo below builds your ModernGPT so.",
        },
    )
    quick_kv
    return (quick_kv,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A reproduction tests the paper's claim only when it runs the paper's experiment: the same data, the same model and the same settings. Every setting you take from memory, or from a default, is a second experiment hidden in the first. So in this step you build the data exactly as the paper makes it, take each setting from the place in the paper that gives it, and write down where your model differs. The model is your own ModernGPT, adapted; the data and the settings go into your reproduction kit.
        """),
        draw.project_steps(["Read the paper and pick the claim", "Build the model and the data", "Sanity checks", "Reproduce the key figure",
                            "The ablation", "Compare with the paper", "The report"],
                           1, here_label="← today", title="Where your project stands: your reproduction"),
    ])
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "size",
        question="The paper's transformer has 2 layers, width 128 and 4 heads, and about 4 · 10⁵ parameters outside its embeddings. Your ModernGPT with the same 2 layers, width 128 and 4 heads, with SwiGLU and RoPE: how many?",
        options={
            "right": "About as many, close to 4 · 10⁵: SwiGLU has three matrices, but narrower ones",
            "more": "About twice as many: SwiGLU has three matrices where the paper's MLP has two",
            "fewer": "Far fewer: RoPE has no parameters, and the paper's table of places does",
        },
        reveal=f"**About as many: {C.number(C.NON_EMBEDDING_COURSE, 0, L.lang)}.** SwiGLU has three matrices, but each is narrower: a hidden width of 8/3 times 128 in place of 4 times 128, so its about 8 · 128² numbers are as many as those of the paper's two. A table of places sits outside the blocks, as the embeddings do, so it is not in the count either way.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_reading():
    mo.md(r"""
    ## Reading guide

    The third pass of the paper, for this step: Appendix A.1, about a page.

    1. **A.1.1, the data:** which equations, how each is written as tokens, and how the training data is chosen.
    2. **A.1.2, the model and its optimization:** every number of the model and of the optimizer. Write down each one with where it stands; you need them all in exercise 3.
    3. **What is not there:** which details of the transformer does the paper leave open?

    The paper's words next to the course's and the code's:

    | Paper | Course | Code |
    |---|---|---|
    | $\langle x \rangle \langle op \rangle \langle y \rangle \langle = \rangle \langle x \circ y \rangle$ | an equation as five tokens | `[a, p, b, p + 1]`, then `c` |
    | training data fraction | the share of all equations used for training | `fraction` |
    | width | the size of each token's vector, $d$ | `width` |
    | optimization budget | the most training steps | `steps` |
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **The data is a table.** For addition modulo $p$ the table has an equation $a + b = c$ for every $a$ and every $b$ from 0 to $p - 1$, with $c = (a + b) \bmod p$: for $p = 97$, $97^2 = 9{,}409$ equations. Each number is a token of its own, with no digits: the model sees symbols, and learns only from how they combine.

    **The split is random, and fixed.** Shuffle all equations with a seed, and take the first `int(fraction * 9409)` for training; the rest is the validation data. The same seed gives the same split, so a run can be repeated.

    Below is the whole table for $p = 97$, a cell for each equation: $a$ down, $b$ across. A filled cell is training data, an open cell validation data. It opens at 50%, with the cell $50 + 10$ pointed at. The slider sets the fraction; the button takes another seed.
    """)
    return


@app.cell(hide_code=True)
def play_table_cell():
    play_table = play.Table(
        C.training, lambda a, b: (a + b) % C.PAPER["p"], p=C.PAPER["p"], fraction=C.TABLE_FRACTION, point=list(C.POINT),
        decimal="," if L.lang == "nl" else ".",
        labels={"task_kick": L.w["task"], "task": "Point at open cells: where do the cells with the same answer lie, and how many of them are filled?",
                "alt": "The table of a + b modulo 97 as a grid of 97 by 97 cells, a down and b across; the cells of the training data are filled. The cell 50 + 10 is pointed at: the cells with its answer, 60, lie on a diagonal that goes on at the other side at the edge.",
                "train": "training data: the model sees the answer", "val": "validation data: the model has to find it",
                "ring": "the same answer as the cell you point at; the other cells fade", "fraction_name": "fraction", "seed": "Other seed", "of": "of",
                "counts": "equations in the training data", "point": "Point at a cell, or use the arrow keys.", "op": "+",
                "is_train": "training data", "is_val": "validation data: the model has to find this itself",
                "same": "the same answer in {n} cells, {k} of them filled"})
    mo.ui.anywidget(play_table)
    return


@app.cell(hide_code=True)
def text_idea_rest():
    mo.md(r"""
    The cells with the same answer lie on one diagonal: one more in $a$ and one less in $b$ keeps $a + b$ the same, and at the edge the diagonal goes on at the other side, as on a clock. Every answer has 97 cells; at 50%, on average about 48 of them are filled. The model never sees the open cells. Addition gives it one shortcut: $b + a$ has the same answer as $a + b$, so an open cell whose mirror cell, across the line from the top left to the bottom right, is filled can be got right from that one cell. At 50%, about half of the open cells have one; $50 + 10$ has none. All other open cells the model can get right only by finding the rule behind the diagonals, not by remembering cells. That is what the validation accuracy measures: the mirror cells give at most about half of it, the rule the rest.

    **The answer is the only target.** The model reads $a$, the operation, $b$ and $=$, and its logits at the place of $=$ are its guess for $c$. The paper computes the loss and the accuracy only there.

    **Every setting from the paper.** Each number of the model and the optimizer comes from the place in the paper that gives it, with that place written next to it. Where the paper is silent, your choice is a **deviation**: write it down, so that a difference in the results can be traced to it later.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **modular arithmetic** | arithmetic on a clock of $p$ hours: after $p - 1$ comes 0 again | `(a + b) % p` |
    | **training data fraction** | the share of all equations of the table used for training | `fraction` |
    | **deviation** | a way your reproduction differs from the paper, chosen or forced, and written down | `# deviation: ...` |
    | `random.shuffle(rows)` | puts the items of a list in a random order, in place; after `random.seed(seed)` always the same order | `random.shuffle(rows)` |

    ### Recipe

    To build the experiment of a paper:

    1. Make the data as the paper describes it: which examples, which tokens, which split.
    2. Take every setting of the model and the optimizer from the paper, with where it stands.
    3. Write down what the paper leaves open, and what you chose there.
    4. Build the model with those settings, and compare what you can count with the paper, such as its size.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One equation of the table for $p = 5$, as tokens, and the size of its split.
    """)
    return


@app.cell
def example_tokens():
    _p = 5
    _a, _b = 3, 4
    _c = (_a + _b) % _p
    print("tokens:", [_a, _p, _b, _p + 1], "answer:", _c)   # the operation is token p, the sign = is token p + 1
    print("equations:", _p * _p, "for training at 50%:", int(0.5 * _p * _p))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $3 + 4 = 7$, and on a clock of 5 hours that is 2. The numbers 0 to 4 are tokens 0 to 4, so the operation and the sign $=$ get the next two, 5 and 6: a vocabulary of $p + 2$ tokens. Of the 25 equations, 50% is 12.5, and `int` keeps 12.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dataset():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The data

        Complete `equations(p)`, every `(a, b, c)` of the table with `c = (a + b) % p`, row by row: `a` in the outer loop, `b` in the inner one, and `dataset(p, fraction, seed)`, which shuffles them after `random.seed(seed)` with `random.shuffle` and gives back the first `int(fraction * len(rows))` for training and the rest for validation. The check calls `dataset(3, 0.5, 0)`, and others. When it is right, it is saved in your project as `repro/data.py`.
        """),
        L.task("Complete `equations` and `dataset`: the table of addition modulo `p`, shuffled and split."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dataset():
    box_dataset = L.code_box("dataset", lines=17)
    box_dataset
    return (box_dataset,)


@app.cell(hide_code=True)
def check_dataset(box_dataset):
    result_dataset = L.check("dataset", box_dataset.value)
    result_dataset
    return (result_dataset,)


@app.cell(hide_code=True)
def hints_dataset():
    L.hints(
        "dataset",
        [
            "`equations` is two loops, one inside the other, and one `append` per pair. Which value does `c` get?",
            "`random.shuffle(rows)` changes `rows` itself and gives back nothing. Then a slice up to `n`, and one from `n`.",
        ],
        why="A split that depends on nothing but the seed lets every later run, and anyone else, use the same training data.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fraction():
    mo.md(r"""
    ### Exercise 2. Which fraction?
    """)
    return


@app.cell(hide_code=True)
def choice_fraction_cell():
    choice_fraction = L.choice(
        "which-fraction",
        question="Section 3.1 says the delay is typical \"for dataset sizes that were close to the minimal dataset size for which the network generalized\", and that for larger ones \"the training and validation curves tend to track each other more closely\". For addition, which fraction does your reproduction take?",
        options={
            "most": "90%: the more training data, the surer the result",
            "figure": "50%: the fraction of Figure 1",
            "right": f"{round(100 * C.FRACTION)}%: a small part of the table, where the delay should show",
        },
        correct="right",
        why={
            "most": f"With most of the table to train on, the curves track each other: in the course's run at 90%, training reached 99% at step {C.FRACTIONS[0.9]['t']} and validation at step {C.number(C.FRACTIONS[0.9]['v'], 0, L.lang)}, too soon for a delay of 10 times.",
            "figure": f"Figure 1 is division, and the paper finds that more complicated operations need more data. In the course's run of addition at 50%, training reached 99% at step {C.FRACTIONS[0.5]['t']} and validation at step {C.number(C.FRACTIONS[0.5]['v'], 0, L.lang)}: {round(C.FRACTIONS[0.5]['v'] / C.FRACTIONS[0.5]['t'])} times as late, still no delay by your plan.",
            "right": f"The paper finds the delay near the smallest fraction that still generalizes. In the course's run at 30%, training reached 99% at step {C.FRACTIONS[0.3]['t']} and validation at step {C.number(C.FRACTIONS[0.3]['v'], 0, L.lang)}: {round(C.FRACTIONS[0.3]['v'] / C.FRACTIONS[0.3]['t'])} times as late.",
        },
    )
    choice_fraction
    return (choice_fraction,)


@app.cell(hide_code=True)
def text_exercise_settings():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The settings

        Fill in `SETTINGS` with the paper's values, from Appendix A.1.2, and the fraction of exercise 2. Write each the way Python reads it: `1e-3` for $10^{-3}$, a tuple for the two betas, and a whole number for a count, such as `100_000` or `10**5` for $10^5$ (`1e5` is a float). Where the paper gives a rule, as for the batch, work it out for your data: the training data has thousands of equations. When it is right, it is saved in your project as `repro/settings.py`.
        """),
        L.task("Fill in `SETTINGS`: every setting of the paper, and your fraction."),
    ])
    return


@app.cell(hide_code=True)
def exercise_settings():
    box_settings = L.code_box("settings", lines=15)
    box_settings
    return (box_settings,)


@app.cell(hide_code=True)
def check_settings(box_settings):
    result_settings = L.check("settings", box_settings.value)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def hints_settings():
    L.hints(
        "settings",
        [
            "Appendix A.1.2 has two sentences with numbers: one about the transformer, one about the optimizer. Take them one at a time.",
            "The batch: 512, or half of the training data, whichever is smaller. How many training equations does your fraction give?",
        ],
        why="A setting with the place it comes from can be checked by anyone; one without it is a guess nobody can find again.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your `dataset` and your `SETTINGS`, from your project, and your ModernGPT made with those settings, before any training: how big the data is, how many parameters the model has outside its embeddings, and one batch of training equations through it. Where a module of your ModernGPT is missing or broken, the course's version stands in.
    """)
    return


@app.cell(hide_code=True)
def demo_build(result_dataset, result_settings):
    result_dataset, result_settings  # run again when an exercise saves a part of the kit
    _r = C.demo()
    if _r is None:
        _out = mo.md("*Finish exercises 1 and 3 first: then your data and your settings are in your project, and this cell uses them.*")
    elif isinstance(_r, str):
        _out = mo.md({"data": "*Your `repro/data.py` stops with an error. Did you change it after the check? Delete it, and check exercise 1 again.*",
                      "settings": "*Your `repro/settings.py` stops with an error. Did you change it after the check? Delete it, and check exercise 3 again.*",
                      "model": "*Your ModernGPT stops with an error with these settings. Did you change `llm/modern_gpt.py` or `repro/settings.py` after its check? Delete the file you changed, and check its exercise again.*"}[_r])
    else:
        _s = _r["settings"]
        _stand = ("The course's version stood in for: " + ", ".join(f"`{_x}`" for _x in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
        _out = mo.md(f"""
| | yours | what to expect |
|---|---|---|
| equations for training | {C.number(_r['train'], 0, L.lang)} | {round(100 * _s['fraction'])}% of {C.number(_s['p'] ** 2, 0, L.lang)} |
| equations for validation | {C.number(_r['val'], 0, L.lang)} | the rest |
| parameters outside the embeddings | {C.number(_r['params'], 0, L.lang)} | about 400,000 |
| a batch of tokens | {_r['batch'][0]} × {_r['batch'][1]} | {_s['batch']} equations of 4 tokens, without the answer |
| logits | {_r['logits'][0]} × {_r['logits'][1]} × {_r['logits'][2]} | one row of {_s['p'] + 2} for each place; the last one guesses `c` |
| loss on the answers, before training | {C.number(_r['loss'], 2, L.lang)} | ln {_s['p'] + 2} = {C.number(_r['chance'], 2, L.lang)}, for a model that guesses evenly |

{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the paper's settings, the data and the size of the model match the paper. Three parts of your model do not come from it: RMSNorm, RoPE and SwiGLU. The paper says only "a standard decoder-only transformer", so each is a deviation: write them down in your kit, and keep them in mind. If your results differ from the paper's, they are among the first places to look. The loss before training is close to that of a model that guesses evenly among all tokens: what the next step, the sanity checks, will look at first.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The plan of [Guided reproduction 1](/course/open/repro-guided-claim?lang=en), now with its data and its settings; your ModernGPT of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en), adapted to the paper; and the care of [From formula to code](/course/open/res-formula-to-code?lang=en), for every number of a paper. The next lesson, *Guided reproduction 3: sanity checks*, sends one batch through again and makes the model learn it by heart, before any long run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_dataset, choice_fraction, result_settings):
    L.wrap_up(result_dataset, choice_fraction, result_settings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
