# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-three-pass. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-three-pass", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The anatomy of a machine learning paper](/course/open/res-paper-anatomy?lang=en) and [The transformer block](/course/open/attn-transformer-block?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_claim():
    quick_claim = L.choice(
        "quick-claim",
        question="You want to know, in a minute, what a paper claims to add. Where do you look?",
        options={
            "method": "In the method section, where the model is described",
            "right": "In the abstract and the list of contributions",
            "appendix": "In the appendix, where the details are",
        },
        correct="right",
        why={
            "method": "The method says in detail what they did; it does not sum up the claim. See [The anatomy of a machine learning paper](/course/open/res-paper-anatomy?lang=en).",
            "appendix": "The appendix holds the details to rebuild it, not the claim. See [The anatomy of a machine learning paper](/course/open/res-paper-anatomy?lang=en).",
            "right": "The abstract says what is claimed, and the contributions list what the rest of the paper has to back up.",
        },
    )
    quick_claim
    return (quick_claim,)


@app.cell(hide_code=True)
def quick_check_block():
    quick_block = L.choice(
        "quick-block",
        question="In the transformer block you built, what happens to `x` right before it goes into the attention?",
        options={
            "mlp": "It goes through the MLP",
            "add": "It is added to the output of the attention",
            "right": "It goes through a layer normalization",
        },
        correct="right",
        why={
            "mlp": "The MLP comes after the attention, in the second branch. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
            "add": "The addition comes after the attention: `x + self.attention(self.norm1(x))`. See [The transformer block](/course/open/attn-transformer-block?lang=en).",
            "right": "Your block is pre-norm: `x + self.attention(self.norm1(x))`. Keep it in mind for the third pass of this lesson.",
        },
    )
    quick_block
    return (quick_block,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You built a GPT, lesson by lesson. The paper its transformer blocks come from, *Attention Is All You Need* (2017), is about fifteen pages on arXiv, dense with formulas and tables. Read front to back, you get stuck at the first formula you do not follow, or you lose the thread halfway through Section 3. Reading in three passes, each with its own goal and its own time, gets you further in less time, and tells you when to stop.

    **An everyday example.** Buying a house: first the photos and the price online, five minutes, to decide whether it is worth a visit. Then a visit of an hour: does it fit, what does it look like inside? And only for the house you really want, an inspection that takes hours, to find what is hidden in the walls.
    """)
    return


@app.cell(hide_code=True)
def predict_time():
    guess_time = L.guess(
        "third-pass-time",
        question="S. Keshav, who wrote down the three-pass method in 2007, gives each pass its own time. How long do you think the third pass, the one that goes through every detail, takes someone new to papers?",
        options={
            "minutes": "About ten minutes",
            "hour": "About an hour",
            "right": "Four or five hours",
        },
        reveal="**Four or five hours** for someone new to it, about an hour for an experienced reader. The first pass takes five to ten minutes, the second up to an hour. That is why you do the third pass only for the few papers you really need: many papers you put away after the first.",
    )
    guess_time
    return (guess_time,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In **three-pass reading** you read a paper three times, each time deeper, and after each pass you decide whether to go on.

        **The first pass** is a quick look: the title, the abstract, the introduction, the headings of the sections, and the conclusion. After it, you can answer five questions, which Keshav calls the five Cs:

        1. **Category:** what kind of paper is it? Use the kinds of contribution from [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en). <!-- termen:ok -->
        2. **Context:** which earlier work does it build on?
        3. **Correctness:** do its assumptions look right?
        4. **Contributions:** what does it claim to add?
        5. **Clarity:** is it well written?

        Then you decide: read on, or put it away.

        **The second pass** reads the whole paper with care, but skips proofs and long derivations. Look at every figure and table as in [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en), and mark the references you have not read. After it, you can tell someone else what the paper does and on what evidence.

        **The third pass** goes through every detail as if you were to rebuild the work yourself: every setting, every assumption, every step that the paper skips. Compare each choice with what you would have done. After it, you could reconstruct the whole paper from memory, and you know its hidden assumptions and its weak spots.
        """),
        mo.Html(C.passes({"title": "Three passes", "read": "Read:", "after": "After it:",
                          "passes": [("First pass", "5 to 10 minutes", "title, abstract, introduction, headings, conclusion", "the five Cs, and whether to read on"),
                                     ("Second pass", "up to an hour", "the whole text and every figure and table; no proofs yet", "what it does and on what evidence; the references to read"),
                                     ("Third pass", "4 to 5 hours at first, later about an hour", "every detail, as if you were to rebuild it", "how to rebuild it, and its hidden assumptions")],
                          "alt": "Three cards. First pass, 5 to 10 minutes: read the title, abstract, introduction, headings and conclusion; after it, the five Cs, and whether to read on. Second pass, up to an hour: read the whole text and every figure and table, no proofs yet; after it, what the paper does and on what evidence, and the references to read. Third pass, 4 to 5 hours at first, later about one: every detail, as if you were to rebuild it; after it, how to rebuild it, and its hidden assumptions."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **three-pass reading** | reading a paper three times, each time deeper: a quick look, a careful read, and every detail |
        | **five Cs** | the questions after the first pass: category, context, correctness, contributions, clarity <!-- termen:ok --> |
        """),
        L.watch_out("In the second pass you may skip what you do not follow yet: a proof, a derivation, a formula. Write down where it is, and come back to it in the third pass, if the paper gets one."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a paper:

    1. **First pass**, 5 to 10 minutes: read the title, the abstract, the introduction, the headings and the conclusion. Answer the five Cs, and decide whether to read on.
    2. **Second pass**, up to an hour: read the whole text, skipping proofs. Read every figure and table with care, and mark the references you have not read. Write down in a few sentences what the paper does, and on what evidence.
    3. **Third pass**, only for a paper you really need: go through every detail as if you were to rebuild it. Write down every setting and every assumption, and what the paper leaves out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first pass of [Attention Is All You Need](https://arxiv.org/abs/1706.03762), by Vaswani and seven others, 2017. Open it next to this lesson if you like: the abstract, the introduction, the headings of Sections 1 to 7, and the conclusion in Section 7, take about ten minutes.

    1. **Category:** a method paper: a new kind of model, the transformer, tested above all on translation. <!-- termen:ok -->
    2. **Context:** it builds on models for translation with an encoder and a decoder that read a sentence one word after another, and on attention, which such models already used; the introduction names both.
    3. **Correctness:** the claim rests on scores for translation on two standard test sets of 2014, English to German and English to French. The assumption: that those scores measure how good a translation is. That looks reasonable for now; how the scores were measured is for a later pass.
    4. **Contributions:** a model built only from attention, without reading word after word; better translations than the best earlier models; and far less training time.
    5. **Clarity:** short and compact, with a clear plan in its headings: the model in Section 3, the training in Section 5, the results in Section 6.

    **Decision:** read on. For you, this paper is worth a second pass: you built most of its parts yourself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which pass?

    Six things you can do with a paper. Put each with the pass it belongs to.
    """)
    return


@app.cell(hide_code=True)
def exercise_pass():
    sort_pass = L.sort("which-pass")
    sort_pass
    return (sort_pass,)


@app.cell(hide_code=True)
def check_pass(sort_pass):
    result_pass = L.check("which-pass", sort_pass)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "which-pass",
        [
            "For each task, ask how deep it goes: a look at the whole in minutes, a careful read of the text and the figures, or every detail as if you were to rebuild it.",
            "Two of the tasks belong to the quick look of the first pass: one is what you read in it, the other names two of the five Cs. Two need the text and the tables read with care, but not every setting.",
        ],
        why="**First pass:** read the headings and the conclusion and decide; name the kind of contribution and the earlier work. **Second pass:** check the rows of the results table; mark the references you still need. **Third pass:** find where each layer normalization goes and compare with your block; write down every setting to train the model yourself.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.md(r"""
    ### Exercise 2. Stuck in the second pass
    """)
    return


@app.cell(hide_code=True)
def choice_stuck_cell():
    choice_stuck = L.choice(
        "stuck",
        question="In your second pass of the transformer paper, you reach the formula for the learning rate in Section 5.3, and you do not follow it yet. What do you do?",
        options={
            "work": "Stop and work through the formula until you understand it, then read on",
            "right": "Note where it is, read on, and work it out in the third pass",
            "skip": "Skip it for good: the learning rate is a detail",
        },
        correct="right",
        why={
            "work": "That is the third pass. In the second, a formula you do not follow yet should not stop you from seeing the whole paper. See the watch-out in *The idea*.",
            "skip": "For rebuilding the model, the learning rate matters. You put it off, you do not drop it. See *The idea*.",
            "right": "The second pass is for the whole picture. The formula, a warmup followed by a slow decrease, is for the third pass.",
        },
    )
    choice_stuck
    return (choice_stuck,)


@app.cell(hide_code=True)
def text_exercise_differences():
    mo.md(r"""
    ### Exercise 3. A piece of the third pass

    Open [the paper](https://arxiv.org/abs/1706.03762) and read Sections 3.1, 3.4 and 5.4: a few paragraphs, less than a page in all. Compare what they say with the GPT you built.
    """)
    return


@app.cell(hide_code=True)
def explain_differences():
    explain_differences_cell = L.explain(
        "differences",
        question="Name three things the transformer of the paper does that your GPT does not do, or does differently. For each, write the section where you found it, and what your GPT does instead.",
        model_answer="**Section 3.1:** the model has two stacks, an encoder and a decoder; your GPT has one stack of blocks with causal attention. **Section 3.1:** each sub-layer gives LayerNorm(x + Sublayer(x)): the layer normalization comes after the addition; your block normalizes before the attention and before the MLP. **Section 3.4:** the embeddings are multiplied by the square root of d_model, which your GPT calls d, and one matrix is shared by the embeddings of the tokens in the encoder and the decoder and the layer before the softmax; your GPT has its own output head. **Section 5.4:** dropout of 0.1 on the output of each sub-layer and on the embeddings, and label smoothing: the target for the right token is about 0.9 instead of 1, and the rest is spread over the other tokens; your GPT has neither.",
        points=[
            "you name three differences, such as the encoder, the place of the layer normalization, the scaled and shared embeddings, dropout, or label smoothing",
            "for each, you name the section where you found it",
            "for each, you say what your GPT does instead",
        ],
    )
    explain_differences_cell
    return (explain_differences_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The anatomy of a machine learning paper](/course/open/res-paper-anatomy?lang=en) told you where each part is; the three passes tell you how deep to read each one, and when to stop. Your third pass found differences between the paper and your GPT: each is a choice someone made, and some were changed later because another choice worked better. At the end of the course you do the third pass for real, on the paper you reproduce. In the next lesson, *Change one thing at a time*, you learn how an experiment tests a choice like these.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stuck, explain_differences_cell, result_pass):
    L.wrap_up(result_pass, choice_stuck, explain_differences_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
