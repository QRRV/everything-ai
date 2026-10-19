# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-paper-anatomy. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-paper-anatomy", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Finding papers](/course/open/res-find-papers?lang=en) and [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_invented():
    quick_invented = L.choice(
        "quick-invented",
        question="A chatbot gives you a reference that looks complete: three authors, a title and a year. What do you do before you use it?",
        options={
            "cite": "Use it as it is: the authors, the title and the year are all there",
            "right": "Look it up on arXiv, Semantic Scholar or Google Scholar",
            "ask": "Ask the chatbot whether the paper really exists",
        },
        correct="right",
        why={
            "cite": "Chatbots can invent references that look complete. See [Finding papers](/course/open/res-find-papers?lang=en).",
            "ask": "The chatbot that invented a reference can also say it exists. Check it in a place that lists real papers. See [Finding papers](/course/open/res-find-papers?lang=en).",
            "right": "A paper you found yourself, in a search engine for papers, really exists, and then you can read it too.",
        },
    )
    quick_invented
    return (quick_invented,)


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="In a results table, model A's 91.8 is in bold, above model B's 91.2. The caption says that B's number was copied from an earlier paper. What do you check first?",
        options={
            "newer": "Whether A's paper is newer than B's",
            "bold": "Whether bold means that A is better on every test",
            "right": "Whether both were measured the same way",
        },
        correct="right",
        why={
            "newer": "Newer does not make a number comparable. See [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en).",
            "bold": "Bold only marks the best number in its column. See [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en).",
            "right": "A number from another paper may come from another test set, metric or amount of training data. Only rows measured the same way can be compared.",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You found a paper. It has thirty pages: formulas, tables, figures, and after the references an appendix. Read front to back like a book, it costs you hours, and you may still miss what you came for. But each part of a paper answers its own question. Know which part answers which, and you go straight to what you need.

    **An everyday example.** The manual of a washing machine: the first page says what the machine can do, a chapter says how to use it, and at the back, in small print, a table gives every program with its temperature and how long it runs. For the wool program, you go straight to that table.
    """)
    return


@app.cell(hide_code=True)
def predict_proof():
    guess_proof = L.guess(
        "adam-proof",
        question="Adam, an optimizer that many models are trained with, was introduced in a paper from 2014. In its appendix, the paper proves that Adam converges. The paper became one of the most cited in machine learning. Did anyone find a mistake in that proof?",
        options={
            "no": "No, the proof has held up",
            "right": "Yes, a few years later",
            "never": "Nobody could check it: the proof was never published",
        },
        reveal="**Yes, a few years later.** In 2018, Reddi, Kale and Kumar showed that the proof has a mistake, and gave a simple problem on which Adam does not find the best solution. Adam still works well in practice; the theorem just did not hold as written. An appendix often gets less attention than the main text, while it holds many of the details you need.",
    )
    guess_proof
    return (guess_proof,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Most papers in machine learning are built from the same parts, and each part answers its own question. The picture shows them in the usual order.
        """),
        mo.Html(C.anatomy({"title": "The parts of a paper, and the question each answers",
                           "main": "Main text: a page limit, at the big conferences about 8 to 10 pages (2026)",
                           "main_parts": [("Title and abstract", "What is claimed?"),
                                          ("Introduction", "Why does it matter, and what is new? Often ends with a list of contributions."),
                                          ("Related work", "How does it differ from earlier work?"),
                                          ("Method", "What exactly did they do or build?"),
                                          ("Experiments", "How was it measured, against what, and what came out?"),
                                          ("Conclusion and limitations", "What does it mean, and where does it stop?")],
                           "after": "After the main text: no page limit",
                           "after_parts": [("References", "What does it build on?"),
                                           ("Appendix A, B, C…", "The details: settings, extra results, proofs.")],
                           "alt": "A paper as two blocks. The main text, with a page limit: title and abstract, what is claimed; introduction, why it matters and what is new, often ending with a list of contributions; related work, how it differs from earlier work; method, what exactly they did or built; experiments, how it was measured, against what, and what came out; conclusion and limitations, what it means and where it stops. After the main text, without a page limit: references, what it builds on; appendix A, B, C, the details: settings, extra results, proofs."})),
        mo.md(r"""
        The names differ from paper to paper. The method may be called *Approach*, *Model* or the name of the method; the experiments *Results* or *Evaluation*; related work may come right after the introduction or just before the conclusion. The questions stay the same.

        The **list of contributions** at the end of the introduction says what the paper claims to add, often as (1), (2), (3). Together with the abstract, it is what the rest of the paper has to back up.

        The main text has a page limit, so the details move to the **appendix**, after the references, with sections lettered A, B, C. The main text points there: "see Appendix B". To rebuild an experiment you need exactly those details: the learning rate, the batch size, the number of epochs, the seeds, how the data was split. They are often only in the appendix, or in the code, if the authors published it.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **paper structure** | the parts most papers have, each answering its own question: abstract, introduction, related work, method, experiments, conclusion, references, appendix |
        | **list of contributions** | the list at the end of the introduction of what the paper claims to add |
        | **related work** | the part that describes earlier work on the same problem and how this paper differs from it |
        | **limitations** | what the paper did not test or cannot claim, often near the conclusion |
        | **appendix** | the part after the references, without a page limit, with the details: settings, extra results, proofs |
        """),
        L.watch_out("*Method* means more than one thing. In a paper, the method is the new way of doing something that the authors propose, as in a method paper, and the Method part says exactly what they did or built. In Python, a method is a function that belongs to an object, such as `append` in `scores.append(0.9)`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find something in a paper:

    1. Say what you are looking for: the claim, what was done, how it was measured, the settings to rebuild it, or where it stops.
    2. Go to the part that answers that question: the abstract and the list of contributions for the claim, the method for what was done, the experiments for how it was measured, the appendix for the settings, the conclusion and limitations for where it stops.
    3. Not there? Search the PDF for the word, such as *learning rate*, *batch* or *seed*, and follow every "see Appendix …".
    4. Write down what you found, with the section it is in, and what you could not find anywhere.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up paper, built like many real ones: **How wide is wide enough? MLPs for handwritten digits.** Its abstract: *We train MLPs with one hidden layer of 16 to 4,096 neurons on handwritten digits. Accuracy on the test set rises with the width up to 512 neurons, and stays at about 98% after that.* You want to rebuild that curve.

    1. **Looking for:** the claim, the model, the data and how it was measured, and the settings.
    2. **The claim** is in the abstract: accuracy rises up to 512 neurons, then stays at about 98%. The introduction ends with the contributions: (1) the curve from 16 to 4,096 neurons, (2) code to rebuild it. **The model** is in Section 3, the method: one hidden layer with ReLU, softmax at the output. **The data and how it was measured** are in Section 4, the experiments: 60,000 images to train on and 10,000 to test on, and Figure 2 shows the accuracy on the test set against the width, the mean of five seeds with their spread. **The settings** are not in Section 4, which says "training details in Appendix B".
    3. In Appendix B.1: SGD, a learning rate of 0.1, a batch size of 128, 20 epochs, seeds 0 to 4. A search for *validation* finds nothing: the paper does not say how it chose the learning rate and the 20 epochs.
    4. **Written down:** claim (abstract), model (Section 3), data and curve (Section 4, Figure 2), settings (Appendix B.1). Not found: how the learning rate and the number of epochs were chosen. Look in the code, or ask the authors.

    Four places in a paper of twenty pages, and one question the paper leaves open.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which part?

    Six pieces of another made-up paper, which compares one wide hidden layer with three narrow ones at the same number of parameters. Put each piece with the part of the paper it most likely comes from.
    """)
    return


@app.cell(hide_code=True)
def exercise_where():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "For each piece, ask which question it answers: what is new, how it differs from others, what the model does, what came out, where it stops, or how exactly it was set up.",
            "Look at the picture in *The idea*. One piece starts with a section number that has a letter in it; one lists what the paper adds, as (1) and (2); one names a table.",
        ],
        why="**Introduction:** \"Our contributions are…\". **Related work:** \"Lee et al. (2019)… Park et al. (2021)…; ours do not\". **Method:** how an image of 28 by 28 pixels goes into the first hidden layer. **Experiments:** Table 2, the accuracy for each width over five seeds. **Conclusion and limitations:** only handwritten digits were tested. **Appendix:** B.2, the training details.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_baselines():
    mo.md(r"""
    ### Exercise 2. Was the comparison fair?
    """)
    return


@app.cell(hide_code=True)
def choice_baselines_cell():
    choice_baselines = L.choice(
        "fair-baselines",
        question="The abstract of a paper says that its method beats every baseline. You want to know whether the baselines got as much tuning as the new method. Where is that most likely written?",
        options={
            "intro": "In the introduction, in the list of contributions",
            "right": "In the experiments section, or in the appendix with the settings",
            "related": "In the related work, where the papers of the baselines are named and compared",
        },
        correct="right",
        why={
            "intro": "The contributions say what the paper claims, not how it was measured. See *The idea*.",
            "related": "Related work says how the baselines differ in their idea, not how they were trained for this paper. See *The idea*.",
            "right": "How each method was tuned is part of how it was measured: in the experiments, or, often, in the appendix. If it is nowhere, write that down: it is worth knowing.",
        },
    )
    choice_baselines
    return (choice_baselines,)


@app.cell(hide_code=True)
def text_exercise_repro():
    mo.md(r"""
    ### Exercise 3. What do you need to rebuild it?

    At the end of this course you reproduce a paper. Before you start, you want to know where to find everything you need.
    """)
    return


@app.cell(hide_code=True)
def explain_repro():
    explain_repro_list = L.explain(
        "repro-list",
        question="Make a list of what you need to rebuild the main result of a paper, and for each item, the part of the paper where you look first.",
        model_answer="**The claim:** the abstract and the list of contributions. **The model:** the method. **The data, the baseline, the metric and the result to match:** the experiments, in the table or figure of the main result. **The settings:** the learning rate, the batch size, the number of epochs, the seeds and the split of the data: the appendix, or the code if the authors published it. **What is missing:** search the PDF, and write down what you could not find.",
        points=[
            "you name settings such as the learning rate, the batch size and the seeds, and look for them in the appendix",
            "you look for the claim in the abstract or the contributions, and for the result to match in the experiments",
            "you write down what you could not find",
        ],
    )
    explain_repro_list
    return (explain_repro_list,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en) you read the parts that hold the measurements; now you know where every other part is, and which question it answers. [Finding papers](/course/open/res-find-papers?lang=en) gave you the papers; now you find your way inside one. In the next lesson, *Reading a paper in three passes*, you read the transformer paper in three passes, from the claim to the details.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_baselines, explain_repro_list, result_where):
    L.wrap_up(result_where, choice_baselines, explain_repro_list)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
