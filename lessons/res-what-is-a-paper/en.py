# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-what-is-a-paper. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("res-what-is-a-paper", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_translate():
    quick_translate = L.choice(
        "quick-translate",
        question="A program translates text into other languages. It learned that from millions of translated sentences, with a neural network of many layers. What is the smallest circle it fits in?",
        options={
            "dl": "Deep learning: it is a neural network with many layers",
            "right": "Generative AI: it makes new text",
            "ml": "Machine learning: it learned from examples",
        },
        correct="right",
        why={
            "dl": "It is deep learning, but the third question of the recipe goes one circle further in: does it make something new? See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "ml": "It learned from examples, so it is machine learning; but ask the next two questions too. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "right": "Learned from examples, a neural network with many layers, and it writes new sentences: generative AI.",
        },
    )
    quick_translate
    return (quick_translate,)


@app.cell(hide_code=True)
def quick_check_tax():
    quick_tax = L.choice(
        "quick-tax",
        question="A program finds out what is wrong with a car by asking you questions, with rules that mechanics wrote down. Where does it fit?",
        options={
            "right": "AI, but not machine learning: people wrote its rules",
            "ml": "Machine learning: it knows what mechanics know",
            "none": "Nowhere: a program with rules is not AI",
        },
        correct="right",
        why={
            "ml": "Machine learning finds its rules in examples; here mechanics wrote them down. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "none": "Some AI follows rules that people wrote, like the chess program: finding a fault in a car is an expert's task. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "right": "The first question of the recipe: people wrote the rules, so it stops in the AI circle.",
        },
    )
    quick_tax
    return (quick_tax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    News about AI often starts with "a new study shows". Behind such a story there is usually a research paper. Whether you can trust the story depends on what kind of paper it is, where it appeared, and who checked it. Later in this course you read papers yourself, and at the end of the research part you repeat the result of one.

    **An everyday example.** Two links to a recipe for the same cake: one to a cookbook's website, where an editor read and corrected every recipe before it went up, and one to a recipe that someone posted the same evening. On your screen they look the same.
    """)
    return


@app.cell(hide_code=True)
def predict_review():
    guess_review = L.guess(
        "review",
        question="A paper passed peer review at a big AI conference: before the conference accepted it, a few other researchers read it and judged it. How sure can you be that its main result is right?",
        options={
            "certain": "Certain: experts checked everything in it",
            "right": "More likely than before, but not certain",
            "same": "No surer than before: review changes nothing",
        },
        reveal="**More likely than before, but not certain.** A few experts read the paper, ask for changes and catch many mistakes. But like the editor of the cookbook, who reads a recipe without baking the cake, they usually do not run the experiments again, they have little time, and they often disagree. In 2014 the big AI conference NeurIPS, then called NIPS, let two separate groups of reviewers judge the same papers, and the two groups disagreed about many of them. A result becomes really solid when other researchers repeat it.",
    )
    guess_review
    return (guess_review,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **paper**, or research paper, is a written report of a piece of research: the question, what the researchers did, what came out, and how it relates to earlier work. Researchers share their work as papers, so that others can read it, check it and build on it.

        A **preprint** is a paper that the authors put online themselves, before or without peer review. Many AI papers appear first on **arXiv**, said "archive", a free website for preprints. That is fast and open to everyone, but nobody has checked whether the paper is right: arXiv only screens that it looks like research. A preprint can get new versions, v1, v2 and so on.

        **Peer review** is the check by other researchers in the field, the peers. A conference or a journal sends a submitted paper to a few experts, who usually stay anonymous and advise whether to publish it and what must be fixed. Only then does a conference put the paper in its proceedings, the collection of the papers it accepted, or a journal in an issue. Peer review catches many mistakes, but it is not a guarantee: reviewers read, they rarely run the experiments again.

        So a claim gets stronger step by step: a preprint, then a paper that passed peer review, then a result that other researchers repeat. A press release or a news story is someone's summary of a paper, often shorter and surer than the paper itself.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **paper** | a written report of a piece of research: question, method, results, and how it relates to earlier work |
        | **preprint** | a paper the authors put online themselves, before or without peer review: nobody has checked it yet |
        | **arXiv** (say "archive") | the free website where most AI preprints appear first |
        | **peer review** | the check by a few other researchers before a conference or journal publishes a paper; it catches many mistakes, but is no guarantee |
        """),
        L.watch_out("\"Published\" means two things. \"Published on arXiv\" is a preprint, put online by its own authors. \"Published at a conference\" or \"in a journal\" means it passed peer review. News stories often do not tell the two apart."),
        mo.md(r"""
        The picture: the usual path of a paper in AI.
        """),
        draw.flow(
            [("Preprint", ["the authors put it", "on arXiv"]),
             ("Peer review", ["a few researchers", "read and judge it"]),
             ("Published", ["in the proceedings", "of a conference,", "or in a journal"])],
            forward=["submitted", "accepted"],
            back=["changes asked,\nor rejected"],
            title="From preprint to published paper"),
        mo.md(r"""
        Not every paper goes all the way: some stay a preprint, and a rejected paper often goes to another conference, after changes.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find out how much to trust a study you read about:

    1. Find the paper itself, not only the story about it: a good news story links to it; otherwise search for its title.
    2. See where it appeared: only on arXiv or a website, so a preprint; or at a conference or in a journal, after peer review.
    3. Look at the date and the version: a preprint can still change.
    4. Ask whether others repeated the result: a result that held up in other hands is the strongest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    **"Attention Is All You Need"**, by Vaswani and seven co-authors, the paper that introduced the transformer: the kind of neural network behind today's chat assistants, which you build yourself later in the course.

    1. **The paper itself.** It is on arXiv, with the number 1706.03762: 17 for 2017 and 06 for June.
    2. **Where it appeared.** First as a preprint, in June 2017. Then it passed peer review at the NeurIPS conference, then called NIPS, and appeared in its proceedings in December 2017.
    3. **Date and version.** arXiv shows seven versions: four in June 2017, one in December 2017 and two in 2023. The authors changed details along the way; read the latest.
    4. **Repeated by others?** Yes, many thousands of times: the transformer is the base of today's chat assistants, and countless later papers built on it and confirmed that it works.

    This claim went all the way, from preprint through peer review to a result others repeated. Many claims in the news have made only the first step.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Which kind of text?

        Six texts about research. Put each in its group.
        """),
    ])
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "For each text, ask: is it the report of the research itself, and if so, did other researchers check it before it appeared?",
            "A text that only tells about the research, without its method and all its results, is not the paper. And a paper passed peer review only if a conference or a journal published it.",
            "The blog post and the press release are not the paper. arXiv and the authors' own website have no peer review. So two texts are preprints, two passed peer review, and two are not papers.",
        ],
        why="**Preprint:** the paper on arXiv, version 1, and the paper on the authors' own website. **Paper that passed peer review:** the paper in the NeurIPS proceedings and the paper JMLR accepted. **Not a paper:** the blog post and the press release.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_arxiv():
    mo.md(r"""
    ### Exercise 2. Published on arXiv
    """)
    return


@app.cell(hide_code=True)
def choice_arxiv_cell():
    choice_arxiv = L.choice(
        "on-arxiv",
        question="A news site writes about a new model: \"the study was published today on arXiv\". What do you know about that study?",
        options={
            "reviewed": "Reviewers checked that its results are right",
            "rejected": "A journal turned it down, so the authors put it on arXiv",
            "right": "Its authors put it online; arXiv did not review it",
        },
        correct="right",
        why={
            "reviewed": "arXiv only screens that a paper looks like research; it has no peer review. See *The idea*.",
            "rejected": "Authors often put a paper on arXiv before or during review; it does not mean anyone turned it down. See *The idea*.",
            "right": "It is a preprint: fast and open, but nobody reviewed it. \"Published\" on arXiv does not mean peer reviewed.",
        },
    )
    choice_arxiv
    return (choice_arxiv,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 3. Before you believe it
    """)
    return


@app.cell(hide_code=True)
def explain_claim():
    explain_check = L.explain(
        "check-claim",
        question="Someone shares a news story with you: \"Scientists show: new AI reads scans better than doctors.\" Which questions would you ask before you believe it? Write them down.",
        model_answer="**Is there a paper, and can I find it?** **Is it a preprint, or did it pass peer review, and where?** **Has anyone else repeated the result?** And: does the paper itself say what the news story says?",
        points=[
            "you ask for the paper itself, not only the news story",
            "you ask whether it is a preprint or passed peer review",
            "you ask whether others repeated the result",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) showed how to place a system; now you can place a claim about one: a preprint that nobody checked yet, a paper that passed peer review, or a result that others repeated. In the next lesson, *Reading AI claims critically*, you compare what a press release, a paper and a benchmark score each say, and ask what was measured.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_arxiv, explain_check, result_kinds):
    L.wrap_up(result_kinds, choice_arxiv, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
