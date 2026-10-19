# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-find-papers. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-find-papers", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_venue():
    quick_venue = L.choice(
        "quick-venue",
        question="You find a paper twice: as version 1 on arXiv, from February 2025, and later in the proceedings of ICML 2025. What happened in between?",
        options={
            "same": "Nothing that matters: it is the same paper in another place",
            "right": "A few reviewers judged it, and the conference accepted it",
            "repeated": "Other researchers repeated its result",
        },
        correct="right",
        why={
            "same": "The proceedings of a conference hold only the papers it accepted after peer review, so the second place tells you how much checking it had. See [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).",
            "repeated": "Repeating a result is a reproduction; being in the proceedings of a conference means peer review. See [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).",
            "right": "ICML is a conference with peer review: a few reviewers read and judged the paper. That is still no guarantee that its result holds.",
        },
    )
    quick_venue
    return (quick_venue,)


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="A paper tests 12 existing image classifiers on photos taken at night, reports how much worse each does than in daylight, and proposes nothing new. What kind of contribution is that?",
        options={
            "method": "A method",
            "position": "A position paper",
            "right": "An empirical analysis",
        },
        correct="right",
        why={
            "method": "A method paper brings a new way to do something, and this paper proposes nothing new. See [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).",
            "position": "A position paper argues for a view; this paper measures. See [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).",
            "right": "It measures how something that already exists behaves. So check how many settings and runs, and whether the pattern holds beyond them.",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You have a question, such as: does filling in missing values with the mean make a model worse? A web search gives you blog posts, adverts and answers from chatbots, some of which name papers that do not exist. You want the papers themselves: who measured this, how, and what came out. With thousands of new papers every month, you need a way to find the few that matter.

    **An everyday example.** A good book has a list of sources at the back: from there you find older books on the same subject. Newer books that mention it are not in it: it was printed before they were written.
    """)
    return


@app.cell(hide_code=True)
def predict_citations():
    guess_citations = L.guess(
        "transformer-citations",
        question="At its end, the transformer paper of 2017 lists a few dozen references: the older work it builds on. How many later papers do you think cite it, by 2026?",
        options={
            "dozens": "A few dozen as well",
            "thousands": "A few thousand",
            "right": "More than 100,000",
        },
        reveal="**More than 100,000**, by the count of Google Scholar in 2026. Its own references you read in a minute. The papers that cite it are more than anyone can read, so there you filter: by year, by words in the title, or by how often each of those papers is cited in turn.",
    )
    guess_citations
    return (guess_citations,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Literature search** is finding the papers that matter for a question. There are three ways in, and a good search uses all three.

        **Start from someone who already searched.** A **survey** is a paper that orders and compares the papers of a whole field; a reading list is a list of papers that a teacher or a researcher recommends for a topic. Search for your topic with the word "survey" added.

        **Search with search terms.** A search engine for papers finds them by their words, above all those in the title and the abstract, the short summary at the top. In 2026 the best known are Semantic Scholar and Google Scholar, for almost every field, with next to each result how often it is cited, and arXiv, for the preprints on it. Use more than one term: researchers name the same thing in different ways, such as *missing values* and *missing data*, and *imputation* for filling them in.

        **Follow citations.** At its end, every paper lists the work it cites: its **references**. Following them goes **backward**, to older work. Following the papers that cite it goes **forward**, to newer work. A paper cannot list who will cite it later, so forward you need a search engine that keeps track: Semantic Scholar and Google Scholar show these papers under *Citations* or *Cited by*, and on the page of every paper on arXiv, the box *References & Citations* links to sites that show them. Following citations both ways, round after round, is called snowballing.

        All papers together, with an arrow from each paper to every paper it cites, form a **citation graph**. The picture shows a few papers around the transformer.
        """),
        mo.Html(C.around("transformer", {"title": "Around the transformer paper", "forward": "Forward: papers that cite it (6 of more than 100,000)",
                                         "back": "Backward: its references (7 of a few dozen)", "cite": "↓ cites",
                                         "alt": "The transformer paper of 2017 in the middle. Above it, six later papers that cite it: gpt 2018, bert 2018, gpt2 2019, scaling 2020, gpt3 2020 and vit 2020. Below it, seven of its references: align 2014, seq2seq 2014, lstm 1997, adam 2014, dropout 2014, resnet 2015 and layernorm 2016. Arrows point down, from a paper to the papers it cites."})),
        mo.md(r"""
        Some fields write a search down so exactly that anyone can repeat it: the search terms, the search engines, the dates, and why each paper was kept or dropped. That is a systematic review, and PRISMA is a common checklist for reporting one. For your own work a lighter version is enough: write down your terms and where you searched.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **literature search** | finding the papers that matter for a question: from surveys and reading lists, with search terms, and by following citations |
        | **survey** | a paper that orders and compares the papers of a whole field; a good place to start in a new topic |
        | **reference** | a paper listed at the end of another paper, as work it builds on or discusses |
        | **citation** | one paper naming another; how often a paper is cited is a rough measure of its influence |
        | **citation graph** | all papers, with an arrow from each paper to every paper it cites |
        | **backward, forward** | backward to the references of a paper, older work; forward to the papers that cite it, newer work |
        """),
        L.watch_out("Chatbots can invent references: a title that sounds right, the names of real researchers, a year, and no such paper. Look up every paper you did not find yourself, on arXiv, Semantic Scholar or Google Scholar, before you use it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the papers for a question:

    1. Write the question in one sentence, and choose two to four search terms, with other words for the same thing.
    2. Look for a survey or a reading list first: search for your terms with the word "survey".
    3. Search Semantic Scholar, Google Scholar or arXiv with your terms. Keep the two or three papers whose title and abstract fit best, and place each with the recipe of [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en).
    4. From each paper you kept, go backward through its references and forward through the papers that cite it, and keep what fits your question.
    5. Do step 4 again with the papers you kept, until a round brings mostly papers you already have. Write down your terms and where you searched.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 in code, on a small part of the real citation graph. For each paper, `_cites` holds some of the papers of this set that it really cites; the table under the cell says which papers the names stand for. Not every link is here: we left some out, and each paper also cites papers outside this set. One thing in the Python is new: `in` also asks whether a value is in a list, as it asks whether a key is in a dictionary. `"gpt" in ["transformer", "gpt"]` gives `True`. The comments say which half of step 4 each part is. Run the cell.
    """)
    return


@app.cell
def example_round():
    # a small part of the real citation graph: for each paper, some of the papers of this set that it cites
    _cites = {
        "lstm": [],
        "dropout": [],
        "adam": [],
        "seq2seq": ["lstm"],
        "align": ["seq2seq"],
        "resnet": [],
        "layernorm": [],
        "transformer": ["align", "seq2seq", "lstm", "adam", "dropout", "resnet", "layernorm"],
        "gpt": ["transformer", "adam", "layernorm"],
        "bert": ["transformer", "gpt"],
        "gpt2": ["transformer", "gpt", "bert"],
        "scaling": ["transformer", "gpt2"],
        "gpt3": ["transformer", "gpt2", "bert", "scaling"],
        "vit": ["transformer", "bert", "resnet"],
    }
    _paper = "gpt2"
    # 4. backward: the references of the paper
    _back = _cites[_paper]
    # 4. forward: every paper that has it among its references
    _forward = []
    for _other in _cites:
        if _paper in _cites[_other]:
            _forward.append(_other)
    print("backward:", _back)
    print("forward:", _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Backward from `gpt2` (2019): the transformer, GPT and BERT, all older. Forward: the scaling laws and GPT-3, both from 2020, newer. Backward is one look-up, because the list is in the paper itself. Forward needs a loop over every paper in the graph: on the real graph, with millions of papers, that is the work Semantic Scholar and Google Scholar do for you.

    | Name | Paper, with the year it first appeared |
    |---|---|
    | `lstm` | Hochreiter and Schmidhuber (1997), *Long Short-Term Memory* |
    | `dropout` | Srivastava et al. (2014), *Dropout* |
    | `adam` | Kingma and Ba (2014), *Adam* |
    | `seq2seq` | Sutskever et al. (2014), *Sequence to Sequence Learning with Neural Networks* |
    | `align` | Bahdanau et al. (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* | <!-- termen:ok -->
    | `resnet` | He et al. (2015), *Deep Residual Learning for Image Recognition* | <!-- termen:ok -->
    | `layernorm` | Ba et al. (2016), *Layer Normalization* |
    | `transformer` | Vaswani et al. (2017), *Attention Is All You Need* |
    | `gpt` | Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* | <!-- termen:ok -->
    | `bert` | Devlin et al. (2018), *BERT* |
    | `gpt2` | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* |
    | `scaling` | Kaplan et al. (2020), *Scaling Laws for Neural Language Models* |
    | `gpt3` | Brown et al. (2020), *Language Models are Few-Shot Learners* |
    | `vit` | Dosovitskiy et al. (2020), *An Image is Worth 16x16 Words* |
    """)
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One round, both ways

        Make `one_round(cites, paper)`: the papers one step from `paper` in the graph `cites`, backward and forward, together in one list in alphabetical order. The box gives you the start and the end; replace the two `...`. At the end, `back + forward` puts the two lists one after the other: `["a"] + ["b"]` gives `["a", "b"]`. The check calls `one_round(cites, "bert")` with the graph of the worked example, and also tries other papers and another graph.
        """),
        L.task("Make `one_round(cites, paper)` give back, in alphabetical order, the references of `paper` and the papers that cite it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("one-round", lines=8)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("one-round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "one-round",
        [
            "Two halves, as in the worked example, with `cites` and `paper` instead of `_cites` and `_paper`: backward is one look-up in `cites`; forward goes through every paper in `cites`, under a name of its own such as `other`, and asks whether `paper` is in its list.",
            "`back = cites[___]`, then `for other in cites:`, with under it `if paper in ___:` and under that `forward.append(___)`.",
        ],
        why="```python\ndef one_round(cites, paper):\n    back = cites[paper]\n    forward = []\n    for other in cites:\n        if paper in cites[other]:\n            forward.append(other)\n    return sorted(back + forward)\n```\n\n`cites[paper]` is the list of references, backward. The loop keeps every paper whose own list holds `paper`: the papers that cite it, forward. `sorted` puts the two lists together in alphabetical order.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ### Exercise 2. Which way?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="You found a good survey from 2021 about filling in missing values. Now you want the papers on that topic from after 2021. What do you do?",
        options={
            "back": "Go through the list of references at the end of the survey, one by one",
            "right": "Look up who cites the survey, on Semantic Scholar or Google Scholar",
            "title": "Search arXiv for the title of the survey",
        },
        correct="right",
        why={
            "back": "The references are older than the survey: backward leads to work from before 2021. See *The idea*.",
            "title": "That finds the survey itself again, not the work that came after it. See *The idea*.",
            "right": "Papers that cite the survey came after it: forward. Filter them by year, or sort them by how often they are cited.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Exercise 3. Plan a search

    Your question: **does filling in missing values with the mean of their column make a model worse than leaving those rows out?**
    """)
    return


@app.cell(hide_code=True)
def explain_plan():
    explain_plan_search = L.explain(
        "plan-search",
        question="Write a search plan with steps 1 to 4 of the recipe: your search terms, with other words for the same thing; where you start; where you search; and how you follow citations.",
        model_answer="**Terms:** missing values, missing data, imputation, mean imputation; for leaving rows out, listwise deletion or complete-case analysis; and the kind of data, such as tabular data. **Start:** search Semantic Scholar or Google Scholar for *missing data imputation survey*, and use the survey's overview. **Search:** the same terms without *survey*; keep the two or three papers whose title and abstract fit best. **Follow citations:** backward through their references for the older, well-known methods, and forward through the papers that cite them for newer comparisons; stop when a round brings mostly papers you already have.",
        points=[
            "you give more than one search term, including another word for the same thing, such as missing data or imputation",
            "you start from a survey or a reading list",
            "you follow citations both ways: backward through the references, forward through the papers that cite them",
        ],
    )
    explain_plan_search
    return (explain_plan_search,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Kinds of papers and where they appear](/course/open/res-how-research-is-published?lang=en) taught you to place a paper; now you can find them. Following citations comes back when you choose a paper to reproduce at the end of the course: the papers that cite it show whether others have already built on it, or tried to repeat it. In the next lesson, *The anatomy of a machine learning paper*, you look inside a paper: what each part answers, and where the details are.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, explain_plan_search, result_round):
    L.wrap_up(result_round, choice_direction, explain_plan_search)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
