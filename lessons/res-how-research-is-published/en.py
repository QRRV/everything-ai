# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-how-research-is-published. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-how-research-is-published", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en) and [Reading AI claims critically](/course/open/res-reading-claims?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_experiment():
    quick_experiment = L.choice(
        "quick-experiment",
        question="A team trains the same model five times with a new learning rate and five times with the old one, keeps everything else the same, and compares the validation losses. Which step of the research cycle is this?",
        options={
            "question": "The question",
            "right": "The experiment",
            "hypothesis": "The hypothesis",
        },
        correct="right",
        why={
            "question": "The question comes first: what do you want to know? This team is already testing. See [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en).",
            "hypothesis": "The hypothesis is the precise expectation they test; training and comparing the losses is the test itself. See [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en).",
            "right": "An experiment changes one thing on purpose, keeps the rest the same and measures what happens.",
        },
    )
    quick_experiment
    return (quick_experiment,)


@app.cell(hide_code=True)
def quick_check_benchmark():
    quick_benchmark = L.choice(
        "quick-benchmark",
        question="A model gets 88% on a benchmark of 500 maths questions. What does that show?",
        options={
            "right": "How it does on those 500 questions, scored that way",
            "all": "That it solves 88% of all maths questions of that kind",
            "beats": "That it beats a model with 80% elsewhere",
        },
        correct="right",
        why={
            "all": "It was tested on 500 questions, not on all of them. See [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
            "beats": "Scores on different tests cannot be compared. See [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
            "right": "A benchmark score says how a model does on those tasks, scored that way, not in general.",
        },
    )
    quick_benchmark
    return (quick_benchmark,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    When you look for a paper, for a project of your own or for the reproduction at the end of this course, you find hundreds: in 2026, thousands of new papers about AI appear on arXiv every month. Before you spend an evening on one, two questions tell you what you are holding. What does this paper add? And where did it appear? The first says what evidence to look for, the second how much checking the paper has had.

    **An everyday example.** A newspaper has news reports, opinion pieces, reviews and corrections. You read an opinion piece differently from a news report: one argues for a view, the other reports what happened. Papers come in kinds too.
    """)
    return


@app.cell(hide_code=True)
def predict_venue():
    guess_venue = L.guess(
        "where-papers-count",
        question="In biology or physics, the papers that count appear in journals, after months of review. Where do most of the papers that count in machine learning go through peer review?",
        options={
            "journal": "In journals, just as in other sciences",
            "right": "At big conferences, such as NeurIPS",
            "arxiv": "Nowhere: they only go on arXiv",
        },
        reveal="**At conferences.** In machine learning the big conferences count as much as journals do in other sciences: a few reviewers judge each paper within weeks, and the accepted ones appear in the proceedings. The transformer paper from [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en) went that way: on arXiv in June 2017, at NeurIPS, then called NIPS, in December. Journals such as JMLR exist too, but far fewer papers go there. And not every well-known paper goes through review: the 2019 report on GPT-2, a well-known language model, appeared only on the website of the lab that made it.",
    )
    guess_venue
    return (guess_venue,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Every paper says what it adds to the field: its **contribution**. Most papers in AI make one of six kinds of contribution. A **method** paper brings a new way to do something; an **empirical analysis** measures how something that already exists behaves; a **theory** paper proves something with mathematics; a **benchmark or dataset** paper gives others new tasks or data to use; a **position paper** argues for a view; a **reproduction** does someone else's experiments again. Some papers make two, such as a new benchmark and an analysis of models on it. A **survey** is a kind of paper of its own: it sums up and compares the papers of a whole field, and the next lesson starts from one.

        The kind tells you what to check. A new method means little without a comparison on the same data, as in [Always beat a simple baseline](/course/open/ml-baselines?lang=en). A benchmark is only worth something if it measures what it says, as you saw in [Reading AI claims critically](/course/open/res-reading-claims?lang=en). A proof holds only as far as its assumptions do.
        """),
        mo.Html(C.kinds({"title": "Six kinds of contribution", "adds": "Adds:", "check": "Check:",
                         "kinds": [("Method", "a new way to do something: a new kind of model, a training trick, an algorithm", "does it beat strong baselines on the same data, on more than one task?", "Example: the transformer, behind today's chatbots (2017)"),
                                   ("Empirical analysis", "measurements of how something behaves, without a new method", "how many settings and seeds, and does the pattern hold beyond them?", "Example: Scaling Laws for Neural Language Models, how the loss falls as models grow (2020)"),
                                   ("Theory", "results proven with mathematics", "which assumptions, and do they hold for real models?", "Example: A Theory of the Learnable, how many examples are enough to learn a rule (1984)"),
                                   ("Benchmark or dataset", "new test tasks or data for others to use", "does it measure what it says, are the answers right, were the questions already online?", "Example: ImageNet, millions of labeled photos (2009)"),
                                   ("Position paper", "an argument for where the field should go", "which evidence supports it, and what would the other side say?", "Example: Troubling Trends in Machine Learning Scholarship (2018)"),
                                   ("Reproduction", "someone else's experiments done again", "what was the same, what differed, and what could not be checked?", "Example: the reports of the ML Reproducibility Challenge")],
                         "alt": "Six cards. Method: adds a new way to do something; check whether it beats strong baselines on the same data. Empirical analysis: adds measurements of how something behaves; check how many settings and seeds. Theory: adds a statement proven with mathematics; check the assumptions. Benchmark or dataset: adds new tasks or data; check whether it measures what it says. Position paper: adds an argument; check the evidence and the other side. Reproduction: does someone else's experiments again; check what was the same and what differed."})),
        mo.md(r"""
        A paper also appears somewhere: its **venue**. The venue tells you how much checking the paper had before you read it.

        | Venue | Who checked it | Examples (2026) |
        |---|---|---|
        | arXiv | nobody yet: the authors put it online | most AI papers, often before any review |
        | workshop | a few reviewers, often lightly | small meetings next to a big conference |
        | conference | a few reviewers, within a few weeks | NeurIPS, ICML, ICLR; ACL for language, CVPR for images |
        | journal | reviewers, often in several rounds | JMLR, TMLR |

        In most sciences the journal is the place that counts. In machine learning the big conferences count just as much, and most reviewed papers appear there, in the proceedings. A workshop is a smaller meeting next to a conference, often for early work. Names of venues come and go, so look at the year next to them.

        You find the venue on the first page (`Published as a conference paper at ICLR 2025`), in the Comments field on arXiv (`Accepted at ICML 2025`), or in a reference: *Advances in Neural Information Processing Systems 30* is volume 30 of the NeurIPS proceedings, from 2017. No venue in any of these places? Then treat it as a preprint until you find one: authors do not always add the venue to arXiv.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **contribution type** | what a paper adds to the field: a method, an empirical analysis, a theory, a benchmark or dataset, a position, or a reproduction |
        | **venue** | where a paper appears: a conference, a workshop, a journal, or arXiv for preprints |
        | **workshop** | a smaller meeting next to a conference, often for early work, with a lighter review |
        | **journal** | a scientific publication that publishes papers after peer review, often in several rounds |
        | **theory** | here: results proven with mathematics, not a guess as in everyday speech |
        | **position paper** | a paper that argues for a view on where the field should go, instead of presenting a new method or experiment |
        | **reproduction** | doing a paper's experiments again, with the authors' code or your own, to see whether you get the same results |
        """),
        L.watch_out("\"Accepted at NeurIPS\" means that a few reviewers found the paper good enough, not that its result holds: peer review is no guarantee, as in [What a paper, a preprint and peer review are](/course/open/res-what-is-a-paper?lang=en). And a paper that is only on arXiv is not wrong for that reason. The venue says how much checking was done; the evidence for the claim you still check yourself."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To place a paper before you read it closely:

    1. Read the title and the abstract, the short summary at the top of the paper, and find the sentences that say what the paper adds: often "we propose", "we show", "we prove", "we introduce", "we argue" or "we reproduce". Those words are a hint, not proof: read what follows them.
    2. Name the kind of contribution. If there are two, note both: the main claim usually rests on both.
    3. Find the venue: on the first page, in the Comments field on arXiv, or in a reference. No venue: treat it as a preprint until you find one.
    4. Decide what to check first: the evidence its kind needs, and how much checking its venue already did.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up paper, built like the real study in the guess of [Reading AI claims critically](/course/open/res-reading-claims?lang=en). On arXiv: **Fresh sums: 1,200 new maths word problems to test language models.** From the abstract: *We wrote 1,200 maths word problems by hand, in the style of a well-known benchmark, and kept them off the internet. We tested 30 language models on the old benchmark and on our problems. Several scored up to 10 percentage points lower on ours, which suggests they had seen the old questions in training.* It is version 1, put online in March 2026, and the Comments field is empty.

    1. **What it adds:** "We wrote 1,200 maths word problems" and "We tested 30 language models". Two sentences, two contributions.
    2. **Kind:** a benchmark, with an empirical analysis on it. The main claim, that some models had seen the old questions, rests on both.
    3. **Venue:** only on arXiv, version 1, no venue in the Comments field: a preprint. Nobody has checked it yet.
    4. **What to check first:** for the benchmark, whether the new problems are really as hard as the old ones and the answers right; if the new ones are harder, the drop could come from that instead. For the analysis, which models dropped and by how much, and whether anything other than having seen the questions could explain it. And because it is a preprint, you are among the first to check.

    Four steps, a few minutes, and you know what this paper must show before you use its claim.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which kind?

    Six made-up papers, each in one sentence. Put each with the kind of contribution it makes.
    """)
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
            "For each paper, ask: what can others take from it? Something new to use, a measurement, a proof, new test data, an argument, or a check of earlier work?",
            "Look at the cards in *The idea*. Only one paper works with someone else's experiments, and only one argues for something. Of the other four: which one brings a new way and compares it with the old way, and which one only measures how something that already exists behaves?",
        ],
        why="**Method:** a new way to fill in missing values. **Empirical analysis:** accuracy for 1,000 to 100,000 examples, five seeds for each amount. **Theory:** proves that a training method ends at the lowest loss. **Benchmark or dataset:** 2,000 questions about Dutch law. **Position paper:** argues that papers should publish their code. **Reproduction:** the experiments of a 2023 paper, run again.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_venue():
    mo.md(r"""
    ### Exercise 2. Where did it appear?
    """)
    return


@app.cell(hide_code=True)
def choice_venue_cell():
    choice_venue = L.choice(
        "venue",
        question="A paper you want to build on is on arXiv. Its Comments field says: \"Accepted at a workshop of ICML 2026.\" What does that tell you about the checking it had?",
        options={
            "main": "The same peer review as the papers of the main conference, ICML itself",
            "none": "None: a paper on arXiv is a preprint that nobody has checked",
            "right": "Some review, often lighter than at the main conference",
        },
        correct="right",
        why={
            "main": "A workshop is a smaller meeting next to the conference, with a review of its own, often a lighter one. See the venues in *The idea*.",
            "none": "Many reviewed papers are on arXiv too; the Comments field tells you where it was accepted. See the venues in *The idea*.",
            "right": "A workshop is often for early work: the review is real, but usually lighter than at the main conference. All the more reason to check the evidence yourself.",
        },
    )
    choice_venue
    return (choice_venue,)


@app.cell(hide_code=True)
def text_exercise_place():
    mo.md(r"""
    ### Exercise 3. Place a paper yourself

    Another made-up paper. **Half the labels: training a classifier on fewer labeled examples.** From the abstract: *We propose a training method that also learns from examples without a label. On two datasets of photos, a model trained with our method on half of the labeled examples reaches the same accuracy on the test set as a standard model trained on all of them.* On the first page: `Published as a conference paper at ICLR 2026.`

    Step 1 of the recipe is done: what it adds is "We propose a training method that also learns from examples without a label".
    """)
    return


@app.cell(hide_code=True)
def explain_place():
    explain_place_paper = L.explain(
        "place-paper",
        question="Do steps 2 to 4 of the recipe: name the kind of contribution, the venue and what it tells you, and what you would check first.",
        model_answer="**Kind:** a method, a new way to train a classifier. **Venue:** a conference, ICLR 2026, so a few reviewers read and judged it. **Check first:** whether the standard model got the same care, with its hyperparameters tuned just as well, and over how many seeds; and that only two datasets of photos were tested, so whether it holds for other kinds of data, the paper does not show.",
        points=[
            "you name the kind: a method",
            "you say that it passed peer review at a conference",
            "you name a check that a method needs: a fair comparison with the standard model, or how far the result reaches beyond two datasets of photos",
        ],
    )
    explain_place_paper
    return (explain_place_paper,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [How research works: question, hypothesis, experiment](/course/open/res-research-cycle?lang=en) showed the cycle that ends in a paper, and [Reading AI claims critically](/course/open/res-reading-claims?lang=en) how far a claim can stray from its measurement. Now you can tell, before you read a paper closely, what kind it is and how much checking it had. At the end of this course you make one of the six kinds yourself: a reproduction. In the next lesson, *Finding papers*, you search for papers and follow their references.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_venue, explain_place_paper, result_kinds):
    L.wrap_up(result_kinds, choice_venue, explain_place_paper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
