# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-whats-next. Route: course/route/lessen/onderzoek.yaml. Checks and the numbers about the domains: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-whats-next", "en")
    C = L.checks   # what every domain borrows, from the route data
    B = C.borrowers()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The map of AI and how this course is organized](/course/open/start-map-of-ai?lang=en) and [Your reproduction 8: report and share](/course/open/repro-own-report?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_locked():
    quick_locked = L.choice(
        "quick-locked",
        question="On the start page, a lesson at the expert level catches your eye while you are still at the basics. What happens when you open it?",
        options={
            "locked": "It stays closed until you have done every intermediate lesson",
            "right": "It opens, and *Before you start* lists the lessons it builds on",
            "count": "It opens, but its exercises count only after the levels before it",
        },
        correct="right",
        why={
            "right": "No lesson is locked: a level only says how advanced a lesson is. *Before you start* shows what it builds on, with a tick for each one you have done.",
            "locked": "A level is a label, not a stage of a game: no lesson is locked. See [The map of AI and how this course is organized](/course/open/start-map-of-ai?lang=en).",
            "count": "Every exercise counts, whenever you do it: a level only says how advanced a lesson is. See [The map of AI and how this course is organized](/course/open/start-map-of-ai?lang=en).",
        },
    )
    quick_locked
    return (quick_locked,)


@app.cell(hide_code=True)
def quick_check_not_checked():
    quick_not_checked = L.choice(
        "quick-not-checked",
        question="The *Not checked* part of the course's report names MNIST itself, larger networks and more seeds. What is it for?",
        options={
            "failed": "It lists the runs that went wrong, so that a reader can leave them out",
            "right": "It keeps a reader from taking the result for more than was tested",
            "modest": "It shows modesty: a report should not sound too sure of itself",
        },
        correct="right",
        why={
            "right": "A reader might think that MNIST was tested too, and it was not. So the part is also a list of open questions, and this lesson comes back to it.",
            "failed": "A run that went wrong goes in the Results like any other. *Not checked* names what was never run. See [Your reproduction 8](/course/open/repro-own-report?lang=en).",
            "modest": "It is not about tone: it says what was not tested, so that a reader can weigh the result. See [Your reproduction 8](/course/open/repro-own-report?lang=en).",
        },
    )
    quick_not_checked
    return (quick_not_checked,)


@app.cell(hide_code=True)
def text_why():
    mo.md(rf"""
    ## Why this matters

    You have done the core and the route of research: from your first line of Python to a paper you reproduced on your own. What now? There are two ways on, and you can take both. The map of the course has {len(C.OTHERS)} more domains, from computer vision to robotics, and each of them builds on what you know now. And your reproduction does not have to end with its report: much research starts as a reproduction, and grows into a contribution of its own when someone asks the next question about the paper.

    **An everyday example.** After a year of lessons you can play the guitar well enough to play songs you have never heard. Now you choose: jazz, classical, or songs of your own. Each of them starts from chords you already know, and a good teacher can tell you which ones.
    """)
    return


@app.cell(hide_code=True)
def predict_shared():
    guess_shared = L.guess(
        "most-borrowed",
        question="Some lessons are shared: written once, and borrowed by every domain that needs them. Which of these do the most domains borrow?",
        options={
            "cnn": "Convolutional networks",
            "attention": "Attention and transformers",
            "math": "Math, formally",
        },
        reveal=f"**Math, formally.** {B['wiskunde']} of the {len(C.DOMAINS)} domains borrow it, attention {B['attention']}, and convolutional networks {B['cnn']}. Past its basics, almost every domain needs the math written out: derivatives, matrices and probability, which the core showed you by example. Convolutional networks are for data on a grid, such as pictures and sound, so fewer domains need them.",
    )
    guess_shared
    return (guess_shared,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Shared modules.** Every domain builds on the core. Past their basics, the domains need more of the same things: the math written out formally, training deep networks, convolutional networks, attention. Those lessons are written once, in shared modules of the foundation, and a domain borrows the ones it needs, at the level where it first needs them. Computer vision, for example, borrows convolutional networks at the intermediate level, and so does speech. A lesson you did for one domain counts for every domain that borrows it: you never take it twice. The route data of the course list what every domain borrows.

        **From a reproduction to your own question.** A reproduction asks whether a claim holds where the paper tested it. The next question asks where else it holds, or why. Change one thing about the paper, keep everything else as in your reproduction, and you have a new experiment, with the claim, plan, seeds and comparison you already have. Three kinds of change are common:

        - an **ablation** takes a part of the method away, to see how much of the gain that part carries, as in Your reproduction 6;
        - **new data** runs the same method on another dataset, to see whether the claim holds there too;
        - a **robustness test** keeps the model and its training, and tests it under a disturbance the paper did not try, such as noise on the pictures.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **shared module** | a chapter of lessons in the foundation of the course that several domains borrow, such as convolutional networks or attention; each domain borrows it at the level where it first needs it |
        | **robustness test** | a test of a trained model under a disturbance it was not trained for, such as noise or shifted pixels, to see whether its result holds up |
        """),
        L.watch_out("Change one thing at a time. Change two at once, such as the data and the size of the network, and a difference you find has two possible causes: you cannot tell which one it was."),
        mo.md(r"""
        Pick a domain below to see what it builds on.
        """),
    ])
    return


@app.cell(hide_code=True)
def pick_domain():
    domain_pick = mo.ui.dropdown(
        options={C.name(_d, L.lang): _d for _d in C.OTHERS},
        value=C.name("beeld", L.lang), label="Domain")
    domain_pick
    return (domain_pick,)


@app.cell(hide_code=True)
def picture_domain(domain_pick):
    _done = C.done_lessons()
    mo.vstack([
        mo.Html(C.domain_picture(domain_pick.value, _done, {
            "lang": L.lang, "title": "What {domain} builds on", "core": "The core",
            "levels": ["Basics", "Intermediate", "Expert"], "count": "{k} of {n} lessons done", "count_one": "{k} of 1 lesson done",
            "later": "lessons still to come", "nothing": "nothing more from the foundation",
            "alt": "What the chosen domain builds on, from the bottom up: the core, and above it, per level, the shared modules the domain borrows, each with how many of its lessons you have done."})),
        mo.md(C.domain_card(domain_pick.value, L.lang, _done, {
            "project": "Its project", "taste": "A first taste of it in the core", "done": "done", "not_done": "not done yet",
            "lessons": "Lessons", "one": "1 lesson ready", "many": "{n} lessons ready", "in": "in the route “{route}”",
            "later": "still to come"})),
    ])
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    Read the picture from the bottom up: the core, which every domain builds on, and above it, per level, the shared modules the domain borrows, each with how many of its lessons you have done. A dashed box is a module whose lessons are still to come. Each domain opens on the start page, under *The whole map*, as soon as its lessons are ready.

    ### Recipe

    To pick your next step:

    1. **Say what you want to be able to do next.** A part of AI, such as pictures or robots, or a question about your paper.
    2. **A part of AI?** Find its domain above: what it borrows, and how far you are with each part. The lessons of the core that gave you a first taste of it, if it has any, show what kind of work it is.
    3. **A question about your paper?** Start from the *Not checked* part of your report, and pick one change: an ablation, new data or a robustness test.
    4. **Write it down as a plan before you run it,** as in Your reproduction 2: what you change, what stays the same, and by which rule you will call the claim held.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course's reproduction of the dropout paper says under *Not checked*: MNIST itself, larger networks, more seeds. Here are three next questions, one of each kind; the first comes straight from that list:

    | Kind | Question | What changes | What stays the same |
    |---|---|---|---|
    | new data | Does the gain hold on MNIST, the paper's own digits? | the dataset | the networks, the dropout, the steps and the seeds |
    | ablation | Does the gain need the full rate of 0.5 in the hidden layers? | a rate of 0.25 in the hidden layers | the data, the input's dropout, the steps and the seeds |
    | robustness test | Does the gain hold when the test pictures have noise? | the test pictures | the networks and their training |

    *Larger networks* is a fourth question. Asked together with MNIST, it would change two things at once: ask it on Fashion-MNIST, as a question of its own.

    Which one first? A good first question changes one thing, fits in your budget, and has an answer that is interesting whichever way it comes out. All three do. The course would start with MNIST: its report found errors eight to nine times the paper's, and a gain larger in points and smaller as a share of the error. On MNIST, with errors closer to the paper's, you see whether that difference in size comes from the data.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What kind of question?

        Each question below changes something about the course's reproduction of the dropout paper: networks of 784-512-512-10 on Fashion-MNIST, with dropout at the input and in the hidden layers, three seeds.
        """),
        L.task("Put each question in its group."),
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
            "An ablation takes a part of the method away; new data changes what the networks learn from and are tested on; a robustness test changes only what the trained networks are tested on. The dropout at the input and the dropout in the hidden layers are the two parts of the method.",
            "Count the changes in each question first. Two changes put it in the last group, whatever they are.",
        ],
        why="Ablation: **Input only** and **Hidden layers only**. New data: **MNIST** and **Kuzushiji-MNIST**. Robustness test: **Noisy pictures** and **Shifted pictures**. Two changes at once: **Wider, on MNIST** and **No input dropout, noisy pictures**.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ready():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Which domains are you ready for?

        For each domain, `uses` lists the shared modules it borrows up to its intermediate level, and `done` lists the shared modules you have finished. Complete `ready(uses, done)`: it gives the domains whose shared modules are all in `done`, in the order of `uses`. A domain that borrows nothing is ready right away. In the lesson's example, `vision` and `build` are ready, and `speech` still needs `attention`:

        ```python
        uses = {"vision": ["math", "cnn"], "build": [], "speech": ["math", "cnn", "attention"]}
        done = ["cnn", "math"]
        ```

        The check tries this example, no domains, nothing done, domains that borrow nothing, and modules done in another order than `uses` lists them.
        """),
        L.task("Complete `ready`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ready():
    box_ready = L.code_box("ready", lines=8)
    box_ready
    return (box_ready,)


@app.cell(hide_code=True)
def check_ready(box_ready):
    result_ready = L.check("ready", box_ready.value)
    result_ready
    return (result_ready,)


@app.cell(hide_code=True)
def hints_ready():
    L.hints(
        "ready",
        [
            "`for domain, modules in uses.items():` gives each domain with its list of shared modules. A domain is ready when none of its modules is missing from `done`.",
            "For each domain, collect the modules that are not in `done`: `[m for m in modules if m not in done]`. When that list is empty, the domain is ready, also when it borrows nothing.",
        ],
        why="A domain is ready when every module it borrows is done; an empty list has nothing missing, so a domain that borrows nothing is ready. Going through `uses` from the start keeps its order.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your progress says

    Your `ready`, on the shared modules that the other {len(C.OTHERS)} domains borrow up to their intermediate level, as far as those modules have lessons yet, and on the modules whose lessons you have all done:
    """)
    return


@app.cell(hide_code=True)
def demo_ready(box_ready, result_ready):
    _ready = C.my_ready(box_ready.value) if result_ready.state == "correct" else None
    if result_ready.state != "correct":
        _out = mo.md("*Make `ready` right in exercise 2 first; then it runs here.*")
    elif _ready is None:
        _out = mo.md("*Your `ready` stops with an error on the route data of the course and your progress. Change it in exercise 2 and check it again.*")
    else:
        _done = C.done_lessons()
        _lines = [f"**For {len(_ready)} of the {len(C.OTHERS)} domains, you have done every shared module they borrow up to intermediate.**", ""]
        _lines += [f"- {C.name(_d, L.lang)}" for _d in _ready if _d in C.OTHERS]
        _wait = [_d for _d in C.OTHERS if _d not in _ready]
        if _wait:
            _lines += ["", "Not yet:", ""]
            for _d in _wait:
                _todo = C.missing(_d, _done)
                _parts = [f"{C.module_name(_m, L.lang)}, {_k} of {_n} {"lesson" if _n == 1 else "lessons"} done, next [{C.course.title(_next, L.lang)}](/course/open/{_next}?lang={L.lang})"
                          for _m, _k, _n, _next in _todo]
                _lines.append(f"- {C.name(_d, L.lang)}: " + "; ".join(_parts))
        _out = mo.md("\n".join(_lines))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Exercise 3. Your next question

    Your report is written, and its *Not checked* part names what you did not test. Turn one of those, or another question about your paper, into the plan of a next experiment.
    """)
    return


@app.cell(hide_code=True)
def explain_next():
    explain_next_cell = L.explain(
        "next-question",
        question="Write the next question for the paper you reproduced: which kind of change it is, what you change and what stays the same, and the rule by which you will call the claim held, written before you run it.",
        model_answer="Does the gain of dropout hold on MNIST, the paper's own digits? New data: I train the same 784-512-512-10 networks, with and without dropout, on 6000 pictures of MNIST instead of Fashion-MNIST, with the same dropout, the same 12,800 steps and the same three seeds. The rule, written before I run it: the claim holds on MNIST if the test error with dropout is lower for all three seeds. If it holds, I also compare the size of the gain with the paper's 0.25 points, to test my guess that the harder pictures explained the difference.",
        points=[
            "you name the kind of change: an ablation, new data or a robustness test",
            "you change one thing, and say what stays the same",
            "you write the rule for the claim before you run it",
            "the question grows out of your reproduction, for example its *Not checked*",
        ],
    )
    explain_next_cell
    return (explain_next_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.vstack([
        mo.md(rf"""
        ## Wrap-up

        **How this connects.** The map you met in [The map of AI and how this course is organized](/course/open/start-map-of-ai?lang=en), at the start of the course, now with the shared modules that each domain borrows; and the *Not checked* part of [Your reproduction 8: report and share](/course/open/repro-own-report?lang=en), turned into your next question. This is the last lesson of the route of research, the last step to the last of these milestones:
        """),
        mo.md(C.milestone_lines(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_kinds, result_ready, explain_next_cell):
    L.wrap_up(result_kinds, result_ready, explain_next_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
