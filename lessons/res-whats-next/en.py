# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-whats-next. Route: course/route/lessen/onderzoek.yaml. Checks and the numbers about the branches: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-whats-next", "en")
    C = L.checks   # the attach points of every branch, from the route data
    FIRST = C.first_levels()


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

    You have done the main route: from your first line of Python to an LLM of your own, pretrained and post-trained, and a paper you reproduced on your own. What now? There are two ways on, and you can take both. The map of the course has {len(C.LATER)} more branches, from computer vision to robotics, and each of them builds on what you know now. And your reproduction does not have to end with its report: much research starts as a reproduction, and grows into a contribution of its own when someone asks the next question about the paper.

    **An everyday example.** After a year of lessons you can play the guitar well enough to play songs you have never heard. Now you choose: jazz, classical, or songs of your own. Each of them starts from chords you already know, and a good teacher can tell you which ones.
    """)
    return


@app.cell(hide_code=True)
def predict_level():
    guess_level = L.guess(
        "level",
        question=f"Of the {len(C.LATER)} branches still to come, after which level of the main route can most of them begin?",
        options={
            "basis": "After the basics",
            "gevorderd": "After intermediate",
            "expert": "Only after expert, at the end of the main route",
        },
        reveal=f"**After the basics.** {FIRST['basis']} of the {len(C.LATER)} branches can begin right after a lesson at the basics level, and {FIRST['gevorderd']} after a lesson at intermediate: {', '.join(C.later_names('en'))}, which builds on your project with a deep network. Every branch builds on the main route, but most of them need only its start to begin. Once you have done all three levels, you can begin any of them.",
    )
    guess_level
    return (guess_level,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Attach points.** Every branch still to come grows out of the main route, like a branch out of a trunk. Where it connects is an attach point: a lesson of the main route that the branch builds on. A branch can have more than one: it can begin after the first, and its later lessons build on the others too. Computer vision, for example, begins after *Convolution: one small filter slides over the image*, at the basics level, and its lessons on deeper CNNs build on *Project: your first CNN*, at intermediate. The route data of the course list the attach points of every branch.

        **From a reproduction to your own question.** A reproduction asks whether a claim holds where the paper tested it. The next question asks where else it holds, or why. Change one thing about the paper, keep everything else as in your reproduction, and you have a new experiment, with the claim, plan, seeds and comparison you already have. Three kinds of change are common:

        - an **ablation** takes a part of the method away, to see how much of the gain that part carries, as in Your reproduction 6;
        - **new data** runs the same method on another dataset, to see whether the claim holds there too;
        - a **robustness test** keeps the model and its training, and tests it under a disturbance the paper did not try, such as noise on the pictures.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **attach point** | a lesson of the main route that a branch of the course connects to; the branch can begin after its first attach point, and from there it builds on what you know |
        | **robustness test** | a test of a trained model under a disturbance it was not trained for, such as noise or shifted pixels, to see whether its result holds up |
        """),
        L.watch_out("Change one thing at a time. Change two at once, such as the data and the size of the network, and a difference you find has two possible causes: you cannot tell which one it was."),
        mo.md(r"""
        Pick a branch below to see where it grows out of the main route.
        """),
    ])
    return


@app.cell(hide_code=True)
def pick_branch():
    branch_pick = mo.ui.dropdown(
        options={C.short_title(_b["id"], L.lang): _b["id"] for _b in C.LATER},
        value=C.short_title(C.LATER[0]["id"], L.lang), label="Branch")
    branch_pick
    return (branch_pick,)


@app.cell(hide_code=True)
def picture_tree(branch_pick):
    _done = C.done_lessons()
    mo.vstack([
        C.tree_picture(branch_pick.value, _done, {
            "lang": L.lang, "title": "Where the branch grows out of the main route",
            "levels": ["Basics", "Intermediate", "Expert"], "you": "you are here",
            "alt": "The main route as a horizontal trunk in three levels. The chosen branch leaves it at its first attach point, and dotted lines join its other attach points to it. Short grey stubs show where the other branches begin."}),
        mo.md(C.branch_card(branch_pick.value, L.lang, _done, {
            "project": "Its project", "attach": "Its attach points", "done": "done", "not_done": "not done yet",
            "extras": "Extra lessons of the main route from this branch, a start you can make now:"})),
    ])
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    Read the trunk from left to right: the main route, from the basics to expert. The numbered dots are the attach points of the branch you picked; it grows out at dot 1. The grey stubs show where the other branches begin, and *you are here* marks the furthest lesson of the main route you have done. Each branch opens on the start page, under *The whole map*, as soon as it is ready.

    ### Recipe

    To pick your next step:

    1. **Say what you want to be able to do next.** A part of AI, such as pictures or robots, or a question about your paper.
    2. **A part of AI?** Find its branch and its attach points above, and look back at those lessons: they are what the branch builds on. Until the branch opens, its extra lessons of the main route, if it has any, are a start you can make now.
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
def text_exercise_can_begin():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Which branches can you begin?

        The route data give each branch its attach points, in route order, and your progress says which lessons you have done. Complete `can_begin(attach, done)`: it gives the branches whose first attach point is in `done`, in the order of `attach`. In the lesson's example, `vision` and `robots` can begin, and `depth` cannot yet:

        ```python
        attach = {"vision": ["cnn-convolution-idea", "cnn-pytorch"],
                  "depth": ["nn-deep-net-project", "llm-modern-gpt"],
                  "robots": ["math-matrix-multiply", "rl-what-is-rl"]}
        done = ["cnn-convolution-idea", "llm-modern-gpt", "math-matrix-multiply", "rl-what-is-rl"]
        ```

        The check tries this example, no branches, no lessons done, a branch with one attach point, and lessons done in another order than the branches.
        """),
        L.task("Complete `can_begin`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_can_begin():
    box_can_begin = L.code_box("can-begin", lines=8)
    box_can_begin
    return (box_can_begin,)


@app.cell(hide_code=True)
def check_can_begin(box_can_begin):
    result_can_begin = L.check("can-begin", box_can_begin.value)
    result_can_begin
    return (result_can_begin,)


@app.cell(hide_code=True)
def hints_can_begin():
    L.hints(
        "can-begin",
        [
            "`for branch, points in attach.items():` gives each branch with its list of attach points; its first one is `points[0]`.",
            "Start with an empty list, and keep only the branches that can begin; what decides that is their first attach point.",
        ],
        why="A branch begins after its first attach point, so that is the one to look at. Going through `attach` from the start keeps its order.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### What your progress says

    Your `can_begin`, on the attach points of all {len(C.LATER)} branches and on the lessons you have done:
    """)
    return


@app.cell(hide_code=True)
def demo_can_begin(box_can_begin, result_can_begin):
    _ready = C.my_branches(box_can_begin.value) if result_can_begin.state == "correct" else None
    if result_can_begin.state != "correct":
        _out = mo.md("*Make `can_begin` right in exercise 2 first; then it runs here.*")
    elif _ready is None:
        _out = mo.md("*Your `can_begin` stops with an error on the route data of the course and your progress. Change it in exercise 2 and check it again.*")
    else:
        _lines = [f"**You can begin {len(_ready)} of the {len(C.LATER)} branches.**", ""]
        _lines += [f"- {C.short_title(_k, L.lang)}" for _k in _ready if _k in C.ATTACH]
        _wait = C.waiting(_ready)
        if _wait:
            _lines += ["", "Not yet:", ""]
            _lines += [f"- {C.short_title(_k, L.lang)}, after [{C.course.title(C.ATTACH[_k][0], L.lang)}](/course/open/{C.ATTACH[_k][0]}?lang={L.lang})" for _k in _wait]
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

        **How this connects.** The map you met in [The map of AI and how this course is organized](/course/open/start-map-of-ai?lang=en), at the start of the course, now with the attach points that show where each branch grows out of the main route; and the *Not checked* part of [Your reproduction 8: report and share](/course/open/repro-own-report?lang=en), turned into your next question. This is the last lesson of the main route, the last step to the third of its milestones:
        """),
        mo.md(C.milestone_lines(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_kinds, result_can_begin, explain_next_cell):
    L.wrap_up(result_kinds, result_can_begin, explain_next_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
