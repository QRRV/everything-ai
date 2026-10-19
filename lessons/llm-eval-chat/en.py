# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-chat. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-chat", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en) and [Keeping test data out of training data](/course/open/llm-decontamination?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_checker():
    quick_checker = L.choice(
        "quick-checker",
        question="In the project about DPO, what decided which answer of a pair was chosen, and what counted the test instructions afterwards?",
        options={
            "people": "People chose the answers; `words_used` counted the test instructions",
            "right": "`words_used` both times: it made the pairs, and it counted",
            "reward": "A reward model chose the answers; people counted the test instructions",
        },
        correct="right",
        why={
            "people": "No people took part: the checker made every pair. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
            "reward": "DPO needs no reward model: the pairs came from the checker. See [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en).",
            "right": "Keep that in mind: the model learned what that checker rewards, and the same checker gave it its score.",
        },
    )
    quick_checker
    return (quick_checker,)


@app.cell(hide_code=True)
def quick_check_clean():
    quick_clean = L.choice(
        "quick-clean",
        question="Half of the questions of a benchmark turn out to be in the training data. Which score still says something about skill?",
        options={
            "all": "The score on all questions: the model still had to answer them",
            "half": "The score on all questions, halved",
            "right": "The score on the clean subset: the questions that were not in the training data",
        },
        correct="right",
        why={
            "all": "On the questions it saw, a model can answer from memory: that score mixes memory and skill. See [Keeping test data out of training data](/course/open/llm-decontamination?lang=en).",
            "half": "Halving fixes nothing: the seen questions can be answered from memory, and they are not simply worth half. See [Keeping test data out of training data](/course/open/llm-decontamination?lang=en).",
            "right": "Today you make that subset for instructions, where it is not the text that must be held out, but its words.",
        },
    )
    quick_clean
    return (quick_clean,)


@app.cell(hide_code=True)
def text_why():
    _c = C.MEASURED["course"]
    mo.md(f"""
    ## Why this matters

    SFT and DPO took the course's model from {_c['base']['with']} to {_c['sft']['with']} to {_c['dpo']['with']} of the {_c['total']['with']} test instructions of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en). Did post-training help? Two doubts. The checker that counts those instructions is the one that made the DPO pairs, so DPO may have learned that checker rather than instructions. And most test instructions were also training instructions: the models were asked for the same words while they trained. A fairer test asks other things a program can check, on prompts the model never trained on. For large models, IFEval (Zhou and others, 2023) tests instruction following with such checks: 25 kinds of verifiable instructions, such as *include these keywords*, *do not include these words* and *answer in at most this many sentences*, in 541 prompts.

    **An everyday example.** A driving student who has practised one route passes a test on that route. To know whether they can drive, test them on another route, and ask for a turn they never practised.
    """)
    return


@app.cell(hide_code=True)
def predict_without():
    guess_without = L.guess(
        "without",
        question="SFT and DPO taught the course's model to write stories that use the words it is asked for. Now ask it for a story *without* the word *dog*. Compared with the model before post-training, does the model after DPO leave the word out more often, about as often, or less often?",
        options={
            "more": "More often: it has learned to follow instructions",
            "same": "About as often: it was never trained on this kind",
            "right": "Less often: it has learned to use the word it is given",
        },
        reveal=f"**Less often.** Of {C.WITHOUT} such prompts, the course's model left the word out {C.MEASURED['course']['base']['without']} times before post-training, {C.MEASURED['course']['sft']['without']} times after SFT, and {C.MEASURED['course']['dpo']['without']} times after DPO. It never learned to read an instruction: it learned that a word in the instruction belongs in the story.",
    )
    guess_without
    return (guess_without,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A verifiable instruction** is one whose result a short program can check: *use the word kite*, *do not use the word dog*, *write at most three sentences*. No person and no model has to judge, so the check is the same for every answer and every model. It is not perfect: a story can use *kite* without being good, and `sentences` counts *Mr.* as the end of one. But a fixed check that is a little wrong is wrong in the same way for every model, and that is what a comparison needs.

        **More than one kind.** A model fine-tuned on one kind of instruction can get better at that kind and worse at others. So the test asks three kinds: the one the models trained on, *with the words*, and two they never saw, *without a word* and *at most a number of sentences*.

        **Held out by what varies.** All prompts of a kind share their template: *Write a story with the words: … Story:*. So a check on n-grams, as in decontamination, cannot tell them apart: with a small n, every prompt shares the n-grams of its template with the training data thousands of times, and with n = 13 a prompt of eight to ten words is shorter than one n-gram, so it shares none. What has to be held out is the part that varies: the requested words. A prompt whose words, in any order, were never a training instruction belongs to the **clean subset**.
        """),
        draw.flow(
            [("Prompt", ["Write a story without", "the word: dog."]),
             ("Model", ["base, SFT", "or SFT and DPO"]),
             ("Check", ["avoids(story, \"dog\")", "True or False"])],
            forward=["answer", "story"],
            back=[],
            title="A verifiable instruction"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### New words and symbols

    | Word or symbol | What it means | In code |
    |---|---|---|
    | **verifiable instruction** | an instruction whose result a short program can check, the same way every time | `avoids(story, word)` |
    | **verifiable eval** | a test of instruction following made of verifiable instructions, on held-out prompts, the same for every model | |

    ### Recipe

    To test whether post-training helped:

    1. Choose instructions a program can check, of more than one kind: also kinds the model was never trained on.
    2. Make the prompts held out: the part that varies must never have been a training instruction.
    3. Ask every model the same prompts, drawn the same way, with the same seed for a prompt.
    4. Per kind, count how many answers pass; and see whether passing depends on what was asked.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first prompt of the kind *without*, and the answers of the course's three models to it, each checked.
    """)
    return


@app.cell
def example_check():
    _d = C.course()
    _k = 100   # the first prompt of the kind "without"
    _kind, _words, _prompt = _d["prompts"][_k]
    print(_prompt.split("\n")[0])
    for _m in C.MODELS:
        _story = _d["answers"][_m][_k]
        _found = {_w.strip(".,!?;:\"'") for _w in _story.lower().split()}
        print(_m, _words[0] not in _found, _story[:60] + " ...")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `True` means the story leaves the word out, as the prompt asks. The check is the inverse of `words_used`, word for word: the same words of the story, with the same punctuation stripped off.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_avoids():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Without the word

        Complete `avoids(story, word)`. Give back `True` when the story does not use `word`, and `False` when it does, seeing the words of the story as `words_used` sees them: in small letters, without the punctuation around them. The check calls `avoids("The kite flew over the park.", "dog")`, and others.
        """),
        L.task("Make `avoids` say whether a story leaves a word out."),
    ])
    return


@app.cell(hide_code=True)
def exercise_avoids():
    box_avoids = L.code_box("avoids", lines=6)
    box_avoids
    return (box_avoids,)


@app.cell(hide_code=True)
def check_avoids(box_avoids):
    result_avoids = L.check("avoids", box_avoids.value)
    result_avoids
    return (result_avoids,)


@app.cell(hide_code=True)
def hints_avoids():
    L.hints(
        "avoids",
        [
            "Your `words_used` of the lesson about SFT already sees the words of a story the way this check must. The story is already in small letters here: what else did it do to every word?",
            "As soon as one word of the story is the word to avoid, the answer is known. And when the loop has seen every word?",
        ],
        why="The two checks see words in exactly the same way, so a word that `words_used` would find is one that `avoids` objects to.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sentences():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Counting sentences

        Complete `sentences(story)`. Give back how many sentences the story has, by this rule: a sentence ends at a `.`, `!` or `?`, and a run of them, such as `...` or `?!`, ends one sentence, not more. The check calls `sentences("Tom ran. He fell... Was he hurt?")`, and others.
        """),
        L.task("Make `sentences` count the runs of `.`, `!` and `?`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sentences():
    box_sentences = L.code_box("sentences", lines=10)
    box_sentences
    return (box_sentences,)


@app.cell(hide_code=True)
def check_sentences(box_sentences):
    result_sentences = L.check("sentences", box_sentences.value)
    result_sentences
    return (result_sentences,)


@app.cell(hide_code=True)
def hints_sentences():
    L.hints(
        "sentences",
        [
            "In `...` only the first dot starts something new. What is true of the character before that first dot, and not of the ones before the other two?",
            "`before` remembers whether the previous character ended a sentence. When does a character add one to `count`?",
        ],
        why="A run of end marks is one end: counting only where a run starts gives one per sentence, whether it ends in `.`, `?!` or `...`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_held():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The clean subset

        Complete `held_out(prompts, train)`. Both are lists of requested words, one list per instruction: `prompts` those of the test, `train` those the models trained on. Give back the prompts, in their order, whose words, in any order, were never a training instruction. The check calls `held_out([["park", "kite"], ["dog"], ["sun", "park"]], [["kite", "park"], ["cat"]])`, and others.
        """),
        L.task("Make `held_out` keep the prompts whose words, together, were never a training instruction."),
    ])
    return


@app.cell(hide_code=True)
def exercise_held():
    box_held = L.code_box("held-out", lines=8)
    box_held
    return (box_held,)


@app.cell(hide_code=True)
def check_held(box_held):
    result_held = L.check("held-out", box_held.value)
    result_held
    return (result_held,)


@app.cell(hide_code=True)
def hints_held():
    L.hints(
        "held-out",
        [
            "*kite and park* and *park and kite* are the same instruction. How do you make both into the same thing, one that a set can hold?",
            "A prompt stays when its words, made the same way, are not among those of the training instructions. Does it go when it shares only one word?",
        ],
        why="The template is the same for every prompt, so the words decide whether a prompt is new; sorted, the order no longer matters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Test the three models

        The course's base model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en), its SFT model, and that model with its DPO adapters answered all {len(C.prompts())} prompts: the {C.T.TESTS} test instructions of the lesson about SFT, {C.WITHOUT} asking for a story without a word, and {C.SENTENCES} asking for at most 1 to 5 sentences. Your checks score their answers below; the clean subset is what your `held_out` keeps of the test instructions, against the instructions of the course's SFT run (for your models: of your SFT run) and of the DPO pairs. If you have models of your own, the button lets them answer the same prompts: on the course's CPU, with one thread, that took about {round(C.MEASURED['yours']['seconds'] / 60)} minutes for its models of the demo runs.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def button_cell():
    _have = [_m for _m, _there in C.mine().items() if _there]
    ask_button = mo.ui.run_button(label="Let my models answer", disabled=not _have)
    ask_button
    return (ask_button,)


@app.cell(hide_code=True)
def demo_ask(ask_button):
    asked = None
    _note = None
    if ask_button.value:
        def _progress(_p):
            mo.output.replace(mo.md(f"**{_p['model']}** · prompt {_p['done']} of {_p['total']} · {C.number(_p['seconds'] / 60, 1, L.lang)} minutes"))
        _r = C.ask(_progress)
        if _r == "run":
            _note = "*Your models cannot answer: a module of your project stops with an error. Do your modules in `work/projects/llm/` still pass their checks?*"
        else:
            asked = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (asked,)


@app.cell(hide_code=True)
def demo_score(asked, box_avoids, box_held, box_sentences, result_avoids, result_held, result_sentences):
    _course = {"avoids": run_code(C.EXERCISES["avoids"].solution).names["avoids"],
               "sentences": run_code(C.EXERCISES["sentences"].solution).names["sentences"],
               "held_out": run_code(C.EXERCISES["held-out"].solution).names["held_out"]}
    _boxes = {"avoids": (box_avoids, result_avoids), "sentences": (box_sentences, result_sentences), "held_out": (box_held, result_held)}
    _fns, _notes = {}, []
    for _k, (_box, _result) in _boxes.items():
        _fns[_k] = run_code(_box.value).names.get(_k) if _result.state == "correct" else None
        if _fns[_k] is None:
            _fns[_k] = _course[_k]
            _notes.append(f"*The course's `{_k}` stands in until yours passes its check.*")
    _t = C.tools()
    _wu = run_code(C.T.EXERCISES["words-used"].solution).names["words_used"] if _t == "run" else _t["words_used"]
    if asked is not None and asked["answers"]:
        _answers, _whose = asked["answers"], "your models"
        _train = asked["train"]
    else:
        _answers, _whose = C.course()["answers"], "the course's models"
        _train = C.trained({"course"})
    _s = C.score(_answers, _train, _fns["avoids"], _fns["sentences"], _fns["held_out"], _wu)
    if isinstance(_s, str):   # one of the checks stops on the answers: the course's four stand in
        _notes.append(f"*Your `{_s if _s != 'words' else 'words_used'}` stops with an error on the answers, or gives no answer of the right kind: the course's checks stand in.*")
        _s = C.score(_answers, _train, _course["avoids"], _course["sentences"], _course["held_out"],
                     run_code(C.T.EXERCISES["words-used"].solution).names["words_used"])
    _names = {"base": "before post-training", "sft": "after SFT", "dpo": "after SFT and DPO"}
    _tot = _s["total"]
    _rows = [f"| {_names[_m]} | {_s[_m]['with']} of {_tot['with']} | {_s[_m]['clean']} of {_tot['clean']} | {_s[_m]['without']} of {_tot['without']} | {_s[_m]['sentences']} of {_tot['sentences']} |"
             for _m in C.MODELS if _m in _s]
    _per = [f"| {_names[_m]} | " + " | ".join(C.number(_s[_m]['per'].get(_n, 0), 1, L.lang) for _n in range(1, 6)) + " |" for _m in C.MODELS if _m in _s]
    mo.vstack([
        mo.md(f"How many answers of {_whose} pass, per kind:\n\n"
              "| | with the words, all | with the words, clean subset | without the word | at most N sentences |\n|---|---|---|---|---|\n" + "\n".join(_rows)
              + "\n\nThe mean number of sentences, for every number N the prompt asked for:\n\n"
              "| | N = 1 | N = 2 | N = 3 | N = 4 | N = 5 |\n|---|---|---|---|---|---|\n" + "\n".join(_per)),
        *[mo.md(_n) for _n in _notes],
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _c, _y = C.MEASURED["course"], C.MEASURED["yours"]
    mo.md(f"""
    On *with the words*, post-training helped a lot: for the course's models from {_c['base']['with']} to {_c['dpo']['with']} of {_c['total']['with']}, for its models of the demo runs from {_y['base']['with']} to {_y['dpo']['with']}. But most of those instructions were also training instructions, and on the clean subset the two differ: the course's model after DPO did {_c['dpo']['clean']} of its {_c['total']['clean']}, its model of the demo runs {_y['dpo']['clean']} of {_y['total']['clean']}. Only the clean subset says whether what a model learned carries over to words it was never asked for together. On *without the word*, DPO made both worse than after SFT: from {_c['sft']['without']} to {_c['dpo']['without']}, and from {_y['sft']['without']} to {_y['dpo']['without']}, of {_c['total']['without']}. And on *at most N sentences*, the models after SFT and DPO write about as many sentences whatever N asks: they do not read the number, so what passes depends on how long their stories happen to be. Post-training taught the kind of instruction it trained on, and its checker; not to follow instructions in general.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 4. Why not?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-not",
        question="After DPO, the course's model did most of the test instructions of the lesson about SFT, against a quarter after SFT. Why does that not show that DPO taught it to follow instructions?",
        options={
            "right": "Most test instructions were training instructions, and their checker made the DPO pairs",
            "small": "A hundred instructions are too few to say anything about a model",
            "base": "The base model did almost none of them, so any model would look better",
        },
        correct="right",
        why={
            "small": "A hundred is few, but the difference is large. The problem is what was measured: instructions the model trained on, counted by the checker it trained against.",
            "base": "The comparison is with the model after SFT, not only the base model. The problem is what was measured: instructions the model trained on, counted by the checker it trained against.",
            "right": "On held-out prompts, and on an instruction it never trained on, the model after DPO does far worse.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The checker `words_used` of [Project: teach your model to follow instructions (SFT)](/course/open/llm-sft?lang=en), now one of three kinds; the clean subset of [Keeping test data out of training data](/course/open/llm-decontamination?lang=en), now for the words of an instruction; and the models of [Project: DPO, preferences without a reward model](/course/open/llm-dpo?lang=en), now put side by side. Next: testing whether a model refuses what it should, and only that.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_avoids, result_sentences, result_held, choice_why):
    L.wrap_up(result_avoids, result_sentences, result_held, choice_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
