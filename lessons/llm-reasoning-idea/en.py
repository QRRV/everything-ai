# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-reasoning-idea. Route: course/route/lessen/taal.yaml. Checks and the course's run: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-reasoning-idea", "en")
    C = L.checks
    R = C.course()      # the course's run: Qwen3-0.6B-Base on the first 30 test questions of GSM8K
    S = C.SCORES        # what that run says, read and voted with the course's own solutions
    Q1 = R["questions"][0]


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Did post-training help?](/course/open/llm-eval-chat?lang=en) and [How sure is a number? Confidence intervals and the bootstrap](/course/open/stats-bootstrap?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_program():
    _quick_program = L.choice(
        "quick-program",
        question="In *Did post-training help?*, what decided whether a story did what its instruction asked?",
        options={
            "right": "A program that checks it, such as whether a word is left out",
            "reward": "A reward model that gives every story a score",
            "people": "You, reading every story",
        },
        correct="right",
        why={
            "right": "The instructions were verifiable: a program checked every answer, the same way for each model.",
            "reward": "There was no reward model: your own functions checked every answer. See [Did post-training help?](/course/open/llm-eval-chat?lang=en).",
            "people": "Your functions checked all 600 answers. See [Did post-training help?](/course/open/llm-eval-chat?lang=en).",
        },
    )
    _quick_program
    return


@app.cell(hide_code=True)
def quick_check_error():
    _quick_error = L.choice(
        "quick-error",
        question="An accuracy measured on 30 questions has a standard error of about 9 points. On about how many questions would it be 3 points?",
        options={
            "three": "90",
            "right": "270",
            "thirty": "900",
        },
        correct="right",
        why={
            "right": "The standard error shrinks with the square root of the number of questions: a third of it needs nine times as many.",
            "three": "Three times the questions divides it by the square root of 3, not by 3. See [How sure is a number?](/course/open/stats-bootstrap?lang=en).",
            "thirty": "Thirty times the questions divides it by the square root of 30, about 5.5. See [How sure is a number?](/course/open/stats-bootstrap?lang=en).",
        },
    )
    _quick_error
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every token a model writes comes out of one forward pass through its blocks, 28 of them in Qwen3-0.6B. A question that needs several steps of work, such as a word problem with three sums, gets no more of that work than a question that needs one, if the model answers at once. People do not answer such questions at once either: they write the steps down.

    **An everyday example.** 47 × 38 in your head, against on paper. On paper you only ever do one small step, and you read back what you wrote.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    _guess_steps = L.guess(
        "steps",
        question=f"Qwen3-0.6B-Base, a small open base model, answers the first {S['n']} problems of GSM8K, OpenAI's benchmark of word problems from primary school (MIT license). With four examples that show only their answer, it gets {S['direct']} right. With the same four examples and their steps written out, how many?",
        options={
            "same": "About as many: it is the same model",
            "few": "A few more, up to twice as many",
            "half": "About half of them",
        },
        reveal=f"**{S['steps']} of {S['n']}** in the course's run: the same weights, the same questions, only the prompt changed. The Qwen3 Technical Report (2025) gives {C.number(C.QWEN3_REPORT, 1, 'en')}% on all 1,319 test problems with such a prompt.",
    )
    _guess_steps
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## The idea

        **One token, one forward pass.** Qwen3's direct answer to question 1 is *{Q1['direct']['text'].strip()}* That is {Q1['direct']['tokens']} tokens, so {Q1['direct']['tokens']} forward passes, and wrong. The answer with steps is {Q1['steps']['tokens']}, each pass reading all written so far through attention, and ends in the right {Q1['answer']}. Steps give more computation for the same question (in theory, a transformer of fixed depth cannot do some tasks in one pass, and can with enough written steps), and smaller problems, each one a single pass can do.
        """),
        L.watch_out("*Reasons* is a picture. The model still predicts one token after another: its steps are text it wrote and then reads."),
        mo.md(rf"""
        **Three ways to get the steps.** Worked examples in the prompt: few-shot prompting with a chain of thought. Asking without examples: zero-shot, *Let's think step by step.* after the question, the experiment of [How research works](/course/open/res-research-cycle?lang=en); in the course's run, {S['zero']} of {S['n']} right. Or training the model to write them: reasoning models such as OpenAI o1 and DeepSeek-R1 (as of October 2026) learned it with reinforcement learning, DeepSeek-R1 on answers a program checks, as in [Reinforcement learning with verifiable rewards](/course/open/llm-rlvr?lang=en).

        **Test-time compute.** Writing steps spends more computation while answering, and so do several answers. In the course's run an answer was {C.number(S['tokens']['direct'], 1, 'en')} tokens direct and {C.number(S['tokens']['steps'], 0, 'en')} with steps on average, and {C.SAMPLES} samples {C.number(C.SAMPLES * S['tokens']['sample'], 0, 'en')} together: on the course's CPU about {C.number(S['seconds']['direct'], 0, 'en')} s, {C.number(S['seconds']['steps'], 0, 'en')} s and {C.number(S['seconds']['votes'] / 60, 0, 'en')} minutes per question.

        **Voting and pass@k.** Sample $k$ answers at a temperature above 0, read the final number of each, and take the one that comes most often: majority voting, the same vote as in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), for chains of thought also called self-consistency. Pass@k asks something else: is *any* of the $k$ right? That needs a check that knows the answer, here the answer key; voting needs none.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **chain of thought** | the steps a model writes before its final answer |
        | **few-shot prompting** | a few worked examples in the prompt, before the real question |
        | **zero-shot prompting** | the question or an instruction, without examples |
        | **test-time compute** | the computation a model spends while it answers |
        | **pass@k**, $k$ | whether at least one of $k$ answers is right, by a check that knows the answer |
        | **reasoning model** | a model trained, mostly with reinforcement learning, to write a long chain of thought before it answers |
        | **faithfulness** | whether the written steps are the real reason for the answer |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play_questions():
    mo.md(rf"""
    **The course's run, question by question.** Qwen3-0.6B-Base on the first {S['n']} test questions of GSM8K, not chosen ones, with both prompts, greedy. The bars are the tokens of each answer, with ✓ or ✗ from the course's checker; under them, both answers in full.
    """)
    return


@app.cell(hide_code=True)
def play_questions():
    _read = C.course_functions()[0]
    _labels = {"title": "Question {i}: the tokens of each answer", "x": "tokens written", "direct": "direct", "steps": "with steps",
               "tokens": "tokens", "read": "read:", "nothing": "nothing", "key": "answer key", "question": "Question {i} of {n}",
               "alt": "The tokens of the direct answer and of the answer with steps to the chosen question, as two bars, each with a tick if the course's checker found the right number and a cross if not."}
    _play = L.explore("questions", "question", lambda _v: C.answer_frame(R, int(_v), _read, _labels), start=1, low=1, high=S["n"], step=1,
              role="", task="Find a question where the direct answer is right and the chain of thought is wrong. Where does the chain go wrong?")
    _play
    return


@app.cell(hide_code=True)
def text_play_questions_after():
    mo.md(rf"""
    Question {C.COUNTER + 1} is one: the direct answer, {R['questions'][C.COUNTER]['answer']}, is right, and the chain of thought writes *60 \* 3 = 180 meters each week*: it forgets that James runs 3 sprints each time. Written steps help on most questions, not on all, and a chain of thought can read well and be wrong.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write a prompt of a few worked examples, each `Question:`, `Answer:` with the steps, `The answer is N.` and a blank line; end with the new question and `Answer:`.
    2. Let the model write until it starts a question of its own or stops; greedy for one answer.
    3. Read its final number with a program: the number after the first `The answer is`, else the last number.
    4. To vote: sample $k$ answers at temperature 0.7, read each, take the number that comes most often.
    5. Compare with the answer key; for pass@k, check whether any of the $k$ is right.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 with one example, and steps 3 to 5 on the course's answers to question 1. `C.examples()` gives the lesson's four examples, each a dict with `"question"`, `"steps"` and `"answer"`. Run the cell.
    """)
    return


@app.cell
def example_question_1():
    _example = C.examples()[0]
    _q = R["questions"][0]
    # 1. one worked example with its steps, then the new question
    _prompt = "Question: " + _example["question"] + "\nAnswer: " + _example["steps"] + "\nThe answer is " + _example["answer"] + ".\n\n"
    _prompt = _prompt + "Question: " + _q["question"] + "\nAnswer:"
    # the prompt line by line: repr puts each line in quotes, so a blank line shows as ''
    for _line in _prompt.split("\n"):
        print(repr(_line))
    # 3. the number after the first "The answer is", in the direct answer and in the one with steps
    for _kind in ["direct", "steps"]:
        _after = _q[_kind]["text"].split("The answer is")[1]
        print(_kind, repr(_after), "->", _after.split()[0].strip("."))
    # 4. what the checker read in the 8 samples, counted
    _read = ["0", "26", "18", "18", "26", "18", "18", "224"]
    _counts = {}
    for _a in _read:
        if _a in _counts:
            _counts[_a] = _counts[_a] + 1
        else:
            _counts[_a] = 1
    print(_counts)
    # 5. the answer key
    print("answer key:", _q["answer"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    `repr` puts every line of the prompt in quotes, so the blank line after the example shows as `''`: there the prompt has `\n\n`, two new lines in a row. The last line is `'Answer:'`, where the model goes on. The checker reads {C.course_functions()[0](Q1['direct']['text'])} in the direct answer and {C.course_functions()[0](Q1['steps']['text'])} in the one with steps; the answer key is {Q1['answer']}. Of the {C.SAMPLES} samples, {C.FIRST_VOTE['top']} comes {C.FIRST_VOTE['count']} times: the vote is {C.FIRST_VOTE['top']}, and right; and as at least one sample is right, so is pass@{C.SAMPLES}.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompt():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A prompt with worked examples

        Complete `few_shot_prompt(examples, question, steps)`: `examples` is a list of dicts as `C.examples()` gives them. With `steps` true, every example is laid out as in the worked example, a blank line after it; with `steps` false, without the steps and their new line: `Answer: The answer is 72.` Then `Question: `, the new question, a new line and `Answer:`, without a space after it. The check tries the lesson's examples with and without steps, an example of its own, and steps of several lines.
        """),
        L.task("Complete `few_shot_prompt`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("few-shot-prompt", lines=12)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("few-shot-prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "few-shot-prompt",
        ["Build a list of texts, one example at a time, with `if steps:` for the part that differs, and join it with `\"\".join(parts)`."],
        why="The model goes on with the pattern of its prompt: examples with their steps before `The answer is` make it write steps before its answer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A program that reads the answer

        Complete `number(word)` and `read_answer(text)`. `number` takes the commas out of a word and the signs `$ % . : ; ! ? * ( )` off its ends, and gives the word if it is then all digits, else `None`; `"1250".isdigit()` is `True`, `"19.50".isdigit()` is `False`. `read_answer` gives the first number after the first `The answer is`; without that phrase, the last word that is a number; `None` if there is none. A number stays text, such as `"18"`, as the answer key is. The check reads answers of the course's run and texts made for it.
        """),
        L.task("Complete `number` and `read_answer`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-answer", lines=18)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-answer", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-answer",
        ["With the phrase, the words of `text.split(\"The answer is\")[1]` in order; without it, the words from the end, `reversed(text.split())`. In both, the first word for which `number` is not `None`."],
        why="The examples teach the model to end with *The answer is* and the number; an answer that ends otherwise mostly has its result as its last number.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The vote

        Complete `majority_vote(answers)`: the answer that comes most often in the list, without `None`; of answers that come as often, the first; `None` if there is none. So `majority_vote(["18", "26", "18", None])` gives `"18"`. The check tries a few groups, and every group of 8 samples of the course's run, as its checker read them.
        """),
        L.task("Complete `majority_vote`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("majority-vote", lines=10)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("majority-vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "majority-vote",
        ["Count in a dict, as in the worked example; then walk the list in order and keep an answer only when it comes more often than the one you kept."],
        why="Walking the list in order and keeping only a larger count gives a tie to the answer that came first.",
    )
    return


@app.cell(hide_code=True)
def text_play_votes():
    mo.md(rf"""
    **More samples: voting and pass@k.** For every question the course drew {C.SAMPLES} answers with steps at temperature {C.number(C.TEMPERATURE, 1, 'en')}. The block reads them with your `read_answer` and votes with your `majority_vote` (the course's, until both are right), over the first $k$ samples of each question.
    """)
    return


@app.cell(hide_code=True)
def play_votes(box_read, box_vote, result_read, result_vote):
    _course_read, _course_vote = C.course_functions()
    _mine = result_read.state == "correct" and result_vote.state == "correct"
    _read = run_code(box_read.value).names["read_answer"] if _mine else _course_read
    _vote = run_code(box_vote.value).names["majority_vote"] if _mine else _course_vote
    _s = C.scores(R, _read, _vote)
    _labels = {"title": f"Right of {_s['n']}, for k samples", "x": "k, samples per question", "y": "right answers", "vote": "voting",
               "pass": "pass@k", "greedy": "greedy chain of thought", "lang": "en",
               "said": "**k = {k}:** voting {vote}, pass@{k} {passes}, greedy {greedy} of {n}; about {tokens} tokens per question.",
               "alt": f"Right answers of {_s['n']} for k from 1 to {C.SAMPLES}: voting as a solid line, pass@k dashed, the greedy chain of thought as a thin flat line, and a dashed line at the chosen k."}
    _play = L.explore("votes", "k", lambda _v: C.votes_frame(_s, int(_v), _labels), start=1, low=1, high=C.SAMPLES, step=1,
              role="hyper", task="Raise k. Which line keeps rising, and why can voting not follow it?")
    _play
    return


@app.cell(hide_code=True)
def text_play_votes_after():
    mo.md(rf"""
    Pass@k never goes down as $k$ grows, and ends at {S['pass'][-1]}: one right answer is enough, when something checks it. Voting gains only when the right answer is the most common one; with the course's checker it ends at {S['vote'][-1]}, against {S['steps']} for greedy, within the noise of 30 questions. Wang and others (2022) found 17.9 points more on GSM8K with far larger models; one small model on 30 questions cannot tell whether the gain is smaller at 0.6B or this is chance.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Qwen3 on your prompt and your checker

    The demo gives Qwen3-0.6B-Base the first {C.QUESTIONS['cpu']} test questions with your `few_shot_prompt`, without and with steps, greedy, and reads every answer with your `read_answer`; on a GPU all {C.QUESTIONS['gpu']}, with a vote over {C.VOTE['gpu']} samples. On the course's CPU with 2 threads it took {C.number(C.DEMO['minutes'], 1, 'en')} minutes, measured; on a GPU it was not measured. Greedy writing is deterministic: on a CPU like the course's, your line with steps lies on the course's, dashed; another device can change a token through rounding.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B-Base (1.2 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Qwen3-0.6B-Base is not on your computer yet. The button downloads it once from Hugging Face, `Qwen/Qwen3-0.6B-Base` at a fixed version, 1.2 GB, under the Apache-2.0 license: the same files as in [The same steps on an open model](/course/open/llm-same-on-open-model?lang=en).*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 1.2 GB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Qwen3-0.6B-Base is downloaded.*")
    return


@app.cell(hide_code=True)
def run_button_cell():
    run_button = mo.ui.run_button(label="Run Qwen3 on your prompt")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_prompt, box_read, box_vote, result_prompt, result_read, result_vote, run_button):
    ran = None
    if result_prompt.state != "correct" or result_read.state != "correct":
        _out = mo.md("*Make `few_shot_prompt` and `read_answer` right in exercises 1 and 2 first: the demo uses them.*")
    elif not C.downloaded():
        _out = mo.md("*Download Qwen3 first, with the button above.*")
    elif not run_button.value:
        _out = mo.md("*Press the button to run Qwen3.*")
    else:
        _n = C.QUESTIONS[C.where()["kind"]]
        _live = L.live({"title": f"Right so far, of the first {_n}", "x": "question", "y": "right so far",
                        "alt": "How many answers of your run are right so far: with steps as the line with dots, direct as a grey dashed line, and the course's run with steps on the same questions as another dashed line."},
                       _n, role="ink", low=0, high=_n)
        _live.reference(C.so_far(R, "steps", _n), "", role="ink")
        _direct = []
        _what = {"direct": "direct", "steps": "with steps", "votes": f"{C.VOTE['gpu']} samples"}

        def _show(_d):
            if _d["phase"] == "load":
                _live.line("**Loading Qwen3** …")
            elif _d["phase"] == "answer":
                _live.line(f"**Question {_d['i']} of {_d['n']}** · {_what[_d['kind']]} …")
            elif _d["phase"] == "answered":
                _live.line(f"**Question {_d['i']} of {_d['n']}** · {_what[_d['kind']]} · {_d['tokens']} tokens · {_d['seconds']:.0f} s · {'right' if _d['right'] else 'wrong'}")
            elif _d["phase"] == "question":
                _direct.append((_d["i"], _d["direct"]))
                _live.reference(_direct, "")
                _live.add(_d["i"], _d["steps"], f"**Question {_d['i']} of {_d['n']}:** right so far {_d['steps']} with steps, the line with dots, and {_d['direct']} direct, the grey dashed line")

        _vote = run_code(box_vote.value).names["majority_vote"] if result_vote.state == "correct" else C.course_functions()[1]
        _r = C.run(run_code(box_prompt.value).names["few_shot_prompt"], run_code(box_read.value).names["read_answer"], _vote, _show)
        if isinstance(_r, str):
            _out = _live.stopped({
                "prompt": "*Your `few_shot_prompt` stops with an error on a question of the demo, or gives no text. Does it still pass its check?*",
                "read": "*Your `read_answer` stops with an error on an answer of Qwen3. Does it still pass its check?*",
                "vote": "*Your `majority_vote` stops with an error on the samples of a question. Does it still pass its check?*",
                "run": "*Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.*",
                "memory": "*The memory of your device is full. Close other programs, and press the button again.*",
            }[_r])
        else:
            ran = _r
            _right = C.run_scores(_r)
            _out = mo.vstack([mo.Html(_live.picture()), mo.md(f"Right: {_right['steps']} of {_right['n']} with steps, the line with dots, and {_right['direct']} direct, the grey dashed line. The course's run with steps is dashed too: where it does not show, it lies under yours.")])
    _out
    return (ran,)


@app.cell(hide_code=True)
def demo_result(ran):
    _c = C.run_scores(R)
    _m = C.run_scores(ran) if ran is not None else None
    _cols = [("the course's run", _c)]
    if _m is not None:
        _cols = [(f"your run, {ran['device']}", _m)]
        if _m["n"] < _c["n"]:   # the course's answers to the same questions, beside all its own
            _cols.append((f"the course's run, its first {_m['n']}", C.run_scores({"questions": R["questions"][:_m["n"]]})))
        _cols.append((f"the course's run, all {_c['n']}", _c))
    _runs = [_s for _n, _s in _cols]

    def _row(_name, _key):
        return f"| {_name} | " + " | ".join("" if _s[_key] is None else f"{_s[_key]} of {_s['n']}" for _s in _runs) + " |\n"

    def _two(_name, _key):
        return f"| {_name} | " + " | ".join(f"{C.number(_s[_key]['direct'], 1, 'en')} and {C.number(_s[_key]['steps'], 1, 'en')}" for _s in _runs) + " |\n"

    _head = "| | " + " | ".join(_n for _n, _s in _cols) + " |\n|---|" + "---|" * len(_cols) + "\n"
    mo.md(_head + _row("right, direct", "direct") + _row("right, with steps", "steps") + _row(f"right, vote of {C.SAMPLES}", "vote")
          + _two("tokens per answer, direct and with steps", "tokens") + _two("seconds per answer, direct and with steps", "seconds")
          + "\nWith 10 questions, one answer more or less is 10 points; with 30, the standard error is about 9 points."
          + ("\n\n*The course's run stands in until you run Qwen3 yourself.*" if _m is None else ""))
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(r"""
    **Your answers, question by question:** the block of *The idea*, now over your run, with what your `read_answer` read.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(box_read, ran):
    if ran is None:
        _out = mo.md("*Once you have run Qwen3, you can go through its answers here.*")
    else:
        _read = run_code(box_read.value).names["read_answer"]
        _labels = {"title": "Question {i}: the tokens of each answer", "x": "tokens written", "direct": "direct", "steps": "with steps",
                   "tokens": "tokens", "read": "read:", "nothing": "nothing", "key": "answer key", "question": "Question {i} of {n}",
                   "alt": "The tokens of your run's direct answer and answer with steps to the chosen question, as two bars, each with a tick if your read_answer found the right number and a cross if not."}
        _out = L.explore("replay", "question", lambda _v: C.answer_frame(ran, int(_v), _read, _labels), start=1, low=1,
                         high=len(ran["questions"]), step=1, role="",
                         task="Find an answer your checker reads as nothing. What did the model write there?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(rf"""
    In the course's run, the checker reads nothing in the chains of thought of questions {C.UNREAD[0]} and {C.UNREAD[1]}: they end in a negative number, `-$10,000` and `-20`, and a word with a minus sign is not all digits. None of these {S['n']} answer keys is negative, so here it changes no score; a few answers of GSM8K are, and a checker for those would have to read the sign.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_faithful():
    mo.md(r"""
    ### Exercise 4. Is the chain the reason?
    """)
    return


@app.cell(hide_code=True)
def explain_faithful():
    explain_faithful_cell = L.explain(
        "faithful",
        question="Qwen3 wrote a chain of thought that ends in the right number. Does that show it found the number *because of* those steps? How could you test it?",
        model_answer="No. A chain of thought can be a story that fits the answer rather than the way to it: when the right answer in a prompt's examples is always (A), models pick (A) more often without saying so in their steps (Turpin and others, 2023). To test it, change the chain and see whether the answer follows: put a mistake in a step, or cut the chain short and ask for the answer at once (Lanham and others, 2023).",
        points=[
            "a right answer with steps does not show that the steps caused it",
            "the steps can fit the answer without causing it, and a right answer can come with wrong steps",
            "change a step, or cut the chain short, and see whether the answer changes",
            "this is faithfulness, and it is an open research question",
        ],
    )
    explain_faithful_cell
    return (explain_faithful_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(rf"""
    ## Wrap-up

    **How this connects.** [How research works](/course/open/res-research-cycle?lang=en) used Kojima's experiment as an example; here is why it works. Your checker is a program like those of [Did post-training help?](/course/open/llm-eval-chat?lang=en), and DeepSeek-R1 was trained with such checkers. In [Project: a model that writes its steps, from scratch](/course/open/llm-reasoning-from-scratch?lang=en) you train a small model from scratch to write its own steps, on data where you control every example.

    **Limits.** Word problems are a narrow part of reasoning, and 30 questions give about 9 points of standard error. GSM8K may have been in Qwen3's pretraining (contamination), which the course cannot check. A hint that text in this form was: without any example, {C.ZERO_STYLE['phrase']} of its {S['n']} zero-shot answers still write *The answer is*. Names and scores of models age fast: these are as of October 2026; the lesson's own numbers are the course's run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_prompt, result_read, result_vote, explain_faithful_cell):
    L.wrap_up(result_prompt, result_read, result_vote, explain_faithful_cell)
    return


@app.cell(hide_code=True)
def look_back():
    _look = L.choice(
        "look-temperature",
        question=f"### Looking back\n\nThe course drew its samples at temperature {C.number(C.TEMPERATURE, 1, 'en')}. What would temperature 0 have given?",
        options={
            "right": f"The same answer {C.SAMPLES} times: the greedy one",
            "wrong": f"{C.SAMPLES} different answers, each one wrong",
            "faster": "The same answers as at 0.7, only faster",
        },
        correct="right",
        why={
            "right": "At temperature 0 the most likely token always wins, so every sample is the greedy answer.",
            "wrong": "At temperature 0 every sample is the greedy answer. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
            "faster": "The temperature changes which tokens are drawn, not how fast. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
        },
    )
    _look
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
