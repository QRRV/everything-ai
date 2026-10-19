# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-multiple-choice. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-multiple-choice", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How good is a language model?](/course/open/lm-nll?lang=en) and [Evaluating your base model](/course/open/llm-eval-base?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="After a context, a model gives the three tokens of a word the log-probabilities −1.2, −0.7 and −2.1, each after the tokens before it. What is the log-probability of the whole word after that context?",
        options={
            "mean": "−1.33, their mean",
            "right": "−4.0, their sum",
            "last": "−2.1, that of the last token",
        },
        correct="right",
        why={
            "mean": "The probability of the word is the product of the three probabilities, and the log of a product is the sum of the logs. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "last": "The word only comes when all three tokens come: the probabilities multiply, the logs add up. See [How good is a language model?](/course/open/lm-nll?lang=en).",
            "right": "Today a model scores whole answers that way: the sum of the log-probabilities of their tokens.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_windows():
    quick_windows = L.choice(
        "quick-windows",
        question="Your `token_losses` scores a text of 1,000 tokens in windows of 256. How many losses does it give?",
        options={
            "right": "999: every token but the first",
            "thousand": "1,000: one per token",
            "windows": "4: one per window of 256 tokens",
        },
        correct="right",
        why={
            "thousand": "The first token has nothing before it, so nothing predicts it. See [Evaluating your base model](/course/open/llm-eval-base?lang=en).",
            "windows": "With `reduction=\"none\"`, every predicted token gets its own loss, not every window. See [Evaluating your base model](/course/open/llm-eval-base?lang=en).",
            "right": "Today you read the log-probabilities of only some tokens: those of an answer, after a question.",
        },
    )
    quick_windows
    return (quick_windows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many benchmarks for base models are multiple-choice: HellaSwag, for one, gives the start of a text and four ways it could go on (Zellers and others, 2019). A base model cannot answer a question, but it can tell how likely each answer is. GPT-3 was tested that way: it compared the likelihoods of the options after the context (Brown and others, 2020). In this lesson you test your own model that way, on 500 questions about stories.

    **An everyday example.** You hear the start of a sentence and three possible endings. You do not have to say the ending yourself to point at the one that sounds most natural.
    """)
    return


@app.cell(hide_code=True)
def predict_sum():
    _m = C.MEASURED["yours"]
    guess_sum = L.guess(
        "sum",
        question=f"Each of the 500 questions has three sentences of a story and four next sentences: the real one, and three from other stories. Guessing gets about 125 right. The course's run of the capstone, a model of {C.number(_m['params'], 0, L.lang)} parameters trained on TinyStories, takes the option with the highest log-probability after the context. How many does it get right?",
        options={
            "right": "Hardly more than guessing",
            "half": "About half of the questions",
            "most": "Most of them",
        },
        reveal=f"**Hardly more than guessing: {_m['right']['sum']} of 500.** The plain log-probability says most of all how likely a sentence is in itself: a short one, or one of common words, wins after any context. This lesson measures what the context adds.",
    )
    guess_sum
    return (guess_sum,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Score every option.** Put the option after the context, and add the log-probabilities of its tokens, each read where the model predicts it: that is the log-probability of the whole option after the context.

    **The sum is not fair.** Every token adds a negative number, so an option of many tokens gets a low sum, also when it fits well. And a sentence of common words is likely after any context. GPT-3 was scored in two fairer ways:

    - **Per token.** Divide the sum by the number of tokens of the option: a long option is no longer punished for its length. Per byte, which the GPT-3 paper does not mention, works the same way and does not depend on the tokenizer.
    - **Against the option on its own.** Score the option also after a neutral start, for the stories here an end-of-text, the start of any story. Its log-probability there says how likely it is anyway. The option whose log-probability rises most through the context is the one the context points to:
    """)
    return


@app.cell(hide_code=True)
def formula_rise():
    L.formula(r"\begin{aligned} \text{rise} &= \log P(\text{option} \mid \text{context}) \\ &\quad - \log P(\text{option} \mid \text{start}) \end{aligned}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        GPT-3 did that for three of its benchmarks, ARC, OpenBookQA and RACE, with "Answer: " or "A: " as the neutral start, and divided the probabilities; the difference of the logs is the log of that division. On the 500 questions of this lesson, the course's run of the capstone gets this many right under each rule:
        """),
        C.rules_picture({"sum": "the plain sum", "token": "per token", "byte": "per byte", "alone": "against the option on its own",
                         "title": "Right answers of 500", "note": "Guessing gets about 125.",
                         "alt": "Right answers of 500 for the four rules: the plain sum 138, per token 151, per byte 156, against the option on its own 220; guessing gets about 125."}),
        mo.md(r"""
        ### New words

        | Word | What it means | In code |
        |---|---|---|
        | **log-likelihood scoring** | testing a model on a multiple-choice question by the log-probability it gives each option after the question, and taking the highest | `option_logprob(model, context, option)` |

        ### Recipe

        To test a base model on a multiple-choice question:

        1. Cut the context and every option into the model's tokens.
        2. Score every option: the sum of the log-probabilities of its tokens after the context.
        3. Score every option on its own too, after a neutral start.
        4. Pick the option whose log-probability rises most; or, more simply, the highest per token.
        5. Count the right answers over many questions, and compare them with guessing.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 for one question with three options. The log-probabilities of their tokens are made up.
    """)
    return


@app.cell
def example_rules():
    _after_context = {"short": [-2.0, -3.5], "right": [-1.5, -1.0, -2.0, -1.5, -0.5], "odd": [-4.0, -6.0, -3.0]}
    _on_its_own = {"short": [-2.5, -3.5], "right": [-3.0, -2.5, -3.0, -2.0, -1.5], "odd": [-4.5, -6.5, -3.0]}
    for _name in _after_context:
        _with = sum(_after_context[_name])
        _alone = sum(_on_its_own[_name])
        print(_name, _with, round(_with / len(_after_context[_name]), 2), _with - _alone)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The plain sum picks "short", −5.5 against −6.5: two tokens cost less than five. Per token, "right" wins, −1.3 against −2.75. Against the option on its own, "right" wins too, and by more: the context raises it by 5.5, "short" by only 0.5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logprob():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The log-probability of an option

        Complete `option_logprob(model, context, option)`: step 2. `context` and `option` are lists of token ids. The function puts them one after the other, lets the model read all but the last id, and turns the logits into log-probabilities: `torch.log_softmax(logits, dim=-1)` is the log of the softmax of every row in one step, the row minus its log-sum-exp, so that no tiny probability rounds to 0 first. Add, for every token of the option, its log-probability in the row of the token before it. The check tries your function on a small model, with `option_logprob(model, [3, 1, 4], [1, 5])`, and more.
        """),
        L.task("Make `option_logprob` give the log-probability of `option` after `context`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logprob():
    box_logprob = L.code_box("option-logprob", lines=12)
    box_logprob
    return (box_logprob,)


@app.cell(hide_code=True)
def check_logprob(box_logprob):
    result_logprob = L.check("option-logprob", box_logprob.value)
    result_logprob
    return (result_logprob,)


@app.cell(hide_code=True)
def hints_logprob():
    L.hints(
        "option-logprob",
        [
            "Row `j` of `logprobs` is the prediction after token `j` of `ids`. `option[0]` stands at place `len(context)`, so the model predicts it in row `len(context) - 1`.",
            "`logprobs[row, token]` is one number in a tensor; `.item()` makes it a Python number to add to `total`.",
        ],
        why="The probability of the whole option is the product of the probabilities of its tokens, each after all tokens before it: the sum of their logs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The option the context points to

        Complete `best_option(with_context, alone)`: step 4. `with_context[i]` is the log-probability of option `i` after the context, `alone[i]` the same after a neutral start. Give back the index of the option whose log-probability rises most, the first one when two rise equally. The check calls `best_option([-12.5, -9.0, -20.1, -15.0], [-14.0, -8.5, -26.0, -16.2])`, and others.
        """),
        L.task("Make `best_option` give the index of the option the context raises most."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-option", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-option", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-option",
        [
            "The rise of option `i` is `with_context[i] - alone[i]`. In the example, option 1 has the highest log-probability after the context, but it is even a little more likely on its own.",
            "Keep the index of the best option so far, starting at 0, and replace it when an option rises more.",
        ],
        why="An option that is likely anyway gets no credit for that; only what the context adds counts.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Your model on 500 questions

        The 500 questions of `data/story-cloze.json` that [Keeping test data out of training data](/course/open/llm-decontamination?lang=en) made from stories of Cosmopedia: three sentences of a story, its real next sentence, and three sentences from other stories, in a shuffled order. No model of the course trained on these stories. The button scores every option with your `option_logprob` on your model of [Capstone: pretrain your own small LLM](/course/open/llm-pretrain-capstone?lang=en), or on the course's model when you have none, after the context and after an end-of-text alone, and counts the right answers of the four rules; the last uses your `best_option`. On the course's computer, with one thread, that took about a minute for the course's run of the capstone, and about two minutes for the course's model, which is five times as large.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def score_button_cell():
    score_button = mo.ui.run_button(label="Score the 500 questions")
    score_button
    return (score_button,)


@app.cell(hide_code=True)
def demo_score(box_best, box_logprob, result_best, result_logprob, score_button):
    scored = None
    _note = None
    if not (result_logprob.state == "correct" and result_best.state == "correct"):   # no mo.stop: the table shows the course's run
        _note = "*Finish exercises 1 and 2 first: the demo scores with your `option_logprob` and picks with your `best_option`.*"
    elif score_button.value:
        _r = C.score_all(run_code(box_logprob.value).names["option_logprob"], run_code(box_best.value).names["best_option"],
                         lambda _p: mo.output.replace(mo.md(f"**Scoring** · {_p['done']} of {_p['total']} questions · {round(_p['seconds'])} s")))
        if _r == "run":
            _note = "*Scoring stops with an error. Do your modules in `work/projects/llm/` still pass their checks? Delete a file you changed, and check its exercise again.*"
        elif _r == "learner":
            _note = "*Your `option_logprob` or `best_option` stops with an error on this model. Does `option_logprob` work for a context of dozens of tokens and an option of a whole sentence?*"
        else:
            scored = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()   # also when scoring is done: no stale progress line
    return (scored,)


@app.cell(hide_code=True)
def demo_table(scored):
    _m = C.MEASURED["yours"]
    _r = scored or _m
    _names = {"sum": "the plain sum", "token": "per token", "byte": "per byte", "alone": "against the option on its own"}
    _rows = "\n".join(f"| {_names[_rule]} | {_r['right'][_rule]} | {C.number(100 * _r['right'][_rule] / _r['total'], 0, L.lang)}% |" for _rule in C.RULES)
    _whose = ("Your model" if scored["source"] == "yours" else "The course's model, on Cosmopedia") if scored else "The course's run of the capstone, on TinyStories, until you score yours"
    mo.md(f"{_whose}: right answers of {_r['total']} questions; guessing gets about 25%.\n\n| rule | right | share |\n|---|---|---|\n{_rows}")
    return


@app.cell(hide_code=True)
def text_exercise_short():
    mo.md(r"""
    ### Exercise 3. Why short options win
    """)
    return


@app.cell(hide_code=True)
def choice_short_cell():
    choice_short = L.choice(
        "why-short",
        question="With the plain sum of log-probabilities, a model often picks a short option, even when it fits the story badly. Why?",
        options={
            "data": "Its training stories were short, so it learned to prefer short sentences to long ones",
            "right": "Every token adds a negative log-probability, so fewer tokens give a higher sum",
            "right-answers": "In questions like these, the short options are more often the right answer",
        },
        correct="right",
        why={
            "data": "Even a model trained on long texts tends to give a short sentence a higher sum: each extra token can only lower it. Per token or against the option on its own, that bias goes away.",
            "right": "A probability is at most 1, so its log is at most 0: every token lowers the sum. Dividing by the length, or subtracting the score of the option on its own, takes that away.",
            "right-answers": "In these 500 questions the right sentences are as long as the others, 87 characters on average. The bias is in the sum, not in the answers.",
        },
    )
    choice_short
    return (choice_short,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The log-likelihood of [How good is a language model?](/course/open/lm-nll?lang=en), now of whole answers; the scoring of [Evaluating your base model](/course/open/llm-eval-base?lang=en), now on questions; and the questions of [Keeping test data out of training data](/course/open/llm-decontamination?lang=en), made from stories no model of the course trained on, so that the model has not seen the right answer before. Later lessons teach your model to answer instructions itself, and test that with checks a program can verify.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_logprob, result_best, choice_short):
    L.wrap_up(result_logprob, result_best, choice_short)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
