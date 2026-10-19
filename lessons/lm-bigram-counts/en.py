# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-bigram-counts. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-counts", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) and [Project: which letters does a book use?](/course/open/py-project-letter-counts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gives():
    quick_gives = L.choice(
        "quick-gives",
        question="What does a language model give back for the text so far?",
        options={
            "one": "The one next token that is right",
            "right": "A probability for every possible next token",
            "label": "A label that says what the text is about",
        },
        correct="right",
        why={
            "one": "A model does not know which token is right; it says how likely each one is. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "label": "A language model predicts what comes next, not what a text is about. See [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en).",
            "right": "A probability for every token it knows, and together they add up to 1.",
        },
    )
    quick_gives
    return (quick_gives,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="What does your `count_pairs(\"Hi, ho\")` give?",
        options={
            "all": "`{\"hi\": 1, \"i,\": 1, \", \": 1, \" h\": 1, \"ho\": 1}`",
            "capital": "`{\"Hi\": 1, \"ho\": 1}`",
            "right": "`{\"hi\": 1, \"ho\": 1}`",
        },
        correct="right",
        why={
            "all": "`count_pairs` counts a pair only when both characters are letters. See [Project: which letters does a book use?](/course/open/py-project-letter-counts?lang=en).",
            "capital": "`count_pairs` first puts the text in small letters, with `text.lower()`. See [Project: which letters does a book use?](/course/open/py-project-letter-counts?lang=en).",
            "right": "The text becomes \"hi, ho\", and only \"hi\" and \"ho\" are two letters next to each other.",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your project `llm` has `count_pairs`: it counts which letter follows which. That is almost a language model, as in [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en): from the pairs that start with a t, you can say what is likely to come after a t. Almost, because it skips spaces and signs, and makes capitals small, so a model made from it could never end a word or start a sentence. Today you count every pair of characters in the plays of Shakespeare, and turn the counts into probabilities: the second step of your own LLM.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           1, here_label="← today", title="Where your project stands: your own LLM"),
        mo.md(r"""
        **An everyday example.** A rule of thumb for the weather: after a dry day, the next day is mostly dry too. You know that from counting days, not from understanding the weather, and you only look at yesterday.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="The plays of Shakespeare in the course use 65 different characters, so there are 65 · 65 = 4,225 possible pairs of characters. How many of them do you think occur at least once in their 1.1 million characters?",
        options={
            "all": "Nearly all 4,225: in so much text, every pair comes by",
            "few": "About 100",
            "right": "About a third of them",
        },
        reveal=f"**About a third: {C.grouped(C.distinct(), L.lang)}.** Most pairs never occur: no word has \"qz\" or \"Xk\", and after a \"q\" comes a \"u\" every single time, all {C.grouped(C.after_q(), L.lang)} times. A model made from these counts gives every pair it never saw a probability of 0.",
    )
    guess_pairs
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **bigram** is a pair of tokens next to each other; here, two characters, such as "th". Count every bigram of a text, as `count_pairs` did, but now every character counts: spaces, capitals, signs and newlines too. Then the counts give the probability that $b$ comes after $a$: of all the pairs that start with $a$, the share that is $ab$.
        """),
        L.formula(r"[[pred:P]](b \mid a) = \frac{[[data:\mathrm{count}(ab)]]}{[[data:\text{how many pairs start with } a]]}"),
        mo.md(r"""
        That is a **bigram model**: a language model that only looks at the last token. The text so far can be long, but for the next character it only uses the last one. That simplification is the **Markov assumption**:
        """),
        L.formula(r"P(x_{t+1} \mid x_1, \dots, x_t) \approx P(x_{t+1} \mid x_t)"),
        mo.md(r"""
        An **n-gram** is a run of $n$ tokens in a row. An n-gram model looks at the last $n - 1$ tokens: a bigram model is a 2-gram model. The model of [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en) looked at the last three characters, so it was a 4-gram model. More tokens back make better predictions, but also many more runs that never occur in the text.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **bigram** | | two tokens next to each other, such as "th" | `text[i:i + 2]` |
        | **n-gram** | | a run of $n$ tokens in a row; an n-gram model looks at the last $n - 1$ | |
        | **Markov assumption** | | the next token depends only on the last few tokens, not on everything before | |
        | $\mathrm{count}(ab)$ | count of a b | how often the pair $ab$ occurs in the text | `counts[a + b]` |
        | how many pairs start with $a$ | | the counts of all pairs that start with $a$, added up, whatever comes second | `total` |
        | $P(b \mid a)$ | P of b given a | the probability that $b$ comes after $a$ | `probs[b]` |
        | $\approx$ | is about | about equal to | |
        """),
        L.watch_out("A pair that never occurs in the text gets a probability of 0: the model then says it can never happen, also when it can, just not in these plays. A later lesson, *Pairs you never saw: smoothing*, fixes that."),
        mo.md(r"""
        The picture: the probabilities after a "t", from the counts of all of Shakespeare's plays in the course. Only the five most likely characters are shown; ␣ is a space.
        """),
        draw.next_word_bars("t", C.top(C.course_probs("t")), picked=0, picked_label="most likely", title="After \"t\""),
        mo.md(r"""
        After a "t" comes an "h" about a third of the time, as in "the", "that" and "thou", and a space about a quarter of the time, at the end of a word.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make a bigram model from a text, and use it after a character `a`:

    1. Count every bigram: for every place `i`, the pair `text[i:i + 2]`, one more each time it comes by.
    2. Take the last character `a` of the text so far.
    3. Add up the counts of all pairs that start with `a`: that is `total`.
    4. Divide the count of each of those pairs by `total`: the probability that its second character comes after `a`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A bigram model of the text "to be or not to be", and its probabilities after an "o". The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_bigrams():
    _text = "to be or not to be"
    # 1. count every bigram
    _counts = {}
    for _i in range(len(_text) - 1):
        _pair = _text[_i:_i + 2]
        if _pair in _counts:
            _counts[_pair] = _counts[_pair] + 1
        else:
            _counts[_pair] = 1
    print(_counts)
    # 2. the last character of the text so far
    _so_far = "to be or no"
    _a = _so_far[-1]
    # 3. how many pairs start with it
    _total = 0
    for _pair in _counts:
        if _pair[0] == _a:
            _total = _total + _counts[_pair]
    print(_total)
    # 4. each next character, with its count divided by the total
    for _pair in _counts:
        if _pair[0] == _a:
            print([_pair[1], _counts[_pair] / _total])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The text so far is "to be or no", and the model only looks at its last character, an "o". Four pairs start with "o": "o " twice, in "to be", and "or" and "ot" once each. So after an "o", the model gives a space 2 / 4 = 0.5, and "r" and "t" 0.25 each. Every other character gets 0: in this short text, nothing else ever came after an "o". Each line is printed as a list, so that the space shows between its quotes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_markov():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Only the last character
    """)
    return


@app.cell(hide_code=True)
def choice_markov_cell():
    choice_markov = L.choice(
        "markov",
        question="Your bigram model gets the text \"the k\", and later the text \"a k\". What does it give after each of them?",
        options={
            "different": "Other probabilities: \"the k\" is more often followed by an \"i\", as in \"the king\"",
            "right": "The same probabilities: the ones after \"k\"",
            "nothing": "Nothing: these texts are longer than one pair",
        },
        correct="right",
        why={
            "different": "That is true of real text, but a bigram model does not see it: it only looks at the last character. See the Markov assumption in *The idea*.",
            "nothing": "The text so far may be any length; the model uses its last character. See *The idea*.",
            "right": "Both texts end in \"k\", and a bigram model only looks at the last character: it gives the probabilities after \"k\" both times.",
        },
    )
    choice_markov
    return (choice_markov,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count every pair

        `count_bigrams(text)` counts every pair of characters next to each other, as your `count_pairs` did, but without skipping anything: spaces, capitals, signs and newlines count too. The loop and the pair are there. Replace the `...` with the counting, step 1 of the recipe. The check calls `count_bigrams("To be, or not")`, and tries other texts too. When your function is right, the check saves it in your project, as `work/projects/llm/bigrams.py`.
        """),
        L.task("Make `count_bigrams` count every pair of characters, whatever they are."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-bigrams", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-bigrams", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-bigrams",
        [
            "Step 1 of the recipe: the same counting as in your `count_pairs`, with an `if` and an `else`.",
            "When `pair` is already in `counts`, it gets 1 more; when it is not, it starts at 1. Leave out the check for letters, and leave the text as it is.",
            "`if pair in counts:`, then `counts[pair] = counts[pair] + ___`, and after `else:` the line `counts[pair] = ___`.",
        ],
        why="\"To be, or not\" has 13 characters, so 12 pairs, all different: \"To\", \"o \", \" b\", \"be\", \"e,\", \", \", \" o\", \"or\", \"r \", \" n\", \"no\" and \"ot\", each once.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. From counts to probabilities

        `bigram_probs(counts, a)` gives the probabilities after the character `a`, as a dictionary: for every character that ever comes after `a`, the probability that it does. The two loops are there. Replace the two `...` with steps 3 and 4 of the recipe. The check calls `bigram_probs({"ab": 3, "ac": 1, "ba": 2}, "a")`, which should give `{"b": 0.75, "c": 0.25}`, and tries other counts too. When your function is right, the check saves it in your project, as `work/projects/llm/bigram_model.py`.
        """),
        L.task("Make `bigram_probs` divide the count of every pair that starts with `a` by `total`: the counts of all those pairs, added up."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("bigram-probs", lines=11)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("bigram-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "bigram-probs",
        [
            "Steps 3 and 4 of the recipe. The first loop only adds up; the second divides.",
            "In the first loop, `total` grows by the count of every pair that starts with `a`. In the second loop, the key is the second character of the pair, and the value its count as a share of `total`.",
            "`total = total + counts[___]`, and `probs[pair[1]] = counts[pair] / ___`.",
        ],
        why="With `{\"ab\": 3, \"ac\": 1, \"ba\": 2}` and `\"a\"`, the pairs \"ab\" and \"ac\" start with \"a\": `total` is 3 + 1 = 4. Then \"b\" gets 3 / 4 = 0.75 and \"c\" gets 1 / 4 = 0.25; \"ba\" starts with \"b\" and does not count.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below counts every pair in the plays of Shakespeare with your own `count_bigrams`, and asks your own `bigram_probs` what comes after a "t" and after a "q". It uses your own modules, so it works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_model(result_count, result_probs):
    result_count, result_probs  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `count_bigrams` and your `bigram_probs` are in your project, and this cell uses them.*")
    elif _result == "slow":
        _out = mo.md("*Counting the plays with your `count_bigrams` in `work/projects/llm/bigrams.py` takes more than 30 seconds. Does it go once over the text?*")
    elif _result == "counts":
        _out = mo.md("*The `count_bigrams` in `work/projects/llm/bigrams.py` no longer counts the way the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.*")
    elif _result == "probs":
        _out = mo.md("*The `bigram_probs` in `work/projects/llm/bigram_model.py` no longer gives the probabilities the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*")
    else:
        _pairs, _after_t, _after_q = _result
        _out = mo.vstack([
            mo.md(f"""Your `count_bigrams` found <span class="ea-num">{C.grouped(_pairs, L.lang)}</span> different pairs in the plays. Your `bigram_probs` says what comes after a "t":"""),
            draw.next_word_bars("t", C.top(_after_t), picked=0, picked_label="most likely", title="Your model, after \"t\""),
            mo.md(f"""And after a "q", it gives a "u" a probability of <span class="ea-num">{L.number(_after_q.get("u", 0), 2)}</span>: every "q" in the plays is followed by a "u". This is your first language model. In the lessons after the next one, it writes text of its own."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The pair counts of [Project: which letters does a book use?](/course/open/py-project-letter-counts?lang=en) became a language model, as in [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en): one that only looks at the last character, by the Markov assumption. Your project `llm` now has `bigrams.py` and `bigram_model.py` as well. In the next lesson, *The bigram table as a matrix of probabilities*, you put all the counts in one table of 65 by 65, and make every row a probability distribution at once.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_markov, result_count, result_probs):
    L.wrap_up(choice_markov, result_count, result_probs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
