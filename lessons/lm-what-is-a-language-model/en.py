# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-what-is-a-language-model. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-what-is-a-language-model", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Conditional probability](/course/open/prob-conditional?lang=en) and [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_given():
    quick_given = L.choice(
        "quick-given",
        question="In a long text, the letter q appears 20 times, and 19 of those times a u follows it. What is P(u next | q)?",
        options={
            "count": "19",
            "right": "0.95",
            "rest": "0.05",
        },
        correct="right",
        why={
            "count": "19 is how often the u follows. A probability is a share, between 0 and 1: count among the 20 cases with a q only. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "rest": "That is the probability that no u follows. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "right": "Among the 20 times there is a q, a u follows 19 times: 19 / 20 = 0.95.",
        },
    )
    quick_given
    return (quick_given,)


@app.cell(hide_code=True)
def quick_check_function():
    quick_function = L.choice(
        "quick-function",
        question="A model is a function. What does that function do?",
        options={
            "right": "It takes the features and gives back a prediction",
            "store": "It keeps every training example, to look it up later",
            "labels": "It writes the right answer next to every example",
        },
        correct="right",
        why={
            "store": "A model keeps its parameters, not the examples: it works out a prediction from the features. See [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "labels": "The labels come with the examples; the model's output is a prediction. See [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en).",
            "right": "Features in, a prediction out, as ŷ = w · x + b, with its parameters w and b.",
        },
    )
    quick_function
    return (quick_function,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A chat program writes its answer one small piece at a time, and each time the model behind it guesses what comes next. To get good at guessing, a model has to learn from far more text than people could ever label by hand: billions of words. So where do its labels come from?

    **An everyday example.** When someone says "Once upon a …", you know the next word before you hear it. Nobody taught you that as a rule: you picked it up from hearing many stories.
    """)
    return


@app.cell(hide_code=True)
def predict_examples():
    guess_examples = L.guess(
        "examples",
        question="A model learns to predict the next character of a text. How many training examples do you think the text \"the cat sat\", 11 characters long, gives?",
        options={
            "all": "11: one for each character",
            "right": "10: one for each character after the first",
            "one": "1: the whole text is one example",
        },
        reveal="**10.** Every character after the first is the label of one example, and the text before it is the input: \"t\" → \"h\", \"th\" → \"e\", \"the\" → a space, and so on up to \"the cat sa\" → \"t\". Not 11: the first character has nothing before it, so it is never a label. The text gives its own labels, so a model can learn from any text, without anyone adding labels: self-supervised learning, as in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
    )
    guess_examples
    return (guess_examples,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **token** is a piece of text that a language model reads and writes. In this lesson a token is one character; real models use bigger pieces, as the next lesson shows. Write the tokens of a text as $x_1, x_2, \dots$: $x_1$ is the first token, $x_t$ the $t$-th. Here the counting starts at 1, as usual in maths; Python counts the places from 0, so $x_t$ is `text[t - 1]`.

        A **language model** gives, for the text so far, the probability of every possible next token:
        """),
        L.formula(r"[[pred:P]](x_{t+1} \mid x_1, \dots, x_t)"),
        mo.md(r"""
        That is a conditional probability, as in [Conditional probability](/course/open/prob-conditional?lang=en): the probability of the next token, given the text so far. Together the probabilities of all possible next tokens add up to 1. And a language model is a function, as in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en): the text so far goes in, a probability for every token comes out.

        **Next-token prediction** is how a language model learns: on real text, it predicts every next token, and its prediction is compared with the token that really comes. The text gives the labels itself: a text of $n$ tokens gives $n - 1$ examples. In code, the text so far is a slice: a slice works on a text as on a list, as in [Lists](/course/open/py-lists?lang=en), so `text[:i]` is the text before place `i`, and `"my lord"[:2]` is `"my"`. To write, a model picks a next token from its probabilities, adds it to the text, and does it again.

        The course's language model, `C.next_probs(text)`, counts in Tiny Shakespeare, about 40,000 lines from plays by Shakespeare, which character comes after the last three characters of `text`; when the text is shorter, or those three never occur, it looks at fewer. It knows 65 different characters. Counting is enough to make a language model: no neural network needed. Real language models look at thousands of tokens at once and learn their probabilities with a neural network, but they give the same thing: a probability for every possible next token.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **token** | | a piece of text that a language model reads and writes; here one character | `text[i]` |
        | **language model** | | a model that gives, for the text so far, the probability of every possible next token | `C.next_probs(text)` |
        | **next-token prediction** | | learning by predicting every next token of a real text: the text gives the labels | |
        | $x_t$ | x t | the $t$-th token of a text, counting from 1 | `text[t - 1]` |
        | $P(x_{t+1} \mid x_1, \dots, x_t)$ | P of x t plus 1 given x 1 to x t | the probability of the next token, given the text so far | `C.next_probs(text[:t])` |
        """),
        L.watch_out("*Token* also means something else: a login token is a key that gives access to a service, and a crypto token is a kind of digital coin. In this course a token is always a piece of text that a language model reads; for the key, the course says *API key*."),
        mo.md(r"""
        The picture: the probabilities the course's model gives the next character after "my lo". Only the five most likely of the 65 are shown.
        """),
        draw.next_word_bars("my lo", C.top("my lo"), picked=0, picked_label="most likely",
                            title="After \"my lo\""),
        mo.md(r"""
        The model only sees the last three characters: a space, "l" and "o", the start of a word. "lord" and "love" are both common words in the plays, so "r" and "v" are almost as likely as each other. A model that only predicted "r" would be wrong a lot; the probabilities say how sure it is. A model that also saw "my" could do better: in the plays, "my lord" comes about ten times as often as "my love".
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To learn and use a language model:

    1. Split the text into tokens: here every character is a token.
    2. Make the examples: for every place `i` from 1 on, the input is the text so far, `text[:i]`, and the label is the next token, `text[i]`.
    3. Give the model the text so far: it gives a probability for every possible next token.
    4. Look up the probability of the token that really comes next: a good model gives it a high probability.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The examples of the text "my lord", and the course's model at place 5, after "my lo". The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_model():
    _text = "my lord"
    # 1. the tokens: every character, so 7 tokens
    print(len(_text))
    # 2. every place from 1 on gives an example: the text so far, and the token that comes next
    _examples = []
    for _i in range(1, len(_text)):
        _examples.append((_text[:_i], _text[_i]))
    print(len(_examples), _examples)
    # 3. the model at place 5: the text so far, "my lo", in; a probability for every possible next token out
    _probs = C.next_probs(_text[:5])
    print(len(_probs))
    # 4. the probability of the token that really comes next, _text[5], which is "r"; and of "v"
    print(_probs[_text[5]], _probs["v"])
    return


@app.cell(hide_code=True)
def text_after_example():
    _probs = C.next_probs("my lo")
    mo.md(f"""
    Seven tokens give six examples, from ("m", "y") to ("my lor", "d"). After "my lo", the model gives all 65 characters a probability; the "r" that really comes next gets {L.number(_probs["r"], 2)}, and "v" gets {L.number(_probs["v"], 2)}. In training, a neural network would change its parameters so that the real next token gets a higher probability, example after example.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_addup():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. All the probabilities together
    """)
    return


@app.cell(hide_code=True)
def choice_addup_cell():
    choice_addup = L.choice(
        "add-up",
        question="After \"my lo\", the course's model gives each of the 65 characters it knows a probability. What do those 65 probabilities add up to?",
        options={
            "right": "1: exactly one of them comes next",
            "many": "65: one for each character",
            "depends": "It depends on the text so far",
        },
        correct="right",
        why={
            "many": "Each probability is a share between 0 and 1, and together they share out all the cases. See *The idea*.",
            "depends": "The probabilities themselves depend on the text, but not their sum. See *The idea*.",
            "right": "Exactly one character comes next, so the probabilities share out all the cases: a probability distribution over the 65 characters.",
        },
    )
    choice_addup
    return (choice_addup,)


@app.cell(hide_code=True)
def text_exercise_examples():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The examples in a text

        `examples(text)` gives the training examples of a text for next-token prediction, each as a pair: the text so far, and the token that comes next. Replace the three `...` with step 2 of the recipe. The check calls `examples("my lord")`, and that should give the six examples of the worked example. It also tries other texts, short ones too.
        """),
        L.task("Make `examples` give, for every place from 1 on, the text so far and the token that comes next."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("examples", lines=7)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    L.hints(
        "examples",
        [
            "Step 2 of the recipe: which places `i`, and what is the input and the label at each place?",
            "The places run from 1 up to the last place of the text, `len(text) - 1`. The input stops just before place `i`; the label is the token at place `i`.",
            "`range(1, len(___))`, and the pair `(text[:___], text[___])`.",
        ],
        why="For \"my lord\", `i` goes from 1 to 6. At `i` = 1 the pair is (\"m\", \"y\"), at `i` = 6 it is (\"my lor\", \"d\"): six examples, one for every token after the first.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_real():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How sure is the model?

        For every place in the text "to be", find the probability the course's model gives the token that really comes next. In the box, the model is called `next_probs`, as `C.next_probs` in the worked example. Replace the two `...` with steps 3 and 4 of the recipe. The last line shows the list `real`, with one probability for each of the four examples. The check also tries your code on other texts.
        """),
        L.task("Fill `real` with the probability the model gives each token that really comes next."),
    ])
    return


@app.cell(hide_code=True)
def exercise_real():
    box_real = L.code_box("real-probs", lines=7)
    box_real
    return (box_real,)


@app.cell(hide_code=True)
def check_real(box_real):
    result_real = L.check("real-probs", box_real.value)
    result_real
    return (result_real,)


@app.cell(hide_code=True)
def hints_real():
    L.hints(
        "real-probs",
        [
            "Steps 3 and 4 of the recipe, at every place `i`: the model gets the text so far, and you look up one probability in what it gives back.",
            "`next_probs` gets the text so far, as in step 2 of the recipe. It gives back a dictionary with a probability for every character: look up the token that really comes next as a key, as `_probs[\"r\"]` in the worked example.",
            "`probs = next_probs(text[:___])` and `real.append(probs[text[___]])`.",
        ],
        why="At `i` = 1 the model gets \"t\" and the real next token is \"o\"; at `i` = 4 it gets \"to b\" and the real next token is \"e\", which it gives a probability of about 0.66.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A language model is a function, as in [A model is a function with parameters](/course/open/ml-model-is-a-function?lang=en), that gives a conditional probability, as in [Conditional probability](/course/open/prob-conditional?lang=en), for every possible next token. It learns from text alone, because the text gives its own labels: self-supervised learning, as in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en). Here a token was a character, as in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en). In the next lesson, *Tokens: how real LLMs chop up text*, you see the pieces real language models use; after that, in *A bigram model by counting*, you build a counting model of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_addup, result_examples, result_real):
    L.wrap_up(choice_addup, result_examples, result_real)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
