# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-kinds-of-learning. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-kinds-of-learning", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Examples, features and labels](/course/open/ml-vocabulary?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="A table of houses has the columns area, rooms, garden and price. You want to predict the price. What is the label?",
        options={
            "area": "The area",
            "all": "All four columns",
            "right": "The price",
        },
        correct="right",
        why={
            "area": "The area is something you know beforehand. What do you want to predict? See *The idea* in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "all": "Some columns are what the model gets, one is what it predicts. See *The idea* in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "right": "The label is the column you want to predict: the right answer. Area, rooms and garden are the features.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_inference():
    quick_inference = L.choice(
        "quick-inference",
        question="A trained model predicts the price of a house that is for sale. What is that called?",
        options={
            "right": "Inference",
            "training": "Training",
            "label": "Labeling",
        },
        correct="right",
        why={
            "training": "Training is learning from examples whose label is known. Is the price of this house known? See the table in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "label": "The model does not write a right answer; it predicts one. See the table in [Examples, features and labels](/course/open/ml-vocabulary?lang=en).",
            "right": "Using a trained model on a new example, whose label is not known yet, is inference.",
        },
    )
    quick_inference
    return (quick_inference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far, every example to learn from had a label: a right answer someone wrote down, such as the price a house sold for. But a large language model learned most of what it knows from billions of pages of text, and nobody labeled them. A robot that learns to walk has no table of right answers either. The kinds of learning differ in one thing: where the right answers come from.

    **An everyday example.** You can learn a language from a teacher who corrects every sentence, by reading books and noticing which words come after which, or by trying to order food abroad and seeing whether you get what you wanted.
    """)
    return


@app.cell(hide_code=True)
def predict_labels():
    guess_labels = L.guess(
        "labels",
        question="A large language model learned from billions of pages of text. Where do you think its labels came from?",
        options={
            "people": "People wrote them, one by one",
            "text": "From the text itself: the next word is the label",
            "none": "It needs no labels at all",
        },
        reveal="**From the text itself.** Take any piece of text, hide the word that comes next, and that word is the label. Every page gives hundreds of examples, and nobody has to write a single label. This is self-supervised learning.",
    )
    guess_labels
    return (guess_labels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        There are four main kinds of learning, and they differ in where the right answers come from.

        - In **supervised learning**, every example has a label that someone gave it, and the model learns to predict the label from the features: houses with their prices.
        - In **unsupervised learning**, there are no labels at all. The model looks for structure in the examples, such as groups of customers who buy alike.
        - In **self-supervised learning**, the labels come from the data itself: hide a part, such as the next word, and predict it. Language models learn this way.
        - In **reinforcement learning**, there are no examples with answers, but actions and rewards: the model tries something, gets a reward or not, and learns which actions pay off, as in a game.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **supervised learning** | learning from examples that have a label |
        | **unsupervised learning** | learning from examples without labels, to find structure in them |
        | **self-supervised learning** | learning from data without labels by making the labels from the data itself, such as the next word |
        | **reinforcement learning** | learning by trying actions and getting rewards for them |
        """),
        L.watch_out("Self-supervised learning is not unsupervised learning: there are labels, and the model learns to predict them. But nobody wrote them: the data gives them. Some sources still call it unsupervised, because nobody labels anything by hand; this course keeps the two apart."),
    ])
    return


@app.cell(hide_code=True)
def picture_kinds():
    mo.vstack([
        draw.data_tables([
            ("Supervised", ["area", "price"], [["62", "245"], ["85", "310"], ["110", "395"]], {(0, 1), (1, 1), (2, 1)}, ""),
            ("Unsupervised", ["bread", "milk"], [["2", "0"], ["0", "3"], ["2", "1"]], set(), ""),
            ("Self-supervised", ["input", "label"], [["the", "cat"], ["cat", "sat"], ["sat", "down"]], {(0, 1), (1, 1), (2, 1)}, ""),
            ("Reinforcement", ["action", "reward"], [["left", "0"], ["up", "0"], ["up", "1"]], {(0, 1), (1, 1), (2, 1)}, ""),
        ], title="Where the answers come from",
            alt="Four small tables, one for each kind of learning. Supervised: houses with their area and their price; the price column is marked. Unsupervised: customers with how much bread and milk they bought; nothing is marked. Self-supervised: pairs from the text the cat sat down, an input word and the next word as its label; the label column is marked. Reinforcement: actions, left, up and up, each with a reward of 0, 0 and 1; the reward column is marked."),
        mo.md(r"""
        Four small tables of examples. A thick border marks the right answers; in reinforcement learning, the rewards take their place. Supervised: the prices were written down when the houses were sold. Unsupervised: only what each customer bought, and no column with answers. Self-supervised: the label of each row is the input of the next row, so the labels come from the text *the cat sat down* itself. Reinforcement: no right answers, only a reward after each action.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tell which kind of learning a task is, ask where the right answers come from:

    1. Did people give each example its label? Supervised learning.
    2. Does the data give the label itself, such as the next word? Self-supervised learning.
    3. Are there no answers at all, only examples to find structure in? Unsupervised learning.
    4. Does the model act, and get rewards for what it does? Reinforcement learning.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four tasks through the recipe. Ask the questions in order: the first *yes* says the kind.

    **A photo app** learns to tell cats from dogs, from 50,000 photos that people tagged "cat" or "dog". Step 1, did people give each example its label? Yes: every photo has the tag people gave it. **Supervised learning.**

    **A phone keyboard** learns to suggest your next word, from messages that people typed. Step 1: no, people typed the messages, but nobody added a label to them. Step 2, does the data give the label itself? Yes: after each word in a message, the next word is already there. **Self-supervised learning.**

    **A music app** puts songs that sound alike in groups, and nobody says which groups there are. Steps 1 and 2: no labels from people, and nothing hidden to predict. Step 3, no answers at all, only examples to find structure in? Yes: only songs, to sort into groups. **Unsupervised learning.**

    **A program learns to park a car** in a simulator: it steers, and gets points when the car ends up in the parking space. Steps 1 to 3: no examples with answers, and no groups to find. Step 4, does it act, and get rewards for what it does? Yes: it steers, and those points are the rewards. **Reinforcement learning.**

    **Step 2 in code.** How does the data give the labels? Training pairs made from the letters of a word, each letter with the letter after it as its label. The comment with `#` says which step of the recipe the code belongs to. Run the cell.
    """)
    return


@app.cell
def example_pairs():
    _text = "hello"
    # 2. the data gives the labels: each letter, with the next letter as its label
    _pairs = []
    for _i in range(len(_text) - 1):
        _pairs.append((_text[_i], _text[_i + 1]))
    print(_pairs)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Five letters give four pairs: the last letter has no letter after it. Each pair is an example, the first letter its input and the second its label, and nobody wrote a label. The first language model you build in this course learns from pairs just like these; a large language model looks at all the words before, not only the last one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spam():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A spam filter
    """)
    return


@app.cell(hide_code=True)
def choice_spam_cell():
    choice_spam = L.choice(
        "spam-kind",
        question="A spam filter learns from 10,000 mails that people marked as spam or not spam. Which kind of learning is that?",
        options={
            "right": "Supervised learning",
            "unsupervised": "Unsupervised learning",
            "self": "Self-supervised learning",
            "reinforcement": "Reinforcement learning",
        },
        correct="right",
        why={
            "unsupervised": "There are answers here: people marked every mail. See step 1 of the recipe.",
            "self": "The marks do not come from the mails themselves: people added them. See steps 1 and 2 of the recipe.",
            "reinforcement": "The filter does not act and get rewards; it learns from examples with answers. See the recipe.",
            "right": "Every mail has a label that people gave it: spam or not spam.",
        },
    )
    choice_spam
    return (choice_spam,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Five tasks, four kinds

        Each task below gets one kind of learning. Use the recipe, as in the worked example: the questions in order.
        """),
        L.task("Put each task with its kind of learning."),
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
            "Take one task at a time through the recipe, as in the worked example: the first *yes* says the kind.",
            "Step 1 for each task: who gave the right answers? Then step 2: does any task hide a part of its own data and predict it?",
            "The fraud check and the translation have answers from people. The photo puzzle predicts a covered square of its own photos. Which task looks for groups, and which one gets points?",
        ],
        why="**Supervised:** the fraud check and the translation; people marked each payment and wrote each translation. **Self-supervised:** the photo puzzle; the covered square is the label, and it comes from the photo itself. **Unsupervised:** the news topics; there is no list of subjects, only articles to put in groups. **Reinforcement:** chess; the program plays, and winning or losing gives its reward.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Labels from text

        Finish `word_pairs(words)`. It gets a list of words and gives back a list of training pairs: each word that has a word after it, with that next word as its label. Replace the `...`. The check calls `word_pairs(["the", "cat", "sat", "down"])`: that should give `[("the", "cat"), ("cat", "sat"), ("sat", "down")]`. It also tries other lists, including one with a single word and an empty one.
        """),
        L.task("Make `word_pairs` give back each word with the next word as its label."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("word-pairs", lines=7)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("word-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "word-pairs",
        [
            "Look at the worked example: the same, with words instead of letters.",
            "A loop over the places `i` of the words. Which places have a word after them, and what is that word?",
            "`for i in range(len(words) - ___):` and inside it `pairs.append((words[i], words[___]))`.",
        ],
        why="`range(len(words) - 1)` goes over every word that has a word after it, and `(words[i], words[i + 1])` pairs it with that next word. Four words give three pairs.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Examples, features and labels](/course/open/ml-vocabulary?lang=en), every example to learn from had a label; now you know that the label can come from people, from the data itself, or not exist at all, and that a model can also learn from rewards. The language model you build later in the course learns self-supervised, from pairs like the ones in exercise 3. Reinforcement learning comes back later in the main route, in *Learning from rewards*. In the next lesson, *Classification: predicting a class*, you look at supervised learning whose label is a class instead of a number.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spam, result_kinds, result_pairs):
    L.wrap_up(choice_spam, result_kinds, result_pairs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
