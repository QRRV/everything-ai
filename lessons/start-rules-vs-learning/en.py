# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-rules-vs-learning. Route: course/route/lessen/stam-1-basis.yaml. Checks and mornings: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-rules-vs-learning", "en")
    C = L.checks   # the mornings, and how a rule's mistakes are counted

    def line(mornings, t=None, drag=False):
        """The mornings on a line; with a rule, the mistakes get a ring. With drag, a play block sets the rule
        where you drag along the line. The title says what the picture shows, for a screen reader."""
        def listed(temps):
            return ", ".join(str(d) for d in temps[:-1]) + " and " + str(temps[-1]) if len(temps) > 1 else str(temps[0])

        title = (f"Twelve mornings on a line from 0 to 26 °C: with a coat at {listed([d for d, c in sorted(mornings) if c])} °C, "
                 f"without a coat at {listed([d for d, c in sorted(mornings) if not c])} °C.")
        if t is not None:
            wrong = [d for d, c in sorted(mornings) if (d < t) != c]
            title += (f" The rule coat when colder than {t} °C makes {count(len(wrong))}"
                      + (f", with a ring at {listed(wrong)} °C." if wrong else "."))
        return draw.threshold_line(
            mornings, t, lo=0, hi=26, axis="temperature in the morning (°C)", yes="coat", no="no coat",
            rule=f"coat when colder than {t} °C" if t is not None else "", wrong="mistake", title=title, drag=drag)

    def count(n):
        """A number of mistakes, in words."""
        return f"{n} mistake" if n == 1 else f"{n} mistakes"

    def bars(mornings):
        """How many mistakes each rule makes."""
        return draw.mistakes_per_rule(
            {t: C.mistakes(mornings, t) for t in C.RULES}, axis="rule: coat when colder than … °C",
            value="mistakes", best="fewest mistakes")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about the lesson before this one.
    """)
    return


@app.cell(hide_code=True)
def quick_check_spam():
    quick_spam = L.choice(
        "quick-spam",
        question="A spam filter learned from many emails which ones are spam. There is no neural network in it. What is the smallest circle it fits in?",
        options={"ai": "AI", "ml": "Machine learning", "dl": "Deep learning"},
        correct="ml",
        why={
            "ai": "It fits in AI, but there is a smaller circle: it learned from examples, so it is machine learning. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
            "ml": "It learned from examples, so it is machine learning, and without a neural network it is not deep learning.",
            "dl": "Deep learning needs a neural network with many layers, and this filter has none. See [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en).",
        },
    )
    quick_spam
    return (quick_spam,)


@app.cell(hide_code=True)
def quick_check_genai():
    quick_genai = L.choice(
        "quick-genai",
        question="Which of these is generative AI?",
        options={
            "poem": "An app that writes a new poem from three words you give it.",
            "birds": "An app that tells you which bird you hear.",
            "route": "A route planner with rules that people wrote.",
        },
        correct="poem",
        why={
            "poem": "It makes something new, a poem, so it is generative AI.",
            "birds": "It names a bird that is already singing; it makes nothing new. So it is not generative AI.",
            "route": "People wrote its rules, so it is AI, but not machine learning, and so not generative AI either.",
        },
    )
    quick_genai
    return (quick_genai,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You want a computer to decide something for you: whether an email is spam, whether a photo shows a cat. One way is to write the rules yourself. That works for simple things, but for spam you would need thousands of rules, and spammers change their emails every week. How else can a computer get a rule?

    **An everyday example.** You do not teach a child what a dog is by giving it rules: four legs, a tail and fur fit a cat too. You point at dogs and say "dog", and at cats and say "cat". After enough examples, the child finds the difference by itself. Giving examples is easy; writing the rule is hard.
    """)
    return


@app.cell(hide_code=True)
def predict_can_find():
    guess_can_find = L.guess(
        "can-find",
        question="You give a computer twelve mornings: the temperature, and whether you wore a coat. Do you think it can find a rule for \"coat or no coat\" without you telling it any rule?",
        options={
            "yes": "Yes, from the examples alone.",
            "no": "No, someone has to give it the rule.",
            "understand": "Only if it understands what cold is.",
        },
        reveal="**Yes.** In this lesson you see how: the computer tries rules and keeps the one that fits the examples best. It does not know what a coat is, or what cold is. It only sees numbers and answers.",
    )
    guess_can_find
    return (guess_can_find,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **example** | one case in the data; here one morning, with its temperature and whether you wore a coat |
    | **rule** | what gives the answer for a case; here "coat when it is colder than 14 °C" |
    | **learning from data** | instead of a person writing the rule, the computer finds it in examples; this is the idea behind all of machine learning |

    ### Your twelve mornings

    For twelve mornings you wrote down the temperature and whether you wore a coat. A filled dot is a morning with a coat, an open dot one without.
    """)
    return


@app.cell(hide_code=True)
def picture_mine():
    line(C.MINE)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    A rule of the kind "coat when it is colder than T °C" splits the line in two: a coat to the left, no coat to the right. Every morning on the wrong side is a **mistake**, and gets a ring. Move the slider, or drag along the line, to set T, and watch the rings.
    """)
    return


@app.cell(hide_code=True)
def explore_mine_cell():
    def _frame(t):
        return (line(C.MINE, t, drag=True).text,
                f"""This rule makes <span class="c-loss">{count(C.mistakes(C.MINE, t))}</span> on your mornings.""")

    explore_mine = L.explore("mine", "T", _frame, start=20, low=0, high=26, step=1, role="param", task="Find a rule that makes no mistakes on your mornings.")
    explore_mine
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    With T at 14 or at 15 °C, the rule makes no mistakes on your mornings. That was you writing the rule. Now the computer. It follows a recipe, and it needs nothing but the examples.

    ### Recipe

    1. Take a rule: "coat when it is colder than T °C", starting at T = 0.
    2. Count the rule's mistakes on the examples.
    3. Do the same for every T, up to 26.
    4. Keep the rule with the fewest mistakes.

    This is a simplified way of learning. It works here because there are only 27 rules to try. Later in the course you see how a computer improves a rule step by step when there are far too many rules to try them all.
    """)
    return


@app.cell(hide_code=True)
def worked_example():
    mo.md(rf"""
        ### Worked example

        The computer follows the recipe on your twelve mornings.

        1. **T = 0**: "coat when colder than 0 °C". No morning was that cold, so the rule says "no coat" every time. You wore a coat on 6 mornings: **{count(C.mistakes(C.MINE, 0))}**.
        2. **T = 10**: the rule says "coat" for 4, 7 and 9 °C, and "no coat" for the rest. Wrong at 10, 12 and 13 °C: **{count(C.mistakes(C.MINE, 10))}**.
        3. **Every other T**: the computer counts the same way. The bars show all 27 rules:
        """)
    return


@app.cell(hide_code=True)
def worked_example_bars():
    bars(C.MINE)
    return


@app.cell(hide_code=True)
def worked_example_end():
    _found = C.learn(C.MINE)
    mo.md(rf"""
        4. **Fewest mistakes**: the rules for **{_found[0]} °C** and **{_found[-1]} °C** both make {count(C.mistakes(C.MINE, _found[0]))}. The computer keeps the first one: coat when it is colder than {_found[0]} °C.
        """)
    return


@app.cell(hide_code=True)
def text_exercise_neighbour():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Your neighbor's mornings

    Your neighbor also wrote down twelve mornings, with the temperature and whether they wore a coat. Now you do what the computer does, on **their** mornings. The slider starts at your own rule, 14 °C.
    """)
    return


@app.cell(hide_code=True)
def exercise_neighbour():
    explore_neighbour = L.explore("neighbour", "T", lambda t: line(C.NEIGHBOUR, t, drag=True).text, start=14, low=0, high=26, step=1,
                                  role="param", task="Move the slider to the rule that makes the fewest mistakes on your neighbor's mornings.")
    explore_neighbour
    return (explore_neighbour,)


@app.cell(hide_code=True)
def check_neighbour(explore_neighbour):
    result_neighbour = L.check("neighbour", explore_neighbour)
    result_neighbour
    return (result_neighbour,)


@app.cell(hide_code=True)
def hints_neighbour():
    L.hints(
        "neighbour",
        [
            "Move the slider one degree at a time and count the rings each time.",
            "Your neighbor wore a coat at 17 and 18 °C. A rule that gets those right has to go above 18.",
            "Compare 19 and 20 °C. Which one makes fewer mistakes?",
        ],
        why="**19 °C**: coat when it is colder than 19 °C. Only the morning of 16 °C is wrong: your neighbor wore no coat that day. No rule of this kind gets that morning right without getting other mornings wrong.",
    )
    return


@app.cell(hide_code=True)
def text_learn():
    mo.md(r"""
    ### Let the computer do it

    You just did by hand what the computer does. Press the button to let the computer follow the recipe on your neighbor's mornings.
    """)
    return


@app.cell(hide_code=True)
def learn_button_cell():
    learn_button = mo.ui.run_button(label="Learn from the examples")
    learn_button
    return (learn_button,)


@app.cell(hide_code=True)
def learned_neighbour(learn_button):
    mo.stop(not learn_button.value)
    _found = C.learn(C.NEIGHBOUR)
    mo.vstack([
        bars(C.NEIGHBOUR),
        line(C.NEIGHBOUR, _found[0]),
        mo.md(f"The computer keeps **coat when it is colder than {_found[0]} °C**, with {count(C.mistakes(C.NEIGHBOUR, _found[0]))}. Same recipe as before, other examples, other rule."),
    ])
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### Exercise 2. What changed?
    """)
    return


@app.cell(hide_code=True)
def choice_changed_cell():
    choice_changed = L.choice(
        "what-changed",
        question="Your rule is 14 °C, your neighbor's is 19 °C. What changed to get your neighbor's rule?",
        options={
            "examples": "Only the examples. The recipe stayed the same.",
            "recipe": "The recipe: the computer needed a new way to learn.",
            "understand": "The computer understood that your neighbor gets cold more easily.",
        },
        correct="examples",
        why={
            "examples": "The computer followed the same recipe on other examples, and found another rule. That is the strength of learning from data: for a new situation you only need new examples.",
            "recipe": "The recipe was exactly the same: try every rule, count the mistakes, keep the fewest. Only the examples were different.",
            "understand": "The computer understands nothing about cold or coats. It only counted mistakes on other examples, and those happened to give another rule.",
        },
    )
    choice_changed
    return (choice_changed,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Exercise 3. Why learn?
    """)
    return


@app.cell(hide_code=True)
def explain_why_learn():
    explain_why = L.explain(
        "why-learn",
        question="A spam filter decides for every email whether it is spam. Why would you rather let it learn from examples than write its rules yourself? Write one or two sentences.",
        model_answer="Spam comes in far too many forms to catch with rules you write yourself, and spammers keep changing their emails. To learn, you only need emails marked spam or not spam, and when spam changes, the filter learns again from new examples.",
        points=[
            "spam has too many forms to write rules for",
            "spam keeps changing, and with new examples the filter learns again",
            "for learning you only need emails marked spam or not spam",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI, machine learning, deep learning and generative AI](/course/open/start-ai-ml-genai?lang=en) the first question of the recipe was: did people write the rules, or did the computer find them in examples? Now you have seen the second way at work. Counting mistakes is the first form of a number you will meet again and again: a number that says how bad a rule is. Here the computer learned one number, T. The models you build later in the course learn thousands to millions of numbers, but the idea stays the same: learn the rule from examples, and measure how good it is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_changed, explain_why, result_neighbour):
    L.wrap_up(result_neighbour, choice_changed, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
