# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-model-card. Route: course/route/lessen/systemen.yaml. Checks, the reader, the digits, the conditions and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-card", "en")
    C = L.checks   # the digit reader, the 2,000 test digits, the conditions, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using a pretrained model](/course/open/torch-pretrained?lang=en) and [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)
    """)
    return


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="A model on a model hub reads 97 of 100 test digits right, its card says. What do you look at on the card before you use it in an app?",
        options={
            "score": "Nothing else: with 97 of 100 right, the model is good enough for almost any app you build",
            "right": "What data it learned from and was tested on, and whether its license allows your use",
            "size": "How many parameters it has, since a model with more parameters always reads better",
            "date": "When it was trained, since a model trained a while ago knows less than a newer one",
        },
        correct="right",
        why={
            "right": "A score holds for the test digits it was measured on. In Using a pretrained model, a card with white digits on black did not fit photos of digits on paper, and a license with NC did not fit an app that you sell.",
            "score": "97 of 100 holds for the card's own test digits: do your images look like those, and may you use the model in your app? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "size": "More parameters say nothing about whether a model fits your images or your use: what on the card does? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "date": "A model does not know more for being newer: what on the card says whether it fits your images and your use? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="A test set has 900 digits of group A and 100 of group B. A reader gets 95% of group A right and 75% of group B. What is its accuracy over the whole test set?",
        options={
            "half": "85%: the mean of 95% and 75%, since each of the two groups counts once",
            "a": "95%: group A is nine tenths of the test set, so only its accuracy counts",
            "right": "93%: in the mean, group A weighs nine times as much as group B",
            "b": "75%: an accuracy is only as good as that of the weakest group",
        },
        correct="right",
        why={
            "right": "0.9 × 95% + 0.1 × 75% = 93%. The overall score lies close to that of group A, and hides that one in four digits of group B goes wrong.",
            "half": "Each digit counts once, not each group: how much does a group of 100 weigh against a group of 900? See [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
            "a": "Group B is a tenth of the test set, and it counts for a tenth: what is 0.9 × 95% + 0.1 × 75%? See [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
            "b": "The overall accuracy is a mean over all digits, weighted by the size of each group: what does that give here? See [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en)",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The card of the digit reader of [Using a pretrained model](/course/open/torch-pretrained?lang=en) says: 929 of the first 1,000 test images of QMNIST right. Someone who wants to read the digits that adults write on forms reads that line and expects about 7 mistakes in 100 digits. But who wrote those 1,000 digits? The card does not say. And a model does not do as well for every writer, or under every condition.

    In 2019, Margaret Mitchell and eight others proposed the model card: a short document that comes with a trained model and says what it is for and what not, which data stand behind it, and how well it does, not only overall but per group and per condition. For a model that looks for smiles in photos, that means per age group, for instance, and in bright and in dim light. Hugging Face shows such a card on the page of a model, and makers of large models publish one with each model. In this lesson you measure the digit reader the way a card should report it, and fill in what its card leaves out.

    **An everyday example.** The leaflet in a box of medicine says what the medicine is for, who should not take it, such as children under 12, which side effects are known and how often they occur, and on whom it was tested. A leaflet with only "works for 9 in 10 people" would leave you guessing whether you are one of the nine.
    """)
    return


@app.cell(hide_code=True)
def predict_employees():
    guess_employees = L.guess(
        "employees",
        question="The 1,000 test digits behind the card's 929 were all written by American high-school students. QMNIST also has digits written by employees of the American government. Take 1,000 of those, by writers who wrote none of the reader's training digits either. How many of them do you think the reader reads right?",
        options={
            "same": "About 930: the same reader, on the same kind of image",
            "fewer": "About 880: grown-ups write faster and less neatly",
            "right": "About 960: the students' digits are the harder ones",
            "far": "About 700: the reader never saw these writers",
        },
        reveal="**About 960: 960 of 1,000.** On the students' digits the reader makes 71 mistakes, on the employees' 40: almost twice as many for the students. QMNIST's own notes say the students' digits are \"considered to be substantially more challenging\". Here the card's one number is the lower of the two, but it could just as well have been the higher one, and the card would not have told you. The reader learned from 6,000 digits, 3,000 by employees and 3,000 by students, so neither group was missing, and the employees' 3,000 even came from just 29 writers, against 250 students. Measuring per group like this has a name, in *The idea*.",
    )
    guess_employees
    return (guess_employees,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **What a model card says.** Mitchell and others list nine parts. They come down to five questions:

        - **What is it?** What goes in and what comes out, who made it, which version it is, and its license.
        - **What is it for, and what not?** The **intended use**: the tasks and the users the makers had in mind. And the uses it is not meant for, such as a decision it should not take on its own.
        - **Which data stand behind it?** What it learned from, what it was tested on, and who is in those data.
        - **How well does it do, and for whom?** The metric over the whole test set, and per group and per condition.
        - **What goes wrong?** The limits that the numbers show, and what someone should check before using it.

        **Per group and per condition.** An accuracy over a whole test set is a mean, and a mean can hide a group for which the model does badly, as in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en) **Disaggregated evaluation** measures the metric for each group and for each condition apart. A group is about who or what is in the data: who wrote a digit, or which digit it is. A condition is about how the input comes in: a form that lies askew in the scanner, or a digit that is not in the middle of its box. Which groups and conditions matter follows from the intended use.
        """),
        mo.Html(C.split_picture(L.lang)),
        mo.md(r"""
        The digit reader on 2,000 test digits: the 1,000 students' digits of its card and 1,000 employees' digits. Each bar shows the share read right in the color of its group, then the share read wrong in purple; the line marks 90%, what this lesson asks for when reading forms. Over all 2,000 digits the reader gets 94.5% right. Per group, the students' digits go wrong almost twice as often. And turned 15° clockwise, as on a form that lies askew, the order flips: then the employees' digits go wrong more often. A card with only the first bar hides both.

        **Small groups.** You can split further, down to each writer. The 1,000 students' digits come from 247 writers, most of whom wrote 2 to 6 of them. But a number over 2 digits can only be 0%, 50% or 100%, and one mistake moves it by half. So a card gives with each number how many examples stand behind it, and leaves out groups too small to say anything.

        **A system card.** A model is seldom used on its own. In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en) the model was a small part of a system. A **system card** does for a whole system what a model card does for one model: which models it uses, what comes before and after them, such as a step that cuts out each digit or a person who checks, and how well the whole does, tested the way it is used. OpenAI published a system card with GPT-4 in 2023: what experts tested it for, such as harmful answers, and the safeguards built around the model.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **intended use** | | what a model is for: the tasks and the users its makers had in mind; a card also names uses it is not meant for | |
        | **disaggregated evaluation** | | measuring a metric for each group and each condition apart, not only over the whole test set | `right[series == 4].float().mean()` |
        | **system card** | | a card for a whole AI system: its models, the steps around them, and how well the whole does, tested as it is used | |
        """),
        L.watch_out("A card can only report the groups and conditions someone measured. A group nobody measured does not show up at all, and its absence looks like good news. So a card also says what was not tested."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To write a model card:

    1. Say what the model is: what goes in, what comes out, who made it, the version and the license.
    2. Write the intended use, and the uses it is not meant for.
    3. Say what data it learned from and was tested on, and who is in them.
    4. Choose the groups and conditions that matter for the intended use, and measure per group and per condition, with the number of examples behind each number.
    5. Write the limits: where the numbers fall below what the intended use asks for, and what is known to go wrong.
    6. Is the model part of a system? Then the system gets a system card too: its parts, and how well the whole does.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 4 for one way of splitting: per digit. The cell loads the digit reader as in [Using a pretrained model](/course/open/torch-pretrained?lang=en), and gets the 2,000 test digits with `C.test_digits()`: `images`, `digits`, the right digit of each, and `series`, the series of each digit's writer, as QMNIST records it. Series 4 was written by students, series 0 and 1 by employees. `right` holds True for each digit the reader reads right. The mask `digits == _digit` keeps the digits of one kind, and `_mask.sum()` counts them: the number of examples behind each number. Run the cell.
    """)
    return


@app.cell
def example_digits():
    import torch
    from everything_ai.models import DigitCNN

    reader = DigitCNN()
    reader.load_state_dict(torch.load(C.hub_file("model.pt")[0], weights_only=True))
    reader.eval()
    images, digits, series = C.test_digits()
    with torch.no_grad():
        right = reader(images).argmax(dim=1) == digits
    # 4. overall, and per digit, with the number of digits behind each number
    print("all", round(right.float().mean().item(), 3), len(right))
    for _digit in range(10):
        _mask = digits == _digit
        print(_digit, round(right[_mask].float().mean().item(), 3), _mask.sum().item())
    return digits, images, reader, series, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Over all 2,000 digits the reader gets 0.945 right. Per digit, that ranges from 0.89 for an 8 to 0.979 for a 0: an 8 goes wrong about five times as often as a 0, 11.0 against 2.1 in 100. Each of those numbers stands on 164 to 238 digits, enough to compare them. Whether this split belongs on the card depends on the intended use: on a form, every digit of an amount counts.
    """)
    return


@app.cell(hide_code=True)
def text_conditions():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Per group and per condition

    The intended use of the card in this lesson: reading the digits in the boxes of scanned forms. A form can lie askew in the scanner, a box can be cut out off-center, and someone can forget to make the digits white on black. The cell makes the 2,000 digits under five conditions: as written, turned 15° counterclockwise and 15° clockwise with `C.turn`, moved 3 pixels to the right with `C.move`, and dark on light with `1 - images`. `right_when` holds, for each condition, whether the reader reads each digit right. Run the cell.
    """)
    return


@app.cell
def example_conditions(digits, images, reader, torch):
    changed = {
        "original": images,
        "turned +15": C.turn(images, 15),    # 15° counterclockwise
        "turned -15": C.turn(images, -15),   # 15° clockwise
        "moved 3": C.move(images, 3),        # 3 pixels to the right
        "inverted": 1 - images,              # dark on light
    }
    right_when = {}
    for _condition in changed:
        with torch.no_grad():
            right_when[_condition] = reader(changed[_condition]).argmax(dim=1) == digits
    list(right_when)
    return (right_when,)


@app.cell(hide_code=True)
def text_exercise_per_group():
    mo.vstack([
        mo.md(r"""
        Now the table of the card. The start code goes through the conditions in `right_when`: for each `condition`, `right` holds whether the reader reads each digit right under it, and `table[condition]` gets the accuracy for the students and for the employees. The mask `students = series == 4` keeps the students' digits; `.float()` turns True and False into 1 and 0, `.mean()` takes the share, and `.item()` makes it a Python number. Replace the `...` on the line for the employees: the digits of series 0 and 1. The last line is `table`, so the check sees it. The check also puts the 2,000 digits in another order.
        """),
        L.task("Complete the line for the employees, so that `table` holds the accuracy of both groups under each condition."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_group():
    box_per_group = L.code_box("per-group", lines=10)
    box_per_group
    return (box_per_group,)


@app.cell(hide_code=True)
def check_per_group(box_per_group, right_when, series):
    result_per_group = L.check("per-group", box_per_group.value, {"right_when": right_when, "series": series})
    result_per_group
    return (result_per_group,)


@app.cell(hide_code=True)
def hints_per_group():
    L.hints(
        "per-group",
        [
            "Step 4 of the recipe: the same as the line for the students, `right[...].float().mean().item()`, with a mask from `series` that keeps only the employees' digits.",
            "Series 0 and 1 are the employees, series 4 the students. Which one comparison of `series` with 4 is True for 0 and for 1, and False for 4?",
            "`\"employees\": right[series ___ 4].float().mean().item(),`",
        ],
        why="`series != 4` is True for every digit of series 0 and of series 1: all 1,000 digits of the employees. `series < 4` gives the same mask. As written, the reader gets 0.929 of the students' digits right and 0.96 of the employees'. Turned 15° clockwise, the order flips: 0.86 against 0.827. Moved 3 pixels, or dark on light, both groups fall far below 90%.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Turn further

    The table has 15° in each direction. With the slider, the reader reads all 2,000 digits turned further, from 45° clockwise to 45° counterclockwise, in steps of 5°. At the top, six of the digits as the reader gets them: three by students, three by employees.
    """)
    return


@app.cell(hide_code=True)
def demo_turn():
    _out = L.explore("turn", "degrees", lambda _d: C.turn_frame(_d, L.lang), start=0, low=-45, high=45, step=5,
                     role="",
                     task="Turn both ways. Where does each group fall below 90%? And in which direction do the employees' digits suffer most?")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Within 10° each way, both groups stay above 90%. Counterclockwise, the employees' digits keep their lead: at 30°, the reader gets 73.6% of them right, against 64.7% of the students'. Clockwise, the employees' digits fall faster: at 30°, 45.3% right, against 63.3%. A card can turn that into a limit someone can act on: straighten every form to within 10° before the reader sees it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sections():
    mo.md(r"""
    ### Exercise 2. The parts of the card

    Eight lines for the card of the digit reader. Put each one in the part of the card where it belongs: what it is for, what it is not for, what it learned from and was tested on, or its known limits.
    """)
    return


@app.cell(hide_code=True)
def sort_sections_cell():
    sort_sections = L.sort("sections")
    sort_sections
    return (sort_sections,)


@app.cell(hide_code=True)
def check_sections(sort_sections):
    result_sections = L.check("sections", sort_sections)
    result_sections
    return (result_sections,)


@app.cell(hide_code=True)
def hints_sections():
    L.hints(
        "sections",
        [
            "Ask for each line: is it a use the makers have in mind, a use they warn against, a fact about the digits behind the reader, or a condition under which it fails?",
            "Two lines are uses the reader fits, and two are uses it does not fit. Which two say where its numbers come from?",
            "What it is for: forms and \\_\\_\\_. What it is not for: money and \\_\\_\\_. Data: \\_\\_\\_ and test digits. Known limits: turned and \\_\\_\\_.",
        ],
        why="**What it is for:** forms and lessons. **What it is not for:** money, since about 7 in 100 students' digits go wrong, and street photos, which look nothing like white digits on black. **What it learned from and was tested on:** the training digits and the test digits. **Known limits:** turned and dark on light, from the numbers of exercise 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_system():
    mo.md(r"""
    ### Exercise 3. A system card
    """)
    return


@app.cell(hide_code=True)
def choice_system_cell():
    choice_system = L.choice(
        "system-card",
        question="A bank's app reads amounts on forms: a photo of the form goes in, a step cuts out each digit, the digit reader reads them, and when the reader is unsure, a person looks. What belongs in the app's system card, and not in the reader's model card?",
        options={
            "groups": "The reader's accuracy per group of writers and per condition, as in exercise 1",
            "license": "The license of the digit reader, and of the digits of QMNIST it learned from",
            "code": "The code of the class `DigitCNN`, so that anyone can load and run the reader on its own",
            "right": "How often the whole app reads an amount right, and how often a person has to look",
        },
        correct="right",
        why={
            "right": "A system card is about the whole: the photo, the cutting, the reader and the person together. A digit cut out wrong, or an amount in which one digit of six goes wrong, only shows when the whole app is tested on real forms.",
            "groups": "That is about the reader alone, so it belongs on its model card. What can only be measured on the whole app, with its other steps?",
            "license": "That is about the reader and its data, so it belongs on its model card. What can only be measured on the whole app, with its other steps?",
            "code": "That is about the reader alone, so it belongs on its model card. What can only be measured on the whole app, with its other steps?",
        },
    )
    choice_system
    return (choice_system,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Using a pretrained model](/course/open/torch-pretrained?lang=en) you read a model card; here you measured a model the way its card should report it: per group of writers and per condition, with the number of examples behind each number, and turned those numbers into its limits. In *Project: tests and a model card* your demand model gets such a card, with its numbers per group of hours and what it is not for. *A model card for your model* writes the card of a chatbot of your own, with a header that Hugging Face reads.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_system, result_per_group, result_sections):
    L.wrap_up(result_per_group, result_sections, choice_system)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
