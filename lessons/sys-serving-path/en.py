# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-serving-path. Route: course/route/lessen/systemen.yaml. Checks, the course's demand model, the serving path and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-serving-path", "en")
    C = L.checks   # the course's demand model, the requests of the lesson, the serving path and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) and [Your model behind an API](/course/open/sys-serving-api?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="A spam filter calls a mail spam once its probability of spam reaches 0.9. You lower the decision threshold to 0.5. What happens?",
        options={
            "same": "Nothing changes: the model gives the same probabilities",
            "right": "More spam is caught, and more good mail is called spam too",
            "fewer": "Less spam gets through, and fewer good mails are called spam",
            "retrain": "The filter first has to be trained again for the new threshold",
        },
        correct="right",
        why={
            "right": "A lower threshold calls more mails spam: recall goes up, and so do the false positives. The model itself stays the same; only the line after it moves. In this lesson, such a line turns the number of a model into a decision.",
            "same": "The probabilities stay the same. But which of them now count as spam? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "fewer": "From 0.5 on, more mails are called spam. Can fewer good mails be called spam when more mails are? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "retrain": "The threshold comes after the model. Does the model have to change when the line moves? See [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def quick_check_ok():
    quick_ok = L.choice(
        "quick-ok",
        question="A server answers a request for a prediction with status 200 and {\"bikes\": 4855}. What does the 200 tell you?",
        options={
            "busy": "That it will be a busy hour",
            "checked": "That the model compared the number with the data of training",
            "right": "That the server could answer, not that the number is right",
            "right-number": "That the number is right, since a wrong number would get status 400",
        },
        correct="right",
        why={
            "right": "200 means that the server could read the request, run the model and answer. A temperature of 26.2 in degrees, where the data has 0.64, still gets 200 and 4,855 bikes. Whether the number is any good depends on what happens before and after the model: the subject of this lesson.",
            "busy": "The status code is about the request, not about the bikes. What does 200 say about how the request went? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
            "checked": "The server of that lesson checked the fields of a request, not the number of the model. What does 200 say, then? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
            "right-number": "A 400 is for a request the server cannot use, such as a missing field. Does a well-formed request with a temperature in degrees get one? See [Your model behind an API](/course/open/sys-serving-api?lang=en).",
        },
    )
    quick_ok
    return (quick_ok,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that did well in validation can do badly once it serves, without a single error. In [Your model behind an API](/course/open/sys-serving-api?lang=en), the answers for 3:00 and 4:00 at night were below 0 bikes, down to −11, and a temperature written in degrees got 4,855 bikes, all with status 200. The model is 34 bikes off per hour on average on the validation days; what went wrong happened before and after it.

    Before the model, every input has to be prepared exactly as in training: the same columns, in the same order, in the same units, scaled with the same numbers. After the model, a number has to become something people can act on: a whole number of bikes, never below 0, and a decision, such as whether an extra crew goes out with a truck to fill the stations. In *Rules of Machine Learning*, Martin Zinkevich wrote down what engineers at Google learned from years of models in production. Rule 32 says to re-use code between the training pipeline and the serving pipeline; rule 37 says to measure the difference between how a model does in training and in serving.

    **An everyday example.** A recipe written for an oven in Celsius says 180 degrees. In an oven that counts in Fahrenheit, 180 degrees is barely warm, and the cake comes out raw. The recipe is fine and the oven is fine, and nothing beeps: they only work together when both mean the same thing by a number.
    """)
    return


@app.cell(hide_code=True)
def predict_batch():
    guess_batch = L.guess(
        "batch",
        question="The model of the project learned from features standardized with the mean and standard deviation of the training hours. A new night batch prepares the 24 hours of a day on its own: it standardizes them with the mean and standard deviation of those 24 hours. What happens to its prediction for 8:00 on Thursday 14 July 2011, a clear summer day, do you think?",
        options={
            "error": "The code stops with an error: the scale no longer fits",
            "same": "Nothing changes: the same hour, with the same weather",
            "more": "Far more bikes, since the morning looks warm",
            "right": "Far fewer bikes, and nothing stops with an error",
        },
        reveal="**Far fewer bikes, and nothing stops with an error.** With the scale of training, 8:00 gets 431 bikes, and 486 were rented. With the scale of the batch, it gets 169. In July, every hour is warm, so the mean of the batch is warm too: temp 0.64 at 8:00 lies below the day's own mean of 0.68, and becomes −0.77 instead of +0.76. To the model, a warm morning now looks like a cool one. The weekday, the month and the season have one value all day, so they become 0, which to the model means an average weekday, month and season. Over all 73 validation days, such a batch is 73 bikes off per hour on average, where the scale of training is 34 off. The code ran, every hour got its number, and the numbers were wrong.",
    )
    guess_batch
    return (guess_batch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The serving path.** Between a request and an answer, the model is one step of several. The *serving path* of this lesson has five: the request; the check of [Your model behind an API](/course/open/sys-serving-api?lang=en), which answers a bad request with 400; *prepare*, which turns the request into the 34 inputs the model learned from; the model; and *decide*, which turns its number into an answer the business can use. Everything before the model has one job: to give the model its input exactly as training did. What happens after the model is up to the business.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Training-serving skew.** When serving prepares an input differently from training, the model gets numbers it never learned from, and gives numbers that are off, without an error. That difference is **training-serving skew**. It comes in many forms:

        - *statistics computed again*: a scaler fitted on the data of the day, instead of the mean and standard deviation saved from training, as in the guess;
        - *units*: a temperature in degrees, where the data of training has degrees divided by 41, as in the 4,855 bikes;
        - *time*: hours in UTC in one place and in local time in another, so that 8:00 in Washington, D.C., becomes 12:00 or 13:00;
        - *order*: the same 10 features in another order, so that each weight multiplies the wrong column.

        The main defense: one function `prepare` that training and serving both call, with the mean and standard deviation computed once, on the training hours, and saved in config.json next to the weights. It is step 4 of the recipe of [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en): a new example is standardized with the same `mean` and `std`. A check of the units and the time zone of every request, a test that sends the same hours through both paths and compares the inputs, and a log of the inputs a service gets catch what slips through anyway.

        **Business rule.** After the model, a number becomes an answer. A **business rule** is a fixed rule of the organization that a system applies around a model, whatever the model says: never fewer than 0 bikes, whole bikes only, no crew between midnight and 6:00. Such rules do not come from data, and a model should not have to learn them: people decide them, they change without training again, and anyone can read them in a few lines of code.

        **A decision threshold from costs.** In a busy hour, with at least 400 rentals, the planners of the bike-share company send an extra crew with a truck to fill the stations. The model gives a number; a decision threshold, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), turns it into yes or no. Where to put it depends on what each mistake costs. A crew that goes out for nothing costs one unit; a busy hour without a crew costs three, in empty stations and rides that never happen. So the threshold need not be the 400 of a busy hour. You choose it on the validation set, in Exercise 3.

        **At scale.** Large teams keep the preparation in one place. TensorFlow Transform computes the statistics once over the training data and writes them into the graph the server runs. TensorFlow Data Validation compares the inputs of training with those of serving, and reports skew. A *feature store* computes each feature once, and serves the same value to training and to the model in production. The idea stays the same: one definition of every input, used on both sides.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **training-serving skew** | | a difference between how inputs are prepared in training and in serving, which makes a good model give wrong numbers without an error | |
        | **business rule** | | a fixed rule of the organization that a system applies around a model, whatever the model says | |
        """),
        L.watch_out("Good numbers in validation say nothing about skew: validation prepared its hours with the code of training. Skew only shows in serving, so log the inputs a service gets, and compare them with those of training."),
        L.watch_out("Keep business rules out of the model. A model trained to never give less than 0, or to stay quiet at night, mixes what it learned from data with what people decided, and nobody can change the rule without training again."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Building the serving path around a model:

    1. Write the preparation once, as a function that training and serving both call.
    2. Save the statistics of training, such as the mean and standard deviation of every feature, next to the weights, and load them when the service starts; never compute them again on the data of serving.
    3. Check every request first, and keep the units, the time zone and the order of the features the way training had them.
    4. Turn the number of the model into an answer with business rules: rounded, never below 0, and whatever else the organization decided.
    5. Choose the decision threshold on the validation set, from what each mistake costs.
    6. Test that both paths give the same inputs for the same hours, and log the inputs the service gets.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.prepare(request)` is the course's preparation, with the mean and standard deviation of the training hours from `C.CONFIG`. `C.expect(x)` runs the model on the 34 inputs and gives the bikes as a plain number, and `C.decide(pred)` turns that number into bikes and a decision. The cell sends two hours of 14 July 2011 through the path: 8:00, and 4:00 at night. The first 24 inputs are the hour; the cell prints the 10 after them, rounded, in the order of `C.FEATURES`: weekday, workingday, holiday, weathersit, temp, atemp, hum, windspeed, season and mnth. Run the cell.
    """)
    return


@app.cell
def example_path():
    for _request in (C.DAY[8], C.DAY[4]):
        _x = C.prepare(_request)
        _pred = C.expect(_x)
        print("hour", _request["hr"], "features:", [round(v, 2) for v in _x[24:].tolist()])
        print("  model:", round(_pred, 1), " decide:", C.decide(_pred))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At 8:00, temp 0.64 goes in as 0.76, the fifth number: a little warmer than the average training hour. The model makes 431.2, and `decide` makes it 431 bikes and sends the extra crew; 486 were rented. At 4:00, the model makes −10.8, a number no planner can use, and `decide` makes it 0 bikes. In Exercise 1, you write the line of `prepare` that standardizes; in Exercise 2, the business rule of `decide`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prepare():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. As in training

        `prepare(request)` gets the body of a request as a dict, with `hr` and the 10 features, and gives back the 34 inputs of the model as one tensor. `one_hot` makes `hour`, the 24 numbers of the hour. `values` holds the 10 features in a tensor made by `torch.tensor`: `float(request[f])` for each name `f` in `FEATURES`, in the order of training. `CONFIG` is what config.json keeps of training: a dict with a list of 10 means under `"mean"` and a list of 10 standard deviations under `"std"`, both plain Python lists. `torch.cat` puts `hour` and `scaled` together. Replace the `...` so that `scaled` is `values` standardized with the numbers of training.

        The check calls `prepare(C.DAY[8])`, and compares the 34 inputs with those of training. Then 17:00 and 3:00 of the same day, and 17:00 on 7 December 2011, in light rain.
        """),
        L.task("Make `scaled` the features standardized with the mean and standard deviation of training."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prepare():
    box_prepare = L.code_box("prepare", lines=5)
    box_prepare
    return (box_prepare,)


@app.cell(hide_code=True)
def check_prepare(box_prepare):
    result_prepare = L.check("prepare", box_prepare.value)
    result_prepare
    return (result_prepare,)


@app.cell(hide_code=True)
def hints_prepare():
    L.hints(
        "prepare",
        [
            "Step 2 of the recipe: standardize with the numbers saved from training, never with numbers of the request itself. From `values`, subtract the 10 means, then divide by the 10 standard deviations. Both are lists in `CONFIG`, under `\"mean\"` and `\"std\"`, and `torch.tensor` makes a tensor of a list. The result goes in `scaled`.",
            "As in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en): `(new - mean) / std`. Python divides before it subtracts, so the subtraction needs parentheses. What goes after the minus sign, as a tensor?",
            "`    scaled = (values - ___) / torch.tensor(CONFIG[\"std\"])`",
        ],
        why="`scaled = (values - torch.tensor(CONFIG[\"mean\"])) / torch.tensor(CONFIG[\"std\"])`: each feature minus its mean over the training hours, divided by its standard deviation over the training hours. At 8:00, temp 0.64 becomes 0.76, as in training, and the model makes 431.2 bikes again.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decide():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A business rule

        `decide(pred)` gets the number the model makes for one hour, a plain Python number such as 431.2 or −10.8, and gives back the answer of the service: a dict with the bikes and the decision. `THRESHOLD` is 400: from that many bikes on, `extra_crew` is `True`, and the planners send the extra crew. Replace the `...` so that `bikes` is `pred` rounded to a whole number of bikes, and never below 0.

        The check calls `decide(-10.8)`: the number of 4:00 at night. Then the numbers of 8:00, 19:00, 17:00 and 3:00 of the same day, and a number just below 400.
        """),
        L.task("Make `bikes` a whole number of bikes, never below 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decide():
    box_decide = L.code_box("decide", lines=4)
    box_decide
    return (box_decide,)


@app.cell(hide_code=True)
def check_decide(box_decide):
    result_decide = L.check("decide", box_decide.value)
    result_decide
    return (result_decide,)


@app.cell(hide_code=True)
def hints_decide():
    L.hints(
        "decide",
        [
            "Step 4 of the recipe: round `pred` with `round`, and keep the result from going below 0, so that the −10.8 of the model becomes 0 bikes. The answer goes in `bikes`.",
            "Python has a function that gives the larger of two numbers. With 0 as one of the two, what comes out is never below 0. Which function is it?",
            "`    bikes = ___(0, round(pred))`",
        ],
        why="`bikes = max(0, round(pred))`: `round` makes 431.2 the whole number 431, and `max` gives the larger of 0 and that number, so −10.8 becomes 0 bikes. The model stays as it is; the rule after it makes its number an answer the planners can use.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Exercise 3. Where to draw the line

    The planners count the decisions of `decide` on the 1,733 hours of the 73 validation days, for four thresholds. 115 of those hours were busy, with at least 400 rentals. A crew that goes out for nothing costs 1; a busy hour without a crew costs 3.

    | Threshold | Crew in a busy hour | Crew for nothing | Busy hour, no crew |
    |---|---|---|---|
    | 300 | 100 | 110 | 15 |
    | 350 | 84 | 47 | 31 |
    | 400 | 62 | 12 | 53 |
    | 450 | 34 | 3 | 81 |
    """)
    return


@app.cell(hide_code=True)
def choice_threshold_cell():
    choice_threshold = L.choice(
        "threshold",
        question="Which threshold costs the planners the least?",
        options={
            "300": "300 bikes: the crew misses only 15 busy hours",
            "right": "350 bikes: 47 trips for nothing and 31 misses",
            "400": "400 bikes: the same line as a busy hour",
            "450": "450 bikes: the crew almost never goes out for nothing",
        },
        correct="right",
        why={
            "right": "350 costs 47 + 3 × 31 = 140, the least of the four: 300 costs 110 + 3 × 15 = 155, 400 costs 12 + 3 × 53 = 171, and 450 costs 3 + 3 × 81 = 246. A miss costs three times as much as a trip for nothing, so the line goes below the 400 of a busy hour: the crew goes out a little more often, and misses fewer busy hours. With other costs, another threshold wins: the costs are a choice of the business, not of the model.",
            "300": "Only 15 misses, but 110 trips for nothing. What does that cost in all, at 1 for each trip and 3 for each miss?",
            "400": "The same number as a busy hour feels right. But what does it cost, with 53 misses at 3 each?",
            "450": "Almost no trips for nothing, but how many busy hours does the crew miss, at 3 each?",
        },
    )
    choice_threshold
    return (choice_threshold,)


@app.cell(hide_code=True)
def text_exercise_sides():
    mo.md(r"""
    ### Exercise 4. Before or after the model?

    Eight steps of the serving path of the bike-share company. Put each on its side: before the model, where every step does what training did, or after the model, where business rules and a threshold turn the number into an answer.
    """)
    return


@app.cell(hide_code=True)
def sort_sides_cell():
    sort_sides = L.sort("sides")
    sort_sides
    return (sort_sides,)


@app.cell(hide_code=True)
def check_sides(sort_sides):
    result_sides = L.check("sides", sort_sides)
    result_sides
    return (result_sides,)


@app.cell(hide_code=True)
def hints_sides():
    L.hints(
        "sides",
        [
            "Ask for each step: does the model need it to read its input as in training, or does it change what comes out?",
            "Four steps are about what goes in: the scale, the hour, the units and the order of the features. Which four are about the number that comes out?",
            "Before: standardize, the hour, \\_\\_\\_ and \\_\\_\\_. After: below 0, the crew, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Before the model:** standardize, the hour, degrees and the order, since the model only gives good numbers for input prepared as in training. **After the model:** below 0, the crew, at night and whole bikes, since the business decides what the number means and what may never happen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The night batch

    A night batch prepares the 24 hours of one day in advance, and the planners act on its decisions in the morning. Choose a day of the validation set, and where the batch takes its mean and standard deviation from: from the training hours, as config.json keeps them; from the 24 hours of the batch itself; or from nowhere, with no scaling at all. The picture shows the rentals, what the model makes with the scale of training and with the scale you chose, and the hours in which the chosen path sends the crew, at the threshold of 350 from Exercise 3.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    _w = C.DEMO_TEXT[L.lang]
    demo_day = mo.ui.dropdown(options={_w["days"][k]: k for k in C.DAYS}, value=_w["days"]["thu"], label="Day")
    demo_source = mo.ui.radio(options={_w["sources"][k]: k for k in C.SOURCES}, value=_w["sources"]["training"],
                              label="Mean and standard deviation from")
    mo.vstack([
        demo_day,
        demo_source,
        L.task("Choose the batch itself for each day, then no scaling. On which day does the batch send no crew at all? And how far off is the path without scaling, over all 73 days?"),
    ])
    return demo_day, demo_source


@app.cell(hide_code=True)
def demo_picture(demo_day, demo_source):
    mo.Html(C.day_picture(demo_day.value, demo_source.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    On Thursday 14 July, the scale of training sends the crew at 8:00, 17:00, 18:00 and 19:00: exactly the four busy hours. The batch sends it at 17:00 and 18:00 only, and gives 169 bikes for 8:00, when 486 were rented. On Wednesday 3 August, the batch sends no crew at all, though three hours were busy. With no scaling, the model is off in its own way, and sends the crew in hours that are not busy. Over all 73 days, the batch is 73 bikes off per hour on average and the path without scaling 120, where the scale of training is 34. Not one of these runs stopped with an error.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Your model behind an API](/course/open/sys-serving-api?lang=en), the server passed on whatever the model said, even a number below 0 bikes, or 4,855 for a temperature in degrees. This lesson put a path around the model. Before it, the input is prepared as in training, standardized as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) with the mean and standard deviation saved from the training hours. After it, business rules and a decision threshold chosen from costs, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), turn the number into an answer. In the next lesson, *Watching a running model: metrics, logs and alerts*, a service keeps an eye on itself: what it gets, what it answers, and when someone has to look.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_threshold, result_decide, result_prepare, result_sides):
    L.wrap_up(result_prepare, result_decide, choice_threshold, result_sides)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
