# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-drift. Route: course/route/lessen/systemen.yaml. Checks, the model, the hours of 2011 and 2012 and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-drift", "en")
    C = L.checks   # the course's model of the project, its answers for every hour of 2011 and 2012, the bins and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How ML systems fail](/course/open/sys-ml-failures?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="A check compares every input of a request with the lowest and the highest value that training saw. Which failure does it let through?",
        options={
            "degrees": "An app that sends the temperature in degrees, 31.2 instead of 0.76",
            "minus": "A humidity sensor with no reading that sends −1",
            "right": "A heat wave of 38.5 degrees, with hours like it in training",
            "kmh": "A step upstream that sends the wind in km/h",
        },
        correct="right",
        why={
            "right": "38.5 degrees lies inside the range of training, since training has hours that hot, so the check lets every request of the heat wave through, while the model was far off. The other three lie outside the range. In this lesson, something slower slips past such a check: the world itself moves.",
            "degrees": "31.2 lies far above the 0.96 of training. Would the check let it through? See [How ML systems fail](/course/open/sys-ml-failures?lang=en).",
            "minus": "−1 lies under the 0 of training. Would the check let it through? See [How ML systems fail](/course/open/sys-ml-failures?lang=en).",
            "kmh": "A wind of 24.0 lies far above the 0.85 of training. Would the check let it through? See [How ML systems fail](/course/open/sys-ml-failures?lang=en).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="`x` holds the temperatures of 1,000 hours. What does `np.quantile(x, 0.1)` give?",
        options={
            "tenth": "The tenth temperature of `x`, which is `x[9]`",
            "right": "The temperature below which a tenth of the hours lie",
            "share": "The share of the hours that are colder than 0.1 degrees",
            "mean": "A tenth of the mean temperature, `x.mean() / 10`",
        },
        correct="right",
        why={
            "right": "The 0.1 quantile is the value below which a tenth of the values lie: of 1,000 hours, about 100 were colder. The nine quantiles 0.1 to 0.9 split the values into ten parts, with about a tenth in each. Those are the bins of this lesson.",
            "tenth": "A quantile does not depend on the order of `x`: it asks how many values lie below. Which value has a tenth of the values below it? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "share": "The 0.1 in `np.quantile` is a share, not a temperature: it asks for a value, not for a count. Which value has that share of the hours below it? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "mean": "A quantile looks at where the values lie, not at their mean. Which value has a tenth of the hours below it? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model learns from the past, and works in the future. The world keeps moving: people move, prices change, a new app comes out, habits shift. A model that was right last year can slowly get worse without any failure: every request gets a fast answer with status 200, and every input lies within what training saw. This slow change is called *drift*. To catch it, you compare what comes in now with what the model learned from, and you watch how far off the model is once the truth comes in.

    **An everyday example.** A map of a city from ten years ago still opens and still shows streets, and most of them are right. But the new neighborhoods are missing, and it sends you the wrong way down a street that became one-way. The map does not know; you notice when you get lost.
    """)
    return


@app.cell(hide_code=True)
def predict_year():
    guess_year = L.guess(
        "year",
        question="The model of the project learned from the hours of 2011, and was 30.3 bikes off per hour over 2011. In 2012, the service answers every hour with the same model: nothing breaks, and every request gets status 200. How far off per hour is the model over 2012?",
        options={
            "same": "About 30, as in 2011: the weather of 2012 is like that of 2011",
            "bit": "About 40: a little more, since 2012 has days the model never saw",
            "right": "About 90: three times as much",
            "lot": "About 300: ten times as much",
        },
        reveal="**About 90: three times as much.** Over 2012, the model was 92.5 bikes off per hour, against 30.3 over 2011. The weather is not the cause: 2012 had about the same temperatures, humidity and wind as 2011, and the model answered about as many bikes, 148.7 per hour against 144.2. But far more people rode: 234.7 bikes per hour against 143.8, and 65 percent more bikes over the whole year. The model still answers for the world of 2011.",
    )
    guess_year
    return (guess_year,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Drift.** *Drift* is a change over time in the data a model meets in production, compared with the data it learned from. There are three places to look, each with its own name:

        - **Data drift**: the inputs change, such as the temperatures in the requests. It shows at once, at every request.
        - **Prediction drift**: the model's answers change. It shows at once too, since the service logs every answer.
        - **Label drift**: the true answers change, here the bikes rented. It shows only once the truth comes in: an hour later for the bikes, much later for many models.

        The picture shows all three for 2012, against the hours of training.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        In 2012, the temperature and the model's answers stay where they were; only the rentals move. The weather gives the same kind of requests, and the model answers as it learned in 2011; but in the same weather, far more bikes are rented. The link between the inputs and the true answer changed. That is called **concept drift**: the same input should now get another answer. Checks on the inputs and on the answers cannot see it; only the truth shows it.

        **Comparing two distributions.** To see whether values moved, you compare their distribution now with their distribution in training. Split the values of training into ten bins with the nine quantiles 0.1 to 0.9, so that each bin holds about a tenth of training. Then count the share of the new values in each bin. Where nothing changed, each bin again holds about a tenth.

        **The population stability index.** The **population stability index**, or **PSI**, turns the two lists of shares into one number. For each bin, take the share now minus the share in training, multiply it by the log of the share now divided by the share in training, and add up the parts of all bins:

        $$\text{PSI} = \sum_{\text{bins}} (\text{now} - \text{train}) \cdot \log \frac{\text{now}}{\text{train}}$$

        Each part is 0 or more: where a share went up, both factors are positive, and where it went down, both are negative. A bin that moved from 0.1 to 0.2 adds 0.1 × log 2 = 0.069. A rule of thumb from banks, which have long watched their loan models with the PSI: under 0.1, nothing to worry about; from 0.1 to 0.25, keep an eye on it; above 0.25, serious drift. The rentals in the picture give 0.24 over the whole year, just under 0.25: a whole year mixes the seasons. Month by month, in *Each month of 2012* further on, the rentals lie above it in most months.

        **Like with like.** A month of 2012 against all hours of training always looks different: July is summer, and training holds a whole year. So compare a month with the same month in training.

        **At scale.** Tools for model monitoring, such as Evidently and NannyML, compute the PSI and numbers like it for every input and for the answers, every day or every week, and raise an alert when one goes too high. Where the truth comes late, they watch the inputs and the answers in the meantime, knowing that concept drift can slip past both.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **data drift** | | a change in the inputs a model gets in production, compared with training | |
        | **prediction drift** | | a change in the answers a model gives in production, compared with its answers in training | |
        | label drift | | a change in the true answers, such as the bikes rented, compared with training; it shows only once the truth is in | |
        | concept drift | | a change in the link between the inputs and the true answer: the same input should now get another answer | |
        | **population stability index** (PSI) | | a number for how far a distribution moved from training: for each bin, the share now minus the share in training, times the log of the share now over the share in training, added up over the bins | `psi(train, now)` |
        | infinity, $\infty$ | | a number above every other number; minus infinity lies below every other | `np.inf`, `-np.inf` |
        """),
        L.watch_out("A calm PSI on the inputs does not mean a healthy model. In 2012, the temperature and the answers stay calm while the model is three times as far off: a change in the link between the inputs and the answer shows only in the truth."),
        L.watch_out("An empty bin has a share of 0, and the log of 0 is minus infinity. Tools give such a bin a tiny share, such as 0.0001, before they take the log."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Checking a model for drift:

    1. Choose what to watch: each input (data drift), the model's answers (prediction drift), and once known, the true answers (label drift).
    2. Choose what to compare with: the data of training, from the same season, such as the same month.
    3. Make ten bins from that data: minus infinity, the quantiles 0.1 to 0.9, and infinity.
    4. Count the share of the values in each bin, of training and of now.
    5. Compute the PSI: for each bin, (now − train) × log(now / train), added up over the bins.
    6. Read it with the rule of thumb, under 0.1 calm and above 0.25 drift, next to how far off the model is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.TEMPS_TRAIN` holds the temperature in degrees of each of the 6,912 hours of training, and `C.TEMPS_2012` of each of the 8,734 hours of 2012. `C.edges(values)` gives the edges of ten bins: minus infinity, the quantiles 0.1 to 0.9 of `values`, and infinity. `C.shares(values, edges)` gives the share of `values` in each bin, and `C.psi(train, now)` the PSI of two lists of shares: the course's own versions of what you write in Exercises 1 and 2. The cell prints the nine edges in the middle, the shares of training and of 2012, and the PSI. Run the cell.
    """)
    return


@app.cell
def example_year():
    import numpy as np

    _edges = C.edges(C.TEMPS_TRAIN)
    _train = C.shares(C.TEMPS_TRAIN, _edges)
    _now = C.shares(C.TEMPS_2012, _edges)
    print(np.round(_edges[1:-1], 1))     # the nine edges between minus infinity and infinity
    print(np.round(_train, 3))
    print(np.round(_now, 3))
    print(round(C.psi(_train, _now), 3))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The nine edges run from 9.0 to 30.3 degrees. Training has between 0.087 and 0.121 of its hours in each bin: about a tenth, not exactly, since the temperatures in the data come in steps of 0.82 degrees and many hours share a value. 2012 has about the same shares, with fewer hours in the coldest bin, under 9.0 degrees: 0.040 against 0.087. The PSI is 0.041, under 0.1: calm. Over the whole year, the weather of 2012 looks like the weather of training, and still the model is three times as far off. In Exercises 1 and 2, you write `shares` and `psi` yourself, and look at a single month.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The share in each bin

        `shares(values, edges)` gets an array of `values` and an array of `edges`, as `C.edges` makes them. Bin `i` runs from `edges[i]` up to `edges[i + 1]`: a value exactly on `edges[i]` belongs to bin `i`, and a value exactly on `edges[i + 1]` to the bin above it. The loop goes over each `i` in `range(len(edges) - 1)`, and the list `out` collects a share per bin. The first line of the loop picks from `values` the values from the bottom of bin `i` up, with a mask, and calls them `inside`. The last line adds the share `len(inside) / len(values)` to `out` with `append`, and the function gives back `np.array(out)`. Replace the `...` so that `inside` keeps only its values under the top of bin `i`.

        The check calls `shares(C.MARCH_2012, C.MARCH_EDGES)`: the temperatures of the 743 hours of March 2012, in the bins of the March hours of training. Then the 566 March hours of training, `C.MARCH_TRAIN`, in the same bins, the whole of 2012 in the bins of `C.EDGES`, and two short arrays with values exactly on an edge.
        """),
        L.task("Make `inside` keep only its values under the top of bin `i`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=7)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Step 4 of the recipe: `inside` holds the values from the bottom of bin `i` up, and `edges[i + 1]` is the top of bin `i`. Which values of `inside` lie in bin `i`?",
            "A mask is a comparison between square brackets, as on the line above: `values[values >= edges[i]]`. This time, the comparison is between `inside` and the top of the bin, and a value exactly on the top belongs to the bin above. With which sign?",
            "`        inside = inside[___]`",
        ],
        why="`inside = inside[inside < edges[i + 1]]`: of the values from the bottom of bin `i` up, the mask keeps those under the top. A value exactly on the top fails `<` here and passes `>=` in the next bin, so each value lands in exactly one bin. In March 2012, a share of 0.506 lands in the top bin.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_psi():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The PSI

        `psi(train, now)` gets two arrays with a share per bin: `train`, the shares of training, and `now`, the shares of the new values. Its first two lines give an empty bin a tiny share with `np.maximum`, which takes the larger of each share and 0.0001. The last line multiplies `now - train` by a factor per bin, and adds up the parts with `.sum()`. Replace the `...` with that factor: the log of the share now divided by the share in training.

        The check calls `psi(C.SHARES_TRAIN, C.SHARES_2012)`: the shares of the March hours of training and of March 2012, in the bins of `C.MARCH_EDGES`, as `shares` gives them. Then the whole year of 2012 against training, July 2012 (`C.JULY_2012`) against all of training, the same shares twice, the two of March swapped, and two short arrays, one with an empty bin.
        """),
        L.task("Replace the `...` with the log of the share now divided by the share in training."),
    ])
    return


@app.cell(hide_code=True)
def exercise_psi():
    box_psi = L.code_box("psi", lines=4)
    box_psi
    return (box_psi,)


@app.cell(hide_code=True)
def check_psi(box_psi):
    result_psi = L.check("psi", box_psi.value)
    result_psi
    return (result_psi,)


@app.cell(hide_code=True)
def hints_psi():
    L.hints(
        "psi",
        [
            "Step 5 of the recipe: in each bin, the part is `now - train` times a factor, and `.sum()` adds up the parts. The factor is a log, and it is 0 where `now` and `train` are equal. The log of what?",
            "The log of 1 is 0, and the share now divided by the share in training is 1 where nothing changed. `np.log` takes the natural log of every number of an array at once. What goes between its parentheses?",
            "`    return ((now - train) * ___).sum()`",
        ],
        why="`return ((now - train) * np.log(now / train)).sum()`: in each bin, how much the share moved, times the log of how many times as large it became, added up over the bins. March 2012 against March in training gives 1.15; the temperature of the whole year, 0.041.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_season():
    mo.md(r"""
    ### Exercise 3. Against which month?

    You compare the temperatures of July 2012 with the bins of all hours of training, `C.EDGES`, and your `psi` gives 4.84, far above 0.25.
    """)
    return


@app.cell(hide_code=True)
def choice_season_cell():
    choice_season = L.choice(
        "season",
        question="What does that number say?",
        options={
            "hot": "July 2012 was far hotter than July 2011, so the model needs training again",
            "right": "Little: July is summer, and the bins come from all seasons of the year",
            "bins": "Ten bins are too few for one month: with a hundred bins, the PSI would lie under 0.25",
            "broken": "The temperature sensor broke: a PSI above 1 can only come from a broken input",
        },
        correct="right",
        why={
            "right": "Against all of training, a summer month always lies in the top bins: every month of 2012 lies above 0.25 that way, even with weather like 2011. Against the July hours of training, July 2012 gives 0.06: calm. Compare like with like, such as the same month.",
            "hot": "July 2012 against the July hours of training gives 0.06. Which comparison makes July look so different?",
            "bins": "More bins would split the year into more pieces, and July would still fill the top ones. What do the bins of all of training hold that July does not?",
            "broken": "A broken sensor can push the PSI up, but so can a season. What do the bins of all of training hold that July does not?",
        },
    )
    choice_season
    return (choice_season,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 4. Which drift?

    Six changes that could happen to the service of the project. Put each with the kind of drift it is.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
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
            "Ask for each change: do the requests change, or only what comes out of the model, or only what riders really do?",
            "Two changes start in the requests, two in the service itself, and two in the riders, while the requests and the answers stay the same.",
            "Data: a warm March, \\_\\_\\_. Prediction: a new version, \\_\\_\\_. Label: more members, \\_\\_\\_.",
        ],
        why="**Data drift:** a warm March and a new sensor, since the inputs change. **Prediction drift:** a new version and another scaling, since only the answers change. **Label drift:** more members and a price rise, since only the bikes rented change.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Each month of 2012

    Choose what to watch and what to compare with. The bars show the PSI of each month of 2012; the lines under them show how far off the model was per hour in each month, in 2011 and in 2012.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_column = mo.ui.dropdown(
        options={
            "The temperature": "temp",
            "The humidity": "hum",
            "The wind": "wind",
            "The model's answers": "answers",
            "The rentals": "rentals",
        },
        value="The temperature",
        label="Watch",
    )
    demo_ref = mo.ui.dropdown(
        options={"the same month in training": "same", "all of training": "all"},
        value="the same month in training",
        label="Compare with",
    )
    mo.vstack([
        mo.hstack([demo_column, demo_ref], justify="start", wrap=True),
        L.task("Start with the temperature against the same month, then the model's answers, then the rentals. The model was further off than in 2011 in every month of 2012: which of the three lies above 0.25 in most months? Then compare with all of training: what happens to July, and to the rentals of January?"),
    ])
    return demo_column, demo_ref


@app.cell(hide_code=True)
def demo_picture(demo_column, demo_ref):
    mo.Html(C.demo_picture(demo_column.value, demo_ref.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In all 12 months of 2012, the model was further off than in the same month of 2011. Against the same month in training, the temperature lies above 0.25 in 5 of them, March most of all, and July is calm, at 0.06, while the model was 86.8 bikes off per hour, against 46.5 in July 2011. The answers stay under 0.1 in every month. The rentals lie above 0.25 in 10 of the 12 months: only the truth gives the alarm in almost every month. The height of the PSI does not tell how far off the model is, though: the rentals of January give the highest PSI, in the month the model was least off. Against all of training, the temperature lies above 0.25 in every month, a false alarm of the seasons, and the rentals of January drop from 0.85 to 0.04: a real drift hides.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How ML systems fail](/course/open/sys-ml-failures?lang=en), a check against the range of training looked at one request at a time and caught a unit that changed; the PSI looks at a whole month of requests at once, and sees a shift inside the range. The bins come from the quantiles of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), and the factor of each bin is a log, as in [Exponents and logarithms](/course/open/math-exp-log?lang=en). In the next lesson, *When a model changes its own data: feedback loops*, the model itself moves the world it learns from. And in *Project: your model as a service, watched for drift*, your service checks the data of 2012 month by month.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_season, result_kinds, result_psi, result_shares):
    L.wrap_up(result_shares, result_psi, choice_season, result_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
