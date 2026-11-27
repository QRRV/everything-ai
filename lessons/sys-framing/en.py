# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-framing. Route: course/route/lessen/systemen.yaml. Checks, the apps, the feed of the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-framing", "en")
    C = L.checks   # the views, the apps and their scores, the feed of the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Classification: predicting a class](/course/open/ml-classification?lang=en) and [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_task():
    quick_task = L.choice(
        "quick-task",
        question="A model predicts the price of a house in euros. What kind of task is that?",
        options={
            "classification": "Classification: each price is a class",
            "right": "Regression: the price is a number in euros",
            "reward": "Reinforcement learning: a reward",
            "ranking": "Ranking: the houses from cheap to expensive",
        },
        correct="right",
        why={
            "right": "A price is a number, so this is a regression. This lesson shows that one wish can often be framed in more than one way, and that the choice matters later.",
            "classification": "Classification picks one class from a fixed set, such as spam or ok. What does a model give that predicts a price? See [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "reward": "Reinforcement learning learns from rewards, and this model learns from houses with their price. What kind of task predicts a number? See [Classification: predicting a class](/course/open/ml-classification?lang=en).",
            "ranking": "A ranking puts items in order, and this model gives one price per house. What kind of task predicts a number? See [Classification: predicting a class](/course/open/ml-classification?lang=en).",
        },
    )
    quick_task
    return (quick_task,)


@app.cell(hide_code=True)
def quick_check_mae():
    quick_mae = L.choice(
        "quick-mae",
        question="A model predicts how many minutes a user watches a video. On the test set, its MAE is 3 minutes. What does that say?",
        options={
            "percent": "3% of its predictions are wrong",
            "every": "It predicts 3 minutes for every video",
            "right": "It is typically about 3 minutes off",
            "squared": "Its squared misses average 3",
        },
        correct="right",
        why={
            "right": "The MAE is the mean of the misses without their sign, in the unit of the label: minutes. Minutes watched come back in this lesson, at a real company.",
            "percent": "The MAE is in the unit of the label, not a percentage. In which unit are the misses? See [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).",
            "every": "That would be a model that always predicts the same. What does the MAE measure? See [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).",
            "squared": "Squaring the misses first belongs to the MSE and the RMSE. What does the MAE do with the sign of each miss? See [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en).",
        },
    )
    quick_mae
    return (quick_mae,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2016, three engineers at Google described how YouTube chooses the videos it recommends (Covington, Adams and Sargin). An obvious way to frame the task is to predict whether a user will click on a video: a classification, yes or no. They wrote that ranking by how often people click "often promotes deceptive videos that the user does not complete": videos with a tempting picture and title, known as clickbait. So they framed the task differently: the model predicts how long the user is expected to watch each video it shows, and the ranking is based mostly on that.

    The wish stayed the same: show people videos they want to see. What changed is what the model predicts, and with it which videos end up at the top. Choosing that is called framing the problem, and it happens before any training. This lesson shows three common framings, how the choice decides what you can change later without a new model, and what to do when a system has two goals at once.

    **An everyday example.** A library wants to help readers find good books. It could predict whether a reader will borrow a book (yes or no), how many pages the reader will read (a number), or the order in which to show ten new books (a ranking). Predicting borrowing rewards nice covers; predicting pages read rewards books people finish.
    """)
    return


@app.cell(hide_code=True)
def predict_new_app():
    guess_new_app = L.guess(
        "new-app",
        question="A phone has 40 apps. A model predicts which app the user opens next: a classification with one output per app, 40 in all. Then the user installs a 41st app. What does the model need to ever pick it?",
        options={
            "right": "Training again, with one more output",
            "nothing": "Nothing: it learns the new app by itself",
            "example": "One example of the new app in its settings",
            "threshold": "A lower threshold for new apps",
        },
        reveal="**Training again.** The model has one output for each of the 40 apps it was trained on, and none for the 41st. Chip Huyen (2022) uses this example in a course on ML systems at Stanford, and frames the problem differently: a model gets the user, the moment and one app, and gives one number from 0 to 1, how likely the user opens that app. To choose, you score every app and take the highest. A new app is then one more app to score, not a new output.",
    )
    guess_new_app
    return (guess_new_app,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Framing a problem.** Before a model is trained, someone decides what exactly it predicts: what goes in, what comes out, and which label it learns from. That is **problem framing**. Most wishes can be framed in more than one way:

        | Framing | The model gives | On the video site | What it means later |
        |---|---|---|---|
        | classification | one class from a fixed set, or a probability per class | will the user click: yes or no | a new class, such as a new app, means training again |
        | regression | a number | how many minutes the user will watch | a threshold can live in code, outside the model |
        | ranking | an order of items | the 10 videos in the order to show them | needs a score per item, often from a regression or a probability |

        **Where the threshold lives.** Google's course on problem framing gives this example: a video site keeps a video in an expensive cache, fast storage close to the viewers, when it is predicted to get at least 50 views. A classification model trained on "50 views or more: yes or no" has the 50 built into its labels. A regression model predicts the number of views, and the 50 is a line of code. Google's advice: when the threshold is fixed, use a classification model; when it may change, predict the number and keep the threshold in the app's code, so it can move without a new model. The spam service in [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en) also keeps its threshold outside the model: its model gives a probability, and the threshold on that probability sits in its settings.

        **New items.** In Huyen's example of the next app, a classification with one output per app has to be trained again for every new app. Framed as a regression, with one score per user, moment and app, a new app is one more item to score. Huyen calls this a regression; you can also see it as a yes-or-no classification per app. Either way, a new app needs no new output. A ranking then takes the app with the highest score, or the ten highest.

        **The label stands in for the wish.** No label is the wish itself. At YouTube, clicks rewarded clickbait; minutes watched, the engineers wrote, capture engagement better, and they are a stand-in too. Ask of every label how a model could get it right without doing what you meant.

        **Two goals.** Many systems have two goals that pull apart. Huyen's example is a news feed that should keep engagement high, how much users click and react, and limit misinformation and extreme views. One way is one model, trained on one combined loss: α × quality loss + (1 − α) × engagement loss. Each time the team wants another balance, it has to train the model again. The other way is to train a quality model and an engagement model, each on its own loss, and to rank by α × quality + (1 − α) × engagement, worked out in code. These are **decoupled objectives**: the team changes α in a second, without training anything. Huyen writes α and β for the two weights; this lesson uses α and 1 − α, so the weights add up to 1.

        Huyen gives a second reason: each model can be updated on its own schedule. In Huyen's example, a spam filter needs updates much more often than a quality model, since spam tricks change much faster than what people see as a good post. YouTube's ranking system of 2019 (Zhao and others) trains one model with an output per goal, such as clicks, time watched and likes, and combines these outputs in a formula whose weights the team tunes by hand. The picture shows both ways for the feed.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **problem framing** | | deciding what a model predicts: what goes in, what comes out, and which label it learns from | `next_app(apps, scores)` |
        | **decoupled objectives** | | one model per goal, with their scores combined by weights in code, so the balance changes without training | `combined(quality, engagement, alpha)` |
        | **α** | alpha | the weight of quality in the combined score; engagement gets 1 − α | `alpha` |
        """),
        L.watch_out("The weight α is a choice, not one of the weights training learns: no loss says how many reactions one misleading post is worth. Write down who chose α and why, as you would for a threshold."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Framing a wish as an ML problem:

    1. Write the wish as the decision the system takes with the output: cache or not, spam or ok, which order.
    2. List the framings: a class, a number, an order. For each, name the label and how you would collect it.
    3. Ask what will change later: new items, a threshold that moves, a new balance between goals. Prefer the framing where that change happens in code, not in training.
    4. With two goals, train one model per goal and combine their scores with a weight in code.
    5. Check what the label rewards: could a model get it right by doing something you did not mean?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.VIEWS` holds the views a regression model predicts for 8 new videos in their first week, invented by the course. A video goes into the expensive cache when its predicted views reach the threshold. The cell counts the videos for a threshold of 50, and then for 80. Run the cell.
    """)
    return


@app.cell
def example_cache():
    views = C.VIEWS
    for threshold in [50, 80]:
        cached = 0
        for v in views:
            if v >= threshold:
                cached = cached + 1
        print(f"threshold {threshold}: {cached} of {len(views)} videos in the cache")
    return cached, threshold, views


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With a threshold of 50, 4 of the 8 videos go into the cache; with 80, 2. Moving the threshold took one number in code, and the model stayed the same. A classification model trained on "50 views or more" would know nothing about 80: it would need new labels and new training.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One score per app

        The model that predicts the next app is now framed as a regression: for each app, it gives a score from 0 to 1, how likely the user opens that app next. `next_app(apps, scores)` gets a list of `apps` and a list of their `scores` in the same order, and gives back the app with the highest score. `best` holds the position of the best app so far, starting with the first. The loop goes over every position `i` from `range(len(apps))`, and when the condition holds, `best = i`. Replace the `...` with a condition that is true when app `i` scores higher than the best app so far.

        The check calls `next_app(C.APPS, C.SCORES)`, with the 8 apps of one user. Then the same apps with a new one, Bus, and other lists.
        """),
        L.task("Keep the position of the app with the highest score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next", lines=7)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next",
        [
            "Step 3 of the recipe: with one score per app, a new app is one more item to score. Compare `scores[i]`, the score of app `i`, with the score of the best app so far.",
            "`best` is a position, not a score. How do you get the score at a position from the list `scores`?",
            "`        if scores[i] > ___:`",
        ],
        why="`scores[i] > scores[best]` compares app `i` with the best app so far; when it scores higher, `best = i` keeps its position. For `C.APPS`, that is Chat with 0.64; after the user installs Bus with 0.81, it is Bus.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_combined():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Two scores, one weight

        `combined(quality, engagement, alpha)` gets the scores of the quality model and of the engagement model, one per post, and the weight `alpha` of quality. It gives back one combined score per post, with `alpha` as the weight of quality and the rest of the weight for engagement. The loop goes over every position `i` from `range(len(quality))`, and `scores.append` adds the score of post `i` to `scores`. Replace the `...` with that score.

        The check calls `combined(C.QUALITY, C.ENGAGEMENT, 0.3)`, the 8 posts of the demo below. Then other weights and lists.
        """),
        L.task("Give each post its quality times alpha, plus its engagement times the rest of the weight."),
    ])
    return


@app.cell(hide_code=True)
def exercise_combined():
    box_combined = L.code_box("combined", lines=6)
    box_combined
    return (box_combined,)


@app.cell(hide_code=True)
def check_combined(box_combined):
    result_combined = L.check("combined", box_combined.value)
    result_combined
    return (result_combined,)


@app.cell(hide_code=True)
def hints_combined():
    L.hints(
        "combined",
        [
            "Step 4 of the recipe: one model per goal, and a weight in code. In `scores.append`, the score of post `i` starts with `alpha * quality[i]`; engagement gets the weight that is left.",
            "The two weights add up to 1. If quality weighs `alpha`, what does engagement weigh, and which item of `engagement` belongs to post `i`?",
            "`        scores.append(alpha * quality[i] + ___)`",
        ],
        why="`alpha * quality[i] + (1 - alpha) * engagement[i]` is the score of post `i`. At `alpha` 0.3, two misleading posts are in the top 3; at 0.8, none, and no model was trained again.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### One weight over the feed

    The city's app shows a feed of 8 posts, invented by the course. For each post, a quality model gives how trustworthy it is, and an engagement model how likely a user reacts to it, both from 0 to 1. Three posts are misleading. The slider sets α, the weight of quality, and the feed is ranked by α × quality + (1 − α) × engagement. Each bar shows the two parts of a post's score, as the legend says, and the dashed line marks the top 3. Nothing is trained while you move the slider.
    """)
    return


@app.cell(hide_code=True)
def demo_alpha():
    L.explore("alpha", "α", lambda _v: C.frame(_v, L.lang), start=0.0, low=0.0, high=1.0, step=0.1, role="",
              task="Find the smallest α with no misleading post in the top 3. What happens to the engagement of the top 3?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From α = 0.4 on, no misleading post is in the top 3, and the top 3 still have an engagement of 0.64 on average, against 0.88 at α = 0. From α = 0.8 on, the engagement of the top 3 drops to 0.37. Which α is right is a choice the team makes, with the people who use the app. With two models, each choice costs no training: 11 positions of the slider, 0 training runs. With one model on one combined loss, each of the 11 would have been a training run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_framing():
    mo.md(r"""
    ### Exercise 3. Which framing?

    A shop keeps a product in the warehouse near the city when it is expected to sell at least 100 times a week. The 100 changes with each season: lower in summer, when the warehouse has more room, and higher in the weeks before the holidays.
    """)
    return


@app.cell(hide_code=True)
def choice_framing_cell():
    choice_framing = L.choice(
        "framing",
        question="How do you frame the model, so that a new season needs no new training?",
        options={
            "classification": "Classification: at least 100 sales, yes or no",
            "right": "Regression: the sales per week, with 100 in code",
            "ranking": "Ranking: the products from most sold to least sold",
            "groups": "Groups of products that sell alike, found without labels",
        },
        correct="right",
        why={
            "right": "The model predicts a number, and the 100 is a line of code: a new season changes one number. That is Google's advice for a threshold that moves.",
            "classification": "The 100 is then built into the labels. What happens when the shop moves it to 120?",
            "ranking": "A ranking says which product sells more than another, not whether it sells 100 times. Which framing predicts the number of sales itself?",
            "groups": "Groups found without labels do not predict sales at all. Which framing predicts how often a product sells?",
        },
    )
    choice_framing
    return (choice_framing,)


@app.cell(hide_code=True)
def text_exercise_goals():
    mo.md(r"""
    ### Exercise 4. A new balance

    The feed of the city's app ranks posts with one model, trained on one loss: 0.7 × engagement loss + 0.3 × quality loss. The city council asks for more weight on quality, and may ask again next month.
    """)
    return


@app.cell(hide_code=True)
def choice_goals_cell():
    choice_goals = L.choice(
        "goals",
        question="Which setup makes such a change, now and next month, a change in code?",
        options={
            "again": "Train the same model again, now with 0.5 for each loss",
            "bigger": "A bigger model, trained on the same combined loss",
            "engagement": "Drop the quality loss and rank by engagement alone from now on",
            "right": "Two models, one per goal, and a weight in code",
        },
        correct="right",
        why={
            "right": "Each model learns one goal, and the weight lives in code: the council's next request is one number, and nothing is trained. These are decoupled objectives.",
            "again": "That works this month. What happens when the council asks again next month?",
            "bigger": "The weight is still inside the loss, so a new balance still means training. Where should the weight live?",
            "engagement": "Then quality has no weight at all, and the misleading posts rise to the top. Which setup keeps both goals?",
        },
    )
    choice_goals
    return (choice_goals,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In *Reliable, scalable, maintainable, adaptable*, you turned the requirements of a system into numbers with a target. This lesson went one step back: how a wish becomes an ML problem at all, and how that choice decides what you can change later without training again. [Classification: predicting a class](/course/open/ml-classification?lang=en) and [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en) gave the two framings you met most. In the next lesson, *The same packages everywhere: environments and pinning*, you make sure the code of a system runs the same on every computer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_framing, choice_goals, result_combined, result_next):
    L.wrap_up(result_next, result_combined, choice_framing, choice_goals)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
