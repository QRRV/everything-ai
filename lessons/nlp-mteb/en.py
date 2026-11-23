# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-mteb. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-mteb", "en")
    C = L.checks   # the scores of 15 embedding models on the 41 tasks of MTEB(eng, v2), the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Same meaning? Similarity and paraphrases](/course/open/nlp-sts?lang=en) and [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sts():
    quick_sts = L.choice(
        "quick-sts",
        question="In *Same meaning? Similarity and paraphrases*, Granite gave *The gate is blue.* and *The gate is yellow.* a cosine similarity of 0.951, and people gave the pair a score of 1.6 out of 5. What does that pair show?",
        options={
            "people": "People gave the pair too low a score",
            "always": "A cosine similarity above 0.9 means a paraphrase, whatever people say about the pair",
            "right": "Two sentences can get a high cosine similarity and still differ in meaning",
            "colors": "Granite cannot see colors at all",
        },
        correct="right",
        why={
            "right": "The two sentences share every word but one, and that one word changes what the gate looks like. A cosine similarity measures one thing about two vectors; this lesson compares models on many tasks, each of which measures something else.",
            "people": "On the STS scale, a 1 means the same topic, and the two sentences describe different gates. The score is right; the cosine similarity is high because the two sentences share every word but one.",
            "always": "The pair shows the opposite: 0.951 and a score of 1.6. A cosine similarity that is high says that two vectors point the same way, not that two sentences mean the same. See [Same meaning? Similarity and paraphrases](/course/open/nlp-sts?lang=en).",
            "colors": "Then the two sentences would get exactly the same vector, and a cosine similarity of 1. They get 0.951: the one word that differs moves the vector a little, and the meaning a lot.",
        },
    )
    quick_sts
    return (quick_sts,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="In *Sums and means along an axis*, `m` is `np.array([[1, 2, 3], [4, 5, 6]])`. What does `m.mean(axis=1)` give?",
        options={
            "columns": "`array([2.5, 3.5, 4.5])`: one mean per column",
            "all": "`3.5`: the mean of all six numbers",
            "first": "`array([1, 4])`: the first of each row",
            "right": "`array([2., 5.])`: one mean per row",
        },
        correct="right",
        why={
            "right": "`axis=1` works along each row: (1 + 2 + 3) / 3 = 2 and (4 + 5 + 6) / 3 = 5.",
            "columns": "That is `axis=0`, which works down the rows: one result per column. `axis=1` works along each row: what does that give for the rows 1, 2, 3 and 4, 5, 6?",
            "all": "That is `m.mean()` without an axis. With `axis=1`, NumPy takes a mean along each row: one number per row.",
            "first": "`m[:, 0]` gives the first column. `mean(axis=1)` takes the mean along each row. See [Sums and means along an axis](/course/open/np-axis?lang=en).",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Everything that works with sentence embeddings starts with a choice: which model makes the vectors? Hundreds of models that turn texts into vectors are free to download, from 20 million parameters to billions. A leaderboard ranks them by one number, and the top of the list is the obvious choice. It can also be the wrong one: built for another kind of task, too big for your computer, or never tested on your language.

    **An everyday example.** A restaurant guide gives each restaurant one score for food, service, price and atmosphere together. If all you want is a quick, cheap lunch, the restaurant with the highest score may be the wrong place.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="A benchmark of embedding models measures them on tasks of several types, such as STS, where the cosine similarities must follow people's scores, as in the previous lesson, and retrieval, where the vectors must find, for a question, the texts that answer it. Of nine models up to 300 million parameters, paraphrase-multilingual-mpnet-base-v2 scores best on the STS tasks. Where does it land on the retrieval tasks, among the same nine?",
        options={
            "first": "1st: finding the answer to a question also asks which texts are alike in meaning",
            "third": "3rd",
            "fifth": "5th, in the middle",
            "right": "8th, second to last",
        },
        reveal="**8th, second to last:** 37.1 on retrieval, against 55.7 for granite-embedding-125m-english, the best of the nine there, while on STS it is first with 80.5. A model that orders pairs of sentences well can be weak at finding documents for a question. The idea shows how one mean hides this, and how to read past it.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One benchmark, many tasks: MTEB

    A model that turns a text into one vector, such as Granite in the previous lesson, is called an **embedding model**. **MTEB**, the Massive Text Embedding Benchmark, compares embedding models on many tasks at once. Niklas Muennighoff, Nouamane Tazi, Loïc Magne and Nils Reimers published it in 2023: "MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages." Each **task** gives a model the texts of one dataset, uses the vectors in a fixed way, and measures the result with one number, the task's **main score**. Tasks that use the vectors in the same way form a **task type**, with the same main score:

    | Task type | What the vectors must do | Main score |
    |---|---|---|
    | retrieval | find, for a query, the documents that answer it among many: those whose vectors have the highest cosine similarity with the query's | nDCG@10: how high the right documents stand among the first 10 |
    | reranking | put a short list of candidates for a query in order | MAP: how high the right candidates stand, as a mean over the queries |
    | classification | serve as the features of a logistic regression that learns the labels of the texts | accuracy |
    | clustering | form groups that match the labels of the texts, with k-means: a method that puts vectors that lie close together in one group | V-measure: how well the groups and the labels match |
    | pair classification | decide whether two texts are duplicates or paraphrases, with a threshold on their cosine similarity | average precision: the precision at each threshold where one more real duplicate turns up, averaged |
    | STS | give pairs of sentences cosine similarities in the order of people's scores, as in the previous lesson | Spearman correlation: how well the two orders agree |
    | summarization | judge summaries a machine made by their cosine similarity with summaries people wrote, in the order of people's judgments | Spearman correlation |

    The eighth type of MTEB, bitext mining, pairs sentences of two languages, and has no place in an English benchmark. nDCG and MAP come back in later lessons, about search engines. The leaderboard shows every main score times 100, so most scores run from 0 to 100, and higher is better.

    ### One number for the leaderboard

    MMTEB, the multilingual successor by Kenneth Enevoldsen and many others (2025), has "over 500 quality-controlled evaluation tasks across 250+ languages". The English benchmark of its leaderboard, **MTEB(eng, v2)**, takes 41 English tasks of seven types: 10 retrieval, 2 reranking, 8 classification, 8 clustering, 3 pair classification, 9 STS and 1 summarization. The STS Benchmark of the previous lesson is one of the 9 STS tasks. A **leaderboard** ranks models by one number, and this one offers two:

    - **Mean(Task)** is the mean over all 41 tasks. In the words of the leaderboard: "This is a naïve average computed across all the tasks within the benchmark." A type with many tasks counts more: retrieval counts for 10 of the 41, summarization for 1. The leaderboard adds: "However, the mean can overvalue tasks with higher variance in its scores." On a retrieval task, the scores of the 15 models of this lesson have a standard deviation of 12.8 points on average; on a reranking task, of 2.8.
    - **Mean(TaskType)** first takes the mean of each type, and then the mean of those seven type means: each type counts one seventh.

    Both are **weighted means** of the seven type means: each type mean times its weight, added up, and divided by the sum of the weights. Mean(TaskType) gives every type the weight 1. Mean(Task) gives each type the number of its tasks: 10 for retrieval, 1 for summarization. A model with 40 on retrieval and 80 on STS gets, with the weights 3 and 1, (3 × 40 + 1 × 80) / 4 = 50.

    ### What one number hides

    The table shows the nine models up to 300 million parameters, of the 15 that the course ships the scores of.
    """)
    return


@app.cell(hide_code=True)
def idea_board():
    mo.Html(C.board_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_board():
    mo.md(r"""
    granite-embedding-125m-english is first on both means, and first on retrieval, reranking and clustering. On classification and on pair classification it is 8th of the nine. multilingual-e5-base is first on classification, multilingual-e5-small on pair classification, paraphrase-multilingual-mpnet-base-v2 on STS and granite-embedding-278m-multilingual on summarization. The two means do not agree everywhere either: granite-embedding-30m-english is 4th by Mean(Task) and 5th by Mean(TaskType), multilingual-e5-small the other way round. The lines below follow the places of each model.
    """)
    return


@app.cell(hide_code=True)
def idea_ranks():
    mo.Html(C.ranks_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_read():
    mo.md(r"""
    Muennighoff and others saw the same on their 33 models: "No model claims the state-of-the-art in all seven English tasks." And: "Models geared towards STS (SimCSE, ST5, SGPT-nli) perform badly on retrieval tasks."

    The table leaves out the six bigger models of the 15. The biggest, Qwen3-Embedding-8B, has 7.6 billion parameters, 333 times as many as all-MiniLM-L6-v2, and more parameters take more memory, and more time for each text.

    ### Read the tasks that look like yours

    The blog post that introduced the current MTEB leaderboard, in 2026, puts it this way: "When selecting a benchmark, it is worth knowing that it rarely measures what you care about." And: "you can often do a lot better by customizing the leaderboard specifically to your use case." So decide what your system does with the vectors, give each task type a weight for how much it looks like that, and rank the models by the weighted mean with your weights.

    Your language counts too: MTEB(eng, v2) holds English tasks only. For Dutch there is MTEB-NL, by Nikolay Banar and others (2026), with 40 Dutch datasets. There, the course's own Granite, granite-embedding-107m-multilingual, comes 25th of 45 models on the mean over the task types, with 57.0; the best model, Qwen3-Embedding-4B, gets 69.2. granite-embedding-125m-english, the best of the nine on the English tasks, was made for English only. The course uses its own Granite because it is small and knows Dutch, one of its 12 languages.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **embedding model** | a model that turns a text into one vector, such as Granite |
        | **MTEB** | the Massive Text Embedding Benchmark: tasks that compare embedding models; MTEB(eng, v2) is the English benchmark of its leaderboard, with 41 tasks |
        | **task** | one dataset, a fixed way to use the vectors of its texts, and one main score |
        | **main score** | the one number that measures a model on a task, such as accuracy for classification |
        | **task type** | tasks that use the vectors in the same way, with the same main score: retrieval, reranking, classification, clustering, pair classification, STS and summarization |
        | **leaderboard** | a table that ranks models by their scores on a benchmark |
        | **Mean(Task)** | the mean of a model's main scores over all tasks of a benchmark |
        | **Mean(TaskType)** | the mean of a model's type means: every task type counts the same |
        | **weighted mean** | each number times its weight, added up, and divided by the sum of the weights |
        | **Spearman correlation** | how well two orders agree, from −1 to 1: 1 when both put the same things in the same order |
        | **zero-shot** | of a model on a task: not trained on the data of that task |
        | `scores[:, types == "STS"]` | all rows of `scores`, and of the columns only those where the mask is True |
        | `300_000_000` | a number with `_` between groups of digits, to read it more easily; Python skips the `_` |
        | `np.array(columns)` | turns a list of arrays of the same length into one table, one row per array, as with a list of lists |
        """),
        L.watch_out("A model can be trained on the data of a task, and then its score on that task says less about texts it has never seen. MMTEB meant MTEB(eng, v2) as a zero-shot benchmark, for models that were not trained on its tasks: \"To prevent overfitting, we intend it as a zero-shot benchmark.\" Of MTEB itself, Muennighoff and others write: \"we ignore overlap of MTEB datasets with model training datasets in MTEB scores.\" The leaderboard marks which tasks each model was trained on: for the four Granite models of this lesson, 5 of the 41, among them AskUbuntuDupQuestions, one of the two reranking tasks; for the Qwen3 and e5 models and GritLM-7B, 2. Without those five tasks, granite-embedding-125m-english stays first of the nine by Mean(Task), with 61.6 against 61.4 for multilingual-e5-base."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down what your system does with the vectors, and in which language.
    2. Give each task type a weight for how much it looks like that: 0 for what your system never does.
    3. Compute each model's mean per task type from its scores per task.
    4. Rank the models by the weighted mean of their type means, with your weights.
    5. Cross off what does not fit: too many parameters for your computer, not your language, or trained on the tasks you weight most.
    6. Try the best two or three on a few hundred examples of your own.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.mteb()` gives five values: `scores`, an array with one row per model and one column per task, each number the model's main score on the task times 100; `models`, the 15 names, as on Hugging Face; `tasks`, the 41 names of the tasks; `types`, an array with the task type of each column; and `parameters`, the number of parameters of each model. The code counts tasks of two types, finds the model with the highest Mean(Task), counts the models up to 300 million parameters, and keeps the columns of the STS tasks.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    scores, models, tasks, types, parameters = C.mteb()
    print(scores.shape)
    print(np.sum(types == "Retrieval"), np.sum(types == "Summarization"))
    print(models[scores.mean(axis=1).argmax()])
    small = parameters <= 300_000_000
    print(np.sum(small))
    sts = scores[:, types == "STS"]
    print(sts.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    15 models and 41 tasks, of which 10 are retrieval tasks and 1 is a summarization task. Over all 41 tasks, Qwen/Qwen3-Embedding-8B has the highest mean: the first place by Mean(Task). Nine models have up to 300 million parameters. `scores[:, types == "STS"]` keeps all 15 rows and the 9 columns of the STS tasks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which task type looks like it?

    Each line is a use of an embedding model. Put each use under the task type of MTEB that looks most like it.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_uses = L.sort("use-cases")
    sort_uses
    return (sort_uses,)


@app.cell(hide_code=True)
def check_sort(sort_uses):
    result_sort = L.check("use-cases", sort_uses)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    A real system often does more than one of these, with the same model. The next exercise computes each model's mean per task type; the one after it weighs the types.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. type_means: one mean per task type

        Write `type_means(scores, types, kinds)`: `scores` holds the scores of each model on each task, one row per model and one column per task; `types` the task type of each column; `kinds` a list of task types. Give back an array with one row per model and one column per type of `kinds`, in that order: the mean of the model's scores on the tasks of that type.

        For `type_means(np.array([[60.0, 70.0, 40.0], [80.0, 60.0, 30.0]]), np.array(["STS", "STS", "Retrieval"]), ["Retrieval", "STS"])` the result is `[[40.0, 65.0], [30.0, 70.0]]`: the first model scores 40 on its one retrieval task and 65 on average on its two STS tasks. With `["STS", "Retrieval"]` the two columns swap. The check calls both, then cases of its own: `kinds` with only some of the types, and the scores of `C.mteb()` with `C.KINDS`, the seven types in the order of this lesson, and with two types.
        """),
        L.task("Give back the mean score per model and per type of kinds, one row per model and one column per type."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("type_means", lines=10)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("type_means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "type_means",
        [
            "Start with an empty list `columns`, and go through `kinds` with a loop. For each `kind`, add one column to `columns` with `append`: the mean of every model on the tasks of that kind. At the end, `np.array(columns)` has one row per type; turn it with `.T`.",
            "The worked example keeps the STS columns with `scores[:, types == \"STS\"]`. What stands there instead of `\"STS\"` in your loop?",
            r"`        columns.append(___)`",
        ],
        why="`columns = []`, then `for kind in kinds:` with `columns.append(scores[:, types == kind].mean(axis=1))` in it; last, `return np.array(columns).T`. For the example: retrieval keeps the third column, with the means `[40, 30]`; STS the first two, with the means `[65, 70]`. `np.array(columns)` is then `[[40, 30], [65, 70]]`, one row per type, and `.T` turns it into `[[40, 65], [30, 70]]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_means():
    mo.md(r"""
    With all 15 models, Qwen3-Embedding-8B has the highest number in five of the seven columns, and e5-mistral-7b-instruct in the other two. Further down, the rows of the small models cross: granite-embedding-125m-english has 55.65 on retrieval, more than the 48.97 of multilingual-e5-base, and 68.29 on classification, less than the 73.81 of multilingual-e5-base.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. best_for: the best model for your weights

        Write `best_for(means, weights, models)`: `means` holds the type means of each model, one row per model and one column per type, as `type_means` gives them; `weights` one weight per column; `models` the names of the models, in the order of the rows. Give back two values: the name of the model with the highest weighted mean, and that weighted mean.

        For `best_for(np.array([[50.0, 80.0], [60.0, 60.0]]), np.array([3.0, 1.0]), ["a", "b"])` the result is `("b", 60.0)`: model a gets (3 × 50 + 1 × 80) / 4 = 57.5, and model b (3 × 60 + 1 × 60) / 4 = 60. With the weights 1 and 1, a wins with 65.0. The check calls both, then cases of its own: a weight of 0, the type means of the 15 models of MTEB with the weight 1 for every type and with the number of tasks of each type, and those of the nine models up to 300 million parameters.
        """),
        L.task("Give back the name of the model with the highest weighted mean, and that weighted mean."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_for", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best_for", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_for",
        [
            "Step 4 of the recipe: compute `total`, the weighted mean of every row of `means` at once, as the weighted sums of the rows divided by `np.sum(weights)`. Then `total.argmax()` gives the place of the highest, `best`; give back `models[best]` and `total[best]`.",
            "In the example, `np.sum(weights)` is 4. Which product of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en) gives the weighted sums 230 and 240 of the two rows in one go?",
            r"`    total = ___ / np.sum(weights)`",
        ],
        why="`total = means @ weights / np.sum(weights)`: `means @ weights` is the dot product of every row with `weights`, the weighted sum of each model. Then `best = total.argmax()` and `return models[best], total[best]`. For the example: `means @ weights` is `[230, 240]`, divided by 4 that is `[57.5, 60]`, and the highest stands at place 1, so `(\"b\", 60.0)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    Among the nine small models, your function chooses granite-embedding-125m-english for Mean(TaskType), Mean(Task), the search box and grouping by topic. For sorting mail it chooses multilingual-e5-base, for finding duplicates multilingual-e5-small, and for scoring similarity paraphrase-multilingual-mpnet-base-v2. Among all 15, Qwen3-Embedding-8B wins six of the seven weightings, and e5-mistral-7b-instruct the one for duplicates. Mean(Task) gives Qwen3-Embedding-8B 75.22 and Mean(TaskType) 68.71: the same model at another height, because summarization and reranking, where every model scores low, weigh 1 and 2 of 41 in Mean(Task), and a seventh each in Mean(TaskType).
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Try it: weights of your own

    Give each task type a weight from 0 to 3 for a use of your own, and choose how big a model may be. The table ranks the models by your weighted mean, and shows how far your weighting moves each model from its place by Mean(TaskType). Tick the box to leave out the five tasks that some of the 15 models were trained on. Among the models up to 300 million parameters, how high can you get the course's own Granite?
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_weights = mo.ui.array([mo.ui.slider(0, 3, step=1, value=1, label=name, show_value=True) for name in C.type_labels(L.lang)])
    explore_limit = mo.ui.dropdown(options=list(C.LIMITS[L.lang]), value=list(C.LIMITS[L.lang])[0], label="Size")
    explore_zero = mo.ui.checkbox(value=False, label="Only tasks none of the models was trained on")
    mo.vstack([explore_weights.vstack(gap=0.25), mo.hstack([explore_limit, explore_zero], justify="start", gap=2)])
    return explore_limit, explore_weights, explore_zero


@app.cell(hide_code=True)
def explore(explore_limit, explore_weights, explore_zero):
    mo.Html(C.explore_picture(explore_weights.value, explore_limit.value, explore_zero.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    No weighting from 0 to 3 puts granite-embedding-107m-multilingual first among the nine, with or without the five tasks: 3rd is the highest it gets, for example with retrieval 1, reranking 1 and summarization 2 on all 41 tasks. For English texts, other small models fit better; the course needs one that also knows Dutch.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Same meaning? Similarity and paraphrases](/course/open/nlp-sts?lang=en) measured Granite on one task, STS. MTEB measures embedding models on many tasks at once, and a leaderboard ranks them by one mean. With a mean along an axis, as in [Sums and means along an axis](/course/open/np-axis?lang=en), you computed the mean per task type yourself, and with a weighted mean you ranked the models for a use of your own. The next lesson of the domain, *A label per word: sequence labeling and BIO*, turns from whole sentences to single words: a label for every word.

    **An everyday example.** A decathlon crowns the best athlete over ten events. A relay team that needs a sprinter looks at the 100 meters, and the winner of the decathlon may not be the fastest there.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_best, result_means, result_sort):
    L.wrap_up(result_sort, result_means, result_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
