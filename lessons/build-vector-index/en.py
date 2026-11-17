# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-vector-index. Route: course/route/lessen/bouwen.yaml. Checks, the index and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-vector-index", "en")
    C = L.checks   # the knowledge base, Granite's embeddings of it, the course's index, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_centroid():
    quick_centroid = L.choice(
        "quick-centroid",
        question="k-means is done. What is the centroid of a cluster?",
        options={
            "first": "The example that k-means picked first for that cluster",
            "far": "The example of the cluster that lies farthest from the other clusters",
            "mean": "The mean of its examples, feature by feature",
        },
        correct="mean",
        why={
            "mean": "Every move step puts a centroid at the mean of the examples in its cluster, and when k-means is done, that step changes nothing any more. So the centroid is the center point of its cluster, usually not an example itself.",
            "first": "k-means can start from examples picked at random, but every move step puts the centroids somewhere else. See [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "far": "A centroid lies in the middle of its own cluster, not at its edge. See [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
        },
    )
    quick_centroid
    return (quick_centroid,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    The search of [Search by meaning](/course/open/build-retrieval-embeddings?lang=en) compares the question with every point: {points} dot products of {dims} numbers, {mult} multiplications for each question. For the course, that is nothing. The knowledge base of a company can have a million chunks: {mmult} multiplications for each question, and {gb} GB only to hold their embeddings in float32. A search engine has billions. An index compares the question with only a small part of the vectors, chosen so that the nearest are almost always among them.

    **An everyday example.** In a library you do not walk past every book. You read the signs above the sections, go to the section that fits your question best, and look only on its shelves. Sometimes the best book stands in the next section, and then you miss it, unless you look there too. That is only a comparison: the search below compares lists of numbers.
    """.format(points=C.number(_f["points"], L.lang), dims=C.DIMS, mult=C.number(_f["multiplications"], L.lang),
               mmult=C.number(_f["million_multiplications"], L.lang), gb=C.number(_f["million_gb"], L.lang, 1)))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **vector index** | a structure, built once over the vectors, that makes searching them fast; comparing with every vector, as in [Search by meaning](/course/open/build-retrieval-embeddings?lang=en), is called *flat search* |
    | **approximate nearest neighbor search** | finding vectors that are very likely among the nearest, by skipping most comparisons; now and then a nearer vector is missed |
    | **IVF index** | an index that puts the vectors in clusters with k-means, and compares a question only with the vectors in the few clusters whose centroids are nearest; IVF stands for *inverted file* |
    | **HNSW** | an index that links every vector to a few vectors near it, on several levels; a search goes from vector to vector, each time to the one nearest the question |
    | `probes` | the number of clusters a search looks in |

    The course built an IVF index of its {points} points once. k-means put their embeddings in {clusters} clusters, of {small} to {large} points each, and a list per cluster holds the numbers of its points. A search then takes two steps. First it compares the question with the {clusters} centroids, by distance as k-means does, and takes the `probes` nearest. Then it compares the question, by dot product, with the points in the lists of those clusters only. With `probes` 1, that is about {per} points instead of {points}.

    The best point is not always in the cluster with the nearest centroid. A point can lie at the edge of its cluster, nearer to the question than every point of a cluster whose centroid is nearer. That is why the search is *approximate*: the more clusters it looks in, the less often it misses a point, and the more vectors it compares.

    IVF is the simplest index. Most vector databases, databases made to keep and search embeddings, use HNSW, which usually finds the same vectors with fewer comparisons, but needs more memory for all its links. Faiss, a library from Meta, has flat search, IVF and HNSW ready-made, for a million vectors and more; it calls `probes` *nprobe*. The idea stays the same: compare with few vectors, and measure what you miss.
    """.format(points=C.number(_f["points"], L.lang), clusters=_f["clusters"], small=_f["smallest"], large=_f["largest"],
               per=C.number(round(_f["points"] / _f["clusters"]), L.lang)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: search with an IVF index

    1. **Once:** cluster the vectors with k-means, and keep the centroids.
    2. **Once:** make a list per cluster with the numbers of its vectors.
    3. **Clusters:** compute the distance from the question to every centroid, and take the `probes` nearest clusters.
    4. **Score:** take the dot product of the question with the vectors in the lists of those clusters only.
    5. **Keep:** the `k` with the highest score, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{question}*

    1. **Clusters:** of the {clusters} centroids, the nearest three lie at distances {d0}, {d1} and {d2}.
    2. **One cluster:** the nearest cluster has {s0} points. With `probes` 1, the search compares the question with {c1} vectors: the {clusters} centroids and those {s0} points. Its three best are points of *{o0}*, *{o1}* and *{o2}*: about the slope of a loss, not about the shape of its curve.
    3. **Three clusters:** the next two add {s1} and {s2} points, {c3} vectors in all. Now the first is a point of *{t0}*: *{p0}* It answers the question.

    That point lies in the third nearest cluster: its centroid is farther from the question, but the point itself is nearer than every point of the nearest cluster. With three clusters, the search finds the same three points as a search through every vector, and compares {c3} vectors instead of {points}.
    """.format(question=_w["question"]["text"][L.lang], clusters=C.CLUSTERS, d0=C.number(_w["distances"][0], L.lang, 3),
               d1=C.number(_w["distances"][1], L.lang, 3), d2=C.number(_w["distances"][2], L.lang, 3), s0=_w["sizes"][0],
               s1=_w["sizes"][1], s2=_w["sizes"][2], c1=_w["compared_one"], c3=_w["compared_three"],
               o0=C.title_of(_w["one"][0], L.lang), o1=C.title_of(_w["one"][1], L.lang), o2=C.title_of(_w["one"][2], L.lang),
               t0=C.title_of(_w["three"][0], L.lang), p0=C.POINTS[_w["three"][0]], points=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course searched its index for all {questions} questions, also the {none} that no point answers, looking in 1 to {clusters} clusters, and compared the three points it found with the three of a search through every vector. Guess first:
    """.format(questions=C.findings()["questions"], none=C.findings()["questions"] - C.findings()["answerable"], clusters=C.CLUSTERS))
    return


@app.cell(hide_code=True)
def guess_one_cell():
    _f = C.findings()
    guess_one = L.guess(
        "one-cluster",
        question="Looking in 1 of the {clusters} clusters, for how many of the {questions} questions does the index find the same three points as a search through every vector?".format(
            clusters=C.CLUSTERS, questions=_f["questions"]),
        options={
            "all": "For all of them, or almost all",
            "most": "For about three quarters",
            "third": "For about a third",
        },
        reveal="**For {one} of the {questions}**, about a third. For {best} questions, the best point is in the nearest cluster, but the three best are all there for only {one}: the second and the third are often in a cluster next door.".format(
            one=_f["one"][1], questions=_f["questions"], best=_f["best_nearest"]),
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def picture_curve():
    mo.Html(C.curve_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    _more = C.more_clusters()
    mo.vstack([
        mo.md(r"""
        Looking in 1 cluster, the index compared {c1} vectors on average and found the same three for {s1} of the {questions} questions; in 2 clusters, {c2} vectors and {s2}; in 4, {c4} and {s4}; in 7, {c7} and {s7}. From {at} clusters on it found the same three for all {questions}, but then it compared {call} vectors, more than three quarters of all {points}. For an answer, what counts is whether the right point is among the three: through every vector it was for {gold} of the {answerable} questions that a point answers, looking in 1 cluster for {g1}, in 7 for {g7}.

        With {more} clusters, the lists are shorter: looking in 1 cluster, the index compared {cm} vectors, and found the same three for only {sm}. On {points} points an index gains little. With a million vectors in {big} clusters of about {per} each, looking in {probes} clusters compares about {big_c} vectors: about one in a hundred.
        """.format(c1=_f["one"][2], s1=_f["one"][1], questions=_f["questions"], c2=_f["two"][2], s2=_f["two"][1], c4=_f["four"][2],
                   s4=_f["four"][1], c7=_f["seven"][2], s7=_f["seven"][1], at=_f["all_at"], call=_f["all_compared"],
                   points=C.number(_f["points"], L.lang), gold=_f["flat_gold"], answerable=_f["answerable"], g1=_f["one"][3],
                   g7=_f["seven"][3], more=C.MORE, cm=_more[1], sm=_more[0], big=C.number(C.BIG_CLUSTERS, L.lang),
                   per=C.number(C.MILLION // C.BIG_CLUSTERS, L.lang), probes=C.BIG_PROBES, big_c=C.number(_f["big_compared"], L.lang))),
        L.watch_out("An index misses quietly: it still gives `k` points, and nothing in them says that a nearer point was skipped. Before you rely on an index, measure on your own questions how often it finds the same points as a search through every vector, and choose the number of clusters to look in from that."),
    ])
    return


@app.cell(hide_code=True)
def text_lists():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A list per cluster

        `make_lists(labels, n)` should give a list of `n` lists: list `c` holds the numbers `i` of the points whose label is `c`, from small to large. `labels` holds the cluster of every point, as `km.labels_` of k-means does: a list or a NumPy array; `n` is the number of clusters. `make_lists([0, 1, 0, 2, 1], 3)` gives `[[0, 2], [1, 4], [3]]`: points 0 and 2 are in cluster 0. The check also draws the lists your function makes of the course's {points} points.
        """.format(points=C.number(len(C.POINTS), L.lang))),
        L.task("Give cluster `c` the numbers `i` of the points whose label is `c`, instead of an empty list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lists():
    box_lists = L.code_box("make_lists", lines=5)
    box_lists
    return (box_lists,)


@app.cell(hide_code=True)
def check_lists(box_lists):
    result_lists = L.check("make_lists", box_lists.value)
    result_lists
    return (result_lists,)


@app.cell(hide_code=True)
def hints_lists():
    L.hints(
        "make_lists",
        [
            "For cluster `c`, go through the numbers `i` of all points, `range(len(labels))`, and keep those whose label is `c`.",
            "A comprehension with a condition does that in one line: `[i for i in range(len(labels)) if ...]`.",
            "In the loop: `lists.append([i for i in range(len(labels)) if ___])`.",
        ],
        why="""```python
def make_lists(labels, n):
    lists = []
    for c in range(n):
        lists.append([i for i in range(len(labels)) if labels[i] == c])
    return lists
```

Because the loop goes over all `n` clusters, a cluster that no point is in still gets its list, an empty one. One loop over the points does it too: start with `n` empty lists, and add every `i` to list `labels[i]`.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Look only in the nearest clusters

        `ivf_search(q, index, k, probes)` should give the numbers of the `k` points with the highest dot product with `q`, among the points of the `probes` clusters whose centroids are nearest to `q`. `index` is a dict with three parts: `"centroids"`, an array with a centroid per row; `"lists"`, the lists of `make_lists`; and `"vectors"`, the embeddings, a row per point. The box already finds the nearest clusters, `nearest`, but then compares `q` with every vector. The small index of the example has six vectors of two numbers in three clusters:

        ```python
        {small}
        ```

        `ivf_search(np.array([0.6, 0.8]), SMALL, 2, 1)` looks only in cluster 1, the nearest, and gives `[2, 3]`. Looking in 2 clusters, it gives `[2, 0]`, the same as a search through every vector: point 0, in cluster 0, fits better than point 3. The check also runs your function on the course's index, for the question of the worked example, and counts the vectors it compares.
        """.format(small=C.SMALL_CODE.replace("\n", "\n        "))),
        L.task("Compare `q` only with the points in the lists of the clusters in `nearest`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("ivf_search", lines=10)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("ivf_search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "ivf_search",
        [
            "`index[\"lists\"]` has a list for each cluster, with the numbers of its points; you need those of every cluster in `nearest`.",
            "Two loops, one inside the other: one over the clusters in `nearest`, and in it one over the numbers of the points of that cluster.",
            "In place of the loop over `range(len(index[\"vectors\"]))`: `for c in nearest:`, and in it `for i in ___:`, with the line with `found.append(...)` in the inner one.",
        ],
        why="""```python
import numpy as np


def ivf_search(q, index, k, probes):
    distances = np.linalg.norm(index["centroids"] - q, axis=1)
    nearest = np.argsort(distances)[:probes]
    found = []
    for c in nearest:
        for i in index["lists"][c]:
            found.append((index["vectors"][i] @ q, i))
    return [i for score, i in sorted(found, reverse=True)[:k]]
```

The picture under the check counts the vectors your function really compares. A function that first computes `index["vectors"] @ q` for all points and then picks the clusters gives the same numbers, but compares all {points}: then the index saves nothing.""".format(points=C.number(len(C.POINTS), L.lang)),
    )
    return


@app.cell(hide_code=True)
def text_find_more():
    mo.md(r"""
    ### Exercise 3. Find the same more often

    Look again at the picture of what the course measured.
    """)
    return


@app.cell(hide_code=True)
def choice_find_more_cell():
    _f = C.findings()
    _more = C.more_clusters()
    choice_find_more = L.choice(
        "find-more",
        question="Looking in 1 of the {clusters} clusters, the index finds the same three points as a search through every vector for {one} of the {questions} questions. What makes it find the same more often?".format(
            clusters=C.CLUSTERS, one=_f["one"][1], questions=_f["questions"]),
        options={
            "k": "Ask for more points: a larger `k`",
            "probes": "Look in more clusters: a larger `probes`",
            "clusters": "Let k-means make more clusters: {more} instead of {clusters}".format(more=C.MORE, clusters=C.CLUSTERS),
        },
        correct="probes",
        why={
            "probes": "Every cluster more adds its points to those the search compares. In 4 clusters the index found the same three for {s4} of the {questions}, from {at} clusters on for all; the price is the vectors it compares: {c4} and {call} on average, against {c1} in 1 cluster.".format(
                s4=_f["four"][1], questions=_f["questions"], at=_f["all_at"], c4=_f["four"][2], call=_f["all_compared"], c1=_f["one"][2]),
            "k": "A larger `k` gives more points from the same clusters. A point in a cluster the search does not look in is still never compared.",
            "clusters": "More clusters make every list shorter, so in 1 cluster the search compares fewer points, and misses more: with {more} clusters, {cm} vectors and the same three for only {sm} of the {questions}.".format(
                more=C.MORE, cm=_more[1], sm=_more[0], questions=_f["questions"]),
        },
    )
    choice_find_more
    return (choice_find_more,)


@app.cell(hide_code=True)
def text_quiet_miss():
    mo.md(r"""
    ### Exercise 4. Does the index miss anything?

    A colleague has built a RAG system with an IVF index. It gives three points for every question, so the colleague says the index misses nothing.
    """)
    return


@app.cell(hide_code=True)
def choice_quiet_miss_cell():
    _w = C.worked()
    choice_quiet_miss = L.choice(
        "quiet-miss",
        question="What do you answer?",
        options={
            "flat": "Search some questions through every vector too, and compare",
            "scores": "Look at the scores: when the index misses the best point, they are low",
            "right": "That is right: an index that gives `k` points has found the `k` nearest",
        },
        correct="flat",
        why={
            "flat": "Only a search through every vector shows what the index skipped. Measured on the course's {questions} questions, looking in 1 cluster the index found the same three for only {one}, while it gave three points for every question.".format(
                questions=C.findings()["questions"], one=C.findings()["one"][1]),
            "scores": "For the question of the worked example, the best score looking in 1 cluster was {one}, through every vector {all}. Does {one} look like a miss, if you do not know {all}?".format(
                one=C.number(_w["best_one"], L.lang, 3), all=C.number(_w["best_all"], L.lang, 3)),
            "right": "An IVF index always gives the `k` best of the points it compared, also when a nearer point is in a cluster it did not look in. Look again at the worked example.",
        },
    )
    choice_quiet_miss
    return (choice_quiet_miss,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Search the course with your own ivf_search

    Once exercises 1 and 2 are right, pick a question of the course and the number of clusters to look in. The course searches its index with your `ivf_search`, and shows below it what a search through every vector finds. Look for a question for which 1 cluster is not enough.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q07")["text"][L.lang], label="Question", full_width=True)
    probes_slider = mo.ui.slider(start=1, stop=C.CLUSTERS, step=1, value=1, label="Clusters to look in", show_value=True)
    mo.vstack([question_pick, probes_slider])
    return probes_slider, question_pick


@app.cell(hide_code=True)
def demo_search(box_lists, box_search, probes_slider, question_pick):
    _fn = C.learner_search(box_lists.value, box_search.value)
    if _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the search uses your own `ivf_search`.*")
    else:
        _out = mo.Html(C.searched(_fn, question_pick.value, [probes_slider.value], L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** An index changes how fast the points are found, not what happens to them next: they go into the prompt as in [Your first RAG](/course/open/build-rag-first?lang=en). In [Filters and vector databases](/course/open/build-metadata-filters?lang=en), every point also carries metadata, such as its language and its lesson, and a search keeps only the points that fit; a vector database keeps embeddings, metadata and index together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_find_more, choice_quiet_miss, result_lists, result_search):
    L.wrap_up(result_lists, result_search, choice_find_more, choice_quiet_miss)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
