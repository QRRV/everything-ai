# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-image-search. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-image-search", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en) and about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="What does `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` give for `a = np.array([1, 2])` and `b = np.array([2, 4])`?",
        options={
            "right": "1.0",
            "dot": "10.0",
            "ratio": "2.0",
            "half": "0.5",
        },
        correct="right",
        why={
            "right": "`b` is `a` times 2: the same direction, so the cosine similarity is 1, whatever the lengths. Today it compares photos.",
            "dot": "10 is the dot product, 1 · 2 + 2 · 4. The cosine similarity divides it by the two norms, √5 and √20, whose product is 10. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "ratio": "`b` is twice as long as `a`, but the cosine similarity only looks at the direction, and that is the same. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "half": "The lengths drop out of a cosine similarity, and `a` and `b` point the same way. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_argsort():
    quick_argsort = L.choice(
        "quick-argsort",
        question="What does `np.argsort(np.array([0.3, 0.9, 0.1]), kind=\"stable\")` give?",
        options={
            "right": "`[2, 0, 1]`",
            "values": "`[0.1, 0.3, 0.9]`",
            "largest": "`[1, 0, 2]`",
            "ranks": "`[1, 2, 0]`",
        },
        correct="right",
        why={
            "right": "The positions from the smallest to the largest: 0.1 is at position 2, then 0.3 at 0 and 0.9 at 1. Today you need the largest first.",
            "values": "Those are the values themselves, sorted, as `sorted` gives them. `np.argsort` gives their positions. See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "largest": "That is from the largest to the smallest. `np.argsort` starts at the smallest, as for the k nearest in k-NN. See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "ranks": "That is the place each value gets in the sorted order. `np.argsort` gives the positions in the sorted order: first the position of the smallest. See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
        },
    )
    quick_argsort
    return (quick_argsort,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You have one photo, and you want the photos that look like it from a large collection in which nobody wrote down what each photo shows: your own photos, the catalog of a shop, an archive that may hold the same photo twice. A classifier does not help: it knows only the classes it learned, and someone has to label photos for it. What helps is comparing the photo with every photo in the collection, and keeping the ones most like it.

    **An everyday example.** Someone shows you a photo of a mushroom from their walk, and you leaf through a guide for the pictures that look most like it. You compare how they look, not their names; the names come after, from the pictures you found.
    """)
    return


@app.cell(hide_code=True)
def predict_five():
    guess_five = L.guess(
        "five",
        question="For each of the 300 photos of the previous lesson, you look for the 5 photos most like it among the 299 others, by the cosine similarity of their DINOv2 features. How many of those 5 show the same kind of object, on average?",
        options={
            "one": "About 1",
            "two": "About 2",
            "three": "About 3.5",
            "almost": "Almost 5",
        },
        reveal=f"**Almost 5: {L.number(C.REFERENCE['of5']['features'], 2)} on average,** without a single label. For 8 of the 10 classes it is 5 or nearly; lotus gets {L.number(C.REFERENCE['class5']['features'][3], 2)} and water lily {L.number(C.REFERENCE['class5']['features'][8], 2)}, because they find each other. The same search on the pixels finds {L.number(C.REFERENCE['of5']['pixels'], 2)}.",
    )
    guess_five
    return (guess_five,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Image retrieval** is finding the photos in a collection that look most like a given photo. With a vision foundation model it takes four steps. Compute the features of every photo in the collection once: a matrix `F` with a row of 384 numbers per photo. Compute the features `q` of the photo you search with. Work out the cosine similarity of `q` with every row of `F`. And keep the rows with the largest. Nobody has to label anything: the course uses the classes only to count how good the search is. On DINOv2's features, two photos of the same class have a cosine similarity of 0.60 on average, two photos of different classes 0.05.

        **All rows at once.** `F @ q` is the dot product of `q` with every row of `F`, as in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en): one number per photo. `np.linalg.norm(F, axis=1)` gives the norm of every row, as it gave the distances in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). Divide the one by the other and by the norm of `q`, and you have the cosine similarity of `q` with every photo of the collection, in one line:

        `sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`

        **Most alike first.** `np.argsort` sorts from small to large, and for k-NN that was right: a small distance means near. A cosine similarity is the other way round: large means alike. So you sort the similarities with a minus sign in front, `np.argsort(-sims, kind="stable")`, and keep the first k positions. Image retrieval is the first half of k-NN: k-NN lets the k photos it found vote on a label, and image retrieval shows them.
        """),
        L.watch_out("When the photo you search with is in the collection itself, it comes first, with a cosine similarity of 1: nothing is more like a photo than the photo itself. Skip it."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **image retrieval** | finding the photos in a collection that look most like a given photo, usually by comparing their features | `np.argsort(-sims, kind="stable")[:k]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_search():
    mo.vstack([
        mo.Html(C.search_picture({"title": "Searching 300 photos for the ones most alike", "collection": "collection: 300 photos",
                                  "query": "the photo you search with", "dino": "DINOv2-small", "once": "once per photo",
                                  "F": "F: 300 rows of 384", "q": "q: 384 features",
                                  "sims": "(F @ q) / (norms of the rows · norm of q)", "sims_note": "300 cosine similarities, one per photo",
                                  "top": "np.argsort(-sims, kind=\"stable\")[:5]", "top_note": "the positions of the 5 photos most alike",
                                  "alt": "On the left, the 300 photos of the collection go through DINOv2-small once each, into F, a matrix of 300 rows of 384 features. On the right, the photo you search with goes through DINOv2-small into q, 384 features. Together they give 300 cosine similarities, one per photo, and np.argsort gives the positions of the 5 photos most alike."})),
        mo.md(r"""
        The left column runs once, however often you search; the right column runs for every search, and then it is one matrix times a vector and one sort.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Every one of the 300 photos searches the 299 others. With the slider you choose how many of the photos most alike each search keeps, from 1 to 29: each class has 29 other photos. You see what percentage of them are of the same class, on DINOv2's features and on the pixels of the prepared photos.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _say(k):
        _k = int(k)
        _f, _p = C.percent(C.REFERENCE["features"][_k], "en"), C.percent(C.REFERENCE["pixels"][_k], "en")
        if _k == 1:
            return f"The photo most alike is of the same class in {_f} percent of the searches on DINOv2's features, and in {_p} percent on the pixels."
        return f"Of the {_k} photos most alike, on average {_f} percent are of the same class on DINOv2's features, and {_p} percent on the pixels."

    def _frame(k):
        _html = C.curve(int(k), {"title": "Of the same class, in percent", "x": "photos kept, most alike first", "y": "percent",
                                 "features": "DINOv2's features", "pixels": "pixels", "chance": "picked at random",
                                 "alt": "Two lines over the photos kept, most alike first, 1 to 29: on DINOv2's features 96 to 91 percent of them are of the same class, on the pixels 39 to 19 percent; photos picked at random would give about 10."}, "en")
        return _html, _say(k)

    explore_k = L.explore("k", "photos most alike", _frame, start=1, values=C.K_VALUES, role="",
                          task="Find the most photos you can keep while on DINOv2's features at least 95 percent of them are of the same class.")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    On DINOv2's features, you can keep up to 13 photos most alike and still have at least 95 percent of them of the same class: exactly 95.0 at 13, and 94.9 at 14. Even with 29, all the other photos a class has, 90.9 percent of them are of the same class: the search finds almost all photos of a class, without any label. On the pixels, the photo most alike is of the same class in 38.7 percent of the searches, and of 29 photos 18.8 percent are: better than photos picked at random, about 10, but far below the features.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To search a collection of photos for the ones most like a given photo:

    1. Compute the features of every photo in the collection once, with a vision foundation model: a row per photo in `F`, of shape `(n, 384)`.
    2. Compute the features `q` of the photo you search with, in the same way.
    3. Work out the cosine similarity of `q` with every row: `sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`.
    4. Take the positions of the k largest, the largest first: `np.argsort(-sims, kind="stable")[:k]`.
    5. Look up the photos at those positions. When the photo you search with is in the collection, skip it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell searches with the first lotus, photo 90 of the 300. `C.collection(dino)` is step 1: the features of the 300 photos, as a NumPy array `F` with a row per photo in the order of `C.CLASSES`, with their labels, the place of each photo's class in `C.CLASSES`, and their files. It computes them once, as `features(photos)` did in the previous lesson, which takes about a quarter of a minute, longer on a slow computer. The cell prints the shape of `F`, the positions of the 6 photos most alike and their cosine similarities, and then shows those photos with their class. `dino` is DINOv2-small, which the cell below loads once both downloads are done.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download DINOv2-small and Caltech-101, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small and the photos of Caltech-101 are not on your computer yet. The button downloads them once: DINOv2-small from Hugging Face, 88 MB, and Caltech-101 from Caltech, 137 MB, whose photos it then unpacks. The worked example and the demo need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 225 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*DINOv2-small and the photos are on your computer: 300 photos to search, 30 of each of 10 classes.*",
    })
    return (dino,)


@app.cell
def example_search(dino):
    import numpy as np

    # 1. the features of the 300 photos, once: a row of F per photo
    F, labels, photos = C.collection(dino)
    print(F.shape)
    # 2. the photo to search with: the first lotus, photo 90
    q = F[90]
    # 3. its cosine similarity with every photo
    sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))
    # 4. the positions from most to least alike
    order = np.argsort(-sims, kind="stable")
    print(order[:6])
    print(sims[order[:6]])
    # 5. the photo itself and the 5 photos most like it, with their class
    C.results(photos, labels, sims, order[:6], "en")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `F` has a row of 384 features for each of the 300 photos. The first position is 90: the lotus itself, with a cosine similarity of 1, printed here as `0.9999999`, perhaps as `1.` or `1.0000001` on your computer, because the computer rounds a little. Then come three lotuses, at 0.776, 0.776 and 0.774, a water lily at 0.749, and another lotus at 0.742. All 300 similarities took one line, and the sort another.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sims():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. All rows at once
    """)
    return


@app.cell(hide_code=True)
def choice_sims_cell():
    choice_sims = L.choice(
        "sims",
        question="`F` holds the features of 1,000 photos, a row of 768 numbers per photo, and `q` the 768 features of one photo. What is `(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`?",
        options={
            "one": "One number: the cosine similarity of `q` with all photos together",
            "right": "1,000 numbers: the cosine similarity of `q` with every photo",
            "features": "768 numbers: one for every feature",
            "distance": "1,000 numbers: the distance from `q` to every photo",
        },
        correct="right",
        why={
            "one": "`F @ q` gives a number for every row of `F`: 1,000 dot products. Divided by the 1,000 norms of the rows and the norm of `q`, they become 1,000 cosine similarities, one per photo.",
            "right": "`F @ q` is the dot product of `q` with every row, `np.linalg.norm(F, axis=1)` the norm of every row: dividing gives the cosine similarity of `q` with every photo, 1,000 numbers, however many features a row has.",
            "features": "The 768 features disappear in the dot product: each row of `F` and `q` give one number together. So there are 1,000, one per photo, each a cosine similarity.",
            "distance": "A distance is the norm of a difference, `np.linalg.norm(F - q, axis=1)`. This is the dot product divided by the norms: the cosine similarity of `q` with every photo, which looks only at the direction.",
        },
    )
    choice_sims
    return (choice_sims,)


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Search a collection

        Complete `search(F, q, k)`: steps 3 and 4 of the recipe. `F` has a row of features per photo, `q` the features of the photo you search with, and `k` is how many positions the function gives back, the most alike first. Replace the two `...`. The check makes up a collection of 6 rows of 3 features, of very different lengths, calls `search(F, q, 3)`, and looks at the positions it gives back. It also tries other collections and other `k`; there `q` is never one of the rows.
        """),
        L.task("Make `search(F, q, k)` give the positions of the `k` rows of `F` most like `q`, the most alike first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    search_code = L.code_box("search", lines=8)
    search_code
    return (search_code,)


@app.cell(hide_code=True)
def check_search(search_code):
    result_search = L.check("search", search_code.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Steps 3 and 4 of the worked example, with `k` instead of 6.",
            "`F @ q` gives the dot product of `q` with every row; divide it by the norms of the rows and of `q`. Then sort the similarities so that the largest come first, and keep the first `k`.",
            "`sims = (F @ q) / (np.linalg.norm(F, axis=___) * np.linalg.norm(q))` and `return np.argsort(___sims, kind=\"stable\")[:k]`.",
        ],
        why="`(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))` is the cosine similarity of `q` with every row at once. `np.argsort(-sims, kind=\"stable\")` gives the positions from the largest similarity to the smallest, and `[:k]` keeps the first `k`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your search on the photos

    When DINOv2 and the photos are downloaded (the button is in the worked example), the cell below lets your `search` loose on the 300 photos if it is right, and the course's otherwise. Every photo searches for the 6 rows most like it; the 5 after the photo itself count. For every class, you see how many of those 5 are of the same class, on average.
    """)
    return


@app.cell(hide_code=True)
def demo_search(dino, search_code, result_search):
    _d = C.demo(dino, search_code.value if result_search.state == "correct" else None)
    _who = "your `search`" if _d["yours"] else "the course's `search`"
    _rows = "\n".join(f"| {C.NAMES['en'][_c]} | {L.number(_d['class5'][_i], 2)} | {L.number(C.REFERENCE['class5']['features'][_i], 2)} |"
                      for _i, _c in enumerate(C.CLASSES))
    mo.md(f"""With {_who}, on DINOv2's features:

| class | of the 5 most alike, of the same class | the course measured |
|---|---|---|
{_rows}""")
    return


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.md(r"""
    ### Exercise 3. A search on the pixels
    """)
    return


@app.cell(hide_code=True)
def choice_pixels_cell():
    choice_pixels = L.choice(
        "pixels",
        question=f"The same search on the pixels of the prepared photos finds on average {L.number(C.REFERENCE['class5']['pixels'][7], 2)} sunflowers among the 5 photos most like a sunflower, but only {L.number(C.REFERENCE['class5']['pixels'][4], 1)} nautiluses for a nautilus. Why do sunflowers do best?",
        options={
            "middle": "Sunflowers fill the middle of the photo, and the pixels compare what lies in the middle",
            "more": "There are more sunflowers among the 300 photos",
            "right": "Sunflower photos share a lot of yellow, and pixels compare colors place by place",
            "shape": "The pixels describe the shape of the petals better than DINOv2's features do",
        },
        correct="right",
        why={
            "middle": f"Many photos have their object in the middle, not only sunflowers. Take the color away and keep the places: on gray pixels, the mean of the three color channels, sunflowers find only {L.number(C.REFERENCE['gray']['sunflower'], 1)} of 5. It is the color: on average about {C.REFERENCE['yellow']['sunflower']} percent of the pixels of a sunflower photo are strongly yellow, the course counted, and about {C.REFERENCE['yellow']['others'][0]} to {C.REFERENCE['yellow']['others'][1]} percent in each other class.",
            "more": f"Every class has 30 photos. What sunflower photos share is their color: on average about {C.REFERENCE['yellow']['sunflower']} percent of their pixels are strongly yellow, the course counted, and about {C.REFERENCE['yellow']['others'][0]} to {C.REFERENCE['yellow']['others'][1]} percent in each other class.",
            "right": f"On average about {C.REFERENCE['yellow']['sunflower']} percent of the pixels of a sunflower photo are strongly yellow, the course counted, and about {C.REFERENCE['yellow']['others'][0]} to {C.REFERENCE['yellow']['others'][1]} percent in each other class. The cosine similarity of two photos' pixels is large when the same colors lie in the same places, whatever the object: color is what pixels say directly. On gray pixels, without the color, sunflowers find only {L.number(C.REFERENCE['gray']['sunflower'], 1)} of 5. A nautilus is white, brown or gray, like much else. On DINOv2's features, sunflowers find {L.number(C.REFERENCE['class5']['features'][7], 2)} of 5.",
            "shape": f"On DINOv2's features, sunflowers find {L.number(C.REFERENCE['class5']['features'][7], 2)} of 5, on the pixels {L.number(C.REFERENCE['class5']['pixels'][7], 2)}. A pixel only says what color one spot is; it knows nothing of petals. What sunflower photos share on the pixels is their color: on average about {C.REFERENCE['yellow']['sunflower']} percent of their pixels are strongly yellow.",
        },
    )
    choice_pixels
    return (choice_pixels,)


@app.cell(hide_code=True)
def text_exercise_look_alike():
    mo.md(r"""
    ### Exercise 4. A water lily for a lotus
    """)
    return


@app.cell(hide_code=True)
def explain_look_alike_cell():
    explain_look_alike = L.explain(
        "look-alike",
        question="In the worked example, a water lily is among the 5 photos most like the lotus, and on average lotuses find only 3.7 lotuses among their 5. Is that a mistake of the search? Say what the search looks for, and what it would take to keep lotuses and water lilies apart.",
        model_answer="**The search finds the photos whose features point most the same way, and nothing else:** it knows no classes. **A lotus and a water lily do look alike,** two large flowers with pointed petals, often among round leaves on the water, and their features are close: on average a lotus and a water lily have a cosine similarity of 0.50, almost as much as two lotuses, 0.52, against 0.05 for two photos of different classes in general. If you want photos that look like this one, the water lily is a good answer; only if you want the same kind of flower is it a mistake, and that only the labels say. Labels can be wrong too: some photos labeled lotus in Caltech-101 seem to show water lilies. **To keep them apart, you need labeled photos:** a probe that learns from them which differences count, or features that hold those small differences more clearly. k-NN helps little: it learns nothing, and its vote counts the same neighbors the search finds. The probe of the previous lesson mixed up the same two classes, so even with labels it stays hard.",
        points=[
            "you say that the search only looks at how alike the features are, and knows no classes",
            "you say that lotus and water lily look alike, so that a water lily is a good answer to 'photos like this one', and a mistake only for 'the same kind'",
            "you say that keeping them apart takes labels, such as a probe that learns from them, or features that hold the difference more clearly",
        ],
    )
    explain_look_alike
    return (explain_look_alike,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The cosine similarity of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), which in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) found the image of a digit most like a given one, now searches 300 photos at once, on the features of [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en). It is the first half of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), with a minus sign, because a large similarity means near. In the next lesson, *Fine-tune a classifier on your own kinds of photos*, the model itself learns a little too: every layer of a pretrained classifier, on a few hundred photos of new kinds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pixels, choice_sims, explain_look_alike, result_search):
    L.wrap_up(choice_sims, result_search, choice_pixels, explain_look_alike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
