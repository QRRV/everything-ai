# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-metadata-filters. Route: course/route/lessen/bouwen.yaml. Checks, the store and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-metadata-filters", "en")
    C = L.checks   # the course's points in both languages with their metadata, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Searching many vectors fast](/course/open/build-vector-index?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_index():
    quick_index = L.choice(
        "quick-index",
        question="An IVF index looks in 1 of its 16 clusters, and gives three points. Which three are they?",
        options={
            "cluster": "The best three of the vectors in the nearest cluster",
            "all": "The three vectors nearest the question, out of all vectors",
            "centroids": "The three centroids that lie nearest the question",
        },
        correct="cluster",
        why={
            "cluster": "The search compares the question only with the vectors in the lists of the clusters it looks in. A nearer vector in another cluster is never compared, so it can be missed.",
            "all": "That is what a search through every vector gives. The index compares only the vectors of the clusters it looks in. See [Searching many vectors fast](/course/open/build-vector-index?lang=en).",
            "centroids": "The centroids only decide which clusters the search looks in; the points it gives come from the lists of those clusters. See [Searching many vectors fast](/course/open/build-vector-index?lang=en).",
        },
    )
    quick_index
    return (quick_index,)


@app.cell(hide_code=True)
def text_why():
    _l = C.languages()["en"]
    mo.md(r"""
    ## Why this matters

    The course exists in English and in Dutch. An assistant for it can search the points of both: {items} items. But search by meaning only compares embeddings; it knows nothing about the language of a point, its lesson or its level. Asked in English, the course's {questions} questions got only {same} English points among their {all} results, and for {twice} of them the three best held the same point twice. A company has the same problem with more at stake: documents of every department in one knowledge base, and some that only a few people may read. The search needs to know more about each text than its embedding.

    **An everyday example.** In a webshop you search for *rain jacket*, tick *size M* and *in stock*, and see only the jackets that fit. The search still goes by what you typed; the ticks decide which products it may show. That is only a comparison: the search below compares lists of numbers.
    """.format(items=C.number(len(C.store()), L.lang), questions=len(C.QUESTIONS), same=_l["same"], all=C.K * len(C.QUESTIONS), twice=_l["twice"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.worked()
    _item = C.store()[_w["first"][0]]
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **metadata** | data about data: here, for every point, its language, its lesson, its domain and its level |
    | **metadata filtering** | limiting a search to the items whose metadata fit a condition, such as the language `"nl"` |
    | **vector database** | a database that keeps embeddings together with their text and metadata, and answers searches fast, usually with filters on the metadata |
    | `where` | a dict with the metadata an item must have, such as `{{"lang": "nl", "level": "basis"}}` |

    The course made a store of its points in both languages: {items} items, one per point and language. Every item is a dict with the text, the embedding and the metadata:

    ```python
    {{"text": "...", "vector": np.array([...]), "lang": "{lang}", "lesson": "{lesson}", "domain": "{domain}", "level": "{level}"}}
    ```

    A filter can come at two moments. *Afterwards*: search all items, take the `k` best, and drop those that do not fit. That is simple, but it can leave fewer than `k`, or none. *First*: keep only the items that fit, and search those. Then you get `k` results whenever `k` items fit.

    A vector database does this for you, and fast. Qdrant, Chroma, Weaviate and Milvus are made for it, and pgvector adds it to PostgreSQL, an ordinary database. They keep the embedding, the text and the metadata of every chunk together, build an index, often HNSW, and filter during the search, so that a filter does not leave too few. Libraries such as LangChain call it a *vector store*. For a few thousand chunks, a list in memory, as here, is enough.
    """.format(items=C.number(len(C.store()), L.lang), lang=_item["lang"], lesson=_item["lesson"], domain=_item["domain"], level=_item["level"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: search with a filter

    1. **Once:** give every item its metadata when it goes into the store.
    2. **Filter:** keep the items whose metadata fit `where`.
    3. **Score:** take the dot product of the question with those items only.
    4. **Keep:** the `k` with the highest score, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Worked example

    *{nl}* (in English: *{en}*) A learner who reads the course in Dutch asks it in a lesson of domain `bouwen`, so the search uses `where = {where}`. {fits} items fit.

    1. **Afterwards:** the three best of all {items} items are points of *{t0}*, *{t1}* and *{t2}*, of the domains `{d0}`, `{d1}` and `{d2}`. Not one fits, so filtering afterwards leaves nothing.
    2. **First:** of the {fits} items that fit, the best is a point of *{f0}*: *{p0}* (in English: *{p0_en}*) It answers the question; among all items it was at place {pl0}. Then come points of *{f1}* and *{f2}*, at places {pl1} and {pl2}.
    """.format(nl=_w["question"]["text"]["nl"], en=_w["question"]["text"]["en"], where=C.where_text(C.WORKED_WHERE), fits=_w["fits"],
               items=C.number(len(C.store()), L.lang), t0=C.item_title(_w["top"][0], L.lang), t1=C.item_title(_w["top"][1], L.lang),
               t2=C.item_title(_w["top"][2], L.lang), d0=C.store()[_w["top"][0]]["domain"], d1=C.store()[_w["top"][1]]["domain"],
               d2=C.store()[_w["top"][2]]["domain"], f0=C.item_title(_w["first"][0], L.lang), p0=C.store()[_w["first"][0]]["text"],
               p0_en=C.store()[C.point_of(_w["first"][0])]["text"],
               pl0=_w["places"][0], f1=C.item_title(_w["first"][1], L.lang), f2=C.item_title(_w["first"][2], L.lang),
               pl1=_w["places"][1], pl2=_w["places"][2]))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### What the course measured

    The course asked all {questions} questions in Dutch, with three filters, and filtered afterwards and first:
    """.format(questions=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _a = C.afterwards()
    _l = C.languages()
    _i = C.with_index()
    mo.md(r"""
    Filtering afterwards on the language alone kept {k0} of the three on average. On `{w1}` it kept nothing for {n1} of the {questions} questions, on `{w2}` for {n2}. Filtering first gave three for every question, each time.

    Without a filter, the English questions got {same_en} English points among their {all} results, and the Dutch ones {same_nl} Dutch points. For {twice_en} of the English questions and {twice_nl} of the Dutch ones, the three best held the same point twice, in both languages. With a filter on the language of the question, the right point was among the three for {rf_en} of the {answerable} questions that a point answers in English, against {r_en} without; in Dutch {rf_nl} against {r_nl}.

    With the IVF index of [Searching many vectors fast](/course/open/build-vector-index?lang=en), a filter is harder: {empty} of its 16 clusters hold no point of domain `{domain}`. Looking in 1 cluster and keeping the points of `{domain}` in it, the search found nothing for {i1} of the {questions} English questions, and fewer than three for {f1} more; looking in 2, nothing for {i2}. That is why vector databases filter during the search, not afterwards.
    """.format(k0=C.number(_a[0]["kept"], L.lang, 1), w1=C.where_text(_a[1]["where"]), n1=_a[1]["nothing"], questions=len(C.QUESTIONS),
               w2=C.where_text(_a[2]["where"]), n2=_a[2]["nothing"], same_en=_l["en"]["same"], all=C.K * len(C.QUESTIONS),
               same_nl=_l["nl"]["same"], twice_en=_l["en"]["twice"], twice_nl=_l["nl"]["twice"], rf_en=_l["en"]["right_filtered"],
               answerable=len(C.ANSWERABLE), r_en=_l["en"]["right"], rf_nl=_l["nl"]["right_filtered"], r_nl=_l["nl"]["right"],
               empty=_i["empty_clusters"], domain=C.DOMAIN, i1=_i[1]["nothing"], f1=_i[1]["fewer"], i2=_i[2]["nothing"]))
    return


@app.cell(hide_code=True)
def text_matches():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Does an item fit?

        `matches(item, where)` should give `True` when the item has, for every key of `where`, the value that `where` has for it, and `False` otherwise. An empty `where` lets every item through. With `item = {note}`, `matches(item, {{"lang": "en", "level": "basis"}})` gives `False`: the level fits, the language does not. The check also counts how many of the course's {items} items fit a few filters, by your function.
        """.format(note=C.where_text(C.NOTE), items=C.number(len(C.store()), L.lang))),
        L.task("Give `False` as soon as one key of `where` has another value in `item`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matches():
    box_matches = L.code_box("matches", lines=5)
    box_matches
    return (box_matches,)


@app.cell(hide_code=True)
def check_matches(box_matches):
    result_matches = L.check("matches", box_matches.value)
    result_matches
    return (result_matches,)


@app.cell(hide_code=True)
def hints_matches():
    L.hints(
        "matches",
        [
            "Go through the keys of `where` one by one, and compare the value of the item with the value of `where`.",
            "`for key in where:` goes through the keys; `item[key] != where[key]` is `True` when the item does not fit for that key.",
            "In the loop: `if item[key] != where[key]:` and then `return False`; after the loop, `return True`.",
        ],
        why="""```python
def matches(item, where):
    for key in where:
        if item[key] != where[key]:
            return False
    return True
```

The function can say no at the first key that does not fit, but yes only after the last: only then do all of them fit.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Filter first

        `filtered_search(q, store, where, k)` should give the texts of the `k` items with the highest dot product with `q`, among the items that fit `where`. `store` is a list of items like the ones above, and `matches` is given. The box filters afterwards: it takes the `k` best of all items, and then drops those that do not fit. The small store of the example has six notes:

        ```python
        {notes}
        ```

        `filtered_search(np.array([1.0, 0.2]), NOTES, {{"lang": "nl"}}, 2)` should give `["k-means in het kort", "HNSW uitgediept"]`. The box gives only `["k-means in het kort"]`, because the second best of all, `"k-means in brief"`, is in English. The check also runs your function on the course's store, for the question of the worked example.
        """.format(notes=C.NOTES_CODE.replace("\n", "\n        "))),
        L.task("Give only the items that fit `where` a score, so that you keep the `k` best of those."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("filtered_search", lines=7)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("filtered_search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "filtered_search",
        [
            "The filter now comes after `sorted(...)[:k]`. It should come before: an item that does not fit should never get a score.",
            "In the loop, put `if matches(store[i], where):` in front of the line with `found.append(...)`, one step further in.",
            "Add `if matches(store[i], where):` above `found.append(...)`, and remove `if matches(store[i], where)` from the last line.",
        ],
        why="""```python
def filtered_search(q, store, where, k):
    found = []
    for i in range(len(store)):
        if matches(store[i], where):
            found.append((store[i]["vector"] @ q, i))
    best = sorted(found, reverse=True)[:k]
    return [store[i]["text"] for score, i in best]
```

The items that do not fit never get a score, so the `k` best are always `k` that fit, as long as there are that many.""",
    )
    return


@app.cell(hide_code=True)
def text_twice():
    mo.md(r"""
    ### Exercise 3. The same point twice

    For the English question *{question}*, the three best of the store with both languages are the same point in Dutch and in English, and one more point.
    """.format(question=C.question("q11")["text"]["en"]))
    return


@app.cell(hide_code=True)
def choice_twice_cell():
    _s = C.twice_scores()
    choice_twice = L.choice(
        "twice",
        question="Why does the same point come twice?",
        options={
            "copy": "The Dutch points have the same embeddings as the English ones",
            "store": "The store holds every Dutch point twice, by mistake",
            "translation": "Granite embeds a text and its translation close together",
        },
        correct="translation",
        why={
            "translation": "Granite was trained so that a text and its translation mean the same to it, which is also why a Dutch question finds an English point. Here the Dutch point scored {nl} and the English one {en}. For an assistant that answers in one language, the second copy only takes the place of another point: filter on the language of the question.".format(
                nl=C.number(_s["nl"], L.lang, 3), en=C.number(_s["en"], L.lang, 3)),
            "copy": "Then they would score exactly the same. The Dutch point scored {nl}, the English one {en}: close, not the same.".format(
                nl=C.number(_s["nl"], L.lang, 3), en=C.number(_s["en"], L.lang, 3)),
            "store": "The store has {items} items: every point once in English and once in Dutch. The two that came are not the same item.".format(
                items=C.number(len(C.store()), L.lang)),
        },
    )
    choice_twice
    return (choice_twice,)


@app.cell(hide_code=True)
def text_access():
    mo.md(r"""
    ### Exercise 4. Who may read it?

    A company keeps the documents of every department in one vector database, for an assistant that answers questions from them. Only HR may read the salary documents.
    """)
    return


@app.cell(hide_code=True)
def choice_access_cell():
    choice_access = L.choice(
        "access",
        question="How do you keep the salary documents out of the answers for everyone else?",
        options={
            "prompt": "Tell the model in its system prompt never to quote salary documents",
            "filter": "Filter in the search on metadata that says who may read a chunk",
            "answer": "Check every answer for amounts of money, and block those",
        },
        correct="filter",
        why={
            "filter": "Then a chunk someone may not read never reaches the prompt, and what the model never sees, it cannot pass on. The filter belongs in the search itself, before the `k` best are taken; afterwards it can leave too few, as the course measured.",
            "prompt": "What is in the prompt, the model can pass on: an instruction lowers the chance, but does not make it zero. Better that the salary documents never reach the prompt.",
            "answer": "Then the salary documents still reach the model, and a salary written in words, or a sentence about who earns the most, passes your check.",
        },
    )
    choice_access
    return (choice_access,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Search the course with your own filtered_search

    Once exercises 1 and 2 are right, pick a question and a filter. The course searches its store with your `filtered_search` and your `matches`, and shows below it the three best without a filter. Try a filter that fits few items, such as a domain with a level.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q11")["text"][L.lang], label="Question", full_width=True)
    lang_pick = mo.ui.dropdown(options={"any": "", "en": "en", "nl": "nl"}, value="any", label="lang")
    domain_pick = mo.ui.dropdown(options={"any": "", **{d: d for d in C.domains()}}, value="any", label="domain")
    level_pick = mo.ui.dropdown(options={"any": "", **{lv: lv for lv in C.LEVELS}}, value="any", label="level")
    mo.vstack([question_pick, mo.hstack([lang_pick, domain_pick, level_pick], justify="start", wrap=True)])
    return domain_pick, lang_pick, level_pick, question_pick


@app.cell(hide_code=True)
def demo_search(box_matches, box_search, domain_pick, lang_pick, level_pick, question_pick):
    _fn = C.learner_search(box_matches.value, box_search.value)
    if _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the search uses your own `matches` and `filtered_search`.*")
    else:
        _where = C.where_of(lang_pick.value, domain_pick.value, level_pick.value)
        _out = mo.Html(C.searched(_fn, question_pick.value, _where, L.lang, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A filter decides which points may be found; which of them are found best is the next question. Search by meaning missed the point about the agent, which BM25 found first, in [Search by meaning](/course/open/build-retrieval-embeddings?lang=en). [Words and meaning together](/course/open/build-hybrid-search?lang=en) merges the two searches, because each finds points the other misses.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_access, choice_twice, result_matches, result_search):
    L.wrap_up(result_matches, result_search, choice_twice, choice_access)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
