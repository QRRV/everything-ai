# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-vector-index. Route: course/route/lessen/bouwen.yaml. Checks, de index en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-vector-index", "nl")
    C = L.checks   # de knowledge base, de embeddings van Granite ervan, de index van de course, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_centroid():
    quick_centroid = L.choice(
        "quick-centroid",
        question="k-means is klaar. Wat is de centroid van een cluster?",
        options={
            "first": "Het voorbeeld dat k-means als eerste voor dat cluster koos",
            "far": "Het voorbeeld van het cluster dat het verst van de andere clusters ligt",
            "mean": "Het gemiddelde van zijn voorbeelden, feature voor feature",
        },
        correct="mean",
        why={
            "mean": "Elke keer verschuiven zet een centroid op het gemiddelde van de voorbeelden in zijn cluster, en als k-means klaar is, verandert die stap niets meer. De centroid is dus het middelpunt van zijn cluster, meestal zelf geen voorbeeld.",
            "first": "k-means kan beginnen met willekeurig gekozen voorbeelden, maar elke keer verschuiven zet de centroids ergens anders neer. Zie [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "far": "Een centroid ligt midden in zijn eigen cluster, niet aan de rand. Zie [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
        },
    )
    quick_centroid
    return (quick_centroid,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    De zoekfunctie van [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl) vergelijkt de vraag met elk punt: {points} inproducten van {dims} getallen, {mult} vermenigvuldigingen voor elke vraag. Voor de course is dat niets. De knowledge base van een bedrijf kan een miljoen chunks hebben: {mmult} vermenigvuldigingen voor elke vraag, en {gb} GB alleen om hun embeddings in float32 te bewaren. Een zoekmachine heeft er miljarden. Een index vergelijkt de vraag met maar een klein deel van de vectoren, zo gekozen dat de dichtstbijzijnde er bijna altijd bij zitten.

    **Een voorbeeld uit het dagelijks leven.** In een bibliotheek loop je niet langs elk boek. Je leest de bordjes boven de afdelingen, gaat naar de afdeling die het best bij je vraag past, en zoekt alleen in die kasten. Soms staat het beste boek in de afdeling ernaast, en dan mis je het, tenzij je daar ook kijkt. Dat is alleen een vergelijking: de zoekfunctie hieronder vergelijkt lijsten van getallen.
    """.format(points=C.number(_f["points"], L.lang), dims=C.DIMS, mult=C.number(_f["multiplications"], L.lang),
               mmult=C.number(_f["million_multiplications"], L.lang), gb=C.number(_f["million_gb"], L.lang, 1)))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **vector index** | een structuur die je één keer over de vectoren bouwt, en die zoeken erin snel maakt; vergelijken met elke vector, zoals in [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl), heet *flat search* |
    | **approximate nearest neighbor search** | vectoren vinden die met grote kans bij de dichtstbijzijnde horen, door de meeste vergelijkingen over te slaan; af en toe mis je er een die dichterbij ligt |
    | **IVF index** | een index die de vectoren met k-means in clusters indeelt, en een vraag alleen vergelijkt met de vectoren in de paar clusters waarvan de centroids het dichtst bij liggen; IVF staat voor *inverted file* |
    | **HNSW** | een index die elke vector verbindt met een paar vectoren dicht bij hem, op meerdere niveaus; een zoekfunctie gaat van vector naar vector, telkens naar de vector die het dichtst bij de vraag ligt |
    | `probes` | het aantal clusters waarin een zoekfunctie kijkt |

    De course bouwde één keer een IVF index van zijn {points} punten. k-means deelde hun embeddings in {clusters} clusters in, van {small} tot {large} punten elk, en per cluster houdt een lijst de nummers van zijn punten bij. Een zoekfunctie gaat dan in twee stappen. Eerst vergelijkt hij de vraag met de {clusters} centroids, op afstand zoals k-means doet, en neemt de `probes` dichtstbijzijnde. Daarna vergelijkt hij de vraag, met het inproduct, alleen met de punten in de lijsten van die clusters. Met `probes` 1 zijn dat zo'n {per} punten in plaats van {points}.

    Het beste punt zit niet altijd in het cluster met de dichtstbijzijnde centroid. Een punt kan aan de rand van zijn cluster liggen, dichter bij de vraag dan elk punt van een cluster waarvan de centroid dichterbij ligt. Daarom zoek je *bij benadering*: hoe meer clusters de zoekfunctie bekijkt, hoe minder vaak hij een punt mist, en hoe meer vectoren hij vergelijkt.

    IVF is de eenvoudigste index. De meeste vector databases, databases die gemaakt zijn om embeddings te bewaren en te doorzoeken, gebruiken HNSW, dat meestal dezelfde vectoren vindt met minder vergelijkingen, maar meer geheugen nodig heeft voor al zijn verbindingen. Faiss, een library van Meta, heeft flat search, IVF en HNSW kant-en-klaar, voor een miljoen vectoren en meer; het noemt `probes` *nprobe*. Het idee blijft hetzelfde: vergelijk met weinig vectoren, en meet wat je mist.
    """.format(points=C.number(_f["points"], L.lang), clusters=_f["clusters"], small=_f["smallest"], large=_f["largest"],
               per=C.number(round(_f["points"] / _f["clusters"]), L.lang)))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: zoeken in een IVF index

    1. **Eén keer:** clusteer de vectoren met k-means, en bewaar de centroids.
    2. **Eén keer:** maak per cluster een lijst met de nummers van zijn vectoren.
    3. **Clusters:** bereken de afstand van de vraag tot elke centroid, en neem de `probes` dichtstbijzijnde clusters.
    4. **Score:** neem het inproduct van de vraag met alleen de vectoren in de lijsten van die clusters.
    5. **Houden:** de `k` met de hoogste score, de hoogste eerst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{question}*

    1. **Clusters:** van de {clusters} centroids liggen de dichtstbijzijnde drie op afstand {d0}, {d1} en {d2}.
    2. **Eén cluster:** het dichtstbijzijnde cluster heeft {s0} punten. Met `probes` 1 vergelijkt de zoekfunctie de vraag met {c1} vectoren: de {clusters} centroids en die {s0} punten. Zijn beste drie zijn punten van *{o0}*, *{o1}* en *{o2}*: over de helling van een loss, niet over de vorm van zijn kromme.
    3. **Drie clusters:** de volgende twee voegen {s1} en {s2} punten toe, {c3} vectoren in totaal. Nu is de eerste een punt van *{t0}*: *{p0}* Dat beantwoordt de vraag.

    Dat punt ligt in het op twee na dichtstbijzijnde cluster: zijn centroid ligt verder van de vraag, maar het punt zelf ligt dichterbij dan elk punt van het dichtstbijzijnde cluster. Met drie clusters vindt de zoekfunctie dezelfde drie punten als een zoekfunctie door elke vector, en vergelijkt hij {c3} vectoren in plaats van {points}.
    """.format(question=_w["question"]["text"][L.lang], clusters=C.CLUSTERS, d0=C.number(_w["distances"][0], L.lang, 3),
               d1=C.number(_w["distances"][1], L.lang, 3), d2=C.number(_w["distances"][2], L.lang, 3), s0=_w["sizes"][0],
               s1=_w["sizes"][1], s2=_w["sizes"][2], c1=_w["compared_one"], c3=_w["compared_three"],
               o0=C.title_of(_w["one"][0], L.lang), o1=C.title_of(_w["one"][1], L.lang), o2=C.title_of(_w["one"][2], L.lang),
               t0=C.title_of(_w["three"][0], L.lang), p0=C.POINTS[_w["three"][0]], points=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course doorzocht zijn index voor alle {questions} vragen, ook de {none} die geen punt beantwoordt, door in 1 tot {clusters} clusters te kijken, en vergeleek de drie punten die hij vond met de drie van een zoekfunctie door elke vector. Eerst raden:
    """.format(questions=C.findings()["questions"], none=C.findings()["questions"] - C.findings()["answerable"], clusters=C.CLUSTERS))
    return


@app.cell(hide_code=True)
def guess_one_cell():
    _f = C.findings()
    guess_one = L.guess(
        "one-cluster",
        question="Als hij in 1 van de {clusters} clusters kijkt, voor hoeveel van de {questions} vragen vindt de index dan dezelfde drie punten als een zoekfunctie door elke vector?".format(
            clusters=C.CLUSTERS, questions=_f["questions"]),
        options={
            "all": "Voor allemaal, of bijna allemaal",
            "most": "Voor ongeveer driekwart",
            "third": "Voor ongeveer een derde",
        },
        reveal="**Voor {one} van de {questions}**, ongeveer een derde. Voor {best} vragen zit het beste punt in het dichtstbijzijnde cluster, maar de beste drie zitten er alle drie maar voor {one}: de tweede en de derde zitten vaak in een cluster ernaast.".format(
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
        In 1 cluster vergeleek de index gemiddeld {c1} vectoren en vond hij dezelfde drie voor {s1} van de {questions} vragen; in 2 clusters {c2} vectoren en {s2}; in 4 {c4} en {s4}; in 7 {c7} en {s7}. Vanaf {at} clusters vond hij dezelfde drie voor alle {questions}, maar dan vergeleek hij {call} vectoren, meer dan driekwart van alle {points}. Voor een antwoord telt of het goede punt bij de drie zit: door elke vector was dat zo voor {gold} van de {answerable} vragen die een punt beantwoordt, in 1 cluster voor {g1}, in 7 voor {g7}.

        Met {more} clusters zijn de lijsten korter: in 1 cluster vergeleek de index {cm} vectoren, en vond hij dezelfde drie voor maar {sm}. Op {points} punten wint een index weinig. Met een miljoen vectoren in {big} clusters van zo'n {per} elk vergelijkt kijken in {probes} clusters zo'n {big_c} vectoren: ongeveer een op de honderd.
        """.format(c1=_f["one"][2], s1=_f["one"][1], questions=_f["questions"], c2=_f["two"][2], s2=_f["two"][1], c4=_f["four"][2],
                   s4=_f["four"][1], c7=_f["seven"][2], s7=_f["seven"][1], at=_f["all_at"], call=_f["all_compared"],
                   points=C.number(_f["points"], L.lang), gold=_f["flat_gold"], answerable=_f["answerable"], g1=_f["one"][3],
                   g7=_f["seven"][3], more=C.MORE, cm=_more[1], sm=_more[0], big=C.number(C.BIG_CLUSTERS, L.lang),
                   per=C.number(C.MILLION // C.BIG_CLUSTERS, L.lang), probes=C.BIG_PROBES, big_c=C.number(_f["big_compared"], L.lang))),
        L.watch_out("Een index mist ongemerkt: hij geeft nog steeds `k` punten, en niets daarin zegt dat er een punt werd overgeslagen dat dichterbij lag. Meet voordat je op een index vertrouwt op je eigen vragen hoe vaak hij dezelfde punten vindt als een zoekfunctie door elke vector, en kies daarmee in hoeveel clusters hij kijkt."),
    ])
    return


@app.cell(hide_code=True)
def text_lists():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een lijst per cluster

        `make_lists(labels, n)` hoort een lijst van `n` lijsten te geven: lijst `c` bevat de nummers `i` van de punten met label `c`, van klein naar groot. `labels` bevat het cluster van elk punt, zoals `km.labels_` van k-means: een lijst of een NumPy-array; `n` is het aantal clusters. `make_lists([0, 1, 0, 2, 1], 3)` geeft `[[0, 2], [1, 4], [3]]`: punten 0 en 2 zitten in cluster 0. De check tekent ook de lijsten die je functie maakt van de {points} punten van de course.
        """.format(points=C.number(len(C.POINTS), L.lang))),
        L.task("Geef cluster `c` de nummers `i` van de punten met label `c`, in plaats van een lege lijst."),
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
            "Ga voor cluster `c` de nummers `i` van alle punten langs, `range(len(labels))`, en houd die met label `c`.",
            "Een comprehension met een voorwaarde doet dat in één regel: `[i for i in range(len(labels)) if ...]`.",
            "In de loop: `lists.append([i for i in range(len(labels)) if ___])`.",
        ],
        why="""```python
def make_lists(labels, n):
    lists = []
    for c in range(n):
        lists.append([i for i in range(len(labels)) if labels[i] == c])
    return lists
```

Omdat de loop alle `n` clusters langsgaat, krijgt een cluster waar geen punt in zit toch zijn lijst, een lege. Eén loop over de punten kan het ook: begin met `n` lege lijsten, en zet elke `i` in lijst `labels[i]`.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Alleen in de dichtstbijzijnde clusters kijken

        `ivf_search(q, index, k, probes)` hoort de nummers te geven van de `k` punten met het hoogste inproduct met `q`, tussen de punten van de `probes` clusters waarvan de centroids het dichtst bij `q` liggen. `index` is een dict met drie delen: `"centroids"`, een array met een centroid per rij; `"lists"`, de lijsten van `make_lists`; en `"vectors"`, de embeddings, een rij per punt. Het vak vindt de dichtstbijzijnde clusters al, `nearest`, maar vergelijkt `q` daarna met elke vector. De kleine index van het voorbeeld heeft zes vectoren van twee getallen in drie clusters:

        ```python
        {small}
        ```

        `ivf_search(np.array([0.6, 0.8]), SMALL, 2, 1)` kijkt alleen in cluster 1, het dichtstbijzijnde, en geeft `[2, 3]`. In 2 clusters geeft hij `[2, 0]`, hetzelfde als een zoekfunctie door elke vector: punt 0, in cluster 0, past beter dan punt 3. De check draait je functie ook op de index van de course, voor de vraag van het uitgewerkte voorbeeld, en telt de vectoren die hij vergelijkt.
        """.format(small=C.SMALL_CODE.replace("\n", "\n        "))),
        L.task("Vergelijk `q` alleen met de punten in de lijsten van de clusters in `nearest`."),
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
            "`index[\"lists\"]` heeft een lijst per cluster, met de nummers van de punten erin; je hebt die van elk cluster in `nearest` nodig.",
            "Twee loops, de een in de ander: een over de clusters in `nearest`, en daarin een over de nummers van de punten van dat cluster.",
            "Op de plaats van de loop over `range(len(index[\"vectors\"]))`: `for c in nearest:`, en daarin `for i in ___:`, met de regel met `found.append(...)` in de binnenste.",
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

Het plaatje onder de check telt de vectoren die je functie echt vergelijkt. Een functie die eerst `index["vectors"] @ q` voor alle punten berekent en daarna de clusters kiest, geeft dezelfde nummers, maar vergelijkt alle {points}: dan bespaart de index niets.""".format(points=C.number(len(C.POINTS), L.lang)),
    )
    return


@app.cell(hide_code=True)
def text_find_more():
    mo.md(r"""
    ### Oefening 3. Vaker hetzelfde vinden

    Kijk nog eens naar het plaatje van wat de course mat.
    """)
    return


@app.cell(hide_code=True)
def choice_find_more_cell():
    _f = C.findings()
    _more = C.more_clusters()
    choice_find_more = L.choice(
        "find-more",
        question="Als hij in 1 van de {clusters} clusters kijkt, vindt de index voor {one} van de {questions} vragen dezelfde drie punten als een zoekfunctie door elke vector. Waardoor vindt hij vaker hetzelfde?".format(
            clusters=C.CLUSTERS, one=_f["one"][1], questions=_f["questions"]),
        options={
            "k": "Vraag om meer punten: een grotere `k`",
            "probes": "Kijk in meer clusters: een grotere `probes`",
            "clusters": "Laat k-means meer clusters maken: {more} in plaats van {clusters}".format(more=C.MORE, clusters=C.CLUSTERS),
        },
        correct="probes",
        why={
            "probes": "Elk cluster erbij voegt zijn punten toe aan die de zoekfunctie vergelijkt. In 4 clusters vond de index dezelfde drie voor {s4} van de {questions}, vanaf {at} clusters voor allemaal; de prijs is het aantal vectoren dat hij vergelijkt: gemiddeld {c4} en {call}, tegen {c1} in 1 cluster.".format(
                s4=_f["four"][1], questions=_f["questions"], at=_f["all_at"], c4=_f["four"][2], call=_f["all_compared"], c1=_f["one"][2]),
            "k": "Een grotere `k` geeft meer punten uit dezelfde clusters. Een punt in een cluster waar de zoekfunctie niet kijkt, wordt nog steeds nooit vergeleken.",
            "clusters": "Meer clusters maken elke lijst korter, dus in 1 cluster vergelijkt de zoekfunctie minder punten, en mist hij meer: met {more} clusters {cm} vectoren, en dezelfde drie voor maar {sm} van de {questions}.".format(
                more=C.MORE, cm=_more[1], sm=_more[0], questions=_f["questions"]),
        },
    )
    choice_find_more
    return (choice_find_more,)


@app.cell(hide_code=True)
def text_quiet_miss():
    mo.md(r"""
    ### Oefening 4. Mist de index iets?

    Een collega heeft een RAG-systeem gebouwd met een IVF index. Het geeft voor elke vraag drie punten, dus volgens de collega mist de index niets.
    """)
    return


@app.cell(hide_code=True)
def choice_quiet_miss_cell():
    _w = C.worked()
    choice_quiet_miss = L.choice(
        "quiet-miss",
        question="Wat antwoord je?",
        options={
            "flat": "Doorzoek een paar vragen ook door elke vector, en vergelijk",
            "scores": "Kijk naar de scores: als de index het beste punt mist, zijn ze laag",
            "right": "Dat klopt: een index die `k` punten geeft, heeft de `k` dichtstbijzijnde gevonden",
        },
        correct="flat",
        why={
            "flat": "Alleen een zoekfunctie door elke vector laat zien wat de index oversloeg. Gemeten op de {questions} vragen van de course vond de index in 1 cluster maar voor {one} dezelfde drie, terwijl hij voor elke vraag drie punten gaf.".format(
                questions=C.findings()["questions"], one=C.findings()["one"][1]),
            "scores": "Voor de vraag van het uitgewerkte voorbeeld was de beste score in 1 cluster {one}, door elke vector {all}. Ziet {one} eruit als een gemist punt, als je {all} niet kent?".format(
                one=C.number(_w["best_one"], L.lang, 3), all=C.number(_w["best_all"], L.lang, 3)),
            "right": "Een IVF index geeft altijd de `k` beste van de punten die hij vergeleek, ook als er een punt dichterbij ligt in een cluster waar hij niet keek. Kijk nog eens naar het uitgewerkte voorbeeld.",
        },
    )
    choice_quiet_miss
    return (choice_quiet_miss,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Doorzoek de course met je eigen ivf_search

    Zodra oefening 1 en 2 goed zijn, kies je een vraag van de course en het aantal clusters waarin hij kijkt. De course doorzoekt zijn index met je `ivf_search`, en laat eronder zien wat een zoekfunctie door elke vector vindt. Zoek een vraag waarvoor 1 cluster niet genoeg is.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q07")["text"][L.lang], label="Vraag", full_width=True)
    probes_slider = mo.ui.slider(start=1, stop=C.CLUSTERS, step=1, value=1, label="Clusters om in te kijken", show_value=True)
    mo.vstack([question_pick, probes_slider])
    return probes_slider, question_pick


@app.cell(hide_code=True)
def demo_search(box_lists, box_search, probes_slider, question_pick):
    _fn = C.learner_search(box_lists.value, box_search.value)
    if _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de zoekfunctie gebruikt je eigen `ivf_search`.*")
    else:
        _out = mo.Html(C.searched(_fn, question_pick.value, [probes_slider.value], L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een index verandert hoe snel de punten gevonden worden, niet wat er daarna mee gebeurt: ze gaan in de prompt zoals in [Je eerste RAG](/course/open/build-rag-first?lang=nl). In [Filters en vector databases](/course/open/build-metadata-filters?lang=nl) heeft elk punt ook metadata, zoals zijn taal en zijn les, en houdt een zoekfunctie alleen de punten die passen; een vector database houdt embeddings, metadata en index bij elkaar.
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
