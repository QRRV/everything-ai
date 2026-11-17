# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-bm25. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-bm25", "nl")
    C = L.checks   # de punten, wat de course mat, de checks, en de demo met jouw BM25


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="In deze course is log zonder grondtal de natuurlijke log, met grondtal e. Welk getal ligt het dichtst bij log(1000)?",
        options={
            "three": "3, want 10 keer 10 keer 10 is 1000",
            "seven": "Ongeveer 6,9, want e tot de macht 6,9 is ongeveer 1000",
            "hundred": "Ongeveer 368, want 1000 gedeeld door e is ongeveer 368",
        },
        correct="seven",
        why={
            "seven": "De log geeft de exponent: e tot de macht 6,9 is ongeveer 1000. Een log groeit langzaam: duizend keer zoveel is maar ongeveer 6,9 meer.",
            "three": "3 is de log met grondtal 10. Zonder grondtal betekent log in deze course grondtal e: e tot de macht 6,9 is ongeveer 1000, dus ongeveer 6,9. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "hundred": "Een log deelt niet: hij geeft de exponent. e tot de macht 6,9 is ongeveer 1000, dus ongeveer 6,9. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    Het zoeken op woorden van [Je eerste RAG](/course/open/build-rag-first?lang=nl) telt de woorden die een punt met de vraag deelt, en elk woord telt even zwaar: *the* en *of* net zo zwaar als *DBSCAN*. Maar *the* staat in bijna elk punt, dus het zegt niets over welk punt past; een zeldzaam woord zegt veel. De course gaf die zoekfunctie de titel van elk van zijn {n} lessen als vraag: hij zette voor {o1} daarvan een punt van die les bovenaan. BM25, een formule uit 1994 die zoekmachines zoals Lucene en Elasticsearch nog steeds gebruiken (voorbeelden in november 2026), zette voor {b1} de goede les bovenaan. Het is de standaardmanier om op woorden te zoeken, en naast zoeken op betekenis zit het nog in veel RAG-systemen.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt een recept met saffraan. Elk recept heeft het woord *de*, dus dat helpt je niet kiezen; maar een paar hebben *saffraan*, en dat zijn de recepten die je wilt. En een recept dat vijf keer *saffraan* zegt, gaat niet vijf keer zoveel over saffraan als een dat het één keer zegt. Dat is maar een vergelijking; hieronder meet je wat deze twee ideeën doen op de eigen samenvattingen van de course.
    """.format(n=_f["titles"], o1=_f["overlap"]["title_1"], b1=_f["bm25"]["title_1"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord of teken | Uitspraak | Wat het betekent |
    |---|---|---|
    | **lexical retrieval** | | teksten doorzoeken op de woorden zelf, zoals ze geschreven zijn; het tegenovergestelde is zoeken op betekenis |
    | **term frequency**, $f$ | | hoe vaak een woord in één tekst voorkomt |
    | **inverse document frequency**, idf | "ie-dee-ef" | hoe zeldzaam een woord is onder alle teksten: hoog als weinig teksten het hebben, bijna 0 als bijna alle het hebben |
    | **BM25** | "bee-em-vijfentwintig" | een score voor hoe goed een tekst bij een vraag past: de idf van elk woord dat ze delen, iets minder voor elke extra keer, en gecorrigeerd voor de lengte van de tekst |
    | $N$, $n$ | | het aantal teksten, en hoeveel daarvan het woord hebben |
    | $k_1$, $b$ | "k-een", "b" | de twee instellingen van BM25: hoe snel een extra keer niets meer toevoegt, en hoeveel een lange tekst gecorrigeerd wordt; meestal 1,2 en 0,75 |

    ### Drie ideeën in één formule

    **Een zeldzaam woord telt zwaarder.** De idf van een woord dat $n$ van de $N$ teksten hebben is

    $$\text{{idf}} = \log\left(\frac{{N - n + 0{{,}}5}}{{n + 0{{,}}5}} + 1\right)$$

    De 0,5 houdt hem gedefinieerd voor een woord dat geen enkele tekst heeft, en de 1 houdt hem boven 0 voor een woord dat bijna elke tekst heeft. Over de {n} punten is dit de idf van de woorden van de titel van een les, gebruikt als vraag:
    """.format(n=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def picture_idf():
    mo.Html(C.idf_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_formula():
    mo.md(r"""
    **Een extra keer telt minder, en een lange tekst krijgt geen voorsprong.** Voor elk woord van de vraag neemt BM25 de idf maal een deel dat groeit met $f$, hoe vaak de tekst het woord heeft, maar steeds minder:

    $$\text{{score}} = \sum_{{\text{{woorden van de vraag}}}} \text{{idf}} \cdot \frac{{f \cdot (k_1 + 1)}}{{f + k_1 \cdot \left(1 - b + b \cdot \frac{{\text{{lengte}}}}{{\text{{gemiddelde lengte}}}}\right)}}$$

    In een tekst van gemiddelde lengte is het deel 1 als het woord er één keer in staat, ongeveer 1,4 als het er twee keer in staat, en nooit meer dan $k_1 + 1 = 2{{,}}2$, hoe vaak ook. Een tekst die twee keer zo lang is als gemiddeld heeft meer woorden, en dus meer kans om toevallig een woord van de vraag te hebben; de lengte onder de streep maakt zijn deel kleiner. Een woord dat de tekst niet heeft, telt 0.

    ### Wat de course mat

    De course gaf de titel van elk van zijn {n} lessen als vraag, en keek of een punt van die les bovenaan kwam, met de gedeelde woorden van [Je eerste RAG](/course/open/build-rag-first?lang=nl) en met BM25:
    """.format(n=C.findings()["titles"]))
    return


@app.cell(hide_code=True)
def picture_titles():
    mo.Html(C.titles_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_titles():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        BM25 zette voor {b1} van de {n} titels de goede les bovenaan, tegen {o1} met de gedeelde woorden, en bij de eerste drie voor {b3}, tegen {o3}. Een titel is kort en vol woorden zoals *the*, *and* en *of*, die de meeste punten hebben: bij de gedeelde woorden beslissen die, BM25 telt ze nauwelijks. Op de {w} vragen uit [Je eerste RAG](/course/open/build-rag-first?lang=nl) die hun woorden delen met hun punt, deden de twee het ongeveer even goed: allebei zetten ze het goede punt {ow1} keer bovenaan, en {ow3} en {bw3} keer bij de eerste drie. Die vragen herhalen de zeldzame woorden van hun punt, dus zelfs gewoon tellen vindt het. Op de {m} vragen die het in andere woorden vragen, vonden de gedeelde woorden er {om3} en BM25 er {bm3} bij de eerste drie.
        """.format(n=_f["titles"], b1=_f["bm25"]["title_1"], o1=_f["overlap"]["title_1"], b3=_f["bm25"]["title_3"], o3=_f["overlap"]["title_3"],
                   w=_f["words"], ow1=_f["overlap"]["words_1"], ow3=_f["overlap"]["words_3"], bw3=_f["bm25"]["words_3"],
                   m=_f["meaning"], om3=_f["overlap"]["meaning_3"], bm3=_f["bm25"]["meaning_3"])),
        L.watch_out("BM25 ziet nog steeds alleen woorden zoals ze geschreven zijn. Een vraag die hetzelfde in andere woorden vraagt, *a classifier that looks at the closest stored examples* waar het punt *k-nearest neighbors* zegt, deelt geen zeldzaam woord met zijn punt, en geen formule over woorden kan het vinden. Zoeken op betekenis, met embeddings, is de volgende stap in deze module."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: zoeken met BM25

    1. **De woorden van elke tekst, als lijst:** BM25 telt hoe vaak elk woord voorkomt, dus houd elke keer.
    2. **De idf van elk woord:** tel in hoeveel teksten het voorkomt, en gebruik de formule.
    3. **De gemiddelde lengte** van een tekst, in woorden.
    4. **Geef elke tekst een score:** voor elk woord van de vraag de idf maal het deel met $f$, $k_1$ en $b$; tel ze op.
    5. **Sorteer** de teksten van de hoogste score naar de laagste, en neem de eerste paar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _bm, _ov = C.first(C.EXAMPLE, "bm25", 1)[0], C.first(C.EXAMPLE, "overlap", 1)[0]
    _rows = []
    for _p, _q in zip(C.parts(C.EXAMPLE, _ov), C.parts(C.EXAMPLE, _bm)):
        _rows.append("| *{w}* | {n} | {idf} | {f1} | {p1} | {f2} | {p2} |".format(
            w=_p["word"], n=C.number(_p["n"], L.lang), idf=C.number(_p["idf"], L.lang, 2), f1=_p["f"], p1=C.number(_p["part"], L.lang, 2), f2=_q["f"], p2=C.number(_q["part"], L.lang, 2)))
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De Engelse titel van de les over DBSCAN is *{q}*. Welk punt komt bovenaan?

    1. **De woorden:** *any*, *clusters*, *dbscan*, *of* en *shape*.
    2. **Hun idf**, over de {n} punten: in het plaatje hierboven. *dbscan* staat in {nd} punten, *of* in {no}.
    3. **De gemiddelde lengte** van een punt is {avg} woorden.
    4. **Score:** twee punten delen drie woorden met de titel. Het eerste, uit *{t1}*, zegt: *{ov}* Het tweede, uit de les over DBSCAN zelf: *{bm}*

    | woord | punten ermee | idf | eerste: $f$ | eerste: deel | tweede: $f$ | tweede: deel |
    |---|---|---|---|---|---|---|
    {rows}

    5. **Sorteer:** met de gedeelde woorden hebben ze allebei 3, en komt het eerste bovenaan; met BM25 heeft het eerste {s1} en het tweede {s2}, dus komt de les over DBSCAN bovenaan. *any* en *shape* zijn niet zeldzaam genoeg om *clusters* en *dbscan* te verslaan.
    """.format(q=C.EXAMPLE, n=C.number(len(C.POINTS), L.lang), nd=C.having("dbscan"), no=C.having("of"), avg=C.number(C.index()["average"], L.lang, 1),
               t1=C.lesson_of(_ov)["title"][L.lang], ov=_ov, bm=_bm, rows="\n".join(_rows),
               s1=C.number(sum(p["part"] for p in C.parts(C.EXAMPLE, _ov)), L.lang, 2), s2=C.number(sum(p["part"] for p in C.parts(C.EXAMPLE, _bm)), L.lang, 2)))
    return


@app.cell(hide_code=True)
def text_rung_up():
    mo.md(r"""
    ### Een trede hoger

    Een loop geeft het ene punt na het andere een score. Met NumPy alle {n} punten tegelijk: een tabel met een rij per punt en een kolom per woord van de vraag, en de formule op de hele tabel. `point.count(word)` telt hoe vaak een lijst een waarde heeft; `lengths` is een kolom, één rij per punt, dus delen door `average` en optellen bij de tabel gaat rij voor rij, zoals broadcasting doet.

    ```python
    {code}
    ```

    | Python, één punt tegelijk | NumPy, alle punten tegelijk |
    |---|---|
    | `for word in asked:` | een kolom per woord: `f` heeft een rij per punt |
    | `f = words.count(word)` | `point.count(word)` voor elk punt en woord, in `f` |
    | `idfs[word] * f * (K1 + 1) / (...)` | dezelfde formule op de tabel: `part` |
    | `score = score + ...` | `(part * idf).sum(axis=1)`: een som langs elke rij |

    Voor de titel van de les over DBSCAN zijn de {n} scores van NumPy de scores van de loop, en de hoogste is {top}.
    """.format(n=C.number(len(C.POINTS), L.lang), code=C.NUMPY_CODE.strip(),
               top=C.number(float(C.numpy_scores(C.EXAMPLE).max()), L.lang, 2)))
    return


@app.cell(hide_code=True)
def text_idf():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe zeldzaam is een woord?

        Schrijf `idf(n, total)`, die de inverse document frequency van BM25 geeft van een woord dat `n` van `total` teksten hebben, met de formule uit het idee. Het vak heeft de klassieke idf, de log van `total / n`: die deelt door nul voor een woord dat geen enkele tekst heeft. `math.log` is de natuurlijke log, net als `np.log`, voor één getal.
        """),
        L.task("Schrijf de idf van BM25: tel boven en onder de streep 0,5 op, en 1 bij de breuk, binnen de log."),
    ])
    return


@app.cell(hide_code=True)
def exercise_idf():
    box_idf = L.code_box("idf", lines=5)
    box_idf
    return (box_idf,)


@app.cell(hide_code=True)
def check_idf(box_idf):
    result_idf = L.check("idf", box_idf.value)
    result_idf
    return (result_idf,)


@app.cell(hide_code=True)
def hints_idf():
    L.hints(
        "idf",
        [
            "Boven de streep: de teksten die het woord niet hebben, plus 0,5. Eronder: de teksten die het wel hebben, plus 0,5. Dan nog 1 erbij, en de log van alles.",
            "De teksten zonder het woord zijn `total - n`. Tel 0,5 op boven en onder de streep, 1 bij de breuk, en neem `math.log` van de som.",
            "`return math.log((___) / (n + 0.5) + 1)`",
        ],
        why="""```python
import math


def idf(n, total):
    return math.log((total - n + 0.5) / (n + 0.5) + 1)
```

Zonder de `+ 1` krijgt een woord dat meer dan de helft van de teksten heeft de log van een getal onder 1: een idf onder 0, die de score zou verlagen van elke tekst die het heeft.""",
    )
    return


@app.cell(hide_code=True)
def text_bm25():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De score van een tekst

        Schrijf `bm25(asked, words, idfs, average)`, die de BM25-score van een tekst voor een vraag geeft. `asked` is de lijst met de woorden van de vraag, `words` de lijst met de woorden van de tekst, elke keer dat er een voorkomt, `idfs` een dictionary met de idf van elk gevraagd woord, en `average` het gemiddelde aantal woorden van een tekst. `K1` en `B` zijn gegeven, 1,2 en 0,75. Het vak telt voor elk gevraagd woord `idf * f` op: elke extra keer telt even zwaar als de eerste, en een lange tekst krijgt voorsprong.
        """),
        L.task("Vervang `f` door het deel van BM25: `f` maal `K1 + 1`, gedeeld door `f` plus `K1` maal de lengtecorrectie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bm25():
    box_bm25 = L.code_box("bm25", lines=6)
    box_bm25
    return (box_bm25,)


@app.cell(hide_code=True)
def check_bm25(box_bm25):
    result_bm25 = L.check("bm25", box_bm25.value)
    result_bm25
    return (result_bm25,)


@app.cell(hide_code=True)
def hints_bm25():
    L.hints(
        "bm25",
        [
            "Alleen de regel die bij `score` optelt verandert: `idfs[word]` blijft, en `f` wordt een breuk.",
            "Boven de streep: `f * (K1 + 1)`. Eronder: `f + K1 *` de lengtecorrectie, en die is `1 - B` plus `B` keer de lengte van de tekst gedeeld door `average`.",
            "`score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (___))`",
        ],
        why="""```python
def bm25(asked, words, idfs, average):
    score = 0
    for word in asked:
        f = words.count(word)
        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))
    return score
```

Als de tekst gemiddelde lengte heeft, is `1 - B + B * len(words) / average` gelijk aan 1, en telt een woord dat er één keer in staat precies zijn idf. Een woord dat de tekst niet heeft, heeft `f` 0 en telt 0.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: jouw BM25 op de punten

    Als oefening 1 en 2 goed zijn, zoekt de demo met jouw `idf` en `bm25` in de {n} punten naar je vraag, en zet de eerste drie naast die van de gedeelde woorden. De punten zijn Engels, dus vraag in het Engels. Probeer een vraag met een zeldzaam woord, en een in je eigen woorden.
    """.format(n=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def question_cell():
    question_text = mo.ui.text(value="Clusters of any shape: DBSCAN", label="Je vraag", full_width=True)
    question_text
    return (question_text,)


@app.cell(hide_code=True)
def demo_bm25(box_bm25, box_idf, question_text):
    _fns = C.learner_bm25(box_idf.value, box_bm25.value)
    if _fns is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo zoekt met je eigen `idf` en `bm25`.*")
    else:
        _words = {"title": "De eerste drie punten voor je vraag", "overlap": "Gedeelde woorden", "bm25": "Jouw BM25",
                  "empty": "Typ eerst een vraag.", "failed": "Je BM25 stopte met een fout op de punten: {why}"}
        _out = mo.Html(C.demo(_fns, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_twice():
    mo.md(r"""
    ### Oefening 3. Tien keer hetzelfde woord
    """)
    return


@app.cell(hide_code=True)
def choice_twice_cell():
    choice_twice = L.choice(
        "twice",
        question="Twee teksten van gemiddelde lengte hebben allebei het woord *loss*: de een één keer, de ander tien keer. Hoeveel meer telt *loss* mee in de score van de tweede, met BM25?",
        options={
            "ten": "Tien keer zoveel, want elke keer dat het voorkomt telt even zwaar",
            "twice": "Ongeveer twee keer zoveel: elke extra keer telt minder",
            "same": "Even veel, want BM25 kijkt alleen of het woord erin staat",
        },
        correct="twice",
        why={
            "twice": "Met f = 1 is het deel 2,2 / 2,2 = 1; met f = 10 is het 10 · 2,2 / 11,2 ≈ 1,96. Hoe vaak een woord ook voorkomt, zijn deel blijft onder k1 + 1 = 2,2.",
            "ten": "Dat doet gewoon tellen. In BM25 staat f ook onder de streep, dus groeit het deel steeds langzamer: 1 bij f = 1, ongeveer 1,96 bij f = 10, en nooit boven 2,2.",
            "same": "BM25 telt wel hoe vaak: f staat in de formule. Maar elke extra keer telt minder: 1 bij f = 1, ongeveer 1,96 bij f = 10.",
        },
    )
    choice_twice
    return (choice_twice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** BM25 vervangt de gedeelde woorden van [Je eerste RAG](/course/open/build-rag-first?lang=nl) als zoekfunctie in een RAG-systeem; waarin hij zoekt, zijn de chunks van [Documenten in stukken knippen](/course/open/build-chunking?lang=nl). Hij werkt met de log van [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl). Waar een vraag andere woorden gebruikt dan het antwoord, helpt zoeken op betekenis, dat hierna in deze module komt; en zoeken dat beide combineert, hybrid search, komt daarna.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_twice, result_bm25, result_idf):
    L.wrap_up(result_idf, result_bm25, choice_twice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
