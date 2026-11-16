# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-metadata-filters. Route: course/route/lessen/bouwen.yaml. Checks, de store en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-metadata-filters", "nl")
    C = L.checks   # de punten van de course in beide talen met hun metadata, de checks en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Snel zoeken in veel vectoren](/course/open/build-vector-index?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_index():
    quick_index = L.choice(
        "quick-index",
        question="Een IVF index kijkt in 1 van zijn 16 clusters, en geeft drie punten. Welke drie zijn dat?",
        options={
            "cluster": "De beste drie van de vectoren in het dichtstbijzijnde cluster",
            "all": "De drie vectoren die het dichtst bij de vraag liggen, van alle vectoren",
            "centroids": "De drie centroids die het dichtst bij de vraag liggen",
        },
        correct="cluster",
        why={
            "cluster": "De zoekfunctie vergelijkt de vraag alleen met de vectoren in de lijsten van de clusters waarin hij kijkt. Een vector in een ander cluster die dichterbij ligt, wordt nooit vergeleken, dus die kan hij missen.",
            "all": "Dat geeft een zoekfunctie door elke vector. De index vergelijkt alleen de vectoren van de clusters waarin hij kijkt. Zie [Snel zoeken in veel vectoren](/course/open/build-vector-index?lang=nl).",
            "centroids": "De centroids bepalen alleen in welke clusters de zoekfunctie kijkt; de punten die hij geeft, komen uit de lijsten van die clusters. Zie [Snel zoeken in veel vectoren](/course/open/build-vector-index?lang=nl).",
        },
    )
    quick_index
    return (quick_index,)


@app.cell(hide_code=True)
def text_why():
    _l = C.languages()["en"]
    mo.md(r"""
    ## Waarom

    De course bestaat in het Engels en in het Nederlands. Een assistent ervoor kan de punten van allebei doorzoeken: {items} items. Maar zoeken op betekenis vergelijkt alleen embeddings; het weet niets van de taal van een punt, zijn les of zijn niveau. In het Engels gesteld kregen de {questions} vragen van de course maar {same} Engelse punten tussen hun {all} resultaten, en voor {twice} ervan bevatten de beste drie hetzelfde punt twee keer. Een bedrijf heeft hetzelfde probleem, met meer op het spel: documenten van elke afdeling in één knowledge base, en sommige die maar een paar mensen mogen lezen. De zoekfunctie moet meer van elke tekst weten dan zijn embedding.

    **Een voorbeeld uit het dagelijks leven.** In een webwinkel zoek je op *regenjas*, vink je *maat M* en *op voorraad* aan, en zie je alleen de jassen die passen. De zoekfunctie gaat nog steeds af op wat je typte; de vinkjes bepalen welke producten hij mag laten zien. Dat is alleen een vergelijking: de zoekfunctie hieronder vergelijkt lijsten van getallen.
    """.format(items=C.number(len(C.store()), L.lang), questions=len(C.QUESTIONS), same=_l["same"], all=C.K * len(C.QUESTIONS), twice=_l["twice"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.worked()
    _item = C.store()[_w["first"][0]]
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **metadata** | data over data: hier voor elk punt zijn taal, zijn les, zijn domein en zijn niveau |
    | **metadata filtering** | een zoekfunctie beperken tot de items waarvan de metadata aan een voorwaarde voldoen, zoals de taal `"nl"` |
    | **vector database** | een database die embeddings bewaart samen met hun tekst en metadata, en snel antwoord geeft op zoekopdrachten, meestal met filters op de metadata |
    | `where` | een dict met de metadata die een item moet hebben, zoals `{{"lang": "nl", "level": "basis"}}` |

    De course maakte een store van zijn punten in beide talen: {items} items, één per punt en taal. Elk item is een dict met de tekst, de embedding en de metadata:

    ```python
    {{"text": "...", "vector": np.array([...]), "lang": "{lang}", "lesson": "{lesson}", "domain": "{domain}", "level": "{level}"}}
    ```

    Een filter kan op twee momenten komen. *Achteraf*: doorzoek alle items, neem de `k` beste, en laat weg wat niet past. Dat is eenvoudig, maar het kan minder dan `k` overlaten, of niets. *Eerst*: houd alleen de items die passen, en doorzoek die. Dan krijg je `k` resultaten zodra er `k` items passen.

    Een vector database doet dit voor je, en snel. Qdrant, Chroma, Weaviate en Milvus zijn ervoor gemaakt, en pgvector voegt het toe aan PostgreSQL, een gewone database. Ze houden de embedding, de tekst en de metadata van elke chunk bij elkaar, bouwen een index, vaak HNSW, en filteren tijdens het zoeken, zodat een filter niet te weinig overlaat. Libraries zoals LangChain noemen het een *vector store*. Voor een paar duizend chunks is een lijst in het geheugen, zoals hier, genoeg.
    """.format(items=C.number(len(C.store()), L.lang), lang=_item["lang"], lesson=_item["lesson"], domain=_item["domain"], level=_item["level"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: zoeken met een filter

    1. **Eén keer:** geef elk item zijn metadata als het in de store komt.
    2. **Filter:** houd de items waarvan de metadata bij `where` passen.
    3. **Score:** neem het inproduct van de vraag met alleen die items.
    4. **Houden:** de `k` met de hoogste score, de hoogste eerst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{nl}* Een leerling die de course in het Nederlands leest, stelt deze vraag in een les van domein `bouwen`, dus de zoekfunctie gebruikt `where = {where}`. Er passen {fits} items.

    1. **Achteraf:** de drie beste van alle {items} items zijn punten van *{t0}*, *{t1}* en *{t2}*, uit de domeinen `{d0}`, `{d1}` en `{d2}`. Geen ervan past, dus achteraf filteren laat niets over.
    2. **Eerst:** van de {fits} items die passen, is het beste een punt van *{f0}*: *{p0}* Dat beantwoordt de vraag; tussen alle items stond het op plaats {pl0}. Daarna komen punten van *{f1}* en *{f2}*, op plaats {pl1} en {pl2}.
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
    ### Wat de course mat

    De course stelde alle {questions} vragen in het Nederlands, met drie filters, en filterde achteraf en eerst:
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
    Achteraf filteren op alleen de taal hield gemiddeld {k0} van de drie over. Op `{w1}` hield het niets over voor {n1} van de {questions} vragen, op `{w2}` voor {n2}. Eerst filteren gaf elke keer drie voor elke vraag.

    Zonder filter kregen de Engelse vragen {same_en} Engelse punten tussen hun {all} resultaten, en de Nederlandse {same_nl} Nederlandse punten. Voor {twice_en} van de Engelse vragen en {twice_nl} van de Nederlandse bevatten de beste drie hetzelfde punt twee keer, in beide talen. Met een filter op de taal van de vraag zat het goede punt bij de drie voor {rf_en} van de {answerable} vragen die een punt beantwoordt in het Engels, tegen {r_en} zonder; in het Nederlands {rf_nl} tegen {r_nl}.

    Met de IVF index van [Snel zoeken in veel vectoren](/course/open/build-vector-index?lang=nl) is een filter lastiger: {empty} van zijn 16 clusters bevatten geen punt van domein `{domain}`. Als hij in 1 cluster keek en daarin de punten van `{domain}` hield, vond de zoekfunctie niets voor {i1} van de {questions} Engelse vragen, en minder dan drie voor nog {f1}; in 2 clusters niets voor {i2}. Daarom filteren vector databases tijdens het zoeken, niet achteraf.
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
        ## Zelf doen

        ### Oefening 1. Past een item?

        `matches(item, where)` hoort `True` te geven als het item voor elke key van `where` de waarde heeft die `where` ervoor heeft, en anders `False`. Een lege `where` laat elk item door. Met `item = {note}` geeft `matches(item, {{"lang": "en", "level": "basis"}})` `False`: het niveau past, de taal niet. De check telt ook hoeveel van de {items} items van de course volgens je functie bij een paar filters passen.
        """.format(note=C.where_text(C.NOTE), items=C.number(len(C.store()), L.lang))),
        L.task("Geef `False` zodra een key van `where` in `item` een andere waarde heeft."),
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
            "Ga de keys van `where` een voor een langs, en vergelijk de waarde van het item met die van `where`.",
            "`for key in where:` gaat de keys langs; `item[key] != where[key]` is `True` als het item voor die key niet past.",
            "In de loop: `if item[key] != where[key]:` en dan `return False`; na de loop `return True`.",
        ],
        why="""```python
def matches(item, where):
    for key in where:
        if item[key] != where[key]:
            return False
    return True
```

De functie kan nee zeggen bij de eerste key die niet past, maar ja pas na de laatste: pas dan passen ze allemaal.""",
    )
    return


@app.cell(hide_code=True)
def text_search():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eerst filteren

        `filtered_search(q, store, where, k)` hoort de teksten te geven van de `k` items met het hoogste inproduct met `q`, tussen de items die bij `where` passen. `store` is een lijst met items zoals hierboven, en `matches` is gegeven. Het vak filtert achteraf: het neemt de `k` beste van alle items, en laat daarna weg wat niet past. De kleine store van het voorbeeld heeft zes notities:

        ```python
        {notes}
        ```

        `filtered_search(np.array([1.0, 0.2]), NOTES, {{"lang": "nl"}}, 2)` hoort `["k-means in het kort", "HNSW uitgediept"]` te geven. Het vak geeft alleen `["k-means in het kort"]`, omdat de op een na beste van allemaal, `"k-means in brief"`, Engels is. De check draait je functie ook op de store van de course, voor de vraag van het uitgewerkte voorbeeld.
        """.format(notes=C.NOTES_CODE.replace("\n", "\n        "))),
        L.task("Geef alleen de items die bij `where` passen een score, zodat je de `k` beste daarvan houdt."),
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
            "Het filter komt nu na `sorted(...)[:k]`. Het hoort ervoor te komen: een item dat niet past, hoort nooit een score te krijgen.",
            "Zet in de loop `if matches(store[i], where):` voor de regel met `found.append(...)`, die een stap verder inspringt.",
            "Zet `if matches(store[i], where):` boven `found.append(...)`, en haal `if matches(store[i], where)` uit de laatste regel.",
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

De items die niet passen, krijgen nooit een score, dus de `k` beste zijn altijd `k` die passen, zolang er zoveel zijn.""",
    )
    return


@app.cell(hide_code=True)
def text_twice():
    mo.md(r"""
    ### Oefening 3. Hetzelfde punt twee keer

    Voor de Engelse vraag *{question}* zijn de drie beste van de store met beide talen hetzelfde punt in het Nederlands en in het Engels, en nog één punt.
    """.format(question=C.question("q11")["text"]["en"]))
    return


@app.cell(hide_code=True)
def choice_twice_cell():
    _s = C.twice_scores()
    choice_twice = L.choice(
        "twice",
        question="Waarom komt hetzelfde punt twee keer?",
        options={
            "copy": "De Nederlandse punten hebben dezelfde embeddings als de Engelse",
            "store": "De store bevat elk Nederlands punt per ongeluk twee keer",
            "translation": "Granite zet een tekst en zijn vertaling dicht bij elkaar",
        },
        correct="translation",
        why={
            "translation": "Granite is zo getraind dat een tekst en zijn vertaling voor hem hetzelfde betekenen, en daarom vindt een Nederlandse vraag ook een Engels punt. Hier scoorde het Nederlandse punt {nl} en het Engelse {en}. Voor een assistent die in één taal antwoordt, neemt de tweede kopie alleen de plek van een ander punt in: filter op de taal van de vraag.".format(
                nl=C.number(_s["nl"], L.lang, 3), en=C.number(_s["en"], L.lang, 3)),
            "copy": "Dan zouden ze precies even hoog scoren. Het Nederlandse punt scoorde {nl}, het Engelse {en}: dicht bij elkaar, niet hetzelfde.".format(
                nl=C.number(_s["nl"], L.lang, 3), en=C.number(_s["en"], L.lang, 3)),
            "store": "De store heeft {items} items: elk punt één keer in het Engels en één keer in het Nederlands. De twee die kwamen, zijn niet hetzelfde item.".format(
                items=C.number(len(C.store()), L.lang)),
        },
    )
    choice_twice
    return (choice_twice,)


@app.cell(hide_code=True)
def text_access():
    mo.md(r"""
    ### Oefening 4. Wie mag het lezen?

    Een bedrijf bewaart de documenten van elke afdeling in één vector database, voor een assistent die er vragen mee beantwoordt. Alleen HR mag de salarisdocumenten lezen.
    """)
    return


@app.cell(hide_code=True)
def choice_access_cell():
    choice_access = L.choice(
        "access",
        question="Hoe houd je de salarisdocumenten buiten de antwoorden voor alle anderen?",
        options={
            "prompt": "Zeg het model in zijn system prompt nooit uit salarisdocumenten te citeren",
            "filter": "Filter bij het zoeken op metadata die zegt wie een chunk mag lezen",
            "answer": "Controleer elk antwoord op geldbedragen, en houd die tegen",
        },
        correct="filter",
        why={
            "filter": "Dan bereikt een chunk die iemand niet mag lezen nooit de prompt, en wat het model nooit ziet, kan het niet doorgeven. Het filter hoort in de zoekfunctie zelf, voordat de `k` beste genomen worden; achteraf kan het te weinig overlaten, zoals de course mat.",
            "prompt": "Wat in de prompt staat, kan het model doorgeven: een instructie maakt de kans kleiner, maar niet nul. Beter dat de salarisdocumenten de prompt nooit bereiken.",
            "answer": "Dan bereiken de salarisdocumenten het model nog steeds, en een salaris in woorden, of een zin over wie het meest verdient, komt door je controle.",
        },
    )
    choice_access
    return (choice_access,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Doorzoek de course met je eigen filtered_search

    Zodra oefening 1 en 2 goed zijn, kies je een vraag en een filter. De course doorzoekt zijn store met je `filtered_search` en je `matches`, en laat eronder de drie beste zonder filter zien. Probeer een filter waar weinig items bij passen, zoals een domein met een niveau.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q11")["text"][L.lang], label="Vraag", full_width=True)
    lang_pick = mo.ui.dropdown(options={"alle": "", "en": "en", "nl": "nl"}, value="alle", label="lang")
    domain_pick = mo.ui.dropdown(options={"alle": "", **{d: d for d in C.domains()}}, value="alle", label="domain")
    level_pick = mo.ui.dropdown(options={"alle": "", **{lv: lv for lv in C.LEVELS}}, value="alle", label="level")
    mo.vstack([question_pick, mo.hstack([lang_pick, domain_pick, level_pick], justify="start", wrap=True)])
    return domain_pick, lang_pick, level_pick, question_pick


@app.cell(hide_code=True)
def demo_search(box_matches, box_search, domain_pick, lang_pick, level_pick, question_pick):
    _fn = C.learner_search(box_matches.value, box_search.value)
    if _fn is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de zoekfunctie gebruikt je eigen `matches` en `filtered_search`.*")
    else:
        _where = C.where_of(lang_pick.value, domain_pick.value, level_pick.value)
        _out = mo.Html(C.searched(_fn, question_pick.value, _where, L.lang, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een filter bepaalt welke punten gevonden mogen worden; welke daarvan het best gevonden worden, is de volgende vraag. Zoeken op betekenis miste het punt over de agent, dat BM25 als eerste vond, in [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl). [Woorden en betekenis samen](/course/open/build-hybrid-search?lang=nl) voegt de twee zoekfuncties samen, omdat elk punten vindt die de ander mist.
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
