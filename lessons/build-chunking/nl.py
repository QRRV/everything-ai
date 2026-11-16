# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-chunking. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-chunking", "nl")
    C = L.checks   # de documenten, wat de course mat, de checks, en de demo met jouw chunk


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Van bestand naar tekst](/course/open/build-documents-to-text?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parse():
    quick_parse = L.choice(
        "quick-parse",
        question="Een webpagina gaat je knowledge base in met alleen de tags eruit gehaald. Wat komt er nog meer in de tekst terecht?",
        options={
            "nothing": "Niets anders: zonder tags is een pagina de tekst die de schrijver schreef",
            "boilerplate": "Het menu, de footer en de code van de pagina, rond de tekst die ertoe doet",
            "pictures": "Een beschrijving van elk plaatje op de pagina, die de browser maakt",
        },
        correct="boilerplate",
        why={
            "boilerplate": "Als je de tags weghaalt, blijft alle tekst ertussen staan: het menu en de footer van elke pagina, en de code van de scripts. Bij ingestion zoek je de hoofdtekst en laat je de rest weg.",
            "nothing": "De tekst tussen de tags is meer dan het stuk zelf: het menu, de footer en de code van de scripts zijn ook tekst. Zie [Van bestand naar tekst](/course/open/build-documents-to-text?lang=nl).",
            "pictures": "Een browser maakt geen beschrijvingen; alleen wat in de pagina zelf staat blijft over, zoals het menu, de footer en de code van de scripts. Zie [Van bestand naar tekst](/course/open/build-documents-to-text?lang=nl).",
        },
    )
    quick_parse
    return (quick_parse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het zoeken op woorden van [Je eerste RAG](/course/open/build-rag-first?lang=nl) ging punten van zo'n dertig woorden langs. Echte documenten zijn langer: een handboek, een rapport, een pagina met notities. Zoek je in hele documenten, dan weet je welk document bij de vraag past, maar niet waar het antwoord erin staat, en het model moet dan het hele document lezen om het te vinden. Drie documenten van een paar duizend woorden zijn bij elke vraag duizenden tokens, en ze verstoppen die ene zin die ertoe doet tussen alle andere. Daarom knipt een RAG-systeem zijn documenten eerst in stukken, en zoekt het in die stukken.

    **Een voorbeeld uit het dagelijks leven.** Het register achter in een boek zegt niet *dit boek gaat over fietsen*; het zegt *remmen, bladzijde 41*. Je slaat één bladzijde open, niet het hele boek. Dat is maar een vergelijking; hieronder meet je hoe vaak het goede stuk bovenaan komt als de course zijn eigen samenvattingen op verschillende manieren knipt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.document_words()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **chunk** | een stuk van een document dat als één geheel wordt bewaard en doorzocht, van een zin tot een paar honderd woorden lang |
    | **chunking** | documenten in chunks knippen voordat je erin zoekt |
    | **chunk overlap** | de woorden die twee naburige chunks delen: de volgende chunk begint iets vóór het eind van de vorige |

    ### Drie manieren om te knippen

    - **Een vaste grootte:** elke chunk heeft hetzelfde aantal woorden (of tokens), achter elkaar. Eenvoudig, maar een knip kan midden in een zin vallen, tussen de woorden van een vraag en het antwoord. Een overlap maakt het minder waarschijnlijk dat wat bij elkaar hoort nooit samen in één chunk staat.
    - **Langs de structuur:** knip waar de tekst zelf grenzen heeft, bij kopjes en alinea's, en knip alleen een stuk dat dan nog te lang is. Een *recursieve* splitter gaat daarna door: een alinea die te lang is, knipt hij bij zinnen, een zin die te lang is bij woorden.
    - **Langs de betekenis:** knip waar het onderwerp wisselt (*semantic chunking*). Daarvoor moet je de betekenis van zinnen kunnen vergelijken, en dat komt in een latere les over embeddings.

    ### Wat de course mat

    De course zette de punten van de samenvattingen van elke module achter elkaar, met een lege regel tussen twee punten: {n} documenten van {lo} tot {hi} woorden, één per module. Daarna knipte hij ze op twaalf manieren, en liet hij voor elk van de {q} vragen uit [Je eerste RAG](/course/open/build-rag-first?lang=nl) die hun woorden delen met hun punt, het zoeken op woorden van die les de chunks op volgorde zetten. Elk vierkantje is een vraag; de legenda onder het plaatje zegt of de chunk die bovenaan kwam het antwoord bevat:
    """.format(n=len(C.DOCUMENTS), lo=_w[0], hi=C.number(_w[-1], L.lang), q=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ways():
    _f = C.findings()
    _k = {w: sum(_f[w]["found"]) for w in C.WAYS}
    mo.vstack([
        mo.md(r"""
        Met chunks van 10 woorden bevatte de eerste chunk het antwoord voor {t10} van de {q} vragen: de woorden van de vraag en het antwoord kwamen vaak in verschillende chunks terecht, en een chunk met veel woorden uit de vraag had niet het getal waar de vraag om vroeg. Grotere chunks vonden meer, tot zo'n 80 woorden, en een overlap vond er bij elke grootte een of twee meer: {t10} en {o10} bij 10 woorden, {t40} en {o40} bij 40, {t80} en {o80} bij 80. Knippen langs de alinea's, die hier de punten zelf zijn, vond er {para}, met {para_words} woorden per chunk. Hele documenten vonden er {whole}, maar een heel document heeft gemiddeld {whole_words} woorden: de drie beste zijn zo'n {three} woorden om te lezen voor het model, waar drie alinea's er ongeveer {three_para} zijn.
        """.format(q=len(C.QUESTIONS), t10=_k[("fixed", 10, 0)], o10=_k[("fixed", 10, 2)], t40=_k[("fixed", 40, 0)], o40=_k[("fixed", 40, 10)],
                   t80=_k[("fixed", 80, 0)], o80=_k[("fixed", 80, 20)], para=_k[("paragraph", 80, 0)], whole=_k[("whole", 0, 0)],
                   para_words=C.number(_f[("paragraph", 80, 0)]["words"], L.lang), whole_words=C.number(_f[("whole", 0, 0)]["words"], L.lang),
                   three=C.number(3 * _f[("whole", 0, 0)]["words"], L.lang, 0), three_para=C.number(3 * _f[("paragraph", 80, 0)]["words"], L.lang, 0))),
        L.watch_out("Zeventien vragen zijn weinig: één vraag meer of minder kan toeval zijn. Wat overeind blijft, is de vorm: chunks die veel te klein zijn verliezen het verband tussen een vraag en het antwoord, chunks die veel te groot zijn laten het model veel lezen, en knippen waar de tekst zelf grenzen heeft doet het goed. Meet je eigen documenten met vragen waarvan je het antwoord kent."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: documenten in chunks knippen

    1. **Knip langs de structuur** die de tekst heeft: kopjes, alinea's.
    2. **Knip wat dan nog te lang is** op een vaste grootte.
    3. **Laat naburige chunks een beetje overlappen**, vaak een tiende tot een kwart van hun grootte, waar je op een vaste grootte knipt.
    4. **Meet** met vragen waarvan je het antwoord kent: hoe vaak komt de goede chunk bovenaan?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _pieces = C.chunk(C.EXAMPLE, 40, 10)
    _n = len(C.EXAMPLE.split())
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het kleinste document met meer dan één les is de module over recurrente netwerken: {n} woorden. Knip het in chunks van 40 woorden met een overlap van 10.

    1. **De structuur:** de zes punten zijn de alinea's; hier knippen we op een vaste grootte, om te zien wat dat doet.
    2. **Een vaste grootte:** de eerste chunk is woord 0 tot en met 39. Met een overlap van 10 begint elke volgende chunk 40 − 10 = 30 woorden verder: bij 0, 30, 60, 90 en 120. De chunk bij 120 heeft woord 120 tot en met 158, het eind van het document, dus dat is de laatste: {k} chunks.
    3. **De overlap:** de eerste chunk eindigt met *{end}*, en de tweede begint met dezelfde tien woorden. Een zin die de eerste knip breekt, staat heel in een van de twee.
    4. **Meten:** dat deed het plaatje hierboven, voor alle {docs} documenten.
    """.format(n=_n, k=len(_pieces), end=" ".join(_pieces[0].split()[-10:]), docs=len(C.DOCUMENTS)))
    return


@app.cell(hide_code=True)
def text_chunk():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Chunks met een overlap

        Schrijf `chunk(text, size, overlap)`, die `text` knipt in chunks van `size` woorden, als teksten met een spatie tussen de woorden. Elke volgende chunk begint `overlap` woorden vóór het eind van de vorige, en de chunk die het eind van de tekst haalt is de laatste: geen chunk mag binnen de vorige liggen. Een lege tekst geeft een lege lijst. Het vak knipt chunks achter elkaar, zonder overlap. `break` stopt een `while`-loop meteen. Als hij goed is, bewaart de check hem ook als bestand, `work/projects/app/chunks.py`: [Project: je assistent leest je documenten](/course/open/build-project-rag?lang=nl) gebruikt hem.
        """),
        L.task("Stap `size - overlap` woorden verder, en stop na de chunk die het eind haalt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chunk():
    box_chunk = L.code_box("chunk", lines=10)
    box_chunk
    return (box_chunk,)


@app.cell(hide_code=True)
def check_chunk(box_chunk):
    result_chunk = L.check("chunk", box_chunk.value)
    result_chunk
    return (result_chunk,)


@app.cell(hide_code=True)
def hints_chunk():
    L.hints(
        "chunk",
        [
            "Twee veranderingen: de stap van de ene chunk naar de volgende, en vlak na `append` een test of deze chunk het eind al haalde.",
            "Het volgende begin is `start + size - overlap`. Deze chunk haalt het eind als `start + size >= len(words)`; dan `break`.",
            "Na de `append`: `if start + size >= len(words):` met `break` eronder, dan `start = start + size - overlap`.",
        ],
        why="""```python
def chunk(text, size, overlap):
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start = start + size - overlap
    return out
```

Zonder de `break` gaat de loop door zolang `start` nog in de tekst valt, en na de chunk die het eind haalde komt er nog één, van de laatste paar woorden, die de chunk ervoor al had.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: jouw chunk op de documenten

    Als oefening 1 goed is, knipt de demo de {n} documenten met jouw `chunk`, op de grootte en overlap die je kiest, en zet het zoeken op woorden voor elk van de {q} vragen de chunks op volgorde. Welke grootte vindt het meest? En wat voegt de overlap toe?
    """.format(n=len(C.DOCUMENTS), q=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def settings_cell():
    size_slider = mo.ui.slider(start=10, stop=160, step=10, value=40, label="Woorden per chunk", show_value=True)
    overlap_choice = mo.ui.dropdown(options={"geen": 0, "een tiende": 10, "een kwart": 4, "de helft": 2}, value="geen", label="Overlap")
    mo.hstack([size_slider, overlap_choice], justify="start", gap=2)
    return overlap_choice, size_slider


@app.cell(hide_code=True)
def demo_chunk(box_chunk, overlap_choice, size_slider):
    _fn = C.learner_chunk(box_chunk.value)
    if _fn is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo knipt met je eigen `chunk`.*")
    else:
        _size = size_slider.value
        _overlap = _size // overlap_choice.value if overlap_choice.value else 0
        _words = {"title": "Jouw chunk op de documenten: {size} woorden, {overlap} overlap",
                  "result": "De eerste chunk bevatte het antwoord voor {k} van de {n} vragen; {chunks} chunks van gemiddeld {w} woorden.",
                  "failed": "Je chunk stopte met een fout op de documenten: {why}"}
        _out = mo.Html(C.demo(_fn, _size, _overlap, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_paragraph():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Langs de alinea's

        Schrijf `by_paragraph(text, size)`, die elke alinea van `text` als chunk geeft, behalve een alinea van meer dan `size` woorden: die knipt hij met `chunk(paragraph, size, 0)`, en de stukken voegt hij één voor één toe. Tussen twee alinea's staat een lege regel, dus `text.split("\n\n")` geeft ze: `split` met een tekst tussen de haakjes knipt bij die tekst in plaats van bij elke spatie. `chunk` is gegeven. Het vak geeft elke alinea als chunk, ook een die te lang is.
        """),
        L.task("Knip alleen een alinea van meer dan `size` woorden, en voeg de stukken één voor één toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paragraph():
    box_paragraph = L.code_box("by_paragraph", lines=9)
    box_paragraph
    return (box_paragraph,)


@app.cell(hide_code=True)
def check_paragraph(box_paragraph):
    result_paragraph = L.check("by_paragraph", box_paragraph.value)
    result_paragraph
    return (result_paragraph,)


@app.cell(hide_code=True)
def hints_paragraph():
    L.hints(
        "by_paragraph",
        [
            "Beslis in de loop per alinea: is hij te lang of niet? Een lange gaat erin als een paar stukken, een korte zoals hij is.",
            "`len(paragraph.split())` telt de woorden. Is dat meer dan `size`, ga dan `chunk(paragraph, size, 0)` langs en `append` elk stuk; anders `append` je de alinea.",
            "`if len(paragraph.split()) > size:` met `for piece in chunk(paragraph, size, 0):` en `out.append(piece)` eronder, dan `else:` met `out.append(paragraph)`.",
        ],
        why="""```python
def by_paragraph(text, size):
    out = []
    for paragraph in text.split("\\n\\n"):
        if len(paragraph.split()) > size:
            for piece in chunk(paragraph, size, 0):
                out.append(piece)
        else:
            out.append(paragraph)
    return out
```

`out.append(chunk(paragraph, size, 0))` zou de hele lijst stukken als één item toevoegen: een lijst binnen je lijst, die een zoekfunctie niet als tekst kan lezen.""",
    )
    return


@app.cell(hide_code=True)
def text_small():
    mo.md(r"""
    ### Oefening 3. Te klein
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    _f = C.findings()
    choice_small = L.choice(
        "small",
        question="Met chunks van 10 woorden bevatte de eerste chunk het antwoord voor maar {k} van de {n} vragen. Wat is de waarschijnlijkste reden?".format(
            k=sum(_f[("fixed", 10, 0)]["found"]), n=len(C.QUESTIONS)),
        options={
            "slow": "Met {c} chunks moet de zoekfunctie zoveel teksten langs dat hij er een paar overslaat".format(c=C.number(_f[("fixed", 10, 0)]["chunks"], L.lang)),
            "split": "Het antwoord en de woorden van de vraag eromheen komen in verschillende chunks",
            "model": "Een chunk van 10 woorden is te kort voor een language model om te lezen, dus wordt hij overgeslagen",
        },
        correct="split",
        why={
            "split": "De zoekfunctie zet de chunk met de meeste woorden uit de vraag bovenaan; in 10 woorden mist die chunk vaak het getal waar de vraag om vraagt, dat in de chunk ernaast staat.",
            "slow": "De zoekfunctie gaat elke chunk langs, hoeveel het er ook zijn; hij slaat niets over. Het probleem is wat een chunk van 10 woorden bevat: de woorden van de vraag en het antwoord komen vaak in verschillende chunks.",
            "model": "Bij deze meting leest geen model iets: alleen het zoeken op woorden zet de chunks op volgorde. Het probleem is dat de woorden van de vraag en het antwoord in verschillende chunks komen.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De documenten komen uit [Van bestand naar tekst](/course/open/build-documents-to-text?lang=nl), en de zoekfunctie die de chunks op volgorde zet, is die van [Je eerste RAG](/course/open/build-rag-first?lang=nl). Die telt elk gedeeld woord even zwaar, *the* net zo zwaar als *perplexity*; [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl) laat een zeldzaam woord zwaarder tellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_chunk, result_paragraph):
    L.wrap_up(result_chunk, result_paragraph, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
