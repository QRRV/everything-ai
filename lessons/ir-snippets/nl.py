# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-snippets. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-snippets", "nl")
    C = L.checks   # de alinea's en vragen van XQuAD, de resultaten van BM25, de snippets en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl) en [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question=r'Wat geeft `re.findall(r"\S+", "U.S. troops (1855), left.")`?',
        options={
            "letters": r"`['U', 'S', 'troops', '1855', 'left']`: de reeksen letters en cijfers",
            "spaces": r"`[' ', ' ', ' ']`",
            "right": r"`['U.S.', 'troops', '(1855),', 'left.']`",
            "sentences": r"`['U.S.', 'troops (1855), left.']`: geknipt na de eerste punt",
        },
        correct="right",
        why={
            "right": r"`\S` past op één teken dat geen witruimte is, dus `\S+` is een reeks tekens tussen spaties, en elk woord houdt de hoofdletters en de leestekens: *(1855),* houdt de haakjes en de komma. Deze les knipt alinea's zo, zodat een snippet elk woord kan tonen zoals het geschreven staat.",
            "letters": r"Dat zijn de reeksen van `\w`, letters en cijfers. Op welke tekens past `\S`? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "spaces": r"Dat zijn de reeksen van `\s`, de witruimte. Welke tekens passen er met een hoofdletter S? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "sentences": r"`re.findall` geeft elke reeks die past, geen zinnen: waar stopt een reeks van `\S`? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def quick_check_bm25():
    quick_bm25 = L.choice(
        "quick-bm25",
        question="Twee alinea's van elk 87 woorden bevatten twee keer *Victoria* en twee keer *parlement*, en geen andere zoekwoorden. In alinea A staan de vier woorden in één zin; in alinea B staan de vier woorden verspreid over de alinea. Welke alinea krijgt de hoogste BM25-score bij de zoekvraag *parlement Victoria*?",
        options={
            "near": "A: BM25 geeft woorden die dicht bij elkaar staan een hogere score",
            "spread": "B: verspreide woorden beslaan meer van de alinea, dus elk woord voegt meer toe aan de score",
            "first": "A, als de zin vooraan staat",
            "right": "even hoog: BM25 ziet de aantallen en de lengte, niet waar de woorden staan",
        },
        correct="right",
        why={
            "right": "BM25 geeft elk zoekwoord een waarde uit het aantal keer dat het voorkomt, de IDF en de lengte van de alinea, en telt deze waarden op: waar een woord staat, speelt geen rol. Dat is genoeg om de alinea's te rangschikken, maar een lezer wil ook zien waar de zoekwoorden staan. Deze les vindt de plek van de zoekwoorden.",
            "near": "Kijkt BM25 naar waar een woord staat, of alleen naar hoe vaak het voorkomt? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "spread": "Welke getallen bepalen de waarde van een woord in BM25: de plaatsen in de alinea, of het aantal, de IDF en de lengte? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "first": "Staat er in de formule van BM25 iets over de volgorde van de zinnen? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
        },
    )
    quick_bm25
    return (quick_bm25,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl) rangschikte BM25 bij elke vraag van XQuAD de 2.067 alinea's. Een zoekmachine laat een lezer deze scores niet zien. De zoekmachine toont een pagina met resultaten, en bij elk resultaat een titel en een paar regels tekst; op grond van deze regels beslist de lezer welk resultaat de moeite van het openen waard is, en welk niet. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze schrijven in *Introduction to Information Retrieval* (2008) dat de gebruikelijke oplossing een snippet is: een korte samenvatting van het document, bedoeld om de gebruiker te laten beslissen of het document relevant is.

    Welke regels moeten dat zijn? De eerste regels van een document zijn bij elke zoekvraag hetzelfde. In 1998 probeerden Anastasios Tombros en Mark Sanderson iets anders. Het tweetal ging uit van een oude overtuiging: wie in de lijst met resultaten de zinnen ziet waarin de zoekwoorden voorkomen, kan beter beoordelen of een document relevant is.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt een woord op in het register achter in een boek en slaat de pagina open. Je oog gaat meteen naar het woord en de regels eromheen; je leest de pagina niet vanaf boven. Een snippet doet dat bij een zoekopdracht voor elk resultaat tegelijk.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_opened = L.guess(
        "guess",
        question="Tombros en Sanderson gaven 20 mensen vijf minuten per zoekvraag om te beoordelen welke documenten van een lijst met resultaten relevant waren. Met de eerste zinnen van elk document als samenvatting openden de deelnemers de volledige tekst van 23,7% van de beoordeelde documenten. Hoe vaak openden de deelnemers de volledige tekst met samenvattingen uit de zinnen met de zoekwoorden?",
        options={
            "same": "ongeveer even vaak: 22%",
            "half": "half zo vaak: 12%",
            "right": "ongeveer één document op de 75: 1,3%",
            "more": "vaker, 31%: door de zoekwoorden wilden de deelnemers verder lezen",
        },
        reveal="Ongeveer één document op de 75: 1,32%, of 0,3 documenten per zoekvraag, tegen 4,74 met de eerste zinnen. Met de samenvattingen voor de zoekvraag beoordeelden de deelnemers in de vijf minuten ook meer documenten, 22,62 tegen 20, en vonden de deelnemers meer van de relevante documenten: 65,6% tegen 49,76%. Volgens Tombros en Sanderson hebben gebruikers meer aanwijzingen nodig om de relevantie van een document vast te stellen, en vooral aanwijzingen over de context waarin de zoekwoorden in het document gebruikt worden.",
    )
    guess_opened
    return (guess_opened,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Snippet.** Volgens *Introduction to Information Retrieval* bestaat een snippet meestal uit de titel van het document en een korte samenvatting, automatisch uit het document gehaald. Een pagina met resultaten toont elk resultaat met de titel en de snippet. De alinea's van XQuAD komen van Wikipedia, dus de titel is de titel van de Wikipedia-pagina.

    **Statische en dynamische samenvattingen.** Het boek onderscheidt twee soorten samenvattingen: statische samenvattingen zijn altijd hetzelfde, welke zoekvraag er ook komt, en dynamische samenvattingen zijn aangepast aan de informatiebehoefte die uit de zoekvraag blijkt. Een statische samenvatting kan het begin van het document zijn, volgens het boek in de eenvoudigste vorm de eerste twee zinnen of de eerste 50 woorden, of een beschrijving die de maker van een webpagina in de HTML zet. Tombros en Sanderson noemen een dynamische samenvatting query biased, en Andrew Turpin, Yohannes Tsegay, David Hawking en Hugh Williams doen dat in 2007 ook: een query-biased snippet is een stuk tekst dat gekozen wordt op grond van het verband met de zoekvraag.

    **Zoekwoorden en de geschreven vorm.** De zoekwoorden van een vraag zijn de woorden van de vraag in kleine letters, zonder de stop words van [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl). Een alinea wordt geknipt bij de spaties, met `re.findall(r"\S+", text)`, dus elk woord houdt de hoofdletters en de leestekens, en een snippet kan het woord tonen zoals het geschreven staat. Om een woord met de zoekwoorden te vergelijken, geeft de functie `plain` van de course de kale vorm van het woord: in kleine letters, zonder de leestekens aan het begin en het eind. *Parlement* wordt *parlement*, en *(NFC),* wordt *nfc*.

    **Venster.** Het boek: als de zoekvraag als woordgroep in het document voorkomt, toont de samenvatting de plekken van deze woordgroep, en anders vensters in het document met meerdere zoekwoorden. Zulke vensters beslaan vaak gewoon een aantal woorden links en rechts van de zoekwoorden. Deze les neemt de eenvoudigste vorm: de les telt de zoekwoorden in elke reeks van 20 woorden op een rij, en neemt de reeks met de meeste, bij een gelijke stand de eerste.

    **Beste zin.** Een venster kan midden in een zin beginnen en stoppen, en volgens het boek hebben gebruikers liever snippets die goed lezen, met volledige zinsdelen. Turpin en collega's schrijven dat een systeem voor een snippet alle zinnen van een document tegen de zoekvraag hoort af te zetten, en de beste twee of drie hoort te tonen. Tombros en Sanderson gaven elke zin een query score: het kwadraat van het aantal verschillende zoekwoorden in de zin, gedeeld door het aantal zoekwoorden. De vraag *Welk document vormde het parlement van Victoria?* heeft vijf zoekwoorden: *document*, *parlement*, *victoria*, *vormde* en *welk*. Het woord *welk* staat niet bij de stop words van Snowball, dus het telt als zoekwoord. Een zin met *parlement* en *Victoria* krijgt 2² / 5 = 0,8, en een zin met alleen *Victoria* krijgt 1² / 5 = 0,2: elk zoekwoord telt één keer, hoe vaak het ook in de zin staat, en twee verschillende zoekwoorden tellen vier keer zo zwaar als één. Deze les neemt de zin met de hoogste query score, bij een gelijke stand de eerste, en knipt de alinea in zinnen met de regel van [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).

    **Vet.** Een pagina met resultaten zet de zoekwoorden van een snippet in het vet, zodat het oog de zoekwoorden meteen vindt. Deze les schrijft vet zoals Markdown dat doet, `**Victoria**`, en een beletselteken, …, waar de snippet de tekst afkapt.

    **De kosten.** Een statische samenvatting wordt één keer gemaakt, als de documenten in de index komen. Een dynamische samenvatting kan volgens het boek niet van tevoren berekend worden, dus de zoekmachine maakt de samenvatting pas als de zoekvraag binnenkomt, uit de tekst van het document, en dat moet snel gaan, omdat een zoekmachine bij elke zoekvraag veel snippets maakt. Turpin en collega's maakten de tien snippets van een zoekopdracht in 77 milliseconden in plaats van 183, op een collectie van 18,5 miljoen webpagina's, met een gecomprimeerde opslag van de documenten.

    **Toont de snippet het antwoord?** Bij elke vraag van XQuAD weet de course waar het antwoord in de eigen alinea staat. De course kan dus meten hoe vaak een snippet van de eigen alinea het antwoord toont: een lezer die het antwoord in de snippet ziet, hoeft het resultaat niet eens te openen. Het plaatje toont de vraag over Victoria met de eerste drie resultaten van BM25, elk met het venster van 20 woorden. De eigen alinea, alinea 513, staat bovenaan. De eerste 20 woorden stoppen zes woorden voor het antwoord, *Victoria Constitution Act 1855*, en het venster toont het antwoord, net als de beste zin. Over alle 1.183 vragen tonen de eerste 20 woorden het antwoord bij 25,5% van de vragen, het venster van 20 woorden bij 47,9%, en het venster van 30 woorden bij 59,2%. De beste zin toont het antwoord bij 75,2%, met gemiddeld 30,7 woorden, ongeveer evenveel als het venster van 30 woorden; de beste zin van alinea 513 heeft 51 woorden.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **snippet** | de korte tekst onder elk resultaat op een pagina met resultaten, naast de titel, zodat een lezer het resultaat kan beoordelen zonder het te openen |
    | **statische samenvatting** | een samenvatting die bij elke zoekvraag hetzelfde is, zoals de eerste zinnen van een document |
    | **dynamische samenvatting**, query-biased summary | een samenvatting voor de zoekvraag, uit de plekken in het document waar de zoekwoorden staan |
    | **venster** | een reeks woorden op een rij uit een document; een snippet toont het venster met de meeste zoekwoorden |
    | **zoekwoorden** | de woorden van een zoekvraag in kleine letters, zonder stop words |
    | `re.findall(r"\S+", text)` | de tekst geknipt bij de spaties: elk woord houdt de hoofdletters en de leestekens |
    | `" ".join(words)` | de woorden weer aan elkaar tot één tekst, met een spatie tussen elke twee |
    | `**word**` | Markdown voor een woord in het vet |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een snippet van een document te maken bij een zoekvraag:

    1. Knip het document bij de spaties, zodat elk woord blijft zoals het geschreven staat.
    2. Neem de zoekwoorden: de woorden van de zoekvraag in kleine letters, zonder stop words.
    3. Tel bij elk venster van `size` woorden op een rij de woorden waarvan de kale vorm een zoekwoord is.
    4. Neem het eerste venster met de meeste zoekwoorden.
    5. Schrijf de woorden van het venster, met elk zoekwoord in het vet, en een beletselteken waar het document voor of na het venster verdergaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.EXAMPLE` is de vraag over Victoria, `C.question_place(C.EXAMPLE, L.lang)` geeft het nummer van deze vraag, `C.query_of(L.lang, k)` de zoekwoorden van vraag `k`, en `C.results(L.lang, k)` de eerste drie alinea's die BM25 bij deze vraag geeft. `C.words_of(L.lang, i)` geeft de woorden van alinea `i`, geknipt bij de spaties, en `C.own_paragraph(L.lang, k)` de alinea met het antwoord. `C.hits`, `C.best_start` en `C.snippet` zijn de versies van de course van de drie oefeningen. `C.span_of` geeft waar een snippet van een soort begint en eindigt: de eerste woorden (`"first"`), het venster (`"window"`) of de beste zin (`"sentence"`), en `C.shows_answer` of deze woorden het antwoord bevatten. Eerst de resultaten: elke alinea, het aantal woorden, het aantal zoekwoorden, en waar het venster begint. Dan de drie soorten snippet van de eigen alinea, en de beste zin uitgeschreven. Tot slot geeft `C.shares` hoe vaak elke soort het antwoord toont over alle vragen, met vensters van 10 tot 40 woorden:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _query = C.query_of(L.lang, _k)
    print(sorted(_query))
    for _i in C.results(L.lang, _k):
        _words = C.words_of(L.lang, _i)
        print(_i, len(_words), C.hits(_words, _query), C.best_start(_words, _query, 20))
    _own = C.own_paragraph(L.lang, _k)
    for _kind in ("first", "window", "sentence"):
        _a, _b = C.span_of(L.lang, _k, _own, _kind)
        print(_kind, _a, _b, C.shows_answer(L.lang, _k, _a, _b))
    print(C.snippet(C.words_of(L.lang, _own), _query, _a, _b - _a))
    for _size in (10, 20, 30, 40):
        _shares = C.shares(L.lang, _size)
        print(_size, round(_shares["first"], 3), round(_shares["window"], 3), round(_shares["sentence"], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste regel bevat de vijf zoekwoorden. Dan de drie resultaten: alinea 513 heeft 97 woorden en zes zoekwoorden, en het venster begint bij woord 13, geteld vanaf 0; alinea 526 heeft 87 woorden en vijf zoekwoorden, en alinea 520 heeft 101 woorden en ook vijf zoekwoorden. Geen van de drie alinea's bevat *welk*, dus dit zoekwoord helpt nergens.

    Bij de eigen alinea tonen de eerste 20 woorden het antwoord niet, en het venster, van woord 13 tot woord 33, wel. De beste zin is de eerste zin van de alinea, de woorden 0 tot 51: de alinea heeft twee zinnen, en beide bevatten *parlement* en *Victoria*; bij een gelijke stand wint de eerste zin. De laatste vier regels geven bij 10 tot 40 woorden hoe vaak de eerste woorden, het venster en de beste zin het antwoord tonen. Het venster toont het antwoord vaker dan de eerste woorden van dezelfde lengte: bij 10 woorden bijna twee keer zo vaak, bij 40 woorden ongeveer anderhalf keer zo vaak. En de beste zin hangt niet af van een lengte, dus het aandeel blijft 0,752.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. hits: de zoekwoorden tellen

        Schrijf `hits`, die `words` krijgt, de woorden van een tekst geknipt bij de spaties, als lijst, en `query`, de zoekwoorden als set, in kleine letters. De functie geeft terug hoeveel van de woorden zoekwoorden zijn; een zoekwoord dat twee keer voorkomt, telt twee keer. In de startcode begint `count` bij 0, en `for word in words` loopt de woorden langs. Een woord zoals het geschreven staat houdt de hoofdletters en de leestekens, dus *Broncos,* met de komma is nog geen zoekwoord. Het vak kent `plain` van de course. Deze functie geeft een woord in kleine letters, zonder de leestekens aan het begin en het eind: `plain("Broncos,")` geeft `"broncos"`. Vervang de `...` door de test: is de kale vorm van het woord een van de zoekwoorden?

        De check roept `hits(["Denver", "Broncos,", "beat", "the", "Panthers."], {"denver", "broncos", "panthers"})` aan, wat `3` geeft. En `hits(["Rijn,", "Maas", "en", "Rijn"], {"rijn"})`, wat `2` geeft: *Rijn* telt elke keer dat het woord voorkomt. Daarna eigen gevallen: een lege lijst, een zoekvraag zonder woorden, een woord tussen haakjes, en de alinea's van de course.
        """),
        L.task("Tel elk woord waarvan de kale vorm een van de zoekwoorden is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=6)
    box_hits
    return (box_hits,)


@app.cell(hide_code=True)
def check_hits(box_hits):
    result_hits = L.check("hits", box_hits.value)
    result_hits
    return (result_hits,)


@app.cell(hide_code=True)
def hints_hits():
    L.hints(
        "hits",
        [
            "De test van de `if` vraagt of het woord, in de vorm die `plain` geeft, een van de woorden in `query` is.",
            "`x in s` vraagt of `x` in de set `s` zit: `\"rijn\" in {\"rijn\", \"maas\"}` geeft `True`. En `plain(\"(NFC)\")` geeft `\"nfc\"`.",
            "`        if ___ in query:`",
        ],
        why="`if plain(word) in query:`: de kale vorm staat in kleine letters en zonder de leestekens aan het begin en het eind, net als de zoekwoorden, dus *Broncos,* en *(NFC)* tellen ook mee.",
    )
    return


@app.cell(hide_code=True)
def text_after_hits():
    mo.md(r"""
    In het plaatje telt je functie de zoekwoorden van de eerste drie resultaten bij de vraag over Victoria. Alinea 513 bevat zes zoekwoorden, en de alinea's 526 en 520 elk vijf; in alle drie staan er hoogstens 3 in één venster van 20 woorden. De staven tonen je hits in elk venster van 20 woorden van alinea 513, van het eerste venster tot het laatste. Bij het venster dat bij woord 13 begint, staan er 3 in, en aan het eind van de alinea staat geen enkel zoekwoord meer. De volgende oefening vindt de eerste hoogste staaf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_start():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. best_start: het venster vinden

        Schrijf `best_start`, die `words` en `query` krijgt zoals in oefening 1, en `size`, het aantal woorden van een snippet. De functie geeft de plaats terug, geteld vanaf 0, waar het venster van `size` woorden met de meeste zoekwoorden begint; bij een gelijke stand het eerste zulke venster. In de startcode beginnen `best` en `most` bij 0. `for start in range(len(words) - size + 1)` loopt elke plaats langs waar een venster van `size` woorden past; als er minder woorden zijn dan `size`, loopt de `for` geen enkele plaats langs, en geeft de functie 0 terug. `count = hits(..., query)` telt de zoekwoorden van een venster, en de `if` houdt het venster als het meer zoekwoorden bevat dan het beste venster tot nu toe. Vervang de `...` door het venster: de `size` woorden van `words` vanaf plaats `start`. Het vak kent `hits` van oefening 1 al.

        De check roept `best_start(["The", "Panthers", "lost", "to", "the", "Denver", "Broncos."], {"denver", "broncos"}, 2)` aan, wat `5` geeft: het venster *Denver Broncos.* bevat beide zoekwoorden. En `best_start(["Rotterdam", "ligt", "aan", "de", "Maas", "en", "de", "Rijn."], {"maas", "rijn"}, 4)`, wat `4` geeft: alleen het laatste venster van vier woorden bevat zowel *Maas* als *Rijn*. Daarna eigen gevallen: minder woorden dan `size`, een alinea zonder zoekwoorden, twee vensters met evenveel zoekwoorden, en de alinea's van de course.
        """),
        L.task("Geef `hits` de `size` woorden van het venster dat bij `start` begint."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_start():
    box_best_start = L.code_box("best_start", lines=7)
    box_best_start
    return (box_best_start,)


@app.cell(hide_code=True)
def check_best_start(box_best_start):
    result_best_start = L.check("best_start", box_best_start.value)
    result_best_start
    return (result_best_start,)


@app.cell(hide_code=True)
def hints_best_start():
    L.hints(
        "best_start",
        [
            "`count` is wat `hits` geeft bij één venster en `query`: het venster is een slice van `words`, vanaf plaats `start`, met `size` woorden.",
            "Een slice `a[i:j]` bevat de elementen vanaf plaats `i` tot plaats `j`, zonder `j` zelf: `a[2:5]` bevat de drie elementen op de plaatsen 2, 3 en 4.",
            "`        count = hits(words[___], query)`",
        ],
        why="`count = hits(words[start:start + size], query)`: de slice stopt net bij plaats `start + size`, zonder deze plaats, en bevat dus precies `size` woorden.",
    )
    return


@app.cell(hide_code=True)
def text_after_best_start():
    mo.md(r"""
    In het plaatje zet je functie een venster op elk van de eerste drie resultaten bij twee vragen. Bij Victoria bevat het venster van alinea 513, het eerste resultaat, het antwoord zelf, *Victoria Constitution Act 1855*: een lezer hoeft het resultaat niet meer te openen. Bij het Amazonebekken bevat het venster van het eerste resultaat *7.000.000 vierkante kilometer*. De vensters van de andere resultaten tonen ook zoekwoorden, maar een lezer ziet dat het over iets anders gaat: alinea 102 gaat over Warschau, met *vierkante kilometer* als enige zoekwoorden, en alinea 512 over landbouwgrond in Victoria. Over alle 1.183 vragen toont je venster van 20 woorden het antwoord bij 47,9% van de vragen, bijna twee keer zo vaak als de eerste 20 woorden, met 25,5%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_snippet():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. snippet: de snippet schrijven

        Schrijf `snippet`, die `words` en `query` krijgt zoals in oefening 1, en `start` en `size` zoals in oefening 2, en de snippet teruggeeft als tekst: de `size` woorden vanaf plaats `start`, met elk zoekwoord in het vet. Vet schrijf je zoals Markdown: twee sterretjes aan elke kant van het woord, `**Denver**`. In de startcode verzamelt `shown` de woorden, `for word in words[start:start + size]` loopt de woorden van het venster langs, de `if` vindt de zoekwoorden met `plain`, en `shown.append(word)` voegt elk woord toe, vet of niet. Dan plakt `text = " ".join(shown)` de woorden aan elkaar, met een spatie tussen elke twee. Een beletselteken, `…`, komt aan het begin van de tekst als het venster na het eerste woord begint, en aan het eind als `start + size < len(words)`: de alinea gaat verder. Vervang de `...` door het zoekwoord in het vet: het woord zoals het geschreven staat, met twee sterretjes aan elke kant.

        De check roept `snippet(["The", "Denver", "Broncos", "beat", "the", "Carolina", "Panthers."], {"denver", "panthers"}, 1, 3)` aan, wat `"… **Denver** Broncos beat …"` geeft: het venster begint na *The* en *the Carolina Panthers.* valt buiten het venster. En `snippet(["De", "Rijn", "en", "de", "Maas"], {"rijn", "maas"}, 0, 5)`, wat `"De **Rijn** en de **Maas**"` geeft: het venster bevat elk woord, dus er komt geen beletselteken. Daarna eigen gevallen: een venster aan het begin of het eind van de woorden, een venster langer dan de woorden, een woord tussen haakjes, en de alinea's van de course.
        """),
        L.task("Zet twee sterretjes aan elke kant van elk zoekwoord, zoals het geschreven staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_snippet():
    box_snippet = L.code_box("snippet", lines=12)
    box_snippet
    return (box_snippet,)


@app.cell(hide_code=True)
def check_snippet(box_snippet):
    result_snippet = L.check("snippet", box_snippet.value)
    result_snippet
    return (result_snippet,)


@app.cell(hide_code=True)
def hints_snippet():
    L.hints(
        "snippet",
        [
            "`word` wordt het woord zelf, zoals het geschreven staat, met twee sterretjes aan elke kant.",
            "Teksten plak je aan elkaar met `+`, in de volgorde waarin je de teksten schrijft: `\"(\" + \"Rijn\" + \")\"` geeft `\"(Rijn)\"`.",
            "`            word = \"**\" + ___`",
        ],
        why="`word = \"**\" + word + \"**\"`: het woord houdt de hoofdletters en de leestekens, dus *Panthers.* wordt `**Panthers.**`, met de punt binnen het vet.",
    )
    return


@app.cell(hide_code=True)
def text_after_snippet():
    mo.md(r"""
    In het plaatje staan je snippets onder de eerste drie resultaten bij twee vragen, zoals een pagina met resultaten de snippets toont: de zoekwoorden in het vet, en een beletselteken waar de alinea verdergaat. Bij de vraag over de belangrijkste rol van priemgetallen toont het eerste resultaat *De fundamentele stelling van de rekenkunde*, het antwoord, met de zoekwoorden *stelling*, *rol*, *priemgetallen* en *getaltheorie* in het vet. De vraag zoekt de *belangrijkste* rol, en de alinea noemt de *centrale* rol: een snippet vindt de woorden van de zoekvraag, niet woorden met dezelfde betekenis.

    ### Probeer het: je eigen pagina met resultaten

    Kies een vraag, een soort snippet en een lengte. Het plaatje toont de eerste vijf resultaten van BM25, elk met de snippet, zegt of de snippet van de eigen alinea het antwoord toont, en geeft in een tabel hoe vaak elke soort het antwoord toont over alle vragen, bij de lengte die je koos. De lengte geldt voor de eerste woorden en het venster; de beste zin houdt de eigen lengte, welke lengte je ook kiest.

    Begin met de vraag over het parlement van Victoria en het venster, en schuif de lengte van 5 naar 40 woorden: vanaf welke lengte toont de snippet van de eigen alinea het antwoord, en blijft de snippet het antwoord tonen? Kies dan de vraag over het Amazonebekken, en vergelijk het venster met de beste zin: toont het venster het antwoord, en de beste zin?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _questions = C.try_options(L.lang)
    try_question = mo.ui.dropdown(options=_questions, value=next(t for t, q in _questions.items() if q == C.EXAMPLE),
                                  label="Vraag")
    try_kind = mo.ui.radio(options=C.KIND_OPTIONS[L.lang],
                           value=next(t for t, k in C.KIND_OPTIONS[L.lang].items() if k == "window"), label="Snippet")
    try_size = mo.ui.slider(C.TRY_SIZES[0], C.TRY_SIZES[1], step=C.TRY_SIZES[2], value=C.SIZE, show_value=True,
                            label="Woorden")
    mo.vstack([try_question, try_kind, try_size])
    return try_kind, try_question, try_size


@app.cell(hide_code=True)
def try_look(try_kind, try_question, try_size):
    mo.Html(C.try_picture(L.lang, try_question.value, try_size.value, try_kind.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Bij Victoria toont het venster het antwoord vanaf 10 woorden, en bij elke grotere lengte ook; vanaf 30 woorden tonen zelfs de eerste woorden het antwoord. Bij het Amazonebekken tonen het venster en de beste zin allebei het antwoord. De zin met het antwoord bevat *vierkante* en *kilometer*, en de eerste zin van de alinea alleen *Amazonebekken*. In het Engels gaat het mis: daar bevatten de eerste zin en de zin met het antwoord elk twee verschillende zoekwoorden, en wint bij een gelijke stand de eerste zin. De Engelse alinea schrijft *kilometres*, waar de vraag *kilometers* schrijft, dus dat zoekwoord telt niet mee: één spelling kan bepalen welke zin een lezer te zien krijgt.

    Is een langere snippet beter? Edward Cutrell en Zhiwei Guan volgden de ogen van mensen die zochten met snippets van één regel, van twee tot drie regels, en van zes tot zeven regels. Meer tekst in de snippet hielp de deelnemers duidelijk bij het zoeken naar informatie, en werkte tegen bij het zoeken naar een bepaalde website. Een lange snippet hielp om een feit te vinden, en zat in de weg bij het vinden van een homepage, waar het adres van de pagina meer zegt. David Maxwell, Leif Azzopardi en Yashar Moshfeghi lieten in 2017 53 mensen zoeken met snippets van vier lengtes. De deelnemers vonden langere samenvattingen over het algemeen beter, omdat deze informatiever leken; toch herkenden de deelnemers de relevante documenten bij alle vier de lengtes ongeveer even goed.

    ### Oefening 4. Statisch of dynamisch?

    Deel elke samenvatting hieronder in: is de samenvatting bij elke zoekvraag hetzelfde, of gemaakt voor de zoekvraag?
    """)
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De resultaten hier komen uit [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), de woorden zijn geknipt zoals in [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl), en de zoekwoorden laten de stop words van [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl) weg. Een snippet is een samenvatting met woorden uit het document zelf; [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl) kiest de zinnen van een tekst voor een samenvatting zonder zoekvraag. De volgende les, *Van tien links naar één antwoord*, kijkt naar zoekmachines die één antwoord schrijven boven de resultaten, of in plaats van de resultaten.

    **Een voorbeeld uit het dagelijks leven.** Een etalage toont een paar dingen uit de winkel, gekozen voor de mensen die langslopen. Een statische samenvatting is een etalage die nooit verandert; een dynamische samenvatting richt de etalage opnieuw in voor elke voorbijganger, met wat deze voorbijganger zocht.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_hits, result_best_start, result_snippet, result_kinds):
    L.wrap_up(result_hits, result_best_start, result_snippet, result_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
