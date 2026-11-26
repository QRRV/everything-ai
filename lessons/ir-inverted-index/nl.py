# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-inverted-index. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-inverted-index", "nl")
    C = L.checks   # de dertien documenten van de course, de alinea's van SQuAD-NL, de inverted index en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reads():
    quick_reads = L.choice(
        "quick-reads",
        question="In *Zoeken: informatiebehoefte en relevantie* gaf `C.search` de documenten waarvoor `matches` `True` teruggaf. Voor welke documenten roept `C.search` de functie `matches` aan, bij één zoekvraag in de dertien documenten van de course?",
        options={
            "first": "alleen bij de documenten met het eerste woord van de zoekvraag",
            "right": "bij alle dertien, voor elke zoekvraag",
            "results": "alleen bij de resultaten",
            "remember": "de eerste keer bij alle dertien; daarna alleen bij de documenten die de functie eerder vond",
        },
        correct="right",
        why={
            "right": "`C.search` gaat elk document langs, en `matches` kijkt naar de woorden van elk document. Het werk groeit mee met de verzameling: dertien documenten, of de honderden miljarden webpagina's van een zoekmachine. Deze les vindt de documenten met alle woorden van een zoekvraag zonder de documenten te lezen.",
            "first": "Om te weten welke documenten het eerste woord bevatten, moet de zoekmachine elk document bekijken. Welke documenten gaat `C.search` langs? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
            "results": "De resultaten zijn wat de zoekmachine vindt. Hoe komt de zoekmachine erachter welke documenten resultaten zijn? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
            "remember": "`C.search` onthoudt niets van de ene zoekvraag tot de volgende. Welke documenten gaat de functie elke keer langs? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
        },
    )
    quick_reads
    return (quick_reads,)


@app.cell(hide_code=True)
def quick_check_extra():
    quick_extra = L.choice(
        "quick-extra",
        question="Een document bevat alle woorden van de zoekvraag, maar helpt niet bij de informatiebehoefte. In welke lijst van `split` uit *Zoeken: informatiebehoefte en relevantie* komt het terecht?",
        options={
            "found": "gevonden: het is een resultaat",
            "missed": "gemist: de zoekmachine gaf het niet terug",
            "right": "te veel: een resultaat dat niet helpt",
            "relevant": "gevonden, want een document met elk woord van de zoekvraag is relevant",
        },
        correct="right",
        why={
            "right": "De zoekmachine geeft het terug, want alle woorden staan erin, en het helpt niet: een resultaat te veel. De index van deze les vindt dezelfde resultaten als `matches`, ook de resultaten te veel, maar dan veel sneller.",
            "found": "Bij gevonden horen de resultaten die helpen bij de informatiebehoefte. Helpt dit document? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
            "missed": "De zoekmachine geeft elk document terug met alle woorden van de zoekvraag. Geeft de zoekmachine dit document terug? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
            "relevant": "Relevantie beoordeel je aan de hand van de informatiebehoefte. Helpt een document met alle woorden altijd? Zie [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).",
        },
    )
    quick_extra
    return (quick_extra,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De zoekmachine van [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl) bekijkt bij elke zoekvraag elk document. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze noemen dat in *Introduction to Information Retrieval* (2008) de eenvoudigste manier om documenten te vinden: de computer leest de documenten een voor een van begin tot eind. Dat heet *grepping*, naar het Unix-programma grep, dat precies zo zoekt. Voor het verzameld werk van Shakespeare, iets minder dan een miljoen woorden, is meer ook niet nodig, schrijven de auteurs. Voor een grote verzameling niet.

    Over Google Search schrijft Google zelf dat de zoekindex honderden miljarden webpagina's bevat en ruim 100.000.000 gigabyte groot is. Google vergelijkt de index met het register achter in een boek, met een ingang voor elk woord op elke webpagina in de index. Een zoekmachine leest het web niet op het moment dat je zoekt: de zoekmachine zoekt je woorden op in een index die al klaarligt.

    In deze les bouw je zo'n index voor de dertien documenten van de course. Daarna doorzoek je 2.067 alinea's van de Engelse Wikipedia, in het Nederlands vertaald in SQuAD-NL, de Nederlandse versie van SQuAD, een verzameling vragen over Wikipedia. Dat doe je met een index die de course op dezelfde manier bouwde, zonder een alinea te lezen.

    **Een voorbeeld uit het dagelijks leven.** Om te vinden wat een kookboek over *saffraan* zegt, lees je niet het hele boek: je zoekt *saffraan* op in het register achterin, en bladert naar de pagina's die erbij staan.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_zeros = L.guess(
        "guess",
        question="In *Een document als vector* had een document-term matrix een rij voor elk document en een kolom voor elk woord, en bestond de matrix vooral uit nullen. De 2.067 alinea's van deze les bevatten 30.090 verschillende woorden. Maak een tabel met een rij per alinea en een kolom per woord: een 1 waar de alinea het woord bevat, en een 0 waar niet. Welk deel van de cellen is 0?",
        options={
            "half": "ongeveer de helft: een alinea bevat veel woorden",
            "nine": "ongeveer negen op de tien",
            "right": "meer dan 99 op de 100",
            "three": "ongeveer drie op de vier",
        },
        reveal="Meer dan 99 op de 100: 99,73% van de 62.196.030 cellen is 0. Slechts 170.719 cellen zijn een 1: een alinea bevat gemiddeld 83 verschillende woorden. Meer dan de helft van de woorden, 17.168, staat in precies één alinea. Manning en collega's rekenen hetzelfde uit voor een miljoen documenten van elk ongeveer 1.000 woorden, met 500.000 verschillende woorden: minstens 99,8% van de cellen is daar 0. Een tabel die elke 0 opslaat, is bijna helemaal verspilling.",
    )
    guess_zeros
    return (guess_zeros,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Een tabel met nullen en enen.** Manning en collega's beginnen met de toneelstukken van Shakespeare en de zoekvraag *Brutus AND Caesar AND NOT Calpurnia*: de stukken met de woorden *Brutus* en *Caesar*, en zonder het woord *Calpurnia*. Geef elk woord een rij en elk stuk een kolom, met een 1 waar het stuk het woord bevat en een 0 waar niet. Zo'n tabel is een **incidentiematrix**: de document-term matrix van *Een document als vector*, gekanteld, met een 1 voor elke telling die niet 0 is. De zoekvraag neemt de rijen van *Brutus* en *Caesar*, maakt in de rij van *Calpurnia* van elke 0 een 1 en van elke 1 een 0, en bewaart de kolommen waar alle drie de rijen een 1 hebben. Het antwoord is *Antony and Cleopatra* en *Hamlet*. Voor een miljoen documenten van ongeveer 1.000 woorden, met 500.000 verschillende woorden, telt de tabel een half biljoen nullen en enen, volgens het boek te veel voor het geheugen van een computer.

    **Bewaar alleen de enen.** Een **inverted index** bewaart voor elk woord alleen de lijst van de documenten met dat woord, op volgorde van nummer: de **postings list** van het woord. Elk nummer erin is een **posting**: een aantekening dat het woord in dat document staat. Het register achter in een boek werkt net zo, van een woord naar de pagina's, en Manning en collega's merken op dat de naam eigenlijk dubbelop is: een index wijst altijd terug van een woord naar de plaatsen waar het staat, dus *inverted* voegt niets toe. Voor de index ga je één keer door de documenten: voeg voor elk document, op volgorde, het nummer toe aan de lijst van elk woord in het document, één keer. Omdat de documenten op volgorde komen, is elke lijst vanzelf gesorteerd.

    **AND: wandel langs twee lijsten.** Neem voor *wijn AND hart* de twee postings lists, en zet een vinger aan het begin van elke lijst. Wijzen beide vingers hetzelfde getal aan, dan bevat dat document beide woorden: bewaar het nummer, en schuif beide vingers door. Verschillen de getallen, schuif dan de vinger bij het kleinste getal door: de vinger in de andere lijst staat al bij een groter getal, dus dit getal komt daar niet meer voor. Is een van de lijsten op, dan stopt de wandeling. Elke stap vergelijkt twee getallen. De wandeling doet hooguit zoveel stappen als er getallen in de twee lijsten samen staan, en opent nooit een document. Het resultaat is de **doorsnede** van de twee lijsten: de getallen die in beide staan.

    **Meer woorden: de kortste lijst eerst.** Wandel bij drie woorden eerst langs twee lijsten, en daarna met het antwoord tot nu toe langs de derde lijst. De volgorde verandert het antwoord niet, wel het werk. Het boek noemt als gangbare vuistregel: werk de woorden af van de kleinste document frequency naar de grootste. De **document frequency** van een woord is het aantal documenten met dat woord: de lengte van de postings list. Begin je bij de kortste lijst, dan wordt het antwoord tot nu toe nooit langer dan deze lijst, en een wandeling stopt zodra het antwoord tot nu toe op is; meestal is er dan minder te wandelen.

    **OR, NOT en Boolean retrieval.** OR wandelt net zo langs twee lijsten en bewaart elk getal dat in minstens een van beide staat. NOT neemt de documenten die niet op een lijst staan. Zoeken met AND, OR en NOT heet **Boolean retrieval**, naar de algebra van de wiskundige George Boole (1847): een document voldoet of voldoet niet, en de resultaten krijgen geen volgorde van goed naar minder goed. Het bleef lang in gebruik. Het boek beschrijft de juridische zoekdienst Westlaw, gestart in 1975, met ruim een half miljoen abonnees die samen miljoenen zoekvragen per dag stellen; in 2005 was Boolean zoeken daar nog steeds de standaard, hoewel er in 1992 een zoekfunctie met gerangschikte resultaten bij was gekomen.

    Het plaatje bouwt de index van de dertien documenten van de course voor de zoekvraag *rode wijn hart*, en wandelt langs de twee kortste lijsten.
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
    | **incidentiematrix** | een tabel met een rij voor elk woord en een kolom voor elk document: een 1 waar het document het woord bevat, een 0 waar niet |
    | **inverted index** | voor elk woord de lijst van de documenten met dat woord; een zoekmachine zoekt woorden daarin op, in plaats van de documenten te lezen |
    | **postings list** | de lijst van een woord in een inverted index: de nummers van de documenten met het woord, gesorteerd; elk nummer is een **posting** |
    | **document frequency** | het aantal documenten met een woord: de lengte van de postings list |
    | **doorsnede** | de getallen die in beide lijsten staan; bij postings lists gevonden met één wandeling langs beide |
    | **Boolean retrieval** | zoeken met AND, OR en NOT; een document voldoet of voldoet niet, en de resultaten krijgen geen volgorde |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo beantwoord je een zoekvraag met AND uit een inverted index:

    1. Bouw de index één keer: voeg voor elk document, op volgorde, het nummer één keer toe aan de postings list van elk woord in het document.
    2. Zoek elk woord van de zoekvraag op. Een woord zonder lijst staat in geen enkel document, en het antwoord is leeg.
    3. Sorteer de lijsten op lengte, de kortste eerst.
    4. Wandel langs de eerste twee lijsten met een vinger in elke lijst: bewaar een getal dat in beide staat, en schuif de vinger bij het kleinste getal door.
    5. Wandel met het antwoord tot nu toe langs elke volgende lijst, een voor een.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.DOCS` bevat de dertien documenten van de course als lijsten woorden, en `C.QUERIES` de zoekvragen van *Zoeken: informatiebehoefte en relevantie*. `C.build_index` bouwt de inverted index van een lijst documenten, `C.intersect` wandelt langs twee postings lists, en `C.search_and` beantwoordt een zoekvraag met AND, de kortste lijst eerst:
    """)
    return


@app.cell
def example_cell():
    _index = C.build_index(C.DOCS[L.lang])
    print(len(_index))
    _query = C.QUERIES[L.lang]["wine"]
    for _word in _query:
        print(_word, _index[_word])
    print(C.intersect(_index[_query[1]], _index[_query[2]]))
    print(C.search_and(_query, _index))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De dertien documenten bevatten 115 verschillende woorden, dus de index heeft 115 postings lists. De woorden *rode* en *wijn* staan in vier documenten, *hart* in drie. De wandeling langs de lijsten van *wijn* en *hart* levert 6, 7 en 8 op. `C.search_and` begint met *hart*, de kortste lijst, en *rode*, en wandelt daarna met het antwoord langs de lijst van *wijn*, waarin alle drie staan. Het antwoord is wat `matches` vond in *Zoeken: informatiebehoefte en relevantie*, de documenten 6, 7 en 8, en de index vond het zonder een document te openen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_build_index():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. build_index: de inverted index

        Schrijf `build_index`, die `docs` krijgt, een lijst documenten, elk een lijst woorden, zoals `C.DOCS["nl"]`. De functie geeft de inverted index terug: een dictionary van elk woord naar de postings list, de nummers van de documenten met dat woord, op volgorde en elk nummer één keer. In de startcode gaat de buitenste loop elk nummer `doc` in `range(len(docs))` langs, en de binnenste loop elk `word` van het document `docs[doc]`. Een woord dat voor het eerst langskomt, krijgt een lege lijst in `index`, en `.append` voegt het nummer van het document toe aan de lijst van het woord. Vervang de `...` door de test die elk nummer niet vaker dan één keer in een lijst toelaat, ook als een document het woord twee keer bevat.

        De check roept `build_index([["red", "wine"], ["white", "wine", "wine"], ["red"]])` aan, wat `{"red": [0, 2], "wine": [0, 1], "white": [1]}` geeft, en `build_index([["python"], [], ["python", "snake", "python"]])`, wat `{"python": [0, 2], "snake": [2]}` geeft. Daarna eigen gevallen: helemaal geen documenten, een document met een woord twee keer en een ander woord ertussen, een woord drie keer in één document, een woord in elk document, en de dertien documenten van de course in beide talen.
        """),
        L.task("Voeg het nummer van elk document één keer toe aan de postings list van elk woord in het document."),
    ])
    return


@app.cell(hide_code=True)
def exercise_build_index():
    box_build_index = L.code_box("build_index", lines=9)
    box_build_index
    return (box_build_index,)


@app.cell(hide_code=True)
def check_build_index(box_build_index):
    result_build_index = L.check("build_index", box_build_index.value)
    result_build_index
    return (result_build_index,)


@app.cell(hide_code=True)
def hints_build_index():
    L.hints(
        "build_index",
        [
            "Een nummer hoort één keer in de lijst van een woord, ook als het woord twee keer in het document staat. `index[word]` is de lijst van `word` tot nu toe, en de test komt in de `if`: `x in een_lijst` is `True` als `x` een van de elementen van een lijst is.",
            "Laat het nummer van dit document, `doc`, alleen door als het nog niet in de lijst van het woord staat. `not in` is het tegenovergestelde van `in`.",
            "`            if ___ index[word]:`",
        ],
        why="`if doc not in index[word]:`: een nummer komt alleen in de lijst van een woord de eerste keer dat het woord in het document voorkomt.",
    )
    return


@app.cell(hide_code=True)
def text_after_build_index():
    mo.md(r"""
    In het plaatje heeft *rode* een lijst van vier documenten, *wijn* ook, *hart* van drie, en *python* van vijf. In de 2.067 alinea's zijn de lijsten veel ongelijker: *de* staat in 2.046 alinea's, en meer dan de helft van de 30.090 woorden staat in precies één alinea. De meeste postings lists zijn kort en een paar zijn heel lang: hetzelfde patroon als de wet van Zipf, uit de les [De wet van Zipf en de wet van Heaps](/course/open/nlp-zipf-heaps?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_intersect():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. intersect: wandelen langs twee lijsten

        Schrijf `intersect`. De functie krijgt twee postings lists `p1` en `p2`, elk gesorteerd van klein naar groot, en geeft de lijst terug van de getallen die in beide staan, op volgorde. In de startcode is `i` de plaats van de vinger in `p1` en `j` de plaats van de vinger in `p2`; `i` en `j` beginnen allebei bij 0. De `while`-loop gaat door zolang geen van de lijsten op is: zolang `i < len(p1)` en `j < len(p2)`. Zijn `p1[i]` en `p2[j]` hetzelfde getal, dan voegt `.append` het getal toe aan `answer`, en schuiven beide vingers door. Anders schuift één vinger door: `i` in de `elif`, `j` in de `else`. Vervang de `...` door de test die bepaalt wanneer de vinger in `p1` doorschuift.

        De check roept `intersect([6, 7, 8, 10], [6, 7, 8, 9])` aan, de lijsten van *wine* en *heart* in de Engelse les, wat `[6, 7, 8]` geeft, en `intersect([2, 4, 8, 16, 32], [1, 2, 3, 5, 8, 13])`, wat `[2, 8]` geeft. Daarna eigen gevallen: een lege lijst aan een van beide kanten, lijsten zonder gemeenschappelijk getal, twee gelijke lijsten, twee lijsten die een paar getallen delen, maar anders beginnen en eindigen, een korte lijst tegen een langere in beide volgordes, en twee lange lijsten met veelvouden van 3 en van 4.
        """),
        L.task("Geef de getallen die in beide gesorteerde lijsten staan, met één wandeling langs beide."),
    ])
    return


@app.cell(hide_code=True)
def exercise_intersect():
    box_intersect = L.code_box("intersect", lines=14)
    box_intersect
    return (box_intersect,)


@app.cell(hide_code=True)
def check_intersect(box_intersect):
    result_intersect = L.check("intersect", box_intersect.value)
    result_intersect
    return (result_intersect,)


@app.cell(hide_code=True)
def hints_intersect():
    L.hints(
        "intersect",
        [
            "Als `p1[i]` en `p2[j]` verschillen, schuift de vinger bij het kleinste getal door. De `elif` beslist wanneer dat de vinger in `p1` is; deze vinger schuift door met `i = i + 1`. `<` vergelijkt twee getallen.",
            "De vinger in `p1` schuift door als het getal daar het kleinste van de twee is: als het getal op plaats `i` in `p1` kleiner is dan het getal bij de vinger in de andere lijst.",
            "`        elif p1[i] ___:`",
        ],
        why="`elif p1[i] < p2[j]:`: het getal in `p1` is het kleinste, dus de andere lijst kan het niet meer bevatten, en de vinger in `p1` schuift door.",
    )
    return


@app.cell(hide_code=True)
def text_after_intersect():
    mo.md(r"""
    In het plaatje bekijkt de wandeling langs *rijn* en *delta* hooguit 47 getallen en vindt vijf alinea's; de alinea's zelf lezen betekent 264.177 woorden lezen. Een wandeling doet hooguit zoveel stappen als er getallen in de twee lijsten samen staan, dus het werk hangt af van de lengte van de lijsten van de zoekvraag, niet van het aantal woorden in de documenten. Zoekmachines maken lange wandelingen nog korter: het boek beschrijft *skip pointers*, waarmee een wandeling over een stuk van een lange lijst heen springt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_search_and():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. search_and: elk woord van de zoekvraag

        Schrijf `search_and`, die `query` krijgt, een lijst van minstens één woord, en `index`, een inverted index, zoals `C.SMALL_INDEX["nl"]`, de index van de dertien documenten van de course. De functie geeft de nummers terug van de documenten met alle woorden van de zoekvraag. In de startcode gaat de eerste loop elk `word` van `query` langs: een woord dat niet in `index` staat, staat in geen enkel document, dus het antwoord is `[]`; anders voegt `.append` de postings list van het woord toe aan `lists`. `sorted` met `key=len` zet de lijsten op volgorde van lengte, de kortste eerst, en `answer` begint als de kortste. De laatste loop gaat elke volgende lijst, `postings`, in `lists[1:]` langs. Vervang de `...` door het nieuwe antwoord: de wandeling van het antwoord tot nu toe langs `postings`, met `intersect` uit oefening 2; het vak kent deze functie al.

        De check roept `search_and(["red", "wine", "heart"], C.SMALL_INDEX["en"])` aan, met de index van de Engelse les, wat `[6, 7, 8]` geeft, en `search_and(["python", "slang"], C.SMALL_INDEX["nl"])`, wat `[2]` geeft. Daarna eigen gevallen: een zoekvraag van één woord, een zoekvraag van vier woorden, een woord dat in geen enkel document staat, twee keer hetzelfde woord, de Nederlandse zoekvraag *rode wijn hart*, een zoekvraag van zes woorden, en een kleine uitgeschreven index.
        """),
        L.task("Geef de documenten met elk woord van de zoekvraag, wandelend langs de lijsten vanaf de kortste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search_and():
    box_search_and = L.code_box("search_and", lines=12)
    box_search_and
    return (box_search_and,)


@app.cell(hide_code=True)
def check_search_and(box_search_and):
    result_search_and = L.check("search_and", box_search_and.value)
    result_search_and
    return (result_search_and,)


@app.cell(hide_code=True)
def hints_search_and():
    L.hints(
        "search_and",
        [
            "`intersect` wandelt langs twee postings lists en geeft de getallen terug die in beide staan. In deze loop bevat `answer` wat de wandelingen tot nu toe overhielden, en is `postings` de lijst van het volgende woord.",
            "Het nieuwe antwoord is wat het antwoord tot nu toe en de volgende lijst allebei bevatten: roep de functie van oefening 2 aan met deze twee.",
            "`        answer = intersect(___)`",
        ],
        why="`answer = intersect(answer, postings)`: elke ronde bewaart de getallen die het antwoord tot nu toe en de volgende lijst allebei bevatten.",
    )
    return


@app.cell(hide_code=True)
def text_after_search_and():
    mo.md(r"""
    In het plaatje vindt *zuurstof water vuur* niets: *vuur* staat in 7 alinea's en *zuurstof* in 42, twee alinea's bevatten beide, en geen van de twee bevat *water*. Bij *stoom machine water* beginnen de wandelingen bij *machine* en *stoom*, 14 en 26 alinea's; er blijven er twee over, en de wandeling met de 61 alinea's van *water* stopt na 7 stappen: 42 vergelijkingen in totaal. Begin je bij *water* en *machine*, dan kost hetzelfde antwoord er 76. Typ de zoekvragen in het vak hieronder om elke wandeling te zien.

    ### Probeer het: doorzoek de alinea's

    Typ een zoekvraag in het vak. Het plaatje zoekt elk woord op in de index van de 2.067 alinea's, wandelt langs de lijsten vanaf de kortste, en laat zien hoeveel alinea's er na elke wandeling over zijn en hoeveel vergelijkingen elke wandeling deed. Probeer *rijn delta*, en voeg dan *water* toe. De derde wandeling houdt één alinea over en doet toch meer vergelijkingen dan de eerste: de vinger in de lange lijst van *water* schuift getal voor getal door, tot een van de lijsten op is. Zet daarna een woord dat in bijna elke alinea staat, zoals *de*, naast een zeldzaam woord.
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.TRY_START[L.lang], label="Je zoekvraag", rows=2, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_answers():
    mo.md(r"""
    ### Oefening 4. Wat beantwoordt deze index?

    De index van deze les bewaart bij elk woord de nummers van de documenten met dat woord, en verder niets. OR wandelt langs twee lijsten en bewaart elk getal dat in minstens een van beide staat; NOT neemt de documenten die niet op een lijst staan. Deel elke zoekvraag hieronder in: deze index beantwoordt de zoekvraag, of er is meer nodig dan de nummers van de documenten met een woord.
    """)
    return


@app.cell(hide_code=True)
def exercise_answers():
    sort_answers = L.sort("answers")
    sort_answers
    return (sort_answers,)


@app.cell(hide_code=True)
def check_answers(sort_answers):
    result_answers = L.check("answers", sort_answers)
    result_answers
    return (result_answers,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl) vond elk document met alle woorden van een zoekvraag door elk document te lezen; de index van deze les vindt dezelfde documenten met alleen de postings lists. De incidentiematrix is de document-term matrix van [Een document als vector](/course/open/nlp-document-vectors?lang=nl) met een 1 voor elke telling die niet 0 is, en de inverted index slaat de matrix sparse op: alleen de enen. De volgende les, *Rangschikken met de cosine similarity: het vector space model*, zet de resultaten op volgorde, met per woord een telling; een index kan zo'n telling naast elke posting bewaren.

    **Een voorbeeld uit het dagelijks leven.** In de catalogus van een bibliotheek vind je elk boek van een schrijver zonder langs de kasten te lopen: je zoekt de naam op en krijgt een lijst met plaatsnummers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_build_index, result_intersect, result_search_and, result_answers):
    L.wrap_up(result_build_index, result_intersect, result_search_and, result_answers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
