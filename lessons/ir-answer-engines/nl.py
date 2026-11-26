# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-answer-engines. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-answer-engines", "nl")
    C = L.checks   # de vragen van XQuAD, de tien links van BM25, de answer engine van de course en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl) en [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rank():
    quick_rank = L.choice(
        "quick-rank",
        question="Het eerste relevante document bij een zoekvraag staat op plaats 4 van de rangschikking. Wat is de reciprocal rank van deze zoekvraag?",
        options={
            "place": "4: de plaats zelf",
            "first": "0: alleen de eerste plaats telt, en het document op plaats 1 is niet relevant",
            "right": "0,25: één gedeeld door de plaats",
            "share": "0,75: drie van de eerste vier plaatsen bevatten geen relevant document",
        },
        correct="right",
        why={
            "right": "De reciprocal rank is één gedeeld door de plaats van het eerste relevante document: 1 / 4 = 0,25. De maat kijkt alleen naar het eerste relevante resultaat, zoals een lezer die stopt bij de eerste goede link. Deze les vraagt nog iets eenvoudigers aan een pagina met links: staat er bij de eerste resultaten überhaupt een relevant document?",
            "place": "De reciprocal rank wordt kleiner als het eerste relevante document lager staat. Wordt 4 kleiner als de plaats hoger wordt? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
            "first": "De reciprocal rank zoekt het eerste relevante document, waar het document ook staat. Wat doet de maat met de plaats van dit document? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
            "share": "Dat telt de plaatsen zonder relevant document. Door welke plaats deelt de reciprocal rank één? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
        },
    )
    quick_rank
    return (quick_rank,)


@app.cell(hide_code=True)
def quick_check_extract():
    quick_extract = L.choice(
        "quick-extract",
        question="Wat doet een extractieve samenvatting van een tekst?",
        options={
            "write": "nieuwe zinnen schrijven in eigen woorden, zoals een mens zou doen",
            "right": "zinnen uit de tekst kiezen, met de woorden ongewijzigd",
            "keywords": "een lijst geven van de woorden die het vaakst in de tekst voorkomen, zonder zinnen eromheen",
            "lead": "altijd de eerste zin nemen",
        },
        correct="right",
        why={
            "right": "Een extractieve samenvatting neemt zinnen ongewijzigd over uit de tekst; een abstractieve samenvatting schrijft nieuwe zinnen. De answer engine van deze les is extractief: het antwoord is één zin uit een resultaat, dus elk woord van het antwoord staat in de geciteerde bron.",
            "write": "Dat is een abstractieve samenvatting. Wat doet een extractieve samenvatting met de zinnen van de tekst? Zie [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl).",
            "keywords": "Een samenvatting in zinnen of een lijst met woorden: welke van de twee is een extractieve samenvatting? Zie [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl).",
            "lead": "De eerste zinnen zijn een baseline om een samenvatting mee te vergelijken; extractief betekent iets anders. Welke zinnen mag een extractieve samenvatting kiezen? Zie [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl).",
        },
    )
    quick_extract
    return (quick_extract,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In deze module gaf een zoekmachine een pagina met resultaten: een rangschikking door [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), gemeten met [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl), en bij elk resultaat een snippet uit [Snippets: de resultaten laten zien](/course/open/ir-snippets?lang=nl). De lezer kiest een resultaat en leest het. Donald Metzler, Yi Tay, Dara Bahri en Marc Najork schreven in 2021 dat klassieke zoeksystemen een informatiebehoefte niet zelf beantwoorden, maar verwijzen naar antwoorden, hopelijk uit een betrouwbare bron. De vier stelden een systeem voor dat de vraag zelf beantwoordt, in plaats van de gebruiker een keuze uit mogelijke antwoorden te geven, wat de gebruiker volgens de auteurs flink wat denkwerk kost.

    Zulke systemen bestaan nu. Google schrijft een AI-samenvatting boven de resultaten, en zoekmachines zoals Perplexity geven één antwoord op de plek van de resultaten, met kleine nummers die naar de bronnen wijzen. Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Yixin Mao en Chien-Sheng Wu noemen zulke systemen answer engines. Een answer engine is een zoekmachine op basis van een LLM en schrijft een samenvatting met bronnen, in plaats van alleen relevante links te geven; volgens de auteurs is dat een bijzonder grote verschuiving ten opzichte van de klassieke zoekmachine. Deze les meet wat deze verschuiving verandert: voor de lezer die het antwoord krijgt, en voor de bronnen waar het antwoord vandaan komt.

    **Een voorbeeld uit het dagelijks leven.** Je vraagt een voorbijganger de weg en krijgt één antwoord: de tweede straat links. Een plattegrond toont elke straat en laat de keuze aan jou. Het antwoord is sneller. Maar als de voorbijganger zich vergist, loop je de verkeerde kant op, en niets vertelt je dat er een andere straat was.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_clicks = L.guess(
        "guess",
        question="In maart 2025 volgde het Pew Research Center de zoekopdrachten op Google van 900 volwassenen in de Verenigde Staten: 68.879 zoekopdrachten, waarvan 18% met een AI-samenvatting bovenaan de pagina. De samenvattingen noemden de bronnen, als links. Hoe vaak klikten mensen op zo'n link in de samenvatting?",
        options={
            "half": "ongeveer de helft van de keren: de samenvatting noemt de bronnen, dus mensen openen deze bronnen om te controleren",
            "eight": "bij 8% van de bezoeken",
            "right": "bij 1% van de bezoeken",
            "fifteen": "bij 15% van de bezoeken",
        },
        reveal="Bij 1% van de bezoeken aan een pagina met een AI-samenvatting. De 8% en de 15% gaan over de links onder de samenvatting, de klassieke resultaten: bij bezoeken met een AI-samenvatting klikten mensen bij 8% op een klassiek resultaat, bij bezoeken zonder samenvatting bij 15%, bijna twee keer zo vaak. Na een pagina met een samenvatting eindigde de browsersessie ook vaker: bij 26% van de pagina's met een AI-samenvatting, tegen 16% van de pagina's met alleen klassieke resultaten.",
    )
    guess_clicks
    return (guess_clicks,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Answer engine.** Een zoekmachine die een vraag beantwoordt met één tekst en de bronnen van deze tekst, in plaats van met een pagina met links. De meeste answer engines zoeken eerst, zoals BM25, en laten dan een LLM de eerste resultaten lezen en daaruit het antwoord schrijven, met na elke zin een nummer voor het resultaat waar de zin vandaan komt: een citatie.

    **De answer engine van de course.** Deze les bouwt er een zonder LLM. De engine leest de eerste resultaten van BM25 en antwoordt met één zin: de zin met de hoogste query score uit [Snippets: de resultaten laten zien](/course/open/ir-snippets?lang=nl), de eerste bij een gelijke stand, gevolgd door de plaats van het resultaat tussen haken, zoals [2]. Het antwoord is een extractieve samenvatting van de resultaten, zoals in [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl): de engine neemt een zin over en schrijft zelf niets, dus elk woord van het antwoord staat in de geciteerde bron. Nelson Liu, Tianyi Zhang en Percy Liang zagen hetzelfde bij echte answer engines: letterlijk overgenomen tekst wordt bijna altijd volledig gesteund door de geciteerde bron. Bij de engine van de course kan alleen de keuze van de zin misgaan: beantwoordt de zin de vraag?

    **success@k.** Een pagina met links werkt als de lezer er een relevant resultaat op vindt. success@k meet hoe vaak dat kan: het deel van de vragen met minstens één relevant document bij de eerste k resultaten. P@k telt hoeveel van de eerste k relevant zijn, en de reciprocal rank kijkt naar de plaats van het eerste relevante document; success@k vraagt alleen of er een is. Met de eigen alinea als enige relevante alinea is success@1 het deel van de vragen waarbij de eigen alinea eerst komt.

    **Toont het antwoord het antwoord?** Net als in [Snippets: de resultaten laten zien](/course/open/ir-snippets?lang=nl) weet de course waar het antwoord van elke vraag van XQuAD in de eigen alinea staat. Een antwoord van de engine van de course telt als goed als de zin uit de eigen alinea komt en het antwoord bevat.

    **Tien links tegen één antwoord.** Het plaatje toont de vraag naar het aantal inwoners van Jacksonville in 2010, met de tien links. BM25 zet alinea 1398 eerst, en de eigen alinea, 1379, op de tweede plaats. Na alleen het eerste resultaat antwoordt de engine met *Jacksonville is de meest bevolkte stad in Florida en de twaalfde meest bevolkte stad in de Verenigde Staten.* Dat klinkt als een antwoord en bevat geen getal. Na de eerste drie resultaten vindt de engine de zin met *1.345.596* in alinea 1379 en citeert [2]. Over alle 1.183 vragen is het eerste resultaat bij 73,5% van de vragen de eigen alinea, en bevatten de tien links de eigen alinea bij 91,9%. Na één resultaat antwoordt de engine bij 58,4% van de vragen goed, en na drie resultaten bij 59,8%; meer dan drie resultaten lezen maakt het een fractie slechter, 59,5% bij tien. Van de 476 vragen met een fout antwoord na drie resultaten staat bij 380 de eigen alinea bij de tien links: 79,8%. Een lezer van de pagina kan het antwoord nog een paar regels lager vinden; een lezer van het ene antwoord ziet alleen de foute zin.

    **Klikken.** De gok liet zien wat er met de links gebeurt: bij bezoeken met een AI-samenvatting klikten mensen bij 8% op een klassiek resultaat, tegen 15% bij bezoeken zonder samenvatting, en bij 1% op een link in de samenvatting. Lydia Chilton en Jaime Teevan zagen dit al in 2011, bij de korte antwoorden die Bing op de pagina met resultaten toonde, zoals de betekenis van een woord: zulke antwoorden kunnen klikken wegnemen bij de gewone resultaten. Een klik die niet gebeurt kost de site die de tekst schreef een bezoek. Voor de lezer is een zoekopdracht zonder klik niet altijd mislukt. Jane Li, Scott Huffman en Akihito Tokuda gaven het goede geval in 2009 een naam: good abandonment, een zoekopdracht waarbij de pagina met resultaten de informatiebehoefte van de gebruiker al vervulde, zodat klikken of de zoekvraag aanpassen niet meer nodig was. Dezelfde ontbrekende klik kan ook betekenen dat de lezer een fout antwoord geloofde. Narayanan Venkit en collega's keken mee met 21 mensen die zochten: bij answer engines klikken gebruikers meestal op één bron en vertrouwen op de keuze van het model. Met een answer engine bewogen mensen de muis over ongeveer 2 bronnen en openden er ongeveer 1; met Google bewogen deze mensen de muis over ongeveer 12 bronnen en openden er ongeveer 4.

    **Citaties.** Als een antwoord een bron citeert, zegt de bron dan wat de zin zegt? Liu, Zhang en Liang meten dit op twee manieren. Citation recall is het deel van de te controleren uitspraken dat volledig gesteund wordt door de citaties erbij. Citation precision is het deel van de citaties dat de uitspraak erbij steunt. Eén maat is niet genoeg. Stel dat een antwoord bij elke uitspraak elke webpagina van het internet citeert: volgens de auteurs haalt zo'n antwoord een hoge citation recall en een lage citation precision. Liu en collega's laten ook meerdere citaties samen een zin steunen, en tellen ook een citatie die een deel van een zin steunt; deze les laat beide weg: een zin telt als gesteund als een van de citaties de zin steunt. In 2023 controleerden de drie onderzoekers de antwoorden van Bing Chat, NeevaAI, perplexity.ai en YouChat op 1.450 zoekvragen. Gemiddeld werd slechts 51,5% van de geschreven zinnen volledig gesteund door de citaties, en steunde slechts 74,5% van de citaties de zin erbij. En hoe nuttiger mensen de antwoorden van een engine vonden, hoe lager de citation precision: een correlatie van r = −0,96.

    **Fouten zonder alternatieven.** In december 2024 stelde de BBC aan ChatGPT, Copilot, Gemini en Perplexity 100 vragen over het nieuws, en 45 journalisten beoordeelden de 362 antwoorden. Bij 51% van alle antwoorden vonden de journalisten ernstige problemen van een of andere soort. En 19% van de antwoorden met een bron van de BBC bracht feitelijke fouten in: onjuiste beweringen, getallen en datums. Een citatie maakt zo'n fout overtuigender: volgens de BBC vertrouwt het publiek een antwoord eerder als een AI-assistent een vertrouwde naam zoals de BBC als bron noemt, ook als het antwoord fout is. Volgens Chirag Shah en Emily Bender (2022) wekt een antwoord de indruk dat de zaak afgehandeld is, terwijl verwijzingen naar plekken om verder te zoeken dat niet doen. En een antwoord uit meerdere bronnen verbergt de rest: door resultaten uit verschillende bronnen samen te voegen, verhult het systeem welke bronnen er zijn, en kan de gebruiker deze ruimte niet meer zelf verkennen.
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
    | **answer engine** | een zoekmachine die antwoordt met één tekst en de bronnen van deze tekst, in plaats van met een pagina met links |
    | **citatie** | het nummer achter een zin van een antwoord, dat wijst naar de bron van de zin |
    | **success@k** | het deel van de vragen met minstens één relevant document bij de eerste k resultaten |
    | **citation recall** | het deel van de zinnen van een antwoord dat door een van de citaties gesteund wordt |
    | **citation precision** | het deel van de citaties van een antwoord dat de zin erbij steunt |
    | **zero-click search** | een zoekopdracht waarna de lezer geen resultaat opent |
    | **good abandonment** | een zoekopdracht zonder klik omdat de pagina met resultaten al gaf wat nodig was |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een pagina met tien links te vergelijken met één antwoord:

    1. Rangschik bij elke vraag de documenten, zoals BM25 doet.
    2. De pagina: tel de vragen met een relevant document bij de eerste k resultaten, en deel door het aantal vragen. Dat is success@k.
    3. Het antwoord: lees de eerste resultaten, neem de zin met de hoogste query score, de eerste bij een gelijke stand, en citeer de plaats van het resultaat.
    4. Tel de antwoorden waarvan de zin uit een relevant document komt en het antwoord bevat.
    5. Controleer bij een antwoord met meer zinnen de citaties: welke zinnen de citaties steunen (citation recall), en welke citaties de zin erbij steunen (citation precision).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.EXAMPLE` is de vraag over het parlement van Victoria uit [Snippets: de resultaten laten zien](/course/open/ir-snippets?lang=nl), en `C.question_place(C.EXAMPLE, L.lang)` geeft het nummer van de vraag. `C.links(L.lang, k)` geeft de tien links van vraag `k`, de eerste tien alinea's van BM25, en `C.relevant_of(L.lang, k)` een lijst met de eigen alinea. `C.success_at_k` is de versie van de course van oefening 1. `C.engine(L.lang, k, read)` geeft het antwoord van de engine van de course na het lezen van de eerste `read` resultaten, als de plaats van het resultaat, de alinea, en het begin en eind van de zin in de woorden; `C.answer_right` zegt of de zin het antwoord toont, en `C.answer_of` schrijft het antwoord uit met de citatie. Tot slot geven `C.success_share` en `C.engine_share` success@k en het deel goede antwoorden over alle vragen, voor elk getal in `C.KS`, en `C.misses` hoeveel antwoorden na drie resultaten fout zijn, en bij hoeveel van deze vragen de eigen alinea bij de tien links staat:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _links = C.links(L.lang, _k)
    _relevant = C.relevant_of(L.lang, _k)
    print(_links, _relevant)
    for _read in (1, 2, 3):
        _right = C.answer_right(L.lang, _k, _read)
        print(_read, C.success_at_k(_links, _relevant, _read), C.engine(L.lang, _k, _read), _right)
    print(C.answer_of(L.lang, _k, 3))
    for _read in C.KS:
        print(_read, round(C.success_share(L.lang, _read), 3), round(C.engine_share(L.lang, _read), 3))
    print(C.misses(L.lang, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste regel geeft de tien links en de eigen alinea, 513, op plaats 1. Bij één, twee en drie resultaten geeft `success_at_k` dus `True`, en antwoordt de engine met de eerste zin van alinea 513, woorden 0 tot 51: de zin bevat *parlement* en *Victoria*, twee van de vijf zoekwoorden, een query score van 2² / 5 = 0,8, en het antwoord, *Victoria Constitution Act 1855*. Alinea 511, over de geschiedenis van Victoria, staat hier op plaats 7.

    De volgende vijf regels geven bij 1, 2, 3, 5 en 10 resultaten success@k en het deel goede antwoorden. success@k stijgt van 0,735 naar 0,919, en het deel goede antwoorden blijft rond 0,6: 0,598 bij drie resultaten, en 0,595 bij vijf en tien. Het verschil groeit van regel tot regel. De laatste regel: na drie resultaten zijn 476 antwoorden fout, en bij 380 van deze vragen staat de eigen alinea bij de tien links.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. success_at_k: staat er een relevant document op de pagina?

        Schrijf `success_at_k`, dat `ranking` krijgt, de nummers van de documenten met het beste eerst, als lijst, `relevant`, de nummers van de relevante documenten, en `k`. De functie geeft `True` terug als een van de eerste `k` documenten van `ranking` in `relevant` staat, en `False` als geen enkel document daar staat. In de startcode gaat `for doc in ...` door documenten, geeft `if doc in relevant` bij het eerste relevante document `True` terug, en geeft de functie na de loop `False` terug. Vervang de `...` door de eerste `k` documenten van `ranking`.

        De check roept `success_at_k([511, 513, 526], [513], 1)` aan, wat `False` geeft: het relevante document 513 staat op plaats 2 in deze kleine voorbeeldrangschikking. En `success_at_k([511, 513, 526], [513], 2)`, wat `True` geeft. Daarna eigen gevallen: een lege rangschikking, een `k` groter dan de rangschikking, een `k` van 0, en de rangschikkingen van de course.
        """),
        L.task("Laat de loop door de eerste `k` documenten van `ranking` gaan, en niet meer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("success_at_k", lines=5)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("success_at_k", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "success_at_k",
        [
            "De loop gaat door `ranking`, één `doc` per keer, maar alleen door de eerste `k` documenten: een slice vanaf het begin van de lijst.",
            "Een slice `a[:j]` bevat de items vanaf het begin tot plaats `j`, geteld vanaf 0, zonder plaats `j` zelf: `a[:2]` bevat de items op plaats 0 en 1, de eerste twee, en een slice langer dan de lijst bevat de hele lijst.",
            "`    for doc in ranking[___]:`",
        ],
        why="`for doc in ranking[:k]:`: de slice bevat de eerste `k` documenten, of alle documenten als de rangschikking korter is, dus een `k` groter dan de rangschikking werkt ook, en een `k` van 0 geeft een lege slice en `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    In het plaatje antwoordt je functie bij de vraag over Victoria: de eigen alinea 513 staat op plaats 1, dus `k` = 1, 2 en 3 geven allemaal `True`. Daaronder je functie over alle 1.183 vragen: success@1 is 73,5%, en de pagina met tien links bevat de eigen alinea bij 91,9% van de vragen. De eerste plaatsen leveren het meeste op: van 1 naar 3 resultaten stijgt success@k met 11,7 punten, van 3 naar 10 met 6,7.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recall():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. citation_recall: welke zinnen worden door de citaties gesteund?

        Schrijf `citation_recall`, dat `cited` en `supports` krijgt, twee lijsten met een lijst voor elke zin van een antwoord. `cited[i]` bevat de bronnen die in zin `i` geciteerd worden, en de lijst op plaats `i` van `supports` bevat de bronnen die steun geven aan zin `i`. De functie geeft het deel van de zinnen terug dat door een van de citaties gesteund wordt. In de startcode telt `supported` deze zinnen, en gaat `for i in range(len(cited))` door de zinnen. Bij elke zin begint `found` op `False`, gaat `for source in cited[i]` door de citaties van de zin, en wordt `found` gelijk aan `True` als een van de citaties de zin steunt. Een zin zonder citaties is nooit gesteund. Vervang de `...` door de test: is `source` een van de bronnen die steun geven aan zin `i`?

        De check roept `citation_recall([[1], [1, 2], [2, 3], []], [[1], [1], [4, 5], [2, 6]])` aan, wat `0.5` geeft: de eerste twee zinnen citeren bron 1, en bron 1 steunt deze zinnen, de derde zin citeert bron 2 en 3 terwijl alleen 4 en 5 de zin steunen, en de vierde zin citeert niets. En `citation_recall([[1, 2, 3, 4]], [[1, 2]])`, wat `1.0` geeft: één zin met vier citaties, waarvan twee de zin steunen. Daarna eigen gevallen: de twee voorbeelden van Liu, Zhang en Liang, een antwoord zonder citaties, en een dubbel geciteerde bron.
        """),
        L.task("Test of `source` een van de bronnen die steun geven aan zin `i` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recall():
    box_recall = L.code_box("citation_recall", lines=10)
    box_recall
    return (box_recall,)


@app.cell(hide_code=True)
def check_recall(box_recall):
    result_recall = L.check("citation_recall", box_recall.value)
    result_recall
    return (result_recall,)


@app.cell(hide_code=True)
def hints_recall():
    L.hints(
        "citation_recall",
        [
            "De test vraagt of `source` voorkomt in de lijst op plaats `i` van `supports`: de lijst bij zin `i`.",
            "`x in a` vraagt of `x` een item van de lijst `a` is: `4 in [4, 5]` geeft `True`. En `a[2]` is het item op plaats 2 van `a`, geteld vanaf 0.",
            "`            if source in ___:`",
        ],
        why="`if source in supports[i]:`: de lijst op plaats `i` van `supports` bevat de bronnen die steun geven aan zin `i`, op dezelfde plaats als de citaties van de zin in `cited`.",
    )
    return


@app.cell(hide_code=True)
def text_after_recall():
    mo.md(r"""
    In het plaatje meet je functie vier antwoorden: de twee van de oefening en twee voorbeelden van Liu, Zhang en Liang. Een citation recall van 1 krijgt het antwoord dat bij de ene zin elke bron citeert, en ook voorbeeld (a), hoewel slechts drie van de acht citaties de zin erbij steunen. In voorbeeld (c) citeert alleen de eerste zin iets, dus twee van de drie zinnen zijn niet gesteund: 1/3. Citation recall kijkt naar de zinnen; een citatie te veel ziet deze maat niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_precision():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. citation_precision: welke citaties steunen de zin?

        Schrijf `citation_precision`, dat `cited` en `supports` krijgt zoals in oefening 2, en teruggeeft welk deel van alle citaties van het antwoord de zin erbij steunt. In de startcode beginnen `good` en `total` op 0, en gaat `for i in range(len(cited))` door de zinnen. Daarbinnen gaat `for source in cited[i]` door de citaties van zin `i`, en telt `if source in supports[i]` 1 op bij `good` voor elke citatie die de zin steunt. Een antwoord zonder citaties geeft `0.0`, zodat de functie nooit door 0 deelt. Vervang de `...` door de nieuwe `total`: de citaties tot nu toe, plus de citaties van zin `i`.

        De check roept `citation_precision([[1], [1, 2], [2, 3], []], [[1], [1], [4, 5], [2, 6]])` aan, wat `0.4` geeft: van de vijf citaties steunen er twee de zin, bron 1 bij de eerste zin en bron 1 bij de tweede. En `citation_precision([[1, 2, 3, 4]], [[1, 2]])`, wat `0.5` geeft: twee van de vier citaties. Daarna eigen gevallen: de twee voorbeelden van Liu, Zhang en Liang, een antwoord zonder citaties, en een dubbel geciteerde bron.
        """),
        L.task("Tel het aantal citaties van zin `i` op bij `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("citation_precision", lines=10)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("citation_precision", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "citation_precision",
        [
            "`total` telt alle citaties van het antwoord: bij elke zin groeit `total` met het aantal citaties van de zin, de lengte van de lijst van de zin in `cited`.",
            "`len(a)` geeft het aantal items van een lijst: `len([2, 3])` geeft `2`. En `cited[1]` is de lijst met citaties van de tweede zin.",
            "`        total = total + ___`",
        ],
        why="`total = total + len(cited[i])`: elke citatie telt, ook een citatie zonder steun, dus meer bronnen citeren verhoogt `total`, en alleen een steunende bron verhoogt `good`.",
    )
    return


@app.cell(hide_code=True)
def text_after_precision():
    mo.md(r"""
    In het plaatje dezelfde vier antwoorden, nu met je citation precision naast de citation recall. Bij het antwoord met elke bron is de citation recall 1 en de citation precision 0,5; voorbeeld (a) haalt 1 en 0,375, de 3/8 van Liu en collega's. Voorbeeld (c) draait het om: citation precision 0,667 en citation recall 0,333, want de gegeven citaties kloppen grotendeels, terwijl twee van de drie zinnen niets citeren. Echte answer engines verschillen op dezelfde manier. In het onderzoek van Liu en collega's had Bing Chat de hoogste citation precision, 89,5, met een citation recall van 58,7; perplexity.ai had de hoogste citation recall, 68,7, met een citation precision van 72,7; en YouChat haalde een citation recall van slechts 11,1.

    ### Probeer het: één antwoord of tien links

    Kies een vraag en hoeveel resultaten de answer engine van de course leest. Het plaatje toont het antwoord met de citatie, of de zin uit de eigen alinea komt en het antwoord bevat, en daaronder de tien links, met de eigen alinea gemarkeerd. De tabel geeft over alle vragen success@k en het deel goede antwoorden bij het aantal resultaten dat je koos, en bij tien.

    Kies eerst de vraag over Jacksonville en schuif de slider van 1 naar 3: vanaf welk aantal toont het antwoord het antwoord? Kies dan de vraag over de vroegere teams van Los Angeles, en schuif de slider van 1 naar 10. Waar staat de eigen alinea bij de links, en verandert het antwoord ooit?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _questions = C.try_options(L.lang)
    try_question = mo.ui.dropdown(options=_questions, value=next(t for t, q in _questions.items() if q == C.EXAMPLE),
                                  label="Vraag")
    try_read = mo.ui.slider(1, C.LINKS, step=1, value=C.READ, show_value=True, label="Gelezen resultaten")
    mo.vstack([try_question, try_read])
    return try_question, try_read


@app.cell(hide_code=True)
def try_look(try_question, try_read):
    mo.Html(C.try_picture(L.lang, try_question.value, try_read.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Bij Jacksonville toont het antwoord het antwoord vanaf twee resultaten. Bij Los Angeles staat de eigen alinea op plaats 2 van de links, en de eerste zin van deze alinea bevat het antwoord: twee voetbalteams, van 2005 tot 2014. Maar het eerste resultaat, over de professionele sportteams van Zuid-Californië, heeft een zin met drie zoekwoorden, *Los*, *Angeles* en *teams*, en de zin van de eigen alinea heeft er slechts twee: *Soccer-teams* telt niet als *teams*. Dus is het antwoord bij elk aantal resultaten de lijst met de teams van nu: een fout antwoord, met niets ernaast dat de fout meldt. De vraag over Sky+ HD gaat op een andere manier mis: de eigen alinea komt eerst, maar de engine kiest de zin over MPEG-4, met *Sky* en *gebruikt*, en niet de volgende zin, met het antwoord, OpenTV. In *OpenTV-systeem* herkent de engine het zoekwoord *systeem* niet, dus deze zin heeft alleen *eigen*. Bij de vragen over Victoria en de Saksische Tuin werkt de engine wel.

    De engine van de course neemt de zin letterlijk over, dus de citatie steunt het antwoord altijd: de fouten zijn een verkeerde zin uit een echte bron. Een answer engine met een LLM schrijft nieuwe zinnen, en kan ook een zin schrijven zonder steun in een bron. Hoe je zo'n engine bouwt en de antwoorden meet, staat in [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) en [Klopt het RAG-systeem?](/course/open/build-rag-eval?lang=nl).

    ### Oefening 4. Tien links of één antwoord?

    Deel elk punt hieronder in: hoort het bij een pagina met tien links, of bij één antwoord?
    """)
    return


@app.cell(hide_code=True)
def exercise_pages():
    sort_pages = L.sort("pages")
    sort_pages
    return (sort_pages,)


@app.cell(hide_code=True)
def check_pages(sort_pages):
    result_pages = L.check("pages", sort_pages)
    result_pages
    return (result_pages,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tien links zijn de rangschikking van [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), success@k hoort bij de maten van [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl), en het antwoord is de beste zin uit [Snippets: de resultaten laten zien](/course/open/ir-snippets?lang=nl), een extractieve samenvatting zoals in [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl). Met deze les eindigt de module over zoeken. De volgende les, *Talen met weinig data*, begint de module over meer talen.

    **Een voorbeeld uit het dagelijks leven.** Geeft een verkoper je meteen één product, dan hoef je niet langs de schappen te zoeken. Als het product niet klopt, merk je dat pas thuis; bij het schap had je de andere producten gezien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_success, result_recall, result_precision, result_pages):
    L.wrap_up(result_success, result_recall, result_precision, result_pages)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
