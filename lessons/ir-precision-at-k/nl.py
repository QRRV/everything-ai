# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-precision-at-k. Route: course/route/lessen/taal-en-zoeken.yaml. Controles: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-precision-at-k", "nl")
    C = L.checks   # de dertien documenten van de course, de rangschikkingen van XQuAD, SciFact en de controles


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) en [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een zoekopdracht geeft 8 documenten, en 6 daarvan zijn relevant. De collectie bevat in totaal 10 relevante documenten. Wat zijn de precision en de recall van deze zoekopdracht?",
        options={
            "swap": "precision 0,6 en recall 0,75",
            "found": "precision 0,8 en recall 0,6, uit 8 van 10 en 6 van 10",
            "right": "precision 0,75 en recall 0,6, uit 6 van 8 en 6 van 10",
            "same": "allebei 0,75, want precision en recall tellen dezelfde 6 relevante documenten",
        },
        correct="right",
        why={
            "right": "Precision deelt de 6 relevante resultaten door de 8 resultaten: 0,75. Recall deelt deze resultaten door de 10 relevante documenten: 0,6. Deze les neemt precision mee naar een gerangschikte lijst: de precision van de eerste k resultaten.",
            "swap": "Welk getal telt de resultaten, en welk de relevante documenten: kijkt precision naar wat de zoekopdracht gaf, of naar wat de zoekopdracht had moeten geven? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "found": "De zoekopdracht geeft 8 documenten, maar zijn alle 8 relevant? Wat deelt precision, en waardoor? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "same": "Allebei tellen de 6 relevante resultaten, maar delen deze twee maten door hetzelfde getal? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_tune():
    quick_tune = L.choice(
        "quick-tune",
        question="In *BM25: de klassieke rangschikking* werden k1 en b afgesteld op de 809 trainingsbeweringen van SciFact, en daarna gemeten op de 300 testbeweringen. Waarom stel je k1 en b niet ook af op de testbeweringen?",
        options={
            "few": "300 beweringen zijn te weinig om alle 651 instellingen van het rooster op te proberen",
            "right": "een instelling gekozen op de testbeweringen doet het daar beter dan straks op nieuwe beweringen",
            "none": "de testbeweringen hebben geen relevante abstracts om op af te stellen",
            "slow": "afstellen duurt te lang op de testbeweringen, omdat de abstracts daar veel langer zijn dan bij de trainingsbeweringen",
        },
        correct="right",
        why={
            "right": "Een instelling die je kiest door naar de testbeweringen te kijken, past bij deze beweringen, en belooft meer dan de instelling bij nieuwe beweringen waarmaakt: afgesteld op de trainingsbeweringen zette BM25 bij 53,7% van de testbeweringen een relevant abstract bovenaan, en de beste instelling op de testbeweringen zelf haalde 56,0%. Deze les gaat over de test zelf: vaste zoekvragen met bekende relevante documenten, en de getallen die een rangschikking daarop een score geven.",
            "few": "651 instellingen proberen lukt bij elk aantal beweringen. Wat gaat er mis als je afstelt op dezelfde beweringen als waarop je meet? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "none": "De testbeweringen hebben relevante abstracts: de les heeft ermee gemeten. Wat gaat er mis als je er ook op afstelt? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "slow": "Beide sets beweringen zoeken in dezelfde 5.183 abstracts. Wat gaat er mis als je afstelt op dezelfde beweringen als waarop je meet? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
        },
    )
    quick_tune
    return (quick_tune,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl) en [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl) besliste één getal welke rangschikking het beter deed: het deel van de vragen van XQuAD met de eigen alinea bovenaan. BM25 haalt daar 73,5%, de cosine similarity 62,6%. Dit getal kijkt alleen naar de eerste plaats: een rangschikking met de eigen alinea op plaats 2 krijgt daar even weinig voor als een rangschikking met de eigen alinea helemaal onderaan. En het werkt alleen omdat elke vraag één alinea heeft waarbij de vraag geschreven is. Een bewering van SciFact kan vijf relevante abstracts hebben, en een zoekmachine laat tien resultaten tegelijk zien.

    Zoeken wordt al sinds het eind van de jaren vijftig op een vaste manier gemeten. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze schrijven in *Introduction to Information Retrieval* (2008) dat je om op de standaardmanier te meten een testcollectie nodig hebt van drie dingen: documenten, informatiebehoeften in de vorm van zoekvragen, en relevance judgments, één voor elk paar van een zoekvraag en een document. Met deze drie krijgt een rangschikking een getal, en kun je twee rangschikkingen op dezelfde test vergelijken. Deze les voegt drie getallen toe die naar meer kijken dan de eerste plaats.

    **Een voorbeeld uit het dagelijks leven.** Een docent beoordeelt elke leerling met dezelfde toets en hetzelfde antwoordmodel. De vragen van de toets zijn de zoekvragen, het antwoordmodel bevat de relevance judgments, en de manier waarop de punten worden geteld, bepaalt wie er het best uitkomt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_p10 = L.guess(
        "guess",
        question="BM25 zet bij 91,9% van de 1.183 vragen de eigen alinea bij de eerste tien. De precision at 10 is het deel relevante alinea's bij deze eerste tien, gemiddeld over de vragen. Hoe groot is deze?",
        options={
            "same": "ongeveer 0,92, even hoog als het deel",
            "eight": "ongeveer 0,74, het deel met de eigen alinea bovenaan",
            "half": "ongeveer 0,5",
            "right": "net onder 0,1",
        },
        reveal="Net onder 0,1: 0,092. Elke vraag heeft één relevante alinea, dus hoogstens één van de eerste tien alinea's is relevant, en zelfs een perfecte rangschikking haalt een precision at 10 van 0,1. De 91,9% is een ander getal: het deel van de vragen met de eigen alinea bij de eerste tien, de recall at 10. Wat een maat kan halen, hangt af van het aantal relevante documenten per zoekvraag.",
    )
    guess_p10
    return (guess_p10,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Het Cranfield-paradigma.** Volgens het boek werden zoeksystemen voor het eerst streng en formeel getest in de Cranfield-experimenten, vanaf het eind van de jaren vijftig. De Cranfield-collectie bevat 1.398 abstracts van papers over aerodynamica, 225 zoekvragen, en een relevance judgment voor elk paar van een zoekvraag en een document. Zo meten, elk systeem op dezelfde vaste testcollectie, heet het Cranfield-paradigma. TREC, de evaluatie van het Amerikaanse instituut NIST sinds 1992, werkt op dezelfde manier, met veel grotere collecties.

    **Relevance judgments.** Een relevance judgment zegt of een document relevant is voor de informatiebehoefte achter een zoekvraag: in deze les relevant of niet relevant, niets ertussenin. Het boek benadrukt dat relevantie wordt beoordeeld ten opzichte van een informatiebehoefte, niet van een zoekvraag. In [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl) hadden twee van de drie behoeften dezelfde zoekvraag, *python*, en had elke behoefte eigen relevante documenten. Dertien documenten kun je voor elke behoefte beoordelen, zoals in Cranfield, maar bij grote collecties beoordeelt men volgens het boek meestal alleen een deel van de documenten per zoekvraag: TREC beoordeelt de documenten die de geteste systemen bij de eerste resultaten zetten. De twee testcollecties van deze module zijn elk op een eigen manier beoordeeld. In XQuAD is de ene relevante alinea van een vraag de alinea waarbij de vraag geschreven is; in SciFact zijn de relevante abstracts van een bewering de abstracts waar de zin naar verwijst, de zin waarbij de bewering geschreven is. Elk ander document telt als niet relevant, ook als het zou helpen.

    **Precision at k.** Knip de rangschikking af na k documenten, en neem de precision van wat er overblijft: het aantal relevante documenten bij de eerste k, gedeeld door k. Het deel dat deze module tot nu toe gebruikte, de vragen met de eigen alinea bovenaan, is de precision at 1. De precision at 10 zegt hoeveel van een pagina met tien resultaten helpt. Het boek noemt het voordeel: je hoeft niet te schatten hoeveel relevante documenten er zijn. En de nadelen: van de gangbare maten is precision at k het minst stabiel, en het gemiddelde over zoekvragen zegt weinig, omdat het aantal relevante documenten van een zoekvraag de precision at k sterk bepaalt.

    **Reciprocal rank.** Neem 1 gedeeld door de plaats van het eerste relevante document: 1 voor de eerste plaats, 0,5 voor de tweede, 0,333 voor de derde, en 0 als er geen relevant document komt. Het gemiddelde over de zoekvragen is de mean reciprocal rank, MRR. Ellen Voorhees en Dawn Tice beschrijven hoe de vraag-en-antwoordtaak van TREC-8, in 1999, vijf antwoorden op elke vraag een score gaf: een vraag kreeg 1 gedeeld door de plaats van het eerste goede antwoord, of 0 als geen van de vijf antwoorden goed was. MRR beloont een tweede plaats, maar kijkt naar niets na het eerste relevante document. Volgens Voorhees en Tice krijgt een systeem geen extra punten voor meer verschillende goede antwoorden.

    **Average precision.** Loop de rangschikking af, en neem bij de plaats van elk relevant document de precision tot deze plaats. Tel deze precisions op, en deel door het aantal relevante documenten van de zoekvraag. Een relevant document dat niet in de rangschikking staat, voegt niets toe, maar telt wel mee in dit aantal: het boek telt de precision daar als 0. Het gemiddelde over de zoekvragen is de mean average precision, MAP. Volgens het boek onderscheidt MAP systemen bijzonder goed en is de maat stabiel. Met één relevant document is de average precision de precision op de plaats van dit document, 1 gedeeld door de plaats: de reciprocal rank.

    Voor een zoekvraag met de verzameling $R$ van relevante documenten:

    $$\text{P@}k = \frac{\text{relevante documenten bij de eerste } k}{k} \qquad \text{RR} = \frac{1}{\text{plaats van het eerste relevante document}} \qquad \text{AP} = \frac{1}{|R|} \sum_{d \in R} \text{P@plaats}(d)$$

    waarin een relevant document $d$ dat niet in de rangschikking staat, 0 toevoegt aan AP. MRR en MAP zijn de gemiddelden van RR en AP over de zoekvragen.

    **De drie behoeften.** Het plaatje laat de drie behoeften bij de dertien documenten van de course zien, elk met de resultaten van BM25 voor de zoekvraag, en de precision op elke plaats. Bij behoefte 1, een python als huisdier kopen, staat document 3 vierde, en document 1, *Koningspythons te koop*, is geen resultaat: er staat *koningspythons*, niet *python*. De reciprocal rank is 0,25, en de average precision 0,125: de precision op plaats 4, 0,25, gedeeld door de twee relevante documenten. Bij behoefte 2, Python leren, staat document 0 pas tweede: reciprocal rank en average precision 0,5. Bij de behoefte over rode wijn staat document 6 bovenaan, maar document 9 is geen resultaat: er staat *hartaanvallen* en *rood*, niet *hart* en *rode*. De reciprocal rank is 1, de average precision (1 + 0) / 2 = 0,5. Over de drie behoeften is de MRR 0,583 en de MAP 0,375.
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
    | **testcollectie** | documenten, zoekvragen en relevance judgments: de vaste test waarop rangschikkingen worden gemeten |
    | **Cranfield-paradigma** | zoeksystemen meten op een vaste testcollectie, zoals de Cranfield-experimenten deden vanaf het eind van de jaren vijftig |
    | **relevance judgment** | het oordeel of een document relevant is voor de informatiebehoefte achter een zoekvraag, gegeven door mensen of door de opbouw van de collectie; hier relevant of niet relevant |
    | **precision at k**, P@k | het deel van de eerste k documenten van een rangschikking dat relevant is |
    | **reciprocal rank**, RR | 1 gedeeld door de plaats van het eerste relevante document, of 0 als er geen komt |
    | **mean reciprocal rank**, MRR | het gemiddelde van de reciprocal rank over de zoekvragen |
    | **average precision**, AP | de precision op de plaats van elk relevant document, opgeteld en gedeeld door het aantal relevante documenten |
    | **mean average precision**, MAP | het gemiddelde van de average precision over de zoekvragen |
    | $R$, $\lvert R \rvert$ | de verzameling relevante documenten van een zoekvraag, en hoeveel het er zijn |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo meet je een rangschikking op een testcollectie:

    1. Rangschik de documenten voor elke zoekvraag, en houd de eerste resultaten, bijvoorbeeld 100.
    2. Zoek de relevante documenten van elke zoekvraag op in de relevance judgments.
    3. Precision at k: tel de relevante documenten bij de eerste k, en deel door k.
    4. Reciprocal rank: 1 gedeeld door de plaats van het eerste relevante document, of 0 als er geen komt.
    5. Average precision: bij elk relevant document de precision tot de eigen plaats; tel deze op, en deel door het aantal relevante documenten.
    6. Neem van elk het gemiddelde over alle zoekvragen: de gemiddelde P@k, de MRR en de MAP.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.rankings(L.lang, "bm25")` geeft voor elke vraag van XQuAD de eerste 100 alinea's van BM25, de beste eerst, en `"cosine"` de alinea's van de cosine similarity, zoals in de lessen hiervoor; alinea's met dezelfde score staan in de volgorde van het nummer. `C.relevant(L.lang)` geeft voor elke vraag een lijst met de ene relevante alinea: de eigen alinea. `C.precision_at_k`, `C.reciprocal_rank` en `C.average_precision` zijn de maten van de course, en `C.question_place(C.EXAMPLE, L.lang)` geeft de plaats van de voorbeeldvraag tussen de vragen. Eerst de vraag door welke mensen het gebied van Jacksonville werd bewoond, toen de Fransen er nog niet waren, daarna de gemiddelden over alle vragen:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _relevant = C.relevant(L.lang)[_k]
    for _how in ("cosine", "bm25"):
        _ranking = C.rankings(L.lang, _how)[_k]
        print(_how, _ranking[:5], _relevant, C.precision_at_k(_ranking, _relevant, 1),
              C.precision_at_k(_ranking, _relevant, 5), round(C.reciprocal_rank(_ranking, _relevant), 3),
              round(C.average_precision(_ranking, _relevant), 3))
    for _how in ("cosine", "bm25"):
        _rankings = C.rankings(L.lang, _how)
        _relevants = C.relevant(L.lang)
        _n = len(_rankings)
        _p1 = sum([C.precision_at_k(_rankings[_q], _relevants[_q], 1) for _q in range(_n)]) / _n
        _p10 = sum([C.precision_at_k(_rankings[_q], _relevants[_q], 10) for _q in range(_n)]) / _n
        _mrr = sum([C.reciprocal_rank(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        _map = sum([C.average_precision(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        print(_how, _n, round(_p1, 3), round(_p10, 3), round(_mrr, 3), round(_map, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste twee regels zijn de vraag over Jacksonville: de eerste vijf alinea's, de relevante alinea, de precision at 1 en at 5, de reciprocal rank en de average precision. BM25 zet alinea 1382 bovenaan en de eigen alinea, 1380, op plaats 2: de precision at 1 is 0, at 5 is deze 0,2, en de reciprocal rank en de average precision zijn allebei 0,5. De cosine similarity zet de eigen alinea pas op plaats 6: bij de eerste vijf staat geen relevante alinea, en de reciprocal rank en de average precision zijn 0,167. Alinea 1382 vertelt dat de Mocama, een groep aan de kust binnen de Timucua, in de 16e eeuw in de regio woonden: deze alinea beantwoordt de vraag ook, maar de testcollectie noemt alleen de eigen alinea relevant.

    De laatste twee regels zijn de gemiddelden over alle 1.183 vragen: de precision at 1 en at 10, de MRR en de MAP. BM25 haalt 0,735 bij 1, de 73,5% van [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), 0,092 bij 10, en een MRR van 0,804, tegen 0,724 voor de cosine similarity. De MAP is bij beide rangschikkingen gelijk aan de MRR: elke vraag heeft één relevante alinea.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_precision_at_k():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. precision_at_k: de eerste k resultaten

        Schrijf `precision_at_k`, met als invoer `ranking`, de documenten van een rangschikking als lijst, de beste eerst, `relevant`, een lijst met de relevante documenten van de zoekvraag, en `k`. De functie geeft de precision at k terug: het aantal relevante documenten bij de eerste `k` van de rangschikking, gedeeld door `k`. In de startcode begint `found` bij 0, en de loop telt er 1 bij op voor elk relevant `doc`. De regel met `return` deelt al door `k`, ook als de rangschikking minder dan `k` documenten bevat: een plaats zonder document telt als niet relevant. Vervang de `...` door de documenten waar de loop doorheen gaat: de eerste `k` van `ranking`.

        De controle roept `precision_at_k([7, 3, 1, 9, 4], [7, 4], 2)` aan, wat `0.5` geeft, en `precision_at_k([5, 2], [2, 8], 4)`, wat `0.25` geeft: de rangschikking bevat twee documenten, en de deling is door 4. Daarna eigen gevallen: k = 1, geen relevante documenten, een lege rangschikking, en de rangschikkingen van de course.
        """),
        L.task("Laat de loop door de eerste k documenten van de rangschikking gaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision_at_k():
    box_precision_at_k = L.code_box("precision_at_k", lines=6)
    box_precision_at_k
    return (box_precision_at_k,)


@app.cell(hide_code=True)
def check_precision_at_k(box_precision_at_k):
    result_precision_at_k = L.check("precision_at_k", box_precision_at_k.value)
    result_precision_at_k
    return (result_precision_at_k,)


@app.cell(hide_code=True)
def hints_precision_at_k():
    L.hints(
        "precision_at_k",
        [
            "De loop moet door de eerste `k` documenten van `ranking` gaan; `doc` is dan om de beurt elk van deze documenten.",
            "Een slice geeft het begin van een lijst: `numbers[:3]` geeft de eerste drie getallen.",
            "`    for doc in ranking[___]:`",
        ],
        why="`for doc in ranking[:k]:`: de loop ziet alleen de eerste k documenten, en elk relevant document telt 1 op bij `found`.",
    )
    return


@app.cell(hide_code=True)
def text_after_precision_at_k():
    mo.md(r"""
    In het plaatje haalt BM25 een precision at 1 van 0,735 en at 10 van 0,092, en een perfecte rangschikking, met de eigen alinea altijd bovenaan, haalt 1 en 0,1. Met één relevante alinea per vraag kan de precision at k niet hoger worden dan 1 gedeeld door k: hier zegt de precision at 10 meer over de testcollectie dan over de rangschikking. Het deel van de vragen met de eigen alinea bij de eerste 10, 91,9% bij BM25, is de recall at 10, hier 10 keer de precision at 10.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reciprocal_rank():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. reciprocal_rank: het eerste relevante document

        Schrijf `reciprocal_rank`, met als invoer `ranking` en `relevant` zoals in oefening 1. De functie geeft 1 gedeeld door de plaats van het eerste relevante document terug, geteld vanaf 1, of 0 als de rangschikking geen relevant document bevat. In de startcode gaat `for place in range(len(ranking))` door de plaatsen van de rangschikking zoals Python telt, vanaf 0. Bij het eerste relevante document geeft de functie meteen de waarde terug; eindigt de loop zonder relevant document, dan geeft de laatste regel 0 terug. Vervang de `...` door de reciprocal rank: 1 gedeeld door de plaats, waarbij de eerste plaats 1 is, niet 0.

        De controle roept `reciprocal_rank([4, 8, 5, 2, 6], [6, 2])` aan, wat `0.25` geeft: document 2 is het eerste relevante document, op plaats 4. En `reciprocal_rank([5, 1], [9])`, wat `0` geeft. Daarna eigen gevallen: een relevant document op de eerste plaats, een lege rangschikking, en de rangschikkingen van de course.
        """),
        L.task("Geef 1 gedeeld door de plaats van het document terug, geteld vanaf 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reciprocal_rank():
    box_reciprocal_rank = L.code_box("reciprocal_rank", lines=5)
    box_reciprocal_rank
    return (box_reciprocal_rank,)


@app.cell(hide_code=True)
def check_reciprocal_rank(box_reciprocal_rank):
    result_reciprocal_rank = L.check("reciprocal_rank", box_reciprocal_rank.value)
    result_reciprocal_rank
    return (result_reciprocal_rank,)


@app.cell(hide_code=True)
def hints_reciprocal_rank():
    L.hints(
        "reciprocal_rank",
        [
            "`place` telt vanaf 0, dus de plaats van het document geteld vanaf 1 is één meer dan `place`.",
            "Deel 1 door dit getal, en zet het getal tussen haakjes, zodat Python eerst optelt en dan deelt.",
            "`            return 1 / ___`",
        ],
        why="`return 1 / (place + 1)`: de eerste plaats geeft 1, de tweede 0,5, de derde 0,333.",
    )
    return


@app.cell(hide_code=True)
def text_after_reciprocal_rank():
    mo.md(r"""
    In het plaatje zet BM25 de eigen alinea bij 870 vragen bovenaan, bij 103 op plaats 2 en bij 35 op plaats 3, en 26 eigen alinea's staan niet bij de eerste 100. De MRR is 0,804, tegen een precision at 1 van 0,735: elke vraag met de eigen alinea op plaats 2 voegt 0,5 toe aan het totaal waarvan het gemiddelde wordt genomen, waar P@1 deze vraag 0 geeft, en elke derde plaats 0,333. Bij de cosine similarity is het verschil groter, 0,724 tegen 0,626, omdat de cosine similarity vaker een eigen alinea op plaats 2 zet: 134 keer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_average_precision():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. average_precision: elk relevant document

        Schrijf `average_precision`, met als invoer `ranking` en `relevant` zoals hiervoor. De functie geeft de average precision terug: bij de plaats van elk relevant document de precision van de rangschikking tot deze plaats, opgeteld en gedeeld door het aantal relevante documenten. In de startcode begint `total` bij 0, vindt de loop `for place in range(len(ranking))` elk relevant document, en deelt de regel met `return` door `len(relevant)`, zodat een relevant document dat de rangschikking mist, 0 toevoegt. Vervang de `...` door wat elk relevant document toevoegt: `precision_at_k` uit oefening 1, met als invoer de rangschikking, de relevante documenten en k, in deze volgorde, waarbij k de plaats van het document is, geteld vanaf 1. De box kent `precision_at_k` al.

        De controle roept `average_precision([2, 5, 1, 7], [2, 7])` aan, wat `0.75` geeft: de precision 1 op plaats 1 en 0,5 op plaats 4, gedeeld door 2. En `average_precision([9, 4, 6], [4, 8])`, wat `0.25` geeft: de precision 0,5 op plaats 2, gedeeld door 2, omdat document 8 ontbreekt. Daarna eigen gevallen: elk relevant document bovenaan, elk relevant document ontbrekend, een lege rangschikking, en de rangschikkingen van de course.
        """),
        L.task("Laat elk relevant document de precision van de rangschikking tot de eigen plaats toevoegen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average_precision():
    box_average_precision = L.code_box("average_precision", lines=6)
    box_average_precision
    return (box_average_precision,)


@app.cell(hide_code=True)
def check_average_precision(box_average_precision):
    result_average_precision = L.check("average_precision", box_average_precision.value)
    result_average_precision
    return (result_average_precision,)


@app.cell(hide_code=True)
def hints_average_precision():
    L.hints(
        "average_precision",
        [
            "Elk relevant document voegt `precision_at_k` van de rangschikking tot de eigen plaats toe: het nieuwe `total` is het oude `total` plus deze precision.",
            "`precision_at_k` krijgt de rangschikking, de relevante documenten, en als k de plaats geteld vanaf 1, één meer dan `place`, zoals in oefening 2.",
            "`            total = total + precision_at_k(___)`",
        ],
        why="`total = total + precision_at_k(ranking, relevant, place + 1)`: elk relevant document voegt de precision van de rangschikking tot de eigen plaats toe, en de laatste regel deelt door alle relevante documenten, gevonden of niet.",
    )
    return


@app.cell(hide_code=True)
def text_after_average_precision():
    mo.md(r"""
    Bij de drie behoeften geeft je functie een average precision van 0,125, 0,5 en 0,5, en een MAP van 0,375, waar de MRR 0,583 is. Op XQuAD is je MAP gelijk aan de MRR: 0,804 bij BM25, omdat met één relevante alinea de average precision 1 gedeeld door de plaats van de alinea is. MAP en MRR verschillen pas waar zoekvragen meer dan één relevant document hebben: van de 300 testbeweringen van SciFact hebben er 23 twee tot vijf.

    ### Probeer het: MRR en MAP op SciFact

    SciFact, uit [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), is een testcollectie van dezelfde soort: 5.183 abstracts, en bij elke bewering de abstracts waar de zin van de bewering naar verwijst als relevante abstracts. Van de 300 testbeweringen hebben er 277 één relevant abstract, en 23 twee tot vijf. SciFact is Engels, dus de beweringen in de demo zijn Engels. Download SciFact, kies een van deze 23 beweringen, en verschuif k. Het plaatje laat de eerste k abstracts van de cosine similarity en van BM25 bij de bewering zien, de plaatsen van alle relevante abstracts, en de precision at k, reciprocal rank (RR) en average precision (AP). Daaronder staan de gemiddelden over alle 300 testbeweringen, naast een perfecte rangschikking.

    Begin met bewering 133: beide rangschikkingen zetten een relevant abstract bovenaan. Welke rangschikking vindt de andere vier eerder, en welke maten laten dat zien, en welke niet? Kies daarna bewering 971, en vergelijk de reciprocal rank met de average precision.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciFact, 8,2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scifact_cell(get_button):
    sci_claims = C.scifact_or_stop(get_button, {
        "ask": "*SciFact staat nog niet op je computer. De knop downloadt de set één keer van de kopie van BEIR op Hugging Face, 8,2 MB: de beweringen onder de licentie CC BY 4.0 en de abstracts onder ODC-By 1.0. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor 8,2 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    _options = C.try_options()
    try_claim = mo.ui.dropdown(options=_options, value=next(t for t, c in _options.items() if c == C.CLAIM), label="Bewering")
    try_k = mo.ui.slider(C.TRY_K[0], C.TRY_K[1], step=1, value=C.TRY_K[2], label="k", show_value=True)
    mo.hstack([try_claim, try_k], justify="start", gap=2)
    return sci_claims, try_claim, try_k


@app.cell(hide_code=True)
def try_look(sci_claims, try_claim, try_k):
    mo.Html(C.try_picture(L.lang, try_claim.value, try_k.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Bij bewering 133 halen beide rangschikkingen een reciprocal rank van 1. Maar de cosine similarity vindt de andere relevante abstracts op de plaatsen 3, 7, 10 en 75, en BM25 op 10, 25 en 31, en het vijfde niet bij de eerste 100: een average precision van 0,512 tegen 0,290. Bij bewering 971 is het andersom: de cosine similarity heeft het eerste relevante abstract op plaats 2 en BM25 op plaats 3, maar BM25 heeft alle vier bij de eerste 7. De reciprocal rank is 0,5 tegen 0,333, de average precision 0,427 tegen 0,501. Over alle 300 testbeweringen doet BM25 het op elke maat beter: een MRR van 0,632 tegen 0,550, en een MAP van 0,620 tegen 0,536. Met k = 10 haalt zelfs de perfecte rangschikking een gemiddelde precision at 10 van slechts 0,113, omdat de meeste beweringen één relevant abstract hebben.

    Er blijven twee grenzen. De relevance judgments missen documenten die zouden helpen: bij 27 van de 313 vragen waar BM25 een andere alinea bovenaan zet, bevat deze alinea ook de tekst van een antwoord, zoals alinea 1382 bij de vraag over Jacksonville. En mensen oordelen verschillend: Voorhees en Tice merkten dat beoordelaars terecht van mening kunnen verschillen over wat goed is, zelfs bij korte, feitelijke vragen. Het boek beschrijft wat zulke verschillen deden met de volgorde van de systemen van TREC, in een onderzoek van Voorhees (2000): de getallen veranderen, maar de volgorde blijft vrij stabiel.

    ### Oefening 4. Welke maat past?

    Deel elke uitspraak hieronder in: deze past bij precision at k, MRR of MAP.
    """)
    return


@app.cell(hide_code=True)
def exercise_measures():
    sort_measures = L.sort("measures")
    sort_measures
    return (sort_measures,)


@app.cell(hide_code=True)
def check_measures(sort_measures):
    result_measures = L.check("measures", sort_measures)
    result_measures
    return (result_measures,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rangschikkingen die je hier meet, komen uit [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl) en [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), en de precision uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) telt nu in een gerangschikte lijst. In het domein *Bouwen met AI* meet [Klopt het RAG-systeem?](/course/open/build-rag-eval?lang=nl) hoe goed het zoeken van een RAG-systeem de juiste teksten vindt. De volgende les, *Relevantie in gradaties: nDCG*, laat een document meer of minder relevant zijn in plaats van alleen relevant of niet; en *Van tien links naar één antwoord* kijkt naar wat er verandert als een zoekmachine één antwoord schrijft in plaats van een gerangschikte lijst.

    **Een voorbeeld uit het dagelijks leven.** Een vriend geeft je een lijst met restaurants. De precision at 3 vraagt hoeveel van de eerste drie tips goed waren; de reciprocal rank vraagt hoe ver naar beneden de eerste goede stond; de average precision vraagt of alle goede restaurants vroeg kwamen, ook de laatste.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_precision_at_k, result_reciprocal_rank, result_average_precision, result_measures):
    L.wrap_up(result_precision_at_k, result_reciprocal_rank, result_average_precision, result_measures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
