# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-bm25. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-bm25", "nl")
    C = L.checks   # de dertien documenten van de course, de alinea's en vragen van SQuAD, SciFact en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="In *Rangschikken met de cosine similarity: het vector space model* stond document 6 er twee keer achter elkaar. Wat gebeurde er met de cosine similarity met de zoekvraag *rode wijn hart*?",
        options={
            "double": "deze verdubbelde, net als het inproduct",
            "right": "deze bleef gelijk: 0,386",
            "half": "deze halveerde",
            "one": "deze werd 1, omdat het document dan elk woord van de zoekvraag twee keer bevat",
        },
        correct="right",
        why={
            "right": "Elke waarde verdubbelt, dus het inproduct verdubbelt, en de lengte van de vector van het document ook: de cosine similarity blijft 0,386. Deze les stelt een andere vraag: wat moeten de tweede en de derde *wijn* toevoegen?",
            "double": "Het inproduct verdubbelde, van 1,454 naar 2,907. Wat gebeurde er met de lengte van de vector van het document, en dus met de deling? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
            "half": "Een langer document heeft een langere vector, maar het inproduct met de zoekvraag groeit ook. Hoe verhouden de twee veranderingen zich? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
            "one": "De cosine similarity is alleen 1 als de twee vectoren dezelfde kant op wijzen. Veranderde de richting van de vector toen het document er twee keer stond? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="In dezelfde les deelde pivoted length normalization het inproduct door een mengsel van de lengte en een pivot. Bij helling 0 wordt elk document door hetzelfde getal gedeeld. Welke volgorde krijgen de documenten dan?",
        options={
            "cosine": "de volgorde van de cosine similarity",
            "right": "de volgorde van het inproduct zelf",
            "short": "het kortste document bovenaan",
            "number": "de volgorde van het nummer, zoals bij AND in een inverted index",
        },
        correct="right",
        why={
            "right": "Elke score door hetzelfde getal delen verandert de volgorde niet: helling 0 geeft de volgorde van het inproduct, en daarmee winnen lange documenten. BM25 heeft ook een instelling voor hoeveel de lengte meetelt, en bij 0 telt de lengte helemaal niet mee.",
            "cosine": "De cosine similarity deelt door de lengte van de vector van het document. Is dat voor elk document hetzelfde getal? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
            "short": "Als elk inproduct door hetzelfde getal gedeeld wordt, blijft het grotere inproduct het grotere. Wat bepaalt dan de volgorde? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
            "number": "Een rangschikking zet de documenten op volgorde van een score. Welke score blijft over als elk document door hetzelfde getal gedeeld wordt? Zie [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl) zette de cosine similarity de eigen alinea bovenaan bij 62,6% van de 1.183 Nederlandse vragen van XQuAD. Neem de vraag *Waar ging de afvoer van gletsjers in Europa in de laatste ijstijd naartoe?* De cosine similarity zet een alinea over de huidige afvoer van de Rijn bovenaan: het woord *afvoer* staat er vier keer in, maar over gletsjers en de ijstijd zegt de alinea niets. De eigen alinea, met het antwoord, *de Rijn*, bevat *afvoer* één keer, *gletsjers* één keer en *Europa* drie keer, en komt op plaats twee. In een vector van TF-IDF-waarden weegt een woord dat vier keer voorkomt vier keer zo zwaar als een woord dat één keer voorkomt. Maar een alinea met vier keer *afvoer* gaat niet vier keer zoveel over afvoer, en een alinea met *afvoer* én *gletsjers* is al een goed teken.

    BM25 rangschikt documenten met dezelfde twee ingrediënten, hoe vaak de woorden van de zoekvraag in een document staan en de IDF van elk woord, en verandert twee dingen: elke herhaling van een woord voegt minder toe dan de vorige, en een eigen instelling bepaalt hoeveel de lengte van een document meetelt. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze schrijven in *Introduction to Information Retrieval* (2008) dat BM25 vaak Okapi-weging heet, naar het systeem waarin het voor het eerst werd ingebouwd, en dat zulke wegingen het goed deden in de evaluaties van TREC en door veel groepen werden overgenomen. Nieuwere manieren van zoeken meten zich nog steeds met BM25: BEIR, een verzameling testcollecties (zoekvragen met bij elke zoekvraag de documenten waarvan bekend is dat deze relevant zijn) van Nandan Thakur en anderen (2021), noemt BM25 in de samenvatting een robuuste baseline.

    **Een voorbeeld uit het dagelijks leven.** De eerste keer dat iemand je vertelt dat een restaurant goed is, leer je veel; de vijfde keer bijna niets nieuws. BM25 laat de vierde *afvoer* minder meetellen dan de eerste.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_bm25 = L.guess(
        "guess",
        question="De cosine similarity zette de eigen alinea bovenaan bij 62,6% van de 1.183 vragen. BM25 rangschikt dezelfde 2.067 alinea's met dezelfde IDF, maar laat elke herhaling van een woord minder toevoegen dan de vorige, met de instellingen die het boek redelijk noemt. Bij welk deel van de vragen zet BM25 de eigen alinea bovenaan?",
        options={
            "fewer": "minder vaak dan de cosine similarity: ongeveer de helft",
            "same": "ongeveer even vaak als de cosine similarity: zes op de tien",
            "right": "meer dan zeven op de tien",
            "most": "bijna altijd: meer dan negen op de tien",
        },
        reveal="Meer dan zeven op de tien: 73,5%, tegen 62,6% met de cosine similarity. Als elk woord één keer de IDF toevoegt, hoe vaak het er ook in staat, haalt BM25 al 70,9%, meer dan de cosine similarity. De winst is het grootst bij de vragen met de eigen alinea in de langste groep: van 54,2% met de cosine similarity naar 69,8%.",
    )
    guess_bm25
    return (guess_bm25,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Dezelfde ingrediënten.** BM25 geeft een document een score op de woorden van de zoekvraag die erin staan. Elk van deze woorden voegt de IDF toe, log10(N / df) zoals in [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl), maal een waarde die groeit met het aantal keer dat het woord in het document staat. Een woord dat niet in het document staat, voegt niets toe, dus net als bij de cosine similarity tellen alleen de postings lists van de woorden van de zoekvraag.

    **Verzadiging.** Bij TF-IDF groeit de waarde van een woord even hard als het aantal: tien keer *afvoer* weegt tien keer zo zwaar als één keer. BM25 laat het aantal verzadigen:

    $$\frac{(k_1 + 1) \cdot \text{tf}}{k_1 + \text{tf}}$$

    Bij een aantal van 1 is dit precies 1. Elke volgende herhaling voegt minder toe dan de vorige, en hoe vaak het woord ook voorkomt, de waarde blijft onder $k_1 + 1$: met $k_1 = 1{,}2$ geeft vier keer *afvoer* 1,69 in plaats van 4. Stephen Robertson en Hugo Zaragoza noemen dit in *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) saturation, in het Nederlands verzadiging. Hoe lager $k_1$, hoe sterker de verzadiging. Volgens het boek geeft $k_1 = 0$ een binair model, zonder term frequency: een woord voegt dan één keer de IDF toe, hoe vaak het er ook in staat. Een grote $k_1$ laat bijna elke herhaling weer helemaal meetellen.

    **Lengte.** Robertson en Zaragoza geven twee redenen waarom het ene document langer is dan het andere. De ene schrijver gebruikt meer woorden om hetzelfde te zeggen; de andere schrijver heeft meer te zeggen, en behandelt in één document meer onderwerpen. Bij de eerste soort moeten de aantallen omlaag met de lengte, bij de tweede niet. BM25 mengt de twee met een tweede instelling, $b$, van 0 tot 1, en vermenigvuldigt $k_1$ met

    $$(1 - b) + b \cdot \frac{L_d}{L_{\text{ave}}}$$

    waarin $L_d$ het aantal woorden van het document is en $L_{\text{ave}}$ het gemiddelde over de collectie. Een document van gemiddelde lengte krijgt 1, en dezelfde verzadiging als hierboven. Een langer document krijgt meer dan 1, dus een woord moet vaker voorkomen voor dezelfde waarde, en een korter document krijgt minder dan 1. Volgens het boek schaalt $b = 1$ de waarde helemaal met de lengte van het document, en doet $b = 0$ niets met de lengte. Anders dan de cosine similarity deelt BM25 de score niet door een lengte: de lengte bepaalt alleen hoe snel de aantallen verzadigen, en een woord voegt nog steeds minder dan $k_1 + 1$ keer de IDF toe.

    **De hele score.** Samen is de BM25-score van een document $d$ voor een zoekvraag $q$

    $$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}_t \cdot \frac{(k_1 + 1) \cdot \text{tf}_{t,d}}{k_1 \cdot \left((1 - b) + b \cdot \frac{L_d}{L_{\text{ave}}}\right) + \text{tf}_{t,d}}$$

    waarin $\text{tf}_{t,d}$ het aantal keer is dat woord $t$ in document $d$ staat; een woord dat niet in $d$ staat, voegt niets toe. Dit is de formule van het boek. Het boek voegt een derde instelling toe, $k_3$, voor woorden die meer dan één keer in een lange zoekvraag staan; de vragen hier zijn kort, en deze les telt elk woord van de zoekvraag één keer.

    **De instellingen.** Het boek raadt aan de instellingen af te stellen op een testcollectie die apart is gehouden om mee te ontwikkelen, en noemt, zonder zo'n afstelling, $k_1$ tussen 1,2 en 2 en $b = 0{,}75$ redelijke waarden. Deze les gebruikt $k_1 = 1{,}2$ en $b = 0{,}75$. Op de alinea's helpt elk van de twee instellingen: met $k_1 = 0$ zakt het deel van de vragen met de eigen alinea bovenaan van 73,5% naar 70,9%, en met $b = 0$ naar 70,1%. Met $k_1 = 5$, waarbij de herhalingen weer bijna helemaal meetellen, zakt het naar 67,8%. De winst ten opzichte van de cosine similarity is het grootst bij de vragen met de eigen alinea in de langste groep: van 54,2% naar 69,8%, en van 65,4% naar 74,8% bij de andere vragen.

    Bij de vraag over de gletsjers zet BM25 de eigen alinea bovenaan: 8,22 tegen 4,47 voor de alinea over de huidige afvoer van de Rijn. In deze alinea voegen de vier keer *afvoer* 4,39 toe, waar de TF-IDF-waarde 10,15 was. Het plaatje laat de vraag zien in beide rangschikkingen, wat elk woord van de vraag toevoegt in de twee alinea's, en alle vragen op zes manieren gerangschikt.
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

    | Woord | Betekenis |
    |---|---|
    | **BM25** | een rangschikking die voor elk woord van de zoekvraag dat in een document staat, de IDF maal een waarde optelt; de waarde verzadigt met het aantal en wordt geschaald met de lengte van het document |
    | **verzadiging** | het verschijnsel dat de waarde van een woord met elke herhaling minder groeit, en onder $k_1 + 1$ blijft; in het Engels saturation |
    | **term weight** | wat één woord in één document toevoegt, zonder de IDF: het aantal, verzadigd en geschaald met de lengte van het document |
    | $k_1$ | de instelling voor hoe sterk de aantallen verzadigen: bij 0 telt elk woord één keer, en hoe hoger, hoe meer de herhalingen meetellen |
    | $b$ | de instelling voor hoeveel de lengte van een document meetelt ten opzichte van het gemiddelde: bij 0 niet, bij 1 helemaal |
    | $L_d$, $L_{\text{ave}}$ | het aantal woorden van document $d$, en het gemiddelde aantal woorden van de documenten van de collectie |
    | **afstellen** | instellingen zoals $k_1$ en $b$ kiezen door de instellingen uit te proberen op zoekvragen waarvan de relevante documenten bekend zijn, en de uitkomst daarna meten op andere zoekvragen |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo rangschik je documenten voor een zoekvraag met BM25:

    1. Tel de woorden van elk document, en neem het gemiddelde aantal woorden over de collectie.
    2. Geef elk woord de IDF, log10(N / df).
    3. Neem voor elk woord van de zoekvraag dat in een document staat, één keer, het aantal en laat het verzadigen met $k_1$, waarbij $b$ bepaalt hoeveel de lengte van het document ten opzichte van het gemiddelde $k_1$ schaalt; vermenigvuldig met de IDF.
    4. Tel deze delen op: de BM25-score van het document.
    5. Sorteer de documenten op de score, de hoogste bovenaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.SMALL_IDF` bevat de IDF van elk woord van de dertien documenten van de course, `C.SMALL_AVERAGE` het gemiddelde aantal woorden, en `C.DOCS` de documenten als lijsten van woorden. `C.saturation` geeft de verzadiging van een aantal, `C.term_weight` de waarde van een woord met een aantal in een document met een lengte, en `C.bm25` de BM25-score van een document; hier steeds met $k_1 = 1{,}2$ en $b = 0{,}75$. De zoekvraag is *rode wijn hart*:
    """)
    return


@app.cell
def example_cell():
    _idf = C.SMALL_IDF[L.lang]
    _average = C.SMALL_AVERAGE[L.lang]
    _query = C.QUERIES[L.lang]["wine"]
    print(round(_average, 3), [round(C.saturation(_count, 1.2), 3) for _count in (1, 2, 3, 4)])
    for _doc in (6, 7, 8):
        _words = C.DOCS[L.lang][_doc]
        _counts = [_words.count(_word) for _word in _query]
        _parts = [_idf[_word] * C.term_weight(_words.count(_word), len(_words), _average, 1.2, 0.75)
                  for _word in _query]
        _score = C.bm25(_query, _words, _idf, _average, 1.2, 0.75)
        print(_doc, len(_words), _counts, [round(_part, 3) for _part in _parts], round(_score, 3))
    _twice = C.DOCS[L.lang][6] * 2
    print(round(C.bm25(_query, _twice, _idf, _average, 1.2, 0.75), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De documenten hebben gemiddeld 15,538 woorden, en de eerste regel laat de verzadiging van de aantallen 1 tot en met 4 zien: drie keer een woord geeft 1,571, niet 3. Elke volgende regel geeft een document, het aantal woorden, hoe vaak *rode*, *wijn* en *hart* erin staan, wat elk van deze woorden toevoegt, en de BM25-score. Document 6 bevat *wijn* drie keer, en is met 23 woorden langer dan gemiddeld: *wijn* voegt daar 0,729 toe, tegen 0,506 in document 7, waar het woord één keer staat. De TF-IDF-score van *TF-IDF: woorden wegen* gaf document 6 een score van 2,684 en document 7 een score van 1,661; BM25 geeft 1,689 tegen 1,641. De documenten 7 en 8 bevatten elk woord één keer, en document 8 is langer, 20 woorden tegen 16, dus elk woord voegt daar minder toe. De laatste regel is document 6 twee keer achter elkaar: de score groeit van 1,689 naar slechts 1,772, waar de TF-IDF-score zou verdubbelen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_saturation():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. saturation: elke herhaling voegt minder toe

        Schrijf `saturation`, die `tf` krijgt, het aantal keer dat een woord in een document staat, en `k1`, de instelling. De functie geeft het verzadigde aantal uit *Het idee* terug: het aantal maal $k_1 + 1$, gedeeld door $k_1$ plus het aantal. Met alleen `+`, `*` en `/` werkt de functie ook als `tf` een NumPy-array met aantallen is. Vervang de `...` door de formule.

        De check roept `saturation(1, 1.2)` aan, wat `1.0` geeft, en `saturation(3, 2.0)`, wat `1.8` geeft. Daarna eigen gevallen: andere aantallen en instellingen, een instelling van 0, een aantal van 0, en een NumPy-array met aantallen.
        """),
        L.task("Geef het verzadigde aantal: het aantal maal k1 + 1, gedeeld door k1 plus het aantal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_saturation():
    box_saturation = L.code_box("saturation", lines=3)
    box_saturation
    return (box_saturation,)


@app.cell(hide_code=True)
def check_saturation(box_saturation):
    result_saturation = L.check("saturation", box_saturation.value)
    result_saturation
    return (result_saturation,)


@app.cell(hide_code=True)
def hints_saturation():
    L.hints(
        "saturation",
        [
            "De formule heeft een deel boven de streep, `tf * (k1 + 1)`, en een deel eronder: het aantal `tf` plus de instelling `k1`.",
            "Deel door de twee samen: zet het aantal en de instelling tussen haakjes, zodat Python eerst optelt en dan deelt.",
            "`    return tf * (k1 + 1) / (___)`",
        ],
        why="`return tf * (k1 + 1) / (tf + k1)`: een aantal van 1 geeft precies 1, en hoe groot het aantal ook is, de uitkomst blijft onder k1 + 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_saturation():
    mo.md(r"""
    In het plaatje voegt een woord dat 20 keer voorkomt met $k_1 = 1{,}2$ een waarde van 2,075 toe, tegen 1 bij één keer: het aantal zelf zou 20 zeggen. Met $k_1 = 0{,}5$ groeit de waarde na een paar herhalingen nauwelijks nog, en met $k_1 = 2{,}0$ blijft de waarde langer groeien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_term_weight():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. term_weight: de lengte van het document

        Schrijf `term_weight`, die `tf` krijgt, het aantal keer dat een woord in een document staat, `length`, het aantal woorden van het document, `average`, het gemiddelde aantal woorden van de documenten, en de twee instellingen `k1` en `b`. De functie geeft de waarde van het woord terug: de verzadiging van oefening 1, met $k_1$ vermenigvuldigd met de schaal uit *Het idee*, waarin 1 meetelt met het deel `1 - b` en de lengte ten opzichte van het gemiddelde met het deel `b`. De startcode noemt dat getal `scale`, en de regel met `return` bevat de verzadiging al, met `k1 * scale`. Vervang de `...` door de schaal. Met alleen `+`, `-`, `*` en `/` werkt de functie ook als `tf` en `length` NumPy-arrays zijn: het plaatje hieronder roept de functie één keer aan, voor elk woord in elke alinea.

        De check roept `term_weight(2, 20, 20.0, 2.0, 0.75)` aan, een document van gemiddelde lengte, wat `1.5` geeft, en `term_weight(2, 60, 20.0, 2.0, 0.5)`, een document dat drie keer zo lang is, wat `1.0` geeft. Daarna eigen gevallen: b = 0, b = 1, k1 = 0, een kort document, en NumPy-arrays.
        """),
        L.task("Schaal k1 met een mengsel van 1 en de lengte ten opzichte van het gemiddelde, waarin de lengte het deel b heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_term_weight():
    box_term_weight = L.code_box("term_weight", lines=4)
    box_term_weight
    return (box_term_weight,)


@app.cell(hide_code=True)
def check_term_weight(box_term_weight):
    result_term_weight = L.check("term_weight", box_term_weight.value)
    result_term_weight
    return (result_term_weight,)


@app.cell(hide_code=True)
def hints_term_weight():
    L.hints(
        "term_weight",
        [
            "`scale` is een mengsel: 1 telt mee met het deel `1 - b`, en de lengte van het document ten opzichte van het gemiddelde, `length / average`, met het deel `b`.",
            "Het tweede deel van het mengsel is de lengte gedeeld door het gemiddelde, vermenigvuldigd met het eigen deel, de instelling b.",
            "`    scale = 1 - b + ___`",
        ],
        why="`scale = 1 - b + b * length / average`: een document van gemiddelde lengte krijgt 1, wat b ook is; bij b = 0 telt de lengte niet mee, bij b = 1 helemaal.",
    )
    return


@app.cell(hide_code=True)
def text_after_term_weight():
    mo.md(r"""
    In de voorbeelden van de check krijgt het document van gemiddelde lengte de schaal 1 en de verzadiging uit oefening 1: 1,5; het document dat drie keer zo lang is, krijgt de schaal 2, en het woord, dat er twee keer in staat, voegt slechts 1,0 toe. In het plaatje voegt een woord dat één keer voorkomt 1,277 toe in een alinea van 60 woorden, 1,0 in een alinea van gemiddelde lengte, en 0,719 in een alinea van 250 woorden. Met de term weights van de course zet BM25 de eigen alinea bovenaan bij 73,5% van de vragen; het plaatje laat zien wat de jouwe geven.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bm25():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. bm25: de score van een document

        Schrijf `bm25`, die `query` krijgt, de woorden van een zoekvraag als een lijst waarin elk woord één keer staat, `doc`, een document als lijst van woorden, `idf`, een dictionary met de IDF van elk woord, `average`, het gemiddelde aantal woorden van de documenten, en de instellingen `k1` en `b`. De functie geeft de BM25-score van het document terug. In de startcode begint `score` bij 0, en de loop gaat langs elk `word` van `query`; als `doc` het woord bevat, telt het woord het eigen deel op bij `score`. Vervang de `...` door de nieuwe `score`: de score tot nu toe, plus de IDF van het woord maal de term weight. De functie `term_weight` van oefening 2 geeft deze waarde bij het aantal keer dat het woord in het document staat, het aantal woorden van het document, het gemiddelde en de twee instellingen, in deze volgorde; het vak kent de functie al. `doc.count(word)` geeft hoe vaak `word` in `doc` staat, en `len(doc)` het aantal woorden van `doc`.

        De check roept `bm25(["red", "wine"], ["red", "wine", "wine", "cheese"], {"red": 0.5, "wine": 1.0, "cheese": 2.0}, 4.0, 1.2, 0.75)` aan, een document van gemiddelde lengte, wat `1.875` geeft, en `bm25(["python", "snake"], ["python", "tank"], {"python": 1.0, "snake": 2.0, "tank": 1.5}, 2.0, 1.2, 0.75)`, wat `1.0` geeft: *snake* staat niet in het document. Daarna eigen gevallen: een lege zoekvraag, k1 = 0, b = 0, b = 1, een woord met IDF 0, en de zoekvraag *red wine heart* met de documenten 6 en 9 van de Engelse les, en *rode wijn hart* met document 6 van de course.
        """),
        L.task("Tel voor elk woord van de zoekvraag dat in het document staat, de IDF maal de term weight op."),
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
            "Elke ronde telt het deel van `word` op bij `score`: de IDF, `idf[word]`, maal wat `term_weight` geeft voor dit woord in `doc`.",
            "`term_weight` krijgt vijf waarden: hoe vaak het woord in het document staat, het aantal woorden van het document, en dan `average`, `k1` en `b`.",
            "`            score = score + idf[word] * ___`",
        ],
        why="`score = score + idf[word] * term_weight(doc.count(word), len(doc), average, k1, b)`: elk woord van de zoekvraag dat in het document staat, voegt de IDF maal de term weight toe.",
    )
    return


@app.cell(hide_code=True)
def text_after_bm25():
    mo.md(r"""
    Bij de vraag over de gletsjers zet je functie de eigen alinea bovenaan, waar de cosine similarity de alinea op plaats twee zette. Bij de vraag over het parlement van Victoria zet je functie de eigen alinea ook bovenaan, net als de cosine similarity. In de Engelse les zet BM25 bij deze vraag een langere alinea over Victoria bovenaan, met *Victoria* vier keer en ook *what*, een woord van de vraag dat niets over het antwoord zegt: geen enkele rangschikking krijgt elke vraag goed.

    ### Probeer het: k1 en b afstellen op SciFact

    Volgens het boek stel je de instellingen het best af op een testcollectie die apart is gehouden om mee te ontwikkelen: zoekvragen waarvan de relevante documenten bekend zijn. SciFact, van David Wadden en anderen (2020), heeft zulke zoekvragen: wetenschappelijke beweringen, door experts geschreven, met de abstracts die de bewering steunen of tegenspreken. Elke bewering is geschreven bij een zin uit een paper dat naar andere papers verwijst. In de kopie die BEIR gebruikt, is de collectie 5.183 abstracts van biomedische papers, zijn de zoekvragen de beweringen, en zijn de abstracts waar de zin naar verwijst de relevante abstracts van de bewering. De beweringen zijn verdeeld over twee sets: 809 trainingsbeweringen om op af te stellen, en 300 testbeweringen om op te meten. SciFact is Engels, dus de demo werkt in het Engels. De abstracts hebben gemiddeld 223,2 woorden, van 38 tot 1.575.

    Stel alleen af op de trainingsbeweringen. Een instelling die je kiest door naar de testbeweringen te kijken, lijkt op de testbeweringen beter dan op nieuwe beweringen. Robertson en Zaragoza waarschuwen bovendien dat afgestelde instellingen bij de eigen collectie horen, en bij een andere collectie misschien slecht werken.

    Download SciFact, en verschuif dan de sliders. Zoek eerst een instelling die bij meer trainingsbeweringen het relevante abstract bovenaan zet dan $k_1 = 1{,}2$ en $b = 0{,}75$; kijk daarna wat je instelling doet bij de testbeweringen. Het plaatje laat ook één testbewering zien, met de eerste drie abstracts bij jouw instelling.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciFact, 8,2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scifact_cell(get_button):
    sci_numbers = C.scifact_or_stop(get_button, {
        "ask": "*SciFact staat nog niet op je computer. De knop downloadt de set één keer van de kopie van BEIR op Hugging Face, 8,2 MB: de beweringen onder de licentie CC BY 4.0 en de abstracts onder ODC-By 1.0. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor 8,2 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    try_k1 = mo.ui.slider(0.0, 3.0, step=0.1, value=C.TRY_START[0], label="k1", show_value=True)
    try_b = mo.ui.slider(0.0, 1.0, step=0.05, value=C.TRY_START[1], label="b", show_value=True)
    mo.hstack([try_k1, try_b], justify="start", gap=2)
    return sci_numbers, try_b, try_k1


@app.cell(hide_code=True)
def try_look(sci_numbers, try_k1, try_b):
    mo.Html(C.try_picture(L.lang, try_k1.value, try_b.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Afgesteld op de trainingsbeweringen is de beste instelling $k_1 = 1{,}1$ en $b = 0{,}65$: 57,4% van de trainingsbeweringen, tegen 56,5% met de gebruikelijke instelling, en 53,7% van de testbeweringen, tegen 53,3%. Afstellen levert hier weinig op, omdat de gebruikelijke instelling al dichtbij zat. De beste instelling op de testbeweringen zelf, $k_1 = 0{,}7$ en $b = 1{,}0$, geeft daar 56,0%: gekozen door naar de test te kijken, belooft deze instelling meer dan afstellen op de trainingsbeweringen oplevert. Elke instelling van de sliders verslaat de cosine similarity, die het relevante abstract bovenaan zet bij 44,7% van de testbeweringen.

    ### Oefening 4. Wat doet k1, en wat doet b?

    Deel elke uitspraak hieronder in: deze gaat over de instelling $k_1$, of over de instelling $b$.
    """)
    return


@app.cell(hide_code=True)
def exercise_settings():
    sort_settings = L.sort("settings")
    sort_settings
    return (sort_settings,)


@app.cell(hide_code=True)
def check_settings(sort_settings):
    result_settings = L.check("settings", sort_settings)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** BM25 gebruikt de IDF van [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) en de postings lists van [De inverted index](/course/open/ir-inverted-index?lang=nl), en waar [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl) door de lengte van een vector deelde, laat BM25 de lengte bepalen hoe snel de aantallen verzadigen. In het domein *Bouwen met AI* gebruikt [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl) BM25 om de teksten te vinden waaruit een assistent antwoordt. De volgende les, *Zoekmachines voor één domein*, rangschikt documenten met velden, zoals een titel en een prijs; en *Werkte het zoeken? P@k, MAP en MRR* meet een rangschikking op SciFact op meer manieren dan alleen de eerste plaats.

    **Een voorbeeld uit het dagelijks leven.** Een kok proeft de soep en doet er zout bij: het eerste snufje maakt veel verschil, het vijfde bijna niets, en een grote pan heeft meer zout nodig dan een klein kommetje voor dezelfde smaak. BM25 behandelt de herhalingen van een woord als de snufjes, en de lengte van een document als de grootte van de pan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_saturation, result_term_weight, result_bm25, result_settings):
    L.wrap_up(result_saturation, result_term_weight, result_bm25, result_settings)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
