# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-ndcg. Route: course/route/lessen/taal-en-zoeken.yaml. Controles: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-ndcg", "nl")
    C = L.checks   # de dertien documenten van de course met cijfers, de rangschikkingen van XQuAD en SciFact, en de controles


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl) en [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ap():
    quick_ap = L.choice(
        "quick-ap",
        question="Een zoekvraag heeft twee relevante documenten, en een rangschikking zet deze op de plaatsen 1 en 4. Wat zijn de reciprocal rank en de average precision van deze rangschikking?",
        options={
            "last": "RR 0,25 en AP 0,75",
            "mean": "RR 1 en AP 0,625, het gemiddelde van 1 gedeeld door elk van de twee plaatsen",
            "right": "RR 1 en AP 0,75, uit de precisions 1 en 0,5",
            "added": "RR 1 en AP 1,5, de precisions 1 en 0,5 opgeteld",
        },
        correct="right",
        why={
            "right": "De reciprocal rank kijkt alleen naar het eerste relevante document: 1 / 1 = 1. De average precision neemt de precision bij elk relevant document, 1 op plaats 1 en 2 / 4 = 0,5 op plaats 4, en deelt door de 2 relevante documenten: 0,75. Beide maten tellen de twee documenten als even relevant; in deze les kan het ene document zwaarder meetellen dan het andere.",
            "last": "Naar welk relevant document kijkt de reciprocal rank: het eerste of het laatste? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
            "mean": "Op plaats 4 is de precision niet 1 gedeeld door 4: hoeveel van de eerste vier documenten zijn relevant? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
            "added": "De precisions bij de twee relevante documenten zijn 1 en 0,5, maar waardoor deelt de average precision? Zie [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl).",
        },
    )
    quick_ap
    return (quick_ap,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Wat geven `np.log2(4)` en `np.log2(1)`?",
        options={
            "power": "16 en 2: 2 tot de macht 4, en 2 tot de macht 1",
            "right": "2 en 0: 2 tot de macht 2 is 4, en 2 tot de macht 0 is 1",
            "one": "2 en 1",
            "natural": "1,386 en 0: de log met grondtal e van 4, en de log met grondtal e van 1, zoals `np.log` deze geeft",
        },
        correct="right",
        why={
            "right": "`np.log2(x)` vraagt: 2 tot welke macht geeft x? 2 tot de macht 2 is 4, en 2 tot de macht 0 is 1. Deze les deelt de waarde van een document door zo'n log: op plaats 1 door de log met grondtal 2 van 2, en dat is 1, en lager in de lijst door meer.",
            "power": "Dat zijn 2 tot de macht 4 en 2 tot de macht 1. De log werkt andersom: 2 tot welke macht geeft 4? Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "one": "2 tot de macht 2 is 4: goed. En welke macht van 2 geeft 1? Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "natural": "`np.log2` neemt de log met grondtal 2, niet e: 2 tot welke macht geeft 4? Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl) was een document relevant of niet. Bij de behoefte over rode wijn en het hart waren de documenten 6 en 9 relevant, en het ene document even relevant als het andere. Toch beantwoordt document 6 de vraag rechtstreeks: het vergelijkt rode en witte wijn voor de kans op een hartaanval. Document 9 vergelijkt rood en wit ook, maar naast bier, in een overzicht over alcohol in het algemeen. En document 7, over eten voor een gezond hart met af en toe een glas rode wijn, beantwoordt de vraag niet, maar staat dichter bij de vraag dan document 8, over een wandeling door een oude stad. Gemeten als relevant of niet krijgen de volgorde 6, 7, 8, 10, 9 en de volgorde 9, 7, 8, 10, 6 dezelfde average precision: 0,7.

    Kalervo Järvelin en Jaana Kekäläinen schreven in 2002 dat documenten niet allemaal even relevant zijn voor de gebruikers, en dat een zoeksysteem de zeer relevante documenten moet herkennen en bovenaan moet zetten. nDCG, de maat uit dit paper, gaat uit van een cijfer voor elk document, telt de cijfers van een rangschikking van boven naar beneden op, elk iets minder naarmate het lager staat, en vergelijkt de uitkomst met de beste volgorde die er bestaat. Veel benchmarks voor zoeken melden deze maat: de Deep Learning-track van TREC in 2019 mat met nDCG@10, de benchmark BEIR doet dat op 18 testcollecties, en de retrieval-score van MTEB in [Een model voor zinnen kiezen: MTEB lezen](/course/open/nlp-mteb?lang=nl) is ook nDCG@10.

    **Een voorbeeld uit het dagelijks leven.** Je vraagt in een vreemde stad aan drie mensen de weg naar het station. De eerste legt de weg uit, de tweede zegt dat het station ergens bij de markt ligt, en de derde praat over het weer. Geteld als behulpzaam of niet zijn de eerste twee even goed; toch wil je het volledige antwoord als eerste, en het halve antwoord nog altijd eerder dan het weer.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_place = L.guess(
        "guess",
        question="nDCG telt een relevant document minder mee naarmate het lager staat, maar minder steil dan de reciprocal rank: deze geeft plaats 10 een tiende van wat plaats 1 krijgt. Hoeveel geeft nDCG een document op plaats 10, vergeleken met hetzelfde document op plaats 1?",
        options={
            "tenth": "een tiende, zoals bij de reciprocal rank",
            "right": "iets minder dan een derde",
            "half": "de helft",
            "most": "negen tiende: lagere plaatsen tellen bijna even zwaar",
        },
        reveal="Iets minder dan een derde: 0,289. nDCG deelt de waarde van het document op plaats 10 door de log met grondtal 2 van 11, en dat is 3,459, en op plaats 1 door de log met grondtal 2 van 2, en dat is 1. Järvelin en Kekäläinen kozen de log omdat wie lang blijft zoeken, ook de lagere plaatsen bereikt. Met de log met grondtal 2 krijgt volgens het tweetal een document op plaats 1024 nog een tiende van de volle waarde.",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Relevantie in gradaties.** In [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl) zei een relevance judgment relevant of niet relevant. Bij relevantie in gradaties geeft de relevance judgment elk document een cijfer. Järvelin en Kekäläinen gaan ervan uit dat zeer relevante documenten meer waard zijn dan documenten die nauwelijks relevant zijn. Bij de Deep Learning-track van TREC in 2019, georganiseerd door Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos en Ellen Voorhees, gaven beoordelaars van NIST passages uit webpagina's een van vier cijfers:

    | Cijfer | Bij TREC Deep Learning 2019 |
    |---|---|
    | 3 | perfect relevant: de passage gaat helemaal over de zoekvraag en bevat precies het antwoord |
    | 2 | zeer relevant: de passage geeft een antwoord op de zoekvraag, maar het antwoord is misschien wat onduidelijk, of verstopt tussen andere informatie |
    | 1 | verwant: de passage lijkt met de zoekvraag te maken te hebben, maar beantwoordt de zoekvraag niet |
    | 0 | irrelevant: de passage heeft niets met de zoekvraag te maken |

    Deze les geeft de dertien documenten van de course cijfers op deze schaal. Bij behoefte 3, rode wijn en het hart, krijgt document 6 een 3, document 9 een 2, document 7 een 1, en de andere documenten een 0. Bij behoefte 1, een python als huisdier kopen, krijgt document 1, *Koningspythons te koop*, een 3, document 3, over een warm terrarium en waar je er een koopt, een 2, en document 5, over een python thuis verzorgen, een 1. Bij behoefte 2, Python leren, krijgt document 0, *Python voor beginners*, een 3, en document 4, over een nieuwe versie van Python, een 1. De documenten met cijfer 2 of 3 zijn de relevante documenten van [Zoeken: informatiebehoefte en relevantie](/course/open/ir-what-is-search?lang=nl).

    **Gain.** Järvelin en Kekäläinen maken van de gerangschikte documenten een lijst met waarden, door elk document te vervangen door de relevantiescore. Elk document van de rangschikking krijgt zo een gain, hier het cijfer, en een document zonder cijfer krijgt 0. De gains van de eerste k plaatsen opgeteld vormen de cumulative gain (Järvelin en Kekäläinen schrijven cumulated gain). Deze groeit met elk document met een cijfer, maar het maakt niet uit waar bij de eerste k zo'n document staat.

    **Discount.** Volgens Järvelin en Kekäläinen is een relevant document minder waard naarmate het lager staat, omdat de kans kleiner is dat de gebruiker het document ooit bekijkt. De gain op elke plaats wordt dus gedeeld door een discount, en de discount groeit met de plaats. Het tweetal koos de log van de plaats: een discount die de score geleidelijk verlaagt naarmate de plaats lager wordt, maar niet zo steil als delen door de plaats, omdat sommige gebruikers volhouden en verder kijken. De gangbare vorm deelt de gain op plaats i door log₂(i + 1): plaats 1 door 1, plaats 2 door 1,585, plaats 3 door 2, en plaats 10 door 3,459. Zo rekent trec_eval, het programma waarmee TREC en BEIR de getallen berekenen. In het paper van 2002 wordt de gain op plaats i, met grondtal 2 zoals in de voorbeelden, vanaf plaats 2 gedeeld door log₂ i, dus tellen de eerste twee plaatsen allebei volledig mee.

    **DCG.** De gains van de eerste k plaatsen, elk gedeeld door de eigen discount en dan opgeteld, vormen de discounted cumulative gain, DCG@k. Bij behoefte 3 geeft BM25 de documenten 6, 7, 8 en 10, met de gains 3, 1, 0 en 0: DCG@5 = 3 / 1 + 1 / 1,585 = 3,631.

    **De ideale volgorde.** Is 3,631 goed? Dat hangt af van de cijfers van de zoekvraag. Järvelin en Kekäläinen vergelijken met de beste volgorde die er bestaat: alle documenten met een cijfer bij de zoekvraag, het hoogste cijfer eerst. Het tweetal benadrukt dat deze ideale volgorde uitgaat van alle bekende relevante documenten van het onderwerp, en niet van de uitkomst van een zoeksysteem: de ideale volgorde bevat elk document met een cijfer, ook als geen enkele rangschikking het document vindt. De DCG@k van deze volgorde is de ideale DCG, IDCG@k. Bij behoefte 3 is de ideale volgorde 3, 2, 1: IDCG@5 = 3 / 1 + 2 / 1,585 + 1 / 2 = 4,762.

    **nDCG.** De DCG gedeeld door de ideale DCG is de normalized DCG, nDCG@k: 1 voor een rangschikking in de ideale volgorde, en 0 voor een rangschikking zonder document met een cijfer bij de eerste k. Bij behoefte 3: 3,631 / 4,762 = 0,763. Omdat elke zoekvraag een getal van 0 tot 1 krijgt, kunnen zoekvragen met veel en met weinig relevante documenten samen één gemiddelde vormen. De Deep Learning-track van TREC nam nDCG@10 als belangrijkste maat, omdat deze maat de vier niveaus van de beoordelingen gebruikt en kijkt naar de eerste resultaten, de resultaten die gebruikers zien.

    **De gain van een cijfer.** trec_eval neemt het cijfer zelf als gain: volgens de handleiding zijn de gains de relevantiewaarden in het bestand met de relevance judgments. Chris Burges en collega's bij Microsoft leerden in 2005 een model de resultaten van een webzoekmachine te rangschikken, op beoordelingen van 1 tot 5. Als gain namen Burges en collega's 2 tot de macht de beoordeling, min 1: een perfect resultaat telt dan veel zwaarder mee dan een redelijk resultaat. Beide heten nDCG, dus een paper hoort te zeggen welke gain het gebruikt. Deze les neemt het cijfer zelf, zoals trec_eval en BEIR; bij Probeer het kun je wisselen.

    **Eén relevant document.** Met één relevant document, met cijfer 1, op plaats p is de ideale DCG 1, en is nDCG@k gelijk aan 1 / log₂(p + 1), of 0 als p na k komt. De reciprocal rank geeft 1 / p: op plaats 2 geeft nDCG 0,631 waar de reciprocal rank 0,5 geeft, en op plaats 10 0,289 tegen 0,1.

    Voor een zoekvraag, met de gain $g_i$ op plaats $i$:

    $$\text{DCG@}k = \sum_{i=1}^{k} \frac{g_i}{\log_2(i + 1)} \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

    waarin IDCG@k de DCG@k is van de ideale volgorde van alle cijfers van de zoekvraag; nDCG@k is 0 als de IDCG@k 0 is. Het gemiddelde over de zoekvragen is de gemiddelde nDCG@k.

    **De drie behoeften.** Het plaatje laat de drie behoeften zien met de resultaten van BM25, elk met de gain, de discount en de gain gedeeld door de discount. Bij behoefte 1 mist BM25 document 1, met cijfer 3, en staan document 5 bovenaan en document 3 vierde: een DCG@5 van 1 / 1 + 2 / 2,322 = 1,861, tegen een ideale 4,762, dus een nDCG@5 van 0,391. Bij behoefte 2 staat document 0 tweede in plaats van eerste, en document 4, met cijfer 1, vijfde in plaats van tweede: een nDCG@5 van 0,628, waar de average precision 0,5 was. Bij behoefte 3 ontbreekt document 9: 0,763. De volgorde 6, 7, 8, 10, 9 zou 0,925 krijgen, en de volgorde 9, 7, 8, 10, 6, met dezelfde average precision, 0,796. Over de drie behoeften is de gemiddelde nDCG@5 0,594, tegen een MAP van 0,375.
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
    | **relevantie in gradaties** | relevance judgments met cijfers, zoals 0 tot 3, in plaats van alleen relevant of niet relevant |
    | **gain** | wat een document van een rangschikking waard is: hier het cijfer, of 0 zonder cijfer |
    | **cumulative gain** | de gains van de eerste k plaatsen, opgeteld |
    | **discount** | waardoor de gain op een plaats gedeeld wordt: log₂(plaats + 1), 1 op plaats 1 en 3,459 op plaats 10 |
    | **discounted cumulative gain**, DCG@k | de gains van de eerste k plaatsen, elk gedeeld door de eigen discount, opgeteld |
    | **ideale DCG**, IDCG@k | de DCG@k van de ideale volgorde: alle cijfers van de zoekvraag, het hoogste eerst |
    | **normalized DCG**, nDCG@k | de DCG@k gedeeld door de IDCG@k: een getal van 0 tot 1 |
    | `sorted(..., reverse=True)` | een gesorteerde lijst van hoog naar laag, waar `sorted` alleen van laag naar hoog gaat |
    | $g_i$ | de gain op plaats $i$ |
    | $\log_2$ | de log met grondtal 2: $\log_2 8 = 3$, omdat $2^3 = 8$ |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een rangschikking te meten met relevantie in gradaties:

    1. Geef elk document een cijfer voor de zoekvraag, zoals 0 tot 3; een document zonder cijfer krijgt 0.
    2. Maak van de rangschikking de gains: het cijfer van elk document, van boven naar beneden.
    3. DCG@k: deel de gain op plaats i door log₂(i + 1), en tel de eerste k op.
    4. Ideale DCG: zet alle cijfers van de zoekvraag van hoog naar laag, en neem de DCG@k van deze volgorde.
    5. nDCG@k: de DCG@k gedeeld door de ideale DCG, of 0 als de ideale DCG 0 is.
    6. Neem het gemiddelde over alle zoekvragen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.small_results(L.lang, need, how)` geeft de resultaten van de cosine similarity (`"cosine"`) of van BM25 (`"bm25"`) voor de zoekvraag van een behoefte, de beste eerst; `C.GRADES` bevat de cijfers van de course bij elke behoefte, en `C.judged(need)` alle cijfers van een behoefte. `C.gains` maakt van een rangschikking de gains, en `C.dcg` en `C.ndcg` zijn de maten van de course. `C.rankings`, `C.relevant` en `C.reciprocal_rank` komen uit [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl). Eerst de drie behoeften: de resultaten, de gains, de DCG@5 en de nDCG@5. Dan de gemiddelde nDCG@10 over alle vragen van XQuAD, waar de eigen alinea cijfer 1 heeft, naast de MRR:
    """)
    return


@app.cell
def example_cell():
    for _need in ("buy", "learn", "wine"):
        for _how in ("cosine", "bm25"):
            _results = C.small_results(L.lang, _need, _how)
            _gains = C.gains(_results, C.GRADES[_need])
            print(_need, _how, _results, _gains, round(C.dcg(_gains, 5), 3),
                  round(C.ndcg(_gains, C.judged(_need), 5), 3))
    for _how in ("cosine", "bm25"):
        _rankings = C.rankings(L.lang, _how)
        _relevants = C.relevant(L.lang)
        _n = len(_rankings)
        _ndcg = sum([C.ndcg(C.gains(_rankings[_q], {_relevants[_q][0]: 1}), [1], 10) for _q in range(_n)]) / _n
        _mrr = sum([C.reciprocal_rank(_rankings[_q], _relevants[_q]) for _q in range(_n)]) / _n
        print(_how, _n, round(_ndcg, 3), round(_mrr, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste zes regels zijn de drie behoeften, voor de cosine similarity en voor BM25. Bij behoefte 3 geven beide rangschikkingen dezelfde vier documenten, met de gains 3, 1, 0 en 0, en een nDCG@5 van 0,763: document 9 ontbreekt. Bij behoefte 1 krijgt de cosine similarity 0,42 en BM25 0,391: beide missen document 1, met cijfer 3, en beide zetten document 5, met cijfer 1, bovenaan, maar de cosine similarity zet document 3, met cijfer 2, derde, waar BM25 dit document vierde zet.

    De laatste twee regels zijn de gemiddelden over alle 1.183 vragen van XQuAD: een nDCG@10 van 0,83 voor BM25 en 0,764 voor de cosine similarity, elk iets boven de MRR, 0,804 en 0,724. Met één relevante alinea met cijfer 1 krijgt een tweede plaats 0,631 in plaats van de 0,5 van de reciprocal rank.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dcg():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. dcg: discounted cumulative gain

        Schrijf `dcg`, met als invoer `gains`, de gains van een rangschikking als lijst, van de bovenste plaats naar beneden, en `k`. De functie geeft de DCG@k terug: elk van de eerste `k` gains gedeeld door de eigen discount, en opgeteld. In de startcode begint `total` bij 0, houdt `top = gains[:k]` de eerste `k` gains, of alle gains als de lijst korter is, en gaat `for place in range(len(top))` door de plaatsen zoals Python telt, vanaf 0. Vervang de `...` door de discount van de plaats: `np.log2` van de plaats geteld vanaf 1, plus 1. De box kent `np` al.

        De controle roept `dcg([3, 2, 3, 0, 0, 1, 2, 2, 3, 0], 3)` aan, wat `5.762` geeft: 3 / 1 + 2 / 1,585 + 3 / 2, de eerste drie gains van het voorbeeld van Järvelin en Kekäläinen. En `dcg([1, 0, 3], 5)`, wat `2.5` geeft: de lijst bevat drie gains, en de functie telt wat er is. Daarna eigen gevallen: k = 1, een lege lijst, en de gains van de course.
        """),
        L.task("Deel elke gain door de log met grondtal 2 van de plaats geteld vanaf 1, plus 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dcg():
    box_dcg = L.code_box("dcg", lines=6)
    box_dcg
    return (box_dcg,)


@app.cell(hide_code=True)
def check_dcg(box_dcg):
    result_dcg = L.check("dcg", box_dcg.value)
    result_dcg
    return (result_dcg,)


@app.cell(hide_code=True)
def hints_dcg():
    L.hints(
        "dcg",
        [
            "Elke gain `top[place]` wordt gedeeld door de eigen discount voordat de gain bij `total` komt; de discount is `np.log2` van de plaats geteld vanaf 1, plus 1.",
            "`place` telt vanaf 0, dus de plaats geteld vanaf 1 is één meer dan `place`, en de log neem je van één meer dan dat: op de eerste plaats de log met grondtal 2 van 2, en dat is 1.",
            "`        total = total + top[place] / np.log2(___)`",
        ],
        why="`total = total + top[place] / np.log2(place + 2)`: de eerste plaats wordt gedeeld door de log met grondtal 2 van 2, en dat is 1, de tweede door 1,585, de tiende door 3,459.",
    )
    return


@app.cell(hide_code=True)
def text_after_dcg():
    mo.md(r"""
    In het plaatje geeft je functie een gain van 1 op plaats 2 de waarde 0,631, en op plaats 10 de waarde 0,289, waar de reciprocal rank 0,5 en 0,1 geeft: de log daalt snel over de eerste plaatsen, en daarna langzaam. Bij de drie behoeften is je DCG@5 van de resultaten van BM25 1,861, 2,280 en 3,631, en de DCG@5 van de ideale volgorden 4,762, 3,631 en 4,762. Behoefte 2 heeft geen document met cijfer 2, dus de ideale volgorde, 3 en 1, heeft een lagere DCG: een DCG alleen zegt weinig zolang je de DCG niet vergelijkt met de ideale DCG van de eigen zoekvraag.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ndcg():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ndcg: vergelijken met de ideale volgorde

        Schrijf `ndcg`, met als invoer `gains`, de gains van een rangschikking zoals in oefening 1, `judged`, een lijst met alle cijfers van de zoekvraag in de relevance judgments, in willekeurige volgorde, en `k`. De functie geeft de nDCG@k terug: de DCG@k van de gains, gedeeld door `best`, de DCG@k van de ideale volgorde. In de startcode geeft `if best == 0` een 0 terug bij een zoekvraag zonder document met cijfer 1 of hoger, en de laatste regel deelt. Vervang de `...` door de ideale volgorde: alle cijfers in `judged`, het hoogste eerst. De box kent `dcg` van oefening 1 al.

        De controle roept `ndcg([0, 3, 1], [2, 3, 1], 3)` aan, wat `0.502` geeft: de ideale volgorde 3, 2, 1 heeft een DCG@3 van 4,762, en de rangschikking krijgt 2,393. En `ndcg([1, 0], [3, 1], 2)`, wat `0.275` geeft: de rangschikking mist het document met cijfer 3, en de ideale volgorde zet dit document toch bovenaan. Daarna eigen gevallen: een zoekvraag zonder relevante documenten, een lege rangschikking, k = 1, en de gains van de course.
        """),
        L.task("Geef dcg alle cijfers van de zoekvraag, van hoog naar laag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ndcg():
    box_ndcg = L.code_box("ndcg", lines=5)
    box_ndcg
    return (box_ndcg,)


@app.cell(hide_code=True)
def check_ndcg(box_ndcg):
    result_ndcg = L.check("ndcg", box_ndcg.value)
    result_ndcg
    return (result_ndcg,)


@app.cell(hide_code=True)
def hints_ndcg():
    L.hints(
        "ndcg",
        [
            "`best` is de `dcg` van de ideale volgorde bij `k`: alle cijfers in `judged`, op volgorde gezet met `sorted`, het hoogste eerst.",
            "`sorted` zet het laagste eerst; met `reverse=True` na de lijst zet de functie het hoogste eerst: `sorted([1, 3, 2], reverse=True)` geeft `[3, 2, 1]`.",
            "`    best = dcg(sorted(___), k)`",
        ],
        why="`best = dcg(sorted(judged, reverse=True), k)`: de ideale volgorde bevat elk document met een cijfer bij de zoekvraag, ook de documenten die de rangschikking mist, dus alleen de beste rangschikking die er bestaat, krijgt 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_ndcg():
    mo.md(r"""
    In het plaatje meet je functie de cosine similarity en BM25 op drie testcollecties. Op de dertien documenten krijgt de cosine similarity een gemiddelde nDCG@5 van 0,603 en BM25 0,594, tegen een MAP van 0,389 en 0,375. Bij behoefte 2 zetten beide rangschikkingen document 0 tweede: een average precision van 0,5, maar een nDCG@5 van 0,628, want de log kost een tweede plaats minder dan de reciprocal rank. Op XQuAD ligt de nDCG@10 bij beide rangschikkingen iets boven de MRR. Met SciFact gedownload krijgt BM25 0,662 en de cosine similarity 0,580. BEIR meldt 0,665 voor de eigen BM25, met k1 = 0,9 en b = 0,4 en de titel als apart veld, waar de BM25 van de course k1 = 1,2 en b = 0,75 gebruikt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dcg_all():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. dcg_all: elke zoekvraag tegelijk

        Schrijf `dcg_all`, met als invoer `gains`, een NumPy-array met een rij gains voor elke zoekvraag en minstens `k` kolommen, en `k`. De functie geeft een array terug met de DCG@k van elke rij, zonder loop. In de startcode houdt `gains[:, :k]` de eerste `k` kolommen van elke rij, deelt `/ discounts` elke kolom door een eigen getal, zoals bij [Broadcasting](/course/open/np-broadcasting?lang=nl), en telt `.sum(axis=1)` elke rij op. Vervang de `...` door `discounts`: een array met de discount van elk van de eerste `k` plaatsen, `np.log2` van 2, 3 en zo verder tot en met `k + 1`.

        De controle roept `dcg_all(np.array([[3, 2, 3, 0], [1, 0, 3, 2]]), 3)` aan, wat `np.array([5.762, 2.5])` geeft: de eerste rij begint als het voorbeeld van oefening 1. En `dcg_all(np.array([[0, 1], [2, 0]]), 2)`, wat `np.array([0.631, 2.0])` geeft. Daarna eigen gevallen: k = 1, een k zo lang als de rijen, en de gains van XQuAD van de course.
        """),
        L.task("Maak de discounts van de eerste k plaatsen als één array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dcg_all():
    box_dcg_all = L.code_box("dcg_all", lines=4)
    box_dcg_all
    return (box_dcg_all,)


@app.cell(hide_code=True)
def check_dcg_all(box_dcg_all):
    result_dcg_all = L.check("dcg_all", box_dcg_all.value)
    result_dcg_all
    return (result_dcg_all,)


@app.cell(hide_code=True)
def hints_dcg_all():
    L.hints(
        "dcg_all",
        [
            "`discounts` bevat de discount van elk van de eerste `k` plaatsen: `np.log2` van de plaats geteld vanaf 1, plus 1, voor alle plaatsen tegelijk.",
            "`np.arange(a, b)` geeft de gehele getallen van `a` tot `b`, zonder `b` zelf: `np.arange(2, 5)` geeft 2, 3 en 4. `np.log2` van een array neemt de log van elk getal erin.",
            "`    discounts = np.log2(___)`",
        ],
        why="`discounts = np.log2(np.arange(2, k + 2))`: de logs van 2 tot en met k + 1, dus plaats 1 wordt gedeeld door 1 en plaats k door de log met grondtal 2 van k + 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_dcg_all():
    mo.md(r"""
    In het plaatje krijgt BM25 een gemiddelde nDCG@1 van 0,735, nDCG@3 van 0,805, nDCG@5 van 0,820 en nDCG@10 van 0,830; de cosine similarity krijgt 0,626, 0,725, 0,747 en 0,764. De nDCG@1 is de precision at 1: met één relevante alinea telt alleen de eerste plaats, en dan volledig. Bij elke grotere k tellen een paar eigen alinea's meer mee, elk voor minder. De perfecte rangschikking krijgt 1 bij elke k, omdat de DCG daar de ideale DCG is.

    ### Probeer het: geef zelf de cijfers

    Cijfers zijn oordelen, en mensen oordelen verschillend. Kies een behoefte, en geef elk document met de slider een cijfer van 0 tot 3; de sliders beginnen bij de cijfers van de course. Het plaatje laat de resultaten van de cosine similarity en van BM25 zien met jouw cijfers, de DCG@5 en nDCG@5, naast de nDCG@5 met de cijfers van de course en de average precision, die de cijfers 2 en 3 als relevant telt. Kies onder de behoefte de gain: het cijfer zelf, of 2 tot de macht het cijfer, min 1.

    Begin met behoefte 3, en geef document 9 ook een 3: waarom gaat de nDCG@5 van beide rangschikkingen omlaag, terwijl de rangschikkingen gelijk blijven? Zet daarna document 9 terug op 2, kies de andere gain, en vergelijk behoefte 3 met behoefte 1: waar gaat de nDCG@5 omhoog, en waar omlaag?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _needs = C.try_options(L.lang)
    try_need = mo.ui.dropdown(options=_needs, value=next(t for t, n in _needs.items() if n == C.TRY_NEED), label="Behoefte")
    try_gain = mo.ui.radio(options=C.GAIN_OPTIONS[L.lang], value=next(iter(C.GAIN_OPTIONS[L.lang])), label="Gain")
    mo.vstack([try_need, try_gain])
    return try_gain, try_need


@app.cell(hide_code=True)
def try_grades_cell(try_need):
    _sliders = C.try_sliders(L.lang, try_need.value)
    try_grades = mo.ui.array([mo.ui.slider(0, 3, step=1, value=v, show_value=True) for _t, v in _sliders])
    mo.vstack([mo.hstack([mo.md(_t), try_grades[_j]], widths=[3, 1], align="center")
               for _j, (_t, _v) in enumerate(_sliders)], gap=0.25).style({"max-width": "44rem"})
    return (try_grades,)


@app.cell(hide_code=True)
def try_look(try_gain, try_grades, try_need):
    mo.Html(C.try_picture(L.lang, try_need.value, try_grades.value, try_gain.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    De rangschikkingen vinden document 9 niet (er staat *hartaanvallen* en *rood*, niet *hart* en *rode*), dus met document 9 op 3 blijft de DCG@5 3,631, en wordt de ideale volgorde beter: 3, 3, 1 geeft een IDCG@5 van 5,393, dus de nDCG@5 daalt van 0,763 naar 0,673. Er horen nu twee documenten met cijfer 3 bovenaan, en de rangschikkingen missen er een. Met de gain 2 tot de macht het cijfer, min 1, telt een 3 voor 7 en een 1 voor 1. Bij behoefte 3, waar beide rangschikkingen document 6 bovenaan zetten, stijgt de nDCG@5 van 0,763 naar 0,812; bij behoefte 1, waar beide rangschikkingen document 1, met cijfer 3, missen, daalt de nDCG@5 van BM25 van 0,391 naar 0,244. De steilere gain beloont het vinden van het beste document meer, en straft het missen ervan meer af.

    Cijfers nemen niet elk zwak punt van de relevance judgments weg: een document dat niemand beoordeelde, telt als 0, ook als het zou helpen. Burges en collega's merken op dat een nDCG van 1 voor een zoekvraag, hier $N_i = 1$, onwaarschijnlijk is, zelfs voor een perfect zoeksysteem, omdat sommige documenten zonder beoordeling in werkelijkheid zeer relevant kunnen zijn.

    ### Oefening 4. Omhoog, omlaag of gelijk?

    Deel elke verandering hieronder in naar wat deze doet met de nDCG@10 van een zoekvraag, met het cijfer als gain.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rangschikkingen die je hier meet, komen uit [Rangschikken met de cosine similarity: het vector space model](/course/open/ir-vector-space?lang=nl) en [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl), en nDCG brengt de maten van [Werkte het zoeken? P@k, MAP en MRR](/course/open/ir-precision-at-k?lang=nl) van relevant of niet naar cijfers. De retrieval-score van MTEB in [Een model voor zinnen kiezen: MTEB lezen](/course/open/nlp-mteb?lang=nl) is de nDCG@10 van deze les, maal 100. De volgende les, *Snippets: de resultaten laten zien*, kijkt naar wat een pagina met resultaten van elk document laat zien; en *Van tien links naar één antwoord* kijkt naar wat er verandert als een zoekmachine één antwoord schrijft in plaats van een gerangschikte lijst.

    **Een voorbeeld uit het dagelijks leven.** Een afspeellijst zet je lievelingsnummer bovenaan, een nummer dat je een beetje leuk vindt wat lager, en nummers die je overslaat aan het eind. De gain is hoe leuk je elk nummer vindt, de discount zegt dat je misschien al eerder stopt met luisteren, en nDCG vergelijkt de afspeellijst met de beste volgorde van dezelfde nummers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_dcg, result_ndcg, result_dcg_all, result_changes):
    L.wrap_up(result_dcg, result_ndcg, result_dcg_all, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
