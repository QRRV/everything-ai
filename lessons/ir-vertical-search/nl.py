# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-vertical-search. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-vertical-search", "nl")
    C = L.checks   # de 24 advertenties van de course, de alinea's en vragen van SQuAD, en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_saturation():
    quick_saturation = L.choice(
        "quick-saturation",
        question="In *BM25: de klassieke rangschikking* heeft een woord dat één keer in een document van gemiddelde lengte staat, met k1 = 1,2 de term weight 1. Wat is de term weight als het woord er drie keer in staat?",
        options={
            "three": "3: drie keer zoveel als bij één keer",
            "right": "1,571: elke herhaling voegt minder toe dan de vorige",
            "k1": "1,2: de instelling k1 zelf",
            "limit": "2,2: het meeste dat een aantal van een woord kan toevoegen, k1 + 1",
        },
        correct="right",
        why={
            "right": "De verzadiging: 2,2 · 3 / (1,2 + 3) = 1,571. Deze les stelt een nieuwe vraag: als een woord in de titel van een document staat en ook in de tekst, moet het dan één keer verzadigen over allebei, of in elk veld apart?",
            "three": "Drie keer zoveel is het aantal zelf, zoals in een vector van TF-IDF-waarden. Wat deed BM25 met elke herhaling van een woord? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "k1": "k1 bepaalt hoe sterk de aantallen verzadigen, maar de term weight hangt ook af van het aantal. Wat geeft de verzadiging bij een aantal van 3? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "limit": "k1 + 1 is de grens waar de term weight dichtbij komt, maar nooit aan komt. Haalt een aantal van 3 de grens al? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
        },
    )
    quick_saturation
    return (quick_saturation,)


@app.cell(hide_code=True)
def quick_check_tune():
    quick_tune = L.choice(
        "quick-tune",
        question="In dezelfde les werden k1 en b afgesteld op de 809 trainingsbeweringen van SciFact, en gemeten op de 300 testbeweringen. Waarom niet afstellen op de testbeweringen?",
        options={
            "fewer": "de testbeweringen zijn te weinig om twee instellingen op af te stellen",
            "right": "een instelling gekozen op de testbeweringen lijkt daar beter dan op nieuwe beweringen",
            "same": "de beste instelling is op allebei gelijk, dus het maakt niet uit",
            "other": "de testbeweringen gaan over andere abstracts, dus de beste instelling daar zegt helemaal niets",
        },
        correct="right",
        why={
            "right": "Een instelling gekozen op de testbeweringen past bij deze beweringen: k1 = 0,7 en b = 1,0 gaf daar 56,0%, meer dan de 53,7% die afstellen op de trainingsbeweringen opleverde. Deze les voegt eigen instellingen toe, zoals hoeveel een titel meetelt, en voor deze instellingen geldt hetzelfde.",
            "fewer": "300 beweringen zijn ruim genoeg om 651 instellingen op uit te proberen. Wat gaat er mis als je een instelling meet op de beweringen waarmee je de instelling koos? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "same": "De beste instelling op de testbeweringen was k1 = 0,7 en b = 1,0, tegen k1 = 1,1 en b = 0,65 op de trainingsbeweringen. Welke van de twee belooft te veel? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
            "other": "Beide sets beweringen zoeken in dezelfde 5.183 abstracts. Wat gaat er mis als je een instelling meet op de beweringen waarmee je de instelling koos? Zie [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl).",
        },
    )
    quick_tune
    return (quick_tune,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op een site voor tweedehands spullen zoek je een tent: hoogstens € 100, niet te ver weg, en nog te koop. De site van de course heeft 24 advertenties. Elke advertentie heeft een titel, zoals *Tent voor vier personen*, een korte tekst, een prijs, het aantal dagen sinds de advertentie geplaatst is, en de afstand tot jou. Twaalf advertenties bevatten het woord *tent*. BM25 op de titel en de tekst als één lijst van woorden zet een slaapzak bovenaan, met twee keer *tent* in de tekst: *Warme slaapzak voor koude nachten in de tent. Rolt klein op, zodat er ruimte overblijft in de tent.* Daarna komen stalen haringen. De advertentie die je zoekt, *Tent voor vier personen*, bevat *tent* één keer, in de titel, en heeft een lange tekst: van de twaalf komt deze advertentie als laatste. Van de andere tenten kost er een € 140 en een € 220, staat er een op 140 km, en werd er een drie maanden geleden geplaatst.

    Een zoekmachine voor het hele web behandelt een pagina vooral als woorden en links. Een zoekmachine voor één domein kent de delen van de documenten: hier de titel, de prijs, de datum en de plek. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze plaatsen zulke zoekmachines in *Introduction to Information Retrieval* (2008) tussen het web en je eigen computer in: zoeken in de documenten van een bedrijf of een instelling, in een database met patenten, of in papers over biochemie. Zoekmachines voor nieuws, voor producten, voor plekken in de buurt, voor de documenten van een bedrijf en voor recht zijn zulke **verticale zoekmachines**: elk zoekt in één domein, en gebruikt wat de zoekmachine weet over de documenten van dat domein.

    **Een voorbeeld uit het dagelijks leven.** In de catalogus van een bibliotheek zoek je op titel, op schrijver of op jaar, en in een schoenenwinkel kies je een maat en sorteer je op prijs. Geen van de twee behandelt de spullen als een hoop woorden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_title = L.guess(
        "guess",
        question="Terug naar de 2.067 alinea's van XQuAD, waar BM25 de eigen alinea bovenaan zette bij 73,5% van de 1.183 vragen. Elke alinea hoort bij een pagina van Wikipedia, zoals *Warschau*. Stel dat elke alinea de titel van de pagina krijgt als extra veld, waarin elk woord twee keer telt. Bij welk deel van de vragen zet BM25 de eigen alinea nu bovenaan?",
        options={
            "less": "minder vaak dan zonder de titels",
            "right": "iets vaker: minder dan twee punten erbij",
            "more": "duidelijk vaker: ongeveer vijf punten erbij",
            "most": "bijna altijd: bij meer dan negen op de tien vragen",
        },
        reveal="Iets vaker: 75,1%, tegen 73,5%. Een pagina heeft gemiddeld 43 alinea's, en de titel voegt aan elk van deze alinea's dezelfde woorden toe: de titel kan hoogstens helpen de pagina te vinden, en de pagina vond BM25 al: bij 90,0% van de vragen stond de eerste alinea in de pagina van de eigen alinea. Binnen de pagina onderscheidt de titel geen enkele alinea, en slechts 28,1% van de vragen bevat een woord uit de titel van de eigen pagina. Een veld helpt het meest als het veld documenten van elkaar onderscheidt, zoals de titel van een advertentie.",
    )
    guess_title
    return (guess_title,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Velden.** Onder metadata verstaat het boek bepaalde soorten gegevens over een document, zoals de schrijvers, de titel en de datum van publicatie. Een deel met vrije tekst, zoals de titel of de samenvatting, heet in het boek een zone, en gegevens met een waarde, zoals de datum waarop het document is gemaakt, een veld. Zoekmachines zoals Elasticsearch noemen allebei een veld, en deze les ook. Een advertentie op de site heeft twee tekstvelden, de titel en de tekst, en drie velden met een waarde: de prijs, het aantal dagen sinds de advertentie geplaatst is, en de afstand.

    **De velden wegen.** Een treffer in de titel zegt meer dan een treffer in de tekst: een titel zegt in een paar woorden waar het document over gaat. Stephen Robertson en Hugo Zaragoza schrijven in *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) dat een treffer in de titel sterker op relevantie kan wijzen dan dezelfde treffer in de tekst. De voor de hand liggende manier is elk veld een eigen BM25-score te geven, en de scores op te tellen, elk maal de weging van het veld. Robertson en Zaragoza zijn daartegen: een woord verzadigt dan in elk veld apart, terwijl het voor het hele document geldt of het document over het woord gaat. BM25F, de variant van Robertson en Zaragoza, telt in de eenvoudigste vorm eerst de aantallen van een woord op, elk maal de weging van het veld, en laat pas dit totaal verzadigen: de verzadiging hoort volgens Robertson en Zaragoza pas bij het totaal van elk woord. Met $v_s$ de weging van veld $s$:

    $$\widetilde{\text{tf}}_{t,d} = \sum_s v_s \cdot \text{tf}_{t,s} \qquad \widetilde{L}_d = \sum_s v_s \cdot L_s$$

    waarin $\text{tf}_{t,s}$ het aantal keer is dat woord $t$ in veld $s$ van document $d$ staat, en $L_s$ het aantal woorden van het veld. Daarna werkt BM25 zoals in *BM25: de klassieke rangschikking*, met $\widetilde{\text{tf}}$ op de plek van het aantal, en $\widetilde{L}_d$ op de plek van de lengte, ten opzichte van het gemiddelde $\widetilde{L}_{\text{ave}}$ over de collectie. Met hele getallen als weging is dit BM25 op een document waarin elk veld zo vaak is uitgeschreven als de weging: Robertson en Zaragoza noemen het een gewone BM25-functie op een document waarin sommige streams herhaald zijn, waarbij stream het woord is voor een veld. De IDF komt uit hele documenten: Robertson en Zaragoza berekenden de IDF over de hele collectie, zonder naar de streams te kijken, en dat werkte in de praktijk goed.

    Het verschil zie je bij een woord dat één keer in een titel met weging 3 staat en twee keer in een tekst, in velden van gemiddelde lengte, met $k_1 = 1{,}2$. BM25F laat het gecombineerde aantal, 3 · 1 + 2 = 5, verzadigen tot 1,774, onder $k_1 + 1 = 2{,}2$. Een BM25-score per veld zou 3 · 1 + 1,375 = 4,375 geven, alsof de titel en de tekst elk apart over het woord gaan, en de som zou kunnen groeien tot 3 · 2,2 + 2,2 = 8,8, waar BM25F onder 2,2 blijft. De wegingen zijn instellingen, net als $k_1$ en $b$. De volledige BM25F van Robertson en Zaragoza geeft elk veld ook een eigen $b$; voor een van de webzoektaken van TREC in 2003 melden Robertson en Zaragoza afgestelde wegingen van 38,4 voor de titel, 1,0 voor de tekst en 35 voor de anchor text van de links naar een pagina, met $k_1 = 27{,}5$. Deze les houdt één $b$, en geeft de titel weging 3.

    **Filters.** Het boek geeft als voorbeeld een zoekvraag naar documenten van William Shakespeare uit 1601 met de zinsnede *alas poor Yorick*. Met een veld met een waarde houdt een zoekmachine alleen de documenten waarvan de waarde past, zoals een prijs van hoogstens € 100 en een afstand van hoogstens 50 km. Een filter verandert de score van een document niet: het filter houdt het document, of laat het afvallen.

    **Decay.** Een filter heeft een harde rand: een tent op 50 km blijft, een tent op 51 km valt af. Voor leeftijd en afstand past een zachte rand vaak beter: een advertentie van een week geleden telt nog mee, iets minder dan een van vandaag. Elasticsearch, een zoekmachine waar veel sites op draaien, heeft hier decay functions voor: functies die de score van een document kleiner laten worden naarmate een getal van het document verder van een gekozen beginpunt ligt, zoals een filter op een bereik, maar met zachte randen in plaats van harde. Deze les gebruikt de decay die de score halveert bij een afstand $h$:

    $$\text{decay}(x) = 0{,}5^{\,x / h}$$

    waarin $x$ de afstand tot de beste waarde is, zoals de leeftijd in dagen of de afstand in km. Bij $x = 0$ is de factor 1, bij $h$ is de factor 0,5, bij $2h$ 0,25. Elasticsearch schrijft de exponentiële decay met $e$, een instelling *decay* en een *scale*; met decay 0,5 en scale $h$ is het deze formule. De eindscore is de BM25F-score maal de decay van de dagen, met $h$ = 14 dagen, maal de decay van de km, met $h$ = 25 km. Ook Elasticsearch vermenigvuldigt de score van de zoekvraag met de score van de functies, tenzij je iets anders kiest. Robertson en Zaragoza noemen zulke gegevens ook: de leeftijd van een bestand, het soort bestand of het aantal links ernaartoe kan iets zeggen over de relevantie van een document.

    **Elk domein vraagt iets anders.** Nieuws gaat over wat er nu gebeurt, dus versheid weegt zwaar. Een winkel filtert op de velden van de producten, zoals een prijs en een maat. Bij plekken in de buurt bepaalt de afstand veel van de volgorde; Google schrijft dat de lokale resultaten vooral afhangen van relevantie, afstand en bekendheid. In een bedrijf is niet elk document voor iedereen bedoeld. Volgens het boek mogen documenten die de zoeker niet mag openen, niet in de resultaten staan, want alleen al het bestaan van een document kan gevoelige informatie zijn. En recht heeft eigen gewoontes. Het boek beschrijft Westlaw, een zoekdienst voor juristen, met lange, precieze zoekvragen en operatoren voor nabijheid, allebei ongewoon bij zoeken op het web. Een operator voor nabijheid vraagt om woorden in dezelfde zin, of binnen een aantal woorden van elkaar. Westlaw zet de documenten volgens het boek op datum, het nieuwste eerst, en dat werkt in de praktijk goed; en in 2007 leken de meeste juridische bibliothecarissen nog Booleaanse zoekvragen aan te raden om niets te missen.

    Het plaatje laat de zoekvraag *tent* zien in vier stappen: BM25 op de titel en de tekst als één lijst van woorden, de titel drie keer, de filters van hoogstens € 100 en hoogstens 50 km, en de decay. *Tent voor vier personen* klimt van de laatste plaats van de twaalf naar de vierde, gedeeld met de grote tent voor zes personen, daarna naar de tweede en de eerste.
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
    | **verticale zoekmachine** | een zoekmachine voor één domein, zoals nieuws, producten, plekken in de buurt, de documenten van een bedrijf of recht; de zoekmachine gebruikt wat deze weet over de documenten van dat domein |
    | **veld** | een deel van een document dat een zoekmachine apart kent: een tekst, zoals de titel, of een waarde, zoals een prijs of een datum |
    | **weging van een veld** | hoe vaak een woord in een veld meetelt, ten opzichte van een woord in een veld met weging 1 |
    | **BM25F** | BM25 voor documenten met velden: de aantallen van een woord in de velden worden opgeteld, elk maal de weging van het veld, voordat het totaal verzadigt |
    | **filter** | een voorwaarde op een veld, waarmee een document blijft of afvalt, zonder dat de score verandert |
    | **decay** | een factor van 1 omlaag naar 0 waarmee een score wordt vermenigvuldigd, kleiner naarmate een waarde verder van de beste waarde ligt, zoals de leeftijd van een document of de afstand |
    | $h$ | de afstand waarbij de decay een score halveert, zoals 14 dagen of 25 km |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je in documenten met velden, zoals een verticale zoekmachine doet:

    1. Verdeel elk document in velden: teksten, zoals de titel en de tekst, en waarden, zoals de prijs, de leeftijd en de afstand.
    2. Laat de documenten afvallen waarvan de waarden niet door de filters komen.
    3. Tel voor elk woord van de zoekvraag de aantallen in de tekstvelden op, elk maal de weging van het veld, en tel de lengtes van de velden op dezelfde manier op.
    4. Geef elk document de BM25-score met deze aantallen en lengtes, en met de IDF over hele documenten: BM25F.
    5. Vermenigvuldig de score met de decay van elke waarde die zacht moet meetellen, zoals de leeftijd en de afstand.
    6. Sorteer de documenten op de eindscore, de hoogste bovenaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.fields(L.lang, i)` geeft de tekstvelden van advertentie `i` als lijsten van woorden, en `C.listing(L.lang, i)` de hele advertentie, met de prijs, de dagen en de km. `C.combined_count` en `C.combined_length` geven het aantal en de lengte van BM25F, `C.term_weight` de term weight uit *BM25: de klassieke rangschikking*, en `C.decay` de decay. `C.average_length` geeft de gemiddelde gecombineerde lengte van de 24 advertenties, en `C.IDF` de IDF van elk woord. De zoekvraag is *tent*, en de titel telt drie keer:
    """)
    return


@app.cell
def example_cell():
    _weights = {"title": 3, "text": 1}
    _average = C.average_length(L.lang, _weights)
    _idf = C.IDF[L.lang]["tent"]
    print(round(_idf, 3), round(_average, 3))
    for _i in (0, 1):
        _fields = C.fields(L.lang, _i)
        _listing = C.listing(L.lang, _i)
        _count = C.combined_count("tent", _fields, _weights)
        _length = C.combined_length(_fields, _weights)
        _score = _idf * C.term_weight(_count, _length, _average, 1.2, 0.75)
        _final = _score * C.decay(_listing["days"], 14) * C.decay(_listing["km"], 25)
        print(_listing["title"], _count, _length, round(_score, 3), round(_final, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Tent* staat in 12 van de 24 advertenties, dus de IDF is log10(24 / 12) = 0,301, en met de titel drie keer heeft een advertentie gemiddeld 17,417 woorden. *Tent voor vier personen* bevat *tent* één keer, in de titel van 4 woorden, dus het aantal is 3 en de lengte 3 · 4 + 16 = 28: de score is 0,419. *Slaapzak* bevat *tent* twee keer, in de tekst van 18 woorden: aantal 2, lengte 3 · 1 + 18 = 21, score 0,391. Als één lijst van woorden won de slaapzak, met 0,357 tegen 0,237. Dan de decay: de tent werd 3 dagen geleden geplaatst en staat op 12 km, dus de score wordt vermenigvuldigd met 0,862 en met 0,717, tot 0,259; de slaapzak, 8 dagen oud en op 20 km, eindigt op 0,151.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_combined_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. combined_count: het aantal van BM25F

        Schrijf `combined_count`, die `word` krijgt, een woord van de zoekvraag, `fields`, een dictionary met de woorden van elk tekstveld van een document, zoals `{"title": ["tent", "voor", "twee"], "text": ["past", "in", "een", "tent"]}`, en `weights`, een dictionary met de weging van elk veld, zoals `{"title": 3, "text": 1}`. De functie geeft het aantal van BM25F terug: voor elk veld het aantal keer dat het woord in de woorden van het veld staat, maal de weging van het veld, bij elkaar opgeteld. In de startcode begint `count` bij 0, en gaat de loop elke `name` van `fields` langs. Vervang de `...` door de nieuwe `count`: de telling tot nu toe, plus de weging van het veld maal hoe vaak `word` in de woorden van het veld staat. `weights[name]` geeft de weging van het veld, en `fields[name]` de woorden: een lijst, met de methode `.count` die elke lijst heeft.

        De check roept `combined_count("tent", {"title": ["tent", "for", "two"], "text": ["a", "small", "tent", "in", "a", "tent", "bag"]}, {"title": 3, "text": 1})` aan, een tent uit de Engelse les, wat `5` geeft, en `combined_count("pomp", {"title": ["luchtbed", "voor", "twee"], "text": ["met", "een", "pomp"]}, {"title": 3, "text": 1})`, wat `1` geeft. Daarna eigen gevallen: een woord in geen enkel veld, een titel met weging 0, drie velden, een weging die geen heel getal is, helemaal geen velden, en advertenties van de course.
        """),
        L.task("Tel voor elk veld de weging maal hoe vaak het woord in de woorden van het veld staat bij elkaar op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_combined_count():
    box_combined_count = L.code_box("combined_count", lines=5)
    box_combined_count
    return (box_combined_count,)


@app.cell(hide_code=True)
def check_combined_count(box_combined_count):
    result_combined_count = L.check("combined_count", box_combined_count.value)
    result_combined_count
    return (result_combined_count,)


@app.cell(hide_code=True)
def hints_combined_count():
    L.hints(
        "combined_count",
        [
            "Elke ronde telt het deel van veld `name` op bij `count`: de weging van het veld, uit `weights`, maal hoe vaak `word` in de woorden van het veld staat.",
            "De woorden van het veld zijn een lijst. Welke methode van een lijst telt hoe vaak een waarde erin staat?",
            "`        count = count + weights[name] * ___`",
        ],
        why="`count = count + weights[name] * fields[name].count(word)`: met weging 3 voor de titel telt een woord dat één keer in de titel staat als 3, voordat BM25 het totaal laat verzadigen.",
    )
    return


@app.cell(hide_code=True)
def text_after_combined_count():
    mo.md(r"""
    In het plaatje krijgt *Tent voor vier personen* het aantal 3 en de slaapzak het aantal 2, en BM25F zet de tenten bovenaan: *Tent voor twee personen*, de nieuwe tent voor twee personen, de tent voor drie personen, en dan *Tent voor vier personen* en de grote tent voor zes personen, met precies dezelfde score. Allebei hebben het aantal 3 en de lengte 28. De lange tekst kost *Tent voor vier personen* nog steeds iets: de titel telt drie keer, maar de lengte van de titel ook.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. keep: de filters

        Schrijf `keep`, die `listing` krijgt, een advertentie als dictionary, zoals `{"title": "Tent voor twee personen", "price": 45, "days": 5, "km": 140}`, `max_price`, de hoogste prijs, en `max_km`, de grootste afstand. De functie geeft `True` terug als de prijs van de advertentie hoogstens `max_price` is en de afstand hoogstens `max_km`, en anders `False`. `listing["price"]` geeft de prijs en `listing["km"]` de afstand. Vervang de `...` door de twee voorwaarden, verbonden met `and`.

        De check roept `keep({"title": "Tent for two people", "price": 45, "days": 5, "km": 140}, 100, 50)` aan, een tent uit de Engelse les, wat `False` geeft, en `keep({"title": "Tent voor vier personen", "price": 85, "days": 3, "km": 12}, 85, 50)`, wat `True` geeft: een prijs van precies de hoogste prijs komt nog door het filter. Daarna eigen gevallen: een afstand van precies de grootste afstand, een prijs die te hoog is, andere grenzen, en advertenties van de course.
        """),
        L.task("Houd een advertentie als de prijs hoogstens max_price is en de afstand hoogstens max_km."),
    ])
    return


@app.cell(hide_code=True)
def exercise_keep():
    box_keep = L.code_box("keep", lines=3)
    box_keep
    return (box_keep,)


@app.cell(hide_code=True)
def check_keep(box_keep):
    result_keep = L.check("keep", box_keep.value)
    result_keep
    return (result_keep,)


@app.cell(hide_code=True)
def hints_keep():
    L.hints(
        "keep",
        [
            'De prijs, `listing["price"]`, moet hoogstens `max_price` zijn, en de afstand moet hoogstens `max_km` zijn. Allebei moeten kloppen, dus verbind de twee met `and`.',
            "*Hoogstens* telt de grens zelf mee. Welke vergelijking geeft `True` als de twee getallen gelijk zijn?",
            '`    return listing["price"] <= max_price and ___`',
        ],
        why='`return listing["price"] <= max_price and listing["km"] <= max_km`: een advertentie blijft alleen als deze door beide filters komt, en een prijs van precies de hoogste prijs komt nog door.',
    )
    return


@app.cell(hide_code=True)
def text_after_keep():
    mo.md(r"""
    In het plaatje vallen drie tenten af: de nieuwe tent voor twee personen kost € 140, de grote tent voor zes personen € 220, en *Tent voor twee personen* staat op 140 km. Van de twaalf advertenties met *tent* blijven er negen over, en de tent voor drie personen staat nu bovenaan: de filters kijken niet naar de leeftijd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_decay():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. decay: zachte randen

        Schrijf `decay`, die `distance` krijgt, hoe ver een waarde van de beste waarde ligt, zoals de leeftijd van een advertentie in dagen, en `half`, de afstand waarbij de score gehalveerd wordt. De functie geeft 0,5 tot de macht van de afstand gedeeld door `half` terug: 1 bij afstand 0, 0,5 bij `half`, 0,25 bij twee keer `half`. In Python neemt `**` een macht: `0.5 ** 2` geeft `0.25`. Met alleen `**` en `/` werkt de functie ook als `distance` een NumPy-array is. Vervang de `...` door de decay.

        De check roept `decay(14, 14)` aan, wat `0.5` geeft, en `decay(50, 25)`, wat `0.25` geeft. Daarna eigen gevallen: een afstand van 0, afstanden die geen veelvoud van `half` zijn, een `half` die geen heel getal is, en een NumPy-array met afstanden.
        """),
        L.task("Geef 0,5 tot de macht van de afstand gedeeld door half."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decay():
    box_decay = L.code_box("decay", lines=3)
    box_decay
    return (box_decay,)


@app.cell(hide_code=True)
def check_decay(box_decay):
    result_decay = L.check("decay", box_decay.value)
    result_decay
    return (result_decay,)


@app.cell(hide_code=True)
def hints_decay():
    L.hints(
        "decay",
        [
            "De uitkomst is `0.5` tot een macht, en de macht is de afstand ten opzichte van `half`: hoe vaak `half` in `distance` past.",
            "Python rekent een macht uit voordat het deelt, dus zet de deling die de macht maakt tussen haakjes.",
            "`    return 0.5 ** (___)`",
        ],
        why="`return 0.5 ** (distance / half)`: elke `half` aan afstand halveert de score, hoe ver de waarde ook ligt.",
    )
    return


@app.cell(hide_code=True)
def text_after_decay():
    mo.md(r"""
    In het plaatje houdt een advertentie van 14 dagen oud de helft van de score, en de tent voor drie personen, 90 dagen oud, houdt 0,012 van de score. *Tent voor vier personen*, 3 dagen oud en op 12 km, houdt 0,862 · 0,717 van de score, en staat bovenaan: 0,259, tegen 0,189 voor de kampeerbrander. Eén tent blijft vooraan: de tent voor drie personen, 90 dagen oud, zakt naar de laatste plaats, en na *Tent voor vier personen* komen spullen voor een tent, zoals de kampeerbrander en de haringen.

    ### Probeer het: je eigen zoekmachine

    Verschuif de sliders: de weging van de titel, de hoogste prijs, de grootste afstand, en de dagen en de km waarna de decay een score halveert. Zet de hoogste prijs op € 300: welke tent staat bovenaan met een korte $h$ voor de dagen, zoals 1, en welke met een lange, zoals 60? Wat gebeurt er met weging 1 voor de titel, zoals bij gewone BM25? En met $h$ = 5 km?
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    try_weight = mo.ui.slider(1, 10, step=1, value=C.TRY_START["weight"], label="Weging van de titel", show_value=True)
    try_price = mo.ui.slider(10, 300, step=10, value=C.TRY_START["price"], label="Hoogste prijs (€)", show_value=True)
    try_km = mo.ui.slider(5, 200, step=5, value=C.TRY_START["km"], label="Grootste afstand (km)", show_value=True)
    try_half_days = mo.ui.slider(1, 60, step=1, value=C.TRY_START["half_days"], label="h voor de dagen", show_value=True)
    try_half_km = mo.ui.slider(5, 100, step=5, value=C.TRY_START["half_km"], label="h voor de km", show_value=True)
    mo.vstack([try_weight, mo.hstack([try_price, try_km], justify="start", gap=2),
               mo.hstack([try_half_days, try_half_km], justify="start", gap=2)])
    return try_half_days, try_half_km, try_km, try_price, try_weight


@app.cell(hide_code=True)
def try_look(try_weight, try_price, try_km, try_half_days, try_half_km):
    mo.Html(C.try_picture(L.lang, try_weight.value, try_price.value, try_km.value, try_half_days.value, try_half_km.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Met $h$ = 5 km zakt *Tent voor vier personen* net onder de kampeerbrander, op 9 km: 0,068 tegen 0,070. Een decay die te sterk is, laat de afstand winnen van de woorden. Met weging 1 voor de titel staan de stalen haringen bovenaan. Met een hoogste prijs van € 300 zet $h$ = 1 dag de nieuwe tent voor twee personen bovenaan, 2 dagen geleden geplaatst, en $h$ = 60 dagen de grote tent voor zes personen, op 3 km. Geen instelling is goed voor elke zoekvraag: een verticale zoekmachine kiest de wegingen en de $h$ door deze af te stellen op zoekvragen met bekende goede resultaten, zoals k1 en b werden afgesteld in *BM25: de klassieke rangschikking*, of laat de zoeker kiezen, zoals deze sliders doen.

    ### Oefening 4. Wat heeft elk domein nodig?

    Deel elke uitspraak hieronder in bij het domein waarvan de zoekmachine doet wat de uitspraak beschrijft.
    """)
    return


@app.cell(hide_code=True)
def exercise_domains():
    sort_domains = L.sort("domains")
    sort_domains
    return (sort_domains,)


@app.cell(hide_code=True)
def check_domains(sort_domains):
    result_domains = L.check("domains", sort_domains)
    result_domains
    return (result_domains,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** BM25F is de BM25 uit [BM25: de klassieke rangschikking](/course/open/ir-bm25?lang=nl) op een document met gewogen velden, en een filter kiest documenten op een waarde, zoals de AND uit [De inverted index](/course/open/ir-inverted-index?lang=nl) documenten koos op de woorden. In het domein *Bouwen met AI* filtert [Filters en vector databases](/course/open/build-metadata-filters?lang=nl) op dezelfde manier de teksten waarin een assistent zoekt. De volgende les, *Werkte het zoeken? P@k, MAP en MRR*, meet een rangschikking op meer manieren dan alleen de eerste plaats.

    **Een voorbeeld uit het dagelijks leven.** Een vriend die je helpt een tweedehands tent te vinden, kijkt eerst naar de tenten en niet naar de haringen, slaat de tenten boven je budget over, en belt eerst over de tent die om de hoek staat en vanochtend is geplaatst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_combined_count, result_keep, result_decay, result_domains):
    L.wrap_up(result_combined_count, result_keep, result_decay, result_domains)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
