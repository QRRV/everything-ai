# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ir-what-is-search. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-what-is-search", "nl")
    C = L.checks   # de verzameling van de course, de informatiebehoeften met de oordelen, de aanwijzingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een document als vector](/course/open/nlp-document-vectors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vector():
    quick_vector = L.choice(
        "quick-vector",
        question='In *Een document als vector* gaf een vocabulary elk woord een kolom. Met de vocabulary `{"de": 0, "hond": 1, "kat": 2, "zat": 3}`: wat is de documentvector van *de hond zag de kat*?',
        options={
            "order": "`[0, 1, 0, 2]`: de kolom van elk bekend woord, op volgorde",
            "once": "`[1, 1, 1, 0]`: een 1 voor elk woord van de vocabulary dat erin staat",
            "right": "`[2, 1, 1, 0]`: een aantal voor elke kolom, zonder *zag*",
            "new": "`[2, 1, 1, 0, 1]`: met een nieuwe kolom voor *zag*",
        },
        correct="right",
        why={
            "right": "Elke kolom houdt bij hoe vaak het woord voorkomt: *de* twee keer, *hond* en *kat* één keer, *zat* niet. *zag* heeft geen kolom en valt weg. De zoekmachine van deze les kijkt naar een deel daarvan: of een document een woord bevat.",
            "order": "Een documentvector heeft één getal voor elke kolom van de vocabulary. Wat telt het getal in een kolom? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "once": "*de* komt twee keer voor in de zin. Wat houdt een documentvector bij van hoe vaak een woord voorkomt? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "new": "De vocabulary ligt vast. Wat gebeurt er met een woord zonder kolom? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_vector
    return (quick_vector,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Een bag of words houdt bij welke woorden een tekst heeft en hoe vaak, en vergeet de volgorde. Welke twee zoekvragen geven dezelfde documentvector, met een vocabulary die elk woord van beide bevat?",
        options={
            "plural": "*een glas rode wijn* en *twee glazen goede rode wijn bij het eten*",
            "right": "*van brazilië naar amerika* en *van amerika naar brazilië*",
            "twice": "*python kopen* en *python python kopen*",
            "the": "*de slang* en *een slang*",
        },
        correct="right",
        why={
            "right": "Beide bevatten *van*, *brazilië*, *naar* en *amerika* één keer, dus de vectoren zijn gelijk, en de reizen gaan in tegengestelde richting. Een zoekmachine die alleen kijkt welke woorden een zoekvraag bevat, ziet geen verschil. Het volgende deel van de les laat een zoekmachine zien die zulke zoekvragen door elkaar haalde.",
            "plural": "*glas* en *glazen* zijn verschillende woorden, elk met een eigen kolom. Krijgt elke kolom hetzelfde aantal? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "twice": "Een documentvector telt hoe vaak elk woord voorkomt. Komt *python* in beide even vaak voor? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "the": "*de* en *een* zijn verschillende woorden. Bevatten beide zoekvragen dezelfde woorden? Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je zoekt elke dag: op het web, in een webwinkel, in je mail, in de catalogus van een bibliotheek. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze openen het boek *Introduction to Information Retrieval* (2008) ermee: honderden miljoenen mensen doen elke dag aan information retrieval, met een zoekmachine op het web of in de eigen mail.

    Een zoekmachine krijgt een paar woorden en moet daaruit opmaken wat iemand wil. Google schreef in 2019 dat het elke dag miljarden zoekopdrachten ziet, en dat 15 procent van de zoekvragen nooit eerder voorbijkwam. In hetzelfde bericht liet Google de zoekvraag *2019 brazil traveler to usa need a visa* zien, getypt door een reiziger uit Brazilië met de Verenigde Staten als bestemming. De zoekmachine van Google gaf vroeger resultaten over reizen van de Verenigde Staten naar Brazilië: de goede woorden, en de reis precies andersom.

    Google laat de resultaten dus beoordelen door mensen. In het overzicht van november 2023 telt Google ongeveer 16.000 Search Quality Raters. Een rater beoordeelt hoe goed een resultaat past bij wat iemand met een zoekvraag zoekt, op een schaal van *Fully Meets* tot *Fails to Meet*. Volgens Google tellen deze oordelen niet direct mee voor de volgorde van de resultaten: met de oordelen test Google een verandering voordat iedereen er gebruik van maakt. In 2023 deed Google 719.326 van zulke tests van de zoekkwaliteit, en voerde het bedrijf 4.781 veranderingen in de zoekmachine door; Google noemt zulke veranderingen *launches*.

    Deze les gaat over het verschil tussen de woorden die iemand typt en wat iemand wil vinden, en over hoe je een zoekmachine op dat tweede beoordeelt.

    **Een voorbeeld uit het dagelijks leven.** Vraag een bibliothecaris om een boek over pythons, en de bibliothecaris vraagt terug: over de slang, of over de programmeertaal?
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_agree = L.guess(
        "guess",
        question="TREC is een jaarlijkse evaluatie van zoeksystemen, georganiseerd door NIST, het Amerikaanse instituut voor standaarden. Bij TREC-4, in 1995, beoordeelde de schrijver van elk van 49 uitgeschreven zoektaken de documenten die voor de taak gevonden waren, en beoordeelden twee anderen een steekproef van deze documenten opnieuw; alle drie waren gepensioneerde informatieanalisten. Neem twee van de drie beoordelaars, en de documenten uit de steekproef die minstens een van de twee relevant vond. Welk deel daarvan vonden beiden relevant, gemiddeld?",
        options={
            "nine": "ongeveer negen op de tien",
            "three": "ongeveer drie op de vier",
            "right": "minder dan de helft",
            "one": "ongeveer één op de tien",
        },
        reveal="Minder dan de helft. Ellen Voorhees van NIST vergeleek in 2000 de drie sets oordelen: bij de drie paren beoordelaars waren de documenten die beiden relevant vonden 42%, 49% en 43% van de documenten die minstens een van de twee relevant vond; bij alle drie de beoordelaars 30%. Van de documenten die de schrijver van een taak relevant vond, vonden beide anderen 30% niet relevant. Toch bleef de volgorde van 33 zoeksystemen bijna gelijk, welke set oordelen Voorhees ook gebruikte. Mensen zijn het oneens over wat helpt, en zoekmachines vergelijken werkt toch.",
    )
    guess_agree
    return (guess_agree,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Information retrieval** is volgens Manning en collega's het vinden van materiaal zonder vaste structuur, meestal tekst in documenten, dat voldoet aan een informatiebehoefte, in een grote verzameling die meestal op computers staat. Een programma voor information retrieval is een zoekmachine. Het boek maakt het onderscheid waar deze les over gaat: een informatiebehoefte is het onderwerp waar iemand meer over wil weten, en verschilt van de zoekvraag: wat iemand aan de computer doorgeeft om de informatiebehoefte over te brengen.

    De **informatiebehoefte** is wat iemand wil weten of doen; de **zoekvraag** bestaat uit de woorden die iemand typt om dat te zeggen. Zoekvragen zijn kort: volgens het boek heeft een zoekvraag op het web gemiddeld twee à drie woorden. Een paar woorden laten veel onuitgesproken.

    Een document is **relevant** als het helpt bij de informatiebehoefte, en **relevantie** beoordeel je op de informatiebehoefte, niet op de woorden. Manning en collega's: een document is relevant als het ingaat op de informatiebehoefte, en niet omdat er toevallig alle woorden van de zoekvraag in staan. Het voorbeeld in het boek is de vraag of rode wijn de kans op een hartaanval meer verlaagt dan witte wijn. Een pagina die bij eten voor een gezond hart een glas rode wijn aanraadt, bevat de woorden *rode*, *wijn* en *hart*, en geeft geen antwoord op de vraag.

    Eén zoekvraag kan voor verschillende informatiebehoeften staan. Weer het boek: wie *python* in een zoekmachine typt, wil misschien weten waar een python als huisdier te koop is, of zoekt informatie over de programmeertaal Python. In 2009 schreef TREC zes informatiebehoeften op bij de zoekvraag *defender*, waaronder het programma Windows Defender, de Land Rover Defender, een arcadespel en de krant Chicago Defender.

    **Drie soorten informatiebehoefte.** In 2002 deelde Andrei Broder, op grond van werk bij de zoekmachine AltaVista, de informatiebehoeften achter zoekvragen op het web in drie soorten in. Navigationeel: iemand wil direct naar een bepaalde site. Informationeel: iemand wil informatie die op een of meer webpagina's te vinden is. Transactioneel: iemand wil via het web iets doen. *youtube* is navigationeel: iemand wil naar de site van YouTube. *leukemie* is informationeel. *koningspython kopen* is transactioneel: iemand wil iets doen, hier kopen. Broder waarschuwde dat je de soort niet met zekerheid uit de zoekvraag kunt afleiden. Hoe vaak elke soort voorkomt, hangt af van hoe je telt. Met een enquête onder gebruikers van AltaVista en 400 zoekvragen uit de logbestanden van AltaVista schatte Broder het aandeel informationele zoekvragen op minder dan 50%. Bernard Jansen, Danielle Booth en Amanda Spink deelden 1.523.793 zoekvragen van één dag in 2005 bij de zoekmachine Dogpile in met regels die naar de woorden kijken, zoals *buy* of de naam van een site; na een controle van de regels met zoekvragen die met de hand waren ingedeeld, kwam de schatting uit op ongeveer 65% informationeel, 15% navigationeel en 20% transactioneel.

    **Mensen beoordelen.** Om een zoekmachine te meten heb je documenten, informatiebehoeften en oordelen nodig. TREC schrijft elke informatiebehoefte uit als een onderwerp, met een titel, een beschrijving en een toelichting, en houdt het onderwerp gescheiden van de zoekvraag: TREC noemt een informatiebehoefte die in gewone taal is uitgeschreven een *topic*, om het verschil met een zoekvraag duidelijk te maken. Mensen beoordelen daarna de bovenste documenten die de zoeksystemen bij een onderwerp teruggeven: relevant of niet. Manning en collega's raden minstens vijftig informatiebehoeften aan: vijftig blijkt meestal genoeg.

    Het plaatje voert de zoekvraag *python* uit op de verzameling van de course, dertien documenten, en beoordeelt elk resultaat op twee informatiebehoeften, zoals mensen een zoekmachine beoordelen.
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
    | **information retrieval** | in een grote verzameling de documenten vinden die helpen bij een informatiebehoefte |
    | **informatiebehoefte** | wat iemand wil weten of doen bij het zoeken |
    | **zoekvraag** | de woorden die iemand typt om een informatiebehoefte uit te drukken, vaak twee of drie |
    | **relevantie** | of een document helpt bij de informatiebehoefte; een document dat helpt, is **relevant** |
    | **navigationeel, informationeel, transactioneel** | de drie soorten informatiebehoefte van Broder: naar een site gaan, iets te weten komen, iets doen |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een zoekmachine te beoordelen op één informatiebehoefte:

    1. Schrijf de informatiebehoefte uit in een zin, zoals een onderwerp van TREC; de zoekvraag alleen zegt het niet.
    2. Voer de zoekvraag uit: de resultaten zijn de documenten die de zoekmachine teruggeeft.
    3. Beoordeel elk resultaat op de informatiebehoefte, niet op de woorden van de zoekvraag: relevant of niet.
    4. Zoek de relevante documenten die de zoekmachine niet teruggaf: een document kan helpen zonder elk woord van de zoekvraag te bevatten.
    5. Tel beide soorten fouten: resultaten die niet helpen, en relevante documenten die gemist zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.TEXTS` bevat de dertien documenten van de verzameling van de course in de taal van deze les, en `C.DOCS` dezelfde documenten als lijsten woorden, gemaakt met `C.words`, dat een tekst in woorden in kleine letters knipt. `C.QUERIES` bevat de twee zoekvragen van de les als lijsten woorden, en `C.NEEDS` de drie informatiebehoeften erachter, elk met de zoekvraag en de nummers van de documenten die erbij helpen, volgens het oordeel van de course; `C.NEED_TEXT` schrijft elke informatiebehoefte uit. `C.search` geeft de nummers van de documenten met elk woord van een zoekvraag, de resultaten, en `C.split` verdeelt de resultaten bij een informatiebehoefte over drie lijsten:
    """)
    return


@app.cell
def example_cell():
    print(C.words(C.TEXTS[L.lang][1]))
    _query = C.QUERIES[L.lang]["python"]
    _results = C.search(_query, C.DOCS[L.lang])
    print(_query, _results)
    for _need in ["buy", "learn"]:
        print(C.NEED_TEXT[L.lang][_need])
        print(C.split(_results, C.NEEDS[_need]["relevant"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vijf van de dertien documenten bevatten het woord *python*: 0, 2, 3, 4 en 5. Bij de informatiebehoefte om een python als huisdier te kopen helpt alleen document 3: het staat in de eerste lijst. Documenten 0, 2, 4 en 5 zijn resultaten die niet helpen, in de tweede lijst. Document 1, *Koningspythons te koop*, helpt wel maar is geen resultaat, in de derde lijst: er staat *koningspythons* in, niet *python*. Bij de informatiebehoefte om de taal te leren helpt document 0, en helpen de andere vier resultaten niet. De zoekvraag en de resultaten zijn bij beide informatiebehoeften hetzelfde; alleen de informatiebehoefte bepaalt welke resultaten goed zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_matches():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. matches: elk woord van de zoekvraag

        Schrijf `matches`, die `query` krijgt, een lijst woorden, en `document`, de woorden van één document als lijst, zoals `C.DOCS["nl"][1]`. De functie geeft `True` terug als elk woord van de zoekvraag in het document staat, en `False` zodra er één woord ontbreekt. Dit is de eenvoudigste zoekmachine: de documenten die matchen, zijn de resultaten. In de startcode gaat de `for` langs elk `word` van `query`, en na de `for` zegt `return True` dat er geen woord ontbrak. Vervang de `...` door de test waarmee de functie stopt met `False`.

        De check roept `matches(["python"], C.DOCS["en"][1])` aan, met een document van de Engelse les, wat `False` geeft, omdat document 1 *pythons* bevat en niet *python*, en `matches(["rode", "wijn", "hart"], C.DOCS["nl"][7])`, wat `True` geeft. Daarna eigen gevallen: een document met een deel van de woorden, met twee zoekvragen; een zoekvraag van één woord die matcht; een document met elk woord in een andere volgorde; twee Nederlandse documenten; een zoekvraag met twee keer hetzelfde woord; en een leeg document.
        """),
        L.task("Geef True als elk woord van de zoekvraag in het document staat, en False als er een ontbreekt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matches():
    box_matches = L.code_box("matches", lines=6)
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
            "De functie kan `False` geven zodra er één `word` van de zoekvraag is dat het document mist. De test komt in de `if`: `x in a_list` is `True` als `x` een van de elementen van een lijst is.",
            "Test het woord van deze ronde van de `for`, `word`, tegen de lijst `document`: het document is geen resultaat als het woord er niet in staat. `not in` is het omgekeerde van `in`.",
            "`        if word ___:`",
        ],
        why="`if word not in document:`: één woord van de zoekvraag dat het document mist, is genoeg voor `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_matches():
    mo.md(r"""
    In het plaatje vindt *python* vijf documenten, en *rode wijn hart* drie: de pagina over rode wijn, witte wijn en het hart, de pagina over eten voor een gezond hart, en de wandeling door de oude stad, waar *hart* het centrum van de stad betekent. *Alcohol en hartaanvallen: verschillen bier, wit en rood?* beantwoordt de vraag over rode en witte wijn, en is geen resultaat: er staat geen enkel woord van de zoekvraag in, want *hartaanvallen* is niet *hart*, en *rood* is niet *rode*. Manning en collega's noemen zoeken op precies de woorden van de zoekvraag Boolean retrieval; een document matcht of matcht niet, en niets zegt welk resultaat het beste is. Latere lessen van deze module maken het snel en zetten de resultaten op volgorde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. split: de resultaten beoordelen op een informatiebehoefte

        Schrijf `split`, die `results` krijgt, de nummers van de documenten die een zoekmachine teruggaf, en `relevant`, de nummers van de documenten die helpen bij de informatiebehoefte. De functie geeft drie lijsten terug: `found`, de resultaten die relevant zijn; `extra`, de resultaten die niet relevant zijn; en `missed`, de relevante documenten die geen resultaat zijn, in de volgorde van `relevant`. In de startcode gaat de `for` langs elk `doc` van `results` en zet het in `found` of in `extra`, met `.append`. Vervang de `...` door de lijst met gemiste documenten.

        De check roept `split([0, 2, 3, 4, 5], [1, 3])` aan, de resultaten van *python* bij de informatiebehoefte om een python als huisdier te kopen, wat `([3], [0, 2, 4, 5], [1])` geeft, en `split([6, 7, 8], [6, 9])`, de resultaten van *rode wijn hart* bij de vraag over rode en witte wijn, wat `([6], [7, 8], [9])` geeft. Daarna eigen gevallen: de informatiebehoefte om de taal te leren, resultaten in een andere volgorde dan de relevante documenten, geen resultaten, geen relevante documenten, en elk relevant document gevonden.
        """),
        L.task("Geef de resultaten die helpen, de resultaten die niet helpen, en de relevante documenten die de zoekmachine miste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=10)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "`missed` kan een list comprehension over `relevant` zijn: `[doc for doc in relevant if test]` houdt elk `doc` van `relevant` waarbij de test klopt.",
            "Een relevant document is gemist als de zoekmachine het niet teruggaf: als het niet een van de nummers in `results` is.",
            "`    missed = [doc for doc in relevant if ___]`",
        ],
        why="`missed = [doc for doc in relevant if doc not in results]`: de relevante documenten die geen resultaat zijn, in de volgorde van `relevant`.",
    )
    return


@app.cell(hide_code=True)
def text_after_split():
    mo.md(r"""
    De twee informatiebehoeften achter *python* krijgen dezelfde vijf resultaten, en je split beoordeelt de resultaten op twee manieren: één resultaat helpt bij de behoefte om te kopen, één bij de behoefte om te leren. Een zoekmachine die alleen de woorden ziet, kan het niet voor beide tegelijk beter doen. De twee lijsten met fouten trekken elk een andere kant op: een zoekmachine die meer documenten teruggeeft, mist niet meer en vaak minder relevante documenten, maar geeft ook meer documenten terug die niet helpen. De les *Werkte het zoeken? P@k, MAP en MRR* maakt van zulke oordelen getallen, en met oordelen bij veel informatiebehoeften vergelijken zulke getallen zoekmachines, zoals bij TREC.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind_of():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. kind_of: de soort informatiebehoefte raden

        Schrijf `kind_of`, die `query` krijgt, een lijst woorden, en `clues`, een dictionary van een soort informatiebehoefte naar de lijst met aanwijzingen voor de soort, zoals `C.CLUES["nl"]`: namen van sites en woorden zoals *inloggen* voor `"navigational"`, en woorden zoals *kopen* en *downloaden* voor `"transactional"`. De functie geeft de eerste soort terug, in de volgorde van `clues`, met een woord van de zoekvraag bij de aanwijzingen, en `"informational"` als geen enkel woord een aanwijzing is. Jansen en collega's gebruikten zulke regels, in deze volgorde: eerst navigationeel, dan transactioneel, en informationeel voor de rest. In de startcode gaat de buitenste `for` langs elke `kind` van `clues`, en de binnenste `for` langs elk `word` van `query`. Vervang de `...` door de test voor een aanwijzing.

        De check roept `kind_of(["facebook", "login"], C.CLUES["en"])` aan, met de aanwijzingen van de Engelse les, wat `"navigational"` geeft, en `kind_of(["hoe", "leer", "ik", "python"], C.CLUES["nl"])`, wat `"informational"` geeft. Daarna eigen gevallen: een transactionele Engelse zoekvraag, een transactionele Nederlandse zoekvraag, een zoekvraag met een aanwijzing van elke soort, dezelfde zoekvraag met de soorten in de andere volgorde, een zoekvraag met alleen een navigationele aanwijzing, een lege zoekvraag, en helemaal geen aanwijzingen.
        """),
        L.task("Geef de eerste soort met een aanwijzing in de zoekvraag, of \"informational\" als er geen is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kind_of():
    box_kind_of = L.code_box("kind_of", lines=7)
    box_kind_of
    return (box_kind_of,)


@app.cell(hide_code=True)
def check_kind_of(box_kind_of):
    result_kind_of = L.check("kind_of", box_kind_of.value)
    result_kind_of
    return (result_kind_of,)


@app.cell(hide_code=True)
def hints_kind_of():
    L.hints(
        "kind_of",
        [
            "`clues` is een dictionary: `clues[\"navigational\"]` is de lijst met aanwijzingen voor deze soort, en `word` is één woord van de zoekvraag. De test komt in de `if`.",
            "Het woord is een aanwijzing voor `kind`, de soort van deze ronde van de buitenste `for`, als het een van de aanwijzingen voor deze soort is: zoek de lijst op in `clues`, met `kind` als sleutel.",
            "`            if word in ___:`",
        ],
        why="`if word in clues[kind]:`: het woord is een van de aanwijzingen voor de soort.",
    )
    return


@app.cell(hide_code=True)
def text_after_kind_of():
    mo.md(r"""
    Met de versie van de course komen de regels in elke taal bij 12 van de 15 zoekvragen overeen met wat de course erin leest. De regels missen *lufthansa*, een site die niet bij de aanwijzingen staat. *geschiedenis van bol.com* lezen de regels als navigationeel, terwijl iemand over de site wil lezen en er niet naartoe wil; en *boeken over slangen* als transactioneel, omdat *boeken* een aanwijzing is voor een hotel of een kaartje boeken. Jansen en collega's controleerden de regels met 400 zoekvragen die met de hand waren ingedeeld: bij 74% gaven de regels dezelfde soort als de indeling met de hand. Woorden kunnen de ene kant op wijzen en de informatiebehoefte de andere, en alleen wie een zoekvraag typte, weet zeker wat de informatiebehoefte is.

    ### Probeer het: doorzoek de verzameling

    Typ een zoekvraag in het vak. Het plaatje laat zien welke van de dertien documenten elk woord ervan bevatten, en bij elk ander document welke woorden van de zoekvraag ontbreken. Probeer *slang*: welke documenten over slangen mist de zoekvraag? Probeer daarna een zoekvraag van drie woorden, en kijk hoe snel de resultaten op zijn.
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
def text_exercise_judge():
    mo.md(r"""
    ### Oefening 4. Gevonden, gemist of te veel?

    Onderwerp 427 van TREC, uit 1999, heeft de titel *UV damage, eyes*. De beschrijving vraagt om documenten over de schade die ultraviolet licht (uv) van de zon aan de ogen kan doen. De toelichting zegt: een relevant document gaat over ziekten door blootstelling van de ogen aan uv-licht, over behandelingen van de schade, of over voorlichting om schade te voorkomen. Documenten over de behandeling van staar en van oogmelanoom zijn relevant, ook als er geen oorzaak genoemd wordt. Documenten over schade door straling uit radioactieve bronnen of door lasers zijn niet relevant. Bij staar wordt de lens van het oog troebel; oogmelanoom is een vorm van kanker in het oog.

    De zoekvraag is *uv schade ogen*, en de zoekmachine is je `matches`. Deel elk document hieronder in: gevonden (relevant, met elk woord van de zoekvraag), gemist (relevant, maar een woord van de zoekvraag staat er niet in, dus de zoekmachine geeft het niet terug) of te veel (elk woord van de zoekvraag staat erin, maar het is niet relevant).
    """)
    return


@app.cell(hide_code=True)
def exercise_judge():
    sort_judge = L.sort("judge")
    sort_judge
    return (sort_judge,)


@app.cell(hide_code=True)
def check_judge(sort_judge):
    result_judge = L.check("judge", sort_judge)
    result_judge
    return (result_judge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een document als vector](/course/open/nlp-document-vectors?lang=nl) hield bij welke woorden een tekst bevat, en vergat de volgorde; de zoekmachine van deze les gebruikt alleen welke woorden een document bevat, en ziet dus niet dat *van brazilië naar amerika* een reis de andere kant op is dan *van amerika naar brazilië*. De intent van [Een trein boeken: frames en slots](/course/open/nlp-slot-filling?lang=nl) lijkt op de soort informatiebehoefte: wat de gebruiker wil doen, gelezen uit de woorden. De volgende les, *De inverted index*, vindt de documenten met elk woord van een zoekvraag zonder elk document te lezen, en latere lessen van deze module zetten de resultaten op volgorde en meten een zoekmachine met oordelen van mensen.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt in je mail op *factuur*, en in de mail die je nodig hebt, staat *rekening*: relevant, en gemist.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_matches, result_split, result_kind_of, result_judge):
    L.wrap_up(result_matches, result_split, result_kind_of, result_judge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
