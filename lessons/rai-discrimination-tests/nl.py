# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-discrimination-tests. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-discrimination-tests", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_opportunity():
    quick_opportunity = L.choice(
        "quick-opportunity",
        question="Het model van een bank geeft een lening aan 0,6 van de vrouwen die de lening zouden terugbetalen, en aan 0,6 van zulke mannen. Aan welk criterium voldoet dat?",
        options={
            "parity": "Demographic parity: evenveel ja per groep",
            "right": "Equal opportunity: dezelfde recall",
            "calibration": "Kalibratie: een ja betekent hetzelfde",
        },
        correct="right",
        why={
            "parity": "Demographic parity telt de ja's onder iedereen, niet alleen onder wie zou terugbetalen. Hetzelfde deel met een ja onder die mensen is dezelfde recall: equal opportunity. Zie *Het idee* in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).",
            "right": "Van wie zou terugbetalen, krijgt in beide groepen hetzelfde deel een lening: dezelfde true positive rate, equal opportunity.",
            "calibration": "Kalibratie begint bij de mensen met een ja, en vraagt hoeveel van hen terugbetalen. Deze vraag begint bij de mensen die terugbetalen. Zie *Het idee* in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).",
        },
    )
    quick_opportunity
    return (quick_opportunity,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="Een model heeft in twee groepen dezelfde precision (onder 1) en dezelfde recall, maar in groep A zijn meer mensen positief. Hoe zit het met de false positive rates van de twee groepen?",
        options={
            "same": "Die zijn ook gelijk",
            "right": "Die moeten verschillen",
            "zero": "Die moeten allebei 0 zijn",
        },
        correct="right",
        why={
            "same": "Ook dan verandert de base rate de false positive rate: in Oefening 2 van [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) gaven dezelfde precision en recall 0,03 en 0,11.",
            "right": "Uit de base rate, de precision en de recall van een groep volgt de false positive rate; verschillende base rates geven verschillende rates.",
            "zero": "Alleen een model zonder false positives heeft voor allebei 0, en dan is de precision 1. Met dezelfde precision onder de 1 verschillen de rates. Zie *Waarom niet allebei* in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige lessen had je het model en de data: je kon elke ja en elke vergissing tellen. Vaak heb je geen van beide. Een verhuurder, een werkgever of een agent beslist, en je ziet alleen wat eruit komt. Behandelden ze groepen verschillend? Deze les laat twee tests zien die van buitenaf werken, en waar ze allebei tekortschieten. De eerste werkt ook op een model waar je alleen invoer naartoe kunt sturen.

    **Een voorbeeld uit het dagelijks leven.** Twee vrienden bellen dezelfde verhuurder over dezelfde kamer, een uur na elkaar. Ze zeggen hetzelfde: hetzelfde inkomen, dezelfde datum om erin te trekken. Alleen hun namen verschillen. De een hoort dat de kamer vrij is, de ander dat hij net verhuurd is. Eén paar telefoontjes bewijst weinig; honderd van zulke paren bewijzen veel.
    """)
    return


@app.cell(hide_code=True)
def predict_callback():
    guess_callback = L.guess(
        "callback",
        question="Onderzoekers reageerden in Boston en Chicago op vacatures met verzonnen cv's die op de naam na gelijk waren: sommige met een naam die wit klonk, sommige met een naam die Afro-Amerikaans klonk. De witte namen werden in 9,65 procent van de gevallen teruggebeld. Hoe vaak denk je dat de Afro-Amerikaanse namen werden teruggebeld?",
        options={
            "same": "Ongeveer even vaak: 9,5 procent",
            "right": "Minder vaak: 6,45 procent",
            "rare": "Bijna nooit: 1 procent",
        },
        reveal="**Minder vaak: 6,45 procent.** Dat is 3,20 procentpunt minder: de witte namen werden 50 procent vaker gebeld, ongeveer één keer per 10 cv's tegen één keer per 15. De cv's verschilden alleen in de naam. *Het idee* hieronder laat zien hoe je zo'n test opzet.",
    )
    guess_callback
    return (guess_callback,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Gelijke sollicitaties.** Om te testen of een beslissing afhangt van een protected attribute, stuur je sollicitaties of aanvragen die in alles gelijk zijn behalve dat attribute, en tel je hoe vaak elke soort een ja krijgt. Zo'n test heet een **praktijktest**. Bij sommige praktijktests gaan getrainde mensen in paren zelf langs; Barocas, Hardt en Narayanan schrijven in *Fairness and Machine Learning* dat het Amerikaanse ministerie van Volkshuisvesting (HUD) er in de jaren zeventig mee begon, om te onderzoeken hoe huizenkopers en huurders uit minderheidsgroepen werden behandeld. Een **correspondentietest** is een praktijktest op papier: de sollicitaties zijn geschreven en verzonnen. Marianne Bertrand en Sendhil Mullainathan (2004) reageerden op meer dan 1300 vacatures met bijna 5000 cv's, met namen als Emily Walsh of Greg Baker, of als Lakisha Washington of Jamal Jones. Hun uitkomst: cv's met een witte naam hadden 9,65 procent kans om teruggebeld te worden, gelijkwaardige cv's met een Afro-Amerikaanse naam 6,45 procent. In gewone woorden: met een witte naam moest je ongeveer 10 cv's sturen voor één telefoontje, met een Afro-Amerikaanse naam ongeveer 15. Zelfs de woonwijk was in elk paar gelijk, dus, merken Barocas, Hardt en Narayanan op, onderschat de opzet waarschijnlijk hoeveel dingen die er moreel niet toe zouden mogen doen in de praktijk meewegen bij het terugbellen.

    **In Nederland.** In 2014 stuurde het Sociaal en Cultureel Planbureau (SCP) 504 sollicitaties die alleen in de naam verschilden (autochtoon Nederlands, Hindoestaans of Marokkaans) naar 176 echte vacatures in de regio Den Haag. Sollicitanten met een autochtoon Nederlandse naam hadden 34 procent kans op succes, met een Hindoestaanse naam 23 procent, en met een Marokkaanse naam 19 procent. Lex Thijssen, Marcel Coenders en Bram Lancee (2019) stuurden 4211 sollicitaties naar vacatures in tien beroepen, en vonden een geschatte kans op een positieve reactie van 46 procent voor sollicitanten van Nederlandse herkomst tegen 35 procent voor sollicitanten met een migratieherkomst. Zo'n test kan ook als bewijs dienen: volgens een handleiding uit 2014 past een praktijktest het best om directe discriminatie aan te tonen, de soort uit [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) waarbij het protected attribute zelf het verschil maakt.

    **Dezelfde test op een model.** De meeste van zulke onderzoeken, schrijven Barocas, Hardt en Narayanan, kun je het best zien als een test van blindheid: gebruikt wie beslist een protected attribute direct? Bij een model verander je alleen dat attribute. Sainyam Galhotra en anderen (2017) meten het zo: laat het model twee mensen beoordelen die in alles gelijk zijn behalve hun afkomst, en kijk of de uitkomst verandert. Sahaj Garg en anderen (2019) verwisselden woorden in zinnen: een model gaf de Engelse zin *Some people are gay* een kans van 98 procent om toxisch te zijn, en *Some people are straight* 2 procent. Maar het model zonder de kolom `sex` uit [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl) doorstaat zo'n test zonder moeite: het ziet de kolom niet, en vindt het geslacht toch via proxies. Een systeem dat een fairnesstest doorstaat, schrijft het boek, moet je niet zien als een certificaat dat het systeem fair is.

    **De outcome test.** Vaak kun je geen testgevallen sturen: de beslissingen zijn al lang genomen. Kijk dan naar wat ervan terechtkwam. Camelia Simoiu, Sam Corbett-Davies en Sharad Goel (2017) beschrijven de **outcome test**, die ze terugvoeren op de econoom Gary Becker. Die test kijkt niet naar hoe vaak een beslissing valt, maar naar hoe vaak ze goed uitpakt. Bij leningen: zonder discriminatie zouden mensen uit een minderheid die een lening krijgen, die even vaak moeten terugbetalen als witte mensen met een lening. Betalen ze vaker terug, dan geeft de geldverstrekker misschien alleen leningen aan uitzonderlijk geschikte mensen uit een minderheid: een hogere lat voor hen, of in de woorden van de vorige les: een hogere threshold. Bij het doorzoeken van auto's heet dat succespercentage de **hit rate**: het deel van de doorzoekingen dat iets verbodens oplevert. Het is de precision per groep uit [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl), en je hebt er alleen de uitkomsten voor nodig van de mensen die een ja kregen, en die kan een buitenstaander zien.

    **Wat het opleverde.** John Knowles, Nicola Persico en Petra Todd (1999) bekeken 1590 doorzoekingen van auto's op een snelweg in Maryland. Er werden drugs gevonden bij 0,34 van de doorzoekingen van zwarte bestuurders, 0,32 van die van witte bestuurders en 0,11 van die van Latijns-Amerikaanse bestuurders. In hun model pasten de eerste twee getallen niet bij vooroordelen over afkomst; bij Latijns-Amerikaanse mannen paste het lage getal wel bij discriminatie van die groep, al was die steekproef klein. Emma Pierson en anderen (2020) verzamelden gegevens over bijna 100 miljoen verkeerscontroles in de Verenigde Staten. Bij de verkeerspolitie van acht staten vonden doorzoekingen van witte bestuurders in 32,0 procent van de gevallen iets verbodens, van zwarte bestuurders in 29,4 procent en van Latijns-Amerikaanse bestuurders in 24,3 procent. Volgens hen wijst de outcome test er dus op dat de beslissingen om te doorzoeken nadelig kunnen zijn voor Latijns-Amerikaanse bestuurders, maar is het bewijs voor zwarte bestuurders minder duidelijk.

    **Gelijke lat, ongelijk succes.** Toch kan de outcome test misleiden. Barocas, Hardt en Narayanan noemen de stap van een ander succespercentage naar een andere lat een denkfout. Het succespercentage is een gemiddelde over iedereen boven de lat, ook over de mensen die er ver boven zitten. Zitten de mensen boven de lat in de ene groep er verder boven, dan is het succespercentage van die groep hoger, met precies dezelfde lat. Dat is **infra-marginality**: de meting middelt over gevallen die ver van de decision threshold liggen. Het kan ook een lagere lat verbergen, stellen Simoiu en anderen. In Raleigh, North Carolina, werden zwarte bestuurders vaker doorzocht dan witte, 4 tegen 2 procent, en vonden doorzoekingen bij zwarte bestuurders ook vaker iets, 16 tegen 13 procent; volgens de outcome test lag de lat dus lager voor witte bestuurders. Simoiu en anderen schatten de lat zelf met een threshold test; volgens hun schatting lag de lat lager voor zwarte bestuurders: 6 tegen 9 procent, wat volgens hen op discriminatie wijst. Hun test heeft een model nodig van hoe de kansen verdeeld zijn, terwijl de outcome test, zoals Barocas, Hardt en Narayanan schrijven, zonder model werkt.

    **Je ziet alleen de ja's.** De outcome test en de threshold test kunnen alleen kijken naar de uitkomsten van de mensen die een ja kregen; een praktijktest heeft die uitkomsten niet nodig. Of iemand een lening terugbetaalt, zie je alleen als die persoon er een kreeg. Himabindu Lakkaraju en anderen (2017) geven borgtocht als voorbeeld: of een verdachte niet op de zitting verschijnt, zie je alleen als de rechter de verdachte op borgtocht vrijlaat. Dat is het **selective labels problem**. Uit eerdere beslissingen kun je de precision per groep meten, maar niet de recall of de false positive rate: daarvoor heb je de uitkomsten nodig van de mensen die een nee kregen. Een test van gelijke fouten op zulke data, schrijven Barocas, Hardt en Narayanan, lijdt onder selection bias. Een uitweg, sommige mensen een ja geven wat de voorspelling ook is, roept ethische bezwaren op, en zulke experimenten worden in de praktijk zelden gedaan.

    **Probeer het.** De cel hieronder verzint 100 bestuurders in elk van twee groepen, met dezelfde kansen dat ze iets verbodens bij zich hebben: 70 bestuurders met een kans van 0,01, 20 met 0,2 en 10 met 0,6. Een agent doorzoekt elke bestuurder met een kans van minstens de lat van de eigen groep. De functie `_search` telt per groep hoeveel bestuurders worden doorzocht, en hoeveel vondsten dat gemiddeld oplevert: 20 bestuurders met een kans van 0,2 geven 4 vondsten. Eerst krijgen beide groepen de lat 0,5, daarna krijgt groep B een lagere lat, 0,1. Voer de cel uit.
    """)
    return


@app.cell
def demo_searches():
    # verzonnen bestuurders: per groep een kans dat ze iets bij zich hebben, en hoeveel bestuurders die kans hebben
    _drivers = {
        "groep A": {0.01: 70, 0.2: 20, 0.6: 10},
        "groep B": {0.01: 70, 0.2: 20, 0.6: 10},
    }


    def _search(bars):
        # per groep: de doorzochte bestuurders, de vondsten gemiddeld, en het deel van de doorzoekingen dat iets vindt
        for group in _drivers:
            searched = 0
            found = 0
            for chance, count in _drivers[group].items():
                if chance >= bars[group]:
                    searched = searched + count
                    found = found + chance * count
            print("  ", group, "| lat", bars[group], "| doorzocht", searched, "van de 100", "| gevonden", round(found, 1),
                  "| hit rate", round(found / searched, 2))


    print("dezelfde lat voor beide groepen")
    _search({"groep A": 0.5, "groep B": 0.5})
    print("een lagere lat voor groep B")
    _search({"groep A": 0.5, "groep B": 0.1})
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_hits({"title": "De doorzoekingen van de agent per groep",
                                "rows": ["Dezelfde lat van 0,5", "Een lat van 0,1 voor groep B"],
                                "xlabels": ["deel doorzocht", "hit rate"],
                                "key": "open: groep A, gevuld: groep B", "decimal": ",",
                                "alt": "Twee panelen met elk twee rijen. Links het deel van de bestuurders dat werd doorzocht in groep A en groep B: dezelfde lat 0,10 en 0,10, een lagere lat voor groep B 0,10 en 0,30. Rechts de hit rate: dezelfde lat 0,60 en 0,60, een lagere lat voor groep B 0,60 en 0,33."})),
        mo.md(r"""
        Met dezelfde lat worden beide groepen even vaak doorzocht, en vinden de doorzoekingen even vaak iets: in 0,6 van de gevallen. Met een lagere lat voor groep B worden drie keer zoveel bestuurders uit groep B doorzocht, en de extra doorzoekingen treffen bestuurders met een kans van maar 0,2: de hit rate van groep B zakt naar 0,33. Zo redeneert de outcome test: een lagere hit rate wijst op een lagere lat. Hier hadden de groepen dezelfde kansen, dus heeft de test gelijk. Oefening 2 laat een geval zien waarin hij zich vergist.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **praktijktest** | een test met paren mensen of sollicitaties die gelijk zijn op een protected attribute na, om te zien of dat attribute de beslissing verandert; in het Engels een audit study |
        | **correspondentietest** | een praktijktest met geschreven, verzonnen sollicitaties, zoals cv's die alleen in de naam verschillen |
        | **outcome test** | vergelijkt per groep het succespercentage van beslissingen: van de mensen met een ja, het deel bij wie die ja goed uitpakte; een lager percentage in een groep wijst op een lagere lat daar |
        | **hit rate** | bij een outcome test van doorzoekingen: het deel van de doorzoekingen dat iets vindt, de precision van de doorzoekingen |
        | **infra-marginality** | succespercentages verschillen tussen groepen bij één lat voor iedereen, omdat de mensen boven de lat in de ene groep er verder boven zitten |
        | **selective labels problem** | je ziet alleen de uitkomst van de mensen die een ja kregen, dus kun je de vergissingen onder de mensen met een nee niet tellen |
        """),
        L.watch_out("Een gelijk succespercentage bewijst niet dat er geen discriminatie is, en een ongelijk percentage bewijst niet dat die er wel is. Pierson en anderen merken op dat de outcome test discriminatie ook kan missen als die er wel is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om van buitenaf uit te zoeken of beslissingen groepen verschillend behandelen:

    1. Kun je testgevallen sturen? Maak paren die gelijk zijn op het protected attribute na, stuur er veel, en vergelijk per groep het deel met een ja: een praktijktest. Verander bij een model ook de proxies van het attribute, niet alleen de kolom.
    2. Alleen eerdere beslissingen? Neem per groep het succespercentage van de ja's: van de mensen met een ja, het deel bij wie die ja goed uitpakte. Een lager percentage in een groep wijst op een lagere lat daar: de outcome test.
    3. Vraag je af of de groepen boven de lat anders verdeeld zijn. Dan kan een gelijke lat ongelijke succespercentages geven, en een ongelijke lat gelijke: infra-marginality.
    4. Zeg wat je niet kunt zien: de uitkomsten van de mensen met een nee. Zonder die uitkomsten is er geen recall en geen false positive rate per groep.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De telefoontjes van Bertrand en Mullainathan: ze stuurden met elke soort naam 2435 cv's. De publicatie geeft percentages; de aantallen telefoontjes in de cel, 235 en 157, zijn uitgerekend uit de 9,65 en 6,45 procent van 2435. De cel rekent per soort naam het deel uit dat werd teruggebeld, en daarna hoeveel keer zo vaak de witte namen werden gebeld: stap 1 van het recept. Voer de cel uit.
    """)
    return


@app.cell
def example_callbacks():
    # Bertrand en Mullainathan (2004): 2435 cv's per soort naam
    _sent = 2435
    # de telefoontjes, uitgerekend uit hun percentages
    _called = {"witte namen": 235, "Afro-Amerikaanse namen": 157}

    _rate = {}
    for _name in _called:
        _rate[_name] = _called[_name] / _sent
    print("teruggebeld:", "witte namen", round(_rate["witte namen"], 4),
          "| Afro-Amerikaanse namen", round(_rate["Afro-Amerikaanse namen"], 4))
    print("keer zo vaak:", round(_rate["witte namen"] / _rate["Afro-Amerikaanse namen"], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De witte namen werden in 0,0965 van de gevallen teruggebeld en de Afro-Amerikaanse namen in 0,0645: 9,65 en 6,45 procent, zoals in de publicatie. De witte namen werden 1,5 keer zo vaak gebeld, de 50 procent meer uit de publicatie. Oefening 1 doet hetzelfde voor het Nederlandse onderzoek van het SCP.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ratio():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoeveel keer zo vaak

        `success` bevat de kans op succes per naam in het onderzoek van het SCP: `"native Dutch"` is autochtoon Nederlands, `"Hindustani"` Hindoestaans en `"Moroccan"` Marokkaans. De loop gaat langs de sleutels, en `group` is bij elke ronde de naam van één groep. Vervang de `...` door één regel: `ratio[group]`, de kans van autochtoon Nederlandse sollicitanten gedeeld door de kans van `group`, zoals het uitgewerkte voorbeeld het ene deel door het andere deelde. De check probeert ook andere kansen.
        """),
        L.task("Reken `ratio[group]` uit voor elke groep in `success`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ratio():
    box_ratio = L.code_box("ratio", lines=8)
    box_ratio
    return (box_ratio,)


@app.cell(hide_code=True)
def check_ratio(box_ratio):
    result_ratio = L.check("ratio", box_ratio.value)
    result_ratio
    return (result_ratio,)


@app.cell(hide_code=True)
def hints_ratio():
    L.hints(
        "ratio",
        [
            "Het uitgewerkte voorbeeld deelde het aandeel teruggebelde witte namen door dat van de Afro-Amerikaanse namen. Deel hier de kans van autochtoon Nederlandse sollicitanten door de kans van elke groep om de beurt.",
            "In de loop is `success[group]` de kans van de groep van deze ronde, en `success[\"native Dutch\"]` de kans van autochtoon Nederlandse sollicitanten.",
            "`ratio[group] = success[\"native Dutch\"] / ___`.",
        ],
        why="`ratio[group] = success[\"native Dutch\"] / success[group]`. Dat geeft 1,0 voor autochtoon Nederlands, 1,48 voor Hindoestaans en 1,79 voor Marokkaans.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén lat, twee hit rates

        Weer verzonnen bestuurders, nu met de kansen uit het voorbeeld van Simoiu, Corbett-Davies en Goel: in groep A 90 bestuurders met een kans van 0,01 en 10 met 0,75; in groep B 80 met 0,01 en 20 met 0,5. De agent doorzoekt in beide groepen elke bestuurder met een kans van minstens `bar`, 0,1. `drivers` bevat per groep hoeveel bestuurders elke kans hebben; de binnenste loop gaat erlangs met `chance` en `count`; `searched` en `found` beginnen per groep bij 0, en de laatste regel deelt ze tot `hit_rate[group]`. Vervang de `...` in de binnenste loop door de `if` uit *Probeer het* en de twee regels eronder, met `bar` in plaats van `bars[group]`. De check probeert ook een andere lat.
        """),
        L.task("Reken `hit_rate[group]` uit voor elke groep in `drivers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=14)
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
            "Alleen de bestuurders met een kans van minstens `bar` worden doorzocht. Tel bij elke soort bestuurder die wordt doorzocht het aantal bestuurders op bij `searched`, en het gemiddelde aantal vondsten bij `found`.",
            "`if chance >= bar:`, en daaronder, vier spaties verder ingesprongen, `searched = searched + count`.",
            "`found = found + ___ * count`.",
        ],
        why="In de binnenste loop: `if chance >= bar:`, dan `searched = searched + count` en `found = found + chance * count`. Dat geeft een hit rate van 0,75 voor groep A en 0,50 voor groep B.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.md(r"""
    ### Oefening 3. Wat een bank kan meten
    """)
    return


@app.cell(hide_code=True)
def choice_labels_cell():
    choice_labels = L.choice(
        "labels",
        question="Alleen bij de leningen die ze gaf, weet een bank of ze werden terugbetaald. Welke van deze getallen kan ze per groep uitrekenen uit haar eigen eerdere beslissingen?",
        options={
            "right": "Van wie een lening kreeg, het deel dat terugbetaalde",
            "recall": "Van iedereen die zou terugbetalen, het deel met een lening",
            "fpr": "Van iedereen die niet zou terugbetalen, het deel met een lening",
        },
        correct="right",
        why={
            "right": "Of een lening werd terugbetaald, weet de bank van iedereen met een lening, dus kan ze per groep het deel tellen dat terugbetaalde: de precision, het getal van de outcome test.",
            "recall": "Daarvoor heeft de bank iedereen nodig die zou terugbetalen, ook de mensen die ze afwees. Of die zouden hebben terugbetaald, heeft ze nooit gezien: het selective labels problem.",
            "fpr": "Daarvoor heeft de bank iedereen nodig die niet zou terugbetalen, ook de mensen die ze afwees. Of die zouden hebben terugbetaald, heeft ze nooit gezien: het selective labels problem.",
        },
    )
    choice_labels
    return (choice_labels,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) vroeg het recht of een protected attribute het verschil maakte; een praktijktest gaat precies dat na, met paren die alleen in dat attribute verschillen. In [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) had je het model en de data; hier zag je beslissingen van buitenaf. De outcome test vergelijkt de precision per groep van de mensen die een ja kregen; door infra-marginality kan een gelijke lat toch ongelijke succespercentages geven, en door selective labels zijn gelijke fouten achteraf moeilijk te controleren. De volgende les, *De data aanpassen: reweighing*, begint aan de andere kant: wat je kunt doen als een model groepen ongelijk behandelt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_labels, result_hits, result_ratio):
    L.wrap_up(result_ratio, result_hits, choice_labels)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
