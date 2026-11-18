# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-abstraction-traps. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-abstraction-traps", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_behind():
    quick_behind = L.choice(
        "quick-behind",
        question="Drie groepen hebben een word error rate van 0,30, 0,20 en 0,26. Een team sprak, nog voordat het keek, een grens af van 0,05 boven de beste groep. Welke groepen blijven achter?",
        options={
            "only": "Alleen 0,30",
            "right": "0,30 en 0,26",
            "all": "Alle drie, ook 0,20",
        },
        correct="right",
        why={
            "only": "Ook 0,26 ligt boven de lijn: de beste groep zit op 0,20, dus de lijn ligt op 0,20 + 0,05 = 0,25. Zie [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl).",
            "right": "De beste groep zit op 0,20, dus de lijn ligt op 0,25: 0,30 en 0,26 liggen erboven.",
            "all": "0,20 is de beste groep zelf, en die kan niet boven de eigen word error rate plus de grens liggen. De lijn ligt op 0,25. Zie [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl).",
        },
    )
    quick_behind
    return (quick_behind,)


@app.cell(hide_code=True)
def quick_check_reactive():
    quick_reactive = L.choice(
        "quick-reactive",
        question="Welke van deze is reactief, in de woorden van Rakova en collega's?",
        options={
            "right": "Repareren na een nieuwsbericht",
            "review": "Een review op vaste momenten vóór de lancering",
            "board": "Een commissie die een systeem op tijd kan stoppen",
        },
        correct="right",
        why={
            "right": "Pas handelen na slechte publiciteit is reactief: de schade heeft mensen dan al bereikt. Vaste reviews en een commissie die een systeem kan stoppen zijn proactief.",
            "review": "Vaste momenten om een systeem vooraf te beoordelen maken het werk juist proactief. Zie [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl).",
            "board": "Sommige proactieve organisaties hebben een interne commissie die erop toeziet; een commissie die een systeem kan stoppen, handelt vóór de schade, niet erna. Zie [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl).",
        },
    )
    quick_reactive
    return (quick_reactive,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2011 nam de Amerikaanse staat Kentucky een wet aan, House Bill 463, die een risicotaxatie vóór het proces verplicht maakte: een tool die het risico van elke verdachte inschat voordat de zaak voorkomt. Voor verdachten met een laag of gemiddeld risico stelde de wet als standaard: meteen vrijlaten, zonder borgsom. Negentig procent van de verdachten kreeg een laag of gemiddeld risico. Megan Stevenson (2018) bestudeerde meer dan een miljoen strafzaken. Als rechters de standaard hadden gevolgd, waren 90 procent meteen zonder borgsom vrijgelaten; in de praktijk was dat 29 procent, bij de eerste beslissing over de borgtocht. Rechters negeerden de standaard in meer dan twee derde van de zaken, en gingen na verloop van tijd terug naar hoe ze het eerder deden. De wet leverde maar een verwaarloosbare toename van vrijlating op. En het verschil tussen witte en zwarte verdachten in vrijlating zonder borgsom sprong van ongeveer 2 naar 10 procentpunten. Niet omdat de risicotaxatie meer bias naar ras had dan de rechters zelf, schrijft Stevenson: rechters in verschillende regio's reageerden verschillend op de wet. Een fairness-check op de scores van de tool had dit verschil niet gezien.

    **Een voorbeeld uit het dagelijks leven.** Een school geeft elke leerling dezelfde laptop. Kijk je alleen naar de laptops, dan is het zo gelijk als het maar kan. Maar een leerling zonder internet thuis kan er minder huiswerk mee maken dan een leerling met internet. Of het plan voor iedereen goed uitpakt, hangt ook af van de huizen, niet alleen van de laptops.
    """)
    return


@app.cell(hide_code=True)
def predict_release():
    guess_release = L.guess(
        "release",
        question="De tool in Kentucky gaf 90 procent van de verdachten een laag of gemiddeld risico, en voor hen stelde de wet als standaard vrijlating zonder borgsom. Welk deel werd bij de eerste beslissing over de borgtocht zonder borgsom vrijgelaten?",
        options={
            "most": "Bijna 90 procent",
            "two": "Ongeveer twee op de drie",
            "right": "Minder dan een derde",
        },
        reveal="**Minder dan een derde: 29 procent.** Als rechters de standaard hadden gevolgd, waren 90 procent meteen zonder borgsom vrijgelaten. Rechters negeerden de standaard in meer dan twee derde van de zaken, en de verandering vlak na de wet vervaagde toen ze teruggingen naar hoe ze het eerder deden.",
    )
    guess_release
    return (guess_release,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het systeem, niet de tool.** Andrew Selbst, danah boyd, Sorelle Friedler, Suresh Venkatasubramanian en Janet Vertesi (2019) betogen dat abstractie en modulair ontwerp, basisideeën van de informatica, technisch werk aan fairness ondoeltreffend, onjuist en soms gevaarlijk misleidend kunnen maken zodra het de samenleving rond een beslissysteem binnengaat. Fairness en rechtvaardigheid, schrijven ze, zijn eigenschappen van sociale en juridische systemen zoals werk en strafrecht, niet van de technische tools daarbinnen. Het vakgebied science and technology studies noemt een systeem van technische en sociale delen samen een *sociotechnisch systeem*.

        **Drie kaders.** Waar je de grens trekt, bepaalt wat je kunt zien. In het *algoritmische kader* zijn er alleen de keuzes hoe je de data weergeeft en wat je labelt; daar is fairness niet eens te definiëren. Het *datakader* voegt de invoer en uitvoer van het algoritme toe: voorspellingen en uitkomsten per groep, waar de maten van deze module naar kijken. Het *sociotechnische kader* haalt de mensen en instellingen die op de uitvoer handelen binnen de grens, zodat fairness een eigenschap van het geheel wordt, van begin tot eind. De uitvoer van een tool vóór het proces is een risico; de uitvoer van de rechtbank is dat een verdachte vrijkomt, vrijkomt op borgtocht, of vast blijft zitten.

        **Vijf valkuilen.** Selbst en collega's noemen vijf *abstraction traps*, valkuilen van abstractie: manieren om te falen doordat het technische en het sociale deel niet samen bekeken worden.

        - De *framing trap*: niet het hele systeem modelleren waarover fairness moet gelden. Rechters nemen de aanbeveling van een tool niet steeds mee: sommigen leunen er misschien te veel op, uit automation bias, anderen wijken er misschien op een vooringenomen manier van af of negeren haar. Een garantie over alleen de tool klopt dan mogelijk niet.
        - De *portability trap*: een oplossing voor de ene sociale context naar een andere verhuizen. De cultuur van de informatica waardeert draagbaarheid en eist die vaak, dus dezelfde classifier lijkt te passen bij recidive, wanbetaling op leningen en werving. Zelfs tussen rechtsgebieden kunnen de lokale zorgen over fairness zo verschillen dat een systeem niet goed over te zetten is.
        - De *formalism trap*: missen dat fairness procedureel, afhankelijk van de context en betwistbaar is, en dat geen formule dat beslecht. Bij een tool die cv's selecteert, weegt een false negative misschien minder dan een false positive, omdat het sollicitatiegesprek later nog filtert; in het strafrecht houden false positives mensen achter de tralies. En de 80%-regel is maar de eerste stap van disparate impact in het Amerikaanse arbeidsrecht, waarna de vraag komt of een test met het werk te maken heeft en past bij wat het bedrijf nodig heeft.
        - De *ripple effect trap*: missen hoe een nieuwe tool de mensen en de waarden verandert van het systeem waar hij in komt. Rechters volgen een score misschien omdat die grondiger oogt, of verzetten zich ertegen. Een tool kan stilletjes wat meetbaar is voorop zetten: iemand vasthouden zodat die geen misdaad kan plegen is maar een van minstens zeven redenen om te straffen.
        - De *solutionism trap*: niet zien dat de beste oplossing voor een probleem misschien geen technologie is. Wie bij de technologie begint, krijgt nooit de kans om te vragen of die technologie er wel moet komen.

        **Geen reden om te stoppen met meten.** De uitweg, schrijven Selbst en collega's, is de stap van een aanpak die oplossingen zoekt naar een aanpak die een proces volgt, met een grens die ook mensen, instellingen en hun omgang met elkaar omvat. Ze stellen voor de valkuilen langs te gaan, 'misschien in omgekeerde volgorde': eerst of een technische oplossing wel bij de situatie past, als laatste of het kader de mensen en de data bevat die ertoe doen. Bij tools vóór het proces betekent dat: de tool afwegen tegen beleid dat hem overbodig maakt, zoals mensen die van een geweldloos misdrijf verdacht worden standaard niet vasthouden, en een proef draaien om te zien of rechters hem volgen. Dat betekent niet dat fairness in machine learning niet in de praktijk te brengen is, schrijven ze, maar wel dat niet alle problemen met technologie op te lossen zijn, of zo opgelost moeten worden. Ben Green (2020) beschrijft hetzelfde uit de praktijk: risicotaxaties verplaatsen de beslisruimte naar andere mensen en plekken, in plaats van haar weg te nemen; rechters in Cook County, Illinois, weken in 85 procent van de gevallen af van de risicotaxatie en lieten veel minder verdachten vrij dan aanbevolen.

        **Solutionisme.** Evgeny Morozov (2013) leende het woord uit de architectuur en de stedenbouw. Voor Morozov maakt solutionisme van ingewikkelde sociale situaties nette problemen met een vaste, berekenbare oplossing, terwijl vaak niet de oplossing betwist is, maar hoe het probleem zelf is gedefinieerd. Selbst en collega's en Ben Green schrijven 'technological solutionism'; *techno-solutionisme* is een gangbare verkorting. De valkuil van Selbst en collega's is milder dan dat geloof: niet dat mensen denken dat een technische oplossing altijd werkt, maar dat ze niet zien dat de beste oplossing misschien niet technisch is.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **sociotechnisch systeem** | een systeem gezien als techniek plus de mensen, organisaties en regels eromheen |
        | **abstraction trap** | een manier waarop een methode voor fairness tekortschiet doordat die een deel weglaat van het systeem waarin ze werkt; Selbst en collega's noemen er vijf |
        | **framing**, **portability**, **formalism**, **ripple effect** en **solutionism trap** | de vijf: niet het hele systeem modelleren; een oplossing naar een andere context verhuizen; fairness in één formule vangen; missen hoe een tool de mensen eromheen verandert; niet zien dat de beste oplossing misschien niet technisch is |
        | **techno-solutionisme** | kort voor technological solutionism: het geloof dat maatschappelijke problemen het best met technologie op te lossen zijn, ook als de oorzaken politiek of sociaal zijn |
        """),
        L.watch_out("De valkuilen zijn geen argument tegen meten. Ze zeggen wat je nog meer moet meten: de beslissingen die mensen met een tool nemen, niet alleen de scores. En ze zeggen wat je eerst moet vragen: of een tool wel het antwoord is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Ga, voordat je een fairness-oplossing bouwt of beoordeelt, de valkuilen langs; Selbst en collega's stellen voor om dat misschien in deze volgorde te doen, de omgekeerde van hun eigen:

    1. Solutionism: zou een verandering die niet technisch is, zoals een nieuwe regel of meer personeel, beter werken? Zo ja, stop dan hier.
    2. Ripple effect: hoe gaan de mensen en de organisatie zich gedragen als de tool er eenmaal is?
    3. Formalism: welke soort fairness past bij deze context, en wie mag daarover meepraten?
    4. Portability: is de tool, of de drempel ervan, gemaakt voor deze plek en deze mensen?
    5. Framing: trek de grens om de tool en de mensen die op de uitvoer handelen, en meet hun beslissingen, niet alleen de scores van de tool.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Tien zaken bij één rechtbank, verzonnen door de course: wat de tool aanbeveelt, en wat de rechter besliste. `"release"` is vrijlating zonder geld; `"bond"` is een borgsom. De loop telt hoe vaak elk van beide vrijlating zegt. Voer de cel uit.
    """)
    return


@app.cell
def example_court():
    # de aanbeveling van de tool en de beslissing van de rechter in tien zaken
    _cases = [
        {"tool": "release", "judge": "release"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "release"},
        {"tool": "bond", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "release"},
        {"tool": "release", "judge": "bond"},
        {"tool": "release", "judge": "bond"},
    ]

    _tool = 0
    _judge = 0
    for _case in _cases:
        if _case["tool"] == "release":
            _tool = _tool + 1
        if _case["judge"] == "release":
            _judge = _judge + 1
    print("tool", _tool / len(_cases))
    print("rechters", _judge / len(_cases))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tool zou 9 van de 10 vrijlaten; de rechters lieten er 3 vrij. Naar de scores gemeten laat deze tool bijna iedereen vrij; wat er met de verdachten gebeurt, beslissen de rechters. Gebeurt dat voor twee groepen op dezelfde manier? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De tool en de rechters

        `cases` bevat 16 zaken van één rechtbank, verzonnen door de course naar het patroon dat Stevenson in Kentucky vond: elk met de groep van de verdachte, A of B, het district (`"rural"` voor het platteland, `"urban"` voor de stad), de aanbeveling van de tool en de beslissing van de rechter. Reken `share` uit: het deel van de zaken van `group` waarin `key` `"release"` zegt. Tel de zaken van de groep in `total`, en de zaken daarvan met `case[key] == "release"` in `released`; dan `share = released / total`. Met `key = "judge"` is `case[key]` hetzelfde als `case["judge"]`. De check probeert ook de tool, en groep A.
        """),
        L.task("Reken `share` uit: het deel van de zaken van `group` waarin `key` vrijlating zegt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=33)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Twee tellingen in één loop over de zaken: de zaken van de groep, en de zaken daarvan die `key` wil vrijlaten.",
            "Onder `for case in cases:` telt `if case[\"group\"] == group:` 1 op bij `total`; daarbinnen telt `if case[key] == \"release\":` 1 op bij `released`.",
            "Na de loop: `share = ___ / total`.",
        ],
        why="`total = 0` en `released = 0`; dan `for case in cases:` met `if case[\"group\"] == group:` en `total = total + 1`, en daaronder `if case[key] == \"release\":` met `released = released + 1`. Na de loop `share = released / total`. Dat geeft `0.375`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_trap():
    mo.md(r"""
    ### Oefening 2. Welke valkuil?
    """)
    return


@app.cell(hide_code=True)
def choice_trap_cell():
    choice_trap = L.choice(
        "trap",
        question="Een bedrijf bouwde een tool die sollicitaties selecteert, en die voldoet daar aan een criterium voor fairness. Een rechtbank koopt hem, traint hem op de eigen dossiers, en voorspelt er nu mee of verdachten op hun zitting verschijnen. Welke valkuil is dit?",
        options={
            "formalism": "De formalism trap",
            "right": "De portability trap",
            "ripple": "De ripple effect trap",
        },
        correct="right",
        why={
            "formalism": "De formalism trap gaat over fairness in één formule vangen. Hier verhuist een tool voor werving naar een rechtbank. Zie *Vijf valkuilen* in *Het idee*.",
            "right": "Een oplossing voor de ene sociale situatie, werving, verhuist naar een andere, een rechtbank. Zelfs tussen rechtbanken, schrijven Selbst en collega's, kunnen de lokale zorgen over fairness zo verschillen dat een systeem niet goed over te zetten is. De tool trainen op de eigen dossiers verandert de data, niet de sociale situatie rond de beslissing.",
            "ripple": "De ripple effect trap gaat over hoe een tool de mensen en waarden eromheen verandert als hij eenmaal gebruikt wordt. Hier begint het probleem eerder: een tool voor werving verhuist naar een rechtbank. Zie *Vijf valkuilen* in *Het idee*.",
        },
    )
    choice_trap
    return (choice_trap,)


@app.cell(hide_code=True)
def text_exercise_overrides():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Waar komt het verschil vandaan?

        De tool wil beide groepen even vaak vrijlaten, de rechters niet. Reken `overrides` uit: een dict van elk district in `counties` naar het deel van de zaken daar waarin de rechter anders besliste dan de tool, `case["judge"] != case["tool"]`. Begin met `overrides = {}`. Zet voor elk district `total = 0` en `changed = 0`, loop over de zaken, en tel de zaken van het district in `total` en de zaken daarvan waarin de rechter anders besliste in `changed`. Dan `overrides[county] = changed / total`. De check probeert ook een andere rechtbank.
        """),
        L.task("Reken `overrides` uit: per district het deel van de zaken waarin de rechter anders besliste dan de tool."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overrides():
    box_overrides = L.code_box("overrides", lines=32)
    box_overrides
    return (box_overrides,)


@app.cell(hide_code=True)
def check_overrides(box_overrides):
    result_overrides = L.check("overrides", box_overrides.value)
    result_overrides
    return (result_overrides,)


@app.cell(hide_code=True)
def hints_overrides():
    L.hints(
        "overrides",
        [
            "Net als in oefening 1, met nog een loop eromheen: voor elk district twee tellingen over de zaken.",
            "Onder `for county in counties:` zet je `total = 0` en `changed = 0`; dan `for case in cases:` met `if case[\"county\"] == county:` dat 1 optelt bij `total`, en daarbinnen `if case[\"judge\"] != case[\"tool\"]:` dat 1 optelt bij `changed`.",
            "Na de binnenste loop, op de hoogte van `for case`: `overrides[___] = changed / total`.",
        ],
        why="`overrides = {}`, dan `for county in counties:` met `total = 0` en `changed = 0`, en `for case in cases:` met `if case[\"county\"] == county:`, `total = total + 1`, en daaronder `if case[\"judge\"] != case[\"tool\"]:` met `changed = changed + 1`. Na de binnenste loop `overrides[county] = changed / total`. Dat geeft `{'rural': 0.0, 'urban': 0.5}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Oefening 4. Wat komt eerst?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="Selbst en collega's stellen voor de vijf valkuilen langs te gaan, misschien in omgekeerde volgorde. Welke vraag stelt een team dan eerst over een nieuwe risicotool voor een rechtbank?",
        options={
            "rates": "Is de error rate voor elke groep even hoog?",
            "right": "Is een tool hier wel het goede antwoord?",
            "elsewhere": "Werkte hij waar hij gemaakt is?",
        },
        correct="right",
        why={
            "rates": "Error rates per groep komen later, bij de formalism en de framing trap. In de omgekeerde volgorde is de eerste vraag of een tool wel bij de situatie past. Zie *Geen reden om te stoppen met meten* in *Het idee*.",
            "right": "De omgekeerde volgorde begint bij de solutionism trap: of een technische oplossing wel bij de situatie past. Bij tools vóór het proces betekent dat: de tool afwegen tegen beleid dat hem overbodig maakt.",
            "elsewhere": "Dat is de portability trap, de vierde in de omgekeerde volgorde. De eerste vraag is of een tool wel bij de situatie past. Zie *Geen reden om te stoppen met meten* in *Het idee*.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Fairness in een organisatie](/course/open/rai-fairness-in-organizations?lang=nl) ging het erom wanneer een team naar fairness kijkt. Hier gaat het erom waar het naar kijkt. De maten van deze module meten een tool in het datakader; het verschil in oefening 1 werd pas zichtbaar toen ook de beslissingen van de rechters meetelden, en oefening 3 vond het in de districten, niet in de tool. Een fairness review hoort de grens te trekken om de beslissingen die mensen met een tool nemen, en eerst te vragen of een tool wel het antwoord is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, choice_trap, result_overrides, result_share):
    L.wrap_up(result_share, choice_trap, result_overrides, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
