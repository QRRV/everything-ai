# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-automate-work. Route: course/route/lessen/bouwen.yaml. Checks, de eventlogs en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-automate-work", "nl")
    C = L.checks   # de verzonnen eventlogs van de retouren van een webshop, en de directly-follows graph
    F = C.facts()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl), en één over [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="Een model deed een taak in alle acht pogingen goed. Een andere taak, die er net zo makkelijk uitziet, deed het elke keer fout. Wat laat dat zien?",
        options={
            "luck": "Dat acht pogingen te weinig zijn om iets te zeggen",
            "jagged": "Dat de frontier grillig is: je zoekt het per taak uit, door te proberen",
            "broken": "Dat het model kapot is, en je het helemaal niet moet gebruiken",
        },
        correct="jagged",
        why={
            "jagged": "Twee taken die er even moeilijk uitzien, kunnen aan verschillende kanten van de frontier liggen. Dus probeer je elke taak op gevallen die je kent.",
            "luck": "Acht van de acht tegen nul van de acht is geen geluk. Wat zegt het over de twee taken? Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
            "broken": "Hetzelfde model deed de eerste taak elke keer goed. Hoe heet de grens tussen de twee taken? Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question='Een tabel met verkopen heeft de kolommen `shop` en `amount`. Wat geeft `df.groupby("shop")["amount"].sum()`?',
        options={
            "total": "Eén totaal: alle bedragen bij elkaar opgeteld",
            "rows": "Hoeveel verkopen elke winkel had",
            "per": "Het totale bedrag van elke winkel, één getal per winkel",
        },
        correct="per",
        why={
            "per": "`groupby(\"shop\")` splitst de rijen per winkel, `[\"amount\"]` neemt hun bedragen, en `.sum()` geeft één totaal per winkel.",
            "total": "Dat is `df[\"amount\"].sum()`, zonder groepen. Wat voegt `groupby` toe? Zie [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "rows": "Rijen tellen is `.size()` of `.count()`. Wat doet `.sum()` met de bedragen van elke winkel? Zie [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel kantoorwerk bestaat uit dezelfde stappen, steeds opnieuw: de gegevens uit een mail overnemen in een systeem, een label sturen, een formulier controleren, betalen. Software kan die stappen overnemen, en met AI kan hij zelfs de mail lezen. Maar software doet precies wat je hem opdroeg, ook als het werk niet gaat zoals je dacht. Dus voordat je automatiseert, kijk je hoe het werk echt gaat.

    **Een voorbeeld uit het dagelijks leven.** Een school wil een nieuw pad over het gras aanleggen, van het hek naar de deur. Een paar weken van tevoren kijkt de conciërge waar het gras plat is gelopen: de kinderen lopen niet van het hek naar de deur, maar vanaf de fietsenstalling. Dat is alleen een vergelijking; hieronder zie je hoe je de olifantenpaadjes van het werk in een log vindt.
    """)
    return


@app.cell(hide_code=True)
def predict_as_drawn():
    guess_as_drawn = L.guess(
        "as-drawn",
        question=f"Een webshop tekende zijn retouren als vijf stappen: retour aangevraagd, label gestuurd, pakket ontvangen, pakket gecontroleerd, geld teruggestort. Het systeem houdt van elke stap een log bij. Hoeveel van de laatste {F['returns']} retouren doorliepen precies die vijf stappen?",
        options={
            "all": "Bijna allemaal",
            "half": "Iets meer dan de helft",
            "few": "Bijna geen",
        },
        reveal=f"**Iets meer dan de helft: {F['as_drawn']} van de {F['returns']}.** Bij {F['no_label']} retouren stuurde de klant het pakket zonder het label; bij {F['photo_cases']} vroeg de controle de klant om foto's en werd er opnieuw gecontroleerd, soms meer dan eens; {F['refused']} werden geweigerd; en bij {F['twice']} retouren werd het geld twee keer teruggestort. Niets daarvan staat in de tekening. Het log is door de course verzonnen, maar logs van echte bedrijven laten hetzelfde zien: werk gaat meer kanten op dan iemand tekende.",
    )
    guess_as_drawn
    return (guess_as_drawn,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **no-code** | een automatisering bouwen door blokken aan elkaar te klikken in plaats van code te schrijven, zoals *als er een mail met een factuur binnenkomt, bewaar de bijlage en zet een rij in een spreadsheet* |
    | **robotic process automation (RPA)** | software die vaste klikken en toetsaanslagen herhaalt in programma's die je al hebt, zoals een mens dat zou doen; ze volgt regels en leert niets |
    | **persoonlijke agent** | een assistent die stappen voor je zet, zoals je mail lezen en een afspraak in je agenda zetten, en die stappen zelf kiest |
    | **eventlog** | een tabel die een systeem van het werk bijhoudt: één rij voor elke stap, met de case waar hij bij hoort, wat er gebeurde (de activiteit) en wanneer |
    | **process mining** | uit een eventlog afleiden hoe een proces echt verloopt: welke stappen, in welke volgorde, en waar de tijd heen gaat |
    | **directly-follows graph** | een plaatje van een proces uit zijn eventlog: een vak voor elke activiteit, en een pijl van de ene naar de andere voor elke keer dat die er in een case direct op volgde |

    ### Drie manieren om te automatiseren

    | Manier | Wat het doet | Goed voor |
    |---|---|---|
    | **no-code**-flow | een trigger en vaste stappen, gebouwd uit blokken in tools zoals Zapier, Make, n8n of Power Automate; veel ervan hebben nu een stap die een language model iets vraagt | werk dat altijd hetzelfde gaat, tussen programma's die een koppeling met de tool hebben |
    | **RPA** | een softwarerobot die door de schermen van een programma klikt, vaak een oud programma zonder koppeling; met machine learning erbij om formulieren te lezen of mails te sorteren heet het *intelligent process automation* | vast werk in programma's waar niets anders bij kan |
    | **persoonlijke agent** | een assistent die zijn eigen stappen kiest, met toegang tot je mail, agenda of bestanden; een latere les laat zien hoe een agent werkt | werk dat elke keer anders is, waarbij je nakijkt wat hij deed |

    De eerste twee doen elke keer precies wat je bouwde. De agent kan met het onverwachte omgaan, en kan ook zelf iets onverwachts doen.

    ### Process mining

    Elk systeem dat cases, bestellingen of retouren afhandelt, schrijft op wat er gebeurde: een eventlog. Process mining leest het. Het eerste plaatje is de directly-follows graph: voor elke twee stappen van één case die direct op elkaar volgen, telt hij het paar. Een dikke pijl is een weg die veel cases gaan; een dunne pijl die niet in de tekening staat, is een omweg; een pijl terug is werk dat opnieuw gedaan wordt. Process mining is voor een groot deel aan de TU Eindhoven ontstaan. *Task mining* doet hetzelfde voor de klikken van één persoon achter een computer.

    Het log van deze les is in het Engels, zoals veel logs van bedrijven. De activiteiten: *Return requested* (retour aangevraagd), *Label sent* (label gestuurd), *Package received* (pakket ontvangen), *Package checked* (pakket gecontroleerd), *Photos asked* (foto's gevraagd), *Refund paid* (geld teruggestort) en *Return refused* (retour geweigerd).
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een directly-follows graph

    1. **Zet het log op volgorde:** per case, en binnen een case op tijd. (Het log van de course is dat al.)
    2. **Ga twee aan twee langs de rijen:** rij `i` en rij `i + 1`.
    3. **Dezelfde case?** Dan vormen de twee stappen een paar: `(steps[i], steps[i + 1])`. Begint er een nieuwe case? Dan is er geen paar: de laatste stap van de ene retour leidt niet naar de eerste van de volgende.
    4. **Tel elk paar,** en teken er een pijl voor, zo dik als het aantal.
    5. **Vergelijk met de tekening:** welke pijlen zijn extra, welke gaan terug, welke zouden er nooit mogen zijn? En waar gaat de tijd heen? Pas dan beslis je wat je automatiseert.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een log van drie retouren, al op volgorde:

    | case | activity | day |
    |---|---|---|
    | R001 | Return requested | 0 |
    | R001 | Label sent | 1 |
    | R001 | Package received | 4 |
    | R002 | Return requested | 0 |
    | R002 | Package received | 3 |
    | R003 | Return requested | 0 |
    | R003 | Label sent | 2 |
    | R003 | Package received | 5 |

    1. **Op volgorde:** ja, per case, dan op dag.
    2. **Twee aan twee:** rij 0 en 1, rij 1 en 2, enzovoort, tot rij 6 en 7: zeven keer.
    3. **Dezelfde case?** Rij 0 en 1: allebei R001, dus het paar *(Return requested, Label sent)*. Rij 2 en 3: R001 en R002, een nieuwe case: geen paar. In totaal vijf paren:
       *(Return requested, Label sent)*, *(Label sent, Package received)*, *(Return requested, Package received)*, *(Return requested, Label sent)*, *(Label sent, Package received)*.
    4. **Tel:** *Return requested → Label sent* 2 keer, *Label sent → Package received* 2 keer, *Return requested → Package received* één keer.
    5. **Vergelijk:** R002 ging van de aanvraag meteen naar het pakket: de klant wachtte niet op het label. Als een robot altijd op *Label sent* wacht, loopt hij vast op retouren zoals R002.
    """)
    return


@app.cell(hide_code=True)
def text_which_automation():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke manier van automatiseren?
    """)
    return


@app.cell(hide_code=True)
def choice_which_automation_cell():
    choice_which_automation = L.choice(
        "which-automation",
        question="Elke ochtend neemt iemand bij een verzekeraar de gegevens van nieuwe schadeclaims over uit een oud programma, dat nergens een koppeling mee heeft, in een spreadsheet: dezelfde twaalf velden, in dezelfde volgorde. Welke manier van automatiseren past het best?",
        options={
            "agent": "Een persoonlijke agent, die zelf beslist wat hij met elke claim doet",
            "rpa": "RPA: een softwarerobot die door het oude programma klikt en de velden overneemt",
            "nocode": "Een no-code-flow die het oude programma start als er een claim binnenkomt",
        },
        correct="rpa",
        why={
            "rpa": "Vaste stappen in een programma waar niets anders bij kan: daar is RPA voor. De robot doet elke ochtend precies hetzelfde, en dat wil je hier.",
            "agent": "Een agent kiest zijn eigen stappen, dus hij kan ook verkeerde kiezen. Veranderen deze stappen ooit? Kijk naar de tabel met de drie manieren.",
            "nocode": "Een no-code-flow heeft een koppeling met het programma nodig, en dit programma heeft er geen. Kijk naar de tabel met de drie manieren.",
        },
    )
    choice_which_automation
    return (choice_which_automation,)


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(rf"""
        ### Oefening 2. De paren van het log

        Het vak leest het log van de webshop met {F['returns']} retouren als `log`, met de kolommen `case`, `activity` en `day`, op volgorde van case en dag: {C.number(F['rows'], 0, L.lang)} rijen. `cases` en `steps` zijn de kolommen als lijsten. Maak de loop af: voeg voor elke twee rijen die op elkaar volgen en bij dezelfde retour horen het paar `(steps[i], steps[i + 1])` aan `pairs` toe. Vervang de `...`. De laatste regel is `pairs`, zodat de check het ziet; onder het vak tekent de course je directly-follows graph. De check probeert ook een log van 40 andere retouren.
        """),
        L.task("Maak `pairs`: elke twee stappen van één retour die direct op elkaar volgen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("pairs", lines=9)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "pairs",
        [
            "Stap 2 en 3 van het recept: een loop over `i`, en een test of rij `i` en `i + 1` bij dezelfde retour horen.",
            "`range(len(steps) - 1)` laat `i` tot de op één na laatste rij gaan, zodat `i + 1` altijd een rij is. De test vergelijkt `cases[i]` met `cases[i + 1]`.",
            "`for i in range(___):` met daarin `if ___ == ___:` en daarin weer `pairs.append((___, ___))`.",
        ],
        why=f"`for i in range(len(steps) - 1):` gaat twee aan twee langs de rijen; `if cases[i] == cases[i + 1]:` houdt alleen twee rijen van dezelfde retour over; `pairs.append((steps[i], steps[i + 1]))` voegt hun stappen als één paar toe. De {C.number(F['rows'], 0, L.lang)} rijen van {F['returns']} retouren geven {C.number(F['pairs'], 0, L.lang)} paren: voor elke retour één minder dan zijn rijen.",
    )
    return


@app.cell(hide_code=True)
def text_graph_after():
    mo.vstack([
        mo.Html(C.course_graph(L.lang)),
        mo.md(rf"""
        Dit is de directly-follows graph van alle {F['returns']} retouren. De dikste pijlen zijn de getekende weg. Maar {F['photos']} keer volgde op een controle *Photos asked*, en daarna telkens weer een controle: werk dat dubbel gedaan werd, bij {F['photo_cases']} retouren. {F['no_label']} retouren gingen van de aanvraag meteen naar het pakket. En een paar pijlen zijn dun, maar niet onschuldig.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_took():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Waar de tijd heen gaat

        Elke retour begint op dag 0, dus de dag van zijn laatste stap is hoeveel dagen hij duurde. Maak `took`: voor elke retour de grootste `day`. Vervang de `...`. De laatste regel is `took`, zodat de check het ziet. Onder het vak splitst de course je dagen in de retouren waarbij foto's werden gevraagd en de rest.
        """),
        L.task("Maak `took`: de grootste dag van elke retour."),
    ])
    return


@app.cell(hide_code=True)
def exercise_took():
    box_took = L.code_box("took", lines=6)
    box_took
    return (box_took,)


@app.cell(hide_code=True)
def check_took(box_took):
    result_took = L.check("took", box_took.value)
    result_took
    return (result_took,)


@app.cell(hide_code=True)
def hints_took():
    L.hints(
        "took",
        [
            "Eén getal per retour: dat is `groupby`, zoals in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "Groepeer op `case`, neem de kolom `day`, en vraag om de grootste waarde, niet om het gemiddelde.",
            "`took = log.groupby(___)[___].max()`.",
        ],
        why=f"`log.groupby(\"case\")` splitst de rijen per retour, `[\"day\"]` neemt hun dagen, en `.max()` geeft de laatste: hoe lang elke retour duurde. De retouren waarbij foto's werden gevraagd, duurden gemiddeld {C.number(F['days_photos'], 1, L.lang)} dagen, de andere {C.number(F['days_other'], 1, L.lang)}.",
    )
    return


@app.cell(hide_code=True)
def text_took_after():
    mo.md(rf"""
    De retouren waarbij foto's werden gevraagd, duurden gemiddeld {C.number(F['days_photos'], 1, L.lang)} dagen; de andere {C.number(F['days_other'], 1, L.lang)}. In die lus gaat de tijd zitten. Een robot die de gegevens sneller overneemt, verandert daar niets aan; de foto's meteen bij de aanvraag vragen, voordat het pakket wordt verstuurd, misschien wel.
    """)
    return


@app.cell(hide_code=True)
def text_never_twice():
    mo.md(r"""
    ### Oefening 4. Een pijl die er niet hoort
    """)
    return


@app.cell(hide_code=True)
def choice_never_twice_cell():
    choice_never_twice = L.choice(
        "never-twice",
        question="Welke pijl in de directly-follows graph van oefening 2 laat een fout zien die de webshop geld kost?",
        options={
            "photos": "Package checked → Photos asked",
            "twice": "Refund paid → Refund paid",
            "nolabel": "Return requested → Package received",
        },
        correct="twice",
        why={
            "twice": f"Geld teruggestort vlak na geld teruggestort: {F['twice']} klanten kregen hun geld twee keer terug. Een dunne pijl, dus makkelijk over het hoofd te zien, en niemand tekende hem. Voordat je de betalingen automatiseert, zoek je uit waarom.",
            "photos": "Foto's vragen kost tijd, geen geld dat per ongeluk wordt uitbetaald. Welke pijl betaalt iets meer dan eens?",
            "nolabel": "De klant stuurde het pakket zonder het label: onschuldig, het pakket kwam toch binnen. Welke pijl betaalt iets meer dan eens?",
        },
    )
    choice_never_twice
    return (choice_never_twice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl) vatte je een tabel per groep samen; hier is elke retour een groep, en telt ook de volgorde van zijn rijen. In [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl) deelde je je eigen werk op in taken; process mining doet dat voor het werk van een heel bedrijf, uit zijn log. Later in het domein *Bouwen met AI* bouw je zelf een agent, en zie je waarom die controles nodig heeft die een vaste flow niet nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never_twice, choice_which_automation, result_pairs, result_took):
    L.wrap_up(choice_which_automation, result_pairs, result_took, choice_never_twice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
