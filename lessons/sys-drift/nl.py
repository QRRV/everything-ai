# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-drift. Route: course/route/lessen/systemen.yaml. Checks, het model, de uren van 2011 en 2012 en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-drift", "nl")
    C = L.checks   # het model van het project van de course, de antwoorden voor elk uur van 2011 en 2012, de bins en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe ML-systemen falen](/course/open/sys-ml-failures?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Een check vergelijkt elke input van een verzoek met de laagste en de hoogste waarde die de training zag. Welke storing laat de check door?",
        options={
            "degrees": "Een app die de temperatuur in graden stuurt, 31,2 in plaats van 0,76",
            "minus": "Een vochtigheidssensor die −1 stuurt als er geen meting is",
            "right": "Een hittegolf van 38,5 graden, met zulke uren in de training",
            "kmh": "Een stap stroomopwaarts die de wind in km/u stuurt",
        },
        correct="right",
        why={
            "right": "38,5 graden ligt binnen het bereik van de training, want de training bevat zulke hete uren, dus laat de check elk verzoek van de hittegolf door, terwijl het model er ver naast zat. De andere drie liggen buiten het bereik. In deze les glipt iets langzamers langs zo'n check: de wereld zelf verschuift.",
            "degrees": "31,2 ligt ver boven de 0,96 van de training. Laat de check dat getal door? Zie [Hoe ML-systemen falen](/course/open/sys-ml-failures?lang=nl).",
            "minus": "−1 ligt onder de 0 van de training. Laat de check dat getal door? Zie [Hoe ML-systemen falen](/course/open/sys-ml-failures?lang=nl).",
            "kmh": "Een wind van 24,0 ligt ver boven de 0,85 van de training. Laat de check dat getal door? Zie [Hoe ML-systemen falen](/course/open/sys-ml-failures?lang=nl).",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="`x` bevat de temperaturen van 1.000 uren. Wat geeft `np.quantile(x, 0.1)`?",
        options={
            "tenth": "De tiende temperatuur van `x`, en dat is `x[9]`",
            "right": "De temperatuur waaronder een tiende van de uren ligt",
            "share": "Het aandeel van de uren dat kouder is dan 0,1 graad",
            "mean": "Een tiende van de gemiddelde temperatuur, `x.mean() / 10`",
        },
        correct="right",
        why={
            "right": "Het 0,1-kwantiel is de waarde waaronder een tiende van de waarden ligt: van 1.000 uren waren er ongeveer 100 kouder. De negen kwantielen 0,1 tot en met 0,9 verdelen de waarden in tien delen, met ongeveer een tiende in elk deel. Dat zijn de bins van deze les.",
            "tenth": "Een kwantiel hangt niet af van de volgorde van `x`: het vraagt hoeveel waarden eronder liggen. Welke waarde heeft een tiende van de waarden onder zich? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "share": "De `0.1` in `np.quantile` is een aandeel, geen temperatuur: het vraagt om een waarde, niet om een aantal. Welke waarde heeft dat aandeel van de uren onder zich? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "mean": "Een kwantiel kijkt naar waar de waarden liggen, niet naar het gemiddelde. Welke waarde heeft een tiende van de uren onder zich? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leert van het verleden, en werkt in de toekomst. De wereld blijft bewegen: mensen verhuizen, prijzen veranderen, er komt een nieuwe app, gewoonten verschuiven. Een model dat vorig jaar klopte kan langzaam slechter worden zonder dat er iets stukgaat: elk verzoek krijgt snel een antwoord met status 200, en elke input ligt binnen wat de training zag. Deze langzame verandering heet *drift*. Om drift te vangen vergelijk je wat er nu binnenkomt met waar het model van leerde, en houd je bij hoe ver het model ernaast zit zodra bekend is wat er echt gebeurde.

    **Een voorbeeld uit het dagelijks leven.** Een plattegrond van een stad van tien jaar geleden opent nog en toont nog straten, en de meeste kloppen. Alleen ontbreken de nieuwe wijken, en stuurt de plattegrond je de verkeerde kant op door een straat met nieuw eenrichtingsverkeer. De plattegrond weet het niet; je merkt het als je verdwaalt.
    """)
    return


@app.cell(hide_code=True)
def predict_year():
    guess_year = L.guess(
        "year",
        question="Het model van het project leerde van de uren van 2011, en zat er over 2011 30,3 fietsen per uur naast. In 2012 beantwoordt de service elk uur met hetzelfde model: niets gaat stuk, en elk verzoek krijgt status 200. Hoe ver zit het model er over 2012 per uur naast?",
        options={
            "same": "Ongeveer 30, net als in 2011: het weer van 2012 lijkt op dat van 2011",
            "bit": "Ongeveer 40: iets meer, want 2012 heeft dagen die het model nog nooit zag",
            "right": "Ongeveer 90: drie keer zoveel",
            "lot": "Ongeveer 300: tien keer zoveel",
        },
        reveal="**Ongeveer 90: drie keer zoveel.** Over 2012 zat het model er 92,5 fietsen per uur naast, tegen 30,3 over 2011. Het weer is niet de oorzaak: 2012 had ongeveer dezelfde temperaturen, vochtigheid en wind als 2011, en het model antwoordde ongeveer evenveel fietsen, 148,7 per uur tegen 144,2. Maar er fietsten veel meer mensen: 234,7 fietsen per uur tegen 143,8, en over het hele jaar 65 procent meer fietsen. Het model antwoordt nog steeds voor de wereld van 2011.",
    )
    guess_year
    return (guess_year,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drift.** *Drift* is een verandering in de loop van de tijd in de data die een model in productie tegenkomt, vergeleken met de data waarvan het model leerde. Er zijn drie plekken om te kijken, elk met een eigen naam:

        - **Data drift**: de inputs veranderen, zoals de temperaturen in de verzoeken. Dat zie je meteen, bij elk verzoek.
        - **Prediction drift**: de antwoorden van het model veranderen. Ook dat zie je meteen, want de service logt elk antwoord.
        - **Label drift**: de echte antwoorden veranderen, hier de verhuurde fietsen. Dat zie je pas als bekend is wat er echt gebeurde: een uur later bij de fietsen, veel later bij veel modellen.

        De afbeelding toont alle drie voor 2012, tegen de uren van de training.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        In 2012 blijven de temperatuur en de antwoorden van het model ongeveer gelijk; alleen de verhuurde fietsen schuiven op. Het weer geeft dezelfde soort verzoeken, en het model antwoordt zoals het in 2011 leerde; maar bij hetzelfde weer worden veel meer fietsen verhuurd. Het verband tussen de inputs en het echte antwoord veranderde. Dat heet **concept drift**: dezelfde input hoort nu een ander antwoord te krijgen. Checks op de inputs en op de antwoorden kunnen concept drift niet zien; alleen wat er echt gebeurde laat het zien.

        **Twee verdelingen vergelijken.** Om te zien of waarden verschoven vergelijk je de verdeling nu met de verdeling bij de training. Verdeel de waarden van de training over tien bins met de negen kwantielen 0,1 tot en met 0,9, zodat elke bin ongeveer een tiende van de training bevat. Tel dan welk aandeel van de nieuwe waarden in elke bin valt. Waar niets veranderde bevat elke bin opnieuw ongeveer een tiende.

        **De population stability index.** De **population stability index**, of **PSI**, maakt van de twee lijsten met aandelen één getal. Neem voor elke bin het aandeel nu min het aandeel bij de training, vermenigvuldig dat met de log van het aandeel nu gedeeld door het aandeel bij de training, en tel de delen van alle bins op:

        $$\text{PSI} = \sum_{\text{bins}} (\text{nu} - \text{training}) \cdot \log \frac{\text{nu}}{\text{training}}$$

        Elk deel is 0 of meer: waar een aandeel steeg zijn beide factoren positief, en waar een aandeel daalde zijn beide negatief. Een bin met een aandeel van 0,1 bij de training en 0,2 nu telt 0,1 × log 2 = 0,069 op. Een vuistregel van banken, waar de PSI al lang de modellen voor leningen bewaakt: onder 0,1 niets aan de hand; van 0,1 tot 0,25 in de gaten houden; boven 0,25 serieuze drift. De verhuurde fietsen in de afbeelding geven over het hele jaar 0,24, net onder 0,25: een heel jaar mengt de seizoenen. Per maand liggen de verhuurde fietsen in de meeste maanden boven 0,25, zoals *Elke maand van 2012* verderop laat zien.

        **Vergelijk gelijk met gelijk.** Een maand van 2012 tegen alle uren van de training ziet er altijd anders uit: juli is zomer, en de training bevat een heel jaar. Vergelijk een maand dus met dezelfde maand in de training.

        **Op grote schaal.** Tools voor model monitoring, zoals Evidently en NannyML, berekenen de PSI en vergelijkbare getallen voor elke input en voor de antwoorden, elke dag of elke week, en laten een alert afgaan als een getal te hoog wordt. Waar pas laat bekend is wat er echt gebeurde houden zulke tools intussen de inputs en de antwoorden in de gaten, in de wetenschap dat concept drift langs allebei kan glippen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **data drift** | | een verandering in de inputs die een model in productie krijgt, vergeleken met de training | |
        | **prediction drift** | | een verandering in de antwoorden die een model in productie geeft, vergeleken met de antwoorden bij de training | |
        | label drift | | een verandering in de echte antwoorden, zoals de verhuurde fietsen, vergeleken met de training; pas te zien als bekend is wat er echt gebeurde | |
        | concept drift | | een verandering in het verband tussen de inputs en het echte antwoord: dezelfde input hoort nu een ander antwoord te krijgen | |
        | **population stability index** (PSI) | | een getal voor hoe ver een verdeling verschoof ten opzichte van de training: per bin het aandeel nu min het aandeel bij de training, keer de log van het aandeel nu gedeeld door het aandeel bij de training, opgeteld over de bins | `psi(train, now)` |
        | oneindig, $\infty$ | | een getal boven elk ander getal; min oneindig ligt onder elk ander getal | `np.inf`, `-np.inf` |
        """),
        L.watch_out("Een rustige PSI op de inputs betekent geen gezond model. In 2012 blijven de temperatuur en de antwoorden rustig, terwijl het model er drie keer zo ver naast zit: een verandering in het verband tussen de inputs en het antwoord zie je alleen aan wat er echt gebeurde."),
        L.watch_out("Een lege bin heeft een aandeel van 0, en de log van 0 is min oneindig. Tools geven zo'n bin een piepklein aandeel, zoals 0,0001, en nemen pas dan de log."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model controleren op drift:

    1. Kies wat je in de gaten houdt: elke input (data drift), de antwoorden van het model (prediction drift), en zodra bekend, de echte antwoorden (label drift).
    2. Kies waarmee je vergelijkt: de data van de training, uit hetzelfde seizoen, zoals dezelfde maand.
    3. Maak tien bins uit deze data: min oneindig, de kwantielen 0,1 tot en met 0,9, en oneindig.
    4. Tel het aandeel van de waarden in elke bin, van de training en van nu.
    5. Bereken de PSI: per bin (nu − training) × log(nu / training), opgeteld over de bins.
    6. Beoordeel het getal met de vuistregel, onder 0,1 rustig en boven 0,25 drift, naast hoe ver het model ernaast zit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.TEMPS_TRAIN` bevat de temperatuur in graden van elk van de 6.912 uren van de training, en `C.TEMPS_2012` van elk van de 8.734 uren van 2012. `C.edges(values)` geeft de randen van tien bins: min oneindig, de kwantielen 0,1 tot en met 0,9 van `values`, en oneindig. `C.shares(values, edges)` geeft het aandeel van `values` in elke bin, en `C.psi(train, now)` de PSI van twee lijsten met aandelen: de eigen versies van de course van wat je in Oefening 1 en 2 schrijft. De cel print de negen randen in het midden, de aandelen van de training en van 2012, en de PSI. Voer de cel uit.
    """)
    return


@app.cell
def example_year():
    import numpy as np

    _edges = C.edges(C.TEMPS_TRAIN)
    _train = C.shares(C.TEMPS_TRAIN, _edges)
    _now = C.shares(C.TEMPS_2012, _edges)
    print(np.round(_edges[1:-1], 1))     # de negen randen tussen min oneindig en oneindig
    print(np.round(_train, 3))
    print(np.round(_now, 3))
    print(round(C.psi(_train, _now), 3))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De negen randen lopen van 9,0 tot 30,3 graden. De training heeft tussen 0,087 en 0,121 van de uren in elke bin: ongeveer een tiende, niet precies, want de temperaturen in de data gaan in stappen van 0,82 graden en veel uren hebben dezelfde waarde. 2012 heeft ongeveer dezelfde aandelen, met minder uren in de koudste bin, onder 9,0 graden: 0,040 tegen 0,087. De PSI is 0,041, onder 0,1: rustig. Over het hele jaar lijkt het weer van 2012 op het weer van de training, en toch zit het model er drie keer zo ver naast. In Oefening 1 en 2 schrijf je zelf `shares` en `psi`, en kijk je naar één maand.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het aandeel in elke bin

        `shares(values, edges)` krijgt een array met `values` en een array met `edges`, zoals `C.edges` de randen maakt. Bin `i` loopt van `edges[i]` tot `edges[i + 1]`: een waarde precies op `edges[i]` hoort bij bin `i`, en een waarde precies op `edges[i + 1]` bij de bin erboven. De loop gaat over elke `i` in `range(len(edges) - 1)`, en de list `out` verzamelt een aandeel per bin. De eerste regel van de loop kiest met een mask de waarden van `values` vanaf de onderkant van bin `i`, en noemt deze waarden `inside`. De laatste regel voegt het aandeel `len(inside) / len(values)` toe aan `out` met `append`, en de functie geeft `np.array(out)` terug. Vervang de `...` zodat van `inside` alleen de waarden onder de bovenkant van bin `i` overblijven.

        De check roept `shares(C.MARCH_2012, C.MARCH_EDGES)` aan: de temperaturen van de 743 uren van maart 2012, in de bins van de maarturen van de training. Dan de 566 maarturen van de training, `C.MARCH_TRAIN`, in dezelfde bins, heel 2012 in de bins van `C.EDGES`, en twee korte arrays met waarden precies op een rand.
        """),
        L.task("Houd van `inside` alleen de waarden onder de bovenkant van bin `i` over."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=7)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Stap 4 van het recept: `inside` bevat de waarden vanaf de onderkant van bin `i`, en `edges[i + 1]` is de bovenkant van bin `i`. Welke waarden van `inside` liggen in bin `i`?",
            "Een mask is een vergelijking tussen vierkante haken, zoals op de regel erboven: `values[values >= edges[i]]`. Deze keer gaat de vergelijking tussen `inside` en de bovenkant van de bin, en een waarde precies op de bovenkant hoort bij de bin erboven. Met welk teken?",
            "`        inside = inside[___]`",
        ],
        why="`inside = inside[inside < edges[i + 1]]`: van de waarden vanaf de onderkant van bin `i` houdt het mask de waarden onder de bovenkant. Een waarde precies op de bovenkant valt hier af door `<` en komt in de volgende bin door `>=`, dus komt elke waarde in precies één bin terecht. In maart 2012 komt een aandeel van 0,506 in de bovenste bin.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_psi():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De PSI

        `psi(train, now)` krijgt twee arrays met een aandeel per bin: `train`, de aandelen van de training, en `now`, de aandelen van de nieuwe waarden. De eerste twee regels geven een lege bin een piepklein aandeel met `np.maximum`, dat van elk aandeel en 0,0001 het grootste neemt. De laatste regel vermenigvuldigt `now - train` met een factor per bin, en telt de delen op met `.sum()`. Vervang de `...` door deze factor: de log van het aandeel nu gedeeld door het aandeel bij de training.

        De check roept `psi(C.SHARES_TRAIN, C.SHARES_2012)` aan: de aandelen van de maarturen van de training en van maart 2012, in de bins van `C.MARCH_EDGES`, zoals `shares` de aandelen geeft. Dan het hele jaar 2012 tegen de training, juli 2012 (`C.JULY_2012`) tegen de hele training, twee keer dezelfde aandelen, de twee van maart omgewisseld, en twee korte arrays, een met een lege bin.
        """),
        L.task("Vervang de `...` door de log van het aandeel nu gedeeld door het aandeel bij de training."),
    ])
    return


@app.cell(hide_code=True)
def exercise_psi():
    box_psi = L.code_box("psi", lines=4)
    box_psi
    return (box_psi,)


@app.cell(hide_code=True)
def check_psi(box_psi):
    result_psi = L.check("psi", box_psi.value)
    result_psi
    return (result_psi,)


@app.cell(hide_code=True)
def hints_psi():
    L.hints(
        "psi",
        [
            "Stap 5 van het recept: in elke bin is het deel `now - train` keer een factor, en `.sum()` telt de delen op. De factor is een log, en de log is 0 waar `now` en `train` gelijk zijn. De log van wat?",
            "De log van 1 is 0, en het aandeel nu gedeeld door het aandeel bij de training is 1 waar niets veranderde. `np.log` neemt de natuurlijke log van elk getal van een array tegelijk. Wat komt er tussen de haakjes?",
            "`    return ((now - train) * ___).sum()`",
        ],
        why="`return ((now - train) * np.log(now / train)).sum()`: in elke bin hoeveel het aandeel verschoof, keer de log van hoeveel keer zo groot het aandeel werd, opgeteld over de bins. Maart 2012 tegen maart in de training geeft 1,15; de temperatuur van het hele jaar 0,041.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_season():
    mo.md(r"""
    ### Oefening 3. Tegen welke maand?

    Je vergelijkt de temperaturen van juli 2012 met de bins van alle uren van de training, `C.EDGES`, en je `psi` geeft 4,84, ver boven 0,25.
    """)
    return


@app.cell(hide_code=True)
def choice_season_cell():
    choice_season = L.choice(
        "season",
        question="Wat zegt dat getal?",
        options={
            "hot": "Juli 2012 was veel heter dan juli 2011, dus moet het model opnieuw getraind worden",
            "right": "Weinig: juli is zomer, en de bins komen uit alle seizoenen van het jaar",
            "bins": "Tien bins zijn te weinig voor één maand: met honderd bins zou de PSI onder 0,25 liggen",
            "broken": "De temperatuursensor ging stuk: een PSI boven 1 komt alleen van een kapotte input",
        },
        correct="right",
        why={
            "right": "Tegen de hele training ligt een zomermaand altijd in de bovenste bins: zo ligt elke maand van 2012 boven 0,25, ook bij weer zoals in 2011. Tegen de juli-uren van de training geeft juli 2012 0,06: rustig. Vergelijk gelijk met gelijk, zoals dezelfde maand.",
            "hot": "Juli 2012 tegen de juli-uren van de training geeft 0,06. Welke vergelijking laat juli er zo anders uitzien?",
            "bins": "Meer bins zouden het jaar in meer stukken verdelen, en juli zou nog steeds de bovenste bins vullen. Wat bevatten de bins van de hele training dat juli niet bevat?",
            "broken": "Een kapotte sensor kan de PSI omhoog duwen, net als een seizoen. Wat bevatten de bins van de hele training dat juli niet bevat?",
        },
    )
    choice_season
    return (choice_season,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 4. Welke drift?

    Zes veranderingen die de service van het project kunnen overkomen. Zet elke verandering bij de soort drift die erbij hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Vraag bij elke verandering: veranderen de verzoeken, of alleen wat er uit het model komt, of alleen wat fietsers echt doen?",
            "Twee veranderingen beginnen in de verzoeken, twee in de service zelf, en twee bij de fietsers, terwijl de verzoeken en de antwoorden hetzelfde blijven.",
            "Data: een warme maart, \\_\\_\\_. Prediction: een nieuwe versie, \\_\\_\\_. Label: meer leden, \\_\\_\\_.",
        ],
        why="**Data drift:** een warme maart en een nieuwe sensor, want de inputs veranderen. **Prediction drift:** een nieuwe versie en een andere schaal, want alleen de antwoorden veranderen. **Label drift:** meer leden en een hogere prijs, want alleen de verhuurde fietsen veranderen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Elke maand van 2012

    Kies wat je in de gaten houdt en waarmee je vergelijkt. De staven tonen de PSI van elke maand van 2012; de lijnen eronder tonen hoe ver het model er per uur naast zat in elke maand, in 2011 en in 2012.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_column = mo.ui.dropdown(
        options={
            "De temperatuur": "temp",
            "De vochtigheid": "hum",
            "De wind": "wind",
            "De antwoorden van het model": "answers",
            "De verhuurde fietsen": "rentals",
        },
        value="De temperatuur",
        label="Houd in de gaten",
    )
    demo_ref = mo.ui.dropdown(
        options={"dezelfde maand in de training": "same", "de hele training": "all"},
        value="dezelfde maand in de training",
        label="Vergelijk met",
    )
    mo.vstack([
        mo.hstack([demo_column, demo_ref], justify="start", wrap=True),
        L.task("Begin met de temperatuur tegen dezelfde maand, dan de antwoorden van het model, dan de verhuurde fietsen. Het model zat in elke maand van 2012 verder ernaast dan in 2011: welke van de drie ligt in de meeste maanden boven 0,25? Vergelijk daarna met de hele training: wat gebeurt er met juli, en met de verhuurde fietsen van januari?"),
    ])
    return demo_column, demo_ref


@app.cell(hide_code=True)
def demo_picture(demo_column, demo_ref):
    mo.Html(C.demo_picture(demo_column.value, demo_ref.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In alle 12 maanden van 2012 zat het model verder ernaast dan in dezelfde maand van 2011. Tegen dezelfde maand in de training ligt de temperatuur in 5 van de 12 maanden boven 0,25, maart het hoogst, en is juli rustig, op 0,06, terwijl het model er 86,8 fietsen per uur naast zat, tegen 46,5 in juli 2011. De antwoorden blijven in elke maand onder 0,1. De verhuurde fietsen liggen in 10 van de 12 maanden boven 0,25: alleen wat er echt gebeurde geeft in bijna elke maand alarm. Hoe ver het model ernaast zit lees je niet af aan de hoogte van de PSI: de verhuurde fietsen van januari geven de hoogste PSI, in de maand waarin het model het minst ernaast zat. Tegen de hele training ligt de temperatuur in elke maand boven 0,25, een vals alarm van de seizoenen, en zakken de verhuurde fietsen van januari van 0,85 naar 0,04: een echte drift verdwijnt uit beeld.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hoe ML-systemen falen](/course/open/sys-ml-failures?lang=nl) keek een check tegen het bereik van de training naar één verzoek tegelijk en ving een eenheid die veranderde; de PSI kijkt naar een hele maand verzoeken tegelijk, en ziet een verschuiving binnen het bereik. De bins komen uit de kwantielen van [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl), en de factor van elke bin is een log, zoals in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl). In de volgende les, *Als een model de eigen data verandert: feedback loops*, verandert het model zelf de wereld waarvan het leert. En in *Project: je model als service, bewaakt op drift* controleert je service de data van 2012 maand na maand.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_season, result_kinds, result_psi, result_shares):
    L.wrap_up(result_shares, result_psi, choice_season, result_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
